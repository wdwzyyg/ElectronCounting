# temp change: position deviation
import cv2
import numpy as np
import torch
from scipy.ndimage import center_of_mass, maximum_position
from scipy.ndimage import label, find_objects
from sklearn.metrics import pairwise_distances_argmin_min  # works better than below


# all methods return the 256x256 counted image and coords array of shape(num, 2)


def cca(img):
  '''
  only returns the stats from cca
  '''
  thresh = np.array(img > 20).astype('int8')
  output = cv2.connectedComponentsWithStatsWithAlgorithm(thresh, 8, cv2.CV_32S, 0) 
  (_, _, stats, centroids) = output
  return stats


def counting_filter_binary_com(image, threshold=20, structure = np.ones((3,3))):
    image_binary = image > threshold  # more readable
    all_labels, num = label(image_binary, structure = np.ones((3,3)))  # get blobs
    m=np.ones(shape=all_labels.shape)
    obj = center_of_mass(m, all_labels, range(1,num))
    obj = np.rint(obj).astype(int)
    x = np.zeros(shape=np.shape(image))
    x[obj[:,0],obj[:,1]]=1
    return x, obj


def counting_filter_com(image, threshold=20, structure = np.ones((3,3))):
    image_binary = image > threshold  # more readable
    all_labels, num = label(image_binary, structure = np.ones((3,3)))  # get blobs
    # m=np.ones(shape=all_labels.shape)
    obj = center_of_mass(image, all_labels, range(1,num))
    obj = np.rint(obj).astype(int)
    x = np.zeros(shape=np.shape(image))
    x[obj[:,0],obj[:,1]]=1
    return x, obj


def counting_filter_max(image, threshold=20, structure = np.ones((3,3))):
    eventsize = []
    image_binary = image > threshold  # more readable
    all_labels, num = label(image_binary, structure = np.ones((3,3)))  # get blobs
    m=np.ones(shape=all_labels.shape)
    obj = maximum_position(image, all_labels, range(1,num))
    obj = np.rint(obj).astype(int)
    x = np.zeros(shape=np.shape(image))
    x[obj[:,0],obj[:,1]]=1
    for i in np.arange(num)[1:]:
      eventsize.append(np.where(all_labels==i)[0].shape[0])
    return x, obj, np.array(eventsize).astype('int')


def counting_filter_random(image, threshold=20, structure = np.ones((3,3))):
    image_binary = image > threshold  # more readable
    all_labels, num = label(image_binary, structure = np.ones((3,3)))  # get blobs
    obj = find_objects(all_labels)
    coords = []
    for i in range(len(obj)):
      coords.append((np.random.randint(obj[i][0].start,obj[i][0].stop),
                    np.random.randint(obj[i][1].start,obj[i][1].stop)))
    coords = np.array(coords)
    x = np.zeros(shape=np.shape(image))
    x[coords[:,0], coords[:,1]] = 1
    return x, coords


def pos_deviation(coords, truth, threshold):
  # elements in pair 1 need to be no less than pair 2 
  distances = []
  if len(coords):
    assigment,distances = pairwise_distances_argmin_min(coords, truth)
  # for p in range(assigment.shape[0]):
  #   if distances[p] < threshold:
  #     dis.append(distances[p])
  # return dis
  return distances

# def fastrcnn_predict(model, arr, method, locating_model, dynamic_param, **kwargs):
#   from MaskRCNN.locator_archive import Locator
#   x = arr[None,...]
#   device =  torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#   counting = Locator(model, device, 64, method, 20, locating_model, dynamic_param, meanADU = kwargs.get('meanADU'), p_list=kwargs.get('p_list'))
#   boxes_list = counting.predict(x) # x as the image array in shape [1,h,w]
#   filtered, coords, eventsize = counting.locate(x[0], boxes_list[0])
#   return filtered, coords, eventsize


def fastrcnn_predict(model, arr, process_stride, mode, **kwargs):
  from CountingNN.locator import Locator
  x = arr[None, ...]
  device =  torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  counting = Locator(model, device, process_stride, 'max', 30, None, mode, meanADU = kwargs.get('meanADU'), 
                     p_list=kwargs.get('p_list'), dynamic_thres = kwargs.get('dynamic_thres'), pretune_thresholding = kwargs.get('pretune_thresholding'))
  filtered, event_sizes =  counting.predict_sequence(x)
  filtered = filtered[0]
  # with this predict function, the event sizes are just not match with position deviations
  coords1 = np.array(np.where(filtered==1))        
  coords2 = np.array(np.where(filtered==2))        
  coords3 = np.array(np.where(filtered==3))        
  coords = np.hstack((coords1, coords2, coords2, coords3, coords3, coords3)).T
  return filtered, coords, event_sizes


def general_evaluation(file, algorithm, repeat, savepath, **kwargs):
  '''
  from the centoids results from the clustering methods, calculate:
  - DCE: count for FPs
  - position deviations
  - corrected DCE with position deviation: get rid of FPs and count for FNs
  '''
  data = np.load(file)

  X = data['X'][:,:repeat]
  y = data['y'][:,:repeat]
  print('Max pixel value in ground truth:', y.max())
  dce = np.zeros(X.shape[:2])
  dce_corrected = np.zeros(X.shape[:2])
  mae = np.zeros(X.shape[:2])
  nume = np.zeros(X.shape[:2])
  recall = np.zeros(X.shape[:2])
  precision = np.zeros(X.shape[:2])
  filtered =  np.zeros(X.shape)
  coords = [ [0] * X.shape[1] ] * X.shape[0]
  deviations = [ [0] * X.shape[1] ] * X.shape[0]
  eventsizes = [ [0] * X.shape[1] ] * X.shape[0]
  coords = np.array(coords, dtype=object)
  deviations = np.array(deviations, dtype=object)
  eventsizes = np.array(eventsizes, dtype=object)
  save_e_size = True

  for i in range(X.shape[0]):
    for j in range(X.shape[1]):
      if algorithm == 'cnn_predict' or algorithm == 'cnn_ccamax_mix' or algorithm == 'fcn_predict':
        model = kwargs.get('model')
        res = eval(algorithm +"(model, X[i][j])")
        filtered[i,j] = res[0]
        coords[i][j] = res[1]
        try:
          eventsizes[i][j] = res[2]
        except:
          save_e_size = False
          
      elif algorithm =='fastrcnn_predict':
        model = kwargs.get('model')
        method = kwargs.get('method')
        stride = kwargs.get('stride')
        mode = kwargs.get('mode')
        meanADU = kwargs.get('meanADU')
        p_list = kwargs.get('p_list')
        dynamic_thres = kwargs.get('dynamic_thres')
        pretune_thresholding = kwargs.get('pretune_thresholding')
        res = eval(algorithm +"(model, X[i][j], stride, mode, meanADU=meanADU, p_list=p_list, dynamic_thres = dynamic_thres,pretune_thresholding = pretune_thresholding )")
        filtered[i,j] = res[0]
        coords[i][j] = res[1]
        try:
          eventsizes[i][j] = res[2]
        except:
          save_e_size = False

      else: 
        res = eval(algorithm + "(X[i][j])")
        filtered[i,j] = res[0]
        coords[i][j] = res[1]
        try:
          eventsizes[i][j] = res[2]
        except:
          save_e_size = False

      truth1 = np.array(np.where(y[i,j]==1))        
      truth2 = np.array(np.where(y[i,j]==2))        
      truth3 = np.array(np.where(y[i,j]==3))
      truth4 = np.array(np.where(y[i,j]==4))
      truth5 = np.array(np.where(y[i,j]==5))       
      truth = np.hstack((truth1, truth2, truth2, truth3, truth3, truth3, truth4, truth4,truth4,truth4,))
      # print(coords[i][j].shape,truth.shape )
      mae[i,j] = np.sum(np.abs(filtered[i,j]-y[i,j]))/65536
      dce[i,j] = np.sum(filtered[i,j])/np.sum(y[i,j])
      nume[i,j] = np.sum(y[i,j])

      tp = 0
      for n, value in enumerate(filtered[i,j].ravel()):

        if (value != 0) & (y[i,j].ravel()[n] != 0):
          tp = tp + np.min((value,y[i,j].ravel()[n])) # multi-class considered

      recall[i,j] = tp/nume[i,j]
      precision[i,j] = tp/np.sum(filtered[i,j])

      deviations[i][j] = pos_deviation(coords[i][j], truth.T, 6)
      #print(len(deviations[i][j]),truth.shape[0])
      dce_corrected[i,j] = len(deviations[i][j])/np.sum(y[i,j])

  path = savepath + file[-12:-4]
  if save_e_size:
    np.savez(path+'_result.npz', coordinates = coords, result = filtered, mae = mae, dce = dce, nume = nume, 
    recall = recall, precision = precision, deviations = deviations, dce_corrected = dce_corrected, eventsizes = eventsizes)
  else:
    np.savez(path+'_result.npz', coordinates = coords, result = filtered, mae = mae, dce = dce, nume = nume, 
    recall = recall, precision = precision, deviations = deviations, dce_corrected = dce_corrected)
