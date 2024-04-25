import cv2
import numpy as np
import torch
from scipy.ndimage import center_of_mass, maximum_position
from scipy.ndimage import label, find_objects
from sklearn.metrics import pairwise_distances_argmin_min


##########################################################################################################################
### Counting methods ###
# Below list six different counting methods, primaryly use connected component labeling(CCL) to find the clusters, 
# and assign the entry position to max intensity pixel, or center of mass, or center of mass after binarization, or random.
# the last one, fastrcnn_predict, is using the ML model, instead of CCL.
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


def fastrcnn_predict(model, arr, process_stride, mode, **kwargs):
  from CountingNN.locator import Locator
  x = arr[None, ...]
  device =  torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  counting = Locator(model, device, process_stride, 'max', 30, None, mode, meanADU = kwargs.get('meanADU'), 
                     p_list=kwargs.get('p_list'), dynamic_thres = kwargs.get('dynamic_thres'), pretune_thresholding = kwargs.get('pretune_thresholding'))
  filtered, event_sizes =  counting.predict_sequence(x)
  filtered = filtered[0]
  all_coords = []
  for value in range(1, 1 + filtered.max()):
      coords = np.array(np.where(filtered==value))        
      all_coords.append([coords]*value)
  all_coords = np.hstack(np.array(all_coords)[0]).T

  return filtered, all_coords, event_sizes

##########################################################################################################################
### Evaluation metrics calculation ###

def pos_deviation(coords, truth, threshold):
    """
    Cal the root mean square error between detected electron incident positions and the ground truth positions in units of pixels.
    """
    # elements in pair 1 need to be no less than pair 2 
    distances = []
    if len(coords):
      assigment,distances = pairwise_distances_argmin_min(coords, truth)

    return distances


def general_evaluation(file, algorithm, repeat, savepath, **kwargs):
  '''
  This function is for calculating the scores for each data file, Stack***.npz together. 
  Arguments
  -----------
  file: the validation data file. e.g., my Stack000.npz contains images with sparsity 0~0.002, array X is the input images, 
  it has the shape of [N, M, 256, 256], N different sparsity ranging from sparsitymin(0) to sparsitymax(0.002), M copies of each same sparsity.
  Then Stack001.npz contains images with sparsity 0.002~0.004, and so on. 

  algorithm: run evaluation of one counting algorithm, string of the function name defined above.

  repeat: set the number of images with identical sparsity in each data file to be used.  
  '''
  data = np.load(file)

  X = data['X'][:,:repeat]
  y = data['y'][:,:repeat]
  print('Max pixel value in ground truth:', y.max())
  # creat blank arrays to store the score values, which has the same first two dimension as X. 
  dce = np.zeros(X.shape[:2]) # dce: simple detector conversion efficiency,  the ratio of input and detected electron counts
  mae = np.zeros(X.shape[:2]) # mae: mean absolute error, the absolute error of electron counts averaged over all pixels in a single image
  nume = np.zeros(X.shape[:2]) # number of actrual electrons in the image
  recall = np.zeros(X.shape[:2]) # recall, true positive / (true positives + false negtives)
  precision = np.zeros(X.shape[:2]) # precision, true positive / (true positives + false positives)
  filtered =  np.zeros(X.shape) # i.e. the counted image

  # saving the coordinate, position deviation and event size for each detected electron event in the image, 
  # so need to create an array of objects, and the object is a list
  coords = [ [0] * X.shape[1] ] * X.shape[0]
  deviations = [ [0] * X.shape[1] ] * X.shape[0]
  eventsizes = [ [0] * X.shape[1] ] * X.shape[0]
  coords = np.array(coords, dtype=object)
  deviations = np.array(deviations, dtype=object)
  eventsizes = np.array(eventsizes, dtype=object)
  save_e_size = True


  # Now go through the NxM images to get the scores
  for i in range(X.shape[0]):
    for j in range(X.shape[1]):
          
      if algorithm =='fastrcnn_predict':
        model = kwargs.get('model')
        method = kwargs.get('method')
        stride = kwargs.get('stride')
        mode = kwargs.get('mode')
        meanADU = kwargs.get('meanADU')
        p_list = kwargs.get('p_list')
        dynamic_thres = kwargs.get('dynamic_thres')
        pretune_thresholding = kwargs.get('pretune_thresholding')
        # by using the "eval", the long string has been read as a line of code, and it runs the algorithm function
        res = eval(algorithm +"(model, X[i][j], stride, mode, meanADU=meanADU, p_list=p_list, dynamic_thres = dynamic_thres,pretune_thresholding = pretune_thresholding )")
        filtered[i,j] = res[0]
        coords[i][j] = res[1]
        # if the algorithm returns eventsize, set to save it.
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
      
      # Get all the ground truth coordinates of electron events
      # For a pixel value 2 for example, indicating 2 electrons here, so we need to add its coordinate twice. 
      truth = []
      for value in range(1, 1+int(y[i,j].max())):
        truth_ = np.array(np.where(y[i,j]==value))        
        truth.append([truth_]*value)
      truth = np.hstack(np.array(truth)[0]).T

      total_pixel = filtered[i,j].shape[0] * filtered[i,j].shape[1]
      mae[i,j] = np.sum(np.abs(filtered[i,j]-y[i,j]))/total_pixel
      dce[i,j] = np.sum(filtered[i,j])/np.sum(y[i,j])
      nume[i,j] = np.sum(y[i,j])

      tp = 0 
      # count how many electron events are well identified, i.e., count the true positives
      for n, value in enumerate(filtered[i,j].ravel()):

        if (value != 0) & (y[i,j].ravel()[n] != 0):
          tp = tp + np.min((value, y[i,j].ravel()[n])) # multi-class considered

      recall[i,j] = tp/nume[i,j]
      precision[i,j] = tp/np.sum(filtered[i,j])

      deviations[i][j] = pos_deviation(coords[i][j], truth, 6)
      dce[i,j] = len(deviations[i][j])/np.sum(y[i,j])

  path = savepath + file[-12:-4]
  if save_e_size:
    np.savez(path+'_result.npz', coordinates = coords, result = filtered, mae = mae, dce = dce, nume = nume, 
    recall = recall, precision = precision, deviations = deviations, eventsizes = eventsizes)
  else:
    np.savez(path+'_result.npz', coordinates = coords, result = filtered, mae = mae, dce = dce, nume = nume, 
    recall = recall, precision = precision, deviations = deviations)
