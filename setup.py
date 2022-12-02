from setuptools import setup, find_packages

setup(
    name='ElectronCounting',
    version='0.1.0',
    packages=find_packages(),
    url='https://github.com/wdwzyyg/ElectronCounting.git',
    license='MIT',
    author='Jingrui Wei',
    author_email='jwei74@wisc.edu',
    description='Count single electron in direct electron detector data',
    keywords=[
        "direct electron detector",
        "electron microscopy",
        "data analysis",
        "object detection",
        "Faster R-CNN",
    ],
    install_requires=[
        "setuptools~=62.0.0"
        "torch~=1.12.0"
        "numpy~=1.22.3"
        "scikit-learn~=1.0.2"
        "matplotlib~=3.5.1"
        "opencv-python~=4.5.5.64"
        "torchvision~=0.13.0+cu113"
    ],

)
