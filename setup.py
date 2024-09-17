from setuptools import setup, find_packages
# from distutils.core import setup

setup(
    name='ElectronCounting',
    version='0.1.5',
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
        "torch>=1.12.0",
        "torchvision>=0.13.0",
        "numpy>=1.20.1",
        # "scikit-learn>=1.0.2",
        # "matplotlib>=3.2.2",
        # "kornia>=0.6.9",
    ],

)
