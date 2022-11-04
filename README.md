# Phase I Instructions
The path to the folder of images can be provided as mentioned below.

``python Wrapper.py --Folder /home/naitri/Downloads/YourDirectoryID_p1/Phase1/Data/Train/Set1
``

# Phase II Instructions

- The ```Wrapper.py``` simply shows an example of generating a pair of images according to the Data Generation technique presented in the Supervised approach paper.
- To train the supervised model run ```python Train.py --ModelType=sup```
- To train the unsupervised model run ```python Train.py --ModelType=unsup```
- To test the supervised model run ```python Test.py --ModelType=sup``` after having trained it.
- To test the unsupervised model run ```python Test.py --ModelType=unsup``` after having trained it.
- All scripts assume that there is a ```Data``` folder, in which MS COCO images are stored according to the given filesystem scheme.
