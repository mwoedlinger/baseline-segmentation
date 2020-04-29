# TUWien-libraryCards

Finds for a given library archive card automatically the text regions for Verfasser, Titel, Signatur, etc. and extracts the corresponding text from the Transkribus page XML file.

## Preprocessing

The folder *demo* contains jupyter notebooks to prepare data for training.
The files contain descriptions on how to use them.

## Training and testing

Training can be done using `segmentation_train_pytorch.py` and `classification_train_pytorch.py`. 
The parameters can be specified in a config file or directly in corresponding python files.
The available models are:
```
DeepLabV3_[bb]   [1]
DeepLabV3_plus   [2]
FCN_[bb]         [3]
GCN              [4]
```
with *[bb]* denoting the backbone (*resnet50*, *resnet101*).

Testing of the segmentation network can be done using `segmentation_test_pytorch.py`.

## Inference
### Classification

To apply the classifier use `apply_classifier.py --options` with the options:
```
--input     The folder containing the scans
--output    the output folder for the text files
--weights   Weights for the classifier
--gpu       GPU that should be used
```
The output consists of two text files containing the filenames of Schemas and Verweise respectively
and two text files containing the files where the model is the least confidence (these files are still contained in the first two text files).

### Segmentation
To perform the segmentation and baseline detection use `apply_segmentation.py --options` with the options:
```
--filenames The txt file containing the filenames
--weights   Weights for the segmentation model
--gpu       GPU that should be used
```
The function assumes folders where for every image **/filename.jpg* there is a corresponding page xml file **/page/filename.jpg* that contains the baselines.
The function modifies the xml file by modifying the custom tag for every baseline and adding its type.

## Demo
The folder *demo* contains the files `classification_demo.ipynb`, `segmentation_demo.ipynb` and `Full_pipeline.ipynb` that contain demonstrations.

## Requirements
The project has been tested with Python 3.5.2.

```
torch 1.2
torchvision 0.4.0
tensorboard 1.15.0
tqdm 4.35.0
numpy 1.17.2
```

## Literature

```
[1] Chen, Liang-Chieh, et al. "Rethinking atrous convolution for semantic image segmentation." 
    arXiv preprint arXiv:1706.05587 (2017).
[2] Chen, Liang-Chieh, et al. "Encoder-decoder with atrous separable convolution for semantic image segmentation." 
    Proceedings of the European conference on computer vision (ECCV). 2018.
[3] Long, Jonathan, Evan Shelhamer, and Trevor Darrell. "Fully convolutional networks for semantic segmentation." 
    Proceedings of the IEEE conference on computer vision and pattern recognition. 2015.
[4] Peng, Chao, et al. "Large Kernel Matters--Improve Semantic Segmentation by Global Convolutional Network." 
    Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.
```