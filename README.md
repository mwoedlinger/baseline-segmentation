# Text baseline segmentation

Performs segmentation of document page images.

## Preprocessing

The folder *demo* contains a jupyter notebook to prepare data for training.

## Training and testing

Training can be done using `train.py`. The parameters can be specified in a config file.
The available models are:
```
DeepLabV3_[bb]   [1]
FCN_[bb]         [2]
GCN              [3]
```
with *[bb]* denoting the backbone (*resnet50*, *resnet101*).

Testing of the segmentation network can be done using `test.py`.

## Literature

```
[1] Chen, Liang-Chieh, et al. "Rethinking atrous convolution for semantic image segmentation." 
    arXiv preprint arXiv:1706.05587 (2017).
[2] Long, Jonathan, Evan Shelhamer, and Trevor Darrell. "Fully convolutional networks for semantic segmentation." 
    Proceedings of the IEEE conference on computer vision and pattern recognition. 2015.
[3] Peng, Chao, et al. "Large Kernel Matters--Improve Semantic Segmentation by Global Convolutional Network." 
    Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.
```
