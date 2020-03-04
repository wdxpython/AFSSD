# An Accurate and Fast Single Shot Multibox Detector

By Lie Guo, Dongxing Wang, Linhui Li 

### Introduction
With the development of deep learning, the performance of object detection has made great progress. But there are still some challenging problems, such as the detection accuracy of small objects and the efficiency of the detector. This paper proposes an accurate and fast single shot multibox detector (AFSSD) , which includes context comprehensive enhancement module (CCE) module and feature enhancement module (FEM). 

<img align="right" src="https://github.com/wdxpython/AFSSD/blob/master/img/919462661.jpg">
&nbsp;
&nbsp;

## Training

- By default, we assume you have downloaded the file in the `AFSSD/weights` dir:
```Shell
mkdir weights
cd weights
wget https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth
```

- To train RFBNet using the train script simply specify the parameters listed in `train_AFSSD.py` as a flag or manually change them.
```Shell
python train_AFSSD.py -d VOC -v AFSSD -s 320 
```
## Evaluation
To evaluate a trained network:

```Shell
python test_AFSSD.py -d VOC -v AFSSD -s 320 
```

## Thank For
	Thanks for [xxx‘s project](https://github.com/ruinmessi/RFBNet) 

