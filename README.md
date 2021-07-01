# 🍕 Food Recognition 🍅

The Food Recognition challenge is a very interesting topic that consists in training a Deep Learning model able to detect the food items in an image and classify them in different categories.
The kind of project falls into the category of the image segmentation tasks, that are based on the process of partitioning a digital image into multiple segments.
We tried to address this problem by trying different deep learning neural networks in order to experiment some specific techniques and then make a comparison between them.
We took inspiration from model architectures that reached very good results in the field of image segmentation and we tried to adapt them to our purpose.
The neural networks that we have implemented are **U-Net**, **LinkNet** and **Mask-RCNN**.

## Dataset
For our project we used the dataset made available by the AIcrowd community.
The dataset is composed in the following way:
* Training Set of 24120 RGB food images, along with their corresponding 39328 annotations in MS-COCO format
* Validation Set of 1269 RGB food images, along with their corresponding 2053 annotations in MS-COCO format

## Implementation
* [Res-U-Net](https://github.com/BeleRicks11/Food_Recognition/blob/main/U-Net/U_net_version.ipynb)
* [Mask-RCNN](https://github.com/BeleRicks11/Food_Recognition/blob/main/Mask_RCNN/maskrecognition.ipynb)
* [Link-Net](https://github.com/BeleRicks11/Food_Recognition/blob/main/LinkNet/LinkNet.ipynb)

Everything is described in the following report:
* [Report](https://github.com/BeleRicks11/Food_Recognition/blob/main/Report.pdf)

## Evaluation criteria
Given a known ground truth mask A, and a predicted mask B, first compute Intersection Over Union (IoU). The prediction is tradionally considered a True detection, when there is at least half an overlap, i.e. IoU >0.5. Then you may define precision and recall. The final scoring parameters are computed by averaging over all the precision and recall.
The final scoring parameters are computed by averaging over all the precision and recall values for all known annotations in the ground truth.

## Libraries
* TensorFlow with Keras backend
* Numpy
* Pandas
* OpenCV
* pycocotools
* matplotlib
* segmentation_models

## Authors
* [Riccardo Fava](https://github.com/BeleRicks11)
* [Luca Bompani](https://github.com/Bomps4)
* [Ganesh Pavan Kartikeya Bharadwaj Kolluri](https://github.com/karthikbharadhwajKB)
