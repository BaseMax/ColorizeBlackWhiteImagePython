# Colorize Black&White Images Python

Colorize black & white images, using machine learning in Python.

## Model

- `colorization_release_v2.caffemodel`: It is a pre-trained model stored in the Caffe framework’s format that can be used to predict new unseen data.
- `colorization_deploy_v2.prototxt`: It consists of different parameters that define the network and it also helps in deploying the Caffe model.
- `pts_in_hull.npy`: It is a NumPy file that stores the cluster center points in NumPy format. It consists of 313 cluster kernels, i.e (0-312).

For downloading the files, click [here](https://drive.google.com/drive/folders/1FaDajjtAsntF_Sw5gqF0WyakviA5l8-a).

### Sources

Note: This is not mine and I used another model which is available on the internet. For reading more you can look at following links:

- https://drive.google.com/drive/folders/1FaDajjtAsntF_Sw5gqF0WyakviA5l8-a
- https://modelzoo.co/model/colorful-image-colorization
- https://github.com/opencv/opencv/blob/master/samples/dnn/colorization.py
- https://github.com/richzhang/colorization/blob/caffe/colorization/resources/pts_in_hull.npy
- https://github.com/richzhang/colorization/tree/caffe/colorization/models

Thanks all and also great OpenCV lib.
