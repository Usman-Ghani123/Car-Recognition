# Car Recognition


This repository is to do car recognition by fine-tuning ResNet-50 with Cars Dataset from Stanford.


## Dependencies

- Numpy 1.16.1
- Tensorflow 1.13.1
- Keras 2.0.0
- OpenCV 4.2.0



## Dataset

We use the Cars Dataset, which contains 16,185 images of 196 classes of cars. The data is split into 8,144 training images and 8,041 testing images, where each class has been split roughly in a 80-20 split.

 ![image](https://github.com/foamliu/Car-Recognition/raw/master/images/random.jpg)

You can get it from [Cars Dataset](https://ai.stanford.edu/~jkrause/cars/car_dataset.html):

```bash
$ cd Car-Recognition
$ wget http://imagenet.stanford.edu/internal/car196/cars_train.tgz
$ wget http://imagenet.stanford.edu/internal/car196/cars_test.tgz
$ wget --no-check-certificate https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz
```

## ImageNet Pretrained Models

Download [ResNet-50](https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5) into models folder.

## Usage

### Data Pre-processing
Extract 8,144 training images, and split them by 80:20 rule (6,515 for training, 1,629 for validation):
```bash
$ python pre_process.py
```

### Train
```bash
$ python train.py
```

If you want to visualize during training, run in your terminal:
```bash
$ tensorboard --logdir path_to_current_dir/logs
```
*Python version must be atleast 3.6.8 to run this command in your cmd or anaconda promopt or miniconda prompt*

 ![image](https://github.com/Usman-Ghani123/Car-Recognition/blob/master/Accuracy_Loss/acc.PNG) ![image](https://github.com/Usman-Ghani123/Car-Recognition/blob/master/Accuracy_Loss/loss.PNG) ![image](https://github.com/Usman-Ghani123/Car-Recognition/blob/master/Accuracy_Loss/val_acc.PNG) ![image](https://github.com/Usman-Ghani123/Car-Recognition/blob/master/Accuracy_Loss/val_loss.PNG)

### Analysis
Update "model_weights_path" in "utils.py" with your best model, and use 1,629 validation images for result analysis:
```bash
$ python analyze.py
```

#### Validation acc:
**97.54%**

#### Confusion matrix:

 ![image](https://github.com/Usman-Ghani123/Car-Recognition/blob/master/Confusion%20Matrix/Normalized%20Confusion%20matrix.png) ![image](https://github.com/Usman-Ghani123/Car-Recognition/blob/master/Confusion%20Matrix/confusion%20matrix.png)


### Test
```bash
$ python test.py
```

Submit predictions of test data set (8,041 testing images) at [Cars Dataset](https://ai.stanford.edu/~jkrause/cars/car_dataset.html), evaluation result:

#### Test acc:
**87.60%**

 ![image](https://github.com/Usman-Ghani123/Car-Recognition/blob/master/Capture.JPG)

### Demo
Download [pre-trained model](https://drive.google.com/file/d/1K7NRmwlTenyerFrW-mrevRuDaNuHDItu/view?usp=sharing) into "models" folder then run:

```bash
$ python demo.py
```
If no argument, a sample image is used:

 ![image](https://github.com/foamliu/Car-Recognition/raw/master/images/samples/07647.jpg)

```bash
$ python demo.py
class_name: Lamborghini Reventon Coupe 2008
prob: 0.9999994
```
### GUI

```bash
$ python gui.py  
```
Select a directory of images of cars. It will create bounding box around a car and will show class name and probability score:

![image](https://github.com/Usman-Ghani123/Car-Recognition/blob/master/gui_capture.JPG)


