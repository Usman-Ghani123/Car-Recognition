import keras
from resnet_50 import resnet50_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.models import Model

img_width, img_height = 224, 224
num_channels = 3
num_classes = 196
resnet = resnet50_model(img_height, img_width, num_channels, num_classes)

for layer in resnet.layers:
  layer.trainable = False

model = Model(inputs=resnet.input, outputs=resnet.output)

model.summary()
