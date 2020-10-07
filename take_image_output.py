import keras
from resnet_50 import resnet50_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import Model
from matplotlib import pyplot
from numpy import expand_dims

img_width, img_height = 224, 224
num_channels = 3
num_classes = 196

# load the model
model = resnet50_model(img_height, img_width, num_channels, num_classes)

# redefine model to output right after the first hidden layer
model = Model(inputs=model.inputs, outputs=model.layers[5].output)

model.summary()

# load the image with the required shape
img = load_img('data/test/00001.jpg', target_size=(224, 224))

# convert the image to an array
img = img_to_array(img)

# expand dimensions so that it represents a single 'sample'
img = expand_dims(img, axis=0)

# get feature map for first hidden layer
feature_maps = model.predict(img)

# plot all 64 maps in an 8x8 squares
square = 8
ix = 1
for _ in range(square):
	for _ in range(square):
		# specify subplot and turn of axis
		ax = pyplot.subplot(square, square, ix)
		ax.set_xticks([])
		ax.set_yticks([])
		# plot filter channel in grayscale
		pyplot.imshow(feature_maps[0, :, :, ix-1], cmap='gray')
		ix += 1
# show the figure
pyplot.show()