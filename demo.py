# import the necessary packages
import json
import os
import random

import cv2 as cv
import keras.backend as K
import numpy as np
import scipy.io

from utils import load_model

if __name__ == '__main__':
    img_width, img_height = 224, 224
    model = load_model()
    model.load_weights('G:/fyp/resnet_50/model/epoch_100_no_augmentation/model.17-0.78.hdf5')

    cars_meta = scipy.io.loadmat('devkit/cars_meta')
    class_names = cars_meta['class_names']  # shape=(1, 196)
    class_names = np.transpose(class_names)

    # search the path with images which are in jpg format
    test_path = 'images/samples/'
    test_images = [f for f in os.listdir(test_path) if
                   os.path.isfile(os.path.join(test_path, f)) and f.endswith('.jpg') or f.endswith('png')]

    
    num_samples = 4

    # it will pick any two random images from that directory
    samples = random.sample(test_images, num_samples)
    results = []
    for i, image_name in enumerate(samples):
        # directory is being made here
        filename = os.path.join(test_path, image_name)
        print('Start processing image: {}'.format(filename))
        # image is opened
        bgr_img = cv.imread(filename)
        # resized the image to 224,224 bcoz resent 50 model take only 224 by 224 size
        bgr_img = cv.resize(bgr_img, (img_width, img_height), cv.INTER_CUBIC)

        # bgr to rgb color conversion
        rgb_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2RGB)

        # 3-D to 4-D
        rgb_img = np.expand_dims(rgb_img, 0)

        # prob
        preds = model.predict(rgb_img)

        # highest prob
        prob = np.max(preds)
        # index of highest prob
        class_id = np.argmax(preds)
        # Name of class + prob
        text = ('Predict: {},\nprob: {}'.format(class_names[class_id][0][0], prob))
        results.append({'label': class_names[class_id][0][0], 'prob': '{:.4}'.format(prob)})
        cv.imwrite('images/{}_out.png'.format(i), bgr_img)

    print(results)
    # json file in which the sample is stored which can be seen through visual studio
    with open('results.json', 'w') as file:
        json.dump(results, file, indent=4)

    K.clear_session()
