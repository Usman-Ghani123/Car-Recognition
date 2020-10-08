# -*- coding: utf-8 -*-

import os

import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing import image
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from utils import load_model


def decode_predictions(preds, top=5):
    results = []
    # preds is basically the list of probabilities
    for pred in preds:
        
        # it will print the last 5 values and will reverse the array
        top_indices = pred.argsort()[-top:][::-1]

        # it will store that array
        result = [(class_names[i], pred[i]) for i in top_indices]

        # It will print the first element of the array
        result.sort(key=lambda x: x[1], reverse=True)

        results.append(result)
    return results


def predict(img_dir, model):
    img_files = []
    
    #we will go down the directory in data/valid
    for root, dirs, files in os.walk(img_dir, topdown=False):
        for name in files:
            # this is the name of files in each directory
            # append is a function of list to make a list of arrays
            img_files.append(os.path.join(root, name))
    # sort the list like it will reverse it
    img_files = sorted(img_files)

    y_pred = []
    y_test = []
    
    for img_path in tqdm(img_files):
        # it will load img 
        img = image.load_img(img_path, target_size=(224, 224))

        # convert the img to array
        x = image.img_to_array(img)

        # predict the image 
        preds = model.predict(x[None, :, :, :])

        # function created above
        decoded = decode_predictions(preds, top=1)

        # predicted label
        pred_label = decoded[0][0][0]
        # print(pred_label)
        
        y_pred.append(pred_label)

        # to split the array
        tokens = img_path.split("\\")

        # extract the label given by us
        class_id = int(tokens[-2])
        
        # print(str(class_id))
        y_test.append(class_id)

    return y_pred, y_test


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        # each value in particular row will be drived by the sum of all elements in that row.
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def calc_acc(y_pred, y_test):
    num_corrects = 0
    for i in range(num_samples):
        pred =int(y_pred[i])
        test = int(y_test[i])
        if pred == test:
            num_corrects += 1
    return num_corrects / num_samples


if __name__ == '__main__':
    img_width, img_height = 224, 224
    num_channels = 3
    num_classes = 196
    class_names = range(1, (num_classes + 1))
    num_samples = 1629

    print("\nLoad the trained ResNet model....")
    model = load_model()

    #  y_predict is the labels predicted by the classifier and y_test are the orignal labels given by use
    y_pred, y_test = predict('data/valid', model)
    print("y_pred: " + str(y_pred))
    print("y_test: " + str(y_test))

    acc = calc_acc(y_pred, y_test)
    print("%s: %.2f%%" % ('acc', acc * 100))

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                          title='Normalized confusion matrix')

    plt.show()



