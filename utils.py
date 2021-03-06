import cv2 as cv

from resnet_50 import resnet50_model


def load_model():
    model_weights_path = 'G:/fyp/resnet_50/model/epoch_100_aug/model.48-0.89.hdf5'
    img_width, img_height = 224, 224
    num_channels = 3
    num_classes = 196
    model = resnet50_model(img_height, img_width, num_channels, num_classes)
    model.load_weights(model_weights_path, by_name=True)
    return model


def draw_str(dst, target, s1, s2):
    x, y = target
    cv.rectangle(dst, (x-220, y-220), (x-10, y-10), (255, 0, 0), 2)
    # cv.putText(dst, s1, (x - 270, y - 200), cv.FONT_HERSHEY_SIMPLEX, 0.40, (255, 0, 255), thickness=2)
    # cv.putText(dst, s2, (x - 270, y - 180), cv.FONT_HERSHEY_SIMPLEX, 0.40, (255, 0, 255), thickness=2)

