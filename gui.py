import tkinter as tk
import tkinter.font
import cv2
from PIL import ImageTk, Image
import os
import random
import cv2 as cv
import keras.backend as K
import numpy as np
import scipy.io
from utils import load_model
from tkinter import filedialog
from utils import draw_str


def open_folder():

    global folder_path
    filename = filedialog.askdirectory()
    folder_path.set(filename)


def get_path(file_path):
      update = ''
      label['text'] = update
      label_3['text'] = update
      predict_image(file_path + '/')


def predict_image(file_path):
  try:
    test_images = [f for f in os.listdir(file_path) if
                   os.path.isfile(os.path.join(file_path, f)) and f.endswith('.jpg')or f.endswith('png')]

    num_samples = 1
    samples = random.sample(test_images, num_samples)
    for i, image_name in enumerate(samples):

        filename = os.path.join(file_path, image_name)
        path = f'Start processing image:\n{filename}'
        label.config(text=label.cget('text') + path + '\n')

        tokens = image_name.split('.')
        img_name = tokens[0]

        bgr_img = cv.imread(filename)
        render_img = cv.resize(bgr_img, (img_width, img_height), cv.INTER_CUBIC)
        rgb_img = cv.cvtColor(render_img, cv.COLOR_BGR2RGB)
        rgb_img = np.expand_dims(rgb_img, 0)
        preds = model.predict(rgb_img)

        prob = np.max(preds)
        class_id = np.argmax(preds)
        format_result(class_names[class_id][0][0], prob)

        render_img = cv.resize(render_img, (img_width, img_height), cv.INTER_CUBIC)
        height, width = render_img.shape[:2]
        model_1 = draw_str(render_img, (height, width), f'Class: {class_names[class_id][0][0]}', 'Prob: {:.4}'.format(prob))
        cv.imwrite(f'img_copy.jpg', render_img)

        load_img = Image.open(f'img_copy.jpg')
        render_img_1 = ImageTk.PhotoImage(load_img)
        label_3.image = render_img_1
        label_3.configure(image=render_img_1)


  except:

        label['text'] = 'There is some problem in directory or image\nTry again'
        label_3['text'] = 'No Image Found'


def format_result(cl_id, prob):
    lab = {}
    lab['cl'] = cl_id
    lab['labl'] = '{:.4}'.format(prob)
    final_str = 'Class: {}\nProbibility: {}\n'.format(cl_id, lab['labl'])
    label.config(text=label.cget('text') + '\n' + final_str + '\n')


def my_quit():

    quit()


# MAIN CODE ######################################################################################################

Height = 480
Width = 700

img_width, img_height = 224, 224


model = load_model()
model.load_weights('G:/fyp/resnet_50/model/epoch_100_aug/model.84-0.88.hdf5')

cars_meta = scipy.io.loadmat('devkit/cars_meta')
class_names = cars_meta['class_names']  # shape=(1, 196)
class_names = np.transpose(class_names)


root = tk.Tk()
root.title('Car Recognition Tool')
root.iconbitmap('icon.ico')

canvas = tk.Canvas(root, height=Height, width=Width)
canvas.pack()

img = cv2.imread('landscape.png')
img = cv2.resize(img, dsize=(Width, Height))
cv2.imwrite('new_landscape.png', img)
#
background_image = tk.PhotoImage(file='new_landscape.png')
image_label = tk.Label(root, image=background_image)
image_label.place(relheight=1, relwidth=1)


frame = tk.Frame(root, bg='Gray', bd=5)
frame.place(relx=0.5, rely=0.1, relheight=0.1, relwidth=0.85, anchor='n')

label_1 = tk.Label(root, bg='light grey', bd=5)
label_1.place(relx=0.120, rely=0.025, relheight=0.07, relwidth=0.6)
label_1['text'] = '*provide the directory of the image e.g data/test\n**1 random image will be recognized at a time'

button_2 = tk.Button(frame, text="Input Folder", activebackground='black', activeforeground='blue', font=('courier', 9), command=lambda: open_folder())
button_2.place(relx=0.01, rely=0, relheight=1, relwidth=0.185)

folder_path = tk.StringVar()
entry = tk.Entry(frame, font=('courier', 8), textvariable=folder_path)
entry.place(relx=0.2, rely=0, relheight=1, relwidth=0.48)


button = tk.Button(frame, text="Enter", activebackground='black', activeforeground='blue', font=('courier', 10), command=lambda: get_path(entry.get()))
button.place(relx=0.7, rely=0, relheight=1, relwidth=0.3)

label_2 = tk.Label(root, bg='light grey', bd=5)
label_2.place(relx=0.25, rely=0.20, relheight=0.05, relwidth=0.40)
label_2['text'] = 'Results will be shown here'

lower_frame = tk.Frame(root, bg='Gray', bd=10)
lower_frame.place(relx=0.5, rely=0.25, relheight=0.6, relwidth=1, anchor='n')

label = tk.Label(lower_frame, font=('courier', 8), anchor='nw', justify='left',  bd=5)
label.place(relheight=1, relwidth=0.56)

label_3 = tk.Label(lower_frame, font=('courier', 8), anchor='n', justify='left',  bd=5)
label_3.place(relx=0.56, rely=0, relheight=1, relwidth=0.45)

button_1 = tk.Button(root, text="Quit", activebackground='Red', activeforeground='black', font=('courier', 10), command=my_quit)
button_1.place(relx=0.47, rely=0.87, relheight=0.1, relwidth=0.09)


root.mainloop()
