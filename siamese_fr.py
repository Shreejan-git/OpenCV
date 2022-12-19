import numpy as np
import cv2
import os
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Layer, Dense, Conv2D, MaxPool2D, Flatten, Input
import uuid

# link to the dataset: http://vis-www.cs.umass.edu/lfw/
# link to the paper: https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf

POS_PATH = os.path.join('data','positive')
NEG_PATH = os.path.join('data','negative')
ANC_PATH = os.path.join('data','anchor')

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
# make_dir(POS_PATH)
# make_dir(NEG_PATH)
# make_dir(ANC_PATH)

def transfer_image():
    for sub_dir in os.listdir('lfw'):
        for image in os.listdir(os.path.join('lfw',sub_dir)):
            current_path = os.path.join('lfw',sub_dir,image)
            destination_path = os.path.join(NEG_PATH,image)
            os.replace(current_path, destination_path)
    
# transfer_image()

def data_collector(ANC_PATH, POS_PATH, name):
    '''
    collecting the image data for anchor and postive class using a webcam.
    '''
    cap = cv2.VideoCapture(0)
    print('dfa')
    while cap.isOpened():
        ret, frame = cap.read()
        
        frame = frame[120:120+250, 200:200+250, :]
        
        cv2.imshow('image data collection', frame) #displaying the image
        
        if cv2.waitKey(1) & 0XFF == ord('a'):
            print('collecting the images for anchor class')
            path = os.path.join(ANC_PATH, f'{name}_{uuid.uuid1()}.jpg' )
            cv2.imwrite(path, frame)
            
        if cv2.waitKey(1) & 0XFF == ord('p'):
            print('Collecting the image for positive class')
            path = os.path.join(POS_PATH, f'{name}_{uuid.uuid1()}.jpg')
            cv2.imwrite(path, frame)
            
        if cv2.waitKey(1) & 0XFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()
    
# data_collector(ANC_PATH, POS_PATH, 'shreejan')