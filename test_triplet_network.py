import config
import utils
from keras.models import load_model
from imutils.paths import list_images
import numpy as np
import cv2
import tensorflow as tf

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

print("[INFO] loading test dataset...")
testImagePaths = list(list_images('examples'))

np.random.seed(42)
pairs = np.random.choice(testImagePaths, size=(15, 2))
# # load the model from disk
print("[INFO] loading siamese model...")
model = load_model(config.MODEL_PATH, compile=False)

for (i, (pathA, pathB)) in enumerate(pairs):
    # load both the images and convert them to grayscale
    imageA = cv2.imread(pathA, 0)
    imageB = cv2.imread(pathB, 0)
    
    # create a copy of both the images for visualization purpose
    origA = imageA.copy()
    origB = imageB.copy()
    
    #resizing the image according to the model's input requirement
    
    imageA = cv2.resize(imageA, (28,28))
    imageB = cv2.resize(imageB, (28,28))
    
    # add channel a dimension to both the images
    imageA = np.expand_dims(imageA, axis=-1)
    imageB = np.expand_dims(imageB, axis=-1)
    
    # scale the pixel values to the range of [0, 1]
    imageA = imageA / 255.0
    imageB = imageB / 255.0
    input_images = np.array([imageA, imageB])
    
    # use our siamese model to make predictions on the image pair,
    # indicating whether or not the images belong to the same class
    preds = model.predict(input_images)
    proba = preds[0][0]
    fig = plt.figure("Pair #{}".format(i + 1), figsize=(4, 2))
    plt.suptitle("Similarity: {:.2f}".format(proba))
    # show first image
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(origA, cmap=plt.cm.gray)
    plt.axis("off")
    # show the second image
    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(origB, cmap=plt.cm.gray)
    plt.axis("off")
    # show the plot
    plt.show()
