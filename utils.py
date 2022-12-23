import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from imutils import build_montages
import cv2


def make_pairs(images, labels):
    '''
    This function takes two inputs:
    images: all the image data
    labels: all the corresponding labels
    '''
    pairImages = [] #images in our dataset
    pairLabels = [] #the class labels associated with the images
    
    # print(labels[:11])
    
    numClassess = len(np.unique(labels))
    #In np.where(labels == i) we must include -> [0]. If not, we will get a list of a tuple of array.
    # https://www.geeksforgeeks.org/numpy-where-in-python/
    idx = [np.where(labels == i)[0] for i in range(0, numClassess)]
    # print('this is idx:', idx)
    
    
    for idxA in range(len(images)):
        currentImage = images[idxA]
        label = labels[idxA]
        idxB = np.random.choice(idx[label])
        # print(idxB)
        posImage = images[idxB] #positive image
        # print(posImage)
        
        pairImages.append([currentImage, posImage])
        pairLabels.append([1])
        
        negIdx = np.where(labels != label)[0]
        # print(negIdx)
        negImage = images[np.random.choice(negIdx)]
        
        pairImages.append([currentImage,negImage])
        pairLabels.append([0])
        
    return (np.array(pairImages), np.array(pairLabels))




def euclidean_distance(vectors):
    '''
    This function has one parameter: vectors.
    Vectors needs to be the 2 tuples of embeddings of two input images.
    '''
    emb1, emb2 = vectors
    sumSquared = K.sum(K.square(emb1-emb2), axis=1, keepdims=True)
    
    return K.sqrt(K.maximum(sumSquared, K.epsilon()))


def plot_training(H, plotPath):
    '''
    H is the history object from model.fit
    plotPath is the path where we are saving our figure
    '''
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H.history["loss"], label="train_loss")
    plt.plot(H.history["val_loss"], label="val_loss")
    plt.plot(H.history["accuracy"], label="train_acc")
    plt.plot(H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(plotPath)
 
 
if __name__ == '__main__':
    print('[INFO] loading MNIST dataset...')
    (trainX, trainY), (testX, testY) = mnist.load_data()

    (pairTrain, labelTrain) = make_pairs(trainX, trainY)
    (pairTest, labelTest) = make_pairs(testX, testY)

    images = []

    # loop over a sample of our training pairs
    for i in np.random.choice(np.arange(0, len(pairTrain)), size=(49,)):
        # grab the current image pair and label
        imageA = pairTrain[i][0]
        imageB = pairTrain[i][1]
        label = labelTrain[i]
        # to make it easier to visualize the pairs and their positive or
        # negative annotations, we're going to "pad" the pair with four
        # pixels along the top, bottom, and right borders, respectively
        output = np.zeros((36, 60), dtype="uint8")
        pair = np.hstack([imageA, imageB])
        output[4:32, 0:56] = pair
        # set the text label for the pair along with what color we are
        # going to draw the pair in (green for a "positive" pair and
        # red for a "negative" pair)
        text = "neg" if label[0] == 0 else "pos"
        color = (0, 0, 255) if label[0] == 0 else (0, 255, 0)
        # create a 3-channel RGB image from the grayscale pair, resize
        # it from 60x36 to 96x51 (so we can better see it), and then
        # draw what type of pair it is on the image
        vis = cv2.merge([output] * 3)
        vis = cv2.resize(vis, (96, 51), interpolation=cv2.INTER_LINEAR)
        cv2.putText(vis, text, (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
            color, 2)
        # add the pair visualization to our list of output images
        images.append(vis)


    #https://pyimagesearch.com/2017/05/29/montages-with-opencv/
    montage = build_montages(images, (96, 51), (7, 7))[0]
    # show the output montage
    cv2.imshow("Siamese Image Pairs", montage)
    cv2.waitKey(0)
