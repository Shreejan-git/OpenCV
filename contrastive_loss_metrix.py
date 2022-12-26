import keras.backend as K
import tensorflow as tf

def contrastive_loss(y, preds, margin=1):
    ''' 
    y: the ground-truth labels of the dataset (1: images of same class and 0: images of different class)
    preds: value returned by euclidean distance 
    margin:
    '''
    y = tf.cast(y, preds.dtype)
	# calculate the contrastive loss between the true labels and
	# the predicted labels
    squaredPreds = K.square(preds)
    squaredMargin = K.square(K.maximum(margin - preds, 0))
    loss = K.mean(y * squaredPreds + (1 - y) * squaredMargin)
    # return the computed contrastive loss to the calling function
    return loss
