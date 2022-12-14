import numpy as np
import cv2
import os
import tensorflow as tf
from keras.models import Model
from keras.layers import Layer, Dense, Conv2D, MaxPool2D, Flatten, Input
import uuid
from keras.metrics import Precision, Recall

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

#taking 60 images of each class
anchor = tf.data.Dataset.list_files(ANC_PATH + '/*.jpg').take(60)
positive = tf.data.Dataset.list_files(POS_PATH + '/*.jpg').take(60)
negative = tf.data.Dataset.list_files(NEG_PATH +'/*.jpg').take(60)

# generator_ = anchor.as_numpy_iterator()
# print(generator_.next())

def image_preprocess(img_path):
    '''
    reading the image and resizing it to (105,105). 
    (105,105) is the recommended size in the paper.
    '''
    image = tf.io.read_file(img_path)
    img = tf.io.decode_jpeg(image)
    
    #scaling (normalizing) the image to 0-1 pixel value
    img = tf.image.resize(img, (100,100))
    
    img = img/255.0
    
    return img

#labeling the image, i.e. (anchor, positive == 1) $ (anchor, negative ==0)
positives = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
negatives = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
data = positives.concatenate(negatives) #this contains the data in the form of generator
# samples = data.as_numpy_iterator()
# print(samples.next())

def preprocess_twin(input_img, validation_img, label):
    return (image_preprocess(input_img), image_preprocess(validation_img), label)


data = data.map(preprocess_twin)
data = data.cache()
data = data.shuffle(buffer_size=60)

#training partition
train_data = data.take(round(len(data)*.7))
train_data = train_data.batch(16)
train_data = train_data.prefetch(8)

train_samples = train_data.as_numpy_iterator()
train_sample = train_samples.next()
# print(len(train_sample[0]))

#testing partition
test_data = data.skip(round(len(data)*.7))
test_data = test_data.take(round(len(data)*.3))
test_data = test_data.batch(16)
test_data = test_data.prefetch(8)

def make_embeddings():
    inp = Input(shape=(100,100,3), name='input_image')
    
    c1 = Conv2D(64, (10,10), activation='relu')(inp)
    m1 = MaxPool2D(64, (2,2), padding='same')(c1)
    
    c2 = Conv2D(64, (7,7), activation='relu')(m1) #originally there was 128 filters
    m2 = MaxPool2D(64, (2,2), padding='same')(c2)
    
    c3 = Conv2D(64, (4,4), activation='relu')(m2) #originally there was 128 filters
    m3 = MaxPool2D(64, (2,2), padding='same')(c3)
    
    c4 = Conv2D(128, (4,4), activation='relu')(m3) #originally there was 255 filters
    
    f1 = Flatten()(c4)
    
    d1 = Dense(4096, activation='sigmoid',)(f1)
    
    return Model(inputs=[inp], outputs=[d1], name='embedding')
    
embedding = make_embeddings()
# embedding.summary()

class L1Dist(Layer):
    def __init__(self, **kwargs):
        super().__init__()
    
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)
        
def make_siamese_network():
    input_image = Input(shape=(100,100,3), name='input_img')
    
    validation_image = Input(shape=(100,100,3), name='validation_img')
    
    siamese_layer = L1Dist()
    siamese_layer._name = 'distance'
    distances = siamese_layer(embedding(input_image),embedding(validation_image))
    
    classifier = Dense(1, activation='sigmoid')(distances)
    
    return Model(inputs=[input_image, validation_image], outputs=[classifier], name='SiameseNetwork')
    
    
# m = make_siamese_network()
# print(m.summary)
binary_cross_loss = tf.losses.BinaryCrossentropy() #from_logit = True 
opt = tf.keras.optimizers.Adam(1e-4)

# establishing the checkpoint
siamese_model = make_siamese_network()
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint = tf.train.Checkpoint(opt=opt, siamese_model=siamese_model)

@tf.function
def train_step(batch):
    
    with tf.GradientTape() as tape:
        
        #x has anchor and positive or negative image
        X = batch[:2]
        #y has the label
        y = batch[2]
        
        #forward pass
        #training = True research
        yhat = siamese_model(X, training=True)
        
        #calculate loss
        
        loss = binary_cross_loss(y, yhat)
        
        
    #calculate gradients
    grad = tape.gradient(loss, siamese_model.trainable_variables)
    
    #calculate updated weights and apply to siamese model
    opt.apply_gradients(zip(grad, siamese_model.trainable_variables))
    
    return loss
  

def train(data, EPOCHS):
    
    #loop through epochs
    for epoch in range(1, EPOCHS+1):
        print('\n Epoch {}/{}'.format(epoch, EPOCHS))
        progbar = tf.keras.utils.Progbar(len(data))
                                            
        #loop through each batch
        for idx, batch in enumerate(data):
            train_step(batch)
            progbar.update(idx+1)
            
        #save checkpoints
        if epoch % 10 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
            
        
    
EPOCHS = 15
# train(train_data, EPOCHS)


#get a btach of test data
test_input, test_val, y_true = test_data.as_numpy_iterator().next()

test_var = test_data.as_numpy_iterator().next() 

y_hat = siamese_model.predict([test_input, test_val])
print(y_hat)

predicted_value = [1 if prediction > 0.5 else 0 for prediction in y_hat]

#creating a matric object
m = Recall()
#calculating the recall value
m.update_state(y_true, y_hat)
#return recall result
m.result().numpy()


#save model
siamese_model.save('siamesemodel.h5')


model = tf.keras.models.load_model('siamesemodel.h5', custom_objects={'L1Dist':L1Dist})

#if we want to predict the using our custom build model
#model.predict([test_input, test_val])

