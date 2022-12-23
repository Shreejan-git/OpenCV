from keras.models import Model
import numpy as np
from keras.layers import Input, Dense, Conv2D, Dropout, GlobalAveragePooling2D, MaxPooling2D


def build_siamese_model(inputShape=np.array([28,28,1]), embeddingDim=48):
    '''
    embeddingDim: Output dimensionality of the final fully-connected
    layer in the network.
    '''
    inputs = Input(inputShape)
    
    c1 = Conv2D(32, (2,2), padding='same', activation='relu')(inputs)
    m1 = MaxPooling2D(pool_size=(2,2))(c1)
    d1 = Dropout(0.4)(m1)
    
    c2 = Conv2D(32, (2,2), padding='same', activation='relu')(d1)
    m2 = MaxPooling2D(pool_size=(2,2))(c2)
    d2 = Dropout(0.4)(m2)
    
    pooledOutput = GlobalAveragePooling2D()(d2)
    outputs = Dense(embeddingDim)(pooledOutput)
    
    return Model(inputs=[inputs], outputs=[outputs])

if __name__ =="__main__":
    embedding = build_siamese_model()
    print(embedding.summary())