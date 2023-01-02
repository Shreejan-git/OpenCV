import tensorflow as tf
import tensorflow_addons as tfa
from keras.datasets import mnist
import tensorflow_datasets as tfd
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout, Input, Lambda, MaxPooling2D
from keras.models import Model
import config
import utils



def _pairwise_distances(embeddings, squared=False):
    """Compute the 2D matrix of distances between all the embeddings.
    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.
    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """
    # Get the dot product between all embeddings
    # shape (batch_size, batch_size)
    dot_product = tf.matmul(embeddings, tf.transpose(embeddings))

    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability (the diagonal of the result will be exactly 0).
    # shape (batch_size,)
    square_norm = tf.diag_part(dot_product) #get a diagonal of a matrix

    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # The shape of square_norm is (batch_size,). tf.expand_dims will make the shape (batch_size, batch_size)
    distances = tf.expand_dims(square_norm, 1) - 2.0 * dot_product + tf.expand_dims(square_norm, 0)

    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances = tf.maximum(distances, 0.0)
    
    if not squared:
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        #tf.euqal is doing: if distances == 0.0
        mask = tf.to_float(tf.equal(distances, 0.0))
        distances = distances + mask * 1e-16

        distances = tf.sqrt(distances)

        # Correct the epsilon added: set the distances on the mask to be exactly 0.0
        distances = distances * (1.0 - mask)

    return distances

def _normalize_img(img, label):
    img = tf.cast(img, tf.float32) / 255.
    return (img, label)

train_dataset,test_dataset = tfd.load(name="mnist", split=['train', 'test'], as_supervised=True)

print('[INFO] this is from tensorflow datast')

# Build your input pipelines
train_dataset = train_dataset.shuffle(1024).batch(config.BATCH_SIZE)
train_dataset = train_dataset.map(_normalize_img)

test_dataset = test_dataset.batch(config.BATCH_SIZE)
test_dataset = test_dataset.map(_normalize_img)

def build_siamese_model(inputShape, embeddingDim=48):
    '''
    embeddingDim: Output dimensionality of the final fully-connected
    layer in the network.
    '''
    inputs = Input(inputShape)
    
    c1 = Conv2D(64, (10,10), padding='same', activation='relu')(inputs)
    m1 = MaxPooling2D(pool_size=2)(c1)
        
    c2 = Conv2D(128, (7,7), activation='relu', padding='same')(m1) #originally there was 128 filters
    # m2 = MaxPool2D(64, pool_size=(2,2), padding='same')(c2)
    m2 = MaxPooling2D(pool_size=2)(c2)
    
    c3 = Conv2D(128, (4,4), activation='relu', padding='same')(m2) #originally there was 128 filters
    # m3 = MaxPool2D(64, pool_size=(2,2), padding='same')(c3)
    
    c4 = Conv2D(256, (4,4), activation='relu', padding='same')(c3) #originally there was 256 filters
    
    f1 = Flatten()(c4)
    
    outputs = Dense(embeddingDim)(f1)
    
    output_ = Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(outputs)
    
    return Model(inputs=[inputs], outputs=[output_])


model = build_siamese_model((28,28,1))


history = model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tfa.losses.TripletSemiHardLoss(),
    metrices=['accuracy'])

history = model.fit(
    train_dataset,
    validation_data = test_dataset,
    epochs=config.EPOCHS)


model.save(config.MODEL_PATH)

print("[INFO] plotting training history...")
utils.plot_training(history, config.PLOT_PATH)