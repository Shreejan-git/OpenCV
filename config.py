import os
'''
Below link is imp to know the process of save our model.
https://www.tensorflow.org/guide/keras/save_and_serialize#savedmodel_format


'''

IMG_SHAPE = (28,28,1)

BATCH_SIZE = 10
EPOCHS = 10

BASE_OUTPUT = 'output'

MODEL_PATH = os.path.sep.join([BASE_OUTPUT, 'siamese_model'])
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, 'plot.png'])

if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)
    
# if not os.path.exists(PLOT_PATH):
