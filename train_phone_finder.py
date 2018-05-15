import keras
from keras import Model
from keras.layers import Conv2D, Input, Dense,Flatten,LeakyReLU, BatchNormalization
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras import backend as K
import tensorflow as tf

import os
import argparse

from image_helper import *
from MyDataGenerator import DataGenerator


###########################################################################################
############################### READ COMMANDLINE ##########################################
###########################################################################################

parser = argparse.ArgumentParser(description='Augments the data and trains the model on this data.')
parser.add_argument('path', metavar='p', help='Path to the train images')
parser.add_argument('use_pretrained', metavar='u', help='Boolean: Whether to use pretrained model and fine tune it, or to train model from scratch', defeault=True)
args = parser.parse_args()

path = args.path
use_pretrained = args.path


###########################################################################################
############################### LOAD DATA #################################################
###########################################################################################

lines = [line.rstrip('\n') for line in open(path + "labels.txt")]
labels = list()
for line in lines:
    labels.append(line.split(" "))
    
    
data, data_label = load_data(path, labels)

split = int(data.shape[0]*0.8)
train_data = data[:split]
validation_data = data[split:]


train_label = data_label[:split]
validation_label = data_label[split:]


###########################################################################################
############################### AUGMENT DATA ##############################################
###########################################################################################

if not os.path.exists("./augmented_data/"):
    os.makedirs("./augmented_data/")

    augmented_train_data, augmented_train_label = augment_all(train_data, train_label, n_translations=0, n_scales=0)
    save_augmented_data("./augmented_data/", augmented_train_data, augmented_train_label, "train")

    augmented_validation_data, augmented_validation_label = augment_all(validation_data, validation_label, n_translations=0, n_scales=0)
    save_augmented_data("./augmented_data/", augmented_validation_data, augmented_validation_label, "val")
else:  
    augmented_train_data, augmented_train_label = load_augmented_data("./augmented_data/", "train")
    augmented_validation_data, augmented_validation_label = load_augmented_data("./augmented_data/", "val")
    augmented_test_data, augmented_test_label = load_augmented_data("./augmented_data/", "test")

# In case dataset wouldbe bigger and doesnt fit in memory, this is for using a data generator:
# partition_dict, labels_dict = get_generator_dicts("./augmented_data/")

# Note: Data is not normalized (only scaled in range [0,1]), as the dataset is to small, so that calculating mean and stdv
# would be biased and not estimate real mean and stdv of the distribution correctly, thereby leading to overfitting, as 
# experiments also confirmed


###########################################################################################
############################### DOWNSAMPLING ##############################################
###########################################################################################

resize_factor = 0.6
n_sample_train = augmented_train_data.shape[0]
n_sample_validation = augmented_train_validation.shape[0]
n_rows_new, n_cols_new, n_channel = imresize(augmented_train_data[0], resize_factor).shape

augmented_train_data_resized = np.zeros((n_sample_train, n_rows_new, n_cols_new, 3))
augmented_validation_data_resized = np.zeros((n_sample_validation, n_cols_new, 3))

for i, im in enumerate(augmented_train_data):
    augmented_train_data_resized[i] = imresize(im, resize_factor)/255.0
    

for i, im in enumerate(augmented_validation_data):
    augmented_validation_data_resized[i] = imresize(im, resize_factor)/255.0
    
augmented_train_data = augmented_train_data_resized
augmented_validation_data = augmented_validation_data_resized


###########################################################################################
############################### TRAIN MODEL ###############################################
###########################################################################################

# Define some callbacks for early stopping and checkpoints, as well as tensorboard log files for analysis of the model
tbCallBack = TensorBoard(log_dir='./logs', histogram_freq=2, batch_size=32, write_graph=True, write_grads=True, write_images=True)
cpCallBack = ModelCheckpoint('./checkpoints/model.hdf5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
esCallBack = EarlyStopping(monitor='val_loss', min_delta=0, patience=6, verbose=0, mode='auto')

# Either load pretrained model to fine tune it, or learn it from scratch
if use_pretrained:
    model = load_model("./pretrained_model/model.hdf5")
else:
    input_img = Input(shape=(augmented_train_data[0].shape[0], augmented_train_data[0].shape[1], 3))

    # Note: No Batch Normalization, as it seemed to cause overfitting. Same reason as before. 
    #(Alternative explanation: As some images look quite similiar, the data could get artificially stretched to much)
    x = Conv2D(8, (3, 3), padding='same')(input_img)
    x = LeakyReLU(alpha=0.3)(x)
    x = Conv2D(8, (3, 3), padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Conv2D(1, (1, 1), padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Flatten()(x)

    # Set bias to false in this layer. This prevents network from learning local minima: "always predict mean (x=y=0.5)" 
    # by learning all weights to become 0 except bias weight to 0.5
    output = Dense(2, use_bias=False)(x)

    model = Model(inputs=input_img, outputs=output)
    model.compile(optimizer="adam", loss='mean_squared_error')


# fit model either with data generator, if data doesnt fit in memory, 
# or just normally (which is done here, as the dataset is small):
#params = {'dim': (augmented_train_data[0].shape[0],augmented_train_data[0].shape[1]),
#          'batch_size': 32,
#          'n_channels': 3,
#          'shuffle': True}
#training_generator = DataGenerator("./augmented_data/", partition_dict['train'], labels_dict, **params)
#validation_generator = DataGenerator("./augmented_data/", partition_dict['val'], labels_dict, **params)

#history = model.fit_generator(generator=training_generator, validation_data=[augmented_validation_data, augmented_validation_label], epochs=30, callbacks=[cpCallBack, esCallBack])

history = model.fit(augmented_train_data, augmented_train_label, validation_data=[augmented_validation_data, augmented_validation_label] ,epochs=500, batch_size=32, callbacks=[tbCallBack, cpCallBack, esCallBack])





