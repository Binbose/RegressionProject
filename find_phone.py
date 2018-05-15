from keras.models import load_model
from image_helper import *
import argparse


###########################################################################################
############################### READ COMMANDLINE ##########################################
###########################################################################################

parser = argparse.ArgumentParser(description='Augments the data and trains the model on this data.')
parser.add_argument('path', metavar='p', help='Path to the train images')
parser.add_argument('--use_pretrained', metavar='u', help='Boolean: Whether the pretrained model should be used or the custom trained model', default=True)
parser.add_argument('--visualize', metavar='v', help='Boolean: Whether the output should be plotted over the image', default=False)
args = parser.parse_args()

path = args.path
use_pretrained = args.use_pretrained
visualize = args.visualize



###########################################################################################
############################### LOAD DATA #################################################
###########################################################################################

data = load_images(path)
for i,im in enumerate(data):
    plt.imshow(im)
    plt.show()
    
    
###########################################################################################
############################### DOWNSAMPLING ##############################################
###########################################################################################

resize_factor = 0.6
n_sample = data.shape[0]
n_rows_new, n_cols_new, n_channel = imresize(data[0], resize_factor).shape

data_resized = np.zeros((n_sample_train, n_rows_new, n_cols_new, 3))

for i, im in enumerate(data):
    data_resized[i] = imresize(im, resize_factor)/255.0
    
data = data_resized


###########################################################################################
############################### LOAD MODEL AND PREDICT ####################################
###########################################################################################

if use_pretrained:
    model = load_model("./pretrained_model/model.hdf5")
else:
    model = load_model("./checkpoints/model.hdf5")

predictions = model.predict(data)

for i, pred in enumerate(predictions):
    print str(pred[0]) + " " + str(pred[1])
    
    if visualize:
        plot_with_label(augmented_validation_data[i], pred[0], pred[1])
    