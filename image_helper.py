import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from scipy.ndimage.interpolation import zoom
from PIL import Image
from scipy.misc import imsave
import glob
from os.path import basename

def convert_coordinate_to_pixel(x,y,n_rows,n_cols):
    return int(x*n_rows), int(y*n_cols)


def plot_with_label(im_array, x, y):
    x = float(x)
    y = float(y)
        
    im_array = np.asarray(im_array)
        
    center_point_x, center_point_y = convert_coordinate_to_pixel(x,y,im_array.shape[1], im_array.shape[0])
        
    fig,ax = plt.subplots(1)
    ax.set_aspect('equal')
    ax.scatter([center_point_x], [center_point_y], s=5, c="r")
    ax.add_patch(Circle((center_point_x, center_point_y), 0.05*np.min([im_array.shape[1], im_array.shape[0]]), fill=False))
    ax.imshow(im_array)
    plt.show()
    
    
def load_data(path, labels):
    # returns list with (image_matrix, x, y)
    
    data = list()
    data_label = list()
    for label in labels:
        x = float(label[1])
        y = float(label[2])
        
        image = Image.open(path + label[0])
        image.load()
        im_array = np.asarray(image)
        
        
        data.append(im_array)
        data_label.append([x,y])
        

    return np.asarray(data)/255, np.asarray(data_label)


############################################################################
######################Augmentation##########################################
############################################################################


def translate_random(im, x, y, n):
    # Translates image randomly n times and folds background back on other side
    c = 1
    result = list()
    result_label = list()
    result.append(im)
    result_label.append([x,y])
    while c < n+1:
        x_translation = (np.random.rand()-0.5)*2
        y_translation = (np.random.rand()-0.5)*2
        
        offset_x = 0.05
        offset_y = 0.05
        if x - x_translation > offset_x and x - x_translation < 1-offset_x and y - y_translation > offset_y and y - y_translation < 1-offset_y:
            result_label.append([x-x_translation,y-y_translation])
            
            x_translation, y_translation = convert_coordinate_to_pixel(y_translation, x_translation, im.shape[0], im.shape[1])
            new_im = np.roll(im, -y_translation, axis=1)
            new_im = np.roll(new_im, -x_translation, axis=0)
            
            result.append(new_im)
            c+=1
            
    return result, result_label


def scale_random(im, x, y, n, order = 1):
    def clipped_zoom(img, zoom_factor, x, y, order = 1):
    
        h, w = img.shape[:2]

        # For multichannel images we don't want to apply the zoom factor to the RGB
        # dimension, so instead we create a tuple of zoom factors, one per array
        # dimension, with 1's for any trailing dimensions after the width and height.
        zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

        # Zooming out
        if zoom_factor < 1:

            # Bounding box of the zoomed-out image within the output array
            zh = int(np.round(h * zoom_factor))
            zw = int(np.round(w * zoom_factor))
            top = (h - zh) // 2
            left = (w - zw) // 2

            # Zero-padding
            out = np.zeros_like(img)
            result_x = x*zoom_factor + left/out.shape[1]
            result_y = y*zoom_factor + top/out.shape[0]



            out[top:top+zh, left:left+zw] = zoom(img, zoom_tuple, order=order)



        # Zooming in
        elif zoom_factor > 1:

            # Bounding box of the zoomed-in region within the input array
            zh = int(np.round(h / zoom_factor))
            zw = int(np.round(w / zoom_factor))
            top = (h - zh) // 2
            left = (w - zw) // 2


            out = zoom(img[top:top+zh, left:left+zw], zoom_tuple, order=order)

            result_x = (x - left/out.shape[1])*zoom_factor 
            result_y = (y - top/out.shape[0])*zoom_factor  

            # `out` might still be slightly larger than `img` due to rounding, so
            # trim off any extra pixels at the edges
            trim_top = ((out.shape[0] - h) // 2)
            trim_left = ((out.shape[1] - w) // 2)
            out = out[trim_top:trim_top+h, trim_left:trim_left+w]

        # If zoom_factor == 1, just return the input array
        else:
            out = img
        return out, result_x, result_y
    
    k = 0
    result = list()
    result_label = list()
    result.append(im)
    result_label.append([x,y])
    while k < n:
        zoom_factor = np.random.rand()+0.75
        new_im, result_x, result_y = clipped_zoom(im,zoom_factor,x,y, order = order)
        if result_x > 0 and result_x < 1 and result_y > 0 and result_y < 1 and new_im.shape == im.shape:
            result.append(new_im)
            result_label.append([result_x, result_y])
        k += 1
            
    return result, result_label




def flip_along_all_axis(im, x, y):
    result = list()
    result_label = list()
    
    result.append(im)
    result_label.append([x,y])
    
    result.append(np.fliplr(im))
    result_label.append([1-x,y])
    
    result.append(np.flipud(im))
    result_label.append([x,1-y])
    
    result.append(np.fliplr(np.flipud(im)))
    result_label.append([1-x,1-y])
    
    return result, result_label








def augment_all(images, labels, n_translations=10, n_scales=1):
    result = list()
    result_label = list()
    for i, im in enumerate(images):
        #print(i)
        augmented_images, augmented_label = flip_along_all_axis(im, labels[i][0], labels[i][1])
        for k, im_ in enumerate(augmented_images):
            augmented_images_, augmented_label_ = translate_random(im_, augmented_label[k][0], augmented_label[k][1],n_translations)
            for s, im__ in enumerate(augmented_images_):
                augmented_images__, augmented_label__ = scale_random(im__, augmented_label_[s][0], augmented_label_[s][1], n_scales)
                result.extend(augmented_images__)
                result_label.extend(augmented_label__)
    return np.array(result), np.array(result_label)

def save_augmented_data(path, augmented_data, augmented_label, qualifier):
    for i in range(augmented_data.shape[0]):
        imsave(path + "data/" + qualifier + "_{0}.png".format(i), augmented_train_data[i])
        np.savetxt(path + "label/" + qualifier + "_{0}.txt".format(i), augmented_train_label[i])
        
def load_augmented_data(path, qualifier):
    augmented_data = list()
    augmented_label = list()
    
    def load_augmented_data(path, qualifier):
    augmented_data = list()
    augmented_label = list()
    
    for filename in glob.glob(path + "data/" + qualifier + "*.png"):
        im=np.array(Image.open(filename))
        im = im/255
        augmented_data.append(im)
        
        label = np.loadtxt(path + "label/" + basename(filename).split(".")[0] + ".txt")
        augmented_label.append(label)
        
    return np.array(augmented_data), np.array(augmented_label)
        
    return np.array(augmented_data), np.array(augmented_label)


def get_generator_dicts(path):
    partition = {"train": [], "val": [], "test": []}
    labels = {}
    
    for filename in glob.glob(path + "data/train_*.png"):
        partition["train"].append(basename(filename).split(".")[0])
        
    for ID in partition["train"]:
        labels[ID] = np.loadtxt(path + "label/" + ID + ".txt")
        
        
    for filename in glob.glob(path + "data/val*.png"):
        partition["val"].append(basename(filename).split(".")[0])
        
    for ID in partition["val"]:
        labels[ID] = np.loadtxt(path + "label/" + ID + ".txt")
        
        
    for filename in glob.glob(path + "data/test*.png"):
        partition["test"].append(basename(filename).split(".")[0])
        
    for ID in partition["test"]:
        labels[ID] = np.loadtxt(path + "label/" + ID + ".txt")
        
    return partition, labels




###################################################################################
#####################################Fully Convolutional###########################
###################################################################################

def transform_label(x,y,n_rows,n_cols, width = 0.01):
    label = np.zeros((n_rows,n_cols))
    width_pixel = np.max([int(np.min([n_rows,n_cols])*width), 1])
    
    label[int(y*n_rows)-width_pixel:int(y*n_rows)+width_pixel, int(x*n_cols)-width_pixel:int(x*n_cols)+width_pixel] = 1
    return label

def evaluate_label(label):
    ys, xs = np.where(label == 1)
    xs = xs/label.shape[1]
    ys = ys/label.shape[0]
    return np.mean(xs), np.mean(ys)



















