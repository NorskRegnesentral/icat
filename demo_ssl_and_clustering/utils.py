import cv2
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from datetime import datetime

def date_string():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def downsample_to_N(array, N, random=False):
    if len(array) <= N:
        return array

    input_was_list = type(array)==list

    array = np.array(array)

    if not random:
            ds = len(array) // N
            array = array[::ds]
    else:
        array = np.random.choice(array, size=N, replace=False)

    if input_was_list:
        array = array.tolist()

    return array


def imscatter(x, y, images, n_imgs_pr_axis = 50, image_resolution = 100, crop_scheme = 'min'):
    assert crop_scheme in ['height', 'width', 'min']
    out_image = np.zeros((n_imgs_pr_axis*image_resolution, n_imgs_pr_axis*image_resolution, 3), dtype='uint8') + 255

    X = np.linspace(0, (n_imgs_pr_axis - 1) * image_resolution, n_imgs_pr_axis, dtype='uint16')
    Y = np.linspace(0, (n_imgs_pr_axis - 1) * image_resolution, n_imgs_pr_axis, dtype='uint16')

    grid_point_has_img = np.zeros((n_imgs_pr_axis, n_imgs_pr_axis), dtype='bool')

    def normalize_feature_vec(x):
        x = np.array(x).copy()
        x -= np.min(x)
        x /= np.max(x)
        x *= (n_imgs_pr_axis - 1) * image_resolution
        x += image_resolution/2
        return x

    x = normalize_feature_vec(x)
    y = normalize_feature_vec(y)

    for x0, y0, image in tqdm(zip(x, y, images), desc='Making an overview of cluster', total=len(images)):

        xi = int(np.argmin(np.abs(X - x0)))
        yi = int(np.argmin(np.abs(Y - y0)))

        if grid_point_has_img[xi,yi]:
            continue

        grid_point_has_img[xi, yi] = True

        image = plt.imread(image)
        if len(image.shape) ==2:
            image = np.concatenate([image[:,:,None]]*3,-1) #For greyscale images
            
        if crop_scheme == 'height' or (crop_scheme == 'min' and image.shape[0] > image.shape[1]):
            if image.shape[0] > image.shape[1]:
                s = (image.shape[0]-image.shape[1])//2
                image = image[s:s+image.shape[1],:,:]
        elif crop_scheme == 'width' or (crop_scheme == 'min' and image.shape[0] < image.shape[1]):
            if image.shape[0] < image.shape[1]:
                s = (image.shape[1]-image.shape[0])//2
                image = image[:, s:s+image.shape[0],:]
        elif crop_scheme == 'min':
            pass
        else:
            raise  NotImplementedError('crop_scheme: {} not implemented'.format(crop_scheme))

        image =  cv2.resize(image, (image_resolution, image_resolution))

        out_image[X[xi]:X[xi]+image_resolution, Y[yi]:Y[yi]+image_resolution] = image
    out_image = np.flipud(np.moveaxis(out_image, 0, 1))
    return out_image

#Take a pytorch variable and make numpy [Copied from the bamjo package]

def var_to_np(var, delete_var=False):
    tmp_var = var
    if type(tmp_var) in [np.array, np.ndarray]:
        return tmp_var

    #If input is list we do this for all elements
    if type(tmp_var) == type([]):
        out = []
        for v in tmp_var:
            out.append(var_to_np(v))
        return out

    try:
        tmp_var = tmp_var.cpu()
    except:
        None
    try:
        tmp_var = tmp_var.data
    except:
        None
    try:
        tmp_var = tmp_var.numpy()
    except:
        None

    if type(tmp_var) == tuple:
        tmp_var = tmp_var[0]

    if delete_var:
        del var
    return tmp_var
