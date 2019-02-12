import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from scipy.ndimage.interpolation import rotate
import tqdm

#Image manipulation functions

def blur_image(image, strength): #Uses gaussian filter to blur image
    return ndimage.gaussian_filter(image,strength)

def shift_image(image, direction): #Shifts direction, with rollover
    return ndimage.shift(image,direction)

def add_gaussian_white_noise(image, mean, std): #Additive white noise
    return image + np.random.normal(mean,std,size=np.shape(image))

def rotate_image(image,rot_angle): #Rotates image around the center
    return rotate(image,angle=rot_angle,mode='nearest',reshape=False)


#Utility
def pop_random_element(array):
    i = np.random.randint(np.shape(array)[0])
    val = array[i]
    array = np.delete(array,i)
    return val, array

#Dataset generation, returns a dataset based on the provided training data (modified)
def generate_extra_datasets(X_train,Y_train):

    """
        Generates more datasets by modifying existing examples.
         -Picks datasets to modify randomly
         -Features
          -Blurring
          -Shifting
          -AGW
          -Rotation
         -Can combine features randomly

    """

    #Number of generated extra datasets
    blurred_extras = 0
    shifted_extras = 0
    agw_extras = 0
    rotated_extras = 0
    combination_extras = 0
    total = blurred_extras+shifted_extras+agw_extras+rotated_extras+combination_extras

    #Strengths of the available modifications, the limits of the probability density function.
    blur_strength_range = (0.01,0.2)
    shift_range = (-3,3)
    agw_mean_range = (-10,10)
    agw_std_range = (2,10)
    rotation_range = (-1,1)

    size_of_dataset = np.shape(X_train)[0]
    available_indexes = np.arange(0,size_of_dataset) #Preventing a dataset from being used twice

    #Saving modified images/results. Prevents us from using hstack() and vstack() too much. (it is slow)
    #This made ~1000 times faster. 
    new_images = []
    new_results = []

    print("Expanding dataset by modifications,", total,"new datasets to be generated.")

    print("By blurring")
    for q in tqdm.trange(blurred_extras):
        i,available_indexes = pop_random_element(available_indexes)
        image = np.reshape(X_train[i,:],(28,28))
        image = blur_image(image,np.random.uniform(blur_strength_range[0],high=blur_strength_range[1]))
        image = np.reshape(image,(1,784))
        new_images.append(image)
        new_results.append(Y_train[i])

    print("By shifting")
    for q in tqdm.trange(shifted_extras):
        i,available_indexes = pop_random_element(available_indexes)
        image = np.reshape(X_train[i,:],(28,28))
        hshift = np.random.randint(shift_range[0],high=shift_range[1])
        vshift = np.random.randint(shift_range[0],high=shift_range[1])
        image = shift_image(image,(vshift,hshift))
        image = np.reshape(image,(1,784))
        new_images.append(image)
        new_results.append(Y_train[i])

    print("By additive gaussian noise")
    for q in tqdm.trange(agw_extras):
        i,available_indexes = pop_random_element(available_indexes)
        image = np.reshape(X_train[i,:],(28,28))
        mean = np.random.uniform(agw_mean_range[0],high=agw_mean_range[1])
        std = np.random.uniform(agw_std_range[0],high=agw_std_range[1])
        image = add_gaussian_white_noise(image,mean,std)
        image = np.reshape(image,(1,784))
        new_images.append(image)
        new_results.append(Y_train[i])

    print("By rotation")
    for q in tqdm.trange(rotated_extras):
        i,available_indexes = pop_random_element(available_indexes)
        image = np.reshape(X_train[i,:],(28,28))
        image = rotate_image(image,np.random.uniform(rotation_range[0],high=rotation_range[1]))
        image = np.reshape(image,(1,784))
        new_images.append(image)
        new_results.append(Y_train[i])

    print("By random combination")
    for q in tqdm.trange(combination_extras):
        i,available_indexes = pop_random_element(available_indexes)
        image = np.reshape(X_train[i,:],(28,28))

        #Vector of shape [x, x, x, x], index 0: blur, index 1: shift, index 2: agw, index 3: rotate
        modification_vector = np.ndarray.tolist(np.random.randint(0,high=2,size=(4,1)))

        #Disabling AGW
        modification_vector[2] = 0

        #Disabling blur
        modification_vector[0] = 0

        if modification_vector[0]:
            image = blur_image(image,np.random.uniform(blur_strength_range[0],high=blur_strength_range[1]))

        if modification_vector[1]:
            hshift = np.random.randint(shift_range[0],high=shift_range[1])
            vshift = np.random.randint(shift_range[0],high=shift_range[1])
            image = shift_image(image,(vshift,hshift))

        if modification_vector[2]:
            mean = np.random.uniform(agw_mean_range[0],high=agw_mean_range[1])
            std = np.random.randint(agw_std_range[0],high=agw_std_range[1])
            image = add_gaussian_white_noise(image,mean,std)
        
        if modification_vector[3]:
            image = rotate_image(image,np.random.randint(rotation_range[0],high=rotation_range[1]))

        image = rotate_image(image,int(np.random.uniform(rotation_range[0],rotation_range[1])))
        image = np.reshape(image,(1,784))
        new_images.append(image)
        new_results.append(Y_train[i])

    #Combining original and modified dataset
    X_train_modified = np.reshape(np.array(new_images),(total,784)).astype(np.uint8)
    Y_train_modified = np.array(new_results).astype(np.uint8)

    return X_train_modified, Y_train_modified