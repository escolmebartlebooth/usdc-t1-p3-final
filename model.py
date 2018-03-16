"""
    behvioural cloning model
    project 3 of udacity self driving cars
    built in March 2018
    David Escolme
"""

# imports
import sys
import pandas as pd
import numpy as np
import cv2
import csv
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import random
from keras.layers import Activation, Dense, Flatten, Lambda, Cropping2D
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.models import Sequential


# globals
FILE_DIR = "../data/"
DATA_FILE = "m_driving_log.csv"


# parameters for training
NB_EPOCHS = 5
BATCH_SIZE = 32


def flip_image(img, angle):
    """
        apply a horizontal flip transformation and
        adjust steering angle for flip

        Args:
            img: an image
            angle: a steering angle

        Returns: a flipped image and new angle
    """
    return cv2.flip(img, 1), angle * -1


def warp_image(img, angle):
    """
        applies a warp on image and calculates new angle
        Args:
            img: an image
            angle: the steering angle
        Returns:
            a warped image with adjusted angle
    """
    h, w, c = img.shape

    WIDTH_SHIFT_RANGE = 100
    HEIGHT_SHIFT_RANGE = 40

    # Translation
    tx = WIDTH_SHIFT_RANGE * np.random.uniform() - WIDTH_SHIFT_RANGE / 2
    ty = HEIGHT_SHIFT_RANGE * np.random.uniform() - HEIGHT_SHIFT_RANGE / 2
    angle = angle + tx / WIDTH_SHIFT_RANGE * 2 * .2

    transform_matrix = np.float32([[1, 0, tx],
                                   [0, 1, ty]])

    warped_image = cv2.warpAffine(img, transform_matrix, (w, h))

    return warped_image, angle


def read_data_from_file():
    """ function to read in image data from csv
        and to correct for image folder
        Returns:
            a list of images for use in training
    """

    data_list = []
    with open(FILE_DIR+DATA_FILE, 'rt') as f:
        # ignore first line if header
        img_data = csv.reader(f)
        firstline = 0
        for line in img_data:
            if firstline == 0:
                firstline = 1
            else:
                data_list.append(line)

    return data_list


def transform_data(X):
    """
        used to provide augmented data to non generator model
        Args:
            X: assumed to be an array
        Returns:
            numpy arrays of Images and Angles
    """

    features = []
    measurements = []

    for item in X:
        # add centre, left and right images and adjust steering
        for i in range(3):
            # check whether data from windows or linux
            features.append(cv2.imread(FILE_DIR+item[i].lstrip()))
            if i == 0:
                correction_factor = 0
            elif i == 1:
                correction_factor = 0.2
            else:
                correction_factor = -0.2
            measurements.append(float(item[3])+correction_factor)

    # now build augmented images
    aug_features, aug_measurements = [], []
    i = 0
    for feature, measurement in zip(features, measurements):
        aug_features.append(feature)
        aug_measurements.append(measurement)
        # now also add a flipped image for every other image
        if i % 2 == 0:
            aug_features.append(cv2.flip(feature, 1))
            aug_measurements.append(measurement*-1.0)
        i += 1

    return np.array(aug_features), np.array(aug_measurements)


def generate_data(X, batch_size=32, validate=False):
    """
        generator function for training and validation data
        this adds more non zero angle images
        this version mimics the first training model data
        Args:
            X: Image and Angle pandas dataframe
            batch_size: batch size for optimisation
            validate: whether the generator is used for validation
        Returns:
            yields numpy batches of images and angles
    """

    sample_size = len(X)

    # run forever...
    while 1:
        # shuffle the data
        shuffle(X)
        # generate a sample batch
        for offset in range(0, sample_size, batch_size):
            # slice off the next batch
            batch_samples = X[offset:offset+batch_size]

            # placeholders for the images and angles
            features = []
            measurements = []

            # loop the batch
            for index, item in batch_samples.iterrows():
                # add all camera images with corrected angle
                for i in range(3):
                    if i == 0:
                        img_path = 'center'
                        correction_factor = 0
                    elif i == 1:
                        img_path = 'left'
                        correction_factor = 0.25
                    else:
                        img_path = 'right'
                        correction_factor = -0.25
                    features.append(cv2.imread(FILE_DIR +
                                               item[img_path].lstrip()))
                    measurements.append(float(item['steering']) +
                                        correction_factor)

            # now build augmented images
            aug_features, aug_measurements = [], []
            for feature, measurement in zip(features, measurements):
                aug_features.append(feature)
                aug_measurements.append(measurement)
                # now also add a flipped image
                aug_features.append(cv2.flip(feature, 1))
                aug_measurements.append(measurement*-1.0)

            yield shuffle(np.array(aug_features),
                          np.array(aug_measurements))


def img_generator(X, batch_size=32, validate=False):
    """
        generator function for training and validation data
        this adds more non zero angle images
        this version uses random generation to balance the data
        Args:
            X: Image and Angle pandas dataframe
            batch_size: batch size for optimisation
            validate: whether the generator is used for validation
        Returns:
            yields numpy batches of images and angles
    """

    sample_size = len(X)

    # run forever
    while 1:
        shuffle(X)

        # slice the shuffled data
        for offset in range(0, sample_size, batch_size):
            # create a batch
            batch_samples = X[offset:offset+batch_size]

            X_train = []
            y_train = []

            for index, item in batch_samples.iterrows():
                # create data - if validate just pass the center image
                if validate is True:
                    img_path = FILE_DIR+item['center'].lstrip()
                    img = cv2.imread(img_path)
                    angle = item['steering']
                else:
                    # random choice of image
                    choice = random.choice([('center', 0), ('left', 0.25),
                                           ('right', -0.25)])
                    img_path = FILE_DIR+item[choice[0]].lstrip()
                    img = cv2.imread(img_path)
                    angle = item['steering']+choice[1]
                    # if zero angle warp by a probability
                    if item['steering'] == 0:
                        keep_prob = random.random()
                        if keep_prob < 0.7:
                            # get warped image x% of time
                            img, angle = warp_image(img, angle)
                    else:
                        # if non zero angle random augmentation
                        prob_image = random.random()
                        if prob_image < 0.3:
                            img, angle = warp_image(img, angle)
                        elif prob_image < 0.6:
                            img, angle = flip_image(img, angle)

                # convert image to RGB to match drive.py
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                X_train.append(img)
                y_train.append(angle)

            yield shuffle(np.array(X_train), np.array(y_train))


def model1(X, y):
    """
        function to train model - over all data
        Args:
            X: training data
            y: training labels
    """

    model = Sequential()
    model.add(Lambda(lambda x: x/255.0 - 0.5,
              input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    model.add(Convolution2D(6, 5, 5))
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(6, 5, 5))
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Activation('relu'))
    model.add(Dense(120, activation='relu'))
    model.add(Dense(84, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X, y, validation_split=0.2, shuffle=True, nb_epoch=5)
    model.save("model1.h5")


def model2(X_train, X_valid):
    """
        function to train model using first generator
        Args:
            X_train: training data
            X_valid: validation data
    """

    # create data generators
    X_gen_train = generate_data(X_train, batch_size=BATCH_SIZE)
    X_gen_valid = generate_data(X_valid, batch_size=BATCH_SIZE,
                                validate=False)

    model = Sequential()
    model.add(Lambda(lambda x: x/255.0 - 0.5,
              input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    model.add(Convolution2D(6, 5, 5))
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(6, 5, 5))
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    history = model.fit_generator(X_gen_train,
                                  samples_per_epoch=len(X_train)*6,
                                  nb_epoch=NB_EPOCHS,
                                  validation_data=X_gen_valid,
                                  nb_val_samples=len(X_valid)*6)
    model.save("model2.h5")


def model3(X_train, X_valid):
    """
        function to train model over final generator
        Args:
            X_train: training data
            X_valid: validation data
    """

    # create data generators
    X_gen_train = img_generator(X_train, batch_size=BATCH_SIZE)
    X_gen_valid = img_generator(X_valid, batch_size=BATCH_SIZE,
                                validate=False)

    model = Sequential()
    model.add(Lambda(lambda x: x/255.0 - 0.5,
              input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    model.add(Convolution2D(6, 5, 5))
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(6, 5, 5))
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    history = model.fit_generator(X_gen_train,
                                  samples_per_epoch=len(X_train)*3,
                                  nb_epoch=NB_EPOCHS,
                                  validation_data=X_gen_valid,
                                  nb_val_samples=len(X_valid)*3)
    model.save("model.h5")


if __name__ == "__main__":
    """
        allows for different models and default model to be run
        if incorrect model passed, error reported
    """
    if sys.argv[-1] == 'model1':
        train_data = read_data_from_file()
        features, measurements = transform_data(train_data)
        model1(features, measurements)
    elif sys.argv[-1] == 'model2':
        df = pd.read_csv(FILE_DIR+DATA_FILE)
        train_data, validation_data = train_test_split(df, test_size=0.2)
        model2(train_data, validation_data)
    elif sys.argv[-1] == 'model3':
        df = pd.read_csv(FILE_DIR+DATA_FILE)
        train_data, validation_data = train_test_split(df, test_size=0.2)
        model3(train_data, validation_data)
    else:
        print('model: {} not understood'.format(sys.argv[-1]))
