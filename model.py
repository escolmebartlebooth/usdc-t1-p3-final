"""
    behvioural cloning model
    project 3 of udacity self driving cars
    built in March 2018
    David Escolme
"""

# imports section
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
CORRECTED_PATH = FILE_DIR + "IMG/"

# if using paths like this \ use 'w' else use 'l'
FILE_FROM = "l"

# parameters for training
NB_EPOCHS = 3
BATCH_SIZE = 126


def flip_image(img, angle):
    """
        apply a horizontal flip transformation
        adjust steering angle for flip
        accepts: an img and a steering angle
        returns: a flipped image and new angle
    """
    return cv2.flip(img, 1), angle * -1


def warp_image(img, angle):
    """
        applies a warp on image
        calculates new angle
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


def img_generator(X, batch_size=32, validate=False):

    sample_size = len(X)

    while 1:
        shuffle(X)

        for offset in range(0, sample_size, batch_size):

            batch_samples = X[offset:offset+batch_size]

            X_train = []
            y_train = []

            for index, item in batch_samples.iterrows():
                # create data - if validate just pass the center image
                if validate == True:
                    img_path = FILE_DIR+item['center'].lstrip()
                    img = cv2.imread(img_path)
                    angle = item['steering']
                else:
                    choice = random.choice([('center', 0), ('left', 0.25),
                                           ('right', -0.25)])
                    # img_path = CORRECTED_PATH+item[choice[0]].split('/')[-1]
                    img_path = FILE_DIR+item[choice[0]].lstrip()
                    img = cv2.imread(img_path)
                    angle = item['steering']+choice[1]
                    if item['steering'] == 0:
                        # do something
                        keep_prob = random.random()
                        if keep_prob < 0.3:
                            # get warped image 90% of time
                            img, angle = warp_image(img, angle)
                    else:
                        prob_image = random.random()
                        if prob_image < 0.3:
                            img, angle = warp_image(img, angle)
                        elif prob_image < 0.6:
                            img, angle = flip_image(img, angle)

                X_train.append(img)
                y_train.append(angle)

            yield shuffle(np.array(X_train), np.array(y_train))


def read_data_from_file():
    """ function to read in image data from csv
        and to correct for image folder
        returns a list of images for use in training
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

    train_data, validation_data = train_test_split(data_list, test_size=0.2)
    return train_data, validation_data


def transform_data(X, file_from="l"):
    features = []
    measurements = []

    for item in X:
        # add centre, left and right images and adjust steering
        for i in range(3):
            # check whether data from windows or linux
            if file_from == "w":
                features.append(cv2.imread(CORRECTED_PATH +
                                           item[i].split("\\")[-1]))
            else:
                features.append(cv2.imread(CORRECTED_PATH +
                                           item[i].split("/")[-1]))
            if i == 0:
                correction_factor = 0
            elif i == 1:
                correction_factor = 0.2
            else:
                correction_factor = -0.2
            measurements.append(float(item[3])+correction_factor)

    # now build augmented images
    aug_features, aug_measurements = [], []
    for feature, measurement in zip(features, measurements):
        aug_features.append(feature)
        aug_measurements.append(measurement)
        # now also add a flipped image
        aug_features.append(cv2.flip(feature, 1))
        aug_measurements.append(measurement*-1.0)

    return np.array(aug_features), np.array(aug_measurements)


def generate_data(X, file_from="l", batch_size=32, validate=False):
    """
        generator function for training and validation data
        this adds more non zero angle images
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
            for item in batch_samples:
                # add centre image for zero angle
                for i in range(3):
                    # check whether data from windows or linux
                    if file_from == "w":
                        features.append(cv2.imread(CORRECTED_PATH +
                                                   item[i].split("\\")[-1]))
                    else:
                        features.append(cv2.imread(CORRECTED_PATH +
                                                   item[i].split("/")[-1]))
                    if i == 0:
                        correction_factor = 0
                    elif i == 1:
                        correction_factor = 0.2
                    else:
                        correction_factor = -0.2
                    measurements.append(float(item[3])+correction_factor)

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


def training_model(X, y):
    """ function to train model """
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
    model.save("model.h5")


def gen_training_model(X_train, X_valid):
    """ function to train model """

    # create data generators
    X_gen_train = img_generator(X_train, batch_size=BATCH_SIZE)
    X_gen_valid = img_generator(X_valid, batch_size=BATCH_SIZE,
                                validate=True)

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
    # train_data, validation_data = read_data_from_file()
    # features, measurements = transform_data(train_data, FILE_FROM)
    # training_model(features, measurements)
    df = pd.read_csv(FILE_DIR+DATA_FILE)
    train_data, validation_data = train_test_split(df, test_size=0.2)
    gen_training_model(train_data, validation_data)
