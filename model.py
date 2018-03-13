""" behvioural cloning model """

# imports
import cv2
import csv
from sklearn.model_selection import train_test_split
from keras.layers import Activation, Dense, Flatten, Lambda, Cropping2D
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.models import Sequential
import numpy as np

# global file locations
FILE_DIR = "../data/"
DATA_FILE = "driving_log.csv"
CORRECTED_PATH = FILE_DIR + "IMG/"

# update this value if path info is windows (w) \ or linux (l) /
FILE_FROM = "l"


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
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X, y, validation_split=0.2, shuffle=True, nb_epoch=5)
    model.save("model.h5")


def gen_training_model(X_train, X_valid):
    """ function to train model """

    # create data generators
    batch_size = 32
    X_gen_train = generate_data(X_train, FILE_FROM, batch_size=batch_size)
    X_gen_valid = generate_data(X_valid, FILE_FROM, batch_size=batch_size)

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
                        samples_per_epoch=len(X_train) * 6,
                        nb_epoch=3, validation_data=X_gen_valid,
                        nb_val_samples=len(X_valid) * 6)
    model.save("model.h5")


if __name__ == "__main__":
    train_data, validation_data = read_data_from_file()
    #features, measurements = transform_data(train_data, FILE_FROM)
    #training_model(features, measurements)
    gen_training_model(train_data, validation_data)
