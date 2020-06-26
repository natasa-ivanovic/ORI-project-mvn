import csv
import gc
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Add,  Input, Maximum, Permute,  Reshape, BatchNormalization, Dense,   Dropout, Flatten
from keras.layers import Cropping1D, Cropping2D, Conv2D, MaxPooling2D

def import_data(normal_set, bacteria_set, virus_set):
    #  Get all images and classify them in lists
    with open('chest_xray_metadata.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            if row[2].lower() == 'normal':
                normal_set.append(row[1])
            else:
                if row[4].lower() == 'bacteria':
                    bacteria_set.append(row[1])
                elif row[4].lower() == 'virus':
                    virus_set.append(row[1])
    return normal_set, bacteria_set, virus_set

def split_data_sets(normal_set, bacteria_set, virus_set, train_set, test_set):
    # Randomly shuffle images before splitting for training and testing
    random.shuffle(normal_set)
    random.shuffle(bacteria_set)
    random.shuffle(virus_set)

    # Split data to training and testing image sets
    # normal_set - 1342 (671)
    # bacteria_set - 2535 (1267)
    # virus_set - 1407 (703)
    train_set = normal_set[:671] + bacteria_set[:1267] + virus_set[:703]
    test_set = normal_set[671:] + bacteria_set[1267:] + virus_set[703:]

    return train_set, test_set

def label_train_set(train_set, x_train, y_train, nRows, nCols):
    # Read and label each image in the training set
    for image in train_set:
        try:
            path = 'chest_xray_data_set/' + image
            x_train.append(cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (nRows, nCols), interpolation=cv2.INTER_CUBIC))
            if image in normal_set:
                y_train.append(1)
            elif image in bacteria_set:
                y_train.append(2)
            elif image in virus_set:
                y_train.append(3)
        except Exception:
            print('Failed to format: ', image)

    return x_train, y_train

def label_test_set(test_set, x_test, y_test, nRows, nCols):
    # Read and label each image in the testing set
    for image in test_set:
        try:
            path = 'chest_xray_data_set/' + image
            x_test.append(cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (nRows, nCols), interpolation=cv2.INTER_CUBIC))
            if image in normal_set:
                y_test.append(1)
            elif image in bacteria_set:
                y_test.append(2)
            elif image in virus_set:
                y_test.append(3)
        except Exception:
            print('Failed to format: ', image)

    return x_test, y_test

def nmp_conversion(x_train, x_test, y_train, y_test, nRows, nCols):
    # Convert to Numpy Arrays
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    x_train = x_train.reshape([len(train_set),nRows, nCols, 1])
    x_test = x_test.reshape([len(test_set),nRows, nCols, 1])
    #y_train = y_train.reshape([len(train_set),nRows, nCols, 1])
    #y_test = y_test.reshape([len(test_set),nRows, nCols, 1])
    return x_train, x_test, y_train, y_test

def clean(train_set, test_set):
    # Garbage collection
    del train_set, test_set
    gc.collect()


def categorical(y_train, y_test):
    # Switch targets to categorical
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    return y_train, y_test


def conv2d_bn(x, filters, kernel_size, strides=1, padding='same', activation='relu', use_bias=False, name=None):
  x = Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias, kernel_initializer='he_normal', name=name)(x)
  if not use_bias:
    bn_axis = 1 if K.image_data_format() == 'channels_first' else 3
    bn_name = None if name is None else name + '_bn'
    x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
  return x

def mfm(x):
  shape = K.int_shape(x)
  x = Permute(dims=(3, 2, 1))(x) # swap 1 <-> 3 axis
  x1 = Cropping2D(cropping=((0, shape[3] // 2), 0))(x)
  x2 = Cropping2D(cropping=((shape[3] // 2, 0), 0))(x)
  x = Maximum()([x1, x2])
  x = Permute(dims=(3, 2, 1))(x) # swap 1 <-> 3 axis
  x = Reshape([shape[1], shape[2], shape[3] // 2])(x)
  return x

def common_conv2d(net, filters, filters2, iter=1):
  res = net

  for v in range(iter):
    net = conv2d_bn(net, filters=filters, kernel_size=3, strides=1, padding='same')
    net = mfm(net)
    net = conv2d_bn(net, filters=filters, kernel_size=3, strides=1, padding='same')
    net = mfm(net)
    net = Add()([net, res]) # residual connection

  net = conv2d_bn(net, filters=filters, kernel_size=1, strides=1, padding='same')
  net = mfm(net)
  net = conv2d_bn(net, filters=filters2, kernel_size=3, strides=1, padding='same')
  net = mfm(net)

  return net

def lcnn29(inputs):
  # Conv1
  net = conv2d_bn(inputs, filters=96, kernel_size=5, strides=1, padding='same')
  net = mfm(net)
  net = MaxPooling2D(pool_size=2, strides=2, padding='same')(net)

  # Block1
  net = common_conv2d(net,filters=96, filters2=192, iter=1)
  net = MaxPooling2D(pool_size=2, strides=2, padding='same')(net)

  # Block2
  net = common_conv2d(net,filters=192, filters2=384, iter=2)
  net = MaxPooling2D(pool_size=2, strides=2, padding='same')(net)

  # Block3
  net = common_conv2d(net,filters=384, filters2=256, iter=3)

  # Block4
  net = common_conv2d(net,filters=256, filters2=256, iter=4)
  net = MaxPooling2D(pool_size=2, strides=2, padding='same')(net)

  net = Flatten()(net)

  return net

def plot_accuracy():
    # Plot accuracy over training period
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def plot_loss():
    # Plot loss over training period
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()



if __name__ == '__main__':
    # Sets for imported data
    normal_set = []
    bacteria_set = []
    virus_set = []

    # Training and validation set
    train_set = []
    test_set = []

    # Training and testing set labeling
    x_train = []
    x_test = []
    y_train = []
    y_test = []

    # Image pre-processing
    nRows = 150  # width
    nCols = 150  # height
    channels = 1  # grayscale

    normal_set, bacteria_set, virus_set = import_data(normal_set, bacteria_set, virus_set)
    train_set, test_set = split_data_sets(normal_set, bacteria_set, virus_set, train_set, test_set)
    x_train, y_train = label_train_set(train_set, x_train,y_train, nRows, nCols)
    x_test, y_test = label_test_set(test_set, x_test, y_test, nRows, nCols)
    x_train, x_test, y_train, y_test = nmp_conversion(x_train, x_test, y_train, y_test, nRows, nCols)
    clean(train_set, test_set)
    y_train, y_test = categorical(y_train, y_test)

    input_image = Input(shape=(nRows, nCols, channels))

    lcnn_output = lcnn29(inputs=input_image)

    fc1 = Dense(512, activation=None)(lcnn_output)
    fc1 = Reshape((512, 1))(fc1)
    fc1_1 = Cropping1D(cropping=(0, 256))(fc1)
    fc1_2 = Cropping1D(cropping=(256, 0))(fc1)
    fc1 = Maximum()([fc1_1, fc1_2])
    fc1 = Flatten()(fc1)

    out = Dense(4, activation='linear')(fc1)
    model = Model(inputs=[input_image], outputs=out)

    # Model summary
    print("Model summary")
    print(model.summary())

    # Compile and train the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, shuffle=True)

    plot_accuracy()
    plot_loss()