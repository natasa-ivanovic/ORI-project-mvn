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


def import_data():
    #  Get all images and classify them in lists
    global normal_set
    global bacteria_set
    global virus_set
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

def split_data_sets():
    # Randomly shuffle images before splitting for training and testing
    global normal_set
    global bacteria_set
    global virus_set
    global train_set
    global test_set
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

def label_train_set():
    global train_set
    global x_train, y_train
    global normal_set
    global bacteria_set
    global virus_set
    global nRows, nCols
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

def label_test_set():
    global test_set
    global normal_set
    global bacteria_set
    global virus_set
    global nRows, nCols
    global x_test, y_test
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

def nmp_conversion():
    global x_train, x_test
    global y_train, y_test
    global nRows, nCols
    # Convert to Numpy Arrays
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    x_train = x_train.reshape([len(train_set),nRows, nCols, 1])
    x_test = x_test.reshape([len(test_set),nRows, nCols, 1])

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

def serialize_model(model, file_name, h5_name):
    print("Started saving model in JSON...")
    model_json = model.to_json()
    with open(file_name, "w") as json_file:
        json_file.write(model_json)

    #serialize weights to HDF5
    model.save_weights(h5_name)
    print("Model successfully saved to model.json")

def load_model(file_name, h5_name):
    try:
        json_file = open(file_name, "r")
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(h5_name)
        print("Loaded model from file")
        return loaded_model, True
    except Exception:
        print("HUR DUR! File does not exists!")
        return None, False

if __name__ == '__main__':
    import_data()
    split_data_sets()
    label_train_set()
    label_test_set()
    nmp_conversion()

    file_name = "test3_lightCNN.json"
    h5_name = "test3_lightCNN_weight.h5"
    model, loaded = load_model(file_name, h5_name)
    if not model:
        # CNN
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

    if not loaded:
        history = model.fit(x_train, to_categorical(y_train), validation_data=(x_test, to_categorical(y_test)), epochs=10, shuffle=True)
        plot_accuracy(history)
        plot_loss(history)
        serialize_model(model, file_name, h5_name)

    total = y_test.size
    # Predict
    predictions = model.predict(x_test[:total])

    # Our model's predictions.
    predictions_max = np.array(np.argmax(predictions, axis=1))

    # Check our predictions against the ground truths.
    check = predictions_max == y_test
    unique, counts = np.unique(check, return_counts=True)
    dict = dict(zip(unique, counts))
    correct = dict[True]
    print(correct)
    print("Finished evaluation")
    print("Result: ", correct,"/", total, " correct")
    print("Accuracy: ", correct/total*100, "%")
