import csv
import gc
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras import regularizers
from keras import Model
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, ZeroPadding2D, BatchNormalization, Input

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
    # Get all images and classify them in lists
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

def plot_accuracy(history):
    # Plot accuracy over training period
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def plot_loss(history):
    # Plot loss over training period
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

if __name__ == '__main__':
    import_data()
    split_data_sets()
    label_train_set()
    label_test_set()
    nmp_conversion()

    file_name = "test2_more_layers.json"
    h5_name = "test2_more_layers_weight.h5";
    model, loaded = load_model(file_name, h5_name)

    if not model:
        # CNN
        visible = Input(shape=(150, 150, 1))
        conv1 = Conv2D(16, kernel_size=(3, 3), activation='relu', strides=(1, 1))(visible)
        conv2 = Conv2D(16, kernel_size=(3, 3), activation='relu', strides=(1, 1))(conv1)
        bat1 = BatchNormalization()(conv2)
        zero1 = ZeroPadding2D(padding=(1, 1))(bat1)

        conv3 = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='valid',
                       kernel_regularizer=regularizers.l2(0.05))(zero1)
        conv4 = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='valid',
                       kernel_regularizer=regularizers.l2(0.05))(conv3)
        bat2 = BatchNormalization()(conv4)

        conv5 = Conv2D(64, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='valid')(bat2)
        conv6 = Conv2D(64, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='valid')(conv5)
        bat3 = BatchNormalization()(conv6)
        pool1 = MaxPooling2D(pool_size=(2, 2))(bat3)
        zero2 = ZeroPadding2D(padding=(1, 1))(pool1)

        conv7 = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='valid',
                       kernel_regularizer=regularizers.l2(0.01))(zero2)
        conv8 = Conv2D(128, kernel_size=(2, 2), activation='relu', strides=(1, 1), padding='valid')(conv7)
        bat4 = BatchNormalization()(conv8)

        conv9 = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='valid',
                       kernel_regularizer=regularizers.l2(0.02))(bat4)
        conv10 = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='valid',
                        kernel_regularizer=regularizers.l2(0.02))(conv9)
        bat5 = BatchNormalization()(conv10)

        conv11 = Conv2D(64, kernel_size=(3, 3), activation='relu', strides=(1, 1))(bat5)
        conv12 = Conv2D(64, kernel_size=(3, 3), activation='relu', strides=(1, 1))(conv11)
        bat6 = BatchNormalization()(conv12)
        pool2 = MaxPooling2D(pool_size=(2, 2))(bat6)

        conv13 = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='valid',
                        kernel_regularizer=regularizers.l2(0.02))(pool2)
        conv14 = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='valid',
                        kernel_regularizer=regularizers.l2(0.02))(conv13)
        bat7 = BatchNormalization()(conv14)

        conv15 = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='valid',
                        kernel_regularizer=regularizers.l2(0.05))(bat7)
        conv16 = Conv2D(128, kernel_size=(2, 2), activation='relu', strides=(1, 1), padding='valid')(conv15)
        bat8 = BatchNormalization()(conv16)

        flat = Flatten()(bat8)
        hidden1 = Dense(32, activation='relu')(flat)
        drop1 = Dropout(0.3)(hidden1)

        hidden2 = Dense(32, activation='relu')(drop1)
        drop2 = Dropout(0.2)(hidden2)

        output = Dense(4, activation='sigmoid')(drop2)
        model = Model(inputs=visible, outputs=output)

    # Model summary
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
