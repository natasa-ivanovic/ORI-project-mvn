import csv
import gc
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model as load_params

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
nRows = 250  # width
nCols = 250  # height
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
                y_train.append(0)
            elif image in bacteria_set:
                y_train.append(1)
            elif image in virus_set:
                y_train.append(2)
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
                y_test.append(0)
            elif image in bacteria_set:
                y_test.append(1)
            elif image in virus_set:
                y_test.append(2)
        except Exception:
            print('Failed to format: ', image)

def nmp_conversion():
    global x_train, x_test
    global y_train, y_test
    global nRows, nCols
    # Convert to Numpy Arrays
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_train = x_train.reshape([len(train_set),nRows, nCols, 1])

    x_test = np.array(x_test)
    y_test = np.array(y_test)

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

def serialize_model(model, file_name, h5_name):
    print("Started saving model in JSON...")
    model_json = model.to_json()
    with open(file_name, "w") as json_file:
        json_file.write(model_json)

    #serialize weights to HDF5
    model.save_weights(h5_name)
    print("Model successfully saved to model.json!")

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

    file_name = "test4_image_preprocessing.json"
    h5_name = "test4_image_preprocessing.h5"
    model, loaded = load_model(file_name, h5_name)

    if not model:
        weight_decay = 1e-4
        # CNN
        model = Sequential()
        model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(250, 250, 1), kernel_regularizer=l2(weight_decay)))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))
        model.add(Conv2D(32, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(weight_decay)))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))
        model.add(Conv2D(64, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(weight_decay)))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))
        model.add(Conv2D(64, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(weight_decay)))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))
        model.add(Conv2D(128, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(weight_decay)))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))
        model.add(Conv2D(128, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(weight_decay)))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(3, activation='softmax'))


    # normalization
    train_mean = np.mean(x_train)
    train_std = np.std(x_train)

    test_mean = np.mean(x_test)
    test_std = np.std(x_test)

    x_train = (x_train-train_mean)/train_std
    x_test = (x_test-test_mean)/test_std

    # adding variation to training set
    image_data_gen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=False
        )
    image_data_gen.fit(x_train)
    # Model summary
    print("Model summary")
    print(model.summary())

    # adding early stopping
    es = EarlyStopping(monitor='val_accuracy', mode='max', min_delta=0.01, patience=100, verbose=1, restore_best_weights=True)

    # adding checkpointing (save best iteration)

    checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', save_best_only=True, verbose=1)

    opt = RMSprop(lr=0.0005, decay=1e-6)

    # Compile and train the model
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    if not loaded:
        history = model.fit(image_data_gen.flow(x_train, to_categorical(y_train), batch_size=32),
                            validation_data=(x_test, to_categorical(y_test)), steps_per_epoch=len(x_train) / 32,
                            epochs=200, shuffle=True, callbacks=[es, checkpoint])

        plot_accuracy(history)
        plot_loss(history)

        # model = load_params('best_model.h5')

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
    print("Finished evaluation on best model")
    print("Result: ", correct,"/", total, " correct")
    print("Accuracy: ", correct/total*100, "%")
