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

normal_set = []
bacteria_set = []
virus_set = []

# 1) Get all images and classify them in lists
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

# 2) Randomly shuffle images before splitting for training and testing
random.shuffle(normal_set)
random.shuffle(bacteria_set)
random.shuffle(virus_set)

# 3) Split data to training and testing image sets
# normal_set - 1342 (671)
# bacteria_set - 2535 (1267)
# virus_set - 1407 (703)
train_set = normal_set[:671] + bacteria_set[:1267] + virus_set[:703]
test_set = normal_set[671:] + bacteria_set[1267:] + virus_set[703:]

# 4) Image pre-processing
nRows = 150 # width
nCols = 150 # height
channels = 1 # grayscale

# 5) Training and testing set labeling
x_train = []
x_test = []
y_train = []
y_test = []

# 6) Read and label each image in the training set
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

# 7) Read and label each image in the testing set
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

# 8) Convert to Numpy Arrays
x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

x_train = x_train.reshape([len(train_set),nRows, nCols, 1])
x_test = x_test.reshape([len(test_set),nRows, nCols, 1])
#y_train = y_train.reshape([len(train_set),nRows, nCols, 1])
#y_test = y_test.reshape([len(test_set),nRows, nCols, 1])

# 9) Garbage collection
del train_set, test_set
gc.collect()


# 10) Switch targets to categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 11) CNN
visible = Input(shape=(150,150,1))
conv1 = Conv2D(16, kernel_size=(3,3), activation='relu', strides=(1, 1))(visible)
conv2 = Conv2D(16, kernel_size=(3,3), activation='relu', strides=(1, 1))(conv1)
bat1 = BatchNormalization()(conv2)
zero1 = ZeroPadding2D(padding=(1, 1))(bat1)

conv3 = Conv2D(32, kernel_size=(3,3), activation='relu', padding='valid', kernel_regularizer=regularizers.l2(0.05))(zero1)
conv4 = Conv2D(32, kernel_size=(3,3), activation='relu', padding='valid', kernel_regularizer=regularizers.l2(0.05))(conv3)
bat2 = BatchNormalization()(conv4)

conv5 = Conv2D(64, kernel_size=(3,3), activation='relu',strides=(1, 1), padding='valid')(bat2)
conv6 = Conv2D(64, kernel_size=(3,3), activation='relu',strides=(1, 1), padding='valid')(conv5)
bat3 = BatchNormalization()(conv6)
pool1 = MaxPooling2D(pool_size=(2, 2))(bat3)
zero2 = ZeroPadding2D(padding=(1, 1))(pool1)

conv7 = Conv2D(128, kernel_size=(3,3), activation='relu', padding='valid', kernel_regularizer=regularizers.l2(0.01))(zero2)
conv8 = Conv2D(128, kernel_size=(2,2), activation='relu', strides=(1, 1), padding='valid')(conv7)
bat4 = BatchNormalization()(conv8)

conv9 = Conv2D(64, kernel_size=(3,3), activation='relu', padding='valid', kernel_regularizer=regularizers.l2(0.02))(bat4)
conv10 = Conv2D(64, kernel_size=(3,3), activation='relu', padding='valid', kernel_regularizer=regularizers.l2(0.02))(conv9)
bat5 = BatchNormalization()(conv10)

conv11 = Conv2D(64, kernel_size=(3,3), activation='relu', strides=(1, 1))(bat5)
conv12 = Conv2D(64, kernel_size=(3,3), activation='relu', strides=(1, 1))(conv11)
bat6 = BatchNormalization()(conv12)
pool2 = MaxPooling2D(pool_size=(2, 2))(bat6)

conv13 = Conv2D(64, kernel_size=(3,3), activation='relu', padding='valid', kernel_regularizer=regularizers.l2(0.02))(pool2)
conv14 = Conv2D(64, kernel_size=(3,3), activation='relu', padding='valid', kernel_regularizer=regularizers.l2(0.02))(conv13)
bat7 = BatchNormalization()(conv14)

conv15 = Conv2D(128, kernel_size=(3,3), activation='relu', padding='valid', kernel_regularizer=regularizers.l2(0.05))(bat7)
conv16 = Conv2D(128, kernel_size=(2,2), activation='relu', strides=(1, 1), padding='valid')(conv15)
bat8 = BatchNormalization()(conv16)

flat = Flatten()(bat8)
hidden1 = Dense(32, activation='relu')(flat)
drop1 = Dropout(0.3)(hidden1)

hidden2 = Dense(32, activation='relu')(drop1)
drop2 = Dropout(0.2)(hidden2)

output = Dense(4, activation='sigmoid')(drop2)
model = Model(inputs=visible, outputs=output)

# 12) Model summary
print("Model summary")
print(model.summary())

# 13) Compile and train the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10)

# 14) Plot accuracy over training period
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# 15) Plot loss over training period
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()