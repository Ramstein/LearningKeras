from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD, RMSprop, Adam
from keras.preprocessing.image import ImageDataGenerator


import matplotlib.pyplot as plt
import numpy as np



#cifar10 is a dataset of 60k images of 32X32 piXels on 3 channels
IMG_CHANNELS = 3
IMG_ROWS = 32
IMG_COLS = 32

#constants
BATCH_SIZE = 128
NB_EPOCH = 20
NB_CLASSES = 10
VERBOSE = 1
VALIDATION_SPLIT = 0.2
'''The training set is used to build our model,
the validation is used to select the best performing approach,
while the test set is to check the performance of our best models on fresh unseen data
'''

OPTIM = Adam()
#load dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print("X_train shape: ",X_train.shape)
print('train samples: ',X_train.shape[0] )
print('test samples: ', X_train.shape[1])


NUM_TO_AUGMENT=5
# augumenting
print("Augmenting training set images...")
datagen = ImageDataGenerator( rotation_range=40, width_shift_range=0.2,
                             height_shift_range=0.2, zoom_range=0.2,
                             horizontal_flip=True, fill_mode='nearest')


xtas, ytas = [], []
for i in range(X_train.shape[0]):
  num_aug = 0
  x = X_train[i] # (3, 32, 32)
  x = x.reshape((1,) + x.shape) # (1, 3, 32, 32)
  for x_aug in datagen.flow(x, batch_size=1, save_to_dir='preview',
                            save_prefix='cifar', save_format='jpeg'):
    if num_aug >= NUM_TO_AUGMENT:
      break
    xtas.append(x_aug[0])
    num_aug += 1
#fit the  datagen
datagen.fit(X_train)



#convert feto catetgorical
y_train = np_utils.to_categorical(y_train, NB_CLASSES)
y_test = np_utils.to_categorical(y_test, NB_CLASSES)

#float and Normalization
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /=255
X_test /=255

#network
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', input_shape = (IMG_ROWS, IMG_COLS, IMG_CHANNELS)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25)) #25% dropout

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(NB_CLASSES))
model.add(Activation('softmax'))
model.summary()

#training the model
model.compile(loss ='categorical_crossentropy', optimizer = OPTIM, matrics = ['accuracy'])
history = model.fit_generator(datagen.flow(X_train, y_train, batch_size = BATCH_SIZE),
                              samples_per_epoch=X_train.shape[0], epochs = NB_EPOCH, verbose = VERBOSE, validation_split = 0.2)
score = model.evaluate(X_test, y_test, batch_size = BATCH_SIZE, verbose = VERBOSE)
print("test score: ", score[0])
print("test accuracy: ", score[1])
