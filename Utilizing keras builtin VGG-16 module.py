from keras.models import Model
from keras.preprocessing import image

import matplotlib.pyplot as plt
import cv2, numpy as np


from keras.callbacks import EarlyStopping


from keras.optimizers import Adam
from keras.applications.resnet50 import ResNet50

from keras.applications.inception_v3 import InceptionV3


from keras.layers import Input, Dense, Flatten, Conv2D, Dropout


#prebuilt model eith prebuilt weights on imagenet
input_tensor = Input(shape=(256, 1600, 3))

base_model = ResNet50(input_tensor=input_tensor, input_shape=(256, 1600, 3),
                 weights='imagenet', include_top=False,
                 pooling='max', classes=4, data_format='channel_last'
                 )

for layer in base_model.layers[:130]:
    layer.trainable = False
for layer in base_model.layers[130:]:
    layer.trainable = True

model = base_model.output

model = Conv2D(512, kernel_size=3,input_shape=(2048, ), activation='relu',padding='same', )(model)  # we add dense layers so that the model can learn more complex functions and classify for better results.
model = Dropout(0.2)(model)

model = Conv2D(256, kernel_size=3, activation='sigmoid',padding='same', )(model)  # we add dense layers so that the model can learn more complex functions and classify for better results.
model = Dropout(0.2)(model)

model = Dropout(rate=0.2, )

predictions = Conv2D(4, kernel_size=3, activation='sigmoid',padding='same', )(model)  # we add dense layers so that the model can learn more complex functions and classify for better results.

model = Model(input=base_model.input, output=predictions)

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy', dice_coef])

model.summary()

from keras.preprocessing.image import ImageDataGenerator

BATCH_SIZE = 8
def create_test_gen():
    return ImageDataGenerator(rescale=1/255.).flow_from_dataframe(
        test_imgs,
        directory='../input/severstal-steel-defect-detection/test_images',
        x_col='ImageId',
        class_mode='categorical',
        target_size=(256, 1600),
        batch_size=BATCH_SIZE,
        shuffle=False
    )
test_gen = create_test_gen()




#resize into VGG16 trained images
im = cv2.resize(cv2.imread('tabby-cat-close-up-portrait-69932.jpeg'), (224, 224))
im = np.expand_dims(im, axis=0)

#predict
out = model.predict(im)

print(np.argmax(out)) #this will print 820 for steaming train
plt.xlabel('prediction')

plt.plot(out.ravel())
plt.show()



