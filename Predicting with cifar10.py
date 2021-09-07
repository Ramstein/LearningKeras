import numpy as np
import scipy.misc
from keras.models import model_from_json
from keras.optimizers import SGD

#load  model
model_architecture = 'DCNN_for_recognising_digits_.json'
model_weights = 'cifar10vgg.h5'

model = model_from_json(open(model_architecture))
model.load_weights(model_weights)


#load images
img_names = ['cheetah-leopard-animal-big-87403.jpeg',
            'pexels-photo-572861.jpeg',
            'pexels-photo-952077.jpeg',
            'tabby-cat-close-up-portrait-69932.jpeg']
imgs = [np.transpose(scipy.misc.imresize(scipy.misc.imread(img_names), (32, 32)),
                    (1, 0, 2)).astype('float32') for img_name in img_names]


imgs = np.array(imgs) / 255

#train
optim = SGD()
model.compile(loss='categorical_crossentropy', optimizer = optim, metrics = ['accuracy'])

#predict
predictions = model.predict_classes(imgs)
print(predictions)