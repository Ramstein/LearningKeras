import matplotlib as mpl
mpl.use("Agg") #this line allows mpl to run with no display defined

'''importing all required modules
'''
import pandas as pd
import numpy as np
from keras.layers import Dense, Reshape, Flatten, Dropout, LeakyReLU, Activation, BatchNormalization, SpatialDropout2D, Input
from keras.layers.convolutional import Conv2D, UpSampling2D, MaxPooling2D, AveragePooling2D
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from keras.regularizers import L1L2
from keras.datasets import mnist


'''
importing Adversarial GANs Modules
'''

from keras_adversarial import AdversarialModel, ImageGridCallback, simple_gan, gan_targets
from keras_adversarial import AdversarialOptimizerSimultaneous, normal_latent_sampling, fix_names
from keras.backend import backend as K
from cifar10_web import cifar10

def dim_ordering_fix(x):
    if K.image_dim_ordering() =='th':
        return x
    else:
        return np.transpose(x, (0, 2, 3, 1))


def dim_ordering_unfix(x):
    if K.image_dim_ordering() =='th':
        return x
    else:
        return np.transpose(x, (0, 3, 1, 2))

def dim_ordering_shape(input_shape):
    if K.image_dim_ordering() == 'th':
        return input_shape
    else:
        return input_shape[1], input_shape[2], input_shape[0]



def dim_ordering_input(input_shape, name):
    if K.image_dim_ordering() == 'th':
        return Input(input_shape, name= name)
    else:
        return Input((input_shape[1], input_shape[2], input_shape[0]), name=name)

def dim_ordering_reshape(k, w, **kwargs):
    if K.image_dim_ordering() =='th':
        return Reshape((k, w, w), **kwargs)
    else:
        return Reshape((w, w, k), **kwargs)


def fix_names(outputs, names):
    if not isinstance(outputs, names):
        outputs=[outputs]
    if not isinstance(names, list):
        names=[names]
    return [Activation('linear', name=name)(output) for output, name in zip(outputs, names)]



def leaky_relu(x):
    return K.relu(x, 0.2)

def gan_targets(n):
    '''
    Standard training target : [generator_fake, generator_real, discriminator_fake, discriminator_real] = [1, 0, 0, 1]
    :param n: number of samples
    :return: array of targets
    '''
    discriminator_fake = np.zeros((n, 1))
    generator_fake = np.ones((n, 1))
    generator_real = np.zeros((n, 1))
    discriminator_real = np.ones((n, 1))
    return [generator_fake, generator_real, discriminator_fake, discriminator_real]


'''
defining generator, Here each module in our pipeline
is simply passed sd input to the following modules
'''
def model_generator():
    nch=256 # no of channels
    g_input = Input(shape=[100])
    H = Dense(nch*14*14, init='glorot_normal')(g_input)
    H = BatchNormalization(mode=2)(H)
    H = Activation('relu')(H)
    H = dim_ordering_reshape(nch, 14)(H)
    H = UpSampling2D(size=(2, 2),)(H)

    H = Conv2D(int(nch/2), 3, 3, border_mode='same', init='glorot_uniform')(H)
    H = BatchNormalization(mode=2, axis=1)(H)
    H = Activation('relu')(H)

    H = Conv2D(int(nch/4),3 ,3 ,border_mode='same', init='glorot_uniform')(H)
    g_V = Activation('sigmoid')(H)
    return Model(g_input, g_V)

'''
defining discriminator, Here discriminator uses LeakyReLu
'''
def model_discriminator(input_shape=(1, 28, 28), dropout_rate=0.5):
    d_input = dim_ordering_input(input_shape, name="input_x")
    nch = 512
    H = Conv2D(int(nch/2), 5, 5, subsamples=(2,2), border_mode='same', activation='relu')(d_input)
    H = LeakyReLU(0.2)(H)
    H = Conv2D(nch, 5, 5, subsample=(2,2), border_mode='same', activation='relu')(H)
    H = LeakyReLU(0.2)(H)
    H = Dropout(dropout_rate)(H)
    H = Flatten()(H)
    H = Dense(int(nch/2))(H)
    H = LeakyReLU(0.2)(H)
    H = Dropout(dropout_rate)(H)
    d_V = Dense(1, activation='sigmoid')(H)
    return Model(d_input, d_V)


'''
defining two functions for loading and normalizing  MNIST data
'''
def mnist_process(x):
    x = x.astype(np.float32) / 255.0
    return x

def mnist_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    return mnist_process(x_train), mnist_process(x_test)

'''
Generator for creating new forgedd images at each epoch durinig training like the original
'''
def generator_samples(latent_dim, generator):
    def fun():
        zsamples = np.random.normal(size=(10*10, latent_dim))
        gen = dim_ordering_unfix(generator.predict(zsamples))
        return gen.reshape((10, 10, 28, 28))
    return fun


'''
Combining generator and discriminator to define a GAN
'''

if __name__=="__main__":
    latent_dim=100 # z in R^100
    input_shape = (1, 28, 28) # x in R^{28*28}
    generator = model_generator() # generator (z -> x)
    descriminator = model_discriminator(input_shape=input_shape) # descriminator (x->y)
    gan = simple_gan(generator, descriminator, normal_latent_sampling((latent_dim,)))

    #printinig summary of the models
    generator.summary()
    descriminator.summary()
    gan.summary()

    # build adversarial models
    model = AdversarialModel(base_model=gan,
                             player_params=[generator.trainable_weights, descriminator.trainable_weights],
                             player_names=["generator", "discrminator"])
    model.adversarial_compile(adversarial_optimizer=AdversarialOptimizerSimultaneous(),
                              player_optimizers=[Adam(1e-4, decay=1e-4), Adam(1e-3, decay=1e-4)],
                              loss='binary_crossentropy')
    #train model
    generator_cb = ImageGridCallback('output/gan_convolutional/epoch-{:03d}.png', generator_samples(latent_dim, generator))
    xtrain, xtest = mnist_data()
    xtrain = dim_ordering_fix(xtrain.reshape((-1, 1, 28, 28)))
    xtest = dim_ordering_fix(xtest.reshape((-1, 1, 28, 28)))
    y = gan_targets(xtest.shape[0])
    xtest = gan_targets(xtest.shape[0])
    history = model.fit(x=xtrain, y=y, validation_data=(xtest, y), callbacks=[generator_cb], nb_epoch=10, batch_size=32)
    df = pd.DataFrame(history.history)
    df.to_csv('output/gan_convolutional/history.csv')
    generator.save("output/gan_convolutional/generator.h5")
    descriminator.save("output/gan_convolutional/desrimiknator.h5")
