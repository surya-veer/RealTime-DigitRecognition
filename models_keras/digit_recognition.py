###############################################################
#########          By: suryaveer @IIT Indore         ##########
#########     GITHUB: https://github.com/surya-veer  ##########
###############################################################

import warnings
warnings.filterwarnings('ignore')

import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
from keras.models import load_model
from scipy import misc
import cv2
import matplotlib.pyplot as plt


IS_TRAIN = False

if(IS_TRAIN):
    seed = 7
    numpy.random.seed(seed)

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # flatten 28*28 images to a 784 vector for each image
    num_pixels = X_train.shape[1] * X_train.shape[2]
    X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')

    # normalize inputs from 0-255 to 0-1
    X_train = X_train / 255
    X_test = X_test / 255

    # one hot encode outputs
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    num_classes = y_test.shape[1]

    # define baseline model
    def Model():
        # create model
        model = Sequential()
        model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
        model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model


    # build the model
    model = Model()
    # Fit the model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=200, verbose=2)
    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("SCORE: %.2f%%" % scores[1]*100)


model = load_model('models_keras/model.h5')

def check():
    #imverting the image
    img = misc.imread('assets/out.png')
    img = (255 - cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))/255
    
    #resizing image into 8X8 matrix
    img = misc.imresize(img,(28,28))
    img = img.astype('float32')
    
    #changing byte scale from 16 to 0 according to imput data
    img = misc.bytescale(img,high=16,low=0)
    
    
    flat_img = img.reshape(1,784)
    result = model.predict_classes(flat_img)
    return result

if __name__ == '__main__':
    """This model is for predicting handwritten digits. Run app.py for testing this."""
