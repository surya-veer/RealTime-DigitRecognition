###############################################################
# ########          By: suryaveer @IIT Indore         ####### #
# ########     GITHUB: https://github.com/surya-veer  ####### #
###############################################################


# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
from sklearn import datasets  # preprocessing
from sklearn.svm import SVC
from sklearn.externals import joblib
from scipy import misc
import cv2
import warnings
warnings.filterwarnings('ignore')

file_name = 'models_svc/clf.pkl'

IS_TRIAN = False

print("Using SVC of sklearn\n")

if (IS_TRIAN is True):
    digits = datasets.load_digits()
    # print(digits.images.dtype) #for getting data type of image which is float64
    X = digits.data
    Y = digits.target
    clf = SVC(gamma=.001, C=10)
    clf.fit(X, Y)
    joblib.dump(clf, file_name)

clf = joblib.load(file_name)


def check():
    img = misc.imread('assets/out.png')
    # imverting the image
    img = 255 - cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # resizing image into 8X8 matrix
    img = misc.imresize(img, (8, 8))
    img = img.astype('float64')

    # changing byte scale from 16 to 0 according to imput data
    img = misc.bytescale(img, high=16, low=0)

    # uncomment if you want to see the image
    # plt.imshow(img, cmap=plt.cm.gray_r, interpolation='nearest')

    flat_img = img.reshape(1, 64)
    result = clf.predict(flat_img)
    return result


if __name__ == '__main__':
    """This model is for predicting handwritten digits. Run app.py for testing this."""
