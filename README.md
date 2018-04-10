# RealTime-DigitRecognition
RealTime DigitRecognition using keras/SVC and pygame.

## Overview
Recently Deep Convolutional Neural Networks (CNNs) becomes one of the most appealing approaches and has been a crucial factor in the variety of recent success and challenging machine learning applications such as object detection, and face recognition. Therefore, CNNs is considered our main model for our challenging tasks of image classification. Specifically, it is used for is one of high research and business transactions. Handwriting digit recognition application is used in different tasks of our real-life time purposes. Precisely, it is used in vehicle number plate detection, banks for reading checks, post offices for sorting letter, and many other related tasks.<br><br>
	![sample images](assets/out.png "applications ")
	
## Description
This is a RealTime-DigitRecognition application which can predict output corresponding to handwritten images. I used **SVC**(support vector classifier) and sequential model of Keras for creating this predictive model. I trained SVC for 8X8 MNIST dataset, but the accuracy of this model is not good when I run this model on my handwritten images(600X600). It is due to resizing images from 600X600 to 8X8.It is important to get good results so I created a sequential model in keras and traied it on 28X28 MNIST dataset. Now it gives very good result on handwritten digits. <br>  

The interface is created by using **Pygame**. The image preprocessing is the most important in this project which I have done by using **Scipy** and **OpenCV**.

## Dataset
MNIST is a widely used dataset for the hand-written digit classification task. It consists of 70,000 labelled 28x28 pixel grayscale images of hand-written digits. The dataset is split into 60,000 training images and 10,000 test images. There are 10 classes (one for each of the 10 digits). The task at hand is to train a model using the 60,000 training images and subsequently test its classification accuracy on the 10,000 test images.<br>

## Sample Images:
These are some sample images of the handwritten character from mnist dataset. <br><br>
	![sample images](assets/sample_images.png "images in mnist dataset")<br><br>

## Dependencies
This is the list of dependencies for running this application.
 * **Skleran**
 * **Keras**
 * **tensorflow/theano**
 * **Opencv**
 * **Pygame**
 * **Pandas**
 * **Numpy**
 * **Secipy**
 * **Matplotlib**
 
  
## How to use
1. Download or clone this repository.
2. Extract to some location
3. First, run **```app.py```** from **```RealTime-DigitRecognition```** folder.<br>
    Now, Pygame window will open. It will look like this.<br><br>
   	![Pygame window](assets/pygame_window.png "Pygame window" )<br><br>

4. Draw the digits on **left** side of the window and output will appear on **right** side of the window. 
5. Mouse handling:<br>
    The **right** button is for resetting screen.<br>
    The **left** button is for drawing.

## Choosing model
Edit in ```app.py``` <br>
**SVC of sklearn:** comment ```KERARS``` and uncomment ```SVC```  <br>
**Sequential model:** comment ```SVC``` and uncomment ```KERARS```<br>
<br>
![Pygame window](assets/choosing_model.png "Choosing model" )<br><br>

## Multi digit reconition
I am developing an efficient model for detection multiple digits on a single frame like number plate, phone number, cheque number etc. <br>
Here are some results:<br><br>
![Pygame window](assets/digits.png "multi digits" )

## Demo
![Pygame window](assets/demo.gif "Demo gif" )<br><br>




### Please commit for any changes or bugs :)

