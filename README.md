# RealTime-DigitRecognition
RealTime DigitRecognition using keras/SVC and pygame.

## Description
This is a RealTime-DigitRecognition application which can predict output corresponding to handwritten images. I used **SVC**(support vector classifier) and sequential model of Keras for creating this predictive model. I trained SVC for 8X8 MNIST dataset, but the accuracy of this model is not good when I run this model on my handwritten images(600X600). It is due to resizing images from 600X600 to 8X8.It is important to get good results so I created a sequential model in keras and traied it on 28X28 MNIST dataset. Now it gives very good result on handwritten digits. <br>  

The interface is created by using **Pygame**. The image preprocessing is the most important in this project which I have done by using **Scipy** and **OpenCV**.

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


## Demo
![Pygame window](assets/demo.gif "Demo gif" )<br><br>




### Please commit for any changes or bugs :)

