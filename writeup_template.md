# **Behavioral Cloning**

Author: David Escolme
Date: March 2018

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/network.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md summarizing the results (this report)

#### 2. Submission includes functional code

The submitted code/model can be tested on either track by running the following commands:

```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py code file is broken into a set of component functions:
* imports section for all dependencies used in the programme
* image file csv processor to create a training and validation generator
* image pre-processor to normalise and centre the image data
* a keras convolutional model
* a network visualiser

In addition, model.py was assessed against Python Coding standards at:
* https://www.python.org/dev/peps/pep-0008/
* http://pep8online.com/

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I would like to acknowledge both the primer videos from the Udacity course and also the following paper which have both heavily influenced my approach to creating a useful model for this project:
* Udacity Course...
* http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf

The model suggested by nvidia consists of:
* a normalisation layer
* i added a cropping layer to remove the top and bottom of the images to remove sky and the car's hood as these parts of the image did not add value to the decision making for steering
* 3 5 x 5 convolutional layers with relu activation and max pooling
* 2 3 x 3 convolutional layers with relu activation and max pooling
* the output is then flattened and followed by 3 dense layers
* the model uses the adam optimiser and a loss function of mean squared error

#### 2. Attempts to reduce overfitting in the model

I introduced dropout in the dense layers to try to reduce overfitting and also augmented the training data set by using 3 approaches:
* flipping each image by 180 degrees and negating the steering angle
* using the left and right camera images from the simulator and adjusting the steering angle by 0.2
* using recovery laps and images from the second test track rather than a single lap of the first track

I retained 20% of the training data to be used as a validation data set.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ...

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I modified the model so that ...

Then I ...

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the layers outlined in section 1 above.

Here is a visualization of the architecture

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
