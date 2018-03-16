# **Behavioral Cloning**

Author: David Escolme
Date: March 2018

---

**Preamble**

This has been a very hard project for me, although it turns out that it was hard only because I suffered from a confirmation bias that drew me away from 2 very important facts which, once discovered, made the training of the model easy enough to complete for the first track. The 2 facts were:
* the drive.py program uses RGB and so i should convert my training data to that format
* the simulator eats up system resources and this affects model performance when testing...so much so, that after a disappointingly long time, once scaled back to 800x600 and simple graphics, the model i had trained 3 weeks earlier....worked.

This was more galling for the fact I had attended the Udacity / Parkopedia meet-up in London and talked about the simulator with Dave and Aaron.......

As well as the meetup, I am indebted to the blogs and forums of other Udacity students who have attempted this project, in particular:

+ https://github.com/georgesung/behavioral_cloning
+ https://medium.com/@dhanoopkarunakaran/147-lines-of-code-to-drive-vehicle-autonomously-around-the-track-in-simulator-c22bcf05f0bb
+ https://towardsdatascience.com/behavioural-cloning-applied-to-self-driving-car-on-a-simulated-track-5365e1082230

for their discussion on augmented image generation and balancing of zero v non-zero angle data.

I have run out of time to work on a more sophisticated model but will come back to this project to get onto testing out the second track.

**Behavioral Cloning Project**

The goals / steps of this project were the following:
* Use the simulator to collect data of good driving behavior
* Build a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/network.png "Model Visualization"
[image2]: ./examples/raw_histogram.png "Raw Data Steering Angles"
[image3]: ./examples/batch_histogram.png "Histogram after augmentation"
[image4]: ./examples/img_process.png "Image Processing Steps"
[image5]: ./examples/recovery.jpg "Recovery Image"

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* **model.h5** containing a trained convolution neural network
* README.md summarizing the results (this report)
* video.mp4 a video of the successful testing run

For this project to run, if cloned, the dependencies are (which might require a Udacity account):
+ https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip - training data generated by Udacity
+ https://github.com/udacity/self-driving-car-sim - the car simulator
+ A working Python 3 environment which can be cloned from: https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md

#### 2. Submission includes functional code

The submitted code/model can be tested on either track by running the following commands:

```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py code file is broken into a set of component functions:
* imports section for all dependencies used in the programme
* 2 helpers for flipping and warping images with angle adjustment
* 2 generators - one which simply uses all images and all images flipped, the other uses random selection of images and augmentation
* 3 identical keras convolutional models including one which operates over all data, one over the first generator, and finally one over the random generator

In addition, model.py was assessed against Python Coding standards at:
* https://www.python.org/dev/peps/pep-0008/
* http://pep8online.com/

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I would like to acknowledge both the primer videos from the Udacity course and also the following paper which have both heavily influenced my approach to creating a useful model for this project:
* http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf

In the end i used a simple 2 layer convolutional model with max pooling and no drop out with 3 fully connected layers. This, in the end, was due to the issues noted in the preamble and a lack of time to go back to the nvidia model i had struggled with.

The model suggested by nvidia consists of:
* a normalisation layer
* i added a cropping layer to remove the top and bottom of the images to remove sky and the car's hood as these parts of the image did not add value to the decision making for steering
* 3 5 x 5 convolutional layers with relu activation and max pooling
* 2 3 x 3 convolutional layers with relu activation and max pooling
* the output is then flattened and followed by 3 dense layers
* the model uses the adam optimiser and a loss function of mean squared error

My model was:
* a normalisation layer
* cropping layer
* 1st convolutional layer with 6 filters and a 5 x 5 pattern with pooling and relu
* 2nd convolutional layer with 6 filters and a 5 x 5 pattern with pooling and relu
* flattened
* 3 dense layers of 120, 84 and 1, the output layer
* an adam optimiser with default params was used
* a batch size of 32 with 5 epochs was used in the submission model

#### 2. Attempts to reduce overfitting in the model

In the model submission i did not need to use DropOut to find a working test lap but I used a number of augmentations over the training data:
* random selection of center, left, right image
* balancing the data away from high zero angle frequency by only taking in 30% of zero angle center image un-warped
* warping all other zero angle center images
* randomly flipping or warping each non-zero image by 180 degrees and adjusting the steering angle
* when using the left and right camera images from the simulator and adjusting the steering angle by 0.2
* using recovery laps on sections of the course the model struggled with

I retained 20% of the training data to be used as a validation data set.

#### 3. Model parameter tuning

I did not tune the adam learing rate but did experiment with different batch sizes and epochs.

The most significant tuning, though, came through using a balanced dataset, tuning the simulator resolution so that model testing was performant and by matching the color space used between model.py and drive.py.

#### 4. Appropriate training data

I used the Udacity provided training data as it seemed to be more 'stable' and was a readily available source. As mentioned above, to reduce the high zero steering angles, augmentation on that data sat was carried out.

I also recorded my own steeper angle recovery data around the section of the course after the bridge where my model was not performing well.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I followed on from the introductory videos and created a 2 layer convolutional model using the Udacity data augmented by flipping each image and using each camera image (adjusting the steering angle by 0.25 as that seemed, from reading the forums, to be the most often used adjustment). I used 20% of the data for validation.

This model drove well until after the bridge where it tracked straight off the course. I examined the histogram of the raw Udacity data center images:

![alt text][image2]

This showed a very large volume of zero angled data which would bias the model - even after my first attempts at augmentation, to drive straight. So, i recorded my own recovery images from the second half of the track to try to have the car respond better in that section.

This didn't work so well as the car continued to drive off the track at the same point. I then decided to try and balance and randomise the data set even more (following on from the blogs i had read) by making random selections of camera angles from the training data and then reducing the zero angle images even more by warping 70% of the zero angle center images chosen. For the remaining images, i randomly applied a flip or a warp or no augmentation.

The histogram of 9 batches of 32 are shown below:

![alt text][image3]

This model, using 5 epochs (where validation loss settled) and a batch size of 32 (the default), drove round the 1st track with no problem.

The model does not perform well on the second track and i believe i need more data to cope better with shadows and steeper curves and possibly a better model similar to the nvidia model described above.

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the layers outlined in section 1 above.

Here is a visualization of the architecture

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I used the Udacity data:

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to navigate between the roadside verges and not continue straight when no verge existed:

![alt text][image5]

To augment the data sat, I used a randomised approach to flipping and warping images and angles. This was inspired by my reading as acknowledged above and was expected to provide more balanced data (flipping) and more examples of verge-side avoidance (warping):

![alt text][image4]