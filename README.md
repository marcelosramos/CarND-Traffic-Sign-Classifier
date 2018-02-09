
##### **Traffic Sign Recognition** 

## Writeup Report

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)
[barchart]: ./report_images/summary_plot.png "Bar Chart"
[samples]: ./report_images/samples.png "Samples"
[preprocessed]: ./report_images/transformed_samples.png "Samples"
[google_images]: ./report_images/google_images.png "Google Images"
[google_images_32x32]: ./report_images/google_images_32x32.png "Google Images"
[predictions]: ./report_images/predictions.png "Predictions"
[hit]: report_images/hit.png "Hit"
[conv1_hit]: report_images/conv1_hit.png "Conv 1 Hit"
[conv2_hit]: report_images/conv2_hit.png "Conv 2 Hit"
[miss]: report_images/miss.png "Miss"
[conv1_miss]: report_images/conv1_miss.png "Conv 1 Miss"
[conv2_miss]: report_images/conv2_miss.png "Conv 2 Miss"


## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---

### Data Set Summary & Exploration

#### 1. Data Set Summary

I used the numpy library and native Python functions/classes to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34,799
* The size of the validation set is 4,410
* The size of test set is 12,630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Exploratory visualization of the dataset.

Here is a bar chart showing the sign count for each dataset. As we can see, the sets are unbalanced.



![alt text][barchart]


In the image below are randomly selected samples of each sign type on the training set.

![alt text][samples]


### Design and Test a Model Architecture

Designing a convolutional neural network is mainly empirical. So I tried several preprocessing techniques, and different architectures as described here.

#### 1. Preprocessing

First I tried converting the images to different colorspaces. On the final chosen arquitecture I got 98.54% for RGB (original images), 98.88% for simple grayscale conversion, 98.56% for YUV using all channels and 98.58% using only the Y channel. I got very similar results for each colorspace, with the best results being for grayscale. 

Actualy, the results look all the same, and because of the randomicity of the augmented training set, ideally, I should have ran a couple times with each configuration and taken the average, but due to time constraints, I didn't take this approach. Furthermore, for each colorspace, there could be a different ideal tunning of preprocessing and model architecture, specially on 1 versus 3 channels, but again, because of limitations, I did all the tunning with grayscale (the simplest, and most performing), and then compared it the other colorspaces.

As a second step I normalized the images so each pixel ranges from -1.0 to 1.0.

As suggested by Yann Lecun in [this paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf), in order to make the model more robust I augmented the training set with randomly tranformed samples of the original images. I have done many trials arbitrarily tweaking each of the transformation parameters, getting the best results with a combination of rotation between -20 and 20 degrees, translation between -3 and 3 pixels and scaling to between 26 and 36 pixels. Also, I noticed thet most of the mispredictions on the validation set, were images with very high brightness levels, so I added one more preprocessing step, in which I randomly increase or reduce the image's brightness.

Here are the same samples shown above after the preprocessing:

![alt text][preprocessed]

I have also tried applying all kinds of noise type (e.g. gaussian, salt&pepper, ...), The accuracy was lower with the random noise than without it, regardless of the noise type, probably, because of the low resolution of the images, the other tranformations were already making enough noise on it.

Finally, the training set was very unbalanced, therefore the features of the most present signs could stand out over the least seen ones. Additionally I treid increasing more and more the size of the training set, but after a threshold, the final accuracy started to fall. Hence I ended up with 5,000 images of each sign, making a total of 215,000 images on the training set.


#### 2. Model architecture.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x108 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x108                |
| Dropout.              | 0.5 keep probability on training              |
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x200 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x200                  |
| Dropout.              | 0.5 keep probability on training              |
| Fully connected		| 5000 input, 200 output.   					|
| RELU					|												|
| Dropout.              | 0.5 keep probability on training              |
| Fully connected		| 200 input, 43 output.   	        			|



#### 3. Model Training

To train the model, I used an Adam optimizer over the softmax cross entropy with a learning rate of 0.0001, and a batch size of 128 through 150 epochs.


#### 4. Solution Approach

My first approach was to use the Lenet5 implementation shown in the classroom adjusted to a 3 color channels input image, only normalizing the image on the preprocessing step, 10 epochs and 0.001 learning rate for model training. With this configuration, the validation accuraccy reaches a maximum of 0.93, then decreaces after epoch 6, ending up with a test accuracy of 0.90. This result suggests a possible overfitting.

In order to handle the overfitting I added a dropout layer after each fully connected layer reaching 0.956 accuracy for validation and 0.929 for testing. Then I tried several configurations varying the number and size of layers, and adding dropout layers after every RELU (including the convolution layers). On the table below I show the architecture at that stage:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x18 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x18                 |
| Dropout               | 0.5 keep probability on training              |
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 48     	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 48                       |
| Dropout               | 0.5 keep probability on training              |
| Fully connected		| 1200 input, 400 output.   					|
| RELU					|												|
| Dropout               | 0.5 keep probability on training              |
| Fully connected		| 400 input, 120 output.   			    		|
| RELU					|												|
| Dropout               | 0.5 keep probability on training              |
| Fully connected		| 120 input, 84 output.   					    |
| RELU					|												|
| Dropout               | 0.5 keep probability on training              |
| Fully connected		| 84 input, 43 output.   	        			|

Small changes on the keep probability didn't seem to make much difference on the results. Arbitrarily changing the architecture from that point, I coun't improve the accuracy.

Then I included the preprocessing as described above and after many architectures tried, the best accuracy I could get was 0.976 (validation) and 0.969 (testing), with the confuguration below.

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x32                 |
| Dropout               | 0.5 keep probability on training              |
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 64     	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 64                       |
| Dropout               | 0.5 keep probability on training              |
| Fully connected		| 1600 input, 200 output.   				    |
| RELU					|												|
| Dropout               | 0.5 keep probability on training              |
| Fully connected		| 200 input, 43 output.   	        			|

From here, small changes on those parameters wouldn't improve the results. At this point, I noticed that too many layers would make it harder to train the model, probably due to atenuation of the derivatives.

Finally, I gave up on arbitrarily changing that model, and tried to follow Yann Lecun's approach as in the paper cited on the assignment. After trying some of the configuration mentioned on the paper, I got my best results with the final model described above. Interestingly my best results came from a single-scale model, not the multi-scale model as described on the paper.

For the final testes I changed the learning rate to 0.0001 and the number of epochs to 150.

I would like to improve those results, maybe trying other preprocessing techniques, using loss or batch normalization, applying other activation functions, etc. The options could be infinit, and those were the best results I've got in the given time and with my available knowledge and resources:

* training set accuracy of 100.00%
* validation set accuracy of 98.89% 
* test set accuracy of 98.88%


 

### Test a Model on New Images

#### 1. Acquiring New Images

I took 12 German traffic sign images from the web:

![alt text][google_images]

In order to use these images here, I manually croped and resized each of the images to 32x32:

![alt text][google_images_32x32]

Note that the images #6 and #13, originaly came from the same image, making it a total of 13 traffic signs to classify.

* Images # 3, 4, 6, 9, 10 and 12 might be difficult to classify because of too much information on the background.
* Images #5 might be difficult to classify because the sign is too small in the image and there are not much contrast/color variations.
* Images #6 might be difficult to classify because of the angle of the point of view.
* Images #10 might be difficult to classify because there seems to be a sticker on the lower part of the sign.
* Images #13 might be difficult to classify because of the angle of the point of view, and because there is a shadow.
* The other images shouldn't be difficult to classify

#### 2. Performance on New Images



Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

![alt text][predictions]

The model was able to correctly guess 13 of the 13 traffic signs, which gives an accuracy of 100.00%. This is better than the accuracy on the test set of 98.88%. It is important to note that the testing set is much more representative, with traffic signs in many real life situations of angles distance an ilumination.

#### 3. Model Certainty - Softmax Probabilities

Image #0 (Bumpy road):

	100.00% chance of Bumpy road
	0.00% chance of Slippery road
	0.00% chance of Bicycles crossing
	0.00% chance of Road work
	0.00% chance of Traffic signals
    
Image #1 (Slippery road):

	100.00% chance of Slippery road
	0.00% chance of Bicycles crossing
	0.00% chance of Beware of ice/snow
	0.00% chance of Wild animals crossing
	0.00% chance of Double curve
    
Image #2 (General caution):

	100.00% chance of General caution
	0.00% chance of Traffic signals
	0.00% chance of Pedestrians
	0.00% chance of Road narrows on the right
	0.00% chance of Right-of-way at the next intersection
    
Image #3 (Road work):

	100.00% chance of Road work
	0.00% chance of Bumpy road
	0.00% chance of Dangerous curve to the right
	0.00% chance of Beware of ice/snow
	0.00% chance of Slippery road
    
Image #4 (End of all speed and passing limits):

	48.69% chance of End of all speed and passing limits
	35.58% chance of Traffic signals
	11.21% chance of General caution
	2.56% chance of Bicycles crossing
	1.22% chance of Priority road
    
Image #5 (Yield):

	100.00% chance of Yield
	0.00% chance of Priority road
	0.00% chance of No vehicles
	0.00% chance of End of no passing
	0.00% chance of Ahead only
    
Image #6 (Right-of-way at the next intersection):

	100.00% chance of Right-of-way at the next intersection
	0.00% chance of Beware of ice/snow
	0.00% chance of Pedestrians
	0.00% chance of Road work
	0.00% chance of Children crossing
    
Image #7 (Speed limit (30km/h)):

	100.00% chance of Speed limit (30km/h)
	0.00% chance of Speed limit (20km/h)
	0.00% chance of Speed limit (50km/h)
	0.00% chance of Speed limit (70km/h)
	0.00% chance of Speed limit (80km/h)
    
Image #8 (Double curve):

	100.00% chance of Double curve
	0.00% chance of Right-of-way at the next intersection
	0.00% chance of Wild animals crossing
	0.00% chance of Road work
	0.00% chance of Dangerous curve to the left
    
Image #9 (Stop):

	100.00% chance of Stop
	0.00% chance of Keep right
	0.00% chance of Speed limit (60km/h)
	0.00% chance of Turn left ahead
	0.00% chance of Priority road
    
Image #10 (Road work):

	100.00% chance of Road work
	0.00% chance of Beware of ice/snow
	0.00% chance of Slippery road
	0.00% chance of Bicycles crossing
	0.00% chance of Bumpy road
    
Image #11 (Speed limit (30km/h)):

	100.00% chance of Speed limit (30km/h)
	0.00% chance of Speed limit (20km/h)
	0.00% chance of Speed limit (50km/h)
	0.00% chance of Speed limit (70km/h)
	0.00% chance of Speed limit (80km/h)
    
Image #12 (Priority road):

	100.00% chance of Priority road
	0.00% chance of Stop
	0.00% chance of Yield
	0.00% chance of Ahead only
	0.00% chance of End of all speed and passing limits

The model predicted with very high certainty (100%) all the  traffic signs, except for one. For image #4, the model was only 48.69% sure it was a 'End of all speed and passing limits' sign. So, even though this sign was correctly predicted, it was not a very strong prection.

---

### (Optional) Visualizing Layers of the Neural Network

I created the visualizations for eache convolutional layer of two images, one correctly predicted and one mispredicted:

##### Correct predicted:

![alt text][hit] 

First convolutional layer:

![alt text][conv1_hit] 

Second convolutional layer:

![alt text][conv2_hit] 

##### Mispredicted:

![alt text][miss] 

First convolutional layer:

![alt text][conv1_miss] 

Second convolutional layer:

![alt text][conv2_miss] 

For the correctly classified image, it is possible to identify some recognizable features on the first convolutional layer's feature maps, the sign's boundary outline can be seen in almost all feature maps, and the inner symbol of the sign is also marked in most of the filters, whilst for the misclassified image it is quite hard to identify any meaningful feature, the sign's symbol can be some how indentified in only a few feature maps.

For the second convolutional layer feature maps, I couldn't visualize any useful information for my human eyes on both images.




```python

```
