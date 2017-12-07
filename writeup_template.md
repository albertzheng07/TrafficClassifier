# **Traffic Sign Recognition**

## This is my solution writeup for the Traffic Sign Recognition Project

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

[image1]: ./examples/Data_Sets_Hist.jpg "Visualization"
[image2]: ./examples/Bicycles_crossing_visual.jpg "Bicycles_crossing_visual"
[image3]: ./examples/No_vehicles_visual.jpg "No_vehicles_visual"
[image4]: ./examples/Slippery_road_visual.jpg "Slippery_road_visual"
[image5]: ./examples/Speed_limit_(100km_h)_visual.jpg "Speed_limit_100km_h_visual"
[image6]: ./examples/Grayscale.jpg "Grayscaling"
[image7]: ./examples/Normalization.jpg "Normalization"
[image8]: ./test_writeup/Yield.jpg "Traffic Sign 1"
[image9]: ./test_writeup/30km.jpg "Traffic Sign 2"
[image10]: ./test_writeup/Stop.jpg "Traffic Sign 3"
[image11]: ./test_writeup/No_Entry.jpg "Traffic Sign 4"
[image12]: ./test_writeup/Priority_road.jpg "Traffic Sign 5"
[image13]: ./examples/Yield_softmax.png "Soft Max"
[image14]: ./examples/Speed_limit_(30km_h)_softmax.png "30 km Soft Max"
[image15]: ./examples/Stop_softmax.png "Stop Soft Max"
[image16]: ./examples/No_entry_softmax.png "No Entry Soft Max"
[image17]: ./examples/Priority_road_softmax.png "Priority Soft Max"

---
### Writeup / README

Here is a link to my [project code](https://github.com/albertzheng07/TrafficClassifier)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32 x 32 x 3.
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

The following is a summary of the training, test, and validation data.

![alt text][image1]

Here are a few of the example images for some of the sign types from the training data before pre-processing.

![alt text][image2]

![alt text][image3]

![alt text][image4]

![alt text][image5]

### Design and Test a Model Architecture

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image6]

Here is an example of an image before and after gray scale and normalization.

![alt text][image7]

I attempted to generate additional data by resizing the image to a larger image dimension such as 64x64. This technique did not improvement my training accuracy when I attempted this.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model is based on the LeNet Architecture which consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 Grayscale image   							|
| Convolution 1x1     	| 1x1 stride, same padding, outputs 28x28x36 |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 1x1     	| 1x1 stride, same padding, outputs 10x10x48 |     
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x48 				|
| Flatten				|     Output 1200    									|
|	Fully Connected					|		Output 200										|
| RELU					|												|
|	Fully Connected					|					Output 84							|
| RELU					|												|
|	Fully Connected					|					Output 43							|



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I trained my model by tuning the following parameters to identify the set of parameters that converged towards an accuracy > 93 %. I tried to first tweak the learning rate by decreasing by half for more accurate solutions, but it didn't benefit greatly. I also attempted to increase it by twice and the learning rate tended to jump back and forth between accuracies. The most beneficial parameter seemed to increase the epoch size and the number of iterations of forward/backward propagation of the network helped the accuracy considerably. I found over 30 epoch would result in diminishing returns.

| Parameter         		|     Value	        					|
|:---------------------:|:---------------------------------------------:|
| Learning Rate         | 0.001  							|
| EPOCH     	|  30	|
|	Batch Size				|		120									|


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.999
* validation set accuracy of 0.939
* test set accuracy of 0.928

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

The baseline architecture selected was the LeNet model. I believed it would perfectly setup to classify traffic signs because it already was designed to classify images that were sized as 32x32. The initial test without tuning any parameters showed that the model was 90% accurate already.

The main adjustment that I made to the architecture was the increase the size of the filters which was based on the intuition more filters would be able to assist in classifying the data. I was careful not to overload the layer with filters in the case of overfitting. I decided to limit the number of filters rather than include additional layers such as dropout layers. I found  

TODO


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image8] ![alt text][image9] ![alt text][image10]
![alt text][image11] ![alt text][image12]

The fifth image will probably the most difficult to classify since the image is cut off on the bottom which doesn't outline the entire diamond shape of the yield sign.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 30 km/h sign      		| 30 km/h sign   									|
| Priority Road     			| Bicycles Crossing 										|
| No Entry					| Turn Left Ahead											|
| Yield	      		| Yield				 				|
| Stop			| Stop      							|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 80%. This actually compares worse to the accuracy on the test set of 12610.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

For the first image, the model is somewhat sure that this is a 30 km/h speed sign, and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .40         			| 30 km/h sign   									|
| .40     				| 30 km/h sign 										|
| .17 					| 100 km/h sign											|
| .02	      			| 70 km/h sign 					 				|
| .01				    | 50 km/h sign      							|


TODO

For the second image, the model is relatively sure that this is a stop sign, and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .60         			| Stop sign   									|
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


The histogram results of the softmax probabilties are below and the code can be found at the end of the project code.

![alt text][image13]

![alt text][image14]

![alt text][image15]

![alt text][image16]

![alt text][image17]
