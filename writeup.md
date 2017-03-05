## Vehicle Detection
### Self-Driving Cars Nanodegree @Udacity

###Credits

- Udacity: Self-Driving Car Nano Degree
- OpenCV: http://opencv-python-tutroals.readthedocs.io/en/latest/
- scikit-learn: http://scikit-learn.org/stable/

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Apply a color transform and append binned color features, as well as histograms of color, to HOG feature vector.
* Note: for those first two steps normalize features and randomize a selection for training and testing.
* Implement a sliding-window technique and use trained classifier to search for vehicles in images.
* Run the pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.png
[image3]: ./examples/sliding_windows.png
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

---

####1. HOG features extraction from training images.

The code for this step is contained in the code cell #3 of the IPython notebook starting at line #10

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

####2. Trained a classifier using your selected HOG features and color features...

I tried various combinations of parameters to maximize accuracy of SVM. Linear SVM (scikit-learn) used over HOG and Color features -Jupyter Notebook CodeCell #6 & #7.

  Final choice of HOG parameters:

  ` HOG orientations = 9  
    HOG pixels per cell = 8
    HOG cells per block = 2
    HOG channel = "ALL"`

  Final choice of histogram for colorspace:

  `Spatial binning dimensions = (32, 32)
   Number of histogram bins = 32`

###Sliding Window Search

####1. Sliding window search.  Scales to search and how much to overlap windows:

Jupyter Notebook code cell #9, line #20
I decided to search window positions across bottom portion of image(by specifying y-start) at multiple random scales and came up with:

![alt text][image3]

####2. Examples of test images.  Optimization of the performance of classifier:

Ultimately I searched on five scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

####1. Link to final video output:
Here's a [link to my video result](./project_video__marked_result.mp4)

####2. Implemented filter for false positives and some method for combining overlapping bounding boxes:

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Problems / issues you faced in your implementation of this project.  
- Even with high SVM accuracy on test data - the model was not capturing the white colored car well (low number of detections, causing fewer additions into heat map)
- Since my final implementation is relatively slow - it took some time to test multiple configurations of the model.

####2. Where will your pipeline likely fail?  
- The pipeline could fail under different camera calibrations (relative vehicle size, position in image)
- The pipeline is currently slow (nowhere close to real time) - hence it will fail when a real-time implementation is required.
- The pipeline is not robust enough to detect each car only once. Hence cannot be used to count the number of cars.

####3. What could you do to make it more robust?
- Leverage the GPU via GPU enabled functions from Tensorflow.
- Include camera calibration and image normalization scripts
- Add centroid tracking to cars, that will help me reduce my search area for consecutive frames
- Add a register counting the number of cars per frame, and leverage the information from previous frames to reduce double counting.
- Also, I have implemented the same using "YOLO", a neural network based approach.
