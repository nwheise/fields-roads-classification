# CNN for Image Classification
## Nicholas Heise
### May 2019

## Objective
<p>The objective of this task is to train a neural network to classify images of roads and fields.</p>

## Setup
Conda is used to manage the environment. The setup.sh shell script extracts the data and updates the conda env from the yaml file included. Clone the repo and then run
```
./setup.sh
source activate bil
python3 solution.py
```

## Approach
<p>Given the success of convolution neural networks (CNN) in image classification, I chose this approach for the image classification task. I train a CNN with two convolutional layers, each followed by a maxpooling, and then finally two fully connected layers. This is inspired by the AlexNet architecture, but is a much simplified version as we are classifying between two outputs rather than 1000 (as in AlexNet).</p>

## Discussion
### Initial Observations
<p>I noticed a few notable things about the data. The data provided is balanced between the two classes (45 images of fields and roads). Further, image dimensions between the images were not the same and so would require resizing or cropping. Finally, the dataset is relatively small for training a neural network (90 total images). To improve this, I perform transformations on the training set that allow the network to train on variations of the training data rather than the exact same images each epoch. This prevents overfitting and leads to a more robust, accurate model after training.</p>

### Transformations (Preprocessing)
<p>Before preprocessing, data is divided into an 80/20 train/test split. The transformations discussed below are only performed on the training data. Test data is only resized to (50, 50) before being passed to the network.</p>
<p>As mentioned above, some transformations of the data were necessary to produce a well-trained network. The primary concern was the small size of the data. I used the PyTorch transformation RandomResizedCrop with size=(50, 50) and scale=(0.5, 1), which crops the image at a scale of 0.5 to 1 of its original size and then resizes the crop to 50x50 so that it can be used as input for the neural network. Further, a horizontal flip of the image is performed at random (probability 0.5). These transformations occur when the data is loaded, so every epoch produces differently modified images. This remedies the small size of the data and prevents overfitting.</p>
<p>Other transformations were considered, such as rotations, vertical flipping, and applying a grayscale filter. These were not chosen, however, as they produce changes to the images that compromise their defining features. For example, a vertically flipped road would come down from the top of the image and would poorly train the neural network. I also chose not to apply grayscale filtering, since it is likely that color gives important clues to the classification (e.g. fields are likely to have more green, roads likely to have more gray).</p>

### CNN Architecture
<p>The CNN used here takes a (3, 50, 50) image (channels, height, width). The input is passed to a 2D convolutional layer with kernel size (5, 5), and produces a (6, 46, 46) output. The output is passed through a ReLU activation and a (2, 2) maxpooling layer. The second 2D convolutional layer then takes a (6, 23, 23) input and uses a (3, 3) kernel to produce a (12, 21, 21) output. This is passed through another ReLU and (2, 2) maxpooling layer. The first fully connected layer takes the flattened (12, 10, 10) data as input and produces a 128-neuron output before applying ReLU again. Finally, the second fully connected layer takes the input of size 128 to produce 2 output features. Passing these outputs through a sigmoid produces probability predictions for each of the 2 output classes.</p>
<p>Use of convolutional layers on images allows the neural network to extract higher-level information about the image, such as edges. Max pooling both reduces noise and reduce dimensionality, allowing for more robust and quick training. ReLU activation is used because it results in much faster training time as compared to logistic or tanh. Finally, sigmoid is used on the output to produce probabilities that sum to 1 over all possible classes.</p>

### Results
<p>Images from the test set and their predictions can be viewed in the "predictions" folder after running the solution script. Further, the average loss per epoch is saved as a plot as "loss.jpg". Experimentally, I found roughly 100 epochs was enough to sufficiently train the model. The neural network typically predicts correctly 15 to 17 of the 18 test images (83.3 to 94.4 %).</p>
 
