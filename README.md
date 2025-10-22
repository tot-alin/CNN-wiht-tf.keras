# Convolutional Neural Networks 2D wiht tf.keras

  The project aims to achieve a classification model using TensorFlow Keras. The model has been trained with the database containing 15 classes ['Bean', 'Bitter_Gourd', 'Bottle_Gourd', 'Brinjal', 'Broccoli', 'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Cucumber', 'Papaya', 'Potato', 'Pumpkin', 'Radish', 'Tomato'], 1500 training dataset, 3000 validatin dataset and 3000 test dataset, taken from: 'https://www.kaggle.com/api/v1/datasets/download/misrakahmed/vegetable-image-dataset'

<p align="center">
<img width="500" height="300" alt="Screenshot from 2025-10-22 20-25-17" src="https://github.com/user-attachments/assets/6ff5883c-9a6b-445e-a131-735da5c9f59f" />
</p>

The project contains:
* [CNN.pdf](https://github.com/user-attachments/files/21350468/CNN.pdf)
* Describes the methods and functions used
* Functionality diagrams
* Diagrams of results obtained
* Project code
<br />
<br />

## Data collection method
A straightforward way to create the dataset is to use the image_dataset_from_directory method, which is found in the Keras TensorFlow library tf.keras.preprocessing.image_dataset_from_directory. This method creates a tensor of the form [ [ [ [image][label] ] ]  [ [image][label] ] [ [ [image][label] ] ]  .....], and the required structure has the form.
<p align="center">
<img width="507" height="252" alt="Screenshot from 2025-10-22 20-50-38" src="https://github.com/user-attachments/assets/f100eb2a-a53b-43b6-b4ae-e97d6d0c487c" />
</p>
<br />
<br />

## Data Normalization
An important step in data pre-processing is feature normalization, which is a method of scaling the features before entering them into a model. Properly, normalization means scaling the data to be analyzed into a specific range [0,0 ... 1]. The purpose of this method is to facilitate the model operation and to find a common denominator in terms of the amplitude of the different datasets that are composited. The normalization equation is:

<img width="115" height="44" alt="image" src="https://github.com/user-attachments/assets/8060b909-6a1a-41dd-a24b-166164f52d69" />
<br />
<br />

## Sequential()
The Sequential class in Keras allows rapid prototyping of machine learning models by sequentially superimposing layers. Sequential models work by layer stacking, i.e., each layer takes the output of the previous layer as input. To create a sequential model in Keras, you can create a list of constructor statements or add layers incrementally using the add() method.
<br />
<br />

## Conv2D()
tf.keras.layers.Conv2D from the TensorFlow Keras libraries is a convolutional element with which conventional neural networks are built. The explanation of these network models is explained in a previous project https://github.com/tot-alin/CNN-with-tf.nn

Conv2D requires some minimal settings Conv2D(filters=nr.maps out, kernel_size=(height,width), strides=moving on (height,width), padding= same or valid, activation=chosen activation function )
<br />
<br />

## Pool()
Pooling is a method of extracting important features from a feature matrix (feature map) by reducing its size. The description of the functionality can be found at https://github.com/tot-alin/CNN-with-tf.nn/blob/main/README.md

The settings for MaxPooling2D(pool_size=(height, width), strides=moving on (height, width), padding=same or valid), used in this project, extract the maximum values from the moving window.
<br />
<br />

## Flatten()
This class transforms an m×n-dimensional matrix or an i×j×k×⋯ tensor into a vector (one-dimensional matrix) 1×(m*n) or 1×(i*j*k*⋯). The role of this transform in this project is to bridge the CNN layer and the Dense layer.
<br />
<br />

## Dense()
tf.keras.layers.layers.Dense() is a fully connected layer where each neuron in the layer is connected to the next layer, more fully described in https://github.com/tot-alin/Multi-Layer-Perceptron-with-NumPy. Setarea clasei dense (unități = numărul de neuroni din strat, activare = funcția de activare aleasă).
<br />
<br />

## Activation function
The activation functions in a neural network have the role of changing the linear character of the output produced by layers of perceptrons, CNN, etc. In this project, there are two types of networks: ReLU and SoftMax

SoftMax is the activation function that converts a raw feature vector from the neural network into a vector expressing the probability corresponding to the input. The computational relation <img width="224" height="67" alt="image" src="https://github.com/user-attachments/assets/050ec122-a469-4113-aa81-e2ee4ddb9358" /> , <img width="12" height="21" alt="image" src="https://github.com/user-attachments/assets/3feae99d-b8a8-41a0-97ce-ef2e2de94480" /> - the output of the previous layer in the network, <img width="10" height="19" alt="image" src="https://github.com/user-attachments/assets/aa0dfb2c-0fa6-4768-a6d0-c1c12e76010f" /> - number of classes,<img width="16" height="21" alt="image" src="https://github.com/user-attachments/assets/e828161e-6132-491a-aa32-aaf0df25510e" /> - represents the exponential of the Zi, <img width="53" height="44" alt="image" src="https://github.com/user-attachments/assets/e324ef93-be58-4a63-9844-6f360a00a0f4" /> - the sum of exponentials across all classes.















## Bibliography:

https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image_dataset_from_directory

https://keras.io/api/data_loading/image/

https://machinelearningmastery.com/how-to-load-large-datasets-from-directories-for-deep-learning-with-keras/

https://www.tensorflow.org/guide/data

https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool2D 

https://www.tensorflow.org/api_docs/python/tf/keras/layers/Flatten

https://en.wikipedia.org/wiki/Activation_function

https://victorzhou.com/blog/softmax/

https://www.datacamp.com/tutorial/loss-function-in-machine-learning

https://www.geeksforgeeks.org/deep-learning/categorical-cross-entropy-in-multi-class-classification/

https://www.geeksforgeeks.org/machine-learning/metrics-for-machine-learning-model/

https://www.geeksforgeeks.org/javascript/tensorflow-js-tf-layersmodel-class-compile-method/

https://kambale.dev/build-compile-and-fit-models-with-tensorflow

https://www.geeksforgeeks.org/deep-learning/model-fit-in-tensorflow/

https://www.geeksforgeeks.org/deep-learning/model-evaluate-in-tensorflow/

https://www.v7labs.com/blog/confusion-matrix-guide

https://www.geeksforgeeks.org/machine-learning/confusion-matrix-machine-learning/

