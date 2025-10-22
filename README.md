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

## Separating data into features and labels

















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

