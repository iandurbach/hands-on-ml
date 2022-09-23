#' ---
#' title: "08 - Neural Nets - MNIST Classification"
#' author: Emmanuel Dufourq 
#' output: html_document
#' ---
#' 
## ----setup, include=FALSE-----------------------------------------------------------------------
knitr::opts_chunk$set(echo = TRUE)

#' 
#' This is a RMarkdown version of the Google Colab notebook at https://colab.research.google.com/drive/1rLJUoP95aVw7n8wwh57w7CCCQKrXUStb. The data imports and normalisation code snippets were extracted from https://www.datacamp.com/community/tutorials/keras-r-deep-learning
#' 
#' Please note that before running this notebook you'll need to have keras and tensorflow installed and linked to R. If running this on RStudio Cloud this will have been done for you, and you just need to run the following lines
#' 
## -----------------------------------------------------------------------------------------------
library(keras)
use_virtualenv("myenv", required = TRUE)   # don't run this unless you installed keras in a virtualenv

#' 
#' ### Load dataset
#' 
## -----------------------------------------------------------------------------------------------
mnist <- dataset_mnist()
mnist

#' 
#' ### Load features
#' 
## -----------------------------------------------------------------------------------------------
x_train <- mnist$train$x
x_test <- mnist$test$x

#' 
#' ### Load targets
#' 
## -----------------------------------------------------------------------------------------------
y_train <- mnist$train$y
y_test <- mnist$test$y

y_train

#' 
#' ### Reshape the values to convert the image into a vector
#' 
#' Each image is originally 28x28x1 (the last x1 is due to the fact that the image is greyscale). So each 28x28 image can be converted into a vector of length 784 (28*28 = 784)
#' 
## -----------------------------------------------------------------------------------------------
c(nrow(x_test))

x_train <- array_reshape(x_train, c(nrow(x_train), 784))
x_test <- array_reshape(x_test, c(nrow(x_test), 784))

#' 
#' ### Rescale values
#' 
## -----------------------------------------------------------------------------------------------
dim(x_train)
dim(x_test)

x_train <- x_train / 255
x_test <- x_test / 255

#' 
#' ### Convert targets/labels to their one-hot encoded equivalent
#' 
## -----------------------------------------------------------------------------------------------
y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)
dim(y_train)

#' 
#' ### Create a model
#' 
## -----------------------------------------------------------------------------------------------
model <- keras_model_sequential()

#' 
#' We can add a number of layers and activation functions to the model. We start off by adding `model %>% ` then append each new layer (with it's activation function) on a new line.
#' 
#' The first layer has one extra thing which the others do not have. For the first layer we specify the input shape, which denotes the shape of the data input. In this case the shape of the input is just a vector of length 784, and thus we add `input_shape = c(784)`.
#' 
#' First we start off with a simple model with two layers then we will add more complexity to the model. 
#' 
#' In each case the line `model <- keras_model_sequential() ` is added otherwise we will keep on adding layers to the first instance of the variable `model` and create a massive model.
#' 
## -----------------------------------------------------------------------------------------------
model <- keras_model_sequential() 
model %>% 
  layer_dense(units = 256, activation = 'relu', input_shape = c(784)) %>% 
  layer_dense(units = 10, activation = 'softmax')

#' 
#' Next, we add an extra layer so the model can learn more complexities.
#' 
## -----------------------------------------------------------------------------------------------
model <- keras_model_sequential() 
model %>% 
  layer_dense(units = 256, activation = 'relu', input_shape = c(784)) %>% 
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dense(units = 10, activation = 'softmax')

#' 
#' Next, we add dropout to the first hidden layer
#' 
## -----------------------------------------------------------------------------------------------
model <- keras_model_sequential() 
model %>% 
  layer_dense(units = 256, activation = 'relu', input_shape = c(784)) %>% 
  layer_dropout(rate = 0.4) %>% 
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dense(units = 10, activation = 'softmax')

#' 
#' Finally, we add dropout to the next hidden layer.
#' 
## -----------------------------------------------------------------------------------------------
model <- keras_model_sequential() 
model %>% 
  layer_dense(units = 256, activation = 'relu', input_shape = c(784)) %>% 
  layer_dropout(rate = 0.4) %>% 
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 10, activation = 'softmax')

#' 
#' Print out a summary of the network architecture
#' 
## -----------------------------------------------------------------------------------------------
summary(model)

#' 
#' ### Compile the model
#' 
#' We need to provide extra information to train the model. We need to specify the loss function, the optimiser and what metric to display to the user.
#' 
## -----------------------------------------------------------------------------------------------
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

#' 
#' ### Training the neural network
#' 
## -----------------------------------------------------------------------------------------------
history <- model %>% fit(
  x_train, y_train, 
  epochs = 10, batch_size = 128, 
  validation_split = 0.2
)

#' 
#' ### Plot the training performance
#' 
#' When calling the fit function, Keras provides feedback of what happens to the loss during training. This is useful in determining if the model was over-fitting for example.
#' 
## -----------------------------------------------------------------------------------------------
plot(history)

#' 
#' ### Evaluate the performance on the test data
#' 
## -----------------------------------------------------------------------------------------------
model %>% evaluate(x_test, y_test)

#' 
#' ### Prediction
#' 
#' To predict we can either predict on an entire matrix or on a subset. In the case of a subset, you need to make sure that the correct dimensions are used as the network has certain input expectations. In this case, the model expects data in this format: [batches, 784]. So you can send any number of batches of data to the network.
#' 
## -----------------------------------------------------------------------------------------------
dim(x_test)

#' 
#' Here we want to predict on the first 10 test examples. But just using `x_test[0:10]` would result in the incorrect dimension. So we need to reshape the data.
#' 
## -----------------------------------------------------------------------------------------------
subset <- array_reshape(array(x_test[0:10,]), c(10, 784))

#' 
#' Here we check the dimensions.
#' 
## -----------------------------------------------------------------------------------------------
dim(subset)

#' 
#' Finally, we can predict on the 10 first examples.
#' 
## -----------------------------------------------------------------------------------------------
model %>% predict_classes(subset)

#' 
#' Here we predict on all of the `x_test `data.
#' 
## -----------------------------------------------------------------------------------------------
model %>% predict_classes(x_test)

