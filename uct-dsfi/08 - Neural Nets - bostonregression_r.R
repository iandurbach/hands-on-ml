#' ---
#' title: "08 - Neural Nets - Boston Regression"
#' author: Emmanuel Dufourq 
#' output: html_document
#' ---
#' 
## ----setup, include=FALSE-----------------------------------------------------------------------
knitr::opts_chunk$set(echo = TRUE)

#' 
#' This is a RMarkdown version of the Google Colab notebook at https://colab.research.google.com/drive/1G1p7FLp8dvaybGG8Y7Wae0EYxuyQ05qd. The data imports and normalisation code snippets were extracted from https://www.datacamp.com/community/tutorials/keras-r-deep-learning
#' 
#' Please note that before running this notebook you'll need to have keras and tensorflow installed and linked to R. If running this on RStudio Cloud this will have been done for you, and you just need to run the following lines
#' 
## -----------------------------------------------------------------------------------------------
library(keras)
use_virtualenv("myenv", required = TRUE)   # don't run this unless you installed keras in a virtualenv

#' 
#' ### Load the boston housing dataset
#' 
## -----------------------------------------------------------------------------------------------
boston_housing <- dataset_boston_housing()

#' 
#' ### Obtain the fetures and labels from the data
#' 
## -----------------------------------------------------------------------------------------------
c(train_data, train_labels) %<-% boston_housing$train
c(test_data, test_labels) %<-% boston_housing$test

#' 
#' ### Check the dimensions for the features
#' 
## -----------------------------------------------------------------------------------------------
dim(train_data)

#' 
#' """## Check the dimensions for the labels"""
#' 
## -----------------------------------------------------------------------------------------------
dim(test_data)
dim(test_labels)

#' 
#' ### Inspect the features to determine if scaling is needed
#' 
## -----------------------------------------------------------------------------------------------
summary(train_data)

#' 
#' ### Scale the features
#' 
#' It's often a good idea to scale data so all variables lie on nearly the same scale. If variables have very different ranges, a one-unit change in one variable might represent a huge change (say on a probability scale), while the same one-unit change might represent a tiny change (say if units are metres and distances are very large). One way to scale variables is to divide each variable by its mean and divide by its standard deviation. 
#' 
#' Note that means and standard deviation's should always come from the TRAINING set, even if scaling the validation and test sets. Otherwise we are using information in the test set in our model building, which we shouldn't do. Most of the time if observations have been randomly allocated to training and test sets it won't make much difference (because the variable means and standard deviations will be similar in training and test sets), but we should do the right thing.
#' 
#' The `scale` function stores means and standard deviations as attributes of the scaled object, so we can extract these and use them to scale variables in the validation and test datasets.
#' 
## -----------------------------------------------------------------------------------------------
train_data <- scale(train_data)
apply(train_data, 2, mean) # mean should be 0
apply(train_data, 2, sd) # sd should be 1
attributes(train_data) # previous means and sds used to scale stored here

#' 
## -----------------------------------------------------------------------------------------------

test_data <- scale(test_data, center = attr(train_data, "scaled:center"), 
                scale = attr(train_data, "scaled:scale"))

#' 
#' ### Define the model
#' 
## -----------------------------------------------------------------------------------------------
dim(train_data)[2]

#' 
## -----------------------------------------------------------------------------------------------
model <- keras_model_sequential() %>%
    layer_dense(units = 64, activation = "relu",input_shape = dim(train_data)[2]) %>%
    layer_dense(units = 64, activation = "relu") %>%
    layer_dense(units = 1)

#' 
#' ### Compile the model
#' 
## -----------------------------------------------------------------------------------------------
model %>% compile(
    loss = "mse",
    optimizer = optimizer_rmsprop(),
    metrics = list("mean_absolute_error")
  )

#' 
#' ### Training the neural network
#' 
## -----------------------------------------------------------------------------------------------
history <- model %>% fit(
  train_data, train_labels, 
  epochs = 50, batch_size = 5, 
  validation_split = 0.2, shuffle = TRUE
)

#' 
#' ### Plot the training performance
#' 
#' When calling the fit function, Keras provides feedback of what happens to the loss during training. This is useful in determining if the model was over-fitting for example.
#' 
## -----------------------------------------------------------------------------------------------
plot(history)

#' 
#' ### Predict on the test data
#' 
## -----------------------------------------------------------------------------------------------
test_predictions <- model %>% predict(test_data)
test_predictions[ , 1]

#' 
#' #### Print first 15 model predictions
#' 
## -----------------------------------------------------------------------------------------------
test_predictions[ , 1][0:15]

#' 
#' #### Print first 15 correct test values
#' 
## -----------------------------------------------------------------------------------------------
test_labels[0:15]

#' 
#' ### Train the model on 100 epochs
#' 
#' Here we introduce something new, this allows us to know when an epoch has completed. In this case, the epoch number is printed on each even numbered epoch.
#' 
## -----------------------------------------------------------------------------------------------
print_dot_callback <- callback_lambda(
  on_epoch_end = function(epoch, logs) {
    if (epoch %% 2 == 0) cat(epoch, '\n')
  }
)

#' 
#' Note that 100 epochs may not be performed since we have added early stopping
#' 
## -----------------------------------------------------------------------------------------------
history <- model %>% fit(
  train_data,
  train_labels,
  epochs = 100,
  validation_split = 0.2,
  verbose = 0,
  callbacks = list(print_dot_callback))

#' 
#' ### Plot the training performance
#' 
## -----------------------------------------------------------------------------------------------
plot(history)

#' 
#' ### Evaluate the model
#' 
## -----------------------------------------------------------------------------------------------
model %>% evaluate(scale(test_data), test_labels, verbose = 0)

