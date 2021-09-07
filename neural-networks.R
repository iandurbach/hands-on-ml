library(keras)
library(dplyr)
library(ggplot2)
library(tidyr)

load("data/aloe.RData")

# have a look at data
head(aloe_pa)
table(aloe_pa$present)
table(aloe_pa$train_id) # 1 = train, 2 = validation, 3 = test

### Step 1: preparing data for keras

## 4 steps:
# 1) identify the observations you want to use in the training, validation, test datasets
x_train <- aloe_pa %>% filter(train_id == 1) 
x_val <- aloe_pa %>% filter(train_id == 2) 
x_test <- aloe_pa %>% filter(train_id == 3) 
# 2) identify the response variable 
y_train <- x_train$present
y_val <- x_val$present
y_test <- x_test$present
# 3) identify the predictor variables 
names(aloe_pa)
names_we_want <- names(aloe_pa)[2:21]
x_train <- x_train %>% select(names_we_want) 
x_val <- x_val %>% select(names_we_want) 
x_test <- x_test %>% select(names_we_want) 
# 4) keras wants input data as matrices, so convert to this format
x_train <- as.matrix(x_train) 
x_val <- as.matrix(x_val) 
x_test <- as.matrix(x_test) 
y_train <- as.matrix(y_train, ncol = 1)
y_val <- as.matrix(y_val, ncol = 1)
y_test <- as.matrix(y_test, ncol = 1)

## check dimensions
dim(x_train)
dim(y_train)
dim(x_val)
dim(y_val)
dim(x_test)
dim(y_test)

## Often good idea to scale data so all variables lie on nearly same scale 
# one way to do this is to divide each variable by its mean and divide by its sd
# means and sd's should come from the TRAINING set, even if scaling the validation and test
# sets.
x_train <- scale(x_train)
# 'scale' function stores means and sd's as attribute of the scaled object, so can use these to 
# scale variables in the validation and test dataset
x_val <- scale(x_val, center = attr(x_train, "scaled:center"), scale = attr(x_train, "scaled:scale"))
x_test <- scale(x_test, center = attr(x_train, "scaled:center"), scale = attr(x_train, "scaled:scale"))

### Step 2: build a feedforward neural network

# initialise model
model <- keras_model_sequential()

# add layers
model %>% 
  layer_dense(units = 128,                  # number of neurons in the hidden layer
              input_shape = c(20)) %>%     # dimension of input array
  layer_activation('relu') %>%             # use a rectified linear unit as an activation function in the hidden layer
  layer_dense(units = 1) %>%               # adds an output layer to the network
  layer_activation('sigmoid')              # use sigmoid activation function in the output layer

# print model so far
summary(model)

# compile model
model %>% compile(
  optimizer = 'adam',
  loss = 'binary_crossentropy',
  metrics = c('accuracy')
)

# fit model
model %>% fit(x_train, y_train, 
              epochs = 20, 
              batch_size = 32,
              validation_data = list(x_val, y_val)) %>% plot()

# model gets updated so if want to rain for more epochs, just run again
model %>% fit(x_train, y_train, epochs = 50, batch_size = 32, validation_data = list(x_val, y_val))

# use fitted model to evaluate on another dataset
model %>% evaluate(x_test, y_test, batch_size=32, verbose = 1)

# get predictions on another dataset
model %>% predict_classes(x_test) 

## ----------------------------------------------------------------------------------------------------------

### Trying a model with an extra layer

model2 <- keras_model_sequential()

model2 %>% 
  layer_dense(units = 64, activation = 'relu', input_shape = c(20)) %>% 
  layer_dense(units = 64, activation = 'relu') %>% 
  layer_dense(units = 1, activation = 'sigmoid') %>% 
  compile(
    loss = 'binary_crossentropy',
    optimizer = 'adam',
    metrics = c('accuracy')
  )

summary(model2)


model2 %>% fit(x_train, y_train, 
               epochs = 70, 
               batch_size = 32,
               validation_data = list(x_val, y_val)) %>% plot()

model2 %>% evaluate(x_test, y_test, batch_size=32, verbose = 1)