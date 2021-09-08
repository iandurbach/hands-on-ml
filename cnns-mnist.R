library(keras)
library(dplyr)
library(ggplot2)
library(tidyr)

# load data
load("data/mnist_small.Rdata")

# set up training and test data (there's no validation data here, for no good reason)
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y

dim(x_train)
dim(x_test)
dim(y_train)
dim(y_test)

# explore the data by plotting a few of the images

# some data manipulation to prepare for ggplot
x_train_long <- as.data.frame(x_train) %>%
  mutate(picture = 1:nrow(.)) %>%
  slice(1:5) %>% # Choose number of pictures to visualise
  pivot_longer(-picture, names_to = "index", values_to = "shade") %>%
  #gather(-picture, key = "index", value = "shade") %>%
  arrange(picture) %>%
  group_by(picture) %>%
  mutate(index = 1:784,
         y = rep(rev(seq(1:28)), 28),
         x = rep(1:28, each = 28)) %>%ungroup()

# visualize
ggplot(data = x_train_long, aes(x = x, y = y, fill = shade)) +
  geom_raster() +
  scale_fill_gradient(low = "white", high = "black") +
  coord_equal(expand = 0) +
  facet_wrap(~picture, nrow = 1)

# raw data for first number
x_train[1,,]

### first we'll ignore the fact that these are images and just use a feedforward NN on the 
### unravelled images

# reshape each images from 28x28 to one long 784x1 vector, this makes it just like the Aloe dataset 
dim(x_train) <- c(nrow(x_train), 784)
dim(x_test) <- c(nrow(x_test), 784)
# scale to lie between 0 and 1 (another way of rescaling)
x_train <- x_train / 255
x_test <- x_test / 255

# response variable is categorical with 10 levels
y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)

# check dims
dim(x_train)
dim(y_train)
dim(x_test)
dim(y_test)

### build a feedforward neural network
model <- keras_model_sequential() 

# we're adding dropout layers here for the first time
# these randomly turn off a proportion of the nodes in a layer each epoch, and help to avoid
# overfitting
model %>% 
  layer_dense(units = 128, activation = 'relu', input_shape = c(784)) %>% 
  layer_dropout(rate = 0.2) %>% 
  layer_dense(units = 64, activation = 'relu') %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 10, activation = 'softmax')

summary(model)

model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = 'adam',
  metrics = c('accuracy')
)

model %>% fit(
  x_train, y_train, 
  epochs = 5, batch_size = 32, 
  validation_split = 0.2
)

model %>% evaluate(x_test, y_test)

### build a convolutional neural network

# prepare input data (different format than for feedforward NN)
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y

dim(x_train)
dim(x_test)
dim(y_train)
dim(y_test)

img_rows <- 28 # size of image in x dim
img_cols <- 28 # size of image in ydim
# reshape as (n images, size in x, size in y, number of channels [3 for RGB, 1 for grayscale])
x_train <- array_reshape(x_train, c(nrow(x_train), img_rows, img_cols, 1))
x_test <- array_reshape(x_test, c(nrow(x_test), img_rows, img_cols, 1))

# scale to lie between 0 and 1 
x_train <- x_train / 255
x_test <- x_test / 255

# response variable is categorical with 10 levels
y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)

# check dims
dim(x_train)
dim(y_train)
dim(x_test)
dim(y_test)

# CNN

model <- keras_model_sequential() 

# add convolutional layers
model %>% 
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = "relu", 
                input_shape = c(28,28,1)) %>% 
  layer_max_pooling_2d(pool_size = c(2,2)) %>% 
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2,2))

# see model so far
summary(model)

# add dense layers at end
model %>% 
  layer_flatten() %>% 
  layer_dense(units = 64, activation = "relu") %>% 
  layer_dense(units = 10, activation = "softmax")

# see model again
summary(model)

model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = 'rmsprop',
  metrics = c('accuracy')
)

model %>% fit(
  x_train, y_train, 
  epochs = 5, batch_size = 32, 
  validation_split = 0.2
)

model %>% evaluate(x_test, y_test)

### Let's have a look at some of the images the model got WRONG

# get IDs of incorrectly predicted test cases
preds <- model %>% predict_classes(x_test)              # get predicted classes for test data
wrong_preds <- tibble(id = 1:nrow(y_test),              # make tibble with obs, pred, and id
                      obs = mnist$test$y, 
                      preds = preds) %>%
  filter(obs != preds)                                # identify incorrect classifications

# create a subset of the test data containing only wrong predictions
x_test_wrong <- x_test[wrong_preds$id, ]

# tranform wide to long, in preparation for ggplot
x_test_wrong_long <- as.data.frame(x_test_wrong) %>%
  mutate(picture = 1:nrow(.)) %>%
  slice(1:5) %>% # Choose number of pictures to visualise
  pivot_longer(-picture, names_to = "index", values_to = "shade") %>%
  arrange(picture) %>%
  group_by(picture) %>%
  mutate(index = 1:784,
         y = rep(rev(seq(1:28)), 28),
         x = rep(1:28, each = 28))

# visualise
ggplot(data = x_test_wrong_long, aes(x = x, y = y, fill = shade)) +
  geom_raster() +
  scale_fill_gradient(low = "white", high = "black") +
  coord_equal(expand = 0) +
  facet_wrap(~picture, nrow = 1)

