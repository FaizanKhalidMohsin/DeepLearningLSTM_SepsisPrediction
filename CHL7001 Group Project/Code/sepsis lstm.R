library(tidyverse) # importing, cleaning, visualising 
library(keras) # deep learning with keras
library(data.table) # fast csv reading
setwd("")

# load data
data <- fread(file.choose())
data <- as.data.frame(data)

data <- ifelse(is.na(data), 0, data)

# split data
ind <- unique(sepsis_data$filename)
split <- sample(c(0,1,2), length(ind), replace = TRUE, prob = c(0.6, 0.2, 0.2))
names(split) <- ind

data$split <- split[data$filename]


# setup input data
train <- data[data$split == 0,]
valid <- data[data$split == 1,]
test <- data[data$split == 2,]

x.train = array(data = lag(cbind(train$price, train$vol), datalags)[-(1:datalags), ], 
                dim = c(nrow(train) - datalags, datalags, 2))
y.train = array(data = train$price[-(1:datalags)], dim = c(nrow(train)-datalags, 1))

x.train = array(data = lag(cbind(train$price, train$vol), datalags)[-(1:datalags), ], 
                dim = c(nrow(train) - datalags, datalags, 2))
y.train = array(data = train$price[-(1:datalags)], dim = c(nrow(train)-datalags, 1))

x.test = array(data = lag(cbind(test$vol, test$price), datalags)[-(1:datalags), ], 
               dim = c(nrow(test) - datalags, datalags, 2))
y.test = array(data = test$price[-(1:datalags)], dim = c(nrow(test) - datalags, 1))

# setup lstm
model <- keras_model_sequential()

model %>%
  layer_lstm(units = dim(y.train)[2], name = "lstm",
             input_shape = dim(x.train),
             dropout = 0.25, recurrent_dropout = 0.25,
             return_sequences = TRUE) %>%
  time_distributed(layer_dense(units = 1, name = "predictions",
                               activation = "sigmoid"))

model %>%
  compile(
    loss = "binary_crossentropy",
    optimizer = optimizer_adam(),
    metrics = list("binary_accuracy")
  )

print(model)

# run lstm
history <- model %>% fit(
  x = x.train,
  y = y.train,
  validation_data = list(x.valid, y.valid),
  batch_size = FLAGS$batch_size,
  epochs = FLAGS$n_epochs,
  shuffle = FALSE
)

print(history)
plot(history)

model %>% reset_states()

# lstm prediction
classes <- model %>% predict_classes(x.test)
table(y.test %*% 0:1, classes)

score <- model %>% evaluate(x.test, y.test)











