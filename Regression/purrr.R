library(tidyverse)
library(lubridate)
library(MLmetrics)

# Import Data

house <- read.csv("Regression/data_input/kc_house_data.csv")

# quick checking

glimpse(house)

# remove unnecessary column

house <- house %>% 
  select(-c(id, zipcode, lat, long, date))

# Cross Validation

set.seed(100)

idx <- sample(x = nrow(house), size = 0.8 * nrow(house))
train <- house[idx,]
test <- house[-idx,]

# Transformation

## original

data_ori <- colnames(train)

## Box-cox

boxcox <- train %>% 
  map(., ~ BoxCox(.,lambda = 2)) %>% 
  as_tibble()
colnames(boxcox) <- paste0(colnames(boxcox), ".bc")
data_boxcox <- colnames(boxcox)

# gathering data

train <- train %>% 
  bind_cols(boxcox)

# make sure all data is numerical, for Xgboost model.

train <- train %>% 
  mutate_all(as.numeric) %>% 
  as.data.frame()

glimpse(train)

# get object name

pred_varsets <- ls(pattern = "data_")
pred_varsets
n_var <- length(pred_varsets)

df_model <- list(train) %>% 
  rep(n_var) %>% 
  enframe(name = "id", value = "data") %>% 
  mutate(transformation = pred_varsets)

# function to filter x & y variable

filtercol <- function(x,y){
  
  x[,(colnames(x) %in% eval(parse(text = y)))]
  
}

# prepare x and y colomn

df_model <- df_model %>% 
  transmute(
    id,
    transformation,
    train_x = map2(data, transformation, ~filtercol(.x, .y)),
    train_y = map(data, ~.x$price)
  )

df_model

# desain model

## decision tree

rpartModel <- function(X, Y) {
  ctrl <- trainControl(
    ## 5-fold CV
    method = "repeatedcv",
    number = 5
  )
  train(
    x = X,
    y = Y,
    method = 'rpart2',
    trControl = ctrl,
    tuneGrid = data.frame(maxdepth=c(2,3,4,5)),
    preProc = c('center', 'scale')
  )
}


## xgboost

xgbTreeModel <- function(X,Y){
  ctrl <- trainControl(
    ## 5-fold CV
    method = "repeatedcv",
    number = 5
  )
  train(
    x=X,
    y=Y,
    method = 'xgbTree',
    trControl = ctrl,
    tuneGrid = expand.grid(nrounds = c(100,300,500), 
                           max_depth = c(2,4,6) ,
                           eta = 0.1,
                           gamma = 1, 
                           colsample_bytree = 1, 
                           min_child_weight = 1, 
                           subsample = 1),
    preProc = c('center', 'scale')
  )
}

model_list <- list(rpartModel = rpartModel,
                   xgbModel = xgbTreeModel) %>%
  enframe(name = 'modelName',value = 'model')

model_list

df_model <- df_model[rep(1:nrow(df_model),nrow(model_list)),]

df_model <- df_model %>% 
  bind_cols(
    model_list[rep(1:nrow(model_list),n_var),] %>% arrange(modelName)
  ) %>% 
  mutate(id = 1:nrow(.))

df_model

# training the model

model_fit <- df_model %>% 
  mutate(
    params = map2(train_x, train_y, ~list(X = .x, Y = .y)),
    modelfits = invoke_map(model, params)
  )

model_fit %>% 
  select(transformation, modelName, params, modelfits)


model_fit %>% 
  mutate(
    prediction = map(modelfits, ~predict(.x, .x$train_x)),
    mae = map2(prediction, train_y, ~MLmetrics::MAE(.x, .y)),
    rmse = map2(prediction, train_y, ~MLmetrics::RMSE(.x, .y))
  ) %>% 
  select(transformation, modelName, mae, rmse) %>% 
  unnest(cols = c(mae, rmse))

