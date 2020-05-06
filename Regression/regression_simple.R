library(tidyverse)
library(lubridate)
library(MLmetrics)

# Import Data

house <- read_csv("Regression/data_input/kc_house_data.csv")

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

# Model Fitting

model_lm <- lm(formula = price ~ ., data = train)

summary(model_lm)

# step wise regression

model_step <- step(model_lm, direction = "backward")

summary(model_step)

# predict to data test

predict_lm <- predict(model_step, newdata = test)

# Model Evaluation

MLmetrics::MAE(y_pred = predict_lm, y_true = test$price)
MLmetrics::RMSE(y_pred = predict_lm, y_true = test$price)
