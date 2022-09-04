
library("caret")
library("gbm")

# df_complete is used to train the models

# df_incomplete is used as the testing set with the best performing model in order to actually predict the dependent variable 


df_complete <- read.csv(file = "C:/Users/cralx2k/Desktop/AAA_UT Data Analytics Certificate Program_XTOL_Materials/Course 3/CompleteResponses.csv")

df_incomplete <- read.csv(file = "C:/Users/cralx2k/Desktop/AAA_UT Data Analytics Certificate Program_XTOL_Materials/Course 3/SurveyIncomplete.csv")


# need to convert "brand" column to be a factor type

df_complete$brand <- as.factor(df_complete$brand)

df_incomplete$brand <- as.factor(df_incomplete$brand)


# create a list of 75% of the rows in the dataframe for training data

in_train <- createDataPartition(y = df_complete$brand,
                                p = 0.75,
                                list = FALSE)

# use 75% of training data to train and test models

training <- df_complete[in_train,]

# use the remainder (25%) of data to validate models

testing <- df_complete[-in_train,]

# sets the seed for all the models

set.seed(123)

# applies 10-fold cross validation + Automatic Tuning Grid onto the model

gbm_control <- trainControl(method="repeatedcv", number=10,
                            repeats = 10)
metric <- "Accuracy"

# stochastic gradient boosting (GBM) algorithm

gbm_fit <- train(brand~., data=training, method="gbm",
                metric=metric, trControl=gbm_control, 
                tuneLength = 5)

# determines and reports how well the model predicts the dependent variable, using the testing data

gbm_predictions <- predict(gbm_fit, testing)

postResample(gbm_predictions, testing$brand)

# ranks importance of features for GBM model

gbm_importance <- varImp(gbm_fit, scale=FALSE)

print(gbm_importance)

plot(gbm_importance)

gbm_predictions_complete <- predict(gbm_fit, df_complete)

summary(gbm_predictions_complete)

# random forest algorithm

# applies 10-fold cross validation + Manual Tuning Grid onto the model

rf_control <- trainControl(method="repeatedcv", 
                        number= 10, repeats = 1)
rf_grid <- expand.grid(mtry=c(1, 2, 3, 4, 5))

rf_fit <- train(brand~., data=training, method="rf",
                metric=metric, trControl=rf_control, 
                tuneGrid=rf_grid)

# determines and reports how well the model predicts the dependent variable, using the testing data

rf_fit

rf_importance <- varImp(rf_fit, scale=FALSE)

print(rf_importance)

plot(rf_importance)

# C5.0 algorithm 

c5_control <- trainControl(method="repeatedcv", 
                           number= 10, repeats = 5)

c5_fit <- train(brand~., data=training, method="C5.0",
                metric=metric, trControl=c5_control,
                tuneLength =  1)

# determines and reports how well the model predicts the dependent variable, using the testing data

c5_predictions <- predict(c5_fit, testing)

postResample(c5_predictions, testing$brand)

# ranks importance of features for C5.0 model

c5_importance <- varImp(c5_fit, scale=FALSE)

print(c5_importance)

plot(c5_importance)

# making predictions using the surveryIncomplete data -- uses best performing model, which was GBM

gbm_real_predictions <- predict(gbm_fit, df_incomplete)

postResample(gbm_real_predictions, df_incomplete$brand)

summary(gbm_real_predictions)

plot(gbm_real_predictions)
