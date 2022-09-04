library("caret")
library("gbm")
library("corrplot")
library("e1071")

# import both datasets 

# epa = existing product attributes (training data)

# npa = new product attributes (test data) 

# trying to predict the volumne for new products (see npa)

epa <- read.csv(file = "C:/Users/cralx2k/Desktop/AAA_UT Data Analytics Certificate Program_XTOL_Materials/Course 3/existingproductattributes2017.csv")

npa <- read.csv(file = "C:/Users/cralx2k/Desktop/AAA_UT Data Analytics Certificate Program_XTOL_Materials/Course 3/newproductattributes2017.csv")

#str(epa)
#str(npa)

# need to preprocess data 
# convert "ProductType" column to be various dummy variable columns

epa_dummy_list <- dummyVars(" ~ .", data=epa)

epa_cleaned <- data.frame(predict(epa_dummy_list, newdata = epa))

npa_dummy_list <- dummyVars(" ~ .", data=npa)

npa_cleaned <- data.frame(predict(npa_dummy_list, newdata = npa))

# drop the columns "ProductNumber", "BestSeller" and "ProfitMargin"

epa_cleaned$ProductNum <- NULL

epa_cleaned$BestSellersRank <- NULL

epa_cleaned$ProfitMargin <- NULL

npa_cleaned$ProductNum <- NULL

npa_cleaned$BestSellersRank <- NULL

npa_cleaned$ProfitMargin <- NULL 
  
# drop the column "x5StarReview" from both dataframes 
  
new_epa_cleaned <- epa_cleaned

new_epa_cleaned$x5StarReviews <- NULL

new_npa_cleaned <- npa_cleaned

new_epa_cleaned$x5StarReviews <- NULL


# check correlations -- focus on what variables correlate with the "Volume" variable

# corr_epa <- cor(epa_cleaned)
# corr_epa


# plots 

#plot(epa_cleaned$Volume, epa_cleaned$x4StarReviews)




# creates a heatmap, which shows the correlations of the epa_cleaned dataframe

# corrplot(corr_epa)

# Four variables had the strongest correlation with "Volume". 
# The strongest correlation was between "5 Star Review" and "Volume", with a correlation of 1.

# create the training and testing set using the epa dataframe

in_train <- createDataPartition(y=epa_cleaned$Volume,
                                p=0.75,
                                list=FALSE)

training <- epa_cleaned[in_train,]

testing <- epa_cleaned[-in_train,]


# create the training and testing set using the new_epa dataframe (experimental purposes)

new_in_train <- createDataPartition(y=new_epa_cleaned$Volume,
                                    p=0.75,
                                    list=FALSE)

new_training <- new_epa_cleaned[new_in_train,]
  
new_testing <- new_epa_cleaned[-new_in_train,]


set.seed(123)

# Linear Model (parametric method)

lm_model = lm(Volume ~ ., data=training)

summary(lm_model)



# SVM model 

svm_control <- trainControl(method="repeatedcv",
                            number=3,
                            repeats=2)

svm_fit <- train(Volume~., data=training,
                 method="svmLinear2",
                 trControl=svm_control,
                 tuneLength=5)

# svm_fit

# expresses what the important predictors are according to the model

#svm_importance <- varImp(svm_fit, scale=FALSE)

#svm_importance

svm_predictions <- predict(svm_fit, newdata=testing)

# svm_predictions


# SVM model using new_epa_cleaned training data

new_svm_control <- trainControl(method="repeatedcv",
                            number=3,
                            repeats=2)

new_svm_fit <- train(Volume~., data=new_training,
                 method="svmLinear2",
                 trControl=new_svm_control,
                 tuneLength=5)

# new_svm_fit

new_svm_predictions <- predict(new_svm_fit, newdata=new_testing)

# new_svm_predictions


# Random Forest Model

rf_control <- trainControl(method="repeatedcv",
                            number=1,
                            repeats=1)

rf_fit <- train(Volume ~ ., data=training,
                 method="rf")

# rf_fit

# expresses what the important predictors are according to the model

#rf_importance <- varImp(rf_fit, scale=FALSE)

#rf_importance

rf_predictions <- predict(rf_fit, newdata=testing)

# rf_predictions



# New RF model using new_epa_cleaned training data

new_rf_control <- trainControl(method="repeatedcv",
                               number=1,
                               repeats=1)

new_rf_fit <- train(Volume~., data=new_training,
                    method="rf")

# new_rf_fit

new_rf_predictions <- predict(new_rf_fit, newdata=new_testing)

# new_rf_predictions

# Gradient Boosting Model 

gbm_control <- trainControl(method="boot",
                            number=1)

gbm_fit <- train(Volume ~ ., data=training,
                 method="gbm",
                 trControl=gbm_control,
                 tuneLength=3)

# gbm_fit

# expresses what the important predictors are according to the model

# gbm_importance <- varImp(gbm_fit, scale=FALSE)

# gbm_importance

gbm_predictions <- predict(gbm_fit, newdata=testing)

# gbm_predictions


# New GBM model using new_epa_cleaned training data

new_gbm_control <- trainControl(method="boot",
                               number=1)

new_gbm_fit <- train(Volume~., data=new_training,
                    method="gbm",
                    trControl=new_gbm_control,
                    tuneLength=3)

# new_gbm_fit

new_gbm_predictions <- predict(new_gbm_fit, newdata=new_testing)

# new_gbm_predictions

# results of models and their predictions (includes 5 Star Review predictor)

svm_fit

svm_predictions


rf_fit

rf_predictions

gbm_fit

gbm_predictions


# results of new models and their new predictions (DOES NOT include 5 Star Review predictor)


new_svm_fit

new_svm_predictions

new_rf_fit

new_rf_predictions

new_gbm_fit

new_gbm_predictions

# prediction scores using the "best" model and the new products dataset 

new_rf_final_predictions <- predict(new_rf_fit, newdata=new_npa_cleaned)

new_rf_final_predictions

# creates a new csv file based on the above code

output <- npa

output$final_predictions <- new_rf_final_predictions

write.csv(output, file="C3T3_output.csv",
          row.names = TRUE)






