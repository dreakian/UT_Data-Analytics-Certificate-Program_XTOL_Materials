# clean data
# EDA
# prepare models 
# test models
# select best performing model

######### dependent variable needs to be a "factor" variable

library(dplyr)
library(caret)
library(doParallel)
library(C50)
library(randomForest)

######################### PARALLEL PROCESSING TO MAKE MODELS EXECUTE FASTER #############

# detectCores() -- 4 cores --- USE 3 SO THAT MEMORY IS NOT EXCEEDED AND THE PROGRAM IS NOT INTERRUPTED / MADE UNABLE TO FUNCTION PROPERLY !!!

cores_for_RS <- makeCluster(4)

registerDoParallel(cores_for_RS)

getDoParWorkers()

data <- read.csv(file="C:/Users/cralx2k/Desktop/AAA_UT Data Analytics Certificate Program_XTOL_Materials/Course 4/trainingData.csv")

# str(data)

# data = data %>% select(LONGITUDE, LATITUDE, FLOOR, BUILDINGID, SPACEID, RELATIVEPOSITION, USERID, PHONEID, everything()) # change position of columns (this was already done in Excel)

# is.null(data) # no null values

# sum(duplicated(data)) # 637 rows are duplicated


# need to reduce the size of the dataset 

# use table function to get a count of the values in each column

# table(data$FLOOR)
# 
# table(data$BUILDINGID)
# 
# table(data$SPACEID)
# 
# table(data$RELATIVEPOSITION)
# 
# table(data$RELATIVEPOSITION, data$FLOOR)
# 
# table(data$RELATIVEPOSITION, data$SPACEID)
# 
# table(data$RELATIVEPOSITION, data$BUILDINGID)
# 
# table(data$BUILDINGID, data$FLOOR)
# 
# table(data$BUILDINGID, data$SPACEID)

# results of previous functions:
# 1. Overwhelming majority of observations had a relative position of 2, which means most of the readings are "outside" a room
# 2. Floor 1 and Floor 3 have the most values. Floor 4 has the least observations.
# 3. Building 2 has the most observations.
# 4. It seems that that the most observations are associated with Relative IDs of 101, 102, 103, 104, 105,
#    106, 107, 108, 110, 111, 112, 122, 201, 202, 203, and 204

# replacing the values of each building ID to a letter

# building 0 = A
# building 1 = B
# building 2 = C

data$BUILDINGID[data$BUILDINGID == 0] <- "A"

data$BUILDINGID[data$BUILDINGID == 1] <- "B"

data$BUILDINGID[data$BUILDINGID == 2] <- "C"

#str(data$BUILDINGID)

################## splitting up the data #################

# creates subsets of the original dataframe based on the building code 

data_building_A <- data[ which(data$BUILDINGID=="A"), ]

data_building_B <- data[ which(data$BUILDINGID=="B"), ]

data_building_C <- data[ which(data$BUILDINGID=="C"), ]

################ organizing sub data frames 

# creates new column for each of the sub data frames which combine relevant location data

# moves the newly created column into the first position of each sub data frames

# BUILDING A #######################

data_building_A <- cbind(data_building_A, paste(data_building_A$BUILDINGID,"_",data_building_A$FLOOR,"_",data_building_A$SPACEID,"_",data_building_A$RELATIVEPOSITION), stringsAsFactors=TRUE)

colnames(data_building_A)[530] <- "Location"

data_building_A <- data_building_A[,c(ncol(data_building_A), 1:(ncol(data_building_A)-1))]

# BUILDING B #########################

data_building_B <- cbind(data_building_B, paste(data_building_B$BUILDINGID,"_",data_building_B$FLOOR,"_",data_building_B$SPACEID,"_",data_building_B$RELATIVEPOSITION), stringsAsFactors=TRUE)

colnames(data_building_B)[530] <- "Location"

data_building_B <- data_building_B[,c(ncol(data_building_B), 1:(ncol(data_building_B)-1))]

# BUILDING C ############################

data_building_C <- cbind(data_building_C, paste(data_building_C$BUILDINGID,"_",data_building_C$FLOOR,"_",data_building_C$SPACEID,"_",data_building_C$RELATIVEPOSITION), stringsAsFactors=TRUE)

colnames(data_building_C)[530] <- "Location"

data_building_C <- data_building_C[,c(ncol(data_building_C), 1:(ncol(data_building_C)-1))]

###### REMOVING UNNECESSARY COLUMNS FROM SUB DATA FRAMES ########################

# removes longitude, latitude, floor #, building ID, space ID, relative position, user ID, phoneID, timestamp

# only the LOCATION column and all the WAP columns should remain in the sub data frames

# removing unnecessary columns for sub data frame one (building A)

dropped_columns <- names(data_building_A) %in% c("LONGITUDE", "LATITUDE", "FLOOR", "BUILDINGID", "SPACEID", "RELATIVEPOSITION", "USERID", "PHONEID", "TIMESTAMP")

data_building_A_cleaned <- data_building_A[!dropped_columns] 

# removing unnecessary columns for sub data frame one (building B)

dropped_columns <- names(data_building_B) %in% c("LONGITUDE", "LATITUDE", "FLOOR", "BUILDINGID", "SPACEID", "RELATIVEPOSITION", "USERID", "PHONEID", "TIMESTAMP")

data_building_B_cleaned <- data_building_B[!dropped_columns] 

# removing unnecessary columns for sub data frame one (building C)

dropped_columns <- names(data_building_C) %in% c("LONGITUDE", "LATITUDE", "FLOOR", "BUILDINGID", "SPACEID", "RELATIVEPOSITION", "USERID", "PHONEID", "TIMESTAMP")

data_building_C_cleaned <- data_building_C[!dropped_columns] 
 
####################### basic EDA for the ENTIRE DATAFRAME ######################

# data <- cbind(data, paste(data$BUILDINGID,"_",data$FLOOR,"_",data$SPACEID,"_",data$RELATIVEPOSITION), stringsAsFactors=TRUE)
# 
# colnames(data)[530] <- "Location"
# 
# location_by_building_counts <- table(data$Location, data$BUILDINGID)
# 
# barplot(location_by_building_counts, main="Location Distribution",
#         xlab="Building Type")
# 
# floor_counts <- table(data$FLOOR)
# 
# barplot(sort(floor_counts, decreasing=TRUE), main="Floor Distribution",
#         xlab="Floor number")
# 
# building_counts <- table(data$BUILDINGID)
# 
# barplot(sort(building_counts, decreasing=TRUE), main="Building Distribution",
#         xlab="Building Type")
# 
# space_counts <- table(data$SPACEID)
# 
# barplot(sort(space_counts, decreasing=TRUE), main="Space Distribution",
#         xlab="Space Number")
# 
# relative_position_counts <- table(data$RELATIVEPOSITION)
# 
# barplot(sort(relative_position_counts, decreasing=TRUE), main="Relative Position Distribution",
#         xlab="Number of Relative Positions")

###################### FURTHER REDUCING THE SIZE OF THE SUB DATAFRAMES ######################

# using a function that identifies columns that have non-zero variance 
# keep the WAP that have a FALSE value for zero variance

# remove WAP that do no detect any wifi signals at all

# building A non-zero variance 

non_zero_variance_metrics_building_A <- nearZeroVar(data_building_A_cleaned, saveMetrics=TRUE)

non_zero_variance_metrics_building_A

zero_variance_building_A <- which(non_zero_variance_metrics_building_A$zeroVar == 1)

data_building_A_cleaned <- data_building_A_cleaned[,-(zero_variance_building_A)]


# building B non-zero variance 

non_zero_variance_metrics_building_B <- nearZeroVar(data_building_B_cleaned, saveMetrics=TRUE)

non_zero_variance_metrics_building_B

zero_variance_building_B <- which(non_zero_variance_metrics_building_B$zeroVar == 1)

data_building_B_cleaned <- data_building_B_cleaned[,-(zero_variance_building_B)]


# building C non-zero variance 

non_zero_variance_metrics_building_C <- nearZeroVar(data_building_C_cleaned, saveMetrics=TRUE)

non_zero_variance_metrics_building_C

zero_variance_building_C <- which(non_zero_variance_metrics_building_C$zeroVar == 1)

data_building_C_cleaned <- data_building_C_cleaned[,-(zero_variance_building_C)]

############################ MODEL CREATION AND EVALUATION #################################################

# need to create separate train/test splits that relate to each specific sub data frame

# models to test:KNN + C5.0 + Random Forest 

# use 10-fold cross validation

# performance metrics: Kappa + Accuracy 

############ ML SECTION ################

########################################################################################
####### 1. Data Building A
#########################################################################################
####### SET SEED FOR MODELS

set.seed(123)

########### MODEL PIPELINE FOR C5.0

### create data partition

in_train_DB_A <- createDataPartition(y = data_building_A_cleaned$Location,
                                     p = 0.75,
                                     list = FALSE)

# use 75% of training data to train and test models

training_DB_A <- data_building_A_cleaned[in_train_DB_A,]

# use the remainder (25%) of data to validate models

testing_DB_A <- data_building_A_cleaned[-in_train_DB_A,]


#### create controls for the model

c50_control <- trainControl(method="repeatedcv", number=10,
                            repeats = 3)
metric <- "Accuracy"

# create C5.0 model for data building A 

c50_fit_DB_A <- train(Location~., data=training_DB_A, method="C5.0",
                      metric=metric, trControl=c50_control, 
                      tuneLength = 2)

# determines and reports how well the model predicts the dependent variable, using the testing data

c50_predictions_DB_A <- predict(c50_fit_DB_A, testing_DB_A)

postResample(c50_predictions_DB_A, testing_DB_A$Location)

############################################################################################
####### 2. Data Building B
############################################################################################

in_train_DB_B <- createDataPartition(y = data_building_B_cleaned$Location,
                                     p = 0.75,
                                     list = FALSE)


training_DB_B <- data_building_B_cleaned[in_train_DB_B,]

testing_DB_B <- data_building_B_cleaned[-in_train_DB_B,]

c50_fit_DB_B <- train(Location~., data=training_DB_B, method="C5.0",
                      metric=metric, trControl=c50_control, 
                      tuneLength = 2)

c50_predictions_DB_B <- predict(c50_fit_DB_B, testing_DB_B)

postResample(c50_predictions_DB_B, testing_DB_B$Location)

#########################################################################################################################
####### 3. Data Building C
#########################################################################################################################

in_train_DB_C <- createDataPartition(y = data_building_C_cleaned$Location,
                                     p = 0.75,
                                     list = FALSE)

training_DB_C <- data_building_C_cleaned[in_train_DB_C,]

testing_DB_C <- data_building_C_cleaned[-in_train_DB_C,]

c50_fit_DB_C <- train(Location~., data=training_DB_C, method="C5.0",
                      metric=metric, trControl=c50_control, 
                      tuneLength = 1)

c50_predictions_DB_C <- predict(c50_fit_DB_C, testing_DB_C)

postResample(c50_predictions_DB_C, testing_DB_C$Location)

#########################################################################################################################################################
##########################################################################################################################################################
###########################################################################################################################################################
########### MODEL PIPELINE FOR KKN
########################################################################################

####### 4. Data Building A
#########################################################################################

### create data partition

in_train_DB_A <- createDataPartition(y = data_building_A_cleaned$Location,
                                     p = 0.75,
                                     list = FALSE)

training_DB_A <- data_building_A_cleaned[in_train_DB_A,]

testing_DB_A <- data_building_A_cleaned[-in_train_DB_A,]

knn_control <- trainControl(method="repeatedcv", number=10,
                            repeats = 3)
metric <- "Accuracy"

knn_fit_DB_A <- train(Location~., data=training_DB_A, method="knn",
                      metric=metric, trControl=knn_control, 
                      tuneLength = 2)

knn_predictions_DB_A <- predict(knn_fit_DB_A, testing_DB_A)

postResample(knn_predictions_DB_A, testing_DB_A$Location)

############################################################################################
####### 5. Data Building B
############################################################################################


in_train_DB_B <- createDataPartition(y = data_building_B_cleaned$Location,
                                     p = 0.75,
                                     list = FALSE)

training_DB_B <- data_building_B_cleaned[in_train_DB_B,]

testing_DB_B <- data_building_B_cleaned[-in_train_DB_B,]

knn_fit_DB_B <- train(Location~., data=training_DB_B, method="knn",
                      metric=metric, trControl=knn_control, 
                      tuneLength = 2)

knn_predictions_DB_B <- predict(knn_fit_DB_B, testing_DB_B)

postResample(knn_predictions_DB_B, testing_DB_B$Location)

#########################################################################################################################
####### 6. Data Building C
#########################################################################################################################

in_train_DB_C <- createDataPartition(y = data_building_C_cleaned$Location,
                                     p = 0.75,
                                     list = FALSE)

training_DB_C <- data_building_C_cleaned[in_train_DB_C,]

testing_DB_C <- data_building_C_cleaned[-in_train_DB_C,]

knn_fit_DB_C <- train(Location~., data=training_DB_C, method="knn",
                      metric=metric, trControl=c50_control, 
                      tuneLength = 2)

knn_predictions_DB_C <- predict(knn_fit_DB_C, testing_DB_C)

postResample(knn_predictions_DB_C, testing_DB_C$Location)

#########################################################################################################################################################
##########################################################################################################################################################
###########################################################################################################################################################

########### MODEL PIPELINE FOR RANDOM FOREST

########################################################################################
####### 7. Data Building A
#########################################################################################


in_train_DB_A <- createDataPartition(y = data_building_A_cleaned$Location,
                                     p = 0.75,
                                     list = FALSE)

training_DB_A <- data_building_A_cleaned[in_train_DB_A,]

testing_DB_A <- data_building_A_cleaned[-in_train_DB_A,]


rf_control <- trainControl(method="repeatedcv", 
                           number= 10, repeats = 1)

# rfGrid <- expand.grid(mtry=c(24))

# 
# rf_fit_DB_A <- train(Location~., data=training_DB_A, method="rf",
#                      trControl=rf_control, 
#                      tuneGrid = rfGrid)

rf_fit_DB_A <- train(Location~., data=training_DB_A, method="rf",
                     trControl=rf_control,
                     tuneLength = 10)

rf_fit_DB_A

########################################################################################
####### 8. Data Building B
#########################################################################################

in_train_DB_B <- createDataPartition(y = data_building_B_cleaned$Location,
                                     p = 0.75,
                                     list = FALSE)

training_DB_B <- data_building_B_cleaned[in_train_DB_B,]

testing_DB_B <- data_building_B_cleaned[-in_train_DB_B,]

rf_fit_DB_B <- train(Location~., data=training_DB_B, method="rf",
                     trControl=rf_control, 
                     tuneLength = 10)

rf_fit_DB_B # USE MTRY 47

########################################################################################
####### 9. Data Building C
#########################################################################################

in_train_DB_C <- createDataPartition(y = data_building_C_cleaned$Location,
                                     p = 0.75,
                                     list = FALSE)

training_DB_C <- data_building_C_cleaned[in_train_DB_C,]

testing_DB_C <- data_building_C_cleaned[-in_train_DB_C,]

rf_control <- trainControl(method="repeatedcv", 
                           number= 10, repeats = 1)

rf_fit_DB_C <- train(Location~., data=training_DB_C, method="rf",
                     trControl=rf_control, 
                     tuneLength = 10)

rf_fit_DB_C # USE MTRY 24

#####################################################################################################
#### MODEL EVALUATIONS


# DB_A MODELS

# model_data_DB_A <- resamples(list(C50_model_DB_A = c50_fit_DB_A, KNN_model_DB_A = knn_fit_DB_A, RF_model_DB_A = rf_fit_DB_A))
# 
# summary(model_data_DB_A)
# 
# # DB_B MODELS
# 
# model_data_DB_B <- resamples(list(C50_model_DB_B = c50_fit_DB_B, KNN_model_DB_B = knn_fit_DB_B, RF_model_DB_B = rf_fit_DB_B))
# 
# Summary(model_data_DB_B)
# 
# # DB_C MODELS
# 
# model_data_DB_C <- resamples(list(C50_model_DB_C = c50_fit_DB_C, KNN_model_DB_C = knn_fit_DB_C, RF_model_DB_C = rf_fit_DB_C))
# 
# Summary(model_data_DB_C)



stopCluster(cores_for_RS)
