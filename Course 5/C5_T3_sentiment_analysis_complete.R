############### the two primary datasets will be modified (have various Feature Selection / Feature Engineering done onto to them so as to train the different models)

############## the models to be created are C5.0, RandomForest, KKNN (NOT KNN), and SVM ()


library(dplyr)
library(caret)
library(doParallel)
library(C50)
library(randomForest)
library(e1071)
library(plotly)
library(corrplot)
library(kknn)
library(sys)
library(tidyr)
library(readxl)

cl <- makeCluster(3)
registerDoParallel(cl)

# getDoParWorkers()


# C:\Users\Lyon P. Abido\Desktop\R_stuff

data_iphone <-
  read.csv(file = "C:/Users/Lyon P. Abido/Desktop/R_stuff/iphone_smallmatrix_labeled_8d.csv")

data_galaxy <-
  read.csv(file = "C:/Users/Lyon P. Abido/Desktop/R_stuff/galaxy_smallmatrix_labeled_9d.csv")


# str(data_iphone)
# summary(data_iphone)
# 
# str(data_galaxy)
# summary(data_galaxy)
# 
# table(data_iphone$ios)
# table(data_iphone$iphonecampos)
# table(data_iphone$htcperunc)
table(data_iphone$iphonesentiment)


#########################################################


# table(data_galaxy$ios)
# table(data_galaxy$iphonecampos)
# table(data_galaxy$htcperunc)
table(data_galaxy$galaxysentiment)

# no null data for either dataset

# is.null(data_iphone)

# is.null(data_galaxy)


# plot_ly(data_iphone,
#         x = ~ data_iphone$iphonesentiment,
#         type = "histogram")
# 
# plot_ly(data_galaxy,
#         x = ~ data_galaxy$galaxysentiment,
#         type = "histogram")

# plot_ly(data_iphone,
#          x= ~ data_iphone$iphone,
#          type="histogram")

# Examine Correlation (Do Classification problems suffer from Collinearity?)


# cor_iphone <- cor(data_iphone)
# 
# cor_galaxy <- cor(data_galaxy)
# 
# options(max.print = 1000000)
# 
# cor_iphone
# 
# cor_galaxy
# 
# corrplot(cor_iphone)
# 
# corrplot(cor_galaxy)

# Correlation Plots for either dataset demonstrate that there are no features that are collinear with the dependent variable



# Examine Feature Variance

nzv_metrics_iphone_true <-
  nearZeroVar(data_iphone, saveMetrics = TRUE)
nzv_metrics_iphone_true

nzv_metrics_iphone_false <-
  nearZeroVar(data_iphone, saveMetrics = FALSE)
nzv_metrics_iphone_false

NZV_iphone <- data_iphone[,-nzv_metrics_iphone_false]

# str(NZV_iphone)
# summary(NZV_iphone)

# cor_NZV_iphone <- cor(NZV_iphone)
# corrplot(cor_NZV_iphone)


###################################################


nzv_metrics_galaxy_true <-
  nearZeroVar(data_galaxy, saveMetrics = TRUE)
nzv_metrics_galaxy_true

nzv_metrics_galaxy_false <-
  nearZeroVar(data_galaxy, saveMetrics = FALSE)
nzv_metrics_galaxy_false

NZV_galaxy <- data_galaxy[,-nzv_metrics_galaxy_false]

# str(NZV_galaxy)
# summary(NZV_galaxy)
# 
# cor_NZV_galaxy <- cor(NZV_galaxy)
# corrplot(cor_NZV_galaxy)



# Recursive Feature Elimination

# RFE is a form of automated feature selection. It used a Random Forest algorithm to ascertain optimal feature subsets




set.seed(123)

iphoneSample <-
  data_iphone[sample(1:nrow(data_iphone), 1000, replace = FALSE), ]

ctrl <- rfeControl(
  functions = rfFuncs,
  method = "repeatedcv",
  repeats = 5,
  verbose = FALSE
)

rfeResults_iphone <- rfe(
  iphoneSample[, 1:58],
  iphoneSample$iphonesentiment,
  sizes = (1:58),
  rfeControl = ctrl
)

# rfeResults_iphone
# 
# plot(rfeResults_iphone, type = c("g", "o"))

RFE_iphone <- data_iphone[, predictors(rfeResults_iphone)]

RFE_iphone$iphonesentiment <- data_iphone$iphonesentiment

# str(RFE_iphone)


######################################################################################

galaxySample <-
  data_galaxy[sample(1:nrow(data_galaxy), 1000, replace = FALSE), ]

ctrl <- rfeControl(
  functions = rfFuncs,
  method = "repeatedcv",
  repeats = 5,
  verbose = FALSE
)

rfeResults_galaxy <- rfe(
  galaxySample[, 1:58],
  galaxySample$galaxysentiment,
  sizes = (1:58),
  rfeControl = ctrl
)

# rfeResults_galaxy
# 
# plot(rfeResults_galaxy, type = c("g", "o"))

RFE_galaxy <- data_galaxy[, predictors(rfeResults_galaxy)]

RFE_galaxy$galaxysentiment <- data_galaxy$galaxysentiment

# str(RFE_galaxy)

########################################


# dependent variables for either dataset = "iphonesentiment" and "galaxysentiment"
# dependent variable needs to be converted to a factor type

data_iphone$iphonesentiment <-
  as.factor(data_iphone$iphonesentiment)

data_galaxy$galaxysentiment <-
  as.factor(data_galaxy$galaxysentiment)


NZV_iphone$iphonesentiment <-
  as.factor(NZV_iphone$iphonesentiment)

NZV_galaxy$galaxysentiment <-
  as.factor(NZV_galaxy$galaxysentiment)


RFE_iphone$iphonesentiment <- 
  as.factor(RFE_iphone$iphonesentiment)

RFE_galaxy$galaxysentiment <- 
  as.factor(RFE_galaxy$galaxysentiment)

###########################################################################################
###########################################################################################

## Model Development

### Models to be used: C5.0 + Random Forest + SVM (e1071 package) + KKNN

### Training datasets to use for models:

# Preprocessed datasets (original)

# Datasets without Non Zero Variance Features (NZV datasets)

# Datasets from Recursive Feature Elimination (RFE datasets)

metric <- "Accuracy"

control <- trainControl(method = "repeatedcv",
                        number = 10,
                        repeats = 2)

################# IPHONE DATASET -- ORIGINAL

in_train_iphone_original <-
  createDataPartition(y = data_iphone$iphonesentiment,
                      p = 0.70,
                      list = FALSE)

training_iphone_original <- data_iphone[in_train_iphone_original, ]

testing_iphone_original <- data_iphone[-in_train_iphone_original, ]

system.time(
  c50_fit_iphone_original <-
    train(
      iphonesentiment ~ .,
      data = training_iphone_original,
      method = "C5.0",
      metric = metric,
      trControl = control,
      tuneLength = 1
    )
)

system.time(
  rf_fit_iphone_original <-
    train(
      iphonesentiment ~ .,
      data = training_iphone_original,
      method = "rf",
      trControl = control,
      metric = metric,
      tuneLength = 1
    )
)

system.time(
  svm_fit_iphone_original <-
    train(
      iphonesentiment ~ .,
      data = training_iphone_original,
      method = "svmLinear2",
      trControl = control,
      metric = metric,
      tuneLength = 1
    )
)

system.time(
  kknn_fit_iphone_original <-
    train(
      iphonesentiment ~ .,
      data = training_iphone_original,
      method = "kknn",
      metric = metric,
      trControl = control,
      tuneLength = 1
    )
)

#### RESULTS FOR ORIGINAL IPHONE

c50_predictions_iphone_original <-
  predict(c50_fit_iphone_original, testing_iphone_original)

postResample(c50_predictions_iphone_original,
             testing_iphone_original$iphonesentiment)

rf_fit_iphone_original

svm_predictions_iphone_original <-
  predict(svm_fit_iphone_original, testing_iphone_original)

postResample(svm_predictions_iphone_original,
             testing_iphone_original$iphonesentiment)

kknn_predictions_iphone_original <-
  predict(kknn_fit_iphone_original, testing_iphone_original)

postResample(kknn_predictions_iphone_original,
             testing_iphone_original$iphonesentiment)

################# GALAXY DATASET -- ORIGINAL

in_train_galaxy_original <-
  createDataPartition(y = data_galaxy$galaxysentiment,
                      p = 0.70,
                      list = FALSE)

training_galaxy_original <- data_galaxy[in_train_galaxy_original, ]

testing_galaxy_original <- data_galaxy[-in_train_galaxy_original, ]

system.time(
  c50_fit_galaxy_original <-
    train(
      galaxysentiment ~ .,
      data = training_galaxy_original,
      method = "C5.0",
      metric = metric,
      trControl = control,
      tuneLength = 1
    )
)

system.time(
  rf_fit_galaxy_original <-
    train(
      galaxysentiment ~ .,
      data = training_galaxy_original,
      method = "rf",
      trControl = control,
      metric = metric,
      tuneLength = 1
    )
)

system.time(
  svm_fit_galaxy_original <-
    train(
      galaxysentiment ~ .,
      data = training_galaxy_original,
      method = "svmLinear2",
      trControl = control,
      metric = metric,
      tuneLength = 1
    )
)

system.time(
  kknn_fit_galaxy_original <-
    train(
      galaxysentiment ~ .,
      data = training_galaxy_original,
      method = "kknn",
      metric = metric,
      trControl = control,
      tuneLength = 1
    )
)

#### RESULTS FOR GALAXY ORIGINAL

c50_predictions_galaxy_original <-
  predict(c50_fit_galaxy_original, testing_galaxy_original)

postResample(c50_predictions_galaxy_original,
             testing_galaxy_original$galaxysentiment)

rf_fit_galaxy_original

svm_predictions_galaxy_original <-
  predict(svm_fit_galaxy_original, testing_galaxy_original)

postResample(svm_predictions_galaxy_original,
             testing_galaxy_original$galaxysentiment)

kknn_predictions_galaxy_original <-
  predict(kknn_fit_galaxy_original, testing_galaxy_original)

postResample(kknn_predictions_galaxy_original,
             testing_galaxy_original$galaxysentiment)

################################################################################################################
################################################################################################################
################################################################################################################

### MODELS FOR NZV DATSET

in_train_NZV_iphone <-
  createDataPartition(y = NZV_iphone$iphonesentiment,
                      p = 0.70,
                      list = FALSE)

training_NZV_iphone <- NZV_iphone[in_train_NZV_iphone, ]

testing_NZV_iphone <- NZV_iphone[-in_train_NZV_iphone, ]

system.time(
  c50_fit_NZV_iphone <-
    train(
      iphonesentiment ~ .,
      data = training_NZV_iphone,
      method = "C5.0",
      metric = metric,
      trControl = control,
      tuneLength = 1
    )
)

system.time(
  rf_fit_NZV_iphone <-
    train(
      iphonesentiment ~ .,
      data = training_NZV_iphone,
      method = "rf",
      trControl = control,
      metric = metric,
      tuneLength = 1
    )
)

system.time(
  svm_fit_NZV_iphone <-
    train(
      iphonesentiment ~ .,
      data = training_NZV_iphone,
      method = "svmLinear2",
      trControl = control,
      metric = metric,
      tuneLength = 1
    )
)

system.time(
  kknn_fit_NZV_iphone <-
    train(
      iphonesentiment ~ .,
      data = training_NZV_iphone,
      method = "kknn",
      metric = metric,
      trControl = control,
      tuneLength = 1
    )
)

#########################################################################

in_train_NZV_galaxy <-
  createDataPartition(y = NZV_galaxy$galaxysentiment,
                      p = 0.70,
                      list = FALSE)

training_NZV_galaxy <- NZV_galaxy[in_train_NZV_galaxy, ]

testing_NZV_galaxy <- NZV_galaxy[-in_train_NZV_galaxy, ]

system.time(
  c50_fit_NZV_galaxy <-
    train(
      galaxysentiment ~ .,
      data = training_NZV_galaxy,
      method = "C5.0",
      metric = metric,
      trControl = control,
      tuneLength = 1
    )
)

system.time(
  rf_fit_NZV_galaxy <-
    train(
      galaxysentiment ~ .,
      data = training_NZV_galaxy,
      method = "rf",
      trControl = control,
      metric = metric,
      tuneLength = 1
    )
)

system.time(
  svm_fit_NZV_galaxy <-
    train(
      galaxysentiment ~ .,
      data = training_NZV_galaxy,
      method = "svmLinear2",
      trControl = control,
      metric = metric,
      tuneLength = 1
    )
)

system.time(
  kknn_fit_NZV_galaxy <-
    train(
      galaxysentiment ~ .,
      data = training_NZV_galaxy,
      method = "kknn",
      metric = metric,
      trControl = control,
      tuneLength = 1
    )
)

########################

# NZV RESULTS

c50_predictions_NZV_iphone <-
  predict(c50_fit_NZV_iphone, testing_NZV_iphone)

postResample(c50_predictions_NZV_iphone,
             testing_NZV_iphone$iphonesentiment)

rf_fit_NZV_iphone

svm_predictions_NZV_iphone <-
  predict(svm_fit_NZV_iphone, testing_NZV_iphone)

postResample(svm_predictions_NZV_iphone,
             testing_NZV_iphone$iphonesentiment)

kknn_predictions_NZV_iphone <-
  predict(kknn_fit_NZV_iphone, testing_NZV_iphone)

postResample(
  kknn_predictions_NZV_iphone,
  testing_NZV_iphone$iphonesentiment)
  
  ###################################################
  
  c50_predictions_NZV_galaxy <-
    predict(c50_fit_NZV_galaxy, testing_NZV_galaxy)
  
  postResample(
    c50_predictions_NZV_galaxy,
    testing_NZV_galaxy$galaxysentiment
  )
  
  rf_fit_NZV_galaxy
  
  svm_predictions_NZV_galaxy <-
    predict(svm_fit_NZV_galaxy, testing_NZV_galaxy)
  
  postResample(
    svm_predictions_NZV_galaxy,
    testing_NZV_galaxy$galaxysentiment
  )
  
  kknn_predictions_NZV_galaxy <-
    predict(kknn_fit_NZV_galaxy, testing_NZV_galaxy)
  
  postResample(
    kknn_predictions_NZV_galaxy,
    testing_NZV_galaxy$galaxysentiment
  )
  
  ################################################################################################################
  ################################################################################################################
  ################################################################################################################
  
  ### MODELS FOR RFE DATSET
  
  in_train_RFE_iphone <-
    createDataPartition(
      y = RFE_iphone$iphonesentiment,
      p = 0.70,
      list = FALSE
    )
  
  training_RFE_iphone <-
    RFE_iphone[in_train_RFE_iphone, ]
  
  testing_RFE_iphone <-
    RFE_iphone[-in_train_RFE_iphone, ]
  
  system.time(
    c50_fit_RFE_iphone <-
      train(
        iphonesentiment ~ .,
        data = training_RFE_iphone,
        method = "C5.0",
        metric = metric,
        trControl = control,
        tuneLength = 1
      )
  )
  
  system.time(
    rf_fit_RFE_iphone <-
      train(
        iphonesentiment ~ .,
        data = training_RFE_iphone,
        method = "rf",
        trControl = control,
        metric = metric,
        tuneLength = 1
      )
  )
  
  system.time(
    svm_fit_RFE_iphone <-
      train(
        iphonesentiment ~ .,
        data = training_RFE_iphone,
        method = "svmLinear2",
        trControl = control,
        metric = metric,
        tuneLength = 1
      )
  )
  
  system.time(
    kknn_fit_RFE_iphone <-
      train(
        iphonesentiment ~ .,
        data = training_RFE_iphone,
        method = "kknn",
        metric = metric,
        trControl = control,
        tuneLength = 1
      )
  )
  
  #########################################################################
  
  in_train_RFE_galaxy <-
    createDataPartition(
      y = RFE_galaxy$galaxysentiment,
      p = 0.70,
      list = FALSE
    )
  
  training_RFE_galaxy <-
    RFE_galaxy[in_train_RFE_galaxy, ]
  
  testing_RFE_galaxy <-
    RFE_galaxy[-in_train_RFE_galaxy, ]
  
  system.time(
    c50_fit_RFE_galaxy <-
      train(
        galaxysentiment ~ .,
        data = training_RFE_galaxy,
        method = "C5.0",
        metric = metric,
        trControl = control,
        tuneLength = 1
      )
  )
  
  system.time(
    rf_fit_RFE_galaxy <-
      train(
        galaxysentiment ~ .,
        data = training_RFE_galaxy,
        method = "rf",
        trControl = control,
        metric = metric,
        tuneLength = 1
      )
  )
  
  system.time(
    svm_fit_RFE_galaxy <-
      train(
        galaxysentiment ~ .,
        data = training_RFE_galaxy,
        method = "svmLinear2",
        trControl = control,
        metric = metric,
        tuneLength = 1
      )
  )
  
  system.time(
    kknn_fit_RFE_galaxy <-
      train(
        galaxysentiment ~ .,
        data = training_RFE_galaxy,
        method = "kknn",
        metric = metric,
        trControl = control,
        tuneLength = 1
      )
  )
  
  
  
  # RFE RESULTS
  
  c50_predictions_RFE_iphone <-
    predict(c50_fit_RFE_iphone, testing_RFE_iphone)
  
  postResample(
    c50_predictions_RFE_iphone,
    testing_RFE_iphone$iphonesentiment
  )
  
  rf_fit_RFE_iphone
  
  svm_predictions_RFE_iphone <-
    predict(svm_fit_RFE_iphone, testing_RFE_iphone)
  
  postResample(
    svm_predictions_RFE_iphone,
    testing_RFE_iphone$iphonesentiment
  )
  
  kknn_predictions_RFE_iphone <-
    predict(kknn_fit_RFE_iphone, testing_RFE_iphone)
  
  postResample(
    kknn_predictions_RFE_iphone,
    testing_RFE_iphone$iphonesentiment)
    
    ###################################################
    
    c50_predictions_RFE_galaxy <-
      predict(c50_fit_RFE_galaxy, testing_RFE_galaxy)
    
    postResample(
      c50_predictions_RFE_galaxy,
      testing_RFE_galaxy$galaxysentiment
    )
    
    rf_fit_RFE_galaxy
    
    svm_predictions_RFE_galaxy <-
      predict(svm_fit_RFE_galaxy, testing_RFE_galaxy)
    
    postResample(
      svm_predictions_RFE_galaxy,
      testing_RFE_galaxy$galaxysentiment
    )
    
    kknn_predictions_RFE_galaxy <-
      predict(kknn_fit_RFE_galaxy, testing_RFE_galaxy)
    
    postResample(
      kknn_predictions_RFE_galaxy,
      testing_RFE_galaxy$galaxysentiment
    )
  
    
#### Apply Model to Data (AWS dataset)
    
    
    # unnecessary code
    
    iphone_large_matrix_data <-
      read.csv(file = "C:/Users/Lyon P. Abido/Desktop/R_stuff/iphoneLargeMatrix.csv")
    
    target <- Reduce(intersect, list(colnames(RFE_iphone), colnames(iphone_large_matrix_data)))

    cleaned_iphone_large_matrix_data <- iphone_large_matrix_data[target]

    cleaned_iphone_large_matrix_data$iphonesentiment <- 0

    cleaned_iphone_large_matrix_data$iphonesentiment <-
      as.factor(cleaned_iphone_large_matrix_data$iphonesentiment)
    
    RF_RFE_fit_iphone_large_matrix_predictions <- predict(rf_fit_RFE_iphone, newdata=cleaned_iphone_large_matrix_data)
    
    summary(RF_RFE_fit_iphone_large_matrix_predictions)
    
    
    ################################################################################
    
    galaxy_large_matrix_data <-
      read.csv("C:/Users/Lyon P. Abido/Desktop/R_stuff/galaxyLargeMatrix.csv")
    
    target <- Reduce(intersect, list(colnames(RFE_galaxy), colnames(galaxy_large_matrix_data)))
    
    cleaned_galaxy_large_matrix_data <- galaxy_large_matrix_data[target]
    
    cleaned_galaxy_large_matrix_data$galaxysentiment <- 0
    
    cleaned_galaxy_large_matrix_data$galaxysentiment <-
      as.factor(cleaned_galaxy_large_matrix_data$galaxysentiment)
    
    RF_RFE_fit_galaxy_large_matrix_predictions <- predict(rf_fit_RFE_galaxy, newdata=cleaned_galaxy_large_matrix_data)
    
    summary(RF_RFE_fit_galaxy_large_matrix_predictions)
    

    #stopCluster(cl)
    