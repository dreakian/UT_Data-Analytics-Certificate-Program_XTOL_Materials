library("readr")

df_iris <- read.csv(file = "C:/Users/cralx2k/Desktop/AAA_UT Data Analytics Certificate Program_XTOL_Materials/Course 3/R Tutorial Data Sets/iris.csv")

attributes(df_iris)

summary(df_iris)

str(df_iris)

names(df_iris)

df_iris$Species.num <- as.numeric(factor(df_iris$Species, 
                                         levels = unique(df_iris$Species)))

plot(df_iris$Sepal.Length)

hist(df_iris$Species.num)

# df_iris$Species <- NULL (drops the "Species" column from the dataframe)

qqnorm(df_iris$Sepal.Length)

qqnorm(df_iris$Petal.Length)

qqnorm(df_iris$Sepal.Length)

qqnorm(df_iris$Sepal.Width)

qqnorm(df_iris$Petal.Width)

qqnorm(df_iris$Species.num)

set.seed(405)

train_size <- round(nrow(df_iris) * 0.8)

train_size

test_size <- nrow(df_iris) - train_size

test_size

training_indices <- sample(seq_len(nrow(df_iris)), size = train_size)

train_set <- df_iris[training_indices, ]

test_set <- df_iris[-training_indices, ]

Linear_Model <- lm(formula=Petal.Width ~ Petal.Length, data=train_set)

summary(Linear_Model)

Predictions_Name <- predict(Linear_Model, test_set)

Predictions_Name

plot(Predictions_Name, test_set$Petal.Width)



