# import modules

library(readr)
library(dplyr)

# creating data frame

df <- read.csv(file = "C:/Users/cralx2k/Desktop/AAA_UT Data Analytics Certificate Program_XTOL_Materials/Course 3/R Tutorial Data Sets/cars.csv")

# exploring data

attributes(df) # displays the # of rows in the data frame

summary(df) # displays the min, max and quartiles of numeric data in the data frame 

str(df) # displays the data type of each column in the data frame

names(df) # displays each column name of the data frame

df$name.of.car # displays all of the data for the "name of car" column

df$speed.of.car # displays all of the data for the "speed of car" column

df$distance.of.car # displays all of the data for the "distance of car" column

names(df) <- c("name", "speed", "distance") #renames each of the columns in the data frame

hist(df$speed)

hist(df$distance)

plot(df$speed,df$distance)

qqnorm(df$speed)

qqnorm(df$distance)

is.na(df) # displays where NA values are in the data frame (FALSE = no NA / TRUE = NA value)

# creating and testing training sets

set.seed(123)

train_size <- round(nrow(df)*0.7) # calculates the size of the train set

test_size <- nrow(df) - train_size # calculates the size of the test set

train_size # 35 (which is 70% of the # of rows of the data frame of 50)

test_size # 15 (which is 30% of the # of rows of the data frame of 50)

# creating the train and test sets

training_indices <- sample(seq_len(nrow(df)), size = train_size)

train_set <- df[training_indices,] # 35

test_set <- df[-training_indices,] # 15

lm <- lm(formula=distance~ speed, data=train_set) # creates a Linear Regression model 
# that predicts "Distance" based on "Speed" using the "train_set" as data

summary(lm) # displays the metrics for the Linear Regression Model

# Multiple R-square = determines how well the regression line fits the data (1 = perfect fit)

# p-value = indicates how much the independent variable ("Speed") affects the
# dependent variable ("Distance") 
# (greater than 0.05 means that the independent variable has no effect)
# (less than 0.05 means that the independent variable does have an effect, 
# which means that both variables are statistically significant)

lm_prediction <- predict(lm, test_set)

lm_prediction

plot(lm_prediction, test_set$distance)
