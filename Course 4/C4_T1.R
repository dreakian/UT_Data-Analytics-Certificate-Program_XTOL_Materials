library(RMariaDB)
library(dplyr)
library(lubridate)


# creates connection to MariaDB

con = dbConnect(MariaDB(), user="deepAnalytics",
                           password="Sqltask1234!",
                           dbname="dataanalytics2018",
                           host="data-analytics-2018.cbrosir2cswx.us-east-1.rds.amazonaws.com")

# shows the tables into the MariaDB object

dbListTables(con)

# dbListFields(con, "iris")

# iris_ALL <- dbGetQuery(con, "SELECT * FROM iris")

# iris_SELECT <- dbGetQuery(con, "SELECT SepalLengthCm, SepalWidthCm FROM iris")

# shows the data fields inside of the yr_2006 table

dbListFields(con, "yr_2006") # do this for 2007, 2008, 2009 and 2010

# creates dataframes which contain all the information from the tables

yr_2006 <- dbGetQuery(con, "SELECT * FROM yr_2006")
yr_2007 <- dbGetQuery(con, "SELECT * FROM yr_2007")
yr_2008 <- dbGetQuery(con, "SELECT * FROM yr_2008")
yr_2009 <- dbGetQuery(con, "SELECT * FROM yr_2009")
yr_2010 <- dbGetQuery(con, "SELECT * FROM yr_2010")

# investigating each of the tables 

str(yr_2006)
#str(yr_2007)
#str(yr_2008)
#str(yr_2009)
#str(yr_2010)

summary(yr_2006)
#summary(yr_2007)
#summary(yr_2008)
#summary(yr_2009)
#summary(yr_2010)

head(yr_2006)
#head(yr_2007)
#head(yr_2008)
#head(yr_2009)
#head(yr_2010)

tail(yr_2006)
#tail(yr_2007)
#tail(yr_2008)
#tail(yr_2009)
#tail(yr_2010)

# verifying if each table spans an entire year 

min(yr_2006$Date)
max(yr_2006$Date)

min(yr_2007$Date)
max(yr_2007$Date)

min(yr_2008$Date)
max(yr_2008$Date)

min(yr_2009$Date)
max(yr_2009$Date)

min(yr_2010$Date)
max(yr_2010$Date)

# only 2007, 2008 and 2009 span an entire year -- drop years 2006 and 2010

# creates new dataframe that will be the primary dataframe for analysis - uses the 2007, 2008 and 2009 tables

df <- bind_rows(yr_2007, yr_2008, yr_2009)

str(df)
summary(df)

min(df$Date)
max(df$Date)

# combines the Date and Time variables into a single variable
# this combined variable will later be changed to a Date-Time data type

df <- cbind(df, paste(df$Date, df$Time), stringsAsFactors=FALSE)

colnames(df)[11] <- "Date_Time"

# str(df)

# moves the newly made "Date_Time" column within the dataframe

df <- df[,c(ncol(df), 1:(ncol(df)-1))]
head(df)

# convert the "Date_Time" variable so it is no longer a "chr" data-type,
# but instead a POSIXct data-type

df$Date_Time <- as.POSIXct(df$Date_Time, "%Y/%m/%d%H:%M:%S")

attr(df$Date_Time, "tzone") <- "UTC"

str(df)

# date_range <- df$Date_Time[1569894] - df$Date_Time[1]

# date_range #1095.999 (1096) days or 3 years or 36 months or 26,280 hours or 1,578,000 minutes

# create different specific time-based attributes, for the sake of later EDA 
# centered on running calculations and aggregations on such attributes



df$Year <- year(df$Date_Time)

df$Month <- month(df$Date_Time)

df$Week <- week(df$Date_Time)

df$Day <- day(df$Date_Time)

df$Hour <- hour(df$Date_Time)

df$Minute <- minute(df$Date_Time)

str(df)

#

# EDA SECTION

# sum(is.null(df)) shows the amount of null values - 0

# duplicated(df) indicates whether or not there are duplicate data - there is not

summary(df)

power_usage_sub_1 <- sum(df$Sub_metering_1) / 1000

power_usage_sub_2 <- sum(df$Sub_metering_2) / 1000

power_usage_sub_3 <- sum(df$Sub_metering_3) / 1000

total_power_usage <- power_usage_sub_1 + power_usage_sub_2 + power_usage_sub_3

power_usage_percentage_sub_1 <- (power_usage_sub_1 / total_power_usage) * 100

power_usage_percentage_sub_2 <- (power_usage_sub_2 / total_power_usage) * 100

power_usage_percentage_sub_3 <- (power_usage_sub_3 / total_power_usage) * 100

