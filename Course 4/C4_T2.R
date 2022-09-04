library(RMariaDB)
library(dplyr)
library(lubridate)
library(plotly)
library(ggplot2)
library(ggfortify)
library(forecast)


# creates connection to MariaDB

con = dbConnect(MariaDB(), user="deepAnalytics",
                password="Sqltask1234!",
                dbname="dataanalytics2018",
                host="data-analytics-2018.cbrosir2cswx.us-east-1.rds.amazonaws.com")

# shows the tables into the MariaDB object

dbListTables(con)

# shows the data fields inside of the yr_2006 table

dbListFields(con, "yr_2006") # do this for 2007, 2008, 2009 and 2010

# creates dataframes which contain all the information from the tables

yr_2006 <- dbGetQuery(con, "SELECT * FROM yr_2006")
yr_2007 <- dbGetQuery(con, "SELECT * FROM yr_2007")
yr_2008 <- dbGetQuery(con, "SELECT * FROM yr_2008")
yr_2009 <- dbGetQuery(con, "SELECT * FROM yr_2009")
yr_2010 <- dbGetQuery(con, "SELECT * FROM yr_2010")

# investigating each of the tables 

#str(yr_2006)
#str(yr_2007)
#str(yr_2008)
#str(yr_2009)
#str(yr_2010)

#summary(yr_2006)
#summary(yr_2007)
#summary(yr_2008)
#summary(yr_2009)
#summary(yr_2010)

#head(yr_2006)
#head(yr_2007)
#head(yr_2008)
#head(yr_2009)
#head(yr_2010)

#tail(yr_2006)
#tail(yr_2007)
#tail(yr_2008)
#tail(yr_2009)
#tail(yr_2010)

# verifying if each table spans an entire year 

# min(yr_2006$Date)
# max(yr_2006$Date)
# 
# min(yr_2007$Date)
# max(yr_2007$Date)
# 
# min(yr_2008$Date)
# max(yr_2008$Date)
# 
# min(yr_2009$Date)
# max(yr_2009$Date)
# 
# min(yr_2010$Date)
# max(yr_2010$Date)

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




power_usage_sub_1 <- sum(df$Sub_metering_1) / 1000

power_usage_sub_2 <- sum(df$Sub_metering_2) / 1000

power_usage_sub_3 <- sum(df$Sub_metering_3) / 1000

total_power_usage <- power_usage_sub_1 + power_usage_sub_2 + power_usage_sub_3

power_usage_percentage_sub_1 <- (power_usage_sub_1 / total_power_usage) * 100

power_usage_percentage_sub_2 <- (power_usage_sub_2 / total_power_usage) * 100

power_usage_percentage_sub_3 <- (power_usage_sub_3 / total_power_usage) * 100

# Visualizations 

# visualizing a week of power consumption

# creates a filter 

### TEST FILTER TO VISUALIZE WHY GRANULARITY IS IMPORTANT ###

test_filter_one <- filter(df, Year == 2009 & Week == 5)

plot(test_filter_one$Sub_metering_1)

# creates an even more specific filter

### TEST FILTER TO VISUALIZE WHY GRANULARITY IS IMPORTANT ###

test_filter_two <- filter(df, Year == 2009 & Month == 5 & Day == 3)

# plots based on the previously created filter, only showing power consumption
# for Sub_meter_1, which isn't very insightful

plot_ly(test_filter_two, x=~test_filter_two$Date_Time, y=~test_filter_two$Sub_metering_1, 
        type="scatter", mode="lines")

# creates labels and a more inclusive filter, but it's still very grainy

plot_ly(test_filter_two, x=~test_filter_two$Date_Time, y=~test_filter_two$Sub_metering_1,
        name="Kitchen", type="scatter", mode="lines") %>%
  add_trace(y=~test_filter_two$Sub_metering_2, name="Laundry Room",
            mode="lines") %>%
  add_trace(y=~test_filter_two$Sub_metering_3, name="Water Heater & AC",
            mode="lines") %>%
  layout(title="Power Consumption May 3rd, 2009",
         xaxis=list(title="Time"),
         yaxis=list(title="Power (watt-hours)"))

# creating new visualization that is even less grainy (less points of data being plotted)\

## Sub-set of PC of May 3rd 2009 - 10 Minute frequency

filter_one <- filter(df, Year == 2009 & Month == 7 & Day == 15 & 
                       (Minute == 0 | Minute == 10 |
                          Minute == 20 | Minute == 30 |
                          Minute == 40 | Minute == 50))

# creating the new, even less grainy visualization based on 10 minute frequency

### SUMMER ### BY DAY

plot_ly(filter_one, x=~filter_one$Date_Time, y=~filter_one$Sub_metering_1,
        name="Kitchen", type="scatter", mode="lines") %>%
  add_trace(y=~filter_one$Sub_metering_2, name="Laundry Room",
            mode="lines") %>%
  add_trace(y=~filter_one$Sub_metering_3, name="Water Heater & AC",
            mode="lines") %>%
  layout(title="Power Consumption of July 15th, 2009",
         xaxis=list(title="Time"),
         yaxis=list(title="Power (watt-hours)"))

# creating visualization based on 10 minute frequency

### WINTER ### BY WEEK

filter_two <- filter(df, Year == 2007 & Week == 1 &
                       (Minute == 0 | Minute == 10 |
                          Minute == 20 | Minute == 30 |
                          Minute == 40 | Minute == 50))

# creating the new, even less grainy visualization based on 10 minute frequency

plot_ly(filter_two, x=~filter_two$Date_Time, y=~filter_two$Sub_metering_1,
        name="Kitchen", type="scatter", mode="lines") %>%
  add_trace(y=~filter_two$Sub_metering_2, name="Laundry Room",
            mode="lines") %>%
  add_trace(y=~filter_two$Sub_metering_3, name="Water Heater & AC",
            mode="lines") %>%
  layout(title="Power Consumption of the First Week of January 2007",
         xaxis=list(title="Time"),
         yaxis=list(title="Power (watt-hours)"))

#### PERIOD of TIME ### SPRING -- 3 months

# filter_three <- filter(df, Year == 2008 & Day == 5 & 
#                          (Month == 1 | Month == 2 |
#                           Month == 3 | Month == 4 |
#                           Month == 5 | Month == 6 |
#                           Month == 7 | Month == 8 |
#                           Month == 9 | Month == 10 |
#                           Month == 11 | Month == 12))

filter_three <- filter(df, Year == 2008 & Month == 12 & (Day == 29 | Day == 30 | Day == 31)  & (Minute == 0 | Minute == 10 | Minute == 20 | Minute == 30 | Minute == 40 | Minute == 50))

plot_ly(filter_three, x=~filter_three$Date_Time, y=~filter_three$Sub_metering_1,
        name="Kitchen", type="scatter", mode="lines") %>%
  add_trace(y=~filter_three$Sub_metering_2, name="Laundry Room",
            mode="lines") %>%
  add_trace(y=~filter_three$Sub_metering_3, name="Water Heater & AC",
            mode="lines") %>%
  layout(title="Power Consumption for December 29-31 2008",
         xaxis=list(title="Time"),
         yaxis=list(title="Power (watt-hours)"))



#########################################

# Pie Charts were created using the link below as a guide. #

# !!!!!!!!!!!! https://www.statmethods.net/graphs/pie.html !!!!!!!!!!!!!!!!!!!!!! # 


#### PIE CHART FOR TOTAL POWER CONSUMPTION (NO FILTERING/AGGREGATION) ###


# Pie Chart with Percentages
slices <- c(power_usage_sub_1, power_usage_sub_2, power_usage_sub_3)
lbls <- c("Kitchen", "Laundry Room", "Water Heater & AC")
pct <- round(slices/sum(slices)*100)
lbls <- paste(lbls, pct) # add percents to labels
lbls <- paste(lbls,"%",sep="") # ad % to labels
pie(slices,labels = lbls, col=rainbow(length(lbls)),
    main="Pie Chart of Total Power Consumption")

##################################################

######## 2007 ###########

filter_2007 <- filter(df, Year == 2007)

SM_1_2007 <- sum(filter_2007$Sub_metering_1)

SM_2_2007 <- sum(filter_2007$Sub_metering_2)

SM_3_2007 <- sum(filter_2007$Sub_metering_3)

slices <- c(SM_1_2007, SM_2_2007, SM_3_2007)
lbls <- c("Kitchen", "Laundry Room", "Water Heater & AC")
pct <- round(slices/sum(slices)*100)
lbls <- paste(lbls, pct) # add percents to labels
lbls <- paste(lbls,"%",sep="") # ad % to labels
pie(slices,labels = lbls, col=rainbow(length(lbls)),
    main="Pie Chart of Total Power Consumption for 2007")

######################################################

######## 2008 ###########

filter_2008 <- filter(df, Year == 2008)

SM_1_2008 <- sum(filter_2008$Sub_metering_1)

SM_2_2008 <- sum(filter_2008$Sub_metering_2)

SM_3_2008 <- sum(filter_2008$Sub_metering_3)

slices <- c(SM_1_2008, SM_2_2008, SM_3_2008)
lbls <- c("Kitchen", "Laundry Room", "Water Heater & AC")
pct <- round(slices/sum(slices)*100)
lbls <- paste(lbls, pct) # add percents to labels
lbls <- paste(lbls,"%",sep="") # ad % to labels
pie(slices,labels = lbls, col=rainbow(length(lbls)),
    main="Pie Chart of Total Power Consumption for 2008")

###########################################################

######## 2009 ###########

filter_2009 <- filter(df, Year == 2009)

SM_1_2009 <- sum(filter_2009$Sub_metering_1)

SM_2_2009 <- sum(filter_2009$Sub_metering_2)

SM_3_2009 <- sum(filter_2009$Sub_metering_3)

slices <- c(SM_1_2009, SM_2_2009, SM_3_2009)
lbls <- c("Kitchen", "Laundry Room", "Water Heater & AC")
pct <- round(slices/sum(slices)*100)
lbls <- paste(lbls, pct) # add percents to labels
lbls <- paste(lbls,"%",sep="") # ad % to labels
pie(slices,labels = lbls, col=rainbow(length(lbls)),
    main="Pie Chart of Total Power Consumption for 2009")


######################################################################


#########   FOR JUNE 19th, 2008    ############


filter_2008_6_19 <- filter(df, Year == 2008 & Month == 6 & Day == 16)

SM_1_2008_6_19 <- sum(filter_2008_6_19$Sub_metering_1) 

SM_2_2008_6_19 <- sum(filter_2008_6_19$Sub_metering_2) 

SM_3_2008_6_19 <- sum(filter_2008_6_19$Sub_metering_3) 

slices <- c(SM_1_2008_6_19, SM_2_2008_6_19, SM_3_2008_6_19)
lbls <- c("Kitchen", "Laundry Room", "Water Heater & AC")
pct <- round(slices/sum(slices)*100)
lbls <- paste(lbls, pct) # add percents to labels
lbls <- paste(lbls,"%",sep="") # ad % to labels
pie(slices,labels = lbls, col=rainbow(length(lbls)),
    main="Pie Chart of Total Power Consumption for June 19th, 2008")


#################################################################################


## time series visualization for SM 3

filter_ts_SM3 <- filter(df, Day == 2 & Hour == 20 & Minute == 1) # One observation per week on Mondays at 8:00pm for 2007, 2008, and 2009

ts_SM3 <- ts(filter_ts_SM3$Sub_metering_3, frequency=7, start=c(2007,1))

autoplot(ts_SM3, ts.colour="black", xlab="Time", ylab="Watt Hours", main="Time Series Visualization for Sub-meter 3")


# plot.ts(ts_SM3_all_years_weekly) --- another way of plotting time-series data


## time series visualizations for SM 1

filter_ts_SM1 <- filter(df, Day == 3 & Hour == 12 & Minute == 1) # One observation per week on Tuesdays at 12:00pm for 2007, 2008, 2009

ts_SM1 <- ts(filter_ts_SM1$Sub_metering_1, frequency=7, start=c(2007, 1))

autoplot(ts_SM1, ts.color="black", xlab="Time", ylab="Watt Hours", main="Time Series Visualization for Sub-meter 1")


## time series visualizations for SM 2

filter_ts_SM2 <- filter(df, Day == 6 & Hour == 5 & Minute == 1)

ts_SM2 <- ts(filter_ts_SM2$Sub_metering_2, frequency=7, start=c(2007, 1))

autoplot(ts_SM2, ts.color="black", xlab="Time", ylab="Watt Hours", main="Time Series Visualization for Sub-meter 2")                    


#### FORECASTING ##########


# SM 3 section

fit_SM3 <- tslm(ts_SM3 ~ trend + season)

summary(fit_SM3)

forecast_fit_SM3 <- forecast(fit_SM3, h=20, level=c(80, 90))

plot(forecast_fit_SM3, ylim=c(0, 20), ylab="Watt Hours", xlab="Time", main="Forecast Visualization for Sub-meter 3")



# SM 1 section

# fit_SM1 <- tslm(ts_SM1_quarterly ~ trend + season)
# 
# summary(fit_SM1)
# 
# forecast_fit_SM1 <- forecast(fit_SM1, h=20)
# 
# plot(forecast_fit_SM1)

fit_SM1 <- tslm(ts_SM1 ~ trend + season)

summary(fit_SM1)

forecast_fit_SM1 <- forecast(fit_SM1, h=20, level=c(80, 90))

plot(forecast_fit_SM1, ylim=c(0,5), ylab="Watt Hours", xlab="Time", main="Forecast Visualization for Sub-meter 1")

# Sm 2 section

fit_SM2 <- tslm(ts_SM2 ~ trend + season)

summary(fit_SM2)

forecast_fit_SM2 <- forecast(fit_SM2, h=20, level=c(80, 90))

plot(forecast_fit_SM2, ylim=c(-1.15, 2.0), ylab="Watt Hours", xlab="Time", main="Forecast Visualization for Sub-meter 2")   

########## DECOMPOSING THE TIME SERIES OBJECTS


## SM3 decomposition

components_ts_SM3 <- decompose(ts_SM3)

plot(components_ts_SM3)

summary(components_ts_SM3$seasonal)

summary(components_ts_SM3$trend)

## SM1 decomposition

components_ts_SM1 <- decompose(ts_SM1)

plot(components_ts_SM1)

summary(components_ts_SM1$seasonal)

summary(components_ts_SM1$trend)


## SM2 decomposition

components_ts_SM2 <- decompose(ts_SM2)

plot(components_ts_SM2)

summary(components_ts_SM2$seasonal)

summary(components_ts_SM2$trend)




######### Holt-Winters Forecasting 


# removing seasonality + creating Holt-Winters forecast for SM3

ts_SM3_adjusted <- ts_SM3 - components_ts_SM3$seasonal

autoplot(ts_SM3_adjusted)

plot(decompose(ts_SM3_adjusted)) 

ts_SM3_HW <- HoltWinters(ts_SM3_adjusted, beta=FALSE, gamma=FALSE)

plot(ts_SM3_HW, ylim=c(0, 25))

ts_SM3_HW_forecast <- forecast(ts_SM3_HW, h=25)

plot(ts_SM3_HW_forecast, ylim=c(-3, 25), ylab="Watt-Hours", xlab="Time - Sub-meter 3")

ts_sm3_HW_forecast_conf <- forecast(ts_SM3_HW, h=25, level=c(10,25)) 

plot(ts_sm3_HW_forecast_conf, ylim=c(0, 20), ylab="Watt-Hours", xlab="Time - Sub-meter 3", start(2010))




# removing seasonality + creating Holt-Winters forecast for SM1

ts_SM1_adjusted <- ts_SM1 - components_ts_SM1$seasonal

autoplot(ts_SM1_adjusted)

plot(decompose(ts_SM1_adjusted)) 

ts_SM1_HW <- HoltWinters(ts_SM1_adjusted, beta=FALSE, gamma=FALSE)

plot(ts_SM1_HW, ylim=c(-0.5, 1.5))

ts_SM1_HW_forecast <- forecast(ts_SM1_HW, h=25)

plot(ts_SM1_HW_forecast, ylim=c(-0.5, 3), ylab="Watt-Hours", xlab="Time - Sub-meter 1")

ts_sm1_HW_forecast_conf <- forecast(ts_SM1_HW, h=25, level=c(10,25)) 

plot(ts_sm1_HW_forecast_conf, ylim=c(0, 3), ylab="Watt-Hours", xlab="Time - Sub-meter 1", start(2010))





# removing seasonality + creating Holt-Winters forecast for SM2

ts_SM2_adjusted <- ts_SM2 - components_ts_SM2$seasonal

autoplot(ts_SM2_adjusted)

plot(decompose(ts_SM2_adjusted)) 

ts_SM2_HW <- HoltWinters(ts_SM2_adjusted, beta=FALSE, gamma=FALSE)

plot(ts_SM2_HW, ylim=c(-0.6, 1.4))

ts_SM2_HW_forecast <- forecast(ts_SM2_HW, h=25)

plot(ts_SM2_HW_forecast, ylim=c(-0.7, 1.8), ylab="Watt-Hours", xlab="Time - Sub-meter 2")

ts_sm2_HW_forecast_conf <- forecast(ts_SM2_HW, h=25, level=c(10,25)) 

plot(ts_sm2_HW_forecast_conf, ylim=c(0, 0.5), ylab="Watt-Hours", xlab="Time - Sub-meter 2", start(2010))
