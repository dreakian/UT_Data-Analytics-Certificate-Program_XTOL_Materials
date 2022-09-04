library("arules")
library("arulesViz")

df <- read.transactions(file="C:/Users/cralx2k/Desktop/AAA_UT Data Analytics Certificate Program_XTOL_Materials/Course 3/ElectronidexTransactions2017.csv", format="basket", sep=",", rm.duplicates=TRUE)

summary(df)

# inspect(df[0:10])

# LIST(df)

itemLabels(df)

itemFrequencyPlot(df, topN=20, type="absolute")

 image(df[0:100])

rules<- apriori(df, parameter=list(supp=0.01, conf=0.40, minlen=3, maxlen=5))

inspect(rules)

summary(rules)

# inspect(rules[0:5]) # looks at the first 5 rules

inspect(sort(rules, by="support"))

inspect(sort(rules, by="confidence"))

inspect(sort(rules, by="lift"))

is.redundant(rules)

hp_laptop_rules <- subset(rules, items %in% "HP Laptop")

summary(hp_laptop_rules)

inspect(hp_laptop_rules)

imac_rules <- subset(rules, items %in% "iMac")

summary(imac_rules)

inspect(imac_rules)

# sum(is.redundant(rules)) # calculates the sum of redundant rules in the data

plot(rules)

plot(rules, method="graph", control=list(type="items"))

plot(rules, method="graph", engine="htmlwidget")

subrules <- head(rules, n=15, by="support")

# subset of the larger rules (48 rules) -- this only takes 10 rules

plot(subrules, method="graph", engine="htmlwidget")

# plot(subrules, method="paracoord")
