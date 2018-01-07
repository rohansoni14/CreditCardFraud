

#Inspired by Neil Schneider's work on Kaggle




library(pROCR)
library(microbenchmark)
library(gbm)
library(xgboost)
library(glm)

#set the seed so that the data is reproducible
set.seed(42)

#Read the data and split it
data = read.csv(""C:/Users/ROHAN/AppData/Roaming/SPB_Data/creditcard.csv")

split <- sample(2, nrow(data), replace=TRUE, prob = c(0.7,0.3))
train = data[split == 1,]
test = data[split == 2,]

#1.) GBM (Generalized Boosted Regression modelling)

#Time to train the GBM model
system.time(gbm.model <- gbm(Class ~ ., distribution = "bernoulli", data=rbind(train,test), n.trees = 500, interaction.depth = 3, n.minobsinnode = 100, shrinakge = 0.01, bag.fraction= 0.5, train.fraction = nrow(train)/(nrow(train)+nrow(test))))

#Determine best iteration based on test data
best.iter = gbm.perf(gbm.model, method="test")

#feature importance
gbm.feature.imp = summary(gbm.model, n.trees= best.iter)

