#Inspired by Neil Schneider's work on Kaggle


library(pROCR)
library(microbenchmark)
library(gbm)
library(xgboost)
library(glm)

#set the seed so that the data is reproducible
set.seed(42)

#Read the data and split it
data = read.csv("C:/Users/ROHAN/AppData/Roaming/SPB_Data/creditcard.csv", sep=",")

split <- sample(2, nrow(data), replace=TRUE, prob = c(0.7,0.3))
train = data[split == 1,]
test = data[split == 2,]

#1.) GBM (Generalized Boosted Regression modelling)

system.time(gbm.model <- gbm(Class ~ ., distribution = "bernoulli", data=rbind(train,test), n.trees = 500, interaction.depth = 3, n.minobsinnode = 100, shrinakge = 0.01, bag.fraction= 0.5, train.fraction = nrow(train)/(nrow(train)+nrow(test))))
best.iter = gbm.perf(gbm.model, method="test")

gbm.feature.imp = summary(gbm.model, n.trees= best.iter)

gbm.test = predict(gbm.model, newdata = test, n.trees = best.iter)
auc.gbm = roc(test$Class, gbm.test, plot = TRUE, col = "red")
print(auc.gbm)

#2.) XGBOOST
xgb.datatrain <- xgb.DMatrix(as.matrix(train[,colnames(train)!= "Class"]), label = train$Class)
xgb.datatest <- xgb.DMatrix(as.matrix(test[,colnames(test)!= "Class"]), label = train$Class)

xgb.bench.acc = microbenchmark(
	xgb.model.acc <- xgb.train(data = xgb.datatrain, params = list(objective = "binary:logistic", eta = 0.1, max.depth = 7, min_child_weight = 100, subsample = 1, colsample_bytree = 1, nthread = 3, eval_metric = "auc"), watchlist = list(test = xgb.datatest), nrounds = 500, early_stopping_rounds = 40, print_every_n = 20), times = 5L)
print(xgb.bench.acc)
print(xgb.model.acc$bestScore)

xgb.feature.imp = xgb.importance(model = xgb.model.acc)

xgb.test.acc = predict(xgb.model.acc
                   , newdata = as.matrix(test[, colnames(test) != "Class"])
                   , ntreelimit = xgb.model.acc$bestInd)
auc.xgb.acc = roc(test$Class, xgb.test.acc, plot = TRUE, col = "green")
print(auc.xgb.acc)

#3.) XGBOOST WITH HISTOGRAM
xgb.bench.hist = microbenchmark(
	xgb.model.hist <- xgb.train(data = xgb.datatrain, params = list(objective = "binary:logistic", eta = 0.1, max.depth = 7, min_child_weight = 100, subsample = 1, colsample_bytree = 1, nthread = 3, eval_metric = "auc", tree_method = "hist", grow_policy = "lossguide"), watchlist = list(test = xgb.datatest), nrounds = 500, early_stopping_rounds = 40, print_every_n = 20), times = 5L)
print(xgb.bench.hist)
print(xgb.model.hist$bestScore)

xgb.feature.imp = xgb.importance(model = xgb.model.hist)

xgb.test.hist = predict(xgb.model.hist
                   , newdata = as.matrix(test[, colnames(test) != "Class"])
                   , ntreelimit = xgb.model.hist$bestInd)
auc.xgb.hist = roc(test$Class, xgb.test.hist, plot = TRUE, col = "blue")
print(auc.xgb.hist)

#4.) LIGHTGBM

lgb.train = lgb.Dataset(as.matrix(train[, colnames(train) != "Class"]), label = train$Class)
lgb.test = lgb.Dataset(as.matrix(test[, colnames(test) != "Class"]), label = test$Class)

params.lgb = list(objective = "binary", metric = "auc", min_data_in_leaf = 1, min_sum_hessian_in_leaf = 100, feature_fraction = 1, bagging_fraction = 1, bagging_freq = 0)

lgb.bench = microbenchmark(
	lgb.model <- lgb.train(params = params.lgb, data = lgb.train, valids = list(test = lgb.test), learning_rate = 0.1, num_leaves = 7, num_threads = 2, nrounds = 500, early_stopping_rounds = 40, eval_freq = 20), times = 5L)
print(lgb.bench)
print(max(unlist(lgb.model$record_evals[["test"]][["auc"]][["eval"]])))

lgb.feature.imp = lgb.importance(lgb.model, percentage = TRUE)

lgb.test = predict(lgb.model, data = as.matrix(test[, colnames(test) != "Class"]), n = lgb.model$best_iter)
auc.lgb = roc(test$Class, lgb.test, plot = TRUE, col = "green")
print(auc.lgb)

#5.) LOGISTIC REGRESSION

mylogit <- glm(Class ~ . , data= train, family="binomial")
summary(mylogit)
lr.speed = microbenchmark(logit.model <- glm(Class ~ ., data=train, family="binomial"), times = 5L)

print(lr.speed)

logit.test = predict(logit.model, newdata=test)
auc.logit = roc(test$Class, logit.test, plot=TRUE, col="red")
print(auc.logit)

roc.default(response = test$Class, predictor = logit.test, plot = TRUE,     col = "red")

logitpredict <- predict(logit.model, type="response")
table(train$Class, logitpredict > 0.5)

ROCRpredlogit <- prediction(logitpredict,  train$Class)
ROCRperflogit <- performance(ROCRpredlogit, 'tpr', 'fpr')
plot(ROCRperflogit, colorize =TRUE ,text.adj=c(-0.2,1.7))
