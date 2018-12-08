# R routine for random forests binary classification of poisonous mushrooms 
# kfb5@njit.edu
# 11/5/2018
# ref: 
#   http://dni-institute.in/blogs/random-forest-using-r-step-by-step-tutorial/
#   https://www.kaggle.com/uciml/mushroom-classification
#
# Load library
library(randomForest)
# Help on randomForest package and function
library(help=randomForest)
help(randomForest)
## Read data
mushroom<-read.csv(file="mushrooms.csv",header = T)
## Explore data frame
names(mushroom)
index <- sample(2,nrow(mushroom),replace=T,prob = c(0.5,0.5))
mtrain <- mushroom[index==1,]
mtest <- mushroom[index==2,]
table(mtrain$class)/nrow(mtrain)
table(mtest$class)/nrow(mtest)
class(mtrain$class)
varNames <- names(mtrain)
# Exclude class variable in training set
varNames <- varNames[!varNames %in% c("class")]
# add + sign between exploratory variables
varNames1 <- paste(varNames, collapse = "+")
# Add class variable and convert to a formula object
rf.form <- as.formula(paste("class", varNames1, sep = " ~ "))
mushroom.rf <- randomForest(rf.form,mtrain,ntree=500,importance=T)
plot(mushroom.rf)
# Variable Importance Plot
varImpPlot(mushroom.rf,sort = T,main="Variable Importance",n.var=5)
# Variable Importance Table
var.imp <- data.frame(importance(mushroom.rf,type=2))
# make row names as columns
var.imp$Variables <- row.names(var.imp)
var.imp[order(var.imp$MeanDecreaseGini,decreasing = T),]
# Predicting class variable
mtrain$predicted.class <- predict(mushroom.rf ,mtrain)
# Load Library or packages
library(e1071)
library(caret)
## Loading required package: lattice
## Loading required package: ggplot2
# Create Confusion Matrix
confusionMatrix(data=mtrain$predicted.class,reference=mtrain$class,positive='p')
# Predicting response variable
mtest$predicted.class <- predict(mushroom.rf ,mtest)
# Create Confusion Matrix
confusionMatrix(data=mtest$predicted.class,reference=mtest$class,positive='p')
#
# end of code
#