###################### Data preparation ######################
rm(list=ls())
library(tidyverse)
library(caret)
Df1<- read.csv("D:/MyWork/R/class15ML/CT_EE/Regression/train_data_df.csv")
head(Df1) # ตรวจสอบdataframe ที่ 1 training dataset
Df2<- read.csv("D:/MyWork/R/class15ML/CT_EE/Regression/test_data_df.csv")
head(Df2) # ตรวจสอบdataframe ที่ 2 testing dataset
# Null column ID and Agemo variables
Df1$ID <- NULL
Df1$HN <- NULL
Df1$T_PRC <- NULL
Df1$T_FFP <- NULL
Df1$Use_FFP <- NULL

Df2$ID <- NULL
Df2$HN <- NULL
Df2$T_PRC <- NULL
Df2$T_FFP <- NULL
Df2$Use_FFP <- NULL


# Define object of outcome (Y)
#Df1$T_PRC <- as.factor(Df1$T_PRC )
#Df2$T_PRC  <- as.factor(Df2$T_PRC )

# Label parameters
#Df1$T_PRC<-factor(Df1$T_PRC,labels=c('no','Transfusion'))
#Df1$DISE_gr2<-factor(Df1$DISE_gr2,labels=c('Tumor','Aneurysm','CVA','TBI','Spine_tumor','Spine_trauma','Spine_infection','Spine_degen','Congenital_brain','Congential_spine','Infection','NPH'))
#Df1$DISE_gr2
#Df1$TYPE_OP2<-factor(Df1$TYPE_OP2,labels=c('Craniotomy','Craniectomy','SOC/Retro','TSS','Cranioplasty','Burr hole','Spine_inst','Spine_non_inst','Spine_congenital','EVD','Shunt','Other'))
#Df1$TYPE_OP2
#Df1$SSI<-factor(Df1$SSI,labels=c('no','SSI'))
#Df1$Emergency<-factor(Df1$Emergency,labels=c('no','Emergency'))
#Df1$SEX<-factor(Df1$SEX,labels=c('Male','Female'))
#Df1$DM<-factor(Df1$DM,labels=c('no','yes'))
#Df1$RF<-factor(Df1$RF,labels=c('no','yes'))
#Df1$WARFARIN<-factor(Df1$WARFARIN,labels=c('no','yes'))


#Df2$T_PRC<-factor(Df2$T_PRC,labels=c('no','Transfusion'))
#Df2$DISE_gr2<-factor(Df2$DISE_gr2,labels=c('Tumor','Aneurysm','CVA','TBI','Spine_tumor','Spine_trauma','Spine_infection','Spine_degen','Congenital_brain','Congential_spine','Infection','NPH'))
#Df2$TYPE_OP2<-factor(Df2$TYPE_OP2,labels=c('Craniotomy','Craniectomy','SOC/Retro','TSS','Cranioplasty','Burr hole','Spine_inst','Spine_non_inst','Spine_congenital','EVD','Shunt','Other'))
#Df2$SSI<-factor(Df2$SSI,labels=c('no','SSI'))
#Df2$Emergency<-factor(Df2$Emergency,labels=c('no','Emergency'))
#Df2$SEX<-factor(Df2$SEX,labels=c('Male','Female'))
#Df2$DM<-factor(Df2$DM,labels=c('no','yes'))
#Df2$RF<-factor(Df2$RF,labels=c('no','yes'))
#Df2$WARFARIN<-factor(Df2$WARFARIN,labels=c('no','yes'))


head(Df1) # ตรวจสอบdata frame ที่ 1 training dataset
head(Df2) # ตรวจสอบdata frame ที่ 2 testing dataset

################################################################
######################################################################
######################################################################
################ R script for k-nearest neighbor ###################
library(tidyverse)
library(caret)
#Fit knn model
set.seed(2)
model.knn <- train(
  Use_PRC ~., data = Df1, method = "knn",
  trControl = trainControl("cv", number = 10),
  preProcess = c("center","scale"),
  tuneLength = 20
)
# Plot model accuracy with different value of k
plot (model.knn)
# Print the best tuning parameter k with maximized model accuracy
model.knn$bestTune
# Make predictions
# Make predictions of regression
pred.knn.class<- model.knn %>% predict(Df2)
head(pred.knn.class)

### save predict DT output เป็นfile csv แล้วเอาไปใช้ทำ 45degree graph ต่อ
pred.knn.class
Df2$Use_PRC
result.regress.knn <-data.frame(Df2$Use_PRC,pred.knn.class)
result.regress.knn
## save predict DT output เป็นfile csv แล้วเอาไปใช้หาROC ต่อ 
write.csv(result.regress.knn,"result.regress.knn.csv") 

# Compute the prediction error RMSE '

#R-squared (R2), which is the proportion of variation in the outcome that is explained
#by the predictor variables. In multiple regression models, R2 corresponds to the squared
#correlation between the observed outcome values and the predicted values by the model.
#*** The Higher the R-squared, the better the model.

#Root Mean Squared Error (RMSE), which measures the average error performed by
#the model in predicting the outcome for an observation.  Mathematically, the RMSE is
#the square root of the mean squared error (MSE), which is the average squared difference
#between the observed actual outome values and the values predicted by the model.  So, 
#MSE = mean((observeds - predicteds)~2) and RMSE = sqrt(MSE). 
#***The lower the RMSE, the better the model.

#Residual Standard Error (RSE), also known as the model sigma, is a variant of the
#RMSE adjusted for the number of predictors in the model. The lower the RSE, the better
#the model. In practice, the difference between RMSE and RSE is very small, particularly
#for large multivariate data.

#Mean Absolute Error (MAE), like the RMSE, the MAE measures the prediction error.
#Mathematically, it is the average absolute difference between observed and predicted 
#outcomes, MAE = mean(abs(observeds - predicteds)). 
# MAE is less sensitive to outliers compared to RMSE.

# R2 higher the better 

cor.test(pred.knn.class, Df2$Use_PRC,  method = "spearman")
cor.test(pred.knn.class, Df2$Use_PRC,  method = "pearson")
##############################################################
R2(pred.knn.class, Df2$Use_PRC)
RMSE(pred.knn.class, Df2$Use_PRC)
MAE(pred.knn.class, Df2$Use_PRC)

## Scaltter plot in r
# Rbase
Df3<- read.csv("D:/MyWork/R/class15ML/CT_EE/Regression/result.regress.knn.csv")

x <- Df3$pred.knn.class
y <- Df3$Df2.Use_PRC
# Plot with main and axis titles
# Change point shape (pch = 19) and remove frame.
plot(x, y, main = "Main title",
     xlab = "X axis title", ylab = "Y axis title",
     pch = 19, frame = FALSE)
# Add regression line
plot(x, y, main = "Main title",
     xlab = "X axis title", ylab = "Y axis title",
     pch = 19, frame = FALSE)
abline(lm(y ~ x, data = mtcars), col = "blue")


#
# Add loess fit
plot(x, y, main = "Main title",
     xlab = "X axis title", ylab = "Y axis title",
     pch = 19, frame = FALSE)
lines(lowess(x, y), col = "blue")

######################################################
#http://www.sthda.com/english/wiki/scatter-plots-r-base-graphs

library("car")
scatterplot( Df2.Use_PRC~  pred.knn.class, data = Df3)

#The plot contains:


#the points
# regression line (in green)
#the smoothed conditional spread (in red dashed line)
#the non-parametric regression smooth (solid line, red)

# Suppress the smoother and frame
scatterplot( Df2.Use_PRC~  pred.knn.class, data = Df3, 
             smoother = FALSE, grid = FALSE, frame = FALSE)

scatterplot( Df2.Use_PRC~  pred.knn.class, data = Df3, 
             smoother = TRUE, grid = FALSE, frame = FALSE)

scatterplot( Df2.Use_PRC~  pred.knn.class, data = Df3, 
             smoother = FALSE, grid = TRUE, frame = FALSE)

scatterplot( Df2.Use_PRC~  pred.knn.class, data = Df3, 
             smoother = FALSE, grid = FALSE, frame = TRUE)

####GG plot
#http://www.sthda.com/english/wiki/ggplot2-scatter-plots-quick-start-guide-r-software-and-data-visualization

library(ggplot2)
# Basic scatter plot
ggplot(Df3, aes(x=pred.knn.class, y=Df2.Use_PRC)) + geom_point()
# Change the point size, and shape
ggplot(Df3, aes(x=pred.knn.class, y=Df2.Use_PRC)) +
  geom_point(size=2, shape=23)

# Change the point size
#ggplot(Df3, aes(x=pred.knn.class, y=Df2.Use_PRC)) + 
#  geom_point(aes(size=qsec))

#Add regression lines
ggplot(Df3, aes(x=pred.knn.class, y=Df2.Use_PRC)) + geom_point()
# Add the regression line
ggplot(Df3, aes(x=pred.knn.class, y=Df2.Use_PRC)) + 
  geom_point()+
  geom_smooth(method=lm)
# Remove the confidence interval
ggplot(Df3, aes(x=pred.knn.class, y=Df2.Use_PRC)) + 
  geom_point()+
  geom_smooth(method=lm, se=FALSE)
# Loess method
ggplot(Df3, aes(x=pred.knn.class, y=Df2.Use_PRC)) + 
  geom_point()+
  geom_smooth()

#Change the appearance of points and lines
# Change the point colors and shapes
# Change the line type and color
ggplot(Df3, aes(x=pred.knn.class, y=Df2.Use_PRC)) + 
  geom_point(shape=18, color="blue")+
  geom_smooth(method=lm, se=FALSE, linetype="dashed",
              color="darkred")
# Change the confidence interval fill color
ggplot(Df3, aes(x=pred.knn.class, y=Df2.Use_PRC)) + 
  geom_point(shape=18, color="blue")+
  geom_smooth(method=lm,  linetype="dashed",
              color="darkred", fill="blue")

##################################################################
####################################################################
###################### Decision tree ############################
library("rpart")
set.seed(2)
model.DT.rpart.big<- rpart(Use_PRC~ ., data = Df1,
                           control=rpart.control(minsplit=20,cp=0))
### Default split by gini
### Decision tree plot
library(rpart.plot)
rpart.plot(model.DT.rpart.big) ### rpart plot
# print cp and error
printcp(model.DT.rpart.big)
min_cp <- model.DT.rpart.big$cptable[which.min(model.DT.rpart.big$cptable[,"xerror"]),"CP"]
print(min_cp)
# min_cp=0.003827019
# Print the prune tree using minimal cp
model.DT.prune = prune(model.DT.rpart.big, cp = min_cp)
# plot pruning decision tree
rpart.plot(model.DT.prune, main="Pruned Classification Tree")
# Make predictions
# Make predictions of regression
pred.DT.class<- model.DT.prune %>% predict(Df2)
head(pred.DT.class)

### save predict DT output เป็นfile csv แล้วเอาไปใช้ทำ 45degree graph ต่อ
pred.DT.class
Df2$Use_PRC

result.regress.DT <-data.frame(Df2$Use_PRC,pred.DT.class)
result.regress.DT
## save predict DT output เป็นfile csv แล้วเอาไปทำ ต่อ 
write.csv(result.regress.DT,"result.regress.DT.csv") 

# Compute the prediction error RMSE '

#R-squared (R2), which is the proportion of variation in the outcome that is explained
#by the predictor variables. In multiple regression models, R2 corresponds to the squared
#correlation between the observed outcome values and the predicted values by the model.
#*** The Higher the R-squared, the better the model.

#Root Mean Squared Error (RMSE), which measures the average error performed by
#the model in predicting the outcome for an observation.  Mathematically, the RMSE is
#the square root of the mean squared error (MSE), which is the average squared difference
#between the observed actual outome values and the values predicted by the model.  So, 
#MSE = mean((observeds - predicteds)~2) and RMSE = sqrt(MSE). 
#***The lower the RMSE, the better the model.

#Residual Standard Error (RSE), also known as the model sigma, is a variant of the
#RMSE adjusted for the number of predictors in the model. The lower the RSE, the better
#the model. In practice, the difference between RMSE and RSE is very small, particularly
#for large multivariate data.

#Mean Absolute Error (MAE), like the RMSE, the MAE measures the prediction error.
#Mathematically, it is the average absolute difference between observed and predicted 
#outcomes, MAE = mean(abs(observeds - predicteds)). 
# MAE is less sensitive to outliers compared to RMSE.

# R2 higher the better 


cor.test(pred.DT.class, Df2$Use_PRC,  method = "spearman")
cor.test(pred.DT.class, Df2$Use_PRC,  method = "pearson")
R2(pred.DT.class, Df2$Use_PRC)
RMSE(pred.DT.class, Df2$Use_PRC)
MAE(pred.DT.class, Df2$Use_PRC)

## Scaltter plot in r
# Rbase
Df4<- read.csv("D:/MyWork/R/class15ML/CT_EE/Regression/result.regress.DT.csv")

x <- Df4$pred.DT.class
y <- Df4$Df2.Use_PRC
# Plot with main and axis titles
# Change point shape (pch = 19) and remove frame.
plot(x, y, main = "Main title",
     xlab = "X axis title", ylab = "Y axis title",
     pch = 19, frame = FALSE)
# Add regression line
plot(x, y, main = "Main title",
     xlab = "X axis title", ylab = "Y axis title",
     pch = 19, frame = FALSE)
abline(lm(y ~ x, data = mtcars), col = "blue")


#
# Add loess fit
plot(x, y, main = "Main title",
     xlab = "X axis title", ylab = "Y axis title",
     pch = 19, frame = FALSE)
lines(lowess(x, y), col = "blue")

######################################################
#http://www.sthda.com/english/wiki/scatter-plots-r-base-graphs

library("car")
scatterplot( Df2.Use_PRC~  pred.DT.class, data = Df4)

#The plot contains:


#the points
# regression line (in green)
#the smoothed conditional spread (in red dashed line)
#the non-parametric regression smooth (solid line, red)

# Suppress the smoother and frame
scatterplot( Df2.Use_PRC~  pred.DT.class, data = Df4, 
             smoother = FALSE, grid = FALSE, frame = FALSE)

scatterplot( Df2.Use_PRC~  pred.DT.class, data = Df4, 
             smoother = TRUE, grid = FALSE, frame = FALSE)

scatterplot( Df2.Use_PRC~  pred.DT.class, data = Df4, 
             smoother = FALSE, grid = TRUE, frame = FALSE)

scatterplot( Df2.Use_PRC~  pred.DT.class, data = Df4, 
             smoother = FALSE, grid = FALSE, frame = TRUE)

####GG plot
#http://www.sthda.com/english/wiki/ggplot2-scatter-plots-quick-start-guide-r-software-and-data-visualization

library(ggplot2)
# Basic scatter plot
ggplot(Df4, aes(x=pred.DT.class, y=Df2.Use_PRC)) + geom_point()
# Change the point size, and shape
ggplot(Df4, aes(x=pred.knn.class, y=Df2.Use_PRC)) +
  geom_point(size=2, shape=23)

# Change the point size
#ggplot(Df3, aes(x=pred.knn.class, y=Df2.Use_PRC)) + 
#  geom_point(aes(size=qsec))

#Add regression lines
ggplot(Df4, aes(x=pred.knn.class, y=Df2.Use_PRC)) + geom_point()
# Add the regression line
ggplot(Df4, aes(x=pred.knn.class, y=Df2.Use_PRC)) + 
  geom_point()+
  geom_smooth(method=lm)
# Remove the confidence interval
ggplot(Df4, aes(x=pred.knn.class, y=Df2.Use_PRC)) + 
  geom_point()+
  geom_smooth(method=lm, se=FALSE)
# Loess method
ggplot(Df4, aes(x=pred.knn.class, y=Df2.Use_PRC)) + 
  geom_point()+
  geom_smooth()

#Change the appearance of points and lines
# Change the point colors and shapes
# Change the line type and color
ggplot(Df4, aes(x=pred.knn.class, y=Df2.Use_PRC)) + 
  geom_point(shape=18, color="blue")+
  geom_smooth(method=lm, se=FALSE, linetype="dashed",
              color="darkred")
# Change the confidence interval fill color
ggplot(Df4, aes(x=pred.knn.class, y=Df2.Use_PRC)) + 
  geom_point(shape=18, color="blue")+
  geom_smooth(method=lm,  linetype="dashed",
              color="darkred", fill="blue")
#################################################################
# SAVE 
tiff(filename = "Pruned_DT_rpart.tif",width = 10, height = 8, units = "in",res=600,compression = "lzw")
rpart.plot(model.DT.prune, main="Pruned Classification Tree")
dev.off()
#############################
#########################################
####################################################################
#####################################################################3
####################### Decision tree-caret #########################
set.seed(2)
model.DT.caret.gini <- train(
  Use_PRC ~., data = Df1, method = "rpart",
  parms = list(split = "gini"),
  trControl = trainControl("cv", number = 5),
  tuneLength = 10
)
plot(model.DT.caret.gini)
print(model.DT.caret.gini)

summary(model.DT.caret.gini$finalModel)
model.DT.caret.gini$bestTune
# Plot final decision tree
library(rpart.plot)
rpart.plot(model.DT.caret.gini$finalModel)

####################################
# SAVE 
tiff(filename = "Pruned_DT_caret.tif",width = 10, height = 8, units = "in",res=600,compression = "lzw")
rpart.plot(model.DT.caret.gini$finalModel)
dev.off()
#####################################################

# Make the predictions 
# Make predictions of regression
predicted.DT.caret.gini<- model.DT.caret.gini %>% predict(Df2)
head(pred.DT.class)

### save predict DT output เป็นfile csv แล้วเอาไปใช้ทำ 45degree graph ต่อ
predicted.DT.caret.gini
Df2$Use_PRC

result.regress.DT.caret <-data.frame(Df2$Use_PRC,predicted.DT.caret.gini)
result.regress.DT.caret
## save predict DT output เป็นfile csv แล้วเอาไปทำ ต่อ 
write.csv(result.regress.DT.caret,"result.regress.DT.caret.csv") 

# Compute the prediction error RMSE '

#R-squared (R2), which is the proportion of variation in the outcome that is explained
#by the predictor variables. In multiple regression models, R2 corresponds to the squared
#correlation between the observed outcome values and the predicted values by the model.
#*** The Higher the R-squared, the better the model.

#Root Mean Squared Error (RMSE), which measures the average error performed by
#the model in predicting the outcome for an observation.  Mathematically, the RMSE is
#the square root of the mean squared error (MSE), which is the average squared difference
#between the observed actual outome values and the values predicted by the model.  So, 
#MSE = mean((observeds - predicteds)~2) and RMSE = sqrt(MSE). 
#***The lower the RMSE, the better the model.

#Residual Standard Error (RSE), also known as the model sigma, is a variant of the
#RMSE adjusted for the number of predictors in the model. The lower the RSE, the better
#the model. In practice, the difference between RMSE and RSE is very small, particularly
#for large multivariate data.

#Mean Absolute Error (MAE), like the RMSE, the MAE measures the prediction error.
#Mathematically, it is the average absolute difference between observed and predicted 
#outcomes, MAE = mean(abs(observeds - predicteds)). 
# MAE is less sensitive to outliers compared to RMSE.

# R2 higher the better 


cor.test(predicted.DT.caret.gini, Df2$Use_PRC,  method = "spearman")
cor.test(predicted.DT.caret.gini, Df2$Use_PRC,  method = "pearson")
##############################################################
R2(predicted.DT.caret.gini, Df2$Use_PRC)
RMSE(predicted.DT.caret.gini, Df2$Use_PRC)
MAE(predicted.DT.caret.gini, Df2$Use_PRC)

## Scaltter plot in r
# Rbase
Df5<- read.csv("D:/MyWork/R/class15ML/CT_EE/Regression/result.regress.DT.caret.csv")

x <- Df5$predicted.DT.caret.gini
y <- Df5$Df2.Use_PRC
# Plot with main and axis titles
# Change point shape (pch = 19) and remove frame.
plot(x, y, main = "Main title",
     xlab = "X axis title", ylab = "Y axis title",
     pch = 19, frame = FALSE)
# Add regression line
plot(x, y, main = "Main title",
     xlab = "X axis title", ylab = "Y axis title",
     pch = 19, frame = FALSE)
abline(lm(y ~ x, data = mtcars), col = "blue")


#
# Add loess fit
plot(x, y, main = "Main title",
     xlab = "X axis title", ylab = "Y axis title",
     pch = 19, frame = FALSE)
lines(lowess(x, y), col = "blue")

######################################################
#http://www.sthda.com/english/wiki/scatter-plots-r-base-graphs

library("car")
scatterplot( Df2.Use_PRC~  predicted.DT.caret.gini, data = Df5)

#The plot contains:


#the points
# regression line (in green)
#the smoothed conditional spread (in red dashed line)
#the non-parametric regression smooth (solid line, red)

# Suppress the smoother and frame
scatterplot( Df2.Use_PRC~  predicted.DT.caret.gini, data = Df5, 
             smoother = FALSE, grid = FALSE, frame = FALSE)

scatterplot( Df2.Use_PRC~  predicted.DT.caret.gini, data = Df5, 
             smoother = TRUE, grid = FALSE, frame = FALSE)

scatterplot( Df2.Use_PRC~  predicted.DT.caret.gini, data = Df5, 
             smoother = FALSE, grid = TRUE, frame = FALSE)

scatterplot( Df2.Use_PRC~  predicted.DT.caret.gini, data = Df5, 
             smoother = FALSE, grid = FALSE, frame = TRUE)

####GG plot
#http://www.sthda.com/english/wiki/ggplot2-scatter-plots-quick-start-guide-r-software-and-data-visualization

library(ggplot2)
# Basic scatter plot
ggplot(Df5, aes(x=predicted.DT.caret.gini, y=Df2.Use_PRC)) + geom_point()
# Change the point size, and shape
ggplot(Df5, aes(x=predicted.DT.caret.gini, y=Df2.Use_PRC)) +
  geom_point(size=2, shape=23)

# Change the point size
#ggplot(Df3, aes(x=pred.knn.class, y=Df2.Use_PRC)) + 
#  geom_point(aes(size=qsec))

#Add regression lines
ggplot(Df5, aes(x=predicted.DT.caret.gini, y=Df2.Use_PRC)) + geom_point()
# Add the regression line
ggplot(Df5, aes(x=predicted.DT.caret.gini, y=Df2.Use_PRC)) + 
  geom_point()+
  geom_smooth(method=lm)
# Remove the confidence interval
ggplot(Df5, aes(x=predicted.DT.caret.gini, y=Df2.Use_PRC)) + 
  geom_point()+
  geom_smooth(method=lm, se=FALSE)
# Loess method
ggplot(Df5, aes(x=predicted.DT.caret.gini, y=Df2.Use_PRC)) + 
  geom_point()+
  geom_smooth()

#Change the appearance of points and lines
# Change the point colors and shapes
# Change the line type and color
ggplot(Df5, aes(x=predicted.DT.caret.gini, y=Df2.Use_PRC)) + 
  geom_point(shape=18, color="blue")+
  geom_smooth(method=lm, se=FALSE, linetype="dashed",
              color="darkred")
# Change the confidence interval fill color
ggplot(Df5, aes(x=predicted.DT.caret.gini, y=Df2.Use_PRC)) + 
  geom_point(shape=18, color="blue")+
  geom_smooth(method=lm,  linetype="dashed",
              color="darkred", fill="blue")
#################################################################
# SAVE 
tiff(filename = "Pruned_DT_rpart.tif",width = 10, height = 8, units = "in",res=600,compression = "lzw")
rpart.plot(model.DT.prune, main="Pruned Classification Tree")
dev.off()
#

#######################################################################
#######################################################################
################# R script for random forest ######################
library(tidyverse)
library(caret)
library(randomForest)
# Fit model
set.seed(2)
model.RF <- train(
  Use_PRC ~., data = Df1, method = "rf",
  trControl = trainControl("cv", number = 10),
  importance = TRUE
)
### best tuning parameter
model.RF$bestTune
### final model
model.RF$finalModel

# Make the predictions 
# Make predictions of regression
predicted.RF.regress<- model.RF %>% predict(Df2)
head(predicted.RF.regress)

### save predict DT output เป็นfile csv แล้วเอาไปใช้ทำ 45degree graph ต่อ
predicted.RF.regress
Df2$Use_PRC

result.regress.rf <-data.frame(Df2$Use_PRC,predicted.RF.regress)
result.regress.rf
## save predict DT output เป็นfile csv แล้วเอาไปทำ ต่อ 
write.csv(result.regress.rf,"result.regress.rf.csv") 

# Compute the prediction error RMSE '

#R-squared (R2), which is the proportion of variation in the outcome that is explained
#by the predictor variables. In multiple regression models, R2 corresponds to the squared
#correlation between the observed outcome values and the predicted values by the model.
#*** The Higher the R-squared, the better the model.

#Root Mean Squared Error (RMSE), which measures the average error performed by
#the model in predicting the outcome for an observation.  Mathematically, the RMSE is
#the square root of the mean squared error (MSE), which is the average squared difference
#between the observed actual outome values and the values predicted by the model.  So, 
#MSE = mean((observeds - predicteds)~2) and RMSE = sqrt(MSE). 
#***The lower the RMSE, the better the model.

#Residual Standard Error (RSE), also known as the model sigma, is a variant of the
#RMSE adjusted for the number of predictors in the model. The lower the RSE, the better
#the model. In practice, the difference between RMSE and RSE is very small, particularly
#for large multivariate data.

#Mean Absolute Error (MAE), like the RMSE, the MAE measures the prediction error.
#Mathematically, it is the average absolute difference between observed and predicted 
#outcomes, MAE = mean(abs(observeds - predicteds)). 
# MAE is less sensitive to outliers compared to RMSE.

# R2 higher the better 

cor.test(predicted.RF.regress, Df2$Use_PRC,  method = "spearman")
cor.test(predicted.RF.regress, Df2$Use_PRC,  method = "pearson")
R2(predicted.RF.regress, Df2$Use_PRC)
RMSE(predicted.RF.regress, Df2$Use_PRC)
MAE(predicted.RF.regress, Df2$Use_PRC)

## Scaltter plot in r
# Rbase
Df6<- read.csv("D:/MyWork/R/class15ML/CT_EE/Regression/result.regress.rf.csv")

x <- Df6$predicted.RF.regress
y <- Df6$Df2.Use_PRC
# Plot with main and axis titles
# Change point shape (pch = 19) and remove frame.
plot(x, y, main = "Main title",
     xlab = "X axis title", ylab = "Y axis title",
     pch = 19, frame = FALSE)
# Add regression line
plot(x, y, main = "Main title",
     xlab = "X axis title", ylab = "Y axis title",
     pch = 19, frame = FALSE)
abline(lm(y ~ x, data = mtcars), col = "blue")


#
# Add loess fit
plot(x, y, main = "Main title",
     xlab = "X axis title", ylab = "Y axis title",
     pch = 19, frame = FALSE)
lines(lowess(x, y), col = "blue")

######################################################
#http://www.sthda.com/english/wiki/scatter-plots-r-base-graphs

library("car")
scatterplot( Df2.Use_PRC~  predicted.RF.regress, data = Df6)

#The plot contains:


#the points
# regression line (in green)
#the smoothed conditional spread (in red dashed line)
#the non-parametric regression smooth (solid line, red)

# Suppress the smoother and frame
scatterplot( Df2.Use_PRC~  predicted.RF.regress, data = Df6, 
             smoother = FALSE, grid = FALSE, frame = FALSE)

scatterplot( Df2.Use_PRC~  predicted.RF.regress, data = Df6, 
             smoother = TRUE, grid = FALSE, frame = FALSE)

scatterplot( Df2.Use_PRC~  predicted.RF.regress, data = Df6, 
             smoother = FALSE, grid = TRUE, frame = FALSE)

scatterplot( Df2.Use_PRC~  predicted.RF.regress, data = Df6, 
             smoother = FALSE, grid = FALSE, frame = TRUE)

####GG plot
#http://www.sthda.com/english/wiki/ggplot2-scatter-plots-quick-start-guide-r-software-and-data-visualization

library(ggplot2)
# Basic scatter plot
ggplot(Df6, aes(x=predicted.RF.regress, y=Df2.Use_PRC)) + geom_point()
# Change the point size, and shape
ggplot(Df6, aes(x=predicted.RF.regress, y=Df2.Use_PRC)) +
  geom_point(size=2, shape=23)

# Change the point size
#ggplot(Df3, aes(x=pred.knn.class, y=Df2.Use_PRC)) + 
#  geom_point(aes(size=qsec))

#Add regression lines
ggplot(Df6, aes(x=predicted.RF.regress, y=Df2.Use_PRC)) + geom_point()
# Add the regression line
ggplot(Df6, aes(x=predicted.RF.regress, y=Df2.Use_PRC)) + 
  geom_point()+
  geom_smooth(method=lm)
# Remove the confidence interval
ggplot(Df6, aes(x=predicted.RF.regress, y=Df2.Use_PRC)) + 
  geom_point()+
  geom_smooth(method=lm, se=FALSE)
# Loess method
ggplot(Df6, aes(x=predicted.RF.regress, y=Df2.Use_PRC)) + 
  geom_point()+
  geom_smooth()

#Change the appearance of points and lines
# Change the point colors and shapes
# Change the line type and color
ggplot(Df6, aes(x=predicted.RF.regress, y=Df2.Use_PRC)) + 
  geom_point(shape=18, color="blue")+
  geom_smooth(method=lm, se=FALSE, linetype="dashed",
              color="darkred")
# Change the confidence interval fill color
ggplot(Df6, aes(x=predicted.RF.regress, y=Df2.Use_PRC)) + 
  geom_point(shape=18, color="blue")+
  geom_smooth(method=lm,  linetype="dashed",
              color="darkred", fill="blue")
###################################################################
######################################################################
####################### gradient-boosting classifer #########################
set.seed(2)
model.GBC <- train(
  Use_PRC ~., data = Df1, method = "xgbTree",
  trControl = trainControl("cv", number = 3)
)


# Best tuning parameter mtry
model.GBC$bestTune
 

####################################
#####################################################

# Make the predictions 
# Make predictions of regression
predicted.GBC<- model.GBC %>% predict(Df2)
head(predicted.GBC)

### save predict DT output เป็นfile csv แล้วเอาไปใช้ทำ 45degree graph ต่อ
predicted.GBC
Df2$Use_PRC

result.regress.GBC<-data.frame(Df2$Use_PRC,predicted.GBC)
result.regress.GBC
## save predict DT output เป็นfile csv แล้วเอาไปทำ ต่อ 
write.csv(result.regress.GBC,"result.regress.GBC.csv") 

# Compute the prediction error RMSE '

#R-squared (R2), which is the proportion of variation in the outcome that is explained
#by the predictor variables. In multiple regression models, R2 corresponds to the squared
#correlation between the observed outcome values and the predicted values by the model.
#*** The Higher the R-squared, the better the model.

#Root Mean Squared Error (RMSE), which measures the average error performed by
#the model in predicting the outcome for an observation.  Mathematically, the RMSE is
#the square root of the mean squared error (MSE), which is the average squared difference
#between the observed actual outome values and the values predicted by the model.  So, 
#MSE = mean((observeds - predicteds)~2) and RMSE = sqrt(MSE). 
#***The lower the RMSE, the better the model.

#Residual Standard Error (RSE), also known as the model sigma, is a variant of the
#RMSE adjusted for the number of predictors in the model. The lower the RSE, the better
#the model. In practice, the difference between RMSE and RSE is very small, particularly
#for large multivariate data.

#Mean Absolute Error (MAE), like the RMSE, the MAE measures the prediction error.
#Mathematically, it is the average absolute difference between observed and predicted 
#outcomes, MAE = mean(abs(observeds - predicteds)). 
# MAE is less sensitive to outliers compared to RMSE.

# R2 higher the better 


cor.test(predicted.GBC, Df2$Use_PRC,  method = "spearman")
cor.test(predicted.GBC, Df2$Use_PRC,  method = "pearson")
##############################################################
R2(predicted.GBC, Df2$Use_PRC)
RMSE(predicted.GBC, Df2$Use_PRC)
MAE(predicted.GBC, Df2$Use_PRC)

## Scaltter plot in r
# Rbase
Df6<- read.csv("D:/MyWork/R/class15ML/CT_EE/Regression/result.regress.GBC.csv")

x <- Df6$predicted.GBC
y <- Df6$Df2.Use_PRC
# Plot with main and axis titles
# Change point shape (pch = 19) and remove frame.
plot(x, y, main = "Main title",
     xlab = "X axis title", ylab = "Y axis title",
     pch = 19, frame = FALSE)
# Add regression line
plot(x, y, main = "Main title",
     xlab = "X axis title", ylab = "Y axis title",
     pch = 19, frame = FALSE)
abline(lm(y ~ x, data = mtcars), col = "blue")


#
# Add loess fit
plot(x, y, main = "Main title",
     xlab = "X axis title", ylab = "Y axis title",
     pch = 19, frame = FALSE)
lines(lowess(x, y), col = "blue")

######################################################
#http://www.sthda.com/english/wiki/scatter-plots-r-base-graphs

library("car")
scatterplot( Df2.Use_PRC~  predicted.GBC, data = Df6)

#The plot contains:


#the points
# regression line (in green)
#the smoothed conditional spread (in red dashed line)
#the non-parametric regression smooth (solid line, red)

# Suppress the smoother and frame
scatterplot( Df2.Use_PRC~  predicted.GBC, data = Df6, 
             smoother = FALSE, grid = FALSE, frame = FALSE)

scatterplot( Df2.Use_PRC~  predicted.GBC, data = Df6, 
             smoother = TRUE, grid = FALSE, frame = FALSE)

scatterplot( Df2.Use_PRC~  predicted.GBC, data = Df6, 
             smoother = FALSE, grid = TRUE, frame = FALSE)

scatterplot( Df2.Use_PRC~  predicted.GBC, data = Df6, 
             smoother = FALSE, grid = FALSE, frame = TRUE)

####GG plot
#http://www.sthda.com/english/wiki/ggplot2-scatter-plots-quick-start-guide-r-software-and-data-visualization

library(ggplot2)
# Basic scatter plot
ggplot(Df6, aes(x=predicted.GBC, y=Df2.Use_PRC)) + geom_point()
# Change the point size, and shape
ggplot(Df6, aes(x=predicted.GBC, y=Df2.Use_PRC)) +
  geom_point(size=2, shape=23)

# Change the point size
#ggplot(Df3, aes(x=pred.knn.class, y=Df2.Use_PRC)) + 
#  geom_point(aes(size=qsec))

#Add regression lines
ggplot(Df6, aes(x=predicted.GBC, y=Df2.Use_PRC)) + geom_point()
# Add the regression line
ggplot(Df6, aes(x=predicted.GBC, y=Df2.Use_PRC)) + 
  geom_point()+
  geom_smooth(method=lm)
# Remove the confidence interval
ggplot(Df6, aes(x=predicted.GBC, y=Df2.Use_PRC)) + 
  geom_point()+
  geom_smooth(method=lm, se=FALSE)
# Loess method
ggplot(Df6, aes(x=predicted.GBC, y=Df2.Use_PRC)) + 
  geom_point()+
  geom_smooth()

#Change the appearance of points and lines
# Change the point colors and shapes
# Change the line type and color
ggplot(Df6, aes(x=predicted.GBC, y=Df2.Use_PRC)) + 
  geom_point(shape=18, color="blue")+
  geom_smooth(method=lm, se=FALSE, linetype="dashed",
              color="darkred")
# Change the confidence interval fill color
ggplot(Df6, aes(x=predicted.GBC, y=Df2.Use_PRC)) + 
  geom_point(shape=18, color="blue")+
  geom_smooth(method=lm,  linetype="dashed",
              color="darkred", fill="blue")
#################################################################
############################### linear regression #####
# Build the model
set.seed(2)
model.lr <- lm(Use_PRC ~., data = Df1)
# Summarize the model
summary(model.lr)
# Make predictions
#####################################################

# Make the predictions 
# Make predictions of regression
predicted.lr<- model.lr %>% predict(Df2)
head(predicted.lr)

### save predict DT output เป็นfile csv แล้วเอาไปใช้ทำ 45degree graph ต่อ
predicted.lr
Df2$Use_PRC

result.regress.lr<-data.frame(Df2$Use_PRC,predicted.lr)
result.regress.lr
## save predict DT output เป็นfile csv แล้วเอาไปทำ ต่อ 
write.csv(result.regress.lr,"result.regress.lr.csv") 

# Compute the prediction error RMSE '

#R-squared (R2), which is the proportion of variation in the outcome that is explained
#by the predictor variables. In multiple regression models, R2 corresponds to the squared
#correlation between the observed outcome values and the predicted values by the model.
#*** The Higher the R-squared, the better the model.

#Root Mean Squared Error (RMSE), which measures the average error performed by
#the model in predicting the outcome for an observation.  Mathematically, the RMSE is
#the square root of the mean squared error (MSE), which is the average squared difference
#between the observed actual outome values and the values predicted by the model.  So, 
#MSE = mean((observeds - predicteds)~2) and RMSE = sqrt(MSE). 
#***The lower the RMSE, the better the model.

#Residual Standard Error (RSE), also known as the model sigma, is a variant of the
#RMSE adjusted for the number of predictors in the model. The lower the RSE, the better
#the model. In practice, the difference between RMSE and RSE is very small, particularly
#for large multivariate data.

#Mean Absolute Error (MAE), like the RMSE, the MAE measures the prediction error.
#Mathematically, it is the average absolute difference between observed and predicted 
#outcomes, MAE = mean(abs(observeds - predicteds)). 
# MAE is less sensitive to outliers compared to RMSE.

# R2 higher the better 


cor.test(predicted.lr, Df2$Use_PRC,  method = "spearman")
cor.test(predicted.lr, Df2$Use_PRC,  method = "pearson")
##############################################################
R2(predicted.lr, Df2$Use_PRC)
RMSE(predicted.lr, Df2$Use_PRC)
MAE(predicted.lr, Df2$Use_PRC)

## Scaltter plot in r
# Rbase
Df7<- read.csv("D:/MyWork/R/class15ML/CT_EE/Regression/result.regress.lr.csv")

x <- Df7$predicted.lr
y <- Df7$Df2.Use_PRC
# Plot with main and axis titles
# Change point shape (pch = 19) and remove frame.
plot(x, y, main = "Main title",
     xlab = "X axis title", ylab = "Y axis title",
     pch = 19, frame = FALSE)
# Add regression line
plot(x, y, main = "Main title",
     xlab = "X axis title", ylab = "Y axis title",
     pch = 19, frame = FALSE)
abline(lm(y ~ x, data = mtcars), col = "blue")


#
# Add loess fit
plot(x, y, main = "Main title",
     xlab = "X axis title", ylab = "Y axis title",
     pch = 19, frame = FALSE)
lines(lowess(x, y), col = "blue")

######################################################
#http://www.sthda.com/english/wiki/scatter-plots-r-base-graphs

library("car")
scatterplot( Df2.Use_PRC~  predicted.lr, data = Df7)

#The plot contains:


#the points
# regression line (in green)
#the smoothed conditional spread (in red dashed line)
#the non-parametric regression smooth (solid line, red)

# Suppress the smoother and frame
scatterplot( Df2.Use_PRC~  predicted.lr, data = Df7, 
             smoother = FALSE, grid = FALSE, frame = FALSE)

scatterplot( Df2.Use_PRC~  predicted.lr, data = Df7, 
             smoother = TRUE, grid = FALSE, frame = FALSE)

scatterplot( Df2.Use_PRC~  predicted.lr, data = Df7, 
             smoother = FALSE, grid = TRUE, frame = FALSE)

scatterplot( Df2.Use_PRC~  predicted.lr, data = Df7, 
             smoother = FALSE, grid = FALSE, frame = TRUE)

####GG plot
#http://www.sthda.com/english/wiki/ggplot2-scatter-plots-quick-start-guide-r-software-and-data-visualization

library(ggplot2)
# Basic scatter plot
ggplot(Df7, aes(x=predicted.lr, y=Df2.Use_PRC)) + geom_point()
# Change the point size, and shape
ggplot(Df7, aes(x=predicted.lr, y=Df2.Use_PRC)) +
  geom_point(size=2, shape=23)

# Change the point size
#ggplot(Df3, aes(x=pred.knn.class, y=Df2.Use_PRC)) + 
#  geom_point(aes(size=qsec))

#Add regression lines
ggplot(Df7, aes(x=predicted.lr, y=Df2.Use_PRC)) + geom_point()
# Add the regression line
ggplot(Df7, aes(x=predicted.lr, y=Df2.Use_PRC)) + 
  geom_point()+
  geom_smooth(method=lm)
# Remove the confidence interval
ggplot(Df7, aes(x=predicted.lr, y=Df2.Use_PRC)) + 
  geom_point()+
  geom_smooth(method=lm, se=FALSE)
# Loess method
ggplot(Df7, aes(x=predicted.lr, y=Df2.Use_PRC)) + 
  geom_point()+
  geom_smooth()

#Change the appearance of points and lines
# Change the point colors and shapes
# Change the line type and color
ggplot(Df7, aes(x=predicted.lr, y=Df2.Use_PRC)) + 
  geom_point(shape=18, color="blue")+
  geom_smooth(method=lm, se=FALSE, linetype="dashed",
              color="darkred")
# Change the confidence interval fill color
ggplot(Df7, aes(x=predicted.lr, y=Df2.Use_PRC)) + 
  geom_point(shape=18, color="blue")+
  geom_smooth(method=lm,  linetype="dashed",
              color="darkred", fill="blue")
#################################################################



#
#################################################################
###############################################################
### save predict DT output เป็นfile csv แล้วเอาไปใช้หาROC ต่อ 
write.csv(predicted.DT.caret.gini,"predicted_DT_caret.csv") 
write.csv(predicted.DT.class,"predicted_DT_rpart.csv") 
write.csv(predicted.RF.class,"predicted_RF.csv") 
write.csv(pred.knn.class,"predicted_knn.csv") 
write.csv(pred.svml,"predicted_svml.csv") 
write.csv(pred.svmr,"predicted_svmr.csv") 
write.csv(Df2$T_PRC ,"Df2_T_PRC.csv") 

## copy ค่า predicted ไปใส่Df2$T_PRCROC.csv
######################################################################
##########################################################
####### R script for artificial neural network using neuralnet package ########
## Fit ann model
library(neuralnet)
set.seed(2)
model.ann.logistic<- neuralnet(Use_PRC~ DISE_gr2+TYPE_OP2+SSI+Emergency+ASA_Class+SEX
                               +RF+WARFARIN+Age+ebl+Pre_WBC+Pre_Hb+Pre_Hct+Pre_PLT
                               +NL_ratio+Pre_rPTT+Pre_PT_INR,
                               data=Df1,
                               hidden = 5 ,
                               act.fct = "logistic",
                               linear.output = T )
# Plot ann model 
plot(model.ann.logistic) 
# Make predictions
predicted.ann <- predict(model.ann.logistic, newdata = Df2, type = "class")
predicted.ann

#the predicted results are compared to the actual results:
results.ann <- data.frame(actual = Df2$T_PRC, prediction = predicted.ann )
results.ann

### Regroup of predicted results 
results.ann$predicted.bi[results.ann$prediction.2< 0.5] <- "Negative"
results.ann$predicted.bi[results.ann$prediction.2>= 0.5] <- "Positive"
results.ann$predicted.bi <-factor(results.ann$predicted.bi)
# Plot confusion matrix
confusionMatrix(results.ann$predicted.bi ,Df2$T_PRC) 
# Plot ROC curve with AUC
library(pROC)
results.ann$predicted.bi<- as.numeric(results.ann$predicted.bi)
res.roc.ann <- roc(Df2$T_PRC, results.ann$predicted.bi)
plot.roc(res.roc.ann, print.auc = TRUE)
# Plot ROC curve using ggplot2
library(ggplot2)
ggroc(res.roc.ann)
ggroc(res.roc.ann, colour = 'steelblue', size = 1) +
  ggtitle(paste0('ROC Curve(AUC=0.662)'))

##############################################################
##############################################################
################ Plot ann model using devtools ####################
library(devtools)
# Connect internet
source_url('https://gist.github.com/fawda123/7471137/raw/cd6e6a0b0bdb4e065c597e52165e5ac887f5fe95/nnet_plot_update.r')
# plot ann model
plot.nnet(model.ann.logistic)
