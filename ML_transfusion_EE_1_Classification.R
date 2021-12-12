###################### Data preparation ######################
rm(list=ls())
library(tidyverse)
library(caret)
Df1<- read.csv("D:/MyWork/R/class15ML/CT_EE/Classfication/train_data_df.csv")
head(Df1) # µÃÇ¨ÊÍºdataframe ·Õè 1 training dataset
Df2<- read.csv("D:/MyWork/R/class15ML/CT_EE/Classfication/test_data_df.csv")
head(Df2) # µÃÇ¨ÊÍºdataframe ·Õè 2 testing dataset
# Null column ID and Agemo variables
Df1$ID <- NULL
Df1$HN <- NULL
Df1$Use_PRC <- NULL
Df1$Use_FFP <- NULL
Df1$T_FFP <- NULL


Df2$ID <- NULL
Df2$HN <- NULL
Df2$Use_PRC <- NULL
Df2$Use_FFP <- NULL
Df2$T_FFP <- NULL

head(Df1)
head(Df2)
# Define object of outcome (Y)
Df1$T_PRC <- as.factor(Df1$T_PRC )
Df2$T_PRC  <- as.factor(Df2$T_PRC )

# Label parameters
Df1$T_PRC<-factor(Df1$T_PRC,labels=c('no','Transfusion'))
Df1$Op<-factor(Df1$Op,labels=c('Craniotomy','Craniectomy','SOC/Retro','TSS','Burr_hole_bx','ETV_Bx'))
Df1$Tumor<-factor(Df1$Tumor,labels=c('Meningioma','Glioma','Pituitary_adenoma','Schawannoma','Metastasis','Lymphoma','Other'))

Df1$HTN<-factor(Df1$HTN,labels=c('no','yes'))
Df1$SEIZURE<-factor(Df1$SEIZURE,labels=c('no','yes'))
Df1$sex<-factor(Df1$sex,labels=c('Male','Female'))
Df1$ASA_gr<-factor(Df1$ASA_gr,labels=c('ASA1-2','ASA3-4'))


Df2$T_PRC<-factor(Df2$T_PRC,labels=c('no','Transfusion'))
Df2$Op<-factor(Df2$Op,labels=c('Craniotomy','Craniectomy','SOC/Retro','TSS','Burr_hole_bx','ETV_Bx'))
Df2$Tumor<-factor(Df2$Tumor,labels=c('Meningioma','Glioma','Pituitary_adenoma','Schawannoma','Metastasis','Lymphoma','Other'))
Df2$HTN<-factor(Df2$HTN,labels=c('no','yes'))
Df2$SEIZURE<-factor(Df2$SEIZURE,labels=c('no','yes'))
Df2$sex<-factor(Df2$sex,labels=c('Male','Female'))
Df2$ASA_gr<-factor(Df2$ASA_gr,labels=c('ASA1-2','ASA3-4'))

head(Df1) # µÃÇ¨ÊÍºdata frame ·Õè 1 training dataset
head(Df2) # µÃÇ¨ÊÍºdata frame ·Õè 2 testing dataset

################################################################
################# R script for naive bayes classifier ##################
library(klaR)
# Building the model
model.NB1 <-NaiveBayes(T_PRC~ ., data = Df1 )
# Make predictions from testing dataset
predicted.NB.class<- predict(model.NB1, Df2, type = "raw") 
predicted.NB.class$class
Df2$T_PRC

predicted_NB<-data.frame(Df2$T_PRC,predicted.NB.class$class)
predicted_NB

## save predict DT output à»ç¹file csv áÅéÇàÍÒä»ãªéËÒROC µèÍ 
write.csv(predicted_NB,"predicted_NB.csv") 

# Confusion matrix
confusionMatrix( predicted.NB.class$class, Df2$T_PRC)

################ R script for plot ROC and print AUC #################
# ROC
library(pROC)

prediction.probabilities <- predicted.NB.class$posterior[,2]
prediction.probabilities
res.roc.NB <- roc(Df2$T_PRC, prediction.probabilities)
plot.roc(res.roc.NB, print.auc = TRUE)
# ROC by ggplot2
library(ggplot2)
ggroc(res.roc.NB)
ggroc(res.roc.NB, colour = 'steelblue', size = 1) +
  ggtitle(paste0('ROC Curve(AUC=0.860)'))
######################################################################
######################################################################
######### R script for Support Vector Machine with linear function ##########
library(tidyverse)
library(caret)
# Fit model
set.seed(123)
model.svml <- train(
  T_PRC ~., data = Df1, method = "svmLinear",
  trControl = trainControl("cv", number = 10),
  tuneGrid = expand.grid(C=seq(0,2, length=20)),
  preProcess = c("center","scale")
)
# Plot model accuracy with different values of COST (C)
plot(model.svml)
# Print the best turning parameter C with maximum model accuracy
model.svml$bestTune
# Make predictions
pred.svml<- model.svml %>% predict(Df2)
pred.svml
# Confusion matrix
confusionMatrix(pred.svml, Df2$T_PRC)


# ROC plot
library(pROC)
pred.svml <- as.numeric(pred.svml)
res.roc.svml <- roc(Df2$T_PRC, pred.svml)
plot.roc(res.roc.svml, print.auc = TRUE)
# ROC using ggplot2
library(ggplot2)
ggroc(res.roc.svml)
ggroc(res.roc.svml, colour = 'steelblue', size = 1) +
  ggtitle(paste0('ROC Curve(AUC=0.691)'))

## save predict DT output à»ç¹file csv áÅéÇàÍÒä»ãªéËÒROC µèÍ 
predicted_svml<-data.frame(pred.svml, Df2$T_PRC)
predicted_svml
write.csv(predicted_svml,"predicted_svml.csv") 
######################################################################
######################################################################
####### R script for Support Vector Machine with radial basis function ########
library(tidyverse)
library(caret)
# Fit model
set.seed(123)
model.svmr <- train(
  T_PRC ~., data = Df1, method = "svmRadial",
  trControl = trainControl("cv", number = 10),
  preProcess = c("center","scale"),
  tuneLenght = 10
)
# Plot model accuracy with different values of COST (C)
plot(model.svmr)
# Print the best turning parameter C with maximum model accuracy
model.svmr$bestTune
# Make predictions
pred.svmr<- model.svmr %>% predict(Df2)
# Confusion matrix
confusionMatrix(pred.svmr, Df2$T_PRC)
# ROC plot
library(pROC)
pred.svmr <- as.numeric(pred.svmr)
res.roc.svmr <- roc(Df2$T_PRC, pred.svmr)
plot.roc(res.roc.svmr, print.auc = TRUE)
# ROC using ggplot2
library(ggplot2)
ggroc(res.roc.svmr)
ggroc(res.roc.svmr, colour = 'steelblue', size = 1) +
  ggtitle(paste0('ROC Curve(AUC=0.720)'))

## save predict DT output à»ç¹file csv áÅéÇàÍÒä»ãªéËÒROC µèÍ 
predicted_svmr<-data.frame(pred.svmr, Df2$T_PRC)
predicted_svmr
write.csv(predicted_svmr,"predicted_svmr.csv") 
######################################################################
######################################################################
################ R script for k-nearest neighbor ###################
library(tidyverse)
library(caret)
#Fit knn model
model.knn <- train(
  T_PRC ~., data = Df1, method = "knn",
  trControl = trainControl("cv", number = 10),
  preProcess = c("center","scale"),
  tuneLength = 20
)
# Plot model accuracy with different value of k
plot (model.knn)
# Print the best tuning parameter k with maximized model accuracy
model.knn$bestTune
# Make predictions
pred.knn.class<- model.knn %>% predict(Df2)
# Print confusion matrix
confusionMatrix(pred.knn.class, Df2$T_PRC)
# Plot ROC curve with AUC
library(pROC)
pred.knn.class <- as.numeric(pred.knn.class)
res.roc.knn <- roc(Df2$T_PRC, pred.knn.class)
plot.roc(res.roc.knn, print.auc = TRUE)
# Plot ROC by ggplot2
library(ggplot2)
ggroc(res.roc.knn)
ggroc(res.roc.knn, colour = 'steelblue', size = 1) +
  ggtitle(paste0('ROC Curve(AUC=0.637)'))


## save predict DT output à»ç¹file csv áÅéÇàÍÒä»ãªéËÒROC µèÍ 
predicted_knn<-data.frame(pred.knn.class, Df2$T_PRC)
predicted_knn
write.csv(predicted_knn,"predicted_knn.csv") 
##################################################################
####################################################################
###################### Decision tree ############################
library("rpart")
model.DT.rpart.big<- rpart(T_PRC~ ., data = Df1,
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
predicted.DT.class <- predict(model.DT.prune, newdata = Df2, type = "class")


# SAVE 
tiff(filename = "Pruned_DT_rpart.tif",width = 10, height = 8, units = "in",res=600,compression = "lzw")
rpart.plot(model.DT.prune, main="Pruned Classification Tree")
dev.off()
#############################
#########################################

# Print confusion matrix
confusionMatrix(predicted.DT.class, Df2$T_PRC) 


# Plot ROC
library(pROC)
predicted.DT.class<- as.numeric(predicted.DT.class)
predicted.DT.class
res.roc.DT.prune <- roc(Df2$T_PRC, predicted.DT.class)
plot.roc(res.roc.DT.prune, print.auc = TRUE)
# Plot ROC by ggplot2
library(ggplot2)
ggroc(res.roc.DT.prune)
ggroc(res.roc.DT.prune, colour = 'steelblue', size = 1) +
  ggtitle(paste0('ROC Curve(AUC=0.793)'))

## save predict DT output à»ç¹file csv áÅéÇàÍÒä»ãªéËÒROC µèÍ 
predicted_DT<-data.frame(predicted.DT.class, Df2$T_PRC)
predicted_DT
write.csv(predicted_DT,"predicted_DT.csv") 

#####################################################################3
####################### Decision tree-caret #########################
model.DT.caret.gini <- train(
  T_PRC ~., data = Df1, method = "rpart",
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

#
# SAVE 
tiff(filename = "Pruned_DT_caret.tif",width = 10, height = 8, units = "in",res=600,compression = "lzw")
rpart.plot(model.DT.caret.gini$finalModel)
dev.off()

# Make the predictions 
predicted.DT.caret.gini <-model.DT.caret.gini %>% predict(Df2)
# Print confusion matrix
confusionMatrix(predicted.DT.caret.gini ,Df2$T_PRC)
# Plot ROC with AUC
library(pROC)
predicted.DT.caret.gini<- as.numeric(predicted.DT.caret.gini)
res.roc.DT.caret.gini <- roc(Df2$T_PRC, predicted.DT.caret.gini)
plot.roc(res.roc.DT.caret.gini , print.auc = TRUE)

# Plot ROC using ggplot2
library(ggplot2)
ggroc(res.roc.DT.caret.gini)
ggroc(res.roc.DT.caret.gini, colour = 'steelblue', size = 1) +
  ggtitle(paste0('ROC Curve(AUC=0.796)'))

## save predict DT output à»ç¹file csv áÅéÇàÍÒä»ãªéËÒROC µèÍ 
predicted_DT_caret<-data.frame(predicted.DT.caret.gini , Df2$T_PRC)
predicted_DT_caret
write.csv(predicted_DT_caret,"predicted_DT_caret.csv") 

#######################################################################
#######################################################################
################# R script for random forest ######################
library(tidyverse)
library(caret)
library(randomForest)
# Fit model
set.seed(123)
model.RF <- train(
  T_PRC ~., data = Df1, method = "rf",
  trControl = trainControl("cv", number = 10),
  importance = TRUE
)
### best tuning parameter
model.RF$bestTune
### final model
model.RF$finalModel

# Make the predictions
predicted.RF.class<- predict(model.RF, Df2, type = "raw")
# Print confusion matrix
confusionMatrix(predicted.RF.class ,Df2$T_PRC) 
# Plot ROC curve with AUC
library(pROC)
predicted.RF.class<- as.numeric(predicted.RF.class)
res.roc.RF.prune <- roc(Df2$T_PRC, predicted.RF.class)
plot.roc(res.roc.DT.prune, print.auc = TRUE)
# Plot ROC using ggplot2
library(ggplot2)
ggroc(res.roc.RF.prune)
ggroc(res.roc.RF.prune, colour = 'steelblue', size = 1) +
  ggtitle(paste0('ROC Curve(AUC=0.842)'))

## save predict DT output à»ç¹file csv áÅéÇàÍÒä»ãªéËÒROC µèÍ 
predicted_rf<-data.frame(predicted.RF.class , Df2$T_PRC)
predicted_rf
write.csv(predicted_rf,"predicted_RF.csv") 
###############################################################
######################################################################
######################################################################
############ Gradient Boosting Classifier #####################
library(tidyverse)
library(caret)
library(xgboost)
# Fit model
set.seed(123)
model.GBC <- train(
  T_PRC ~., data = Df1, method = "xgbTree",
  trControl = trainControl("cv", number = 10)
)

# Make the predictions
predicted.GBC.class<- predict(model.GBC, Df2, type = "raw")
# Print confusion matrix
confusionMatrix(predicted.GBC.class ,Df2$T_PRC) 
# Plot ROC curve with AUC
library(pROC)
predicted.GBC.class<- as.numeric(predicted.GBC.class)
res.roc.GBC.prune <- roc(Df2$T_PRC, predicted.GBC.class)
plot.roc(res.roc.GBC.prune, print.auc = TRUE)
# Plot ROC using ggplot2
library(ggplot2)
ggroc(res.roc.DT.prune)
ggroc(res.roc.DT.prune, colour = 'steelblue', size = 1) +
  ggtitle(paste0('ROC Curve(AUC=0.842)'))

## save predict DT output à»ç¹file csv áÅéÇàÍÒä»ãªéËÒROC µèÍ 
predicted_GBC<-data.frame(predicted.GBC.class , Df2$T_PRC)
predicted_GBC
write.csv(predicted_GBC,"predicted_GBC.csv") 

### save predict DT output à»ç¹file csv áÅéÇàÍÒä»ãªéËÒROC µèÍ 
write.csv(predicted.DT.caret.gini,"predicted_DT_caret.csv") 
write.csv(predicted.DT.class,"predicted_DT_rpart.csv") 
write.csv(predicted.RF.class,"predicted_RF.csv") 
write.csv(pred.knn.class,"predicted_knn.csv") 
write.csv(pred.svml,"predicted_svml.csv") 
write.csv(pred.svmr,"predicted_svmr.csv") 
write.csv(Df2$T_PRC ,"Df2_T_PRC.csv") 
write.csv(predicted_GBC,"predicted_GBC.csv") 

## copy ¤èÒ predicted ä»ãÊèDf2$T_PRCROC.c
