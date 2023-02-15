setwd("C:/Users/au557539/OneDrive - Aarhus universitet/Ph.d-D37803/R_practising")

#Machine learning with the mlr package

############
#loading packages and data
library(mlr)
library(tidyverse)
data(iris)

##################################
#Exercise 5: 
#build a kNN model to classify its three species of iris(including the k hyperparameter)

#create iris data as tibble. 
iristib <- as_tibble(iris)

#inspect the variable of interest (Species)
iris$Species

#Defining the task 
iristask <- makeClassifTask (data = iristib, target = "Species")

iristask

#defining the learner
knn <- makeLearner("classif.knn", par.vals = list("k" = 2)) #par.vals = allows us to specify the numbers of k-nearest neighbors. 

#training the model
knnmodel <- train(knn, iristask)

#run the model = predict values for unobserved values
knnpred <- predict(knnmodel, newdata = iristib)

#this simple model should never be evaluated like the following simple model performance evaluation
performance(knnpred, measures = list(mmce, acc))

#tuning k to improve the model

knnparamspace <- makeParamSet(makeDiscreteParam("k", values = 1:10))
gridsearch <- makeTuneControlGrid()

####crossvalidation for tuning
#defining the cross validation
cvfortuning <- makeResampleDesc("RepCV", folds = 10, reps = 20)

#Calling the tuning function
tunedk <- tuneParams("classif.knn", task = iristask, 
                     resampling = cvfortuning, 
                     par.set = knnparamspace, control = gridsearch)

#call the tundk to get the best performing value of k (k=9)
tunedk


###################################
#Exercise 6 : 
#cross-validate thes iris kNN model using nested CV, where the outer CV is holdout with a two-thirds split. 
#=Including hyperparameter tuning in cross-validation:

#Inner CV
inner <- makeResampleDesc("CV")

#Outer hold out
outerholdout <- makeResampleDesc("holdout", split = 2/3, stratify = T)

#Create wrapper, this is basicly a learner tied to some preprocessesing step
knnwrapper <- makeTuneWrapper("classif.knn", resampling = inner,
                              par.set = knnparamspace,
                              control = gridsearch)

#run the nested CV procedure
holdoutcvtuning <- resample(knnwrapper, iristask, resampling = outer)

holdoutcvtuning #mmce = mean measurements classification error. 
