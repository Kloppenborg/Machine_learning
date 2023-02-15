#this script is based on the book "Machine learning with R, the tidyverse & mlr", pp. 215

#clear the environment
rm(list=ls())

#Load libraries
library(tidyverse)
library(mlr) #mlr3 is the new package, and mlr will not have any developments

#Read the ozone data from the mlbench package
data(Ozone, package = "mlbench")

#Convert data into a tibble
ozoneTib <- as_tibble(Ozone)

#Convert the variablenames into meanigful names
names(ozoneTib) <- c("month","date","day","ozone","press_height",
                     "wind", "humid", "temp_sand","temp_monte",
                     "inv_height","press_grad","inv_temp", "visib")

#Convert all variables into numeric variables & remove observations with NA in the ozone variable
ozone_clean <- mutate_all(ozoneTib, as.numeric) %>% #OBS mutate_all is not coming after a pipe
  filter(is.na(ozone)==FALSE) #Removes all NA´s in the Ozone variable

#gather data, so we can plot each variable on separate facets
ozoneuntidy <- gather(ozone_clean, key="Variable", 
                      value= "Value", 
                      -ozone) #This last line prevent the ozone variable from being gatherede with the others

#plotting the data
ggplot(ozoneuntidy,
       aes(x=Value,
           y=ozone))+
  facet_wrap(~Variable, scale = "free_x")+
  geom_point()+
  geom_smooth()+
  geom_smooth(mehod="lm", col="red")+
  theme_bw()

###Using rpart to impute missing values####
#Defining what algorithm to impute missing values
imputemethod <- imputeLearner("regr.rpart")

#Use impute funktion to create imputed dataset, to which the first argument is the data. 
ozoneImp <- impute(as.data.frame(ozone_clean),
                   classes = list(numeric = imputemethod))

#Defining our task and learner
ozoneTask <- makeRegrTask(data=ozoneImp$data, target = "ozone")

lin <- makeLearner("regr.lm")




######Filter method for feature selection#####
#using a filter method for feature selection
filterVals <- generateFilterValuesData(ozoneTask,
                                       method = "linear.correlation") #to see other methods us listFilterMethods()

##Get a table of predictors with their Pearsons correlation coefficients
filterVals$data

#Plotting the information
plotFilterValues(filterVals)+
  theme_bw()



#Manually selecting which foeatures to drop; instead of this, we want to wrap together our learner.
#The abs argument allows us to specify the absolute number of best predictors to retain
#ozoneFiltTask <- filterFeatures(ozoneTask, fval = filterVals, abs=6)

#The per argument allows us to specify a top percentage of best predictors to retain
#ozoneFiltTask <- filterFeatures(ozoneTask, fval = filterVals, per=0.25)

#The threshold argument allows us to specify a value of our filtering metric, that a predictor must exceed in order to be retain.
#ozoneFiltTask <- filterFeatures(ozoneTask, fcal=filterVals, threshold = 0.2)



#####Creating a filter wrapper
filterwrapper = ?makeFilterWrapper(learner = lin,
                                  fw.method = "linear.correlation")



####Tuning the number of predictors to retain
#Define the hyperparameter space & define fw.abs as the min and max nr. of features to retain
set.seed(1209)
lmParamSpace <- makeParamSet(
  makeIntegerParam("fw.abs", lower = 1, upper = 12)
)

#Define the grid search ; This will try every value of our hyperparameter
gridSearch <- makeTuneControlGrid()

#Make an ordinary 10-fold cross-validation
kFold <- makeResampleDesc("CV", iters = 10)

#First argument in tuneParams() is our wrapped learner, the our task(ozoneTask), 
#cross validation(resampling), hyperparameter(par.set) space and search procedure(control)
tunedFeats <- tuneParams(filterwrapper, 
                         task = ozoneTask, 
                         resampling = kFold,
                         par.set = lmParamSpace,
                         control = gridSearch)

#See the results
tunedFeats #mse.test.mean=20.xxx = mean square error. 

#Training the model with filtered features
filteredTask <- filterFeatures(ozoneTask, fval = filterVals,
                               abs = unlist(tunedFeats$x))

filteredModel <- train(lin, filteredTask)

#Get data to interpret the model
filterModelData <- getLearnerModel(filteredModel)

summary(filterModelData)
