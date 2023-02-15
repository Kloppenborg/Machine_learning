library(tidyverse)
library(mlr)

#Read data
data(titanic_train, package ="titanic")

#Convert data to a tibble
titanicTib <- as_tibble(titanic_train)

#Create a vector to define which variables to be converted to a factor
fctrs <- c("Survived", "Sex", "Pclass")

titanic_clean <- titanicTib %>% 
  mutate(across(.cols = fctrs, .fns= factor), #Creates a factor on the former defined colums. 
         Famsize = SibSp + Parch) %>% 
  select(Survived, Pclass, Sex, Famsize, Age, Fare)

#Create a long format DF to plot data
titanicUntidy <- gather(titanic_clean, key = "variable", value ="value", 
                        -Survived)
#Warning message: attributes are not identical across measure variables; they will be dropped 

#Create a violin subplots for each continuous variable
titanicUntidy %>% 
  filter(variable != "Pclass" & variable != "Sex") %>% #Removes all non-numeric values in the variable row
  ggplot(aes(x=Survived, y = as.numeric(value))) +
  facet_wrap(~variable, scales = "free_y")+
  geom_violin(draw_quantiles = c(0.25, 0.5, 0.75))+
  theme_bw()+
  geom_point(alpha = 0.05, size = 3)
#The command remows/ignores 177 rows, do to that fact that vi have 177 missing values in the value row.
#apply(titanicUntidy, 2,  function(x) sum(is.na(x))) # Remove the # to call this function. 

#Create a box subplots for each continuous variable
titanicUntidy %>% 
  filter(variable != "Pclass" & variable != "Sex") %>% #Removes all non-numeric values in the variable row
  ggplot(aes(x=Survived, y = as.numeric(value))) +
  facet_wrap(~variable, scales = "free_y")+
  geom_boxplot(draw_quantiles = c(0.25, 0.5, 0.75))+
  theme_bw()+
  geom_point(alpha = 0.05, size = 3)
#The command remows/ignores 177 rows, do to that fact that vi have 177 missing values in the value row. 

#Create subplots for each categorical variable
titanicUntidy %>% 
  filter(variable == "Pclass" | variable == "Sex") %>% 
  ggplot(aes(value, fill = Survived)) +
  facet_wrap( ~ variable, scales = "free_x") +
  geom_bar(position = "fill")+ #Try change the position argument to "stack" & "dodge"
  theme_bw()

#Creating a task
titanicTask <- makeClassifTask(data = titanic_clean, target = "Survived")

sum(is.na(titanic_clean$Age)) #We have 177 missing values in the Age variable.

#we can either discard this observations or impute the mean of the variable value to the missing values. I am going to impute
imputed <- impute(titanic_clean, cols = list(Age = imputeMean()))

#Check formissing
sum(is.na(imputed$data$Age))

#Creat a new task on none-missing data
titanic_Task <- makeClassifTask(data = imputed$data, target = "Survived")

#create a learner
logReg <- makeLearner("classif.logreg", predict.type = "prob")

#Create trining model on imputed data
logRegModel <- train(learner =  logReg, task = titanic_Task)

#CROSS-VALIDATING

#Wrapping the learner and imputation method together
logRegWrapper <- makeImputeWrapper("classif.logreg", #This creates the same learner as logReg
                                   cols = list(Age =imputeMean())) #This creates an imputation for each cross-validation

#Create a kfold object
kFold <- makeResampleDesc(method = "RepCV", folds = 10, reps = 50, 
                          stratify = TRUE)

#Train the model including kFold strategi
logRegwithImpute <- resample(learner = logRegWrapper,
                             task = titanic_Task, 
                             resampling = kFold,
                             measures = list(acc, fpr, fnr))

logRegwithImpute #FPR = False positive rate, FNR = false negative rate, ACC = accuracy test 
