df <- read.csv(file="http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com/pp-2018.csv", header=FALSE, sep=",")


library(data.table)
setnames(df, 
         old = c('V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11','V12','V13','V14','V15','V16'), 
         new = c('ID','PRICE','DATE','POSTCODE','PROPERTY_TYPE','OLD_NEW','DURATION','PAON','SAON','STREET','LOCALITY','CITY_TOWN','DISTRICT','COUNTY','PPD_CAT','RECORD_STATUS'))

## Extract columns of interest from dataset
df <- df[,c('PRICE','PROPERTY_TYPE','OLD_NEW','CITY_TOWN','DISTRICT')]

## Decrease factor levels by removing city_town
str(df)
df <- df[,c('PRICE','PROPERTY_TYPE','OLD_NEW','DISTRICT')]

## PLOT DIST OF PRICE
library(ggplot2)
ggplot(df, aes(x=PRICE)) + geom_histogram() 


## PLOT BOXPLOT OF OUTLIERS
ggplot(data = df, aes(x = RECORD_STATUS, y = PRICE)) + geom_boxplot() + geom_point(aes(fill = RECORD_STATUS), size = 6, shape = 21) 


## REMOVE OUTLIERS IQR
nrow(df)
keep <- !df$PRICE %in% boxplot.stats(df$PRICE)$out
df <- df[keep, ]
nrow(df)


## PLOT BOXPLOT OF OUTLIERS
ggplot(data = df, aes(x = RECORD_STATUS, y = PRICE)) + geom_boxplot() + geom_point(aes(fill = RECORD_STATUS), size = 6, shape = 21) 


## PLOT DIST OF PRICE
library(ggplot2)
ggplot(df, aes(x=PRICE)) + geom_histogram() 

library(psych)
describe(df$PRICE)


## create another dataframe and log transform the skewed data
SKEW_COR <- df
SKEW_COR$PRICE <- log(SKEW_COR$PRICE)
ggplot(SKEW_COR, aes(x=PRICE)) + geom_histogram() 


## REMOVE OUTLIERS IQR for skew corrected data
nrow(SKEW_COR)
keep <- !SKEW_COR$PRICE %in% boxplot.stats(SKEW_COR$PRICE)$out
SKEW_COR <- SKEW_COR[keep, ]
nrow(SKEW_COR)

library(psych)
describe(SKEW_COR$PRICE)


## Dummify variables
library(caret)
dmy <- dummyVars(" ~ .", data = df, fullRank=T)
data_skewed <- data.frame(predict(dmy, newdata = df))

dmy <- dummyVars(" ~ .", data = SKEW_COR, fullRank=T)
data_log_tf <- data.frame(predict(dmy, newdata = SKEW_COR))

## Split datasets for training/test 70/30 split
set.seed(1234)
splitIndex <- createDataPartition(data_skewed[,1], p = .7, list = FALSE, times = 1)
trainDF <- data_skewed[ splitIndex,]
testDF  <- data_skewed[-splitIndex,]

set.seed(1234)
splitIndex <- createDataPartition(data_log_tf[,1], p = .7, list = FALSE, times = 1)
trainDF_log <- data_log_tf[ splitIndex,]
testDF_log  <- data_log_tf[-splitIndex,]

set.seed(1234)
splitIndex <- createDataPartition(df[,1], p = .7, list = FALSE, times = 1)
trainDF_raw <- df[ splitIndex,]
testDF_raw  <- df[-splitIndex,]

set.seed(1234)
splitIndex <- createDataPartition(SKEW_COR[,1], p = .7, list = FALSE, times = 1)
trainDF_raw_log <- SKEW_COR[ splitIndex,]
testDF_raw_log  <- SKEW_COR[-splitIndex,]

## Modelling
#Parrallel processing start
library(doParallel)
cl <- makeCluster(detectCores()-1)
registerDoParallel(cl)

## 10 fold cross-validation setup, 
objControl <- trainControl(method='cv',number=10, selectionFunction = "oneSE")

## Setting up search grids for tuning parameter selection
alpha.grid <- 1  # mixing parameter; alpha=1 => lasso; alpha=0 => Ridge; 0<alpha<1 => elasticnet
lambda.grid <- 10^seq(4,-4,length=100) #10^seq(.8,-1,length=50)
srchGrd = expand.grid(.alpha=alpha.grid, .lambda = lambda.grid)
gbmGrid <-  expand.grid(interaction.depth = c(1, 5, 9), 
                        n.trees = (1:30)*50, 
                        shrinkage = 0.1,
                        n.minobsinnode = 20)

objModellasso <- train(trainDF[,2:ncol(trainDF)], trainDF[,1], 
                       method='glmnet', 
                       trControl=objControl,  
                       metric = "RMSE",
                       tuneGrid = srchGrd,
                       preProc = c("center", "scale"))

objModellasso_log <- train(trainDF_log[,2:ncol(trainDF_log)], trainDF_log[,1], 
                           method='glmnet', 
                           trControl=objControl,  
                           metric = "RMSE",
                           tuneGrid = srchGrd,
                           preProc = c("center", "scale"))


objModelgbm <- train(trainDF_raw[,2:ncol(trainDF_raw)], trainDF_raw[,1], 
                     method='gbm', 
                     trControl=objControl,  
                     metric = "RMSE",
                     tuneGrid = gbmGrid)

objModelgbm_log <- train(trainDF_raw_log[,2:ncol(trainDF_raw_log)], trainDF_raw_log[,1], 
                         method='gbm', 
                         trControl=objControl,  
                         metric = "RMSE",
                         tuneGrid = gbmGrid)


stopCluster(cl)

registerDoSEQ()

## Plot cross validated model results
plot(objModellasso)
plot(objModellasso_log)
plot(objModelgbm)
plot(objModelgbm_log)

##Plot important variables
ggplot(varImp(objModellasso, scale = FALSE), top = 20)
ggplot(varImp(objModellasso_log, scale = FALSE), top = 20)
ggplot(varImp(objModelgbm, scale = FALSE))
ggplot(varImp(objModelgbm_log, scale = FALSE))

#Model Compare
results <- resamples(list(GBM=objModelgbm, GBM_log=objModelgbm_log, Lasso=objModellasso, Lasso_log=objModellasso_log))
summary(results)

## Predict test data
testDF$LASSO <- predict(object=objModellasso, testDF[,2:ncol(testDF)], type='raw')
testDF_log$LASSO_LOG <- predict(object=objModellasso_log, testDF_log[,2:ncol(testDF_log)], type='raw')
testDF_raw$GBM <- predict(object=objModelgbm, testDF_raw[,2:ncol(testDF_raw)], type='raw')
testDF_raw_log$GBM_LOG <- predict(object=objModelgbm_log, testDF_raw_log[,2:ncol(testDF_raw_log)], type='raw')

## Get metrics from predicted to actual on validation data
LASSO <- lm(data = testDF,x = testDF$LASSO,y = testDF$PRICE)
LASSO_LOG <- lm(data = testDF_log,x = testDF_log$LASSO_LOG,y = testDF_log$PRICE)

write.table(testDF_raw, file = "C:/Users/501671/Downloads/CA3_raw.csv", sep=",",row.names = FALSE)