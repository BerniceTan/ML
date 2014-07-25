## Practical Machine Learning Course Project


### Background
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset). 


### Data Sources 
The data for this project come from this source: http://groupware.les.inf.puc-rio.br/har


### Objective
The goal for this exercise is to predict the manner in which they did the exercise.


### Course Project Research Areas :
0) Getting and Cleaning Data
1) How you built your model     
2) How you used cross validation    
3) What you think the expected out of sample error is   
4) Why you made the choices you did   


### 0) Getting and Cleaning Data

#### a) Load Data



```r
library(caret)
```

```
## Warning: package 'caret' was built under R version 3.0.3
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```
## Warning: package 'ggplot2' was built under R version 3.0.3
```

```r
setwd("C:/Users/1483299/Documents/Ting Fui/Machine Learning")
pmltraining <- read.csv("pml-training.csv")
testing <- read.csv("pml-testing.csv")
```

#### b) Create Data Partition

```r
inTrain <- createDataPartition(pmltraining$classe, p=0.60, list=FALSE)
training <- pmltraining[inTrain,]
validation <- pmltraining[-inTrain,]
```

#### c) Cleaning Data

```r
training<-training[,colSums(is.na(training)) == 0]
classe<-training$classe
nums <- sapply(training, is.numeric)
training<-cbind(classe,training[,nums])
training$X<-training$num_window<-NULL

validation<-validation[,colSums(is.na(validation)) == 0]
vclasse<-validation$classe
vnums <- sapply(validation, is.numeric)
validation<-cbind(vclasse,validation[,vnums])
colnames(validation)[1]<-"classe"
validation$X<-validation$num_window<-NULL

testing<-testing[,colSums(is.na(testing)) == 0]
tnums <- sapply(testing, is.numeric)
testing<-testing[,tnums]
testing$X<-testing$num_window<-NULL
```


### 1) Model Building
Fit a model using random forest


```r
fit <- train(training$classe~.,data=training, method="rf")
```

```
## Loading required package: randomForest
```

```
## Warning: package 'randomForest' was built under R version 3.0.3
```

```
## randomForest 4.6-7
## Type rfNews() to see new features/changes/bug fixes.
```

```
## Warning: package 'e1071' was built under R version 3.0.3
```


```r
fit$results
```

```
##   mtry Accuracy  Kappa AccuracySD  KappaSD
## 1    2   0.9909 0.9885   0.001232 0.001554
## 2   28   0.9958 0.9947   0.001094 0.001381
## 3   54   0.9918 0.9896   0.004155 0.005262
```


### 2) Cross Validation

Using the model that we've trained, we're performing a cross validation with the validation dataset. 


```r
traincontrol <- trainControl(method = "cv", number = 5)
```


```r
fit_crossvalidation <- train(validation$classe~.,data=validation, method="rf",trControl=traincontrol)
```

### 3) Expected out of sample error

The out of error rate is expected to be less than 1%, as the accuracy of the model observed above is >99%.


```r
fit_crossvalidation$resample
```

```
##   Accuracy  Kappa Resample
## 1   0.9936 0.9919    Fold1
## 2   0.9911 0.9887    Fold3
## 3   0.9930 0.9911    Fold2
## 4   0.9904 0.9879    Fold5
## 5   0.9911 0.9887    Fold4
```

```r
fit_crossvalidation$results
```

```
##   mtry Accuracy  Kappa AccuracySD  KappaSD
## 1    2   0.9866 0.9831   0.003601 0.004556
## 2   28   0.9918 0.9897   0.001377 0.001742
## 3   54   0.9885 0.9855   0.002948 0.003731
```

```r
confusionMatrix(predict(fit_crossvalidation, newdata=validation), validation$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2232    0    0    0    0
##          B    0 1518    0    0    0
##          C    0    0 1368    0    0
##          D    0    0    0 1286    0
##          E    0    0    0    0 1442
## 
## Overall Statistics
##                                 
##                Accuracy : 1     
##                  95% CI : (1, 1)
##     No Information Rate : 0.284 
##     P-Value [Acc > NIR] : <2e-16
##                                 
##                   Kappa : 1     
##  Mcnemar's Test P-Value : NA    
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             1.000    1.000    1.000    1.000    1.000
## Specificity             1.000    1.000    1.000    1.000    1.000
## Pos Pred Value          1.000    1.000    1.000    1.000    1.000
## Neg Pred Value          1.000    1.000    1.000    1.000    1.000
## Prevalence              0.284    0.193    0.174    0.164    0.184
## Detection Rate          0.284    0.193    0.174    0.164    0.184
## Detection Prevalence    0.284    0.193    0.174    0.164    0.184
## Balanced Accuracy       1.000    1.000    1.000    1.000    1.000
```
By calculating the out of sample error (the cross-validation estimate is an out-of-sample estimate) we get the value of `0.52%`:


```r
fit_crossvalidation$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 28
## 
##         OOB estimate of  error rate: 0.5%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 2230    1    0    1    0   0.0008961
## B    8 1508    2    0    0   0.0065876
## C    0    5 1361    2    0   0.0051170
## D    0    0   12 1271    3   0.0116641
## E    0    0    0    5 1437   0.0034674
```

### Predict the 20 test cases

Finally, to predict the classe of the testing dataset, we will use the model we've trained and output the results in the respective files.


```r
test_prediction<-predict(fit, newdata=testing)
test_prediction
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

```r
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
pml_write_files(test_prediction)
```
