library(caret)
# attach the iris dataset to the environment
data(iris)
# rename the dataset
dataset <- iris

# create a list of 80% of the rows in the original dataset we can use for training
# Wir teilen createDataPartition mit, welche Spalte der Factor ist. Es werden
# dann Records gemäss Häufigkeit der Factoren genommen
#
validation_index <- createDataPartition(dataset$Species, p=0.80, list=FALSE)

# select 20% of the data for validation
validation <- dataset[-validation_index,]

# use the remaining 80% of data to training and testing the models
dataset <- dataset[validation_index,]

# dimensions of dataset
dim(dataset)

# list types for each attribute
sapply(dataset, class)

# take a peek at the first 5 rows of the data
head(dataset)

# list the levels for the class
levels(dataset$Species)

# summarize the class distribution
percentage <- prop.table(table(dataset$Species)) * 100
cbind(freq=table(dataset$Species), percentage=percentage)

# summarize attribute distributions
summary(dataset)

# split input and output
x <- dataset[,1:4]
y <- dataset[,5]

# boxplot for each attribute on one image
par(mfrow=c(1,4))
for(i in 1:4) {
  boxplot(x[,i], main=names(iris)[i])
}

# barplot for class breakdown
plot(y)

# scatterplot matrix
featurePlot(x=x, y=y, plot="ellipse")

# Run algorithms using 10-fold cross validation
control <- trainControl(method="cv", number=10)
metric <- "Accuracy"


# a) linear algorithms
set.seed(7)
fit.lda <- train(Species~., data=dataset, method="lda", metric=metric, trControl=control)
# b) nonlinear algorithms
# CART
set.seed(7)
fit.cart <- train(Species~., data=dataset, method="rpart", metric=metric, trControl=control)
# kNN
set.seed(7)
fit.knn <- train(Species~., data=dataset, method="knn", metric=metric, trControl=control)
# c) advanced algorithms
# SVM
set.seed(7)
fit.svm <- train(Species~., data=dataset, method="svmRadial", metric=metric, trControl=control)
# Random Forest
set.seed(7)
fit.rf <- train(Species~., data=dataset, method="rf", metric=metric, trControl=control)

# summarize accuracy of models
results <- resamples(list(lda=fit.lda, cart=fit.cart, knn=fit.knn, svm=fit.svm, rf=fit.rf))
summary(results)

# compare accuracy of models
dotplot(results)

# summarize Best Model
print(fit.lda)

# estimate skill of LDA on the validation dataset
predictions <- predict(fit.lda, validation)
confusionMatrix(predictions, validation$Species)

# Wir erzeugen einen neuen Datensatz mit den Messwerten und verwenden das Modell, um die Spezies / Art 
# vorauszusagen. Messwert + Modell -> Vorausage der Spezies. Oder Messwerte + Modell liefert den
# Level des Factors

# Werte der Spalten angeben.
sepalLenght <- c(5.9, 1)
sepalWidth <- c(3.3, 1)
petalLenght <- c(5.1, 1)
petalWidth <- c(1.9, 1)

# Datenframe erzeugen
# 
new.flowerMeasurement <- data.frame(sepalLenght, sepalWidth, petalLenght, petalWidth)

# Spaltenname hinzufügen. Das Modell wurde mit den Spaltennamen erzeugt, darum müssen die
# die neu vorherzusagenden Datensätze auch Spaltennamen haben.    
colnames(new.flowerMeasurement) <- c("Sepal.Length","Sepal.Width", "Petal.Length", "Petal.Width") 

predict(fit.lda, newdata=new.flowerMeasurement)