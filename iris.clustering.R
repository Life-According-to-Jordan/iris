#read iris data set 
iris<-read.csv("iris.csv")
#create 2nd data frame 
iris.features<-iris
#remove names from iris.features 
iris.features$Name <- NULL
#use kmeans clustering analysis 
results <- kmeans(iris.features, 3)
#view results from algorithm 
results
#which species lie in which cluster 
table(iris$Name, results$cluster)
#visualization of our 3 slusters 
plot(iris[c("SepalLength", "SepalWidth")], col = results$cluster)
 