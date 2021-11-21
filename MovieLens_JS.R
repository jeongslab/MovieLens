########################################################################
# "Movie Recommendation System Using MovieLens Dataset" by Jeong Sukchan
########################################################################

##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse) 
library(caret, warn.conflicts = FALSE) # warn.conflicts = FALSE is to avoid clash)
library(data.table)
library(tidyr)
library(dslabs)
library(dplyr)
library(ggplot2)
library(lubridate)
library(stringr)
library(recosystem)
library(Rcpp)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

#Define RMSE_Root Mean Squared Error
RMSE<-function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings-predicted_ratings)^2, na.rm=T))
}

# Generate the validation set (final hold-out test set)             
# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)
rm(test_index, temp, removed)

# Check there is any missing value
anyNA(edx)

# Check outliers of rating. For visualization, we can use the following code.
boxplot(edx$rating, main="rating spread", col="purple")

# Let's check the number of outliers. 
sum(edx$rating<0.5| edx$rating>5)

# timestamp: The timestamp column represents seconds since midnight Coordinated Universal Time (UTC) of January 1, 1970.). We will convert it into a familiar datetime format making datetime column and removing timestamp column.  
edx_tidy<-edx%>%mutate(datetime=as.POSIXct(edx$timestamp,origin = "1970-01-01",tz = "UTC"))%>%select(-timestamp)

# title: We sill saperate title column into title and releaseyear columns. Along with the movie name there is the year which the movie was released. This can be seperated into 2 different features.  
# library(stringr)
# extract release year from title
pattern <- "(?<=\\()\\d{4}(?=\\))"
edx_tidy$releaseyear <- edx_tidy$title %>% str_extract(pattern) %>% as.integer()

#extract only title without release year from title column removing redundant columns
edx_tidy%>%mutate(title = as.character(str_sub(edx_tidy$title,1, -7)))# the title is from the first to the seventh place from the end

# splitting edx_tidy dataset into train_set and test_set
set.seed(1, sample.kind="Rounding") 
# if using R 3.5 or earlier, use `set.seed(123)`
test_index2 <- createDataPartition(y = edx_tidy$rating, times = 1, p = 0.2, list = FALSE)
train_set <- edx_tidy[-test_index2,] 
test_set <- edx_tidy[test_index2,]
test_set <- test_set %>% semi_join(train_set, by = "movieId") %>% 
  semi_join(train_set, by = "userId")

# sampling: We will use the sampling from train_set dataset in case there will come across limitation and slowness due to the lack of efficiency of my computer. 
# Sampling from train_set 
train_set_sample <- train_set[sample(nrow(train_set), 100000, replace = FALSE),]
set.seed(1, sample.kind="Rounding") 
# if using R 3.5 or earlier, use `set.seed(123)`
test_index3 <- createDataPartition(y = train_set$rating, times = 1, p = 0.2, list = FALSE)
train_set_sample <- train_set_sample[-test_index3,] 
test_set_sample <- train_set_sample[test_index3,]
test_set_sample <- test_set_sample %>% semi_join(train_set_sample, by = "movieId") %>% 
  semi_join(train_set_sample, by = "userId")

dim(train_set)
dim(train_set_sample)
dim(test_set_sample)

###########
# EDA
###########

# Overviewing
str(edx_tidy)

# Size of the dataset
edx_tidy%>% summarize(n_rating=n(), n_movies=n_distinct(movieId), n_users=n_distinct(userId))

# Summary
summary(edx_tidy)

# Variables
# 1)   MovieId
movie_sum <- edx_tidy %>% group_by(movieId) %>%
  summarize(n_rating_of_movie = n(), 
            mu_movie = mean(rating)) 
head(movie_sum)
n_distinct(edx_tidy$movieId)

# Distribution of MovieId
edx_tidy %>% group_by(movieId) %>%
  summarise(n_rating=n()) %>%
  ggplot(aes(n_rating)) +
  geom_histogram(binwidth = 0.25, color = "white") +
  scale_x_log10() +
  ggtitle("Distribution of Movie Ratings") +
  xlab("Number of Ratings") +
  ylab("Number of Movies")

# Distribution of the Movie Effect (Difference from mean)
mu<-mean(edx_tidy$rating)
movie_mean_norm <- edx_tidy %>% 
  group_by(movieId) %>% 
  summarize(movie_effect = mean(rating - mu))
movie_mean_norm %>% qplot(movie_effect, geom ="histogram", bins = 20, data = ., color = I("black")) +
  ggtitle("Distribution of Difference (b_i)", 
          subtitle = "The distribution of the difference from mean shows a tendency") +
  xlab("Difference from mean = movie effect (b_i)") +
  ylab("Count")

# 2)  UserId
# Distribution of userId
# Histogram of User Ratings
edx_tidy %>% group_by(userId) %>%
  summarise(n=n()) %>%
  ggplot(aes(n)) +
  geom_histogram(color = "white") +
  scale_x_log10() + 
  ggtitle("Distribution of Number of Rating by userId") +
  xlab("Number of Ratings") +
  ylab("Number of Users") + 
  geom_density()

# Distribution of the User Effect (Difference from user mean)
user_mean_norm <- edx_tidy %>% 
  left_join(movie_mean_norm, by='movieId') %>%
  group_by(userId) %>%
  summarize(user_effect = mean(rating - mu - movie_effect))
user_mean_norm %>% qplot(user_effect, geom ="histogram", bins = 30, data = ., color = I("black"))+
  ggtitle("Distribution of Difference by UserId (b_u)", 
          subtitle = "The distribution of the difference from mean shows a tendency") +
  xlab("Difference from mean = user effect (b_u)")

# Matrix between movieId and userId
users <- sample(unique(edx_tidy$userId), 100)
edx_tidy%>% filter(userId %in% users) %>% 
  select(userId, movieId, rating) %>%
  mutate(rating = 1) %>%
  spread(movieId, rating) %>% 
  select(sample(ncol(.), 100)) %>% 
  as.matrix() %>% t(.) %>%
  image(1:100, 1:100,. , xlab="Movies", ylab="Users")
abline(h=0:100+0.5, v=0:100+0.5, col = "grey")
title("User x Movie Matrix")

# 3)   rating
edx_tidy %>% group_by(rating) %>% summarize(count = n()) %>% arrange(desc(count))

# Distribution of rating
rating_count<-edx_tidy %>%
  group_by(rating) %>%
  summarize(count = n()) %>%
  ggplot(aes(x = rating, y = count)) +
  geom_line()+
  ggtitle("Number of occurence of each rating")
rating_count

# 4)  releaseyear 
# show relation between releaseyear and rating
edx_tidy %>% group_by(releaseyear) %>%
  summarize(rating = mean(rating)) %>%
  ggplot(aes(releaseyear, rating, fill = releaseyear)) +
  geom_point() + 
  geom_smooth(method = "lm" ) +
  ggtitle("raing over years") +
  theme(plot.title = element_text(hjust = 0.5))

# 5)   genres
# Number of different combinations of genre sets
edx_tidy%>% summarize(n_genres=n_distinct(genres))

# Examples of the combinations
edx_tidy%>%group_by(genres)%>% 
  summarise(n=n()) %>% 
  head()

###############
# Modeling
###############

# Modeling Method: We will use the typical error loss, the root mean squared error (RMSE) on a test_set to decide our algorithm is good. If RMSE is larger than 1, it indicates our typical error is larger than one star, which is not good. 

# Loss function: Our target is already set: the RMSE of our model should be less than 0.86490. Let's check how to describe RMSE assessments. 

# Define RMSE_Root Mean Squared Error
RMSE<-function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings-predicted_ratings)^2, na.rm=T))
}

# 1.   A Naive Model
# Average of rating
mu<-mean(train_set$rating) 

# Estimate RMSE
naive_rmse<-RMSE(test_set$rating, mu)

# Make a results table of the RMSE value and save the results to keep track of it for the sake of comparison with results of other models.  
rmse_results <- tibble(Method = "Model 1: Mean Effect (Naive Model)", RMSE = naive_rmse)
rmse_results %>% knitr::kable()

# 2.  Movie Effect Model  
# Movie Effect algorithm using mean statistic
mu<-mean(train_set$rating)
movie_avgs<-train_set%>%
  group_by(movieId)%>%
  summarise(b_i=mean(rating-mu))

predicted_ratings <- mu+test_set %>% 
  left_join(movie_avgs, by='movieId') %>% 
  pull(b_i)

# Estimate RMSE
movie_rmse<-RMSE(predicted_ratings, test_set$rating)

# RMSE table
rmse_results <- bind_rows(rmse_results,
                         tibble(Method = "Model 2: Movie Effect", 
                                RMSE = movie_rmse))
rmse_results %>% knitr::kable()

# 3.  A Linear Regression Algorithm_Movie Model
# The first try to use a linear regression algorithm using lm() function
fit<-lm(rating~movieId, data=train_set)
y_hat <- fit$coef[1] + fit$coef[2]*test_set$movieId

# Estimate RMSE
regression_movie_rmse<-RMSE(test_set$rating, y_hat)

# RMSE table
rmse_results <- bind_rows(rmse_results, 
                          tibble(Method = "Model 3: Linear Regression Model_Movie", 
                                 RMSE = regression_movie_rmse))
rmse_results %>% knitr::kable()

# 4.  User Effect Model  

# User Effect algorithm calculating the user average effect based on the movie effect algorithm
mu<-mean(train_set$rating)
movie_avgs<-train_set%>%
  group_by(movieId)%>%
  summarise(b_i=mean(rating-mu))

user_avgs <- train_set %>% 
  left_join(movie_avgs, by='movieId') %>% 
  group_by(userId) %>% 
  summarize(b_u = mean(rating - mu - b_i))

predicted_ratings <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>% 
  left_join(user_avgs, by='userId') %>% 
  mutate(pred = mu + b_i + b_u) %>% 
  pull(pred)

# RMSE
user_rmse<-RMSE(predicted_ratings, test_set$rating)

rmse_results <- bind_rows(rmse_results,
                          tibble(Method = "Model 4: Movie + User Effect", 
                                 RMSE = user_rmse))

rmse_results %>% knitr::kable()

# 5.  A Linear Regression Algorithm_User Model
fit2<-lm(rating~userId + movieId, data=train_set)
y_hat2 <- fit2$coef[1] + fit2$coef[2]*test_set$userId

# Estimate RMSE
regression_user_rmse<-RMSE(test_set$rating, y_hat2)

# RMSE table
rmse_results <- bind_rows(rmse_results,
                          tibble(Method = "Model 5: Linear Regression Model_User", 
                                 RMSE = regression_user_rmse))
rmse_results %>% knitr::kable()

### 6.  Trying Genre Effect Model  

# Genre Effect algorithm calculating the genre effect based on the user effect algorithm  <br>
# genre_avgs <- train_set %>% 
#   left_join(movie_avgs, by='movieId') %>% 
#   left_join(user_avgs, by='userId') %>% 
#   group_by(genres)

# predicted_ratings <- test_set %>% 
#   left_join(movie_avgs, by='movieId') %>% 
#   left_join(user_avgs, by='userId') %>% 
#   left_join(genre_avgs, by="genres")%>%
#   mutate(pred = mu + b_i + b_u + b_g) %>% 
#   pull(pred)

# RMSE  <br>
# genre_rmse<-RMSE(predicted_ratings, test_set$rating)
# rmse_results <- bind_rows(rmse_results,
#                         tibble(Method = "Model 6: Movie + User + Genre Effect", 
#                                RMSE = genre_rmse))
# rmse_results


# 7.  Regularization Model
# Regularized Movie + User effects 

lambdas <- seq(0, 10, 0.25) 
rmses <- sapply(lambdas, function(l){
  
mu <- mean(train_set$rating)

b_i <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu)/(n()+l))

b_u <- train_set %>% 
  left_join(b_i, by="movieId") %>% 
  group_by(userId) %>% 
  summarize(b_u = sum(rating - b_i - mu)/(n()+l))

predicted_ratings <- test_set %>% 
  left_join(b_i, by = "movieId") %>% 
  left_join(b_u, by = "userId") %>% 

  mutate(pred = mu + b_i + b_u ) %>% 
  .$pred
  return(RMSE(predicted_ratings, test_set$rating)) }) 
qplot(lambdas, rmses)

# Use the lambda which minimises the RMSE to train the model and predict the test_set
lambda<-lambdas[which.min(rmses)]

mu <- mean(train_set$rating) 

movie_reg_avgs<-train_set%>%
  group_by(movieId)%>%
  summarise(b_i=sum(rating-mu)/(n()+lambda))

user_reg_avgs <- train_set %>% 
  left_join(movie_reg_avgs, by = "movieId") %>% 
  group_by(userId) %>% 
  summarize(b_i = sum(rating - mu)/(n()+lambda), b_u = sum(rating - mu - b_i)/(n()+lambda), n_i = n()) 

predicted_ratings <- mu + test_set %>% 
  left_join(movie_reg_avgs, by = "movieId") %>% 
  left_join(user_reg_avgs, by = "userId") %>% 
  pull(b_u)

# RMSE
reg_genre_rmse<- RMSE(predicted_ratings, test_set$rating)
rmse_results <- bind_rows(rmse_results,
                          tibble (Method="Model 7: Regularized_Movie + User Effect",  
                                     RMSE = min(rmses)))
rmse_results %>% knitr::kable()

# 8.  Matrix Factorization
# Fyi, it took 15mins to run below chunk of code in my computer.
# Select movieId, userId, and rating variables only
edx_fac <- edx_tidy %>% select(movieId, userId, rating)
validation_fac <- validation %>% select(movieId, userId,  rating)
train_set_fac<-train_set%>%select (movieId, userId,  rating)
test_set_fac<-test_set%>%select (movieId, userId, rating)

# Arrange the datasets into matrix forms
edx_fac <- as.matrix(edx_fac)
validation_fac <- as.matrix(validation_fac)
train_set_fac <- as.matrix(train_set_fac)
test_set_fac <- as.matrix(test_set_fac)

# Save the datasets as tables
write.table(edx_fac, file = "edxset.txt", sep = " ", row.names = FALSE, 
            col.names = FALSE)
write.table(validation_fac, file = "validationset.txt", sep = " ", 
            row.names = FALSE, col.names = FALSE)
write.table(train_set_fac, file = "trainset.txt", sep = " ", row.names = FALSE, 
            col.names = FALSE)
write.table(test_set_fac, file = "testset.txt", sep = " ", row.names = FALSE, 
            col.names = FALSE)
set.seed(1)
edx_dataset <- data_file("edxset.txt")
trainset_dataset <- data_file("trainset.txt")
testset_dataset <- data_file("testset.txt")
validation_dataset <- data_file("validationset.txt")

# Create a model object
r = Reco() # this will create a model object

# Tune the algorithm to find the optimal answer
opts = r$tune(trainset_dataset, opts = list(dim = c(10, 20, 30), lrate = c(0.1,
    0.2), costp_l1 = 0, costq_l1 = 0, nthread = 1, niter = 10))

# Train the model using the tuned parameters
r$train(trainset_dataset, opts = c(opts$min, nthread = 1, niter = 20))
stored_prediction = tempfile()

# Predict on the testset_dataset
r$predict(testset_dataset, out_file(stored_prediction))
real_ratings <- read.table("testset.txt", header = FALSE, sep = " ")$V3
pred_ratings <- scan(stored_prediction)

# RMSE
matrix_rmse <- RMSE(real_ratings, pred_ratings)
rmse_results <- bind_rows(rmse_results,
                          data_frame(Method = "Model 8: Matrix Factorization Model", 
                                     RMSE = matrix_rmse ))
rmse_results %>% knitr::kable()

#############
# Results
#############
# Regularized Movie and User Effect Model(Model 7)

# Fyi, it took around 10 mins with my computer to run the below chunk of codes. 
lambdas <- seq(0, 10, 0.25) 

rmses <- sapply(lambdas, function(l){
  mu <- mean(edx$rating)
  b_i <- edx %>% 
    group_by(movieId) %>% 
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- edx %>% 
    left_join(b_i, by="movieId") %>% 
    group_by(userId) %>% 
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  
  predicted_ratings <- validation %>% 
    left_join(b_i, by = "movieId") %>% 
    left_join(b_u, by = "userId") %>% 
    
    mutate(pred = mu + b_i + b_u ) %>% 
    .$pred
  return(RMSE(predicted_ratings, validation$rating)) }) 
qplot(lambdas, rmses)

lambda<-lambdas[which.min(rmses)]
lambda

lambda <- 5.25
mu <- mean(edx$rating) 

movie_reg_avgs<-edx%>%
  group_by(movieId)%>%
  summarise(b_i=sum(rating-mu)/(n()+lambda))

user_reg_avgs <- edx %>% 
  left_join(movie_reg_avgs, by = "movieId") %>% 
  group_by(userId) %>% 
  summarize(b_i = sum(rating - mu)/(n()+lambda), b_u = sum(rating - mu - b_i)/(n()+lambda), n_i = n()) 

predicted_ratings <- mu + validation %>% 
  left_join(movie_reg_avgs, by = "movieId") %>% 
  left_join(user_reg_avgs, by = "userId") %>% 
  pull(b_u)
reg_genre_rmse<- RMSE(predicted_ratings, validation$rating)
reg_genre_rmse


rmse_results <- bind_rows(rmse_results,
                          tibble (Method="The Final Good Model: Regularized_Movie + User Effect",
                                  RMSE = min(rmses)))
rmse_results %>% knitr::kable()

# the Matrix Factorization Model (Model 8)

# Fyi, it took around 20 mins with my computer to run the below chunk of codes.

# Tune the algorithm to find the optimal answer
opts = r$tune(edx_dataset, opts = list(dim = c(10, 20, 30), lrate = c(0.1,
                                                                      0.2), costp_l1 = 0, costq_l1 = 0, nthread = 1, niter = 10))

# Train the model using the tuned parameters
r$train(edx_dataset, opts = c(opts$min, nthread = 1, niter = 20))
stored_prediction = tempfile()

# Predict on the validation_dataset
r$predict(validation_dataset, out_file(stored_prediction))
real_ratings <- read.table("validationset.txt", header = FALSE, sep = " ")$V3
pred_ratings <- scan(stored_prediction)

# RMSE
best_rmse <- RMSE(real_ratings, pred_ratings)
rmse_results <- bind_rows(rmse_results,
                          data_frame(Method = "The Final Best Model: Matrix Factorization Model",
                                     RMSE = best_rmse ))
rmse_results %>% knitr::kable()