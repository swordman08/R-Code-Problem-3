library('tidyverse')
library('forcats')

movies <- read.csv("datasets/IMDB_movies.csv")

movies_clean <- 
  movies %>% 
  mutate(budgetM = budget/1000000,
         grossM = gross/1000000,
         profitM = grossM - budgetM,
         ROI = profitM/budgetM,
         blockbuster = ifelse(profitM > 100, 1,0) %>% factor(., levels = c("0","1")),
         blockbuster_numeric = ifelse(profitM > 100, 1,0), 
         genre_main = as.factor(unlist(map(strsplit(as.character(movies$genres),"\\|"),1))) %>% fct_lump(10),
         rating_simple = fct_lump(content_rating, n = 5)
  ) %>%
  filter(country == "USA", 
         content_rating != "", 
         content_rating != "Not Rated",
         !is.na(gross)) %>% 
  mutate(rating_simple = rating_simple %>% fct_drop()) %>% 
  rename(director = director_name, 
         title = movie_title,
         year = title_year) %>% 
  select(-c(actor_1_name, actor_2_name,actor_1_facebook_likes, actor_2_facebook_likes, 
            budget, gross, aspect_ratio, num_voted_users,num_user_for_reviews)) %>% 
  relocate(title, year, country, director, budgetM, grossM, profitM, ROI, imdb_score, genre_main, rating_simple, language, duration) %>% 
  distinct()

#b.

library(rsample)

set.seed(42)
split <- initial_split(movies_clean, prop = 0.75)
train_set <- training(split)
test_set <- testing(split)

#c.

movies_logit1 <- glm(blockbuster ~ imdb_score + budgetM + year + director_facebook_likes + genre_main, 
                     data = train_set,
                     family = binomial())

summary(movies_logit1)

#d , e 

#crime makes the movie less likely to be blockbuster as its coefficant is 0.14, which is below 1, so unlikely to happen

#imdb has a coefficant of of 3.77 so its not true that blockbusters get a lower score.
#they are more likely to be blockbusters as its above 1.00.
exp(coef(movies_logit1))


#f.

train_predictions <- predict(movies_logit1, train_set, type = "response")
test_predictions <- predict(movies_logit1, test_set, type = "response")

head(train_predictions)

head(test_predictions)

library(ggplot2)
library(plotROC)

#g.

# Plot for Training Set
roc_train = ggplot(train_set, aes(d = blockbuster_numeric, m = train_predictions)) + geom_roc(cutoffs.at = c(0.9,0.7,0.5,0.3,0.1,0.05)) + ggtitle("ROC Curve for Training Set")

# Plot for Test Set
roc_test = ggplot(test_set, aes(d = blockbuster_numeric, m = test_predictions)) + geom_roc(cutoffs.at = c(0.9,0.7,0.5,0.3,0.1,0.05)) + ggtitle("ROC Curve for Testing Set")

roc_test

roc_train

#h,   chose cutoff of .1, that minimizes false_positive_fraction and maximizes true positive fraction.
#a smaller cutoff would result in more false positive, I picked a cutoff the maximizes positives while minimizing false positives.


#i.

cutoff = 0.1
library(yardstick)
train_classified <- ifelse(train_predictions > cutoff, 1, 0)
test_classified <- ifelse(test_predictions > cutoff, 1, 0)


results_train = data.frame(
  true = factor(train_set$blockbuster_numeric),
  pred = factor(train_classified),
  scores = train_predictions
)
results_train %>% glimpse()
summary(results_train)

results_test = data.frame(
  true = factor(test_set$blockbuster_numeric),
  pred = factor(test_classified),
  scores = test_predictions
)

results_test %>% glimpse()

# J. Confusion matrix

cm_test = conf_mat(results_test,
              truth = true,
              estimate = pred)

cm_test

cm_train = conf_mat(results_train,
                    truth = true,
                    estimate = pred)
cm_train


#k. 

TN_train = 1784
TP_train = 103
FP_train = 306
FN_train = 47

train_acc = (TP_train + TN_train) / (TP_train + TN_train + FP_train + FN_train)
print(train_acc)
train_sen = TP_train / (TP_train + FN_train)
train_sen

train_spe = TN_train / (TN_train + FP_train)

print(train_spe)

TN = 595
TP = 27
FP = 103
FN = 22

test_acc = (TP + TN) / (TP + TN + FP + FN)
print(test_acc)

test_sen = TP / (TP + FN)

test_sen

test_spe = TN / (TN + FP)

print(test_spe)
library(forcats)

# L. 

#Given the AUC scores, the model does not appear to be severely over fitting,
#as the performance on the test set remains relatively close to the training set performance. However, 
#the modest drop from training to test AUC suggests there is some over fitting happening
#if concerned about the over fitting, we could apply regularization techniques to penalize complex models and reduce overfitting.
# we could also simplify the model by removing irrelevant input features that do not contribute much to the predictive power of the model but may be contributing to noise.

calc_auc(roc_train)
calc_auc(roc_test)
  


