#
# DAT-690 Final Project
# Employee Attrition Project Two
# John Dirgo Deweese
# Start: 4/21/2024
#

# Set up working directory
setwd("G:/My Drive/24TW3 - DAT 690 - Capstone in Data Analytics/DAT-690 Final Project")

# Put libraries you'll need at the start of this file
# It just makes life easier


#### Data Wrangling
# Tidyverse contains ggplot2, tibble, tidyr, readr, 
#    dplyr, stringr, purr, and forcats
library(tidyverse)
library(mltools)
library(data.table)
library(skimr)
library(rstatix)
library(inspectdf)
library(janitor)
library(ggrepel)
library(qqplotr)

# Data Validator
library(validate)
library(Rfast)

# Plotting
library(lattice)
library(ggvis)
library(plotly)
library(RColorBrewer)
library(gridExtra)
library(corrplot)
library(ggcorrplot)

# Machine Learning
library(nnet)
library(keras3)


# Machine Learning Evaluation
library(caret)        # an aggregator package for performing many ML models
library(h2o)          # an extremely fast java-based platform
library(car)          # Companion to Applied Regression

# Feature Selection Algorithm Package
library(Boruta)

# Tidymodels includes broom, dials, infer, modeldata, parsnip, recipes,
#     rsample, tune, workflows, workflowsets, yardstick
library(tidymodels)
library(corpcor)
library(mctest)
library(pROC)
library(ROCR)
library(MASS)       # Modern Applied Statistics with S
library(caTools)    # For Linear regression     

source("FinalProject_functions.R")



# Import training data file
trn_dataV1 <- read_csv("data/EmployeeSalary_Data.csv", show_col_types = FALSE)

# We know that some of these are categorical variables (factors) even thought they are numeric
# but we are not going to use the col_factor option on the read_csv in the beginning
# We have a dictionary of the factors and their values, so we first want to make sure we don't
# have out-of-range data.  Once the data quality has been checked we can convert to a factor
# where that is appropriate.

trn_validation_checks <- do_validation_check(trn_dataV1)

#############################################
# The value passed back is the number of errors and warnings 
# from the validation check function
#############################################
print(paste("There are ", trn_validation_checks$validation_errors, "errors in the data validation process"))  
print(paste("There are ", trn_validation_checks$validation_warnings, "warning in the data validation"))

if(trn_validation_checks$validation_errors + trn_validation_checks$validation_warnings == 0) {
  print("All validation checks passed without error")
}  else {
  print("It is recommended that this process be stopped and the data errors be corrected")
  print("before continuing")
}

# MAKE A BACKUP COPY!!!
trn_data_backup01 <- trn_dataV1

trn_dataV2 <- do_feature_engineering(trn_dataV1)

multi_col <- do_multicollinearity_check(trn_dataV2)

# Print the significant correlations (row and column indices
multi_col1 <- multi_col[order(multi_col$p, decreasing=FALSE),]

print(multi_col1)

########################################
# Why we're not removing features as a result of the
# multicollinearity check..
#
# First -- the fields that are showing the highest multicollinearity
# are the fields that we would expect to show that. A higher retention percentage
# (pay increase) would have a relationship with Performance Rating. Percentage of
# time with the company working for the current manager  would have a relationship
# with the percentage of time in their current role.
#
# Second -- We found that the Boruta feature reduction removed nearly all
# of fields involved in multicollinearity
#########################################
#Make a backup
trn_data_backup02 <- trn_dataV2

# Take our current numeric categoricals and turn them into factors (either 
# ordinal or nominal)
trn_dataV3 <- do_factor_conversion(trn_dataV2)

## Backup pre-normalization
trn_data_backup03 <- trn_dataV3 

## Normalize the numeric attributes (essentially convert to a z-score)

trn_sal_list <- do_standardize_numerics(trn_dataV3)

# Capture and save the mean and standard deviations
# so that predictions can be converted back from scaled numbers
# to real numbers (these are returned by the do_standardize_numerics function)
# along with the data frame

trn_cur_sal_mean <- trn_sal_list$mean_cur_sal
trn_cur_sal_sd <- trn_sal_list$sd_cur_sal
trn_ret_pct_mean <- trn_sal_list$mean_ret_pct
trn_ret_pct_sd <- trn_sal_list$sd_ret_pct

trn_sal_data <- trn_sal_list$data_frame 
###################################################################
# Categorical Variable Independence Check!
###################################################################

trn_dep_vars <- do_categorical_independence_check(trn_sal_data)
# Look at the Dependent ones
print(trn_dep_vars)

# # Since JobLevel appear in both of the dependent pairs, remove it
# It was also high-scoring in multicollinearity

trn_sal_data <- subset(trn_sal_data, select = -c(JobLevel))

########################################################
# Boruta for feature reduction
########################################################

boruta.train <- Boruta::Boruta(RetentionPercentNeeded_Std ~ ., data=trn_sal_data, doTrace=0)

print(boruta.train)

# We only have a few (if any) tentative options left so let's figure those out
# their median z-score will be compared to the median z-score of the best shadow
# attribute and either in/out based on that

final.boruta <- TentativeRoughFix(boruta.train)
print(final.boruta)

# Graph it out - with boxplots!

plot(final.boruta, xlab = "", xaxt = "n")
lz<-lapply(1:ncol(final.boruta$ImpHistory),function(i)
  final.boruta$ImpHistory[is.finite(final.boruta$ImpHistory[,i]),i])
names(lz) <- colnames(final.boruta$ImpHistory)
Labels <- sort(sapply(lz,median))
axis(side = 1,las=2,labels = names(Labels),
     at = 1:ncol(final.boruta$ImpHistory), cex.axis = 0.7)

#Too tight to see, so let's build a little dataframe of the values
boruta.df <- attStats(final.boruta)

# Extract the attributes that Boruta found valuable to keep
retained_attributes <- boruta.df[boruta.df$decision == "Confirmed", ]

# Pull out the names into a list so we can extract only those into our dataset
retained_attributes_list <- rownames(retained_attributes)

# Add the target attribute
retained_attributes_list <- c(retained_attributes_list,"RetentionPercentNeeded_Std")

# Pull those fields into a final training database
fnl_trn_db <- trn_sal_data[retained_attributes_list]

# Determine with attributes in the final training dataframe are factors 
# (so they can be converted to one-hot mapping)
factor_cols <- names(fnl_trn_db)[sapply(fnl_trn_db, is.factor)]

# Convert the data frame into a data table
fnl_trn_DT <- data.table(fnl_trn_db)

# Do the one_hot encoding of any categorical (factor) attributes
# Be sure to eliminate the original column once converted (dropCols = TRUE)
fnl_trn_DT <- mltools::one_hot(fnl_trn_DT, cols=factor_cols, dropCols = TRUE)

#######################################################################
#   Begin Baseline Testing -- Linear Regression
#######################################################################
trn_linear_reg <- lm(RetentionPercentNeeded_Std ~ ., data = fnl_trn_DT)

summary(trn_linear_reg)

############
# Calculate mse (mean squared error), standard error and AIC 
# (Aikaike Information Criterion)
# In all cases lower is better.  If AIC is negative, you want the most negative
############
trn_lm_residuals <- trn_linear_reg$residuals
trn_lm_mse <- mean(trn_lm_residuals^2)

trn_std_error_lm <- sqrt(trn_lm_mse)

trn_linear_AIC <- extractAIC(trn_linear_reg)

print(paste("Mean Squared Error of the linear regression model is ", trn_lm_mse))
print(paste("Standard error of linear regression model is ", trn_std_error_lm))
print(paste("AIC of linear regression model is ", trn_linear_AIC[2]," with ",trn_linear_AIC[1], " degrees of freedom"))

#######################################################################
#   Begin Baseline Testing -- Logistic Regression
#######################################################################
trn_log_reg <- glm(RetentionPercentNeeded_Std ~ ., data = fnl_trn_DT, family=gaussian)

summary(trn_log_reg)

############
# Calculate mse (mean squared error), standard error and AIC 
# (Aikaike Information Criterion)
# In all cases lower is better.  If AIC is negative, you want the most negative
############
trn_glm_residuals <- trn_log_reg$residuals
trn_glm_mse <- mean(trn_glm_residuals^2)

trn_std_error_glm <- sqrt(trn_glm_mse)

trn_logistic_AIC <- extractAIC(trn_log_reg)

print(paste("Mean Squared Error of the logistic regression model is ", trn_glm_mse))
print(paste("Standard error of logistic regression model is ", trn_std_error_glm))
print(paste("AIC of logistic regression model is ", trn_logistic_AIC[2]," with ",trn_logistic_AIC[1], " degrees of freedom"))

##########################################################
# Linear is better but still shitty
# Continue with linear regression as the baseline
##########################################################

# TRAINING DATA
# Create a dataframe of the actuals and predicted (linear regression)
trn_lm_accuracy <- data.frame(Actual = fnl_trn_DT$RetentionPercentNeeded_Std, 
                                      Predicted = trn_linear_reg$fitted.values)

# Unconvert the Actual percent and Predicted percent
trn_lm_accuracy$Descaled_Actual <- (trn_lm_accuracy$Actual * trn_ret_pct_sd) +
  trn_ret_pct_mean

trn_lm_accuracy$Descaled_Predicted <- (trn_lm_accuracy$Predicted * trn_ret_pct_sd) +
  trn_ret_pct_mean

# Calculate the difference between the actual and predicted percentages of additional
# retention salary needed

trn_lm_accuracy$Difference = (abs(trn_lm_accuracy$Descaled_Predicted - trn_lm_accuracy$Descaled_Actual) / 
                            abs(trn_lm_accuracy$Descaled_Actual))*100

# Create boolean fields of prediction within 10, 15 or 20%
trn_lm_accuracy$Within_10_Percent <- trn_lm_accuracy$Difference <= 10
trn_lm_accuracy$Within_15_Percent <- trn_lm_accuracy$Difference <= 15
trn_lm_accuracy$Within_20_Percent <- trn_lm_accuracy$Difference <= 20

# Store the percentages of those variances
trn_lm_percent_within_10 <- mean(trn_lm_accuracy$Within_10_Percent) * 100
trn_lm_percent_within_15 <- mean(trn_lm_accuracy$Within_15_Percent) * 100
trn_lm_percent_within_20 <- mean(trn_lm_accuracy$Within_20_Percent) * 100

# Graph it to show the percentage within each bin range
histPercent(trn_lm_accuracy$Difference, xlab="Percentage Difference (Actual vs Predicted)", 
            main="Training Data Accuracy of Linear Regression Model",
            xlim=c(0,60), ylim=c(0,60), col="green", border="black")

##########################################################
# Let's do the neural network
##########################################################

train_nn <- nnet(RetentionPercentNeeded_Std ~ ., data = fnl_trn_DT, 
              size=75, linout=TRUE, maxit=500)

# TRAINING DATA
# Create a dataframe of the actuals and predicted (neural network)
trn_nn_results <- data.frame(actual = fnl_trn_DT$RetentionPercentNeeded_Std, 
                         prediction = train_nn$fitted.values)

colnames(trn_nn_results) <- c("actual","prediction")

############
# Calculate mse (mean squared error), standard error and AIC 
# (Aikaike Information Criterion)
# In all cases lower is better.  If AIC is negative, you want the most negative
############
trn_nn_residuals <- train_nn$residuals
trn_nn_mse <- mean(trn_nn_residuals^2)

trn_nn_sse <- sum(trn_nn_residuals^2)
trn_std_error_nn <- sqrt(trn_nn_mse)
trn_n <- nrow(fnl_trn_DT)
trn_p <- ncol(fnl_trn_DT)
trn_neural_AIC <- (trn_n*log(trn_nn_sse/trn_n)) + 2*trn_p


trn_std_error_nn <- sqrt(nn_mse)

print(paste("Mean Squared Error of the neural network model is ", trn_nn_mse))
print(paste("Standard error of neural network model is ", trn_std_error_nn))
print(paste("AIC of linear regression model is ", trn_neural_AIC))

trn_nn_plot <- ggplot(trn_nn_results, aes(x = actual, y = prediction))+ geom_point(color="orange") + geom_abline(aes(intercept = 0, slope = 1)) +
  labs(y= "Predicted Standardized Retention Percent", x = "Real Standardized Retention Percent")

trn_nn_plot

# DE-standardize the training results (multiple by std deviation and add the mean)
# so that comparisons can be made based on actual results
destd_trn_ret_pct_actual = (trn_nn_results$actual * trn_ret_pct_sd) + trn_ret_pct_mean
destd_trn_ret_pct_predict = (trn_nn_results$prediction * trn_ret_pct_sd) + trn_ret_pct_mean
destd_trn_ret_pct_diff = (abs(destd_trn_ret_pct_predict - destd_trn_ret_pct_actual) / 
  abs(destd_trn_ret_pct_actual))*100

# TUrn those into a data frame
trn_nn_accuracy <- data.frame(Actual_Pct = destd_trn_ret_pct_actual,Predict_Pct = destd_trn_ret_pct_predict,
                           Difference = destd_trn_ret_pct_diff)

# Create boolean fields of prediction within 10, 15 or 20%
trn_nn_accuracy$Within_10_Percent <- trn_nn_accuracy$Difference <= 10
trn_nn_accuracy$Within_15_Percent <- trn_nn_accuracy$Difference <= 15
trn_nn_accuracy$Within_20_Percent <- trn_nn_accuracy$Difference <= 20

# Store the percentages of those variances
trn_nn_percent_within_10 <- mean(trn_nn_accuracy$Within_10_Percent) * 100
trn_nn_percent_within_15 <- mean(trn_nn_accuracy$Within_15_Percent) * 100
trn_nn_percent_within_20 <- mean(trn_nn_accuracy$Within_20_Percent) * 100

trn_nn_plot2 <- ggplot(trn_nn_accuracy, aes(x = Actual_Pct, y = Predict_Pct))+ 
  geom_point(color="purple") + geom_abline(aes(intercept = 0, slope = 1)) +
  labs(y= "Predicted Retention Percent", x = "True Retention Percent")

trn_nn_plot2

histPercent(trn_nn_accuracy$Difference, xlab="Percentage Difference (Actual vs Predicted)", 
            main="Training Data Accuracy of Neural Network Model",
            xlim=c(0,60), ylim=c(0,60), col="green", border="black")

# Build a little dataframe to show the comparisons
trn_models <- c("Linear Regression","Logistic Regression","Neural Network")
trn_mse <- c(trn_lm_mse,trn_glm_mse,trn_nn_mse)
trn_std_error <- c(trn_std_error_lm,trn_std_error_glm,trn_std_error_nn)
trn_AIC <- c(trn_linear_AIC[2],trn_logistic_AIC[2],trn_neural_AIC)

trn_model_comparison <- data.frame(trn_models,trn_mse,trn_std_error,trn_AIC)
print(trn_model_comparison)


#################################################################################
######################### Testing Dataset Preparation ###########################
#################################################################################

# Import testing data file
tst_dataV1 <- read_csv("data/EmployeeSalary_Verify.csv", show_col_types = FALSE)

# We know that some of these are categorical variables (factors) even thought they are numeric
# but we are not going to use the col_factor option on the read_csv in the beginning
# We have a dictionary of the factors and their values, so we first want to make sure we don't
# have out-of-range data.  Once the data quality has been checked we can convert to a factor
# where that is appropriate.

tst_validation_checks <- do_validation_check(tst_dataV1)

#############################################
# The value passed back is the number of errors and warnings 
# from the validation check function
#############################################
print(paste("There are ", tst_validation_checks$validation_errors, "errors in the data validation process"))  
print(paste("There are ", tst_validation_checks$validation_warnings, "warning in the data validation"))

if(tst_validation_checks$validation_errors + tst_validation_checks$validation_warnings == 0) {
  print("All validation checks passed without error")
}  else {
  print("It is recommended that this process be stopped and the data errors be corrected")
  print("before continuing")
}

# MAKE A BACKUP COPY!!!
tst_dataV1_backup <- tst_dataV1

tst_dataV2 <- do_feature_engineering(tst_dataV1)

# Make a backup
tst_data_backup2 <- tst_dataV2

tst_dataV3 <- do_factor_conversion(tst_dataV2)

## Backup pre-normalization
tst_data_backup3 <- tst_dataV3 
###############################################################
# Make copy of entire database for normalization
###############################################################

tst_sal_list <- do_standardize_numerics(tst_dataV3)

# Capture and save the mean and standard deviations
# so that predictions can be converted back from scaled numbers
# to real numbers

tst_cur_sal_mean <- tst_sal_list$mean_cur_sal
tst_cur_sal_sd <- tst_sal_list$sd_cur_sal
tst_ret_pct_mean <- tst_sal_list$mean_ret_pct
tst_ret_pct_sd <- tst_sal_list$sd_ret_pct

tst_sal_data <- tst_sal_list$data_frame 

# JobLevel was removed for the training set, so for consistency, remove it again
tst_sal_data <- subset(tst_sal_data, select = -c(JobLevel))

fnl_tst_db <- tst_sal_data[retained_attributes_list]

factor_cols <- names(fnl_tst_db)[sapply(fnl_tst_db, is.factor)]

# Convert the data frame into a data table
fnl_tst_DT <- data.table(fnl_tst_db)

# Convert categoricals to one-hot attributes

fnl_tst_DT <- one_hot(fnl_tst_DT, cols=factor_cols, dropCols = TRUE)

################################################
#
#  BEGIN BASELINE AND NEURAL NETWORK TESTS
#
################################################


################################################
# First run our baseline linear regression and check it
################################################

tst_lin_pred <- predict(trn_linear_reg, fnl_tst_DT)

tst_lm_accuracy <- data.frame(Actual = fnl_tst_DT$RetentionPercentNeeded_Std, 
                              Predicted = tst_lin_pred)
tst_lm_accuracy$Descaled_Actual <- (tst_lm_accuracy$Actual * tst_ret_pct_sd) +
  tst_ret_pct_mean
tst_lm_accuracy$Descaled_Predicted <- (tst_lm_accuracy$Predicted * tst_ret_pct_sd) +
  tst_ret_pct_mean
tst_lm_accuracy$Difference = (abs(tst_lm_accuracy$Descaled_Predicted - tst_lm_accuracy$Descaled_Actual) / 
                                abs(tst_lm_accuracy$Descaled_Actual))*100

tst_lm_residuals <- tst_lm_accuracy$Difference
tst_lm_mse <- mean(tst_lm_residuals^2)
tst_lm_sse <- sum(trn_lm_residuals^2)

tst_std_error_lm <- sqrt(tst_lm_mse)

tst_n <- nrow(fnl_tst_DT)
tst_p <- ncol(fnl_tst_DT)
tst_lm_AIC <- (tst_n*log(tst_lm_sse/tst_n)) + 2*tst_p

# Create boolean fields of prediction within 10, 15 or 20%
tst_lm_accuracy$Within_10_Percent <- tst_lm_accuracy$Difference <= 10
tst_lm_accuracy$Within_15_Percent <- tst_lm_accuracy$Difference <= 15
tst_lm_accuracy$Within_20_Percent <- tst_lm_accuracy$Difference <= 20

# Store the percentages of those variances
tst_lm_percent_within_10 <- mean(tst_lm_accuracy$Within_10_Percent) * 100
tst_lm_percent_within_15 <- mean(tst_lm_accuracy$Within_15_Percent) * 100
tst_lm_percent_within_20 <- mean(tst_lm_accuracy$Within_20_Percent) * 100

tst_lm_plot2 <- ggplot(tst_lm_accuracy, aes(x = Descaled_Actual, y = Descaled_Predicted))+ geom_point() + geom_abline(aes(intercept = 0, slope = 1)) +
  labs(y= "Predicted Retention Percent", x = "True Retention Percent")

tst_lm_plot2

hist(tst_lm_accuracy$Difference, xlab="Percentage Difference (Actual vs Predicted)", 
     main="Testing Data Accuracy of Linear Regression Model", xlim=c(0,60), 
     ylim=c(0,100), breaks=10, col="blue", border="black")

histPercent(tst_lm_accuracy$Difference, xlab="Percentage Difference (Actual vs Predicted)", 
            main="Training Data Accuracy of Linear Regression Model",
            xlim=c(0,60), ylim=c(0,60), col="green", border="black")


ggplot(tst_lm_accuracy, aes(x=Descaled_Actual, y = Descaled_Predicted))  + 
  geom_point(color="red")  +  
  geom_smooth(method="lm")  + 
  ggtitle("Testing Linear Regression - Actual vs Predicted") +
  xlab("Actual") + 
  ylab("Predicted")

################################################
#   Run the TEST Dataset through the Neural Network Model
################################################

predict_tstNN <- predict(train_nn, fnl_tst_DT[,c(1:10)])

# Build dataframe of results

tst_nn_results <- data.frame(actual = fnl_tst_DT$RetentionPercentNeeded_Std, 
                         prediction = predict_tstNN)

colnames(tst_nn_results) <- c("actual","prediction")

# Plot it
tst_NN_plot <- ggplot(tst_nn_results, aes(x = actual, y = prediction))+ geom_point() + geom_abline(aes(intercept = 0, slope = 1)) +
  labs(y= "Predicted Standardized Retention Percent (Test Data)", x = "Real Standardized Retention Percent(Test Data")

tst_NN_plot

# Re-create the orignal Retention Percentage field
destd_tst_ret_pct_actual = (tst_nn_results$actual * tst_ret_pct_sd) + tst_ret_pct_mean
destd_tst_ret_pct_predict = (tst_nn_results$prediction * tst_ret_pct_sd) + tst_ret_pct_mean
destd_tst_ret_pct_diff = (abs(destd_tst_ret_pct_predict - destd_tst_ret_pct_actual) / 
                            abs(destd_tst_ret_pct_actual))*100

# Build a dataframe of "destandardized" results including the difference between
# actual and predicted

tst_nn_accuracy <- data.frame(Actual_Pct = destd_tst_ret_pct_actual,Predict_Pct = destd_tst_ret_pct_predict,
                           Difference = destd_tst_ret_pct_diff)

tst_nn_residuals <- tst_nn_accuracy$Difference
tst_nn_mse <- mean(tst_nn_residuals^2)
tst_nn_sse <- sum(tst_nn_residuals^2)
tst_std_error_nn <- sqrt(tst_nn_mse)
tst_n <- nrow(fnl_tst_DT)
tst_p <- ncol(fnl_tst_DT)
tst_nn_AIC <- (tst_n*log(tst_nn_sse/tst_n)) + 2*tst_p

tst_nn_accuracy$Within_10_Percent <- tst_nn_accuracy$Difference <= 10
tst_nn_accuracy$Within_15_Percent <- tst_nn_accuracy$Difference <= 15
tst_nn_accuracy$Within_20_Percent <- tst_nn_accuracy$Difference <= 20

tst_percent_within_10 <- mean(tst_nn_accuracy$Within_10_Percent) * 100
tst_percent_within_15 <- mean(tst_nn_accuracy$Within_15_Percent) * 100
tst_percent_within_20 <- mean(tst_nn_accuracy$Within_20_Percent) * 100

tst_NN_plot2 <- ggplot(tst_nn_accuracy, aes(x = Actual_Pct, y = Predict_Pct))+ geom_point() + geom_abline(aes(intercept = 0, slope = 1)) +
  labs(y= "Predicted Retention Percent", x = "True Retention Percent")

tst_NN_plot2

histPercent(tst_nn_accuracy$Difference, xlab="Percentage Difference (Actual vs Predicted)", 
            main="Testing Data Accuracy of Neural Network Model",
              xlim=c(0,90), ylim=c(0,60), col="green", border="black")

# Build a little dataframe to show the comparisons
tst_models <- c("Linear Regression","Neural Network")
tst_mse <- c(tst_lm_mse,tst_nn_mse)
tst_std_error <- c(tst_std_error_lm,tst_std_error_nn)
tst_AIC <- c(tst_lm_AIC,tst_nn_AIC)

tst_model_comparison <- data.frame(tst_models,tst_mse,tst_std_error,tst_AIC)
print(tst_model_comparison)



