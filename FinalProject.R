#
# DAT-690 Final Project
# Employee Attrition Project Two
# John Dirgo Deweese
# Start: 3/7/2024
#

# Set up working directory
setwd("G:/My Drive/24TW3 - DAT 690 - Capstone in Data Analytics/DAT-690 Final Project")

# Put libraries you'll need at the start of this file
# It just makes life easier

# Data Wrangling
library(dplyr)
library(tidyverse)
library(rsample)    # data splitting 
library(smotefamily)       # SMOTE & Adasyn for balancing datasets
library(readr)

# Data Validator
library(validate)

# Plotting
library(lattice)
library(ggplot2)
library(ggvis)
library(GGally)
library(plotly)

# Machine Learning

# Training/Testing Dataset Split
library(rsample)

# Oversampling
library(ROSE)

## arules = Association RULES including apriori
library(arules)
library(arulesViz)
library(RColorBrewer)

## Decision Trees and Random Forest
library(rpart)      # performing regression trees
library(rpart.plot) # plotting regression trees
library(ipred)      # bagging
library(randomForest) # basic implementation
library(ranger)       # a faster implementation of randomForest
library(partykit)


# Machine Learning Evaluation
library(caret)        # an aggregator package for performing many machine learning models
library(h2o)          # an extremely fast java-based platform
library(car)          # Companion to Applied Regression

# Feature Selection Algorithm Package
library(Boruta)

library(corpcor)
library(mctest)
library(corrplot)
library(pROC)
library(ROCR)
library(MASS)       # Modern Applied Statistics with S
library(C50)
library(caTools)    # For Linear regression        
library(quantmod)

# Import training data file
sal_dataV1 <- read_csv("data/EmployeeSalary_Data.csv")

# We know that some of these are categorical variables (factors) even thought they are numeric
# but we are not going to use the col_factor option on the read_csv in the beginning
# We have a dictionary of the factors and their values, so we first want to make sure we don't
# have out-of-range data.  Once the data quality has been checked we can convert to a factor
# where that is appropriate.

# Set up validation rules
#
# Rules: 
#
# Check the entire dataframe for NA (!is.na() - returns true if data exists)
# Employees (Age) should be between 18 and 100
# Categorical values coded as numbers should be within their value range
# AverageOvertime must be positive and shouldn't be over 40
# YearsAtCompany must be equal to or less than TotalWorkingYears
# YearsAtCompany must be greater than or equal to YearsInCurrentRole
# YearsAtCompany must be greater than or equal to YearsWithCurrentManager
rules <- validator(CheckNA = !is.na(sal_dataV1),
                   CheckAge = in_range(sal_dataV1$Age, min = 18, max = 100),
                   CheckEducation = in_range(sal_dataV1$Education, min = 1, max = 5),
                   CheckJobInvolve = in_range(sal_dataV1$JobInvolvement, min = 1, max = 4),
                   CheckJobSatis = in_range(sal_dataV1$JobSatisfaction, min = 1, max = 4),
                   CheckPerfRating = in_range(sal_dataV1$PerformanceRating, min = 1, max = 4),
                   CheckRelSatis = in_range(sal_dataV1$RelationshipSatisfaction, min = 1, max = 4),
                   CheckWorkLife = in_range(sal_dataV1$WorkLifeBalance, min = 1, max = 4),
                   CheckAvgOT = in_range(sal_dataV1$AvgOverTime, min = 0, max = 40),
                   CheckTotWorkVsCompany = sal_dataV1$YearsAtCompany <= sal_dataV1$TotalWorkingYears,
                   CheckYrsCompVsCurRole = sal_dataV1$YearsAtCompany >= sal_dataV1$YearsInCurrentRole,
                   CheckYrsCompVsCurMgt = sal_dataV1$YearsAtCompany >= sal_dataV1$YearsWithCurrManager)

# 'Confront' the data with rules, save results
rule_check <- confront(sal_dataV1, rules)

# What do the results say?

summary(rule_check)

