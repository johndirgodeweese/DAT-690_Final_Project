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

library(corpcor)      # 
library(mctest)
library(corrplot)
library(pROC)
library(ROCR)
library(MASS)         # Modern Applied Statistics with S
library(C50)
library(caTools)    # For Linear regression        
library(quantmod)