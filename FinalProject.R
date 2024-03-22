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
library(tidyverse)
library(rsample)    # data splitting 
library(smotefamily)       # SMOTE & Adasyn for balancing datasets


# Data Validator
library(validate)

# Plotting
library(lattice)
library(ggvis)
library(GGally)
library(plotly)
library(skimr)

# Machine Learning
library(neuralnet)
library(NeuralNetTools)


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

library(tidymodels)
library(corpcor)
library(mctest)
library(corrplot)
library(ggcorrplot)
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

# MAKE A BACKUP COPY!!!
sal_dataV1_backup <- sal_dataV1

# OK, now that we've done this, let's start doing some modification and analysis
# First, let's convert the factors to factors...
factor_cols <- c("Education", "EnvironmentSatisfaction", "JobInvolvement",
                 "JobLevel", "JobSatisfaction", "PerformanceRating", 
                 "RelationshipSatisfaction","StockOption","WorkLifeBalance")
sal_dataV1[,factor_cols] <- lapply(sal_dataV1[,factor_cols], factor, ordered= TRUE)

# Get rid of the EMPID field because it is worthless
sal_dataV1 <- subset(sal_dataV1, select = -c(EMPID))

####################################################################
# OK, now that we've done this, let's start doing some modification and analysis
# First, let's convert the factors to factors...
factor_cols <- c("Education", "EnvironmentSatisfaction", "JobInvolvement",
                 "JobLevel", "JobSatisfaction", "PerformanceRating", 
                 "RelationshipSatisfaction","StockOption","WorkLifeBalance")
sal_dataV1[,factor_cols] <- lapply(sal_dataV1[,factor_cols], factor, ordered= TRUE)
#####################################################################
sal_dataV1$AvgTimePerCompany <- round((sal_dataV1$TotalWorkingYears/(sal_dataV1$NumCompaniesWorked+1)),2)

sal_dataV1$CompanyPercentOfCareer <- ifelse(sal_dataV1$TotalWorkingYears == 0, 0, round(((sal_dataV1$YearsAtCompany/sal_dataV1$TotalWorkingYears)*100),2))

sal_dataV1$CurRolePercent <- ifelse(sal_dataV1$YearsAtCompany == 0, 0, round(((sal_dataV1$YearsInCurrentRole/sal_dataV1$YearsAtCompany)*100),2))

sal_dataV1$CurMgrPercent <- ifelse(sal_dataV1$YearsAtCompany == 0, 0, round(((sal_dataV1$YearsWithCurrManager/sal_dataV1$YearsAtCompany)*100),2))

sal_dataV1$NoPromoPercent <- ifelse(sal_dataV1$YearsAtCompany == 0, 0, round(((sal_dataV1$YearsSinceLastPromotion/sal_dataV1$YearsAtCompany)*100),2))

sal_dataV1$RetentionPercentNeeded <- round(((sal_dataV1$DiffFromSalary/sal_dataV1$CurrentSalary)*100),2)

# For our correlation check, remove DiffFromSalary (but keep RetentionPercentNeeded, for now)
sal_dataV1 <- subset(sal_dataV1, select = -c(DiffFromSalary))

# Eliminate AGE because 1) multicollinearity with TotalWorkingYears, etc and
# Also HUGE bias risk
sal_dataV1 <- subset(sal_dataV1, select = -c(Age))

# Let's take a quick peek at the data, including distributions
skim(sal_dataV1)

# Now let's look at the variables with near zero variance
zeroVarVariables <- nearZeroVar(sal_dataV1)

# Show the names of the columns without variability (not useful for modeling)
colnames(sal_dataV1[zeroVarVariables])

# Extract the numeric variables names
numeric_cols <- names(sal_dataV1)[sapply(sal_dataV1, is.numeric)]

################################################################
# Create a Z-Score like standardization routine for the numeric
# attributes
#
# <new x> = (<old x> - <mean X>)/ <std dev x>
# 
###############################################################

data_standardize <- function(x, na.rm= TRUE) {
  return((x - mean(x))/sd(x))
}

# First let's NOT standardize the salary because that has actual meaning
for(looper in numeric_cols) {
  if(looper != 'CurrentSalary' && looper!='AnnualIncomeNeeded')
    {
    sal_dataV1[[looper]] <- data_standardize(sal_dataV1[[looper]]) 
  }
}

# Create a numeric variable dataframe from the main dataframe
sal_num_DF <- sal_dataV1[numeric_cols]

# Create a categorical variable dataframe from the main dataframe
sal_cat_DF <- sal_dataV1[factor_cols]

# Let's set up for correlation plots!
sal_num_corr <- round(cor(sal_num_DF),2)
sal_num_pval <- cor_pmat(sal_num_DF)

# Correlation Plot!
ggcorrplot(sal_num_corr, hc.order=TRUE, type="lower", lab=TRUE, p.mat=sal_num_pval)

target_var <- 'AnnualIncomeNeeded'

# Get a list of the numeric variable names without the target variable
# This will be used for the VIF and Stepwise AIC function

num_cols_no_target <- colnames(sal_num_DF)
num_cols_no_target <- num_cols_no_target[!num_cols_no_target %in% c('AnnualIncomeNeeded','RetentionPercentNeeded')]

# Build the complete formula
salLMformula <- as.formula(paste(target_var,paste(num_cols_no_target, collapse = " + "), sep = " ~"))

# Display the formula
print(salLMformula)

# Perform VIF evalation (Variance Inflation Factor)
VIF_evaluation <- vif(lm(salLMformula, data=sal_dataV1))

# Determine the attribute with the highest VIF value
highest_VIF_location <- which.max(VIF_evaluation)
highest_VIF_index_loc <- highest_VIF_location[1]
highest_VIF_name <- names(highest_VIF_location)

# Now perform the stepwise regression to see which set of attributes
# gives the lowest AIC (Akaike Information Criteria)
# AIC = mathematical evaluation of how well the data fits the model used
#       to generate the model

stepwiseR <- stepAIC(lm(salLMformula,data=sal_dataV1))
summary(stepwiseR)

###################################################################
# Categorical Variable Independence Check!
###################################################################
# Now combine them into every possible set of pairs
factor_pairs <- combn(factor_cols,2)

# Create a function that performs the chi-squared test of independence for a pair of variables
CategoricalIndependenceTest <- function(var1, var2) {
  ContingencyTable <- table(sal_cat_DF[[var1]], sal_cat_DF[[var2]])
  Chi_squared_result <- chisq.test(ContingencyTable)
  return(list(Variable1 = var1, Variable2 = var2, 
              ChiSquaredStatistic = Chi_squared_result$statistic, 
              PValue = Chi_squared_result$p.value))
}

# Run CategoricalIndependenceTest of each pair of variables
IndTestResults <- map2(factor_pairs[1, ], factor_pairs[2, ], CategoricalIndependenceTest)

# Set up blank vectors to hold the data
Ind_Test_Var1 <- rep(NA, length(IndTestResults))
Ind_Test_Var2 <- Ind_Test_Var1
Ind_Test_Chi <- Ind_Test_Var1
Ind_Test_PVal <- Ind_Test_Var1

# Load the vectors
for(MyLoop in 1:length(IndTestResults))
{
  Checkit <- IndTestResults[[MyLoop]]
  Ind_Test_Var1[MyLoop] <- Checkit$Variable1
  Ind_Test_Var2[MyLoop] <- Checkit$Variable2
  Ind_Test_Chi[MyLoop] <- Checkit$ChiSquaredStatistic
  Ind_Test_PVal[MyLoop] <- Checkit$PValue
}

# Convert into a data frame
IndTestResults_df <- data.frame(Ind_Test_Var1,Ind_Test_Var2,Ind_Test_Chi,Ind_Test_PVal)

# Remove the NA values
IndTestResults_df <- IndTestResults_df[complete.cases(IndTestResults_df), ] 

# Examine Results Based on p-values - low p = dependence, high p = independence, cutoff 0.05
IndependentVariables <- IndTestResults_df[IndTestResults_df$Ind_Test_PVal > 0.05, ]
DependentVariables <- IndTestResults_df[IndTestResults_df$Ind_Test_PVal <= 0.05, ]

# Look at the Dependent ones
View(DependentVariables)

