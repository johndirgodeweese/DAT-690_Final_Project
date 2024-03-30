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

#### Data Wrangling
# Tidyverse contains ggplot2, tibble, tidyr, readr, 
#    dplyr, stringr, purr, and forcats
library(tidyverse)

#library(smotefamily)       # SMOTE & Adasyn for balancing datasets
#library(e1071)

library(mltools)
library(data.table)
library(skimr)

# Data Validator
library(validate)

# Plotting
library(lattice)
library(ggvis)
#library(GGally)
library(plotly)

# Machine Learning
library(neuralnet)
library(NeuralNetTools)
library(keras3)

# Oversampling
#library(ROSE)

## arules = Association RULES including apriori
#library(arules)
#library(arulesViz)
library(RColorBrewer)

## Decision Trees and Random Forest
# library(rpart)      # performing regression trees
# library(rpart.plot) # plotting regression trees
# library(ipred)      # bagging
# library(randomForest) # basic implementation
# library(ranger)       # a faster implementation of randomForest
# library(partykit)


# Machine Learning Evaluation
library(caret)        # an aggregator package for performing many machine learning models
library(h2o)          # an extremely fast java-based platform
library(car)          # Companion to Applied Regression

# Feature Selection Algorithm Package
#library(Boruta)

# Tidymodels includes broom, dials, infer, modeldata, parsnip, recipes,
#     rsample, tune, workflows, workflowsets, yardstick
library(tidymodels)
library(corpcor)
library(mctest)
library(corrplot)
library(ggcorrplot)
library(pROC)
library(ROCR)
library(MASS)       # Modern Applied Statistics with S
#library(C50)
library(caTools)    # For Linear regression        
#library(quantmod)

# Import training data file
sal_dataV1 <- read_csv("data/EmployeeSalary_Data.csv", show_col_types = FALSE)

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
                   CheckStockOpt = in_range(sal_dataV1$StockOption, min = 0, max =3),
                   CheckJobLevel = in_range(sal_dataV1$JobLevel, min = 1, max = 5),
                   CheckEnvSatis = in_range(sal_dataV1$EnvironmentSatisfaction, min = 1, max = 4),
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

# Get rid of the EMPID field because it is worthless
sal_dataV1 <- subset(sal_dataV1, select = -c(EMPID))

####################################################################
# OK, now that we've done this, let's start doing some modification and analysis
# First, let's convert the factors to factors...

factor_cols <- c("Education", "EnvironmentSatisfaction", "JobInvolvement",
                 "JobLevel", "JobSatisfaction", "PerformanceRating", 
                 "RelationshipSatisfaction","StockOption","WorkLifeBalance")

#sal_dataV1[,factor_cols] <- lapply(sal_dataV1[,factor_cols], factor, ordered= TRUE)

sal_dataV1$Education <- factor(sal_dataV1$Education, 
                              levels=c("1","2","3","4","5"),
                              labels=c("Below College",
                                "College",
                                "Bachelor",
                                "Master",
                                "Doctor"), 
                              ordered=TRUE)
sal_dataV1$EnvironmentSatisfaction <- factor(sal_dataV1$Education, 
                                            levels=c("1","2","3","4"),
                                            labels=c("Low",
                                              "Medium",
                                              "High",
                                              "Very High"),
                                            ordered=TRUE)
sal_dataV1$JobInvolvement <- factor(sal_dataV1$JobInvolvement, 
                                             levels=c("1","2","3","4"),
                                             labels=c("Low",
                                                      "Medium",
                                                      "High",
                                                      "Very High"),
                                             ordered=TRUE)
sal_dataV1$JobSatisfaction <- factor(sal_dataV1$JobSatisfaction, 
                                             levels=c("1","2","3","4"),
                                             labels=c("Low",
                                                      "Medium",
                                                      "High",
                                                      "Very High"),
                                             ordered=TRUE)
sal_dataV1$PerformanceRating <- factor(sal_dataV1$PerformanceRating, 
                                     levels=c("1","2","3","4"),
                                     labels=c("Low",
                                              "Medium",
                                              "High",
                                              "Very High"),
                                     ordered=TRUE)
sal_dataV1$RelationshipSatisfaction <- factor(sal_dataV1$RelationshipSatisfaction, 
                                     levels=c("1","2","3","4"),
                                     labels=c("Low",
                                              "Medium",
                                              "High",
                                              "Very High"),
                                     ordered=TRUE)
sal_dataV1$WorkLifeBalance <- factor(sal_dataV1$WorkLifeBalance, 
                                     levels=c("1","2","3","4"),
                                     labels=c("Low",
                                              "Medium",
                                              "High",
                                              "Very High"),
                                     ordered=TRUE)
###################
# For the next/final 2 factors, we don't know what the level means, just what they are
###################
sal_dataV1$JobLevel <- factor(sal_dataV1$JobLevel, 
                                     levels=c("1","2","3","4","5"),
                                     ordered=TRUE)
sal_dataV1$StockOption <- factor(sal_dataV1$StockOption, 
                              levels=c("0","1","2","3"),
                              ordered=TRUE)

#####################################################################
# Quickly do a quick histogram for the factor variables
#####################################################################

# Assuming your data is in a dataframe named 'data' with a factor column 'factor_attribute'
ggplot(sal_dataV1, aes(x = Education)) + geom_bar(fill="blue") +
  scale_fill_manual(labels=c("Below College","College","Bachelor","Master","Doctor"))


#####################################################################
# Now Create Some New Attributes That are More Useful
# Average Time Per Company = Total Working Years /(Number Companies Worked + 1)
#             Apparently 'Number of Companies Worked' doesn't include this one
# Company Percent of Career = Years at Company / Total Working Years
# Current Role Percent = Years in Current Role / Years at Company
# Current Manager Percent = Years with Current Manager / Years at Company
# Percent Since Last Promotion = Years since Last Promotion / Year at Company
# Retention Percent Needed = Amount of Salary Change Needed / Current Salary
#####################################################################

sal_dataV1$AvgTimePerCompany <- round((sal_dataV1$TotalWorkingYears/(sal_dataV1$NumCompaniesWorked+1)),2)

sal_dataV1$CompanyPercentOfCareer <- ifelse(sal_dataV1$TotalWorkingYears == 0, 0, round(((sal_dataV1$YearsAtCompany/sal_dataV1$TotalWorkingYears)*100),2))

sal_dataV1$CurRolePercent <- ifelse(sal_dataV1$YearsAtCompany == 0, 0, round(((sal_dataV1$YearsInCurrentRole/sal_dataV1$YearsAtCompany)*100),2))

sal_dataV1$CurMgrPercent <- ifelse(sal_dataV1$YearsAtCompany == 0, 0, round(((sal_dataV1$YearsWithCurrManager/sal_dataV1$YearsAtCompany)*100),2))

sal_dataV1$NoPromoPercent <- ifelse(sal_dataV1$YearsAtCompany == 0, 0, round(((sal_dataV1$YearsSinceLastPromotion/sal_dataV1$YearsAtCompany)*100),2))

sal_dataV1$RetentionPercentNeeded <- round(((sal_dataV1$DiffFromSalary/sal_dataV1$CurrentSalary)*100),2)


# For our correlation check, remove DiffFromSalary (but keep RetentionPercentNeeded, for now)
sal_dataV1 <- subset(sal_dataV1, select = -c(DiffFromSalary, AnnualIncomeNeeded))

# Eliminate AGE because 1) multicollinearity with TotalWorkingYears, etc and
# Also HUGE bias risk
sal_dataV1 <- subset(sal_dataV1, select = -c(Age))

# Let's take a quick peek at the data, including distributions
skim(sal_dataV1)

# Now let's look at the variables with near zero variance
zeroVarVariables <- nearZeroVar(sal_dataV1)

# Show the names of the columns without variability (not useful for modeling)
colnames(sal_dataV1[zeroVarVariables])

# In this case, there are none so retain all the attributes

###############################################################
# Make copy of entire database for normalization
###############################################################
train_sal_data <- sal_dataV1

################################################################
# Create a Z-Score like standardization routine for the numeric
# attributes
#
# <new x> = (<old x> - <mean X>)/ <std dev x>
# 
###############################################################

# Extract the numeric variables names
numeric_cols <- names(sal_dataV1)[sapply(sal_dataV1, is.numeric)]

# 
data_standardize <- function(x, na.rm= TRUE) {
  return((x - mean(x))/sd(x))
}

# First let's NOT standardize the salary or the target variable because 
# they are money and have actual meaning
for(looper in numeric_cols) {
  if(looper != 'CurrentSalary' && looper!='RetentionPercentNeeded') {
    # Create a new name for the created standardized column    
    newcol_name <- paste(looper,"Std", sep="_")
    # Create a vector containing the standardized values
    new_vector <- data_standardize(sal_dataV1[[looper]])
    # Add the vector to the dataframe
    train_sal_data <- cbind(train_sal_data,new_vector)
    # Rename the column
    colnames(train_sal_data)[which(names(train_sal_data) == "new_vector")] <- newcol_name
    # Remove the old non-standardized data (NEW DATAFRAME ONLY)
    train_sal_data <- train_sal_data[, names(train_sal_data) != looper]
  }
}

# Create a numeric variable dataframe from the modeling dataframe
new_numeric_cols <- names(train_sal_data)[sapply(train_sal_data, is.numeric)]

# nDF at the end indicates numeric dataframe
train_sal_data_nDF <- train_sal_data[new_numeric_cols]


# Let's set up for correlation plots!
sal_num_corr <- round(cor(train_sal_data_nDF),2)
sal_num_pval <- cor_pmat(train_sal_data_nDF)

# Correlation Plot!
ggcorrplot(sal_num_corr, hc.order=TRUE, type="lower", lab=TRUE, p.mat=sal_num_pval)

target_var <- 'RetentionPercentNeeded'

# Get a list of the numeric variable names without the target variable
# This will be used for the VIF and Stepwise AIC function

num_cols_no_target <- colnames(train_sal_data_nDF)
num_cols_no_target <- num_cols_no_target[!num_cols_no_target %in% c('RetentionPercentNeeded')]

# Build the complete formula
salLMformula <- as.formula(paste(target_var,paste(num_cols_no_target, collapse = " + "), sep = " ~"))

# Display the formula
print(salLMformula)

# Perform VIF evalation (Variance Inflation Factor)
VIF_evaluation <- vif(lm(salLMformula, data=train_sal_data_nDF))

# Determine the attribute with the highest VIF value
highest_VIF_location <- which.max(VIF_evaluation)
highest_VIF_index_loc <- highest_VIF_location[1]
highest_VIF_name <- names(highest_VIF_location)

# Now perform the stepwise regression to see which set of attributes
# gives the lowest AIC (Akaike Information Criteria)
# AIC = mathematical evaluation of how well the data fits the model used
#       to generate the model

stepwiseR <- MASS::stepAIC(lm(salLMformula,data=train_sal_data_nDF, direction="both"))
summary(stepwiseR)

###
# Save the AIC selected features
###
coefficients <- coef(stepwiseR)
# Ignore intercept (that's why it starts at 2)
selected_features <- names(coefficients)[coefficients != 0][2:length(coefficients)]

###################################################################
# Categorical Variable Independence Check!
###################################################################
# Create a categorical variable dataframe from the modeling dataframe
new_factor_cols <- names(train_sal_data)[sapply(train_sal_data, is.factor)]

# fDF at the end indicates factor dataframe
train_sal_data_fDF <- train_sal_data[new_factor_cols]


# Now combine them into every possible set of pairs
factor_pairs <- combn(new_factor_cols,2)

# Create a function that performs the chi-squared test of independence for a pair of variables
CategoricalIndependenceTest <- function(var1, var2) {
  ContingencyTable <- table(train_sal_data_fDF[[var1]], train_sal_data_fDF[[var2]])
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

# Since JobLevel appear in both of the dependent pairs, remove it
new_factor_cols <- new_factor_cols[!new_factor_cols %in% c('JobLevel')]

# Convert the data frame into a data table
train_sal_data_DT <- data.table(train_sal_data)

# ML tools? one_hot?
train_sal_data_DT <- one_hot(train_sal_data_DT, cols=new_factor_cols, dropCols = TRUE)

# Build final training database
# Assembled from the attributes chosen in the stepwise regression (for numerics)
# and the independence check (categoricals)
train_db_field_list <- c("RetentionPercentNeeded",selected_features,new_factor_cols)

train_final_db <- train_sal_data[,c(train_db_field_list)]

# Convert the data frame into a data table
train_final_db <- data.table(train_final_db)

# Run the one-hot encoding on the factor columns
train_final_db <- one_hot(train_final_db, cols=new_factor_cols, dropCols = TRUE)

# Let's try a basic linear regression model
train_linear_reg <- lm(RetentionPercentNeeded ~ ., data = train_final_db)

# Let's try a GLM model
train_glm <- glm(RetentionPercentNeeded ~ ., data = train_final_db, family = gaussian)

# Let's do a basic neural network
train_nn <- neuralnet(RetentionPercentNeeded ~ ., data = train_final_db, 
                      hidden = c(10, 5), linear.output = FALSE)

# Plot that bad boy out!
plot(train_nn,rep="best")

## Let's do some diagnostics and visualization of the LR model
library(inspectdf)
library(janitor)
library(ggrepel)
library(qqplotr)

# Let's 'tidy' up our linear model
tidy(train_linear_reg)

# Add some additional useful fields
augment(train_linear_reg)

# And look at the overall model stats
glance(train_linear_reg)

# Add the column augments to see how the Q-Q plot looks
aug_lm_sal <- broom::augment_columns(train_linear_reg, data = train_final_db)

aug_lm_sal %>% dplyr::select(contains(".")) %>% dplyr::glimpse(78)

# Graph it!
gg_lm_sal_QQPlot <- train_final_db %>%
  # name the 'sample' the outcome variable (norm_y)
  ggplot(mapping = aes(sample = RetentionPercentNeeded)) +
  # add the stat_qq_band
  qqplotr::stat_qq_band(
    bandType = "pointwise",
    mapping = aes(fill = "Normal"), alpha = 0.5,
    show.legend = FALSE
  ) +
  # add the lines
  qqplotr::stat_qq_line() +
  # add the points
  qqplotr::stat_qq_point() +
  # add labs
  ggplot2::labs(
    x = "Theoretical Quantiles",
    y = "Sample Residuals",
    title = "Normal Q-Q plot for Salary Data"
  )
gg_lm_sal_QQPlot
