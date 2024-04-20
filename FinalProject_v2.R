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
library(mltools)
library(data.table)
library(skimr)
library(rstatix)

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
library(neuralnet)
library(NeuralNetTools)
library(keras3)
library(randomForest)

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
sal_dataV1 <- read_csv("data/EmployeeSalary_Data.csv", show_col_types = FALSE)

# We know that some of these are categorical variables (factors) even thought they are numeric
# but we are not going to use the col_factor option on the read_csv in the beginning
# We have a dictionary of the factors and their values, so we first want to make sure we don't
# have out-of-range data.  Once the data quality has been checked we can convert to a factor
# where that is appropriate.

num_validation_errors <- do_validation_check(sal_dataV1)
if(num_validation_errors == 0) {
  print("All validation checks passed without error")  
} else {
  print("There were",sum(validation_check$fails,"errors in the data validation"))
  print("It is recommended that this process be stopped and the data checked for errors")
}

# MAKE A BACKUP COPY!!!
sal_dataV1_backup <- sal_dataV1

sal_dataV2 <- do_feature_engineering(sal_dataV1)

multi_col <- do_multicollinearity_check(sal_dataV2)

# Print the significant correlations (row and column indices
multi_col1 <- multi_col[order(multi_col$p, decreasing=FALSE),]

print(multi_col1)

# Make a backup
sal_data_backup2 <- sal_dataV2

sal_dataV3 <- do_factor_conversion(sal_dataV2)

## Backup pre-normalization
sal_data_backup3 <- sal_dataV3 
###############################################################
# Make copy of entire database for normalization
###############################################################

################################################################
# Create a Z-Score like standardization routine for the numeric
# attributes
#
# <new x> = (<old x> - <mean X>)/ <std dev x>
# 
###############################################################
train_sal_data <- sal_dataV3

train_sal_data <- do_standardize_numerics(train_sal_data)

###################################################################
# Categorical Variable Independence Check!
###################################################################

DependentVariables <- do_categorical_independence_check(train_sal_data)
# Look at the Dependent ones
View(DependentVariables)

# # Since JobLevel appear in both of the dependent pairs, remove it
train_sal_data <- subset(train_sal_data, select = -c(JobLevel))

########################################################
# Random Forest Model (for Feature Elimination)
########################################################

RF_Model_All <- randomForest::randomForest(RetentionPercentNeeded ~ ., data=train_sal_data)

RF_Model_All


boruta.train <- Boruta::Boruta(RetentionPercentNeeded ~ ., data=train_sal_data, doTrace=1)

print(boruta.train)

boruta.df <- attStats(boruta.train)

retained_attributes <- boruta.df[boruta.df$decision == "Confirmed", ]

retained_attributes_list <- rownames(retained_attributes)

retained_attributes_list <- c(retained_attributes_list,"RetentionPercentNeeded")

redacted_train_db <- train_sal_data[retained_attributes_list]

factor_cols <- names(redacted_train_db)[sapply(redacted_train_db, is.factor)]

# Convert the data frame into a data table
redacted_train_DT <- data.table(redacted_train_db)

# ML tools? one_hot?
redacted_train_DT <- one_hot(redacted_train_DT, cols=factor_cols, dropCols = TRUE)

train_linear_reg <- lm(RetentionPercentNeeded ~ ., data = redacted_train_DT)

summary(train_linear_reg)

train_logistic_reg <- glm(RetentionPercentNeeded ~ ., data = redacted_train_DT, family=gaussian)

summary(train_logistic_reg)

linear_AIC <- extractAIC(train_linear_reg)
logistic_AIC <- extractAIC(train_logistic_reg)

##########################################################
# Linear is better but still shitty
##########################################################

# Let's do a basic neural network
train_nn <- neuralnet(RetentionPercentNeeded ~ ., data = redacted_train_DT, 
                      hidden = c(10, 5), linear.output=TRUE,threshold = 1)

train_nn$result.matrix

nn_results <- data.frame(actual = redacted_train_DT$RetentionPercentNeeded, 
                         prediction = train_nn$net.result)

colnames(nn_results) <- c("actual","prediction")

predicted=nn_results$prediction * abs(diff(range(redacted_train_DT$RetentionPercentNeeded))) + 
  min(redacted_train_DT$RetentionPercentNeeded, na.rm=TRUE)
actual=nn_results$actual * abs(diff(range(redacted_train_DT$RetentionPercentNeeded))) + 
  min(redacted_train_DT$RetentionPercentNeeded, na.rm=TRUE)
comparison=data.frame(predicted,actual)
deviation=((actual-predicted)/actual)
comparison=data.frame(predicted,actual,deviation)
accuracy=1-abs(mean(deviation))
accuracy

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


#################################################################################
######################### Testing Dataset Preparation ###########################
#################################################################################

# Import testing data file
test_dataV1 <- read_csv("data/EmployeeSalary_Verify.csv", show_col_types = FALSE)

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
rules <- validator(CheckNA = !is.na(test_dataV1),
                   CheckAge = in_range(test_dataV1$Age, min = 18, max = 100),
                   CheckEducation = in_range(test_dataV1$Education, min = 1, max = 5),
                   CheckJobInvolve = in_range(test_dataV1$JobInvolvement, min = 1, max = 4),
                   CheckJobSatis = in_range(test_dataV1$JobSatisfaction, min = 1, max = 4),
                   CheckPerfRating = in_range(test_dataV1$PerformanceRating, min = 1, max = 4),
                   CheckRelSatis = in_range(test_dataV1$RelationshipSatisfaction, min = 1, max = 4),
                   CheckWorkLife = in_range(test_dataV1$WorkLifeBalance, min = 1, max = 4),
                   CheckStockOpt = in_range(test_dataV1$StockOption, min = 0, max =3),
                   CheckJobLevel = in_range(test_dataV1$JobLevel, min = 1, max = 5),
                   CheckEnvSatis = in_range(test_dataV1$EnvironmentSatisfaction, min = 1, max = 4),
                   CheckAvgOT = in_range(test_dataV1$AvgOverTime, min = 0, max = 40),
                   CheckTotWorkVsCompany = test_dataV1$YearsAtCompany <= test_dataV1$TotalWorkingYears,
                   CheckYrsCompVsCurRole = test_dataV1$YearsAtCompany >= test_dataV1$YearsInCurrentRole,
                   CheckYrsCompVsCurMgt = test_dataV1$YearsAtCompany >= test_dataV1$YearsWithCurrManager)

# 'Confront' the data with rules, save results
rule_check <- confront(test_dataV1, rules)

# What do the results say?

summary(rule_check)

# MAKE A BACKUP COPY!!!
test_dataV1_backup <- test_dataV1

# Get rid of the EMPID field because it is worthless
test_dataV1 <- subset(test_dataV1, select = -c(EMPID))

####################################################################
# OK, now that we've done this, let's start doing some modification and analysis
# First, let's convert the factors to factors...

factor_cols <- c("Education", "EnvironmentSatisfaction", "JobInvolvement",
                 "JobLevel", "JobSatisfaction", "PerformanceRating", 
                 "RelationshipSatisfaction","StockOption","WorkLifeBalance")


test_dataV1$Education <- factor(test_dataV1$Education, 
                               levels=c("1","2","3","4","5"),
                               labels=c("Below_College",
                                        "College",
                                        "Bachelor",
                                        "Master",
                                        "Doctor"), 
                               ordered=TRUE)
test_dataV1$EnvironmentSatisfaction <- factor(test_dataV1$EnvironmentSatisfaction, 
                                             levels=c("1","2","3","4"),
                                             labels=c("Low",
                                                      "Medium",
                                                      "High",
                                                      "Very_High"),
                                             ordered=TRUE)
test_dataV1$JobInvolvement <- factor(test_dataV1$JobInvolvement, 
                                    levels=c("1","2","3","4"),
                                    labels=c("Low",
                                             "Medium",
                                             "High",
                                             "Very_High"),
                                    ordered=TRUE)
test_dataV1$JobSatisfaction <- factor(test_dataV1$JobSatisfaction, 
                                     levels=c("1","2","3","4"),
                                     labels=c("Low",
                                              "Medium",
                                              "High",
                                              "Very_High"),
                                     ordered=TRUE)
test_dataV1$PerformanceRating <- factor(test_dataV1$PerformanceRating, 
                                       levels=c("1","2","3","4"),
                                       labels=c("Low",
                                                "Medium",
                                                "High",
                                                "Very_High"),
                                       ordered=TRUE)
test_dataV1$RelationshipSatisfaction <- factor(test_dataV1$RelationshipSatisfaction, 
                                              levels=c("1","2","3","4"),
                                              labels=c("Low",
                                                       "Medium",
                                                       "High",
                                                       "Very_High"),
                                              ordered=TRUE)
test_dataV1$WorkLifeBalance <- factor(test_dataV1$WorkLifeBalance, 
                                     levels=c("1","2","3","4"),
                                     labels=c("Low",
                                              "Medium",
                                              "High",
                                              "Very_High"),
                                     ordered=TRUE)
###################
# For the next/final 2 factors, we don't know what the level means, just what they are
###################
test_dataV1$JobLevel <- factor(test_dataV1$JobLevel, 
                              levels=c("1","2","3","4","5"),
                              ordered=TRUE)
test_dataV1$StockOption <- factor(test_dataV1$StockOption, 
                                 levels=c("0","1","2","3"),
                                 ordered=TRUE)

#####################################################################
# Quickly do a quick histogram for the factor variables
#####################################################################
# Get a list of them to loop through

factor_cols <- names(test_dataV1)[sapply(test_dataV1, is.factor)]
num_factor_cols <- length(factor_cols)

for(looper in factor_cols) {
  # Create names for the new temp dataframe and plots
  graph_name <- paste(looper,"Plot", sep="_")
  frame_name <- paste(looper,"Frame", sep="_")
  # Get the counts for each level
  level_counts <- table(test_dataV1[[looper]])
  temp_graph <- ggplot(data.frame(level_counts), aes(x = Var1, y = Freq)) +
    geom_bar(stat = "identity", fill='cadetblue') +
    geom_text(aes(x=Var1, y=Freq,label=Freq)) +
    # Use stat="identity" to keep original counts
    labs(title = paste("Counts of ",looper, " Levels", sep=""), x = "Level", y = "Count") +
    theme_classic()
  assign(graph_name,temp_graph)
}

grid.arrange(Education_Plot,EnvironmentSatisfaction_Plot,JobInvolvement_Plot,
             JobLevel_Plot,JobSatisfaction_Plot,RelationshipSatisfaction_Plot,
             PerformanceRating_Plot,StockOption_Plot,WorkLifeBalance_Plot,
             ncol=3)


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

test_dataV1$AvgTimePerCompany <- round((test_dataV1$TotalWorkingYears/(test_dataV1$NumCompaniesWorked+1)),2)

test_dataV1$CompanyPercentOfCareer <- ifelse(test_dataV1$TotalWorkingYears == 0, 0, round(((test_dataV1$YearsAtCompany/test_dataV1$TotalWorkingYears)*100),2))

test_dataV1$CurRolePercent <- ifelse(test_dataV1$YearsAtCompany == 0, 0, round(((test_dataV1$YearsInCurrentRole/test_dataV1$YearsAtCompany)*100),2))

test_dataV1$CurMgrPercent <- ifelse(test_dataV1$YearsAtCompany == 0, 0, round(((test_dataV1$YearsWithCurrManager/test_dataV1$YearsAtCompany)*100),2))

test_dataV1$NoPromoPercent <- ifelse(test_dataV1$YearsAtCompany == 0, 0, round(((test_dataV1$YearsSinceLastPromotion/test_dataV1$YearsAtCompany)*100),2))

test_dataV1$RetentionPercentNeeded <- round(((test_dataV1$DiffFromSalary/test_dataV1$CurrentSalary)*100),2)


# For our correlation check, remove DiffFromSalary and AnnualIncomeNeeded
# Can always recreate using the current Salary and RetentionPercentNeeded
test_dataV1 <- subset(test_dataV1, select = -c(DiffFromSalary, AnnualIncomeNeeded))

# Eliminate AGE because 1) multicollinearity with TotalWorkingYears, etc and
# Also HUGE bias risk
test_dataV1 <- subset(test_dataV1, select = -c(Age))

# Let's take a quick peek at the data, including distributions
skim(test_dataV1)

# Now let's look at the variables with near zero variance
zeroVarVariables <- nearZeroVar(test_dataV1)

# Show the names of the columns without variability (not useful for modeling)
colnames(test_dataV1[zeroVarVariables])

# In this case, there are none so retain all the attributes

# Let's start some other graphs


# Scatterplot of RetentionPercentage vs Salary
BonusVSal <- ggplot(test_dataV1, aes(y=RetentionPercentNeeded,x=CurrentSalary,
                                    color=factor(JobLevel))) + geom_point(size=2.5)
BonusVEdu <- ggplot(test_dataV1, aes(Education,RetentionPercentNeeded, fill=JobLevel)) + 
  geom_dotplot(binaxis="y",stackdir="center",dotsize=1)



###############################################################
# Make copy of entire database for normalization
###############################################################
test_sal_data <- test_dataV1

################################################################
# Create a Z-Score like standardization routine for the numeric
# attributes
#
# <new x> = (<old x> - <mean X>)/ <std dev x>
# 
###############################################################

# Extract the numeric variables names
numeric_cols <- names(test_dataV1)[sapply(test_dataV1, is.numeric)]

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
    new_vector <- data_standardize(test_dataV1[[looper]])
    # Add the vector to the dataframe
    test_sal_data <- cbind(test_sal_data,new_vector)
    # Rename the column
    colnames(test_sal_data)[which(names(test_sal_data) == "new_vector")] <- newcol_name
    # Remove the old non-standardized data (NEW DATAFRAME ONLY)
    test_sal_data <- test_sal_data[, names(test_sal_data) != looper]
  }
}

# Create a numeric variable dataframe from the modeling dataframe
new_numeric_cols <- names(test_sal_data)[sapply(test_sal_data, is.numeric)]

# nDF at the end indicates numeric dataframe
test_sal_data_nDF <- test_sal_data[new_numeric_cols]


# Create a categorical variable dataframe from the modeling dataframe
#new_factor_cols <- names(test_sal_data)[sapply(test_sal_data, is.factor)]

# fDF at the end indicates factor dataframe
#test_sal_data_fDF <- test_sal_data[new_factor_cols]

#test_sal_data_fDF <- cbind(test_sal_data_fDF,test_sal_data$RetentionPercentNeeded)

#names(test_sal_data_fDF)[names(test_sal_data_fDF) == 'test_sal_data$RetentionPercentNeeded'] <- 'RetentionPercentNeeded'

# Let's set up for correlation plots!


sal_num_corr <- round(cor(test_sal_data_nDF),2)
sal_num_pval <- cor_pmat(test_sal_data_nDF)
#sal_fac_corr <- round(cor(test_sal_data_fDF, method='spearman'),2)

# Assuming your data is in a dataframe named 'data' with an ordinal column 'ordinal_attribute' and a continuous column 'target_variable'
#spearman_rho <- cor(test_sal_data$RetentionPercentNeeded, test_sal_data$Education, method = "spearman")

# spearman_rho will contain the correlation coefficient

#sal_fac_pval <- cor_pmat(test_sal_data_fDF)

# Correlation Plot!
ggcorrplot(sal_num_corr, hc.order=TRUE, type="lower", lab=TRUE, p.mat=sal_num_pval)
#ggcorrplot(sal_fac_corr, hc.order=TRUE, type="lower", lab=TRUE, p.mat=sal_fac_pval)


target_var <- 'RetentionPercentNeeded'

# Get a list of the numeric variable names without the target variable
# This will be used for the VIF and Stepwise AIC function

num_cols_no_target <- colnames(test_sal_data_nDF)
num_cols_no_target <- num_cols_no_target[!num_cols_no_target %in% c('RetentionPercentNeeded')]

# Build the complete formula
salLMformula <- as.formula(paste(target_var,paste(num_cols_no_target, collapse = " + "), sep = " ~"))

# Display the formula
print(salLMformula)

# Perform VIF evalation (Variance Inflation Factor)
VIF_evaluation <- vif(lm(salLMformula, data=test_sal_data_nDF))

# Determine the attribute with the highest VIF value
highest_VIF_location <- which.max(VIF_evaluation)
highest_VIF_index_loc <- highest_VIF_location[1]
highest_VIF_name <- names(highest_VIF_location)

# Now perform the stepwise regression to see which set of attributes
# gives the lowest AIC (Akaike Information Criteria)
# AIC = mathematical evaluation of how well the data fits the model used
#       to generate the model

stepwiseR <- MASS::stepAIC(lm(salLMformula,data=test_sal_data_nDF, direction="both"))
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
new_factor_cols <- names(test_sal_data)[sapply(test_sal_data, is.factor)]

# fDF at the end indicates factor dataframe
test_sal_data_fDF <- test_sal_data[new_factor_cols]


# Now combine them into every possible set of pairs
factor_pairs <- combn(new_factor_cols,2)

# Create a function that performs the chi-squared test of independence for a pair of variables
CategoricalIndependenceTest <- function(var1, var2) {
  ContingencyTable <- table(test_sal_data_fDF[[var1]], test_sal_data_fDF[[var2]])
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
test_sal_data_DT <- data.table(test_sal_data)

# ML tools? one_hot?
test_sal_data_DT <- one_hot(test_sal_data_DT, cols=new_factor_cols, dropCols = TRUE)

# Build final training database
# Assembled from the attributes chosen in the stepwise regression (for numerics)
# and the independence check (categoricals)
test_db_field_list <- c("RetentionPercentNeeded",selected_features,new_factor_cols)

test_final_db <- test_sal_data[,c(test_db_field_list)]

# Convert the data frame into a data table
test_final_db <- data.table(test_final_db)

# Run the one-hot encoding on the factor columns
test_final_db <- one_hot(test_final_db, cols=new_factor_cols, dropCols = TRUE)

test_predictions <- predict(train_linear_reg, test_final_db)
# Build Data Frame of Predicted versus Actual
test_actual_pred_lm <- data.frame(cbind(actuals=test_final_db$RetentionPercentNeeded,
                                        predicted=test_predictions))

correlation_accuracy <- cor(test_actual_pred_lm)
print(correlation_accuracy)
mape <- mape(test_actual_pred_lm,actuals,predicted)
print(mape)

plot(train_nn)

predict_testNN = compute(train_nn, test_final_db)
predict_testNN_2 = (predict_testNN$net.result * (max(test_final_db$RetentionPercentNeeded) - 
                                                 min(test_final_db$RetentionPercentNeeded))) + min(test_final_db$RetentionPercentNeeded)

NN_Test_DF <- data.frame(cbind(actual= test_final_db$RetentionPercentNeeded,
                               predicted = predict_testNN$net.result))

NN_plot <- ggplot(NN_Test_DF, aes(x = actual, y = V2))+ geom_point() + geom_abline(aes(intercept = 0, slope = 1)) +
  labs(y= "Predicted Percent NN", x = "Real Percent")

NN_plot

abline(0,1)