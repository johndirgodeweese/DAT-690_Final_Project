do_validation_check <- function(data_frame) {
  data_frame <- trn_dataV1
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
  rules <- validate::validator(CheckNA = !is.na(data_frame),
                               CheckAge = in_range(data_frame$Age, min = 18, max = 100),
                               CheckEducation = in_range(data_frame$Education, min = 1, max = 5),
                               CheckJobInvolve = in_range(data_frame$JobInvolvement, min = 1, max = 4),
                               CheckJobSatis = in_range(data_frame$JobSatisfaction, min = 1, max = 4),
                               CheckPerfRating = in_range(data_frame$PerformanceRating, min = 1, max = 4),
                               CheckRelSatis = in_range(data_frame$RelationshipSatisfaction, min = 1, max = 4),
                               CheckWorkLife = in_range(data_frame$WorkLifeBalance, min = 1, max = 4),
                               CheckStockOpt = in_range(data_frame$StockOption, min = 0, max =3),
                               CheckJobLevel = in_range(data_frame$JobLevel, min = 1, max = 5),
                               CheckEnvSatis = in_range(data_frame$EnvironmentSatisfaction, min = 1, max = 4),
                               CheckAvgOT = in_range(data_frame$AvgOverTime, min = 0, max = 40),
                               CheckTotWorkVsCompany = data_frame$YearsAtCompany <= data_frame$TotalWorkingYears,
                               CheckYrsCompVsCurRole = data_frame$YearsAtCompany >= data_frame$YearsInCurrentRole,
                               CheckYrsCompVsCurMgt = data_frame$YearsAtCompany >= data_frame$YearsWithCurrManager)
  
  # 'Confront' the data with rules, save results
  rule_check <- validate::confront(data_frame, rules)
  
  # What do the results say?
  
  validation_check <- summary(rule_check)
  
  return(list(validation_errors = sum(validation_check$fails),validation_warnings = (sum(validation_check$warning, na.rm = TRUE) - 1)))
}  

do_feature_engineering <- function(data_frame) {
  
  # Get rid of the EMPID field because it is worthless
  data_frame <- subset(data_frame, select = -c(EMPID))
  
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
  
  data_frame$AvgTimePerCompany <- round((data_frame$TotalWorkingYears/(data_frame$NumCompaniesWorked+1)),2)
  
  data_frame$CompanyPercentOfCareer <- ifelse(data_frame$TotalWorkingYears == 0, 0, round(((data_frame$YearsAtCompany/data_frame$TotalWorkingYears)*100),2))
  
  data_frame$CurRolePercent <- ifelse(data_frame$YearsAtCompany == 0, 0, round(((data_frame$YearsInCurrentRole/data_frame$YearsAtCompany)*100),2))
  
  data_frame$CurMgrPercent <- ifelse(data_frame$YearsAtCompany == 0, 0, round(((data_frame$YearsWithCurrManager/data_frame$YearsAtCompany)*100),2))
  
  data_frame$NoPromoPercent <- ifelse(data_frame$YearsAtCompany == 0, 0, round(((data_frame$YearsSinceLastPromotion/data_frame$YearsAtCompany)*100),2))
  
  data_frame$RetentionPercentNeeded <- round(((data_frame$DiffFromSalary/data_frame$CurrentSalary)*100),2)
  
  #data_frame$Retention_Ratio <- round((data_frame$RetentionPercentNeeded/(data_frame$PercentSalaryHike*100)),2)
  
  # For our correlation check, remove DiffFromSalary and AnnualIncomeNeeded
  # Can always recreate using the current Salary and RetentionPercentNeeded
  data_frame <- subset(data_frame, select = -c(DiffFromSalary, AnnualIncomeNeeded))
  
  # Eliminate AGE because 1) multicollinearity with TotalWorkingYears, etc and
  # Also HUGE bias risk
  data_frame <- subset(data_frame, select = -c(Age))
  
  # Remove the fields relating to the number of years and companies
  # now that we have them expressed in percentages of time with this company
  data_frame <- subset(data_frame, select = -c(NumCompaniesWorked,YearsAtCompany,
                                               TotalWorkingYears,YearsInCurrentRole,
                                               YearsSinceLastPromotion,YearsWithCurrManager))
  
  # Eliminate PercentSalaryHike (last raise) because 1) multicollinearity with several factors, etc and
  # Also direct relation to RetentionPercentNeeded HUGE  
  data_frame <- subset(data_frame, select = -c(PercentSalaryHike))
  
  return(data_frame)
}

do_multicollinearity_check <- function(data_frame) {
  # Run correlation and p-values on all combinations of remaining attributes
  correl_mat <- data_frame %>% rstatix::cor_mat()
  correl_pmat <- correl_mat %>% rstatix::cor_get_pval()
  correl_mat_long <- correl_mat %>% rstatix::cor_gather()
  
  # Define the significance level (alpha)
  alpha <- 0.05
  
  # Calculate the number of comparisons (number of correlations excluding the diagonal)
  n_comparisons <- nrow(correl_pmat) * ncol(correl_pmat) - nrow(correl_pmat)
  
  
  # Apply Bonferroni correction
  correl_mat_long$p <- correl_mat_long$p * n_comparisons
  
  # Identify significant correlations (p-value less than adjusted alpha)
  sig_correl <- correl_mat_long[which(correl_mat_long$p < alpha & 
                                                      correl_mat_long$cor != 1),]
  
  num_rows <- nrow(sig_correl)
  processed_rows <- c()
  mat <- matrix(ncol=4, nrow=0)
  kept_rows <- data.frame(mat)
  for (i in 1:num_rows) {
    if (i %in% processed_rows) next
    processed_rows <- c(processed_rows, i)
    attrib_1 <- as.character(sig_correl[i,1])
    attrib_2 <- as.character(sig_correl[i,2])
    saved_k <- 0
    for (k in (i + 1):num_rows) {
      if (sig_correl[k,2] == attrib_1 && sig_correl[k,1] == attrib_2) 
      {
        saved_k <- k
      }
    }
    if(saved_k != 0) {
      kept_rows <- rbind(kept_rows,sig_correl[i,])
      processed_rows <- c(processed_rows, saved_k)
    }
  }
  return(kept_rows)
}

do_factor_conversion <- function(data_frame) {
  # Let's convert the factors to factors...
  
  factor_cols <- c("Education", "EnvironmentSatisfaction", "JobInvolvement",
                   "JobLevel", "JobSatisfaction", "PerformanceRating", 
                   "RelationshipSatisfaction","StockOption","WorkLifeBalance")
  
  
  data_frame$Education <- factor(data_frame$Education, 
                                 levels=c("1","2","3","4","5"),
                                 labels=c("Below_College",
                                          "College",
                                          "Bachelor",
                                          "Master",
                                          "Doctor"), 
                                 ordered=TRUE)
  data_frame$EnvironmentSatisfaction <- factor(data_frame$EnvironmentSatisfaction, 
                                               levels=c("1","2","3","4"),
                                               labels=c("Low",
                                                        "Medium",
                                                        "High",
                                                        "Very_High"),
                                               ordered=TRUE)
  data_frame$JobInvolvement <- factor(data_frame$JobInvolvement, 
                                      levels=c("1","2","3","4"),
                                      labels=c("Low",
                                               "Medium",
                                               "High",
                                               "Very_High"),
                                      ordered=TRUE)
  data_frame$JobSatisfaction <- factor(data_frame$JobSatisfaction, 
                                       levels=c("1","2","3","4"),
                                       labels=c("Low",
                                                "Medium",
                                                "High",
                                                "Very_High"),
                                       ordered=TRUE)
  data_frame$PerformanceRating <- factor(data_frame$PerformanceRating, 
                                         levels=c("1","2","3","4"),
                                         labels=c("Low",
                                                  "Medium",
                                                  "High",
                                                  "Very_High"),
                                         ordered=TRUE)
  data_frame$RelationshipSatisfaction <- factor(data_frame$RelationshipSatisfaction, 
                                                levels=c("1","2","3","4"),
                                                labels=c("Low",
                                                         "Medium",
                                                         "High",
                                                         "Very_High"),
                                                ordered=TRUE)
  data_frame$WorkLifeBalance <- factor(data_frame$WorkLifeBalance, 
                                       levels=c("1","2","3","4"),
                                       labels=c("Low",
                                                "Medium",
                                                "High",
                                                "Very_High"),
                                       ordered=TRUE)
  ###################
  # For the next/final 2 factors, we don't know what the level means, just what they are
  ###################
  data_frame$JobLevel <- factor(data_frame$JobLevel, 
                                levels=c("1","2","3","4","5"),
                                ordered=TRUE)
  data_frame$StockOption <- factor(data_frame$StockOption, 
                                   levels=c("0","1","2","3"),
                                   ordered=TRUE)
  return(data_frame)
}

################################################################
# Create a Z-Score standardization routine for the numeric
# attributes
#
# <new x> = (<old x> - <mean X>)/ <std dev x>
# 
###############################################################
# For each item
data_standardize <- function(x, na.rm= TRUE) {
  return((x - mean(x))/sd(x))
}


do_standardize_numerics <- function(data_frame) {

# Extract the numeric variables names
numeric_cols <- names(data_frame)[sapply(data_frame, is.numeric)]

# Store and return the mean and standard deviation of the 
# Current Salary and Retention Percent Needed so that the values
# can be "un-scaled" after predicting

mean_cur_sal <- mean(data_frame$CurrentSalary)
sd_cur_sal <- sd(data_frame$CurrentSalary)
mean_ret_pct <- mean(data_frame$RetentionPercentNeeded)
sd_ret_pct <- sd(data_frame$RetentionPercentNeeded)

for(looper in numeric_cols) {
    # Create a new name for the created standardized column    
    newcol_name <- paste(looper,"Std", sep="_")
    # Create a vector containing the standardized values
    new_vector <- data_standardize(data_frame[[looper]])
    # Add the vector to the dataframe
    data_frame <- cbind(data_frame,new_vector)
    # Rename the column
    colnames(data_frame)[which(names(data_frame) == "new_vector")] <- newcol_name
    # Remove the old non-standardized data (NEW DATAFRAME ONLY)
    data_frame <- data_frame[, names(data_frame) != looper]
}

return(list(data_frame = data_frame, mean_cur_sal = mean_cur_sal, 
            sd_cur_sal = sd_cur_sal, mean_ret_pct = mean_ret_pct, 
            sd_ret_pct = sd_ret_pct))
}

do_categorical_independence_check <- function(data_frame) {
  # Create a categorical variable dataframe from the modeling dataframe
  factor_cols <- names(data_frame)[sapply(data_frame, is.factor)]
  
  # fDF at the end indicates factor dataframe
  data_frame_f <- data_frame[factor_cols]
  
  
  # Now combine them into every possible set of pairs
  factor_pairs <- combn(factor_cols,2)
  
  # Create a function that performs the chi-squared test of independence for a pair of variables
  CategoricalIndependenceTest <- function(var1, var2) {
    ContingencyTable <- table(data_frame_f[[var1]], data_frame_f[[var2]])
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
  
  return(DependentVariables)
}

histPercent <- function(x, ...) {
  H <- hist(x, plot = FALSE)
  H$density <- with(H, 100 * density* diff(breaks)[1])
  labs <- paste(round(H$density), "%", sep="")
  plot(H, freq = FALSE, labels = labs,...)
}
