#####################################################################
# Quickly do a quick histogram for the factor variables
#####################################################################
# Get a list of them to loop through

# factor_cols <- names(sal_dataV3)[sapply(sal_dataV3, is.factor)]
# num_factor_cols <- length(factor_cols)
# 
# for(looper in factor_cols) {
#     # Create names for the new temp dataframe and plots
#     graph_name <- paste(looper,"Plot", sep="_")
#     frame_name <- paste(looper,"Frame", sep="_")
#     # Get the counts for each level
#     level_counts <- table(sal_dataV3[[looper]])
#     temp_graph <- ggplot(data.frame(level_counts), aes(x = Var1, y = Freq)) +
#       geom_bar(stat = "identity", fill='cadetblue') +
#       geom_text(aes(x=Var1, y=Freq,label=Freq)) +
#   # Use stat="identity" to keep original counts
#       labs(title = paste("Counts of ",looper, " Levels", sep=""), x = "Level", y = "Count") +
#       theme_classic()
#     assign(graph_name,temp_graph)
# }
# 
# grid.arrange(Education_Plot,EnvironmentSatisfaction_Plot,JobInvolvement_Plot,
#              JobLevel_Plot,JobSatisfaction_Plot,RelationshipSatisfaction_Plot,
#              PerformanceRating_Plot,StockOption_Plot,WorkLifeBalance_Plot,
#              ncol=3)
# 

# # Now let's look at the variables with near zero variance
# zeroVarVariables <- nearZeroVar(sal_dataV2)
# 
# # Show the names of the columns without variability (not useful for modeling)
# colnames(sal_dataV2[zeroVarVariables])
# 
# # In this case, there are none so retain all the attributes

# Let's start some other graphs


# Scatterplot of RetentionPercentage vs Salary
# BonusVSal <- ggplot(sal_dataV1, aes(y=RetentionPercentNeeded,x=CurrentSalary,
#                                     color=factor(JobLevel))) + geom_point(size=2.5)
# BonusVEdu <- ggplot(sal_dataV1, aes(Education,RetentionPercentNeeded, fill=JobLevel)) + 
#   geom_dotplot(binaxis="y",stackdir="center",dotsize=1)
# 

# # Create a numeric variable dataframe from the modeling dataframe
# new_numeric_cols <- names(train_sal_data)[sapply(train_sal_data, is.numeric)]
# 
# # nDF at the end indicates numeric dataframe
# train_sal_data_nDF <- train_sal_data[new_numeric_cols]
# 
# sal_num_corr <- round(cor(train_sal_data_nDF),2)
# sal_num_pval <- cor_pmat(train_sal_data_nDF)
# 
# # Correlation Plot!
# ggcorrplot(sal_num_corr, hc.order=TRUE, type="lower", lab=TRUE, p.mat=sal_num_pval)
# #ggcorrplot(sal_fac_corr, hc.order=TRUE, type="lower", lab=TRUE, p.mat=sal_fac_pval)
# 
# 
# target_var <- 'Retention_Ratio'
# 
# # Get a list of the numeric variable names without the target variable
# # This will be used for the VIF and Stepwise AIC function
# 
# num_cols_no_target <- colnames(train_sal_data_nDF)
# num_cols_no_target <- num_cols_no_target[!num_cols_no_target %in% c('Retention_Ratio',
#                                                                     'RetentionPercentNeeded',
#                                                                     'PercentSalaryHike_Std')]
# 
# # Build the complete formula
# salLMformula <- as.formula(paste(target_var,paste(num_cols_no_target, collapse = " + "), sep = " ~"))
# 
# # Display the formula
# print(salLMformula)
# 
# #############################
# # TEMP
# # Let's just see what the LR shows
# #############################
# standard_num_only_lr <- lm(salLMformula, data=train_sal_data_nDF)
# 
# # Perform VIF evalation (Variance Inflation Factor)
# VIF_evaluation <- vif(lm(salLMformula, data=train_sal_data_nDF))
# 
# # Determine the attribute with the highest VIF value
# highest_VIF_location <- which.max(VIF_evaluation)
# highest_VIF_index_loc <- highest_VIF_location[1]
# highest_VIF_name <- names(highest_VIF_location)
# 
# # Now perform the stepwise regression to see which set of attributes
# # gives the lowest AIC (Akaike Information Criteria)
# # AIC = mathematical evaluation of how well the data fits the model used
# #       to generate the model
# 
# stepwiseR <- MASS::stepAIC(lm(salLMformula,data=train_sal_data_nDF, direction="both"))
# summary(stepwiseR)
# 
# ###
# # Save the AIC selected features
# ###
# coefficients <- coef(stepwiseR)
# # Ignore intercept (that's why it starts at 2)
# selected_features <- names(coefficients)[coefficients != 0][2:length(coefficients)]


# new_factor_cols <- new_factor_cols[!new_factor_cols %in% c('JobLevel')]
# 
#sal_fac_corr <- round(cor(train_sal_data_fDF, method='spearman'),2)

# Assuming your data is in a dataframe named 'data' with an ordinal column 'ordinal_attribute' and a continuous column 'target_variable'
#spearman_rho <- cor(train_sal_data$RetentionPercentNeeded, train_sal_data$Education, method = "spearman")

# spearman_rho will contain the correlation coefficient

#sal_fac_pval <- cor_pmat(train_sal_data_fDF)

# Convert the data frame into a data table
# train_sal_data_DT <- data.table(train_sal_data)
# 
# # ML tools? one_hot?
# train_sal_data_DT <- one_hot(train_sal_data_DT, cols=new_factor_cols, dropCols = TRUE)
# 
# # Build final training database
# # Assembled from the attributes chosen in the stepwise regression (for numerics)
# # and the independence check (categoricals)
# #train_db_field_list <- c("Retention_Ratio",selected_features,new_factor_cols)
# 
# #train_final_db <- train_sal_data[,c(train_db_field_list)]
# 
# # Convert the data frame into a data table
# train_final_db <- data.table(train_final_db)
# 
# # Run the one-hot encoding on the factor columns
# train_final_db <- one_hot(train_final_db, cols=new_factor_cols, dropCols = TRUE)
# 
# # Let's try a basic linear regression model
# train_linear_reg <- lm(RetentionPercentNeeded ~ ., data = train_final_db)
# train_linear_reg_ratio <- lm(Retention_Ratio ~ ., data = train_final_db)
# Let's try a GLM model
#train_glm <- glm(RetentionPercentNeeded ~ ., data = train_final_db, family = gaussian)

# Build the formula for the neural network
#dt_cols_no_target <- colnames(train_final_db)
#dt_cols_no_target <- dt_cols_no_target[!dt_cols_no_target %in% c('RetentionPercentNeeded')]

# Build the complete formula
#finalNNformula <- as.formula(paste(target_var,paste(dt_cols_no_target, collapse = " + "), sep = " ~"))


#Boruta graph
plot(boruta.train, xlab = "", xaxt = "n")
lz<-lapply(1:ncol(boruta.train$ImpHistory),function(i)
boruta.train$ImpHistory[is.finite(boruta.train$ImpHistory[,i]),i])
names(lz) <- colnames(boruta.train$ImpHistory)
Labels <- sort(sapply(lz,median))
axis(side = 1,las=2,labels = names(Labels),
  at = 1:ncol(boruta.train$ImpHistory), cex.axis = 0.7)

# Let's examine the results of the linear model
# Let's do some diagnostics and visualization of the LR model

# Let's 'tidy' up our linear model
# tidy(trn_linear_reg)
# 
# # Add some additional useful fields
# augment(trn_linear_reg)
# 
# # And look at the overall model stats
# glance(trn_linear_reg)
# 
# # Add the column augments to see how the Q-Q plot looks
# aug_lm_sal <- broom::augment_columns(trn_linear_reg, data = fnl_trn_DT)
# 
# aug_lm_sal %>% dplyr::select(contains(".")) %>% dplyr::glimpse(78)

# Graph it!
# gg_lm_sal_QQPlot <- aug_lm_sal %>%
#   # name the 'sample' the outcome variable (norm_y)
#   ggplot(mapping = aes(sample = RetentionPercentNeeded_Std)) +
#   # add the stat_qq_band
#   qqplotr::stat_qq_band(
#     bandType = "pointwise",
#     mapping = aes(fill = "Normal"), alpha = 0.5,
#     show.legend = FALSE
#   ) +
#   # add the lines
#   qqplotr::stat_qq_line() +
#   # add the points
#   qqplotr::stat_qq_point() +
#   # add labs
#   ggplot2::labs(
#     x = "Theoretical Quantiles",
#     y = "Sample Residuals",
#     title = "Normal Q-Q plot for Salary Data (Linear Regression)"
#   )
# gg_lm_sal_QQPlot
# 
# trn_lm_plot2 <- ggplot(trn_lm_accuracy, aes(x = Descaled_Actual, y = Descaled_Predicted))+ geom_point() + geom_abline(aes(intercept = 0, slope = 1)) +
#   labs(y= "Predicted Retention Percent", x = "True Retention Percent")
# 
# trn_lm_plot2
# 
# abline(0,1)

# hist(trn_lm_accuracy$Difference, xlab="Percentage Difference (Actual vs Predicted)", 
#      main="Training Data Accuracy of Linear Regression Model", xlim=c(0,60), 
#      ylim=c(0,300), breaks=10, col="blue", border="black")

# ggplot(trn_lin_reg_act_vs_pred, aes(x=Descaled_Actual, y = Descaled_Predicted))  + 
#   geom_point(color="red")  +  
#   geom_smooth(method="lm")  + 
#   ggtitle("Training Linear Regression - Actual vs Predicted") +
#   xlab("Actual") + 
#   ylab("Predicted")

hist(trn_nn_accuracy$Difference, xlab="Percentage Difference (Actual vs Predicted)", 
     main="Training Data Accuracy of Neural Network Model", xlim=c(0,60), ylim=c(0,700),
     breaks=10, col="blue", border="black")
