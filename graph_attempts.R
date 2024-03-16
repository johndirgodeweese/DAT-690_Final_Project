GraphSet <- sal_dataV1 %>% 
  subset(select = c("JobLevel", "StockOption","WorkLifeBalance","EnvironmentSatisfaction",
                    "JobSatisfaction","RelationshipSatisfaction","JobInvolvement"))

# P1 
ggplot(GraphSet) +
  geom_boxplot(aes(x = JobSatisfaction, y = WorkLifeBalance, 
                   color = JobSatisfaction)) +
  labs(x = 'Job Satisfaction', y = "Work/Life Balance") +
  ggtitle("Job Level ~ Job Satisfaction, Work/Life Balance") +
  theme_bw() +
  theme(axis.text.x = element_text(face = 'bold', size = 10),
        axis.text.y = element_text(face = 'bold', size = 10))

ggpairs(GraphSet, aes(color = JobLevel)) + theme_bw()

# Pairs Plots of the Variable Possibly Related to JobLevel
# (StockOption,WorkLifeBalance,EnvironmentSatisfaction,
# JobSatisfaction,RelationshipSatisfaction,JobInvolvement)
# Different Colors by Job Level

JobLevelGraphPointSize <- rep(1,nrow(sal_dataV1))
JobLevelGraphColor <- rep("blue",nrow(sal_dataV1))
JobLevelGraphColor[sal_dataV1$JobLevel=="2"] <-- "green"
JobLevelGraphColumns <- c("StockOption","WorkLifeBalance","EnvironmentSatisfaction",
                          "JobSatisfaction","RelationshipSatisfaction","JobInvolvement")

pairs(sal_dataV1[,6:7],pch=JobLevelGraphPointSize,col=JobLevelGraphColor,cex=0.75)


# Pairs Plots of the Variable Possibly Related to JobLevel
# (StockOption,WorkLifeBalance,EnvironmentSatisfaction,
# JobSatisfaction,RelationshipSatisfaction,JobInvolvement)
# Different Colors by Job Level

JobLevelGraphPointSize <- rep(1,nrow(sal_dataV1))
JobLevelGraphColor <- rep("blue",nrow(sal_dataV1))
JobLevelGraphColor[sal_dataV1$JobLevel==2] <-- "green"
JobLevelGraphColumns <- c("StockOption","WorkLifeBalance","EnvironmentSatisfaction",
                          "JobSatisfaction","RelationshipSatisfaction","JobInvolvement")

pairs(sal_dataV1[,6:7],pch=JobLevelGraphPointSize,col=JobLevelGraphColor,cex=0.75)

# Convert categorical variables to factors
# Create a list
columnsToConvert <- c("Education", "EnvironmentSatisfaction", "JobInvolvement",
                      "JobLevel", "JobSatisfaction", "PerformanceRating",
                      "RelationshipSatisfaction", "StockOption", "WorkLifeBalance")

# Convert and save back to the dataframe
sal_dataV1[,columnsToConvert] <- lapply(sal_dataV1[,columnsToConvert],factor,ordered=T)

str(sal_dataV1)

#Extract the ordinals into their own list for graphing
Ordinal_Data = select_if(sal_dataV1, is.factor)

# Distribution plot of Ordinals
Ordinal_Data %>%
  gather() %>%                             
  ggplot(aes(value)) +                    
  facet_wrap(~ key, scales = "free") +   
  geom_bar(fill = "blue") 
