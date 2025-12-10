
# libraries
install.packages("elasticnet")
install.packages("flexdashboard")
library(tidyverse)
library(caret)
library(glmnet)
library(rpart)
library(rpart.plot)
library(randomForest)
library(elasticnet)
library(ggplot2)
library(dplyr)
library(patchwork)

# CSV Loading
file <- "mlb-player-stats-Batters.csv"
df <- read_csv(file)

head(df)

# Drop rows missing OPS or AB(Two most important stats)
df <- df %>% filter(!is.na(OPS), AB > 0)

# Derived features
df <- df %>%
  mutate(
    AVG = H / AB,
    BB_rate = ifelse(AB > 0, BB / AB, NA_real_),
    K_rate = ifelse(AB > 0, SO / AB, NA_real_),
    Pos_group = as.character(Pos)
  )

# Remove empty row
df <- df %>% drop_na(OPS, Age, AB, HR, AVG, BB_rate, K_rate)

head(df)

# Define response and predictors
response <- "OPS"
predictors_all <- c("Age","AB","HR","AVG","BB_rate","K_rate","Pos_group")
predictors_small <- c("HR","AVG","BB_rate","K_rate","Age","Pos_group")

set.seed(42)

# 1) Scatter plot OPS vs AVG
EXP_plot1 <- ggplot(df, aes(x = AVG, y = OPS)) +
  geom_point(alpha = 0.6) +
  geom_smooth(method="lm", se=FALSE) +
  labs(title = "OPS vs Batting Average (AVG)",
       subtitle = paste0("Correlation: ", round(cor(df$AVG, df$OPS, use="complete.obs"), 3)),
       x = "AVG", y = "OPS")
print(EXP_p1ot1)

# 2) Scatter plot OPS vs HR 
EXP_plot2 <- ggplot(df, aes(x = HR, y = OPS)) +
  geom_point(alpha = 0.6) +
  geom_smooth(method="lm", se=FALSE) +
  labs(title = "OPS vs Home Runs", subtitle = "Do power hitters have higher OPS?",
       x = "Home Runs (HR)", y = "OPS") +
  annotate("text", x = max(df$HR, na.rm=TRUE)*0.6, y = max(df$OPS, na.rm=TRUE)*0.95,
           label = paste0("n = ", nrow(df)))
print(EXP_plot2)


# 3) Scatter plot OPS vs Age 
age_summary <- df %>% group_by(Age) %>% summarise(meanOPS = mean(OPS, na.rm=TRUE))
EXP_plot3 <- ggplot(df, aes(x=Age,y=OPS)) +
  geom_point(alpha=0.4) +
  geom_line(data=age_summary, aes(x=Age,y=meanOPS), color="black", size=1.1) +
  labs(title="OPS vs Age", subtitle = "Average OPS by age overlaid", x="Age", y="OPS")
print(EXP_plot3)

# 4) Boxplot PS by Pos_group
EXP_plot4 <- ggplot(df, aes(x = Pos_group, y = OPS)) +
  geom_boxplot(outlier.shape = NA) +
  labs(title = "Distribution of OPS by Position Group",
       subtitle = "Comparing offensive output across position groups",
       x = "Position Group", y = "OPS") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
print(EXP_plot4)

# 5) Barchart Average OPS by Team 
team_avg <- df %>% group_by(Team) %>% summarise(meanOPS=mean(OPS,na.rm=TRUE), cnt = n()) %>%
  arrange(desc(cnt)) %>% slice(1:15)
EXP_plot5 <- ggplot(team_avg, aes(x=reorder(Team, -meanOPS), y=meanOPS)) +
  geom_col() +
  labs(title="Top 15 Teams by Average OPS (in dataset)", x="Team", y="Average OPS") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))
print(EXP_plot5)

set.seed(123)

df_300 <- df %>% sample_n(300)

# Preserve OPS
ops_only <- df_300$OPS

# Dummy encode all other variables
dummies <- dummyVars(" ~ .", data = df_300 %>% select(-OPS))
df_num <- data.frame(predict(dummies, newdata = df_300 %>% select(-OPS)))

# Add OPS back
df_num$OPS <- ops_only

# Remove near-zero variance predictors
nzv <- nearZeroVar(df_num)
df_num <- df_num[, -nzv]

# Remove any remaining NA
df_num <- na.omit(df_num)

# Train Model

train_ctrl <- trainControl(
  method = "cv",
  number = 5
)

# Linear regression

model_lm <- train(
  OPS ~ ., 
  data = df_num,
  method = "lm",
  trControl = train_ctrl
)
print(model_lm)

# lasso

model_lasso <- train(
  OPS ~ ., 
  data = df_num,
  method = "lasso",
  preProcess = c("center", "scale"),
  trControl = train_ctrl,
  tuneLength = 15
)
 print(model_lasso)
 
# Random forest

model_rf <- train(
  OPS ~ ., 
  data = df_num,
  method = "rf",
  trControl = train_ctrl,
  tuneLength = 5
)
print(model_rf)

# Neural network

model_nnet <- train(
  OPS ~ ., 
  data = df_num,
  method = "nnet",
  preProcess = c("center", "scale"),
  trControl = train_ctrl,
  tuneLength = 10,
  linout = TRUE,
  trace = FALSE
)

print(model_nnet)

# Regression tree

model_tree_small <- train(
  OPS ~ ., 
  data = df_num,
  method = "rpart",
  trControl = train_ctrl
)
print(model_tree_small)

# Compare models (RMSE)

model_results <- data.frame(
  Model = c("Linear Regression", "LASSO", "Random Forest", "Neural Network", "Regression Tree"),
  RMSE = c(
    min(model_lm$results$RMSE),
    min(model_lasso$results$RMSE),
    min(model_rf$results$RMSE),
    min(model_nnet$results$RMSE),
    min(model_tree_small$results$RMSE)
  )
)

print(model_results)

# RMSE bar chart

RMSE_Comp <- ggplot(model_results, aes(x = reorder(Model, RMSE), y = RMSE, fill = Model)) +
  geom_col(show.legend = FALSE) +
  coord_flip() +
  geom_text(aes(label = round(RMSE, 3)), 
            hjust = -0.1, 
            size = 4) +
  labs(
    title = "Model Comparison Using RMSE",
    subtitle = "Lower RMSE Indicates Better Predictive Accuracy",
    x = "Model",
    y = "RMSE"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 16, face = "bold"),
    plot.subtitle = element_text(size = 12),
    axis.text.y = element_text(size = 11)
  )


dashboard <- (EXP_plot1 | EXP_plot2) /
  (EXP_plot3 | EXP_plot4) / (EXP_plot5 | RMSE_Comp)

print(dashboard) 