# ==============================================================================
# Q1 COMPREHENSIVE CLINICAL DATA SCIENCE SCRIPT (MERGED & FINAL)
# Methodology: UMAP Manifolds, Causal Bayesian Networks, & Decision Curves
# ==============================================================================

# 1. Load/Install Required Libraries
if (!require("pacman")) install.packages("pacman")
# Added 'parameters' and 'broom.helpers' to ensure gtsummary can tidy 'rms' models
pacman::p_load(tidyverse, umap, dbscan, bnlearn, dcurves, ggdist, rms, 
               gtsummary, ggforce, patchwork, Rgraphviz, corrplot, ggcorrplot,
               parameters, broom.helpers)

# 2. Data Loading & Preprocessing
file_path <- "C:/Users/SUBHRAJIT SAHA/Downloads/Heart_Attack.csv"
heart_data <- read.csv(file_path)
colnames(heart_data) <- make.names(colnames(heart_data))

# Create binary target for clinical utility (Disease Presence vs Absence)
heart_clean <- heart_data %>%
  mutate(target_bin = ifelse(target > 0, 1, 0)) %>%
  mutate(across(c(sex, Chest.pain.type, fasting.blood.sugar, resting.ecg, 
                  exercise.angina, ST.slope), as.factor))

# ==============================================================================
# STEP 1: TOPOLOGICAL MANIFOLD MAPPING (UMAP)
# ==============================================================================
cat("Step 1: Computing UMAP Manifolds...\n")
set.seed(42) # Reproducibility
numeric_cols <- c("age", "trestbps", "cholesterol", "max.heart.rate", "oldpeak")

# UMAP Configuration with Random Init to avoid "Spectral Embedding" failure
umap_config <- umap.defaults
umap_config$init <- "random"

umap_results <- umap(scale(heart_data[, numeric_cols]), config = umap_config)

umap_df <- as.data.frame(umap_results$layout) %>%
  rename(UMAP1 = V1, UMAP2 = V2) %>%
  bind_cols(heart_clean)

p_umap <- ggplot(umap_df, aes(x = UMAP1, y = UMAP2, color = as.factor(target))) +
  geom_point(alpha = 0.7, size = 2.5) +
  geom_mark_hull(aes(fill = as.factor(target)), alpha = 0.1, show.legend = FALSE) +
  scale_color_viridis_d(option = "plasma", name = "Target Severity") +
  theme_minimal(base_size = 14) +
  labs(title = "Topological Patient Manifold (UMAP)",
       subtitle = "Visualizing non-linear clinical phenotypes")

# ==============================================================================
# STEP 2: CAUSAL DISCOVERY & CLINICAL CORRELATIONS
# ==============================================================================
cat("Step 2: Causal Discovery & Correlation Matrix...\n")

# A. Correlation Heatmap (Clinical Standard)
corr_matrix <- cor(heart_data %>% select(where(is.numeric)))
p_corr <- ggcorrplot(corr_matrix, hc.order = TRUE, type = "lower",
                     lab = TRUE, lab_size = 3, method = "circle",
                     colors = c("#6D9EC1", "white", "#E46726"),
                     title = "Clinical Feature Correlation Matrix")

# B. Bayesian Network DAG
bn_df <- heart_data %>% 
  mutate(across(everything(), as.numeric)) %>%
  mutate(target = as.factor(target))
dag <- hc(bn_df)

cat("Generating Causal DAG...\n")
graphviz.plot(dag, highlight = list(nodes = "target", fill = "tomato", col = "darkred"),
              main = "Causal Structure: Directed Acyclic Graph")

# ==============================================================================
# STEP 3: CLINICAL NOMOGRAM & PROBABILITY CALIBRATION
# ==============================================================================
cat("Step 3: Generating Nomogram & Calibration...\n")
dd <- datadist(heart_clean)
options(datadist = "dd")

f <- lrm(target_bin ~ age + sex + Chest.pain.type + trestbps + oldpeak + ST.slope, 
         data = heart_clean, x=TRUE, y=TRUE)

# Nomogram Plot
plot(nomogram(f, fun=plogis, 
              fun.at=c(.001, .01, .05, seq(.1, .9, by=.2), .95, .99, .999),
              funlabel="Probability of Heart Attack"), xfrac=.3)

# Calibration Plot (B=500 for Q1 Stability)
cal <- calibrate(f, method="boot", B=500)
plot(cal, subtitles = FALSE, main = "Model Calibration Plot",
     xlab = "Predicted Probability", ylab = "Actual Observed Fraction")
title(sub = "Black line: Ideal | Blue line: Model Performance", col.sub = "blue", adj = 0)

# ==============================================================================
# STEP 4: DECISION CURVE ANALYSIS & FEATURE IMPORTANCE
# ==============================================================================
cat("Step 4: Performing DCA & Feature Importance...\n")

heart_clean$model_prob <- predict(f, type="fitted")

# A. DCA Plot
dca_res <- dca(target_bin ~ model_prob, 
               data = heart_clean,
               label = list(model_prob = "Novel Multi-modal Model")) %>%
  plot() +
  scale_color_manual(values = c("black", "red", "blue")) +
  theme_minimal() +
  labs(title = "Clinical Decision Curve Analysis",
       subtitle = "Demonstrating Clinical Net Benefit")

# B. Model Importance (Odds Ratios Visualization)
p_importance <- ggplot(as.data.frame(summary(f)), aes(x = reorder(rownames(summary(f)), Effect), y = Effect)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  theme_minimal() +
  labs(title = "Clinical Feature Importance", x = "Predictors", y = "Effect Size (Log-Odds)")

# ==============================================================================
# STEP 5: RAINCLOUD DISTRIBUTIONS & TABLES
# ==============================================================================
cat("Step 5: Visualizing Distributions & Generating Tables...\n")

p_rain <- ggplot(heart_clean, aes(x = as.factor(target), y = max.heart.rate, fill = as.factor(target))) +
  stat_halfeye(adjust = .5, width = .6, justification = -.3, .width = 0, point_colour = NA) +
  geom_boxplot(width = .1, outlier.shape = NA, alpha = 0.5) +
  geom_point(shape = 95, size = 10, alpha = 0.2, position = position_nudge(x = -0.2)) +
  scale_fill_viridis_d(option = "viridis") +
  theme_minimal() +
  coord_flip() +
  labs(title = "Max Heart Rate Distribution", x = "Target Class", y = "Max HR")

# Display Advanced Visuals
print(p_umap)
print(p_corr)
print(dca_res)
print(p_importance)
print(p_rain)

# Publication Table 1 (Patient Characteristics)
table1 <- heart_clean %>%
  select(target, age, sex, Chest.pain.type, trestbps, cholesterol, max.heart.rate, ST.slope) %>%
  tbl_summary(by = target, 
              percent = "column",
              statistic = list(all_continuous() ~ "{mean} ({sd})",
                               all_categorical() ~ "{n} ({p}%)")) %>%
  add_p() %>%
  add_overall() %>%
  bold_labels() %>%
  modify_header(label = "**Variable**") %>%
  modify_caption("**Table 1. Baseline Clinical Characteristics Stratified by Heart Disease Severity**")

# Publication Table 2 (Logistic Regression Results)
table2 <- f %>%
  tbl_regression(exponentiate = TRUE) %>%
  bold_labels() %>%
  italicize_levels() %>%
  modify_caption("**Table 2. Multivariable Logistic Regression for Predicting Heart Attack Risk**")

print(table1)
print(table2)
