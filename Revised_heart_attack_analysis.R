# ==============================================================================
# HEART ATTACK ANALYSIS SCRIPT (FINAL METHODOLOGY)
# 
# Methodology Mapping:
# 1. Topological Phenotyping (UMAP) -> Exploratory visualization
# 2. Bayesian Network (HC + BIC) -> Conditional dependency analysis
# 3. Ordinal Regression (CLM) -> Primary severity prediction (0, 1, 2)
# 4. Binary Utility Model (LRM) -> Nomogram & Decision Curve Analysis (DCA)
# 5. Validation -> Bootstrapping (B=500) and Integrated Calibration Index (ICI)
# ==============================================================================

# ----------------------------- 0. SETUP & LIBRARIES ---------------------------
B_BOOT     <- 500        # Bootstrap iterations for validation
SEED       <- 42
OUTPUT_DIR <- "HeartAttack_Final_Outputs"

# Install and load required packages
if (!require("pacman")) install.packages("pacman")
pacman::p_load(
  tidyverse, janitor, umap, bnlearn, dcurves, ggdist, rms,
  ordinal, gtsummary, ggforce, ggcorrplot, patchwork, igraph,
  ggraph, gt, zip, DescTools, scales, ggeffects, yardstick
)

# File Paths (Fallback mechanism to ensure the script runs)
find_data_file <- function() {
  candidates <- c(
    "C:/Users/SUBHRAJIT SAHA/Downloads/Heart_Attack.csv",
    "Heart_Attack.csv",
    file.path(getwd(), "Heart_Attack.csv")
  )
  hit <- candidates[file.exists(candidates)]
  if (length(hit) == 0) {
    stop("Could not locate 'Heart_Attack.csv'. Ensure the file is in C:/Users/SUBHRAJIT SAHA/Downloads/ or the working directory.")
  }
  normalizePath(hit[1], winslash = "/", mustWork = TRUE)
}

file_path <- find_data_file()
message("Using data file: ", file_path)

# Initialize Output Directories
dir.create(OUTPUT_DIR, showWarnings = FALSE)
dir.create(file.path(OUTPUT_DIR, "figures"), showWarnings = FALSE)
dir.create(file.path(OUTPUT_DIR, "tables"),  showWarnings = FALSE)

# ------------------------ 1. DATA LOADING & PREPROCESSING ---------------------
set.seed(SEED)

heart_raw <- read.csv(file_path, check.names = FALSE) %>%
  janitor::clean_names()

heart <- heart_raw %>%
  mutate(
    # Primary Ordinal Target (0, 1, 2)
    target_num = as.integer(target),
    target_ord = factor(target_num, levels = c(0, 1, 2), ordered = TRUE),
    
    # Secondary Binary Target for Clinical Utility Y* = 1(Y > 0)
    target_bin_num = ifelse(target_num > 0, 1, 0),
    target_bin = factor(target_bin_num, levels = c(0, 1)),
    
    # Categorical predictors
    across(c(sex, chest_pain_type, fasting_blood_sugar, resting_ecg, 
             exercise_angina, st_slope), as.factor)
  ) %>%
  drop_na()

numeric_cols <- c("age", "trestbps", "cholesterol", "max_heart_rate", "oldpeak")
factor_cols  <- c("sex", "chest_pain_type", "fasting_blood_sugar",
                  "resting_ecg", "exercise_angina", "st_slope")

# Helper: Save dual plots (PNG/PDF)
save_plot <- function(plot_obj, filename, width = 10, height = 8) {
  ggsave(file.path(OUTPUT_DIR, "figures", paste0(filename, ".png")), plot = plot_obj, width = width, height = height, dpi = 300)
  ggsave(file.path(OUTPUT_DIR, "figures", paste0(filename, ".pdf")), plot = plot_obj, width = width, height = height)
}

# ------------------- 2. TOPOLOGICAL MANIFOLD LEARNING (UMAP) ------------------
message("Section A: Computing UMAP (Exploratory Visualization)...")
umap_cfg <- umap.defaults
umap_cfg$init <- "random" # Prevents spectral convergence errors

umap_fit <- umap(scale(heart[, numeric_cols]), config = umap_cfg)
umap_df <- as.data.frame(umap_fit$layout) %>%
  setNames(c("UMAP1", "UMAP2")) %>%
  bind_cols(heart)

# Plot A1: Standard UMAP with Hulls
p_umap <- ggplot(umap_df, aes(UMAP1, UMAP2, color = target_ord)) +
  geom_point(alpha = 0.7, size = 2) +
  geom_mark_hull(aes(fill = target_ord), alpha = 0.1, show.legend = FALSE) +
  scale_color_viridis_d(option = "plasma", name = "Severity") +
  theme_minimal(base_size = 14) +
  labs(title = "Exploratory UMAP Embedding", 
       subtitle = "Geometric summary of patient phenotypes in latent space")

save_plot(p_umap, "Fig_01a_UMAP_Manifold")
print(p_umap)

# Plot A2: Faceted Density UMAP (Advanced Visualization)
p_umap_density <- ggplot(umap_df, aes(UMAP1, UMAP2)) +
  stat_density_2d(aes(fill = after_stat(level)), geom = "polygon", alpha = 0.6) +
  geom_point(aes(color = target_ord), alpha = 0.4, size = 1) +
  scale_fill_viridis_c(option = "mako", guide = "none") +
  scale_color_viridis_d(option = "plasma", name = "Severity") +
  facet_wrap(~ st_slope, labeller = label_both) +
  theme_minimal(base_size = 14) +
  labs(title = "Topological Density Stratified by ST Slope", 
       subtitle = "Revealing non-linear phenotypic concentrations across ischemic states")

save_plot(p_umap_density, "Fig_01b_UMAP_Density_Facet")
print(p_umap_density)

# ------------------- 2b. EXPLORATORY RAINCLOUD MATRIX -------------------------
message("Section A2: Generating Advanced Raincloud Feature Matrix...")

create_raincloud <- function(data, y_var, title, fill_pal = "viridis") {
  ggplot(data, aes(x = target_ord, y = .data[[y_var]], fill = target_ord)) +
    stat_halfeye(adjust = 0.5, width = 0.6, justification = -0.2, .width = 0, point_colour = NA) +
    geom_boxplot(width = 0.15, outlier.shape = NA, alpha = 0.5) +
    geom_jitter(width = 0.05, alpha = 0.2, size = 1, color = "darkgray") +
    scale_fill_viridis_d(option = fill_pal, guide = "none") +
    theme_minimal() +
    coord_flip() +
    labs(x = "Target Severity", y = y_var, title = title)
}

p_rc1 <- create_raincloud(heart, "oldpeak", "ST Depression (Oldpeak)", "magma")
p_rc2 <- create_raincloud(heart, "max_heart_rate", "Max Heart Rate", "cividis")
p_rc3 <- create_raincloud(heart, "age", "Patient Age", "plasma")
p_rc4 <- create_raincloud(heart, "trestbps", "Resting Blood Pressure", "viridis")

p_raincloud_matrix <- (p_rc1 | p_rc2) / (p_rc3 | p_rc4) + 
  plot_annotation(title = "Physiological Distributions Across Heart Disease Severity",
                  subtitle = "Combined half-violin and boxplot geometries")
save_plot(p_raincloud_matrix, "Fig_02_Raincloud_Matrix")
print(p_raincloud_matrix)

# ------------------- 3. BAYESIAN NETWORK (DEPENDENCY ANALYSIS) ----------------
message("Section B: Bayesian Network for Conditional Dependencies...")

# Fix for bnlearn: Convert integers to numeric before discretization
bn_num_data <- heart[, numeric_cols] %>% mutate(across(everything(), as.numeric))
bn_discretized <- bnlearn::discretize(bn_num_data, method = "quantile", breaks = 3)

bn_data <- bind_cols(
  bn_discretized,
  heart %>% select(all_of(factor_cols), target_ord)
) %>% 
  mutate(across(everything(), as.factor)) # Ensure all are discrete factors

# Blacklist: Target cannot predict upstream clinical variables
blacklist <- data.frame(from = "target_ord", to = names(bn_data))
dag <- hc(bn_data, score = "bic", blacklist = blacklist)

# Plot DAG using ggraph
arc_tbl <- as.data.frame(arcs(dag))
node_tbl <- tibble(name = nodes(dag)) %>%
  mutate(role = ifelse(name == "target_ord", "Target", "Predictor"))
g_dag <- igraph::graph_from_data_frame(arc_tbl, vertices = node_tbl, directed = TRUE)

p_dag <- ggraph(g_dag, layout = "fr") +
  geom_edge_link(alpha = 0.5, end_cap = circle(2.8, "mm"), 
                 arrow = arrow(length = unit(3.2, "mm"), type = "closed")) +
  geom_node_point(aes(color = role), size = 7) +
  geom_node_text(aes(label = name), repel = TRUE, size = 4.5, fontface = "bold") +
  scale_color_manual(values = c("Predictor" = "steelblue", "Target" = "tomato")) +
  theme_void(base_size = 14) +
  labs(title = "Bayesian Dependency Network", 
       subtitle = "Conditional association patterns via Hill-Climbing BIC")

save_plot(p_dag, "Fig_03_Bayesian_DAG")
print(p_dag)

# ------------------- 4. ORDINAL REGRESSION (PRIMARY MODEL) --------------------
message("Section C: Ordinal Cumulative Link Modeling (Primary Inference)...")
ord_formula <- target_ord ~ age + sex + chest_pain_type + trestbps + 
  cholesterol + max_heart_rate + oldpeak + st_slope

ord_fit <- ordinal::clm(ord_formula, data = heart, link = "logit")

# Extract Coefficients for Custom Export & Forest Plot
coef_df <- as.data.frame(coef(summary(ord_fit))) %>%
  rownames_to_column("term") %>%
  filter(!grepl("\\|", term)) %>% # Filter out cut-points
  mutate(
    OR = exp(Estimate),
    CI_Low = exp(Estimate - 1.96 * `Std. Error`),
    CI_High = exp(Estimate + 1.96 * `Std. Error`),
    p_value = `Pr(>|z|)`
  ) %>%
  arrange(desc(OR))

write.csv(coef_df, file.path(OUTPUT_DIR, "tables", "Table_02_Ordinal_Coefficients.csv"), row.names = FALSE)

# Plot C1: Forest Plot of Ordinal Odds Ratios
p_forest <- ggplot(coef_df, aes(x = OR, y = reorder(term, OR))) +
  geom_vline(xintercept = 1, linetype = "dashed", color = "darkgray", size = 1) +
  geom_errorbarh(aes(xmin = CI_Low, xmax = CI_High), height = 0.2, color = "#2c3e50", size = 1) +
  geom_point(aes(color = p_value < 0.05), size = 4) +
  scale_color_manual(values = c("TRUE" = "#e74c3c", "FALSE" = "#7f8c8d"), name = "Significant (p<0.05)") +
  scale_x_log10(breaks = c(0.1, 0.5, 1, 2, 5, 10)) +
  theme_minimal(base_size = 14) +
  labs(title = "Forest Plot of Cumulative Odds Ratios",
       subtitle = "Predictors of advanced severity in the Ordinal CLM Framework",
       x = "Odds Ratio (Log Scale)", y = "Clinical Feature")

save_plot(p_forest, "Fig_04a_Ordinal_Forest_Plot")
print(p_forest)

# Plot C2: Marginal Effects Plot (Predicted Probabilities across Oldpeak)
# FIX: Explicitly factorize response.level to avoid discrete scale error
eff_oldpeak <- ggeffects::ggpredict(ord_fit, terms = "oldpeak [all]") %>%
  mutate(response.level = factor(response.level))

p_eff <- ggplot(eff_oldpeak, aes(x = x, y = predicted, color = response.level)) +
  geom_line(size = 1.2) +
  geom_ribbon(aes(ymin = conf.low, ymax = conf.high, fill = response.level), alpha = 0.1, color = NA) +
  scale_color_viridis_d(option = "plasma", name = "Target Class") +
  scale_fill_viridis_d(option = "plasma", name = "Target Class") +
  theme_minimal(base_size = 14) +
  labs(title = "Ordinal Marginal Effects of ST Depression",
       subtitle = "Smooth predicted probabilities of severity classes across 'oldpeak'",
       x = "ST Depression (Oldpeak)", y = "Predicted Probability")

save_plot(p_eff, "Fig_04b_Ordinal_Marginal_Effects")
print(p_eff)

# Model Performance Metrics (Ordinal Class Prediction Summaries)
# FIX: Ensure factor levels match exactly between truth and estimate
pred_probs <- predict(ord_fit, type = "prob")$fit
# Map numeric index of max probability to actual level names
level_names <- levels(heart$target_ord)
heart$pred_class <- factor(
  level_names[max.col(pred_probs)], 
  levels = level_names, 
  ordered = TRUE
)

ord_metrics <- metric_set(accuracy, kap)
perf_results <- ord_metrics(heart, truth = target_ord, estimate = pred_class)
write.csv(perf_results, file.path(OUTPUT_DIR, "tables", "Table_03_Ordinal_Performance_Metrics.csv"), row.names = FALSE)

# ------------------- 5. BINARY UTILITY MODEL (NOMOGRAM & DCA) -----------------
message("Section D: Binary Utility Model (Nomogram, DCA, Calibration)...")

# Data Distribution for rms
dd <- datadist(heart)
options(datadist = "dd")

# Fit Logistic Regression for Y* = 1(Y > 0)
bin_formula <- target_bin ~ age + sex + chest_pain_type + trestbps + oldpeak + st_slope
bin_fit <- lrm(bin_formula, data = heart, x = TRUE, y = TRUE)
heart$prob_any <- predict(bin_fit, type = "fitted")

# Nomogram
png(file.path(OUTPUT_DIR, "figures", "Fig_05_Clinical_Nomogram.png"), width = 1200, height = 800, res = 150)
plot(nomogram(bin_fit, fun = plogis, funlabel = "Pr(Heart Disease Y > 0)"), xfrac = 0.4)
dev.off()

# Calibration Curve (Bootstrap B=500)
png(file.path(OUTPUT_DIR, "figures", "Fig_06_Bootstrap_Calibration.png"), width = 1000, height = 800, res = 150)
cal_obj <- calibrate(bin_fit, method = "boot", B = B_BOOT)
plot(cal_obj, subtitles = FALSE, main = sprintf("Bootstrap Calibration (B=%d)", B_BOOT),
     xlab = "Predicted Probability", ylab = "Observed Proportion")
dev.off()

# Compute Integrated Calibration Index (ICI)
lo_cal <- loess(as.numeric(as.character(target_bin)) ~ prob_any, data = heart, span = 0.75)
obs_smooth <- pmin(pmax(predict(lo_cal, newdata = data.frame(prob_any = heart$prob_any)), 0), 1)
ici_val <- mean(abs(heart$prob_any - obs_smooth), na.rm = TRUE)
cat("\nIntegrated Calibration Index (ICI):", round(ici_val, 4), "\n")

# Decision Curve Analysis (DCA)
dca_fit <- dca(target_bin_num ~ prob_any, data = heart)
p_dca <- plot(dca_fit) + 
  theme_minimal(base_size = 14) + 
  scale_color_manual(values = c("black", "gray", "red")) +
  labs(title = "Decision Curve Analysis", 
       subtitle = "Net Benefit for Secondary Binary Endpoint Y* = 1(Y > 0)")
save_plot(p_dca, "Fig_07_DCA")
print(p_dca)

# ------------------- 6. DESCRIPTIVE TABLES AND EXPORT -------------------------
message("Section E: Generating Baseline Tables & Archiving...")

# Table 1: Baseline Characteristics by Ordinal Target
table1 <- heart %>%
  select(target_ord, age, sex, chest_pain_type, trestbps, cholesterol, max_heart_rate, oldpeak) %>%
  tbl_summary(
    by = target_ord, 
    statistic = list(all_continuous() ~ "{mean} ({sd})", all_categorical() ~ "{n} ({p}%)"),
    missing = "no"
  ) %>%
  add_overall() %>%
  add_p(test = list(all_continuous() ~ "kruskal.test", all_categorical() ~ "chisq.test")) %>%
  bold_labels() %>%
  modify_caption("**Table 1: Baseline Characteristics Stratified by Disease Severity**")

as_gt(table1) %>% gt::gtsave(file.path(OUTPUT_DIR, "tables", "Table_01_Baseline.html"))

# Table 2: Formatted Ordinal Regression Table
table2 <- ord_fit %>%
  tbl_regression(exponentiate = TRUE) %>%
  bold_labels() %>%
  bold_p(t = 0.05) %>%
  modify_caption("**Table 2: Ordinal Cumulative Link Model Predictors**")

as_gt(table2) %>% gt::gtsave(file.path(OUTPUT_DIR, "tables", "Table_02_Ordinal_Regression_Summary.html"))

# Zip the entire outputs folder for easy extraction
zipfile <- paste0(OUTPUT_DIR, ".zip")
if(file.exists(zipfile)) file.remove(zipfile)
zip::zipr(zipfile, files = OUTPUT_DIR)

message("Analysis Complete! All outputs have been saved and zipped to: ", getwd(), "/", zipfile)
