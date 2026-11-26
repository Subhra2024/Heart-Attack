# ==============================================================================
# HEART ATTACK ANALYSIS SCRIPT (REVISED FOR COMMENT 3: BASELINE BENCHMARKING)
#
# Methodology Mapping:
# 1. Topological Phenotyping (UMAP) -> Exploratory visualization
# 2. Bayesian Network (HC + BIC) -> Conditional dependency analysis
# 3. Ordinal Regression (CLM) -> Primary severity prediction (0, 1, 2)
# 4. Baseline Classifiers -> Logistic/Multinomial Reg., Random Forest,
#    Gradient Boosting, and SVM baselines under identical conditions
# 5. Binary Utility Model (LRM) -> Nomogram & Decision Curve Analysis (DCA)
# 6. Validation -> Repeated stratified CV / bootstrap (B=500) & ICI
# ==============================================================================

# ----------------------------- 0. SETUP & LIBRARIES ---------------------------
B_BOOT     <- 500        # Bootstrap iterations for validation
N_FOLDS    <- 5          # Identical stratified CV folds for baseline benchmarking
N_REPEATS  <- 3          # Repeated CV for stable multi-class & binary comparison
SEED       <- 42
OUTPUT_DIR <- "HeartAttack_Final_Outputs"

# Install and load required packages
if (!require("pacman")) install.packages("pacman")
pacman::p_load(
  tidyverse, janitor, umap, bnlearn, dcurves, ggdist, rms,
  ordinal, gtsummary, ggforce, ggcorrplot, patchwork, igraph,
  ggraph, gt, zip, DescTools, scales, ggeffects, yardstick,
  nnet, ranger, xgboost, e1071, rsample
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
    target_nom = factor(target_num, levels = c(0, 1, 2), ordered = FALSE),
    
    # Secondary Binary Target for Clinical Utility Y* = 1(Y > 0)
    target_bin_num = ifelse(target_num > 0, 1L, 0L),
    target_bin = factor(target_bin_num, levels = c(0, 1)),
    
    # Categorical predictors
    across(c(sex, chest_pain_type, fasting_blood_sugar, resting_ecg,
             exercise_angina, st_slope), as.factor)
  ) %>%
  drop_na()

numeric_cols <- c("age", "trestbps", "cholesterol", "max_heart_rate", "oldpeak")
factor_cols  <- c("sex", "chest_pain_type", "fasting_blood_sugar",
                  "resting_ecg", "exercise_angina", "st_slope")

# Reusable formula objects (avoids bare column name lookup errors)
rhs_formula <- ~ age + sex + chest_pain_type + trestbps +
  cholesterol + max_heart_rate + oldpeak + st_slope

# Helper: Save dual plots (PNG/PDF)
save_plot <- function(plot_obj, filename, width = 10, height = 8) {
  ggsave(file.path(OUTPUT_DIR, "figures", paste0(filename, ".png")), plot = plot_obj, width = width, height = height, dpi = 300)
  ggsave(file.path(OUTPUT_DIR, "figures", paste0(filename, ".pdf")), plot = plot_obj, width = width, height = height)
}

# Helper: Compute ICI for binary probabilities p_hat vs binary truth y_bin (0/1)
compute_ici <- function(p_hat, y_bin) {
  p_hat <- pmin(pmax(as.numeric(p_hat), 1e-6), 1 - 1e-6)
  lo <- tryCatch(
    loess(as.numeric(y_bin) ~ p_hat, span = 0.75),
    error = function(e) NULL
  )
  if (is.null(lo)) return(NA_real_)
  obs_smooth <- pmin(pmax(as.numeric(predict(lo, newdata = data.frame(p_hat = p_hat))), 0), 1)
  mean(abs(p_hat - obs_smooth), na.rm = TRUE)
}

# Helper: Safely coerce predicted class probability objects to an N x K matrix
as_prob_matrix <- function(pred_obj, expected_cols) {
  mat <- as.matrix(pred_obj)
  k <- length(expected_cols)
  if (ncol(mat) == 1L && nrow(mat) == k) {
    mat <- matrix(as.vector(mat), nrow = 1L, ncol = k, byrow = TRUE)
  } else if (nrow(mat) == 1L && ncol(mat) != k && length(mat) == k) {
    mat <- matrix(as.vector(mat), nrow = 1L, ncol = k, byrow = TRUE)
  }
  if (!is.null(colnames(mat)) && all(expected_cols %in% colnames(mat))) {
    mat <- mat[, expected_cols, drop = FALSE]
  } else if (ncol(mat) == k) {
    colnames(mat) <- expected_cols
  }
  mat
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
ord_formula <- stats::update(rhs_formula, target_ord ~ .)
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

# Apparent Model Performance Metrics (Ordinal Class Prediction Summaries)
level_names <- levels(heart$target_ord)
pred_probs <- as_prob_matrix(predict(ord_fit, type = "prob")$fit, as.character(level_names))
heart$pred_class <- factor(
  level_names[max.col(pred_probs)],
  levels = level_names,
  ordered = TRUE
)

ord_metrics <- metric_set(accuracy, kap)
perf_results <- ord_metrics(heart, truth = target_ord, estimate = pred_class)
write.csv(perf_results, file.path(OUTPUT_DIR, "tables", "Table_03_Ordinal_Performance_Metrics.csv"), row.names = FALSE)

# ------------------- 4b. BASELINE BENCHMARKING (IDENTICAL CONDITIONS) ---------
message("Section C2: Benchmarking Against Conventional Statistical & ML Classifiers...")

# Standardize continuous features strictly within each training fold to prevent leakage
preprocess_fold <- function(train_df, test_df, num_cols) {
  means <- sapply(train_df[, num_cols], mean, na.rm = TRUE)
  sds   <- sapply(train_df[, num_cols], function(x) {
    s <- sd(x, na.rm = TRUE)
    if (is.na(s) || s == 0) 1 else s
  })
  for (col in num_cols) {
    train_df[[col]] <- (train_df[[col]] - means[[col]]) / sds[[col]]
    test_df[[col]]  <- (test_df[[col]]  - means[[col]]) / sds[[col]]
  }
  list(train = train_df, test = test_df)
}

# Build model matrix for xgboost and e1071 using identical predictors
rhs_vars <- c(numeric_cols, factor_cols)
mm_formula <- as.formula(paste("~", paste(rhs_vars, collapse = " + "), "- 1"))

make_x_matrix <- function(df, ref_cols = NULL) {
  x <- model.matrix(mm_formula, data = df)
  if (!is.null(ref_cols)) {
    missing_cols <- setdiff(ref_cols, colnames(x))
    for (mc in missing_cols) {
      x <- cbind(x, setNames(rep(0, nrow(x)), mc))
    }
    x <- x[, ref_cols, drop = FALSE]
  }
  x
}

cv_splits <- rsample::vfold_cv(heart, v = N_FOLDS, repeats = N_REPEATS, strata = target_ord)
level_chr <- as.character(level_names)

clm_cv_formula  <- stats::update(rhs_formula, target_ord ~ .)
mnom_cv_formula <- stats::update(rhs_formula, target_nom ~ .)
rf_cv_formula   <- stats::update(rhs_formula, target_nom ~ .)
svm_cv_formula  <- stats::update(rhs_formula, target_nom ~ .)

# Helper to evaluate a single fold's 3-class probabilities (matrix N x 3) and binary P(Y>0)
eval_fold_predictions <- function(test_df, prob_mat_3class, p_bin_gt0, model_name, split_id) {
  prob_mat_3class <- as_prob_matrix(prob_mat_3class, level_chr)
  pred_cls <- factor(
    level_chr[max.col(prob_mat_3class, ties.method = "first")],
    levels = level_chr,
    ordered = TRUE
  )
  acc_val <- yardstick::accuracy_vec(truth = test_df$target_ord, estimate = pred_cls)
  kap_val <- yardstick::kap_vec(truth = test_df$target_ord, estimate = pred_cls)
  brier_bin <- mean((as.numeric(test_df$target_bin_num) - as.numeric(p_bin_gt0))^2, na.rm = TRUE)
  ici_bin <- compute_ici(p_bin_gt0, test_df$target_bin_num)
  
  # DCA net benefit at threshold p_t = 0.20 (illustrative screening threshold)
  pt <- 0.20
  pred_pos <- as.integer(p_bin_gt0 >= pt)
  tp <- sum(pred_pos == 1L & test_df$target_bin_num == 1L)
  fp <- sum(pred_pos == 1L & test_df$target_bin_num == 0L)
  n_obs <- nrow(test_df)
  nb_20 <- (tp / n_obs) - (fp / n_obs) * (pt / (1 - pt))
  
  tibble(
    split_id = split_id,
    model = model_name,
    accuracy = acc_val,
    kappa = kap_val,
    brier_y_gt0 = brier_bin,
    ici_y_gt0 = ici_bin,
    nb_at_pt020 = nb_20
  )
}

bench_records <- list()

for (i in seq_len(nrow(cv_splits))) {
  sp <- cv_splits$splits[[i]]
  tr_raw <- rsample::analysis(sp)
  te_raw <- rsample::assessment(sp)
  prep <- preprocess_fold(tr_raw, te_raw, numeric_cols)
  tr <- prep$train
  te <- prep$test
  
  # 1. Ordinal CLM (Proposed)
  m_clm <- ordinal::clm(clm_cv_formula, data = tr, link = "logit")
  p_clm <- as_prob_matrix(predict(m_clm, newdata = te, type = "prob")$fit, level_chr)
  p_clm_gt0 <- 1 - p_clm[, 1]
  bench_records[[length(bench_records) + 1]] <- eval_fold_predictions(te, p_clm, p_clm_gt0, "Ordinal CLM (Proposed)", i)
  
  # 2. Multinomial Logistic Regression (Baseline)
  m_mnom <- nnet::multinom(mnom_cv_formula, data = tr, trace = FALSE)
  p_mnom <- as_prob_matrix(predict(m_mnom, newdata = te, type = "probs"), level_chr)
  p_mnom_gt0 <- 1 - p_mnom[, 1]
  bench_records[[length(bench_records) + 1]] <- eval_fold_predictions(te, p_mnom, p_mnom_gt0, "Multinomial Logistic Reg.", i)
  
  # 3. Random Forest (Baseline)
  set.seed(SEED + i)
  m_rf <- ranger::ranger(rf_cv_formula, data = tr, probability = TRUE, num.trees = 500, seed = SEED + i)
  p_rf <- as_prob_matrix(predict(m_rf, data = te)$predictions, level_chr)
  p_rf_gt0 <- 1 - p_rf[, 1]
  bench_records[[length(bench_records) + 1]] <- eval_fold_predictions(te, p_rf, p_rf_gt0, "Random Forest", i)
  
  # 4. Gradient Boosting / XGBoost (Baseline)
  x_tr <- make_x_matrix(tr)
  x_te <- make_x_matrix(te, ref_cols = colnames(x_tr))
  y_tr_idx <- as.integer(tr$target_nom) - 1L
  dtrain <- xgboost::xgb.DMatrix(data = x_tr, label = y_tr_idx)
  dtest  <- xgboost::xgb.DMatrix(data = x_te)
  set.seed(SEED + i)
  m_xgb <- xgboost::xgb.train(
    params = list(
      objective = "multi:softprob",
      num_class = 3L,
      eval_metric = "mlogloss",
      max_depth = 3L,
      eta = 0.05,
      subsample = 0.8,
      colsample_bytree = 0.8
    ),
    data = dtrain,
    nrounds = 100L,
    verbose = 0
  )
  p_xgb <- as_prob_matrix(matrix(predict(m_xgb, dtest), ncol = 3L, byrow = TRUE), level_chr)
  p_xgb_gt0 <- 1 - p_xgb[, 1]
  bench_records[[length(bench_records) + 1]] <- eval_fold_predictions(te, p_xgb, p_xgb_gt0, "Gradient Boosting (XGBoost)", i)
  
  # 5. Support Vector Machine with RBF Kernel (Baseline)
  m_svm <- e1071::svm(svm_cv_formula, data = tr, probability = TRUE, kernel = "radial")
  p_svm <- as_prob_matrix(attr(predict(m_svm, newdata = te, probability = TRUE), "probabilities"), level_chr)
  p_svm_gt0 <- 1 - p_svm[, 1]
  bench_records[[length(bench_records) + 1]] <- eval_fold_predictions(te, p_svm, p_svm_gt0, "SVM (RBF)", i)
}

bench_fold_df <- bind_rows(bench_records)
bench_summary_df <- bench_fold_df %>%
  group_by(model) %>%
  summarise(
    mean_accuracy = mean(accuracy, na.rm = TRUE),
    sd_accuracy   = sd(accuracy, na.rm = TRUE),
    mean_kappa    = mean(kappa, na.rm = TRUE),
    sd_kappa      = sd(kappa, na.rm = TRUE),
    mean_brier_gt0 = mean(brier_y_gt0, na.rm = TRUE),
    mean_ici_gt0   = mean(ici_y_gt0, na.rm = TRUE),
    mean_nb_pt020  = mean(nb_at_pt020, na.rm = TRUE),
    .groups = "drop"
  )

write.csv(bench_fold_df, file.path(OUTPUT_DIR, "tables", "Table_04a_Baseline_CV_Fold_Metrics.csv"), row.names = FALSE)
write.csv(bench_summary_df, file.path(OUTPUT_DIR, "tables", "Table_04b_Baseline_Benchmark_Summary.csv"), row.names = FALSE)
print(bench_summary_df)

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
ici_val <- compute_ici(heart$prob_any, heart$target_bin_num)
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

# Table 4: Formatted Baseline Benchmark Table
table4_gt <- bench_summary_df %>%
  mutate(
    across(c(mean_accuracy, sd_accuracy, mean_kappa, sd_kappa, mean_brier_gt0, mean_ici_gt0, mean_nb_pt020), ~ round(.x, 3))
  ) %>%
  gt::gt() %>%
  gt::tab_header(title = "Table 4: Benchmark Comparison Under Identical Repeated Stratified CV")

gt::gtsave(table4_gt, file.path(OUTPUT_DIR, "tables", "Table_04_Baseline_Benchmark.html"))

# Zip the entire outputs folder for easy extraction
zipfile <- paste0(OUTPUT_DIR, ".zip")
if (file.exists(zipfile)) file.remove(zipfile)
zip::zipr(zipfile, files = OUTPUT_DIR)

message("Analysis Complete! All outputs have been saved and zipped to: ", getwd(), "/", zipfile)
