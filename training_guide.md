# Model Training Guide

This guide walks you through training your first tennis match prediction model step-by-step.

## Step-by-Step Training Roadmap

### Step 1: Prepare Your Data

**Goal:** Separate features from the target.

**What to do:**
1. Load train/val/test (same as EDA).
2. Separate features (X) from target (y):
   - X = all columns except `label`
   - y = `label` column
3. Convert boolean columns to integers (if needed):
   - LightGBM/XGBoost can handle booleans, but integers are safer
   - Use `.astype(int)` on boolean columns

**Why:** Models need X (features) and y (target) separated.

---

### Step 2: Choose Your First Model

**Goal:** Start with a baseline.

**Recommendation:** Start with LightGBM.

**Why LightGBM:**
- Handles missing values natively
- Fast training
- Good default hyperparameters
- Works well on tabular data

**Key parameters to understand:**
- `objective='binary'` (binary classification)
- `metric='binary_logloss'` or `'auc'` (evaluation metric)
- `n_estimators` (number of trees)
- `learning_rate` (how fast the model learns)
- `max_depth` (tree depth; controls complexity)
- `random_state` (for reproducibility)

---

### Step 3: Train Your Baseline Model

**Goal:** Get a working model quickly.

**What to do:**
1. Create a LightGBM classifier with simple defaults:
   - Start with `n_estimators=100`, `learning_rate=0.1`, `max_depth=5`
2. Fit the model:
   - Use `model.fit(X_train, y_train)`
   - Optionally use `eval_set=[(X_val, y_val)]` to monitor validation performance during training
3. Make predictions:
   - `predict()` for class predictions (0 or 1)
   - `predict_prob()` for probabilities (0.0 to 1.0)

**Why:** Establish a baseline before tuning.

---

### Step 4: Evaluate Your Model

**Goal:** Understand how well it performs.

**Metrics to calculate:**
1. **Accuracy:** percentage of correct predictions
   - Use `accuracy_score(y_true, y_pred)`
2. **AUC-ROC:** ability to distinguish classes
   - Use `roc_auc_score(y_true, y_pred_proba)`
   - Higher is better (1.0 = perfect, 0.5 = random)
3. **Confusion matrix:** breakdown of predictions
   - Shows true positives, false positives, true negatives, false negatives

**Evaluate on:**
- Training set (to check for overfitting)
- Validation set (main evaluation)
- Test set (final check; use sparingly)

**Why:** Accuracy alone can be misleading with balanced data. AUC-ROC is more informative.

---

### Step 5: Check for Overfitting

**Goal:** Ensure the model generalizes.

**What to look for:**
- Training accuracy >> validation accuracy = overfitting
- Training AUC >> validation AUC = overfitting

**If overfitting:**
- Reduce `max_depth` (simpler trees)
- Increase `learning_rate` and reduce `n_estimators`
- Add regularization (`reg_alpha`, `reg_lambda`)

**Why:** Overfitting means the model memorizes training data and won't generalize.

---

### Step 6: Understand Feature Importance

**Goal:** See which features matter.

**What to do:**
1. Get feature importance: `model.feature_importances_`
2. Create a DataFrame with feature names and importance scores
3. Sort by importance and visualize (bar chart)

**Why:** Helps interpret the model and validate feature engineering.

---

### Step 7: Make Probability Predictions

**Goal:** Get win probabilities.

**What to do:**
- Use `predict_prob()` to get probabilities
- The second column `[:, 1]` gives P(win)
- This is your desired output (0.0 to 1.0)

**Why:** Probabilities are more useful than binary predictions.

---

## Concepts to Understand

1. **Binary classification:** predicting 0 (loss) or 1 (win)
2. **Probability vs prediction:**
   - `predict()` → 0 or 1
   - `predict_prob()` → probability (0.0 to 1.0)
3. **Train/val/test:**
   - Train: model learns from this
   - Val: tune hyperparameters and check performance
   - Test: final evaluation (use once)
4. **Overfitting:** model performs well on training but poorly on validation
5. **AUC-ROC:** measures how well the model separates classes (higher is better)

---

## Things to Watch Out For

1. **Don't use test set for tuning:** only for final evaluation
2. **Missing values:** LightGBM handles them, but verify behavior
3. **Boolean columns:** convert to int if you see issues
4. **Random state:** set it for reproducibility
5. **Memory:** large datasets may need chunking

---

## Suggested Notebook Structure

1. Imports (pandas, numpy, lightgbm, sklearn metrics)
2. Load data (train, val, test)
3. Prepare features (separate X and y, convert booleans)
4. Train baseline model (simple hyperparameters)
5. Evaluate (accuracy, AUC, confusion matrix)
6. Feature importance (visualize)
7. Make predictions (probabilities)
8. Interpret results (what did the model learn?)

---

## Success Criteria

- Validation AUC > 0.65 (better than random)
- Training and validation performance are close (no severe overfitting)
- Feature importance makes sense (rank-related features likely top)
- Can generate probabilities for new matches

---

**Start with a simple baseline, then iterate. What would you like to dive deeper into?**

