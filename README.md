# Tennis Match Prediction with Machine Learning

A machine learning project that predicts the probability of a player winning a tennis match against an opponent using historical ATP match data from 2013-2024. This project employs gradient boosting models to learn patterns from player attributes, match characteristics, and tournament features.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset Description](#dataset-description)
3. [Exploratory Data Analysis](#exploratory-data-analysis)
4. [Feature Engineering](#feature-engineering)
5. [Model Selection](#model-selection)
6. [Model Training and Results](#model-training-and-results)
7. [Key Findings](#key-findings)
8. [Setup and Usage](#setup-and-usage)

## Project Overview

The goal of this project is to build a predictive model that estimates the probability of a player winning a tennis match based on various features including player rankings, age differences, physical attributes, tournament characteristics, and match conditions. The model outputs a probability score between 0.0 and 1.0, where 1.0 indicates a 100% predicted win probability.

### Problem Formulation

This is a binary classification problem where:
- **Target Variable**: `label` (1 = player wins, 0 = player loses)
- **Input Features**: 24 engineered features derived from player attributes, match characteristics, and tournament information
- **Evaluation Metrics**: Accuracy and AUC-ROC (Area Under the ROC Curve)

The dataset is designed with a 50/50 class balance by representing each match from both players' perspectives, ensuring the model learns balanced win/loss patterns.

## Dataset Description

### Data Source

The project uses ATP (Association of Tennis Professionals) match data from 2013 to 2024, containing detailed information about professional tennis matches including player statistics, tournament details, and match outcomes.

### Dataset Splits

The data is split temporally to ensure realistic evaluation and prevent data leakage:

- **Training Set**: 2013-2019 (35,364 matches, 59.9% of data)
- **Validation Set**: 2020-2021 (7,660 matches, 13.0% of data)
- **Test Set**: 2022-2024 (15,990 matches, 27.1% of data)

This time-based split ensures the model generalizes to future matches and prevents overfitting to historical patterns that may not persist.

### Data Characteristics

- **Total Features**: 24 features (excluding target variable)
- **Target Distribution**: Balanced 50/50 split by design (each match appears twice: once from winner's perspective, once from loser's)
- **Missing Values**: 176 total missing values, all height-related (0.25% of height_diff, 0.12% of player_ht and opponent_ht)
- **Data Types**: Mix of numerical (float64, int64) and boolean features

## Exploratory Data Analysis

### Target Variable Analysis

The target variable (`label`) shows a perfect 50/50 distribution across all splits:
- Training: 50.00% win rate
- Validation: 50.00% win rate
- Test: 50.00% win rate

This balanced distribution is intentional and achieved by representing each match from both players' perspectives, ensuring the model sees balanced examples from both viewpoints.

### Feature Distributions

Key insights from feature distribution analysis:

1. **Rank Difference (`rank_diff`)**: Normally distributed with range from -2125 to 2125, indicating balanced matchups across the dataset
2. **Age Difference (`age_diff`)**: Normally distributed with range from -21.5 to 21.5 years
3. **Player Rank (`player_rank`)**: Right-skewed distribution, as expected since fewer players achieve higher rankings
4. **Player Age (`player_age`)**: Right-skewed distribution, with mean age around 26-27 years

The distributions show that difference features (rank_diff, age_diff) are normally distributed, while absolute features (player_rank, player_age) are right-skewed, which is expected for ranking and age data.

**Visualizations**: The EDA notebook (`notebooks/01_eda.ipynb`) contains histogram visualizations comparing feature distributions across train/val/test splits for `rank_diff`, `age_diff`, `player_rank`, and `player_age`. These visualizations confirm consistent distributions across splits and validate the temporal splitting approach.

### Temporal Trends

Analysis of age distribution across time splits reveals an interesting trend:

- **Training Set (2013-2019)**: Mean age ~26.5 years, with 13.4% of players in the 24-26 age range
- **Test Set (2022-2024)**: Mean age shifts younger, with 20.7% of players in the 24-26 age range

This temporal shift indicates that newer generations of players are entering professional tennis at younger ages. While this could potentially affect model performance if age proves to be a critical factor, the model's reliance on relative differences (age_diff) rather than absolute age helps mitigate this concern.

### Missing Data Analysis

Missing values are minimal and concentrated in height-related features:
- `height_diff`: 88 missing values (0.25%)
- `player_ht`: 44 missing values (0.12%)
- `opponent_ht`: 44 missing values (0.12%)

These missing values are handled natively by LightGBM, which can split on missing values during tree construction, making explicit imputation unnecessary.

### Data Leakage Prevention

A comprehensive check confirmed that all identity columns have been removed to prevent data leakage:
- Player identifiers: `player_id`, `player_name`, `player_ioc`
- Opponent identifiers: `opponent_id`, `opponent_name`, `opponent_ioc`
- Tournament identifiers: `tourney_id`, `tourney_name`

This ensures the model learns generalizable patterns based on player attributes rather than memorizing specific players or tournaments.

### Feature Distribution Comparison

Visualizations comparing feature distributions across train/val/test splits confirm:
- Consistent distributions across splits, validating proper temporal splitting
- No significant distribution shifts that would indicate data leakage or improper splitting
- Outliers are present but tree-based models are robust to them

**Visualizations**: See `notebooks/01_eda.ipynb` for detailed histogram comparisons showing the distribution of key features (`rank_diff`, `age_diff`, `player_rank`, `player_age`) across all three data splits. The visualizations use 1st-99th percentile limits to focus on the main distribution while annotating outlier counts.

## Feature Engineering

### Why Tabular Dataset?

This project uses a tabular dataset approach rather than other data formats (e.g., time series, graph structures, or text-based features) for several reasons:

1. **Natural Tabular Structure**: Tennis match data inherently consists of structured features (player attributes, match characteristics) that map naturally to a tabular format
2. **Tree-Based Model Compatibility**: Gradient boosting models like LightGBM excel at tabular data, automatically learning feature interactions and handling mixed data types
3. **Interpretability**: Tabular features allow for clear feature importance analysis and model interpretability
4. **Efficiency**: Tabular data is computationally efficient for training and inference
5. **Feature Engineering Flexibility**: Easy to create derived features (differences, ratios) that capture relative advantages

### Feature Engineering Process

#### Player View Transformation

Each match is represented from both players' perspectives:
- **Row A**: Player = winner, Opponent = loser, Label = 1 (win)
- **Row B**: Player = loser, Opponent = winner, Label = 0 (loss)

This transformation ensures:
- Balanced 50/50 class distribution
- Model learns from both perspectives of each match
- Prevents bias toward always predicting from the "winner's" perspective

#### Difference Features

Key engineered features capture relative advantages:
- `rank_diff`: Player rank - Opponent rank (negative = player ranked higher)
- `rank_points_diff`: Player ranking points - Opponent ranking points
- `age_diff`: Player age - Opponent age
- `height_diff`: Player height - Opponent height

These difference features proved to be the most important predictors, as they capture relative matchups rather than absolute values.

#### Categorical Encoding

- **Surface**: One-hot encoded as `surface_Grass`, `surface_Hard` (Clay is the baseline/dropped category)
- **Tournament Level**: One-hot encoded as `tourney_level_F`, `tourney_level_G`, `tourney_level_M`, `tourney_level_O` (one category dropped to avoid multicollinearity)
- **Handedness**: Binary features `player_is_left` and `opponent_is_left` to capture left-handed advantage

#### Temporal Features

- `year`: Match year (used for splitting, not as a model feature)
- `month`: Match month to capture seasonality effects
- `rounds_remaining`: Tournament rounds remaining (captures match importance and pressure)

#### Removed Features

The following features were intentionally removed to prevent bias and overfitting:

- **Identity Features**: Player names, IDs, countries (IOC codes), tournament names and IDs
  - Reason: Prevents model from memorizing specific players/tournaments rather than learning generalizable patterns
- **Davis Cup**: Team competition data removed as it's not representative of individual ATP match outcomes
- **Olympic Games**: Removed for similar reasons (team/representative competition format differs from standard ATP matches)

## Model Selection

### Why LightGBM over XGBoost?

LightGBM was chosen as the primary model for this project over XGBoost for several technical and practical reasons:

#### Performance and Speed

1. **Training Speed**: LightGBM uses a leaf-wise tree growth strategy and histogram-based algorithm, making it significantly faster than XGBoost's level-wise approach, especially on large datasets
2. **Memory Efficiency**: LightGBM's memory usage is lower, allowing for faster training iterations and easier hyperparameter tuning
3. **Native Missing Value Handling**: LightGBM can handle missing values natively without requiring imputation, which is beneficial given the small amount of missing height data

#### Model Characteristics

1. **Gradient-Based One-Side Sampling (GOSS)**: LightGBM's built-in GOSS algorithm focuses on samples with larger gradients, improving training efficiency
2. **Exclusive Feature Bundling (EFB)**: Automatically bundles sparse features, reducing the number of features and improving training speed
3. **Good Default Hyperparameters**: LightGBM's default parameters work well out-of-the-box, reducing the need for extensive hyperparameter tuning

#### Practical Considerations

1. **CPU Performance**: LightGBM performs excellently on CPU, which is sufficient for this dataset size. XGBoost would be preferred if GPU acceleration were available, but for CPU-based training, LightGBM is the optimal choice
2. **Tabular Data Optimization**: LightGBM is specifically optimized for tabular data, which aligns perfectly with this project's data format
3. **Lower Accuracy Trade-off**: While XGBoost may achieve slightly higher accuracy in some cases, the speed advantage of LightGBM allows for faster experimentation and iteration, which is valuable during model development

#### When XGBoost Would Be Preferred

XGBoost would be a better choice if:
- GPU acceleration is available (XGBoost has better GPU support)
- Maximum accuracy is the primary concern and training time is not a constraint
- The dataset requires more sophisticated regularization techniques

For this project's requirements (CPU training, fast iteration, good accuracy), LightGBM is the optimal choice.

### Model Architecture

The baseline model uses the following configuration:

```python
LGBMClassifier(
    objective='binary',           # Binary classification
    metric='binary_logloss',      # Optimization metric
    n_estimators=200,              # Number of trees
    learning_rate=0.05,            # Learning rate
    max_depth=5,                   # Maximum tree depth
    random_state=42,               # Reproducibility
    verbose=-1                     # Suppress output
)
```

Early stopping is implemented with a patience of 10 rounds, monitoring validation loss to prevent overfitting.

## Model Training and Results

### Training Process

The model was trained with:
- **Early Stopping**: Stops training if validation loss doesn't improve for 10 consecutive rounds
- **Best Iteration**: Training stopped at iteration 64 based on validation performance
- **Validation Monitoring**: Model performance monitored on validation set during training

### Validation Set Results

- **Accuracy**: 64.15% (better than 50% random baseline)
- **AUC-ROC**: 0.7029 (indicates good class separation ability)
- **Confusion Matrix**:
  - True Negatives (correct losses): 2,465
  - False Positives (predicted win, actual loss): 1,365
  - False Negatives (predicted loss, actual win): 1,381
  - True Positives (correct wins): 2,449

### Overfitting Analysis

Training set performance compared to validation:
- **Training Accuracy**: 67.21%
- **Training AUC-ROC**: 0.7431
- **Train-Val Accuracy Gap**: 3.06%
- **Train-Val AUC Gap**: 4.02%

The small gap between training and validation performance indicates minimal overfitting. The model generalizes well without requiring additional regularization (L1/L2 regularization was not needed).

### Test Set Results (Final Evaluation)

- **Test Accuracy**: 64.40%
- **Test AUC-ROC**: 0.7013
- **Confusion Matrix**:
  - True Negatives: 5,154
  - False Positives: 2,841
  - False Negatives: 2,852
  - True Positives: 5,143

### Generalization Analysis

Comparison between validation and test performance:
- **Accuracy Difference**: +0.25% (test slightly better than validation)
- **AUC-ROC Difference**: -0.16% (minimal difference)

The model generalizes well to the test set, with performance consistent between validation and test sets. This indicates the model learned generalizable patterns rather than overfitting to the validation set.

### Feature Importance Analysis

The top 10 most important features are:

1. `rank_points_diff` (349) - Ranking points difference
2. `rank_diff` (168) - Rank difference
3. `age_diff` (159) - Age difference
4. `player_age` (126) - Player's absolute age
5. `player_rank_points` (121) - Player's ranking points
6. `height_diff` (116) - Height difference
7. `opponent_age` (113) - Opponent's age
8. `opponent_rank_points` (110) - Opponent's ranking points
9. `best_of` (89) - Number of sets (3 vs 5)
10. `year` (86) - Match year

**Key Insights from Feature Importance**:
- Difference features dominate the top 6 positions, confirming that relative advantages matter more than absolute values
- Ranking-related features (rank_points_diff, rank_diff) are the strongest predictors
- Age and height differences are significant factors
- Tournament characteristics (best_of, year) also contribute to predictions

This feature importance ranking validates the feature engineering approach, showing that creating difference features was the right strategy.

**Visualization**: The model training notebook (`notebooks/02_first_model.ipynb`) contains a horizontal bar chart visualization showing the top 15 most important features, clearly illustrating the dominance of difference-based features in the model's decision-making process.

## Key Findings

### Model Performance

1. **Baseline Performance**: The model achieves 64.40% accuracy and 0.7013 AUC-ROC on the test set, significantly better than random guessing (50% accuracy)
2. **AUC-ROC Focus**: For tennis match prediction, AUC-ROC is more important than accuracy because it measures the model's ability to rank predictions by confidence, which is valuable for probability-based betting or decision-making
3. **Generalization**: The model generalizes well across temporal splits, with consistent performance on validation and test sets

### Feature Engineering Insights

1. **Difference Features Are Key**: The top 6 most important features are all difference-based or relative measures, confirming that matchups are more important than absolute player attributes
2. **Ranking Dominance**: Ranking-related features (rank_points_diff, rank_diff) are the strongest predictors, aligning with tennis domain knowledge
3. **Feature Engineering Success**: The player view transformation (representing matches from both perspectives) successfully created a balanced dataset and improved model learning

### Data Quality

1. **Temporal Consistency**: The time-based split ensures realistic evaluation, and the consistent 50/50 label distribution across splits validates proper data transformation
2. **Minimal Missing Data**: Only 0.25% of data has missing values (height-related), and LightGBM handles these natively
3. **No Data Leakage**: Comprehensive checks confirmed all identity columns are removed, ensuring the model learns generalizable patterns

### Model Characteristics

1. **Minimal Overfitting**: Small gap between training and validation performance (3-4%) indicates good generalization
2. **Early Stopping Effective**: Training stopped at iteration 64, preventing overfitting while maintaining good performance
3. **LightGBM Suitability**: The model's performance and training efficiency confirm LightGBM was the right choice for this tabular dataset

## Setup and Usage

### Prerequisites

- Python 3.8 or higher
- Virtual environment (recommended)

### Installation

1. Create a virtual environment:
```bash
python3 -m venv venv
```

2. Activate the virtual environment:
```bash
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Project Structure

```
tennis-ml-predictions/
├── data/
│   ├── raw/              # Raw ATP match data (2013-2024)
│   └── processed/        # Processed datasets (train.csv, val.csv, test.csv)
├── notebooks/
│   ├── 01_eda.ipynb      # Exploratory Data Analysis
│   └── 02_first_model.ipynb  # Model training and evaluation
├── src/
│   ├── config.py         # Configuration settings
│   ├── load_data.py      # Data loading utilities
│   ├── feature_engineering.py  # Feature engineering pipeline
│   └── train_model.py    # Model training utilities
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

### Usage

1. **Data Processing**: Run the feature engineering pipeline to create train/val/test splits:
   - Processed data should be in `data/processed/` directory

2. **Exploratory Data Analysis**: Open `notebooks/01_eda.ipynb` to explore the dataset

3. **Model Training**: Open `notebooks/02_first_model.ipynb` to train and evaluate the model

4. **Making Predictions**: Use the trained model to predict match outcomes:
   ```python
   import lightgbm as lgb
   import pandas as pd
   
   # Load trained model
   model = lgb.Booster(model_file='model.txt')
   
   # Prepare features for new match
   features = prepare_features(match_data)
   
   # Get win probability
   win_probability = model.predict(features)[0]
   ```

### Dependencies

Key dependencies include:
- `pandas`: Data manipulation
- `numpy`: Numerical computations
- `lightgbm`: Gradient boosting model
- `scikit-learn`: Model evaluation metrics
- `matplotlib` & `seaborn`: Data visualization

See `requirements.txt` for the complete list.

---

## Future Improvements

Potential areas for enhancement:
- Hyperparameter tuning using more sophisticated search strategies
- Feature engineering experiments (interaction features, polynomial features)
- Ensemble methods combining multiple models
- Time-series features capturing player form trends
- Advanced regularization techniques if overfitting becomes an issue
