# Auto MPG — Linear Regression

Predicting car fuel efficiency (mpg) from vehicle specs using multiple linear regression. This is my second machine learning project, built as guided practice (not a copied tutorial) to move from "following steps" to "making my own decisions" about data cleaning, encoding, and evaluation.

## Dataset

[Auto MPG dataset](https://archive.ics.uci.edu/dataset/9/auto+mpg) (also available via `seaborn.load_dataset('mpg')`) — 398 cars with specs (cylinders, displacement, horsepower, weight, acceleration, model year, origin) and their mpg.

## What this project covers

- **Data cleaning**: the `horsepower` column had missing values disguised as the string `'?'` instead of `NaN` — had to detect and fix that before the column could be converted to numeric
- **Handling missing data**: dropped 6 incomplete rows (~1.5% of the data) after converting `horsepower` to numeric
- **One-hot encoding**: `origin` is stored as integers (1/2/3) but represents a category (USA/Europe/Japan), not a quantity — encoded it with `pd.get_dummies()` so the model doesn't assume a false numeric ranking
- **Train/test split**: 80/20 split with `random_state=42` for reproducibility
- **Model**: `sklearn.linear_model.LinearRegression`
- **Evaluation**: MAE, MSE, RMSE, and R² — calculated on both the test set and the training set to check for overfitting
- **Visualization**: EDA scatterplot (weight vs. mpg) before modeling, and an actual-vs-predicted scatterplot with a reference diagonal after modeling

## Results

| Metric | Value |
|---|---|
| MAE | 2.46 |
| MSE | 10.60 |
| RMSE | 3.26 |
| R² (test) | 0.79 |
| R² (train) | 0.83 |

The small gap between train and test R² (0.83 vs. 0.79) indicates the model generalizes reasonably well and isn't overfitting.

## What I'd improve next

- Polynomial regression — the weight-vs-mpg scatterplot shows a slight curve that a straight line can't fully capture
- Feature engineering (e.g. horsepower-to-weight ratio)
- Checking for multicollinearity between weight, displacement, and cylinders

## Tools

Python, pandas, NumPy, seaborn, matplotlib, scikit-learn
