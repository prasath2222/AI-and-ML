
# Robust Regression

## 1. What Is Robust Regression?

Imagine you want to predict house prices.

Your dataset contains:

| House | Size (sq ft) | Price (lakhs) |
|---|---:|---:|
| A | 1000 | 20 |
| B | 1500 | 30 |
| C | 2000 | 40 |
| D | 2500 | 50 |
| E | 3000 | 200 |

Look at House E.

Most houses follow a pattern:

- 1000 sq ft → 20 lakhs
- 1500 sq ft → 30 lakhs
- 2000 sq ft → 40 lakhs
- 2500 sq ft → 50 lakhs

But House E has a price of 200 lakhs.

This observation is very different from the others.

It could be:
- A data-entry error.
- A luxury property.
- A house in an unusually valuable location.
- A genuine observation that requires additional features.

A normal Linear Regression model using Ordinary Least Squares (OLS) can be strongly affected by unusual observations.

Robust Regression provides alternative methods that can reduce the influence of certain unusual observations.

### Simple definition

Robust Regression is a family of regression methods designed to reduce sensitivity to outliers or violations of certain modelling assumptions.

Important:

Robust regression does not mean that the model automatically ignores every unusual observation.

Different robust methods handle different types of unusual data.

---

# 2. Why Is Ordinary Linear Regression Sensitive to Outliers?

First, remember how ordinary least squares works.

OLS minimizes the sum of squared residuals.

\[
SSE=\sum_{i=1}^{n}(y_i-\hat y_i)^2
\]

Where:

- \(y_i\) = actual value.
- \(\hat y_i\) = predicted value.
- Residual = actual minus predicted.

The important part is that residuals are squared.

### Example 1: Small error

```python
error = 2

squared_error = error ** 2

print(squared_error)

# 4
```

### Example 2: Large error

```python
error = 20

squared_error = error ** 2

print(squared_error)

# 400
```

Compare:

| Error | Squared error |
|---:|---:|
| 2 | 4 |
| 5 | 25 |
| 10 | 100 |
| 20 | 400 |
| 50 | 2500 |

Large errors receive disproportionately large penalties.

This is why an observation with a very large residual can have a substantial influence on an OLS model.

### Important distinction

A large residual is not the same as high leverage.

- A large residual means the prediction is far from the actual target.
- High leverage means the input features are unusual compared with the other observations.

An observation with high leverage can strongly influence the fitted regression line, even if its residual is not particularly large.

Robust regression methods do not all solve both problems in the same way.

---

# 3. Understanding Robust Loss Functions

Before learning individual algorithms, understand one important concept.

A loss function measures how much a model's predictions differ from actual values.

Ordinary least squares uses squared errors.

\[
L(e)=e^2
\]

Here, \(e\) is the residual.

Robust regression methods can use different loss functions that reduce the influence of large residuals.

For example:

- Absolute error.
- Huber loss.
- Other robust loss functions.

The goal is to prevent a small number of extreme residuals from dominating the model's objective.

---

# 4. Huber Regression

Huber Regression is a robust regression method that combines two types of loss:

1. Squared error for smaller residuals.
2. Absolute-error-like loss for larger residuals.

This is called Huber loss.

## 4.1 Understanding Huber Loss

The Huber loss is:

\[
L_\delta(e)=
\begin{cases}
\frac{1}{2}e^2 & |e|\leq\delta\\
\delta(|e|-\frac{1}{2}\delta) & |e|>\delta
\end{cases}
\]

Understand the symbols:

| Symbol | Meaning |
|---|---|
| e | Residual |
| δ | Threshold controlling the transition |
| \(|e|\) | Absolute value of the residual |
| L | Loss value |

### When the error is small

Huber loss behaves like squared error.

For example:

```text
Error = 2
Threshold = 5
```

Since 2 is smaller than 5, the squared-error part is used.

### When the error is large

Huber loss behaves like absolute error, with a linear penalty for larger residuals.

For example:

```text
Error = 20
Threshold = 5
```

The loss increases linearly rather than quadratically for this large residual.

This reduces the influence of extreme residuals compared with ordinary squared-error loss.

## 4.2 Python implementation

Scikit-learn provides `HuberRegressor`.

```python
from sklearn.linear_model import HuberRegressor

model = HuberRegressor(
    epsilon=1.35,
    max_iter=1000
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print(predictions)
```

### Understanding the hyperparameters

**epsilon**

Controls the point at which residuals are treated as outliers by the Huber loss.

A smaller value makes the method more resistant to large residuals, but can also reduce efficiency when the data follows the assumed model well.

Scikit-learn requires epsilon to be at least 1.

**max_iter**

Maximum number of iterations used by the optimization procedure.

### When is Huber Regression useful?

- When your dataset contains some unusually large residuals.
- When you want a linear model with reduced sensitivity to those residuals.
- When ordinary least squares is being strongly affected by unusual target values.

Important:

Huber Regression mainly addresses unusual residuals. It does not automatically solve every high-leverage problem.

---

# 5. RANSAC Regression

RANSAC stands for:

Random Sample Consensus.

It is a robust estimation technique that tries to identify a group of observations that agree with a model.

It separates observations into two conceptual groups:

1. Inliers.
2. Outliers.

### Understanding inliers

Inliers are observations that fit the candidate model reasonably well.

### Understanding outliers

Outliers are observations that do not fit the candidate model within a chosen error threshold.

## 5.1 How RANSAC Works

Imagine you have 100 house records.

Most follow a relationship between house size and price.

But 10 records are unusual.

RANSAC works approximately like this:

**Step 1:** Randomly select a small subset of observations.

**Step 2:** Train a model using that subset.

**Step 3:** Check how many other observations agree with the model.

**Step 4:** Count the observations that fall within the chosen residual threshold.

**Step 5:** Repeat with other random subsets.

**Step 6:** Select a suitable candidate based on consensus among observations.

**Step 7:** Refit the model using the selected inliers.

The exact implementation includes additional rules for choosing and validating candidate models.

### Simple illustration

Imagine:

```text
100 observations

90 observations follow a similar relationship.

10 observations are unusual.
```

RANSAC attempts to identify the group of 90 observations that agrees with a common model.

The unusual observations may be excluded from the final inlier-based fit.

This can be useful when a dataset contains a substantial number of outliers.

## 5.2 Python implementation

```python
from sklearn.linear_model import RANSACRegressor

from sklearn.linear_model import LinearRegression

model = RANSACRegressor(
    estimator=LinearRegression(),
    min_samples=0.5,
    residual_threshold=10,
    max_trials=100,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print(predictions)
```

### Understanding the hyperparameters

| Hyperparameter | Meaning |
|---|---|
| estimator | Base regression model |
| min_samples | Minimum sample size used to fit a candidate model |
| residual_threshold | Residual threshold used to identify inliers |
| max_trials | Maximum number of random candidate trials |
| random_state | Controls randomness for reproducibility |

### Understanding residual_threshold

Suppose:

```python
residual_threshold = 10
```

An observation with a residual within the threshold can be classified as an inlier under the configured loss criterion.

An observation outside the threshold is treated as an outlier for that candidate.

The threshold must be chosen carefully.

It depends on:
- The scale of the target.
- The noise in the data.
- The problem.
- The model.

### Inspecting inliers and outliers

```python
inliers = model.inlier_mask_

print("Inliers:")
print(inliers)

print("Number of inliers:")
print(inliers.sum())

print("Number of outliers:")
print((~inliers).sum())
```

The mask identifies which training observations RANSAC classified as inliers.

### When is RANSAC useful?

- When a substantial portion of observations may be outliers.
- When a meaningful group of observations follows a consistent relationship.
- When you want a model that is less influenced by observations that disagree with that relationship.

Important:

RANSAC can fail if too many observations are outliers, if the inliers do not provide enough information, or if the threshold and sampling settings are unsuitable.

It can also exclude genuine but unusual observations.

---

# 6. Theil-Sen Regression

Theil-Sen is another robust regression method.

It estimates regression coefficients using medians of slopes calculated from subsets of observations.

Let's understand the basic idea using a simple example.

Suppose you have:

| Size | Price |
|---:|---:|
| 1000 | 20 |
| 1500 | 30 |
| 2000 | 40 |
| 2500 | 50 |

For two observations, the slope is:

\[
\text{Slope}=
\frac{y_2-y_1}{x_2-x_1}
\]

For the first and second houses:

\[
\text{Slope}=
\frac{30-20}{1500-1000}
\]

\[
\text{Slope}=0.02
\]

Theil-Sen calculates slopes from many pairs or subsets and uses their median to obtain a robust coefficient estimate.

### Why use the median?

Imagine the calculated slopes are:

```text
0.02
0.021
0.019
0.020
0.80
```

The mean is affected strongly by 0.80.

The median is:

```text
0.020
```

The median is less sensitive to that extreme value.

This is the basic intuition behind Theil-Sen estimation.

For multiple features, the method uses a more general subset-based coefficient estimation procedure.

## 6.1 Python implementation

```python
from sklearn.linear_model import TheilSenRegressor

model = TheilSenRegressor(
    max_subpopulation=10000,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print(predictions)
```

### When is Theil-Sen useful?

- When you want robust linear regression.
- When the dataset contains unusual observations.
- When the dataset is not extremely large.
- When you want an alternative to OLS that uses median-based estimation.

### Limitations

- Can become computationally expensive for large datasets.
- May not be suitable for every high-dimensional problem.
- Its performance depends on the dataset and estimator settings.

---

# 7. Huber vs RANSAC vs Theil-Sen

These are three different robust regression approaches.

| Method | Main idea | Useful situation |
|---|---|---|
| Huber Regression | Uses a loss that changes from quadratic to linear | Some large residuals |
| RANSAC | Searches for a model supported by a consensus of inliers | Many observations may be outliers |
| Theil-Sen | Uses median-based coefficient estimation | Robust linear relationship estimation |

### Important differences

**Huber:**

Does not normally discard unusual observations. It reduces the penalty growth for large residuals.

**RANSAC:**

Explicitly identifies observations as inliers or outliers according to its consensus procedure.

**Theil-Sen:**

Uses robust coefficient estimation based on medians.

No method is automatically best.

You should compare them on your actual dataset.

---

# 8. Quantile Regression

Quantile Regression is another important regression technique.

Ordinary least squares estimates the conditional mean of the target.

Quantile Regression estimates a chosen conditional quantile.

### First understand the median

Suppose house prices are:

```text
20, 25, 30, 40, 100
```

The median is:

```text
30
```

The median represents the middle value.

Quantile Regression can estimate the median or other quantiles of the target, conditional on the input features.

## 8.1 Understanding quantiles

| Quantile | Meaning |
|---|---|
| 0.10 | 10th percentile |
| 0.25 | 25th percentile |
| 0.50 | Median |
| 0.75 | 75th percentile |
| 0.90 | 90th percentile |

For example, you could build models to estimate:

- Lower expected house prices.
- Median house prices.
- Upper expected house prices.

This can provide a more complete understanding of the target distribution than estimating only its mean.

## 8.2 Python implementation

Scikit-learn provides `QuantileRegressor`.

```python
from sklearn.linear_model import QuantileRegressor

model = QuantileRegressor(
    quantile=0.50,
    alpha=0.0,
    solver="highs"
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
```

The quantile value 0.50 represents the conditional median.

You can also choose:

```python
quantile=0.90
```

to estimate the conditional 90th percentile.

### When is Quantile Regression useful?

- When the median is more useful than the mean.
- When you want lower or upper conditional quantiles.
- When prediction uncertainty differs across the input range.
- When you want to understand different parts of the target distribution.

Important:

Quantile Regression is not simply another outlier-removal technique. It estimates a different statistical quantity from ordinary least squares.

---

# 9. Robust Regression vs Ordinary Linear Regression

Let's compare the methods.

| Method | Objective or approach | Outlier handling |
|---|---|---|
| Ordinary Linear Regression | Squared residuals | Sensitive to large residuals |
| Ridge Regression | Squared residuals plus L2 penalty | Does not specifically solve outlier sensitivity |
| Lasso Regression | Squared residuals plus L1 penalty | Does not specifically solve outlier sensitivity |
| Huber Regression | Huber loss | Reduces influence of large residuals |
| RANSAC | Consensus-based fitting | Can identify and exclude outliers |
| Theil-Sen | Median-based coefficient estimation | Robust coefficient estimation |
| Quantile Regression | Quantile loss | Estimates conditional quantiles rather than the mean |

Very important:

Regularization and robust regression solve different problems.

- Regularization controls coefficient complexity.
- Robust regression reduces sensitivity to certain unusual observations or changes the quantity being estimated.

You can sometimes combine robust estimation with other modelling techniques, but you must understand how their objectives interact.

---

# 10. Robust Scaling vs Robust Regression

Do not confuse these two concepts.

They sound similar, but they are different.

## Robust Scaling

Robust scaling changes the numerical scale of input features using statistics such as the median and interquartile range.

Example:

```python
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)
```

It helps reduce the influence of extreme feature values on the scaling parameters.

It does not remove outliers.

## Robust Regression

Robust Regression changes how the model estimates its relationship with the target.

Examples:

- Huber Regression.
- RANSAC.
- Theil-Sen.

### Comparison

| Robust Scaling | Robust Regression |
|---|---|
| Data preprocessing technique | Regression method |
| Changes feature scales | Changes estimation or loss procedure |
| Uses median and IQR | Uses robust estimation approaches |
| Does not automatically reduce regression outlier influence | Designed to reduce sensitivity to certain unusual observations |

You may use robust scaling before a suitable regression model, but it does not replace robust regression.

---

# 11. Comparing Robust Regression Models in Python

Let's compare several methods.

```python
from sklearn.linear_model import (
    LinearRegression,
    HuberRegressor,
    RANSACRegressor,
    TheilSenRegressor
)

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_error

import numpy as np

models = {
    "Linear Regression": LinearRegression(),

    "Huber Regression": HuberRegressor(),

    "RANSAC Regression": RANSACRegressor(
        estimator=LinearRegression(),
        random_state=42
    ),

    "Theil-Sen Regression": TheilSenRegressor(
        random_state=42
    )
}

for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    print(name)

    print("MAE:", mae)

    print("RMSE:", rmse)

    print()
```

This example assumes that the input features are numerical, properly prepared and suitable for the estimators.

For Huber and Theil-Sen, feature scaling is often useful when features have very different numerical scales.

For RANSAC, choose an appropriate base estimator and residual threshold.

### How should you compare them?

Do not look only at training performance.

Investigate:

- Validation MAE.
- Validation RMSE.
- Performance on genuine unusual observations.
- Performance on ordinary observations.
- Stability across different data splits.
- Training time.
- Whether the model is excluding genuine observations.

If your real application must predict unusual but valid houses, you should not evaluate performance only on the majority of ordinary houses.

---

# 12. How to Decide Which Robust Method to Use

Use the following as a starting guide.

| Situation | Method to investigate |
|---|---|
| A few unusually large residuals | Huber Regression |
| Many observations disagree with a common pattern | RANSAC |
| Want median-based robust coefficient estimates | Theil-Sen |
| Want to estimate median house prices | Quantile Regression |
| Feature values have extreme scales | Robust Scaling |
| Coefficients are too large or model is overfitting | Ridge, Lasso or Elastic Net |

These are starting points, not guaranteed solutions.

You must understand why the observations are unusual before selecting a method.

---

# 13. Important Limitations of Robust Regression

Robust methods are useful, but they have limitations.

### A. They do not fix incorrect data automatically

If a house price is entered as 5000 instead of 50, you should investigate and correct the data if it is confirmed to be an error.

A robust model is not a substitute for data quality.

### B. They can ignore genuine observations

RANSAC may classify a genuine luxury house as an outlier if it does not agree with the majority relationship.

That may be undesirable if you need to predict luxury houses.

### C. They do not solve every high-leverage problem

Some robust regression methods are more resistant to large residuals than to unusual predictor values.

Understand the particular method's robustness properties.

### D. They may require additional computation

Theil-Sen and RANSAC can be more computationally expensive than ordinary linear regression.

### E. They may not perform better on every dataset

If your data contains no meaningful outlier problem, ordinary linear regression may perform just as well or better.

---

# 14. Practical Workflow for Robust Regression

A sensible workflow:

1. Train a baseline Linear Regression model.
2. Calculate MAE and RMSE.
3. Inspect residual plots.
4. Identify observations with large residuals.
5. Investigate high-leverage observations separately.
6. Check whether unusual values are data errors or genuine cases.
7. Train one or more robust regression models.
8. Compare validation performance.
9. Examine performance on unusual but valid observations.
10. Select the method that best matches your actual prediction objective.
11. Evaluate the final model on untouched test data.

Do not remove observations or choose a robust estimator simply because it produces a better score.

The method must make sense for the real problem.

---

# 15. What You Must Remember

| Concept | Simple meaning |
|---|---|
| Robust Regression | Regression methods designed to reduce sensitivity to certain unusual observations |
| OLS | Minimizes squared residuals |
| Huber Regression | Uses quadratic loss for smaller residuals and linear loss for larger ones |
| RANSAC | Finds a model supported by a group of inliers |
| Inlier | Observation that agrees with the candidate model |
| Outlier | Observation that does not agree within the selected threshold |
| Theil-Sen | Uses median-based coefficient estimation |
| Quantile Regression | Estimates conditional quantiles |
| Robust Scaling | Scales features using robust statistics |
| Residual | Actual value minus predicted value |
| Leverage | Measures how unusual input-feature values are |

---

# 16. Final Understanding

Ordinary Linear Regression can be sensitive to unusual observations because it minimizes squared errors.

Robust Regression provides alternative approaches.

Remember these four important methods:

1. **Huber Regression:** Reduces the influence of large residuals using Huber loss.
2. **RANSAC:** Searches for a model supported by a consensus of inliers.
3. **Theil-Sen:** Uses median-based coefficient estimation.
4. **Quantile Regression:** Estimates a chosen conditional quantile instead of the conditional mean.

The most important lesson:

Do not automatically remove unusual observations. Understand whether they are errors, genuine cases, or evidence that your model is missing an important relationship.

**End of Robust Regression.**
