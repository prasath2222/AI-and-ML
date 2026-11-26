
# Regression Statistical Inference

## 1. What Is Statistical Inference?

Imagine you collect information about 1,000 houses.

You train a Linear Regression model to understand how house size relates to house price.

Your model learns:

```python
prediction = 0.02 * size + 5
```

This tells us that the model has learned:

- Weight = 0.02
- Bias = 5

But you may have additional questions:

1. Does house size really have a relationship with house price?
2. Is the learned coefficient reliable?
3. Could the observed relationship have happened because of random variation?
4. How uncertain is our estimated coefficient?
5. How much might the coefficient change if we collected another sample?

Ordinary prediction metrics such as MAE and RMSE do not directly answer all these questions.

Statistical inference provides methods to investigate them.

### Simple definition

Statistical inference uses sample data to estimate population relationships and quantify uncertainty about those estimates.

---

# 2. Understanding Population and Sample

First, understand two important terms.

## 2.1 Population

Population means the complete group you want to understand.

Example:

Suppose you want to study the relationship between house size and house prices across every house in a particular housing market.

All those houses form your population.

## 2.2 Sample

A sample is a smaller group selected from that population.

Example:

Instead of collecting information about every house, you collect data about 1,000 houses.

Those 1,000 houses form your sample.

### Why do we use samples?

Collecting information about every house may be:

- Expensive.
- Time-consuming.
- Difficult.
- Sometimes impossible.

Therefore, we use samples to estimate relationships in the larger population.

### Important

A sample may not perfectly represent the population.

If your sample contains only luxury houses, your conclusions may not apply to ordinary houses.

Statistical inference helps quantify uncertainty, but it cannot automatically fix a biased or unrepresentative sample.

---

# 3. Understanding Ordinary Least Squares (OLS)

OLS stands for Ordinary Least Squares.

It is a method used to estimate the coefficients of a linear regression model.

Remember the regression equation:

\[
\hat y=w_1x+b
\]

OLS chooses the coefficients that minimize the sum of squared residuals.

A residual is:

\[
e_i=y_i-\hat y_i
\]

Where:

- \(y_i\) = actual value.
- \(\hat y_i\) = predicted value.
- \(e_i\) = residual for observation i.

The OLS objective is:

\[
SSE=\sum_{i=1}^{n}(y_i-\hat y_i)^2
\]

SSE means Sum of Squared Errors, also called the residual sum of squares.

### Simple example

Actual house prices:

```python
actual = [20, 30, 40, 50]
```

Predicted house prices:

```python
predicted = [22, 28, 43, 47]
```

Calculate residuals:

```python
residuals = []

for a, p in zip(actual, predicted):
    residuals.append(a - p)

print(residuals)
```

Output:

```text
[-2, 2, -3, 3]
```

Square each residual:

```text
4, 4, 9, 9
```

Add them:

```text
SSE = 4 + 4 + 9 + 9

SSE = 26
```

OLS tries to find coefficients that minimize this sum of squared residuals.

### Important distinction

- OLS is an estimation method.
- Linear Regression is the model.
- Statistical inference investigates uncertainty and relationships associated with the fitted model.

---

# 4. Understanding Regression Coefficients

A regression coefficient tells us how the model's prediction changes when an input feature increases by one unit, while the other features are held constant.

Consider this multiple linear regression model:

\[
\hat y=0.02x_1+3x_2-0.5x_3+5
\]

Where:

- \(x_1\) = house size.
- \(x_2\) = bedrooms.
- \(x_3\) = house age.

The coefficients are:

| Feature | Coefficient |
|---|---:|
| Size | 0.02 |
| Bedrooms | 3 |
| Age | -0.5 |
| Intercept | 5 |

## 4.1 Interpreting the size coefficient

The size coefficient is 0.02.

This means:

For a one-unit increase in house size, the model's predicted price increases by 0.02 units, assuming the other features remain unchanged.

If size is measured in square feet and price in lakhs, the interpretation is:

An additional square foot is associated with an increase of 0.02 lakhs in predicted price, holding bedrooms and age constant.

## 4.2 Interpreting the bedroom coefficient

The bedroom coefficient is 3.

This means:

One additional bedroom is associated with an increase of 3 units in predicted price, holding size and age constant.

## 4.3 Interpreting the age coefficient

The age coefficient is -0.5.

This means:

One additional year of house age is associated with a decrease of 0.5 units in predicted price, holding size and bedrooms constant.

## 4.4 Interpreting the intercept

The intercept is 5.

It represents the predicted target value when all input features equal zero.

However, that situation may not be realistic.

For example, a house with zero size and zero bedrooms may not be meaningful.

Therefore, an intercept does not always have a useful real-world interpretation.

### Important

A regression coefficient describes an estimated relationship.

It does not automatically prove that changing the feature causes the target to change.

For example, house size and house price may be associated, but the coefficient alone does not prove causation.

---

# 5. Understanding Positive and Negative Coefficients

A coefficient can be positive, negative, or zero.

### Positive coefficient

```text
Size coefficient = +0.02
```

As size increases, the model's prediction increases, holding other features constant.

### Negative coefficient

```text
Age coefficient = -0.5
```

As age increases, the model's prediction decreases, holding other features constant.

### Zero coefficient

```text
Feature coefficient = 0
```

The feature contributes nothing through that coefficient in the fitted linear prediction.

For ordinary least squares, coefficients are not generally forced to zero.

For Lasso, some coefficients can become exactly zero.

---

# 6. What Is a Standard Error?

A coefficient is estimated using sample data.

If you collected another sample of houses and trained the same regression model, the estimated coefficient might be different.

Why?

Because different samples contain different observations.

A standard error measures the estimated uncertainty or sampling variability of a statistical estimate.

For a regression coefficient, its standard error describes how precisely that coefficient has been estimated under the statistical model.

### Example

Imagine two estimated size coefficients:

| Model | Size coefficient | Standard error |
|---|---:|---:|
| Model A | 0.02 | 0.001 |
| Model B | 0.02 | 0.01 |

Both coefficients are 0.02.

But their standard errors are different.

Model A has a smaller estimated standard error.

This suggests that its coefficient estimate is more precise under the model's assumptions.

Model B has greater estimated uncertainty.

### Important

A smaller standard error does not automatically mean that a model predicts better.

Standard errors describe uncertainty in estimated statistical parameters, not the same thing as prediction error.

---

# 7. Understanding Hypothesis Testing

Hypothesis testing is a statistical method used to evaluate a claim about a population parameter.

In regression, we commonly test whether a coefficient is different from zero.

Suppose you want to investigate whether house size is associated with house price after accounting for the other features in your model.

We create two hypotheses.

## 7.1 Null Hypothesis (H0)

The null hypothesis says that the population coefficient is zero.

\[
H_0:\beta_1=0
\]

Here:

- \(H_0\) = null hypothesis.
- \(\beta_1\) = population coefficient for house size.

In simple language:

The model's population relationship between house size and price, after accounting for the other included features, is zero.

## 7.2 Alternative Hypothesis (H1)

The alternative hypothesis says that the population coefficient is not zero.

\[
H_1:\beta_1\ne0
\]

In simple language:

There is a nonzero population relationship between house size and price, after accounting for the other included features.

This is a two-sided test.

### Important

A hypothesis test does not directly prove that a feature causes a change in price.

It evaluates a statistical claim under the assumptions of the model and sampling process.

---

# 8. Understanding the t-Statistic

The t-statistic compares an estimated coefficient with its standard error.

The formula is:

\[
t=\frac{\hat\beta-\beta_0}{SE(\hat\beta)}
\]

Understand the symbols:

| Symbol | Meaning |
|---|---|
| t | t-statistic |
| β̂ | Estimated coefficient |
| β0 | Coefficient assumed under the null hypothesis |
| SE | Standard error of the estimated coefficient |

For the common test that the coefficient equals zero:

\[
t=\frac{\hat\beta}{SE(\hat\beta)}
\]

### Example

Suppose:

```text
Estimated coefficient = 0.02

Standard error = 0.005
```

Then:

```text
t = 0.02 / 0.005

t = 4
```

The t-statistic is 4.

It means the estimated coefficient is four standard errors away from zero.

A larger absolute t-statistic generally provides stronger evidence against the null hypothesis, when the relevant t-distribution and assumptions apply.

You do not need to memorize the formula immediately.

First understand what it compares:

**Estimated coefficient versus its uncertainty.**

---

# 9. Understanding P-Values

A p-value is one of the most important concepts in statistical inference.

A p-value measures how unusual the observed test statistic, or something more extreme, would be if the null hypothesis and the test assumptions were true.

In regression, we often use it to evaluate whether a coefficient is statistically distinguishable from zero.

### Example

Suppose your regression output shows:

```text
Feature: Size

Coefficient: 0.02

P-value: 0.001
```

A p-value of 0.001 means:

If the null hypothesis were true and the test assumptions held, the probability of obtaining a test statistic at least this extreme would be 0.1%.

This provides evidence against the null hypothesis.

### Common significance threshold

A frequently used significance level is:

```text
alpha = 0.05
```

The significance level is a threshold chosen before interpreting the test.

| P-value | Decision at 0.05 |
|---:|---|
| 0.001 | Reject the null hypothesis |
| 0.01 | Reject the null hypothesis |
| 0.03 | Reject the null hypothesis |
| 0.05 | At the threshold; conventionally reject if p ≤ 0.05 |
| 0.20 | Do not reject the null hypothesis |

### Very important: What a p-value does NOT mean

A p-value of 0.01 does NOT mean:

- There is a 99% probability that the feature is important.
- There is a 1% probability that the null hypothesis is true.
- The feature causes the target to change.
- The model has 99% prediction accuracy.

A p-value is calculated under the assumption that the null hypothesis is true.

### Statistical significance vs practical importance

Imagine:

```text
Coefficient = 0.0001

P-value = 0.001
```

The coefficient may be statistically distinguishable from zero.

But the actual effect might be too small to matter in a practical business decision.

Statistical significance and practical importance are different.

---

# 10. Understanding Confidence Intervals

A confidence interval provides a range of plausible values for a population parameter under a specified statistical procedure.

A common choice is a 95% confidence interval.

Imagine your estimated size coefficient is:

```text
Coefficient = 0.02

95% confidence interval:

0.015 to 0.025
```

This interval gives a range of plausible population coefficient values under the model's assumptions.

### What does 95% confidence mean?

The correct frequentist interpretation is:

If we repeatedly collected samples and constructed intervals using the same procedure, approximately 95% of those intervals would contain the true population coefficient, assuming the model and sampling assumptions hold.

It does not mean that there is a 95% probability that this particular fixed population coefficient lies inside the interval.

### Understanding zero

Suppose the confidence interval is:

```text
0.015 to 0.025
```

It does not contain zero.

This is consistent with rejecting the two-sided null hypothesis that the coefficient is zero at the corresponding 5% significance level, under the same assumptions and test procedure.

Another example:

```text
-0.01 to 0.03
```

This interval contains zero.

The data do not provide sufficient evidence to rule out a zero coefficient at the corresponding significance level.

### Important

A wide confidence interval indicates greater uncertainty about the coefficient.

A narrow confidence interval indicates greater precision.

---

# 11. Understanding the F-Test

The t-test commonly investigates an individual regression coefficient.

The F-test can investigate multiple coefficients together or the overall explanatory contribution of a regression model.

### Example

Suppose your model uses:

```text
Size
Bedrooms
Age
Location
```

You may want to investigate whether all the feature coefficients are simultaneously zero.

The hypotheses are:

\[
H_0:
\beta_1=\beta_2=\beta_3=\beta_4=0
\]

\[
H_1:
\text{At least one coefficient is nonzero}
\]

The F-test evaluates this joint hypothesis.

### What does the overall F-test tell you?

A statistically significant overall F-test provides evidence that at least one included predictor has a nonzero population coefficient under the model assumptions.

It does not tell you:

- Which individual feature is significant.
- Whether the model predicts accurately.
- Whether the model has a causal interpretation.
- Whether the model is correctly specified.

For individual coefficients, we commonly inspect their t-tests and confidence intervals.

---

# 12. R-Squared and Statistical Inference

You already learned R-squared in Regression Evaluation Metrics.

Let's connect it to statistical inference.

R-squared measures the proportion of the observed target variation explained by the fitted model relative to a mean-only baseline, for the usual regression setting with an intercept.

Example:

```text
R² = 0.80
```

This means the fitted model accounts for 80% of the variation in the observed target values in the dataset being evaluated.

But:

```text
R² = 0.80
```

does not automatically mean:

- 80% prediction accuracy.
- Every feature is statistically significant.
- The model is causal.
- The model will achieve the same performance on unseen data.

R-squared, p-values, confidence intervals and prediction metrics answer different questions.

---

# 13. Prediction Intervals vs Confidence Intervals

These two concepts are easy to confuse.

## 13.1 Confidence Interval for the Mean Response

A confidence interval estimates the uncertainty around the average target value for observations with particular input features.

Example:

You want to estimate the average house price for houses of a particular size.

The confidence interval describes uncertainty around that average predicted price.

## 13.2 Prediction Interval

A prediction interval estimates a range for an individual future observation.

Example:

You want to predict the price of one particular new house.

Its actual price may differ from the average price because of individual variation and unpredictable factors.

Therefore, a prediction interval is generally wider than a confidence interval for the mean response at the same input values.

### Simple comparison

| Confidence interval | Prediction interval |
|---|---|
| Uncertainty about the mean response | Uncertainty about an individual observation |
| Usually narrower | Usually wider |
| Estimates an average relationship | Estimates a range for a new observation |

---

# 14. Performing Statistical Inference Using Python

Let's use `statsmodels`.

It is a Python library commonly used for statistical modelling.

### Step 1: Create a dataset

```python
import pandas as pd

data = {
    "size": [
        1000, 1200, 1500, 1800, 2000,
        2200, 2500, 2700, 3000, 3200
    ],

    "bedrooms": [
        2, 2, 3, 3, 3,
        4, 4, 4, 5, 5
    ],

    "age": [
        10, 9, 8, 7, 6,
        5, 4, 4, 3, 2
    ],

    "price": [
        20, 23, 30, 34, 39,
        44, 48, 52, 59, 63
    ]
}

df = pd.DataFrame(data)
```

### Step 2: Separate features and target

```python
X = df[
    ["size", "bedrooms", "age"]
]

y = df["price"]
```

### Step 3: Add an intercept

```python
import statsmodels.api as sm

X = sm.add_constant(X)
```

The constant column allows the model to estimate an intercept.

### Step 4: Train OLS

```python
model = sm.OLS(y, X).fit()
```

### Step 5: Display statistical results

```python
print(model.summary())
```

The summary contains:

- Coefficients.
- Standard errors.
- t-statistics.
- P-values.
- Confidence intervals.
- R-squared.
- Adjusted R-squared.
- F-statistic.
- Overall model p-value.

### Step 6: Extract coefficients

```python
print(model.params)
```

### Step 7: Extract standard errors

```python
print(model.bse)
```

### Step 8: Extract p-values

```python
print(model.pvalues)
```

### Step 9: Extract confidence intervals

```python
print(model.conf_int())
```

### Step 10: Extract overall F-test results

```python
print("F-statistic:", model.fvalue)

print("Overall F-test p-value:", model.f_pvalue)
```

Important:

This is a small educational dataset. Its results should not be treated as reliable conclusions about the real housing market.

With only ten observations and three predictors plus an intercept, statistical estimates can be highly uncertain.

---

# 15. Important OLS Assumptions for Statistical Inference

Earlier, you studied regression diagnostics.

Now let's connect those assumptions to statistical inference.

For the usual classical OLS standard errors and inference procedures, important assumptions include:

### 1. Linear specification in coefficients

The model must correctly represent the conditional mean using a linear combination of the included coefficients.

### 2. No perfect multicollinearity

The predictors must not have an exact linear dependency that prevents the individual coefficients from being estimated.

### 3. Zero conditional mean of errors

The errors should have an expected value of zero conditional on the included predictors.

This is an important assumption for unbiased coefficient estimation.

### 4. Independence or suitable error dependence assumptions

The observations and errors must satisfy the dependence assumptions required by the chosen inference method.

For time-series data, ordinary independent-error calculations may be inappropriate.

### 5. Constant error variance

The usual classical OLS standard-error formulas assume homoscedasticity.

If error variance changes, heteroscedasticity-robust standard errors may be appropriate.

### 6. Normality of errors

Normality is used for exact finite-sample t and F inference under the classical model assumptions.

It is not required for OLS coefficients to be unbiased under the other relevant assumptions.

Large-sample inference may use other approximations.

### Important distinction

These assumptions are primarily about coefficient estimation and statistical inference.

They are not a universal checklist that every predictive machine learning model must satisfy.

---

# 16. Statistical Inference vs Machine Learning Prediction

This distinction is extremely important.

Imagine you have two goals.

### Goal A: Understand relationships

You want to investigate:

- Is house size associated with house price?
- How large is the estimated relationship?
- How uncertain is the coefficient?
- Is the coefficient statistically distinguishable from zero?

You may use:

- OLS.
- Standard errors.
- t-tests.
- p-values.
- Confidence intervals.

### Goal B: Predict accurately

You want to predict prices for new houses.

You may focus on:

- MAE.
- RMSE.
- R-squared on validation/test data.
- Cross-validation.
- Generalization performance.
- Model selection.

You may use:

- Linear Regression.
- Ridge.
- Random Forest.
- XGBoost.
- LightGBM.
- Other suitable algorithms.

### Comparison

| Statistical inference | Predictive machine learning |
|---|---|
| Understand estimated relationships | Predict unseen values |
| Coefficients and uncertainty | Prediction errors |
| P-values and confidence intervals | MAE, RMSE and validation metrics |
| Often emphasizes interpretation | Often emphasizes generalization |
| Requires assumptions for valid inference | Requires suitable evaluation and leakage control |

These goals can overlap.

You can build a model that is useful for both explanation and prediction, but you must evaluate each goal appropriately.

---

# 17. Common Statistical Inference Mistakes

| Mistake | Correct understanding |
|---|---|
| P-value is the probability that the null hypothesis is true | It is calculated assuming the null hypothesis and test assumptions hold |
| Small p-value means a feature causes the target to change | Statistical association does not prove causation |
| Large R-squared means every feature is important | R-squared does not establish individual feature significance |
| Small standard error means low prediction error | Standard error measures uncertainty in an estimate |
| Confidence interval gives a 95% probability for this fixed coefficient | It describes the long-run coverage of the interval procedure |
| Prediction interval and confidence interval are the same | They represent different types of uncertainty |
| Statistically significant means practically important | Practical importance depends on the size and context of the effect |
| A nonsignificant result proves there is no relationship | It means the data did not provide sufficient evidence against the tested null hypothesis |

---

# 18. What You Must Remember

| Concept | Simple meaning |
|---|---|
| Statistical inference | Drawing conclusions about population relationships from sample data |
| Population | Complete group being studied |
| Sample | Smaller group of observations |
| OLS | Method that minimizes squared residuals |
| Coefficient | Estimated change in prediction for a one-unit feature increase, holding other features constant |
| Standard error | Estimated uncertainty of a coefficient |
| Hypothesis testing | Evaluating a statistical claim |
| Null hypothesis | Claim being tested, commonly that a coefficient is zero |
| Alternative hypothesis | Competing claim |
| t-statistic | Coefficient estimate relative to its standard error |
| P-value | Probability of a result at least as extreme under the null and test assumptions |
| Confidence interval | Interval procedure for estimating a population parameter |
| F-test | Tests joint restrictions on coefficients |
| Prediction interval | Range for an individual future observation |
| Statistical significance | Evidence against a specified null hypothesis at a chosen significance level |

---

# 19. Final Understanding

Regression is useful for two different purposes.

First, it can help us predict values.

Second, it can help us investigate relationships between variables.

Statistical inference helps answer questions about estimated coefficients and uncertainty.

Remember these five concepts:

1. **Coefficient:** What relationship did the model estimate?
2. **Standard error:** How uncertain is that estimate?
3. **t-test:** How strong is the evidence against a specified coefficient value?
4. **P-value:** How unusual is the observed test statistic under the null hypothesis?
5. **Confidence interval:** What range of parameter values is supported by the interval procedure?

Do not confuse statistical significance with prediction accuracy or causation.

**End of Regression Statistical Inference.**
