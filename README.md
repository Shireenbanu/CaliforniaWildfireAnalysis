# California Wildfire Impact On Home Insurance Premium
Dataset:
This dataset contains information about home insurance and wildfire risk in different ZIP codes across California. It includes average coverage amounts for buildings (Coverage A) and personal items (Coverage C), the average fire risk score for each area, and how many homes fall into different fire risk levels. It also shows how much insurance companies earned in premiums and how much they paid out in claims, both for regular and major (catastrophic) fire and smoke damage. This data can help analyze how wildfire risk affects insurance prices and predict future premiums.
Note: This dataset is obtained from the calfiornia department of insurance (https://www.insurance.ca.gov/01-consumers/200-wrr/DataAnalysisOnWildfiresAndInsurance.cfm).

## Objectives:
1. To perform in-depth exploratory data analysis of the both datasets (tabular and graph)
To engineer new predictive features from the available graphs
To develop a supervised model to classify behaviour into normal and anomalous

## Data Pre-processing 

The dataset is already well-structured and clean, requiring minimal preprocessing. However, I made a small adjustment by __removing newline characters__ from column headers to improve accessibility and usability.

Since the analysis focuses on understanding how catastrophic losses in 2020 influenced premium rates in 2021, I will:

* Merge the 2020 and 2021 datasets into a single DataFrame based on ZIP codes.

* Exclude details related to 2021 catastrophic losses, as they are not relevant to the prediction objective.

### Univariate Analysis:

* Most features exhibit heavy right skewness, indicating the presence of significant outliers far from the mean.

* These outliers are not noise but rather reflect rare, high-impact events in certain ZIP codes.

* Rather than removing them entirely, we can apply __capping (also known as winsorization)__ to retain meaningful outliers while limiting the influence of extreme values that may not add predictive value.

Impact:
Only extreme outliers were handled by capping values at the 99th percentile, ensuring that rare but impactful data points are retained while eliminating excessively extreme cases.

### Bivariate Analysis:

1. Scatter Plots – To visually inspect the relationship between individual features and the target variable.

2. Variance Thresholding – To eliminate features with near-constant values, as they provide little to no predictive power.

3. Correlation Matrix – To identify pairs of features with high correlation.

4. Collinearity Reduction – To remove one of the highly correlated features, improving model stability and avoiding multicollinearity.

5. Predictive Power Score: If two feature has a high collinearity, the feature with low predictive power is removed. 

Impact:


High Collinearity Removal: Removed 9 features that showed high collinearity (correlation > 0.9) with other variables cosnidering the predictive power to reduce multicollinearity. However, a few correlated features were retained intentionally, considering their potential usefulness for future feature engineering.

In Total, __9 features are removed out of 29__.

# Feature Engineering Summary: Insurance Risk Metrics

### Objective
Derived the **exposure-adjusted risk metrics** that standardize claims, losses, and premiums across policies.

### New Features Added

| Feature                | Formula                                  | Purpose                                                                 |
|------------------------|------------------------------------------|-------------------------------------------------------------------------|
| **Avg CAT Loss**       | `Total CAT Loss / Number of Exposures`   | Measures catastrophic loss burden per unit of exposure                  |
| **Avg Non-CAT Loss**   | `Total Non-CAT Loss / Exposures`         | Quantifies non-catastrophic loss density (e.g., theft, accidents)       |
| **Avg CAT Claims**     | `Total CAT Claims / Exposures`           | Normalizes catastrophic claim frequency by exposure count               |
| **Avg Non-CAT Claims** | `Total Non-CAT Claims / Exposures`       | Standardizes non-catastrophic claim frequency                          |
| **Avg Premium 2021**   | `Total Premium 2021 / Exposures`         | Evaluates premium pricing consistency relative to coverage size         |
| **Claim Frequency**    | `Total Claims / Exposures`               | General claim rate (combines CAT and Non-CAT)                          |
| **Avg Claim Severity** | `Total Loss / Total Claims`              | Measures average cost per claim (higher = more severe claims)           |
| **Avg Fire Risk Score** | `Fire Risk Score / Exposures`              | Normalizes fire risk by exposure count to compare risk density across policies.           |

### Feature Selection Process
1. **Correlation Analysis**:
   - Removed newly derived features showing correlation > 0.85 with existing features
   - Example: Dropped `Avg Total Loss` as it correlated strongly (0.92) with `Avg CAT Loss`

2. **Skewness Treatment**:
   - Applied quadratic scaling to features with skewness > 2.0
   - Example: `log(Claim Frequency + 1)` for right-skewed claim data
   - Preserved original scaling for features where transformation reduced predictive power


I started with a Ridge Regression model(after exploring the other linear models) using raw features derived from historical insurance loss and exposure data. The initial model yielded a moderate R² score of 0.5186 and RMSE of 446.57 on the validation set, indicating limited explanatory power.


Model Performance Comparison After Feature Engineering

| Metric   | Before Feature Engineering | After Feature Engineering | Percentage |
| -------- | -------------------------- | ------------------------- | ------------------------- |
| R² Score | __0.5186__                 | __0.7826__                 | +50.91% | 
| RMSE     | 446.57                     | 454.4                      | +1.75%| 
| MAE      | __262.86__                 | __212.48__                 | -19.17%| 

Note: Premium rates from 2020 were excluded in both models to avoid data leakage. Including them inflated the R² to 99%, which aligns with actuarial practices where previous year premiums often serve as a baseline.

### Coefficient Plot:

#### Before feature Engineering:

![images/before_feature_engineering.png](attachment:image-3.png)

#### After feature Engineering:

![images/after_feature_engineering.png](attachment:image.png)

Interpretation:

The Avg Fire Risk Score was a strong predictor in both versions. After feature engineering, the model started leveraging more meaningful features like Avg Non-CAT Claims, Claim Frequency, and Avg CAT Claims, which enhanced generalization and interpretability. These features suggest a correlation between higher claim frequency/fire risk and increased premiums—but do not imply causation.

#### Residual Plot

![images/residual.png](attachment:image-2.png)

Residual = Actual - Predicted

The residual plot shows that the model predicts well for lower premium values (residuals centered around zero), but consistently underestimates higher premiums (predicted low value for higher premiums makes sense due to the large outliers), as seen by large positive residuals. This suggests potential heteroscedasticity and indicates the model struggles with high-value predictions, possibly due to skewed data


Future Work:

The aim of this analysis is to explore whether higher historical losses or elevated fire risk scores contribute to higher insurance premiums. While linear models like Ridge Regression have helped identify strong correlations—most notably with the Average Fire Risk Score—they do not establish causal relationships. Given the consistent prominence of this feature across both pre- and post-feature-engineered models, the next logical step is to apply causal inference techniques to investigate whether such variables directly influence premium increases, beyond mere correlation.



