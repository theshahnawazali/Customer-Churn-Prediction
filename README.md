# Customer Churn Prediction using Machine Learning

## Project Overview

Customer Churn is one of the most critical challenges that a subscription-based organization faces; some examples include telecom operators, Internet Service Providers (ISPs), and Software as a Service (SaaS). In other words, customer churn refers to customers terminating their use of the services provided by an organization. The early prediction of customer churn helps the organization act proactively to minimize losses.

The purpose of this project is to develop a machine learning model for customer churn prediction based on the Telco Customer Churn dataset. Various machine learning models have been developed, trained, tested, and compared in order to find the best performing one.

---

## Objectives

* Perform data preprocessing and feature engineering.
* Handle categorical and numerical features using machine learning pipelines.
* Train multiple machine learning classification models.
* Compare model performance using multiple evaluation metrics.
* Perform hyperparameter tuning to improve model performance.
* Select the best-performing model based on business-relevant metrics.

---

## Dataset Information

The dataset contains customer information including:

* Customer demographics
* Account information
* Subscription details
* Internet services
* Contract types
* Payment methods
* Monthly charges
* Total charges
* Customer churn status

Target Variable:

* Churn

  * Yes = Customer left the service
  * No = Customer stayed with the service

---

## Data Preprocessing

Several preprocessing steps were performed before training the models:

### Handling Missing Values

Missing values were identified and handled appropriately to ensure data quality.

### Categorical Feature Encoding

Different encoding techniques were used depending on the feature type:

* Ordinal Encoding for binary categorical features.
* One-Hot Encoding for multi-category features.

### Feature Scaling

Numerical features were standardized using StandardScaler to ensure consistent feature ranges and improve model performance.

### Pipeline Implementation

Scikit-Learn Pipelines were used to automate preprocessing and model training. This approach prevents data leakage and ensures reproducibility.

---

## Machine Learning Models Used

The following classification algorithms were implemented and evaluated:

### 1. Logistic Regression

Linear classifier chosen for benchmarking was Logistic Regression. Though Logistic Regression is very simple, it performed very well in terms of prediction accuracy and generalization capability.


### 2. K-Nearest Neighbors (KNN)

For KNN, manual parameter tuning was performed using various values of:

* n_neighbors
* distance metric
* leaf size

Optimal results were obtained using:

* n_neighbors = 45
* metric = euclidean

### 3. Decision Tree

Decision Tree hyperparameters were manually tuned to improve performance and reduce overfitting.

Best configuration:

* criterion = gini
* max_depth = 7
* min_samples_split = 20
* min_samples_leaf = 20

### 4. Random Forest

Random Forest was implemented as an ensemble learning technique and tuned using:

* n_estimators
* max_depth
* min_samples_split
* min_samples_leaf

Best configuration:

* n_estimators = 300
* max_depth = 10
* min_samples_split = 10
* min_samples_leaf = 2

---

## Model Evaluation Metrics

The following evaluation metrics were used:

* Accuracy Score
* Precision Score
* Recall Score
* F1 Score
* ROC-AUC Score

Since customer churn prediction is an imbalanced classification problem, special attention was given to:

* Recall
* F1 Score
* ROC-AUC Score

rather than relying solely on accuracy.

---

## Model Comparison

| Model                 | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| --------------------- | -------- | --------- | ------ | -------- | ------- |
| Logistic Regression   | 0.8219   | 0.6860    | 0.6032 | 0.6419   | 0.7519  |
| KNN (Tuned)           | 0.8141   | 0.6637    | 0.6032 | 0.6320   | 0.7466  |
| Random Forest (Tuned) | 0.8148   | 0.7000    | 0.5255 | 0.6003   | 0.7222  |
| Decision Tree (Tuned) | 0.8020   | 0.6382    | 0.5818 | 0.6087   | 0.7315  |
| KNN (Base)            | 0.7764   | 0.5863    | 0.5282 | 0.5557   | 0.6970  |
| Random Forest (Base)  | 0.7949   | 0.6544    | 0.4772 | 0.5519   | 0.6932  |
| Decision Tree (Base)  | 0.7182   | 0.4672    | 0.4584 | 0.4628   | 0.6351  |

---

## Results and Findings

Some of the key observations from the experiment include:

* The Logistic Regression classifier was found to perform the best overall among the algorithms.
* The KNN algorithm performed significantly better post-hyperparameter tuning and emerged as the second best-performing algorithm.
* The performance of the Decision Tree classifier was found to be significantly better post-hyperparameter tuning.
* The Random Forest performed very well in terms of accuracy but did not beat Logistic Regression and KNN in terms of Recall and ROC-AUC scores.

---

## Final Model Selection

Logistic Regression was selected as the final model because it achieved the highest overall performance across multiple evaluation metrics.

Final Logistic Regression Performance:

* Accuracy: 82.19%
* Precision: 68.60%
* Recall: 60.32%
* F1 Score: 64.19%
* ROC-AUC Score: 75.19%

The model demonstrated strong predictive capability while maintaining good balance between precision and recall.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Matplotlib
* Seaborn
* Jupyter Notebook
* Streamlit

---

## Future Improvements

Potential future enhancements include:

* XGBoost implementation
* LightGBM implementation
* CatBoost implementation
* Feature selection techniques


---

## Conclusion

This project presents an end-to-end application of machine learning to predict customer churn involving data preprocessing, feature engineering, modeling, hyperparameter tuning, and model evaluation steps. Various classification models were tried, out of which logistic regression performed better than other models. This project underscores the need for experimental approaches and model comparisons to solve business problems with machine learning.