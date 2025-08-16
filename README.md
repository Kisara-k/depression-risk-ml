# Predicting Depression Risk Using XGBoost

**Authors:** Kodithuwakku Kisara, Dr. Uthayasanker Thayasivam

## Abstract

In this study, we use machine learning to predict depression risk based on lifestyle and demographic factors. We analyze a dataset with a mixture of categorical and numerical features, such as age, work/study hours, and family history of mental illness. The aim is to build a model that predicts if someone is at risk of depression. Our approach includes data cleaning, feature engineering, and the use of the XGBoost algorithm. The model achieved strong performance, with an accuracy score of 0.9381 and an F1 score of 0.8252, showing its potential to help identify people at risk of depression.

## 1. Introduction

Depression is a common mental health problem that can have a significant impact on people's lives. Early detection of those at risk is important so they can get the help they need. Machine learning offers a way to predict depression risk by analyzing data about a person's lifestyle and background. In this study, we use a dataset that includes information such as age, work/study hours, occupation, work satisfaction, stress, and sleep patterns to predict whether someone is at risk of depression. Our goal is to build a model that can accurately make this prediction. We follow a step-by-step process, including exploring the data, cleaning it, creating new features, and training a machine learning model. The results show that our model works well, making it a useful tool for identifying people who might need support.

## 2. Methodology

### 2.1 Overview of the Dataset

The dataset contains a combination of categorical and numerical features intended to help identify how everyday factors could be linked to mental health risks, making it an invaluable resource for developing machine learning models aimed at predicting mental health outcomes [2].

**The dataset consists of the following features:**

**Numerical Features:**

- Age
- Work/Study Hours: Number of hours spent working or studying
- CGPA: Cumulative Grade Point Average

**Ordinal Features:**

- Academic Pressure
- Study Satisfaction
- Work Pressure
- Job Satisfaction
- Financial Stress

**Categorical Features:**

- Gender
- Working Professional or Student
- Dietary Habits
- Have you ever had suicidal thoughts?
- Family History of Mental Illness
- Sleep Duration (later converted to an ordinal feature)

**Categorical Features (High Cardinality):**

- Name
- City: City of residence
- Profession: Professional occupation
- Degree: Educational degree held by the individual

**Target Variable (Binary):**

- Depression: Indicates whether the individual experiences depression

### 2.2 Data Preprocessing and Feature Engineering

To begin, the presence of missing values in the dataset was assessed using the function `df.isna().sum()`. This step is important for understanding how much missing data there is, as it can affect the accuracy and effectiveness of later analysis and modeling work.

The dataset had missing values in some fields related to students and working professionals. These missing values were explored separately for each category to ensure they were handled properly based on their context.

Several feature engineering methods were considered to improve model performance, such as creating composite features like "Pressure" and "Satisfaction" (common to both students and working professionals), and using imputation methods like mean/mode and KNN imputation. However, none of these methods resulted in significant improvements in model accuracy.

For ordinal features and the "CGPA" column, constant imputation with a value of -1 was chosen. This simple method ensured that missing values were dealt with without disturbing the analysis or modeling process.

High cardinality categorical features, such as "Profession", "Degree", "Dietary Habits", and "City", created additional challenges. For these, missing values were filled with "NA". Also, categories with fewer than 50 occurrences were grouped into an "Other" category to reduce noise and help model performance.

The "Sleep Duration" column was mapped to an engineered feature "Sleep Quality" scale by a predefined mapping. This conversion turned categorical sleep duration values into a numerical scale, making it easier for the model to interpret.

In the end, columns considered not relevant to the analysis, like "Name" and "Sleep Duration", were removed from the dataset. This simplification helped focus the analysis on the most important features, making the modeling process more efficient.

### 2.3 Exploratory Data Analysis

**Correlation Analysis:**

A correlation matrix was computed for numerical features to identify relationships between them. This helps in understanding how features interact with each other and can guide feature selection or engineering.

![](assets\04_Image_0001.jpg)

**Violin Plots for Numerical and Ordinal Features:**

Violin plots were used to visualize the distribution of numerical and ordinal features against the target variable (Depression). These plots provide insights into how the distributions differ between the two target classes, which can be useful for identifying patterns or anomalies.

#### Numerical Features
![](assets\05_Image_0001.jpg)
![](assets\05_Image_0002.jpg)
![](assets\05_Image_0003.jpg)

#### Ordinal Features
![](assets\05_Image_0004.jpg)
![](assets\05_Image_0005.jpg)
![](assets\06_Image_0001.jpg)
![](assets\06_Image_0002.jpg)
![](assets\06_Image_0003.jpg)
![](assets\06_Image_0004.jpg)

**Count Plots for Categorical Features:**

Count plots were used to visualize the distribution of categorical features with respect to the target variable. This helps in understanding the frequency of each category and its association with the target.

![](assets\06_Image_0005.jpg)
![](assets\06_Image_0006.jpg)
![](assets\07_Image_0001.jpg)
![](assets\07_Image_0002.jpg)
![](assets\07_Image_0003.jpg)

**Count Plots for Top-N High Cardinality Features:**

For high cardinality features, count plots were generated for the most frequent categories to better understand their distribution.

![](assets\07_Image_0004.jpg)
![](assets\08_Image_0001.jpg)
![](assets\08_Image_0002.jpg)

### 2.4 Feature Encoding

To prepare data for modeling, feature encoding was applied to transform both categorical and numerical features into formats compatible with machine learning algorithms. This process involved several key steps:

1. **Standardization:** Numerical features like 'Age', 'Work/Study Hours', and 'CGPA' were standardized using StandardScaler. This ensured that all numerical features had a mean of 0 and a standard deviation of 1, preventing features with larger magnitude from dominating the model.
2. **Ordinal Encoding:** Ordinal features, such as 'Academic Pressure' and 'Sleep Time', were encoded with OrdinalEncoder. This encoding preserved the order of categories, allowing the model to interpret the relationship between these features correctly.
3. **One-Hot Encoding:** For low-cardinality categorical features, like 'Gender' and 'Dietary Habits', one-hot encoding was used. This method created binary columns for each category, ensuring the model could process the categorical data without assuming any ordinal relationship.
4. **Target Encoding:** High-cardinality categorical features, like 'City' and 'Profession', were target encoded using TargetEncoder. This technique reduced dimensionality while preserving the relationship between the feature and target variable, improving model performance.
5. **Consistency:** To maintain consistency between the training and test datasets, any missing columns in the test set were filled with zero, and columns were reordered to match those in the training set. This step ensured that the model processed data uniformly during evaluation.

### 2.5 Model Selection and Evaluation

The XGBoost algorithm was chosen for its ability to handle mixed data types and provide robust performance. The model was trained and evaluated through several key steps to ensure reliability and effectiveness:

1. **Stratified K-Fold Cross-Validation:** The dataset was split into five folds, ensuring that the proportion of the target variable was maintained across all folds. This provided a reliable estimate of model performance and helped to prevent overfitting.
2. **Hyperparameter Tuning:** The following hyperparameters were chosen:
   - `n_estimators=1000` (number of boosting rounds)
   - `learning_rate=0.01` (step size shrinkage to avoid overfitting)
   - `max_depth=5` (maximum depth of each tree)
   - `min_child_weight=3` (minimum sum of instance weights in a child)
   - `subsample=0.8` (fraction of samples used for training each tree)
   - `colsample_bytree=0.8` (fraction of features used)
   - `gamma=0.1` (minimum loss reduction required to make a split)
3. **Early Stopping:** The model was trained with early stopping, which monitored validation performance and stopped training if no improvement was seen for 50 rounds. This helped prevent overfitting and ensured the model was optimized.
4. **Performance:** During cross-validation, the model achieved a mean AUC-ROC score of 0.9754, showing strong performance in distinguishing between individuals at risk of depression and those who are not. Finally, the model was trained on the entire training dataset to maximize data usage, and this trained model was used to make predictions on the test dataset.

## 3. Results

The model's performance was evaluated on the test set using several key metrics:

- **AUC-ROC score:** 0.9745, showing the model's ability to effectively distinguish between the two classes.
- **Accuracy:** 0.9381, meaning it correctly classified 93.81% of test instances.
- **F1-score:** 0.8252, reflecting the model's ability to balance precision and recall, especially in the context of an imbalanced dataset.

To make predictions on the final test dataset, the model was retrained on the entire training dataset. This approach ensured that the model used all available data in the best way. The final predictions were then generated, with an accuracy score of 0.94005.

## 4. Conclusion

In conclusion, the feature encoding, model selection, and evaluation process demonstrated the effectiveness of the XGBoost algorithm in predicting depression risk. With an AUC-ROC score of 0.9745 and accuracy of 0.9381, the model performed strongly, highlighting the potential of machine learning to identify individuals at risk of depression. These results provide a strong foundation for future research and applications in the field of mental health.

## 5. References

1. Dr. Uthayasanker Thayasivam, PhD (U. Georgia), BSc Eng. (Hons) (Moratuwa).
2. "Predicting Depression: Machine Learning Challenge," [Online]. Available: https://www.kaggle.com/competitions/predicting-depression-machine-learning-challenge.
