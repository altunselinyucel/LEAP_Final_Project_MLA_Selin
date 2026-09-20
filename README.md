# LEAP_Final_Project_MLA_Selin.ipynb
Adult Census Model Comparison

# Predicting Income Brackets: Neural Networks vs. Classical Models

## 1. Project Overview & Problem Statement
This project was developed as part of the LEAP Machine Learning pipeline. The objective is to identify households likely to fall into a higher-income bracket (>50K) using survey data. The public agency requesting this model requires a comparison between a highly interpretable classical model (Logistic Regression) and a complex Deep Learning approach (Neural Network) to determine if the added complexity of a neural network is justified for this tabular classification task.

## 2. The Data Set
The analysis uses the **Adult Census Income** dataset from OpenML. Each row represents an individual, featuring a mix of continuous numeric variables (e.g., age, hours worked per week) and categorical variables (e.g., education, occupation). The target variable is binary, indicating whether the individual earns `>50K` (Positive Class, 1) or `<=50K` (Negative Class, 0).

## 3. Methodology
To ensure a fair comparison, both models were trained and evaluated on the exact same stratified data splits. 
*   **Preprocessing:** A `ColumnTransformer` was utilized. Numeric features were median-imputed and scaled via `StandardScaler`. Categorical features were mode-imputed and encoded using `OneHotEncoder`.
*   **Classical Model:** A Logistic Regression pipeline was validated using 5-fold Stratified Cross-Validation.
*   **Neural Network:** A Keras Sequential model (Dense layers with Dropout for regularization) was trained using the Adam optimizer and binary cross-entropy loss, with an `EarlyStopping` callback to prevent overfitting.

## 4. Key Results
Since the dataset is highly imbalanced (the >50K class is the minority), **F1 Score** was used as the primary evaluation metric, supported by overall accuracy.

*   **Logistic Regression:** F1 Score: ~0.655 | Accuracy: ~0.850
*   **Neural Network:** F1 Score: ~0.669 | Accuracy: ~0.855

*(Note: You can insert your confusion matrix images here using `![LR Matrix](images/lr_matrix.png)`)*

## 5. Interpretation & Ethical Considerations
Both models performed similarly. While the Neural Network achieved a marginally higher F1 score, both models struggled primarily with false negatives (failing to identify high-income households). Given the structured, tabular nature of this dataset, the Neural Network's slight performance edge does not justify its lack of interpretability. 

**Ethical Risks:** The dataset includes sensitive attributes such as race, sex, and nationality. Deploying a black-box neural network carries a significant risk of entrenching structural inequalities, as its internal decision-making cannot be easily audited. In a public sector setting where a false positive might unfairly exclude a vulnerable household from targeted benefits, the transparent and explainable Logistic Regression model is the mathematically and ethically sound choice.

## 6. Reflection
If granted more time, I would improve this pipeline by implementing advanced feature engineering and conducting a rigorous hyperparameter tuning phase using GridSearch for the Logistic Regression model. Additionally, applying SMOTE or adjusting class weights could help mitigate the severe class imbalance and improve the models' recall on the minority class.
