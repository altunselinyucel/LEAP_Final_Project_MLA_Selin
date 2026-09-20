# Project Evaluation & Summary Report: Machine Learning Classification Pipeline

## 1. Project Overview & Objective
This project serves as the final submission for the Machine Learning course, focusing on **Option 2: Comparing a Neural Network against a Classical Model**. The main objective was to build and evaluate two distinct models on the same tabular dataset to answer a central question: *Does a neural network outperform a well-crafted classical model on tabular data, and is its added complexity worthwhile?*

## 2. Dataset & Features
* **Dataset:** Adult Census Income dataset from OpenML.
* **Task:** Binary classification to predict whether an individual's annual income exceeds \$50,000 (`>50K` vs. `<=50K`).
* **Features:** A mix of continuous numerical features (e.g., age, hours worked per week) and categorical features (e.g., education, occupation).

## 3. Methodology & Workflow
To ensure a fair and robust comparison, both models were trained and tested on the exact same stratified data split:
* **Preprocessing Pipeline:** Implemented using scikit-learn's `ColumnTransformer` and `Pipeline`. Numeric values were median-imputed and scaled using `StandardScaler`. Categorical values were mode-imputed and encoded via `OneHotEncoder` (with `handle_unknown='ignore'`).
* **Classical Model:** A `LogisticRegression` classifier trained and validated using 5-fold Stratified Cross-Validation.
* **Neural Network:** A custom deep learning model built with Keras (Sequential API), featuring dense layers, ReLU activations, Dropout layers (0.3) for regularization, and a sigmoid output layer. It was optimized using Adam, binary cross-entropy loss, and an `EarlyStopping` callback to prevent overfitting.

## 4. Key Results & Evaluation
Since the income dataset is heavily imbalanced (the minority class represents high-income earners), **F1 Score** was prioritized alongside overall accuracy:
* **Logistic Regression:** Achieved a balanced F1 score and high interpretability, demonstrating that linear relationships capture the vast majority of predictive patterns in this dataset.
* **Neural Network:** Performed competitively, yielding metrics very close to the classical model without providing a substantial performance leap.

## 5. Critical Reflection & Ethical Considerations
* **Model Complexity vs. Value:** The added complexity, computational cost, and "black-box" nature of the neural network were ultimately not justified, as the classical Logistic Regression model performed nearly identically on this structured, tabular data.
* **Fairness & Ethics:** The dataset contains sensitive socioeconomic and demographic variables (such as race, sex, and native country). Deploying an opaque neural network in a public sector or benefit-targeting context creates high risks of reproducing hidden biases and institutional inequalities. Transparent models like Logistic Regression remain vastly superior when decisions directly impact human livelihoods.

## 6. Future Improvements
Given more time, I would expand this project by implementing advanced feature engineering (such as interaction terms), performing automated hyperparameter tuning via GridSearchCV for the classical model, and applying class-rebalancing techniques (like SMOTE) to improve minority-class recall.
