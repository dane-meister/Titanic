# 🚢 Titanic Survival Prediction - ML & Data Science Project

> Predicting survival outcomes from the Titanic disaster using supervised machine learning.

## Project Objective

This project explores the famous Titanic dataset to predict passenger survival using machine learning. We aim to analyze which factors impacted survival the most and develop predictive models with solid performance using feature engineering, data visualization, and classification techniques.

## Contributors

- Dane Meister (Data Preprocessing, EDA, Visualizations, Presentation)
- Michael Lieberman (Model Development, Evaluation, Documentation)

## Project Structure

- `train.csv` / `test.csv` — Raw datasets
- `submission_logreg.csv`, `submission_knn.csv`, `submission_svm.csv` — Final model outputs
- `Titanic_Project_Presentation.pptx` — Summary slides
- `cse351project.ipynb` — All code and workflow in Google Colab

## Key Steps

### Data Cleaning

- **Age**: Filled with median grouped by `Title`
- **Fare**: Median imputation + log transform
- **Embarked**: Filled with mode (`'S'`)
- **Outliers**: Clipped Fare at 99th percentile
- **Dropped**: `Cabin`, `Ticket`, `Name` (after extracting info)

### Feature Engineering

- `Title` extracted from names, rare titles grouped
- `FamilySize` = `SibSp` + `Parch` + 1
- `IsAlone` binary indicator
- `HasCabin` binary indicator
- Binned `AgeGroup` into 5 ranges
- Label encoded `Sex`, `Embarked`, `Title`

### Exploratory Data Analysis (EDA)

- Women and children had highest survival
- First-class passengers fared much better
- Family presence generally increased odds
- Passengers with cabins had better survival chances

### Models & Performance

| Model               | Accuracy | Precision | Recall | F1 Score |
|--------------------|----------|-----------|--------|----------|
| Logistic Regression| 80.6%    | 0.758     | **0.725**  | 0.741    |
| KNN (k=5)           | 80.4%    | 0.765     | 0.705  | 0.734    |
| SVM (RBF)           | **81.6%**| **0.801** | 0.693  | **0.743**|

> **SVM** had the best accuracy and F1 score.  
> **Logistic Regression** had the best recall (most sensitive).  
> All models used 5-fold cross-validation for evaluation.

## Outputs

Final predictions saved as:

- `submission_logreg.csv`
- `submission_knn.csv`
- `submission_svm.csv`


## Dependencies

```bash
pandas
numpy
seaborn
matplotlib
scikit-learn

```
Note that this was as an academic project at Stony Brook University.
