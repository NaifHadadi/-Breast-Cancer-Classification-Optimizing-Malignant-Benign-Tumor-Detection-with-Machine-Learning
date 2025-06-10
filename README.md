# Breast Cancer Classification: Optimizing Malignant/Benign Tumor Detection with Machine Learning

This project, developed for an ML course, classifies breast tumors as malignant or benign using Logistic Regression and an MLP neural network. We carefully preprocess the data and select the most important features through tree-based analysis and correlation pruning. Our evaluation focuses on key metrics like precision, recall, and especially ROC/AUC scores – crucial for handling the dataset's imbalance and ensuring reliable cancer detection. The MLP model achieves 94.7% accuracy with a near-perfect 0.992 AUC, demonstrating strong performance for this medical diagnostic task.

## Tools Used

- Python
- PyTorch
- Scikit-learn
- Pandas
- Matplotlib / Seaborn



Key Results
--
Using a randomly selected three features

<img width="275" alt="image" src="https://github.com/user-attachments/assets/7cba107e-6064-484f-b33f-3112a57ad0ed" />

We got a reasonable result for both models:

<img width="534" alt="image" src="https://github.com/user-attachments/assets/ad7dede1-807d-4122-8845-f680fc7e167e" />

# Model Enhancement

To improve model performance and interpretability, we applied a structured feature selection pipeline using a tree-based approach and correlation analysis.
<br>
<br>

### Step 1: Feature Importance with Extra Trees
We trained an ExtraTreesClassifier with 100 estimators on the standardized breast cancer dataset to quantify the relative importance of all 30 features. The model computes feature importances based on how frequently and effectively features are used in decision splits.

The top 15 most informative features were extracted and visualized using a color-normalized bar plot

![image](https://github.com/user-attachments/assets/52e691d2-c921-40a2-a7ed-eaf27449ed05)

<br>
<br>

### Step 2: Correlation Pruning

To address redundancy and multicollinearity:

We filtered out features with importance scores below 5% (i.e., importance ≥ 0.05), yielding a more focused set with 8 features.

A correlation matrix of the selected features (below) revealed several pairs with correlations above 85%:
![image](https://github.com/user-attachments/assets/0ba1d371-238d-404f-acb8-972bf6e4bb08)


For each highly correlated pair, we retained only the feature with the higher importance score, ensuring minimal loss of information and avoiding overfitting.

The resulting feature set was reduced from 8 to the top 3 slightly correlated features:
| Index | worst radius | mean concavity | worst concave points |
|-------|--------------|----------------|-----------------------|
| 0     | 25.38        | 0.3001         | 0.2654                |
| 1     | 24.99        | 0.0869         | 0.1860                |
| 2     | 23.57        | 0.1974         | 0.2430                |
| 3     | 14.91        | 0.2414         | 0.2575                |
| 4     | 22.54        | 0.1980         | 0.1625                |

<br>

### Model Performance Comparison Before and After Feature Selection

The table below summarizes the key evaluation metrics for the MLP and Logistic Regression models on the breast cancer classification task, comparing their performance before and after the feature selection process. Notably, both models show improved accuracy and reduced loss post-selection.
| Model                   | Metric   | Before Feature Selection | After Feature Selection |
| ----------------------- | -------- | ------------------------ | ----------------------- |
| **MLP Model**           | Accuracy | 91.40%                   | 94.70%                  |
|                         | Loss     | 0.2363                   | 0.1153                  |
| **Logistic Regression** | Accuracy | 90.97%                   | 93.92%                  |
|                         | Loss     | 0.2760                   | 0.2147                  |

<br>

# Evaluation

To comprehensively assess model performance, we use multiple metrics and visualizations beyond simple accuracy. This is critical for imbalanced datasets like breast cancer classification, where correctly detecting malignant cases is essential. 

<img width="800" alt="image" src="https://github.com/user-attachments/assets/69d18429-19ad-4ed7-a468-c9ca21b84481" />

### Model Performance Summary

| Metric                | MLP Model  | Logistic Regression |
|-----------------------|------------|---------------------|
| **Accuracy**          | 94.70%     | 93.92%              |
| **Loss**              | 0.1153     | 0.2147              |
| **Precision**         | 0.9714     | 0.9333              |
| **Recall**              | 0.9444     | 0.9722              |
| **F1-Score**          | 0.9577     | 0.9524              |

- **Accuracy & Loss:**  
  MLP outperforms Logistic Regression with 94.7% vs. 93.92% accuracy and lower loss (0.1153 vs. 0.2147), which suggests better optimization.

- **Precision vs. Recall:**  
  MLP has higher precision (97.1%) but lower recall (94.4%) — fewer false positives but misses some cancers.  
  Logistic Regression has higher recall (97.2%) but lower precision (93.3%) — catches more cancers but more false alarms.  
  Choice depends on clinical priorities.

-  **F1-Score:**  
  MLP shows a comparable F1-score, making it more balanced overall.

#### The above metrics suggest that:
- If false negatives are critical (e.g., cancer screening), LR’s higher recall might be preferable despite lower precision.

- If false positives are costly (e.g., unnecessary biopsies), MLP’s precision wins.

<br>

## Threshold Study, ROC & AUC 

The Receiver Operating Characteristic (ROC) curve and Area Under the Curve (AUC) score evaluate how well the models distinguish between malignant and benign cases across different classification thresholds. Accuracy alone can be misleading in imbalanced data, while ROC curves show the trade-off between true positive rate and false positive rate. Below plots show a comparison between ROC for enhanced versions of the two models


![image](https://github.com/user-attachments/assets/be84345c-7f94-468c-8c41-87d72ed69b55)

<br>

## Conclusion:

Feature selection improved both models' performance. The MLP model outperforms Logistic Regression in most metrics, with some threshold tuning for the MLP model based on the  ROC data above,    it will become the preferred final model for distinguishing malignant from benign cases and other problems as well. This project underscores how model selection, feature engineering, and threshold tuning collectively enhance real-world AI solutions. The methodology can be replicated in other domains where interpretability and performance are equally critical.



# Visualization of Model Predictions
Below is a 2D projection visualization showing two key features from the best-performing model (MLP), highlighting the correct and incorrect predictions:

![image](https://github.com/user-attachments/assets/7b5a30e3-35ec-4ef4-9520-37e730df934a)


