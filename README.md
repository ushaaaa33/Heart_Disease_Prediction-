# Heart Disease Prediction System

## Overview
This project is a machine learning-based system designed to predict the likelihood of heart disease in patients using health-related parameters. The system leverages a Random Forest classifier trained on a heart disease dataset from Kaggle. It includes data exploration, model training, evaluation, and a user-friendly web interface built with Streamlit for real-time predictions.

The goal is to assist healthcare professionals by providing an accessible tool for early detection of heart disease based on clinical features.

## Dataset Description
The dataset used is the **Heart Disease UCI** dataset from Kaggle, which contains 303 instances with 14 attributes. It includes the following features:

- **age**: Age of the patient (in years)
- **sex**: Gender (1 = male, 0 = female)
- **cp**: Chest pain type (0-3)
- **trestbps**: Resting blood pressure (in mm Hg)
- **chol**: Serum cholesterol (in mg/dl)
- **fbs**: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
- **restecg**: Resting electrocardiographic results (0-2)
- **thalach**: Maximum heart rate achieved
- **exang**: Exercise-induced angina (1 = yes, 0 = no)
- **oldpeak**: ST depression induced by exercise relative to rest
- **slope**: Slope of the peak exercise ST segment (0-2)
- **ca**: Number of major vessels colored by fluoroscopy (0-3)
- **thal**: Thalassemia (0 = normal, 1 = fixed defect, 2 = reversible defect)
- **target**: Presence of heart disease (1 = yes, 0 = no)

The dataset has no missing values and is balanced for binary classification.

## Tech Stack
- **Programming Language**: Python
- **Libraries**:
  - Data Manipulation: Pandas, NumPy
  - Visualization: Matplotlib, Seaborn
  - Machine Learning: Scikit-learn
  - Web Interface: Streamlit
  - Model Serialization: Joblib
- **Environment**: Jupyter Notebook for development, Streamlit for deployment
- **Other Tools**: Git for version control

## Project Workflow
1. **Data Loading and Exploration**:
   - Load the dataset (`heart.csv`).
   - Perform exploratory data analysis (EDA) including checking for null values, data distribution, correlations, and visualizations (histograms, box plots, heatmaps).

2. **Data Preprocessing**:
   - Split the data into features (X) and target (y).
   - Perform train-test split (80% train, 20% test).

3. **Model Training and Evaluation**:
   - Train multiple machine learning models: Logistic Regression, Decision Tree, Gradient Boosting, Gaussian Naive Bayes, K-Nearest Neighbors, Multi-Layer Perceptron, Support Vector Machine, and Random Forest.
   - Evaluate models based on accuracy scores.
   - Select the best-performing model (Random Forest) and save it using Joblib.

4. **Prediction Script**:
   - Create a `predict.py` script to load the saved model and make predictions on new input data.

5. **Web Application**:
   - Build a Streamlit app (`app.py`) for user interaction, allowing input of patient details and displaying prediction results.

6. **Deployment**:
   - Run the Streamlit app locally or deploy to a platform like Heroku or Streamlit Cloud.

## Machine Learning Model
### Algorithm Used: Random Forest Classifier
- **Why Chosen**: Among the tested models (Logistic Regression, Decision Tree, Gradient Boosting, etc.), Random Forest achieved the highest accuracy on the test set. Random Forest is an ensemble method that combines multiple decision trees, reducing overfitting and improving generalization. It handles both numerical and categorical features well, is robust to outliers, and provides feature importance insights.

### Model Training Details
- **Hyperparameters**: Default settings were used (e.g., n_estimators=100 for Random Forest).
- **Scaling**: Applied StandardScaler for SVM; other models used raw features.
- **Cross-Validation**: Not explicitly performed; relied on train-test split for evaluation.

## Model Performance
The following table summarizes the accuracy scores of various models on the test set:

| Model                  | Accuracy (%) |
|------------------------|--------------|
| Random Forest         | 88.52       |
| Decision Tree         | 78.69       |
| Gradient Boosting     | 81.97       |
| Gaussian Naive Bayes  | 86.89       |
| K-Nearest Neighbors   | 68.85       |
| Multi-Layer Perceptron| 83.61       |
| Support Vector Machine| 83.61       |

- **Best Model**: Random Forest with ~88.52% accuracy.
- **Evaluation Metrics**: Accuracy was the primary metric. Additional metrics like precision, recall, and F1-score were computed for Logistic Regression (as an example), showing balanced performance.
- **Confusion Matrix**: Generated for Logistic Regression, indicating good classification with minimal false positives/negatives.

## Results & Key Insights
- **Key Findings**:
  - Random Forest outperformed other models due to its ensemble nature, making it suitable for this dataset.
  - Feature correlations (e.g., via heatmap) showed relationships between variables like age, cholesterol, and heart disease.
  - Data distribution analysis revealed no significant outliers, and the dataset is suitable for modeling.
  - The target variable is balanced (165 positive, 138 negative cases).

- **Insights from EDA**:
  - Histograms and box plots indicated normal distributions for most features.
  - Correlation heatmap highlighted key predictors like `thalach`, `oldpeak`, and `ca`.
  - No null values were present, simplifying preprocessing.

- **Prediction Example**:
  - Input: [52, 1, 0, 125, 212, 0, 1, 168, 0, 1.0, 2, 2, 3]
  - Output: No heart disease (prediction = 0)

- **Limitations**:
  - Model accuracy could be improved with hyperparameter tuning or more data.
  - The dataset is relatively small (303 samples), which may limit generalization.
  - Assumes input data is clean and in the correct format.

## Installation and Usage
1. **Clone the Repository**:
   ```
   git clone <repository-url>
   cd heart-disease-prediction
   ```

2. **Install Dependencies**:
   ```
   pip install -r requirements.txt
   ```

3. **Run the Jupyter Notebook** (for exploration):
   ```
   jupyter notebook main.ipynb
   ```

4. **Run the Streamlit App**:
   ```
   streamlit run app.py
   ```
   - Open the provided URL in your browser and input patient details for predictions.

5. **Make Predictions via Script**:
   - Use `predict.py` in your code or modify it for batch predictions.

## Contributing
Contributions are welcome! Please fork the repository, make changes, and submit a pull request. Ensure to follow best practices for code quality and documentation.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
- Dataset sourced from Kaggle (Heart Disease UCI).
- Inspired by various ML tutorials and healthcare AI projects.
