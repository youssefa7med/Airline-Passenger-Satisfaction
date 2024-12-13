# Airline Passenger Satisfaction

![Airline Passenger Satisfaction](https://cdn.dribbble.com/users/1081586/screenshots/3321346/flight_loading_vikas-singh.gif)

This project delves into the classification of airline passenger satisfaction, using advanced machine learning and deep learning techniques to uncover patterns and insights from the dataset. By leveraging a diverse range of models and extensive hyperparameter tuning, this project identifies the optimal approach for predicting passenger satisfaction.

## Overview
The **Airline Passenger Satisfaction** project is a comprehensive analysis aimed at predicting customer satisfaction levels based on various features like service quality, in-flight experience, and customer feedback. The project involved:

- Cleaning and preprocessing the dataset.
- Experimenting with multiple machine learning and deep learning models.
- Performing hyperparameter optimization to achieve the best results.

### Key Objectives
- Build a reliable classification model for predicting passenger satisfaction.
- Compare performance metrics across various algorithms.
- Utilize hyperparameter tuning to refine model performance.

## Dataset
The dataset for this project is sourced from Kaggle: [Airline Passenger Satisfaction](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction). It includes features such as:
- Customer type
- Travel class
- In-flight service ratings
- Satisfaction labels (Satisfied/Neutral or Dissatisfied)

### Dataset Preparation
#### Data Cleaning and Preprocessing:
1. **Outlier Removal**:
   - Identified and removed outliers using statistical methods to improve model robustness.
2. **Handling Missing Values**:
   - Addressed missing data through imputation techniques.
3. **Feature Encoding**:
   - Applied one-hot encoding for categorical variables.
4. **Feature Scaling**:
   - Standardized numerical features to ensure uniformity across model inputs.

## Methodology

### Model Exploration
The project experimented with a variety of machine learning models:

1. **Logistic Regression**
   - A baseline linear model for classification tasks.

2. **K-Nearest Neighbors (KNN)**
   - A non-parametric model that predicts based on the majority vote of nearest neighbors.

3. **Decision Tree**
   - A tree-based model for capturing feature interactions.

4. **Ensemble Methods**
   - **Random Forest**: An ensemble of decision trees to reduce overfitting.
   - **XGBoost**: A gradient-boosting framework with high accuracy.
   - **GradientBoostingClassifier**: A sequential boosting algorithm that improves weak learners.

### Deep Learning
Explored deep learning by constructing a neural network model. The architecture included:
- **Input Layer**: Processed numerical and categorical features.
- **Hidden Layers**: Fully connected layers with ReLU activation functions.
- **Output Layer**: Sigmoid activation for binary classification.

### Hyperparameter Tuning
Utilized **Hyperband** for hyperparameter tuning, optimizing parameters such as:
- Learning rate
- Number of layers and neurons
- Dropout rate
- Regularization parameters

### Evaluation Metrics
Models were evaluated using:
- **AUC-ROC Score**: To measure the model's ability to distinguish between classes.
- **Precision**: To assess the accuracy of positive predictions.
- **Recall**: To evaluate the sensitivity of the model.
- **F1-Score**: A harmonic mean of precision and recall.

## Results

### Performance Comparison
- **Logistic Regression**: Achieved baseline results with acceptable AUC-ROC scores.
- **KNN**: Performed moderately well but was computationally expensive for large datasets.
- **Decision Tree**: Showed high variance but was interpretable.
- **Random Forest**: Delivered robust results with balanced precision and recall.
- **XGBoost**: Outperformed other models with the highest AUC-ROC scores.
- **Deep Learning**: Provided competitive results after hyperparameter tuning, especially for complex feature interactions.

## Technologies Used
- **Python**: For implementation and analysis.
- **pandas**: Data manipulation and cleaning.
- **numpy**: Numerical computations.
- **scikit-learn**: Machine learning models and tools.
- **xgboost**: Gradient boosting framework.
- **tensorflow/keras**: Deep learning framework.
- **plotly express**: Interactive data visualization.

## Getting Started
### Prerequisites
Ensure the following libraries are installed:
- pandas
- numpy
- scikit-learn
- xgboost
- tensorflow
- plotly

### Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/youssefa7med/AirlineSatisfaction.git
   ```
2. **Navigate to the Project Directory**:
   ```bash
   cd AirlineSatisfaction
   ```
3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application
1. **Prepare Data**:
   - Place the dataset in the designated folder.
2. **Run the Script**:
   ```bash
   python main.py
   ```

## Usage
- Train multiple models and compare their performance.
- Visualize results using AUC-ROC curves and confusion matrices.
- Predict satisfaction levels for new data.

## Contributing
Contributions are welcome! Fork the repository, make improvements, and submit a pull request.

## License
This project is licensed under the MIT License. Refer to the [LICENSE](LICENSE) file for details.

## Acknowledgments
Special thanks to Kaggle for the dataset and the open-source community for the tools and resources used in this project.

