# Diabetes Prediction using K-Nearest Neighbors

This project aims to predict the occurrence of diabetes in patients based on various features using the K-Nearest Neighbors (KNN) algorithm from the scikit-learn library.

## Prerequisites

- Python (version 3.6 or higher)
- pandas
- numpy
- scikit-learn

## Dataset

The dataset used in this project is the `diabetes.csv` file, which should be placed in the `C:\Users\Admin\OneDrive\Documents\VSCode Practice\Practice\ML\` directory. This file contains various features related to diabetes, including glucose, blood pressure, skin thickness, insulin, BMI, and more.

## Code Explanation

1. The code starts by importing the necessary libraries: pandas, numpy, and several modules from scikit-learn.

2. The `diabetes.csv` file is read into a pandas DataFrame using `pd.read_csv()`.

3. Any zero values in the specified columns (`Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`) are replaced with the mean value of the respective column.

4. The features (X) and target variable (y) are separated from the DataFrame.

5. The dataset is split into training and testing sets using `train_test_split()` from scikit-learn, with a test size of 20%.

6. Feature scaling is performed using `StandardScaler()` from scikit-learn to ensure all features are on the same scale.

7. The KNN classifier is instantiated with `n_neighbors=11` (number of neighbors), `p=2` (Minkowski distance metric), and `metric="euclidean"` (Euclidean distance).

8. The classifier is trained on the training data using `fit()`.

9. Predictions are made on the test set using `predict()`.

10. The confusion matrix, F1-score, and accuracy score are calculated and printed using functions from `sklearn.metrics`.

## Usage

1. Ensure that you have the required Python libraries installed.
2. Place the `diabetes.csv` file in the specified directory (`C:\Users\Admin\OneDrive\Documents\VSCode Practice\Practice\ML\`).
3. Run the code script.
4. The output will display the confusion matrix, F1-score, and accuracy score for the predictions made by the KNN classifier.

## Contributing

Contributions to this project are welcome. If you find any issues or have suggestions for improvement, please open an issue or submit a pull request.

