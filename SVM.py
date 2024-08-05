import pandas as pd
from sklearn import svm
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, make_scorer, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


def feature_scaling(x_train, x_test):
    numerical_features = ["MinTemp", "MaxTemp", "Pressure9am", "Humidity3pm", "WindGustSpeed", "Temp3pm", "Sunshine",
                          "Humidity9am", "Humidity3pm", "Pressure3pm", "Cloud3pm",
                          "HumidityChange", "PressureChange"]

    scaler = StandardScaler()
    x_train[numerical_features] = scaler.fit_transform(x_train[numerical_features])
    x_test[numerical_features] = scaler.fit_transform(x_test[numerical_features])
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=123)
    return x_train, x_test, cv


def read_data(path_to_file: str):
    df = pd.read_csv(path_to_file, encoding='ISO-8859-1')
    print(f'Data Describing')
    print(df.describe())
    print(f'Null values summary')
    print(df.isnull().sum())

    duplicates_exist = df.duplicated().any()
    if duplicates_exist:
        print(f'Duplicate data is exist in this dataset.')

    return df


def svm_hyperparameter_tuning(x_train, y_train, x_test, y_test):
    smote = SMOTE()
    x_train, y_train = smote.fit_resample(x_train, y_train)
    svm_model = SVC(kernel='linear')

    # Define the parameter grid
    param_grid = {
        'C': [1, 10, 100]

    }
    # Setup the grid search
    grid_search = GridSearchCV(estimator=svm_model, param_grid=param_grid, scoring=make_scorer(accuracy_score),
                               cv=5, verbose=1, n_jobs=-1)

    grid_search.fit(x_train, y_train)

    print("Best parameters found: ", grid_search.best_params_)
    print("Best cross-validation accuracy_score: ", grid_search.best_score_)

    best_model = grid_search.best_estimator_
    test_accuracy = accuracy_score(y_test, best_model.predict(x_test))
    print("Test set accuracy_score: ", test_accuracy)

    svm_best = grid_search.best_estimator_
    print("Model intercept: ", svm_best.intercept_)
    print("Model coefficients: ", svm_best.coef_)

    y_pred_test = svm_model.predict(x_test)
    final_accuracy = accuracy_score(y_test, y_pred_test)
    final_conf_matrix = confusion_matrix(y_test, y_pred_test)
    final_classification_report = classification_report(y_test, y_pred_test)

    print("Test Set accuracy_score:", final_accuracy)
    print("Test Set Confusion Matrix:\n", final_conf_matrix)
    print("Test Set Classification Report:\n", final_classification_report)

    # Print SVM model coefficients and support vectors
    # print("Model coefficients (weights):", svm_model.coef_)
    print("Model intercept:", svm_model.intercept_)
    print("Support vectors (first few shown):", svm_model.support_vectors_[:5])
    print("Number of support vectors for each class:", svm_model.n_support_)

    return best_model, svm_best


def train_and_evaluate_svm(x_train, y_train, x_test, y_test, cv):
    smote = SMOTE()
    x_train, y_train = smote.fit_resample(x_train, y_train)
    svm_model = SVC(kernel='linear', C=10)  # baseline_parameters
    accuracies = []
    conf_matrices = []

    # Cross-validation
    fold = 1
    for train_index, test_index in cv.split(x_train, y_train):
        # Split data into k-fold train/test sets
        X_train_fold, X_test_fold = x_train.iloc[train_index], x_train.iloc[test_index]
        y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]

        # Train SVM
        svm_model.fit(X_train_fold, y_train_fold)
        y_pred_fold = svm_model.predict(X_test_fold)

        # Evaluate SVM
        score = accuracy_score(y_test_fold, y_pred_fold)
        conf_matrix = confusion_matrix(y_test_fold, y_pred_fold)
        accuracies.append(score)
        conf_matrices.append(conf_matrix)

        print(f"Fold {fold}: score = {score}")
        fold += 1


    mean_score = np.mean(score)
    print(f"Mean score across all folds: {mean_score}")

    svm_model.fit(x_train, y_train)

    y_pred_test = svm_model.predict(x_test)
    final_mean_score = accuracy_score(y_test, y_pred_test)
    final_conf_matrix = confusion_matrix(y_test, y_pred_test)
    final_classification_report = classification_report(y_test, y_pred_test)

    print("Test Set Accuracy:", final_mean_score)
    print("Test Set Confusion Matrix:\n", final_conf_matrix)
    print("Test Set Classification Report:\n", final_classification_report)

    # print("Model coefficients (weights):", svm_model.coef_)
    print("Model intercept:", svm_model.intercept_)
    print("Support vectors (first few shown):", svm_model.support_vectors_[:5])
    print("Number of support vectors for each class:", svm_model.n_support_)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    sns.heatmap(confusion_matrix(y_test, y_pred_test), annot=True, fmt="d", ax=axes[0], cmap="Blues")
    axes[0].set_title('Training Set Confusion Matrix')
    axes[0].set_xlabel('Predicted labels')
    axes[0].set_ylabel('True labels')

    sns.heatmap(confusion_matrix(y_test, y_pred_test), annot=True, fmt="d", ax=axes[1], cmap="Blues")
    axes[1].set_title('Test Set Confusion Matrix')
    axes[1].set_xlabel('Predicted labels')
    axes[1].set_ylabel('True labels')

    plt.show()

    feature_weights = svm_model.coef_[0]
    features = x_train.columns

    for feature, weight in zip(features, feature_weights):
        print(f"Feature: {feature}, Weight: {weight}")

    feature_importance = pd.DataFrame({"Feature": features, "Weight": feature_weights})
    print(feature_importance)
    return final_mean_score, final_conf_matrix, final_classification_report


def main(path_to_file: str, path_to_file_train: str, path_to_file_test: str):
    df_train = pd.read_csv(path_to_file_train)
    df_test = pd.read_csv(path_to_file_test)
    df = read_data(path_to_file)
    selected = ['MinTemp', 'MaxTemp', 'Sunshine', 'WindGustSpeed', 'Humidity9am', 'Humidity3pm', 'Pressure9am',
                'Pressure3pm', 'Cloud3pm', 'Temp3pm', 'HumidityChange', 'PressureChange', 'Location_Bendigo',
                'Location_Canberra', 'Location_Hobart', 'Location_Melbourne',
                'Location_Sydney', 'Location_Uluru', 'WindGustDir_E', 'WindGustDir_N', 'WindGustDir_S', 'WindGustDir_W',
                'WindDir9am_E', 'WindDir9am_N', 'WindDir9am_S', 'WindDir9am_W', 'WindDir3pm_E', 'WindDir3pm_N',
                'WindDir3pm_S', 'WindDir3pm_W', 'WindSpeed9am_calm'
        , 'WindSpeed9am_moderate_breeze', 'WindSpeed9am_strong_breeze', 'WindSpeed3pm_calm',
                'WindSpeed3pm_moderate_breeze', 'WindSpeed3pm_strong_breeze']
    features = selected + ["RainTomorrow"]
    print(f'Test set samples: {len(df_test)}')
    print(f'Train set samples: {len(df_train)}')
    x_train = df_train[selected]
    x_test = df_test[selected]
    print(x_train.columns)
    print(x_train.columns)
    y_train = df_train['RainTomorrow']
    y_test = df_test['RainTomorrow']
    x_train, x_test, cv = feature_scaling(x_train, x_test)
   # svm_hyperparameter_tuning(x_train, y_train, x_test, y_test)

    train_and_evaluate_svm(x_train, y_train, x_test, y_test, cv)


if __name__ == '__main__':
    path_to_file = r"C:\Users\edenm\OneDrive\שולחן העבודה\לימודים\Xy_train_processed.csv"
    path_to_file_train = r"C:\Users\edenm\OneDrive\שולחן העבודה\לימודים\Xy_post_process_train.csv"
    path_to_file_test = r"C:\Users\edenm\OneDrive\שולחן העבודה\לימודים\Xy_post_process_internal_test.csv"
    main(path_to_file, path_to_file_train, path_to_file_test)