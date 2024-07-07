import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

import json
import time

import matplotlib.pyplot as plt
import seaborn as sns
def handle_manual(df:pd.DataFrame):
    """
    function that handle all the manual changes in the dataset.
    """
    df['Rainfall'] = df['Rainfall'].replace(-3, 0)
    df['Cloud9am'] = df['Cloud9am'].replace(999,7)
    df['WindDir3pm'] = df['WindDir3pm'].replace('zzzzzzzzzz\x85...', 'N')
    df = df.drop('CloudsinJakarta', axis=1)
    df = df.drop(index=5777)
    df['RainToday'] = df['RainToday'].replace({'Yes': 1, 'No': 0})
    df['RainTomorrow'] = df['RainTomorrow'].replace({'Yes': 1, 'No': 0})

    return df

def handle_NaNs(df:pd.DataFrame, categorials,debug_mode=False):
    """
    This function fills NaN values in each column with the mean value of the column.
    The mean value is calculated separately for rows where the 'Raintomorrow' property is 'yes' or 'no'.
    """
    method = 'Replace' #'Replace', DropColumns, DropRows
    df_copy = df.copy()
    if method == 'Replace':
        criteria = 'RainToday'
        for column in df_copy.columns:
            if column in categorials: # skip already categorials (manually selected)
                pass
            else:
                if df_copy[column].isna().any():
                    print(f'{column} has {df_copy[column].isna().sum()} missing values') if debug_mode else None  # Debug - print column name

                    # Calculate the mean values separately for 'yes' and 'no' in criteria
                    mean_yes = df_copy[df_copy[criteria] == 1][column].mean()
                    median_yes = df_copy[df_copy[criteria] == 1][column].median()

                    print(f'mean yes : {mean_yes}') if debug_mode else None  # Debug - print column mean yes
                    print(f'median yes : {median_yes}') if debug_mode else None  # Debug - print column mean yes

                    mean_no = df_copy[df_copy[criteria] == 0][column].mean()
                    median_no = df_copy[df_copy[criteria] == 0][column].median()

                    print(f'mean no : {mean_no}') if debug_mode else None  # Debug - print column mean no
                    print(f'median no : {median_no}') if debug_mode else None  # Debug - print column mean yes

                    # Replace NaN values with the corresponding mean value
                    if column == 'Sunshine':
                        df_copy.loc[(df_copy[criteria] == 1) & (df_copy[column].isna()), column] = mean_yes
                        df_copy.loc[(df_copy[criteria] == 0) & (df_copy[column].isna()), column] = mean_no
                    elif column == 'Evaporation':
                        df_copy.loc[(df_copy[criteria] == 1) & (df_copy[column].isna()), column] = median_yes
                        df_copy.loc[(df_copy[criteria] == 0) & (df_copy[column].isna()), column] = median_no
                else:
                    print(f'{column} has no missing values') if debug_mode else None  # Debug - print column name
    if method == 'DropColumns':
        df_copy = df_copy.drop('Sunshine', axis=1)
        df_copy = df_copy.drop('Evaporation', axis=1)
    if method == 'DropRows':
        df_copy = df_copy.dropna(subset=['Evaporation'])
        df_copy = df_copy.dropna(subset=['Sunshine'])
    return df_copy

def categorize_df(df: pd.DataFrame, categorials:list, N:int,
                  bins_mode:str='equal_range',df_mode:str='add'):
    """
    This function takes a DataFrame and makes all the non-categorical columns categorical.
    It converts the continuous values in the column to one of N categories representing the value magnitude.
    1 is the smallest, N is the largest.
    The function has 2 modes of bin separating:
    1. equal_range: each bin represents the same range of data // default mode
    2. equal_bins: each bin has the same amount of data
    The function has 2 modes of df management:
    1. replace - replace the original column with the categorical column
    2. add - add the new categorical column with the name feature_categorical
    """
    def handle_rainFall(df_copy):
        """
        Function that handles rainfall manually - make sure first column is 0-1 mm of rain.
        """
        if 'Rainfall' in df.columns:
            new_column_name = 'Rainfall_categorized'

            if bins_mode == 'equal_range':
                # Define custom bins for 'Rainfall'
                rainfall_min = df['Rainfall'].min()
                rainfall_max = df['Rainfall'].max()
                custom_bins = [rainfall_min, 1] + list(np.linspace(1, rainfall_max, N))

                # Remove duplicate bin edges if any
                custom_bins = sorted(set(custom_bins))

                df_copy[new_column_name] = pd.cut(df['Rainfall'], bins=custom_bins, labels=False, include_lowest=True) + 1
                print(f"Custom Equal Bins for 'Rainfall': {custom_bins}")
            elif bins_mode == 'equal_bins':
                    # Define custom bin for 0-1 and use qcut for the remaining bins
                    custom_bins = [0, 1] + list(pd.qcut(df['Rainfall'][df['Rainfall'] > 1], q=N - 1,
                                                       duplicates='drop').unique().categories.right)
                    df_copy[new_column_name] = pd.cut(df['Rainfall'], bins=custom_bins, labels=False, include_lowest=True) + 1
                    print(f"Custom Equal Bins for 'Rainfall': {custom_bins}")
            # Manage the DataFrame mode
            if df_mode == 'replace':
                df_copy['Rainfall'] = df_copy[new_column_name]
                df_copy.drop(columns=[new_column_name], inplace=True)
            elif df_mode == 'add':
                # Reorder columns to place new column next to the original
                cols = list(df_copy.columns)
                original_idx = cols.index('Rainfall')
                cols.insert(original_idx + 1, cols.pop(cols.index(new_column_name)))
                df_copy = df_copy[cols]
            else:
                raise ValueError("Invalid df_mode. Choose 'replace' or 'add'.")
        return df_copy
    # Make a copy of the DataFrame to avoid modifying the original one
    df_copy = df.copy()

    df_copy = handle_rainFall(df_copy)
    # Iterate over columns to categorize
    for column in df.columns:
        # not handle categorials and Rainfall (specific handling)
        if column not in categorials  and column != 'Rainfall':
            # Define new column name
            print(column)
            new_column_name = f"{column}_categorized"
            if bins_mode == 'equal_range':
                # Categorize the non-NaN values into N categories of equal range
                df_copy[new_column_name], bins = pd.cut(df_copy[column], bins=N, labels=False, retbins=True)
                df_copy[new_column_name] += 1
                print(f"Equal Range Bins for {column}: {bins}")
            elif bins_mode == 'equal_bins':
                # Categorize the non-NaN values into N categories with equal number of data points
                df_copy[new_column_name], bins = pd.qcut(df_copy[column], q=N, labels=False, retbins=True)
                df_copy[new_column_name] += 1
                print(f"Equal Size Bins for {column}: {bins}")
            else:
                raise ValueError("Invalid mode. Choose 'equal_range' or 'equal_bins'.")

            # Convert the new column to integer type for categorization
            df_copy[new_column_name] = df_copy[new_column_name].astype(int)

            # Manage the DataFrame mode
            if df_mode == 'replace':
                df_copy[column] = df_copy[new_column_name]
                df_copy.drop(columns=[new_column_name], inplace=True)
            elif df_mode == 'add':
                # Keep both original and new categorized column
                # Reorder columns to place new column next to the original
                cols = list(df_copy.columns)
                original_idx = cols.index(column)
                cols.insert(original_idx + 1, cols.pop(cols.index(new_column_name)))
                df_copy = df_copy[cols]
            else:
                raise ValueError("Invalid df_mode. Choose 'replace' or 'add'.")

    return df_copy


if __name__ == '__main__':
    NormalizationMethod = 'MinMaxScaler'


    path_to_file = r'E:\t2\machine\project\part_1\pythonProject\Xy_train.csv'
    df = pd.read_csv(path_to_file, encoding='ISO-8859-1')
    categorials = ['Location', 'WindGustDir', 'WindDir3pm', 'WindDir9am',
                   'Cloud9am', 'Cloud3pm', 'CloudsinJakarta', 'RainToday', 'RainTomorrow']
    df = handle_manual(df)
    df = handle_NaNs(df,categorials)
    df = pd.get_dummies(df,columns =['WindGustDir','WindDir9am','WindDir3pm','Location'])

    X = df.drop('RainTomorrow', 1).values
    Y = df['RainTomorrow'].values

    if NormalizationMethod == 'MinMaxScaler':
        minmax_scaler = MinMaxScaler()
        X_normalized = minmax_scaler.fit_transform(X)
        Y_normalized = Y
    elif NormalizationMethod == 'StandardScaler':
        standard_scaler = StandardScaler()
        X_normalized = standard_scaler.fit_transform(X)
        Y_normalized = Y


    X_train, X_test, Y_train, Y_test = train_test_split(X_normalized, Y_normalized, test_size=0.2, random_state=112)
    print(f"Train size: {X_train.shape[0]}")
    print(f"Test size: {X_test.shape[0]}")

    param_grid = {
        'hidden_layer_sizes': [(72,36,18,9)],
        'solver': ['adam'],
        'activation': ['logistic'],
        'learning_rate_init': [0.00105,0.00108,0.0011],
        'max_iter':[400,450,500,550,600],
        'alpha': [0.001]
    }

    model = MLPClassifier(random_state=1, verbose=False)
    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    # Define scoring metrics
    scoring = {
        'accuracy': 'accuracy',
        'f1': 'f1'
    }
    print("start time: " + str(time.ctime()))

    # Do grid search
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                               cv=kf, scoring=scoring, n_jobs=-1,
                             verbose=2,
                               refit='accuracy')
    # Fit GridSearchCV on the training data
    grid_search.fit(X_train, Y_train)


    # Print the best parameters and best score for the selected refit scoring method
    print("Best Parameters:", grid_search.best_params_)
    print("Best F1 Score (on training set):", grid_search.best_score_)

    results = {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'all_results': []
    }

    for i in range(len(grid_search.cv_results_['params'])):
        result = {
            'params': grid_search.cv_results_['params'][i],
            'mean_test_accuracy': grid_search.cv_results_['mean_test_accuracy'][i].item() if isinstance(
                grid_search.cv_results_['mean_test_accuracy'][i], np.generic) else
            grid_search.cv_results_['mean_test_accuracy'][i],
            'mean_test_f1': grid_search.cv_results_['mean_test_f1'][i].item() if isinstance(
                grid_search.cv_results_['mean_test_f1'][i], np.generic) else grid_search.cv_results_['mean_test_f1'][i],
            'rank_test_accuracy': grid_search.cv_results_['rank_test_accuracy'][i].item() if isinstance(
                grid_search.cv_results_['rank_test_accuracy'][i], np.generic) else
            grid_search.cv_results_['rank_test_accuracy'][i],
            'rank_test_f1': grid_search.cv_results_['rank_test_f1'][i].item() if isinstance(
                grid_search.cv_results_['rank_test_f1'][i], np.generic) else grid_search.cv_results_['rank_test_f1'][i]
        }
        result['params'] = {key: value.item() if isinstance(value, np.generic) else value for key, value in
                            result['params'].items()}
        results['all_results'].append(result)


    with open('grid_search_results.json', 'w') as f:
        json.dump(results, f, indent=4)

    # Evaluate the best model on the test set
    best_model = grid_search.best_estimator_
    Y_pred = best_model.predict(X_test)

    # Calculate and print the chosen metrics on the test set
    accuracy = accuracy_score(Y_test, Y_pred)
    f1 = f1_score(Y_test, Y_pred)

    print(f"Test Accuracy: {accuracy}")
    print(f"Test F1 Score: {f1}")
    print("finish time: " + str(time.ctime()))


