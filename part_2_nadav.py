import copy

import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score,precision_score,recall_score, confusion_matrix
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
    # file params:
    PrePrepareData = True
    PrepareData = True
    splitType = 'cache' #original or from cache
    RonData = False
    singleTest = False
    GridSearch = True
    ## Data loading:
    path_to_file = r'E:\t2\machine\project\part_1\pythonProject\Xy_train.csv'


    if PrePrepareData:
        ## Data Preperation part:
        NormalizationMethod = 'MinMaxScaler'# MinMaxScalar or StandardScaler

        # Data analysis in case it is not yet prepared
        df = pd.read_csv(path_to_file, encoding='ISO-8859-1')
        categorials = ['Location', 'WindGustDir', 'WindDir3pm', 'WindDir9am',
                       'Cloud9am', 'Cloud3pm', 'CloudsinJakarta', 'RainToday', 'RainTomorrow']
        df = handle_manual(df)
        df = handle_NaNs(df,categorials)

    if PrepareData:
        # Create dummy variables
        df = pd.get_dummies(df,columns =['WindGustDir','WindDir9am','WindDir3pm','Location'])

        # Create X and Y vectors:
        X = df.drop('RainTomorrow', 1).values
        Y = df['RainTomorrow'].values

        # Normalize Data:
        if NormalizationMethod == 'MinMaxScaler':
            minmax_scaler = MinMaxScaler()
            X_normalized = minmax_scaler.fit_transform(X)
            Y_normalized = Y
        elif NormalizationMethod == 'StandardScaler':
            standard_scaler = StandardScaler()
            X_normalized = standard_scaler.fit_transform(X)
            Y_normalized = Y

        # Split Train and Test
        if splitType == 'original':
            X_train, X_test, Y_train, Y_test = train_test_split(X_normalized, Y_normalized, test_size=0.1, random_state=158)
        elif splitType == 'cache':
            # match indices:
            modified_i_test = pd.read_csv('Xy_post_process_internal_test.csv', encoding='ISO-8859-1')
            modified_i_train = pd.read_csv('Xy_post_process_train.csv', encoding='ISO-8859-1')
            df_copy= copy.deepcopy(df)
            merged_df = df_copy.merge(modified_i_test, on=['Pressure9am', 'Pressure3pm','MinTemp','Temp9am', 'Temp3pm','Evaporation'],
                                      how='left',indicator=True)
            test_indices = merged_df[merged_df['_merge'] == 'both'].index.tolist()

            df_copy= copy.deepcopy(df)
            merged_df = df_copy.merge(modified_i_train, on=['Pressure9am', 'Pressure3pm', 'Temp9am', 'Temp3pm'],
                                      how='left',indicator=True)
            train_indices = merged_df[merged_df['_merge'] == 'both'].index.tolist()

            # Check for similaity:
            print(len(train_indices) + len(test_indices))
            train_indices_set = set(train_indices)
            test_indices_set = set(test_indices)
            # Find intersection
            common_indices = train_indices_set.intersection(test_indices_set)
            print(common_indices)



            # Create training and testing sets
            X_train = (X_normalized[train_indices])
            Y_train = (Y_normalized[train_indices])
            X_test = (X_normalized[test_indices])
            Y_test = (Y_normalized[test_indices])
            print(len(test_indices))
            print(len(train_indices))

    if RonData:
        NormalizationMethod = 'MinMaxScaler'# MinMaxScalar or StandardScaler

        modified_i_train = pd.read_csv('Xy_post_process_train.csv', encoding='ISO-8859-1')
        modified_i_test = pd.read_csv('Xy_post_process_internal_test.csv', encoding='ISO-8859-1')
        full_df = pd.concat([modified_i_train, modified_i_test],axis=0)
        X = full_df.drop('RainTomorrow', 1).values
        Y = full_df['RainTomorrow'].values
        if NormalizationMethod == 'MinMaxScaler':
            minmax_scaler = MinMaxScaler()
            X_normalized = minmax_scaler.fit_transform(X)
            Y_normalized = Y
        elif NormalizationMethod == 'StandardScaler':
            standard_scaler = StandardScaler()
            X_normalized = standard_scaler.fit_transform(X)
            Y_normalized = Y

        X_train = X_normalized[0:len(modified_i_train)]
        Y_train = Y_normalized[0:len(modified_i_train)]
        X_test = X_normalized[len(modified_i_train):]
        Y_test = Y_normalized[len(modified_i_train):]

    if singleTest:
        print("train T proportion: " + str((np.count_nonzero(Y_train == 1)) / (len(Y_train))))
        print("test T proportion: " + str((np.count_nonzero(Y_test == 1)) / (len(Y_test))))

        model = MLPClassifier(random_state=1, verbose=False)
        model.fit(X_train, Y_train)

        # Predict on the test set
        y_pred = model.predict(X_test)

        # Evaluate the model using F1 score
        f1 = f1_score(Y_test, y_pred)
        accuracy = accuracy_score(Y_test, y_pred)
        precision = precision_score(Y_test, y_pred)
        recall = recall_score(Y_test, y_pred)
        print(f"F1 Score: {f1}")
        print(f"Accuracy Score: {accuracy}")
        print(f"precision: {precision}")
        print(f"recall Score: {recall}")
        # Predict on the training set
        y_train_pred = model.predict(X_train)


        # Evaluate the model using F1 score and accuracy on the training set
        f1_train = f1_score(Y_train, y_train_pred)
        accuracy_train = accuracy_score(Y_train, y_train_pred)
        precision = precision_score(Y_train, y_train_pred)
        recall = recall_score(Y_train, y_train_pred)
        print('---')
        print(f"Train F1 Score: {f1_train}")
        print(f"Train Accuracy Score: {accuracy_train}")
        print(f"Train F1 precision: {precision}")
        print(f"Train recall Score: {recall}")

        conf_matrix = confusion_matrix(Y_test, y_pred)

        plt.figure(figsize=(10, 7))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 20})
        plt.title('Confusion Matrix Heatmap', fontsize=24)
        plt.xlabel('Predicted', fontsize=24)
        plt.ylabel('Actual', fontsize=24)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.show()
    # Do Grid Search:
    if GridSearch:
        param_grid = {
            'solver': ['adam'],
            'hidden_layer_sizes': [200,250,(36,18),(36,18,3)],
            'activation': ['relu', 'tanh', 'logistic','identity','softmax'],
            'learning_rate_init': [0.001],
            'max_iter':[500],
        }
        # Set model
        model = MLPClassifier(random_state=1, verbose=False)
        kf = KFold(n_splits=5, shuffle=True, random_state=1)
        # Define scoring metrics
        print("start time: " + str(time.ctime()))

        # # Define grid search

        # grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
        #                            cv=kf, scoring='f1', verbose=2)
        # #
        # # Do the actual grid
        # grid_search.fit(X_train, Y_train)
        # results = pd.DataFrame(grid_search.cv_results_)
        # results.to_csv('grid_search5.csv', index=False)

        results = pd.read_csv('grid_search5.csv')
        # Filter the relevant columns
        results = results[['param_hidden_layer_sizes', 'param_activation', 'mean_test_score']]
        results_pivot = results.pivot('param_hidden_layer_sizes', 'param_activation', 'mean_test_score')

        # Plot the heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(results_pivot, annot=True, fmt=".3f", cmap="viridis", annot_kws={"size": 20})
        plt.title('Grid Search f1 Heatmap', fontsize=24)
        plt.xlabel('learning rate', fontsize=24)
        plt.ylabel('param_hidden_layer_sizes', fontsize=24)
        plt.show()

        # Print the best parameters and best score for the selected refit scoring method
        print("Best Parameters:", grid_search.best_params_)
        print("Best F1 Score (on training set):", grid_search.best_score_)

        # Show best results from grid search:
        best_model = grid_search.best_estimator_
        Y_pred = best_model.predict(X_test)

        # Calculate and print the chosen metrics on the test set
        accuracy = accuracy_score(Y_test, Y_pred)
        f1 = f1_score(Y_test, Y_pred)

        print(f"Test Accuracy: {accuracy}")
        print(f"Test F1 Score: {f1}")
        print("finish time: " + str(time.ctime()))



