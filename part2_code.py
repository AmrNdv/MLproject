"""
Here is our code for part 2 of the project.
The code is devided by machine learning method.
To run each part separately - copy the right part to a clean python notebook
"""
# Decision tree and random forest: Lines 10 - 605
# Neural Networks: Lines 610 - 988
# Kmeans - unsupervised learning: 992 - 1254
# Support Vector Machines: 1257 - 1453


#Start Desicion tree part

import time
import joblib
import os.path
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.tree import plot_tree
from matplotlib import pyplot as plt
import statsmodels.formula.api as smf
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV, RandomizedSearchCV


pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


def visualize_tree(tree, feature_names, max_depth=None):
    plt.figure(figsize=(20, 10))
    plot_tree(tree, feature_names=feature_names, class_names=['No', 'Yes'],
              filled=True, rounded=True, max_depth=max_depth)
    plt.savefig(r'C:\Users\Ron Gabo\Desktop\ml_course\תמונות של גרפים\decision_tree.png', format='png', dpi=300)  # Save with high DPI
    plt.show()


def read_data(path_to_file: str):
    df = pd.read_csv(path_to_file,encoding='ISO-8859-1')
    print(f'Data Describing')
    print(df.describe())
    print(f'Null values summary')
    print(df.isnull().sum())

    duplicates_exist = df.duplicated().any()
    if duplicates_exist:
        print(f'Duplicate data is exist in this dataset.')

    return df


def write_list_to_excel(data_list: [], file_path: str):
    df = pd.DataFrame(data_list, columns=['target'])
    df.to_excel(file_path, index=False)


def split_data_train_test(df: pd.DataFrame, test_pcg=10):
    rain_tomorrow_yes = df[df['RainTomorrow'] == 1]
    rain_tomorrow_no = df[df['RainTomorrow'] == 0]

    df_train_yes, df_test_yes = train_test_split(rain_tomorrow_yes, test_size=test_pcg / 100, random_state=123)
    df_train_no, df_test_no = train_test_split(rain_tomorrow_no, test_size=test_pcg / 100, random_state=123)
    df_train = pd.concat([df_train_yes, df_train_no], ignore_index=True)
    df_test = pd.concat([df_test_yes, df_test_no], ignore_index=True)

    return df_train, df_test


def feature_scaling(x_train, x_test):
    def identify_continuous_numerical_columns(df: pd.DataFrame, unique_threshold: int = 20) -> list:
        continuous_numerical_columns = []

        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                unique_values = df[col].nunique()
                if unique_values > unique_threshold:
                    continuous_numerical_columns.append(col)

        return continuous_numerical_columns

    numerical_features = identify_continuous_numerical_columns(x_train, 10)
    scaler = StandardScaler()
    x_train[numerical_features] = scaler.fit_transform(x_train[numerical_features])
    x_test[numerical_features] = scaler.fit_transform(x_test[numerical_features])
    cv = StratifiedKFold(n_splits=10, shuffle=True)
    return x_train, x_test, cv


def evaluate_model(model, x, y_gt, y_predict):
    accuracy = accuracy_score(y_gt, y_predict)
    f1 = f1_score(y_gt, y_predict, average='weighted')
    roc_auc = roc_auc_score(y_gt, y_predict)
    print('Accuracy: ', accuracy)
    print('F1 Score: ', f1)
    print('AUC(ROC): ', roc_auc)
    print()
    print("Classification Report: ")
    print(classification_report(y_gt, y_predict))

    # ROC AUC
    prob = model.predict_proba(x)
    prob = prob[:, 1]
    fper, tper, _ = roc_curve(y_gt, prob)
    auc_scr = auc(fper, tper)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].plot(fper, tper, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc_scr)
    axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('Receiver Operating Characteristic (ROC) Curve')
    axes[0].legend(loc="lower right")

    sns.heatmap(confusion_matrix(y_gt, y_predict), ax=axes[1], annot=True, cbar=False, fmt='.0f')
    axes[1].set_xlabel('Prediction')
    axes[1].set_ylabel('Ground Truth')

    plt.show()
    return accuracy, f1, roc_auc


def train_model_with_grid_search(model, param_grid, x_train, y_train, x_val, y_val, skf, output_path):

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', n_jobs=-1, cv=skf)
    grid_search.fit(x_train, y_train)

    best_model = grid_search.best_estimator_
    best_param = grid_search.best_params_
    joblib.dump(best_model, output_path)
    y_pred = best_model.predict(x_val)
    print(f'Test Eval:')
    accuracy, f1, roc_auc = evaluate_model(model=best_model, x=x_val, y_gt=y_val,
                                           y_predict=y_pred)
    print(f'Train Eval:')
    y_train_pred = best_model.predict(x_train)
    accuracy_train, f1_train, roc_auc_train = evaluate_model(model=best_model, x=x_train, y_gt=y_train,
                                                             y_predict=y_train_pred)


    results = pd.DataFrame(grid_search.cv_results_)
    return model, accuracy, f1, roc_auc, best_param, results


def full_decision_tree_model(x_train, y_train, x_test, y_test):
    full_tree = DecisionTreeClassifier()

    full_tree.fit(x_train, y_train)
    # train set
    y_train_pred = full_tree.predict(x_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print("Training Accuracy: ", train_accuracy)
    print("\nTraining Set Classification Report:")
    print(classification_report(y_train, y_train_pred))

    # test set
    y_test_pred = full_tree.predict(x_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print("Test Accuracy: ", test_accuracy)
    print("\nTest Set Classification Report:")
    print(classification_report(y_test, y_test_pred))

    # Confusion Matrix for both sets
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    sns.heatmap(confusion_matrix(y_train, y_train_pred), annot=True, fmt="d", ax=axes[0], cmap="Blues")
    axes[0].set_title('Training Set Confusion Matrix')
    axes[0].set_xlabel('Predicted labels')
    axes[0].set_ylabel('True labels')

    sns.heatmap(confusion_matrix(y_test, y_test_pred), annot=True, fmt="d", ax=axes[1], cmap="Blues")
    axes[1].set_title('Test Set Confusion Matrix')
    axes[1].set_xlabel('Predicted labels')
    axes[1].set_ylabel('True labels')

    plt.show()
    print(f'Tree Depth: {full_tree.tree_.max_depth}')
    params = full_tree.get_params()
    for key, val in params.items():
        print(f"{key}:{val}")


def decision_tree_model(x_train, y_train, x_test, y_test, cv, output_path):

    dt_param_grid = {
        'max_depth': [None, 3, 5, 7],
        'min_samples_split': [2, 4, 5, 6, 8],
        'min_samples_leaf': [1, 3, 4, 5],
        'max_features': [None, 0.4, 0.8, 0.6],
        'splitter': ['best'],
        'criterion': ['gini'],
        'ccp_alpha': np.arange(0, 1, 0.1)
    }
    model_dt = DecisionTreeClassifier()

    time_start = time.time()
    model_dt, acc_dt, f1_dt, roc_auc_dt, param_dt, results = train_model_with_grid_search(model_dt, dt_param_grid,
                                                                                   x_train, y_train, x_test, y_test,
                                                                                    cv, output_path)
    time_taken_dt = time.time() - time_start
    plot_all_hyperparameter_effect(results, dt_param_grid)
    return model_dt


def show_feature_importance(columns, feature_importances):
    feature_importances_df = pd.DataFrame({
        'Feature': columns,
        'Importance': feature_importances
    })
    feature_importances_df = feature_importances_df.sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(12, 8))
    plt.barh(feature_importances_df['Feature'], feature_importances_df['Importance'], color='skyblue')
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Feature')
    plt.title('Feature Importances in Decision Tree')
    plt.gca().invert_yaxis()
    plt.subplots_adjust(left=0.25, right=0.95, top=0.95, bottom=0.1)
    plt.show()

    print(feature_importances_df)


def random_forest_model(x_train, y_train, x_test, y_test, cv, output_path):
    rf_param_grid = {
        'n_estimators': [50, 100, 200, 400],
        'criterion': ['gini'],
        'max_depth': [None,3, 5,10,20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4, 30],
        'bootstrap': [True, False],
        'max_features': [None, 0.4]
    }

    model_rf = RandomForestClassifier()

    time_start = time.time()
    model_rf, acc_rf, f1_rf, roc_auc_rf, param_rf, results = train_model_with_grid_search(model_rf, rf_param_grid,
                                                                                          x_train, y_train, x_test,
                                                                                          y_test, cv,
                                                                                          output_path)
    time_taken_rf = time.time() - time_start
    return model_rf


def plot_hyperparameter_effect(results: pd.DataFrame, param_name: str):
    mean_scores = results.pivot_table(index=param_name, values='mean_test_score')
    mean_scores.plot()
    plt.title(f'Effect of {param_name} on Accuracy')
    plt.xlabel(param_name)
    plt.ylabel('Mean Test Accuracy')
    plt.show()


def plot_all_hyperparameter_effect(results, param_dist):
    for param in param_dist.keys():
        plot_hyperparameter_effect(results, f'param_{param}')


def get_model_from_pkl_file(path: str):
    loaded_model = joblib.load(path)
    return loaded_model


def add_features_to_part_a(df):

    temp_norm = (df['Temp3pm'] - df['Temp3pm'].mean()) / df['Temp3pm'].std()
    pressure_norm = (df['Pressure3pm'] - df['Pressure3pm'].mean()) / df['Pressure3pm'].std()
    tpi = temp_norm + pressure_norm
    df['TempPressureIndex'] = tpi

    humidity_norm = (df['Humidity3pm'] - df['Humidity3pm'].mean()) / df['Humidity3pm'].std()
    wind_speed_norm = (df['WindGustSpeed'] - df['WindGustSpeed'].mean()) / df['WindGustSpeed'].std()

    wci = temp_norm - (0.55 - 0.55 * humidity_norm) * (temp_norm - 58) + wind_speed_norm
    df['WCI'] = wci

    heat_index = -42.379 + 2.04901523 * df['Temp3pm'] + 10.14333127 * df['Humidity3pm'] \
                 - 0.22475541 * (df['Temp3pm'] * df['Humidity3pm'] - 0.00683783 * df['Temp3pm'])**2 \
                 - 0.05481717 * df['Humidity3pm']**2 + 0.00122874 * df['Temp3pm']**2 * df['Humidity3pm'] \
                 + 0.00085282 * df['Temp3pm'] * df['Humidity3pm']**2 - 0.00000199 * df['Temp3pm']**2 * df['Humidity3pm']**2
    df['HeatIndex'] = heat_index

    return df


def handle_extreme_values(df: pd.DataFrame):

    def _explore_cloud9am_variable():
        print(f'Explore extreme value of Cloud9am: 999')
        cloud9am_999 = (df['Cloud9am'] == 999)
        df_cloud9am_999 = df[cloud9am_999]

        hobart_and_no_rain = ((df['Location'] == 'Hobart') & (df['RainToday'] == 'No'))
        df_hobart_and_no_rain = df[hobart_and_no_rain]

        hobart_and_no_rain_and_cloud3pm_is_5_6_7 = (hobart_and_no_rain & (df['Cloud3pm'].isin([5, 6, 7])))
        df_hobart_and_no_rain_and_cloud3pm_is_5_6_7 = df[hobart_and_no_rain_and_cloud3pm_is_5_6_7]

        hobart_and_no_rain_and_cloud3pm_and_9am_is_5_6_7 = (
                    hobart_and_no_rain & (df['Cloud3pm'].isin([5, 6, 7])) & (df['Cloud9am'].isin([5, 6, 7])))
        df_hobart_and_no_rain_and_cloud3pm_and_9am_is_5_6_7 = df[hobart_and_no_rain_and_cloud3pm_and_9am_is_5_6_7]

        df.loc[cloud9am_999, 'Cloud9am'] = 7


    def _explore_rainfall_variable():
        condition_rainfall_negative = (df['Rainfall'] < 0)
        df_rainfall_negative = df[condition_rainfall_negative]
        df.loc[condition_rainfall_negative, 'Rainfall'] = 0.0

    def _explore_winddir3pm_variable():
        wind_dir_3pm_zzz = (df['WindDir3pm'].astype(str).str.startswith("zzz"))
        # Set the 'zzz..' value to 'N'
        df.loc[wind_dir_3pm_zzz, 'WindDir3pm'] = 'N'

    # Sanity check that there is no data with contradiction.
    condition_rainfall_and_rain_today = ((df['Rainfall'] > 1.0) & (df['RainToday'] == 'No'))
    data_rainfall_and_rain_today = df[condition_rainfall_and_rain_today]


    #Deleting the empty sample  from the data
    empty_row_condition = (df['MaxTemp'].isna() & df['Rainfall'].isna() & df['Evaporation'].isna() &
                            df['Sunshine'].isna() & df['WindGustDir'].isna() & df['WindDir9am'].isna() &
                            df['WindDir3pm'].isna() & df['WindSpeed9am'].isna() & df['WindSpeed3pm'].isna() &
                            df['Humidity9am'].isna() & df['Humidity3pm'].isna() & df['Pressure9am'].isna() &
                            df['Pressure3pm'].isna() & df['Cloud9am'].isna() & df['Cloud3pm'].isna() & df['Temp9am'].isna() &
                            df['Temp3pm'].isna())

    df = df[~empty_row_condition]

    # Explore data of extreme values
    _explore_rainfall_variable()
    _explore_winddir3pm_variable()
    _explore_cloud9am_variable()

    return df


def re_handle_missing_values(df: pd.DataFrame):
    df['Evaporation'] = df.groupby('RainToday')['Evaporation'].transform(lambda x: x.fillna(x.median()))
    df['Sunshine'] = df.groupby('RainToday')['Sunshine'].transform(lambda x: x.fillna(x.mean()))

    return df


def categorized_features(df: pd.DataFrame):

    def categorized_column_by_equal_range(column: str, num_of_bins: int):
        # Categorize the non-NaN values into N categories of equal range

        new_column_name = f'{column}_Category'
        df[column], bins = pd.cut(df[column], bins=num_of_bins, labels=False, retbins=True)
        df[column] += 1
        df[column].astype(str)


    def categorized_column_by_equal_bins(column: str, num_of_bins: int):
        # Categorize the non-NaN values into N categories with equal number of data points


        new_column_name = f'{column}_Category'
        df[new_column_name], bins = pd.qcut(df[column], q=num_of_bins, labels=False, retbins=True)
        df[new_column_name] += 1


    def categorize_temp(temp):
        if temp < 7.5:
            return 'cold'
        elif 7.5 <= temp < 30:
            return 'mild'
        else:
            return 'hot'


    def categorize_rainfall(rainfall):
        if rainfall == 0:
            return 'none'
        elif 0 < rainfall <= 20:
            return 'light'
        elif 20 < rainfall <= 60:
            return 'moderate'
        else:
            return 'heavy'


    def categorize_evaporation(evaporation):
        if evaporation == 0:
            return 'dry'
        elif 0 < evaporation <= 2:
            return 'minimal'
        elif 2 < evaporation <= 6:
            return 'average'
        else:
            return 'significant'


    def categorize_sunshine(sunshine):
        if sunshine == 0:
            return 'no_sun'
        elif 0 < sunshine <= 3:
            return 'little_sun'
        elif 4 < sunshine <= 7:
            return 'some_sun'
        else:
            return 'a_lot_of_sun'


    def categorize_wind_speed(speed):
        if speed < 20:
            return 'calm'
        elif 20 <= speed < 40:
            return 'moderate_breeze'
        elif 40 <= speed < 61:
            return 'strong_breeze'
        else:
            return 'gale'


    def categorize_humidity(humidity):
        if humidity < 25:
            return 'dry'
        elif 25 <= humidity < 60:
            return 'comfortable'
        else:
            return 'very_humid'


    def categorize_pressure(pressure):
        if pressure < 1000:
            return 'low'
        elif 1000 <= pressure < 1030:
            return 'normal'
        else:
            return 'high'


    def categorize_cloud(cloud):
        if cloud == 0:
            return 'clear'
        elif 1 <= cloud <= 4:
            return 'partly_cloudy'
        elif 5 <= cloud <= 7:
            return 'mostly_cloudy'
        else:
            return 'overcast'


    categorized_column_by_equal_bins(column='MaxTemp', num_of_bins=5)
    categorized_column_by_equal_bins(column='MinTemp', num_of_bins=5)
    categorized_column_by_equal_range(column='WindGustSpeed', num_of_bins=5)
    df['Temp9am_Category'] = df['Temp9am'].apply(categorize_temp)
    df['Temp3pm_Category'] = df['Temp3pm'].apply(categorize_temp)
    df['Rainfall_Category'] = df['Rainfall'].apply(categorize_rainfall)
    df['Evaporation_Category'] = df['Evaporation'].apply(categorize_evaporation)
    df['Sunshine_Category'] = df['Sunshine'].apply(categorize_sunshine)
    df['WindSpeed9am_Category'] = df['WindSpeed9am'].apply(categorize_wind_speed)
    df['WindSpeed3pm_Category'] = df['WindSpeed3pm'].apply(categorize_wind_speed)
    df['Humidity9am_Category'] = df['Humidity9am'].apply(categorize_humidity)
    df['Humidity3pm_Category'] = df['Humidity3pm'].apply(categorize_humidity)
    df['Pressure9am_Category'] = df['Pressure9am'].apply(categorize_pressure)
    df['Pressure3pm_Category'] = df['Pressure3pm'].apply(categorize_pressure)
    df['Cloud9am_Category'] = df['Cloud9am'].apply(categorize_cloud)
    df['Cloud3pm_Category'] = df['Cloud3pm'].apply(categorize_cloud)
    df.loc[df['Location'] == "SydneyAirport", "Location"] = "Sydney"

    return df


def create_new_features_combinations(df: pd.DataFrame):

    def _are_opposite_direction(direction_a, direction_b):
        opposite_dir_map = {"N":"S", "S":"N", "E":"W","W":"E"}
        return opposite_dir_map.get(direction_a) == direction_b

    df['TempRange'] = df['MaxTemp'] - df['MinTemp']
    df['TempChange'] = df['Temp3pm'] - df['Temp9am']
    df['HumidityChange'] = df['Humidity3pm'] - df['Humidity9am']
    df['AvgWindSpeed'] = (df['WindSpeed9am'] + df['WindSpeed3pm']) / 2
    df['WindGustRatio'] = df['WindGustSpeed'] / df['AvgWindSpeed']
    df['PressureChange'] = df['Pressure3pm'] - df['Pressure9am']
    df['SameWindDir'] = (df['WindDir9am'] == df['WindDir3pm'])
    df['OppositeWindDirDiff'] = df.apply(lambda row: _are_opposite_direction(row['WindDir9am'], row['WindDir3pm']), axis=1)
    df['RainToday'] = df['RainToday'].map({"Yes":1,"No":0})
    df['RainTomorrow'] = df['RainTomorrow'].map({"Yes":1,"No":0})

    return df


def fix_part_a(df: pd.DataFrame):
    df = handle_extreme_values(df)
    df = re_handle_missing_values(df) # New in part b (missing values completion by RainToday)
    df = categorized_features(df)
    df = create_new_features_combinations(df)

    return df


def feature_selection(df: pd.DataFrame):
    def _forwardSelection(df, response, significance_level=0.05, r_squared_threshold=0.001):
        selected = []
        current_score, best_new_score = 0.0, 0.0
        iteration = 0

        while True:
            remaining = set(df.columns) - set(selected) - {response}
            best_candidate = None

            scores_with_candidates = []
            for candidate in remaining:
                formula = "{} ~ {} ".format(response, ' + '.join(selected + [candidate]))
                try:
                    model = smf.ols(formula, df).fit()
                    score = model.rsquared_adj
                    if candidate in list(model.pvalues._info_axis.values):
                        p_value = model.pvalues[candidate]
                    elif f'{candidate}[T.True]' in list(model.pvalues._info_axis.values):
                        p_value = model.pvalues[f'{candidate}[T.True]']
                    else:
                        p_value = model.pvalues[f'{candidate}[F.False]']
                    if p_value < significance_level:
                        scores_with_candidates.append((score, candidate))
                except Exception as e:
                    print(f"Skipping {candidate} due to error: {e}")

            if not scores_with_candidates:
                break

            scores_with_candidates.sort()
            best_new_score, best_candidate = scores_with_candidates.pop()

            improvement = best_new_score - current_score
            # Debugging output to trace the loop execution and variable selection
            print(
                f"Iteration {iteration}: Best candidate: {best_candidate}, Adjusted R-squared: {best_new_score:.4f}, Improvement: {improvement:.4f}")

            if improvement > 0 and improvement > r_squared_threshold:
                selected.append(best_candidate)
                current_score = best_new_score
                print(f"Selected: {best_candidate} | Adjusted R-squared: {best_new_score:.4f}")
            elif current_score >= r_squared_threshold:
                print(f"Stopping iterations. Final model reached adjusted R-squared threshold: {current_score:.4f}")
                break
            else:
                print(f'Not Selected: {best_candidate}..')
                break

            iteration += 1

        formula = "{} ~ {} ".format(response, ' + '.join(selected))
        model = smf.ols(formula, df).fit()
        print(f"Final model formula: {formula}")
        return model, selected

    # Example usage
    # final_model = _forwardSelection(df, 'RainTomorrow')
    print(df.columns)
    lm , selected = _forwardSelection(df, 'RainTomorrow')
    lm.summary()
    print(lm.model.formula)
    print(lm.rsquared_adj)

    return selected


def prepare_data(df:pd.DataFrame):
    df = fix_part_a(df)
    df = pd.get_dummies(df)
    # selected_features = feature_selection(df) + ["RainTomorrow"]
    selected_features = ["Sunshine", "Sunshine_Category_some_sun", "Pressure3pm", "Pressure9am",
                         "Humidity3pm", "WindGustSpeed", "Evaporation_Category_minimal",
                         "Cloud9am_Category_mostly_cloudy", "Cloud3pm_Category_overcast", "Cloud9am",
                         "RainToday", "Rainfall", "TempRange", "RainTomorrow"]
    return df[selected_features]


def main(path_to_file: str):

    df = read_data(r"C:\Users\Ron Gabo\Desktop\ml_course\data_test_v1.csv")
    df2 = read_data(r"C:\Users\Ron Gabo\Desktop\ml_course\data_train_v1.csv")
    df = pd.concat([df,df2])
    df_train, df_test = split_data_train_test(df, 11)
    print(f'Test set samples: {len(df_test)}')
    print(f'Train set samples: {len(df_train)}')
    x_train = df_train.drop(columns=["RainTomorrow"])
    x_test = df_test.drop(columns=["RainTomorrow"])
    y_train = df_train['RainTomorrow']
    y_test = df_test['RainTomorrow']
    x_train, x_test, cv = feature_scaling(x_train, x_test)
    dt_output_path = os.path.join(os.path.dirname(path_to_file),'dt_best_model.pkl')
    rf_output_path = os.path.join(os.path.dirname(path_to_file),'rf_best_model.pkl')
    full_decision_tree_model(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
    dt = decision_tree_model(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test, cv=cv, output_path=dt_output_path)
    show_feature_importance(columns=x_train.columns, feature_importances=dt.feature_importances_)
    rf = random_forest_model(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test, cv=cv,
                             output_path=rf_output_path)


if __name__ == '__main__':
    path_to_file: str = r'C:\Users\Ron Gabo\Desktop\ml_course\Xy_train.csv'
    main(path_to_file=path_to_file)

#Finish Decision tree part




# Start NN part

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
from imblearn.over_sampling import SMOTE,ADASYN,BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler
import random
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
    splitType = 'cache'# original split or from cache (group split)
    RonData = False # external pre process data - loaded from external file
    EnableSMOTE = False # enable SMOT on data for both single test or grid search
    singleTest = True
    GridSearch = False
    RUN_gridTest = False # if false - load results from file else run grid search and saves the results
    ## Data loading:
    path_to_file = r'E:\t2\machine\project\part_1\pythonProject\Xy_train.csv'


    if PrePrepareData:


        # Data analysis in case it is not yet prepared
        df = pd.read_csv(path_to_file, encoding='ISO-8859-1')
        categorials = ['Location', 'WindGustDir', 'WindDir3pm', 'WindDir9am',
                       'Cloud9am', 'Cloud3pm', 'CloudsinJakarta', 'RainToday', 'RainTomorrow']
        df = handle_manual(df)
        df = handle_NaNs(df,categorials)

    if PrepareData:
        ## Data Preperation part:
        NormalizationMethod = 'MinMaxScaler'# MinMaxScalar or StandardScaler
        # Create dummy variables
        df = pd.get_dummies(df,columns =['WindGustDir','WindDir9am','WindDir3pm','Location'])

        # Create X and Y vectors:
        X = df.drop('RainTomorrow', axis=1).values
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
            random.seed() # Generates random test train split
            X_train, X_test, Y_train, Y_test = train_test_split(X_normalized, Y_normalized, test_size=0.1, random_state=23)#random.randint(1,10000)
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
        X = full_df.drop('RainTomorrow',axis =  1).values
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
        model = MLPClassifier(random_state=1, verbose=True)
        # max_iter=500,learning_rate_init=0.001,hidden_layer_sizes=(150),
        if EnableSMOTE:
            smote = SMOTE()
            X_train_smote, y_train_smote = smote.fit_resample(X_train, Y_train)
            print("SMOTE ACTIVATED")
            model.fit(X_train_smote, y_train_smote)
        else:
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
        print(f"Train  precision: {precision}")
        print(f"Train recall Score: {recall}")

        conf_matrix = confusion_matrix(Y_train, y_train_pred)

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
            'hidden_layer_sizes': [100,200,(100,100),(36,18,3),(36,18)],
            'max_iter':[200,500],
            'activation': ['relu','tanh']
        }
        # Set model
        model = MLPClassifier(random_state=1, verbose=False)
        kf = KFold(n_splits=5, shuffle=True, random_state=1)
        # Define scoring metrics
        print("start time: " + str(time.ctime()))

        # # Define grid search

        grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                                   cv=kf, scoring='accuracy', verbose=2)
        if RUN_gridTest:
            if EnableSMOTE:
                smote = SMOTE()
                X_train_smote, y_train_smote = smote.fit_resample(X_train, Y_train)
                grid_search.fit(X_train_smote, y_train_smote)
            else:
                grid_search.fit(X_train, Y_train)
            results = pd.DataFrame(grid_search.cv_results_)
            results.to_csv('grid_search11.csv', index=False)
        else:
            results = pd.read_csv('grid_search11.csv')
        # Filter the relevant columns
        results = results[['param_hidden_layer_sizes', 'param_activation', 'param_learning_rate','param_max_iter','mean_test_score']]
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

# End NN part

# Start Kmeans - Unsupervised learning
import pandas as pd
import numpy as np
from part_2_nadav import handle_manual,handle_NaNs
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score,precision_score,recall_score, confusion_matrix
import seaborn as sns
import itertools


if __name__ == '__main__':
    PrePrepareData = True
    PrepareData = True
    DO_PCA = False
    DO_Kmeans = True
    DO_Kmeans_performanceAnalysis = False
    DO_Kmeans_performanceAnalysis_manyClusters = False
    showWinner = True
    winner_transform_fromcode = True
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
        X = df.drop('RainTomorrow', axis=1)
        Y = df['RainTomorrow']

        # Normalize Data:
        if NormalizationMethod == 'MinMaxScaler':
            minmax_scaler = MinMaxScaler()
            X_normalized = minmax_scaler.fit_transform(X)
            Y_normalized = Y
        elif NormalizationMethod == 'StandardScaler':
            standard_scaler = StandardScaler()
            X_normalized = standard_scaler.fit_transform(X)
            Y_normalized = Y


    # First Part - Do Principal Components Analysis on our Dataset.
    # Present a graph of the principal components, which correspond to the first 2 eigenvectors of the data matrix.

    if DO_PCA:
        pca = PCA(n_components=2)
        pca.fit(X)
        print(pca.explained_variance_ratio_)
        print(pca.explained_variance_ratio_.sum())
        df_pca = pca.transform(X)
        df_pca = pd.DataFrame(df_pca, columns=['PC1', 'PC2'])
        df_pca['RainTommorow'] = Y
        # Transform the data and create a DataFrame
        df_pca = pca.transform(X)
        df_pca = pd.DataFrame(df_pca, columns=['PC1', 'PC2'])
        df_pca['RainTomorrow'] = Y

        # Plot the PCA results with different colors for RainTomorrow categories
        colors = {0: 'blue', 1: 'red'}
        plt.figure(figsize=(10, 8))

        for category in [0,1]:
            subset = df_pca[df_pca['RainTomorrow'] == category]
            plt.scatter(subset['PC1'], subset['PC2'], c=colors[category], label=f'RainTomorrow = {category}', alpha=0.6)

        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('PCA of Dataset with RainTomorrow Categories')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # Second Part - Do Principal Components Analysis on our Dataset.

    if DO_Kmeans:
        # Perform KMeans clustering
        n_classes = 8
        kmeans = KMeans(n_clusters=n_classes, max_iter=300, n_init=10, random_state=42)
        kmeans.fit(X)
        df['cluster'] = kmeans.predict(X)

        # Perform PCA for visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
        df_pca['cluster'] = df['cluster']
        df_pca['label'] = Y.values

        colors = {0: 'blue', 1: 'green', 2:'red',3:'purple',4:'yellow',5:'brown',6:'black', 7:'magenta',8:'lightblue',9:'ocean'}
        markers = {0: 'o', 1: '*'}  # Circle for label 0, star for label 1

        # Plot the PCA results with clusters using Matplotlib
        plt.figure(figsize=(10, 8))
        for cluster in range(n_classes):
            subset = df_pca[df_pca['cluster'] == cluster]
            for label in range(n_classes):
                label_subset = subset[subset['label'] == label]
                plt.scatter(label_subset['PC1'], label_subset['PC2'], c=colors[cluster],
                            marker='o', label=f'Cluster {cluster}, RainTomorrow - {str(bool(label))}', alpha=0.6)

        # Plot the cluster centers
        centers_pca = pca.transform(kmeans.cluster_centers_)
        plt.scatter(centers_pca[:, 0], centers_pca[:, 1], marker='+', s=100, color='red', label='Centroids')

        # Customize the plot
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('KMeans Clustering with PCA')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()


        colors = {1: 'blue', 0: 'green'}
        # Create a new figure for the real labels plot
        plt.figure(figsize=(10, 8))
        # Scatter plot for each real label
        for label in [0, 1]:
            label_subset = df_pca[df_pca['label'] == label]
            plt.scatter(label_subset['PC1'], label_subset['PC2'], c=colors[label], marker=markers[label],
                        label=f'RainTomorrow - {str(bool(label))}', alpha=0.6)

        # Customize the plot
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('Real Labels with PCA')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Show both plots
        plt.show()

    if DO_Kmeans_performanceAnalysis:

        original_labels = Y.values
        predicted_labels = df['cluster']


        predicted_labels_fixed_labels = predicted_labels.replace({0:1,1:0})
        # Adjust predicted labels to match original labels
        conf_matrix = confusion_matrix(original_labels, predicted_labels_fixed_labels)

        f1 = f1_score(original_labels, predicted_labels_fixed_labels)
        accuracy = accuracy_score(original_labels, predicted_labels_fixed_labels)
        precision = precision_score(original_labels, predicted_labels_fixed_labels)
        recall = recall_score(original_labels, predicted_labels_fixed_labels)

        print(f"F1 Score: {f1}")
        print(f"Accuracy Score: {accuracy}")
        print(f"precision: {precision}")
        print(f"recall Score: {recall}")

        plt.figure(figsize=(10, 7))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 20})
        plt.title('Confusion Matrix Heatmap', fontsize=24)
        plt.xlabel('Predicted', fontsize=24)
        plt.ylabel('Actual', fontsize=24)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.show()

if DO_Kmeans_performanceAnalysis_manyClusters:


        original_labels = Y.values
        predicted_labels = df['cluster']

        # Define the range of original labels (e.g., 0 to 3)
        original_labels_range = range(n_classes)  # Labels are 0, 1, 2, 3

        # Generate all possible mappings using itertools.product
        all_mappings = list(itertools.product([0, 1], repeat=len(original_labels_range)))
        all_label_mappings = [{original: new for original, new in zip(original_labels_range, mapping)} for mapping in
                              all_mappings]
        max_accuracy = 0
        winner_transform = []
        for i, mapping in enumerate(all_label_mappings):
            # Apply the mapping using the replace method
            transformed_series = predicted_labels.replace(mapping)
            print(f"Mapping {i}: {mapping}, Transformed Series: {transformed_series.tolist()}")

            conf_matrix = confusion_matrix(original_labels, transformed_series)

            f1 = f1_score(original_labels, transformed_series)
            accuracy = accuracy_score(original_labels, transformed_series)
            precision = precision_score(original_labels, transformed_series)
            recall = recall_score(original_labels, transformed_series)
            if accuracy>max_accuracy:
                max_accuracy = accuracy
                winner_transform = mapping

        # Winner
        transformed_series = predicted_labels.replace(winner_transform)
        conf_matrix = confusion_matrix(original_labels, transformed_series)
        f1 = f1_score(original_labels, transformed_series)
        accuracy = accuracy_score(original_labels, transformed_series)
        precision = precision_score(original_labels, transformed_series)
        recall = recall_score(original_labels, transformed_series)
        print(winner_transform)
        print(f"winner F1 Score: {f1}")
        print(f"winner Accuracy Score: {accuracy}")
        print(f"winner precision: {precision}")
        print(f"winner recall Score: {recall}")

        plt.figure(figsize=(10, 7))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 20})
        plt.title('Confusion Matrix Heatmap', fontsize=24)
        plt.xlabel('Predicted', fontsize=24)
        plt.ylabel('Actual', fontsize=24)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.show()

if showWinner:
    if winner_transform_fromcode:
        winner_transform = {0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 0, 6: 0, 7: 1}

    df['cluster'] = kmeans.predict(X)

    # Perform PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    df_pca['cluster'] = df['cluster']
    df_pca['cluster'] = df_pca['cluster'].map(winner_transform)

    df_pca['label'] = Y.values

    colors = {1: 'blue', 0: 'green', 2: 'red', 3: 'purple', 4: 'yellow', 5: 'brown', 6: 'black', 7: 'magenta',
              8: 'lightblue', 9: 'ocean'}
    markers = {0: 'o', 1: '*'}  # Circle for label 0, star for label 1

    # Plot the PCA results with clusters using Matplotlib
    plt.figure(figsize=(10, 8))
    for cluster in range(2):
        print(cluster)
        label_subset = df_pca[df_pca['label'] == cluster]
        plt.scatter(label_subset['PC1'], label_subset['PC2'], c=colors[cluster],
                        marker='o', label='RainTomorrow True prediction' if cluster ==1 else 'RainTomorrow False prediction', alpha=0.6)
    # Customize the plot
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('KMeans Clustering with PCA')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.show()
# End Kmeans - Unsupervised learning

# Start svm part:
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

#End SVM part