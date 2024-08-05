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
