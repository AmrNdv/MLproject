import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import pearsonr, skew, normaltest, chi2_contingency, f_oneway
import statsmodels.api as sm
from statsmodels.formula.api import ols
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


### Statistical tests functions ###

# Single variable

def check_normal_distribution(df: pd.DataFrame, x_value: str, p_value_threshold: float = 0.05):
    data = df[x_value].dropna(inplace=False)
    stat, p = normaltest(data)
    if p <= p_value_threshold:
        print(f'Normal distribution test: {x_value} is distributed normally, p-value: {round(p,2)}')
    else:
        print(f'Normal distribution test: {x_value} is NOT distributed normally, p-value: {round(p,2)}')


def check_skewness_distribution(df: pd.DataFrame, x_value: str):
    data = df[x_value].dropna(inplace=False)
    skewness = skew(data)
    if -0.5 <= skewness <= 0.5:
        print(f'Skewness distribution test: {x_value} data is approximately symmetric, skewness: {round(skewness,2)}')
    elif 0.5 < skewness <= 1:
        print(f'Skewness distribution test: {x_value} data is moderately right skewed, skewness: {round(skewness,2)}')
    elif skewness > 1:
        print(f'Skewness distribution test: {x_value} data is highly right skewed, skewness: {round(skewness,2)}')
    elif -0.5 >= skewness > -1:
        print(f'Skewness distribution test: {x_value} data is moderately left skewed, skewness: {round(skewness,2)}')
    elif skewness < -1:
        print(f'Skewness distribution test: {x_value} data is highly left skewed, skewness: {round(skewness,2)}')


# Multiple variables

def check_chi2_square(df:pd.DataFrame, x_value: str, y_value: str, p_value_threshold: float = 0.05):
    crosstab = pd.crosstab(df[x_value], df[y_value])
    chi2, p, dof, expected = chi2_contingency(crosstab)
    if p <= p_value_threshold:
        print(f'Chi square test: The correlation between {x_value}, {y_value} are statistically significant, p-value: {round(p,2)}')
    else:
        print(f'Chi square test: The correlation between {x_value}, {y_value} are NOT statistically significant, p-value: {round(p,2)}')


def calculate_pearson_coefficient(df: pd.DataFrame, x_value: str, y_value: str, p_value_threshold: float = 0.05):
    data = df.dropna(subset=x_value, inplace=False)
    data = data.dropna(subset=y_value, inplace=False)
    correlation, p_value = pearsonr(data[x_value], data[y_value])
    if p_value <= p_value_threshold:
        print(f'Pearson test: The correlation between {x_value}, {y_value} are statistically significant, p-values: {round(p_value,2)}')
    else:
        print(f'Pearson test: The correlation between {x_value}, {y_value} are NOT statistically significant, p-values: {round(p_value,2)}')


### plots functions ###

# Single Variable

def create_box_plot(df: pd.DataFrame, y_label: str, show: bool = True):
    # Box plots for Continuous variables
    if not show:
        return

    boxPlot = plt.axes()
    sns.boxplot(y=y_label, data=df)
    boxPlot.set_title(f'{y_label} Boxplot')
    plt.show()


def create_histogram(df: pd.DataFrame, y_label: str, bins: int = 10, show: bool = True):
    # Histogram - Continuous variables
    if not show:
        return

    plt.hist(df[y_label], bins=bins, density=True)
    plt.title(f"{y_label} histogram", fontsize=20)
    plt.ylabel('frequency', fontsize=15)
    sns.histplot(df[y_label], bins=20, kde=True, fill=True, color='blue', line_kws={'color': 'red'})
    plt.xlabel(f'Amount of {y_label}', fontsize=15)
    plt.show()


def create_count_plot(df: pd.DataFrame, y_label: str, rng: range = range(0,6), show: bool = True):
    # Categorical variables
    if not show:
        return

    sns.countplot(x=y_label, data=df, palette='Set1')
    plt.title(f"Countplot of {y_label}", fontsize=20)
    plt.xticks(rng)
    plt.xlabel(y_label, fontsize=15)
    plt.ylabel('Frequency', fontsize=15)
    plt.show()

# Multiple variables

def create_strip_plot(df: pd.DataFrame, x_value: str, y_value: str, show: bool = True):
    # Categorical - Continuous variables
    if not show:
        return

    sns.stripplot(x=x_value, y=y_value, data=df, jitter=True)
    plt.title(f'Strip Plot of {x_value} by {y_value}')
    plt.xlabel(x_value)
    plt.ylabel(y_value)
    plt.show()


def plot_heatmap(df: pd.DataFrame, variables_dict: dict, show: bool = True):
    # Continuous variables
    if not show:
        return

    heatMap = plt.axes()
    sns.heatmap(df[variables_dict['continuous']].corr(), annot=True, cmap='coolwarm')
    heatMap.set_title('Variables correlations')
    plt.xticks(rotation=45, ha='right')
    plt.show()


def plot_scatter(df: pd.DataFrame, x_value: str, y_value: str, show: bool = True):
    # Continuous-Continuous variables
    if not show:
        return

    plt.figure(figsize=(8, 6))
    plt.scatter(df[x_value], df[y_value], color='blue', alpha=0.7)
    plt.title(f'Scatter Plot of Columns {x_value} and {y_value}')
    plt.xlabel(f'{x_value}')
    plt.ylabel(f'{y_value}')
    plt.grid(True)
    plt.show()


def heat_map_categorical(df: pd.DataFrame, x_value: str, y_value: str, show: bool = True):
    # Categorical - Categorical variables
    if not show:
        return

    cross_tab = pd.crosstab(df[x_value], df[y_value])
    sns.heatmap(cross_tab, annot=True, cmap='YlGnBu', fmt='d')
    plt.xlabel(f'{x_value}')
    plt.ylabel(f'{y_value}')
    plt.title(f'Relationship between {x_value} and {y_value}')
    plt.show()


def plot_stacked_bar(df:pd.DataFrame, x_label: str, y_label: str, show: bool = True):
    # Categorical-Categorical Variables
    if not show:
        return

    cross_tab = pd.crosstab(df[x_label], df[y_label])
    cross_tab.plot(kind='bar')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f'Relationship between {x_label} and {y_label}')
    plt.legend(title=f'{y_label}')
    plt.show()


# General Functions

def plot_graphs(df: pd.DataFrame, variables_dict: dict, show: bool = True):
    for key, variables in variables_dict.items():
        cat_pairs = set()
        for variable in variables:
            if key == 'continuous':
                create_box_plot(df=df, y_label=variable, show=show)
                create_histogram(df=df, y_label=variable, show=show)
                print(f'{variable}')
            else:
                create_count_plot(df=df, y_label=variable,rng=range(0,10),show=show)
                for var2 in variables_dict['categorical']:
                    if (variable, var2) not in cat_pairs:
                        cat_pairs.add((variable,var2))
                        cat_pairs.add((var2,variable))
                        heat_map_categorical(df, variable, var2, show)


def show_graphs_and_statistical_test_for_specific_variables(df: pd.DataFrame, variables_correlation_focus: dict, show: bool = True):
    #After first analyse of graphs - sending here specific variables and connection to represent correlations and graphs.

    for dtype, list_of_pairs in variables_correlation_focus.items():
        for pair in list_of_pairs:
            if dtype == "categorical-categorical":
                plot_stacked_bar(df,pair[0], pair[1], show)
                heat_map_categorical(df, pair[0], pair[1], show)
                check_chi2_square(df=df, x_value=pair[0], y_value=pair[1])
            elif dtype == "continuous-continuous":
                plot_scatter(df, pair[0], pair[1], show)
                calculate_pearson_coefficient(df,pair[0], pair[1])
            else: # categorical-continuous
                create_strip_plot(df, pair[0], pair[1], show)


def describe_variables(df: pd.DataFrame, variables_dict:dict):
    for key, variables in variables_dict.items():
        for variable in variables:
            print(f'\n--{variable}--')
            print(df[variable].describe())
            print(f'Missing values: {df[variable].isna().sum()}')
            if key in ['categorical', 'target']:
                print(f'Unique Values: {df[variable].unique()}')
            else:
                check_normal_distribution(df=df, x_value=variable)
                check_skewness_distribution(df=df, x_value=variable)


def data_exploratory(df: pd.DataFrame):
    all_variables = {
        'target': ["RainTomorrow"],
        'continuous': ["MinTemp", "MaxTemp", "Rainfall", "Evaporation", "Sunshine", "WindGustSpeed", "WindSpeed9am", "WindSpeed3pm", "Humidity9am", "Humidity3pm", "Pressure9am","Pressure3pm","Temp9am","Temp3pm"],
        'categorical': ["RainToday", "Location", "WindGustDir", "WindDir9am", "WindDir3pm", "Cloud9am", "Cloud3pm", "CloudsinJakarta"]
    }

    plot_graphs(df=df, variables_dict=all_variables, show=False)
    describe_variables(df=df, variables_dict=all_variables)
    plot_heatmap(df=df, variables_dict=all_variables, show=False)

    variables_correlation_focus = {"categorical-categorical":[],
                                    "continuous-continuous": [("Temp9am","MinTemp"),("Temp3pm","MaxTemp"), ("WindSpeed3pm","WindGustSpeed")],
                                   "categorical-continuous": [("Sunshine","Cloud3pm")]}
    show_graphs_and_statistical_test_for_specific_variables(df=df,
                                                            variables_correlation_focus=variables_correlation_focus,
                                                            show=False)


def handle_missing_and_extreme_values(df: pd.DataFrame):


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

        print(f'-Cloud9am Extreme Values-')
        print(f'Cloud9am Extreme values count: {len(df_cloud9am_999)}')
        print(f'Rows:\n', df_cloud9am_999)
        print(f'Further Analysis:\n'
              f'a) Data from Location: Hobart, Cloud3pm: 6.0, Rainfall/RainToday = No')
        print(f'b) Total data of Hobart and No rain:', len(df_hobart_and_no_rain))
        print(f'c) Total data of Hobart and No rain and Cloud3pm is between 5-7:',
              len(df_hobart_and_no_rain_and_cloud3pm_is_5_6_7))
        print(f'd) Total data of Hobart and No Rain and Cloud9am and Cloud3pm is between 5-7:',
              len(df_hobart_and_no_rain_and_cloud3pm_and_9am_is_5_6_7))


        create_count_plot(df=df_hobart_and_no_rain_and_cloud3pm_is_5_6_7, y_label='Cloud9am', show=False)
        print(f"e) Median value of Cloud9am in Hobart, No rain, Cloud3pm between 5-7: {df_hobart_and_no_rain_and_cloud3pm_is_5_6_7['Cloud9am'].median()}")
        print(f'Decision: ' 
              f'Based on plot and Median value of Cloud9am we are setting value of Cloud9am as 7')
        df.loc[cloud9am_999, 'Cloud9am'] = 7


    def _explore_rainfall_variable():
        condition_rainfall_negative = (df['Rainfall'] < 0)
        df_rainfall_negative = df[condition_rainfall_negative]
        df.loc[condition_rainfall_negative, 'Rainfall'] = 0.0

        print(f'-Rainfall Extreme Values-')
        print(f'Rainfall negative values: {len(df_rainfall_negative)}')
        print(f'Rows:\n', df_rainfall_negative)
        print(f'Decision: Modify Rainfall -3.0 value to 0.0.\n'
              f'Reason: RainToday variable value is 0.0,\n'
              f'Which means Rainfall values is between 0-1, assuming this 0 because of majority of 0s and simplifications')


    def _explore_winddir3pm_variable():
        wind_dir_3pm_zzz = (df['WindDir3pm'].astype(str).str.startswith("zzz"))
        wind_dir_3pm_zzz_row = df[wind_dir_3pm_zzz]
        canberra_and_no_rain = ((df['Location'] == 'Canberra') & (df['RainToday'] == 'No'))
        consistence_wind_dir = ((df['WindDir9am'] == df['WindDir3pm']))
        df_consistence_wind_dir_in_canberra_and_no_rain = df[canberra_and_no_rain & consistence_wind_dir]
        df_in_canberra_and_no_rain_and_wind_dir_9am_is_North = df[(canberra_and_no_rain) & (df['WindDir9am'] == 'N')]
        canberra_and_no_rain_and_wind_dir_9am_equals_3pm_and_is_N = (canberra_and_no_rain & consistence_wind_dir & (df['WindDir9am'] == 'N'))
        df_consistence_wind_dir_in_canberra_and_North_wind_dir_9am = df[canberra_and_no_rain & consistence_wind_dir & (df['WindDir9am'] == 'N')]

        print(f'-WindDir3pm Extreme Values-\n'
              f'Rows:\n'
              f'{wind_dir_3pm_zzz_row}')
        print(f'Further Analysis:\n'
              f'   a) This sample features are: Canberra, No Rain, and WindDir9am is N (North)\n'
              f'   b) All samples size from Canberra: {len(df[df["Location"] == "Canberra"])}\n'
              f'   c) All samples size from Canberra and with No Rain: { len(df[canberra_and_no_rain])}\n'
              f'   e) Samples size of consistency when WinDir9am equals to WindDir3pm in Canberra and No Rain:'
              f'{len(df_consistence_wind_dir_in_canberra_and_no_rain)}'
              f'   f) Ratio of e/c:'
              f'{round(len(df_consistence_wind_dir_in_canberra_and_no_rain)/len(df[canberra_and_no_rain]),2)}\n'
              f'   g) Samples size of samples from Canberra, No Rain and WindDir9am is N (North) {len(df_in_canberra_and_no_rain_and_wind_dir_9am_is_North)}'
              f'   g) Consistency when WinDir9am equals to WindDir3pm in Canberra, No Rain and WinDir9am is N (North): {len(canberra_and_no_rain_and_wind_dir_9am_equals_3pm_and_is_N)}'
              f'   g) Ratio of g/e: {round(len(df_consistence_wind_dir_in_canberra_and_North_wind_dir_9am) / len(df_in_canberra_and_no_rain_and_wind_dir_9am_is_North), 2)}')
        print(f'Decision: Set WinDir3pm value to N (North)\n'
              f'Reason: Seems to have high probability when grouped those values above so.\n'
              f'North is most common among this data type.')

        create_count_plot(df=df_in_canberra_and_no_rain_and_wind_dir_9am_is_North, y_label="WindDir3pm", show=True)
        # By this plot we can see that most of samples in canberra, no rain and windDir9am is N - WindDir3pm value is N as well. So we can assume to fill in this value with 'N'.

        # Set the 'zzz..' value to 'N'
        df.loc[wind_dir_3pm_zzz, 'WindDir3pm'] = 'N'


    # Variables with no changes to make

    #RainToday - no missing values, values make sense to reality
    #MinTemp - no missing values, values make sense to reality
    #RainToday - no missing value,  values make sense to reality
    #Location - no missing value,  values make sense to reality
    #CloudsinJakarta - no missing values, values make sense to reality

    # Variables with missing values alone

    #Evaporation - 1239 missing values, values make sense to reality
    #Sunshine - 1850 missing values, values make sense to reality

    # Single missing values
    #MaxTemp - 1 single missing value, values make sense to reality
    #WindGustSpeed - 1 missing value, values make sense to reality
    #WindSpeed9am - 1 missing value,  values make sense to reality
    #WindSpeed3pm - 1 missing value,  values make sense to reality
    #Humidity9am - 1 missing value,  values make sense to reality
    #Humidity3pm - 1 missing value,  values make sense to reality
    #Pressure9am - 1 missing value,  values make sense to reality
    #Pressure3pm- 1 missing value,  values make sense to reality
    #Temp9am - 1 missing value,  values make sense to reality
    #Temp3pm - 1 missing value,  values make sense to reality
    #WindGustDir - 1 missing value,  values make sense to reality
    #WindDir9am - 1 missing value,  values make sense to reality
    #Cloud3pm - 1 missing value, values make sense to reality

    # Variables with missing values and extreme values

    #WindDir3pm - 1 missing value,  value of zzz make no sense
    #Rainfall - 1 single missing value, -3 value of mm rain no make sense
    #Cloud9am - 1 missing value, value of 999 make no sense

    # First let's see if all those with 1 single missing values are same sample.
    # Seems that all those records are from same sample. All data is from Uluru, which he has only 178 records.
    # While the second-smallest location count is 1195 (Canberra).
    # Therefore, The influence of this sample for Uluru might be big.
    # All this with trying filling 18 values for variables based on small category.


    print(f'--- Handling Missing Values and Extreme Values ---')

    print(f'Sanity Check:')
    # Sanity check that there is no data with contradiction.
    condition_rainfall_and_rain_today = ((df['Rainfall'] > 1.0) & (df['RainToday'] == 'No'))
    data_rainfall_and_rain_today = df[condition_rainfall_and_rain_today]
    print('Samples with RainToday="No" but with more than 1.0 mm Rainfall:', len(data_rainfall_and_rain_today))
    if len(data_rainfall_and_rain_today) == 0:
        print(f'Sanity check passed.')
    else:
        print(f'Sanity check failed.')


    print(f'Short Summary:\n'
          f'* 1 single sample with 18 different missing values\n'
          f'* 3 variables with extreme values as following:\n'
          f'  a) Rainfall (Negative).\n'
          f'  b) Cloud9am (999 - not in category list).\n'
          f'  c) WindDir3pm (ZZZ.. - not in category list).\n'
          f'* 2 continuous variables with a lot of missing values: Evaporation and Sunshine.\n\n'
          f'Handling those issues in data:')


    #Deleting the empty sample  from the data
    empty_row_condition = (df['MaxTemp'].isna() & df['Rainfall'].isna() & df['Evaporation'].isna() &
                            df['Sunshine'].isna() & df['WindGustDir'].isna() & df['WindDir9am'].isna() &
                            df['WindDir3pm'].isna() & df['WindSpeed9am'].isna() & df['WindSpeed3pm'].isna() &
                            df['Humidity9am'].isna() & df['Humidity3pm'].isna() & df['Pressure9am'].isna() &
                            df['Pressure3pm'].isna() & df['Cloud9am'].isna() & df['Cloud3pm'].isna() & df['Temp9am'].isna() &
                            df['Temp3pm'].isna())


    print(f'Sample with 18 missing values:\n{df[empty_row_condition]}')
    print(f'Decision: Delete sample\n'
          f'Reason: All data is from Uluru, only 178 samples out of 9000.\n'
          f'Any sample from Uluru might be meaningful for the model because we believe that Location might be an important feature.\n'
          f'The alternative of fill na values is probably not the right approach since 18 variables are almost the amount of all features.')
    df = df[~empty_row_condition]

    # Explore data of extreme values
    _explore_rainfall_variable()
    _explore_winddir3pm_variable()
    _explore_cloud9am_variable()


    # Filling Missing values

    #Evaporation - 1239 missing values

    print(f'-- Fill NA values --')
    print(f'-Evaporation-')
    print(f'Total missing values: 1239\n'
          f'Evaporation is a bit skewed and not symmetric, there are a lot of outliers (box-plot).\n'
          f'Therefore, we will add missing values by median.')
    evaporation_median = df['Evaporation'].median()
    df['Evaporation'].fillna(evaporation_median, inplace=True)


    #Sunshine - 1850 missing values, values make sense to reality

    #Sunshine is normally distributed, no skewed and no outliers.
    #Therefore, we will add missing values by mean.
    print(f'-Sunshine-')
    print(f'Total missing values: 1850\n'
          f'Sunshine is a not skewed and looks symmetric and passed normal dist test,there are no outliers (box-plot).\n'
          f'Therefore, we will add missing values by mean.')
    sunshine_mean = df['Sunshine'].mean()
    df['Sunshine'].fillna(sunshine_mean, inplace=True)


def feature_extraction(df: pd.DataFrame):
    pass


def write_df_to_path(df: pd.DataFrame, path: str):
    df.to_csv(path,sep='\t')
    print(f'Wrote dataframe to {path}')


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


def main(path_to_file):
    df = read_data(path_to_file)
    data_exploratory(df)
    handle_missing_and_extreme_values(df)
    feature_extraction(df)


def check_dependency_between_categorials(df:pd.DataFrame, categorial1,categorial2):


if __name__ == '__main__':
    path_to_file = r'E:\t2\machine\project\part_1\pythonProject\Xy_train.csv'
    main(path_to_file=path_to_file)
