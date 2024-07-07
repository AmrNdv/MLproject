import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import pearsonr, skew, normaltest, chi2_contingency, f_oneway
import statsmodels.api as sm
import statsmodels.formula.api as smf


from statsmodels.formula.api import ols
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

ALL_VARIABLES = {
    'target': ["RainTomorrow"],
    'continuous': ["MinTemp", "MaxTemp", "Rainfall", "Evaporation", "Sunshine", "WindGustSpeed", "WindSpeed9am",
                   "WindSpeed3pm", "Humidity9am", "Humidity3pm", "Pressure9am", "Pressure3pm", "Temp9am", "Temp3pm"],
    'categorical': ["RainToday", "Location", "WindGustDir", "WindDir9am", "WindDir3pm", "Cloud9am", "Cloud3pm",
                    "CloudsinJakarta"]
}
### Statistical tests functions ###

# Single variable

def check_normal_distribution(df: pd.DataFrame, x_value: str, p_value_threshold: float = 0.05):
    data = df[x_value].dropna(inplace=False)
    stat, p = normaltest(data)
    if p <= p_value_threshold:
        print(f'Normal distribution test: {x_value} is distributed normally, p-value: {round(p,5)}')
    else:
        print(f'Normal distribution test: {x_value} is NOT distributed normally, p-value: {round(p,5)}')


def check_skewness_distribution(df: pd.DataFrame, x_value: str):
    data = df[x_value].dropna(inplace=False)
    skewness = skew(data)
    if -0.5 <= skewness <= 0.5:
        print(f'Skewness distribution test: {x_value} data is approximately symmetric, skewness: {round(skewness,5)}')
    elif 0.5 < skewness <= 1:
        print(f'Skewness distribution test: {x_value} data is moderately right skewed, skewness: {round(skewness,5)}')
    elif skewness > 1:
        print(f'Skewness distribution test: {x_value} data is highly right skewed, skewness: {round(skewness,5)}')
    elif -0.5 >= skewness > -1:
        print(f'Skewness distribution test: {x_value} data is moderately left skewed, skewness: {round(skewness,5)}')
    elif skewness < -1:
        print(f'Skewness distribution test: {x_value} data is highly left skewed, skewness: {round(skewness,5)}')


# Multiple variables

def check_chi2_square(df:pd.DataFrame, x_value: str, y_value: str, p_value_threshold: float = 0.05):
    crosstab = pd.crosstab(df[x_value], df[y_value])
    chi2, p, dof, expected = chi2_contingency(crosstab)
    if p <= p_value_threshold:
        print(f'Chi square test: The correlation between {x_value}, {y_value} are statistically significant, p-value: {round(p,5)}')
    else:
        print(f'Chi square test: The correlation between {x_value}, {y_value} are NOT statistically significant, p-value: {round(p,5)}')


def calculate_pearson_coefficient(df: pd.DataFrame, x_value: str, y_value: str, p_value_threshold: float = 0.05):
    data = df.dropna(subset=x_value, inplace=False)
    data = data.dropna(subset=y_value, inplace=False)
    correlation, p_value = pearsonr(data[x_value], data[y_value])
    if p_value <= p_value_threshold:
        print(f'Pearson test: The correlation between {x_value}, {y_value} are statistically significant, p-values: {round(p_value,5)}')
    else:
        print(f'Pearson test: The correlation between {x_value}, {y_value} are NOT statistically significant, p-values: {round(p_value,5)}')


### plots functions ###

# Single Variable

def create_box_plot(df: pd.DataFrame, y_label: str, show: bool = True, title: str = None):
    # Box plots for Continuous variables
    if not show:
        return

    boxPlot = plt.axes()
    sns.boxplot(y=y_label, data=df)
    if title:
        boxPlot.set_title(title)
    else:
        boxPlot.set_title(f'{y_label} Boxplot')
    plt.show()


def create_histogram(df: pd.DataFrame, y_label: str, bins: int = 10, show: bool = True, title: str = None):
    # Histogram - Continuous variables
    if not show:
        return

    plt.hist(df[y_label], bins=bins, density=True)
    if title:
        plt.title(title)
    else:
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

    plot_graphs(df=df, variables_dict=ALL_VARIABLES, show=False)
    describe_variables(df=df, variables_dict=ALL_VARIABLES)
    plot_heatmap(df=df, variables_dict=ALL_VARIABLES, show=False)

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
              f'{round(len(df_consistence_wind_dir_in_canberra_and_no_rain)/len(df[canberra_and_no_rain]),5)}\n'
              f'   g) Samples size of samples from Canberra, No Rain and WindDir9am is N (North) {len(df_in_canberra_and_no_rain_and_wind_dir_9am_is_North)}'
              f'   g) Consistency when WinDir9am equals to WindDir3pm in Canberra, No Rain and WinDir9am is N (North): {len(canberra_and_no_rain_and_wind_dir_9am_equals_3pm_and_is_N)}'
              f'   g) Ratio of g/e: {round(len(df_consistence_wind_dir_in_canberra_and_North_wind_dir_9am) / len(df_in_canberra_and_no_rain_and_wind_dir_9am_is_North), 2)}')
        print(f'Decision: Set WinDir3pm value to N (North)\n'
              f'Reason: Seems to have high probability when grouped those values above so.\n'
              f'North is most common among this data type.')

        create_count_plot(df=df_in_canberra_and_no_rain_and_wind_dir_9am_is_North, y_label="WindDir3pm", show=False)
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
          f'Therefore, we will add missing values by median.'
          f'Showing plot of Box-Plot and Histogram when By RainTomorrow')
    create_histogram(df=df[df['RainTomorrow'] == "Yes"], y_label="Evaporation",  title="Evaporation Histogram, RainTomorrow='Yes'", show=False)
    create_box_plot(df=df[df['RainTomorrow'] == "Yes"], y_label="Evaporation",  title="Evaporation Boxplot, RainTomorrow='Yes'", show=False)
    create_histogram(df=df[df['RainTomorrow'] == "No"], y_label="Evaporation",  title="Evaporation Histogram, RainTomorrow='No'", show=False)
    create_box_plot(df=df[df['RainTomorrow'] == "No"], y_label="Evaporation", title="Evaporation Boxplot, RainTomorrow='No'", show=False)
    print(f'Calculating Normal and Skewed Distribution p-value:')
    check_normal_distribution(df=df[df['RainTomorrow'] == 'Yes'], x_value="Evaporation")
    check_skewness_distribution(df=df[df['RainTomorrow'] == 'Yes'], x_value="Evaporation")
    evaporation_raintomorrow_yes_median = df[df['RainTomorrow'] == "Yes"]['Evaporation'].median()
    evaporation_raintomorrow_no_median = df[df['RainTomorrow'] == "No"]['Evaporation'].median()
    df.loc[df["RainTomorrow"] == "Yes", 'Evaporation'] = df.loc[df["RainTomorrow"] == "Yes", 'Evaporation'].fillna(
        evaporation_raintomorrow_yes_median)
    df.loc[df["RainTomorrow"] == "No", 'Evaporation'] = df.loc[df["RainTomorrow"] == "No", 'Evaporation'].fillna(
        evaporation_raintomorrow_no_median)

    #Sunshine - 1850 missing values, values make sense to reality

    #Sunshine is normally distributed, no skewed and no outliers.
    #Therefore, we will add missing values by mean.
    print(f'-Sunshine-')
    print(f'Total missing values: 1850\n'
          f'Sunshine is a not skewed and looks symmetric and passed normal dist test,there are no outliers (box-plot).\n'
          f'Therefore, we will add missing values by mean.'
          f'Showing plot of Box-Plot and Histogram when By RainTomorrow')
    create_histogram(df=df[df['RainTomorrow'] == "Yes"], y_label="Sunshine", title="Sunshine Histogram, RainTomorrow='Yes'", show=False)
    create_box_plot(df=df[df['RainTomorrow'] == "Yes"], y_label="Sunshine", title="Sunshine Boxplot, RainTomorrow='Yes'", show=False)
    create_histogram(df=df[df['RainTomorrow'] == "No"], y_label="Sunshine",  title="Sunshine Histogram, RainTomorrow='No'",show=False)
    create_box_plot(df=df[df['RainTomorrow'] == "No"], y_label="Sunshine",  title="Sunshine Boxplot, RainTomorrow='No'", show=False)

    print(f'Calculating Normal and Skewed Distribution p-value:')
    check_normal_distribution(df=df[df['RainTomorrow'] == 'Yes'], x_value="Sunshine")
    check_skewness_distribution(df=df[df['RainTomorrow'] == 'Yes'], x_value="Sunshine")

    sunshine_raintomorrow_yes_mean = df[df['RainTomorrow'] == "Yes"]['Sunshine'].mean()
    sunshine_raintomorrow_no_mean = df[df['RainTomorrow'] == "No"]['Sunshine'].mean()
    df.loc[df["RainTomorrow"] == "Yes", 'Sunshine'] = df.loc[df["RainTomorrow"] == "Yes", 'Sunshine'].fillna(
        sunshine_raintomorrow_yes_mean)
    df.loc[df["RainTomorrow"] == "No", 'Sunshine'] = df.loc[df["RainTomorrow"] == "No", 'Sunshine'].fillna(
        sunshine_raintomorrow_no_mean)

    return df

def categorized_features(df: pd.DataFrame):

    def categorized_column_by_equal_range(column: str, num_of_bins: int):
        # Categorize the non-NaN values into N categories of equal range

        new_column_name = f'{column}_Category'
        df[column], bins = pd.cut(df[column], bins=num_of_bins, labels=False, retbins=True)
        df[column] += 1
        df[column].astype(str)
        print(f"Equal Range Bins for {column}: {bins}")


    def categorized_column_by_equal_bins(column: str, num_of_bins: int):
        # Categorize the non-NaN values into N categories with equal number of data points


        new_column_name = f'{column}_Category'
        df[new_column_name], bins = pd.qcut(df[column], q=num_of_bins, labels=False, retbins=True)
        df[new_column_name] += 1
        print(f"Equal Size Bins for {column}: {bins}")


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

def feature_extraction(df: pd.DataFrame):
    categorized_features(df)
    create_new_features_combinations(df)

def calculate_vif(input_data, dependent_col):
    x_vars=input_data.drop([dependent_col], axis=1)
    xvar_names=x_vars.columns
    for i in range(0,xvar_names.shape[0]):
        y=x_vars[xvar_names[i]]
        x=x_vars[xvar_names.drop(xvar_names[i])]
        rsq=smf.ols(formula="y~x", data=x_vars).fit().rsquared
        vif=round(1/(1-rsq),2)
        print (xvar_names[i], " VIF = " , vif)

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
        return model

    # Example usage
    # final_model = _forwardSelection(df, 'RainTomorrow')
    print(df.columns)
    lm = _forwardSelection(df, 'RainTomorrow')
    lm.summary()
    print(lm.model.formula)
    print(lm.rsquared_adj)


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

def validate_categorical_dummy(df):
    categorical_columns = [col for col in df.columns if 'categorical' in col.lower() or col.lower() in ALL_VARIABLES['categorical']]
    for col in categorical_columns:
        if col not in df.columns:  # Check if dummy variables already exist
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df, dummies], axis=1)
    return df

# 5 functions that shows dependencies between variables #
"""
Run the function check_dependencies_procedure(path)
with path = data frame path
to see dependencies graph.
please edit graph inputs inside check_dependencies_procedure function.
"""
def preview_dependency_between_categorials(df:pd.DataFrame, categorial1:str,
                                           categorial2:str,meassured_category:str):
    """"
    Description:
    inputs:
    1. path to file - string
    2. categorial1 - first categorial feature
    3. categiral2 - second categorial feature
    4. meassured_category - one category of the second variable  (e.g "Yes")
        that we want to show how it behave in different categories of categorial1
    """
    contingency_table = pd.crosstab(df[categorial1], df[categorial2])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    # Calculate the probabilities for feature value 2
    rates = contingency_table[meassured_category] / contingency_table.sum(axis=1)

    # Calculate the overall probability of 'yes'
    overall_yes_rate = df[categorial2].value_counts(normalize=True)[meassured_category]

    # Add the overall probability to the rates series
    rates['Ovel All'] = overall_yes_rate

    # Plot the rates
    rates.plot(kind='bar',fontsize=16, color='blue', alpha=0.7, label=f'Probability of {categorial2}-{meassured_category}'
                                                          f' for different categories of {categorial1}')
    plt.title(f'Rate of {meassured_category} for {categorial2} for Each Category of {categorial1}',
              fontsize=16)
    plt.xlabel(f'{categorial1} Categories',fontsize=16)
    plt.ylabel(f'Probability of {categorial2}',fontsize=16)
    plt.xticks(rotation=0)

    yaxis_ceil = min(np.ceil(rates.max() * 10)/10+0.15,1) # Automatic definition of graph y axis size
    plt.ylim(0, yaxis_ceil)

    plt.axhline(y=overall_yes_rate, color='red', linestyle='--', linewidth=1,
                label=f'Overall Probability of {categorial2}-{meassured_category} ')

    plt.text(len(rates) - 1, overall_yes_rate + 0.005, f'{overall_yes_rate:.2f}', color='red', ha='center'
             ,fontsize=18)

    for i in range(len(rates) - 1):
        plt.text(i, rates.iloc[i] + 0.005, f'{rates.iloc[i]:.2f}', color='black', ha='center'
                 ,fontsize=18)
    plt.legend(fontsize=16,loc='upper left')

    plt.show()

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

def check_dependencies_procedure(path_to_file):
    """
    Stand Alone function that produce dependencies graphs as shown in the project.
    Need to edit the parameters from inside the function
    :return:
    """
    df = pd.read_csv(path_to_file, encoding='ISO-8859-1')
    categorials = ['Location', 'WindGustDir', 'WindDir3pm', 'WindDir9am',
                   'Cloud9am', 'Cloud3pm', 'CloudsinJakarta', 'RainToday', 'RainTomorrow']

    df = handle_missing_and_extreme_values(df)
    df = categorize_df(df,categorials,4,bins_mode='equal_range')
    preview_dependency_between_categorials(df,
                                          'Rainfall_categorized',
                                           'RainTomorrow','Yes')

# UnUsed code
# def handle_manual(df:pd.DataFrame):
#     """
#     function that handle all the manual changes in the dataset.
#     """
#     df['Rainfall'] = df['Rainfall'].replace(-3, 0)
#     df['Cloud9am'] = df['Cloud9am'].replace(999,7)
#     df['WindDir3pm'] = df['WindDir3pm'].replace('zzzzzzzzzz…...', 'N')
#     # df = df.drop('CloudsinJakarta', axis=1)
#     return df
#
# def handle_NaNs(df:pd.DataFrame, categorials:list,debug_mode:bool=False):
#     """
#     This function fills NaN values in each column with the mean value of the column.
#     The mean value is calculated separately for rows where the 'Raintomorrow' property is 'yes' or 'no'.
#     """
#     df_copy = df.copy()
#
#     for column in df_copy.columns:
#         if column in categorials: # skip already categorials (manually selected)
#             pass
#         else:
#             if df_copy[column].isna().any():
#                 print(f'{column} has {df_copy[column].isna().sum()} missing values') if debug_mode else None  # Debug - print column name
#
#                 # Calculate the mean values separately for 'yes' and 'no' in 'Raintomorrow'
#                 mean_yes = df_copy[df_copy['RainTomorrow'] == 'Yes'][column].mean()
#                 print(f'mean yes : {mean_yes}') if debug_mode else None  # Debug - print column mean yes
#
#                 mean_no = df_copy[df_copy['RainTomorrow'] == 'No'][column].mean()
#                 print(f'mean no : {mean_no}') if debug_mode else None  # Debug - print column mean no
#
#                 # Replace NaN values with the corresponding mean value
#                 df_copy.loc[(df_copy['RainTomorrow'] == 'Yes') & (df_copy[column].isna()), column] = mean_yes
#                 df_copy.loc[(df_copy['RainTomorrow'] == 'No') & (df_copy[column].isna()), column] = mean_no
#             else:
#                 print(f'{column} has no missing values') if debug_mode else None  # Debug - print column name
#
#     return df_copy

def main(path_to_file):

    df = read_data(path_to_file)
    data_exploratory(df)
    df = handle_missing_and_extreme_values(df)
    feature_extraction(df)
    df_with_dummies = pd.get_dummies(df, drop_first=True)
    df_with_dummies = validate_categorical_dummy(df_with_dummies)
    print(df.columns)
    feature_selection(df_with_dummies)

if __name__ == '__main__':
    path_to_file = r'E:\t2\machine\project\part_1\pythonProject\Xy_train.csv'
    main(path_to_file=path_to_file)
    # path_to_file = r'E:\t2\machine\project\part_1\pythonProject\Xy_train.csv'
    # check_dependencies_procedure(path_to_file)
