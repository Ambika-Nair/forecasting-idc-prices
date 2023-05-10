import os
import pandas as pd
import numpy as np
import time
import plotly.express as px
import xgboost as xgb
from datetime import datetime
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from collections import defaultdict, Counter
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score, KFold
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from math import sqrt
import pytz
import mlflow
import mlflow.sklearn
mlflow.set_experiment('Tracking_Forecasting_XGBoost')


def reduce_mem_usage(df):
    """Optimizes the memory usage of a DataFrame by modifying the datatype of each column.

    Args:
        df (pd.DataFrame): The Dataframe to optimize

    Returns:
        pd.DataFrame: The optimized DataFrame
    """

    ###iterate through all the columns of a dataframe and modify the data type to reduce memory usage.
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')
    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df

##Accessing csv files from directory
csv_files = []
startdate  = datetime.strptime("2022-01-31 00:00:00", "%Y-%m-%d %H:%M:%S")
enddate = datetime.strptime("2022-02-05 23:45:00", "%Y-%m-%d %H:%M:%S")
path = os.path.realpath(os.path.join(os.getcwd(),os.path.dirname(__file__)))
print("Path is:", path)
for root, dirs, files in os.walk(path):
    for file in files:
        if file.startswith("pri_de_intraday_vwap_last5min_EURmwh_cet_min15_ca_") and file.endswith(".csv"):
            file_date = datetime.strptime(os.path.basename(file), "pri_de_intraday_vwap_last5min_EURmwh_cet_min15_ca_%Y-%m-%d.csv")
            if startdate <= file_date <= enddate:
                csv_files.append(os.path.join(root, file))
csv_files.reverse()

##Reading csv file
def read_csv(file_name):
    """ Reads a CSV file and returns a pandas DataFrame object containing the data from the file.
    Args:
        file_name (str): The name of the CSV file to be read.

    Returns:
        pandas.DataFrame: A pandas dataframe object containing the data from the CSV file.
    """
    df = pd.read_csv(file_name, decimal=",", delimiter=";", index_col=0)
    df.index = pd.to_datetime(df.index, dayfirst=False, utc = True)
    df.index = df.index.tz_convert('Europe/Berlin')
    df.rename_axis('date', inplace = True)
    #Dropping the last rows of the following month to have dates in sync with columns.
    df = df.drop(df.loc[df.index > pd.Timestamp(enddate).tz_localize('Europe/Berlin')].index)  
    return df

##Data preprocessing
def preprocess_data(df):
    """ Preprocesses the raw data structure from the CSV files.
    Args:
        df (pandas.DataFrame): The pandas dataframe with the raw data from csv files

    Returns:
        pandas.DataFrame: A preprocessed dataframe
    """
    df.drop(df.columns[0:180], axis = 1, inplace = True)  #Dropping all columns for day before until 15:00
    df.fillna(axis=1, method='backfill', inplace = True)   #Backfill 'NaN' with next corresponding values

    #Removing prices after fullfillment time
    for idx, column in enumerate(df.columns):
        df.loc[df.index == column, df.columns[(idx+1):]]=None
        # df_main.reset_index(inplace=True)
    return df

###Restructuring the dataframe
def restructure(df_preprocessed):
    """Restructures the pre-processed dataframe such that it has a single date column and multiple columns representing 
    the time difference between the date and the original columns in the preprocessed dataframe.

    Args:
        df_preprocessed (pandas.DataFrame): A pandas dataframe object containing the preprocessed data.

    Returns:
        pandas.DataFrame: A pandas dataframe object containing the restructured data.
    """
    df_preprocessed.reset_index(inplace=True)
    df_preprocessed = (df_preprocessed.melt('date', var_name='date2') # reshape the columns to rows
   # convert the date strings to datetime and compute the timedelta
   .assign(date=lambda d: pd.to_datetime(d['date']),
           date2=lambda d: pd.to_datetime(d['date2']),
           delta=lambda d: d['date'].sub(d['date2'])
                           .dt.total_seconds().floordiv(60)
          )
   # filter out negative timedelta
   .loc[lambda d: d['delta'].ge(0)]
   # reshape the rows back to columns
   .pivot('date', 'delta', 'value')
   # rename columns from integer to "Xmins"
   .rename(columns=lambda x: f'{x:.0f}') 
   # remove columns axis label
   .rename_axis(columns=None)
)
    df_preprocessed.loc[df_preprocessed.isna().all(axis=1)] = 0     #Checking for rows with all na values and replacing it with 0
    df_preprocessed.dropna(axis='columns', inplace = True)  ##Dropping columns with NaN values
    df_preprocessed = df_preprocessed[df_preprocessed.columns[::-1]]        #Reversing the order of columns

    return df_preprocessed
#For all the rows (products), all prices are available starting from 540 mins(9 hours) before fulfillment time. 

def access_features_csv(file_name:str):
    """ Reads a CSV file with the given name and returns a pandas dataframe object containing the data from the file.

    Args:
        file_name (str): A string that represents the name of the CSV file to be read.

    Returns:
        pandas.DataFrame: A pandas dataframe object containing the data from the CSV file.
    """
    path = os.path.realpath(os.path.join(os.getcwd(),os.path.dirname(__file__)))
    # print("Path of csv is:", path)
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.startswith(file_name) and file.endswith(".csv"):
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path, decimal=",", delimiter=";", index_col=0)
                df.index = pd.to_datetime(df.index, dayfirst=False, utc = True)     #Converting index into Timestamp type
                df = df.fillna(0)
    return df

def feature_sys_imb(df_product, df_imb):
    """Adds system imbalance volume column with an offset of 30 minutes to the input dataframe.

    Args:
        df_product (pandas.DataFrame): Pandas restructed dataframe that contains the product data.
        df_imb (pandas.DataFrame): Pandas dataframe containing the system imbalance data.

    Returns:
        pandas.DataFrame: A pandas dataframe containing the input product data with an additional column for system imbalance volume.
    """
    anchor_time = pd.to_datetime(df_product.columns[-1])
    lookback_minutes = 30
    df_product['date'] = pd.to_numeric(df_product['date'])
    df_product_imb = pd.DataFrame(df_product.set_index(anchor_time - pd.to_timedelta(df_product['date'],unit='min') - pd.to_timedelta(lookback_minutes, unit='min'))
        .join(df_imb).reset_index(drop=True))
    df_product_imb['VOL_IMB_DE'].fillna(method='ffill',inplace=True)
    return df_product_imb

def feature_rdl(df_product_imb, df_rdl):
    """Adds residual load column with an offset of 30 minutes to the input Dataframe 'df_product_imb'.

    Args:
        df_product_imb (pandas.DataFrame): Pandas dataframe that contains the product data and system imbalance volume data.
        df_rdl (pandas.DataFrame): Pandas dataframe containing residual load data.

    Returns:
        pandas.DataFrame: It returns a pandas DataFrame with residual load column added to the input df_product_imb.
    """
    anchor_time = pd.to_datetime(df_product_imb.columns[-2])
    lookback_minutes = 60
    df_imb_rdl = pd.DataFrame(df_product_imb.set_index(anchor_time - pd.to_timedelta(df_product_imb['date'] + lookback_minutes, unit='min'))
        .join(df_rdl).reset_index(drop=True))
    df_imb_rdl['Residual_Load_DE'].fillna(method='ffill',inplace=True)
    # df_imb_rdl.set_index("date", inplace = True)
    return df_imb_rdl

def feature_spv(df_imb_rdl, df_spv):
    """Adds the solar pv protection with no offset to input DataFrame 'df_imb_rdl'.

    Args:
        df_imb_rdl (pandas.DataFrame): Pandas dataframe that contains product data, system imbalance volume data and residual load.
        df_spv (pandas.DataFrame): Pandas dataframe containing solar pv data.

    Returns:
        pandas.DataFrame: Pandas DataFrame with solar pv column added to the input 'df_imb_rdl'.
    """
    df_imb_rdl_spv = pd.DataFrame(df_imb_rdl.set_index(pd.to_datetime(df_imb_rdl.columns[-3]) + pd.to_timedelta(-df_imb_rdl['date'], unit='min'))
        .join(df_spv).reset_index(drop=True) )
    df_imb_rdl_spv['Solar_Power_DE'].fillna(method='ffill',inplace=True)
    # df_imb_rdl_spv.set_index("date", inplace = True) 
    return df_imb_rdl_spv

def feature_wnd(df_imb_rdl_spv, df_wnd):
    """Adds the wind protection with no offset to input DataFrame 'df_imb_rdl_spv'.

    Args:
        df_imb_rdl_spv (pandas.DataFrame): Pandas dataframe that contains product data, system imbalance volume data,residual load and solar pv data.
        df_wnd (_type_): Pandas dataframe containing wind production data.

    Returns:
        pandas.DataFrame: Pandas DataFrame with wind production column added to the input 'df_imb_rdl_spv'.
    """
    df_imb_rdl_spv_wnd = pd.DataFrame(df_imb_rdl_spv.set_index(pd.to_datetime(df_imb_rdl_spv.columns[-4]) + pd.to_timedelta(-df_imb_rdl_spv['date'], unit='min'))
        .join(df_wnd).reset_index(drop=True) )
    df_imb_rdl_spv_wnd['Wind_Power_DE'].fillna(method='ffill',inplace=True)
    df_imb_rdl_spv_wnd.set_index("date", inplace = True) 
    return df_imb_rdl_spv_wnd

def minutes_to_timestamp(df):
    """Converts the 'time_before_fulfilment' column in minutes to timestamp format and set it as the index of the input dataframe.

    Args:
        df (pandas.DataFrame): The input dataframe containing the 'time_before_fulfilment' column to be converted to timestamp format.

    Returns:
        pandas.DataFrame: The modified dataframe with the 'time_before_fulfilment' column converted to timestamp format and set as the index.
    """
### Replacing minutes before fulfilment column to timestamp 
    df['time_before_fulfilment'] = (pd.to_timedelta(df['time_before_fulfilment'], unit='min')
                    .rsub(pd.to_datetime(df.columns[1]))
                    )
    df.set_index('time_before_fulfilment', inplace=True)
    df.index = df.index.tz_convert(pytz.FixedOffset(60))

    ##Converting the column name from type timestamp to str
    timestamp = pd.Timestamp(df.columns[0])
    date_string = timestamp.strftime('%Y-%m-%d %H:%M:%S%z')
    df = df.rename(columns={df.columns[0]: date_string})
    df
    return df

def add_lags(df):
    """Adds three lag columns to the input DataFrame.


    Args:
        df (pandas.DataFrame): The input DataFrame is the one that was modified in function 'minutes_to_timestamp'.

    Returns:
        pandas.DataFrame: The DataFrame with three additional columns 'lag1', 'lag2', and 'lag3'.
    """
    col = df.iloc[:, 0]
    target_map = col.to_dict()
    target_map
    df['lag1'] = (df.index -pd.Timedelta('3 hours')).map(target_map)
    df['lag2'] = (df.index -pd.Timedelta('4 hours')).map(target_map)
    df['lag3'] = (df.index -pd.Timedelta('5 hours')).map(target_map)

    return df

def cross_validation(df):
    """Splits the input dataframe into train and test sets using time series split and returns the necessary features and targets for each set.

    Args:
        df (pandas.DataFrame): Input dataframe is the dataframe that includes lags.

    Returns:
        tuple: A tuple containing the following elements:
            - pandas.DataFrame: The training data.
            - pandas.DataFrame: The validation data.
            - pandas.DataFrame: The features for the training data.
            - pandas.DataFrame: The target for the training data.
            - pandas.DataFrame: The features for the validation data.
            - pandas.DataFrame: The target for the validation data.
            - list: A list of feature names.
            - list: A list of target names.
    """
    tss = TimeSeriesSplit(n_splits = 5)
    df = df.sort_index()

    for train_idx, val_idx in tss.split(df):
        train = df.iloc[train_idx]
        test = df.iloc[val_idx]

        FEATURES = ['VOL_IMB_DE','Residual_Load_DE','Solar_Power_DE','Wind_Power_DE', 'lag1', 'lag2', 'lag3']
        TARGET = [df.columns[0]]

        X_train = train[FEATURES] 
        y_train = train[TARGET]

        X_test = test[FEATURES] 
        y_test = test[TARGET]

    return train, test, X_train, y_train, X_test, y_test, FEATURES, TARGET

def random_search_optimization(X_train, y_train):
    """Performs hyperparameter optimization using Random Search algorithm on a XGBoost model.

    Args:
        X_train (pandas.DataFrame): The feature matrix of training data.
        y_train (pandas.DataFrame): The target vector of training data.

    Returns:
        sklearn.model_selection.RandomizedSearchCV: The optimized XGBoost model.
    """
    ### Optimization Algorithm - Random Search
    params = {
           'max_depth': range(3,10),
           'learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.02, 0.2, 0.03, 0.3, 0.04, 0.4, 0.05, 0.5, 0.6, 1.0],
           'n_estimators' : [100,500,1000,2000],
           'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
           'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
           'colsample_bylevel': np.arange(0.4, 1.0, 0.1),
           'gamma' :  [0, 0.1, 0.2, 0.3, 0.4],
           'min_child_weight' : [1,5,10,15],
           'reg_alpha': [0, 0.1, 0.5, 1, 10],
           'reg_lambda': [0, 0.1, 0.5, 1, 10]
            }
    ### Set up the k-fold cross-validation
    kfold = KFold(n_splits=4, shuffle=True, random_state=10)

    xgb_rs = xgb.XGBRegressor(seed = 20).fit(X_train, y_train)

    model_random_search = RandomizedSearchCV(
                            estimator=xgb_rs,
                            param_distributions=params,
                            scoring='neg_mean_squared_error',
                            cv = kfold,
                            n_jobs = -1,       
                            n_iter=25,
                            verbose=1)

    ###Fit Random Search
    rs_params = model_random_search.fit(X_train, y_train)

    return rs_params
    
def get_common_parameters(xgboost_rs_results):
    """Get the highest occurring parameter values.

    Args:
         xgboost_rs_results (list): A list of dictionaries containing the results of random search optimization for XGBoost.

    Returns:
        list: A list of dictionaries containing the top parameter values that occurred the most frequently in the random search optimization.
    """
    param_counts_by_value = defaultdict(Counter)
    for d in xgboost_rs_results:
        for k, v in d.items():
            param_counts_by_value[k][v] += 1

    top_param_values = [{top_params: value_counts.most_common(1)[0][0]} for top_params, value_counts in param_counts_by_value.items()]

    return top_param_values

def  top_parameters(top_param_values):
    """Combines list of dictionaries into one dictionary.

    Args:
        top_param_values (list[dict]): A list of dictionaries containing the top parameter values.

    Returns:
        dict: A dictionary containing the combined top parameter values.
    """
    parameters_dict = {}
    parameters_dict = {k: v for d in top_param_values for k, v in d.items()}
    return parameters_dict

def xgbmodel(X_train, y_train, X_test, y_test, top_parameters_dict):
    """Trains an XGBoost model with the specified hyperparameters using the given training data, and evaluates its performance on the given test data.

    Args:
        X_train (numpy.ndarray): A 2D array containing the features of the training data.
        y_train (numpy.ndarray): A 1D array containing the target values of the training data.
        X_test (numpy.ndarray): A 2D array containing the features of the test data.
        y_test (numpy.ndarray): A 1D array containing the target values of the test data.
        top_parameters_dict (dict): A dictionary containing the best hyperparameters found during hyperparameter tuning.

    Returns:
        xgb.XGBRegressor: The trained XGBoost model.
    """
    model = xgb.XGBRegressor(base_score=0.5, objective= 'reg:squarederror', booster = 'gbtree', early_stopping_rounds = 20, 
                             n_estimators = top_parameters_dict['n_estimators'], max_depth = top_parameters_dict['max_depth'], 
                             learning_rate = top_parameters_dict['learning_rate'], colsample_bytree = top_parameters_dict['colsample_bytree'],
                             colsample_bylevel = top_parameters_dict['colsample_bylevel'], subsample = top_parameters_dict['subsample'],
                             min_child_weight = top_parameters_dict['min_child_weight'], gamma = top_parameters_dict['gamma'],
                             reg_alpha = top_parameters_dict['reg_alpha'], reg_lambda = top_parameters_dict['reg_lambda'] )    
    model.fit(X_train, y_train,
    eval_set = [(X_train, y_train), (X_test, y_test)],
    verbose = 1, eval_metric = 'rmse')
    return model

def plot_product_graph(df):
    """Plots the time-series graph of product and its features for a given DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame containing the product and features data.
    """
    fig = px.line(df, x = df.index, y = [df.columns[0],'VOL_IMB_DE','Residual_Load_DE','Solar_Power_DE','Wind_Power_DE'] , markers='.')

    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count = 1, label = "1H", step = "hour", stepmode ="backward"),
                dict(step="all")
            ])
        )
    )
    fig.show()

def predict_against_test(model, X_test,):
    """Predicts target values against test dataset using trained model.

    Args:
        model (object): A trained model object which can predict.
        X_test (pd.DataFrame): The test input data for prediction.

    Returns:
        np.ndarray: The predicted target values.
    """
    y_pred = model.predict(X_test)
    return y_pred

def forecast_plot(df):
    """Generates plot for IDC electricity market price forecast using XGBoost.

    Args:
        df (pandas.DataFrame): A dataframe containing columns with past prices, forecasted prices and real prices.
    """
    fig = px.line(df, x = df.index, y = [df.columns[0],'Forecasted Prices', 'Real Prices'], markers='.',
                  title = f'IDC electricity market price forecast using XGBoost for: {df.columns[0]}'
                )   
    fig.update_xaxes(
        title = 't-minutes before fulfillment',
        tickangle = -90,

        # autorange="reversed",
        # rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count = 1, label = "1H", step = "hour", stepmode ="backward"),
                dict(step="all")
            ])
        )
    )
    fig.update_yaxes(title="IDC Price VWAP5Minutes [€/MWh]")
    fig.update_traces(
        name="Past Prices", # change the name of the first y-axis attribute
        selector=dict(name=df.columns[0])
    )
    fig.update_layout(legend_title_text="")
    fig.show()

def eval_metrics(actual, pred, df):
    """Calculate evaluation metrics and log them to MLflow.

    Args:
        actual (np.array): Array of actual values.
        pred (np.array): Array of predicted values.
        df (pd.DataFrame): Dataframe containing the original data.

    Returns:
        Tuple(float, float, float): Tuple containing the MAE, RMSE, and MAPE.
    """
    mae = mean_absolute_error(actual, pred)
    print("MAE : "+str(mae))

    rmse = sqrt(mean_squared_error(actual, pred))
    print("RMSE : "+str(rmse))

    r2 = r2_score(actual, pred)
    print("R2 : "+str(r2))

    mape = mean_absolute_percentage_error (actual, pred)
    print("MAPE : "+str(mape))

    mlflow.log_param("Timestamp", df.columns[0])
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("mape", mape)

    return mae, rmse, mape
 

############################## Mainline processing starts here #############################################
def main():
    start = time.time()

    print("**Execution Starts!!**")
    ##Merging the files into one dataframe
    df = pd.merge(
        read_csv(csv_files.pop()),
        read_csv(csv_files.pop()),
        left_index=True, 
        right_index=True
    )
    while csv_files:
        df = pd.merge(
            df, 
            read_csv(csv_files.pop()), 
            left_index=True, 
            right_index=True,
            how='outer'
        )
    print("Files merged into one dataframe")
    print(df)

    print("################################################################################################################################")

    print("**Preprocessing starts!**")
    df_pre = preprocess_data(df)
    print("**Preprocessing done!**")
    print(df_pre)

    print("################################################################################################################################")

    print("**Restructuring starts!**")
    df_re = restructure(df_pre)
    print("**Restructuring done!**")
    print(df_re)

    print("################################################################################################################################")

    ##Transposing the dataframe.
    df_transposed = df_re.rename_axis(index=None, columns='date').T
    print("Transposed Data")

    print("################################################################################################################################")

    df_full = df_transposed.copy()
    reduce_mem_usage(df_full)
    print("DF Full")
    print(df_full)

    print("################################################################################################################################")

    ###Reading the sys vol imb csv
    print("Reading system volume imbalance data...")
    df_imb = access_features_csv('vol_de_imb_sys_mw_cet_min15_a_2022')
    df_imb.rename(columns={'vol de imb sys mw cet min15 a':'VOL_IMB_DE'}, inplace=True)        #Renaming column
    df_imb = df_imb.fillna(0)

    ###Reading file for residual load
    print("Reading residual load data...")
    df_rdl = access_features_csv('rdl_de_mwhh_cet_min15_a_2022')
    df_rdl.rename(columns={'rdl de mwh/h cet min15 a':'Residual_Load_DE'}, inplace=True)        #Renaming column
    df_rdl = df_rdl.fillna(0)

    ###Reading file for solar pv production
    print("Reading Solar PV production data...")
    df_spv = access_features_csv('pro_de_spv_mwhh_cet_min15_a_2022')
    df_spv.rename(columns={'pro de spv mwh/h cet min15 a':'Solar_Power_DE'}, inplace=True)        #Renaming column
    df_spv = df_spv.fillna(0)
   
    ###Reading file for wind power
    print("Reading wind production data...")
    df_wnd = access_features_csv('pro_de_wnd_mwhh_cet_min15_a_2022.csv')
    df_wnd.rename(columns={'pro de wnd mwh/h cet min15 a':'Wind_Power_DE'}, inplace=True)        #Renaming column
    df_wnd = df_wnd.fillna(0)
   
    print("################################################################################################################################")

    ###Creating lists to store results. 
    df_fi = pd.DataFrame()
    xgboost_rs_results = []
    preds = []

    print("Running functions on each column...")
    for col in df_full.columns[:]:
        # create a new DataFrame with only the current column
        col_df = df_full[[col]]
        col_df.reset_index(inplace=True)
        print("Col_DF:")
        print(col_df)

        ### perform functions on col_df here
        print("Adding the sys imb vol column with offset of 30 mins.")
        df_product_imb = feature_sys_imb(col_df, df_imb)
        # print(df_product_imb)
        print("Adding the residual column with offset of 60 mins.")
        df_imb_rdl = feature_rdl(df_product_imb, df_rdl)
        # print(df_imb_rdl)
        print("Adding the solar pv protection with no offset.")
        df_imb_rdl_spv = feature_spv(df_imb_rdl, df_spv)
        # print(df_imb_rdl_spv)
        print("Adding the wind power with no offset.")
        df_imb_rdl_spv_wnd = feature_wnd(df_imb_rdl_spv, df_wnd)
        # print(df_imb_rdl_spv_wnd)

        print("################################################################################################################################")

        df_main = df_imb_rdl_spv_wnd.copy()
        df_main.reset_index(inplace = True)
        df_main.rename(columns={'date':'time_before_fulfilment'},inplace=True)
        print("Df of",df_main.columns[0])
        print(df_main)
    
        print("################################################################################################################################")

        df_main = minutes_to_timestamp(df_main)
        print("DF_MAIN", df_main)

        print("################################################################################################################################")

        df_main = add_lags(df_main)
        print("Adding lags to dataframe...")
        print(df_main)

        print("################################################################################################################################")

        print("Splitting dataset using cross validation...")
        train, test, X_train, y_train, X_test, y_test, FEATURES, TARGET = cross_validation(df_main)
       
        print("################################################################################################################################")

        print("Running XGBoost model with Random Search Optimization Algorithm for", df_main.columns[0])
        model_rs_result = random_search_optimization(X_train, y_train)
        print("Best parameters : ", model_rs_result.best_params_)
        print("Lowest RMSE : ", (-model_rs_result.best_score_)**(1/2.0))
        print("Optimized learing rate is : ",model_rs_result.best_params_['learning_rate'])
        xgboost_rs_results.append(model_rs_result.best_params_)

        print("################################################################################################################################")

        ###"Getting the highest occuring parameter values..."
        top_param_values = get_common_parameters(xgboost_rs_results)
    
        print("################################################################################################################################")

        ###Combining list of dictionaries into one dictionary
        top_parameters_dict = top_parameters(top_param_values)
    print("Common parameters in a dict: ",top_parameters_dict)
    print("Length of List of XGBoost RS result : ", len(xgboost_rs_results))

    print("################################################################################################################################")

    ### *********Enter the product for which you wish to forecast********
    df_future = pd.DataFrame(df_full['2022-02-04 19:45:00+01:00'])    
     
    ### ***Will pick a random product to forecast on ***
    # df_future = df_full.sample(axis='columns')

    df_future.reset_index(inplace=True)
    print("Product to forecast on is: ",df_future.columns[1])
    print(df_future)

    ### Adding corresponding features to the product selected.
    print("Adding the sys imb vol column with offset of 30 mins.")
    df_future_imb = feature_sys_imb(df_future, df_imb)
    print("Adding the residual column with offset of 60 mins.")
    df_future_imb_rdl = feature_rdl(df_future_imb, df_rdl)
    print("Adding the solar pv protection with no offset.")
    df_future_imb_rdl_spv = feature_spv(df_future_imb_rdl, df_spv)
    print("Adding the wind power with no offset.")
    df_future_imb_rdl_spv_wnd = feature_wnd(df_future_imb_rdl_spv, df_wnd)

    print("################################################################################################################################")
    
    df_future = df_future_imb_rdl_spv_wnd.copy()
    df_future.reset_index(inplace = True)
    df_future.rename(columns={'date':'time_before_fulfilment'},inplace=True)
    print("Future Df of product: ",df_future.columns[0])
    print(df_future)

    print("################################################################################################################################")

    df_future = minutes_to_timestamp(df_future)

    df_future = add_lags(df_future )
    print("Adding lags to dataframe...")
    
    print("################################################################################################################################")

    print("Splitting future dataset using cross validation...")
    train_future, test_future, X_train_future, y_train_future, X_test_future, y_test_future, FEATURES_future, TARGET_future = cross_validation(df_future)
    print("################################################################################################################################")

    print("Applying top optimized parameters to train the XGBoost model...")
    mlflow.xgboost.autolog()
    with mlflow.start_run():
        model_xgb = xgbmodel(X_train_future, y_train_future, X_test_future, y_test_future, top_parameters_dict)
        print("Model applied on: ",df_future.columns[0])
        print("Model: ",model_xgb)

        print("################################################################################################################################")

        print("Forecasting future values: ",df_future.columns[0])
        y_pred = predict_against_test(model_xgb, X_test_future)
        preds.append(y_pred)

        # print("################################################################################################################################")

        test_future['Forecasted Prices'] = y_pred
        df_future = df_future.merge(test_future[['Forecasted Prices']], how = 'left', left_index = True, right_index = True)

        # print("################################################################################################################################")

        ### Setting index from timestamp to t-mins 
        print("Setting index from timestamp to t-mins")
        df_future.index = ((pd.to_datetime(df_future.columns[0]) - 
                                    (df_future.index))
                                        .astype('<m8[m]').astype(int))
        df_future.index = [str(idx) for idx in df_future.index]

        ### Adding a new column 'Real prices', to create plot. Create a new column and populate it with df_main.columns[0] values where 'Forecasted_Prices' are not NaN
        print("Creating new column for making plot.")
        df_future['Real Prices'] = np.nan
        df_future.loc[~df_future['Forecasted Prices'].isna(), 'Real Prices'] = df_future.loc[~df_future['Forecasted Prices'].isna(), df_future.columns[0]]
        print(df_future)

        # print("################################################################################################################################")

        print("Plotting the product graph with features and a graph with forecasted values...")
        plot_product_graph(df_future)
        forecast_plot(df_future)

        eval_metrics(test_future[TARGET_future], test_future['Forecasted Prices'], df_future)

        print("################################################################################################################################")

        print("Calculating feature importance...")

        booster = model_xgb.get_booster()

        ##Get the feature importance scores as a dictionary
        feature_importance_dict = booster.get_score(importance_type='weight')               
        # Convert the dictionary to a list of tuples
        feature_importance_list = [(k, v) for k, v in feature_importance_dict.items()]

        # Log the feature importance values in MLflow
        for feature, importance in feature_importance_dict.items():
            metric_name = "FI_" + feature
            mlflow.log_metric(metric_name, importance)

    mlflow.end_run()

    print("################################################################################################################################")

    print("Execution ends!!!")
    end = time.time()
    print("The time of execution of above program is :",(end-start) / 60, "minutes")
    
if __name__ == "__main__":
    main()