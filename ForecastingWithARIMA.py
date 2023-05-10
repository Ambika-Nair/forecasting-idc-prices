##Importing Libraries
import os
import pandas as pd
import numpy as np
import time
from datetime import datetime
from matplotlib import pyplot as plt
import pytz
import plotly.express as px
import pmdarima as pm
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from sklearn.model_selection import TimeSeriesSplit
from pmdarima.arima import ADFTest
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from math import sqrt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import mlflow
import mlflow.sklearn
mlflow.set_experiment('Tracking_Forecasting_ARIMA')

def reduce_mem_usage(df):
    """Optimizes the memory usage of a pandas DataFrame by modifying the data type of each column.

    Args:
        df (pd.DataFrame): The DataFrame to optimize.

    Returns:
        pd.DataFrame: The optimized DataFrame.
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

###Accessing csv files from directory
csv_files = []
startdate  = datetime.strptime("2022-05-31 00:00:00", "%Y-%m-%d %H:%M:%S")
enddate = datetime.strptime("2022-06-04 23:45:00", "%Y-%m-%d %H:%M:%S")
path = os.path.realpath(os.path.join(os.getcwd(),os.path.dirname(__file__)))
print("Path is:", path)
for root, dirs, files in os.walk(path):
    for file in files:
        if file.startswith("pri_de_intraday_vwap_last5min_EURmwh_cet_min15_ca_") and file.endswith(".csv"):
            file_date = datetime.strptime(os.path.basename(file), "pri_de_intraday_vwap_last5min_EURmwh_cet_min15_ca_%Y-%m-%d.csv")
            if startdate <= file_date <= enddate:
                csv_files.append(os.path.join(root, file))
csv_files.reverse()

###Reading csv file
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

###Data preprocessing
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

###Flatten columns
def melt_columns(df, chunk_size=100):
    """Melt columns of a given DataFrame into a single column containing the prices

    Args:
        df (pandas DataFrame): Input DataFrame containing columns to be melted
        chunk_size (int, optional): Number of columns to process at a time. Defaults to 100.

    Returns:
        pandas DataFrame: DataFrame with a single column containing the prices
    """
    num_cols = len(df.columns)
    chunks = range(0, num_cols, chunk_size)
    melted_dfs = []
    for i in chunks:
        chunk = df.iloc[:, i:i+chunk_size]
        melted = pd.melt(chunk, value_name='Prices', value_vars=chunk.columns).drop('variable', axis=1)
        melted_dfs.append(melted)
    melted_df = pd.concat(melted_dfs, axis=0)
    return melted_df[['Prices']]

##Visualizing some columns from the main dataframe
def plot_graph(df_flat):
    """Plots the flattened dataframe

    Args:
        df_flat (pandas.DataFrame): The flattened dataframe. 
    """
    fig = px.line(df_flat, x = df_flat.index, y = df_flat['Prices'], title = 'IDC electricity prices')

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

##Running auto-arima on the flattened columns
def autoarima(df_flat):
    """ Fits an AutoARIMA model to the flattened DataFrame.

    Args:
        df_flat (pandas.DataFrame): A DataFrame with a single column of flat time series data.

    Returns:
        list: A list containing the optimal (p, d, q) order values for the AutoARIMA model.
    """
    autoarima_results=[]  
    Auto_ARIMA = pm.auto_arima(df_flat['Prices'], 
                        start_p=1, start_q=1,
                        test='adf',
                        max_p=5, max_q=5,
                        d=1 ,          
                        trace=True,
                        error_action='ignore',  
                        suppress_warnings=True, 
                        n_jobs = -1,
                        stepwise=False)
    autoarima_results.append(Auto_ARIMA.order)
    return autoarima_results

def minutes_to_timestamp(df):
### Replacing minutes before fulfilment column to timestamp 
    df['time_before_fulfilment'] = (pd.to_timedelta(df['time_before_fulfilment'].astype(int), unit = "min")
                    .rsub(pd.to_datetime(df.columns[1]))
                    )
    df.set_index('time_before_fulfilment', inplace=True)
    return df

def cross_validation(df):
    """Split time series data into training and testing sets using time series split.

    Args:
        df (pandas.DataFrame):  The input DataFrame is the one that was modified in function 'minutes_to_timestamp'.

    Returns:
       tuple: A tuple containing training and testing sets as pandas.DataFrames.
    """
    tss = TimeSeriesSplit(n_splits = 6)
    df = df.sort_index()
    for train_idx, val_idx in tss.split(df):
        train_data = df.iloc[train_idx]
        test_data = df.iloc[val_idx]

    return train_data, test_data

def train_model(train_data, top_order):
    """Train an ARIMA model on the training data with the given order.

    Args:
        train_data (pandas.DataFrame): The training data to use for training the model.
        order (tuple): A tuple containing the order (p, d, q) of the ARIMA model to train.

    Returns:
        statsmodels.tsa.arima_model.ARMAResultsWrapper: The fitted ARIMA model.
    """
    order = top_order
    model = ARIMA(train_data, order = order)
    model.initialize_approximate_diffuse()
    fitted_model = model.fit()
    warnings.simplefilter('ignore', ConvergenceWarning)
    print(fitted_model.summary())   
    mlflow.log_param("P_D_Q Orders", order)
    return fitted_model

def predict_prices(train_data, test_data, fitted_model):
    """Predicts prices using the trained ARIMA model.

    Args:
        train_data (pandas.DataFrame): DataFrame containing the training data.
        test_data (pandas.DataFrame): DataFrame containing the test data.
        fitted_model (statsmodels.tsa.arima.model.ARIMAResultsWrapper): Trained ARIMA model.

    Returns:
        numpy.ndarray: Array containing the predicted prices.
    """
    start = len(train_data)
    end = len(train_data) + len(test_data) -1
    y_pred = fitted_model.predict(start = start, end = end, typ = 'levels')
    return y_pred

def forecast_plot(df, order):

    fig = px.line(df, x = df.index, y = [df.columns[0],'Forecasted Prices', 'Real Prices'], markers='.',
                  title = f'IDC electricity market price forecast for: {df.columns[0]} ARIMA orders - {order}'
                )   
    fig.update_xaxes(
        title = 't-minutes before fulfillment',
        tickangle = -90,
        rangeselector=dict(
            buttons=list([
                dict(count = 1, label = "1H", step = "hour", stepmode ="backward"),
                dict(step="all")
            ])
        )
    )
    fig.update_yaxes(title="IDC Price VWAP5Minutes [€/MWh]")
    fig.update_traces(
        name="Past Prices", # changes the name of the first y-axis attribute
        selector=dict(name=df.columns[0])
    )

    fig.update_layout(legend_title_text="")
    fig.show()

## Verifying performance of the model by calculating Mean absolute error and root mean squared error. 
def eval_metrics(actual, pred, df):
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
    ###start clocking time
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

    #Transposing the dataframe.
    df_transposed = df_re.rename_axis(index=None, columns='date').T
    print("Transposed Data")
    print(df_transposed)

    print("################################################################################################################################")

    df_full = df_transposed.copy()
    reduce_mem_usage(df_full)
    print("DF Full")
    print(df_full)

    print("################################################################################################################################")

    #### Flattening columns
    print("Flattening the columns...")
    melted_df = melt_columns(df_full, chunk_size=100)
    print(melted_df) 

    print("################################################################################################################################")

    ## Running auto arima to get suggested p,d,q values
    print("Running Auto Arima... ")
    autoarima_results = autoarima(melted_df)

    print("################################################################################################################################")

    top_order = autoarima_results[0]
    print("P,D,Q values are:", top_order) 

    print("################################################################################################################################")

    ### *********Enter the product for which you wish to forecast********
    # df_product = pd.DataFrame(df_full['2022-06-02 08:15:00+02:00'])

    ### ***Will pick a random product to forecast on ***
    df_product = df_full.sample(axis='columns')

    df_product.reset_index(inplace=True)
    df_product.rename(columns={'date':'time_before_fulfilment'},inplace=True)
    print("Product to forecast on is: ",df_product.columns[1])
    print(df_product)

    print("Converting minutes to timestamp...")
    df_product = minutes_to_timestamp(df_product)
    print("DF_Product", df_product)

    print("################################################################################################################################")

    print("Splitting dataset using cross validation...")
    train_data, test_data = cross_validation(df_product)

    print("################################################################################################################################")

    print("Fitting the model...")
    mlflow.statsmodels.autolog()
    with mlflow.start_run():
        try:
            arimamodel_fit = train_model(train_data,top_order)

            print("################################################################################################################################")

            print("Forecasting future values: ",df_product.columns[0])
            y_pred = predict_prices(train_data, test_data, arimamodel_fit)

            test_data['Forecasted Prices'] = y_pred
            df_product = df_product.merge(test_data[['Forecasted Prices']], how = 'left', left_index = True, right_index = True)
            print(df_product)

            print("################################################################################################################################")

            ### Setting index from timestamp to t-mins 
            print("Setting index from timestamp to t-mins")
            df_product.index = ((pd.to_datetime(df_product.columns[0]) - 
                                    (df_product.index))
                                        .astype('<m8[m]').astype(int))
            df_product.index = [str(idx) for idx in df_product.index]

            print(df_product)

            print("################################################################################################################################")

            ### Adding a new column 'Real prices', to create plot. Create a new column and populate it with df_main.columns[0] values where 'Forecasted_Prices' are not NaN
            print("Creating new column for making plot.")
            df_product['Real Prices'] = np.nan
            df_product.loc[~df_product['Forecasted Prices'].isna(), 'Real Prices'] = df_product.loc[~df_product['Forecasted Prices'].isna(), df_product.columns[0]] 
            print(df_product)

            ###Converting the column name from type timestamp to str
            timestamp = pd.Timestamp(df_product.columns[0])
            date_string = timestamp.strftime('%Y-%m-%d %H:%M:%S%z')
            df_product = df_product.rename(columns={df_product.columns[0]: date_string})

            print("################################################################################################################################")

            print("Plotting the product graph with features and a graph with forecasted values...")
            forecast_plot(df_product, top_order)

            print("################################################################################################################################")

            #Getting eval metrics
            print("Evaluation metrics of the forecast is:")
            eval_metrics(test_data.iloc[:,:1], test_data['Forecasted Prices'], df_product)
        finally:
            mlflow.end_run()

    ("################################################################################################################################")
   
    print("**Execution Ends!**")
    end = time.time()
    print("The time of execution of above program is :",(end-start) / 60, "minutes")

if __name__ == "__main__":
    main()