# Forecasting short-term price movements on the German intraday continuous electricity market.


## Points to know about this project:
1. MLFlow - This project uses MLFlow to track machine learning experiments and monitor performance. MLflow is an open-source platform that helps data scientists and machine learning engineers manage their machine learning lifecycle. It provides a suite of tools to help with tasks such as tracking experiments, packaging and deploying models, and monitoring performance.
2. Two models have been implemented to generate forecasts- ARIMA (Auto Regressive Integrated Moving Average) and XGBoost (eXtreme Gradient Boosting)

## To begin with :
1. Install the packages mentioned in requirements.txt file with 
```bash
pip install -r requirements.txt
```

2. Once the packages are installed, you want to download the relevant data files needed from AWS S3 bucket. To do that, open ```download_wattsight_from_s3.py``` file and change the year for which you wish the files are to be fetched then run the file. (Not on git for privacy reasons)
3. All the files will be downloaded and stored in your local folder where the code resides. The files are stored in a folder named 'data'. 

## To generate forecast using ARIMA and XGBoost. 
1. The script that generates forecast using ARIMA is named as ```ForecastingWithARIMA.py```
2. The script that generates forecast using ARIMA is named as ```ForecastingWithXGBoost.py```
3. The next steps are common to both the scripts:
4. If you wish to log the run in MLflow to track its various metrics, open terminal and run as below. This starts a local server that hosts the MLflow UI. You can then access the UI by opening a web browser and navigating to the URL displayed in the terminal (typically http://localhost:5000)
```bash
mlflow ui
```
5. The script has 2 instances where the user has to make changes as required. 
    1. The user has to enter the start and end timestamps of their choice, based on which the files will be accessed from the 'data' folder for the implementation to          start. The block of code that takes teh timestamps looks like below:
    ```bash
    ###Accessing csv files from directory
    csv_files = []
    startdate  = datetime.strptime("2022-01-01 00:00:00", "%Y-%m-%d %H:%M:%S")
    enddate = datetime.strptime("2022-01-10 23:45:00", "%Y-%m-%d %H:%M:%S")
    ```
    2. The user has the option to let the script choose a random product from the dataframe to forecast upon or can enter a product of their choice. Note that the            product entered should fall within the range of the timestamps as entered in the point above. 
       The block of code where the product of choice can be entered is as follows:
      ```bash
      ### *********Enter the product for which you wish to forecast********
        df_future = pd.DataFrame(df_full['2022-02-01 19:45:00+01:00']) 
      ```
6. Once that is done, the script can be run. 
