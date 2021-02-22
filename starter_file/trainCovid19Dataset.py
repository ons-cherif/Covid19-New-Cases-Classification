import argparse
import os
import numpy as np
import pandas as pd
from shutil import copyfile

from zipfile import ZipFile
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory
#from azureml.core import Dataset, Datastore
#from azureml.data.datapath import DataPath

# Create .kaggledirectory to store kaggle's Api token
os.mkdir("~/.kaggle") 
copyfile("~/sandbox/sandbox_env/Udacity/kaggle.json", "~/.kaggle/")
# cat ~/.kaggle/kaggle.json 
os.chmod("~/.kaggle/kaggle.json", 600)

# Download dataset from kaggle
# ~/.local/bin/kaggle datasets download -d gpreda/covid-world-vaccination-progress -p ./starter_file/kaggle/
import kaggle
kaggle.api.authenticate()
kaggle.api.dataset_download_files('gpreda/covid-world-vaccination-progress', path='./starter_file/kaggle/', unzip=True)

with ZipFile('./starter_file/kaggle/covid-world-vaccination-progress.zip', 'r') as datasetZip:
   # Extract all the contents of zip file in current directory
   datasetZip.extractall('./starter_file/kaggle')
    
#ds = TabularDatasetFactory.from_delimited_files(path=datastore_path, infer_column_types=True, separator=',', header=True, encoding='utf8')
run = Run.get_context()  

data = pd.read_csv("./starter_file/kaggle/country_vaccinations.csv").dropna()

def clean_data(data):
    # Clean and one hot encode data
    x_df = data
    # take the latest number of vaccinated people by country
    x_df['used_Vaccine'] = np.where(x_df.groupby('country')['total_vaccinations'].transform('max') > 0, True, False)
    y_df = x_df.pop("used_Vaccine").apply(lambda s: 1 if s == True else 0)
    
    countries = pd.get_dummies(x_df.country, prefix="country")
    iso_codes = pd.get_dummies(x_df.iso_code, prefix="iso_code")
    vaccines = pd.get_dummies(x_df.vaccines, prefix="vaccine")
    source_names = pd.get_dummies(x_df.source_name, prefix="source")
    #source_websites = pd.get_dummies(x_df.source_name, prefix="source_website")
    x_df.drop(["country","iso_code","vaccines","source_name","source_website"], inplace=True, axis=1)
    x_df = x_df.join([countries,iso_codes,vaccines,source_names])
    
    x_df['month']= pd.DatetimeIndex(x_df['date']).month
    x_df['date']=pd.to_datetime(x_df['date'], format='%Y-%m-%d')
    x_df['date']= pd.DatetimeIndex(x_df['date']).year
    
    return x_df,y_df
    
def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()
    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")
    args = parser.parse_args()  
   
    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))
    
    x, y = clean_data(data)

    # TODO: Split data into train and test sets.
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state = 42,shuffle=True)
    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
   
    os.makedirs('./outputs', exist_ok=True)
    joblib.dump(model,'./outputs/model.joblib')
    run.log("Accuracy", np.float(accuracy))

if __name__ == '__main__':
    main()
