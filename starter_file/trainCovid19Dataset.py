import os
#os.environ['KAGGLE_USERNAME']= 'ocherif'
#os.environ['KAGGLE_KEY']= 'cb037f99cae382b7a67c68f8048e01be'
#import kaggle
import argparse

import numpy as np
import pandas as pd
import shutil
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from azureml.core.run import Run

#kaggle.api.authenticate()
#kaggle.api.dataset_download_files('gpreda/covid-world-vaccination-progress', path='kaggle/', unzip=True)
    
run = Run.get_context()  

x_df = pd.read_csv("kaggle/country_vaccinations.csv")

def clean_data(x_df):
    # Clean and one hot encode data
    x_df.drop(["people_fully_vaccinated_per_hundred"], inplace=True, axis=1)
    x_df = x_df.fillna(0)
    # take the latest number of vaccinated people by country
    x_df['used_Vaccine'] = np.where(x_df.people_fully_vaccinated > 0, True, False)
    y_df = x_df.pop("used_Vaccine").apply(lambda s: 1 if s == True else 0)
    
    #countries = pd.get_dummies(x_df.country, prefix="country")
    iso_codes = pd.get_dummies(x_df.iso_code, prefix="iso_code")
    vaccines = pd.get_dummies(x_df.vaccines, prefix="vaccines")
    source_names = pd.get_dummies(x_df.source_name, prefix="source")
    x_df.drop(["country","iso_code","vaccines","source_name","source_website"], inplace=True, axis=1)
    x_df = x_df.join([iso_codes,vaccines,source_names])
    
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
    
    x, y = clean_data(x_df)

    # TODO: Split data into train and test sets.
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state = 42,shuffle=True)
    #x_train.to_csv(path_or_buf='x_trainToCSV.csv',header = True, encoding='UTF8',index=False)
    #y_train.to_csv(path_or_buf='y_trainToCSV.csv',header = True, encoding='UTF8',index=False)
    
    model = LogisticRegression(C=args.C,max_iter=args.max_iter,multi_class='ovr').fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
   
    os.makedirs('./outputs', exist_ok=True)
    joblib.dump(model,'./outputs/model.joblib')
    run.log("Accuracy", np.float(accuracy))

if __name__ == '__main__':
    main()
