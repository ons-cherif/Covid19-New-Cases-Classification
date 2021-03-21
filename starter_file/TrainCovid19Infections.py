import argparse

import numpy as np
import pandas as pd
import shutil
import joblib
import datetime as dt
import os

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory

#original_path = 'https://covid.ourworldindata.org/data/owid-covid-data.csv'
#ds = TabularDatasetFactory.from_delimited_files(original_path, infer_column_types=True, separator=',', header=True)

run = Run.get_context()  
x_df = pd.read_csv("github/owid-covid-data.csv").fillna(0)

def clean_data(x_df):
    # take the latest number of vaccinated people by country
    x_df['tests_units'] = np.where(x_df.tests_units == 'tests performed', True, False)
    x_df['tests_units'] = x_df.pop("tests_units").apply(lambda s: 1 if s == True else 0)
  
    iso_codes = pd.get_dummies(x_df.iso_code, prefix="iso_code")
    continent = pd.get_dummies(x_df.continent, prefix="continent")
    x_df.drop([
        "location",
        "iso_code",
        "continent",
        "hosp_patients_per_million",
        "weekly_icu_admissions",
        "weekly_icu_admissions_per_million",
        "weekly_hosp_admissions",
        "weekly_hosp_admissions_per_million",
        "new_tests_smoothed_per_thousand",
        "new_tests_smoothed","new_tests_per_thousand",
        "people_vaccinated",
        "people_fully_vaccinated",
        "new_vaccinations",
        "new_vaccinations_smoothed",
        "total_vaccinations_per_hundred",
        "people_vaccinated_per_hundred",
        "people_fully_vaccinated_per_hundred",
        "new_vaccinations_smoothed_per_million",
        "total_vaccinations"],
         inplace=True, axis=1)
    x_df = x_df.join([iso_codes,continent])
    
    x_df['date']=pd.to_datetime(x_df['date'])
    x_df['date']=x_df['date'].map(dt.datetime.toordinal)
   
    #x_df['total_vaccinations'] =  x_df['total_vaccinations'].astype(float)
    
    y_df = x_df.pop("new_cases").apply(lambda s: 1 if s > 1 else 0)
    #y_df = x_df['new_cases']
    return x_df,y_df
    
def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()
    parser.add_argument('--C', type=float, default=0.5, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=50, help="Maximum number of iterations to converge")
    args = parser.parse_args()  
   
    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))
    
    x, y = clean_data(x_df)


    # TODO: Split data into train and test sets.
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.6, random_state = 42,shuffle=True)
    
    model = LogisticRegression(C=args.C,max_iter=args.max_iter,multi_class='ovr').fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
   
    os.makedirs('./outputs', exist_ok=True)
    joblib.dump(model,'./outputs/model.joblib')
    run.log("Accuracy", np.float(accuracy))

if __name__ == '__main__':
    main()
