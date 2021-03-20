import json
import numpy as np
import os
import joblib
import pandas as pd

def init():
    #This function initialises the model. The model file is retrieved used within the script.
    global model
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'automl_covid19_model.pkl')
    print("Found model:", os.path.isfile(model_path))
    model = joblib.load(model_path)

def run(data):
    try:
        data = np.array(json.loads(data))
        #data = np.array(json.loads(data))
        #data = json.loads(data)['data'] # raw = pd.read_json(data) 
        #data = pd.DataFrame.from_dict(data)
        result = model.predict(data)
        # You can return any data type, as long as it is JSON serializable.
        return result.tolist()
    except Exception as e:
        error = str(e)
        return error