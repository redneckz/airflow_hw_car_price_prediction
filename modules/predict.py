import sys
from os import environ
from os import listdir
from os.path import join, expanduser
from datetime import datetime
import dill
import json
import pandas as pd

path = environ.get('PROJECT_PATH', '.')
data_path = join(path, 'data')
models_path = join(data_path, 'models')
test_path = join(data_path, 'test')
predictions_path = join(data_path, 'predictions')


airflow_hw_path = expanduser('~/airflow_hw')
sys.path.insert(0, path)


def load_latest_model():
    model_files = listdir(models_path)
    latest_model_file = join(models_path, model_files[-1])
    with open(latest_model_file, 'rb') as file:
       return dill.load(file)


def load_test_df(test_file):
    with open(join(test_path, test_file), 'r') as file:
        return pd.DataFrame.from_dict([json.load(file)])


def load_tests():
    tests = [load_test_df(test_file) for test_file in listdir(test_path)]
    return pd.concat(tests, ignore_index=True)


def dump_prediction(tests_df, pred):
    result = pd.DataFrame(data={ 'car_id': tests_df['id'], 'pred': pred })
    result.to_csv(
        join(predictions_path, f'preds_{datetime.now().strftime("%Y%m%d%H%M")}.csv'),
        index=None)


def predict():
    model = load_latest_model()
    tests_df = load_tests()
    pred = model.predict(tests_df)
    dump_prediction(tests_df, pred)
    

if __name__ == '__main__':
    predict()
