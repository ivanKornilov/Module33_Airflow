import dill
import pandas as pd
from pydantic import BaseModel
import os
from datetime import datetime
import json
import numpy as np
from os import listdir
from os.path import isfile, join

# Укажем путь к файлам проекта:
# -> $PROJECT_PATH при запуске в Airflow
# -> иначе - текущая директория при локальном запуске
path = os.environ.get('PROJECT_PATH', '.')

class Form(BaseModel):
    description: str
    fuel: str
    id: int
    image_url: str
    lat: float
    long: float
    manufacturer: str
    model: str
    odometer: int
    posting_date: str
    price: int
    region: str
    region_url: str
    state: str
    title_status: str
    transmission: str
    url: str
    year: int


class Prediction(BaseModel):
    id: int
    price_category: str
    price: int


def version():
    return model['metadata']


def predict(model):
    data = []
    mypath = 'C://Users//user//airflow_hw//modules//data//test//'
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    for _ in range(len(onlyfiles)):
        with open(f'{mypath}{onlyfiles[_]}') as file1:
            content = json.load(file1)
        data.append(content)
    df = pd.DataFrame.from_dict(data, orient='columns')
    onlyfiles_ids = []
    for _ in range(len(onlyfiles)):
        onlyfiles_ids.append(onlyfiles[_].split('.')[0])
    result_of_prediction = []
    for _ in range(len(onlyfiles)):
        y = model['model'].predict(df.loc[_:_])
        print(y[0])
        result_of_prediction.append(y[0])
    result_df = pd.DataFrame(
        {'car_id': onlyfiles_ids,
         'pred': result_of_prediction
         })
    result_df.to_csv('resuts_prediction.csv', sep='\t')


if __name__ == '__main__':
    model_filename = f'{path}/data/models/cars_pipe_{datetime.now().strftime("%Y%m%d%H")}.pkl'
    with open(model_filename, 'rb') as file:
        model = dill.load(file)
    predict(model)


