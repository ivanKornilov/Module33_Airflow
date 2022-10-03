import dill
import pandas as pd
import os
from datetime import datetime
import json
from os import listdir
from os.path import isfile, join
import logging


path = os.environ.get('PROJECT_PATH', '.')


def version():
    return model['metadata']


def predict(model):
    data = []
    mypath = f'{path}/data/test/'
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
        result_of_prediction.append(y[0])
    result_df = pd.DataFrame(
        {'car_id': onlyfiles_ids,
         'pred': result_of_prediction
         })
    preds_filename = f'{path}/data/predictions/preds_{datetime.now().strftime("%Y%m%d%H%M")}.csv'
    result_df.to_csv(preds_filename, index=False)
    logging.info(f'Predictions are saved as {preds_filename}')

if __name__ == '__main__':
    latest_model = sorted(os.listdir(f'{path}/data/models'))[-1]
    with open(f'{path}/data/models/{latest_model}', 'rb') as file:
        model = dill.load(file)
    predict(model)


