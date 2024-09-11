import pandas as pd
import numpy as np
import gc


for filetype in [60, 90, 120]:
    data = pd.read_csv(f'./data/predict_{filetype}_train.csv')
    target = data.values[:,-1]
    mean, std = np.mean(data.values, axis=0), np.std(data.values, axis=0)
    data = (data - data.mean())/ data.std()
    data.iloc[:,-1] = target
    data.to_csv(f'./data/predict_{filetype}_train_std.csv', encoding='utf-8', index=None)
    del data
    gc.collect()

    valid = pd.read_csv(f'./data/predict_{filetype}_valid.csv')
    target = valid.values[:,-1]
    valid = (valid - mean) / std 
    valid.iloc[:,-1] = target
    valid.to_csv(f'./data/predict_{filetype}_valid_std.csv', encoding='utf-8', index=None)
    del valid
    gc.collect()
    print(f'finish {filetype}')

test = pd.read_csv(f'./data/predict_{filetype}_test.csv')
target = test.values[:,-1]
test = (test- mean) / std 
test.iloc[:,-1] = target
test.to_csv(f'./data/predict_{filetype}_test_std.csv', encoding='utf-8', index=None)