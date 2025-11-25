import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import numpy as np
np.random.seed(42)

def train_val_test_split(X, y, train_size=0.8, val_size=0.1, test_size=0.1, random_state=1234):    
    # Split dataset into Train + tmp
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(1 - train_size), random_state=random_state)
    
    # Split dataset into Test and Validation
    val_test_split = test_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_test_split, random_state=random_state)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def TransDataApp(labels, dic, columns=None):
    data_frames = {
        'X_train': pd.DataFrame(),
        'Y_train': pd.Series(dtype='int'),
        'X_test': pd.DataFrame(),
        'Y_test': pd.Series(dtype='int'),
        'X_val': pd.DataFrame(),
        'Y_val': pd.Series(dtype='int')
    }

    for i, label in enumerate(labels, start=0):
        # print(f"Add {label}")
        # dataFrame = dic[label].reset_index(drop=False)
        dataFrame = dic[label].reset_index(drop=True)
        if columns is not None:
            dataFrame = dataFrame[columns]
        y = pd.Series([i] * dataFrame.shape[0]) # Label generated according to i
        x_train, x_val, x_test, y_train, y_val, y_test = train_val_test_split(dataFrame, y)
        data_frames['X_train'] = pd.concat([data_frames['X_train'], x_train], axis=0)
        data_frames['Y_train'] = pd.concat([data_frames['Y_train'], y_train], axis=0)
        data_frames['X_test'] = pd.concat([data_frames['X_test'], x_test], axis=0)
        data_frames['Y_test'] = pd.concat([data_frames['Y_test'], y_test], axis=0)
        data_frames['X_val'] = pd.concat([data_frames['X_val'], x_val], axis=0)
        data_frames['Y_val'] = pd.concat([data_frames['Y_val'], y_val], axis=0)
    return data_frames['X_train'], data_frames['Y_train'], data_frames['X_test'], data_frames['Y_test'], data_frames['X_val'], data_frames['Y_val']