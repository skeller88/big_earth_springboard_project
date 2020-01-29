from typing import List

import numpy as np
import pandas as pd


def balanced_class_train_test_splits(*dataframes: List[pd.DataFrame]):
    train = []
    valid = []
    test = []

    train_size = .8
    for dataframe in dataframes:
        random_mask = np.random.rand(len(dataframe))
        train_mask = random_mask < train_size
        valid_mask = (train_size <= random_mask) & (random_mask < .9)
        test_mask = random_mask >= .9

        train.append(dataframe[train_mask])
        valid.append(dataframe[valid_mask])
        test.append(dataframe[test_mask])

    train = pd.concat(train)
    valid = pd.concat(valid)
    test = pd.concat(test)

    print("len(train)", len(train), "len(valid)", len(valid), "len(test)", len(test))
    return train, valid, test