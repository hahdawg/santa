from os.path import join
from pathlib import Path

import dask
import dask.dataframe as dd
import pandas as pd
from tqdm import tqdm

SRC_DIR = Path(__file__).parent.absolute()
ROOT_DIR = Path(SRC_DIR).parent
DATA_DIR = join(ROOT_DIR, "data")
PROC_DATA_PATH = join(DATA_DIR, "proc.h5")


def load_raw_train():
    path = join(DATA_DIR, "train.csv")
    return pd.read_csv(path)


class CategoryEncoder:

    def __init__(self, columns):
        self.columns = columns
        self.encode_name = "encoded"
        self.mapping = None

    def _fit_column(self, x, y, c):
        df = pd.concat([x, y], axis=1)
        grouped = df.groupby(x.name)["y"].agg(["mean", "count"])
        unique_idx = grouped["count"] == 1
        grouped[self.encode_name] = 0

        unique = grouped.loc[unique_idx == 1, self.encode_name].copy()

        repeated = grouped.loc[~unique_idx].copy()
        positive = repeated["mean"] == 1
        negative = repeated["mean"] == 0
        mixed = ~(positive | negative)
        repeated = repeated[self.encode_name]
        repeated.loc[negative] = 1
        repeated.loc[mixed] = 2
        repeated.loc[positive] = 3

        res = pd.concat((unique, repeated), axis=0).sort_index()
        res.name = f"{c}_encoded"
        return res

    def _transform_column(self, x, c):
        x = x.map(self.mapping[c])
        x.name = f"{c}_encoded"
        return x

    def fit(self, X, y):
        X = dask.delayed(X)
        mapping = {
            c: dask.delayed(self._fit_column)(X[c], y, c)
            for c in self.columns
        }
        self.mapping = dask.compute(mapping, scheduler="threads")[0]

    def transform(self, X):
        X = dask.delayed(X)
        encoded = [
            dask.delayed(self._transform_column)(X[c], c)
            for c in self.columns
        ]
        encoded = dask.compute(encoded, scheduler="threads")[0]
        encoded = pd.concat(encoded, axis=1)
        X = pd.concat((X.compute(), encoded), axis=1)
        return X

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)


def process_data(df):
    df.columns = df.columns.str.lower()
    df = df.drop("id_code", axis=1)
    df = df.rename(columns={"target": "y"})

    X = df.drop(["y"], axis=1)
    cols_orig = X.columns
    for c in tqdm(cols_orig):
        X[f"{c}_count"] = X[c].groupby(X[c]).transform("count")
    dmatrix = X.copy()
    y = df["y"]
    dmatrix["y"] = y
    dmatrix.to_hdf(PROC_DATA_PATH, "df")


def load_processed_data():
    df = pd.read_hdf(PROC_DATA_PATH)
    X = df.drop("y", axis=1)
    y = df["y"]
    return X, y
