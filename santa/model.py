from functools import partial

import copy
from hyperopt import hp, tpe
from hyperopt.fmin import fmin
import lightgbm as lgb
from sklearn.model_selection import train_test_split

HP_NAMES = ("min_data_in_leaf", "num_leaves", "feature_fraction")

DEFAULT_HP = dict(
    objective="binary",
    learning_rate=0.2,
    min_data_in_leaf=1,
    num_leaves=int(3),
    bagging_freq=1,
    bagging_fraction=0.9,
    feature_fraction=0.1,
    metric="auc",
    feature_pre_filter=False
)


def compute_tts(X, y):
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, stratify=y, test_size=0.2)
    dtrain = lgb.Dataset(X_tr.values, y_tr.values)
    dvalid = lgb.Dataset(X_val.values, y_val.values)
    return dtrain, dvalid


def fit(hyperparams, dtrain, dvalid):

    gbm = lgb.train(
        params=hyperparams,
        train_set=dtrain,
        num_boost_round=5000,
        valid_sets=(dtrain, dvalid),
        early_stopping_rounds=1000,
        verbose_eval=50,
    )
    return gbm


def _hp_objective(hyperparams, dtrain, dvalid):
    params = copy.deepcopy(DEFAULT_HP)
    for p, v in hyperparams.items():
        if p in {"num_leaves", "min_data_in_leaf"}:
            v = int(v)
        params[p] = v
    params["verbose"] = -1

    gbm = lgb.train(
        params=params,
        train_set=dtrain,
        num_boost_round=5000,
        valid_sets=(dtrain, dvalid),
        early_stopping_rounds=30,
        verbose_eval=0,
    )
    return -gbm.best_score["valid_1"]["auc"]


def optimize_hyperparams(dtrain, dvalid, max_evals=50):
    hp_objective = partial(_hp_objective, dtrain=dtrain, dvalid=dvalid)
    param_space = {
        "min_data_in_leaf": hp.quniform("min_data_in_leaf", 1, 500, 5),
        "num_leaves": hp.quniform("num_leaves", 2, 32, 1),
        "feature_fraction": hp.quniform("feature_fraction", 0.05, 1.0, 0.01),
        "bagging_fraction": hp.quniform("bagging_fraction", 0.05, 1.0, 0.01)
    }
    best = fmin(
        fn=hp_objective,
        space=param_space,
        algo=tpe.suggest,
        max_evals=max_evals
    )
    return best
