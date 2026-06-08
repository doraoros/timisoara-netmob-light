import pandas as pd
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, f1_score, average_precision_score
from src.config import RANDOM_STATE

def train_regression(df_num: pd.DataFrame):
    y = df_num["dl_mbps"].values
    X = df_num.drop(columns=["dl_mbps"]).values
    model = XGBRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, random_state=RANDOM_STATE
    )
    model.fit(X, y)
    pred = model.predict(X)
    return model, {"mae": mean_absolute_error(y, pred), "rmse": mean_squared_error(y, pred, squared=False)}

def train_classification(df_num: pd.DataFrame):
    y = df_num["ho_next"].astype(int).values
    X = df_num.drop(columns=["ho_next"]).values
    clf = XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, random_state=RANDOM_STATE, eval_metric="logloss"
    )
    clf.fit(X, y)
    p = clf.predict_proba(X)[:,1]
    return clf, {"f1": f1_score(y, (p>=0.5).astype(int)), "pr_auc": average_precision_score(y, p)}
