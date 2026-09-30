# (0) integrations
import os, io, time, requests
import numpy as np, pandas as pd

from dotenv import load_dotenv
load_dotenv()

import matplotlib.pyplot as plt
plt.style.use("dark_background")
plt.rcParams.update({
    "figure.figsize": (12, 8),
    "font.size": 12,
    "axes.grid": True,
    "grid.alpha": 0.2
})

import tabpfn_client
from tabpfn_client import TabPFNRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from tqdm.notebook import tqdm

API_TOKEN = os.environ["TABPFN_TOKEN"]
tabpfn_client.set_access_token(API_TOKEN)


# (1) load data
CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph/csv?id=HOUSTNSA"
raw_csv = requests.get(CSV_URL, timeout=15).text

df = (
    pd.read_csv(io.StringIO(raw_csv), parse_dates=["observation_date"])
        .rename(columns={"observation_date": "date", "HOUSTNSA": "value"})
        .set_index("date")
        .sort_index()
)

# (2) feature engineering
df = df.loc["2022-01-01":]

df["month"]     = df.index.month
df["run_idx"]   = np.arange(len(df)) / len(df)

# (3) train test split
HORIZON     = 24
train, test = df.iloc[:-HORIZON], df.iloc[-HORIZON:]
X_tr, y_tr = train.drop(columns="value"), train["value"]
X_te, y_te = test.drop(columns="value"), test["value"]

# (4) model creation

## (4.1) tabpfn
tab = TabPFNRegressor(model_path="v3.5_default", n_estimators=8, random_state=67)
tab.fit(X_tr.values, y_tr.values)
y_tab = tab.predict(X_te.values, output_type="median")
q10, q90 = tab.predict(X_te.values, output_type="quantiles", quantiles=[0.1,0.9])

## (4.2) xgboost
xgb = XGBRegressor()
xgb.fit(X_tr, y_tr)
y_xgb = xgb.predict(X_te)

# (5) mae comparison
print(f"TabPFN  MAE : {mean_absolute_error(y_te, y_tab):.3f}")
print(f"XGBoost  MAE : {mean_absolute_error(y_te, y_xgb):.3f}")

# (6) chart comparison
plt.plot(df.index, df.value, label="History")
plt.plot(y_te.index, y_tab, "o-", label="TabPFN Median")
plt.fill_between(y_te.index, q10, q90, alpha=0.25, label="TabPFN 80 % PI")
plt.plot(y_te.index, y_xgb, "s--", label="XGBoost point-forecast")
plt.title("Housing Starts (HOUSTNSA) – TabPFN vs XGBoost")
plt.ylabel("1e3 Units")
plt.legend(); plt.tight_layout()
plt.savefig("output.png")
