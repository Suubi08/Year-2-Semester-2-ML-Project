"""
PM2.5 Air Quality Prediction - Makerere University
Full pipeline: Cleaning → Preprocessing → Feature Engineering →
Models: Linear Regression, Random Forest (tuned), Gradient Boosting, ARIMA, LSTM
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 65)
print("  PM2.5 AIR QUALITY PREDICTION - MAKERERE UNIVERSITY")
print("=" * 65)

# ─────────────────────────────────────────────────────────────────
# 1. LOAD & CLEAN DATA
# ─────────────────────────────────────────────────────────────────
print("\n[1] DATA LOADING & CLEANING")
print("-" * 40)

df_raw = pd.read_csv('/mnt/user-data/uploads/data.csv')
print(f"Raw dataset shape: {df_raw.shape}")

# Filter to Station 94 only
df = df_raw[df_raw['site_name'].str.contains('94', na=False)].copy()
print(f"Station 94 rows: {len(df)}")

# Parse datetime with UTC
df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
df = df.sort_values('datetime').reset_index(drop=True)

# Replace zero temperatures and humidity with NaN (sensor errors)
zero_temp = (df['temperature'] == 0).sum()
zero_hum  = (df['humidity'] == 0).sum()
df['temperature'] = df['temperature'].replace(0, np.nan)
df['humidity']    = df['humidity'].replace(0, np.nan)
print(f"Zero temperatures replaced with NaN: {zero_temp}")
print(f"Zero humidity values replaced with NaN: {zero_hum}")

# Drop rows with missing PM2.5 (target variable)
before = len(df)
df = df.dropna(subset=['pm2_5_calibrated_value'])
print(f"Rows dropped (missing PM2.5): {before - len(df)}")

# ─────────────────────────────────────────────────────────────────
# 2. PREPROCESSING — Hourly aggregation
# ─────────────────────────────────────────────────────────────────
print("\n[2] PREPROCESSING")
print("-" * 40)

df = df.set_index('datetime')
df_hourly = df[['pm2_5_calibrated_value', 'temperature', 'humidity']].resample('h').mean()
print(f"Rows before hourly aggregation: {len(df)}")
print(f"Rows after hourly aggregation:  {len(df_hourly)}")

# Forward-fill small gaps in temperature and humidity (sensor drop-outs)
df_hourly['temperature'] = df_hourly['temperature'].ffill(limit=3)
df_hourly['humidity']    = df_hourly['humidity'].ffill(limit=3)

# Drop remaining NaN rows
df_hourly = df_hourly.dropna()
print(f"Rows after dropping residual NaNs: {len(df_hourly)}")
print(f"Date range: {df_hourly.index.min().date()} → {df_hourly.index.max().date()}")
print(f"PM2.5 stats → Mean: {df_hourly['pm2_5_calibrated_value'].mean():.2f}  "
      f"Std: {df_hourly['pm2_5_calibrated_value'].std():.2f}  "
      f"Min: {df_hourly['pm2_5_calibrated_value'].min():.2f}  "
      f"Max: {df_hourly['pm2_5_calibrated_value'].max():.2f}")

# ─────────────────────────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────
print("\n[3] FEATURE ENGINEERING")
print("-" * 40)

df_feat = df_hourly.copy()

# Temporal features
df_feat['hour']       = df_feat.index.hour
df_feat['day_of_week']= df_feat.index.dayofweek

# Lag feature: PM2.5 one hour ago
df_feat['pm25_lag_1h']= df_feat['pm2_5_calibrated_value'].shift(1)

# Rolling averages
df_feat['pm25_rolling_3h'] = df_feat['pm2_5_calibrated_value'].shift(1).rolling(3).mean()
df_feat['pm25_rolling_12h']= df_feat['pm2_5_calibrated_value'].shift(1).rolling(12).mean()

# Interaction feature
df_feat['temp_hum_interaction'] = df_feat['temperature'] * df_feat['humidity']

# Drop NaNs introduced by lag/rolling
df_feat = df_feat.dropna()
print(f"Final feature matrix shape: {df_feat.shape}")
print(f"Features: {[c for c in df_feat.columns if c != 'pm2_5_calibrated_value']}")

# ─────────────────────────────────────────────────────────────────
# 4. TRAIN / TEST SPLIT (chronological — no shuffling)
# ─────────────────────────────────────────────────────────────────
print("\n[4] TRAIN/TEST SPLIT (chronological 80/20)")
print("-" * 40)

FEATURE_COLS = ['humidity', 'temperature', 'hour', 'day_of_week',
                'pm25_lag_1h', 'temp_hum_interaction',
                'pm25_rolling_3h', 'pm25_rolling_12h']
TARGET = 'pm2_5_calibrated_value'

X = df_feat[FEATURE_COLS].values
y = df_feat[TARGET].values

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
print(f"Training samples: {len(X_train)}  |  Test samples: {len(X_test)}")

# ─────────────────────────────────────────────────────────────────
# HELPER: metrics
# ─────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred, name):
    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    r2   = r2_score(y_true, y_pred)
    print(f"\n  {'Metric':<8} {'Value':>10}")
    print(f"  {'MAE':<8} {mae:>10.4f} ug/m3")
    print(f"  {'RMSE':<8} {rmse:>10.4f} ug/m3")
    print(f"  {'MSE':<8} {mse:>10.4f}")
    print(f"  {'MAPE':<8} {mape:>10.2f} %")
    print(f"  {'R2':<8} {r2:>10.4f}")
    return {'Model': name, 'MAE': round(mae,4), 'RMSE': round(rmse,4),
            'MSE': round(mse,4), 'MAPE': round(mape,2), 'R2': round(r2,4)}

results = []

# ─────────────────────────────────────────────────────────────────
# 5. MODEL 1: LINEAR REGRESSION (baseline)
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  MODEL 1: LINEAR REGRESSION (Baseline)")
print("=" * 65)

lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
results.append(compute_metrics(y_test, lr_pred, 'Linear Regression'))

# ─────────────────────────────────────────────────────────────────
# 6. MODEL 2: RANDOM FOREST + HYPERPARAMETER TUNING
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  MODEL 2: RANDOM FOREST + HYPERPARAMETER OPTIMIZATION")
print("=" * 65)

print("\n  Running GridSearchCV (TimeSeriesSplit, 3 folds)...")
tscv = TimeSeriesSplit(n_splits=3)
rf_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'max_features': ['sqrt', 0.5]
}
rf_base = RandomForestRegressor(random_state=42, n_jobs=-1)
rf_gs = GridSearchCV(rf_base, rf_param_grid, cv=tscv,
                     scoring='neg_mean_absolute_error', n_jobs=-1, verbose=0)
rf_gs.fit(X_train, y_train)
print(f"  Best parameters: {rf_gs.best_params_}")
print(f"  Best CV MAE: {-rf_gs.best_score_:.4f}")

rf_best = rf_gs.best_estimator_
rf_pred = rf_best.predict(X_test)
results.append(compute_metrics(y_test, rf_pred, 'Random Forest (Tuned)'))

# Feature importances
feat_imp = pd.Series(rf_best.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
print(f"\n  Feature importances:")
for f, v in feat_imp.items():
    bar = '█' * int(v * 40)
    print(f"  {f:<28} {bar} {v:.3f}")

# ─────────────────────────────────────────────────────────────────
# 7. MODEL 3: GRADIENT BOOSTING + HYPERPARAMETER TUNING
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  MODEL 3: GRADIENT BOOSTING + HYPERPARAMETER OPTIMIZATION")
print("=" * 65)

print("\n  Running GridSearchCV (TimeSeriesSplit, 3 folds)...")
gb_param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5],
    'subsample': [0.8, 1.0]
}
gb_base = GradientBoostingRegressor(random_state=42)
gb_gs = GridSearchCV(gb_base, gb_param_grid, cv=tscv,
                     scoring='neg_mean_absolute_error', n_jobs=-1, verbose=0)
gb_gs.fit(X_train, y_train)
print(f"  Best parameters: {gb_gs.best_params_}")
print(f"  Best CV MAE: {-gb_gs.best_score_:.4f}")

gb_best = gb_gs.best_estimator_
gb_pred = gb_best.predict(X_test)
results.append(compute_metrics(y_test, gb_pred, 'Gradient Boosting (Tuned)'))

# ─────────────────────────────────────────────────────────────────
# 8. MODEL 4: ARIMA (implemented with numpy/scipy)
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  MODEL 4: ARIMA (p=2, d=1, q=2)")
print("=" * 65)

class ARIMA:
    """
    ARIMA(p,d,q) implemented from scratch using numpy.
    Uses Yule-Walker for AR estimation and residual-based MA fitting.
    """
    def __init__(self, p=2, d=1, q=2):
        self.p = p
        self.d = d
        self.q = q
        self.ar_params  = None
        self.ma_params  = None
        self.ar_const   = None
        self.resid_std  = None
        self._orig_vals = None
        self._diff_vals = None

    def _difference(self, x, d):
        result = x.copy()
        for _ in range(d):
            result = np.diff(result)
        return result

    def _undifference(self, diff_vals, orig_start, d):
        result = diff_vals.copy()
        for _ in range(d):
            result = np.concatenate([[orig_start], result])
            result = np.cumsum(result)
        return result[1:]

    def _yule_walker(self, x, order):
        """Yule-Walker equations for AR parameter estimation."""
        n = len(x)
        x = x - np.mean(x)
        r = np.array([np.dot(x[:n-k], x[k:]) / n for k in range(order + 1)])
        R = np.array([[r[abs(i-j)] for j in range(order)] for i in range(order)])
        try:
            params = np.linalg.solve(R, r[1:])
        except np.linalg.LinAlgError:
            params = np.zeros(order)
        return params, np.mean(x)

    def fit(self, series):
        self._orig_vals = series.copy()
        diff = self._difference(series, self.d)
        self._diff_vals = diff

        # Fit AR part using Yule-Walker
        self.ar_params, self.ar_const = self._yule_walker(diff, self.p)

        # Compute AR residuals
        ar_resids = np.zeros(len(diff))
        diff_m = diff - np.mean(diff)
        for t in range(self.p, len(diff)):
            ar_hat = np.mean(diff) + np.dot(self.ar_params,
                             diff_m[t-self.p:t][::-1])
            ar_resids[t] = diff[t] - ar_hat

        # Fit MA part: regress residuals on lagged residuals
        if self.q > 0:
            X_ma = np.array([ar_resids[t-self.q:t][::-1]
                             for t in range(self.q, len(ar_resids))])
            y_ma = ar_resids[self.q:]
            if len(X_ma) > 0:
                try:
                    self.ma_params = np.linalg.lstsq(X_ma, y_ma, rcond=None)[0]
                except:
                    self.ma_params = np.zeros(self.q)
            else:
                self.ma_params = np.zeros(self.q)
        else:
            self.ma_params = np.array([])

        self.resid_std = np.std(ar_resids[self.p:])
        self._fitted_resids = ar_resids
        return self

    def forecast(self, steps):
        diff = self._diff_vals.copy()
        diff_m = diff - np.mean(diff)
        resids = self._fitted_resids.copy()
        forecasts_diff = []

        for _ in range(steps):
            ar_part = np.mean(diff) + np.dot(self.ar_params,
                              diff_m[-self.p:][::-1]) if self.p > 0 else np.mean(diff)
            ma_part = np.dot(self.ma_params, resids[-self.q:][::-1]) if self.q > 0 else 0.0
            fc = ar_part + ma_part
            forecasts_diff.append(fc)
            resid = 0.0
            resids = np.append(resids, resid)
            diff = np.append(diff, fc)
            diff_m = diff - np.mean(diff)

        last_orig = self._orig_vals[-1]
        if self.d == 1:
            result = np.cumsum(np.concatenate([[last_orig], forecasts_diff]))[1:]
        else:
            result = np.array(forecasts_diff)
        return result

# Use the raw PM2.5 time series (no feature engineering needed for ARIMA)
pm25_series = df_feat[TARGET].values
pm25_split  = int(len(pm25_series) * 0.8)
pm25_train  = pm25_series[:pm25_split]
pm25_test   = pm25_series[pm25_split:]

print(f"\n  Fitting ARIMA(2,1,2) on {len(pm25_train)} training observations...")
arima = ARIMA(p=2, d=1, q=2)
arima.fit(pm25_train)

print(f"  AR parameters: {arima.ar_params.round(4)}")
print(f"  MA parameters: {arima.ma_params.round(4)}")
print(f"  Residual std:  {arima.resid_std:.4f}")

print(f"\n  Forecasting {len(pm25_test)} steps ahead (rolling one-step forecast)...")
# Rolling one-step forecast for fair evaluation
arima_preds = []
history = pm25_train.copy()
batch = 50  # refit every 50 steps for speed
for i in range(0, len(pm25_test), batch):
    chunk_end = min(i + batch, len(pm25_test))
    m = ARIMA(p=2, d=1, q=2)
    m.fit(history)
    fc = m.forecast(chunk_end - i)
    arima_preds.extend(fc[:chunk_end - i])
    history = np.append(history, pm25_test[i:chunk_end])

arima_pred = np.array(arima_preds[:len(pm25_test)])
# Clip negative predictions (PM2.5 cannot be negative)
arima_pred = np.clip(arima_pred, 0, None)
results.append(compute_metrics(pm25_test, arima_pred, 'ARIMA(2,1,2)'))

# ─────────────────────────────────────────────────────────────────
# 9. MODEL 5: LSTM (implemented from scratch in numpy)
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  MODEL 5: LSTM (implemented from scratch, numpy)")
print("=" * 65)

class LSTMVectorised:
    """
    Batch-vectorised LSTM for time series regression.
    Processes the entire sequence in a single matrix operation per timestep.
    Input: X shape (N, T) — N samples, T timesteps
    Predicts the value at timestep T+1 for each sample.
    Uses Adam optimiser with gradient clipping.
    """
    def __init__(self, hidden=32, lr=0.003):
        np.random.seed(42)
        self.H  = hidden
        self.lr = lr
        s = 0.08
        # Combined weight matrix for all 4 gates: [f, i, g, o]
        # W: (4H, H+1)  U: (4H, H)
        self.W  = np.random.randn(4*hidden, 1)      * s  # input weights
        self.U  = np.random.randn(4*hidden, hidden) * s  # recurrent weights
        self.b  = np.zeros((4*hidden, 1))
        self.Wy = np.random.randn(1, hidden) * s
        self.by = np.zeros((1, 1))
        # Adam state
        self.t = 0
        self.ms = {k: np.zeros_like(v) for k,v in self._p().items()}
        self.vs = {k: np.zeros_like(v) for k,v in self._p().items()}

    def _p(self):
        return {'W':self.W,'U':self.U,'b':self.b,'Wy':self.Wy,'by':self.by}

    @staticmethod
    def sig(x): return 1/(1+np.exp(-np.clip(x,-10,10)))

    def forward(self, X):
        """X: (N, T) → returns h_final (N, H), all_h (T, N, H), caches"""
        N, T = X.shape
        H = self.H
        h = np.zeros((N, H))
        c = np.zeros((N, H))
        caches = []
        for t in range(T):
            x = X[:, t:t+1]                  # (N,1)
            gates = x @ self.W.T + h @ self.U.T + self.b.T  # (N, 4H)
            f = self.sig(gates[:, :H])
            i = self.sig(gates[:, H:2*H])
            g = np.tanh(gates[:, 2*H:3*H])
            o = self.sig(gates[:, 3*H:])
            c_new = f*c + i*g
            h_new = o * np.tanh(c_new)
            caches.append((x, h, c, f, i, g, o, c_new, h_new))
            h, c = h_new, c_new
        return h, caches

    def predict(self, X):
        h, _ = self.forward(X)
        return (h @ self.Wy.T + self.by.T).flatten()  # (N,)

    def train_step(self, X, y_true):
        N, T = X.shape
        H = self.H
        h_final, caches = self.forward(X)
        y_pred = (h_final @ self.Wy.T + self.by.T)  # (N,1)
        dy = (y_pred - y_true.reshape(-1,1)) / N     # (N,1)
        loss = float(np.mean((y_pred.flatten() - y_true)**2))

        gWy = dy.T @ h_final                          # (1,H)
        gby = dy.sum(keepdims=True).T                 # (1,1)
        dh  = dy @ self.Wy                            # (N,H)

        gW = np.zeros_like(self.W)
        gU = np.zeros_like(self.U)
        gb = np.zeros_like(self.b)
        dc = np.zeros((N, H))

        for t in reversed(range(T)):
            x, h_prev, c_prev, f, i, g, o, c_new, h_new = caches[t]
            tanh_c = np.tanh(c_new)
            dh_total = dh
            dc_total = dh_total * o * (1 - tanh_c**2) + dc

            df = dc_total * c_prev * f * (1-f)
            di = dc_total * g      * i * (1-i)
            dg = dc_total * i      * (1-g**2)
            do_g = dh_total * tanh_c * o * (1-o)
            dc   = dc_total * f

            dgates = np.concatenate([df, di, dg, do_g], axis=1)  # (N,4H)
            gW += dgates.T @ x
            gU += dgates.T @ h_prev
            gb += dgates.sum(axis=0, keepdims=True).T
            dh  = dgates @ self.U

        grads = {'W': np.clip(gW,-1,1), 'U': np.clip(gU,-1,1),
                 'b': np.clip(gb,-1,1), 'Wy': np.clip(gWy,-1,1),
                 'by': np.clip(gby,-1,1)}
        self._adam(grads)
        return loss

    def _adam(self, grads):
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        p = self._p()
        for k in p:
            self.ms[k] = b1*self.ms[k] + (1-b1)*grads[k]
            self.vs[k] = b2*self.vs[k] + (1-b2)*grads[k]**2
            mh = self.ms[k]/(1-b1**self.t)
            vh = self.vs[k]/(1-b2**self.t)
            p[k] -= self.lr * mh / (np.sqrt(vh)+eps)

    def fit(self, X_tr, y_tr, epochs=40, batch=64, verbose=True):
        N = len(X_tr)
        losses = []
        for ep in range(epochs):
            idx = np.random.permutation(N)
            ep_loss = 0.0
            for start in range(0, N, batch):
                b_idx = idx[start:start+batch]
                ep_loss += self.train_step(X_tr[b_idx], y_tr[b_idx])
            avg = ep_loss / max(1, N//batch)
            losses.append(avg)
            if verbose and (ep % 8 == 0 or ep == epochs-1):
                print(f"  Epoch {ep+1:>3}/{epochs}  Loss: {avg:.6f}")
        return losses


# Prepare LSTM data
scaler_lstm = MinMaxScaler(feature_range=(0, 1))
pm25_scaled = scaler_lstm.fit_transform(pm25_series.reshape(-1,1)).flatten()

SEQ_LEN = 24
X_lstm = np.array([pm25_scaled[i-SEQ_LEN:i] for i in range(SEQ_LEN, len(pm25_scaled))])
y_lstm = np.array([pm25_scaled[i]            for i in range(SEQ_LEN, len(pm25_scaled))])

lstm_split = int(len(X_lstm) * 0.8)
X_tr_l, X_te_l = X_lstm[:lstm_split], X_lstm[lstm_split:]
y_tr_l, y_te_l = y_lstm[:lstm_split], y_lstm[lstm_split:]

print(f"\n  LSTM architecture: input=1, hidden=32, output=1")
print(f"  Sequence length: {SEQ_LEN} hours  |  Batch size: 64")
print(f"  Training sequences: {len(X_tr_l)}  |  Test sequences: {len(X_te_l)}")
print(f"  Optimiser: Adam (lr=0.003)  |  Epochs: 40")

lstm = LSTMVectorised(hidden=32, lr=0.003)
losses = lstm.fit(X_tr_l, y_tr_l, epochs=40, batch=64, verbose=True)

lstm_preds_scaled = lstm.predict(X_te_l).reshape(-1,1)
lstm_preds = scaler_lstm.inverse_transform(lstm_preds_scaled).flatten()
lstm_preds = np.clip(lstm_preds, 0, None)

# Align test targets
y_test_lstm = y_te_l
y_test_lstm_orig = scaler_lstm.inverse_transform(y_test_lstm.reshape(-1,1)).flatten()
min_len = min(len(y_test_lstm_orig), len(lstm_preds))
y_test_lstm_orig = y_test_lstm_orig[:min_len]
lstm_preds = lstm_preds[:min_len]

results.append(compute_metrics(y_test_lstm_orig, lstm_preds, 'LSTM'))

# ─────────────────────────────────────────────────────────────────
# 10. RESULTS SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  COMPREHENSIVE RESULTS SUMMARY")
print("=" * 65)

results_df = pd.DataFrame(results)
results_df = results_df.set_index('Model')
print(f"\n{results_df.to_string()}")
best_mae = results_df['MAE'].idxmin()
best_r2  = results_df['R2'].idxmax()
print(f"\n  Best MAE:  {best_mae} ({results_df.loc[best_mae,'MAE']:.4f})")
print(f"  Best R²:   {best_r2}  ({results_df.loc[best_r2,'R2']:.4f})")

# ─────────────────────────────────────────────────────────────────
# 11. VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────
print("\n[11] GENERATING VISUALIZATIONS...")

fig = plt.figure(figsize=(20, 26))
fig.patch.set_facecolor('#FAFAFA')
gs_main = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.3)

COLORS = {
    'Linear Regression':         '#2196F3',
    'Random Forest (Tuned)':     '#4CAF50',
    'Gradient Boosting (Tuned)': '#FF9800',
    'ARIMA(2,1,2)':              '#9C27B0',
    'LSTM':                      '#F44336',
    'actual':                    '#333333'
}

# ── Plot 1: Raw PM2.5 time series ──────────────────────────────
ax1 = fig.add_subplot(gs_main[0, :])
ax1.plot(df_hourly.index, df_hourly['pm2_5_calibrated_value'],
         color='#1565C0', alpha=0.7, linewidth=0.6, label='PM2.5 (ug/m³)')
ax1.axhline(df_hourly['pm2_5_calibrated_value'].mean(),
            color='red', linestyle='--', linewidth=1.2, label='Mean')
ax1.axhline(25, color='orange', linestyle=':', linewidth=1.2,
            label='WHO 24h guideline (25 ug/m³)')
ax1.set_title('PM2.5 Air Quality — Makerere University Station 94\n(Hourly readings, full dataset)',
              fontsize=13, fontweight='bold', pad=10)
ax1.set_xlabel('Date', fontsize=11)
ax1.set_ylabel('PM2.5 (ug/m³)', fontsize=11)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_facecolor('#F8F9FA')

# ── Plot 2: Metrics bar chart ──────────────────────────────────
ax2 = fig.add_subplot(gs_main[1, 0])
metrics_plot = ['MAE', 'RMSE']
x = np.arange(len(results_df))
width = 0.35
bars1 = ax2.bar(x - width/2, results_df['MAE'],  width, label='MAE',
                color=[COLORS.get(m, '#999') for m in results_df.index], alpha=0.85)
bars2 = ax2.bar(x + width/2, results_df['RMSE'], width, label='RMSE',
                color=[COLORS.get(m, '#999') for m in results_df.index], alpha=0.5,
                hatch='//')
ax2.set_xticks(x)
ax2.set_xticklabels([m.replace(' (Tuned)', '\n(Tuned)').replace(' (numpy)', '\n(numpy)')
                     for m in results_df.index], fontsize=8)
ax2.set_title('MAE and RMSE by Model', fontsize=12, fontweight='bold')
ax2.set_ylabel('Error (ug/m³)', fontsize=10)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_facecolor('#F8F9FA')
for bar in bars1:
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
             f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=7)

# ── Plot 3: R² comparison ──────────────────────────────────────
ax3 = fig.add_subplot(gs_main[1, 1])
colors_r2 = [COLORS.get(m, '#999') for m in results_df.index]
bars_r2 = ax3.barh(results_df.index, results_df['R2'],
                   color=colors_r2, alpha=0.85, edgecolor='white', height=0.6)
ax3.axvline(0, color='black', linewidth=0.8)
ax3.set_xlim(-0.1, 1.05)
ax3.set_title('R² Score by Model\n(Higher = Better)', fontsize=12, fontweight='bold')
ax3.set_xlabel('R² Score', fontsize=10)
ax3.grid(True, alpha=0.3, axis='x')
ax3.set_facecolor('#F8F9FA')
for bar, val in zip(bars_r2, results_df['R2']):
    ax3.text(max(val + 0.01, 0.02), bar.get_y() + bar.get_height()/2,
             f'{val:.3f}', va='center', fontsize=9, fontweight='bold')

# ── Plot 4: Actual vs Predicted — ML models ────────────────────
ax4 = fig.add_subplot(gs_main[2, :])
n_show = min(150, len(y_test))
t_range = np.arange(n_show)
ax4.plot(t_range, y_test[:n_show], color=COLORS['actual'],
         linewidth=1.5, label='Actual', zorder=5)
ax4.plot(t_range, lr_pred[:n_show], color=COLORS['Linear Regression'],
         linewidth=1, alpha=0.8, label='Linear Regression', linestyle='--')
ax4.plot(t_range, rf_pred[:n_show], color=COLORS['Random Forest (Tuned)'],
         linewidth=1, alpha=0.8, label='Random Forest (Tuned)')
ax4.plot(t_range, gb_pred[:n_show], color=COLORS['Gradient Boosting (Tuned)'],
         linewidth=1, alpha=0.8, label='Gradient Boosting (Tuned)', linestyle=':')
ax4.set_title('Actual vs Predicted PM2.5 — ML Models (First 150 test hours)',
              fontsize=12, fontweight='bold')
ax4.set_xlabel('Hours', fontsize=10)
ax4.set_ylabel('PM2.5 (ug/m³)', fontsize=10)
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)
ax4.set_facecolor('#F8F9FA')

# ── Plot 5: Actual vs Predicted — Time Series models ──────────
ax5 = fig.add_subplot(gs_main[3, :])
n_show2 = min(150, len(arima_pred), len(lstm_preds))
t2 = np.arange(n_show2)
ax5.plot(t2, pm25_test[:n_show2], color=COLORS['actual'],
         linewidth=1.5, label='Actual', zorder=5)
ax5.plot(t2, arima_pred[:n_show2], color=COLORS['ARIMA(2,1,2)'],
         linewidth=1, alpha=0.85, label='ARIMA(2,1,2)', linestyle='--')
ax5.plot(t2, lstm_preds[:n_show2], color=COLORS['LSTM'],
         linewidth=1, alpha=0.85, label='LSTM')
ax5.set_title('Actual vs Predicted PM2.5 — Time Series Models (First 150 test hours)',
              fontsize=12, fontweight='bold')
ax5.set_xlabel('Hours', fontsize=10)
ax5.set_ylabel('PM2.5 (ug/m³)', fontsize=10)
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3)
ax5.set_facecolor('#F8F9FA')

fig.suptitle('PM2.5 Air Quality Prediction — Makerere University\nComprehensive Model Comparison',
             fontsize=15, fontweight='bold', y=0.995)

plt.savefig('/mnt/user-data/outputs/ml_project_results.png',
            dpi=150, bbox_inches='tight', facecolor='#FAFAFA')
print("  Saved: ml_project_results.png")

# ── Feature importance plot ────────────────────────────────────
fig2, axes = plt.subplots(1, 2, figsize=(14, 5))
fig2.patch.set_facecolor('#FAFAFA')

# RF feature importance
feat_imp_sorted = feat_imp.sort_values()
axes[0].barh(feat_imp_sorted.index, feat_imp_sorted.values,
             color='#4CAF50', alpha=0.85, edgecolor='white')
axes[0].set_title('Random Forest\nFeature Importance', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Importance Score', fontsize=10)
axes[0].grid(True, alpha=0.3, axis='x')
axes[0].set_facecolor('#F8F9FA')

# GB feature importance
gb_feat = pd.Series(gb_best.feature_importances_,
                    index=FEATURE_COLS).sort_values()
axes[1].barh(gb_feat.index, gb_feat.values,
             color='#FF9800', alpha=0.85, edgecolor='white')
axes[1].set_title('Gradient Boosting\nFeature Importance', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Importance Score', fontsize=10)
axes[1].grid(True, alpha=0.3, axis='x')
axes[1].set_facecolor('#F8F9FA')

plt.suptitle('Feature Importance — Tree-Based Models', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/feature_importance.png',
            dpi=150, bbox_inches='tight', facecolor='#FAFAFA')
print("  Saved: feature_importance.png")

# LSTM training curve
fig3, ax = plt.subplots(figsize=(8, 4))
fig3.patch.set_facecolor('#FAFAFA')
ax.plot(range(1, len(losses)+1), losses, color='#F44336', linewidth=2)
ax.set_title('LSTM Training Loss (MSE per epoch)', fontsize=12, fontweight='bold')
ax.set_xlabel('Epoch', fontsize=10)
ax.set_ylabel('Loss', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_facecolor('#F8F9FA')
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/lstm_training_curve.png',
            dpi=150, bbox_inches='tight', facecolor='#FAFAFA')
print("  Saved: lstm_training_curve.png")

# ─────────────────────────────────────────────────────────────────
# Save results CSV for reference
# ─────────────────────────────────────────────────────────────────
results_df.to_csv('/mnt/user-data/outputs/model_metrics.csv')
print("\n  Saved: model_metrics.csv")

print("\n" + "=" * 65)
print("  ALL DONE")
print("=" * 65)
print(f"\n  Output files:")
print(f"  • ml_project_results.png    — main comparison plots")
print(f"  • feature_importance.png    — RF and GB feature importance")
print(f"  • lstm_training_curve.png   — LSTM convergence")
print(f"  • model_metrics.csv         — metrics table")
