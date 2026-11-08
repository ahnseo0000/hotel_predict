"""
Part 5-1. 예측 시점(H일 전) 민감도.

투숙 H일 전에 그날의 수요/매출을 예측한다고 할 때 성능이 어떻게 변하는지 본다.
7일은 설계 가정이었으므로 H = 1, 3, 7, 14, 28로 바꿔 같은 방식으로 비교한다.

간소 설계 (15개 모델 중첩 절차를 시점마다 돌리면 시점당 1시간 이상 걸림):
  - 모델 설정 고정 (선택, 튜닝 없음): 계절 naive, Ridge, LightGBM, SARIMAX[OTB만]
  - 바깥 평가는 asof_nested.py와 같은 마지막 168일 (56일 x 3구간, 구간마다 그 이전 데이터로만 학습)
  - OTB 정의 두 가지를 같은 조건에서 비교: asof(투숙 H일 전에 이미 취소된 예약 제외) vs legacy(lead_time >= H 전체)
  - 따라서 7장의 최종 수치(선택 절차 A)와는 직접 비교하지 않는다. 같은 간소 설계 안에서의 상대 비교용이다.

환경변수: TARGET=demand|revenue
"""
import os, sys, warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from statsmodels.tsa.statespace.sarimax import SARIMAX

sys.stdout.reconfigure(encoding='utf-8')
TARGET = os.environ.get('TARGET', 'demand'); REV = TARGET == 'revenue'
HORIZONS = [1, 3, 7, 14, 28]
TEST_LEN, N_OUTER = 56, 3
SCALE = 1000.0 if REV else 1.0

df = pd.read_csv('hotel_bookings.csv')
df['arrival_date'] = pd.to_datetime(df['arrival_date_year'].astype(str) + '-' + df['arrival_date_month'] + '-'
                                    + df['arrival_date_day_of_month'].astype(str), format='%Y-%B-%d')
df['status_date'] = pd.to_datetime(df['reservation_status_date'])
df['nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
df['valid_price'] = (df['adr'] > 0) & (df['adr'] < 1000)
df['revenue'] = np.where(df['valid_price'], df['adr'] * df['nights'], 0.0)

def mape(y, p): return float(np.mean(np.abs(y - p) / (np.maximum(y, 1) if REV else y)))
def wape(y, p): return float(np.abs(y - p).sum() / y.sum())

def build(hotel, H, asof):
    h = df[df['hotel'] == hotel]
    idx = pd.date_range('2015-08-01', h['arrival_date'].max())
    kept = h[h['is_canceled'] == 0]
    y = (kept.groupby('arrival_date')['revenue'].sum() if REV else kept.groupby('arrival_date').size()).reindex(idx).fillna(0.0)
    d = pd.DataFrame({'y': y.values}, index=idx)
    d['dow'] = d.index.dayofweek; d['month'] = d.index.month
    lw = 7 * int(np.ceil((H + 1) / 7))
    d['lag_s'] = d['y'].shift(H + 1); d['lag_w'] = d['y'].shift(lw)
    d['roll7'] = d['y'].shift(H + 1).rolling(7).mean(); d['roll28'] = d['y'].shift(H + 1).rolling(28).mean()
    booked = h['lead_time'] >= H
    if asof:
        booked = booked & ~((h['is_canceled'] == 1) & (h['status_date'] <= h['arrival_date'] - pd.Timedelta(days=H)))
    s = h[booked]
    edges = [(H, H + 7), (H + 7, H + 30), (H + 30, H + 90), (H + 90, 10 ** 6)]
    w = s['revenue'] if REV else pd.Series(1.0, index=s.index)
    cols = []
    for i, (lo, hi) in enumerate(edges):
        m = (s['lead_time'] >= lo) & (s['lead_time'] < hi)
        d[f'b{i}'] = w[m].groupby(s.loc[m, 'arrival_date']).sum().reindex(idx).fillna(0.0).values; cols.append(f'b{i}')
    d['otb'] = d[cols].sum(axis=1)
    return d.dropna(subset=['lag_s', 'lag_w', 'roll7', 'roll28']), cols

FEATS = lambda cols: ['dow', 'month', 'lag_s', 'lag_w', 'roll7', 'roll28', 'otb'] + cols

def ridge_pred(tr, te, feats):
    num = [c for c in feats if c not in ('dow', 'month')]
    pre = ColumnTransformer([('n', Pipeline([('i', SimpleImputer(strategy='median')), ('s', StandardScaler())]), num),
                             ('c', OneHotEncoder(handle_unknown='ignore'), ['dow', 'month'])])
    m = Pipeline([('p', pre), ('r', Ridge(alpha=10.0))]).fit(tr[feats], tr['y'])
    return m.predict(te[feats])

def lgb_pred(tr, te, feats):
    out = []
    for s in (0, 1, 2):
        m = lgb.train({'objective': 'regression', 'metric': 'l1', 'verbose': -1, 'seed': s, 'num_leaves': 15, 'learning_rate': 0.05,
                       'feature_fraction': 0.8, 'min_child_samples': 10, 'bagging_fraction': 0.8, 'bagging_freq': 1},
                      lgb.Dataset(tr[feats], label=tr['y']), num_boost_round=300)
        out.append(m.predict(te[feats]))
    return np.mean(out, axis=0)

def sarimax_pred(d, ts_pos, te_pos, H):
    y = d['y'].values / SCALE; ex = d[['otb']].values / SCALE; out = np.full(te_pos - ts_pos, np.nan)
    for b0 in range(ts_pos, te_pos, 7):
        o = b0 - H - 1; n = min(7, te_pos - b0)
        try:
            f = SARIMAX(y[:o + 1], exog=ex[:o + 1], order=(1, 0, 1), seasonal_order=(1, 0, 0, 7)).fit(disp=False, maxiter=100) \
                .forecast(H + 7, exog=ex[o + 1:o + H + 8])[H:H + n]
        except Exception:
            f = d['lag_w'].values[b0:b0 + n] / SCALE
        out[b0 - ts_pos:b0 - ts_pos + n] = np.asarray(f)[:n] * SCALE
    return out

rows = []
for hotel in ['City Hotel', 'Resort Hotel']:
    for H in HORIZONS:
        for asof in (1, 0):
            d, cols = build(hotel, H, asof); feats = FEATS(cols); N = len(d)
            preds = {k: [] for k in ('계절naive', 'Ridge', 'LightGBM', 'SARIMAX[OTB만]')}; ys = []
            for k in range(N_OUTER):
                ts = N - (N_OUTER - k) * TEST_LEN; te = ts + TEST_LEN
                tr, tst = d.iloc[:ts], d.iloc[ts:te]
                ys.append(tst['y'].values)
                preds['계절naive'].append(tst['lag_w'].values)
                preds['Ridge'].append(ridge_pred(tr, tst, feats))
                preds['LightGBM'].append(lgb_pred(tr, tst, feats))
                preds['SARIMAX[OTB만]'].append(sarimax_pred(d, ts, te, H))
            y = np.concatenate(ys)
            for name, p in preds.items():
                p = np.concatenate(p)
                rows.append({'target': TARGET, 'hotel': hotel.split()[0], 'H': H, 'OTB정의': 'asof' if asof else 'legacy', '모델': name,
                             'MAPE': mape(y, p), 'WAPE': wape(y, p)})
            print(f'{TARGET} {hotel} H={H} asof={asof} 완료', flush=True)
res = pd.DataFrame(rows)
res.round(4).to_csv(f'results/horizon_{TARGET}_results.csv', index=False, encoding='utf-8-sig')
for hotel in ['City', 'Resort']:
    for dfn in ('asof', 'legacy'):
        pv = res[(res.hotel == hotel) & (res['OTB정의'] == dfn)].pivot(index='모델', columns='H', values='MAPE')
        print(f'\n===== [{TARGET}] {hotel} OTB={dfn}: 예측 시점 H일 전별 168일 MAPE =====')
        print(pv.round(4).to_string())
