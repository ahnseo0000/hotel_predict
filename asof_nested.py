"""
Part 5. 운영 시점 OTB(as-of OTB)로 다시 계산한 중첩 walk-forward + 비교 모델 확장.

Part 3-1, 4-1의 OTB는 "lead_time >= n인 예약 전체"라서, 투숙 n일 전 이전에 이미 취소된 예약까지 센다.
운영 시점(투숙 n일 전)에는 그 예약들이 취소로 확정돼 있으므로 "그 시점까지 취소되지 않은 예약"만 센다.
이 스크립트는 그 정의(as-of)로 모든 OTB 계열 피처를 다시 만들고, 같은 중첩 절차로 모델을 비교한다.

환경변수
  TARGET    demand | revenue
  ASOF      1(기본, 운영 시점 정의) | 0(Part 3-1, 4-1 정의. 재현 확인용)
  OUTER_RUN 계산할 바깥 구간 (예: 0,1,2,3)
  HOTEL_RUN 계산할 호텔 (예: City,Resort)
  EXTRA     1(기본, 확장 모델 포함) | 0(기존 9개 모델만)
"""
import os, sys, time, pickle, warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import holidays
from itertools import product
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, ExtraTreesRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.forecasting.theta import ThetaModel

sys.stdout.reconfigure(encoding='utf-8')
TARGET = os.environ.get('TARGET', 'demand')
ASOF = int(os.environ.get('ASOF', '1'))
EXTRA = int(os.environ.get('EXTRA', '1'))
OUTER_RUN = [int(x) for x in os.environ.get('OUTER_RUN', '0,1,2,3').split(',') if x.strip() != '']
HOTEL_RUN = [x for x in os.environ.get('HOTEL_RUN', 'City,Resort').split(',') if x.strip() != '']
assert TARGET in ('demand', 'revenue')
REV = TARGET == 'revenue'
YCOL = 'revenue' if REV else 'actual_stays'
CACHE = f"cache_asof_{TARGET}" + ('' if ASOF else '_legacy')
os.makedirs(CACHE, exist_ok=True); os.makedirs('cache_asof', exist_ok=True)
print(f'TARGET={TARGET} ASOF={ASOF} EXTRA={EXTRA} OUTER_RUN={OUTER_RUN} HOTEL_RUN={HOTEL_RUN} -> {CACHE}', flush=True)

df = pd.read_csv('hotel_bookings.csv')
df['arrival_date'] = pd.to_datetime(
    df['arrival_date_year'].astype(str) + '-' + df['arrival_date_month'] + '-'
    + df['arrival_date_day_of_month'].astype(str), format='%Y-%B-%d')
df['nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
df['valid_price'] = (df['adr'] > 0) & (df['adr'] < 1000)
df['revenue'] = df['adr'] * df['nights']
df['status_date'] = pd.to_datetime(df['reservation_status_date'])

# ---------------- 취소 확률 (Part 3-1과 동일, 캐시 공유) ----------------
RAW_COLS = ['hotel', 'lead_time', 'arrival_date_month', 'arrival_date_week_number', 'arrival_date_day_of_month',
            'stays_in_weekend_nights', 'stays_in_week_nights', 'adults', 'children', 'babies', 'meal',
            'country', 'market_segment', 'distribution_channel', 'is_repeated_guest', 'previous_cancellations',
            'previous_bookings_not_canceled', 'reserved_room_type', 'deposit_type', 'customer_type', 'adr', 'agent']
CAT_COLS = ['hotel', 'meal', 'country', 'market_segment', 'distribution_channel', 'reserved_room_type',
            'deposit_type', 'customer_type', 'arrival_date_month']
raw = df[RAW_COLS].copy()
raw['children'] = raw['children'].fillna(0); raw['country'] = raw['country'].fillna('Unknown'); raw['agent'] = raw['agent'].fillna(0)
y_all = df['is_canceled'].values
CANCEL_PARAMS = {'objective': 'binary', 'learning_rate': 0.05, 'num_leaves': 31, 'feature_fraction': 0.8,
                 'bagging_fraction': 0.8, 'bagging_freq': 5, 'verbose': -1, 'seed': 42}

def cancel_fit_predict(tr_idx, ap_idx, rounds=300):
    tr, ap = raw.iloc[tr_idx].copy(), raw.iloc[ap_idx].copy()
    ytr = y_all[tr_idx]; gm = ytr.mean()
    agg = pd.DataFrame({'a': tr['agent'].values, 'y': ytr}).groupby('a')['y'].agg(['mean', 'count'])
    sm = (agg['mean'] * agg['count'] + gm * 100) / (agg['count'] + 100)
    for d in (tr, ap):
        d['agent_te'] = d['agent'].map(sm).fillna(gm); d.drop(columns=['agent'], inplace=True)
    top = tr['country'].value_counts().head(20).index
    for d in (tr, ap):
        d['country'] = d['country'].where(d['country'].isin(top), 'Other')
    for c in CAT_COLS:
        cats = pd.Categorical(tr[c]).categories
        tr[c] = pd.Categorical(tr[c], categories=cats); ap[c] = pd.Categorical(ap[c], categories=cats)
    return lgb.train(CANCEL_PARAMS, lgb.Dataset(tr, label=ytr), num_boost_round=rounds).predict(ap)

CHUNK_DAYS, MIN_CHUNKS = 45, 3
t0 = df['arrival_date'].min()
chunk = ((df['arrival_date'] - t0).dt.days // CHUNK_DAYS).values
P_START = t0 + pd.Timedelta(days=MIN_CHUNKS * CHUNK_DAYS)
assert os.path.exists('cache_nested/p_cancel.npy'), 'cache_nested/p_cancel.npy 가 필요합니다 (Part 3-1 먼저 실행)'
df['p_cancel'] = np.load('cache_nested/p_cancel.npy')

# ---------------- 운영 시점 활성 예약 마스크 ----------------
# act[n]: 투숙 n일 전 시점에 "이미 예약돼 있고, 아직 취소되지 않은" 예약
def active_mask(n):
    booked = df['lead_time'] >= n
    if not ASOF: return booked
    cancelled_by = (df['is_canceled'] == 1) & (df['status_date'] <= df['arrival_date'] - pd.Timedelta(days=n))
    return booked & ~cancelled_by
OTB_DAYS = [7, 14, 30, 60, 90]
for n in OTB_DAYS: df[f'act_{n}'] = active_mask(n)
print('활성 예약 비율 (lead_time >= n 대비):', {n: round(df[f'act_{n}'].sum() / (df['lead_time'] >= n).sum(), 3) for n in OTB_DAYS}, flush=True)

# 조건부 취소 확률: 투숙 7일 전까지 살아남은 예약이 그 뒤에 취소될 확률 (생존 조건 반영)
LATE_PATH = 'cache_asof/p_late.npy'
if ASOF and os.path.exists(LATE_PATH): p_late = np.load(LATE_PATH)
else:
    p_late = np.full(len(df), np.nan)
    if ASOF:
        a7 = df['act_7'].values
        for k in range(MIN_CHUNKS, chunk.max() + 1):
            tr_idx, ap_idx = np.where((chunk < k) & a7)[0], np.where(chunk == k)[0]
            p_late[ap_idx] = cancel_fit_predict(tr_idx, ap_idx)
        np.save(LATE_PATH, p_late)
df['p_late'] = p_late
if ASOF:
    ok = ~np.isnan(p_late) & df['act_7'].values
    print(f"조건부 취소 모델 AUC (활성 예약 {ok.sum():,}건): {roc_auc_score(y_all[ok], p_late[ok]):.4f}", flush=True)

# ---------------- 피처 ----------------
b = df.copy()
b['is_groups'] = (b['market_segment'] == 'Groups').astype(int)
b['is_online'] = (b['market_segment'] == 'Online TA').astype(int)
b['is_offline'] = (b['market_segment'] == 'Offline TA/TO').astype(int)
b['is_direct'] = (b['market_segment'] == 'Direct').astype(int)
b['is_corp'] = (b['market_segment'] == 'Corporate').astype(int)
b['is_nonref'] = (b['deposit_type'] == 'Non Refund').astype(int)
b['is_prt'] = (b['country'] == 'PRT').astype(int)
b['is_agent1'] = (b['agent'] == 1.0).astype(int)
b['has_prevcancel'] = (b['previous_cancellations'] > 0).astype(int)
b['guests'] = b['adults'] + b['children'].fillna(0) + b['babies']
b['cust_transient'] = (b['customer_type'] == 'Transient').astype(int)
b['cust_party'] = (b['customer_type'] == 'Transient-Party').astype(int)
b['cust_contract'] = (b['customer_type'] == 'Contract').astype(int)
b['cust_group'] = (b['customer_type'] == 'Group').astype(int)
b['has_kids'] = ((b['children'].fillna(0) > 0) | (b['babies'] > 0)).astype(int)
b['meal_bb'] = (b['meal'] == 'BB').astype(int)
b['meal_hbfb'] = b['meal'].isin(['HB', 'FB']).astype(int)
SHARE = {
    'segment': {'is_groups': 'groups_sh', 'is_online': 'online_sh', 'is_offline': 'offline_sh', 'is_direct': 'direct_sh', 'is_corp': 'corp_sh'},
    'biased': {'is_nonref': 'nonref_sh', 'is_prt': 'prt_sh', 'is_agent1': 'agent1_sh'},
    'history': {'has_prevcancel': 'prevcancel_sh', 'is_repeated_guest': 'repeat_sh'},
    'attrs': {'cust_transient': 'cust_transient_sh', 'cust_party': 'cust_party_sh', 'cust_contract': 'cust_contract_sh',
              'cust_group': 'cust_group_sh', 'has_kids': 'kids_sh', 'meal_bb': 'meal_bb_sh', 'meal_hbfb': 'meal_hbfb_sh'},
}
YEARS = [2015, 2016, 2017]
pt_h = sorted(pd.to_datetime(list(holidays.Portugal(years=YEARS).keys())))
foreign = [set(pd.to_datetime(list(holidays.country_holidays(c, years=YEARS).keys()))) for c in ['GB', 'FR', 'DE', 'ES', 'IE']]
pt_set = set(pt_h)

def calendar_feats(dates):
    dates = pd.to_datetime(pd.Series(dates).values); nxt, prv = [], []
    for d in dates:
        later = [(h - d).days for h in pt_h if h >= d]; earlier = [(d - h).days for h in pt_h if h <= d]
        nxt.append(min(later[0] if later else 14, 14)); prv.append(min(earlier[-1] if earlier else 14, 14))
    return pd.DataFrame({'is_holiday': [int(d in pt_set) for d in dates], 'days_to_next_hol': nxt, 'days_since_hol': prv,
                         'n_foreign_hol': [sum(d in s for s in foreign) for d in dates],
                         'weekofyear': pd.Series(dates).dt.isocalendar().week.astype(int).values})

def make_daily(hotel_name):
    h = df[df['hotel'] == hotel_name]
    if REV:
        cr = h.groupby('arrival_date')['is_canceled'].mean().rename('cancel_rate')
        rv = h[(h['is_canceled'] == 0) & h['valid_price']].groupby('arrival_date')['revenue'].sum().rename('revenue')
        daily = pd.concat([rv, cr], axis=1).reset_index(); daily['revenue'] = daily['revenue'].fillna(0)
    else:
        daily = h.groupby('arrival_date').agg(actual_stays=('is_canceled', lambda x: (x == 0).sum()), cancel_rate=('is_canceled', 'mean')).reset_index()
    return daily[daily['arrival_date'] >= '2015-08-01'].sort_values('arrival_date').reset_index(drop=True)

def build_frame(hotel_name):
    d = make_daily(hotel_name); y = d[YCOL]
    d['dow'] = d['arrival_date'].dt.dayofweek; d['month'] = d['arrival_date'].dt.month
    d['is_weekend'] = (d['dow'] >= 5).astype(int)
    d['lag_8'], d['lag_14'] = y.shift(8), y.shift(14)
    d['roll7_h7'], d['roll14_h7'] = y.shift(8).rolling(7).mean(), y.shift(8).rolling(14).mean()
    d['level_28'], d['level_56'] = y.shift(8).rolling(28).mean(), y.shift(8).rolling(56, min_periods=28).mean()
    d['cancel_rate_7'], d['cancel_rate_28'] = d['cancel_rate'].shift(8).rolling(7).mean(), d['cancel_rate'].shift(8).rolling(28).mean()
    h = b[b['hotel'] == hotel_name]
    for n in OTB_DAYS:
        d[f'otb_{n}'] = d['arrival_date'].map(h[h[f'act_{n}']].groupby('arrival_date').size()).fillna(0)
    d['otb_pace'] = d['otb_7'] - d['otb_30']
    s7 = h[h['act_7']]; g = s7.groupby('arrival_date')
    parts = [g[list(cols)].mean().rename(columns=cols) for cols in SHARE.values()]
    parts.append(pd.DataFrame({'lead_mean_7': g['lead_time'].mean()}))
    parts.append(pd.DataFrame({'adr_mean_7': s7[s7['valid_price']].groupby('arrival_date')['adr'].mean()}))
    parts.append(pd.DataFrame({'nights_mean_7': g['nights'].mean(), 'guests_mean_7': g['guests'].mean()}))
    risk = pd.DataFrame({'cancel_prob_7': g['p_cancel'].mean(), 'exp_net_7': g.size() - g['p_cancel'].sum(),
                         'exp_net_7c': g.size() - g['p_late'].sum()})
    for n in [14, 30]:
        sn = h[h[f'act_{n}']].groupby('arrival_date'); risk[f'exp_net_{n}'] = sn.size() - sn['p_cancel'].sum()
    parts.append(risk)
    for part in parts:
        for c in part.columns:
            d[c] = d['arrival_date'].map(part[c])
    for c in ['exp_net_7', 'exp_net_7c', 'exp_net_14', 'exp_net_30']:
        d[c] = d[c].fillna(0)
    if REV:
        hv = h[h['valid_price']]
        for n in OTB_DAYS:
            sub = hv[hv[f'act_{n}']]
            d[f'otb_rev_{n}'] = d['arrival_date'].map(sub.groupby('arrival_date')['revenue'].sum()).fillna(0)
            d[f'exp_rev_{n}'] = d['arrival_date'].map((sub['revenue'] * (1 - sub['p_cancel'])).groupby(sub['arrival_date']).sum()).fillna(0)
        sub = hv[hv['act_7']]
        d['exp_rev_7c'] = d['arrival_date'].map((sub['revenue'] * (1 - sub['p_late'])).groupby(sub['arrival_date']).sum()).fillna(0)
        d['otb_rev_pace'] = d['otb_rev_7'] - d['otb_rev_30']
    d = pd.concat([d, calendar_feats(d['arrival_date'])], axis=1)
    d = d[d['arrival_date'] >= P_START]
    return d.dropna(subset=['lag_14', 'roll14_h7', 'lag_8', 'level_28', 'cancel_rate_28']).reset_index(drop=True)

frames = {'City': build_frame('City Hotel'), 'Resort': build_frame('Resort Hotel')}
HOTELS = {'City': 'City Hotel', 'Resort': 'Resort Hotel'}
for k, f in frames.items():
    print(k, f.shape, f['arrival_date'].min().date(), '~', f['arrival_date'].max().date(), flush=True)
assert len(frames['City']) == len(frames['Resort'])

BASE = ['dow', 'month', 'is_weekend', 'lag_8', 'lag_14', 'roll7_h7', 'roll14_h7'] + [f'otb_{n}' for n in OTB_DAYS] + ['otb_pace']
if REV: BASE += [f'otb_rev_{n}' for n in OTB_DAYS] + ['otb_rev_pace']
GROUPS = {
    'segment': list(SHARE['segment'].values()), 'biased': list(SHARE['biased'].values()), 'history': list(SHARE['history'].values()),
    'leadtime': ['lead_mean_7'], 'price': ['adr_mean_7'], 'size': ['nights_mean_7', 'guests_mean_7'], 'attrs': list(SHARE['attrs'].values()),
    'cancel': ['cancel_prob_7', 'exp_net_7', 'exp_net_14', 'exp_net_30'] + (['exp_rev_7', 'exp_rev_14', 'exp_rev_30'] if REV else []),
    'cancel_lag': ['cancel_rate_7', 'cancel_rate_28'],
    'calendar': ['is_holiday', 'days_to_next_hol', 'days_since_hol', 'n_foreign_hol', 'weekofyear'], 'trend': ['level_28', 'level_56'],
}

# ---------------- 중첩 검증 틀 ----------------
TEST_LEN, VAL_LEN, N_INNER = 56, 56, 4
N = len(frames['City'])
OUTER_TS = {0: N - 3 * TEST_LEN, 1: N - 2 * TEST_LEN, 2: N - TEST_LEN, 3: N}

def inner_folds(ts):
    return [(ts - TEST_LEN * (N_INNER - f) - VAL_LEN, ts - TEST_LEN * (N_INNER - f), ts - TEST_LEN * (N_INNER - f) + TEST_LEN) for f in range(N_INNER)]

def mape(y, p): return float(np.mean(np.abs(y - p) / (np.maximum(y, 1) if REV else y)))
def to_target(d, mode): return d[YCOL] / d['level_28'] if mode == 'ratio' else d[YCOL]
def from_target(pred, d, mode): return pred * d['level_28'].values if mode == 'ratio' else pred

def run_cv(d, fp, folds):
    ys, ps = [], []
    for tr_end, va_end, te_end in folds:
        tr, va, te = d.iloc[:tr_end], d.iloc[tr_end:va_end], d.iloc[va_end:te_end]
        ps.append(np.asarray(fp(tr, va, te), dtype=float)); ys.append(te[YCOL].values)
    return np.concatenate(ys), np.concatenate(ps)

DEFAULT_LGB = {'num_leaves': 31, 'learning_rate': 0.05, 'feature_fraction': 0.8, 'min_child_samples': 20, 'lambda_l2': 0.0}

def lgb_fp(feats, params, mode, seeds):
    def fp(tr, va, te):
        out = []
        for s in seeds:
            pr = {'objective': 'regression', 'metric': 'l1', 'verbose': -1, 'seed': s, 'bagging_fraction': 0.8, 'bagging_freq': 1, **params}
            dtr = lgb.Dataset(tr[feats], label=to_target(tr, mode)); dva = lgb.Dataset(va[feats], label=to_target(va, mode), reference=dtr)
            m = lgb.train(pr, dtr, num_boost_round=1000, valid_sets=[dva], callbacks=[lgb.early_stopping(50, verbose=False)])
            out.append(m.predict(te[feats]))
        return from_target(np.mean(out, axis=0), te, mode)
    return fp

def xgb_fp(feats, params, mode, seeds):
    def fp(tr, va, te):
        out = []
        for s in seeds:
            m = xgb.XGBRegressor(n_estimators=1000, objective='reg:absoluteerror', eval_metric='mae', early_stopping_rounds=50,
                                 subsample=0.8, random_state=s, n_jobs=2, verbosity=0, **params)
            m.fit(tr[feats], to_target(tr, mode), eval_set=[(va[feats], to_target(va, mode))], verbose=False)
            out.append(m.predict(te[feats]))
        return from_target(np.mean(out, axis=0), te, mode)
    return fp

def cat_fp(feats, params, mode, seeds):
    def fp(tr, va, te):
        out = []
        for s in seeds:
            m = CatBoostRegressor(iterations=1000, loss_function='MAE', eval_metric='MAE', early_stopping_rounds=50,
                                  random_seed=s, verbose=0, thread_count=2, allow_writing_files=False, **params)
            m.fit(tr[feats], to_target(tr, mode), eval_set=(va[feats], to_target(va, mode)))
            out.append(m.predict(te[feats]))
        return from_target(np.mean(out, axis=0), te, mode)
    return fp

def sk_fp(make_model, feats, mode):
    def fp(tr, va, te):
        full = pd.concat([tr, va]); mdl = make_model(feats)
        mdl.fit(full[feats], to_target(full, mode))
        return from_target(mdl.predict(te[feats]), te, mode)
    return fp

def make_ridge(alpha):
    def mk(feats):
        cat = [c for c in ['dow', 'month'] if c in feats]; num = [c for c in feats if c not in cat]
        pre = ColumnTransformer([('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())]), num),
                                 ('cat', OneHotEncoder(handle_unknown='ignore'), cat)])
        return Pipeline([('pre', pre), ('m', Ridge(alpha=alpha))])
    return mk
def make_rf(min_leaf, max_feat):
    return lambda feats: Pipeline([('imp', SimpleImputer(strategy='median')),
                                   ('m', RandomForestRegressor(n_estimators=150, min_samples_leaf=min_leaf, max_features=max_feat, random_state=0, n_jobs=-1))])
def make_et(min_leaf, max_feat):
    return lambda feats: Pipeline([('imp', SimpleImputer(strategy='median')),
                                   ('m', ExtraTreesRegressor(n_estimators=200, min_samples_leaf=min_leaf, max_features=max_feat, random_state=0, n_jobs=-1))])
def make_hgb(lr, depth, iters, leaf):
    return lambda feats: HistGradientBoostingRegressor(learning_rate=lr, max_depth=depth, max_iter=iters, min_samples_leaf=leaf, random_state=0)

SCALE = 1000.0 if REV else 1.0
def classical_fp(kind, exog_cols, order=(1, 0, 1), sorder=(1, 0, 0, 7)):
    def predict_window(d, ts, te_end):
        y = d[YCOL].values / SCALE; ex = d[exog_cols].values / SCALE if exog_cols else None; out = np.full(te_end - ts, np.nan)
        for b0 in range(ts, te_end, 7):
            o = b0 - 8; yt = y[:o + 1]
            try:
                if kind == 'ets':
                    f = ExponentialSmoothing(yt, trend='add', damped_trend=True, seasonal='add', seasonal_periods=7).fit().forecast(14)[7:14]
                elif kind == 'theta':
                    f = ThetaModel(yt, period=7).fit().forecast(14)[7:14]
                else:
                    f = SARIMAX(yt, exog=ex[:o + 1], order=order, seasonal_order=sorder).fit(disp=False, maxiter=100).forecast(14, exog=ex[o + 1:o + 15])[7:14]
            except Exception:
                f = d['lag_14'].values[b0:b0 + 7] / SCALE
            n = min(7, te_end - b0)
            out[b0 - ts:b0 - ts + n] = np.asarray(f)[:n] * SCALE
        return out
    return predict_window

def run_classical(d, pw, folds):
    ys, ps = [], []
    for tr_end, va_end, te_end in folds:
        ps.append(pw(d, va_end, te_end)); ys.append(d[YCOL].values[va_end:te_end])
    return np.concatenate(ys), np.concatenate(ps)

# ---------------- 한 (바깥 구간, 호텔) 단위 ----------------
MIN_GAIN = 0.003
SEEDS_SEL, SEEDS_TUNE, SEEDS_FINAL = [0], [0, 1], [0, 1, 2, 3, 4]
LGB_GRID = {'num_leaves': [7, 15, 31], 'min_child_samples': [5, 10, 20], 'learning_rate': [0.03, 0.05],
            'feature_fraction': [0.6, 0.8, 1.0], 'lambda_l2': [0.0, 1.0, 10.0]}
if REV:
    EXOG, EXOG_C, EXOG_ONLY_EXP, OTB_ONLY = ['otb_rev_7', 'exp_rev_7'], ['otb_rev_7', 'exp_rev_7c'], ['exp_rev_7'], ['otb_rev_7']
else:
    EXOG, EXOG_C, EXOG_ONLY_EXP, OTB_ONLY = ['otb_7', 'exp_net_7'], ['otb_7', 'exp_net_7c'], ['exp_net_7'], ['otb_7']
PART = 'Part4' if REV else 'Part3'
N_NAIVE, N_ETS, N_SAR = '계절naive(lag_14)', 'ETS(감쇠추세)', 'SARIMAX+OTB'
N_RIDGE, N_RF, N_HGB, N_LGB0, N_LGBT, N_BLEND = 'Ridge', 'RandomForest', 'HistGB', f'LightGBM({PART} 설정)', 'LightGBM(튜닝)', '블렌드(LGBM+Ridge)'
ORIG9 = [N_NAIVE, N_ETS, N_SAR, N_RIDGE, N_RF, N_HGB, N_LGB0, N_LGBT, N_BLEND]
EXTRA_MODELS = ['Theta', 'SARIMAX+OTB(조건부c)', 'SARIMAX 차수(2,0,1)(1,0,1)', 'ExtraTrees', 'XGBoost', 'CatBoost']
ABLATION = ['SARIMAX[OTB만]', 'SARIMAX[exp만]', 'SARIMAX[exp_c만]']

def run_unit(hotel, k):
    ts = OUTER_TS[k]; d = frames[hotel]; d_dev = d.iloc[:ts]; folds = inner_folds(ts); T = time.time()
    def tick(msg): print(f'   [{TARGET} {hotel} outer{k}] {msg} ({time.time() - T:.0f}s)', flush=True)
    rng = np.random.default_rng(1000 * k + (0 if hotel == 'City' else 1))
    rng2 = np.random.default_rng(7000 + 1000 * k + (0 if hotel == 'City' else 1))
    def score(feats): y, pr = run_cv(d_dev, lgb_fp(feats, DEFAULT_LGB, 'raw', SEEDS_SEL), folds); return mape(y, pr)
    cur_feats, chosen, remaining, log = list(BASE), [], list(GROUPS), []
    cur = base_score = score(cur_feats); step = 0
    while remaining:
        step += 1
        scores = {g: score(cur_feats + GROUPS[g]) for g in remaining}
        log += [{'step': step, 'group': g, 'score': s, 'delta': s - cur} for g, s in scores.items()]
        best = min(scores, key=scores.get)
        if cur - scores[best] >= MIN_GAIN:
            chosen.append(best); cur_feats += GROUPS[best]; cur = scores[best]; remaining.remove(best)
        else: break
    feats = cur_feats; tick(f'선택 완료 {chosen}')
    cands = [(dict(DEFAULT_LGB), 'raw')]; combos = [dict(zip(LGB_GRID, v)) for v in product(*LGB_GRID.values())]
    for i in rng.choice(len(combos), 10, replace=False): cands.append((combos[i], str(rng.choice(['raw', 'ratio']))))
    best_l = None
    for pr, mode in cands:
        y, ps = run_cv(d_dev, lgb_fp(feats, pr, mode, SEEDS_TUNE), folds); s = mape(y, ps)
        if best_l is None or s < best_l[0]: best_l = (s, pr, mode)
    tuned = {'lgb': best_l}
    grids = {'ridge': [(make_ridge(a), {'alpha': a}, m) for a in [0.1, 1, 10, 100, 1000] for m in ['raw', 'ratio']],
             'rf': [(make_rf(l, mf), {'leaf': l, 'max_feat': mf}, m) for l in [2, 5, 10] for mf in [0.5, 1.0] for m in ['raw', 'ratio']],
             'hgb': [(make_hgb(lr, dp, it, lf), {'lr': float(lr), 'depth': dp, 'iters': it, 'leaf': lf}, m)
                     for lr, dp, it, lf, m in [(rng.choice([0.03, 0.1]), int(rng.choice([3, 5])), int(rng.choice([150, 300])),
                                                int(rng.choice([5, 15])), str(rng.choice(['raw', 'ratio']))) for _ in range(6)]]}
    if EXTRA:
        grids['et'] = [(make_et(l, mf), {'leaf': l, 'max_feat': mf}, m) for l in [2, 5, 10] for mf in [0.5, 1.0] for m in ['raw', 'ratio']]
    for name, grid in grids.items():
        best = None
        for mk, info, m in grid:
            y, ps = run_cv(d_dev, sk_fp(mk, feats, m), folds); s = mape(y, ps)
            if best is None or s < best[0]: best = (s, mk, info, m)
        tuned[name] = best
    if EXTRA:
        xgb_cands = [({'max_depth': int(rng2.choice([2, 3, 4])), 'learning_rate': float(rng2.choice([0.03, 0.1])),
                       'colsample_bytree': float(rng2.choice([0.6, 0.8, 1.0])), 'min_child_weight': int(rng2.choice([1, 3, 5])),
                       'reg_lambda': float(rng2.choice([1.0, 10.0]))}, str(rng2.choice(['raw', 'ratio']))) for _ in range(6)]
        cat_cands = [({'depth': int(rng2.choice([3, 4, 6])), 'learning_rate': float(rng2.choice([0.03, 0.08])),
                       'l2_leaf_reg': float(rng2.choice([1.0, 5.0, 10.0]))}, str(rng2.choice(['raw', 'ratio']))) for _ in range(6)]
        for name, fpf, cs in [('xgb', xgb_fp, xgb_cands), ('cat', cat_fp, cat_cands)]:
            best = None
            for pr, m in cs:
                y, ps = run_cv(d_dev, fpf(feats, pr, m, [0]), folds); s = mape(y, ps)
                if best is None or s < best[0]: best = (s, pr, m)
            tuned[name] = best
    tick('튜닝 완료')
    lgb_s, lgb_pr, lgb_mode = tuned['lgb']
    models = {N_NAIVE: ('fp', lambda tr, va, te: te['lag_14'].values),
              N_ETS: ('cl', classical_fp('ets', [])), N_SAR: ('cl', classical_fp('sarimax', EXOG)),
              N_RIDGE: ('fp', sk_fp(tuned['ridge'][1], feats, tuned['ridge'][3])),
              N_RF: ('fp', sk_fp(tuned['rf'][1], feats, tuned['rf'][3])),
              N_HGB: ('fp', sk_fp(tuned['hgb'][1], feats, tuned['hgb'][3])),
              N_LGB0: ('fp', lgb_fp(BASE, DEFAULT_LGB, 'raw', SEEDS_FINAL)),
              N_LGBT: ('fp', lgb_fp(feats, lgb_pr, lgb_mode, SEEDS_FINAL))}
    if EXTRA:
        models.update({'Theta': ('cl', classical_fp('theta', [])),
                       'SARIMAX+OTB(조건부c)': ('cl', classical_fp('sarimax', EXOG_C)),
                       'SARIMAX 차수(2,0,1)(1,0,1)': ('cl', classical_fp('sarimax', EXOG, (2, 0, 1), (1, 0, 1, 7))),
                       'ExtraTrees': ('fp', sk_fp(tuned['et'][1], feats, tuned['et'][3])),
                       'XGBoost': ('fp', xgb_fp(feats, tuned['xgb'][1], tuned['xgb'][2], [0, 1, 2])),
                       'CatBoost': ('fp', cat_fp(feats, tuned['cat'][1], tuned['cat'][2], [0, 1, 2])),
                       'SARIMAX[OTB만]': ('cl', classical_fp('sarimax', OTB_ONLY)),
                       'SARIMAX[exp만]': ('cl', classical_fp('sarimax', EXOG_ONLY_EXP)),
                       'SARIMAX[exp_c만]': ('cl', classical_fp('sarimax', [EXOG_C[1]]))})
    inner_pred, outer_pred = {}, {}
    ofold = [(ts - VAL_LEN, ts, min(ts + TEST_LEN, N))]
    for name, (kind, fp) in models.items():
        tick(f'평가 {name}')
        if kind == 'cl':
            yi, inner_pred[name] = run_classical(d, fp, folds)
            if k < 3: yo, outer_pred[name] = run_classical(d, fp, ofold)
        else:
            yi, inner_pred[name] = run_cv(d_dev, fp, folds)
            if k < 3: yo, outer_pred[name] = run_cv(d, fp, ofold)
    inner_pred[N_BLEND] = (inner_pred[N_LGBT] + inner_pred[N_RIDGE]) / 2
    if k < 3: outer_pred[N_BLEND] = (outer_pred[N_LGBT] + outer_pred[N_RIDGE]) / 2
    inner_mape = {n: mape(yi, p) for n, p in inner_pred.items()}
    cand9 = {n: inner_mape[n] for n in ORIG9}
    cand_all = {n: inner_mape[n] for n in ORIG9 + (EXTRA_MODELS if EXTRA else [])}
    return {'hotel': hotel, 'k': k, 'ts': ts, 'base_score': base_score, 'sel_score': cur, 'groups': chosen, 'n_feats': len(feats), 'log': log,
            'lgb': (lgb_s, lgb_pr, lgb_mode), 'tuned_info': {n: v[2] if n in ('ridge', 'rf', 'hgb', 'et') else v[1:] for n, v in tuned.items()},
            'inner_mape': inner_mape, 'final9': min(cand9, key=cand9.get), 'final_all': min(cand_all, key=cand_all.get), 'y_inner': yi,
            'y_outer': yo if k < 3 else None, 'outer_pred': outer_pred if k < 3 else None,
            'outer_dates': d['arrival_date'].iloc[ts:min(ts + TEST_LEN, N)].values if k < 3 else None}

if __name__ == '__main__':
    for k in OUTER_RUN:
        for hotel in HOTEL_RUN:
            path = f'{CACHE}/outer{k}_{hotel}.pkl'
            if os.path.exists(path): print(f'outer {k} {hotel}: 저장된 결과 사용'); continue
            t1 = time.time(); r = run_unit(hotel, k)
            pickle.dump(r, open(path, 'wb'))
            print(f"outer {k} {hotel}: 완료 {time.time() - t1:.0f}초 | 선택 그룹 {r['groups']} | 선택 모델(기존9) {r['final9']} | (전체) {r['final_all']}", flush=True)
