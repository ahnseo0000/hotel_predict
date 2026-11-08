"""
Part 2-4. 취소 예측 개선 실험: 시간순 타깃 인코딩, 추가 피처, CatBoost, 확률 보정.

비교 (모두 Part 2와 같은 분할 70/15/15, 예약 시점에 알 수 있는 변수만 사용):
  기본            Part 2 그대로 (agent는 train 전체 취소율의 고정 타깃 인코딩)
  시간순 TE(a)     agent 취소율을 "예약 시점 이전에 결과가 확정된 예약"만으로 시간순 누적 계산
  시간순 TE(a, c)  agent와 country 모두
  + 추가 피처      위에 총 숙박일, 인원, 1인당 adr, 예약 월, 예약 요일 등
  CatBoost         타깃 인코딩 없이 agent, country 등을 범주로 직접 처리
확정 시점: 예약의 결과는 reservation_status_date(취소일 또는 퇴실일)에 알려진다고 본다 (엄격히 예약일보다 이전).
확률 보정: valid 예측으로 Platt, isotonic을 맞추고 test에서 평가. 보정이 Part 2-1 순효과에 주는 영향도 본다.
"""
import sys, warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

sys.stdout.reconfigure(encoding='utf-8')
df = pd.read_csv('hotel_bookings.csv')
df['arrival_date'] = pd.to_datetime(df['arrival_date_year'].astype(str) + '-' + df['arrival_date_month'] + '-'
                                    + df['arrival_date_day_of_month'].astype(str), format='%Y-%B-%d')
df['booking_date'] = df['arrival_date'] - pd.to_timedelta(df['lead_time'], unit='D')
df['status_date'] = pd.to_datetime(df['reservation_status_date'])
df = df.sort_values('booking_date').reset_index(drop=True)
q70, q85 = df['booking_date'].quantile(0.70), df['booking_date'].quantile(0.85)
is_tr = (df['booking_date'] <= q70).values
is_va = ((df['booking_date'] > q70) & (df['booking_date'] <= q85)).values
is_te = (df['booking_date'] > q85).values
y_all = df['is_canceled'].values

DROP = ['reservation_status', 'reservation_status_date', 'assigned_room_type', 'required_car_parking_spaces', 'arrival_date', 'booking_date',
        'booking_ym', 'company', 'booking_changes', 'total_of_special_requests', 'days_in_waiting_list', 'arrival_date_year', 'status_date']
CATS = ['hotel', 'meal', 'country', 'market_segment', 'distribution_channel', 'reserved_room_type', 'deposit_type', 'customer_type', 'arrival_date_month']
PARAMS = {'objective': 'binary', 'metric': 'auc', 'learning_rate': 0.05, 'num_leaves': 31, 'feature_fraction': 0.8,
          'bagging_fraction': 0.8, 'bagging_freq': 5, 'verbose': -1}
SEEDS = (42, 43, 44)


def expanding_te(key, m=100):
    """예약 i의 인코딩 = (예약일 이전에 결과가 확정된 같은 key 예약들의 취소 수 + m x 전체율) / (확정 수 + m). 자기 자신은 제외(엄격히 이전)."""
    b = df['booking_date'].values; r = df['status_date'].values
    out = np.full(len(df), np.nan)
    # 전체 확정 취소율(시간순)
    order = np.argsort(r); rs = r[order]; cs = np.cumsum(y_all[order])
    k = np.searchsorted(rs, b, side='left')
    prior = np.where(k > 0, cs[np.maximum(k - 1, 0)] / np.maximum(k, 1), np.nan)
    prior = np.where(np.isnan(prior), y_all[is_tr].mean(), prior)
    for _, idx in df.groupby(key).indices.items():
        o = np.argsort(r[idx]); rs_ = r[idx][o]; cs_ = np.cumsum(y_all[idx][o])
        k_ = np.searchsorted(rs_, b[idx], side='left')
        c_ = np.where(k_ > 0, cs_[np.maximum(k_ - 1, 0)], 0)
        out[idx] = (c_ + m * prior[idx]) / (k_ + m)
    return out


def extras():
    e = pd.DataFrame(index=df.index)
    e['nights_total'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
    e['guests'] = df['adults'] + df['children'].fillna(0) + df['babies']
    e['adr_per_guest'] = df['adr'] / e['guests'].clip(lower=1)
    e['weekend_share'] = df['stays_in_weekend_nights'] / e['nights_total'].clip(lower=1)
    e['book_month'] = df['booking_date'].dt.month; e['book_dow'] = df['booking_date'].dt.dayofweek
    e['has_agent'] = (df['agent'].fillna(0) != 0).astype(int)
    return e


def build(variant):
    d = df.drop(columns=[c for c in DROP if c in df.columns]).copy()
    d['children'] = d['children'].fillna(0); d['country'] = d['country'].fillna('Unknown'); d['agent'] = d['agent'].fillna(0)
    if variant == 'catboost':
        for c in CATS + ['agent']: d[c] = d[c].astype(str)
        X = d.drop(columns=['is_canceled']); return X, [c for c in CATS + ['agent']]
    if variant == '기본':
        gm = y_all[is_tr].mean(); a = pd.DataFrame({'a': d.loc[is_tr, 'agent'], 'y': y_all[is_tr]}).groupby('a')['y'].agg(['mean', 'count'])
        sm = (a['mean'] * a['count'] + gm * 100) / (a['count'] + 100)
        d['agent_te'] = d['agent'].map(sm).fillna(gm)
    else:
        d['agent_te'] = expanding_te(df['agent'].fillna(0).values)
        if variant in ('시간순 TE(a, c)', '+ 추가 피처'):
            d['country_te'] = expanding_te(df['country'].fillna('Unknown').values)
        if variant == '+ 추가 피처':
            d = pd.concat([d, extras()], axis=1)
    d = d.drop(columns=['agent'])
    top = d.loc[is_tr, 'country'].value_counts().head(20).index
    d['country'] = d['country'].where(d['country'].isin(top), 'Other')
    for c in CATS:
        cc = pd.Categorical(d.loc[is_tr, c]).categories; d[c] = pd.Categorical(d[c], categories=cc)
    return d.drop(columns=['is_canceled']), None


def fit_lgb(X):
    pv, pt, its = [], [], []
    for s in SEEDS:
        dtr = lgb.Dataset(X[is_tr], label=y_all[is_tr]); dva = lgb.Dataset(X[is_va], label=y_all[is_va], reference=dtr)
        m = lgb.train({**PARAMS, 'seed': s}, dtr, num_boost_round=1000, valid_sets=[dva], callbacks=[lgb.early_stopping(50, verbose=False)])
        pv.append(m.predict(X[is_va], num_iteration=m.best_iteration)); pt.append(m.predict(X[is_te], num_iteration=m.best_iteration)); its.append(m.best_iteration)
    return np.mean(pv, 0), np.mean(pt, 0), its


def fit_cat(X, cats):
    m = CatBoostClassifier(iterations=1500, learning_rate=0.05, depth=6, eval_metric='AUC', random_seed=42, verbose=0,
                           od_type='Iter', od_wait=50, cat_features=cats)
    m.fit(X[is_tr], y_all[is_tr], eval_set=(X[is_va], y_all[is_va]), use_best_model=True)
    return m.predict_proba(X[is_va])[:, 1], m.predict_proba(X[is_te])[:, 1], [m.get_best_iteration()]


def ece(y, p, bins=10):
    q = pd.qcut(p, bins, duplicates='drop'); t = pd.DataFrame({'p': p, 'y': y, 'q': q}).groupby('q', observed=True).agg(p=('p', 'mean'), y=('y', 'mean'), n=('y', 'size'))
    return float((t['n'] * (t['p'] - t['y']).abs()).sum() / t['n'].sum())


def metrics(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {'AUC': roc_auc_score(y, p), 'LogLoss': log_loss(y, p), 'Brier': brier_score_loss(y, p), 'ECE': ece(y, p)}


yv, yt = y_all[is_va], y_all[is_te]
preds, rows = {}, []
for v in ['기본', '시간순 TE(a)', '시간순 TE(a, c)', '+ 추가 피처', 'catboost']:
    X, cats = build(v)
    pv, pt, its = fit_cat(X, cats) if v == 'catboost' else fit_lgb(X)
    preds[v] = (pv, pt)
    rows.append({'변형': v if v != 'catboost' else 'CatBoost(범주 직접)', '피처 수': X.shape[1], 'Valid AUC': roc_auc_score(yv, pv), **{f'Test {k}': x for k, x in metrics(yt, pt).items()}, 'iter': its})
    print(f'{v} 완료', flush=True)
res = pd.DataFrame(rows)
pd.set_option('display.width', 250)
print('\n===== 변형별 성능 (LightGBM은 시드 3개 평균) =====')
print(res.round(4).to_string(index=False))

# 기본 대비 AUC 차이 (쌍체 부트스트랩, 예약을 독립으로 가정)
rng = np.random.default_rng(0); n = len(yt); B = 300; ids = [rng.integers(0, n, n) for _ in range(B)]
print('\n===== 기본 대비 Test AUC 차이 [95% 부트스트랩 구간] =====')
for v in list(preds)[1:]:
    d = [roc_auc_score(yt[i], preds[v][1][i]) - roc_auc_score(yt[i], preds['기본'][1][i]) for i in ids]
    print(f'{v}: {roc_auc_score(yt, preds[v][1]) - roc_auc_score(yt, preds["기본"][1]):+.4f} [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}]')

# ---------- 확률 보정 ----------
print('\n===== 확률 보정 (valid로 맞추고 test 평가) =====')
cal_rows = []
ev = df[['hotel', 'adr', 'stays_in_weekend_nights', 'stays_in_week_nights']].iloc[np.where(is_te)[0]].copy()
nights = ev['stays_in_weekend_nights'] + ev['stays_in_week_nights']
rev = np.where((ev['adr'] > 0) & (ev['adr'] < 1000), ev['adr'] * nights, 0.0).astype(float); lost = rev * yt
ths = np.round(np.arange(0.05, 0.96, 0.01), 2); S0, C0 = 0.3, 50
prt = (df.loc[is_te, 'country'].fillna('Unknown') == 'PRT').values


def net(mask): return float(S0 * lost[mask].sum() - C0 * mask.sum())


def business(pv_, pt_):
    # (a) 임계값을 valid에서 선택: valid의 손실 매출 계산
    evv = df[['adr', 'stays_in_weekend_nights', 'stays_in_week_nights']].iloc[np.where(is_va)[0]]
    nv = evv['stays_in_weekend_nights'] + evv['stays_in_week_nights']
    lost_v = np.where((evv['adr'] > 0) & (evv['adr'] < 1000), evv['adr'] * nv, 0.0) * yv
    vals = np.array([S0 * lost_v[pv_ >= t].sum() - C0 * (pv_ >= t).sum() for t in ths]); t_v = ths[vals.argmax()]
    a = net(pt_ >= t_v)
    # (b) 기대값 규칙: S0 x p x 예약매출 >= C0 이면 대응 (임계값 튜닝 없음)
    b = net(S0 * pt_ * rev >= C0)
    return t_v, a, b, int((pt_ >= t_v).sum()), int((S0 * pt_ * rev >= C0).sum())


for v in ['기본', '+ 추가 피처']:
    pv, pt = preds[v]
    platt = LogisticRegression(C=1e6).fit(np.log(np.clip(pv, 1e-6, 1 - 1e-6) / (1 - np.clip(pv, 1e-6, 1 - 1e-6))).reshape(-1, 1), yv)
    lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))).reshape(-1, 1)
    iso = IsotonicRegression(out_of_bounds='clip', y_min=1e-4, y_max=1 - 1e-4).fit(pv, yv)
    for nm, (cv, ct) in [('보정 없음', (pv, pt)), ('Platt', (platt.predict_proba(lg(pv))[:, 1], platt.predict_proba(lg(pt))[:, 1])),
                         ('Isotonic', (iso.predict(pv), iso.predict(pt)))]:
        m = metrics(yt, ct); t_v, a, b, na, nb = business(cv, ct)
        cal_rows.append({'모델': v, '보정': nm, 'LogLoss': m['LogLoss'], 'Brier': m['Brier'], 'ECE': m['ECE'], '예측 평균': ct.mean(), '실제 취소율': yt.mean(),
                         '포르투갈 예측평균': ct[prt].mean(), '포르투갈 실제': yt[prt].mean(), 'valid 임계값': t_v, '순효과(valid 임계값)': a, '순효과(기대값 규칙)': b, '기대값 규칙 대상 수': nb,
                         '예측 순효과(기대값 규칙)': float(S0 * (ct * rev)[S0 * ct * rev >= C0].sum() - C0 * nb)})
cal = pd.DataFrame(cal_rows); print(cal.round(4).to_string(index=False))
print(f'\n참고: 전수 대응 {net(np.ones(len(yt), bool)):,.0f}, test에서 임계값을 고른 최적(기본 모델) {max(net(preds["기본"][1] >= t) for t in ths):,.0f}')

res.round(4).to_csv('cancel_improve_results.csv', index=False, encoding='utf-8-sig')
cal.round(4).to_csv('cancel_calibration_results.csv', index=False, encoding='utf-8-sig')
