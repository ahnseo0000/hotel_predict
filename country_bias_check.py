"""
Part 2-2. 포르투갈 편향 점검 (취소 예측).

Part 2의 취소 모델(예약 이후 변수 제거 버전, Test AUC 0.827)이 포르투갈 예약에 쏠려 있지는 않은지 확인한다.
  1) 국가 집단별 성능: 포르투갈(PRT) vs 그 외, 전체 AUC가 집단 간 취소율 차이만으로 만들어진 것은 아닌지
  2) country 피처를 빼고 다시 학습했을 때의 변화
  3) 포르투갈만 / 그 외만으로 학습해서 서로 평가했을 때의 변화 (집단 이동 일반화)
  4) SHAP(LightGBM pred_contrib) 기여도에서 country가 차지하는 비중
  5) 집단별 보정(예측 평균 vs 실제 취소율)
Part 2와 같은 분할(예약일 70/15/15), 같은 전처리, 같은 파라미터.
"""
import sys, warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

sys.stdout.reconfigure(encoding='utf-8')
df = pd.read_csv('hotel_bookings.csv')
df['arrival_date'] = pd.to_datetime(df['arrival_date_year'].astype(str) + '-' + df['arrival_date_month'] + '-'
                                    + df['arrival_date_day_of_month'].astype(str), format='%Y-%B-%d')
df['booking_date'] = df['arrival_date'] - pd.to_timedelta(df['lead_time'], unit='D')
df = df.sort_values('booking_date').reset_index(drop=True)
q70, q85 = df['booking_date'].quantile(0.70), df['booking_date'].quantile(0.85)
train = df[df['booking_date'] <= q70]; valid = df[(df['booking_date'] > q70) & (df['booking_date'] <= q85)]; test = df[df['booking_date'] > q85]

DROP = ['reservation_status', 'reservation_status_date', 'assigned_room_type', 'required_car_parking_spaces',
        'arrival_date', 'booking_date', 'booking_ym', 'company',
        'booking_changes', 'total_of_special_requests', 'days_in_waiting_list', 'arrival_date_year']
CATS = ['hotel', 'meal', 'country', 'market_segment', 'distribution_channel', 'reserved_room_type',
        'deposit_type', 'customer_type', 'arrival_date_month']
PARAMS = {'objective': 'binary', 'metric': 'auc', 'learning_rate': 0.05, 'num_leaves': 31, 'feature_fraction': 0.8,
          'bagging_fraction': 0.8, 'bagging_freq': 5, 'verbose': -1, 'seed': 42}


def prep(tr, va, te, drop_country=False):
    """Part 2와 같은 전처리. 학습 구간 통계(agent 타깃 인코딩, 상위 20개 국가)만 사용."""
    parts = []
    for d in (tr, va, te):
        d = d.drop(columns=[c for c in DROP if c in d.columns]).copy()
        d['children'] = d['children'].fillna(0); d['country'] = d['country'].fillna('Unknown'); d['agent'] = d['agent'].fillna(0)
        parts.append(d)
    a, b, c = parts
    gm = a['is_canceled'].mean(); agg = a.groupby('agent')['is_canceled'].agg(['mean', 'count'])
    sm = (agg['mean'] * agg['count'] + gm * 100) / (agg['count'] + 100)
    for d in parts:
        d['agent_te'] = d['agent'].map(sm).fillna(gm); d.drop(columns=['agent'], inplace=True)
    top = a['country'].value_counts().head(20).index
    for d in parts: d['country'] = d['country'].where(d['country'].isin(top), 'Other')
    cats = [x for x in CATS if x != 'country'] if drop_country else CATS
    if drop_country:
        for d in parts: d.drop(columns=['country'], inplace=True)
    for x in cats:
        cc = pd.Categorical(a[x]).categories
        for d in parts: d[x] = pd.Categorical(d[x], categories=cc)
    return [d.drop(columns=['is_canceled']) for d in parts], [d['is_canceled'].values for d in parts]


def fit(X, y):
    dtr = lgb.Dataset(X[0], label=y[0]); dva = lgb.Dataset(X[1], label=y[1], reference=dtr)
    return lgb.train(PARAMS, dtr, num_boost_round=1000, valid_sets=[dva],
                     callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])


def auc(y, p):
    return roc_auc_score(y, p) if 0 < y.mean() < 1 else np.nan


prt_te = (test['country'].fillna('Unknown') == 'PRT').values
prt_tr = (train['country'].fillna('Unknown') == 'PRT').values
prt_va = (valid['country'].fillna('Unknown') == 'PRT').values
rows = []

# ---------- 기본 모델 (Part 2 재현) ----------
X, y = prep(train, valid, test)
m = fit(X, y); p = m.predict(X[2], num_iteration=m.best_iteration)
print(f'[재현] Test AUC {auc(y[2], p):.4f} (Part 2 보고값 0.827), best_iter {m.best_iteration}')
print(f'\n== 집단 구성 (test) ==')
for nm, msk in [('포르투갈', prt_te), ('그 외', ~prt_te)]:
    print(f'{nm}: {msk.sum()}건 ({msk.mean():.1%}), 실제 취소율 {y[2][msk].mean():.3f}, 예측 평균 {p[msk].mean():.3f}')
print(f'학습 구간 포르투갈 비중 {prt_tr.mean():.1%}, 취소율 PRT {y[0][prt_tr].mean():.3f} / 그 외 {y[0][~prt_tr].mean():.3f}')

print('\n== 1) 집단별 AUC (기본 모델) ==')
a_all, a_p, a_n = auc(y[2], p), auc(y[2][prt_te], p[prt_te]), auc(y[2][~prt_te], p[~prt_te])
print(f'전체 {a_all:.4f} / 포르투갈 안에서 {a_p:.4f} / 그 외 안에서 {a_n:.4f}')
rows += [('기본', '전체', a_all), ('기본', '포르투갈 안', a_p), ('기본', '그 외 안', a_n)]

# 집단 안에서 순위를 얼마나 맞히는지 vs 집단 평균만 쓰는 경우
grp_only = np.where(prt_te, y[0][prt_tr].mean(), y[0][~prt_tr].mean())
print(f'참고: 국가 집단 평균 취소율만 쓴 예측의 전체 AUC {auc(y[2], grp_only):.4f} (집단 정보만으로 얻는 순위 정보)')

print('\n== 상위 국가별 AUC (test 200건 이상) ==')
ct = test['country'].fillna('Unknown').values
for c_, n_ in pd.Series(ct).value_counts().items():
    if n_ < 200: continue
    msk = ct == c_
    print(f'{c_:>4} n={n_:>5} 취소율 {y[2][msk].mean():.3f} 예측평균 {p[msk].mean():.3f} AUC {auc(y[2][msk], p[msk]):.3f}')

# ---------- 2) country 제외 ----------
print('\n== 2) country 피처를 뺀 모델 ==')
Xc, yc = prep(train, valid, test, drop_country=True)
mc = fit(Xc, yc); pc = mc.predict(Xc[2], num_iteration=mc.best_iteration)
for nm, msk in [('전체', np.ones(len(yc[2]), bool)), ('포르투갈 안', prt_te), ('그 외 안', ~prt_te)]:
    ab, ac = auc(y[2][msk], p[msk]), auc(yc[2][msk], pc[msk])
    print(f'{nm}: country 포함 {ab:.4f} → 제외 {ac:.4f} (차이 {ac - ab:+.4f})')
    rows.append(('country 제외', nm, ac))

# ---------- 3) 집단 이동 일반화 ----------
print('\n== 3) 집단 이동: 한쪽 집단만으로 학습해서 서로 평가 ==')
tr_p, va_p, te_p = train[prt_tr], valid[prt_va], test[prt_te]
tr_n, va_n, te_n = train[~prt_tr], valid[~prt_va], test[~prt_te]
for nm_tr, (a_, b_) in [('포르투갈만 학습', (tr_p, va_p)), ('그 외만 학습', (tr_n, va_n))]:
    Xs, ys = prep(a_, b_, test, drop_country=True)
    ms = fit(Xs, ys); ps = ms.predict(Xs[2], num_iteration=ms.best_iteration)
    print(f'{nm_tr}: 포르투갈 test {auc(ys[2][prt_te], ps[prt_te]):.4f} / 그 외 test {auc(ys[2][~prt_te], ps[~prt_te]):.4f} (학습 {len(a_)}건)')
    rows.append((nm_tr, '포르투갈 안', auc(ys[2][prt_te], ps[prt_te]))); rows.append((nm_tr, '그 외 안', auc(ys[2][~prt_te], ps[~prt_te])))
print(f'참고 전체 학습(country 제외): 포르투갈 {auc(yc[2][prt_te], pc[prt_te]):.4f} / 그 외 {auc(yc[2][~prt_te], pc[~prt_te]):.4f}')

# ---------- 4) SHAP ----------
print('\n== 4) SHAP 기여도 (test, 평균 |기여|) ==')
sh = m.predict(X[2], num_iteration=m.best_iteration, pred_contrib=True)[:, :-1]
imp = pd.Series(np.abs(sh).mean(axis=0), index=X[2].columns); imp = imp / imp.sum()
print('전체:'); print((imp.sort_values(ascending=False).head(8)).round(3).to_string())
for nm, msk in [('포르투갈', prt_te), ('그 외', ~prt_te)]:
    ii = pd.Series(np.abs(sh[msk]).mean(axis=0), index=X[2].columns); ii = ii / ii.sum()
    print(f'{nm} 상위 5:', ', '.join(f'{k} {v:.3f}' for k, v in ii.sort_values(ascending=False).head(5).items()))
print(f'country 비중 {imp["country"]:.3f}, agent_te {imp["agent_te"]:.3f}, deposit_type {imp["deposit_type"]:.3f}')
pr = X[2]['country'].astype(str).values == 'PRT'
print(f'포르투갈 예약의 country 평균 기여(로짓) {sh[pr, list(X[2].columns).index("country")].mean():+.3f}, 그 외 {sh[~pr, list(X[2].columns).index("country")].mean():+.3f}')

# ---------- 5) 보정 ----------
print('\n== 5) 집단별 보정 (예측 평균 vs 실제 취소율, 예측 구간별) ==')
for nm, msk in [('포르투갈', prt_te), ('그 외', ~prt_te)]:
    q = pd.qcut(p[msk], 5, duplicates='drop')
    t = pd.DataFrame({'p': p[msk], 'y': y[2][msk], 'q': q}).groupby('q', observed=True).agg(예측=('p', 'mean'), 실제=('y', 'mean'), n=('y', 'size'))
    print(nm); print(t.round(3).to_string())

pd.DataFrame(rows, columns=['설정', '평가 집단', 'AUC']).round(4).to_csv('cancel_country_check.csv', index=False, encoding='utf-8-sig')
