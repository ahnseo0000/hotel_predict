"""
Part 2-3. 취소 대응 효과(Part 2-1)의 포르투갈 의존도 점검 + 임계값을 valid에서 정하는 점검.

Part 2-1은 최적 임계값을 test 위에서 골랐다(낙관적). 여기서는 두 가지를 본다.
  1) 평가 집단별 순효과: 전체 / 포르투갈 제외 / 포르투갈만 (모델은 그대로)
  2) 임계값을 valid에서 정해 test에 적용했을 때의 순효과 (test에서 고른 값과 비교)
  3) 포르투갈을 학습과 평가에서 모두 뺀 재학습 모델
대표 시나리오 s=0.3, c=50 (Part 2-1과 동일), 모델 설정과 전처리도 Part 2-1과 동일.
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

DROP = ['reservation_status', 'reservation_status_date', 'assigned_room_type', 'required_car_parking_spaces', 'arrival_date', 'booking_date',
        'company', 'booking_changes', 'total_of_special_requests', 'days_in_waiting_list', 'arrival_date_year']
CATS = ['hotel', 'meal', 'country', 'market_segment', 'distribution_channel', 'reserved_room_type', 'deposit_type', 'customer_type', 'arrival_date_month']
PARAMS = {'objective': 'binary', 'metric': 'auc', 'learning_rate': 0.05, 'num_leaves': 31, 'feature_fraction': 0.8,
          'bagging_fraction': 0.8, 'bagging_freq': 5, 'verbose': -1, 'seed': 42}
S0, C0 = 0.3, 50
ths = np.round(np.arange(0.05, 0.96, 0.01), 2)


def prep(d):
    d = d.drop(columns=DROP).copy()
    d['children'] = d['children'].fillna(0); d['country'] = d['country'].fillna('Unknown'); d['agent'] = d['agent'].fillna(0)
    return d


def fit_predict(tr_df, va_df, te_df):
    tr, va, te = prep(tr_df), prep(va_df), prep(te_df)
    gm = tr['is_canceled'].mean(); agg = tr.groupby('agent')['is_canceled'].agg(['mean', 'count'])
    sm = (agg['mean'] * agg['count'] + gm * 100) / (agg['count'] + 100)
    for d in (tr, va, te):
        d['agent_te'] = d['agent'].map(sm).fillna(gm); d.drop(columns=['agent'], inplace=True)
    top = tr['country'].value_counts().head(20).index
    for d in (tr, va, te):
        d['country'] = d['country'].where(d['country'].isin(top), 'Other')
        for c in CATS: d[c] = d[c].astype('category')
    for c in CATS:
        cats = tr[c].cat.categories
        for d in (va, te): d[c] = pd.Categorical(d[c].astype(object), categories=cats)
    X = [d.drop(columns=['is_canceled']) for d in (tr, va, te)]; y = [d['is_canceled'] for d in (tr, va, te)]
    dtr = lgb.Dataset(X[0], label=y[0]); dva = lgb.Dataset(X[1], label=y[1], reference=dtr)
    m = lgb.train(PARAMS, dtr, num_boost_round=1000, valid_sets=[dva], callbacks=[lgb.early_stopping(50, verbose=False)])
    return m.predict(X[1]), m.predict(X[2])


def build_ev(d, p):
    e = d[['hotel', 'adr', 'stays_in_weekend_nights', 'stays_in_week_nights', 'is_canceled']].copy()
    e['p'] = p; e['y'] = e['is_canceled'].values
    nights = e['stays_in_weekend_nights'] + e['stays_in_week_nights']
    e['rev'] = np.where((e['adr'] > 0) & (e['adr'] < 1000), e['adr'] * nights, 0.0); e['lost'] = e['rev'] * e['y']
    return e


def effect(e, th, s=S0, c=C0):
    f = e['p'] >= th
    return (s * e.loc[f, 'lost']).sum() - c * f.sum(), int(f.sum())


def best(e):
    v = np.array([effect(e, th)[0] for th in ths]); k = v.argmax(); return ths[k], v[k]


pv, pt = fit_predict(train, valid, test)
ev_va, ev_te = build_ev(valid, pv), build_ev(test, pt)
prt_te = (test['country'].fillna('Unknown') == 'PRT').values
print(f'Test AUC {roc_auc_score(ev_te.y, ev_te.p):.4f}')

print(f'\n== 1) 평가 집단별 순효과 (s={S0}, c={C0}, 모델 그대로, 최적 임계값은 각 집단의 test에서 선택) ==')
rows = []
for nm, e in [('전체', ev_te), ('포르투갈 제외', ev_te[~prt_te]), ('포르투갈만', ev_te[prt_te])]:
    th, nb = best(e); nb_all = (S0 * e['lost']).sum() - C0 * len(e)
    rows.append({'평가 집단': nm, '예약수': len(e), '취소율': e.y.mean(), 'AUC': roc_auc_score(e.y, e.p), '손실 매출': e.lost.sum(),
                 '최적 임계값': th, '모델 순효과': nb, '전수 대응': nb_all, '순효과/손실매출': nb / e.lost.sum()})
t1 = pd.DataFrame(rows); print(t1.round(3).to_string(index=False))

print('\n== 2) 임계값을 valid에서 정해 test에 적용 (전체 모델) ==')
th_v, nb_v = best(ev_va)
nb_t_at_v, n_t = effect(ev_te, th_v); th_t, nb_t_best = best(ev_te)
print(f'valid에서 고른 임계값 {th_v} → test 순효과 {nb_t_at_v:,.0f} (대상 {n_t:,}건)')
print(f'test에서 고른 임계값 {th_t} → test 순효과 {nb_t_best:,.0f}  (차이 {nb_t_at_v - nb_t_best:,.0f}, 낙관 정도 {1 - nb_t_at_v / nb_t_best:.1%})')
print(f'전수 대응 {(S0 * ev_te.lost).sum() - C0 * len(ev_te):,.0f}')
for nm, msk in [('포르투갈 제외', ~prt_te), ('포르투갈만', prt_te)]:
    e = ev_te[msk]; print(f'  {nm}: valid 임계값 적용 순효과 {effect(e, th_v)[0]:,.0f} / 그 집단 최적 {best(e)[1]:,.0f}')

print('\n== 3) 포르투갈을 학습, 평가에서 모두 제외하고 재학습 ==')
nprt = lambda d: d[d['country'].fillna('Unknown') != 'PRT']
tr2, va2, te2 = nprt(train), nprt(valid), nprt(test)
pv2, pt2 = fit_predict(tr2, va2, te2)
e2v, e2t = build_ev(va2, pv2), build_ev(te2, pt2)
th2v, _ = best(e2v); th2t, nb2t = best(e2t)
print(f'AUC {roc_auc_score(e2t.y, e2t.p):.4f}, 예약수 {len(e2t):,}, 손실 매출 {e2t.lost.sum():,.0f}')
print(f'test 최적 임계값 {th2t} 순효과 {nb2t:,.0f} / valid 임계값 {th2v} 적용 순효과 {effect(e2t, th2v)[0]:,.0f} / 전수 대응 {(S0 * e2t.lost).sum() - C0 * len(e2t):,.0f}')

t1.round(4).to_csv('cancel_country_impact.csv', index=False, encoding='utf-8-sig')
