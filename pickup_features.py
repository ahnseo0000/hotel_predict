"""
Part 5-3. 막판 순증(pickup) 타깃과 막판 예약 설명 피처로 성능을 올릴 수 있는가.

배경: 7-F장에서 투숙 7일 전에 수요의 83~96%가 이미 알려져 있고, 모델은 나머지(직전 신규 예약 B - 이후 취소 C)의 날짜별 변동을
거의 설명하지 못했다. 단순 기준(활성 OTB + 최근 28일 평균 순증)과 구분되지 않았다. 이를 직접 겨냥해 본다.

변경 두 가지를 분리해서 비교한다 (모델 설정은 고정, 예측 시점 7일, 운영 시점 OTB, 바깥 3구간 168일 = horizon_sensitivity.py와 같은 설계).
  타깃: 수요 y를 직접 예측(level) vs 순증 D = y - A(활성 OTB)를 예측하고 A를 더함(pickup)
  피처: 기본(base) / 순증 이력(pickup 지연값, 같은 요일 평균, 순증 비율) / 장부 구성(세그먼트 비중, 평균 리드타임, 평균 adr 등) / 둘 다
기준선: 활성 OTB만, 단순 기준 A + 28일 평균 순증, A + 같은 요일 4주 평균 순증.
판정: 단순 기준(A + 28일 평균 순증) 대비 MAPE 차이의 블록 부트스트랩 95% 구간.
비교 변형이 많으므로(호텔 x 과제당 12개) 하나쯤 우연히 유의해질 수 있다는 점을 감안해 읽는다.
"""
import sys, warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge

sys.stdout.reconfigure(encoding='utf-8')
H, TEST_LEN, N_OUTER = 7, 56, 3
LW = 7 * int(np.ceil((H + 1) / 7))

df = pd.read_csv('hotel_bookings.csv')
df['arrival_date'] = pd.to_datetime(df['arrival_date_year'].astype(str) + '-' + df['arrival_date_month'] + '-'
                                    + df['arrival_date_day_of_month'].astype(str), format='%Y-%B-%d')
df['status_date'] = pd.to_datetime(df['reservation_status_date'])
df['nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
df['valid_price'] = (df['adr'] > 0) & (df['adr'] < 1000)
df['revenue'] = np.where(df['valid_price'], df['adr'] * df['nights'], 0.0)
df['guests'] = df['adults'] + df['children'].fillna(0) + df['babies']
df['adr_v'] = df['adr'].where(df['valid_price'])
for nm, cond in [('s_groups', df['market_segment'] == 'Groups'), ('s_onlineTA', df['market_segment'] == 'Online TA'),
                 ('s_offlineTA', df['market_segment'] == 'Offline TA/TO'), ('s_direct', df['market_segment'] == 'Direct'),
                 ('s_corp', df['market_segment'] == 'Corporate'), ('s_nonref', df['deposit_type'] == 'Non Refund'),
                 ('s_transient', df['customer_type'] == 'Transient')]:
    df[nm] = cond.astype(float)
COMP = ['s_groups', 's_onlineTA', 's_offlineTA', 's_direct', 's_corp', 's_nonref', 's_transient']


def build(hotel, rev):
    h = df[df['hotel'] == hotel]
    idx = pd.date_range('2015-08-01', h['arrival_date'].max())
    w = h['revenue'] if rev else pd.Series(1.0, index=h.index)
    kept = h['is_canceled'] == 0
    y = w[kept].groupby(h.loc[kept, 'arrival_date']).sum().reindex(idx).fillna(0.0)
    A_m = (h['lead_time'] >= H) & ~((h['is_canceled'] == 1) & (h['status_date'] <= h['arrival_date'] - pd.Timedelta(days=H)))
    s = h[A_m]; ws = w[A_m]
    d = pd.DataFrame({'y': y.values}, index=idx)
    edges = [(H, H + 7), (H + 7, H + 30), (H + 30, H + 90), (H + 90, 10 ** 6)]; bcols = []
    for i, (lo, hi) in enumerate(edges):
        m = (s['lead_time'] >= lo) & (s['lead_time'] < hi)
        d[f'b{i}'] = ws[m].groupby(s.loc[m, 'arrival_date']).sum().reindex(idx).fillna(0.0).values; bcols.append(f'b{i}')
    d['A'] = d[bcols].sum(axis=1)
    d['D'] = d['y'] - d['A']
    d['dow'] = d.index.dayofweek; d['month'] = d.index.month
    sh = H + 1
    # 기본 피처: 수요 이력과 OTB
    d['lag_s'] = d['y'].shift(sh); d['lag_w'] = d['y'].shift(LW)
    d['roll7'] = d['y'].shift(sh).rolling(7).mean(); d['roll28'] = d['y'].shift(sh).rolling(28).mean()
    base = ['dow', 'month', 'lag_s', 'lag_w', 'roll7', 'roll28', 'A'] + bcols
    # 순증 이력 피처 (모두 t-8 이전 정보)
    d['D_s'] = d['D'].shift(sh); d['D_w'] = d['D'].shift(LW)
    d['D_roll7'] = d['D'].shift(sh).rolling(7).mean(); d['D_roll28'] = d['D'].shift(sh).rolling(28).mean()
    d['D_dow4'] = sum(d['D'].shift(LW + 7 * j) for j in range(4)) / 4
    d['D_ratio28'] = (d['D'] / d['A'].clip(lower=1)).shift(sh).rolling(28).mean()
    d['A_w'] = d['A'].shift(LW)  # 같은 요일 지난주 장부 규모 (순증 대비 규모 참고)
    d['b0_share'] = d['b0'] / d['A'].clip(lower=1)  # 장부 중 직전 한 주에 쌓인 비중 (예약 속도)
    pick = ['D_s', 'D_w', 'D_roll7', 'D_roll28', 'D_dow4', 'D_ratio28', 'b0_share']
    # 장부 구성 피처: 활성 OTB 예약의 세그먼트 비중, 평균 리드타임, 평균 adr, 평균 인원
    g = s.groupby('arrival_date')
    comp = g[COMP].mean().reindex(idx)
    comp['A_lead'] = g['lead_time'].mean().reindex(idx); comp['A_adr'] = g['adr_v'].mean().reindex(idx); comp['A_guests'] = g['guests'].mean().reindex(idx)
    comp['A_cnt'] = g.size().reindex(idx).fillna(0)
    for c in comp.columns: d[c] = comp[c].values
    compc = COMP + ['A_lead', 'A_adr', 'A_guests', 'A_cnt']
    need = ['lag_s', 'lag_w', 'roll7', 'roll28', 'D_s', 'D_w', 'D_roll7', 'D_roll28', 'D_dow4']
    return d.dropna(subset=need), base, pick, compc


def ridge_pred(tr, te, feats, target):
    num = [c for c in feats if c not in ('dow', 'month')]
    pre = ColumnTransformer([('n', Pipeline([('i', SimpleImputer(strategy='median')), ('s', StandardScaler())]), num),
                             ('c', OneHotEncoder(handle_unknown='ignore'), ['dow', 'month'])])
    return Pipeline([('p', pre), ('r', Ridge(alpha=10.0))]).fit(tr[feats], tr[target]).predict(te[feats])


def lgb_pred(tr, te, feats, target):
    out = []
    for sd in (0, 1, 2):
        m = lgb.train({'objective': 'regression', 'metric': 'l1', 'verbose': -1, 'seed': sd, 'num_leaves': 15, 'learning_rate': 0.05,
                       'feature_fraction': 0.8, 'min_child_samples': 10, 'bagging_fraction': 0.8, 'bagging_freq': 1},
                      lgb.Dataset(tr[feats], label=tr[target]), num_boost_round=300)
        out.append(m.predict(te[feats]))
    return np.mean(out, axis=0)


def boot_ci(dape, B=2000, seed=0):
    rng = np.random.default_rng(seed); nb = int(np.ceil(len(dape) / 7))
    bs = [np.concatenate([dape[s * 7:s * 7 + 7] for s in rng.integers(0, nb, nb)]).mean() for _ in range(B)]
    return np.percentile(bs, [2.5, 97.5])


rows = []
for task, rev in [('수요', False), ('매출', True)]:
    floor = 1.0 if rev else 0.0
    for hotel in ['City Hotel', 'Resort Hotel']:
        d, base, pick, compc = build(hotel, rev); N = len(d)
        variants = {'L0 level 기본': ('y', base), 'L3 level 전체피처': ('y', base + pick + compc),
                    'P0 pickup 기본': ('D', base), 'P1 pickup +순증이력': ('D', base + pick),
                    'P2 pickup +장부구성': ('D', base + compc), 'P3 pickup 전체피처': ('D', base + pick + compc)}
        preds = {f'{vn} {mn}': [] for vn in variants for mn in ('Ridge', 'LGBM')}
        preds.update({'기준 OTB만': [], '기준 단순 A+28일평균': [], '기준 단순 A+같은요일4주': []}); ys = []
        for k in range(N_OUTER):
            ts = N - (N_OUTER - k) * TEST_LEN; te = ts + TEST_LEN
            tr, tst = d.iloc[:ts], d.iloc[ts:te]; ys.append(tst['y'].values)
            preds['기준 OTB만'].append(tst['A'].values)
            preds['기준 단순 A+28일평균'].append(tst['A'].values + tst['D_roll28'].values)
            preds['기준 단순 A+같은요일4주'].append(tst['A'].values + tst['D_dow4'].values)
            for vn, (tg, fs) in variants.items():
                for mn, fn in (('Ridge', ridge_pred), ('LGBM', lgb_pred)):
                    p = fn(tr, tst, fs, tg)
                    preds[f'{vn} {mn}'].append(p + tst['A'].values if tg == 'D' else p)
        y = np.concatenate(ys); ape = {nm: np.abs(y - np.concatenate(p)) / np.maximum(y, floor) for nm, p in preds.items()}
        ref = ape['기준 단순 A+28일평균']
        for nm, a in ape.items():
            ci = boot_ci(a - ref)
            rows.append({'과제': task, '호텔': hotel.split()[0], '모델': nm, 'MAPE': a.mean(), '단순기준 대비': (a - ref).mean(), 'CI하한': ci[0], 'CI상한': ci[1]})
        print(f'{task} {hotel} 완료', flush=True)

res = pd.DataFrame(rows)
res.round(4).to_csv('pickup_results.csv', index=False, encoding='utf-8-sig')
pd.set_option('display.width', 250)
for (task, hotel), g in res.groupby(['과제', '호텔'], sort=False):
    g = g.copy(); g['단순기준 대비 [95% CI]'] = g.apply(lambda r: f'{r["단순기준 대비"]:+.4f} [{r["CI하한"]:+.4f}, {r["CI상한"]:+.4f}]', axis=1)
    print(f'\n===== {task} {hotel} (바깥 168일 MAPE, 7일 전 예측) =====')
    print(g[['모델', 'MAPE', '단순기준 대비 [95% CI]']].round(4).to_string(index=False))
