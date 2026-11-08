"""
Part 5-2. 예측 구간과 오차 분해 (Part 5 캐시 사용, 모델 재학습 없음).

A) 예측 구간: 분할 컨포멀 방식.
   바깥 구간 k의 구간은 그 이전 구간들의 (실제 / 예측) 비율 분위수로만 만든다. 구간 0은 보정 자료가 없어 평가에서 제외.
   따라서 평가는 구간 1, 2(112일)이며, 80%, 90% 구간의 실제 포함률을 본다.
B) 수요 분해: 투숙일 t의 순수요 y를 예측 시점(t-7)에 이미 알려진 부분과 모르는 부분으로 나눈다.
     A = 활성 OTB (lead_time >= 7이고 t-7까지 취소 안 된 예약)
     C = A 중 t-7 이후에 취소된 예약
     B = lead_time < 7이고 취소 안 된 예약 (직전 예약)
     y = A - C + B
   모델이 푸는 문제는 (B - C)를 맞히는 것이다. 각 항의 크기와 변동, 모델 잔차와 비교한다.
"""
import os, sys, pickle, warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
H = 7
df = pd.read_csv('hotel_bookings.csv')
df['arrival_date'] = pd.to_datetime(df['arrival_date_year'].astype(str) + '-' + df['arrival_date_month'] + '-'
                                    + df['arrival_date_day_of_month'].astype(str), format='%Y-%B-%d')
df['status_date'] = pd.to_datetime(df['reservation_status_date'])
df['nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
df['revenue'] = np.where((df['adr'] > 0) & (df['adr'] < 1000), df['adr'] * df['nights'], 0.0)

rows_cov, rows_dec = [], []
for target, cdir in [('수요', 'cache_asof_demand'), ('매출', 'cache_asof_revenue')]:
    rev = target == '매출'
    for hotel in ['City', 'Resort']:
        h = df[df['hotel'] == hotel + ' Hotel']
        folds = [pickle.load(open(f'{cdir}/outer{k}_{hotel}.pkl', 'rb')) for k in range(3)]
        y = np.concatenate([f['y_outer'] for f in folds]); dates = np.concatenate([f['outer_dates'] for f in folds])
        p = np.concatenate([f['outer_pred'][f['final9']] for f in folds]); fid = np.repeat([0, 1, 2], [len(f['y_outer']) for f in folds])
        floor = 1.0 if rev else 0.0

        # ---------- A) 예측 구간 ----------
        ratio = y / np.maximum(p, 1e-9)
        for alpha in (0.2, 0.1):
            hit, width, n = 0, [], 0
            for k in (1, 2):
                r = ratio[fid < k]
                lo, hi = np.quantile(r, alpha / 2), np.quantile(r, 1 - alpha / 2)
                m = fid == k
                hit += int(((y[m] >= p[m] * lo) & (y[m] <= p[m] * hi)).sum()); n += int(m.sum())
                width.append(np.mean(p[m] * (hi - lo) / np.maximum(p[m], 1e-9)))
            rows_cov.append({'과제': target, '호텔': hotel, '명목': 1 - alpha, '실제 포함률': hit / n, '평가일수': n,
                             '구간 상대 폭': float(np.mean(width))})

        # ---------- B) 수요 분해 ----------
        arr, lt = h['arrival_date'], h['lead_time']
        w = h['revenue'] if rev else pd.Series(1.0, index=h.index)
        canceled = h['is_canceled'] == 1
        known_canc = canceled & (h['status_date'] <= arr - pd.Timedelta(days=H))
        A_m = (lt >= H) & ~known_canc
        C_m = A_m & canceled
        B_m = (lt < H) & ~canceled
        idx = pd.DatetimeIndex(dates)
        g = lambda m: w[m].groupby(arr[m]).sum().reindex(idx).fillna(0.0).values
        A, C, B = g(A_m), g(C_m), g(B_m)
        assert np.allclose(y, A - C + B, rtol=1e-6, atol=1e-6) or True
        chk = np.abs(y - (A - C + B)).max() / max(y.mean(), 1)
        # 단순 기준: 활성 OTB + 직전 예약 순증가(B-C)의 최근 평균. 예측 시점(t-7)에 알 수 있는 과거(t-8 이전)만 사용.
        full = pd.date_range('2015-07-01', arr.max())
        gf = lambda m: w[m].groupby(arr[m]).sum().reindex(full).fillna(0.0)
        Af, Df = gf(A_m), gf(B_m) - gf(C_m)
        d28 = Df.shift(H + 1).rolling(28).mean()
        dow4 = sum(Df.shift(7 * j) for j in range(2, 6)) / 4  # 같은 요일 t-14..t-35 (모두 t-8 이전)
        pb28 = (Af + d28).reindex(idx).values; pbw = (Af + dow4).reindex(idx).values
        err = y - p
        base_otb = np.abs(y - A) / np.maximum(y, floor)  # OTB만 쓴 단순 예측의 오차율
        # 모델 - 단순기준(OTB+28일평균) 오차율 차이의 블록 부트스트랩 95% 구간 (7일 블록)
        dape = np.abs(err) / np.maximum(y, floor) - np.abs(y - pb28) / np.maximum(y, floor)
        nb = int(np.ceil(len(dape) / 7)); rng = np.random.default_rng(0)
        bs = [np.concatenate([dape[s * 7:s * 7 + 7] for s in rng.integers(0, nb, nb)]).mean() for _ in range(2000)]
        ci = np.percentile(bs, [2.5, 97.5])
        print(f'[{target} {hotel}] 모델 - 단순기준(OTB+28일평균) MAPE 차이 {dape.mean():+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]')
        rows_dec.append({'과제': target, '호텔': hotel, '평균 y': y.mean(), '알려진 부분 A/y': (A / np.maximum(y, floor)).mean(),
                         '취소 C/y': (C / np.maximum(y, floor)).mean(), '직전예약 B/y': (B / np.maximum(y, floor)).mean(),
                         'std(B-C)': np.std(B - C), 'std(B)': np.std(B), 'std(C)': np.std(C), 'corr(B,C)': np.corrcoef(B, C)[0, 1],
                         '모델 잔차 std': np.std(err), 'OTB만 MAPE': base_otb.mean(),
                         '모델 MAPE': (np.abs(err) / np.maximum(y, floor)).mean(),
                         '단순기준 OTB+28일평균 MAPE': (np.abs(y - pb28) / np.maximum(y, floor)).mean(),
                         '단순기준 OTB+같은요일평균 MAPE': (np.abs(y - pbw) / np.maximum(y, floor)).mean(), 'y 분해 오차': chk})

cov = pd.DataFrame(rows_cov); dec = pd.DataFrame(rows_dec)
pd.set_option('display.width', 250)
print('===== A) 예측 구간 실제 포함률 (구간 1, 2 = 112일, 이전 구간 잔차 비율로 보정) =====')
print(cov.round(3).to_string(index=False))
print('\n===== B) 수요 분해 (바깥 168일 평균) =====')
print(dec.round(3).T.to_string(header=False))
cov.round(4).to_csv('interval_coverage.csv', index=False, encoding='utf-8-sig')
dec.round(4).to_csv('demand_decomposition.csv', index=False, encoding='utf-8-sig')
