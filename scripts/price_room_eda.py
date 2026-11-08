"""
Part 1-4. 가격대, 룸 유형별 EDA (가볍게).
  1) 가격대(저, 중, 고)와 룸 유형별 취소율
  2) 가격대, 룸 유형별 막판 신규 예약 비중 (lead_time < 7, 취소 안 된 예약 / 취소 안 된 전체)
  3) Resort의 계절별 가격대 구성
가격대는 호텔별 adr 3분위(전체 기간 기준)이며, 계절 효과를 분리하려고 "같은 달 안에서의 상대 가격대"도 함께 본다.
adr은 예약 시점에 합의된 가격이므로 예측 시점에 알 수 있는 값이다.
"""
import sys, warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
pd.set_option('display.width', 220)
df = pd.read_csv('hotel_bookings.csv')
df['arrival_date'] = pd.to_datetime(df['arrival_date_year'].astype(str) + '-' + df['arrival_date_month'] + '-'
                                    + df['arrival_date_day_of_month'].astype(str), format='%Y-%B-%d')
df = df[(df['adr'] > 0) & (df['adr'] < 1000)].copy()
df['month'] = df['arrival_date'].dt.month
df['late'] = (df['lead_time'] < 7).astype(int)
df['kept'] = (df['is_canceled'] == 0).astype(int)
labels = ['저', '중', '고']
for h, g in df.groupby('hotel'):
    df.loc[g.index, 'tier'] = pd.qcut(g['adr'], 3, labels=labels)
    # 같은 호텔, 같은 달 안에서의 상대 가격대
    df.loc[g.index, 'tier_m'] = g.groupby('month')['adr'].transform(lambda s: pd.qcut(s, 3, labels=labels, duplicates='drop')).astype(str)

for h in ['City Hotel', 'Resort Hotel']:
    d = df[df['hotel'] == h]
    print(f'\n==================== {h} (유효 가격 {len(d):,}건) ====================')
    cuts = d.groupby('tier', observed=True)['adr'].agg(['min', 'max', 'median']).round(0)
    print('가격대 경계(adr):'); print(cuts.to_string())

    print('\n[1] 취소율: 전체 가격대 / 같은 달 안 상대 가격대')
    t = pd.DataFrame({'전체기준 취소율': d.groupby('tier', observed=True)['is_canceled'].mean(),
                      '월내 상대 취소율': d.groupby('tier_m')['is_canceled'].mean().reindex(labels)})
    t['건수'] = d.groupby('tier', observed=True).size(); print(t.round(3).to_string())

    print('\n[1b] 룸 유형별 취소율 (건수 500 이상)')
    r = d.groupby('reserved_room_type').agg(건수=('is_canceled', 'size'), 취소율=('is_canceled', 'mean'), adr중앙값=('adr', 'median'))
    print(r[r['건수'] >= 500].round(3).to_string())

    print('\n[2] 막판 신규 예약(리드타임 7일 미만) 비중, 취소 안 된 예약 기준')
    k = d[d['kept'] == 1]
    t2 = pd.DataFrame({'전체기준': k.groupby('tier', observed=True)['late'].mean(), '월내 상대': k.groupby('tier_m')['late'].mean().reindex(labels)})
    t2['평균 리드타임(일)'] = k.groupby('tier', observed=True)['lead_time'].mean(); print(t2.round(3).to_string())
    r2 = k.groupby('reserved_room_type').agg(건수=('late', 'size'), 막판비중=('late', 'mean'), 평균리드타임=('lead_time', 'mean'))
    print('\n룸 유형별:'); print(r2[r2['건수'] >= 500].round(3).to_string())

    print('\n[2b] 시장 세그먼트를 통제한 막판 비중 (월내 상대 가격대 x 세그먼트)')
    seg = k[k['market_segment'].isin(['Online TA', 'Offline TA/TO', 'Direct', 'Groups', 'Corporate'])]
    print(seg.pivot_table(index='market_segment', columns='tier_m', values='late', aggfunc='mean')[labels].round(3).to_string())

d = df[df['hotel'] == 'Resort Hotel']
print('\n==================== [3] Resort 계절별 가격대 구성 (도착 월 기준, 전체 3분위 기준) ====================')
m = d.groupby('month').agg(건수=('adr', 'size'), adr중앙값=('adr', 'median'))
for lb in labels: m[f'{lb} 비중'] = d.assign(x=(d['tier'] == lb)).groupby('month')['x'].mean()
print(m.round(3).to_string())
print('\nResort 룸 유형 구성 (상위 4개, 월별 비중)')
top = d['reserved_room_type'].value_counts().head(4).index
print(pd.crosstab(d['month'], d['reserved_room_type'], normalize='index')[top].round(3).to_string())
print('\nResort 월별 조건: 성수기(7,8월) vs 비수기(11~2월) 가격대별 취소율')
d = d.assign(season=np.where(d['month'].isin([7, 8]), '성수기(7-8월)', np.where(d['month'].isin([11, 12, 1, 2]), '비수기(11-2월)', '그 외')))
print(d.pivot_table(index='season', columns='tier', values='is_canceled', aggfunc='mean', observed=True)[labels].round(3).to_string())
