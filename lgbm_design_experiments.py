"""
Part 5-4. LightGBM 피처, 학습 설계 개선 실험 (7일 전 수요, 매출). 사후 분석.

[사전 지정: 아래 후보 목록, 순서, 판정 규칙은 결과를 보기 전에 확정했고 실행 중 바꾸지 않는다]

기준 모델 (a): pickup_features.py의 level 기본 LightGBM (고정 설정, 시드 3개 평균, 300 round)
  피처: dow, month, lag_s(y 8일 전), lag_w, roll7, roll28, A(활성 OTB), b0~b3(A의 리드타임 구간별)
  타깃: y 직접 예측 (L2 손실)
비교 (b): 단순 기준 = A + 직전 28일 평균 순증(y-A).  비교 (c): Ridge(alpha=10), 같은 기본 피처.
평가: 예측 시점 7일, as-of 정의(취소일 <= 투숙일-7인 예약은 A에서 제외), 바깥 3구간 168일, 구간마다 그 이전 데이터로만 학습,
      7일 블록 부트스트랩 95% CI (2000회). 모든 후보가 같은 평가일을 쓴다.
      (학습 행은 구간 직전 날짜까지의 y를 쓴다. 기존 horizon_sensitivity.py, pickup_features.py와 같은 설계라 결과를 비교할 수 있다.)

후보 (이 순서, 하나씩 기준 모델에 더함)
  1 stack  : 예측 시점 H in {3,5,7,10,14}로 행을 쌓고 H를 피처로 넣음 (평가는 H=7만)
  2 joint  : City, Resort 합동 학습 + 호텔 구분 피처
  3 init   : init_score = A (잔차 학습). 로그 링크 손실이면 log(max(A,1))
  4 glm    : 수요는 Poisson 손실, 매출은 Tweedie(분산 지수 1.5) 손실
  5 decay  : 최근 데이터 가중치, 지수 감쇠 반감기 90일 (한 값만)
  6 nbr    : 이웃 날짜 A (D-1, D+1, 같은 주(월~일) 합). 모두 D-7 시점에 알 수 있는 값(이웃 날짜의 예약 중 D-7까지 잡힌 것)
  7 calendar: 건너뜀. Part 3-1, 5의 calendar 그룹(공휴일 여부, 다음, 직전 공휴일까지 일수, 방문국 공휴일 수, 연중 주차)이
            이미 후보였고 채택 기준 미달 (연휴 길이, 방학은 없음)
  8 comp   : A 구성. 리드타임 60일 이상 비중, Groups 예약 건수
  9 real   : 실현된 오차(y-A) 피처. 7일 이상 지난 날의 최근 7, 14, 28일 평균, 같은 요일 4주 평균, 추세(7일 평균 - 28일 평균)
조합: 아래 규칙으로 "효과 있음"이 된 후보만 묶어 1회 평가.
  효과 있음 = 4개 셀(호텔 x 과제) 중 2개 이상에서 기준 대비 MAPE 차이의 CI 상한 < 0 이고, CI 하한 > 0 인 셀이 없음.
다중 비교: 후보 8개 x 셀 4개. 하나만 유의하면 우연일 수 있다.
"""
import sys, warnings, math; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge

sys.stdout.reconfigure(encoding='utf-8')
TEST_LEN, N_OUTER, EVAL_H = 56, 3, 7
HALFLIFE = 90
STACK_HS = [3, 5, 7, 10, 14]

df = pd.read_csv('hotel_bookings.csv')
df['arrival_date'] = pd.to_datetime(df['arrival_date_year'].astype(str) + '-' + df['arrival_date_month'] + '-'
                                    + df['arrival_date_day_of_month'].astype(str), format='%Y-%B-%d')
df['status_date'] = pd.to_datetime(df['reservation_status_date'])
df['nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
df['revenue'] = np.where((df['adr'] > 0) & (df['adr'] < 1000), df['adr'] * df['nights'], 0.0)

BASE = ['dow', 'month', 'lag_s', 'lag_w', 'roll7', 'roll28', 'A', 'b0', 'b1', 'b2', 'b3']
GCOLS = {'nbr': ['A_prev', 'A_next', 'A_week'], 'comp': ['lead60_share', 'groups_cnt'],
         'real': ['D_roll7', 'D_roll14', 'D_roll28', 'D_dow4', 'D_trend']}
BASEP = {'objective': 'regression', 'metric': 'l1', 'verbose': -1, 'num_leaves': 15, 'learning_rate': 0.05,
         'feature_fraction': 0.8, 'min_child_samples': 10, 'bagging_fraction': 0.8, 'bagging_freq': 1}


class Ctx:
    def __init__(self, hotel, rev):
        self.h = df[df['hotel'] == hotel]; self.rev = rev
        self.idx = pd.date_range('2015-08-01', self.h['arrival_date'].max())
        self.w = self.h['revenue'] if rev else pd.Series(1.0, index=self.h.index)
        kept = self.h['is_canceled'] == 0
        self.y = self.w[kept].groupby(self.h.loc[kept, 'arrival_date']).sum().reindex(self.idx).fillna(0.0)
        self._parts, self._rows = {}, {}

    def parts(self, hh):
        """예측 시점이 투숙 hh일 전일 때의 활성 OTB(A)와 구성."""
        if hh in self._parts: return self._parts[hh]
        h = self.h
        A_m = (h['lead_time'] >= hh) & ~((h['is_canceled'] == 1) & (h['status_date'] <= h['arrival_date'] - pd.Timedelta(days=hh)))
        s = h[A_m]; ws = self.w[A_m]; out = pd.DataFrame(index=self.idx)
        edges = [(hh, hh + 7), (hh + 7, hh + 30), (hh + 30, hh + 90), (hh + 90, 10 ** 6)]
        for i, (lo, hi) in enumerate(edges):
            m = (s['lead_time'] >= lo) & (s['lead_time'] < hi)
            out[f'b{i}'] = ws[m].groupby(s.loc[m, 'arrival_date']).sum().reindex(self.idx).fillna(0.0).values
        out['A'] = out[['b0', 'b1', 'b2', 'b3']].sum(axis=1)
        g = s.groupby('arrival_date')
        out['cnt'] = g.size().reindex(self.idx).fillna(0).values
        out['lead60'] = (s['lead_time'] >= 60).groupby(s['arrival_date']).sum().reindex(self.idx).fillna(0).values
        out['groups'] = (s['market_segment'] == 'Groups').groupby(s['arrival_date']).sum().reindex(self.idx).fillna(0).values
        self._parts[hh] = out; return out

    def rows(self, H, groups):
        key = (H, tuple(sorted(groups)))
        if key in self._rows: return self._rows[key]
        p = self.parts(H); idx = self.idx; sh = H + 1; LW = 7 * int(math.ceil((H + 1) / 7))
        d = pd.DataFrame({'y': self.y.values}, index=idx)
        d['dow'] = idx.dayofweek; d['month'] = idx.month
        d['lag_s'] = d['y'].shift(sh); d['lag_w'] = d['y'].shift(LW)
        d['roll7'] = d['y'].shift(sh).rolling(7).mean(); d['roll28'] = d['y'].shift(sh).rolling(28).mean()
        for c in ['b0', 'b1', 'b2', 'b3', 'A']: d[c] = p[c].values
        D = d['y'] - d['A']
        d['D_roll7'] = D.shift(sh).rolling(7).mean(); d['D_roll14'] = D.shift(sh).rolling(14).mean()
        d['D_roll28'] = D.shift(sh).rolling(28).mean()
        d['D_dow4'] = sum(D.shift(LW + 7 * j) for j in range(4)) / 4
        d['D_trend'] = d['D_roll7'] - d['D_roll28']
        if 'nbr' in groups:
            # 기준일 D의 예측 시점은 D-H. 이웃 날짜 D'의 A는 같은 시점(D-H)까지 잡힌 예약이므로 D' 기준 horizon = H + (D'-D)
            d['A_prev'] = self.parts(max(H - 1, 1))['A'].shift(1).values
            d['A_next'] = self.parts(H + 1)['A'].shift(-1).values
            wk = np.full(len(idx), np.nan)
            for w in range(7):
                m = (idx.dayofweek == w)
                tot = sum(self.parts(max(H + j - w, 1))['A'].shift(-(j - w)).values for j in range(7))
                wk[m] = tot[m]
            d['A_week'] = wk
        if 'comp' in groups:
            d['lead60_share'] = (p['lead60'] / p['cnt'].clip(lower=1)).values; d['groups_cnt'] = p['groups'].values
        d = d.dropna(subset=['lag_s', 'lag_w', 'roll7', 'roll28', 'D_roll28', 'D_dow4'])  # 모든 후보가 같은 행 집합
        self._rows[key] = d; return d


def feats_of(spec):
    f = list(BASE)
    for g in spec['groups']: f += GCOLS[g]
    if spec['stack']: f += ['Hh']
    if spec['joint']: f += ['hotel']
    return f


def lgb_fit_predict(Xtr, ytr, Atr, wtr, Xte, Ate, spec, rev):
    loss = 'l2' if not spec['glm'] else ('tweedie' if rev else 'poisson'); log_link = loss != 'l2'
    out = []
    for sd in (0, 1, 2):
        p = dict(BASEP, seed=sd)
        if loss == 'poisson': p['objective'] = 'poisson'
        if loss == 'tweedie': p.update(objective='tweedie', tweedie_variance_power=1.5)
        init_tr = (np.log(np.maximum(Atr, 1)) if log_link else Atr) if spec['init'] else None
        m = lgb.train(p, lgb.Dataset(Xtr, label=ytr, weight=wtr, init_score=init_tr), num_boost_round=300)
        if spec['init']:
            init_te = np.log(np.maximum(Ate, 1)) if log_link else Ate
            raw = m.predict(Xte, raw_score=True) + init_te; out.append(np.exp(raw) if log_link else raw)
        else:
            out.append(m.predict(Xte))
    return np.mean(out, axis=0)


def ridge_pred(tr, te, feats):
    num = [c for c in feats if c not in ('dow', 'month')]
    pre = ColumnTransformer([('n', Pipeline([('i', SimpleImputer(strategy='median')), ('s', StandardScaler())]), num),
                             ('c', OneHotEncoder(handle_unknown='ignore'), ['dow', 'month'])])
    return Pipeline([('p', pre), ('r', Ridge(alpha=10.0))]).fit(tr[feats], tr['y']).predict(te[feats])


def boot_ci(x, B=2000, seed=0):
    rng = np.random.default_rng(seed); nb = int(np.ceil(len(x) / 7))
    bs = [np.concatenate([x[s * 7:s * 7 + 7] for s in rng.integers(0, nb, nb)]).mean() for _ in range(B)]
    return np.percentile(bs, [2.5, 97.5])


def run_spec(ctx, rev, spec):
    """ctx: {'City': Ctx, 'Resort': Ctx}. 반환: {호텔: 바깥 168일 예측}."""
    feats = feats_of(spec); res = {h: [] for h in ctx}
    ref = ctx['City'].rows(EVAL_H, ())  # 평가일 기준 (모든 후보 동일)
    N = len(ref)
    for k in range(N_OUTER):
        ts = N - (N_OUTER - k) * TEST_LEN; te_ = ts + TEST_LEN
        ts_date = ref.index[ts]; te_dates = ref.index[ts:te_]
        def train_part(hotel):
            parts = []
            for H in (STACK_HS if spec['stack'] else [EVAL_H]):
                d = ctx[hotel].rows(H, spec['groups']); d = d[d.index < ts_date].copy()
                d['Hh'] = H; d['hotel'] = 0 if hotel == 'City' else 1
                parts.append(d)
            return pd.concat(parts)
        def test_part(hotel):
            d = ctx[hotel].rows(EVAL_H, spec['groups']).loc[te_dates].copy(); d['Hh'] = EVAL_H; d['hotel'] = 0 if hotel == 'City' else 1
            return d
        def weights(tr):
            return None if not spec['decay'] else 0.5 ** (((ts_date - tr.index).days.values) / HALFLIFE)
        if spec['joint']:
            tr = pd.concat([train_part(h) for h in ctx]); w = weights(tr)
            for h in ctx:
                te = test_part(h); res[h].append(lgb_fit_predict(tr[feats], tr['y'].values, tr['A'].values, w, te[feats], te['A'].values, spec, rev))
        else:
            for h in ctx:
                tr = train_part(h); te = test_part(h); w = weights(tr)
                res[h].append(lgb_fit_predict(tr[feats], tr['y'].values, tr['A'].values, w, te[feats], te['A'].values, spec, rev))
    return {h: np.concatenate(v) for h, v in res.items()}


NOSPEC = dict(stack=False, joint=False, init=False, glm=False, decay=False, groups=())
CANDS = [('1 stack', dict(stack=True)), ('2 joint', dict(joint=True)), ('3 init', dict(init=True)), ('4 glm', dict(glm=True)),
         ('5 decay', dict(decay=True)), ('6 nbr', dict(groups=('nbr',))), ('8 comp', dict(groups=('comp',))), ('9 real', dict(groups=('real',)))]

rows = []
for task, rev in [('수요', False), ('매출', True)]:
    ctx = {h: Ctx(h + ' Hotel', rev) for h in ('City', 'Resort')}
    floor = 1.0 if rev else 0.0
    ref = ctx['City'].rows(EVAL_H, ()); N = len(ref)
    test_idx = ref.index[N - N_OUTER * TEST_LEN:]
    y = {h: ctx[h].rows(EVAL_H, ()).loc[test_idx, 'y'].values for h in ctx}
    ape = lambda h, p, y=y, floor=floor: np.abs(y[h] - p) / np.maximum(y[h], floor)  # 루프 변수가 바뀌어도 이 과제의 y, floor를 쓰도록 묶음
    # (b) 단순 기준, (c) Ridge
    simple = {h: ctx[h].rows(EVAL_H, ()).loc[test_idx, 'A'].values + ctx[h].rows(EVAL_H, ()).loc[test_idx, 'D_roll28'].values for h in ctx}
    ridge = {}
    for h in ctx:
        d = ctx[h].rows(EVAL_H, ()); out = []
        for k in range(N_OUTER):
            ts = N - (N_OUTER - k) * TEST_LEN
            out.append(ridge_pred(d.iloc[:ts], d.iloc[ts:ts + TEST_LEN], BASE))
        ridge[h] = np.concatenate(out)
    base = run_spec(ctx, rev, NOSPEC)
    preds = {'(a) 기준 LightGBM': base, '(b) 단순 기준': simple, '(c) Ridge': ridge}
    for name, over in CANDS:
        preds[name] = run_spec(ctx, rev, dict(NOSPEC, **over)); print(task, name, '완료', flush=True)
    store = {'ctx': ctx, 'y': y, 'ape': ape, 'preds': preds, 'rev': rev}
    # 조합 판정 (사전 규칙)
    eff = []
    for name, over in CANDS:
        up, lo = 0, 0
        for h in ctx:
            diff = ape(h, preds[name][h]) - ape(h, base[h]); ci = boot_ci(diff)
            up += int(ci[1] < 0); lo += int(ci[0] > 0)
        # 규칙은 두 과제를 합쳐 판정해야 하므로 과제별 집계를 저장
        rows.append({'과제': task, 'name': name, 'up': up, 'lo': lo})
    globals().setdefault('STORE', {})[task] = store

# ---- 효과 판정은 4개 셀(호텔 x 과제) 전체에서 ----
tally = pd.DataFrame(rows).groupby('name')[['up', 'lo']].sum()
eff = [n for n in tally.index if tally.loc[n, 'up'] >= 2 and tally.loc[n, 'lo'] == 0]
print('\n효과 있음 판정 (CI 상한<0인 셀 수, CI 하한>0인 셀 수):'); print(tally.to_string()); print('효과 있음:', eff)

combo_spec = dict(NOSPEC)
for name, over in CANDS:
    if name in eff:
        for k_, v_ in over.items():
            combo_spec[k_] = (tuple(sorted(set(combo_spec[k_]) | set(v_))) if k_ == 'groups' else v_)
for task in STORE:
    S = STORE[task]
    if eff:
        S['preds']['조합'] = run_spec(S['ctx'], S['rev'], combo_spec); print(task, '조합 완료', combo_spec, flush=True)

out_rows = []
for task, S in STORE.items():
    floor = 1.0 if S['rev'] else 0.0
    for h in ('City', 'Resort'):
        a_base = S['ape'](h, S['preds']['(a) 기준 LightGBM'][h]); a_simple = S['ape'](h, S['preds']['(b) 단순 기준'][h])
        for name, p in S['preds'].items():
            a = S['ape'](h, p[h]); d1 = a - a_base; d2 = a - a_simple; c1 = boot_ci(d1); c2 = boot_ci(d2)
            out_rows.append({'과제': task, '호텔': h, '모델': name, 'MAPE': a.mean(), '기준(a) 대비': d1.mean(), 'CI(a) 하한': c1[0], 'CI(a) 상한': c1[1],
                             '단순(b) 대비': d2.mean(), 'CI(b) 하한': c2[0], 'CI(b) 상한': c2[1]})
res = pd.DataFrame(out_rows); res.round(4).to_csv('lgbm_design_results.csv', index=False, encoding='utf-8-sig')
pd.set_option('display.width', 250)
def verdict(lo, hi): return '구분 안 됨' if lo <= 0 <= hi else ('개선' if hi < 0 else '악화')
for (task, h), g in res.groupby(['과제', '호텔'], sort=False):
    g = g.copy()
    g['기준(a) 대비 [95% CI]'] = g.apply(lambda r: f'{r["기준(a) 대비"]:+.4f} [{r["CI(a) 하한"]:+.4f}, {r["CI(a) 상한"]:+.4f}] {verdict(r["CI(a) 하한"], r["CI(a) 상한"])}', axis=1)
    g['단순(b) 대비 [95% CI]'] = g.apply(lambda r: f'{r["단순(b) 대비"]:+.4f} [{r["CI(b) 하한"]:+.4f}, {r["CI(b) 상한"]:+.4f}] {verdict(r["CI(b) 하한"], r["CI(b) 상한"])}', axis=1)
    print(f'\n===== {task} {h} (바깥 168일 MAPE) ====='); print(g[['모델', 'MAPE', '기준(a) 대비 [95% CI]', '단순(b) 대비 [95% CI]']].round(4).to_string(index=False))
