"""asof_nested.py 결과 집계: 기존 정의(Part 3-1, 4-1) vs 운영 시점(as-of) OTB, 확장 모델 비교."""
import os, sys, pickle
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
pd.set_option('display.width', 250)
HOTELS = ['City', 'Resort']
ORIG9 = ['계절naive(lag_14)', 'ETS(감쇠추세)', 'SARIMAX+OTB', 'Ridge', 'RandomForest', 'HistGB', None, 'LightGBM(튜닝)', '블렌드(LGBM+Ridge)']
EXTRA_MODELS = ['Theta', 'SARIMAX+OTB(조건부c)', 'SARIMAX 차수(2,0,1)(1,0,1)', 'ExtraTrees', 'XGBoost', 'CatBoost']
ABL = ['SARIMAX[OTB만]', 'SARIMAX[exp만]', 'SARIMAX[exp_c만]']
rng = np.random.default_rng(0)

def block_ci(diff, block=7, B=3000):
    n = len(diff); nb = int(np.ceil(n / block)); means = []
    for _ in range(B):
        starts = rng.integers(0, n - block + 1, nb)
        means.append(np.concatenate([diff[s:s + block] for s in starts])[:n].mean())
    return np.percentile(means, [2.5, 97.5])

def verdict(diff):
    lo, hi = block_ci(diff)
    return lo, hi, ('유의한 개선' if hi < 0 else ('유의한 악화' if lo > 0 else '불확실'))

def report(target):
    rev = target == 'revenue'
    legacy_dir = 'cache_rev' if rev else 'cache_nested'
    new_dir = f'cache_asof_{target}'
    part = 'Part4' if rev else 'Part3'
    lgb0 = f'LightGBM({part} 설정)'
    names9 = [lgb0 if n is None else n for n in ORIG9]
    mape = (lambda y, p: np.mean(np.abs(y - p) / np.maximum(y, 1))) if rev else (lambda y, p: np.mean(np.abs(y - p) / y))
    wape = lambda y, p: np.abs(y - p).sum() / y.sum()
    L = {(k, h): pickle.load(open(f'{legacy_dir}/outer{k}_{h}.pkl', 'rb')) for k in range(4) for h in HOTELS}
    A = {(k, h): pickle.load(open(f'{new_dir}/outer{k}_{h}.pkl', 'rb')) for k in range(4) for h in HOTELS}
    rows, sel_rows, out = [], [], {}
    for h in HOTELS:
        ys = np.concatenate([A[(k, h)]['y_outer'] for k in range(3)])
        ys_l = np.concatenate([L[(k, h)]['y_outer'] for k in range(3)])
        assert np.allclose(ys, ys_l), '정답이 달라졌습니다'
        fold = np.concatenate([np.full(len(A[(k, h)]['y_outer']), k) for k in range(3)])
        # 기존 정의 예측 (위치로 이름 매핑)
        leg = {}
        for i, n in enumerate(names9):
            leg[n] = np.concatenate([L[(k, h)]['outer_pred'][list(L[(k, h)]['inner_mape'])[i]] for k in range(3)])
        leg_sel = np.concatenate([L[(k, h)]['outer_pred'][L[(k, h)]['final']] for k in range(3)])
        new = {n: np.concatenate([A[(k, h)]['outer_pred'][n] for k in range(3)]) for n in names9 + EXTRA_MODELS + ABL}
        sel9 = np.concatenate([A[(k, h)]['outer_pred'][A[(k, h)]['final9']] for k in range(3)])
        selall = np.concatenate([A[(k, h)]['outer_pred'][A[(k, h)]['final_all']] for k in range(3)])
        out[h] = dict(ys=ys, leg=leg, leg_sel=leg_sel, new=new, sel9=sel9, selall=selall, fold=fold)
        for n in names9:
            rows.append({'hotel': h, '모델': n, '구분': '기존9', '기존 정의 MAPE': mape(ys, leg[n]), '운영 시점 MAPE': mape(ys, new[n]),
                         '운영 시점 WAPE': wape(ys, new[n]), '운영 시점 편향': float(np.mean(new[n] - ys)),
                         **{f'구간{k+1}': mape(ys[fold == k], new[n][fold == k]) for k in range(3)}})
        for n in EXTRA_MODELS + ABL:
            rows.append({'hotel': h, '모델': n, '구분': '추가' if n in EXTRA_MODELS else '외생변수 ablation', '기존 정의 MAPE': np.nan,
                         '운영 시점 MAPE': mape(ys, new[n]), '운영 시점 WAPE': wape(ys, new[n]), '운영 시점 편향': float(np.mean(new[n] - ys)),
                         **{f'구간{k+1}': mape(ys[fold == k], new[n][fold == k]) for k in range(3)}})
        for lab, pr in [('선택 절차 A, 기존9 (운영 시점)', sel9), ('선택 절차 A, 전체 모델 (운영 시점)', selall), ('선택 절차 A (기존 정의)', leg_sel)]:
            sel_rows.append({'hotel': h, '구분': lab, 'MAPE': mape(ys, pr), 'WAPE': wape(ys, pr), '편향': float(np.mean(pr - ys)),
                             **{f'구간{k+1}': mape(ys[fold == k], pr[fold == k]) for k in range(3)}})
    res = pd.DataFrame(rows); sel = pd.DataFrame(sel_rows)
    for h in HOTELS:
        print(f"\n===== [{target}] {h}: 바깥 168일 MAPE (기존 정의 vs 운영 시점) =====")
        print(res[res.hotel == h].drop(columns='hotel').sort_values('운영 시점 MAPE').round(4).to_string(index=False))
        print(sel[sel.hotel == h].drop(columns='hotel').round(4).to_string(index=False))
        o = out[h]; ys = o['ys']
        ape = lambda pr: np.abs(ys - pr) / (np.maximum(ys, 1) if rev else ys)
        print('  -- 비교 (MAPE 차이, 음수가 개선) --')
        for lab, a, bb in [('선택A(전체모델,운영시점) vs 계절naive', o['selall'], o['new']['계절naive(lag_14)']),
                           ('선택A(전체모델,운영시점) vs LightGBM(기존 설정)', o['selall'], o['new'][lgb0]),
                           ('선택A(기존9,운영시점) vs 선택A(기존 정의)', o['sel9'], o['leg_sel']),
                           ('선택A(전체모델) vs 선택A(기존9) [운영시점]', o['selall'], o['sel9'])]:
            diff = ape(a) - ape(bb); lo, hi, v = verdict(diff)
            print(f'  {lab:<46} {diff.mean():+.4f}  95%CI [{lo:+.4f}, {hi:+.4f}]  {v}')
        print('  선택된 모델 (바깥 구간별) 기존9:', [A[(k, h)]['final9'] for k in range(3)], '| 전체:', [A[(k, h)]['final_all'] for k in range(3)],
              '| 기존 정의:', [L[(k, h)]['final'] for k in range(3)])
        print('  배포용(전체 데이터) 선택: 기존9', A[(3, h)]['final9'], '| 전체', A[(3, h)]['final_all'])
        print('  채택된 EDA 그룹:', [A[(k, h)]['groups'] for k in range(4)])
        # 외생변수 ablation 유의성
        e = o['new']; cur = e['SARIMAX+OTB']
        for nm in ABL + ['SARIMAX+OTB(조건부c)']:
            diff = ape(e[nm]) - ape(cur); lo, hi, v = verdict(diff)
            print(f'  {nm:<26} vs SARIMAX+OTB(exp 사용)  {diff.mean():+.4f}  [{lo:+.4f}, {hi:+.4f}]  {v}')
    res.round(4).to_csv(f'asof_{target}_results.csv', index=False, encoding='utf-8-sig')
    sel.round(4).to_csv(f'asof_{target}_selection.csv', index=False, encoding='utf-8-sig')

for t in (sys.argv[1:] or ['demand', 'revenue']):
    report(t)
