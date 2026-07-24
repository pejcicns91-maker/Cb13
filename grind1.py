# W1 GRIND v2 — grind unchanged; finalize rebuilt memory-lean (no wide-table load,
# int32 preallocation, staged prints). Finalize emits w1_digest.csv, w1_extinction.csv,
# w1_summary.csv ONLY; full register is regenerable from committed out_counts (stated law).
# Seed 20260723. n>=40 floor, BH q=.10/family over the finalized set, extinction logged.
import numpy as np, pandas as pd, json, os, time, glob, hashlib, argparse
from itertools import combinations
np.random.seed(20260723)
ap = argparse.ArgumentParser()
ap.add_argument("--budget-min", type=float, default=230)
ap.add_argument("--b4", type=int, default=0)
ap.add_argument("--finalize", type=int, default=0)
A = ap.parse_args()
OUT = "out_counts"; os.makedirs(OUT, exist_ok=True)
SF = "gha_state.json"

def combo_list(b4):
    cs = [(5, c) for c in combinations(range(25), 5)]
    NEWT = set(range(25, 60))
    for k in range(1, 4):
        cs += [(k, c) for c in combinations(range(60), k) if set(c) & NEWT]
    if b4:
        cs += [(4, c) for c in combinations(range(60), 4) if set(c) & NEWT]
    return cs

if A.finalize:
    from scipy import stats
    from scipy.stats import binom
    ncmb = len(combo_list(A.b4))
    st = json.load(open(SF))
    assert st["cursor"] >= ncmb, f"grind incomplete: {st['cursor']}/{ncmb} — finalize refused"
    B2 = pd.read_parquet("bf_vantage_ALL_wide.parquet", columns=['fwd_favU','fwd_advU'])
    base_b = float(((B2.fwd_favU>=0.25)&(B2.fwd_advU<0.25)).mean())
    base_t = float((B2.fwd_advU>=0.6).mean()); del B2
    print(f"stage 1: bases {base_b:.4f}/{base_t:.4f}", flush=True)
    import pyarrow.parquet as pq
    parts = sorted(glob.glob(f"{OUT}/cnt_*.parquet"))
    sizes = [pq.ParquetFile(p).metadata.num_rows for p in parts]
    tot = sum(sizes)
    kb = np.empty(tot, np.int32); kt = np.empty(tot, np.int32)
    n_ = np.empty(tot, np.int32); dep = np.empty(tot, np.int8)
    off = 0
    for p, sz in zip(parts, sizes):
        B = pd.read_parquet(p, columns=['depth','kb','kt','n'])
        kb[off:off+sz] = B.kb; kt[off:off+sz] = B.kt
        n_[off:off+sz] = B.n; dep[off:off+sz] = B.depth; off += sz; del B
    print(f"stage 2: loaded {tot:,} cells from {len(parts)} parts", flush=True)
    def pvec(k, n, p0):
        out = np.empty(len(k)); o = np.argsort(n, kind='stable')
        ns = n[o]; ks = k[o]
        uq, a = np.unique(ns, return_index=True); b = np.append(a[1:], len(ns))
        for nv, i0, i1 in zip(uq, a, b):
            pmf = binom.pmf(np.arange(int(nv)+1), int(nv), p0)
            d = pmf[ks[i0:i1]] * (1+1e-7)
            sp = np.sort(pmf); cs = np.cumsum(sp)
            pos = np.searchsorted(sp, d, side='right')
            out[o[i0:i1]] = np.minimum(np.where(pos>0, cs[np.maximum(pos-1,0)], 0.0), 1.0)
        return out
    p_b = pvec(kb, n_, base_b); print("stage 3: p bounce done", flush=True)
    p_t = pvec(kt, n_, base_t); print("stage 4: p through done", flush=True)
    def bh(ps, q=0.10):
        m = len(ps); o = np.argsort(ps, kind='stable')
        sat = np.nonzero(ps[o] <= q*np.arange(1, m+1)/m)[0]
        kmax = int(sat[-1]+1) if len(sat) else 0
        ok = np.zeros(m, bool); ok[o[:kmax]] = True; return ok
    cb = bh(p_b); ct = bh(p_t); print("stage 5: BH done", flush=True)
    rng = np.random.default_rng(20260723)
    ix = rng.choice(tot, 200, replace=False)
    md = max(max(abs(stats.binomtest(int(kb[i]),int(n_[i]),base_b).pvalue-p_b[i]) for i in ix),
             max(abs(stats.binomtest(int(kt[i]),int(n_[i]),base_t).pvalue-p_t[i]) for i in ix))
    print(f"stage 6: scipy check {md:.2e}", flush=True)
    digest = []
    off = 0
    for p, sz in zip(parts, sizes):
        B = pd.read_parquet(p); s = slice(off, off+sz); off += sz
        for fam, kv, cc, bb in [('bounce', B.kb, cb[s], base_b), ('through', B.kt, ct[s], base_t)]:
            m = cc & (B.n.values >= 100)
            if not m.any(): continue
            D = B[m].copy()
            D['family'] = fam; D['rate'] = np.round(kv[m]/D.n, 3); D['base'] = round(bb, 3)
            D['lift'] = (D.rate - D.base).abs()
            digest.append(D.nlargest(min(800, len(D)), 'lift')
                          [['family','depth','f1','v1','f2','v2','rate','base','n','lift']])
        del B
    DG = pd.concat(digest, ignore_index=True)
    DG = pd.concat([DG[(DG.family==f)&(DG.depth==d)].nlargest(200,'lift')
                    for f in ['bounce','through'] for d in sorted(set(dep.tolist()))],
                   ignore_index=True)
    DG.drop(columns=['lift']).to_csv("w1_digest.csv", index=False)
    print("stage 7: digest done", flush=True)
    E = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(f"{OUT}/ext_*.parquet"))],
                  ignore_index=True)
    E['extinct'] = E.cells == 0; E.to_csv("w1_extinction.csv", index=False)
    rows = []
    for d in sorted(set(dep.tolist())):
        m = dep == d
        rows.append(dict(depth=int(d), cells=int(m.sum()),
                         cert_bounce=int(cb[m].sum()), cert_through=int(ct[m].sum())))
    S = pd.DataFrame(rows); S.to_csv("w1_summary.csv", index=False)
    print(f"FINALIZED | cells/fam {tot:,} | cert b {int(cb.sum()):,} t {int(ct.sum()):,} | "
          f"extinct {int(E.extinct.sum())}/{len(E)} | pval {md:.2e}", flush=True)
    for f in ["w1_digest.csv","w1_extinction.csv","w1_summary.csv"]:
        print(f, "sha", hashlib.sha256(open(f,'rb').read()).hexdigest()[:12])
    raise SystemExit(0)

# ---------------- grind mode (unchanged behavior) ----------------
V = pd.read_parquet("bf_vantage_ALL_wide.parquet")
V['bounce'] = (V.fwd_favU >= 0.25) & (V.fwd_advU < 0.25)
V['through'] = V.fwd_advU >= 0.6
for col, q in [('dq','distU'),('sq','speedUh'),('vq','relvol'),('wq','widthU'),
               ('rq','rng_used'),('kq','wk_used'),('uq','u_trend')]:
    try: V[col] = pd.qcut(pd.to_numeric(V[q], errors='coerce'), 4, duplicates='drop')
    except Exception: V[col] = np.nan
V['cb'] = pd.cut(V.contact, [-1,30,49.5,101], labels=['c<30','c30-49','c>=50'])
V['tn'] = pd.cut(pd.to_numeric(V.test_no, errors='coerce'), [0,1,2,99], labels=['t1','t2','t3+'])
FE25 = ['coin','etype','zone','virgin','cb','wq','session','wknd','hayden','hayden_btc','btc_pi',
        'origin','tn','dq','sq','vq','rq','kq','uq','yd_arch','ob55','dtype','lean','scen_state','scen_failed']
T2 = [f"{b}_{f}" for b in ['net100','net20','rng100','pos100','volr','zt100','zlast']
      for f in ['5m','15m','1h','4h','1d']]
for c in T2:
    try: V['q_'+c] = pd.qcut(pd.to_numeric(V[c], errors='coerce'), 4, duplicates='drop')
    except Exception: V['q_'+c] = np.nan
FE = FE25 + ['q_'+c for c in T2]
codes = {}; labels = {}
for f in ['station'] + FE:
    s = V[f]
    if isinstance(s.dtype, pd.CategoricalDtype):
        codes[f] = s.cat.codes.to_numpy(np.int32); labels[f] = [str(x) for x in s.cat.categories]
    else:
        c, u = pd.factorize(s, use_na_sentinel=True)
        codes[f] = c.astype(np.int32); labels[f] = [str(x) for x in u]
card = {f: len(labels[f]) for f in codes}
b_arr = V.bounce.to_numpy(bool); t_arr = V.through.to_numpy(bool)
combos = combo_list(A.b4)
print("total combos this configuration:", len(combos), flush=True)
def process(depth, combo):
    cols = ['station'] + [FE[i] for i in combo]
    cm = np.stack([codes[c] for c in cols]); mask = (cm >= 0).all(axis=0)
    dims = [card[c] for c in cols]
    idx = np.ravel_multi_index(tuple(cm[:, mask]), dims)
    n2 = np.bincount(idx, minlength=int(np.prod(dims)))
    sel = np.nonzero(n2 >= 40)[0]
    er = (depth, '|'.join(cols[1:]), len(sel))
    if not len(sel): return [], er
    kb2 = np.bincount(idx[b_arr[mask]], minlength=len(n2))[sel]
    kt2 = np.bincount(idx[t_arr[mask]], minlength=len(n2))[sel]
    uidx = np.unravel_index(sel, dims)
    lab = [np.array(labels[c], dtype=object)[uidx[d]] for d, c in enumerate(cols)]
    head = ['|'.join(x) for x in zip(*[lab[d] for d in range(len(cols)-1)])]
    f1 = '|'.join(cols[:-1]); f2 = cols[-1]
    return [(depth, f1, head[r], f2, lab[-1][r], int(kb2[r]), int(kt2[r]), int(n2[sel][r]))
            for r in range(len(sel))], er
st = json.load(open(SF)) if os.path.exists(SF) else {"cursor": 0, "b4": A.b4}
cur = st["cursor"]; BLOCK = 200; t0 = time.time()
while cur < len(combos) and (time.time() - t0) / 60 < A.budget_min:
    rows, ext = [], []
    for depth, combo in combos[cur:cur+BLOCK]:
        rr, er = process(depth, combo); rows += rr; ext.append(er)
    tag = f"{cur:07d}"
    pd.DataFrame(rows, columns=['depth','f1','v1','f2','v2','kb','kt','n']) \
      .to_parquet(f"{OUT}/cnt_{tag}.parquet", index=False, compression='zstd')
    pd.DataFrame(ext, columns=['depth','combo','cells']).to_parquet(f"{OUT}/ext_{tag}.parquet", index=False)
    cur += min(BLOCK, len(combos) - cur)
    json.dump({"cursor": cur, "b4": A.b4}, open(SF, 'w'))
    print(f"cursor {cur}/{len(combos)}  {(time.time()-t0)/60:.1f}min", flush=True)
print("chunk done; complete" if cur >= len(combos) else "chunk done; resume next run", flush=True)
