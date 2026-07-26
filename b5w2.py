# B5-W2 — W1's forecaster (b5.py header = frozen prereg) rerun on the W2-augmented table.
# PREREG REPLICATED VERBATIM FROM b5.py: walk-forward by calendar month of t_station;
# train = all months < m (warmup >= 12 months), test = month m; model =
# sklearn LogisticRegression(C=1.0, max_iter=200, class prior free), seed 20260723;
# targets per frame F in {5m,15m,1h,4h,1d}: bounce_F=(fav_F>=.25)&(adv_F<.25),
# through_F=(adv_F>=.6) on rows where both fields exist; baseline = train-set base rate
# (constant predictor); Brier + skill (1 - Brier/Brier_base) per (frame, family, month).
# No peeking, no refits, no tuning. ONLY THE FEATURE SET CHANGES.
#
# FEATURES = W1's exact set (FE25 one-hot + 35 t2 numerics, median-imputed on TRAIN only,
# standardized on TRAIN only) PLUS the W2 columns encoded per B5_HANDOFF prereg item 2:
#   relation cols                -> one-hot (<=5 levels)
#   swing words (*_psw/*_rsw)    -> PER-POSITION: 4 positional cols x {HH,HL,LH,LL,na}
#   slope words (*_ps8/*_hy8)    -> PER-POSITION: 7 positional cols x {U,F,D,na}
#   reduced words (*_ps2/hy2/hb2)-> one-hot (<=4 levels)
#   divergence cols (dbct/dbsn/dsct/dssn) -> one-hot (4 levels)
#   scalars (*age/*fl/*dom/*prev/*scage/*scfl) -> quartile RANK one-hot, na its own level
# Sparse: scipy CSR built from explicit deterministic integer codes (no hashing, no
# get_dummies(sparse=True)). Seed 20260723 everywhere.
#
# SPARSE-SAFE STANDARDIZATION: b5.py does (X-mu)/sd. Centering a design matrix is a
# no-op for a logistic model with a free, UNPENALIZED intercept (sklearn default): the
# map (b0,b) -> (b0 - sum(bj*muj), b) preserves eta and the L2 penalty on b, so the
# fitted slopes and predicted probabilities are identical. We therefore apply scale-only
# (divide by TRAIN sd + 1e-9), which is mathematically equivalent AND preserves sparsity.
import numpy as np, pandas as pd, json, os, time, argparse, hashlib, re, gc
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression

ap = argparse.ArgumentParser()
ap.add_argument("--budget-min", type=float, default=1e9)
ap.add_argument("--frames", type=str, default="5m,15m,1h,4h,1d")
ap.add_argument("--secondary", type=int, default=0)   # b50/fastres table (separate, never blended)
ap.add_argument("--probe", type=int, default=0)       # time N late months and exit
ap.add_argument("--out", type=str, default="b5w2_scores.csv")
ap.add_argument("--state", type=str, default="b5w2_state.json")
A = ap.parse_args()
np.random.seed(20260723)
SEED = 20260723

# ---------------------------------------------------------------- PREPARE (cached)
CACHE = "b5w2_cache.npz"
T2 = [f"{b}_{f}" for b in ['net100','net20','rng100','pos100','volr','zt100','zlast']
      for f in ['5m','15m','1h','4h','1d']]
FR = {'5m': ('fwd_favU','fwd_advU'), '15m': ('fav_15m','adv_15m'), '1h': ('fav_1h','adv_1h'),
      '4h': ('fav_4h','adv_4h'), '1d': ('fav_1d','adv_1d')}

if os.path.exists("cache_CODES.npy"):
    CODES = np.load("cache_CODES.npy", mmap_mode="r")     # streamed, not resident
    NUM = np.load("cache_NUM.npy")
    Z = np.load("cache_meta.npz", allow_pickle=True)
    LVS = Z["LVS"]; MONTH = Z["MONTH"]; FAV = Z["FAV"]; ADV = Z["ADV"]
    SECB = Z["SECB"]; GNAMES = list(Z["GNAMES"]); del Z
    print("PREPARE: cache loaded (mmap)", flush=True)
elif os.path.exists(CACHE):
    Z = np.load(CACHE, allow_pickle=True)
    CODES = Z["CODES"]; NUM = Z["NUM"]; LVS = Z["LVS"]; MONTH = Z["MONTH"]
    FAV = Z["FAV"]; ADV = Z["ADV"]; SECB = Z["SECB"]; GNAMES = list(Z["GNAMES"])
    print("PREPARE: cache loaded", CACHE, flush=True)
else:
    # ---------------------------------------------------------------- load + W1 features
    V = pd.read_parquet("bf_vantage_ALL_wide.parquet")
    ts = pd.to_datetime(V.t_station, unit="ms", utc=True)
    V["month"] = ts.dt.strftime("%Y-%m")

    FE25 = ['station','coin','etype','zone','virgin','cb_','wq_','session','wknd','hayden','hayden_btc','btc_pi',
            'origin','tn_','dq_','sq_','vq_','rq_','kq_','uq_','yd_arch','ob55','dtype','lean','scen_state','scen_failed']
    for col, q in [('dq_','distU'),('sq_','speedUh'),('vq_','relvol'),('wq_','widthU'),
                   ('rq_','rng_used'),('kq_','wk_used'),('uq_','u_trend')]:
        try: V[col] = pd.qcut(pd.to_numeric(V[q], errors='coerce'), 4, duplicates='drop').astype(str)
        except Exception: V[col] = "na"
    V['cb_'] = pd.cut(V.contact, [-1,30,49.5,101], labels=['c<30','c30-49','c>=50']).astype(str)
    V['tn_'] = pd.cut(pd.to_numeric(V.test_no, errors='coerce'), [0,1,2,99], labels=['t1','t2','t3+']).astype(str)
    for c in FE25: V[c] = V[c].astype(str).fillna("na")

    T2 = [f"{b}_{f}" for b in ['net100','net20','rng100','pos100','volr','zt100','zlast']
          for f in ['5m','15m','1h','4h','1d']]
    NUM = V[T2].to_numpy(np.float32)

    # ---------------------------------------------------------------- W2 table + provenance
    W = pd.read_parquet("bf_w2cols.parquet")
    assert len(W) == len(V), "row count mismatch W2 vs wide"
    # bf_w2cols carries no coin col; assembly asserts per-coin block order against the wide table
    W2COLS = [c for c in W.columns if c not in ("b50","fastres")]
    assert ("psw_5m" in W.columns) and not any(c.startswith("w2_") for c in W.columns) \
           and 100 <= len(W.columns) <= 140, "PROVENANCE FAIL on bf_w2cols.parquet"

    groups = []          # (name, codes int32 array, n_levels)
    def add_group(name, ser_str, vocab):
        cc = pd.Categorical(pd.Index(ser_str).astype(str), categories=list(vocab)).codes.astype(np.int16)
        if (cc < 0).any(): cc = np.where(cc < 0, len(vocab)-1, cc).astype(np.int16)
        groups.append((name, cc, len(vocab)))

    # --- W1 categoricals: global one-hot vocabulary, deterministic (sorted)
    for c in FE25:
        v = sorted(V[c].astype(str).unique())
        add_group(f"FE:{c}", V[c].astype(str).values, v)

    # --- W2 columns (certified psw-named schema, bf_w2cols.parquet)
    SWING_LV = ['HH','HL','LH','LL','na']
    SLOPE_LV = ['U','F','D','na']
    REL_LIKE = lambda c: c.endswith('_rel') or c.startswith('hay_cross') or c.startswith('dv_last')
    for c in W2COLS:
        if c.endswith('_sw'):                                  # 4 positional swing tokens
            arr = W[c].astype(str).values
            for pos in range(4):
                toks = np.array([('na' if x in ('na','nan','None') or x.count('-')!=3
                                  else x.split('-')[pos]) for x in arr], object)
                add_group(f"{c}#p{pos}", toks, SWING_LV)
        elif c.endswith('_sl'):                                # 8 positional slope segments
            arr = W[c].astype(str).values
            for pos in range(8):
                toks = np.array([('na' if x in ('na','nan','None') or len(x)!=8
                                  else x[pos]) for x in arr], object)
                add_group(f"{c}#s{pos}", toks, SLOPE_LV)
        elif REL_LIKE(c) and W[c].dtype == object:             # relations -> direct one-hot
            u = W[c].astype(str).fillna('na')
            add_group(c, u.values, sorted(u.unique()))
        else:                                                  # scalars -> quartile, na its level
            num = pd.to_numeric(W[c], errors='coerce')
            try: q = pd.qcut(num, 4, duplicates='drop', labels=False)
            except Exception: q = pd.Series(np.nan, index=num.index)
            qv = q.to_numpy(float) if hasattr(q,'to_numpy') else np.asarray(q,float)
            lab = np.where(np.isfinite(qv),
                           np.char.add("q", np.nan_to_num(qv, nan=0).astype(int).astype(str)),
                           "na").astype(object)
            vocab = sorted(set(lab.tolist()) - {"na"}) + ["na"]
            add_group(c, lab, vocab)

    LVS = np.array([g[2] for g in groups], np.int64)
    CODES = np.empty((len(V), len(groups)), np.int16)
    for j, (nm, cd, nl) in enumerate(groups): CODES[:, j] = cd
    GNAMES = [g[0] for g in groups]
    del groups; gc.collect()

    MONTH = V.month.values.astype("U7")
    FAV = np.stack([V[FR[f][0]].to_numpy(float) for f in FR]); ADV = np.stack([V[FR[f][1]].to_numpy(float) for f in FR])
    SECB = np.stack([W.b50.to_numpy(bool), W.fastres.to_numpy(bool)])
    del V, W; gc.collect()
    np.savez(CACHE, CODES=CODES, NUM=NUM, LVS=LVS, MONTH=MONTH, FAV=FAV, ADV=ADV,
             SECB=SECB, GNAMES=np.array(GNAMES, object))
    print("PREPARE: cache written", CACHE, flush=True)

NG = CODES.shape[1]
OFF = np.concatenate([[0], np.cumsum(LVS)])[:-1].astype(np.int32)
NCAT = int(LVS.sum()); NCOL = NCAT + len(T2)
W2MAX = max([l for nm, l in zip(GNAMES, LVS) if not str(nm).startswith("FE:")])
print(f"ENCODER: {NG} one-hot groups -> {NCAT} sparse cat cols + {len(T2)} numerics = {NCOL} features")
print(f"         nnz/row = {NG + len(T2)} | max levels any group = {LVS.max()} | max levels W2 group = {W2MAX} (cap 8)", flush=True)
assert W2MAX <= 8, "PREREG VIOLATION: a W2 group exceeds 8 one-hot levels"

NUMIDX = np.arange(NCAT, NCOL, dtype=np.int32)
CH = 20000
_BUF = {}
def _buf(tag, n, k):
    b = _BUF.get(tag)
    if b is None or b[0].size < n*k:
        _BUF[tag] = (np.empty(n*k, np.int32), np.empty(n*k, np.float32))
        b = _BUF[tag]
    return b[0][:n*k].reshape(n, k), b[1][:n*k].reshape(n, k)
def build(rows, med, sd, tag="tr"):
    """CSR for the given row indices; numerics median-imputed; all cols scaled by 1/sd.
    Chunked fill: no temporary ever exceeds CH x k."""
    n = len(rows); k = NG + len(T2)
    inv = np.where(sd > 2e-9, 1.0 / sd, 0.0).astype(np.float32)   # sd==0 -> exact-zero column,
    # which is precisely what b5.py's (x-mu)/(sd+1e-9) yields for a zero-variance TRAIN column
    # (and the fitted coefficient of such a column is 0, so test rows contribute 0 either way).
    ind, dat = _buf(tag, n, k)
    for a in range(0, n, CH):
        b = min(a+CH, n); r = rows[a:b]
        np.add(OFF[None, :], CODES[r], out=ind[a:b, :NG], casting="unsafe")
        ind[a:b, NG:] = NUMIDX
        nm = NUM[r]
        bad = ~np.isfinite(nm)
        dat[a:b, NG:] = np.where(bad, med, nm) if bad.any() else nm
        dat[a:b, :NG] = 1.0
        np.multiply(dat[a:b], inv[ind[a:b]], out=dat[a:b])
    return sp.csr_matrix((dat.reshape(-1), ind.reshape(-1),
                          np.arange(0, n*k+1, k, dtype=np.int32)), shape=(n, NCOL), copy=False)

def colstats(rows, med):
    """TRAIN sd per column, matching b5.py's std over the assembled matrix."""
    sd = np.empty(NCOL, np.float64)
    n = len(rows)
    cnt = np.zeros(NCAT, np.int64)
    for a in range(0, n, CH):
        r = rows[a:a+CH]
        cnt += np.bincount((OFF[None, :] + CODES[r]).reshape(-1), minlength=NCAT)
    p = cnt / n
    sd[:NCAT] = np.sqrt(p * (1 - p))
    ss = np.zeros(len(T2)); s2 = np.zeros(len(T2))
    for a in range(0, n, CH):
        r = rows[a:a+CH]; nm = NUM[r].astype(np.float64)
        bad = ~np.isfinite(nm)
        if bad.any(): nm = np.where(bad, med, nm)
        ss += nm.sum(0); s2 += (nm*nm).sum(0)
    mu = ss/n
    sd[NCAT:] = np.sqrt(np.maximum(s2/n - mu*mu, 0.0))
    return np.where(sd > 0, sd + 1e-9, 0.0)

# ---------------------------------------------------------------- walk-forward
months = sorted(set(MONTH.tolist()))
FRIX = {f: i for i, f in enumerate(FR)}
FRAMES = [f for f in A.frames.split(",") if f]

st = json.load(open(A.state)) if os.path.exists(A.state) else {"done": []}
res = pd.read_csv(A.out).to_dict("records") if os.path.exists(A.out) else []
done = set(tuple(x) for x in st["done"])
t0 = time.time()

for fr in FRAMES:
    fav = FAV[FRIX[fr]]; adv = ADV[FRIX[fr]]
    ok = np.isfinite(fav) & np.isfinite(adv)
    if A.secondary:
        fams = [("b50", SECB[0]), ("fastres", SECB[1])]
    else:
        fams = [("bounce", (fav >= .25) & (adv < .25)), ("through", adv >= .6)]
    idx = np.nonzero(ok)[0]
    order = np.argsort(MONTH[idx], kind="stable")     # month-contiguous; order-invariant model
    idx = idx[order]
    mon = MONTH[idx]
    bnd = {m: (np.searchsorted(mon, m, 'left'), np.searchsorted(mon, m, 'right')) for m in months}
    todo = [mi for mi in range(12, len(months)) if (fr, months[mi]) not in done]
    if A.probe: todo = todo[-A.probe:]
    for mi in todo:
        if (time.time()-t0)/60 > A.budget_min: break
        m = months[mi]
        lo, hi = bnd[m]
        if lo < 5000 or (hi-lo) < 50: 
            done.add((fr, m)); continue
        tr = idx[:lo]; te = idx[lo:hi]
        if len(tr) < 5000 or len(te) < 50:
            done.add((fr, m)); continue
        tt = time.time()
        med = np.nanmedian(NUM[tr], axis=0)
        sd = colstats(tr, med)
        Xtr = build(tr, med, sd, "tr"); Xte = build(te, med, sd, "te")
        for fam, y in fams:
            ytr = y[tr].astype(int); yte = y[te].astype(int)
            clf = LogisticRegression(C=1.0, max_iter=200, random_state=SEED)
            clf.fit(Xtr, ytr)
            p = clf.predict_proba(Xte)[:, 1]
            base = ytr.mean()
            br = float(np.mean((p-yte)**2)); brb = float(np.mean((base-yte)**2))
            res.append(dict(month=m, frame=fr, family=fam, n_test=int(len(te)), n_train=int(len(tr)),
                            brier=round(br,5), brier_base=round(brb,5),
                            skill=round(1-br/brb,4) if brb>0 else np.nan, base=round(base,4)))
        del Xtr, Xte; gc.collect()
        done.add((fr, m))
        pd.DataFrame(res).to_csv(A.out, index=False)
        json.dump({"done": sorted(list(done))}, open(A.state, "w"))
        print(f"{fr} {m} ntr={len(tr)} nte={len(te)} {time.time()-tt:.1f}s "
              f"(elapsed {(time.time()-t0)/60:.1f}m)", flush=True)

R = pd.DataFrame(res)
if len(R):
    agg = R.groupby(["frame","family"]).apply(
        lambda g: pd.Series(dict(months=len(g), med_skill=g.skill.median(),
                                 pos_months=(g.skill>0).mean())), include_groups=False).round(4)
    print(agg.to_string())
    print("sha", hashlib.sha256(open(A.out,'rb').read()).hexdigest()[:12])
