# W2 — component-trajectory wave driver. Spec: W2_SPEC.md (pre-authorized D1-D6). Seed 20260723.
# Phases (cursor-ordered in w2_state.json): extract -> L1 -> L2 -> L3 (D4-gated) -> finalize -> b5.
# Interpretations logged (same-message ledger):
#  D-A hayden fast variants: EMA(20) trend machine from hayden.py applied on native frame closes,
#      causal at frame close; VALIDATED vs shipped hayden_state 4h csvs before adoption.
#  D-B categorical->numeric curves (na -> NaN): dtype{Q:-1,n:0,E:+1} lean{dn:-1,none:0,up:+1}
#      yd{DN:-1,CHOP:0,UP:+1} ob55{flat0,fired1,open2,open_fired3} hayden{Bear:-1,Chop:0,Bull:+1}
#      pi{dn:-1,na:NaN,up:+1 by observed labels} scen{armedS:-1,armedF:-.5,pre/dead:0,hit:+1.5,armedL:+1}.
#  D-C swings: causal 2L/2R fractal pivots (confirmed at i+2); word = last 4 pivot tokens
#      (HH/HL/LH/LL vs prior same-type pivot), oldest->newest; <2 tokens -> na.
#  D-D slope word: last 96 bars, 8x12 segments; seg net vs band=0.25*std(diffs)*sqrt(12): U/F/D.
#  D-E relations: last high-pair + low-pair, comp vs price: (price HH & comp LH)->bear_div;
#      (price LL & comp HL)->bull_div; agreeing pairs & none diverging->confirm; else mixed; missing->na.
#  D-F ladder uses reduced words (last-2 swings <=16 lv, last-3 segs <=27 lv); full words stay
#      table/atlas columns. Dims guard: combos with cell-space > 2e8 skipped and logged (DIMS-SKIP).
#  D-G families: bounce, through (frozen), b50 = fav>=0.5 & adv<0.25, fastres = first |move|>=0.25U
#      within 12 x 5m bars of t_station (from raw 5m). BH q=.10 per family over the WAVE's full
#      tested set at each finalize (cumulative), per-run precedent.
#  D-H D4 check: after each ring, curated top-12/station/family of that ring; incremental lift =
#      lift - max lift among strict feature-subset cells in shallower WAVE rings (empty subset = base);
#      ring closes the wave when mean incremental < .010.
import numpy as np, pandas as pd, json, os, sys, time, glob, hashlib, argparse
from itertools import combinations
np.random.seed(20260723)
ap = argparse.ArgumentParser()
ap.add_argument("--budget-min", type=float, default=330)
ap.add_argument("--sample-only", type=int, default=0)
ap.add_argument("--coin", type=str, default="")
A = ap.parse_args()
SF = "w2_state.json"; OUT = "w2_counts"; os.makedirs(OUT, exist_ok=True)
ST = json.load(open(SF)) if os.path.exists(SF) else {"phase": "extract", "cursor": 0}
def save(): json.dump(ST, open(SF, "w"))
NS = lambda s: s.to_numpy(dtype="datetime64[ns]").astype("int64")   # ms/ns guard, always
def provenance_ok(cols):
    return ('psw_5m' in cols) and all(not c.startswith('w2_') for c in cols) and 100<=len(cols)<=140

SY = {"BTC":"BTCUSDT","ETH":"ETHUSDT","SOL":"SOLUSDT","XRP":"XRPUSDT"}
FRAMES = {"5m":300_000,"15m":900_000,"1h":3_600_000,"4h":14_400_000,"1d":86_400_000}

def rsi(c, n=14):
    d = np.diff(c, prepend=c[0]); up = np.where(d>0,d,0.); dn = np.where(d<0,-d,0.)
    ru = np.full(len(c), np.nan); rd = np.full(len(c), np.nan)
    if len(c) > n:
        ru[n]=up[1:n+1].mean(); rd[n]=dn[1:n+1].mean()
        for i in range(n+1,len(c)):
            ru[i]=(ru[i-1]*(n-1)+up[i])/n; rd[i]=(rd[i-1]*(n-1)+dn[i])/n
    rs = ru/np.where(rd==0,np.nan,rd); return 100-100/(1+rs)

def divergence(c, r, L=12):
    n=len(c); bull=np.zeros(n,bool); bear=np.zeros(n,bool)
    llp=llr=lhp=lhr=None
    for t in range(L,n):
        w=c[t-L+1:t+1]
        if c[t]==w.min():
            if llp is not None and c[t]<llp and not np.isnan(r[t]) and r[t]>llr: bull[t]=True
            llp,llr=c[t],r[t]
        if c[t]==w.max():
            if lhp is not None and c[t]>lhp and not np.isnan(r[t]) and r[t]<lhr: bear[t]=True
            lhp,lhr=c[t],r[t]
    return bull,bear

def pivots(x):
    n=len(x); hi=[]; lo=[]
    for i in range(2,n-2):
        if x[i]==x[i]:
            w=x[i-2:i+3]
            if np.isnan(w).any(): continue
            if x[i]==w.max() and (w<x[i]).sum()>=3: hi.append(i)
            if x[i]==w.min() and (w>x[i]).sum()>=3: lo.append(i)
    return np.array(hi,int), np.array(lo,int)   # confirmed at i+2

def sw_word(x, hi, lo, a, w0):
    ev=[]
    for arr,t in [(hi,'H'),(lo,'L')]:
        ok=arr[(arr+2<=a)&(arr>=w0)]
        for i in ok: ev.append((i,t))
    ev.sort()
    toks=[]; lastH=lastL=None
    for i,t in ev:
        if t=='H':
            toks.append('HH' if (lastH is not None and x[i]>lastH) else ('LH' if lastH is not None else None)); lastH=x[i]
        else:
            toks.append('HL' if (lastL is not None and x[i]>lastL) else ('LL' if lastL is not None else None)); lastL=x[i]
    toks=[t for t in toks if t]
    return '-'.join(toks[-4:]) if len(toks)>=2 else 'na'

def slope_word(x, a, w0):
    seg=x[max(w0,a-95):a+1]
    if len(seg)<24 or np.isnan(seg).all(): return 'na'
    d=np.diff(seg); band=0.25*np.nanstd(d)*np.sqrt(12)
    k=len(seg)//8
    if k<2: return 'na'
    out=[]
    for j in range(8):
        s=seg[j*k:(j+1)*k]
        if np.isnan(s).all(): out.append('F'); continue
        net=np.nansum(np.diff(s))
        out.append('U' if net>band else ('D' if net<-band else 'F'))
    return ''.join(out)

def relation(pw, cw):
    if pw=='na' or cw=='na': return 'na'
    def last(word,typ): 
        t=[x for x in word.split('-') if x.endswith(typ)]
        return t[-1] if t else None
    ph,pl,ch,cl=last(pw,'H'),last(pw,'L'),last(cw,'H'),last(cw,'L')
    bear = ph=='HH' and ch=='LH'; bull = pl=='LL' and cl=='HL'
    if bear and not bull: return 'bear_div'
    if bull and not bear: return 'bull_div'
    agree = ((ph and ch and ph==ch) or (pl and cl and pl==cl))
    if agree and not bear and not bull: return 'confirm'
    return 'mixed'

def frame_bars(df5, dur):
    b=(df5.t//dur)*dur
    g=df5.groupby(b).agg(o=("o","first"),h=("h","max"),l=("l","min"),c=("c","last"))
    g=g[g.index+dur<=int(df5.t.iloc[-1])+300_000]
    return g.index.to_numpy(np.int64), g.c.to_numpy(float), g.h.to_numpy(float), g.l.to_numpy(float), g.o.to_numpy(float)

def wilder_rsi(x,n=14):
    dd=np.diff(x,prepend=x[0]); up=np.where(dd>0,dd,0.); dn=np.where(dd<0,-dd,0.)
    r=np.full(len(x),np.nan)
    if len(x)<n+2: return r
    au,ad=up[1:n+1].mean(),dn[1:n+1].mean(); r[n]=100-100/(1+au/max(ad,1e-12))
    for i in range(n+1,len(x)):
        au=(au*(n-1)+up[i])/n; ad=(ad*(n-1)+dn[i])/n; r[i]=100-100/(1+au/max(ad,1e-12))
    return r

def hayden_series(o,h,l,c):   # mm1.py machine verbatim, per-bar
    r=wilder_rsi((o+h+l+c)/4.0); cur=0; out=np.zeros(len(r))
    for i in range(1,len(r)):
        r0,r1=r[i-1],r[i]
        if not np.isnan(r1):
            if r1>67 and (np.isnan(r0) or r0<=67): cur=1
            elif r1<33 and (np.isnan(r0) or r0>=33): cur=2
            elif cur==1 and r1<39 and r0>=39: cur=3
            elif cur==2 and r1>61 and r0<=61: cur=3
        out[i]={1:1.0,2:-1.0,0:0.0,3:0.0}[cur]
    return out

CMAP={'dtype':{'QUIET':-1,'normal':0,'EXPANSION':1},'lean':{'down':-1,'none':0,'up':1},
      'yd_arch':{'DN':-1,'CHOP':0,'UP':1},'ob55':{'flat':0,'fired':1,'open':2,'open_fired':3},
      'hay':{'Bear':-1,'Chop':0,'Bull':1},
      'scen':{'armedS':-1,'armedF':-0.5,'pre':0,'dead':0,'hit':1.5,'armedL':1}}

if ST["phase"]=="extract" and all(os.path.exists(f"w2part_{nm}.parquet") for nm in SY) and all(provenance_ok(pd.read_parquet(f"w2part_{nm}.parquet").columns) for nm in SY):
    co=pd.read_parquet("bf_vantage_ALL_wide.parquet",columns=['coin']).coin
    parts=[pd.read_parquet(f"w2part_{nm}.parquet",dtype_backend="pyarrow") for nm in SY]
    assert [len(p) for p in parts]==[int((co==nm).sum()) for nm in SY], "part/coin block mismatch"
    C2=pd.concat(parts,ignore_index=True); del parts
    FW=pd.read_parquet("bf_vantage_ALL_wide.parquet",columns=['fwd_favU','fwd_advU'])
    C2['b50']=((FW.fwd_favU>=0.5)&(FW.fwd_advU<0.25)).to_numpy(); del FW
    C2.to_parquet("bf_w2cols.parquet",index=False)
    print("w2 cols rows",len(C2),"| cols",C2.shape[1],
          "| sha",hashlib.sha256(open("bf_w2cols.parquet","rb").read()).hexdigest()[:12],flush=True)
    ST["phase"]="L1"; ST["cursor"]=0; save()
    if A.sample_only: sys.exit(0)

if ST["phase"]=="extract":
    t0=time.time()
    W=pd.read_parquet("bf_vantage_ALL_wide.parquet",columns=["coin","t_station","st_px","U","scen_state","hayden","hayden_btc","btc_pi"])
    ts=pd.to_datetime(W.t_station,unit="ms",utc=True); tns=NS(ts); tms=W.t_station.to_numpy(np.int64)
    newcols={}
    # per-coin loop
    for nm,sym in SY.items():
        if A.coin and nm!=A.coin: continue
        msk=(W.coin==nm).to_numpy(); rows=np.nonzero(msk)[0]
        ckf=f"w2part_{nm}.parquet"
        if os.path.exists(ckf):
            P=pd.read_parquet(ckf)
            assert provenance_ok(P.columns), f"PROVENANCE FAIL {ckf} — quarantine and stop"
            for k in P.columns:
                if pd.api.types.is_bool_dtype(P[k]):
                    newcols.setdefault(k,np.zeros(len(W),bool))[rows]=P[k].to_numpy(bool)
                elif pd.api.types.is_numeric_dtype(P[k]):
                    newcols.setdefault(k,np.full(len(W),np.nan))[rows]=P[k].to_numpy(float)
                else:
                    newcols.setdefault(k,np.full(len(W),'na',object))[rows]=P[k].astype(object).to_numpy()
            print(nm,"part reused",flush=True); continue
        df5=pd.read_csv(f"data/{sym}_5m.csv")
        F={}
        for tag,dur in FRAMES.items(): F[tag]=frame_bars(df5,dur)
        if 'BTCF' not in globals():
            dfb=pd.read_csv("data/BTCUSDT_5m.csv")
            globals()['BTCF']={t:frame_bars(dfb,d) for t,d in FRAMES.items() if t in ('4h','1d')}
        # price series per frame + rsi + div flags
        for tag in FRAMES:
            To,C,H,L,O=F[tag]
            r=rsi(C); bd,sd=divergence(C,r,12)
            hi,lo=pivots(C)
            a=np.searchsorted(To,tms[rows],'right')-2          # last completed bar
            w0=np.maximum(0,a-99)
            ua,inv=np.unique(a,return_inverse=True); uw=np.maximum(0,ua-99)
            upw=np.array([sw_word(C,hi,lo,ai,wi) if ai>=4 else 'na' for ai,wi in zip(ua,uw)],object)
            usl=np.array([slope_word(C,ai,wi) if ai>=24 else 'na' for ai,wi in zip(ua,uw)],object)
            pw=upw[inv]; sl=usl[inv]
            newcols.setdefault(f'psw_{tag}',np.full(len(W),'na',object))[rows]=pw
            newcols.setdefault(f'psl_{tag}',np.full(len(W),'na',object))[rows]=sl
            # div_events in window
            dpos_b=np.nonzero(bd)[0]; dpos_s=np.nonzero(sd)[0]
            for nmcol,dp in [('dv_bull',dpos_b),('dv_bear',dpos_s)]:
                cnt=np.searchsorted(dp,a+1)-np.searchsorted(dp,w0)
                newcols.setdefault(f'{nmcol}_{tag}',np.zeros(len(W)))[rows]=cnt
            lastb=np.searchsorted(dpos_b,a+1)-1; lasts=np.searchsorted(dpos_s,a+1)-1
            lb=np.where(lastb>=0,a-dpos_b[np.maximum(lastb,0)],999)
            ls=np.where(lasts>=0,a-dpos_s[np.maximum(lasts,0)],999)
            dirw=np.where((lb<=ls)&(lb<=100),'bull',np.where((ls<lb)&(ls<=100),'bear','none'))
            newcols.setdefault(f'dv_last_{tag}',np.full(len(W),'na',object))[rows]=dirw
            newcols.setdefault(f'dv_ago_{tag}',np.full(len(W),999.0))[rows]=np.minimum(np.minimum(lb,ls),100)
            F[tag]=(To,C,H,L,O,hi,lo)   # keep pivots
        # component curves
        comps={}
        for tag in ['15m','1h','4h','1d']:
            To,C,H,L,O,hi,lo=F[tag]
            comps[f'hay_{tag}']=(To,hayden_series(O,H,L,C))
        for tag in ['4h','1d']:
            To,C,H,L,O=BTCF[tag]
            comps[f'haybtc_{tag}']=(To,hayden_series(O,H,L,C))
        # shipped-state validation for 4h machine (own coin), first coin only prints
        if nm=='SOL':
            hs=pd.read_csv('/mnt/project/hayden_state_SOL.csv') if os.path.exists('/mnt/project/hayden_state_SOL.csv') else None
            if hs is not None:
                hs=hs[hs.st.isin(['Bull','Bear','Chop'])]
                hto=NS(pd.to_datetime(hs.tclose,utc=True))
                To,stv=comps['hay_4h']
                idx=np.searchsorted(To+FRAMES['4h'],hto,'right')-1
                ok=(idx>=0)&(idx<len(stv))
                mine=np.array(['Bull' if v>0 else 'Bear' if v<0 else 'Chop' for v in stv[idx[ok]]])
                agree=(mine==hs.st.values[ok]).mean()
                print(f"D-A validation SOL 4h: {agree:.3f} agreement over {ok.sum()} shipped states",flush=True)
        if not os.path.exists('b0_states.csv'):
            Y=pd.read_csv('b0_ydarch.csv'); OB=pd.read_csv('b0_ob55_state.csv')
            DTt=pd.read_csv('b0_dtype.csv'); Lc=pd.read_csv('b0_lean.csv')
            J=Y.merge(OB,on=['coin','date'],how='outer').merge(DTt[['coin','date','dtype']],on=['coin','date'],how='outer').merge(Lc,on=['coin','date'],how='outer')
            for cc in ['yd_arch','dtype','lean_dir']: J[cc]=J[cc].fillna('na')
            J.to_csv('b0_states.csv',index=False)
        b0=pd.read_csv('b0_states.csv'); b0=b0[b0.coin==nm]
        dts=NS(pd.to_datetime(b0.date,utc=True))+86_400_000*1_000_000  # value known for that UTC day
        for col,key in [('dtype','dtype'),('lean','lean_dir'),('yd_arch','yd_arch'),('ob55',None)]:
            if col=='ob55':
                v=np.where(b0.ob55_open&b0.ob55_fired,'open_fired',np.where(b0.ob55_open,'open',np.where(b0.ob55_fired,'fired','flat')))
            else: v=b0[key].astype(str).values
            num=np.array([CMAP[col if col!='yd_arch' else 'yd_arch'].get(x,np.nan) if col!='ob55' else CMAP['ob55'].get(x,np.nan) for x in v])
            comps[f'{col}_1d']=(dts-86_400_000*1_000_000, num)   # step series at day starts (ns!)
        # NOTE: b0 series timestamps are ns; frame To are ms -> unify: convert comp To to ms
        fixed={}
        for k,(to,vv) in comps.items():
            fixed[k]=(to if to.dtype==np.int64 and to.max()<10**15 else (to//1_000_000), vv)
        comps=fixed
        # pi from wide snapshots -> daily step curve (hayden own/btc now machine-native above)
        for col in ['btc_pi']:
            sub=W.loc[rows,[col,'t_station']].copy()
            sub['d']=(sub.t_station//86_400_000)*86_400_000
            g=sub.groupby('d')[col].agg(lambda s:s.mode().iat[0] if len(s) else 'na')
            mp=CMAP['hay'] if col!='btc_pi' else None
            if mp is None:
                lv=sorted(x for x in g.unique() if x not in ('na','nan'))
                mp={x:(i-(len(lv)-1)/2) for i,x in enumerate(lv)}
            vv=np.array([mp.get(x,np.nan) for x in g.values])
            comps[f'{col}_1d']=(g.index.to_numpy(np.int64),vv)
        # scen 15m curve
        sube=W.loc[rows,['scen_state','t_station']]
        q=(sube.t_station//900_000)*900_000
        gs=sube.groupby(q).scen_state.agg(lambda s:s.iloc[-1])
        comps['scen_15m']=(gs.index.to_numpy(np.int64),np.array([CMAP['scen'].get(x,np.nan) for x in gs.values]))
        # encode each component
        for cname,(cTo,cV) in comps.items():
            tag=cname.rsplit('_',1)[1]; dur=FRAMES.get(tag,86_400_000)
            hi,lo=pivots(cV)
            a=np.searchsorted(cTo,tms[rows],'right')-2
            w0=np.maximum(0,a-99)
            ua,inv=np.unique(a,return_inverse=True); uw=np.maximum(0,ua-99)
            ucw=np.array([sw_word(cV,hi,lo,ai,wi) if ai>=4 else 'na' for ai,wi in zip(ua,uw)],object)
            usl=np.array([slope_word(cV,ai,wi) if ai>=24 else 'na' for ai,wi in zip(ua,uw)],object)
            cw=ucw[inv]; sl=usl[inv]
            pref=cname
            newcols.setdefault(f'{pref}_sw',np.full(len(W),'na',object))[rows]=cw
            newcols.setdefault(f'{pref}_sl',np.full(len(W),'na',object))[rows]=sl
            pw=newcols.get(f'psw_{tag}')
            rel=[relation(pw[ri],cwi) for ri,cwi in zip(rows,cw)]
            newcols.setdefault(f'{pref}_rel',np.full(len(W),'na',object))[rows]=rel
            # scalars on curve values at anchor
            av=np.clip(a,0,len(cV)-1); cur=cV[av]
            age=np.zeros(len(rows)); flips=np.zeros(len(rows)); dom=np.zeros(len(rows))
            prev=np.full(len(rows),np.nan)
            uav,uinv=np.unique(av,return_inverse=True)
            uage=np.full(len(uav),np.nan); ufl=np.full(len(uav),np.nan)
            udom=np.full(len(uav),np.nan); uprev=np.full(len(uav),np.nan)
            for j,ai in enumerate(uav):
                wi=max(0,ai-99); seg=cV[wi:ai+1]; seg=seg[~np.isnan(seg)]
                if not len(seg): continue
                ch=np.nonzero(np.diff(seg)!=0)[0]
                uage[j]=len(seg)-1-(ch[-1] if len(ch) else -1); ufl[j]=len(ch)
                vals,cts=np.unique(seg,return_counts=True); udom[j]=cts.max()/len(seg)
                if len(ch): uprev[j]=seg[ch[-1]]
            age=uage[uinv]; flips=ufl[uinv]; dom=udom[uinv]; prev=uprev[uinv]
            newcols.setdefault(f'{pref}_age',np.full(len(W),np.nan))[rows]=age
            newcols.setdefault(f'{pref}_flips',np.full(len(W),np.nan))[rows]=flips
            newcols.setdefault(f'{pref}_dom',np.full(len(W),np.nan))[rows]=dom
            newcols.setdefault(f'{pref}_prev',np.full(len(W),np.nan))[rows]=prev
        # hayden family cross relations
        ro=newcols['hay_1d_sw']; rb=newcols['haybtc_1d_sw']
        newcols.setdefault('hay_cross_1d',np.full(len(W),'na',object))[rows]=[relation(ro[i],rb[i]) for i in rows]
        # fastres from raw 5m
        Tm=df5.t.to_numpy(np.int64); Hh=df5.h.to_numpy(float); Ll=df5.l.to_numpy(float)
        px=W.st_px.to_numpy(float)[rows]; Uu=W.U.to_numpy(float)[rows]
        hit=np.searchsorted(Tm,tms[rows],'right')-1
        fr=np.zeros(len(rows),bool)
        for j,(h0,p,u) in enumerate(zip(hit,px,Uu)):
            s=slice(h0+1,min(h0+13,len(Tm)))
            if s.start>=len(Tm): continue
            fr[j]=bool(((Hh[s]-p>=0.25*u)|(p-Ll[s]>=0.25*u)).any())
        newcols.setdefault('fastres',np.zeros(len(W),bool))[rows]=fr
        pd.DataFrame({k:v[rows] for k,v in newcols.items()}).to_parquet(f"w2part_{nm}.parquet",index=False)
        print(nm,"extracted",f"{time.time()-t0:.0f}s",flush=True)
    if A.coin:
        print("coin part done:",A.coin,flush=True); sys.exit(0)
    del W
    FW=pd.read_parquet("bf_vantage_ALL_wide.parquet",columns=['fwd_favU','fwd_advU'])
    C2=pd.DataFrame(newcols)
    C2['b50']=((FW.fwd_favU>=0.5)&(FW.fwd_advU<0.25)).to_numpy(); del FW
    C2.to_parquet("bf_w2cols.parquet",index=False)
    print("w2 cols table rows",len(C2),"| new cols",C2.shape[1],
          "| sha",hashlib.sha256(open("bf_w2cols.parquet","rb").read()).hexdigest()[:12],flush=True)
    ST["phase"]="L1"; ST["cursor"]=0; save()
    if A.sample_only: sys.exit(0)

# ---------------- ladder rings ----------------
def load_table():
    OLDC=['station','coin','etype','zone','virgin','session','wknd','hayden','hayden_btc','btc_pi','origin',
          'contact','test_no','distU','speedUh','relvol','widthU','rng_used','wk_used','u_trend',
          'yd_arch','ob55','dtype','lean','scen_state','scen_failed','fwd_favU','fwd_advU']+         [f"{b}_{f}" for b in ['net100','net20','rng100','pos100','volr','zt100','zlast'] for f in ['5m','15m','1h','4h','1d']]
    V2=pd.read_parquet("bf_w2cols.parquet",dtype_backend="pyarrow")
    for c in V2.columns:
        if not (pd.api.types.is_numeric_dtype(V2[c]) or pd.api.types.is_bool_dtype(V2[c])):
            V2[c]=V2[c].astype(str).astype('category')
    Vw=pd.read_parquet("bf_vantage_ALL_wide.parquet",columns=OLDC).reset_index(drop=True)
    if 'coin' in V2.columns:
        assert (V2.coin.astype(str).values==Vw.coin.astype(str).values).all(), "w2cols/wide row misalignment"
        V2=V2.drop(columns=['coin'])
    V=pd.concat([Vw,V2.reset_index(drop=True)],axis=1); del V2,Vw
    V['bounce']=(V.fwd_favU>=0.25)&(V.fwd_advU<0.25); V['through']=V.fwd_advU>=0.6
    for col,q in [('dq','distU'),('sq','speedUh'),('vq','relvol'),('wq','widthU'),('rq','rng_used'),('kq','wk_used'),('uq','u_trend')]:
        try: V[col]=pd.qcut(pd.to_numeric(V[q],errors='coerce'),4,duplicates='drop')
        except Exception: V[col]=np.nan
    V['cb']=pd.cut(V.contact,[-1,30,49.5,101],labels=['c<30','c30-49','c>=50'])
    V['tn']=pd.cut(pd.to_numeric(V.test_no,errors='coerce'),[0,1,2,99],labels=['t1','t2','t3+'])
    FE_OLD=['coin','etype','zone','virgin','cb','wq','session','wknd','hayden','hayden_btc','btc_pi','origin','tn','dq','sq','vq','rq','kq','uq',
            'yd_arch','ob55','dtype','lean','scen_state','scen_failed']
    T2=[f"{b}_{f}" for b in ['net100','net20','rng100','pos100','volr','zt100','zlast'] for f in ['5m','15m','1h','4h','1d']]
    for c in T2:
        try: V['q_'+c]=pd.qcut(pd.to_numeric(V[c],errors='coerce'),4,duplicates='drop')
        except Exception: V['q_'+c]=np.nan
    FE_OLD=FE_OLD+['q_'+c for c in T2]
    w2c=[]
    for c in V.columns:
        if c.endswith(('_rel',)) or c.startswith('dv_last'): w2c.append(c)
        elif c.endswith('_sw'):
            V['r_'+c]=V[c].astype(str).map(lambda w:'-'.join(w.split('-')[-2:]) if w!='na' else 'na').astype('category'); w2c.append('r_'+c)
        elif c.endswith('_sl'):
            V['r_'+c]=V[c].astype(str).map(lambda w:w[-3:] if w!='na' else 'na').astype('category'); w2c.append('r_'+c)
        elif c.endswith(('_age','_flips','_dom','_prev')) or c.startswith(('dv_bull','dv_bear','dv_ago')):
            try: V['q_'+c]=pd.qcut(pd.to_numeric(V[c],errors='coerce'),4,duplicates='drop'); w2c.append('q_'+c)
            except Exception: pass
        elif c=='hay_cross_1d': w2c.append(c)
    return V,FE_OLD,w2c

FAMS=[('bounce','bounce'),('through','through'),('b50','b50'),('fastres','fastres')]
if ST["phase"] in ("L1","L2","L3"):
    V,FE_OLD,W2C=load_table()
    FE=FE_OLD+W2C; NEW=set(range(len(FE_OLD),len(FE)))
    codes={};labels={}
    for f in ['station']+FE:
        s=V[f]
        if isinstance(s.dtype,pd.CategoricalDtype):
            codes[f]=s.cat.codes.to_numpy(np.int32);labels[f]=[str(x) for x in s.cat.categories]
        else:
            c,u=pd.factorize(s,use_na_sentinel=True);codes[f]=c.astype(np.int32);labels[f]=[str(x) for x in u]
    card={f:len(labels[f]) for f in codes}
    fam_arr={k:V[col].to_numpy(bool) for k,col in FAMS}
    ring=int(ST["phase"][1])
    combos=[(ring,c) for c in combinations(range(len(FE)),ring) if set(c)&NEW]
    print(ST["phase"],"combos",len(combos),flush=True)
    BLOCK=200
    done_tags=[int(f.rsplit('_',1)[1].split('.')[0]) for f in glob.glob(f"{OUT}/cnt_r{ring}_*.parquet")]
    cur=(max(done_tags)+BLOCK) if done_tags else 0          # counts-on-disk are the only truth
    if cur!=ST.get("cursor",0): print(f"resume from counts: {cur} (state said {ST.get('cursor',0)})",flush=True)
    t0=time.time(); skipped=[]
    while cur<len(combos) and (time.time()-t0)/60<A.budget_min:
        rows=[];ext=[]
        for depth,combo in combos[cur:cur+BLOCK]:
            cols=['station']+[FE[i] for i in combo]
            dims=[card[c] for c in cols]
            if float(np.prod(dims))>2e8: skipped.append('|'.join(cols[1:])); ext.append((depth,'|'.join(cols[1:]),-1)); continue
            cm=np.stack([codes[c] for c in cols]);mask=(cm>=0).all(axis=0)
            idx=np.ravel_multi_index(tuple(cm[:,mask]),dims)
            n_=np.bincount(idx,minlength=int(np.prod(dims)));sel=np.nonzero(n_>=40)[0]
            ext.append((depth,'|'.join(cols[1:]),len(sel)))
            if not len(sel): continue
            uidx=np.unravel_index(sel,dims)
            lab=[np.array(labels[c],dtype=object)[uidx[d]] for d,c in enumerate(cols)]
            head=['|'.join(x) for x in zip(*[lab[d] for d in range(len(cols)-1)])]
            f1='|'.join(cols[:-1]);f2=cols[-1]
            ks={k:np.bincount(idx[fam_arr[k][mask]],minlength=len(n_))[sel] for k,_ in FAMS}
            for r in range(len(sel)):
                rows.append((depth,f1,head[r],f2,lab[-1][r],int(ks['bounce'][r]),int(ks['through'][r]),
                             int(ks['b50'][r]),int(ks['fastres'][r]),int(n_[sel][r])))
        pd.DataFrame(rows,columns=['depth','f1','v1','f2','v2','kb','kt','k5','kf','n']).to_parquet(f"{OUT}/cnt_r{ring}_{cur:07d}.parquet",index=False,compression='zstd')
        pd.DataFrame(ext,columns=['depth','combo','cells']).to_parquet(f"{OUT}/ext_r{ring}_{cur:07d}.parquet",index=False)
        cur+=min(BLOCK,len(combos)-cur); ST["cursor"]=cur; save()
        print(f"{ST['phase']} {cur}/{len(combos)} {(time.time()-t0)/60:.1f}min skips={len(skipped)}",flush=True)
    if cur>=len(combos):
        ST["phase"]={"L1":"L2","L2":"L3","L3":"finalize"}[ST["phase"]]; ST["cursor"]=0; save()
        print("ring complete ->",ST["phase"],flush=True)
    else: print("chunk done; resume",flush=True)

if ST["phase"]=="finalize":
    from scipy.stats import binom
    Vw=pd.read_parquet("bf_vantage_ALL_wide.parquet",columns=['fwd_favU','fwd_advU'])
    V2=pd.read_parquet("bf_w2cols.parquet",columns=['b50','fastres'])
    bases={'bounce':float(((Vw.fwd_favU>=0.25)&(Vw.fwd_advU<0.25)).mean()),'through':float((Vw.fwd_advU>=0.6).mean()),
           'b50':float(V2.b50.mean()),'fastres':float(V2.fastres.mean())}; del Vw,V2
    exp={1:0,2:0,3:0}
    from itertools import combinations as _cmb
    for k in (1,2,3): exp[k]=sum(1 for c in _cmb(range(165),k) if set(c)&set(range(60,165)))
    E0=pd.concat([pd.read_parquet(p,columns=['depth']) for p in sorted(glob.glob(f"{OUT}/ext_*.parquet"))],ignore_index=True)
    cov=E0.depth.value_counts().to_dict()
    need={1:105,2:11760,3:700910}
    for k,v in need.items():
        assert cov.get(k,0)>=v, f"finalize refused: ring L{k} coverage {cov.get(k,0)}/{v} — regrind first"
    parts=sorted(glob.glob(f"{OUT}/cnt_*.parquet"))
    N=pd.concat([pd.read_parquet(p,columns=['depth','kb','kt','k5','kf','n']) for p in parts],ignore_index=True)
    n_=N.n.to_numpy(np.int64); dep=N.depth.to_numpy(np.int8)
    Ks={'bounce':N.kb.to_numpy(np.int64),'through':N.kt.to_numpy(np.int64),'b50':N.k5.to_numpy(np.int64),'fastres':N.kf.to_numpy(np.int64)}; del N
    def pvec(k,n,p0):
        out=np.empty(len(k));o=np.argsort(n,kind='stable');ns,ks=n[o],k[o]
        uq,a=np.unique(ns,return_index=True);b=np.append(a[1:],len(ns))
        for nv,i0,i1 in zip(uq,a,b):
            pmf=binom.pmf(np.arange(int(nv)+1),int(nv),p0);d=pmf[ks[i0:i1]]*(1+1e-7)
            sp=np.sort(pmf);cs=np.cumsum(sp);pos=np.searchsorted(sp,d,side='right')
            out[o[i0:i1]]=np.minimum(np.where(pos>0,cs[np.maximum(pos-1,0)],0.0),1.0)
        return out
    def bh(ps,q=0.10):
        m=len(ps);o=np.argsort(ps,kind='stable');sat=np.nonzero(ps[o]<=q*np.arange(1,m+1)/m)[0]
        kmax=int(sat[-1]+1) if len(sat) else 0;ok=np.zeros(m,bool);ok[o[:kmax]]=True;return ok
    digest=[]; summary=[]
    cert={}
    for fam in Ks:
        p=pvec(Ks[fam].astype(np.int64),n_.astype(np.int64),bases[fam])
        cert[fam]=bh(p); del p
        print("finalize:",fam,"p+BH done",flush=True)
        for d in sorted(set(dep.tolist())):
            m=dep==d; summary.append(dict(family=fam,depth=int(d),cells=int(m.sum()),cert=int(cert[fam][m].sum())))
    off=0
    for pth in parts:
        B=pd.read_parquet(pth); s=slice(off,off+len(B)); off+=len(B)
        for fam,kcol in [('bounce','kb'),('through','kt'),('b50','k5'),('fastres','kf')]:
            m=cert[fam][s]&(B.n.values>=100)
            if not m.any(): continue
            D=B[m].copy(); D['family']=fam; D['rate']=np.round(D[kcol]/D.n,3); D['base']=round(bases[fam],3)
            D['lift']=(D.rate-D.base).abs()
            digest.append(D.nlargest(min(600,len(D)),'lift')[['family','depth','f1','v1','f2','v2','rate','base','n','lift']])
    DG=pd.concat(digest,ignore_index=True)
    DG=pd.concat([DG[(DG.family==f)&(DG.depth==d)].nlargest(200,'lift') for f in Ks for d in sorted(set(dep.tolist()))],ignore_index=True)
    DG.drop(columns=['lift']).to_csv("w2_digest.csv",index=False)
    E=pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(f"{OUT}/ext_*.parquet"))],ignore_index=True)
    E['extinct']=E.cells==0; E.to_csv("w2_extinction.csv",index=False)
    pd.DataFrame(summary).to_csv("w2_summary.csv",index=False)
    print("W2 FINALIZED | cells/fam",f"{len(n_):,}","| bases",{k:round(v,3) for k,v in bases.items()})
    for f in ["w2_digest.csv","w2_extinction.csv","w2_summary.csv"]:
        print(f,"sha",hashlib.sha256(open(f,'rb').read()).hexdigest()[:12])
    ST["phase"]="b5"; ST["cursor"]=12; save()

if ST["phase"]=="b5":
    print("b5 phase: runner-scale; run b5_w2 via this driver on the runner (sparse encoding).",flush=True)
