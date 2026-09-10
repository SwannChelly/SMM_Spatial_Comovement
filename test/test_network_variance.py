"""End-to-end gate: a synthetic finite-variety economy with the model's own structure
(Frechet prices, CES weights, per-sector N_s), decomposed and checked against theory."""
import json, numpy as np, pandas as pd, matplotlib, warnings, os
matplotlib.use("Agg"); import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

nb=json.load(open('/home/user/SMM_Spatial_Comovement/diffusion.ipynb'))
code=''.join(nb['cells'][26]['source'])

THETA=1.0; NU=1.5; R=45; B=400
rng=np.random.default_rng(7)
Ns={0:3,1:6,2:12,3:24}; buyers=[1,2,3,4]
RHO={}; rows=[]
for s,N in Ns.items():
    cells=np.sort(rng.choice(R, size=int(rng.integers(10,R)), replace=False))
    T=np.exp(rng.normal(0,0.9,size=len(cells)))
    for r in buyers:
        d=np.maximum(rng.lognormal(5.5,0.6,size=len(cells)),1.0)
        psi=T*d**(-THETA*0.3); p=psi/psi.sum()
        full=np.zeros(R); full[cells]=p; RHO[(s,r)]=full
        for b in range(B):
            win=rng.choice(cells, size=N, p=p)                 # iid winners
            # Frechet winning price, independent of the winner: p_rho^{-theta} ~ Exp
            price=rng.exponential(size=N)**(-1.0/THETA)
            v=price**(1-NU); v=v/v.sum()                        # CES weights
            for k in range(N):
                rows.append((b,s,r,win[k]+1,k,v[k]))
sup=pd.DataFrame(rows,columns=['replication','A129','ze2010_downstream','ze2010',
                               'variety','share'])
sup['A129']+=1                                                  # model index 1..S

data={"R":R,"S":len(Ns),"sector_names":[f"S{s}" for s in Ns],"suppliers":sup,
      "post_hoc_N_hat":np.array([Ns[s] for s in sorted(Ns)],float),
      "folder":".","step_dir":"step3"}
ns={'np':np,'pd':pd,'plt':plt,'os':os,'toulouse_color':(.2,.4,.6),'sim_color':(.7,.3,.2),
    'NU_S_DEFAULT':NU,'THETA_DEFAULT':THETA,
    '_parquet_sector_index':lambda d,s: s['A129'].to_numpy().astype(int)-1,
    '_n_hat_from_diagnostics':lambda d: None,
    '_region_labels':lambda d: pd.DataFrame({'index':np.arange(1,R+1),
                                             'ze2010':[str(i) for i in range(1,R+1)],
                                             'ze2010_name':[f"z{i}" for i in range(1,R+1)]}),
    'sourcing_geometry':None}
exec(code, ns)
# rho_incidence is fed directly, bypassing sourcing_geometry
ns['rho_incidence']=lambda d,rows_,geom=None: np.array(
    [RHO[(int(s),int(r))] for s,r in zip(rows_['sector'],rows_['ze2010_downstream'])])

sec=ns['network_variance_decomposition'](data, level="sector")
by =ns['network_variance_by_sector'](sec, data)
t,ts,_=ns['network_variance_theory'](data, sec, None)
print("\nIDENTITY  max|V_real-(V_S+V_G)| =", np.nanmax(np.abs(sec.identity_gap)))
print("HHI form  max|E[H]-H(rho)-H_v(1-H(rho))| =",
      np.nanmax(np.abs(t.hhi_realised - t.hhi_realised_theory)))
print("\nRATIOS simulated/predicted (target 1.00 for `exact`):")
print(ts[['sector_name','N_hat','n_eff_varieties','ratio_exact','ratio_frechet',
          'ratio_equal']].round(3).to_string(index=False))
