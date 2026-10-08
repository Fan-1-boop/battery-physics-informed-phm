import pandas as pd, numpy as np, os, time, json
from scipy.interpolate import interp1d

BASE='/mnt/data/che2025_official/ResValData'
OUT='/mnt/data/che2025_official/cache_dataset2.npz'
RATES=['C/80_Cycle','C/40_Cycle','0.05A_Cycle_mistake','C/10_Cycle','C/7_cycle','C/5_Cycle']
CYCLE_TYPES=['start_discharge','C/80_Cycle','GITT','C/40_Cycle','0.05A_Cycle_mistake','C/10_Cycle','C/7_cycle','C/5_Cycle','1C_Cycle','2C_Cycle','charge_for_storage']

def quantity_sum(data, quantity, state_type):
    state_code={'charge':'C','discharge':'D'}[state_type]
    q=data['_'+quantity].where(data['_state']==state_code, other=0).copy()
    if data['_wf_chg_cap'].notna().sum():
        key=(state_type,quantity)
        if key==('discharge','capacity'): col='_wf_dis_cap'
        elif key==('charge','capacity'): col='_wf_chg_cap'
        elif key==('discharge','energy'): col='_wf_dis_e'
        elif key==('charge','energy'): col='_wf_chg_e'
        else: col=None
        if col: q=data[col].where(data[col].notna(), other=q)
    end_step=data['_ending_status'].apply(lambda x: 128 <= x <= 255)
    is_step_change=data['step_index'].diff(periods=-1).fillna(0)!=0
    end_step_inds=end_step.index[np.logical_and(end_step.to_numpy(), is_step_change.to_numpy())]
    if end_step_inds.size==0: return q
    cycle_sum=0.0; begin=q.index[0]+1
    for end in end_step_inds:
        if data.loc[begin-1,'cycle_index'] != data.loc[begin,'cycle_index']:
            cycle_sum=0.0
        q.loc[begin:end] += cycle_sum
        cycle_sum=q.loc[end]; begin=end+1
    if end_step_inds[-1] < q.index[-1]: q.loc[begin:] += cycle_sum
    return q

def parse(path):
    usecols=['Cyc#','Step','Step (Sec)','Amp-hr','Watt-hr','Amps','Volts','State','ES','WF Chg Cap','WF Dis Cap','WF Chg E','WF Dis E']
    d=pd.read_csv(path,sep='\t',skiprows=1,usecols=usecols,na_values='N/A')
    d=d.rename(columns={'Cyc#':'cycle_index','Step':'step_index','Step (Sec)':'step_time','Amp-hr':'_capacity','Watt-hr':'_energy','Amps':'current','Volts':'voltage','State':'_state','ES':'_ending_status','WF Chg Cap':'_wf_chg_cap','WF Dis Cap':'_wf_dis_cap','WF Chg E':'_wf_chg_e','WF Dis E':'_wf_dis_e'})
    d['discharge_capacity']=quantity_sum(d,'capacity','discharge')
    d['cycle_type']=d['cycle_index'].map(lambda x:CYCLE_TYPES[int(x)])
    return d

def curve_from_cycle(data, ctype, input_1c=False, num_points=200, nominal=4.84):
    f=data[data.cycle_type==ctype]
    if len(f)==0: return None
    V=f.voltage.to_numpy(); I=f.current.to_numpy(); Q=f.discharge_capacity.to_numpy(); t=f.step_time.to_numpy()
    charge_end_idx=None
    for j in range(2,len(Q)):
        if Q[j]>Q[j-1] and I[j]<0:
            charge_end_idx=j; break
    if charge_end_idx is None: return None
    discharge_end_idx=None
    for j in range(2,len(Q)):
        if V[j]<=2.7:
            discharge_end_idx=j; break
    if discharge_end_idx is None: discharge_end_idx=len(Q)-1
    start=charge_end_idx
    if np.max(Q[charge_end_idx:discharge_end_idx])<2: return None
    if input_1c:
        end=None
        for j in range(start,discharge_end_idx):
            if (t[j]-t[start])>1200:
                end=j; break
        if end is None: end=discharge_end_idx
        V=V[start:end]/4.2
        Q=Q[start:end]/nominal-Q[start]/nominal
    else:
        V=V[start:discharge_end_idx+1]/4.2
        Q=Q[start:discharge_end_idx+1]/nominal-Q[start]/nominal
    if len(V)<2: return None
    x=np.linspace(0,1,num_points)
    vi=interp1d(np.linspace(0,1,len(V)),V,kind='linear')(x)
    qi=interp1d(np.linspace(0,1,len(Q)),Q,kind='linear')(x)
    return vi,qi

def main():
    files=sorted([f for f in os.listdir(BASE) if f.startswith('ResVal') and f!='ResVal_000084_0000ED.072'])
    X=[]; Ys={r:[] for r in RATES}; kept=[]; skipped=[]
    t0=time.time()
    for i,fn in enumerate(files):
        d=parse(os.path.join(BASE,fn))
        inp=curve_from_cycle(d,'1C_Cycle',input_1c=True)
        if inp is None:
            skipped.append((fn,'input')); continue
        outs={r:curve_from_cycle(d,r,input_1c=False) for r in RATES}
        if any(v is None for v in outs.values()):
            skipped.append((fn,'target')); continue
        vi,qi=inp
        X.append(np.stack([vi,qi,np.ones(200)],axis=-1))
        for r,(vo,qo) in outs.items(): Ys[r].append(np.stack([vo,qo],axis=-1))
        kept.append(fn)
        if (i+1)%10==0: print(i+1,'/',len(files),'elapsed',round(time.time()-t0,1))
    X=np.array(X,dtype=np.float32)
    arr={'X':X,'files':np.array(kept)}
    for k,r in enumerate(RATES): arr[f'Y{k}']=np.array(Ys[r],dtype=np.float32)
    np.savez_compressed(OUT,**arr)
    meta={'n':len(kept),'skipped':skipped,'rates':RATES,'files':kept,'shape_X':list(X.shape),'shape_Y':{r:list(np.array(Ys[r]).shape) for r in RATES}}
    with open('/mnt/data/che2025_official/cache_dataset2_meta.json','w') as f: json.dump(meta,f,indent=2)
    print('DONE',meta,'elapsed',time.time()-t0)

if __name__=='__main__': main()