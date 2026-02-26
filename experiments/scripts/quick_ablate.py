"""Quick ablation: 30-epoch runs to measure learning curves.

~12 min per experiment on CPU. Enough to distinguish converging vs non-converging configs.
Uses 20K samples/epoch for better signal (matches original data rate better).
"""

import argparse, math, os, time, sys, json
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np

VOCAB_SIZE=14; D_MODEL=6; TOK_DIM=3; POS_DIM=3
N_HEADS=2; HEAD_DIM=3; FFN_DIM=6
MAX_SEQ_LEN=34; MAX_DIGITS=10; ANSWER_LEN=11; N_POS=10
PLUS_TOKEN=10; EQUALS_TOKEN=11; EOS_TOKEN=12; PAD_TOKEN=13
EQ_POS=21

BATCH_SIZE=512; LR=1e-3
TRAIN_N=20000; VAL_N=1000; EPOCHS=30

def _spiral(i,p):
    return math.cos(2*math.pi*i/p), math.sin(2*math.pi*i/p), i/max(p-1,1)

def build_tok():
    e=torch.zeros(VOCAB_SIZE,TOK_DIM)
    for d in range(10): e[d,0],e[d,1],e[d,2]=_spiral(d,10)
    e[10]=torch.tensor([2.,0.,-1.]); e[11]=torch.tensor([0.,2.,-1.])
    e[12]=torch.tensor([-2.,0.,-1.]); e[13]=torch.tensor([0.,-2.,-1.])
    return e

def build_pos():
    pe=torch.zeros(MAX_SEQ_LEN,POS_DIM)
    for i in range(10): pe[i,0],pe[i,1],pe[i,2]=_spiral(i,10)
    for i in range(10): pe[11+i,0],pe[11+i,1],pe[11+i,2]=_spiral(i,10)
    for i in range(min(11,10)): pe[22+i,0],pe[22+i,1],pe[22+i,2]=_spiral(i,10)
    pe[32]=torch.tensor([0.,0.,1.5])
    pe[10]=torch.tensor([2.,0.,-1.]); pe[21]=torch.tensor([0.,2.,-1.]); pe[33]=torch.tensor([-2.,0.,-1.])
    return pe

_PM=22; _AM=12; _LM=np.array([0]*_PM+[1]*_AM,dtype=np.int64)

def gen(n,rng,donly=None):
    nd=np.full(n,donly,np.int64) if donly else rng.integers(1,11,size=n)
    lo=np.where(nd==1,0,10**(nd-1)); hi=10**nd
    x=(rng.random(n)*(hi-lo)+lo).astype(np.int64)
    y=(rng.random(n)*(hi-lo)+lo).astype(np.int64)
    z=x+y; t=np.empty((n,34),np.int64)
    tmp=x.copy()
    for d in range(10): t[:,d]=tmp%10; tmp//=10
    t[:,10]=10; tmp=y.copy()
    for d in range(10): t[:,11+d]=tmp%10; tmp//=10
    t[:,21]=11; tmp=z.copy()
    for d in range(11): t[:,22+d]=tmp%10; tmp//=10
    t[:,33]=12; return t,nd

class DS(Dataset):
    def __init__(s,n,seed=None,donly=None):
        rng=np.random.default_rng(seed); t,nd=gen(n,rng,donly)
        s.t=torch.from_numpy(t); s.m=torch.from_numpy(np.tile(_LM,(n,1)))
        s.nd=torch.from_numpy(nd)
    def __len__(s): return len(s.t)
    def __getitem__(s,i): return s.t[i],s.m[i],s.nd[i]

# Attn
class Attn(nn.Module):
    """QK=pos, V=tok"""
    def __init__(s,nh=2):
        super().__init__(); s.nh=nh; s.hd=HEAD_DIM
        s.qp=nn.Linear(POS_DIM,nh*HEAD_DIM,bias=False)
        s.kp=nn.Linear(POS_DIM,nh*HEAD_DIM,bias=False)
        s.vp=nn.Linear(TOK_DIM,nh*HEAD_DIM,bias=False)
        s.op=nn.Linear(nh*HEAD_DIM,D_MODEL,bias=False)
        s.register_buffer("cm",torch.tril(torch.ones(34,34))[None,None])
    def forward(s,x):
        B,T,_=x.shape; xp=x[:,:,TOK_DIM:]; xt=x[:,:,:TOK_DIM]
        q=s.qp(xp).view(B,T,s.nh,s.hd).transpose(1,2)
        k=s.kp(xp).view(B,T,s.nh,s.hd).transpose(1,2)
        v=s.vp(xt).view(B,T,s.nh,s.hd).transpose(1,2)
        a=(q@k.transpose(-2,-1))/math.sqrt(s.hd)
        a=a.masked_fill(s.cm[:,:,:T,:T]==0,float('-inf'))
        return s.op((F.softmax(a,dim=-1)@v).transpose(1,2).contiguous().view(B,T,s.nh*s.hd))

class AttnSh(nn.Module):
    """Shared QK from pos, V from tok"""
    def __init__(s,nh=2):
        super().__init__(); s.nh=nh; s.hd=HEAD_DIM
        s.qkp=nn.Linear(POS_DIM,nh*HEAD_DIM,bias=False)
        s.vp=nn.Linear(TOK_DIM,nh*HEAD_DIM,bias=False)
        s.op=nn.Linear(nh*HEAD_DIM,D_MODEL,bias=False)
        s.register_buffer("cm",torch.tril(torch.ones(34,34))[None,None])
    def forward(s,x):
        B,T,_=x.shape; xp=x[:,:,TOK_DIM:]; xt=x[:,:,:TOK_DIM]
        qk=s.qkp(xp).view(B,T,s.nh,s.hd).transpose(1,2)
        v=s.vp(xt).view(B,T,s.nh,s.hd).transpose(1,2)
        a=(qk@qk.transpose(-2,-1))/math.sqrt(s.hd)
        a=a.masked_fill(s.cm[:,:,:T,:T]==0,float('-inf'))
        return s.op((F.softmax(a,dim=-1)@v).transpose(1,2).contiguous().view(B,T,s.nh*s.hd))

class AttnStd(nn.Module):
    """Standard attention from full d_model"""
    def __init__(s,nh=2):
        super().__init__(); s.nh=nh; s.hd=D_MODEL//nh
        s.qp=nn.Linear(D_MODEL,D_MODEL,bias=False)
        s.kp=nn.Linear(D_MODEL,D_MODEL,bias=False)
        s.vp=nn.Linear(D_MODEL,D_MODEL,bias=False)
        s.op=nn.Linear(D_MODEL,D_MODEL,bias=False)
        s.register_buffer("cm",torch.tril(torch.ones(34,34))[None,None])
    def forward(s,x):
        B,T,_=x.shape
        q=s.qp(x).view(B,T,s.nh,s.hd).transpose(1,2)
        k=s.kp(x).view(B,T,s.nh,s.hd).transpose(1,2)
        v=s.vp(x).view(B,T,s.nh,s.hd).transpose(1,2)
        a=(q@k.transpose(-2,-1))/math.sqrt(s.hd)
        a=a.masked_fill(s.cm[:,:,:T,:T]==0,float('-inf'))
        return s.op((F.softmax(a,dim=-1)@v).transpose(1,2).contiguous().view(B,T,D_MODEL))

# Blocks
class BF(nn.Module):
    def __init__(s,a):
        super().__init__(); s.ln1=nn.LayerNorm(D_MODEL); s.a=a
        s.ln2=nn.LayerNorm(D_MODEL)
        s.ffn=nn.Sequential(nn.Linear(D_MODEL,FFN_DIM),nn.GELU(),nn.Linear(FFN_DIM,D_MODEL))
    def forward(s,x): x=x+s.a(s.ln1(x)); return x+s.ffn(s.ln2(x))

class BN(nn.Module):
    def __init__(s,a): super().__init__(); s.ln=nn.LayerNorm(D_MODEL); s.a=a
    def forward(s,x): return x+s.a(s.ln(x))

# Model
class Mdl(nn.Module):
    def __init__(s,ft=False,fp=False,ffn=True,atype='qkpv',nl=1,nh=2):
        super().__init__()
        if ft: s.register_buffer("te",build_tok()); s._tm='b'
        else:
            s.te_p=nn.Embedding(VOCAB_SIZE,TOK_DIM)
            s.te_p.weight.data.copy_(build_tok()); s._tm='p'
        if fp: s.register_buffer("pe",build_pos()); s._pm='b'
        else: s.pe_p=nn.Parameter(build_pos()); s._pm='p'
        blks=[]
        for _ in range(nl):
            if atype=='qkpv': a=Attn(nh)
            elif atype=='shqk': a=AttnSh(nh)
            elif atype=='std': a=AttnStd(nh)
            else: raise ValueError(atype)
            blks.append(BF(a) if ffn else BN(a))
        s.blocks=nn.ModuleList(blks)
        s.lnf=nn.LayerNorm(D_MODEL); s.head=nn.Linear(D_MODEL,VOCAB_SIZE,bias=False)
        for n,p in s.named_parameters():
            if n in ('te_p.weight','pe_p'): continue
            if p.dim()>1: nn.init.xavier_uniform_(p)

    def forward(s,idx):
        B,T=idx.shape
        tok=s.te[idx] if s._tm=='b' else s.te_p(idx)
        pos=(s.pe if s._pm=='b' else s.pe_p)[:T][None].expand(B,-1,-1)
        x=torch.cat([tok,pos],dim=-1)
        for b in s.blocks: x=b(x)
        return s.head(s.lnf(x))

    @torch.no_grad()
    def generate(s,prompt):
        s.eval(); B,Tp=prompt.shape
        fs=torch.zeros(B,Tp+12,dtype=torch.long,device=prompt.device)
        fs[:,:Tp]=prompt
        for st in range(12):
            T=Tp+st; lg=s.forward(fs[:,:T]); fs[:,T]=lg[:,-1].argmax(-1)
        return fs[:,Tp:]

    def npar(s): return sum(p.numel() for p in s.parameters() if p.requires_grad), sum(p.numel() for p in s.parameters())

def evaluate(model,loader,ar=False):
    model.eval(); tl=tc=tt=te=ts=0
    with torch.no_grad():
        for tok,msk,nd in loader:
            lg=model(tok[:,:-1]); tg=tok[:,1:]; ms=msk[:,1:]
            l=F.cross_entropy(lg.reshape(-1,lg.size(-1)),tg.reshape(-1),reduction='none').reshape(tg.shape)
            ml=(l*ms).sum(); nt=ms.sum()
            if nt>0: tl+=ml.item(); tt+=nt.item()
            tc+=((lg.argmax(-1)==tg)&(ms==1)).sum().item()
            ts+=tok.size(0)
            if ar:
                g=model.generate(tok[:,:22])
                te+=((g==tok[:,22:]).all(-1)).sum().item()
    return tl/max(tt,1), tc/max(tt,1), te/max(ts,1) if ar else None

CFGS={
    'baseline':    dict(ft=False,fp=False,ffn=True, atype='qkpv',nl=1,nh=2,donly=None),
    'freeze_tok':  dict(ft=True, fp=False,ffn=True, atype='qkpv',nl=1,nh=2,donly=None),
    'freeze_pos':  dict(ft=False,fp=True, ffn=True, atype='qkpv',nl=1,nh=2,donly=None),
    'freeze_both': dict(ft=True, fp=True, ffn=True, atype='qkpv',nl=1,nh=2,donly=None),
    'no_ffn':      dict(ft=False,fp=False,ffn=False,atype='qkpv',nl=1,nh=2,donly=None),
    'nf_fb':       dict(ft=True, fp=True, ffn=False,atype='qkpv',nl=1,nh=2,donly=None),
    'nf_2L':       dict(ft=False,fp=False,ffn=False,atype='qkpv',nl=2,nh=2,donly=None),
    'd10':         dict(ft=False,fp=False,ffn=True, atype='qkpv',nl=1,nh=2,donly=10),
    'std':         dict(ft=False,fp=False,ffn=True, atype='std', nl=1,nh=2,donly=None),
    'shqk':        dict(ft=False,fp=False,ffn=True, atype='shqk',nl=1,nh=2,donly=None),
    'shqk_nf':     dict(ft=False,fp=False,ffn=False,atype='shqk',nl=1,nh=2,donly=None),
    'minimal':     dict(ft=True, fp=True, ffn=False,atype='shqk',nl=1,nh=2,donly=None),
    '1h':          dict(ft=False,fp=False,ffn=True, atype='qkpv',nl=1,nh=1,donly=None),
    'nf_1h':       dict(ft=False,fp=False,ffn=False,atype='qkpv',nl=1,nh=1,donly=None),
}

def run(name,cfg,epochs=EPOCHS):
    donly=cfg.pop('donly',None)
    model=Mdl(**cfg); nt,np_=model.npar()
    print(f"\n{'='*60}\n{name} | {nt} trainable / {np_} counted | donly={donly}\n{'='*60}",flush=True)
    opt=torch.optim.Adam(model.parameters(),lr=LR)
    vl=DataLoader(DS(VAL_N,seed=42),batch_size=512)
    t0=time.time(); hist=[]

    for ep in range(1,epochs+1):
        tl=DataLoader(DS(TRAIN_N,seed=None,donly=donly),batch_size=BATCH_SIZE,shuffle=True)
        model.train(); el=et=0
        for tok,msk,_ in tl:
            lg=model(tok[:,:-1]); tg=tok[:,1:]; ms=msk[:,1:]
            l=F.cross_entropy(lg.reshape(-1,lg.size(-1)),tg.reshape(-1),reduction='none').reshape(tg.shape)
            ml=(l*ms).sum(); nt_=ms.sum()
            if nt_>0: (ml/nt_).backward(); opt.step(); opt.zero_grad(); el+=ml.item(); et+=nt_.item()

        # Eval every epoch (TF only - fast), AR at end
        do_ar = (ep==epochs) or (ep%10==0)
        vl_,da,ea=evaluate(model,vl,ar=do_ar)
        s=time.time()-t0
        msg=f"E{ep:2d} [{s:5.0f}s] loss={el/max(et,1):.4f} da={da:.4f}"
        if ea is not None: msg+=f" ea={ea:.4f}"
        print(msg,flush=True)
        hist.append(dict(ep=ep,da=da,ea=ea,loss=el/max(et,1)))

    tt=time.time()-t0
    # Final AR eval
    _,_,fea=evaluate(model,DataLoader(DS(2000,seed=999),batch_size=512),ar=True)
    print(f"RESULT {name}: P={np_} final_ea={fea:.4f} da@30={hist[-1]['da']:.4f} time={tt:.0f}s",flush=True)
    return dict(name=name,params=np_,final_ea=fea,hist=hist,time=tt)

def main():
    pa=argparse.ArgumentParser()
    pa.add_argument('--exp',nargs='+',default=['baseline'])
    pa.add_argument('--epochs',type=int,default=EPOCHS)
    a=pa.parse_args()

    groups={
        'all': list(CFGS.keys()),
        'quick': ['baseline','freeze_both','no_ffn','nf_fb','minimal'],
        'freeze': ['baseline','freeze_tok','freeze_pos','freeze_both'],
        'ffn': ['baseline','no_ffn','nf_fb','nf_2L'],
        'attn': ['baseline','std','shqk'],
        'key': ['baseline','freeze_both','no_ffn','nf_fb','d10','shqk','minimal'],
    }

    names=[]
    for e in a.exp:
        if e in groups: names.extend(groups[e])
        elif e in CFGS: names.append(e)
        else: print(f"Unknown: {e}")
    seen=set(); names=[n for n in names if not (n in seen or seen.add(n))]

    results=[]
    for n in names:
        results.append(run(n,CFGS[n].copy(),epochs=a.epochs))

    print(f"\n{'='*70}\nSUMMARY (30-epoch learning curves)\n{'='*70}")
    print(f"{'Name':<15} {'P':>5} {'DA@10':>7} {'DA@20':>7} {'DA@30':>7} {'EA@30':>7} {'Time':>5}")
    print('-'*60)
    for r in results:
        h=r['hist']
        da10=h[9]['da'] if len(h)>=10 else h[-1]['da']
        da20=h[19]['da'] if len(h)>=20 else h[-1]['da']
        da30=h[-1]['da']
        ea30=r['final_ea']
        print(f"{r['name']:<15} {r['params']:>5} {da10:>7.4f} {da20:>7.4f} {da30:>7.4f} {ea30:>7.4f} {r['time']:>4.0f}s")

    # Save results
    with open('ablation_results.json','w') as f:
        json.dump([{k:v for k,v in r.items() if k!='hist'} for r in results], f, indent=2)

if __name__=='__main__':
    main()
