"""Ultra-fast ablation runner for CPU.

Optimized: 5K train/epoch, 500 val, digit acc as primary metric,
AR eval only when digit acc > 0.98 or every 30 epochs.
~25s/epoch -> 200 epochs in ~80 min per experiment.
"""

import argparse, math, os, time, sys
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict
import numpy as np

# Constants
VOCAB_SIZE = 14; D_MODEL = 6; TOK_DIM = 3; POS_DIM = 3
N_HEADS = 2; HEAD_DIM = 3; FFN_DIM = 6
MAX_SEQ_LEN = 34; MAX_DIGITS = 10; ANSWER_LEN = 11; N_POS = 10
PLUS_TOKEN = 10; EQUALS_TOKEN = 11; EOS_TOKEN = 12; PAD_TOKEN = 13
EQ_POS = 21; X_START = 0; PLUS_POS = 10; Y_START = 11; Z_START = 22; EOS_POS = 33

BATCH_SIZE = 512; LR = 1e-3
TRAIN_N = 5000; VAL_N = 500


def _spiral(i, p):
    return math.cos(2*math.pi*i/p), math.sin(2*math.pi*i/p), i/max(p-1,1)

def build_tok_emb():
    e = torch.zeros(VOCAB_SIZE, TOK_DIM)
    for d in range(10): e[d,0],e[d,1],e[d,2] = _spiral(d,10)
    e[10]=torch.tensor([2.,0.,-1.]); e[11]=torch.tensor([0.,2.,-1.])
    e[12]=torch.tensor([-2.,0.,-1.]); e[13]=torch.tensor([0.,-2.,-1.])
    return e

def build_pos_enc():
    pe = torch.zeros(MAX_SEQ_LEN, POS_DIM)
    for i in range(10): pe[i,0],pe[i,1],pe[i,2] = _spiral(i,10)
    for i in range(10): pe[11+i,0],pe[11+i,1],pe[11+i,2] = _spiral(i,10)
    for i in range(min(11,10)): pe[22+i,0],pe[22+i,1],pe[22+i,2] = _spiral(i,10)
    pe[32]=torch.tensor([0.,0.,1.5])
    pe[10]=torch.tensor([2.,0.,-1.]); pe[21]=torch.tensor([0.,2.,-1.]); pe[33]=torch.tensor([-2.,0.,-1.])
    return pe

# Data
_PM = 22; _AM = 12
_LM = np.array([0]*_PM+[1]*_AM, dtype=np.int64)

def gen_data(n, rng, donly=None):
    nd = np.full(n, donly, np.int64) if donly else rng.integers(1,11,size=n)
    lo = np.where(nd==1,0,10**(nd-1)); hi = 10**nd
    x = (rng.random(n)*(hi-lo)+lo).astype(np.int64)
    y = (rng.random(n)*(hi-lo)+lo).astype(np.int64)
    z = x+y
    t = np.empty((n,34),np.int64)
    tmp=x.copy()
    for d in range(10): t[:,d]=tmp%10; tmp//=10
    t[:,10]=10
    tmp=y.copy()
    for d in range(10): t[:,11+d]=tmp%10; tmp//=10
    t[:,21]=11
    tmp=z.copy()
    for d in range(11): t[:,22+d]=tmp%10; tmp//=10
    t[:,33]=12
    return t, nd

class DS(Dataset):
    def __init__(self, n, seed=None, donly=None):
        rng = np.random.default_rng(seed)
        t, nd = gen_data(n, rng, donly)
        self.t = torch.from_numpy(t)
        self.m = torch.from_numpy(np.tile(_LM,(n,1)))
        self.nd = torch.from_numpy(nd)
    def __len__(self): return len(self.t)
    def __getitem__(self, i): return self.t[i], self.m[i], self.nd[i]

# Attention modules
class Attn_QKPV(nn.Module):
    """QK=pos, V=tok"""
    def __init__(self, nh=2):
        super().__init__()
        self.nh=nh; self.hd=HEAD_DIM
        self.qp=nn.Linear(POS_DIM,nh*HEAD_DIM,bias=False)
        self.kp=nn.Linear(POS_DIM,nh*HEAD_DIM,bias=False)
        self.vp=nn.Linear(TOK_DIM,nh*HEAD_DIM,bias=False)
        self.op=nn.Linear(nh*HEAD_DIM,D_MODEL,bias=False)
        self.register_buffer("cm", torch.tril(torch.ones(34,34)).unsqueeze(0).unsqueeze(0))
    def forward(self, x):
        B,T,_=x.shape
        xp,xt = x[:,:,TOK_DIM:], x[:,:,:TOK_DIM]
        q=self.qp(xp).view(B,T,self.nh,self.hd).transpose(1,2)
        k=self.kp(xp).view(B,T,self.nh,self.hd).transpose(1,2)
        v=self.vp(xt).view(B,T,self.nh,self.hd).transpose(1,2)
        a=(q@k.transpose(-2,-1))/math.sqrt(self.hd)
        a=a.masked_fill(self.cm[:,:,:T,:T]==0,float('-inf'))
        a=F.softmax(a,dim=-1)
        o=(a@v).transpose(1,2).contiguous().view(B,T,self.nh*self.hd)
        return self.op(o)

class Attn_SharedQK(nn.Module):
    """Shared QK from pos, V from tok"""
    def __init__(self, nh=2):
        super().__init__()
        self.nh=nh; self.hd=HEAD_DIM
        self.qkp=nn.Linear(POS_DIM,nh*HEAD_DIM,bias=False)
        self.vp=nn.Linear(TOK_DIM,nh*HEAD_DIM,bias=False)
        self.op=nn.Linear(nh*HEAD_DIM,D_MODEL,bias=False)
        self.register_buffer("cm", torch.tril(torch.ones(34,34)).unsqueeze(0).unsqueeze(0))
    def forward(self, x):
        B,T,_=x.shape
        xp,xt = x[:,:,TOK_DIM:], x[:,:,:TOK_DIM]
        qk=self.qkp(xp).view(B,T,self.nh,self.hd).transpose(1,2)
        v=self.vp(xt).view(B,T,self.nh,self.hd).transpose(1,2)
        a=(qk@qk.transpose(-2,-1))/math.sqrt(self.hd)
        a=a.masked_fill(self.cm[:,:,:T,:T]==0,float('-inf'))
        a=F.softmax(a,dim=-1)
        o=(a@v).transpose(1,2).contiguous().view(B,T,self.nh*self.hd)
        return self.op(o)

class Attn_Std(nn.Module):
    """Standard attention from full d_model"""
    def __init__(self, nh=2):
        super().__init__()
        self.nh=nh; self.hd=D_MODEL//nh
        self.qp=nn.Linear(D_MODEL,D_MODEL,bias=False)
        self.kp=nn.Linear(D_MODEL,D_MODEL,bias=False)
        self.vp=nn.Linear(D_MODEL,D_MODEL,bias=False)
        self.op=nn.Linear(D_MODEL,D_MODEL,bias=False)
        self.register_buffer("cm", torch.tril(torch.ones(34,34)).unsqueeze(0).unsqueeze(0))
    def forward(self, x):
        B,T,_=x.shape
        q=self.qp(x).view(B,T,self.nh,self.hd).transpose(1,2)
        k=self.kp(x).view(B,T,self.nh,self.hd).transpose(1,2)
        v=self.vp(x).view(B,T,self.nh,self.hd).transpose(1,2)
        a=(q@k.transpose(-2,-1))/math.sqrt(self.hd)
        a=a.masked_fill(self.cm[:,:,:T,:T]==0,float('-inf'))
        a=F.softmax(a,dim=-1)
        o=(a@v).transpose(1,2).contiguous().view(B,T,D_MODEL)
        return self.op(o)

# Blocks
class BF(nn.Module):
    """Block with FFN"""
    def __init__(self, attn):
        super().__init__()
        self.ln1=nn.LayerNorm(D_MODEL); self.attn=attn
        self.ln2=nn.LayerNorm(D_MODEL)
        self.ffn=nn.Sequential(nn.Linear(D_MODEL,FFN_DIM),nn.GELU(),nn.Linear(FFN_DIM,D_MODEL))
    def forward(self, x):
        x=x+self.attn(self.ln1(x)); return x+self.ffn(self.ln2(x))

class BN(nn.Module):
    """Block no FFN"""
    def __init__(self, attn):
        super().__init__()
        self.ln=nn.LayerNorm(D_MODEL); self.attn=attn
    def forward(self, x): return x+self.attn(self.ln(x))

# Model
class M(nn.Module):
    def __init__(self, ft=False, fp=False, ffn=True, atype='qkpv', nl=1, nh=2):
        super().__init__()
        if ft:
            self.register_buffer("te", build_tok_emb()); self._tm='b'
        else:
            self.te_p=nn.Embedding(VOCAB_SIZE,TOK_DIM)
            self.te_p.weight.data.copy_(build_tok_emb()); self._tm='p'
        if fp:
            self.register_buffer("pe", build_pos_enc()); self._pm='b'
        else:
            self.pe_p=nn.Parameter(build_pos_enc()); self._pm='p'

        blocks=[]
        for _ in range(nl):
            if atype=='qkpv': a=Attn_QKPV(nh)
            elif atype=='shqk': a=Attn_SharedQK(nh)
            elif atype=='std': a=Attn_Std(nh)
            else: raise ValueError(atype)
            blocks.append(BF(a) if ffn else BN(a))
        self.blocks=nn.ModuleList(blocks)
        self.lnf=nn.LayerNorm(D_MODEL)
        self.head=nn.Linear(D_MODEL,VOCAB_SIZE,bias=False)
        for n,p in self.named_parameters():
            if n in ('te_p.weight','pe_p'): continue
            if p.dim()>1: nn.init.xavier_uniform_(p)

    def forward(self, idx):
        B,T=idx.shape
        tok = self.te[idx] if self._tm=='b' else self.te_p(idx)
        pos = (self.pe if self._pm=='b' else self.pe_p)[:T].unsqueeze(0).expand(B,-1,-1)
        x=torch.cat([tok,pos],dim=-1)
        for b in self.blocks: x=b(x)
        return self.head(self.lnf(x))

    @torch.no_grad()
    def generate(self, prompt):
        self.eval()
        B,Tp=prompt.shape
        fs=torch.zeros(B,Tp+12,dtype=torch.long,device=prompt.device)
        fs[:,:Tp]=prompt
        for s in range(12):
            T=Tp+s
            logits=self.forward(fs[:,:T])
            fs[:,T]=logits[:,-1].argmax(dim=-1)
        return fs[:,Tp:]

    def nparams(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad), \
               sum(p.numel() for p in self.parameters())

# Eval
def ev(model, loader, ar=False):
    model.eval()
    tl=tc=tt=te=ts=0
    with torch.no_grad():
        for tok,msk,nd in loader:
            lg=model(tok[:,:-1]); tg=tok[:,1:]; ms=msk[:,1:]
            l=F.cross_entropy(lg.reshape(-1,lg.size(-1)),tg.reshape(-1),reduction='none').reshape(tg.shape)
            ml=(l*ms).sum(); nt=ms.sum()
            if nt>0: tl+=ml.item(); tt+=nt.item()
            pr=lg.argmax(dim=-1)
            tc+=((pr==tg)&(ms==1)).sum().item()
            ts+=tok.size(0)
            if ar:
                gen=model.generate(tok[:,:22])
                te+=((gen==tok[:,22:]).all(dim=-1)).sum().item()
    return tl/max(tt,1), tc/max(tt,1), te/max(ts,1) if ar else None

# Configs
CFGS = {
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
    'fb_1h':       dict(ft=True, fp=True, ffn=True, atype='qkpv',nl=1,nh=1,donly=None),
}

def run(name, cfg, epochs=200, patience=40):
    donly=cfg.pop('donly',None)
    model=M(**cfg); nt,np_=model.nparams()
    print(f"\n{'='*60}\n{name} | {nt} trainable / {np_} counted\n{'='*60}")
    opt=torch.optim.Adam(model.parameters(),lr=LR)
    vl=DataLoader(DS(VAL_N,seed=42),batch_size=512)
    best_da=best_ea=0; pat=0; ep99=None; t0=time.time()

    for ep in range(1,epochs+1):
        tl=DataLoader(DS(TRAIN_N,seed=None,donly=donly),batch_size=BATCH_SIZE,shuffle=True)
        model.train(); el=et=0
        for tok,msk,_ in tl:
            lg=model(tok[:,:-1]); tg=tok[:,1:]; ms=msk[:,1:]
            l=F.cross_entropy(lg.reshape(-1,lg.size(-1)),tg.reshape(-1),reduction='none').reshape(tg.shape)
            ml=(l*ms).sum(); nt_=ms.sum()
            if nt_>0:
                (ml/nt_).backward(); opt.step(); opt.zero_grad()
                el+=ml.item(); et+=nt_.item()

        do_ev = (ep%3==0) or ep<=3
        if not do_ev: continue

        do_ar = (ep%20==0) or (best_da>0.97)
        vl_,da,ea = ev(model, vl, ar=do_ar)
        s=time.time()-t0
        msg=f"E{ep:3d} [{s:5.0f}s] loss={el/max(et,1):.4f} da={da:.4f}"
        if ea is not None:
            msg+=f" ea={ea:.4f}"
            if ea>best_ea:
                best_ea=ea; torch.save(model.state_dict(),f'exp_{name}.pt')
            if ep99 is None and ea>=0.99: ep99=ep; msg+=" *99%*"
        print(msg,flush=True)

        if da>best_da: best_da=da; pat=0
        else: pat+=1
        if pat>=patience:
            print(f"  Stop ep{ep}"); break

    tt=time.time()-t0
    # Final
    if os.path.exists(f'exp_{name}.pt'):
        model.load_state_dict(torch.load(f'exp_{name}.pt',weights_only=True))
    fl=DataLoader(DS(3000,seed=999),batch_size=512)
    _,_,fea=ev(model,fl,ar=True)
    print(f"RESULT {name}: params={np_} final_ea={fea:.4f} best_ea={best_ea:.4f} ep@99={ep99} time={tt:.0f}s")
    return dict(name=name,params=np_,final_ea=fea,best_ea=best_ea,ep99=ep99,best_da=best_da,time=tt)

def main():
    pa=argparse.ArgumentParser()
    pa.add_argument('--exp',nargs='+',default=['baseline'])
    a=pa.parse_args()

    groups = {
        'all': list(CFGS.keys()),
        'quick': ['baseline','freeze_both','no_ffn','nf_fb','minimal'],
        'freeze': ['baseline','freeze_tok','freeze_pos','freeze_both'],
        'ffn': ['baseline','no_ffn','nf_fb'],
        'attn': ['baseline','std','shqk','shqk_nf'],
        'heads': ['baseline','1h','nf_1h','fb_1h'],
    }

    names=[]
    for e in a.exp:
        if e in groups: names.extend(groups[e])
        elif e in CFGS: names.append(e)
        else: print(f"Unknown: {e}")
    # dedupe preserving order
    seen=set(); names=[n for n in names if not (n in seen or seen.add(n))]

    results=[]
    for n in names:
        cfg=CFGS[n].copy()
        results.append(run(n,cfg))

    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    print(f"{'Name':<15} {'P':>5} {'Final':>7} {'Best':>7} {'Ep99':>5} {'DA':>6} {'Time':>5}")
    print('-'*55)
    for r in results:
        print(f"{r['name']:<15} {r['params']:>5} {r['final_ea']:>7.4f} {r['best_ea']:>7.4f} "
              f"{str(r['ep99']):>5} {r['best_da']:>6.4f} {r['time']:>4.0f}s")

if __name__=='__main__':
    main()
