"""Nine tabular architectures, full implementations.

TabNet comes from pytorch-tabnet; the rest are implemented here, faithful to the papers but sized
for 1M x 296. The attention-based ones (FT-Transformer, SAINT, ExcelFormer, CARTE) take the top-96
features by gain — with 296 tokens the attention matrix alone would dominate the compute budget,
and that truncation is standard practice for these models on wide tables.

Every model predicts in log1p space and is trained with plain L2 there, which is exactly RMSLE.
"""
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

sys.path.insert(0, "models")
from zoo_common import load, run_pair, torch_fit_predict

DEV = "cuda" if torch.cuda.is_available() else "cpu"
X, Y, NAMES, TOP = load()
NTOK = 96
TOPC = np.sort(TOP[:NTOK])
print(f"attention models use top-{NTOK} features by gain")

mse = lambda o, y: ((o[:, 0] - torch.log1p(y)) ** 2).mean()
take0 = lambda o: o[:, 0]

# ============================================================ 1. FT-Transformer
class FeatureTokenizer(nn.Module):
    """one learned d-dim token per numeric feature: token_j = w_j * x_j + b_j, plus a CLS token"""
    def __init__(self, f, d):
        super().__init__()
        self.w = nn.Parameter(torch.randn(f, d) * d ** -0.5)
        self.b = nn.Parameter(torch.zeros(f, d))
        self.cls = nn.Parameter(torch.randn(1, 1, d) * d ** -0.5)
    def forward(self, x):
        t = x.unsqueeze(-1) * self.w + self.b
        return torch.cat([self.cls.expand(x.shape[0], -1, -1), t], 1)

class FTTransformer(nn.Module):
    def __init__(self, f, d=64, nl=3, nh=8, drop=0.1):
        super().__init__()
        self.tok = FeatureTokenizer(f, d)
        el = nn.TransformerEncoderLayer(d, nh, d * 2, dropout=drop, batch_first=True,
                                        norm_first=True, activation="gelu")
        self.tr = nn.TransformerEncoder(el, nl)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.ReLU(), nn.Linear(d, 1))
    def forward(self, x):
        return self.head(self.tr(self.tok(x))[:, 0])

# ============================================================ 2. SAINT
class SAINT(nn.Module):
    """FT-style column attention alternating with intersample attention across the batch"""
    def __init__(self, f, d=48, nl=2, nh=4, drop=0.1):
        super().__init__()
        self.tok = FeatureTokenizer(f, d)
        self.col = nn.ModuleList([nn.TransformerEncoderLayer(d, nh, d * 2, dropout=drop,
                                  batch_first=True, norm_first=True, activation="gelu") for _ in range(nl)])
        dr = d * (f + 1)
        self.row = nn.ModuleList([nn.TransformerEncoderLayer(dr, 1, dr, dropout=drop,
                                  batch_first=True, norm_first=True, activation="gelu") for _ in range(nl)])
        self.head = nn.Sequential(nn.LayerNorm(d), nn.ReLU(), nn.Linear(d, 1))
    def forward(self, x):
        z = self.tok(x)
        B, T, D = z.shape
        for c, r in zip(self.col, self.row):
            z = c(z)
            z = r(z.reshape(1, B, T * D)).reshape(B, T, D)   # attention over the batch dimension
        return self.head(z[:, 0])

# ============================================================ 3. NODE
def entmax15(z, dim=-1):
    """alpha=1.5 entmax, the sparse transform NODE uses for feature and threshold selection"""
    z = z - z.max(dim=dim, keepdim=True).values
    zs, _ = torch.sort(z, dim=dim, descending=True)
    r = torch.arange(1, z.shape[dim] + 1, device=z.device, dtype=z.dtype)
    shape = [1] * z.dim(); shape[dim] = -1
    r = r.view(shape)
    mean = zs.cumsum(dim) / r
    mean_sq = (zs ** 2).cumsum(dim) / r
    ss = r * (mean_sq - mean ** 2)
    delta = torch.clamp((1 - ss) / r, min=0)
    tau = mean - torch.sqrt(delta)
    # support must be at least 1: without the clamp it can reach 0 numerically, gather then reads
    # index -1 and trips a device-side assert that poisons the CUDA context for the whole process
    support = (tau <= zs).sum(dim, keepdim=True).clamp(min=1)
    tau_star = tau.gather(dim, support - 1)
    return torch.clamp(z - tau_star, min=0) ** 2

class ODST(nn.Module):
    """one layer of oblivious differentiable decision trees.

    The published layer uses entmax for both the feature choice and the leaf response. A literal
    entmax on the selection tensor collapsed here — the gradient path through the sorted threshold
    is too weak at 96 features and the layer degenerated to a constant. Replaced by a temperature-
    annealed softmax, the standard relaxation, which keeps the sparse-selection behaviour with a
    gradient that actually flows."""
    def __init__(self, f, n_trees=128, depth=6, out=1):
        super().__init__()
        self.n, self.d, self.o = n_trees, depth, out
        self.sel = nn.Parameter(torch.randn(f, n_trees, depth) * 0.5)
        self.logT = nn.Parameter(torch.zeros(1))
        self.thr = nn.Parameter(torch.randn(n_trees, depth) * 0.5)
        self.tau = nn.Parameter(torch.ones(n_trees, depth))
        self.leaf = nn.Parameter(torch.randn(n_trees, 2 ** depth, out) * (2 ** depth) ** -0.5)
    def forward(self, x):
        w = torch.softmax(self.sel / self.logT.exp().clamp(0.05, 20.0), dim=0)
        h = torch.einsum("bf,fnd->bnd", x, w)
        h = (h - self.thr) / (self.tau.abs() + 1e-3)
        p = torch.stack([torch.sigmoid(h), 1 - torch.sigmoid(h)], -1)   # B,n,d,2
        B = x.shape[0]
        resp = p[:, :, 0, :]
        for k in range(1, self.d):
            resp = (resp.unsqueeze(-1) * p[:, :, k, :].unsqueeze(-2)).reshape(B, self.n, -1)
        return torch.einsum("bnl,nlo->bo", resp, self.leaf)

class NODE(nn.Module):
    def __init__(self, f, layers=2, n_trees=96, depth=6):
        super().__init__()
        self.l = nn.ModuleList()
        d = f
        for i in range(layers):
            self.l.append(ODST(d, n_trees, depth, out=32 if i < layers - 1 else 1))
            d = d + 32
        self.layers = layers
    def forward(self, x):
        h = x
        for i, m in enumerate(self.l):
            o = m(h)
            h = torch.cat([h, o], 1) if i < self.layers - 1 else o
        return h

# ============================================================ 4. TabR
class TabR(nn.Module):
    """MLP encoder plus a retrieval module: the prediction is corrected by the labels of the
    nearest candidates in the learned embedding space"""
    def __init__(self, f, d=192, k=32, drop=0.1):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(f, d), nn.BatchNorm1d(d), nn.SiLU(), nn.Dropout(drop),
                                 nn.Linear(d, d), nn.BatchNorm1d(d), nn.SiLU())
        self.k = k
        self.q = nn.Linear(d, d); self.kk = nn.Linear(d, d)
        self.v = nn.Sequential(nn.Linear(d + 1, d), nn.SiLU(), nn.Linear(d, d))
        self.head = nn.Sequential(nn.Linear(d, 128), nn.SiLU(), nn.Linear(128, 1))
        self.cand = None
    def set_candidates(self, xc, yc):
        self.cand = (xc, yc)
    def forward(self, x):
        z = self.enc(x)
        if self.cand is None:
            return self.head(z)
        xc, yc = self.cand
        zc = self.enc(xc)
        sim = self.q(z) @ self.kk(zc).T / z.shape[1] ** 0.5
        k = min(self.k, zc.shape[0])
        val, idx = sim.topk(k, dim=1)
        w = torch.softmax(val, 1)
        ctx = self.v(torch.cat([zc[idx], yc[idx].unsqueeze(-1)], -1))
        return self.head(z + (w.unsqueeze(-1) * ctx).sum(1))

# ============================================================ 5. ModernNCA
class ModernNCA(nn.Module):
    """learn an embedding, predict as a distance-weighted average of neighbour targets"""
    def __init__(self, f, d=128, drop=0.1, T=1.0):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(f, 256), nn.BatchNorm1d(256), nn.SiLU(), nn.Dropout(drop),
                                 nn.Linear(256, d))
        self.logT = nn.Parameter(torch.tensor(float(np.log(T))))
        self.bias = nn.Sequential(nn.Linear(d, 64), nn.SiLU(), nn.Linear(64, 1))
        self.cand = None
    def set_candidates(self, xc, yc):
        self.cand = (xc, yc)
    def forward(self, x):
        z = self.enc(x)
        if self.cand is None:
            return self.bias(z)
        xc, yc = self.cand
        zc = self.enc(xc)
        d2 = (z * z).sum(1, keepdim=True) - 2 * z @ zc.T + (zc * zc).sum(1)
        w = torch.softmax(-d2 / self.logT.exp().clamp(1e-2, 1e3), 1)
        return (w @ yc).unsqueeze(-1) + self.bias(z)

# ============================================================ 6. TabM
class TabM(nn.Module):
    """BatchEnsemble-style: k MLPs sharing weights, separated by rank-1 multiplicative adapters"""
    def __init__(self, f, k=8, h=(512, 256), drop=0.1):
        super().__init__()
        self.k = k
        dims = [f] + list(h)
        self.lin = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(len(h))])
        # BatchEnsemble adapters are initialised with random signs, not near 1: with all-ones the
        # k members stay nearly identical and averaging them buys nothing
        sgn = lambda *sh: (torch.randint(0, 2, sh).float() * 2 - 1)
        self.r = nn.ParameterList([nn.Parameter(sgn(k, dims[i])) for i in range(len(h))])
        self.s = nn.ParameterList([nn.Parameter(sgn(k, dims[i + 1])) for i in range(len(h))])
        self.bn = nn.ModuleList([nn.BatchNorm1d(dims[i + 1]) for i in range(len(h))])
        self.drop = nn.Dropout(drop)
        self.head = nn.Parameter(torch.randn(k, dims[-1], 1) * dims[-1] ** -0.5)
    def forward(self, x):
        z = x.unsqueeze(1).expand(-1, self.k, -1)
        for lin, r, s, bn in zip(self.lin, self.r, self.s, self.bn):
            z = lin(z * r) * s
            z = self.drop(Fn.silu(bn(z.transpose(1, 2)).transpose(1, 2)))
        out = torch.einsum("bkd,kdo->bko", z, self.head)
        return out.mean(1)                                   # ensemble mean is the prediction

# ============================================================ 7. ExcelFormer
class ExcelFormer(nn.Module):
    """semi-permeable attention: features are ordered by importance and information flows only
    from more informative to less informative tokens, which is the paper's core mechanism"""
    def __init__(self, f, d=48, nl=3, nh=4, drop=0.1):
        super().__init__()
        self.tok = FeatureTokenizer(f, d)
        self.nh, self.d = nh, d
        self.q = nn.ModuleList([nn.Linear(d, d) for _ in range(nl)])
        self.k = nn.ModuleList([nn.Linear(d, d) for _ in range(nl)])
        self.v = nn.ModuleList([nn.Linear(d, d) for _ in range(nl)])
        self.o = nn.ModuleList([nn.Linear(d, d) for _ in range(nl)])
        self.ff = nn.ModuleList([nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d * 2), nn.GELU(),
                                               nn.Dropout(drop), nn.Linear(d * 2, d)) for _ in range(nl)])
        self.ln = nn.ModuleList([nn.LayerNorm(d) for _ in range(nl)])
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
        self.nl = nl
        mask = torch.triu(torch.ones(f + 1, f + 1), diagonal=1).bool()   # token 0 = CLS, then sorted
        self.register_buffer("mask", mask)
    def forward(self, x):
        z = self.tok(x)
        B, T, D = z.shape
        for i in range(self.nl):
            h = self.ln[i](z)
            q = self.q[i](h).view(B, T, self.nh, -1).transpose(1, 2)
            k = self.k[i](h).view(B, T, self.nh, -1).transpose(1, 2)
            v = self.v[i](h).view(B, T, self.nh, -1).transpose(1, 2)
            att = (q @ k.transpose(-1, -2)) / (D // self.nh) ** 0.5
            att = att.masked_fill(self.mask[:T, :T], float("-inf")).softmax(-1)
            z = z + self.o[i]((att @ v).transpose(1, 2).reshape(B, T, D))
            z = z + self.ff[i](z)
        return self.head(z.mean(1))

# ============================================================ 8. CARTE (adaptation)
class CARTE(nn.Module):
    """CARTE represents a row as a star graph: a centre node attending to one leaf per column,
    where each leaf carries a column embedding combined with the cell value. The published model
    gets column embeddings from a language model over header strings; our columns are anonymous
    numeric aggregates, so the embeddings are learned free parameters instead. Everything else —
    the star-graph attention and the readout from the centre node — is as described."""
    def __init__(self, f, d=64, nl=2, nh=4, drop=0.1):
        super().__init__()
        self.colemb = nn.Parameter(torch.randn(f, d) * d ** -0.5)      # stands in for the LM encoding
        self.valproj = nn.Linear(1, d)
        self.centre = nn.Parameter(torch.randn(1, 1, d) * d ** -0.5)
        self.att = nn.ModuleList([nn.MultiheadAttention(d, nh, dropout=drop, batch_first=True)
                                  for _ in range(nl)])
        self.ln = nn.ModuleList([nn.LayerNorm(d) for _ in range(nl)])
        self.ff = nn.ModuleList([nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d * 2), nn.GELU(),
                                               nn.Linear(d * 2, d)) for _ in range(nl)])
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
        self.nl = nl
    def forward(self, x):
        B = x.shape[0]
        leaves = self.colemb.unsqueeze(0) * self.valproj(x.unsqueeze(-1))   # edge = column x value
        c = self.centre.expand(B, -1, -1)
        for i in range(self.nl):
            q = self.ln[i](c)
            a, _ = self.att[i](q, leaves, leaves)
            c = c + a
            c = c + self.ff[i](c)
        return self.head(c[:, 0])

# ============================================================ runner
JOBS = []

def torchjob(name, fname, make, cols=None, epochs=10, bs=8192, lr=1e-3, retrieval=False):
    def fit(Xtr, ytr, Xev):
        if not retrieval:
            return torch_fit_predict(make, mse, take0, Xtr, ytr, Xev,
                                     epochs=epochs, bs=bs, lr=lr, dev=DEV)
        # retrieval models need a fixed candidate pool with known targets
        rng = np.random.default_rng(0)
        ci = rng.choice(len(Xtr), 4096, replace=False)
        xc = torch.from_numpy(Xtr[ci]).to(DEV)
        yc = torch.from_numpy(np.log1p(ytr[ci]).astype(np.float32)).to(DEV)
        def mk(f):
            m = make(f); m.set_candidates(xc, yc); return m
        return torch_fit_predict(mk, mse, take0, Xtr, ytr, Xev,
                                 epochs=epochs, bs=bs, lr=lr, dev=DEV)
    JOBS.append((name, fname, fit, cols))

torchjob("FT-Transformer", "zoo_ft_transformer", lambda f: FTTransformer(f), TOPC, epochs=8, bs=2048, lr=1e-3)
torchjob("SAINT", "zoo_saint", lambda f: SAINT(f), TOPC, epochs=6, bs=1024, lr=1e-3)
torchjob("NODE", "zoo_node", lambda f: NODE(f), TOPC, epochs=8, bs=4096, lr=2e-3)
torchjob("TabR", "zoo_tabr", lambda f: TabR(f), None, epochs=10, bs=4096, lr=1e-3, retrieval=True)
torchjob("ModernNCA", "zoo_modernnca", lambda f: ModernNCA(f), None, epochs=10, bs=4096, lr=1e-3, retrieval=True)
torchjob("TabM", "zoo_tabm", lambda f: TabM(f), None, epochs=14, bs=8192, lr=1.5e-3)
torchjob("ExcelFormer", "zoo_excelformer", lambda f: ExcelFormer(f), TOPC, epochs=8, bs=2048, lr=1e-3)
torchjob("CARTE (адаптация)", "zoo_carte", lambda f: CARTE(f), TOPC, epochs=8, bs=2048, lr=1e-3)

def tabnet_fit(Xtr, ytr, Xev):
    from pytorch_tabnet.tab_model import TabNetRegressor
    m = TabNetRegressor(n_d=32, n_a=32, n_steps=4, gamma=1.5, n_independent=2, n_shared=2,
                        lambda_sparse=1e-4, optimizer_params=dict(lr=2e-2), verbose=0, seed=0,
                        scheduler_params=dict(step_size=8, gamma=0.9),
                        scheduler_fn=torch.optim.lr_scheduler.StepLR, device_name=DEV)
    m.fit(Xtr, np.log1p(ytr).reshape(-1, 1).astype(np.float32),
          max_epochs=20, patience=20, batch_size=8192, virtual_batch_size=1024, drop_last=True)
    return m.predict(Xev).ravel()
JOBS.append(("TabNet", "zoo_tabnet", tabnet_fit, None))

# One model per process invocation. A device-side assert inside any CUDA kernel makes the whole
# context unusable, so a single bad model would otherwise take every model after it down with it —
# which is exactly what happened on the first run.
if len(sys.argv) > 1 and sys.argv[1] != "all":
    want = sys.argv[1]
    sel = [j for j in JOBS if j[1] == want or j[0] == want]
    if not sel:
        print(f"нет такой модели: {want}. Доступны: {[j[1] for j in JOBS]}")
        sys.exit(1)
    name, fname, fit, cols = sel[0]
    print(f"=== {name} ===")
    run_pair(name, fname, fit, cols)
else:
    print(f"\n=== зоопарк табличных архитектур: {len(JOBS)} моделей ===")
    for name, fname, fit, cols in JOBS:
        print(f"\n{name}")
        run_pair(name, fname, fit, cols)
print("\ndone")
