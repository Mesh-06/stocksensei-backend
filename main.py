"""
Stock Sensei AI — FastAPI Backend
Architecture exactly matches StockSensei_Universal_TGT_v3.ipynb
"""

import os, json, math, time, warnings, logging, re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests

# ── One-time volume copy ──────────────────────────────────────
_src = Path("base_model.pt")
_dst_dir = Path(os.getenv("MODEL_DIR", "./models"))
_dst_dir.mkdir(parents=True, exist_ok=True)
_dst = _dst_dir / "base_model.pt"
if _src.exists() and not _dst.exists():
    import shutil
    shutil.copy(_src, _dst)
    logging.info("Copied base_model.pt to volume.")

# ── Heavy imports ─────────────────────────────────────────────
try:
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from scipy.stats import pearsonr
    import yfinance as yf
    yf.set_tz_cache_location("/tmp/yfinance_cache")
    import ta, joblib
    HEAVY_OK = True
except ImportError as e:
    HEAVY_OK = False
    logging.warning(f"Heavy imports failed: {e}")

# ─────────────────────────────────────────────────────────────
#  App & CORS
# ─────────────────────────────────────────────────────────────
app = FastAPI(title="Stock Sensei AI", version="3.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────
#  Config — EXACTLY matching notebook
# ─────────────────────────────────────────────────────────────
MODEL_DIR       = Path(os.getenv("MODEL_DIR", "./models"))
BASE_MODEL_PATH = MODEL_DIR / "base_model.pt"
REGISTRY_PATH   = MODEL_DIR / "registry.json"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if (HEAVY_OK and torch.cuda.is_available()) else "cpu"
SEED   = 42

FEATURE_COLS = [
    "Close","log_return","ema_9","ema_21","ema_50","sma_20",
    "macd","macd_sig","macd_diff","adx",
    "rsi_14","stoch_k","stoch_d","cci","williams_r","roc",
    "bb_high","bb_low","bb_mid","bb_width","atr",
    "obv","vwap","mfi","cmf","volatility_5","price_range"
]
N_FEATURES = len(FEATURE_COLS)
CLOSE_IDX  = FEATURE_COLS.index("Close")

BASE_CONFIG = dict(
    seq_len=60, pred_horizon=5,
    hidden_dim=128, num_gru_layers=2, num_heads=4, num_tf_layers=3,
    dropout=0.2, gcn_out_dim=64,
    epochs=120, batch_size=32, lr=3e-4, weight_decay=1e-5,
    patience=18, val_split=0.15, test_split=0.10,
)
FINETUNE_CONFIG = dict(
    seq_len=60, pred_horizon=5,
    hidden_dim=128, num_gru_layers=2, num_heads=4, num_tf_layers=3,
    dropout=0.2, gcn_out_dim=64,
    epochs=20, batch_size=16, lr=8e-5, weight_decay=1e-5,
    patience=6, val_split=0.15, test_split=0.10,
)

SECTOR_PEERS = {
    "Technology":            ["TCS.NS","INFY.NS","WIPRO.NS","HCLTECH.NS","TECHM.NS"],
    "Energy":                ["RELIANCE.NS","ONGC.NS","IOC.NS","BPCL.NS","GAIL.NS"],
    "Financial Services":    ["HDFCBANK.NS","ICICIBANK.NS","SBIN.NS","AXISBANK.NS","KOTAKBANK.NS"],
    "Consumer Defensive":    ["HINDUNILVR.NS","NESTLEIND.NS","BRITANNIA.NS","DABUR.NS"],
    "Healthcare":            ["SUNPHARMA.NS","DRREDDY.NS","CIPLA.NS","DIVISLAB.NS"],
    "Industrials":           ["LT.NS","SIEMENS.NS","ABB.NS","BEL.NS","HAL.NS"],
    "Basic Materials":       ["TATASTEEL.NS","JSWSTEEL.NS","HINDALCO.NS","COALINDIA.NS"],
    "Consumer Cyclical":     ["MARUTI.NS","TATAMOTORS.NS","M&M.NS","BAJAJ-AUTO.NS"],
    "Communication Services":["BHARTIARTL.NS","IDEA.NS","TATACOMM.NS"],
    "US_Technology":         ["AAPL","MSFT","GOOGL","META","NVDA","AMD"],
    "US_Financial":          ["JPM","BAC","GS","MS","WFC"],
    "US_Healthcare":         ["JNJ","PFE","UNH","MRK","ABBV"],
    "US_Energy":             ["XOM","CVX","COP","SLB","EOG"],
    "US_Consumer":           ["AMZN","TSLA","HD","MCD","NKE"],
    "UK_Energy":             ["BP.L","SHEL.L","SSE.L"],
    "UK_Financial":          ["HSBA.L","LLOY.L","BARC.L","NWG.L"],
}
EXCHANGE_REGION = {".NS":"IN",".BO":"IN",".L":"UK",".T":"JP","":"US"}

# ─────────────────────────────────────────────────────────────
#  Pydantic models
# ─────────────────────────────────────────────────────────────
class OHLCVPoint(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float

class AnalyzeRequest(BaseModel):
    ticker: str
    force_retrain: bool = False
    ohlcv: Optional[List[OHLCVPoint]] = None

class PredictionResult(BaseModel):
    ticker: str
    last_close: float
    last_date: str
    predicted_prices: list
    predictions: dict
    change_pct_day1: float
    trend: str
    signal: str
    peers_used: list
    generated_at: str
    metrics: Optional[dict] = None

# ─────────────────────────────────────────────────────────────
#  Registry
# ─────────────────────────────────────────────────────────────
def load_registry():
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH) as f: return json.load(f)
    return {}

def save_registry(reg):
    with open(REGISTRY_PATH, "w") as f: json.dump(reg, f, indent=2)

def register_stock(ticker, metrics, peers, model_path):
    reg = load_registry()
    reg[ticker] = {
        "ticker": ticker, "model_path": str(model_path),
        "peers": peers, "last_trained": datetime.now().isoformat(),
        "metrics": {k: round(float(v), 4) for k, v in metrics.items()},
        "status": "ready",
    }
    save_registry(reg)

# ─────────────────────────────────────────────────────────────
#  ML code (only defined when imports available)
# ─────────────────────────────────────────────────────────────
if HEAVY_OK:

    # ── Model architecture — EXACT copy from notebook ────────
    class GCNLayer(nn.Module):
        def __init__(self, in_f, out_f, adj, drop=0.1):
            super().__init__()
            self.fc = nn.Linear(in_f, out_f)
            self.drop = nn.Dropout(drop)
            self.act = nn.GELU()
            A = torch.tensor(adj, dtype=torch.float32)
            D = A.sum(1); Di = torch.diag(D.pow(-0.5))
            self.register_buffer("AN", Di @ A @ Di)
        def forward(self, x):
            return self.drop(self.act(self.fc(self.AN @ x)))

    class GCNEncoder(nn.Module):
        def __init__(self, in_d, hid, out_d, adj, drop=0.1):
            super().__init__()
            self.g1 = GCNLayer(in_d, hid, adj, drop)
            self.g2 = GCNLayer(hid, out_d, adj, drop)
            self.proj = nn.Linear(in_d, out_d)
            self.norm = nn.LayerNorm(out_d)
        def forward(self, x):
            return self.norm(self.g2(self.g1(x)) + self.proj(x))

    class PosEnc(nn.Module):
        def __init__(self, d, maxlen=512, drop=0.1):
            super().__init__(); self.drop = nn.Dropout(drop)
            pe = torch.zeros(maxlen, d)
            pos = torch.arange(maxlen).unsqueeze(1).float()
            div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000) / d))
            pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
            self.register_buffer("pe", pe.unsqueeze(0))
        def forward(self, x):
            return self.drop(x + self.pe[:, :x.size(1)])

    class TGT(nn.Module):
        def __init__(self, n_feat, n_stocks, adj, gcn_out=64, hidden=128,
                     n_gru=2, n_heads=4, n_tf=3, horizon=5, drop=0.2, seq_len=60):
            super().__init__()
            self.gcn_out = gcn_out; self.hidden = hidden
            self.gcn = GCNEncoder(n_feat, gcn_out * 2, gcn_out, adj, drop)
            self.gru_proj = nn.Linear(n_feat + gcn_out, hidden)
            self.gru = nn.GRU(hidden, hidden, n_gru, batch_first=True,
                              dropout=drop if n_gru > 1 else 0)
            self.gru_norm = nn.LayerNorm(hidden)
            self.tf_proj = nn.Linear(n_feat, hidden)
            self.pe = PosEnc(hidden, seq_len + 10, drop)
            enc = nn.TransformerEncoderLayer(hidden, n_heads, hidden * 4, drop,
                                             batch_first=True, activation="gelu")
            self.tf = nn.TransformerEncoder(enc, n_tf, nn.LayerNorm(hidden))
            self.alpha = nn.Parameter(torch.ones(3) / 3)
            self.head = nn.Sequential(
                nn.Linear(hidden, hidden // 2), nn.GELU(),
                nn.Dropout(drop), nn.Linear(hidden // 2, horizon))
            self._init()

        def _init(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None: nn.init.zeros_(m.bias)

        def replace_adj(self, adj):
            A = torch.tensor(adj, dtype=torch.float32).to(next(self.parameters()).device)
            D = A.sum(1); Di = torch.diag(D.pow(-0.5)); AN = Di @ A @ Di
            for g in [self.gcn.g1, self.gcn.g2]: g.AN = AN

        def forward(self, xg, xs):
            B, T, N, F_ = xg.shape
            gcn_out = self.gcn(xg.view(B * T, N, F_))[:, 0, :].view(B, T, -1)
            gi = self.gru_proj(torch.cat([xs, gcn_out], -1))
            go, _ = self.gru(gi); gl = self.gru_norm(go)[:, -1, :]
            tl = self.tf(self.pe(self.tf_proj(xs)))[:, -1, :]
            pad = self.hidden - self.gcn_out
            gs_ = F.pad(gcn_out.mean(1), (0, pad)) if pad > 0 else gcn_out.mean(1)
            a = torch.softmax(self.alpha, 0)
            return self.head(a[0] * gl + a[1] * tl + a[2] * gs_)

    def build_model(n_stocks, adj, cfg=None):
        cfg = cfg or BASE_CONFIG
        return TGT(
            N_FEATURES, n_stocks, adj,
            gcn_out=cfg["gcn_out_dim"], hidden=cfg["hidden_dim"],
            n_gru=cfg["num_gru_layers"], n_heads=cfg["num_heads"],
            n_tf=cfg["num_tf_layers"], horizon=cfg["pred_horizon"],
            drop=cfg["dropout"], seq_len=cfg["seq_len"],
        ).to(DEVICE)

    # ── Feature engineering — EXACT copy from notebook ───────
    def add_features(df):
        c = df["Close"].squeeze(); h = df["High"].squeeze()
        l = df["Low"].squeeze();   v = df["Volume"].squeeze()
        df["ema_9"]    = ta.trend.EMAIndicator(c, 9).ema_indicator()
        df["ema_21"]   = ta.trend.EMAIndicator(c, 21).ema_indicator()
        df["ema_50"]   = ta.trend.EMAIndicator(c, 50).ema_indicator()
        df["sma_20"]   = ta.trend.SMAIndicator(c, 20).sma_indicator()
        macd = ta.trend.MACD(c)
        df["macd"] = macd.macd(); df["macd_sig"] = macd.macd_signal(); df["macd_diff"] = macd.macd_diff()
        df["adx"]      = ta.trend.ADXIndicator(h, l, c).adx()
        df["rsi_14"]   = ta.momentum.RSIIndicator(c, 14).rsi()
        st = ta.momentum.StochasticOscillator(h, l, c)
        df["stoch_k"] = st.stoch(); df["stoch_d"] = st.stoch_signal()
        df["cci"]      = ta.trend.CCIIndicator(h, l, c).cci()
        df["williams_r"] = ta.momentum.WilliamsRIndicator(h, l, c).williams_r()
        df["roc"]      = ta.momentum.ROCIndicator(c).roc()
        bb = ta.volatility.BollingerBands(c)
        df["bb_high"] = bb.bollinger_hband(); df["bb_low"] = bb.bollinger_lband()
        df["bb_mid"] = bb.bollinger_mavg();   df["bb_width"] = bb.bollinger_wband()
        df["atr"]      = ta.volatility.AverageTrueRange(h, l, c).average_true_range()
        df["obv"]      = ta.volume.OnBalanceVolumeIndicator(c, v).on_balance_volume()
        df["vwap"]     = ta.volume.VolumeWeightedAveragePrice(h, l, c, v).volume_weighted_average_price()
        df["mfi"]      = ta.volume.MFIIndicator(h, l, c, v).money_flow_index()
        df["cmf"]      = ta.volume.ChaikinMoneyFlowIndicator(h, l, c, v).chaikin_money_flow()
        df["log_return"]    = np.log(c / c.shift(1))
        df["volatility_5"]  = df["log_return"].rolling(5).std()
        df["price_range"]   = (h - l) / c
        df.dropna(inplace=True)
        keep = [f for f in FEATURE_COLS if f in df.columns]
        extra = [x for x in ["Open","High","Low","Volume"] if x in df.columns and x not in keep]
        return df[keep + extra]

    # ── Convert frontend OHLCV → DataFrame with features ─────
    def ohlcv_to_df(ohlcv_points):
        records = [{"Date": p.date, "Open": p.open, "High": p.high,
                    "Low": p.low, "Close": p.close, "Volume": p.volume}
                   for p in ohlcv_points]
        df = pd.DataFrame(records)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").sort_index()
        df = add_features(df)
        return df

    # ── Download (for peers only) ─────────────────────────────
    def download_data(tickers, start, end, verbose=True):
        result = {}
        session = requests.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        })
        for t in tickers:
            for attempt in range(3):
                try:
                    yticker = yf.Ticker(t, session=session)
                    df = yticker.history(start=start, end=end, auto_adjust=True)
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    df.dropna(inplace=True)
                    if len(df) < 120:
                        time.sleep(2); continue
                    df = add_features(df)
                    result[t] = df
                    if verbose: logging.info(f"  {t}: {len(df)} rows")
                    break
                except Exception as e:
                    logging.warning(f"  Attempt {attempt+1} failed for {t}: {e}")
                    time.sleep(3)
        return result

    def get_suffix(ticker):
        m = re.search(r"(\.[A-Z]+)$", ticker)
        return m.group(1) if m else ""

    def auto_peers(ticker, n=4):
        suffix = get_suffix(ticker)
        region = EXCHANGE_REGION.get(suffix, "US")
        peers = []
        for key, plist in SECTOR_PEERS.items():
            if any(ticker in plist for _ in [1]):
                peers = [p for p in plist if p != ticker][:n]; break
        if not peers:
            if region == "IN":   peers = ["RELIANCE.NS","TCS.NS","HDFCBANK.NS","INFY.NS"]
            elif region == "UK": peers = ["BP.L","HSBA.L","BARC.L","LLOY.L"]
            else:                peers = ["SPY","QQQ","MSFT","GOOGL"]
        return [p for p in peers if p != ticker][:n]

    def build_graph(processed, threshold=0.25):
        tickers = list(processed.keys()); n = len(tickers)
        rets = {t: processed[t]["log_return"].dropna() for t in tickers if "log_return" in processed[t]}
        common = None
        for s in rets.values(): common = s.index if common is None else common.intersection(s.index)
        corr = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                try:
                    r, _ = pearsonr(rets[tickers[i]].loc[common], rets[tickers[j]].loc[common])
                    corr[i, j] = corr[j, i] = r
                except Exception: pass
        adj = (np.abs(corr) > threshold).astype(float); np.fill_diagonal(adj, 1.0)
        return adj, tickers, corr

    def scale_data(processed, target, fit=True, existing=None):
        common = None
        for df in processed.values():
            idx = df[[f for f in FEATURE_COLS if f in df.columns]].dropna().index
            common = idx if common is None else common.intersection(idx)
        scalers, scaled = {}, {}
        for t, df in processed.items():
            fc = [f for f in FEATURE_COLS if f in df.columns]
            arr = df.loc[common, fc].values
            if fit or existing is None:
                sc = MinMaxScaler((0, 1)); arr = sc.fit_transform(arr)
            else:
                sc = existing.get(t, MinMaxScaler((0, 1)).fit(arr)); arr = sc.transform(arr)
            scalers[t] = sc; scaled[t] = arr
        csc = MinMaxScaler()
        cv = processed[target].loc[common, ["Close"]].values
        if fit or existing is None: csc.fit(cv)
        else: csc = existing.get("__close__", MinMaxScaler().fit(cv))
        scalers["__close__"] = csc
        return scaled, scalers, common

    class StockDataset(Dataset):
        def __init__(self, scaled, tickers, target, seq_len, horizon):
            self.seq_len = seq_len; self.horizon = horizon
            self.target_idx = tickers.index(target)
            T = scaled[tickers[0]].shape[0]
            self.data = np.stack([scaled[t] for t in tickers], axis=1)
            self.idx = list(range(seq_len, T - horizon + 1))
        def __len__(self): return len(self.idx)
        def __getitem__(self, i):
            t = self.idx[i]
            xg = torch.tensor(self.data[t - self.seq_len:t], dtype=torch.float32)
            xs = xg[:, self.target_idx, :]
            y = torch.tensor(self.data[t:t + self.horizon, self.target_idx, CLOSE_IDX], dtype=torch.float32)
            return xg, xs, y

    def make_loaders(scaled, tickers, target, cfg):
        ds = StockDataset(scaled, tickers, target, cfg["seq_len"], cfg["pred_horizon"])
        N = len(ds)
        nte = max(1, int(N * cfg.get("test_split", 0.10)))
        nva = max(1, int(N * cfg.get("val_split", 0.15)))
        ntr = N - nva - nte
        tr, va, te = torch.utils.data.random_split(
            ds, [ntr, nva, nte], generator=torch.Generator().manual_seed(SEED))
        bs = cfg["batch_size"]
        return (DataLoader(tr, batch_size=bs, shuffle=True, drop_last=True),
                DataLoader(va, batch_size=bs, shuffle=False),
                DataLoader(te, batch_size=bs, shuffle=False),
                ntr, nva, nte)

    class HybridLoss(nn.Module):
        def __init__(self, a=0.7):
            super().__init__(); self.a = a
            self.mse = nn.MSELoss(); self.mae = nn.L1Loss()
        def forward(self, p, t): return self.a * self.mse(p, t) + (1 - self.a) * self.mae(p, t)

    def train_model(model, tr_l, va_l, cfg, label=""):
        for p in model.parameters(): p.requires_grad = True
        crit = HybridLoss()
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 1e-5))
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=cfg["lr"] * 8, epochs=cfg["epochs"],
            steps_per_epoch=len(tr_l), pct_start=0.3, anneal_strategy="cos")
        best_val, best_state, pat = float("inf"), None, 0
        for ep in range(1, cfg["epochs"] + 1):
            model.train(); tl = 0
            for xg, xs, yb in tr_l:
                xg, xs, yb = xg.to(DEVICE), xs.to(DEVICE), yb.to(DEVICE)
                loss = crit(model(xg, xs), yb)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); tl += loss.item() * xg.size(0)
            model.eval(); vl = 0
            with torch.no_grad():
                for xg, xs, yb in va_l:
                    vl += crit(model(xg.to(DEVICE), xs.to(DEVICE)), yb.to(DEVICE)).item() * xg.size(0)
            tl /= len(tr_l.dataset); vl /= len(va_l.dataset)
            if vl < best_val:
                best_val = vl
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                pat = 0
            else:
                pat += 1
            interval = max(1, cfg["epochs"] // 6)
            if ep % interval == 0 or ep == 1:
                logging.info(f"  [{label}] Ep {ep}/{cfg['epochs']} train={tl:.5f} val={vl:.5f}")
            if pat >= cfg["patience"]:
                logging.info(f"  Early stop @ epoch {ep}"); break
        model.load_state_dict(best_state); model.eval()
        return best_val

    def compute_metrics(t, p):
        mae  = mean_absolute_error(t, p)
        rmse = np.sqrt(mean_squared_error(t, p))
        mape = np.mean(np.abs((t - p) / (np.abs(t) + 1e-8))) * 100
        r2   = r2_score(t, p)
        corr = np.corrcoef(t.ravel(), p.ravel())[0, 1]
        da   = (np.sign(np.diff(t)) == np.sign(np.diff(p))).mean() * 100
        return dict(MAE=mae, RMSE=rmse, MAPE=mape, R2=r2, Correlation=corr, DirectionalAccuracy=da)

    def get_predictions(model, loader, close_scaler):
        model.eval(); ps, ts = [], []
        with torch.no_grad():
            for xg, xs, yb in loader:
                ps.append(model(xg.to(DEVICE), xs.to(DEVICE)).cpu().numpy())
                ts.append(yb.numpy())
        ps = np.concatenate(ps); ts = np.concatenate(ts)
        p_inv = close_scaler.inverse_transform(ps[:, 0:1]).ravel()
        t_inv = close_scaler.inverse_transform(ts[:, 0:1]).ravel()
        return ps, ts, p_inv, t_inv

    def _run_prediction(ticker, model, scalers, graph_t, raw_override=None):
        end   = datetime.today().strftime("%Y-%m-%d")
        start = (datetime.today() - timedelta(days=400)).strftime("%Y-%m-%d")
        if raw_override:
            fresh = {t: raw_override[t] for t in graph_t if t in raw_override}
        else:
            fresh = download_data(graph_t, start, end, verbose=False)

        common = None
        for t in fresh:
            idx = fresh[t][[f for f in FEATURE_COLS if f in fresh[t].columns]].dropna().index
            common = idx if common is None else common.intersection(idx)

        seq_len = 60
        if common is None or len(common) < seq_len:
            raise ValueError(f"Not enough data ({len(common) if common is not None else 0} rows).")

        def _sc(t):
            fc  = [f for f in FEATURE_COLS if f in fresh[t].columns]
            arr = fresh[t].loc[common, fc].values
            return scalers[t].transform(arr) if t in scalers else MinMaxScaler((0, 1)).fit_transform(arr)

        sf   = {t: _sc(t) for t in graph_t if t in fresh}
        x_g  = np.stack([sf[t][-seq_len:] for t in graph_t if t in sf], axis=1)
        x_g  = torch.tensor(x_g, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        x_s  = torch.tensor(sf[graph_t[0]][-seq_len:], dtype=torch.float32).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            ps = model(x_g, x_s).cpu().numpy()

        preds = scalers["__close__"].inverse_transform(ps).ravel()
        lc    = float(np.array(fresh[graph_t[0]]["Close"].squeeze().values).ravel()[-1])
        ld    = common[-1]
        fut   = pd.bdate_range(start=ld, periods=len(preds) + 1)[1:]

        def _f(x): return round(float(x), 4)
        prices = [_f(p) for p in preds]
        return {
            "ticker":           ticker,
            "last_close":       _f(lc),
            "last_date":        str(ld.date()),
            "predicted_prices": prices,
            "predictions":      {d.strftime("%Y-%m-%d"): p for d, p in zip(fut, prices)},
            "change_pct_day1":  _f((prices[0] - lc) / (lc + 1e-8) * 100),
            "trend":            "UPTREND" if prices[-1] > prices[0] else "DOWNTREND",
            "signal":           "BUY" if prices[0] > lc else "SELL",
            "peers_used":       graph_t[1:],
            "generated_at":     datetime.now().isoformat(),
        }

    def run_fine_tune(ticker: str, force_retrain: bool = False, ohlcv_points=None) -> dict:
        ticker = ticker.upper()
        mp = MODEL_DIR / f"{ticker.replace('.','_')}_model.pt"
        sp = MODEL_DIR / f"{ticker.replace('.','_')}_scalers.pkl"
        reg = load_registry()

        # ── Already trained → quick predict ──────────────────
        if ticker in reg and not force_retrain and mp.exists() and sp.exists():
            ckpt = torch.load(mp, map_location=DEVICE)
            art  = joblib.load(sp)
            sc   = art["scalers"]; gt = art["graph_tickers"]
            cfg  = ckpt.get("base_config", BASE_CONFIG)
            m    = build_model(len(gt), np.array(ckpt["adj_matrix"]), cfg)
            m.load_state_dict(ckpt["model_state"]); m.eval()
            raw_override = None
            if ohlcv_points and len(ohlcv_points) >= 5:
                provided_df = ohlcv_to_df(ohlcv_points)
                raw_override = {ticker: provided_df}
            result = _run_prediction(ticker, m, sc, gt, raw_override)
            result["metrics"] = reg[ticker].get("metrics")
            return result

        # ── New stock: fine-tune ──────────────────────────────
        DATA_START = "2019-01-01"
        DATA_END   = datetime.today().strftime("%Y-%m-%d")
        peers      = auto_peers(ticker, n=4)
        all_t      = [ticker] + peers

        if ohlcv_points and len(ohlcv_points) >= 60:
            logging.info(f"  Using frontend OHLCV for {ticker} ({len(ohlcv_points)} points)")
            provided_df = ohlcv_to_df(ohlcv_points)
            raw   = {ticker: provided_df}
            peers = []
            all_t = [ticker]
        else:
            raw = download_data(all_t, DATA_START, DATA_END)

        avail = [t for t in all_t if t in raw]
        if ticker not in raw:
            raise ValueError(f"Cannot download data for {ticker}. Check the ticker symbol.")

        adj, graph_t, _ = build_graph({t: raw[t] for t in avail})
        scaled, scalers, _ = scale_data({t: raw[t] for t in graph_t}, ticker, fit=True)
        tr_l, va_l, te_l, ntr, nva, nte = make_loaders(scaled, graph_t, ticker, FINETUNE_CONFIG)

        if not BASE_MODEL_PATH.exists():
            raise FileNotFoundError("Base model not found. Upload base_model.pt to the models/ directory.")

        ckpt     = torch.load(BASE_MODEL_PATH, map_location=DEVICE)
        base_cfg = ckpt.get("base_config", BASE_CONFIG)
        base_n   = len(ckpt["base_tickers"])

        ft_model = build_model(base_n, np.array(ckpt["adj_matrix"]), base_cfg)
        ft_model.load_state_dict(ckpt["model_state"])

        if len(graph_t) != base_n:
            ft_model.gcn = GCNEncoder(
                N_FEATURES, base_cfg["gcn_out_dim"] * 2,
                base_cfg["gcn_out_dim"], adj, base_cfg["dropout"],
            ).to(DEVICE)
            ft_model.alpha = nn.Parameter(torch.ones(3) / 3).to(DEVICE)
        else:
            ft_model.replace_adj(adj)

        train_model(ft_model, tr_l, va_l, FINETUNE_CONFIG, label=ticker)

        _, _, te_l2, _, _, _ = make_loaders(scaled, graph_t, ticker, FINETUNE_CONFIG)
        _, _, te_pi, te_ti   = get_predictions(ft_model, te_l2, scalers["__close__"])
        te_m                  = compute_metrics(te_ti, te_pi)

        torch.save({
            "model_state":   ft_model.state_dict(),
            "base_config":   base_cfg,
            "graph_tickers": graph_t,
            "adj_matrix":    adj.tolist(),
            "ticker":        ticker,
            "peers":         peers,
            "metrics":       te_m,
            "fine_tuned_on": datetime.now().isoformat(),
        }, mp)
        joblib.dump({"scalers": scalers, "feature_cols": FEATURE_COLS,
                     "graph_tickers": graph_t}, sp)
        register_stock(ticker, te_m, peers, mp)

        result = _run_prediction(ticker, ft_model, scalers, graph_t, raw)
        result["metrics"] = te_m
        return result

# ─────────────────────────────────────────────────────────────
#  Routes
# ─────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "heavy_imports": HEAVY_OK,
        "device": DEVICE,
        "base_model_exists": BASE_MODEL_PATH.exists(),
        "registered_stocks": list(load_registry().keys()),
    }

@app.get("/registry")
def get_registry():
    return load_registry()

@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    if not HEAVY_OK:
        raise HTTPException(503, "ML libraries not installed.")
    try:
        result = run_fine_tune(req.ticker.upper().strip(), req.force_retrain, req.ohlcv)
        return result
    except FileNotFoundError as e:
        raise HTTPException(503, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logging.exception("Prediction error")
        raise HTTPException(500, f"Prediction failed: {str(e)}")

@app.get("/predict/{ticker}")
def predict(ticker: str, force_retrain: bool = False):
    return analyze(AnalyzeRequest(ticker=ticker, force_retrain=force_retrain))
