"""
Stock Sensei AI — FastAPI Backend
Deploy on Railway (free tier) or any Python host.
Serves the TGT model predictions to the Vercel frontend.
"""
import shutil
from pathlib import Path

# One-time copy of base_model.pt to persistent volume
_src = Path("base_model.pt")
_dst = Path("/app/models/base_model.pt")
if _src.exists() and not _dst.exists():
    print("Copying base_model.pt to volume...")
    _dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(_src, _dst)
    print("✓ base_model.pt copied to volume.")

import os
import json
import math
import time
import warnings
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore")

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Optional heavy imports (only needed when model is loaded) ──
try:
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from sklearn.preprocessing import MinMaxScaler
    import yfinance as yf
    import ta
    import joblib
    from scipy.stats import pearsonr
    HEAVY_IMPORTS_OK = True
except ImportError as e:
    HEAVY_IMPORTS_OK = False
    logging.warning(f"Heavy ML imports failed: {e}. Install requirements.txt first.")

# ─────────────────────────────────────────────────────────────
#  App & CORS
# ─────────────────────────────────────────────────────────────
app = FastAPI(title="Stock Sensei AI", version="3.0.0")

ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:5173,http://localhost:3000,https://stock-sensei-ai-05.vercel.app"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten to ALLOWED_ORIGINS after confirming your Vercel URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────
#  Paths & Config
# ─────────────────────────────────────────────────────────────
MODEL_DIR      = Path(os.getenv("MODEL_DIR", "./models"))
BASE_MODEL_PATH = MODEL_DIR / "base_model.pt"
REGISTRY_PATH  = MODEL_DIR / "registry.json"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if (HEAVY_IMPORTS_OK and torch.cuda.is_available()) else "cpu"

N_FEATURES = 18   # must match notebook
SEQ_LEN    = 30

BASE_CONFIG = {
    "seq_len": SEQ_LEN, "pred_len": 1,
    "gcn_out_dim": 32, "gru_hidden": 64,
    "transformer_dim": 64, "nhead": 4,
    "dropout": 0.15, "lr": 3e-4,
    "epochs": 60, "batch_size": 64,
    "patience": 12, "weight_decay": 1e-5,
}

FINETUNE_CONFIG = {**BASE_CONFIG, "epochs": 30, "lr": 5e-5, "patience": 8}

FEATURE_COLS = [
    "Close","Open","High","Low","Volume",
    "rsi","macd","macd_signal","bb_upper","bb_lower","bb_mid",
    "ema_20","ema_50","atr","obv","cci","stoch_k","return_1d",
]

# ─────────────────────────────────────────────────────────────
#  Pydantic Models
# ─────────────────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    ticker: str
    force_retrain: bool = False

class PredictionResult(BaseModel):
    ticker: str
    last_close: float
    last_date: str
    predicted_prices: list[float]
    predictions: dict
    change_pct_day1: float
    trend: str
    signal: str
    peers_used: list[str]
    generated_at: str
    metrics: Optional[dict] = None

# ─────────────────────────────────────────────────────────────
#  Registry helpers
# ─────────────────────────────────────────────────────────────
def load_registry() -> dict:
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH) as f:
            return json.load(f)
    return {}

def save_registry(reg: dict):
    with open(REGISTRY_PATH, "w") as f:
        json.dump(reg, f, indent=2)

def register_stock(ticker, metrics, peers, model_path):
    reg = load_registry()
    reg[ticker] = {
        "ticker": ticker,
        "model_path": str(model_path),
        "peers": peers,
        "last_trained": datetime.now().isoformat(),
        "metrics": {k: round(float(v), 4) for k, v in metrics.items()},
        "status": "ready",
    }
    save_registry(reg)

# ─────────────────────────────────────────────────────────────
#  ML Utilities  (only executed when HEAVY_IMPORTS_OK)
# ─────────────────────────────────────────────────────────────
if HEAVY_IMPORTS_OK:

    # ── Data download ──────────────────────────────────────────
    def download_data(tickers, start, end, verbose=True):
        raw = {}
        for t in tickers:
            try:
                df = yf.download(t, start=start, end=end, progress=False, auto_adjust=True)
                if len(df) > 60:
                    raw[t] = df
                    if verbose:
                        logging.info(f"  Downloaded {t}: {len(df)} rows")
            except Exception as e:
                logging.warning(f"  Failed {t}: {e}")
        return raw

    # ── Feature engineering ────────────────────────────────────
    def add_features(df):
        d = df.copy()
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        for col in ["Close","Open","High","Low","Volume"]:
            if col not in d.columns:
                d[col] = 0.0
        cl = d["Close"].squeeze()
        d["rsi"]         = ta.momentum.RSIIndicator(cl, window=14).rsi()
        macd_obj         = ta.trend.MACD(cl)
        d["macd"]        = macd_obj.macd()
        d["macd_signal"] = macd_obj.macd_signal()
        bb               = ta.volatility.BollingerBands(cl, window=20)
        d["bb_upper"]    = bb.bollinger_hband()
        d["bb_lower"]    = bb.bollinger_lband()
        d["bb_mid"]      = bb.bollinger_mavg()
        d["ema_20"]      = ta.trend.EMAIndicator(cl, window=20).ema_indicator()
        d["ema_50"]      = ta.trend.EMAIndicator(cl, window=50).ema_indicator()
        d["atr"]         = ta.volatility.AverageTrueRange(d["High"].squeeze(), d["Low"].squeeze(), cl).average_true_range()
        d["obv"]         = ta.volume.OnBalanceVolumeIndicator(cl, d["Volume"].squeeze()).on_balance_volume()
        d["cci"]         = ta.trend.CCIIndicator(d["High"].squeeze(), d["Low"].squeeze(), cl).cci()
        d["stoch_k"]     = ta.momentum.StochasticOscillator(d["High"].squeeze(), d["Low"].squeeze(), cl).stoch()
        d["return_1d"]   = cl.pct_change()
        d = d[FEATURE_COLS].copy()
        d.replace([np.inf, -np.inf], np.nan, inplace=True)
        d.ffill(inplace=True)
        d.bfill(inplace=True)
        d.dropna(inplace=True)
        return d

    # ── Graph builder ──────────────────────────────────────────
    def build_graph(raw_dict, threshold=0.5):
        tickers = list(raw_dict.keys())
        closes  = {}
        for t in tickers:
            df = raw_dict[t]
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            closes[t] = df["Close"].squeeze()
        common = sorted(set.intersection(*[set(closes[t].index) for t in tickers]))
        mat    = pd.DataFrame({t: closes[t].loc[common] for t in tickers})
        n      = len(tickers)
        corr   = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    corr[i, j] = 1.0
                elif j > i:
                    try:
                        r, _ = pearsonr(mat.iloc[:, i].values, mat.iloc[:, j].values)
                        corr[i, j] = corr[j, i] = abs(r)
                    except Exception:
                        corr[i, j] = corr[j, i] = 0.0
        adj = (corr >= threshold).astype(float)
        np.fill_diagonal(adj, 1.0)
        return adj, tickers, corr

    # ── Scaling ────────────────────────────────────────────────
    def scale_data(raw_dict, target, fit=True):
        tickers = list(raw_dict.keys())
        feat    = {t: add_features(raw_dict[t]) for t in tickers}
        common  = sorted(set.intersection(*[set(feat[t].index) for t in tickers]))
        scalers = {}
        scaled  = {}
        for t in tickers:
            scaler = MinMaxScaler()
            data   = feat[t].loc[common]
            if fit:
                scaled[t] = scaler.fit_transform(data.values)
            else:
                scaled[t] = scaler.transform(data.values)
            scalers[t] = scaler
        close_scaler = MinMaxScaler()
        close_col    = feat[target]["Close"].loc[common].values.reshape(-1, 1)
        close_scaler.fit_transform(close_col)
        scalers["__close__"] = close_scaler
        return scaled, scalers, common

    # ── Dataset ────────────────────────────────────────────────
    class StockDataset(Dataset):
        def __init__(self, scaled_dict, tickers, target, seq_len, pred_len=1):
            n, f = scaled_dict[target].shape
            self.X, self.y = [], []
            for i in range(seq_len, n - pred_len + 1):
                x = np.stack([scaled_dict[t][i - seq_len: i] for t in tickers], axis=0)
                y = scaled_dict[target][i: i + pred_len, 0]
                self.X.append(x)
                self.y.append(y)
            self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
            self.y = torch.tensor(np.array(self.y), dtype=torch.float32)

        def __len__(self):  return len(self.X)
        def __getitem__(self, i): return self.X[i], self.y[i]

    def make_loaders(scaled, tickers, target, cfg):
        ds  = StockDataset(scaled, tickers, target, cfg["seq_len"], cfg["pred_len"])
        n   = len(ds)
        ntr = int(n * 0.7)
        nva = int(n * 0.15)
        nte = n - ntr - nva
        tr, va, te = torch.utils.data.random_split(ds, [ntr, nva, nte])
        return (
            DataLoader(tr, batch_size=cfg["batch_size"], shuffle=True),
            DataLoader(va, batch_size=cfg["batch_size"]),
            DataLoader(te, batch_size=cfg["batch_size"]),
            ntr, nva, nte,
        )

    # ── Model ──────────────────────────────────────────────────
    class GCNEncoder(nn.Module):
        def __init__(self, in_f, hidden, out_f, adj, dropout=0.15):
            super().__init__()
            self.adj = torch.tensor(adj, dtype=torch.float32).to(DEVICE)
            self.fc1 = nn.Linear(in_f, hidden)
            self.fc2 = nn.Linear(hidden, out_f)
            self.drop = nn.Dropout(dropout)
        def forward(self, x):
            # x: (B, N, seq, F)
            B, N, S, F = x.shape
            a = self.adj.unsqueeze(0).expand(B, -1, -1)
            h = x.mean(dim=2)              # (B, N, F)
            h = torch.bmm(a, h)            # (B, N, F)
            h = F.relu(self.fc1(h))
            h = self.drop(h)
            h = torch.bmm(a, h)
            h = self.fc2(h)
            return h[:, 0, :]              # target node

    class TGTModel(nn.Module):
        def __init__(self, n_nodes, adj, cfg):
            super().__init__()
            d = cfg["gcn_out_dim"]
            self.gcn = GCNEncoder(N_FEATURES, d * 2, d, adj, cfg["dropout"])
            self.gru = nn.GRU(N_FEATURES, cfg["gru_hidden"], batch_first=True, num_layers=2,
                              dropout=cfg["dropout"])
            enc_layer = nn.TransformerEncoderLayer(d_model=cfg["transformer_dim"],
                                                   nhead=cfg["nhead"], dropout=cfg["dropout"],
                                                   batch_first=True)
            self.transformer = nn.TransformerEncoder(enc_layer, num_layers=2)
            self.proj = nn.Linear(N_FEATURES, cfg["transformer_dim"])
            total_in = d + cfg["gru_hidden"] + cfg["transformer_dim"]
            self.alpha = nn.Parameter(torch.ones(3) / 3)
            self.head = nn.Sequential(
                nn.Linear(total_in, 64), nn.SiLU(), nn.Dropout(cfg["dropout"]),
                nn.Linear(64, 32),       nn.SiLU(),
                nn.Linear(32, cfg["pred_len"]),
            )

        def replace_adj(self, new_adj):
            self.gcn.adj = torch.tensor(new_adj, dtype=torch.float32).to(DEVICE)

        def forward(self, x):
            # x: (B, N, seq, F)
            B, N, S, F = x.shape
            gcn_out = self.gcn(x)                         # (B, d)
            gru_in  = x[:, 0, :, :]                       # (B, seq, F)
            gru_out, _ = self.gru(gru_in)
            gru_out = gru_out[:, -1, :]                   # (B, gru_hidden)
            tr_in   = self.proj(gru_in)                   # (B, seq, tr_dim)
            tr_out  = self.transformer(tr_in)[:, -1, :]   # (B, tr_dim)
            w = F.softmax(self.alpha, dim=0)
            fused = w[0] * gcn_out + w[1] * gru_out + w[2] * tr_out
            return self.head(fused)

    def build_model(n_nodes, adj, cfg):
        return TGTModel(n_nodes, adj, cfg).to(DEVICE)

    # ── Training ───────────────────────────────────────────────
    def train_model(model, tr_loader, va_loader, cfg, label="", freeze_backbone=False):
        if freeze_backbone:
            for name, p in model.named_parameters():
                if "head" not in name:
                    p.requires_grad = False
        opt = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg["lr"], weight_decay=cfg["weight_decay"],
        )
        sched   = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=cfg["lr"], steps_per_epoch=len(tr_loader), epochs=cfg["epochs"],
        )
        loss_fn = nn.HuberLoss()
        best_val = float("inf")
        best_state = None
        patience_ctr = 0
        history = {"train": [], "val": []}

        for epoch in range(cfg["epochs"]):
            model.train()
            tr_loss = 0
            for xb, yb in tr_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                opt.zero_grad(); loss.backward(); opt.step(); sched.step()
                tr_loss += loss.item()
            tr_loss /= len(tr_loader)

            model.eval()
            va_loss = 0
            with torch.no_grad():
                for xb, yb in va_loader:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    va_loss += loss_fn(model(xb), yb).item()
            va_loss /= len(va_loader)
            history["train"].append(tr_loss)
            history["val"].append(va_loss)

            if va_loss < best_val:
                best_val = va_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= cfg["patience"]:
                    break

        if best_state:
            model.load_state_dict(best_state)
        return history, best_val

    # ── Metrics ────────────────────────────────────────────────
    def compute_metrics(actual, predicted):
        a, p = np.array(actual), np.array(predicted)
        rmse = float(np.sqrt(np.mean((a - p) ** 2)))
        mape = float(np.mean(np.abs((a - p) / (np.abs(a) + 1e-8))) * 100)
        dir_acc = float(np.mean(np.sign(np.diff(a)) == np.sign(np.diff(p))) * 100)
        return {"RMSE": rmse, "MAPE": mape, "DirectionalAccuracy": dir_acc}

    def get_predictions(model, loader, close_scaler):
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb in loader:
                out = model(xb.to(DEVICE)).cpu().numpy()
                preds.append(out)
                trues.append(yb.numpy())
        preds = np.concatenate(preds)
        trues = np.concatenate(trues)
        p_inv = close_scaler.inverse_transform(preds)
        t_inv = close_scaler.inverse_transform(trues)
        return preds, trues, p_inv.squeeze(), t_inv.squeeze()

    # ── Auto peer detection ────────────────────────────────────
    SECTOR_PEERS = {
        "tech":    ["TCS.NS","INFY.NS","WIPRO.NS","TECHM.NS","HCLTECH.NS","AAPL","MSFT","GOOGL","META","NVDA"],
        "bank":    ["HDFCBANK.NS","ICICIBANK.NS","AXISBANK.NS","KOTAKBANK.NS","SBIN.NS","JPM","BAC","WFC","GS"],
        "energy":  ["ONGC.NS","RELIANCE.NS","BPCL.NS","IOC.NS","XOM","CVX","BP.L","SHEL.L"],
        "pharma":  ["SUNPHARMA.NS","DRREDDY.NS","CIPLA.NS","DIVISLAB.NS","JNJ","PFE","MRK","ABBV"],
        "auto":    ["MARUTI.NS","TATAMOTORS.NS","BAJAJ-AUTO.NS","HEROMOTOCO.NS","TSLA","F","GM","TM"],
        "fmcg":    ["HINDUNILVR.NS","ITC.NS","NESTLEIND.NS","BRITANNIA.NS","PG","KO","PEP","NESN.SW"],
        "metal":   ["TATASTEEL.NS","JSWSTEEL.NS","HINDALCO.NS","VEDL.NS","NUE","X","AA"],
        "telecom": ["BHARTIARTL.NS","VIL.NS","RELIANCE.NS","T","VZ","VOD.L"],
        "realty":  ["DLF.NS","GODREJPROP.NS","OBEROIRLTY.NS","PLD","AMT","SPG"],
        "us_tech": ["AAPL","MSFT","GOOGL","META","NVDA","AMZN","TSLA","AMD","INTC","CRM"],
    }

    def auto_peers(ticker, n=4):
        t = ticker.upper()
        for sector, peers in SECTOR_PEERS.items():
            if t in peers:
                return [p for p in peers if p != t][:n]
        # fallback
        if ".NS" in t or ".BO" in t:
            return ["TCS.NS","INFY.NS","HDFCBANK.NS","RELIANCE.NS"][:n]
        return ["AAPL","MSFT","GOOGL","AMZN"][:n]

    # ── Core prediction ────────────────────────────────────────
    def _run_prediction(ticker, model, scalers, graph_tickers, raw_dict=None):
        ticker = ticker.upper()
        if raw_dict is None:
            end   = datetime.today().strftime("%Y-%m-%d")
            start = (datetime.today() - timedelta(days=200)).strftime("%Y-%m-%d")
            raw_dict = download_data(graph_tickers, start, end, verbose=False)

        df = raw_dict.get(ticker)
        if df is None:
            raise ValueError(f"No data for {ticker}")

        feat       = add_features(df)
        close_sc   = scalers["__close__"]
        target_sc  = scalers.get(ticker, scalers.get(list(scalers.keys())[0]))
        raw_seq    = feat[FEATURE_COLS].values[-SEQ_LEN:]
        scaled_seq = target_sc.transform(raw_seq)
        # build fake multi-node tensor using only target (simplification for inference)
        x = torch.tensor(scaled_seq, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
        # replicate for n_nodes
        n_nodes = model.gcn.adj.shape[0]
        x = x.expand(-1, n_nodes, -1, -1)

        model.eval()
        prices = []
        with torch.no_grad():
            inp = x.clone()
            for _ in range(5):
                out = model(inp)
                price_sc = out.cpu().numpy()[0, 0]
                price    = float(close_sc.inverse_transform([[price_sc]])[0, 0])
                prices.append(price)
                # shift window
                new_row = inp[:, :, -1:, :].clone()
                new_row[:, 0, 0, 0] = out.squeeze()
                inp = torch.cat([inp[:, :, 1:, :], new_row], dim=2)

        lc        = float(feat["Close"].iloc[-1])
        last_date = str(feat.index[-1].date())
        _f        = lambda v: round(v, 4)

        future_dates = []
        d = datetime.strptime(last_date, "%Y-%m-%d")
        count = 0
        while count < 5:
            d += timedelta(days=1)
            if d.weekday() < 5:
                future_dates.append(d.strftime("%Y-%m-%d"))
                count += 1

        predictions = {dt: round(p, 2) for dt, p in zip(future_dates, prices)}

        return {
            "ticker":          ticker,
            "last_close":      round(lc, 2),
            "last_date":       last_date,
            "predicted_prices": [round(p, 2) for p in prices],
            "predictions":     predictions,
            "change_pct_day1": _f((prices[0] - lc) / (lc + 1e-8) * 100),
            "trend":           "UPTREND" if prices[-1] > prices[0] else "DOWNTREND",
            "signal":          "BUY" if prices[0] > lc else "SELL",
            "peers_used":      graph_tickers[1:],
            "generated_at":    datetime.now().isoformat(),
        }

    # ── Fine-tune flow ─────────────────────────────────────────
    def run_fine_tune(ticker: str, force_retrain: bool = False) -> dict:
        ticker = ticker.upper()
        mp = MODEL_DIR / f"{ticker.replace('.','_')}_model.pt"
        sp = MODEL_DIR / f"{ticker.replace('.','_')}_scalers.pkl"

        reg = load_registry()
        if ticker in reg and not force_retrain and mp.exists() and sp.exists():
            # load and predict
            ckpt = torch.load(mp, map_location=DEVICE)
            art  = joblib.load(sp)
            sc   = art["scalers"]
            gt   = art["graph_tickers"]
            m    = build_model(len(gt), np.array(ckpt["adj_matrix"]), ckpt["base_config"])
            m.load_state_dict(ckpt["model_state"])
            result = _run_prediction(ticker, m, sc, gt)
            result["metrics"] = reg[ticker].get("metrics")
            return result

        DATA_START = "2019-01-01"
        DATA_END   = datetime.today().strftime("%Y-%m-%d")
        peers      = auto_peers(ticker, n=4)
        all_t      = [ticker] + peers
        raw        = download_data(all_t, DATA_START, DATA_END)
        avail      = [t for t in all_t if t in raw]
        if ticker not in raw:
            raise ValueError(f"Cannot download data for {ticker}. Check the ticker symbol.")

        adj, graph_t, _ = build_graph({t: raw[t] for t in avail})
        scaled, scalers, _ = scale_data({t: raw[t] for t in graph_t}, ticker, fit=True)
        tr_l, va_l, te_l, ntr, nva, nte = make_loaders(scaled, graph_t, ticker, FINETUNE_CONFIG)

        if not BASE_MODEL_PATH.exists():
            raise FileNotFoundError(
                "Base model not found. Run Phase 1 training in the notebook first, "
                "then upload base_model.pt to the models/ directory."
            )

        ckpt     = torch.load(BASE_MODEL_PATH, map_location=DEVICE)
        base_cfg = ckpt["base_config"]
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
            "model_state":  ft_model.state_dict(),
            "base_config":  base_cfg,
            "graph_tickers": graph_t,
            "adj_matrix":   adj.tolist(),
            "ticker":       ticker,
            "peers":        peers,
            "metrics":      te_m,
            "fine_tuned_on": datetime.now().isoformat(),
        }, mp)
        joblib.dump({
            "scalers": scalers,
            "feature_cols": FEATURE_COLS,
            "graph_tickers": graph_t,
        }, sp)
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
        "heavy_imports": HEAVY_IMPORTS_OK,
        "device": DEVICE,
        "base_model_exists": BASE_MODEL_PATH.exists(),
        "registered_stocks": list(load_registry().keys()),
    }

@app.get("/registry")
def get_registry():
    return load_registry()

@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    if not HEAVY_IMPORTS_OK:
        raise HTTPException(503, "ML libraries not installed. Run: pip install -r requirements.txt")
    try:
        result = run_fine_tune(req.ticker.upper().strip(), req.force_retrain)
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
    """GET convenience endpoint — same as POST /analyze"""
    return analyze(AnalyzeRequest(ticker=ticker, force_retrain=force_retrain))
