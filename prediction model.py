import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer
from datetime import datetime
import pytz
import bisect

# --- 1. Configuration ---
PSE_TZ = pytz.timezone('Asia/Manila')
MODEL_SAVE_PATH = "finbert_rep_engine.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NEWS_CSV_PATH = "/kaggle/input/data-csv/headlines.csv"       # CSV containing at least columns: 'date' and 'text'
OHLCV_CSV_PATH = "/kaggle/input/ohlcv-csv/ohlcv.csv"     # CSV containing daily OHLCV data: date, Open, High, Low, Close, Volume

# --- 2. Dataset Definition ---
class FinancialNewsDataset(Dataset):
    """
    Loads news from a CSV (with columns 'date' and 'text'), loads OHLCV from ohlcv.csv,
    generates labels (direction, confidence, volatility) based on the next trading day in OHLCV data,
    and tokenizes text via FinBERT to get embeddings.
    """
    def __init__(self, news_csv_path=NEWS_CSV_PATH, ohlcv_csv_path=OHLCV_CSV_PATH):
        # 2.1 Load news CSV
        self.df = pd.read_csv(news_csv_path)
        if 'date' not in self.df.columns or 'text' not in self.df.columns:
            raise ValueError(f"News CSV must contain 'date' and 'text' columns")
        # Parse dates robustly
        self.df['date'] = pd.to_datetime(self.df['date'], format=None, errors='coerce')
        self.df = self.df.dropna(subset=['date']).copy()
        # Localize/convert to Asia/Manila
        # If naive datetime, localize; if tz-aware, convert
        if self.df['date'].dt.tz is None:
            self.df['date'] = self.df['date'].dt.tz_localize(PSE_TZ)
        else:
            self.df['date'] = self.df['date'].dt.tz_convert(PSE_TZ)
        # Create a date-only column for matching with OHLCV dates
        self.df['date_only'] = self.df['date'].dt.date
        
        # 2.2 Load OHLCV CSV
        self._load_ohlcv(ohlcv_csv_path)
        
        # 2.3 Generate labels: direction, confidence, volatility
        self._generate_labels()
        
        # After label generation, drop any rows without labels
        # Keep only rows where we found a next trading day
        self.df = self.df.dropna(subset=['direction', 'confidence', 'volatility']).reset_index(drop=True)
        
        # 2.4 Load tokenizer and FinBERT model
        self.tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
        self.bert = BertModel.from_pretrained('yiyanghkust/finbert-tone').to(DEVICE)
        self.bert.eval()

    def _load_ohlcv(self, ohlcv_csv_path):
        if not os.path.exists(ohlcv_csv_path):
            raise FileNotFoundError(f"OHLCV CSV not found at {ohlcv_csv_path}")
        df_ohlcv = pd.read_csv(ohlcv_csv_path)
        # Normalize column names to lowercase for flexibility
        df_ohlcv.columns = [c.lower() for c in df_ohlcv.columns]
        # Expect columns: 'date', 'open', 'high', 'low', 'close', 'volume'
        required = {'date', 'open', 'high', 'low', 'close', 'volume'}
        if not required.issubset(set(df_ohlcv.columns)):
            missing = required - set(df_ohlcv.columns)
            raise ValueError(f"OHLCV CSV missing columns: {missing}")
        # Parse date column
        df_ohlcv['date'] = pd.to_datetime(df_ohlcv['date'], format=None, errors='coerce')
        df_ohlcv = df_ohlcv.dropna(subset=['date']).copy()
        # We only care about date part for alignment: convert to date
        df_ohlcv['date_only'] = df_ohlcv['date'].dt.date
        # Sort by date_only ascending
        df_ohlcv = df_ohlcv.sort_values('date_only').reset_index(drop=True)
        # Keep only one row per date_only (if duplicates, take first or average? Here we take first)
        df_ohlcv = df_ohlcv.drop_duplicates(subset=['date_only'], keep='first').copy()
        # Set index for fast lookup
        df_ohlcv = df_ohlcv.set_index('date_only')
        self.market_data = df_ohlcv  # index: python date, columns: open, high, low, close, volume
        # Prepare a sorted list of dates for bisect
        self._market_dates = list(self.market_data.index)  # sorted python dates

        # Pre-compute rolling median volume (over previous 5 trading days including current)
        # Because market_data is sorted by date_only index, we can do rolling on the DataFrame.
        # Note: rolling(window=5) will take previous 4 + current row.
        self.market_data['vol_median_5'] = self.market_data['volume'].rolling(window=5, min_periods=1).median()

    def _generate_labels(self):
        # Prepare arrays to collect labels
        directions = []
        confidences = []
        volatilities = []
        has_label = []
        
        for news_date in self.df['date_only']:
            # Find next trading day strictly after news_date
            # Use bisect on sorted list self._market_dates
            pos = bisect.bisect_right(self._market_dates, news_date)
            if pos >= len(self._market_dates):
                # no next trading day in our OHLCV data
                has_label.append(False)
                directions.append(np.nan)
                confidences.append(np.nan)
                volatilities.append(np.nan)
                continue
            next_date = self._market_dates[pos]
            # Extract OHLCV for next_date
            row = self.market_data.loc[next_date]
            open_p = float(row['open'])
            close_p = float(row['close'])
            high_p = float(row['high'])
            low_p = float(row['low'])
            vol = float(row['volume'])
            vol_med = float(row['vol_median_5']) if not np.isnan(row['vol_median_5']) else np.nan
            
            # 1) Direction: continuous label in [-1,1], e.g., tanh of scaled return
            #    return = (close - open)/open
            ret = (close_p - open_p) / open_p if open_p != 0 else 0.0
            direction = np.tanh(ret * 3.0)
            
            # 2) Confidence: volume relative to 5-day median volume, clipped to [0,1]
            if not np.isnan(vol_med) and vol_med > 0:
                conf = vol / vol_med
                # clip between 0 and 1
                conf = float(np.clip(conf, 0.0, 1.0))
            else:
                conf = 0.0
            
            # 3) Volatility: (High - Low)/Open, clipped to [0,1]
            if open_p != 0:
                volat = (high_p - low_p) / open_p
                volat = float(np.clip(volat, 0.0, 1.0))
            else:
                volat = 0.0
            
            has_label.append(True)
            directions.append(direction)
            confidences.append(conf)
            volatilities.append(volat)
        
        # Assign to dataframe
        self.df['direction'] = directions
        self.df['confidence'] = confidences
        self.df['volatility'] = volatilities
        # Rows without label will have NaNs; drop later

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Tokenize text
        text = self.df.iloc[idx]["text"]
        # Retrieve labels
        label = self.df.iloc[idx][["direction", "confidence", "volatility"]].astype(np.float32).values

        # Tokenize and get FinBERT embeddings
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding="max_length"
        ).to(DEVICE)

        with torch.no_grad():
            outputs = self.bert(**inputs)
            # Mean pooling over sequence
            embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()  # shape [768]
        return embeddings, torch.FloatTensor(label)

    def tokenize_sample(self, text):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding="max_length"
        ).to(DEVICE)
        with torch.no_grad():
            outputs = self.bert(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()
        return embeddings.unsqueeze(0)  # shape [1, 768]


# --- 3. Model Architecture ---
class RepresentationEngine(nn.Module):
    """
    Simple MLP: maps 768-dim FinBERT embedding to 3 outputs: direction, confidence, volatility.
    """
    def __init__(self, input_dim=768, hidden_dim=256):
        super().__init__()
        self.alignment = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.ReLU()
        )
        self.predictor = nn.Linear(input_dim, 3)

    def forward(self, x):
        aligned = self.alignment(x)
        pred = self.predictor(aligned)
        # For direction: we want roughly in [-1,1], so apply tanh on that logit? 
        # For confidence and volatility: they were clipped to [0,1]; 
        # we can apply sigmoid to those outputs to get [0,1], but since predictor outputs 3 values,
        # we might apply: direction via tanh, confidence/volatility via sigmoid.
        # Here, we split:
        direction_raw = pred[:, 0:1]          # shape [batch,1]
        conf_raw = pred[:, 1:2]
        vol_raw = pred[:, 2:3]
        direction = torch.tanh(direction_raw)  # in [-1,1]
        confidence = torch.sigmoid(conf_raw)   # in [0,1]
        volatility = torch.sigmoid(vol_raw)    # in [0,1]
        out = torch.cat([direction, confidence, volatility], dim=1)
        return aligned, out


# --- 4. Training Loop ---
def train_model(news_csv_path=NEWS_CSV_PATH, ohlcv_csv_path=OHLCV_CSV_PATH, epochs=10, batch_size=32, lr=1e-4):
    dataset = FinancialNewsDataset(news_csv_path, ohlcv_csv_path)
    if len(dataset) == 0:
        raise RuntimeError("No data after label generation. Check date alignment between news and OHLCV.")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = RepresentationEngine().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for embeddings, labels in loader:
            embeddings = embeddings.to(DEVICE)  # shape [batch,768]
            labels = labels.to(DEVICE)          # shape [batch,3]
            optimizer.zero_grad()
            _, pred = model(embeddings)
            loss = loss_fn(pred, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * embeddings.size(0)
        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    return model

# --- 5. Execution ---
if __name__ == "__main__":
    print("Starting training...")
    model = train_model()
    print(f"Model saved to {MODEL_SAVE_PATH}")

    # Example inference
    sample_text = "SM Prime announces record profits"
    ds = FinancialNewsDataset(news_csv_path=NEWS_CSV_PATH, ohlcv_csv_path=OHLCV_CSV_PATH)
    sample_embedding = ds.tokenize_sample(sample_text).to(DEVICE)
    model.eval()
    with torch.no_grad():
        _, prediction = model(sample_embedding)  # shape [1,3]
    pred_np = prediction.cpu().numpy().squeeze()
    print(f"Prediction for sample: direction={pred_np[0]:.4f}, confidence={pred_np[1]:.4f}, volatility={pred_np[2]:.4f}")
