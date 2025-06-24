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
from tqdm.auto import tqdm  # tqdm for progress bars

# --- 1. Configuration ---
PSE_TZ = pytz.timezone('Asia/Manila')
MODEL_SAVE_PATH = "finbert_rep_engine.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NEWS_CSV_PATH = "/kaggle/input/data-csv/headlines.csv"  # CSV containing 'date' and 'text'
OHLCV_CSV_PATH = "/kaggle/input/ohlcv-csv/ohlcv.csv"    # CSV containing daily OHLCV data: date, Open, High, Low, Close, Volume

# --- 2. Dataset Definition ---
class FinancialNewsDataset(Dataset):
    """
    Loads news from a CSV (with columns 'date' and 'text'), loads OHLCV from ohlcv.csv,
    generates labels (direction, confidence, volatility) based on the next trading day in OHLCV data,
    and tokenizes text via FinBERT tokenizer, returning input_ids & attention_mask.
    """
    def __init__(self, news_csv_path=NEWS_CSV_PATH, ohlcv_csv_path=OHLCV_CSV_PATH, tokenizer_name='yiyanghkust/finbert-tone'):
        # 2.1 Load news CSV
        self.df = pd.read_csv(news_csv_path)
        if 'date' not in self.df.columns or 'text' not in self.df.columns:
            raise ValueError(f"News CSV must contain 'date' and 'text' columns")
        # Parse dates robustly
        self.df['date'] = pd.to_datetime(self.df['date'], format=None, errors='coerce')
        self.df = self.df.dropna(subset=['date']).copy()
        # Localize/convert to Asia/Manila
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
        self.df = self.df.dropna(subset=['direction', 'confidence', 'volatility']).reset_index(drop=True)
        if len(self.df) == 0:
            # Let caller handle zero-length dataset
            return

        # 2.4 Initialize tokenizer (BERT model will be in the main model)
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)

    def _load_ohlcv(self, ohlcv_csv_path):
        if not os.path.exists(ohlcv_csv_path):
            raise FileNotFoundError(f"OHLCV CSV not found at {ohlcv_csv_path}")
        df_ohlcv = pd.read_csv(ohlcv_csv_path)
        df_ohlcv.columns = [c.lower() for c in df_ohlcv.columns]
        required = {'date', 'open', 'high', 'low', 'close', 'volume'}
        if not required.issubset(set(df_ohlcv.columns)):
            missing = required - set(df_ohlcv.columns)
            raise ValueError(f"OHLCV CSV missing columns: {missing}")
        df_ohlcv['date'] = pd.to_datetime(df_ohlcv['date'], format=None, errors='coerce')
        df_ohlcv = df_ohlcv.dropna(subset=['date']).copy()
        df_ohlcv['date_only'] = df_ohlcv['date'].dt.date
        df_ohlcv = df_ohlcv.sort_values('date_only').reset_index(drop=True)
        df_ohlcv = df_ohlcv.drop_duplicates(subset=['date_only'], keep='first').copy()
        df_ohlcv = df_ohlcv.set_index('date_only')
        self.market_data = df_ohlcv
        # Sorted list of dates for bisect
        self._market_dates = list(self.market_data.index)
        # Pre-compute rolling median volume
        self.market_data['vol_median_5'] = self.market_data['volume'].rolling(window=5, min_periods=1).median()

    def _generate_labels(self):
        directions = []
        confidences = []
        volatilities = []

        for news_date in self.df['date_only']:
            pos = bisect.bisect_right(self._market_dates, news_date)
            if pos >= len(self._market_dates):
                # no next trading day
                directions.append(np.nan)
                confidences.append(np.nan)
                volatilities.append(np.nan)
                continue
            next_date = self._market_dates[pos]
            row = self.market_data.loc[next_date]
            open_p = float(row['open'])
            close_p = float(row['close'])
            high_p = float(row['high'])
            low_p = float(row['low'])
            vol = float(row['volume'])
            vol_med = float(row['vol_median_5']) if not np.isnan(row['vol_median_5']) else np.nan

            # Direction: tanh-scaled return
            ret = (close_p - open_p) / open_p if open_p != 0 else 0.0
            direction = np.tanh(ret * 3.0)

            # Confidence: volume relative to 5-day median, clipped [0,1]
            if not np.isnan(vol_med) and vol_med > 0:
                conf = vol / vol_med
                conf = float(np.clip(conf, 0.0, 1.0))
            else:
                conf = 0.0

            # Volatility: (High - Low)/Open, clipped [0,1]
            if open_p != 0:
                volat = (high_p - low_p) / open_p
                volat = float(np.clip(volat, 0.0, 1.0))
            else:
                volat = 0.0

            directions.append(direction)
            confidences.append(conf)
            volatilities.append(volat)

        self.df['direction'] = directions
        self.df['confidence'] = confidences
        self.df['volatility'] = volatilities

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        Return a dict: {'input_ids': ..., 'attention_mask': ...}, and a label tensor [3].
        All tensors on CPU; training loop moves them to DEVICE.
        """
        text = self.df.iloc[idx]["text"]
        label = self.df.iloc[idx][["direction", "confidence", "volatility"]].astype(np.float32).values
        # Tokenize (returns CPU tensors)
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding="max_length"
        )
        # encoding is a dict with keys 'input_ids', 'attention_mask'
        # Squeeze the batch dimension:
        item = {
            'input_ids': encoding['input_ids'].squeeze(0),         # shape [seq_len]
            'attention_mask': encoding['attention_mask'].squeeze(0) # shape [seq_len]
        }
        label_tensor = torch.FloatTensor(label)  # shape [3]
        return item, label_tensor

# --- 3. Model Architecture including FinBERT ---
class RepresentationEngine(nn.Module):
    """
    Combines FinBERT embedding (frozen) + MLP predicting 3 outputs.
    Inputs: tokenized inputs (input_ids, attention_mask).
    Outputs: aligned embedding and predictions [direction, confidence, volatility].
    """
    def __init__(self, bert_model_name='yiyanghkust/finbert-tone', input_dim=768, hidden_dim=256):
        super().__init__()
        # Load FinBERT model
        self.bert = BertModel.from_pretrained(bert_model_name)
        # Freeze BERT parameters
        for param in self.bert.parameters():
            param.requires_grad = False
        self.bert.eval()  # we'll run it in eval mode; forward in with torch.no_grad()

        # MLP layers
        self.alignment = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.ReLU()
        )
        self.predictor = nn.Linear(input_dim, 3)

    def forward(self, input_ids, attention_mask):
        # input_ids, attention_mask: shape [batch, seq_len] on some GPU
        # Get BERT embeddings without gradient
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            # Mean pooling over sequence dimension
            # outputs.last_hidden_state: [batch, seq_len, hidden_dim]
            embeddings = outputs.last_hidden_state.mean(dim=1)  # [batch, hidden_dim]
        # Pass through alignment MLP
        aligned = self.alignment(embeddings)  # [batch, hidden_dim->input_dim]
        pred = self.predictor(aligned)        # [batch, 3]
        # Split outputs
        direction_raw = pred[:, 0:1]
        conf_raw = pred[:, 1:2]
        vol_raw = pred[:, 2:3]
        direction = torch.tanh(direction_raw)      # in [-1,1]
        confidence = torch.sigmoid(conf_raw)       # in [0,1]
        volatility = torch.sigmoid(vol_raw)        # in [0,1]
        out = torch.cat([direction, confidence, volatility], dim=1)  # [batch, 3]
        return aligned, out

# --- 4. Training Loop with multi-GPU and tqdm ---
def train_model(news_csv_path=NEWS_CSV_PATH, ohlcv_csv_path=OHLCV_CSV_PATH,
                epochs=10, batch_size=32, lr=1e-4, num_workers=4):
    # Instantiate dataset
    dataset = FinancialNewsDataset(news_csv_path, ohlcv_csv_path)
    if len(dataset) == 0:
        raise RuntimeError("No data after label generation. Check date alignment between news and OHLCV.")
    # DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, pin_memory=True)

    # Instantiate model
    model = RepresentationEngine().to(DEVICE)
    # If multiple GPUs, wrap with DataParallel
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)
    else:
        print("Using single GPU or CPU")

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        model.train()  # MLP layers in train mode; BERT remains in eval mode inside forward
        total_loss = 0.0
        # Epoch-level tqdm
        epoch_iter = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for batch in epoch_iter:
            # batch is tuple: (item_dict, label_tensor)
            inputs, labels = batch
            # Move inputs to DEVICE
            input_ids = inputs['input_ids'].to(DEVICE, non_blocking=True)         # [batch, seq_len]
            attention_mask = inputs['attention_mask'].to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)                         # [batch, 3]

            optimizer.zero_grad()
            _, pred = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(pred, labels)
            loss.backward()
            optimizer.step()

            batch_loss = loss.item() * input_ids.size(0)
            total_loss += batch_loss
            # Optionally, update tqdm with batch loss
            epoch_iter.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.6f}")

    # Save model (if DataParallel, save module.state_dict())
    if isinstance(model, nn.DataParallel):
        torch.save(model.module.state_dict(), MODEL_SAVE_PATH)
    else:
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")
    return model

# --- 5. Execution ---
if __name__ == "__main__":
    print("Starting training...")
    model = train_model()
    # Example inference
    sample_text = "SM Prime announces record profits"
    # We need a small helper to tokenize and run inference
    tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
    device = DEVICE
    # Put model in eval
    model.eval()
    # Tokenize sample
    encoding = tokenizer(
        sample_text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding="max_length"
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    with torch.no_grad():
        # If DataParallel, wrap inputs similarly
        _, prediction = model(input_ids=input_ids, attention_mask=attention_mask)
    pred_np = prediction.cpu().numpy().squeeze()
    print(f"Prediction for sample: direction={pred_np[0]:.4f}, confidence={pred_np[1]:.4f}, volatility={pred_np[2]:.4f}")
