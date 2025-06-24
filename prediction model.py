import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertModel, BertTokenizer
from datetime import datetime
import pytz
import bisect
from tqdm.auto import tqdm  # tqdm for progress bars
from sklearn.metrics import accuracy_score

# --- 1. Configuration ---
PSE_TZ = pytz.timezone('Asia/Manila')
MODEL_SAVE_PATH = "finbert_rep_engine_classification.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths (adjust if needed)
NEWS_CSV_PATH = "/kaggle/input/data-csv/headlines.csv"  # CSV containing 'date' and 'text'
OHLCV_CSV_PATH = "/kaggle/input/ohlcv-csv/ohlcv.csv"    # CSV containing daily OHLCV data: date, Open, High, Low, Close, Volume

# Classification thresholds
# If next-day close-to-close return > NEUTRAL_THRESHOLD -> class 2 (up)
# If next-day close-to-close return < -NEUTRAL_THRESHOLD -> class 0 (down)
# Else -> class 1 (neutral)
NEUTRAL_THRESHOLD = 0.005  # e.g., 0.5%; adjust based on distribution of returns

# --- 2. Dataset Definition ---
class FinancialNewsDataset(Dataset):
    """
    Loads news from a CSV (with columns 'date' and 'text'), loads OHLCV from ohlcv.csv,
    generates a 3-class label (down=0, neutral=1, up=2) based on next trading day's
    close-to-close return, and tokenizes text via FinBERT tokenizer,
    returning input_ids & attention_mask and a label int.
    """
    def __init__(self, news_csv_path=NEWS_CSV_PATH, ohlcv_csv_path=OHLCV_CSV_PATH,
                 tokenizer_name='yiyanghkust/finbert-tone', neutral_threshold=NEUTRAL_THRESHOLD):
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

        # 2.3 Generate 3-class labels: down / neutral / up using close-to-close
        self._generate_labels(neutral_threshold)

        # After label generation, drop any rows without labels
        # label column named 'label' with values 0,1,2
        self.df = self.df.dropna(subset=['label']).reset_index(drop=True)
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
        # Set index to date_only for fast lookup
        df_ohlcv = df_ohlcv.set_index('date_only')
        self.market_data = df_ohlcv
        # Sorted list of dates for bisect
        self._market_dates = sorted(self.market_data.index)
        # Optionally: Pre-compute any rolling stats if needed
        # self.market_data['vol_median_5'] = self.market_data['volume'].rolling(window=5, min_periods=1).median()

    def _generate_labels(self, neutral_threshold):
        labels = []
        for news_date in self.df['date_only']:
            # Find insertion point: first date > news_date
            pos = bisect.bisect_right(self._market_dates, news_date)
            # For close-to-close, we need:
            #   close_t = last trading day <= news_date  (index pos-1, if pos>0 and date matches or before)
            #   close_t+1 = first trading day > news_date (index pos)
            if pos == 0 or pos >= len(self._market_dates):
                # No valid prev or next trading day
                labels.append(np.nan)
                continue
            date_t = self._market_dates[pos - 1]
            date_t1 = self._market_dates[pos]
            # Ensure date_t <= news_date < date_t1
            # Fetch closes
            try:
                close_t = float(self.market_data.loc[date_t, 'close'])
                close_t1 = float(self.market_data.loc[date_t1, 'close'])
            except Exception:
                labels.append(np.nan)
                continue
            # Compute close-to-close return
            if close_t != 0:
                ret = (close_t1 - close_t) / close_t
            else:
                ret = 0.0
            # Determine class
            if ret > neutral_threshold:
                label = 2  # up
            elif ret < -neutral_threshold:
                label = 0  # down
            else:
                label = 1  # neutral
            labels.append(label)
        self.df['label'] = labels

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        Return a dict: {'input_ids': ..., 'attention_mask': ...}, and a label int.
        All tensors on CPU; training loop moves them to DEVICE.
        """
        text = self.df.iloc[idx]["text"]
        label = int(self.df.iloc[idx]["label"])
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
        label_tensor = torch.tensor(label, dtype=torch.long)
        return item, label_tensor

# --- 3. Model Architecture including FinBERT for 3-class classification ---
class RepresentationEngine(nn.Module):
    """
    Combines FinBERT embedding (frozen or partially unfrozen) + MLP predicting 3 logits for classes.
    Inputs: tokenized inputs (input_ids, attention_mask).
    Outputs: aligned embedding and logits [batch, 3].
    """
    def __init__(self, bert_model_name='yiyanghkust/finbert-tone',
                 input_dim=768, hidden_dim=256,
                 unfreeze_last_n_layers=0):
        """
        If unfreeze_last_n_layers > 0, unfreezes that many final encoder layers.
        e.g., unfreeze_last_n_layers=2 unfreezes encoder.layer.10 and 11 (for bert-base with 12 layers).
        """
        super().__init__()
        # Load FinBERT model
        self.bert = BertModel.from_pretrained(bert_model_name)
        # Freeze all parameters first
        for param in self.bert.parameters():
            param.requires_grad = False

        # Optionally unfreeze last N layers
        if unfreeze_last_n_layers > 0:
            total_layers = len(self.bert.encoder.layer)
            for i in range(total_layers - unfreeze_last_n_layers, total_layers):
                for param in self.bert.encoder.layer[i].parameters():
                    param.requires_grad = True
            # Also unfreeze pooler if present
            if hasattr(self.bert, 'pooler'):
                for param in self.bert.pooler.parameters():
                    param.requires_grad = True

        # MLP layers on top of pooled embedding
        # We use mean pooling in forward, then pass through alignment MLP
        self.alignment = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.ReLU()
        )
        # Final classifier: 3 classes
        self.classifier = nn.Linear(input_dim, 3)

    def forward(self, input_ids, attention_mask):
        # input_ids, attention_mask: shape [batch, seq_len]
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Mean pooling over sequence dimension
        embeddings = outputs.last_hidden_state.mean(dim=1)  # [batch, hidden_dim]
        # Pass through alignment MLP
        aligned = self.alignment(embeddings)  # [batch, input_dim]
        logits = self.classifier(aligned)     # [batch, 3]
        return aligned, logits

# --- 4. Training Loop with multi-GPU, validation, and tqdm ---
def train_model(news_csv_path=NEWS_CSV_PATH, ohlcv_csv_path=OHLCV_CSV_PATH,
                epochs=2, batch_size=32, lr=1e-4, num_workers=4,
                val_split=0.2, unfreeze_last_n_layers=0):
    # Instantiate dataset
    dataset = FinancialNewsDataset(news_csv_path, ohlcv_csv_path)
    if len(dataset) == 0:
        raise RuntimeError("No data after label generation. Check date alignment between news and OHLCV.")
    total_size = len(dataset)
    # Create train/val split
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size],
                                              generator=torch.Generator().manual_seed(42))
    print(f"Dataset size: {total_size}, train: {train_size}, val: {val_size}")

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    # Instantiate model
    model = RepresentationEngine(unfreeze_last_n_layers=unfreeze_last_n_layers).to(DEVICE)
    # If multiple GPUs, wrap with DataParallel
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)
    else:
        print("Using single GPU or CPU")

    # Prepare optimizer: only train params with requires_grad=True
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(trainable_params, lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        # ---- Training ----
        model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
        for batch in train_iter:
            inputs, labels = batch
            input_ids = inputs['input_ids'].to(DEVICE, non_blocking=True)
            attention_mask = inputs['attention_mask'].to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)  # [batch]

            optimizer.zero_grad()
            _, logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * input_ids.size(0)
            # Track predictions
            preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.detach().cpu().numpy())
            train_iter.set_postfix(loss=loss.item())

        avg_train_loss = total_loss / train_size
        all_preds_arr = np.concatenate(all_preds)
        all_labels_arr = np.concatenate(all_labels)
        train_acc = accuracy_score(all_labels_arr, all_preds_arr)

        # ---- Validation ----
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels_list = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]  ", leave=False):
                inputs, labels = batch
                input_ids = inputs['input_ids'].to(DEVICE, non_blocking=True)
                attention_mask = inputs['attention_mask'].to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)

                _, logits = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(logits, labels)
                val_loss += loss.item() * input_ids.size(0)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                val_preds.append(preds)
                val_labels_list.append(labels.cpu().numpy())

        avg_val_loss = val_loss / val_size if val_size > 0 else 0.0
        val_preds_arr = np.concatenate(val_preds) if val_size>0 else np.array([])
        val_labels_arr = np.concatenate(val_labels_list) if val_size>0 else np.array([])
        val_acc = accuracy_score(val_labels_arr, val_preds_arr) if val_size>0 else 0.0

        print(f"Epoch {epoch+1}/{epochs}: "
              f"Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.4f} | "
              f"Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.4f}")

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
    # You can set unfreeze_last_n_layers=2 to fine-tune last 2 BERT layers if desired,
    # but be cautious with learning rate (might want smaller LR). By default we keep BERT frozen.
    model = train_model(epochs=2, batch_size=32, lr=1e-4, unfreeze_last_n_layers=0)

    # Example inference
    sample_texts = [
        "SM Prime announces record profit",
        "Market shows slight decline amid global concerns",
        "No significant movement expected from the index tomorrow",
        "Inflation at an all-time high"
    ]
    tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
    model.eval()
    for sample_text in sample_texts:
        encoding = tokenizer(
            sample_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding="max_length"
        )
        input_ids = encoding['input_ids'].to(DEVICE)
        attention_mask = encoding['attention_mask'].to(DEVICE)
        with torch.no_grad():
            _, logits = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(logits, dim=1).cpu().numpy().squeeze()  # [3]
            pred_class = int(np.argmax(probs))
        class_map = {0: "Down", 1: "Neutral", 2: "Up"}
        print(f"Text: \"{sample_text}\"  → Pred: {class_map[pred_class]}, probs={probs.tolist()}")

