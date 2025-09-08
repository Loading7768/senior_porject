import os
import json
import numpy as np
from glob import glob
import random
from tqdm import tqdm
import sys

from sklearn.metrics import accuracy_score

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

from transformers import BertForSequenceClassification, Trainer, TrainingArguments
import transformers





'''可修改變數'''
SAMPLE_RATIO = 0.00001  # random sampling 取的比例 (0.1 -> 10%)

NUM_LABELS = 5  # 標籤數量

EPOCHS = 3

SAVE_PATH = "../data/ml/classification/BERT"
'''可修改變數'''

os.makedirs(SAVE_PATH, exist_ok=True)



def load_tweets(data_dir, coin_short_name, json_dict_name):
    """讀取某幣種所有推文，回傳 [list of texts]"""
    files = glob(os.path.join(data_dir, coin_short_name, "*", "*", f"{coin_short_name}_*_normal.json"))
    texts = []
    for f in tqdm(files, desc=f"Loading tweets for {coin_short_name}"):
        with open(f, "r", encoding="utf-8-sig") as fp:
            data = json.load(fp)
            for tw in data[json_dict_name]:
                texts.append(tw["text"])
    return texts



def load_price_diff(price_path, coin_short_name):
    """讀取某幣種的價差 (N, 5)"""
    return np.load(os.path.join(price_path, f"{coin_short_name}_price_diff.npy"))



def categorize_array_multi(Y, t1=0.1, t2=0.00125):
    # Y shape = (N, 5)
    labels = np.full_like(Y, 2, dtype=int)  # 預設持平
    labels[Y >= t1] = 4
    labels[(Y >= t2) & (Y < t1)] = 3
    labels[(Y > -t1) & (Y <= -t2)] = 1
    labels[Y <= -t1] = 0
    return labels



class TweetDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

        # 事先將文本 tokenized，避免 __getitem__ 每次重複計算
        self.encodings = []
        print("Tokenizing texts...")
        for txt in tqdm(self.texts, desc="Tokenizing"):
            encoding = self.tokenizer(
                txt,
                truncation=True,
                padding="max_length",
                max_length=self.max_len,
                return_tensors="pt"
            )
            self.encodings.append({
                "input_ids": encoding["input_ids"].flatten(),
                "attention_mask": encoding["attention_mask"].flatten()
            })

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        item = self.encodings[idx]
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item




def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}



def train_single_model(texts, labels, num_labels=5, model_dir="./models",
                       epochs=3, sample_ratio=0.5):
    """
    texts: list of 推文文字
    labels: np.array, shape=(N,)
    sample_ratio: 每次訓練隨機抽樣比例 (0~1)
    """
    # 隨機抽樣
    n = len(texts)
    sample_size = int(n * sample_ratio)
    sampled_indices = random.sample(range(n), sample_size)
    sampled_texts = [texts[i] for i in sampled_indices]
    sampled_labels = labels[sampled_indices]

    # Tokenizer + Dataset
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    dataset = TweetDataset(sampled_texts, sampled_labels, tokenizer)

    # train/val split
    split = int(0.8 * len(dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [split, len(dataset)-split])

    # 模型
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)

    training_args = TrainingArguments(
        output_dir=model_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        load_best_model_at_end=True,
        logging_steps=10,          # 每10 step印一次loss
        report_to="none",          # 避免冗餘輸出
        remove_unused_columns=False # 避免 Trainer 警告
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    print(f"=== Training {model_dir} ===")
    train_result = trainer.train()  # Trainer 會自動顯示 batch/epoch 進度條

    # 儲存模型
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    # 輸出訓練結果
    metrics = train_result.metrics
    print(f"Training metrics for {model_dir}:")
    print(json.dumps(metrics, indent=4))

    # 存成 json
    with open(os.path.join(model_dir, "train_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    # 驗證集 accuracy
    eval_metrics = trainer.evaluate()
    print(f"Validation metrics for {model_dir}:")
    print(json.dumps(eval_metrics, indent=4))
    with open(os.path.join(model_dir, "eval_metrics.json"), "w") as f:
        json.dump(eval_metrics, f, indent=4)

    return trainer




def main():
    # print("Transformers version:", transformers.__version__)
    # print("Python executable:", sys.executable)
    # input("Pause...")

    data_dir = "../data/filtered_tweets"
    price_dir = "../data/ml/dataset/coin_price"

    COIN_SHORT_NAME = ["DOGE", "PEPE", "TRUMP"]
    JSON_DICT_NAME = ["dogecoin", "PEPE", "(officialtrump OR \"official trump\" OR \"trump meme coin\" OR \"trump coin\" OR trumpcoin OR $TRUMP OR \"dollar trump\")"]

    all_texts = []
    all_Y = []

    # 先把三種幣的資料合併
    for coin_short_name, json_dict_name in zip(COIN_SHORT_NAME, JSON_DICT_NAME):
        print(f"=== Loading data for {coin_short_name} ===")
        texts = load_tweets(data_dir, coin_short_name, json_dict_name)
        Y = load_price_diff(price_dir, coin_short_name)  # (N_coin, 5)

        # 確認長度一致
        assert len(texts) == Y.shape[0], f"{coin_short_name} texts and Y length mismatch!"

        all_texts.extend(texts)
        all_Y.append(Y)

    # 合併所有幣種的 Y
    all_Y = np.vstack(all_Y)  # shape = (N_total, 5)

    # 對每一組 Y 訓練一個模型
    for i in range(5):
        print(f"=== Training model for Y[:, {i}] (all coins combined) ===")
        labels_i = categorize_array_multi(all_Y)[:, i]

        trainer = train_single_model(
            all_texts,
            labels_i,
            num_labels = NUM_LABELS,
            model_dir=f"{SAVE_PATH}/allcoins_y{i}",
            epochs = EPOCHS,
            sample_ratio=SAMPLE_RATIO  # 隨機抽 50% 訓練
        )




if __name__ == "__main__":
    main()