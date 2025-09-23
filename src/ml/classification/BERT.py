import os
import json
import numpy as np
from glob import glob
import random
from tqdm import tqdm
import sys
import math
import pandas as pd

from collections import Counter

from sklearn.metrics import accuracy_score

import torch
from torch.utils.data import Dataset

from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments
import transformers

from sklearn.model_selection import train_test_split





'''可修改變數'''
N_SAMPLES = 250_000  # random sampling 取的數量

NUM_CATEGORIES = 5  # 類別數量

EPOCHS = 5

SAVE_PATH = "../data/ml/classification/BERT"

ISTRAIN = False
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



# --- 五元分類 ---
def categorize_array_multi(Y, t1=0.0590, t2=0.0102, t3=0.0060, t4=0.0657):
    # 五元分類
    labels = np.full_like(Y, 2, dtype=int)  # 預設持平
    labels[Y <= -t1] = 0  # 大跌
    labels[(Y > -t1) & (Y <= -t2)] = 1  # 跌
    labels[(Y >= t3) & (Y < t4)] = 3  # 漲
    labels[Y >= t4] = 4  # 大漲

    if np.any(Y == 0):  # 檢查是否有任何元素等於 0
        count = np.sum(Y == 0)
        print(f"共有 {count} 個 Y == 0")
        labels[Y == 0] = 4  # 為了校正 TRUMP 前兩天的價格相同 第一天設為大漲

    return labels



class TweetDataset(Dataset):
    def __init__(self, texts=None, labels=None, tokenizer=None, max_len=64, pre_tokenized=None):
        if pre_tokenized is not None:
            # 已經處理好的 encoding
            self.encodings = pre_tokenized
            self.labels = labels
        else:
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_len = max_len
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
        return len(self.encodings)

    def __getitem__(self, idx):
        item = self.encodings[idx]
        if hasattr(self, "labels") and self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item





def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}



def balanced_sampling(texts, labels, n_samples, num_categories, random_state=42):
    """
    Balanced sampling, with a target total sample count n_samples.
    Attempts to have roughly equal number of samples per class.
    
    texts: list of text samples
    labels: np.array of shape (N,)
    n_samples: total number of samples to draw
    num_categories: number of classes
    """

    # np.random.seed(random_state)
    # labels = np.array(labels)
    # sampled_texts = []
    # sampled_labels = []

    # # 計算每類目標樣本數
    # target_per_class = n_samples // num_categories
    # all_indices = set(range(len(labels)))
    # sampled_indices_set = set()

    # # 每類抽樣
    # for cls in range(num_categories):
    #     cls_indices = np.where(labels == cls)[0]
    #     n_cls_sample = min(len(cls_indices), target_per_class)
    #     selected = np.random.choice(cls_indices, size=n_cls_sample, replace=False)
    #     sampled_indices_set.update(selected)

    # # 剩餘數量
    # remaining = n_samples - len(sampled_indices_set)
    # if remaining > 0:
    #     available_indices = np.array(list(all_indices - sampled_indices_set))
    #     extra_selected = np.random.choice(available_indices, size=min(len(available_indices), remaining), replace=False)
    #     sampled_indices_set.update(extra_selected)

    # # 最終抽樣
    # sampled_indices = np.array(list(sampled_indices_set))
    # sampled_texts = [texts[i] for i in sampled_indices]
    # sampled_labels = labels[sampled_indices]

    # # 類別統計
    # counter_sampled = Counter(sampled_labels)
    # total = len(sampled_labels)
    # print("Balanced sampled class distribution (approx):")
    # for k in range(num_categories):
    #     print(f"  Class {k}: {counter_sampled[k]} samples, {counter_sampled[k]/total*100:.2f}%")
    # print(f"  Total sampled: {total} samples\n")

    # return sampled_texts, sampled_labels


    np.random.seed(random_state)
    labels = np.array(labels)
    sampled_texts = []
    sampled_labels = []

    # 計算每類目標樣本數
    target_per_class = n_samples // num_categories

    # 用索引操作，避免重複
    all_indices = np.arange(len(labels))
    used_indices = set()

    for cls in range(num_categories):
        cls_indices = np.where(labels == cls)[0]
        if len(cls_indices) == 0:
            continue
        n_cls_sample = min(len(cls_indices), target_per_class)
        selected = np.random.choice(cls_indices, size=n_cls_sample, replace=False)
        sampled_texts.extend([texts[i] for i in selected])
        sampled_labels.extend(labels[selected])
        used_indices.update(selected)

    # 剩餘樣本數，從未用過的樣本隨機分配
    remaining = n_samples - len(sampled_labels)
    if remaining > 0:
        available_indices = np.array([i for i in all_indices if i not in used_indices])
        if len(available_indices) > 0:
            extra_selected = np.random.choice(
                available_indices, 
                size=min(len(available_indices), remaining), 
                replace=False
            )
            sampled_texts.extend([texts[i] for i in extra_selected])
            sampled_labels.extend(labels[extra_selected])

    sampled_labels = np.array(sampled_labels)

    # 統計抽樣結果
    counter = Counter(sampled_labels)
    print("Balanced sampled class distribution (approx):")
    total = len(sampled_labels)
    for cls in range(num_categories):
        print(f"  Class {cls}: {counter[cls]} samples, {counter[cls]/total*100:.2f}%")
    print(f"  Total sampled: {total} samples")

    return sampled_texts, sampled_labels




def train_single_model(texts, labels, num_categories, model_dir=None,
                       epochs=3, n_samples=None, balanced=True):
    """
    texts: 訓練用的推文
    labels: 對應標籤
    all_texts_for_pred: 要全部丟去預測的推文 (包含訓練用的)
    """
    labels = np.array(labels)
    sampled_texts = []
    sampled_labels = []

    if n_samples is None:
        n_samples = len(texts)

    if balanced:
        sampled_texts, sampled_labels = balanced_sampling(
            texts, labels, n_samples, num_categories=num_categories
        )
    else:
        n = len(texts)
        sampled_indices = np.random.choice(range(n), size=min(n_samples, n), replace=False)
        sampled_texts = [texts[i] for i in sampled_indices]
        sampled_labels = labels[sampled_indices]

    # if balanced:
    #     # 計算每類目標樣本數
    #     target_per_class = n_samples // num_categories
    #     all_indices = set(range(len(labels)))
    #     sampled_indices_set = set()

    #     # 每類抽樣
    #     for cls in range(num_categories):
    #         cls_indices = np.where(labels == cls)[0]
    #         n_cls_sample = min(len(cls_indices), target_per_class)
    #         selected = np.random.choice(cls_indices, size=n_cls_sample, replace=False)
    #         sampled_indices_set.update(selected)

    #     # 剩餘數量
    #     remaining = n_samples - len(sampled_indices_set)
    #     if remaining > 0:
    #         available_indices = np.array(list(all_indices - sampled_indices_set))
    #         extra_selected = np.random.choice(available_indices, size=min(len(available_indices), remaining), replace=False)
    #         sampled_indices_set.update(extra_selected)

    #     # 最終抽樣
    #     sampled_indices = np.array(list(sampled_indices_set))
    #     sampled_texts = [texts[i] for i in sampled_indices]
    #     sampled_labels = labels[sampled_indices]

    #     # 類別統計
    #     counter_sampled = Counter(sampled_labels)
    #     total = len(sampled_labels)
    #     print("Balanced sampled class distribution (approx):")
    #     for k in range(num_categories):
    #         print(f"  Class {k}: {counter_sampled[k]} samples, {counter_sampled[k]/total*100:.2f}%")
    #     print(f"  Total sampled: {total} samples\n")
    # else:
    #     # 普通隨機抽樣
    #     n = len(texts)
    #     sampled_indices = np.random.choice(range(n), size=min(n_samples, n), replace=False)
    #     sampled_texts = [texts[i] for i in sampled_indices]
    #     sampled_labels = labels[sampled_indices]

    # Tokenizer + Dataset
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    dataset = TweetDataset(sampled_texts, sampled_labels, tokenizer)

    # train/val split
    split = int(0.8 * len(dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [split, len(dataset)-split])

    # 模型
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_categories=num_categories)

    training_args = TrainingArguments(
        output_dir=model_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        load_best_model_at_end=True,
        logging_steps=10,
        report_to="none",
        remove_unused_columns=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    print(f"=== Training {model_dir} ===")
    train_result = trainer.train()

    # 儲存模型
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    # 訓練結果
    metrics = train_result.metrics
    print(f"Training metrics for {model_dir}:")
    print(json.dumps(metrics, indent=4))
    with open(os.path.join(model_dir, "train_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    # 驗證集結果
    eval_metrics = trainer.evaluate()
    print(f"Validation metrics for {model_dir}:")
    print(json.dumps(eval_metrics, indent=4))

    return trainer



def tokenize_and_save(all_texts, tokenizer, max_len=64, save_path=None): 
    if os.path.exists(save_path): 
        print(f"Loading pre-tokenized tweets from {save_path}") 
        return torch.load(save_path) 
    
    encodings = [] 
    print("Tokenizing texts...") 
    for txt in tqdm(all_texts, desc="Tokenizing"): 
        encoding = tokenizer( 
            txt, 
            truncation=True, 
            padding="max_length", 
            max_length=max_len, 
            return_tensors="pt" 
        ) 
        encodings.append({ 
            "input_ids": encoding["input_ids"].flatten(), 
            "attention_mask": encoding["attention_mask"].flatten() 
        }) 
        
    torch.save(encodings, save_path) 
    print(f"Saved tokenized tweets to {save_path}") 
    
    return encodings



def fast_predict_all_models(all_texts, all_Y, tokenized_path=None,
                            save_path=SAVE_PATH, batch_size=512, device=None):
    """
    分批預測所有模型，完成一個 label 就單獨存成一個檔案並釋放記憶體
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 載入 pre-tokenized
    tokenizer = BertTokenizerFast.from_pretrained(f"{save_path}/allcoins_y0")  # 用 model0 的 tokenizer
    encodings = tokenize_and_save(all_texts, tokenizer, max_len=64, save_path=tokenized_path)
    input_ids = torch.stack([e["input_ids"] for e in encodings]).to(device)
    attention_mask = torch.stack([e["attention_mask"] for e in encodings]).to(device)
    n_samples = input_ids.size(0)
    n_labels = all_Y.shape[1]

    os.makedirs(save_path, exist_ok=True)

    for i in range(n_labels):
        model_dir = f"{save_path}/allcoins_y{i}"
        print(f"=== Predicting Y[:, {i}] with model {model_dir} ===")

        model = BertForSequenceClassification.from_pretrained(model_dir).to(device)
        model.eval()

        preds_i = []

        with torch.no_grad():
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch_input_ids = input_ids[start:end]
                batch_attention_mask = attention_mask[start:end]

                outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
                logits = outputs.logits
                batch_pred = torch.argmax(logits, dim=-1).cpu().numpy()
                preds_i.append(batch_pred)

                if (start // batch_size) % 50 == 0:
                    print(f"  Processed {end}/{n_samples} samples")

        preds_i = np.concatenate(preds_i)

        # 建立 DataFrame，只包含當前 label
        df = pd.DataFrame({
            "text": all_texts,
            f"true_y{i}": all_Y[:, i],
            f"pred_y{i}": preds_i,
        })
        df[f"correct_y{i}"] = df[f"true_y{i}"] == df[f"pred_y{i}"]

        # 存檔（每個 label 獨立檔案）
        csv_path = os.path.join(save_path, f"predictions_y{i}.csv")
        json_path = os.path.join(save_path, f"predictions_y{i}.json")
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        df.to_json(json_path, orient="records", force_ascii=False, indent=4)

        print(f"Saved predictions for label {i} to {csv_path} and {json_path}")

        # 清理記憶體
        del model
        torch.cuda.empty_cache()

    print("✅ 全部 label 預測完成！")







def main():
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
        Y = load_price_diff(price_dir, coin_short_name)  # (N_coin, 1)

        assert len(texts) == Y.shape[0], f"{coin_short_name} texts and Y length mismatch!"

        all_texts.extend(texts)
        all_Y.append(Y)

    all_Y = np.vstack(all_Y)  # shape = (N_total, 1)

    if ISTRAIN:
        print(f"=== Processing Y (all coins combined) ===")
        labels = categorize_array_multi(all_Y)
        model_dir = f"{SAVE_PATH}/allcoins_y"

        # 訓練 + 預測全部推文
        trainer = train_single_model(
            all_texts,
            labels,
            num_categories=NUM_CATEGORIES,
            model_dir=model_dir,
            epochs=EPOCHS,
            n_samples=N_SAMPLES,
            balanced=True
        )
        
    print("\n開始預測全部推文...")
    # 預測全部推文 + 輸出 CSV/JSON
    fast_predict_all_models(all_texts, all_Y, tokenized_path=f"{SAVE_PATH}/tokenized_tweets.pt")





if __name__ == "__main__":
    main()