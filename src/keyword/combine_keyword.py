#!/usr/bin/env python3
import os
import json
import glob

def load_keywords(path):
    """讀取單一 JSON 檔案，回傳 set 格式的 keywords"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        if isinstance(data, list):
            return set(data)
        else:
            raise ValueError(f"{path} 格式錯誤，應該是 list")

def combine_keywords(input_dir, output_path):
    """把資料夾內所有 *_keywords.json 合併"""
    all_keywords = set()

    # 找所有 keyword 檔案
    json_files = glob.glob(os.path.join(input_dir, "*_keywords.json"))
    print(f"找到 {len(json_files)} 個檔案可處理")

    for jf in json_files:
        try:
            kws = load_keywords(jf)
            all_keywords |= kws  # 合併成一個 set
            print(f"{jf} ✅ 讀入 {len(kws)} 個 keywords")
        except Exception as e:
            print(f"{jf} ❌ 發生錯誤: {e}")

    # 排序後存成新的 JSON
    merged = sorted(list(all_keywords))
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"合併完成，總共 {len(merged)} 個 keywords，已輸出到 {output_path}")

if __name__ == "__main__":
    # 你可以改這裡的路徑
    INPUT_DIR = "../data/keyword/machine_learning"         # 放 keyword 檔的資料夾
    OUTPUT_PATH = "../data/keyword/machine_learning/all_keywords.json"

    combine_keywords(INPUT_DIR, OUTPUT_PATH)