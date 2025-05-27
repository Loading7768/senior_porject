import json
from pathlib import Path
import subprocess



'''可修改參數'''
FOLDER_PATH = "../project_vscode/data/DOGE/2024/11"  # 設定要看的 json 檔資料夾

ALL_JSON_DATA = "./data/combined/DOGE_2024_11.json"  # 設定要把所有指定推文儲存到的地方

JSON_DICT_NAME = "dogecoin"  # 設定推文所存的 json 檔中字典的名稱

COIN_SHORT_NAME = "DOGE"  # 檔案名中的幣種名稱
'''可修改參數'''



# 指定資料夾路徑
folder_path = Path(FOLDER_PATH)  # 替換為你的資料夾路徑

# 用來儲存所有 JSON 內容的列表
all_json_data = {JSON_DICT_NAME:[]}

# 查找所有 .json 檔案
for json_file in folder_path.glob("*.json"):
    if not json_file.name.startswith(COIN_SHORT_NAME):
        continue  # 不是 COIN_SHORT_NAME 開頭就跳過

    with json_file.open('r', encoding='utf-8-sig') as file:
        data = json.load(file)  # 讀取 JSON 內容
        all_json_data[JSON_DICT_NAME].extend(data[JSON_DICT_NAME])  # 存儲到列表

print(f"讀取了 {len(all_json_data[JSON_DICT_NAME])} 個 JSON 檔案")

# 寫入結果到 combined 資料夾中 
with open(ALL_JSON_DATA, 'w', encoding='utf-8-sig') as file:
    json.dump(all_json_data, file, ensure_ascii=False, indent=4)


# 執行另一個 Python 檔案 -> resetCount.py 用來重新對 tweet_count 編號
script_path = Path("../project_vscode/resetCount.py").resolve()

try:
    # 執行 resetCount.py 並傳遞 --filename 參數
    result = subprocess.run(
        # 等同於在終端機運行 python resetCount.py --filename ALL_JSON_DATA
        ["python", str(script_path), "--filename", f"../Kmeans{ALL_JSON_DATA.lstrip(".")}"],
        capture_output=True,
        text=True,
        check=True
    )
    if result.stdout != "":
        print("標準輸出：", result.stdout)
    if result.stderr != "":
        print("標準錯誤：", result.stderr)

except subprocess.CalledProcessError as e:
    print(f"運行 {script_path} 時發生錯誤：{e}")
except FileNotFoundError:
    print(f"找不到檔案：{script_path}")

print(f"✅ 已整合指定資料夾中的所有 json 檔至 {ALL_JSON_DATA}")