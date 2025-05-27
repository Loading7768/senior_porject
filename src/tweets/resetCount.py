# 如果 tweet_count 不是從 1 開始的話可以使用
import json
import argparse


'''可修改參數'''
JSON_DICT_NAME = "dogecoin"

# filename = '../Kmeans/data/combined/DOGE_2025_3.json'  # 修改成指定檔案
'''可修改參數'''

'''
如果要執行 resetCount.py 需在終端機上打
python resetCount.py --filename "(要修改的檔案路徑)"

如果要直接在程式打上要修改檔案路徑
把 17 - 23 註解 且 filename 取消註解
'''


# 設置命令列參數解析
parser = argparse.ArgumentParser(description="Reset tweet_count in a JSON file")
parser.add_argument('--filename', type=str, required=True, help="Path to the JSON file")
args = parser.parse_args()

# 從參數獲取檔案路徑
filename = args.filename


with open(filename, 'r', encoding='utf-8-sig') as file:
    data_json = json.load(file)

count = 0
last_text = data_json[JSON_DICT_NAME][-1]['text']
while True:
    number = data_json[JSON_DICT_NAME][count]['tweet_count']
    data_json[JSON_DICT_NAME][count]['tweet_count'] = count + 1
    
    if data_json[JSON_DICT_NAME][count]['text'] == last_text:
        break

    count += 1

with open(filename, 'w', encoding='utf-8-sig') as file:
    json.dump(data_json, file, indent=4, ensure_ascii=False)

print(f"成功重置 {filename} 的 tweet_count 總共有 {count + 1} 則推文")