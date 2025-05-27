import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import re  # re 模組來進行文字的模式比對與取代（比 str.replace() 更強大）
import os
from tqdm import tqdm



'''可修改參數'''
FOLDER_PATH = "../temp_data"  # 選擇要對哪個資料夾執行

OUTPUT_FOLDER_NAME = "test"  # 設定要儲存到的資料夾名稱   ex. "../LCS/analysis/{OUTPUT_FOLDER_NAME}/"

JSON_DICT_NAME = "dogecoin"  # 設定推文所存的 json 檔中字典的名稱
'''可修改參數'''




# 載入模型
model_name = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# 定義預處理函數（移除@user與http連結）
def preprocess(text):
    # 移除提及（@username）  
    # \w+：代表一個以上的「英數底線組成的字元」，也就是帳號本體    re.sub(…)：表示將符合這個模式的部分通通移除
    text = re.sub(r'@\w+', '', text)

    # 移除網址（http 開頭）
    # \S+：一串不是空白的字元
    text = re.sub(r'http\S+', '', text)

    # 移除多餘空白
    text = text.strip()
    return text


if __name__ == "__main__":
    all_files = [f for f in os.listdir(FOLDER_PATH) if f.endswith(".json")]
    print(f"📂 總共找到 {len(all_files)} 個檔案要處理")

    for file in all_files:
        filepath = os.path.join(FOLDER_PATH, file)
        filename = os.path.basename(filepath)  # ex: DOGE_20210428.json
        analysis_name = os.path.splitext(filename)[0]  # ex: DOGE_20210428

        # 設定 txtname, json_output_path 的名稱
        txtname = f"./analysis/{OUTPUT_FOLDER_NAME}/{analysis_name}.txt"
        json_output_path = f"./analysis/{OUTPUT_FOLDER_NAME}/{analysis_name}.json"

        # 確認是否有輸出時需使用的資料夾
        output_folder_path = f"./analysis/{OUTPUT_FOLDER_NAME}/"
        os.makedirs(output_folder_path, exist_ok=True)

        # 讀入 json 檔
        with open(filepath, 'r', encoding="utf-8-sig") as file:
            data_json = json.load(file)

        # 推文資料
        tweets = data_json[JSON_DICT_NAME]
        print(f"\n📄 正在處理檔案：{filename}，共 {len(tweets)} 筆推文")

        # 先把 txt 檔裡清空
        with open(txtname, 'w', encoding="utf-8-sig") as filetxt:
            filetxt.write("")

        
        # 先把輸出的 json 檔裡清空
        json_output = {JSON_DICT_NAME: []}
        with open(json_output_path, 'w', encoding='utf-8-sig') as f_json:
            json.dump(json_output, f_json, indent=4, ensure_ascii=False)

        # 分析每條推文
        labels = ['negative', 'neutral', 'positive']

        for tweet in tqdm(tweets, desc=f"🔍 處理 {filename}"):
            # 先清除 @mention, http
            text = preprocess(tweet["text"])

            # 把文字轉成模型需要的格式（token ids、attention mask），並指定回傳 PyTorch tensor（'pt'）
            encoded_input = tokenizer(text, return_tensors='pt')

            # 得到輸出結果（logits），這是一個張量（tensor），代表每個情緒類別的「未經 softmax」分數
            output = model(**encoded_input)

            # 把模型輸出的 logits 轉成機率
            # output.logits：取出原始的分數（tensor）
            # .detach()：切斷跟 PyTorch 訓練圖的關聯，這是因為你只要推論
            # .numpy()：把 tensor 轉成 NumPy 陣列
            # [0]：因為你只輸入一筆文字，batch size = 1，所以取出第 0 筆
            # softmax(...)：將三個類別分數轉成機率（加總 = 1）
            scores = softmax(output.logits.detach().numpy()[0])

            # argmax() 會找出機率最大值的索引
            sentiment = labels[scores.argmax()]

            with open(txtname, 'a', encoding='utf-8-sig') as txtfile:
                txtfile.write(f"Tweet {tweet['tweet_count']}\n"
                              f"Username: {tweet['username']}\n"
                              f"Text: [{repr(text)[1:-1]}]\n"
                              f"Sentiment: {sentiment}\n"
                              f"Scores: {dict(zip(labels, [round(float(s), 4) for s in scores]))}\n\n")

            # 把情緒分析加進 json 檔中
            tweet["sentiment"] = sentiment

        
        with open(json_output_path, 'w', encoding='utf-8-sig') as f_json:
            json.dump(data_json, f_json, indent=4, ensure_ascii=False)
