使用手冊(抓推文)：

    執行前：
        1. 將終端機的檔案變更為 senior_project\project_vscode>
        2. 更改 X 帳號步驟
            (1)使用瀏覽器開啟 X（Twitter）
                確保你已經登入 X（Twitter）。
            (2)取得 Cookies
                Google Chrome / Edge
                打開 X（Twitter）網站
                按 F12 或 Ctrl + Shift + I 進入 開發者工具
                前往 Application → Storage → Cookies
                找到 https://twitter.com
                取得 auth_token 和其他相關 Cookies
            (3)儲存 Cookies 到 cookies.json
                將 Cookies 存入 cookies.json 檔案
                {
                    "auth_token": "你的 auth_token 值",
                    "ct0": "你的 ct0 值"
                }
        3. 在 main.py 裡的 "可修改參數" 中設定好需要的參數
        4. 開始執行

    執行後：
        1. 可在 {名稱}_{日期}.json 裡看到抓到的推文，並移動副本至適當資料夾中
        2. 可在 analysis.txt 裡看到執行的基本資料與重要 timestamp
        3. 若執行過程中應出現錯誤而中止 可去 analysis_temp.txt 裡查看目前的執行分析
        5. 若 json 的 tweet_count 沒有從 1 開始，可用 resetCount.py 來修正
        6. 可繼續執行

使用手冊(Kmeans)：

    先將終端機的檔案變更為 senior_project\Kmeans>

    combined_data.py：
        1. 設定好 "可修改參數"
        2. 把一整個資料夾中的 json 檔合在一起
        3. 執行完成可在 Keams/data/combined 裡看到合併一起的 json 檔

    BERT_best_cluster.py：(不一定要執行)
        1. 設定好 "可修改參數"
        2. 用合在一起後的 json 檔來判斷分多少群比較適合
        3. 可把執行完成後的 plt.figure 儲存起來

    BERT.py：
        1. 設定好 "可修改參數"
        2. 執行完成可在 Keams/data/culstered 裡看到分好群的 json 檔

使用手冊(LCS)：(已放棄使用)

    先將終端機的檔案變更為 senior_project\LCS>

    tweet_LCS.py：
        1. 需先至 Kmeans 裡將要比對的檔案分群 (若檔案數很小可不用 ex. 1000 筆推文內)
        2. 設定好 "可修改參數"
        3. 執行完成可在 LCS/data/analysis 裡看到符合比對標準的推文 txt 檔

使用手冊(Dice)：

    先將終端機的檔案變更為 senior_project\Dice>

    pip install nltk

    dice.py：
        1. 設定好 "可修改參數"
        2. 如果是第一次執行 需要載
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('punkt_tab')
        3. 執行完成可在 Dice/analysis 裡看到符合比對標準的推文 txt, json 檔
        4. 在 Dice/robot_account 裡有可能是機器人的帳號

使用手冊(Sentiment)：

    先將終端機的檔案變更為 senior_project\Sentiment>

    pip install transformers scipy tqdm

    sentiment_analysis.py：
        1. 設定好 "可修改參數"
        2. 執行完成可在 Sentiment/analysis 裡看到新增加 "sentiment": sentiment 的推文 json 檔，以及詳細分類評分在 txt 檔中 
