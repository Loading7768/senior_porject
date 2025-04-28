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

    BERT_best_cluster.py：
        1. 設定好 "可修改參數"
        2. 用合在一起後的 json 檔來判斷分多少群比較適合
        3. 可把執行完成後的 plt.figure 儲存起來

    BERT.py：
        1. 設定好 "可修改參數"
        2. 執行完成可在 Keams/data/culstered 裡看到分好群的 json 檔