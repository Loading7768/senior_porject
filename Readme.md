使用手冊：

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
        3. 在 main.py 裡的可修改參數中設定好需要的參數
        4. 開始執行

    執行後：
        1. 可在 data{編號}.json 裡看到抓到的推文
        2. 可在 analysis.txt 裡看到執行的基本資料與重要 timestamp
        3. 更改為適當的檔名後 (ex. DOGE_20250307)，移動副本至適當資料夾中
        4. 將執行後的 data{編號}.json 刪除 (不需留任何 json 格式)
        5. 若 data{編號}.json 的 tweet_count 沒有從 1 開始，可用 resetCount.py 來修正
        6. 可繼續執行