import time
import pandas as pd
from FinMind.data import DataLoader
import yfinance as yf

# 初始化 FinMind 資料載入器
api = DataLoader()

print("🔍 1. 開始利用 FinMind 獲取最新上市櫃股票清單...")

try:
    df_info = api.taiwan_stock_info()
    
    # 定義我們要的科技業（電子股）子產業名稱
    tech_categories = [
        "半導體業", "電腦及週邊設備業", "光電業", "通信網路業", 
        "電子零組件業", "電子通路業", "資訊服務業", "其他電子業"
    ]
    
    # 篩選科技股
    df_tech = df_info[
        df_info["industry_category"].isin(tech_categories) & 
        df_info["type"].isin(["twse", "tpex"]) & 
        (df_info["stock_id"].str.len() == 4)
    ]
    
    stock_list = {}
    for _, row in df_tech.iterrows():
        sid = row["stock_id"]
        suffix = ".TW" if row["type"] == "twse" else ".TWO"
        stock_list[f"{sid}{suffix}"] = {
            "name": row["stock_name"],
            "is_listed": 1 if row["type"] == "twse" else 0
        }
        
    total_count = len(stock_list)
    print(f"🎯 成功鎖定全台股共 {total_count} 檔上市櫃科技股！")

except Exception as e:
    print(f"❌ 獲取股票清單失敗，錯誤原因: {e}")
    stock_list = {}

if stock_list:
    print("\n🚀 2. 開始批次下載所有科技股的歷史數據...")
    all_data = []
    success_count = 0
    start_date = "2025-01-01"
    
    for i, (symbol, info) in enumerate(stock_list.items(), 1):
        print(f"[{i}/{total_count}] 正在下載 {symbol} ({info['name']})...", end="", flush=True)
        
        try:
            # 關鍵修正 1：加上 group_by='ticker'，讓格式完全統一，不再有亂七八糟的多層結構
            df = yf.download(symbol, start=start_date, progress=False, group_by='ticker')
            
            if df.empty or len(df) < 25:  # 至少要 25 天的資料才能算 20 日均線
                print(" ❌ (資料太少或無資料，跳過)")
                continue
                
            df = df.reset_index()
            
            # 關鍵修正 2：強制將欄位名稱攤平成單層字串（例如把 ('Date', '') 變成 'Date'）
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[1] if col[1] != '' else col[0] for col in df.columns]
            
            # 檢查必備欄位是否存在
            if "Close" not in df.columns:
                print(" ❌ (缺少 Close 欄位)")
                continue

            # 計算均線特徵
            df["MA_5"] = df["Close"].rolling(window=5).mean()
            df["MA_20"] = df["Close"].rolling(window=20).mean()
            
            # 建立核心欄位
            df["stock_id"] = symbol.split(".")[0]
            df["stock_name"] = info["name"]
            df["is_listed"] = info["is_listed"]
            
            # 重新命名欄位以防衝突
            df = df.rename(columns={
                "Open": "Open_Price",
                "High": "High_Price",
                "Low": "Low_Price",
                "Close": "Close_Price",
                "Volume": "Volume_Size"
            })
            
            # 清除因計算均線產生的 NaN
            df = df.dropna(subset=["MA_20"])
            
            # 篩選最終要用的特徵欄位
            df = df[[
                "Date", "stock_id", "stock_name", "is_listed", 
                "Open_Price", "High_Price", "Low_Price", "Close_Price", "Volume_Size", 
                "MA_5", "MA_20"
            ]]
            
            all_data.append(df)
            success_count += 1
            print(" 行 ✅")
            
            # 控制速度防止被鎖 IP
            if i % 30 == 0:
                time.sleep(1.5)
                
        except Exception as e:
            print(f" ❌ (錯誤: {e})")
            
    if all_data:
        final_table = pd.concat(all_data, ignore_index=True)
        print(f"\n📊 全部下載完成！成功收集 {success_count} 檔股票數據。")
        output_filename = "all_tech_stocks_data.csv"
        final_table.to_csv(output_filename, index=False, encoding="utf-8-sig")
        print(f"💾 大數據表格已成功儲存為 '{output_filename}'！")


#測試