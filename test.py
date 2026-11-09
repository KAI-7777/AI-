import pandas as pd

# 讀取剛剛下載的大表格
df = pd.read_csv("all_tech_stocks_data.csv")

# 篩選出台積電 (2330) 在 2026 年 11 月 6 日（最近一個交易日）的資料
tsmc_check = df[(df['stock_id'] == 2330) & (df['Date'] == '2026-11-03')]

print(tsmc_check[['Date', 'stock_name', 'Close_Price', 'Volume_Size']])
