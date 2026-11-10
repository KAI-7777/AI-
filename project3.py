import pandas as pd
import numpy as np
import yfinance as yf
import xgboost as xgb

print("🔍 1. 載入最新科技股與大盤數據...")
df = pd.read_csv("all_tech_stocks_data.csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by=['stock_id', 'Date']).reset_index(drop=True)

# 下載最新 0050 數據
tw50 = yf.download("0050.TW", start="2025-01-01", progress=False, auto_adjust=True, group_by='ticker').reset_index()

# 【完美校正】還原你最原始精準的 MultiIndex 攤平邏輯
if isinstance(tw50.columns, pd.MultiIndex):
    tw50.columns = [col[1] if col[1] != '' else col[0] for col in tw50.columns]

tw50['Date'] = pd.to_datetime(tw50['Date'])
tw50['大盤_20日均線'] = tw50['Close'].rolling(20).mean()
tw50['大盤_20日乖離率'] = (tw50['Close'] - tw50['大盤_20日均線']) / tw50['大盤_20日均線']
tw50['大盤_5日漲跌幅'] = tw50['Close'].pct_change(5)

df = pd.merge(df, tw50[['Date', '大盤_20日乖離率', '大盤_5日漲跌幅']], on='Date', how='left')

# 計算特徵
g = df.groupby('stock_id')
df['5日均量'] = g['Volume_Size'].transform(lambda x: x.rolling(5).mean())
df['當日漲跌幅'] = (df['Close_Price'] - df['Open_Price']) / df['Open_Price']
df['當日振幅'] = (df['High_Price'] - df['Low_Price']) / df['Open_Price']
df['5日乖離率'] = (df['Close_Price'] - df['MA_5']) / df['MA_5']
df['20日乖離率'] = (df['Close_Price'] - df['MA_20']) / df['MA_20']
df['均線差比率'] = (df['MA_5'] - df['MA_20']) / df['MA_20']
df['量比'] = df['Volume_Size'] / df['5日均量']
df['過去5日漲跌幅'] = g['Close_Price'].pct_change(5)
df['過去5日波動率'] = g['當日漲跌幅'].transform(lambda x: x.rolling(5).std())

low_20 = g['Low_Price'].transform(lambda x: x.rolling(20).min())
high_20 = g['High_Price'].transform(lambda x: x.rolling(20).max())
df['20日價格位置'] = (df['Close_Price'] - low_20) / (high_20 - low_20 + 1e-8)

def calc_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-8)
    return 100 - (100 / (1 + rs))

df['RSI_14'] = g['Close_Price'].transform(lambda x: calc_rsi(x, 14))

# 貼近實戰的標籤
df['Next_Open'] = g['Open_Price'].shift(-1)
df['Future_Close'] = g['Close_Price'].shift(-6)
df['Real_Return'] = (df['Future_Close'] - df['Next_Open']) / df['Next_Open']
df['Target'] = np.where(df['Real_Return'] >= 0.03, 1, 0)

feature_cols = [
    'is_listed', '當日漲跌幅', '當日振幅', '5日乖離率', '20日乖離率', 
    '均線差比率', '量比', '過去5日漲跌幅', '過去5日波動率', 
    '20日價格位置', 'RSI_14', '大盤_20日乖離率', '大盤_5日漲跌幅'
]

latest_date = df['Date'].max()
today_df = df[df['Date'] == latest_date].dropna(subset=feature_cols).copy()
train_df = df[df['Date'] < latest_date].dropna(subset=feature_cols + ['Real_Return']).copy()

# 訓練完整模型
X_train = train_df[feature_cols]
y_train = train_df['Target']
weight_ratio = (y_train == 0).sum() / (y_train == 1).sum()

print(f"🚀 2. 使用截至 {latest_date.strftime('%Y-%m-%d')} 前的所有歷史數據訓練模型...")
model = xgb.XGBClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.03,
    scale_pos_weight=weight_ratio, subsample=0.8, colsample_bytree=0.7,
    reg_alpha=1.0, reg_lambda=2.0, random_state=42, eval_metric='logloss'
)
model.fit(X_train, y_train)

# 預測最新交易日
today_df['AI_Prob'] = model.predict_proba(today_df[feature_cols])[:, 1]

# 【完整修正】確保讀取歷史數據特徵時不會出錯
market_bias = today_df['大盤_20日乖離率'].values[0]
print(f"\n📊 最新交易日 ({latest_date.strftime('%Y-%m-%d')}) 大盤 20 日乖離率：{market_bias:+.2%}")

if False:
    print("⚠️ 警告：目前大盤處於偏空跌勢...")
else:
    # 這裡將信心度稍微放寬到 0.45~0.75，並強行撈取前 10 名
    picks = today_df[
        (today_df['AI_Prob'] >= 0.45) & 
        (today_df['AI_Prob'] <= 0.75) & 
        (today_df['Volume_Size'] >= 500000)
    ].sort_values(by='AI_Prob', ascending=False).head(10)
    
    print("\n🎯 明日 AI 精選科技股關注名單（按信心度排序）：")
    if picks.empty:
        print("今日無符合流動性與放寬後條件的標的。")
    else:
        for _, row in picks.iterrows():
            vol_lots = int(row['Volume_Size'] / 1000)
            print(f"代號: {row['stock_id']} ({row['stock_name']}) | 收盤價: {row['Close_Price']:.2f} | 成交量: {vol_lots} 張 | AI信心度: {row['AI_Prob']:.2%}")

