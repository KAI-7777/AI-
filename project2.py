import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb

# 1. 載入個股資料
print("💾 1. 載入個股歷史數據...")
df = pd.read_csv("all_tech_stocks_data.csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by=['stock_id', 'Date']).reset_index(drop=True)

# 2. 下載 0050 作為台股大盤多空濾網
print("🌐 2. 下載 0050 數據建立大盤環境特徵...")
tw50 = yf.download("0050.TW", start="2025-01-01", progress=False, auto_adjust=True, group_by='ticker').reset_index()

# 正確攤平 MultiIndex 欄位
if isinstance(tw50.columns, pd.MultiIndex):
    tw50.columns = [col[1] if col[1] != '' else col[0] for col in tw50.columns]

tw50['Date'] = pd.to_datetime(tw50['Date'])
tw50['大盤_20日均線'] = tw50['Close'].rolling(20).mean()
tw50['大盤_20日乖離率'] = (tw50['Close'] - tw50['大盤_20日均線']) / tw50['大盤_20日均線']
tw50['大盤_5日漲跌幅'] = tw50['Close'].pct_change(5)

# 將大盤特徵合併進個股資料表
df = pd.merge(df, tw50[['Date', '大盤_20日乖離率', '大盤_5日漲跌幅']], on='Date', how='left')

# 3. 進階特徵工程（趨勢、動能、相對位置）
print("⚙️ 3. 計算進階技術指標特徵...")
g = df.groupby('stock_id')

# 基礎比率特徵
df['5日均量'] = g['Volume_Size'].transform(lambda x: x.rolling(5).mean())
df['當日漲跌幅'] = (df['Close_Price'] - df['Open_Price']) / df['Open_Price']
df['當日振幅'] = (df['High_Price'] - df['Low_Price']) / df['Open_Price']
df['5日乖離率'] = (df['Close_Price'] - df['MA_5']) / df['MA_5']
df['20日乖離率'] = (df['Close_Price'] - df['MA_20']) / df['MA_20']
df['均線差比率'] = (df['MA_5'] - df['MA_20']) / df['MA_20']
df['量比'] = df['Volume_Size'] / df['5日均量']

# 新增：過去 5 日累積漲跌幅（動能）與 5 日波動率
df['過去5日漲跌幅'] = g['Close_Price'].pct_change(5)
df['過去5日波動率'] = g['當日漲跌幅'].transform(lambda x: x.rolling(5).std())

# 新增：近 20 日價格相對位置（0 代表 20 日最低點，1 代表 20 日最高點）
low_20 = g['Low_Price'].transform(lambda x: x.rolling(20).min())
high_20 = g['High_Price'].transform(lambda x: x.rolling(20).max())
df['20日價格位置'] = (df['Close_Price'] - low_20) / (high_20 - low_20 + 1e-8)

# 新增：14 日 RSI 指標
def calc_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-8)
    return 100 - (100 / (1 + rs))

df['RSI_14'] = g['Close_Price'].transform(lambda x: calc_rsi(x, 14))

# 4. 標籤工程（預測未來 5 個交易日漲幅 >= 3%）
df['Future_Close'] = g['Close_Price'].shift(-5)
df['Future_Return'] = (df['Future_Close'] - df['Close_Price']) / df['Close_Price']
df['Target'] = np.where(df['Future_Return'] >= 0.03, 1, 0)

df = df.dropna().reset_index(drop=True)

feature_cols = [
    'is_listed', '當日漲跌幅', '當日振幅', '5日乖離率', '20日乖離率', 
    '均線差比率', '量比', '過去5日漲跌幅', '過去5日波動率', 
    '20日價格位置', 'RSI_14', '大盤_20日乖離率', '大盤_5日漲跌幅'
]

X = df[feature_cols]
y = df['Target']

# 5. 嚴格時間軸切割（留緩衝期防止洩漏）
split_date = pd.to_datetime("2026-08-01")
buffer_date = pd.to_datetime("2026-07-24")

train_mask = df['Date'] < buffer_date
test_mask = df['Date'] >= split_date

X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test = X[test_mask], y[test_mask]

# 6. 訓練模型（調整正則化參數防止過度擬合雜訊）
neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
weight_ratio = neg_count / pos_count

print("🚀 4. 開始訓練升級版 XGBoost 模型...")
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.03,
    scale_pos_weight=weight_ratio,
    subsample=0.8,
    colsample_bytree=0.7,
    reg_alpha=1.0,
    reg_lambda=2.0,
    random_state=42,
    eval_metric='logloss'
)
model.fit(X_train, y_train)

# 7. 信心門檻與真實報酬率體檢
y_probs = model.predict_proba(X_test)[:, 1]
test_returns = df.loc[test_mask, 'Future_Return']

print("\n🎯 升級版模型：不同信心門檻下的勝率與平均報酬率：")
for threshold in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
    mask = (y_probs >= threshold)
    count = mask.sum()
    if count > 0:
        win_3pct = (y_test[mask] == 1).mean()
        win_pos = (test_returns[mask] > 0).mean()
        avg_ret = test_returns[mask].mean()
        print(f"門檻 >= {threshold:.0%} | 出手: {count:5d} 次 | 大漲>=3%率: {win_3pct:.2%} | 正報酬勝率: {win_pos:.2%} | 平均5日報酬: {avg_ret:+.2%}")
    else:
        print(f"門檻 >= {threshold:.0%} | 出手:     0 次")

importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\n🔥 升級版 AI 特徵重要性排名：")
print(importances)

# 8. 模擬實戰交易：甜蜜區間 + 每日 Top 3 選股回測（含交易成本）
print("\n💼 5. 開始進行模擬資金回測（2026/08 以後）...")

# 建立回測專用表格
bt_df = df.loc[test_mask, ['Date', 'stock_id', 'stock_name', 'Future_Return', '過去5日波動率', '大盤_20日乖離率']].copy()
bt_df['AI_Prob'] = y_probs

# 設定實戰濾網：信心度落在 58% ~ 66% 甜蜜區間，且大盤處於偏多格局（20日乖離 > -1%）
cond = (
    (bt_df['AI_Prob'] >= 0.58) & 
    (bt_df['AI_Prob'] <= 0.66) & 
    (bt_df['大盤_20日乖離率'] > -0.01)
)
filtered_signals = bt_df[cond].copy()

# 每天只挑選信心度最高的前 3 檔股票（模擬真實資金有限的情況）
daily_top3 = filtered_signals.sort_values(by=['Date', 'AI_Prob'], ascending=[True, False]).groupby('Date').head(3)

# 計算扣除台股交易成本後的淨報酬（手續費折讓後約 0.1% + 證交稅 0.3% = 0.4%）
trading_cost = 0.004
daily_top3['Net_Return'] = daily_top3['Future_Return'] - trading_cost

# 計算每日平均報酬（因為持有 5 天，假設資金分成 5 等份每日輪動）
daily_portfolio = daily_top3.groupby('Date')['Net_Return'].mean() / 5.0
cumulative_return = (1 + daily_portfolio).cumprod() - 1

print(f"📅 測試期間總交易天數：{len(bt_df['Date'].unique())} 天（實際有出手天數：{len(daily_portfolio)} 天）")
print(f"🎯 總計買進次數：{len(daily_top3)} 筆（平均每天買 {len(daily_top3)/max(len(daily_portfolio),1):.1f} 檔）")
print(f"✅ 扣稅費後單筆正報酬勝率：{(daily_top3['Net_Return'] > 0).mean():.2%}")
print(f"💵 扣稅費後平均每筆淨報酬：{daily_top3['Net_Return'].mean():+.2%}")
print(f"📈 期間資金輪動累積總報酬率：{cumulative_return.iloc[-1]:+.2%}")

