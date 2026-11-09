import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False

print("💾 1. 正在載入數據並鎖定【台積電 (2330)】進行獨立高精準訓練...")
df_all = pd.read_csv("all_tech_stocks_data.csv")

# 【關鍵修正 1】只抓取台積電，完全隔離其他 898 檔股票的雜訊
df = df_all[df_all['stock_id'] == 2330].copy()

df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by='Date').reset_index(drop=True)

print("🛠️ 2. 獨立計算台積電的季線特徵...")
# 計算台積電自己的 60日均線
df['MA_60'] = df['Close_Price'].rolling(window=60).mean()
df['Close_60_days_ago'] = df['Close_Price'].shift(60)
df = df.dropna(subset=['MA_60', 'Close_60_days_ago'])

# 相對百分比特徵
df['MA60_Bias'] = df['Close_Price'] / df['MA_60'] - 1
df['Deduct_Strength'] = (df['Close_Price'] - df['Close_60_days_ago']) / df['Close_60_days_ago']
df['Day_Amplitude'] = (df['High_Price'] - df['Low_Price']) / df['Close_Price']
df['Volume_Ratio'] = df['Volume_Size'] / df['Volume_Size'].rolling(window=20).mean()
df = df.dropna(subset=['Volume_Ratio'])

# 預測目標：5天後的實際季線
df['Future_MA60_Value'] = df['MA_60'].shift(-5)
df = df.dropna(subset=['Future_MA60_Value'])

feature_cols = ['MA60_Bias', 'Deduct_Strength', 'Day_Amplitude', 'Volume_Ratio']

# 【關鍵修正 2】微調盲測時間軸，確保訓練集與測試集都有足夠的台積電資料
# 拿前 75% 的日子來訓練，最後 25% 的日子（近期盲測）來考試
split_idx = int(len(df) * 0.75)
train_df = df.iloc[:split_idx]
test_df = df.iloc[split_idx:]

X_train, y_train = train_df[feature_cols], train_df['Future_MA60_Value']
X_test, y_test = test_df[feature_cols], test_df['Future_MA60_Value']

print(f"📚 訓練樣本數（歷史）：{len(X_train)} 天")
print(f"📝 測試樣本數（未來）：{len(X_test)} 天")

# 3. 訓練模型
print("🚀 3. 開始訓練台積電專屬 XGBoost 趨勢模型...")
model = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)
model.fit(X_train, y_train)

# 4. 預測與評估
test_df['AI_Predicted_MA60'] = model.predict(X_test)

mae = mean_absolute_error(test_df['Future_MA60_Value'], test_df['AI_Predicted_MA60'])
r2 = r2_score(test_df['Future_MA60_Value'], test_df['AI_Predicted_MA60'])

print("\n================== 📈 AI 實戰期末考數據分析（修正版） ==================")
print(f"🎯 平均絕對誤差 (MAE): {mae:.2f} 元 (AI 提早5天預測，平均只差了 {mae:.2f} 元！)")
print(f"📈 趨勢擬合度 (R² Score): {r2:.4f} (越接近 1.0 代表預測越完美)")
print("========================================================================")

# 5. 繪製合併比較圖
plt.figure(figsize=(12, 6))
plt.plot(test_df['Date'], test_df['Future_MA60_Value'], label='真實發生的未來季線', color='#1f77b4', linewidth=2.5)
plt.plot(test_df['Date'], test_df['AI_Predicted_MA60'], label=f'AI 提早5天預測線 (誤差: ±{mae:.1f}元)', color='#d62728', linestyle='--', linewidth=2)

plt.title('台積電 (2330) 專屬模型 - 季線預測與真實未來趨勢對照圖', fontsize=14, fontweight='bold')
plt.xlabel('日期', fontsize=12)
plt.ylabel('季線價格 (元)', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(fontsize=11, loc='upper left')
plt.xticks(rotation=30)
plt.tight_layout()

plt.savefig('verify_tsmc_ma60_fixed.png', dpi=300)
plt.show()
