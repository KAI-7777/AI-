import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb

# 1. 載入資料
print("💾 正在載入 899 檔科技股的大數據...")
df = pd.read_csv("all_tech_stocks_data.csv")

# 確保日期格式正確並排序
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by=['stock_id', 'Date']).reset_index(drop=True)

# 2. 標籤工程（預測目標）
df['Future_Close'] = df.groupby('stock_id')['Close_Price'].shift(-5)
df['Future_Return'] = (df['Future_Close'] - df['Close_Price']) / df['Close_Price']
df['Target'] = np.where(df['Future_Return'] >= 0.03, 1, 0)
df = df.dropna(subset=['Future_Close'])

# 3. 準備特徵（將原本的英文欄位對應到中文欄位名稱）
df = df.rename(columns={
    'is_listed': '是否上市',
    'Open_Price': '今日開盤價',
    'High_Price': '今日最高價',
    'Low_Price': '今日最低價',
    'Close_Price': '今日收盤價',
    'Volume_Size': '今日成交量',
    'MA_5': '5日均線值',
    'MA_20': '20日均線值'
})

# 定義要餵給 AI 學習的中文特徵清單
feature_cols = ['是否上市', '今日開盤價', '今日最高價', '今日最低價', '今日收盤價', '今日成交量', '5日均線值', '20日均線值']
X = df[feature_cols]
y = df['Target']

# 4. 【實戰修正】嚴格依據時間軸切割資料
print("\n🔍 4. 嚴格依據時間軸切割資料...")

# 2026/08/01 以前作為歷史訓練，以後作為未來模擬考
split_date = pd.to_datetime("2026-08-01")

train_mask = df['Date'] < split_date
test_mask = df['Date'] >= split_date

X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test = X[test_mask], y[test_mask]

print(f"📚 訓練集（歷史）：{len(X_train)} 筆資料")
print(f"📝 測試集（未來）：{len(X_test)} 筆資料")

# 5. 訓練模型
print("\n🚀 5. 開始訓練 XGBoost AI 模型...")
model = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, eval_metric='logloss')
model.fit(X_train, y_train)

# 6. 模擬考評分
print("\n🎯 6. 訓練完成！開始對測試集（2026年8月以後）進行模擬考測驗...")
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"📈 AI 模型的真實預測準確度: {accuracy:.2%}")

print("\n📄 詳細預測報告：")
report = classification_report(y_test, y_pred, output_dict=True)

# 將英文的評分報告表格轉為漂亮的中文呈現
report_df = pd.DataFrame(report).transpose()
report_df = report_df.rename(index={
    '0': '沒大漲 (0)',
    '1': '有大漲 (1)',
    'accuracy': '整體準確度',
    'macro avg': '總體平均',
    'weighted avg': '加權平均'
}, columns={
    'precision': '精準度(抓對率)',
    'recall': '捕捉率(沒漏掉率)',
    'f1-score': '綜合分數(F1)',
    'support': '樣本總數'
})
print(report_df.round(2))

# 印出中文特徵重要性排名
importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\n🔥 AI 心中最重要的特徵排名：")
print(importances)
