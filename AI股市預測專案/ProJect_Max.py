import subprocess
import sys
import os
import time

# 取得目前程式所在的資料夾路徑，確保不會找不到檔案
current_dir = os.path.dirname(os.path.abspath(__file__))
p1_path = os.path.join(current_dir, "project1.py")
p3_path = os.path.join(current_dir, "project3.py")

print("==================================================")
print("🤖 AI 量化選股自動化系統啟動")
print("==================================================\n")

start_time = time.time()

# 步驟 1：執行 project1.py 下載最新台股數據
print("▶️ [步驟 1/2] 正在執行 project1.py 更新全台科技股數據...")
res1 = subprocess.run([sys.executable, p1_path])

if res1.returncode != 0:
    print("\n❌ project1.py 執行發生錯誤，已中止後續選股程序！請檢查網路或錯誤訊息。")
    sys.exit(1)

print("\n✅ [步驟 1/2] 數據更新完成！\n")
print("--------------------------------------------------\n")

# 步驟 2：執行 project3.py 進行 AI 運算並產出明日選股清單
print("▶️ [步驟 2/2] 正在執行 project3.py 進行 AI 模型運算與選股...")
res3 = subprocess.run([sys.executable, p3_path])

if res3.returncode != 0:
    print("\n❌ project3.py 執行發生錯誤，請檢查錯誤訊息。")
    sys.exit(1)

elapsed = time.time() - start_time
print("\n==================================================")
print(f"🎉 全部流程執行完畢！總共耗時：{elapsed:.1f} 秒")
print("==================================================")
