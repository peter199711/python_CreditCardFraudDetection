
# 🛡️ 個人化信用卡詐欺偵測系統 Fraud Detection Project

## 📌 專題簡介

本專題旨在建立一個能有效辨識信用卡詐欺交易的預警系統。透過機器學習技術與資料處理流程，從大量交易資料中偵測出潛在異常行為，協助金融單位與消費者即時發現風險，減少損失。

- **主題關鍵詞**：詐欺偵測、機器學習、資料不平衡、Logistic Regression、XGBoost
- **資料來源**：Kaggle - [Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

## 🔍 資料簡介

- **筆數**：284,807 筆交易紀錄
- **欄位數**：30 欄（28 個匿名化特徵 V1-V28 + 時間 Time + 金額 Amount + 類別 Class）
- **資料不平衡**：詐欺樣本僅佔 0.17%

## 🛠️ 資料處理流程

1. **資料探索（EDA）**
   - 檢查欄位分布與異常值
   - `Class` 欄分佈極端偏斜（正常交易佔 99.8%）

2. **重複值處理**
   ```python
   print(df.duplicated().sum())  # 1081 筆重複
   df = df.drop_duplicates()
   ```

3. **特徵縮放（MinMaxScaler）**
   - 對 `Amount` 欄進行標準化
   - `Time` 欄未列入模型訓練

4. **資料平衡處理（Undersampling）**
   - 將正常與詐欺樣本各取相同數量（492 筆）
   ```python
   fraud = df[df['Class'] == 1]
   non_fraud = df[df['Class'] == 0].sample(n=len(fraud), random_state=42)
   balanced_df = pd.concat([fraud, non_fraud])
   ```

5. **資料切分（Stratified Split）**
   - 確保訓練與測試資料中詐欺比例一致
   ```python
   X = balanced_df.drop('Class', axis=1)
   y = balanced_df['Class']
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
   ```

## 🤖 模型訓練與評估

### 使用模型：
- Logistic Regression
- （可擴充：XGBoost、Random Forest、異常偵測模型等）

### 評估指標：
- 準確率 Accuracy
- 精確率 Precision
- 召回率 Recall（關鍵指標）
- F1 分數

### 混淆矩陣與指標顯示：
| 指標 | 說明 |
|------|------|
| Precision | 預測為詐欺中真正為詐欺的比例 |
| Recall | 真正詐欺中被成功預測的比例 |
| F1-score | Precision 與 Recall 的加權平均 |

## 📊 特徵重要性分析

使用邏輯回歸的係數分析各特徵對結果的貢獻程度（正負值代表趨勢）：

```python
coef = model.coef_[0]
features = X.columns
plt.barh(features, coef)
plt.title("Feature Importance - Logistic Regression")
```

> 可見某些匿名變數（如 V14、V10、V17）對偵測詐欺交易具有高度貢獻。

## 🔍 分析結果與發現

- 在重採樣資料上，Logistic Regression 能有效偵測詐欺樣本（Recall > 90%）
- 特定特徵（如 V14、V10）有明顯的詐欺特徵權重
- 若改用 XGBoost，預期將提高偵測精度與處理能力
- 未來應評估異常偵測模型（Isolation Forest、Autoencoder）於原始不平衡資料集的表現

## 📌 未來方向

- 採用原始資料處理不平衡問題（SMOTE、異常偵測）
- 模型部署：串接 API 實現即時預警
- 加入時序元素、客戶 ID，打造更個人化的動態風控系統

## 📘 AlternativeMethods.ipynb 概覽

此筆記本示範在原始極度不平衡資料上，除了基礎邏輯回歸外的多種「替代方法」，並以更適合不平衡場景的指標（PR-AUC）做比較。

### 方法
- 監督式：
  - Logistic Regression（class_weight='balanced'，搭配閾值調整）
  - RandomForestClassifier（class_weight='balanced_subsample'）
- 非監督/半監督（異常偵測）：
  - IsolationForest（contamination≈訓練集詐欺比例）
  - Local Outlier Factor（novelty=True）

### 評估與視覺化
- 指標：ROC-AUC、PR-AUC（Average Precision）
- 曲線：ROC、Precision-Recall 曲線
- 閾值：可依目標 Recall（例：80%）自動挑選對應的決策閾值
- 解釋性：使用 SHAP 對 RandomForest 進行特徵重要度解讀（摘要圖/長條圖）

### 比較表如何解讀
- 表格按 PR-AUC 由高到低排序；在極不平衡情境下，PR-AUC比ROC-AUC更具代表性。
- PR-AUC 的隨機基線 ≈ 詐欺率；高於基線才有實質價值。
- 同一模型下，"~80%R" 代表將閾值調到達到目標 Recall 的運作點，可觀察 Precision 與誤報的取捨。
- 綜合常見結果：若有足夠且近期標註，監督式通常優於非監督；非監督適合補捉新型未知樣態。

### 如何執行
1. 安裝依賴：`pip install -r requirements.txt`
2. 確認專案根目錄存在 `creditcard.csv`
3. 開啟並依序執行 `AlternativeMethods.ipynb`

### 實務建議
- 部署以監督式為主（依業務目標 Recall 選閾值），非監督作為新型可疑流量的補捉器與特徵來源。
- 加上機率校準（Platt/Isotonic）、漂移監控與定期重訓排程。