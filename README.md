## 信用卡詐欺偵測：分析報告與結果總結

本專案針對 `creditcard.csv`（284,807 筆、詐欺率約 0.17%）建立偵測模型，核心目標是：在極度不平衡資料下，以穩健的資料處理與合理的決策閾值，取得高召回率的同時控制誤報。

---

### 1) 資料與問題特性
- 交易筆數 284,807、欄位 31（匿名化特徵 `V1–V28`、`Time`、`Amount`、`Class`）。
- 正負樣本極不平衡（詐欺約 0.17%）；僅看 Accuracy 會嚴重誤導，因此評估以 Recall、Precision、F1 與 PR-AUC 為主，並搭配 ROC-AUC 參考。
- 發現 1,081 筆重複資料，已移除以降低偏差。

---

### 2) 資料處理流程（重點）
- 數值縮放：將 `Amount`、`Time` 標準化為 `scaled_amount`、`scaled_time`，再移除原欄位；其餘 `V1–V28` 直接使用。
- 資料分割：採 `Stratified` 方式切 8:2（訓練/測試），保持詐欺比例一致。
- 不平衡處理（兩路線並行測試）：
  - A. SMOTE：僅對訓練集過採樣（避免洩漏），再訓練 LR/XGBoost。
  - B. 不重抽樣：維持原始分佈，於 XGBoost 使用 `scale_pos_weight`（≈ 負/正樣本比）處理不平衡，並以閾值調整控制誤報/漏報。

---

### 3) 模型選擇與調整策略
- 基線模型：Logistic Regression（易解釋、訓練快速），作為可比較的下限。
- 進階模型：XGBoost（處理非線性與交互效應、對不平衡具彈性），同時觀察特徵重要度以輔助解釋。
- 閾值管理：在極不平衡情境下，固定 0.5 並不合適；依目標 Recall 或成本函數（漏報成本 ≫ 誤報成本）選擇決策閾值，能大幅改善實務效益。

---

### 4) 關鍵比較：調整前後的差異
以下指標取自專案 Notebook 的實驗結果（不同設定對照），著重「取捨」而非單一數字的極致：

| 設定 | 抽樣/權重 | 主要處理 | 指標（測試集） |
|---|---|---|---|
| Logistic Regression（基線） | 不重抽樣 | 僅縮放 `Amount/Time` | Precision 0.829、Recall 0.643、F1 0.724、ROC-AUC 0.958 |
| Logistic + SMOTE | 訓練集過採樣 | 追求召回 | Precision 0.058、Recall 0.918、F1 0.109（召回高但誤報過多） |
| XGBoost + SMOTE | 訓練集過採樣 | 非線性學習 | Precision 0.731、Recall 0.888、F1 0.802 |
| XGBoost + scale_pos_weight | 不重抽樣 | 權重處理不平衡 + 閾值調整 | 預設閾值：Recall ≈ 0.82、AUC ≈ 0.972；最佳 F1 閾值：Precision 0.96、Recall 0.76、F1 0.85 |

重點觀察：
- 單純 SMOTE 套在 LR 能大幅拉高 Recall，但 Precision 崩跌，營運上成本極高。
- XGBoost 在兩種路線皆穩定，`scale_pos_weight` 方案不需改變分佈，搭配「以目標 Recall/成本選閾值」可取得更實用的取捨（例如 Precision 0.96、Recall 0.76 的平衡點）。

---

### 5) 特徵重要性與解釋
- XGBoost 重要度一致顯示：`V14` 影響最大，其次常見於前排的有 `V4`、`V12`、`V8`、`V1`、`V13`、`V3` 等；`scaled_amount`、`scaled_time` 亦具輔助性。
- 隨機森林與 LR 係數排名亦支持 `V10`、`V14`、`V16–V20` 等特徵的區辨度，整體趨勢一致，提升了結論可信度。

---

### 6) 閾值與成本：實務運作建議
- 先明確「目標召回」（如 ≥0.85）或「誤報/漏報成本比」（如 C_fn=100、C_fp=1），再由驗證集/測試集的分數曲線選擇閾值。
- 對需要機率輸出管理告警量的情境，可進一步做機率校準（如 isotonic），以取得更可靠的機率值與可解釋的告警策略。

---

### 7) 推薦預設方案（MVP）
1. 使用 `XGBoost + scale_pos_weight`（不重抽樣），以 `stratified` 切分，僅縮放 `Amount`、`Time`。
2. 以驗證集 PR 曲線選擇達成業務目標的決策閾值；若需成本最小化，最小化 `C_fn*FN + C_fp*FP` 取得閾值。
3. 上線時持續監控正負例比例與特徵漂移，定期重訓與閾值微調。

---

### 8) 專案檔案導覽（摘要）
- `FE_LogisticRegressionXGBoost.ipynb`：從基線 LR 到 SMOTE 與 XGBoost 的整體流程與對照。
- `XGBoost.ipynb`：兩路線比較（重抽樣 vs `scale_pos_weight`），含 AUC、PR-AUC、閾值調整與 SHAP。
- `AlternativeMethods.ipynb`：在原始極不平衡資料上的監督/非監督方法（RF、IsolationForest、LOF）、PR-AUC 與成本導向閾值。
- `basic_LogisticRegression.ipynb`：簡化版本的 LR 基線流程（可作為教學/起點）。

---

### 9) 快速開始（一句話即可）
安裝 `requirements.txt`、確保根目錄有 `creditcard.csv`，依序執行上列 Notebook；若需中文圖表且無警告，可在第一個 cell 設定字型並 `warnings.filterwarnings("ignore")`。

> 工具/套件：主要使用 scikit-learn、XGBoost、seaborn/matplotlib、imbalanced-learn（SMOTE）、SHAP；Python 版本與完整相依請參見 `requirements.txt`。


