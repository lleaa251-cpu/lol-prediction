# 🏆 League of Legends 勝負預測

> 使用前 10 分鐘的遊戲數據，透過機器學習預測藍方是否獲勝

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-red)
![License](https://img.shields.io/badge/License-MIT-green)

-----

## 📖 專案簡介

本專案利用 **League of Legends 鑽石段位天梯對戰資料**，
分析前 10 分鐘的遊戲數據（金幣差、擊殺數、龍等），
訓練二元分類模型預測最終勝負。

**核心問題：前 10 分鐘的優勢真的能決定比賽結果嗎？**

-----

## 📊 資料集

|項目|說明                                                                                                                                     |
|--|---------------------------------------------------------------------------------------------------------------------------------------|
|來源|[Kaggle - League of Legends Diamond Ranked Games](https://www.kaggle.com/datasets/bobbyscience/league-of-legends-diamond-ranked-10-min)|
|筆數|約 10,000 場對戰                                                                                                                           |
|標籤|`blueWins`（1 = 藍方勝，0 = 紅方勝）                                                                                                            |

### 主要特徵

|特徵                      |說明    |
|------------------------|------|
|`blueKills` / `redKills`|擊殺數   |
|`blueGold` / `redGold`  |金幣總量  |
|`blueTotalExperience`   |藍方總經驗值|
|`blueDragons`           |龍擊殺數  |
|`blueFirstBlood`        |是否拿到一血|
|`blueTotalMinionsKilled`|補兵數   |
|`goldDiff` *(自建)*       |藍紅金幣差 |
|`expDiff` *(自建)*        |藍紅經驗值差|
|`killDiff` *(自建)*       |藍紅擊殺差 |

-----

## 🚀 快速開始

### 1. 克隆專案

```bash
git clone https://github.com/your-username/lol-win-prediction.git
cd lol-win-prediction
```

### 2. 安裝套件

```bash
pip install -r requirements.txt
```

### 3. 下載資料集

前往 [Kaggle](https://www.kaggle.com/datasets/bobbyscience/league-of-legends-diamond-ranked-10-min)
下載 `high_diamond_ranked_10min.csv`，放入專案根目錄。

### 4. 執行訓練

```bash
python lol_prediction.py
```

-----

## 🗂️ 專案結構

```
lol-win-prediction/
│
├── lol_prediction.py          # 主程式（訓練 + 評估 + 視覺化）
├── requirements.txt           # 套件清單
├── high_diamond_ranked_10min.csv  # 資料集（自行下載）
│
├── outputs/
│   ├── feature_importance_random_forest.png
│   ├── feature_importance_xgboost.png
│   ├── confusion_matrix_random_forest.png
│   └── confusion_matrix_xgboost.png
│
└── README.md
```

-----

## 🧠 模型與結果

### 使用模型

|模型               |說明           |
|-----------------|-------------|
|**Random Forest**|集成決策樹，抗過擬合強  |
|**XGBoost**      |梯度提升，通常有更高準確率|

### 實驗結果（參考）

|模型           |測試集準確率|
|-------------|------|
|Random Forest|~72%  |
|XGBoost      |~74%  |


> ⚠️ 準確率上限天然偏低，因為前 10 分鐘不能完全決定勝負 — 這正是這個題目最有趣的地方！

### 最重要的特徵 TOP 5

1. 💰 `goldDiff` — 金幣差距
1. 📈 `expDiff` — 經驗值差距
1. 🐉 `blueDragons` — 龍數量
1. ⚔️ `killDiff` — 擊殺差距
1. 🏰 `blueTowersDestroyed` — 摧毀塔數

-----

## 🔧 ML 流程

```
原始資料
   ↓
特徵工程（差值特徵）
   ↓
資料清理 & 標準化
   ↓
Train / Test Split（8:2）
   ↓
Random Forest + XGBoost 訓練
   ↓
評估（Accuracy / F1 / 混淆矩陣）
   ↓
特徵重要性視覺化
```

-----

## 📦 套件需求

```
pandas>=1.5
numpy>=1.23
scikit-learn>=1.3
xgboost>=2.0
matplotlib>=3.7
seaborn>=0.12
```

安裝：

```bash
pip install -r requirements.txt
```

-----

## 🌱 未來改進方向

- [ ] 加入 SHAP 值解釋模型決策
- [ ] 嘗試 LightGBM、CatBoost
- [ ] 用 GridSearchCV 做超參數調整
- [ ] 串接 Riot API 取得即時對戰資料
- [ ] 部署為 Web App（Streamlit）

-----

## 📄 License

本專案採用 [MIT License](LICENSE)。

-----

## 🙏 致謝

- 資料集來源：[bobbyscience @ Kaggle](https://www.kaggle.com/bobbyscience)
- 靈感來自對 LOL 勝負關鍵因素的好奇心 🎮