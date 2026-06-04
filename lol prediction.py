"""
LOL 遊戲勝負預測 - League of Legends Win Prediction
使用前10分鐘數據預測藍方是否獲勝
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

def load_data(filepath: str) -> pd.DataFrame:
    """載入 CSV 資料集"""
    df = pd.read_csv(filepath)
    print(f"資料載入成功！共 {len(df)} 筆，{df.shape[1]} 個欄位")
    print(df.head())
    return df

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # 金幣優勢
    if "blueGold" in df.columns and "redGold" in df.columns:
        df["goldDiff"] = df["blueGold"] - df["redGold"]
    # 經驗值優勢
    if "blueTotalExperience" in df.columns and "redTotalExperience" in df.columns:
        df["expDiff"] = df["blueTotalExperience"] - df["redTotalExperience"]
    # KDA 差
    if all(c in df.columns for c in ["blueKills", "redKills"]):
        df["killDiff"] = df["blueKills"] - df["redKills"]
    # CS 差
    if all(c in df.columns for c in ["blueTotalMinionsKilled", "redTotalMinionsKilled"]):
        df["csDiff"] = df["blueTotalMinionsKilled"] - df["redTotalMinionsKilled"]

    print(f"特徵工程完成，新增欄位後共 {df.shape[1]} 個特徵")
    return df
    
def preprocess(df: pd.DataFrame, target: str = "blueWins"):
    """分割特徵與標籤、標準化"""
    df = df.dropna()
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"訓練集：{len(X_train)} 筆 | 測試集：{len(X_test)} 筆")
    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, X.columns
    
def train_random_forest(X_train, y_train):
    """Random Forest"""
    model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    print("Random Forest 訓練完成")
    return model
    
def train_xgboost(X_train_scaled, y_train):
    """XGBoost"""
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train_scaled, y_train)
    print("XGBoost 訓練完成")
    return model

def evaluate(model, X_test, y_test, model_name: str):
    """輸出準確率與分類報告"""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{'─'*40}")
    print(f"📊 [{model_name}] 測試集準確率：{acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["紅方勝", "藍方勝"]))
    return y_pred, acc

def plot_feature_importance(model, feature_names, model_name: str, top_n: int = 15):
    """特徵重要性長條圖"""
    importances = pd.Series(model.feature_importances_, index=feature_names)
    top = importances.nlargest(top_n).sort_values()

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top)))
    top.plot(kind="barh", ax=ax, color=colors)
    ax.set_title(f"Top {top_n} 特徵重要性 [{model_name}]", fontsize=14)
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig(f"feature_importance_{model_name.lower().replace(' ', '_')}.png", dpi=150)
    plt.show()
    print(f"特徵重要性圖已儲存")

def plot_confusion_matrix(y_test, y_pred, model_name: str):
    """混淆矩陣"""
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["紅方勝", "藍方勝"],
                yticklabels=["紅方勝", "藍方勝"], ax=ax)
    ax.set_title(f"混淆矩陣 [{model_name}]")
    ax.set_ylabel("實際")
    ax.set_xlabel("預測")
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    # ── 資料路徑（Kaggle: high_diamond_ranked_10min.csv）
    DATA_PATH = "high_diamond_ranked_10min.csv"

    df = load_data(DATA_PATH)
    df = feature_engineering(df)
    X_train, X_test, X_train_s, X_test_s, y_train, y_test, feat_names = preprocess(df)

    # ── Random Forest
    rf_model = train_random_forest(X_train, y_train)
    rf_pred, rf_acc = evaluate(rf_model, X_test, y_test, "Random Forest")
    plot_feature_importance(rf_model, feat_names, "Random Forest")
    plot_confusion_matrix(y_test, rf_pred, "Random Forest")

    # ── XGBoost
    xgb_model = train_xgboost(X_train_s, y_train)
    xgb_pred, xgb_acc = evaluate(xgb_model, X_test_s, y_test, "XGBoost")
    plot_confusion_matrix(y_test, xgb_pred, "XGBoost")

    print(f"\n 模型比較")
    print(f"   Random Forest 準確率：{rf_acc:.4f}")
    print(f"   XGBoost       準確率：{xgb_acc:.4f}")
    winner = "XGBoost" if xgb_acc >= rf_acc else "Random Forest"
    print(f" 最佳模型：{winner}")
