import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
import joblib
from config import OUTPUT_PATH, TARGET_YEAR

# 创建模型保存目录
MODEL_PATH = os.path.join(OUTPUT_PATH, "models/")
os.makedirs(MODEL_PATH, exist_ok=True)

def train_employment_change_model(df: pd.DataFrame) -> tuple:
    """
    训练就业变化率预测模型
    :param df: 预处理后的数据集
    :return: 训练好的模型和评估指标
    """
    # 选择特征和目标变量
    features = [
        "automation_risk_norm", "salary_usd_norm", "experience_years_norm",
        "remote_ratio_norm", "ai_impact_level_norm",
        "industry_encoded", "country_encoded", "education_encoded"
    ]
    features = [f for f in features if f in df.columns]
    target = "employment_change_rate"
    
    # 去除目标变量为空的样本
    df_model = df.dropna(subset=[target] + features)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        df_model[features], df_model[target], test_size=0.2, random_state=42
    )
    
    # 定义模型 pipeline
    model = Pipeline([
        ("regressor", RandomForestRegressor(random_state=42))
    ])
    
    # 网格搜索参数
    param_grid = {
        "regressor__n_estimators": [100, 200],
        "regressor__max_depth": [None, 10, 20],
        "regressor__min_samples_split": [2, 5]
    }
    
    # 网格搜索优化模型
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring="neg_mean_squared_error")
    grid_search.fit(X_train, y_train)
    
    # 评估模型
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    # 保存模型
    model_path = os.path.join(MODEL_PATH, "employment_change_model.pkl")
    joblib.dump(best_model, model_path)
    print(f"📦 就业变化预测模型已保存：{model_path}")
    
    # 特征重要性
    feature_importance = pd.DataFrame({
        "feature": features,
        "importance": best_model.named_steps["regressor"].feature_importances_
    }).sort_values(by="importance", ascending=False)
    
    return best_model, {"rmse": rmse, "feature_importance": feature_importance}

def train_automation_risk_model(df: pd.DataFrame) -> tuple:
    """
    训练自动化风险分类模型（预测高风险工作）
    :param df: 预处理后的数据集
    :return: 训练好的模型和评估指标
    """
    # 选择特征和目标变量
    features = [
        "salary_usd_norm", "experience_years_norm", "remote_ratio_norm",
        "ai_impact_level_norm", "employment_change_rate",
        "industry_encoded", "country_encoded", "education_encoded"
    ]
    features = [f for f in features if f in df.columns]
    # 目标变量：是否为高风险岗位
    target = "is_high_risk"
    df[target] = (df["risk_level"] == "high").astype(int)
    
    # 去除目标变量为空的样本
    df_model = df.dropna(subset=[target] + features)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        df_model[features], df_model[target], test_size=0.2, random_state=42, stratify=df_model[target]
    )
    
    # 定义模型 pipeline
    model = Pipeline([
        ("classifier", GradientBoostingClassifier(random_state=42))
    ])
    
    # 网格搜索参数
    param_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [3, 5, 10],
        "classifier__learning_rate": [0.01, 0.1]
    }
    
    # 网格搜索优化模型
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring="roc_auc")
    grid_search.fit(X_train, y_train)
    
    # 评估模型
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    # 计算评估指标
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    report = classification_report(y_test, y_pred)
    
    # 保存模型
    model_path = os.path.join(MODEL_PATH, "automation_risk_model.pkl")
    joblib.dump(best_model, model_path)
    print(f"📦 自动化风险预测模型已保存：{model_path}")
    
    # 特征重要性
    feature_importance = pd.DataFrame({
        "feature": features,
        "importance": best_model.named_steps["classifier"].feature_importances_
    }).sort_values(by="importance", ascending=False)
    
    return best_model, {"roc_auc": roc_auc, "classification_report": report, "feature_importance": feature_importance}

def predict_future_employment(df: pd.DataFrame, model=None) -> pd.DataFrame:
    """
    预测未来就业市场变化
    :param df: 预处理后的数据集
    :param model: 可选，已训练好的模型
    :return: 包含预测结果的DataFrame
    """
    if model is None:
        # 加载预训练模型
        model_path = os.path.join(MODEL_PATH, "employment_change_model.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在：{model_path}，请先训练模型")
        model = joblib.load(model_path)
    
    # 选择特征
    features = [
        "automation_risk_norm", "salary_usd_norm", "experience_years_norm",
        "remote_ratio_norm", "ai_impact_level_norm",
        "industry_encoded", "country_encoded", "education_encoded"
    ]
    features = [f for f in features if f in df.columns]
    
    # 预测就业变化率
    df["predicted_employment_change"] = model.predict(df[features])
    
    # 预测2030年岗位数量
    df["predicted_openings_2030"] = df.apply(
        lambda row: row["openings_2024"] * (1 + row["predicted_employment_change"]), axis=1
    )
    
    return df

def identify_high_risk_jobs(df: pd.DataFrame, model=None) -> pd.DataFrame:
    """
    识别高风险岗位
    :param df: 预处理后的数据集
    :param model: 可选，已训练好的模型
    :return: 包含风险预测的DataFrame
    """
    if model is None:
        # 加载预训练模型
        model_path = os.path.join(MODEL_PATH, "automation_risk_model.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在：{model_path}，请先训练模型")
        model = joblib.load(model_path)
    
    # 选择特征
    features = [
        "salary_usd_norm", "experience_years_norm", "remote_ratio_norm",
        "ai_impact_level_norm", "employment_change_rate",
        "industry_encoded", "country_encoded", "education_encoded"
    ]
    features = [f for f in features if f in df.columns]
    
    # 预测高风险概率
    df["high_risk_probability"] = model.predict_proba(df[features])[:, 1]
    
    # 预测是否为高风险岗位
    df["predicted_risk_level"] = model.predict(df[features])
    df["predicted_risk_level"] = df["predicted_risk_level"].map({0: "not_high", 1: "high"})
    
    return df