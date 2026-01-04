import pandas as pd
import numpy as np
from config import RISK_THRESHOLDS
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """处理缺失值：数值型用行业中位数，分类型用众数"""
    # 数值型字段填充
    numeric_fields = ["automation_risk", "openings_2024", "openings_2030", "salary_usd", "experience_years"]
    for field in numeric_fields:
        if field in df.columns:
            df[field] = df.groupby("industry")[field].transform(lambda x: x.fillna(x.median()))
    
    # 分类型字段填充
    categorical_fields = ["job_role", "industry", "country", "education", "job_status"]
    for field in categorical_fields:
        if field in df.columns:
            df[field] = df[field].fillna(df[field].mode()[0])
    
    return df

def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """处理异常值（IQR法，仅处理核心数值字段）"""
    outlier_fields = ["openings_2024", "openings_2030", "salary_usd", "automation_risk"]
    for field in outlier_fields:
        if field in df.columns and not df[field].isnull().all():
            Q1 = df[field].quantile(0.25)
            Q3 = df[field].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[field] >= lower_bound) & (df[field] <= upper_bound)]
    
    print(f"✅ 异常值处理完成，剩余数据：{df.shape[0]}行")
    return df

def create_risk_level(df: pd.DataFrame) -> pd.DataFrame:
    """创建自动化风险等级字段"""
    def _assign_risk(score):
        if pd.isna(score):
            return "unknown"
        elif score <= RISK_THRESHOLDS["low"]:
            return "low"
        elif score <= RISK_THRESHOLDS["medium"]:
            return "medium"
        else:
            return "high"
    
    df["risk_level"] = df["automation_risk"].apply(_assign_risk)
    return df

def create_encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """编码分类型特征（标签编码）"""
    categorical_fields = ["industry", "country", "education", "job_status", "risk_level"]
    le = LabelEncoder()
    for field in categorical_fields:
        if field in df.columns:
            df[f"{field}_encoded"] = le.fit_transform(df[field].astype(str))
    return df

def create_normalize_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """改进版归一化数值特征，处理极值影响"""
    numeric_fields = ["automation_risk", "salary_usd", "experience_years", 
                      "remote_ratio", "employment_change_rate"]
    
    for field in numeric_fields:
        if field not in df.columns:
            continue
            
        data = df[field].copy()
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR  
        # 对极端异常值进行截断或缩尾处理
        clipped_data = np.clip(data, lower_bound, upper_bound)
        scaler = MinMaxScaler()
        df[f"{field}_norm"] = scaler.fit_transform(clipped_data.values.reshape(-1, 1)) 
    return df

def create_ai_impact(df: pd.DataFrame) -> pd.DataFrame:
    if "ai_impact_level" in df.columns:
        impact_mapping = {"Low": 0.0, "Moderate": 0.0, "High": 1.0, "Unknown": 0}
        df["ai_impact_level_norm"] = df["ai_impact_level"].map(impact_mapping).fillna(0)
    return df

def create_future_features(df: pd.DataFrame) -> pd.DataFrame:
    """创建用于未来预测的特征"""
    # AI普及度趋势特征（假设每年增长）
    current_year = 2024
    target_year = 2030
    years_diff = target_year - current_year
    
    # 假设AI影响随时间增长
    df["ai_adoption_trend"] = df["ai_impact_level_norm"] * (1 + 0.1 * years_diff)
    df["ai_adoption_trend"] = df["ai_adoption_trend"].clip(upper=1.0)  # 上限为1
    
    # 远程工作趋势（假设增长）
    df["remote_work_trend"] = df["remote_ratio_norm"] * (1 + 0.05 * years_diff)
    df["remote_work_trend"] = df["remote_work_trend"].clip(upper=1.0)
    
    # 行业自动化成熟度
    industry_automation = df.groupby("industry")["automation_risk"].mean().to_dict()
    df["industry_automation_maturity"] = df["industry"].map(industry_automation)
    
    return df

def preprocess_for_prediction(df: pd.DataFrame) -> pd.DataFrame:
    """完整的预测预处理流程"""
    df = handle_missing_values(df)
    df = handle_outliers(df)
    df = create_risk_level(df)
    df = create_encode_categorical(df)
    df = create_normalize_numeric(df)
    df = create_ai_impact(df)
    df = create_future_features(df)
    return df