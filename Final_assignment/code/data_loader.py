import pandas as pd
import os
from config import DATA_PATH, FIELD_MAPPING

def load_data(file_path: str = DATA_PATH) -> pd.DataFrame:
    """
    加载CSV数据，适配字段映射，处理百分比/数值转换
    :return: 标准化后的DataFrame
    """
    # 1. 校验文件存在性
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据文件不存在：{file_path}\n请检查DATA_PATH配置")
    
    # 2. 加载原始数据
    df = pd.read_csv(file_path, encoding="utf-8")
    print(f"✅ 原始数据加载完成：{df.shape[0]}行 × {df.shape[1]}列")
    print(f"原始字段列表：{df.columns.tolist()}")
    
    # 3. 校验核心字段
    required_csv_fields = list(FIELD_MAPPING.keys())
    missing_fields = [f for f in required_csv_fields if f not in df.columns]
    if missing_fields:
        raise ValueError(f"❌ 缺失核心字段：{missing_fields}\n请检查CSV表头或FIELD_MAPPING配置")
    
    # 4. 字段重命名（标准化）
    df = df.rename(columns=FIELD_MAPPING)
    
    # 5. 数据类型转换
    numeric_fields = [
        "automation_risk", "openings_2024", "openings_2030", 
        "salary_usd", "gender_diversity", "remote_ratio", "experience_years"
    ]
    for field in numeric_fields:
        if field in df.columns:
            df[field] = pd.to_numeric(df[field], errors="coerce")  # 非数值转NaN
    
    # 6. 百分比转小数（0-1）
    percent_fields = ["automation_risk", "gender_diversity", "remote_ratio"]
    for field in percent_fields:
        if field in df.columns:
            df[field] = df[field] / 100  # 如80% → 0.8
    
    # 7. 补充衍生字段（基础）
    df["employment_change_rate"] = (df["openings_2030"] - df["openings_2024"]) / df["openings_2024"]
    df["employment_change_rate"] = df["employment_change_rate"].fillna(0)  # 空值填充为0
    
    # 8. 空值提示
    null_summary = df[numeric_fields].isnull().sum()
    if null_summary.sum() > 0:
        print(f"\n⚠️  数值字段空值统计：\n{null_summary[null_summary > 0]}")
    
    print(f"\n✅ 数据标准化完成，核心字段：{list(FIELD_MAPPING.values())}")
    return df

if __name__ == "__main__":
    # 测试加载
    try:
        df = load_data()
        print(df[["job_role", "industry", "country", "automation_risk", "openings_2024", "openings_2030"]])
    except Exception as e:
        print(f"❌ 加载失败：{e}")