# 路径配置
DATA_PATH = "./Final_assignment/data/ai_job_trends_dataset.csv"  # 数据文件路径
OUTPUT_PATH = "./Final_assignment/output/"      # 结果输出路径
VISUALIZATION_PATH = "./Final_assignment/output/visualizations/"  # 可视化结果路径

# 字段映射：CSV实际字段名 → 代码内部标准化字段名
FIELD_MAPPING = {
    "Job Title": "job_role",
    "Industry": "industry",
    "Location": "country",
    "Automation Risk (%)": "automation_risk",
    "Job Openings (2024)": "openings_2024",
    "Projected Openings (2030)": "openings_2030",
    "Median Salary (USD)": "salary_usd",
    "Required Education": "education",
    "Gender Diversity (%)": "gender_diversity",
    "Remote Work Ratio (%)": "remote_ratio",
    "AI Impact Level": "ai_impact_level",
    "Experience Required (Years)": "experience_years",
    "Job Status": "job_status"
}

# 核心参数配置
TARGET_YEAR = 2030  # 预测目标年份
RISK_THRESHOLDS = {  # 自动化风险等级划分（0-1）
    "low": 0.3,
    "medium": 0.7,
    "high": 1.0
}
AI_IMPACT_WEIGHTS = {  # AI影响指数权重
    "automation_risk": 0.5,
    "ai_impact_level": 0.5
}

# 可视化配置
PLOT_STYLE = "seaborn-v0_8-whitegrid"
PLOT_FIGSIZE = (12, 6)
PLOT_FONT = {"family": "SimHei", "size": 10}