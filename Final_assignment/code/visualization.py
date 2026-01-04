import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
from config import VISUALIZATION_PATH, PLOT_STYLE, PLOT_FIGSIZE, PLOT_FONT

# 初始化可视化配置
plt.style.use(PLOT_STYLE)
plt.rcParams["font.family"] = PLOT_FONT["family"]
plt.rcParams["font.size"] = PLOT_FONT["size"]
plt.rcParams["axes.unicode_minus"] = False  # 负号显示

def create_visualization_dir():
    """创建可视化目录（不存在则创建）"""
    if not os.path.exists(VISUALIZATION_PATH):
        os.makedirs(VISUALIZATION_PATH)
    return VISUALIZATION_PATH

def plot_risk_distribution(df: pd.DataFrame):
    """绘制自动化风险分布（按行业）"""
    save_path = os.path.join(create_visualization_dir(), "automation_risk_by_industry.png")
    plt.figure(figsize=PLOT_FIGSIZE)
    sns.boxplot(data=df, x="industry", y="automation_risk")
    plt.title("各行业自动化风险分布（2024）", fontsize=12)
    plt.xlabel("行业")
    plt.ylabel("自动化风险（0-1）")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"📈 自动化风险分布图已保存：{save_path}")

def plot_employment_change(df: pd.DataFrame):
    """绘制2024-2030就业岗位变化率（Top10岗位）"""
    save_path = os.path.join(create_visualization_dir(), "employment_change_top10.png")
    # 取变化率绝对值Top10的岗位
    top10_jobs = df.sort_values(by="employment_change_rate", key=abs, ascending=False).head(10)
    plt.figure(figsize=PLOT_FIGSIZE)
    sns.barplot(data=top10_jobs, x="job_role", y="employment_change_rate")
    plt.title("2024-2030岗位变化率Top10", fontsize=12)
    plt.xlabel("岗位")
    plt.ylabel("就业变化率（正增长/负减少）")
    plt.xticks(rotation=45, ha="right")
    plt.axhline(y=0, color="red", linestyle="--", alpha=0.5)  # 零轴参考线
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"📈 就业变化率图已保存：{save_path}")

def plot_high_risk_jobs(df: pd.DataFrame):
    """绘制高风险岗位Top10"""
    save_path = os.path.join(create_visualization_dir(), "high_risk_jobs_top10.png")
    high_risk_df = df[df["risk_level"] == "high"].sort_values(by="automation_risk", ascending=False).head(10)
    plt.figure(figsize=PLOT_FIGSIZE)
    sns.barplot(data=high_risk_df, x="job_role", y="automation_risk")
    plt.title("高风险岗位Top10（自动化风险）", fontsize=12)
    plt.xlabel("岗位")
    plt.ylabel("自动化风险（0-1）")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"📈 高风险岗位图已保存：{save_path}")

def plot_y_by_x(label,x,y,df: pd.DataFrame):
    save_path = os.path.join(create_visualization_dir(), f"{label}.png")
    industry_impact = df.groupby(x)[y].mean().sort_values(ascending=False)
    plt.figure(figsize=PLOT_FIGSIZE)
    sns.barplot(x=industry_impact.index, y=industry_impact.values)
    plt.title(f"{label}",fontsize=12)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"📈 {label}图已保存：{save_path}")

def plot_future_employment_forecast(df: pd.DataFrame):
    """绘制未来就业预测对比图"""
    save_path = os.path.join(create_visualization_dir(), "future_employment_forecast.png")
    
    # 选择有代表性的岗位
    Representative_Position = df.groupby("job_role")["predicted_employment_change"].std().sort_values().head(10).index
    forecast_df = df[df["job_role"].isin(Representative_Position)][["job_role", "openings_2024", "predicted_openings_2030"]]
    forecast_df = forecast_df.melt(id_vars="job_role", var_name="year", value_name="openings")
    forecast_df["year"] = forecast_df["year"].map({
        "openings_2024": 2024, 
        "predicted_openings_2030": 2030
    })
    
    plt.figure(figsize=PLOT_FIGSIZE)
    sns.barplot(data=forecast_df, x="job_role", y="openings", hue="year")
    plt.title("2024与2030年岗位数量对比预测", fontsize=12)
    plt.xlabel("岗位")
    plt.ylabel("岗位数量")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"📈 未来就业预测图已保存：{save_path}")

def plot_ai_impact_by_region(df: pd.DataFrame):
    """绘制不同地区的AI影响热力图"""
    save_path = os.path.join(create_visualization_dir(), "ai_impact_by_region.png")
    
    # 准备数据
    impact_data = df.groupby(["country", "industry"]).agg({
        "automation_risk": "mean",
        "predicted_employment_change": "mean"
    }).reset_index()
    
    # 转换为透视表
    pivot_risk = impact_data.pivot(index="country", columns="industry", values="automation_risk")
    pivot_change = impact_data.pivot(index="country", columns="industry", values="predicted_employment_change")
    
    # 创建子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # 热图1：自动化风险
    sns.heatmap(pivot_risk, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax1)
    ax1.set_title("各地区各行业自动化风险均值", fontsize=12)
    
    # 热图2：就业变化预测
    sns.heatmap(pivot_change, annot=True, fmt=".2f", cmap="RdBu", ax=ax2)
    ax2.set_title("各地区各行业就业变化率预测", fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"📈 地区AI影响热力图已保存：{save_path}")

def plot_risk_vs_salary(df: pd.DataFrame):
    """绘制风险与薪资关系图"""
    save_path = os.path.join(create_visualization_dir(), "risk_vs_salary.png")
    
    plt.figure(figsize=PLOT_FIGSIZE)
    sns.scatterplot(
        data=df, 
        x="automation_risk", 
        y="salary_usd",
        hue="predicted_risk_level",
        size="predicted_employment_change",
        sizes=(50, 200),
        alpha=0.7
    )
    plt.title("自动化风险与薪资关系", fontsize=12)
    plt.xlabel("自动化风险（0-1）")
    plt.ylabel("薪资（USD）")
    plt.axvline(x=0.7, color="red", linestyle="--", alpha=0.5, label="高风险阈值")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"📈 风险与薪资关系图已保存：{save_path}")
