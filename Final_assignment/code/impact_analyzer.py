import pandas as pd
import os
from config import OUTPUT_PATH, AI_IMPACT_WEIGHTS, TARGET_YEAR
from visualization import (plot_y_by_x, plot_risk_distribution, 
                           plot_employment_change, plot_high_risk_jobs,
                           plot_future_employment_forecast, 
                           plot_ai_impact_by_region, plot_risk_vs_salary)
from predictor import (train_employment_change_model, train_automation_risk_model,
                       predict_future_employment, identify_high_risk_jobs)
from data_preprocessor import preprocess_for_prediction

def analyze_ai_impact_by_industry(df: pd.DataFrame) -> pd.DataFrame:
    """分析AI对不同行业的影响"""
    impact_metrics = df.groupby("industry").agg({
        "automation_risk": "mean",
        "employment_change_rate": "mean",
        "ai_impact_level": lambda x: x.value_counts(normalize=True).get("High", 0),
        "salary_usd": "mean"
    }).rename(columns={
        "ai_impact_level": "high_ai_impact_ratio",
        "automation_risk": "avg_automation_risk",
        "employment_change_rate": "avg_employment_change",
        "salary_usd": "avg_salary"
    }).sort_values(by="avg_automation_risk", ascending=False)
    
    # 保存分析结果
    output_path = os.path.join(OUTPUT_PATH, "ai_impact_by_industry.csv")
    impact_metrics.to_csv(output_path)
    print(f"📄 行业AI影响分析已保存：{output_path}")
    
    return impact_metrics

def analyze_ai_impact_by_country(df: pd.DataFrame) -> pd.DataFrame:
    """分析AI对不同国家的影响"""
    country_impact = df.groupby("country").agg({
        "automation_risk": "mean",
        "employment_change_rate": "mean",
        "remote_ratio": "mean",
        "job_role": "count"
    }).rename(columns={
        "automation_risk": "avg_automation_risk",
        "employment_change_rate": "avg_employment_change",
        "remote_ratio": "avg_remote_ratio",
        "job_role": "total_jobs"
    }).sort_values(by="avg_automation_risk", ascending=False)
    
    # 保存分析结果
    output_path = os.path.join(OUTPUT_PATH, "ai_impact_by_country.csv")
    country_impact.to_csv(output_path)
    print(f"📄 国家AI影响分析已保存：{output_path}")
    
    return country_impact

def generate_high_risk_jobs_report(df: pd.DataFrame) -> pd.DataFrame:
    """生成高风险工作报告"""
    high_risk_jobs = df[df["predicted_risk_level"] == "high"].groupby(["industry", "job_role"]).agg({
        "high_risk_probability": "mean",
        "automation_risk": "mean",
        "predicted_employment_change": "mean"
    }).sort_values(by="high_risk_probability", ascending=False).head(20)
    
    # 保存报告
    output_path = os.path.join(OUTPUT_PATH, "high_risk_jobs_report.csv")
    high_risk_jobs.to_csv(output_path)
    print(f"📄 高风险工作报告已保存：{output_path}")
    
    return high_risk_jobs

if __name__ == "__main__":
    from data_loader import load_data
    
    try:
        # 加载并预处理数据
        print("1. 加载并预处理数据...")
        df_raw = load_data()
        df_processed = preprocess_for_prediction(df_raw)
        
        # 训练预测模型
        print("\n2. 训练预测模型...")
        employment_model, employment_metrics = train_employment_change_model(df_processed)
        print(f"就业变化预测模型性能: RMSE = {employment_metrics['rmse']:.4f}")
        print("特征重要性:")
        print(employment_metrics["feature_importance"])
        
        risk_model, risk_metrics = train_automation_risk_model(df_processed)
        print(f"\n自动化风险预测模型性能: ROC-AUC = {risk_metrics['roc_auc']:.4f}")
        print("分类报告:")
        print(risk_metrics["classification_report"])
        
        # 进行预测
        print("\n3. 进行未来就业市场预测...")
        df_with_predictions = predict_future_employment(df_processed, employment_model)
        df_with_predictions = identify_high_risk_jobs(df_with_predictions, risk_model)
        
        # 生成分析报告
        print("\n4. 生成分析报告...")
        industry_impact = analyze_ai_impact_by_industry(df_with_predictions)
        country_impact = analyze_ai_impact_by_country(df_with_predictions)
        high_risk_report = generate_high_risk_jobs_report(df_with_predictions)
        
        # 生成可视化结果
        print("\n5. 生成可视化结果...")
        plot_risk_distribution(df_with_predictions)
        plot_employment_change(df_with_predictions)
        plot_high_risk_jobs(df_with_predictions)
        
        # 新增预测相关可视化
        plot_future_employment_forecast(df_with_predictions)
        plot_ai_impact_by_region(df_with_predictions)
        plot_risk_vs_salary(df_with_predictions)
        
        # 按不同维度分析就业变化率
        value = "predicted_employment_change"
        for feature in ["industry", "country", "education", "ai_impact_level"]:
            plot_y_by_x(f"{value}_by_{feature}", feature, value, df_with_predictions)
            print(f"\n📊 {TARGET_YEAR}年就业变化预测({feature})：")
            print(df_with_predictions.groupby(feature)[value].mean().sort_values(ascending=False))
            
        print("\n✅ 所有分析完成！结果已保存至输出目录。")
        
    except Exception as e:
        print(f"❌ 分析失败：{e}")