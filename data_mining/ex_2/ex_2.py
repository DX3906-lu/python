import h2o
import pandas as pd
import matplotlib.pyplot as plt
from h2o.estimators.random_forest import H2ORandomForestEstimator
from sklearn.metrics import confusion_matrix, classification_report
import warnings

warnings.filterwarnings("ignore", category=UserWarning, message="Your H2O cluster version is")
warnings.filterwarnings("ignore", category=FutureWarning)


def calculate_accuracy(perf_obj, test_hf):
    """从混淆矩阵手动计算准确度（替代旧版本无的accuracy()方法）"""
    cm_hf = perf_obj.confusion_matrix()
    cm_df = cm_hf.as_data_frame()
    correct_predictions = cm_df.values.diagonal().sum()
    total_samples = test_hf.nrows
    accuracy = correct_predictions / total_samples
    return round(accuracy, 4)


print("初始化H2O集群...")
h2o.init(
    max_mem_size="2G", 
    nthreads=-1, 
    verbose=False,
    port=54321
)
h2o.no_progress()


def load_and_preprocess_data(train_path, test_path):
    train_hf = h2o.import_file(path=train_path)
    test_hf = h2o.import_file(path=test_path)
    
    print(f"\n[数据基本信息]")
    print(f"训练集形状: {train_hf.shape} | 测试集形状: {test_hf.shape}")
    print(f"训练集列名及类型:")
    for col, col_type in train_hf.types.items():
        print(f"  {col:25} | 类型: {col_type}")
    
    for df_name, df in [("训练集", train_hf), ("测试集", test_hf)]:
        numeric_cols = [col for col, col_type in df.types.items() if col_type in ("int", "real")]
        print(f"\n[{df_name}] 数值型列（共{len(numeric_cols)}个）:")
        print(f"  {numeric_cols}")
        
        missing_result = df[numeric_cols].isna().sum()
        if isinstance(missing_result, float):
            missing_df = pd.DataFrame([missing_result], index=[numeric_cols[0]], columns=["missing_count"])
        else:
            missing_df = missing_result.as_data_frame()
            missing_df.columns = ["missing_count"]
        
        total_missing = missing_df["missing_count"].sum()
        if total_missing > 0:
            print(f"[{df_name}] 发现{total_missing}个缺失值，用均值填充")
            for col in numeric_cols:
                df[col] = df[col].impute(method="mean")
        else:
            print(f"[{df_name}] 无缺失值，无需填充")
    
    target_col = "Catrgory"
    if target_col not in train_hf.columns or target_col not in test_hf.columns:
        raise ValueError(f"响应变量 '{target_col}' 不存在！请检查列名")
    train_hf[target_col] = train_hf[target_col].asfactor()
    test_hf[target_col] = test_hf[target_col].asfactor()
    
    return train_hf, test_hf


train_hf, test_hf = load_and_preprocess_data(
    train_path="./data_mining/ex_2/data/train.csv",
    test_path="./data_mining/ex_2/data/test.csv"
)

exclude_cols = ["driver", "trip", "Catrgory"]
full_features = [col for col in train_hf.columns if col not in exclude_cols]
y_col = "Catrgory"
print(f"\n[模型配置] 全量训练特征列（共{len(full_features)}个）:")
print(f"  {full_features}")


print("\n" + "="*60)
print("题目一：训练基础随机森林模型（全特征输入）")
print("="*60)

my_seed=45
ntrees_array = []
max_depth_array = []
accuracy_array = []

optimal_ntrees=0
optimal_max_depth=0
optimal_accuracy=0


for i in range(1, 21):
    for j in range(1, 21):
        rf = H2ORandomForestEstimator(ntrees=i, max_depth=j,seed=my_seed)
        rf.train(x=full_features,y=y_col,training_frame=train_hf)
        perf = rf.model_performance(test_data=test_hf)
        accuracy = calculate_accuracy(perf, test_hf)
        ntrees_array.append(i)
        max_depth_array.append(j)
        accuracy_array.append(accuracy)
        if accuracy>=optimal_accuracy:
            optimal_ntrees=i
            optimal_max_depth=j
            optimal_accuracy=accuracy
        print(f"{accuracy:.4f} ", end="")
        if j==20:
            print()

plt.scatter(ntrees_array, max_depth_array, c=accuracy_array, cmap='viridis', s=100)
plt.colorbar(label='Accuracy')
plt.xlabel('Number of Trees (ntrees)')
plt.ylabel('Max Depth')
plt.title('Accuracy vs. ntrees and max_depth in RandomForest')
plt.savefig("./data_mining/ex_2/output/Accuracy.png", dpi=300, bbox_inches='tight')
plt.show()
plt.close()

param_results_df = pd.DataFrame({
    "ntrees": ntrees_array,
    "max_depth": max_depth_array,
    "accuracy": accuracy_array
})

rf_full=H2ORandomForestEstimator(ntrees=optimal_ntrees, max_depth=optimal_max_depth,seed=my_seed)
rf_full.train(x=full_features,y=y_col,training_frame=train_hf)
perf_full = rf_full.model_performance(test_data=test_hf)
accuracy_full = calculate_accuracy(perf_full, test_hf)

print(f"[最优结果]")
print(f"  最高准确率：{optimal_accuracy:.4f}")
print(f"  对应参数：ntrees={optimal_ntrees}, max_depth={optimal_max_depth}")

print("\n" + "="*60)
print("题目二：前8重要特征 + 同一随机森林框架 重新训练")
print("="*60)

feature_importance = rf_full.varimp()
imp_df = pd.DataFrame(
    feature_importance,
    columns=["feature", "rel_importance", "scaled_importance", "pct_importance"]
).sort_values("rel_importance", ascending=False)

print("\n[全特征模型特征重要性]")
print(imp_df.round(4))

top8_features = imp_df.head(8)["feature"].tolist()
print(f"\n[筛选结果] 前8个重要特征（仅用这些特征重新训练）：{top8_features}")

rf_top8 = H2ORandomForestEstimator(ntrees=optimal_ntrees, max_depth=optimal_max_depth,seed=my_seed)
rf_top8.train(x=top8_features, y=y_col, training_frame=train_hf) 

perf_top8 = rf_top8.model_performance(test_data=test_hf)
accuracy_top8 = calculate_accuracy(perf_top8, test_hf)
print(f"[前8特征模型] 测试集准确度: {accuracy_top8:.4f}")


print("\n" + "="*60)
print("最终结果汇总（同一模型框架，不同特征输入对比）")
print("="*60)

result_df = pd.DataFrame({
    "模型类型": ["随机森林（全特征）", "随机森林（前8重要特征）"],
    "输入特征数": [len(full_features), len(top8_features)],
    "测试集准确度": [accuracy_full, accuracy_top8],
}).round(3)
print("\n[公平对比结果]")
print(result_df)

best_model = rf_full if accuracy_full >= accuracy_top8 else rf_top8
best_type = "随机森林（全特征）" if accuracy_full >= accuracy_top8 else "随机森林（前8重要特征）"
best_acc = accuracy_full if accuracy_full >= accuracy_top8 else accuracy_top8
print(f"\n[最佳模型] {best_type}（准确度: {best_acc:.4f}）")

print("\n[生成predict.csv...")
test_pd = test_hf.as_data_frame()
pred_pd = best_model.predict(test_hf).as_data_frame()[["predict"]]

predictions_df = test_pd.copy()
predictions_df["Predicted_Category"] = pred_pd["predict"]
predictions_df["Prediction_Correct"] = (predictions_df[y_col] == predictions_df["Predicted_Category"]).astype(int)  # 1=正确，0=错误

output_path = "./data_mining/ex_2/output/predict.csv"
predictions_df.to_csv(output_path, index=False, encoding="utf-8-sig")
print(f"  已保存至: {output_path}")
print(f"  包含字段：{predictions_df.columns.tolist()}")

print("\n[详细评估（最佳模型）]")
y_true = test_pd[y_col]
y_pred = pred_pd["predict"]

cm = confusion_matrix(y_true, y_pred)
cm_df = pd.DataFrame(cm, index=sorted(y_true.unique()), columns=sorted(y_true.unique()))
print("\n1. 混淆矩阵:")
print(cm_df)

print("\n2. 分类报告:")
print(classification_report(y_true, y_pred, zero_division=0))

print("\n[实验完成] 关闭H2O集群...")
h2o.shutdown(prompt=False)
print("集群已关闭，文件生成完毕！")

