# config.py
# 数据配置：路径、列名、参数名（必须与你的数据列名完全一致）
class DataConfig:
    # 1. 数据文件路径（替换为你的数据文件实际路径！）
    # 示例：如果数据文件在multi_model_framework文件夹内，直接写文件名；否则写完整路径
    DATA_PATH = "数据 - 副本.xlsx"  # 重点：确认这个路径正确！
    
    # 2. 输入特征列名（必须与你的Excel/Csv中的列名完全一致）
    INPUT_FEATURES = ["燃油", "燃油变化率"]  # 检查数据中是否有这两列，无则删除多余的
    
    # 3. 预测目标参数名（5个参数，必须与数据中的“实际值/仿真值”列名匹配）
    TARGET_PARAMS = ["P3", "P7", "T3", "T7", "转速"]  # 例如数据中需有“P3实际值”“P3仿真值”
    
    # 4. 时序模型专用：时间步长（非时序模型忽略，如决策树）
    TIME_STEP = 10  # LSTM等时序模型的窗口大小，无需修改


# 训练配置：交叉验证、模型参数（无需修改，除非需调参）
class TrainConfig:
    # 五折交叉验证（符合作业要求）
    N_SPLITS = 5
    RANDOM_SEED = 55

    ML_PARAMS = {
        "max_depth": 20000,
        "min_samples_split": 3000,
        "n_estimators": 1000
    }
    
    DL_PARAMS = {
        "epochs": 50,
        "batch_size": 64,
        "learning_rate": 0.001,
        "lstm_units": [64, 32]
    }


# 结果保存配置：无需修改
class SaveConfig:
    RESULT_DIR = "model_results/"  # 结果保存目录（自动创建）
    ERROR_SUMMARY_FILE = "误差汇总表.xlsx"
    PLOT_PREFIX = "模型对比图_"