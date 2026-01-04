# data_modules.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from config import DataConfig

class DataProcessor:
    def __init__(self, is_time_series=False):
        """
        初始化数据处理器
        :param is_time_series: 是否为时序模型（True=需构建时序序列，如LSTM；False=非时序，如决策树）
        """
        self.is_time_series = is_time_series
        self.scaler_X = MinMaxScaler(feature_range=(0, 1))  # 输入特征标量器
        self.scaler_y = MinMaxScaler(feature_range=(0, 1))  # 输出标签标量器（深度学习用）
        self.params = DataConfig.TARGET_PARAMS  # 目标参数名
        self.input_features = DataConfig.INPUT_FEATURES  # 输入特征名

    def load_data(self):
        """加载原始数据（支持Excel/Csv）"""
        path = DataConfig.DATA_PATH
        if path.endswith(".xlsx"):
            df = pd.read_excel(path, sheet_name="Sheet1")
        elif path.endswith(".csv"):
            df = pd.read_csv(path)
        else:
            raise ValueError("仅支持.xlsx或.csv格式数据")
        print(f"✅ 加载数据完成，共{len(df)}行 × {len(df.columns)}列")
        return df

    def preprocess(self, df):
        """
        统一数据预处理：计算补偿值 → 处理异常值 → 标准化 → 格式转换
        :return: X（输入特征）、y（输出标签：补偿值）、valid_df（预处理后的数据框）、元数据（标量器等）
        """
        # 1. 计算补偿值（补偿值 = 实际值 - 仿真值）
        for param in self.params:
            df[f"{param}_补偿值"] = df[f"{param}实际值"] - df[f"{param}仿真值"]

        """
        # 2. 去除极端异常值（3σ原则）
        for param in self.params:
            mean = df[f"{param}_补偿值"].mean()
            std = df[f"{param}_补偿值"].std()
            df = df[(df[f"{param}_补偿值"] >= mean - 3*std) & 
                    (df[f"{param}_补偿值"] <= mean + 3*std)]
        
        
        # 3. 特征工程：添加移动平均和差分特征（仅对输入特征进行处理）
        # 注意：避免使用未来数据，移动平均使用历史窗口
        window_sizes = [3, 5]  # 可以尝试不同的窗口大小
        for feature in self.input_features:
            for window in window_sizes:
                # 移动平均特征
                df[f'{feature}_MA_{window}'] = df[feature].rolling(window=window, min_periods=1).mean()
                # 移动标准差特征
                df[f'{feature}_MSTD_{window}'] = df[feature].rolling(window=window, min_periods=1).std()
            # 一阶差分
            df[f'{feature}_diff'] = df[feature].diff()
        # 填充因差分产生的NaN
        df = df.fillna(method='bfill').fillna(method='ffill')

        # 4. 更新输入特征列表
        enhanced_features = self.input_features.copy()
        for feature in self.input_features:
            for window in window_sizes:
                enhanced_features.append(f'{feature}_MA_{window}')
                enhanced_features.append(f'{feature}_MSTD_{window}')
            enhanced_features.append(f'{feature}_diff')
        """
        # 3. 提取输入特征X和输出标签y
        X = df[self.input_features].values
        y = df[[f"{param}_补偿值" for param in self.params]].values
        
        # 4. 处理缺失值
        X = np.nan_to_num(X, nan=0.0)
        y = np.nan_to_num(y, nan=0.0)
        
        # 5. 标准化（非时序模型仅标准化X，时序模型X/y均标准化）
        X_scaled = self.scaler_X.fit_transform(X)
        if self.is_time_series:
            y_scaled = self.scaler_y.fit_transform(y)
        else:
            y_scaled = y  # 传统ML模型无需标准化y

        # 6. 时序模型：构建时序序列（samples, time_step, features）
        if self.is_time_series:
            X_processed, y_processed = self._build_time_series(X_scaled, y_scaled)
            # 时序模型的有效数据框（跳过前time_step行）
            valid_df = df.iloc[DataConfig.TIME_STEP:].reset_index(drop=True)
        else:
            X_processed, y_processed = X_scaled, y_scaled
            valid_df = df.reset_index(drop=True)

        # 整理元数据（供后续反标准化、可视化使用）
        meta_data = {
            "scaler_X": self.scaler_X,
            "scaler_y": self.scaler_y if self.is_time_series else None,
            "params": self.params,
            "input_features": self.input_features,
            "time_step": DataConfig.TIME_STEP if self.is_time_series else 0,
            "y_true_val": y  # 新增！传入完整真实标签y，用于加权投票计算基模型权重
        }

        print(f"✅ 数据预处理完成：")
        print(f"  - 输入X形状：{X_processed.shape}")
        print(f"  - 输出y形状：{y_processed.shape}")
        print(f"  - 有效数据行数：{len(valid_df)}")
        return X_processed, y_processed, valid_df, meta_data

    def _build_time_series(self, X, y):
        """时序模型专用：构建时序序列（内部方法，用户无需修改）"""
        time_step = DataConfig.TIME_STEP
        X_seq, y_seq = [], []
        for i in range(time_step, len(X)):
            X_seq.append(X[i-time_step:i, :])
            y_seq.append(y[i, :])
        return np.array(X_seq), np.array(y_seq)