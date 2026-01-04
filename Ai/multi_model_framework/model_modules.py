import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor  # 新增GBR
from sklearn.svm import SVR  
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D  # 新增1D卷积（TCN用）
from config import TrainConfig


# -------------------------- 模型基类：所有模型必须继承此类 --------------------------
class BaseModel:
    def __init__(self, model_name, is_time_series=False):
        self.model_name = model_name  # 模型名称（用于保存结果和投票标识）
        self.is_time_series = is_time_series  # 是否为时序模型
        self.models = None  # 存储训练后的模型（多输出任务：每个参数1个模型）
        self.meta_data = None  # 存储数据元数据（标量器、参数名等）

    def train(self, X, y, meta_data):
        raise NotImplementedError(f"{self.model_name}需实现train方法！")

    def predict(self, X):
        raise NotImplementedError(f"{self.model_name}需实现predict方法！")

    def _inverse_scale_y(self, y_scaled, meta_data):
        """辅助方法：反标准化y（时序模型用）"""
        if meta_data["scaler_y"] is None:
            return y_scaled
        return meta_data["scaler_y"].inverse_transform(y_scaled)


# -------------------------- 一、传统机器学习模型（非时序） --------------------------
# 1. 决策树模型（原有）
class DecisionTreeModel(BaseModel):
    def __init__(self):
        super().__init__(model_name="决策树", is_time_series=False)
        self.params = TrainConfig.ML_PARAMS

    def train(self, X, y, meta_data):
        self.meta_data = meta_data
        n_params = len(meta_data["params"])
        self.models = []

        for i in range(n_params):
            dt = DecisionTreeRegressor(
                max_depth=self.params["max_depth"],
                min_samples_split=self.params["min_samples_split"],
                random_state=TrainConfig.RANDOM_SEED
            )
            dt.fit(X, y[:, i])
            self.models.append(dt)
        print(f"✅ {self.model_name}训练完成（{n_params}个参数模型）")
        return self.models

    def predict(self, X):
        preds = [model.predict(X).reshape(-1, 1) for model in self.models]
        return np.hstack(preds)


# 2. 随机森林模型（原有）
class RandomForestModel(BaseModel):
    def __init__(self):
        super().__init__(model_name="随机森林", is_time_series=False)
        self.params = TrainConfig.ML_PARAMS

    def train(self, X, y, meta_data):
        self.meta_data = meta_data
        n_params = len(meta_data["params"])
        self.models = []

        for i in range(n_params):
            rf = RandomForestRegressor(
                n_estimators=self.params["n_estimators"],
                max_depth=self.params["max_depth"],
                min_samples_split=self.params["min_samples_split"],
                random_state=TrainConfig.RANDOM_SEED,
                n_jobs=-1
            )
            rf.fit(X, y[:, i])
            self.models.append(rf)
        print(f"✅ {self.model_name}训练完成（{n_params}个参数模型）")
        return self.models

    def predict(self, X):
        preds = [model.predict(X).reshape(-1, 1) for model in self.models]
        return np.hstack(preds)


# 3. 新增：支持向量机（SVM）模型（非时序，适合小样本高精度场景）
class SVMModel(BaseModel):
    def __init__(self):
        super().__init__(model_name="SVM", is_time_series=False)
        self.params = {
            "kernel": "rbf",  # 径向基核（适合非线性关系）
            "C": 1.0,         # 正则化强度
            "gamma": "scale", # 核函数系数
            "epsilon": 0.1    # 回归任务的误差容忍度
        }

    def train(self, X, y, meta_data):
        self.meta_data = meta_data
        n_params = len(meta_data["params"])
        self.models = []

        for i in range(n_params):
            svm = SVR(
                kernel=self.params["kernel"],
                C=self.params["C"],
                gamma=self.params["gamma"],
                epsilon=self.params["epsilon"]
            )
            svm.fit(X, y[:, i])
            self.models.append(svm)
        print(f"✅ {self.model_name}训练完成（{n_params}个参数模型）")
        return self.models

    def predict(self, X):
        preds = [model.predict(X).reshape(-1, 1) for model in self.models]
        return np.hstack(preds)


# 4. 新增：梯度提升回归（GBR）模型（非时序，比随机森林更精准）
# 4. 梯度提升回归（GBR）模型（非时序，比随机森林更精准）
class GBRModel(BaseModel):
    def __init__(self):
        super().__init__(model_name="梯度提升", is_time_series=False)
        self.params = TrainConfig.ML_PARAMS  # 复用配置文件中的ML参数

    def train(self, X, y, meta_data):
        self.meta_data = meta_data
        n_params = len(meta_data["params"])  # 5个参数（P3/P7/T3/T7/转速）
        self.models = []

        # 为每个参数独立训练1个GBR模型
        for i in range(n_params):
            gbr = GradientBoostingRegressor(
                n_estimators=self.params["n_estimators"],  # 树的数量（从配置文件读取）
                max_depth=self.params["max_depth"],        # 树深度（从配置文件读取）
                min_samples_split=self.params["min_samples_split"],  # 最小分裂样本数
                random_state=TrainConfig.RANDOM_SEED,      # 固定随机种子
                learning_rate=0.05  # GBR专用参数：控制每棵树的贡献度
            )
            gbr.fit(X, y[:, i])  # 单参数训练（适配多输出任务）
            self.models.append(gbr)
        
        print(f"✅ {self.model_name}训练完成（{n_params}个参数模型）")
        return self.models

    def predict(self, X):
        # 遍历每个参数模型预测，合并结果
        preds_list = []
        for model in self.models:
            pred = model.predict(X)  # 单参数预测
            preds_list.append(pred.reshape(-1, 1))  # 转为列向量
        return np.hstack(preds_list)  # 合并为(样本数, 5)的矩阵


# -------------------------- 二、深度学习模型（时序） --------------------------
# 1. LSTM模型（原有）
class LSTMModel(BaseModel):
    def __init__(self):
        super().__init__(model_name="LSTM", is_time_series=True)
        self.params = TrainConfig.DL_PARAMS

    def train(self, X, y, meta_data):
        self.meta_data = meta_data
        input_shape = (X.shape[1], X.shape[2])
        n_params = len(meta_data["params"])

        self.models = Sequential(name=self.model_name)
        self.models.add(LSTM(
            units=self.params["lstm_units"][0],
            return_sequences=True,
            input_shape=input_shape,
            kernel_regularizer=tf.keras.regularizers.l2(0.001)
        ))
        self.models.add(Dropout(0.2))
        self.models.add(LSTM(
            units=self.params["lstm_units"][1],
            return_sequences=False
        ))
        self.models.add(Dropout(0.2))
        self.models.add(Dense(units=n_params, activation="linear"))

        self.models.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.params["learning_rate"]),
            loss="mse",
            metrics=["mae"]
        )

        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        )
        self.models.fit(
            X, y,
            epochs=self.params["epochs"],
            batch_size=self.params["batch_size"],
            validation_split=0.2,
            callbacks=[early_stop],
            verbose=1
        )
        print(f"✅ {self.model_name}训练完成（多输出模型）")
        return self.models

    def predict(self, X):
        y_scaled_pred = self.models.predict(X, verbose=0)
        return self._inverse_scale_y(y_scaled_pred, self.meta_data)




# -------------------------- 三、核心新增：投票集成模型（支持自由选择基模型） --------------------------
class VotingEnsembleModel(BaseModel):
    def __init__(self, base_model_instances, voting_type="weighted"):
        """
        投票集成模型（支持多模型融合）
        :param base_model_instances: 基模型实例列表（必须≥2个，如[DecisionTreeModel(), RandomForestModel()]）
        :param voting_type: 投票类型："weighted"（MAE加权，MAE越小权重越大）或 "uniform"（等权重）
        """
        # 验证输入合法性
        if len(base_model_instances) < 2:
            raise ValueError("投票集成至少需要2个基模型！")
        if not all(isinstance(model, BaseModel) for model in base_model_instances):
            raise TypeError("所有基模型必须继承BaseModel类！")
        if voting_type not in ["weighted", "uniform"]:
            raise ValueError("voting_type只能是'weighted'或'uniform'！")

        # 初始化集成模型信息
        self.base_models = base_model_instances  # 基模型实例列表
        self.voting_type = voting_type          # 投票类型
        self.base_model_names = [model.model_name for model in self.base_models]  # 基模型名称
        super().__init__(
            model_name=f"投票集成（{'+'.join(self.base_model_names)}）",
            is_time_series=self.base_models[0].is_time_series  # 所有基模型必须同类型（时序/非时序）
        )

        # 验证所有基模型时序类型一致
        if not all(model.is_time_series == self.is_time_series for model in self.base_models):
            raise ValueError("所有基模型必须同为时序模型或非时序模型！")

    def train(self, X, y, meta_data):
        """训练所有基模型，并存储训练后的模型"""
        self.meta_data = meta_data
        print(f"\n=== 开始训练投票集成的基模型（共{len(self.base_models)}个） ===")
        
        # 训练每个基模型
        self.trained_base_models = []  # 存储训练后的基模型
        for model in self.base_models:
            print(f"\n--- 训练基模型：{model.model_name} ---")
            trained_model = model.train(X, y, meta_data)
            self.trained_base_models.append(model)  # 存储训练后的完整模型实例（含predict方法）
        
        print(f"\n✅ 所有基模型训练完成！集成模型名称：{self.model_name}")
        return self.trained_base_models

    def predict(self, X):
        """
        投票预测：
        - 等权重（uniform）：所有基模型预测结果取平均
        - 加权（weighted）：按基模型在验证集的MAE倒数加权（MAE越小权重越大）
        """
        # 1. 获取所有基模型的预测结果
        base_preds = []  # 存储每个基模型的预测结果（形状：[n_models, n_samples, n_params]）
        for model in self.trained_base_models:
            pred = model.predict(X)
            base_preds.append(pred)
        base_preds = np.array(base_preds)  # 转为数组：(n_models, n_samples, n_params)

        # 2. 计算投票权重
        if self.voting_type == "uniform":
            # 等权重：每个模型权重=1/模型数量
            weights = np.ones(len(self.base_models)) / len(self.base_models)
        else:  # weighted：按MAE加权（需先计算每个基模型的MAE）
            weights = self._calculate_base_model_weights(X, base_preds)

        # 3. 按权重融合预测结果（权重广播到每个样本和参数）
        weights = weights.reshape(-1, 1, 1)  # 调整权重形状：(n_models, 1, 1)
        ensemble_pred = np.sum(base_preds * weights, axis=0)  # 加权求和：(n_samples, n_params)

        print(f"✅ 投票集成预测完成（权重：{dict(zip(self.base_model_names, weights.squeeze()))}）")
        return ensemble_pred

    # model_modules.py 的 _calculate_base_model_weights 方法（完全替换）
    def _calculate_base_model_weights(self, X_val, base_preds):
        """辅助方法：计算基模型权重（使用当前折的验证集标签，确保形状匹配）"""
        # 1. 从 meta_data 中获取当前折的验证集真实标签（而非全量标签）
        if "current_fold_y_true" not in self.meta_data:
            raise KeyError("请在 train_modules.py 的 cross_validate 中添加 'current_fold_y_true' 到 meta_data！")
        y_val_true = self.meta_data["current_fold_y_true"]  # 当前折验证集真实标签

        # 2. 验证形状匹配（基模型预测结果与验证集标签必须同形状）
        base_pred_shape = base_preds[0].shape  # 单个基模型的预测形状（n_val_samples, n_params）
        y_true_shape = y_val_true.shape        # 验证集真实标签形状（n_val_samples, n_params）
        if base_pred_shape != y_true_shape:
            raise ValueError(
                f"基模型预测形状 {base_pred_shape} 与验证集标签形状 {y_true_shape} 不匹配！"
                f"请检查基模型预测逻辑（X_val形状：{X_val.shape}）"
            )

        # 3. 计算每个基模型在当前折的MAE（按参数平均后作为模型整体MAE）
        base_maes = []
        for pred in base_preds:
            # 计算每个参数的MAE（形状：(n_params,)）
            param_mae = np.mean(np.abs(pred - y_val_true), axis=0)
            # 计算模型整体MAE（所有参数的平均）
            model_mae = np.mean(param_mae)
            base_maes.append(model_mae)
            print(f"  基模型 {self.base_model_names[len(base_maes)-1]} 的MAE：{model_mae:.6f}")

        # 4. MAE倒数归一化（MAE越小，权重越大）
        weights = 1 / (np.array(base_maes) + 1e-8)  # 避免MAE=0导致除以0
        weights = weights / weights.sum()  # 归一化：权重和为1
        return weights

    def _inverse_scale_y(self, y_scaled, meta_data):
        """复用基模型的反标准化逻辑（所有基模型反标准化一致）"""
        return self.base_models[0]._inverse_scale_y(y_scaled, meta_data)