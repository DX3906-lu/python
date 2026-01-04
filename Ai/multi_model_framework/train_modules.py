# train_modules.py
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from config import TrainConfig  # 必须导入配置文件，获取交叉验证参数


class Trainer:  # 类名必须是Trainer，大小写完全一致！
    """通用训练验证器：支持所有继承BaseModel的模型（决策树、LSTM等）"""
    def __init__(self, model):
        """
        初始化训练器
        :param model: 模型实例（需继承model_modules中的BaseModel类）
        """
        self.model = model  # 传入的模型对象（如DecisionTreeModel实例）
        self.all_preds = None  # 存储所有样本的预测结果（原始尺度）
        self.all_true = None   # 存储所有样本的真实标签（原始尺度）
        
        # 初始化五折交叉验证（参数来自config.py，无需修改）
        self.kf = KFold(
            n_splits=TrainConfig.N_SPLITS,
            shuffle=True,
            random_state=TrainConfig.RANDOM_SEED
        )

    # train_modules.py 完整的 cross_validate 方法（确保与以下代码一致）
    def cross_validate(self, X, y, meta_data):
        """五折交叉验证核心逻辑"""
        self.all_true = self._inverse_scale_y(y, meta_data)
        self.all_preds = np.zeros_like(self.all_true)

        for fold_idx, (train_indices, val_indices) in enumerate(self.kf.split(X)):
            print(f"\n=== 第{fold_idx+1}/{TrainConfig.N_SPLITS}折训练 ===")
            
            # 1. 划分当前折的训练/验证数据
            X_train, X_val = X[train_indices], X[val_indices]
            y_train, y_val = y[train_indices], y[val_indices]

            # 2. 计算当前折的验证集真实标签（反标准化后）
            y_val_true = self._inverse_scale_y(y_val, meta_data)  # 这一行必须有！
            
            # 3. 关键：将当前折的验证集真实标签添加到 meta_data
            meta_data["current_fold_y_true"] = y_val_true  # 这一行是修复的核心！

            # 4. 训练模型（投票模型会自动训练所有基模型）
            trained_model = self.model.train(X_train, y_train, meta_data)

            # 5. 预测验证集（此时 meta_data 中已包含 current_fold_y_true）
            y_val_pred = self.model.predict(X_val)

            # 6. 保存当前折的结果并计算MAE
            self.all_preds[val_indices] = y_val_pred
            fold_mae = mean_absolute_error(y_val_true, y_val_pred)
            print(f"第{fold_idx+1}折验证MAE：{fold_mae:.6f}")

        # 计算整体MAE
        overall_mae = mean_absolute_error(self.all_true, self.all_preds)
        print(f"\n✅ 五折交叉验证完成！整体平均MAE：{overall_mae:.6f}")
        return self.all_preds, self.all_true

    def _inverse_scale_y(self, y, meta_data):
        """辅助方法：调用模型的反标准化逻辑（统一接口，适配时序/非时序模型）"""
        return self.model._inverse_scale_y(y, meta_data)