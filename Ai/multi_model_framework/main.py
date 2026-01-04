# main.py
import numpy as np
from data_modules import DataProcessor
from model_modules import (
    DecisionTreeModel, RandomForestModel, GBRModel,SVMModel,LSTMModel, VotingEnsembleModel
)
from train_modules import Trainer
from visual_modules import Visualizer
from config import DataConfig,SaveConfig

def run_experiment(model):
    """
    单模型实验流程：数据加载→预处理→训练→可视化→结果保存
    """

    data_processor = DataProcessor(is_time_series=model.is_time_series)
    raw_df = data_processor.load_data()
    X, y, valid_df, meta_data = data_processor.preprocess(raw_df)
    trainer = Trainer(model=model)
    all_preds, all_true = trainer.cross_validate(X, y, meta_data)
    common_time = valid_df["时间"].values if model.is_time_series else None
    
    visualizer = Visualizer(
        model_name=model.model_name,    
        params=meta_data["params"],     
        valid_df=valid_df,              
        common_time=common_time         
    )
    
    visualizer.run_all_visualization(all_preds)

    print(f"\n🎉 {model.model_name}实验全流程完成！")
    print(f"📁 所有结果保存在：{SaveConfig.RESULT_DIR}")

if __name__ == "__main__":
    base_models = [
        DecisionTreeModel(),    
        RandomForestModel(),         
    ]
    
    current_model = VotingEnsembleModel(
        base_model_instances=base_models,
        voting_type="weighted" 
    )
    current_model=LSTMModel()
    run_experiment(current_model)