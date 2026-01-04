# visual_modules.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from config import SaveConfig  # 导入保存配置（路径、文件名）


# 提前创建结果保存目录（避免后续报错）
os.makedirs(SaveConfig.RESULT_DIR, exist_ok=True)


class Visualizer:  # 类名必须是Visualizer，大小写完全一致！
    """通用可视化器：生成误差表、时序图、箱线图，适配所有模型"""
    def __init__(self, model_name, params, valid_df, common_time=None):
        self.model_name = model_name
        self.params = params
        self.valid_df = valid_df
        self.common_time = common_time  # 时序模型专用
        self.error_dict = None  # 存储计算后的误差数据

        # 配置中文字体（避免乱码）
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        # 颜色配置（统一风格）
        self.colors = {"真实值": "#1f77b4", "预测值": "#2ca02c", "误差线": "#ff7f0e"}

    def calculate_error(self, all_preds):
        """
        计算核心误差：相对误差 = (仿真值 + 预测补偿值 - 实际值) / 实际值 * 100%
        :param all_preds: 模型预测的补偿值（形状：(样本数, 5)）
        :return: 误差字典（含逐点误差、最大/平均误差、预测值等）
        """
        self.error_dict = {}
        for param_idx, param in enumerate(self.params):
            # 从数据框中提取当前参数的仿真值、实际值
            sim_val = self.valid_df[f"{param}仿真值"].values  # 仿真值
            true_val = self.valid_df[f"{param}实际值"].values  # 实际值
            pred_comp = all_preds[:, param_idx]  # 模型预测的补偿值

            # 计算相对误差（避免除以0，加1e-8微小值）
            relative_error = (sim_val + pred_comp - true_val) / (true_val + 1e-8) * 100

            # 存储当前参数的所有误差相关数据
            self.error_dict[param] = {
                "逐点相对误差(%)": relative_error,
                "绝对相对误差(%)": np.abs(relative_error),
                "最大相对误差(%)": np.max(np.abs(relative_error)),
                "平均相对误差(%)": np.mean(np.abs(relative_error)),
                "预测补偿值": pred_comp,
                "预测实际值": sim_val + pred_comp,  # 仿真值+补偿值=预测实际值
                "真实实际值": true_val
            }
        print(f"✅ {self.model_name}误差计算完成（5个参数）")
        return self.error_dict

    def plot_error_summary(self):
        """1. 生成误差汇总表（Excel + 可视化表格）"""
        # 1.1 保存到Excel（支持多模型结果追加）
        summary_data = []
        for param, err_data in self.error_dict.items():
            summary_data.append({
                "模型名称": self.model_name,
                "参数名称": param,
                "最大相对误差(%)": round(err_data["最大相对误差(%)"], 6),
                "平均相对误差(%)": round(err_data["平均相对误差(%)"], 6)
            })
        summary_df = pd.DataFrame(summary_data)

        # 拼接或新建Excel文件
        excel_path = os.path.join(SaveConfig.RESULT_DIR, SaveConfig.ERROR_SUMMARY_FILE)
        if os.path.exists(excel_path):
            # 若文件已存在，追加数据（避免覆盖）
            existing_df = pd.read_excel(excel_path)
            summary_df = pd.concat([existing_df, summary_df], ignore_index=True)
        # 保存Excel
        summary_df.to_excel(excel_path, index=False)

        # 1.2 生成可视化表格（PNG图片）
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.axis('tight')  # 紧凑布局
        ax.axis('off')    # 隐藏坐标轴

        # 提取当前模型的误差数据用于表格
        current_model_data = summary_df[summary_df["模型名称"] == self.model_name]
        table_cell_text = [
            [row["参数名称"], f'{row["最大相对误差(%)"]:.6f}', f'{row["平均相对误差(%)"]:.6f}']
            for _, row in current_model_data.iterrows()
        ]

        # 创建表格
        table = ax.table(
            cellText=table_cell_text,
            colLabels=["参数名称", "最大相对误差(%)", "平均相对误差(%)"],
            cellLoc="center",  # 文字居中
            loc="center",      # 表格居中
            bbox=[0, 0, 1, 1]  # 表格占满整个子图
        )

        # 美化表格样式
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 2)  # 缩放表格（宽度×1.2，高度×2，避免文字拥挤）

        # 表头样式（深蓝色背景+白色文字）
        for col_idx in range(3):
            table[(0, col_idx)].set_facecolor("#4472C4")
            table[(0, col_idx)].set_text_props(weight="bold", color="white")

        # 表格内容行交替背景色（便于阅读）
        for row_idx in range(1, len(table_cell_text) + 1):
            bg_color = "#F8F9FA" if row_idx % 2 == 0 else "white"
            for col_idx in range(3):
                table[(row_idx, col_idx)].set_facecolor(bg_color)

        # 添加表格标题
        plt.title(f"{self.model_name}误差汇总表", fontsize=14, fontweight="bold", pad=20)
        
        # 保存表格图片
        table_path = os.path.join(SaveConfig.RESULT_DIR, f"{self.model_name}_误差汇总表.png")
        plt.savefig(table_path, bbox_inches="tight", dpi=300)
        plt.close()

        print(f"✅ 误差汇总表已保存：\n  - Excel：{excel_path}\n  - 图片：{table_path}")
        return summary_df

    def plot_time_series_comparison(self):
        """2. 生成时序预测对比图（时序模型专用）"""
        if self.common_time is None:
            print("⚠️ 非时序模型，跳过时序对比图绘制")
            return

        for param in self.params:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # 绘制真实值和预测值
            ax.plot(
                self.common_time, 
                self.error_dict[param]["真实实际值"], 
                label="真实值", 
                color=self.colors["真实值"], 
                linewidth=2
            )
            ax.plot(
                self.common_time, 
                self.error_dict[param]["预测实际值"], 
                label="预测值", 
                color=self.colors["预测值"], 
                linestyle='--', 
                linewidth=2
            )
            
            # 添加标题和标签
            ax.set_title(f"{self.model_name} - {param}实际值 vs 预测值", fontsize=14, fontweight="bold")
            ax.set_xlabel("时间", fontsize=12)
            ax.set_ylabel(f"{param}值", fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(alpha=0.3)
            
            # 自动调整x轴标签角度
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            
            # 保存图片
            plot_path = os.path.join(SaveConfig.RESULT_DIR, f"{self.model_name}_{param}_时序对比图.png")
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close()

        print(f"✅ 时序对比图生成完成（{len(self.params)}个参数）")

    def plot_error_distribution(self):
        """3. 生成误差分布箱线图"""
        # 准备误差数据
        error_data = []
        param_names = []
        for param in self.params:
            error_data.append(self.error_dict[param]["逐点相对误差(%)"])
            param_names.append(param)
        
        # 创建箱线图
        fig, ax = plt.subplots(figsize=(10, 6))
        bp = ax.boxplot(error_data, patch_artist=True, labels=param_names)
        
        # 美化箱线图样式
        for box in bp['boxes']:
            box.set(facecolor='#e0e0e0', edgecolor='#4472C4', linewidth=2)
        for whisker in bp['whiskers']:
            whisker.set(color='#4472C4', linewidth=2)
        for cap in bp['caps']:
            cap.set(color='#4472C4', linewidth=2)
        for median in bp['medians']:
            median.set(color='#ff7f0e', linewidth=2)
        for flier in bp['fliers']:
            flier.set(marker='o', color='#d62728', alpha=0.5)
        
        # 添加标题和标签
        ax.set_title(f"{self.model_name} - 各参数相对误差分布(%)", fontsize=14, fontweight="bold")
        ax.set_ylabel("相对误差(%)", fontsize=12)
        ax.grid(alpha=0.3, axis='y')
        plt.tight_layout()
        
        # 保存图片
        plot_path = os.path.join(SaveConfig.RESULT_DIR, f"{self.model_name}_误差分布箱线图.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"✅ 误差分布箱线图生成完成")

    def run_all_visualization(self, all_preds):
        self.calculate_error(all_preds)       # 1. 计算误差
        self.plot_error_summary()             # 2. 生成误差表（含图片）
        self.plot_time_series_comparison()    # 3. 生成时序对比图
        self.plot_error_distribution()        # 4. 生成误差分布箱线图
        print(f"\n🎉 {self.model_name}所有可视化完成！结果保存在：{SaveConfig.RESULT_DIR}")