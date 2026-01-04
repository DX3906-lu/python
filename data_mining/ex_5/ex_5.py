import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

data = pd.read_csv("./data_mining/ex_5/data/trip.csv") 
print("原始数据形状：", data.shape)
print("原始数据前5行：")
print(data.head())

feature_cols = data.columns.drop(['driver'])
data_features = data[feature_cols].copy()
print("\n参与聚类的特征列：", feature_cols.tolist())

scaler = StandardScaler()
data_features_scaled = scaler.fit_transform(data_features)
data_features_scaled = pd.DataFrame(data_features_scaled, columns=feature_cols)
print("\n标准化后特征前5行：")
print(data_features_scaled.head())

param_grid = {
    'init': ['k-means++', 'random'], 
    'n_init': [10, 20, 30],          
    'max_iter': [300, 500]            
}

best_inertia = float('inf')
best_kmeans = None
for init in param_grid['init']:
    for n_init in param_grid['n_init']:
        for max_iter in param_grid['max_iter']:
            kmeans = KMeans(
                n_clusters=3,        
                init=init,
                n_init=n_init,
                max_iter=max_iter,
                random_state=42,
                verbose=0
            )
            kmeans.fit(data_features_scaled)

            if kmeans.inertia_ < best_inertia:
                best_inertia = kmeans.inertia_
                best_kmeans = kmeans

print(f"\n最优K-means模型参数：")
print(f"初始化方式：{best_kmeans.init}")
print(f"初始化次数：{best_kmeans.n_init}")
print(f"最大迭代次数：{best_kmeans.max_iter}")
print(f"最优惯性值：{best_inertia:.4f}")


cluster_counts = pd.Series(best_kmeans.labels_, name='jllable').value_counts().sort_index()
print("\n各聚类类别的数目：")
for cluster_id, count in cluster_counts.items():
    print(f"类别{cluster_id}（{['普通驾驶', '激进驾驶', '超冷静驾驶'][cluster_id]}）：{count}条")


centroids_scaled = best_kmeans.cluster_centers_ 
centroids_original = scaler.inverse_transform(centroids_scaled)  
centroids_df = pd.DataFrame(
    centroids_original,
    columns=feature_cols,
    index=['普通驾驶', '激进驾驶', '超冷静驾驶']
)
print("\n聚类中心（原始尺度，每行代表一类驾驶的特征均值）：")
print(centroids_df.round(4))

new_df = data.copy()
new_df['jllable'] = best_kmeans.labels_  
new_df.to_csv("./data_mining/ex_5/output/new_df.csv", index=False, encoding='utf-8-sig')
print(f"\n包含聚类类别的数据集已保存为 new_df.csv，形状：{new_df.shape}")
print("new_df前5行：")
print(new_df[['driver', 'v_avg', 'a_var', 'jllable']].head())


pca = PCA(n_components=2, random_state=42)
reduced_data = pca.fit_transform(data_features_scaled)  

new_pca = pd.DataFrame(
    reduced_data,
    columns=['PC1', 'PC2']  
)
print(f"\nPCA降维后的数据（new_pca）形状：{new_pca.shape}")
print("new_pca前5行：")
print(new_pca.head())

explained_variance = pca.explained_variance_ratio_
print(f"\nPCA主成分解释方差比例：PC1={explained_variance[0]:.4f}, PC2={explained_variance[1]:.4f}")
print(f"累计解释方差比例：{sum(explained_variance):.4f}")


plt.figure(figsize=(10, 8))

d = new_pca[new_df['jllable'] == 0]
plt.plot(d['PC1'], d['PC2'], 'r.', markersize=8, label='普通驾驶', alpha=0.7)

d = new_pca[new_df['jllable'] == 1]
plt.plot(d['PC1'], d['PC2'], 'go', markersize=8, label='激进驾驶', alpha=0.7)

d = new_pca[new_df['jllable'] == 2]
plt.plot(d['PC1'], d['PC2'], 'b*', markersize=8, label='超冷静驾驶', alpha=0.7)

centroids_pca = pca.transform(centroids_scaled)
plt.scatter(
    centroids_pca[:, 0],
    centroids_pca[:, 1],
    c='black',
    marker='^',
    s=200,
    edgecolor='red',
    linewidth=2,
    label='聚类中心'
)

plt.title('驾驶类型聚类结果（PCA降维可视化）', fontsize=16)
plt.xlabel(f'主成分1（解释方差：{explained_variance[0]:.2%}）', fontsize=12)
plt.ylabel(f'主成分2（解释方差：{explained_variance[1]:.2%}）', fontsize=12)
plt.legend(loc='best', fontsize=12)
plt.grid(alpha=0.3, linestyle='--')
plt.tight_layout()

plt.savefig("./data_mining/ex_5/output/kmeans.png", dpi=300, bbox_inches='tight')
plt.show()
print("\n聚类可视化图已保存为 kmeans.png")

print("\n=== 聚类结果解释 ===")
radical_features = centroids_df.loc['激进驾驶']
calm_features = centroids_df.loc['超冷静驾驶']
normal_features = centroids_df.loc['普通驾驶']

print(f"激进驾驶特征：高平均速度（{radical_features['v_avg']:.2f}）、高加速度方差（{radical_features['a_var']:.2f}）")
print(f"超冷静驾驶特征：低平均速度（{calm_features['v_avg']:.2f}）、低加速度方差（{calm_features['a_var']:.2f}）")
print(f"普通驾驶特征：介于两者之间（平均速度：{normal_features['v_avg']:.2f}）")