import numpy as np
file_path = r'./data/R.capsulatus/numpy.test.esm.feature.npy'
X = np.load(file_path, allow_pickle=True)
# X=np.load('C:/Users/22750/Desktop/iEnhancer-DCSV-main/data/R.capsulatus/numpy.test.esm.feature.npy')
# 将特征矩阵展平成二维矩阵，每行表示一个样本
X_flattened = X.reshape(X.shape[0], -1)
cov_matrix = np.cov(X_flattened, rowvar=False)
print("协方差矩阵：",cov_matrix.shape)
# 计算协方差矩阵的特征值和特征向量
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
print("协方差矩阵的特征值：")
print(eigenvalues)
print("协方差矩阵的特征向量矩阵：",eigenvectors.shape)
print(eigenvectors)
# 取特征向量矩阵的前 1232 列
selected_eigenvectors = eigenvectors[:, :200]

print("选取的特征向量矩阵：",selected_eigenvectors.shape)


