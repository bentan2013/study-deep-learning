#导入所需的模块和包
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor 
from sklearn.ensemble import AdaBoostRegressor
#创造数据集
rng = np.random.RandomState(1)
X = np.linspace(0, 6, 100).reshape(-1,1)
y = np.sin(X).ravel() + np.sin(6 * X).ravel() + rng.normal(0, 0.1, X.shape[0])

#训练回归模型
DTR = DecisionTreeRegressor(max_depth=4)
ABR = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),n_estimators=300, random_state=rng)
DTR.fit(X, y)
ABR.fit(X, y)
#预测结果
y_1 = DTR.predict(X) 
y_2 = ABR.predict(X)

#绘制可视化图形
plt.figure(figsize=(15,6))
plt.scatter(X, y, c="k", label="training samples") 
plt.plot(X, y_1, c="orange", label="DTR", linewidth=2) 
plt.plot(X, y_2, c="g", label="ABR", linewidth=2) 
plt.xlabel("data")
plt.ylabel("target")
plt.title("Boosted Decision Tree Regression", fontsize=15) 
plt.legend()
plt.show()