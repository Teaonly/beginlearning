# easyLearning.js

基于Javascript简单易用的机器学习库，从Karpathy的[convnetjs]( http://cs.stanford.edu/people/karpathy/convnetjs/) 仿制而来。

主要的特征包括：
* 基础的神经网络BP算法实现，基于梯度下降的最优化算法实现
* 支持逻辑回归、SVM(线性模型最大间隔)、Softmax等常规算法
* 通过Node.js的C++扩展，支持本地代码优化，同时支持浏览器运行
* 支持卷积网络，计划扩展支持RNN等其他深度学习模型

本项目目的是通过完全自造一遍轮子，完全掌握深度神经网络以及常规的机器学习算法。
同时形成一个自用的家酿(homebrew)工具，满足小规模的数据下的机器学习。
