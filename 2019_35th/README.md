# 话务预测

## deep Spatial-Temporal neural Network

## LSTM RNN（2017）

网络流量特征：自相似、multiscalarity、long-range dependence、a highly nonlinear nature  
更高效地使用模型参数训练预测模型，进行大规模流量矩阵预测。 

## RNN（2018）

RNN、LSTM和GRU进行流量预测；RNN进行协议分类；数据包分布预测  
视作一个时间序列预测问题。统计方法：ARIMA、EM和Holt-Winters算法，概率方法：贝叶斯网络或者隐马尔可夫模型。神经网络表现优异。  
RNN及RNN的变体：LSTM、GRU。理论上可以以任意精度接近非线性动力系统、可以捕获时间序列的长期依赖关系。

## Joint Spatial and Temporal Classification of Mobile Traffic Demands（2017）

移动流量数据的时空分类，基于Exploratory Factor Analysis（EFA）。

## Large-scale Mobile Traffic Analysis: a Survey(2015)

## Adaptive Resource Scheduling based on Neural Network and Mobile Traffic Prediction

基于NN结构的AdaptSch调度框架。使用热力图

## Traffic Prediction for Mobile Network using Holt-Winter's Exponential Smoothing(2007)

针对周期性时间序列，基于定量预测方法，使用数学递归函数预测趋势行为。假定未来会跟随过去的相同模式。

## Understanding Mobile Traffic Patterns of Large Scale Cellular Towers in Urban Environment

城市移动数据使用可以分为5种基础的时间领域模式。  
4种基础组件线性组合。  

## Big Data Driven Mobile Traffic Understanding and Forecasting: A Time Series Approach(2016)

## ZipNet-GAN: Inferring Fine-grained Mobile Traffic Pattern via a Generative Adversarial Neural Network

## Multiscale Internet traffic forecasting using neural network and time series methods(2012)

在五分钟级和小时级，ARIMA和NNE（neural network ensemble）产生更小的误差。  
ARIMA不适合在线预测系统，因为对算力需求太高。  
NNE的搜索空间很高，但是启发式方法显著降低计算负担，且易于实现。NNE可以被用于实时预测。  
HW在天级预测有最好表现。

## Comparetive evaluation of ARIMA and ANFIS for modeling of wireless network traffic time series(2014)

* 统计/回归模型：AR、ARMA、GARMA、ARIMA、FARIMA
* 分数高斯噪音和分数布朗运动：在长期依赖数据上，比回归模型有更好的精确度
* 人工神经网络和逻辑模糊方法

Adaptive neuro fuzzy inference system（ANFIS）结合了模糊逻辑和神经网络。  
文中的三个场景下，ANFIS均优于ARIMA，但是代价是过高的计算复杂度，只适合算力充足的场景下。
