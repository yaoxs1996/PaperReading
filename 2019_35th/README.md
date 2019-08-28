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

## A novel hybridization of echo state networks and multiplicative seasonal ARIMA model for mobile communication traffic series forecasting(2014)

时间序列预测模型被分为单模型和组合模型。单模型不足以捕获复杂时间序列的全部特征。经典线性模型被广泛使用，诸如AR、ARMA和ARIMA，针对非线性时间序列，SVM更有效。  
因为高精确度和良好的泛化能力，SVM可以用于时间序列预测，但是SVM难以在不同应用中选择核函数。  
ANNs拥有灵活的非线性映射能力，可以以任意期望的精度近似任意连续可测量函数，但是慢收敛和高计算训练代价，ANNs在实践中难以应用。  
Echo state network（ESNs）是ANN的一个新结构。不同于传统的ANN方法，ESN中，只有从动态储层到输出神经元的连接需要被训练，因此训练ESNs成为一个线性回归任务，解决了ANNs的收敛慢和次优解的问题。  
本文提出基于ESNs和乘法周期ARIMA模型的小波多分辨率分析（MRA）预测模型。将时间序列分解成平滑部分和周期部分，再使用适当的模型分别单独进行预测。

## Computer network traffic prediction: a comparion between traditional and deep learning neural networks

就计算机网络流量预测，比较4种人工神经网络：

* 多层感知机（MLP），使用反向传播作为训练算法
* Rprop MLP
* RNN
* 深度学习栈式自动编码器（SAE）

对于网络流量预测，MLP和RNN优于SAE。SAE深度神经网络训练时的计算复杂度更高。  
最佳预测方法应该是RNN。最佳精度和计算时间由JNN获得，一种SRN。带Rprop的RNN在短期和实时预测最好。  

## Prediction Future Traffic using Hidden Markov Models(2016)

现有两种主流的研究：

* 假定任意给定的时间槽，在特定的源-目的对之间，合计的传输流量数可以被测量，使用时间序列模型进行预测
  * 线性模型：AR、ARMA、ARIMA、FARIMA
  * 非线性模型：ANN、RNN、GARCH
  缺点：对流量数进行直接测量代价高，尤其是面对大规模高速网络，因此尽管这种方式简单，但是实际中的可拓展性较差。
* 网络断层扫描（network tomography），是第一种方式的补充。基于其他观测值，比如链路利用率，进行估计网络流量数。通常有确定的线性关系可以描述链路利用率和隐性流量数。
  缺点：网络中，链路数目是远小于源-目的对的数量，使用有限的链路利用率数目，很难还原隐性的流量数。
  
本文利用流总数和流容量估计流量数。使用隐马尔可夫模型描述流总数和流容量的关系和行为。使用核贝叶斯规则和带LSTM的RNN进行模型的训练和预测未来流量。

本方法避免了直接测量，在复杂度和存储要求方面降低代价，特别在大规模高速网络下很有用。  
