# Modeling Extreme Events in Time Series Prediction

## ABSTRACT

以往的工作忽视了 _极端事件_。  
基于极值理论，提出Extreme Value Loss（EVL），探测未来发生极端事件。  
Memory Network

## 1 INTRODUCTION

DNN在近些年的成功。RNN  
不平衡数据  
致力于提升DNN在预测时间序列的极端事件的能力

## 2 PRELIMINARIES

### 2.1 Time Series Prediction

### 2.2 Extreme Events

#### 2.2.1 Heavy-tailed Distributions

现实世界数据经验分布总是 _heavy-tailed_ 。  

#### 2.2.2 Extreme Value Theory

#### 2.2.3 Modeling The Tail

## 3 PROBLEMS CAUSED BY EXTREME EVENTS

### 3.1 Empirical Distribution After Optimization

### 3.2 Why Deep Neural Network Could Suffer Extreme Event Problem

## PREDICTING TIME-SERIES DATA WITH EXTREME EVENTS

利用尾部先前观测数据，关注两个要素：

* 记忆极端事件：记忆网络
* 建模尾分布：基于观测数据的近似尾分布和EVL

### 4.1 Memory Network Module

极端事件的时间规律性

#### 4.1.1 Historical Window

对于每个时间间隔t，采样窗口M，使用GRU构建记忆网络模块  

#### 4.1.2 Attention Mechanism

### 4.2 Extreme Value Loss

### 4.3 Optimization

首先，结合预测输出$o_t$和极端事件发生预测；  
为每个窗口项j添加惩罚项

## 5 EMPIRICAL RESULTS

### 5.1 Experimental Settings

### 5.2 Effectiveness of Time Series Prediction

### 5.3 Effectiveness of EVL

### 5.4 Influence of Hyper-parameters in Memory Network

## 6 RELATED WORK

## 7 CONCLUSION

致力于深度学习方法在序列预测的表现，尤其是细粒度的极端事件模型
