# Deep Mixture Point Processes:Spatio-temporal Event Prediction with Rich Contentual Information

## 摘要

## 1 INTRODUCTION

事件预测是诸多领域应用的关键组件，例如城市规划、交通优化以及基于地点的营销。  
我们关注使用点处理方法对连续时间空间的事件序列建模，不使用聚类。  
点处理直接估计强度函数（描述时间发生率）。上下文特征的影响可以被其进行建模，但点处理也有很大限制：基于协变量的函数形态的假设受限去捕获上下文特征的影响，无法适应非结构化的数据。  
本文使用dl方法强化点处理方法。使用dnn对强度建模。但是需要使用积分计算估计的可能性。  
本文设计强度作为专家深度混合，混合权重由dnn建模。DMPP使得能将非结构化上下文特征结合为预测模型，自动学习对事件发生的复杂影响。允许简单集成决定可能性。简单的后向传播。  
使用自注意力机制，更好得理解上下文特征。

## 2 RELATED WORK

## 3 PRELIMINARIES

点处理是一个领域内事件发生的随机序列。已知时间和地点的事件$x=(t,s)$，时间$t \in \Bbb{T}$与地点$s \in \Bbb{S}$，其中$\Bbb{T} \times \Bbb{S}$是$\R \times \R^2$的子集。  
我们定义$\Bbb{T} \times \Bbb{S}$的子集A中的事件下降的数目为$N(A)$。强度$\lambda(x)$代表小区域中的事件发生率：

$$\lambda(x)=\lambda(t,s)\equiv\lim_{|dt| \to 0,|ds| \to 0}{{\Bbb{E}[N(dt\times ds)]}\over{|dt||ds|}}$$

$dt$代表时间$t$的一个小间隔，$|dt|$是持续时间，$ds$是包含地点$s$的小区域，$|ds|$是其区域。$\Bbb{E}$是期望。函数强度用来捕获合适的事件发生的潜在动力学。  
事件序列$\mathcal{X}=\{x_i=(t_i,s_i)\}^N_{i=1}$，$t_i\in\mathbb{T}$以及$s_i \in \mathbb{S}$。

$$p(\mathcal{X}|\lambda(x))=\prod^N_{i=1}\lambda(x_i)\cdot \exp(-\int_{\mathbb{T} \times \mathbb{S}}\lambda(x)dx)$$

## 4 DEEP MIXTURE POINT PROCESSES

## 4.1 Probleam Definition

$\mathcal{D}=A_1,A_2,...,A_K$是上下文特征的集合。上下文特征包括天气、社会/交通事件信息以及地理特征。A的格式为$<time,latitude,longitude,description>$。  
给定时间$T+\Delta{T}$的上下文特征$\mathcal{D}$以及时间$T$的事件序列$\mathcal{X}$：

* 预测未来时间窗口$[T,T+\Delta{T}]$的事件时间和地点。
* 使用任意给定的空间区域和时间周期$[T,T+\Delta{T}]$预测时间数目。

### 4.2 Model Formulation
