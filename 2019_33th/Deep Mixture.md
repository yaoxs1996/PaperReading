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

### 4.1 Probleam Definition

$\mathcal{D}=A_1,A_2,...,A_K$是上下文特征的集合。上下文特征包括天气、社会/交通事件信息以及地理特征。A的格式为$<time,latitude,longitude,description>$。  
给定时间$T+\Delta{T}$的上下文特征$\mathcal{D}$以及时间$T$的事件序列$\mathcal{X}$：

* 预测未来时间窗口$[T,T+\Delta{T}]$的事件时间和地点。
* 使用任意给定的空间区域和时间周期$[T,T+\Delta{T}]$预测时间数目。

### 4.2 Model Formulation
dl模型被证明非结构数据中提取有意义的数据极其有用。点处理模型集成了dl方法。我们使用神经网络函数建模强度，用于接收上下文特征作为输入。  
__Intensity function__。灵活、计算高效，使用核卷积。

$$\tag{3} \lambda(x|\mathcal{D})=\int{f(u,Z(u;\mathcal{D});\theta)k(x,u)du}$$

其中$u=(\tau,r)$，$\tau \in \mathbb{T}$和$r \in \mathbb{S}$，$k(\cdot,u)$是中心在u的核函数。  
$f(\cdot)$是任意返回非负标量的深度学习模型，$\theta$代表深度学习网络的参数集合。  
$Z(u,\mathcal{D})=\{Z_1(u;A_1),...,Z_K(u;A_K)\}$时空点u的特征值的集合，$Z_K$提取u的第k个特征的值的操作符。  
集成的神经网络函数$f(\cdot)$是难处理的，所以引入$J$代表时空区域点$\mathcal{U}=\{u_j\}^J_{j=1}$，获得等式(10)的一个独立组件：

$$\tag{4} \lambda(x|\mathcal{D})=\sum^J_{j=1}f(u_j,z_j;\theta)k(x,u_j)$$

每个点$u_j=(\tau_j,r_j)$由时间$\tau_j \in \mathbb{T}$和位置$r_j \in \mathbb{S}$组成。  
我们将$Z(u_j;\mathcal{D})$定义为$z_j$，代表点$u_j$的第j个关联的上下文特征向量。  
强度被描述成核多专家模型，混合权重由深度神经网络建模，输入为上下文特征。

__Configuration of representative point__。多离散点$M$均匀分布在时间轴$[0,T+\Delta T]$，定义多时间点$\mathcal{T}:0=\tau'_1<...<\tau'_M=T+\Delta T$。  
用空间区域的多离散点$L$定义空间点$\mathcal{S}:r'_1,...,r'_L$其中$r'_l \in \mathbb{S}$。  
多代表点集合被定义成空间和时间点的笛卡尔积：$\mathcal{U}=\{(\tau,r)|\tau \in \mathcal{T} \land r \in \mathcal{S}$，因此$J=ML$。  
本文使用在常规网格中固定点来定位多代表点。多代表点的数目J决定了近似准确性和计算复杂度之间的权衡。越多的J会提高近似度，同时会降低计算开销（有疑问？）。  
__Neural network model__。我们提出了一个注意力网络结构能完全利用视觉和文本信息。该模型有三个组件：图像注意力网络负责提取图像特征、文本注意力网络负责编码文本特征、多模型融合模块。  
将提取的特征通过多模型融合模块融合成单个代表，作为强度函数的输入。

### 4.3 Parameter Learning

