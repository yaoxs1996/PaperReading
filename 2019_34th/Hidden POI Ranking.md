# Hidden POI Ranking with Spatial Crowdsourcing

缩写：

* LBSNs：Location Based Social Networks
* TCS：Tree-constrained Skip
* TCSS：Tree-constrained Skip with Supervision
* PRR：POI Relevance Relation
* CRR：Category Relevance Relation

## ABSTRACT

本文研究如何使用异质结对任务来增强传统的众包评级聚类框架，来清除H-POIs的隐藏特征。  
在离线阶段，通过取得地理文本有效的异质结对作为初始候选对象来减小搜索空间，然后开发两种使用的数据驱动策略计算工作者品质。  
在在线阶段，我们引入主动学习算法使用品质共同选择结对和工作者，以减少评估开销。  
为减少时间开销，而提出（Minimum Spanning） Tree-constrained Skip搜索策略。

## KEYWORDS

* Hidden POI
* POI ranking
* spatial crowdsourcing

## 1 INTRODUCTION

POI推荐系统致力于推荐那些用户感兴趣，但是从来没有去过的地点。  
出现一些挑战：首先用户选择的多样性情况恶化；其次推荐结果偏向于在总体LBSNs数据库中有很多份额的P-POIs。所有致力于发现潜在的POIs。  
众包市场的出现，使得个人和企业可以使用人类智囊团处理数据。利用群众力量，将那些在LBSNs中很少被提及，但是在实际生活中有很大兴趣的地点。  
我们使用异质结对$<H-POI,P-POI>$。且只有当H-POI和P-POI同类且H-POI的服务区域被包含在P-POI中，结对才是有效的。  

## 2 PROBLEM STATEMENT

### 2.1 Preliminary

### 2.2 Problem Definition

* 定义6.（SPATIAL TASK）：$s=<p_h,p_p>$，工作者需要给出他们偏好的POI。
* 定义7.（COMPARABLE HIDDEN POINT OF INTEREST）：存在至少$\kappa$个P-POIs，每个P-POI都满足
  1. $C_{p_h}=C_{p_p}$
  2. $d(l_{p_h},l_{p_p}) \leq r_{p_p}-r_{p_h}$，$d(a,b)$是a和b的欧氏距离。

第二个限制确保H-POI的服务区域在P-POI的区域内，表明工作者曾今访问过$p_h$。

* 定义8.（VALID SPATIAL TASK SET）
* 定义9.（MINIMUM VALID SPATIAL TASK SET）

__PROBLEM STATEMENT__。给定H-POI集合$P_h$，P-POI集合$P_p$以及空间众包工作者集合$W$。研究问题，如何生成候选任务，将合适的工作分给合适的人。

## 3 PROPOSED METHOD

### 3.1 Pair Generation

为减少在后续的主动学习阶段在VTS中寻找最佳结对的开销，构建MinVTS。  
__MinVTS Greedy Search Algorithm__。$P_h$、$P_p$和$\kappa$（H-POI应当匹配的P-POI的最小数目）作为算法的输入。

### 3.2 Category and Area Aware Worker Reliability Calculation

事前评估工作者的可靠性。元路径和网络嵌入的方法。

#### 3.2.1 Category Reliability

现实世界中的众包工作者的移动行为特征可以被建模为一个异质信息网络$G=(V,E)$，结点V有三种类型，边E有两种类型。  

* 定义10（WORKER-BASED NODE）：一个特定的工作者，表示为W。
* 定义11（POI-BASED NODE）：一个特定的POI，表示为P
* 定义12（CATEGORY-BASED NODE）：一个POI分类，表示为C
* 定义13（POI RELEVANCE EDGE）：只存在于W结点与P结点之间，表示为e(W,P)。表示一个工作者和一个POI之间的POI关联关系。边权重$w(W,P)$
* 定义14（CATEGORY RELEVANCE EDGE）：只存在于结点P和结点C之间，表示为$e(P,C)$，权重为1。

异质网络表征学习主要目的是学得低维潜在表征$\bold{X} \in \R^{|V|\times d}$。包含多个种类以及互联对象的基于元路径的随机游走被证明能高效地处理异质信息网络。三种元路径：

* CPC路径：表示一个POI(P)和两个categories(C)的关系
* PWP路径：勾连起两个POIs(P)和一个worker(W)的关系
* CPWPC路径：表示由一个worker(W)访问的两个categories(C)和POIs(P)。

基于元路径的随机游走策略确保不同类的结点的语义关系可以被合适的合并到skip-gram。优化函数被定义成：

$$\tag{1} \argmax_{\theta}\sum_{v \in V}\sum_{t \in T_V}\sum_{c_t \in N_t(v)}\log{p(c_t|v;\theta)}$$

$v$是$V$中的结点，$T_V$代表$V$中结点种类的集合，$N_t(v)$是类别t的v的邻居。$p(c_t|v;\theta)$代表给定结点$v$，有一个context node $c_t$的条件可能性。  
获得每个结点的表征向量，就可以评估基于分类k的工作人员w的可靠性：

$$\tag{2} \alpha^k_w={{sim(X_w,X_{C_k})} \over {\sum^{|C|}_{j=1}sim(X_w,X_{C_j})}}$$

$sim(a,b)$代表$(a,b)$的点积。

#### 3.2.2 Area Reliability

理解工作者的活跃区域，经纬坐标。基于聚类算法，将n个签到地点聚类为k个集合，关注工作区域的形心，表示为$\mu_i$。使用X-means快速估计k-means算法划分的个数。  
本文结合Bayesian Information Criterion（BIC）和Akaike Information Criterion（AIC）去评估聚类模型。  
修改X-means算法，设置阈值H，样本数大于H，使用BIC评估，否则使用AIC。  
A-B X-means算法可以合适地确定工作者的形心x。截止值集合$0 \equiv \gamma_0 < \gamma_1<...<\gamma_{m-1} \equiv inf$。一个形心o和一个POI p的观测距离为$d(o,p)$，如果$\gamma_{i-1}<d(o,p)<\gamma_i$，则$D(o,p)=i$。任务s中，在POI地点的工作者w的可靠性表示为$l_s$，被计算为：

$$\tag{5} \beta^{l_s}_w=(1-\lambda)\max_{n \in [1,x]}\{{{1} \over {D(o_n,p_h)}}\}+\lambda \max_{n \in [1,x]}\{{{1} \over {D(o_n,p_p)}}\}$$

$\lambda$代表H-POIs和P-POIs间的取舍。

### 3.3 Ranking Aggregation

评级的主要思想是为每个对象赋予一个评分$s_i$，对分数排序获得评级。

#### 3.3.1 Crowed-BT

#### 3.3.2 Computation of Overall Worker Quality

从LBSNs中提取信息，工作者质量分数$\mu_w$。分类可靠性$\alpha$，地点可靠性$\beta$，任务s中工作者w的总体质量计算为：

$$\mu^s_w=\epsilon \alpha^{C_{(s)}}_w+(1-\epsilon)\beta^{l_s}_w$$

$\epsilon \in [0,1]$，是手动设置参数，是分类和地点的取舍。$C_(s)$是任务s的分类。

#### 3.3.3 Tree-Constrained Skip Search

本文提出Tree-constrained Skip搜索算法，在Minimum Spanning Tree（MST）结点中找到一个局部最优对，从选定的对中调到两个次优解。  
