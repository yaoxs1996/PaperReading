# Hidden POI Ranking with Spatial Crowdsourcing

缩写：

* LBSNs：Location Based Social Networks
* TCS：Tree-constrained Skip
* TCSS：Tree-constrained Skip with Supervision

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

