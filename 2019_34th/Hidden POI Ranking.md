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

