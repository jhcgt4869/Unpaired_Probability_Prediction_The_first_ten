# Unpaired_Probability_Prediction_The_first_ten
螺旋桨RNA结构预测竞赛第10名方案
<font size=3>
  
队伍名：**白鹤亮对翅，黑熊飞双桨**   
成员：刘建建、史靖玮、项建彪 、杨静俐  
  成绩情况：`score`：3.722  `rmsd_avg`:0.269  `rmsd_std`:0.067

# 赛题介绍

“RNA碱基不成对概率”衡量了RNA序列在各个点位是否能形成稳定的碱基对（base pair），是RNA结构的重要属性，并可被应用在mRNA疫苗序列设计、药物研发等领域。例如mRNA疫苗序列通常不稳定，而RNA碱基不成对概率较高的点位正是易被降解的位置；又如RNA 碱基不成对概率较高的点位通常更容易与其他RNA序列相互作用，形成RNA-RNA binding等，这一特性也被广泛应用于疾病诊断和RNA药物研发。

本次比赛提供了5000条训练数据，请选手基于训练数据和飞桨平台，开发模型预测RNA碱基不成对概率。

（<span style='color:red'>Tips：机器学习框架方面只允许使用飞桨深度学习框架哦</span>）

[比赛地址：https://aistudio.baidu.com/aistudio/competition/detail/61](https://aistudio.baidu.com/aistudio/competition/detail/61)

### 比赛思路
<font size=3>
  
先对基线进行了解，使用基线进行提交发现效果不错寻求进一步提升。    
在实践过程中发现效果呈现过拟合的情况，进行了提前停止。    
再寻求其他的优化模式，最后找到了一个成绩的段点（比较好的地方）   
通过这个值周边最近的几个`model`进行融合得到了最后的结果并进行提交    
