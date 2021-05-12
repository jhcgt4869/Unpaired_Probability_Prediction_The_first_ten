#!/usr/bin/env python
# coding: utf-8

# ## 螺旋桨RNA结构预测竞赛第10名方案
# <font size=3>
#   
# 队伍名：**白鹤亮对翅，黑熊飞双桨**  
# 成员：刘建建、史靖玮、项建彪 、杨静俐   
#   成绩情况：`score`：3.722  `rmsd_avg`:0.269  `rmsd_std`:0.067

# # 赛题介绍
# 
# “RNA碱基不成对概率”衡量了RNA序列在各个点位是否能形成稳定的碱基对（base pair），是RNA结构的重要属性，并可被应用在mRNA疫苗序列设计、药物研发等领域。例如mRNA疫苗序列通常不稳定，而RNA碱基不成对概率较高的点位正是易被降解的位置；又如RNA 碱基不成对概率较高的点位通常更容易与其他RNA序列相互作用，形成RNA-RNA binding等，这一特性也被广泛应用于疾病诊断和RNA药物研发。
# 
# 本次比赛提供了5000条训练数据，请选手基于训练数据和飞桨平台，开发模型预测RNA碱基不成对概率。
# 
# （<span style='color:red'>Tips：机器学习框架方面只允许使用飞桨深度学习框架哦</span>）
# 
# [比赛地址：https://aistudio.baidu.com/aistudio/competition/detail/61](https://aistudio.baidu.com/aistudio/competition/detail/61)

# ### 比赛思路
# <font size=3>
#   
# 先对基线进行了解，使用基线进行提交发现效果不错寻求进一步提升。    
# 在实践过程中发现效果呈现过拟合的情况，进行了提前停止。    
# 再寻求其他的优化模式，最后找到了一个成绩的段点（比较好的地方）   
# 通过这个值周边最近的几个`model`进行融合得到了最后的结果并进行提交    

# # 竞赛数据集 

# In[ ]:


# 检查数据集所在路径
get_ipython().system('tree /home/aistudio/data')


# # 基线系统代码结构
# 
# 本次基线基于飞桨PaddlePaddle2.0版本。

# In[ ]:


# 检查源代码文件结构
# !cd work; mkdir model
get_ipython().system('tree /home/aistudio/work -L 2')


# ## 训练脚本
# 
# `python src/main.py train --model-path-base [model_directory_name]`
# 
# 本代码会训练一个模型，并且保存到指定位置，训练日志默认保存到文件`train_log.txt`   
# 注意：由于初始化的不稳定，可能需要多次训练，比较合理的验证集(dev)均方误差损失值(MSE loss)为0.05-0.08  
# 
# #### 样例
# `python src/main.py train --model-path-base model`
# 
# #### 你将会看到类似如下的训练日志
# ```
# epoch 1 batch 40 processed 640 batch-loss 0.1984 epoch-elapsed 0h00m10s total-elapsed 0h00m11s 
# epoch 1 batch 41 processed 656 batch-loss 0.2119 epoch-elapsed 0h00m10s total-elapsed 0h00m11s 
# epoch 1 batch 42 processed 672 batch-loss 0.2205 epoch-elapsed 0h00m11s total-elapsed 0h00m11s 
# epoch 1 batch 43 processed 688 batch-loss 0.2128 epoch-elapsed 0h00m11s total-elapsed 0h00m11s 
# # Dev Average Loss: 0.212 (MSE) -> 0.461 (RMSD)
# ```
# 
# #### 注意事项
# 请使用<span style='color:red'>GPU版本</span>的配置环境运行本模块

# In[1]:


# To train:
# python src/main.py train --model-path-base [model_directory_name]

get_ipython().system('cd work; python src/main.py train --model-path-base model')


# ## 预测脚本
# 
# `python src/main.py test --model-path-base [saved_model_directory]`  
# 
# 本代码会预测一个模型，日志和结果默认保存到文件`test_log.txt` 
# 
# #### 样例  
# 1. 用<span style='color:red'>不带标签</span>的测试集来预测：  
# `python src/main.py test --model-path-base model-0/model_dev\=0.0772/`
# 2. 用<span style='color:red'>带标签</span>的测试集来预测并评估:  
# 	`python src/main.py test_withlabel --model-path-base model-0/model_dev\=0.0772/`  
# 	样例输出
#     ```
#     # python3 src/main.py test_withlabel --model-path-base model-0/model_dev=0.0772
#     Loading data...
#     Loading model...
#     initializing vacabularies... done.
#     Sequence(6): ['<START>', '<STOP>', 'A', 'C', 'G', 'U']
#     Brackets(5): ['<START>', '<STOP>', '(', ')', '.']
#     W0113 21:57:44.871776   221 device_context.cc:252] Please NOTE: device: 0, CUDA Capability: 70, Driver API Version: 11.0, Runtime API Version: 9.0
#     W0113 21:57:44.878015   221 device_context.cc:260] device: 0, cuDNN Version: 7.6.
#     #  Dev Average Loss: 0.0772 (MSE) -> 0.2778 (RMSD)
#     # Test Average Loss: 0.0445 (MSE) -> 0.2111 (RMSD)
#     ```
# 
# - 由于比赛的公开数据不提供测试集的标签，故本基线模型无法运行预设的`test_withlabel`，除非用户自己生成一个带标签的测试集`~/data/test.txt`。  
# 
# 
# 
# 
# #### 注意事项
# 请使用<span style='color:red'>GPU版本</span>的配置环境运行本模块

# ### 预测说明
# <font size=3>
#   
# 在初赛时发现最后的结果呈现过拟合的情况，效果不是非常好。于是决定提前停止。  
# 在0.075的位置发现效果比较好。    
# 最后A榜结果为第4名 `score`：4.66  
# ****  
# 复赛时经过提交测试发现0.0762和0.0749的效果均不好最佳为0.0752次佳伟0.0756  
# 最后把结果进行融合得到最终结果。

# In[2]:


get_ipython().system('cd work; python src/main.py test --model-path-base model-0/model_dev\\=0.0772')


# In[3]:


get_ipython().system('cd work; python src/main.py test --model-path-base model/model_dev\\=0.0752')
count=0
list=[]
with open("work/test_log.txt","r")as f:
    for i in f.readlines():
        count+=1
        
        list.append(i.split(" "))
        with open(f"predict.files752/{count}.predict.txt","w")as ff:
            for i in range(len(list[count-1])):
                ff.write(list[count-1][i]+"\n")


# In[4]:


get_ipython().system('cd work; python src/main.py test --model-path-base model/model_dev\\=0.0756')
count=0
list=[]
with open("work/test_log.txt","r")as f:
    for i in f.readlines():
        count+=1
        
        list.append(i.split(" "))
        with open(f"predict.files756/{count}.predict.txt","w")as ff:
            for i in range(len(list[count-1])):
                ff.write(list[count-1][i]+"\n")


# ### 代码融合

# In[29]:


list1=[]
list2=[]
list3=[]
for i in range(1,113):
    with open(f"/home/aistudio/predict.files752/{i}.predict.txt", "r") as f:
        with open(f"/home/aistudio/predict.files756/{i}.predict.txt","r") as ff:
            with open(f"/home/aistudio/predict.files/{i}.predict.txt","a") as fff:
                for m in f.readlines():
                    list1.append((m.replace("\n","")))
                for n in ff.readlines():
                    list2.append((n.replace("\n","")))
                for j in range(0,len(list1)):
                    # print(list1[j])
                    if list1[j] and list2[j] !="\n":
                        a = (float(list1[j])+float(list2[j]))/2
                        list3.append(a)
                        # print(list3)
                        fff.write(str(list3[j])+"\n")


# ### 文件打包

# In[31]:


get_ipython().run_line_magic('cd', '/home/aistudio/predict.files')
get_ipython().system('zip -r -o /home/aistudio/predict.files.zip ./')
get_ipython().run_line_magic('cd', '/home/aistudio')


# <font size=3>
#   
#   这次的分享就到这里啦，感谢大家的支持!
