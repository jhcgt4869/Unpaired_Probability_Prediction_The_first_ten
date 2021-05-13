## 螺旋桨RNA结构预测竞赛第10名方案
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

# 竞赛数据集 


```python
# 检查数据集所在路径
!tree /home/aistudio/data
```

    /home/aistudio/data
    ├── data67691
    │   └── test_log.txt
    └── data82504
        └── B_board_112_seqs.txt
    
    2 directories, 2 files


# 基线系统代码结构

本次基线基于飞桨PaddlePaddle2.0版本。


```python
# 检查源代码文件结构
# !cd work; mkdir model
!tree /home/aistudio/work -L 2
```

    /home/aistudio/work
    ├── data
    │   ├── dev.txt
    │   ├── test_nolabel.txt
    │   └── train.txt
    ├── model
    │   ├── model_dev=0.0673
    │   ├── model_dev=0.0674
    │   ├── model_dev=0.0678
    │   ├── model_dev=0.0749
    │   ├── model_dev=0.0752
    │   ├── model_dev=0.0756
    │   ├── model_dev=0.0762
    │   └── placeholder.txt
    ├── model-0
    │   └── model_dev=0.0772
    ├── README.txt
    ├── src
    │   ├── const.py
    │   ├── dataset.py
    │   ├── __init__.py
    │   ├── main.py
    │   ├── network.py
    │   ├── __pycache__
    │   ├── utils.py
    │   └── vocabulary.py
    ├── test_log.txt
    └── train_log.txt
    
    13 directories, 14 files


## 训练脚本

`python src/main.py train --model-path-base [model_directory_name]`

本代码会训练一个模型，并且保存到指定位置，训练日志默认保存到文件`train_log.txt`   
注意：由于初始化的不稳定，可能需要多次训练，比较合理的验证集(dev)均方误差损失值(MSE loss)为0.05-0.08  

#### 样例
`python src/main.py train --model-path-base model`

#### 你将会看到类似如下的训练日志
```
epoch 1 batch 40 processed 640 batch-loss 0.1984 epoch-elapsed 0h00m10s total-elapsed 0h00m11s 
epoch 1 batch 41 processed 656 batch-loss 0.2119 epoch-elapsed 0h00m10s total-elapsed 0h00m11s 
epoch 1 batch 42 processed 672 batch-loss 0.2205 epoch-elapsed 0h00m11s total-elapsed 0h00m11s 
epoch 1 batch 43 processed 688 batch-loss 0.2128 epoch-elapsed 0h00m11s total-elapsed 0h00m11s 
# Dev Average Loss: 0.212 (MSE) -> 0.461 (RMSD)
```

#### 注意事项
请使用<span style='color:red'>GPU版本</span>的配置环境运行本模块


```python
# To train:
# python src/main.py train --model-path-base [model_directory_name]

!cd work; python src/main.py train --model-path-base model
```

    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/utils.py:26: DeprecationWarning: `np.int` is a deprecated alias for the builtin `int`. To silence this warning, use `int` by itself. Doing this will not modify any behavior and is safe. When replacing `np.int`, you may wish to use e.g. `np.int64` or `np.int32` to specify the precision. If you wish to review your current use, check the release note link for additional information.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      def convert_to_list(value, n, name, dtype=np.int):
    2021-05-12 13:40:41.025758
    # python3 src/main.py train --model-path-base model
    # Training set contains 4750 Sequences.
    # Validation set contains 250 Sequences.
    # Paddle: Using device: CUDAPlace(0)
    # Initializing model...
    initializing vacabularies... done.
    Sequence(6): ['<START>', '<STOP>', 'A', 'C', 'G', 'U']
    Brackets(5): ['<START>', '<STOP>', '(', ')', '.']
    # Checking validation 10 times an epoch (every 475 batches)
    W0512 13:40:42.600050   511 device_context.cc:362] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 11.0, Runtime API Version: 10.1
    W0512 13:40:42.604969   511 device_context.cc:372] device: 0, cuDNN Version: 7.6.
    # Epoch 1 starting.
    epoch 1 batch 1 processed 1 batch-loss 0.2189 epoch-elapsed 0h00m00s total-elapsed 0h00m04s 
    epoch 1 batch 2 processed 2 batch-loss 0.1878 epoch-elapsed 0h00m00s total-elapsed 0h00m04s 
    epoch 1 batch 3 processed 3 batch-loss 0.2259 epoch-elapsed 0h00m00s total-elapsed 0h00m04s 
    epoch 1 batch 4 processed 4 batch-loss 0.2033 epoch-elapsed 0h00m00s total-elapsed 0h00m04s 
    epoch 1 batch 5 processed 5 batch-loss 0.2201 epoch-elapsed 0h00m01s total-elapsed 0h00m05s 
    epoch 1 batch 6 processed 6 batch-loss 0.2206 epoch-elapsed 0h00m01s total-elapsed 0h00m05s 
    epoch 1 batch 7 processed 7 batch-loss 0.2026 epoch-elapsed 0h00m01s total-elapsed 0h00m05s 
    epoch 1 batch 8 processed 8 batch-loss 0.2187 epoch-elapsed 0h00m01s total-elapsed 0h00m05s 
    epoch 1 batch 9 processed 9 batch-loss 0.2143 epoch-elapsed 0h00m01s total-elapsed 0h00m05s 
    epoch 1 batch 10 processed 10 batch-loss 0.2183 epoch-elapsed 0h00m02s total-elapsed 0h00m06s 
    epoch 1 batch 11 processed 11 batch-loss 0.2130 epoch-elapsed 0h00m02s total-elapsed 0h00m06s 
    epoch 1 batch 12 processed 12 batch-loss 0.2179 epoch-elapsed 0h00m02s total-elapsed 0h00m06s 
    epoch 1 batch 13 processed 13 batch-loss 0.2330 epoch-elapsed 0h00m02s total-elapsed 0h00m06s 
    epoch 1 batch 14 processed 14 batch-loss 0.2232 epoch-elapsed 0h00m02s total-elapsed 0h00m06s 
    epoch 1 batch 15 processed 15 batch-loss 0.1971 epoch-elapsed 0h00m02s total-elapsed 0h00m06s 
    epoch 1 batch 16 processed 16 batch-loss 0.1919 epoch-elapsed 0h00m03s total-elapsed 0h00m07s 
    epoch 1 batch 17 processed 17 batch-loss 0.1845 epoch-elapsed 0h00m03s total-elapsed 0h00m07s 
    epoch 1 batch 18 processed 18 batch-loss 0.1756 epoch-elapsed 0h00m03s total-elapsed 0h00m07s 
    epoch 1 batch 19 processed 19 batch-loss 0.2241 epoch-elapsed 0h00m03s total-elapsed 0h00m07s 
    epoch 1 batch 20 processed 20 batch-loss 0.2204 epoch-elapsed 0h00m03s total-elapsed 0h00m07s 
    epoch 1 batch 21 processed 21 batch-loss 0.2264 epoch-elapsed 0h00m03s total-elapsed 0h00m07s 
    epoch 1 batch 22 processed 22 batch-loss 0.1991 epoch-elapsed 0h00m03s total-elapsed 0h00m07s 
    epoch 1 batch 23 processed 23 batch-loss 0.2336 epoch-elapsed 0h00m03s total-elapsed 0h00m07s 
    epoch 1 batch 24 processed 24 batch-loss 0.2168 epoch-elapsed 0h00m04s total-elapsed 0h00m08s 
    epoch 1 batch 25 processed 25 batch-loss 0.2207 epoch-elapsed 0h00m04s total-elapsed 0h00m08s 
    epoch 1 batch 26 processed 26 batch-loss 0.2035 epoch-elapsed 0h00m04s total-elapsed 0h00m08s 
    epoch 1 batch 27 processed 27 batch-loss 0.2133 epoch-elapsed 0h00m04s total-elapsed 0h00m08s 
    epoch 1 batch 28 processed 28 batch-loss 0.2210 epoch-elapsed 0h00m04s total-elapsed 0h00m08s 
    epoch 1 batch 29 processed 29 batch-loss 0.2162 epoch-elapsed 0h00m04s total-elapsed 0h00m08s 
    epoch 1 batch 30 processed 30 batch-loss 0.2223 epoch-elapsed 0h00m05s total-elapsed 0h00m09s 
    epoch 1 batch 31 processed 31 batch-loss 0.2235 epoch-elapsed 0h00m05s total-elapsed 0h00m09s 
    epoch 1 batch 32 processed 32 batch-loss 0.2076 epoch-elapsed 0h00m05s total-elapsed 0h00m09s 
    epoch 1 batch 33 processed 33 batch-loss 0.2024 epoch-elapsed 0h00m05s total-elapsed 0h00m09s 
    epoch 1 batch 34 processed 34 batch-loss 0.2153 epoch-elapsed 0h00m05s total-elapsed 0h00m09s 
    epoch 1 batch 35 processed 35 batch-loss 0.2099 epoch-elapsed 0h00m05s total-elapsed 0h00m09s 
    epoch 1 batch 36 processed 36 batch-loss 0.2023 epoch-elapsed 0h00m05s total-elapsed 0h00m09s 
    epoch 1 batch 37 processed 37 batch-loss 0.2232 epoch-elapsed 0h00m06s total-elapsed 0h00m10s 
    epoch 1 batch 38 processed 38 batch-loss 0.2079 epoch-elapsed 0h00m06s total-elapsed 0h00m10s 
    epoch 1 batch 39 processed 39 batch-loss 0.2055 epoch-elapsed 0h00m06s total-elapsed 0h00m10s 
    epoch 1 batch 40 processed 40 batch-loss 0.2025 epoch-elapsed 0h00m06s total-elapsed 0h00m10s 
    epoch 1 batch 41 processed 41 batch-loss 0.2248 epoch-elapsed 0h00m06s total-elapsed 0h00m10s 
    epoch 1 batch 42 processed 42 batch-loss 0.2072 epoch-elapsed 0h00m06s total-elapsed 0h00m10s 
    epoch 1 batch 43 processed 43 batch-loss 0.2107 epoch-elapsed 0h00m06s total-elapsed 0h00m10s 
    epoch 1 batch 44 processed 44 batch-loss 0.2194 epoch-elapsed 0h00m07s total-elapsed 0h00m11s 
    epoch 1 batch 45 processed 45 batch-loss 0.1972 epoch-elapsed 0h00m07s total-elapsed 0h00m11s 
    epoch 1 batch 46 processed 46 batch-loss 0.1819 epoch-elapsed 0h00m07s total-elapsed 0h00m11s 
    epoch 1 batch 47 processed 47 batch-loss 0.1911 epoch-elapsed 0h00m07s total-elapsed 0h00m11s 
    epoch 1 batch 48 processed 48 batch-loss 0.2107 epoch-elapsed 0h00m07s total-elapsed 0h00m11s 
    epoch 1 batch 49 processed 49 batch-loss 0.1512 epoch-elapsed 0h00m07s total-elapsed 0h00m11s 
    epoch 1 batch 50 processed 50 batch-loss 0.1901 epoch-elapsed 0h00m07s total-elapsed 0h00m11s 
    epoch 1 batch 51 processed 51 batch-loss 0.1936 epoch-elapsed 0h00m08s total-elapsed 0h00m11s 
    epoch 1 batch 52 processed 52 batch-loss 0.1962 epoch-elapsed 0h00m08s total-elapsed 0h00m12s 
    epoch 1 batch 53 processed 53 batch-loss 0.1924 epoch-elapsed 0h00m08s total-elapsed 0h00m12s 
    epoch 1 batch 54 processed 54 batch-loss 0.2275 epoch-elapsed 0h00m08s total-elapsed 0h00m12s 
    epoch 1 batch 55 processed 55 batch-loss 0.1977 epoch-elapsed 0h00m08s total-elapsed 0h00m12s 
    epoch 1 batch 56 processed 56 batch-loss 0.2144 epoch-elapsed 0h00m08s total-elapsed 0h00m12s 
    epoch 1 batch 57 processed 57 batch-loss 0.1948 epoch-elapsed 0h00m09s total-elapsed 0h00m13s 
    epoch 1 batch 58 processed 58 batch-loss 0.1993 epoch-elapsed 0h00m09s total-elapsed 0h00m13s 
    epoch 1 batch 59 processed 59 batch-loss 0.2186 epoch-elapsed 0h00m09s total-elapsed 0h00m13s 
    epoch 1 batch 60 processed 60 batch-loss 0.1905 epoch-elapsed 0h00m09s total-elapsed 0h00m13s 
    epoch 1 batch 61 processed 61 batch-loss 0.2148 epoch-elapsed 0h00m09s total-elapsed 0h00m13s 
    epoch 1 batch 62 processed 62 batch-loss 0.1968 epoch-elapsed 0h00m10s total-elapsed 0h00m14s 
    epoch 1 batch 63 processed 63 batch-loss 0.1934 epoch-elapsed 0h00m10s total-elapsed 0h00m14s 
    epoch 1 batch 64 processed 64 batch-loss 0.1935 epoch-elapsed 0h00m10s total-elapsed 0h00m14s 
    epoch 1 batch 65 processed 65 batch-loss 0.2298 epoch-elapsed 0h00m10s total-elapsed 0h00m14s 
    epoch 1 batch 66 processed 66 batch-loss 0.1970 epoch-elapsed 0h00m10s total-elapsed 0h00m14s 
    epoch 1 batch 67 processed 67 batch-loss 0.1793 epoch-elapsed 0h00m10s total-elapsed 0h00m14s 
    epoch 1 batch 68 processed 68 batch-loss 0.2281 epoch-elapsed 0h00m11s total-elapsed 0h00m14s 
    epoch 1 batch 69 processed 69 batch-loss 0.1924 epoch-elapsed 0h00m11s total-elapsed 0h00m15s 
    epoch 1 batch 70 processed 70 batch-loss 0.1790 epoch-elapsed 0h00m11s total-elapsed 0h00m15s 
    ^C
    Traceback (most recent call last):
      File "src/main.py", line 392, in <module>
        main()
      File "src/main.py", line 387, in main
        args.callback(args)
      File "src/main.py", line 165, in run_train
        return_numpy=False)
      File "/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/executor.py", line 1108, in run
        return_merged=return_merged)
      File "/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/executor.py", line 1238, in _run_impl
        use_program_cache=use_program_cache)
      File "/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/executor.py", line 1328, in _run_program
        [fetch_var_name])
    KeyboardInterrupt


## 预测脚本

`python src/main.py test --model-path-base [saved_model_directory]`  

本代码会预测一个模型，日志和结果默认保存到文件`test_log.txt` 

#### 样例  
1. 用<span style='color:red'>不带标签</span>的测试集来预测：  
`python src/main.py test --model-path-base model-0/model_dev\=0.0772/`
2. 用<span style='color:red'>带标签</span>的测试集来预测并评估:  
	`python src/main.py test_withlabel --model-path-base model-0/model_dev\=0.0772/`  
	样例输出
    ```
    # python3 src/main.py test_withlabel --model-path-base model-0/model_dev=0.0772
    Loading data...
    Loading model...
    initializing vacabularies... done.
    Sequence(6): ['<START>', '<STOP>', 'A', 'C', 'G', 'U']
    Brackets(5): ['<START>', '<STOP>', '(', ')', '.']
    W0113 21:57:44.871776   221 device_context.cc:252] Please NOTE: device: 0, CUDA Capability: 70, Driver API Version: 11.0, Runtime API Version: 9.0
    W0113 21:57:44.878015   221 device_context.cc:260] device: 0, cuDNN Version: 7.6.
    #  Dev Average Loss: 0.0772 (MSE) -> 0.2778 (RMSD)
    # Test Average Loss: 0.0445 (MSE) -> 0.2111 (RMSD)
    ```

- 由于比赛的公开数据不提供测试集的标签，故本基线模型无法运行预设的`test_withlabel`，除非用户自己生成一个带标签的测试集`~/data/test.txt`。  




#### 注意事项
请使用<span style='color:red'>GPU版本</span>的配置环境运行本模块

### 预测说明
<font size=3>
  
在初赛时发现最后的结果呈现过拟合的情况，效果不是非常好。于是决定提前停止。  
在0.075的位置发现效果比较好。    
最后A榜结果为第4名 `score`：4.66  
****  
复赛时经过提交测试发现0.0762和0.0749的效果均不好最佳为0.0752次佳为0.0756  
最后把结果进行融合得到最终结果。


```python
!cd work; python src/main.py test --model-path-base model-0/model_dev\=0.0772
```

    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/utils.py:26: DeprecationWarning: `np.int` is a deprecated alias for the builtin `int`. To silence this warning, use `int` by itself. Doing this will not modify any behavior and is safe. When replacing `np.int`, you may wish to use e.g. `np.int64` or `np.int32` to specify the precision. If you wish to review your current use, check the release note link for additional information.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      def convert_to_list(value, n, name, dtype=np.int):
    Loading data...
    Loading model...
    W0512 13:46:49.052799   817 device_context.cc:362] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 11.0, Runtime API Version: 10.1
    W0512 13:46:49.058928   817 device_context.cc:372] device: 0, cuDNN Version: 7.6.
    0.8769019 0.827059 0.124781355 0.07532252 0.07499607 0.1314325 0.08891902 0.13985236 0.09831676 0.08501828 0.07340165 0.07567302 0.1133676 0.09805709 0.06310087 0.056796186 0.057331596 0.0815781 0.27457792 0.9096467 0.93526053 0.9451846 0.9142914 0.109518185 0.076339886 0.048351273 0.049028248 0.05358911 0.060041353 0.074196644 0.13099869 0.88650066 0.8841263 0.108059675 0.078317374 0.05504336 0.056563973 0.081898704 0.243508 0.9161593 0.9409232 0.9298708 0.30104664 0.096637875 0.054647654 0.051664583 0.06739785 0.09219342 0.7264107 0.726659 0.73035246 0.7605198 0.89717275 0.90631706 0.8475187 0.9132216 0.91307867 0.7822155 0.079098985 0.06328807 0.09567865 0.09337111 0.056361835 0.05519329 0.07371606 0.20833306 0.8994241 0.9335145 0.15200244 0.16098142 0.13979791 0.83967584 0.9244599 0.93794787 0.93146956 0.8610643 0.14420095 0.104273684 0.08480854 0.0657236 0.07158427 0.104142815 0.69858843 0.8231929 0.7192306 0.7296086 0.8188443 0.70629466 0.73674756 0.86907995 0.8761509 0.7557834 0.27506483 0.27363357 0.2096101 0.12540138 0.09406437 0.12995595 0.14843474 0.88662666 0.9157279 0.92791003 0.921483 0.86955506 0.86588264 0.9174033 0.89678645 0.10066239 0.08461384 0.07529054 0.12906869 0.12231052 0.1496461 0.21498549 0.14801437 0.2764888  0.081257366 0.9033271 0.9342948 0.89194506 0.8807766 0.10806908 0.09785647 0.07428005 0.100757055 0.82896733 0.21201192 0.20115143 0.091858394 0.07151817 0.090858996 0.08316694 0.08005128 0.07875547 0.07702952 0.07359996 0.09018034 0.08683971 0.11228195 0.7524584 0.87401533 0.3552185 0.32328513
    0.85190153 0.8004978 0.7834312 0.87545794 0.80943245 0.8141827 0.8870493 0.87456256 0.2247908 0.1776992 0.1528118 0.20111498 0.15980801 0.15832028 0.16394168 0.5951297 0.19600783 0.16838795 0.20453955 0.29506925 0.59307003 0.565762 0.30068 0.34491885 0.39538962 0.3886126 0.34946576 0.82175004 0.69261444 0.80722576 0.32902294 0.23999384 0.24694005 0.8471683 0.8440093 0.19210613 0.18218043 0.15031442 0.8006151 0.8973477 0.89688265 0.74138224 0.6770714 0.16547869 0.21786806 0.15676819 0.7439129 0.7421006 0.85367626 0.22139461 0.19860391 0.39737356 0.866………… 0.7619047



```python
!cd work; python src/main.py test --model-path-base model/model_dev\=0.0756
count=0
list=[]
with open("work/test_log.txt","r")as f:
    for i in f.readlines():
        count+=1

        list.append(i.split(" "))
        with open(f"predict.files756/{count}.predict.txt","w")as ff:
            for i in range(len(list[count-1])):
                ff.write(list[count-1][i]+"\n")
```

    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/utils.py:26: DeprecationWarning: `np.int` is a deprecated alias for the builtin `int`. To silence this warning, use `int` by itself. Doing this will not modify any behavior and is safe. When replacing `np.int`, you may wish to use e.g. `np.int64` or `np.int32` to specify the precision. If you wish to review your current use, check the release note link for additional information.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      def convert_to_list(value, n, name, dtype=np.int):
    Loading data...
    Loading model...
    W0512 13:47:54.666482   981 device_context.cc:362] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 11.0, Runtime API Version: 10.1
    W0512 13:47:54.671232   981 device_context.cc:372] device: 0, cuDNN Version: 7.6.
    0.92458844 0.8973756 0.083819814 0.048119456 0.04318866 0.06369634 0.061531093 0.08912887 0.07272421 0.06061156 0.047806975 0.04566294 0.05193742 0.0465386 0.038122084 0.035683643 0.035520226 0.044568405 0.089197926 0.9403354 0.95413107 0.9659275 0.947999 0.091526456 0.043470513 0.033027984 0.03248403 0.03333013 0.03653355 0.046932414 0.07692799 0.90853477 0.92064553 0.10450742 0.06477215 0.04051999 0.03910085 0.06434644 0.19825488 0.9514736 0.9613649 86 0.27746683 0.56432915 0.27165216 0.22117698 0.6343458 0.28548688 0.1490245 0.30219793 0.07924749 0.061275408 0.0712523 0.09512221 0.71393305 0.7728529 0.78344065 0.12242744 0.077154376 0.07725689 0.058522586 0.04672195 0.04593994 0.052065138 0.113416955 0.14666957 0.85096854 0.81819427 0.8156704 0.84576315 0.08336176 0.057449628 0.059872873 0.0978862 0.8592945 0.14375012 0.12232697 0.69887173 0.13994755 0.11359459 0.15944806 0.10881757 0.7315731 0.105488054 0.077173166 0.12365675 0.8588115 0.24234413 0.7861132 0.107930176 0.07159182 0.09324458 0.09120768 0.07186025 0.8105579 0.79349583 0.83615726 0.8965029 0.9502591 0.94215065 0.9564655 0.9442282 0.06240864 0.03893294 0.03351966 0.035541855 0.040954556 0.04281014 0.04482754 0.047412407 0.050068595 0.051022653 0.052231997 0.049969897 0.050768476 0.058887467………………………… 0.84340286     



### 代码融合


```python
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
```

### 文件打包


```python
%cd /home/aistudio/predict.files
!zip -r -o /home/aistudio/predict.files.zip ./
%cd /home/aistudio
```

    /home/aistudio/predict.files
      adding: 52.predict.txt (deflated 69%)
      adding: 9.predict.txt (deflated 84%)
      adding: 71.predict.txt (deflated 68%)
      adding: 37.predict.txt (deflated 69%)
      adding: 38.predict.txt (deflated 69%)
      adding: 19.predict.txt (deflated 74%)
      adding: 86.predict.txt (deflated 67%)
      adding: 83.predict.txt (deflated 67%)
      adding: 17.predict.txt (deflated 74%)
      adding: 28.predict.txt (deflated 71%)
      adding: 101.predict.txt (deflated 67%)
      adding: 74.predict.txt (deflated 68%)
      adding: 80.predict.txt (deflated 68%)
      adding: 26.predict.txt (deflated 71%)
      adding: 33.predict.txt (deflated 70%)
      adding: 85.predict.txt (deflated 67%)
      adding: 90.predict.txt (deflated 67%)
      adding: 107.predict.txt (deflated 67%)
      adding: 24.predict.txt (deflated 72%)
      adding: 75.predict.txt (deflated 68%)
      adding: 44.predict.txt (deflated 69%)
      adding: 35.predict.txt (deflated 69%)
      adding: 82.predict.txt (deflated 68%)
      adding: 10.predict.txt (deflated 82%)
      adding: 64.predict.txt (deflated 68%)
      adding: 14.predict.txt (deflated 77%)
      adding: 48.predict.txt (deflated 69%)
      adding: 100.predict.txt (deflated 67%)
      adding: 68.predict.txt (deflated 68%)
      adding: 15.predict.txt (deflated 76%)
      adding: 66.predict.txt (deflated 68%)
      adding: 16.predict.txt (deflated 75%)
      adding: 70.predict.txt (deflated 68%)
      adding: 45.predict.txt (deflated 69%)
      adding: 98.predict.txt (deflated 67%)
      adding: 110.predict.txt (deflated 66%)
      adding: 36.predict.txt (deflated 69%)
      adding: 104.predict.txt (deflated 67%)
      adding: 27.predict.txt (deflated 71%)
      adding: 60.predict.txt (deflated 69%)
      adding: 8.predict.txt (deflated 85%)
      adding: 65.predict.txt (deflated 68%)
      adding: 77.predict.txt (deflated 68%)
      adding: 55.predict.txt (deflated 69%)
      adding: 92.predict.txt (deflated 67%)
      adding: 43.predict.txt (deflated 69%)
      adding: 81.predict.txt (deflated 68%)
      adding: 1.predict.txt (deflated 97%)
      adding: 29.predict.txt (deflated 70%)
      adding: 88.predict.txt (deflated 67%)
      adding: 25.predict.txt (deflated 71%)
      adding: 21.predict.txt (deflated 73%)
      adding: 46.predict.txt (deflated 69%)
      adding: 18.predict.txt (deflated 74%)
      adding: 13.predict.txt (deflated 78%)
      adding: 39.predict.txt (deflated 69%)
      adding: 84.predict.txt (deflated 67%)
      adding: 30.predict.txt (deflated 70%)
      adding: 22.predict.txt (deflated 72%)
      adding: 7.predict.txt (deflated 83%)
      adding: 91.predict.txt (deflated 67%)
      adding: 62.predict.txt (deflated 68%)
      adding: 20.predict.txt (deflated 73%)
      adding: 54.predict.txt (deflated 69%)
      adding: 97.predict.txt (deflated 67%)
      adding: 63.predict.txt (deflated 68%)
      adding: 99.predict.txt (deflated 67%)
      adding: 23.predict.txt (deflated 72%)
      adding: 76.predict.txt (deflated 68%)
      adding: 78.predict.txt (deflated 68%)
      adding: 51.predict.txt (deflated 69%)
      adding: 73.predict.txt (deflated 68%)
      adding: 5.predict.txt (deflated 83%)
      adding: 40.predict.txt (deflated 69%)
      adding: 103.predict.txt (deflated 67%)
      adding: 34.predict.txt (deflated 69%)
      adding: 67.predict.txt (deflated 68%)
      adding: 109.predict.txt (deflated 66%)
      adding: 79.predict.txt (deflated 68%)
      adding: 11.predict.txt (deflated 80%)
      adding: 6.predict.txt (deflated 80%)
      adding: 42.predict.txt (deflated 69%)
      adding: 2.predict.txt (deflated 79%)
      adding: 112.predict.txt (deflated 66%)
      adding: 50.predict.txt (deflated 69%)
      adding: 47.predict.txt (deflated 69%)
      adding: 72.predict.txt (deflated 68%)
      adding: 87.predict.txt (deflated 67%)
      adding: 105.predict.txt (deflated 67%)
      adding: 102.predict.txt (deflated 67%)
      adding: 32.predict.txt (deflated 70%)
      adding: 31.predict.txt (deflated 70%)
      adding: 69.predict.txt (deflated 68%)
      adding: 61.predict.txt (deflated 68%)
      adding: 94.predict.txt (deflated 67%)
      adding: 3.predict.txt (deflated 73%)
      adding: 53.predict.txt (deflated 69%)
      adding: 95.predict.txt (deflated 67%)
      adding: 41.predict.txt (deflated 69%)
      adding: 58.predict.txt (deflated 69%)
      adding: 96.predict.txt (deflated 67%)
      adding: 4.predict.txt (deflated 79%)
      adding: 56.predict.txt (deflated 69%)
      adding: 108.predict.txt (deflated 66%)
      adding: 59.predict.txt (deflated 69%)
      adding: 106.predict.txt (deflated 66%)
      adding: 49.predict.txt (deflated 69%)
      adding: 111.predict.txt (deflated 66%)
      adding: 93.predict.txt (deflated 67%)
      adding: 89.predict.txt (deflated 67%)
      adding: 57.predict.txt (deflated 69%)
      adding: 12.predict.txt (deflated 78%)
    /home/aistudio


<font size=3>
  
  这次的分享就到这里啦，感谢大家的支持!
