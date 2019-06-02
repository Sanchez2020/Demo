## 摘要

基于PyTorch实现GAT和GCN在Cora，CiteSeer，PubMed三个引文数据集上以半监督学习方式进行节点分类任务，并对节点的嵌入表示进行降维可视化。

### 1. 安装需求

|     package     | version |
| :-------------: | :-----: |
|   matplotlib    |  3.0.3  |
|      numpy      | 1.16.2  |
|     pandas      | 0.24.2  |
|     python      |  3.7.3  |
|     pytorch     |  1.1.0  |
|  scikit-learn   | 0.21.0  |
|     seaborn     |  0.9.0  |
|    texttable    |  1.6.1  |
|  torch-cluster  |  1.3.0  |
| torch-geometric |  1.2.0  |
|  torch-scatter  |  1.2.0  |
|  torch-sparse   |  0.4.0  |

### 2. 数据集

[planetoid](https://github.com/kimiyoung/planetoid)

### 3. 输出结果

|                  格式                   |      说明      |
| :-------------------------------------: | :------------: |
|  `model_name_dataset_name`_result.csv   |  节点预测结果  |
| `model_name_dataset_name`_embedding.csv |  嵌入表示结果  |
|   `model_name_dataset_name`_accs.svg    |  准确率变化图  |
|   `model_name_dataset_name`_model.pkl   |      模型      |
| `model_name_dataset_name`_embedding.svg | 嵌入降维可视化 |

### 4. 选项

|           参数           |  类型  |      说明      |   默认值    |
| :----------------------: | :----: | :------------: | :---------: |
|     --dataset-folder     | string |  数据集文件夹  | "./input/"  |
|      --dataset-name      | string |   数据集名称   |   "Cora"    |
|      --result-path       | string | 输出结果文件夹 | "./output/" |
|         --model          | string |      模型      |    "GAT"    |
|     --number-layers      |  int   |   图网络层数   |      2      |
| --attention-out-channels |  int   | GAT层输出通道  |      8      |
|       --multi-head       |  int   |    GAT多头     |      8      |
|    --gcn-out-channels    |  int   | GCN层输出通道  |     64      |
|        --dropout         | float  |    Dropout     |     0.6     |
|     --learning-rate      | float  |     学习率     |    0.005    |
|      --weight-decay      | float  |    权重衰减    |   0.0005    |
|         --epochs         |  int   |    训练轮数    |     200     |
|                          |  int   | t-SNE迭代次数  |    10000    |

### 5. 示例

```python
python scr/main.py
```

!["动态演示"](./Demo演示.gif"动态演示")

指定训练100轮

```python
python scr/main.py --epochs 100
```

### 6. 引用

[GAT](https://arxiv.org/abs/1710.10903)

[GCN](https://arxiv.org/abs/1609.02907)

### 7. 代码参考

[PyTorch Geometric](<https://github.com/rusty1s/pytorch_geometric>)

[CapsGNN](<https://github.com/benedekrozemberczki/CapsGNN#outputs>)