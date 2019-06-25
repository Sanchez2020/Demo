## 摘要

基于PyTorch实现GAT和GCN在**Cora，CiteSeer，PubMed**三个引文数据集上以半监督学习方式进行节点分类任务，并对节点的嵌入表示进行降维可视化。在**PPI**数据集（多图）上训练GAT模型，表明GAT模型可用于归纳式学习。

### 1. 安装需求

#### 1.1 系统环境

|      环境      |          版本           |
| :------------: | :---------------------: |
|      系统      |   Ubuntu 16.04.6 LTS    |
|  深度学习框架  |      PyTorch1.1.0       |
|      CUDA      |      CUDA 10.0.130      |
| 几何深度学习库 | PyTorch-Geometric 1.2.0 |
|    编程语言    |       Python3.7.3       |

#### 1.2 依赖库

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

[planetoid](https://github.com/kimiyoung/planetoid)（包含三个引文网络数据集，Cora，CiteSeer，PubMed），[PPI](https://s3.us-east-2.amazonaws.com/dgl.ai/dataset/ppi.zip)（蛋白质相互作用网络数据集）

### 3. 输出结果

#### 3.1 GCN和GAT在三个引文数据集上的输出

|                  格式                   |      说明      |
| :-------------------------------------: | :------------: |
|  `model_name_dataset_name`_result.csv   |  节点预测结果  |
| `model_name_dataset_name`_embedding.csv |  嵌入表示结果  |
|   `model_name_dataset_name`_accs.svg    |  准确率变化图  |
|   `model_name_dataset_name`_model.pkl   |      模型      |
| `model_name_dataset_name`_embedding.svg | 嵌入降维可视化 |

#### 3.2 GAT在PPI数据集上的输出

|       格式        |      说明      |
| :---------------: | :------------: |
| GAT_PPI_model.pkl |      模型      |
|    GAT_PPI.svg    | 训练过程变化图 |
|    pred_*.csv     |  节点预测结果  |
|    real_*.csv     |  节点真实结果  |
|  embedding_*.csv  |  嵌入表示结果  |

### 4. 选项

#### 4.1 输入输出路径

|       参数       |  类型  |      说明      | **默认值**  |
| :--------------: | :----: | :------------: | :---------: |
| --dataset-folder | string |  数据集文件夹  | "./input/"  |
|  --result-path   | string | 输出结果文件夹 | "./output/" |

#### 4.2 GAT模型参数

|           参数           |  类型  |     说明      | 默认值 |
| :----------------------: | :----: | :-----------: | :----: |
|           GAT            | string |    GAT模型    |   -    |
|      --dataset-name      | string |  数据集名称   | "Cora" |
|     --number-layers      |  int   |  图网络层数   |   2    |
| --attention-out-channels |  int   | GAT层输出通道 |   8    |
|       --multi-head       |  int   |    GAT多头    |   8    |
|        --dropout         | float  |    Dropout    |  0.6   |
|     --learning-rate      | float  |    学习率     | 0.005  |
|      --weight-decay      | float  |   权重衰减    | 0.0005 |
|         --epochs         |  int   |   训练轮数    |  200   |
|         --n-iter         |  int   | t-SNE迭代次数 | 10000  |

#### 4.3 GCN模型参数

|        参数        |  类型  |     说明      | 默认值 |
| :----------------: | :----: | :-----------: | :----: |
|        GCN         | string |    GCN模型    |   -    |
|   --dataset-name   | string |  数据集名称   | "Cora" |
|  --number-layers   |  int   |  图网络层数   |   2    |
| --gcn-out-channels |  int   |  图网络层数   |   64   |
|  --learning-rate   | float  |    学习率     |  0.01  |
|   --weight-decay   | float  |   权重衰减    | 0.0005 |
|      --epochs      |  int   |   训练轮数    |  200   |
|      --n-iter      |  int   | t-SNE迭代次数 | 10000  |

#### 4.4 在PPI数据集上训练GAT参数设置

|           参数           |  类型  |      说明      | 默认值 |
| :----------------------: | :----: | :------------: | :----: |
|         GAT_PPI          | string | PPI上的GAT模型 |   -    |
|      --dataset-name      | string |   数据集名称   | "PPI"  |
|     --number-layers      |  int   |  神经网络层数  |   3    |
| --attention-out-channels |  int   | GAT层输出通道  |  256   |
|       --multi-head       |  int   |    多头注意    |   4    |
|     --learning-rate      | float  |     学习率     | 0.005  |
|         --epochs         |  int   |    训练轮数    |  100   |

### 5. 示例

```python
python src/main.py
```

!["动态演示"](./Demo演示.gif"动态演示")

指定训练100轮

```python
python src/main.py --epochs 100
```

### 6. 参考文献

[GAT](https://arxiv.org/abs/1710.10903)

[GCN](https://arxiv.org/abs/1609.02907)

### 7. 代码参考

[PyTorch Geometric](<https://github.com/rusty1s/pytorch_geometric>)

[CapsGNN](<https://github.com/benedekrozemberczki/CapsGNN#outputs>)