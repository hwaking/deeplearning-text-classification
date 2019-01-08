** 基于 dennybritz's 项目 [cnn-text-classification-tf](https://github.com/dennybritz/cnn-text-classification-tf), 添加RNN+Attention 实现，同时对代码进行了简化和修改**


## Requirements

- Python 3
- Tensorflow > 0.12
- Numpy

## Training

- 首先在config.py中设置模型参数，具体参数含义如下：

```
config parameters:
   # 常规参数
  -- learning_rate    学习率
  -- training_steps   迭代次数
  -- batch_size       批数据量
  -- display_step     多少次打印一次结果
  -- evaluate_every   多少次评估一次模型
  -- checkpoint_every 多少次保存一次模型
  -- num_checkpoints  保存模型个数
  -- early_stop_steps 提前停止
  
  
  # 网络参数
  -- num_hidden        隐藏神经元个数
  -- num_classes       类别数目
  -- dropout_keep_prob dropout比例
  -- l2_reg_lambda     l2正则化强度
  
  # CNN 网络参数
  -- filter_sizes      卷积核规格
  # RNN 网络参数
  -- network           网络类型lstm/gru
  -- bi_drection       是否选择双向网络
  -- timesteps = 56    序列长度
  -- attention_size    attention神经元个数
  
   # 硬件设置
  -- allow_soft_placement = True
  -- log_device_placement = False
  
   # 数据路径
  -- dev_sample_percentage = 0.1    验证集比例
  -- positive_data_file = "./data/rt-polaritydata/rt-polarity.pos"
  -- negative_data_file = "./data/rt-polaritydata/rt-polarity.neg"

```

Train:

```
python train.py
```

## Evaluating

'''
待加入
'''

## References

- [Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882)
- [A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1510.03820)
- [Hierarchical Attention Networks for Document Classification](http://www.aclweb.org/anthology/N16-1174)
