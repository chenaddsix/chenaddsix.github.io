---
layout: post
title: Investigation on Meta Learning
subtitle:   "\"Investigation on Meta Learning.\""
date: 2018-04-30
header-img: "img/post-bg-2015.jpg"
author: "Jiale Chen"
catalog: true
tags:
  - Machine Learning
---

<script type="text/javascript" async src="//cdn.bootcss.com/mathjax/2.7.0/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_CHTML"></script>

由于最近科研任务需要，开始关注meta learning的研究进展，发现内容繁多，方法也很多，所以写一篇调研报告帮助梳理思路。

Meta learning领域经常应用于Few-shot learning的问题，也就是如何在小数据量的问题上使模型快速收敛，这一点不管是在deep learning或者reinforcement learning领域都很重要。

**那么，meta-learning 和 few-shot learning 的动机是什么呢**？当我们没有足够多的标注数据来提供训练时，我们往往想从其他的无标注数据（无监督学习）、多模态数据（多模态学习）和多领域数据（领域迁移、迁移学习）里来获得更多帮助。在这种动机的驱动下，few-shot learning 便是希望针对一组小样本，得到一个能学习到很好的模型参数的算法。而如果我们能用端对端的方式学习到这种算法，那么就可以称为 **meta-learning 或者 learning to learn**。

Meta Learning的实现方法并不单一，只要是具有快速学习功能的算法都是meta learning相关的方法，下面做了一个从方法类型和研究领域做了一个大概的归类。

### Metric Learning
度量学习的方法是对样本间距离分布进行建模，使得属于同类样本靠近，异类样本远离。简单地，我们可以采用无参估计的方法，如KNN。KNN虽然不需要训练，但效果依赖距离度量的选取。但目前比较好的方法是通过学习一个端到端的最近邻分类器，它同时受益于带参数和无参数的优点，使得不但能快速的学习到新的样本，而且能对已知样本有很好的泛化性。

##### Siamese Neural Network

比较早的工作是15年NIPS的 Siamese neural network[16]，如Fig 1所示，输入分为两个部分，其中一个输入是support set中的数据，另外一个是要测试的数据，最后输出两个输入是同一个类型的概率，遍历所有support set后，概率大的为预测类型。
<div align=center> <img src="https://calebchen-1256449519.cos.ap-guangzhou.myqcloud.com/18.04/Investigation_meta_learning_2.png"  alt=" " width="50%"/> </div>
<div align=center>Fig 1. Siamese neural network</div>

<div align=center> <img src="https://calebchen-1256449519.cos.ap-guangzhou.myqcloud.com/18.04/Investigation_meta_learning_1.png"  alt=" " width="70%"/>  </div>
<div align=center>Fig 2. Training and Testing strategy </div>

Siamese neural network较早得提出了在feature map层面对比图片来提升few-shot问题的性能，比较的功能也是由网络学习获得的。

17ICCV上Facebook发表的[18]也是通过先学习feature map层面的特征，再通过网络学习feature map的差异来解决few-shot learning问题。

##### Matching Network

基于比较的思想，[1]提出了更加合理的Matching方案，作者引入了attention机制和memory机制。

他们把问题建模成：

$$
\hat{y}=\sum_{i=1}^{k}a(\hat{x},x_i)y_i
$$

其中$S= \{ (x_i, y_i) \}_{i=1}^k$为support set，$a$为attension机制。

这里的attention机制是先将输入$\hat{x}$与support set的$x$分别做embeding（分别用$f,g$表示），然后计算cosine distance $c$，再输入到softmax中归一化到0-1，即下式所示：

$$
a(x,\hat{x})=\frac{e^{c(f(\hat{x}),g(x))}}{\sum_{j=1}^{k} e^{c(f(\hat{x}),g(x))}}
$$

attention机制可以理解为最终对于距离$\hat{x}$的$x$的响应会更多得被考虑，那么这样embeding操作就决定了最后attention机制的响应。作者认为对于单个样本$x,\hat{x}$的embeding都应该考虑support set $S$的情况。所以$f(\hat{x}),g(x)$修改为$f(\hat{x},S),g(x,S)$。

- 对于$g(x,S)$，作者将support set中的样本看作一个序列，利用BiLSTM作为embeding的网络结构，对每一个$x_i$进行编码。
- 对于$f(\hat{x},S)$，作者利用一个带有attention的LSTM结构（具体细节看论文Appendix）

这种embeding使得网络能够忽视一些不重要的类别，在性能上也得到了很大的提升，也是该领域Matching网络的基础。