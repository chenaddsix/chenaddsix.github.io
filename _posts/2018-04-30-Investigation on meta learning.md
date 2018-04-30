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
 
 <center><img src="https://calebchen-1256449519.cos.ap-guangzhou.myqcloud.com/18.04/Investigation_meta_learning_2.png"  alt=" "  width="50%"/> </center>
<center>Fig 1. Siamese neural network</center>

<center> <img src="https://calebchen-1256449519.cos.ap-guangzhou.myqcloud.com/18.04/Investigation_meta_learning_1.png"  alt=" " width="70%"/>  </center>
<center>Fig 2. Training and Testing strategy </center>

Siamese neural network较早得提出了在feature map层面对比图片来提升few-shot问题的性能，比较的功能也是由网络学习获得的。

17ICCV上Facebook发表的[18]也是通过先学习feature map层面的特征，再通过网络学习feature map的差异来解决few-shot learning问题。

##### Matching Network

基于比较的思想，[1]提出了更加合理的Matching方案，作者引入了attention机制和memory机制。

他们把问题建模成：

$$$$
\hat{y}=\sum_{i=1}^{k}a(\hat{x},x_i)y_i
$$$$

其中$$S= \{ (x_i, y_i) \}_{i=1}^k$$为support set，$$a$$为attension机制。

这里的attention机制是先将输入$$\hat{x}$$与support set的$$x$$分别做embeding（分别用$$f,g$$表示），然后计算cosine distance $$c$$，再输入到softmax中归一化到0-1，即下式所示：

$$$$
a(x,\hat{x})=\frac{e^{c(f(\hat{x}),g(x))}}{\sum_{j=1}^{k} e^{c(f(\hat{x}),g(x))}}
$$$$

attention机制可以理解为最终对于距离$$\hat{x}$$的$$x$$的响应会更多得被考虑，那么这样embeding操作就决定了最后attention机制的响应。作者认为对于单个样本$$x,\hat{x}$$的embeding都应该考虑support set $$S$$的情况。所以$$f(\hat{x}),g(x)$$修改为$$f(\hat{x},S),g(x,S)$$。

- 对于$$g(x,S)$$，作者将support set中的样本看作一个序列，利用BiLSTM作为embeding的网络结构，对每一个$$x_i$$进行编码。
- 对于$$f(\hat{x},S)$$，作者利用一个带有attention的LSTM结构（具体细节看论文Appendix）

这种embeding使得网络能够忽视一些不重要的类别，在性能上也得到了很大的提升，也是该领域Matching网络的基础。

##### Prototypical Network
17年NIPS，[2]提出了一种Prototypical Network，该网络的思路很简单，对于每个类型学习一个原型表达(Prototy)，然后对于测试样本计算与这些原型的距离即可。如Fig 3所示

<center> <img src="https://calebchen-1256449519.cos.ap-guangzhou.myqcloud.com/18.04/Investigation_meta_learning_3.png"  alt=" " width="70%"/>  </center>
<center>Fig 3. Prototypical Network on few-shot and zero-shot scenarios </center>

伪代码如下：
代码中的$$d$$采用了Bregman divergence，实验证明效果优于cosine distance。
<center> <img src="https://calebchen-1256449519.cos.ap-guangzhou.myqcloud.com/18.04/Investigation_meta_learning_4.png"  alt=" " width="80%"/>  </center>

### Gradient Descent

基于梯度的方法中，最流行的meta learning方法如Google在16年发表的[3]，这类方法会构建一个meta learner和learner两个部分，meta learner负责学习一个更新learner的策略或者直接是网络参数

<center> <img src="https://calebchen-1256449519.cos.ap-guangzhou.myqcloud.com/18.04/Investigation_meta_learning_5.png"  alt=" " width="50%"/>  </center>
<center>Fig 4. LSTM Optimizer </center>

<center> <img src="https://calebchen-1256449519.cos.ap-guangzhou.myqcloud.com/18.04/Investigation_meta_learning_6.png"  alt=" " width="60%"/>  </center>
<center>Fig 5. Computational graph used for computing the gradient of the optimizer. </center>

如Fig 4所示，作者训练了一个两层的LSTM的优化器，输入是网络参数在Loss Function上梯度，然后Optimizer输出网络参数的增量。具体如Fig 5所示，这样其实相当于将learner的learning rate也省去了，直接学习对每个参数的数值。从实验结果上看，这样的方法甚至超过了例如adam,rmsprop等常用的优化器。另外类似的工作是17NIPS的hypernetwork[14]，方法也是用一个网络产生另外一个网络的权值，同时实验了CNN和RNN在meta learning上的效果。

而[15]将这样的框架应用到了few shot问题中并取得了不错的效果。具体算法如下，论文的使用的meta learning算法与[3]类似，同样是训练一个meta learner和一个用于分类的learner网络，不同之处是[15]把这种框架在few shot问题上做了扩展，算法的优化目标是得到一个对于few shot问题最优的meta learner，即最优的$$\Theta$$参数

<center> <img src="https://calebchen-1256449519.cos.ap-guangzhou.myqcloud.com/18.04/Investigation_meta_learning_11.png"  alt=" " width="70%"/>  </center>

[3],[14],[15]都是从deep learning的角度来求解最优的更新的策略，而[12]从强化学习的角度来解释更新策略的求解，算法模型如下：

<center> <img src="https://calebchen-1256449519.cos.ap-guangzhou.myqcloud.com/18.04/Investigation_meta_learning_10.png"  alt=" " width="70%"/>  </center>

作者将优化问题看作一个马尔可夫过程，并建模了优化问题一般的形式，把问题转化为学习一个优化策略$$\pi(f,\{x^{(0)},\cdots,x^{(i-1)}\})=-\gamma \nabla f(x^{(i-1)})$$，策略网络的输出即梯度下降的结果，也就是learner网络参数的增量。

除了上述meta learner和learner这种两个网络的模型，在17年ICML上UC Berkeley提出一种免模型的meta learning方法[4]，该方法不需要一个meta learner来指导learner的更新，他们的出发点是学习一个对于各种同类型任务都非常敏感的参数模型$$\theta$$，在这样一种参数情况下，对于任意任务的Loss Function都能很快收敛，可以认为是学习一个对于特定问题的最优网络初始化。具体算法如下：

<center> <img src="https://calebchen-1256449519.cos.ap-guangzhou.myqcloud.com/18.04/Investigation_meta_learning_7.png"  alt=" " width="50%"/>  </center>

该算法的思想是学习一个对于各种同质任务很敏感的初始化网络参数$$\theta$$，如Fig 6所示。这个网络参数可以在任意Loss Function的第一次迭代中快速收敛。算法流程：

- 首次构建一个同质任务的任务库$$T$$，可以理解为meta learning的数据集，用于后面采样。
- 然后中这些任务上做单次更新得到$${\theta}'$$并记录。
- 最后通过$${\theta}'$$计算meta learning的Loss Function，更新初始化参数$$\theta$$。

<center> <img src="https://calebchen-1256449519.cos.ap-guangzhou.myqcloud.com/18.04/Investigation_meta_learning_8.png"  alt=" " width="50%"/>  </center>
<center>Fig 6. Diagram of MAML. </center>

目前在meta learning领域最流行的框架就是一个meta learner和一个learner的模型，这样的结构相当于生成网络参数，这种结构优点是对于网络的每一个权值都能自适应一个特定的更新增量，但这种结构的对于很深的大规模网络需要训练很多meta learner，训练复杂度高。而[12]的方法在一定程度上解决了这个问题。这几篇论文的测试数据集都有不同，所以暂时没有性能对比。

### Memory-Augmented Network

Meta Learning领域一种基于记忆增强神经网络的方法，这种方法通过加入可读写的外部存储器层，实现用极少量新观测数据就能有效对模型进行调整，从而快速获得识别未见过的目标类别的meta-learning能力，也就是可以利用极少量样本学习。

[5]将这种记忆增强神经网络[18]用到了meta learning的领域，并提出了新的存储读写更新策略——LRUS(Least Recently Used Access)，每次写操作只选择最少被用到的存储位置或者最近被用的存储位置。这样的策略完全由内容决定，不依赖于存储的位置。

这种方式能够起作用的原因是网络的记忆功能用一个外部存储来代替，迫使网络去学习一些high-level的信息。

另外一篇很重要的记忆增强神经网络的论文[6]获得了17ICLR的oral，作者是李飞飞的学生。作者提出了一种用于深度学习的大规模终身记忆模块，利用了快速最近邻算法加快了查询效率。一个新的样本出现，先在memory中找到最近邻特征，如果类别不同，就在存放最久的样本中随机选取一个位置存放新的样本特征，因为这个样本更加rare；如果类别相同，则合并两个特征。具体做法如Fig 7。

<center> <img src="https://calebchen-1256449519.cos.ap-guangzhou.myqcloud.com/18.04/Investigation_meta_learning_9.png"  alt=" " width="70%"/>  </center>
<center>Fig 7. The operation of the memory module on a query q with correct value v. </center>

综上，[5]是记忆增强网络这few shot问题上的首次应用，论文创新性很高，直接将网络的记忆功能用一个可查询的存储结构代替。[6]在这个基础上对速度，匹配方法做了改进。这种方法的优势是可解释性比较强，网络的记忆部分不再只由LSTM的参数拟合，但缺点就是维护一个很大的memory在某些问题上成本很高，对速度要求高的场景也很难适应。

