---
layout: post
title: 分解机
subtitle:   "\"Factorization Machines.\""
date: 2018-03-24
header-img: "img/post-bg-2015.jpg"
author: "Jiale Chen"
catalog: false
tags:
  - Mechine Learning
---

<script type="text/javascript" async src="//cdn.bootcss.com/mathjax/2.7.0/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_CHTML"></script>

<center> <img src="https://github.com/starfolder/MarkdownPic/blob/Razor_Atmel/FM1.png?raw=true"  alt=" " /> </center>
<center> <img src="https://github.com/starfolder/MarkdownPic/blob/Razor_Atmel/FM2.png?raw=true"  alt=" " /> </center>
<center> <img src="https://github.com/starfolder/MarkdownPic/blob/Razor_Atmel/FM3.png?raw=true"  alt=" " /> </center>
<center> <img src="https://github.com/starfolder/MarkdownPic/blob/Razor_Atmel/FM4.png?raw=true"  alt=" " /> </center>
<center> <img src="https://github.com/starfolder/MarkdownPic/blob/Razor_Atmel/FM5.png?raw=true"  alt=" " /> </center>

> 在基于Model-Based的协同过滤中，一个rating矩阵可以分解为user矩阵和item矩阵，每个user和item都可以采用一个隐向量表示。如下图所示。
> <center> <img src="https://github.com/starfolder/MarkdownPic/blob/Razor_Atmel/ffm_mf.png?raw=true", alt=" "/> </center>
> 上图把每一个user表示成了一个二维向量，同时也把item表示成一个二维向量，两个向量的内积就是矩阵中user对item的打分。

<center> <img src="https://github.com/starfolder/MarkdownPic/blob/Razor_Atmel/FM6.png?raw=true"  alt=" " /> </center>
<center> <img src="https://github.com/starfolder/MarkdownPic/blob/Razor_Atmel/FM7.png?raw=true"  alt=" " /> </center>

解读第（1）步到第（2）步，这里用A表示系数矩阵V的上三角元素，B表示对角线上的交叉项系数。由于系数矩阵V是一个对称阵，所以下三角与上三角相等，有下式成立：
$$ A = \frac{1}{2} (2A+B) - \frac{1}{2} B.  \quad \underline{ A=\sum_{i=1}^{n} \sum_{j=i+1}^{n} {\langle \mathbf{v}_i, \mathbf{v}_j \rangle} x_i x_j } ; \quad \underline{ B = \frac{1}{2} \sum_{i=1}^{n} {\langle \mathbf{v}_i, \mathbf{v}_i \rangle} x_i x_i } \quad (fm) $$

<center> <img src="https://github.com/starfolder/MarkdownPic/blob/Razor_Atmel/FM8.png?raw=true"  alt=" " /> </center>
<center> <img src="https://github.com/starfolder/MarkdownPic/blob/Razor_Atmel/FM9.png?raw=true"  alt=" " /> </center>
<center> <img src="https://github.com/starfolder/MarkdownPic/blob/Razor_Atmel/FM10.png?raw=true"  alt=" " /> </center>
<center> <img src="https://github.com/starfolder/MarkdownPic/blob/Razor_Atmel/FM11.png?raw=true"  alt=" " /> </center>
<center> <img src="https://github.com/starfolder/MarkdownPic/blob/Razor_Atmel/FM12.png?raw=true"  alt=" " /> </center>
<center> <img src="https://github.com/starfolder/MarkdownPic/blob/Razor_Atmel/FM13.png?raw=true"  alt=" " /> </center>
<center> <img src="https://github.com/starfolder/MarkdownPic/blob/Razor_Atmel/FM14.png?raw=true"  alt=" " /> </center>
<center> <img src="https://github.com/starfolder/MarkdownPic/blob/Razor_Atmel/FM15.png?raw=true"  alt=" " /> </center>
<center> <img src="https://github.com/starfolder/MarkdownPic/blob/Razor_Atmel/FM16.png?raw=true"  alt=" " /> </center>
<center> <img src="https://github.com/starfolder/MarkdownPic/blob/Razor_Atmel/FM17.png?raw=true"  alt=" " /> </center>

### Field-aware Factorization Machine 

场感知分解机（Field-aware Factorization Machine ，简称FFM）最初的概念来自于Yu-Chin Juan与其比赛队员，它们借鉴了辣子Michael Jahrer的论文中field概念，提出了FM的升级版模型。

通过引入field的概念，FFM吧相同性质的特征归于同一个field。在FM开头one-hot编码中提到用于访问的channel，编码生成了10个数值型特征，这10个特征都是用于说明用户PV时对应的channel类别，因此可以将其放在同一个field中。那么，我们可以把同一个categorical特征经过one-hot编码生成的数值型特征都可以放在同一个field中。

在FFM中，每一维特征$$x_i$$，针对其它特征的每一种”field” $$f_j$$，都会学习一个隐向量$$v_i,f_j$$。因此，隐向量不仅与特征相关，也与field相关。
假设每条样本的n个特征属于$$f$$个field，那么FFM的二次项有$nf$个隐向量。而在FM模型中，每一维特征的隐向量只有一个。因此可以吧FM看作是FFM的特例，即把所有的特征都归属到一个field是的FFM模型。根据FFM的field敏感特性，可以导出其模型表达式： \

$$
\hat{y}(\mathbf{x}) := w_0 + \sum_{i=1}^{n} w_i x_i + \sum_{i=1}^{n} \sum_{j=i+1}^{n} \langle \mathbf{v}_{i,\,f_j}, \mathbf{v}_{j,\,f_i} \rangle x_i x_j \qquad(ffm)
$$

其中，$$f_j$$是第$$j$$个特征所属的field。如果隐向量的长度为$$k$$，那么FFM的二交叉项参数就有$$nfk$$个，远多于FM模型的$$nk$$个。此外，由于隐向量与field相关，FFM的交叉项并不能够像FM那样做化简，其预测复杂度为$$O(kn^2)$$。

给出一下输入数据:  
|User|Movie|Genre|Price|  
|:--------|---------:|:-------:|---------:|  
|YuChin | 3Idiots | Comedy, Drama | $9.99|
Price是数值型特征，实际应用中通常会把价格划分为若干个区间（即连续特征离散化），然后再one-hot编码，这里假设$9.99对应的离散化区间tag为”2”。当然不是所有的连续型特征都要做离散化，比如某广告位、某类广告／商品、抑或某类人群统计的历史CTR（pseudo－CTR）通常无需做离散化。
该条记录可以编码为5个数值特征，即User^YuChin, Movie^3Idiots, Genre^Comedy, Genre^Drama, Price^2。其中Genre^Comedy, Genre^Drama属于同一个field。为了说明FFM的样本格式，我们把所有的特征和对应的field映射成整数编号。
|Field Name | Field Index | Feature Name | Feature Index|
|:--------|---------:|:-------:|---------:|
|User | 1 | User^YuChin | 1|
Movie | 2 | Movie^3Idiots | 2
Genre	3 | Genre^Comedy | 3
－ |	－ | Genre^Drama | 4
Price | 4 | Price^2 | 5
那么，FFM所有的（二阶）组合特征共有10项 $$\mathbf{C}_{5}^{2} = \frac{5 \times 4}{ 2!}= 10$$, 即为：
<center> <img src="https://github.com/starfolder/MarkdownPic/blob/Razor_Atmel/ffm_samples.png?raw=true"  alt=" " /> </center>

Yu-Chin Juan实现了一个C++版的FFM模型，源码可从Github下载。这个版本的FFM省略了常数项和一次项，模型方程如下。
$$
\phi(\mathbf{w}, \mathbf{x}) = \sum_{j_1, j_2 \in \mathcal{C}_2} \langle \mathbf{w}_{j_1, f_2}, \mathbf{w}_{j_2, f_1} \rangle x_{j_1} x_{j_2} \label{eq:phi}\tag{5}
$$
其中，$$C_2$$是非零特征的二元组合，$$j_1$$ 是特征，属于field $$f_1，w_{j1,f2}$$ 是特征 $$j_1$$ 对field $$f_2$$ 的隐向量。此FFM模型采用logistic loss作为损失函数，和L2惩罚项，因此只能用于二元分类问题。  
$$
\min_{\mathbf{w}} \sum_{i=1}^L \log \big( 1 + \exp\{ -y_i \phi (\mathbf{w}, \mathbf{x}_i ) \} \big) + \frac{\lambda}{2} \| \mathbf{w} \|^2
$$
其中，$$yi\in{−1,1}$$ 是第 $$i$$ 个样本的label，$$L$$ 是训练样本数量，$$\lambda$$ 是惩罚项系数。模型采用SGD优化，优化流程如下。
<center> <img src="https://github.com/starfolder/MarkdownPic/blob/Razor_Atmel/ffm_samples.png?raw=true"  alt=" " /> </center>
