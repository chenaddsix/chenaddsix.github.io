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

在FFM中，每一维特征$x_i$，针对其它特征的每一种”field” $f_j$，都会学习一个隐向量$v_i,f_j$。因此，隐向量不仅与特征相关，也与field相关。
假设每条样本的n个特征属于$f$个field，那么FFM的二次项有$nf$个隐向量。而在FM模型中，每一维特征的隐向量只有一个。因此可以吧FM看作是FFM的特例，即把所有的特征都归属到一个field是的FFM模型。根据FFM的field敏感特性，可以导出其模型表达式：
$$
\hat{y}(\mathbf{x}) := w_0 + \sum_{i=1}^{n} w_i x_i + \sum_{i=1}^{n} \sum_{j=i+1}^{n} \langle \mathbf{v}_{i,\,f_j}, \mathbf{v}_{j,\,f_i} \rangle x_i x_j \qquad(ffm)
$$

