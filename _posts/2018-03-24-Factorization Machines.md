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
$ A = \frac{1}{2} (2A+B) - \frac{1}{2} B.  \quad \underline{ A=\sum_{i=1}^{n} \sum_{j=i+1}^{n} {\langle \mathbf{v}_i, \mathbf{v}_j \rangle} x_i x_j } ; \quad \underline{ B = \frac{1}{2} \sum_{i=1}^{n} {\langle \mathbf{v}_i, \mathbf{v}_i \rangle} x_i x_i } \quad (n.ml.1.9.4) $

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



