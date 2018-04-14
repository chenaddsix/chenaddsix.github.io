---
layout: post
title: Prioritied
subtitle:   "\"Prioritied Experience Replay.\""
date: 2018-04-14
header-img: "img/post-bg-2015.jpg"
author: "Jiale Chen"
catalog: true
tags:
  - Reinforcement Learning
---

<script type="text/javascript" async src="//cdn.bootcss.com/mathjax/2.7.0/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_CHTML"></script>

为了构建自己的代码库，从本周开始每周实现一个到两个经典的强化学习算法，并做一定笔记对该算法进行记录。

这周记录的算法是Prioritied Experience Replay[1]，该算法是Google在16年提出的，基本思想很简单，就是为了增加对TD-error比较大的experience的训练次数，得以让网络更好拟合环境的Reward空间。
最简单的方法是将experience按照TD-error进行排序，然后优先训练TD-error较大的experience。但这样的方法很容易导致一些TD-error较小的experience很难被训练到，容易导致过拟合。

Google提出的方法是利用了线段树的存储和查找方式。这个方法之前在数据结构中没有遇到，所以本文更多的笔墨会说明这种存储方式。
先看算法的完整伪代码：




