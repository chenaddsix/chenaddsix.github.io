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