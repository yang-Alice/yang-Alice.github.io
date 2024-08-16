---
layout: post
title: My Journey Learning CUDA
subtitle:  Introduction to Parallel Computing
author: https://yang-alice.github.io/
categories: CUDA
banner:
  #video: https://vjs.zencdn.net/v/oceans.mp4
  loop: true
  volume: 0.8
  start_at: 8.5
  #image: /assets/images/leetcode/33/Figure_5.png
  opacity: 0.618
  background: "#000"
  height: "100vh"
  min_height: "38vh"
  heading_style: "font-size: 4.25em; font-weight: bold; text-decoration: underline"
  subheading_style: "color: gold"
tags: CUDA
sidebar: []
---
# My Journey Learning CUDA: Introduction to Parallel Computing

---

Hello, and welcome to my blog! My name is yang, and I am excited to share my journey of learning CUDA with you. In this series, I will be exploring the world of parallel computing, starting with the basics and diving into more advanced topics as we progress.

### Why CUDA?

As a C++ software developer, I have always been fascinated by the potential of parallel computing to speed up complex calculations and processes. When I came across CUDA, a parallel computing platform developed by NVIDIA, I knew I had to delve into this technology to expand my skills and knowledge.

### What to Expect

In this series, I will be covering a wide range of topics related to CUDA, including:

1. Introduction to parallel computing and GPU architecture
2. Setting up the development environment for CUDA programming
3. Basic CUDA programming concepts and syntax
4. Advanced CUDA techniques and optimization strategies
5. Real-world applications of CUDA in machine learning, image processing, and more

### Get Involved

I invite you to join me on this journey of exploration and discovery as we dive into the world of parallel computing with CUDA. Feel free to ask questions, share your own experiences, and engage with the content by leaving comments on the blog posts.

### Upcoming Posts

Stay tuned for the first post in this series, where we will explore the fundamentals of parallel computing and GPU architecture. I can't wait to share my knowledge and experiences with you as we learn and grow together in the exciting world of CUDA programming.

Thank you for joining me on this adventure. Let's start learning CUDA together!

---

# link
[NVIDIA's Official Tutorials: CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
 
 
官方文档及书籍
英文好、时间充裕的同学可以精读官方文档或者著作。

NVIDIA CUDA C++ Programming Guide
地址：
https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html

这是英伟达官方的CUDA编程教程，但是我英文一般，简单过了一遍之后感觉很多细节没讲，有一定的跳跃性，所以我看完还是很朦胧。

CUDA C++ Best Practices Guide
地址：
https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html

这也是英伟达官方的CUDA编程教程，不过侧重点在实践方面，比如如何编程才能最大化利用GPU特性提升性能，建议基础打好之后再来看这个。

CUDA C编程权威指南
这么经典的书就不用我多说了，英文原版叫《Professional CUDA C Programming》，pdf地址
http://www.hds.bme.hu/~fhegedus/C++/Professional%20CUDA%20C%20Programming.pdf

个人博客
像我这种英文差、想快速入门的只能找找中文博客看看了，还是找到不少非常奈斯的教程的。

谭升的博客（强推！！！）
地址：
https://face2ai.com/program-blog/#GPU编程（CUDA）

这是我最近发现的又一个宝藏博主，看完他的GPU编程系列教程后感觉豁然开朗，很多底层的原理和细节都通彻了，强烈安利！

他在github还开源了教程对应的示例代码：
https://github.com/Tony-Tan/CUDA_Freshman

CUDA编程入门极简教程
地址：
https://zhuanlan.zhihu.com/p/34587739

速览即可，看完就会写最简单的CUDA代码了。

《CUDA C Programming Guide》《CUDAC编程指南》
导读
地址：
https://zhuanlan.zhihu.com/p/53773183

这是NVIDIA CUDA C++ Programming Guide和《CUDA C编程权威指南》两者的中文解读，加入了很多作者自己的理解，对于快速入门还是很有帮助的。但还是感觉细节欠缺了一点，建议不懂的地方还是去看原著。

CUDA编程入门系列
地址：
https://zhuanlan.zhihu.com/p/97044592

这位大佬写了六篇，主要是通过一个简单的加法的例子，一步步讲了CUDA优化的若干种方法，拿来上手实践一下还是很棒的。

CUDA编程系列
地址：
https://blog.csdn.net/sunmc1204953974/article/details/51000970

这个系列写的也是很全了，十几篇，建议快速通读一下。

开源代码
有很多的CUDA源码可以供我们慢慢学习，我这就简单给几个典型的Transformer系列的加速代码了。

LightSeq
地址：
https://github.com/bytedance/lightseq

这是字节跳动开源的生成模型推理加速引擎，BERT、GPT、VAE等等全都支持，速度也是目前业界最快的之一。

FasterTransformer
地址：
https://github.com/NVIDIA/DeepLearningExamples/tree/master/FasterTransformer

这是英伟达开源的Transformer推理加速引擎。

TurboTransformers
地址：
https://github.com/Tencent/TurboTransformers

这是腾讯开源的Transformer推理加速引擎。

DeepSpeed
地址：
https://github.com/microsoft/DeepSpeed

这是微软开源的深度学习分布式训练加速引擎。



Official documentation and books
Students with good English skills and sufficient time can meticulously read official documentation or books.

NVIDIA CUDA C++ Programming Guide
Link:
https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html

This is NVIDIA's official CUDA programming tutorial, but for those with average English proficiency like me, after a quick read, many details may seem vague, with some topics appearing disjointed, leaving a somewhat blurry understanding.

CUDA C++ Best Practices Guide
Link:
https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html

Also an official CUDA programming guide by NVIDIA, this one emphasizes practical aspects, focusing on how to program to maximize GPU utilization for enhanced performance. It's advisable to have a solid foundation before delving into this.

CUDA C Programming: An Authoritative Guide
This classic book needs no elaborate introduction. The English original version is titled "Professional CUDA C Programming," and the PDF link is provided below. If the loading is slow, you can reply with "cuda" in the background to receive the PDF file:
http://www.hds.bme.hu/~fhegedus/C++/Professional%20CUDA%20C%20Programming.pdf

Personal Blogs
For individuals with limited English proficiency seeking quick entry into the subject, exploring Chinese blogs might prove valuable. I came across some excellent tutorials.

Recommended Blog by Tan Sheng
Link:
https://face2ai.com/program-blog/#GPU编程（CUDA）

I recently discovered this gem of a blogger. After studying his GPU programming series, I found a newfound clarity as he thoroughly explains many underlying principles and details. Highly recommended!

He has also open-sourced the tutorial's accompanying sample code on GitHub:
https://github.com/Tony-Tan/CUDA_Freshman

Ultra-Concise Guide to CUDA Programming
Link:
https://zhuanlan.zhihu.com/p/34587739

A quick read is sufficient; after going through it, you'll be able to write basic CUDA code.

Guided Readings of "CUDA C Programming Guide" and "CUDA C Programming: An Authoritative Guide"
Link:
https://zhuanlan.zhihu.com/p/53773183

This provides a Chinese interpretation of NVIDIA's CUDA C++ Programming Guide and "CUDA C Programming: An Authoritative Guide,” incorporating the author's insights. It's quite helpful for a rapid start, yet some details are still lacking, so it's recommended to consult the original texts for areas of confusion.

Introduction to CUDA Programming Series
Link:
https://zhuanlan.zhihu.com/p/97044592

This expert has written six pieces, mainly illustrating various CUDA optimization methods through a simple addition example, making it excellent for hands-on practice.

CUDA Programming Series
Link:
https://blog.csdn.net/sunmc1204953974/article/details/51000970

This comprehensive series, comprising over ten articles, is best skimmed through quickly.

Open Source Code
There are numerous CUDA source codes available for gradual learning. Here are a few notable examples of Transformer series acceleration codes.

LightSeq
Link:
https://github.com/bytedance/lightseq

This is Bytedance's open-source engine for inference acceleration in generative models, supporting BERT, GPT, VAE, and more, known for its leading industry speed.

FasterTransformer
Link:
https://github.com/NVIDIA/DeepLearningExamples/tree/master/FasterTransformer

This is NVIDIA's open-source Transformer inference acceleration engine.

TurboTransformers
Link:
https://github.com/Tencent/TurboTransformers

This is Tencent's open-source Transformer inference acceleration engine.

DeepSpeed
Link:
https://github.com/microsoft/DeepSpeed

This is Microsoft's open-source engine for accelerating distributed deep learning training.
https://godweiyang.com/2021/01/25/cuda-reading/

https://developer.nvidia.com/blog/even-easier-introduction-cuda/