---
layout: post
published: true
title: Distributed Deep Learning Using Synchronous Stochastic Gradient Descent
subtitle: Paper Review
date: '2024-02-26'
tags:
  - ML Acceleration
  - SGD
  - Deep Learning
  - CNN
  - Distributed Processing
---


# Distributed Deep Learning Using Synchronous Stochastic Gradient Descent

## Introduction

In the landscape of deep learning, the capacity to efficiently scale up the training of large neural networks is a pivotal challenge. This paper tackles the optimization of synchronous Stochastic Gradient Descent (SGD), a cornerstone algorithm for distributed deep learning. The authors embark on an exploration to enhance the scalability of training processes without the need to modify hyperparameters, compress data, or alter the algorithmic essence. By delving into a comprehensive analysis of scaling behaviors, the paper identifies optimal design points for various network architectures, showcasing significant improvements in training throughput on large-scale computational resources. The authors demonstrate the feasibility of scaling a conventional CNN training session by up to **90 times across 128 nodes**, alongside similar scalability achievements for other network types on **a 64-node cluster**. The paper further underscores the **versatility** of their approach by presenting notable scaling efficiencies on different hardware configurations, including a 14X scaling on an Ethernet-based cluster. This inquiry into synchronous SGD not only offers insights into achieving then (2016) record training throughputs but also paved the way for more accessible and efficient deep learning training methodologies across varied computational environments.



## Background and Related Work

The advancement of distributed deep learning is primarily driven by the escalating computational demands of training large neural networks. The quest is to accelerate training times from days to hours or even minutes, despite these networks requiring substantial computational resources. The traditional approach, utilizing single nodes or GPUs, is increasingly inadequate for the task, leading to the exploration of distributed training methods.

Frameworks such as TensorFlow, FireCaffe, and DeepImage have emerged to support multi-node or multi-GPU training environments. However, scaling synchronous Stochastic Gradient Descent (SGD), a fundamental algorithm in neural network training, presents a significant challenge. Synchronous SGD distributes the workload of processing a minibatch of data points across multiple nodes. Yet, many existing frameworks find it difficult to scale this process effectively beyond a limited number of nodes.

In response to these challenges, various adaptations of synchronous SGD have been developed, including 1-bit SGD, elastic-SGD, and asynchronous SGD. These methods aim to improve scalability by modifying the algorithm, adjusting hyperparameters, or implementing data compression techniques. Unlike these approaches, the focus of this paper is on optimizing the original synchronous SGD algorithm to enhance scalability without altering its fundamental principles or resorting to data compression.

The approach involves a detailed analysis to understand the balance between computation and communication in distributed systems. The paper explores **data parallelism**, **model parallelism**, and introduces a new algorithm for **hybrid parallelism**. This analysis is crucial for identifying the most effective strategy for scaling distributed training across various neural network layers.

Optimizing single-node performance is a prerequisite for efficient distributed training. This involves maximizing the computational efficiency of a single node through **cache optimization**, **register blocking**, **threading**, and instruction sequencing. These optimizations are foundational for achieving high levels of efficiency on individual nodes, which, in turn, facilitates effective scaling in a distributed training environment.

The following sections set the groundwork for the exploration of a scalable, distributed multi-node synchronous SGD algorithm. The paper's thorough analysis aims to advance the field of distributed deep learning by enabling efficient training of large neural networks across multiple computational nodes without compromising the integrity of the original SGD algorithm.



## Optimizing Computation in Neural Network Training

The "Optimizing Computation" section of the paper delves into several critical strategies aimed at enhancing the computational efficiency of neural network training. These strategies cover a wide spectrum, from optimizing the computation patterns of convolutional and fully-connected layers to advanced cache blocking, data layout optimization, vectorization, register blocking, and effective threading and work partitioning. Each of these strategies contributes to the overall goal of maximizing computational throughput and efficiency, critical for scaling deep learning algorithms.



Neural network training is computationally intensive, with convolutional and fully-connected layers being particularly compute-heavy. These layers involve operations on multi-dimensional tensors, transforming inputs into outputs through a series of complex mathematical manipulations. The paper outlines these operations as $2k+2$-dimensional loops for forward and backward propagation and weight gradient determination, with $2k$ denoting the dimensions of a feature map or kernel. The additional two dimensions represent minibatches and feature identifiers, or input-output feature pairs for weights.



**Cache Blocking**

Cache blocking is a technique designed to maximize the utilization of the cache memory by organizing the data access patterns in such a way that once data is loaded into the cache, it can be reused multiple times before being evicted. This method reduces the frequency of memory accesses to slower, external DRAM, thereby minimizing latency and improving overall performance. The paper highlights the importance of traversing along consecutive blocks in any dimension to achieve memory reuse and better Bytes/FLOPs (B/F) ratios, crucial for computational layers like convolutional layers where data reuse is significant. By performing a brute-force state space search over all values of loop iterators, the paper outlines an approach to find the minimum B/F ratio for different 2-D convolutional layers, given a limit on cache size.



**Data Layout and Vectorization**

Vectorization involves reorganizing data and computations to leverage Single Instruction, Multiple Data (SIMD) capabilities of modern CPUs, where a single operation can be performed on multiple data points simultaneously. The paper discusses laying out data in a manner that aligns with SIMD-width for output feature maps, ensuring that operations like multiply-and-accumulate can be efficiently vectorized. This not only improves the utilization of cache lines and bandwidth but also enhances prefetcher performance, significantly speeding up computations.



**Register Blocking**

Register blocking aims to optimize the use of CPU registers by carefully organizing data in registers to minimize load/store operations and maximize the computational throughput. The technique improves the ratio of vector fused multiply-and-add (VFMA) operations to load/store operations and is essential for hiding the latency of these instructions. By adjusting the register block size, the paper ensures that a sequence of VFMA instructions can be executed without delays, further boosting the performance of forward propagation operations in neural networks.



**Threading and Work Partitioning**

Fine-grained threading and work partitioning strategies are critical for leveraging multi-core CPUs effectively. The paper describes how to divide the computation workload across multiple threads, ensuring balanced utilization of processing resources. This involves partitioning tasks in a way that matches the hardware's parallel processing capabilities, thereby reducing execution time and improving overall computational efficiency.



### Optimizing Communication in Distributed Deep Learning

The section on optimizing communication delves into the strategies and mathematical formulations necessary for minimizing communication overhead in distributed deep learning, particularly focusing on synchronous stochastic gradient descent (SGD) algorithms. It covers the core challenges of data parallelism, model parallelism, and introduces a hybrid approach that aims to leverage the best of both worlds. Furthermore, it outlines deep learning communication primitives that facilitate efficient multi-node data transfer.



**Data Parallelism**

Data parallelism involves dividing the workload across minibatches and distributing these across multiple nodes. The primary mathematical formulation presented for a convolutional layer considers the amount of computation (����*C**o**m**p*) in FLOPS required for forward, backward, and weight gradient computation steps. This is given by:

����=3×2×������×���×���×��×�ℎ×����×���ℎ*C**o**m**p*=3×2×*M**B**n**o**d**e*×*i**f**m*×*o**f**m*×*k**w*×*kh*×*o**u**tw*×*o**u**t**h*

where ������*M**B**n**o**d**e* represents the number of data points assigned to a node, ���*i**f**m* and ���*o**f**m* are the number of input and output feature maps, ��*k**w* and �ℎ*kh* are the kernel width and height, and ����*o**u**tw* and ���ℎ*o**u**t**h* are the dimensions of the output feature map.

The total communication per iteration (����*C**o**mm*) for a data-parallel approach, where each node sends partial weight gradients to other nodes and receives updated weights, is expressed as:

����=��������×���×���×��×�ℎ×(2−�������)*C**o**mm*=*s**i**z**e**d**a**t**a*×*i**f**m*×*o**f**m*×*k**w*×*kh*×(2−*o**v**er**l**a**p*)

The equation emphasizes the impact of software optimization for overlapping send and receive operations to improve efficiency.



**Analyzing Model Parallelism**

Model parallelism entails partitioning the model across different nodes, such that each node computes a portion of the forward pass, backward pass, or weight gradient update. The communication requirements for model parallelism are less straightforward but generally involve transferring intermediate data between nodes to ensure consistency and correctness of the model updates.



**Analyzing Hybrid Parallelism**

The hybrid parallelism approach combines data and model parallelism by partitioning both along the minibatch and the feature map dimensions. It aims to optimize communication volumes by adjusting the partitioning scheme based on the specific layer and network architecture being used. The mathematical optimization involves finding the minimum total communication volume by differentiating the expression for overall communication volume with respect to �*G* (the number of groups) and solving for the point where the derivative is zero.



**Deep Learning Communication Primitives**

The paper introduces two key communication primitives: part-reduce and part-broadcast, crucial for implementing the hybrid parallel approach. These operations are optimized for multi-node environments and are designed to efficiently aggregate and distribute data among nodes to minimize communication bottlenecks.

- **Part-reduce** operation involves a reduction over partial data computed locally on each node, followed by scattering the reduced data to all nodes in the group. This is essential for aggregating partial weight gradients during the SGD update step.
- **Part-broadcast** operation involves broadcasting a node's locally owned data segment to all other nodes in the group, ensuring each node has the complete updated model or data necessary for the next computation step.




