---
layout: post
published: true
title: 'HyPar: Towards Hybrid Parallelism for Deep Learning Accelerator Array'
subtitle: Paper Review
date: '2024-02-26'
tags:
  - ML Acceleration
  - Deep Learning
  - Parallel Computing
  - Distributed Processing
---


# HyPar: Towards Hybrid Parallelism for Deep Learning Accelerator Array



## I. Introduction

The rapid ascent of artificial intelligence, particularly Deep Neural Networks (DNNs), has revolutionized various fields by offering remarkable accuracy, scalability, and adaptability. DNNs form the backbone of numerous cutting-edge applications, ranging from facial recognition systems and speech processing to scene parsing, showcasing their versatility and effectiveness. Given their computational and memory-intensive nature, DNNs present significant challenges, straining the conventional Von Neumann architecture where computational logic and data storage are distinctly separated. This separation often leads to a bottleneck in performance and energy efficiency due to the extensive data movement required between processing units and memory stores. 

Acknowledging these challenges, there has been a concerted effort within both academic and industrial spheres to enhance the performance and energy efficiency of DNNs through hardware acceleration. Notably, advancements in this domain have seen the development of specialized DNN accelerators and neuro-processors by leading tech companies, alongside innovative architectural solutions like spatial architectures and in-memory processing in academia. Despite these efforts, the training of DNNs, particularly large models, remains an area less explored, plagued by the inefficiencies of off-chip memory accesses and a lack of effective parallelism strategies. 

HyPar emerges as a solution to these challenges, proposing a novel approach to optimize layer-wise parallelism in DNN training across an array of accelerators. By focusing on partitioning and communication optimization, HyPar aims to minimize the data movement and enhance the performance and energy efficiency of DNN training, marking a significant step forward in the quest for more capable and efficient DNN training methodologies.



## II. Background and Motivation

The drive to enhance Deep Neural Networks (DNNs) transcends the mere pursuit of computational efficiency; it is fundamentally about harnessing the potential of DNNs to solve complex, real-world problems across various domains. The advent of DNNs has introduced a paradigm shift in how data is processed, analyzed, and understood, offering unprecedented accuracy in tasks like image and speech recognition. However, the computational and memory demands of training these sophisticated models challenge traditional computing architectures. The separation of processing and memory in these architectures often leads to significant inefficiencies, primarily due to the extensive data movement required.

The quest for solutions has led to a surge in the development of specialized hardware accelerators, designed to meet the unique demands of DNN training and inference. These accelerators aim to reduce the reliance on off-chip memory and optimize data flow within the computing architecture, thereby improving both performance and energy efficiency. Despite these advancements, the training of large-scale DNNs remains a significant hurdle, highlighting the need for innovative approaches that can effectively leverage hardware capabilities to support the complex, iterative process of DNN training.

Enter the realm of parallel computing strategies—where the potential for accelerating DNN training lies in the efficient partitioning of tasks across multiple processors. Yet, the challenge persists in balancing the load across these processors while minimizing the communication overhead, ensuring that the parallelization efforts lead to tangible improvements in training speed and efficiency. It is within this context that the proposal of HyPar becomes particularly relevant, offering a new approach to optimize the distribution of DNN training tasks across an array of accelerators. By minimizing data movement and strategically managing the communication between processors, HyPar aims to address the core challenges of DNN training, paving the way for more efficient and scalable solutions.



## III. HyPar: Concept and Design

The authors of the paper introduce HyPar, a novel approach aimed at optimizing layer-wise parallelism for deep neural network (DNN) training across an array of DNN accelerators. HyPar stands out by partitioning various tensor types crucial in DNN operations—feature map tensors, kernel tensors, gradient tensors, and error tensors—across the accelerators. This partitioning strategy revolves around the choice of parallelism for each weighted layer within a network, targeting the reduction of total communication during the training process.

A core component of HyPar's methodology is its unique communication model, meticulously designed to quantify and minimize the communication overhead inherent in DNN training. This model explicates the sources and quantities of inter-accelerator communication triggered by the distributed tensor operations, providing a framework to systematically address the communication challenges.

To navigate the complex optimization landscape of tensor partitioning across layers and accelerators, the authors employ a hierarchical layer-wise dynamic programming algorithm. This algorithm is both efficient, with a linear time complexity for partition searches, and practical, ensuring scalability and adaptability to various DNN architectures and sizes. By applying HyPar in an HMC-based DNN training architecture, the solution notably reduces data movement, a critical factor in enhancing training performance and energy efficiency.

The effectiveness of HyPar is underscored through evaluations with ten DNN models, revealing significant gains in performance and energy efficiency compared to conventional data and model parallelism strategies. These findings highlight HyPar's potential to advance the state-of-the-art in DNN training acceleration, addressing the pressing challenges of large model sizes and the inefficiencies of existing parallelism approaches.

![]({{site.baseurl}}/img/hypar-1.png)

### Data Parallelism and Model Parallelism Illustrated

The authors of the paper provide concrete examples to elucidate the concepts of data parallelism and model parallelism, two pivotal types of parallelism used in DNN training. In data parallelism, each accelerator holds a complete copy of the model (kernel or weight matrix) but processes a unique portion of the input data. This approach is apt for scenarios where neural networks are convolution-rich, as it allows for the efficient distribution of computational workload across multiple accelerators without necessitating significant inter-accelerator communication for the forward pass and error backward propagation.

Conversely, model parallelism involves partitioning the model itself across accelerators, with each accelerator responsible for a distinct segment of the model while all accelerators work on the same input data. This method is particularly useful for networks with large models, reducing the memory footprint on each accelerator but requiring more communication for the assembly of intermediate outputs and gradients.

The examples provided in the paper serve to illustrate these concepts vividly. For instance, considering a fully-connected layer with a specific number of input and output neurons, the authors show how data and model parallelism would partition the workload between two accelerators, highlighting the communication needs and computational strategies for each type of parallelism.

### Algorithm for Partitioning Between Accelerators

The partitioning algorithm presented by the authors, Algorithm 1, aims to optimize the parallelism strategy for each layer in a DNN to minimize total communication during training. This algorithm operates under the premise that the choice between data parallelism and model parallelism can significantly affect the performance and efficiency of DNN training on multiple accelerators. It takes into account the number of weighted layers in a DNN model, the batch size, and a list of hyperparameters such as layer types (convolutional or fully connected), kernel sizes, and activation functions.

The core of the algorithm involves iterating through each layer of the DNN and calculating the communication costs associated with data parallelism and model parallelism, based on pre-defined tables that quantify intra-layer and inter-layer communication amounts for different parallelism strategies. By dynamically programming these costs layer by layer, the algorithm determines the optimal partition strategy that minimizes total communication.

This partitioning logic is critical in scenarios where different layers of a DNN might benefit from different parallelism strategies due to their unique computational and memory requirements. By systematically evaluating and selecting the parallelism strategy for each layer, the algorithm ensures efficient utilization of accelerator resources, significantly reducing the communication overhead that can bottleneck the training process.

![]({{site.baseurl}}/img/hypar-2.png)



## IV. Implementation of HyPar

The implementation of HyPar, as delineated by the authors, presents a nuanced approach to the partitioning and management of computational tasks across an array of DNN accelerators. This process is essential for optimizing the layer-wise parallelism within deep neural networks (DNNs), thus addressing the dual challenges of minimizing communication overhead and maximizing computational efficiency.

#### Partition Between Two Accelerators

At the heart of HyPar's implementation is the principle that each DNN layer can be configured with either data parallelism or model parallelism. The decision hinges on a strategic evaluation of communication costs—both intra-layer and inter-layer. A dynamic programming method, specifically designed for this task, facilitates the search for optimal partitions layer by layer, maintaining a linear time complexity relative to the number of weighted layers in the network. The pseudocode provided (Algorithm 1) outlines this process, beginning with the initialization of communication costs and parallelism lists and proceeding through each layer to compute and update these values based on the calculated communication costs.

This algorithmic approach enables a systematic evaluation of parallelism strategies for each layer, aiming to minimize the total communication overhead incurred during DNN training. By considering both data and model parallelism, HyPar ensures that the partitioning strategy is precisely tailored to the characteristics of each layer, thereby optimizing the use of accelerator resources.

#### Hierarchical Partition

To extend the partitioning scheme to an array of accelerators, HyPar employs a hierarchical approach. This method conceptualizes the network of accelerators as a binary tree, where each node represents a group of accelerators that can be further divided. This hierarchical partitioning allows for the efficient distribution of computational tasks across a larger array of accelerators, significantly enhancing scalability and flexibility.

The hierarchical partition algorithm (Algorithm 2) recursively applies the partitioning logic to increasingly granular subdivisions of the accelerator array. This process effectively manages the complexity of partitioning across multiple accelerators, ensuring that communication overhead is minimized at every level of the hierarchy. The result is a highly optimized configuration that leverages both data and model parallelism to achieve superior performance and energy efficiency in DNN training.



![]({{site.baseurl}}/img/hypar-3.png)![]({{site.baseurl}}/img/hypar-4.png)



## V. HyPar Architecture

The architecture proposed by the authors for HyPar is centered around a sophisticated array of accelerators, each based on the Hybrid Memory Cube (HMC) technology. This choice is strategic, as HMC integrates stacked DRAM dies with a logic die, interconnected through Silicon Vias (TSVs), offering an impressive memory bandwidth of up to 320 GB/s. This bandwidth is crucial for DNN accelerators due to their intense memory access patterns and computational demands. Processing Units (PUs) are integrated into the logic die, allowing for in-memory processing, which significantly reduces data movement—a major bottleneck in traditional computing architectures.

#### Row Stationary Design

For the processing units within each accelerator, the authors adopt a row stationary design. This design choice is aimed at optimizing convolutional computations, which are prevalent in DNNs. In a row stationary design, weight rows are shared horizontally across processing engines, feature map rows are shared diagonally, and partial sum rows are accumulated vertically. This arrangement ensures efficient data reuse and minimizes memory accesses, which is vital for enhancing computational efficiency and reducing energy consumption.

#### 2-D Array of Accelerators

The overall architecture is composed of a 2-D array consisting of sixteen accelerators. This design not only leverages the high memory bandwidth provided by HMC but also facilitates a scalable and efficient parallelism framework as dictated by the HyPar algorithm. The partitioning strategy determined by HyPar dictates the parallelism setting across this array, optimizing the distribution of computational tasks to minimize communication overhead and maximize performance.

#### Network Topologies: H Tree vs. Torus

The authors explore two network topologies for connecting the accelerators within the array: the H tree and the torus topology. The H tree topology aligns with the binary tree pattern of the hierarchical partitioning approach, naturally supporting the communication patterns between accelerator subarrays as defined by HyPar. On the other hand, while the torus topology is a common choice for connecting processors in parallel computing, it does not align as closely with the partition patterns generated by HyPar, leading to less efficient communication.

The H tree topology, with its capability to match the communication patterns necessitated by the hierarchical partitioning of HyPar, emerges as the more effective configuration. It supports efficient hierarchical communication, allowing accelerator subarrays to communicate effectively, mirroring the binary tree structure inherent in the HyPar algorithm. This topology choice underscores the architecture's focus on minimizing data movement and communication overhead, which are pivotal for achieving high performance and energy efficiency in DNN training.
