---
layout: post
published: true
title: 'RecNMP: Accelerating Personalized Recommendation with Near-Memory Processing'
subtitle: Paper Review
date: '2024-03-18'
tags:
  - Machine Learning Acceleration
  - Personalized Recommendation
  - Software-Hardware Codesign
  - DNN
---
# RecNMP: Accelerating Personalized Recommendation with Near-Memory Processing

### Introduction & Motivation

In an age where digital content and products inundate every aspect of our lives, personalized recommendation systems have emerged as critical tools for filtering and directing relevant information to users. Powering a diverse array of internet services—from search engines and social networks to online retail and content streaming—these systems rely on sophisticated deep learning models to predict and tailor content that aligns with individual preferences. The efficiency and accuracy of these systems not only enhance user engagement but are also pivotal for the operational success of the platforms employing them.

Despite their significant impact, the deployment of personalized recommendation models faces notable challenges:

- **Memory-Bound Operations**: The core of recommendation models involves sparse embedding operations characterized by irregular memory access patterns. This irregularity fundamentally challenges acceleration efforts, making traditional computational architectures inefficient for optimizing these models.

- **Massive Embedding Tables**: The embedding tables, essential for capturing categorical (sparse) features, can grow to tens or hundreds of GBs in size, vastly exceeding on-chip memory capacities. This exacerbates the memory bottleneck, limiting the models' performance.

- **Computational Inefficiencies**: Existing architectures struggle with the computationally dense layers of deep neural networks (DNNs), as well as the memory-intensive non-DNN layers specific to recommendation models. The inadequacy in handling these layers presents significant performance bottlenecks.

These challenges underscore the limitations of conventional computing architectures in meeting the unique demands of personalized recommendation systems. There is a pressing need for innovative computational approaches that can navigate these hurdles efficiently, enhancing both the performance and energy efficiency of recommendation models at scale.

RecNMP (Recommendation Near-Memory Processing) presents a solution to these challenges by leveraging near-memory processing technology. Designed specifically for personalized recommendation systems, RecNMP aims to alleviate the memory bandwidth bottleneck through direct integration of lightweight computation with memory. This novel approach promises significant improvements in system throughput and energy efficiency, marking a pivotal step forward in the deployment of AI and deep learning models for personalized recommendation at scale.



### Paper's Contribution

The paper identifies significant contributions to the field of personalized recommendation systems. These include:

- The introduction of RecNMP, a near-memory processing solution designed to accelerate the inference of personalized recommendation models by leveraging the unique properties of memory-bound sparse embedding operations.
- An in-depth workload characterization that highlights the unique challenges posed by the memory bandwidth saturation in production recommendation models, showing the necessity for a specialized acceleration approach like RecNMP.
- RecNMP's hardware architecture, which is compatible with DDR4 and focuses on maximizing rank-level parallelism and exploiting temporal locality in embedding table accesses, offering a scalable and practical solution for deploying advanced recommendation models in production environments.
- A suite of hardware-software co-optimization techniques, including memory-side caching, table-aware packet scheduling, and hot entry profiling, designed to enhance the performance and efficiency of RecNMP. These optimizations enable improvements in memory latency for embedding operations and a significant overall throughput enhancement for end-to-end recommendation model inference.
- A comprehensive evaluation of RecNMP's performance improvements, demonstrating a speedup in end-to-end recommendation inference and a reduction in memory energy consumption compared to highly-optimized baseline systems .

![]({{site.baseurl}}/img/recnmp-1.png)

> Fig. 1 Model Architecture for a production-scale recommendation model



### Characterization Study

The authors do a characterization study to identify and understand the unique computational and memory access patterns of deep learning-based personalized recommendation models. This study is critical for designing the RecNMP architecture tailored to address these specific challenges. Key insights from the characterization study include:

- **Dominance of Embedding Operations**: The study highlights that personalized recommendation models are heavily dominated by memory-bound sparse embedding operations. These operations are characterized by sparse and irregular memory access patterns that significantly challenge traditional computing architectures .

- **Memory Bandwidth Saturation**: Through an evaluation of embedding operations on real systems, the study demonstrates how these operations can easily saturate memory bandwidth, especially as the batch size and the number of threads increase. This saturation points to a need for systems that can perform Gather-Reduce operations near memory to alleviate bandwidth constraints .

- **Locality in Embedding Table Lookups**: Contrary to prior assumptions that embedding table lookups are random, the study reveals a modest level of locality, primarily due to temporal reuse, within traces from production traffic. This finding suggests that caching strategies can improve performance by leveraging this locality .

- **Load Imbalance and Optimization Techniques**: The paper discusses the load imbalance among DRAM ranks when performing embedding table operations in parallel. It suggests several optimization techniques, including software methods for embedding table allocation and hardware optimizations like memory-side caching, to address this imbalance and improve overall system efficiency .

These findings underscore the necessity of a near-memory processing solution like RecNMP, designed to overcome the unique challenges posed by personalized recommendation systems. By tailoring the architecture to the specific needs identified through the characterization study, RecNMP aims to significantly enhance the performance and energy efficiency of these critical systems.



### ReNMP System Design

#### Hardware Architecture

The RecNMP system introduces an innovative approach to address the challenges posed by personalized recommendation systems through a unique hardware architecture. This design is crucial for enhancing the efficiency and performance of embedding operations in deep learning-based recommendation models. The key components and features of the RecNMP hardware architecture are:

- **System Overview**: RecNMP is strategically positioned within the buffer chip on the DIMM, serving as a bridge between the memory channel interface from the host and the standard DRAM device interface. This placement allows for seamless integration with existing memory infrastructure while providing the computational capabilities necessary for accelerating recommendation inference.

- **Processing Units**: Each buffer chip incorporates a RecNMP processing unit (PU), which consists of a DIMM-NMP module and multiple rank-NMP modules. This modular design ensures scalability and flexibility, allowing for easy adaptation to different memory configurations and capacities. The architecture supports concurrent processing across multiple ranks, significantly enhancing throughput and bandwidth utilization for embedding operations.

- **Communication and Instruction Handling**: The host-side memory controller communicates with RecNMP PUs using customized, compressed-format NMP instructions. These instructions are efficiently decoded and executed to perform local computation of embedding vectors, with the system designed to support aggregation across ranks within a processing unit. This mechanism facilitates efficient management of embedding operations, from instruction reception to final result accumulation.

- **DIMM-NMP and Rank-NMP Modules**: The DIMM-NMP module plays a critical role in dispatching NMP instructions to appropriate rank-NMP modules based on rank addresses. The rank-NMP modules, on the other hand, focus on the local computation of embedding vectors, leveraging the internal bandwidth of a DIMM to maximize the effective bandwidth of embedding table operations. This design ensures that embedding operations are performed close to memory, reducing latency and improving overall system performance.

Through these design elements, RecNMP offers a non-intrusive, scalable solution that enhances the performance of personalized recommendation systems. By addressing the unique challenges of sparse embedding operations directly within the memory hierarchy, RecNMP represents a significant step forward in the efficient deployment of deep learning-based recommendation models in production environments.

![]({{site.baseurl}}/img/recnmp-2.png)

> Fig. 2 (a) Architecture overview of *RecNMP* architecture; (b) DIMM-NMP; (c) Rank-NMP; (d) NMP instruction format.

#### C/A Bandwidth Expansion

The RecNMP architecture incorporates innovative strategies to overcome the limitations imposed by Command/Address (C/A) bandwidth, which is critical for maximizing the efficiency of memory operations in personalized recommendation systems. The highlights of the C/A bandwidth expansion strategy are as follows:

- **Challenge of Sparse Data Access**: The primary challenge in accessing the vast embedding tables is the sparse and irregular memory access patterns, which lead to frequent row buffer misses and conflicts. These patterns necessitate a high volume of ACT and PRE commands to access embedding vector entries, exacerbating the C/A bandwidth limitation.

- **Addressing Low Spatial Locality**: Embedding vectors, typically ranging from 64B to 256B, exhibit low spatial locality. This limitation results in a scenario where consecutive row buffer hits are minimal, highlighting the need for a solution that can efficiently manage the C/A bandwidth to support concurrent rank activations.

- **Innovative C/A Bandwidth Solution**: RecNMP introduces a customized instruction format, NMP-Inst, that utilizes compressed DDR commands. This approach significantly reduces the C/A bandwidth consumed per operation, enabling up to 8× bandwidth expansion for embedding vectors with low spatial locality. This compression scheme allows for parallel activation of multiple ranks, thus addressing the bandwidth bottleneck effectively.

![]({{site.baseurl}}/img/recnmp-3.png)

> Fig. 3 Timing diagram of (a) ideal DRAM bank interleaving read operations; (b) The proposed *RecNMP* concurrent rank activation.

#### Programming Model and Execution Flow

RecNMP adopts a heterogeneous computing programming model, similar to frameworks like OpenCL, to streamline the execution of personalized recommendation models. The programming model and execution flow incorporate the following key elements:

- **Heterogeneous Computing Model**: The system divides applications into host calls, which run on the CPU, and NMP kernels, offloaded to RecNMP Processing Units (PUs). This division allows for efficient use of computational resources by offloading specific tasks to the most suitable processors.
- **NMP Instruction Packets**: NMP kernels are compiled into packets of NMP-Insts, which are then transmitted over the DIMM interface to the RecNMP PUs. The results of these operations are sent back to the host CPU, ensuring a seamless integration between host and NMP computations.
- **Detailed Execution Flow**: An example using a simple SLS function call demonstrates the execution flow within the RecNMP programming model. This involves memory allocation for input and output data, initialization and loading of host-visible data, and execution of NMP kernels. The NMP-Insts are designed to fit within the standard C/A and DQ interface, showcasing the system's ability to integrate with existing memory architectures without significant modifications.

![]({{site.baseurl}}/img/recnmp-4.png)

> Fig. 4 (a) *RecNMP* SLS example code; (b) NMP packet; (c) NMP kernel offloading; (d) NMP-enabled memory controller.

#### HW/SW Co-optimization

RecNMP's design is enhanced through a series of hardware-software co-optimization techniques that improve performance and efficiency. These optimizations include:

- **Memory-Side Caching**: By exploiting temporal and spatial locality within embedding table operations, memory-side caching strategies are employed to reduce memory access latency. This is facilitated through techniques like table-aware packet scheduling and hot entry profiling, which optimize memory access patterns for embedding operations.

- **Hot Entry Profiling and Table-Aware Packet Scheduling**: These techniques identify frequently accessed embedding vectors and schedule memory accesses to optimize for spatial locality. The system dynamically adjusts based on the observed access patterns, reducing the latency of embedding operations and improving overall system throughput.

- **Customized Instruction Formats and Execution Optimizations**: RecNMP uses compressed NMP instruction formats to reduce the C/A bandwidth required for memory operations. Additionally, execution optimizations such as the pooling of embedding vectors and efficient memory allocation strategies further enhance the system's performance.

![]({{site.baseurl}}/img/recnmp-5.png)




