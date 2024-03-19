---
layout: post
published: true
title: >-
  Centaur: A Chiplet-based, Hybrid Sparse-Dense Accelerator for Personalized
  Recommendations
subtitle: Paper Review
date: '2024-03-18'
tags:
  - Machine Learning Acclerator Design
  - Personalized Recommendation
  - FPGA
  - Software-Hardware Codesign
  - DNN
---
# Centaur: A Chiplet-based, Hybrid Sparse-Dense Accelerator for Personalized Recommendations

### Introduction & Motivation

In today's digital landscape, the capability of platforms like Netflix, Amazon, and Facebook to offer personalized content has become a cornerstone of user engagement. These platforms deploy recommendation models at an immense scale within their data centers, navigating through vast product catalogs to deliver individualized suggestions. At the heart of these recommendation systems lie embedding tables, which due to the extensive range of products and user interactions, can expand to hundreds of GBs. For example:

- It's estimated that over 79% of all AI inference cycles across Facebook’s production datacenters are dedicated to operating these intricate recommendation models.

Deploying these models at such a grand scale introduces several challenges:

- **Sparse Data Access Patterns:** The lack of reusable dataflow patterns and spatial locality in the embedding layers of recommendation models means data accessed during inference is highly sparse and irregular. This complexity hinders the exploitation of spatial locality, failing to maximize memory-level parallelism.

- **Low Computational Throughput of CPUs:** Despite their general utility, CPUs are not inherently designed to efficiently process the computationally dense layers of deep neural networks (DNNs), nor the memory-intensive non-DNN layers. These layers are increasingly recognized as significant performance bottlenecks in modern ML workloads.

These challenges — inefficient memory bandwidth utilization due to sparse data access patterns and the inadequate computational throughput of CPUs — represent significant bottlenecks for the deployment of personalized recommendation systems. There is a clear demand for a new computational approach that can specifically address these hurdles by optimizing both memory usage and computational efficiency for the demanding needs of large-scale recommendation system deployments.

The introduction of Centaur, a chiplet-based hybrid sparse-dense accelerator, aims to directly tackle the unique challenges posed by personalized recommendation systems. By combining the computational flexibility of FPGAs with the processing power of CPUs, Centaur seeks to improve the efficiency and performance of personalized recommendation systems at scale, presenting a promising solution to the outlined challenges.



### Paper's Contribution

This paper introduces Centaur, a pioneering approach in the realm of machine learning acceleration, particularly tailored to the nuanced requirements of personalized recommendation systems. Centaur’s unique value proposition lies in its innovative architecture and targeted optimization strategies, which collectively address the dual challenges of sparse data processing and dense computation inherent in recommendation models. Below, we detail the key contributions of this paper:

- **Innovative Hybrid Architecture:** At the core of Centaur is a hybrid architecture that thoughtfully integrates a CPU with an FPGA, leveraging the strengths of both to achieve unprecedented levels of efficiency and performance. This heterogenous design is purpose-built to handle the diverse computational patterns encountered in recommendation systems.
- **"Sparse" Accelerator for Embeddings:**
  - Centaur introduces a "Sparse" accelerator, specifically designed to optimize the processing of embedding layers. These layers, characterized by their sparse data access patterns and significant memory bandwidth demands, are efficiently managed through the FPGA component of Centaur.
  - The Sparse accelerator significantly improves memory bandwidth utilization by directly accessing embedding tables stored in the CPU's memory, bypassing traditional bottlenecks and enhancing throughput for sparse data operations.
- **"Dense" Accelerator for GEMM (General Matrix Multiplication):**
  - Alongside the Sparse accelerator, Centaur incorporates a "Dense" accelerator dedicated to the compute-intensive aspects of recommendation models, particularly GEMM operations. GEMM is pivotal for executing multi-layer perceptrons (MLPs) and other dense computations within the recommendation workflow.
  - By offloading GEMM computations to the FPGA, the Dense accelerator achieves substantial improvements in computational throughput. This ensures that the backend dense computations, which are crucial for the final stages of the recommendation pipeline, are executed swiftly and efficiently.

The integration of these two specialized accelerators within a single architecture represents a significant leap forward in the field of machine learning acceleration. By addressing the specific computational and memory access challenges of personalized recommendation systems, Centaur not only boosts the performance of these systems but also enhances their energy efficiency. This targeted approach ensures that large-scale deployments of recommendation models, which are integral to the operations of major internet platforms, can be carried out more effectively and sustainably.



![]({{site.baseurl}}/img/centaur-1.png)

> Fig 1. Workflow of a Personalized Recommendation Model



![]({{site.baseurl}}/img/centaur-2.png)
> Fig 2. Comparison of CPU-FPGA Integration Approaches



### Workload Characterization

The paper presents an in-depth workload characterization of DNN-based personalized recommendation systems using the deep learning recommendation model (DLRM) to root-cause performance bottlenecks and motivate the design of their hybrid sparse-dense FPGA accelerator, Centaur. The paper's characterization study addresses the design space of recommendations by varying the number of embedding tables, the number of gather operations per table, and the total memory usage of embedding tables and MLP layers.

**Key Findings:**

- **Embedding Layers:**
  - These layers can consume up to several hundreds of GBs of memory, particularly in inference mode.
  - The aggregate size of gathered embeddings for inference is much smaller than the size of embedding tables, resulting in extremely sparse operations with low spatial/temporal locality.
  - CPUs fail to maximize memory-level parallelism, thus significantly under-utilizing memory bandwidth for sparse embedding gather operations.
  - The characterization shows that embedding layers can account for a significant fraction of inference time, up to 79%, becoming a prominent performance bottleneck 【3†source】.

- **Multi-layer Perceptrons (MLPs):**
  - Despite the smaller relative size compared to embedding layers, MLP layers still account for a non-trivial portion of runtime, especially with smaller batch sizes.
  - The inference latency of MLP layers increases with larger batch sizes due to the cost of uploading weights on-chip, although they exhibit low last-level cache (LLC) miss rates and low misses per thousand instructions (MPKI) compared to embedding layers【3†source】.

- **End-to-End Inference Time:**
  - With varying input batch sizes from 1 to 128, the embedding layers showed a significant fraction of the execution time compared to MLP layers and others.
  - MLP layers experience a slower increase in execution time with larger batch sizes compared to embedding layers, except for DLRM(6), which has a compute-intensive MLP layer【3†source】.

- **Caching Efficiency:**
  - Embedding layers' LLC miss rate is sensitive to the input batch size, with an increasing number of misses as the batch size is increased. 
  - MLP layers show less sensitivity to batch size as the aggregate model size of these layers is typically small enough to fit inside the CPU's on-chip caches .

- **Effective Memory Throughput:**
  - The effective memory bandwidth utilized for gathering embedding vectors is quite low compared to the potential maximum bandwidth of the baseline CPU memory system.
  - Despite high LLC miss rates and MPKI for embedding layers compared to MLP layers, embedding operations use a fraction of the available memory bandwidth  .



### Architecture of Centaur

The Centaur architecture is a novel response to the intricate demands of DNN-based personalized recommendation systems that are extensively deployed in modern data centers. These systems must process vast and complex data sets in real time, a task that involves handling hundreds of gigabytes of model data and balancing computational intensity with memory bandwidth. The paper presents Centaur as a solution with a detailed examination of its components and their roles within the hybrid structure.

**Motivation for CPU + FPGA:**

- Personalized recommendation models necessitate a processing platform capable of managing the high memory usage of embedding layers, often reaching hundreds of gigabytes, while simultaneously offering the computational capacity for dense MLP layers.
- The CPU+FPGA combination emerges as an effective design, balancing the flexibility of FPGA programmable logic with the high memory capacity and efficiency of CPUs, crucial for handling embedding layers which can account for a significant portion of the inference time in recommendation models .

**Chiplet-based CPU+FPGA Architecture:**

- Centaur proposes a chiplet-based architecture that integrates CPU and FPGA in a minimally invasive design that is socket-compatible with current systems, allowing easy implementation in existing data center infrastructures.
- It features two communication paths: a cache-coherent path for data locality and a cache-bypassing path for memory-intensive operations. This dual approach allows the FPGA to either work with the CPU cache or directly access memory, significantly boosting throughput for embedding layers .

**Sparse Accelerator:**

- The sparse accelerator is designed to facilitate high-throughput, low-latency operations for embedding gathers and reductions. It leverages the direct access to shared physical memory systems enabled by package-integrated CPU+FPGA devices.
- This component includes an embedding streaming unit (EB-Streamer), which performs multiple embedding vector gather operations in parallel, leading to a substantial increase in throughput and efficient utilization of available memory bandwidth .

![]({{site.baseurl}}/img/centaur-3.png)

> Fig 3. Microarchitecture of Centaur sparse accelerator.



**Dense Accelerator:**

- The dense accelerator focuses on accelerating GEMM computations, which are essential for MLP layers and feature interactions within the recommendation model.
- Utilizing Altera’s FPGA floating-point IP cores optimized for matrix multiplication, the dense accelerator complex is equipped to deliver a high aggregate computational throughput, significantly outperforming CPU-only systems in GEMM operations .

![]({{site.baseurl}}/img/centaur-4.png)

> Fig 4. Microarchitecture of Centaur dense accelerator.



Together, the sparse and dense accelerators form the Centaur architecture, designed to overcome the bottlenecks of memory-intensive embedding layers and compute-intensive MLP layers, which are characteristic of modern recommendation systems. This integration not only provides a substantial performance uplift but also improves energy efficiency when deploying personalized recommendation models at scale .

