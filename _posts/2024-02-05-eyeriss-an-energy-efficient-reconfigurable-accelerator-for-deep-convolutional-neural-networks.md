---
layout: post
published: true
title: >-
  Eyeriss: An Energy-Efficient Reconfigurable Accelerator for Deep Convolutional
  Neural Networks
subtitle: Paper Review
date: '2024-02-05'
image: img/eyeriss-2
tags:
  - ML Accelerators
  - CNN
  - Deep Learning
  - Deep Learning Accelerators
  - Energy Efficient Accelerators
---

## Introduction

Convolutional Neural Network (CNN) accelerators have emerged as a pivotal solution to the computational and energy demands of deep learning applications. These specialized hardware accelerators are designed to efficiently process the vast amount of data and complex operations involved in CNNs. Motivated by the need to enhance performance and energy efficiency beyond what general-purpose computing systems offer, CNN accelerators leverage optimized data processing architectures. They aim to reduce latency and power consumption, enabling real-time processing and deployment of AI applications in resource-constrained environments such as mobile devices and embedded systems.



## Eyeriss: An Energy-Efficient Reconfigurable Accelerator for Deep Convolutional Neural Networks

Eyeriss is an energy-efficient reconfigurable accelerator for CNNs. The paper addresses the challenges of high throughput and energy efficiency in processing CNNs due to significant data movement from both on-chip and off-chip memory, which is more energy-consuming than computation. The key to achieving high throughput and energy efficiency lies in **minimizing data movement energy costs** for any CNN shape. Eyeriss accomplishes this through a novel processing dataflow called **row stationary (RS)** on a spatial architecture with 168 processing elements. This dataflow optimizes energy efficiency by maximizing data reuse locally, reducing the need for expensive data movements like DRAM accesses. Additionally, techniques such as compression and data gating further enhance energy efficiency.



### Introduction and Motivation

The next section of the paper details the various innovative features that enhance the energy efficiency of the Eyeriss accelerator for deep convolutional neural networks (CNNs):

- **Spatial Architecture:** Utilizes 168 processing elements to create a hierarchical memory structure that minimizes high-cost data movements.
- **Row Stationary Dataflow:** A unique dataflow that optimizes energy efficiency by enhancing local data reuse, significantly reducing the need for data movement.
- **Network-on-Chip Architecture:** Features a specialized NoC that supports efficient data delivery mechanisms, crucial for the row stationary dataflow.
- **Compression and Data Gating:** Implements run-length compression and data gating to exploit data sparsity, reducing energy consumption by minimizing unnecessary computations and memory accesses.



### System Architecture

This section elaborates on the innovative design strategies implemented in Eyeriss to **enhance data efficiency** and processing throughput for deep convolutional neural networks, emphasizing the architecture's **adaptability and efficiency** in handling various data types and convolutional operations.

- **Spatial Array of PEs**: The Eyeriss architecture includes a spatial array of 168 processing elements (PEs) arranged in a 12x14 rectangle, featuring a **global buffer (GLB)** of 108-kB, an **RLC CODEC** for run-length compression, and a **ReLU module** for activation functions. This design facilitates efficient data movement between PEs, GLB, and DRAM, reducing energy consumption.
- **Control and Configuration**: The system employs a two-level control hierarchy. The top level manages traffic between off-chip DRAM and the GLB, and between the GLB and the PE array, as well as operations of the RLC CODEC and ReLU module. The lower level includes independent control logic in each PE, allowing for asynchronous processing across PEs.
- **PE Sets for 2-D Convolution**: To handle 2-D convolutions, Eyeriss **groups PEs into sets** that share filter weights and ifmap values for efficient data reuse and psum accumulation. The dimension and mapping of these PE sets are adaptable to different layer shapes, optimizing for energy efficiency.
- **Multiple PE Sets and Dimensions Beyond 2-D**: Eyeriss can map multiple PE sets onto the PE array to increase processing throughput and data reuse. The architecture supports additional dimensions by processing multiple 2-D convolutions simultaneously, adjusting for different channels and filters.

![eyeriss-1.png]({{site.baseurl}}/img/eyeriss-1.png)

### Energy-Efficient Features

The "Energy-Efficient Features" section delves into strategies for enhancing the Eyeriss accelerator's energy efficiency, focusing on two key approaches:

**A. Energy-Efficient Dataflow: Row Stationary**

- **Row Stationary (RS) Dataflow**: Optimizes energy efficiency by minimizing data movement for all data types (input feature maps, filter weights, and partial sums/output feature maps), taking into account the energy costs at different memory hierarchy levels. It significantly reduces high-cost DRAM and global buffer accesses by maximizing data reuse from low-cost scratchpad memories and inter-PE communication. The RS dataflow is found to be 1.4–2.5 times more energy-efficient in AlexNet compared to existing dataflows.
- **1-D Convolution Primitive in a PE**: RS dataflow divides the computation into 1-D convolution primitives that run in parallel. Each primitive operates on one row of filter weights and one row of ifmap values, generating one row of psums. By mapping each primitive to one PE, computation remains stationary in the PE, utilizing local scratchpad memories for both data reuse and psum accumulation efficiently.
- **Figures 3, 4, and 5** in the paper illustrate the processing sequence of a 1-D convolution primitive in a PE, the dataflow in a PE set for processing a 2-D convolution, and the mapping of the PE sets on the spatial array of 168 PEs for the CONV layers in AlexNet, respectively.

![eyeriss-2.png]({{site.baseurl}}/img/eyeriss-2.png)

![eyeriss-3.png]({{site.baseurl}}/img/eyeriss-3.png)

**B. Exploit Data Statistics**

- **Leveraging Data Sparsity**: The inherent sparsity in CNN data, such as zeros introduced by the ReLU function and the possibility of pruning filter weights to zeros, is exploited to improve energy efficiency. This sparsity is utilized through run-length coding (RLC) for compression, significantly reducing DRAM bandwidth needs.
- **Figures 8 and 9** showcase the RLC encoding process and compare DRAM accesses (read and write) for filters, ifmaps, and ofmaps before and after using RLC in the five CONV layers of AlexNet, highlighting the substantial reduction in DRAM accesses and thus energy savings.

These features target the minimization of data movement and the efficient use of data statistics, leading to notable improvements in energy efficiency for CNN processing on the Eyeriss accelerator.



![eyeriss-4.png]({{site.baseurl}}/img/eyeriss-4.png)![eyeriss-5.png]({{site.baseurl}}/img/eyeriss-5.png)



### System Modules

This section outlines the key architectural components designed to enhance the efficiency and flexibility of the accelerator, focusing on handling deep convolutional neural networks (CNNs):

**A. Global Buffer (GLB)**

- **Capacity and Function**: The GLB has a 108-kB capacity, interfacing with DRAM via an asynchronous interface and the PE array through the NoC. It stores inputs (ifmaps), weights (filters), and outputs (psums/ofmaps), allocating 100 kB for ifmaps and psums as per the RS dataflow requirements and 8 kB for filters to alleviate off-chip bandwidth constraints.
- **Data Preloading**: While the PE array processes one pass, the GLB preloads filters for the next pass to optimize processing efficiency.
- **Reconfigurability**: The storage for ifmaps and psums is divided into 25 reconfigurable 4 kB banks to accommodate different data proportions, enabling simultaneous access to both data types by the PE array.

**B. Network-on-Chip (NoC)**

- **Function and Design Goals**: The NoC facilitates data movement between the GLB and PE array, and among PEs, with the architecture tailored to support the RS dataflow's delivery patterns. It aims to manage different convolution strides, segmented sets, and simultaneous mappings of multiple sets while maximizing energy efficiency and bandwidth.
- **Architecture**: Comprising global input and output networks (GIN and GON) for multicast and direct data transfers, and a local network (LN) for vertical psum passing between PEs. The GIN uses a hierarchical structure to efficiently deliver data to PEs, optimizing for single-cycle multicast to groups of PEs.

**C. Processing Element (PE) and Data Gating**

- **Architecture**: Each PE incorporates FIFOs for workload balancing, a configurable control for processing multiple filters and channels, and a three-stage pipeline for computation, including a 16-bit multiplier and adder.
- **Data Gating for Energy Efficiency**: Data gating logic exploits zero values in inputs (ifmaps) to save energy by disabling unnecessary filter reads and MAC operations, reducing power consumption by up to 45%.
  
  

**Key takeaway:** The architecture is specifically designed to adapt to different CNN layer shapes and sizes, demonstrating a flexible and energy-efficient approach to deep learning acceleration.
