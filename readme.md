# Optimized GRU Model for Sentiment Analysis Using LTH, KD, and QAT

## Table of Contents
- [Optimized GRU Model for Sentiment Analysis Using LTH, KD, and QAT](#optimized-gru-model-for-sentiment-analysis-using-lth-kd-and-qat)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Problem Statement](#problem-statement)
  - [Motivation](#motivation)
  - [Objectives](#objectives)
  - [Methodology](#methodology)
    - [Lottery Ticket Hypothesis (LTH)](#lottery-ticket-hypothesis-lth)
    - [Knowledge Distillation (KD)](#knowledge-distillation-kd)
    - [Quantization-Aware Training (QAT)](#quantization-aware-training-qat)
  - [Implementation Details](#implementation-details)
    - [Dataset](#dataset)
    - [Data Preprocessing](#data-preprocessing)
    - [Model Architecture](#model-architecture)
    - [Pruning Process](#pruning-process)
    - [Training Process](#training-process)
    - [Model Evaluation](#model-evaluation)
  - [Results and Discussion](#results-and-discussion)
  - [Future Enhancements](#future-enhancements)
  - [Conclusion](#conclusion)

---

## Introduction
Deep Neural Networks (DNNs) have shown remarkable success in various AI applications, particularly in Natural Language Processing (NLP). However, deploying deep learning models like Gated Recurrent Units (GRUs) in resource-constrained environments remains a challenge due to their computational and memory requirements.

This project aims to optimize GRU models for sentiment analysis by implementing model compression techniques such as:
- Lottery Ticket Hypothesis (LTH)
- Knowledge Distillation (KD)
- Quantization-Aware Training (QAT)

These techniques significantly reduce model size and computation while maintaining accuracy, making them deployable on edge devices like IoT platforms.

## Problem Statement
Traditional GRU-based sentiment analysis models require substantial computational power and memory, making them unsuitable for real-time AI applications on low-resource devices like Raspberry Pi, ESP32, and embedded systems. This project optimizes a GRU model using advanced compression techniques to enhance efficiency while retaining accuracy.

## Motivation
As AI applications expand into real-time analytics and IoT, deep learning models must be efficient enough for real-world deployment. The project addresses this challenge by leveraging structured pruning, knowledge distillation, and quantization to develop a lightweight yet effective sentiment analysis model.

## Objectives
- Reduce the size and computational complexity of GRU models while maintaining accuracy above 92%.
- Implement LTH for structured pruning, KD for knowledge transfer, and QAT for efficient deployment.
- Develop a model optimization pipeline that enables real-time sentiment analysis on edge devices.

## Methodology
### Lottery Ticket Hypothesis (LTH)
LTH suggests that within a large network, there exists a smaller, highly optimized subnetwork capable of achieving similar performance. We apply LTH to prune unnecessary weights while preserving accuracy.

### Knowledge Distillation (KD)
KD via Attention-based feature distillation(AFD) transfers knowledge from a large, well-trained "teacher" model to a smaller "student" model. This ensures that the compressed model retains most of the predictive capabilities of the original network.

### Quantization-Aware Training (QAT)
QAT reduces model precision from 32-bit floating points to lower-bit representations (e.g., INT8), minimizing memory footprint and increasing inference speed with negligible accuracy loss.

## Implementation Details
### Dataset
The Amazon Reviews dataset is used for sentiment analysis. It includes:
- Text reviews
- Star ratings (1 to 5 stars)
- Balanced selection of 5M positive (4-5 stars) and 5M negative (1-3 stars) reviews

### Data Preprocessing
- Tokenization using Byte-Pair Encoding (BPE)
- Stopword removal and text normalization
- Conversion of ratings into binary sentiment labels

### Model Architecture
- Embedding Layer: Encodes input text representations
- Bi-directional GRU Layers: Captures sequential dependencies
- Dense Layers: Transforms features for classification
- Output Layer: Binary sentiment classification

### Pruning Process
- Calculate sparsity and define pruning threshold
- Generate a global pruning mask for the entire model
- Retrain the pruned subnetwork using LTH methodology

### Training Process
- Train the base GRU model to convergence
- Apply pruning via LTH and fine-tune
- Perform KD to transfer knowledge from teacher to student model
- Implement QAT to simulate lower-bit precision inference

### Model Evaluation
- Accuracy, Precision, Recall, and F1-score
- ROC-AUC for classification performance
- Confusion matrix analysis

## Results and Discussion
The optimized GRU model achieves:
- **Baseline Model:** 92.79% accuracy, 24MB size
- **Pruned Model:** 92.57% accuracy, 8MB size
- **Distilled Model:** 92.79% accuracy, 8MB size
- **Quantized Model:** 92.72% accuracy, 2.5MB size

Inference time reduced from **0.0009s to 0.000867s**, demonstrating efficiency gains with minimal performance loss.

## Future Enhancements
- **IoT Integration:** Deploy the model on smartwatches, security systems, and healthcare monitors.
- **Further Optimizations:** Implement hardware-aware optimizations for faster processing.
- **Scalability:** Extend support for multiple languages and larger datasets.

## Conclusion
This project successfully optimizes GRU models for sentiment analysis using LTH, KD, and QAT. The results show significant improvements in computational efficiency while maintaining high accuracy, making the model suitable for deployment in real-world, resource-constrained environments.

