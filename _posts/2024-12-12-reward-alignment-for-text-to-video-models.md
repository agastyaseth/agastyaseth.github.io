---
layout: post
published: true
title: Reward Alignment for Text-to-Video Models
subtitle: What happened when we brought ReNO and RAFT from image diffusion to Pyramid-Flow
date: '2024-12-12'
image: /img/t2v-reno-comparison.jpg
tags:
  - Generative Models
  - Diffusion
  - Alignment
  - Text-to-Video
---

Reward-based alignment has done a lot for text-to-image (T2I) diffusion. Methods like ReNO and RAFT measurably improve prompt adherence and image quality. Text-to-video (T2V) is a much harder setting: every frame has to look good, and consecutive frames also have to agree with each other. This post is about a team project at ASU where we tried to carry those alignment methods, along with two distillation methods, over to video models. I led the ReNO integration.

The short version: the methods don't transfer cleanly, and understanding why is the useful part.

## The pieces

- **Pyramid-Flow** generates video hierarchically. It starts at low resolution and refines in stages, which keeps compute manageable.
- **CogVideoX** is a T2V model built on an expert transformer and a 3D causal VAE, with strong temporal consistency.
- **ReNO** (Reward-based Noise Optimization) leaves the model weights alone. Instead it optimizes the *initial noise* at inference time, using gradients from reward models such as CLIPScore.
- **RAFT** (Reward rAnked FineTuning) generates candidates, ranks them with a reward model, and fine-tunes on the best ones.
- **InstaFlow** and **SlimFlow** distill multi-step rectified-flow models into fast one-step, or smaller, students.

## ReNO on Pyramid-Flow

ReNO needs gradients to flow from the reward back to the initial noise, so gradient tracking has to stay on during inference. On a video model that's expensive. At 256×256 per frame, a single video used the full 80 GB of an NVIDIA A100. At 512×512 every run ran out of memory. We were limited to a batch size of one, 20 to 50 optimization steps per run, and learning rates between 1e-3 and 1e-4.

The reward combined CLIPScore for semantic alignment with a custom heuristic for temporal coherence. For comparison, we ran the same ReNO workflow on single-step image models (HyperSD, SD, SDXL).

![ReNO across image models and Pyramid-Flow](/img/t2v-reno-comparison.jpg)

*Same prompts, same ReNO workflow. On single-step image models ReNO sharpens detail and follows the prompt more closely. On Pyramid-Flow (last column) the output collapses into blur.*

The contrast is stark. HyperSD+ReNO and SD+ReNO clearly improve on their baselines. The turtle appears, the umbrellas get their color, and the oasis shows up. Pyramid-Flow+ReNO produces almost featureless color fields.

## Why it breaks

ReNO assumes the initial noise has a direct, single-shot influence on the output. In a single-step model it does. Pyramid-Flow instead passes the latent through several refinement stages, and each stage partly washes out what the optimized noise was steering toward. Worse, any small inconsistency introduced early is amplified as it moves up the pyramid. That shows up as blur within frames and abrupt jumps between them. Our temporal-coherence heuristic wasn't strong enough to counteract it.

The lesson isn't that ReNO is bad. It's that noise optimization has to be adapted to the generation schedule. One option is to optimize per stage, or to apply rewards to intermediate latents instead of only the final frames.

## RAFT and distillation

Teammates built a RAFT fine-tuning pipeline on Pyramid-Flow's 384p diffusion transformer, with CLIP-based aesthetic rewards and a reconstruction loss. The hard parts were balancing those two losses and handling videos of different lengths and resolutions. On the distillation side, we set up InstaFlow and SlimFlow baselines with Pyramid-Flow and CogVideoX on two A100s, using a mixed JourneyDB and OpenVid dataset. Compute limits kept us from fully fine-tuning any of them.

## Takeaways

- Alignment methods built for single-step image diffusion assume a direct line from noise to pixels. Hierarchical video models break that assumption.
- Temporal coherence needs its own reward signal. Frame-level CLIP scores aren't enough. Video-native evaluators like VIPER, V-JEPA, or Open-VCLIP are the obvious next step.
- Memory is the real constraint. Inference-time gradient methods on video need memory-efficient tricks (checkpointing, latent-space rewards) before they're practical.
