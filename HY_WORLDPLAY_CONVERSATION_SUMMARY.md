# HY-WorldPlay: Project and Conversation Summary

## Scope

This note summarizes:

- my understanding of `HY-WorldPlay` after reading the technical report and tracing the main code paths;
- the key questions answered in this chat (training data, inference inputs, and outputs).

## What HY-WorldPlay Is

`HY-WorldPlay` (HY-World 1.5) is a real-time interactive world model built on video diffusion.
Given an initial world condition (image/prompt) and a control trajectory (camera/actions), it autoregressively predicts future video chunks while maintaining long-term geometric consistency.

In this project, "world model" means a memory-aware video generation system rather than an explicit physics simulator.

## Core Ideas (Paper-Level)

From `HYWorld_1.5_Tech_Report.txt`, the system emphasizes four ingredients:

1. Dual action representation (discrete + camera pose cues for control),
2. Reconstituted context memory (retrieve useful past frames),
3. WorldCompass RL post-training (improve action following and quality),
4. Context forcing distillation (retain memory while enabling fast inference).

It frames generation as chunk-wise next-video prediction (16 frames per chunk at concept level), aiming for real-time interactive streaming with long-horizon consistency.

## What Is Implemented in This Repo (Code-Level)

Primary runnable inference path (Hunyuan branch):

- `hyvideo/generate.py`
- `hyvideo/pipelines/worldplay_video_pipeline.py`
- `hyvideo/models/transformers/worldplay_1_5_transformer.py`
- `hyvideo/utils/retrieval_context.py`

Primary open training path:

- `scripts/training/hyvideo15/run_ar_hunyuan_action_mem.sh`
- `trainer/training/ar_hunyuan_w_mem_training_pipeline.py`
- `trainer/training/ar_hunyuan_mem_training_pipeline.py`
- `trainer/dataset/ar_camera_hunyuan_w_mem_dataset.py`

Notable engineering in inference includes KV cache, sequence parallel support, optional SageAttention, optional fp8 GEMM quantization, and optional VAE parallel decode.

## Paper Claims vs Open-Sourced Status

- Memory-aware AR rollout and action/camera conditioning are clearly implemented and runnable.
- RL/WorldCompass is described in report/README, but a clearly exposed public RL training pipeline is not obvious in the main released training path.
- Context forcing distillation is described in report; distillation knobs exist, but a clearly labeled full context-forcing training pipeline is not straightforwardly exposed in the main open training route.

## Conversation Q&A Recap

### 1) What does training data look like?

Two levels:

- Paper data mixture: 320K curated clips across AAA games, real-world 3D-derived data, synthetic 4D, and natural video.
- Open training format: preprocessed feature data (not raw video decode in-loop).

The training loader expects a dataset manifest (`--json_path`) where each sample points to:

- latent feature file (`latent_path`, `.pt`),
- pose file (`pose_path`, JSON),
- optional action labels (`action_path`, JSON for some datasets).

The latent `.pt` typically contains precomputed tensors used during training (latent, prompt embeddings/masks, image condition, vision states, byT5 states/mask).

### 2) During inference, what can be used as input?

In the released CLI path:

- image (`--image_path`, required by current script assertion),
- prompt (`--prompt`),
- pose trajectory (`--pose`), either:
  - pose string (e.g., `w-3, right-1, d-4`),
  - pose JSON with intrinsics/extrinsics.

So practical public usage is image-to-video style conditioned interactive generation.

### 3) What is the output? Video only or geometry too?

Direct output in this repo's inference path is video:

- `gen.mp4`,
- optionally `gen_sr.mp4` with super-resolution.

No direct point-cloud/mesh export is provided in the released inference path.
The report's geometry/point-cloud examples are downstream applications that combine generated consistent views with a separate reconstruction system.

## Practical "Start Here" Reading Order

1. `README.md` (claims, setup, run modes),
2. `hyvideo/generate.py` (CLI contract and preprocessing of pose/action),
3. `hyvideo/pipelines/worldplay_video_pipeline.py` (AR rollout + memory use),
4. `hyvideo/utils/retrieval_context.py` (memory frame selection logic),
5. `trainer/training/ar_hunyuan_mem_training_pipeline.py` and `trainer/dataset/ar_camera_hunyuan_w_mem_dataset.py` (training behavior).

## One-Line Takeaway

`HY-WorldPlay` is best understood as a memory-aware, action-conditioned, autoregressive video diffusion system optimized for interactive long-horizon consistency, with video as the primary delivered artifact in open-source inference.

