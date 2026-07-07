# arXiv:2512.22615v2[cs.CV]4Jan2026

Dream-VL & Dream-VLA: Open Vision-Language and Vision-Language-Action Models with Diffusion Language Model Backbone

Jiacheng Ye1,∗ Shansan Gong1,∗ Jiahui Gao2,†,♢ Junming Fan2 Shuang Wu2 Wei Bi‡ Haoli Bai2 Lifeng Shang2 Lingpeng Kong1,† 1The University of Hong Kong 2Huawei Technologies

Dream-VL Dream-VLA DreamLM/Dream-VLX

December 23, 2025

Abstract

While autoregressive Large Vision-Language Models (VLMs) have achieved remarkable success, their sequential generation often limits their efficacy in complex visual planning and dynamic robotic control. In this work, we investigate the potential of constructing Vision-Language Models upon diffusion-based large language models (dLLMs) to overcome these limitations. We introduce Dream-VL, an open diffusion-based VLM (dVLM) that achieves state-of-the-art performance among previous dVLMs. Dream-VL is comparable to top-tier AR-based VLMs trained on open data on various benchmarks but exhibits superior potential when applied to visual planning tasks. Building upon Dream-VL, we introduce Dream-VLA, a dLLM-based Vision-Language-Action model (dVLA) developed through continuous pre-training on open robotic datasets. We demonstrate that the natively bidirectional nature of this diffusion backbone serves as a superior foundation for VLA tasks, inherently suited for action chunking and parallel generation, leading to significantly faster convergence in downstream fine-tuning. Dream-VLA achieves top-tier performance of 97.2% average success rate on LIBERO, 71.4% overall average on SimplerEnv–Bridge, and 60.5% overall average on SimplerEnv–Fractal, surpassing leading models such as π0 and GR00T-N1. We also validate that dVLMs surpass AR baselines on downstream tasks across different training objectives. We release both Dream-VL and Dream-VLA to facilitate further research in the community.

## 1 Introduction

Vision-language models are reshaping AI across diverse domains – from medical diagnosis and scientific discovery to autonomous systems and robotic manipulation (Tu et al., 2024; Zhang et al., 2024a,c; Zhou et al., 2024b; Driess et al., 2023; Brohan et al., 2023). As these applications grow in complexity, a fundamental capability emerges as critical: visual planning – the ability to perceive visual context, reason about goals, and formulate coherent action sequences. Whether analyzing multi-step experimental procedures in scientific papers, guiding a surgical robot through delicate operations, orchestrating a delivery drone’s route, or enabling a home robot to organize a kitchen, success hinges on the capacity for sophisticated long-horizon planning.

Large vision-language models (VLMs) have demonstrated remarkable progress in grounding language with visual representations, powering applications from multimodal dialogue to scene understanding. Building on these foundations, vision-language-action (VLA) models extend these capabilities to action generation, enabling agents to interact with physical and virtual environments. However, current VL and VLA models overwhelmingly rely on autoregressive large language models (LLMs) as their backbone. This architectural choice presents a fundamental bottleneck: the autoregressive paradigm, trained on next-token prediction, struggles with tasks requiring long-horizon planning and global reasoning (Bachmann and Nagarajan, 2024; Ye et al., 2025a; Weng et al., 2025). Besides, its sequential

∗Equal contribution †Corresponding author ♢Project Lead ‡Independent Researcher

High-level action

Multimodal reasoning

Text planning

Low-level action planning

Applied Scenarios

planning {

Text reasoning

Deploy

[0.1,

[Figure 1]

[Figure 2]

- -0.2, 0, 10º, ... 25º,
- -7º, 0]

`action`:

| |
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 3]

[Figure 4]

Each box has 6

`move`

...

pencils. She buys

`params`: {

Chunks = 4

4 boxes, so 4*6=24 pencils.

...

} }

(Tokens = 28)

...

Cups are on the table, …

Cups are on the table, …

Get cups and …

Generation

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

…

…

| |
|---|

…

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Dream-VLA (dVLA)

Dream (dLLM)

Dream-VL (dVLM)

Backbone

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

…

…

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

…

Diffusion Input

Vision Get cups and … Encoder

Vision

Cups are [m] the table, …

Cups are [m] the table, …

Empty Action

Encoder

|[Figure 5]|
|---|

|[Figure 6]|
|---|

|Δx | Δy | Δz | Δ⍺ | Δβ | Δγ | G|
|---|

Robotic Action

- Figure 1: Overview of the Dream family, with bi-directional masked diffusion modeling. Built on the diffusion language model Dream-7B, we introduce Dream-VL, a state-of-the-art diffusion VLM that demonstrates strong multimodal understanding, reasoning, and effective long-horizon planning. Building further on Dream-VL, we perform VLA pretraining to obtain the first pretrained diffusion-based VLA model (dVLA), which serves as a strong backbone for downstream VLA tasks.

generation nature suffers from error accumulation during inference, further hindering global reasoning capabilities (Zhang et al., 2023a; Rohatgi et al., 2025).

Diffusion large language models (dLLMs) offer a compelling alternative to address these challenges (Inception Labs, 2025; Nie et al., 2025; Ye et al., 2025c; DeepMind, 2025). Unlike autoregressive models that generate tokens sequentially, diffusion models learn to iteratively refine noisy sequences into coherent outputs. This iterative refinement process naturally encourages global coherence, making dLLMs particularly well-suited for planning tasks where long-range dependencies and goal-oriented reasoning are critical (Ye et al., 2025c; Zhang et al., 2025a). Moreover, the parallel decoding capability of diffusion language models offers computational advantages during inference (Inception Labs, 2025; DeepMind, 2025; Wang et al., 2025b; Wu et al., 2025b). Given these promising characteristics in language modeling, a natural question arises: can these advantages translate to vision-language tasks, where planning and coherent reasoning are equally critical?

To explore this potential, we build Dream-VL, a diffusion LLM-based vision-language model (dVLM) for general visual understanding, and Dream-VLA, a diffusion LLM-based vision-languageaction model (dVLA) for robotic manipulation. We conduct comprehensive experiments across visual understanding, visual planning, and embodied action benchmarks. Our results show that:

- • Dream-VL achieves comparable performance to top-tier autoregressive VLMs trained on open data, and significantly surpasses existing dVLMs such as LLaDA-V (You et al., 2025) and Dimple (Yu et al., 2025). Notably, Dream-VL shows superior performance over strong AR baselines on visual planning tasks (e.g., LIBERO (Liu et al., 2023a), ViPlan (Merler et al., 2025)) that require long-horizon planning.
- • Built upon Dream-VL, Dream-VLA establishes top-tier performance on both the evaluated SimplerEnv (Li et al., 2024e) and LIBERO (Liu et al., 2023a) benchmark, surpassing top VLA

models such as π0 (Black et al., 2024a) and GR00T-N1 (Bjorck et al., 2025). Notably, Dream-VLA consistently outperforms the strong AR-based baseline OpenVLA-OFT (Kim et al., 2025) across diverse finetuning objectives, demonstrating that a natively bidirectional dVLM can serve as a superior backbone for VLA tasks.

We attribute these strong results to three inherent advantages of diffusion-based VLMs over autoregressive models: (1) Bidirectional attention enables richer information fusion between visual and text features; (2) The text planning capability in dLLMs enhances visual planning in VLMs by generating global plans aligned with predefined goals; (3) Diffusion VLMs naturally support action chunking and parallel generation without architectural modification, which are crucial for low-level action prediction in VLA models and enable faster convergence during downstream fine-tuning.

## 2 Related Work

Diffusion Language Model Diffusion models are now considered a promising alternative to traditional autoregressive generation methods, as they support flexible generation orders and work through iterative refinement. The idea of using discrete diffusion for language modeling was first explored by Austin et al. (2021); Hoogeboom et al. (2021), which provided an initial framework for text modeling. Since then, various studies have been proposed to improve text diffusion models (Campbell et al., 2022; Li et al., 2022; Chen et al., 2023; Gong et al., 2023; Zheng et al., 2023; Sahoo et al., 2024; Shi et al., 2024; Ou et al., 2025; Xu et al., 2025b; Liu et al., 2025b; Zhang et al., 2025b; Zheng et al., 2025a; Pynadath et al., 2025; Havasi et al., 2025; Yang et al., 2025a; Wu et al., 2025c; Kang et al., 2025; Campbell et al., 2025). Larger models have been trained from scratch, such as SEDD (Lou et al., 2024), Plaid (Gulrajani and Hashimoto, 2023), MDGA (Ni et al., 2025) and LLaDA (Nie et al., 2025; Zhu et al., 2025a,b). Meanwhile, methods that adapt pretrained autoregressive models for diffusion, such as DiffuLLaMA (Gong et al., 2025a) and Dream (Ye et al., 2025c), have led to significant reductions in the training costs of dLLMs. DLLMs have exhibited promising capabilities in planning (Zhang et al.,

- 2023b; Ye et al., 2024, 2025a,b) and code generation (Khanna et al., 2025; Gong et al., 2025b; Xie et al., 2025). Recently, DeepMind (2025); Arriola et al. (2025); Ma et al. (2025a); Wang et al. (2025b); Cheng et al. (2025); Wu et al. (2025b,a); Gat et al. (2025); Liu et al. (2025a) have further advanced dLLMs by improving generation quality and inference efficiency.

Vision-Language Model (VLM) Autoregressive vision-language models have achieved remarkable success across various multimodal tasks. Pioneering works (Alayrac et al., 2022; Li et al., 2023b; Liu et al., 2023b; Dai et al., 2023; Liu et al., 2024a,b; Chen et al., 2024b,c; Dai et al., 2024; Wang et al., 2024c; OpenAI, 2024; Bai et al., 2025b; Guo et al., 2025) have demonstrated strong capabilities in visual understanding, reasoning, and instruction following by aligning pretrained vision encoders with large language models. Extending dLLMs to vision-language tasks remains an emerging research area. LaViDA (Li et al., 2025c) explores a discrete diffusion transformer with multi-view image encoding and masked-denoising objectives. LLaDA-V (You et al., 2025) applies masked diffusion with visual instruction tuning, which supports parallel decoding and controllable infilling. MMaDA (Yang et al.,

- 2025b) proposes a unified diffusion transformer that aims to align reasoning between text and vision through chain-of-thought supervision and reinforcement learning. Dimple (Yu et al., 2025) addresses the training instability of discrete diffusion by employing a hybrid autoregressive-then-diffusion paradigm. Lavida-O (Li et al., 2025b) introduces a unified masked diffusion model for multimodal understanding and generation, incorporates planning and iterative self-reflection in image generation and editing tasks. LLaDA-MedV (Dong et al., 2025) utilizes masked diffusion in medical applications, integrating imaging and textual data for diagnosis.

Vision-Language-Action Model (VLA) Building a generalist robotic policy has long been a central goal in embodied AI. Driven by the rapid advances in autoregressive Vision-Language, researchers have increasingly focused on Vision-Language-Action (VLA) models to leverage strong multimodal generalization for robotic control (Driess et al., 2023; Brohan et al., 2023; Black et al., 2024b; Kim et al.,

- 2024; Li et al., 2024c; Wen et al., 2025a; Team et al., 2025; Bjorck et al., 2025). Pioneering work such as RT-2 (Brohan et al., 2023) established the VLA paradigm by jointly fine-tuning VLMs on web-scale data and low-level robot demonstrations. OpenVLA (Kim et al., 2024) is further introduced as the first open-source VLA model. To address the computational demands of these models, subsequent work has introduced token compression (Pertsch et al., 2025) and action chunking (Zhao et al., 2023; Kim et al., 2025), enabling higher-frequency control and faster adaptation. Build upon diffusion models (Chi et al., 2023; Pearce et al.; Reuss et al., 2023) for action prediction, a VLM with an additional action expert architecture has also been actively explored to predict coherent continuous actions (Li et al., 2024c; Black et al., 2024a; Wen et al., 2025a; Bjorck et al., 2025; Shukor et al., 2025; Zhang et al., 2025c). While standard VLAs rely on autoregressive modeling, discrete diffusion VLA has also been recently studied for robotic control (Liang et al., 2025; Wen et al., 2025b) and autonomous driving (Cui et al.,
- 2025; Li et al., 2025a; Ma et al., 2025b; Xu et al., 2025a). DiscreteDiffusionVLA (Liang et al., 2025) and LLaDA-VLA (Wen et al., 2025b) have introduced discrete masked diffusion as an alternative by directly finetuning specific robotic tasks on an autoregressive dLLM backbone (Liang et al., 2025) or dVLM backbone (Wen et al., 2025b) without robotic pretraining.

## 3 Dream-VL: Diffusion-based VLM Built on Dream 7B

To fully and controllably analyze the model characteristics, we first train Dream-VL, a well-aligned diffusion-based VLM built upon Dream 7B (Ye et al., 2025c).

### 3.1 Setup

Multi-modal Training. Following MAmmoTH-VL (Guo et al., 2025), we collect 12M open-source multimodal data to train the VL model, which contains instruction-response pairs from diverse, real-world tasks such as mathematical problem-solving, OCR, and domain-specific reasoning. We incorporate a Qwen2ViT (Wang et al., 2024c) module to encode visual inputs into latent features, which are concatenated with text features as model input. The model is trained using the same discrete diffusion loss as Dream 7B, followed by a multi-stage training paradigm. The detailed training parameters of each stage are shown in Table 1.

Table 1: Training parameters of Dream-VL.

Stage-1 Stage-2 Stage-3 Dataset LCS Single Image (SI) Single, Multi-Image & Video #Samples 558K 10M 2M Vision Tower Qwen2ViT Qwen2ViT Qwen2ViT LLM Backbone Dream-v0-Instruct-7B Dream-v0-Instruct-7B Dream-v0-Instruct-7B Trainable Model Parameters Projector: 25.7M Full Model: 8.3B Full Model: 8.3B Batch Size 512 256 256 Model Max Length 8192 8192 8192 Learning Rate / Epoch 1e-3 / 1ep 1e-5 / 1ep + 5e-6 / 2ep 5e-6 / 1ep

Evaluation. Our experimental evaluation incorporates a wide range of vision-language understanding benchmarks: (1) Multidisciplinary Knowledge (MMStar (Chen et al., 2024a), MMMU (Yue et al., 2024a), MMMU-Pro (Yue et al., 2024b), SeedBench (Li et al., 2023a), MMBench (Liu et al.,

- 2025c), MMvet (Yu et al., 2023)); (2) Mathematical Reasoning (Mathverse (Zhang et al., 2024b), Mathvista (Lu et al., 2023)); (3) Chart/Doc Understanding (AI2D (Kembhavi et al., 2016), ChartQA (Masry et al., 2022), DocVQA (Mathew et al., 2020), InfoVQA (Mathew et al., 2021)); (4)

- Table 2: Performance comparison on multi-discipline knowledge and mathematical reasoning benchmarks. “Open Data” indicates whether the multimodal training data is open-source. Best results among diffusion models are bolded. SI indicates Single Image.

Open Multi-Discipline Knowledge Mathematical Reasoning Model LLM Backbone Data MMMU MMMU Pro MMStar MMBench SeedBench MathVista MathVerse Proprietary Models GPT-4o (OpenAI, 2024) - ✗ 69.1 49.7 64.7 82.1 76.2 63.8 50.2 Gemini-1.5 Pro (Reid et al., 2024) - ✗ 65.8 44.4 59.1 73.9 76.0 63.9 Claude 3.5 Sonnet (Anthropic, 2024) - ✗ 68.3 48.0 62.2 79.7 72.2 67.7 Autoregressive VLMs DeepSeek-VL2 (Wu et al., 2024) DeepSeek-MoE 27B ✗ 51.1 - 61.3 79.6 - 62.8 Qwen2.5-VL (Bai et al., 2025a) Qwen2.5-7B ✗ 58.6 - 63.9 83.5 - 68.2 49.2 InternVL3 (Zhu et al., 2025c) Qwen2.5-7B ✗ 62.7 - 68.2 83.4 - 71.6 39.8 MiMo-VL-RL (Yue et al., 2025) MiMo-7B ✗ 66.7 40.3 - 84.4 - 81.5 71.5 Cambrian-1 (Tong et al., 2024) LLaMA3-8B ✓ 42.7 14.7 - 74.6 73.3 49.0 Llava-CoT-11B (Xu et al., 2024) LLaMA3.2-11B ✓ 48.9 18.5 57.6 75.0 75.2 54.8 24.2 Molmo-8B-D (Deitke et al., 2024) Qwen2 7B ✓ 45.3 18.9 50.5 73.6 74.1 51.6 21.5 LLaVA-OV (SI) (Li et al., 2024b) Qwen2-7B ✓ 47.3 16.8 60.9 80.5 74.8 56.1 26.9 LLaVA-OV (Li et al., 2024b) Qwen2-7B ✓ 48.8 18.7 61.7 80.8 75.4 63.2 26.2 MAmmoTH-VL (SI) (Guo et al., 2025) Qwen2.5-7B ✓ 49.4 26.0 55.4 83.0 73.3 67.6 35.0 MAmmoTH-VL (Guo et al., 2025) Qwen2.5-7B ✓ 50.8 25.3 63.0 83.4 76.0 67.6 34.2 Diffusion VLMs LLaDA-V (You et al., 2025) LLaDA 8B ✓ 48.6 18.6 60.1 - 74.8 50.6 28.5 MMaDA (Yang et al., 2025b) LLaDA 8B ✗ 30.2 - - 68.5 64.2 - FUDOKI (Wang et al., 2025a) Janus-1.5B ✓ 34.7 - - 73.6 68.2 - Dimple (Yu et al., 2025) Dream 7B ✓ 45.2 - - 74.6 - 42.4 LaViDa-D (Li et al., 2025c) Dream 7B ✓ 42.6 - - 73.8 - 42.1 24.1 Dream-VL (SI) Dream 7B ✓ 51.8 25.8 60.2 83.6 75.9 64.5 30.7 Dream-VL Dream 7B ✓ 52.2 26.0 59.9 83.0 76.4 63.1 31.5

##### Table 3: Performance comparison on chart & document understanding and real-world multimodal interactions benchmarks. “Open Data” indicates whether the multimodal training data is open-source. Best results among diffusion models are bolded.

Open Chart & Doc Understanding Interact Model LLM Backbone Data AI2D ChartQA InfoVQA DocVQA RealWorldQA Proprietary Models

GPT-4o (OpenAI, 2024) - ✗ 94.2 85.7 79.2 92.8 76.5 Gemini-1.5 Pro (Reid et al., 2024) - ✗ 94.4 87.2 81.0 93.1 70.4 Claude 3.5 Sonnet (Anthropic, 2024) - ✗ 94.7 90.8 49.7 95.2 60.1

Autoregressive VLMs DeepSeek-VL2 (Wu et al., 2024) DeepSeek-MoE 27B ✗ 81.4 86.0 61.3 - 68.4 Qwen2.5-VL (Bai et al., 2025a) Qwen2.5-7B ✗ 83.9 87.3 82.6 95.7 68.5 InternVL3 (Zhu et al., 2025c) Qwen2.5-7B ✗ 85.2 86.6 76.8 92.7 70.8 MiMo-VL-RL (Yue et al., 2025) MiMo-7B ✗ 83.5 91.7 88.0 95.7 Cambrian-1 (Tong et al., 2024) LLaMA3-8B ✓ 73.0 73.3 41.6 77.8 64.2 Llava-CoT-11B (Xu et al., 2024) LLaMA-3.2-11B ✓ - 67.0 44.8 - Molmo-8B-D (Deitke et al., 2024) Qwen2 7B ✓ 81.0 84.1 72.6 92.2 70.7 LLaVA-OV (SI) (Li et al., 2024b) Qwen2-7B ✓ 81.6 78.8 65.3 86.9 65.5 LLaVA-OV (Li et al., 2024b) Qwen2-7B ✓ 81.4 80.0 68.8 87.5 66.3 MAmmoTH-VL (SI) (Guo et al., 2025) Qwen2.5-7B ✓ 83.4 85.9 74.8 93.8 71.3 MAmmoTH-VL (Guo et al., 2025) Qwen2.5-7B ✓ 84.0 86.2 73.1 93.7 69.9

Diffusion VLMs LLaDA-V (You et al., 2025) LLaDA 8B ✓ 77.8 78.3 66.3 83.9 63.2 Dimple (Yu et al., 2025) Dream 7B ✓ 74.4 63.4 - - LaViDa-D (Li et al., 2025c) Dream 7B ✓ 69.0 61.0 36.2 56.1 -

Dream-VL (SI) Dream 7B ✓ 80.6 86.8 81.4 94.4 68.4 Dream-VL Dream 7B ✓ 81.2 84.5 81.0 94.4 66.3

Multimodal Interaction (RealworldQA (x.ai, 2024), WildVision (Lu et al., 2024), Llava-WilderSmall (Li et al., 2024a)); and (5) Multi-image/Video Understanding (MuirBench (Wang et al.,

- 2024a), SeedBench-Video (Li et al., 2023a), MLVU (Zhou et al., 2024a), VideoMME (Fu et al., 2024)).

##### Table 4: Performance comparison on multi-image and video understanding benchmarks. “Open Data” denotes open-source multimodal training data. Best results among diffusion models are bolded.

Open Multi-image and Video Understanding Model LLM Backbone Data Seed-video VideoMME MuirBench MLVU Proprietary Models GPT-4o (OpenAI, 2024) - ✗ - 71.9 68.0 64.6 Gemini-1.5 Pro (Reid et al., 2024) - ✗ - - - Claude 3.5 Sonnet (Anthropic, 2024) - ✗ - - - Autoregressive VLMs DeepSeek-VL2 (Wu et al., 2024) DeepSeek-MoE 27B ✗ - - - Qwen2.5-VL (Bai et al., 2025a) Qwen2.5-7B ✗ - 65.1 - InternVL3 (Zhu et al., 2025c) Qwen2.5-7B ✗ - 66.3 - MiMo-VL-RL (Yue et al., 2025) MiMo-7B ✗ - 67.4 - Cambrian-1 (Tong et al., 2024) LLaMA3-8B ✓ - - - Llava-CoT-11B (Xu et al., 2024) LLaMA-3.2-11B ✓ - - - Molmo-8B-D (Deitke et al., 2024) Qwen2 7B ✓ - - - LLaVA-OV (Li et al., 2024b) Qwen2-7B ✓ 56.9 58.2 41.8 64.7 MAmmoTH-VL (Guo et al., 2025) Qwen2.5-7B ✓ 57.1 58.8 55.1 64.7 Diffusion VLMs

LLaDA-V LLaDA 8B ✓ 53.7 56.1 48.3 59.4 Dream-VL Dream 7B ✓ 58.1 61.5 51.2 61.1

- 3.2 Experimental Results

Dream-VL exhibits substantial improvements over existing diffusion-based VLMs. Tables 2-

- 4 show the experimental results. Dream-VL significantly outperforms previous diffusion-based VLMs, demonstrating the effectiveness of our approach. Compared to LLaDA-V, Dream-VL achieves superior performance while using comparable training data (∼12M in Dream-VL vs. ∼13M in LLaDA-V). This improvement can be attributed to the enhanced capabilities of the base Dream model and differences

High-level action planning

Low-level action planning

[Figure 7]

[Figure 8]

Action Planner

Action Planner

Dream-VL

Dream-VLA

Long action chain…

Chunk 1

…

|Action 1|
|---|

|Action 2|
|---|

|Action n|
|---|

|!!!" …|
|---|

|Chunk 2|
|---|

|Chunk n|
|---|

{

|Task: pick up cups on the table|
|---|

|Task: pick up cups on the table|
|---|

{

{

Long action chain

“action”:

“action”:

“action”:

“move” “params”:

“place-on” “params”:

“grasp” “params”:

[Figure 9]

[Figure 10]

|Δx | Δy | Δz | Δ⍺ | Δβ | Δγ | G|
|---|

[“table1”] }

[“table2”] }

[“cup1”] }

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

Execution

Execution

Environment

Environment

- Figure 2: High-level and low-level action planning. Both forms of planning require multi-turn interaction with the environment, typically modeled as a Markov Decision Process. The task requires the model to perform long-horizon planning with high consistency. High-level planning operates in a symbolic or semantic action space, producing abstract commands. Low-level planning directly focuses on precise robot behavior, handling fine-grained, continuous control such as grasping and motion trajectories. Additional robotic training is required for bridging the vision-language model with executable robotic actions.

in training strategies. When compared to other Dream-based VLMs (e.g., Dimple/LaViDa), the performance gains primarily result from the substantially larger training dataset (∼12M vs. ∼2M samples), highlighting the importance of sufficient training for diffusion-based VLMs to fully explore their potential.

Dream-VL achieves competitive performance with autoregressive VLMs and shows distinct advantages. Compared to state-of-the-art autoregressive models trained on similar open datasets, Dream-VL achieves competitive overall performance, with notable advantages on multi-discipline visual reasoning (MMMU) and document understanding tasks (DocVQA, ChartQA), demonstrating its distinct advantages. These results suggest that diffusion-based modeling can match or exceed autoregressive approaches in some scenarios. However, a performance gap remains when compared to leading closed-data autoregressive models (e.g., Qwen2.5-VL), indicating opportunities for further improvement through data scaling and model optimization.

### 3.3 Analysis on Planning Ability

Most of the aforementioned tasks primarily reflect the model’s general understanding capabilities. In this section, we investigate the effectiveness of dLLMs as VL backbones for planning tasks. Specifically, we focus on two types of planning: high-level action planning that operates in a symbolic or semantic action space, and low-level action planning that focuses on precise control of robot behavior. The two types of planning are shown in Figure 2.

#### 3.3.1 Dream-VL for High-level Action Planning

Task Formulation. High-level action planning requires generating a sequence of symbolic actions {a1,a2,...,aT} that transform an initial scene state s0 to a goal state sg, given visual observations I

and a natural-language instruction l. Each action at represents a structured, discrete operation (e.g., pick(block1), place(block2, left)) executable by downstream controllers. This task evaluates whether a model can understand the scene, ground entities, and reason over multi-step procedures.

Setup. We evaluate Dream-VL on the ViPlan benchmark (Merler et al., 2025). In each task instance, the input consists of an image I depicting the scene and a language instruction l describing the goal. The benchmark operates in two modes: (1) Grounding mode, where the model outputs a binary response indicating whether a state description is satisfied in the current scene, and (2) Planning mode, where the model generates a sequence of symbolic actions to achieve the specified goal. ViPlan comprises two domains: BlockWorlds, involving colored blocks manipulated under symbolic relational predicates (e.g., on, clear, incolumn), and Household, featuring everyday objects in complex spatial scenarios. Each domain includes three difficulty tiers (simple, medium, hard), varying in scene complexity, required plan length, and subtlety of visual cues. Beyond grounding relations in the image, the model is also required to produce structured, symbolic actions that are similar to function or tool calls. Thus,

|Grounding Mode|Planning Mode|
|---|---|
|Output Format: The model outputs a binary “yes” or “no” indicating whether the state description is satisfied in the current scene. The grounding results assist the planner in producing executable actions.<br><br>Example: Input: [image] + “Is the hardback on top of the shelf?” Output: "No"<br><br>|Output Format: The model outputs a JSON list of symbolic actions. For each iteration, only the first action is executed; the scene is updated and fed into the next iteration until the goal is satisfied or the maximum steps are reached.<br><br>Example: Input: [image] + “Place the hardback on shelf.” Output:<br><br>{"plan": [{"action": "navigate-to",<br><br>"parameters": ["shelf_1"]}, {"action": "place-on",<br><br>"parameters": ["hardback_1"]}] }|
|Domains: (1) BlockWorlds — colored blocks are arranged and manipulated under symbolic relational predicates (on, clear, incolumn); (2) Household — everyday objects in a home environment must be manipulated and moved in more complex spatial and commonsense scenarios.| |

Figure 3: Output formats and domains of the ViPlan benchmark.

ViPlan evaluates a hybrid capability that blends vision-textual understanding and planning, and precise symbolic planning. We report task success rate (task accuracy), measured by whether the final state satisfies the goal description, and action accuracy, defined as the proportion of valid actions among all generated actions.

Results and Analysis. We evaluate Dream-VL on the ViPlan benchmark in a zero-shot manner, comparing it against both autoregressive and diffusion-based vision-language models. The results are presented in Figure 4. We have the following observations:

- • Compared to autoregressive VLMs, Dream-VL achieves competitive performance with InternVL3. While a performance gap remains relative to Qwen2.5-VL-Instruct, which has undergone extensive vision-language alignment, Dream-VL demonstrates promising capabilities as a diffusion-based alternative. More importantly, when controlling for training data and procedures, specifically comparing Dream-VL with MAmmoTH-VL-7B, which shares similar training recipes, Dream-VL achieves higher accuracy across most settings. This controlled comparison validates the benefits of the diffusion-based architecture for high-level planning tasks.
- • Within the diffusion model family, Dream-VL exhibits substantial improvements and consistently outperforms LLaDA-V across all evaluation modes and difficulty tiers. We attribute this advantage to the strong planning capabilities inherited from the base Dream 7B model, which is good at text planning tasks such as Sudoku.
- • Across both model families, Dream-VL narrows the performance gap with top autoregressive baselines, with particularly notable advantages in the Household grounding task. This suggests that diffusion-based modeling may offer specific benefits for complex visual grounding scenarios that require understanding spatial and commonsense relations in realistic environments.

#### Highlights

- • Under controlled comparison, Dream-VL outperforms the autoregressive baseline (MAmmoTH-VL-7B) on most grounding and planning scenarios, highlighting the architectural advantages of diffusion-based modeling for high-level planning.
- • Dream-VL achieves state-of-the-art performance among diffusion-based VLMs, consistently outperforming LLaDA-V across all evaluation modes.

###### Note: values > 0.40 are clipped ( ) BlockWorlds (grounding)

0.40

0.87

0.84

|Action Simple Action Medium Action Hard Task Simple Task Medium Task Hard<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>|
|---|

AR Diffusion

0.35

0.34

0.32

0.30

0.26

0.25

0.24

Accuracy

0.20

0.16

0.15

0.12

0.12

0.11

0.10

0.07

0.07

0.05

0.04

0.04

0.03

0.03

0.02

0.01

0 0

0 0 0 0 0 0 0 0 0 0

0

0

0

0

0

0

0.00

InternVL3-8BQwen2.5-VL-7B-Instruct LLaVA-OV-7BMAmmoTH-VL-7B Dream-VL-7B LLaDA-V-8B

###### BlockWorlds (planning)

0.200

AR Diffusion

|Action Simple Action Medium Action Hard Task Simple Task Medium Task Hard<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>|
|---|

0.18

0.175

0.17

0.150

0.125

Accuracy

0.11

0.10

0.100

0.09

0.08

0.08

0.075

0.06

0.050

0.05

0.04

0.03

0.02

0.025

0.01

0 0.00 0.00 0.01

0

0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0

0

0

0.000

InternVL3-8BQwen2.5-VL-7B-Instruct LLaVA-OV-7BMAmmoTH-VL-7B Dream-VL-7B LLaDA-V-8B

###### HouseHold (grounding)

AR Diffusion

Action Simple Action Medium Action Hard Task Simple Task Medium Task Hard

0.25

| |
|---|

0.23

| |
|---|

0.21

| |
|---|

0.20

| |
|---|

0.18

0.17

| |
|---|

Accuracy

0.15

0.10

0.10

0.10

0.09

0.09

0.08

0.08

0.07

0.06

0.05

0.05

0.05

0.05

0.04 0.04 0.04 0.04

0.04

0.03

0.02

0.02

0 0

0 0 0 0 0 0

0 0 0

0

0.00

InternVL3-8BQwen2.5-VL-7B-Instruct LLaVA-OV-7BMAmmoTH-VL-7B Dream-VL-7B LLaDA-V-8B

###### HouseHold (planning)

0.8

AR Diffusion 0.71

0.72

Action Simple Action Medium Action Hard Task Simple Task Medium Task Hard

0.71

0.7

| |
|---|

0.65

0.65

0.63

| |
|---|

0.61

0.61

0.59

0.6

| |
|---|

0.56

| |
|---|

0.5

| |
|---|

Accuracy

0.41

0.4

0.32

0.30

0.3

0.2

0.16

0.12

0.11

0.10

0.1

0.08

0.04 0.04 0 0 0 0 0

0

0

0

0 0 0 0

0 0 0 0

0.0

InternVL3-8BQwen2.5-VL-7B-Instruct LLaVA-OV-7BMAmmoTH-VL-7B Dream-VL-7B LLaDA-V-8B

Figure 4: Performance comparison for VLMs on the ViPlan benchmark (Merler et al., 2025).

#### 3.3.2 Dream-VL for Low-level Action Planning

Task Formulation. Low-level action planning requires predicting continuous motor controls that can be directly executed by controllers. Given visual observations I and a natural-language instruction l, the model generates a sequence of robot control actions {a1,a2,...,aT}. Following Brohan et al. (2023), for each action, the model predicts 7-dimensional robot control to indicate the delta end-effector pose (∆pos, ∆rotation, gripper).

Setup. We conducted experiments on the LIBERO (Liu et al., 2023a) benchmark. We use the LIBERO-Goal subtask, which tests procedural learning through varying task goals with fixed objects, and the LIBERO-Long subtask, which includes 10 long-horizon tasks with different objects, layouts, and goals. To see the ability of Dream-VL in such low-level action planning, we first discretize each dimension of the robot actions separately into one of 256 bins following (Brohan et al., 2023; Kim et al., 2024). Then we finetune with autoregressive loss for Qwen2.5-VL and discrete diffusion loss for Dream-VL. We follow the same preprocessed pipeline as in OpenVLA (Kim et al., 2024) and do not use wrist camera image and robot proprioceptive state in this experiment. Both Qwen2.5-VL and Dream-VL are directly finetuned on the LIBERO data without robotic pretraining. We set the training action chunk to be 8 to be consistent with OpenVLA, and compare the three models in Table 5. The diffusion timestep is set to be 1 for Dream-VL. Additionally, we launch another comparison between Qwen2.5-VL and Dream-VL by setting the training action chunk size to be larger than 12 and varying the predicted action chunk during inference time to see the performance of the two models in long-horizon action chunk prediction, as shown in Figure 5.

- Table 5: Performance comparison between autoregressive and diffusion-based VLMs on LIBERO benchmarks. Despite Qwen2.5-VL achieving stronger results on standard vision-language tasks (Table 24), Dream-VL substantially outperforms it on robotic planning tasks, suggesting that diffusion-based modeling offers advantages for long-horizon action prediction.

Model VLM Type Size Robotics Pretraining LIBERO-Goal (%) LIBERO-Long (%)

OpenVLA AR 7B ✓ 79.2 53.7 Qwen2.5-VL AR 7B ✗ 68.0 34.0 Dream-VL Diffusion 7B ✗ 83.2 59.0

LIBERO-Goal Performance

LIBERO-Long Performance

Speedup (Relative to Qwen2.5-VL)

90

30

Dream-VL

Dream-VL

Dream-VL

27.4×

Qwen2.5-VL

Qwen2.5-VL

Qwen2.5-VL

60

25

23.4×

22.6×

80

SuccessRate(%)

SuccessRate(%)

19.0×

Speedupratio

20

14.0×

40

15

13.1×

10.7×

70

10

7.9×

6.1×

3.7×

5

20

60

0

2 4 6 8 10 12

2 4 6 8 10 12

2 4 6 8 10 12

Action Chunk

Action Chunk

Action Chunk

Figure 5: Comparison of model performance and speed with varying action chunk sizes.

Results and Analysis. We have the following observations:

- • As shown in Table 5, we observed a significant performance disparity between architectures. Although Qwen2.5-VL generally exhibits stronger performance on standard vision-language benchmarks in Table 2, after SFT, it achieves only a 34.0% success rate on LIBERO-Long. In contrast, the diffusion-based Dream-VL achieves 59.0%, which also outperforms the large-scale robotic-pretrained OpenVLA. This result suggests that the diffusion-based VLM holds superior potential for learning long-horizon planning compared to autoregressive backbones.
- • As shown in Figure 5, Qwen2.5-VL tends to accumulate errors more easily when generating too many actions at once. This can be seen from the fact that Qwen2.5-VL reaches the maximum benefit of action chunking faster than Dream-VL. For instance, the optimal chunk sizes for Qwen2.5-VL are 3 for LIBERO-Goal and 5 for LIBERO-Long, compared to 9 and 10 for Dream-VL. After reaching the maximum, Qwen2.5-VL is affected by error accumulation from autoregressive generation. When this error accumulation becomes excessive, action chunking can even have a negative impact. For example, predicting and executing 9 or more actions at once in the LIBERO-Goal task performs worse than predicting and executing just one action at a time.
- • Dream-VL can maximize its parallel generation capability when predicting low-level actions. We found that for predicting multiple low-level actions, only one diffusion step is needed to achieve good performance, resulting in a significant speed advantage. This is different from text generation, where predicting multiple tokens in one step is often insufficient. As shown in the figure on the right, generating 12 actions at once can achieve a 27× speedup.

#### Highlights

- • Unlike autoregressive models (e.g., Qwen2.5-VL) where performance degrades with larger chunk sizes due to error accumulation, dVLMs such as Dream-VL maintain robustness over longer horizons.
- • Dream-VL exhibits exceptional efficiency in predicting low-level actions; remarkably, it requires only a single diffusion step to achieve competitive performance, delivering a 27× speedup over autoregressive generation.

## 4 Dream-VLA: Dream-VL with Large-scale Robotic Pretraining

Given the potential of Dream-VL in low-level action planning, we further conduct a large-scale robotic pretraining over Dream-VL to obtain a general vision-language-action (VLA) model.

### 4.1 Setup.

Robotic Pretraining. Following OpenVLA (Kim et al., 2024), we use a large, diverse dataset of 970k robot manipulation trajectories from the Open-X Embodiment dataset (O’Neill et al., 2024b),

Parallel Decoding

Next-token Prediction

Action Expert

AR-based LLM

AR-based LLM

Parallel Decoding

Action Chunks:

VIT

VIT

𝜋 -FAST

Diffusion-based LLM (dLLM)

𝜋

Bidirectional Attention

Parallel Decoding

Parallel Decoding

VIT

Empty Action

Get cups and …

Causal to Bidirectional Attention

|Δx | Δy | Δz | Δ⍺ | Δβ | Δγ | G|
|---|

|[Figure 15]|
|---|

AR-based LLM System 1

AR-based LLM

Robotic Action

VIT

VIT

Dream-VLA

OpenVLA-OFT

GR00T

Figure 6: Architectural comparison of various VLA models.

which spans a wide range of robot embodiments, tasks, and scenes. Similar to training Dream-VL, we also use the same discrete diffusion loss as in the backbone language model Dream 7B during robotic pretraining. We continue training Dream-VL with a global batch size of 1024, a constant learning rate of 1e-5, an action chunk size of 8, and a total of 610,000 training steps.

Evaluation. To assess Dream-VLA’s capabilities, we conduct experiments across three diverse robotic manipulation benchmarks:

- • LIBERO (Liu et al., 2023a), which consists of four task suites designed for studying lifelong learning in robotic manipulation: LIBERO-Spatial (same objects, different spatial layouts), LIBERO-Object (fixed layout, different objects), LIBERO-Goal (fixed objects and layout, different goals), and LIBERO-Long (longer tasks that span multiple objects, layouts, and operations). LIBERO is built on a Franka arm with diverse scenes and expert demonstrations for each task;
- • SimplerEnv (Li et al., 2024e), a simulated evaluation suite for real robot setups, including Google Robot tasks with Visual Matching and Variant Aggregation metrics across diverse visual variations, and WidowX robot tasks aligned with BridgeData-V2 (Walke et al., 2023);
- • Real-World PiPER Robot, a tabletop manipulation evaluation on our PiPER platform with a single front-facing camera observing the robot and the table workspace. Following our concurrent work, we finetune only on simulation data synthesized with domain randomization (e.g., textures, scene, lighting, and camera pose), and evaluate in two settings: (i) sim2sim evaluation in simulation on picking up varying target objects randomly placed on the table in the presence of multiple distractors without test-time domain randomization, and (ii) zero-shot sim2real transfer on the real PiPER robot without any real-world finetuning.

Finetuning. Following standard finetuning protocols (Kim et al., 2025), our policy takes as input RGB observations—specifically, one third-person camera and one wrist-mounted camera for LIBERO, or a single third-person camera for SimplerEnv—along with natural language task descriptions. Proprioceptive end-effector positions are optionally provided when available. When finetuning for downstream tasks, we can use any finetuning objective without changing the model architecture (e.g., L1 regression, continuous diffusion, discrete in OpenVLA-OFT (Kim et al., 2025), and discrete diffusion (Liang et al.,

- 2025)). To predict as fine-grained a move as possible, we prioritize a continuous action approach. By

default, we use a Flow Matching (Lipman et al., 2023) loss similar to π0 (Black et al., 2024a) but without introducing a separate action expert, and the flow matching timesteps are set to 4 during inference. For SimplerEnv–Fractal and SimplerEnv–Bridge, we fine-tune on Fractal (Brohan et al., 2022) and BridgeData V2 (Walke et al., 2023), respectively. We use an action chunk size of 8 for LIBERO and 5 for SimplerEnv to match the widely used settings. The fine-tuning is launched with LoRA (Hu et al., 2022) with rank 32, batch size 64, and learning rate 1e-4. We train Dream-VLA on LIBERO for 100k steps and 200k for SimplerEnv, without careful checkpoint selection.

- Table 6: Success rate (%) on LIBERO benchmark suites. We compare Dream-VLA with state-of-the-art policies and VLA models. “-” indicates the metric is not reported in the original paper.

Model VLM Type Spatial Object Goal Long Average Diffusion Policy (Chi et al., 2023) - 78.3 92.5 68.3 50.5 72.4 MDT (Reuss et al., 2024) - 78.5 87.5 73.5 64.8 76.1 Seer (scratch) (Tian et al., 2025) - - - - 78.7 Octo-Base (Ghosh et al., 2024) - 78.9 85.7 84.6 51.1 75.1 Seer (fine-tuned) (Tian et al., 2025) - - - - 87.7 DiT Policy (Hou et al., 2025) - 84.2 96.3 85.4 63.8 82.4 TraceVLA (Zheng et al., 2025b) AR 84.6 85.2 75.1 54.1 74.8 SpatialVLA (Qu et al., 2025) AR 88.2 89.9 78.6 55.5 78.1 π0 + FAST (Pertsch et al., 2025) AR 96.4 96.8 88.6 60.2 85.5 π0 (Black et al., 2024a) AR 96.8 98.8 95.8 85.2 94.2 OpenVLA (Kim et al., 2024) AR 84.7 88.4 79.2 53.7 76.5 CoT-VLA (Zhao et al., 2025) AR 87.5 91.6 87.6 69.0 81.1 OpenVLA-OFT (Kim et al., 2025) AR 97.6 98.4 97.9 94.5 97.1 GR00T-N1 (Bjorck et al., 2025) AR 94.4 97.6 93.0 90.6 93.9 Discrete Diffusion VLA (Liang et al., 2025) AR 97.2 98.6 97.4 92.0 96.3 Dream-VLA (Ours) Diffusion 97.6 98.8 97.2 95.0 97.2

- Table 7: Real-world performance evaluation on WidowX Robot tasks. We report the Grasping Success Rate (Grasp) and Task Success Rate (Succ) for each task, along with the average task success rate. All values are in percentages (%).

Spoon on Towel Carrot on Plate Stack Green Block Eggplant in Basket Average Grasp Success Grasp Success Grasp Success Grasp Success Success Overall

Method

Octo-Base (Ghosh et al., 2024) 50.0 33.0 50.0 25.0 29.2 0.0 40.0 23.3 20.3 31.3 RoboVLM (Li et al., 2024d) 37.5 20.8 33.3 25.0 8.3 8.3 0.0 0.0 13.5 16.7 SpatialVLA (Qu et al., 2025) 20.8 16.7 29.2 25.0 62.5 29.2 100.0 100.0 42.7 47.9 RT-1-X (O’Neill et al., 2024a) 4.2 0.0 16.7 0.0 0.0 0.0 3.3 0.0 0.0 3.0 OpenVLA (Kim et al., 2024) 4.1 0.0 33.0 0.0 12.5 0.0 8.3 4.1 1.0 7.8 OpenVLA-OFT (Kim et al., 2025) 50.0 12.5 41.7 4.2 70.8 20.8 91.7 37.5 18.8 41.2 π0 (Black et al., 2024a) 45.8 29.1 25.0 0.0 50.0 16.6 91.6 62.5 27.1 40.1 π0+FAST (Pertsch et al., 2025) 62.5 29.1 58.5 21.9 54.0 10.8 83.3 66.6 32.1 48.3 GR00T-N1 (Bjorck et al., 2025) 83.3 62.5 54.2 45.8 70.8 16.7 41.7 20.8 36.5 49.5 DiscreteDiffusionVLA (Liang et al., 2025) 70.8 29.2 58.3 29.2 62.5 20.8 91.7 70.8 37.5 54.2 LLaDA-VLA (Wen et al., 2025b) - 56.9 - 76.3 - 30.6 - 58.3 55.5 Dream-VLA (Ours) 91.7 79.2 58.3 41.7 79.2 20.8 100.0 100.0 60.4 71.4

### 4.2 Main Results

LIBERO results. We evaluate Dream-VLA on the LIBERO benchmark, with results shown in Table 6. Our model demonstrates strong performance across four task suites, reaching 97.6% on LIBERO-Spatial, 98.8% on LIBERO-Object, 97.2% on LIBERO-Goal, and 95.0% on LIBERO-Long, yielding an overall average of 97.2%. Dream-VLA shows that a model pretrained on a diffusion backbone can achieve superior performance than previous state-of-the-art autoregressive VLA model (OpenVLA-OFT, 97.1% average). These results show that diffusion-based modeling offers significant potential for vision-language-action tasks, providing a strong alternative approach to VLA modeling.

WidowX Robot results. Table 7 presents real-world evaluation results on the WidowX robot across four manipulation tasks. Dream-VLA achieves an overall average performance of 71.4%, establishing new state-of-the-art results among VLA models. Our approach substantially outperforms diffusion/flowmatching-based policies such as π0 (40.1%) by 31.3 percentage points. When compared to autoregressive baselines, Dream-VLA surpasses leading methods such as OpenVLA-OFT (33.3%), demonstrating the effectiveness of diffusion-based modeling. Among diffusion-based approaches, Dream-VLA outperforms DiscreteDiffusionVLA by 17.2 percentage points. Unlike DiscreteDiffusionVLA, which adapts diffusion modeling from pretrained autoregressive VLA model, Dream-VLA builds from dLLM (Dream-7B) through sufficient vision-language alignment followed by VLA pretraining, thereby fully unlocking the potential of dLLMs for robotic manipulation tasks. The per-task results reveal consistent gains across both grasp and task success metrics. Dream-VLA demonstrates particularly strong performance

- Table 8: Performance evaluation on Google Robot tasks. All values are success rates in percentages (%). “-” denotes missing data.

Visual Matching Variant Aggregation Pick Coke Mv Near Drawer Avg Pick Coke Mv Near Drawer Avg Overall

Method

- RT-1-X (O’Neill et al., 2024a) 56.7 31.7 59.7 53.4 49.0 32.3 29.4 39.6 46.5
- RT-2-X (O’Neill et al., 2024a) 78.7 77.9 25.0 60.7 82.3 79.2 35.3 64.3 62.5 Octo-Base (Ghosh et al., 2024) 17.0 4.2 22.7 16.8 0.6 3.1 1.1 1.1 9.0 OpenVLA (Kim et al., 2024) 16.3 46.2 35.6 27.7 54.5 47.7 17.7 39.8 33.8 HPT (Wang et al., 2024b) 56.0 60.0 24.0 46.0 - - - - Moto (Chen et al., 2025) 74.0 60.4 43.1 59.2 - - - - RoboVLM (Li et al., 2024d) 77.3 61.7 43.5 63.4 75.6 60.0 10.6 51.3 57.4 TraceVLA (Zheng et al., 2025b) 28.0 53.7 57.0 42.0 60.0 56.4 31.0 45.0 43.5 π0 (Black et al., 2024a) 72.7 65.3 38.3 58.8 75.2 63.7 25.6 54.8 56.8 π0+FAST (Pertsch et al., 2025) 75.3 67.5 42.9 61.9 77.6 68.2 31.3 59.0 60.5 OpenVLA-OFT (Kim et al., 2025) 72.3 69.6 47.2 63.0 65.3 59.0 12.2 45.5 54.3 GR00T-N1 (Bjorck et al., 2025) 47.0 70.0 18.1 45.0 78.8 62.5 13.2 51.5 48.4 Discrete Diffusion VLA (Liang et al., 2025) 85.4 67.5 60.6 71.2 82.5 64.6 23.6 56.9 64.1 Dream-VLA (Ours) 80.3 78.3 40.7 66.5 70.5 66.3 27.0 54.6 60.5

on tasks such as Put Eggplant (100% task success) and Put Spoon on Towel (79.2% task success), highlighting its capability in fine-grained manipulation.

Google Robot results. Table 8 reports performance on Google Robot tasks in the SimplerEnv benchmark. Our model obtains an overall average of 60.5%, performing on par with π0+FAST and outperforming widely-adopted methods including π0 (56.8%), OpenVLA-OFT (54.3%), and GR00T-N1 (48.4%). Within the Visual Matching setting, Dream-VLA exhibits particularly strong results on Pick Coke (80.3%) and Move Near (78.3%) tasks. When examining the more challenging Variant Aggregation setting, Dream-VLA achieves performance (54.6%) competitive with strong baseline π0-FAST (59.0%) and Discrete Diffusion VLA (56.9%). These results show that Dream-VLA can generalize effectively across different visual conditions and tasks.

### 4.3 Effectiveness of Robotic Pretraining

Figure 7 shows the impact of robotic pretraining. The result shows that robotic pretraining consistently improves fine-tuning outcomes across the majority of tasks, demonstrating its effectiveness for enhancing VLA capabilities. The magnitude of improvement varies across different tasks, with many showing substantial gains from pretraining, while a few tasks such as LIBERO-Goal and Stack Green Block On Yellow exhibit more modest improvements. The overall trend confirms that large-scale robotic pretraining provides valuable prior knowledge that transfers effectively to downstream tasks.

Effect of Robotic Pretraining on Success Rate (%)

+41.7%

|Dream-VL (Finetuned)<br><br>| |
|---|
<br><br>Dream-VLA (Finetuned)|
|---|

+15.0% +5.2%

100

100.0%

+33.4% 97.2%

95.0%

92.0%

SuccessRate(%)

75

80.0%

79.2%

+22.9%

60.4%

+12.5%

58.3%

50

45.8%

41.7%

+4.1%

37.5%

25

29.2%

20.8%

16.7%

0 Spoon on Towel Carrot on Plate Green on Yellow Eggplant in Basket Bridge Overall LIBERO-Long LIBERO-Goal

##### Figure 7: Performance gains from robotic pretraining across diverse tasks. Flow matching loss is used for all the downstream task fine-tuning.

### 4.4 What can we get from Dream-VLA?

Architectural Consistency & Superior Performance. For AR-based VLA backbones such as OpenVLA, downstream adaptation often requires structural modifications (i.e., attention mask adjustments) to enable action chunking, which is critical for AR-based models as it enhances both inference efficiency and task success rates. In contrast, DreamVLA, as a diffusion-based backbone, inherently supports action chunking without requiring architectural changes. In fact, the model architecture remains consistent starting from the LLM stage, thereby minimizing performance loss caused by structural variations across stages. As a result (Table 9), we find that Dream-VLA consistently achieves superior downstream performance across various finetuning objectives.

- Table 9: Comparison of OpenVLA-OFT and Dream-VLA with different SFT training objectives on SimplerEnv-WidowX benchmark.

Spoon on Towel Carrot on Plate Stack Green Block Eggplant in Basket Overall WidowX Robot Grasp Success Grasp Success Grasp Success Grasp Success Success OpenVLA-OFT

w/ L1 62.5 20.8 45.8 20.8 66.7 16.7 100.0 87.5 36.5 w/ Discrete 91.7 29.2 58.3 12.5 58.3 8.3 95.8 75.0 31.3 w/ Cont. Diffusion 50.0 12.5 33.3 0.0 37.5 0.0 66.7 4.2 4.2 w/ Disc. Diffusion 75.0 37.5 58.3 37.5 58.3 12.5 87.5 50.0 34.4 w/ Flow Matching 58.3 12.5 45.8 4.2 37.5 0.0 91.7 25.0 10.4

###### Dream-VLA

w/ L1 75.0 62.5 66.7 54.2 79.2 8.3 100.0 100.0 56.3 w/ Discrete 70.8 37.5 62.5 41.7 79.2 8.3 95.8 91.7 44.8 w/ Cont. Diffusion 79.2 66.7 66.7 54.2 95.8 16.7 91.7 91.7 57.3 w/ Disc. Diffusion 79.2 45.8 62.5 45.8 83.3 25.0 100.0 87.5 51.0 w/ Flow Matching 91.7 79.2 58.3 41.7 79.2 20.8 100.0 100.0 60.4

Accelerated Convergence in Fine-tuning. Due to the absence of architectural modifications, we find the model tends to converge faster and achieves a lower loss in finetuning as compared to OpenVLA-OFT. The difference is most pronounced under discrete diffusion finetuning because it shares the same training objective with LLM, VLM, and VLA pretraining.

|Flow M|atching|
|---|---|
| ||Dream-VLA<br><br>OpenVLA-OFT|
|---|
|
| | |
| | |

|Continue|Diffusion|
|---|---|
|| |
|---|
|Dream-VLA OpenVLA-OFT|
| | |
| | |

|Discrete|Diffusion|
|---|---|
| ||Dream-VLA<br><br>OpenVLA-OFT|
|---|
|
| | |
| | |
| | |

L1 Regression

1.0

1.0

1.0

- 0
- 1
- 2
- 3
- 4
- 5

|Dream-VLA<br><br>OpenVLA-OFT|
|---|

0.8

0.8

0.8

0.6

0.6

0.6

Loss

0.4

0.4

0.4

0.2

0.2

0.2

0.0

0.0

0.0

0 20k 40k 60k 80k 100k

0 50k 100k 150k 200k

0 50k 100k 150k 200k

0 20k 40k 60k 80k 100k

Step

Step

Step

Step

Figure 8: Loss curves of Dream-VLA and OpenVLA-OFT under different training objectives.

#### Highlights

- • Dream-VLA achieves a new state-of-the-art on the WidowX Robot with 71.4% overall average score, substantially outperforming all prior methods (previous best 54.2%), and achieves a new state-of-the-art result on the LIBERO (97.2%).
- • Dream-VLA consistently outperforms the strong AR-based baseline OpenVLA-OFT across diverse finetuning objectives, demonstrating the advantage of a diffusion-based backbone.
- • Diffusion-based backbone requires no architectural modifications compared to AR-based VLA models to enable action chunking, providing inherent advantages in structural consistency and faster training convergence.

### 4.5 Real-World Performance

PiPER Sim2Sim evaluation. We first benchmark Dream-VLA under the PiPER sim-to-sim tabletop pick setting and compare against a strong autoregressive baseline, OpenVLA-OFT. We finetune both models on domain-randomized simulation data with L1 regression using the same LoRA setting as mentioned above, for 60k steps. We use an action chunk size of 50 and execute the first 20 steps for each action chunk. Two evaluation protocols are considered: (i) a simplified setting where both training and evaluation are restricted to 10 objects; (ii) a full setting with 75 target objects. Each object is evaluated for 10 trials, where the target is randomly placed among several distractors. Table 10 reports success counts for a representative subset of objects, while overall success rates are calculated over 100 and 750 total trials, respectively. Overall, on this relatively simple pick-up setting, Dream-VLA slightly outperforms OpenVLA-OFT, suggesting that diffusion-based VLA modeling remains competitive under controlled sim2sim evaluation. In addition, Dream-VLA achieves a ∼1.31× training speedup over OpenVLA-OFT for the same number of steps, and the absolute wall-clock time savings grow with longer training schedules. In future work, we will extend the comparison to more diverse and challenging scenarios to better investigate the strengths of different models.

- Table 10: Comparison of OpenVLA-OFT and Dream-VLA under PiPER sim-to-sim. We report successes out of 10 trials for each object (for the 75-obj setting, only the first 10 objects are shown for readability). Overall is computed over all objects in the corresponding setting.

Setting Model Apple Avocado Banana Bear Melon Bowl Tea Brush Cabbage Camel Others Overall 10 objs

OpenVLA-OFT 10/10 9/10 10/10 10/10 10/10 5/10 9/10 10/10 10/10 10/10 – 93.00% Dream-VLA 10/10 10/10 10/10 10/10 10/10 5/10 10/10 9/10 10/10 10/10 – 94.00%

OpenVLA-OFT 6/10 6/10 9/10 10/10 9/10 5/10 8/10 10/10 9/10 10/10 · · · 81.87% Dream-VLA 10/10 10/10 9/10 10/10 10/10 3/10 9/10 10/10 10/10 10/10 · · · 83.73%

75 objs

PiPER Sim2Real evaluation. We further perform zero-shot sim-to-real transfer on the real PiPER robot picking task using a single front-facing Intel RealSense D455 camera, without any real-world fine-tuning. Since the camera pose is randomized during training, we do not require rigorous extrinsic calibration at test time; instead, we place the camera in front of the workspace and manually ensure that its view roughly matches the simulated viewpoint. Nevertheless, we observe that camera placement can still have a noticeable impact on real-world performance. Due to unavoidable variations in scene resets, lighting, and camera pose across trials, we do not aim for a strictly controlled quantitative comparison. Qualitatively, Dream-VLA can reliably pick up a wide range of objects under moderate appearance shifts, demonstrating strong generalization. Remaining failures mainly stem from residual sim-to-real gaps (e.g., viewpoint and appearance discrepancies), suggesting that better calibration and improved robustness to visual shifts would further benefit real-world deployment.

## 5 Conclusion and Future Work

In this paper, we release Dream-VL and Dream-VLA as preliminary attempts at dLLMs for visual planning. Dream-VL achieves competitive performance with current open-source state-of-the-art AR VLMs on general visual understanding tasks and significantly surpasses previous dVLMs. Although general capabilities still lag behind top closed-data AR models, Dream-VL demonstrates significant advantages for tasks requiring global planning. Followed with robotic pretraining, Dream-VLA stands as a pioneering pretrained VLA based on dLLMs, achieving strong performance and showcasing its considerable promise.

However, there is still much room for improvement. Firstly, we do not systematically study data in this work; both the VL and VLA training data largely follow prior setups. There is still substantial room on this side to further improve performance. Second, an interesting direction is mixture training that jointly covers high-level planning and low-level control, such as in RT-2 (Brohan et al., 2023). Since dLLMs already exhibit advantages over AR LLMs in high-level planning, it is promising to investigate VLA backbones based on dLLMs that jointly improve both high-level plan generation and low-level action prediction. Thirdly, we empirically observe that continuous action spaces often outperform discrete ones in many downstream SFT settings. This suggests that (i) continuous-space

robotic pretraining over discrete dVLMs or continuous dVLMs is a promising direction, and (ii) on top of a discrete dVLM, designing better discrete action representations such as FAST (Pertsch et al., 2025) could further close the gap for discrete VLAs. Finally, our current real-robot experiments are still preliminary. A natural next step is to collect larger and more diverse real-world datasets to systematically evaluate how Dream-VLA behaves in realistic environments, and how it performs compared with AR-VLM-based VLAs regarding generalization.

## Acknowledgments

We thank Zhili Liu, Lei Li, Yiheng Xu, and Mukai Li for the valuable discussion on VLM, and Shengliang Deng for the valuable discussion on VLA.

## References

Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. Advances in neural information processing systems, 35:23716–23736, 2022.

Anthropic. Claude 3.5 sonnet, 2024. URL https://www.anthropic.com/news/claude-3-5-sonnet. Accessed on February 11, 2024.

Marianne Arriola, Aaron Gokaslan, Justin T. Chiu, Zhihan Yang, Zhixuan Qi, Jiaqi Han, Subham Sekhar Sahoo, and Volodymyr Kuleshov. Block diffusion: Interpolating between autoregressive and diffusion language models, 2025. URL https://arxiv.org/abs/2503.09573.

Jacob Austin, Daniel D Johnson, Jonathan Ho, Daniel Tarlow, and Rianne Van Den Berg. Structured denoising diffusion models in discrete state-spaces. Advances in neural information processing systems, 34:17981–17993, 2021.

Gregor Bachmann and Vaishnavh Nagarajan. The pitfalls of next-token prediction, 2024. URL https://arxiv.org/abs/2403.06963.

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report. ArXiv, abs/2502.13923, 2025a. URL https://api.semanticscholar.org/CorpusID:276449796.

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025b.

Johan Bjorck, Fernando Castañeda, Nikita Cherniadev, Xingye Da, Runyu Ding, Linxi Fan, Yu Fang, Dieter Fox, Fengyuan Hu, Spencer Huang, et al. Gr00t n1: An open foundation model for generalist humanoid robots. arXiv preprint arXiv:2503.14734, 2025.

Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, et al. π0: A vision-language-action flow model for

- general robot control. arXiv preprint arXiv:2410.24164, 2024a.

Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, et al. pi_0: A vision-language-action flow model for

- general robot control. arXiv preprint arXiv:2410.24164, 2024b.

Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Joseph Dabis, Chelsea Finn, Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, Jasmine Hsu, et al. Rt-1: Robotics transformer for real-world control at scale. arXiv preprint arXiv:2212.06817, 2022.

Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromanski, Tianli Ding, Danny Driess, Avinava Dubey, Chelsea Finn, et al. Rt-2: Vision-language-action models transfer web knowledge to robotic control. arXiv preprint arXiv:2307.15818, 2023.

Andrew Campbell, Joe Benton, Valentin De Bortoli, Thomas Rainforth, George Deligiannidis, and Arnaud Doucet. A continuous time framework for discrete denoising models. Advances in Neural Information Processing Systems, 35:28266–28279, 2022.

Andrew Campbell, Valentin De Bortoli, Jiaxin Shi, and Arnaud Doucet. Self-speculative masked diffusions. arXiv preprint arXiv:2510.03929, 2025.

Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, et al. Are we on the right way for evaluating large vision-language models? ArXiv preprint, abs/2403.20330, 2024a. URL https://arxiv.org/abs/2403.20330.

Ting Chen, Ruixiang ZHANG, and Geoffrey Hinton. Analog bits: Generating discrete data using diffusion models with self-conditioning. In The Eleventh International Conference on Learning Representations, 2023.

Yi Chen, Yuying Ge, Yizhuo Li, Yixiao Ge, Mingyu Ding, Ying Shan, and Xihui Liu. Moto: Latent motion token as the bridging language for robot manipulation. In Proceedings of the IEEE/CVF international conference on computer vision, 2025.

Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, et al. Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling. arXiv preprint arXiv:2412.05271, 2024b.

Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, et al. How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites. arXiv preprint arXiv:2404.16821, 2024c.

Shuang Cheng, Yihan Bian, Dawei Liu, Linfeng Zhang, Qian Yao, Zhongbo Tian, Wenhai Wang, Qipeng Guo, Kai Chen, Biqing Qi, and Bowen Zhou. Sdar: A synergistic diffusion-autoregression paradigm for scalable sequence generation, 2025. URL https://arxiv.org/abs/2510.06303.

Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, and Shuran Song. Diffusion policy: Visuomotor policy learning via action diffusion. The International Journal of Robotics Research, page 02783649241273668, 2023.

Can Cui, Yupeng Zhou, Juntong Peng, Sung-Yeon Park, Zichong Yang, Prashanth Sankaranarayanan, Jiaru Zhang, Ruqi Zhang, and Ziran Wang. Vilad: A large vision language diffusion framework for end-to-end autonomous driving. arXiv preprint arXiv:2508.12603, 2025.

Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven Hoi. Instructblip: Towards general-purpose vision-language models with instruction tuning, 2023.

Wenliang Dai, Nayeon Lee, Boxin Wang, Zhuolin Yang, Zihan Liu, Jon Barker, Tuomas Rintamaki, Mohammad Shoeybi, Bryan Catanzaro, and Wei Ping. Nvlm: Open frontier-class multimodal llms,

2024. URL https://arxiv.org/abs/2409.11402. DeepMind. Gemini diffusion. 2025. URL https://deepmind.google/models/gemini-diffusion/.

Matt Deitke, Christopher Clark, Sangho Lee, Rohun Tripathi, Yue Yang, Jae Sung Park, Mohammadreza Salehi, Niklas Muennighoff, Kyle Lo, Luca Soldaini, et al. Molmo and pixmo: Open weights and open data for state-of-the-art multimodal models. ArXiv preprint, abs/2409.17146, 2024. URL https://arxiv.org/abs/2409.17146.

Xuanzhao Dong, Wenhui Zhu, Xiwen Chen, Zhipeng Wang, Peijie Qiu, Shao Tang, Xin Li, and Yalin Wang. Llada-medv: Exploring large language diffusion models for biomedical image understanding. arXiv preprint arXiv:2508.01617, 2025.

Danny Driess, Fei Xia, Mehdi SM Sajjadi, Corey Lynch, Aakanksha Chowdhery, Brian Ichter, Ayzaan Wahid, Jonathan Tompson, Quan Vuong, Tianhe Yu, et al. Palm-e: An embodied multimodal language model. arXiv preprint arXiv:2303.03378, 2023.

Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. ArXiv preprint, abs/2405.21075, 2024. URL https://arxiv.org/abs/2405.21075.

Itai Gat, Heli Ben-Hamu, Marton Havasi, Daniel Haziza, Jeremy Reizenstein, Gabriel Synnaeve, David Lopez-Paz, Brian Karrer, and Yaron Lipman. Set block decoding is a language model inference accelerator. arXiv preprint arXiv:2509.04185, 2025.

Dibya Ghosh, Homer Rich Walke, Karl Pertsch, Kevin Black, Oier Mees, Sudeep Dasari, Joey Hejna, Tobias Kreiman, Charles Xu, Jianlan Luo, et al. Octo: An open-source generalist robot policy. In Robotics: Science and Systems, 2024.

Shansan Gong, Mukai Li, Jiangtao Feng, Zhiyong Wu, and Lingpeng Kong. DiffuSeq: Sequence to sequence text generation with diffusion models. In International Conference on Learning Representations, ICLR, 2023.

Shansan Gong, Shivam Agarwal, Yizhe Zhang, Jiacheng Ye, Lin Zheng, Mukai Li, Chenxin An, Peilin Zhao, Wei Bi, Jiawei Han, Hao Peng, and Lingpeng Kong. Scaling diffusion language models via adaptation from autoregressive models. International Conference on Learning Representations, 2025a.

Shansan Gong, Ruixiang Zhang, Huangjie Zheng, Jiatao Gu, Navdeep Jaitly, Lingpeng Kong, and Yizhe Zhang. Diffucoder: Understanding and improving masked diffusion models for code generation. arXiv preprint arXiv:2506.20639, 2025b.

Ishaan Gulrajani and Tatsunori B Hashimoto. Likelihood-based diffusion language models. Advances in Neural Information Processing Systems, 36:16693–16715, 2023.

Jarvis Guo, Tuney Zheng, Yuelin Bai, Bo Li, Yubo Wang, King Zhu, Yizhi Li, Graham Neubig, Wenhu Chen, and Xiang Yue. Mammoth-vl: Eliciting multimodal reasoning with instruction tuning at scale,

2025. URL https://arxiv.org/abs/2412.05237. Marton Havasi, Brian Karrer, Itai Gat, and Ricky TQ Chen. Edit flows: Flow matching with edit operations. arXiv preprint arXiv:2506.09018, 2025.

Emiel Hoogeboom, Didrik Nielsen, Priyank Jaini, Patrick Forré, and Max Welling. Argmax flows and multinomial diffusion: Learning categorical distributions. Advances in neural information processing systems, 34:12454–12465, 2021.

Zhi Hou, Tianyi Zhang, Yuwen Xiong, Haonan Duan, Hengjun Pu, Ronglei Tong, Chengyang Zhao, Xizhou Zhu, Yu Qiao, Jifeng Dai, et al. Dita: Scaling diffusion transformer for generalist visionlanguage-action policy. arXiv preprint arXiv:2503.19757, 2025.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=nZeVKeeFYf9.

Inception Labs. Mercury: Ultra-fast language models based on diffusion. https://www.inceptionlabs.ai/introducing-mercury, 2025. Accessed: 2025-06-16.

Haoqiang Kang, Yizhe Zhang, Nikki Lijing Kuang, Nicklas Majamaki, Navdeep Jaitly, Yi-An Ma, and Lianhui Qin. Ladir: Latent diffusion enhances llms for text reasoning. arXiv preprint arXiv:2510.04573, 2025.

Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A diagram is worth a dozen images. In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11–14, 2016, Proceedings, Part IV 14, pages 235–251. Springer, 2016. URL https://link.springer.com/chapter/10.1007/978-3-319-46493-0_15.

Samar Khanna, Siddhant Kharbanda, Shufan Li, Harshit Varma, Eric Wang, Sawyer Birnbaum, Ziyang Luo, Yanis Miraoui, Akash Palrecha, Stefano Ermon, et al. Mercury: Ultra-fast language models based on diffusion. arXiv e-prints, pages arXiv–2506, 2025.

Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, et al. Openvla: An open-source vision-languageaction model. arXiv preprint arXiv:2406.09246, 2024.

Moo Jin Kim, Chelsea Finn, and Percy Liang. Fine-tuning vision-language-action models: Optimizing speed and success, 2025. URL https://arxiv.org/abs/2502.19645.

Bo Li, Kaichen Zhang, Hao Zhang, Dong Guo, Renrui Zhang, Feng Li, Yuanhan Zhang, Ziwei Liu, and Chunyuan Li. Llava-next: Stronger llms supercharge multimodal capabilities in the wild, 2024a. URL https://llava-vl.github.io/blog/2024-05-10-llava-next-stronger-llms/.

Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. Llava-onevision: Easy visual task transfer. ArXiv preprint, abs/2408.03326, 2024b. URL https://arxiv.org/abs/2408.03326.

Bohao Li, Rui Wang, Guangzhi Wang, Yuying Ge, Yixiao Ge, and Ying Shan. Seed-bench: Benchmarking multimodal llms with generative comprehension. ArXiv preprint, abs/2307.16125, 2023a. URL https://arxiv.org/abs/2307.16125.

Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pretraining with frozen image encoders and large language models. In International conference on machine learning, pages 19730–19742. PMLR, 2023b.

Pengxiang Li, Yinan Zheng, Yue Wang, Huimin Wang, Hang Zhao, Jingjing Liu, Xianyuan Zhan, Kun Zhan, and Xianpeng Lang. Discrete diffusion for reflective vision-language-action models in autonomous driving. arXiv preprint arXiv:2509.20109, 2025a.

Qixiu Li, Yaobo Liang, Zeyu Wang, Lin Luo, Xi Chen, Mozheng Liao, Fangyun Wei, Yu Deng, Sicheng Xu, Yizhong Zhang, et al. Cogact: A foundational vision-language-action model for synergizing cognition and action in robotic manipulation. arXiv preprint arXiv:2411.19650, 2024c.

Shufan Li, Jiuxiang Gu, Kangning Liu, Zhe Lin, Zijun Wei, Aditya Grover, and Jason Kuen. Lavida-o: Elastic large masked diffusion models for unified multimodal understanding and generation. arXiv preprint arXiv:2509.19244, 2025b.

Shufan Li, Konstantinos Kallidromitis, Hritik Bansal, Akash Gokul, Yusuke Kato, Kazuki Kozuka, Jason Kuen, Zhe Lin, Kai-Wei Chang, and Aditya Grover. Lavida: A large diffusion model for vision-language understanding. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025c.

Xiang Lisa Li, John Thickstun, Ishaan Gulrajani, Percy Liang, and Tatsunori B Hashimoto. Diffusion-lm improves controllable text generation. In Conference on Neural Information Processing Systems, NeurIPS, 2022.

Xinghang Li, Peiyan Li, Minghuan Liu, Dong Wang, Jirong Liu, Bingyi Kang, Xiao Ma, Tao Kong, Hanbo Zhang, and Huaping Liu. Towards generalist robot policies: What matters in building vision-language-action models. arXiv preprint arXiv:2412.14058, 2024d.

Xuanlin Li, Kyle Hsu, Jiayuan Gu, Karl Pertsch, Oier Mees, Homer Rich Walke, Chuyuan Fu, Ishikaa Lunawat, Isabel Sieh, Sean Kirmani, et al. Evaluating real-world robot manipulation policies in simulation. arXiv preprint arXiv:2405.05941, 2024e.

Zhixuan Liang, Yizhuo Li, Tianshuo Yang, Chengyue Wu, Sitong Mao, Tian Nian, Liuao Pei, Shunbo Zhou, Xiaokang Yang, Jiangmiao Pang, Yao Mu, and Ping Luo. Discrete diffusion vla: Bringing discrete diffusion to action decoding in vision-language-action policies, 2025. URL https://arxiv. org/abs/2508.20072.

Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow matching for generative modeling. In The Eleventh International Conference on Learning Representations, 2023.

Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, and Peter Stone. Libero: Benchmarking knowledge transfer for lifelong robot learning. Advances in Neural Information Processing Systems, 36:44776–44791, 2023a.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36:34892–34916, 2023b.

Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26296–26306, 2024a.

Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Improved reasoning, ocr, and world knowledge, 2024b.

Jingyu Liu, Xin Dong, Zhifan Ye, Rishabh Mehta, Yonggan Fu, Vartika Singh, Jan Kautz, Ce Zhang, and Pavlo Molchanov. Tidar: Think in diffusion, talk in autoregression. arXiv preprint arXiv:2511.08923, 2025a.

Sulin Liu, Juno Nam, Andrew Campbell, Hannes Stark, Yilun Xu, Tommi Jaakkola, and Rafael Gomez-Bombarelli. Think while you generate: Discrete diffusion with planned denoising. In The Thirteenth International Conference on Learning Representations, 2025b.

Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? In European Conference on Computer Vision, pages 216–233. Springer, 2025c. URL https://arxiv.org/abs/2307.06281.

Aaron Lou, Chenlin Meng, and Stefano Ermon. Discrete diffusion language modeling by estimating the ratios of the data distribution. In International Conference on Machine Learning, ICML, 2024.

Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. ArXiv preprint, abs/2310.02255, 2023. URL https: //arxiv.org/abs/2310.02255.

Yujie Lu, Dongfu Jiang, Wenhu Chen, William Yang Wang, Yejin Choi, and Bill Yuchen Lin. Wildvision: Evaluating vision-language models in the wild with human preferences. ArXiv preprint, abs/2406.11069, 2024. URL https://arxiv.org/abs/2406.11069.

Xinyin Ma, Runpeng Yu, Gongfan Fang, and Xinchao Wang. dkv-cache: The cache for diffusion language models. arXiv preprint arXiv:2505.15781, 2025a.

Yingzi Ma, Yulong Cao, Wenhao Ding, Shuibai Zhang, Yan Wang, Boris Ivanovic, Ming Jiang, Marco Pavone, and Chaowei Xiao. dvlm-ad: Enhance diffusion vision-language-model for driving via controllable reasoning. arXiv preprint arXiv:2512.04459, 2025b.

Ahmed Masry, Xuan Long Do, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. ChartQA: A benchmark for question answering about charts with visual and logical reasoning. In Findings of the Association for Computational Linguistics: ACL 2022, pages 2263–2279, Dublin, Ireland, 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.findings-acl.177. URL https://aclanthology.

org/2022.findings-acl.177.

Minesh Mathew, Dimosthenis Karatzas, R Manmatha, and CV Jawahar. Docvqa: A dataset for vqa on document images. corr abs/2007.00398 (2020). ArXiv preprint, abs/2007.00398, 2020. URL https://arxiv.org/abs/2007.00398.

Minesh Mathew, Viraj Bagal, Rubèn Pérez Tito, Dimosthenis Karatzas, Ernest Valveny, and C. V Jawahar. Infographicvqa. ArXiv preprint, abs/2104.12756, 2021. URL https://arxiv.org/abs/ 2104.12756.

Matteo Merler, Nicola Dainese, Minttu Alakuijala, Giovanni Bonetta, Pietro Ferrazzi, Yu Tian, Bernardo Magnini, and Pekka Marttinen. Viplan: A benchmark for visual planning with symbolic predicates and vision-language models, 2025. URL https://arxiv.org/abs/2505.13180.

Jinjie Ni, Qian Liu, Longxu Dou, Chao Du, Zili Wang, Hang Yan, Tianyu Pang, and Michael Qizhe Shieh. Diffusion language models are super data learners. arXiv preprint arXiv:2511.03276, 2025.

Shen Nie, Fengqi Zhu, Zebin You, Xiaolu Zhang, Jingyang Ou, Jun Hu, Jun Zhou, Yankai Lin, Ji-Rong

Wen, and Chongxuan Li. Large language diffusion models. arXiv preprint arXiv:2502.09992, 2025. OpenAI. Gpt-4o system card, 2024. URL https://arxiv.org/abs/2410.21276. Jingyang Ou, Shen Nie, Kaiwen Xue, Fengqi Zhu, Jiacheng Sun, Zhenguo Li, and Chongxuan Li. Your

absorbing discrete diffusion secretly models the conditional distributions of clean data. International Conference on Learning Representations, 2025.

Abby O’Neill, Abdul Rehman, Abhiram Maddukuri, Abhishek Gupta, Abhishek Padalkar, Abraham Lee, Acorn Pooley, Agrim Gupta, Ajay Mandlekar, Ajinkya Jain, et al. Open x-embodiment: Robotic learning datasets and rt-x models: Open x-embodiment collaboration 0. In 2024 IEEE International

- Conference on Robotics and Automation (ICRA), pages 6892–6903. IEEE, 2024a.

Abby O’Neill, Abdul Rehman, Abhiram Maddukuri, Abhishek Gupta, Abhishek Padalkar, Abraham Lee, Acorn Pooley, Agrim Gupta, Ajay Mandlekar, Ajinkya Jain, et al. Open x-embodiment: Robotic learning datasets and rt-x models: Open x-embodiment collaboration 0. In 2024 IEEE International

- Conference on Robotics and Automation (ICRA), pages 6892–6903. IEEE, 2024b.

Tim Pearce, Tabish Rashid, Anssi Kanervisto, David Bignell, Mingfei Sun, Raluca Georgescu, Sergio Valcarcel Macua, Shan Zheng Tan, Ida Momennejad, Katja Hofmann, et al. Imitating human behaviour with diffusion models. In Deep Reinforcement Learning Workshop NeurIPS 2022.

Karl Pertsch, Kyle Stachowicz, Brian Ichter, Danny Driess, Suraj Nair, Quan Vuong, Oier Mees, Chelsea Finn, and Sergey Levine. Fast: Efficient action tokenization for vision-language-action models, 2025. URL https://arxiv.org/abs/2501.09747.

Patrick Pynadath, Jiaxin Shi, and Ruqi Zhang. Candi: Hybrid discrete-continuous diffusion models. arXiv preprint arXiv:2510.22510, 2025.

Delin Qu, Haoming Song, Qizhi Chen, Yuanqi Yao, Xinyi Ye, Yan Ding, Zhigang Wang, JiaYuan Gu, Bin Zhao, Dong Wang, et al. Spatialvla: Exploring spatial representations for visual-language-action model. arXiv preprint arXiv:2501.15830, 2025.

Machel Reid, Nikolay Savinov, Denis Teplyashin, Dmitry Lepikhin, Timothy P. Lillicrap, Jean-Baptiste Alayrac, Radu Soricut, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. ArXiv, abs/2403.05530, 2024. URL https://api.semanticscholar.org/ CorpusID:268297180.

Moritz Reuss, Maximilian Li, Xiaogang Jia, and Rudolf Lioutikov. Goal conditioned imitation learning using score-based diffusion policies. In Robotics: Science and Systems, 2023.

Moritz Reuss, Ömer Erdinç Yağmurlu, Fabian Wenzel, and Rudolf Lioutikov. Multimodal diffusion transformer: Learning versatile behavior from multimodal goals. In First Workshop on VisionLanguage Models for Navigation and Manipulation at ICRA 2024, 2024.

Dhruv Rohatgi, Adam Block, Audrey Huang, Akshay Krishnamurthy, and Dylan J. Foster. Computational-statistical tradeoffs at the next-token prediction barrier: Autoregressive and imitation learning under misspecification, 2025. URL https://arxiv.org/abs/2502.12465.

Subham Sekhar Sahoo, Marianne Arriola, Aaron Gokaslan, Edgar Mariano Marroquin, Alexander M Rush, Yair Schiff, Justin T Chiu, and Volodymyr Kuleshov. Simple and effective masked diffusion language models. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. URL https://openreview.net/forum?id=L4uaAR4ArM.

Jiaxin Shi, Kehang Han, Zhe Wang, Arnaud Doucet, and Michalis K. Titsias. Simplified and generalized masked diffusion for discrete data. Neural Information Processing Systems, 2024.

Mustafa Shukor, Dana Aubakirova, Francesco Capuano, Pepijn Kooijmans, Steven Palma, Adil Zouitine, Michel Aractingi, Caroline Pascal, Martino Russi, Andres Marafioti, et al. Smolvla: A vision-languageaction model for affordable and efficient robotics. arXiv preprint arXiv:2506.01844, 2025.

Gemini Robotics Team, Saminda Abeyruwan, Joshua Ainslie, Jean-Baptiste Alayrac, Montserrat Gonzalez Arenas, Travis Armstrong, Ashwin Balakrishna, Robert Baruch, Maria Bauza, Michiel Blokzijl, et al. Gemini robotics: Bringing ai into the physical world. arXiv preprint arXiv:2503.20020, 2025.

Yang Tian, Sizhe Yang, Jia Zeng, Ping Wang, Dahua Lin, Hao Dong, and Jiangmiao Pang. Predictive inverse dynamics models are scalable learners for robotic manipulation. In The Thirteenth International Conference on Learning Representations, 2025.

Shengbang Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Manoj Middepogu, Sai Charitha Akula, Jihan Yang, Shusheng Yang, Adithya Iyer, Xichen Pan, et al. Cambrian-1: A fully open, vision-centric exploration of multimodal llms. ArXiv preprint, abs/2406.16860, 2024. URL https://arxiv.org/ abs/2406.16860.

Tao Tu, Shekoofeh Azizi, Danny Driess, Mike Schaekermann, Mohamed Amin, Pi-Chuan Chang, Andrew Carroll, Charles Lau, Ryutaro Tanno, Ira Ktena, et al. Towards generalist biomedical ai. Nejm Ai, 1(3):AIoa2300138, 2024.

Homer Rich Walke, Kevin Black, Tony Z Zhao, Quan Vuong, Chongyi Zheng, Philippe Hansen-Estruch, Andre Wang He, Vivek Myers, Moo Jin Kim, Max Du, et al. Bridgedata v2: A dataset for robot learning at scale. In Conference on Robot Learning, pages 1723–1736. PMLR, 2023.

Fei Wang, Xingyu Fu, James Y. Huang, Zekun Li, Qin Liu, Xiaogeng Liu, Mingyu Derek Ma, Nan Xu, Wenxuan Zhou, Kai Zhang, Tianyi Lorena Yan, Wenjie Jacky Mo, Hsiang-Hui Liu, Pan Lu, Chunyuan Li, Chaowei Xiao, Kai-Wei Chang, Dan Roth, Sheng Zhang, Hoifung Poon, and Muhao Chen. Muirbench: A comprehensive benchmark for robust multi-image understanding. ArXiv preprint, abs/2406.09411, 2024a. URL https://arxiv.org/abs/2406.09411.

Jin Wang, Yao Lai, Aoxue Li, Shifeng Zhang, Jiacheng Sun, Ning Kang, Chengyue Wu, Zhenguo Li, and Ping Luo. Fudoki: Discrete flow-based unified understanding and generation via kinetic-optimal velocities. ArXiv, abs/2505.20147, 2025a. URL https://api.semanticscholar.org/CorpusID: 278911369.

Lirui Wang, Xinlei Chen, Jialiang Zhao, and Kaiming He. Scaling proprioceptive-visual learning with heterogeneous pre-trained transformers. Advances in neural information processing systems, 37: 124420–124450, 2024b.

Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024c.

Xu Wang, Chenkai Xu, Yijie Jin, Jiachun Jin, Hao Zhang, and Zhijie Deng. Diffusion llms can do fasterthan-ar inference via discrete diffusion forcing, 2025b. URL https://arxiv.org/abs/2508.09192.

Junjie Wen, Yichen Zhu, Jinming Li, Minjie Zhu, Zhibin Tang, Kun Wu, Zhiyuan Xu, Ning Liu, Ran Cheng, Chaomin Shen, et al. Tinyvla: Towards fast, data-efficient vision-language-action models for robotic manipulation. IEEE Robotics and Automation Letters, 2025a.

Yuqing Wen, Hebei Li, Kefan Gu, Yucheng Zhao, Tiancai Wang, and Xiaoyan Sun. Llada-vla: Vision language diffusion action models, 2025b. URL https://arxiv.org/abs/2509.06932.

Gwen Yidou Weng, Benjie Wang, and Guy Van den Broeck. Trace back from the future: A probabilistic reasoning approach to controllable language generation, 2025. URL https://arxiv.org/abs/2504. 18535.

Chengyue Wu, Hao Zhang, Shuchen Xue, Shizhe Diao, Yonggan Fu, Zhijian Liu, Pavlo Molchanov, Ping Luo, Song Han, and Enze Xie. Fast-dllm v2: Efficient block-diffusion llm, 2025a. URL https://arxiv.org/abs/2509.26328.

Chengyue Wu, Hao Zhang, Shuchen Xue, Zhijian Liu, Shizhe Diao, Ligeng Zhu, Ping Luo, Song Han, and Enze Xie. Fast-dllm: Training-free acceleration of diffusion llm by enabling kv cache and parallel decoding. arXiv preprint arXiv:2505.22618, 2025b.

Zhiyu Wu, Xiaokang Chen, Zizheng Pan, Xingchao Liu, Wen Liu, Damai Dai, Huazuo Gao, Yiyang Ma, Chengyue Wu, Bing-Li Wang, Zhenda Xie, Yu Wu, Kai Hu, Jiawei Wang, Yaofeng Sun, Yukun Li, Yishi Piao, Kang Guan, Aixin Liu, Xin Xie, Yu mei You, Kaihong Dong, Xingkai Yu, Haowei Zhang, Liang Zhao, Yisong Wang, and Chong Ruan. Deepseek-vl2: Mixture-of-experts visionlanguage models for advanced multimodal understanding. ArXiv, abs/2412.10302, 2024. URL https://api.semanticscholar.org/CorpusID:274762981.

Zirui Wu, Lin Zheng, Zhihui Xie, Jiacheng Ye, Jiahui Gao, Yansong Feng, Zhenguo Li, Victoria W., Guorui Zhou, and Lingpeng Kong. Dreamon: Diffusion language models for code infilling beyond fixed-size canvas, 2025c. URL https://hkunlp.github.io/blog/2025/dreamon.

x.ai. Grok 1.5v: A New Era in AI Understanding, 2024. URL https://x.ai/blog/grok-1.5v. Accessed: 2024-11-07.

Zhihui Xie, Jiacheng Ye, Lin Zheng, Jiahui Gao, Jingwei Dong, Zirui Wu, Xueliang Zhao, Shansan Gong, Xin Jiang, Zhenguo Li, et al. Dream-coder 7b: An open diffusion language model for code. arXiv preprint arXiv:2509.01142, 2025.

Guowei Xu, Peng Jin, Hao Li, Yibing Song, Lichao Sun, and Li Yuan. Llava-cot: Let vision language models reason step-by-step. ArXiv preprint, abs/2411.10440, 2024. URL https://arxiv.org/abs/ 2411.10440.

Mingwang Xu, Jiahao Cui, Feipeng Cai, Hanlin Shang, Zhihao Zhu, Shan Luan, Yifang Xu, Neng Zhang, Yaoyi Li, Jia Cai, et al. Wam-diff: A masked diffusion vla framework with moe and online reinforcement learning for autonomous driving. arXiv preprint arXiv:2512.11872, 2025a.

Minkai Xu, Tomas Geffner, Karsten Kreis, Weili Nie, Yilun Xu, Jure Leskovec, Stefano Ermon, and Arash Vahdat. Energy-based diffusion language models for text generation. In The Thirteenth International Conference on Learning Representations, 2025b.

Chenxiao Yang, Cai Zhou, David Wipf, and Zhiyuan Li. On powerful ways to generate: Autoregression, diffusion, and beyond. arXiv preprint arXiv:2510.06190, 2025a.

Ling Yang, Ye Tian, Bowen Li, Xinchen Zhang, Ke Shen, Yunhai Tong, and Mengdi Wang. Mmada: Multimodal large diffusion language models. arXiv preprint arXiv:2505.15809, 2025b.

Jiacheng Ye, Shansan Gong, Liheng Chen, Lin Zheng, Jiahui Gao, Han Shi, Chuan Wu, Xin Jiang, Zhenguo Li, Wei Bi, et al. Diffusion of thought: Chain-of-thought reasoning in diffusion language models. Advances in Neural Information Processing Systems, 37:105345–105374, 2024.

Jiacheng Ye, Jiahui Gao, Shansan Gong, Lin Zheng, Xin Jiang, Zhenguo Li, and Lingpeng Kong. Beyond autoregression: Discrete diffusion for complex reasoning and planning. International Conference on Learning Representations, 2025a.

Jiacheng Ye, Zhenyu Wu, Jiahui Gao, Zhiyong Wu, Xin Jiang, Zhenguo Li, and Lingpeng Kong. Implicit search via discrete diffusion: A study on chess. In The Thirteenth International Conference on Learning Representations, 2025b.

Jiacheng Ye, Zhihui Xie, Lin Zheng, Jiahui Gao, Zirui Wu, Xin Jiang, Zhenguo Li, and Lingpeng Kong. Dream 7b: Diffusion large language models, 2025c. URL https://arxiv.org/abs/2508.15487.

Zebin You, Shen Nie, Xiaolu Zhang, Jun Hu, Jun Zhou, Zhiwu Lu, Ji-Rong Wen, and Chongxuan Li. Llada-v: Large language diffusion models with visual instruction tuning. arXiv preprint arXiv:2505.16933, 2025.

Runpeng Yu, Xinyin Ma, and Xinchao Wang. Dimple: Discrete diffusion multimodal large language model with parallel decoding. arXiv preprint arXiv:2505.16990, 2025.

Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. Mm-vet: Evaluating large multimodal models for integrated capabilities. ArXiv preprint, abs/2308.02490, 2023. URL https://arxiv.org/abs/2308.02490.

Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, Cong Wei, Botao Yu, Ruibin Yuan, Renliang Sun, Ming Yin, Boyuan Zheng, Zhenzhu Yang, Yibo Liu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of CVPR, 2024a. URL https://arxiv.org/abs/2311.16502.

Xiang Yue, Tianyu Zheng, Yuansheng Ni, Yubo Wang, Kai Zhang, Shengbang Tong, Yuxuan Sun, Botao Yu, Ge Zhang, Huan Sun, Yu Su, Wenhu Chen, and Graham Neubig. Mmmu-pro: A more robust multi-discipline multimodal understanding benchmark. ArXiv preprint, abs/2409.02813, 2024b. URL https://arxiv.org/abs/2409.02813.

Xiaomi LLM-Core Team Zihao Yue, Zhenrui Lin, Yi-Hao Song, Weikun Wang, Shu-Qin Ren, Shuhao Gu, Shicheng Li, Peidian Li, Liang Zhao, Lei Li, Kainan Bao, Hao Tian, Hailin Zhang, Gang Wang, Dawei Zhu, Cici, Chenhong He, Bowen Ye, Bowen Shen, Zihan Zhang, Zi-Ang Jiang, Zhixian Zheng, Zhichao Song, Zhen Luo, Yue Yu, Yudong Wang, Yu Tian, Yu Tu, Yihan Yan, Yi Huang, Xu Wang, Xin dan Xu, Xin Ran Song, Xing Zhang, Xing Yong, Xin Zhang, Xia Deng, Wenyu Yang, Wenhan Ma, Weiwei Lv, Weiji Zhuang, Wei Liu, Sirui Deng, Shuo Liu, Shimao Chen, Shi liang Yu, Shao yang Liu, Shan yong Wang, Rui Ma, Qiantong Wang, Peng Wang, Nuo Chen, Menghang Zhu, Kang Zhou, Kang Zhou, Kai Fang, Jun-Miao Shi, Jinhao Dong, Jiebao Xiao, Jiaming Xu, Huaqiu Liu, Hongsheng Xu, Hengxu Qu, Hao-Song Zhao, Hanglong Lv, Guoan Wang, Duo Zhang, Dong Zhang, Di Zhang, Chong-Yi Ma, Chang Liu, Can Cai, and Bing Xia. Mimo-vl technical report. ArXiv, abs/2506.03569, 2025. URL https://api.semanticscholar.org/CorpusID:279155294.

Jingyao Zhang, Tianlin Li, Xiaoyu Zhang, Qiang Hu, and Bin Shi. Exploring the power of diffusion large language models for software engineering: An empirical investigation, 2025a. URL https: //arxiv.org/abs/2510.04605.

Kai Zhang, Rong Zhou, Eashan Adhikarla, Zhiling Yan, Yixin Liu, Jun Yu, Zhengliang Liu, Xun Chen, Brian D Davison, Hui Ren, et al. A generalist vision–language foundation model for diverse biomedical tasks. Nature Medicine, 30(11):3129–3141, 2024a.

Muru Zhang, Ofir Press, William Merrill, Alisa Liu, and Noah A Smith. How language model hallucinations can snowball. arXiv preprint arXiv:2305.13534, 2023a.

Renrui Zhang, Dongzhi Jiang, Yichi Zhang, Haokun Lin, Ziyu Guo, Pengshuo Qiu, Aojun Zhou, Pan Lu, Kai-Wei Chang, Peng Gao, and Hongsheng Li. Mathverse: Does your multi-modal llm truly see the diagrams in visual math problems? ArXiv preprint, abs/2403.14624, 2024b. URL https://arxiv.org/abs/2403.14624.

Ruixiang Zhang, Shuangfei Zhai, Yizhe Zhang, James Thornton, Zijing Ou, Joshua M Susskind, and Navdeep Jaitly. Target concrete score matching: A holistic framework for discrete diffusion. In Forty-second International Conference on Machine Learning, 2025b.

Wenyao Zhang, Hongsi Liu, Zekun Qi, Yunan Wang, Xinqiang Yu, Jiazhao Zhang, Runpei Dong, Jiawei He, He Wang, Zhizheng Zhang, Li Yi, Wenjun Zeng, and Xin Jin. Dreamvla: A vision-languageaction model dreamed with comprehensive world knowledge. CoRR, abs/2507.04447, 2025c. doi: 10.48550/ARXIV.2507.04447. URL https://doi.org/10.48550/arXiv.2507.04447.

Yizhe Zhang, Jiatao Gu, Zhuofeng Wu, Shuangfei Zhai, Joshua Susskind, and Navdeep Jaitly. Planner: Generating diversified paragraph via latent language diffusion model. Advances in Neural Information Processing Systems, 36:80178–80190, 2023b.

Yu Zhang, Xiusi Chen, Bowen Jin, Sheng Wang, Shuiwang Ji, Wei Wang, and Jiawei Han. A comprehensive survey of scientific large language models and their applications in scientific discovery. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 8783–8817, 2024c.

Qingqing Zhao, Yao Lu, Moo Jin Kim, Zipeng Fu, Zhuoyang Zhang, Yecheng Wu, Zhaoshuo Li, Qianli Ma, Song Han, Chelsea Finn, et al. Cot-vla: Visual chain-of-thought reasoning for vision-languageaction models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 1702–1713, 2025.

Tony Z Zhao, Vikash Kumar, Sergey Levine, and Chelsea Finn. Learning fine-grained bimanual manipulation with low-cost hardware. 2023.

Huangjie Zheng, Shansan Gong, Ruixiang Zhang, Tianrong Chen, Jiatao Gu, Mingyuan Zhou, Navdeep Jaitly, and Yizhe Zhang. Continuously augmented discrete diffusion model for categorical generative modeling. arXiv preprint arXiv:2510.01329, 2025a.

Lin Zheng, Jianbo Yuan, Lei Yu, and Lingpeng Kong. A reparameterized discrete diffusion model for text generation. arXiv preprint arXiv:2302.05737, 2023.

Ruijie Zheng, Yongyuan Liang, Shuaiyi Huang, Jianfeng Gao, Hal Daumé III, Andrey Kolobov, Furong Huang, and Jianwei Yang. Tracevla: Visual trace prompting enhances spatial-temporal awareness for generalist robotic policies. In The Thirteenth International Conference on Learning Representations, 2025b.

Junjie Zhou, Yan Shu, Bo Zhao, Boya Wu, Shitao Xiao, Xi Yang, Yongping Xiong, Bo Zhang, Tiejun Huang, and Zheng Liu. Mlvu: A comprehensive benchmark for multi-task long video understanding. ArXiv preprint, abs/2406.04264, 2024a. URL https://arxiv.org/abs/2406.04264.

Xingcheng Zhou, Mingyu Liu, Ekim Yurtsever, Bare Luka Zagar, Walter Zimmer, Hu Cao, and Alois C Knoll. Vision language models in autonomous driving: A survey and outlook. IEEE Transactions on Intelligent Vehicles, 2024b.

Fengqi Zhu, Rongzhen Wang, Shen Nie, Xiaolu Zhang, Chunwei Wu, Jun Hu, Jun Zhou, Jianfei Chen, Yankai Lin, Ji-Rong Wen, et al. Llada 1.5: Variance-reduced preference optimization for large language diffusion models. arXiv preprint arXiv:2505.19223, 2025a.

Fengqi Zhu, Zebin You, Yipeng Xing, Zenan Huang, Lin Liu, Yihong Zhuang, Guoshan Lu, Kangyu Wang, Xudong Wang, Lanning Wei, et al. Llada-moe: A sparse moe diffusion language model. arXiv preprint arXiv:2509.24389, 2025b.

Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Yuchen Duan, Hao Tian, Weijie Su, Jie Shao, Zhangwei Gao, Erfei Cui, Yue Cao, Yangzhou Liu, Haomin Wang, Weiye Xu, Hao Li, Jiahao Wang, Han Lv, Dengnian Chen, Songze Li, Yinan He, Tan Jiang, Jiapeng Luo, Yi Wang, Cong He, Botian Shi, Xingcheng Zhang, Wenqi Shao, Junjun He, Ying Xiong, Wenwen Qu, Peng Sun, Penglong Jiao, Lijun Wu, Kai Zhang, Hui Deng, Jiaye Ge, Kaiming Chen, Limin Wang, Min Dou, Lewei Lu, Xizhou Zhu, Tong Lu, Dahua Lin, Yu Qiao, Jifeng Dai, and Wenhai Wang. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. ArXiv, abs/2504.10479, 2025c. URL https://api.semanticscholar.org/CorpusID:277780955.

