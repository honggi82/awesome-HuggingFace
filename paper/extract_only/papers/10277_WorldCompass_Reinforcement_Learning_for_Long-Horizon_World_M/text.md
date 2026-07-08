# arXiv:2602.09022v1[cs.CV]9Feb2026

## WorldCompass: Reinforcement Learning for Long-Horizon World Models

Zehan Wang1 Tengfei Wang2 Haiyu Zhang2 Xuhui Zuo2 Junta Wu2 Haoyuan Wang2 Wenqiang Sun2 Zhenwei Wang2 Chenjie Cao2 Hengshuang Zhao3 Chunchao Guo2 Zhou Zhao1

### Abstract

This work presents WorldCompass, a novel Reinforcement Learning (RL) post-training framework for the long-horizon, interactive video-based world models, enabling them to explore the world more accurately and consistently based on interaction signals. To effectively “steer” the world model’s exploration, we introduce three core innovations tailored to the autoregressive video generation paradigm: 1) Clip-level rollout Strategy: We generate and evaluate multiple samples at a single target clip, which significantly boosts rollout efficiency and provides fine-grained reward signals. 2) Complementary Reward Functions: We design reward functions for both interactionfollowing accuracy and visual quality, which provide direct supervision and effectively suppress reward-hacking behaviors. 3) Efficient RL Algorithm: We employ the negative-aware finetuning strategy coupled with various efficiency optimizations to efficiently and effectively enhance model capacity. Evaluations on the SoTA open-source world model, WorldPlay, demonstrate that WorldCompass significantly improves interaction accuracy and visual fidelity across various scenarios. Project page and online demo can be found: https://3d-models.hunyuan.

tencent.com/world/

### 1. Introduction

World models enable humans or agents to interact with generative world environments. The emergence of Cosmos (Agarwal et al., 2025; Ali et al., 2025; Azzolini et al., 2025), HY-World (HunyuanWorld, 2025; Huang et al., 2025; Sun et al., 2025) and Genie series (Bruce et al., 2024; ParkerHolder et al., 2024; Ball et al., 2025), have significantly advanced the field, demonstrating the profound impact of video-based world models on both fundamental AI research and practical generative media applications (Liao et al.,

1Zhejiang University 2Tencent Hunyuan 3The University of Hong Kong. Correspondence to: Chunchao Guo <chunchaoguo@gmail.com>, Zhou Zhao <zhaozhou@zju.edu.cn>.

Preprint. February 10, 2026.

2025; Liu et al., 2025c; Ding et al., 2025; Li et al., 2025c; Liu et al., 2025d; Team et al., 2025a; Li et al., 2025b).

However, despite this immense potential, current opensource video-based world model solutions remain predominantly confined to the pre-training stage. These methods (Tang et al., 2025; He et al., 2025c; Sun et al., 2025; Yu et al., 2025) typically rely on pixel supervision from raw visual data to implicitly learn how to follow input actions.

In this work, we focus on post-training for long-horizon video-based world models. We attempt to employ reinforcement learning (RL) to more directly teach the model to explore the world with better accuracy and consistency, grounded in interaction signals. To achieve this, we propose WorldCompass, a novel RL framework specifically designed to “steer” the world exploration.

Specifically, we redesign each stages of RL process, grounding them in the autoregressive, interactive and long-horizon generation paradigm of world model. 1) We introduce cliplevel rollout for autoregressive video generation. This strategy significantly boosts both rollout efficiency and the granularity of the reward signal. Besides, it compels the model to rely on its own imperfect predictions, thus effectively mitigating the challenge of exposure bias. 2) We design two complementary reward functions tailored to the main characteristics of world modeling: action following and visual quality of the generated content. This complementary reward feedback effectively suppresses reward hacking. 3) We utilize the negative-aware fine-tuning strategy for RL training, supplemented by a suite of multi-faceted efficiency optimizations. Together, these methods drive the autoregressive video model toward the desired direction, ensuring both robust learning and computational feasibility.

We evaluate the WorldCompass framework by performing post-training on WorldPlay (Sun et al., 2025), a recent state-of-the-art open-source world model. Through comprehensive evaluation, we find that our RL training substantially improves the model’s interaction accuracy and visual quality across various scenarios: varying durations (from short-term to long-term), and different action complexities (basic actions or composite actions). This comprehensive improvement demonstrates that WorldCompass possesses high generalizability and effectively strengthens the model’s fundamental capabilities.

Our contribution can be summarized as follows:

- • We highlight the value of post-training for advanced world models and introduce WorldCompass, a novel RL framework specifically designed for video-based world models.
- • We dive into the autoregressive, interactive and longhorizon characteristics of the world model, redesigning the RL framework to achieve efficient training and fine-grained feedback signals.
- • We provide a comprehensive evaluation on WorldPlay, demonstrating that RL post-training significantly enhances the capabilities of this state-of-the-art opensource world model across various scenarios.

### 2. Related Work

#### 2.1. Video-based World Model

World model aims to predict future states, adhering to physical and geometric laws based on current and past observations and actions. It enables users or agents to interact with any environment. The recent Genie series (Bruce et al., 2024; Parker-Holder et al., 2024; Ball et al., 2025) demonstrate the significant potential of video-based world models in embodied intelligence and content creation. This approach utilizes a video generation model (Google, 2025; Wu et al., 2025; Wan et al., 2025) to interactively generate videos and explore the world by subjecting the generation process to a discrete action signal. This intrinsic need for interactive exploration requires the generation process to be autoregressive and long-horizon, and simultaneously demands precise fidelity to diverse action control conditions. Recent work revolves around these requirements. Diffusion Forcing (Chen et al., 2024) enables the autoregressive generation of long video clips by using variable timesteps during training. Another line of works (Wang et al., 2024; He et al., 2024; Valevski et al., 2024) embed discrete or continuous control signals into the video generation model to govern camera movements within the generated video. More recent efforts (Yu et al., 2025; Li et al., 2025a; Sun et al., 2025) integrate both aspects, allowing models to autoregressively generate video clips following action conditions and finally compose long-horizon sequences.

However, these methods primarily focus on the pre-training stage, where models implicitly learn to follow actions through pixel supervision from videos. This approach limits the existing methods’ ability over action switching or complex composite actions. In contrast, our approach introduces a post-training phase for the world model. By providing direct supervision for both action fidelity and visual quality, we significantly enhance the model’s capabilities.

#### 2.2. Reinforcement Learning

RL for Autoregressive LLM The recent success of DeepSeek-R1 (Guo et al., 2025) demonstrates that largescale on-policy Reinforcement Learning (RL), when coupled with a reliable reward function, can guide autoregressive LLM towards emergent capabilities growth. The GRPO (Shao et al., 2024) algorithm utilized in DeepSeekR1 has attracted significant attention. By leveraging the mean and variance of policy groups, GRPO removes the need for a separate value network (Schulman et al., 2017), resulting in a more memory-efficient approach. The effectiveness of on-policy RL has been wildly validated in large-scale experiments of LLMs (Yang et al., 2025; Liu et al., 2025a; Zheng et al., 2025a).

RL for Diffusion Model Inspired by the success of RL in LLMs, recent research has explored adapting RL algorithms for the post-training of diffusion models. While DiffusionDPO (Wallace et al., 2024) achieves alignment using off-policy preference pairs, more recent works like Flow-GRPO (Liu et al., 2025b) and Dance-GRPO (Xue et al., 2025) have successfully integrated the GRPO algorithm with diffusion models. By utilizing SDE solvers (Song et al., 2020) to enable on-policy RL, these methods have demonstrated significant performance gains. Furthermore, DiffusionNFT (Zheng et al., 2025b) builds upon the concept of group-wise advantage estimation, combining it with negative-aware fine-tuning to provide a more computationally efficient and effective refinement process.

However, current RL framework for diffusion models primarily target the paradigm where the entire content sequence (e.g., image, video, or audio) is generated in parallel within a single diffusion process. In contrast, world models necessitate sequential generation in an autoregressive manner, and often involve very long-horizon sequences. This fundamental architectural shift prevents the direct application of existing RL pipelines to our task. To bridge this gap, we propose a novel RL framework specifically tailored for the unique requirements of video-based world models.

### 3. WorldCompass 3.1. Preliminaries

Interactive World Modeling The core concept of world model is a generative system that enables a human or agent to interact with a world environment. Following the definition of Genie 2 (Parker-Holder et al., 2024), we frame this problem as autoregressive streaming video generation. This process is realized through a video diffusion model, denoted as πθ(xn|x1:n−1,an,c), which generates the future state (i.e., the next video clip) xn conditioned on the history of world observations x1:n−1, the user’s interaction actions an, and a textual or visual world prompt c.

- Sec 3.2 Clip-Level Rollout for Autoregressive Generation
- Sec 3.3 Reward Functions for Interactive Generation

Sec 3.4 Efficient RL Optimization

|W<br><br>D W<br><br>Action<br><br>Sequence …| |
|---|---|
| | |

Auto-Regressive Video Diffusion (𝜋 )

|EMA Update|
|---|

Shared Prefix from Clip 1 to Clip 𝑛-1 (𝑥 …𝑥 )

| | | |
|---|---|---|
|[Figure 1]<br><br>[Figure 2]<br><br>[Figure 3]<br><br>[Figure 4]<br><br>[Figure 5]<br><br>[Figure 6]|……|[Figure 7]<br><br>[Figure 8]|

|𝑣| |
|---|---|
| | |

− 𝑣

𝜋

Noise

Rollout Candidates at Target Clip 𝑛 (𝑥 … 𝑥 )

1−r

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

|……|
|---|

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

𝑚𝑖𝑛

r

| | |
|---|---|
|𝑣| |

| | | | | |
|---|---|---|---|---|
| | | | | |
| |Reward Fu|nctions (Action|Following & Visual Quality)| |

− 𝑣

𝜋

……

|Optimality Reward|
|---|

|−0.7|
|---|

|+0.4|
|---|

|+0.8|
|---|

|−0.3|
|---|

Figure 1. Overview of WorldCompass. 1) Starting from environmental prompts and action sequences, we generate shared prefix video clips. At the n-th target clip, we perform clip-level rollouts to generate a set of candidate video clips. 2) We design reliable reward functions to evaluate the action-following accuracy and visual quality of rollout samples. 3) We employ efficient RL algorithm to optimize the model, steering it toward generating high-scoring video clips.

Reinforcement Learning for Diffusion Models The fundamental loop of on-policy RL comprises three core stages: 1) Rollout: a group of G rollouts is generated from a pretrained diffusion model; 2) Evaluation: each sample is scored according to a set of predefined reward functions; and

However, we find that this intuitive sequence-level rollout scheme is fundamentally unsuitable for autoregressive video generation. The core issue is that the feedback signal is too sparse. A single, final reward score for an entire video sequence cannot precisely identify which individual clips adhere to their action conditions or pinpoint when visual quality degradation begins.

- 3) Optimization: these generated samples and their associated rewards are utilized to update the model, incentivizing the generation of high-reward trajectories.

To address this, we formulate clip-level rollout strategy specifically for autoregressive video generation. Considering semantic textual or image condition c, a sequence of action conditions a1:N and a target clip index n to rollout. we first autoregressively generate the preceding n − 1 clip. Subsequently, we generate G candidate rollouts for the n-th target clip. The generation and reward evaluation of the i-th sample at the n-th clip are formulated as follows:

To employ RL to autoregressive, interactive and longhorizon video-based world models, we should address three corresponding challenges:

- 1. How to generate rollout samples for the autoregressive video generation model? (Sec. 3.2)
- 2. How to design reward functions that reliably evaluate interactive generation? (Sec. 3.3)
- 3. How to effectively and efficiently optimize world modeling using RL? (Sec. 3.4)

x1:n−1 = πθ(·|a1:n−1,c) (2)

x(ni) = πθ(·|x1:n−1,an,c),s(i) = r(x(ni),an,c) (3) The clip-level rollout strategy offers two key advantages:

1. Rollout Efficiency: For world modeling, long-term interactive generation is crucial. Meanwhile, for RL, a large number of rollout samples is vital for stable training and strong performance, implying that both the video length (N) and rollout number (G) should be large. With our clip-level rollout, the preceding n − 1 clips are sampled only once and reused repeatedly. The model only performs G repetitive samplings for the nth clip. Consequently, the computational complexity of the rollout process is roughly reduced from O(N · G) to O(N + G), which significantly improves rollout sampling efficiency.

#### 3.2. Clip-level Rollout for Autoregressive Generation

As the most typical autoregressive generation model, we first study how a LLM performs its rollout. Given an input prompt c, the LLM model πllm, and the reward function r, RL scheme for LLMs would independently generate G sequences and score each. The computation process for the i-th sample x(1:i)n and its corresponding score s(i) can be formulated as:

x(1:i)n = πllm(·|c), s(i) = r(x(1:i)n,c) (1)

- 2. Consistent and Fine-grained Reward: Generating rollout samples from identical historical observations mitigates the potential inconsistencies that diverse prefixes introduce into current predictions. Furthermore, by evaluating these samples within the same context, we derive a more granular and comparable reward signal. These advantages enable a more targeted optimization for the model to synthesize high-fidelity clips, ensuring that gradient updates are driven by the quality of the current generation rather than historical variance.
- 3.3. Reward Functions for Interactive Generation

The effectiveness of an RL framework depends on its reward functions, as they dictate the direction of model improvement. In the context of interactive video generation, we focus on rewarding two fundamental attributes: interaction following and visual quality. Accordingly, we implement specific reward functions to assess each sampled clip.

Interaction Following Score For world model, interaction following evaluates whether the generated clip moves as the given action condition indicated. Following recent work like Genie 3 (Ball et al., 2025), the action signal is decomposed into: translation and rotation.

To assess these actions, we utilize advanced 3D foundation model (Liu et al., 2025d; Lin et al., 2025), to estimate the camera trajectory within the generated clip. This continuous trajectory is then mapped to a predefined discrete action space to calculate adherence accuracy.

For rotation, we determine the action between adjacent frames by comparing the estimated relative camera rotation against a predefined threshold, τrot. Evaluating translation action is more challenging because the detected positional scale varies across different scenes. Since a single universal threshold, τtrans, cannot generalize across all environments, we set multiple translation thresholds. A translation action is deemed correct if it matches the conditional input under any of the defined thresholds, ensuring a robust evaluation across diverse scenes.

Finally, to provide more discriminative guidance during optimization, we calculate the rotation and translation accuracies independently and define the final interaction following score as their average.

Visual Quality Score To evaluate the visual fidelity of the generated video, we employ HPSv3 (Ma et al., 2025) as our reward model, which inherently assesses both text-visual alignment and aesthetic quality. Specifically, we sample frames from each clip at 4-frame intervals and compute the average HPSv3 score across these frames to represent the overall visual quality.

Discussion Reward hacking is a common challenge in RL for generative models. Due to the inherent error accumulation characteristic in autoregressive video generation, world models are even more prone to such problem. In practice, we find that our two reward functions act as mutual regularizers. By balancing these two objectives, the framework prevents the model from optimizing one at the expense of the other, thereby suppressing reward hacking and leading to more robust training.

#### 3.4. Efficient RL Optimization

RL Algorithem Preliminary post-training methods for world models (Ye et al., 2025; He et al., 2025a) typically rely on FlowGRPO (Liu et al., 2025b), which employs SDE sampling to achieve stochastic exploration by generating multiple samples from the same initial noise. However, our investigation reveals that SDE sampling over same noise only diversifies the visual scenes but leaves the camera movement virtually unchanged. This limitation in exploring diverse camera trajectories constrains the potential improvement of the action following capabilities.

Inspired by DiffusionNFT (Zheng et al., 2025b), we perform policy optimization using a negative-aware fine-tuning strategy. The rollout data are sampled from different initial noises, and the model is directly trained with the flow matching objective. Given G samples at the n-th video clip

index, {x(ni)}Gi=1, with corresponding interaction following scores {s(IFi)}Gi=1 and visual quality scores {s(VQi)}Gi=1, we first compute the advantage for each reward dimension:

s(ji) − mean({s(ji)}Gi=1) std({s(ji)}Gi=1)

a(ji) =

, for j ∈ {IF, VQ} (4)

Then, we derive the optimality probability r(i) of the i-th sample through a clipped linear combination of the two normalized advantages:

r(i) =

- 1

- 2

- 1

- 2

+

clip

λa(IFi) + (1 − λ)a(VQi) Z

,−1,1 (5)

where λ is the trade-off hyperparameter, and Z is the normalizing factor. The final optimization loss is defined as:

2 2

2 2

L(θ) = Eit∼∼GT n∼N

+ (1 − r(i)) vθ− − v(i)

r(i) vθ+ − v(i)

(6)

where zt(i) = (1 − t)x(ni) + tϵ; ϵ ∼ N(0, I) vθ+ = (1 − β)vθold(zt(i), x1:n−1, a, c, t) + βvθ(zt(i), x1:n−1, a, c, t) vθ− = (1 + β)vθold(zt(i), x1:n−1, a, c, t) − βvθ(zt(i), x1:n−1, a, c, t) v(i) = x(0i) − ϵ

Algorithm 1 WorldCompass Training Process Require: Initial policy model πθ, its EMA copy πθ

; reward functions RIF,RVQ; prompt and action dataset D Ensure: Optimized policy model πθ

old

- 1: for each training iteration k do
- 2: Sample batch Db ∼ D // Clip-Level Rollout & Rewards, Sec. 3.2 & 3.3
- 3: Select target clip index for progressive training: n = (k modN) + 1
- 4: for each (c,a1:n) ∈ Db do
- 5: Generate shared prefix x1:n−1 and G rollout clips {x(ni)}Gi=1 using πθ

old

- 6: Compute rollout samples’ advantages {a(IFi),a(VQi)}Gi=1 with reward functions
- 7: Compute optimality probability r(i) by combining and clipping the advantages
- 8: end for
- 9: Subsample Best-of-N samples Gsub ⊂ 1 : G // Efficient RL Optimization, Sec. 3.4
- 10: Subsample a random set of timesteps Tsub ⊂ 1 : T
- 11: for i ∈ Gsub and t ∈ Tsub do
- 12: Forward diffusion process: zt(i) = (1 − t)x(ni) + tϵ; v(i) = x(ni) − ϵ
- 13: Calculate implicit positive/negative velocity vθ+,vθ− (Eq. 6)
- 14: Compute weighted loss: Li = r(i) vθ+ − v(i) 22 + (1 − r(i)) vθ− − v(i) 22
- 15: Update policy: θ ← θ − λlr∇θLi
- 16: end for
- 17: Update old policy: θold ← η θold + (1 − η)θ
- 18: end for

where the T denotes the diffusion sampling timesteps, and N represents the maximum length of clips used in training. The vθ

and vθ are predictions from the old model for rollout sampling and the current model undergoing training. Both models are initialized from the base model, and the old model is EMA updated by the trained model.

old

Notably, we omit the KL divergence loss used in the original DiffusionNFT (Zheng et al., 2025b). While KL regularization between the training and original model’s outputs typically mitigates mode collapse and reward hacking, we empirically observe limited performance improvement when including it. Instead, we employ a lower learning rate and the EMA update strategy to prevent over-optimization and achieve superior final results.

Efficient Training Strategy As shown in Eq. 6, the optimization theoretically involves iterations over the diffusion sampling timesteps (T), the number of rollout samples (G), and the maximum length of clips (N). This process is timeconsuming and computationally intensive. In practice, we employ several strategies to accelerate the training process:

- 1. For sampling timestep T, following observations in previous work (Xue et al., 2025; Liu et al., 2025b), we randomly select subsets of timesteps within a denoising trajectory for each training iteration, rather than processing all T steps. This significantly reduces computational overhead without compromising performance.
- 2. For rollout samples G, we adopt the Best-of-N selection strategy. Specifically, we select only the subsets of samples

ActionAccuracy(%)

HPSv3Score

Training Step Training Step

Figure 2. Evolution of interaction following and visual quality scores during the RL training of WorldPlay (HunyuanVideo-1.5 version). These reward metrics are evaluated on a fixed subset of the test set with complex combined action.

corresponding to the top 3 and bottom 3 rewards for training. This approach enhances training efficacy by focusing the model on the most informative cases, leveraging highreward samples for reinforcement and low-reward samples for error correction.

3. For clips length N, we implement a progressive optimization strategy. During training, the target clip index n (used for rollout and optimization) cycles incrementally from 1 to N, as shown in Line 3 of Alg. 1. This scheduling naturally induces a curriculum learning effect, adhering to RL principles by gradually increasing the task horizon. Furthermore, maintaining a unified video length across all parallel compute nodes maximizes hardware utilization and sampling efficiency.

### 4. Experiments

#### 4.1. Experimental Setup

Base Model We evaluate our framework using the recent WorldPlay (Sun et al., 2025) as our base world model. For a rigorous validation, we test two distinct variants of WorldPlay: HunyuanVideo-1.5-8B (Wu et al., 2025) and Wan2.25B (Wan et al., 2025). These models utilize eight basic actions: forward, backward, left, right movement, and up, down, left, right rotation. These individual basic movements can also be combined into complex compositional actions. Under the autoregressive generation paradigm, the video token sequence is processed as discrete chunks, where each chunk corresponds to a 16-frame video clip. To optimize the model for long-horizon generation, we set the maximum generation length N to 16 clips, totaling approximately 256 decoded frames.

Training Data Since all supervision is derived directly from reward functions, our framework eliminates the need for explicit manual annotations. We utilize a training set representative of real-world inference inputs, consisting of 4,000 diverse images and corresponding caption. Furthermore, to enhance the model’s robustness in challenging scenarios, we randomly constructed complex action sequence, primarily consisting of combinations of the basic actions.

Hyper-parameter During training, each step comprised 64 groups with a rollout group size of G = 16. The sampling step T for rollout is set to 40, from which 50% of the sub-sampling steps are randomly selected for training. The rotation threshold for the action-following score is set to 1◦. Based on the inherent predicted position scale of 3D foundation models, the multiple translation thresholds are set to [0.01,0.02,0.03,0.04,0.05]. The parameter λ and normalizing factor Z in Eq. 5 is 2/3 and 2 respectively. The parameter β in Eq. 6 is 1. We use the Muon (Team et al., 2025b) optimizer with a learning rate of 1e-5, and the EMA factor for the old model is linearly annealed from 0.4 to 0.8. The training process spans 3 days across 64 H20 GPUs.

#### 4.2. Main Results

Evaluation Protocol We evaluate our model using 600 cases from the WorldPlay test set. To better simulate realworld application scenarios, we redesign the action control sequences for each test sample into two categories: basic actions and composite actions. To comprehensively assess the model’s capabilities across various conditions, we evaluate different model variants across three video lengths: short (∼ 125 frames), medium (∼ 253 frames), and long (∼ 381 frames). We report the average action-following accuracy and the HPSv3 visual quality score, both computed across all generated clips using a 4-frame sampling interval.

Combined Action Basic Action Accaction HPSv3 Accaction HPSv3

HY-Video-1.5 21.74 -1.05 62.33 1.96 + WorldCompass 58.20 0.42 68.62 3.77 ∆ +36.46 +1.47 +6.29 +1.81

Short-term (125 frames)

Wan2.2 22.87 -1.10 58.28 1.83 +WorldCompass 49.81 0.20 64.72 3.17 ∆ +26.94 +1.30 +6.44 +1.34

HY-Video-1.5 19.73 -0.19 63.35 1.91 + WorldCompass 55.01 0.37 74.09 3.61 ∆ +35.28 +0.56 +10.74 +1.70

Mid-term (253 frames)

Wan2.2 20.33 -1.67 57.94 1.63 + WorldCompass 50.32 0.27 63.87 3.37 ∆ +29.99 +1.94 +5.93 +1.74

HY-Video-1.5 19.70 -0.33 64.28 1.90 + WorldCompass 54.82 0.73 76.56 3.72 ∆ +35.12 +1.06 +12.28 +1.82

Long-term (381 frames)

Wan2.2 19.58 -0.80 55.59 1.91 + WorldCompass 42.92 0.59 63.91 3.59 ∆ +23.34 +1.39 +8.32 +1.68

Table 1. Quantitative Results on two version of WorldPlay: HunyuanVideo-1.5-8B (HY-Video-1.5) and Wan2.2-5B (Wan-2.2). We compare the performance of two model variants across different scenarios: 1) the base model and 2) the model after WorldCompass post-training. The better results are highlighted in bold.

Quantitative Results Fig. 2 illustrates the evolution of reward scores throughout the RL training process. Notably, our method achieves significant performance gains in both interaction following and visual quality on challenging action inputs within a remarkably small number of training steps. In Table 1, we provide a comprehensive comparative analysis. Overall, WorldCompass yields substantial performance improvements across various base model versions and different video lengths. Specifically, for complex composite action inputs, our method improves average accuracy from approximately 20% to 55%. For basic action inputs, we observe a 10% improvement in interaction accuracy. Beyond achieving superior interaction following, the RL training also enhances the overall visual quality of the generated videos.

It is worth noting that our action accuracy is computed by matching every 4 frame with its corresponding action condition, which represents a rigorous evaluation standard. At lower score ranges such as 10% to 30%, errors primarily stem from the failure of the model to comprehend and execute the input action. At higher ranges between 50% and 60%, errors are mainly characterized by latencies during action switching. Consequently, the jump from 20% to 55% for complex actions represents a fundamental shift from failing to follow actions to successfully executing them. For simple actions, the improvement from 60% to 70% reflects more rapid responses during action switching.

|[Figure 17]<br><br>Base Model|
|---|

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

W

A W A

← ←

|[Figure 22]<br><br>+ WorldCompass|
|---|

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

W

A W A

← ←

|[Figure 27]<br><br>Base Model|
|---|

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

←

A

A

←

←

←

←

|[Figure 32]<br><br>+ WorldCompass|
|---|

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

←

A

A

←

←

←

←

Figure 3. Qualitative comparisons under complex combined action sequence.

Combined Action Basic Action Accaction HPSv3 Accaction HPSv3

Row Rollout Type IF Score VQ Score RL Algorithm

- 0 - - - - 19.70 -0.33 64.28 1.90

- 1 clip-level ✓ ✓ DiffusionNFT 54.82 0.73 76.56 3.72
- 2 sample-level ✓ ✓ DiffusionNFT 12.45 0.19 58.42 2.69
- 3 clip-level ✓ × DiffusionNFT 36.39 -2.67 67.60 -1.83
- 4 clip-level × ✓ DiffusionNFT 11.51 1.01 35.94 4.19
- 5 clip-level ✓ ✓ DanceGRPO 20.02 0.59 67.43 3.97

Table 2. Ablation study for the core components in WorldCompass. Line 0 provide the results of baseline without RL training.

Qualitative Results In Fig. 3 and 4, we provide visual comparisons of generated results with and without RL for both complex combined actions and simple basic action sequences, respectively. The qualitative results align with our quantitative findings: the WorldCompass post training framework significantly enhances the ability of the model to follow interactions while improving visual quality. Furthermore, as discussed in (He et al., 2025b), 3D foundation model successfully predict camera trajectories from a generated video clip that match the input condition also indirectly implies that the content possesses spatial geometric consistency. We also observe that the generated content demonstrates improved spatial consistency.

Subset of Timestep

Best-of-N Sampling

Accaction HPSv3 Iteration Time

✓ ✓ 54.82 0.73 1.00× ✓ × 55.28 0.75 1.42× × × 54.68 0.78 2.26×

Table 3. Ablation study of efficiency optimization strategies. The results are reported under combined action setting.

Sample- or Clip-level Rollout Comparing rows 0,1,2 in Tab. 2, we can summarize that a clip-level rollout is crucial for the effectiveness of WorldCompass. Specifically, samplelevel rollout tends to degrade the action following capability of the model, yielding only minor gains in visual quality. This phenomenon occurs because sample-level rollout results in an overly sparse reward density for long-duration videos. While visual quality often exhibits strong temporal dependencies that allow a holistic score to represent overall visual fidelity, action-following accuracy remains relatively independent across clips. Consequently, an aggregate sample-level score for action following averages out

#### 4.3. More Discussion

To further dissect our framework, we conduct more in-depth ablation studies. All the experiments are performed on the HY-Video-1.5 version of WorldPlay, and all results are reported under long-term generation setting.

|[Figure 37]<br><br>Base Model|
|---|

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

← ←

← ←

|[Figure 42]<br><br>+ WorldCompass|
|---|

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

← ←

← ←

|[Figure 47]<br><br>Base Model|
|---|

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

← ←

← ←

|[Figure 52]<br><br>+ WorldCompass|
|---|

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

← ←

← ←

Figure 4. Qualitative comparisons under simple basic action sequence.

Training Efficiency To investigate the impact of our efficiency optimization strategies on final performance and training iteration time, we provide a detailed comparative study in Tab. 2. The results demonstrate that our strategies reduces training overhead by 50% while maintaining competitive results. Selecting a subset of timesteps and employing a Best-of-N sample selection strategy preserves the most critical information within the RL training process, thereby significantly enhancing overall efficiency.

specific successes and failures, thereby failing to provide a discriminative reward signal. This coarse grained signal becomes uninformative or even misleading, ultimately hindering the model from improving its core capabilities.

Complementary Reward Functions We study the effect of individual reward functions by comparing rows 1, 3, and

- 4 in Tab. 2. Our results indicate that relying on a single reward function makes the model highly susceptible to reward hacking. Specifically, optimizing solely for the Interaction Following (IF) Score improves action accuracy but triggers a noticeable decline in visual quality. This degradation in visual fidelity severely undermines training stability, often leading to model collapse. Conversely, using only the Visual Quality (VQ) Score yields aesthetically pleasing results but produces static or motionless content. When both rewards are applied simultaneously, they act as a mutual constraint. This synergy guides the model toward the desired direction of improvement, achieving superior performance across both dimension.

### 5. Conclusion

We introduce WorldCompass, a novel online reinforcement learning framework tailored specifically for world models. Recognizing the interactive, long-horizon, and autoregressive nature of world modeling tasks, we redesign the RL training pipeline for diffusion models to better address these unique characteristics. Specifically, we introduce a clip-level rollout strategy that provides fine-grained rewards while boosting rollout efficiency for long-term autoregressive video generation. Furthermore, we incorporate complementary reward functions that act as mutual constraints to stably improve both interactive accuracy and visual quality. Finally, we utilize a negative-aware fine-tuning algorithm for RL model optimization. Experimental results demonstrate that WorldCompass significantly enhances the state-of-theart world model, WorldPlay, particularly when handling challenging compositional action sequences.

Alternative RL Algorithm In row 4 of Tab. 2, we valuate the necessity of DiffusionNFT (Zheng et al., 2025b) by replacing it with DanceGRPO (Xue et al., 2025), another widely-used RL algorithm for diffusion models. As analyzed in Sec. 3.4, DanceGRPO generates rollout video clips with minimal camera motion variance. This lack of diversity restricts the search space for optimal policies within the world modeling task, which ultimately results in marginal improvements in interaction-following capability.

### Impact Statements

This paper presents work whose goal is to advance the field of machine learning. There are many potential societal consequences of our work, none of which we feel must be specifically highlighted here.

### References

Agarwal, N., Ali, A., Bala, M., Balaji, Y., Barker, E., Cai, T., Chattopadhyay, P., Chen, Y., Cui, Y., Ding, Y., et al. Cosmos world foundation model platform for physical ai. arXiv preprint arXiv:2501.03575, 2025.

Ali, A., Bai, J., Bala, M., Balaji, Y., Blakeman, A., Cai, T., Cao, J., Cao, T., Cha, E., Chao, Y.-W., et al. World simulation with video foundation models for physical ai. arXiv preprint arXiv:2511.00062, 2025.

Azzolini, A., Bai, J., Brandon, H., Cao, J., Chattopadhyay, P., Chen, H., Chu, J., Cui, Y., Diamond, J., Ding, Y., et al. Cosmos-reason1: From physical common sense to embodied reasoning. arXiv preprint arXiv:2503.15558, 2025.

Ball, P. J., Bauer, J., Belletti, F., Brownfield, B., Ephrat, A., Fruchter, S., Gupta, A., Holsheimer, K., Holynski, A., Hron, J., et al. Genie 3: A new frontier for world models. 2025.

Bruce, J., Dennis, M. D., Edwards, A., Parker-Holder, J., Shi, Y., Hughes, E., Lai, M., Mavalankar, A., Steigerwald, R., Apps, C., et al. Genie: Generative interactive environments. In Forty-first International Conference on Machine Learning, 2024.

Chen, B., Mart´ı Mons´o, D., Du, Y., Simchowitz, M., Tedrake, R., and Sitzmann, V. Diffusion forcing: Nexttoken prediction meets full-sequence diffusion. Advances in Neural Information Processing Systems, 37:24081– 24125, 2024.

Ding, J., Zhang, Y., Shang, Y., Zhang, Y., Zong, Z., Feng, J., Yuan, Y., Su, H., Li, N., Sukiennik, N., et al. Understanding world or predicting future? a comprehensive survey of world models. ACM Computing Surveys, 58(3):1–38, 2025.

Google. Veo 3. 2025. URL https://deepmind. google/models/veo/.

He, H., Xu, Y., Guo, Y., Wetzstein, G., Dai, B., Li, H., and Yang, C. Cameractrl: Enabling camera control for textto-video generation. arXiv preprint arXiv:2404.02101, 2024.

He, H., Patrikar, J., Kim, D.-K., Smith, M., McGann, D., Agha-mohammadi, A.-a., Omidshafiei, S., and Scherer, S. Grndctrl: Grounding world models via self-supervised reward alignment. arXiv preprint arXiv:2512.01952, 2025a.

He, H., Yang, C., Lin, S., Xu, Y., Wei, M., Gui, L., Zhao, Q., Wetzstein, G., Jiang, L., and Li, H. Cameractrl ii: Dynamic scene exploration via camera-controlled video diffusion models. arXiv preprint arXiv:2503.10592, 2025b.

He, X., Peng, C., Liu, Z., Wang, B., Zhang, Y., Cui, Q., Kang, F., Jiang, B., An, M., Ren, Y., et al. Matrix-game 2.0: An open-source real-time and streaming interactive world model. arXiv preprint arXiv:2508.13009, 2025c.

Huang, T., Zheng, W., Wang, T., Liu, Y., Wang, Z., Wu, J., Jiang, J., Li, H., Lau, R., Zuo, W., et al. Voyager: Longrange and world-consistent video diffusion for explorable 3d scene generation. ACM Transactions on Graphics (TOG), 44(6):1–15, 2025.

HunyuanWorld, T. Hunyuanworld 1.0: Generating immersive, explorable, and interactive 3d worlds from words or pixels. arXiv preprint, 2025.

Li, J., Tang, J., Xu, Z., Wu, L., Zhou, Y., Shao, S., Yu, T., Cao, Z., and Lu, Q. Hunyuan-gamecraft: High-dynamic interactive game video generation with hybrid history condition. arXiv preprint arXiv:2506.17201, 2025a.

- Li, W., Pan, W., Luan, P.-C., Gao, Y., and Alahi, A. Stable video infinity: Infinite-length video generation with error recycling. arXiv preprint arXiv:2510.09212, 2025b.
- Li, X., Wang, T., Gu, Z., Zhang, S., Guo, C., and Cao, L. Flashworld: High-quality 3d scene generation within seconds, 2025c.

Liao, Y., Zhou, P., Huang, S., Yang, D., Chen, S., Jiang, Y., Hu, Y., Cai, J., Liu, S., Luo, J., et al. Genie envisioner: A unified world foundation platform for robotic manipulation. arXiv preprint arXiv:2508.05635, 2025.

Lin, H., Chen, S., Liew, J., Chen, D. Y., Li, Z., Shi, G., Feng, J., and Kang, B. Depth anything 3: Recovering the visual space from any views. arXiv preprint arXiv:2511.10647, 2025.

Guo, D., Yang, D., Zhang, H., Song, J., Zhang, R., Xu, R., Zhu, Q., Ma, S., Wang, P., Bi, X., et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

Liu, A., Mei, A., Lin, B., Xue, B., Wang, B., Xu, B., Wu, B., Zhang, B., Lin, C., Dong, C., et al. Deepseek-v3. 2: Pushing the frontier of open large language models. arXiv preprint arXiv:2512.02556, 2025a.

Liu, J., Liu, G., Liang, J., Li, Y., Liu, J., Wang, X., Wan, P., Zhang, D., and Ouyang, W. Flow-grpo: Training flow matching models via online rl. arXiv preprint arXiv:2505.05470, 2025b.

Liu, Y., Chen, W., Bai, Y., Liang, X., Li, G., Gao, W., and Lin, L. Aligning cyber space with physical world: A comprehensive survey on embodied ai. IEEE/ASME Transactions on Mechatronics, 2025c.

Liu, Y., Min, Z., Wang, Z., Wu, J., Wang, T., Yuan, Y., Luo, Y., and Guo, C. Worldmirror: Universal 3d world reconstruction with any-prior prompting. arXiv preprint arXiv:2510.10726, 2025d.

Ma, Y., Wu, X., Sun, K., and Li, H. Hpsv3: Towards widespectrum human preference score. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 15086–15095, 2025.

Parker-Holder, J., Ball, P., Bruce, J., Dasagi, V., Holsheimer, K., Kaplanis, C., Moufarek, A., Scully, G., and others, J. S. Genie 2: A large-scale foundation world model. 2024.

Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang, H., Zhang, M., Li, Y., Wu, Y., et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., and Poole, B. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020.

Sun, W., Zhang, H., Wang, H., Wu, J., Wang, Z., Wang, Z., Wang, Y., Zhang, J., Wang, T., and Guo, C. Worldplay: Towards long-term geometric consistency for real-time interactive world modeling. arXiv preprint arXiv:2512.14614, 2025.

Tang, J., Liu, J., Li, J., Wu, L., Yang, H., Zhao, P., Gong, S., Yuan, X., Shao, S., and Lu, Q. Hunyuan-gamecraft2: Instruction-following interactive game world model. arXiv preprint arXiv:2511.23429, 2025.

Team, G., Ye, A., Wang, B., Ni, C., Huang, G., Zhao, G., Li, H., Li, J., Zhu, J., Feng, L., et al. Gigabrain-0: A world model-powered vision-language-action model. arXiv preprint arXiv:2510.19430, 2025a.

Team, K., Bai, Y., Bao, Y., Chen, G., Chen, J., Chen, N., Chen, R., Chen, Y., Chen, Y., Chen, Y., et al. Kimi k2: Open agentic intelligence. arXiv preprint arXiv:2507.20534, 2025b.

Valevski, D., Leviathan, Y., Arar, M., and Fruchter, S. Diffusion models are real-time game engines. arXiv preprint arXiv:2408.14837, 2024.

Wallace, B., Dang, M., Rafailov, R., Zhou, L., Lou, A., Purushwalkam, S., Ermon, S., Xiong, C., Joty, S., and Naik, N. Diffusion model alignment using direct preference optimization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 8228–8238, 2024.

Wan, T., Wang, A., Ai, B., Wen, B., Mao, C., Xie, C.-W., Chen, D., Yu, F., Zhao, H., Yang, J., et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

Wang, Z., Yuan, Z., Wang, X., Li, Y., Chen, T., Xia, M., Luo, P., and Shan, Y. Motionctrl: A unified and flexible motion controller for video generation. In ACM SIGGRAPH 2024 Conference Papers, pp. 1–11, 2024.

Wu, B., Zou, C., Li, C., Huang, D., Yang, F., Tan, H., Peng, J., Wu, J., Xiong, J., Jiang, J., et al. Hunyuanvideo 1.5 technical report. arXiv preprint arXiv:2511.18870, 2025.

Xue, Z., Wu, J., Gao, Y., Kong, F., Zhu, L., Chen, M., Liu, Z., Liu, W., Guo, Q., Huang, W., et al. Dancegrpo: Unleashing grpo on visual generation. arXiv preprint arXiv:2505.07818, 2025.

Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Gao, C., Huang, C., Lv, C., et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

Ye, Y., He, T., Yang, S., and Bian, J. Reinforcement learning with inverse rewards for world model post-training. arXiv preprint arXiv:2509.23958, 2025.

Yu, J., Qin, Y., Wang, X., Wan, P., Zhang, D., and Liu, X. Gamefactory: Creating new games with generative interactive videos. arXiv preprint arXiv:2501.08325, 2025.

Zheng, C., Liu, S., Li, M., Chen, X.-H., Yu, B., Gao, C., Dang, K., Liu, Y., Men, R., Yang, A., et al. Group sequence policy optimization. arXiv preprint arXiv:2507.18071, 2025a.

Zheng, K., Chen, H., Ye, H., Wang, H., Zhang, Q., Jiang, K., Su, H., Ermon, S., Zhu, J., and Liu, M.-Y. Diffusionnft: Online diffusion reinforcement with forward process. arXiv preprint arXiv:2509.16117, 2025b.

### A. Limitation

Currently, there is a lack of reliable metrics to evaluate visual quality drift and spatial memory retention in long-form video generation. Consequently, our reward signal lacks the direct constraints to penalize such drift, which leads to cumulative quality degradation when applying large-scale RL training to long-duration autoregressive video generation. While we currently mitigate this issue through a conservative training strategy by employing fewer iterations and a reduced learning rate, a more fundamental solution would involve a robust reward function specifically designed to evaluate visual drift and spatial memory. This represents a promising direction for future research, although it remains beyond the scope of this work.

### B. More Qualitative Results

We provide additional qualitative results comparing the results before and after WorldCompass RL training in Fig. 5, 6, 7 and 8. To facilitate better visualization, we reconstructed the 3D scenes and camera trajectories from the generated videos. For a more intuitive comparison, we evaluate the models across multiple distinct scenarios using a consistent action sequence: the first half consists of the “W+A” command (moving forward-left), followed by a “→” command (turning right) in the second half. As illustrated, after undergoing WorldCompass training, the world model demonstrates significantly improved action-following accuracy and superior geometric consistency.

Base Model + WorldCompass

[Figure 57]

[Figure 58]

- Figure 5. Visualization Case 1. The input action sequence consists of W+A” (moving forward-left) for the first half, followed by →” (turning right) in the second half.

Base Model

[Figure 59]

+ WorldCompass

[Figure 60]

- Figure 6. Visualization Case 2. The input action sequence consists of W+A” (moving forward-left) for the first half, followed by →” (turning right) in the second half.

Base Model

[Figure 61]

+ WorldCompass

[Figure 62]

- Figure 7. Visualization Case 3. The input action sequence consists of W+A” (moving forward-left) for the first half, followed by →” (turning right) in the second half.

Base Model

[Figure 63]

+ WorldCompass

[Figure 64]

- Figure 8. Visualization Case 4. The input action sequence consists of W+A” (moving forward-left) for the first half, followed by →” (turning right) in the second half.

