# arXiv:2601.09708v2[cs.CV]24Feb2026

[Figure 1]

2026-2-25

## Fast-ThinkAct: Efficient Vision-Language-Action Reasoning via Verbalizable Latent Planning

Chi-Pin Huang1, Yunze Man2, Zhiding Yu, Min-Hung Chen, Jan Kautz, Yu-Chiang Frank Wang1, Fu-En Yang

NVIDIA

### Abstract

Vision-Language-Action (VLA) tasks require reasoning over complex visual scenes and executing adaptive actions in dynamic environments. While recent studies on reasoning VLAs show that explicit chain-of-thought (CoT) can improve generalization, they suffer from high inference latency due to lengthy reasoning traces. We propose Fast-ThinkAct, an efficient reasoning framework that achieves compact yet performant planning through verbalizable latent reasoning. Fast-ThinkAct learns to reason efficiently with latent CoTs by distilling from a teacher, driven by a preference-guided objective to align manipulation trajectories that transfers both linguistic and visual planning capabilities for embodied control. This enables reasoning-enhanced policy learning that effectively connects compact reasoning to action execution. Extensive experiments across diverse embodied manipulation and reasoning benchmarks demonstrate that Fast-ThinkAct achieves strong performance with up to 89.3% reduced inference latency over state-of-the-art reasoning VLAs, while maintaining effective long-horizon planning, few-shot adaptation, and failure recovery.

Links: Project Page

### 1. Introduction

Recent large vision-language models (VLMs) Liu

- et al. (2023); Comanici et al. (2025); Liu et al.

- (2024); Bai et al. (2025); Shi et al. (2024); Li et al. (2025); Chen et al. (2025); Wang et al. (2025); Xie et al. (2024) have achieved remarkable capabilities in visual-language understanding across diverse multimodal tasks. To extend these capabilities to embodied-centric tasks, recent works leverage large-scale robot demonstrations O’Neill et al. (2024) to develop Vision-Language-Action (VLA) foundation models Brohan et al. (2022, 2023); Team

et al. (2024); Bjorck et al. (2025); Li et al.

- (2025); Black et al. (2024); Yang et al. (2025); Kim et al. (2024). These VLA tasks require agents to perceive complex visual scenes, reason over spatial and temporal contexts, and execute adaptive actions within dynamic environments, demanding robust long-horizon planning and contextual adaptation. However, as these VLA models primarily rely on supervised training from action data, they excel at basic skills (e.g., pick-and-place) but struggle to generalize beyond training distributions, such as long-horizon planning, self-

Previous Works (e.g., ThinkACt)

Actions

"To" "put" "the"

"[" "(0.12" ","

...

...

~250 tokens

Reasoning VLA

...

[Figure 2]

"Put the 🍓 in the drawer"

[Figure 3]

Ours To put the 🍓 ...

(0.12, 0.34)

(0.12, 0.34)

Actions

...

6 continuous tokens

9.3x faster

Reasoning VLA

...

[Figure 4]

"Put the 🍓 in the drawer"

Figure 1: Overview of Fast-ThinkAct. Previous reasoning VLAs generate lengthy reasoning traces (∼250 tokens). Our approach learns compact continuous tokens (e.g., 6) (blue) and parallel spatial tokens (green) as internal reasoning. The bottom-right plot shows that we achieve 9.3× faster inference than ThinkAct-7B Huang et al. (2025), while delivering improved performance on the SimplerEnv-Google benchmark.

Additional affiliations: 1 National Taiwan University. 2 University of Illinois Urbana-Champaign. © 2026 NVIDIA. All rights reserved.

correction from failures, and adaptation to novel scenarios, due to the impracticality of collecting exhaustive robot demonstrations.

Reasoning VLAs Zawalski et al. (2024); Zhao et al. (2025); Lee et al. (2025); Wu et al. (2025); Qu et al. (2025); Huang et al. (2025); Kim et al. (2025) address these limitations by incorporating intermediate thinking processes, improving generalization and task-solving capability. Supervised chain-of-thought (CoT) methods Zawalski et al. (2024); Zhao et al. (2025); Lee et al. (2025); Qu et al. (2025) address this by learning from intermediate reasoning annotations. These approaches can be categorized into textual reasoning methods that leverage off-the-shelf LLMs and VLMs to generate pseudo CoT labels Zawalski et al. (2024), and visual reasoning methods that generate structured visual reasoning representations such as sub-goal images, image depth, and 2D visual traces Zhao et al. (2025); Lee et al. (2025). However, these supervised approaches require substantial reasoning annotations and remain limited by training data coverage. To address this, ThinkAct Huang et al. (2025) employs RL-based reasoning Shao et al. (2024) to generate long textual CoTs guided by action-aligned visual rewards. While these reasoning methods effectively improve task generalization and planning capabilities, they require generating lengthy chain-of-thought steps that introduce substantial reasoning latency, which hampers embodied applications with real-time requirements.

In embodied AI applications such as robotic manipulation and autonomous driving, agents must make rapid decisions at high frequencies (e.g., 1-15 Hz) Guan et al. (2025). However, generating lengthy reasoning traces can take several seconds per decision (e.g., 0.1 Hz) Huang et al. (2025); Lee et al. (2025), creating a critical bottleneck that limits real-time performance Guan et al. (2025); Yu et al. (2025) and poses safety risks in time-critical scenarios Wang et al. (2025). To mitigate this efficiency bottleneck while preserving reasoning capabilities, very recent works Chen et al. (2025); Yu et al. (2025); Guan et al. (2025) have explored approaches to reduce inference latency in embodied reasoning. For instance, ECoT-Lite Chen et al. (2025) proposes reasoning dropout to accelerate inference, yet directly reducing textual reasoning length risks performance degradation due to critical information loss. How to preserve reasoning capability while enabling compact representations that properly capture essential spatial-temporal dynamics remains a crucial challenge for reasoning VLA models.

In this paper, we propose Fast-ThinkAct, an efficient embodied reasoning framework for Vision-Language-Action tasks that achieves compact yet expressive planning through verbalizable latent reasoning. As depicted in Figure 1, unlike prior reasoning VLAs that generate lengthy explicit textual CoT traces, we introduce rewardguided preference distillation with visual trajectory alignment to compress linguistic and visual planning into compact continuous latents that enable implicit internal reasoning. Our student VLM encodes reasoning into compact latents decodable by a verbalizer, enabling preference-based optimization that leverages RL-derived reward signals to distill high-quality reasoning patterns from a textual teacher VLM while suppressing lowquality ones. We further align trajectory latents between teacher and student to transfer visual planning capabilities essential for embodied control. Once trained, the student VLM enables reasoning-enhanced policy learning that bridges implicit multimodal planning with action execution, achieving significantly faster inference while outperforming existing reasoning VLAs.

Our contributions can be summarized as follows:

- • We propose Fast-ThinkAct, an efficient reasoning framework that compresses reasoning into verbalizable latent thoughts while maintaining expressive planning abilities.
- • We introduce preference-guided distillation with manipulation trajectory alignment that compresses linguistic and visual planning into compact continuous latents.
- • We bridge high-level visual planning to low-level action execution through reasoning-enhanced policy learning guided by manipulation trajectory latents.
- • We achieve up to 89.3% inference latency reduction over state-of-the-art reasoning VLAs while maintaining strong performance across diverse embodied benchmarks.

### 2. Related Works

- 2.1. Vision-Language-Action (VLA) Models Foundation VLAs. Vision-Language-Action (VLA) models Brohan et al. (2022, 2023); Team et al. (2024); Bjorck et al. (2025); Li et al. (2025); Black et al. (2024); Yang et al. (2025); Pertsch et al. (2025); Driess et al. (2025); Bu et al. (2025); Team et al. (2025); Wang et al. (2025) have recently emerged as a promising paradigm for embodied AI by training vision-language backbones on large-scale robot demonstrations. Works

such as OpenVLA Kim et al. (2024) and 𝜋0 Black et al. (2024) achieve language-conditioned manipulation through end-to-end policy learning, while Magma Yang et al. (2025) co-trains on heterogeneous human and robot data. HAMSTER Li et al. (2025) and TraceVLA Zheng et al. (2024) further leverage 2D visual trajectories to boost spatial-action connections. Despite success on routine manipulation, these imitation-based approaches struggle with long-horizon planning and generalization to novel scenarios due to limited training data coverage.

Reasoning VLAs. To overcome these limitations, recent works Zawalski et al. (2024); Zhao et al. (2025); Lee et al. (2025); Wu et al. (2025); Qu et al. (2025); Huang et al. (2025); Kim et al. (2025); Yuan et al. (2025); Abdolmaleki et al. (2025) integrate explicit reasoning mechanisms into VLA architectures. Supervised approaches Zawalski et al. (2024); Zhao et al. (2025); Lee et al. (2025); Qu et al. (2025) introduce intermediate reasoning through chain-of-thought annotations. Embodied CoT Zawalski et al. (2024) and Hi-Robot Shi et al. (2025) synthesize reasoning labels via pretrained foundation models. To perform vision-centric reasoning Man et al. (2025); Sarch et al. (2025) beyond pure text, CoT-VLA Zhao et al. (2025) employs visual goal generation and MolmoAct Lee et al. (2025) structures reasoning by spatial representations. Additionally, EO-1 Qu et al. (2025) introduces interleaved vision-language-action pre-training to bridge reasoning and interaction. Recent works Yuan et al. (2025); Huang et al. (2025) alternatively leverage reinforcement fine-tuning to generate reasoning chains with designed rewards. Despite improved generalization, these reasoning VLAs suffer from high inference latency and inevitably introduce extraneous information that degrades action quality.

- 2.2. Efficient Reasoning To address the inference latency of reasoning, recent LLM research explores various efficiency techniques Lee et al. (2025); Dai et al. (2025); Yuan et al. (2025); Xiang et al. (2025); Aggarwal and Welleck (2025); Lee et al.

(2025). For example, RL-based approaches Dai et al. (2025); Yuan et al. (2025); Xiang et al. (2025); Aggarwal and Welleck (2025) introduce length penalties to encourage shorter reasoning chains, though such methods can suffer from training instability. Beyond length control, latent reasoning methods Hao et al. (2024); Shen et al. (2025); Zhang et al. (2025); Cheng and Van Durme (2024); Xu et al. (2025) enable reasoning in continuous spaces, such as Coconut Hao et al. (2024) using hidden states as continuous thoughts, CODI Shen et al. (2025) distilling explicit CoT into continuous space via teacher-student alignment, and Soft Thinking Zhang et al. (2025) generating weighted concept tokens. However, these LLM techniques cannot directly transfer to VLA tasks due to the need for spatial-temporal understanding and bridging semantic reasoning with embodied control. Recently, ECoT-Lite Chen et al. (2025) proposes reasoning dropout to accelerate embodied reasoning by skipping test-time reasoning traces. However, reasoning dropout can lead to inconsistent planning as it builds on supervised embodied CoT. Our proposed Fast-ThinkAct distills reasoning into compact latent representations that naturally encode multimodal information, enabling robust reasoning-enhanced policy learning.

- 3. Method

##### 3.1. Problem Formulation We first define the setting and notations. At each timestep 𝑡, given a language instruction 𝑙, the model observes

a visual input 𝑜𝑡 and generates an action chunk 𝑎𝑡, represented as a sequence of continuous robot control vectors (e.g., 7- or 14-DOF for single- or bimanual robots, respectively).

To address this problem, we propose Fast-ThinkAct, an efficient reasoning framework that bridges high-level

[Figure 5]

<think> To put the ... </think> <ans>[(0.12, 0.34), ...]</ans>

[Figure 6]

###### GRPO Rollouts

Reward

... ... ... ...

[Figure 7]

...

The image is ... 0.3

[Figure 8]

...

...

Textual Teacher

Spatial KV

0.9

To put the 🍓 ...

Action Model

...

[Figure 9]

[Figure 10]

"Put the 🍓 in the drawer"

... ...

<think> To

</think>

<ans>

State Encoder

...

To put the 🍓 in the ...

MLP

...

...

[Figure 11]

Latent Student

Spatial KV

Latent Student

Verbalizer LLM

...

[Figure 12]

... ...

"Put the 🍓 in the drawer"

<think>

</think> <ans>

(a) (b)

- Figure 2: Overview of Fast-ThinkAct. (a) Given observation 𝑜𝑡 and instruction 𝑙, the Textual Teacher VLM

ℱ𝜃𝑇 generates explicit reasoning chains. The Latent Student VLM ℱ𝜃 distills these into compact latent tokens z guided by reward preferences. Verbalizer LLM 𝒱𝜓 decodes latents to text for preference-based learning via ℒverb, while ℒdistill transfers visual planning capability from teacher, and spatial tokens enable parallel visual trajectory prediction via ℒans, ensuring latents are verbalizable and grounded in visual planning. (b) Reasoning-Enhanced Policy Learning. The Action Model 𝜋𝜑 is trained with ℒIL while freezing the latent student ℱ𝜃 and state encoder.

planning with low-level action execution. Our approach employs a VLM ℱ𝜃 to perform reasoning in continuous latent space, integrated with an action model 𝜋𝜑 for executable action generation. Specifically, ℱ𝜃 processes observation-instruction pairs (𝑜𝑡,𝑙) through latent chain-of-thought (CoT) reasoning to produce a compact visual plan latent 𝑐𝑡 that encapsulates the intended trajectory in visual space (Sec. 3.2). This 𝑐𝑡 subsequently guides 𝜋𝜑 to predict executable actions 𝑎𝑡 (Sec. 3.3). By distilling reasoning into a continuous latent space rather than discrete text, Fast-ThinkAct achieves significantly improved inference efficiency while enhancing action performance through better preservation of spatial and visual information.

- 3.2. Efficient Embodied Reasoning To enable efficient embodied reasoning that meets the real-time requirements of embodied AI tasks, we aim to compress long textual CoTs into a compact set of continuous latent representations. However, compressing reasoning traces into latents is challenging, as there is no direct supervision signal in the latent space to guide what reasoning patterns should be encoded.

- 3.2.1. Verbalizable Latent CoT by Reward Preferences To address this challenge, we propose to perform distillation in natural language space by introducing a verbalizer LLM that decodes latents into verbalizable reasoning. This approach grounds latent learning in an interpretable textual form, ensuring that the learned latents faithfully preserve the underlying reasoning

structure. Since reasoning traces generated by the teacher model ℱ𝜃𝑇 exhibit varying quality, we adopt a preference-based learning framework that exploits reward signals from the teacher’s GRPO training to guide

the latent student ℱ𝜃 toward high-quality reasoning patterns while suppressing low-quality ones. Specifically, we employ a teacher-student framework where a textual teacher model ℱ𝜃𝑇 first learns explicit reasoning through GRPO Shao et al. (2024) training by maximizing:

[︁min

)︀]︁, (1)

(︀

𝑟𝜃(𝜏)𝐴(𝜏),clip(𝑟𝜃(𝜏),1 − 𝜖,1 + 𝜖)𝐴(𝜏)

###### 𝒥GRPO(𝜃) = E𝜏∼ℱ𝑇

𝜃

where 𝜏 denotes a reasoning trace and 𝑟𝜃(𝜏) = ℱ

ℱold𝑇 (𝜏) is the probability ratio. The advantage function for group

𝑇 𝜃 (𝜏)

rewards {𝑅𝑖}𝑖∈𝐺(𝜏) is represented as:

𝐴(𝜏) =

𝑅𝜏 − mean({𝑅𝑖}𝑖∈𝐺(𝜏)) std({𝑅𝑖}𝑖∈𝐺(𝜏))

. (2)

This training process produces textual CoTs with varying quality, where the advantage function 𝐴(𝜏) naturally serves as a quality indicator. To construct preference pairs for distillation, we select the highest and lowest advantage traces from each rollout group:

𝐴(𝜏) and 𝜏− = arg min 𝜏∈𝐺

𝐴(𝜏). (3)

𝜏+ = arg max 𝜏∈𝐺

Instead of generating textual tokens, the student model ℱ𝜃 performs latent reasoning by autoregressively generating 𝑀 continuous latent vectors z = {𝑧𝑚}𝑀𝑚=1 with 𝑧𝑚 ∈ R𝑑, where 𝑑 is the hidden size. We then train the verbalizer LLM 𝒱𝜓 to decode these latents z into natural language. The training objective encourages the verbalizer to assign a higher likelihood to decoding latents into high-quality reasoning 𝜏+ than low-quality reasoning 𝜏−. Inspired by DPO Rafailov et al. (2023), we formulate this as an optimization guided by the reward preferences:

𝑝ref(𝜏−) )︀)︁]︁, (4)

ℒverb = −E[︁log 𝜎(︁𝛽

(︀

−|z)

+|z)

𝑝ref(𝜏+) − log 𝑝𝜓(𝜏

log 𝑝𝜓(𝜏

where 𝑝ref is the reference model (i.e., 𝒱𝜓 without latent conditioning), 𝜎 is the sigmoid function, and 𝛽 = 0.1 controls preference strength. This encourages the student VLM ℱ𝜃 to encode latents that the verbalizer decodes into high-quality reasoning while suppressing low-quality patterns.

###### 3.2.2. Action-Aligned Visual Plan Distillation

While the verbalizer loss (Eq. 4) enables the student ℱ𝜃 to capture high-level reasoning patterns, it does not explicitly ensure that latent representations encode the visual planning capability crucial for embodied

control. To address this, we introduce action-aligned visual plan distillation to transfer the teacher ℱ𝜃𝑇’s spatial reasoning ability to the student ℱ𝜃.

We distill spatial reasoning from the teacher, which is trained with trajectory-level rewards (e.g., goal completion and trajectory alignment Huang et al. (2025)) for grounded visual planning. We align the trajectory-level representations by minimizing the L2 distance between hidden states of the <answer> token that encodes the visual plan:

ℒdistill = ‖ℎ𝑇𝑡 − ℎ𝑡‖22, (5) where ℎ𝑇𝑡 and ℎ𝑡 are the hidden states from teacher (corresponding to 𝜏+) and student, respectively.

To enable efficient parallel trajectory prediction, unlike the textual teacher that autoregressively generates verbose text sequences of waypoints {𝑝𝑘}𝐾𝑘=1 with 𝑝𝑘 ∈ [0,1]2 (tokenized into 60-70 tokens when 𝐾 = 5), the student uses 𝐾 learnable spatial tokens {s𝑖}𝐾𝑖=1 appended to the reasoning latent sequence, with each output hidden state simultaneously projected to a waypoint via an MLP. The total objective for training ℱ𝜃 combines all three components:

ℒstudent = ℒverb + ℒdistill + ℒans, where

(6)

∑︁𝐾

‖𝑝𝑖 − 𝑝ˆ𝑖‖22, with 𝑝𝑖 = MLP(ℎ′(s𝑖)),

ℒans =

𝑖=1

where ℎ′(s𝑖) denotes the output hidden state of the 𝑖-th spatial token and 𝑝ˆ𝑖 are ground-truth waypoints. Through this unified framework, the student model ℱ𝜃 performs compact yet expressive latent reasoning and generates visual trajectory plans efficiently.

OpenVLA CoT-VLA

84.7

OpenVLA CoT-VLA

88.4 91.6

OpenVLA CoT-VLA

79.2

87.5 88.3 87.0

87.6

ThinkAct MolmoAct

91.4

ThinkAct MolmoAct

ThinkAct MolmoAct

87.1

95.4 97.2

87.6 90.2

92.0

Ours

Ours

Ours

0 20 40 60 80

0 20 40 60 80

100

0 20 40 60 80

Success rate (%)

Success rate (%)

Success rate (%)

(a) LIBERO-Spatial (b) LIBERO-Object (c) LIBERO-Goal

Pre-fill Reasoning Action Predicting

OpenVLA CoT-VLA

53.7

40.2

OpenVLA

|805| |-89.3%|67 5674<br><br>|7513 23|
|---|---|---|---|---|
| | | | | |

69.0 70.9 77.2 79.4

ThinkAct-7B MolmoAct-7B

ThinkAct

68.3

ThinkAct MolmoAct

64.9

MolmoAct

ThinkAct-3B

8

Ours

68.7

Ours

Ours (3B)

0 20 40 60 80

0 20 40 60 80

0 1500 3000 4500 6000 7500

Success rate (%)

Success rate (%)

Inference latency (ms)

(d) LIBERO-Long (e) SimplerEnv-Google (f) Latency of Reasoning VLAs

- Figure 3: Evaluation of robot manipulation and reasoning efficiency. (a)-(e) Success rates on LIBERO Liu et al. (2023) and SimplerEnv Li et al. (2024) benchmarks compared with state-of-the-art 7B reasoning VLAs. (f) Latency comparison across 3B and 7B reasoning VLAs. Our approach achieves up to 89.3% inference latency reduction while maintaining superior task success rates.

#### 3.3. Reasoning-Enhanced Policy Learning

After the student VLM ℱ𝜃 performs compact latent reasoning and generates visual trajectory planning through spatial tokens, we leverage these representations to guide a diffusion Transformer-based action model 𝜋𝜑 (e.g., RDT Liu et al. (2024)) for action prediction. To bridge the high-level visual planning with low-level action generation, we connect the visual latent planning 𝑐𝑡 encoded in the key-value cache corresponding to the spatial tokens to the action model.

Specifically, we extract visual latent planning 𝑐𝑡 from the KV cache of spatial tokens in earlier VLM layers (since ℱ𝜃 has more layers than 𝜋𝜑) and concatenate with KV pairs from the action model’s state encoder. The action model’s cross-attention then attends to both the visual planning context and state observations. We post-train on action-annotated robot data by freezing ℱ𝜃 and the state encoder while updating only 𝜋𝜑 with the imitation learning objective:

ℒIL(𝜑) = ℓ(𝜋𝜑(𝑜𝑡,𝑙,𝑐𝑡),𝑎ˆ𝑡), (7)

where ℓ denotes the denoising objective for diffusion policy and 𝑎ˆ𝑡 is the ground-truth action. Through this posttraining, the action model effectively translates visual planning from compact latent reasoning into low-level robot actions.

#### 3.4. Learning Strategy and Inference

Training Strategy. We initialize both teacher ℱ𝜃𝑇 and student ℱ𝜃 from the same checkpoint obtained through SFT and CoT-SFT on a pre-trained VLM. The teacher is trained with GRPO using action-aligned rewards Huang

et al. (2025), while the student is trained with ℒstudent to compress reasoning into compact latents. We then connect the trained ℱ𝜃 with action model 𝜋𝜑 (initialized from Liu et al. (2024)) by freezing ℱ𝜃 and the state encoder while updating the latent projector and 𝜋𝜑 with ℒIL on large-scale robotic data. For target environment adaptation (e.g., LIBERO Liu et al. (2023), RoboTwin2.0 Chen et al. (2025)), we fine-tune on environment-specific demonstrations.

Inference. The ℱ𝜃 processes (𝑜𝑡,𝑙) by compact latent reasoning, generating visual trajectories via 𝐾 spatial tokens. The visual latent planning 𝑐𝑡, extracted from the spatial tokens’ KV cache, conditions 𝜋𝜑 to predict

actions 𝑎𝑡. Inference requires only ℱ𝜃 and 𝜋𝜑; the verbalizer 𝒱𝜓 is used solely during training and optionally for interpretability.

### 4. Experiment

#### 4.1. Experimental Setup

###### Implementation Details.

We use Qwen2.5-VL 3B Bai et al. (2025) as the VLM backbone. The SFT stage runs for 1 epoch with batch size 64 and learning rate 1e−5, followed by CoT-SFT for 15K iterations with the same hyperparameters. For teacher-student training, both ℱ𝜃𝑇 and ℱ𝜃 are initialized from the CoT-SFT checkpoint and trained for 4,500 iterations with batch size 128 and learning rate 1e−6. The teacher is optimized with GRPO Shao et al. (2024) using action-aligned visual rewards Huang et al. (2025) and QA-style rewards (detailed in supplementary material). For the first 3,000 iterations of the student training, we train the verbalizer 𝒱𝜓 with standard language modeling loss, then switch to ℒverb for the remaining 1,500 iterations. For reasoning-enhanced policy learning, we initialize 𝜋𝜑 from DiT-Policy Chi et al. (2023) pre-trained on OXE O’Neill et al. (2024) for SimplerEnv, and from RDT Liu et al. (2024) for LIBERO and RoboTwin2.0. A linear projection adapts the VLM’s KV cache to the action model dimension (1,024 for DiT-Policy and 2,048 for RDT). Training runs for 20K iterations with batch size 256 and learning rate 1e−4. All diffusion hyperparameters follow those of the respective action models. All experiments are conducted on 16 NVIDIA A100 GPUs with 80 GB memory.

###### Training Datasets and Evaluation Benchmarks.

For reasoning VLM training, we utilize single-arm visual trajectories labeled by Lee et al. (2025) and dual-arm visual trajectories from the AIST dataset Motoda et al. (2025), along with QA tasks from PixMo Deitke et al.

- (2024), RoboFAC Lu et al. (2025), RoboVQA Sermanet et al. (2024), ShareRobot Ji et al. (2025), EgoPlan Chen

- et al. (2023), and Video-R1 Feng et al. (2025). For reasoning-enhanced policy learning, we use action data from the OXE dataset O’Neill et al. (2024) (following OpenVLA Kim et al. (2024)) when training with DiT-Policy, and augment with bimanual data from the static Aloha dataset Shi et al. (2023); Zhao et al. (2023) when training with RDT.

We evaluate Fast-ThinkAct on four embodied reasoning benchmarks and three robot manipulation benchmarks. For embodied reasoning, we use EgoPlan-Bench2 Qiu et al. (2024) (accuracy on multiple-choice questions), RoboVQA Sermanet et al. (2024) (BLEU score Papineni et al. (2002)), OpenEQA Majumdar et al. (2024), and RoboFAC Lu et al. (2025) (both using LLM-based scoring). Notably, RoboVQA and RoboFAC contain videos captured from real robots. For robot manipulation, we evaluate on SimplerEnv Li et al. (2024), which demonstrates strong correlation with real-world performance, LIBERO Liu et al. (2023) covering diverse manipulation tasks including long-horizon scenarios, and RoboTwin2.0 Chen et al. (2025) for complex bimanual manipulation. All robot manipulation tasks use task success rate as the metric. Additional details are provided in the supplementary material.

#### 4.2. Quantitative Evaluation

Robot Manipulation. We evaluate Fast-ThinkAct on robotic manipulation using LIBERO Liu et al. (2023) and SimplerEnv Li et al.

- (2024) benchmarks. LIBERO covers diverse subtasks, including Spatial, Object, Goal, and Long, while SimplerEnv provides a simulated benchmark with strong real-world correlation, featuring variations in lighting, object appearance, and camera viewpoints. As shown in Fig. 3(a)-(e), Fast-ThinkAct consistently outperforms all baselines, achieving the highest success rates across all LIBERO subtasks and SimplerEnv-Google. This includes substantial improvements over foundation VLAs such as OpenVLA Kim et al. (2024), and reasoning VLAs including CoT-VLA Zhao et al. (2025), ThinkAct Huang et al. (2025), and MolmoAct Lee et al. (2025). Moreover, as shown in Fig. 3(f), our compact latent reasoning achieves 89.3% and 88.0% latency reduction

- Table 1: Quantitative evaluation on RoboTwin2.0 Chen et al. (2025). E and H denote easy and hard settings (without/with domain randomization). Background colors indicate task length based on expert demonstrations:

short (80-100) , medium (110-220) , long (270-470) steps.

Model

click alarm

click bell

turn switch

adjust bottle

beat block

handover mic

handover block

hanging mug

stack blocks two

stack bowls three

Average E H E H E H E H E H E H E H E H E H E H E H

DP Chi et al. (2023) 61 5 54 0 36 1 97 0 42 0 53 0 10 0 8 0 7 0 63 0 43.1 0.6 ACT Zhao et al. (2023) 32 4 58 3 5 2 97 23 56 3 85 0 42 0 7 0 25 0 48 0 45.5 3.5

𝜋0 Black et al. (2024) 63 11 44 3 27 23 90 56 43 21 98 13 45 8 11 3 42 1 66 24 52.9 16.3 RDT Liu et al. (2024) 61 12 80 9 35 15 81 75 77 37 90 31 45 14 23 16 21 2 51 17 56.4 22.8 ThinkAct Huang et al. (2025) 64 13 84 11 40 19 94 70 79 33 92 40 56 15 31 18 30 5 54 23 62.4 24.7

Fast-ThinkAct 70 17 82 12 37 21 92 72 82 33 99 42 65 15 30 22 45 5 55 25 65.7 26.4

- Table 2: Quantitative evaluation on EgoPlan-Bench2 Qiu et al. (2024), RoboVQA Sermanet et al. (2024), and OpenEQA Majumdar et al. (2024) benchmarks for embodied reasoning.

Method EgoPlan-Bench2 RoboVQA OpenEQA Overall

Daily. Work. Rec. Hobbies Avg. B-1 B-2 B-3 B-4 B-Avg. Score Avg.

GPT-4V Achiam et al. (2023) 36.7 27.7 33.9 32.5 32.6 32.2 26.5 24.7 23.9 26.8 49.6 36.4 Gemini-2.5-Flash Comanici et al. (2025) 44.2 42.3 43.2 39.1 42.4 39.1 31.6 22.9 22.1 28.9 45.3 38.9

InternVL2.5-2B Chen et al. (2024) 30.9 27.8 28.6 33.1 30.1 36.6 33.7 31.0 29.4 32.7 47.1 36.6 InternVL3-2B Zhu et al. (2025) 36.9 29.9 35.6 31.5 33.4 34.4 33.9 33.5 33.3 33.8 48.8 38.7 NVILA-2B Liu et al. (2024) 34.6 26.7 33.3 31.6 31.4 38.7 34.3 31.1 29.2 33.3 47.0 37.2 Qwen2.5-VL-3B Bai et al. (2025) 29.0 27.0 30.2 28.9 28.5 42.5 36.3 28.7 31.8 34.8 43.4 35.6 Magma-8B Yang et al. (2025) 32.1 25.7 34.4 29.3 29.8 38.6 31.5 28.1 26.7 31.2 49.1 36.7 RoboBrain2.0-3B Team et al. (2025) 45.3 37.6 45.9 39.7 41.8 54.4 47.7 43.1 41.0 46.5 50.1 46.1 ThinkAct-3B Huang et al. (2025) 46.6 41.4 45.9 42.5 44.0 62.4 57.3 52.0 49.6 55.3 48.9 49.4

Fast-ThinkAct-3B 50.3 44.3 46.4 43.2 46.4 70.1 63.0 57.2 53.0 60.8 51.2 52.8

compared to ThinkAct-7B Huang et al. (2025) and MolmoAct-7B Lee et al. (2025) respectively, and 7× faster inference than ThinkAct-3B, demonstrating substantial efficiency gains without sacrificing performance.

To further validate Fast-ThinkAct on more complex scenarios, we evaluate on RoboTwin2.0 Chen et al. (2025), a challenging bimanual manipulation benchmark requiring long-horizon planning. As shown in Tab. 1, FastThinkAct significantly outperforms previous VLAs including DP Chi et al. (2023), ACT Zhao et al. (2023), 𝜋0 Black et al. (2024), RDT Liu et al. (2024), and ThinkAct Huang et al. (2025) across both easy and hard settings. Compared to RDT, Fast-ThinkAct achieves 9.3% and 3.6% higher success rates on easy and hard settings, respectively. Against the reasoning VLA ThinkAct, it improves success by 3.3% and 1.7% while maintaining substantially higher efficiency, as shown in Fig. 3(f). These results demonstrate that our compact reasoning design enables both superior accuracy and computational efficiency on complex bimanual manipulation tasks.

###### Embodied Reasoning.

- In Tab. 2, we evaluate the reasoning capabilities of Fast-ThinkAct in embodied scenarios across three benchmarks: EgoPlan-Bench2 Qiu et al. (2024), RoboVQA Sermanet et al. (2024), and OpenEQA Majumdar et al.

- (2024). These benchmarks assess multi-step planning in egocentric everyday scenarios, long-horizon reasoning for robotic manipulation tasks, and zero-shot understanding of embodied scenes in diverse environments, respectively. We observed that, Fast-ThinkAct surpasses all comparison methods, including two proprietary models (i.e., GPT-4V Achiam et al. (2023) and Gemini-2.5-Flash Comanici et al. (2025)), exceeding the runner-up by 2.4% on EgoPlan-Bench2, 5.5 BLEU score on RoboVQA, and 1.1 points on OpenEQA. These results demonstrate that Fast-ThinkAct effectively handles complex planning sequences and extended reasoning horizons while generalizing to novel environments, showcasing robust capabilities for scene comprehension and multi-step task execution in embodied AI applications.

- 4.3. Analysis of Fast-ThinkAct Reasoning Enables Long-Horizon Planning.

Observation Visualized Trace Execution Result

Instruction

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

"Move 7up can near apple."

- (a)
- (b)
- (c)

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

"Turn on the stove and put the moka pot on it."

"Use the left arm to grab the red block, transfer it to the right hand, and set it

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

onto the blue pad."

- Figure 4: Visualization of predicted visual trajectories and action execution results on long-horizon tasks. Examples from (a) SimplerEnv-Google, (b) LIBERO-Long, and (c) RoboTwin2.0-Hard with long (278) steps. Yellow traces indicate single-arm/left gripper trajectories; red traces indicate right gripper trajectories for bimanual tasks.

[Figure 34]

[Figure 35]

[Figure 36]

To recover from the failure, the robot arm should first reposition itself to ensure proper alignment above the target socket. It must then execute the alignment subtask by carefully adjusting its position to accurately line up the plug with the socket before proceeding to insert the plug. This ensures that the subsequent insertion action is executed with precision, minimizing the risk of further errors.

Q: This is a video of a robotic arm performing a task, an error occurred during execution. Please provide highlevel corrective instructions to help the robot recover and complete the task successfully.

missed

| |Qwen2.5 RoboFAC Fast-Thi|-VL-3B<br>-3B nkAct-3B<br>|
|---|---|---|
| | |80.2<br><br>91.1|

RoboFAC-Sim.

RoboFAC-Real

0 25 50 75 100

29.5

1

16.8

62.5

78.9

[Figure 37]

[Figure 38]

[Figure 39]

First, move the robot arm slightly backward to create distance from the microwave. Then, adjust the arm to the left to align with the spoon. Finally, lower the arm gently to the appropriate height to ensure it is directly above the spoon, ready to grasp it securely.

Q: This is a video of a robotic arm performing a task, an error occurred during execution. Please provide low-level corrective commands to help the robot recover and complete the task successfully.

dropped

- Figure 5: Failure recovery capability on RoboFAC Lu et al. (2025). Left: Qualitative examples (from both simulation and real robot) of corrective guidance for manipulation errors. Right: Quantitative evaluation on simulation (RoboFAC-Sim) and real-robot (RoboFAC-Real) settings.

[Figure 40]

We analyze Fast-ThinkAct’s capability on long-horizon tasks in Tab. 1 and Fig. 4. We focus on longhorizon tasks (average length exceeding 270 steps) in RoboTwin2.0 Chen et al. (2025) that require multistep reasoning and extended planning horizons. As shown in Tab. 1, Fast-ThinkAct achieves average scores of 48.8 and 16.8 on easy and hard settings of longhorizon tasks, respectively, surpassing RDT (35.0/12.3) and ThinkAct (42.8/15.3). Fig. 4 visualizes predicted 2D visual traces and execution results on representative tasks from SimplerEnv-Google Li et al. (2024), LIBEROLong Liu et al. (2023), and RoboTwin2.0 Chen et al.

56.5

51.7

47.8

36.6

33.2

33.2

31.7

16.5

20.1

16.8

15.6 14.8

7.0

Figure 6: Few-shot adaptation results on RoboTwin2.0 benchmark. We use 10 demonstrations per task for fine-tuning.

- (2025). For example, the LIBERO-Long task requires sequentially turning on the stove and placing a moka pot on it, while the RoboTwin2.0 handover task requires bimanual coordination to transfer a block between grippers. The visual traces successfully predict feasible solution paths, with their corresponding representations serving as visual planning guidance for successful execution. These results demonstrate that our compact latent reasoning effectively supports long-horizon

planning in complex manipulation scenarios. Reasoning Enables Failure Recovery.

A key advantage of reasoning-based VLAs Huang et al. (2025); Abdolmaleki et al. (2025) is their ability to identify runtime failures and provide corrective guidance for recovery. To evaluate this capability, we conduct experiments on RoboFAC Lu et al.

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

Question: Pick up the glass. Is it satisfied? GT: yes

- (2025), a benchmark specifically designed to assess failure identification and correction in embodied VLMs. As shown in Fig. 5, Fast-ThinkAct substantially outperforms the second-best baseline RoboFAC3B Lu et al. (2025) by 10.9 points on the simulation split and 16.4 points on the real-world split. The qualitative examples demonstrate Fast-ThinkAct’s ability to reason over manipulation videos, identify failures, and propose recovery steps. For instance, in the right example where the target object drops mid-execution, Fast-ThinkAct generates a concrete recovery plan: first moving the arm backward to create space, then adjusting laterally to align with the target object, and finally lowering to the appropriate height for a secure grasp. These results demonstrate that our latent reasoning supports both fast task execution and crucial failure analysis capabilities essential for robust robotic manipulation. Reasoning Enables Few-Shot Adaptation.

Let me think about this. I can see a cluttered table with several objects scattered around. The robot arm moves slowly toward the glass, adjusting its position to avoid bumping into nearby items. The gripper closes gently around the glass and lifts it from the table with a steady motion. The glass is held securely and remains stable in the air. Everything looks correct, so the action is satisfied.

(a) Textual reasoning (from teacher model)

Let me think. The robot arm moves carefully over the table, aligns with the glass, and grips it. The glass lifts smoothly without tipping or slipping. It looks like the task is successful and satisfied.

(b) Verbalized reasoning

Figure 7: Reasoning trace comparison on RoboVQA. (a) Teacher’s textual reasoning. (b) Student’s verbalized latent reasoning. Green: relevant content; orange: less relevant content.

To assess how reasoning capability improves few-shot adaptation, we conduct few-shot experiments on the RoboTwin2.0 benchmark Chen et al. (2025), fine-tuning models using only 10 demonstrations per task. As illustrated in Fig. 6, Fast-ThinkAct significantly enhances our adopted action model RDT Liu

Table 3: Ablation study of training objectives and learning stages. Note that Fast-ThinkAct w/o ℒverb,ℒdistill denotes the student VLM ℱ𝜃 trained without the corresponding loss components.

Method EgoPlan RoboVQA OpenEQA Average Fast-ThinkAct 46.4 60.8 51.2 52.8

- et al. (2024) and outperforms the state-of-the-

w/o ℒverb 42.1 53.8 49.5 48.5 w/o ℒverb,ℒdistill 41.6 52.7 48.9 47.7

art VLAs, including 𝜋0 Black et al. (2024) and ThinkAct Huang et al. (2025) on both medium and long-horizon tasks. Notably, our method achieves these gains while operating with significantly lower reasoning latency compared to ThinkAct, highlighting the advantage of efficient yet effective reasoning for few-shot action adaptation in complex robot manipulation scenarios.

Textual Teacher ℱ𝜃𝑇 41.7 58.2 49.4 49.8 SFT + CoT-SFT 40.0 46.1 48.8 45.0 SFT only 40.5 53.6 45.3 46.5

###### Visualization of Verbalizable Latent Reasoning.

- In Fig. 7, we compare the teacher’s textual reasoning with the student’s verbalized latent reasoning on RoboVQA. While both capture task-relevant information (green), the teacher generates verbose outputs with less directly relevant content (orange), whereas our student produces more concise and focused responses when verbalized. This demonstrates that our preference-guided distillation not only reduces computational cost but also distills concise reasoning patterns while filtering out redundant information.

###### Ablation Study.

- In Tab. 3, we ablate training stages and loss components. Starting from the full Fast-ThinkAct, removing ℒverb causes performance drops as latent CoTs lack preference-based guidance to align with high-quality reasoning and suppress low-quality patterns. Further removing ℒdistill leads to additional decline, indicating that aligning trajectory-level representations is crucial for transferring visual planning capabilities. Comparing training strategies, CoT-SFT underperforms SFT on EgoPlan-Bench2 and RoboVQA but improves on OpenEQA, suggesting naïve chain-of-thought supervision benefits open-ended QA but introduces verbosity that hinders structured reasoning tasks. Our preference-guided approach distills high-quality reasoning while maintaining efficiency. This validates the necessity of our proposed distillation framework and visual trajectory alignment. We provide additional ablation studies in the supplementary material.

### 5. Conclusion

We presented Fast-ThinkAct, an efficient reasoning framework for vision-language-action tasks that achieves compact yet expressive planning through verbalizable latent reasoning. By distilling lengthy textual reasoning into compact latent representations via preference-guided distillation and visual trajectory alignment, our approach bridges high-level embodied reasoning with low-level action execution through reasoning-enhanced policy learning. Extensive experiments across diverse robotic manipulation and embodied reasoning benchmarks demonstrate that Fast-ThinkAct achieves strong performance with significantly reduced inference latency while enabling effective long-horizon planning, few-shot adaptation, and failure recovery capabilities.

Limitations and Future Works. As our verbalizer 𝒱𝜓 is built upon a pre-trained LLM, it inevitably inherits language model limitations, including hallucination, occasionally producing plausible but inaccurate descriptions. However, this does not affect action execution during inference, as the verbalizer serves only for interpretability while action prediction uses the grounded latent representations from visual plan distillation. To further improve the faithfulness of verbalized reasoning, we can consider incorporating grounding-aware objectives or hallucination suppression techniques in future work.

### A. Additional Experimental Setup

- A.1. Algorithm Algorithm 1: Training Fast-ThinkAct (Sec. 3.2)

Input: CoT-SFT checkpoint ℱ𝜃0

, training data 𝒟, rollout size 𝑁, latent reasoning steps 𝑀, number of

waypoints 𝐾, total iterations 𝑇total Output: Trained student model ℱ𝜃 // Initialize models

ℱ𝜃𝑇 ← ℱ𝜃0

, ℱ𝜃 ← ℱ𝜃0

; Initialize verbalizer 𝒱𝜓 from pre-trained LLM; 𝑡 ← 0; while 𝑡 < 𝑇total do

Sample batch (𝑜,𝑙,𝑝ˆ) from 𝒟; // Suppose bs=1 for simplicity // Teacher GRPO training

Generate 𝑁 rollouts {𝜏𝑖}𝑁𝑖=1 from ℱ𝜃𝑇(𝑜,𝑙); Compute trajectory rewards {𝑟𝑖}𝑁𝑖=1; Compute group-wise advantages {𝐴𝑖}𝑁𝑖=1; Update ℱ𝜃𝑇 with 𝒥GRPO (Eq. 1);

𝜏+ ← arg max𝑖 𝐴𝑖, 𝜏− ← arg min𝑖 𝐴𝑖 (Eq. 3) ; // For student distillation ℎ𝑇𝑡 ← hidden state of 𝜏+ at <answer> token from ℱ𝜃𝑇 ; // For distillation loss // Student latent distillation

z = {𝑧𝑚}𝑀𝑚=1 ← ℱ𝜃(𝑜,𝑙) ; // Perform auto-regressive latent reasoning Compute ℒverb with z, 𝒱𝜓, 𝜏+, 𝜏− (Eq. 4); Forward 𝐾 spatial tokens from ℱ𝜃(𝑜,𝑙,z) to obtain ℎ𝑡 and {ℎ′(s𝑖)}𝐾𝑖=1; Compute ℒdistill with ℎ𝑇𝑡 , ℎ𝑡 (Eq. 5); Compute ℒans with {ℎ′(s𝑖)}𝐾𝑖=1, 𝑝ˆ (Eq. 6); Update ℱ𝜃 with ℒstudent = ℒverb + ℒdistill + ℒans; 𝑡 ← 𝑡 + 1;

end return ℱ𝜃;

Algorithm 1 presents the complete training procedure corresponding to Sec. 3.2. It shows how we jointly optimize the teacher model with GRPO and distill its reasoning into the student’s compact latent representations.

- A.2. Implementation Details Our implementation follows the setup described in Sec. 4.1 of the main paper. Here we provide additional

details. The verbalizer 𝒱𝜓 is initialized from a small LLM, Qwen3-0.6B, with cross-attention layers inserted at each layer to condition on latent CoTs z. For the student model training, in the first 3,000 iterations, we replace verbalization loss ℒverb with language modeling loss using 𝜏+ as ground truth to warm up 𝒱𝜓’s alignment with the latent representations z. We then freeze 𝒱𝜓 and use the ℒverb for the remaining 1,500 iterations. The student ℱ𝜃 is optimized throughout both phases. For waypoint prediction in Eq. 6, each 𝑝𝑖 ∈ R6 encodes coordinates in the format [𝑥single,𝑦single,𝑥left,𝑦left,𝑥right,𝑦right], where the first two dimensions are for single-arm and the last four are for bimanual robot. For ground-truth 𝑝𝑖ˆ , we fill the corresponding dimensions based on robot type and mask out the unused dimensions when computing ℒans. For GRPO training, we follow the configuration of ThinkAct Huang et al. (2025), using rollout size 𝑁 = 5. Following Lee et al. (2025), we set the number of waypoints in trajectory to 𝐾 = 5. We use 𝑀 = 6 latent reasoning tokens, with ablation study provided in Fig. 8.

During reasoning-enhanced policy learning, for SimplerEnv Li et al. (2024) evaluation, to ensure fair comparison with previous works Kim et al. (2024); Lee et al. (2025), we initialize 𝜋𝜑 from DiT-Policy Chi et al. (2023) pre-trained on the same OXE dataset O’Neill et al. (2024); Kim et al. (2024) and conduct reasoning-enhanced policy learning (Sec. 3.3) using the same OXE data. For LIBERO Liu et al. (2023) and RoboTwin2.0 Chen et al.

- (2025) evaluations, we initialize 𝜋𝜑 from RDT Liu et al. (2024), which has demonstrated strong performance on RoboTwin2.0, and conduct policy learning using OXE O’Neill et al. (2024) and static ALOHA datasets Shi et al.

- (2023); Zhao et al. (2023). Our method further enhances RDT’s manipulation capabilities on both benchmarks. The use of different action models also demonstrates that our approach is agnostic to the underlying action model choice.

A.3. Training Data Details

- A.3.1. Dataset Sources 2D Visual Trace of Manipulation Tasks. For single-arm manipulation, we utilize 2D visual trajectories labeled by MolmoAct Lee et al. (2025) from the Open X-Embodiment (OXE) dataset O’Neill et al. (2024), comprising approximately 1.3M trajectories. For bimanual manipulation, we extract dual-arm visual trajectories from the AIST dataset Motoda et al.

(2025), resulting in approximately 92K trajectory samples. Specifically, we first use Molmo-72B Deitke et al.

- (2024) to detect left and right gripper positions (following Lee et al. (2025)) in the first frame, then apply CoTracker3 Karaev et al. (2025) to track and parse the manipulation trajectories throughout the video sequences.

###### RoboFAC Lu et al. (2025).

RoboFAC is a robotic failure analysis dataset containing 9,440 erroneous manipulation trajectories across 16 tasks in both simulated and real-world environments. We utilize the training set with 64K QA pairs covering various failure types for developing failure identification and correction planning capabilities.

###### RoboVQA Sermanet et al. (2024).

RoboVQA contains robot manipulation videos with QA tasks covering task understanding. The dataset includes approximately 5K long-horizon and 92K medium-horizon video sequences from diverse robotic platforms, resulting in total 798K QA pairs. Videos are annotated with multiple questions probing spatial reasoning, action prediction, and task comprehension.

###### ShareRobot Ji et al. (2025).

ShareRobot is a large-scale dataset collected by RoboBrain Ji et al. (2025), containing over 1M QA pairs covering task planning, object affordances, and manipulation strategies across diverse robot embodiments and scenes. The dataset features fine-grained annotations linking task descriptions to frame-level execution details, facilitating learning of transferable manipulation knowledge.

###### EgoPlan-Bench Chen et al. (2023).

EgoPlan-Bench features egocentric videos of daily activities annotated with task planning information including goals, execution history, and current states. The dataset contains approximately 53K video-text pairs for training long-horizon planning and progress tracking capabilities from egocentric view.

###### Video-R1-CoT Feng et al. (2025).

Video-R1 comprises 165K video question-answer pairs with chain-of-thought reasoning annotations generated by large-scale vision-language models. The dataset covers diverse reasoning domains including mathematical logic, spatial understanding, OCR, and visual analytics. All samples are quality-filtered to ensure annotation consistency and correctness.

###### PixMo Deitke et al. (2024).

PixMo is a general-purpose vision-language dataset with diverse image captions and question-answer pairs. Following MolmoAct Lee et al. (2025), we incorporate PixMo dataset to preserve general visual understanding

###### Table 4: Quantitative results with larger model size (7B or 8B) on embodied reasoning benchmarks.

Method EgoPlan-Bench2 RoboVQA OpenEQA Overall

Daily. Work. Rec. Hobbies Avg. B-1 B-2 B-3 B-4 B-Avg. Score Avg.

InternVL2.5-8B Chen et al. (2024) 36.2 28.7 34.4 35.4 33.5 40.5 33.3 29.6 27.5 32.7 54.4 40.2 InternVL3-8B Zhu et al. (2025) 38.5 32.9 36.1 37.2 36.2 44.3 36.5 31.6 28.9 35.3 55.5 42.3 NVILA-8B Liu et al. (2024) 35.8 28.7 37.2 35.4 33.7 42.7 39.7 37.6 36.1 39.0 54.0 42.2 Qwen2.5-VL-7B Bai et al. (2025) 31.4 26.7 29.5 28.6 29.1 47.8 41.2 36.2 33.7 39.7 50.8 39.9 Magma-8B Yang et al. (2025) 32.1 25.7 34.4 29.3 29.8 38.6 31.5 28.1 26.7 31.2 49.1 36.7 RoboBrain2.0-7B Team et al. (2025) 39.4 27.0 33.9 32.2 33.2 44.9 38.2 34.7 33.5 37.8 51.1 40.7 ThinkAct-7B Huang et al. (2025) 50.1 49.8 44.8 45.2 48.2 69.1 61.8 56.0 52.4 59.8 56.2 54.7

Fast-ThinkAct-7B 51.3 47.3 41.5 45.9 47.5 70.4 63.3 57.3 53.2 61.1 59.0 55.9

###### Table 5: Results on LIBERO and SimplerEnv benchmarks with additional ThinkAct-3B comparison.

###### Method LIBERO SimplerEnv-Google Latency (↓)

OpenVLA-7B Kim et al. (2024) 76.5 40.2 N/A CoT-VLA-7B Zhao et al. (2025) 83.9 N/A N/A ThinkAct-7B Huang et al. (2025) 84.4 68.3 7513 MolmoAct-7B Lee et al. (2025) 86.8 64.9 6723

ThinkAct-3B Huang et al. (2025) 83.1 64.7 5674 Fast-ThinkAct-3B 89.7 68.7 805 (↓7.0×)

and prevent catastrophic forgetting when training on embodied dataset. Specifically, we use approximately 726K samples from the ask_model_anything, cap, and cap-qa splits.

- A.3.2. Data Processing and Formatting Supervised Fine-Tuning (SFT). To enhance foundational embodied knowledge, we perform supervised fine-tuning on approximately 4M samples combining 2D visual trajectories from MolmoAct Lee et al. (2025) and AIST Motoda et al. (2025), along with QA data from PixMo Deitke et al. (2024), RoboFAC Lu et al. (2025), RoboVQA Sermanet et al.

(2024), ShareRobot Ji et al. (2025), and EgoPlan Chen et al. (2023). This stage enables the model to acquire basic visual understanding, task comprehension, and manipulation knowledge across diverse embodiments and scenarios.

###### Chain-of-Thought SFT (CoT-SFT).

To develop reasoning capabilities while preserving embodied understanding, we sample 5% from the SFT data (approximately 200K samples) and augment with 165K samples from Video-R1-CoT Feng et al. (2025). For data with CoT annotations, we format prompts to elicit structured reasoning enclosed in <think> tags followed by answers in <answer> tags; for data without CoT annotations, we prompt for direct answers only. This enables the model to learn reasoning capabilities from CoT-annotated data and generalize them to embodied tasks.

###### Teacher-Student Training.

Building upon the CoT-SFT checkpoint, we curate a balanced training set by sampling approximately 5,000 instances from each dataset and data type, totaling nearly 50K samples. We adopt the prompt formatting strategy from CoT-SFT for both teacher GRPO training and student latent distillation. We train both the teacher with GRPO and the student with latent distillation (as detailed in Sec. 3.2) on this data, efficiently transferring high-quality reasoning patterns into compact latent representations.

###### Table 6: Comparison with efficient textual reasoning methods.

EgoPlanBench2

Method

RoboVQA OpenEQA Average

Textual Teacher ℱ𝜃𝑇 41.7 58.2 49.4 49.8 ℱ𝜃𝑇 Inference w/o thinking 42.7 55.0 41.7 46.5 ℱ𝜃𝑇 Inference w/ 6 textual tokens 39.3 53.0 46.5 46.3 ℱ𝜃𝑇 w/ RL Length-Penalty Arora and Zanette (2025) 41.2 57.5 44.7 47.8 Fast-ThinkAct-3B 46.4 60.8 52.8 53.3

Observation Instruction Visualized Trace Execution Result

"Lift the medium mug for hot drinks from the table, spin it, set it down, and hang it on the dark gray rack

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

- (a)

"Grab the handheld microphone from the table and pass

it over."

- (b)

with flat back."

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

- Figure 8: Visualization of predicted visual trajectories and action execution results on RoboTwin2.0. Yellow traces indicate left gripper trajectories; red traces indicate right gripper trajectories for bimanual tasks.

#### A.4. Evaluation Setup

- A.4.1. Embodied Reasoning Benchmarks We evaluate on three benchmarks assessing different aspects of embodied reasoning. EgoPlan-Bench2 Chen et al. (2023) tests egocentric task planning across 24 daily-life scenarios with 1,321 multiple-choice questions, measuring accuracy in predicting next steps given task goals and progress history. RoboVQA Sermanet et al.

(2024) evaluates visual reasoning in manipulation contexts through 1,893 free-form QA pairs from robot and human demonstrations, assessed via BLEU score. OpenEQA Majumdar et al. (2024) assesses spatial and functional understanding through 1,600+ questions spanning 180+ real-world environments, evaluated using LLM-based scoring aligned with human preferences. These benchmarks comprehensively evaluate embodied reasoning capability across planning, manipulation, and spatial understanding.

- A.4.2. Robotic Manipulation Benchmarks We evaluate on three simulation benchmarks covering diverse manipulation scenarios. SimplerEnv Li et al.

(2024) provides manipulation tasks with strong sim-to-real correlation, featuring diverse visual variations in lighting, textures, backgrounds, and camera poses. Following MolmoAct Lee et al. (2025), we evaluate on the Google Robot tasks using the standard protocol Kim et al. (2024); Lee et al. (2025) of directly evaluating on SimplerEnv after training on OXE. LIBERO Liu et al. (2023) targets different generalization challenges through four task suites: spatial layout variation (LIBERO-Spatial), object diversity (LIBERO-Object), goal variation (LIBERO-Goal), and long-horizon planning with mixed variations (LIBERO-Long). We evaluate each suite over 500 trials using 3 random seeds following prior works Kim et al. (2024); Lee et al. (2025). RoboTwin2.0 Chen

- et al. (2025) features challenging bimanual manipulation with easy and hard difficulty settings, where the hard setting introduces domain randomization including clutter, lighting variations, diverse textures, and height changes. Following the original protocol, we train on 50 clean expert demonstrations per task and evaluate with 100 rollouts under both settings. We assess 10 tasks categorized into short, medium, and long horizons based on demonstration lengths.

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

missed

wrong position

Q: From the video of the robotic arm carrying out a task, identify the phase of the task where the error happened. Options: ['Align the card with the slot.', 'Move the long block to the red bowl.', 'Move the ball to the target position.', 'Grasp the dice.', 'Reach for the cube.']

Q: From the video of the robotic arm performing a task, what kind of error can be observed during the task? Options: ['Orientation deviation.', 'Step omission.', 'Wrong target object.', 'Timing error.', 'Grasping error.', 'Position deviation.']

A: Position deviation.

A: Reach for the cube

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

unstable grasp

missed

Q: Based on the video, provide a detailed explanation of why the robotic arm failed to complete the task.

Q: From the video of the robotic arm executing a task, give a detailed explanation of the reason behind the task failure.

The task failed because the robot arm did not properly align itself directly above the L-shaped tool during the reaching phase. This misalignment resulted in a position deviation that prevented the robot from successfully grasping the tool, which is critical for the subsequent steps of the task. Without the proper alignment, the robot risks grasping the tool in an unstable manner, leading to the potential dropping of the tool and failure to complete the task of pulling the cube back.

The task failed because the robot arm did not reach the correct position above the red cube during the reach stage. This position deviation led to a misalignment of the gripper with the red cube, causing the grasping attempt to miss the target object entirely. The robot's end-effector was not accurately aligned, which resulted in the failure of the subsequent grasp and lift operations.

- Figure 9: Failure identification and analysis capabilities on RoboFAC Lu et al. (2025). Top row shows identification of failure types and execution stages. Bottom row demonstrates failure root cause analysis.

### B. Additional Experiment Results

#### B.1. Additional Quantitative Results

###### Results of Larger Model Size.

To demonstrate the scalability of our approach, we apply Fast-ThinkAct to a larger backbone, Qwen2.5-VL-7B, and evaluate its performance on embodied reasoning benchmarks. As shown in Tab. 4, Fast-ThinkAct consistently achieves strong performance across EgoPlan-Bench2 Qiu et al. (2024), RoboVQA Sermanet et al. (2024), and OpenEQA Majumdar et al. (2024), validating that our latent reasoning distillation method effectively scales to larger model backbones.

###### Performance Comparison with ThinkAct-3B.

Tab. 5 presents detailed numerical results corresponding to Fig. 3 with additional ThinkAct-3B results. At the same 3B model size, Fast-ThinkAct achieves notable performance gains (89.7 vs. 83.1 on LIBERO, 68.7 vs. 64.7 on SimplerEnv-Google) while dramatically improving efficiency with 7× faster inference (805ms vs. 5674ms).

###### Comparison with Efficient Reasoning Baselines.

- Table 6 compares our method with efficient textual reasoning alternatives applied to the textual teacher ℱ𝜃𝑇. We evaluate three baselines: removing reasoning during inference entirely (0 tokens), constraining the teacher to generate only 6 textual tokens during inference, and applying RL training with a length penalty Arora and Zanette (2025) to encourage concise reasoning (∼50 tokens). These achieve 46.5, 46.3, and 47.8 respectively, all degrading from the teacher’s 49.8. In contrast, Fast-ThinkAct uses only 6 latent tokens and achieves 53.3, demonstrating superior efficiency and performance.

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

Question: Is this room on the ground floor of the building? GT: no

Let's carefully analyze the image to determine if this room is on the ground floor of the building.

First, I notice that there is a window with a view of greenery outside. This suggests we are at a level that allows natural light to filter in, which is typical for ground floor windows.

Next, I see a white desk and a chair, which seem like standard office furniture for a ground floor workspace. The desk appears to be at a height that would be comfortable for someone to sit and work, which is also consistent with a ground floor layout.

Additionally, there's a white bookshelf that looks like it could be used for storage and decoration, which is a common feature in office spaces on the ground floor.

However, there is a cabinet in the background that seems a bit unusual for a ground floor room. It could be a storage cabinet for a specific purpose, but it might not be typical for a ground floor space. Let's also consider the lighting. The natural light coming through the

window seems to be coming from the side, which is more likely if we are on the ground floor and looking towards a window that is part of the building's exterior.

After considering all these factors, I think the room is likely on the ground floor, but it's not impossible for it to be on an upper floor with a unique window design or a specialized storage cabinet. Final thought: Given the typical layout of office spaces and the visible elements, it's reasonable to conclude that this room is on the ground floor.

(a) Textual reasoning (from teacher model)

Let's break this down. The image shows a study room scene with a desk, a chair, and some shelves. There's no clear indication that this is on the ground floor. It's obviously indoors, but I should check if anything suggests a different level. Wait, I should also consider if there's any sign of elevation. Hmm, it seems like the view outside the window shows that the room is positioned above the ground. Oh, I see, the perspective through the window suggests it’s not on the ground floor. Let's confirm that. Yes, it checks out.

(b) Verbalized reasoning

- Figure 10: Reasoning trace comparison on OpenEQA. (a) Teacher’s textual reasoning. (b) Student’s verbalized latent reasoning. Green: reasonable reasoning trace; red: incorrect trace.

#### B.2. Additional Qualitative Results

###### Qualitative Robot Execution.

We provide qualitative robot execution comparisons between the base action model RDT Liu et al. (2024) and Fast-ThinkAct in the supplementary video Fast-ThinkAct.mp4. Our method shows substantial improvements on challenging robotic execution tasks, where reasoning capabilities provide better spatial understanding and coordination for successful manipulation.

###### Bimanual Manipulation Results.

- In Fig. 8, we present visualized trajectories and execution results for hanging mug and handover mic tasks under easy and hard settings in RoboTwin2.0 Chen et al. (2025). The hard setting includes different backgrounds and distractor objects. These examples show successful bimanual coordination where predicted waypoints accurately guide both grippers through the manipulation sequence, demonstrating Fast-ThinkAct’s spatial reasoning ability across varied visual conditions.

###### Table 7: Additional ablation study of training objectives and learning stages on robot manipulation benchmarks.

SimplerEnv Google

Method LIBERO

RoboTwin2.0 Average

Fast-ThinkAct 89.7 68.7 46.1 68.2 w/o ℒverb 88.6 67.3 44.9 66.9 w/o ℒverb,ℒdistill 86.3 65.7 42.6 64.9 Textual Teacher 88.5 67.3 45.8 67.2 SFT + CoT-SFT 87.2 65.8 43.3 65.4 SFT only 86.9 64.5 42.8 64.7

LIBERO-Average

- 86
- 87
- 88
- 89
- 90

SuccessRate(%)

M=1 M=6 (Ours) M=30 M=100

Table 8: Ablation of Latent Reasoning Steps 𝑀.

Failure Identification and Recovery.

- In Fig. 9, we demonstrate Fast-ThinkAct’s failure identification and analysis capabilities, complementing the recovery planning shown in the main paper. The top row shows that Fast-ThinkAct identifies failure types (e.g., position deviation) and execution stages (e.g., reaching for the cube). The bottom row illustrates root cause analysis, for instance, in the bottom-right example, the model correctly infers that the failure to push the cube with an L-shaped tool stems from an improper initial grasp. These results demonstrate Fast-ThinkAct’s comprehensive understanding of manipulation failures beyond recovery planning.

###### Verbalized Latent Reasoning.

Fig. 10 visualizes teacher textual reasoning and student verbalized reasoning. While the student generates compact and correct (green) reasoning, the teacher’s lengthy output sometimes contains erroneous steps (red) that might degrade the performance.

#### B.3. Additional Ablation Study and Analysis

###### Additional Ablation Results on Manipulation Benchmarks.

- Table 7 shows ablation results on LIBERO Liu et al. (2023), SimplerEnv-Google Li et al. (2024), and RoboTwin2.0 Chen

et al. (2025). Removing ℒverb or ℒdistill progressively degrades performance, confirming their contributions. Our full model consistently outperforms the textual teacher and models without teacher-student training (CoT-SFT, SFT only), demonstrating the benefits of compact latent reasoning distillation.

###### Ablation Study on Action Model Conditioning.

In Sec. 3.3, we extract visual latent planning 𝑐𝑡 from early-layer KV cache of spatial tokens to condition the action model. We compare this against using late-layer KV cache (last 𝑁 layers, where 𝑁 is the action model depth) and directly using spatial tokens’ output hidden states. Our approach achieves 89.7 on LIBERO, outperforming late-layer KV at 88.3 and output hidden states at 87.1, demonstrating that early-layer representations better capture visual planning information for action prediction. Therefore, we adopt early-layer KV conditioning as our default configuration.

###### Ablation Study on Latent Reasoning Steps.

In Fig. 8, we study the effect of latent reasoning steps 𝑀. We observe that too few steps (𝑀 = 1) limit reasoning capacity, while excessive steps (𝑀 = 30,100) might introduce redundant or noisy information. Therefore, we adopt 𝑀 = 6, which achieves optimal performance, as our default.

### References

- [1] Abbas Abdolmaleki, Saminda Abeyruwan, Joshua Ainslie, Jean-Baptiste Alayrac, Montserrat Gonzalez Arenas, Ashwin Balakrishna, Nathan Batchelor, Alex Bewley, Jeff Bingham, Michael Bloesch, et al. Gemini robotics 1.5: Pushing the frontier of generalist robots with advanced embodied reasoning, thinking, and motion transfer. arXiv preprint arXiv:2510.03342, 2025. 3, 10
- [2] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023. 8
- [3] Pranjal Aggarwal and Sean Welleck. L1: Controlling how long a reasoning model thinks with reinforcement learning. arXiv preprint arXiv:2503.04697, 2025. 3
- [4] Daman Arora and Andrea Zanette. Training language models to reason efficiently. arXiv preprint arXiv:2502.04463, 2025. 15, 16
- [5] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025. 1, 7, 8, 14
- [6] Johan Bjorck, Fernando Castañeda, Nikita Cherniadev, Xingye Da, Runyu Ding, Linxi Fan, Yu Fang, Dieter Fox, Fengyuan Hu, Spencer Huang, et al. Gr00t n1: An open foundation model for generalist humanoid robots. arXiv preprint arXiv:2503.14734, 2025. 1, 3
- [7] Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy

Groom, Karol Hausman, Brian Ichter, et al. 𝜋0: A vision-language-action flow model for general robot control. arXiv preprint arXiv:2410.24164, 2024. 1, 3, 8, 10

- [8] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Joseph Dabis, Chelsea Finn, Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, Jasmine Hsu, et al. Rt-1: Robotics transformer for real-world control at scale. arXiv preprint arXiv:2212.06817, 2022. 1, 3
- [9] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromanski, Tianli Ding, Danny Driess, Avinava Dubey, Chelsea Finn, et al. Rt-2: Vision-language-action models transfer web knowledge to robotic control. arXiv preprint arXiv:2307.15818, 2023. 1, 3
- [10] Qingwen Bu, Jisong Cai, Li Chen, Xiuqi Cui, Yan Ding, Siyuan Feng, Shenyuan Gao, Xindong He, Xuan Hu, Xu Huang, et al. Agibot world colosseo: A large-scale manipulation platform for scalable and intelligent embodied systems. arXiv preprint arXiv:2503.06669, 2025. 3
- [11] Guo Chen, Zhiqi Li, Shihao Wang, Jindong Jiang, Yicheng Liu, Lidong Lu, De-An Huang, Wonmin Byeon, Matthieu Le, Tuomas Rintamaki, et al. Eagle 2.5: Boosting long-context post-training for frontier vision-language models. arXiv preprint arXiv:2504.15271, 2025. 1
- [12] Tianxing Chen, Zanxin Chen, Baijun Chen, Zijian Cai, Yibin Liu, Zixuan Li, Qiwei Liang, Xianliang Lin, Yiheng Ge, Zhenyu Gu, et al. Robotwin 2.0: A scalable data generator and benchmark with strong domain randomization for robust bimanual robotic manipulation. arXiv preprint arXiv:2506.18088, 2025. 6, 7, 8, 9, 10, 13, 15, 17, 18
- [13] William Chen, Suneel Belkhale, Suvir Mirchandani, Oier Mees, Danny Driess, Karl Pertsch, and Sergey Levine. Training strategies for efficient embodied reasoning. arXiv preprint arXiv:2505.08243, 2025. 2, 3
- [14] Yi Chen, Yuying Ge, Yixiao Ge, Mingyu Ding, Bohao Li, Rui Wang, Ruifeng Xu, Ying Shan, and Xihui Liu. Egoplan-bench: Benchmarking multimodal large language models for human-level planning. arXiv preprint arXiv:2312.06722, 2023. 7, 13, 14, 15

- [15] Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, et al. Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling. arXiv preprint arXiv:2412.05271, 2024. 8, 14
- [16] Jeffrey Cheng and Benjamin Van Durme. Compressed chain of thought: Efficient reasoning through dense representations. arXiv preprint arXiv:2412.13171, 2024. 3
- [17] Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, and Shuran Song. Diffusion policy: Visuomotor policy learning via action diffusion. The International Journal of Robotics Research, page 02783649241273668, 2023. 7, 8, 13
- [18] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025. 1, 8
- [19] Muzhi Dai, Shixuan Liu, and Qingyi Si. Stable reinforcement learning for efficient reasoning. arXiv preprint arXiv:2505.18086, 2025. 3
- [20] Matt Deitke, Christopher Clark, Sangho Lee, Rohun Tripathi, Yue Yang, Jae Sung Park, Mohammadreza Salehi, Niklas Muennighoff, Kyle Lo, Luca Soldaini, et al. Molmo and pixmo: Open weights and open data for state-of-the-art multimodal models. arXiv e-prints, pages arXiv–2409, 2024. 7, 13, 14
- [21] Danny Driess, Jost Tobias Springenberg, Brian Ichter, Lili Yu, Adrian Li-Bell, Karl Pertsch, Allen Z Ren, Homer Walke, Quan Vuong, Lucy Xiaoyang Shi, et al. Knowledge insulating vision-language-action models: Train fast, run fast, generalize better. arXiv preprint arXiv:2505.23705, 2025. 3
- [22] Kaituo Feng, Kaixiong Gong, Bohao Li, Zonghao Guo, Yibing Wang, Tianshuo Peng, Junfei Wu, Xiaoying Zhang, Benyou Wang, and Xiangyu Yue. Video-r1: Reinforcing video reasoning in mllms. arXiv preprint arXiv:2503.21776, 2025. 7, 13, 14
- [23] Weifan Guan, Qinghao Hu, Aosheng Li, and Jian Cheng. Efficient vision-language-action models for embodied manipulation: A systematic survey. arXiv preprint arXiv:2510.17111, 2025. 2
- [24] Shibo Hao, Sainbayar Sukhbaatar, DiJia Su, Xian Li, Zhiting Hu, Jason Weston, and Yuandong Tian. Training large language models to reason in a continuous latent space. arXiv preprint arXiv:2412.06769,

- 2024. 3

[25] Chi-Pin Huang, Yueh-Hua Wu, Min-Hung Chen, Yu-Chiang Frank Wang, and Fu-En Yang. Thinkact: Vision-language-action reasoning via reinforced visual latent planning. arXiv preprint arXiv:2507.16815,

- 2025. 1, 2, 3, 5, 6, 7, 8, 10, 12, 14

- [26] Yuheng Ji, Huajie Tan, Jiayu Shi, Xiaoshuai Hao, Yuan Zhang, Hengyuan Zhang, Pengwei Wang, Mengdi Zhao, Yao Mu, Pengju An, et al. Robobrain: A unified brain model for robotic manipulation from abstract to concrete. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 1724–1734,

2025. 7, 13, 14

- [27] Nikita Karaev, Yuri Makarov, Jianyuan Wang, Natalia Neverova, Andrea Vedaldi, and Christian Rupprecht. Cotracker3: Simpler and better point tracking by pseudo-labelling real videos. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 6013–6022, 2025. 13
- [28] Dongyoung Kim, Sumin Park, Huiwon Jang, Jinwoo Shin, Jaehyung Kim, and Younggyo Seo. Robot-r1: Reinforcement learning for enhanced embodied reasoning in robotics. arXiv preprint arXiv:2506.00070,

2025. 2, 3

- [29] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, Quan Vuong, Thomas Kollar, Benjamin Burchfiel, Russ Tedrake, Dorsa Sadigh, Sergey Levine, Percy Liang, and Chelsea Finn. Openvla: An open-source vision-language-action model. arXiv preprint arXiv:2406.09246, 2024. 1, 3, 7, 13, 14, 15
- [30] Byung-Kwan Lee, Ryo Hachiuma, Yong Man Ro, Yu-Chiang Frank Wang, and Yueh-Hua Wu. Unified reinforcement and imitation learning for vision-language models. arXiv preprint arXiv:2510.19307, 2025. 3
- [31] Byung-Kwan Lee, Ryo Hachiuma, Yu-Chiang Frank Wang, Yong Man Ro, and Yueh-Hua Wu. Vlsi: Verbalized layers-to-interactions from large to small vision language models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 29545–29557, 2025. 3
- [32] Jason Lee, Jiafei Duan, Haoquan Fang, Yuquan Deng, Shuo Liu, Boyang Li, Bohan Fang, Jieyu Zhang, Yi Ru Wang, Sangho Lee, et al. Molmoact: Action reasoning models that can reason in space. arXiv preprint arXiv:2508.07917, 2025. 2, 3, 7, 8, 12, 13, 14, 15
- [33] Xuanlin Li, Kyle Hsu, Jiayuan Gu, Karl Pertsch, Oier Mees, Homer Rich Walke, Chuyuan Fu, Ishikaa Lunawat, Isabel Sieh, Sean Kirmani, Sergey Levine, Jiajun Wu, Chelsea Finn, Hao Su, Quan Vuong, and Ted Xiao. Evaluating real-world robot manipulation policies in simulation. arXiv preprint arXiv:2405.05941,

2024. 6, 7, 9, 13, 15, 18

- [34] Yi Li, Yuquan Deng, Jesse Zhang, Joel Jang, Marius Memmel, Raymond Yu, Caelan Reed Garrett, Fabio Ramos, Dieter Fox, Anqi Li, et al. Hamster: Hierarchical action models for open-world robot manipulation. arXiv preprint arXiv:2502.05485, 2025. 1, 3
- [35] Zhiqi Li, Guo Chen, Shilong Liu, Shihao Wang, Vibashan VS, Yishen Ji, Shiyi Lan, Hao Zhang, Yilin Zhao, Subhashree Radhakrishnan, et al. Eagle 2: Building post-training data strategies from scratch for frontier vision-language models. arXiv preprint arXiv:2501.14818, 2025. 1
- [36] Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, and Peter Stone. Libero: Benchmarking knowledge transfer for lifelong robot learning. arXiv preprint arXiv:2306.03310, 2023. 6, 7, 9, 13, 15, 18
- [37] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36:34892–34916, 2023. 1
- [38] Songming Liu, Lingxuan Wu, Bangguo Li, Hengkai Tan, Huayu Chen, Zhengyi Wang, Ke Xu, Hang Su, and Jun Zhu. Rdt-1b: a diffusion foundation model for bimanual manipulation. arXiv preprint arXiv:2410.07864, 2024. 6, 7, 8, 10, 13, 17
- [39] Zhijian Liu, Ligeng Zhu, Baifeng Shi, Zhuoyang Zhang, Yuming Lou, Shang Yang, Haocheng Xi, Shiyi Cao, Yuxian Gu, Dacheng Li, et al. Nvila: Efficient frontier visual language models. arXiv preprint arXiv:2412.04468, 2024. 1, 8, 14
- [40] Weifeng Lu, Minghao Ye, Zewei Ye, Ruihan Tao, Shuo Yang, and Bo Zhao. Robofac: A comprehensive framework for robotic failure analysis and correction. arXiv preprint arXiv:2505.12224, 2025. 7, 9, 10, 13, 14, 16
- [41] Arjun Majumdar, Anurag Ajay, Xiaohan Zhang, Pranav Putta, Sriram Yenamandra, Mikael Henaff, Sneha Silwal, Paul Mcvay, Oleksandr Maksymets, Sergio Arnaud, et al. Openeqa: Embodied question answering in the era of foundation models. In CVPR, pages 16488–16498, 2024. 7, 8, 15, 16

- [42] Yunze Man, De-An Huang, Guilin Liu, Shiwei Sheng, Shilong Liu, Liang-Yan Gui, Jan Kautz, Yu-Xiong Wang, and Zhiding Yu. Argus: Vision-centric reasoning with grounded chain-of-thought. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 14268–14280, 2025. 3
- [43] Tomohiro Motoda, Masaki Murooka, Ryoichi Nakajo, Muhammad A. Muttaqien, Koshi Makihara, Hanbit Oh, Keisuke Shirai, Floris Erich, Ryo Hanai, and Yukiyasu Domae. Aist-bimanual manipulation, 2025. 7, 13, 14
- [44] Abby O’Neill, Abdul Rehman, Abhiram Maddukuri, Abhishek Gupta, Abhishek Padalkar, Abraham Lee, Acorn Pooley, Agrim Gupta, Ajay Mandlekar, Ajinkya Jain, et al. Open x-embodiment: Robotic learning datasets and rt-x models: Open x-embodiment collaboration 0. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 6892–6903. IEEE, 2024. 1, 7, 13
- [45] Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. Bleu: a method for automatic evaluation of machine translation. In Proceedings of the 40th annual meeting of the Association for Computational Linguistics, pages 311–318, 2002. 7
- [46] Karl Pertsch, Kyle Stachowicz, Brian Ichter, Danny Driess, Suraj Nair, Quan Vuong, Oier Mees, Chelsea Finn, and Sergey Levine. Fast: Efficient action tokenization for vision-language-action models. arXiv preprint arXiv:2501.09747, 2025. 3
- [47] Lu Qiu, Yi Chen, Yuying Ge, Yixiao Ge, Ying Shan, and Xihui Liu. Egoplan-bench2: A benchmark for multimodal large language model planning in real-world scenarios. arXiv preprint arXiv:2412.04447,

2024. 7, 8, 16

- [48] Delin Qu, Haoming Song, Qizhi Chen, Zhaoqing Chen, Xianqiang Gao, Xinyi Ye, Qi Lv, Modi Shi, Guanghui Ren, Cheng Ruan, et al. Eo-1: Interleaved vision-text-action pretraining for general robot control. arXiv preprint arXiv:2508.21112, 2025. 2, 3
- [49] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in neural information processing systems, 36:53728–53741, 2023. 5
- [50] Gabriel Sarch, Snigdha Saha, Naitik Khandelwal, Ayush Jain, Michael J Tarr, Aviral Kumar, and Katerina Fragkiadaki. Grounded reinforcement learning for visual reasoning. arXiv preprint arXiv:2505.23678,

2025. 3

- [51] Pierre Sermanet, Tianli Ding, Jeffrey Zhao, Fei Xia, Debidatta Dwibedi, Keerthana Gopalakrishnan, Christine Chan, Gabriel Dulac-Arnold, Sharath Maddineni, Nikhil J Joshi, et al. Robovqa: Multimodal long-horizon reasoning for robotics. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 645–652. IEEE, 2024. 7, 8, 13, 14, 15, 16
- [52] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Y Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024. 2, 4, 7
- [53] Zhenyi Shen, Hanqi Yan, Linhai Zhang, Zhanghao Hu, Yali Du, and Yulan He. Codi: Compressing chain-of-thought into continuous space via self-distillation. arXiv preprint arXiv:2502.21074, 2025. 3
- [54] Lucy Xiaoyang Shi, Archit Sharma, Tony Z Zhao, and Chelsea Finn. Waypoint-based imitation learning for robotic manipulation. arXiv preprint arXiv:2307.14326, 2023. 7, 13
- [55] Lucy Xiaoyang Shi, Brian Ichter, Michael Equi, Liyiming Ke, Karl Pertsch, Quan Vuong, James Tanner, Anna Walling, Haohuan Wang, Niccolo Fusai, et al. Hi robot: Open-ended instruction following with hierarchical vision-language-action models. arXiv preprint arXiv:2502.19417, 2025. 3

- [56] Min Shi, Fuxiao Liu, Shihao Wang, Shijia Liao, Subhashree Radhakrishnan, Yilin Zhao, De-An Huang, Hongxu Yin, Karan Sapra, Yaser Yacoob, et al. Eagle: Exploring the design space for multimodal llms with mixture of encoders. arXiv preprint arXiv:2408.15998, 2024. 1
- [57] BAAI RoboBrain Team, Mingyu Cao, Huajie Tan, Yuheng Ji, Xiansheng Chen, Minglan Lin, Zhiyu Li, Zhou Cao, Pengwei Wang, Enshen Zhou, et al. Robobrain 2.0 technical report. arXiv preprint arXiv:2507.02029,

2025. 8, 14

- [58] Gemini Robotics Team, Saminda Abeyruwan, Joshua Ainslie, Jean-Baptiste Alayrac, Montserrat Gonzalez Arenas, Travis Armstrong, Ashwin Balakrishna, Robert Baruch, Maria Bauza, Michiel Blokzijl, et al. Gemini robotics: Bringing ai into the physical world. arXiv preprint arXiv:2503.20020, 2025. 3
- [59] Octo Model Team, Dibya Ghosh, Homer Walke, Karl Pertsch, Kevin Black, Oier Mees, Sudeep Dasari, Joey Hejna, Tobias Kreiman, Charles Xu, et al. Octo: An open-source generalist robot policy. arXiv preprint arXiv:2405.12213, 2024. 1, 3
- [60] Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, et al. Internvl3. 5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. arXiv preprint arXiv:2508.18265, 2025. 1
- [61] Yan Wang, Wenjie Luo, Junjie Bai, Yulong Cao, Tong Che, Ke Chen, Yuxiao Chen, Jenna Diamond, Yifan Ding, Wenhao Ding, et al. Alpamayo-r1: Bridging reasoning and action prediction for generalizable autonomous driving in the long tail. arXiv preprint arXiv:2511.00088, 2025. 2
- [62] Yihao Wang, Pengxiang Ding, Lingxiao Li, Can Cui, Zirui Ge, Xinyang Tong, Wenxuan Song, Han Zhao, Wei Zhao, Pengxu Hou, et al. Vla-adapter: An effective paradigm for tiny-scale vision-language-action model. arXiv preprint arXiv:2509.09372, 2025. 3
- [63] Yilin Wu, Anqi Li, Tucker Hermans, Fabio Ramos, Andrea Bajcsy, and Claudia P’erez-D’Arpino. Do what you say: Steering vision-language-action models via runtime reasoning-action alignment verification. arXiv preprint arXiv:2510.16281, 2025. 2, 3
- [64] Violet Xiang, Chase Blagden, Rafael Rafailov, Nathan Lile, Sang Truong, Chelsea Finn, and Nick Haber. Just enough thinking: Efficient reasoning with adaptive length penalties reinforcement learning. arXiv preprint arXiv:2506.05256, 2025. 3
- [65] Jinheng Xie, Weijia Mao, Zechen Bai, David Junhao Zhang, Weihao Wang, Kevin Qinghong Lin, Yuchao Gu, Zhijie Chen, Zhenheng Yang, and Mike Zheng Shou. Show-o: One single transformer to unify multimodal understanding and generation. arXiv preprint arXiv:2408.12528, 2024. 1
- [66] Yige Xu, Xu Guo, Zhiwei Zeng, and Chunyan Miao. Softcot: Soft chain-of-thought for efficient reasoning with llms. arXiv preprint arXiv:2502.12134, 2025. 3
- [67] Jianwei Yang, Reuben Tan, Qianhui Wu, Ruijie Zheng, Baolin Peng, Yongyuan Liang, Yu Gu, Mu Cai, Seonghyeon Ye, Joel Jang, et al. Magma: A foundation model for multimodal ai agents. arXiv preprint arXiv:2502.13130, 2025. 1, 3, 8, 14
- [68] Zhaoshu Yu, Bo Wang, Pengpeng Zeng, Haonan Zhang, Ji Zhang, Lianli Gao, Jingkuan Song, Nicu Sebe, and Heng Tao Shen. A survey on efficient vision-language-action models. arXiv preprint arXiv:2510.24795,

2025. 2

- [69] Danlong Yuan, Tian Xie, Shaohan Huang, Zhuocheng Gong, Huishuai Zhang, Chong Luo, Furu Wei, and Dongyan Zhao. Efficient rl training for reasoning models via length-aware optimization. arXiv preprint arXiv:2505.12284, 2025. 3

- [70] Yifu Yuan, Haiqin Cui, Yaoting Huang, Yibin Chen, Fei Ni, Zibin Dong, Pengyi Li, Yan Zheng, and Jianye Hao. Embodied-r1: Reinforced embodied reasoning for general robotic manipulation. arXiv preprint arXiv:2508.13998, 2025. 3
- [71] Michał Zawalski, William Chen, Karl Pertsch, Oier Mees, Chelsea Finn, and Sergey Levine. Robotic control via embodied chain-of-thought reasoning. arXiv preprint arXiv:2407.08693, 2024. 2, 3
- [72] Zhen Zhang, Xuehai He, Weixiang Yan, Ao Shen, Chenyang Zhao, Shuohang Wang, Yelong Shen, and Xin Eric Wang. Soft thinking: Unlocking the reasoning potential of llms in continuous concept space. arXiv preprint arXiv:2505.15778, 2025. 3
- [73] Qingqing Zhao, Yao Lu, Moo Jin Kim, Zipeng Fu, Zhuoyang Zhang, Yecheng Wu, Zhaoshuo Li, Qianli Ma, Song Han, Chelsea Finn, et al. Cot-vla: Visual chain-of-thought reasoning for vision-language-action models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 1702–1713, 2025. 2, 3, 7, 14
- [74] Tony Z Zhao, Vikash Kumar, Sergey Levine, and Chelsea Finn. Learning fine-grained bimanual manipulation with low-cost hardware. arXiv preprint arXiv:2304.13705, 2023. 7, 8, 13
- [75] Ruijie Zheng, Yongyuan Liang, Shuaiyi Huang, Jianfeng Gao, Hal Daumé III, Andrey Kolobov, Furong Huang, and Jianwei Yang. Tracevla: Visual trace prompting enhances spatial-temporal awareness for generalist robotic policies. arXiv preprint arXiv:2412.10345, 2024. 3
- [76] Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025. 8, 14

