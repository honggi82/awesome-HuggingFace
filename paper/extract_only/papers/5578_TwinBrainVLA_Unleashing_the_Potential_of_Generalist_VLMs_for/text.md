2026-02-02 Work in progress.

# arXiv:2601.14133v2[cs.RO]30Jan2026

[Figure 1]

## TwinBrainVLA : Unleashing the Potential of Generalist VLMs for Embodied Tasks via Asymmetric Mixture-of-Transformers

### Bin Yu1,2,* Shijie Lian2,4,* Xiaopeng Lin2,5,* Yuliang Wei1,† Zhaolong Shen2,6 Changti Wu2,7

Yuzhuo Miao1,2 Xinming Wang2,8 Bailing Wang1 Cong Huang2,3 Kai Chen2,3,9,† 1HIT 2ZGCA 3ZGCI 4HUST 5HKUST(GZ) 6BUAA 7ECNU 8CASIA 9DeepCybo

https://github.com/ZGC-EmbodyAI/TwinBrainVLA

### Abstract

The fundamental premise of Vision-Language-Action (VLA) models is to harness the extensive general capabilities of pre-trained Vision-Language Models (VLMs) for generalized embodied intelligence. However, standard robotic ﬁne-tuning inevitably disrupts the pre-trained feature space, leading to "catastrophic forgetting" that compromises the general visual understanding we aim to leverage. To effectively utilize the uncorrupted general capabilities of VLMs for robotic tasks, we propose TwinBrainVLA , which coordinates two isomorphic VLM pathways: a frozen generalist (also called "Left Brain") and a trainable specialist (also called "Right Brain"). Our architecture utilizes a Asymmetric Mixture-of-Transformers (AsyMoT) mechanism, enabling the Right Brain to dynamically query and fuse intact semantic knowledge from the Left Brain with proprioceptive states. This fused representation conditions a ﬂow-matching action expert for precise continuous control. Empirical results on SimplerEnv and RoboCasa benchmarks demonstrate that by explicitly retaining general capabilities, TwinBrainVLA achieves substantial performance gains over baseline models in complex manipulation tasks.

[Figure 2]

### 1 Introduction

The pursuit of embodied artiﬁcial intelligence has recently converged on the paradigm of VisionLanguage-Action (VLA) model (Kim et al., 2024; Intelligence et al., 2025; NVIDIA et al., 2025). By grafting robotic control heads onto pre-trained Vision-Language Models (VLMs), the fundamental premise is to harness the rich semantic reasoning and open-world generalization capabilities learned from internet-scale data to enable generalized robotic control. Ideally, such a system should not merely execute motor commands but leverage its "VLM brain" to understand complex instructions, reason about unseen objects, and plan over long horizons.

However, a critical paradox undermines current VLA architectures: the optimization process required to learn low-level sensorimotor control inevitably disrupts the high-level semantic features we aim to leverage. During standard robotic ﬁne-tuning, the VLM backbone is aggressively updated to minimize action prediction errors on narrow, domain-speciﬁc datasets. This creates a conﬂict between the original objective of semantic understanding and the new objective of proprioceptive control, precipitating "catastrophic forgetting" (Hancock et al., 2025a; Zhang et al., 2026; Zhou et al., 2025; Fang et al., 2025). Consequently, the model sacriﬁces its pre-trained general capabilities to become a specialized controller. This defeats the original purpose of the VLA paradigm: instead of inheriting a generalized "brain," the ﬁne-tuned model degenerates into a localized policy, stripping away the very open-world understanding required for robust generalization.

*Equal contribution †Corresponding author

Work done at Zhongguancun Academy (Beijing).

Therefore, the central challenge is to effectively utilize the uncorrupted general capabilities of VLMs to enhance robotic performance. We posit that a generalizable VLA model must simultaneously possess both general visual understanding capabilities and low-level servo control proﬁciency. Drawing inspiration from the biological principle of hemispheric lateralization, where the brain allocates distinct functions to specialized hemispheres, we propose that a VLA should maintain a dedicated "semantic anchor" separate from its "motor controller."

[Figure 3]

[Figure 4]

Action Expert

Action Expert

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

KV

Left Brain

Right Brain

VLM as Backbone

(specialist-only)

(generalist)

(specialist)

[Figure 10]

###### a) Vanilla VLA b) TwinBrainVLA

image token text token state info

Figure 1: Architectural comparison between Vanilla VLA and TwinBrainVLA.

[Figure 11]

In this paper, we introduce TwinBrainVLA , a novel VLA framework that orchestrates two isomorphic VLM pathways via an asymmetric dual-stream joint training strategy. Structurally, our model comprises a frozen "Left Brain" (generalist) and a trainable "Right Brain" (specialist). The Left Brain preserves the intact, open-world knowledge of the pre-trained VLM, while the Right Brain specializes in processing proprioceptive states and generating actions. Crucially, we propose a novel Asymmetric Mixture-of-Transformers (AsyMoT) mechanism, which enables the trainable Right Brain to dynamically query and fuse semantic features from the frozen Left Brain. This design ensures that the action expert is conditioned on uncorrupted general knowledge, allowing the robot to "think" with a generalist brain while "acting" with a specialist body.

Our contributions are summarized as follows:

- • We quantitatively analyzed the impairment of the general capabilities of VLMs caused by VLA training through preliminary experiments, namely the occurrence of the "catastrophic forgetting" phenomenon.
- • We propose TwinBrainVLA , an asymmetric dual-stream architecture that structurally decouples semantic understanding from embodied control, allowing the system to leverage pre-trained capabilities without disruption.

[Figure 12]

- • We introduce the Asymmetric Mixture-of-Transformers (AsyMoT) mechanism to facilitate efﬁcient information interaction between the two VLM pathways, enabling the specialist Right Brain to dynamically query general knowledge from the frozen Left Brain.
- • Extensive experiments and evaluations on the SimplerEnv and RoboCasa benchmarks, as well as real-robot settings, demonstrate the effectiveness of the TwinBrainVLA architecture, the AsyMoT mechanism, and our proposed training strategy.

### 2 Related Work

Multimodal understanding. The landscape of computer vision and natural language processing has been revolutionized by VLMs (Liu et al., 2023b; Bai et al., 2025b,a). By seamlessly integrating visual encoders with Large Language Models (LLMs) through sophisticated projection layers or adapter modules (Jiang et al., 2025a), these models exhibit emergent abilities in semantic understanding and visual question answering (VQA). Given the inherent limitations of general-purpose VLMs in spatial perception (Xiao et al., 2026; Dang et al., 2025; Zhang et al., 2025; Chen et al., 2025; Yang et al., 2025b), recent studies have increasingly focused on employing post-training strategies to enhance spatial intelligence (Yang et al., 2025a; Tan et al., 2025; Zhou et al., 2026; Zhu et al., 2025; Li et al., 2025) and

construct specialized embodied foundation models tailored for embodied scenarios (Team et al., 2025a; Yang et al., 2025a; Hao et al., 2025; Lin et al., 2025c).

Vision-Language-Action (VLA) Models. Building upon the zero-shot generalization and rich semantic priors of VLMs, VLA models have emerged as a scalable paradigm for embodied intelligence (Pertsch et al., 2025; Zheng et al., 2025a; Wen et al., 2025; Shukor et al., 2025; Zheng et al., 2025b; Lin et al., 2025b; Cai et al., 2026). By ﬁne-tuning pre-trained VLMs on large-scale robotic manipulation datasets, these models learn to map multimodal instructions and observations to low-level control signals (Jiang et al., 2025b; Fan et al., 2025; Zhai et al., 2025). To mitigate the degradation of general conversational capabilities in VLMs during VLA training, several approaches have been proposed. LangForce (Lian et al., 2026) attempts to enhance VLA performance by introducing mechanisms to mitigate visual shortcuts. ChatVLA (Zhou et al., 2025) and VLM2VLA (Hancock et al., 2025b) attempts to incorporate general dialogue data for co-training; however, this method still suffers from catastrophic forgetting and requires careful curation of general QA data. ThinkAct (Huang et al., 2025) and related works (Yin et al., 2025; Lin et al., 2025a; Lee et al., 2025) introduce Chain-of-Thought (CoT) reasoning into VLA training. Nevertheless, current mainstream VLA datasets lack CoT supervision signals, and the annotation process remains expensive and difﬁcult to scale. Instead of relying on extensive data engineering, we propose a structural solution that explicitly decouples semantic preservation from motor learning, enabling the VLA to directly harness the uncorrupted general capabilities of the VLM backbone.

###### POPE Bench (%)

###### SeedBench (%)

: only robot data

[Figure 13]

100

90

88.87

78.79

VLA-Training images instruction state Co-Training

VLA-Training Co-Training

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

80

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

80

70

[Figure 37]

[Figure 38]

actions

[Figure 39]

[Figure 40]

[Figure 41]

60

[Figure 42]

[Figure 43]

60

[Figure 44]

50

[Figure 45]

[Figure 46]

40

[Figure 47]

40

[Figure 48]

policy

30

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

10.64 0.04

[Figure 54]

20

[Figure 55]

20

5.95 3.51

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

10

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

: robot data + visual-text data

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

0

0

[Figure 91]

original 10K 20K 30K 40K 50K 60K 70K 80K 90K 100K

original 10K 20K 30K 40K 50K 60K 70K 80K 90K 100K

actions

images instruction state

###### ScienceQA Bench (%)

###### MME Cognition Score

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

700

100

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

94.65 643.93

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

VLA-Training Co-Training

[Figure 108]

VLA-Training Co-Training

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

600

[Figure 116]

80

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

500

[Figure 121]

chat

[Figure 122]

60

images instruction

400

policy

300

[Figure 123]

[Figure 124]

40

[Figure 125]

[Figure 126]

200

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

20

[Figure 131]

[Figure 132]

1.88 0.00 32.14 26.07

100

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

Input Output

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

0

0

original 10K 20K 30K 40K 50K 60K 70K 80K 90K 100K

original 10K 20K 30K 40K 50K 60K 70K 80K 90K 100K

(a) Diagram of VLA-Training and Co-Training.

(b) Catastrophic forgetting in the VLM during VLA training.

- Figure 2: Empirical Evidence of Catastrophic Forgetting. (a) Overview of two prevailing training paradigms: standard VLA-Training (utilizing robot-only data) and Co-Training (mixing robot data with general visionlanguage data). (b) Evaluation of the VLMs general visual understanding capabilities across four benchmarks (POPE (Li et al., 2023b), SEED-Bench (Li et al., 2023a), ScienceQA (Lu et al., 2022), and MME (Fu et al., 2025)) during the VLA ﬁne-tuning process. Both strategies exhibit a precipitous decline in performance as training steps increase, demonstrating that co-training with general data is insufﬁcient to prevent the collapse of the VLMs pretrained semantic reasoning when optimizing for robotic control.

### 3 Motivation: The Dilemma of Utilizing General Capabilities in VLA

##### 3.1 The Conﬂict Between Specialization and Generalization

Standard VLA models are built on a compelling premise: by ﬁne-tuning pre-trained VLMs on robotic datasets, we aim to endow robots with the rich semantic reasoning and open-world understanding inherent in the backbone. Ideally, the VLM should serve as a "generalist brain" that guides the robotic body through complex, unseen environments.

However, a fundamental paradox exists in the current transfer learning paradigm. To adapt the VLM for precise sensorimotor control, the model is ﬁne-tuned on robotic datasets (Drobot) that are fundamentally different from the massive internet-scale data used during pre-training. This aggressive optimization for low-level action prediction inevitably disrupts the pre-trained feature space, triggering "catastrophic forgetting". This phenomenon turns the VLM backbone from a "generalist" into a "specialist", sacriﬁcing its linguistic brain to gain a robotic body.

##### 3.2 Empirical Analysis: The Severity of Catastrophic Forgetting

To empirically quantify the extent of catastrophic forgetting discussed above, we conducted a preliminary study using VLM backbones: Qwen2.5-VL-3B-Instruct and Qwen3-VL-4B-Instruct. We evaluate the models’ general visual capabilities before and after ﬁne-tuning under two distinct paradigms: (1) VLA Training: The model is ﬁne-tuned exclusively on robotic trajectory data to optimize action generation.

- (2) Co-Training: Following recent strategies (Mazzaglia et al., 2025; Zhou et al., 2025) to mitigate forgetting, we mix general visual-QA data with robotic data at a 1:1 ratio during ﬁne-tuning, aiming to preserve visual conversational capabilities.

Result Analysis. The results are presented in Figure 2b. First, standard VLA Training leads to a complete collapse of general visual understanding. For instance, the POPE score of Qwen3-VL drops from 88.87% to near-zero (0.04%). This conﬁrms that optimizing for proprioceptive action control aggressively overwrites the semantic features required for VQA tasks. More critically, the co-training strategy also fails to prevent this degradation.

##### Key Findings

- 1. Training VLA models with robotics data induces severe catastrophic forgetting in the VLM, which undermines the original premise of the paradigm that builds VLAs from VLMs.
- 2. While Co-Training is frequently suggested as a countermeasure, our experiments show that it merely mitigates the symptoms rather than providing a fundamental solution.

These ﬁndings expose a critical bottleneck: standard ﬁne-tuning effectively erases the ’VLM brain’ before it can be deployed for robotic reasoning, rendering it impossible to harness pre-trained general knowledge for complex, open-ended tasks. While the success of multimodal applications has proven that general understanding is essential for downstream performance which is also a key insight driving the VLA paradigm. However, current training paradigms lead to severe catastrophic forgetting, which directly contradicts the initial design philosophy of VLAs. Therefore, the challenge remains: How can we construct VLA models to leverage the uncorrupted multimodal understanding capabilities of VLMs to enhance their own performance?

[Figure 170]

### 4 Method: TwinBrainVLA

[Figure 171]

In this section, we present TwinBrainVLA , a novel framework designed to enable VLA models to effectively utilize the general visual understanding capabilities of pre-trained VLMs for embodied tasks. While standard monolithic architectures suffer from feature degradation during ﬁne-tuning, our approach aims to retain the full spectrum of pre-trained semantic knowledge and make it explicitly accessible for robotic control. To this end, we introduce an asymmetric dual-stream design that structurally separates the preservation of general capabilities from the learning of sensorimotor skills, ensuring that the control policy can leverage uncorrupted multimodal understanding capabilities.

##### 4.1 Asymmetric Dual-VLM Backbone

The backbone of TwinBrainVLA comprises two parallel VLM pathways: the "Left Brain" (ML) and the "Right Brain" (MR). Both streams are initialized with identical pre-trained weights (e.g., Qwen3-VL series) but play distinct roles to facilitate the utilization of general capabilities for robot control.

The "Left Brain" as a General Capability Reservoir. The Left Brain serves as the provider of general visual understanding. To ensure that the pre-trained capabilities remain intact, it is kept strictly frozen throughout the training phase. It processes the visual observation I and instruction T to generate semantic representations:

HL0 = [V(I);T (T)] (1) where V and T denote the vision encoder and text tokenizer. By freezing this pathway, the Left Brain

acts as a stable semantic anchor, offering a continuous source of open-world knowledge that is immune

independent weights, joint attention

[Figure 172]

continuous actions

Asymmetric MoT

[Figure 173]

[Figure 174]

-1.8 3.7 1.4

[Figure 175]

Causal Self-Attention

Left Brain Right Brain

KV

Condition

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

Generalist Specialist for robot

Generalist Specialist for robot

[Figure 180]

KV

[Figure 181]

denoise

[Figure 182]

[Figure 183]

Action Expert

Transformer Block Transformer Block

Wq Wk Wv Wq Wk Wv

vision encoder

text tokenizer

vision encoder

text tokenizer

state encoder

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

Flow-Matching Algorithm

task instruction

task instruction

Left Brain Input Right Brain Input

[Figure 199]

a) Overview of TwinBrainVLA

b) Asymmetric MoT

[Figure 200]

- Figure 3: The framework of TwinBrainVLA . (a) Overall Architecture. The model features an Asymmetric Mixture-of-Transformers design composed of two distinct pathways: a frozen "Left Brain" (Generalist) for semantic reasoning and a trainable "Right Brain" (Specialist) for embodied motor control. The Right Brain fuses visual, textual, and proprioceptive state inputs to provide conditioning for the Action Expert, which utilizes a Flow-Matching algorithm to denoise continuous robotic actions. (b) Asymmetric MoT Mechanism (AsyMoT). Through causal self-attention, the trainable Right Brain attends to the frozen Key-Value (KV) pairs of the Left Brain, enabling the transfer of general semantic knowledge to the robotic control policy without catastrophic forgetting.

to the catastrophic forgetting typically induced by robotic ﬁne-tuning.

The "Right Brain" as a Semantic-Augmented Controller. The Right Brain is a trainable specialist responsible for executing robotic actions. To ground the model in the physical world, we introduce a lightweight State Encoder ϕ, which projects the robot’s proprioceptive state s (e.g., joint angles) into the VLM’s embedding space:

HR0 = [V(I);T (T);ϕ(s)] (2)

Unlike a standard VLA that relies solely on its own degrading parameters, the Right Brain is designed to actively harness the capabilities of the Left Brain to inform its decision-making.

Asymmetric Mixture-of-Transformers (AsyMoT). Mixture-of-Transformers (MoT) (Liang et al., 2025) is a model architecture that implements forward propagation by connecting two independent Transformer networks via joint attention at each corresponding layer. Its advantage lies in the ability to facilitate information interaction between multiple modalities while still maintaining their independent computation. Inspired by this work, we propose the Asymmetric Mixture-of-Transformers (AsyMoT) mechanism to enable the Right Brain to utilize the general visual understanding from the Left Brain. This mechanism establishes a unidirectional information bridge, allowing the control policy to query the frozen general knowledge.

Let HlL and HlR denote the hidden states of the Left and Right Brains at layer l, respectively. the Left Brain remains frozen during training. Its self-attention mechanism operates independently to

preserve pre-trained general capabilities:

HLl+1 = Attn(QlL,KLl ,VLl) + FFN(HlL), (3) where Q,K,V are derived from HlL using frozen projection weights.

The Right Brain is trainable and employs an asymmetric joint attention mechanism. It queries not only its own context but also the semantic features from the Left Brain. Speciﬁcally, at each attention head, the Right Brain computes its Query (QR), while the Key (K) and Value (V ) are constructed by concatenating the representations from both brains:

Kjoint = [sg(KLl ) ; KRl ], (4)

Vjoint = [sg(VLl) ; VRl ], (5) HlR+1 = Softmax(

)Vjoint + FFN(HlR), (6)

QlR(Kjoint)T √dk

where [ ; ] denotes concatenation along the sequence length dimension, and sg(·) indicates the stopgradient operation.

AsyMoT v.s. Cross-Attention. The key to the AsyMoT mechanism lies in the asymmetric joint attention at each layer. It allows the ’right brain’ to attend to both the ’left brain’ and itself, whereas the ’left brain’ is restricted to attending only to itself, resulting in an asymmetric phenomenon. In contrast, Cross Attention is a different multimodal fusion architecture. It enables a Query from one modality to attend to the Key-Values of another modality, but precludes it from attending to itself. This constitutes the critical distinction between AsyMoT and Cross Attention."

##### 4.2 Flow-Matching Action Expert

Following the paradigm of recent mainstream VLAs (NVIDIA et al., 2025; Intelligence et al., 2025), we employ a generative action expert based on ﬂow matching to enable high-precision continuous control. This module is implemented as a Diffusion Transformer (DiT) (Peebles and Xie, 2023), which operates as a conditional decoder. Speciﬁcally, it takes the representations HR from the Right Brain as the condition to synthesize continuous action trajectories from noise. The training objective is deﬁned by the standard vector ﬁeld regression loss:

[||vψ(at,t,HR) − (a1 − a0)||2]

, (7) where vψ denotes the DiT network, a0 is the Gaussian noise, and a1 represents the ground-truth action.

Laction(ψ) = Et,a0,a1

##### 4.3 Training Strategy

Optimization Objective. Consistent with standard VLA ﬁne-tuning paradigms (NVIDIA et al., 2025), we train TwinBrainVLA using exclusively the robotic action objective. The training objective is to minimize the Flow-Matching loss:

Ltotal = Laction(θR,ψ,ϕ;Drobot), (8)

where Drobot represents the robotic demonstration dataset, and {θR,ψ,ϕ} denote the trainable parameters of the Right Brain, Action Expert, and State Encoder, respectively. It is worth noting that in monolithic VLA architectures, relying solely on action loss typically leads to severe catastrophic forgetting of general semantic capabilities. However, our dual-stream design structurally immunizes the model against this degradation: the "Right Brain" is free to specialize entirely in control dynamics, while the frozen "Left Brain" implicitly safeguards the linguistic and semantic priors.

### 5 Experiment

To comprehensively evaluate the efﬁcacy of TwinBrainVLA, we conduct extensive experiments on two simulation benchmarks: SimplerEnv (Li et al., 2024c) and RoboCasa (Nasiriany et al., 2024). Our training pipeline is built upon the starVLA framework (starVLA, 2025), distributed across 16 × NVIDIA H100 GPUs. We strictly follow its default training protocols to ensure a fair comparison. More extensive simulation benchmarks and real-world robotic experiments are in progress.

##### 5.1 Experiments on SimplerEnv (OOD)

Implementation and Training Setup. To demonstrate the scalability and effectiveness of our architecture, we instantiate TwinBrainVLA with two state-of-the-art VLM backbones: Qwen2.5-VL-3B-Instruct and Qwen3-VL-4B-Instruct. Consistent with our asymmetric design, both the frozen Left Brain and the trainable Right Brain are initialized from these pre-trained checkpoints to ensure aligned feature spaces. For the training data, we utilize two large-scale subsets from the Open X-Embodiment (OXE) dataset: the Bridge-V2 and the Fractal dataset. Comprehensive implementation details are provided in Appendix B.

Evaluation. The benchmark consists of four manipulation tasks: "put spoon on towel", "put carrot on plate", "stack green block on yellow block", "put eggplant in the yellow basket". For each task, we evaluate our VLA policy using the evaluation script provided by the SimplerEnv repository (Li et al., 2024c).

- Table 1: Results of evaluating the VLA models with the WidowX robot in the SimplerEnv simulation environment. We highlight the best results in bold and the second-best results with underline. Here, Vanilla VLA refers to a VLA constructed with a single VLM backbone. Its only difference from TwinBrainVLA is the removal of the frozen Left Brain.

Put Spoon on Towel

Put Carrot on Plate

Stack Green Block on Yellow Block

Put Eggplant in Yellow Basket

Method

Average

RT-1-X (ONeill et al., 2024) 0.0 4.2 0.0 0.0 1.1 Octo-Base (Team et al., 2024) 15.8 12.5 0.0 41.7 17.5 Octo-Small (Team et al., 2024) 41.7 8.2 0.0 56.7 26.7 OpenVLA (Kim et al., 2024) 4.2 0.0 0.0 12.5 4.2 RoboVLM (Li et al., 2024b) 50.0 37.5 0.0 83.3 42.7 TraceVLA (Zheng et al., 2025c) 12.5 16.6 16.6 65.0 27.7 SpatialVLA (Qu et al., 2025) 20.8 20.8 25.0 70.8 34.4 ThinkAct (Huang et al., 2025) 58.3 37.5 8.7 70.8 CogACT (Li et al., 2024a) 71.7 50.8 15.0 67.5 51.3 VideoVLA (Shen et al., 2025) 75.0 20.8 45.8 70.8 53.1 π0 (Black et al., 2024) 29.1 0.0 16.6 62.5 27.1 π0.5 (Intelligence et al., 2025) 49.3 64.7 44.7 69.7 57.1 Isaac-GR00T-N1.6-Bridge (Team et al., 2025b) 64.5 65.5 5.5 93.0 57.1 Vanilla VLA + Qwen2.5-VL-3B-Instruct 74.9 44.7 9.3 81.2 52.5 Vanilla VLA + Qwen2.5-VL-7B-Instruct 68.7 35.4 16.7 62.5 44.8 Vanilla VLA + Qwen3-VL-4B-Instruct 87.5 50.0 29.2 54.2 55.2 Vanilla VLA + Qwen3-VL-8B-Instruct 68.7 38.5 30.2 87.9 56.3

###### ours

TwinBrainVLA + Qwen2.5-VL-3B-Instruct 83.3 41.7 31.5 77.1 58.4 TwinBrainVLA + Qwen3-VL-4B-Instruct 87.5 58.3 33.3 79.1 64.5

To mitigate randomness, we run 480 independent trials and report the mean performance (Avg@480). Since the evaluation environments are completely absent from the training set, this constitutes an Outof-Domain (OOD) evaluation.

Results. The quantitative results on the SimplerEnv benchmark are presented in Table 1. Notably, despite not undergoing large-scale pre-training for robotic action prediction, TwinBrainVLA achieves state-of-the-art performance among all listed methods. Our framework demonstrates strong generalizability across different VLM families, attaining competitive success rates of 58.4% with Qwen2.5-VL-

- 3B-Instruct and 64.5% with Qwen3-VL-4B-Instruct. The latter conﬁguration surpasses the strongest baseline, Isaac-GR00T-N1.6 (57.1%), by a notable margin of +7.4%, validating the effectiveness of our asymmetric dual-brain architecture in bridging high-level semantic understanding and low-level robotic control.

5.2 Experiments on RoboCasa

Implementation. We train TwinBrainVLA on the Humanoid Robot Tabletop Manipulation subset from the PhysicalAI-Robotics-GR00T-X-Embodiment-Sim dataset (NVIDIA et al., 2025). All other experimental settings, including model initialization, distributed training infrastructure, and hyperparameters, remain identical to those described in Sec. 5.1.

Evaluation. We evaluate our policy on the RoboCasa GR1 Tabletop Benchmark, which comprises a diverse suite of 24 tabletop manipulation tasks. These tasks involve complex interactions with articulated objects and varying geometries, such as PnPBottleToCabinetClose (placing a bottle into a cabinet and closing the door) and PnPCanToDrawerClose. To ensure statistical reliability while maintaining consistency with our previous experiments, we report the success rate averaged over 50 independent trials for each of the 24 tasks.

Results. The quantitative results on the RoboCasa GR1 Tabletop Benchmark are presented in Table 8. TwinBrainVLA achieves the best performance across all 24 manipulation tasks, with our Qwen3-VL-

- 4B-Instruct variant attaining an average success rate of 54.6%, followed closely by the Qwen2.5-VL3B-Instruct variant at 53.5%. These results substantially outperform all baseline methods: our best

- Table 2: Results of evaluating the VLA models with the GR1 robot in the RoboCasa Tabletop simulation environment. The results for Isaac-GR00T N1.5 and Isaac-GR00T N1.6 are sourced from the ofﬁcial IsaacGR00T github repository (Team et al., 2025b). The results for the ﬁrst four baseline methods are sourced from the ofﬁcial starVLA experiments (starVLA, 2025). Performance on all 24 tasks can be found in Table 8 of the Appendix.

Model +Qwen3VLQwenFAST QwenGR00T+Qwen3VL +Qwen3VLQwenPI +Qwen3VLQwenOFT Average 39.0 47.8 43.9 48.8

Model Isaac-GR00TN1.5 Isaac-GR00TN1.6 TwinBrainVLA+ Qwen2.5VL TwinBrainVLA+ Qwen3VL Average 48.2 47.6 53.5 54.6

model surpasses Isaac-GR00T-N1.6 (47.6%) by +7.0%, QwenGR00T (47.8%) by +6.8%, and QwenPI (43.9%) by +10.7%. These results reinforce our hypothesis that decoupling semantic understanding from embodied perception enables more effective learning of ﬁne-grained manipulation skills in complex tabletop scenarios.

##### 5.3 Experiments on LIBERO Benchmark

We also evaluate TwinBrainVLA on the LIBERO benchmark (Liu et al., 2023a). However, given that current VLA research has saturated this benchmark (over 95%) and its tasks are strictly in-domain, it offers limited insight into the beneﬁts of general semantic preservation. To better evaluate the model’s generalization capabilities, we combined the training data from all four LIBERO task suites for VLA training and evaluated a single model across all four suites. In this setting, TwinBrainVLA attains a 97.6% average success rate without relying on massive pre-training. See Appendix F for implementation and evaluation details.

Table 3: Comparison on the LIBERO benchmark.

Method Spatial Object Goal Long Avg OpenVLA 87.4 88.4 79.2 53.7 76.5 π0 96.8 98.8 95.8 85.2 94.1 π0.5 98.8 98.2 98.0 92.4 96.9 TwinBrainVLA 99.2 99.0 96.8 95.4 97.6

##### 5.4 Real-robot Experiments

###### To demonstrate the practical applicability of TwinBrainVLA in real world, we conduct experiments using a Franka Research 3 (7-DOF).

[Figure 201]

Training Data Evaluation pick up the blue block and put it into the gray box.

OOD

instruction

pick up the red block and put it into the gray box.

pick up all blocks and put it into the gray box.

Pick All

Figure 4: Real-robot Experiment Environment.

Pre-training Setup. We curate a high-quality pre-training corpus by selecting 20 subsets from the Open X-Embodiment (OXE) dataset. Our selection criteria prioritize high video resolution, diverse task instructions, and uniﬁed end-effector position control formats. This dataset covers multiple robot embodiments, including WidowX, Google Robot, and Franka (see Appendix A for details). TwinBrainVLA is pre-trained on this datasets to acquire generalizable sensorimotor priors before speciﬁc task adaptation.

Fine-tuning and Evaluation. For downstream adaptation, we collect 300 teleoperated demonstration trajectories on the Franka robot for a pick-and-place task (instruction: “pick up the red block and put it into the gray box”). To rigorously assess robustness, we design three evaluation protocols: (1) InDomain: The test environment and objects are identical to the training data; (2) Out-of-Domain (OOD): The target object’s attributes (e.g., block color) are altered to evaluate zero-shot visual generalization.

- (3) Pick-All: In this task, the robot is required to place all blocks into the box. We expect the model to complete this long-horizon task by leveraging the atomic actions learned from the training set. We report the average success rate over 30 independent trials for each setting.

Results. In real-robot experiments, despite utilizing signiﬁcantly less pre-training data than π0.5, our method achieves comparable performance after ﬁne-tuning on teleoperation data. Furthermore, it demonstrates robust results in both OOD and Pick-All evaluations. This validates the effectiveness of our model architecture and underscores the importance of retaining generalizability for task generalization.

- Table 4: Real-Robot Experiment Results. Success rates (%) of the pick-and-place task on the Franka Research 3 robot.

Method

Success Rate In-Domain Out-of-Domain Pick-All

OpenVLA (Kim et al., 2024) 7/30 0/30 0/30 π0 (Black et al., 2024) 23/30 9/30 0/30 π0.5 (Intelligence et al., 2025) 28/30 13/30 2/30 Vanilla VLA 22/30 9/30 0/30 TwinBrainVLA (Ours) 28/30 15/30 3/30

5.5 Ablation Experiments

Impact of asymmetric freezing strategy. To validate our asymmetric design, we ablate the freezing strategy by making the Left Brain trainable. Results show that this conﬁguration hurts performance, causing a 7% drop in success rate for the Qwen3-VL-4B backbone on SimplerEnv. This result highlights that explicitly retaining general capabilities as an accessible resource is paramount for achieving high performance in VLA tasks.

- Table 5: Ablation on Freezing Strategy. Quantitative evaluation of success rates (%) on SimplerEnv tasks. We benchmark TwinBrainVLA against the Single-Stream VLA (QwenGR00T) under identical training settings using Qwen2.5-VL-3B and Qwen3-VL-4B backbones.

Put Spoon on Towel

Put Carrot on Plate

Stack Block

Eggplant in Basket

Avg Qwen2.5-VL-3B-Instruct No Freezing 64.6 45.8 20.8 64.6 49.0 TwinBrainVLA 83.3 41.7 31.5 77.1 58.4 Qwen3-VL-4B-Instruct No Freezing 85.4 50.0 29.1 70.8 58.8 TwinBrainVLA 87.5 58.3 33.3 79.1 64.5

Method

Impact of interaction density in AsyMoT. In TwinBrainVLA, joint attention is performed layerwise by default. We further investigated the impact of interaction density by sparsifying the connections. Speciﬁcally, we experimented with interaction intervals of k = {0,1,2,4,8} layers. When k = 1, the asymmetric joint attention and naive self-attention are performed alternately. The results in Table 6 demonstrate that effective information fusion yields performance gains, whereas performance degrades when the connections become overly sparse.

Impact of dual-VLM. If the left brain and the attention connections between the two VLMs are removed, the network architecture degenerates into a Vanilla VLA. As shown in Table 1, this degradation leads to a performance drop of nearly 7 percentage points.

- Table 6: Ablation on AsyMoT Interaction Frequency. We investigate the impact of sparsifying the joint attention connections by varying the interaction interval k on SimplerEnv benchmark.

###### Interval (k) 0 (ours) 1 2 4 8

Qwen2.5-VL-3B-Instruct based 58.4 59.4 57.4 56.5 55.9 Qwen3-VL-4B-Instruct based 64.5 64.1 66.7 62.5 61.9

### 6 Discussion

##### 6.1 From Forgetting Mitigation to Capability Utilization

TwinBrainVLA effectively prevents catastrophic forgetting by maintaining a strictly frozen "Left Brain" throughout the training process. This strategy explicitly preserves the general semantic understanding capabilities acquired during VLM pre-training, ensuring that the model retains its robust open-world knowledge while the "Right Brain" specializes in embodied control.

Attention Map (Tokens from Right Brain -> All Tokens)

[Figure 202]

TokensfromtheRightBrain

All Tokens (Left + Right)

Figure 5: TwinBrainVLA Attention Map Visualization at Runtime. It can be seen that the right brain attends to the semantic information of the left brain via the AsyMoT mechanism.

##### 6.2 The Necessity of General Semantics for Embodied Intelligence

The fundamental premise of VLA models is to harness the extensive general capabilities acquired during the pre-training of VLMs to achieve generalized embodied intelligence. However, current VLA training paradigms often suffer from severe overﬁtting to speciﬁc robotic datasets (Wanna et al., 2026), leading to "catastrophic forgetting" of the backbone’s original capabilities. Addressing this paradox is the primary motivation behind TwinBrainVLA. TwinBrainVLA addresses this by explicitly preserving the backbone’s semantic integrity. Our results conﬁrm that retaining general capabilities directly boosts performance: our approach matches commercial models (e.g., π0.5) without massive pre-training. Furthermore, this preserved knowledge enables robust instruction following and semantic generalization in real-world scenarios, handling complex tasks that traditional ﬁne-tuning struggles to address."

##### 6.3 "Twin-to-One" via Knowledge Distillation

While TwinBrainVLA successfully achieves the utilization of general capabilities by incorporating a frozen VLM as a semantic anchor, we acknowledge that this dual-stream architecture inevitably introduces additional computational overhead during deployment. To bridge the gap between high-level capability utilization and low-latency deployment, we conducted a preliminary exploration into a "Twinto-One" knowledge distillation paradigm. In this experiment, we employed the feature representations from the Right Brain as a source of supervisory signals to re-train a Vanilla VLA architecture. The results revealed a tangible improvement in performance, which substantiates that TwinBrainVLA achieves superior training outcomes through the effective fusion of the Left Brain. This ﬁnding also highlights the potential for deeper exploration into internalizing these fused capabilities into simpler architectures. The details of these experiments are provided in Appendix G.

### 7 Conclusion

In this work, we identiﬁed a critical paradox in current VLA models: the ﬁne-tuning process required for robotic control inevitably leads to the "catastrophic forgetting" of the VLM backbone’s general capabilities. In order to harness the pre-trained general capabilities of VLMs to achieve generalization in embodied AI, we introduce TwinBrainVLA which is a framework that structurally decouples multimodal understanding from sensorimotor control. By orchestrating a frozen "Left Brain" for openworld understanding and a trainable "Right Brain" for embodied action via Asymmetric Mixture-ofTransformers (AsyMoT) mechanism, our architecture allows the policy to query high-level semantics without corrupting them. Extensive empirical evaluations on SimplerEnv, RoboCasa benchmarks and real-robot experiments demonstrate that TwinBrainVLA signiﬁcantly outperforms mainstream VLAs.

[Figure 203]

While TwinBrainVLA marks a meaningful step towards general embodied AI, the challenges of achieving higher efﬁciency and deeper semantic fusion remain. We provide a detailed discussion on these limitations and future directions in Appendix H.

### References

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhifang Guo, Qidong Huang, Jie Huang, Fei Huang, Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng Li, and 45 others. 2025a. Qwen3-vl technical report.

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, and 8 others. 2025b. Qwen2.5-vl technical report.

Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy

Groom, Karol Hausman, Brian Ichter, and 1 others. 2024. π0: A vision-language-action ﬂow model for general robot control. arXiv preprint arXiv:2410.24164.

Junhao Cai, Zetao Cai, Jiafei Cao, Yilun Chen, Zeyu He, Lei Jiang, Hang Li, Hengjie Li, Yang Li, Yufei Liu, Yanan Lu, Qi Lv, Haoxiang Ma, Jiangmiao Pang, Yu Qiao, Zherui Qiu, Yanqing Shen, Xu Shi, Yang Tian, and 23 others. 2026. Internvla-a1: Unifying understanding, generation and action for robotic manipulation. Preprint, arXiv:2601.02456.

Shiqi Chen, Tongyao Zhu, Ruochen Zhou, Jinghan Zhang, Siyang Gao, Juan Carlos Niebles, Mor Geva, Junxian He, Jiajun Wu, and Manling Li. 2025. Why is spatial reasoning hard for VLMs? an attention mechanism perspective on focus areas. In International conference on machine learning (ICML).

Ronghao Dang, Yuqian Yuan, Yunxuan Mao, Kehan Li, Jiangpin Liu, Zhikai Wang, Xin Li, Fan Wang, and Deli Zhao. 2025. Rynnec: Bringing mllms into embodied world. Preprint, arXiv:2508.14160.

Shichao Fan, Kun Wu, Zhengping Che, Xinhua Wang, Di Wu, Fei Liao, Ning Liu, Yixue Zhang, Zhen Zhao, Zhiyuan Xu, Meng Li, Qingjie Liu, Shanghang Zhang, Min Wan, and Jian Tang. 2025. Xr-1: Towards versatile vision-language-action models via learning uniﬁed vision-motion representations. Preprint, arXiv:2511.02776.

Zhen Fang, Zhuoyang Liu, Jiaming Liu, Hao Chen, Yu Zeng, Shiting Huang, Zehui Chen, Lin Chen, Shanghang Zhang, and Feng Zhao. 2025. Dualvla: Building a generalizable embodied agent via partial decoupling of reasoning and action. Preprint, arXiv:2511.22134.

Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, Yunsheng Wu, Rongrong Ji, Caifeng Shan, and Ran He. 2025. MME: A comprehensive evaluation benchmark for multimodal large language models. In The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track.

- Asher J. Hancock, Xindi Wu, Lihan Zha, Olga Russakovsky, and Anirudha Majumdar. 2025a. Actions as language: Fine-tuning vlms into vlas without catastrophic forgetting. Preprint, arXiv:2509.22195.
- Asher J. Hancock, Xindi Wu, Lihan Zha, Olga Russakovsky, and Anirudha Majumdar. 2025b. Actions as language: Fine-tuning vlms into vlas without catastrophic forgetting. Preprint, arXiv:2509.22195.

Xiaoshuai Hao, Lei Zhou, Zhijian Huang, Zhiwen Hou, Yingbo Tang, Lingfeng Zhang, Guang Li, Zheng Lu, Shuhuai Ren, Xianhui Meng, Yuchen Zhang, Jing Wu, Jinghui Lu, Chenxu Dang, Jiayi Guan, Jianhua Wu, Zhiyi Hou, Hanbing Li, Shumeng Xia, and 25 others. 2025. Mimo-embodied: X-embodied foundation model technical report. Preprint, arXiv:2511.16518.

Chi-Pin Huang, Yueh-Hua Wu, Min-Hung Chen, Yu-Chiang Frank Wang, and Fu-En Yang. 2025. Thinkact: Vision-language-action reasoning via reinforced visual latent planning. Preprint, arXiv:2507.16815.

Physical Intelligence, Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Manuel Y. Galliker, Dibya Ghosh, Lachy Groom, Karol Hausman, Brian Ichter, Szymon Jakubczak, Tim Jones, Liyiming Ke, Devin LeBlanc, and 17 others. 2025. π0.5: a visionlanguage-action model with open-world generalization. Preprint, arXiv:2504.16054.

Jingjing Jiang, Chao Ma, Xurui Song, Hanwang Zhang, and Jun Luo. 2025a. Corvid: Improving multimodal large language models towards chain-of-thought reasoning. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 3034–3046.

Tao Jiang, Tianyuan Yuan, Yicheng Liu, Chenhao Lu, Jianning Cui, Xiao Liu, Shuiqi Cheng, Jiyang Gao, Huazhe Xu, and Hang Zhao. 2025b. Galaxea open-world dataset and g0 dual-system vla model. Preprint, arXiv:2509.00576.

Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan P Foster, Pannag R Sanketi, Quan Vuong, Thomas Kollar, Benjamin Burchﬁel, Russ Tedrake, Dorsa Sadigh, Sergey Levine, Percy Liang, and Chelsea Finn. 2024. Openvla: An open-source vision-language-action model. In Annual Conference on Robot Learning (CoRL).

Jason Lee, Jiafei Duan, Haoquan Fang, Yuquan Deng, Shuo Liu, Boyang Li, Bohan Fang, Jieyu Zhang, Yi Ru Wang, Sangho Lee, Winson Han, Wilbert Pumacay, Angelica Wu, Rose Hendrix, Karen Farley, Eli VanderBilt, Ali Farhadi, Dieter Fox, and Ranjay Krishna. 2025. Molmoact: Action reasoning models that can reason in space. Preprint, arXiv:2508.07917.

Bohao Li, Rui Wang, Guangzhi Wang, Yuying Ge, Yixiao Ge, and Ying Shan. 2023a. Seed-bench: Benchmarking multimodal llms with generative comprehension. arXiv preprint arXiv:2307.16125.

Fuhao Li, Wenxuan Song, Han Zhao, Jingbo Wang, Pengxiang Ding, Donglin Wang, Long Zeng, and Haoang Li.

2025. Spatial forcing: Implicit spatial representation alignment for vision-language-action model. Preprint, arXiv:2510.12276.

Qixiu Li, Yaobo Liang, Zeyu Wang, Lin Luo, Xi Chen, Mozheng Liao, Fangyun Wei, Yu Deng, Sicheng Xu, Yizhong Zhang, Xiaofan Wang, Bei Liu, Jianlong Fu, Jianmin Bao, Dong Chen, Yuanchun Shi, Jiaolong Yang, and Baining Guo. 2024a. Cogact: A foundational vision-language-action model for synergizing cognition and action in robotic manipulation. Preprint, arXiv:2411.19650.

Xinghang Li, Peiyan Li, Minghuan Liu, Dong Wang, Jirong Liu, Bingyi Kang, Xiao Ma, Tao Kong, Hanbo Zhang, and Huaping Liu. 2024b. Towards generalist robot policies: What matters in building vision-language-action models. Preprint, arXiv:2412.14058.

Xuanlin Li, Kyle Hsu, Jiayuan Gu, Oier Mees, Karl Pertsch, Homer Rich Walke, Chuyuan Fu, Ishikaa Lunawat, Isabel Sieh, Sean Kirmani, Sergey Levine, Jiajun Wu, Chelsea Finn, Hao Su, Quan Vuong, and Ted Xiao. 2024c. Evaluating real-world robot manipulation policies in simulation. In Annual Conference on Robot Learning (CoRL).

Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. 2023b. Evaluating object hallucination in large vision-language models. Preprint, arXiv:2305.10355.

Shijie Lian, Bin Yu, Xiaopeng Lin, Laurence T. Yang, Zhaolong Shen, Changti Wu, Yuzhuo Miao, Cong Huang, and Kai Chen. 2026. Langforce: Bayesian decomposition of vision language action models via latent action queries. Preprint, arXiv:2601.15197.

Weixin Liang, LILI YU, Liang Luo, Srini Iyer, Ning Dong, Chunting Zhou, Gargi Ghosh, Mike Lewis, Wen tau Yih, Luke Zettlemoyer, and Xi Victoria Lin. 2025. Mixture-of-transformers: A sparse and scalable architecture for multi-modal foundation models. Transactions on Machine Learning Research.

Fanqi Lin, Ruiqian Nai, Yingdong Hu, Jiacheng You, Junming Zhao, and Yang Gao. 2025a. Onetwovla: A uniﬁed vision-language-action model with adaptive reasoning. Preprint, arXiv:2505.11917.

Tao Lin, Yilei Zhong, Yuxin Du, Jingjing Zhang, Jiting Liu, Yinxinyu Chen, Encheng Gu, Ziyan Liu, Hongyi Cai, Yanwen Zou, Lixing Zou, Zhaoye Zhou, Gen Li, and Bo Zhao. 2025b. Evo-1: Lightweight vision-languageaction model with preserved semantic alignment. Preprint, arXiv:2511.04555.

Xiaopeng Lin, Shijie Lian, Bin Yu, Ruoqi Yang, Changti Wu, Yuzhuo Miao, Yurun Jin, Yukun Shi, Cong Huang, Bojun Cheng, and Kai Chen. 2025c. Physbrain: Human egocentric data as a bridge from vision language models to physical intelligence. Preprint, arXiv:2512.16793.

Yixin Lin, Austin S. Wang, Giovanni Sutanto, Akshara Rai, and Franziska Meier. 2021. Polymetis. https: //facebookresearch.github.io/fairo/polymetis/.

Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, and Peter Stone. 2023a. Libero: Benchmarking knowledge transfer for lifelong robot learning. arXiv preprint arXiv:2306.03310.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2023b. Visual instruction tuning. In Advances in

neural information processing systems (NeurIPS). Ilya Loshchilov and Frank Hutter. 2019. Decoupled weight decay regularization. Preprint, arXiv:1711.05101. Pan Lu, Swaroop Mishra, Tony Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark,

and Ashwin Kalyan. 2022. Learn to explain: Multimodal reasoning via thought chains for science question answering. In The 36th Conference on Neural Information Processing Systems (NeurIPS).

Pietro Mazzaglia, Cansu Sancaktar, Markus Peschl, and Daniel Dijkman. 2025. Hybrid training for visionlanguage-action models. Preprint, arXiv:2510.00600.

Soroush Nasiriany, Abhiram Maddukuri, Lance Zhang, Adeet Parikh, Aaron Lo, Abhishek Joshi, Ajay Mandlekar, and Yuke Zhu. 2024. Robocasa: Large-scale simulation of everyday tasks for generalist robots. In Robotics: Science and Systems.

NVIDIA, :, Johan Bjorck, Fernando Castañeda, Nikita Cherniadev, Xingye Da, Runyu Ding, Linxi "Jim" Fan, Yu Fang, Dieter Fox, Fengyuan Hu, Spencer Huang, Joel Jang, Zhenyu Jiang, Jan Kautz, Kaushil Kundalia, Lawrence Lao, Zhiqi Li, Zongyu Lin, and 24 others. 2025. Gr00t n1: An open foundation model for generalist humanoid robots. Preprint, arXiv:2503.14734.

Abby ONeill, Abdul Rehman, Abhiram Maddukuri, Abhishek Gupta, Abhishek Padalkar, Abraham Lee, Acorn Pooley, Agrim Gupta, Ajay Mandlekar, Ajinkya Jain, and 1 others. 2024. Open x-embodiment: Robotic learning datasets and rt-x models: Open x-embodiment collaboration. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 6892–6903. IEEE.

William Peebles and Saining Xie. 2023. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 4195–4205.

Karl Pertsch, Kyle Stachowicz, Brian Ichter, Danny Driess, Suraj Nair, Quan Vuong, Oier Mees, Chelsea Finn, and Sergey Levine. 2025. Fast: Efﬁcient action tokenization for vision-language-action models. Preprint, arXiv:2501.09747.

Delin Qu, Haoming Song, Qizhi Chen, Yuanqi Yao, Xinyi Ye, Yan Ding, Zhigang Wang, JiaYuan Gu, Bin Zhao, Dong Wang, and Xuelong Li. 2025. Spatialvla: Exploring spatial representations for visual-language-action model. Preprint, arXiv:2501.15830.

Yichao Shen, Fangyun Wei, Zhiying Du, Yaobo Liang, Yan Lu, Jiaolong Yang, Nanning Zheng, and Baining Guo.

2025. Videovla: Video generators can be generalizable robot manipulators. Preprint, arXiv:2512.06963.

Mustafa Shukor, Dana Aubakirova, Francesco Capuano, Pepijn Kooijmans, Steven Palma, Adil Zouitine, Michel Aractingi, Caroline Pascal, Martino Russi, Andres Maraﬁoti, Simon Alibert, Matthieu Cord, Thomas Wolf, and Remi Cadene. 2025. Smolvla: A vision-language-action model for affordable and efﬁcient robotics. Preprint, arXiv:2506.01844.

starVLA. 2025. Starvla: A lego-like codebase for vision-language-action model developing. GitHub repository. Huajie Tan, Yuheng Ji, Xiaoshuai Hao, Xiansheng Chen, Pengwei Wang, Zhongyuan Wang, and Shanghang

Zhang. 2025. Reason-rft: Reinforcement ﬁne-tuning for visual reasoning of vision language models. Preprint, arXiv:2503.20752.

BAAI RoboBrain Team, Mingyu Cao, Huajie Tan, Yuheng Ji, Xiansheng Chen, Minglan Lin, Zhiyu Li, Zhou Cao, Pengwei Wang, Enshen Zhou, Yi Han, Yingbo Tang, Xiangqi Xu, Wei Guo, Yaoxu Lyu, Yijie Xu, Jiayu Shi, Mengfei Du, Cheng Chi, and 34 others. 2025a. Robobrain 2.0 technical report. Preprint, arXiv:2507.02029.

GEAR Team, Allison Azzolini, Johan Bjorck, Valts Blukis, Fernando Castañeda, Rahul Chand, and 1 others. 2025b. Gr00t n1.6: An improved open foundation model for generalist humanoid robots. https://research. nvidia.com/labs/gear/gr00t-n1_6/.

Octo Model Team, Dibya Ghosh, Homer Walke, Karl Pertsch, Kevin Black, Oier Mees, Sudeep Dasari, Joey Hejna, Tobias Kreiman, Charles Xu, Jianlan Luo, You Liang Tan, Lawrence Yunliang Chen, Pannag Sanketi, Quan Vuong, Ted Xiao, Dorsa Sadigh, Chelsea Finn, and Sergey Levine. 2024. Octo: An open-source generalist robot policy. Preprint, arXiv:2405.12213.

Selma Wanna, Agnes Luhtaru, Jonathan Salﬁty, Ryan Barron, Juston Moore, Cynthia Matuszek, and Mitch Pryor.

2026. Limited linguistic diversity in embodied ai datasets. Preprint, arXiv:2601.03136.

Junjie Wen, Yichen Zhu, Jinming Li, Minjie Zhu, Kun Wu, Zhiyuan Xu, Ning Liu, Ran Cheng, Chaomin Shen, Yaxin Peng, Feifei Feng, and Jian Tang. 2025. Tinyvla: Towards fast, data-efﬁcient vision-language-action models for robotic manipulation. Preprint, arXiv:2409.12514.

Yuxi Xiao, Longfei Li, Shen Yan, Xinhang Liu, Sida Peng, Yunchao Wei, Xiaowei Zhou, and Bingyi Kang. 2026. Spatialtree: How spatial abilities branch out in mllms. Preprint, arXiv:2512.20617.

Rui Yang, Ziyu Zhu, Yanwei Li, Jingjia Huang, Shen Yan, Siyuan Zhou, Zhe Liu, Xiangtai Li, Shuangye Li, Wenqian Wang, Yi Lin, and Hengshuang Zhao. 2025a. Visual spatial tuning. Preprint, arXiv:2511.05491.

Shusheng Yang, Jihan Yang, Pinzhi Huang, Ellis Brown, Zihao Yang, Yue Yu, Shengbang Tong, Zihan Zheng, Yifan Xu, Muhan Wang, Daohan Lu, Rob Fergus, Yann LeCun, Li Fei-Fei, and Saining Xie. 2025b. Cambrians: Towards spatial supersensing in video. Preprint, arXiv:2511.04670.

Cheng Yin, Yankai Lin, Wang Xu, Sikyuen Tam, Xiangrui Zeng, Zhiyuan Liu, and Zhouping Yin. 2025. Deepthinkvla: Enhancing reasoning capability of vision-language-action models. Preprint, arXiv:2511.15669.

Andy Zhai, Brae Liu, Bruno Fang, Chalse Cai, Ellie Ma, Ethan Yin, Hao Wang, Hugo Zhou, James Wang, Lights Shi, Lucy Liang, Make Wang, Qian Wang, Roy Gan, Ryan Yu, Shalfun Li, Starrick Liu, Sylas Chen, Vincent Chen, and Zach Xu. 2025. Igniting vlms toward the embodied space. Preprint, arXiv:2509.11766.

Jianke Zhang, Xiaoyu Chen, Qiuyue Wang, Mingsheng Li, Yanjiang Guo, Yucheng Hu, Jiajun Zhang, Shuai Bai, Junyang Lin, and Jianyu Chen. 2026. Vlm4vla: Revisiting vision-language-models in vision-language-action models. Preprint, arXiv:2601.03309.

Ziang Zhang, Zehan Wang, Guanghao Zhang, Weilong Dai, Yan Xia, Ziang Yan, Minjie Hong, and Zhou Zhao.

2025. Dsi-bench: A benchmark for dynamic spatial intelligence. Preprint, arXiv:2510.18873.

Jinliang Zheng, Jianxiong Li, Dongxiu Liu, Yinan Zheng, Zhihao Wang, Zhonghong Ou, Yu Liu, Jingjing Liu, Ya-Qin Zhang, and Xianyuan Zhan. 2025a. Universal actions for enhanced embodied foundation models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 22508– 22519.

Jinliang Zheng, Jianxiong Li, Zhihao Wang, Dongxiu Liu, Xirui Kang, Yuchun Feng, Yinan Zheng, Jiayin Zou, Yilun Chen, Jia Zeng, Ya-Qin Zhang, Jiangmiao Pang, Jingjing Liu, Tai Wang, and Xianyuan Zhan. 2025b. X-vla: Soft-prompted transformer as scalable cross-embodiment vision-language-action model. Preprint, arXiv:2510.10274.

Ruijie Zheng, Yongyuan Liang, Shuaiyi Huang, Jianfeng Gao, Hal Daumé III, Andrey Kolobov, Furong Huang, and Jianwei Yang. 2025c. Tracevla: Visual trace prompting enhances spatial-temporal awareness for generalist robotic policies. Preprint, arXiv:2412.10345.

Enshen Zhou, Jingkun An, Cheng Chi, Yi Han, Shanyu Rong, Chi Zhang, Pengwei Wang, Zhongyuan Wang, Tiejun Huang, Lu Sheng, and Shanghang Zhang. 2026. Roborefer: Towards spatial referring with reasoning in vision-language models for robotics. Preprint, arXiv:2506.04308.

Zhongyi Zhou, Yichen Zhu, Minjie Zhu, Junjie Wen, Ning Liu, Zhiyuan Xu, Weibin Meng, Ran Cheng, Yaxin Peng, Chaomin Shen, and Feifei Feng. 2025. Chatvla: Uniﬁed multimodal understanding and robot control with vision-language-action model. In Proceedings of the Empirical Methods in Natural Language Processing (EMNLP). Association for Computational Linguistics.

Chenming Zhu, Tai Wang, Wenwei Zhang, Jiangmiao Pang, and Xihui Liu. 2025. Llava-3d: A simple yet effective pathway to empowering lmms with 3d-awareness. Preprint, arXiv:2409.18125.

- A Real Robot Experiments Detail This section describes the training implementation and evaluation setup for our real-robot experiments.

##### A.1 Real Robot Pre-training

Table 7 presents the OXE subsets used for our pre-training.

- Table 7: Selected OXE Subsets for Pre-training. We curate 20 subsets from the Open X-Embodiment dataset that utilize end-effector position control. The table details the dataset source, the robot platform, and the number of episodes utilized.

Dataset Name Robot Episodes

Berkeley Bridge WidowX 25,460 RT-1 Robot Action Google Robot 79,499 DROID Franka 92,233 BC-Z Google Robot 39,350 Stanford HYDRA Franka 550 Austin VIOLA Franka 135 Berkeley Autolab UR5 UR5 896 NYU Franka Play Franka 456 Stanford Kuka Multimodal Kuka iiwa 3,000 Freiburg Franka Play Franka 3,242 USC Jaco Play Jaco 2 976 NYU VINN Hello Stretch 435 Austin BUDS Franka 50 UCSD Kitchen xArm 150 CMU Franka Pick-Insert Data Franka 520 Austin Mutex Franka 1,500 Berkeley Fanuc Manipulation Fanuc Mate 415 CMU Play Fusion Franka 576 CMU Stretch Hello Stretch 135 DobbE Hello Stretch 5,208

Total 254,786

##### A.2 Data Collection via Teleoperation

We use Polymetis (Lin et al., 2021) for teleoperation and collect human expert data to support imitation learning for VLA models. For real-world experiments, we utilize a Franka Research 3 robot arm equipped with two Intel RealSense D435 cameras. We collected 300 expert human demonstrations, following the instruction format: “pick up the {color} block and put it into the gray box”. The dataset is balanced, containing 100 trajectories for each of the three colors: red, green, and blue. A schematic illustration of the demonstration samples is presented in Figure 6.

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

Figure 6: Diagram of human expert data acquired through teleoperation

##### A.3 Fine Tuning on Human Expert Demonstration Samples

We perform ﬁne-tuning on the checkpoints pretrained on the OXE subset, using the human expert demonstration data we collected. The training is conducted on 8 × H100 GPUs with a per-device batch size of 16 for 80K steps.

- A.4 Real Robot Evaluation Our evaluation is divided into three settings: in-domain, out-of-domain and pick-all.

- • In-Domain. In this setting, the evaluation tasks use exactly the same objects and task instructions as those in the training set, with the positions of the blocks altered randomly.
- • Out-of-Domain. In this setting, the instructions for the evaluation tasks differ from those in the training set, primarily in terms of the color of the blocks to be grasped being altered. Speciﬁcally, the OOD evaluation task is deﬁned by the instruction: “pick up the yellow block and put it into the gray box”.
- • Pick-All. In this setting, the task instruction is distinct from those in the training set, requiring the robotic arm to place all blocks into the box. Speciﬁcally, the task is deﬁned by the instruction: “pick up all blocks and put them into the gray box“, with four blocks (red, blue, green, and yellow) present in the workspace. The model is required to comprehend the task semantics and achieve long-horizon generalization by composing atomic actions learned from the training dataset.

### B Implementation Details

We instantiate the VLM backbones by initializing the weights from Qwen2.5-VL-3B-Instruct and Qwen3-VL-4B-Instruct. The model is ﬁne-tuned for 40K steps on a cluster of 16 GPUs (batch size 16 per device). We employ the AdamW (Loshchilov and Hutter, 2019) optimizer initialized with a learning rate of 1e-5 and a cosine annealing schedule. System-level optimizations include DeepSpeed ZeRO-2 (Loshchilov and Hutter, 2019), gradient clipping at a norm of 1.0, and no gradient accumulation. Our training pipeline is built upon the starVLA (starVLA, 2025) framework.

For ﬁne-tuning, we set the default training duration to 40K steps. This choice aligns with standard practices in related work and ensures that the training can be completed within one day on NVIDIA H100 GPUs.

Our model architecture incorporates a state encoder designed to process proprioceptive information, such as joint angles and gripper states. However, since mainstream simulation benchmarks (e.g., SimplerEnv) primarily evaluate vision-only policies and do not typically involve state observations in their standard protocols, we omitted the state encoder during all simulation experiments to ensure fair comparison. Conversely, in real-robot settings where precise proprioception is crucial for control, the state encoder was active and utilized throughout both the pre-training and ﬁne-tuning phases.

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

Figure 7: Examples from the Bridge V2 dataset in the training set.

### C QwenGR00T, QwenPI and QwenOFT

For certain simulation experiments, we adopted the built-in baselines provided by StarVLA, reimplementing, training, and evaluating them under our uniform experimental setup. The architectures of these baselines are detailed in this section. As illustrated in Figure 8, these models differ primarily in their action modeling and VLM-policy interaction mechanisms:

- • QwenGR00T (Figure 8a): Incorporating a Flow Matching-based Action Expert, this model follows a late-fusion strategy (NVIDIA et al., 2025). The VLM functions as a high-level semantic encoder, passing only its last hidden states to the Action Expert. This design treats the VLM representation as a ﬁxed holistic condition for the ﬂow generation process.
- • QwenPI (Figure 8b): Also leveraging Flow Matching for continuous control, QwenPI adopts a deep-fusion architecture (Black et al., 2024). Unlike GR00T, it establishes dense connections between the VLM and the Action Expert, allowing for layer-wise attention interactions. This enables the policy to access and leverage multi-scale semantic features from the VLM’s intermediate layers during action generation.
- • QwenOFT (Figure 8c): Representing the discrete control paradigm (Kim et al., 2024), this model discretizes the action space into token bins and predicts actions autoregressively, treating manipulation as a standard next-token prediction task.

The implementations of the aforementioned architectures are all adopted from the built-in modules of the starVLA (starVLA, 2025) framework and are not claimed as our original contributions.

continuous action

continuous action

discrete action

Action Expert

Action Expert

VLM Backbone

noise

VLM Backbone

VLM Backbone

[Figure 217]

instruction

[Figure 218]

instruction noise

[Figure 219]

instruction

(a) QwenGR00T

(b) QwenPI

(c) QwenOFT

Figure 8: Architectural overview of VLA baselines with distinct fusion strategies. (a) QwenGR00T adopts a late-fusion design where a ﬂow-matching Action Expert is conditioned solely on the last hidden states of the VLM. (b) QwenPI employs a deep-fusion strategy, enabling the ﬂow-matching policy to interact with the VLM via layer-wise attention mechanisms. (c) QwenOFT follows the discrete token paradigm, predicting action bins autoregressively.

### D RoboCasa Tabletop Benchmark

The RoboCasa Tabletop benchmark comprises 24 distinct sub-tasks. We present the detailed success rates for all these tasks in Table 8.

#### E VLM Prompt Template This section presents the prompt templates used for VLM input during the VLA training process.

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

Figure 9: Schematic of RoboCasa Tabletop Evaluation.

##### System Prompt Template for Left Brain

You are a helpful robot brain that can understand images and texts.

##### System Prompt Template for Right Brain

You are a helpful robot brain that can understand images, texts, and robot states. You will be provided with observation, an instruction, and the robot state. Take action to execute the instruction.

##### User Prompt Template for Left Brain

<image> Instruction: {instruction}

##### User Prompt Template for Right Brain

<image> Instruction: {instruction} Robot Type: {robot_type} Predict the next action for the robot based on the observation, instruction and robot type.

### F Implementation Details on the LIBERO Benchmark

Training Dataset To evaluate the model’s multi-task generalization capabilities, we adopted a uniﬁed training strategy. We aggregated the training datasets from all four LIBERO task suites (LIBERO-Spatial, -Object, -Goal, and -Long) into a single combined dataset. Instead of training separate experts for each suite, we trained a single generalist model on this uniﬁed data.

Training Conﬁguration. We utilized Qwen3-VL-4B as the visual-language backbone. The model was trained on a cluster of 8 NVIDIA H100 GPUs. We set the per-device batch size to 16, resulting in a global batch size of 128. The training process was conducted for a total of 100,000 steps.

Evaluation Protocol. For evaluation, we strictly followed the ofﬁcial LIBERO evaluation protocol. We utilized the standard evaluation scripts provided by the benchmark. The performance was assessed across a total of 500 trials, consisting of 10 distinct tasks with 50 evaluation episodes per task. The ﬁnal reported result represents the average success rate across these 500 trials.

###### Table 8: Results of evaluating the VLA models with the GR1 robot in the RoboCasa Tabletop simulation environment. We highlight the best results in bold and the second-best results with underline.

Task Isaac-GR00TN1.6 QwenGR00T+ Qwen3VL + Qwen3VLQwenPI TwinBrainVLA+ Qwen2.5VL TwinBrainVLA+ Qwen3VL

PnP Bottle To Cabinet Close 51.5 46.0 26.0 62.0 74.0 PnP Can To Drawer Close 13.0 80.0 62.0 66.0 72.0 PnP Cup To Drawer Close 8.5 54.0 42.0 46.0 52.0 PnP Milk To Microwave Close 14.0 48.0 50.0 52.0 60.0 PnP Potato To Microwave Close 41.5 28.0 42.0 56.0 36.0 PnP Wine To Cabinet Close 16.5 46.0 32.0 58.0 46.0 PnP Novel From Cuttingboard To Basket 58.0 48.0 40.0 50.0 62.0 PnP Novel From Cuttingboard To Cardboardbox 46.5 40.0 46.0 44.0 46.0 PnP Novel From Cuttingboard To Pan 68.5 68.0 60.0 64.0 70.0 PnP Novel From Cuttingboard To Pot 65.0 52.0 40.0 60.0 66.0 PnP Novel From Cuttingboard To Tieredbasket 46.5 56.0 44.0 52.0 52.0 PnP Novel From Placemat To Basket 58.5 42.0 44.0 56.0 30.0 PnP Novel From Placemat To Bowl 57.5 44.0 52.0 59.0 54.0 PnP Novel From Placemat To Plate 63.0 48.0 50.0 62.0 64.0 PnP Novel From Placemat To Tieredshelf 28.5 18.0 28.0 28.0 38.0 PnP Novel From Plate To Bowl 57.0 60.0 52.0 62.0 60.0 PnP Novel From Plate To Cardboardbox 43.5 50.0 40.0 52.0 58.0 PnP Novel From Plate To Pan 51.0 54.0 36.0 56.0 56.0 PnP Novel From Plate To Plate 78.7 70.0 48.0 68.0 66.0 PnP Novel From Tray To Cardboardbox 51.5 38.0 34.0 40.0 46.0 PnP Novel From Tray To Plate 71.0 56.0 64.0 56.0 72.0 PnP Novel From Tray To Pot 64.5 50.0 44.0 60.0 56.0 PnP Novel From Tray To Tieredbasket 57.0 36.0 50.0 42.0 46.0 PnP Novel From Tray To Tieredshelf 31.5 16.0 28.0 34.0 28.0

Average 47.6 47.8 43.9 53.5 54.6

### G "Twin-to-One" via Knowledge Distillation

##### G.1 Motivation and Methodology

While TwinBrainVLA achieves superior performance by leveraging a dual-stream architecture to utilize general capabilities, the inclusion of a second VLM inevitably increases computational overhead during deployment. To bridge the gap between high-level capability utilization and deployment efﬁciency, we propose a "Twin-to-One" knowledge distillation framework.

Our core hypothesis is that the "Right Brain" of TwinBrainVLA, having dynamically fused semantic knowledge from the frozen "Left Brain" via AsyMoT, possesses a superior feature space compared to a standard single-stream VLA. Therefore, we treat the fully trained TwinBrainVLA as the Teacher and a standard Vanilla VLA as the Student. The goal is to force the Student to internalize the Teacher’s fused general capabilities by mimicking its internal representations.

Distillation Objective. We utilize the last hidden states of the Teacher’s Right Brain as the supervision signal. Formally, let Hteacher ∈ RL×D denote the ﬁnal hidden states of the Right Brain in TwinBrainVLA, and Hstudent ∈ RL×D denote the ﬁnal hidden states of the Student Vanilla VLA. In addition to the standard action prediction loss Laction, we introduce a feature alignment loss Lfeat:

∑L

1 L

∥Hstudent(i) − sg(Hteacher(i) )∥22 (9)

Lfeat =

i=1

where sg(·) indicates the stop-gradient operation, ensuring the Teacher remains ﬁxed. The total training objective for the Student is:

Ltotal = Laction + λLfeat (10) where λ is a balancing coefﬁcient (set to 1.0 in our experiments).

##### G.2 Experiment Setup

We conducted the distillation experiment on the SimplerEnv benchmark. The training data (BridgeV2 and Fractal subsets) and hyperparameters (e.g., learning rate, batch size, optimizer) are kept strictly identical to the main experiments described in Appendix B. The Student model architecture is identical to the Right Brain in TwinBrainVLA.

###### G.3 Results and Analysis The quantitative results are presented in Table 9.

- Table 9: "Twin-to-One" Distillation Results on SimplerEnv. We compare the distilled single-stream model (Student) against the standard Vanilla VLA and the Teacher (TwinBrainVLA). All models use the Qwen3-VL-4BInstruct backbone.

Method Spoon on Towel Carrot on Plate Stack Block Eggplant in Basket Average

|Baselines Vanilla VLA (Standard Training)|87.5 50.0 29.2 54.2|55.2|
|---|---|---|
|Teacher TwinBrainVLA (dual-stream)|87.5 58.3 33.3 79.1|64.5|
|"Twin-to-One" Distilled Student (single-stream)|83.3 54.2 29.2 66.7|58.4|

Performance Gains. As shown in Table 9, the Distilled Student achieves an average success rate of 58.4%, outperforming the standard Vanilla VLA (55.2%) by +3.2%.

Conclusion. These results substantiate that the performance gains of TwinBrainVLA stem from the high-quality, semantics-rich features fused from the Left Brain. Furthermore, it demonstrates that TwinBrainVLA can serve as an effective supervisor, enabling single-stream models to internalize general capabilities and break the ceiling of standard ﬁne-tuning paradigms.

### H Limitation and Future Work

Efﬁciency. The addition of a second VLM component in TwinBrainVLA leads to higher training and deployment costs compared to vanilla VLAs. While our approach prioritizes generalizability over efﬁciency, we acknowledge these overheads as a limitation. Nevertheless, it is encouraging that our architecture, consisting of two 4B models, outperforms a larger 8B-parameter model, demonstrating the effectiveness of this design.

Towards Deeper Semantic Fusion. Although our architecture successfully enables the retention and retrieval of general capabilities, how to seamlessly fuse this semantic information into downstream VLA tasks for true generalization is not yet fully resolved. This paper identiﬁes a viable pathway towards this goal, but achieving a deﬁnitive solution requires further exploration. We believe that more advanced mechanisms for integrating high-level semantics with low-level policies will be a critical area for future investigation.

