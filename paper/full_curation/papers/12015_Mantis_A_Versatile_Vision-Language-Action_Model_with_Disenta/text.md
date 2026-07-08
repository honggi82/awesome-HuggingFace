## Mantis: A Versatile Vision-Language-Action Model with Disentangled Visual Foresight

Yi Yang1,2 Xueqi Li2,3 Yiyang Chen1 Jin Song4 Yihan Wang5 Zipeng Xiao1 Jiadi Su5 You Qiaoben6 Pengfei Liu1,2 Zhijie Deng1,*

1SJTU 2SII 3SUSTech 4NJUPT 5FDU 6BOSCH

Code: https://github.com/SJTU-DENG-Lab/Mantis

# arXiv:2511.16175v2[cs.CV]23Feb2026

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

LIBERO Benchmark

[Figure 7]

[Figure 8]

[Figure 9]

Hi Mantis, how much is

[Figure 10]

DiT

96.7

[Figure 11]

[Figure 12]

100

the number of cubes

81.1

76.5

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

minus that of cups?

50

actions

Connector

[Figure 20]

220k Human

[Figure 21]

There are three cubes and

Videos

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

| |VLM<br><br>|
|---|---|
| | |

0

one cup, so the answer is two.

OpenVLA CoT-VLA Mantis

- 1

- 2

- 3

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

Convergence Speed

[Figure 34]

Who's on the magazine cover?

|SR(%)<br><br>epoch|
|---|

[Figure 35]

[Figure 36]

latent-act. queries

[Figure 37]

[Figure 38]

SR(%)

It's Iron Man, a Marvel superhero.

###### Disentangled Visual Foresight

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

I'm thirsty. Could you give me a hand?

76k Robot Demonstrations

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

Real World Experiments

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

Multi-gap Frame Prediction

Visual

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

10

7.83

7.25

Foresight

6.58

5

[Figure 68]

2.83

Yes

|𝜋0.5|
|---|

[Figure 69]

[Figure 70]

Action

[Figure 71]

[Figure 72]

Dynamic Patches

Prediction

[Figure 73]

ShareGPT4V GQA

0

COCO

[Figure 74]

OOD Instructions

ID

Dense inference

Instructions

Overlap?

VISION-FLAN

Cambrian Sherlock

…

38 Multimodal

[Figure 75]

Inference Counts

Basic Reasoning World Knowledge

Target Patches

LLaVA-Instruct

- 1

- 2

- 3

[Figure 76]

Datasets

TE

154.6

OpenOrca

No Sparse inference

Mantis

|× 50%|
|---|

Intent Detection

77.3

ATE

ALFWorld

… ALLaVA

Adaptive Temporal Ensemble

### Abstract

(DiT) head. With the current visual state provided to the DiT via a residual connection, a simple next-state prediction objective enables the meta queries to automatically capture the latent actions that delineate the visual trajectory, and hence boost the learning of explicit actions. The disentanglement reduces the burden of the VLA backbone, enabling it to maintain comprehension and reasoning capabilities through language supervision. Empirically, pretrained on human manipulation videos, robot demonstrations, and image-text pairs, Mantis achieves a 96.7% success rate on LIBERO benchmark after fine-tuning, surpassing powerful baselines while exhibiting high convergence speed. Real-world evaluations show that Mantis outperforms π0.5, a leading open-source VLA model, particularly in instruction-following capability, generalization to unseen instructions, and reasoning ability. We also introduce the adaptive temporal ensemble (ATE) strategy to balance com-

Recent advances in Vision-Language-Action (VLA) models demonstrate that visual signals can effectively complement sparse action supervisions. However, letting VLA directly predict high-dimensional visual states can distribute model capacity and incur prohibitive training cost, while compressing visual states into more compact supervisory signals inevitably incurs information bottlenecks. Moreover, existing methods often suffer from poor comprehension and reasoning capabilities due to the neglect of language supervision. This paper introduces Mantis, a novel framework featuring a Disentangled Visual Foresight (DVF) to tackle these issues. Specifically, Mantis decouples visual foresight prediction from the backbone with the combination of meta queries and a diffusion Transformer

*Corresponding author

putational efficiency and motion stability during inference, yielding the Mantis-ATE variant, which reduces inference counts by 50% while maintaining performance. Code and weights are released to support the open-source community.

### 1. Introduction

Robotic learning has witnessed remarkable progress, particularly in developing robust control strategies for diverse tasks across heterogeneous environments. Among the most promising approaches are Vision-Language-Action (VLA) models [7, 10, 25, 33, 40, 53, 58], which leverage pretrained Vision-Language Models (VLMs) [2, 3] to translate linguistic instructions and visual observations into executable robotic actions. Despite these advancements, existing VLA approaches face a fundamental challenge: the low-dimensional action signals can be too sparse to adequately supervise the large VLA model that processes highdimensional sensory inputs [27]. This mismatch leaves much of the model’s representational capacity underutilized, thereby constraining overall performance.

A remedy is to integrate the visual foresight prediction problem into VLA training, with the model asked to predict dense future visual states in addition to forecasting actions [43, 55]. However, high-dimensional visual states usually contain redundancies that distract the model from action prediction, resulting in high training costs and slow convergence during downstream fine-tuning [17]. Alternative approaches compress visual states into more compact supervisory signals [44, 51]. Yet, compression inevitably diminishes subtle variations among visual states that convey fine-grained motions, thereby creating an information bottleneck. Furthermore, existing methods often overlook language supervision, rendering the context understanding and reasoning capabilities of the model unguarded [57].

We propose Mantis, a novel VLA model featuring a Disentangled Visual Foresight (DVF) to address these issues. As illustrated in Figure 2, Mantis combines meta queries [34] with a Diffusion Transformer (DiT) [35] head to predict future visual states. Despite its simplicity, this design can effectively disentangle the tight coupling of action learning and foresight prediction, considering its efficacy in text-to-image generation tasks [12, 34]. Furthermore, we advocate feeding the current visual state also to the DiT through a residual connection [19], which enables the meta queries to automatically capture the inter-frame dynamics that delineate the visual trajectory. Intuitively, these queries extract latent actions which can facilitate more effective explicit action generation; we therefore term them latent-action queries. Then, we use learnable action queries to extract information from both the input and latent-action queries via causal attention. We apply an action head to map the extraction outcomes to explicit actions as in [54].

The disentangled design of Mantis reduces the representation burden on the VLA backbone, thereby freeing up more capacity for language supervision, which helps preserve the model’s semantic understanding and reasoning capabilities after action learning. To minimize the competition between learning signals from various modalities, we develop a progressive training recipe that incrementally incorporates additional modalities for smooth fusion. For inference, Mantis leverages the Temporal Ensemble [56] method for motion stability. Considering its inefficiency issues, we also propose Adaptive Temporal Ensemble (ATE), which dynamically adjusts the ensemble strength according to the motion stability demand of the current timestep. E.g., finegrained object manipulation can require much higher motion stability than unladen movements. We term the corresponding model variant as Mantis-ATE.

We pretrain Mantis based on three data sources: the SSV2 dataset [18] containing 220K human manipulation videos, the DROID dataset [24] with 76K robot demonstrations (covering video and action), and image–text pairs drawn from 38 multimodal datasets. We then evaluate Mantis on the LIBERO simulation benchmark [28] and in real-world settings. Extensive experiments validate our approach from multiple perspectives: (1) Mantis surpasses several strong baselines [8, 9, 32, 43, 44, 55], achieving 96.7% success on the LIBERO benchmark and showing faster convergence than previous visual foresight approaches [43]. (2) Experiments on the Agilex platform confirm Mantis’s robust instruction-following and generalization capabilities. Across three test scenarios, it accurately executes in-domain commands and effectively generalizes to out-of-domain instructions, outperforming the leading open-source VLA model π0.5 [21]. (3) The Mantis-ATE variant reduces inference calls by up to 50% while maintaining comparable task success rates.

In summary, our main contributions are three-folds:

- • We propose Disentangled Visual Foresight (DVF) that provides concise and instructive look-ahead cues for action prediction and construct a new VLA model Mantis.
- • We design a progressive training recipe for modality fusion, enabling Mantis to effectively integrate action prediction with language understanding.
- • Mantis achieves 96.7% success on the LIBERO benchmark and demonstrates superior instruction-following capabilities in real-world robot experiments.

### 2. Related Work

##### 2.1. Vision-Language-Action Models

The rapid evolution of vision-language models (VLMs) [2, 3, 29, 39, 47] has catalyzed the emergence of VisionLanguage-Action (VLA) models [7, 10, 25, 26, 33, 40– 42, 45, 53, 58]. These systems integrate a vision-language

(a) Visual Foresight

backbone with an action prediction component, extending their functional capabilities beyond pure perception. By leveraging the rich perceptual and linguistic representations of pretrained VLMs, VLA models enable robots to interpret and execute human commands with improved adaptability, transcending the limitations of purely reactive control policies. Despite this potential, existing VLA models suffer from limited understanding capabilities. Robot-specific training often overwrites the critical vision-text alignments acquired during pretraining [57], degrading instructionfollowing performance and undermining fundamental reasoning abilities. While some approaches employ explicit language assistance to mitigate this issue [38, 53], they are prone to error accumulation [17]. Other methods introduce implicit language supervision through auxiliary losses on language representations [13, 22, 57], but lack the visual foresight capabilities necessary for frame-level planning. In contrast, we propose a framework that incorporates visual foresight while simultaneously preserving semantic understanding and reasoning capabilities.

[Figure 77]

[Figure 78]

|𝑜𝑡+𝑛|
|---|

|𝑜𝑡|
|---|

[Figure 79]

[Figure 80]

(a) Visual Foresight (b) Track Guidance

[Figure 81]

VLA

[Figure 82]

[Figure 83]

|𝑜𝑡+𝑛|
|---|

|𝑎𝑡:𝑡+𝑛|
|---|

|𝑜𝑡+𝑛|
|---|

|𝑎𝑡:𝑡+𝑛|
|---|

[Figure 84]

|𝑎𝑡:𝑡+𝑛|
|---|

text

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

(b) Control-specific Information Guidance

###### VLA

###### VLA

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

|𝑜𝑡|
|---|

traj.

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

VLA

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

|𝑎𝑡:𝑡+𝑛|
|---|

text

[Figure 119]

[Figure 120]

text

text

𝑜𝑡

𝑜𝑡

(c) Latent Action Supervision

(c) Latent Action Supervision

[Figure 121]

[Figure 122]

latent act.

|𝑎𝑡:𝑡+𝑛|
|---|

latent act.

|𝑎𝑡:𝑡+𝑛|
|---|

VLA

VLA

Act. Quant.

Act. Quant.

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

text

[Figure 140]

[Figure 141]

[Figure 142]

text

|𝑜𝑡|
|---|

|𝑜𝑡+𝑛|
|---|

|𝑜𝑡|
|---|

𝑜𝑡 𝑜𝑡

𝑜𝑡+𝑛

##### 2.2. Vision-Augmented Action Learning

Figure 1. Vision-augmented action learning paradigms. (a) Visual Foresight enhances action prediction by forecasting future frames. (b) Track Guidance employs compressed visual state representations to guide action prediction. (c) Latent Action Supervision improves action learning through auxiliary latent actions.

Vision-augmented action learning effectively complements sparse action signals and better exploits the representational capacity of VLA models. We categorize vision-augmented action learning approaches as follows:

- 1) Visual Foresight. This approach enhances action pre-

diction by forecasting future frames, using either explicit or implicit methods. Explicit methods generate future frames through autoregressive discrete image tokens [32, 55] or learnable query tokens with a vision decoder [46, 54], which then inform action prediction. Implicit methods jointly train the VLA model on video generation and action prediction tasks, establishing an implicit connection between visual forecasting and action prediction [9, 23, 43]. However, pixel-level forecasting can introduce unnecessary information, distracting the model from action prediction and resulting in high training costs. Moreover, this approach may lead the model to mistakenly associate physical motion with visual appearances, such as texture or lighting changes, potentially resulting in hallucinations [16].

- 2) Track Guidance. To mitigate these limitations, sev-

eral studies compress visual states into compact, controloriented representations such as keypoint tracks [4, 5, 44]. These tracks capture essential physical dynamics and facilitate action prediction. Nonetheless, this compression can cause information bottlenecks, and the precision of pointtracking extracted from videos is often limited, leading to higher action prediction errors.

- 3) Latent Action Supervision. Other approaches em-

The VLA model is then trained to predict these latent actions before being fine-tuned on robot manipulation data. This strategy leverages the insight that inter-frame dynamics can be represented as action primitives that aid prediction. However, training an additional quantization model increases computational complexity. An overview of these vision-augmentation paradigms is provided in Fig. 1. In contrast to prior work, Mantis decouples visual foresight prediction from the backbone, yielding more compact and accurate auxiliary information for action prediction.

### 3. Methodology

This section provides an overview of Mantis, beginning with its model architecture and specifications, followed by detailed descriptions of the progressive training recipe and the adaptive temporal ensemble mechanism.

##### 3.1. Model Overview

As illustrated in Fig. 2, Mantis comprises the following components: a backbone model P, a connector C, a DVF head D, an action head π, a set of trainable latent-action queries [LAT], and action queries [ACT]. The task is specified by a language instruction l. At each time step t, the backbone P receives l and a visual state (e.g., a raw

ploy latent actions to supervise action prediction [8, 14, 51]. Typically, an action quantization model is first trained to learn discrete latent actions from inter-frame differences.

###### Progressive Training Recipe

#### Mantis Adaptive Temporal Ensemble

[Figure 143]

[Figure 144]

DVF

[Figure 145]

Target Patches Dynamic Patches

- 1 Multiple Gap Vision Training

- 2 Vision-Action Joint Training

Head

|𝑎𝑡:𝑡+𝑛|
|---|

[Figure 146]

[Figure 147]

DVF Head & Connector

𝑜𝑡+𝑛

Action Head Backbone

[Figure 148]

[Figure 149]

[Figure 150]

Action

Connector

Head

[GAP] [LAT] [ACT]

[Figure 151]

[Figure 152]

Overlap?

DVF Head

Yes No

Action Head Backbone

Language

###### AR Transformer

& Connector

[Figure 153]

[Figure 154]

[Figure 155]

Supervision

t = 0

t = 0

[GAP] [LAT] [ACT]

[Figure 156]

[Figure 157]

t = 1

t = 1

t = 2

2 Language Supervised Mix Training

t = 2

Instruction

[GAP] [ACT]

[LAT]

DVF Head & Connector

[Figure 158]

Action Head Backbone

[Figure 159]

[Figure 160]

[Figure 161]

|×[0.7, 0.2, 0.1]|
|---|

|× [0.74, 0.26]|
|---|

RGB

[GAP] [LAT] [ACT]

[Figure 162]

[Figure 163]

𝑜𝑡

- Figure 2. Left: Progressive training recipe. Mantis progressively integrates multiple modalities to achieve stable and well-balanced optimization. Center: Overview of Mantis. The framework consists of a backbone network, a DVF head, and an action head. The DVF head predicts future frames to facilitate latent action learning, thereby improving action prediction. Language supervision helps maintain the backbone’s capability for understanding and reasoning. Right: Adaptive Temporal Ensemble. Mantis-ATE dynamically adjusts the ensemble strength based on the overlap between target tokens and dynamic tokens.

##### 3.2. Model Specification

image frame) ot. These inputs, together with [LAT], are packed into a sequence and mapped as

VLM Backbone. Considering the robust understanding and reasoning capabilities of Qwen2.5-VL [2], we adopt it as our backbone. Qwen2.5-VL natively supports flexible input resolutions, allowing us to assign higher resolution to the primary camera while allocating lower resolution to the wrist camera, which captures less spatial detail.

ht = P(ot,l,[LAT]). (1)

ht are then concatenated with ot and fed into the connector C, which projects them into the conditional input of D to generate the future image frame ot+n with n as the time step gap between the current and future frames:

DVF Head. We employ Sana [48] as the DVF head due to its superior performance in text-to-image generation. Sana is a highly efficient DiT that integrates a deep compression autoencoder [11]. We define a connector comprising 12 transformer encoder layers and a projection layer that bridges the backbone’s output space with the DiT’s input space, following the Qwen2.5 LLM architecture [50] while using bidirectional attention. During action inference, the DVF head is omitted to reduce computational overhead, as visual state prediction is not required for robot execution.

ot+n = D(C(ot,ht)). (2)

This design prevents the VLA backbone from producing redundant visual information directly. Besides, feeding ot into D via a residual connection [19] enables the [LAT] queries to capture inter-frame dynamics that characterize the visual trajectory rather than reconstructing complete frames. These dynamics connect to latent actions—a visual manifestation of explicit robot motions—thereby providing targeted guidance for action prediction.

Action Head. Following [54], we employ a DiT-based action head for action prediction. Learnable action queries first aggregate information from input and latent-action queries via causal attention, then the action head is used to denoise Gaussian noise into an n-step action trajectory.

Next, the actions for the next n time steps can be generated by the action head π via

at:t+n = π(P(ot,l,[LAT],[ACT])), (3)

##### 3.3. Progressive Training Recipe

where the latent-action queries [LAT] is added to the context and the action queries [ACT] are used to extract information from the context.

Directly fusing vision, language, and action during pretraining can bias learning toward the easiest signal (e.g., actions) or overfit dominant modalities (e.g., language), causing cross-modal competition and unstable convergence. To address this, we adopt a progressive training recipe that introduces modalities in stages, promoting more stable optimization (Fig. 2, left). The training proceeds in three stages:

Furthermore, to produce denser visual predictions during training and accommodate diverse downstream tasks, we introduce multi-gap queries [GAP]. These queries are inserted before [LAT] to guide the generation of future frames at varying time step intervals.

Instruction：“Put both moka pots on the stove”

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

| |
|---|

| |
|---|

[Figure 169]

[Figure 170]

…

t = T t = T + 1 t = T + 2 t = T + 3 t = T + 4

- Figure 3. Visualization of multi-gap future frame generation.

| |
|---|

| |
|---|

- Stage 1: Multiple Gap Vision Training. We first train Mantis on videos without action annotations to predict future frames, encouraging the model to infer latent actions from visual dynamics. We use human manipulation videos so that Mantis learns general manipulation skills and broad world knowledge [51]. During training, we unfreeze the DVF head and latent-action queries, optimizing the diffu-

sion loss LDVF while keeping the backbone frozen to preserve its pretrained language representations. Additionally, the multi-gap queries are unfrozen to guide future frame generation across different time step intervals. Fig. 3 visualizes examples of multi-gap future frame generation.

- Stage 2: Vision-Action Joint Training. Then we introduce the action modality using a robot demonstration dataset. The time step gap is fixed to match the action chunk size for temporal alignment between visual and action streams.

The model is optimized with αLDVF + Laction, where Laction denotes the action head’s diffusion loss and α balances the terms. In this stage, we unfreeze the action queries while keeping the backbone frozen.

- Stage 3: Language Supervised Mix Training. To supervise the language modality, we jointly train on a collection of multimodal datasets together with the robot demonstration data. The backbone is unfrozen, and a cross-entropy loss Llang is applied to the language outputs. The overall objective is αLDVF + Laction + βLlang, where α and β weight the contributions of the vision and language losses. This progressive fusion yields stable and efficient optimization, producing a vision-augmented foundation with robust multimodal understanding for downstream tasks.

[Figure 171]

[Figure 172]

| |
|---|

t = 40

t = 95

attention scores heatmap cosine similarity heatmap target patches

dynamic patches

overlap patches

| |
|---|

| |
|---|

| |
|---|

Figure 4. Visualization of ATE. The attention heatmap uses darker colors to represent higher values, whereas in the cosine similarity heatmap the opposite holds. The parameters are set as τtarget = 1 and τdynamic = 12.

scores. (2) Dynamic patches correspond to regions that exhibit significant visual changes. To detect them, we divide the current and previous input images into patches aligned with the vision tokens, compute the cosine similarity between corresponding patches in pixel space, and select the top τdynamic% with the lowest similarity values.

Intuitively, dynamic patches capture the motion of the robotic arm and end-effector, while target patches highlight instruction-relevant objects. Overlap between them indicates fine-grained manipulations, such as grasping. As shown in Fig. 4, visualization of both patch types during Mantis inference supports this observation. When such overlap occurs, the Temporal Ensemble is activated to enhance motion stability; otherwise, it is disabled to improve computational efficiency. Integrating ATE into Mantis yields a more efficient variant, termed Mantis-ATE. For more details about ATE, please refer to Appendix A.

##### 3.4. Adaptive Temporal Ensemble

During inference, Mantis incorporates the commonly used Temporal Ensemble method [56] to enhance motion stability. Considering its high computational overhead, we also introduce an Adaptive Temporal Ensemble (ATE) strategy that dynamically adjusts the ensemble strength according to the motion stability required at each inference time step.

### 4. Experiments

##### 4.1. Implementation Details

Basic configuration. Mantis comprises 5.8 billion parameters in total: 3.7B in the backbone, 1.4B in the DVF head, 0.3B in the action head, and 0.3B in the VAE. We set the number of [LAT] to 9, [ACT] to 6, and [GAP] to 6 × 3 corresponding to time step gaps from 1 to 6. The diffusion process uses 30 steps for the DVF head and 10 steps

As shown in Fig. 2 (right), ATE maintains two sets of input vision patches at each time step. (1) Target patches denote image regions that are most relevant to the language instruction. Following [49], we compute text-to-vision attention scores from Mantis backbone’s cross-attention module and select the top τtarget% of vision tokens with the highest

Table 1. Comparison on the LIBERO benchmark. Mantis exhibits superior performance on 3 of 4 tasks and attains the highest average success rate compared to existing baseline methods, demonstrating the effectiveness of leveraging DVF for action prediction. Bold indicates the best performance, and Italics indicates the second-best performance.

###### Spatial Object Goal Long Avg.

Diffusion Policy [15] 78.3 92.5 68.3 50.5 72.4 OpenVLA [25] 84.7 88.4 79.2 53.7 76.5

non-visionaugmented

π0 [6] 96.8 98.8 95.8 85.2 94.2 π0-FAST [36] 96.4 96.8 88.6 60.2 85.5

NORA [20] 92.2 95.4 89.4 74.6 87.9

ATM [44] 68.5 68.0 77.8 39.3 63.4 CoT-VLA [55] 87.5 91.6 87.6 69.0 81.1 WorldVLA [9] 87.6 96.2 83.4 60.0 81.8

UniVLA [8] 96.5 96.8 95.6 92.0 95.2 UnifiedVLA [43] 95.4 98.8 93.6 94.0 95.5 DreamVLA [54] 97.5 94.0 89.5 89.5 92.6

visionaugmented

F1 [32] 98.2 97.8 95.4 91.3 95.7 Mantis (Ours) 98.8 99.2 94.4 94.2 96.7

for the action head. Training adopts AdamW [31] with 0.1 weight decay and 0.5 gradient clipping, and leverages DeepSpeed [37] for efficient distributed training.

|0<br><br>10<br><br>20<br><br>30<br><br>40<br><br>50<br><br>60<br><br>70<br><br>80<br><br>90<br><br>100<br><br>0 1 2 3 4 5 6 7 8 9 10 15 20<br><br>SR(%)<br><br>epoch|
|---|

Mantis

UniVLA

OpenVLA

Pretraining Setup. In Stage 1, the model is pretrained on the SSV2 dataset [18], which contains approximately 220K videos of human manipulations. The time step gap is randomly sampled between 1 and 6. In Stage 2, the model is pretrained on the DROID dataset [24], consisting of 76K robot episodes. The vision loss weight α is set to 0.1. In Stage 3, we introduce language supervision by jointly training on 38 multimodal datasets and DROID for 1.5 epochs.

ATM

UnifiedVLA

Finetuning Setup. For downstream finetuning on the LIBERO benchmark [28], we use the same learning rate configuration as in pretraining stages 1 and 2 and train for 30 epochs without language supervision. The vision loss weight is set to α = 0.1. The checkpoint with the highest validation success rate (SR) is selected for final evaluation. Further implementation details are provided in Appendix B.

Figure 5. Convergence speed comparison. Compared with traditional visual foresight methods such as UnifiedVLA [43], Mantis achieves significantly faster convergence speed, underscoring the necessity of decoupling foresight prediction from action learning.

##### 4.2. Simulation Experiments

We evaluate Mantis on the widely adopted LIBERO benchmark [28], which comprises four task suites: Spatial, Object, Goal, and Long. Each suite contains 10 tasks, with 50 trials per task to reduce random variation. Performance is measured using Success Rate (SR) from 0 to 100, where higher is better. For comprehensive comparison, we select high-performance VLA models both with [8, 9, 32, 43, 44, 54, 55] and without [6, 15, 20, 25, 36] vision augmentation as baselines. We also compare the convergence speed of Mantis against four vision-augmented baselines.

Main Results. As shown in Table 1, Mantis exhibits superior performance on 3 of 4 task suites and attains the highest average SR, outperforming previous vision-augmented and non-vision-augmented methods. This underscores the efficacy of leveraging DVF to enhance action prediction. Furthermore, vision-augmented methods mostly outperform their non-vision-augmented counterparts, corroborating the findings in [27] that dense visual states effectively complement sparse action signals. Notably, ATM [44] exhibits inferior performance, which we attribute to the limited accuracy of point trajectories extracted via video tracking methods, resulting in accumulated errors.

(a) (b)

[Figure 173]

Scenario 1 Scenario 2 Scenario 3

Wrist Camera Front Camera

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

| |
|---|

| |
|---|

ID: OOD:

“Put the cup on the female singer”

“Put the bread in the basket” “I’m hungry. Can you help?”

“Put the bear on the number 8”

“Put the cup on Taylor Swift”

“Put the bear on the number (3+5)”

(c) (d)

Mantis ID Mantis OOD 𝜋0.5 ID 𝜋0.5 OOD Task1 Task2 Task3 Task4

10

Mantis ID

8

Mantis OOD

6

4

|𝜋0.5 ID|
|---|

2

𝜋0.5 OOD

0

0 10 20 30

Scenario 1 Scenario 2 Scenario 3

- Figure 6. Real World Experiments. (a) The Agilex platform. (b) Scenario setups and example instructions. Each scenario shows one ID instruction and the corresponding OOD instruction. (c) Average success counts for Mantis and π0.5 on ID and OOD tasks across three scenarios. (d) Per-task success counts for Mantis and π0.5 in Scenario 1.

Convergence Speed. We evaluate the convergence speed of Mantis and four baseline methods representing different learning paradigms: non-vision-augmentation (OpenVLA [25]), visual foresight (UnifiedVLA [43]), track guidance (ATM [44]), and latent action supervision (UniVLA [8]). Each model is fine-tuned from its pretrained checkpoint for 20 epochs on the LIBERO Spatial suite following the prescribed configuration. Evaluation is conducted at each epoch, and the results are presented as an epoch-SR line graph in Fig. 5. As illustrated, Mantis demonstrates a relatively fast convergence speed, comparable to the non-vision-augmented OpenVLA and the latent action supervision method UniVLA. In contrast, the entangled visual foresight approach, UnifiedVLA, converges the slowest, maintaining a success rate of zero during the first ten epochs. This observation highlights the necessity of decoupling foresight prediction from action learning in order to achieve efficient optimization.

##### 4.3. Real World Experiments

We evaluate Mantis on an Agilex platform (Fig. 6 (a)) to investigate the effect of language supervision on preserving the backbone’s capabilities. We design three experimental scenarios, each containing four in-domain (ID) instructions to assess instruction following and four out-of-domain (OOD) instructions to evaluate generalization. Fig. 6 (b) shows the setup of the scenarios and provides example instructions. Effective generalization demands a nuanced un-

derstanding and reasoning. For example, the first scenario requires world knowledge (e.g., Taylor Swift), while the second involves basic arithmetic logic. We compare Mantis with π0.5 [21], a state-of-the-art open-source VLA model. Both are fine-tuned separately on the three scenarios, with Mantis maintaining language supervision throughout training. Each instruction is executed 10 times, and performance is reported as the average number of successful executions, allowing up to 5 consecutive attempts per trial.

As shown in Fig. 6 (c), Mantis consistently outperforms π0.5 on both ID and OOD instructions across all three scenarios, demonstrating its superior instruction-following capability, generalization to unseen instructions, and fundamental reasoning ability. In contrast, π0.5 exhibits subpar instruction-following ability and almost no generalization capability to OOD instructions (Fig. 6 (d)). This corroborates the efficacy of language supervision in preserving the backbone’s understanding and reasoning capability. Additionally, we activated the DVF head in certain experiments to collect the generated future frame sequences, as visualized in Fig. 7. Further details are provided in Appendix C.

##### 4.4. Ablation Study

ATE Analysis. We evaluated ATE’s impact on task execution speed by comparing the standard Mantis (TE) and Mantis-ATE across four LIBERO suites. We measured the average inference count (IC) and success rate (SR), where a lower IC indicates higher efficiency. The parameters are

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

Put the bowl on the plate

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

Put the cup on

the basketball player

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

Put the bread in the basket

Initial States Generated Future Frames Final States

- Figure 7. Visualization of Generated Future Frames. The last generated future frame closely mirrors the ground truth final state, substantiating the efficacy of the DVF in refining action prediction across diverse manipulation tasks.

TE SR ATE SR TE IC ATE IC

|0<br><br>50<br><br>100<br><br>150<br><br>200<br><br>250<br><br>300<br><br>90<br><br>92<br><br>94<br><br>96<br><br>98<br><br>100<br><br>spatial object goal long<br><br>98.6<br><br>98.8<br><br>103.6<br><br>136.5<br><br>117.8<br><br>260.5<br><br>94.4<br><br>99.2<br><br>94.2<br><br>58.3<br><br>98.8<br><br>60.8<br><br>86.4<br><br>92.8 92.4<br><br>103.8|
|---|

- Figure 8. Comparison between standard Mantis (TE) and Mantis-ATE. The primary vertical axis denotes success rate (SR), and the secondary vertical axis denotes inference count (IC).

Table 2. Comparison of four DVF variants. Bold indicates the best performance, Italics indicates the second-best performance.

Spatial Object Goal Long Avg.

vanilla-DVF 98.2 98.8 93.6 92.2 95.7 flawed-DVF 96.8 97.0 93.8 89.8 94.4

no-DVF 93.4 93.0 90.4 88.2 91.3 pretrained-DVF 98.4 99.0 94.0 93.2 96.2

worst. These results support the following conclusions: (a) DVF facilitates action learning, (b) the residual connection enables DVF to better capture latent actions, and (c) video pretraining further enhances the performance of DVF.

set as τtarget = 1 and τdynamic = 12. As shown in Fig. 8, Mantis-ATE reduces the IC by nearly 50% while maintaining comparable performance, demonstrating a substantial improvement in inference efficiency.

Language Supervision Ablations. We assessed the understanding and reasoning abilities of Mantis using multiple multimodal benchmarks and performed ablation studies without language supervision in real-world settings. The results are provided in Appendix D.

DVF Ablations. We evaluated the effect of the DVF through four architectural variants across four LIBERO suites: (1) vanilla-DVF, (2) flawed-DVF (DVF without the residual connection), (3) no-DVF (only the action head), and (4) pretrained-DVF (DVF pretrained on human and robot videos). The first three models were trained from scratch, while all four were trained for 30 epochs on each task suite. The best-performing checkpoint from each model was used for evaluation. The results, summarized in Table 2, show that the pretrained-DVF achieved the highest success rate (SR), followed by the vanilla-DVF, then the flawed-DVF, with the no-DVF configuration performing the

### 5. Conclusion, Limitations and Future Work

In this work, we present Mantis, a framework featuring Disentangled Visual Foresight (DVF). Simulation experiments demonstrate that DVF enhances performance and convergence speed, while real-world experiments validate the effectiveness of language supervision. Limitations include minor motion rollbacks in real-world scenarios due to missing robot state inputs. Future work will integrate richer inputs (e.g., 3D point clouds) and optimize inference speed.

### References

- [1] Xiang An, Yin Xie, Kaicheng Yang, Wenkang Zhang, Xiuwei Zhao, Zheng Cheng, Yirui Wang, Songcen Xu, Changrui Chen, Chunsheng Wu, et al. Llava-onevision-1.5: Fully open framework for democratized multimodal training. arXiv preprint arXiv:2509.23661, 2025. 1, 2
- [2] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025. 2, 4
- [3] Lucas Beyer, Andreas Steiner, Andr´e Susano Pinto, Alexander Kolesnikov, Xiao Wang, Daniel Salz, Maxim Neumann, Ibrahim Alabdulmohsin, Michael Tschannen, Emanuele Bugliarello, et al. Paligemma: A versatile 3b vlm for transfer. arXiv preprint arXiv:2407.07726, 2024. 2
- [4] Homanga Bharadhwaj, Debidatta Dwibedi, Abhinav Gupta, Shubham Tulsiani, Carl Doersch, Ted Xiao, Dhruv Shah, Fei Xia, Dorsa Sadigh, and Sean Kirmani. Gen2act: Human video generation in novel scenarios enables generalizable robot manipulation. arXiv preprint arXiv:2409.16283,

2024. 3

- [5] Homanga Bharadhwaj, Roozbeh Mottaghi, Abhinav Gupta, and Shubham Tulsiani. Track2act: Predicting point tracks from internet videos enables generalizable robot manipulation. In ECCV, pages 306–324. Springer, 2024. 3
- [6] Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, et al. π0 : A vision-languageaction flow model for general robot control. arXiv preprint arXiv:2410.24164, 2024. 6
- [7] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Joseph Dabis, Chelsea Finn, Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, Jasmine Hsu, et al. Rt-1: Robotics transformer for real-world control at scale. arXiv preprint arXiv:2212.06817, 2022. 2
- [8] Qingwen Bu, Yanting Yang, Jisong Cai, Shenyuan Gao, Guanghui Ren, Maoqing Yao, Ping Luo, and Hongyang Li. Learning to act anywhere with task-centric latent actions. arXiv preprint arXiv:2502.14420, 2025. 2, 3, 6, 7
- [9] Jun Cen, Chaohui Yu, Hangjie Yuan, Yuming Jiang, Siteng Huang, Jiayan Guo, Xin Li, Yibing Song, Hao Luo, Fan Wang, et al. Worldvla: Towards autoregressive action world model. arXiv preprint arXiv:2506.21539, 2025. 2, 3, 6
- [10] Chi-Lam Cheang, Guangzeng Chen, Ya Jing, Tao Kong, Hang Li, Yifeng Li, Yuxiao Liu, Hongtao Wu, Jiafeng Xu, Yichu Yang, et al. Gr-2: A generative video-language-action model with web-scale knowledge for robot manipulation. arXiv preprint arXiv:2410.06158, 2024. 2
- [11] Junyu Chen, Han Cai, Junsong Chen, Enze Xie, Shang Yang, Haotian Tang, Muyang Li, Yao Lu, and Song Han. Deep compression autoencoder for efficient high-resolution diffusion models. arXiv preprint arXiv:2410.10733, 2024. 4
- [12] Jiuhai Chen, Zhiyang Xu, Xichen Pan, Yushi Hu, Can Qin, Tom Goldstein, Lifu Huang, Tianyi Zhou, Saining Xie, Silvio Savarese, et al. Blip3-o: A family of fully open unified multimodal models-architecture, training and dataset. arXiv preprint arXiv:2505.09568, 2025. 2

- [13] William Chen, Suneel Belkhale, Suvir Mirchandani, Oier Mees, Danny Driess, Karl Pertsch, and Sergey Levine. Training strategies for efficient embodied reasoning. arXiv preprint arXiv:2505.08243, 2025. 3
- [14] Yi Chen, Yuying Ge, Weiliang Tang, Yizhuo Li, Yixiao Ge, Mingyu Ding, Ying Shan, and Xihui Liu. Moto: Latent motion token as the bridging language for learning robot manipulation from videos. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 19752– 19763, 2025. 3
- [15] Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, and Shuran Song. Diffusion policy: Visuomotor policy learning via action diffusion. The International Journal of Robotics Research, 44 (10-11):1684–1704, 2025. 6
- [16] Yilun Du, Sherry Yang, Bo Dai, Hanjun Dai, Ofir Nachum, Josh Tenenbaum, Dale Schuurmans, and Pieter Abbeel. Learning universal policies via text-guided video generation. Advances in neural information processing systems, 36:9156–9172, 2023. 3
- [17] Chongkai Gao, Zixuan Liu, Zhenghao Chi, Junshan Huang, Xin Fei, Yiwen Hou, Yuxuan Zhang, Yudi Lin, Zhirui Fang, Zeyu Jiang, et al. Vla-os: Structuring and dissecting planning representations and paradigms in vision-languageaction models. arXiv preprint arXiv:2506.17561, 2025. 2, 3
- [18] Raghav Goyal, Samira Ebrahimi Kahou, Vincent Michalski, Joanna Materzynska, Susanne Westphal, Heuna Kim, Valentin Haenel, Ingo Fruend, Peter Yianilos, Moritz Mueller-Freitag, et al. The” something something” video database for learning and evaluating visual common sense. In Proceedings of the IEEE international conference on computer vision, pages 5842–5850, 2017. 2, 6
- [19] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770–778, 2016. 2, 4
- [20] Chia-Yu Hung, Qi Sun, Pengfei Hong, Amir Zadeh, Chuan Li, U Tan, Navonil Majumder, Soujanya Poria, et al. Nora: A small open-sourced generalist vision language action model for embodied tasks. arXiv preprint arXiv:2504.19854, 2025. 6
- [21] Physical Intelligence, Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, et al. π0.5: a vision-language-action model with open-world generalization. arXiv preprint arXiv:2504.16054, 2025. 2, 7
- [22] Yuheng Ji, Huajie Tan, Jiayu Shi, Xiaoshuai Hao, Yuan Zhang, Hengyuan Zhang, Pengwei Wang, Mengdi Zhao, Yao Mu, Pengju An, et al. Robobrain: A unified brain model for robotic manipulation from abstract to concrete. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 1724–1734, 2025. 3
- [23] Yuming Jiang, Siteng Huang, Shengke Xue, Yaxi Zhao, Jun Cen, Sicong Leng, Kehan Li, Jiayan Guo, Kexiang Wang, Mingxiu Chen, et al. Rynnvla-001: Using human demonstrations to improve robot manipulation. arXiv preprint arXiv:2509.15212, 2025. 3

- [24] Alexander Khazatsky, Karl Pertsch, Suraj Nair, Ashwin Balakrishna, Sudeep Dasari, Siddharth Karamcheti, Soroush Nasiriany, Mohan Kumar Srirama, Lawrence Yunliang Chen, Kirsty Ellis, et al. Droid: A large-scale in-the-wild robot manipulation dataset. arXiv preprint arXiv:2403.12945, 2024. 2, 6
- [25] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, et al. Openvla: An open-source vision-language-action model. arXiv preprint arXiv:2406.09246, 2024. 2, 6, 7
- [26] Xinghang Li, Minghuan Liu, Hanbo Zhang, Cunjun Yu, Jie Xu, Hongtao Wu, Chilam Cheang, Ya Jing, Weinan Zhang, Huaping Liu, et al. Vision-language foundation models as effective robot imitators. arXiv preprint arXiv:2311.01378,

2023. 2

- [27] Yingyan Li, Shuyao Shang, Weisong Liu, Bing Zhan, Haochen Wang, Yuqi Wang, Yuntao Chen, Xiaoman Wang, Yasong An, Chufeng Tang, et al. Drivevla-w0: World models amplify data scaling law in autonomous driving. arXiv preprint arXiv:2510.12796, 2025. 2, 6
- [28] Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, and Peter Stone. Libero: Benchmarking knowledge transfer for lifelong robot learning. Advances in Neural Information Processing Systems, 36:44776–44791, 2023. 2, 6
- [29] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36:34892–34916, 2023. 2, 1
- [30] Yuliang Liu, Zhang Li, Mingxin Huang, Biao Yang, Wenwen Yu, Chunyuan Li, Xu-Cheng Yin, Cheng-Lin Liu, Lianwen Jin, and Xiang Bai. Ocrbench: on the hidden mystery of ocr in large multimodal models. Science China Information Sciences, 67(12):220102, 2024. 1
- [31] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017. 6
- [32] Qi Lv, Weijie Kong, Hao Li, Jia Zeng, Zherui Qiu, Delin Qu, Haoming Song, Qizhi Chen, Xiang Deng, and Jiangmiao Pang. F1: A vision-language-action model bridging understanding and generation to actions. arXiv preprint arXiv:2509.06951, 2025. 2, 3, 6
- [33] Yao Mu, Qinglong Zhang, Mengkang Hu, Wenhai Wang, Mingyu Ding, Jun Jin, Bin Wang, Jifeng Dai, Yu Qiao, and Ping Luo. Embodiedgpt: Vision-language pre-training via embodied chain of thought. Advances in Neural Information Processing Systems, 36:25081–25094, 2023. 2
- [34] Xichen Pan, Satya Narayan Shukla, Aashu Singh, Zhuokai Zhao, Shlok Kumar Mishra, Jialiang Wang, Zhiyang Xu, Jiuhai Chen, Kunpeng Li, Felix Juefei-Xu, et al. Transfer between modalities with metaqueries. arXiv preprint arXiv:2504.06256, 2025. 2
- [35] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205,

2023. 2

- [36] Karl Pertsch, Kyle Stachowicz, Brian Ichter, Danny Driess, Suraj Nair, Quan Vuong, Oier Mees, Chelsea Finn, and

- Sergey Levine. Fast: Efficient action tokenization for visionlanguage-action models. arXiv preprint arXiv:2501.09747, 2025. 6
- [37] Jeff Rasley, Samyam Rajbhandari, Olatunji Ruwase, and Yuxiong He. Deepspeed: System optimizations enable training deep learning models with over 100 billion parameters. In Proceedings of the 26th ACM SIGKDD international conference on knowledge discovery & data mining, pages 3505– 3506, 2020. 6
- [38] Qi Sun, Pengfei Hong, Tej Deep Pala, Vernon Toh, U-Xuan Tan, Deepanway Ghosal, and Soujanya Poria. Emma-x: An embodied multimodal action model with grounded chain of thought and look-ahead spatial reasoning. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 14199– 14214, 2025. 3
- [39] Chameleon Team. Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818, 2024. 2
- [40] Octo Model Team, Dibya Ghosh, Homer Walke, Karl Pertsch, Kevin Black, Oier Mees, Sudeep Dasari, Joey Hejna, Tobias Kreiman, Charles Xu, et al. Octo: An open-source generalist robot policy. arXiv preprint arXiv:2405.12213, 2024. 2
- [41] Yang Tian, Sizhe Yang, Jia Zeng, Ping Wang, Dahua Lin, Hao Dong, and Jiangmiao Pang. Predictive inverse dynamics models are scalable learners for robotic manipulation. arXiv preprint arXiv:2412.15109, 2024.
- [42] Yihao Wang, Pengxiang Ding, Lingxiao Li, Can Cui, Zirui Ge, Xinyang Tong, Wenxuan Song, Han Zhao, Wei Zhao, Pengxu Hou, et al. Vla-adapter: An effective paradigm for tiny-scale vision-language-action model. arXiv preprint arXiv:2509.09372, 2025. 2
- [43] Yuqi Wang, Xinghang Li, Wenxuan Wang, Junbo Zhang, Yingyan Li, Yuntao Chen, Xinlong Wang, and Zhaoxiang Zhang. Unified vision-language-action model. arXiv preprint arXiv:2506.19850, 2025. 2, 3, 6, 7
- [44] Chuan Wen, Xingyu Lin, John So, Kai Chen, Qi Dou, Yang Gao, and Pieter Abbeel. Any-point trajectory modeling for policy learning. arXiv preprint arXiv:2401.00025, 2023. 2, 3, 6, 7
- [45] Junjie Wen, Yichen Zhu, Jinming Li, Minjie Zhu, Zhibin Tang, Kun Wu, Zhiyuan Xu, Ning Liu, Ran Cheng, Chaomin Shen, et al. Tinyvla: Towards fast, data-efficient visionlanguage-action models for robotic manipulation. IEEE Robotics and Automation Letters, 2025. 2
- [46] Hongtao Wu, Ya Jing, Chilam Cheang, Guangzeng Chen, Jiafeng Xu, Xinghang Li, Minghuan Liu, Hang Li, and Tao Kong. Unleashing large-scale video generative pretraining for visual robot manipulation. arXiv preprint arXiv:2312.13139, 2023. 3
- [47] Yecheng Wu, Zhuoyang Zhang, Junyu Chen, Haotian Tang, Dacheng Li, Yunhao Fang, Ligeng Zhu, Enze Xie, Hongxu Yin, Li Yi, et al. Vila-u: a unified foundation model integrating visual understanding and generation. arXiv preprint arXiv:2409.04429, 2024. 2
- [48] Enze Xie, Junsong Chen, Junyu Chen, Han Cai, Haotian Tang, Yujun Lin, Zhekai Zhang, Muyang Li, Ligeng Zhu,

- Yao Lu, et al. Sana: Efficient high-resolution image synthesis with linear diffusion transformers. arXiv preprint arXiv:2410.10629, 2024. 4
- [49] Siyu Xu, Yunke Wang, Chenghao Xia, Dihao Zhu, Tao Huang, and Chang Xu. Vla-cache: Towards efficient visionlanguage-action model via adaptive token caching in robotic manipulation. arXiv preprint arXiv:2502.02175, 2025. 5
- [50] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025. 4
- [51] Seonghyeon Ye, Joel Jang, Byeongguk Jeon, Sejune Joo, Jianwei Yang, Baolin Peng, Ajay Mandlekar, Reuben Tan, Yu-Wei Chao, Bill Yuchen Lin, et al. Latent action pretraining from videos. arXiv preprint arXiv:2410.11758, 2024. 2, 3, 5
- [52] Shukang Yin, Chaoyou Fu, Sirui Zhao, Ke Li, Xing Sun, Tong Xu, and Enhong Chen. A survey on multimodal large language models. National Science Review, 11(12): nwae403, 2024. 1
- [53] Michał Zawalski, William Chen, Karl Pertsch, Oier Mees, Chelsea Finn, and Sergey Levine. Robotic control via embodied chain-of-thought reasoning. arXiv preprint arXiv:2407.08693, 2024. 2, 3, 1
- [54] Wenyao Zhang, Hongsi Liu, Zekun Qi, Yunnan Wang, Xinqiang Yu, Jiazhao Zhang, Runpei Dong, Jiawei He, Fan Lu, He Wang, et al. Dreamvla: a vision-language-action model dreamed with comprehensive world knowledge. arXiv preprint arXiv:2507.04447, 2025. 2, 3, 4, 6
- [55] Qingqing Zhao, Yao Lu, Moo Jin Kim, Zipeng Fu, Zhuoyang Zhang, Yecheng Wu, Zhaoshuo Li, Qianli Ma, Song Han, Chelsea Finn, et al. Cot-vla: Visual chain-of-thought reasoning for vision-language-action models. In CVPR, pages 1702–1713, 2025. 2, 3, 6
- [56] Tony Z Zhao, Vikash Kumar, Sergey Levine, and Chelsea Finn. Learning fine-grained bimanual manipulation with low-cost hardware. arXiv preprint arXiv:2304.13705, 2023.

- 2, 5

[57] Zhongyi Zhou, Yichen Zhu, Minjie Zhu, Junjie Wen, Ning Liu, Zhiyuan Xu, Weibin Meng, Yaxin Peng, Chaomin Shen, Feifei Feng, et al. Chatvla: Unified multimodal understanding and robot control with vision-language-action model. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 5377–5395, 2025. 2,

- 3, 1

- [58] Brianna Zitkovich, Tianhe Yu, Sichun Xu, Peng Xu, Ted Xiao, Fei Xia, Jialin Wu, Paul Wohlhart, Stefan Welker, Ayzaan Wahid, et al. Rt-2: Vision-language-action models transfer web knowledge to robotic control. In Conference on Robot Learning, pages 2165–2183. PMLR, 2023. 2

### A. Adaptive Temporal Ensemble

Both training stage 1 and 2 are trained for one epoch using a cosine learning rate schedule with 500 warm-up steps, a base learning rate of 1e − 4, and a minimum learning rate of 1e − 5. For stage 3, a fixed learning rate of 1e − 5 is used, with vision loss weight α = 0.1 and language loss weight β = 0.005. Our 38 language-supervision datasets are sourced from those used in the instructiontraining stage of LLaVA-OneVision-1.5-Instruct [1]. We include datasets covering visual question answering, OCR, embodied planning, and other general-purpose tasks, while excluding those focused on chart question answering, medical imaging, and other highly specialized domains. A complete list of the selected datasets is provided in Table 4.

When applying ATE, each input image is divided into 18 × 18 patches. Unless otherwise specified, we set τtarget = 1 and τdynamic = 12 for all experiments. We then analyze the computational complexity. For standard Mantis inference, the theoretical FLOPs for each backbone layer are:

FLOPs ≈ 4LD2 + 2L2D + 2LDM (4)

where L is the number of input tokens to the first backbone layer, D is the hidden-state dimension, and M denotes the feed-forward network (FFN) intermediate dimension. Dynamic-patch identification introduces an additional cost of O(LvDpatch), incurred by patch-similarity computations, where Lv is the number of image patches and Dpatch is the dimension of each patch. Target-patch selection further requires cross-modal attention aggregation, adding O(LtLvD), with Lt representing the number of text tokens. The thresholding step includes a sorting operation of complexity O(Lv log Lv). Empirically, ATE reduces the number of inferences by more than 40%, and the overhead from dynamic- and target-patch selection is negligible relative to the computation in Eq. 4. As a result, ATE delivers an effective acceleration of the inference process.

### C. Real World Experiments

All in-domain (ID) and out-of-domain (OOD) instructions for the three scenarios are summarized in Table 5. For each scenario, we collected 100 teleoperated demonstration episodes, with 25 episodes per task. Both Mantis and π0.5 were fine-tuned on the combined dataset containing all four tasks within each scenario. For each model, we adopted the same learning-rate settings used in pretraining stage 1 and trained for 10 epochs. Mantis was supervised with the LLaVA-Instruct dataset [29], whereas π0.5 did not use any language supervision. We set the vision loss weight to α = 0.1 and the language loss weight for Mantis to β = 0.1. Task-level success counts for both models across the three scenarios are reported in Table 6. Execution examples on real-world tasks is shown in Fig. 10.

### B. Implementation Details

For visual inputs, video frames and the robot’s primary camera images are cropped and resized to 512 × 512 pixels, while the robot’s wrist camera images are at 256×256.

### D. Language Supervision Ablations

- Table 3. Comparison on VQA and multimodal understanding benchmarks. Mantis achieves superior performance on 2 of the 3 benchmarks. Compared with the original backbone, its performance decreases only marginally. Bold denotes the best results.

To assess whether language supervision preserves the backbone’s capabilities, we conducted ablation studies comparing Mantis with other VLA models [53, 57] on several VQA and multimodal understanding benchmarks [30, 52]. The original Qwen2.5-VL model was also included for reference, and the results are shown in Table 3. Mantis achieves the best performance on two of the three benchmarks, with only a marginal drop relative to the original backbone, confirming the effectiveness of the language supervision.

MME OCRBench RealWorldQA Qwen2.5-VL 2217.3 807 62.1

ECoT 0 12 0 ChatVLA 1435.2 729 57.0 Mantis(Ours) 2070.2 757 56.9

In the Real-World Experiments, we also trained a language-unsupervised variant, Mantis-LU. Figure 9 compares its performance with the original Mantis on both indomain (ID) and out-of-domain (OOD) instructions across the three scenarios. While Mantis-LU retains reasonably strong performance on ID instructions, it performs poorly on OOD instructions, indicating that language supervision is crucial for instruction generalization.

Mantis ID Mantis OOD Mantis-LU ID Mantis-LU OOD

10

8

6

4

2

0

Scenario 1 Scenario 2 Scenario 3

Figure 9. Comparison between Mantis and Mantis-LU.

- Table 4. The 38 datasets used for Mantis language supervision. These datasets are sourced from the instruction-tuning data of LLaVAOneVision-1.5-Instruct [1] and cover general visual-language tasks while excluding specialized domains.

alfworld allava instruct laion4v allava instruct vflan4v allava cambrian coco Evol-Instruct-GPT4-Turbo gpt4o gpt4v gqa laion 220k infographic azuregpt4v llava cot 100k llava instruct llava wild llrv gpt4v magpie pro magpie ultra magpie ultra open orca orca 994k orca agentinstruct sharegpt4o sherlock vflan

CLEVR-Math image textualization Super-CLEVR wikipedia 2m textocr gpt4v sharegpt4v-part-00 sharegpt4v-part-01 sharegpt4v-part-03 sharegpt4v-part-04 sharegpt4v-part-05 sharegpt4v-part-06 sharegpt4v-part-07 sharegpt4v-part-08

- Table 5. In-distribution (ID) and out-of-distribution (OOD) instructions for the three scenarios. Each scenario comprises four ID instructions and four corresponding OOD instructions. The OOD instructions evaluate world knowledge, basic reasoning abilities, and understanding of human intent across the three scenarios, respectively.

ID instructions OOD instructions

- Scenario 1

- Task1 Put the cup on the female singer Put the cup on Taylor Swift

- Task2 Put the cup on the basketball player Put the cup on Michael Jordan

- Task3 Put the cup on the Marvel superhero Put the cup on Iron Man

- Task4 Put the cup on the English playwright Put the cup on Shakespeare

- Scenario 2

- Task1 Put the bear on the number 8 Put the bear on the number (3+5)

- Task2 Put the bear on the letter A Put the bear on the first letter

- Task3 Put the bear on the letter Z Put the bear on the last letter

- Task4 Put the bear on the number 7 Put the bear on the number (9-2)

- Scenario 3

- Task1 Put the bottle in the basket Put a thing that can quench thirst in the basket

- Task2 Put the Rubik’s Cube in the basket I want to play with something. Can you help?

- Task3 Put the bread in the basket I’m hungry. Can you help?

- Task4 Put the watch in the basket Put a thing that can tell the time in the basket

- Table 6. Task-level success counts. Mantis outperforms π0.5 on both in-domain (ID) and out-of-domain (OOD) instructions, demonstrating strong instruction-following and generalization capabilities through effective language supervision.

Scenario 1 2 3 Task 1 2 3 4 Avg. 1 2 3 4 Avg. 1 2 3 4 Avg.

ID 9 8 9 7 8.25 9 10 9 9 9.25 6 7 6 5 6 OOD 10 8 9 6 8.25 9 8 9 6 7.75 6 0 6 3 3.75

Mantis

ID 10 9 6 6 7.75 10 6 8 9 8.25 5 7 5 6 5.75

π0.5

OOD 10 4 0 0 3.5 10 0 0 0 2.5 0 10 0 0 2.5

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

Instruction: Put the cup on Taylor Swift

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

Instruction: Put the watch in the basket

###### Figure 10. Execution examples on real-world tasks.

