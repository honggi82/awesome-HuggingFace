# arXiv:2503.06955v2[cs.CV]12Mar2025

## Motion Anything: Any to Motion Generation

Zeyu Zhang1∗ Yiran Wang2∗ Wei Mao3 Danning Li4 Rui Zhao5 Biao Wu6 Zirui Song67 Bohan Zhuang8 Ian Reid7 Richard Hartley19 1ANU 2USYD 3Tencent 4McGill 5JD.com 6UTS 7MBZUAI 8ZJU 9Google

https://steve-zeyu-zhang.github.io/MotionAnything

[Figure 1]

Figure 1. Motion Anything is an any-to-motion method for generating high-quality, controllable human motion under multimodal conditions, including text queries, background music and a mix of both. Video demos are included in the supplementary material.

### Abstract

Conditional motion generation has been extensively studied in computer vision, yet two critical challenges remain. First, while masked autoregressive methods have recently outperformed diffusion-based approaches, existing masking models lack a mechanism to prioritize dynamic frames and body parts based on given conditions. Second, existing methods for different conditioning modalities often fail to integrate multiple modalities effectively, limiting control and coherence in generated motion. To address these challenges, we propose Motion Anything, a multimodal motion generation framework that introduces an Attention-

based Mask Modeling approach, enabling fine-grained spatial and temporal control over key frames and actions. Our model adaptively encodes multimodal conditions, including text and music, improving controllability. Additionally, we introduce Text-Music-Dance (TMD), a new motion dataset consisting of 2,153 pairs of text, music, and dance, making it twice the size of AIST++, thereby filling a critical gap in the community. Extensive experiments demonstrate that Motion Anything surpasses state-of-the-art methods across multiple benchmarks, achieving a 15% improvement in FID on HumanML3D and showing consistent performance gains on AIST++ and TMD.

∗Equal contribution.

Random Masking (Previous)

Random Masking Mask Restoration

### 1. Introduction

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

Transformer

Transformer

Mask

Mask

...

[Figure 6]

[Figure 7]

Human motion generation [85] has been widely explored in recent years due to its broad applications in film production, video gaming, augmented and virtual reality (AR/VR), and embodied AI for human-robot interaction. Recent advancements in conditional motion generation, including text-tomotion [22, 45, 64] and music-to-dance [36, 51] models have shown promising potential in 3D motion generation. These developments mark significant progress in generating motion sequences directly from textual descriptions and background music. However, despite extensive research in motion generation, the field still faces two significant challenges.

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

Condition

###### Attention-based Masking (Ours)

Key Frame Masking

Key Joints Masking

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

TemporalAdaptive

Rearrange

Rearrange

SpatialAligning

Transformer

Transformer

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

Attention-guided

Attention-guided

[Figure 25]

[Figure 26]

Condition

Condition

- (1) Recently, masked autoregressive methods [21, 44]

have shown a promising trend, outperforming diffusionbased methods [10, 53, 70]. However, existing masking models have been underexplored in generating motion that prioritizes dynamic frames and body parts in motion sequences based on given conditions.

- (2) Although specialized and multitask methods [5, 17,

Figure 2. Masking strategy comparison. This figure demonstrates the key differences between the previous random masking strategy [21] (top) and our attention-based masking (bottom). Our masking strategy focuses on the more significant and dynamic parts of the motion (colored) corresponding to the condition.

work that can seamlessly and adaptively encode multimodal conditions for more controllable motion generation. This fills the gap of multimodal conditioning in previous motion generation research and represents a significant improvement.

73, 83] exist for different conditioning modalities, they often overlook the importance of integrating multiple modalities to achieve more controllable generation, as shown in Table 1. For example, enhancing music-to-dance generation with precise text descriptions can improve control and coherence, whereas relying on only a single modality as a condition leads to underperformance.

- • For generating more controllable motion, we design an Attention-based Mask Modeling approach across both temporal and spatial dimensions, focusing on key frames and key actions corresponding to the condition. We further customize the mask transformer to adaptively handle different modalities of conditions, enhancing motion generation by integrating multimodal conditioning.
- • For exploring multi-conditioning motion generation, we introduce the new Text-Music-Dance (TMD) dataset, which includes 2,153 paired samples of text, music, and dance, making it twice as large as AIST++ [34]. We also conducted extensive experiments on standard benchmarks across multiple motion generation tasks. Our method achieved a 15% improvement in FID on HumanML3D [19] and consistent improvement on AIST++ [34] and TMD datasets.

Our motivation is to address these challenges by presenting an innovative method that tackles them in a nutshell. To overcome the first challenge, we designed a conditional masking approach within an autoregressive generation paradigm across both spatial and temporal dimensions, enabling the model to focus on key frames and actions corresponding to the given condition, as shown in Figure 2. The conditional masking strategy also dynamically adjusts based on the modality of the condition, whether it is text or music. To tackle the second challenge, we design our architecture to handle multimodal conditions adaptively and simultaneously. From a temporal perspective, our model aligns different input modalities to control motion generation in a time-sensitive manner. Meanwhile, from a spatial perspective, it maps action queries to specific body-part movements and aligns music genres with corresponding dance styles. Moreover, since multi-conditioning in motion generation is underexplored, there is no motion dataset with paired music and text available in the current community. Hence, we have curated a new motion dataset with paired music and text as a benchmark to help advance the community’s exploration of multimodal conditioning in motion generation.

### 2. Related Works

Text-to-Motion Generation. Recent advancements in human motion generation have skillfully combined diffusion and autoregressive models, achieving more realistic, versatile, and scalable motion synthesis. Foundational work like MDM [53] introduced a transformer-based diffusion approach for lifelike, text-driven motion generation. Expanding on this, MotionDiffuse [72] added refined control and diversity mechanisms, while MLD [10] boosted efficiency by operating within a latent space, reducing computational demands without sacrificing quality. Motion Mamba [78]

In general, our contributions can be summarized as follows:

• We present Motion Anything, an any-to-motion frame-

(a) Attention-based Temporal Masking

(b) Multimodal Motion Generation Architecture

(c) Attention-based Spatial Masking

(d) A Block of Motion Generator

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

VQ-VAE Decoder

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

(spatial)

[Figure 38]

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

rearrange

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

rearrange

[Figure 54]

[Figure 55]

Spatial Aligning Transformer

[Figure 56]

Spatial Aligning Transformer

Key Frame Masking

Key Joint Masking

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

Temporal Adaptive Transformer

[Figure 62]

[Figure 63]

Temporal Attention

Spatial Attention

...

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

T2M:

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

Q KV

[Figure 79]

T2M:

[Figure 80]

T2M:

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

Spatial Aligning Transformer

M2D:

QKV M2D:

Q

KV M2D:

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

KV TM2D:

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

Q

Linear

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

Q KV TM2D:

[Figure 114]

Q KV TM2D:

[Figure 115]

rearrange

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

Q KV

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

Q KV

Q KV

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

(spatial masking)

Attention-based Key Frame Selection

[Figure 143]

Attention-based Key Joint Selection

Temporal Adaptive Transformer

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

(spatial)

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

rearrange

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

(temporal)

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

Temporal Adaptive Transformer

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

Audio Motion Sequence Encoder

VQ-VAE CLIP Encoder

[Figure 182]

Motion Sequence

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

M2D:

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

A man bending down picking up something, then raise up left hand, then walking forward while holding armrest.

[Figure 194]

T2M:

Embeddings:

Q KV TM2D:

Mask Token Motion Mask Restoration

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

QKV

[Figure 202]

[Figure 203]

[Figure 204]

Text

Music

Q KV

Figure 3. Motion Anything architecture. The multimodal architecture consists of several key components: (a) temporal and (c) spatial attention-based masking, (b) motion generator, and (d) a single block of motion generator. These components enable the model to learn key motions corresponding to the given conditions, and facilitate alignment between multi-modal conditions and motion features.

Models Text-to-Motion Music-to-Dance Text and Music to Dance TM2D [17] ✓ ✓ ✗

Music-to-Dance Generation. Recent work in musicdriven dance generation has leveraged autoregressive and diffusion-based models to achieve more synchronized, diverse, and controllable dance motions. TSMT [33] pioneered using transformer architectures to model complex dance motions. Subsequently, early methods like DanceNet [86] and Dance Revolution [24] customize autoregressive and sequence-to-sequence models to establish foundational mappings between music and movement. FACT [34] and Bailando [51] build upon this by incorporating 3D motion data and actor-critic memory models to capture richer choreography, and Bailando++ [52] enhances this framework further for refined generation quality. EDGE [54] introduces user-editable dance generation for greater customization. In recent work, Lodge [36] and Lodge++ [35] apply coarse-to-fine diffusion methods to extend sequence length and create vivid choreography patterns, while BeatIt [26] achieves beat-synchronized dance generation under multiple musical conditions. Lastly, the BADM [66] merges autoregressive and diffusion models, producing coherent, music-aligned dance sequences. Together, these works illustrate the field’s progression toward adaptable, high-fidelity dance generation tightly integrated with musical features.

UDE [83] ✓ ✓ ✗ UDE-2 [84] ✓ ✓ ✗

MoFusion [12] ✓ ✓ ✗ MCM [39] ✓ ✓ ✓ LMM [73] ✓ ✓ ✗

MotionCraft [5] ✓ ✓ ✗

MagicPose4D [67] ✗ ✗ ✗ STAR [8] ✓ ✗ ✗ TC4D [3] ✓ ✗ ✗

Motion Avatar [77] ✓ ✗ ✗ Motion Anything (Ours) ✓ ✓ ✓

- Table 1. Methods comparison. Either single-task or multi-task models can handle only one condition at a time, overlooking the importance of integrating multiple modalities for more controllable generation. Our Motion Anything introduces an innovative approach that encodes different modalities simultaneously and adaptively for more controllable generation.

addressed the challenge of generating longer sequences, and ReMoDiffuse [70] further enriched motion variability by incorporating retrieval-augmented diffusion. Meanwhile, autoregressive models like MoMask [21] enhanced temporal coherence through generative masked modeling, selectively revealing segments of the motion sequence. BAMM [44] introduced a bidirectional model to capture detailed motion with forward and backward dependencies. InfiniMotion [76] optimized transformer memory to support extended sequences, and KMM [75] prioritized essential frames to balance continuity and computational efficiency. MoGenTS [64] added spatial-temporal joint modeling for further structural consistency in generated motions.

Multi-Task Motion Generation. Human motion generation has evolved through multi-modal approaches, enabling contextually adaptive synthesis across diverse inputs like music, text, and visual cues. TM2D [17] introduced a bi-

[Figure 205]

[Figure 206]

[Figure 207]

modal framework integrating music and text for 3D dance generation, using VQ-VAE to encode motion in a shared latent space for flexible control. MotionCraft [5] builds on this by offering whole-body motion generation with adaptable multi-modal controls, from high-level semantics to specific joint details. MCM [39] further employs a transformer-based model to process varied inputs—text, audio, and video—generating motion that reflects both style and context. LMM [73] extends these capabilities by integrating large-scale pre-trained models for complex human motion across modalities. Meanwhile, UDE [83] provides a cohesive framework for motion synthesis, and UDE-2 [84] expands this to synchronize multi-part, multi-modal movements. MoFusion [12] complements these advances with a diffusion-based denoising framework focused on robustness and quality in diverse motion styles. These works collectively guide the field toward adaptive, multi-modal, and context-aware human motion generation.

Q(Text,Motion)

Q(Music,Text)

Q(Music)

K (Text, Motion)

K (Motion)

K (Motion)

- (1) Temporal Attention Map
- (2) Spatial Attention Map

[Figure 208]

[Figure 209]

[Figure 210]

Q(Music,Text)

Q(Music)

Q(Text)

K (Motion)

K (Motion)

K (Motion)

Figure 4. Attention map. The attention map provides a direct visualization of our attention-based masking approach, which selectively masks regions in the motion sequence with high attention scores.

### 3. Methodology

audio, or a combination of both. This process highlights specific regions in the attention map, indicating the key motions. We designed attention-based masking on both temporal and spatial dimensions to ensure the model focuses on learning key frames and joints in the motion sequence that correspond to the conditions, as shown in Figure 3 (a) and (c). This enables the model to learn more robust motion representations compared to traditional random masking [21, 64].

#### 3.1. Overview

Motion Anything presents an innovative any-to-motion approach that generates controllable human motion by focusing on the dynamic and significant parts of human motion sequences and adaptively aligning with different condition modalities. As shown in Figure 3, Motion Anything can take different modalities either separately or simultaneously, enabling multimodal conditioning to enhance controllable motion generation instead of relying on a single condition. The conditions are first encoded by text and audio encoders, then used to guide both masking and motion generation. We propose an attention-based masking approach that identifies the most significant parts of the motion corresponding to the conditions across both spatial and temporal dimensions by selecting high-attention scores as masking guidance. The guided mask tokens, along with condition embeddings, are then fed into a masked transformer for guided mask restoration. We customize masked transformers into a Temporal Adaptive Transformer and a Spatial Aligning Transformer to adaptively align overall control and specific actions to the motion sequence.

As shown in Algorithm 1, given a motion sequence M, a condition C, and a masking ratio α, the model masks the top α% of attention scores, which represent the most important motions that are also most relevant to the corresponding condition.

Algorithm 1 Attention-based Masking

- 1: Input: Motion M, Condition C, Masking Ratio α
- 2: Define: T: Text space, D: Audio space
- 3: Step 1: Define temporal Qtemp,Ktemp,Vtemp and spatial Qspatial,Kspatial,Vspatial
- 4: if C ∈ T then
- 5: Qtemp,Ktemp,Vtemp ← (C,M),(C,M),(C,M)
- 6: Qspatial,Kspatial,Vspatial ← C,M,M
- 7: else if C ∈ D or C ∈ T ∩ D then
- 8: Qtemp,Ktemp,Vtemp ← C,M,M
- 9: Qspatial,Kspatial,Vspatial ← C,M,M
- 10: end if
- 11: Step 2: Compute Attention Scores
- 12: Atemp = Attention(Qtemp,Ktemp,Vtemp)
- 13: Aspatial = Attention(Qspatial,Kspatial,Vspatial)
- 14: Step 3: Apply Masking
- 15: Sort Atemp and mask top α percent: masktemp = {i | Atemp,i in top α%}
- 16: Sort Aspatial and mask top α percent: maskspatial = {i | Aspatial,i in top α%}
- 17: Output: Masked motion sequence Mmasked

#### 3.2. Architecture

Attention-based Masking. The core of attention-based masking involves guiding the condition modality to select key frames in the temporal dimension and key actions in the spatial dimension, allowing for the masking of these motions. As shown in the attention map in Figure 4, both temporal and spatial attention rely on either self-attention or cross-attention [56], depending on the condition modality. The condition serves as query Q, and motion serves as key K and value V , where the condition can be text,

Temporal Adaptive Transformer. The Temporal Adaptive Transformer (TAT) aligns the temporal tokens of the motion sequence with the temporal condition by dynamically adjusting its attention calculation according to the modality of the condition. This enables the TAT to align key frames of motion with keywords in text and beats in music.

As shown in Algorithm 2 and Figure 3 (d), after

attention-based masking, the key frames of the motion sequence are masked. The TAT then learns the motion representation by restoring the masked frames with guidance from the condition. If the condition consists only of text, it contains a single token in the temporal dimension from CLIP, making self-attention in the temporal dimension more suitable. Otherwise, the motion sequence serves as Q, and the condition serves as KV , performing cross-attention to align the temporal information of the motion with music or the combination of music and text. This enables the Temporal Adaptive Transformer to become more adaptable and robust for different modalities of input conditions.

- Algorithm 2 Temporal Adaptive Transformer

- 1: Input: Motion M, Condition C, Mask Ratio α
- 2: Define: T: Text space, A: Audio space
- 3: Step 1: Apply Attention-based Masking to M
- 4: Mmasked ← Attention-based Masking(M,C,α)
- 5: Step 2: Define Q, K, V
- 6: if C ∈ T then
- 7: Q,K,V ← (C,Mmasked),(C,Mmasked),(C,Mmasked)
- 8: else if C ∈ A or C ∈ T ∩ A then
- 9: Q,K,V ← Mmasked,C,C
- 10: end if
- 11: Step 3: Compute Attention Scores
- 12: Mrestored = Attention(Q,K,V )
- 13: Output: Restored Motion Sequence Mrestored

Spatial Aligning Transformer. In the Spatial Aligning Transformer (SAT), both the condition and motion embeddings are rearranged to expose the spatial dimension. As shown in Algorithm 3 and Figure 3 (d), during attentionbased masking, the key action in each frame, which refers to the key motion of a specific body part in the spatial dimension, is masked. The SAT restores this feature with the guidance of the spatial condition. Aligning the spatial pose in each frame with the spatial condition is essential, especially in text-to-motion generation, where certain keywords describe specific body parts. In music-to-dance generation, the spectrum of each audio frame indicates the music genre [32, 55], which is crucial for generating the appropriate type of dance.

- Algorithm 3 Spatial Aligning Transformer

- 1: Input: Motion M′, Condition C′, Mask Ratio α
- 2: Step 1: Apply Attention-based Masking to M′
- 3: M′masked ← Attention-based Masking(M′,C′,α)
- 4: Step 2: Define Q, K, V
- 5: Q,K,V ← M′masked,C′,C′
- 6: Step 3: Compute Attention Scores
- 7: M′restored = Attention(Q,K,V )
- 8: Output: Restored Motion Sequence M′restored

- 4. Experiments

#### 4.1. Datasets and Evaluation Matrices

TMD Dataset. Our Text-Music-Dance (TMD) dataset introduces a pioneering benchmark with 2,153 pairs of text, music, and motion. We extract dance motions and corresponding text annotations from Motion-X [38], including

AIST++ [34] and other datasets. For motion-text pairs without music, we generate corresponding music by implementing Stable Audio Open [14] with beat adjustment and evaluate the generated music through human expert assessments, ensuring inter-rater reliability.

Public Benchmarks. To ensure a fair comparison, we evaluate our method against both specialized and unified motion generation approaches on standard benchmarks including HumanML3D [19] and KIT-ML [46] for text-to-motion generation, and AIST++ [34] for music-to-dance generation.

Evaluation Matrices. We adapt standard evaluation metrics to assess various aspects of our experiments. For text-to-motion generation, we implement FID and R precision to quantify the realism and robustness of generated motions, MultiModal Distance to measure motiontext alignment, and the diversity metric to calculate variance in motion features. Additionally, we apply the multimodality (MModality) metric to evaluate diversity among motions sharing the same text description. For music-todance generation, we follow AIST++ [34] to evaluate generated dances from three perspectives: quality, diversity, and music-motion alignment. For quality, we calculate FID between the generated dance and motion sequence features (kinetic, FIDk, and geometric, FIDg) using the toolbox in [18]. For diversity, we compute the average feature distance as in AIST++ [34]. For alignment, we calculate the Beat Align Score (BAS) as the average temporal distance between music beats and their closest dance beats.

#### 4.2. Model and Implementation Details

Our model consists of 2 TAT and 2 SAT layers, with 12.65M parameters and 137.35 GFLOPs. The learning rate increases to 2 × 10−4 after 2000 iterations using a linear warm-up schedule for all models. The mini-batch size is set to 512 for training the VQ-VAE tokenizer and 64 for training the masked transformers. All experiments were conducted on an Intel Xeon Platinum 8360Y CPU at 2.40GHz, equipped with a single NVIDIA A100 40GB GPU and 32GB of RAM.

#### 4.3. Comparative Study

Text-to-Motion. We compared our method with other state-of-the-art approaches on both HumanML3D [19] and KIT-ML [46]. The results in Table 2 demonstrate that our method consistently outperforms specialized text-to-motion models and surpasses recent multi-task methods.

Music-to-Dance. To highlight the music-to-dance capability of our method, we conducted evaluations on AIST++ [34]. The results in Table 3 indicate that our method surpasses previous state-of-the-art specialized and unified approaches, demonstrating superior motion quality, enhanced diversity, and better beat alignment in music-to-dance gen-

R Precision ↑ FID↓ MultiModal Dist↓ Diversity→ MultiModality↑ Top 1 Top 2 Top 3

Datasets Method

Ground Truth 0.511±.003 0.703±.003 0.797±.002 0.002±.000 2.974±.008 9.503±.065 TM2D [17] 0.319±.000 - - 1.021±.000 4.098±.000 9.513±.000 4.139±.000 MotionCraft [5] 0.501±.003 0.697±.003 0.796±.002 0.173±.002 3.025±.008 9.543±.098 ReMoDiffuse [70] 0.510±.005 0.698±.006 0.795±.004 0.103±.004 2.974±.016 9.018±.075 1.795±.043 MMM [45] 0.504±.003 0.696±.003 0.794±.002 0.080±.003 2.998±.007 9.411±.058 1.164±.041 DiverseMotion [41] 0.515±.003 0.706±.002 0.802±.002 0.072±.004 2.941±.007 9.683±.102 1.869±.089 BAD [22] 0.517±.002 0.713±.003 0.808±.003 0.065±.003 2.901±.008 9.694±.068 1.194±.044 BAMM [44] 0.525±.002 0.720±.003 0.814±.003 0.055±.002 2.919±.008 9.717±.089 1.687±.051 MCM [39] 0.502±.002 0.692±.004 0.788±.006 0.053±.007 3.037±.003 9.585±.082 0.810±.023 MoMask [21] 0.521±.002 0.713±.002 0.807±.002 0.045±.002 2.958±.008 - 1.241±.040 LMM [73] 0.525±.002 0.719±.002 0.811±.002 0.040±.002 2.943±.012 9.814±.076 2.683±.054 MoGenTS [64] 0.529±.003 0.719±.002 0.812±.002 0.033±.001 2.867±.006 9.570±.077 Motion Anything (Ours) 0.546±.003 0.735±.002 0.829±.002 0.028±.005 2.859±.010 9.521±.083 2.705±.068

Human ML3D [19]

Ground Truth 0.424±.005 0.649±.006 0.779±.006 0.031±.004 2.788±.012 11.08±.097 -

ReMoDiffuse [70] 0.427±.014 0.641±.004 0.765±.055 0.155±.006 2.814±.012 10.80±.105 1.239±.028 MMM [45] 0.404±.005 0.621±.005 0.744±.004 0.316±.028 2.977±.019 10.91±.101 1.232±.039 DiverseMotion [41] 0.416±.005 0.637±.008 0.760±.011 0.468±.098 2.892±.041 10.87±.101 2.062±.079 BAD [22] 0.417±.006 0.631±.006 0.750±.006 0.221±.012 2.941±.025 11.00±.100 1.170±.047 BAMM [44] 0.438±.009 0.661±.009 0.788±.005 0.183±.013 2.723±.026 11.01±.094 1.609±.065 MoMask [21] 0.433±.007 0.656±.005 0.781±.005 0.204±.011 2.779±.022 - 1.131±.043 LMM [73] 0.430±.015 0.653±.017 0.779±.014 0.137±.023 2.791±.018 11.24±.103 1.885±.127 MoGenTS [64] 0.445±.006 0.671±.006 0.797±.005 0.143±.004 2.711±.024 10.92±.090 -

KITML [46]

Motion Anything (Ours) 0.449±.007 0.678±.004 0.802±.006 0.131±.003 2.705±.024 10.94±.098 1.374±.069

- Table 2. Quantitative comparison on HumanML3D [19] and KIT-ML [46]. The best and runner-up values are bold and underlined. The right arrow → indicates that closer values to ground truth are better. Multimodal motion generation methods are highlighted in blue.

Motion Quality Motion Diversity

Method FIDk ↓ FIDg ↓ Divk ↑ Divg ↑ BAS ↑ Ground Truth 17.10 10.60 8.19 7.45 0.2374 TSMT [33] 86.43 43.46 6.85 3.32 0.1607 Dance Revolution [24] 73.42 25.92 3.52 4.87 0.1950 DanceNet [86] 69.18 25.49 2.86 2.85 0.1430 MoFusion [12] 50.31 - 9.09 - 0.2530 EDGE [54] 42.16 22.12 3.96 4.61 0.2334 Lodge [36] 37.09 18.79 5.58 4.85 0.2423 FACT [34] 35.35 22.11 5.94 6.18 0.2209 Bailando [51] 28.16 9.62 7.83 6.34 0.2332 TM2D [17] 23.94 9.53 7.69 4.53 0.2127 BADM [66] - - 8.29 6.76 0.2366 LMM [73] 22.08 21.97 9.85 6.72 0.2249 Bailando++ [52] 17.59 10.10 8.64 6.50 0.2720 UDE [83] 17.25 8.69 7.78 5.81 0.2310 MCM [39] 15.57 25.85 6.50 5.74 0.2750 Motion Anything (Ours) 17.22 8.56 9.91 6.79 0.2757

- Table 3. Quantitative comparison on AIST++ [34]. The best and runner-up values are bold and underlined. Multimodal motion generation methods are highlighted in blue.

4 demonstrate superior performance by our method when handling different modalities simultaneously.

Motion Quality Motion Diversity Method FIDk ↓ FIDg ↓ Divk ↑ Divg ↑ BAS ↑ MMDist↓ MModality↑ Ground Truth 20.72 11.37 7.42 6.94 0.2105 5.07 -

TM2D [17] 26.78 12.04 6.25 4.41 0.2001 6.13 2.232 MotionCraft [5] 24.21 26.39 7.02 5.79 0.2036 5.82 2.481 Motion Anything 21.46 11.44 7.04 6.15 0.2094 5.34 2.424

Table 4. Quantitative comparison on TMD. The best and runnerup values are bold and underlined.

#### 4.4. Ablation Study

Masking Strategy. To evaluate the effectiveness of our attention-based masking approach across both temporal and spatial dimensions, we conducted comprehensive experiments on HumanML3D [19], comparing it to other masking strategies such as random masking [21, 64], KMeans [40], GMM [49], confidence-based masking [45], and densitybased masking [75]. The results in Table 5 demonstrate that our attention-based masking outperforms these other strategies in human motion generation, yielding promising results for learning robust motion representations.

eration.

Text-and-Music-to-Dance. For paired text-and-music-todance (TM2D) generation, we evaluated open-source multimodal motion generation methods by directly combining their condition embeddings and compared them with our approach on the TMD dataset. The results in Table

Text

BAD BAMM MoMask Motion Anything

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

A man walk forward with both hands above head.

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

A man walk clockwise in a circle.

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

A man picks up something and throw it away.

- Figure 5. Qualitative evaluation on text-to-motion generation. We qualitatively compared the visualizations generated by our method with those produced by BAD [22], BAMM [44], and MoMask [21].

R Precision ↑ FID↓ MM Dist↓ Diversity→ MModality↑ Top 1 Top 2 Top 3

Method

Ground Truth 0.511±.003 0.703±.003 0.797±.002 0.002±.000 2.974±.008 9.503±.065 -

Random Masking [21] 0.522±.004 0.714±.003 0.818±.006 0.049±.023 2.945±.027 9.633±.218 2.538±.035 KMeans [40] 0.528±.003 0.709±.004 0.823±.006 0.042±.032 2.871±.035 9.549±.173 2.548±.023 GMM [49] 0.531±.002 0.721±.004 0.826±.008 0.039±.021 2.887±.024 9.602±.138 2.488±.031 Confidence-based Masking [45] 0.524±.007 0.731±.001 0.818±.004 0.047±.023 2.928±.009 9.530±.095 2.574±.039 Density-based Masking [75] 0.538±.005 0.733±.002 0.819±.006 0.031±.035 2.913±.021 9.518±.138 2.608±.043

Attention-based Masking 0.546±.003 0.735±.002 0.829±.002 0.028±.005 2.859±.010 9.521±.083 2.705±.068

Table 5. Ablation study of the masking strategy on HumanML3D [19]. The best and runner-up values are bold and underlined. The right arrow → indicates that closer values to ground truth are better.

Masking Ratio. To demonstrate the robustness of our method across various hyperparameters and the impact of different masking ratios on overall performance, we conducted comprehensive ablation studies with different attention-based masking ratios on HumanML3D [19], as shown in Table 6. We conducted an ablation study on the masking ratio for temporal and spatial attention-based masking separately. The results show that our method is relatively robust across different masking ratios, with 30% identified as the superior setting in our paper.

Cross-modal TAT for Text-to-Motion. To verify the necessity of self-attention in the Temporal Adaptive Transformer (TAT) for text-to-motion generation, we modified a cross-modal attention layer in TAT to resemble the setup implemented for music-to-dance and text&music-to-dance generation. The results in Table 7 indicate that the crossmodal attention layer performs worse compared to the selfattention layer in TAT for text-to-motion on HumanML3D

R Precision ↑ FID↓ MM Dist↓ Diversity→ MModality↑ Top 1 Top 2 Top 3

Method

Ground Truth 0.511±.003 0.703±.003 0.797±.002 0.002±.000 2.974±.008 9.503±.065 -

T:15% S:15% 0.523±.005 0.716±.002 0.818±.005 0.047±.034 2.920±.026 9.625±.145 2.580±.064 T:15% S:30% 0.529±.002 0.718±.005 0.822±.003 0.044±.046 2.914±.023 9.573±.163 2.631±.024 T:15% S:50% 0.530±.002 0.715±.007 0.820±.007 0.045±.035 2.918±.019 9.632±.217 2.611±.026

T:30% S:15% 0.535±.007 0.728±.001 0.823±.004 0.036±.027 2.873±.037 9.527±.116 2.709±.027 T:30% S:30% 0.546±.003 0.735±.002 0.829±.002 0.028±.005 2.859±.010 9.521±.083 2.705±.068 T:30% S:50% 0.541±.004 0.726±.003 0.821±.005 0.033±.035 2.926±.054 9.519±.196 2.710±.037

T:50% S:15% 0.525±.005 0.720±.003 0.820±.009 0.043±.028 2.940±.044 9.620±.134 2.584±.063 T:50% S:30% 0.525±.007 0.723±.004 0.819±.007 0.040±.042 2.937±.063 9.617±.115 2.701±.031 T:50% S:50% 0.524±.006 0.712±.003 0.822±.006 0.048±.025 2.943±.037 9.623±.153 2.620±.025

- Table 6. Ablation study of masking ratio on HumanML3D [19]. The best and runner-up values are bold and underlined. The right arrow → indicates that closer values to ground truth are better.

Method

R Precision ↑ FID↓ MM Dist↓ Diversity→ MModality↑ Top 1 Top 2 Top 3

Ground Truth 0.511±.003 0.703±.003 0.797±.002 0.002±.000 2.974±.008 9.503±.065 -

Cross-modal Attention 0.347±.006 0.587±.007 0.726±.005 0.583±.024 3.356±.022 9.032±.153 2.153±.056 Motion Anything 0.546±.003 0.735±.002 0.829±.002 0.028±.005 2.859±.010 9.521±.083 2.705±.068

- Table 7. Ablation study of the TAT on HumanML3D [19]. The best values are bold. The right arrow → indicates that closer values to ground truth are better.

[19]. This underperformance can be attributed to the fact that the text embeddings from CLIP consist of only a single token along the temporal dimension. Consequently, the temporal dimension does not align with the motion embeddings, making it unsuitable for effective cross-modal fusion with motion in the temporal context.

Effectiveness of Multimodal Conditioning. To evaluate the effectiveness of multimodal conditioning, we examine whether paired text descriptions enable more controllable music-to-dance generation and whether our any-to-motion

Music

EDGE Lodge Bailando Motion Anything

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

Marshall Jefferson Move Your Body (Chicago House)

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

Stardust - Music Sounds Better With You (French House)

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

Paul Kalkbrenner Sky and Sand (Tech House)

- Figure 6. Qualitative evaluation on music-to-dance generation. We qualitatively compared the visualizations generated by our method with those produced by EDGE [54], Lodge [36], and Bailando [51].

method can seamlessly leverage both conditions. We conduct ablation studies on multimodal conditioning using our TMD dataset, comparing it with the single-condition setting using only music. The results in Table 8 demonstrate that introducing multimodal conditioning is more effective than using a single modality, and our model can effectively and adaptively handle multimodal conditions.

Motion Quality Motion Diversity Method FIDk ↓ FIDg ↓ Divk ↑ Divg ↑ BAS ↑ MMDist↓ MModality↑ Ground Truth 20.72 11.37 7.42 6.94 0.2105 5.07 -

Motion Anything w/o text 25.07 14.23 6.95 6.01 0.2077 6.24 2.398 Motion Anything 21.46 11.44 7.04 6.15 0.2094 5.34 2.424

- Table 8. Single-modal vs. multimodal generation on TMD dataset.

Number of Layers. To investigate the impact of our model by varying the number of layers N in masked transformers, we conduct ablation studies on HumanML3D. Table 9 demonstrates the robustness of our model across different layer configurations.

Method

R Precision ↑ FID↓ MM Dist↓ Diversity→ MModality↑ Top 1 Top 2 Top 3

Ground Truth 0.511±.003 0.703±.003 0.797±.002 0.002±.000 2.974±.008 9.503±.065 -

N = 2 0.521±.006 0.725±.008 0.819±.005 0.079±.019 2.916±.033 9.598±.117 2.503±.024 N = 4 0.546±.003 0.735±.002 0.829±.002 0.028±.005 2.859±.010 9.521±.083 2.705±.068 N = 6 0.541±.007 0.733±.002 0.826±.010 0.029±.004 2.861±.010 9.517±.094 2.673±.019 N = 8 0.544±.009 0.734±.003 0.826±.007 0.028±.014 2.851±.011 9.519±.057 2.711±.032

- Table 9. Ablation study of number of layers on HumanML3D [19].

#### 4.5. Qualitative Evaluation

Text-to-Motion Generation. To qualitatively evaluate our performance in text-to-motion generation, we compare the visualizations generated by our method with those produced by previous state-of-the-art methods specializing in text-tomotion generation, including BAD [22], BAMM [44], and MoMask [21]. The text prompts are customized based on the HumanML3D [19] test set. As shown in Figure 5 and video demos, our method generates motions with superior quality, greater diversity, and better alignment between text and motion compared to the previous state-of-the-art methods.

Music-to-Dance Generation. To evaluate the quality of our music-to-dance generation, we compare the dances generated by our method against those from state-of-the-art approaches, including EDGE [54], Lodge [36], and Bailando [51]. Training on AIST++ [34], we ensure the evaluation reflects diverse musical styles. As shown in Figure 6 and demonstrated in the accompanying videos, our method produces dances with better visual quality and achieves superior alignment with the beat and genre of the music, surpassing previous state-of-the-art techniques.

For more qualitative evaluations and videos, please refer to the supplementary materials.

### 5. Conclusion

In conclusion, Motion Anything presents a significant forward to motion generation by enabling adaptive and controllable multimodal conditioning. Our model introduces the attention-based masking within an autoregressive framework to focus on key frames and actions, addressing the challenge of prioritizing dynamic frames and body parts. Additionally, it bridges the gap in multimodal motion generation by aligning different input modalities both temporally and spatially, enhancing control and coherence. To further advance research in this area, we introduce the TextMusic-Dance (TMD) dataset, a pioneering benchmark with paired music and text. Extensive experiments demonstrate that our method outperforms prior approaches, achieving substantial improvements across multiple benchmarks. By tackling these challenges, Motion Anything establishes a new paradigm for motion generation, offering a more versatile and precise framework for motion generation.

### References

- [1] Unlock the power of 3d design with tripo: The aipowered 3d model generator, 2024. 2, 3
- [2] Sherwin Bahmani, Ivan Skorokhodov, Victor Rong, Gordon Wetzstein, Leonidas Guibas, Peter Wonka, Sergey Tulyakov, Jeong Joon Park, Andrea Tagliasacchi, and David B Lindell. 4d-fy: Text-to-4d generation using hybrid score distillation sampling. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7996–8006, 2024. 2
- [3] Sherwin Bahmani, Xian Liu, Wang Yifan, Ivan Skorokhodov, Victor Rong, Ziwei Liu, Xihui Liu, Jeong Joon Park, Sergey Tulyakov, Gordon Wetzstein, et al. Tc4d: Trajectory-conditioned text-to-4d generation. In European Conference on Computer Vision, pages 53–72. Springer, 2025. 3
- [4] Ilya Baran and Jovan Popovi´c. Automatic rigging and animation of 3d characters. ACM Transactions on graphics (TOG), 26(3):72–es, 2007. 2
- [5] Yuxuan Bian, Ailing Zeng, Xuan Ju, Xian Liu, Zhaoyang Zhang, Wei Liu, and Qiang Xu. Adding multi-modal controls to whole-body human motion generation. arXiv preprint arXiv:2407.21136, 2024. 2, 3, 4, 6, 5
- [6] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023. 2
- [7] P´eter Boros´an, Ming Jin, Doug DeCarlo, Yotam Gingold, and Andrew Nealen. Rigmesh: automatic rig-

- ging for part-based shape modeling and deformation. ACM Transactions on Graphics (TOG), 31(6):1–9, 2012. 2
- [8] Zenghao Chai, Chen Tang, Yongkang Wong, and Mohan Kankanhalli. Star: Skeleton-aware text-based 4d avatar generation with in-network motion retargeting. arXiv preprint arXiv:2406.04629, 2024. 3
- [9] Ling-Hao Chen, Wenxun Dai, Xuan Ju, Shunlin Lu, and Lei Zhang. Motionclr: Motion generation and training-free editing via understanding attention mechanisms. arXiv preprint arXiv:2410.18977, 2024. 5
- [10] Xin Chen, Biao Jiang, Wen Liu, Zilong Huang, Bin Fu, Tao Chen, and Gang Yu. Executing your commands via motion diffusion in latent space. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18000–18010,

2023. 2, 5

- [11] Yushuo Chen, Zerong Zheng, Zhe Li, Chao Xu, and Yebin Liu. Meshavatar: Learning high-quality triangular human avatars from multi-view videos. arXiv preprint arXiv:2407.08414, 2024. 2
- [12] Rishabh Dabral, Muhammad Hamza Mughal, Vladislav Golyanik, and Christian Theobalt. Mofusion: A framework for denoising-diffusion-based motion synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9760–9770, 2023. 3, 4, 6
- [13] Wenxun Dai, Ling-Hao Chen, Jingbo Wang, Jinpeng Liu, Bo Dai, and Yansong Tang. Motionlcm: Realtime controllable motion generation via latent consistency model. arXiv preprint arXiv:2404.19759, 2024. 5
- [14] Zach Evans, Julian D Parker, CJ Carr, Zack Zukowski, Josiah Taylor, and Jordi Pons. Stable audio open. arXiv preprint arXiv:2407.14358, 2024. 5, 2
- [15] Andrew Feng, Dan Casas, and Ari Shapiro. Avatar reshaping and automatic rigging using a deformable model. In Proceedings of the 8th ACM SIGGRAPH Conference on Motion in Games, pages 57–64, 2015. 2
- [16] Xuehao Gao, Yang Yang, Zhenyu Xie, Shaoyi Du, Zhongqian Sun, and Yang Wu. Guess: Gradually enriching synthesis for text-driven human motion generation. IEEE Transactions on Visualization and Computer Graphics, 2024. 5
- [17] Kehong Gong, Dongze Lian, Heng Chang, Chuan Guo, Zihang Jiang, Xinxin Zuo, Michael Bi Mi, and Xinchao Wang. Tm2d: Bimodality driven 3d dance generation via music-text integration. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 9942–9952, 2023. 2, 3, 6, 4, 5

- [18] Deepak Gopinath and Jungdam Won. Fairmotiontools to load, process and visualize motion capture data, 2020. 5
- [19] Chuan Guo, Shihao Zou, Xinxin Zuo, Sen Wang, Wei Ji, Xingyu Li, and Li Cheng. Generating diverse and natural 3d human motions from text. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5152–5161, 2022. 2, 5, 6, 7, 8, 1
- [20] Chuan Guo, Xinxin Zuo, Sen Wang, and Li Cheng. Tm2t: Stochastic and tokenized modeling for the reciprocal generation of 3d human motions and texts. In European Conference on Computer Vision, pages 580–597. Springer, 2022. 5
- [21] Chuan Guo, Yuxuan Mu, Muhammad Gohar Javed, Sen Wang, and Li Cheng. Momask: Generative masked modeling of 3d human motions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1900–1910, 2024. 2, 3, 4, 6, 7, 8, 5
- [22] S Rohollah Hosseyni, Ali Ahmad Rahmani, S Jamal Seyedmohammadi, Sanaz Seyedin, and Arash Mohammadi. Bad: Bidirectional auto-regressive diffusion for text-to-motion generation. arXiv preprint arXiv:2409.10847, 2024. 2, 6, 7, 8, 5
- [23] Liangxiao Hu, Hongwen Zhang, Yuxiang Zhang, Boyao Zhou, Boning Liu, Shengping Zhang, and Liqiang Nie. Gaussianavatar: Towards realistic human avatar modeling from a single video via animatable 3d gaussians. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 634–644, 2024. 2
- [24] Ruozi Huang, Huang Hu, Wei Wu, Kei Sawada, Mi Zhang, and Daxin Jiang. Dance revolution: Long-term dance generation with music via curriculum learning. In International Conference on Learning Representations, 2021. 3, 6
- [25] Yiheng Huang, Hui Yang, Chuanchen Luo, Yuxi Wang, Shibiao Xu, Zhaoxiang Zhang, Man Zhang, and Junran Peng. Stablemofusion: Towards robust and efficient diffusion-based motion generation framework. arXiv preprint arXiv:2405.05691, 2024. 5
- [26] Zikai Huang, Xuemiao Xu, Cheng Xu, Huaidong Zhang, Chenxi Zheng, Jing Qin, and Shengfeng He. Beat-it: Beat-synchronized multi-condition 3d dance generation. arXiv preprint arXiv:2407.07554, 2024. 3
- [27] Biao Jiang, Xin Chen, Wen Liu, Jingyi Yu, Gang Yu, and Tao Chen. Motiongpt: Human motion as a foreign language. Advances in Neural Information Processing Systems, 36:20067–20079, 2023. 5
- [28] Yanqin Jiang, Li Zhang, Jin Gao, Weimin Hu, and Yao Yao. Consistent4d: Consistent 360 {\deg} dy-

- namic object generation from monocular video. arXiv preprint arXiv:2311.02848, 2023. 2
- [29] Peng Jin, Yang Wu, Yanbo Fan, Zhongqian Sun, Wei Yang, and Li Yuan. Act as you wish: Fine-grained control of motion diffusion model with hierarchical semantic graphs. Advances in Neural Information Processing Systems, 36, 2024. 5
- [30] Hanyang Kong, Kehong Gong, Dongze Lian, Michael Bi Mi, and Xinchao Wang. Priority-centric human motion generation in discrete latent space. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 14806–14816, 2023. 5
- [31] Binh Huy Le and Zhigang Deng. Smooth skinning decomposition with rigid bones. ACM Transactions on Graphics (TOG), 31(6):1–10, 2012. 3
- [32] Chang-Hsing Lee, Jau-Ling Shih, Kun-Ming Yu, and Hwai-San Lin. Automatic music genre classification based on modulation spectral analysis of spectral and cepstral features. IEEE Transactions on Multimedia, 11(4):670–682, 2009. 5
- [33] Jiaman Li, Yihang Yin, Hang Chu, Yi Zhou, Tingwu Wang, Sanja Fidler, and Hao Li. Learning to generate diverse dance motions with transformer. arXiv preprint arXiv:2008.08171, 2020. 3, 6
- [34] Ruilong Li, Shan Yang, David A Ross, and Angjoo Kanazawa. Ai choreographer: Music conditioned 3d dance generation with aist++. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 13401–13412, 2021. 2, 3, 5, 6, 8
- [35] Ronghui Li, Hongwen Zhang, Yachao Zhang, Yuxiang Zhang, Youliang Zhang, Jie Guo, Yan Zhang, Xiu Li, and Yebin Liu. Lodge++: High-quality and long dance generation with vivid choreography patterns. arXiv preprint arXiv:2410.20389, 2024. 3
- [36] Ronghui Li, YuXiang Zhang, Yachao Zhang, Hongwen Zhang, Jie Guo, Yan Zhang, Yebin Liu, and Xiu Li. Lodge: A coarse to fine diffusion network for long dance generation guided by the characteristic dance primitives. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1524–1534, 2024. 2, 3, 6, 8
- [37] Zhiqi Li, Yiming Chen, and Peidong Liu. Dreammesh4d: Video-to-4d generation with sparsecontrolled gaussian-mesh hybrid representation. arXiv preprint arXiv:2410.06756, 2024. 2
- [38] Jing Lin, Ailing Zeng, Shunlin Lu, Yuanhao Cai, Ruimao Zhang, Haoqian Wang, and Lei Zhang. Motion-x: A large-scale 3d expressive whole-body human motion dataset. Advances in Neural Information Processing Systems, 36, 2024. 5
- [39] Zeyu Ling, Bo Han, Yongkang Wongkan, Han Lin, Mohan Kankanhalli, and Weidong Geng. Mcm:

- Multi-condition motion synthesis framework. arXiv preprint arXiv:2404.12886, 2024. 3, 4, 6, 5
- [40] Stuart Lloyd. Least squares quantization in pcm. IEEE transactions on information theory, 28(2):129–137,

1982. 6, 7

- [41] Yunhong Lou, Linchao Zhu, Yaxiong Wang, Xiaohan Wang, and Yi Yang. Diversemotion: Towards diverse human motion generation via discrete diffusion. arXiv preprint arXiv:2309.01372, 2023. 6, 5
- [42] Natapon Pantuwong and Masanori Sugimoto. A fully automatic rigging algorithm for 3d character animation. In SIGGRAPH Asia 2011 Posters, pages 1–1.

2011. 2

- [43] Mathis Petrovich, Michael J Black, and G¨ul Varol. Temos: Generating diverse human motions from textual descriptions. In European Conference on Computer Vision, pages 480–497. Springer, 2022. 5
- [44] Ekkasit Pinyoanuntapong, Muhammad Usama Saleem, Pu Wang, Minwoo Lee, Srijan Das, and Chen Chen. Bamm: Bidirectional autoregressive motion model. arXiv preprint arXiv:2403.19435, 2024. 2, 3, 6, 7, 8, 5
- [45] Ekkasit Pinyoanuntapong, Pu Wang, Minwoo Lee, and Chen Chen. Mmm: Generative masked motion model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1546–1555, 2024. 2, 6, 7, 5
- [46] Matthias Plappert, Christian Mandery, and Tamim Asfour. The kit motion-language dataset. Big data, 4(4): 236–252, 2016. 5, 6, 1
- [47] Jiawei Ren, Liang Pan, Jiaxiang Tang, Chi Zhang, Ang Cao, Gang Zeng, and Ziwei Liu. Dreamgaussian4d: Generative 4d gaussian splatting. arXiv preprint arXiv:2312.17142, 2023. 2
- [48] Jiawei Ren, Kevin Xie, Ashkan Mirzaei, Hanxue Liang, Xiaohui Zeng, Karsten Kreis, Ziwei Liu, Antonio Torralba, Sanja Fidler, Seung Wook Kim, et al. L4gm: Large 4d gaussian reconstruction model. arXiv preprint arXiv:2406.10324, 2024. 2
- [49] Douglas A Reynolds et al. Gaussian mixture models. Encyclopedia of biometrics, 741(659-663), 2009. 6, 7
- [50] Yichun Shi, Peng Wang, Jianglong Ye, Mai Long, Kejie Li, and Xiao Yang. Mvdream: Multi-view diffusion for 3d generation. arXiv preprint arXiv:2308.16512,

2023. 2

- [51] Li Siyao, Weijiang Yu, Tianpei Gu, Chunze Lin, Quan Wang, Chen Qian, Chen Change Loy, and Ziwei Liu. Bailando: 3d dance generation by actor-critic gpt with choreographic memory. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11050–11059, 2022. 2, 3, 6, 8

- [52] Li Siyao, Weijiang Yu, Tianpei Gu, Chunze Lin, Quan Wang, Chen Qian, Chen Change Loy, and Ziwei Liu. Bailando++: 3d dance gpt with choreographic memory. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2023. 3, 6
- [53] Guy Tevet, Sigal Raab, Brian Gordon, Yoni Shafir, Daniel Cohen-or, and Amit Haim Bermano. Human motion diffusion model. In The Eleventh International Conference on Learning Representations, 2022. 2, 5
- [54] Jonathan Tseng, Rodrigo Castellon, and Karen Liu. Edge: Editable dance generation from music. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 448–458, 2023. 3, 6, 8
- [55] George Tzanetakis and Perry Cook. Musical genre classification of audio signals. IEEE Transactions on speech and audio processing, 10(5):293–302, 2002. 5
- [56] A Vaswani. Attention is all you need. Advances in Neural Information Processing Systems, 2017. 4
- [57] Shaofei Wang, Bozidar Antic, Andreas Geiger, and Siyu Tang. Intrinsicavatar: Physically based inverse rendering of dynamic humans from monocular videos via explicit ray tracing. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1877–1888, 2024. 2
- [58] Wenjing Wang, Huan Yang, Zixi Tuo, Huiguo He, Junchen Zhu, Jianlong Fu, and Jiaying Liu. Swap attention in spatiotemporal diffusions for text-to-video generation. arXiv preprint arXiv:2305.10874, 2023. 2
- [59] Yin Wang, Zhiying Leng, Frederick WB Li, ShunCheng Wu, and Xiaohui Liang. Fg-t2m: Fine-grained text-driven human motion generation via diffusion model. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 22035–22044,

2023. 5

- [60] Yuan Wang, Di Huang, Yaqi Zhang, Wanli Ouyang, Jile Jiao, Xuetao Feng, Yan Zhou, Pengfei Wan, Shixiang Tang, and Dan Xu. Motiongpt-2: A general-purpose motion-language model for motion generation and understanding. arXiv preprint arXiv:2410.21747, 2024. 5
- [61] Qi Wu, Yubo Zhao, Yifan Wang, Yu-Wing Tai, and Chi-Keung Tang. Motionllm: Multimodal motionlanguage learning with large language models. arXiv preprint arXiv:2405.17013, 2024. 5
- [62] Zijie Wu, Chaohui Yu, Yanqin Jiang, Chenjie Cao, Fan Wang, and Xiang Bai. Sc4d: Sparse-controlled video-to-4d generation and motion transfer. In European Conference on Computer Vision, pages 361–379. Springer, 2025. 2
- [63] Yuyang Yin, Dejia Xu, Zhangyang Wang, Yao Zhao, and Yunchao Wei. 4dgen: Grounded 4d content

- generation with spatial-temporal consistency. arXiv preprint arXiv:2312.17225, 2023. 2
- [64] Weihao Yuan, Weichao Shen, Yisheng He, Yuan Dong, Xiaodong Gu, Zilong Dong, Liefeng Bo, and Qixing Huang. Mogents: Motion generation based on spatial-temporal joint modeling. arXiv preprint arXiv:2409.17686, 2024. 2, 3, 4, 6, 5
- [65] Yifei Zeng, Yanqin Jiang, Siyu Zhu, Yuanxun Lu, Youtian Lin, Hao Zhu, Weiming Hu, Xun Cao, and Yao Yao. Stag4d: Spatial-temporal anchored generative 4d gaussians. In European Conference on Computer Vision, pages 163–179. Springer, 2025. 2
- [66] Canyu Zhang, Youbao Tang, Ning Zhang, Ruei-Sung Lin, Mei Han, Jing Xiao, and Song Wang. Bidirectional autoregessive diffusion model for dance generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 687– 696, 2024. 3, 6
- [67] Hao Zhang, Di Chang, Fang Li, Mohammad Soleymani, and Narendra Ahuja. Magicpose4d: Crafting articulated models with appearance and motion control. arXiv preprint arXiv:2405.14017, 2024. 3, 2
- [68] Haiyu Zhang, Xinyuan Chen, Yaohui Wang, Xihui Liu, Yunhong Wang, and Yu Qiao. 4diffusion: Multiview video diffusion model for 4d generation. arXiv preprint arXiv:2405.20674, 2024. 2
- [69] Jianrong Zhang, Yangsong Zhang, Xiaodong Cun, Yong Zhang, Hongwei Zhao, Hongtao Lu, Xi Shen, and Ying Shan. Generating human motion from textual descriptions with discrete representations. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 14730–14740,

2023. 5

- [70] Mingyuan Zhang, Xinying Guo, Liang Pan, Zhongang Cai, Fangzhou Hong, Huirong Li, Lei Yang, and Ziwei Liu. Remodiffuse: Retrieval-augmented motion diffusion model. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 364– 373, 2023. 2, 3, 6, 5
- [71] Mingyuan Zhang, Huirong Li, Zhongang Cai, Jiawei Ren, Lei Yang, and Ziwei Liu. Finemogen: Finegrained spatio-temporal motion generation and editing. Advances in Neural Information Processing Systems, 36:13981–13992, 2023. 5
- [72] Mingyuan Zhang, Zhongang Cai, Liang Pan, Fangzhou Hong, Xinying Guo, Lei Yang, and Ziwei Liu. Motiondiffuse: Text-driven human motion generation with diffusion model. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024. 2, 5
- [73] Mingyuan Zhang, Daisheng Jin, Chenyang Gu, Fangzhou Hong, Zhongang Cai, Jingfang Huang, Chongzhi Zhang, Xinying Guo, Lei Yang, Ying He,

- et al. Large motion model for unified multi-modal motion generation. arXiv preprint arXiv:2404.01284, 2024. 2, 3, 4, 6, 5
- [74] Yaqi Zhang, Di Huang, Bin Liu, Shixiang Tang, Yan Lu, Lu Chen, Lei Bai, Qi Chu, Nenghai Yu, and Wanli Ouyang. Motiongpt: Finetuned llms are general-purpose motion generators. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 7368–7376, 2024. 5
- [75] Zeyu Zhang, Hang Gao, Akide Liu, Qi Chen, Feng Chen, Yiran Wang, Danning Li, and Hao Tang. Kmm: Key frame mask mamba for extended motion generation. arXiv preprint arXiv:2411.06481, 2024. 3, 6, 7
- [76] Zeyu Zhang, Akide Liu, Qi Chen, Feng Chen, Ian Reid, Richard Hartley, Bohan Zhuang, and Hao Tang. Infinimotion: Mamba boosts memory in transformer for arbitrary long motion generation. arXiv preprint arXiv:2407.10061, 2024. 3
- [77] Zeyu Zhang, Yiran Wang, Biao Wu, Shuo Chen, Zhiyuan Zhang, Shiya Huang, Wenbo Zhang, Meng Fang, Ling Chen, and Yang Zhao. Motion avatar: Generate human and animal avatars with arbitrary motion. arXiv preprint arXiv:2405.11286, 2024. 3, 2
- [78] Zeyu Zhang, Akide Liu, Ian Reid, Richard Hartley, Bohan Zhuang, and Hao Tang. Motion mamba: Efficient and long sequence motion generation. In European Conference on Computer Vision, pages 265–282. Springer, 2025. 2, 5
- [79] Yuyang Zhao, Zhiwen Yan, Enze Xie, Lanqing Hong, Zhenguo Li, and Gim Hee Lee. Animate124: Animating one image to 4d dynamic scene. arXiv preprint arXiv:2311.14603, 2023. 2
- [80] Yufeng Zheng, Xueting Li, Koki Nagano, Sifei Liu, Otmar Hilliges, and Shalini De Mello. A unified approach for text-and image-guided 4d scene generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7300– 7309, 2024. 2
- [81] Chongyang Zhong, Lei Hu, Zihao Zhang, and Shihong Xia. Attt2m: Text-driven human motion generation with multi-perspective attention mechanism. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 509–519, 2023. 5
- [82] Wenyang Zhou, Zhiyang Dou, Zeyu Cao, Zhouyingcheng Liao, Jingbo Wang, Wenjia Wang, Yuan Liu, Taku Komura, Wenping Wang, and Lingjie Liu. Emdm: Efficient motion diffusion model for fast, high-quality motion generation. arXiv preprint arXiv:2312.02256, 2, 2023. 5
- [83] Zixiang Zhou and Baoyuan Wang. Ude: A unified driving engine for human motion generation. In Proceedings of the IEEE/CVF Conference on Computer

Vision and Pattern Recognition, pages 5632–5641,

2023. 2, 3, 4, 6

- [84] Zixiang Zhou, Yu Wan, and Baoyuan Wang. A unified framework for multimodal, multi-part human motion synthesis. arXiv preprint arXiv:2311.16471, 2023. 3, 4
- [85] Wentao Zhu, Xiaoxuan Ma, Dongwoo Ro, Hai Ci, Jinlu Zhang, Jiaxin Shi, Feng Gao, Qi Tian, and Yizhou Wang. Human motion generation: A survey. IEEE Transactions on Pattern Analysis and Machine Intelligence, 46(4):2430–2449, 2023. 2
- [86] Wenlin Zhuang, Congyi Wang, Jinxiang Chai, Yangang Wang, Ming Shao, and Siyu Xia. Music2dance: Dancenet for music-driven dance generation. ACM Transactions on Multimedia Computing, Communications, and Applications (TOMM), 18(2):1–21, 2022. 3, 6
- [87] Qiran Zou, Shangyuan Yuan, Shian Du, Yu Wang, Chang Liu, Yi Xu, Jie Chen, and Xiangyang Ji. Parco: Part-coordinating text-to-motion synthesis. arXiv preprint arXiv:2403.18512, 2024. 5

## Motion Anything: Any to Motion Generation Supplementary Material

### 1. Full Comparison for Text-to-Motion

To comprehensively showcase our method’s performance in text-to-motion generation, we present a full comparison with previous T2M approaches in Table 1. The results demonstrate that our method consistently outperforms others, achieving state-of-the-art performance on both HumanML3D [19] and KIT-ML [46] datasets.

### 2. User Study

This study provides a comprehensive evaluation of our motion generation. We assessed the real-world applicability of four motion videos generated by Motion Anything and baseline models, as evaluated by 50 participants through a Google Forms survey. Figure 1 displays the User Interface (UI) used in our user study, showcasing 3–4 videos (Video 1 to 3/4), each featuring distinct motion animations from the same model, and four videos for comparing different models (Video A to D). Participants evaluate these animations based on aspects such as motion accuracy and overall user experience. They rate each aspect from 1 (low) to 3 (high) to assess how well the animations mirror realworld movements and their engagement level. In the comparison section, participants select the model with the best performance. This evaluation aims to determine the realism and engagement effectiveness of each motion. The evaluation consisted of three groups of motions: text-to-motion, music-to-dance, and text-and-music-to-dance. The results are as follows:

##### Text-to-Motion:

- • Our motion quality rating is 2.84. Additionally, 86% of participants believe that our method demonstrates high-quality motion generation with minimal jitter, sliding, and unrealistic movements.
- • Our motion diversity rating is 2.68. Additionally, 72% of participants believe that our method generates complex and diverse motions.
- • Our text-motion alignment rating is 2.74. Additionally, 76% of participants believe that our method generates motion that is well-aligned with their text condition.
- • 88% of participants believe that our method outperforms other methods.

##### Music-to-Dance:

- • Our dance quality rating is 2.82. Additionally, 82% of participants believe that our method demonstrates high-quality dance generation with minimal jitter, sliding, and unrealistic movements.
- • Our dance diversity rating is 2.88. Additionally, 88% of participants believe that our method generates com-

[Figure 235]

Figure 1. User study form. The User Interface (UI) used in our user study.

[Figure 236]

[Figure 237]

plex and diverse dance.

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

- • Our music-dance alignment rating is 2.66. Additionally, 70% of participants believe that our method generates dance that is well-aligned with the music genre and beats.
- • 74% of participants believe that our method outperforms other methods.

[Figure 244]

[Figure 245]

Stable Audio Open

Music

Kung Fu Render

[Figure 246]

Panda twists side to side, raises both arms, and alternates stretching hands forward.

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

Retarget

[Figure 253]

[Figure 254]

[Figure 255]

Motion Generator

Motion Sequence Selected Rigged Avatar

[Figure 256]

##### Text and Music to Dance:

[Figure 257]

SRM

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

- • Our dance quality rating is 2.74. Additionally, 76% of participants believe that our method demonstrates high-quality dance generation with minimal jitter, sliding, and unrealistic movements.
- • Our multimodal condition rating is 2.88. Additionally, 88% of participants believe that text enhances music’s conditioning in dance generation.
- • Our text&music-dance aligment rating is 2.76. There are 80% participants believe that our method generates dance that is well-aligned with the both music and text.
- • 78% of participants believe that our method outperforms other methods.

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

Auto-Rig

Text Queries

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

Tripo AI 2.0

Candidate Avatar

Candidate Rigged Avatar

Figure 3. 4D Avatar Generation. This approach enables 4D avatar generation conditioned on multimodal inputs, achievable with just a single text prompt.

### 4. Application: 4D Avatar Generation

One of the most significant applications for conditional motion generation is 4D avatar generation. Previous methods [11, 23, 28, 37, 47, 48, 57, 62, 63, 65] for 4D avatar generation reconstruct dynamic avatars using 4D Gaussians to model both spatial and temporal dimensions from video data. Other approaches [2, 68, 79, 80] integrate video diffusion [6, 58] with geometry-aware diffusion models [50] to ensure spatial consistency and achieve a visually appealing appearance. However, these methods face two significant challenges: (1) limited diversity and control over motion [28, 37, 62], and (2) inconsistencies in the mesh appearance at different time points [47].

### 3. Model Efficiency

We evaluate the inference efficiency of motion generation compared to various methods. The inference cost is calculated as the average inference time over 100 samples on one NVIDIA GeForce RTX 2080 Ti device. Compared to baseline methods, Motion Anything achieves an outstanding balance between generation quality and efficiency, as shown in Figure 2.

Therefore, we propose a comprehensive feedforward approach for 4D avatar generation with only a single prompt, leveraging both Motion Anything and off-the-shelf tools, as illustrated in Figure 3. A single prompt serves as the input, when music is not provided as a condition, Stable Audio Open [14] generates background music as the audio condition. The text and audio conditions are then processed by the multimodal motion generator to create a high-quality motion sequence. Simultaneously, 3D avatar generation employs Tripo AI 2.0 [1] to produce candidate meshes, as shown in Figure 4, and the Selective Rigging Mechanism (SRM) identifies the optimal rigged mesh. The motion sequence is then retargeted to the avatar meshes, resulting in a complete 4D avatar.

Comparisons on FID and Inference Cost

1.4

- FID: 0.04 AIT: 0.12

MLD

- FID: 0.47 AIT: 0.22

T2M

- FID: 1.09 AIT: 0.03

TM2T

- FID: 1.50 AIT: 0.76

1.2

- 0.8
- 1

MotionDiffuse FID: 0.63 AIT: 10.89

FID

0.6

MDM FID: 0.54 AIT: 18.20

T2M-GPT

0.4

FID: 0.12 AIT: 0.38

MoMask

As shown in the Figure 3, automatic rigging is crucial for 4D generation, directly affecting the precision and realism of avatar movements [67, 77]. Although numerous optimization-based approaches [4, 7, 15, 42] have been proposed to achieve fully automated rigging, the outcomes are often unsatisfactory due to the diverse appearances of meshes. Hence, achieving high-quality automatic rigging has become an important challenge to address in skeleton-

0.2

###### Motion Anything

FID: 0.028 AIT: 0.095

0

0 2 4 6 8 10 12 14 16 18 20

AIT (Average Inference Time, s)

- Figure 2. Comparisons on FID and AIT. All tests are conducted on the same NVIDIA GeForce RTX 2080 Ti. The closer the model is to the origin, the better.

based avatar generation. To improve the underperformance of automatic rigging in generated avatars, we introduced a straightforward yet effective Selective Rigging Mechanism that selects the best-rigged 3D avatar from multiple candidates, enhancing the realism of the avatar’s motion visualization.

#### 4.1. Selective Rigging Mechanism

To improve automatic rigging performance and reduce the need for human-in-the-loop adjustments, the Selective Rigging Mechanismn (SRM) presents a two-stage selection process with constraints. This mechanism identifies the optimal rigging from a set of candidate avatars, as shown in

- Figure 3.

- Stage 1: Centroid-Based Filtering. The purpose of this stage is to identify point clouds with centroids positioned within a balanced and plausible bounding region for rigging. Each animated point cloud is defined by a set of 3D coordinates P = {p1,p2,...,pN}, where pi = (xi,yi,zi) ∈ R3, representing the character’s surface. The centroid Gcloud of the point cloud, serving as an approximate center of mass, is calculated as

Gcloud =

1 N

N

i=1

xi,

1 N

N

i=1

yi,

1 N

N

i=1

zi .

The bounding box and stability filters ensure that Gcloud falls within a spatial region aligned with the character’s rigging needs. Specifically, the bounding box constraint requires −1 < XG < 1, −1 < YG < 1, −1 < ZG < 1, while the stability constraint approximates balance by enforcing |XG| ≈ 0, |YG| ≈ 0, and ZG > 0.

These constraints are physically motivated: (1) Minimal lateral displacement, represented by |XG| ≈ 0 and |YG| ≈ 0, keeps the center of mass near the z-axis, avoiding lateral imbalances. (2) Ensuring ZG > 0 places the centroid above the ground plane, maintaining a logical upright character orientation.

- Stage 2: Joint Weight Optimization. This stage’s goal is to select the point cloud configuration with the best joint weight distribution to support stable, smooth deformation during animation. Each vertex i has joint weights

wi1,wi2,...,win, where each wij denotes the influence of joint j on vertex i. To achieve realistic deformation, these weights must sum to 1 across all joints for each vertex, as specified by [31]. The weight normalization condition is expressed as:

n

wij = 1, ∀i ∈ {1,2,...,N}

j=1

where N is the total number of points in P.

To evaluate and select the optimal point cloud from M candidates, we define a loss function based on the average

[Figure 272]

Figure 4. 3D Avatars. This figure shows examples of 3D avatars generated by Tripo AI 2.0 [1]. These avatars will later serve as candidates for our Selective Rigging Mechanism.

deviation from the ideal weight sum. For each point cloud, we calculate the average weight sum S as

n

N

1 N

S =

wij.

j=1

i=1

Our loss function, defined as the absolute difference |S−1|, quantifies how close each point cloud’s weight distribution is to the ideal configuration. Minimizing |S − 1| allows us to select the point cloud whose joint weights best satisfy the normalization condition. Thus, the optimal point cloud Poptimal is chosen as:

|Sk − 1|,

Poptimal = arg min

Pk

where k ∈ {1,2,...,M}, Pk is the k-th candidate point cloud, and Sk is the average weight sum for Pk. By selecting the configuration with the smallest |S − 1|, we ensure a joint weight distribution close to ideal, supporting stable, natural rigging and deformation in animation.

To showcase our improvement and efficiency in rigging with SRM on TMD dataset, we create 200 avatar descriptions along with text randomly selected from TMD to generate corresponding avatars using TripoAI 2.0 [1]. We then evaluate our SRM with different candidate numbers k based on the average weighted sum S in terms of rigging quality, where a value of S closer to 1 indicates better performance. The results show in in Table 2. Video demos of the generated 4D avatars can also be found in the supplementary materials.

Method S → 1 AIT(s) ↓ MagicPose4D 1.93 0.138 SRM (k = 1) 1.78 0.094 SRM (k = 3) 1.36 0.105 SRM (k = 5) 1.06 0.117

Table 2. SRM evaluation.

TM2D MotionCraft Motion Anything

Text

Music

[Figure 273]

[Figure 274]

[Figure 275]

A man is doing groove and swaying steps along with the beat.

Daft Punk - Get Lucky (Disco)

[Figure 276]

[Figure 277]

[Figure 278]

A man is dancing along the beats while use both hands to touch legs and swing back and forth.

Daft Punk - One More Time (French House)

Human Music Generated Music

[Figure 279]

[Figure 280]

[Figure 281]

An energetic dance track with 120–130 BPM, vibrant synths, punchy beats, and uplifting melodies.

A man is doing street dance, kick side step along the beats.

[Figure 282]

[Figure 283]

[Figure 284]

An energetic dance track with 120–130 BPM, dynamic bass lines, punchy beats, and modern electronic elements.

A man alternates lifting his arms overhead, following the beats.

Figure 5. Qualitative evaluation on text-&-music-to-dance generation. We qualitatively compared the visualizations generated by our method with those produced by TM2D [17] and MotionCraft [5].

Algorithm 1 Selective Rigging Mechanism

- 1: Input: M point clouds {P1,...,PM}, each with N 3D vertices and joint weights {wi1,...,win} for each vertex i
- 2: Stage 1: Centroid Filtering
- 3: for each Pk do
- 4: Compute centroid ck = (Xk,Yk,Zk)
- 5: if ck is outside −1 < Xk,Yk,Zk < 1 or not close to (0,0,Z > 0) then
- 6: Discard Pk
- 7: end if
- 8: end for
- 9: Stage 2: Weight Optimization
- 10: for each remaining Pk do
- 11: Calculate Sk = N1 Ni=1 nj=1 wij

- 12: Compute deviation ∆k = |Sk − 1|
- 13: end for
- 14: Select Poptimal = arg minP

k

∆k

- 15: Output: Optimal point cloud Poptimal

### 5. Qualitative Evaluation

In the main text, we have already demonstrated the qualitative evaluation of text-to-motion generation and music-to-

dance generation. Here, we showcase some examples of the visualization of text-and-music-to-dance generation.

#### 5.1. Text-&-Music-to-Dance

To evaluate the quality of our text-and-music-to-dance generation, we compare the dances generated by our method against those from open-source state-of-the-art multi-task models, including TM2D [17] and MotionCraft [5]. Similar to Section 4.3 in the main text, despite these multitask models lacking the ability to simultaneously take two modalities of conditions, we combined their condition embeddings and compared them with our approach. Trained on our TMD dataset, we ensure that the evaluation reflects diverse musical styles. As shown in Figure 5 and demonstrated in the accompanying videos, our method produces dances with better visual quality and achieves superior alignment with both text description, beat, and genre of the music, surpassing previous state-of-the-art techniques.

R Precision ↑ FID↓ MultiModal Dist↓ Diversity→ MultiModality↑ Top 1 Top 2 Top 3

Datasets Method

Ground Truth 0.511±.003 0.703±.003 0.797±.002 0.002±.000 2.974±.008 9.503±.065 TEMOS [43] 0.424±.002 0.612±.002 0.722±.002 3.734±.028 3.703±.008 8.973±.071 0.368±.018 TM2T [20] 0.424±.003 0.618±.003 0.729±.002 1.501±.017 3.467±.011 8.589±.076 2.424±.093 T2M [19] 0.457±.002 0.639±.003 0.740±.003 1.067±.002 3.340±.008 9.188±.002 2.090±.083 TM2D [17] 0.319±.000 - - 1.021±.000 4.098±.000 9.513±.000 4.139±.000 MotionGPT (Zhang et al.) [74] 0.364±.005 0.533±.003 0.629±.004 0.805±.002 3.914±.013 9.972±.026 2.473±.041 MotionDiffuse [72] 0.491±.001 0.681±.001 0.782±.001 0.630±.001 3.113±.001 9.410±.049 1.553±.042 MDM [53] 0.320±.005 0.498±.004 0.611±.007 0.544±.044 5.566±.027 9.559±.086 2.799±.072 MotionLLM [61] 0.482±.004 0.672±.003 0.770±.002 0.491±.019 3.138±.010 9.838±.244 MLD [10] 0.481±.003 0.673±.003 0.772±.002 0.473±.013 3.196±.010 9.724±.082 2.413±.079 M2DM [30] 0.497±.003 0.682±.002 0.763±.003 0.352±.005 3.134±.010 9.926±.073 3.587±.072 MotionLCM [13] 0.502±.003 0.698±.002 0.798±.002 0.304±.012 3.012±.007 9.607±.066 2.259±.092 Motion Mamba [78] 0.502±.003 0.693±.002 0.792±.002 0.281±.009 3.060±.058 9.871±.084 2.294±.058 Fg-T2M [59] 0.492±.002 0.683±.003 0.783±.002 0.243±.019 3.109±.007 9.278±.072 1.614±.049 MotionGPT (Jiang et al.) [27] 0.492±.003 0.681±.003 0.778±.002 0.232±.008 3.096±.008 9.528±.071 2.008±.084 MotionGPT-2 [60] 0.496±.002 0.691±.003 0.782±.004 0.191±.004 3.080±.013 9.860±.026 2.137±.022 MotionCraft [5] 0.501±.003 0.697±.003 0.796±.002 0.173±.002 3.025±.008 9.543±.098 FineMoGen [71] 0.504±.002 0.690±.002 0.784±.002 0.151±.008 2.998±.008 9.263±.094 2.696±.079 T2M-GPT [69] 0.492±.003 0.679±.002 0.775±.002 0.141±.005 3.121±.009 9.722±.082 1.831±.048 GraphMotion [29] 0.504±.003 0.699±.002 0.785±.002 0.116±.007 3.070±.008 9.692±.067 2.766±.096 EMDM [82] 0.498±.007 0.684±.006 0.786±.006 0.112±.019 3.110±.027 9.551±.078 1.641±.078 AttT2M [81] 0.499±.003 0.690±.002 0.786±.002 0.112±.006 3.038±.007 9.700±.090 2.452±.051 GUESS [16] 0.503±.003 0.688±.002 0.787±.002 0.109±.007 3.006±.007 9.826±.104 2.430±.100 ParCo [87] 0.515±.003 0.706±.003 0.801±.002 0.109±.005 2.927±.008 9.576±.088 1.382±.060 ReMoDiffuse [70] 0.510±.005 0.698±.006 0.795±.004 0.103±.004 2.974±.016 9.018±.075 1.795±.043 MotionCLR [9] 0.542±.001 0.733±.002 0.827±.003 0.099±.003 2.981±.011 - 2.145±.043 StableMoFusion [25] 0.553±.003 0.748±.002 0.841±.002 0.098±.003 - 9.748±.092 1.774±.051 MMM [45] 0.504±.003 0.696±.003 0.794±.002 0.080±.003 2.998±.007 9.411±.058 1.164±.041 DiverseMotion [41] 0.515±.003 0.706±.002 0.802±.002 0.072±.004 2.941±.007 9.683±.102 1.869±.089 BAD [22] 0.517±.002 0.713±.003 0.808±.003 0.065±.003 2.901±.008 9.694±.068 1.194±.044 BAMM [44] 0.525±.002 0.720±.003 0.814±.003 0.055±.002 2.919±.008 9.717±.089 1.687±.051 MCM [39] 0.502±.002 0.692±.004 0.788±.006 0.053±.007 3.037±.003 9.585±.082 0.810±.023 MoMask [21] 0.521±.002 0.713±.002 0.807±.002 0.045±.002 2.958±.008 - 1.241±.040 LMM [73] 0.525±.002 0.719±.002 0.811±.002 0.040±.002 2.943±.012 9.814±.076 2.683±.054 MoGenTS [64] 0.529±.003 0.719±.002 0.812±.002 0.033±.001 2.867±.006 9.570±.077 Motion Anything (Ours) 0.546±.003 0.735±.002 0.829±.002 0.028±.005 2.859±.010 9.521±.083 2.705±.068

Human ML3D [19]

Ground Truth 0.424±.005 0.649±.006 0.779±.006 0.031±.004 2.788±.012 11.08±.097 -

TEMOS [43] 0.353±.006 0.561±.007 0.687±.005 3.717±.051 3.417±.019 10.84±.100 0.532±.034 TM2T [20] 0.280±.005 0.463±.006 0.587±.005 3.599±.153 4.591±.026 9.473±.117 3.292±.081 T2M [19] 0.370±.005 0.569±.007 0.693±.007 2.770±.109 3.401±.008 10.91±.119 1.482±.065 MotionDiffuse [72] 0.417±.004 0.621±.004 0.739±.004 1.954±.062 2.958±.005 11.10±.143 0.753±.013 MDM [53] 0.164±.004 0.291±.004 0.396±.004 0.497±.021 9.190±.022 10.85±.109 1.907±.214 MLD [10] 0.390±.003 0.609±.003 0.734±.002 0.404±.013 3.204±.010 10.80±.082 2.192±.079 M2DM [30] 0.405±.003 0.629±.005 0.739±.004 0.502±.049 3.012±.015 11.38±.079 3.273±.045 Motion Mamba [78] 0.419±.006 0.645±.005 0.765±.006 0.307±.041 3.021±.025 11.02±.098 1.678±.064 Fg-T2M [59] 0.418±.005 0.626±.004 0.745±.004 0.571±.047 3.114±.015 10.93±.083 1.019±.029 MotionGPT (Zhang et al.) [74] 0.340±.002 0.570±.003 0.660±.004 0.868±.032 3.721±.018 9.972±.026 2.296±.022 MotionGPT (Jiang et al.) [27] 0.366±.005 0.558±.004 0.680±.005 0.510±.016 3.527±.021 10.35±.084 2.328±.117 MotionGPT-2 [60] 0.427±.003 0.627±.002 0.764±.003 0.614±.005 3.164±.013 11.26±.026 2.357±.022 FineMoGen [71] 0.432±.006 0.649±.005 0.772±.006 0.178±.007 2.869±.014 10.85±.115 1.877±.093 T2M-GPT [69] 0.416±.006 0.627±.006 0.745±.006 0.514±.029 3.007±.023 10.92±.108 1.570±.039 GraphMotion [29] 0.417±.008 0.635±.006 0.755±.004 0.262±.021 3.085±.031 11.21±.106 3.568±.132 EMDM [82] 0.443±.006 0.660±.006 0.780±.005 0.261±.014 2.874±.015 10.96±.093 1.343±.089 AttT2M [81] 0.413±.006 0.632±.006 0.751±.006 0.870±.039 3.039±.021 10.96±.123 2.281±.047 GUESS [16] 0.425±.005 0.632±.007 0.751±.005 0.371±.020 2.421±.022 10.93±.110 2.732±.084 ParCo [87] 0.430±.004 0.649±.007 0.772±.006 0.453±.027 2.820±.028 10.95±.094 1.245±.022 ReMoDiffuse [70] 0.427±.014 0.641±.004 0.765±.055 0.155±.006 2.814±.012 10.80±.105 1.239±.028 StableMoFusion [25] 0.445±.006 0.660±.005 0.782±.004 0.258±.029 - 10.94±.077 1.362±.062 MMM [45] 0.404±.005 0.621±.005 0.744±.004 0.316±.028 2.977±.019 10.91±.101 1.232±.039 DiverseMotion [41] 0.416±.006 0.637±.008 0.760±.011 0.468±.098 2.892±.041 10.87±.101 2.062±.079 BAD [22] 0.417±.006 0.631±.006 0.750±.006 0.221±.012 2.941±.025 11.00±.100 1.170±.047 BAMM [44] 0.438±.009 0.661±.009 0.788±.005 0.183±.013 2.723±.026 11.01±.094 1.609±.065 MoMask [21] 0.433±.007 0.656±.005 0.781±.005 0.204±.011 2.779±.022 - 1.131±.043 LMM [73] 0.430±.015 0.653±.017 0.779±.014 0.137±.023 2.791±.018 11.24±.103 1.885±.127 MoGenTS [64] 0.445±.006 0.671±.006 0.797±.005 0.143±.004 2.711±.024 10.92±.090 -

KITML [46]

Motion Anything (Ours) 0.449±.007 0.678±.004 0.802±.006 0.131±.003 2.705±.024 10.94±.098 1.374±.069

Table 1. Comprehensive comparison on HumanML3D [19] and KIT-ML [46]. The best and runner-up values are bold and underlined. The right arrow → indicates that closer values to ground truth are better. Multimodal motion generation methods are highlighted in blue.

