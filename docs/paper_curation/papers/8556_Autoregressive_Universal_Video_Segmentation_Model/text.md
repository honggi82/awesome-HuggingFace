[Figure 1]

2025-8-27

## Autoregressive Universal Video Segmentation Model

Miran Heo*,†,1,2 Sukjun Hwang*,3 Min-Hung Chen1 Yu-Chiang Frank Wang1,4 Albert Gu3 Seon Joo Kim‡,2 Ryo Hachiuma‡,1

1 NVIDIA 2 Yonsei University 3 Carnegie Mellon University 4 National Taiwan University

# arXiv:2508.19242v1[cs.CV]26Aug2025

### Abstract

Recent video foundation models such as SAM2 excel at prompted video segmentation by treating masks as a general-purpose primitive. However, many real-world settings require unprompted segmentation that aims to detect and track all objects in a video without external cues, leaving today’s landscape fragmented across task-specific models and pipelines. We recast streaming video segmentation as sequential mask prediction, analogous to language modeling, and introduce the Autoregressive Universal Segmentation Model (AUSM), a single architecture that unifies both prompted and unprompted video segmentation. Built on recent state-space models, AUSM maintains a fixed-size spatial state and scales to video streams of arbitrary length. Furthermore, all components of AUSM are designed for parallel training across frames, yielding substantial speedups over iterative training. On standard benchmarks (DAVIS17, YouTube-VOS 2018 & 2019, MOSE, YouTube-VIS 2019 & 2021, and OVIS) AUSM outperforms prior universal streaming video segmentation methods and achieves up to 2.5× faster training on 16-frame sequences.

### 1. Introduction

Language and video are both sequential modalities that arrive as streams, yet their modeling trajectories have diverged. In language, decoder-only large language models (LLMs) show that a single scalable architecture trained on massive corpora can subsume diverse tasks. Video perception would benefit from the same unification: annotations are fragmented across tasks while video data is expensive to curate, so a unified model could amortize supervision and deployment. We focus on streaming video segmentation, and an ideal universal model should: (i) accommodate a broad set of tasks, (ii) preserve fine-grained spatio-temporal details from past inputs, (iii) support inference over long videos, and (iv) enable training that scales efficiently with sequence length. Despite the structural parallels to language, current practice in video remains partitioned into task-specific architectures and training protocols, and no existing approach simultaneously meets all four criteria.

We categorize streaming video segmentation into two regimes: prompted and unprompted. Prompted video segmentation, exemplified by video object segmentation (VOS), takes an initial human cue (e.g., mask, box, point, or text) and propagates specified targets over time; it is effective for user-interactive editing, but does not naturally handle the emergence of novel instances without re-prompting (e.g., in autonomous driving). Unprompted video segmentation – comprising video instance and panoptic segmentation (VIS/VPS) – aims to detect and track all instances of predefined categories throughout a video. Many detect-then-track pipelines (Yang et al., 2019; Huang et al., 2022; Wu et al., 2022) process frames independently and associate post hoc; even approaches that leverage past information (Wu et al., 2022; Ying et al., 2023; Kim et al., 2024) compress each instance into a few vectors, preserving history mainly for identity association and discarding fine-grained spatio-temporal details needed for detection. Recent attempts at universal models (Yan et al., 2023; Li et al., 2024; Athar et al., 2023) retrofit VOS into VIS-style architectures by encoding an instance as a heavily compressed token, which yields noticeable performance drops on VOS. Furthermore, to our knowledge, existing training frameworks for video segmentation lack LLM-style parallelized training, limiting scalability with respect to sequence length.

* Equal contribution. † Work partially done during internship at NVIDIA. ‡ Corresponding authors. © 2025 NVIDIA. All rights reserved.

In this paper, we present an autoregressive universal video segmentation model, AUSM, that unifies both prompted and unprompted video segmentation tasks while scaling to long video settings.

Unified Structure. First, we identify a key connection between the autoregressive pipeline of decoder-only LLMs and streaming video perception: next-word prediction as next-frame mask prediction. Based on this perspective, we present a unification of prompted and unprompted video segmentation and design a universal model that achieves state-of-the-art performance among online universal methods on seven benchmarks comprising both VOS and VIS using shared weights.

Architecture. AUSM is designed with components specialized for streaming videos, History Marker and History Compressor. History Marker removes abstraction of instances by leveraging Token Mark (Heo et al., 2025) and dissolves segmentation masks into frame features for retrieval of instance-wise information. This effectively preserves fine-grained information, demonstrating a nearly 10% improvement in VOS performance compared to previous unified online architectures. History Compressor then takes the output and compresses spatio-temporal information of all past frames into a single spatial state. While video segmentation models typically store fewer than ten frame features (Oh et al., 2019; Ravi et al., 2025), our design makes processing arbitrarily long streams feasible.

Training Acceleration. AUSM supports parallel training, a critical property of the building blocks (Vaswani et al., 2017; Gu and Dao, 2023) used in decoder-only LLMs for extending to long sequences. Compared to existing frameworks that recurrently process frames during training (Heo et al., 2025; Yang et al., 2021; Ravi et al., 2025), our training pipeline shows significant speedups.

Finally, we evaluate AUSM across a diverse set of benchmarks spanning both prompted and unprompted video segmentation: DAVIS 2017 (Pont-Tuset et al., 2017), YouTube-VOS 18&19 (Xu et al., 2018), MOSE (Ding et al., 2023), YouTube-VIS 19&21 (Yang et al., 2019), and OVIS (Qi et al., 2022). AUSM delivers strong performance across all benchmarks, outperforming previous online universal video segmentation models (Li et al., 2024; Yan

- et al., 2023). Importantly, these results are achieved without relying on FIFO memory buffers (Oh et al., 2019; Kirillov et al., 2023), highlighting the efficiency and scalability of our autoregressive design. Furthermore, parallel training becomes faster compared to recurrent training frameworks as the sequence length increases, achieving up to 2.5× faster training than iterative baselines with 16-frame sequences.

### 2. Autoregressive Universal Video Segmentation Model (AUSM)

In Sec. 2.1, we first formalize video segmentation within an autoregressive framework inspired by LLMs, which provides a seamless unification of prompted and unprompted video segmentation tasks. In Sec. 2.2, we then introduce the architectural components and overall algorithm of AUSM using the recurrent form (Fig. 1-Left). Finally, in Sec. 2.3, we demonstrate how each component of AUSM is designed for parallel training (Fig. 1-Right).

- 2.1. Video Segmentation as Language Modeling Video segmentation encompasses a family of tasks (Yang et al., 2019; Pont-Tuset et al., 2017; Kim et al.,

- 2020; Voigtlaender et al., 2019; Cordts et al., 2016) with varying objectives but a shared core definition. Given a video consisting of 𝑇 frames, (ℐ1,...,ℐ𝑇), where each frame ℐ𝑡 ∈ R𝐻×𝑊×3 is an RGB image with spatial resolution 𝐻 × 𝑊, the goal is to produce (ˆ𝑦1,...,𝑦ˆ𝑇) that predicts a sequence of segmentation ground

gt

truth 𝑌 = (𝑦1,...,𝑦𝑇). Each 𝑦𝑡 represents ground truth at time 𝑡 and is defined as a set {(𝑐𝑖𝑡,𝑚𝑖𝑡)}𝑁

𝑖=1, where 𝑐𝑖𝑡 ∈ {1,...,𝐾} denotes the class label of the 𝑖-th object, 𝑚𝑖𝑡 ∈ {0,1}𝐻×𝑊 is its segmentation mask, and 𝑁gt is the number of foreground objects. Similarly, each 𝑦ˆ𝑡 = {(ˆ𝑐𝑖𝑡,𝑚ˆ 𝑖𝑡)}𝑁

det

𝑖=1 represents predictions at time 𝑡, where 𝑐ˆ ∈ {1,...,𝐾,bg} denotes the class label (bg indicates background), 𝑚ˆ ∈ {0,1}𝐻×𝑊 is the predicted mask, and 𝑁det is the number of object queries. We observe that video segmentation can be reformulated within an autoregressive framework used in modern

| |ℳ2| |
|---|---|---|

|𝒜| |
|---|---|
| | |

| |ℳ| |
|---|---|---|

| |ℳ0| |
|---|---|---|

| |𝒜0| | |
|---|---|---|---|
| | | | |

| |ℳ1| |
|---|---|---|

| |𝒜1| | |
|---|---|---|---|
| | | | |

| | |
|---|---|
|Pixel Decoder| |

| | |
|---|---|
|Pixel Decoder| |

|History<br><br>Marker| |
|---|---|
| | |

|Pixel Decoder|
|---|

History Marker

History Marker

| | |
|---|---|
|History Decoder| |

| | |
|---|---|
|History Decoder| |

|History<br><br>Comp.| |
|---|---|
| | |

|History<br><br>Comp.| |
|---|---|
| | |

|History<br><br>Comp.| |
|---|---|
| | |

|History Decoder|
|---|

|…|
|---|

|𝑋0|
|---|

|𝑋1|
|---|

|𝑋2|
|---|

|𝑋1|
|---|

|𝑋2|
|---|

|𝑋3|
|---|

|𝑋4|
|---|

|𝑋5|
|---|

Inference

Training

Figure 1: High-level overview of AUSM during training and inference. Left: At inference time, AUSM processes frames in a recurrent manner with constant decoding time per frame. Rather than maintaining an explicit memory buffer, temporal history is compressed via the Mamba layer within the History Compressor module and passed through time. Right: During training, a parallel formulation is used to jointly optimize over multiple frames, enabling efficient and scalable learning.

LLMs (Mann et al., 2020; Grattafiori et al., 2024). Language models generate sequences by conditioning each token on all previous tokens:

∏︁𝑇

𝑃(𝑦𝑡 | 𝑦<𝑡). (1)

𝑃(𝑦1:𝑇) =

𝑡=1

The sequential nature of language naturally aligns with this autoregressive formulation, which provides an elegant unification of different tasks in language into a single universal architecture (Radford et al., 2019). Similarly, the segmentation of a streaming video can be expressed as

∏︁𝑇

##### 𝑃(𝑦𝑡 | 𝑦0,𝑦<𝑡,ℐ≤𝑡), (2)

𝑃(𝑦1:𝑇 | ℐ1:𝑇) =

𝑡=1

which explicitly models the dependence of each frame’s segmentation 𝑦𝑡 on the current frame ℐ𝑡, all previous frames ℐ<𝑡, previous segmentations 𝑦<𝑡, and potentially an initial prompt 𝑦0 (Pont-Tuset et al., 2017). This formulation covers all video segmentation tasks: prompted video segmentation begins with an initial human prompt 𝑦0 = 𝑚0, while no initial prompt is given for unprompted video segmentation (𝑦0 = ∅) (Yang et al., 2019; Kim et al., 2020).

- 2.2. AUSM Architecture The design of AUSM allows sophisticated conditioning on past history while maintaining constant memory to

process arbitrarily long video sequences. As shown in Alg. 1, AUSM uses a pool of object queries 𝒱 ∈ R𝑁

det×𝐷 and keeps a set of buffer ID vectors ℬ ∈ R𝑁

id×𝐷 that are essential for tracking identified instances, where 𝑁id is the total number of available ID vectors. Additionally, AUSM maintains two sets during inference: 𝒜 for storing allocated ID vectors and ℳ for storing previous mask predictions. These sets maintain a one-to-one mapping, ensuring that 𝒜𝑖𝑡 (𝑖-th element of 𝒜𝑡) corresponds to ℳ𝑖𝑡 (𝑖-th element of ℳ𝑡), thus |𝒜𝑡| = |ℳ𝑡| at all times. We assume each frame is encoded by a frame-independent backbone, yielding features {𝑋1,...,𝑋𝑇} where 𝑋𝑡 = backbone(ℐ𝑡) ∈ R𝐻×𝑊×𝐷 for 𝑡 ∈ {1,...,𝑇}, and 𝐷 is the channel dimension. Additionally, we define 𝑋0 ∈ R𝐻×𝑊×𝐷 by spatially repeating a 𝐷-dimensional trainable vector.

Unification of Tasks. By altering the initialization of 𝒜 and ℳ, AUSM handles both prompted and unprompted video segmentation. For unprompted video segmentation, both 𝒜0 and ℳ0 are initialized as empty sets (Line

- 3 in Alg. 1). The initialization for prompted video segmentation involves Sampler(ℬ,𝑛), which uniformly samples 𝑛 vectors from ℬ without replacement and returns them as a matrix whose rows are the sampled

- Algorithm 1 AUSM in Recurrent Form for Inference.

- 1: Initialize ℬ ∈ R𝑁id×𝐷, 𝒱 ∈ R𝑁det×𝐷
- 2: if 𝑦0 = ∅ then
- 3: 𝒜0, ℳ0 ← ∅, ∅
- 4: ℬ0 ← ℬ
- 5: else
- 6: 𝒜0 ← Sampler(ℬ, |𝑦0|)
- 7: ℳ0 ← 𝑚0
- 8: ℬ0 ← ℬ ∖ 𝒜0
- 9: end if
- 10: for 𝑡 = 1 to 𝑇 do
- 11: 𝐸𝑡 ← 𝑋𝑡−1 + HistoryMarker (𝒜𝑡−1, ℳ𝑡−1)

- 12: 𝐹𝑡 ← HistoryCompressor (𝐸𝑡)

- 13: 𝐺𝑡 ← HistoryDecoder (𝑄 = 𝑋𝑡, 𝐾𝑉 = 𝐹𝑡)

- 14: 𝑦^𝑡trk, 𝑦^𝑡det ← PixelDecoder (𝑄 = concat(𝒜𝑡−1, 𝒱), 𝐾𝑉 = 𝐺𝑡)

- 15: 𝒟 ← filter_fg(^𝑦𝑡det)
- 16: 𝒜′ ← Sampler(ℬ𝑡−1, |𝒟|)
- 17: 𝒜𝑡 ← concat(𝒜𝑡−1, 𝒜′)
- 18: ℬ𝑡 ← ℬ𝑡−1 ∖ 𝒜′
- 19: ℳ𝑡 ← concat( ^𝑚trk𝑡 , 𝒟)
- 20: end for

vectors. Formally,

Sampler(ℬ,𝑛) = [𝑏1,...,𝑏𝑛]⊤ ∈ R𝑛×𝐷, where (𝑏1,...,𝑏𝑛) ∼ Uniform{(𝑏′1,...,𝑏′𝑛) ∈ ℬ𝑛 : 𝑏′𝑖 ̸= 𝑏′𝑗,∀𝑖 ̸= 𝑗}.

In the prompted setting, 𝒜0 = Sampler(ℬ,|𝑦0|) and ℳ0 = 𝑚0, while ℬ0 = ℬ ∖ 𝒜0 (Lines 6–7 in Alg. 1).

History Marker. The vectorization of objects widely used in VIS (Heo et al., 2022; Li et al., 2024; Heo et al., 2023) loses spatial detail due to excessive compression of object-wise information. In contrast, History Marker preserves fine details similarly to memory-based methods in VOS (Oh et al., 2019). Specifically, it leverages Token Mark (Heo et al., 2025) to dissolve instance masks into a spatial feature map, minimizing information loss (Yang et al., 2021; Heo et al., 2025). At time 𝑡, given allocated vectors 𝒜𝑡−1 and segmentation masks ℳ𝑡−1, this module operates as:

##### HistoryMarker(𝒜𝑡−1,ℳ𝑡−1) = 𝑆𝑡 ∈ R𝐻×𝑊×𝐷, 𝑆𝑡[ℎ,𝑤,:] = ∑︀|𝒜

𝑡−1|

𝑖=1 ℳ𝑖𝑡−1[ℎ,𝑤] · 𝒜𝑖𝑡−1 𝜖 +

,

∑︀|𝒜

𝑡−1|

𝑖=1 ℳ𝑖𝑡−1[ℎ,𝑤]

where 𝜖 is a small number to prevent division by zero. The output 𝑆𝑡 is added to 𝑋𝑡−1 to form 𝐸𝑡, which is then passed to the History Compressor.

History Compressor. This module encodes visual features from previous frames and instance-specific masks into a single spatial state, enabling inference over arbitrarily long videos with constant memory. As shown in Fig. 2, each layer in the History Compressor consists of three components: Mamba (Gu and Dao, 2023), SelfAttention (Vaswani et al., 2017), and a feed-forward network. By decomposing video features into spatial and temporal dimensions, these components operate on different axes: Mamba processes the temporal dimension while self-attention handles the spatial dimension. We choose Mamba for the temporal dimension for two reasons: (1) videos are inherently sequential in time, aligning with SSM architectures; and (2) modeling videos is memory-intensive on GPUs because each frame contributes many tokens. The recurrent design of Mamba enables a single spatial state that is updated each frame, eliminating the need to store spatio-temporal features from all previous frames.

History Decoder. The History Decoder is a stack of Transformer decoder layers that outputs 𝐺𝑡 by taking the current-frame features 𝑋𝑡 as queries and the compressed state 𝐹𝑡 as keys and values. This spatial feature 𝐺𝑡 has two important properties: (1) it incorporates current-frame information through the image encoder, and (2) it retains fine-grained information about objects from previous frames through the compressed state.

|History Decoder| |
|---|---|
| |𝐺|

𝑋

𝐹

FFN

HistoryCompressor

|Spatial Self-Attention|
|---|

Pixel Decoder and Update Process. The Pixel Decoder follows Cheng et al. (2022), comprising Transformer decoder layers with masked attention, and takes object queries as in Hungarian matching–based detection methods (Carion et al., 2020; Cheng et al., 2021; Hwang et al., 2021). Specifically, the final predictions for frame 𝑡 are obtained using both previously allocated ID vectors 𝒜𝑡−1 and object queries 𝒱 as inputs, with 𝐺𝑡 providing keys and values. Each ID vector in 𝒜𝑡−1 tracks its corresponding object, and 𝒱 detects instances not yet assigned to any vector in 𝒜𝑡−1. Therefore, this process yields two types of predictions: tracking predictions 𝑦ˆ𝑡trk from 𝒜𝑡−1 and detection predictions 𝑦ˆ𝑡det from 𝒱.

| |T × HW × D|
|---|---|
|Temporal Mamba| |

HW × T × D

𝐸

Figure 2: HistoryCompressor module. Mamba encodes temporal dependencies, while self-attention captures spatial structure, enabling recurrent compression of 𝐸 with constant memory. Specifically, the temporal mamba layer operates pixel-wise, mixing information of each pixel throughout the time dimension 𝑇. The spatial self-attention layer, by contrast, is frame-independent that fuses information spanning over 𝐻𝑊 pixels.

After prediction, the sets 𝒜, ℬ, and ℳ are updated for the next frame. We define the following operations used in Alg. 1:

- • filter_fg(ˆ𝑦𝑡det): Filters predictions from 𝑦ˆ𝑡det to retain only foreground objects, returning a set 𝒟 of newly detected objects.
- • concat(𝐴,𝐵): Concatenates two sets of vectors 𝐴 and 𝐵.

The update process proceeds as follows: (1) foreground detections 𝒟 are filtered from 𝑦ˆ𝑡det; (2) |𝒟| new vectors 𝒜′ are sampled from the remaining buffer ℬ𝑡−1; (3) 𝒜𝑡 is formed by concatenating 𝒜𝑡−1 and 𝒜′, while ℬ𝑡 is obtained by removing 𝒜′ from ℬ𝑡−1; (4) ℳ𝑡 is updated by concatenating mask predictions from 𝑦ˆ𝑡trk and from the newly detected objects 𝒟.

#### 2.3. Parallel Training as Language Models

- Algorithm 2 AUSM in Parallel Form for Training. 𝐴𝐵:𝐵+𝐶 denotes (𝐴𝐵,...,𝐴𝐵+𝐶) for brevity.

- 1: Initialize ℬ ∈ R𝑁id×𝐷, 𝒱 ∈ R𝑁det×𝐷
- 2: 𝒜 ← Sampler(ℬ, 𝑁gt)
- 3: (𝑦1:trk𝑇, 𝑦1:det𝑇), 𝒜0:𝑇−1, ℳ0:𝑇−1 ← Preprocess(𝑦1:𝑇, 𝒜)
- 4: 𝐸1:𝑇 ← (︁𝑋𝑡−1 + HistoryMarker (𝒜𝑡−1, ℳ𝑡−1))︁

1:𝑇

- 5: 𝐹1:𝑇 ← HistoryCompressor (𝐸1:𝑇)

- 6: 𝐺1:𝑇 ← HistoryDecoder (𝑄 = 𝑋𝑡, 𝐾𝑉 = 𝐹𝑡)1:𝑇

- 7:

(︁𝑦^𝑡trk, 𝑦^𝑡det)︁𝑇

𝑡=1

← PixelDecoder (𝑄 = Concat(𝒜𝑡−1, 𝒱), 𝐾𝑉 = 𝐺𝑡)1:𝑇

- 8: ℒtotal =

[︁𝐿trk(𝑦𝑡trk, 𝑦^𝑡trk) + 𝐿det(𝑦𝑡det, 𝑦^𝑡det)]︁

∑︀𝑇

𝑡=1

Modern decoder-only language models Mann et al. (2020); Grattafiori et al. (2024) significantly benefit from parallel training by adopting building blocks (e.g., Transformers and SSMs) that support the teacher-forcing technique Sutskever et al. (2014). However, this parallel training cannot be readily applied to existing video segmentation methods that rely on frame-by-frame propagation of outputs, such as those employing query propagation (Meinhardt et al., 2022; Heo et al., 2023, 2025). Therefore, existing frameworks train video segmentation models by recurrently processing frames, leading to severely inefficient training.

In contrast, as shown in Alg. 2, all modules in AUSM are compatible with teacher forcing. Therefore, AUSM supports parallel training across the temporal dimension, yielding substantial improvements in training efficiency. The parallel training begins by sampling 𝑁gt vectors from the buffer ℬ to form the set 𝒜, with a one-to-one mapping to ground-truth instances. A critical component is the Preprocess function, which prepares 𝑦1:trk𝑇, 𝑦1:det𝑇, ℳ0:𝑇−1, and 𝒜0:𝑇−1.

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

As illustrated in Fig. 3, Preprocess randomly samples a timestep 𝑡𝑖sample for each instance 𝑖 (highlighted with colored contours). This timestep determines when an instance shifts from being treated as a detection target to a tracking target. Using the ground truth 𝑦1:𝑖 𝑇 and sampled index 𝑡𝑖sample, it constructs, for each instance 𝑖 and frame 𝑡,

!!"#!

!!!$%

𝑦𝑡det,𝑖 = {︃𝑦𝑡𝑖 if 𝑡 ≤ 𝑡𝑖sample ∅ otherwise

ℳ!

𝑦𝑡trk,𝑖 = {︃𝑦𝑡𝑖 if 𝑡 > 𝑡𝑖sample ∅ otherwise

#!

ℳ𝑖𝑡 = {︃𝑦𝑡𝑖 if 𝑡 ≥ 𝑡𝑖sample ∅ otherwise

! = 1 ! = 2 ! = 3 ! = 4

Figure 3: Schematic of Preprocess. The video contains three ground-truth instances: person1 (the person riding the horse), person2 (the person standing), and horse. Each instance is matched with a vector from 𝒜, represented by colored circles: red for person1, green for person2, and blue for horse. The highlighted contours indicate the randomly sampled timesteps (𝑡𝑖sample) for each instance.

The set 𝒜𝑡−1 is then constructed to contain vectors corresponding to foreground instances in ℳ𝑡−1, maintaining the one-to-one mapping between allocated vectors and instances.

Once preprocessing is complete, all subsequent operations, from applying History Marker to calculating losses, can be executed in parallel across frames. The training loss decomposes into tracking loss 𝐿trk and detection loss 𝐿det. For tracking, the one-to-one mapping between 𝑦𝑡trk and 𝒜𝑡−1 enables direct loss computation. For detection, where no predetermined mapping exists between 𝑦𝑡det and the detection queries 𝒱, we employ the Hungarian algorithm (Carion et al., 2020) to obtain optimal assignments before computing the loss.

### 3. Experiments

- 3.1. Datasets Training Datasets. We train AUSM on diverse public segmentation datasets encompassing both prompted and unprompted paradigms. Specifically, we use COCO (Lin et al., 2014), DAVIS 2017 (Pont-Tuset et al., 2017), MOSE (Ding et al., 2023), SA-V (Ravi et al., 2025), YouTube-VIS 2019 & 2021 (Yang et al., 2019), and OVIS (Qi et al., 2022). For datasets that include semantic labels (Lin et al., 2014; Yang et al., 2019; Qi et al., 2022), we provide an additional classification objective so the model learns category-aware segmentation.

Evaluation Benchmarks and Metrics. For prompted tasks, we follow established semi-supervised VOS protocols on four benchmarks: DAVIS 2017, YouTube-VOS 2018 & 2019 (Xu et al., 2018), and MOSE, which focuses on multi-object segmentation with complex instance interactions. For DAVIS and MOSE, we report the standard 𝒥 &ℱ metric, which averages region similarity and contour accuracy. For YouTube-VOS, we report 𝒢, the average of 𝒥 &ℱ computed across both seen and unseen categories. For unprompted tasks, we evaluate on VIS benchmarks: YouTube-VIS 2019 & 2021 and OVIS, the latter featuring heavy occlusion and longer videos. Following standard practice, we report Average Precision (AP).

Table 1: Quantitative results on Prompted and Unprompted Video Segmentation benchmarks. We report 𝒥 &ℱ for DAVIS and MOSE, 𝒢 for YouTube-VOS, and AP for YouTube-VIS and OVIS. † denotes methods additionally trained on private datasets. "✗" indicates tasks that are architecturally incompatible with the models. "–" denotes tasks that are feasible to handle with the models but for which results are not reported.

Prompted Unprompted DAVIS MOSE YTVOS18 YTVOS19 YTVIS19 YTVIS21 OVIS

Method Backbone

Task-specialized Models Xmem Cheng and Schwing (2022) ResNet50 86.2 57.6 85.7 85.5 ✗ ✗ ✗ DeAOT Yang and Yang (2022) ResNet50 85.2 59.4 86.0 85.9 ✗ ✗ ✗ DeAOT Yang and Yang (2022) Swin-B 86.2 – 86.2 86.1 ✗ ✗ ✗ SAM2 Ravi et al. (2025)† Hiera-B+ 90.2 76.6 – 88.6 ✗ ✗ ✗ SAM2 Ravi et al. (2025)† Hiera-L 90.7 77.9 – 89.3 ✗ ✗ ✗ UniRef++ Wu et al. (2023) Swin-L 83.9 59.0 83.2 83.0 ✗ ✗ ✗ GenVIS Heo et al. (2023) Swin-L ✗ ✗ ✗ ✗ 64.0 59.6 45.2 DVIS Zhang et al. (2023) Swin-L ✗ ✗ ✗ ✗ 63.9 58.7 47.1 VISAGE Kim et al. (2024) Swin-L ✗ ✗ ✗ ✗ 64.2 59.6 46.5 Video K-Net Li et al. (2022) Swin-B ✗ ✗ ✗ ✗ 54.1 – –

###### Universal Offline Models

TarViS Athar et al. (2023) Swin-T 82.8 – – – – 50.9 34.0 TarViS Athar et al. (2023) Swin-L 85.3 – – – – 60.2 43.2

###### Universal Streaming Models

UNINEXT Yan et al. (2023) ResNet50 74.5 – 77.0 – 53.0 – 34.0 UNINEXT Yan et al. (2023) ConvNeXt-L 77.2 – 78.1 – 64.3 – 41.1 UniVS Li et al. (2024) Swin-T 71.7 – 70.3 – 52.4 51.6 33.0 UniVS Li et al. (2024) Swin-B 75.0 – 70.9 – 57.8 56.5 39.0 UniVS Li et al. (2024) Swin-L 76.2 – 71.5 – 60.0 57.9 41.7 AUSM Swin-T 76.4 58.8 79.5 78.3 54.9 52.1 39.4 AUSM Swin-B 81.6 62.1 80.2 79.1 62.6 58.6 45.5

- 3.2. Training Details A core design choice in AUSM is enabling parallel training over frame sequences. Our architecture processes all frames in a training clip concurrently while preserving the autoregressive training objective over the output sequence. The training process is divided into three stages, progressively increasing temporal complexity and data diversity.

- Stage 1 (Pseudo-video pretraining): We pretrain AUSM on COCO (Lin et al., 2014) using a pseudo-video augmentation strategy (COCO-pseudo). Each image is transformed into a 3-frame sequence via random spatial augmentations following (Heo et al., 2022).
- Stage 2 (Multi-source short-clip training): This stage introduces real video data and train on 5-frame clips sampled from a mixture of COCO-pseudo, MOSE (Ding et al., 2023), SA-V (Ravi et al., 2025), YouTube-VIS 2019 & 2021 (Yang et al., 2019), and OVIS (Qi et al., 2022).
- Stage 3 (Long-clip adaptation): We fine-tune AUSM on 16-frame clips to strengthen long-range temporal modeling. To reduce memory usage, we freeze the image backbone and update only the temporal modules and prediction heads. This stage uses the same datasets as Stage 2, with the addition of the DAVIS 2017 training set (Pont-Tuset et al., 2017).

- 3.3. Main Results Table 1 reports AUSM’s performance across a range of prompted and unprompted video segmentation benchmarks. We split the table into two categories: (1) specialized methods tailored to a single setting – prompted or unprompted – and (2) universal frameworks that support both within one architecture. All results of AUSM are obtained with a single model trained using our joint learning framework, without

10

+1.9 (2.5%) +3.1 (4.1%)

Iterative Training

80.2

Parallel Training

79.1

78.3

80

###### TrainingTime(sec/iter)

76.0

8

###### PerformanceScore

70

+4.5 (7.8%)

2.5×

6

62.1

60

57.6

4

2.1×

+5.2 (12.9%)

50

1.6×

2

45.5

- Stage 2 (5 frames)

- Stage 3 (16 frames)

40.3

40

0

MOSE YTVOS18 YTVOS19 OVIS

1 2 4 8 16

Dataset

Sequence Length

Figure 5: Performance comparison between Stage 2 (5-frame) and Stage 3 (16-frame) training across four benchmark datasets.

Figure 4: Comparison of training time (sec/iter) between iterative and parallel training approaches across different sequence lengths.

task-specific fine-tuning. We report AUSM with Swin-T and Swin-B backbones (Liu et al., 2021).

Prompted Video Segmentation. A leading specialized method for prompted video segmentation is SAM2 (Ravi et al., 2025), a memory-based mask-propagation approach built on STM (Oh et al., 2019), where each object is processed independently using a dedicated memory buffer. With additional private data, SAM2 attains strong performance across standard benchmarks. In contrast, AUSM processes multiple objects jointly in a single forward pass and operates without explicit memory buffers. Among online universal frameworks, AUSM performs strongly, demonstrating the benefits of our autoregressive formulation. Notably, relative to UniVS (Li

- et al., 2024), we surpass its Swin-L variant on YouTube-VOS 2018 by +8.7 (in 𝒢) despite using a smaller Swin-B backbone. This highlights AUSM as a unified, memory-efficient alternative that balances generality and scalability.

Unprompted Video Segmentation. Specialized VIS models achieve strong results across benchmarks, but their decoupled designs, often tailored for object-query propagation (Heo et al., 2023) or short-term tracking during training (Kim et al., 2024), limit extensibility to prompted tasks. Despite being universal, AUSM is competitive with these specialized approaches, capturing complex object dynamics without taskspecific architectural constraints. This underscores the strength of our autoregressive formulation in handling unprompted segmentation while preserving compatibility with prompted settings. Notably, AUSM attains the highest OVIS score among universal models, a dataset characterized by heavy occlusion and long-range interactions.

- 3.4. Ablation Studies Parallel vs. Iterative Training Efficiency. To quantify the benefit of parallel training, we measure training time per iteration (s/iter) over sequence lengths of 1, 2, 4, 8, and 16 using the Swin-B backbone. As shown in Fig. 4, our parallel approach scales substantially better with increasing sequence length compared to the iterative baseline. Whereas the iterative method grows from 1.47s to 8.75s per iteration, our parallel approach increases only from 1.47s to 3.45s at length 16. This yields a 2.5× speedup at sequence length 16, with larger gains expected at longer horizons. These results highlight the efficiency and scalability of our parallel training pipeline, making it especially well-suited for learning from long video sequences where temporal modeling is critical.

Effect of Training with Longer Sequences. We compare Stage 2 (5-frame clips) and Stage 3 (16-frame clips) to assess our long-clip adaptation strategy. As shown in Fig. 5, the transition to longer sequences consistently improves performance across all datasets. The largest gains are on MOSE (+4.52) and OVIS (+5.2),

indicating that longer temporal context improves modeling of complex dynamics and appearance changes. These improvements are achieved without any explicit memory buffer (e.g., FIFO-style spatio-temporal caches); instead, our History Comperssor enables efficient long-term reasoning with constant memory.

Table 2: Effect of varying the foreground threshold during inference on Unprompted Video Segmentation performance.

Effect of Foreground Threshold at Inference. In unprompted inference, new objects enter the buffer via filter_fg(ˆ𝑦𝑡det), which selects confident detections using a foreground-probability threshold. As shown in Tab. 2, performance is relatively robust across thresholds from 0.3 to 0.7. While OVIS benefits from a higher threshold (up to 46.5 AP at 0.7), YouTube-VIS performs best around 0.4–0.5. We therefore use a fixed threshold of 0.5 for all benchmarks to ensure consistency and avoid dataset-specific tuning.

Thres. YTVIS19 YTVIS21 OVIS

- 0.3 62.4 57.8 44.5

- 0.4 62.6 58.1 45.1

- 0.5 62.6 58.6 45.5

- 0.6 61.8 58.1 46.4

- 0.7 61.6 57.9 46.5

Scaling Inference Compute in AUSM. Recent work in language modeling shows that increasing inference-time computation can improve accuracy (Wei et al., 2022); even simple input-repetition strategies (Springer et al., 2024; Arora et al., 2024) help across architectures (Vaswani et al., 2017; Gu and Dao, 2023). Inspired by this, we scale inference compute for AUSM in both images and videos by constructing repeated sequences that allow iterative refinement.

Table 3: Results measured using Swin-B backbone. We relegate more experimental details in the supplementary.

For a single image ℐ, we construct a pseudo-video sequence ℐaug = (ℐ1,...,ℐ𝑇) where ∀𝑡 ∈ {1,...,𝑇},ℐ𝑡 = ℐ. Processing this sequence encourages the model to refine predictions by revisiting uncertain regions and leveraging object interactions over repetitions. Similarly, we extend this concept to video sequences as well. Given a video (ℐ1,...,ℐ𝑇), we construct an augmented sequence by repeating the video frames as:

# Repetition COCO YTVIS19

- ×1 34.2 62.6
- ×2 34.9 63.3
- ×3 35.0 63.5

ℐaug = (ℐ1,...,ℐ𝑇,ℐ𝑇−1,...,ℐ1,ℐ2,...,ℐ𝑇)

When processing this augmented sequence, we take predictions from only the final 𝑇 frames as the refined output.

As shown in Tab. 3, scaling inference compute improves performance. While the primary focus of AUSM is unifying video segmentation tasks within one framework, these results show it can also benefit from test-time compute scaling for further gains in both images and videos.

### 4. Related Work

Unprompted Video Segmentation. A key distinction among existing models is whether they condition on past predictions 𝑦ˆ<𝑡 to generate the current output during inference. Conventional online tracking-by-detection approaches completely exclude 𝑦ˆ<𝑡 in their model design, treating each visual feature independently from previous predictions (Huang et al., 2022; Wu et al., 2022; Kim et al., 2024; Ying et al., 2023). While these methods benefit from parallelizable computation during training, they often require external memory banks to maintain temporal consistency at inference time.

In contrast, models such as GenVIS (Heo et al., 2023) adopt query propagation by conditioning on prior predictions 𝑦ˆ<𝑡 through object-level vector representations. Although this formulation enhances temporal coherence, object vectorization significantly degrades the granularity of mask predictions (Kim et al., 2024). RoCoVIS (Heo et al., 2025) addresses this by introducing instance mask propagation, which greatly improves mask quality. However, this design inherently breaks parallelism, as predictions must be generated sequentially,

leading to reduced training efficiency.

Another line of work follows an offline paradigm, where the entire video ℐ1:𝑇 is available in advance during inference (Wang et al., 2020; Hwang et al., 2021; Heo et al., 2022). While this setting enables models to access long-range context, such models are typically not conditioned on intermediate outputs 𝑦ˆ and therefore cannot refine predictions based on prior object states. As a result, despite high computational cost, these models often underperform recurrent methods (Wu et al., 2022) that explicitly leverage temporal feedback.

Prompted Video Segmentation. Prompted Video Segmentation focuses on segmenting and tracking an object throughout a video, given a specified target in the first frame, without requiring any class labels. Unlike Unprompted Segmentation tasks that involve discovering and classifying all objects, this task aims to track a user-specified target. Prompted Video Segmentation is highly practical, particularly for interactive and general-purpose applications.

One of the most influential paradigms is the Space-Time Memory Network (STM) (Oh et al., 2019), which has inspired a wide range of follow-up methods. STM maintains a memory bank consisting of past frames and their corresponding masks, and performs dense matching between the current frame and the stored memory to retrieve relevant information for accurate mask propagation. Building on this, XMem (Cheng and Schwing,

- 2022) introduces enhanced memory mechanisms for improved long-term tracking. More recently, SAM2 (Ravi

- et al., 2025), building upon SAM (Kirillov et al., 2023), supports more flexible input representations such as points and boxes, and also introduces large-scale datasets to enable broader evaluation and training. Another line of research explores hierarchical propagation using transformer-based architectures (Yang et al.,

- 2021; Yang and Yang, 2022). These methods gradually propagate identity information from past frames to the current frame through a hierarchical structure. It supports multi-object processing in a unified manner, unlike STM-based approaches that typically handle each object separately.

Universal Video Segmentation. The emergence of transformer-based detection and segmentation architectures (Carion et al., 2020; Wang et al., 2021; Cheng et al., 2022) has facilitated early attempts to unify multiple tasks within unprompted video segmentation (e.g. VIS, VPS, and VSS) under a single framework (Kim et al., 2022; Li et al., 2022). In parallel, similar efforts have been made in the prompted setting, aiming to jointly handle VOS and referring VOS within a shared architecture (Wu et al., 2023,).

Recent research has progressed toward bridging the gap between Unprompted and Prompted Video Segmentation. These efforts aim to develop universal models that can support both interaction-driven and fully automatic scenarios within a unified framework, thereby reducing reliance on task-specific designs and enabling broader applicability across diverse video understanding tasks. An offline method, TarViS (Athar et al., 2023), represents the first attempt to jointly model both settings by encoding task-specific targets as a set of queries. UNINEXT (Yan et al., 2023) further introduces a prompt-guided object discovery and retrieval paradigm. UniVS (Li et al., 2024) leverages prompts as queries by treating predicted masks from previous frames as visual prompts for the current frame.

State Space Models. State Space Models (SSMs) (Gu et al., 2021, 2022; Gu and Dao, 2023) have recently emerged as a promising alternative to Transformer (Vaswani et al., 2017) for sequence modeling. While Transformer requires storing all tokens in key-value caches, SSMs compress all history into a single state. This property gives SSMs constant computational complexity and memory requirements during inference, providing significant advantages for modeling long sequences compared to Transformers’ linearly growing complexity and memory.

### 5. Discussion

Our reformulation of video segmentation as an autoregressive modeling problem draws a direct conceptual connection to language modeling and opens several promising research directions. One central direction is

handling long sequences, with two complementary areas: (i) retrieval for long contexts (e.g., needle-in-ahaystack (NIAH) evaluations) (Hsieh et al., 2024) and (ii) length extrapolation (Press et al., 2021; Xiao et al.,

- 2023). Our model demonstrates strong performance and is capable of processing arbitrarily long streams. However, similar to LLMs, we observe performance degradation on extremely long sequences. Recent longsequence techniques from language modeling could be adapted to video to maintain quality and extend the effective context beyond the training sequence length.

Another promising direction is extending AUSM to additional video perception tasks. While this work focuses on both prompted and unprompted video segmentation (VOS and VIS), other tasks can be integrated with minimal modifications. For example:

- • Object tracking (Wu et al., 2013) and multi-object tracking (Milan et al., 2016): convert bounding boxes into mask prompts by filling corresponding regions with masks so they fit the same segmentation interface.
- • Taking the same approach as bounding boxes, other prompt styles (e.g., scribbles, points) can be unified by converting annotations into mask signals.
- • Referring video object segmentation (Seo et al., 2020): initialize the History Compressor’s state with text embeddings (e.g., from a frozen text encoder) corresponding to the language prompt.

We expect incorporating more tasks and data to yield further performance gains, and training on longer videos should also improve long-context modeling. Although our experiments train on clips up to 16 frames due to memory constraints, the framework is designed to scale and will benefit from next-generation hardware.

Limitations. While AUSM attains strong results – and among universal/online methods, state-of-the-art performance on unprompted VIS (e.g., OVIS) – we observe a modest gap on prompted VOS compared to specialized, memory-heavy systems. We believe these results are primarily from our architectural choice: most modules of AUSM take coarse frame features (e.g. stride of 8), which saves memory than finer features (e.g., stride of 4) and is better suited for object-level understanding, but marginally worse in capturing details. Future work could be a new video-specialized backbone to temporal modeling (e.g., reducing frame-independent layers while strengthening frame-dependent modules such as History Compressor/Decoder and prompt conditioning) which may close the VOS gap without sacrificing unprompted performance.

### 6. Conclusion

We introduce AUSM as a step toward a unified, scalable, and general-purpose formulation of video segmentation. Leveraging the connection between language modeling and video segmentation, AUSM supports parallel training and performs temporal modeling without explicit memory buffers, enabling efficient long-range reasoning with constant-memory inference. Across diverse benchmarks, our experiments show that autoregressive modeling provides a practical and scalable solution for video segmentation. We hope this perspective serves as both a strong baseline and a useful direction for future research in video perception.

Acknowledgements. We thank De-An Huang for feedback and helpful discussions.

### A. Implementation Details

We further summarize key architectural configurations and experimental setups used throughout the training and evaluation of AUSM.

Model Configuration. The number of object queries for detection 𝑁det and ID vectors 𝑁id are both set to 100. The History Compressor module consists of 6 layers, and the following Transformer decoder contains 6 layers. For spatio-temporal fusion in History Compressor, we use the 1/8 resolution feature map from the Swin backbone (Liu et al., 2021), with a fusion feature dimension of 256.

Our implementation does not use vision-language supervision (e.g., CLIP (Radford et al., 2021) or other pretrained image-text models); we instead employ dataset-specific classification heads. During training, each head is conditionally selected according to the dataset from which the input sample originates.

Training Setup. All experiments are conducted using 16 NVIDIA A100 GPUs with a batch size of 16 and an initial learning rate of 1 × 10−4, optimized using AdamW (Loshchilov and Hutter, 2019) across all stages.

We begin the training process with pseudo-video training on the COCO dataset (Lin et al., 2014), using image instance segmentation pretrained Mask2Former (Cheng et al., 2022) weights as initialization. This stage is trained for 20 epochs, corresponding to 147,500 iterations.

Next, we perform multi-source short-clip training using 5-frame video segments drawn from multiple datasets. This stage runs for 32,000 iterations and serves to adapt the model to short-term temporal dynamics and diverse visual domains.

Finally, we apply long-clip adaptation by fine-tuning the model on 16-frame clips for 40,000 iterations, enabling the model to better capture long-range temporal dependencies.

Inference Setup. Given a frame, we set the resolution of the shorter side to 1024. For Unprompted Video Segmentation, we apply a fixed foreground probability threshold of 0.5 to select confident detection predictions. Additionally, we perform top-𝑘 selection, retaining the top 10 instances per frame for YouTube-VIS 2019/2021 (Yang et al., 2019) and the top 20 for OVIS (Qi et al., 2022), to measure benchmark evaluation. These are the only post-processing steps employed; no other heuristics are introduced.

In contrast, prompted video segmentation involves no thresholding, filtering, or post-processing. Inference in this setting is fully model-driven, without reliance on manually designed components.

Implementation Details for Inference Scaling Experiments. We present additional methodological considerations for the scaling inference compute experiments. Due to the dense number of object appearing in the COCO validation set, we set dataset-specific foreground confidence thresholds to mitigate propagation of erroneous detections. Specifically, we use foreground confidence thresholds of 0.9 and 0.5 for COCO and YTVIS datasets, respectively. This stringent threshold for COCO results in a relatively modest performance metric (34.2 AP), as a considerable proportion of valid but lower-confidence detections are excluded from evaluation.

We further explore an alternative data augmentation strategy beyond simple image repetition. This approach involves strategic spatial decomposition of the input image into four overlapping quadrants: upper-left (ℐUL), upper-right (ℐUR), bottom-right (ℐBR), and bottom-left (ℐBL). Each quadrant maintains substantial spatial redundancy, with dimensions set to 90% of the original image size.

The resulting augmented sequence is formulated as:

ℐaug = (ℐ,ℐUL,ℐUR,ℐBR,ℐBL,ℐ) (3)

This configuration enables the model to systematically traverse local regions while maintaining global context through the inclusion of the full image at sequence boundaries. Empirical evaluation demonstrates that this

spatial traversal strategy yields statistically significant performance improvements, increasing mAP from 34.2 to 35.9 on the COCO validation set.

### B. Qualitative Results

In Fig. 6 and Fig. 7, we present qualitative results to illustrate the capability of AUSM in both prompted and unprompted video segmentation settings. Notably, a single trained model is used for all results without any task-specific tuning or architectural changes. This underscores the unified nature of AUSM, which can seamlessly switch between prompted and unprompted modes at inference time.

For prompted segmentation, we visualize the model’s ability to accurately segment target objects given masks in the keyframe. For unprompted segmentation, we show how the model discovers and tracks multiple objects throughout a video without external guidance. These results demonstrate that AUSM effectively handles both interaction-based and autonomous video understanding scenarios within a unified architecture.

Unprompted Mode

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

Prompted Mode

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

Unprompted Mode

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

Prompted Mode

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

Unprompted Mode

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

Prompted Mode

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

Unprompted Mode

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

Prompted Mode

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

Unprompted Mode

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

Prompted Mode

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

Unprompted Mode

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

Prompted Mode

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

### References

- [1] Simran Arora, Aman Timalsina, Aaryan Singhal, Benjamin Spector, Sabri Eyuboglu, Xinyi Zhao, Ashish Rao, Atri Rudra, and Christopher Ré. Just read twice: closing the recall gap for recurrent language models. arXiv preprint arXiv:2407.05483, 2024. 9
- [2] Ali Athar, Alexander Hermans, Jonathon Luiten, Deva Ramanan, and Bastian Leibe. Tarvis: A unified approach for target-based video segmentation. In CVPR, 2023. 1, 7, 10
- [3] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-to-end object detection with transformers. In ECCV, 2020. 5, 6, 10
- [4] Bowen Cheng, Alex Schwing, and Alexander Kirillov. Per-pixel classification is not all you need for semantic segmentation. In NeurIPS, 2021. 5
- [5] Bowen Cheng, Ishan Misra, Alexander G Schwing, Alexander Kirillov, and Rohit Girdhar. Masked-attention mask transformer for universal image segmentation. In CVPR, 2022. 5, 10, 12
- [6] Ho Kei Cheng and Alexander G Schwing. Xmem: Long-term video object segmentation with an atkinsonshiffrin memory model. In ECCV, 2022. 7, 10
- [7] Marius Cordts, Mohamed Omran, Sebastian Ramos, Timo Rehfeld, Markus Enzweiler, Rodrigo Benenson, Uwe Franke, Stefan Roth, and Bernt Schiele. The cityscapes dataset for semantic urban scene understanding. In CVPR, 2016. 2
- [8] Henghui Ding, Chang Liu, Shuting He, Xudong Jiang, Philip HS Torr, and Song Bai. Mose: A new dataset for video object segmentation in complex scenes. In ICCV, 2023. 2, 6, 7
- [9] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024. 3, 5
- [10] Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. arXiv preprint arXiv:2312.00752, 2023. 2, 4, 9, 10
- [11] Albert Gu, Karan Goel, and Christopher Ré. Efficiently modeling long sequences with structured state spaces. arXiv preprint arXiv:2111.00396, 2021. 10
- [12] Albert Gu, Karan Goel, Ankit Gupta, and Christopher Ré. On the parameterization and initialization of diagonal state space models. In NeurIPS, 2022. 10
- [13] Miran Heo, Sukjun Hwang, Seoung Wug Oh, Joon-Young Lee, and Seon Joo Kim. Vita: Video instance segmentation via object token association. In NeurIPS, 2022. 4, 7, 10
- [14] Miran Heo, Sukjun Hwang, Jeongseok Hyun, Hanjung Kim, Seoung Wug Oh, Joon-Young Lee, and Seon Joo Kim. A generalized framework for video instance segmentation. In CVPR, 2023. 4, 5, 7, 8, 9
- [15] Miran Heo, Min-Hung Chen, De-An Huang, Sifei Liu, Subhashree Radhakrishnan, Seon Joo Kim, YuChiang Frank Wang, and Ryo Hachiuma. Omni-rgpt: Unifying image and video region-level understanding via token marks. In CVPR, 2025. 2, 4
- [16] Miran Heo, Seoung Wug Oh, Seon Joo Kim, and Joon-Young Lee. Robust and consistent online video instance segmentation via instance mask propagation. In AAAI, 2025. 2, 4, 5, 9
- [17] Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, Yang Zhang, and Boris Ginsburg. Ruler: What’s the real context size of your long-context language models? arXiv preprint arXiv:2404.06654, 2024. 11

- [18] De-An Huang, Zhiding Yu, and Anima Anandkumar. Minvis: A minimal video instance segmentation framework without video-based training. In NeurIPS, 2022. 1, 9
- [19] Sukjun Hwang, Miran Heo, Seoung Wug Oh, and Seon Joo Kim. Video instance segmentation using inter-frame communication transformers. In NeurIPS, 2021. 5, 10
- [20] Dahun Kim, Sanghyun Woo, Joon-Young Lee, and In So Kweon. Video panoptic segmentation. In CVPR,

2020. 2, 3

- [21] Dahun Kim, Jun Xie, Huiyu Wang, Siyuan Qiao, Qihang Yu, Hong-Seok Kim, Hartwig Adam, In So Kweon, and Liang-Chieh Chen. Tubeformer-deeplab: Video mask transformer. In CVPR, 2022. 10
- [22] Hanjung Kim, Jaehyun Kang, Miran Heo, Sukjun Hwang, Seoung Wug Oh, and Seon Joo Kim. Visage: Video instance segmentation with appearance-guided enhancement. In ECCV, 2024. 1, 7, 8, 9
- [23] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollar, and Ross Girshick. Segment anything. In ICCV, 2023. 2, 10
- [24] Minghan Li, Shuai Li, Xindong Zhang, and Lei Zhang. Univs: Unified and universal video segmentation with prompts as queries. In CVPR, 2024. 1, 2, 4, 7, 8, 10
- [25] Xiangtai Li, Wenwei Zhang, Jiangmiao Pang, Kai Chen, Guangliang Cheng, Yunhai Tong, and Chen Change Loy. Video k-net: A simple, strong, and unified baseline for video segmentation. In CVPR, 2022. 7, 10
- [26] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In ECCV, 2014. 6, 7, 12
- [27] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In ICCV, 2021. 8, 12
- [28] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In ICLR, 2019. 12
- [29] Ben Mann, N Ryder, M Subbiah, J Kaplan, P Dhariwal, A Neelakantan, P Shyam, G Sastry, A Askell, S Agarwal, et al. Language models are few-shot learners. arXiv preprint arXiv:2005.14165, 1:3, 2020. 3, 5
- [30] Tim Meinhardt, Alexander Kirillov, Laura Leal-Taixe, and Christoph Feichtenhofer. Trackformer: Multiobject tracking with transformers. In CVPR, 2022. 5
- [31] Anton Milan, Laura Leal-Taixé, Ian Reid, Stefan Roth, and Konrad Schindler. Mot16: A benchmark for multi-object tracking. arXiv preprint arXiv:1603.00831, 2016. 11
- [32] Seoung Wug Oh, Joon-Young Lee, Ning Xu, and Seon Joo Kim. Video object segmentation using space-time memory networks. In ICCV, 2019. 2, 4, 8, 10
- [33] Jordi Pont-Tuset, Federico Perazzi, Sergi Caelles, Pablo Arbeláez, Alex Sorkine-Hornung, and Luc Van Gool. The 2017 davis challenge on video object segmentation. arXiv preprint arXiv:1704.00675, 2017. 2, 3, 6, 7
- [34] Ofir Press, Noah A Smith, and Mike Lewis. Train short, test long: Attention with linear biases enables input length extrapolation. arXiv preprint arXiv:2108.12409, 2021. 11
- [35] Jiyang Qi, Yan Gao, Yao Hu, Xinggang Wang, Xiaoyu Liu, Xiang Bai, Serge Belongie, Alan Yuille, Philip HS Torr, and Song Bai. Occluded video instance segmentation: A benchmark. IJCV, 2022. 2, 6, 7, 12
- [36] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019. 3

- [37] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, 2021. 12
- [38] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, Eric Mintun, Junting Pan, Kalyan Vasudev Alwala, Nicolas Carion, Chao-Yuan Wu, Ross Girshick, Piotr Dollár, and Christoph Feichtenhofer. Sam 2: Segment anything in images and videos. In ICLR, 2025. 2, 6, 7, 8, 10
- [39] Seonguk Seo, Joon-Young Lee, and Bohyung Han. Urvos: Unified referring video object segmentation network with a large-scale benchmark. In ECCV, 2020. 11
- [40] Jacob Mitchell Springer, Suhas Kotha, Daniel Fried, Graham Neubig, and Aditi Raghunathan. Repetition improves language model embeddings. arXiv preprint arXiv:2402.15449, 2024. 9
- [41] Ilya Sutskever, Oriol Vinyals, and Quoc V Le. Sequence to sequence learning with neural networks. In NeurIPS, 2014. 5
- [42] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In NeurIPS, 2017. 2, 4, 9, 10
- [43] Paul Voigtlaender, Michael Krause, Aljosa Osep, Jonathon Luiten, Berin Balachandar Gnana Sekar, Andreas Geiger, and Bastian Leibe. Mots: Multi-object tracking and segmentation. In CVPR, 2019. 2
- [44] Huiyu Wang, Yukun Zhu, Hartwig Adam, Alan Yuille, and Liang-Chieh Chen. Max-deeplab: End-to-end panoptic segmentation with mask transformers. In CVPR, 2021. 10
- [45] Yuqing Wang, Zhaoliang Xu, Xinlong Wang, Chunhua Shen, Baoshan Cheng, Hao Shen, and Huaxia Xia. End-to-end video instance segmentation with transformers. In CVPR, 2020. 10
- [46] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. In NeurIPS, 2022. 9
- [47] Jiannan Wu, Yi Jiang, Bin Yan, Huchuan Lu, Zehuan Yuan, and Ping Luo. Segment every reference object in spatial and temporal spaces. In ICCV, 2023. 10
- [48] Jiannan Wu, Yi Jiang, Bin Yan, Huchuan Lu, Zehuan Yuan, and Ping Luo. Uniref++: Segment every reference object in spatial and temporal spaces. arXiv preprint arXiv:2312.15715, 2023. 7, 10
- [49] Junfeng Wu, Qihao Liu, Yi Jiang, Song Bai, Alan Yuille, and Xiang Bai. In defense of online models for video instance segmentation. In ECCV, 2022. 1, 9, 10
- [50] Yi Wu, Jongwoo Lim, and Ming-Hsuan Yang. Online object tracking: A benchmark. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2411–2418, 2013. 11
- [51] Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. Efficient streaming language models with attention sinks. arXiv preprint arXiv:2309.17453, 2023. 11
- [52] Ning Xu, Linjie Yang, Yuchen Fan, Dingcheng Yue, Yuchen Liang, Jianchao Yang, and Thomas Huang. Youtube-vos: A large-scale video object segmentation benchmark. arXiv preprint arXiv:1809.03327, 2018. 2, 6
- [53] Bin Yan, Yi Jiang, Jiannan Wu, Dong Wang, Ping Luo, Zehuan Yuan, and Huchuan Lu. Universal instance perception as object discovery and retrieval. In CVPR, 2023. 1, 2, 7, 10
- [54] Linjie Yang, Yuchen Fan, and Ning Xu. Video instance segmentation. In ICCV, 2019. 1, 2, 3, 6, 7, 12

- [55] Zongxin Yang and Yi Yang. Decoupling features in hierarchical propagation for video object segmentation. In NeurIPS, 2022. 7, 10
- [56] Zongxin Yang, Yunchao Wei, and Yi Yang. Associating objects with transformers for video object segmentation. In NeurIPS, 2021. 2, 4, 10
- [57] Kaining Ying, Qing Zhong, Weian Mao, Zhenhua Wang, Hao Chen, Lin Yuanbo Wu, Yifan Liu, Chengxiang Fan, Yunzhi Zhuge, and Chunhua Shen. Ctvis: Consistent training for online video instance segmentation. In ICCV, 2023. 1, 9
- [58] Tao Zhang, Xingye Tian, Yu Wu, Shunping Ji, Xuebo Wang, Yuan Zhang, and Pengfei Wan. Dvis: Decoupled video instance segmentation framework. In ICCV, 2023. 7

