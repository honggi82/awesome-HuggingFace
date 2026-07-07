# arXiv:2603.26639v1[cs.CV]27Mar2026

## Make Geometry Matter for Spatial Reasoning

##### Shihua Zhang, Qiuhong Shen, Shizun Wang, Tianbo Pan, and Xinchao Wang⋆

National University of Singapore, Singapore suhzhang001@gmail.com, qiuhong.shen@u.nus.edu, shizun.wang@u.nus.edu, e1583392@u.nus.edu, xinchao@nus.edu.sg

Abstract. Empowered by large-scale training, vision-language models (VLMs) achieve strong image and video understanding, yet their ability to perform spatial reasoning in both static scenes and dynamic videos remains limited. Recent advances try to handle this limitation by injecting geometry tokens from pretrained 3D foundation models into VLMs. Nevertheless, we observe that naive token fusion followed by standard finetuning in this line of work often leaves such geometric cues underutilized for spatial reasoning, as VLMs tend to rely heavily on 2D visual cues. In this paper, we propose GeoSR, a framework designed to make geometry matter by encouraging VLMs to actively reason with geometry tokens. GeoSR introduces two key components: (1) Geometry-Unleashing Masking, which strategically masks portions of 2D vision tokens during training to weaken non-geometric shortcuts and force the model to consult geometry tokens for spatial reasoning; and (2) Geometry-Guided Fusion, a gated routing mechanism that adaptively amplifies geometry token contributions in regions where geometric evidence is critical. Together, these designs unleash the potential of geometry tokens for spatial reasoning tasks. Extensive experiments on both static and dynamic spatial reasoning benchmarks demonstrate that GeoSR consistently outperforms prior methods and establishes new state-of-the-art performance by effectively leveraging geometric information. The project page is available at https://suhzhang.github.io/GeoSR/.

Keywords: VLMs · Spatial reasoning · 3D foundation models

### 1 Introduction

Vision-language models (VLMs) that combine visual perception with language comprehension have advanced rapidly with large-scale training, showing strong capability in image and video understanding [8,16,26,42,43,47]. However, many real applications demand more than recognizing objects and events. They require spatial reasoning, namely answering questions about where things are, how they relate in 3D space, and how these relations evolve over time. This capability is essential in both static settings, where scenes remain largely rigid but viewpoints and visibility vary, and dynamic settings, where motion, occlusion, and temporal

⋆ Corresponding author.

continuity play central roles. Recent benchmarks consistently reveal a significant gap: while prevalent VLMs excel at general visual semantics, they become brittle when faced with spatial questions involving viewpoint changes, motion continuity, and quantitative spatiotemporal judgments [9,17,18,29,41,53,54,57].

A natural direction to bridge this gap is introducing geometric cues that provide 3D structural information for spatial reasoning. Early 3D-aware MLLMs often rely on explicit 3D inputs, such as depth signals [5,55], point clouds [12,40, 51,52], or pre-built 3D maps [20,39]. However, such pipelines typically require additional sensors or multi-stage 3D reconstruction processes, and the resulting geometry can be noisy, potentially degrading downstream performance and limiting scalability in the common scenarios where only monocular images or videos are available [38, 50]. Recent work has shifted toward a more scalable paradigm that extracts implicit geometric features from monocular videos via pretrained visual geometry grounding models [32,34,36], and injects them into VLMs as geometry tokens to complement vanilla vision tokens [7,38,50,53]. This geometry-token injection paradigm appears straightforward by fusing geometry tokens with visual tokens and then fine-tuning the model under spatial reasoning supervision. While these pretrained priors excel at visual geometry grounding and integrating them appears intuitively promising, a fundamental question remains unexplored. Do geometry tokens actually help spatial reasoning, or are they merely dispensable side signals?

Surprisingly, we observe a counterintuitive but reproducible phenomenon that under the common practice of naive token fusion followed by standard fine-tuning, geometry tokens are often underutilized for spatial reasoning. As illustrated in Figure 1, for static scenes, performance gains can be limited, and for dynamic scenes, injecting geometry tokens may even yield negative effects. This suggests that the models frequently default to appearance-driven shortcuts in 2D vision tokens, treating geometry as a dispensable auxiliary signal, rather than as actionable evidence for spatial reasoning. Moreover, indiscriminately mixing geometry with appearance may dilute useful geometric cues, thus the model still avoids consulting geometry when appearance shortcuts are barely sufficient. Clearly, it is insufficient to merely inject geometry tokens naively. We must compel the VLM to effectively rely on geometric information when spatial reasoning demands it.

Motivated by this principle, we propose GeoSR, a framework that makes geometry matter for spatial reasoning in both static and dynamic settings. GeoSR addresses the problem from two complementary angles. First, to make geometry usage effective, we introduce Geometry-Unleashing Masking. During training, it strategically masks parts of the 2D vision tokens to weaken nongeometric shortcuts, so the model is encouraged to consult geometry tokens to answer spatial questions. Second, to make geometry usage reasonable, we propose Geometry-Guided Fusion, a gated routing mechanism that adaptively increases the contribution of geometry tokens at regions where geometric evidence is needed, instead of treating geometry as uniformly helpful across all

[Figure 1]

[Figure 2]

- Fig. 1: Geometry injection can be underutilized and even harmful for spatial reasoning. We compare three variants: w/o Geo. (removing the geometry branch), w/ Geo. (injecting geometry tokens via naive fusion with standard fine-tuning), and Ours. The plots reveal a counterintuitive yet reproducible pattern: in static scenes, w/ Geo. brings only marginal gains over w/o Geo., whereas in dynamic videos, w/ Geo. can even underperform the w/o Geo. baseline. This suggests that VLMs often fall back to appearance-driven shortcuts in 2D visual tokens and treat geometry as a dispensable side signal. These observations motivate GeoSR (Ours), which compels models to use geometry as actionable evidence and yields consistent improvements across both static and dynamic spatial reasoning.

tokens and frames. Together, these designs make geometry tokens actionable and controllable, rather than being easily ignored or indiscriminately fused.

In summary, our main contributions are as follows.

- – We report a reproducible finding in spatial reasoning that implicit geometry injection can be ineffective under naive fusion and standard fine-tuning. Geometry tokens may be largely ignored, with limited or even negative gains on spatial reasoning in practice.
- – We propose GeoSR, a simple framework that encourages effective and reasonable use of geometry tokens. Geometry-Unleashing Masking reduces reliance on appearance shortcuts, and Geometry-Guided Fusion routes geometry information adaptively when geometric evidence is required.
- – We validate GeoSR on both static and dynamic spatial reasoning benchmarks, showing consistent improvements over prior methods.

### 2 Related Work

#### 2.1 General Video Understanding

Recent advances have extended vision-language models from static images to videos by representing a clip as multiple sampled frames and modeling temporal

context for tasks such as video captioning [19, 45] and VideoQA [44, 49]. Proprietary video-capable VLMs provide strong upper-bound baselines for general video understanding [6, 13, 24]. In parallel, several open-source models are explicitly tailored for video understanding, typically enhancing temporal modeling and long-context handling for multi-frame inputs [4,44,49]. Meanwhile, generalpurpose VLMs have also been rapidly extended to video inputs via multi-frame prompting and instruction tuning, offering broad coverage across diverse video tasks [2,3,35]. Despite strong performance on general video understanding, these models are often brittle on spatial reasoning [18,41,53]. This is because common supervision mainly rewards semantic alignment, making models unstable under viewpoint changes, occlusion, and motion, thus requiring geometry-grounded evidence for reliable spatial reasoning [18,41,53].

#### 2.2 Spatial Reasoning with VLMs

Spatial reasoning aims to answer questions about spatial layout and measurable relations, such as distance, direction, and motion changes over time. Recent benchmarks show that such queries remain challenging for modern VLMs, especially when viewpoints vary or relations evolve dynamically [17,18,41,53,54,57]. Static scenes. One of the common settings are reasoning in static scenes, where the environment is largely rigid while camera viewpoint and visibility vary across frames. VSI-Bench [41] is a representative benchmark for this regime and highlights the need for viewpoint-robust spatial understanding beyond frame-specific appearance cues. To strengthen spatial capability, one line of work scales spatial supervision through large spatial question-answer (QA) pairs and instruction tuning [22,46]. Another line of work injects geometric priors from pretrained 3D foundation models [32,34,36] and fuses them with 2D vision tokens [7,38,50]. VGLLM [50] extracts implicit geometry tokens from [32] and performs token-level fusion to improve spatial reasoning under viewpoint change. Spatial-MLLM [38] further argues that conventional semantic encoders under-emphasize structure, and proposes a spatial branch with space-aware frame sampling to better preserve geometry-relevant evidence under limited video context. VLM-3R [7] augments VLMs with instruction-aligned 3D reconstruction priors and integrates reconstruction-aware representations to improve static spatial reasoning. While these methods demonstrate the promise of geometry-aware enhancements, they often rely on uniform or naive fusion, which can leave geometry cues underutilized. Our work targets this gap by making injected geometry evidence more actionable during reasoning.

Dynamic scenes. Dynamic spatial reasoning, also referred to as 4D reasoning, considers scenarios where spatial relations change over time due to camera motion and object motion. Correct answers require spatiotemporal consistency rather than single-frame cues [53]. As an early attempt toward 4D scene understanding, LLaVA-4D [52] embeds spatiotemporal prompts into MLLMs to reason about evolving object states in 4D settings. Most closely related, GSM [53] retrieves question-relevant dynamic geometric evidence from pretrained dynamic geometry priors [36], and appends them to vision tokens as the input of VLM.

All these methods try to inject geometry cues to improve 4D reasoning. However, the former requires multiple videos as input and relies on explicit SfM-style reconstruction [23,48], which limits scalability in monocular in-the-wild videos. The latter actually shows limited gains from geometry injection, which will be discussed in our paper, highlighting that indiscriminate geometry fusion can be ineffective or even harmful without proper control. In contrast, our work focuses on making geometry tokens from monocular videos actionable for spatial reasoning.

### 3 Method

In this section, we first review the commonly used geometry injection framework for spatial reasoning, which augments a VLM with a pretrained geometry branch and a fusion module. We then present two key strategies in our GeoSR, Geometry-Unleashing Masking and Geometry-Guided Fusion, to make geometry tokens actionable rather than ignorable. Finally, we provide implementation details for both static and dynamic spatial reasoning.

#### 3.1 Preliminary: Geometry-Aware Framework

Traditional VLMs have to infer the 3D structure and dynamics of the scene from only 2D vision tokens for text-prompted spatial reasoning, which can be challenging and may lead to suboptimal performance. Accordingly, recent methods attempt to improve the geometric awareness of VLMs by introducing geometry tokens, forming a widely-used geometry-aware framework. The overall architecture of this framework is illustrated in Figure 2. Given a sequence of images or video frames {It|It ∈ RH

I×WI×3}Tt=1 and a text prompt Q, the goal is to answer the question based on the visual and textual information. For the tokenizer, except for the vision branch that encodes {It}Tt=1 to 2D tokens FV ∈ R(H

VWVT)×CV and the prompt branch that encodes Q to text tokens FP ∈ RL

P×CP, it introduces an additional geometry branch that extracts implicit geometry cues from {It}Tt=1 typically using a pretrained 3D model [32,34,36]. The geometry tokens FG ∈ R(H

GWGT)×CG are then merged with the 2D vision tokens FV in a fusion module F(·), and the fused tokens F ∈ RL×C are fed into the VLM backbone together with FP, generating the answer to the question. Generally, CV = CP = C are the same, while CG can be different. And the fusion module F(·) often varies across different methods. We specify this in the following.

- (1) Additive Fusion is commonly used in static spatial reasoning [50]: F = FV + MLP(Reshape(FG)), (1)

where Reshape(·) and MLP(·) are used to align the geometry tokens with the vision tokens in terms of resolution and dimensionality [50].

- (2) QFormer fusion, a more sophisticated module, is often needed to capture the complex spatiotemporal dynamics [53]. It first summarizes the text prompt’s

[Figure 3]

[Figure 4]

[Figure 5]

Vison-Language Model

Decoder

[Figure 6]

[Figure 7]

A: (1) Behind

𝑭

𝑭

Text Tokens

Fused Tokens

[Figure 8]

[Figure 9]

Fusion Module

Prompt Tokenizer

Q: Between 2s and 4s, from camera’s viewpoint, how does direction of camera to the car within bounding box (432, 180, 461, 192) change?

𝑭 𝑭G

Vision Tokens Geometry Tokens

[Figure 10]

[Figure 11]

Visual Tokenizer

Geometric Tokenizer

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

- A: (1) Behind
- B: (1) Behind|Left
- C: (1) Left. (2) Behind
- D: (1) Above

[Figure 17]

[Figure 18]

###### … …

[Figure 19]

[Figure 20]

Language Instructions 𝑄 Wild Video 𝑰

- Fig. 2: Overview of the Geometry-Aware Framework for spatial reasoning. It augments a VLM with an additional geometry branch. A pretrained geometric tokenizer extracts geometry tokens FG from the input video, which are fused with standard vision tokens FV by a fusion module to form F. The VLM answers the query based on the fused evidence together with the text tokens FP. Snow and flame icons denote frozen and trainable components, respectively. This paradigm serves as our baseline. Our key observation is that naive fusion under standard fine-tuning can leave FG underutilized, which motivates GeoSR to make geometry really matter for spatial reasoning.

intent using a small set of learnable bottleneck tokens B ∈ RL

B×C with crossattention [31]:

FB = CrossAttn1(B,FP), (2)

where B acts as the query set and FP serves as the key and value sets. LB is the length of bottleneck tokens, which is typically small to avoid high computational cost. Then QFormer queries these tokens among geometry tokens FG:

ZG = CrossAttn2(FB,Fˆ G) = CrossAttn2(FB,MLP(FG)), (3) FB is the query set, and FG ∈ R(H

GWGT)×CG is projected to Fˆ G ∈ R(H

GWGT)×C with MLP(·) to match the dimensionality, serving as the key and value sets. The output ZG is full of geometry cues, which is appended to the vision tokens:

F = [FV,MLP(ZG)], (4) where [·,·] denotes concatenation along the token dimension.

Note that for these spatiotemporal inputs, tokens are typically equipped with spatiotemporal positional encodings, such as Rotary Position Embedding

[Figure 21]

[Figure 22]

###### (a) Geometry-Unleashing Masking (b) Geometry-Guided Fusion

Vision Tokens

𝑭

𝑭

- (1) Random Mask
- (2) TopK Mask

| | |
|---|---|
| | |

[Figure 23]

C

𝑭G

Bottleneck 𝑩

𝒔

𝜶

𝑭

q k/v

𝒁

| | |
|---|---|
| | |

q k/v

Text Tokens 𝑭

Cross Att.

| | |
|---|---|
| | |

k/v Cross Att. q

Cross Att.

| | |
|---|---|
| | |

𝑭

MLP 𝑭G

| | |
|---|---|
| | |

Fused Tokens

Interpolation 𝑭G′

Geometry Tokens 𝑭G

- Fig. 3: Overview of the proposed strategies in GeoSR. (a) Geometry-Unleashing Masking suppresses appearance shortcuts during training by masking a subset of vision tokens FV. For static settings, the mask is sampled randomly. For dynamic settings, bottleneck tokens B first attend to text tokens FP to obtain FB, which then attend to geometry tokens FG to produce question-relevant geometry evidence ZG and a relevance score used for TopK masking. (b) Geometry-Guided Fusion redistributes the compact evidence ZG back to token-level geometry features (if applicable) and applies a learned gate α to control the contributions of masked vision features F˜ V and geometry features F˜ G, producing fused tokens F for the VLM backbone.

(RoPE) [25] and its spatiotemporal extensions [10,33], which we omit for clarity. In practice, all prompt, visual, and geometric tokenizers are typically kept frozen, while the fusion module F(·) is trained and the VLM backbone is finetuned on spatial reasoning datasets [7,38,50,53]. Albeit straightforward, we find that this common recipe does not make geometry tokens actively useful for spatial reasoning. The VLM can still rely on 2D visual cues and largely ignore the geometry stream, leading to suboptimal performance. And in some cases, performance even improves after removing the geometry branch, as shown in Fig. 1. To address this counterintuitive issue, we propose GeoSR with two strategies to encourage the VLM to reason with geometry cues, which will be introduced in the following sections.

#### 3.2 Geometry-Unleashing Masking

As discussed in Section 1, the core challenge is that the geometry-aware VLM may still answer spatial reasoning questions from 2D appearance shortcuts, even when geometry tokens are provided. We therefore introduce GeometryUnleashing Masking to suppress the shortcuts during training and force the model to effectively use geometric evidence. The detailed architecture of this is illustrated in Figure 3(a), where the masking process can be formulated as:

F˜ V = m ⊙ FV, (5)

- 0, if δ = 1 ∧ j ∈ M,
- 1, otherwise,

δ ∼ Bernoulli(β), (6)

mj =

VWVT) is a binary mask over vision tokens, M is the mask set, β controls the probability of enabling masking during training, and ⊙ denotes element-wise multiplication. So, now the question is how to define m in Eq. (5). A simple idea is to randomly mask vision tokens, such as MAE [11], which has been proven to be effective in many vision tasks [15,30,37]. Inspired by this, GeoSR uniformly samples M of size K from visual-token positions whose resolution is equal to HVWVT:

where m = {mj} ∈ {0,1}(H

M = Rand(HVWVT,K), K = ⌈γ(HVWVT)⌉, (7)

where γ is the masking ratio. Random masking is a simple yet effective choice for static spatial reasoning baselines that use Additive Fusion as in Eq. (1) [7,38, 50]. Meanwhile, for dynamic spatial reasoning baselines that use QFormer Fusion as in Eqs. (2) to (4) [53], GeoSR instead selects mask regions under this query-style mechanism. With Eqs. (2) and (3) that effectively condense the question-related geometric information into ZG, we can identify the most relevant 3D tokens for answering the question, ranking all tokens according to the relevance score s ∈ [0,1](HGWGT). Then with the score, we select the positions of the most critical geometry tokens and mask the corresponding 2D vision tokens during training, which encourages the model to seek these important geometric cues from the geometry stream instead of relying on 2D appearance shortcuts. We use the attention weights in Eq. (3) to derive the relevance score s for masking. This process can be formulated as:

ZG,A = CrossAttn2(FB,Fˆ G) = CrossAttn2(FB,MLP(FG)), (8)

LB

h

uj − min(u) max(u) − min(u) + ϵ

1 hLB

Ak,i,j. (9)

s = {sj} =

, uj =

i=1

k=1

Here, A = {Ak,i,j} ∈ [0,1]h×LB×(HGWGT) is the attention probability tensor, where h is the number of attention heads. ui is the average attention probability of each geometry token across all heads and bottleneck tokens, regarding as the relevance measurement, and we further normalize u to get the final relevance score s. ϵ is a small value to ensure numerical stability. Ideally, after obtaining s, we can directly identify the mask set on high-relevant locations with TopK selection:

M = TopK(s,K), K = ⌈γ(HGWGT)⌉, (10) which replaces the random mask in Eq. (7), and then perform masking process in Eqs. (5) and (6) for dynamic spatial reasoning. However, there is one more issue to be considered. Note that m in Eq (5) should satisfy the resolution of HV × WV × T, which is the same as FV, while M is obtained from FG with the resolution of HG × WG × T. The resolution may mismatch, silimar to Eq. (1). Therefore, the same 2D spatial interpolation may be needed here if the geometry tokens and vision tokens have different spatial resolutions:

Fˆ ′G = Interp(Fˆ G,(HV,WV)) = MLP(Reshape(Fˆ G)). (11)

In this case, Fˆ ′G will replace Fˆ G in Eq. (8) while others remain unchanged.

From Eqs. (5) to (11), the Geometry-Unleashing Masking attempts to suppress some of the vision tokens during training, thus forcing the model to unleash geometry from the 3D branch, making effective geometric information utilization. Then, the model has to fuse these geometric cues into the VLM backbone reasonably, fully leveraging their capabilities for both static and dynamic spatial reasoning.

#### 3.3 Geometry-Guided Fusion

Building on the above discussions, we next fuse the geometric cues with the visual tokens and feed the fused representations into the VLM backbone. However, uniformly mixing in Eq. (1) [38,50] or simply concatenating in Eq. (4) [53] still leaves the model an easy option to down-weight geometry everywhere and fall back to the remaining visual evidence. GeoSR therefore introduces GeometryGuided Fusion, as shown in Figure 3(b), which adaptively controls how much geometric evidence is injected into each fused token in a fine-grained manner, ensuring that geometry dominates when it is informative and necessary.

Generally, this designed module fuses the geometry features F˜ G (as defined in following Eqs. (13) and (14)) with the vision tokens F˜ V that will be masked during training (as in Eq. (5)) adaptively under the guidance of a learned gate mechanism, which allows the model to increase reliance on geometry when visual cues are suppressed and geometry is informative, rather than treating geometry as uniformly helpful. The gated fusion weights are computed as follows:

α = σ(Wg[V ∥G] + bg), V = LNv(F˜ V), G = LNg(F˜ G), (12)

where [·∥·] denotes concatenation along the channel dimension, Wg and bg are learnable parameters, and σ(·) is the sigmoid function. LN(·) here denotes layer normalization [1]. This operation outputs α ∈ (0,1)(H

VWVT)×C, which is a tokenand-channel-wise gate, enabling fine-grained routing of geometry evidence. For static spatial reasoning, the geometry feature is defined as:

F˜ G = MLP(Reshape(FG)), (13)

While for dynamic spatial reasoning that conducts Eq. (8), the compact geometry evidence ZG is first redistributed back to the full spatiotemporal geometry tokens Fˆ ′G aligned with the vision tokens (as in Eq. (11)). Then, Similar to Eqs. (2) and (8), we use cross-attention to retrieve location-aware fine-grained geometry features:

F˜ G = CrossAttn3 F ˆ ′G,ZG , (14)

where Fˆ ′G is the query set, and ZG ∈ RL

B×C serves as the key and value sets. The output F˜ G ∈ R(H

VWVT)×C is a fine-grained geometry feature map then fed to Eq. (12). This retrieval step emphasizes the dynamics condensed in ZG.

Finally, after Eq. (12), we fuse the two streams adaptively with the learned gate for both static and dynamic spatial reasoning:

F = α ⊙ V + (1 − α) ⊙ G, (15)

where ⊙ denotes an element-wise multiplication. F is the fused feature map, which inherits both visual and geometry cues reasonably under the control of the gate α. Typically, in static spatial reasoning, F will be fed into the VLM backbone directly, together with the prompt tokens, to achieve spatial reasoning. In dynamic spatial reasoning, the overall dynamic geometry information in ZG (as in Eq. (8)) is also helpful to perceive the global context, which is appended further to F as [F,ZG] and then fed into VLM in practice.

From Eqs. (14) to (15), Geometry-Guided Fusion adaptively mixes geometric cues with vision tokens using a learned gate α. This design allows geometry features to dominate at locations where they provide useful evidence.

#### 3.4 Implementation Details

This section provides implementation details of GeoSR, including training and inference settings for both static and dynamic spatial reasoning.

Static spatial reasoning. For static spatial reasoning, we use Qwen2.5-VL-7B as the backbone VLM [3] and VGGT [32] as the pretrained 3D model to extract geometry tokens. Following our static masking design from Eqs. (5) to (7), we adopt MAE-style random masking with masking ratio γ = 0.8 in Eq. (7) and masking enable probability β = 0.5 in Eq. (6). During fine-tuning, we update the VLM backbone together with the Geometry-Guided Fusion module, while keeping the tokenizers and the 3D geometry model frozen. We use the same data splits of SPAR-7M [46] and LLaVA-Hound [49] as in [50]. We train our model for 1 epoch with Adam [14] optimizer, a batch size of 64, and a learning rate of 1×10−5 with linear warmup for 150 steps followed by cosine decay to 0. Inference follows training except that masking is disabled and the full vision tokens are fed into the fusion module.

Dynamic spatial reasoning. For dynamic spatial reasoning, we keep the Qwen2.5-VL-7B backbone [3] but switch the 3D model to π3 [36] to better capture dynamic geometry cues. We follow the dynamic masking design in Eqs. (2) to (10) with bottleneck length LB = 32 in Eq. (2), masking ratio γ = 0.8 in Eq. (10), and enable masking with probability β = 0.5 in Eq. (6). The training set is DSR-Train [53] following its official protocol. We train the model for

- 1 epoch with Adam [14] optimizer, a batch size of 32, and a learning rate of
- 2 × 10−7 with linear warmup for 50 steps. Inference is the same as training except that masking is disabled.

All experiments are conducted on 4×H200 GPUs with 141GB memory each. Training takes about 14 hours for static spatial reasoning with DeepSpeed [21] ZeRO-2 and about 20 hours for dynamic spatial reasoning with ZeRO-3 Offload.

###### Table 1: Performance on static spatial reasoning. Best , second-best , and third-best results are highlighted.

Subtask Types

Models

Avg. Numerical Questions Multiple-Choice Questions

Obj. Cnt. Abs. Dist. Obj. Size Room Size Rel. Dist. Rel. Dir. Route Plan Appr. Order

Proprietary Models (API) GPT-4o [13] 43.1 34.1 68.6 64.2 48.3 43.1 29.4 51.3 47.8

- Gemini-1.5 Flash [27] 50.8 33.6 56.5 45.2 48.0 39.8 32.7 59.2 45.7

- Gemini-1.5-Pro [27] 49.6 28.8 58.6 49.4 46.0 48.1 42.0 68.0 48.8

General Video Understanding Models Qwen2.5-VL-7B [3] 40.9 14.8 43.4 10.7 38.6 38.5 33.0 29.8 33.0 Qwen2.5-VL-72B [3] 25.1 29.3 54.5 38.8 38.2 37.0 34.0 28.9 37.0 LLaVA-Video-7B [49] 48.5 14.0 47.8 24.2 43.5 42.4 34.0 30.6 35.6 LLaVA-Video-72B [49] 48.9 22.8 57.4 35.3 42.4 36.7 35.0 48.6 40.9 InternVL2-8B [28] 25.7 24.0 20.0 29.2 32.1 44.1 30.4 6.3 26.5 InternVL3-78B [56] 71.2 53.7 44.4 39.5 55.9 39.5 28.9 54.5 48.4

Spatial Reasoning Models SAT-LLaVA-Video [22] – – – 47.3 41.1 37.1 36.1 40.4 – SPAR [46] – – – – – – – – 41.1 Spatial-MLLM [38] 65.3 34.8 63.1 45.1 41.3 46.2 33.5 46.3 48.4 VG-LLM [50] 67.9 37.7 58.6 62.0 46.6 40.7 32.4 59.2 50.7 GeoSR (Ours) 68.3 38.7 57.4 62.3 48.7 44.4 35.6 59.5 51.9

### 4 Experiments

We perform extensive experiments on both static and dynamic spatial reasoning benchmarks to evaluate the effectiveness of GeoSR, proving that it can significantly improve spatial reasoning performance by making geometry matter. We further conduct ablation studies to validate the contribution of each component and analyze the hyperparameters and the computational cost.

#### 4.1 Static Spatial Reasoning

Benchmarks. We evaluate static spatial reasoning on VSI-Bench [41], which contains over 5k QA pairs from 288 real videos. It focuses on mostly rigid scenes where viewpoint and visibility change across frames, so correct answers often require stable geometric understanding rather than appearance cues. VSI-Bench includes numerical and multiple-choice questions. Numerical subtasks cover object counting (Obj. Cnt.), absolute distance estimation (Abs. Dist.), object size (Obj. Size), and room size (Room Size). Multiple-choice subtasks cover relative distance (Rel. Dist.), relative direction (Rel. Dir.), route planning (Route Plan), and appearance ordering (Appr. Order). Following the official protocol, we report accuracy for multiple-choice questions, and mean relative accuracy for numerical questions, which aggregates scores under several relative error tolerances.

Baselines and results. We compare GeoSR with three groups of baselines: (1) proprietary API models as strong reference [13,27], (2) representative opensource video models that are trained for general video understanding [3,28,49,56],

- (3) spatial reasoning models that are enhanced by spatial supervision and geometry priors [22,38,46,50]. Overall, as shown in Table 1, GeoSR improves over the

###### Table 2: Performance on dynamic spatial reasoning. Best , second-best , and third-best results are highlighted.

Subtask Types

Models

Avg. Absolute Relative Non.

Temp. Dist. Dir. Ori. Spd.

Dir. Pred. Proprietary Models (API)

Spd. Comp.

Dir. Pred.

Spd. Comp.

Dist. Dir. Ori. Spd.

- GPT-4o [13] 18.8 29.2 26.8 29.7 21.5 26.2 24.1 23.8 17.2 32.3 22.8 24.4 34.7 26.4

- GPT-5 [24] 21.1 41.5 48.7 34.5 33.3 34.7 17.2 44.3 41.9 21.2 25.0 30.9 26.7 30.8

- Gemini-2.5-Flash [6] 18.8 27.6 19.5 25.0 23.6 22.0 11.2 28.4 30.8 23.2 17.1 22.6 38.8 24.9

- Gemini-2.5-Pro [6] 20.0 44.6 53.6 27.3 38.7 30.5 23.2 32.9 43.2 17.1 28.5 27.9 34.3 31.7

General Video Understanding Models Qwen2.5-VL-7B [3] 18.8 15.3 14.6 42.8 29.0 19.4 31.8 19.3 11.1 22.2 19.2 20.2 30.1 23.5 Qwen2.5-VL-32B [3] 31.7 21.5 23.1 44.0 36.5 25.4 27.5 23.8 37.0 27.2 29.2 21.4 36.2 29.9 Qwen3-VL-8B-Instruct [2] 23.5 24.6 42.6 29.7 27.9 33.8 18.1 28.4 34.5 24.2 22.1 27.9 33.5 28.7 Qwen3-VL-30B-A3B-Instruct [2] 25.8 27.6 46.3 30.9 31.1 34.7 20.6 29.5 37.0 28.2 24.2 31.5 35.4 31.1 LLaVA-Video-7B [49] 22.3 16.9 25.6 33.3 45.1 24.5 24.1 15.9 17.2 19.1 24.2 21.4 33.9 25.9 VideoRefer [44] 23.5 18.4 25.6 33.5 45.4 27.1 25.0 16.1 18.5 20.2 26.4 22.6 34.7 26.9 InternVL3.5-8B [35] 23.5 27.6 28.0 34.5 24.7 27.9 22.4 17.0 19.7 28.2 30.0 14.2 30.1 25.4 InternVL3.5-38B [35] 25.8 27.8 29.2 34.2 24.7 28.5 26.7 16.3 23.4 29.2 32.1 15.4 31.3 26.7

Spatial Reasoning Models VLM-3R [7] 28.2 27.6 31.7 42.8 38.7 33.0 34.4 23.8 30.8 22.2 26.4 29.1 35.0 31.4 VG-LLM [50] 55.2 32.3 58.5 57.1 51.6 32.2 56.0 36.3 32.0 30.3 32.1 29.1 27.9 38.4 GSM [53] 87.0 73.8 84.1 73.8 72.0 35.5 75.8 76.1 77.7 60.6 37.1 35.1 46.4 58.9 GeoSR (Ours) 88.0 84.4 87.8 83.3 78.5 41.0 79.5 83.0 83.8 76.5 48.9 39.8 52.9 66.1

corresponding baselines under the same evaluation settings, which suggests that geometry tokens become more useful during reasoning rather than remaining auxiliary context.

#### 4.2 Dynamic Spatial Reasoning

Benchmarks. We evaluate dynamic spatial reasoning on DSR-Bench [53], which contains 1484 QA pairs from 575 in-the-wild videos with evolving dynamic spatial relations due to object and camera motion. Correct answers require spatiotemporal consistency rather than static cues. DSR-Bench is organized into absolute, relative, and non-template question types. Absolute types include distance (Dist.), direction (Dir.), orientation (Ori.), and speed (Spd.), and also include comparison and prediction variants such as speed comparison (Spd. Comp.) and direction prediction (Dir. Pred.). Relative types ask about relations defined with respect to another entity, another time point, or another reference frame, including similar types to the absolute ones under such references. Beyond these structured categories, Non-Temp. contains questions with more flexible phrasing that are not restricted to fixed templates. We follow the official evaluation protocol in [53] and report accuracy for each type and the overall average.

Baselines and results. Similar to the static spatial reasoning settings, Table 2 compares GeoSR with proprietary API models [6,13,24], general video understanding models [2,3,35,44,49] and spatial reasoning models [7,50,53]. Results in Table 2 show that DSR-Bench is challenging for general models while spatial

###### Table 3: Ablations on GeoSR components for static spatial reasoning.

Module Subtask Types

Setting

Avg. Geo. Mask.

Geo. Fus.

Ori. Fus.

Obj. Cnt.

Abs. Dist.

Obj. Size

Room Size

Rel. Dist.

Rel. Dir.

Route Plan

Appr. Order

- (a) ✓ ✓ 68.3 38.7 57.4 62.3 48.7 44.4 35.6 59.5 51.9
- (b) ✓ ✓ 67.7 38.5 57.7 61.6 46.5 40.9 34.0 52.9 50.0
- (c) ✓ 68.1 38.8 57.7 60.5 46.8 42.1 26.9 55.7 49.6
- (d) ✓ 68.2 38.0 58.7 57.6 45.6 42.5 34.5 61.8 50.9
- (e) ✓ 67.2 37.8 58.0 60.9 44.9 41.8 34.0 56.6 50.2
- (f) 68.6 36.0 57.9 59.7 45.8 38.9 30.4 60.2 49.8

###### Table 4: Ablations on GeoSR components for dynamic spatial reasoning.

Module Subtask Types

Setting

Avg. Geo. Mask.

Abs. Spd. Comp.

Abs. Dir. Pred.

Rel. Spd. Comp.

Rel. Dir. Pred.

Geo. Fus.

Ori. Fus.

Abs. Dist.

Abs. Dir.

Abs. Ori.

Abs. Spd.

Rel. Dist.

Rel. Dir.

Rel. Ori.

Rel. Spd.

Non. Temp.

- (a) ✓ ✓ 88.0 84.4 87.8 83.3 78.5 41.0 79.5 83.0 83.8 76.5 48.9 39.8 52.9 66.1
- (b) ✓ ✓ 86.7 81.3 87.8 78.6 78.5 41.9 78.6 84.1 83.8 75.5 43.9 34.9 53.8 64.7
- (c) ✓ 86.7 67.2 80.5 76.2 67.7 40.2 77.7 75.0 71.3 55.1 36.0 32.5 48.7 58.1
- (d) ✓ 88.0 84.4 89.0 77.4 78.5 41.9 77.7 87.5 82.5 70.4 41.7 38.6 54.2 64.9
- (e) ✓ 88.0 78.1 86.6 75.0 76.3 40.2 75.9 79.5 81.3 69.4 41.7 35.5 53.4 62.8
- (f) 86.7 79.7 85.3 75.0 79.6 41.0 75.0 83.0 82.5 74.5 43.9 39.8 51.7 64.0

reasoning models perform significantly better, which suggests that geometry cues are particularly helpful for dynamic spatial reasoning. Furthermore, GeoSR rivals the best-performing model and achieves the highest score on each subtask type, which demonstrates that the proposed design highly benefits dynamic spatial reasoning by unleashing geometry cues and adaptively token fusing.

#### 4.3 Ablation Studies

We conduct ablations on both static (Table 3) and dynamic (Table 4) benchmarks to assess the effects of Geometry-Unleashing Masking (Geo. Mask.), Geometry-Guided Fusion (Geo. Fus.), and the original naive fusion used in prior geometry-aware frameworks (Ori. Fus.). We take the full GeoSR model (a) as the reference. Replacing Geo. Fus. with the Ori. Fus. (b) reduces the performance on both benchmarks, indicating that naive fusion is less effective at turning geometry tokens into actionable evidence. Removing Geo. Fus. and Ori. Fus. together while keeping Geo. Mask. (c) further degrades the results, suggesting that masking alone cannot fully exploit geometry without a well-designed mechanism to route geometry cues into the VLM backbone. Conversely, removing Geo. Mask. but keeping Geo. Fus. (d) also leads to a noticeable drop compared to (a), showing that the model can still underutilize geometry without shortcut suppression. Retaining only Ori. Fus. (e) yields an inferior variant to (d), again

Table 5: Hyperparameter analysis. The best configuration is bolded.

Table 6: Computational overhead. Model size includes the VLM backbone and all embedders.

γ 0.8 0.6 0.4 Avg. 66.1 65.9 65.0

β 0.7 0.5 0.3 Avg. 64.5 66.1 64.9

Models Time Size Mem.

Qwen2.5-VL-7B [3] 0.37s 8.76B 18.04GB w/ Geo. (Baseline) 0.40s 9.16B 18.81GB GeoSR (Ours) 0.41s 9.23B 18.95GB

highlighting the benefit of adaptive routing over the original fusion. Finally, we examine the no-geometry model (f). On the static benchmark, the performance drop from (e) to (f) is relatively moderate, suggesting that for static scenes, the gains from directly injecting geometry can be limited. On the dynamic benchmark, (f) even outperforms (e), indicating that simple geometry token injection may yield negative effects when geometry cues are fused without proper control, consistent with our motivation in the introduction. Overall, Geo. Mask. makes geometry effective by weakening appearance shortcuts during training, while Geo. Fus. makes geometry usage reasonable via adaptive routing. The improvements are more pronounced on the dynamic benchmark, where motion and occlusion make appearance cues less stable and thus benefit more from enforced geometry reliance and fine-grained fusion.

#### 4.4 Analysis

Hyperparameter settings. We validate the masking hyperparameter settings of γ in Eq. (10) and β in Eq. (6) on DSR-Bench [53]. Table 5 shows that small γ and β leave appearance shortcuts largely intact and bring limited gains, while overly large β removes too much context and can hurt stability. We find that γ = 0.8 and β = 0.5 strike a good balance.

Computational overhead. We report runtime, model size, and peak memory for dynamic reasoning on DSR-Bench [53] in Table 6, testing on a single H200 GPU. The extra parameters of GeoSR mainly come from the geometry projection and the gating module, so the parameter increase is minor relative to the geometry-aware baseline. GeoSR also introduces a negligible runtime increase, showing that the proposed design is efficient and does not require heavy computation to unleash geometry cues and boost the spatial reasoning performance.

### 5 Conclusion

In this paper, we study a failure mode in existing geometry-aware VLMs for spatial reasoning: under the common recipe of naive token fusion and standard fine-tuning, injected geometry tokens can be largely underutilized, bringing limited gains and even harming performance. This suggests that geometry priors, while informative, are not automatically treated as actionable evidence by the VLM backbone. To address this issue, we propose GeoSR, a new framework

that makes geometry matter by encouraging effective and reasonable geometry usage. Geometry-Unleashing Masking weakens vision-driven shortcuts during training and forces the model to consult geometry tokens, while GeometryGuided Fusion adaptively routes geometry evidence via a fine-grained gate so that geometry contributes more when it is informative and necessary. Extensive experiments on both static and dynamic spatial reasoning benchmarks validate the effectiveness of these designs, with consistent improvements over prior baselines, especially in dynamic settings.

### References

- 1. Ba, J.L., Kiros, J.R., Hinton, G.E.: Layer normalization. In: Proceedings of the Advances in Neural Information Processing Systems. pp. 1–14 (2016) 9
- 2. Bai, S., Cai, Y., Chen, R., Chen, K., Chen, X., Cheng, Z., Deng, L., Ding, W., Gao, C., Ge, C., et al.: Qwen3-vl technical report. arXiv preprint arXiv:2511.21631

(2025) 4, 12

- 3. Bai, S., Chen, K., Liu, X., Wang, J., Ge, W., Song, S., Dang, K., Wang, P., Wang, S., Tang, J., Zhong, H., Zhu, Y., Yang, M., Li, Z., Wan, J., Wang, P., Ding, W., Fu, Z., Xu, Y., Ye, J., Zhang, X., Xie, T., Cheng, Z., Zhang, H., Yang, Z., Xu, H., Lin, J.: Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923 (2025) 4, 10, 11, 12, 14
- 4. Chen, Y., Huang, W., Shi, B., Hu, Q., Ye, H., Zhu, L., Liu, Z., Molchanov, P., Kautz, J., Qi, X., et al.: Scaling rl to long videos. In: Proceedings of the Advances in Neural Information Processing Systems. pp. 1–29 (2025) 4
- 5. Cheng, A.C., Fu, Y., Chen, Y., Liu, Z., Li, X., Radhakrishnan, S., Han, S., Lu, Y., Kautz, J., Molchanov, P., et al.: 3d aware region prompted vision language model. In: Proceedings of the International Conference on Learning Representations. pp. 1–23 (2026) 2
- 6. Comanici, G., Bieber, E., Schaekermann, M., Pasupat, I., Sachdeva, N., Dhillon,

I., Blistein, M., Ram, O., Zhang, D., Rosen, E., et al.: Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261 (2025) 4, 12

- 7. Fan, Z., Zhang, J., Li, R., Zhang, J., Chen, R., Hu, H., Wang, K., Qu, H., Wang, D., Yan, Z., et al.: Vlm-3r: Vision-language models augmented with instructionaligned 3d reconstruction. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. pp. 1–21 (2026) 2, 4, 7, 8, 12
- 8. Feng, S., Fang, G., Ma, X., Wang, X.: Efficient reasoning models: A survey. Transactions on Machine Learning Research pp. 1–34 (2025) 1
- 9. Feng, S., Wang, S., Ouyang, S., Kong, L., Song, Z., Zhu, J., Wang, H., Wang, X.: Can mllms guide me home? a benchmark study on fine-grained visual reasoning from transit maps. arXiv preprint arXiv:2505.18675 (2025) 2
- 10. Gao, M., Liu, J., Li, M., Xie, J., Liu, Q., Zhao, K., Chen, X., Xiong, H.: Tc-llava: Rethinking the transfer of llava from image to video understanding with temporal considerations. In: Proceedings of the AAAI Conference on Artificial Intelligence. pp. 3086–3094 (2025) 7
- 11. He, K., Chen, X., Xie, S., Li, Y., Dollár, P., Girshick, R.: Masked autoencoders are scalable vision learners. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. pp. 16000–16009 (2022) 8

- 12. Hong, Y., Zhen, H., Chen, P., Zheng, S., Du, Y., Chen, Z., Gan, C.: 3d-llm: Injecting the 3d world into large language models. In: Proceedings of the Advances in Neural Information Processing Systems. vol. 36, pp. 20482–20494 (2023) 2
- 13. Hurst, A., Lerer, A., Goucher, A.P., Perelman, A., Ramesh, A., Clark, A., Ostrow, A., Welihinda, A., Hayes, A., Radford, A., et al.: Gpt-4o system card. arXiv preprint arXiv:2410.21276 (2024) 4, 11, 12
- 14. Kingma, D.P., Ba, J.: Adam: A method for stochastic optimization. In: Proceedings of the International Conference on Learning Representation. pp. 1–15 (2015) 10
- 15. Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S., Berg, A.C., Lo, W.Y., et al.: Segment anything. In: Proceedings of the IEEE International Conference on Computer Vision. pp. 4015–4026 (2023) 8
- 16. Li, Q., Wang, X.: Sponge tool attack: Stealthy denial-of-efficiency against toolaugmented agentic reasoning. arXiv preprint arXiv:2601.17566 (2026) 1
- 17. Li, Y., Zhang, Y., Lin, T., Liu, X., Cai, W., Liu, Z., Zhao, B.: Sti-bench: Are mllms ready for precise spatial-temporal world understanding? In: Proceedings of the IEEE International Conference on Computer Vision. pp. 5622–5632 (2025) 2, 4
- 18. Ma, W., Chen, H., Zhang, G., Chou, Y.C., Chen, J., de Melo, C., Yuille, A.:

- 3dsrbench: A comprehensive 3d spatial reasoning benchmark. In: Proceedings of the IEEE International Conference on Computer Vision. pp. 6924–6934 (2025) 2,
- 4

- 19. Maaz, M., Raasheed, H., Khan, S., Khan, F.: Video-chatgpt: Towards detailed video understanding via large vision and language models. In: Proceedings of the Annual Meeting of the Association for Computational Linguistics. pp. 12585–12602

(2024) 4

- 20. Qi, Z., Zhang, Z., Fang, Y., Wang, J., Zhao, H.: Gpt4scene: Understand 3d scenes from videos with vision-language models. arXiv preprint arXiv:2501.01428 (2025) 2
- 21. Rasley, J., Rajbhandari, S., Ruwase, O., He, Y.: Deepspeed: System optimizations enable training deep learning models with over 100 billion parameters. In: Proceedings of the International Conference on Knowledge Discovery and Data Mining. pp. 3505–3506 (2020) 10
- 22. Ray, A., Duan, J., Brown II, E.L., Tan, R., Bashkirova, D., Hendrix, R., Ehsani, K., Kembhavi, A., Plummer, B.A., Krishna, R., Zeng, K.H., Saenko, K.: Sat: Dynamic spatial aptitude training for multimodal language models. In: Proceedings of the Conference on Language Modeling. pp. 1–27 (2025) 4, 11
- 23. Schonberger, J.L., Frahm, J.M.: Structure-from-motion revisited. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. pp. 4104– 4113 (2016) 5
- 24. Singh, A., Fry, A., Perelman, A., Tart, A., Ganesh, A., El-Kishky, A., McLaughlin, A., Low, A., Ostrow, A., Ananthram, A., et al.: Openai gpt-5 system card. arXiv preprint arXiv:2601.03267 (2025) 4, 12
- 25. Su, J., Ahmed, M., Lu, Y., Pan, S., Bo, W., Liu, Y.: Roformer: Enhanced transformer with rotary position embedding. Neurocomputing 568, 127063 (2024) 7
- 26. Tang, Y., Bi, J., Xu, S., Song, L., Liang, S., Wang, T., Zhang, D., An, J., Lin, J., Zhu, R., et al.: Video understanding with large language models: A survey. IEEE Transactions on Circuits and Systems for Video Technology 36(2), 1355– 1376 (2026) 1
- 27. Team, G., Georgiev, P., Lei, V.I., Burnell, R., Bai, L., Gulati, A., Tanzer, G., Vincent, D., Pan, Z., Wang, S., et al.: Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530 (2024) 11

- 28. Team, O.: Internvl2: Better than the best—expanding performance boundaries of open-source multimodal models with the progressive scaling strategy (2024) 11
- 29. Tong, P., Brown, E., Wu, P., Woo, S., Iyer, A.J.V., Akula, S.C., Yang, S., Yang, J., Middepogu, M., Wang, Z., et al.: Cambrian-1: A fully open, vision-centric exploration of multimodal llms. In: Proceedings of the Advances in Neural Information Processing Systems. vol. 37, pp. 87310–87356 (2024) 2
- 30. Tong, Z., Song, Y., Wang, J., Wang, L.: Videomae: Masked autoencoders are dataefficient learners for self-supervised video pre-training. In: Proceedings of the Advances in Neural Information Processing Systems. vol. 35, pp. 10078–10093 (2022) 8
- 31. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, Ł., Polosukhin, I.: Attention is all you need. In: Proceedings of the Advances in Neural Information Processing Systems. vol. 30, pp. 1–11 (2017) 6
- 32. Wang, J., Chen, M., Karaev, N., Vedaldi, A., Rupprecht, C., Novotny, D.: Vggt: Visual geometry grounded transformer. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. pp. 5294–5306 (2025) 2, 4, 5, 10
- 33. Wang, P., Bai, S., Tan, S., Wang, S., Fan, Z., Bai, J., Chen, K., Liu, X., Wang, J., Ge, W., et al.: Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191 (2024) 7
- 34. Wang, Q., Zhang, Y., Holynski, A., Efros, A.A., Kanazawa, A.: Continuous 3d perception model with persistent state. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. pp. 10510–10522 (2025) 2, 4, 5
- 35. Wang, W., Gao, Z., Gu, L., Pu, H., Cui, L., Wei, X., Liu, Z., Jing, L., Ye, S., Shao, J., et al.: Internvl3.5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. arXiv preprint arXiv:2508.18265 (2025) 4, 12
- 36. Wang, Y., Zhou, J., Zhu, H., Chang, W., Zhou, Y., Li, Z., Chen, J., Pang, J., Shen, C., He, T.: π3: Permutation-equivariant visual geometry learning. In: Proceedings of the International Conference on Learning Representations. pp. 1–14 (2026) 2, 4, 5, 10
- 37. Weinzaepfel, P., Lucas, T., Leroy, V., Cabon, Y., Arora, V., Brégier, R., Csurka, G., Antsfeld, L., Chidlovskii, B., Revaud, J.: Croco v2: Improved cross-view completion pre-training for stereo matching and optical flow. In: Proceedings of the IEEE International Conference on Computer Vision. pp. 17969–17980 (2023) 8
- 38. Wu, D., Liu, F., Hung, Y.H., Duan, Y.: Spatial-mllm: Boosting mllm capabilities in visual-based spatial intelligence. In: Proceedings of the Advances in Neural Information Processing Systems. pp. 1–21 (2025) 2, 4, 7, 8, 9, 11, 19
- 39. Xu, Q., Chen, S., Chen, G., Wang, Y., Zhang, Y.: Chatbev: A visual language model that understands bev maps. arXiv preprint arXiv:2503.13938 (2025) 2
- 40. Xu, R., Wang, X., Wang, T., Chen, Y., Pang, J., Lin, D.: Pointllm: Empowering large language models to understand point clouds. In: Proceedings of the European Conference on Computer Vision. pp. 131–147 (2024) 2
- 41. Yang, J., Yang, S., Gupta, A.W., Han, R., Fei-Fei, L., Xie, S.: Thinking in space: How multimodal large language models see, remember, and recall spaces. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. pp. 10632–10643 (2025) 2, 4, 11, 19, 21
- 42. Yin, S., Fu, C., Zhao, S., Li, K., Sun, X., Xu, T., Chen, E.: A survey on multimodal large language models. National Science Review 11(12), nwae403 (2024) 1
- 43. Yu, R., Li, Q., Wang, X.: Discrete diffusion in large language and multimodal models: A survey. arXiv preprint arXiv:2506.13759 (2025) 1

- 44. Yuan, Y., Zhang, H., Li, W., Cheng, Z., Zhang, B., Li, L., Li, X., Zhao, D., Zhang, W., Zhuang, Y., et al.: Videorefer suite: Advancing spatial-temporal object understanding with video llm. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. pp. 18970–18980 (2025) 4, 12
- 45. Zhang, H., Li, X., Bing, L.: Video-llama: An instruction-tuned audio-visual language model for video understanding. In: Proceedings of the Conference on Empirical Methods in Natural Language Processing. pp. 543–553 (2023) 4
- 46. Zhang, J., Chen, Y., Zhou, Y., Xu, Y., Huang, Z., Mei, J., Chen, J., Yuan, Y.J., Cai, X., Huang, G., et al.: From flatland to space: Teaching vision-language models to perceive and reason in 3d. In: Proceedings of the Advances in Neural Information Processing Systems. pp. 1–33 (2025) 4, 10, 11
- 47. Zhang, J., Huang, J., Jin, S., Lu, S.: Vision-language models for vision tasks: A survey. IEEE Transactions on Pattern Analysis and Machine Intelligence 46(8), 5625–5644 (2024) 1
- 48. Zhang, S., Li, Z., Zhang, K., Lu, Y., Deng, Y., Tang, L., Jiang, X., Ma, J.: Deep learning reforms image matching: A survey and outlook. arXiv preprint arXiv:2506.04619 (2025) 5
- 49. Zhang, Y., Wu, J., Li, W., Li, B., Ma, Z., Liu, Z., Li, C.: Llava-video: Video instruction tuning with synthetic data. Transactions on Machine Learning Research pp. 1–24 (2025) 4, 10, 11, 12
- 50. Zheng, D., Huang, S., Li, Y., Wang, L.: Learning from videos for 3d world: Enhancing mllms with 3d vision geometry priors. In: Proceedings of the Advances in Neural Information Processing Systems. pp. 1–20 (2025) 2, 4, 5, 7, 8, 9, 10, 11, 12, 19
- 51. Zheng, D., Huang, S., Wang, L.: Video-3d llm: Learning position-aware video representation for 3d scene understanding. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. pp. 8995–9006 (2025) 2
- 52. Zhou, H., Lee, G.H.: Llava-4d: Embedding spatiotemporal prompt into lmms for 4d scene understanding. arXiv preprint arXiv:2505.12253 (2025) 2, 4
- 53. Zhou, S., Chen, Y., Ge, Y., Huang, W., Lin, J., Shan, Y., Qi, X.: Learning to reason in 4d: Dynamic spatial understanding for vision language models. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. pp. 1–18

(2026) 2, 4, 5, 7, 8, 9, 10, 12, 14, 19, 22

- 54. Zhou, S., Vilesov, A., He, X., Wan, Z., Zhang, S., Nagachandra, A., Chang, D., Chen, D., Wang, X.E., Kadambi, A.: Vlm4d: Towards spatiotemporal awareness in vision language models. In: Proceedings of the IEEE International Conference on Computer Vision. pp. 8600–8612 (2025) 2, 4
- 55. Zhu, C., Wang, T., Zhang, W., Pang, J., Liu, X.: Llava-3d: A simple yet effective pathway to empowering lmms with 3d capabilities. In: Proceedings of the IEEE International Conference on Computer Vision. pp. 4295–4305 (2025) 2
- 56. Zhu, J., Wang, W., Chen, Z., Liu, Z., Ye, S., Gu, L., Tian, H., Duan, Y., Su, W., Shao, J., et al.: Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479 (2025) 11
- 57. Zhu, W., Li, B., Zheng, C., Mai, J., Chen, J., Jiang, L., Hamdi, A., Martinez, S.R., Lin, C.W., Elhoseiny, M., et al.: 4d-bench: Benchmarking multi-modal large language models for 4d object understanding. In: Proceedings of the IEEE International Conference on Computer Vision. pp. 21129–21143 (2025) 2, 4

Appendix

- A Qualitative Results

We present qualitative results of GeoSR on both static and dynamic spatial reasoning benchmarks [41,53]. The examples are shown in Figures 4 and 5, together with comparisons to strong baselines, including VG-LLM [50] and GSM [53]. Each example is annotated with its video ID, and the corresponding videos are provided in the supplementary video file.

- B More Analysis

In the main paper, following prior meticulous architecture in [38,50,53], the baseline for dynamic spatial reasoning adopts a QFormer-style architecture, whereas the static one does not. This design choice is reasonable because dynamic spatial reasoning requires aggregating motion patterns and temporal relations across frames, for which QFormer-style query tokens provide an effective mechanism to summarize global video dynamics. To examine whether such a design is also beneficial for static spatial reasoning, we additionally implement a QFormer-style variant of our GeoSR. As shown in Table 7, introducing QFormer does not improve performance. This suggests that, unlike dynamic scenarios, static spatial reasoning relies less on explicit global temporal aggregation, and its main challenge lies more in grounding spatial judgments on geometric cues than in modeling long-range dynamics.

Table 7: Performance of GeoSR and GeoSR with QFormer on static spatial reasoning.

Models

Subtask Types

Avg. Numerical Questions Multiple-Choice Questions

Obj. Cnt. Abs. Dist. Obj. Size Room Size Rel. Dist. Rel. Dir. Route Plan Appr. Order

GeoSR 68.3 38.7 57.4 62.3 48.7 44.4 35.6 59.5 51.9 +QFormer 68.6 38.8 57.8 60.6 49.0 43.5 35.6 60.2 51.8

- C Limitations and Future Work

GeoSR focuses on model-side improvements that better exploit geometric cues for spatial reasoning on existing benchmarks [41, 53]. Nevertheless, the quality of current datasets may also limit further progress. Since both training and evaluation data are constructed through automatic or semi-automatic pipelines, some question formulations can be ambiguous from a geometric perspective, and some annotations may not perfectly align with the underlying visual evidence. Figures 6 and 7 provide two representative examples. Such issues may

##### affect the performance of all methods. An important future direction is therefore to improve dataset quality, especially in terms of geometry-aware question construction and annotation consistency, alongside continued advances in model design.

###### ID: arkitscenes/42899696

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

Q: If I am standing by the bed and facing the sofa, is the tv to my front-left, front-right, back-left, or back-right?

- A. back-left
- B. front-left
- C. back-right
- D. front-right

GT: D VG-LLM: C

Ours: D Q: If I am standing by the bed and facing the stool, is the tv to my front-left, front-right, back-left, or back-right?

- A. front-left
- B. back-right
- C. back-left
- D. front-right

GT: D VG-LLM: B

Ours: D

###### ID: arkitscenes/47331972

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

Q: Measuring from the closest point of each object, which of these objects (table, fireplace, chair, sofa) is the closest to the dishwasher?

- A. table
- B. fireplace
- C. chair
- D. sofa

GT: C VG-LLM: A

Ours: C Q: Measuring from the closest point of each object, which of these objects (sofa, table, refrigerator, dishwasher) is the closest to the tv?

- A. sofa
- B. table
- C. refrigerator
- D. dishwasher

GT: A VG-LLM: B

Ours: A

ID: scannet/scene0316_00

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

Q: You are a robot beginning at the door facing the brown single-seat sofa. You want to navigate to the chair next to water fountain . You will perform the following actions (Note: for each [please fill in], choose either 'turn back,' 'turn left,' or 'turn right.'): 1. Go forward until the brown single-seat sofa. 2. [please fill in] 3. Go forward until passing by the brown two-seats sofa. 4. [please fill in] 5. Go forward until the chair. You have reached the final destination.

- A. Turn Left, Turn Left
- B. Turn Back, Turn Right
- C. Turn Left, Turn Right
- D. Turn Right, Turn Left

GT: A VG-LLM: C

Ours: A

ID: arkitscenes/41159572

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

Q: How many sofa(s) are in this room? GT: 4, VG-LLM: 2, Ours: 3 Q: How many table(s) are in this room? GT: 2, VG-LLM: 3, Ours: 2

ID: scannetpp/3f15a9266d

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

Q: Measuring from the closest point of each object, what is the distance between the keyboard and the telephone (in meters)?

GT: 0.7, VG-LLM: 1.0, Ours: 0.4 Q: Measuring from the closest point of each object, what is the distance between the computer mouse and the exhaust fan (in meters)?

GT: 2.1, VG-LLM: 3.1, Ours: 3.0 Q: Measuring from the closest point of each object, what is the distance between the telephone and the exhaust fan (in meters)?

- GT: 2.1, VG-LLM: 3.1, Ours: 2.3

Q: Measuring from the closest point of each object, what is the distance between the computer mouse and the door (in meters)?

- GT: 3.2, VG-LLM: 2.7, Ours: 3.1

###### Fig. 4: Visualization of the static spatial reasoning results on VSI-Bench [41].

ID: vwhHZeKT09I_12

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

###### t=4.8s t=9.6s

Q: Between 4.8s and 9.6s, from the perspective of man with bounding box coordinates [230,55,420,358] at 9.6s, how does the speed of man with initial bounding box coordinates [172,97,363,358] change?

- A. (1) Becomes smaller. (2) Becomes larger. (3) Becomes smaller.
- B. (1) Becomes smaller. (2) Keep nearly constant and then becomes larger. (3) Becomes larger.
- C. (1) Keep nearly constant and then becomes larger. (2) Becomes larger.
- D. (1) Becomes smaller. (2) Keep nearly constant and then become smaller.

- GT: B GSM: C Ours: B

GT: A

- GSM: B

- Ours: A

GT: B GSM: D

- Ours: B

GT: A

- GSM: C

- Ours: A

GT: B GSM: C

- Ours: B

- GT: B GSM: A Ours: B

ID: OSFBlKqGx9o_11

| | |
|---|---|
| | |
| | |

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

###### t=15.5s t=43.5s

Q: From the perspective of camera at 40.5s, between 15.5s and 43.5s, which object exhibits a more significant apparent vertical displacement relative to its initial position over the observed time period?

- A. The object with initial bounding box [287,69,485,358]
- B. The object with initial bounding box [481,0,537,279]
- C. Both objects exhibit similar apparent vertical displacement
- D. Neither object exhibits significant apparent vertical displacement

ID: sirSE8kj-ds_3

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

| |
|---|
| |

| |
|---|
| |

###### t=16.8s t=19.7s

Q: From the perspective of camera at 21.1s, between 16.8s and 19.7s, which object is positioned directly above the horse with initial bounding box coordinates [92,147,130,249] throughout the observed period?

- A. the horse with initial bounding box coordinates [239,146,295,279]
- B. the jockey with initial bounding box coordinates [91,136,132,193]
- C. the horse with initial bounding box coordinates [440,181,483,288]
- D. the jockey with initial bounding box coordinates [243,137,299,222]

ID: 8YdGtkqIDKc_29

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

###### t=46.0s t=47.5s

Q: From the perspective of camera at 5.5s, between 46.0s and 47.5s, which dancer moves predominantly leftward relative to the others over time?

- A. The dancer with initial bounding box coordinates [327,48,426,266]
- B. The dancer with initial bounding box coordinates [56,50,269,353]
- C. The dancer with initial bounding box coordinates [127,90,250,297]
- D. No dancer moves predominantly leftward relative to the others

ID: I30-f0VI218_16

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

| | |
|---|---|
| | |
| | |

###### t=9.5s t=11.0s

Q: Following the perspective of camera, between 9.5s and 11.0s, over the time from 9.5 to 11.0 seconds, which person is moving closer in 3d space to the paddleboard with initial bounding box [98,149,223,223]?

- A. The person with initial bounding box [430,112,472,228]
- B. The person with initial bounding box [168,111,223,252]
- C. The person with initial bounding box [238,188,287,258]
- D. No person is moving closer to that paddleboard

ID: IPY6Y0lSyMk_31

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

t=23.5s t=25.9s

Q: Following the perspective of camera, between 23.5s and 25.9s, based on their movement, is the car with initial bounding box [155,232,192,264] getting closer to or farther from the bus with initial bounding box [207,217,245,261]?

- A. Closer to
- B. Farther from
- C. Maintaining the same distance
- D. Moving alongside without changing distance

###### Fig. 5: Visualization of the dynamic spatial reasoning results on DSR-Bench [53].

ID: uH97lwo0T10_21

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

t=13.0s t=19.0s

Q: Between 13.0s and 19.0s, following the perspective of camera, compare the speed between elephant with initial bounding box coordinates [419,113,453,161] and human with initial bounding box coordinates [342,124,379,211].

- A. (1) The latter is faster. (2) The former is faster. (3) Nearly the same.
- B. (1) The latter is faster. (2) The former is faster.
- C. (1) The former is always faster.
- D. (1) The former is faster. (2) Nearly the same. (3) The former is faster.

GT: C

- Fig. 6: Example illustrating ambiguity in the benchmark. The question asks for a speed comparison between the elephant and the person, but the visual evidence is ambiguous from a geometric perspective. The ground-truth answer is “(1) The latter is faster. (2) The former is faster.” However, in the video segment from 13.0s to 19.0s, it is difficult to determine which one is faster, since the person and the elephant appear to move at similar speeds.

Q: Following the perspective of character with initial bounding box coordinates [102,28,364,319], between 8.5s and 11.0s, between 8.5s and 11.0s, is the table with initial bounding box [234,240,433,321] always located between the character with initial bounding box [102,28,364,319] and the character with initial bounding box [360,81,620,318]?

- A. Yes, the table consistently separates the two characters throughout the interval.
- B. No, during some moments, the table is not between them and is located behind the character at [102,28,364,319].
- C. No, the two characters move to the same side relative to the table at certain times.
- D. The 2D information is sufficient to confirm that the table remains between them at all times.

t=8.5s t=11.0s

ID: r3guADzIOfE_103

GT: B

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

- Fig. 7: Example illustrating ambiguity in the benchmark. The question asks about the relative spatial relation among the table and the two cartoon characters, and the ground-truth answer states that “No, during some moments, the table is not between them and is located behind the character at [102,28,364,319].” However, in the video segment from 8.5s to 11.0s, this relation remains difficult to judge. Although the purple character moves toward the table, the character is partially occluded by it, making it unclear whether the table is indeed behind the character.

