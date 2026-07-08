# arXiv:2601.16296v2[cs.CV]23Mar2026

## Memory-V2V: Memory-Augmented Video-to-Video Diffusion for Consistent Multi-Turn Editing

Dohun Lee1,2,∗, Chun-Hao Paul Huang1, Xuelin Chen1, Jong Chul Ye2, Duygu Ceylan1,†, and Hyeonho Jeong1,†

1 Adobe Research 2 KAIST AI ∗ Adobe Internship † Project Lead Project page: https://dohunlee1.github.io/MemoryV2V/

[Figure 1]

Fig. 1: Memory-V2V enables iterative video editing with long-term memory, producing results that remain consistent with edits from previous turns. (A) and (B) show results for video novel view synthesis and text-guided long video editing, respectively. Colored boxes in (A) highlight novel-view regions that are expected to remain consistent across editing iterations. Each iteration performs independent denoising. Additional video results are provided on the supplementary project page.

Abstract. Video-to-video diffusion models achieve impressive singleturn editing performance, but practical editing workflows are inherently iterative. When edits are applied sequentially, existing models treat each turn independently, often causing previously generated regions to drift or be overwritten. We identify this failure mode as the problem of cross-turn consistency in multi-turn video editing. We introduce Memory-V2V, a

memory-augmented framework that treats prior edits as structured constraints for subsequent generations. Memory-V2V maintains an external memory of previous outputs, retrieves task-relevant edits, and integrates them through relevance-aware tokenization and adaptive compression. These technical ingredients enable scalable conditioning without linear growth in computation. We demonstrate Memory-V2V on iterative video novel view synthesis and text-guided long video editing. Memory-V2V substantially enhances cross-turn consistency while maintaining visual quality, outperforming strong baselines with modest overhead.

Keywords: Video editing · Multi-turn editing · Video diffusion models

### 1 Introduction

Videos are rapidly becoming a dominant medium of modern communication and expression, spanning domains from entertainment to robotics simulation. With the advent of large-scale generative models, users increasingly expect video editing tools that enable attribute-level control — allowing modification of subjects [1,26,28], motion [1,23,26], or viewpoint [2,42,57]. However, in practice, video editing workflows are inherently iterative: users progressively refine outputs over multiple interactions, a setting we term multi-turn video editing.

While recent video editing frameworks demonstrate impressive results under single-pass interaction, they often fail to maintain cross-turn consistency across sequential edits. For example, as shown in Fig.1(A), a state-of-the-art video novel view synthesis model, ReCamMaster [2], successfully re-renders the input video from multiple target viewpoints, yet the novel-view regions across iterations remain inconsistent. Similarly, in Fig.1(B), when segments of a long video are iteratively edited using LucyEdit [1], each segment adheres to the local editing prompt, but the global appearance gradually drifts. For instance, an apple edited into an orange appears as visually different oranges across segments.

In this work, we formulate consistent multi-turn video editing as a distinct problem setting and introduce Memory-V2V, a memory-augmented framework tailored to this scenario. Our central design principle is that multi-turn editing requires the memory to function as a constraint mechanism. This is fundamentally different from existing memory-augmented long video generators that assume temporal continuity. Hence, instead of treating previous outputs as additional temporal context, we regard them as a structured external memory that selectively enforces consistency across editing iterations.

A naive solution for enforcing multi-turn consistency is to condition on all previous edits. However, this leads to prohibitive computational growth and introduces redundant or irrelevant context. Instead, we introduce Memory-V2V guided by two design principles: selective retrieval and relevance-aware capacity allocation. First, rather than conditioning on all previous edits or temporal adjacency retrieval used in autoregressive generation, we retrieve only the most relevant prior videos for the current turn. Crucially, retrieval is task-aware and non-temporal. For video novel view synthesis, we introduce a geometric Field-ofView (FOV) retrieval mechanism that ranks prior edits based on camera overlap.

For text-guided long video editing, we retrieve segments according to semantic similarity of their corresponding source videos. Second, we allocate computational capacity proportionally to memory relevance. Retrieved videos are dynamically tokenized with varying spatial resolutions, preserving fine-grained detail for highly relevant edits while aggressively compressing less relevant ones. Furthermore, we introduce adaptive token merging based on attention responsiveness, reducing redundant computation while maintaining essential visual cues. Together, these components enable scalable conditioning on heterogeneous memory sources without linear growth in cost.

We evaluate Memory-V2V on two representative video-to-video editing tasks: video novel view synthesis and text-guided long video editing. Memory-V2V substantially improves cross-iteration consistency in both tasks while also outperforming state-of-the-art baselines in single-turn editing performance. Our contributions can be summarized as follows:

- – We formulate multi-turn video editing as a distinct problem setting, highlighting the challenge of preserving consistency across independently denoised edits.
- – We propose Memory-V2V, a memory augmented framework for iterative editing, where prior edits are treated as structured constraints rather than temporal continuations.
- – We introduce task-aware retrieval mechanisms, including FOV-based geometric retrieval and semantic segment retrieval, that select relevant prior edits without incurring linear context growth. We develop dynamic tokenization and adaptive token merging enabling scalable multi-turn conditioning with reduced computational overhead.
- – We demonstrate substantial improvements in cross-turn consistency on iterative novel view synthesis and long video editing. To enable training for long video editing, we introduce a dataset augmentation pipeline that temporally extends only target videos from short source–target pairs, eliminating the need for long paired video-to-video training data.

### 2 Related Work

Video novel view synthesis. Given a monocular video of a dynamic scene, video novel view synthesis aims to generate plausible videos captured from unseen camera trajectories, while preserving the underlying 3D structure and temporal dynamics [45]. Due to the scarcity of large-scale paired multi-view video datasets, prior works have focused on test-time optimization or scenespecific overfitting of existing monocular video generators [22,34,60]. Recently, approaches such as ReCamMaster [2] leverage large-scale synthetic 4D datasets rendered from simulation engines [12,17] to finetune text-to-video diffusion models into multi-view video-to-video models. As shown in Fig. 1(A), when such models are applied iteratively, the generated outputs fail to maintain consistency for regions that were not visible in the input video. In contrast, as demonstrated in Fig. 1(A), we incorporate an explicit visual memory into pre-trained models such as ReCamMaster [2] to ensure strong cross-iteration consistency.

Another line of work [22, 42, 47, 57] employs point-cloud renderings, which, though incomplete, approximate the desired camera trajectories as geometric proxies. Video diffusion models then use these renderings as spatial guidance to synthesize complete and geometrically consistent videos. While some works iteratively refine point clouds to improve static scene novel view synthesis [42,59], they fail to generalize to dynamic scenes with non-rigid motion.

Text-guided video editing. Early text-driven video editing approaches can be broadly divided into two categories [1]. The first relies on inference-time solutions, often using deterministic inversion with test-time optimization to align edits with text prompts [5, 16, 23, 49]. The second conditions video diffusion models on explicit visual cues such as depth maps or optical flow fields [9,10, 18, 24]. While effective in constrained settings, they often introduce temporal artifacts and struggle to generalize to flexible, open-domain editing scenarios.

Recently, instruction-based foundational video-to-video editing models [1,26, 28,39,40] have emerged, jointly conditioning on source videos and text instructions. These models deliver precise, high-fidelity edits while preserving subject identity and motion. However, these models are constrained by a limited temporal context window. A naive workaround is to split a long video into shorter segments and edit each independently, but as shown in Fig. 1(b), this approach leads to significant appearance inconsistencies across segments. In contrast, as illustrated in Fig. 1(b), our memory mechanism enables coherent long-form video editing with consistent appearance and motion across segments. By casting long video editing as a multi-turn editing problem and equipping video-to-video diffusion models [1] with explicit visual memory, we achieve precise and temporally consistent results, even when segments are denoised independently.

Long video generation with memory or context. Although modern video models generate visually impressive results, they inherently lack 3D and longterm consistency due to their 2D frame-based representations and limited temporal context [44]. To address this, recent works first reformulate full-sequence video diffusion models into auto-regressive generators [6, 19, 43, 55], then incorporate memory modules that capture information from past generations. One line of work maintains an external cache of previous generations and conditions the current generation by retrieving a subset of them, either as raw RGB frames [32, 51, 56] or as hidden states [4, 58]. Alternatively, other methods aim to compress the long-term context into compact forms such as latent states [11, 38, 65], semantic video tokens [25, 37, 62], or vision–language model features [7]. Unlike prior literature, we address memory across independently generated editing iterations. In this setup, the model must selectively maintain consistency across multiple edited videos rather than a continuous frame stream.

### 3 Memory-V2V

- 3.1 Overview Existing video-to-video diffusion models are single-turn video editing models, which are trained to generate an output video conditioned on a single source

[Figure 2]

- Fig. 2: Memory-V2V framework. (a) From an external cache of previously edited videos, only the top-k most relevant videos are retrieved and used as memory inputs to ensure cross-iteration consistency. (b) Dynamic tokenizers allocate more tokens to highly relevant videos—preserving fine details while maintaining an efficient overall token budget. (c) Adaptive token merging further reduces redundant computations by compressing less informative frames based on their attention-based responsiveness to the target query.

video and auxiliary editing signals, approximating the conditional distribution p(x|y, c), where x is the output video, y is the user-provided source video, and c denotes additional conditions like text or camera pose. In contrast, our multiturn video editing task requires incorporating prior editing history by allowing y to include both the current input video and previously edited videos, enabling the model to maintain cross-edit consistency across iterations.

Memory-V2V achieves this through a hybrid retrieval and compression strategy, built upon an effective memory representation. In the following sections, we first detail our framework in the context of video novel view synthesis. We consider ReCamMaster [2], a pretrained video DiT model capable of single-turn video novel view synthesis, as our base model. We then introduce the core components of Memory-V2V, including an efficient video retrieval mechanism that selectively fetches relevant past videos from an external cache, dynamic tokenization strategies to represent the retrieved videos, and a compressor that effectively processes the retrieved information (Sec. 3.3). Retrieval is non-parametric, while the tokenization and merging modules are learned end-to-end. Finally, we extend Memory-V2V to text-guided, long-form video editing by formulating it as a multi-turn video editing task (Sec. 3.4).

- 3.2 Video Retrieval-based Dynamic Tokenization After each editing iteration j, we store the latent video E(xj) ∈ RF×H×W×C in

an external cache Ω, indexed by its corresponding camera trajectory ccamj . The latent video’s temporal and spatial dimensions are denoted by F,H,W, and C

is the latent channel size, and the context encoder E corresponds to the VAE encoder. At the i-th editing iteration (i>j>0), our goal is to produce a new video xi that remains consistent with both the user-input video y and the previously generated videos in Ω = {(E(xj),ccamj )|0 < j < i}, while accurately following the new target camera trajectory ccami . Since the total number of cached frames in Ω increases rapidly over multiple editing turns, it is infeasible to condition the pretrained video-to-video DiT model on the entire editing history. We argue that only a subset of the previously edited videos is necessary to maintain spatial and temporal consistency. To this end, we sort all cached latent videos in Ω and dynamically retrieve a set of the top-k most relevant ones, denoted as Ω˜.

Video FOV retrieval. In case of video novel-view synthesis, we determine the relevance via our proposed VideoFOV retrieval algorithm, which quantifies the geometric overlap between the field-of-view (FOV) of the target camera trajectory ccami and those of cached videos. Specifically, given a target camera trajectory ccami = {ccami,t |1 ≤ t ≤ F}, where F is the number of frames, we place a unit sphere at the first camera position ccami,1 and uniformly sample M viewing directions on its surface 3. For each frame along the trajectory, a sampled point is marked as visible if it lies within the projected image bounds and has positive depth after being transformed to the camera’s local coordinate system. The perframe FOV Fframe(ccami,t ) is defined as the set of visible sampled points, and the video-level FOV Fvideo(ccami ) is obtained as the union of all frame-level FOVs:

Fvideo(ccami ) =

F

Fframe(ccami,t ). (1)

t=1

Given a target trajectory ccami and a cached trajectory ccamj , we define two complementary FOV similarity metrics:

soverlap(ccami ,ccamj ) = |Fvideo(ccami ) ∩ Fvideo(ccamj )| |Fvideo(ccami ) ∪ Fvideo(ccamj )|

,

scontain(ccami ,ccamj ) = |Fvideo(ccami ) ∩ Fvideo(ccamj )| |Fvideo(ccami )|

.

The final relevance score is a weighted combination:

(2)

s(ccami ,ccamj ) = λsoverlap(ccami ,ccamj ) + (1 − λ)scontain(ccami ,ccamj ), (3) where we set λ=0.5.

All cached videos in Ω are then ranked by s(ccami ,ccamj ), and the top-k highest-scoring videos are selected as the retrieved video set Ω˜. The full algorithm and implementation details are provided in the supplementary material.

- 3 In all experiments, we set M = 64, 800.

[Figure 3]

- Fig. 3: Qualitative results for multi-turn video novel view synthesis. Compared to baselines, Memory-V2V (Ours) generates videos from new camera trajectories while maintaining consistency across all novel regions generated (e.g., red & blue box).

Dynamic tokenization. Unlike long video generation [32,51], where the memory context typically consists of a small number of frames (<10), the multi-turn video editing treats entire videos as retrieval units. As iterations accumulate, the total number of past conditioning frames can easily reach hundreds 4, making direct encoding computationally prohibitive. Hence, we propose to tokenize the conditional videos dynamically based on their relevance to the current edit.

Our goal is to apply multiple learnable tokenizers to each retrieved video. In practice, we train tokenizers with spatio-temporal compression factors (f′×h′×w′) of 1×2×2, 1×4×4, and 1×8×8, where the 1×2×2 tokenizer processes the userinput video, the 1×4×4 tokenizer processes the top-3 most relevant retrieved videos, and the 1×8×8 tokenizer processes the remaining ones. This adaptive tokenization preserves fine-grained details for the most relevant videos while keeping the total token count manageable.

- 3.3 Adaptive Token Merging While retrieval-based dynamic tokenization effectively allocates the token budget, the quadratic complexity of the DiT’s self-attention still leads to high computational cost as the token sequence length increases. Recent studies [50,61,64] reveal that DiT attention maps are inherently sparse, with only a small subset of entries contributing meaningfully to the output. Hence, we propose to adaptively merge unresponsive tokens before the attention operation. This strategy avoids redundant computation while preserving the essential context information.

Frame-level token responsiveness. Previous studies [4,21,52,63] show that the spatially averaged attention features (e.g., query or key features) serve as

- 4 For video novel view synthesis, a single video contains 81 frames.

- Table 1: Pearson/Spearman correlations and Bottom-k overlap (k=50%) evaluating consistency of frame responsiveness across transformer blocks. Correlations measure linear and rank-order alignment. Bottom-k overlap reflects whether low-responsive frames remain consistently uninformative across layers.

Pearson r ↑ Spearman ρ ↑ Bottom-k overlap (%) ↑

DiT Block 1 vs 2–30 0.608 ± 0.137 0.506 ± 0.267 0.730 ± 0.160 DiT Block 11 vs 12–30 0.723 ± 0.115 0.657 ± 0.139 0.758 ± 0.120 DiT Block 21 vs 22–30 0.753 ± 0.144 0.683 ± 0.126 0.793 ± 0.114

a strong proxy for measuring how much each frame contributes to the model’s prediction. Formally, given query, key, and value matrices Q,K,V ∈ RN×D within a DiT block, we represent all tokens belonging to frame t with a single aggregated vector K¯t obtained by spatially averaging the key features:

1 |It| i∈I

K¯t =

Ki, (4)

t

where It denotes the set of token indices for frame t. Next, we estimate the responsiveness of each frame by computing its maximum attention response to the target queries Qtgt:

q · K¯t⊤

(softmax

). (5)

√

Rt = max

D

q∈Qtgt

The responsiveness score Rt measures how strongly frame t influences the current generation. A higher Rt indicates that at least one target query token strongly attends to that frame whereas low responsive frames can be safely compressed.

Adaptive token merging. We introduce a learnable convolutional operator Cθ that adaptively merges tokens belonging to frames with low responsiveness. Given a set of tokens {Xi}i∈It

from frame t with low responsiveness score Rt, the operator fuses them into a compact representation:

Nt

X˜t = Cθ({Xi}i∈It

), X˜t ∈ R

r ×D, (6)

where r is the spatial-temporal reduction factor. We scale r proportionally to the number of conditional videos, as a larger memory context introduces higher redundancy and thus benefits from stronger compression. Empirically, we find that our adaptive merging strategy better preserves generation quality compared to completely discarding low-importance tokens (see Fig. 7).

Block selection for token merging. To determine where to apply token merging within the DiT architecture, we analyze the stability of responsiveness scores Rt across transformer blocks. In a 30-block DiT, we group layers into early (1–10), middle (11–20), and late (21–30) stages and measure how well the first block in each stage (1, 11, 21) correlates with subsequent blocks. As shown in Tab. 1, the first block (Block 1) shows weak correlation with the rest (Blocks 2–30), meaning early merging risks discarding frames that later become important. Responsiveness stabilizes in mid and late transformer stages, where

[Figure 4]

- Fig. 4: Long-video editing as multi-turn video editing with Memory-V2V. (a) We extend target videos from an existing video editing dataset for Memory-V2V training. (b) During inference, the external cache Ω stores the editing history as source–target video pairs. At the i-th editing turn, relevant memory videos are retrieved based on the similarity between source video segments.

low-importance frames remain consistently uninformative. We therefore apply token merging at Blocks 10 and 20, where representations are sufficiently mature for reliable compression.

#### 3.4 Extension to Text-guided Long Video Editing

We show that Memory-V2V can be applied to other video editing tasks such as text-guided long video editing. We assume access to a pretrained single-turn text-based video editing model, LucyEdit [1], as our base model in this experiment. Given a long user-input video y typically exceeding 200 frames, which is significantly longer than the temporal context window of the base model (∼ 81 frames), we reformulate the task as a memory-aware iterative editing.

We first divide the long source video y into shorter segments y1,...,yT that fit within the base model’s temporal window and iteratively edit each segment. When editing the i-th source segment yi, the top-k most relevant edited segments are retrieved from an external cache Ω = {E(xj) | 0 < j < i}, which stores all past edited segments as video latents. Unlike camera-conditioned video novel view synthesis where each generated video xj can naturally be indexed by its camera pose ccamj for retrieval, text-conditioned editing poses a unique challenge. Text instructions are often ambiguous, making text-based similarity unreliable. Instead, we retrieve memory segments based on visual similarity between source video segments using DINO features [36]. Detailed retrieval algorithms are pre-

[Figure 5]

- Fig. 5: Text-guided long video editing results by iterative multi-turn editing. In contrast to (iterative) LucyEdit [1] or other frame level propagation baselines, Memory-V2V (Ours) consistently inserts the same ‘hat’ (left) and changes apple into the same ‘orange’ (right) across all segments.

##### Table 2: Quantitative comparison on multi-turn video novel view synthesis.

Multi-view Consistency ↓ Camera Accuracy ↓ Visual Quality ↑ 1st Iter. vs 2nd Iter.

1st Iter. vs 3rd Iter.

2nd Iter. vs 3rd Iter.

Subject Consistency

Imaging Quality

Temporal Flickering

Motion Smoothness

Avg. Score RotErr TransErr

TrajCrafter [57] 0.1254 0.2110 0.2090 0.1818 3.66 57.44 0.9452 0.7385 0.9661 0.9880 ReCam (Ind) [2] 0.1665 0.1982 0.2031 0.1892 1.97 24.23 0.9483 0.7186 0.9765 0.9931 ReCam (AR) [2] 0.1181 0.1985 0.1290 0.1485 - - - - - Memory-V2V (Ours) 0.1168 0.1525 0.1379 0.1357 1.65 13.47 0.9494 0.7242 0.9728 0.9933

sented in the supplementary. The retrieved videos are then dynamically tokenized where adaptive token merging is also applied when generating the edited segment xi. Once all segments are iteratively edited, they are stitched together to form a single output video x (see Fig. 4(b) for overview).

By casting long video editing as a multi-turn memory-conditioned process, we eliminate the need for long source–target video pairs, which are difficult to obtain in practice. Instead of requiring long paired data, Memory-V2V can be trained using short source–target pairs by extending only the target videos. Concretely, starting from a public short video editing dataset, we generatively extend each target video using an off-the-shelf video extension model [62]. The overall pipeline is visualized in Fig. 4(a).

- 4 Experiments

- 4.1 Implementation Details

Video novel view synthesis. Starting from the single-turn video novel view synthesis model ReCamMaster [2], we finetune it for iterative novel view synthesis using a synthetic multi-camera video dataset containing 10 synchronized

##### Table 3: Quantitative comparison results for text-guided long video editing. The higher the better for all metrics.

Subject Consistency

Background Consistency

Aesthetic Quality

Imaging Quality

Temporal Flickering

Motion Smoothness

DINO-F CLIP-F

LucyEdit (Ind) 0.8683 0.9026 0.4601 0.6429 0.9844 0.9915 0.6856 0.8225 LucyEdit (FIFO) 0.8737 0.9042 0.4527 0.5598 0.9844 0.9919 0.6784 0.8198 TokenFlow [16] 0.9191 0.9172 0.4206 0.6649 0.9845 0.9896 0.6480 0.7563 RAVE [29] 0.9220 0.9322 0.4703 0.6012 0.9813 0.9872 0.7308 0.7713 CCEdit [15] 0.8698 0.9101 0.4113 0.6592 0.9788 0.9762 0.6088 0.7228 Memory-V2V (Ours) 0.9326 0.9233 0.4950 0.6759 0.9862 0.9939 0.8019 0.8741

##### Table 4: Quantitative ablation on each component of Memory-V2V.

Time (s) ↓

Subject Consistency ↑

Aesthetic Quality ↑

Imaging Quality ↑

Motion Smoothness ↑

MEt3R ↓

Dynamic Tokenization Only 980.80 0.2234 0.9368 0.5632 0.7307 0.9917 + Video Retrieval 965.95 0.2169 0.9338 0.5632 0.7288 0.9918 + Adaptive Token Merging 661.31 0.2344 0.9361 0.5626 0.7287 0.9921

+ All (Full Model) 648.5 0.2208 0.9351 0.5636 0.7300 0.9919

videos per scene, each captured from distinct camera trajectories. We finetune the self-attention layers, MLP projector, and camera encoder from the base model, together with the newly introduced dynamic tokenizers and adaptive token compressors. During training, 1–6 videos are randomly sampled at each iteration to serve as memory videos Ω˜. To ensure all tokenizers with varying kernel sizes are trained without reliance on the other, we randomly use one of the tokenizer for the 50% of the training time, while for the other 50% we use a mix of them. Moreover, adaptive token merging is enabled with 50% probability, using a compression factor r randomly sampled from [0.3,0.7].

Text-guided long video editing. We extend the single-turn, instructiondriven video editing model LucyEdit [1] to a long video editor. We finetune it to take both the source video and previously edited (history) videos as input. Training is conducted on 56K samples [28] filtered from the publicly available Señorita-2M dataset [67]. Each sample in Señorita contains a triplet of text instruction, source video, and target video. However, since both the source and target clips in Señorita are short-form videos, we employ a generative video extension model [62] to temporally extend the target videos, using these extended segments as the memory videos during training (see Fig. 4(a)).

Both video novel synthesis and long video editing models are trained with the rectified flow matching loss [13] on 32 A100 GPUs with a total batch size of 32. Due to strong pretrained initialization, stable convergence is achieved within 1–2K finetuning steps.

#### 4.2 Memory-V2V for Video Novel View Synthesis

In this experiment, we evaluate on 40 publicly available videos [2, 35, 48, 57], each serving as a user input. We compare Memory-V2V against ReCamMaster (ReCam) [2] and TrajectoryCrafter (TrajCrafter) [57]. For ReCamMaster, two inference modes are evaluated: (1) ReCam (Ind), which reuses the same userinput video as the source for every iteration, and (2) ReCam (AR), which uses

the previous output as the next input in a sequential manner. ReCam (Ind) avoids error accumulation but lacks cross-turn memory, while ReCam (AR) enforces sequential conditioning yet accumulates drift across iterations. Qualitative comparisons are shown in Fig. 3, and quantitative results in Tab. 2. For each input video, we iteratively generate three outputs with highly overlapping camera trajectories (e.g., pan-left and orbit-right) and measure pairwise consistency across the 1st–2nd, 1st–3rd, and 2nd–3rd iterations. We adopt MEt3R to measure cross-view geometric consistency, VBench [20] to evaluate visual quality, and report rotation & translation errors. We observe that as the number of iterations increases, the cross-consistency significantly drops for the base models (TrajCrafter and ReCam (Ind)). While ReCam (AR) preserves consistency between subsequent iterations (e.g., 2nd and 3rd), it fails to keep other iterations (e.g., 1st and 3rd) consistent. In contrast, Memory-V2V results in consistent generations across all iterations while successfully preserving the visual quality and improving camera adherence of the base model.

- 4.3 Memory-V2V for Long Video Editing We evaluate the long video editing performance on 50 videos from the Señorita test set [67]. We first compare against two variants of LucyEdit [1]: LucyEdit (Ind), which edits each video segment independently, and LucyEdit (FIFO), which follows FIFO-Diffusion’s diagonal denoising strategy to process consecutive frames with increasing noise levels, simulating autoregressive generation [30]. In addition, we consider three more baselines: TokenFlow [16], which propagates diffusion features across frames to encourage temporal consistency; RAVE [29], which employs randomized noise shuffling to produce consistent videos while enabling fast video editing; and CCEdit [15], which uses keyframe appearance and structure guidance for consistent editing. As shown in Fig. 5, while TokenFlow and RAVE can maintain coarse edited-object consistency across frames, they often fail to preserve fine-grained subject identity and unintentionally modify background regions during editing. CCEdit frequently exhibits limited editing controllability, leading to incomplete or inconsistent edits. LucyEdit preserves subject identity and background structure via token reuse through self-attention, but still struggles to maintain edited-object consistency over long sequences. In contrast, Memory-V2V simultaneously preserves subject identity, background structure, and edited-object consistency by leveraging explicit memory conditioning, enabling stable and coherent edits across extended video sequences (>200 frames). We report visual quality and consistency metrics in Tab. 3, using VBench and cross-frame DINO/CLIP similarity metrics [36, 41], where Memory-V2V consistently outperforms all baselines.
- 4.4 Ablation Study Ablation on each component. We first ablate the effects of main components of Memory-V2V, as summarized in Tab. 4 and Fig. 6. Dynamic tokenization combined with video retrieval improves cross-iteration consistency, while adaptive token merging notably reduces computational cost. As shown in Fig. 6, retrieval guided by VideoFOV effectively preserves long-term consistency even

[Figure 6]

- Fig. 6: Qualitative ablation on video retrieval and adaptive token merging.

[Figure 7]

- Fig. 7: Qualitative ablation on token responsiveness

[Figure 8]

Fig. 8: Mean responsiveness Rt analysis.

and reduction strategies.

after multiple iterations (e.g., between the 1st and 5th generations), whereas token merging maintains efficiency without visible degradation.

Ablation on token responsiveness and token reduction. We analyze the effect of conditioning tokens ranked by their responsiveness (Sec. 3.3) under two reduction strategies: merging and discarding. 5 Multi-turn video editing needs to achieve cross-iteration consistency across edits while preserving the visual quality of the generated videos.

Highly responsive conditioning tokens are essential for cross-consistency across

edits. When these tokens are either merged (Fig. 7(d)) or discarded (f), the overlapping novel-view regions become inconsistent. The red-boxed region, which should match (b), fails to remain consistent in both cases. This observation is also supported by the frame-responsiveness analysis in Fig. 8(i), where highresponsive tokens are strongly correlated with the target camera trajectory.

5 Tokens are partitioned by responsiveness score Rt: the top 50% are high-responsive, and the bottom 50% are low-responsive. Additional experimental details are provided in the supplementary material.

[Figure 9]

- Fig. 9: Computational cost (FLOPs, Latency) analysis. (a) Comparison between Memory-V2V and naive context-window scaling for memory videos. (b) Comparison of Memory-V2V with and without adaptive token merging. Notably, the computational gains increase as the number of memory videos grows.

Discarding low-responsive tokens may degrade fine visual details and motion. Fig. 7 shows that discarding these tokens introduces unnatural artifacts in the blue-highlighted regions. Furthermore, Fig. 8(ii) shows that these tokens are less correlated with the video frames generated in the current turn, but they may still contribute to structural consistency. Therefore, instead of discarding them, merging low-responsive tokens reduces computation without affecting cross-consistency and visual fidelity. Adaptive merging preserves cross-turn consistency while reducing computation, whereas discarding low-responsive tokens introduces visual artifacts. Additional examples are provided on the supplementary project page.

Computational cost analysis. We analyze the computational gains of Memory-

V2V in Fig. 9. As shown in Fig. 9(a), compared to naive context-window scaling, our method reduces FLOPs and latency by over 90%. Furthermore, Fig. 9(b) shows that adaptive token merging further reduces FLOPs and latency by 30%.

### 5 Conclusion

We formulated multi-turn video editing as a distinct consistency challenge and introduced Memory-V2V, a memory-augmented framework that treats prior edits as structured constraints. By combining task-aware retrieval and relevanceaware compression, Memory-V2V enables scalable and consistent iterative editing across diverse video tasks.

Limitations. Memory-V2V inherits the limitations (e.g., large view changes) of the underlying single-turn editing model on which it is built. Additionally, since the training data consists only of continuous single-shot videos, MemoryV2V may struggle with editing multi-shot long videos containing abrupt scene transitions. Multi-shot video editing is an exciting research direction we would like to explore in the future.

### Acknowledgements

We thank Minguk Kang and Sihyun Yu for their insightful discussions at the early stage of the project.

This supplementary material is organized as follows. Sec. A introduces the rectified flow matching loss used to train Memory-V2V. Sec. B provides experimental details for the ablation study. Sec. C presents toy experiments analyzing suitable context encoders for memory representation, and Sec. C.1 provides further implementation details and results. Sec. D describes the retrieval mechanisms for video novel view synthesis and text-guided long video editing. Sec. E details the training and inference setups, including RoPE design and camera conditioning. Sec. F presents additional qualitative results. Finally, Sec. G discusses limitations, failure cases, and future directions.

A full set of video results is available on the project page. Please refer to the included index.html).

### A Preliminary: Rectified Flow Matching

Recently, flow-based training frameworks [8,14,33,66] have gained attention for their ability to progressively transform samples from a prior distribution to the target data distribution through a series of linear interpolations. Specifically, we define a velocity field vt(x) of a flow ψt(x) : [0,1] × Rd → Rd that satisfies ψt(x0) = xt and ψ1(x0) = x1. Here, the ψt is uniquely characterized by a flow ODE:

dψt(x) = vt(ψt(x))dt (7)

In particular, linear conditional flow defines the flow as xt = ψt(x1|x0) = (1 − t)x0+tx1. Then, we can compute the velocity field vt(xt|x0) = ψ˙t(ψt−1(xt|x0)|x0) = x1 − x0, leading to the following conditional flow matching loss:

0,ϵ∼N(0,I) ∥(x0 − ϵ) − vθ(xt,t)∥2 . (8)

LRF = Et,x

During inference, the predicted velocity is utilized to guide the transformation of an initial noise sample towards the target data distribution through a reverse integration process. This approach enables efficient and controlled generation of high-quality samples.

### B Additional Ablation Study Details

- B.1 Video retrieval and adaptive token merging experiments.

We design this experiment to analyze how video retrieval and adaptive token merging affect long-term consistency. Specifically, the first and fifth iterations use camera trajectories with overlapping visible regions (e.g., pan-right followed by orbit-left, or pan-left followed by orbit-right), while the remaining iterations use randomly sampled trajectories.

For runtime and visual quality evaluation, metrics are computed on the fifthiteration outputs. For MEt3R, we measure consistency between the videos generated in the first and fifth iterations to assess whether multi-view consistency is preserved over long editing sequences. The results show that video retrieval improves cross-iteration multi-view consistency, while adaptive token merging significantly reduces runtime without degrading generation quality or consistency.

- B.2 Token reduction experiments.

In Fig. 7 (main) and Fig. 8 (main), we conduct three-turn novel view synthesis. The first and third iterations use camera trajectories with overlapping viewpoints, while the second iteration uses an unrelated trajectory. For example, when the first and third iterations correspond to pan-right and orbit-left, the second iteration uses a trajectory such as tilt-up with minimal overlap.

During the third iteration, we manipulate tokens from 50% of the frames by either merging or discarding them. For high-responsiveness manipulation, we operate on tokens from the top 50% of frames with the largest responsiveness scores (Rt). For low-responsiveness manipulation, we operate on tokens from the bottom 50% of frames with the smallest responsiveness scores.

For Fig. 8, we analyze the responsiveness of historical frames when generating the third iteration. We first measure the responsiveness of frames that are geometrically related to the target camera trajectory (e.g., frames from videos with trajectories such as pan-left when generating orbit-right). The responsiveness scores are averaged across transformer blocks.

We then perform the same analysis for frames unrelated to the target trajectory, including frames from the static source video and from videos generated with unrelated trajectories (e.g., tilt-up). Their responsiveness scores are also averaged across blocks.

As shown in Fig. 8 (main), frames with high responsiveness in early transformer blocks tend to remain consistently high across later blocks, while lowresponsive frames remain consistently low. Moreover, frames geometrically aligned with the target trajectory generally exhibit higher responsiveness than unrelated frames.

### C Proof-of-Concept Experiment: Ideal Context Encoder

Maintaining semantic consistency and persistency when editing long videos in chunks or in multi-turn editing setup requires to encode prior editing results as

[Figure 10]

- Fig. 10: Comparison of different memory encoders on two-turn novel view synthesis. The red-colored box depicts the novel region which are expected to be consistent between x1 and x2.

additional context which can then be utilized as constraints for the next editing round. Hence, an important design choice is how the context is represented. We devise a simple experiment setup to evaluate the representation capability of different potential context encodings. We limit our experiment to the novel-view synthesis task in a constrained two-turn editing setup using ReCamMaster as the base model. In particular, given an input video y, in the first turn we generate a video x1 from a new camera trajectory ccam1 . Using an arbitrary encoder E, we extract the context E(x1) from x1 and in the second turn, we generate a video

- x2 from a camera trajectory ccam2 , conditioning on both the original source video
- y as well as the encoded context E(x1). If E(x1) has sufficient representation power, we expect that the novel content generated in x1 and x2 to be consistent.

We experiment with three different choices of pretrained context encoders for the novel view synthesis task: (i) a recurrent 3D reconstructor model CUT3R [46] (ii) a state of the art novel view synthesis network LVSM [27], and (iii) a video VAE [31]. For each setting, encoded representations E(x1) are patchified to match the dimensionality of the DiT’s video tokens, allowing interaction within the self-attention layers. During finetuning, we freeze the context encoder E and train only the patchification layer and the ReCamMaster model to accept additional memory context (see the supplementary for details). Moreover, during training, we randomly sample multiple videos of the same scene: one as the original source video y and another as the source of contextual memory x1.

We provide examples from each of the context encodings in Fig. 10. We observe that the state tokens from CUT3R or LVSM fall short for encoding

detailed appearance in newly generated regions. In contrary, using the same video latent space achieves the best performance in terms of quality and consistency of the results. Hence, we adopt this option for the context representation and focus on how to efficiently retrieve and process the context representations obtained from multiple past generations as will be discussed next.

#### C.1 Context Encoder Setup

Recurrent 3D reconstructor CUT3R. CUT3R [46] is designed to predict the 3D point representation rom a video stream by maintaining a recurrent geometric state. Given an input frame It, the model first extracts a feature map using a ViT encoder,

Ft = Encoder(It). To aggregate geometry over time, these features are passed to a decoder along with the previous state st−1, producing

Ft′, st = Decoder(Ft,st−1),

where the updated state st ∈ R768×768 accumulates 3D cues across frames that function as the model’s internal geometric memory. 3D point map is then pre-

dicted from Ft′.

To evaluate whether this recurrent state can serve as a memory representation, we sequentially update st over video frames, project the state via a learnable MLP into the transformer latent space, and condition the diffusion model through a dedicated branch (following the multimodal conditioning strategy used in SD3 [13]). The branch shares self-attention with the backbone while using separate modulation and feed-forward layers. In specific, we fine-tune for 2K steps with a batch size of 32. However, the diffusion transformer backbone fails to effectively leverage the CUT3R state: generations remain unstable and lack coherent geometry, as can be seen in Fig. 10.

These observations further align with our reconstruction analysis. Following CUT3R’s original procedure, we attempted to reconstruct scenes using only the recurrent state and a ray map. As shown in Fig. 11, the resulting geometry is coarse and blurry, lacking meaningful fine-grained detail. This suggests that the recurrent state encodes limited and lossy structural information. As a result, when used as a conditioning signal, it fails to provide sufficiently informative guidance for generation—leading to the degraded performance observed in our CUT3R-conditioned model.

- 3D Novel view synthesis model LVSM. LVSM [27] synthesizes novel views by jointly encoding input images and their corresponding Plücker ray embed-

dings. Given an input image Ii and Plücker embedding P˜i, the model projects their concatenation into a shared feature space:

xi = Linear([Ii, P˜i]) ∈ Rd.

[Figure 11]

[Figure 12]

- Fig. 11: Reconstruction results using CUT3R state s.

Fig. 12: Novel view synthesis results using LVSM state z.

These tokens, together with an initial latent token e, are processed by the encoder,

x1,x2,...,xn, z = Encoder(x1,x2,...,xn,e),

producing a latent state z. For novel view rendering, the decoder conditions on the Plücker embedding q of the target camera rays and predicts

z′, y = Decoder(z,q), where y is the rendered view.

We evaluate whether the latent state z could serve as a meaningful memory representation. Similar to the CUT3R setup, we project z into the DiT feature dimension via a learnable MLP and inject it through a dedicated conditioning branch that shares self-attention layers with the main model but uses separate modulation and feed-forward layers. The model is trained for 2K steps with batch size 32.

However, as illustrated in Fig. 10, conditioning on z provides little benefit. While LVSM’s decoder can exploit z for its own renderer, the representation lacks the transferable, fine-grained geometric cues needed to guide a diffusionbased generator. Consequently, the conditioned synthesis is unstable and does not improve multi-turn consistency, suggesting that the LVSM latent space is not structured for effective memory conditioning.

### D Video Retrieval Algorithms

#### D.1 Retrieval for Video Novel View Synthesis

To compute the VideoFOV retrieval score, we first uniformly sample a dense set of points on a unit sphere centered at the target camera position (see Fig. 13(a)). In specific, we generate a latitude–longitude grid with:

Nθ = 180, Nϕ = 360,

[Figure 13]

Fig. 13: Sphere sampling and Field-of-View visualization.

resulting in a total of |M| = NθNϕ = 64,800 sampled directions. The sampling index set is defined as:

M = {(u,v) | u ∈ {0,...,Nθ−1}, v ∈ {0,...,Nϕ−1}}, (9) where each pair (u,v) uniquely maps to a 3D point on the sphere.

Fig. 13(b) illustrates how the sampled spherical points are tested for visibility given camera intrinsics and extrinsics. During training, intrinsics are taken directly from the dataset; during inference, we adopt a fixed intrinsic matrix from the training set. To ensure consistent evaluation, each frame’s camera pose is represented as a relative transform with respect to the first frame of the target video.

Each sampled point (u,v) ∈ M is then projected to the image plane. point is considered visible if it falls within image bounds with a valid (positive) depth value. The per-frame field of view (FOV) is defined as:

Fframe(ccami,t ) = {(u,v) ∈ M | (u,v) projects inside the image}. (10)

We accumulate these visibility sets across all frames to compute the video-level FOV, and then evaluate the relevance score using Eq. (1), (2), and (3). The complete retrieval workflow is provided in Alg. 1.

Example retrieval results using the proposed VideoFOV algorithm are shown in Fig. 14. The top row displays the target (query) video, while subsequent rows present retrieved videos ranked by descending relevance.

#### D.2 Retrieval for Text-guided Long Video Editing

For text-guided long video editing, retrieval is performed by selecting previously edited segments whose corresponding source segments are most visually similar to the current input segment. Segment-level features are computed by extracting DINOv2 [36] embeddings for all frames within a segment and averaging them into a single descriptor. Cosine similarity between descriptors is then used to rank

[Figure 14]

Fig. 14: Example video retrieval results using Alg. 1.

[Figure 15]

Fig. 15: Example video retrieval results using Alg. 2.

relevance. To maintain temporal coherence between consecutive segments in the output, the most recently edited segment is always included in the retrieved set. The full procedure is provided in Alg. 2. Example retrieval results are shown in Fig. 15, where semantically similar segments consistently rank highest.

### E Additional Training and Inference Details

#### E.1 Video Novel View Synthesis

Positional embedding (RoPE) design. Video diffusion models typically embed the spatiotemporal structure of videos using 3D Rotary Position Embeddings (3D-RoPE), where each token is assigned a positional encoding derived from its spatial coordinates (x,y) and its temporal frame index t. Since all frames share the same spatial resolution, the spatial RoPE indices remain fixed across the video. In contrast, the temporal RoPE indices are bounded by the maximum video length seen during training. When the model is required to generate videos longer than this training horizon or incorporate additional conditioning videos during multi-turn editing, it must extrapolate to unseen temporal positions, which often leads to temporal drift or inconsistencies.

To resolve this issue, we first formalize the three types of videos involved in multi-turn video editing. The target video is the video being generated in the current iteration and has temporal length T. The user-input video is an externally provided video used to guide generation. The memory videos consist of previously generated videos that are cached and reused as conditioning inputs.

Unlike the target or user-input videos, the number of memory videos may grow unboundedly across editing iterations, requiring a positional encoding scheme that can support an expanding conditioning set without encountering unseen RoPE indices.

To provide a stable positional structure across these heterogeneous video sources, we assign each category a disjoint range of temporal RoPE indices. For a target video of length T, we define

τtgt = {0,1,...,T−1}, (11) τusr = {T,T + 1,...,2T−1}, (12)

τmem = {2T,2T + 1,...,3T−1}. (13)

Each token at spatial location (x,y) and temporal index t receives a positional encoding given by

RoPE(x,y,t) = [RoPEx(x),RoPEy(y),RoPEt(t)], (14) where t is sampled from τtgt, τusr, or τmem depending on the video type.

Because this multi-video type configuration does not naturally arise during training, we adopt a mixed training strategy that enables the model to correctly interpret and utilize the hierarchical RoPE layout during inference. When both the user-input and memory RoPE ranges are active, we perturb the memoryvideo tokens with Gaussian noise to prevent the model from overfitting to artifacts that may accumulate across iterations. Specifically, for a memory token zmem, we use

z˜mem = zmem + ϵ, ϵ ∼ N(0,σ2I). (15) This encourages the model to prioritize the cleaner and more reliable user-input video. Complementarily, we disable the user-input RoPE range with probability p during training, forcing the model to rely solely on τmem. This RoPE dropout enables memory videos to serve as an independent conditioning signal.

Through this formulation and training procedure, the model learns a coherent and scalable RoPE structure that remains stable even as memory videos accumulate. The resulting system supports multi-turn video generation by enabling controlled information flow from the user-input video to the target video while mitigating the risk of error propagation through the memory video sets.

Camera conditioning strategy. To incorporate viewpoint geometry into the generative process, we condition the feature representations on the camera extrinsics associated with each video in the conditioning set. For a video v consisting of f frames, we denote its camera representation by

camv ∈ Rf×(3×4),

where each 3 × 4 block corresponds to the rigid transformation [R | t] describing the rotation and translation of the camera for an individual frame. These

[Figure 16]

Fig. 16: RoPE assignments to mitigate train-inference gap.

parameters are then projected into the feature space through a fully connected layer Ec(·).

In ReCamMaster [2], target camera camt information is injected uniformly across all features. Let Fo denote the output feature of the spatial attention layer and Fi denote the input feature to the subsequent 3D self-attention layer. The baseline formulation applies a residual update of the form

Fi = Fo + Ec(camt), (16)

implicitly assuming that all conditioning features are derived from a single video captured under a single camera trajectory.

However, within our memory-driven framework, each memory video is generated under its own distinct camera extrinsics. This means that the conditioning set does not share a unified viewpoint, and the model must interpret the geometric context of each video individually. To achieve this, we assign to every feature token the camera embedding corresponding to the specific video from which it originates. If Fo(v) denotes a feature coming from video v, and camv is the associated camera trajectory, the camera-aware conditioning becomes

Fi(v) = Fo(v) + Ec(camv). (17)

By embedding camera trajectories on a per-video basis in this manner, the model becomes capable of distinguishing heterogeneous viewpoints present across the target, user-input, and memory videos. This explicit geometric conditioning improves the model’s ability to handle complex camera motion and rotation, enabling more stable and coherent viewpoint reasoning during multi-turn video generation.

#### E.2 Text-guided Long Video Editing

Dataset construction. Existing video editing datasets [3, 67] are not suited for long video editing, as they contain only short clips with limited temporal context. To enable training of Memory-V2V in this setting, we construct a longform dataset by temporally extending video editing pairs from Señorita-2M [67]. Señorita-2M provides stable local editing pairs—where foreground edits minimally affect the background—but each clip is only 33 frames long, which is insufficient for long-horizon editing. To address this limitation, we apply an offthe-shelf generative video extension model, FramePack [62]. For each target clip, we retain the original 33 frames and generate 200 additional frames via forward temporal extension, resulting in a 233-frame sequence. The extended portion serves as memory during training; at each iteration, we randomly sample segments from this extended portion to function as memory-conditioning videos.

Positional embedding (RoPE) design. Long-video editing follows the similar RoPE strategy used for video novel view synthesis. For each target segment of T frames, we allocate disjoint temporal RoPE index ranges to distinguish token groups. Specifically, the target segment is assigned as

τtgt = {0,1,...,T−1},

while the immediately preceding segment is assigned as

τprev = {T,T + 1,...,2T−1}, and other remaining segments are assigned as

τmem = {2T,2T + 1,...,3T−1}.

This hierarchical indexing preserves strong continuity with the most recent segment while still providing broader historical context from earlier memory clips.

Mitigating training-inference gap. During training, as shown in Fig. 16 (a), the target video is temporally extended using a generative model; consequently, the target frames appear earlier in the sequence, while the extended (memory) frames appear later. RoPE indices are therefore assigned in chronological order from target to memory frames. However, during inference, the memory frames correspond to previously generated results, which naturally occur before the target frames in time. If RoPE indices are assigned in the same forward order as shown in Fig. 16 (b), this mismatch leads to a clear training-inference gap. To eliminate this discrepancy, we reverse the RoPE assignment for memory frames during inference, as illustrated in Fig. 16 (c). By flipping the RoPE index order, the positional structure of the conditioning sequence becomes consistent with the ordering used during training, thereby preventing the training-inference mismatch.

[Figure 17]

Fig. 17: Failure case. Memory-V2V exhibits difficulties when the input long video contains multiple shots with large scene transitions.

### F Additional Qualitative Results

Additional results for multi-turn video novel view synthesis can be found in Fig. 18, while further results for text-guided long video editing are provided in Fig. 19.

### G Additional Discussion

The dataset used to train Memory-V2V for text-guided long video editing consists exclusively of continuous, single-shot videos, without abrupt scene transitions. As a result, Memory-V2V struggles when applied to real long-form content containing multiple shots, where substantial visual discontinuities occur at shot boundaries (see Fig. 17). In such cases, the model may incorrectly propagate objects or textures from the preceding shot into the next (e.g., a hand or accessory reappearing), even if the high-level semantics remain similar.

Additionally, the target videos used during training are extended using a generative video model. These generated extensions exhibit mild flickering at the junction between real and synthesized segments due to temporal inconsistencies, slight changes in tone, or blur artifacts. When such imperfect segments are stored in memory, these deviations accumulate over repeated denoising steps, especially in long sequences, ultimately leading to visible appearance drift.

We believe this limitation can be addressed by training with multi-shot datasets and higher-quality long-video data pairs, and we leave these directions to future work. Moreover, to further enhance the interactivity of Memory-V2V, future work could integrate it with diffusion distillation or autoregressive generation frameworks [6,53–55].

[Figure 18]

##### Fig. 18: Additional qualitative results for multi-turn video novel view synthesis. Refer to our project page for video results.

[Figure 19]

##### Fig. 19: Additional qualitative results for text-guided long video editing. Refer to our project page for video results.

Algorithm 1: VideoFOV Retrieval: top-k memory selection by spherical FOV overlap

Input : Target video v∗ with per-frame cameras {Kt∗, Rt∗, t∗t}Tt=1∗ ; candidate

memory videos {vi}Ni=1 with per-frame cameras {Kti, Rti, tit}Tt=1i ; sphere radius r; number of sphere samples M; balance λ∈[0, 1]; top-k.

Output: Ω˜ = top-k videos ranked by FOV score.

- 1 Function RelativePose(R, t, Rref, tref):

- 2 R′ ← Rref⊤ R
- 3 t′ ← Rref⊤ (t − tref)
- 4 return (R′, t′)

- 5 Function InFOV(p, K, R, t):

- 6 xc ← R(p − t) // World → camera
- 7 if (xc)z ≤ 0 then

- 8 return false

- 9 u˜ ← K xc/(xc)z // Project to pixels
- 10 return u˜ inside image bounds

- 11 Function VideoFOV({Kt, Rt, tt}Tt=1, S = {pm}Mm=1):

- 12 F ← ∅
- 13 for t ← 1 to T do

- 14 for m ← 1 to M do

- 15 if InFOV(pm, Kt, Rt, tt) then

- 16 F ← F ∪ {pm}

- 17 return F

- 18 Reference the target’s first frame as origin
- 19 (Rref, tref) ← (R1∗, t∗1)
- 20 Relativize all frames of v∗ and each vi via RelativePose()
- 21 Sample S = {pm}Mm=1 uniformly on the sphere of radius r
- 22 F∗ ← VideoFOV({Kt∗, Rt∗, t∗t}, S)
- 23 for i ← 1 to N do

- 24 Fi ← VideoFOV({Kti, Rti, tit}, S)
- 25 soverlap ←

|F∗ ∩ Fi| |F∗ ∪ Fi|

- 26 scontain ←

|F∗ ∩ Fi| |F∗|

- 27 si ← λ soverlap + (1 − λ) scontain

- 28 return Ω˜ ← top-k videos by si

Algorithm 2: DINOv2-based Segment Similarity Ranking

Input: Target segment Xtar with frames {Fttar}Tt=1tar; previous segments {Xk}Nk=1 with frames {Ft(k)}Tt=1k ; DINO backbone fθ; flag enforce_recent_first. Output: Sorted indices π; similarity scores {simk}Nk=1.

- 1 Function Descriptor(X):
- 2 Let X contain frames {Ft}Tt=1
- 3 d ← T1 Tt=1 fθ(Ft)

- 4 return d
- 5 Function Sim(di, dj):
- 6 sim ←

⟨di, dj⟩ ∥di∥2 ∥dj∥2

- 7 return sim
- 8 dtar ← Descriptor(Xtar)
- 9 for k ← 1 to N do

- 10 dk ← Descriptor(Xk)
- 11 simk ← Sim(dtar, dk)

- 12 π ← argsort({simk}, descending)
- 13 if enforce_recent_first then

- 14 Move index N to the front of π

- 15 return π, {simk}Nk=1

### References

- 1. AI, D.: Open-weight text-guided video editing (2025), https://platform.decart.ai/
- 2. Bai, J., Xia, M., Fu, X., Wang, X., Mu, L., Cao, J., Liu, Z., Hu, H., Bai, X., Wan, P., et al.: Recammaster: Camera-controlled generative rendering from a single video. arXiv preprint arXiv:2503.11647 (2025)
- 3. Bai, Q., Wang, Q., Ouyang, H., Yu, Y., Wang, H., Wang, W., Cheng, K.L., Ma, S., Zeng, Y., Liu, Z., Xu, Y., Shen, Y., Chen, Q.: Scaling instruction-based video editing with a high-quality synthetic dataset. arXiv preprint arXiv:2510.15742 (2025)
- 4. Cai, S., Yang, C., Zhang, L., Guo, Y., Xiao, J., Yang, Z., Xu, Y., Yang, Z., Yuille, A., Guibas, L., et al.: Mixture of contexts for long video generation. arXiv preprint arXiv:2508.21058 (2025)
- 5. Ceylan, D., Huang, C.H.P., Mitra, N.J.: Pix2video: Video editing using image diffusion. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 23206–23217 (2023)
- 6. Chen, B., Martí Monsó, D., Du, Y., Simchowitz, M., Tedrake, R., Sitzmann, V.: Diffusion forcing: Next-token prediction meets full-sequence diffusion. Advances in Neural Information Processing Systems 37, 24081–24125 (2024)
- 7. Chen, N., Huang, M., Meng, Y., Mao, Z.: Longanimation: Long animation generation with dynamic global-local memory. arXiv preprint arXiv:2507.01945 (2025)
- 8. Chen, S., Ge, C., Zhang, Y., Zhang, Y., Zhu, F., Yang, H., Hao, H., Wu, H., Lai, Z., Hu, Y., Lin, T.C., Zhang, S., Li, F., Li, C., Wang, X., Peng, Y., Sun, P., Luo, P., Jiang, Y., Yuan, Z., Peng, B., Liu, X.: Goku: Flow based video generative foundation models. arXiv preprint arXiv:2502.04896 (2025)
- 9. Chen, W., Ji, Y., Wu, J., Wu, H., Xie, P., Li, J., Xia, X., Xiao, X., Lin, L.: Controla-video: Controllable text-to-video generation with diffusion models. arXiv e-prints pp. arXiv–2305 (2023)
- 10. Cong, Y., Xu, M., Simon, C., Chen, S., Ren, J., Xie, Y., Perez-Rua, J.M., Rosenhahn, B., Xiang, T., He, S.: Flatten: optical flow-guided attention for consistent text-to-video editing. arXiv preprint arXiv:2310.05922 (2023)
- 11. Dalal, K., Koceja, D., Xu, J., Zhao, Y., Han, S., Cheung, K.C., Kautz, J., Choi, Y., Sun, Y., Wang, X.: One-minute video generation with test-time training. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 17702– 17711 (2025)
- 12. Engine, U.: Unreal engine. Retrieved from Unreal Engine: https://www. unrealengine. com/en-US/what-is-unreal-engine-4 (2018)
- 13. Esser, P., Kulal, S., Blattmann, A., Entezari, R., Müller, J., Saini, H., Levi, Y., Lorenz, D., Sauer, A., Boesel, F., et al.: Scaling rectified flow transformers for high-resolution image synthesis. In: Forty-first international conference on machine learning (2024)
- 14. Esser, P., Kulal, S., Blattmann, A., Entezari, R., Müller, J., Saini, H., Levi, Y., Lorenz, D., Sauer, A., Boesel, F., Podell, D., Dockhorn, T., English, Z., Lacey, K., Goodwin, A., Marek, Y., Rombach, R.: Scaling rectified flow transformers for high-resolution image synthesis (2024), https://arxiv.org/abs/2403.03206
- 15. Feng, R., Weng, W., Wang, Y., Yuan, Y., Bao, J., Luo, C., Chen, Z., Guo, B.: Ccedit: Creative and controllable video editing via diffusion models. arXiv preprint arXiv:2309.16496 (2023)
- 16. Geyer, M., Bar-Tal, O., Bagon, S., Dekel, T.: Tokenflow: Consistent diffusion features for consistent video editing. arXiv preprint arXiv:2307.10373 (2023)

- 17. Greff, K., Belletti, F., Beyer, L., Doersch, C., Du, Y., Duckworth, D., Fleet, D.J., Gnanapragasam, D., Golemo, F., Herrmann, C., et al.: Kubric: A scalable dataset generator. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 3749–3761 (2022)
- 18. Gu, Y., Zhou, Y., Wu, B., Yu, L., Liu, J.W., Zhao, R., Wu, J.Z., Zhang, D.J., Shou, M.Z., Tang, K.: Videoswap: Customized video subject swapping with interactive semantic point correspondence. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 7621–7630 (2024)
- 19. Huang, X., Li, Z., He, G., Zhou, M., Shechtman, E.: Self forcing: Bridging the train-test gap in autoregressive video diffusion. arXiv preprint arXiv:2506.08009

(2025)

- 20. Huang, Z., He, Y., Yu, J., Zhang, F., Si, C., Jiang, Y., Zhang, Y., Wu, T., Jin, Q., Chanpaisit, N., et al.: Vbench: Comprehensive benchmark suite for video generative models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 21807–21818 (2024)
- 21. Jeong, H., Chang, J., Park, G.Y., Ye, J.C.: Dreammotion: Space-time self-similar score distillation for zero-shot video editing. In: European Conference on Computer Vision. pp. 358–376. Springer (2024)
- 22. Jeong, H., Lee, S., Ye, J.C.: Reangle-a-video: 4d video generation as video-to-video translation. arXiv preprint arXiv:2503.09151 (2025)
- 23. Jeong, H., Park, G.Y., Ye, J.C.: Vmc: Video motion customization using temporal attention adaption for text-to-video diffusion models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 9212– 9221 (2024)
- 24. Jeong, H., Ye, J.C.: Ground-a-video: Zero-shot grounded video editing using textto-image diffusion models. arXiv preprint arXiv:2310.01107 (2023)
- 25. Jiang, J., Li, W., Ren, J., Qiu, Y., Guo, Y., Xu, X., Wu, H., Zuo, W.: Lovic: Efficient long video generation with context compression. arXiv preprint arXiv:2507.12952

(2025)

- 26. Jiang, Z., Han, Z., Mao, C., Zhang, J., Pan, Y., Liu, Y.: Vace: All-in-one video creation and editing. arXiv preprint arXiv:2503.07598 (2025)
- 27. Jin, H., Jiang, H., Tan, H., Zhang, K., Bi, S., Zhang, T., Luan, F., Snavely, N., Xu, Z.: Lvsm: A large view synthesis model with minimal 3d inductive bias. arXiv preprint arXiv:2410.17242 (2024)
- 28. Ju, X., Wang, T., Zhou, Y., Zhang, H., Liu, Q., Zhao, N., Zhang, Z., Li, Y., Cai, Y., Liu, S., et al.: Editverse: Unifying image and video editing and generation with in-context learning. arXiv preprint arXiv:2509.20360 (2025)
- 29. Kara, O., Kurtkaya, B., Yesiltepe, H., Rehg, J.M., Yanardag, P.: Rave: Randomized noise shuffling for fast and consistent video editing with diffusion models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (2024)
- 30. Kim, J., Kang, J., Choi, J., Han, B.: Fifo-diffusion: Generating infinite videos from text without training. Advances in Neural Information Processing Systems 37, 89834–89868 (2024)
- 31. Kingma, D.P., Welling, M.: Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114 (2013)
- 32. Li, R., Torr, P., Vedaldi, A., Jakab, T.: Vmem: Consistent interactive video szicene generation with surfel-indexed view memory. arXiv preprint arXiv:2506.18903

(2025)

- 33. Liu, X., Gong, C., Liu, Q.: Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003 (2022)

- 34. Meng, Y., Zhu, Z., Hui, L., Hou, J.: Nvs-solver: Video diffusion model as zero-shot novel view synthesizer. In: The Thirteenth International Conference on Learning Representations
- 35. Nan, K., Xie, R., Zhou, P., Fan, T., Yang, Z., Chen, Z., Li, X., Yang, J., Tai, Y.: Openvid-1m: A large-scale high-quality dataset for text-to-video generation. arXiv preprint arXiv:2407.02371 (2024)
- 36. Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., et al.: Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193 (2023)
- 37. Ouyang, W., Xiao, Z., Yang, D., Zhou, Y., Yang, S., Yang, L., Si, J., Pan, X.: Tokensgen: Harnessing condensed tokens for long video generation. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 18197–18206

(2025)

- 38. Po, R., Nitzan, Y., Zhang, R., Chen, B., Dao, T., Shechtman, E., Wetzstein, G., Huang, X.: Long-context state-space video world models. arXiv preprint arXiv:2505.20171 (2025)
- 39. Polyak, A., Zohar, A., Brown, A., Tjandra, A., Sinha, A., Lee, A., Vyas, A., Shi, B., Ma, C.Y., Chuang, C.Y., et al.: Movie gen: A cast of media foundation models. arXiv preprint arXiv:2410.13720 (2024)
- 40. Qin, B., Li, J., Tang, S., Chua, T.S., Zhuang, Y.: Instructvid2vid: Controllable video editing with natural language instructions. In: 2024 IEEE International Conference on Multimedia and Expo (ICME). pp. 1–6. IEEE (2024)
- 41. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from natural language supervision. In: International conference on machine learning. pp. 8748–8763. PmLR (2021)
- 42. Ren, X., Shen, T., Huang, J., Ling, H., Lu, Y., Nimier-David, M., Müller, T., Keller, A., Fidler, S., Gao, J.: Gen3c: 3d-informed world-consistent video generation with precise camera control. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 6121–6132 (2025)
- 43. Ruhe, D., Heek, J., Salimans, T., Hoogeboom, E.: Rolling diffusion models, 2024. URL https://arxiv. org/abs/2402.09470
- 44. Team, H., Wang, Z., Liu, Y., Wu, J., Gu, Z., Wang, H., Zuo, X., Huang, T., Li, W., Zhang, S., et al.: Hunyuanworld 1.0: Generating immersive, explorable, and interactive 3d worlds from words or pixels. arXiv preprint arXiv:2507.21809 (2025)
- 45. Van Hoorick, B., Wu, R., Ozguroglu, E., Sargent, K., Liu, R., Tokmakov, P., Dave, A., Zheng, C., Vondrick, C.: Generative camera dolly: Extreme monocular dynamic novel view synthesis. In: European Conference on Computer Vision. pp. 313–331. Springer (2024)
- 46. Wang, Q., Zhang, Y., Holynski, A., Efros, A.A., Kanazawa, A.: Continuous 3d perception model with persistent state. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 10510–10522 (2025)
- 47. Wang, Z., Cho, J., Li, J., Lin, H., Yoon, J., Zhang, Y., Bansal, M.: Epic: Efficient video camera control learning with precise anchor-video guidance. arXiv preprint arXiv:2505.21876 (2025)
- 48. Wiedemer, T., Li, Y., Vicol, P., Gu, S.S., Matarese, N., Swersky, K., Kim, B., Jaini, P., Geirhos, R.: Video models are zero-shot learners and reasoners. arXiv preprint arXiv:2509.20328 (2025)
- 49. Wu, J.Z., Ge, Y., Wang, X., Lei, S.W., Gu, Y., Shi, Y., Hsu, W., Shan, Y., Qie, X., Shou, M.Z.: Tune-a-video: One-shot tuning of image diffusion models for text-

- to-video generation. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 7623–7633 (2023)
- 50. Xi, H., Yang, S., Zhao, Y., Xu, C., Li, M., Li, X., Lin, Y., Cai, H., Zhang, J., Li, D., et al.: Sparse videogen: Accelerating video diffusion transformers with spatialtemporal sparsity. arXiv preprint arXiv:2502.01776 (2025)
- 51. Xiao, Z., Lan, Y., Zhou, Y., Ouyang, W., Yang, S., Zeng, Y., Pan, X.: Worldmem: Long-term consistent world simulation with memory. arXiv preprint arXiv:2504.12369 (2025)
- 52. Yatim, D., Fridman, R., Bar-Tal, O., Kasten, Y., Dekel, T.: Space-time diffusion features for zero-shot text-driven motion transfer. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 8466– 8476 (2024)
- 53. Yin, T., Gharbi, M., Park, T., Zhang, R., Shechtman, E., Durand, F., Freeman, B.: Improved distribution matching distillation for fast image synthesis. Advances in neural information processing systems 37, 47455–47487 (2024)
- 54. Yin, T., Gharbi, M., Zhang, R., Shechtman, E., Durand, F., Freeman, W.T., Park, T.: One-step diffusion with distribution matching distillation. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 6613– 6623 (2024)
- 55. Yin, T., Zhang, Q., Zhang, R., Freeman, W.T., Durand, F., Shechtman, E., Huang, X.: From slow bidirectional to fast causal video generators. arXiv e-prints pp. arXiv–2412 (2024)
- 56. Yu, J., Bai, J., Qin, Y., Liu, Q., Wang, X., Wan, P., Zhang, D., Liu, X.: Context as memory: Scene-consistent interactive long video generation with memory retrieval. arXiv preprint arXiv:2506.03141 (2025)
- 57. YU, M., Hu, W., Xing, J., Shan, Y.: Trajectorycrafter: Redirecting camera trajectory for monocular videos via diffusion models. arXiv preprint arXiv:2503.05638

(2025)

- 58. Yu, S., Hahn, M., Kondratyuk, D., Shin, J., Gupta, A., Lezama, J., Essa, I., Ross, D., Huang, J.: Malt diffusion: Memory-augmented latent transformers for anylength video generation. arXiv preprint arXiv:2502.12632 (2025)
- 59. Yu, W., Xing, J., Yuan, L., Hu, W., Li, X., Huang, Z., Gao, X., Wong, T.T., Shan, Y., Tian, Y.: Viewcrafter: Taming video diffusion models for high-fidelity novel view synthesis. arXiv preprint arXiv:2409.02048 (2024)
- 60. Zhang, D.J., Paiss, R., Zada, S., Karnad, N., Jacobs, D.E., Pritch, Y., Mosseri, I., Shou, M.Z., Wadhwa, N., Ruiz, N.: Recapture: Generative video camera controls for user-provided videos using masked video fine-tuning. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 2050–2062 (2025)
- 61. Zhang, J., Xiang, C., Huang, H., Wei, J., Xi, H., Zhu, J., Chen, J.: Spargeattn: Accurate sparse attention accelerating any model inference. arXiv preprint arXiv:2502.18137 (2025)
- 62. Zhang, L., Agrawala, M.: Packing input frame context in next-frame prediction models for video generation. arXiv preprint arXiv:2504.12626 2(3), 5 (2025)
- 63. Zhang, P., Chen, Y., Huang, H., Lin, W., Liu, Z., Stoica, I., Xing, E., Zhang, H.: Vsa: Faster video diffusion with trainable sparse attention. arXiv preprint arXiv:2505.13389 (2025)
- 64. Zhang, P., Huang, H., Chen, Y., Lin, W., Liu, Z., Stoica, I., Xing, E.P., Zhang, H.: Faster video diffusion with trainable sparse attention. arXiv e-prints pp. arXiv– 2505 (2025)

- 65. Zhang, T., Bi, S., Hong, Y., Zhang, K., Luan, F., Yang, S., Sunkavalli, K., Freeman, W.T., Tan, H.: Test-time training done right. arXiv preprint arXiv:2505.23884

(2025)

- 66. Zheng, Z., Peng, X., Yang, T., Shen, C., Li, S., Liu, H., Zhou, Y., Li, T., You, Y.: Open-sora: Democratizing efficient video production for all (March 2024), https: //github.com/hpcaitech/Open-Sora
- 67. Zi, B., Ruan, P., Chen, M., Qi, X., Hao, S., Zhao, S., Huang, Y., Liang, B., Xiao, R., Wong, K.F.: Se\˜ norita-2m: A high-quality instruction-based dataset for general video editing by video specialists. arXiv preprint arXiv:2502.06734 (2025)

