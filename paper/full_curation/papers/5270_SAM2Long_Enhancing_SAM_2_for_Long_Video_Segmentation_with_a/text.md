# arXiv:2410.16268v3[cs.CV]29Jul2025

## SAM2Long: Enhancing SAM 2 for Long Video Segmentation with a Training-Free Memory Tree

### Shuangrui Ding1 Rui Qian1 Xiaoyi Dong1,2* Pan Zhang2

Yuhang Zang2 Yuhang Cao2 Yuwei Guo1 Dahua Lin1,2,3 Jiaqi Wang2* 1 The Chinese University of Hong Kong 2 Shanghai AI Laboratory 3 CPII under InnoHK

https://mark12ding.github.io/project/SAM2Long/

### Abstract

The Segment Anything Model 2 (SAM 2) has emerged as a powerful foundation model for object segmentation in both images and videos. The crucial design of SAM 2 for video segmentation is its memory module, which prompts object-aware memories from previous frames for current frame prediction. However, its greedy-selection memory design suffers from the “error accumulation” problem, where an errored or missed mask will cascade and influence the segmentation of the subsequent frames, which limits the performance of SAM 2 toward complex long-term videos. To this end, we introduce SAM2Long, an improved trainingfree video object segmentation strategy, which considers the segmentation uncertainty within each frame and chooses the video-level optimal results from multiple segmentation pathways in a constrained tree search manner. In practice, we maintain a fixed number of segmentation pathways throughout the video. For each frame, multiple masks are proposed based on the existing pathways, creating various candidate branches. We then select the same fixed number of branches with higher cumulative scores as the new pathways for the next frame. After processing the final frame, the pathway with the highest cumulative score is chosen as the final segmentation result. Benefiting from its heuristic search design, SAM2Long is robust toward occlusions and object reappearances, and can effectively segment and track objects for complex long-term videos. Without further training, SAM2Long significantly and consistently outperforms SAM 2 on nine VOS benchmarks and three VOT benchmarks. Notably, SAM2Long achieves an average improvement of 3.7 points across all 12 direct comparisons, with gains of up to 5.3 points in J &F on long-term video object segmentation benchmarks such as SA-V and LVOS. The code is released at https://github.com/Mark12Ding/SAM2Long.

*Corresponding Author.

### 1. Introduction

The Segment Anything Model 2 (SAM 2) has gained significant attention as a unified foundational model for promptable object segmentation in both images and videos. Notably, SAM 2 [50] has achieved state-of-the-art performance across various video object segmentation tasks, significantly surpassing previous methods. Building upon the original SAM [33], SAM 2 incorporates a memory module that enables it to generate masklet predictions using stored memory contexts from previously observed frames. This module allows SAM 2 to seamlessly extend SAM into the video domain, processing video frames sequentially, attending to the prior memories of the target object, and maintaining object coherence over time.

While SAM 2 demonstrates strong performance in video segmentation and downstream tasks [25], its greedy segmentation strategy struggles to handle complex video scenarios with frequent occlusions and object reappearance. In detail, SAM 2 confidently and accurately segments frames when clear visual cues are present. However, in scenarios with occlusions or reappearing objects, it can produce mask proposals that are highly variable and uncertain. Regardless of the frame’s complexity, a uniform greedy selection strategy is applied to both scenarios: the mask with the highest predicted IoU is selected. Such greedy choice works well for the easy cases but raises the error potential for the challenging frames. Once an incorrect mask is selected into memory, it is uncorrectable and will mislead the segmentation of the subsequent frames. Figure 1 illustrates the problem of “error accumulation” both qualitatively and quantitatively. The performance of SAM 2 gradually declines as the propagation progresses into later temporal segments, underscoring its limitations in keeping accurate tracking over time.

To this end, we redesign the memory module of SAM 2 to enhance its long-term capability and robustness against occlusions and error propagation. Our approach is motivated by the observation that the SAM 2 mask decoder generates multiple diverse masks, accompanied by predicted IoU

[Figure 1]

[Figure 2]

[Figure 3]

LVOS val SA-V test SA-V val

Time Flow

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

Performance(J&FScore)

SAM2’s Prediction

Wrong Tracking Error Accumulates ⚠ Occlusion Occurs SAM2Long’s Prediction Occlusion Solved

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

Object Reappears

[Figure 16]

[Figure 17]

[Figure 18]

Elapsed Time (seconds)

(a) Comparison in handling object occlusion over time.

(b) Per-frame performance comparison across three benchmarks.

- Figure 1. Comparison of occlusion handling and long-term capability between SAM 2 and SAM2Long. (a) When an occlusion occurs, SAM 2 may lose track or follow the wrong object, leading to accumulated errors. In contrast, SAM2Long utilizes memory tree search to recover when the object reappears. (b) Per-frame J &F scores of the predicted masks with different timestamp are plotted on the LVOS and SA-V datasets. SAM2Long demonstrates greater resilience to long video compared to SAM 2.

scores and an occlusion score when handling challenging and ambiguous cases. However, SAM 2 only selects a single mask as memory, sometimes disregarding the correct one. To address this limitation, we propose integrating SAM 2 with multiple memory pathways, inspired by prior Multiple Hypothesis Tracking (MHT) techniques [15, 32, 51]. These techniques, widely adopted in the tracking community, enable the storage of various masks as memory at each time step and delay the association decision to obtain video-level optimal result. In particular, we maintain a fixed number of memory pathways over time to explore multiple segmentation hypotheses with efficiently managed computational resources. At each time step, based on a set of memory pathways, each with its own memory bank and cumulative score (accumulated logarithm of the predicted IoU scores across the pathway), we produce multiple candidate branches for the current frame. Then, among all the branches, we only keep the same number of branches with higher cumulative scores and prune other branches, thereby constraining the tree’s growth. After processing the final frame, the pathway with the highest cumulative score is selected as the final segmentation result. Moreover, to prevent premature convergence on incorrect predictions, we select pathways with distinct predicted masks when their occlusion scores indicate uncertainty. In this way, we maintain diversity in the tree branches in the challenging and distraction scenario. This tree-like memory structure augments SAM 2’s ability to effectively overcome error accumulation.

segmented. This ensures that the memory bank provides effective object cues for the current frame’s segmentation. Additionally, we modulate the memory attention calculation by weighting memory entries according to their occlusion scores, emphasizing more reliable entries during crossattention. These strategies help SAM 2 focus on reliable object clues and improve segmentation accuracy with negligible computational overhead. As evidenced in Figure 1(a), our approach successfully resolves occlusions and re-tracks the recurring dancers, where SAM 2 fails.

Our improvement is completely free of additional training and does not introduce any external parameters, but simply unleashes the potential of SAM 2 itself. We provide a comprehensive evaluation that SAM2Long consistently outperforms SAM 2 across nine VOS benchmarks and three VOT benchmarks, particularly excelling in long-term and occlusion-heavy scenarios. Moreover, datasets with longer video durations generally lead to greater performance gains, which aligns with the motivation of our design. For instance, on the challenging SA-V test set, SAM2Long-L improves the J &F score by 5.3 points, and SAM2Long-S shows an impressive 4.7-point gain over the same size SAM 2 model on SA-V val set. Similar trends are observed on the LVOS validation set, where SAM2Long-S surpasses SAM 2-S by 3.5 points. These consistent improvements across different model sizes, including both SAM 2 and the more recent SAM 2.1 model weights, clearly demonstrate the generalization ability of our proposed method. Furthermore, as shown in Figures 1(b), the per-frame performance gap between SAM2Long and SAM 2 increases over time. This highlights SAM2Long’s strength in long-term tracking scenarios. With these results, we believe SAM2Long sets a new standard for video object segmentation based on SAM 2 in complex, real-world applications.

Within each pathway, we construct an object-aware memory bank that selectively includes frames with confidently detected objects and high-quality segmentation masks, based on the predicted occlusion scores and IoU scores. Instead of simply storing the nearest frames as SAM 2 does, we filter out frames where the object may be occluded or poorly

### 2. Related work

#### 2.1. Video Object Segmentation

Perceiving the environment in terms of objects is a fundamental cognitive ability of humans. In computer vision, Video Object Segmentation (VOS) tasks aim to replicate this capability by requiring models to segment and track specified objects within video sequences. A substantial amount of research has been conducted on video object segmentation in recent decades [3, 18, 19, 21, 28, 29, 31, 34– 36, 43, 44, 46, 48, 49, 55, 58, 59, 61, 70, 71]. There are two main protocols for evaluating VOS models [45, 47]: semi-supervised and unsupervised video object segmentation. In semi-supervised VOS, the first-frame mask of the objects of interest is provided, and the model tracks these objects in subsequent frames. In unsupervised VOS, the model directly segments the most salient objects from the background without any reference. It is important to note that these protocols are defined in the inference phase, and VOS methods can leverage ground truth annotations during the training stage. In this paper, we explore SAM 2 [50], for its application in semi-supervised VOS. We enhance the memory design of SAM 2, significantly improving mask propagation performance without any additional training.

#### 2.2. Memory-Based VOS

Video object segmentation remains an unsolved challenge due to the inherent complexity of video scenes. Objects in videos can undergo deformation [54], exhibit dynamic motion [6], reappear over long durations [26, 27], and experience occlusion [17], among other challenges. To address the above challenges, adopting a memory architecture to store the object information from past frames is indispensable for accurately tracking objects in video [5, 7, 10, 29, 40, 52, 56, 57, 64, 66, 68]. Recent approaches have introduced efficient memory reading mechanisms, utilizing either pixel-level attention [11, 13, 20, 37, 43, 53, 62, 65, 67, 73] or objectlevel attention [1, 2, 14]. Among these, Multiple Hypothesis Tracking (MHT) [15, 32, 51] stands out as a robust paradigm that maintains tree search to construct and manage memory. A prominent example is XMem [11], which leverages a hierarchical memory structure for pixel-level memory reading combined. Building on XMem’s framework, Cutie [14] further improves segmentation accuracy by processing pixel features at the object level to better handle complex scenarios. The latest SAM 2 [50] incorporates a simple memory module on top of the image-based SAM [33], enabling it to function for VOS tasks. However, SAM 2 struggles with challenging cases involving long-term reappearing objects and confusingly similar objects. Motivated by MHT, we redesign SAM 2’s memory to maintain multiple potential correct masks, making the model more object-aware and robust.

### 3. Method

#### 3.1. Preliminary on SAM 2

SAM 2 [50] begins with an image encoder that encodes each input frame into embeddings. In contrast to SAM, where frame embeddings are fed directly into the mask decoder, SAM 2 incorporates a memory module that conditions the current frame’s features on both previous and prompted frames. Specifically, for semi-supervised video object segmentation tasks, SAM 2 maintains a memory bank at each time step t ≥ 1:

Mt = Mτ ∈ RK×C τ∈I ,

where K is the number of memory tokens per frame, C is the channel dimension, and I is the set of frame indices included in the memory. In SAM 2, memory set I stores up to N of the most recent frames, along with the initial mask, using a First-In-First-Out (FIFO) queue mechanism.

Each memory entry consists of two components: (1) the spatial embedding fused with the predicted mask (generated by the memory encoder), and (2) the object-level pointer (generated by the mask decoder). After cross-attending to the memory, the current frame’s features integrate both finegrained correspondences and object-level semantic information.1

The mask decoder, which is lightweight and retains the efficiency of SAM, then generates three predicted masks for the current frame. Each mask is accompanied by a predicted Intersection over Union (IoU) score IoUt ≥ 0 and an output mask token. Additionally, the mask decoder predicts a single occlusion score ot for the frame, where ot > 0 indicates object presence, ot < 0 indicates absence, and the absolute value |ot| depicts the model’s confidence. The mask with the highest predicted IoU score is selected as the final prediction, and its corresponding output token is transformed into the object pointer for use as the memory.

#### 3.2. Constrained Tree Memory with Uncertainty Handling

To enhance SAM 2’s robustness towards long-term and ambiguous cases, we propose a constrained tree memory structure that enables the model to explore various object states over time with minimal computational overhead. We show the high-level pipeline in Figure 2. Note that we store memory separately for each target. Each target maintains its own individual tree memory bank. This tree-based approach maintains multiple plausible pathways and mitigates the effects of erroneous predictions.

Specifically, at each time step t, we maintain a set of P memory pathways, each with a memory bank Mpt and a

1In practice, SAM 2 stores more object pointers than spatial embeddings, as pointers are lighter. We assume equal numbers of both components solely for illustrative purposes, without altering the actual implementation.

[Figure 19]

If Certain

- 𝑆 ,  𝑡 = 1.4

𝑆 ,  𝑡 = 1.6

- 𝑆 ,  𝑡 = 1.5

𝑆 ,  𝑡 + 1 = 2.0

𝑆 𝑡 − 1 = 0.9 𝑆 𝑡 = 1.6 𝑆 𝑡 + 1 = 2.4

[Figure 20]

- Memory Bank 1

Frame t

- Memory Bank 2

Memory Bank 1

Memory Bank 1

[Figure 21]

[Figure 22]

Occlusion Score 𝑜 = 9 > 𝛿 𝑜 = 8 > 𝛿

[Figure 23]

[Figure 24]

[Figure 25]

Cumulative IoU score 𝑆 , 

1.5 1.6 1.5 1.6 1.4 1.5

- 𝑆 ,  𝑡 = 1.5

𝑆 ,  𝑡 = 1.3

- 𝑆 ,  𝑡 = 1.6

𝑆 ,  𝑡 + 1 = 2.1

[Figure 26]

⋮ ⋮

⋮

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

| |
|---|

[Figure 35]

[Figure 36]

Mask Decoder

Mask Decoder

| |
|---|

| |
|---|

⋯

Memory update

Mask candidates

- 𝑆 ,  𝑡 + 1 = 2.1

- 𝑆 ,  𝑡 + 1 = 2.2

𝑆 ,  𝑡 + 1 = 2.4

- 𝑆 ,  𝑡 + 1 = 2.3

[Figure 37]

[Figure 38]

##### Memory update

High-scoring Mask

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

If Uncertain

[Figure 43]

###### Mask Selection

Mask Selection

Frame t + 1

𝑜 = 1 < 𝛿 𝑜 = 0.6 < 𝛿

Occlusion Score

[Figure 44]

Cumulative IoU score 𝑆 ,  1.4 1.7 1.6 1.5 1.7 1.5

| |
|---|

[Figure 45]

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

[Figure 56]

⋮

⋮

⋮

[Figure 57]

Mask candidates

| |
|---|

[Figure 58]

[Figure 59]

Mask Decoder

Mask Decoder

Memory update

Memory update

⋯

| |
|---|

| |
|---|

[Figure 60]

[Figure 61]

Memory Bank 2

Memory Bank 2

[Figure 62]

[Figure 63]

Distinct-shaped Mask

𝑆 𝑡 − 1 = 0.9

𝑆 𝑡 = 1.6 𝑆 𝑡 + 1 = 2.3

(a) The Pipeline of Constrained Memory Tree

(b) Mask Selection with Uncertainty Handling

- Figure 2. (a) The pipeline of constrained memory tree: At each time step t, we maintain multiple memory pathways, each containing a memory bank and a cumulative score Sp[t]. The input frame is processed through the mask decoder conditioned on the memory bank, generating three mask candidates for each pathway. The candidates with the highest updated cumulative scores Sp,k[t] are carried forward to the next time step. (b) Mask selection with uncertainty handling: When the maximum absolute occlusion score exceeds the threshold δconf (Certain), the high-scoring mask is selected. Otherwise (Uncertain), distinct mask candidates are picked to avoid incorrect convergence.

cumulative score Sp[t], representing a possible segmentation hypothesis up to frame t. Conditioned on the memory bank of each pathway p, the SAM 2 decoder head generates three mask candidates along with their predicted IoU scores, denoted as IoUp,t 1, IoUp,t 2, and IoUp,t 3. This process expands the tree by branching each existing pathway into three new candidates. As a result, there are a total of 3P possible pathways at each time step. We then calculate the cumulative scores for each possible pathway by adding the logarithm of its IoU score to the pathway’s previous score:

SAM 2-Large is 224M. Since we process the image encoder only once just as SAM 2 does, the introduction of a memory tree adds negligible computational cost while significantly enhancing SAM 2’s robustness against error-prone cases.

Uncertainty Handling. Unfortunately, there are times when all pathways are uncertain. To prevent the model from improperly converging on incorrect predictions, we implement a strategy to maintain diversity among the pathways by deliberately selecting distinct masks. That is, if the maximum absolute occlusion score across all pathways at time t, max({|opt|}Pp=1), is less than a predefined uncertainty threshold δconf, we enforce the model to select mask candidates with unique IoU values. This design is inspired by the observation that, within the same frame, different IoU scores typically correspond to distinct masks, which we verify in Table 8. In practice, we round each IoU score IoUp,kt to two decimal places and only select those hypotheses with distinct rounded values.

Sp,k[t] = Sp[t − 1] + log(IoUp,kt + ϵ), for k = 1,2,3, where ϵ is a small constant to prevent the logarithm of zero.

However, continuously tripling the pathways would lead to unacceptable computational and memory costs. Therefore, to manage computational complexity and memory usage, we implement a pruning strategy that selects the top P pathways with the highest cumulative scores to carry forward to the next time step. This selection not only retains the most promising segmentation hypotheses but also constrains the tree-based memory, ensuring computational efficiency. Finally, we output the segmentation pathway with the highest cumulative score as the ultimate result.

Overall, the integration of constrained tree memory with uncertainty handling offers a balanced strategy that leverages multiple segmentation hypotheses to enhance robustness toward the long-term complex video and achieve more accurate and reliable segmentation performance by effectively mitigating error accumulation.

Compared to SAM 2, our approach introduces additional computation mainly by increasing the number of passes through the mask decoder and memory module. Notably, these components are lightweight relative to the image encoder. For instance, the image encoder of SAM 2-Large consists of 212M parameters while the total parameter of

#### 3.3. Object-aware Memory Bank Construction

In each memory pathway, we devise object-aware memory selection to retrieve frames with discriminative objects. Meanwhile, we modulate the memory attention calculation

SA-V val SA-V test LVOS v2 val

Method

J &F J F J &F J F J &F J F

SAM2-T† 73.5 70.1 76.9 74.6 71.1 78.0 77.8 74.5 81.2 SAM2Long-T 77.0 (3.5↑) 73.2 80.7 78.7 (4.1↑) 74.6 82.7 81.4 (3.6↑) 77.7 85.0 SAM2.1-T† 75.1 71.6 78.6 76.3 72.7 79.8 81.6 77.9 85.2 SAM2.1Long-T 78.9 (3.8↑) 75.2 82.7 79.0 (2.7↑) 75.2 82.9 82.4 (0.8↑) 78.8 85.9 SAM2-S† 73.0 69.7 76.3 74.6 71.0 78.1 79.7 76.2 83.3 SAM2Long-S 77.7 (4.7↑) 73.9 81.5 78.1 (3.5↑) 74.1 82.0 83.2 (3.5↑) 79.5 86.8 SAM2.1-S† 76.9 73.5 80.3 76.9 73.3 80.5 82.1 78.6 85.6 SAM2.1Long-S 79.6 (2.7↑) 75.9 83.3 80.4 (3.5↑) 76.6 84.1 84.3 (2.2↑) 80.7 88.0 SAM2-B+† 75.4 71.9 78.8 74.6 71.2 78.1 80.2 76.8 83.6 SAM2Long-B+ 78.4 (3.0↑) 74.7 82.1 78.5 (3.9↑) 74.7 82.2 82.3 (2.1↑) 78.8 85.9 SAM2.1-B+† 78.0 74.6 81.5 77.7 74.2 81.2 83.1 79.6 86.5 SAM2.1Long-B+ 80.5 (2.5↑) 76.8 84.2 80.8 (3.1↑) 77.1 84.5 85.2 (2.1↑) 81.5 88.9 SAM2-L† 76.3 73.0 79.5 75.5 72.2 78.9 83.0 79.6 86.4 SAM2Long-L 80.8 (4.5↑) 77.1 84.5 80.8 (5.3↑) 76.8 84.7 85.4 (2.4↑) 81.8 88.7 SAM2.1-L† 78.6 75.1 82.0 79.6 76.1 83.2 84.0 80.7 87.4 SAM2.1Long-L 81.1 (2.5↑) 77.5 84.7 81.2 (1.6↑) 77.6 84.9 85.3 (1.3↑) 81.9 88.8

Table 1. Performance comparison on SA-V [50] and LVOS v2 [27] datasets between SAM 2 and SAM2Long across all model sizes. † We report the re-produced performance of SAM 2(2.1) using its open-source code and checkpoint.

to further strengthen the model’s focus on the target objects. Memory Frame Selection. To construct a memory bank that provides effective object cues, we selectively choose frames from previous time steps based on the predicted object presence and segmentation quality. Starting from the frame immediately before the current frame t, we iterate backward through the prior frames i = {t − 1,t − 2,...,1} in sequence. For each frame i, we retrieve its predicted occlusion score oi and IoU score IoUi as reference. We include frame i in the memory bank if it satisfies the following criteria:

IoUi > δIoU and oi > 0, where δIoU is a predefined IoU threshold. This ensures that only frames with confidently detected objects and reasonable segmentation masks contribute to the memory. We continue this process until we have selected up to N frames. In contrast to SAM 2, which directly picks the nearest N frames as the memory entries, this selection process effectively filters out frames where the object may be occluded, absent, or poorly segmented, thereby providing more robust object cues for the segmentation of the current frame.

Memory Attention Modulation. To further emphasize more reliable memory entries during the cross-attention computation, we utilize the associated occlusion score ot to modulate the contribution of each memory entry. Assuming the memory set consists of N frames plus the initial frame, totaling N + 1 masks, we define a set of standard weights Wstd that are linearly spaced between a lower bound wlow and an upper bound whigh:

i − 1 N

Wstd = wlow +

(whigh − wlow)

N+1

.

i=1

Next, we sort the occlusion scores in ascending order to obtain sorted indices I′ = {Ii}Ni=1+1 such that:

1 ≤ oI

2 ≤ ··· ≤ oI

oI

.

N+1

We then assign the standard weights to the memory entries based on these sorted indices:

= Wistd, for i = 1,2,...,N + 1.

wI

i

This assignment ensures that memory entries with higher occlusion scores, which indicate object presence with higher confidence, receive higher weights. Then, we linearly scale the original keys Mτ with their corresponding weights:

Mτ = wτ · Mτ, for τ ∈ I.

Finally, the modulated memory keys Mt = { Mτ}τ∈I are used in the memory module’s cross-attention mechanism to update the current frame’s features. By using the available occlusion scores as indicators, we effectively emphasize memory entries with more reliable object cues.

### 4. Experiments

#### 4.1. Experiments Setup

Datasets. To evaluate our method, we select nine standard VOS benchmarks: SA-V val and test [50], LVOS v1 [26], LVOS v2 [27], MOSE [17], VOST [54], PUMaVOS [4], DAVIS [47], and YTVOS [63]. We report the following metrics: J (region similarity), F (contour accuracy), and the combined J &F. All evaluations are conducted in a semisupervised setting, where the first-frame mask is provided. Details on the benchmarks are provided in the Appendix.

Evaluation. SAM2Long is modified from the official codebase of SAM2 and we report the re-produced performance of SAM2 with the same official code and checkpoint. We noticed a slight difference between the re-produced results and the performance reported in the SAM2 paper, but for a strictly fair comparison, we tend to report both the SAM2 and SAM2Long results under the same hardware/software environment and settings. We set the uncertainty threshold to δconf = 2, IoU threshold to δIoU = 0.3, modulation weight to [wlow,whigh] = [0.95,1.05]. We find that the performance remains fairly robust across datasets when tuning these hyperparameters; thus, we use the same values to report results for all experiments. The detailed findings are provided in the Appendix for reference.

#### 4.2. Main Results

SAM2Long consistently improves SAM 2 over all model sizes and datasets. Table 1 presents an overall comparison between SAM 2 and SAM2Long across various model sizes on the SA-V validation and test sets, as well as the LVOS v2 validation set. In total, the table includes 8 model variations, covering SAM 2 and the latest SAM 2.1 across four model sizes. The average performance across 12 experiments shows an improvement of 3.7 points in the J &F score. These results confirm that SAM2Long consistently outperforms the SAM 2 baseline by a large margin. For instance, for SAM2Long-Large, achieves an improvement of 4.5 and 5.3 over SAM 2 on SA-V val and test sets.

SAM2Long outperforms previous methods. We also compare our proposed method, SAM2Long, with various state-of-the-art VOS methods on both the SA-V [50] and LVOS [26, 27] datasets, as shown in Table 2 and 3. Previous methods such as STCN[12] and XMem[11] adopt memory-based strategies to improve performance and efficiency. STCN models object relations using a single affinity matrix, whereas XMem maintains a three-tier hierarchical memory structure to balance accuracy and GPU usage. While both methods focus on learning feature-based memory representations, SAM2Long takes a different approach by refining memory at the mask level rather than the feature level. Our method efficiently selects the optimal sequence of memory masks using the proposed tree structure, eliminating the need to retrain the original memory features. As a result, SAM2Long surpasses XMem and STCN by over 20 points on both LVOS v2 and SA-V val, demonstrating its superior ability to handle long-term object segmentation.

Performance gains of SAM2Long positively correlate with video duration. The results shown in the Table 4 reveal a clear trend: SAM2Long tends to exhibit greater performance gains with longer videos. Although SAM2Long performs comparably to SAM 2 on shorter-duration datasets [47, 63], this aligns with the limited long-term tracking opportunities provided by short video sequences. However,

SAM2Long can still maintain the strong segmentation performance of SAM 2.

#### 4.3. Additional Results on VOT Benchmarks

To further evaluate the effectiveness of SAM2Long, we also apply it to three widely used single-object tracking benchmarks, namely LaSOT [22], LaSOText [23], and GOT10k [30]. Without whistles and bells, we first generate an initial mask from the bounding box provided for the first frame using the SAM2 Mask decoder. Subsequently, all subsequent masks are transformed into bounding box format by identifying the minimum and maximum coordinates of their non-zero indices. SAM2Long achieves competitive performance with state-of-the-art VOT methods and SAM 2 baseline, demonstrating its strong generalization ability across various tracking scenarios.

#### 4.4. Ablation Study

We perform a series of ablation studies on the validation subsets of the SA-V and LVOS v2 datasets, utilizing SAM2Large as the default model size.

Number of Memory Pathways P. We ablate the number of memory pathways to assess their impact on SAM2Long in Table 6. Note that setting P = 1 reverts to the SAM 2 baseline. Increasing the number of memory pathways to P = 2 yields a notable improvement, raising the J &F score to 80.1. This result demonstrates that the proposed memory tree effectively boosts the model’s ability to track the correct object while reducing the impact of occlusion. Further increasing the number of memory pathways to P = 3 achieves the best performance. However, using P = 4 shows no additional gains, suggesting that three pathways strike the optimal balance between accuracy and computational efficiency for the SAM 2 model.

In terms of speed, maintaining a fixed number of memory pathways at each time step ensures efficiency. Using three pathways incurs only a 14% FPS slowdown, an 8% increase in GFlops, and a 4% rise in memory usage, while boosting performance by 4.5 points on SA-V and 2.4 points on LVOS. Memory Frame Selection. Building on IoU-based memory frame selection, we incorporate two additional mechanisms: temporal and spatial selection. The temporal method applies a time decay factor to the IoU score, prioritizing frames closer in time to the current frame. The spatial method integrates frame-wise feature similarity into the IoU score, favoring frames with similar spatial contexts. However, as shown in Table 7, neither approach enhances performance, showing that IoU filtering remains the key factor.

Rounding Predicted IoU on Mask Diversity. To evaluate the impact of rounding predicted IoU in selecting diverse masks, we compute the actual IoU between candidate masks chosen based on rounded predicted IoU. As shown in Table 8, rounding to two decimal places significantly reduces the ac-

SA-V val SA-V test

Method

J &F J F J &F J F

SwinB-DeAOT [65] 61.4 56.6 66.2 61.8 57.2 66.3 DEVA [13] 55.4 51.5 59.2 56.2 52.4 60.1 Cutie-base+ [14] 61.3 58.3 64.4 62.8 59.8 65.8 STCN [12] 61.0 57.4 64.5 62.5 59.0 66.0 XMem [11] 60.1 56.3 63.9 62.3 58.9 65.8

- SAM 2 [50] 76.1 72.9 79.2 76.0 72.6 79.3

- SAM2Long (ours) 80.8 74.7 84.7 80.8 76.8 84.7 SAM 2.1† [50] 78.6 75.1 82.0 79.6 76.1 83.2 SAM2.1Long (ours) 81.1 77.5 84.7 81.2 77.6 84.9

Table 2. Performance comparison with the-state-of-the-arts methods on SA-V dataset.

Method

LVOS v1 LVOS v2

J &F J F J &F Js Fs Ju Fu LWL [5] 56.4 51.8 60.9 60.6 58.0 64.3 57.2 62.9 RDE [34] 53.7 48.3 59.2 62.2 56.7 64.1 60.8 67.2 STCN [12] 48.9 43.9 54.0 60.6 57.2 64.0 57.5 63.8 XMem [11] 52.9 48.1 57.7 64.5 62.6 69.1 60.6 65.6 DDMemory [26] 60.7 55.0 66.3 - - - - SAM 2 [50] 77.9 73.1 82.7 79.8 80.0 86.6 71.6 81.1

- SAM2Long (ours) 81.3 76.4 86.2 84.2 82.3 89.2 79.1 86.2 SAM 2.1† [50] 80.2 75.4 84.9 84.1 80.7 87.4 80.6 87.7 SAM2.1Long (ours) 83.4 78.4 88.5 85.9 81.7 88.6 83.0 90.5

- Table 3. Performance comparison with state-of-the-art methods on validation set of LVOS dataset. Subscript s and u denote scores in seen and unseen categories. Unlike the results presented in Table 1, we use the evaluation code from the LVOS official repository.

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

GTSAM2LongFailureCaseSAM2

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

[Figure 91]

- Figure 3. Qualitative comparison between SAM 2 and SAM2Long, with GT (Ground Truth) provided for reference. The last row shows a failure case. Best viewed when zoomed in.

Duration SAM 2.1† SAM2.1Long

Dataset

seconds J &F J &F

LVOS v1 [26] 95.4 80.2 83.4 (3.2↑) LVOS v2 [27] 68.4 84.1 85.9 (1.8↑) SA-V val [50] 13.8 78.6 81.1 (2.5↑) SA-V test [50] 13.8 79.6 81.2 (1.6↑) PUMaVOS [4] 28.7 81.1 82.4 (1.2↑) VOST [54] 21 53.1 54.0 (0.9↑) MOSE [17] 12.4 74.5 75.2 (0.7↑) YTVOS [63] 4.5 88.7 88.8 (0.1↑) DAVIS-17 [47] 1.8 90.1 90.2 (0.1↑)

- Table 4. The performance comparisons between SAM 2 and SAM2Long on other VOS benchmarks. We list the average duration for each benchmark. All experiments use SAM2.1-L model.

Dataset LaSoT LaSoText GoT10k

KeepTrack [41] 67.1 48.2 TOMP [42] 68.5 - DropTrack [60] 71.8 52.7 75.9 SeqTrack [9] 72.5 50.7 74.8 MixFormer [16] 70.1 - 71.2 GRM-256 [24] 69.9 - 73.4 ROMTrack [8] 71.4 51.3 74.2 OSTrack [69] 71.1 50.5 73.7 DiffusionTrack [39] 72.3 - 74.7 ODTrack [72] 74.0 53.9 78.2 LORAT [38] 75.1 56.6 78.2

SAM 2 [50] 70.0 56.9 80.7 SAM2Long (ours) 73.9 60.9 81.1

- Table 5. Performance comparison with state-of-the-art VOT methods on LaSoT, LaSoText, and GOT-10k benchmarks. The results are reported in terms of AUC (Area Under the success rate Curve) for LaSoT and LaSoText, and AO (Average Overlap) for GOT-10k.

P SA-V LVOS FPS GFlops Memory

- 1 76.3 83.0 22 844.1 5.1GB
- 2 80.1 85.0 21 878.2 5.2GB

- 3 80.8 85.4 19 912.3 5.3GB

- 4 80.7 85.2 17 946.4 5.4GB

- Table 6. Ablation study on number of pathways P. We report J &F performance along with FPS, computational cost (GFlops), and memory usage. Throughput and memory consumption are measured on a RTX 3090 GPU with 24 GB of memory.

tual IoU, highlighting increased variation among candidates. This enhanced diversity benefits our tree search strategy and improves performance in handling uncertain cases.

#### 4.5. Qualitative Comparison

We present a qualitative comparison between SAM 2 and SAM2Long in Figure 3. As illustrated in Figure 3, SAM 2

Memory Selection SA-V LVOS Only by IoU 80.8 85.4 w. temporal 80.4 85.2 w. spatial 80.7 84.8

Table 7. Ablation study on memory frame selection methods. We compare the standard IoU-based selection with additional temporal and spatial selection mechanisms.

Rounding SA-V LVOS actual pairwise IoU

✓ 80.8 85.4 51.4 ✕ 80.4 84.4 84.5

Table 8. Ablation study on rounding predicted IoU.

successfully tracks the correct person in a group of dancing people. However, when occlusion occurs, SAM 2 mistakenly switches to tracking a different, misleading individual. In contrast, SAM2Long handles this ambiguity effectively. Even during the occlusion, SAM2Long manages to resist the tracking error and correctly resumes tracking the original dancer when she reappears. We also present a failure case. When the video features dynamic background changes and distracting elements, SAM2Long struggles to maintain accurate tracking. The model either mistakenly tracks the wrong shirt or completely loses the target. We attribute this to SAM 2’s over-reliance on fine-grained visual details and its lack of semantic understanding.

### 5. Conclusion

In this paper, we introduce SAM2Long, a training-free enhancement to SAM 2 that alleviates its limitations in longterm video object segmentation. By employing a constrained tree memory structure with object-aware memory modulation, SAM2Long effectively mitigates error accumulation and improves robustness against occlusions, resulting in a more reliable segmentation process over extended periods. Extensive evaluations on six VOS benchmarks demonstrate that SAM2Long consistently outperforms SAM 2, especially in complex video scenarios. Notably, SAM2Long achieves up to a 5-point improvement in J &F scores on challenging long-term video benchmarks such SA-V and LVOS.

Limitations. While SAM2Long is effective, it has a few limitations. First, its performance is constrained by the inherent capacity of SAM2, as no learnable parameters are modified, restricting its optimization potential. Second, the method primarily focuses on single-object tracking rather than carefully designed multi-object scenarios. Researching specific multi-object techniques will be a promising future direction. However, our experiments demonstrate that SAM2Long already performs exceptionally well in both single-object and multi-object cases. We list the comparison in the Appendix.

### Acknowledgemnts

This work was supported by National Key R&D Program of China 2022ZD0161600, Shanghai Artificial Intelligence Laboratory, Hong Kong RGC TRS T41-603/20-R, the Centre for Perceptual and Interactive Intelligence (CPII) Ltd under the Innovation and Technology Commission (ITC)’s InnoHK. Dahua Lin is a PI of CPII under the InnoHK.

### References

- [1] Ali Athar, Jonathon Luiten, Alexander Hermans, Deva Ramanan, and Bastian Leibe. Hodor: High-level object descriptors for object re-segmentation in video learned from static images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3022–3031,

2022. 3

- [2] Ali Athar, Alexander Hermans, Jonathon Luiten, Deva Ramanan, and Bastian Leibe. Tarvis: A unified approach for target-based video segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18738–18748, 2023. 3
- [3] Linchao Bao, Baoyuan Wu, and Wei Liu. Cnn in mrf: Video object segmentation via inference in a cnn-based higher-order spatio-temporal mrf. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 5977– 5986, 2018. 3
- [4] Maksym Bekuzarov, Ariana Bermudez, Joon-Young Lee, and Hao Li. Xmem++: Production-level video segmentation from few annotated frames. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 635– 644, 2023. 5, 8, 12
- [5] Goutam Bhat, Felix J¨aremo Lawin, Martin Danelljan, Andreas Robinson, Michael Felsberg, Luc Van Gool, and Radu Timofte. Learning what to learn for video object segmentation. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part II 16, pages 777–794. Springer, 2020. 3, 7
- [6] Thomas Brox and Jitendra Malik. Object segmentation by long term analysis of point trajectories. In European conference on computer vision, pages 282–295. Springer, 2010. 3
- [7] Sergi Caelles, Kevis-Kokitsi Maninis, Jordi Pont-Tuset, Laura Leal-Taix´e, Daniel Cremers, and Luc Van Gool. One-shot video object segmentation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 221–230, 2017. 3
- [8] Yidong Cai, Jie Liu, Jie Tang, and Gangshan Wu. Robust object modeling for visual tracking. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 9589–9600, 2023. 8
- [9] Xin Chen, Houwen Peng, Dong Wang, Huchuan Lu, and Han Hu. Seqtrack: Sequence to sequence learning for visual object tracking. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 14572– 14581, 2023. 8
- [10] Yuhua Chen, Jordi Pont-Tuset, Alberto Montes, and Luc Van Gool. Blazingly fast video object segmentation with

- pixel-wise metric learning. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1189–1198, 2018. 3
- [11] Ho Kei Cheng and Alexander G Schwing. Xmem: Long-term video object segmentation with an atkinson-shiffrin memory model. In European Conference on Computer Vision, pages 640–658. Springer, 2022. 3, 6, 7
- [12] Ho Kei Cheng, Yu-Wing Tai, and Chi-Keung Tang. Rethinking space-time networks with improved memory coverage for efficient video object segmentation. Advances in Neural Information Processing Systems, 34:11781–11794, 2021. 6, 7
- [13] Ho Kei Cheng, Seoung Wug Oh, Brian Price, Alexander Schwing, and Joon-Young Lee. Tracking anything with decoupled video segmentation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 1316– 1326, 2023. 3, 7
- [14] Ho Kei Cheng, Seoung Wug Oh, Brian Price, Joon-Young Lee, and Alexander Schwing. Putting the object back into video object segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3151–3161, 2024. 3, 7
- [15] Ingemar J. Cox and Sunita L. Hingorani. An efficient implementation of reid’s multiple hypothesis tracking algorithm and its evaluation for the purpose of visual tracking. IEEE Transactions on pattern analysis and machine intelligence, 18(2):138–150, 1996. 2, 3
- [16] Yutao Cui, Cheng Jiang, Limin Wang, and Gangshan Wu. Mixformer: End-to-end tracking with iterative mixed attention. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 13608–13618,

2022. 8

- [17] Henghui Ding, Chang Liu, Shuting He, Xudong Jiang, Philip HS Torr, and Song Bai. MOSE: A new dataset for video object segmentation in complex scenes. In ICCV, 2023. 3, 5, 8, 12
- [18] Shuangrui Ding, Weidi Xie, Yabo Chen, Rui Qian, Xiaopeng Zhang, Hongkai Xiong, and Qi Tian. Motion-inductive self-supervised object discovery in videos. arXiv preprint arXiv:2210.00221, 2022. 3
- [19] Shuangrui Ding, Rui Qian, Haohang Xu, Dahua Lin, and Hongkai Xiong. Betrayed by attention: A simple yet effective approach for self-supervised video object segmentation. arXiv preprint arXiv:2311.17893, 2023. 3
- [20] Brendan Duke, Abdalla Ahmed, Christian Wolf, Parham Aarabi, and Graham W Taylor. Sstvos: Sparse spatiotemporal transformers for video object segmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5912–5921, 2021. 3
- [21] Deng-Ping Fan, Wenguan Wang, Ming-Ming Cheng, and Jianbing Shen. Shifting more attention to video salient object detection. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8554–8564,

2019. 3

- [22] Heng Fan, Liting Lin, Fan Yang, Peng Chu, Ge Deng, Sijia Yu, Hexin Bai, Yong Xu, Chunyuan Liao, and Haibin Ling. Lasot: A high-quality benchmark for large-scale single object tracking. In Proceedings of the IEEE/CVF conference on

computer vision and pattern recognition, pages 5374–5383,

2019. 6, 12

- [23] Heng Fan, Hexin Bai, Liting Lin, Fan Yang, Peng Chu, Ge Deng, Sijia Yu, Harshit, Mingzhen Huang, Juehuan Liu, et al. Lasot: A high-quality large-scale single object tracking benchmark. International Journal of Computer Vision, 129:439– 461, 2021. 6, 12
- [24] Shenyuan Gao, Chunluan Zhou, and Jun Zhang. Generalized relation modeling for transformer tracking. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18686–18695, 2023. 8
- [25] Yuwei Guo, Ceyuan Yang, Anyi Rao, Chenlin Meng, Omer Bar-Tal, Shuangrui Ding, Maneesh Agrawala, Dahua Lin, and Bo Dai. Keyframe-guided creative video inpainting. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 13009–13020, 2025. 1
- [26] Lingyi Hong, Wenchao Chen, Zhongying Liu, Wei Zhang, Pinxue Guo, Zhaoyu Chen, and Wenqiang Zhang. Lvos: A benchmark for long-term video object segmentation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 13480–13492, 2023. 3, 5, 6, 7, 8, 12
- [27] Lingyi Hong, Zhongying Liu, Wenchao Chen, Chenzhi Tan, Yuang Feng, Xinyu Zhou, Pinxue Guo, Jinglun Li, Zhaoyu Chen, Shuyong Gao, et al. Lvos: A benchmark for largescale long-term video object segmentation. arXiv preprint arXiv:2404.19326, 2024. 3, 5, 6, 8, 12
- [28] Ping Hu, Gang Wang, Xiangfei Kong, Jason Kuen, and YapPeng Tan. Motion-guided cascaded refinement network for video object segmentation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1400–1409, 2018. 3
- [29] Yuan-Ting Hu, Jia-Bin Huang, and Alexander G Schwing. Videomatch: Matching based video object segmentation. In Proceedings of the European conference on computer vision (ECCV), pages 54–70, 2018. 3
- [30] Lianghua Huang, Xin Zhao, and Kaiqi Huang. Got-10k: A large high-diversity benchmark for generic object tracking in the wild. IEEE transactions on pattern analysis and machine intelligence, 43(5):1562–1577, 2019. 6, 12
- [31] Joakim Johnander, Martin Danelljan, Emil Brissman, Fahad Shahbaz Khan, and Michael Felsberg. A generative appearance model for end-to-end video object segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8953–8962, 2019. 3
- [32] Chanho Kim, Fuxin Li, Arridhana Ciptadi, and James M Rehg. Multiple hypothesis tracking revisited. In Proceedings of the IEEE international conference on computer vision, pages 4696–4704, 2015. 2, 3
- [33] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4015–4026, 2023. 1, 3
- [34] Mingxing Li, Li Hu, Zhiwei Xiong, Bang Zhang, Pan Pan, and Dong Liu. Recurrent dynamic embedding for video object segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1332– 1341, 2022. 3, 7

- [35] Xiaoxiao Li and Chen Change Loy. Video object segmentation with joint re-identification and attention-aware mask propagation. In Proceedings of the European conference on computer vision (ECCV), pages 90–105, 2018.
- [36] Yu Li, Zhuoran Shen, and Ying Shan. Fast video object segmentation using the global context module. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part X 16, pages 735–

750. Springer, 2020. 3

- [37] Yongqing Liang, Xin Li, Navid Jafari, and Jim Chen. Video object segmentation with adaptive feature bank and uncertainregion refinement. Advances in Neural Information Processing Systems, 33:3430–3441, 2020. 3
- [38] Liting Lin, Heng Fan, Zhipeng Zhang, Yaowei Wang, Yong Xu, and Haibin Ling. Tracking meets lora: Faster training, larger model, stronger performance. In European Conference on Computer Vision, pages 300–318. Springer, 2025. 8
- [39] Run Luo, Zikai Song, Lintao Ma, Jinlin Wei, Wei Yang, and Min Yang. Diffusiontrack: Diffusion model for multi-object tracking. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 3991–3999, 2024. 8
- [40] K-K Maninis, Sergi Caelles, Yuhua Chen, Jordi Pont-Tuset, Laura Leal-Taix´e, Daniel Cremers, and Luc Van Gool. Video object segmentation without temporal information. IEEE transactions on pattern analysis and machine intelligence, 41

(6):1515–1530, 2018. 3

- [41] Christoph Mayer, Martin Danelljan, Danda Pani Paudel, and Luc Van Gool. Learning target candidate association to keep track of what not to track. In Proceedings of the IEEE/CVF international conference on computer vision, pages 13444– 13454, 2021. 8
- [42] Christoph Mayer, Martin Danelljan, Goutam Bhat, Matthieu Paul, Danda Pani Paudel, Fisher Yu, and Luc Van Gool. Transforming model prediction for tracking. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8731–8740, 2022. 8
- [43] Seoung Wug Oh, Joon-Young Lee, Kalyan Sunkavalli, and Seon Joo Kim. Fast video object segmentation by referenceguided mask propagation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 7376–7385, 2018. 3
- [44] Seoung Wug Oh, Joon-Young Lee, Ning Xu, and Seon Joo Kim. Video object segmentation using space-time memory networks. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 9226–9235, 2019. 3
- [45] Federico Perazzi, Jordi Pont-Tuset, Brian McWilliams, Luc Van Gool, Markus Gross, and Alexander Sorkine-Hornung. A benchmark dataset and evaluation methodology for video object segmentation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 724–732,

2016. 3

- [46] Federico Perazzi, Anna Khoreva, Rodrigo Benenson, Bernt Schiele, and Alexander Sorkine-Hornung. Learning video object segmentation from static images. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2663–2672, 2017. 3
- [47] Jordi Pont-Tuset, Federico Perazzi, Sergi Caelles, Pablo Arbel´aez, Alex Sorkine-Hornung, and Luc Van Gool. The 2017

- davis challenge on video object segmentation. arXiv preprint arXiv:1704.00675, 2017. 3, 5, 6, 8, 12
- [48] Rui Qian, Shuangrui Ding, Xian Liu, and Dahua Lin. Semantics meets temporal correspondence: Self-supervised objectcentric learning in videos. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 16675– 16687, 2023. 3
- [49] Rui Qian, Shuangrui Ding, and Dahua Lin. Rethinking image-to-video adaptation: An object-centric perspective. In European Conference on Computer Vision, pages 329–348. Springer, 2024. 3
- [50] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman R¨adle, Chloe Rolland, Laura Gustafson, et al. Sam 2: Segment anything in images and videos. arXiv preprint arXiv:2408.00714,

2024. 1, 3, 5, 6, 7, 8, 12

- [51] Donald Reid. An algorithm for tracking multiple targets. IEEE transactions on Automatic Control, 24(6):843–854,

1979. 2, 3

- [52] Andreas Robinson, Felix Jaremo Lawin, Martin Danelljan, Fahad Shahbaz Khan, and Michael Felsberg. Learning fast and robust target models for video object segmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 7406–7415, 2020. 3
- [53] Hongje Seong, Junhyuk Hyun, and Euntai Kim. Kernelized memory network for video object segmentation. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XXII 16, pages 629–645. Springer, 2020. 3
- [54] Pavel Tokmakov, Jie Li, and Adrien Gaidon. Breaking the “object” in video object segmentation. In CVPR, 2023. 3, 5, 8, 12
- [55] Carles Ventura, Miriam Bellver, Andreu Girbau, Amaia Salvador, Ferran Marques, and Xavier Giro-i Nieto. Rvos: Endto-end recurrent network for video object segmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5277–5286, 2019. 3
- [56] Paul Voigtlaender and Bastian Leibe. Online adaptation of convolutional neural networks for video object segmentation. arXiv preprint arXiv:1706.09364, 2017. 3
- [57] Paul Voigtlaender, Yuning Chai, Florian Schroff, Hartwig Adam, Bastian Leibe, and Liang-Chieh Chen. Feelvos: Fast end-to-end embedding learning for video object segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9481–9490, 2019. 3
- [58] Junke Wang, Dongdong Chen, Zuxuan Wu, Chong Luo, Chuanxin Tang, Xiyang Dai, Yucheng Zhao, Yujia Xie, Lu Yuan, and Yu-Gang Jiang. Look before you match: Instance understanding matters in video object segmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 2268–2278, 2023. 3
- [59] Qiang Wang, Li Zhang, Luca Bertinetto, Weiming Hu, and Philip HS Torr. Fast online object tracking and segmentation: A unifying approach. In Proceedings of the IEEE/CVF conference on Computer Vision and Pattern Recognition, pages 1328–1338, 2019. 3
- [60] Qiangqiang Wu, Tianyu Yang, Ziquan Liu, Baoyuan Wu, Ying Shan, and Antoni B Chan. Dropmae: Masked autoencoders

- with spatial-attention dropout for tracking tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14561–14571, 2023. 8
- [61] Qiangqiang Wu, Tianyu Yang, Wei Wu, and Antoni B Chan. Scalable video object segmentation with simplified framework. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 13879–13889, 2023. 3
- [62] Haozhe Xie, Hongxun Yao, Shangchen Zhou, Shengping Zhang, and Wenxiu Sun. Efficient regional memory network for video object segmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1286–1295, 2021. 3
- [63] Ning Xu, Linjie Yang, Yuchen Fan, Dingcheng Yue, Yuchen Liang, Jianchao Yang, and Thomas Huang. Youtube-vos: A large-scale video object segmentation benchmark. arXiv preprint arXiv:1809.03327, 2018. 5, 6, 8, 12
- [64] Linjie Yang, Yanran Wang, Xuehan Xiong, Jianchao Yang, and Aggelos K Katsaggelos. Efficient video object segmentation via network modulation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 6499–6507, 2018. 3
- [65] Zongxin Yang and Yi Yang. Decoupling features in hierarchical propagation for video object segmentation. Advances in Neural Information Processing Systems, 35:36324–36336,

2022. 3, 7

- [66] Zongxin Yang, Yunchao Wei, and Yi Yang. Collaborative video object segmentation by foreground-background integration. In European Conference on Computer Vision, pages 332–348. Springer, 2020. 3
- [67] Zongxin Yang, Yunchao Wei, and Yi Yang. Associating objects with transformers for video object segmentation. Advances in Neural Information Processing Systems, 34:2491– 2502, 2021. 3
- [68] Zongxin Yang, Yunchao Wei, and Yi Yang. Collaborative video object segmentation by multi-scale foregroundbackground integration. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(9):4701–4712, 2021. 3
- [69] Botao Ye, Hong Chang, Bingpeng Ma, Shiguang Shan, and Xilin Chen. Joint feature learning and relation modeling for tracking: A one-stream framework. In European Conference on Computer Vision, pages 341–357. Springer, 2022. 8
- [70] Jiaming Zhang, Yutao Cui, Gangshan Wu, and Limin Wang. Joint modeling of feature, correspondence, and a compressed memory for video object segmentation. arXiv preprint arXiv:2308.13505, 2023. 3
- [71] Lu Zhang, Zhe Lin, Jianming Zhang, Huchuan Lu, and You He. Fast video object segmentation via dynamic targeting network. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 5582–5591, 2019. 3
- [72] Yaozong Zheng, Bineng Zhong, Qihua Liang, Zhiyi Mo, Shengping Zhang, and Xianxian Li. Odtrack: Online dense temporal token learning for visual tracking. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 7588– 7596, 2024. 8
- [73] Junbao Zhou, Ziqi Pang, and Yu-Xiong Wang. Rmem: Restricted memory banks improve video object segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18602–18611, 2024. 3

### A. Dataset Information

SA-V [50] is a large-scale video segmentation dataset designed for promptable visual segmentation across diverse scenarios. It encompasses 50.9K video clips, aggregating to 642.6K masklets with 35.5M meticulously annotated masks. The dataset presents a challenge with its inclusion of small, occluded, and reappearing objects throughout the videos. The dataset is divided into training, validation, and testing sets, with most videos allocated to the training set for robust model training. The validation set has 293 masklets across 155 videos for model tuning, while the testing set includes 278 masklets across 150 videos for comprehensive evaluation.

- LVOS v1 [26] is a VOS benchmark for long-term video object segmentation in realistic scenarios. It comprises 720 video clips with 296,401 frames and 407,945 annotations, with an average video duration of over 60 seconds. LVOS introduces challenging elements such as long-term object reappearance and cross-temporal similar objects. In LVOS v1, the dataset includes 120 videos for training, 50 for validation, and 50 for testing.
- LVOS v2 [27] expends LVOS v1 and provides 420 videos for training, 140 for validation, and 160 for testing. This paper primarily utilizes v2, as it already includes the sequences present in v1. The dataset spans 44 categories, capturing typical everyday scenarios, with 12 of these categories deliberately left unseen to evaluate and better assess the generalization capabilities of VOS models.

MOSE [17] is a challenging VOS dataset targeted on complex, real-world scenarios, featuring 2,149 video clips with 431,725 high-quality segmentation masks. These videos are split into 1,507 training videos, 311 validation videos, and 331 testing videos.

VOST [54] is a semi-supervised video object segmentation benchmark that emphasizes complex object transformations. Unlike other datasets, VOST includes objects that are broken, torn, or reshaped, significantly altering their appearance. It comprises more than 700 high-resolution videos, captured in diverse settings, with an average duration of 21 seconds, all densely labeled with instance masks.

PUMaVOS [4] is a novel video dataset designed for benchmarking challenging segmentation tasks. It includes 24 video clips, each ranging from 13.5 to 60 seconds (28.7 seconds on average) at 480p resolution with varying aspect ratios. PUMaVOS focuses on difficult scenarios where annotation boundaries do not align with clear visual cues, such as half faces, necks, tattoos, and pimples, commonly encountered in video production.

YouTubeVOS-2019 [63] is a large-scale video object segmentation dataset featuring 3,252 sequences with detailed annotations at 6 FPS across 78 diverse categories, including humans, animals, vehicles, and accessories. Each video

clip is between 3 to 6 seconds long and frequently contains multiple objects, which have been manually segmented by professional annotators.

DAVIS2017 [47] is a well-known benchmark dataset comprising 60 training videos and 30 validation videos, with a total of 6,298 frames. It offers high-quality, pixel-level annotations for every frame, making it a standard resource for evaluating different VOS methods.

LaSOT [22] is a large-scale tracking dataset designed for long-term visual object tracking. It contains 1,400 videos spanning 70 object categories, with an average sequence length exceeding 2,500 frames, making it a challenging benchmark for evaluating tracking algorithms. LaSOText[23] is an extension of LaSOT, featuring a subset of 15 categories with 150 videos.

GOT-10k [30] is a large-scale generic object tracking dataset that covers over 10,000 video sequences, spanning more than 560 object classes. It provides strict one-shot evaluation settings and diverse real-world tracking scenarios, making it a widely used benchmark for developing and assessing tracking models.

### B. More Ablation Study

Iou Threshold δiou. The choice of the IoU threshold δiou is crucial for selecting frames with reliable object cues. As shown in Table 9, setting 0.1 ≥ δiou ≥ 0.7 yields the competitive J &F, indicating an effective trade-off between filtering out poor-quality frames and retaining valuable segmentation information. In contrast, setting no quality requirement for masks (δiou = 0) lowers the score to 80.0, as unreliable frames with poor segmentation harm SAM 2. Conversely, an overly strict selection (δiou ≥ 0.8) further degrades performance by excluding important neighboring frames, forcing the model to rely on distant frames as memory.

Uncertainty Threshold δconf. The uncertainty threshold δconf controls the selection of hypotheses under uncertain conditions. Our results in Table 10 indicate that setting δconf to 2 provides the highest J &F score, indicating an optimal level for uncertainty handling. Lower values (e.g., 0) lead to suboptimal performance by committing to incorrect segmentations, causing error propagation. Higher values (e.g., 4) do not improve performance, indicating that beyond a certain threshold, the model efficiently relies on top-scoring masks without needing additional diversity.

Memory Attention Modulation [wlow,whigh]. We explore the effect of modulating the attention weights for memory entries using different ranges in Table 11. The configuration [1,1] means no modulation is applied. We find that the configuration of [0.95,1.05] achieves the best performance while increasing the modulation range decreases performance. This result indicates that slight modulation sufficiently emphasizes reliable memory entries.

δiou SAM 2 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 SA-V 76.3 80.0 80.7 80.6 80.8 80.7 81.0 80.6 80.8 80.0 77.8 LVOS 83.0 84.0 85.1 85.6 85.4 85.4 85.1 85.6 85.2 84.1 83.5

Table 9. Ablation study on IoU threshold δiou.

δconf SAM 2 0 1 2 3 4 5 SA-V 76.3 80.4 80.4 80.8 80.8 79.7 78.8 LVOS 83.0 84.4 84.6 85.4 85.2 84.1 83.9

Table 10. Ablation study on uncertainty threshold δconf.

[wlow, whigh] SAM 2 [0.6, 1.4] [0.7, 1.3] [0.8, 1.2] [0.9, 1.1] [0.95, 1.05] [1, 1] SA-V 76.3 77.1 79.3 79.8 80.4 80.8 80.5 LVOS 83.0 77.0 78.8 82.9 85.0 85.4 84.9

Table 11. Ablation study on modulation weight [wlow, whigh].

SA-V val Single-object Multi-object Overall # of seq 73 82 155 SAM 2.1 79.0 78.4 78.6 SAM2.1Long 80.6 (1.6% ↑) 81.3 (2.9% ↑) 81.1 (2.5% ↑)

Table 12. Performance comparison on single-object sequence and multi-object sequence. SAM2Long performs exceptionally well in both single-object and multi-object cases.

### C. More Quantitive Comparion

In this section, we present a detailed comparison of SAM2.1 and SAM2Long across single-object and multi-object sequences, as shown in Table 12. Our experiments demonstrate that SAM2Long outperforms SAM2.1 in both singleobject and multi-object scenarios. Specifically, SAM2Long achieves a 1.6% improvement in single-object sequences, a 2.9% improvement in multi-object sequences, and an overall 2.5% enhancement in performance. These results highlight SAM2Long’s robustness and effectiveness in various video segmentation tasks.

### D. More Visualization

We present additional comparisons between SAM2 and SAM2Long in Figure 4. SAM2Long significantly reduces segmentation errors, showing improved accuracy and consistency in object tracking across frames. Notably, in the Fast & Furious movie scene, SAM2Long successfully tracks the green car, even under challenging dynamic camera movements. Overall, SAM2Long offers substantial improvements over SAM2, especially in handling object occlusion and reappearance, leading to better performance in long-term video segmentation tasks.

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

SAM2LongGTSAM2SAM2SAM2SAM2LongSAM2LongGT

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

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

[Figure 114]

[Figure 115]

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

Figure 4. Qualitative comparison between SAM 2 and SAM2Long, with GT (Ground Truth) provided for reference. The last row shows an in-wild case. Best viewed when zoomed in.

