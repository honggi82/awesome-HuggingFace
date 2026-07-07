# arXiv:2510.20579v2[cs.CV]18Mar2026

[Figure 1]

[Figure 2]

[Figure 3]

## Open-o3-Video: Grounded Video Reasoning with Explicit Spatio-Temporal Evidence

Jiahao Meng1,2 , Xiangtai Li2 , Haochen Wang2,3 , Yue Tan1 , Tao Zhang2,4 , Lingdong Kong2,5 , Yunhai Tong1 †, Anran Wang2 , Zhiyang Teng2 , Yujing Wang2 ,

Zhuochen Wang2

PKU1 ByteDance2 CASIA3 WHU4 NUS5 †Corresponding Author Project Page: https://marinero4972.github.io/projects/Open-o3-Video/

### Abstract

Most video reasoning models only generate textual reasoning traces without indicating when and where key evidence appears. Recent models such as OpenAI-o3 have sparked wide interest in evidence-centered reasoning for images, yet extending this ability to videos is more challenging due to the need for joint temporal tracking and spatial localization across dynamic scenes. We introduce Open-o3-Video, a non-agent framework that integrates explicit spatio-temporal evidence into video reasoning by highlighting key timestamps, objects, and bounding boxes, making the reasoning process traceable and verifiable. To enable this capability, we first construct high-quality datasets STGR that provide unified spatio-temporal supervision, which is absent in existing resources. We further adopt a cold-start reinforcement learning strategy with specially designed rewards that jointly encourage answer accuracy, temporal alignment, and spatial precision. On the V-STAR benchmark, Open-o3-Video achieves state-of-the-art performance, improving mAM by 14.4% and mLGM by 24.2% over the Qwen2.5-VL baseline, and shows consistent gains across a range of video understanding benchmarks. Beyond accuracy, the grounded reasoning traces produced by Open-o3-Video support confidence-aware test-time scaling, improving answer reliability.

Date: March 20, 2026 Correspondence: Yunhai Tong at yhtong@pku.edu.cn, Xiangtai Li at xiangtai.li@bytedance.com

### 1 Introduction

Understanding complex video content is a long-standing goal for large multimodal models [7, 45, 46, 48, 57, 60, 63, 65], as videos encapsulate rich temporal dynamics and spatial interactions that far exceed the information in static images. While recent progress has improved performance on tasks such as video question answering [3, 27, 59, 65, 68], building models that can perform reliable, fine-grained reasoning over long, cluttered scenes remains challenging.

Recent “Thinking with Images” attempts [36, 47, 49, 67] leverage explicit operations (such as cropping, zoom-in, and region selection) to interleave detailed visual evidence with language, achieving superior performance on fine-grained image comprehension. This success motivates extending a similar paradigm to the video domain.

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

- Figure 1 While prior video reasoning models (e.g., Video-R1 [16], VideoRFT [50]) only generate textual rationales, Open-o3-Video integrates explicit spatio-temporal grounding into the reasoning process. The model highlights key timestamps and object regions that directly support the answer, providing verifiable evidence for its prediction. More visualizations are provided in Appendix A.11.

However, this extension is difficult and non-trivial due to the requirement for coherent localization across both time and space precisely. The complexity of dynamic scenes, e.g., replete with motion, occlusions, and camera changes, makes it incredibly challenging to pinpoint when and where events of interest occur. As a result, previous attempts to incorporate explicit reasoning in video have often been limited to textual rationales [16, 50] or, coarse, temporal-only grounding [28, 52], failing to achieve the fine-grained spatio-temporal precision necessary for complex video reasoning. This gap is largely due to two interconnected obstacles: (1) the absence of high-quality datasets that provide joint spatio-temporal supervision for reasoning, and (2) the inherent difficulty of training a model to precisely localize objects in time and space simultaneously.

To address these challenges, we introduce Open-o3-Video, a framework that embeds joint spatio-temporal evidence directly into the reasoning process. Our first key contribution is the creation of a comprehensive training corpus designed to bridge this data gap. We have curated two training datasets, STGR-CoT-30k and STGR-RL-36k, for supervised fine-tuning and reinforcement learning, respectively. These datasets integrate existing temporal-only and spatial-only grounding resources with 5.9k newly annotated high-quality spatiotemporal samples. Each instance contains a question-answer pair, timestamped key frames, localized bounding boxes, and a chain of thought that explicitly links the visual evidence to the reasoning steps.

Building on this dataset, our second contribution is a two-stage training strategy with adaptive temporal proximity and temporal gating to stably and efficiently optimize the model’s spatio-temporal reasoning capability. Although the model has acquired preliminary capabilities for generating structured, grounded chains of thought during the supervised fine-tuning stage, the subsequent reinforcement learning stage still cannot achieve stable training due to a critical spatial collapse issue. This is because spatial grounding rewards are usually conditioned on correctly identifying the timestamp. When temporal predictions are imprecise in the early stages, this leads to near-zero spatial rewards, stalling the learning process for localization ability. Therefore, we propose a novel adaptive temporal proximity technique that relaxes the temporal requirement during early training to reduce reward sparsity and gradually increases the precision demand over time. This training strategy prevents premature saturation of the temporal reward signal and ensures that predicted timestamps continue to approach the ground truth, which is crucial for reliable spatial evaluation. In parallel, a complementary temporal gating mechanism computes spatial rewards only when temporal predictions are sufficiently accurate, preventing irrelevant objects from being rewarded and enforcing precise spatio-temporal

[Figure 11]

[Figure 12]

- Figure 2 Overview of our data construction pipeline and dataset composition. Left: The annotation pipeline includes Gemini 2.5 Pro initial annotation, bounding box filtering, and self-consistency checking. Right: Distribution of data categories in STGR-CoT-30k (SFT) and STGR-RL-36k (RL), showing a balanced coverage across temporal, spatial, spatio-temporal, and general QA.

alignment. Together, these mechanisms provide dense yet reliable feedback, forming a smoother learning curriculum that progressively strengthens both temporal accuracy and spatial grounding.

Through this combination of curated data and our training procedure, as shown in Figure 1, Open-o3-Video produces reasoning that is accurate, interpretable, and grounded in the visual evidence. We evaluate Open-o3Video on the V-STAR benchmark [10] and other video understanding tasks. On V-STAR, our model achieves state-of-the-art performance, surpassing GPT-4o and improving over Qwen2.5-VL by +14.4% mAM and +24.2% mLGM with a small amount of training data. Beyond V-STAR, Open-o3-Video also delivers consistent gains on VideoMME, WorldSense, VideoMMMU, LongVideo-Reason-eval, and TVGBench, demonstrating advantages in long-video reasoning, perception-oriented tasks, and fine-grained temporal localization. We further find that the generated evidence is informative, as removing evidence-aligned frames causes larger performance degradation, and it can also be leveraged for test-time scaling, where evidence-based confidence-aware voting outperforms majority voting (e.g., +1.2% on WorldSense and 1.0% on VideoMMMU).

### 2 Related works

Video Reasoning. Recent advances in video reasoning [8, 13, 17, 28, 39, 50, 53, 55, 61, 64] have largely been driven by reinforcement learning based post-training on multi-modal large language models (MLLMs), which encourages models to move beyond direct question answering and exhibit step-by-step reasoning. Video-R1 [17] shows that temporal-aware GRPO with curated reasoning data improves video understanding benchmarks, while VideoChat-R1 [28] extends to spatio-temporal perception tasks such as grounding and tracking without harming QA. Other variants, including Video-RTS [53] and DeepVideo-R1 [39], combine reinforcement learning with test-time scaling or difficulty-aware regularization to better exploit temporal information. Despite their success, most methods rely on text-only reasoning. However, our approach incorporates spatio-temporal evidence for transparent and verifiable grounding.

Temporal and Spatial Grounding in Video. The problem of locating when and where relevant evidence appears in a video has attracted increasing attention, leading to progress in both temporal and spatial grounding [6, 20, 28, 29, 37, 51, 52]. Temporal grounding methods such as Time-R1 [52] and TVG-R1 [6] leverage verifiable rewards and curated RL data to improve temporal localization, while spatial grounding approaches like SpaceR [37] focus on object-centric localization and geometric reasoning. Several works further explore spatio-temporal localization through architectural or training designs, including STCAT [24], LRR [4], EgoMask [30], and LLaVA-ST [26]. However, existing methods do not align spatio-temporal localization with chain-of-thought reasoning. Our approach addresses this gap by explicitly linking object regions with temporal positions and incorporating spatio-temporal evidence into reasoning, enabling more verifiable video

understanding.

Thinking with Images. A growing line of research [15, 36, 47, 49, 67] explores how multi-modal models improve reasoning by performing explicit visual operations such as cropping, zoom-in, and region selection, thereby producing intermediate evidence that is consumed within the reasoning chain. OpenAI-o3 [36] formalizes “thinking with images,” while DeepEyes [67] shows end-to-end RL can incentivize image–tool reasoning, and TreeBench [47] provides methodology for traceable, box-level evidence. These advances demonstrate the promise of evidence-centric visual reasoning but are largely image-centric. Extending to videos adds challenges in temporal consistency, motion, and fine-grained event alignment. Several concurrent works discussed in Appendix A.2 like VITAL [61] and Conan [38] adapt the paradigm via an agent-based, tool-augmented RL pipeline, yielding gains but relying on external orchestration. In contrast, our framework “thinks with frames” in a single round of inference, directly emitting timestamped crops and bounding boxes as evidence without complex tool pipelines.

### 3 STGR Data Construction

#### 3.1 Data Source and Statistics

Building robust spatio-temporal reasoning models requires training signals that jointly supervise when and where evidence appears and how it is used in reasoning. Existing resources fall short in three ways: (i) temporal-only grounding datasets provide time spans but lack object regions; (ii) spatial or frame-level caption corpora offer boxes on isolated frames without timestamps; and (iii) most lack a chain of thought that explicitly ties objects and timestamps to the answer. These gaps make it impossible to learn coherent localization in dynamic scenes and to compute verifiable rewards for RL, since temporal and spatial supervision are not synchronized and reasoning traces are text-only.

To bridge this gap, we curate two complementary corpora: STGR-CoT-30k for supervised fine-tuning (SFT) and STGR-RL-36k for reinforcement learning (RL). Both combine existing temporal-only and spatial-only resources with 5.9k newly annotated, high-quality spatio-temporal samples produced by our pipeline (Sec. 3.2). Each new instance includes a question–answer pair, timestamped key frames, localized boxes, and a structured chain of thought that links visual evidence to reasoning steps. This design provides synchronized temporal and spatial supervision for SFT to acquire grounded reasoning formats and reliable, verifiable signals for RL to optimize alignment under complex video dynamics.

The SFT corpus consists of four components: (i) 4.1k temporal grounding CoT samples (TVG-Coldstart) [6], (ii) 5k spatial grounding CoT samples (TreeVGR-SFT) [47], (iii) 5.9k spatio-temporal samples curated by us, including 3.9k from temporal grounding datasets (video sources are shown in Appendix A.3) and 2k from PLM-Rdcap [11], and (iv) 15k Video-R1-CoT samples [17]. The RL corpus further expands diversity: (i) 5.2k temporal grounding samples, including 2.3k from Time-R1 [52] and 2.9k from TVG-RL [6], (ii) 5k spatial grounding samples from VisCoT [40], (iii) 10.9k spatio-temporal samples, comprising our 5.9k constructed data (via the pipeline) and an additional 5k filtered from VideoEspresso [21] with consistency checks, and (iv) 15k Video-R1 samples [17].

Overall, as shown in Figure 2 (right), the SFT set covers 13.7% temporal, 16.7% spatial, 19.7% spatio-temporal, and 50.0% general QA data, while the RL set includes 14.4% temporal, 13.9% spatial, 30.3% spatio-temporal, and 41.7% QA data. This design ensures that both phases expose the model to diverse supervisory signals while emphasizing spatio-temporal reasoning as the central capability. More details of training data are provided in the Appendix A.3.

#### 3.2 Data Annotation Pipeline

Spatio-temporal reasoning requires chain-of-thought data that include both temporal spans and spatial grounding. We construct 5.9k such samples by combining temporal grounding datasets with PLM-Rdcap sources (Figure 2, left). The pipeline follows three stages below.

Data Preparation and Initial Annotation. We begin by collecting two types of sources: temporal grounding datasets and PLM-Rdcap data that provide region-level dense captions. All videos are passed through the

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

- Figure 3 Overview of Open-o3-Video. We adopt a two-stage training paradigm: (a) cold-start initialization to learn structured, grounded outputs; (b) reinforcement learning with a composite reward that sharpens temporal alignment and spatial precision with adaptive temporal proximity and temporal gating.

Gemini 2.5 Pro [12] API, with carefully designed prompts (shown in Appendix A.4) to generate structured annotations. Each annotation contains (i) a question-answer pair centered on a specific object or person, (ii) one to five key frames sampled from the annotated segment, (iii) bounding boxes for one to three salient objects in each key frame, and (iv) a reasoning process that must reference every object with explicit format: <obj>object_name</obj><box>[x_min, y_min, x_max, y_max]</box>at<t>timestamp</t>s.

Bounding Box Filtering. Initial annotations may contain noisy or incorrect boxes. We filter them with two rules: (i) boxes covering over 80% of the frame are removed as uninformative; (ii) each crop is verified by Qwen2.5-VL-7B [3] with the query “Is this a {object_name}?”. Only samples that answered “yes” are kept, ensuring that object mentions match the validated boxes.

Self-consistency Checking and Quality Control. Our consistency checking enforces alignment between timestamps, bounding boxes, and the spatio-temporal reasoning chain. For each annotated sample, we verify that all temporal and spatial references appearing in the reasoning text are covered by the corresponding “key_frames” and “key_items” annotations. Samples with missing elements are discarded. We further assess the relevance between each reasoning sentence and its associated visual evidence by cropping the referenced region and querying Qwen2.5-VL to judge semantic consistency. Samples with inconsistent visual-textual alignment are removed. These checks improve annotation quality and provide reliable supervision for cold-start grounded training.

- 4 Open-o3-Video

As shown in Figure 3, our training recipe comprises two stages: a cold-start initialization phase followed by reinforcement learning that enhances spatio-temporal reasoning through carefully designed rewards with adaptive temporal proximity and temporal gating mechanisms.

#### 4.1 Cold Start Initialization

We initialize our model from Qwen2.5-VL-7B [3], and fine-tune it on the constructed STGR-CoT-30k corpus. This cold-start stage equips the model with basic spatio-temporal grounding and structured reasoning capabilities, reducing reward sparsity and stabilizing subsequent reinforcement learning.

#### 4.2 Reinforcement Learning with GSPO

We adopt Group Sequence Policy Optimization (GSPO) [66] as our reinforcement learning algorithm. Compared with GRPO [41], which operates at the token level, GSPO defines importance ratios and clipping at the sequence level, ensuring that optimization is aligned with sequence-level rewards. This eliminates high-variance

token-wise corrections, stabilizes long-horizon training, and avoids collapse in chain-of-thought reasoning. In our video spatio-temporal grounded reasoning setting, rewards are defined over complete reasoning traces that include timestamps and bounding boxes, rather than individual tokens. GSPO better matches by optimizing whole sequences as atomic units, allowing the policy to consistently improve global grounding quality instead of overfitting to local token-level signals. As a result, GSPO yields higher grounding accuracy and more stable training than GRPO in our experiments (Section 5.2).

During training, given a video-question pair x, each generated response y is evaluated with a scalar reward r(x,y) that reflects both correctness and reasoning quality. This reward serves as the optimization signal in GSPO, and more details of the GSPO algorithm are provided in Appendix A.9.

- 4.3 Reward Design For each query–completion pair (x,y), the scalar reward is defined as

r(x,y) = racc(x,y) + rthk(x,y) + rfmt(x,y), (1) which is group-normalized to obtain the advantage used by GSPO. Below, we describe the three components. Accuracy Reward racc. Since the training data span multiple tasks, we design task-specific accuracy rewards. Let τ ∈ {MCQ,QA,SG,TG} denote the task type, where MCQ is a multiple-choice question, QA is free-form question answering, SG is spatial grounding, and TG is temporal grounding. For spatial grounding, we denote the predicted and ground-truth bounding boxes by Bpred and Bgt, respectively. For temporal grounding, we denote the predicted and ground-truth temporal segments by Spred = [spred,epred] and Sgt = [sgt,egt]. We then define:

 

I ypred = ygt , τ = MCQ,

ROUGE ypred,ygt , τ = QA, vIoU Bpred,Bgt , τ = SG, tIoU Spred,Sgt , τ = TG.

(2)

racc(x,y) =



Thinking Reward rthk. We define the thinking reward as the sum of temporal and spatial terms:

rthk(x,y) = rt(x,y) + rs(x,y). (3)

Temporal term with adaptive temporal proximity. Let M be the number of timestamps {tm}Mm=1 parsed from <think>. The temporal reward rt(x,y) depends on the supervision type τt ∈ {Int,Pt,∅}: Interval supervision (Int) provides a ground-truth span [sgt,egt], point supervision (Pt) provides ground-truth timestamps {tgtj }; and ∅ indicates no timestamp evidence. For point supervision, we define the closest temporal distance ∆tm = minj|tm − tgtj |.



M

1 M

1 sgt ≤ tm ≤ egt , τt = Int, 1



m=1

(4)

M

rt(x,y) =

∆t2m 2σ2

, τt = Pt, 0, τt = ∅.

exp −

M



m=1

A key difficulty is that spatial rewards depend on accurate temporal predictions: IoU can only be computed reliably when the timestamp is close to the ground truth. If the temporal constraint is too strict (i.e., σ very small), the model receives little reward when its early temporal predictions are inaccurate, which slows down temporal learning and, in turn, prevents spatial grounding from being learned effectively. Conversely, if the constraint is always loose (i.e., σ large), temporal rewards quickly saturate and stop driving predicted timestamps closer to the ground truth, which again undermines spatial reward reliability. To resolve this trade-off, we propose adaptive temporal proximity: σ is large in early training to provide dense signals, and gradually decreases to enforce stricter alignment. This strategy ensures that the model first obtains stable gradients and later achieves precise timestamping, providing a solid foundation for spatial evaluation.

- Table 1 Performance on the V-STAR benchmark, which evaluates spatio-temporal reasoning across three dimensions. Chain1 denotes what–when–where, while Chain2 corresponds to what–where–when. mAM is the average of the arithmetic mean, and mLGM is the average of the modified logarithmic geometric mean, combining temporal and spatial alignment. * indicates we re-evaluate using the vLLM framework with 16 sampled frames. Bold numbers denote the best results, while underlined numbers indicate the second best. More experimental results based on Qwen3-VL models are provided in Appendix A.6.

Model What When (Temporal IoU) Where (Visual IoU) Overall

Acc Chain1 Chain2 Chain1 Chain2 mAM mLGM

GPT-4o 60.8 16.7 12.8 6.5 3.0 26.8 38.2 Gemini-2-Flash 53.0 24.5 23.8 4.6 2.2 26.9 35.6

Video-LLaMA3 41.9 23.0 23.1 0.9 0.2 21.7 27.0 LLaVA-Video 49.5 10.5 12.2 1.9 1.3 20.8 27.3 VideoChat2 36.2 13.7 12.5 2.5 1.0 17.0 20.3 Oryx-1.5-7B 20.5 13.5 14.8 10.1 3.5 15.1 13.8 InternVL-2.5-8B 44.2 8.7 7.8 0.7 0.1 17.6 24.9 Qwen2.5-VL-7B*(base) 33.5 15.4 13.8 17.0 2.5 19.3 22.4

TRACE 17.6 19.1 17.1 0.0 0.0 12.0 13.3 Sa2VA-8B 16.4 0.1 0.0 32.3 37.5 17.1 20.3 Open-o3-Video-7B (Ours) 61.0 24.5 24.0 25.4 6.0 33.7 46.6 ∆ vs. Qwen2.5-VL-7B ↑ 27.5 ↑ 9.1 ↑ 10.2 ↑ 8.4 ↑ 3.5 ↑ 14.4 ↑ 24.2

Spatial term with temporal gating. For each predicted timestamp tm, let j⋆(m) = arg minj |tm − tgtj | be the nearest annotated time. Let Bm be the predicted boxes at tm and Bjgt⋆(m) be the ground-truth boxes at the matched frame. We define the per-frame maximal overlap as vm = maxb∈B

m, bgt∈Bjgt⋆(m) IoU(b,bgt). The spatial reward is

M

1 M

1 |tm − tgtj⋆(m)| ≤ τ · vm, (5)

rs(x,y) =

m=1

where τ is a temporal threshold. We propose a temporal gating mechanism to guarantee the reliability of spatial supervision. Specifically, spatial rewards are only computed when temporal predictions are sufficiently close to the ground truth. This prevents rewarding salient but irrelevant objects at wrong timestamps, enforces spatio-temporal alignment, and ultimately improves both the interpretability and reliability of the reasoning process. Together, adaptive temporal proximity and temporal gating provide complementary solutions: the former provides stable, progressive temporal supervision, while the latter ensures accurate, trustworthy spatial evaluation.

Format Reward rfmt. Strict usage of <think> and <answer> with correct <obj> <box> <t> gives 1.0. Having only <think> and <answer> yields 0.5. Otherwise, the reward is 0.0.

### 5 Experiments

Implementation Details. We build upon the Qwen2.5-VL-7B model and train on 8 NVIDIA H100 GPUs. During training, we uniformly sample 16 frames from each video, where each frame has a resolution not exceeding 128 × 28 × 28. If annotated key frames are available, they are inserted in addition to the uniformly sampled frames. To strengthen the model’s perception of temporal information, we prepend each frame with its absolute timestamp. More implementation details are provided in Appendix A.1.

Benchmarks. We adopt V-STAR [10] as the main benchmark, since it is specifically designed to measure spatio-temporal grounding in videos. Unlike conventional video QA datasets, V-STAR requires models not only to answer questions but also localize when and where the supporting evidence occurs. It introduces

- Table 2 Performance across different video understanding, reasoning, and temporal grounding benchmarks. “LRR” refers to LongVideo-Reason-eval Benchmark. Open-o3-Video achieves comparable or even superior results to other video understanding and reasoning models, while providing more intuitive spatiotemporal evidence. Evaluation results on more benchmarks are provided in Appendix A.7.

Model VideoMME WorldSense VideoMMMU LRR TVGBench Avg

Overall Long Overall Recognition Overall Perception Acc mIoU

GPT-4o 71.9 - 42.6 - 61.2 66.0 - - -

VideoLLaMA3-7B 60.6 48.7 37.3 38.1 46.5 59.7 59.8 22.2 45.3 InternVL-2.5-8B 62.3 51.2 39.6 38.5 42.4 57.0 62.0 6.3 42.5 Qwen2.5-VL-7B (Base) 62.4 50.8 36.1 33.7 51.2 64.7 59.3 16.3 45.1 VideoRFT-7B 59.8 50.7 38.2 36.6 51.1 66.0 69.4 14.3 46.6 VideoR1-7B 61.4 50.6 35.5 32.8 52.4 65.3 68.9 9.6 45.6

Open-o3-Video-7B (Ours) 63.6 54.9 37.5 36.8 52.3 68.0 69.4 20.8 48.7 ∆ vs. Qwen2.5-VL-7B ↑ 1.2 ↑ 4.1 ↑ 1.4 ↑ 3.1 ↑ 1.1 ↑ 3.3 ↑ 10.1 ↑ 4.5 ↑ 3.6

two structured reasoning chains (“what–when–where” and “what–where–when”) and composite metrics that combine accuracy with temporal and spatial IoU, thereby enabling comprehensive evaluation of spatiotemporal reasoning. We further evaluate on broader video understanding benchmarks. VideoMME [18] and VideoMMMU [23] assess general video QA and multimodal comprehension across diverse domains, while WorldSense [22] emphasizes integrating multimodal signals with commonsense reasoning, and LongVideoReason-eval [8] evaluates long-range reasoning on videos. In addition, TVGBench [52] focuses on fine-grained temporal localization, STAR [54] tests situated reasoning, and CameraBench [31] measures robustness under diverse camera motions.

#### 5.1 Main Results

Results on V-STAR. On the V-STAR benchmark, we compare our method with three groups of baselines: (i) closed-source commercial models such as GPT-4o [35] and Gemini-2-Flash [44], which represent the current frontier of proprietary video LLMs. (ii) open-source general-purpose video understanding models, including Video-LLAMA3 [59], LLaVA-Video [65], VideoChat2 [27], Oryx-1.5-7B [32], InternVL-2.5-8B [9], and Qwen2.5VL-7B [3]. (iii) task-specialized approaches such as TRACE [20], designed for temporal video grounding, and Sa2VA [58], optimized for fine-grained spatial grounding. As summarized in Table 1, our model consistently outperforms the baseline across different evaluation dimensions. In video question answering (What), our model achieves an accuracy of 61.03, representing a +27.6% point improvement over Qwen2.5-VL-7B. For temporal grounding (When), we report strong gains on both reasoning chains: Chain1 (what–when–where) improves by +9.1% points and Chain2 (what–where–when) by +10.2% points, showing robust performance regardless of the reasoning order. For spatial grounding (Where), our method surpasses the baseline by +8.4% points on Chain1 and +3.5% points on Chain2. Overall, compared with the Qwen2.5-VL baseline, our model improves performance by +14.4% mAM and +24.2% mLGM on V-STAR. It further surpasses proprietary models such as GPT-4o [35] and Gemini-2-Flash [12], achieving state-of-the-art performance. By extracting key frames and precise bounding boxes, Open-o3-Video brings o3-style, evidence-guided reasoning to videos, supplying more reliable and verifiable visual evidence during inference. We further train our framework on Qwen3-VL [2] models and observe consistent gains on V-STAR. Detailed results are provided in Appendix A.6.

Results on General Video Understanding and Temporal Grounding Benchmarks. We further evaluate our method on a broad suite of video understanding benchmarks, comparing against three categories of baselines: (i) closed-source models such as GPT-4o [35], (ii) open-source general video LLMs including Qwen2.5-VL-7B [3], Video-LLAMA3 [59], and InternVL-2.5-8B [9], and (iii) recent reasoning-focused models such as VideoRFT-7B [50] and VideoR1-7B [17], which treat video understanding as text-only reasoning. In contrast, our method combines reasoning with explicit spatio-temporal grounding, enabling evidence-based inference. As shown in Table 2, Open-o3-Video achieves consistent improvements across all datasets. Across VideoMME, WorldSense, and VideoMMMU, our model shows consistent gains over Qwen2.5-VL-7B, with

Table 3 Ablation on Different Training Strategies.

Setting mAM mLGM

Baseline (Qwen2.5-VL-7B) 19.3 22.4 Pure SFT 28.5 37.1 Pure RL (GSPO) 30.4 40.7 SFT+RL (GRPO) 32.8 45.3 SFT+RL (GSPO) (Ours) 33.7 46.6

Table 5 Impact of Two Reward Designs in the Thinking Reward.

Ada. Gat. mAM mLGM

× ✓ 33.0 45.2 ✓ × 32.3 44.9 ✓ ✓ 33.7 46.6

Table 4 Impact of different spatio-temporal grounded reasoning data for training data. More ablations about training data are provided in the Appendix A.3.

Training data mAM mLGM

w/o spatio-temporal data 28.3 36.2 + VideoEspresso 31.1 43.6 + Our annotated data 33.7 46.6

Table 6 Effectiveness of the extracted temporal evidence.

Setting Accuracy (%)

Uniform sampling (64 frames) 68.7 Removed predicted evidence 66.0 (-2.7) Random removal 68.0 (-0.7)

notable improvements on long videos (+4.1%) and perception-related tasks (+3.1% on WorldSense recognition and +3.3% on VideoMMMU perception), highlighting enhanced temporal reasoning and perceptual grounding. For long-range video reasoning, our model achieves 69.4% accuracy on the LongVideoReason-eval benchmark (LRR), and outperforms the baseline by +10.1%. Compared with dedicated video reasoning methods, our model achieves results comparable to or even superior to theirs while providing more interpretable evidence in its reasoning process. On TVGBench, which directly measures temporal grounding, our model surpasses the baseline by a large margin (+4.5% mIoU), indicating gains in temporal localization. These results show that our approach maintains the QA strength of general video LLMs while enhancing the spatio-temporal grounding capability.

#### 5.2 Ablation and Analysis

Training strategy: RL provides larger gains than SFT, while their combination yields the best results, with GSPO offering the most stable improvements. As shown in Table 3, on the V-STAR benchmark, both SFT and RL substantially improve grounding over the base model. RL outperforms SFT (+2.1% mAM, +4.6% mLGM) by directly optimizing temporal and spatial alignment, while SFT ensures stable reasoning formats and basic grounding under supervision. Their combination is highly synergistic, reaching 33.7% mAM and 46.6% mLGM. Within this joint training, GSPO further surpasses GRPO (+0.9% mAM, +1.3% mLGM) by providing more stable rewards and better long-horizon temporal localization (+2.9% Chain1 tIoU). More ablations about training are provided in Appendix A.5.

Reward design: Both adaptive temporal proximity and temporal gating are effective. In the thinking reward, we introduce two mechanisms: adaptive temporal proximity (Ada.) and temporal gating (Gat.). To validate their effectiveness, we conduct ablation experiments on the V-STAR benchmark, shown in Table 5. Removing the proximity reward reduces performance by 0.7% mAM and 1.4% mLGM, showing that adaptive scaling helps the model better align predicted timestamps with annotated windows. Removing temporal gating causes larger drops of 1.4% mAM and 1.7% mLGM, confirming that gating is crucial for filtering irrelevant segments and preventing noisy spatial boxes. These results verify that our reward design effectively couples temporal and spatial grounding, leading to strong performance.

Training data: High-quality spatio-temporal annotations boost grounding. Without spatio-temporal supervision, the model exhibits substantially weaker performance, underscoring the necessity of Spatio-temporal annotations for effective grounding. As shown in Table 4, incorporating 9.6k filtered and rewritten VideoEspresso [21] samples enables the model to perform basic spatio-temporal reasoning, leading to improvements of +2.8% mAM and +7.4% mLGM on V-STAR. Moreover, further adding our spatio-temporal annotations can yield gains of +5.4% mAM and +10.4% mLGM. This shows the effectiveness of our annotation pipeline

and the critical role of high-quality spatio-temporal supervision.

Evidence faithfulness: Generated spatio-temporal evidence is concise and informative. We analyze the generated evidence on a randomly sampled VideoMME subset of 150 questions. On average, the model produces 1.15 bounding boxes and 1.37 timestamps per instance. To assess the relevance of the identified evidence, we remove two frames temporally closest to each predicted timestamp. As shown in Table 6, removing evidence-related frames causes a larger performance drop than randomly removing the same number of frames, indicating that the generated evidence can capture some informative visual signals.

Test-time scaling with grounded evidence: Confidence-aware voting with Open-o3-Video outperforms naive majority voting. Inspired by the scoring and adaptive voting mechanisms for video reasoning in CyberV [33], we introduce a confidence-aware voting scheme that leverages grounded evidence to verify predictions at inference, as shown in Figure 4 in the appendix. Details, including scoring schemes, prompts, and results on WorldSense and VideoMMMU, are provided in Appendix A.10.

### 6 Conclusion

We introduce Open-o3-Video, a grounded video reasoning framework that enables a single model to generate explicit spatio-temporal evidence (timestamps and bounding boxes) as part of its reasoning process, without relying on external models or tools. To the best of our knowledge, we are the first to achieve this ability in Video MLLMs. With carefully curated high-quality training data, a two-stage strategy combining supervised fine-tuning and GSPO-based reinforcement learning, and novel thinking rewards incorporating adaptive temporal proximity and temporal gating, our method substantially improves answer accuracy, temporal alignment, and spatial grounding. Comprehensive experiments demonstrate that Open-o3-Video achieves state-of-the-art performance on the V-STAR benchmark, surpassing strong baselines including GPT-4o, while remaining broadly competitive across diverse video understanding tasks. Despite these results, our approach still has some limitations, which we discuss together with potential future directions in Appendix A.12.

### References

- [1] Lisa Anne Hendricks, Oliver Wang, Eli Shechtman, Josef Sivic, Trevor Darrell, and Bryan Russell. Localizing moments in video with natural language. In Proceedings of the IEEE international conference on computer vision, pages 5803–5812, 2017.

- [2] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao Chunjiang Ge, Wenbin Ge, Zhifang Guo, Qidong Huang, Jie Huang, Fei Huang, Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng Li, Mei Li, Kaixin Li, Zicheng Lin, Junyang Lin, Xuejing Liu, Jiawei Liu, Chenglong Liu, Yang Liu, Dayiheng Liu, Shixuan Liu, Dunjie Lu, Ruilin Luo, Chenxu Lv, Rui Men, Lingchen Meng, Xuancheng Ren, Xingzhang Ren, Sibo Song, Yuchong Sun, Jun Tang, Jianhong Tu, Jianqiang Wan, Peng Wang, Pengfei Wang, Qiuyue Wang, Yuxuan Wang, Tianbao Xie, Yiheng Xu, Haiyang Xu, Jin Xu, Zhibo Yang, Mingkun Yang, Jianxin Yang, An Yang, Bowen Yu, Fei Zhang, Hang Zhang, Xi Zhang, Bo Zheng, Humen Zhong, Jingren Zhou, Fan Zhou, Jing Zhou, Yuanzhi Zhu, and Ke Zhu. Qwen3-vl technical report, 2025. URL https://arxiv.org/abs/2511.21631.
- [3] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.

- [4] Apratim Bhattacharyya, Sunny Panchal, Mingu Lee, Reza Pourreza, Pulkit Madan, and Roland Memisevic. Look, remember and reason: Grounded reasoning in videos with language models. arXiv preprint arXiv:2306.17778, 2023.

- [5] Fabian Caba Heilbron, Victor Escorcia, Bernard Ghanem, and Juan Carlos Niebles. Activitynet: A large-scale video benchmark for human activity understanding. In Proceedings of the ieee conference on computer vision and pattern recognition, pages 961–970, 2015.

- [6] Ruizhe Chen, Zhiting Fan, Tianze Luo, Heqing Zou, Zhaopeng Feng, Guiyang Xie, Hansheng Zhang, Zhuochen Wang, Zuozhu Liu, and Huaijian Zhang. Datasets and recipes for video temporal grounding via reinforcement learning. arXiv preprint arXiv:2507.18100, 2025.

- [7] Yukang Chen, Fuzhao Xue, Dacheng Li, Qinghao Hu, Ligeng Zhu, Xiuyu Li, Yunhao Fang, Haotian Tang, Shang Yang, Zhijian Liu, et al. Longvila: Scaling long-context visual language models for long videos. arXiv preprint arXiv:2408.10188, 2024.

- [8] Yukang Chen, Wei Huang, Baifeng Shi, Qinghao Hu, Hanrong Ye, Ligeng Zhu, Zhijian Liu, Pavlo Molchanov, Jan Kautz, Xiaojuan Qi, et al. Scaling rl to long videos. arXiv preprint arXiv:2507.07966, 2025.

- [9] Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, et al. Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling. arXiv preprint arXiv:2412.05271, 2024.

- [10] Zixu Cheng, Jian Hu, Ziquan Liu, Chenyang Si, Wei Li, and Shaogang Gong. V-star: Benchmarking video-llms on video spatio-temporal reasoning. arXiv preprint arXiv:2503.11495, 2025.

- [11] Jang Hyun Cho, Andrea Madotto, Effrosyni Mavroudi, Triantafyllos Afouras, Tushar Nagarajan, Muhammad Maaz, Yale Song, Tengyu Ma, Shuming Hu, Suyog Jain, et al. Perceptionlm: Open-access data and models for detailed visual understanding. arXiv preprint arXiv:2504.13180, 2025.

- [12] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.

- [13] Jisheng Dang, Jingze Wu, Teng Wang, Xuanhui Lin, Nannan Zhu, Hongbo Chen, Wei-Shi Zheng, Meng Wang, and Tat-Seng Chua. Reinforcing video reasoning with focused thinking. arXiv preprint arXiv:2505.24718, 2025.

- [14] Yang Ding, Yizhen Zhang, Xin Lai, Ruihang Chu, and Yujiu Yang. Videozoomer: Reinforcement-learned temporal focusing for long video reasoning. arXiv preprint arXiv:2512.22315, 2025.

- [15] Yue Fan, Xuehai He, Diji Yang, Kaizhi Zheng, Ching-Chen Kuo, Yuting Zheng, Sravana Jyothi Narayanaraju, Xinze Guan, and Xin Eric Wang. Grit: Teaching mllms to think with images. arXiv preprint arXiv:2505.15879, 2025.

- [16] Kaituo Feng, Kaixiong Gong, Bohao Li, Zonghao Guo, Yibing Wang, Tianshuo Peng, Benyou Wang, and Xiangyu Yue. Video-r1: Reinforcing video reasoning in mllms. arXiv preprint arXiv:2503.21776, 2025.

- [17] Kaituo Feng, Kaixiong Gong, Bohao Li, Zonghao Guo, Yibing Wang, Tianshuo Peng, Junfei Wu, Xiaoying Zhang, Benyou Wang, and Xiangyu Yue. Video-r1: Reinforcing video reasoning in mllms. arXiv preprint arXiv:2503.21776, 2025.

- [18] Chaoyou Fu, Yuhan Dai, Yondong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. In CVPR, 2025.

- [19] Xin Gu, Haoji Zhang, Qihang Fan, Jingxuan Niu, Zhipeng Zhang, Libo Zhang, Guang Chen, Fan Chen, Longyin Wen, and Sijie Zhu. Thinking with bounding boxes: Enhancing spatio-temporal video grounding via reinforcement fine-tuning. arXiv preprint arXiv:2511.21375, 2025.

- [20] Yongxin Guo, Jingyu Liu, Mingda Li, Qingbin Liu, Xi Chen, and Xiaoying Tang. Trace: Temporal grounding video llm via causal event modeling. arXiv preprint arXiv:2410.05643, 2024.

- [21] Songhao Han, Wei Huang, Hairong Shi, Le Zhuo, Xiu Su, Shifeng Zhang, Xu Zhou, Xiaojuan Qi, Yue Liao, and Si Liu. Videoespresso: A large-scale chain-of-thought dataset for fine-grained video reasoning via core frame selection. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 26181–26191, 2025.

- [22] Jack Hong, Shilin Yan, Jiayin Cai, Xiaolong Jiang, Yao Hu, and Weidi Xie. Worldsense: Evaluating real-world omnimodal understanding for multimodal llms. arXiv preprint arXiv:2502.04326, 2025.

- [23] Kairui Hu, Penghao Wu, Fanyi Pu, Wang Xiao, Yuanhan Zhang, Xiang Yue, Bo Li, and Ziwei Liu. Video-mmmu: Evaluating knowledge acquisition from multi-discipline professional videos. arXiv preprint arXiv:2501.13826, 2025.

- [24] Yang Jin, Zehuan Yuan, Yadong Mu, et al. Embracing consistency: A one-stage approach for spatio-temporal video grounding. Advances in Neural Information Processing Systems, 35:29192–29204, 2022.

- [25] Jie Lei, Tamara L Berg, and Mohit Bansal. Detecting moments and highlights in videos via natural language queries. Advances in Neural Information Processing Systems, 34:11846–11858, 2021.

- [26] Hongyu Li, Jinyu Chen, Ziyu Wei, Shaofei Huang, Tianrui Hui, Jialin Gao, Xiaoming Wei, and Si Liu. Llava-st: A multimodal large language model for fine-grained spatial-temporal understanding. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 8592–8603, 2025.

- [27] Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Luo, et al. Mvbench: A comprehensive multi-modal video understanding benchmark. In CVPR, pages 22195–22206, 2024.

- [28] Xinhao Li, Ziang Yan, Desen Meng, Lu Dong, Xiangyu Zeng, Yinan He, Yali Wang, Yu Qiao, Yi Wang, and Limin Wang. Videochat-r1: Enhancing spatio-temporal perception via reinforcement fine-tuning. arXiv preprint arXiv:2504.06958, 2025.

- [29] Zeqian Li, Shangzhe Di, Zhonghua Zhai, Weilin Huang, Yanfeng Wang, and Weidi Xie. Universal video temporal grounding with generative multi-modal large language models. arXiv preprint arXiv:2506.18883, 2025.

- [30] Shuo Liang, Yiwu Zhong, Zi-Yuan Hu, Yeyao Tao, and Liwei Wang. Fine-grained spatiotemporal grounding on egocentric videos. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 9385–9395, 2025.

- [31] Zhiqiu Lin, Siyuan Cen, Daniel Jiang, Jay Karhade, Hewei Wang, Chancharik Mitra, Tiffany Ling, Yuhan Huang, Sifan Liu, Mingyu Chen, et al. Towards understanding camera motions in any video. arXiv preprint arXiv:2504.15376, 2025.

- [32] Zuyan Liu, Yuhao Dong, Ziwei Liu, Winston Hu, Jiwen Lu, and Yongming Rao. Oryx mllm: On-demand spatial-temporal understanding at arbitrary resolution. arXiv preprint arXiv:2409.12961, 2024.

- [33] Jiahao Meng, Shuyang Sun, Yue Tan, Lu Qi, Yunhai Tong, Xiangtai Li, and Longyin Wen. Cyberv: Cybernetics for test-time scaling in video understanding. arXiv preprint arXiv:2506.07971, 2025.

- [34] Andreea-Maria Oncescu, Joao F Henriques, Yang Liu, Andrew Zisserman, and Samuel Albanie. Queryd: A video dataset with high-quality text and audio narrations. In ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 2265–2269. IEEE, 2021.

- [35] OpenAI. Hello gpt4-o. https://openai.com/index/hello-gpt-4o/, 2024.
- [36] OpenAI. Openai-o3. https://openai.com/index/introducing-o3-and-o4-mini/, 2025.

- [37] Kun Ouyang, Yuanxin Liu, Haoning Wu, Yi Liu, Hao Zhou, Jie Zhou, Fandong Meng, and Xu Sun. Spacer: Reinforcing mllms in video spatial reasoning. arXiv preprint arXiv:2504.01805, 2025.

- [38] Kun Ouyang, Yuanxin Liu, Linli Yao, Yishuo Cai, Hao Zhou, Jie Zhou, Fandong Meng, and Xu Sun. Conan: Progressive learning to reason like a detective over multi-scale visual evidence. arXiv preprint arXiv:2510.20470, 2025.

- [39] Jinyoung Park, Jeehye Na, Jinyoung Kim, and Hyunwoo J Kim. Deepvideo-r1: Video reinforcement fine-tuning via difficulty-aware regressive grpo. arXiv preprint arXiv:2506.07464, 2025.

- [40] Hao Shao, Shengju Qian, Han Xiao, Guanglu Song, Zhuofan Zong, Letian Wang, Yu Liu, and Hongsheng Li. Visual cot: Advancing multi-modal language models with a comprehensive dataset and benchmark for chain-of-thought reasoning. Advances in Neural Information Processing Systems, 37:8612–8642, 2024.

- [41] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Y Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

- [42] Xi Tang, Jihao Qiu, Lingxi Xie, Yunjie Tian, Jianbin Jiao, and Qixiang Ye. Adaptive keyframe sampling for long video understanding. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 29118–29128, 2025.

- [43] Yansong Tang, Dajun Ding, Yongming Rao, Yu Zheng, Danyang Zhang, Lili Zhao, Jiwen Lu, and Jie Zhou. Coin: A large-scale dataset for comprehensive instructional video analysis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1207–1216, 2019.

- [44] Gemini Team, Petko Georgiev, Ving Ian Lei, Ryan Burnell, Libin Bai, Anmol Gulati, Garrett Tanzer, Damien Vincent, Zhufeng Pan, Shibo Wang, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530, 2024.

- [45] Kwai Keye Team, Biao Yang, Bin Wen, Changyi Liu, Chenglong Chu, Chengru Song, Chongling Rao, Chuan Yi, Da Li, Dunju Zang, et al. Kwai keye-vl technical report. arXiv preprint arXiv:2507.01949, 2025.

- [46] Haochen Wang, Anlin Zheng, Yucheng Zhao, Tiancai Wang, Zheng Ge, Xiangyu Zhang, and Zhaoxiang Zhang. Reconstructive visual instruction tuning. arXiv preprint arXiv:2410.09575, 2024.

- [47] Haochen Wang, Xiangtai Li, Zilong Huang, Anran Wang, Jiacong Wang, Tao Zhang, Jiani Zheng, Sule Bai, Zijian Kang, Jiashi Feng, et al. Traceable evidence enhanced visual grounded reasoning: Evaluation and methodology. arXiv preprint arXiv:2507.07999, 2025.

- [48] Haochen Wang, Yucheng Zhao, Tiancai Wang, Haoqiang Fan, Xiangyu Zhang, and Zhaoxiang Zhang. Ross3d: Reconstructive visual instruction tuning with 3d-awareness. arXiv preprint arXiv:2504.01901, 2025.

- [49] Jiacong Wang, Zijian Kang, Haochen Wang, Haiyong Jiang, Jiawen Li, Bohong Wu, Ya Wang, Jiao Ran, Xiao Liang, Chao Feng, et al. Vgr: Visual grounded reasoning. arXiv preprint arXiv:2506.11991, 2025.

- [50] Qi Wang, Yanrui Yu, Ye Yuan, Rui Mao, and Tianfei Zhou. Videorft: Incentivizing video reasoning capability in mllms via reinforced fine-tuning. arXiv preprint arXiv:2505.12434, 2025.

- [51] Shihao Wang, Guo Chen, De-an Huang, Zhiqi Li, Minghan Li, Guilin Li, Jose M Alvarez, Lei Zhang, and Zhiding Yu. Videoitg: Multimodal video understanding with instructed temporal grounding. arXiv preprint arXiv:2507.13353, 2025.

- [52] Ye Wang, Ziheng Wang, Boshen Xu, Yang Du, Kejun Lin, Zihan Xiao, Zihao Yue, Jianzhong Ju, Liang Zhang, Dingyi Yang, et al. Time-r1: Post-training large vision language model for temporal video grounding. arXiv preprint arXiv:2503.13377, 2025.

- [53] Ziyang Wang, Jaehong Yoon, Shoubin Yu, Md Mohaiminul Islam, Gedas Bertasius, and Mohit Bansal. Video-rts: Rethinking reinforcement learning and test-time scaling for efficient and enhanced video reasoning. arXiv preprint arXiv:2507.06485, 2025.

- [54] Bo Wu and Shoubin Yu Star. A benchmark for situated reasoning in real-world videos. Advances in Neural Information Processing Systems (NeurIPS), 3, 2024.

- [55] Yuan Xie, Tianshui Chen, Zheng Ge, and Lionel Ni. Video-mtr: Reinforced multi-turn reasoning for long video understanding. arXiv preprint arXiv:2508.20478, 2025.

- [56] Zuhao Yang, Sudong Wang, Kaichen Zhang, Keming Wu, Sicong Leng, Yifan Zhang, Bo Li, Chengwei Qin, Shijian Lu, Xingxuan Li, et al. Longvt: Incentivizing" thinking with long videos" via native tool calling. arXiv preprint arXiv:2511.20785, 2025.

- [57] Jiabo Ye, Haiyang Xu, Haowei Liu, Anwen Hu, Ming Yan, Qi Qian, Ji Zhang, Fei Huang, and Jingren Zhou. mplug-owl3: Towards long image-sequence understanding in multi-modal large language models. arXiv preprint arXiv:2408.04840, 2024.

- [58] Haobo Yuan, Xiangtai Li, Tao Zhang, Zilong Huang, Shilin Xu, Shunping Ji, Yunhai Tong, Lu Qi, Jiashi Feng, and Ming-Hsuan Yang. Sa2va: Marrying sam2 with llava for dense grounded understanding of images and videos. arXiv, 2025.

- [59] Boqiang Zhang, Kehan Li, Zesen Cheng, Zhiqiang Hu, Yuqian Yuan, Guanzheng Chen, Sicong Leng, Yuming Jiang, Hang Zhang, Xin Li, et al. Videollama 3: Frontier multimodal foundation models for image and video understanding. arXiv preprint arXiv:2501.13106, 2025.

- [60] Hang Zhang, Xin Li, and Lidong Bing. Video-llama: An instruction-tuned audio-visual language model for video understanding. arXiv preprint arXiv:2306.02858, 2023.

- [61] Haoji Zhang, Xin Gu, Jiawen Li, Chixiang Ma, Sule Bai, Chubin Zhang, Bowen Zhang, Zhichao Zhou, Dongliang He, and Yansong Tang. Thinking with videos: Multimodal tool-augmented reinforcement learning for long video reasoning. arXiv preprint arXiv:2508.04416, 2025.

- [62] Jinglei Zhang, Yuanfan Guo, Rolandos Alexandros Potamias, Jiankang Deng, Hang Xu, and Chao Ma. Vtimecot: Thinking by drawing for video temporal grounding and reasoning. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 24203–24213, 2025.

- [63] Peiyuan Zhang, Kaichen Zhang, Bo Li, Guangtao Zeng, Jingkang Yang, Yuanhan Zhang, Ziyue Wang, Haoran Tan, Chunyuan Li, and Ziwei Liu. Long context transfer from language to vision. arXiv preprint arXiv:2406.16852, 2024.

- [64] Xingjian Zhang, Siwei Wen, Wenjun Wu, and Lei Huang. Tinyllava-video-r1: Towards smaller lmms for video reasoning. arXiv preprint arXiv:2504.09641, 2025.

- [65] Yuanhan Zhang, Jinming Wu, Wei Li, Bo Li, Zejun Ma, Ziwei Liu, and Chunyuan Li. Video instruction tuning with synthetic data. arXiv preprint arXiv:2410.02713, 2024.

- [66] Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui Chen, Bowen Yu, Chang Gao, Kai Dang, Yuqiong Liu, Rui Men, An Yang, et al. Group sequence policy optimization. arXiv preprint arXiv:2507.18071, 2025.

- [67] Ziwei Zheng, Michael Yang, Jack Hong, Chenxiao Zhao, Guohai Xu, Le Yang, Chao Shen, and Xing Yu. Deepeyes: Incentivizing" thinking with images" via reinforcement learning. arXiv preprint arXiv:2505.14362, 2025.

- [68] Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Yuchen Duan, Hao Tian, Weijie Su, Jie Shao, et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025.

## Appendix

### A Appendix

Overview. This appendix provides additional details and analyses to complement the main paper.

- • Section A.1 provides additional implementation details.
- • Section A.2 discusses additional related works that extend the “thinking with images” paradigm to the video domain.
- • Section A.3 describes training dataset preparation and ablations on the ratio of general VideoQA data.
- • Section A.4 presents the prompts used for data annotation with Gemini.
- • Section A.5 provides more ablation studies on hyper-parameters and o3-style training objectives.
- • Section A.6 reports additional results based on Qwen3-VL models.
- • Section A.7 presents more experimental results on the STAR benchmark and CameraBench.
- • Section A.8 analyzes inference frame rate.
- • Section A.9 provides the full mathematical formulation of the GSPO algorithm.
- • Section A.10 details the confidence-aware test-time scaling procedure and reports additional results.
- • Section A.11 provides further qualitative visualizations of spatio-temporal reasoning.
- • Section A.12 discusses limitations of the current framework and directions for future work.

#### A.1 More Implementation Details

The training process of Open-o3-Video consists of two stages. In the cold-start stage, we train on the STGR-CoT-30k dataset for one epoch with a learning rate of 1 × 10−6. In the GSPO stage, we further train on the STGR-RL-36k dataset for one epoch, also with a learning rate of 1 × 10−6. For the thinking reward, the standard deviation parameter σ is annealed from 4 to 1 and then kept constant. The gating mechanism employs a temporal threshold τ of 3s. At test time, we employ the vLLM framework, requiring the model to first produce a spatio-temporal grounded reasoning process, followed by the final answer. For evaluation on V-STAR, we uniformly sample 16 frames per video, and for other video understanding benchmarks, we uniformly sample 64 frames. Additional comparisons and analyses on inference frame rates are provided in Appendix A.8.

#### A.2 More Related Works

In the second half of 2025, several concurrent works extend “thinking with images” to videos, mostly by improving temporal evidence seeking. VITAL [61] enables an agent to crop temporally relevant clips on demand via a visual toolbox, while LongVT [56] and VideoZoomer [14] iteratively invoke temporal zoom-in tools to retrieve relevant clips. Conan [38] teaches the model to identify evidence frames, perform cross-frame deduction, and decide whether to conclude or continue exploring the video. VTimeCoT [62] proposes a training-free “thinking by drawing” scheme that uses progress-bar tools to improve temporal grounding and reasoning. STVG-o1 [19] instead targets spatio-temporal video grounding, but is not designed as an end-to-end video QA method. Compared with these concurrent works, our method jointly integrates temporal localization, spatial grounding, and video reasoning within a single model, and performs single-round inference without external tool use, enabling native “thinking with frames” capability.

#### A.3 More Details and Ablation on Training Data

Data Preparation. Beyond reporting corpus sizes, we describe here the sampling and filtering strategy applied to each source. For temporal grounding data, we adopt strict constraints to ensure annotation quality and to keep the reasoning process manageable. Specifically, for TVG-Coldstart, we retain only samples with chain-of-thought length under 6,000 characters and with ground-truth spans covering less than 70% of the total video duration. The same filtering is applied to Time-R1, resulting in 2.3k samples. For additional temporal grounding video sources (ActivityNet [5], COIN [43], QueryD [34], QVHighlight [25], DiDeMo [1]), we keep videos of duration between 10 seconds and 3 minutes, further discarding those where the annotated action

Table 7 Impact of different amounts of general VideoQA data. 15k achieves the best balance between grounding and general QA performance.

VideoQA Data V-STAR (mAM) VideoMME (Acc)

w/o Video-R1 data 33.4 60.7 +5k 33.0 63.2 +15k 33.7 63.6 +30k 31.7 63.6

lasts more than 50% of the video; TVG-RL is filtered with the same rules, and 2.9k samples are randomly selected. For spatial grounding data, we randomly sample 5k instances from both TreeVGR-SFT and VisCoT. For general video QA data, 15k Video-R1 samples are randomly drawn without additional filtering. For PLM-based video dense captioning data (PLM-Rdcap), we initially sample 3k videos for annotation, from which 2k remain after filtering for quality and consistency. This careful selection yields a high-quality dataset that balances temporal, spatial, and general reasoning tasks. The resulting dataset provides diverse yet clean supervision signals, making it particularly suitable for training and evaluating spatio-temporal reasoning models.

Ablation on Different Ratios of General VideoQA Data. To enhance the model’s grounding ability, we emphasize temporal and spatial grounding data during training. However, excessive focus on grounding may weaken the model’s original strength in general VideoQA. Thus, an important design choice is how much general VideoQA data to include in the STGR dataset. We compare different ratios and evaluate performance on both grounding-oriented (V-STAR) and QA-oriented (VideoMME) benchmarks. As shown in Table 7, adding 15k VideoQA samples from Video-R1 [17] significantly improves QA accuracy without harming grounding performance. In contrast, adding 30k yields no further QA gain while slightly reducing grounding accuracy. Therefore, we adopt 15k VideoQA samples as a balanced choice, offering strong QA capability while preserving grounding ability and maintaining training efficiency.

#### A.4 Prompt for Data Annotation

To obtain high-quality spatio-temporal annotations, we design structured prompts for the Gemini 2.5 Pro API, separately tailored to the two data sources described in Section 3: PLM-Rdcap data and temporal grounding datasets. The goal of these prompts is to guide the model to produce question-answer pairs, key frame selection, bounding boxes, and reasoning chains in a consistent JSON format.

For PLM-Rdcap, as shown in Figure 5, the input is the dense video captions and total frame count, and the output is a JSON with question, answer, key_frames, and reasoning_process. Since only frame indices are given, we post-process them into timestamps and align reasoning mentions with annotated object names and boxes.

For temporal grounding datasets, as shown in Figure 6, the input includes the annotated segment, video duration, and segment descriptions, and the output JSON contains the question, answer, key_frames with timestamps, objects and boxes, and the spatio-temporal grounded reasoning_process.

We further apply strict filtering and consistency checks, retaining only annotations with validated boxes, aligned timestamps, and coherent reasoning. This ensures a high-quality dataset with reliable spatio-temporal evidence, essential for robust training and evaluation.

#### A.5 More Ablation Studies on Hyper-parameters and Training Objectives

Adaptive temporal proximity parameter. As described in the main paper, we adopt an adaptive temporal proximity schedule in the temporal grounding reward, where the parameter σ is gradually annealed from a larger value to a smaller one during training. For a single instance, the temporal reward for point supervision can be written as

x2 2σ2

, (6)

rt = exp −

Table 8 Ablation on the adaptive temporal proximity parameter σ on V-STAR.

Setting mAM mLGM

Open-o3-Video (adaptive σ) 33.7 46.6 Fixed σ = 1.0 32.6 44.5 Fixed σ = 4.0 33.0 45.2

Table 9 Performance comparison between Open-o3-Video and a non-o3-like variant (pure textual reasoning).

Model V-STAR (what) V-STAR (mLGM) VideoMME VideoMME (long) WorldSense WorldSense (Rec.)

Open-o3-Video 61.0 46.6 63.6 54.9 37.5 36.8 Non-o3-like 58.6 41.0 62.3 52.8 37.1 36.4

where x denotes the absolute difference between the predicted timestamp and the ground-truth timestamp. When σ is small (e.g., σ = 1), the reward rapidly decays for moderate temporal errors, leading to sparse rewards during early training stages. In contrast, a large σ (e.g., σ = 4) assigns relatively high rewards to imprecise predictions, weakening the learning signal for fine-grained temporal refinement. The adaptive schedule alleviates both issues by allowing the model to transition from coarse temporal alignment to precise localization. Empirically, both fixed σ = 1 and σ = 4 underperform the adaptive schedule on V-STAR, validating the effectiveness of the proposed design.

Comparison with a non-o3-like variant. We further study the importance of o3-style spatio-temporal grounded reasoning by comparing Open-o3-Video with a non-o3-like variant. The non-o3-like model removes spatiotemporal evidence from SFT and does not receive thinking rewards during reinforcement learning, resulting in a text-only reasoning model. As shown in Table 9, removing o3-style grounding consistently degrades performance across multiple benchmarks, including V-STAR, VideoMME, and WorldSense. The degradation is particularly evident on grounding-intensive settings, such as V-STAR mLGM and long video reasoning in VideoMME, indicating that incorporating spatio-temporal supervision and grounding-aware rewards during training plays a critical role in improving grounded video reasoning performance.

#### A.6 Results based on Qwen3-VL Models

Beyond the main results on Qwen2.5-VL, we apply the same training data and optimization pipeline to Qwen3-VL models and report the results in Table 10. We observe that Qwen3-VL already exhibits basic spatio-temporal grounding alongside strong high-level understanding. Building on this foundation, our method yields consistent improvements across model scales: mAM/mLGM increase by +2.7%/+3.6% on the 4B model, +7.5%/+14.6% on the 8B model, and +4.0%/+8.1% on the 32B model. These results demonstrate that Open-o3-Video can strengthen spatio-temporal grounded reasoning on top of increasingly strong MLLMs.

#### A.7 Results on More Benchmarks

As shown in Table 11, Open-o3-Video performs better than the base model on both STAR and CameraBench. On STAR, it improves accuracy by 3.2%, demonstrating that Open-o3-Video can better handle situated reasoning tasks involving spatio-temporal cues. We also evaluate our model on the CameraBench VQA task and compare it with the baseline model as well as models trained without adaptive temporal proximity or without temporal gating. We find that our model performs better than the baseline and shows gains in challenging motion settings, such as confusable motion, motion and steadiness, and different motion speeds. It also outperforms the variants without Ada. or without Gat. These results indicate that both the model and the training techniques remain stable under camera motions that differ from the training data distribution.

#### A.8 Ablation Studies on Inference Frame Rate

We analyze the effect of inference frame rate on long-video understanding using the LongVideo-Reason-eval benchmark, as shown in Table 12. For the comparison between high and low frame rates, we find that higher

Table 10 Results on V-STAR based on Qwen3-VL models.

Model What When (Temporal IoU) Where (Visual IoU) Overall

Acc Chain1 Chain2 Chain1 Chain2 mAM mLGM

Qwen3-VL-4B 59.6 18.6 19.2 26.8 7.7 31.9 43.7 Open-o3-Video-4B 59.7 25.1 23.8 31.2 8.1 34.6 47.3

Qwen3-VL-8B 42.9 23.5 24.4 27.3 7.1 28.0 34.4 Open-o3-Video-8B 61.1 24.9 23.5 33.2 8.8 35.5 49.0

Qwen3-VL-32B 47.5 24.5 25.7 30.9 7.5 33.9 45.6 Open-o3-Video-32B 64.6 26.4 26.0 33.6 11.9 37.9 53.7

Table 11 Performance on STAR and CameraBench.

Models STAR CameraBench VQA

Overall Overall Confusable Motion Motion and Steadiness Motion Speed

Qwen2.5-VL-7B 67.3 57.5 49.3 56.7 69.0 Open-o3-Video-7B 70.5 58.8 51.3 57.6 69.3 w/o Ada. 70.1 58.5 50.0 56.7 68.7 w/o Gat. 69.6 57.8 50.3 55.9 67.0

frame rates (64 frames) give some improvement, but even with only 16 frames, our model performs well and surpasses the baseline and other reasoning models. And increasing the number of frames from 16 to 64 leads to only marginal gains (e.g., 69.2% → 69.4%), indicating diminishing returns from denser frame sampling. For variable frame rates, we follow AKS [42] and use the key-frame selection strategy. This strategy achieves 70.1% accuracy, demonstrating that key-frame sampling offers a small improvement over uniform sampling when spatio-temporal reasoning is involved.

#### A.9 Details of GSPO Training

For completeness, we provide the full formulation of Group Sequence Policy Optimization (GSPO) [66], which is used in our reinforcement learning stage.

Given a query x, the model generates a group of G candidate responses {yi}Gi=1 sampled from the old policy πθ

(·|x). Each response is scored by a reward function r(x,yi), and its normalized advantage is computed as

old

r(x,yi) − mean({r(x,yj)}Gj=1) std({r(x,yj)}Gj=1)

Aˆi =

. (7)

The importance ratio is defined at the sequence level as

  1

1 |yi|

|yi|

πθ(yi|x) πθ

si(θ) =

= exp

|yi|

(yi|x)

old

t=1

where |yi| denotes the response length. The GSPO objective is then

 , (8)

πθ(yi,t|x,yi,<t) πθ

log

(yi,t|x,yi,<t)

old

JGSPO(θ) = Ex,{y

i}∼πθold

with ϵ controlling the clipping range.

G

1 G

min si(θ)Aˆi, clip(si(θ), 1 − ϵ, 1 + ϵ)Aˆi , (9)

i=1

Unlike GRPO, which clips per-token updates, GSPO clips entire responses, thereby aligning reward assignment with optimization granularity. In practice, this leads to more stable gradients and better performance on long chain-of-thought reasoning tasks.

Table 12 Ablation on inference frame rate on LongVideo-Reason-eval.

Models Qwen2.5-VL Video-R1 VideoRFT Open-o3-Video

Number of Frames 64 64 16 64 16 64 64 (+AKS) 16 LongVideo-Reason-eval 59.3 68.9 67.3 69.4 68.0 69.4 70.1 69.2

Table 13 Test-time scaling results on WorldSense and VideoMMMU, showing that the confidence-aware voting (N=8) with grounded evidence consistently outperforms base model (N=1) and naive majority voting (N=8).

Setting WorldSense VideoMMMU

Base 37.5 52.3 Majority Voting 37.3 53.1 Confidence-aware Voting 38.5 54.1

#### A.10 More Details about Test Time Scaling To further enhance robustness at inference, we adopt a confidence-aware test-time scaling procedure,

- as shown in Figure 4. Given a video question, the model first generates N independent responses in parallel (In our experiments, N = 8, with temperature set to 1.0). Each response contains spatio-temporal grounding annotations in the format <obj>...</obj><box>...</box>at<t>...</t>s, from which we extract the predicted bounding boxes. The corresponding regions are then cropped from the original video frames and paired with the question to form a new input. This input is passed back into the model to obtain a confidence score s ∈ {0,1,2}, where:

- • s = 2: the cropped evidence is highly supportive for answering the question,
- • s = 1: the evidence may be partially useful,
- • s = 0: the evidence is irrelevant.

Each initial response is assigned a confidence-weighted score by averaging its evidence scores across all mentioned objects. The final prediction is selected via weighted voting over the N responses. This process effectively filters out hallucinated reasoning traces and highlights consistent evidence across responses.

As reported in Table 13, confidence-aware voting consistently improves over naive majority voting, achieving +1.0 on WorldSense and +1.0 on VideoMMMU. This demonstrates that our o3-like spatio-temporal evidence not only enhances grounding, but also provides a natural mechanism for scalable inference and self-correction

- at test time.

#### A.11 More Visualizations

As shown in Figure 7,8,9,10, we provide additional qualitative examples to illustrate the spatio-temporal reasoning ability of Open-o3-Video. These visualizations demonstrate that our model can obtain spatiotemporal evidence and achieve better results.

#### A.12 Limitations and Future Works

While our framework demonstrates strong performance, several limitations remain. First, handling longer videos with complex scenes and smaller objects remains challenging, as high-quality spatiotemporal data for such cases remains relatively scarce. Second, reasoning-intensive queries that require multi-step inference beyond direct grounding remain difficult to fully address. Finally, our current design does not integrate audio or speech information, which often carries crucial cues for understanding video content. Future work will focus on extending the approach to longer and more complex videos, enriching supervision for fine-grained object grounding, and unifying multimodal signals, including speech, to further enhance logical reasoning.

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

Figure 4 Illustration of our confidence-aware test-time scaling. The model generates multiple responses with spatio-temporal traces, from which visual regions are cropped and scored for evidence relevance (s ∈ {0, 1, 2}). Final predictions are obtained by confidence-weighted voting. Unlike naive majority voting, which is misled by spurious patterns (predicting “C”), our method highlights consistent supportive evidence and correctly predicts “A”, thereby improving robustness in inference.

Figure 5 Annotation Prompt for PLM-Video-Human Region Dense Temporal Captioning Data source.

###### Figure 6 Annotation Prompt for Temporal Grounding Data Source.

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

###### Figure 7 Visualization. On simple appearance perception tasks, both our model and related baselines can provide correct answers; however, our approach additionally offers explicit spatio-temporal evidence.

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

###### Figure 8 Visualization. For action recognition, our model precisely localizes both the time and location of the action, achieving superior performance compared to Video-R1.

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

###### Figure 9 Visualization. In weather reasoning tasks, our model identifies more effective supporting evidence, whereas related video reasoning models perform poorly.

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

###### Figure 10 Visualization. For open-ended QA, in this example, Video-RFT produces an incorrect analysis, whereas Open-o3-Video answers correctly and provides supporting visual evidence.

