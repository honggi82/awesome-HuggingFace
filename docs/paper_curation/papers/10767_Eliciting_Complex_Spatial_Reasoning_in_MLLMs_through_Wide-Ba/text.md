# arXiv:2606.03577v1[cs.CV]2Jun2026

## Eliciting Complex Spatial Reasoning in MLLMs through Wide-Baseline Matching

Hao Zhong1,2* Muzhi Zhu1,2* Shenyan Zeng1* Anzhou Li2 Cong Chen1,2 Hua Geng1 Duochao Shi1 Wentao Ye2 Tao Lin3,2† Hao Chen1† Chunhua Shen1,2†

1 State Key Laboratory of CAD & CG, Zhejiang University 2 Ant Group 3 Westlake University

#### Abstract

Wide-baseline matching (WBM) requires integrating geometric understanding, viewpoint changes, fine-grained perception, and occlusion reasoning, making it a challenging testbed for spatial reasoning in multimodal large language models (MLLMs) deployed in physical environments. However, current MLLMs lack systematic evaluation and training frameworks for these capabilities. We introduce ReasonMatch-Bench, a benchmark stratified by viewpoint displacement and matching granularity across indoor, outdoor, and object-centric scenarios, and show that current MLLMs still struggle with fine-grained wide-baseline correspondence: on a difficult 90-sample subset, human annotators achieve 84.0 F1, while the best existing baseline reaches 37.2. To bridge this gap, we build a scalable data-generation pipeline that automatically extracts wide-baseline view pairs from large-scale video-3D corpora, including RGB-D videos and SfM reconstructions, yielding diverse and verifiable supervision. We further propose Dynamic Correspondence Reinforcement Learning (DCRL), which combines Image-Level Viewpoint Progression and Point-Level Correspondence Curriculum to improve WBM training through verifiable rewards without explicit CoT supervision. Extensive experiments show that DCRL substantially improves ReasonMatch-Bench and transfers to related spatial benchmarks, while maintaining general visual understanding performance with modest gains on several benchmarks.

#### 1. Introduction

Deploying multimodal large language models (MLLMs) [28, 48, 64] in the physical world requires more than object recognition or captioning: it requires spatial reasoning across disparate viewpoints. Such reasoning in-

*Equal contributions. †Corresponding authors.

volves geometric understanding [47, 49], viewpoint imagination [56, 57], fine-grained perception like segmentation and detection [25, 65, 66], occlusion and topological reasoning [3, 21, 36], and scale or depth estimation [7, 20]. Despite rapid progress in MLLMs, how to train and evaluate these capabilities in a unified, scalable, and verifiable manner remains open.

A central obstacle is data. Curating supervision that truly elicits spatial reasoning is expensive and brittle. Manual annotation rarely captures the full mix of geometry, semantics, and context in a single example, while synthetic setups often struggle to match real-world diversity and verification at scale. This motivates a practical question: can we leverage existing large-scale video-3D data to both test and improve spatial reasoning in MLLMs with minimal human effort?

We revisit this challenge through the lens of WideBaseline Matching (WBM) [37, 42]: deciding whether two views separated by large baselines, strong perspective and appearance changes, repetitive structures, illumination shifts, and semantic occlusions depict the same physical scene element. Classical feature-based pipelines can be effective under small viewpoint changes or dense frame sampling, but they frequently fail in the extreme regime. Humans, on the contrary, still succeed by jointly exploiting geometric regularities, semantic knowledge, and contextual cues. This raises two questions: how well do current MLLMs handle WBM task, and what data and training paradigm can reliably improve this ability?

To answer these questions, we introduce ReasonMatchBench, a comprehensive benchmark for assessing crossview spatial reasoning in MLLMs through wide-baseline matching. ReasonMatch-Bench stratifies difficulty by viewpoint change magnitude and matching granularity, spanning indoor, outdoor, and object-centric scenarios. Our study reveals that current MLLMs still struggle on wide-baseline matching. On a difficult 90-sample human-study subset shown in Table 3, human annotators achieve 84.0 F1, while the best existing baseline reaches 37.2, and smaller models perform worse still. To improve this ability, we intro-

Training Curriculum

###### Benchmark Performance

ReasonMatch-Bench

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

Raw RGDB Pairs Matched Pairs

Image-Level Viewpoint Progression

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

Identical . . . Hard

Reward

[Figure 20]

Point-Level Correspondence Curriculum

Holistic Matching Reward

1 n

[Figure 21]

[Figure 22]

n

1[ f(̂ i) = f*(i)]

rmatch =

L1: Unambiguous matching L2: Selective matching L3: Partial matching

∑

[Figure 23]

i=1

Thinking

Spatial Distribution Refinement

[Figure 24]

Qwen3-VL+DCRL

[Figure 25]

[Figure 26]

<thinking> The two images depict the same office desk scene from different camera angles. The first image is a wider shot showing the desk, monitors, and the wall behind. The second image is a closer, more focused shot from a different perspective, showing the desk from the side and revealing more of the floor and chair. </thinking>

[Figure 27]

[Figure 28]

- Figure 1. Wide-Baseline Matching exposes major spatial-reasoning gaps in current MLLMs. Right: even frontier models struggle on ReasonMatch-Bench. Middle: DCRL improves ReasonMatch and transfers to other spatial intelligence benchmarks. Left: Dataset curation pipeline from RGBD datas to different WBM question-answer pairs.

duce a scalable data-generation pipeline that automatically harvests wide-baseline view pairs from large-scale video3D corpora, including RGB-D videos and Structure-fromMotion (SfM) reconstructions, yielding diverse, verifiable supervision across scenarios and matching granularities.

door, and object-centric scenarios with stratified difficulty levels.

- • We propose Dynamic Correspondence Reinforcement Learning (DCRL), a curriculum-based framework with dual-level adaptive curricula that enables MLLMs to progressively master complex spatial reasoning through verifiable rewards.
- • We show that DCRL improves ReasonMatch-Bench, transfers to related spatial benchmarks, and does not degrade general visual understanding.

Leveraging the verifiable nature of wide-baseline matching, we optimize MLLMs via reinforcement learning with verifiable rewards (RLVR) [18, 51, 61]: the model is rewarded according to its matching accuracy, enabling it to improve spatial reasoning without explicit reasoning supervision. To train this capability effectively, we introduce Dynamic Correspondence Reinforcement Learning (DCRL), which combines Image-Level Viewpoint Progression and Point-Level Correspondence Curriculum to perform a sample-efficient training process. Our experiments show that DCRL improves ReasonMatch-Bench substantially to 70.5% F1 score, outperforming both open-source and closed-source baselines including GPT-5-mini (57.9%) and Gemini-2.5-Pro (42.8%), and also transfers to related spatial benchmarks including OmniSpatial (+5.27%) and MindCube (+3.51%) while maintaining general visual understanding performance with modest gains on several benchmarks.

#### 2. Related Work

Spatial Reasoning in MLLMs. Evaluating and eliciting complex spatial reasoning in MLLMs remains an open challenge [14, 23, 32, 50, 55]. Existing benchmarks such as OmniSpatial [23] and VSI-Bench [55] assess various facets of spatial understanding, yet individual samples typically probe isolated capabilities such as relative positioning or viewpoint prediction, rather than requiring integrated reasoning across geometry, semantics, and context. On the training side, methods like SAT [38], RoboSpatial [44], and RoboRefer [62] focus primarily on visual grounding or simple relational reasoning, staying mainly on textual reasoning and MCQ evaluation. Multi-SpatialMLLM [54] explores correspondence matching but is limited to small viewpoint changes, restricted task formats (e.g., multiplechoice), and supervised fine-tuning (SFT) alone, which may

Our work makes the following contributions:

• We introduce ReasonMatch-Bench, a comprehensive benchmark for evaluating spatial reasoning in MLLMs through wide-baseline matching, spanning indoor, out-

not sufficiently elicit deeper spatial reasoning. In contrast, our work starts from wide-baseline matching (WBM), a fundamental yet challenging visual task that naturally demands complex spatial reasoning. Drawing inspiration from the success of reinforcement learning in DeepSeekR1 [18] and leveraging the inherent verifiability of matching tasks through geometric constraints, we employ reinforcement learning to enable MLLMs to autonomously explore and acquire complex spatial reasoning capabilities. This approach allows the model to discover reasoning strategies beyond what supervised annotations can provide, with the goal of improving performance on broader spatial intelligence tasks.

Wide Baseline View Matching. Wide baseline view matching refers to the task of finding correspondences between two or more views of a scene captured from significantly different viewpoints. This problem is fundamental to many computer vision applications, including 3D reconstruction and re-localization [24]. Traditional approaches relied on a pipeline of extracting handcrafted local features (e.g., SIFT [31], SURF [6], ORB [41]), matching them, and using a robust estimator like RANSAC [16] to find the epipolar geometry [19]. Subsequent research improved each component, introducing learned descriptors [33, 46], end-to-end feature networks [13, 15, 40], and advanced robust estimators [4, 5]. However, these feature-centric methods frequently fail in the extreme regime [24]. The severe changes in perspective, illumination, and occlusion inherent to this task demand robust reasoning about geometry, semantics, and context, which these methods lack.

#### 3. Method

We now describe our approach for eliciting complex spatial reasoning in MLLMs through Wide-Baseline Matching. Our method comprises three main components: task formulation for MLLMs to perform WBM, a scalable dataset and benchmark generation pipeline, a reinforcement learning framework using verifiable rewards combined with a curriculum strategy to progressively enhance spatial reasoning capabilities.

##### 3.1. Task Formulation for Wide-Baseline Matching

Problem Definition. We define the cross-view matching problem as follows. Given two images I1,I2 depicting the same 3D scene but captured from different viewpoints, let each camera be parameterized by its intrinsic matrix Ki and extrinsic parameters (Ri,ti), where Ri ∈ SO(3) and ti ∈ R3 denote the rotation and translation with respect to a common world coordinate frame. For a 3D point X ∈ R3, the image projection is given by the standard camera model in homogeneous coordinates: πi(X) = Ki[Ri |ti]X. The

goal is to predict a set of correspondences

M = {(xi,x′i)}Ni=1, (1) such that each pair corresponds to the same 3D point in the scene, i.e., xi = π1(Xi) and x′i = π2(Xi). Given sufficient and accurate correspondences, the relative camera pose (R,t) between the two views can be recovered via the epipolar constraint. Classical matching methods directly predict a large set of correspondences M by computing appearance similarity between local features [6, 31, 41], followed by geometric verification [19, 43].

Text-driven correspondence reasoning. Unlike classical matchers that output a continuous score matrix S ∈ Rn×m, the MLLM performs matching in a discrete, languagemediated manner. Given two pre-marked point sets X = {xi}ni=1 and Y = {yj}mj=1, the model receives (I1,X; I2,Y) as input, with visual prompts indicating point indices, and produces a textual mapping

fˆ: {1,...,n} → {1,...,m} ∪ {∅}, (2)

where fˆ(i) = j means that xi in I1 corresponds to yj in I2, and fˆ(i) = ∅ denotes no confident match. The predicted correspondence set is

M = {(xi,yfˆ(i)) | fˆ(i) ̸= ∅}. (3)

Conceptually, this process can be viewed as a partial bipartite matching [8, 17] between the two point sets X and Y, where each point may correspond to at most one counterpart or remain unmatched due to occlusion or limited overlap. This formulation treats the MLLM as a reasoning engine that performs symbolic association between visual entities rather than continuous feature matching, allowing it to integrate geometric, semantic, and contextual cues through complex spatial reasoning.

##### 3.2. Dataset Generation Pipeline

We now describe how to construct data samples (I1,X;I2,Y) with ground-truth correspondences. Each point set consists of matchable and distractor subsets:

X = Xmatch ∪ Xdist, Y = Ymatch ∪ Ydist, (4)

where Xmatch,Ymatch have valid correspondences and Xdist,Ydist are distractors. Our data generation pipeline focuses on obtaining image pairs (I1,I2) with verified correspondences that serve as the foundation for constructing these point sets.

Image pair selection and correspondence extraction. We source image pairs from diverse RGB-D datasets (CO3D [39], uCO3D [29], ScanNet [12]) and RGB videos with SfM reconstructions (RealEstate10k [63], DL3DV [27]). For RGB-D data, we obtain correspondences

###### Q: Two images show the same scene from different viewpoints. Match the region IDs from Image A to Image B.

[Figure 29]

[Figure 30]

[Figure 31]

Qwen3-VL+DCRL: GPT5-mini:

Infer matching relations using anchors (endpost, hanging pots overhead):

Key observations:

- - In the first image, we see a long shelf of tomato plants with two specific regions labeled: 1 and 2. Region 1 is on the lower shelf, and region 2 is on the upper shelf, both on the right side of the plant display.
- - In the second image, we see the same plant display from a different angle, with regions labeled 1 through 6. Region 1 is on the bottom shelf, near the floor, and region 6 is on the middle shelf, near the top of the plant display.

- - A-1 (middle-tier label slightly left of the very end in A): In image B the middle-tier label that sits very close to the rack’s right end is B-6. B-6 is described as middle tier and adjacent to the right end-post and near hanging pots — same 3D context as A-

1. A-1 therefore corresponds best to B-6.

- - A-2 (upper-tier label at the very right end near vertical support and hanging pots): In image B the overhead hanging cluster near the rack’s right end is B-4; this is the top-level/hanging-pot anchor near the rightmost support. A-2 is at the upper/rightmost area and aligns with the hanging-pot/end-post context of B-4. A-2 therefore corresponds to B-4.

A

By comparing the spatial context:

- - Region 1 in the first image is on the lower shelf, near the front of the display. In the second image, region 1 is also on the lower shelf, near the front, and is positioned similarly relative to the shelf structure.
- - Region 2 in the first image is on the upper shelf, near the back of the display. In the second image, region 6 is on the upper shelf, near the back, and is positioned similarly relative to the shelf structure.

[Figure 32]

[Figure 33]

Multi-tier shelf reasoning

Wrong tier inference

B

Viewpoint-invariant matching

Over-reliance on nearest anchors

- Figure 2. Comparison of cross-view region matching. Our method correctly preserves global spatial consistency across viewpoints, reasoning over multi-tier shelf structure and stable anchors. GPT-5-mini fails to perform viewpoint-consistent alignment and confuses shelf tiers, leading to incorrect cross-view correspondences. Note: Green highlights indicate correct spatial reasoning; red highlights indicate incorrect reasoning.

via geometric reprojection: each pixel in I1 with valid depth is back-projected to 3D and reprojected into I2, then verified using depth consistency and photometric consistency checks (see supplementary for details). For SfM data, we extract correspondences from shared 3D landmarks in COLMAP reconstructions [43], which have already passed geometric verification. This process yields a dense correspondence set M with thousands of matches per pair.

quality correspondence pool from which matchable and distractor points can be flexibly sampled for various training and evaluation scenarios.

##### 3.3. DCRL: Dynamic Correspondence Reinforcement Learning

Since WBM admits verifiable rewards but remains difficult under extreme viewpoint changes, we optimize with RLVR on our preprocessed samples (I1,I2,P) to enable MLLMs to autonomously develop spatial reasoning capabilities through exploration rather than merely imitating supervised demonstrations.

Viewpoint difficulty quantification. We quantify the viewpoint change between (I1,I2) using an overlap score ω ∈ [0,1]: for RGB-D pairs, ω measures the fraction of successfully matched pixels; for SfM pairs, ω reflects the proportion of shared 3D landmarks (details in supplementary). We use this score for source-aware difficulty stratification rather than direct cross-source comparison. We define the viewpoint-change magnitude as ∆v = 1−ω, which increases with baseline distance and occlusion. This metric enables us to stratify pairs by viewpoint difficulty and supports curriculum-based data organization.

However, directly training on extreme matching scenarios can lead to inefficient exploration and poor convergence. We therefore introduce a progressive curriculum that decomposes the difficulty along two complementary dimensions: image-level viewpoint progression that gradually increases geometric transformation complexity, and pointlevel correspondence progression that adaptively adjusts the number and spatial distribution of matchable points and distractors. This hierarchical decomposition enables the model to build spatial reasoning capabilities incrementally, mastering simpler configurations before tackling extreme scenarios.

Constructing the verified correspondence pool. The raw dense matches M are unsuitable for direct use: they cause severe visual overlap when marked on images and exceed practical input limits for MLLMs. We therefore apply clustering-based spatial filtering to subsample M into

a moderate-sized verified pool P = {(p1i,p2i)}Ni=1p with typically Np ∈ [10,50] spatially well-separated correspondences per pair. Specifically, we cluster matches in joint image-coordinate space and retain one representative per cluster to ensure adequate spacing for visual prompting. The final preprocessed samples (I1,I2,P) provide a high-

DCRL comprises: (1) a holistic matching reward that evaluates all query regions including unmatched points, encouraging comprehensive spatial reasoning; (2) an imagelevel progression that stages training by viewpoint divergence; (3) a point-level curriculum with two subdimensions—correspondence cardinality and spatial distri-

bution—that dynamically adjusts task construction within each viewpoint stage.

Holistic Matching Reward. Traditional partial bipartite matching evaluates only matched pairs, ignoring unmatched points. To encourage comprehensive spatial reasoning over all regions including occluded or out-of-view areas, we explicitly assign a dummy target (∅ in Eq. (2)) to unmatched points and reward correct “no match” predictions. This design eliminates objective ambiguity and prevents the model from focusing solely on easily matchable salient features, instead requiring deliberate reasoning about viewpointdependent visibility and geometric constraints across the entire scene.

Given predicted mapping fˆ and ground-truth f∗ over n query regions, we define matching correctness as:

n

1 n

rmatch =

i=1

f ˆ(i) = f∗(i) , (5)

which measures prediction accuracy over all regions, rewarding correct predictions including unmatched regions (∅). We additionally incorporate a format compliance component to ensure well-formed outputs, yielding final reward r = wf · rformat + wm · rmatch. Importantly, rmatch serves not only as the training signal for policy optimization but also as the control signal for dynamically adapting task difficulty across the curriculum dimensions described below.

Image-Level Viewpoint Progression. Rather than training on randomly shuffled image pairs, we partition the dataset by viewpoint overlap score ω to enable gradual adaptation to geometric complexity. Specifically, the dataset is organized into bins {Ds}Ss=1 by overlap intervals [ωs,ωs], where bin s = 1 contains high-overlap pairs with minimal viewpoint change and bin s = S contains extreme viewpoint divergence. Training proceeds sequentially through these bins: once sustained performance on Ds exceeds a threshold, we advance to Ds+1 and permanently exclude the easier bin from the training set. This staged progression offers two key benefits. In early training, the model focuses on simpler geometric transformations, rapidly building foundational spatial reasoning capabilities and achieving faster convergence. In later stages, the model leverages its established understanding to tackle extreme viewpoint changes, where the more challenging scenarios provide richer learning signals and greater information gain. By filtering out mastered configurations, we maintain training efficiency while progressively pushing the frontier of the model’s spatial reasoning capabilities.

Point-Level Correspondence Curriculum. A key feature of our approach is that point sets X,Y are not pre-

marked offline on images. Instead, we dynamically sample them from the verified pool via X,Y = g(P), where the sampling strategy g adapts to control task difficulty. Within each viewpoint stage, we modulate task complexity along two sub-dimensions: (1) Cardinality adaptation adjusts the number of matchable points and distractors to vary selection ambiguity; (2) Spatial distribution refinement modulates the spatial arrangement of sampled points to influence local versus global reasoning demands.

Cardinality adaptation. When sampling point sets X,Y from the verified pool P, we dynamically adjust the number of matchable and distractor points to control task difficulty. Recall the point partition in Eq. 4. We construct three progressive difficulty levels:

- • Unambiguous matching (L1): Xdist = Ydist = ∅ with |Xmatch| = |Ymatch| = n. One-to-one correspondence eliminates selection ambiguity, allowing the model to focus on geometric transformation understanding. The model learns to reason about appearance changes, occlusion boundaries, and 3D structure without distractor interference.
- • Selective matching (L2): Xdist = ∅, |Ydist| > 0. Introduces the challenge of selecting correct matches from multiple candidates in Y. This simulates asymmetric scene coverage where one view observes additional regions, requiring the model to distinguish geometrically consistent correspondences from visually similar but incorrect candidates.
- • Partial matching (L3): Both |Xdist| > 0 and |Ydist| > 0. Models realistic scenarios with bidirectional occlusion and incomplete overlap. The model must explicitly reason about which regions are visible in both views versus occluded or out-of-frame, integrating geometric constraints with semantic understanding of scene structure and viewpoint-dependent visibility.

The curriculum dynamically promotes to higher levels as performance improves and demotes upon performance degradation, adapting task complexity to the model’s evolving capabilities.

Spatial distribution refinement. Beyond cardinality, the spatial distribution of matchable points sampled from P significantly impacts task difficulty. Points that are too densely clustered become difficult to distinguish or form fixed patterns that enable matching without visual context reasoning. Conversely, overly sparse points limit the model to coarse object-level understanding without learning finegrained spatial relationships. We therefore dynamically control spatial distribution through clustering radius and sampling strategies to progressively refine spatial reasoning capabilities.

Specifically, we employ a cluster-based sampling approach where correspondences are first grouped by spatial proximity, then representatives are selected per cluster. We

Table 1. Performance comparison on ReasonMatch-Bench. We report F1, Precision, and Recall scores across three scenarios (Indoor, Outdoor, Object) and three difficulty levels.

Model F1 Precision Recall Indoor Outdoor Object L1 L2 L3 L1 L2 L3 L1 L2 L3

GPT-5-mini 57.9 56.9 59.4 68.2 65.3 47.0 75.8 75.0 51.4 45.3 60.4 27.8 GPT-5-Chat 51.5 50.6 52.8 52.4 62.5 39.3 61.8 75.1 44.9 44.1 57.2 28.6 GPT-4o-241106 33.5 32.7 34.7 31.9 43.7 21.7 33.9 54.0 23.3 31.9 42.2 19.9 Gemini-2.5-Pro 42.8 42.4 43.4 48.3 49.3 36.7 63.4 64.5 46.6 29.5 35.9 19.0 Claude-4.5-Sonnet 41.7 43.7 41.1 40.9 50.2 29.9 50.3 56.9 37.4 33.9 43.4 19.4 Claude-4.1-Opus 33.7 37.2 32.2 31.0 36.0 23.8 45.1 45.3 31.6 26.3 37.2 16.1 Claude-4-Sonnet 34.8 34.2 35.5 33.0 43.7 21.3 45.0 54.5 29.8 30.8 44.1 18.3

Qwen3-VL-235B [1] 49.2 50.7 48.7 52.0 60.5 35.8 58.8 71.1 41.7 35.1 49.4 22.1 Qwen3-VL-8B-Instruct [1] 27.5 27.1 29.1 25.5 38.8 17.6 30.4 39.5 18.8 21.1 34.5 11.5

Qwen3-VL-8B + DCRL 70.5 70.3 71.1 84.6 75.1 67.0 90.9 80.2 73.6 45.6 63.1 33.7 ∆ vs. Qwen3-VL-8B-Instruct 43.0 43.2 42.0 59.0 36.3 49.4 60.5 40.7 54.8 24.5 28.6 22.2

progress through three stages: (1) Maximally sparse sampling: selecting one point per cluster with large clustering radius, producing globally distributed points that require object-level reasoning; (2) Moderate clustering: reducing the clustering radius to allow multiple points per region, introducing finer spatial structure; (3) Dense sampling: transitioning to random sampling with minimal spacing constraints, requiring the model to reason about detailed geometric relationships and subtle appearance variations at fine granularity. This progression gradually eliminates spatial cues that facilitate matching, forcing the model to develop comprehensive geometric understanding rather than relying on global landmark distribution alone.

This two-level hierarchy—image-level viewpoint filtering as the outer loop and point-level adaptive construction as the inner loop—enables efficient exploration by aligning task difficulty with the model’s evolving spatial reasoning capabilities.

#### 4. Experiments

##### 4.1. Dataset and Benchmark Statistics

TestSet Composition and Balance. To evaluate WBM in MLLMs, we curate 2,810 image pairs from our 220k-pair corpus as ReasonMatch-Bench. The benchmark balances data sources (ScanNet 27.7%, uCO3D 28.0%, DL3DV 27.0%, RE10K 17.2%), task levels (L1 32.5%, L2 36.8%, L3 30.7% as defined in Sec. 3.3), and scene types (indoor 55.1%, object 28.0%, outdoor 16.9%). This balance extends to cross-dimensional distributions: within each dataset, task composition remains approximately uniform. It ensures that evaluation metrics reflect broad spatial reasoning performance rather than biases toward particular sources or configurations. See the supplementary material for detailed statistics.

##### 4.2. Implementation Details

RLVR Configuration and Training Setup. We apply GRPO on Qwen3-VL-8B-Instruct [1] with a group size G = 32 to ensure sufficient variance reduction and exploration of diverse reasoning paths during policy optimization. The effective batch size is 16×32 trajectories per update. The KL loss coefficient is set to β = 0.005. Each generated prediction is capped at 5120 tokens and the temperature is set T = 1.0 for the policy rollouts to maintain sufficient exploration without introducing excessive randomness. We use AdamW optimizer [30] with a linear warmup over the first 10 steps, and a constant learning rate of 10−6. Reward and Curriculum. The reward function uses weights (wf,wm) = (1.0,1.0) as described in Section 3, with format compliance and matching correctness weighted equally. For curriculum configuration, we organize training along three hierarchical dimensions. At the viewpoint level, we partition the dataset into 10 overlap bins and advance to the next bin once the average accuracy reward exceeds 0.8 over a sliding window of 20 training steps. The cardinality adaptation strategy and spatial correspondence distribution settings are shown in the supplement.

##### 4.3. Main Results on ReasonMatch-Bench

Performance Analysis across Scene Types and Task Levels. Table 1 presents a comprehensive comparison across three scene types and three task difficulty levels. Additional qualitative analysis of model CoT outputs is provided in the supplement. Several notable patterns emerge from this evaluation.

Our method achieves the best overall result and strong gains in challenging settings. The improvement is particularly pronounced in difficult scenarios: while maintaining strong performance on easier outdoor scenes, our model also shows robust performance on complex indoor and instance-level matching tasks where baseline models

Table 2. Results on the OmniSpatial benchmark. Accuracy (%) is reported for overall and the four major reasoning dimensions.

Model Size Overall Dynamic Reasoning Spatial Interaction Complex Logic Perspective Taking

GPT-4-turbo [34] – 34.06 38.39 36.49 24.80 33.69 Gemini-2.5-flash-preview-05-20 [45] – 52.12 63.59 67.46 35.67 43.10 LLaVA-1.5-vicuna-7B [28] 7B 34.97 35.38 35.13 25.99 38.82 InternVL3-78B [64] 78B 49.33 63.24 55.61 29.23 44.93 Qwen2.5-VL [2] 7B 39.25 46.30 30.06 35.65 39.68

Qwen3-VL [1] 8B 43.60 51.90 51.90 24.40 42.50 DCRL 8B 48.87 61.48 55.33 32.78 43.21

struggle significantly.

Scene-related difficulty reveals dataset characteristics. Outdoor scenes prove most tractable across all models, with even baseline systems achieving reasonable performance on L1 tasks. Indoor scenes present moderate complexity, where our method maintains substantial advantages over strong proprietary baselines on many difficult configurations. Instance-level matching emerges as the most challenging scenario—baseline models show dramatic performance degradation, particularly on L3 tasks, while our approach maintains relatively stable performance. This pattern reflects the fundamental challenge of object-centric matching: isolated objects lack environmental context that aids correspondence reasoning in full scenes.

To calibrate benchmark difficulty against human performance, we additionally evaluate annotators on the 90 largest-view-divergence samples from DL3DV, RE10K, and uCO3D including indoor, outdoor and object level scenes. Table 3 shows that humans achieve 84.0 F1 overall, compared with 52.0 for DCRL. The remaining gap is especially large on object-centric uCO3D (62.1 vs. 27.8), indicating that challenging wide-baseline correspondence remains far from solved even after DCRL training.

Qualitative analysis of failure modes. Examining model outputs reveals distinct reasoning patterns across baselines. Gemini-2.5-Pro demonstrates accurate pointlevel descriptions, providing detailed local appearance characterizations for each annotated region. However,

Table 3. Human study on a 90-sample high-divergence subset. We report F1 only; the full precision/recall/F1 breakdown is provided in the supplement.

###### Method Overall DL3DV RE10K uCO3D

GPT-5-mini 37.2 35.9 49.7 25.8 Gemini-2.5-Pro 29.5 26.5 44.1 18.0 Claude-4.5-Sonnet 24.0 22.2 30.5 19.2 Qwen3-VL-235B [1] 29.9 25.3 45.7 18.7 DCRL 52.0 57.7 70.6 27.8

Human 84.0 93.5 94.7 62.1

Table 4. Results on MindCube and SAT Real benchmarks. Accuracy (%) for spatial reasoning tasks.

MindCube SAT Overall Rotation Among Around Real

Model

Open-Weight Multi Image Models LongVA-7B [59] 29.46 35.89 29.55 24.88 – InternVL2.5-8B [11] 18.68 36.45 18.20 13.11 – Qwen2.5-VL-7B [2] 29.26 38.76 29.50 21.35 56.33 Idefics3-8B-Llama3 [26] 35.86 35.15 35.94 35.49 –

Proprietary Model GPT-4o [35] 38.81 32.65 40.17 29.16 57.50

Spatial Models

RoboBrain [22] 37.38 35.80 38.28 29.53 – SpatialVLM [9] 22.81 37.65 21.26 29.39 –

Our Models

Qwen3-VL-8B [1] 40.01 53.20 41.00 34.33 70.00 DCRL 43.52 59.20 43.50 37.00 75.30

these descriptions, while locally correct, lack global specificity and discriminative power within the full scene context—descriptions like “white wall region” or “wooden surface” may accurately characterize local appearance but fail to uniquely identify the target point when multiple similar regions exist. This inability to leverage holistic scene understanding and 3D spatial relationships leads the model to perform ambiguous local feature matching rather than geometric correspondence reasoning. The Qwen3-VL series exhibits complementary strengths and weaknesses: these models show strong awareness of viewpoint changes and can reason about cross-view geometric transformations effectively. However, they suffer from frequent visual label misidentification and reasoning-answer inconsistencies, where the model’s Chain-of-Thought reasoning arrives at correct correspondences but the final formatted output contradicts this reasoning. We attribute this to Qwen3-VL’s prior training on spatial intelligence tasks providing geometric intuition, but insufficient exposure to fine-grained cross-view scenarios and multi-image contexts leads to persistent hallucination issues when parsing dense visual annotations.

Table 5. Performance on general visual understanding benchmarks, measured using lmms-eval [58].

Model MME-RealWorld MMStar RealWorldQA V*Bench

Qwen3-VL-8B [1] 62.8 59.8 69.5 84.8 DCRL 63.8 62.5 70.5 85.9

##### 4.4. Generalization on other Spatial and Visual Understanding Benchmarks

To evaluate transfer beyond ReasonMatch-Bench and check whether spatial training affects broader vision-language performance, we test our model on both spatial intelligence benchmarks and general vision-language benchmarks. We compare our model against the base model on three spatial intelligence benchmarks: OmniSpatial [23], SAT [38], and MindCube [57]. We also report results on four general visual understanding benchmarks: MMStar [10], RealWorldQA [53], MME-RealWorld [60], and V*Bench [52].

Table 6. Performance of RL and SFT training objectives.

Method OmniSpatial MindCube SAT ReasonMatch

Base 43.6 40.0 70.0 27.5 SFT 42.6 45.1 41.3 51.0 DCRL 48.9 43.5 75.3 70.5

Performance on Spatial Intelligence and General Visual Benchmarks. Table 4 and Table 2 present results on three spatial intelligence benchmarks. Our model improves over the base model on the reported spatial benchmarks. These gains suggest positive transfer from cross-view matching to related spatial benchmarks beyond the specific matching task. Examining OmniSpatial’s sub-categories reveals heterogeneous transfer patterns. Dynamic Reasoning (9.6%) and Complex Logic (8.38%) show the largest gains among the reported OmniSpatial sub-categories, while Spatial Interaction exhibits moderate gains (3.4%) and Perspective Taking remains nearly stable. One possible explanation is the composition of our training data: many indoor scenes come from room navigation videos, which often include camera rotation and egocentric motion and may therefore better match Complex Logic tasks (3D mental rotation, geometric pattern completion) and Dynamic Reasoning tasks (motion prediction, viewpoint changes). This interpretation is consistent with the MindCube results, where the Rotation sub-task evaluating spatial reasoning under viewpoint rotations exhibits the largest gain (6.0%), notably exceeding improvements on Among and Around sub-tasks.

Table 5 shows no degradation on the reported general visual understanding benchmarks, together with modest gains on all four, suggesting that this spatial training does not harm general visual performance in our evaluation.

##### 4.5. Analysis and Ablation Studies

Trained on a CoT-annotated WBM dataset, SFT substantially improves in-domain ReasonMatch performance over the base model, but transfers inconsistently on other benchmarks. In contrast, DCRL improves all reported spatial benchmarks and outperforms SFT by +19.5 on ReasonMatch and +34.0 on SAT. This contrast in Table 6 suggests that teacher-forced imitation can overfit to correspondence patterns, whereas reinforcement learning with verifiable rewards develops more transferable spatial reasoning.

[Figure 34]

Figure 3. Training curves of DCRL. Left: reward curve. Right: Mean response length per step.

The curriculum ablation further shows that progressive difficulty adjustment is useful. Uniformly sampled RL already outperforms easy-only or hard-only subsets, but the proposed dynamic curriculum delivers the best result, improving over uniformly sampling by +5.2 points. Figure 3 further shows stable convergence during DCRL training. Full ablation details remain in the supplement.

#### 5. Conclusion

We introduced ReasonMatch-Bench, a benchmark for evaluating wide-baseline spatial correspondence in MLLMs, together with a scalable video-3D data pipeline and DCRL, a reinforcement-learning framework for training on this task with verifiable rewards. Across open- and closed-source baselines, current models remain substantially below human performance on a difficult 90-sample subset, where human annotators achieve 84.0 F1 and our best model reaches 52.0. Our experiments show that DCRL improves widebaseline matching and yields positive transfer to several related spatial benchmarks while maintaining general visual understanding performance. These results suggest that wide-baseline correspondence is a useful testbed for studying cross-view spatial reasoning in MLLMs, and that substantial headroom remains.

#### Acknowledgments

This work was supported in part by the Pioneer R&D Program of Zhejiang (Grant No. 2025C01011), by the Ant Group Research Intern Program, and by and the National Natural Science Foundation of China (Grant No. 62576315).

#### References

- [1] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025. 6, 7, 8
- [2] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report, 2025. 7
- [3] Prithviraj Banerjee, Sindi Shkodrani, Pierre Moulon, Shreyas Hampali, Shangchen Han, Fan Zhang, Linguang Zhang, Jade Fountain, Edward Miller, Selen Basol, et al. Hot3d: Hand and object tracking in 3d from egocentric multi-view videos. In CVPR, pages 7061–7071, 2025. 1
- [4] Daniel Barath and Jiri Matas. Graph-cut ransac. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 6733–6741, 2018. 3
- [5] Daniel Barath, Jiri Matas, and Jana Noskova. Magsac: marginalizing sample consensus. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10197–10205, 2019. 3
- [6] Herbert Bay, Tinne Tuytelaars, and Luc Van Gool. Surf: Speeded up robust features. In European conference on computer vision, pages 404–417. Springer, 2006. 3
- [7] Aleksei Bochkovskii, AmaA ¸Gl˜ Delaunoy, Hugo Germain, Marcel Santos, Yichao Zhou, Stephan R Richter, and Vladlen Koltun. Depth pro: Sharp monocular metric depth in less than a second. arXiv preprint arXiv:2410.02073, 2024. 1
- [8] Rainer Burkard, Mauro Dell’Amico, and Silvano Martello. Assignment problems: revised reprint. SIAM, 2012. 3
- [9] Boyuan Chen, Zhuo Xu, Sean Kirmani, Brian Ichter, Danny Driess, Pete Florence, Dorsa Sadigh, Leonidas Guibas, and Fei Xia. Spatialvlm: Endowing vision-language models with spatial reasoning capabilities. arXiv preprint arXiv:2401.12168, 2024. 7
- [10] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, et al. Are we on the right way for evaluating large vision-language models? Advances in Neural Information Processing Systems, 37:27056–27087, 2024. 8
- [11] Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, et al. Expanding performance boundaries of open-source multimodal models with model, data, and testtime scaling. arXiv preprint arXiv:2412.05271, 2024. 7
- [12] Angela Dai, Angel X Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias Nießner. Scannet: Richly-annotated 3d reconstructions of indoor scenes. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 5828–5839, 2017. 3, 1
- [13] Daniel DeTone, Tomasz Malisiewicz, and Andrew Rabinovich. Superpoint: Self-supervised interest point detection and description. In Proceedings of the IEEE conference on

- computer vision and pattern recognition workshops, pages 337–348, 2018. 3
- [14] Zihao Dongfang, Xu Zheng, Ziqiao Weng, Yuanhuiyi Lyu, Danda Pani Paudel, Luc Van Gool, Kailun Yang, and Xuming Hu. Are multimodal large language models ready for omnidirectional spatial reasoning? arXiv preprint arXiv:2505.11907, 2025. 2
- [15] Mihai Dusmanu, Ignacio Rocco, Tomas Pajdla, Marc Pollefeys, Josef Sivic, Akihiko Torii, and Torsten Sattler. D2-net: A trainable cnn for joint detection and description of local features. In CVPR 2019-IEEE Conference on Computer Vision and Pattern Recognition, 2019. 3
- [16] Martin A Fischler and Robert C Bolles. Random sample consensus: a paradigm for model fitting with applications to image analysis and automated cartography. Communications of the ACM, 24(6):381–395, 1981. 3
- [17] Steven Gold and Anand Rangarajan. A graduated assignment algorithm for graph matching. IEEE Transactions on pattern analysis and machine intelligence, 18(4):377–388, 2002. 3
- [18] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, et al. Deepseek-r1 incentivizes reasoning in llms through reinforcement learning. Nature, 645(8081):633– 638, 2025. 2, 3
- [19] Richard I Hartley. Projective reconstruction and invariants from multiple images. IEEE Transactions on Pattern Analysis and Machine Intelligence, 16(10):1036–1041, 1994. 3
- [20] Mu Hu, Wei Yin, Chi Zhang, Zhipeng Cai, Xiaoxiao Long, Hao Chen, Kaixuan Wang, Gang Yu, Chunhua Shen, and Shaojie Shen. Metric3d v2: A versatile monocular geometric foundation model for zero-shot metric depth and surface normal estimation. IEEE TPAMI, 2024. 1
- [21] Zheng Huang, Mingyu Liu, Xiaoyi Lin, Muzhi Zhu, Canyu Zhao, Zongze Du, Xiaoman Li, Yiduo Jia, Hao Zhong, Hao Chen, et al. Notvla: Narrowing of dense action trajectories for generalizable robot manipulation. arXiv preprint arXiv:2510.03895, 2025. 1
- [22] Yuheng Ji, Huajie Tan, Jiayu Shi, Xiaoshuai Hao, Yuan Zhang, Hengyuan Zhang, Pengwei Wang, Mengdi Zhao, Yao Mu, Pengju An, et al. Robobrain: A unified brain model for robotic manipulation from abstract to concrete. arXiv preprint arXiv:2502.21257, 2025. 7
- [23] Mengdi Jia, Zekun Qi, Shaochen Zhang, Wenyao Zhang, Xinqiang Yu, Jiawei He, He Wang, and Li Yi. Omnispatial: Towards comprehensive spatial reasoning benchmark for vision language models. arXiv preprint arXiv:2506.03135,

2025. 2, 8

- [24] Yuhe Jin, Dmytro Mishkin, Anastasiia Mishchuk, Jiri Matas, Pascal Fua, Kwang Moo Yi, and Eduard Trulls. Image matching across wide baselines: From paper to practice. arXiv preprint arXiv:2003.01587, 2020. 3
- [25] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4015–4026, 2023. 1

- [26] Hugo Lauren¸con, Andr´es Marafioti, Victor Sanh, and L´eo Tronchon. Building and better understanding visionlanguage models: Insights and future directions. In Workshop on Responsibly Building the Next Generation of Multimodal Foundational Models, 2024. 7
- [27] Lu Ling, Yichen Sheng, Zhi Tu, Wentian Zhao, Cheng Xin, Kun Wan, Lantao Yu, Qianyu Guo, Zixun Yu, Yawen Lu, et al. Dl3dv-10k: A large-scale scene dataset for deep learning-based 3d vision. In CVPR, pages 22160–22169,

2024. 3, 1

- [28] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 1, 7
- [29] Xingchen Liu, Piyush Tayal, Jianyuan Wang, Jesus Zarzar, Tom Monnier, Konstantinos Tertikas, Jiali Duan, Antoine Toisoul, Jason Y. Zhang, Natalia Neverova, Andrea Vedaldi, Roman Shapovalov, and David Novotny. Uncommon objects in 3d. In arXiv, 2025. 3, 1
- [30] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017. 6
- [31] David G Lowe. Distinctive image features from scaleinvariant keypoints. International journal of computer vision, 60(2):91–110, 2004. 3
- [32] Wufei Ma, Haoyu Chen, Guofeng Zhang, Yu-Cheng Chou, Jieneng Chen, Celso de Melo, and Alan Yuille. 3dsrbench: A comprehensive 3d spatial reasoning benchmark. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 6924–6934, 2025. 2
- [33] Anastasiia Mishchuk, Dmytro Mishkin, Filip Radenovic, and Jiri Matas. Working hard to know your neighbor’s margins: Local descriptor learning loss. In Advances in neural information processing systems, pages 4826–4837, 2017. 3
- [34] OpenAI. Gpt-4 technical report. CoRR, abs/2303.08774,

2023. 7

- [35] OpenAI. Hello gpt-4o. OpenAI Blog, 2024. Accessed: November 22, 2024. 7
- [36] Youxin Pang, Ruizhi Shao, Jiajun Zhang, Hanzhang Tu, Yun Liu, Boyao Zhou, Hongwen Zhang, and Yebin Liu. Manivideo: Generating hand-object manipulation video with dexterous and generalizable grasping. In CVPR, pages 12209–12219, 2025. 1
- [37] Philip Pritchett and Andrew Zisserman. Wide baseline stereo matching. In ICCV, pages 754–760. IEEE, 1998. 1
- [38] Arijit Ray, Jiafei Duan, Ellis Brown, Reuben Tan, Dina Bashkirova, Rose Hendrix, Kiana Ehsani, Aniruddha Kembhavi, Bryan A. Plummer, Ranjay Krishna, Kuo-Hao Zeng, and Kate Saenko. Sat: Dynamic spatial aptitude training for multimodal language models. arXiv preprint arXiv:2412.07755, 2024. 2, 8
- [39] Jeremy Reizenstein, Roman Shapovalov, Philipp Henzler, Luca Sbordone, Patrick Labatut, and David Novotny. Common objects in 3d: Large-scale learning and evaluation of real-life 3d category reconstruction. In Proceedings of the IEEE/CVF international conference on computer vision, pages 10901–10911, 2021. 3, 1

- [40] J´erˆome Revaud, Philippe Weinzaepfel, C´esar De Souza, No´e Pion, Gabriela Csurka, Yohann Cabon, and Martin Humenberger. R2d2: Repeatable and reliable detector and descriptor. arXiv preprint arXiv:1906.06195, 2019. 3
- [41] Ethan Rublee, Vincent Rabaud, Kurt Konolige, and Gary Bradski. Orb: An efficient alternative to sift or surf. In International conference on computer vision, pages 2564–2571. IEEE, 2011. 3
- [42] Cordelia Schmid and Roger Mohr. Matching by local invariants. PhD thesis, INRIA, 1995. 1
- [43] Johannes L Schonberger and Jan-Michael Frahm. Structurefrom-motion revisited. In CVPR, pages 4104–4113, 2016. 3, 4, 1
- [44] Chan Hee Song, Valts Blukis, Jonathan Tremblay, Stephen Tyree, Yu Su, and Stan Birchfield. Robospatial: Teaching spatial understanding to 2d and 3d vision-language models for robotics. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 15768–15780, 2025. 2
- [45] Gemini Team, Rohan Anil, Sebastian Borgeaud, JeanBaptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M. Dai, Anja Hauth, Katie Millican, et al. Gemini: A family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023. 7
- [46] Yurun Tian, Xin Yu, Bin Fan, Fuchao Wu, Huub Heijnen, and Vassileios Balntas. Sosnet: Second order similarity regularization for local descriptor learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11494–11503, 2019. 3
- [47] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. Vggt: Visual geometry grounded transformer. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 5294–5306, 2025. 1
- [48] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, and Junyang Lin. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. CoRR, abs/2409.12191, 2024. 1
- [49] Ruicheng Wang, Sicheng Xu, Cassie Dai, Jianfeng Xiang, Yu Deng, Xin Tong, and Jiaolong Yang. Moge: Unlocking accurate monocular geometry estimation for open-domain images with optimal training supervision. In CVPR, pages 5261–5271, 2025. 1
- [50] Wenqi Wang, Reuben Tan, Pengyue Zhu, Jianwei Yang, Zhengyuan Yang, Lijuan Wang, Andrey Kolobov, Jianfeng Gao, and Boqing Gong. Site: towards spatial intelligence thorough evaluation. arXiv preprint arXiv:2505.05456,

2025. 2

- [51] Xumeng Wen, Zihan Liu, Shun Zheng, Shengyu Ye, Zhirong Wu, Yang Wang, Zhijian Xu, Xiao Liang, Junjie Li, Ziming Miao, et al. Reinforcement learning with verifiable rewards implicitly incentivizes correct reasoning in base llms. arXiv preprint arXiv:2506.14245, 2025. 2
- [52] Penghao Wu and Saining Xie. V?: Guided visual search as a core mechanism in multimodal llms. In Proceedings of

- the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13084–13094, 2024. 8
- [53] xAI. Realworldqa: Real-world visual question answering benchmark. https://x.ai/news/grok-1.5v, 2024. 8
- [54] Runsen Xu, Weiyao Wang, Hao Tang, Xingyu Chen, Xiaodong Wang, Fu-Jen Chu, Dahua Lin, Matt Feiszli, and Kevin J Liang. Multi-spatialmllm: Multi-frame spatial understanding with multi-modal large language models. arXiv preprint arXiv:2505.17015, 2025. 2
- [55] Jihan Yang, Shusheng Yang, Anjali W Gupta, Rilyn Han, Li Fei-Fei, and Saining Xie. Thinking in space: How multimodal large language models see, remember, and recall spaces. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 10632–10643, 2025. 2
- [56] Chun-Hsiao Yeh, Chenyu Wang, Shengbang Tong, Ta-Ying Cheng, Ruoyu Wang, Tianzhe Chu, Yuexiang Zhai, Yubei Chen, Shenghua Gao, and Yi Ma. Seeing from another perspective: Evaluating multi-view understanding in mllms. arXiv preprint arXiv:2504.15280, 2025. 1
- [57] Baiqiao Yin, Qineng Wang, Pingyue Zhang, Jianshu Zhang, Kangrui Wang, Zihan Wang, Jieyu Zhang, Keshigeyan Chandrasegaran, Han Liu, Ranjay Krishna, Saining Xie, Manling Li, Jiajun Wu, and Li Fei-Fei. Mindcube: Spatial mental modeling from limited views. arXiv preprint arXiv:2506.21458, 2025. 1, 8
- [58] Kaichen Zhang, Bo Li, Peiyuan Zhang, Fanyi Pu, Joshua Adrian Cahyono, Kairui Hu, Shuai Liu, Yuanhan Zhang, Jingkang Yang, Chunyuan Li, and Ziwei Liu. Lmmseval: Reality check on the evaluation of large multimodal models, 2024. 8
- [59] Peiyuan Zhang, Kaichen Zhang, Bo Li, Guangtao Zeng, Jingkang Yang, Yuanhan Zhang, Ziyue Wang, Haoran Tan, Chunyuan Li, and Ziwei Liu. Long context transfer from language to vision. arXiv preprint arXiv:2406.16852, 2024. 7
- [60] Yi-Fan Zhang, Huanyu Zhang, Haochen Tian, Chaoyou Fu, Shuangqing Zhang, Junfei Wu, Feng Li, Kun Wang, Qingsong Wen, Zhang Zhang, et al. Mme-realworld: Could your multimodal llm challenge high-resolution real-world scenarios that are difficult for humans? arXiv preprint arXiv:2408.13257, 2024. 8
- [61] Hao Zhong, Muzhi Zhu, Zongze Du, Zheng Huang, Canyu Zhao, Mingyu Liu, Wen Wang, Hao Chen, and Chunhua Shen. Omni-r1: Reinforcement learning for omnimodal reasoning via two-system collaboration. arXiv preprint arXiv:2505.20256, 2025. 2
- [62] Enshen Zhou, Jingkun An, Cheng Chi, Yi Han, Shanyu Rong, Chi Zhang, Pengwei Wang, Zhongyuan Wang, Tiejun Huang, Lu Sheng, et al. Roborefer: Towards spatial referring with reasoning in vision-language models for robotics. arXiv preprint arXiv:2506.04308, 2025. 2
- [63] Tinghui Zhou, Richard Tucker, John Flynn, Graham Fyffe, and Noah Snavely. Stereo magnification: learning view synthesis using multiplane images. ACM Transactions on Graphics (TOG), 37(4):1–12, 2018. 3, 1
- [64] Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Yuchen Duan, Hao Tian, Weijie Su,

- Jie Shao, et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. CoRR, abs/2504.10479, 2025. 1, 7
- [65] Muzhi Zhu, Yuzhuo Tian, Hao Chen, Chunluan Zhou, Qingpei Guo, Yang Liu, Ming Yang, and Chunhua Shen. Segagent: Exploring pixel understanding capabilities in mllms by imitating human annotator trajectories. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 3686–3696, 2025. 1
- [66] Muzhi Zhu, Hao Zhong, Canyu Zhao, Zongze Du, Zheng Huang, Mingyu Liu, Hao Chen, Cheng Zou, Jingdong Chen, Ming Yang, et al. Active-o3: Empowering multimodal large language models with active perception via grpo. arXiv preprint arXiv:2505.21457, 2025. 1

## Eliciting Complex Spatial Reasoning in MLLMs through Wide-Baseline Matching

### Supplementary Material

#### 6. Appendix Overview

This supplementary material provides comprehensive details supporting our main paper, organized as follows:

- Sec. 7 – Implementation Details: Complete specifi-

cations of data generation pipeline, experimental setup, prompt template, and curriculum progression schedules.

- Sec. 8 – Additional Ablations: Extended analyses on

curriculum design variants, overlap scheduling strategies, and detailed RL vs. SFT comparisons across training stages.

- Sec. 9 – Benchmark Analysis: Detailed benchmark

statistics, per-category model performance, failure mode analyses, and qualitative examples illustrating model capabilities and limitations.

- Sec. 10 – Extended Discussion: Expanded analysis of

limitations and concrete future research directions for advancing MLLM spatial intelligence.

#### 7. Implementation Details

##### 7.1. Dataset Generation

Correspondence Generation from RGB-D Data. Starting from RGB-D videos (or RGB with known intrinsics and camera-to-world poses), we back-project every pixel x with valid depth in image I1 into a 3D point X in the world coordinate system under a static-scene assumption, and reproject it to the consecutive image I2 using the known camera parameters to obtain the image location yproj and its camera depth zproj. At yproj we bilinearly sample the observed depth zobs and the color I2. We source our data from diverse datasets including uCO3D [29], CO3D [39], and ScanNet [12]. We compute two verifiable consistency terms: (i) relative depth consistency

zproj − zobs zproj + ε

, (6)

edepth =

and (ii) photometric consistency defined as the channelaveraged RGB difference with bilinear sampling at yproj

I1(c)(x) − Iˆ2(c)(yproj) . (7)

ephoto = 13

c∈{r,g,b}

A correspondence is considered valid if it meets basic visibility, boundedness criteria:

I finite(yproj) ∧ in-bounds(yproj) ∧ zproj > 0 ∧ zobs > 0 ,

(8)

and error criteria:

I edepth < τd · I ephoto < τp . (9)

Here τd and τp are pre-defined thresholds for depth and photometric errors, respectively. Optional masks (e.g., dynamic-object or invalid-depth masks) can be applied on the image I1 and I2.

Correspondence Generation from SfM Data. For datasets lacking ground-truth depth, such as RealEstate10k [63] and DL3DV [27], we leverage 3D reconstructions from Structure-from-Motion (SfM). For DL3DV, we utilize the provided COLMAP [43] models directly. For RealEstate10k, which provides only RGB video streams, we first process the raw sequences with COLMAP to generate these 3D reconstructions. Groundtruth correspondences are then extracted by identifying shared 3D landmarks between two image views. Specifically, a pair of 2D keypoints (xi,yj), where xi is in image I1 and yj is in image I2, is considered a valid match if they both correspond to the same 3D point X in the COLMAP model’s sparse reconstruction. This inlier set, having already passed geometric verification within COLMAP, serves as our reliable matches, bypassing the consistency checks (Eq. (6) and (7)) used for RGB-D data.

Quantifying Viewpoint Change. To categorize the difficulty of a pair, we use a scalar overlap score ω ∈ [0,1] as a proxy for co-visibility (higher means more similar views). For pairs derived from RGB-D data, the overlap ω is defined as the proportion of validly matched pixel pairs M relative to the total number of pixels in each frame with height H and width W:

ω = |M| H × W

(10)

For pairs derived from SfM data, we define this overlap based on the proportion of shared 3D landmarks. Let L1 and L2 be the sets of 3D landmarks (i.e., 3D points) visible in images I1 and I2, respectively. The overlap ω is computed as:

ω = |L1 ∩ L2| min(|L1|,|L2|)

(11)

where | · | denotes the cardinality of the set. The viewpointchange magnitude is then defined as

∆v = 1 − ω, (12)

which is monotonically larger for more disparate viewpoints. In curriculum-based generation, stages specify an

admissible overlap interval [ωs,ωs], and a sample is routed to stage s if

I ωs ≤ ω ≤ ωs = 1. (13) Query Construction. Given a raw pair routed to stage s via its overlap score (Eq. (13)), we take its set of candidate matches M = {(xk,yk)} generated from either the RGBD or SfM pipeline. We then select a spatially diverse core subset C ⊂ M under stage-specific sparsity constraints. By default, the tool uses a cluster-then-prune policy: matches are clustered in the joint space with DBSCAN at radius ε = α · τmin-dist (with α ∈ (0,1)), one representative per cluster is kept (closest to the cluster centroid), and a greedy max-spacing pass further reduces to at most K points if needed. Core matches are given a one-to-one mapping by sampling label subsets LcoreA ,LcoreB of equal size, which satisfy the bijective constraint

fcore : LcoreA ↔ LcoreB . (14) To increase difficulty, we can optionally add distractor points DA,DB sampled from the leftover matches in each view, ensuring no overlap with core matches. Shuffling is applied to the labels in each view to avoid positional bias. Finally, the benchmark sample is packaged as

(I1,X = CA ∪ DA; I2,Y = CB ∪ DB), (15) with ground-truth mapping

f : LA → LB ∪ {∅}, (16) where f extends fcore by assigning distractors to ∅.

##### 7.2. Dynamic Curriculum Strategy

Verifiable Reward Design. Our reward function comprises two components that jointly encourage geometric accuracy and structured reasoning. Given the predicted mapping fˆ : {1,...,n} → {1,...,m} ∪ {∅} and groundtruth mapping f∗ with the same domain, where n is the total number of query regions (including those with no correspondences in ground truth), we define:

Format compliance rformat ∈ {0,1,2} verifies both structural and syntactic validity:

rformat = [structure] + [JSON], (17)

where the structure component checks that the output follows the required format with reasoning confined within thinking tags, and the JSON component verifies that the content within answer tags forms a valid JSON mapping with fˆ(i) ∈ {1,...,m} ∪ {∅} for all i ∈ {1,...,n}.

Matching correctness rmatch ∈ [0,1] measures prediction accuracy over all n query regions:

n

1 n

rmatch =

i=1

f ˆ(i) = f∗(i) , (18)

which counts exact agreements including correct predictions of unmatched regions (fˆ(i) = ∅ when f∗(i) = ∅). This formulation ensures that all regions are evaluated, rewarding comprehensive spatial reasoning.

The final reward combines these components:

r = wf · rformat + wm · rmatch, (19)

wf and wm being reward weights. Both components must be satisfied for high rewards, encouraging the model to produce well-structured and geometrically accurate predictions.

##### 7.3. Curriculum Learning Implementation Details

Our curriculum design operates at two hierarchical levels: (1) three cardinality settings (L1, L2, L3) that define matching task configurations, and (2) three training stages that progressively combine these settings with increasing complexity.

Cardinality settings. We define three cardinality configurations corresponding to different matching scenarios:

- • L1 (Unambiguous matching): n ∼ U(3,5) matchable points with |Xdist| = |Ydist| = 0. This yields 3–5 one-toone correspondences without distractors.
- • L2 (Selective matching): n ∼ U(1,2) matchable points with |Xdist| = 0 and |Ydist| ∼ U(3,6). The model must identify 1–2 correct matches among 4–8 candidates in Y.
- • L3 (Partial matching): n ∼ U(3,6) matchable points with |Xdist|,|Ydist| ∼ U(3,6). Both views contain 6–12 points with bidirectional occlusion.

Three-stage curriculum progression. Training progresses through three stages that combine cardinality settings with increasing complexity:

- • Stage 1 (L1 only): The model trains exclusively on unambiguous matching (L1) to establish foundational geometric reasoning. Stage transition occurs when the sliding window reward r¯ > 0.7 for 10 consecutive evaluations.
- • Stage 2 (L2 only): After mastering L1, training shifts entirely to selective matching (L2) to learn distractor rejection. The model progresses when r¯ > 0.7 is maintained.
- • Stage 3 (L1/L2/L3 mixed): The final stage samples from

all three settings with probability pL1 : pL2 : pL3 = 0.3 : 0.3 : 0.4, ensuring comprehensive exposure to diverse matching scenarios.

Adaptive demotion: If performance degrades (r¯match < 0.2 for 10 consecutive steps) during any stage, the curriculum temporarily reverts to the previous stage for stabilization before resuming progression.

Spatial distribution refinement within settings. Independent of the curriculum stage, each cardinality setting (L1, L2, L3) undergoes its own spatial refinement process. This creates a two-dimensional curriculum: the stage determines which cardinality settings are active, while spatial refinement modulates difficulty within each active setting.

For each cardinality setting, we progressively eliminate local spatial cues:

###### • Initial sampling (clustered): Points are sampled using

DBSCAN clustering at radius ε = α · τmin-dist followed by greedy max-spacing selection. Initially, α = 2.0 and τmin-dist equals the minimal non-overlap margin for annotations, producing spatially coherent clusters.

###### • Progressive tightening: Every time a level’s internal reward threshold is satisfied, we update:

τmin-dist ← max(safe margin,τmin-dist − 20) pixels (20)

This gradually decreases minimum point separation and reduces clustering radius.

###### • Final sampling (dispersed): When the final threshold is achieved inside a level, sampling transitions to pure greedy max-spacing, maximizing spatial dispersion and eliminating local context cues.

Importantly, spatial refinement operates independently for each cardinality setting. When Stage 2 begins, L2 starts with clustered sampling even though L1 may have progressed to dispersed sampling. This ensures each matching configuration is learned from coarse to fine spatial distributions.

##### 7.4. Prompt Design

We design a structured prompt that clearly specifies the cross-view matching task, input format, and expected output structure. As illustrated in Fig. 4, the prompt instructs the model to identify corresponding keypoint locations in a target image given marked query points in the source image. The prompt explicitly defines reasoning requirements, output format requirements (JSON structure with match validity flags). This design ensures consistent task interpretation across different models while enabling verifiable evaluation through structured output parsing.

#### 8. Ablation Studies

Reinforcement Learning vs. Supervised Fine-tuning. To assess our curriculum-based reinforcement learning approach, we compare against supervised fine-tuning (SFT) on the same cross-view matching data. The SFT baseline uses 300 steps of supervised training with teacher-forced matching predictions, while our DCRL method employs reinforcement learning with holistic geometric rewards as described in Sec. 3.

- Table 7. Ablation study on general vision-language understanding benchmarks. Both SFT and DCRL maintain strong performance on general tasks, with DCRL showing slight advantages in our evaluation.

Method MME-RealWorld MMStar RealWorldQA V*

Base Model 62.8 59.8 69.5 84.8 SFT 62.4 56.4 68.8 82.2

Ours (DCRL) 63.8 62.5 70.5 85.9

- Table 8. Ablation study on spatial intelligence and geometric reasoning benchmarks. DCRL substantially outperforms SFT, particularly on tasks requiring fine-grained geometric correspondence (SAT, ReasonMatch), which is consistent with a benefit from curriculum-based reinforcement learning for spatial reasoning.

Method OmniSpatial MindCube SAT ReasonMatch

Base Model 43.6 40.0 70.0 27.5 SFT 42.6 45.1 41.3 51.0

Ours (DCRL) 48.9 43.5 75.3 70.5

General vision-language understanding. As shown in Table 7, DCRL maintains strong performance on general vision-language benchmarks and modestly improves over both the base model and the SFT baseline. While SFT degrades on MMStar (−3.4 points) and V* (−2.6 points), potentially due to catastrophic forgetting under domain shift, our RL-based approach preserves and slightly improves general capabilities (+2.7 on MMStar, +1.1 on V*). This result suggests that curriculum-based reinforcement learning with geometric rewards can strengthen spatial training without harming pre-existing vision-language understanding.

Spatial intelligence and geometric reasoning. Table 8 shows notable advantages of RL over SFT on spatial reasoning tasks. On our cross-view matching benchmark (ReasonMatch), DCRL achieves 70.5% compared with 51.0% for SFT, a 19.5-point improvement. The contrast is even larger on SAT (+34.0 points), where SFT drops from the base model (70.0 → 41.3) while RL improves over it (70.0 → 75.3). This difference is consistent with the view that teacher-forced imitation of specific matching patterns may be less robust when task distributions differ.

One possible explanation for the performance contrast is the difference in training dynamics. SFT encourages the model to imitate exact correspondence patterns from training data, which may create rigid associations that generalize poorly and can override useful pre-existing knowledge. In contrast, RL permits exploration of diverse matching strategies guided by holistic geometric feedback, which may help preserve prior capabilities while improving the target task. The dynamic curriculum may further support this process by progressively exposing harder scenarios rather than rely-

#### Matching Prompt

- • You are given two images of the same physical scene, each having several regions annotated with circles and IDs (e.g., ”1”, ”2”, ”3”).Your task is to identify the underlying correspondence between regions in the two images. Note that maybe not every region in the first image has a match in the second.Please provide your response in two parts: thinking process and final answer, each wrapped by some special tags.
- • thinking process: Your analysis where you show your analyzing and thinking process wrapped in <thinking> </thinking> tags, which includes but not limited to:

- – 1. Describe Visual Regions: For each annotated area in both images, describe the key visual characteristics and **spatial context within the scene**, including: - **Intrinsic properties**: color, shape, texture, size, material - **Spatial relationships in the 3D scene** (NOT pixel coordinates): * What objects/structures are directly adjacent to this region? (e.g., ”attached to a wooden door”, ”sitting on a metal shelf”) * What is this region positioned relative to in the physical space? (e.g., ”below the

window”, ”behind the chair”, ”left side of the bookshelf”) * Semantic context: What functional area or object group does it belong to? (e.g., ”part of the dining area”, ”on the workspace desk”) - **Avoid** describing regions by their pixel locations (top-left, center-right, etc.) unless necessary for disambiguation - Focus on **scene-level landmarks** as reference points (e.g., ”near the entrance”, ”opposite to the main table”, ”in the corner with the lamp”)

- – 2. Compare Viewpoints: Analyze the geometric or perspective relationship between the two images, such as: - Overview of the two images’ contents, focusing on **how the physical scene layout appears** in each view - **Camera transformation**: rotation angle (e.g., ”camera rotated 90° clockwise around the room center”), translation (e.g., ”camera moved closer to the left wall”), zoom/scale differences - **Occlusion changes**: which objects/regions become visible or hidden due to viewpoint change? - **Perspective distortion**: how do spatial relationships appear to change due to different viewing angles? (e.g., ”objects on the right side now appear more frontal”)
- – 3. Infer Matching Relations: Based on region appearance and **scene-relative spatial relationships**, establish correspondences: - Iterate through each annotated region in image 1, and for each one, compare it sequentially with every annotated

region in image 2 to infer the matching likelihood and reasonableness of region correspondences between the two images Prioritize matching based on **what surrounds each region** and **3D spatial context** rather than 2D image positions - Use **stable scene anchors** (walls, large furniture, architectural features) to reason about region identity across views - Consider how the viewpoint change transforms the **spatial relationships** you identified in step 1 - Example reasoning: ”Region A-1 is next to a red door and below a window. Region B-3 is also adjacent to the same red door (now seen from a different angle) and below the same window structure, so A-1 matches B-3”

- • final answer: Your final answer based on your thinking part wrapped by <answer> </answer> tags. The JSON object is a mapping of region IDs from image A (as string keys) to the corresponding region IDs in image B (as string values). **For regions in A that have no match in B, use ”none” as the value.**

Figure 4. Prompt For Wide-Baseline Matching Task.

ing on a single fixed difficulty level.

Across spatial benchmarks, DCRL improves over the base model on all four reported tasks (+43.0 on ReasonMatch, +5.3 on SAT, +5.3 on OmniSpatial, and +3.5 on MindCube). By contrast, SFT shows mixed behavior: it improves on ReasonMatch (+23.5) and MindCube (+5.1), is roughly flat on OmniSpatial (−1.0), and degrades substantially on SAT (−28.7). This pattern suggests that supervised imitation may be less robust for transferable spatial reasoning in this setting, while our RL approach produces gains that transfer more consistently across the reported geometric benchmarks.

Curriculum Learning. To assess the importance of our dynamic three-dimensional curriculum, we compare against

three ablated training strategies: (1) uniform sampling without curriculum, (2) training only on easy samples (first quartile by viewpoint overlap), and (3) training only on hard samples (last quartile with minimal overlap). All variants use identical RL training setup and total training steps.

As shown in Table 9, our dynamic curriculum achieves 70.5% F1, outperforming uniform sampling (65.3%, +5.2 points). This result suggests that progressively increasing viewpoint divergence, correspondence complexity, and spatial distribution challenges can be more effective than treating all samples equally. A plausible explanation is that the curriculum allows the model to build useful intermediate competence on simpler configurations before tackling extreme viewpoint changes.

Training exclusively on easy or hard samples yields sig-

- Table 9. Ablation study on curriculum learning. Progressive difficulty adjustment across viewpoint divergence and correspondence complexity improves performance over uniform sampling and fixed-difficulty training.

Training Strategy ReasonMatch No Curriculum (uniform cardinality) 65.3 Only 1/4 minimal ∆v samples (Easy) 59.9 Only 1/4 maximum ∆v samples (Hard) 62.3 Dynamic Curriculum (Ours) 70.5 ∆ (Ours vs. No Curriculum) +5.2 ∆ (Ours vs. Easy) +10.6 ∆ (Ours vs. Hard) +8.2

nificantly worse results. The easy-only strategy (59.9%) underperforms uniform sampling by 5.4 points, suggesting that limiting training to small viewpoint changes and simple correspondence structures is insufficient in this setup. Conversely, training only on hard samples (62.3%) also underperforms uniform sampling (-3.0 points), suggesting that starting with extreme viewpoint divergence and complex correspondences is also less effective than gradual difficulty scheduling.

The substantial gaps between curriculum and singledifficulty training (+10.6 over easy-only, +8.2 over hardonly) are consistent with the value of progressive learning. Our curriculum’s adaptive adjustment mechanism monitors sustained performance improvements before increasing difficulty, helping the model stabilize before moving to harder settings. This staged progression across three dimensions (viewpoint span, correspondence complexity, and spatial distribution) is also consistent with the transfer gains observed on OmniSpatial, MindCube, and SAT.

Interestingly, hard-only training (62.3%) slightly outperforms easy-only (59.9%), suggesting that exposure to challenging cases, even without curriculum structure, may provide more useful learning signals than trivial examples. However, both approaches remain well below curriculum learning, indicating that progressive difficulty scheduling is beneficial in this setting.

#### 9. ReasonMatch-Bench Results

In this section, we report detailed results on ReasonMatchBench and analyze how different multimodal LLMs behave under our cross-view region matching task. We first summarize the overall quantitative performance across scenarios and difficulty levels and then provide a fine-grained comparison along several error dimensions where models exhibit the largest discrepancies. Finally, we briefly discuss a visualization that highlights these differences.

##### 9.1. Overall Quantitative Performance

Table 1 reports F1, Precision, and Recall on ReasonMatchBench for a broad range of multimodal LLMs, together with per-scenario scores on three difficulty levels (Indoor/Outdoor/Object, L1–L3).

ReasonMatch-Bench is challenging for strong generalpurpose MLLMs. Even the best general-purpose models only achieve F1 scores in the 50–60 range on our benchmark. For example, GPT-5-mini obtains an overall F1 of 57.9, while GPT-5-Chat and Qwen3-VL-235B reach 51.5 and 49.2, respectively. Other strong closed-source systems such as Gemini-2.5-Pro, Claude-4.5-Sonnet, and GPT-4o remain in the 33–43 F1 range. This indicates that ReasonMatch-Bench is substantially harder than standard VQA-style benchmarks, even for large commercial MLLMs.

DCRL turns a weak open-weight model into the best performer. The baseline performs poorly on ReasonMatch-Bench (27.5 %), substantially below both commercial models and the larger Qwen3-VL-235B (49.2 %). After applying our DCRL training, the same 8B backbone achieves 70.5% F1, outperforming all other models in the table by a large margin. This suggests that targeted training on ReasonMatch-Bench-style tasks can, in this setting, be more effective than simply scaling up model size.

Performance consistently drops with difficulty, especially on Object L3. Across models, scores tend to decrease from L1 to L3 within each scenario, indicating that our difficulty annotation captures genuine changes in task hardness. For instance, GPT-5-mini drops from 68.2 to 47.0 on Indoor (L1→L3) and from 75.8 to 51.4 on Outdoor, and similar declines appear for other models. The Object L3 column is consistently the hardest setting, and our Qwen3VL+DCRL model still struggles on this regime (33.7) compared to easier levels, but remains clearly above all baselines.

##### 9.2. Fine-Grained Error Analysis Across Models

Although Table 1 reflects how often models select the correct match, it does not explain why they fail. To better characterize model behavior, we use Qwen3-VL-235B as a blind evaluator to score the complete ReasonMatch-Bench outputs of the five models shown in Figure 5. For each example, the evaluator receives the two annotated images, the ground-truth mapping, the verbatim reasoning trace, and the parsed final answer, and assigns discrete scores in {0,1,2} to a broader rubric, where 2 indicates no issue and 0 indicates a severe issue. Below, we report the five dimensions

- Table 10. Human study on the 90 largest view divergence samples across three datasets. Human annotators achieve substantially higher performance than all tested models, with the performance gap most pronounced on object-centric scenes (uCO3D).

Method Overall DL3DV RE10K uCO3D

P R F1 P R F1 P R F1 P R F1 Closed-Source Models

GPT-5-mini 34.5 40.7 37.2 32.5 40.4 35.9 47.4 52.7 49.7 23.6 29.1 25.8 Gemini-2.5-Pro 28.3 31.1 29.5 25.1 28.3 26.5 42.8 45.6 44.1 17.1 19.3 18.0 Claude-4.5-Sonnet 22.1 26.7 24.0 20.6 24.8 22.2 28.1 33.6 30.5 17.4 21.6 19.2

Open-Source Models

Qwen3-VL-235B 27.5 33.0 29.9 23.2 28.1 25.3 42.3 50.2 45.7 17.1 20.9 18.7 Ours (DCRL) 48.3 56.8 52.0 53.3 63.2 57.7 65.7 76.6 70.6 25.9 30.4 27.8 Human 87.5 81.3 84.0 94.3 93.3 93.5 96.1 93.9 94.7 72.2 56.6 62.1 ∆ (Human vs. Ours) +39.2 +24.5 +32.0 +41.0 +30.1 +35.8 +30.4 +17.3 +24.1 +46.3 +26.2 +34.3

with the clearest cross-model differences: Local-Cue Reliance (F1), Global Layout Misalignment (F2), ReasoningAnswer Mismatch (F4), Overuse of “None” (F5), and Reasoning Coherence.

Local-Cue Reliance (F1). This score measures how well a model can integrate correct local observations into a globally consistent cross-view mapping. Low F1 reflects the typical failure pattern where the chain-of-thought describes the correct shelf, tier, or surrounding objects, but the final correspondence still points to a wrong region. Models with higher F1 scores are better at binding fine-grained visual cues into a coherent global decision.

Global Layout Misalignment (F2). F2 captures largescale geometric errors such as swapped left/right ordering, incorrect vertical tiers, or inconsistent depth relations between regions. This dimension directly measures a model’s ability to maintain a viewpoint-consistent global layout rather than reasoning in a purely local, patch-based manner.

Reasoning-Answer Mismatch (F4). F4 quantifies inconsistencies between the model’s chain-of-thought and its final structured output. A low score indicates cases where the reasoning text identifies the correct target region, but the final JSON or <answer> block encodes a different ID or swaps indices. This dimension probes how reliably a model can translate its internal reasoning into a precise, machineusable prediction.

Overuse of “None” (F5). ReasonMatch-Bench includes regions that legitimately have no correspondence in the other view. F5 measures whether the model uses the “none” option in a calibrated way. Over-using “none”—especially on regions that do have correct matches—reveals an overly

conservative behavior: the model can often describe the local content but refuses to commit to a specific correspondence.

Reasoning Coherence. Finally, we rate the overall logical coherence of the chain-of-thought: whether observations are ordered in a reasonable way, whether there are self-contradictions, and whether the final decision is clearly connected to previous observations. This dimension complements F1/F2 by highlighting cases where the answer is correct but the reasoning is disorganized, as well as cases where an apparently detailed reasoning trail actually does not support the final mapping.

Figure 5 summarizes these five dimensions across all evaluated models. We report mean rubric scores in [0,2], so that higher values consistently correspond to better behavior (fewer or milder errors). Several patterns emerge:

- • Models differ most strongly on F1 and F2, suggesting that binding local cues into a globally consistent spatial alignment is the central bottleneck on ReasonMatch-Bench.
- • F4 reveals that some models frequently “think correctly but answer incorrectly”, exposing weaknesses in reasoning-to-action alignment that are invisible to raw F1 / Precision / Recall.
- • Differences in F5 highlight distinct calibration behaviors: some models are overly cautious and tend to abstain by predicting “none”, while others are more willing to commit to concrete correspondences.
- • The logical coherence scores show that even when overall accuracy is similar, models can vary substantially in how structured and self-consistent their reasoning is.

Overall, these error dimensions provide a complementary perspective to the aggregate metrics in Table 1. They show that current multimodal LLMs are limited not only by perception, but also by their ability to maintain a globally

[Figure 35]

- Figure 5. Comparison of five models across key error dimensions: Local-Cue Reliance (F1), Global Layout Misalignment (F2), Reasoning-Answer Mismatch (F4), Overuse of “None” (F5), and Reasoning Coherence. Higher bars indicate fewer or milder errors.

consistent 3D picture of the scene, propagate that picture through a coherent reasoning process, and translate it into a precise region-level correspondence.

Qualitative examples. To make the above error dimensions more concrete, we highlight a representative case for F2 from ReasonMatch-Bench. In a ScanNet indoor scene (scannet 000751), several floor markers form a small cluster that appears in both views under essentially the same physical configuration. The ground truth correspondence in Image B is region B-1, which is the leftmost marker in this

cluster. However, the model often selects region B-3 and describes it as the leftmost marker. This behavior indicates that the model has roughly identified the correct cluster, but fails to maintain a consistent left–right ordering when projecting the 3D layout into the two viewpoints, which is precisely what F2 is designed to capture.

##### 9.3. Visualization of Model Behaviors

To make the above differences more concrete, Figure 5 provides a compact visual summary of model behaviors along five error dimensions: Local-Cue Reliance (F1), Global Layout Misalignment (F2), Reasoning-Answer Mismatch (F4), Overuse of “None” (F5), and Reasoning Coherence.

This visualization highlights which aspects of spatial reasoning each model handles well and where it fails:

- • Models with high scores on F1 and F2 are able to build a coherent 3D understanding from two wide-baseline views, rather than relying on local texture cues alone.
- • Low F4 and logical coherence expose brittle reasoning pipelines, where correct evidence is not reliably turned into the correct structured prediction.
- • Variations in F5 reveal different risk profiles: some models trade recall for precision by overusing “none”, while others achieve better coverage at the cost of more misaligned matches.

In combination with the quantitative scores in Table 1, this figure shows that ReasonMatch-Bench is not just a harder matching dataset, but also a targeted probe of multiview spatial understanding, calibration, and reasoning quality in multimodal LLMs.

To further illustrate how models behave on individual instances, Figures 6 and 7 show full chain-of-thought traces and JSON outputs on two representative successful examples from ReasonMatch-Bench.

##### 9.4. Human Study

To assess how the task aligns with natural human spatial reasoning and to measure the gap between human performance and current models, we conduct a human study on a stratified 90-sample subset with the largest view divergence. Two non-expert annotators, i.e., ordinary participants without task-specific training, complete the task independently using the same instructions as the models. They do not discuss their answers during annotation, and we report the average of their scores as the final human result.

Despite not having specialized training, the human annotators achieve an overall F1 of 84.0%, substantially outperforming all tested models, including frontier systems (Table 10). The 32-point gap between the non-expert annotators (84.0%) and our best model (52.0%) highlights a large remaining challenge in fine-grained geometric correspondence.

Performance patterns differ markedly across scene types.

On structured indoor and outdoor environments, humans achieve near-perfect accuracy (F1 > 93%), likely aided by rich contextual cues and stronger geometric regularity. Both humans and models struggle more on object-centric scenes (uCO3D), where humans achieve 62.1% F1 compared with 27.8% for our best model. Analysis of human annotations suggests that errors primarily occur when objects exhibit severe self-occlusion or lack distinctive surface features, which are also challenging conditions for current models. This pattern is consistent with the view that the benchmark captures genuine spatial reasoning challenges rather than artifacts of imperfect ground truth.

The human-model gap underscores substantial limitations in current MLLMs’ spatial reasoning. While DCRL improves over strong baselines (+14.8% over GPT-5-mini) and reaches 62% of non-expert human performance overall, there remains considerable room before models approach human-level geometric reasoning. The fact that non-expert annotators can solve many of these tasks reliably further motivates the benchmark and continued work on spatial understanding in MLLMs.

#### 10. Limitations and Future Work

Despite substantial improvements over baselines, our approach reveals both the promise and challenges of developing human-level spatial intelligence in MLLMs. Our best model achieves 52.0% F1 compared to untrained humans’ 84.0%, with the gap most pronounced on objectcentric scenes where even humans struggle (27.8% vs. 62.1%). This persistent performance gap highlights fundamental limitations in current architectures’ ability to perform fine-grained geometric reasoning under extreme viewpoint changes and limited contextual cues.

Our work focuses on pairwise cross-view matching as a foundational capability, but comprehensive spatial intelligence demands reasoning over multiple views simultaneously, integrating geometric correspondence with 3D scene understanding, temporal dynamics, and semantic knowledge. Future research should extend beyond pairwise matching toward holistic multi-view reasoning that mirrors human spatial cognition—synthesizing information across viewpoints to construct coherent 3D mental models.

[Figure 36]

[Figure 37]

[Figure 38]

Two images show the same scene from different viewpoints. Match the region IDs from Image A to Image B

[Figure 39]

<thinking> The two images depict the same outdoor basketball court from different camera angles. The first image is a wide shot from the side of the court, while the second image is a closer, more centered perspective looking down the length of the court.

- Region 1 in the first image is the backboard and hoop assembly on the far left side of the court, near the hedge. In the second image, this same structure is now visible from the front, and is labeled as region 1.
- Region 2 in the first image is the base of a basketball hoop on the far side of the court, near the fence. In the second image, this same base is visible from the front and is labeled as region 3. The second image provides a clearer view of the court's layout, showing multiple hoops and their bases. Region 4 in the second image is the base of the hoop on the left side of the court, which corresponds to the base of the hoop in the first image (region 1). Region 5 in the second image is the backboard and hoop assembly on the far right side of the court, which corresponds to the hoop in the first image (region 2). The spatial relationships are consistent: the hoop bases are positioned along the court's perimeter, and the backboards are mounted on top of the poles. The camera has rotated to face down the court, making previously side-viewed objects now appear more frontal. </thinking> <answer> ```json { "1": "1", "2": "3" } ``` </answer>

- Figure 6. Qualitative example on ReasonMatch-Bench (Example 1). The model is given two views of the same scene and asked to match region IDs from Image A to Image B. Its chain-of-thought correctly identifies the corresponding regions across viewpoints and produces a consistent JSON mapping for the cross-view region matching task.

[Figure 40]

[Figure 41]

[Figure 42]

Two images show the same scene from different viewpoints. Match the region IDs from Image A to Image B

[Figure 43]

<thinking> The two images depict the same industrial ceiling scene from different camera angles. The first image is a wider shot looking up at the ceiling, while the second image is a closer, more centered view of the same area, showing more details of the crane and ceiling structure.

- Region 1 in the first image is a small green circle on the corrugated metal ceiling, located near the bottom right of the frame, adjacent to a long fluorescent light fixture. In the second image, this same region is now visible as region 3, which is also on the ceiling, directly below the crane's hoist cable and near the same light fixture.
- Region 2 in the first image is a green circle on the red crane beam, specifically highlighting the point where the hoist motor is attached. In the second image, this corresponds to region 1, which is also on the crane beam, directly above the hoist motor and next to the Chinese characters "品质优先" (Quality First).
- Region 3 in the second image is a green circle on the ceiling, located directly below the crane's hoist cable. This region is not present in the first image.
- Region 4 in the second image is a green circle on the ceiling, located above the crane beam and near the top of the frame. This region is not present in the first image.
- Region 5 in the second image is a green circle on the ceiling, located directly below the crane's hoist cable and near the same light fixture as region 3. This region is not present in the first image. Based on the spatial relationships and scene context, the matching is as follows:

- - Region 1 in the first image matches region 3 in the second image, as they are the same physical location on the ceiling, viewed from different angles.
- - Region 2 in the first image matches region 1 in the second image, as they are the same physical point on the crane beam, viewed from different angles. </thinking> <answer> ```json { "1": "3", "2": "1" } ``` </answer>

- Figure 7. Qualitative example on ReasonMatch-Bench (Example 2). Another successful case where the model aligns regions across widebaseline views, using textual reasoning over geometric layout and shared objects to produce a correct JSON correspondence. Together with Figure 5, these examples illustrate how multimodal LLMs solve our pixel-anchored cross-view matching task when their spatial reasoning is reliable.

