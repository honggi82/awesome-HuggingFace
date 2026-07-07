# arXiv:2606.11129v2[cs.CV]23Jun2026

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

## WorldOlympiad: Can Your World Model Survive a Triathlon?

Yuke Zhao1∗, Wangbo Zhao3∗, Weijie Wang1∗, Zeyu Zhang2∗†, Dakai An3, Akide Liu4, Yinghao Yu5, Jiasheng Tang2‡, Fan Wang2, Wei Wang3, Bohan Zhuang1‡

1Zhejiang University 2DAMO Academy, Alibaba Group 3The Hong Kong University of Science and Technology 4Monash University 5TRE, Alibaba Group ∗Equal contribution., †Project lead., ‡Corresponding authors.

We introduce WorldOlympiad, a benchmark for diagnosing video-based world models across physical faithfulness, geometric consistency, and interaction fidelity. While existing benchmarks often focus on visual quality, semantic alignment, or short-term temporal coherence, they provide limited insight into whether generated videos obey physical rules, preserve coherent 3D structure, and sustain controllable interactions over long horizons. To address this gap, WorldOlympiad decomposes world-model evaluation into three complementary dimensions. The physical track uses object segmentation and MLLM-as-judge to assess whether generated videos follow interpretable rules in mechanics, thermal phenomena, and material properties. The geometry track reconstructs generated videos with Gaussian splatting and evaluates structural consistency, cross-view coherence, and camera-trajectory alignment. The interaction track assesses whether generated rollouts follow complex action prompts and maintain smooth, coherent transitions across consecutive video chunks. WorldOlympiad further covers three major downstream scenarios, including gaming, robotics, and general real-world videos, capturing diverse challenges from interactive control and embodied manipulation to open-domain motion and camera dynamics. Together, these tracks and scenarios form a scalable and interpretable evaluation suite that exposes failure modes beyond generic video quality. Experiments on state-of-the-art models reveal substantial gaps in physical reasoning, 3D consistency, and long-horizon interaction, underscoring the need for more structured evaluation protocols for generative world models.

Date: June 24, 2026 Project website: https://alibaba-damo-academy.github.io/WorldOlympiad Code: https://github.com/alibaba-damo-academy/WorldOlympiad Correspondence: jiasheng.tjs@alibaba-inc.com, bohan.zhuang@gmail.com

1 Introduction

Recent advances in video generation [36, 34, 17, 46, 3, 4] have expanded its scope from passive content creation to video-based world modeling. A video-based world model is expected to predict future visual states from historical observations and control signals, which is crucial for game simulation [6, 51, 11], robotic policy development [1, 2, 7], and real-world scene generation [26, 35, 40, 31, 45]. In these applications, high visual fidelity alone is insufficient: models must preserve state continuity, respect physical and geometric constraints, respond to user actions, and maintain plausible dynamics over long generation horizons. Together, these requirements make developing a comprehensive framework for evaluating video-based world models across multiple capability dimensions a key challenge.

To address this challenge, a straightforward solution is to reuse evaluation approaches originally designed for traditional video generation models, such as VBench [15], VBench-2.0 [54], and CLIP-based metrics [45, 14, 22]. However, these approaches primarily measure perceptual quality or text–video alignment on short videos and fail to fully capture core world-modeling properties. Although the the subsequent VBench++ [16] extends evaluation to long-video generation, it still focuses largely on visual appearance and temporal smoothness, leaving key world-modeling capabilities underexplored. In particular, existing approaches pay limited attention to whether generated videos can consistently adhere to physical laws, preserve coherent 3D structure, and

[Figure 5]

[Figure 6]

[Figure 7]

###### WorldOlympiad

[Figure 8]

Physical

[Figure 9]

physical rules

[Figure 10]

[Figure 11]

Physical pass rate: 12/14

[Figure 12]

SAM

[Figure 13]

[Figure 14]

Object Segment

bbox & mask video

MLLM

[Figure 15]

Geometry

[Figure 16]

[Figure 17]

[Figure 18]

DA

[Figure 19]

Geometry score

Depth Estimation

[Figure 20]

MLLM

Meta-view Render frames Camera trajectory

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

Interaction

single chunk

CLIP

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

Interaction clip score score

[Figure 30]

[Figure 31]

Alignment Model

chunk transition

[Figure 32]

[Figure 33]

[Figure 34]

video clips

MLLM

[Figure 35]

[Figure 36]

[Figure 37]

whole video

- Figure 1 Overview of the WorldOlympiad pipeline for data collection, long-video generation, and multi-dimensional evaluation.

support controllable interactions over long horizons.

Moreover, recent benchmarks specifically designed for video-based world models often focus on a single downstream domain, such as gaming [47] or robotics [29, 9, 23], making it difficult to compare models under a unified protocol across gaming, robotics, and general real-world scenarios. As a result, current benchmarks still cannot fully answer a central question: Can existing video-based world models reliably simulate world dynamics across multiple domains, over long horizons, and in interactive settings?

Building on prior efforts, we identify two key challenges in evaluating video-based world models: developing capability-oriented metrics and ensuring task diversity. Evaluation should span heterogeneous domains, while the metrics should assess whether generated videos maintain long-horizon consistency, obey physical laws, preserve 3D geometry, and respond faithfully to control signals. Although existing benchmarks partially address these challenges, they do not comprehensively cover both aspects, as summarized in Table 1.

To this end, we introduce WorldOlympiad, a unified benchmark for evaluating video-based world models across gaming, robotics, and real-world scenarios. WorldOlympiad is constructed from 1,000 high-quality long videos that cover diverse downstream requirements, including interactive control in games, embodied manipulation in robotics, and open-domain motion and camera dynamics in real-world scenes. Each video is processed into structured evaluation instances with scenario-specific prompts and chunk-level temporal descriptions, enabling controlled long-video generation and fine-grained diagnosis. Built on this benchmark, we evaluate generated videos from three complementary perspectives: physical faithfulness, geometric consistency, and interaction fidelity. With 8 representative long-video generation pipelines, WorldOlympiad reveals systematic limitations in current models and provides diagnostic evidence for developing more reliable video-based world models.

Our contributions are summarized as follows:

- • We propose WorldOlympiad, a unified benchmark for evaluating interactive long-video world models across gaming, robotics, and real-world scenarios.
- • We design multi-dimensional judge metrics that systematically assess physical-law adherence, 3D geometric consistency, and chunk-by-chunk interactive generation.
- • We construct a dataset of 1,000 high-quality long videos and benchmark 8 long-video generation pipelines, providing a systematic evaluation of their reliability in downstream world-model applications.

### 2 Related Work

- 2.1 Video Generation

Diffusion-based video generation models [3, 46, 17, 36, 34] have demonstrated emergent physical consistency through large-scale training [4, 39], including object permanence, 3D coherence, and plausible motion dynamics. Despite these compelling properties, many early diffusion-based video generators are optimized for short clips, often on the order of 5–10 seconds, which limits their direct use as persistent world model simulators. Recently, block diffusion has emerged as a promising paradigm for scalable long-horizon video synthesis. By performing iterative diffusion denoising within each block and conditioning on previously generated content via cross-block KV caching, this approach combines the high-quality parallel generation of diffusion models with the sequential consistency of autoregressive conditioning [14, 48, 55, 52]. Such a design preserves intra-block denoising quality while enabling scalable temporal extension, positioning block diffusion as a viable road toward video-based world models.

- Table 1 Comparison of existing benchmarks across evaluation metrics and video tasks.

Benchmark

Eval Metrics Video Tasks

Long Video Physical Geometry Interaction Gaming Robotics Real-world

VBench [15] ✗ ✗ ✗ ✗ ✗ ✗ ✓ VBench++ [16] ✓ ✗ ✗ ✗ ✗ ✗ ✓ VBench 2.0 [54] ✗ ✓ ✗ ✗ ✗ ✗ ✓ MIND [47] ✓ ✗ ✗ ✓ ✓ ✗ ✗ EWMBench [12] ✗ ✗ ✗ ✓ ✗ ✓ ✗ WorldEval [20] ✗ ✗ ✗ ✓ ✗ ✓ ✗ WorldArena [29] ✗ ✓ ✓ ✓ ✗ ✓ ✗

WorldOlympiad ✓ ✓ ✓ ✓ ✓ ✓ ✓

- 2.2 Video Generation Models as World Models

The rapid advancement of world models has enabled video generation to be deployed across diverse domains, including interactive game generation [35] and robotics simulation [1, 2]. In the gaming domain, models such as GameGen-X [6] and Matrix Game [51] have demonstrated compelling interactive game simulation with controllable character actions and environment dynamics. In robotics and embodied intelligence, dedicated interactive world models provide policy generation and data augmentation capabilities for robotic agents [1, 2, 7]. However, simultaneously maintaining persistent world state and supporting real-time interaction remains a significant challenge, giving rise to two core research directions. For memory and long-context modeling, some approaches adopt implicit memory mechanisms. For instance, LongLive [45] introduces KV caching to enable long-range consistent generation. In contrast, other works explicitly incorporate 3D memory mechanisms to preserve world-state consistency over extended horizons [43, 13, 19, 53, 50, 42, 38]. More recently, MosaicMem [50] and Inspatio World [33] have begun exploring hybrid memory mechanisms, demonstrating substantial promise. On the interactive generation front, the dominant paradigm adapts controllable video generation techniques within the block diffusion framework [24, 30], enabling interactive video synthesis. Works such as LingBot-World [35] have shown strong performance on downstream tasks such as interactive game generation through scaling. Regardless of the target application, real-time interaction remains a central capability requirement in this field.

- 2.3 World Model Benchmarks

Existing benchmarks for short video generation have introduced a broad set of general evaluation metrics, as exemplified by VBench [15] and its successor VBench 2.0 [54], which cover multi-dimensional criteria spanning visual quality, motion authenticity, semantic consistency, and physical plausibility. More recently, benchmarks specifically targeting world model capabilities have been proposed, evaluating models along dimensions such as physical law adherence, simulation fidelity, and functional world modeling [10, 27, 18, 49]. Newer

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

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

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

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

- Figure 2 Data collection overview across robotics, gaming, and real-world video sources.

benchmarks tailored to robotics downstream tasks [29, 9, 23] extend the evaluation scope to controllability, action conditioning, and closed-loop interaction. Despite this progress, existing benchmarks lack unified coverage across multiple downstream application domains, including gaming, robotics, and general scene generation, within a single evaluation framework. Moreover, assessments of interactive functionality, which is arguably the most critical capability of world models, remain notably absent. To address these gaps, we propose WorldOlympiad, a comprehensive benchmark that unifies the evaluation of game, robotics, and real-world environments, comprising 1,000 high-quality video samples spanning diverse downstream scenarios, and jointly assessing perceptual quality alongside functional world modeling capabilities.

- 3 WorldOlympiad

- 3.1 Data Collection

- Figure 2 summarizes the data collection process of WorldOlympiad. The benchmark contains 400 robotics videos, 400 gaming videos, and 200 real-world videos, covering complementary world-modeling requirements: robotics videos emphasize object manipulation and physical interaction, gaming videos emphasize interactive control and long-context state evolution, and real-world videos emphasize open-domain motion and camera dynamics. This diverse composition enables a comprehensive evaluation of video-based world models across their three most critical application domains.

- 3.1.1 Source Domains

Robotics Domain. The robotics subset is built from RoboCOIN [41], an open-source bimanual robotic manipulation data collection. We use this source because bimanual manipulation naturally contains object contact, gripper motion, state changes, and physically grounded interactions. RoboCOIN also includes multiple bimanual robot embodiments, giving the subset broad coverage for evaluating whether generated videos preserve action-consistent dynamics. From the downloaded RoboCOIN videos, we manually filter 400 videos as the robotics portion of the benchmark test set.

Gaming Domain. The gaming subset is built from GameGen-X [6], an interactive open-world game video dataset. We randomly sample videos from the official OGameData_50K.csv metadata file and download the corresponding videos. Since some gameplay videos are usually long and contain multiple interaction stages, we split long videos into shorter video chunks with 60 seconds before constructing the final 400-video gaming

- Table 2 Data composition of the WorldOlympiad benchmark test set. Domain Count Source Selection rule

Robotics 400 RoboCOIN [41] Downloaded videos that are manually filtered.

Gaming 400 GameGen-X [6] Randomly sampled videos from the official OGameData_50K.csv; long videos are split into shorter evaluation chunks.

Real-world 200 LVD-2M [44] Videos selected from ytb_600k_720p.csv with duration longer than 60 seconds and motion score greater than 50.

subset. This subset targets interactive world-modeling behavior such as camera movement, player navigation, combat events, skill execution, and game-state changes.

Real-world Domain. The real-world subset is built from LVD-2M [44], a long-take video dataset with temporally dense captions. We use the official ytb_600k_720p.csv subset and randomly select videos whose duration is longer than 60 seconds and whose motion score is greater than 50. This filtering rule favors long videos with sufficient visible motion, making the subset suitable for evaluating open-domain dynamics, camera movement, and geometric consistency in everyday scenes.

- 3.1.2 Temporal Chunking and Captioning

Detailed video captions are essential for subsequent evaluation. Instead of relying on a single-pass MLLM, we design a three-stage chunk-caption-refine pipeline to ensure the resulting annotations are both accurate and comprehensive, as illustrated in Figure 3. We adopt Gemini-3-Pro-Preview [8] across all stages, owing to its superior performance in multimodal understanding.

StageI-Chunking. The pipeline first identifies the main continuous execution interval in a video and divides it into at most six contiguous chunks. All chunks follow a left-closed, right-open interval convention, and adjacent chunks are required to have no temporal gaps or overlaps. For gaming videos, the chunking prompt focuses on gameplay execution such as combat, traversal, skill casting, and camera transitions; for real-world videos, the prompt focuses on continuous visual actions, object motion, interaction events, and view transitions.

StageII-Caption. After temporal chunking, the captioning model generates chunk-level captions for each video chunk. For each robotics, gaming, or real-world chunk, the captioning model outputs two fields: an action field and a caption field. The action field maps camera movement to WASD-style controls, with None used when the camera does not move noticeably. This action label is intentionally based on camera movement only; it is not inferred from character animation, subject motion, visual effects, or UI changes. The caption field describes the scene, visible entities, events, interactions, and outcomes in English.

StageIII-Refine. We then refine the chunk-level captions with the full video as context. Given the full video and the time-ordered chunk captions, the refinement step corrects hallucinated details, standardizes terminology across adjacent chunks, improves narrative continuity, and validates the camera-movement action label. This final pass is important for long-video evaluation because adjacent chunks often share objects, locations, player states, or scene context, and inconsistent captions would weaken the reliability of interaction and long-context assessment.

We take the outputs from Stage III as the final captions for each video chunk, which are subsequently used for evaluation. The active judge prompts used by WorldOlympiad are provided in Appendix A.

- chunk1
- chunk2

Actions: left robotic hand retracts... Scene: ...operating over a patterned mat... Camera: Stay static

[Figure 100]

[Figure 101]

[Figure 102]

gaming data

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

Actions: the right hand moves the gripped... Scene: ...operating over a patterned mat... Camera: Stay static

embodied data

Internet Data

general data

Video Chunks Chunk Caption Refined Caption

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

segmentation Per-chunk caption

Whole video refine

###### Figure 3 Data standardization pipeline from raw videos to refined action-caption annotations.

3.2 Evaluation Metrics

- 3.2.1 Physical Evaluation

We evaluate physical faithfulness with a rule-based benchmark spanning three subsets: mechanics, thermodynamics, and material properties. The pipeline first uses an MLLM to identify the moving or deforming entities that are most relevant to physical reasoning, and then applies SAM3 [5] to produce object-centric visualizations that expose their masks and trajectories more clearly. After this preprocessing stage, each metric is evaluated in two steps. A relevance judge first determines whether the target phenomenon is actually present in the ground-truth reference video under the given prompt; unrelated metrics are marked as not related and excluded from scoring. For each relevant metric, a compliance judge then compares the generated video with a ground-truth reference video and predicts whether the observed behavior follows the corresponding physical rule, together with a confidence score and a short explanation. Final physical results are reported by averaging compliance over the applicable metrics within each subset and then across subsets. The active mechanics, thermodynamics, and material rule prompt templates used by the physical MLLLM judges are provided in Appendix A.2.

Mechanics. Gravity evaluates whether unsupported objects move downward under gravity, rather than floating upward or accelerating in physically implausible directions. Buoyancy focuses on whether objects in fluids remain near the surface or sink in accordance with their apparent density. Compression measures whether solids deform plausibly under load, instead of staying unrealistically rigid or buckling without sufficient cause. Impact examines whether collisions lead to reasonable post-impact dynamics, including momentum transfer, rebound, fracture, or eventual rest.

[Figure 113]

###### Figure 4 Pipeline statistics for data processing, annotation coverage, and evaluation-ready samples.

Thermodynamics. Melting assesses whether a heated solid gradually transitions into a liquid state. Sublimation captures direct solid-to-gas transitions without an intermediate liquid phase. Vaporization considers whether liquids turn into vapor through evaporation or boiling when heated or exposed over time. Condensation evaluates the formation of liquid droplets from cooled gas. Deposition describes the direct transformation from gas to solid without first becoming liquid. Freezing measures whether a cooled liquid solidifies into a stable solid state.

Material. Color mixing evaluates whether mixed colored liquids or paints yield the expected resultant color. Solubility focuses on whether soluble substances disperse and dissolve into the solvent, rather than remaining intact. Hardness distinguishes whether soft materials bend or tear easily while hard materials resist deformation or break sharply. Combustibility examines whether flammable materials ignite and produce physically consistent fire, smoke, or charring behavior.

- 3.2.2 Geometry Evaluation

We evaluate geometric consistency with three complementary signals [37]: Srecon scores the rendered GaussianSplat video, Smeta scores a diagnostic meta-view, and Straj scores agreement between the recovered and reference camera trajectories. Given a generated video V = {It}Tt=1, we uniformly sample V¯ = {Ii}Ni=1, with N ≤ 32 in the implementation. When dynamic-object masks are available, foreground Gaussians are removed before rendering so that the 3D judge focuses on the static scene. Depth Anything 3 [21] estimates a Gaussian scene and camera parameters, and the Gaussian-Splat renderer produces two diagnostic artifacts:

FDA3(V¯) → G,{Ei,Ki}Ni=1 , VˆGS = R(G,{Ei,Ki}Ni=1), Iˆmeta = R(G,Ei⋆,Ki⋆), (1)

where G is the reconstructed Gaussian representation, Ei and Ki are recovered extrinsics and intrinsics, and i⋆ denotes the recovered camera pose farthest from the reconstruction origin.

The reconstruction and meta-view scores are produced by the same calibrated MLLM judge used in the implementation. The judge inspects whether the rendered static scene preserves a recognizable layout, coherent

- 3D structure, stable cross-view geometry, and prompt-consistent scene organization. The judge is instructed to return a strict JSON score in [0,1], and the parsed scores are clamped to [0,1] to avoid ambiguity with the CLIP model used in the interaction metric:

Srecon = clamp Jvid(VˆGS,p),0,1 , Smeta = clamp Jimg(Iˆmeta,p),0,1 , (2)

where p is the static-scene prompt used for 3D judging. In the optional LPIPS setting, the Gaussian-Splat video score is replaced by clamp(1 − LPIPS(VˆGS,V¯),0,1).

For camera motion, let {Tˆi}Li=1 and {Ti}Li=1 denote the predicted and reference camera-to-world trajectories after temporal resampling to a shared length. If the reference contains non-negligible translation, the predicted trajectory is first aligned to the reference by a similarity transform. Both trajectories are then expressed relative to their first frame:

T˜ˆi = Tˆ1−1Tˆi, i = 1,...,L. (3)

T˜i = T1−1Ti,

The translation score St combines path-shape similarity, motion-extent agreement, and mean camera-center error. The rotation score Sr combines mean geodesic rotation error, final-frame rotation error, and total rotation-extent agreement. The final trajectory score is computed by an adaptive aggregation function Amotion:

Straj = Amotion St,Sr;{T˜i}Li=1 . (4)

This aggregation is selected from the reference motion profile. For nearly static trajectories, the score penalizes reconstructed camera jitter directly. For translation-dominant or rotation-dominant trajectories, the corresponding component receives the larger weight; for mixed motion, translation and rotation are weighted evenly.

The implementation records the raw 3D reward as the sum of the three bounded subscores, while all tables report the normalized geometry score:

1 3

(Srecon + Smeta + Straj). (5)

S3D =

- 3.2.3 Interaction Evaluation

We evaluate interaction fidelity under the chunk-by-chunk generation setting. Given a generated video divided into T chunks {vi}Ti=1 and their corresponding captions {pi}Ti=1, the interaction benchmark measures whether each chunk follows its local instruction, whether adjacent chunks transition coherently, and whether the full video remains temporally fluent. This design matches the way interactive video world models are typically rolled out: each new chunk is conditioned on the previous visual context and a new control or action caption, so a model must satisfy both local caption alignment and long-range continuity.

The first component is a CLIP-based semantic-adherence score. For each chunk, we uniformly sample a fixed number of frames within its temporal interval Fi = {fi,j}m sampled frame and the corresponding chunk caption with a CLIP model [28, 45], convert both embeddings to unit-length vectors, and compute their dot-product similarity. The chunk-level score is the mean similarity over sampled frames,

- i
- j=1, where mi is 8 by default. We encode each

mi

1 mi

sclipi =

sim CLIPv(fi,j),CLIPt(pi) , (6) and the video-level semantic-adherence score is the weighted mean over all valid sampled frames:

j=1

T i=1

mi j=1 sim CLIPv(fi,j),CLIPt(pi)

. (7)

Sclip =

T i=1 mi

Because this is a cosine similarity computed from normalized CLIP embeddings, the raw score remains on its native [−1,1] scale. To use it as a bounded auxiliary interaction signal, we convert it into a calibrated semantic score with fixed thresholds:

Sclip − τmin τmax − τmin

,0,1 , τmin = 0.20, τmax = 0.40. (8)

Sclip = clip

The thresholds are fixed across all evaluated models, so adding a new model does not change previously reported CLIP auxiliary scores. This component provides an automatic and lightweight estimate of whether the generated chunks contain the semantic content requested by their captions.

The second component uses an MLLM as a structured rubric-based judge. We query the MLLM at three complementary levels, and all returned scores are clipped to the requested 0–5 range before being normalized to [0,1] for reporting. First, the MLLM receives each chunk vi and its caption pi, and scores visual quality, text alignment, and an overall chunk score ai. Second, the MLLM receives each adjacent pair (vi,vi+1) together with their captions (pi,pi+1), and scores transition smoothness and an overall transition score bi. Third, the MLLM receives the full generated video and scores long-range consistency, global text alignment, and a global overall score g. The final MLLM interaction score averages the overall scores from the chunk, transition, and global judgments:

T−1

T

1 5(T − 1)

g 5

1 5T

, (9)

ai, Strans =

bi, Sglobal =

Schunk =

i=1

i=1

1 3

(Schunk + Strans + Sglobal). (10) The final interaction score uses the calibrated CLIP score as a lightweight semantic auxiliary term:

Smllm =

Sinteract = (1 − λ)Smllm + λ Sclip, λ = 0.1. (11)

This design lets CLIP contribute frame-caption semantic grounding while keeping the interaction metric dominated by the structured MLLM judge, which evaluates temporal properties such as chunk-level instruction following, boundary smoothness, state preservation, and full-video fluency.

Finally, WorldOlympiad reports an overall score by averaging the three core evaluation tracks: Sall =

1 3

(Sphys + S3D + Sinteract). (12)

This equal-weight aggregation keeps the leaderboard aligned with the benchmark design: physical faithfulness, geometric consistency, and interaction fidelity contribute symmetrically to the final model ranking.

### 4 Experiment

- 4.1 Experimental Setup

Evaluationmodels. We evaluate eight publicly available video-generation pipelines through OpenWorldLib [32]. These pipelines cover three major families of video world models. The gaming-centric group includes MatrixGame 2.0 [11] and LingBot-World [35]; the robotics-centric group includes Cosmos-Predict-2.5 [2] and WoW [7]; and the general long-video group includes Rolling Forcing [22], LongLive [45], Yume-1.5 [25], and HunyuanWorldPlay [31]. In our experiments, we test these pipelines across different downstream scenarios, including gaming, robotics, and general real-world videos.

Implementation details. For fairness, we use each released pipeline with its official default generation configuration whenever possible. Since different pipelines may adopt different chunk sizes or segment-level generation settings, we dynamically map the temporal information in the chunk captions to each model’s native generation configuration. This allows the temporal proportions of the original chunk captions to be retained while respecting each generation pipeline’s native training and inference configuration. For methods that include an explicit memory or long-context mechanism, such as Rolling Forcing, we preserve the official memory-management strategy during rollout. For pipelines without a dedicated long-horizon memory module, such as WoW, we perform long-video generation through video continuation, using the previously generated context as the condition for the next segment.

All generated videos are evaluated by the same automatic WorldOlympiad evaluator. The evaluator reports physical faithfulness, 3D consistency, CLIP-augmented interaction fidelity, and an overall composite score. Judge-based component scores are reported after averaging their normalized subscores into the [0,1] range. Physical faithfulness aggregates rule-level judgments over mechanics, thermodynamics, and material behavior;

- 3D consistency combines reconstruction quality, meta-view quality, and camera trajectory consistency; and interaction fidelity measures chunk-level instruction following, CLIP-based semantic grounding, adjacent transition smoothness, and long-range coherence over the full generated video.
- 4.2 Main Benchmark Results

- Table 3 summarizes the video world models evaluated in OpenWorldLib, grouped by gaming, robotics, and general world-model categories. The table reports physical faithfulness, 3D consistency, CLIP-augmented interaction fidelity, and the overall score. Figure 5 further visualizes the score distribution across pipelines and evaluation dimensions.

From visual synthesis to stateful world simulation. The most salient trend in Table 3 is that the best models are no longer distinguished only by visual plausibility, but by their ability to preserve physical state and interaction semantics over extended rollouts. LingBot-World achieves the highest overall score (0.683), with particularly strong physical faithfulness (0.942) and interaction fidelity (0.734). Notably, LingBot-World is a 14B-activated model, suggesting that large-scale capacity can substantially improve long-horizon state preservation, scene continuity, and action-conditioned dynamics. However, model scale is not the only factor that determines world-model quality. Cosmos-Predict-2.5, with only 2B parameters, reaches a comparable overall score of (0.671). Although it is categorized as a robotics-centric pipeline in our evaluation, Cosmos-Predict-2.5 is optimized for physical-world prediction, which helps it generalize beyond embodied manipulation scenarios and achieve strong physical fidelity across diverse downstream settings. This comparison suggests that targeted physical-world training and rollout design can partly compensate for smaller activated model scale, leading to competitive performance in stateful world simulation.

Physicalregularityisemergingasasharedcapability. A second trend is that several recent pipelines already show strong compliance with common physical regularities. LingBot-World (0.942), Cosmos-Predict-2.5 (0.906), Rolling Forcing (0.873), LongLive (0.863), and Yume-1.5 (0.863) all achieve high physical scores, suggesting that current video world models have begun to internalize frequent patterns of motion, contact, support, and material behavior. This progress is consistent with the increasing attention to physical plausibility in recent evaluation suites such as VBench 2.0. However, the capability is still uneven: fine-grained results in the

- Table 3 Main benchmark results on WorldOlympiad. We evaluate eight representative video world models across gaming, robotics, and general long-video generation settings. Physical (Sphys): physical faithfulness; 3D Cons. (S3D): 3D spatial consistency; Interact. (Sinteract): interaction fidelity with CLIP-based semantic grounding; All (Sall): overall composite score. Best and second-best results are marked in bold and underlined, respectively.

Evaluation Metrics

Category Model

Rank Physical 3D Cons. Interact. All

Matrix-Game 2.0 [11] 0.325 0.255 0.113 0.231 8 LingBot-World [35] 0.942 0.373 0.734 0.683 1

Gaming World Model

Cosmos-Predict-2.5 [2] 0.906 0.399 0.707 0.671 2 WoW [7] 0.708 0.250 0.345 0.434 7

Robotics World Model

Rolling Forcing [22] 0.873 0.321 0.636 0.610 3 LongLive [45] 0.863 0.363 0.526 0.584 5 Yume-1.5 [25] 0.863 0.301 0.649 0.604 4 Hunyuan-WorldPlay [31] 0.692 0.424 0.316 0.477 6

General World Model

All is the average of Physical, 3D Cons., and Interact.; overall ranks are computed by the unrounded All score. Displayed scores are rounded to three decimal places.

(a) Overall Ranking (b) Capability Radar (c) Metric Heatmap

Physical

1.0

|0.942|0.373|0.734|0.683|
|---|---|---|---|
|0.906|0.399|0.707|0.671|
|0.873|0.321|0.636|0.610|
|[Figure 114]<br><br>0.863<br><br>0.863|0.301<br><br>0.363|0.649<br><br>0.526|0.604<br><br>0.584|
|0.692|0.424|0.316|0.477|
|0.708|0.250|0.345|0.434|
|0.325|0.255|0.113|0.231|

| | |0.23|1|0 0.43|0<br><br>.477 4|0<br><br>0. 0.610<br><br>0.604<br><br>.584|.683 671|
|---|---|---|---|---|---|---|---|
| | | | | | | | |

[Figure 115]

| |3D<br><br>0.25<br><br>0.50<br><br>0.75<br><br>1.00|
|---|---|
| | |

LingBot-World Cosmos-Predict-2.5 Rolling Forcing Yume-1.5

0.8

0.6

Score

All

Cons.

LongLive Hunyuan-WorldPlay

0.4

WoW Matrix-Game 2.0

0.2

0.0

Physical 3D Cons. Interact. All

0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8

Interact.

All score

LingBot-World Cosmos-Predict-2.5 Rolling Forcing Yume-1.5 LongLive Hunyuan-WorldPlay WoW Matrix-Game 2.0

- Figure 5 Result statistics of WorldOlympiad across evaluated world-model pipelines and scoring dimensions.

appendix show that thermodynamics and material-level questions remain more fragile than many mechanics questions, and weaker models still violate basic constraints under long-horizon generation.

The geometry-simulation gap remains unresolved. Geometric consistency remains one of the most important unresolved weaknesses across current video world models. Even the strongest pipeline on this dimension, Hunyuan-WorldPlay, reaches only (0.424), while most models remain in the (0.25)–(0.40) range. Notably, models represented by Hunyuan-WorldPlay rely more heavily on camera or viewpoint control as their primary form of interaction. This design encourages the model to preserve spatial layout under view changes, which helps explain its relatively stronger 3D consistency. However, such interaction is also more constrained than open-ended action-conditioned generation: controlling the camera or viewpoint does not necessarily require the model to reason about complex object manipulation, agent behavior, or multi-step state transitions. As a result, these models can obtain better geometry scores while still achieving limited overall performance. This highlights a key trade-off in current world models: view-control pipelines may better preserve cross-view structure, but robust world simulation requires both stable 3D geometry and flexible interactive dynamics.

The specialization-generalization trade-off. LingBot-World and Cosmos-Predict-2.5 have both undergone sustained training in specific domains such as gaming and robotics. Their strong performance in our benchmark suggests that continuous domain-specific training can effectively generalize to broader evaluation settings.

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

Physical

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

Violating the laws of gravity Move without contact

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

Geometry

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

The soccer goal that suddenly deformed The stone pillar that appeared suddenly

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

Interaction

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

Slow motion tends towards stillness Chaotic drifting camera

- Figure6 Representative WorldOlympiad case studies detected by the benchmark. The upper examples show high-quality generations that preserve the intended physical behavior, scene structure, or interaction state, while the lower examples show typical failure cases with visible rule violations, geometric inconsistency, or interaction drift.

In particular, the fact that these two specialized pipelines rank at the top indicates that targeted training does not necessarily limit a model to its original domain; instead, it can provide transferable world knowledge that benefits performance across different scenarios.However, not all specialized models show the same generalization ability. WoW performs better in embodied scenarios than in other domains, but its scores drop on gaming and general real-world videos. As shown in Table 6, WoW reaches 0.502 on embodied videos, but only 0.368 on gaming videos and 0.415 on general videos. These results suggest that specialization is useful only when the learned knowledge can transfer beyond a narrow domain. Future models should therefore combine sustained domain-specific training with broader cross-domain world knowledge.

Fine-grained diagnostics. WorldOlympiad is designed to be diagnostic rather than only leaderboard-driven. Beyond the aggregate scores in Table 3, we decompose model behavior into domain-level results, physical dimensions and questions, 3D reconstruction submetrics, and interaction submetrics. These breakdowns make it possible to identify whether a low score is caused by a specific physical rule, unstable geometry, poor semantic grounding, or long-range interaction drift. Detailed tables for these fine-grained results are provided in Appendix B.

- 4.3 Qualitative Case Studies and Failure Modes

Quantitative scores are paired with qualitative cases because a leaderboard alone cannot explain model failures. As shown in Figure 6, WorldOlympiad reveals three recurring failure modes. Physical metrics identify implausible dynamics, such as objects moving against gravity, deforming without contact, or changing state abruptly. Geometry metrics expose videos that look reasonable in the original view but fail under 3D reconstruction, meta-view rendering, or camera-trajectory comparison. Interaction metrics capture rollouts that follow isolated captions but reset state, lose objects, or break action continuity across chunks. Additional qualitative examples and discussion are provided in Appendix C.

- 4.4 Human Preference Alignment

To examine whether the WorldOlympiad automatic evaluator is consistent with human preference, we conduct a controlled alignment study on the evaluated world models. Since long-video world modeling requires more than visual realism alone, human annotators compare generated videos from multiple complementary aspects, including overall perceived quality, physical plausibility, temporal coherence, and interaction fidelity. These criteria are designed to reflect the key capabilities targeted by WorldOlympiad and provide a human-centered reference for evaluating model behavior in downstream scenarios.

We aggregate the annotations into a pairwise human preference score Shuman, where higher values indicate stronger human preference. Table 4 compares the resulting human ranking with the WorldOlympiad automatic ranking over the eight annotated models. The two rankings are highly consistent, with a Spearman correlation coefficient of ρ = 0.95. This strong agreement suggests that WorldOlympiad’s automatic evaluation captures model-level quality differences that are also perceived by human annotators. Meanwhile, unlike human evaluation, the automatic evaluator can be applied at a larger scale and provides more fine-grained diagnostic scores across physical, geometric, and interaction-related dimensions. These results indicate that WorldOlympiad offers a scalable yet human-aligned evaluation protocol for long-video world models. Additional annotation and aggregation details are provided in Appendix D.

- Table 4 Alignment between human preference rankings and WorldOlympiad automatic rankings. Shuman denotes the pairwise human preference score, and Sauto denotes the WorldOlympiad All score. Rank gap is computed as human rank minus automatic rank.

Category Model Shuman Sauto Human Rank Auto Rank Rank Gap

Gaming World Model LingBot-World 0.721 0.683 1 1 0 Robotics World Model Cosmos-Predict-2.5 0.648 0.671 2 2 0 General World Model Rolling Forcing 0.579 0.610 3 3 0 General World Model LongLive 0.532 0.584 4 5 -1 General World Model Yume-1.5 0.491 0.604 5 4 1 General World Model Hunyuan-WorldPlay 0.423 0.477 6 6 0 Gaming World Model Matrix-Game 2.0 0.309 0.231 7 8 -1 Robotics World Model WoW 0.271 0.434 8 7 1

- 5 Conclusion

We presented WorldOlympiad, a benchmark for evaluating video world models beyond surface-level visual quality, measuring three core capabilities: physical faithfulness, geometric consistency, and interaction fidelity. WorldOlympiad combines rule-based physical judging, 3D reconstruction-based geometry diagnostics, and chunk-level plus long-range interaction evaluation, providing a unified protocol for diagnosing whether generated videos behave as reliable world simulations. Experiments across gaming-centric, robotics-centric, and general world-model pipelines reveal that current models remain far from reliable world simulators: even strong models fail on physical rules, 3D structure, or long-horizon state preservation, exposing important gaps between perceptually plausible generation and controllable world modeling.

Future work. Future work will extend WorldOlympiad to study how different memory mechanisms affect long-horizon consistency and interactive controllability. Although many recent pipelines introduce memory modules to improve long-video generation, their varying model scales, training data, and architectural designs make it difficult to isolate whether performance gains stem from the memory mechanism itself or from confounding factors. We therefore aim to build a controlled evaluation environment that disentangles memory design from other variables. Relevant designs include KV-cache reuse, explicit 3D scene memory, linear attention, and hybrid temporal-spatial mechanisms. By comparing these under shared data, comparable model capacity, and a unified protocol, future analysis can more clearly reveal which memory forms best support physical consistency, geometric stability, and reliable long-horizon interaction.

References

- [1] Niket Agarwal, Arslan Ali, Maciej Bala, Yogesh Balaji, Erik Barker, Tiffany Cai, Prithvijit Chattopadhyay, Yongxin Chen, Yin Cui, Yifan Ding, et al. Cosmos world foundation model platform for physical ai. arXiv preprint arXiv:2501.03575, 2025.
- [2] Arslan Ali, Junjie Bai, Maciej Bala, Yogesh Balaji, Aaron Blakeman, Tiffany Cai, Jiaxin Cao, Tianshi Cao, Elizabeth Cha, Yu-Wei Chao, et al. World simulation with video foundation models for physical ai. arXiv preprint arXiv:2511.00062, 2025.
- [3] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023.
- [4] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Leo Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, et al. Video generation models as world simulators. OpenAI Blog, 1(8):1, 2024.
- [5] Nicolas Carion, Laura Gustafson, Yuan-Ting Hu, Shoubhik Debnath, Ronghang Hu, Didac Suris, Chaitanya Ryali, Kalyan Vasudev Alwala, Haitham Khedr, Andrew Huang, et al. Sam 3: Segment anything with concepts. arXiv preprint arXiv:2511.16719, 2025.
- [6] Haoxuan Che, Xuanhua He, Quande Liu, Cheng Jin, and Hao Chen. Gamegen-x: Interactive open-world game video generation. In International Conference on Learning Representations, volume 2025, pages 37546–37593, 2025.
- [7] Xiaowei Chi, Peidong Jia, Chun-Kai Fan, Xiaozhu Ju, Weishi Mi, Kevin Zhang, Zhiyuan Qin, Wanxin Tian, Kuangzhi Ge, Hao Li, et al. Wow: Towards a world omniscient world model through embodied interaction. arXiv preprint arXiv:2509.22642, 2025.
- [8] Google DeepMind. Gemini 3 pro model card, 2025.
- [9] Yufan Deng, Zilin Pan, Hongyu Zhang, Xiaojie Li, Ruoqing Hu, Yufei Ding, Yiming Zou, Yan Zeng, and Daquan Zhou. Rethinking video generation model for the embodied world. arXiv preprint arXiv:2601.15282, 2026.
- [10] Haoyi Duan, Hong-Xing Yu, Sirui Chen, Li Fei-Fei, and Jiajun Wu. Worldscore: A unified evaluation benchmark for world generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 27713–27724, 2025.
- [11] Xianglong He, Chunli Peng, Zexiang Liu, Boyang Wang, Yifan Zhang, Qi Cui, Fei Kang, Biao Jiang, Mengyin An, Yangyang Ren, et al. Matrix-game 2.0: An open-source real-time and streaming interactive world model. arXiv preprint arXiv:2508.13009, 2025.
- [12] Yue Hu, Siyuan Huang, Yue Liao, Shengcong Chen, Pengfei Zhou, Liliang Chen, Maoqing Yao, and Guanghui Ren. Ewmbench: Evaluating scene, motion, and semantic quality in embodied world models. arXiv preprint arXiv:2505.09694, 2025.
- [13] Junchao Huang, Xinting Hu, Boyao Han, Shaoshuai Shi, Zhuotao Tian, Tianyu He, and Li Jiang. Memory forcing: Spatio-temporal memory for consistent scene generation on minecraft. arXiv preprint arXiv:2510.03198, 2025.
- [14] Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, and Eli Shechtman. Self forcing: Bridging the train-test gap in autoregressive video diffusion. arXiv preprint arXiv:2506.08009, 2025.
- [15] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21807–21818, 2024.

- [16] Ziqi Huang, Fan Zhang, Xiaojie Xu, Yinan He, Jiashuo Yu, Ziyue Dong, Qianli Ma, Nattapol Chanpaisit, Chenyang Si, Yuming Jiang, et al. Vbench++: Comprehensive and versatile benchmark suite for video generative models. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025.
- [17] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024.
- [18] Dacheng Li, Yunhao Fang, Yukang Chen, Shuo Yang, Shiyi Cao, Justin Wong, Michael Luo, Xiaolong Wang, Hongxu Yin, Joseph E Gonzalez, et al. Worldmodelbench: Judging video generation models as world models. arXiv preprint arXiv:2502.20694, 2025.
- [19] Runjia Li, Philip Torr, Andrea Vedaldi, and Tomas Jakab. Vmem: Consistent interactive video scene generation with surfel-indexed view memory. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 25690–25699, 2025.
- [20] Yaxuan Li, Yichen Zhu, Junjie Wen, Chaomin Shen, and Yi Xu. Worldeval: World model as real-world robot policies evaluator, 2025.
- [21] Haotong Lin, Sili Chen, Junhao Liew, Donny Y Chen, Zhenyu Li, Guang Shi, Jiashi Feng, and Bingyi Kang. Depth anything 3: Recovering the visual space from any views. arXiv preprint arXiv:2511.10647, 2025.
- [22] Kunhao Liu, Wenbo Hu, Jiale Xu, Ying Shan, and Shijian Lu. Rolling forcing: Autoregressive long video diffusion in real time. arXiv preprint arXiv:2509.25161, 2025.
- [23] Mingxin Liu, Shuran Ma, Shibei Meng, Xiangyu Zhao, Zicheng Zhang, Shaofeng Zhang, Zhihang Zhong, Peixian Chen, Haoyu Cao, Xing Sun, et al. Rise-video: Can video generators decode implicit world rules? arXiv preprint arXiv:2602.05986, 2026.
- [24] Wei Liu, Ziyu Chen, Zizhang Li, Yue Wang, Hong-Xing Yu, and Jiajun Wu. Realwonder: Real-time physical action-conditioned video generation. arXiv preprint arXiv:2603.05449, 2026.
- [25] Xiaofeng Mao, Zhen Li, Chuanhao Li, Xiaojie Xu, Kaining Ying, Tong He, Jiangmiao Pang, Yu Qiao, and Kaipeng Zhang. Yume-1.5: A text-controlled interactive world generation model. arXiv preprint arXiv:2512.22096, 2025.
- [26] Xiaofeng Mao, Shaoheng Lin, Zhen Li, Chuanhao Li, Wenshuo Peng, Tong He, Jiangmiao Pang, Mingmin Chi, Yu Qiao, and Kaipeng Zhang. Yume: An interactive world generation model. arXiv preprint arXiv:2507.17744, 2025.
- [27] Yiran Qin, Zhelun Shi, Jiwen Yu, Xijun Wang, Enshen Zhou, Lijun Li, Zhenfei Yin, Xihui Liu, Lu Sheng, Jing Shao, et al. Worldsimbench: Towards video generation models as world simulators. arXiv preprint arXiv:2410.18072, 2024.
- [28] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021.
- [29] Yu Shang, Zhuohang Li, Yiding Ma, Weikang Su, Xin Jin, Ziyou Wang, Lei Jin, Xin Zhang, Yinzhou Tang, Haisheng Su, et al. Worldarena: A unified benchmark for evaluating perception and functional utility of embodied world models. arXiv preprint arXiv:2602.08971, 2026.
- [30] Joonghyuk Shin, Zhengqi Li, Richard Zhang, Jun-Yan Zhu, Jaesik Park, Eli Shechtman, and Xun Huang. Motionstream: Real-time video generation with interactive motion controls. arXiv preprint arXiv:2511.01266, 2025.
- [31] Wenqiang Sun, Haiyu Zhang, Haoyuan Wang, Junta Wu, Zehan Wang, Zhenwei Wang, Yunhong Wang, Jun Zhang, Tengfei Wang, and Chunchao Guo. Worldplay: Towards long-term geometric consistency for real-time interactive world modeling. arXiv preprint arXiv:2512.14614, 2025.

- [32] DataFlow Team, Bohan Zeng, Daili Hua, Kaixin Zhu, Yifan Dai, Bozhou Li, Yuran Wang, Chengzhuo Tong, Yifan Yang, Mingkun Chang, et al. Openworldlib: A unified codebase and definition of advanced world models. arXiv preprint arXiv:2604.04707, 2026.
- [33] InSpatio Team, Donghui Shen, Guofeng Zhang, Haomin Liu, Haoyu Ji, Hujun Bao, Hongjia Zhai, Jialin Liu, Jing Guo, Nan Wang, et al. Inspatio-world: A real-time 4d world simulator via spatiotemporal autoregressive modeling. arXiv preprint arXiv:2604.07209, 2026.
- [34] Meituan LongCat Team, Xunliang Cai, Qilong Huang, Zhuoliang Kang, Hongyu Li, Shijun Liang, Liya Ma, Siyu Ren, Xiaoming Wei, Rixu Xie, et al. Longcat-video technical report. arXiv preprint arXiv:2510.22200, 2025.
- [35] Robbyant Team, Zelin Gao, Qiuyu Wang, Yanhong Zeng, Jiapeng Zhu, Ka Leong Cheng, Yixuan Li, Hanlin Wang, Yinghao Xu, Shuailei Ma, et al. Advancing open-source world models. arXiv preprint arXiv:2601.20540, 2026.
- [36] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.
- [37] Weijie Wang, Xiaoxuan He, Youping Gu, Yifan Yang, Zeyu Zhang, Yefei He, Yanbo Ding, Xirui Hu, Donny Y Chen, Zhiyuan He, et al. World-r1: Reinforcing 3d constraints for text-to-video generation. arXiv preprint arXiv:2604.24764, 2026.
- [38] Weijie Wang, Haoyu Zhao, Yifan Yang, Feng Chen, Zeyu Zhang, Yefei He, Zicheng Duan, Donny Y. Chen, Yuqing Yang, and Bohan Zhuang. Latent spatial memory for video world models. arXiv preprint arXiv:2606.09828, 2026.
- [39] Thaddäus Wiedemer, Yuxuan Li, Paul Vicol, Shixiang Shane Gu, Nick Matarese, Kevin Swersky, Been Kim, Priyank Jaini, and Robert Geirhos. Video models are zero-shot learners and reasoners. arXiv preprint arXiv:2509.20328, 2025.
- [40] Ruiqi Wu, Xuanhua He, Meng Cheng, Tianyu Yang, Yong Zhang, Zhuoliang Kang, Xunliang Cai, Xiaoming Wei, Chunle Guo, Chongyi Li, et al. Infinite-world: Scaling interactive world models to 1000-frame horizons via pose-free hierarchical memory. arXiv preprint arXiv:2602.02393, 2026.
- [41] Shihan Wu, Xuecheng Liu, Shaoxuan Xie, Pengwei Wang, Xinghang Li, Bowen Yang, Zhe Li, Kai Zhu, Hongyu Wu, Yiheng Liu, et al. Robocoin: An open-sourced bimanual robotic data collection for integrated manipulation. arXiv preprint arXiv:2511.17441, 2025.
- [42] Tong Wu, Shuai Yang, Ryan Po, Yinghao Xu, Ziwei Liu, Dahua Lin, and Gordon Wetzstein. Video world models with long-term spatial memory. arXiv preprint arXiv:2506.05284, 2025.
- [43] Zeqi Xiao, Yushi Lan, Yifan Zhou, Wenqi Ouyang, Shuai Yang, Yanhong Zeng, and Xingang Pan. Worldmem: Long-term consistent world simulation with memory. arXiv preprint arXiv:2504.12369, 2025.
- [44] Tianwei Xiong, Yuqing Wang, Daquan Zhou, Zhijie Lin, Jiashi Feng, and Xihui Liu. Lvd-2m: A long-take video dataset with temporally dense captions. Advances in Neural Information Processing Systems, 37:16623–16644, 2024.
- [45] Shuai Yang, Wei Huang, Ruihang Chu, Yicheng Xiao, Yuyang Zhao, Xianbang Wang, Muyang Li, Enze Xie, Yingcong Chen, Yao Lu, et al. Longlive: Real-time interactive long video generation. arXiv preprint arXiv:2509.22622, 2025.
- [46] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024.
- [47] Yixuan Ye, Xuanyu Lu, Yuxin Jiang, Yuchao Gu, Rui Zhao, Qiwei Liang, Jiachun Pan, Fengda Zhang, Weijia Wu, and Alex Jinpeng Wang. Mind: Benchmarking memory consistency and action control in world models. arXiv preprint arXiv:2602.08025, 2026.

- [48] Tianwei Yin, Qiang Zhang, Richard Zhang, William T Freeman, Fredo Durand, Eli Shechtman, and Xun Huang. From slow bidirectional to fast autoregressive video diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22963–22974, 2025.
- [49] Kaining Ying, Hengrui Hu, Siyu Ren, Jiamu Li, Fengjiao Chen, Ziwen Wang, Xuezhi Cao, Xunliang Cai, and Henghui Ding. Wbench: A comprehensive multi-turn benchmark for interactive video world model evaluation. arXiv preprint arXiv:2605.25874, 2026.
- [50] Wei Yu, Runjia Qian, Yumeng Li, Liquan Wang, Songheng Yin, Dennis Anthony, Yang Ye, Yidi Li, Weiwei Wan, Animesh Garg, et al. Mosaicmem: Hybrid spatial memory for controllable video world models. arXiv preprint arXiv:2603.17117, 2026.
- [51] Yifan Zhang, Chunli Peng, Boyang Wang, Puyi Wang, Qingcheng Zhu, Fei Kang, Biao Jiang, Zedong Gao, Eric Li, Yang Liu, et al. Matrix-game: Interactive world foundation model. arXiv preprint arXiv:2506.18701, 2025.
- [52] Zeyu Zhang, Shuning Chang, Yuanyu He, Yizeng Han, Jiasheng Tang, Fan Wang, and Bohan Zhuang. Blockvid: Block diffusion for high-quality and consistent minute-long video generation. arXiv preprint arXiv:2511.22973, 2025.
- [53] Jinjing Zhao, Fangyun Wei, Zhening Liu, Hongyang Zhang, Chang Xu, and Yan Lu. Spatia: Video generation with updatable spatial memory. arXiv preprint arXiv:2512.15716, 2025.
- [54] Dian Zheng, Ziqi Huang, Hongbo Liu, Kai Zou, Yinan He, Fan Zhang, Lulu Gu, Yuanhan Zhang, Jingwen He, Wei-Shi Zheng, et al. Vbench-2.0: Advancing video generation benchmark suite for intrinsic faithfulness. arXiv preprint arXiv:2503.21755, 2025.
- [55] Hongzhou Zhu, Min Zhao, Guande He, Hang Su, Chongxuan Li, and Jun Zhu. Causal forcing: Autoregressive diffusion distillation done right for high-quality real-time interactive video generation. arXiv preprint arXiv:2602.02214, 2026.

### A WorldOlympiad Judge Prompt Templates

The prompt templates below cover dynamic-object extraction, physical consistency, interaction quality, and 3D reconstruction quality.

- Table 5 Judge-related prompt families used by WorldOlympiad. Component Prompt family Role in evaluation

Physical Relevance and compliance judges

Select applicable physical rules and judge whether the generated video follows them against the reference.

Interaction Chunk, transition, and global judges

Score local caption following, boundary smoothness, and long-range consistency.

3D Static-scene rewrite and 3D MLLM scorers

Remove dynamic actors from the judging target and score Gaussiansplat reconstruction quality.

Preprocessing Dynamic-object extraction

Select moving or deforming foreground actors for SAM-based masking and diagnostic videos.

- A.1 Dynamic-Object Extraction Prompt

Before physical and 3D scoring, WorldOlympiad uses a MLLM prompt to identify the primary dynamic or deforming objects for SAM-based visualization, masking, and background completion.

Dynamic-object extraction

System role: The model acts as an expert at spotting only the primary physical actors that visibly move or deform in a video.

Selection rules: Return the fewest distinct moving or deforming objects, with a maximum of three. Prefer dominant moving foreground objects and omit secondary or uncertain objects. Do not include static background, scenery, floors, tables, walls, tools, supports, containers, or objects that are merely visible. If an object does not visibly change position, orientation, or shape, do not return it. Merge duplicates and synonyms into one concise noun of one to three words.

User query: Watch the video and list only the main objects that visibly move or deform. The query includes

{video_prompt} as the video description. Output format: Return only a JSON array with one to three concise nouns, such as ["person", "ball"]. No explanations are allowed.

- A.2 Physical Judge Prompts

The physical pipeline first runs a relevance judge on the reference video to determine which physical rules are applicable. It then runs a compliance judge that compares the generated candidate against the reference.

Physical batch relevance judge

System role: PhysicsFilterBatch, an expert video physics relevance evaluator. Input: One reference or ground-truth video, its textual prompt, and a list of physics questions. Each question contains a question_id, dimension, rule text, and success condition.

Decision rule: For every question, decide whether the reference video contains enough visual evidence to judge the physical rule. Use related=true when the rule can be judged from visible objects, materials, contacts, state changes, motion, support, heat or cold cues, liquid/gas/solid behavior, deformation, color or material behavior, burning, dissolving, or other relevant physical evidence. Use related=false only when the rule truly cannot be evaluated from the shown scene.

User query: The prompt provides {video_prompt} and {question_list_json}, and asks the judge to evaluate relevance for every question_id. Output format: Return strict JSON with one result per input question: question_id, related, confidence, and a short evidence-based reason.

##### Physical batch compliance judge

System role: PhysicsJudgeBatch, an expert video physics evaluator. Input: Two videos are provided: the generated candidate video to judge and the ground-truth reference video, which supplies context for the intended event, scene, timing, and physical evidence. The prompt also provides {video_prompt} and the related subset of {question_list_json}.

Decision rule: For each related physics question, judge whether the generated candidate follows the physical rule and remains consistent with the reference. The judge does not require frame-exact matching, but it requires plausible physics, object and material identity consistency, correct temporal order, and visible evidence.

Output format: Return strict JSON with one result per input question: question_id, compliant, confidence, a 3–5 sentence explanation with specific visual evidence, and short concrete observations.

Physical question context. The question_list_json variable contains rule identifiers, dimensions, questions, and success conditions. These rules cover mechanics (gravity, buoyancy, compression, impact), thermodynamics (melting, sublimation, vaporization, condensation, deposition, freezing), and material behavior (color_mixing, solubility, hardness, combustibility).

Mechanics rule contexts

gravity rule: Do free-moving objects move downward consistently with gravity? expected_behavior: First judge whether objects are unsupported, airborne, falling, jumping, driving over uneven terrain, or staying grounded on a support surface. Unsupported objects should fall or arc downward under gravity; supported ground vehicles, people, and objects should remain in plausible contact with the ground unless a visible jump, ramp, collision, or lift explains vertical motion. Penalize floating, sinking through the ground, sudden vertical pops, or hovering without support. buoyancy rule: Do objects on or in a fluid behave consistently with buoyancy, such that floating items stay near the surface and sinking items submerge? expected_behavior: Floating objects should remain on or near the surface; dense objects should descend. compression rule: When objects or support surfaces are stressed, loaded, squeezed, or pressed, do they deform or remain rigid in a plausible manner?

expected_behavior: Cans may dent when crushed, soft materials should compress smoothly under load, and rigid vehicles or metal bodies should mostly keep their shape unless there is collision or heavy force. In robot manipulation, vehicle, racing, or navigation scenes, grasping, pinching, pressing, tire-ground contact, suspension loading, soil displacement, body rigidity, or lack of impossible warping can make stress and deformation relevant. Deformation should start only after visible squeezing, support, load, contact, or other applied stress, and the same object, person, or vehicle should stay visually consistent instead of morphing into a different shape or identity.

impact rule: Do contact, traction, collisions, impacts, and momentum changes produce reasonable motion transitions? expected_behavior: Look for momentum transfer, bouncing, shattering, resting poses, ground contact, tire or foot contact, traction, braking, turning, or acceleration that matches visible forces and contacts. In robot manipulation, placing or releasing an object into contact also counts as an impact or contact event. In vehicle, racing, sports, or navigation scenes, tire-ground contact, acceleration from rest, sliding, dust kick-up, braking, steering, direction changes, speed changes, near-collisions, or collisions are relevant. Abrupt deformation, direction change, or speed change should happen after visible contact, impact, or control input rather than before.

##### Thermodynamics rule contexts

melting description: A solid substance should gradually transition to liquid when heated above its melting point. expected_behavior: The solid should decrease in size or volume as it transforms into liquid, and the process should be gradual and continuous. violations: Solid increasing in size, liquid turning into solid, instantaneous disappearance without liquid formation, solid remaining unchanged despite heat, or no liquid formation. sublimation description: A solid should directly transition to gas without passing through the liquid phase.

expected_behavior: The solid should rapidly decrease or disappear while gas or vapor appears around it, with no visible liquid phase. violations: Solid melting to liquid first, gas condensing to solid, solid remaining stable, or liquid forming as an intermediate phase. vaporization description: A liquid should transition to gas when heated or over time. expected_behavior: The liquid should decrease in volume or disappear over time; bubbles, boiling, evaporation, or visible gas or vapor may appear. violations: Liquid increasing in volume, gas condensing to liquid, liquid remaining constant despite heat or time, or liquid freezing. condensation description: A gas should transition to liquid when cooled below its condensation point. expected_behavior: Liquid should form and increase in volume; small droplets may appear and merge, and gas may become less visible as it condenses. violations: Liquid evaporating, gas remaining as gas, liquid freezing directly to solid, or no liquid formation. deposition description: A gas should directly transition to solid without passing through the liquid phase. expected_behavior: Solid should form and increase in size or volume directly from gas, often as crystals or frost-like structures, with no visible liquid intermediate. violations: Gas condensing to liquid first, solid melting, liquid freezing to solid, or no solid formation. freezing description: A liquid should transition to solid when cooled below its freezing point. expected_behavior: The liquid should solidify and may expand slightly or contract; the surface may become rigid and the shape should become more defined and regular. violations: Liquid remaining liquid, solid melting, liquid evaporating, or no solidification occurring.

##### Material rule contexts

color_mixing question: When different colored liquids or paints mix, do they produce the correct resulting color? success_condition: Colored liquids, paints, powders, smoke, dye, pigments, or other visibly colored substances should blend into plausible resulting colors when they contact and mix. Red and yellow should produce orange, blue and yellow should produce green, and red and blue should produce purple. If colored objects merely pass near each other without material transfer or blending, they should not change color. solubility question: Do soluble materials such as sugar or salt dissolve properly when placed in water or other solvents? success_condition: Soluble or dispersible substances such as sugar, salt, powder, dye, tablets, or granular material should gradually disperse, dissolve, fade, or become suspended or invisible in a liquid solvent. Insoluble solids should remain visible or settle. hardness question: Do materials with different hardness levels behave correctly when grasped, pressed, cut, folded, or broken? success_condition: Soft materials such as paper, cloth, foam, food, plants, soil, loose dirt, or powder should bend, fold, tear, compress, scatter, or deform when appropriate. Hard materials such as metal, stone, glass, rigid vehicle bodies, tools, or containers should resist deformation unless force or collision is strong enough. In robot, human, vehicle, racing, sports, or navigation scenes, grasping, pinching, pressing, stepping, tire-ground contact, placing, sliding, collision, or load-bearing can reveal rigidity versus softness. Shape change should follow visible contact or applied force, and the acted-on object, person, vehicle, or material should remain visually consistent. combustibility question: Do flammable materials burn correctly, producing fire, smoke, or char? success_condition: Wood, paper, and fabric should ignite and produce flames or smoke; non-flammable materials should not.

- A.3 Interaction Judge Prompts

The interaction pipeline evaluates chunk-generated long videos at three levels: individual chunks, adjacent chunk transitions, and the stitched full video.

##### Interaction judge shared instruction

System role: A strict evaluator for chunk-generated interactive videos. Judgment scope: Judge whether each generated chunk follows its intended action and caption, whether adjacent chunks transition smoothly, and whether the full stitched video remains globally consistent. The reference video is used only as context for scene, style, camera, and intended interaction, not as a requirement for frame-exact matching. Output rule: Always return strict JSON only. All scores must be numbers from 0 to 5.

##### Chunk-level interaction judge

Inputmetadata: {chunk_index}, {source_interval}, generated interval [{generated_start_sec}, {generated_end_sec}), {action}, and {caption}. Primary evidence: Use the generated chunk frames as the primary evidence. Reference-video frames are used only for context about the intended scene and style.

Scoring criteria: Score visual_quality by clarity, realism, temporal stability, color and lighting consistency, and lack of artifacts. Score text_alignment by whether the visible content matches the intended action and caption. Output format: Return strict JSON with chunk_index, visual_quality, text_alignment, overall, and a short evidence-based reason.

##### Transition-level interaction judge

Input metadata: The previous and next chunk indices, generated intervals, actions, and captions. Decision rule: Judge whether the transition is smooth and continuous. Scene, lighting, style, and subject identity should remain coherent. Motion trajectory and camera movement should evolve naturally. Penalize abrupt jumps, object identity resets, impossible camera jumps, and visible stitching artifacts. Output format: Return strict JSON with from_chunk_index, to_chunk_index, transition_smoothness, overall, and a short evidence-based reason.

##### Global interaction judge

Input: The whole stitched generated video and {prompt_summary_json}, which lists each chunk index, action, and caption. Primary evidence: Use generated full-video frames as the primary evidence and reference frames only as context. Scoring criteria: Judge whether subject, character, and object identity remain stable across the video; whether scene style, visual tone, lighting, and camera behavior remain coherent; and whether global semantics align with the combined intent of all chunk prompts. Output format: Return strict JSON with long_range_consistency, global_text_alignment, overall, and a short evidence-based reason.

- A.4 3D Judge Prompts

The 3D pipeline first rewrites the original generation prompt into a static scene prompt, because dynamic foreground actors are masked and video-inpainted before Depth Anything 3 reconstruction and Gaussian-Splat rendering. The MLLM then scores the Gaussian-Splat video and a meta-view image. The camera trajectory score Straj is computed from DA3 camera motion similarity.

Static-scene prompt rewrite

System role: Rewrite video generation prompts for static 3D reconstruction evaluation. The reconstruction input has dynamic foreground actors masked and video-inpainted before DA3 reconstruction.

Rewrite requirements: Keep static background, environment, layout, materials, lighting, weather, terrain, large structures, and camera/view behavior if present. Remove or explicitly ignore dynamic foreground actors and actions, including people, animals, vehicles, limbs, clothing motion, object manipulation, running, turning, and other moving subjects. If the original prompt has no camera information, set camera_behavior exactly to "camera is static". Mention that dynamic foreground actors may be absent because they were masked and video-inpainted before reconstruction.

Input placeholder: {prompt}.

Output format: Return strict JSON with exactly static_scene_description, camera_behavior, and ignore_for_3d.

##### Gaussian-Splat video 3D judge

System role: A calibrated 3D reconstruction judge that follows the scoring rubric and returns strict JSON only. Input: Several sampled frames from a Gaussian-Splat render reconstructed from a source video after dynamic foreground regions were masked and video-inpainted, together with the static-scene prompt {prompt}.

Robustness instruction: Be robust to normal Gaussian-Splat artifacts, video-inpainting artifacts, blur, rain/fog/lowlight appearance, and moderate texture noise. Dynamic foreground actors and actions may be absent or incomplete because they were intentionally removed before reconstruction; do not penalize their absence.

Scoring focus: Judge whether the reconstructed static background geometry is coherent across views or time, whether static structures and scene layout are spatially plausible, whether the render preserves recognizable camera-consistent organization, whether it is faithful to the static-scene description and expected camera behavior, and whether artifacts dominate the render. If the prompt says the camera is static, do not require camera motion or parallax.

Calibration: Scores in 0.8–1.0 indicate a clear coherent 3D scene; 0.6–0.8 indicate a recognizable mostly coherent scene with noticeable artifacts; 0.4–0.6 indicate partial recognition with significant artifacts; 0.2–0.4 indicate mostly failed reconstruction; and 0.0–0.2 indicate an unusable render. Do not assign a very low score solely because the render is blurry or noisy; if the static layout is recognizable, the score should usually be at least 0.5. Output format: Return strict JSON with score in [0, 1] and a short reason.

##### Meta-view image 3D judge

System role: A calibrated 3D reconstruction judge that follows the scoring rubric and returns strict JSON only. Input: A single meta-view image rendered from a 3D reconstruction of a source video after dynamic foreground regions were masked and video-inpainted, together with the static-scene prompt {prompt}.

Robustness instruction: Be robust to normal Gaussian-Splat artifacts, video-inpainting artifacts, blur, rain/fog/lowlight appearance, and moderate texture noise. Dynamic foreground actors and actions may be absent or incomplete because they were intentionally removed before reconstruction; do not penalize their absence.

Scoring focus: Judge whether the static background layout is recognizable, geometrically plausible, structurally coherent, and not catastrophically flat, floating, duplicated, or collapsed. If the prompt says the camera is static, do not require camera motion or parallax.

Calibration: Scores in 0.8–1.0 indicate a clear coherent static scene; 0.6–0.8 indicate a recognizable mostly coherent scene with noticeable artifacts; 0.4–0.6 indicate a partially recognizable scene; 0.2–0.4 indicate a mostly failed reconstruction; and 0.0–0.2 indicate an unusable image. Do not assign a very low score solely because the meta-view is blurry or noisy; if the static scene layout is recognizable, the score should usually be at least 0.5.

Output format: Return strict JSON with score in [0, 1] and a short reason.

### B Detailed Results

This section reports domain-wise scores, physical pass rates, interaction diagnostics, geometry diagnostics, and model-level submetrics.

- B.1 Domain-wise Results

- Table 6 reports the detailed scores on the same-scene subset, grouped by evaluation domain. The table includes physical faithfulness, 3D consistency, CLIP-augmented interaction fidelity, raw and calibrated CLIP semantic alignment, and the overall score. The overall score is computed as the equal-weight average of physical faithfulness, 3D consistency, and interaction fidelity.

###### Table 6 Detailed WorldOlympiad scores on the same-scene subset across gaming, robotics, and general domains. All is the equal-weight average of Physical, 3D Cons., and Interact.

Domain Pipeline Physical 3D Cons. Interact. CLIP Raw CLIP Aux. All

Matrix-Game 2.0 0.332 0.189 0.111 0.230 0.150 0.211 LingBot-World 0.884 0.366 0.778 0.315 0.575 0.676 Cosmos-Predict-2.5 0.867 0.361 0.679 0.306 0.530 0.636 WoW 0.633 0.223 0.249 0.247 0.235 0.368 Rolling Forcing 0.853 0.289 0.675 0.332 0.660 0.606 LongLive 0.851 0.292 0.554 0.322 0.610 0.566 Yume-1.5 0.813 0.352 0.659 0.291 0.455 0.608 Hunyuan-WorldPlay 0.852 0.348 0.471 0.296 0.480 0.557

Gaming

Matrix-Game 2.0 0.364 0.338 0.139 0.252 0.260 0.280 LingBot-World 0.949 0.393 0.710 0.314 0.570 0.684 Cosmos-Predict-2.5 0.937 0.479 0.721 0.321 0.605 0.712 WoW 0.787 0.272 0.447 0.288 0.440 0.502 Rolling Forcing 0.870 0.389 0.566 0.329 0.645 0.608 LongLive 0.857 0.472 0.470 0.327 0.635 0.600 Yume-1.5 0.851 0.288 0.624 0.312 0.560 0.588 Hunyuan-WorldPlay 0.630 0.600 0.262 0.309 0.545 0.497

Robotics

Matrix-Game 2.0 0.216 0.220 0.067 0.222 0.110 0.168 LingBot-World 0.963 0.335 0.767 0.311 0.555 0.688 Cosmos-Predict-2.5 0.939 0.317 0.736 0.313 0.565 0.664 WoW 0.692 0.251 0.302 0.256 0.280 0.415 Rolling Forcing 0.933 0.285 0.657 0.314 0.570 0.625 LongLive 0.909 0.290 0.579 0.315 0.575 0.593 Yume-1.5 0.925 0.302 0.694 0.302 0.510 0.640 Hunyuan-WorldPlay 0.389 0.219 0.097 0.235 0.175 0.235

General

B.2 Fine-grained Physical Results

##### Table 7 reports physical pass rates aggregated by physical dimension. Table 8 further breaks these scores down into individual physical questions.

- Table 7 Physical dimension pass rates on the same-scene subset. Domain Pipeline Overall Mechanics Thermodynamics Material

Gaming

Matrix-Game 2.0 0.332 0.433 0.172 0.184 LingBot-World 0.884 0.983 0.450 0.969 Cosmos-Predict-2.5 0.867 0.951 0.418 0.884 WoW 0.633 0.806 0.226 0.446 Rolling Forcing 0.853 0.941 0.418 0.854 LongLive 0.851 0.941 0.377 0.865 Yume-1.5 0.813 0.942 0.365 0.902 Hunyuan-WorldPlay 0.852 0.944 0.426 0.843

Robotics

Matrix-Game 2.0 0.364 0.366 0.000 0.372 LingBot-World 0.949 0.961 0.000 0.957 Cosmos-Predict-2.5 0.937 0.939 0.000 0.968 WoW 0.787 0.798 0.111 0.788 Rolling Forcing 0.870 0.857 0.000 0.935 LongLive 0.857 0.864 0.000 0.869 Yume-1.5 0.851 0.857 0.000 0.872 Hunyuan-WorldPlay 0.630 0.577 0.000 0.810

General

Matrix-Game 2.0 0.216 0.246 0.097 0.036 LingBot-World 0.963 1.000 0.519 1.000 Cosmos-Predict-2.5 0.939 0.977 0.613 0.875 WoW 0.692 0.743 0.300 0.562 Rolling Forcing 0.933 0.968 0.581 0.938 LongLive 0.909 0.952 0.581 0.812 Yume-1.5 0.925 0.979 0.370 0.906 Hunyuan-WorldPlay 0.389 0.430 0.323 0.036

- Table 8 Physical question pass rates on the same-scene subset.

Domain Pipeline Grav. Buoy. Comp. Impact Melt Sub. Vap. Cond. Dep. Freez. Color Sol. Hard. Comb.

Matrix-Game 2.0 0.494 0.479 0.324 0.247 0.214 0.000 0.146 0.179 0.167 0.231 – – 0.168 0.222 LingBot-World 0.986 0.944 1.000 1.000 0.429 – 0.462 0.417 0.500 0.500 – – 0.976 0.957 Cosmos-Predict-2.5 0.958 0.986 0.986 0.868 0.357 0.000 0.292 0.513 0.833 0.538 – – 0.901 0.843 WoW 0.850 0.944 0.774 0.540 0.200 0.000 0.161 0.296 0.200 0.333 – – 0.486 0.355 Rolling Forcing 0.949 0.987 0.947 0.871 0.357 0.500 0.298 0.475 0.667 0.615 – – 0.854 0.854 LongLive 0.955 0.986 0.946 0.846 0.429 0.000 0.292 0.436 0.500 0.462 – – 0.875 0.843 Yume-1.5 0.955 1.000 0.900 0.786 0.600 0.000 0.286 0.333 0.500 0.600 – – 0.902 0.900 Hunyuan-WorldPlay 0.957 0.986 0.959 0.844 0.500 0.500 0.271 0.513 0.667 0.538 – – 0.870 0.780

Gaming

Matrix-Game 2.0 0.427 0.467 0.290 0.239 – – 0.000 – – – – 0.000 0.374 – LingBot-World 0.965 1.000 0.985 0.935 – – 0.000 – – – 1.000 0.500 0.962 – Cosmos-Predict-2.5 0.945 1.000 1.000 0.894 – – 0.000 – – – 1.000 0.500 0.972 – WoW 0.840 0.929 0.845 0.662 – – 0.111 – – – 0.000 0.000 0.800 – Rolling Forcing 0.889 1.000 0.889 0.766 – – 0.000 – – – – 0.000 0.941 – LongLive 0.879 0.933 0.938 0.791 – – 0.000 – – – – 0.000 0.873 – Yume-1.5 0.874 1.000 0.938 0.765 – – 0.000 – – – 0.000 0.000 0.885 – Hunyuan-WorldPlay 0.644 0.812 0.615 0.373 – – 0.000 – – – 0.000 0.000 0.822 –

Robotics

Matrix-Game 2.0 0.310 0.267 0.111 0.130 0.125 – 0.000 0.000 1.000 0.000 – – 0.037 0.000 LingBot-World 1.000 1.000 1.000 1.000 0.600 – 0.400 0.500 1.000 0.667 – – 1.000 1.000 Cosmos-Predict-2.5 0.972 1.000 1.000 0.976 0.875 – 0.333 1.000 1.000 0.800 – – 0.903 0.000 WoW 0.767 0.806 0.889 0.634 0.500 – 0.133 – 0.500 0.400 – – 0.581 0.000 Rolling Forcing 0.978 0.935 1.000 0.951 0.875 – 0.267 1.000 1.000 0.800 – – 0.935 1.000 LongLive 0.956 1.000 1.000 0.915 0.875 – 0.400 0.000 0.500 0.800 – – 0.839 0.000 Yume-1.5 0.983 1.000 1.000 0.958 0.400 – 0.267 0.500 1.000 0.333 – – 0.903 1.000 Hunyuan-WorldPlay 0.494 0.467 0.500 0.260 0.500 – 0.067 0.000 1.000 0.600 – – 0.037 0.000

General

#### B.3 Fine-grained Interaction Results

Table 9 reports fine-grained interaction diagnostics. The chunk score measures local caption and action following, the transition score measures boundary smoothness between adjacent chunks, and the global score measures long-range consistency over the stitched video. The raw CLIP score is calibrated into a bounded auxiliary score with fixed thresholds, and the interaction score corresponds to the aggregate interaction metric reported in Table 6.

- Table 9 Fine-grained interaction diagnostics on the same-scene subset. Domain Pipeline Chunk Trans. Global Long Range Global Text CLIP Raw CLIP Aux. Interact.

Gaming

Matrix-Game 2.0 0.135 0.074 0.087 0.087 0.087 0.230 0.150 0.111 LingBot-World 0.796 0.767 0.862 0.875 0.848 0.315 0.575 0.778 Cosmos-Predict-2.5 0.704 0.677 0.700 0.718 0.680 0.306 0.530 0.679 WoW 0.267 0.233 0.247 0.250 0.244 0.247 0.235 0.249 Rolling Forcing 0.665 0.681 0.704 0.733 0.675 0.332 0.660 0.675 LongLive 0.595 0.444 0.625 0.640 0.606 0.322 0.610 0.554 Yume-1.5 0.645 0.727 0.668 0.702 0.632 0.291 0.455 0.659 Hunyuan-WorldPlay 0.483 0.458 0.440 0.464 0.415 0.296 0.480 0.471

Robotics

Matrix-Game 2.0 0.136 0.167 0.041 0.042 0.041 0.252 0.260 0.139 LingBot-World 0.670 0.714 0.881 0.893 0.869 0.314 0.570 0.710 Cosmos-Predict-2.5 0.682 0.707 0.896 0.908 0.885 0.321 0.605 0.721 WoW 0.413 0.501 0.472 0.493 0.451 0.288 0.440 0.447 Rolling Forcing 0.498 0.600 0.632 0.661 0.603 0.329 0.645 0.566 LongLive 0.484 0.288 0.587 0.619 0.556 0.327 0.635 0.470 Yume-1.5 0.553 0.715 0.694 0.722 0.667 0.312 0.560 0.624 Hunyuan-WorldPlay 0.245 0.254 0.134 0.140 0.129 0.309 0.545 0.262

General

Matrix-Game 2.0 0.069 0.064 0.031 0.031 0.029 0.222 0.110 0.067 LingBot-World 0.752 0.819 0.829 0.838 0.812 0.311 0.555 0.767 Cosmos-Predict-2.5 0.746 0.755 0.764 0.782 0.746 0.313 0.565 0.736 WoW 0.314 0.294 0.292 0.293 0.285 0.256 0.280 0.302 Rolling Forcing 0.620 0.727 0.661 0.684 0.629 0.314 0.570 0.657 LongLive 0.598 0.520 0.639 0.661 0.603 0.315 0.575 0.579 Yume-1.5 0.637 0.814 0.718 0.744 0.684 0.302 0.510 0.694 Hunyuan-WorldPlay 0.130 0.030 0.073 0.073 0.073 0.235 0.175 0.097

B.4 Fine-grained Geometry Results

- Table 10 reports fine-grained geometry diagnostics. Srecon measures the quality of the Gaussian-splat reconstruction video, Smeta measures the quality of the rendered meta-view image, and Straj measures cameratrajectory consistency. The 3D consistency score corresponds to the aggregate geometry metric reported in Table 6.

###### Table 10 Fine-grained geometry diagnostics on the same-scene subset. Domain Pipeline Srecon Smeta Straj 3D Cons.

Matrix-Game 2.0 0.160 0.159 0.247 0.189 LingBot-World 0.389 0.372 0.337 0.366 Cosmos-Predict-2.5 0.415 0.388 0.280 0.361 WoW 0.232 0.205 0.231 0.223 Rolling Forcing 0.324 0.292 0.250 0.289 LongLive 0.328 0.292 0.256 0.292 Yume-1.5 0.381 0.361 0.315 0.352 Hunyuan-WorldPlay 0.397 0.363 0.284 0.348

Gaming

Matrix-Game 2.0 0.283 0.298 0.432 0.338 LingBot-World 0.416 0.416 0.348 0.393 Cosmos-Predict-2.5 0.451 0.464 0.523 0.479 WoW 0.297 0.289 0.232 0.272 Rolling Forcing 0.458 0.432 0.278 0.389 LongLive 0.476 0.483 0.458 0.472 Yume-1.5 0.337 0.340 0.185 0.288 Hunyuan-WorldPlay 0.574 0.566 0.660 0.600

Robotics

Matrix-Game 2.0 0.191 0.196 0.271 0.220 LingBot-World 0.373 0.319 0.312 0.335 Cosmos-Predict-2.5 0.341 0.322 0.288 0.317 WoW 0.243 0.225 0.286 0.251 Rolling Forcing 0.283 0.266 0.306 0.285 LongLive 0.289 0.280 0.300 0.290 Yume-1.5 0.318 0.306 0.282 0.302 Hunyuan-WorldPlay 0.177 0.191 0.288 0.219

General

- B.5 Model-level Fine-grained Results

##### Table 11 and Table 12 aggregate the fine-grained geometry and interaction diagnostics at the model-category level.

- Table 11 Model-level 3D consistency submetrics. Category Model GS Meta Camera Motion 3D Cons.

Matrix-Game 2.0 0.216 0.222 0.326 0.255 LingBot-World 0.400 0.383 0.337 0.373

Gaming World Model

Cosmos-Predict-2.5 0.415 0.405 0.378 0.399 WoW 0.262 0.245 0.244 0.250

Robotics World Model

Rolling Forcing 0.359 0.332 0.272 0.321 LongLive 0.379 0.365 0.345 0.363 Yume-1.5 0.338 0.334 0.231 0.301 Hunyuan-WorldPlay 0.426 0.412 0.436 0.424

General World Model

###### Table 12 Model-level interaction submetrics.

Category Model Chunk Trans. Global Long Range Global Text CLIP Raw CLIP Aux. Interact.

Matrix-Game 2.0 0.123 0.109 0.058 0.058 0.057 0.237 0.185 0.113 LingBot-World 0.709 0.751 0.864 0.875 0.850 0.314 0.570 0.734

Gaming World Model

Cosmos-Predict-2.5 0.704 0.705 0.791 0.807 0.775 0.313 0.565 0.707 WoW 0.339 0.359 0.352 0.362 0.341 0.267 0.335 0.345

Robotics World Model

Rolling Forcing 0.600 0.666 0.671 0.699 0.641 0.327 0.635 0.636 LongLive 0.552 0.398 0.613 0.636 0.585 0.323 0.615 0.526 Yume-1.5 0.590 0.745 0.697 0.726 0.667 0.306 0.530 0.649 Hunyuan-WorldPlay 0.320 0.294 0.247 0.259 0.235 0.290 0.450 0.316

General World Model

C Case Study

We provide representative qualitative cases that illustrate how WorldOlympiad diagnoses different failure modes beyond generic video quality. Each case uses the same source prompt or reference context across models, so the comparison focuses on model behavior rather than prompt variation.

###### Table 13 Representative case studies and the corresponding diagnostic signals.

Case Evaluation focus Typical success pattern Typical failure pattern

Physical dynamics

Gravity, impact, deformation, or phase transition

The object follows the expected temporal order, preserves contact constraints, and changes state gradually when required.

The object floats, teleports, deforms without contact, changes phase instantaneously, or violates the expected direction of motion.

3D consistency Gaussian-splat reconstruction and camera trajectory

The reconstructed scene remains stable under novel views, with consistent foreground objects and plausible camera motion.

The reconstruction contains stretched geometry, missing background structure, unstable object identity, or camera motion that disagrees with the reference trajectory.

Interactive rollout

Chunk-level instruction following and transition coherence

Each generated chunk follows its action caption, and the next chunk preserves scene state, agent pose, and object layout.

The model resets the scene at chunk boundaries, ignores control changes, changes object identity, or accumulates visual drift over long horizons.

Gaming case study. Figure 8 shows a gaming case study, where the main diagnostic signals come from geometry consistency and interaction fidelity. The geometry metric examines whether the generated video preserves a stable and spatially coherent game scene under camera movement. In particular, it checks whether the visual content remains consistent with the textual description of the scene, including the expected environment, objects, style, and spatial layout. When the camera moves, a strong model should maintain stable geometry and avoid sudden scene deformation, object disappearance, or inconsistent background structure. The interaction metric further evaluates whether the generated rollout follows the intended action sequence and preserves the game state across chunks. Failure cases include drifting away from the described scene, producing unstable camera transitions, resetting the environment between chunks, or generating actions that no longer match the corresponding captions.

Robotics case study. Figure 7 presents an robotics manipulation case, where WorldOlympiad jointly examines physical plausibility, scene-level geometric consistency, and instruction-following behavior. For physical evaluation, the case highlights failures such as an apple floating in mid-air despite the absence of visible support, indicating a violation of gravity and object-support constraints. For geometry evaluation, the benchmark further checks whether the scene layout remains coherent throughout the rollout. For example, a drawer may suddenly appear or disappear across frames, revealing inconsistent spatial structure and unstable background reconstruction. For interaction evaluation, the judge focuses on whether the robot follows the intended manipulation instruction, such as reaching toward the correct object, grasping the target item rather than a distractor, and maintaining a plausible object state after contact. This case shows that visually plausible robotics videos can still fail when object dynamics, scene consistency, or robot-action alignment are not faithfully preserved.

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

Prompt: Applying backward force, the arm slides the drawer open

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

Prompt: firmly grasping the toy pie slice

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

Prompt: The arm moves steadily downward, positioning its open gripper directly over the green apple

###### Figure 7 Robotics case study from WorldOlympiad. The example visualizes how the benchmark diagnoses physical interaction, object-state consistency, and temporal coherence in robotics world-model rollouts.

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

Prompt: Along the way, they pass by patches of red flowers and several NPCs

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

Prompt: The camera begins with a view of an ancient, open-air amphitheater at night

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

Prompt: The player drives the vehicle forward down the street

###### Figure 8 Gaming case study from WorldOlympiad. The example highlights how interactive game rollouts expose action-following, scene-state preservation, and cross-chunk transition failures.

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

Prompt: She expertly manipulates a long, colorful ribbon, creating large, flowing circles and spirals around her as she gracefully dances and leaps

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

Prompt: In an artificial wave pool, a young boy wearing a black long-sleeved swimsuit

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

Prompt: throws a white frisbee to the right. A black and white dog immediately sprints across the field to chase it

- Figure 9 Real-world case study from WorldOlympiad. The example illustrates how open-domain videos reveal geometric consistency, camera-motion, and long-range visual-coherence issues.

General case study. Figure 9 presents a real-world case study, where all three evaluation dimensions are informative. For physical evaluation, the case checks whether the motion of a thrown frisbee follows a plausible trajectory, rather than floating, stopping unnaturally, or changing direction without visible cause. For geometry evaluation, the benchmark inspects whether the scene remains spatially and semantically consistent over time. For instance, a failure case may abruptly change an indoor scene into an outdoor scene, indicating severe scene-level inconsistency and poor long-range coherence. For interaction evaluation, the judge examines whether the generated video contains meaningful temporal evolution rather than becoming overly static. A strong sample should preserve realistic motion, maintain a coherent scene layout, and continue to reflect the intended event throughout the video. These qualitative examples demonstrate that WorldOlympiad can reveal complementary failure modes in physical dynamics, 3D consistency, and interactive temporal behavior.

- D Human Preference Study Details The human preference alignment study in Table 4 uses the following annotation and aggregation protocol.

Annotation protocol. For each selected evaluation prompt, annotators compare anonymized generated videos from the evaluated models under the same prompt or reference context. Five annotators participate in the study. We sample 20 prompts from the evaluation set and compare all 82 = 28 unordered model pairs under each prompt, resulting in 560 prompt-level pairwise comparisons. Each comparison is independently labeled by all five annotators, yielding 2,800 individual preference labels. Annotators are instructed to judge the overall preference using four criteria: visual quality, physical plausibility, temporal coherence, and interaction fidelity. Model names are hidden during annotation. Ties are allowed when two videos are indistinguishable or when their strengths and weaknesses are balanced.

[Figure 242]

[Figure 243]

- Figure 10 Human preference annotation interface used in the alignment study. Human Preference Annotation Prompt

Input: Two anonymized generated videos, Video A and Video B, produced from the same prompt or reference context.

Instruction: Choose the video that better satisfies the prompt and shows more realistic world dynamics. Consider visual quality, physical plausibility, temporal coherence, object and scene consistency, and interaction fidelity. If neither video is clearly better, select Tie.

Output format: Return one label from {A, B, Tie} and a one-sentence rationale.

Score aggregation. Let ym,n,a denote the preference outcome assigned by annotator a for model m in comparison n. A win contributes 1, a tie contributes 0.5, and a loss contributes 0. We first average the five annotator labels for each pairwise comparison and then compute the model-level preference rate:

1 5

y¯m,n =

5

1 Nm

ym,n,a, Smhuman =

a=1

Nm

y¯m,n,

n=1

##### where Nm = 140 is the number of aggregated valid comparisons involving each model. Human ranks are obtained by sorting Shuman in descending order. WorldOlympiad ranks are obtained by sorting the automatic

overall evaluation score in descending order, where Sauto is the same three-track average Sall used in the main benchmark table.

Rank correlation. We measure alignment using Spearman’s rank correlation:

6 Mm=1 d2m M(M2 − 1)

, dm = rmhuman − rmauto,

ρ = 1 −

where M is the number of evaluated models. For the eight models with human preference annotations, the resulting correlation is 0.95, indicating strong agreement between human preference and the WorldOlympiad automatic ranking.

The rank disagreements occur only in two adjacent pairs: LongLive and Yume-1.5, and Matrix-Game 2.0 and WoW. These swaps have a limited effect on the overall correlation and suggest that the automatic evaluator preserves the main model ordering while still exposing borderline cases where human preference and rubric-based automatic scores differ.

