# arXiv:2602.09146v1[cs.CV]9Feb2026

## SemanticMoments: Training-Free Motion Similarity via Third Moment Features

Saar Huberman1,2 Kfir Goldberg1 Or Patashnik2 Sagie Benaim3 Ron Mokady1

1BRIA AI 2Tel Aviv University 3Hebrew University of Jerusalem

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

Refence video Similar Apperance Similar Semantic Motion

Similar Apperance Similar Semantic Motion

[Figure 12]

[Figure 13]

Figure 1. Motion-centric retrieval with Semantic Moments. Existing video-similarity methods over-rely on static appearance and scene context, overlooking temporal dynamics. Our approach retrieves clips that match the semantic motion. We retrieve the drinking-coffee motion across identities, disentangling motion from appearance, while all baselines similarly return look-alikes and miss the action.

### Abstract

Retrieving videos based on semantic motion is a fundamental, yet unsolved, problem. Existing video representation approaches overly rely on static appearance and scene context rather than motion dynamics, a bias inherited from their training data and objectives. Conversely, traditional motion-centric inputs like optical flow lack the semantic grounding needed to understand high-level motion. To demonstrate this inherent bias, we introduce the SimMotion benchmarks, combining controlled synthetic data with a new human-annotated real-world dataset. We show that existing models perform poorly on these benchmarks, often failing to disentangle motion from appearance. To address this gap, we propose SemanticMoments, a simple, training-free method that computes temporal statistics (specifically, higher-order moments) over features from pretrained semantic models. Across our benchmarks, SemanticMoments consistently outperforms existing RGB, flow,

and text-supervised methods. This demonstrates that temporal statistics in a semantic feature space provide a scalable and perceptually grounded foundation for motioncentric video understanding.

### 1. Introduction

Humans perceive motion not as raw pixel displacement, but as meaningful, structured, and semantic change over time [6, 13, 27, 33, 36]. Two videos may differ visually, but convey similar motion when comparable entities undergo analogous temporal transformations (as can be seen in Fig. 1). In this view, motion similarity captures how perceptual structure unfolds over time at the semantic level. Retrieving videos that share such similar motion is a fundamental yet largely unsolved problem, which we tackle in this paper. Such capability would benefit a wide range of applications, from constructing motion-centric datasets to enhancing motion control in generative video models.

We find that most existing video retrieval methods rely on representations that are biased towards static appearance and scene context rather than motion dynamics, often producing results that are visually similar but dynamically unrelated1. This bias stems from the data and objectives used for training. The problem often begins with the datasets themselves: action-recognition labels are a flawed proxy for motion, as identical actions can exhibit distinct motions while different actions can share similar dynamics.

Consequently, models trained on such data learn to exploit these dataset biases. Supervised approaches [5, 8, 11, 31], for instance, can often recognize an action category from a single frame ( Fig. 2), learning to encode static appearance cues (e.g., background, objects, and clothing) rather than true temporal structure. We find this appearance bias is general; it also occurs in self-supervised RGB methods [10, 14, 25, 29, 34], which often learn to prioritize simple appearance consistency over modeling complex temporal change. The primary alternative, methods relying solely on optical flow, presents the opposite problem: while robust to appearance, they are unable to capture the important semantic information that defines perceptual motion.

To illustrate this bias concretely, we introduce the SimMotion-Synthetic dataset. This synthetic dataset consists of video pairs that share identical motion but differ in controlled factors such as viewpoint and visual style. As our retrieval experiments show, existing methods often fail to identify videos with the same motion, as their representations are sensitive to these non-motion factors. This emphasizes the need for representations that explicitly capture motion. However, collecting large-scale annotated data for such training is prohibitively expensive.

To this end, we introduce SemanticMoments, a simple, training-free motion representation designed to move beyond static appearance cues. Our core insight is that standard temporal aggregation, such as average pooling (first statistical moment), effectively captures average appearance but discards the rich temporal dynamics of how features change. To explicitly capture this motion, SemanticMoments computes a richer set of temporal statistics, specifically the higher-order moments (e.g., variance and skewness), over patch-level embeddings from pretrained semantic models like DINO [7, 24]. This results in a compact descriptor that summarizes high-level, structured change. It also leverages the strong semantic correspondences of DINO features, which track meaningful object parts as trajectories in the feature space [35]. Our approach directly captures semantic dynamics and requires no optical flow, labeled data, or additional training, making it broadly compatible with off-the-shelf backbones.

1A similar phenomenon was observed in image classification networks [12], where models trained on ImageNet are biased toward recognizing local texture rather than global structure.

To evaluate motion-based retrieval in realistic settings, we introduce SimMotion-Real, our second benchmark, which consists of human-annotated video pairs labeled for perceptual motion similarity, independently of appearance. Since producing such annotations is time-consuming, the dataset is intentionally small but carefully curated to capture diverse, natural motion patterns. This benchmark enables rigorous, real-world evaluation of motion representations and complements our controlled synthetic analysis. We show that our method consistently retrieves videos with similar motion, outperforming existing approaches. To summarize, we provide the following contributions:

- • We identify and analyze the dominant appearance bias in current video representations, showing they prioritize static cues over motion dynamics.
- • We introduce the SimMotion benchmarks, a new suite of synthetic and real-world human-annotated datasets rigorously evaluating perceptual motion similarity.
- • We propose SemanticMoments, a simple, efficient and training-free method that represents motion using the temporal statistics of semantic features. SemanticMoments outperforms existing state-of-the-art approaches.

### 2. Related Work

Video Representation Learning. Prior work retrieves videos based on the similarity of features produced by some pretrained model, trained using different supervision modes: action recognition, multimodal supervision, and self-supervision.

Action Recognition. Early models, trained on UCF101 [32], HMDB-51 [18], and Kinetics [17], learn to classify videos into predefined categories. The architectures used include Two-Stream Networks [31] and I3D [8] with parallel multi-stream inputs (RGB and optical flow), or SlowFast [11] with varying frame rates. While these architectures were designed to decouple appearance and motion, their shared training objective undermined this goal. Since both paths were trained to predict the same action labels, which are often defined by static objects or scenes, the models learned that the appearance-based path was the most reliable predictor for minimizing loss. As a result, they inherited the dataset’s bias, continuing to rely on appearance cues rather than motion.

Multimodal Supervision. Recent methods adapt models pretrained on text-image/video data via contrastive learning. CLIP4Clip [21] aggregates CLIP [30] frame features for video-text alignment; VideoCLIP [40] jointly learns audio, visual, and textual embeddings; and X-CLIP [22] uses cross-modal transformers for fine-grained frame-text interaction. Despite strong retrieval results, these models inherit a limitation of language supervision: motion is often underspecified (e.g., “a person walking” or “an object rotating” can match many distinct videos).

###### Kinetics UCF101 HMDB-51

[Figure 14]

[Figure 15]

[Figure 16]

Dunking basketball Playing Cello Ride horse

[Figure 17]

[Figure 18]

[Figure 19]

Juggling balls Shaving beard Brush hair

Figure 2. Current benchmarks are appearance-centric. We show random frames from popular video-retrieval datasets. In many cases, static objects (e.g., a cello, a razor) or scene context (e.g., a basketball court) suffice to identify the action label (e.g., Playing Cello, Shaving Beard) without observing motion. This bias enables high accuracy from purely appearance-based cues, discouraging models from learning true temporal dynamics.

Self Supervision. Self-supervised approaches learn embeddings without labels using contrastive (e.g., CVRL [29], VideoMoCo [25]) or masked-prediction objectives (e.g., VideoMAE [34], VideoPrism [42]). While many of these methods attempt to target motion, such as SCVRL [10] (via shuffling) or the more recent V-JEPA [3, 4] (by predicting latent spatiotemporal regions based on JEPA [2]), they remain sensitive to their prediction target design. Their objectives often make learning static appearance consistency the simplest path to minimizing loss. For example, in both masked and predictive modeling, a model is ”incentivized to preserve appearance even as motion changes”, thus inheriting the very appearance bias we aim to solve.

Disentangling Motion from Appearance. Recent works seek to isolate motion, primarily for generative tasks like motion transfer. MoFT [39] identifies motion-sensitive components in diffusion models via PCA; DIFTFlow [28] extracts motion trajectories from attention maps; SMM [41] reduces appearance bias using inter-frame differences and spatial marginal means; and MotionClone [19] derives sparse temporal attention to guide motion imitation. While they pursue disentanglement for generative tasks, their goal is not to produce the compact, global embeddings required for large-scale retrieval, which is the focus of our work.

In parallel, self-supervised vision transformers such as DINO support motion reasoning: DiVE [16] uses DINOv2 to extract localized trajectories and preserve subject identity via LoRA adapters, and MotionShot [20] fuses DINO with Stable Diffusion features to align high-level semantics with low-level structure for controllable transfer. Together, these generative approaches show that pretrained visual features capture motion-relevant structure across time. We adopt this insight but shift the goal: rather than transfer, we use pretrained feature extractors to obtain a compact, training-free representation for motion-based video similarity.

Static Dyn-App Dyn-Obj View Style

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

Figure 3. Controlled variation in SimMotion-Synthetic. We visualize sample pairs from the five distinct categories in our benchmark. From left to right: Static Object (background varies), Dynamic Appearance (subject clothing/attributes vary), Dynamic Object (subject identity varies), View (camera angle varies), and Scene Style (rendering style varies). In each column, the top and bottom videos are temporally synchronized and share identical motion dynamics, differing only in the specified visual factor.

### 3. The “SimMotion” benchmarks

Existing methods for motion similarity or motion retrieval are typically evaluated on action recognition benchmarks. However, as discussed earlier, defining motion through discrete categories or textual descriptions is inherently limited. Categories such as “walking”, “jumping”, or “dancing” provide only coarse, high-level descriptions, useful for naming an action but insufficient to convey the structure and the dynamics of the motion. In fact, as demonstrated in Fig. 2, such categories can be identified from single frames only without observing motion.

Human perception of motion operates across multiple levels of abstraction: at a coarse level, we distinguish action types (e.g., walking vs. jumping), while at a finer level, we perceive variations in structure and dynamics [6, 13, 27, 33, 36]. For example, the category ’dancing’ (coarse level) includes both ’waltz’ and ’breakdancing’ (fine level), which have entirely different motion structures. Furthermore, even two different instances of a ’waltz’ will vary significantly in their execution and dynamics, yet are perceptually similar. As categorical labels capture only part of this hierarchy, measuring motion similarity should also incorporate structural and dynamic properties. Because similarity in these aspects is continuous rather than discrete, benchmarks should be based on relative similarity measures rather than categorical labels.

To better evaluate motion similarity, we introduce the SimMotion benchmarks, where similarity is defined through relative comparisons rather than action classification. Specifically, each SimMotion benchmark includes both intra-class pairs, where videos share the same coarse motion type but differ in finer details, and inter-class pairs that differ in their motion altogether (e.g., running vs. jumping). This composition enables evaluation of both finegrained similarity and broader action-type discrimination.

Original Pixar Paintbrush

[Figure 30]

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

[Figure 41]

(a)

X-CLIP SlowFast VideoMoCo MaCLR

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

(b) DINOv2 VideoPrism VideoMAE

[Figure 46]

[Figure 47]

[Figure 48]

GlobalMean3rdMoment

[Figure 49]

[Figure 50]

[Figure 51]

(c)

- Figure 4. Motion-focused similarity with moment statistics. (a) Appearance-altered edits preserve the same underlying motion for each motion group mi, while changing visual style. (b) Baseline embeddings yield similarity heatmaps that are sensitive to appearance rather than motion. (c) Our moment-based embedding (using the first three moments over patch features) produces clearer motion-consistent clusters (corresponding to shared motion mi) than global mean pooling. Brighter cells indicate higher cosine similarity.

Each benchmark serves a distinct purpose. SimMotionSynthetic serves as a controlled diagnostic benchmark, constructed to isolate and test the failure modes of existing representations. Its design allows us to systematically analyze how non-motion factors, such as appearance or viewpoint, influence similarity results while the underlying motion is held constant. SimMotion-Real complements this by testing generalization to real-world in-the-wild scenarios. It is manually curated and therefore smaller in scale. Its purpose is not to isolate specific factors, but to evaluate a model’s alignment with human perception on realistic videos where motion is perceptually similar but not identical, and confounding factors like appearance and execution vary naturally. Both benchmarks will be made fully available and open source. In the following, we describe each benchmark in detail.

#### 3.1. SimMotion-Synthetic

Benchmark Structure. The benchmark contains 250 triplets (750 videos in total), each composed of a reference video, a positive that shares the same motion, and a hard negative with similar appearance but different motion. Triplets are organized into five categories, each defining how the positive video differs from the reference while preserving motion, with 50 instances per category: (1) Static Object, where non-moving elements in the scene are added, removed, or replaced; (2) Dynamic Object, where the main moving subject is replaced by another entity performing the same motion; (3) Dynamic Appearance, where visual attributes such as clothing, tattoos, or accessories are modified on the moving subject; (4) Scene Style, where the ren-

dering style of the scene is changed (e.g., realistic, painting, or sketch); and (5) View, where the camera viewpoint is altered. Visual examples of these categories are shown in Fig. 3. Each category isolates a different type of variation, with positives preserving motion dynamics and negatives sharing appearance but differing in motion.

Generation Pipeline. For each triplet, we first generate four textual prompts using GPT-4.1 [1]: (1) a base prompt describing the scene, (2) a prompt defining the modification in the scene according to the category, (3) a video prompt specifying the intended motion, and (4) a negative motion prompt describing a distinct motion for the same subject.

Prompts (1) and (2) are used to synthesize reference and positive images with Gemini2.5-Flash [9], which are concatenated and passed to WAN 2.2 (Image-to-Video) [37] to generate temporally synchronized videos sharing identical motion using a single, shared prompt (3). We follow a synchronization strategy, conceptually similar to ICLoRA [15], which ensures WAN 2.2 applies an identical temporal transformation to both (semantically-aligned) start frames. This results in two videos that are temporally synchronized and share the exact same motion dynamics while differing only in the specified visual factors.

Finally, a hard negative video is produced from the same base image using prompt (4), yielding identical appearance but different motion. This pipeline ensures strong temporal alignment while enabling systematic control over appearance, subject, and viewpoint. All videos are 5 seconds long, sampled at 16 fps with a spatial resolution of 512×512.

For evaluation, all other videos in the benchmark are treated as additional negatives during retrieval, providing a

[Figure 52]

Patch-wise Temporal moments Mean(𝑀( , )( ) )

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

Spatial Mean

[Figure 58]

[Figure 59]

Patch–wise Embedder

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

Variance (𝑀( , )( ) )

[𝛼 𝑀 ,𝛼 𝑀 ,𝛼 𝑀 ]

[Figure 65]

skew (𝑀( , )( ) )

- Figure 5. SemanticMoments pipeline. Patch-wise features are extracted per frame using a pretrained embedder (e.g., DINO) and summarized over time using the first three temporal moments (mean, variance, and skewness). Spatial aggregation yields one descriptor per moment, which are combined into a global motion-centric video embedding.

mal for modeling motion similarity.

diverse pool and increasing the difficulty of motion discrimination. This setup provides a clean and controlled basis for evaluating motion representations.

To probe this limitation, we construct a controlled set of four walking sequences and synthesize motion-preserving variants that vary only in style while maintaining identical dynamics, as can be seen in Fig. 4a. These controlled variations isolate motion similarity as the sole factor of interest.

#### 3.2. SimMotion-Real

The benchmark contains 40 examples, each centered on a reference video paired with a positive and a negative counterpart. The positive video depicts the same underlying motion despite differences in appearance or context, while the negative shares a similar appearance but differs in motion. Negative pairs are obtained by sampling different short clips with different motion from the same source video, ensuring comparable visual context. Positive candidates are retrieved from Pexels [26] using text-based motion descriptions and ranked through crowd-sourced annotation, where annotators judged which clips exhibited the most similar motion to the reference, regardless of appearance or scene differences (see supplementary for further details). This process grounds similarity in human perception of motion rather than in visual or categorical cues, yielding a benchmark that reflects naturally occurring motion variability beyond the controlled design of SimMotion-Synthetic. In addition to the hard negatives, we include randomly sampled videos from the Kinetics-400 test set as random negatives. Together, these examples form a realistic benchmark for evaluating whether representations can generalize to motion similarity under natural variation.

Ideally, a motion-sensitive representation should cluster each clip with its motion-equivalent variants while distinguishing distinct walking styles. We evaluate this by computing pairwise cosine-similarity matrices over video embeddings and visualizing them as heatmaps. As shown in Fig. 4b, existing methods show partial grouping but fail to consistently isolate shared motion across different styles.

We conclude that embeddings from self-supervised image and video encoders (e.g., DINOv2, VideoPrism) fail to consistently capture fine-grained motion similarity. Leveraging their semantic correspondence and summarizing temporal variation with higher-order moments (e.g., variance, skewness) yields motion-sensitive embeddings that correctly cluster motion-equivalent variants while distinguishing distinct walking style, as we demonstrate in Fig. 4c.

### 5. SemanticMoments

Our analysis in Sec. 4 reveals that current video encoders tend to focus on appearance and scene context. Their embeddings shift noticeably in feature space under style-only edits that preserve geometry and motion. Our key observation is that incorporating higher-order temporal moments result in features that better represent motion, as such embeddings yielding similarities that better reflect underlying semantic motion patterns. Motivated by this, we define in Sec. 5.1 a general, moment-based representation space M+ that encodes temporal statistics into structured video descriptors We then instantiate it in Sec. 5.2 with SemanticMoments, a practical, training-free method that aggre-

### 4. Analysis

Prior work primarily optimizes for action recognition or video–text alignment, which under-specify temporal structure. In addition, self-supervised approaches such as masked autoencoding often under-represent motion and remain biased toward appearance. We therefore hypothesize that embeddings learned from these objectives are subopti-

gates the first three temporal moments of pretrained features to form motion-centric representations (see Fig. 5). Finally, in Sec. 6 we evaluate SemanticMoments, demonstrating strong performance on motion-based video retrieval.

#### 5.1. M+: A Parametric View of Temporal Statistics

We define M+, a moment space over temporal visual features, extending conventional video representations from a single embedding to a structured set of temporal descriptors. Rather than collapsing temporal information into one pooled vector, we compute multiple statistical moments across time, where each moment captures a distinct temporal characteristic.

Patch-wise temporal moments. Given a Video C and feature extractor F, feeding each frame t ∈ {1,...,T} to the feature extractor produce P patch features Ft ∈ RP×d. We denote the d-dimensional feature of patch p by ft,p ∈ Rd. We define the first temporal moment as the (noncentral) mean:

1 T

µ(1)p = µp =

T

ft,p.

t=1

For higher-order orders k > 1, we compute central temporal moments:

1 T

µ(pk) =

T

(ft,p − µp)k.

t=1

Intuitively, µ(1)p encodes the average appearance of patch p across time, µ(2)p captures the magnitude of temporal variation (motion energy), and µ(3)p reflects the directional asymmetry of change (motion polarity).

Spatial aggregation. So far, moments consists of significant spatial representation. To obtain a global motion representation per moment order, we aggregate the per-patch moments spatially:

M(k) =

P

1 P

µ(pk) ∈ Rd, k = 1,...,K.

p=1

This yields one descriptor M(k) per statistical moment, each summarizing a distinct aspect of temporal variation.

Moment embedding. The general form of the M+ videolevel representation is obtained by a weighted concatenation of the different moment vectors, where αk ∈ R is the relative contribution of the k-th moment:

ϕvideo = [α1M(1); α2M(2); ...; αKM(K) ] ∈ RKd,

#### 5.2. Design choices

We now present our video representation, designed to excel in motion-focused video similarity. We operationalize M+ using pretrained semantic backbones: DINOv2, VideoMAE, and VideoPrism. We use the first three temporal moments (k = 1,2,3): M(1) corresponds to average pooling, while M(2) and M(3) capture the magnitude and polarity of temporal change. The moment vectors are concatenated using weights αk, yielding a final embedding ϕvideo ∈ R3d.

For all experiments, we sample T = 32 frames uniformly per video at the backbone’s native resolution and compute patch features from the final encoder layer. Unless otherwise stated, we use α1 = 1, α2 = 8, and α3 = 4. The entire process is training-free and introduces minimal additional computational cost, making it scalable to largescale video collections.

### 6. Experiments

We evaluate the effectiveness of our method on both synthetic and real-world motion-similarity benchmarks introduced in Sec. 3. We begin by outlining the baselines and evaluation protocol, followed by quantitative results on SimMotion-Synthetic (Sec. 6.1) and SimMotion-Real (Sec. 6.2). Finally, we present ablation studies (Sec. 6.4) and discuss limitations (Sec. 6.5). Videos and additional results are provided in the supplementary materials.

Baselines. We compare our method against a range of established approaches in video representation learning. Multimodal retrieval models, including CLIP4Clip [21] and XCLIP [22], leverage large-scale image–text pretraining for video retrieval. Optical-flow–based models such as I3D [8], CoCLR [14], and MaCLR [38] explicitly incorporate motion via flow. I3D and CoCLR use two-stream RGB–flow architectures, while MaCLR employs flow supervision only during training to guide motion-aware features. RGBbased supervised architectures, including SlowFast [11] and TimeSformer [5], are trained on large-scale action recognition datasets (e.g., Kinetics [17]) to learn spatiotemporal representations directly from frames. Self-supervised transformer encoders such as VideoMAE [34] and VideoPrism [42] are pretrained with masked reconstruction objectives, and DINOv2 [24] serves as a strong image-only self-supervised baseline. We use the publicly available implementations for all methods. Together, these baselines cover most of video representation techniques, enabling a comprehensive evaluation of our moment-based approach.

Evaluation Protocol Since our primary application is motion-focused retrieval, we adopt a retrieval-based evaluation. For each method, we extract video embeddings, ℓ2normalize them, and compute cosine similarity between a

Reference video Reference video

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

Similar motion Similar motion

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

Similar appearance Similar appearance

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

(A) (B)

- Figure 6. Motion vs. Appearance Bias. Left: the dominant motion is a dog walking; while VideoPrism retrieves a static yoga pose (bottom row) based on the background or inferred label ”woman doing yoga”, our Semantic Moments (middle row) successfully retrieves a dog walking despite the different background. Right: although the motion is opening a door, VideoMAE retrieves a video matching the “ambulance” context. In contrast, our method aligns with the underlying dynamics, ignoring static appearance or coarse semantics.

query and all candidates. Given a query video, we rank motion-preserving positives and distractors by similarity and report the success rate of closest video retrieval. Because our motion-focused datasets are medium in scale and we emphasize on precision, we highlight this as the most informative metric.

#### 6.1. Evaluation on SimMotion-Synthetic

We begin with SimMotion-Synthetic, a controlled benchmark that isolates motion similarity under systematic appearance variations. It defines five motion-preserving edit categories (Sec. 3.1, examples in Fig. 3) and reports retrieval accuracy. Tab. 1 summarizes results.

SimMotion-Synthetic reveals complementary weaknesses across different baselines. CLIP-based multimodal models tend to under-specify motion: when appearance shifts while motion is held fixed, their similarity is unstable. RGB-supervised representations are dominated by appearance and are sensitive to style changes. Optical-flow models are motion-aware but struggle to generalize across subjects and viewpoints, as flow fields vary with shape and camera geometry even for identical dynamics.

A category-wise analysis clarifies these effects. In Static Object, where changes occur only in static regions, CLIPbased and RGB-trained models often collapse when backgrounds differ, whereas flow-based methods work well as static regions contribute little to optical flow. In Dynamic Appearance, I3D excels by discounting texture, and SEMANTICMOMENTS matches this robustness without explicit flow. In Dynamic Object (replacing the moving subject while preserving dynamics), all baselines degrade: flow features drift with subject identity, while SEMANTICMOMENTS maintains high similarity by summarizing temporal evolution in a semantics-aware feature space. View

Table 1. Synthetic motion-similarity on SimMotion-Synthetic. Retrieval accuracy (higher is better) across motion-preserving edit categories. The benchmark holds motion fixed while varying appearance factors (object identity/attributes, view, and scene style), exposing where representations over-index on appearance. SemanticMoments denotes our moment-based representation instantiated with different frame encoders. Our method achieves the best overall average score.

Static Dyn-App Dyn-Obj View Style Avg

CLIP4Clip 48.00 64.00 18.00 62.00 20.00 42.20 X-CLIP 68.00 80.00 12.00 64.00 32.00 51.20

SlowFast 86.00 96.00 74.00 76.00 58.00 78.00 TimeSformer 76.00 86.00 36.00 62.00 32.00 58.40 VideoMoCo 78.00 84.00 82.00 46.00 32.00 64.40 VideoMAE 90.00 98.00 64.00 78.00 66.00 79.20 VideoPrism 64.00 90.00 28.00 68.00 22.00 54.40 V-JEPA2 92.00 98.00 72.00 64.00 46.00 74.40

I3D 98.00 96.00 74.00 74.00 80.00 84.40 CoCLR 76.00 82.00 52.00 40.00 26.00 55.20 MaCLR 56.00 78.00 68.00 38.00 10.0 50.00

DINOv2 30.00 88.00 18.00 62.00 24.00 44.40 SemanticMomentsDINO 96.00 98.00 78.00 82.00 78.00 86.40 SemanticMomentsVideoPrism 94.00 100.00 56.00 74.00 74.00 79.60 SemanticMomentsVideoMAE 92.00 94.00 76.00 82.00 74.00 83.60 SemanticMomentsV-JEPA2 100.00 100.00 80.00 70.00 72.00 84.40

changes hurt both RGB and flow baselines due to geometric misalignment, whereas SEMANTICMOMENTS is comparatively robust by aggregating frame-wise semantics over time rather than relying on correspondence.

Overall, SEMANTICMOMENTS achieves the best or competitive performance across categories (Tab. 1), with strong gains in Dynamic Object and View, and near–flowlevel robustness in Static Object and Dynamic Appearance. This indicates that simple moment statistics over semantic features mitigate the primary failure modes exposed by SimMotion-Synthetic while remaining training-free and

Table 2. Real-world motion retrieval on SimMotion-Real-1K. Retrieval accuracy with 1,000 candidates per query (one motionpreserving positive). SemanticMoments denotes our momentbased representation instantiated with different frame encoders. As can be seen, SemanticMoments improve the baselines significantly for semantic motion retrieval.

##### Models Retrieval Accuracy

CLIP4Clip 10.00 X-CLIP 10.00

I3D 27.50 CoCLR 5.00 MaCLR 2.50

SlowFast 15.00 TimeSformer 12.50 VideoMoCo 12.50 VideoMAE 17.50 VideoPrism 12.50 V-JEPA2 27.50

DINOv2 12.50 SemanticMomentsDINO 42.50 SemanticMomentsVideoPrism 22.50 SemanticMomentsVideoMAE 22.50 SemanticMomentsV-JEPA2 27.50

encoder-agnostic.

#### 6.2. Evaluation on SimMotion-Real

While the synthetic benchmark enables granular, categorylevel analysis, real-world evaluation is essential: shifts in appearance, camera motion, timing, and scene complexity often break controlled setting assumptions. SimMotionReal comprises unconstrained videos with unsynchronized motion, making it a test of semantic motion similarity rather than geometric correspondence, since actions are related but rarely identical. As shown in Tab. 2, all methods struggle to get high scores, reflecting real-world variability and noise. Flow-based approaches (e.g., I3D) excel when motion is aligned but lose effectiveness on unsynchronized clips, where flow consistency breaks despite similar semantics; CLIP-based and RGB-only models remain dominated by appearance. In contrast, SEMANTICMOMENTS maintains strong retrieval accuracy and attains the best overall scores, suggesting that temporal statistics over semantic features are more robust to in-the-wild variability. Despite these gains, the absolute numbers indicate that motionsimilarity retrieval in the wild remains a challenging open problem. A visual example is provided in Fig. 6, demonstrating the challenge of separating motion from appearance.

Table 3. Gesture classification on Jester benchmark. Top-1 majority vote and weighted kNN accuracy on the Jester validation set (K=20). SemanticMoments consistently improves performance across different backbones.

Method Acc@1maj Acc@1w-kNN Acc@5w-kNN

X-CLIP 26.0 20.2 42.2 Clip4CLIP 8.5 8.3 11.3 TimeSFormer 16.7 10.3 20.1 SlowFast 19.7 17.2 34.8 VideoMoCo 12.1 12.0 24.4 I3D 26.8 25.8 53.2 VideoMAE 23.8 22.7 43.4 V-JEPA2 12.5 12.3 20.7 DINOv2 8.5 7.8 9.7

SemanticMomentsDINO 25.7 25.0 50.8 SemanticMomentsVideoMAE 26.9 26.5 55.0 SemanticMomentsV-JEPA2 28.6 28.3 47.2

#### 6.3. Gesture-Level Evaluation on Jester

We further extend our evaluation to the publicly available Jester gesture benchmark [23], which contains videos annotated with distinct gesture motion categories.

We evaluate whether SemanticMoments improves gesture-level separability in the embedding space under different video representations. To quantify this effect without training an additional classifier, we adopt a standard kNN evaluation protocol (K=20) on the validation set. For each query video, we retrieve its K nearest neighbors and predict the gesture label using their annotations. Majority-vote accuracy assigns the most frequent neighbor label, while weighted kNN additionally weights each neighbor contribution by its similarity to the query.

As shown in Table 3, applying SemanticMoments consistently improves the metrics across all backbones, indicating stronger motion representations.

#### 6.4. Ablation Studies

We evaluate SemanticMoments on SimMotion-Real by varying three design axes: the order and weighting of temporal moments, the representation level at which moments operate (frame, patch, or patch-difference), and the fusion strategy used to combine moment embeddings. As summarized in Tab. 4, incorporating higher-order moments consistently improves motion alignment over single-moment baselines, indicating that richer temporal statistics capture complementary dynamics beyond average trends.

Tab. 4 further shows that operating at localized, patchlevel granularity better preserves fine motion structure than global frame-level representations, and that applying moments directly on raw patch embeddings outperforms applying them to patch-difference representations. For fusion, additive integration provides strong precision with a compact representation, while concatenation can favor broader recall at the cost of higher dimensionality. Overall, the abla-

- Table 4. Ablation on SimMotion-Real. We systematically analyze our DINO-based moment representation across three complementary axes: (1) Moment configuration — comparing singlemoment versus multi-moment setups to assess how temporal coverage influences motion alignment. (2) Representation level applying moments at different abstraction levels, including framelevel (global embeddings), patch-level (spatially localized features), and patch-difference (temporal gradients between patches), to isolate the contribution of spatial granularity and motion sensitivity. (3) Embedding combination — evaluating how final representations are merged, either by simple summation or concatenation, to study the effect of interaction strength between moment features. Together, these experiments disentangle how moment granularity, representation hierarchy, and feature fusion each contribute to accurate motion-centric retrieval.

Configuration Retrieval Accuracy

- DINO(1,0,0)-patch-concat 12.50

- DINO(0,1,0)-patch-concat 10.00
- DINO(1,1,0)-patch-concat 12.50 DINO(1,8,0)-patch-concat 15.00

- DINO(1,1,1)-patch-concat 30.00

DINO(1,8,4)-patch-sum 30.00 DINO(1,8,4)-frame-concat 12.50 DINO(1,8,4)-diff-patch-concat 22.50

DINO(1,8,4)-patch-concat 42.50

- Table 5. Effect of temporal sampling on SimMotion-Real retrieval. Retrieval accuracy with different numbers of uniformly sampled frames per video. Best results are achieved at 32 frames.

number of frames 4 8 16 32 64 Retrieval Accuracy 22.5 35 40 42.5 35

tions support our design choice of multi-order localized moment modeling. Finally, Tab. 5 analyzes the effect of temporal sampling density, showing that retrieval performance improves up to 32 uniformly sampled frames, after which gains saturate.

- 6.5. Limitations

While our method performs well in most cases, motionsimilarity retrieval in the wild remains challenging. First, some motions are inherently difficult (e.g., fine hand gestures, long-horizon actions, multi-agent interactions). Being training-free, our approach cannot be tuned to these corner cases as effectively as methods finetuned on designated datasets — an avenue for future work. Second, the field lacks a strong, universal video representation comparable to CLIP/DINO for images. Since our method depends on such backbones, its ceiling is bounded by their quality, and we expect video-native backbones to yield significant gains.

Finally, failures persist for extremely subtle dynamics (e.g., breathing) and for motions defined by the absence of motion (e.g., waiting). Overall, we view this work as a step toward motion-centric video understanding, and anticipate that improved video backbones and targeted training will help close these gaps.

### 7. Conclusion

We introduce the task of motion-centric video similarity, targeting how well representations capture and compare motion. Existing retrieval benchmarks are suboptimal for this goal, as labels are often recoverable from static appearance or scene context rather than dynamics. To address this, we propose two dedicated evaluations: SimMotion-Synthetic, a controlled, diagnostic benchmark, and SimMotion-Real, an unconstrained, real-world benchmark. Together, they form a focused testbed for analyzing motion perception in video representations and reveal systematic limitations of current models. Building on these insights, we present SemanticMoments, a training-free representation that encodes motion via temporal statistics of pretrained semantic features. Despite its simplicity, SemanticMoments achieves strong motion alignment across multiple backbones and consistently outperforms prior approaches. Still, results on real-world data indicate that models remain far from human-level motion perception, underscoring both the challenge and opportunity of this task. We position SimMotion and SemanticMomentsas an initial but foundational step toward robust, motion-aware, and perceptually aligned video representations.

### References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774,

2023. 4

- [2] Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, and Nicolas Ballas. Self-supervised learning from images with a joint-embedding predictive architecture. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15619–15629, 2023. 3
- [3] Mido Assran, Adrien Bardes, David Fan, Quentin Garrido, Russell Howes, Matthew Muckley, Ammar Rizvi, Claire Roberts, Koustuv Sinha, Artem Zholus, et al. V-jepa 2: Selfsupervised video models enable understanding, prediction and planning. arXiv preprint arXiv:2506.09985, 2025. 3
- [4] Adrien Bardes, Quentin Garrido, Jean Ponce, Xinlei Chen, Michael Rabbat, Yann LeCun, Mahmoud Assran, and Nicolas Ballas. Revisiting feature prediction for learning visual representations from video. arXiv preprint arXiv:2404.08471, 2024. 3
- [5] Gedas Bertasius, Heng Wang, and Lorenzo Torresani. Is

- space-time attention all you need for video understanding? In Proceedings of the International Conference on Machine Learning (ICML), 2021. 2, 6
- [6] Randolph Blake and Maggie Shiffrar. Perception of human motion. Annu. Rev. Psychol., 58(1):47–73, 2007. 1, 3
- [7] Mathilde Caron, Hugo Touvron, Ishan Misra, Herv´e J´egou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9650–9660, 2021. 2
- [8] Joao Carreira and Andrew Zisserman. Quo vadis, action recognition? a new model and the kinetics dataset. In proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 6299–6308, 2017. 2, 6
- [9] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025. 4
- [10] Michael Dorkenwald, Fanyi Xiao, Biagio Brattoli, Joseph Tighe, and Davide Modolo. Scvrl: Shuffled contrastive video representation learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4132–4141, 2022. 2, 3
- [11] Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He. Slowfast networks for video recognition. In Proceedings of the IEEE/CVF international conference on computer vision, pages 6202–6211, 2019. 2, 6
- [12] Robert Geirhos, Patricia Rubisch, Claudio Michaelis, Matthias Bethge, Felix A Wichmann, and Wieland Brendel. Imagenet-trained cnns are biased towards texture; increasing shape bias improves accuracy and robustness. In International conference on learning representations, 2018. 2
- [13] Martin A Giese, Ian M Thornton, and Shimon Edelman. Metric category spaces of biological motion. Journal of Vision, 3(9):83–83, 2003. 1, 3
- [14] Tengda Han, Weidi Xie, and Andrew Zisserman. Selfsupervised co-training for video representation learning. Advances in neural information processing systems, 33:5679– 5690, 2020. 2, 6
- [15] Lianghua Huang, Wei Wang, Zhi-Fan Wu, Yupeng Shi, Huanzhang Dou, Chen Liang, Yutong Feng, Yu Liu, and Jingren Zhou. In-context lora for diffusion transformers. arXiv preprint arXiv:2410.23775, 2024. 4
- [16] Yi Huang, Wei Xiong, He Zhang, Chaoqi Chen, Jianzhuang Liu, Mingfu Yan, and Shifeng Chen. Dive: Taming dino for subject-driven video editing. arXiv preprint arXiv:2412.03347, 2024. 3
- [17] Will Kay, Joao Carreira, Karen Simonyan, Brian Zhang, Chloe Hillier, Sudheendra Vijayanarasimhan, Fabio Viola, Tim Green, Trevor Back, Paul Natsev, et al. The kinetics human action video dataset. arXiv preprint arXiv:1705.06950,

2017. 2, 6

- [18] Hildegard Kuehne, Hueihan Jhuang, Est´ıbaliz Garrote, Tomaso Poggio, and Thomas Serre. Hmdb: a large video database for human motion recognition. In 2011 Inter-

- national conference on computer vision, pages 2556–2563. IEEE, 2011. 2
- [19] Pengyang Ling, Jiazi Bu, Pan Zhang, Xiaoyi Dong, Yuhang Zang, Tong Wu, Huaian Chen, Jiaqi Wang, and Yi Jin. Motionclone: Training-free motion cloning for controllable video generation. arXiv preprint arXiv:2406.05338, 2024. 3
- [20] Yanchen Liu, Yanan Sun, Zhening Xing, Junyao Gao, Kai Chen, and Wenjie Pei. Motionshot: Adaptive motion transfer across arbitrary objects for text-to-video generation. arXiv preprint arXiv:2507.16310, 2025. 3
- [21] Huaishao Luo, Lei Ji, Ming Zhong, Yang Chen, Wen Lei, Nan Duan, and Tianrui Li. Clip4clip: An empirical study of clip for end to end video clip retrieval and captioning. Neurocomputing, 508:293–304, 2022. 2, 6
- [22] Yiwei Ma, Guohai Xu, Xiaoshuai Sun, Ming Yan, Ji Zhang, and Rongrong Ji. X-clip: End-to-end multi-grained contrastive learning for video-text retrieval. In Proceedings of the 30th ACM international conference on multimedia, pages 638–647, 2022. 2, 6
- [23] Joanna Materzynska, Guillaume Berger, Ingo Bax, and Roland Memisevic. The jester dataset: A large-scale video dataset of human gestures. In Proceedings of the IEEE/CVF international conference on computer vision workshops, pages 0–0, 2019. 8
- [24] Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023. 2, 6
- [25] Tian Pan, Yibing Song, Tianyu Yang, Wenhao Jiang, and Wei Liu. Videomoco: Contrastive video representation learning with temporally adversarial examples. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11205–11214, 2021. 2, 3
- [26] Pexels. Pexels: Free stock photos. https://www. pexels.com/, 2025. Accessed: 2025-11-12. 5
- [27] Frank E Pollick, Joshua G Hale, and Phil McAleer. Visual perception of humanoid movement. 2003. 1, 3
- [28] Alexander Pondaven, Aliaksandr Siarohin, Sergey Tulyakov, Philip Torr, and Fabio Pizzati. Video motion transfer with diffusion transformers. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 22911– 22921, 2025. 3
- [29] Rui Qian, Tianjian Meng, Boqing Gong, Ming-Hsuan Yang, Huisheng Wang, Serge Belongie, and Yin Cui. Spatiotemporal contrastive video representation learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6964–6974, 2021. 2, 3
- [30] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021. 2
- [31] Karen Simonyan and Andrew Zisserman. Two-stream convolutional networks for action recognition in videos. Advances in neural information processing systems, 27, 2014. 2

- [32] Khurram Soomro, Amir Roshan Zamir, and Mubarak Shah. Ucf101: A dataset of 101 human actions classes from videos in the wild. arXiv preprint arXiv:1212.0402, 2012. 2
- [33] Steven M Thurman and Hongjing Lu. Revisiting the importance of common body motion in human action perception. Attention, Perception, & Psychophysics, 78(1):30–36, 2016. 1, 3
- [34] Zhan Tong, Yibing Song, Jue Wang, and Limin Wang. Videomae: Masked autoencoders are data-efficient learners for self-supervised video pre-training. Advances in neural information processing systems, 35:10078–10093, 2022. 2, 3, 6
- [35] Narek Tumanyan, Assaf Singer, Shai Bagon, and Tali Dekel. Dino-tracker: Taming dino for self-supervised point tracking in a single video. In European Conference on Computer Vision, pages 367–385. Springer, 2024. 2
- [36] Joris Vangeneugden, Frank Pollick, and Rufin Vogels. Functional differentiation of macaque visual temporal cortical neurons using a parametric action space. Cerebral cortex, 19(3):593–611, 2009. 1, 3
- [37] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, Jianyuan Zeng, Jiayu Wang, Jingfeng Zhang, Jingren Zhou, Jinkai Wang, Jixuan Chen, Kai Zhu, Kang Zhao, Keyu Yan, Lianghua Huang, Mengyang Feng, Ningyi Zhang, Pandeng Li, Pingyu Wu, Ruihang Chu, Ruili Feng, Shiwei Zhang, Siyang Sun, Tao Fang, Tianxing Wang, Tianyi Gui, Tingyu Weng, Tong Shen, Wei Lin, Wei Wang, Wei Wang, Wenmeng Zhou, Wente Wang, Wenting Shen, Wenyuan Yu, Xianzhong Shi, Xiaoming Huang, Xin Xu, Yan Kou, Yangyu Lv, Yifei Li, Yijing Liu, Yiming Wang, Yingya Zhang, Yitong Huang, Yong Li, You Wu, Yu Liu, Yulin Pan, Yun Zheng, Yuntao Hong, Yupeng Shi, Yutong Feng, Zeyinzi Jiang, Zhen Han, Zhi-Fan Wu, and Ziyu Liu. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025. 4
- [38] Fanyi Xiao, Joseph Tighe, and Davide Modolo. Maclr: Motion-aware contrastive learning of representations for videos. In European conference on computer vision, pages 353–370. Springer, 2022. 6
- [39] Zeqi Xiao, Yifan Zhou, Shuai Yang, and Xingang Pan. Video diffusion models are training-free motion interpreter and controller. arXiv preprint arXiv:2405.14864, 2024. 3
- [40] Hu Xu, Gargi Ghosh, Po-Yao Huang, Dmytro Okhonko, Armen Aghajanyan, Florian Metze, Luke Zettlemoyer, and Christoph Feichtenhofer. Videoclip: Contrastive pre-training for zero-shot video-text understanding. arXiv preprint arXiv:2109.14084, 2021. 2
- [41] Danah Yatim, Rafail Fridman, Omer Bar-Tal, Yoni Kasten, and Tali Dekel. Space-time diffusion features for zero-shot text-driven motion transfer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8466–8476, 2024. 3
- [42] Long Zhao, Nitesh B Gundavarapu, Liangzhe Yuan, Hao Zhou, Shen Yan, Jennifer J Sun, Luke Friedman, Rui Qian, Tobias Weyand, Yue Zhao, et al. Videoprism: A foundational visual encoder for video understanding. arXiv preprint arXiv:2402.13217, 2024. 3, 6

