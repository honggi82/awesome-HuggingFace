# arXiv:2510.02283v1[cs.CV]2Oct2025

[Figure 1]

[Figure 2]

## Self-Forcing++: Towards Minute-Scale High-Quality Video Generation

### Justin Cui1,2 Jie Wu2,† Ming Li2,3 Tao Yang2 Xiaojie Li2 Rui Wang2 Andrew Bai1 Yuanhao Ban1 Cho-Jui Hsieh1,‡

1UCLA, 2ByteDance Seed, 3University of Central Florida ‡Corresponding author, †Project lead

### Abstract

Diffusion models have revolutionized image and video generation, achieving unprecedented visual quality. However, their reliance on transformer architectures incurs prohibitively high computational costs, particularly when extending generation to long videos. Recent work has explored autoregressive formulations for long video generation, typically by distilling from short-horizon bidirectional teachers. Nevertheless, given that teacher models cannot synthesize long videos, the extrapolation of student models beyond their training horizon often leads to pronounced quality degradation, arising from the compounding of errors within the continuous latent space. In this paper, we propose a simple yet effective approach to mitigate quality degradation in long-horizon video generation without requiring supervision from long-video teachers or retraining on long video datasets. Our approach centers on exploiting the rich knowledge of teacher models to provide guidance for the student model through sampled segments drawn from self-generated long videos. Our method maintains temporal consistency while scaling video length by up to 20× beyond teacher’s capability, avoiding common issues such as over-exposure and error-accumulation without recomputing overlapping frames like previous methods. When scaling up the computation, our method shows the capability of generating videos up to 4 minutes and 15 seconds, equivalent to 99.9% of the maximum span supported by our base model’s position embedding and more than 50x longer than that of our baseline model. Experiments on standard benchmarks and our proposed improved benchmark demonstrate that our approach substantially outperforms baseline methods in both fidelity and consistency. Our long-horizon videos demo can be found at https://self-forcing-plus-plus.github.io/.

### 1 Introduction

The field of video generation is advancing at a remarkable pace, catalyzed by the advent of diffusion models. Seminal works such as Sora [43], Wan [56], Hunyuan-DiT [29], and Veo [13] are progressively closing the gap between generated content and reality. Despite this progress, a formidable challenge remains: the majority of state-of-the-art models are confined to generating short-form videos, typically capped at 5-10 seconds. This constraint is inherent to the architectural design of the underlying Diffusion Transformers (DiT) [45], the inherently non-streaming and non-causal nature of the vanilla DiT architecture poses a significant challenge to achieving temporal scalability,

A promising avenue for transcending this limitation lies in shifting from bidirectional diffusion architectures to autoregressive, streaming-based models. One such approach, Diffusion Forcing [5, 25], applies heterogeneous

[Figure 3]

Video Horizon Scaling 🎬 0s-60s (First Minute)

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

60s-120s (Second Minute)

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

120s-180s (Third Minute)

[Figure 12]

[Figure 13]

[Figure 14]

180s-240s (Fourth Minute)

[Figure 15]

[Figure 16]

[Figure 17]

- Figure 1 Self-forcing++ generates videos up to four minutes long. The radar chart highlights our model’s superiority, while the line plot shows its sustained motion dynamics over long durations.

noise schedules across frames to enable sequential generation. However, the combinatorial complexity of noise scheduling often leads to training instability and has proven difficult to scale [6, 15]. A more tractable strategy involves predicting the next frame or chunk from a clean context, with KV caching emerging as a key mechanism for enabling performant, real-time streaming. For instance, CausVid [69] proposes a method to distill a bidirectional teacher model into a streaming student model using heterogeneous distillation. However, its reliance on overlapping frames for temporal consistency and a pronounced train-inference mismatch often results in over-exposure artifacts. The Self-Forcing [21] method mitigates the over-exposure issue by aligning the training and inference distributions. While this sets a new benchmark for short-form video quality, its capacity remains bottlenecked by the fixed-duration teacher model. Consequently, when tasked with generating content beyond this intrinsic temporal window (e.g., >10 seconds), the model’s visual quality degrades precipitously.

A primary challenge limiting the quality of autoregressive long-video generation models is a significant training-inference misalignment. This misalignment manifests in two principal ways. First, a temporal mismatch occurs: during training, models generate short clips of up to 5 seconds—the maximum horizon of the teacher model—whereas at inference, they must generate videos of significantly greater length. Second, error accumulation caused by supervision misalignment during long-horizon generation. In training, the teacher model provides abundant supervision for every frame within the short clip. This intensive guidance, however, means the student model is rarely exposed to the compounding errors that naturally arise in long rollouts, leaving it ill-equipped to handle them. As a result, generation quality rapidly deteriorates beyond the 5-second training horizon, often collapsing into static or stalled content.

In this paper, we introduce Self-Forcing++, which directly targets the above two issues. Building upon the observation from previous works [1, 3, 31] that a teacher model, despite its own 5-second generation limit, possesses rich knowledge for correcting errors in quality-degraded videos due to its training on a vast video corpus. We leverage this insight by extending the student’s generation horizon far beyond 5 seconds (up to 100 seconds in our experiments). This process intentionally produces candidate long videos that contain accumulated errors. To enable the student model to handle these errors, we then re-inject noise

into these degraded rollouts and apply distribution-matching distillation with the strong teacher model, a process combined with a long-horizon rolling KV cache and windowed sampling. This strategy teaches the student to recover from degraded states and sustain high-quality, coherent video generation over extended durations. Experimental results demonstrate that our method can scale video generation up to 100 seconds, a

- 20× increase over the baseline, while maintaining high visual quality. By scaling up computation through extended training, our method is capable of generating videos up to 4 minutes and 15 seconds1, utilizing 99.9% of the base model’s positional embedding capacity and representing a 50× improvement over the baseline. Furthermore, our investigation revealed that the widely used VBench [22] benchmark exhibits a bias that favors over-exposed and degraded frames when evaluating long videos, undermining the reliability of its results. To remedy this, we propose a new metric, Visual Stability, designed to systematically capture both quality degradation and over-exposure in long video generation. Our work paves the way for building more robust and reliable long video generation models. Our contributions are summarized as follows:

- • Identifying Horizon Scaling Bottlenecks: We reveal the primary obstacle to extending the generation horizon of autoregressive models: a dual mismatch in temporality and supervision during training versus inference. This insight provides a clear target for overcoming previous limitations on the generation length.

- • A Simple Solution: We propose a simple training framework, named Self-Forcing++. By generating beyond the teacher’s horizon and correcting the student model on its own long, error-accumulated rollout trajectories, Self-Forcing++ extends high-quality video generation to 100 seconds, far surpassing previous state-of-the-art methods without reusing overlapping frames.

- • SOTA Performance and Horizon Scalability: Self-Forcing++ achieves state-of-the-art (SOTA) performance in long-video generation across a range of durations (e.g., 10s, 50s, 100s). Furthermore, we discover a significant scaling property: by scaling the training computation, our model’s generation capability extends to multiple minutes, a feat previously considered out of reach.

### 2 Related Work

Video Diffusion Models Video diffusion models have advanced rapidly, beginning with UNet [50] based approaches that extended image diffusion backbones into the temporal domain [2, 16–18]. These early models enabled short-form video generation but faced limitations in scalability. The introduction of the Diffusion Transformer (DiT)[45] represented a turning point by replacing convolutional hierarchies with transformer blocks, which allowed models to capture global spatio-temporal dependencies more effectively and to scale with larger datasets and computational resources. This shift led to a new wave of architectures, such as Sora[43], which produces realistic, coherent videos with strong temporal consistency and diverse motion, and Hunyuan Video [29], which employs a causal 3D VAE [26] for spatio-temporal token compression in latent space combined with a large language model for text conditioning. Wan 2.1 [56] further demonstrates the benefits of massive pretraining for high-resolution video generation. CogVideoX [19, 66] introduces an expert transformer with adaptive LayerNorm to enhance cross-modal fusion, supported by a 3D VAE, progressive training, and multi-resolution frame packing, thereby achieving strong text alignment and motion coherence. Open-Sora [33, 47] and Open-Sora-Plan [33] extend these advancements in the open-source community, delivering high-quality video generation and significantly accelerating progress in efficiency and realism. Collectively, these works illustrate how scaling strategies and architectural innovations have transformed video diffusion from modest UNet adaptations into transformer-driven models capable of generating controllable and high-quality videos.

Long Video Generation Due to the substantial training and inference cost of DiT-based architectures, most stateof-the-art models remain limited to generating videos of 5–10 seconds. To overcome this constraint, a number of techniques have been introduced to extend generation to longer durations [12, 23, 28, 34]. RIFLEx [71] is a training-free approach that revisits positional encoding, effectively doubling the generation length by avoiding encodings that induce repetitive motion, and surpassing prior methods by a large margin [7, 46, 72]. Another promising direction is autoregressive video generation. Nova [10] reformulates video synthesis as

1The maximum number of latent frames Wan2.1-T2V-1.3B supports is 1024, since we generate videos in a trunk size of 3, the maximum length we can reach is 1023 which is 99.9% of maximally supported length 1024.

a non-quantized autoregressive problem, jointly modeling temporal frame-by-frame prediction and spatial set-by-set prediction, which enables flexible in-context learning. Pyramid-Flow [24] interprets denoising as a hierarchical process across multi-stage pyramids, linking flows across resolutions and time to support end-to-end autoregressive video generation with a single diffusion transformer. SkyReels-V2 adopts diffusion forcing [5] to support potentially infinite rollouts, while MAGI-1 [54] trains a model to progressively denoise per-chunk noise that increases over time, autoregressively predicting fixed-length segments of consecutive frames. CausVid [69] employs block causal attention and a KV cache to autoregressively extend sequences, and Self-Forcing [21] further aligns training with inference by incorporating the KV cache directly during training, producing high-quality short videos.

Reinforcement Learning Reinforcement learning has become a central component in the post-training of large language models [20, 30, 44, 61]. With the rise of image generation, it has also proven effective for improving generative models, with early efforts introducing reward models tailored to images, such as ImageReward [62], Pick-a-Pic [27], and HPS V2 [59]. These concepts have since been extended to video through reward functions like VideoReward [36] and VisionReward [63], which assess temporal coherence and motion quality. Building on these reward signals, optimization techniques first developed for language models, including Direct Preference Optimization (DPO)[49] and Group Relative Policy Optimization (GRPO)[52], have been adapted to diffusionbased generation. Notably, Diffusion-DPO [55] applies preference-based training directly to diffusion models, while Flow-GRPO [35, 64] leverages GRPO to fine-tune video diffusion models, resulting in improvements in both visual fidelity and motion consistency.

### 3 Method

This section details our methodology for long video generation. We begin by revisiting the conversion of bidirectional models into streaming autoregressive generators [21, 69]. Building upon this, we introduce our novel strategies tailored for long-form video synthesis. The complete generative process is formalized in Algorithm 1.

#### 3.1 Background

Video diffusion models, while powerful, typically require denoising along a multi-step noise schedule, which renders the generation process computationally intensive. A prevalent strategy to mitigate this computational burden is to distill the foundational model into a few-step generator. Prominent approaches in this domain include Distribution Matching (DM) [41, 67, 68] and Consistency Models (CM) [53, 57]. Building upon the methodologies of CausVid and Self-Forcing, we distill the original bidirectional teacher model into a few-step generator, then convert it into an autoregressive model. This conversion is accomplished by training a student model to replicate the Ordinary Differential Equation (ODE) trajectories sampled from the teacher. We refer to this procedure as an initialization stage (see section 8.3 for implementation details). The Self-Forcing method extends this approach by training the distilled model on self-generated rollouts of up to five seconds using techniques such as Distribution Matching Distillation (DMD) loss [67]. While this technique effectively mitigates the over-exposure artifacts present in CausVid, it exhibits a critical limitation: a significant degradation in generative quality when producing sequences that exceed its constrained training horizon.

#### 3.2 Extend training beyond teacher’s limit

Motivation As discussed earlier, the teacher model is trained exclusively on five-second video segments. Consequently, distillation-based methods such as CausVid [69] and Self-Forcing [21] only enforce studentteacher distribution alignment within this limited temporal window. This constrained training objective leads to a precipitous decline in quality when generation extends beyond this five-second horizon. Despite this performance collapse, we make a critical observation: videos rolled out beyond the training horizon often retain structural coherence, even if this coherence manifests as undesirable artifacts such as motion stagnation (a common failure mode in Self-Forcing). This suggests that the core problem is not a fundamental breakdown of the autoregressive mechanism, which correctly leverages the history KV cache to maintain context. Rather, the primary issue is the compounding of autoregressive errors during extended rollouts.

###### Real Data Self-Forcing++

###### Self Forcing

###### CausVid

Self-Rollout N s with Rolling KV Cache

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

###### Extended Training Duration

Variable Noises Fixed Training Duration

###### Fixed Training Duration (N=5S) (N >> 5 seconds (e.g. 100s))

Backward Noise Initialization

[Figure 29]

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

Teacher/Student Model (with Block Causal)

Bi-Directional Teacher Model (Fixed Duration)/Autoregressive Student Model (Extended Duration)

|[Figure 40]<br><br>[Figure 41]<br><br>[Figure 42]<br><br>[Figure 43]<br><br>| |[Figure 44]<br><br>|
|---|---|---|

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

DMD Alignment

DMD Alignment

Segment-2 Teacher Outputs Segment-n Teacher Outputs

Segment-1 Teacher Outputs

Student Outputs

Teacher Outputs Extended Student Outputs

Teacher Outputs Student Outputs

[Figure 66]

Training Stage Inference Stage

Extended-DMD Alignment

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

t=0s t=100s

t=0s t=100s t=0s t=100s

Training Horizon Training Horizon

Use Overlapping Frame Autoregressive decoding

All Frames in Training Horizon , Self-Rollout with KV Cache via Autoregressive decoding Training-Inference Misalignment

Self-Rollout Autoregressive decoding

Training-Inference Misalignment

Training-Inference Alignment

- Figure 2 Workflow between baselines and Self-Forcing++. Our method employ backward noise initialization, extended DMD and rolling KV Cache to effectively mitigates train-test discrepancies.

These errors accumulate and eventually manifest as motion loss, scene freezing, and catastrophic degradation of visual fidelity. This insight motivates us to introduce a simple yet effective method to mitigate error accumulation, which is described in the following sections.

Backwards Noise Initialization A central challenge in extending student-teacher distillation to long-horizon video generation resides in the noise initialization strategy. In the short-horizon setting (i.e., for videos with a length up to T frames), the student model can be directly supervised on complete trajectories sampled from the teacher, each originating from random noise. However, for long-horizon generation, a trajectory initialized from pure random noise is decoupled from the preceding video content, leading to a fundamental context misalignment since the sampled noise does not preserve the temporal dependencies of previously generated frames. Based on the observation mentioned above, we add noise back to the denoised latent vectors and use it as the starting noise which is also shown to boost the performance of distillation [67]. While similar techniques of re-injecting noise have been employed in prior work [21, 67, 69], our motivation and application are distinct. Whereas they used this for short-video distillation, primarily to enhance single-shot quality or circumvent the need for real training data. We leverage it as a mechanism to enforce temporal consistency across long videos. Specifically, the student model is first rolled out to a sequence of N clean frames, with N ≫ T, where T denotes the maximum horizon the teacher can reliably generate such as 5 seconds. We then re-inject noise into the student roll-out according to the same diffusion noise schedule {σt}Nt=1. Formally, given the clean trajectory {xSt }Nt=1 generated by the student, the generation is perturbed as:

xt = (1 − σt)x0 + σtϵ, where x0 = xt−1 − σt−1 ϵˆθ(xt−1,t − 1), (1)

where ϵ ∼ N(0,I) denotes Gaussian noise, and ϵˆθ is the noise prediction network parameterized by θ. which serves as the initial state for computing the teacher and student distributions. This approach ensures that the distribution divergence between student and teacher model is evaluated on trajectories that retain temporal consistency and correctly structured according to the prescribed noise schedule.

Extended Distribution Matching Distillation Our strategy for extending training to long videos is grounded in the observation that although the bidirectional teacher model is trained exclusively on short, five-second clips, it implicitly captures the underlying data distribution of the “world” from its training data. From this perspective, any short, contiguous video segment can be viewed as a sample from the marginal distribution of a valid, longer video sequence [1, 3, 31]. This intuition motivates our core methodological extension. Since our baseline method Self-Forcing [21] restricts the training duration to the first T frames (typically ∼5 seconds),

we instruct the student model to roll out to N frames where N ≫ T. We then uniformly sample a contiguous window of length T from the generated sequence, and compute the distributional discrepancy between the student and teacher models within this window. This sliding-window distillation process is formalized as equation (2):

= EtEz ∇θ KL pSθ,t(z) ∥ pTt (z)

∇θL DMD

extended

≈ − Et Ei∼Unif{1,...,N−K+1} sT Φ(Gθ(zi), t), t − sSθ Φ(Gθ(zi), t), t

dGθ(zi) dθ

dzi ,

(2)

Here, Gθ(z) denotes the student generator rollout given a latent z, and Φ(·,t) is the transformation process at timestep t. pSθ,t and pTt represent the student and teacher distributions at time t, with corresponding scores sSθ and sT. We uniformly sample a starting index i ∼ Unif{0,...,N − K} from the student rollout of length N, and extract a window of length K. The student is then trained to minimize the average KL divergence between its distribution and the teacher’s distribution across this window. The window size K is typically chosen to match the horizon to which the teacher model was originally trained to generate.

Remark Bi-directional diffusion can be seen as a process to gradually restore a degraded target in different denoising time-steps. Our method adapts the idea to autoregressive video generation regime by having a short-horizon teacher gradually restore student’s degraded rollouts at different temporal time-frames and then distills these correction knowledge back into the student model.

Training with rolling KV Cache Despite using KV cache at inference time, CausVid [69] still relies on recomputing overlapping frames and suffers from a severe over-exposure problem. Self-Forcing [21] attempts to address this but introduces a train-inference mismatch by using a fixed cache during training and a rolling cache at inference. Although this is partially mitigated by masking the first latent frame, the mismatch still leads to substantial error accumulation and temporal flickering in long videos (see figure 4). In contrast, our method naturally eliminates this mismatch by employing a rolling KV cache during both training and inference. At training time, this cache is used to roll out sequences far beyond the teacher’s supervisory horizon to compute the extended DMD as detailed above. Consequently, our approach greatly simplifies the entire process, requiring neither the recomputation of overlapping frames nor latent frame masking.

#### 3.3 Improving Long-Term Smoothness via GRPO

A common drawback of generative models [40, 60] employing sliding-window or sparse attention mechanisms for long sequences generation is the gradual loss of long-term memory. This degradation often manifests as temporal inconsistencies such as objects abruptly emerging or vanishing or unnaturally rapid scene transitions. Although the method we proposed above has achieved strong results, we show that Group Relative Policy Optimization (GRPO), a reinforcement learning technique [35, 64], can be utilized in autoregressive video

generation framework when such phenomenon presents. The per step importance weight ρt,i = ππθ(at,i |st,i)

θold(at,i | st,i) where πθ(at,i|st,i) denotes the policy function for output oi at time step t can be computed according to equation (1) and the overall generation probability can be computed as the sum of all the log probabilities in current autoregressive rollouts which we show in section 8.4. To guide the optimization process towards temporally smooth outputs, we follow prior work [4, 42] and use the relative magnitude of optical flow between consecutive frames as a proxy for motion continuity.

Algorithm 1 Self-Forcing++ with Backward Noise Initialization (ours)

Require: Student Gθ, teacher Tϕ, cache size L; rollout length N ≫ 5s; slice length K (5s); denoise steps {t1, . . . , tT}

- 1: loop
- 2: V ← Rollout(Gθ, N, L)
- 3: Pick i ∼ {1, . . . , N−K+1}, set W ← V [i : i+K−1] ▷ uniform slice
- 4: Sample t ∼ {t1, . . . , tT}
- 5: Backward noise initialization: xt(W) ← BackwardNoiseInit(W, t)
- 6: LDMD ← DMD(Gθ(xt(W), t), Tϕ(xt(W), t))
- 7: θ ← θ − η∇θLDMD
- 8: end loop
- 9: R ← OpticalFlowReward(Gθ); θ ← GRPO_update(θ, R)

#### 3.4 New metrics for long videos evaluation

Most prior works rely on VBench [22] to assess image and aesthetic quality in long video generation. We find, however, that outdated evaluation models make the benchmark favor over-exposed videos (e.g., CausVid) and degraded long videos (e.g., Self-Forcing), leading to inaccurate scores. To address this, we adopt Gemini2.5-Pro [9], a state-of-the-art video MLLM with strong reasoning ability [8, 38]. Our protocol defines key long-video issues such as over-exposure and error accumulation, prompts Gemini-2.5-Pro to rate videos along these axes, and aggregates the results onto a 0 − 100 scale termed visual stability for consistent comparison. More details are provided in figure 3 and section 8.5.

Aesthetic Quality

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

Normal exposure

Little Degradation

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

ImageQuality

Degradation

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

Exposure

77.13 69.03

59.05 64.01

43.03 50.88

67.29 53.64

Severe Degradation Over Exposure

- Figure 3 The left figure shows the issue of image score issue and the right figure shows the issue of aesthetic score of regular and degraded images from earlier and later frames of the same video. VBench tends to overrate degraded and over-exposed frames rendering these two metrics unreliable.
- 4 Experiments

#### 4.1 Settings

Baseline methods We include the following baseline methods such as NOVA [11], Pyramid Flow [24], SkyReelsV2-1.3B [6], MAGI-1-4.5B [54] distilled to 16 steps for long video generation, CausVid [69] and Self-Forcing [21], both 1.3B distilled few-step generators similar to ours. Additional two state-of-the-art bidirectional models LTX-Video [14] and Wan2.1 [56] are included for references.

Evaluation metrics We conduct evaluations under two primary settings. The first setting follows the general VBench protocol [22], which measures generation quality on short videos of 5 seconds using 946 prompts across 16 dimensions. The second setting examines the model’s capacity to extend generation up to 50/75/100 seconds with the same prompt set used in CausVid, consisting of 128 prompts from MovieGen [48]. Performance in this setting is assessed with both VBench Long and our proposed improved evaluation metric.

#### 4.2 Empirical Results in Long Video Generation

Quantitative and qualitative results are presented in tables 1 and 2, and figures 1 and 4, respectively. Our method achieves competitive performance in short-horizon generation and demonstrates substantial advantages as the generation horizon extends.

Short-Horizon (5s): Although not specifically trained for the initial 5 seconds, our model performs comparably to Self-Forcing on short clips, achieving strong overall results with a semantic score of 80.37 and a total score of 83.11, both surpassing the remaining baselines.

- Table 1 Performance comparisons on 5s short videos and 50s long videos. Baseline methods achieve high temporal quality scores primarily due to stagnation reflected by their dynamic degree.

Model #Params Throughput

Results on 5s ↑ Results on 50s ↑ (FPS) ↑ Total Quality Semantic Text Temporal Dynamic Visual Framewise†

Score Score Score Alignment Quality Degree Stability Quality Bidirectional models

LTX-Video 1.9B 8.98 80.00 82.30 70.79 - - - - Wan2.1 1.3B 0.78 84.67 85.69 80.60 - - - - -

Autoregressive models NOVA 0.6B 0.88 80.12 80.39 79.05 24.58 86.53 31.96 45.94 34.45 Pyramid Flow 2B 6.7 81.72 84.74 69.62 - - - - MAGI-1 4.5B 0.19 79.18 82.04 67.74 26.04 88.34 28.49 51.25 54.20 SkyReels-V2 1.3B 0.49 82.67 84.70 74.53 23.73 88.78 39.15 60.41 54.13 CausVid 1.3B 17.0 82.46 83.61 77.84 25.25 89.34 37.35 40.47 61.56 Self Forcing 1.3B 17.0 83.00 83.71 80.14 24.77 88.17 34.35 40.12 61.06 Ours 1.3B 17.0 83.11 83.79 80.37 26.37 91.03 55.36 90.94 60.82

- indicates that the model either fails to generate videos at the specified length or that the output collapses into random noise. † As discussed in section 3.4, framewise quality is unreliable for long videos, we include it here for reference.

Long-Horizon (50s/75s/100s): The superiority of our method becomes more pronounced in long-horizon generation. We observe consistent improvements across key metrics. E.g. our model achieves a text alignment score of 26.04 and dynamic degree of 54.12 with 100-second video, outperforming CasuVid by 6.67% and 56.4% respectively which relies on recomputing overlapping frames and our baseline method Self-Forcing by 18.36% and 104.9% respectively as shown in figure 4. This suggests that our approach effectively mitigates error accumulation during long rollouts.

- Table 2 Performance comparisons on 75s and 100s long videos. Baseline methods achieve high temporal quality scores primarily due to stagnation or degrade to pure noise.

Results on 75s ↑ Results on 100s ↑

Model

Text Temporal Dynamic Visual Framewise Text Temporal Dynamic Visual Framewise Alignment Quality Degree Stability Quality Alignment Quality Degree Stability Quality

Autoregressive models

NOVA 23.37 86.32 31.24 34.06 31.53 22.89 86.24 31.09 32.97 31.03 MAGI-1 24.95 87.89 24.82 43.28 52.04 23.75 87.62 22.21 39.38 50.90 SkyReels-V2 22.70 88.99 39.89 55.47 51.55 22.05 88.80 38.75 56.72 50.48 CausVid 24.76 89.14 35.82 39.84 60.96 24.41 89.06 34.60 39.21 61.01 Self Forcing 23.39 87.79 29.15 35.00 60.02 22.00 87.39 26.41 32.03 58.25 Ours 26.31 91.00 55.62 86.10 60.67 26.04 90.87 54.12 84.22 60.66

In contrast, baseline methods exhibit significant degradation when generating long videos. Their primary failure modes are: i) Motion Collapse: While maintaining short-term temporal structure, their videos frequently collapse into nearly static sequences, as reflected by their low dynamic degree scores. Our method, however, sustains coherent motion throughout the entire sequence. ii) Fidelity Degradation: Baselines often suffer from exposure instability. For instance, CausVid trends towards over-exposure, while Self-Forcing videos progressively darken. Our model maintains stable brightness and visual quality. This degradation in SelfForcing is a direct consequence of accumulated errors without explicit long-horizon training. While some diffusion forcing methods show sporadic recovery from noise collapse such as SkyReels, the resulting content is of low fidelity.

#### 4.3 Ablation Study

###### 4.3.1 Length of attention window

A straightforward way to mitigate Self-Forcing’s training–inference mismatch is to shorten the attention span during training, exposing the model to more diverse cache states within a limited horizon.

t=0s t=25s t=50s t=75s t=100s

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

SkyReelsMAGI-1CausVidSelfForcingOurs

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

Figure 4 100-second video generated for prompt “A vibrant tropical fish glides gracefully through colorful ocean reefs, surrounded by swaying coral...”. Baseline methods usually suffer from error accumulation and over-exposure, causing severe quality degradation when generating long videos.

For instance, a 5-second clip corresponds to

Table 3 Ablation study on various methods to reduce error accumulation measured by visual stability on 50s videos.

- 21 latent frames; by reducing the attention window, the model is forced to slide attention multiple times. As shown in table 3 and visualized in Appendix figure 7, smaller windows bring modest gains. For example, visual stability improves from 40.12 to 52.50 with a window of 9 latent frames. However, this comes at the cost of increased inconsistency, since the model now relies on much less context compared to the original 21-frame history.

Causvid Self-Forcing Attn-15 Attn-12 Attn-9 Ours 40.47 40.12 44.69 42.19 52.50 90.94

###### 4.3.2 The effect of GRPO with optical-flow reward

Here we show its effectiveness for enhancing temporal consistency by examining the optical flow magnitude, a proxy for temporal stability. As visualized in fig. 5, videos generated without GRPO may suffer from abrupt scene transitions. These transitions manifest as sharp spikes in the optical flow magnitude, an artifact that is exacerbated by the rolling window mechanism used during inference. By promoting smoother temporal transitions, our GRPO method effectively suppresses these spikes. This results in a marked improvement in long-range consistency and overall perceptual quality of the generated videos.

w/o GRPO w GRPO variance

20.0

| | | | | |24.5|2| | | |
|---|---|---|---|---|---|---|---|---|---|
| | |6.89| | | | |2|0.82| |
| | | | | | | | | | |
| | | |2.0|0| | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |

OpticalFlowMagnitude

17.5

15.0

12.5

10.0

7.5

5.0

2.5

0.0

0 25 50 75 100 125 150 175 200

Frame Index

Figure 5 Comparison of generation outcomes with and without GRPO. Variance is computed with window size 8.

#### 4.4 Training Budget Scaling

Finally, we investigate the effect of scaling the training budget on the model’s long-duration video generation capabilities. As illustrated in figure 6, our model, following ODE initialization, exhibits only a nascent ability to generate short, low-fidelity clips. We establish a baseline (1× budget) as the training required to produce a coherent 5-second video. At this scale, extending generation leads to significant temporal flickering and error accumulation, a failure mode similar to that of Self-Forcing [21]. Increasing the budget to 4× enables the model to maintain semantic coherence over longer horizons, successfully rendering a consistent subject like the specified elephant. At 8×, the model begins to generate detailed backgrounds and more semantically accurate subjects, although motion dynamics remain limited and temporal quality degradation persists. A further scaling to 20× yields a substantial improvement, producing high-fidelity videos that remain stable for over 50 seconds. Remarkably, at a 25× budget, the model successfully generates a 255-second video with negligible quality loss. These findings indicate that scaling the training budget is a viable path toward high-quality, long-duration video synthesis, circumventing the reliance on large-scale real video datasets, which are notoriously difficult to acquire.

[Figure 127]

Training Budget Scaling 🔥

1X

4X 8X 20X 25X

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

0-5s

[Figure 135]

🎬VideoHorizonScaling

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

255s

Self Forcing

ODE

Self Forcing++

- Figure 6 Scaling phenomenon observed in 255-second generation for prompt: “A massive elephant walks slowly across a sunlit savannah, dust rising around its feet, the warm glow of sunset...”.

### 5 Conclusion

We introduce Self-Forcing++, a method that mitigates error accumulation in autoregressive long-video generation. By leveraging a short-video teacher to guide the student on its own self-generated long rollouts, our approach learns to correct errors without requiring long-video supervision. Experiments demonstrate that our method significantly extends video length to even over 4 minutes (a 50× improvement over the baseline) while maintaining high fidelity. We also propose a new metric, Visual Stability, to address critical biases in existing long-video evaluation benchmarks. Our contributions pave the way for more robust and scalable long-video synthesis.

### 6 Limitations and Further work

Our method, while effective, inherits certain limitations from its Self-Forcing foundation and the capacity of the underlying Wan2.1-T2V-1.3B model. Key drawbacks include slower training speed compared to teacher-forcing and a lack of long-term memory, which can cause content divergence in regions occluded for extended periods. To address these challenges, we identify several promising future directions. First, to tackle the high training cost of self-rollout, we will explore parallelizing the training process. Second, to further mitigate quality degradation over long sequences, we plan to investigate techniques for controlling the fidelity of latent vectors. This includes quantizing latent representations stored in the KV cache, as suggested by prior works [70], or normalizing the KV cache to prevent distributional shift. Finally, we aim to incorporate

long-term memory mechanisms [32, 39] into our autoregressive framework, which we believe is crucial for achieving true long-range temporal coherence.

### 7 Discussion

We next discuss concurrent works related to ours and highlight their key differences. Rolling Forcing [37] extends the concept of Rolling Diffusion [51] by applying progressively varied noise levels to different video frames. It integrates attention sink frames for balancing short- and long-term consistency, while training efficiency is improved by sampling non-overlapping frames. LongLive [65] builds upon Self Forcing [21], introducing KV re-caching for prompt switching and leveraging clean contexts. It further employs attention sink frames to mitigate error accumulation by repeatedly applying DMD to future frames beyond the teacher’s horizon. Our approach is most closely related to LongLive where we also incorporate DMD into long self-rolled sequences in a windowed fashion with clean context as detailed in section 3.2. Unlike LongLive, however, our simplified design avoids reliance on attention sink frames to counter error accumulation, which was shown to be a key design of LongLive.

Both Rolling Forcing [37] and LongLive [65], as well as our method, are able to generate high-quality videos up to several minutes long, which marks a significant advance in autoregressive long video generation compared to previous methods.

### References

- [1] Eloi Alonso, Adam Jelley, Vincent Micheli, Anssi Kanervisto, Amos J Storkey, Tim Pearce, and François Fleuret. Diffusion for world modeling: Visual details matter in atari. Advances in Neural Information Processing Systems, 37:58757–58791, 2024.

- [2] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023.

- [3] Jake Bruce, Michael D Dennis, Ashley Edwards, Jack Parker-Holder, Yuge Shi, Edward Hughes, Matthew Lai, Aditi Mavalankar, Richie Steigerwald, Chris Apps, et al. Genie: Generative interactive environments. In Forty-first International Conference on Machine Learning, 2024.

- [4] Hila Chefer, Uriel Singer, Amit Zohar, Yuval Kirstain, Adam Polyak, Yaniv Taigman, Lior Wolf, and Shelly Sheynin. Videojam: Joint appearance-motion representations for enhanced motion generation in video models. arXiv preprint arXiv:2502.02492, 2025.

- [5] Boyuan Chen, Diego Martí Monsó, Yilun Du, Max Simchowitz, Russ Tedrake, and Vincent Sitzmann. Diffusion forcing: Next-token prediction meets full-sequence diffusion. Advances in Neural Information Processing Systems, 37:24081–24125, 2024.

- [6] Guibin Chen, Dixuan Lin, Jiangping Yang, Chunze Lin, Junchen Zhu, Mingyuan Fan, Hao Zhang, Sheng Chen, Zheng Chen, Chengcheng Ma, et al. Skyreels-v2: Infinite-length film generative model. arXiv preprint arXiv:2504.13074, 2025.

- [7] Shouyuan Chen, Sherman Wong, Liangjian Chen, and Yuandong Tian. Extending context window of large language models via positional interpolation. arXiv preprint arXiv:2306.15595, 2023.

- [8] Wei-Lin Chiang, Anastasios Angelopoulos, L Zheng, Y Sheng, L Dunlap, C Chou, T Li, E Frick, N Jain, D Li, et al. Chatbot arena. LMArena, https://lmarena. ai, 2024.

- [9] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.

- [10] Haoge Deng, Ting Pan, Haiwen Diao, Zhengxiong Luo, Yufeng Cui, Huchuan Lu, Shiguang Shan, Yonggang Qi, and Xinlong Wang. Autoregressive video generation without vector quantization. arXiv preprint arXiv:2412.14169, 2024.

- [11] Haoge Deng, Ting Pan, Haiwen Diao, Zhengxiong Luo, Yufeng Cui, Huchuan Lu, Shiguang Shan, Yonggang Qi, and Xinlong Wang. Autoregressive video generation without vector quantization. In ICLR, 2025.

- [12] Xueji Fang, Liyuan Ma, Zhiyang Chen, Mingyuan Zhou, and Guo-jun Qi. Inflvg: Reinforce inference-time consistent long video generation with grpo. arXiv preprint arXiv:2505.17574, 2025.

- [13] Google DeepMind. Veo. https://deepmind.google/models/veo/, 2025. Accessed: 2025-09-09.
- [14] Yoav HaCohen, Nisan Chiprut, Benny Brazowski, Daniel Shalem, Dudu Moshe, Eitan Richardson, Eran Levin, Guy Shiran, Nir Zabari, Ori Gordon, et al. Ltx-video: Realtime video latent diffusion. arXiv preprint arXiv:2501.00103, 2024.

- [15] Xianglong He, Chunli Peng, Zexiang Liu, Boyang Wang, Yifan Zhang, Qi Cui, Fei Kang, Biao Jiang, Mengyin An, Yangyang Ren, et al. Matrix-game 2.0: An open-source, real-time, and streaming interactive world model. arXiv preprint arXiv:2508.13009, 2025.

- [16] Yingqing He, Tianyu Yang, Yong Zhang, Ying Shan, and Qifeng Chen. Latent video diffusion models for high-fidelity long video generation. arXiv preprint arXiv:2211.13221, 2022.

- [17] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, et al. Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303, 2022.

- [18] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video diffusion models. Advances in neural information processing systems, 35:8633–8646, 2022.

- [19] Wenyi Hong, Ming Ding, Wendi Zheng, Xinghan Liu, and Jie Tang. Cogvideo: Large-scale pretraining for text-to-video generation via transformers. arXiv preprint arXiv:2205.15868, 2022.

- [20] Zhenyu Hou, Xin Lv, Rui Lu, Jiajie Zhang, Yujiang Li, Zijun Yao, Juanzi Li, Jie Tang, and Yuxiao Dong. Advancing language model reasoning through reinforcement learning and inference scaling. arXiv preprint arXiv:2501.11651, 2025.

- [21] Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, and Eli Shechtman. Self forcing: Bridging the train-test gap in autoregressive video diffusion. arXiv preprint arXiv:2506.08009, 2025.

- [22] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21807–21818, 2024.

- [23] Jiaxiu Jiang, Wenbo Li, Jingjing Ren, Yuping Qiu, Yong Guo, Xiaogang Xu, Han Wu, and Wangmeng Zuo. Lovic: Efficient long video generation with context compression. arXiv preprint arXiv:2507.12952, 2025.

- [24] Yang Jin, Zhicheng Sun, Ningyuan Li, Kun Xu, Hao Jiang, Nan Zhuang, Quzhe Huang, Yang Song, Yadong Mu, and Zhouchen Lin. Pyramidal flow matching for efficient video generative modeling. In ICLR, 2025.

- [25] Jihwan Kim, Junoh Kang, Jinyoung Choi, and Bohyung Han. Fifo-diffusion: Generating infinite videos from text without training. Advances in Neural Information Processing Systems, 37:89834–89868, 2024.

- [26] Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013.

- [27] Yuval Kirstain, Adam Polyak, Uriel Singer, Shahbuland Matiana, Joe Penna, and Omer Levy. Pick-a-pic: An open dataset of user preferences for text-to-image generation. Advances in neural information processing systems, 36:36652–36663, 2023.

- [28] Akio Kodaira, Tingbo Hou, Ji Hou, Masayoshi Tomizuka, and Yue Zhao. Streamdit: Real-time streaming text-to-video generation. arXiv preprint arXiv:2507.03745, 2025.

- [29] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024.

- [30] Aviral Kumar, Vincent Zhuang, Rishabh Agarwal, Yi Su, John D Co-Reyes, Avi Singh, Kate Baumli, Shariq Iqbal, Colton Bishop, Rebecca Roelofs, et al. Training language models to self-correct via reinforcement learning. arXiv preprint arXiv:2409.12917, 2024.

- [31] Jiaqi Li, Junshu Tang, Zhiyong Xu, Longhuang Wu, Yuan Zhou, Shuai Shao, Tianbao Yu, Zhiguo Cao, and Qinglin Lu. Hunyuan-gamecraft: High-dynamic interactive game video generation with hybrid history condition. arXiv preprint arXiv:2506.17201, 2025.

- [32] Jiaqi Li, Junshu Tang, Zhiyong Xu, Longhuang Wu, Yuan Zhou, Shuai Shao, Tianbao Yu, Zhiguo Cao, and Qinglin Lu. Hunyuan-gamecraft: High-dynamic interactive game video generation with hybrid history condition,

2025. URL https://arxiv.org/abs/2506.17201.

- [33] Bin Lin, Yunyang Ge, Xinhua Cheng, Zongjian Li, Bin Zhu, Shaodong Wang, Xianyi He, Yang Ye, Shenghai Yuan, Liuhan Chen, et al. Open-sora plan: Open-source large video generation model. arXiv preprint arXiv:2412.00131, 2024.

- [34] Shanchuan Lin, Ceyuan Yang, Hao He, Jianwen Jiang, Yuxi Ren, Xin Xia, Yang Zhao, Xuefeng Xiao, and Lu Jiang. Autoregressive adversarial post-training for real-time interactive video generation. arXiv preprint arXiv:2506.09350, 2025.

- [35] Jie Liu, Gongye Liu, Jiajun Liang, Yangguang Li, Jiaheng Liu, Xintao Wang, Pengfei Wan, Di Zhang, and Wanli Ouyang. Flow-grpo: Training flow matching models via online rl. arXiv preprint arXiv:2505.05470, 2025.

- [36] Jie Liu, Gongye Liu, Jiajun Liang, Ziyang Yuan, Xiaokun Liu, Mingwu Zheng, Xiele Wu, Qiulin Wang, Wenyu Qin, Menghan Xia, et al. Improving video generation with human feedback. arXiv preprint arXiv:2501.13918, 2025.

- [37] Kunhao Liu, Wenbo Hu, Jiale Xu, Ying Shan, and Shijian Lu. Rolling forcing: Autoregressive long video diffusion in real time, 2025. URL https://arxiv.org/abs/2509.25161.

- [38] Yuanxin Liu, Kun Ouyang, Haoning Wu, Yi Liu, Lin Sui, Xinhao Li, Yan Zhong, Y Charles, Xinyu Zhou, and Xu Sun. Videoreasonbench: Can mllms perform vision-centric complex video reasoning? arXiv preprint arXiv:2505.23359, 2025.

- [39] Zhiheng Liu, Xueqing Deng, Shoufa Chen, Angtian Wang, Qiushan Guo, Mingfei Han, Zeyue Xue, Mengzhao Chen, Ping Luo, and Linjie Yang. Worldweaver: Generating long-horizon video worlds via rich perception. arXiv preprint arXiv:2508.15720, 2025.

- [40] Yang Luo, Xuanlei Zhao, Mengzhao Chen, Kaipeng Zhang, Wenqi Shao, Kai Wang, Zhangyang Wang, and Yang You. Enhance-a-video: Better generated video for free. arXiv preprint arXiv:2502.07508, 2025.

- [41] Yihong Luo, Tianyang Hu, Jiacheng Sun, Yujun Cai, and Jing Tang. Learning few-step diffusion models by trajectory distribution matching. arXiv preprint arXiv:2503.06674, 2025.

- [42] Hyelin Nam, Jaemin Kim, Dohun Lee, and Jong Chul Ye. Optical-flow guided prompt optimization for coherent video generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 7837–7846, 2025.

- [43] OpenAI. Video generation models as world simulators. Technical report, OpenAI, February 2024. URL https://openai.com/index/video-generation-models-as-world-simulators/. Technical report.
- [44] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744, 2022.

- [45] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205, 2023.

- [46] Bowen Peng, Jeffrey Quesnelle, Honglu Fan, and Enrico Shippole. Yarn: Efficient context window extension of large language models. arXiv preprint arXiv:2309.00071, 2023.

- [47] Xiangyu Peng, Zangwei Zheng, Chenhui Shen, Tom Young, Xinying Guo, Binluo Wang, Hang Xu, Hongxin Liu, Mingyan Jiang, Wenjun Li, et al. Open-sora 2.0: Training a commercial-level video generation model in 200 k. arXiv preprint arXiv:2503.09642, 2025.

- [48] Adam Polyak, Amit Zohar, Andrew Brown, Andros Tjandra, Animesh Sinha, Ann Lee, Apoorv Vyas, Bowen Shi, Chih-Yao Ma, Ching-Yao Chuang, et al. Movie gen: A cast of media foundation models. arXiv preprint arXiv:2410.13720, 2024.

- [49] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in neural information processing systems, 36:53728–53741, 2023.

- [50] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In International Conference on Medical image computing and computer-assisted intervention, pages 234–241. Springer, 2015.

- [51] David Ruhe, Jonathan Heek, Tim Salimans, and Emiel Hoogeboom. Rolling diffusion models, 2024. URL https://arxiv.org/abs/2402.09470.
- [52] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

- [53] Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models. 2023.
- [54] Hansi Teng, Hongyu Jia, Lei Sun, Lingzhi Li, Maolin Li, Mingqiu Tang, Shuai Han, Tianning Zhang, WQ Zhang, Weifeng Luo, et al. Magi-1: Autoregressive video generation at scale. arXiv preprint arXiv:2505.13211, 2025.

- [55] Bram Wallace, Meihua Dang, Rafael Rafailov, Linqi Zhou, Aaron Lou, Senthil Purushwalkam, Stefano Ermon, Caiming Xiong, Shafiq Joty, and Nikhil Naik. Diffusion model alignment using direct preference optimization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8228–8238, 2024.

- [56] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, Jianyuan Zeng, Jiayu Wang, Jingfeng Zhang, Jingren Zhou, Jinkai Wang, Jixuan Chen, Kai Zhu, Kang Zhao, Keyu Yan, Lianghua Huang, Mengyang Feng, Ningyi Zhang, Pandeng Li, Pingyu Wu, Ruihang

- Chu, Ruili Feng, Shiwei Zhang, Siyang Sun, Tao Fang, Tianxing Wang, Tianyi Gui, Tingyu Weng, Tong Shen, Wei Lin, Wei Wang, Wei Wang, Wenmeng Zhou, Wente Wang, Wenting Shen, Wenyuan Yu, Xianzhong Shi, Xiaoming Huang, Xin Xu, Yan Kou, Yangyu Lv, Yifei Li, Yijing Liu, Yiming Wang, Yingya Zhang, Yitong Huang, Yong Li, You Wu, Yu Liu, Yulin Pan, Yun Zheng, Yuntao Hong, Yupeng Shi, Yutong Feng, Zeyinzi Jiang, Zhen Han, Zhi-Fan Wu, and Ziyu Liu. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.
- [57] Fu-Yun Wang, Zhaoyang Huang, Alexander Bergman, Dazhong Shen, Peng Gao, Michael Lingelbach, Keqiang Sun, Weikang Bian, Guanglu Song, Yu Liu, et al. Phased consistency models. Advances in neural information processing systems, 37:83951–84009, 2024.

- [58] Wenhao Wang and Yi Yang. Vidprom: A million-scale real prompt-gallery dataset for text-to-video diffusion models. Advances in Neural Information Processing Systems, 37:65618–65642, 2024.

- [59] Xiaoshi Wu, Yiming Hao, Keqiang Sun, Yixiong Chen, Feng Zhu, Rui Zhao, and Hongsheng Li. Human preference score v2: A solid benchmark for evaluating human preferences of text-to-image synthesis. arXiv preprint arXiv:2306.09341, 2023.

- [60] Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. Efficient streaming language models with attention sinks. arXiv preprint arXiv:2309.17453, 2023.

- [61] Zhihui Xie, Liyu Chen, Weichao Mao, Jingjing Xu, Lingpeng Kong, et al. Teaching language models to critique via reinforcement learning. arXiv preprint arXiv:2502.03492, 2025.

- [62] Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong. Imagereward: learning and evaluating human preferences for text-to-image generation. In Proceedings of the 37th International Conference on Neural Information Processing Systems, pages 15903–15935, 2023.

- [63] Jiazheng Xu, Yu Huang, Jiale Cheng, Yuanming Yang, Jiajun Xu, Yuan Wang, Wenbo Duan, Shen Yang, Qunlin Jin, Shurun Li, Jiayan Teng, Zhuoyi Yang, Wendi Zheng, Xiao Liu, Ming Ding, Xiaohan Zhang, Xiaotao Gu, Shiyu Huang, Minlie Huang, Jie Tang, and Yuxiao Dong. Visionreward: Fine-grained multi-dimensional human preference learning for image and video generation, 2024. URL https://arxiv.org/abs/2412.21059.
- [64] Zeyue Xue, Jie Wu, Yu Gao, Fangyuan Kong, Lingting Zhu, Mengzhao Chen, Zhiheng Liu, Wei Liu, Qiushan Guo, Weilin Huang, et al. Dancegrpo: Unleashing grpo on visual generation. arXiv preprint arXiv:2505.07818, 2025.

- [65] Shuai Yang, Wei Huang, Ruihang Chu, Yicheng Xiao, Yuyang Zhao, Xianbang Wang, Muyang Li, Enze Xie, Yingcong Chen, Yao Lu, and Song Hanand Yukang Chen. Longlive: Real-time interactive long video generation. 2025.
- [66] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024.

- [67] Tianwei Yin, Michaël Gharbi, Taesung Park, Richard Zhang, Eli Shechtman, Fredo Durand, and Bill Freeman. Improved distribution matching distillation for fast image synthesis. Advances in neural information processing systems, 37:47455–47487, 2024.

- [68] Tianwei Yin, Michaël Gharbi, Richard Zhang, Eli Shechtman, Fredo Durand, William T Freeman, and Taesung Park. One-step diffusion with distribution matching distillation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6613–6623, 2024.

- [69] Tianwei Yin, Qiang Zhang, Richard Zhang, William T Freeman, Fredo Durand, Eli Shechtman, and Xun Huang. From slow bidirectional to fast autoregressive video diffusion models. In CVPR, 2025.

- [70] Lvmin Zhang and Maneesh Agrawala. Packing input frame context in next-frame prediction models for video generation. arXiv preprint arXiv:2504.12626, 2025.

- [71] Min Zhao, Guande He, Yixiao Chen, Hongzhou Zhu, Chongxuan Li, and Jun Zhu. Riflex: A free lunch for length extrapolation in video diffusion transformers. arXiv preprint arXiv:2502.15894, 2025.

- [72] Le Zhuo, Ruoyi Du, Han Xiao, Yangguang Li, Dongyang Liu, Rongjie Huang, Wenze Liu, Xiangyang Zhu, Fu-Yun Wang, Zhanyu Ma, et al. Lumina-next: Making lumina-t2x stronger and faster with next-dit. Advances in Neural Information Processing Systems, 37:131278–131315, 2024.

### 8 Appendix

#### 8.1 Detailed evaluation results for all dimensions

Due to limited space, we only report VBench aggregated metrics in tables 1 and 2 in the main text. Here we show the full evaluated data for each dimension in table 4.

Table 4 Comparison of models across multiple quality metrics on 50s, 75s, and 100s videos for our main results. The gray metrics are aggregated metrics by VBench Long.

text alignment

overall consistency

clip score

temporal quality

subject consistency

imaging quality 50 seconds

background consistency

motion smoothness

dynamic degree

frame-wise quality

aesthetic quality

NOVA 24.58 20.55 28.61 86.53 92.48 95.21 99.18 31.96 34.45 39.85 29.06 MAGI-1 26.04 21.55 30.53 88.34 98.08 97.50 99.36 28.49 54.20 52.15 56.26

SkyReels-V2 23.73 18.94 28.53 88.78 96.06 96.43 98.67 39.15 54.13 50.44 57.82

CausVid 25.25 20.71 29.80 89.34 98.29 97.27 98.45 37.35 61.56 57.91 65.21 Self-Forcing 24.77 20.27 29.27 88.17 96.77 96.24 98.40 34.35 61.06 54.95 67.17

Ours 26.37 22.03 30.71 91.03 97.00 95.55 98.39 55.36 60.82 53.76 67.87 75 seconds

NOVA 23.37 19.19 27.54 86.32 91.78 95.57 99.14 31.24 31.53 37.34 25.73 MAGI-1 24.95 20.36 29.53 87.89 98.20 97.70 99.29 24.82 52.04 49.38 54.71

SkyReels-V2 22.70 17.60 27.81 88.99 96.08 96.54 98.88 39.89 51.55 47.55 55.55

CausVid 24.76 19.99 29.53 89.14 98.32 97.28 98.50 35.82 60.96 56.87 65.06 Self-Forcing 23.39 18.85 27.93 87.79 97.43 96.50 98.77 29.15 60.02 53.28 66.77

Ours 26.31 22.09 30.53 91.00 96.93 95.45 98.29 55.62 60.67 53.38 67.97 100 seconds

NOVA 22.89 18.63 27.15 86.24 91.66 95.50 99.13 31.09 31.03 36.64 25.42 MAGI-1 23.75 18.94 28.57 87.62 98.35 97.99 99.20 22.21 50.90 47.25 54.55

SkyReels-V2 22.05 16.84 27.25 88.80 96.05 96.52 98.86 38.75 50.48 46.33 54.62

CausVid 24.41 19.61 29.22 89.06 98.41 97.46 98.54 34.60 61.01 57.22 64.79 Self-Forcing 22.00 17.16 26.84 87.39 97.39 96.76 98.52 26.41 58.25 51.16 65.35

Ours 26.04 21.75 30.34 90.87 97.09 95.53 98.35 54.12 60.66 53.00 68.31

Implementation details We adopt the same base model Wan2.1-T2V-1.3B [56] as Causvid and Self-Forcing, which is later converted into an autoregressive model as describe above. The model is initialized with sampled 16K ODE training trajectories by optimizing the loss in equation (4). We use the same filtered and LLM-extended version of VidProM [58] as Self-Forcing for training. In the training phase, since we utilize backward noise initialization, thus we don’t need real data for training. We utilize the same Wan2.1-T2V-1.3B as the teacher model.

Training details Self-Forcing++ is trained with a training batch size of 8. The hyperparameters are mostly adopted from Self-Forcing such as the denoising steps of 1000,750,500 and 250 with a generator learning rate of 2e−6 and critic learning rate of 4e−7. The generator and critic update ratio is 5. AdamW optimizer is used for both generator and critic both with β1 = 0 and β2 = 0.999. Our rolling KV cache window size is 21 latent frames in all cases except ablation study. The model is updated with EMA starting at 200 epochs. We have also inspected the version without EMA which can also generate long high quality videos but the EMA version performs better. Our method can already generate consistent high quality long videos such as videos up to 4minute 15 seconds before GPRO, in the ablation study, we show that it’s possible to further boost the model’s performance with properly designed rewards.

#### 8.2 Adding noise to context window

As demonstrated in table 1, methods such as MAGI-1 [54] and SkyReels-V2 [6], which rely on variable noisy context injection following a predefined schedule [5], are insufficient to mitigate error accumulation when rolling to long videos. To further investigate its effect on methods, we conduct an experiment where noise is manually injected into the KV cache to explicitly simulate the effect of accumulated errors over extended sequences. Specifically, before adding any query or key to the KV cache, we inject random Gaussian noise into them. While this strategy yields a slight improvement in both image quality and visual stability compare to the original Self-Forcing, it nonetheless fails to prevent substantial degradation in long-horizon video generation which can be seen in figure 7.

t=0s t=10s t=20s t=30s t=50s

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

Attn-9Attn-12Attn-15Noisy-KV

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

- Figure 7 Ablation study for various methods of mitigating error accumulation. Here is one visualization of generated 50-second video for prompt: “Drone view of waves crashing against the rugged cliffs along Big Sur’s garay point beach...”

#### 8.3 More Background

Video diffusion models are typically trained to denoise along a fixed noise schedule, which makes video generation computationally expensive. A common strategy to reduce this cost is to distill the model into a few-step generator, using approaches such as Distribution Matching [41, 67, 68] and Consistency Model [53, 57]. In line with CausVid and Self-Forcing, we also adopt DMD to distill the original bidirectional model into a few-step model, which can be viewed as minimizing the reverse KL divergence between the student and teacher models, as formulated in equation (3).

∇θLDMD = Et ∇θ KL(pfake,t ∥preal,t)

= −Et sreal(Φ(Gθ(z),t),t) − sfake(Φ(Gθ(z),t),t)

dGθ(z) dθ

dz .

(3)

After the model is distilled into few-step form, it is converted into an autoregressive model by introducing causal attention. The conversion is carried out by sampling ODE trajectories from the teacher and training the autoregressive model on these trajectories. This stage functions as a warm-up phase, distinct from the main training procedure described below. The ODE training process is formally expressed in equation (4).

Lode = Ex,t Gϕ {x(ti)

}Ni=1, {ti}Ni=1 − {x(teacheri) }Ni=1

i

2

. (4)

##### 8.4 Improving Long-Term Smoothness via GRPO Following the discussion in the main text, here we show the general form of GRPO which can be written as:

i}Gi=1∼πθold(·|c)Eat,i∼πθ

J (θ) = E{o

old(·|st,i)

G

T

1 G

1 T

min ρt,iAi, clip(ρt,i, 1 − ϵ, 1 + ϵ)Ai , (5)

t=1

i=1

where ρt,i = ππθ(at,i |st,i)

θold(at,i |st,i) is the importance weight, πθ(at,i|st,i) denotes the policy function for output oi at time step t whose value can be computed according to equation (1), ϵ is a clipping hyper-parameter, and

Ai is the advantage computed across a generation group. The advantage is computed across a group of G outputs as:

ri − mean({r1,r2,...,rG}) std({r1,r2,...,rG})

. (6)

Ai =

In order to generate adopt our model for GRPO, we inject Gaussian noise at each non-terminal step according to the noise scheduler. The probability of the final generated video can be formulated as below.

log p(x1:N) =

=

N

log p(xn | x<n)

n=1

x(t,in) − (1 − σt)x(0n,i) 2 2σt2 − log σt − 12 log(2π) ,

N

T

D

−

n=1

t=1

i=1

(7)

where x(0n,i) is computed following equation (1) conditioned on the previously generated samples x<n, D denotes the latent dimension size, T is the number of non-terminal sampling steps, and N is the total number

of autoregressive steps.

###### 8.4.1 Temporal Repetition

As highlighted in RIFLEx [71], one of the primary challenges in long video generation is temporal repetition, where videos begin to cycle with fixed, recurring patterns. In contrast, autoregressive approaches such as Self-Forcing [21] and our method, which rely exclusively on the KV cache to produce new frames, are less prone to this failure mode. To quantify this, we adopt the NoRepeat Score introduced in RIFLEx and report the results in table 5.

From table 5, we observe that NOVA, MAGI-1 and CausVid are more susceptible to repeated temporal patterns when extended to long videos. In contrast, methods that generate solely from the KV cache such as Self-Forcing and ours achieve stronger resistance to temporal repetition without requiring recomputation or overlapping frames.

Table 5 NoRepeat scores (↑) across different methods, computed following RIFLEx. The RIFLEx score reported here corresponds to its best published result and serves only as a reference.

RIFLEx Nova MAGI-1 SkyReels-V2 CausVid Self-Forcing Ours 89.0 67.19 73.44 95.31 92.97 100.0 98.44

#### 8.5 Evaluation with gemini-2.5-pro and manually verification

We present several representative results with Gemini-2.5-Pro on 50-second videos generated by our method and by baseline methods, as shown in figures 8 and 9. None of the baselines sustain high quality at this length, and each displays distinct failure modes. CausVid consistently shows pronounced over-exposure even within its trained 5-second horizon, which worsens as the video progresses until motion collapses entirely. Self-Forcing suffers from severe error accumulation, leading to global darkening and stagnation. MAGI-1 initially avoids over-exposure, likely due to its reliance on diffusion forcing, but rapidly deteriorates into heavy over-exposure and structural collapse. SkyReels-V2, as seen in figure 9, generally preserves structure but exhibits moderate to severe over-exposure, resembling CausVid’s failure pattern.

As further illustrated in figures 8 and 9 and on our project page self-forcing-plus-plus.github.io , all baselines demonstrate systematic breakdown in long-video generation that state-of-the-art MLLMs readily detect. To ensure alignment with human judgment, we conducted manual verification: 20 randomly sampled MovieGen videos were independently annotated by two authors, and the averaged scores were compared with Gemini2.5-Pro. For 50-second sequences, Spearman’s rank correlation reached 100% for the top three methods and 94.2% across all six baselines. Similar results are observed for the 75-second and 100-second videos, where the generation quality of baseline methods further declines.

Overall, our method achieves sustained long-term visual stability. Methods trained with diffusion forcing, such as SkyReels-V2 and MAGI-1, rank next, followed by CausVid, which maintains structure only under severe exposure. Both Self-Forcing and NOVA degrade to comparable low levels.

CausVid Self Forcing

Ours

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

Rating: 2 – Noticeable Exposure Problems Reason: The large celestial body in the

Rating: 2 – Noticeable Exposure Problems Reason: The video consistently displays

Rating: 5/5 – Well-Exposed Reason: The video is well-exposed, demonstrating excellent handling of a highcontrast scene. The deep blacks of the space background and the harsh shadows on the lunar surface are appropriate for the setting and do not represent crushed blacks or loss of detail. Detail is well-preserved throughout the tonal range

background is significantly overexposed, with large areas blown out to pure white, causing a complete loss of surface detail.

significant exposure issues, primarily with high contrast that leads to a loss of detail.

While the scene remains perfectly understandable, these "clear and persistent clipping in highlights or shadows" affect major portions of the frame (the main subject and its shadow), which aligns perfectly with the definition for a rating of 2.

The highlights on the astronaut's suit are also very bright and appear clipped.

These exposure issues are not fleeting; they are constant throughout the entire video.

- Figure 8 Example evaluation using Gemini-2.5-pro on the results generated by our method, CausVid and Self-Forcing for prompt “An astronaut runs on the surface of the moon, the low angle shot shows the vast background of the moon, the movement is smooth and appears lightweight.”. Gemini-2.5-Pro is tasked to rate the whole video with thinking and output reasoning first before outputting the final rating.

Reason: The video maintains a balanced and consistent exposure throughout. There are no noticeable areas where highlights are blown out to pure white or where shadows

are crushed to pure black. All the details in the colorful fish, coral, and background are clearly visible, resulting in a clean and wellpresented image.

Rating: 5/5 – Well-Exposed

Ours

Rating:1 – Severe Exposure Failure Reason: While the video begins with a

reasonably well-exposed image (0:00-0:09), it undergoes an artistic transition where the exposure rapidly and deliberately fails. From

approximately the 15-second mark onward, large portions of the foreground coral and anemones become completely blown out to pure white. This over-exposure dominates the lower half of the frame for the majority of the video's runtime, severely distorting visibility and losing all detail in those areas.

Rating: 1 – Severe Exposure Failure

Reason: The video begins well-exposed for the first 8-10 seconds, showing a clear and balanced underwater scene. However, after approximately the 15-second mark, the

image quality degrades catastrophically. The latter two-thirds of the video is dominated by extreme over-exposure. Large areas of the frame become a flat, blown-out cyan.

SkyReelsV2 MAGI-1

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

- Figure 9 Example evaluation using Gemini-2.5-pro on the results generated by our method, CausVid and Self-Forcing for prompt “A gorgeously rendered papercraft world of a coral reef, rife with colorful fish and sea creatures.”. Gemini2.5-Pro is tasked to rate the whole video with thinking and output reasoning first before outputting the final rating.

The prompt we use for evaluating is as following:

You are tasked with rating the exposure stability of a video. Assign a score according to the following scale:

- 0: Catastrophic Exposure. Nearly the entire frame is either blown out (pure white) or crushed (pure black), rendering the scene unreadable.
- 1: Severe Exposure Failure. Large portions of the frame are dominated by over-exposure or under-exposure, substantially impairing visibility.
- 2: Noticeable Exposure Problems. Persistent clipping is present in highlights or shadows. Significant areas lose detail, though the frame remains viewable.
- 3: Moderate Exposure Issues. Over-exposed highlights or under-exposed shadows occur but are limited in extent or duration.
- 4: Minor Exposure Flaws. Small regions are occasionally too bright or too dark, but these do not meaningfully disrupt overall visibility.
- 5: Well-Exposed. Balanced lighting across the frame. No distracting over-exposure or darkening; both highlights and shadows retain detail.

Do not claim that the observations in any video are of a specific artistic style or scene transitions unless the prompt explicitly states so. The prompt for generating the video is as follows:

First, provide a brief explanation of your reasoning, describing the observed exposure characteristics. Then, state your final score according to the scale.

#### 8.6 Discussion of Diffusion Forcing vs Autoregressive

Both diffusion forcing such as SkyReels and MAGI and autoregressive models with clean context such as Self-Forcing and ours can generate long videos. Diffusion forcing works by keeping a large number of frames in the current stage and apply different noise level for different frames. Thus, it naturally comes with better long term memory. However, such as long term memory comes at the cost of training instability as the number of different noise level combinations can be extremely huge due to the nature of diffusion models which needs multiple step denoising. Thus methods such as StreamDiT [28] has opted to distill the model first to limit the number of combinations which reduces the tranining instability. However, as the resutls shown in our work tables 1, 2 and 4 that a context with variable noises it not absolutely required to achieve long horizon generation with little quality degradation. As shown in our ablation study, the model is gradually learns to generate long videos with increased training budget even without using long video training set. We hope our work can help the community with generating better and more consistent long videos.

##### 8.7 More Visualizations Please checkout our demo page for more videos at https://self-forcing-plus-plus.github.io/

