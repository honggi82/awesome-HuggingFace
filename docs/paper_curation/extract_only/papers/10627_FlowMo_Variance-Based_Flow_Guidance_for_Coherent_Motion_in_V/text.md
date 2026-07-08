arXiv:2506.01144v2[cs.CV]4Jun2025

# FlowMo: Variance-Based Flow Guidance for Coherent Motion in Video Generation

Ariel Shaulov∗ Itay Hazan∗ Lior Wolf Hila Chefer School of Computer Science Tel Aviv University, Israel

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

Wan2.1-1.3B+FlowMo

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

“A woman performing a challenging exercise”

“A dolphin jumping out of ocean waves”

- (a)
- (b)

[Figure 11]

[Figure 12]

CogVideoX-5B+FlowMo

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

“A pair of flamingos wading through shallow water”

“A ballerina leaping through the air”

Figure 1: Text-to-video results before and after applying FlowMo on (a) Wan2.1 [1] and CogVideoX-5B [2]. We present FlowMo, an inference-time guidance method to enhance temporal coherence in text-to-video models. Our method mitigates severe temporal artifacts, such as additional limbs (woman, 1st row, 2nd row), objects that appear or disappear (flamingo, 2nd row), and object distortions (woman, dolphin, 1st row), without requiring additional training or conditioning signals.

## Abstract

Text-to-video diffusion models are notoriously limited in their ability to model temporal aspects such as motion, physics, and dynamic interactions. Existing approaches address this limitation by retraining the model or introducing external

∗Equal contribution. Project page: https://arielshaulov.github.io/FlowMo/ Correspondence to: Ariel Shaulov: arielshaulov@mail.tau.ac.il, Itay Hazan: itay.hzn@gmail.com.

Preprint. Under review.

conditioning signals to enforce temporal consistency. In this work, we explore whether a meaningful temporal representation can be extracted directly from the predictions of a pre-trained model without any additional training or auxiliary inputs. We introduce FlowMo, a novel training-free guidance method that enhances motion coherence using only the model’s own predictions in each diffusion step. FlowMo first derives an appearance-debiased temporal representation by measuring the distance between latents corresponding to consecutive frames. This highlights the implicit temporal structure predicted by the model. It then estimates motion coherence by measuring the patch-wise variance across the temporal dimension and guides the model to reduce this variance dynamically during sampling. Extensive experiments across multiple text-to-video models demonstrate that FlowMo significantly improves motion coherence without sacrificing visual quality or prompt alignment, offering an effective plug-and-play solution for enhancing the temporal fidelity of pre-trained video diffusion models.

## 1 Introduction

Despite recent progress, text-to-video diffusion models remain far from faithfully capturing the temporal dynamics of the real world. Generated videos frequently exhibit temporal artifacts such as objects appearing and disappearing, duplicated or missing limbs, and abrupt motion discontinuities [3, 4, 5]. These issues highlight the limited capability of text-to-video models to reason about motion, physics, and dynamic interactions over time. To mitigate these shortcomings, prior works have proposed fine-tuning models with explicit motion-related objectives [3], conditioning the generation on external motion signals such as optical flow or pixel trajectories [6, 7, 8, 9], or designing complex model architectures tailored to capture temporal dependencies [10, 11, 12].

However, these approaches require either retraining the model [3, 11, 12] or introducing rigid external constraints that dictate motion [6, 7], limiting flexibility and generality. In this work, we propose an alternative strategy dubbed FlowMo, a training-free guidance method that improves temporal consistency using the model’s own internal representations during sampling. FlowMo extracts a latent temporal signal directly from the pre-trained model during inference and leverages its statistics to derive a guidance signal, without any architectural modifications, training, or external supervision.

Our method is grounded in the following key observation: the temporal evolution of individual spatial patches tends to be smooth when the motion is coherent. Namely, the shifts in the representation of each patch over time are expected to be relatively small, leading to low patch-wise variance across frames. In contrast, incoherent motion intuitively manifests as abrupt changes in appearance or structure, producing high temporal variance in the patches that display temporal artifacts.

Notably, measuring temporal relations in a way that is disentangled from appearance information is challenging. As observed by previous works [3], the predictions of text-to-video models are biased toward appearance-based features. To obtain a meaningful appearance-debiased representation, we use the model’s latent predictions to compute pairwise distances between frames. This enables us to measure the shifts in patch representations using patch-wise variance over time while neutralizing their shared appearance content. This is motivated by prior works demonstrating that the latent spaces of generative models capture semantically meaningful transformations, where simple vector operations correspond to interpretable changes [13, 14, 15].

We explore the above intuition extensively in Sec. 3.2. First, we collect a set of generated videos that exhibit significant motion. We categorize these videos into coherent and incoherent sets, and compute the patch-based variance over time given the appearance-debiased representations discussed above. Our experiments yield two complementary observations. First, we find a clear correlation between high patch-based variance and motion incoherence, indicating that measuring the shift in patch representations over time can serve as a reliable metric to estimate coherence. Second, we observe both qualitatively and quantitatively that while coarse appearance-based features such as scene layout and spatial structure are established very early in the generation process, temporal information emerges only at later, intermediate denoising steps.

Motivated by these findings, we present FlowMo, a method that dynamically guides text-to-video diffusion models toward temporally coherent generations. At selected timesteps in the denoising process, we compute the maximal patch-wise variance over time, given the appearance-debiased

latent prediction. We then optimize the model’s prediction to reduce this temporal variance, thereby encouraging smoother, more coherent motion. This guidance is applied iteratively across timesteps, allowing FlowMo to influence both coarse and fine motion dynamics in the generation process.

We demonstrate our method’s effectiveness on two of the most popular open-source models, Wan2.11.3B [1] and CogVideoX-5B [2]. Across a wide range of metrics, we evaluate the impact of our method on motion quality, overall video quality, and prompt alignment, using both the automatic evaluation metrics proposed by VBench [16] and human-based assessments. In all cases, we find that FlowMo consistently and significantly improves the temporal coherence of the generated videos, while preserving the aesthetic quality, text alignment, and motion magnitude (see Fig. 1).

Our results show that it is possible to extract meaningful temporal signals from the learned latent representations of text-to-video models. Such signals not only encapsulate the temporal structure of the generated videos but also serve as actionable guidance cues.

## 2 Related Work

Text-to-video generation Diffusion models have revolutionized visual content creation, starting with image generation [17, 18, 19] and rapidly expanding to diverse applications such as text-to-image synthesis [20, 21] and image editing [22, 23, 24, 25, 26, 27, 28, 29, 30, 31]. This success has spurred their adoption for video generation, from cascaded diffusion models [32, 33, 34, 35, 36, 37, 38, 39, 40, 41], to most recently Diffusion Transformers (DiT) [42, 43] based on Flow Matching (FM) [44, 45, 46], which constitute the current state-of-the-art, and form the basis of our work.

Inference-time guidance has emerged as a powerful technique to steer and refine the outputs of generative models across various tasks without training [47, 48, 49, 28, 50]. Such methods typically optimize the model predictions based on an auxiliary loss. Inference-time guidance for video generation has only recently emerged as a promising research vector [51, 52]. While our work also explores inference-time optimization for video generation, existing objectives and guiding signals inherently differ from ours. Li et al. [51] focus on steering video models using external motion priors, which requires access to additional motion-specific inputs, while Wei et al. [52] propose to minimize a global 3D variance loss. In contrast, our method leverages the internal latent-space dynamics to perform guidance without any auxiliary networks, perceptual objectives, or task-specific priors, making it a lightweight and fully self-supervised plug-and-play module.

Improving temporal coherence in video generation Temporal coherence remains a core challenge in video synthesis [3, 4, 5, 12], and existing solutions generally fall into three categories. First, training with temporal objectives [3, 11, 53, 54], which improves consistency but demands significant compute and access to training data. Second, guiding the generation with external motion signals such as optical flow or trajectories [6, 7, 8, 9], which enforce coherence but require external inputs and are restricted to the conditioning motion. Third, architectures designed for temporal modeling [10, 11, 55, 56, 57, 58], which are often complex and not easily applied to pre-trained models. In contrast, FlowMo improves temporal coherence directly at inference time by leveraging the model’s internal representations, without additional data, inputs, or retraining.

Closest to our work, FreeInit [59] and VideoGuide [60] propose methods to reduce spatio-temporal incoherence in video generation. However, both were designed for earlier UNet-based models trained with DDPM or DDIM samplers [61, 45], which suffered from severe signal-to-noise ratio (SNR) mismatches between training and inference [59]. In contrast, modern Transformer-based FM architectures are substantially more robust, rendering these techniques less effective. For completeness, we include a comparison to FreeInit (which can be reasonably adapted to DiTs) in Appendix A. In our experiments, we find that applying FreeInit to DiTs results in a drop in key metrics such as the overall video quality, as well as a significant drop in the amount of generated motion.

## 3 Method

### 3.1 Preliminaries: Flow Matching in a VAE Latent Space

Following common practice in state-of-the-art image and video generation models [62, 63, 1], we consider models that leverage FM [44] to define the objective function and operate in the learned latent space of a Variational Autoencoder (VAE) for efficiency. The VAE consists of an encoderdecoder pair (E,D), where E maps input data x ∼ X from the pixel space to a lower-dimensional latent representation z = E(x) ∈ Z, and D yields a reconstruction x ≈ D(z). Given a pre-trained VAE, FM learns a transformation from a standard Gaussian distribution in latent space z0 ∼ N(0,I), to a target distribution z1 observed from applying E on the data.

At each training step, FM draws a timestep t ∈ [0,1], and obtains a noised intermediate latent by interpolating between z0 and z1, namely zt = (1 − t) · z1 + t · z0. The model uθ is then optimized to predict the velocity vt = dz

dt = z0 − z1, namely:

t

1,t∼U(0,1),z0∼N(0,I) ∥uθ(zt,t) − (z0 − z1)∥2 . (1)

LFM = Ex

Once trained, samples can be generated from an initial noisy latent z0 ∼ N(0,I) by applying a sequence of denoising steps over a discrete schedule. At time ti, zt

by applying zt

is denoised to produce zt

i

i+1

is an interpolation coefficient determined by the scheduler.

,ti), where σt

= (1−σt

)·zt

i −σt

i ·uθ(zt

i+1

i

i

i

### 3.2 Motivation

In the following, we conduct qualitative and quantitative experiments to motivate the construction of FlowMo. The experiments in this section are conducted on Wan2.1-1.3B [1] for efficiency.

We begin by describing the latent representation on which FlowMo operates. As mentioned in Sec. 3.1, at each denoinsing step, the model prediction uθ,t := uθ(zt,t) is an estimate of the velocity vt, which represents the direction from the noise distribution to the latent space distribution. To extract a temporal representation from the prediction, we propose a debiasing operator ∆, which computes the ℓ1-distance between consecutive latent frames to eliminate their common appearance information. Formally, ∆: RF×W×H×C → R(F−1)×W×H×C is defined as: ∀f ∈ [F − 1],∀w ∈ [W],∀h ∈ [H],∀c ∈ [C]

(∆uθ,t)f,w,h,c = ∥(uθ,t)f+1,w,h,c − (uθ,t)f,w,h,c∥1. (2)

Next, we describe the motivational experiments conducted to examine the statistical characteristics of our proposed latent space.

Mean Patch-wise Variance 2

Quantitative motivation. Our central hypothesis is that temporally coherent motion corresponds to a form of local stability in uθ,t. Specifically, in videos with smooth and consistent motion, object trajectories evolve gradually, yielding lower temporal variance in uθ,t. Incoherent motion, in contrast, introduces abrupt changes, manifesting as larger fluctuations and higher patch-wise variance in the latent predictions. Formally, given uθ,t, we define its temporal patch-wise variance tensor σ2 ∈ RW×H×C as the variance across frames per patch and channel, i.e. ∀w ∈ [W],∀h ∈ [H],∀c ∈ [C],

Incoherent (59 videos)

0.68

Coherent (61 videos)

0.66

0.64

0.62

0.60

0.58

0 5 10 15 20 25 30 35 40

Timestep

Figure 2: Quantitative motivation. We measure the mean temporal variance of spatial patches for coherent and incoherent videos. Incoherent videos portray higher variance. The separation is visible from step 5 onward. 95%-confidence interval was computed using the seaborn python package.

σw,h,c2 = Vf∼[F−1] [(∆uθ,t)f,w,h,c] , (3) where V(X) = E[(X − E[X])2].

To empirically validate our hypothesis, we conducted a user study wherein several hundred generated

videos were rated on a 1-5 scale for both coherence and perceived amount of motion (higher is more motion/better coherence). To isolate the effects of motion magnitude on video coherence, we focused on videos with a substantial amount of motion (rated ≥ 3), and compared those labeled

- as completely incoherent (1), or completely coherent (5). As illustrated in Fig. 2, a clear negative correlation emerges: low-coherence videos consistently exhibit higher variance. This supports our intuition that temporal patch-wise variance is a meaningful measure of perceived coherence.

Notably, the separation in variance becomes prominent from approximately the fifth generation timestep onward. Next, we wish to conduct a qualitative experiment to motivate this phenomenon.

|[Figure 17]|[Figure 18]|[Figure 19]|
|---|---|---|

t=0

Frame 8 Frame 14 Frame 20

|[Figure 20]|[Figure 21]|[Figure 22]|
|---|---|---|

t=4

|[Figure 23]|[Figure 24]|[Figure 25]|
|---|---|---|

t=8

Figure 3: Qualitative motivation. We visualize the model prediction per timestep across the generation. Coarse spatial information is determined in the first steps (0-4), whereas motion is determined at steps 5-8, and refined in later steps.

Qualitative motivation. To qualitatively explore the process of motion generation in text-to-video models, we visualize the evolution of the model’s latent space prediction across the generation steps.

First, observe that by Sec. 3.1, the model prediction uθ,t estimates the velocity vt = z0 −z1. We can thus obtain an estimation of the fully denoised latent, z¯1,

- at any intermediary step t as follows: z¯1 = zt − σt · uθ,t, (4)

where σt is the signal-to-noise ratio at step t, which is defined by the noise scheduler.

Notably, z¯1 is a latent-space representation where channels do not represent RGB information. We thus

arbitrarily select the first channel, to obtain z¯1,c0 ∈ RF×H×W, and visualize the grayscale video via:

z¯1,c0 − minF,H,W(¯z1,c0) maxF,H,W(¯z1,c0) − minF,H,W(¯z1,c0)

. (5)

= 255 ·

Vz¯

1,c0

in representative timesteps are presented in Fig. 3. As can be observed, coarse spatial information begins to emerge from the first generation step, where the training bar and some of the outline of the person are visible. By step 4, most of the structure of the scene is already determined. Conversely, the motion appears to be added to the scene between steps 4 and 8, as evidenced by the strong similarity between all frames in times 0,4, whereas step 8 shows significant variance between these same frames (e.g., the person bends down between frames 14 and 20). Visualizations on additional timesteps and latent channels are provided in Appendix B. Note that the above supports the quantitative experiment presented in Fig. 2. Since motion emerges only around the later initial steps of the denoising process (4-8), we would expect our variance-based metric to be meaningful only in the steps that depict measurable differences over time.

The resulting visualizations of Vz¯

1,c0

In App. C, we show quantitative evidence that, in addition to enhancing coherence, FlowMo reduces the variance in generation steps that correspond to the above intuition.

### 3.3 FlowMo

Motivated by the previous section, Algorithm 1 outlines the FlowMo guidance mechanism, applied within a single FM denoising step. We perform the FlowMo guidance at specific timesteps {τ1,...,τℓ}, corresponding to the early-to-mid stages of generation, following the motivation presented in Fig. 3.

, given an input text prompt P. To encourage alignment between the prediction and the textual prompt, Classifier-free guidance (CFG) [64] is first employed with a scale of ρ (Line 3). If we are not in a refinement step, we jump to Line 14, in which we perform a standard FM step to obtain the next latent zt

Each denoising step ti begins by obtaining the model prediction uθ,t

i

as a linear combination of the current latent and the predicted velocity:

i+1

, (6) where σt

= (1 − σt

) · zt

i − σt

i · uθ,t

zt

i+1

i

i

is a time-dependent coefficient representing the signal-to-noise ratio.

i

If, however, ti ∈ {τi}ℓ1, a FlowMo refinement step is performed (Line 5 to Line 12). We first compute the appearance-debiased representation, ∆uθ,t

Algorithm 1 A Single FlowMo Denoising Step Input: A text prompt P, a timestep ti, a set of

, as defined in Eq. (2) (Line 5).

i

iterations for refinement {τ1,...,τℓ}, and a trained Flow Matching model FM.

Subsequently, drawing on the motivation presented in Fig. 2, we calculate the temporal variance σw,h,c2 for each spatial patch (w,h,c) as defined in Eq. (3) (Line 6). These patch-wise variances are then averaged across the channel dimension to produce a single spatial map sw,h indicating a motion coherence score per patch (Line 7): ∀w ∈ [W],∀h ∈ [H]

Output: A noised latent zt

for the next timestep ti+1

i+1

- 1: uθ,t

i|P ← FM(zt

i

,ti,P)

- 2: uθ,t

i|∅ ← FM(zt

i

,ti,∅)

- 3: uθ,t

i ← uθ,t

i|P + ρ · uθ,t

i|P − uθ,t

i|∅

- 4: if ti ∈ {τ1,...,τℓ} then
- 5: Compute (∆uθ,t

i

) as in Eq. (2)

- 6: Compute σ2 as in Eq. (3)
- 7: sw,h ← Ec∼[C] σw,h,c2 ∀w∀h
- 8: L ← maxw∼[W],h∼[H] sw,h
- 9: zt

i ← zt

i − η · ∇zti

L

- 10: uθ,t

i|P ← FM(zt

i

,ti,P)

- 11: uθ,t

i|∅ ← FM(zt

i

,ti,∅)

- 12: uθ,ti ← uθ,ti|P + ρ · uθ,ti|P − uθ,ti|∅
- 13: end if
- 14: zt

i+1 ← (1 − σt

i

) · zt

i − σt

i · uθ,t

i

- 15: Return zt

C

1 C

sw,h = Ec∼[C] σw,h,c2 =

σw,h,c2 . (7)

c=1

The final FlowMo loss L is then determined by the maximal value in this map, thereby targeting the most dynamically-incoherent patch (Line 8):

sw,h. (8)

L = max

w∼[W],h∼[H]

Intuitively, this formulation encourages the model to produce a prediction uθ,t

wherein the latent space distances of each spatial patch over time are smoother, resulting in more coherent and gradual transitions in the generated video.

i

i+1

Inspired by existing guidance mechanisms that optimize spatial information [47], we propose to use the loss in Eq. 7 to optimize the input latent to the diffusion step, zt

(Line 9). Intuitively, this allows our optimization to modify low-level features in the generated video, including the coarse motion. Thus, the optimization is performed as a gradient descent step:

i

L, (9)

i − η · ∇zti

zt

= zt

i

where η is the learning rate. Following this refinement, we repeat the denoising step ti with the optimized latent zt

(Lines 10 to 12).

i

## 4 Experiments

We conduct qualitative and quantitative experiments to demonstrate FlowMo’s effectiveness. Our experiments evaluate the improvement in temporal coherence enabled by our method, as well as its ability to maintain or even enhance other aspects of the generation, such as appearance quality and text alignment. We provide our code and a website with video results in the supplemental materials.

Implementation details We employ two of the most popular publicly available text-to-video models: Wan2.1-1.3B [1] and CogVideoX-5B [2], using their officially provided weights and default configurations. Motivated by the insights from Sec. 3.2, we apply FlowMo in the first 12 timesteps of the generation, since these are responsible for coarse motion and structure. All our experiments employ a learning rate of η = 0.005, using the Adam optimizer, on two NVIDIA H100 GPUs, with 80GB memory each. Wan2.1 is evaluated at a resolution of 480×832, and CogVideoX at 480×720, both generating 81 frames at 16 frames per second, resulting in 5-second videos.

### 4.1 Qualitative Results

Figures 1, 4 contain representative results demonstrating the impact of FlowMo on pre-trained text-tovideo models. As can be observed, our method mitigates severe temporal artifacts that are common to text-to-video models. For example, the generations tend to display extra limbs (women in Fig. 1(a) and Fig. 4, 2nd, 3rd row), distortions of objects over time (dolphin in Fig. 1(a) and deer in Fig. 4, 4th

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

+FlowMoWan2.1-1.3B

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

“A roulette wheel in a dimly lit room or casino floor” “A boy with glasses flying a kite in a grassy field”

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

+FlowMoWan2.1-1.3BCogVideoX-5BCogVideoX-5B+FlowMo+FlowMo

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

“A boy skipping rope in a backyard” “A senior couple dancing at sunset on a pier”

(a)

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

“A figure skater gliding across the ice” “A violinist performing a solo on stage”

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

“A deer leaping over a fallen log”

“A teenager skateboarding down a handrail”

(b)

- Figure 4: Qualitative results. Text-to-video results before and after applying FlowMo on (a) Wan2.1 [1] and (b) CogVideoX [2]. FlowMo mitigates severe temporal artifacts, e.g., extra limbs (women, 2nd, 3rd row), objects that appear or disappear (2nd, 3rd row), and distortions (4th row).

row), and objects that suddenly appear or disappear (flamingo in Fig. 1(b) and rope, violin in Fig. 4, 2nd, 3rd row). These results demonstrate that temporal artifacts correspond to abrupt changes in the latent representations of video patches. This, in turn, drives our optimization process to encourage smoother representations of the affected patches, resulting in improved temporal coherence.

Wan2.1-1.3B

CogVideoX-5B

Text alignment

14.9%

77.9%

7.2%

15.6%

75.7%

8.7%

31.1%

54.9%

14.0%

31.7%

51.3%

17.1%

Quality

44.3%

39.5%

16.2%

43.0%

39.4%

17.6%

Motion

0 20 40 60 80 100

0 20 40 60 80 100

Win Rate

Win Rate

Ours Both Baseline

- Figure 5: User study conducted on Wan2.1-1.3B [1] (left) and CogVideoX-5B [2] (right) using VideoJAM-bench [3], designed specifically to evaluate motion coherence. Our method significantly improves temporal coherence in all models, while maintaining or improving the visual quality and the text alignment of the resulting videos. 95%-confidence intervals were calculated using Dirichlet sampling, assuming a multinomial distribution with Laplace smoothing applied to the counts.

- Table 1: VBench evaluation results. A comparison of the overall video quality before and after applying FlowMo on Wan2.1-1.3B [1] and CogVideoX-5B [2] using VBench [16]. We enclose both the motion-specific and the aggregated scores. FlowMo consistently improves the Final Score representing the overall video quality by at least 5%.

Motion Metrics Aggregated Scores

|Models<br><br>|Motion Smoothness<br><br>Dynamic Degree|Semantic Score<br><br>Quality Score<br><br>Final Score|
|---|---|---|
|Wan2.1-1.3B<br><br>+ FlowMo<br><br>|96.43% 83.21% 98.56% 81.96%|84.70% 65.58% 75.14% 89.11% 73.58% 81.34% (+6.20%)|
|CogVideoX-5B<br><br>+ FlowMo|95.01% 65.29% 97.29% 63.92%|70.03% 60.83% 65.43% 69.26% 72.11% 70.69% (+5.26%)|

### 4.2 Quantitative Results

We employ both the VBench benchmark [16] and human-based evaluations, which serve as the standard evaluation protocols for measuring the quality of text-to-video generation [2, 65, 66, 67, 68].

User study. We conduct a human preference study using the VideoJAM benchmark [3], which was specifically designed to test motion coherence. For each prompt, we generate a pair of videos (with and without FlowMo) with a fixed seed in the setting described above, and randomly shuffle the order of the results. Each prompt was evaluated by five different participants, resulting in 640 unique responses per baseline. Annotators were asked to compare the videos based on their alignment to the text prompt, the aesthetic quality of the videos, and the motion quality (see App. D).

The results, presented in Fig. 5, demonstrate a consistent human preference for FlowMo-guided videos across all criteria. Specifically for Motion Coherence, FlowMo was favored in 44.3% of comparisons for Wan2.1 (vs. 16.2% for baseline) and 43.0% for CogVideoX (vs. 17.6% for baseline). A similar trend was observed for Aesthetic Quality, where FlowMo was preferred in 31.1% of Wan2.1 pairs (vs. 14.0% for baseline) and 31.7% of CogVideoX pairs (vs. 17.1% for baseline). Interestingly, FlowMo also showed improved Text-Video Alignment, with preference rates of 14.9% for Wan2.1 (vs. 7.2% for baseline) and 15.6% for CogVideoX (vs. 8.7% for baseline).

These findings highlight that FlowMo not only enhances temporal coherence but also contributes positively to the overall perceived video quality and faithfulness to the input prompt.

Automatic metrics. The results of the automatic metrics on the VBench benchmark [16] are summarized in Tab. 1. We enclose both the motion-based metrics, and the aggregated metrics, which constitute an average of all the benchmark dimensions, and measure the overall quality of the generations. A full breakdown of all metrics is provided in App. E.

Notably, FlowMo significantly improves the Final Score by 6.2%, 5.26% for Wan, CogVideoX, respectively. This metric represents the overall quality score, considering all the evaluation dimensions. This improvement is supported by gains in the Quality Score (Wan2.1: +8.0%; CogVideoX: +11.28%) and Semantic Score for Wan2.1 (+4.41%), with a negligible decrease of 0.77% for CogVideoX.

Considering the motion metrics, FlowMo boosts Motion Smoothness (Wan2.1: +2.13%; CogVideoX: +2.28%), which is a key metric that evaluates the motion coherence. Finally, note that some decrease to the dynamic degree is expected. This is since temporal artifacts such as objects appearing and disappearing increases the amount on motion in the video. With that, we observe that the metric does not decrease significantly (less than 1.5% for both models), especially compared to the 16.20% decrease in this metric demonstrated by FreeInit, FlowMo’s most closely-related method (see App. A).

In summary, both human evaluations and automated VBench metrics consistently demonstrate FlowMo’s effectiveness in improving motion coherence and overall video quality.

### 4.3 Ablation Study

We ablate the primary design choices of FlowMo, namely using the patch with the maximal variance to derive the loss (Eq. 8), the appearance debiasing operator ∆ (Eq. 2), and the selection of diffusion steps to apply the optimization (1-12).

“A ninja flipping through a bamboo forest” “Pikachu with Charmander at a campfire”

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

AllStepsFlowMoW/ODebiasMean

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

Wan2.1-1.3

B

Results of the ablation study on Wan2.1 are reported in Fig. 6. For each prompt, we enclose the result with FlowMo (1st row), Wan2.1 (2nd row) and the ablations (3rd-5th row). Replacing the maximum with the mean (3rd row) significantly weakens the optimization effect, likely because most patches are static, leading to a smaller loss and diminished gradients. Removing the debiasing operator (4th row) yields a similar effect. This can be attributed to the fact that, as observed by previous works [3], predictions by text-to-video models tend to be appearance-based, reducing the influence of motion on the loss. Finally, applying FlowMo across all diffusion steps (5th row) introduces artifacts, as the optimization interferes with high frequencies and fine details in steps where motion is already determined.

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

Figure 6: Ablation study. We ablate the main design choices of FlowMo, i.e., using the maximal variance for the objective (3rd row), using the appearance-debiasing operator (4th row), the selection of the optimization steps (5th row), and show that FlowMo is significantly superior to all variants.

### 4.4 Limitations

While our method enables a substantial improvement in the motion coherence and overall quality of generated videos, it still has a few limitations. First, due to the calculation and propagation of gradients by our method (see Alg. 1), there is some slowdown in the inference time. On average, generating a video with FlowMo takes 234.30 seconds compared to 99.27 seconds without it, corresponding to a ×2.39 increase. This overhead could be mitigated by integrating FlowMo into the training phase, eliminating the need for gradient-based optimization at inference time.

Second, since FlowMo does not modify the model weights, it is bounded by the learned capabilities of the pre-trained model. While it can improve the coherence of motion predicted by the model, it cannot synthesize motion types the model has not learned to represent. We believe this limitation can be addressed by incorporating motion-based objectives based on the model’s internal representations during training, encouraging richer temporal understanding in generative video models.

## 5 Conclusions

Can we extract meaningful temporal representations from a model with limited temporal understanding? In this work, we propose a new approach to address temporal artifacts in text-to-video models. Instead of relying on external signals, additional data, or specialized architectures, we repurpose the model’s own learned representations as a source of temporal guidance. Specifically, we examine the semantic latent space learned by text-to-video diffusion models and find that it implicitly encodes valuable temporal information. Through extensive analysis (Sec. 3.2), we show that distances between pairs of frames in this latent space correlate with intuitive measures of temporal artifacts, such as patch-wise variance over time. Building on these insights, we implement an inference-time guidance method that encourages smoother transitions in the latent space, and observe that this maps to smoother behavior in pixel space as well, significantly boosting motion coherence while preserving and even improving other aspects of the generation. We hope this work sparks further interest in exploring the temporal properties of semantic latent spaces and encourages the development of methods that improve temporal coherence by looking inward rather than outward.

## 6 Acknowledgments

This work was supported by a grant from the Tel Aviv University Center for AI and Data Science (TAD).

## References

- [1] Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, Jianyuan Zeng, Jiayu Wang, Jingfeng Zhang, Jingren Zhou, Jinkai Wang, Jixuan Chen, Kai Zhu, Kang Zhao, Keyu Yan, Lianghua Huang, Mengyang Feng, Ningyi Zhang, Pandeng Li, Pingyu Wu, Ruihang Chu, Ruili Feng, Shiwei Zhang, Siyang Sun, Tao Fang, Tianxing Wang, Tianyi Gui, Tingyu Weng, Tong Shen, Wei Lin, Wei Wang, Wei Wang, Wenmeng Zhou, Wente Wang, Wenting Shen, Wenyuan Yu, Xianzhong Shi, Xiaoming Huang, Xin Xu, Yan Kou, Yangyu Lv, Yifei Li, Yijing Liu, Yiming Wang, Yingya Zhang, Yitong Huang, Yong Li, You Wu, Yu Liu, Yulin Pan, Yun Zheng, Yuntao Hong, Yupeng Shi, Yutong Feng, Zeyinzi Jiang, Zhen Han, Zhi-Fan Wu, and Ziyu Liu. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.
- [2] Wenyi Hong, Ming Ding, Wendi Zheng, Xinghan Liu, and Jie Tang. Cogvideo: Large-scale pretraining for text-to-video generation via transformers. arXiv preprint arXiv:2205.15868, 2022.
- [3] Hila Chefer, Uriel Singer, Amit Zohar, Yuval Kirstain, Adam Polyak, Yaniv Taigman, Lior Wolf, and Shelly Sheynin. Videojam: Joint appearance-motion representations for enhanced motion generation in video models. arXiv preprint arXiv:2502.02492, 2025.
- [4] Bingyi Kang, Yang Yue, Rui Lu, Zhijie Lin, Yang Zhao, Kaixin Wang, Gao Huang, and Jiashi Feng. How far is video generation from world model: A physical law perspective, 2024.
- [5] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, Clarence Ng, Ricky Wang, and Aditya Ramesh. Video generation models as world simulators. 2024.
- [6] Daniel Geng, Charles Herrmann, Junhwa Hur, Forrester Cole, Serena Zhang, Tobias Pfaff, Tatiana LopezGuevara, Carl Doersch, Yusuf Aytar, Michael Rubinstein, Chen Sun, Oliver Wang, Andrew Owens, and Deqing Sun. Motion prompting: Controlling video generation with motion trajectories, 2024.
- [7] Wan-Duo Kurt Ma, J. P. Lewis, and W. Bastiaan Kleijn. Trailblazer: Trajectory control for diffusion-based video generation, 2023.
- [8] Shaowei Liu, Zhongzheng Ren, Saurabh Gupta, and Shenlong Wang. Physgen: Rigid-body physicsgrounded image-to-video generation. In European Conference on Computer Vision ECCV, 2024.
- [9] Yuren Cong, Mengmeng Xu, Christian Simon, Shoufa Chen, Jiawei Ren, Yanping Xie, Juan-Manuel Perez-Rua, Bodo Rosenhahn, Tao Xiang, and Sen He. Flatten: optical flow-guided attention for consistent text-to-video editing. arXiv preprint arXiv:2310.05922, 2023.
- [10] S. Tulyakov, Ming-Yu Liu, Xiaodong Yang, and Jan Kautz. Mocogan: Decomposing motion and content for video generation. 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1526–1535, 2017.
- [11] Yang Jin, Zhicheng Sun, Kun Xu, Kun Xu, Liwei Chen, Hao Jiang, Quzhe Huang, Chengru Song, Yuliang Liu, Di Zhang, Yang Song, Kun Gai, and Yadong Mu. Video-lavit: Unified video-language pre-training with decoupled visual-motional tokenization. ArXiv, abs/2402.03161, 2024.
- [12] Lvmin Zhang and Maneesh Agrawala. Packing input frame context in next-frame prediction models for video generation. arXiv preprint arXiv:2504.12626, 2025.
- [13] Alec Radford, Luke Metz, and Soumith Chintala. Unsupervised representation learning with deep convolutional generative adversarial networks. CoRR, abs/1511.06434, 2015.
- [14] Yujun Shen, Jinjin Gu, Xiaoou Tang, and Bolei Zhou. Interpreting the latent space of gans for semantic face editing. 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 9240–9249, 2019.
- [15] Rinon Gal, Or Patashnik, Haggai Maron, Amit H. Bermano, Gal Chechik, and Daniel Cohen-Or. Stylegannada. ACM Transactions on Graphics (TOG), 41:1 – 13, 2021.
- [16] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, Yaohui Wang, Xinyuan Chen, Limin Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. VBench: Comprehensive benchmark suite for video generative models. In CVPR, 2024.
- [17] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In NeurIPS, 2020.
- [18] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P. Kingma, Ben Poole, Mohammad Norouzi, David J. Fleet, and Tim Salimans. Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303, 2022.

- [19] Prafulla Dhariwal and Alex Nichol. Diffusion models beat GANs on image synthesis. arXiv preprint arXiv:2105.05233, 2021.
- [20] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation, 2021.
- [21] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily Denton, Seyed Kamyar Seyed Ghasemipour, Burcu Karagol Ayan, S. Sara Mahdavi, Rapha Gontijo Lopes, Tim Salimans, Jonathan Ho, David J Fleet, and Mohammad Norouzi. Photorealistic text-to-image diffusion models with deep language understanding, 2022.
- [22] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H Bermano, Gal Chechik, and Daniel Cohen-Or. An image is worth one word: Personalizing text-to-image generation using textual inversion. arXiv preprint arXiv:2208.01618, 2022.
- [23] Hila Chefer, Shiran Zada, Roni Paiss, Ariel Ephrat, Omer Tov, Michael Rubinstein, Lior Wolf, Tali Dekel, Tomer Michaeli, and Inbar Mosseri. Still-moving: Customized video generation without customized video data. arXiv preprint arXiv:2407.08674, 2024.
- [24] Shelly Sheynin, Adam Polyak, Uriel Singer, Yuval Kirstain, Amit Zohar, Oron Ashual, Devi Parikh, and Yaniv Taigman. Emu edit: Precise image editing via recognition and generation tasks. In CVPR, 2024.
- [25] Uriel Singer, Amit Zohar, Yuval, Shelly Sheynin, Adam Polyak, Devi Parikh, and Yaniv Taigman. Video editing via factorized diffusion distillation. In ECCV, 2024.
- [26] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Prompt-toprompt image editing with cross attention control. arXiv preprint arXiv:2208.01626, 2022.
- [27] Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 18392–18402, 2023.
- [28] Yoad Tewel, Rinon Gal, Dvir Samuel, Yuval Atzmon, Lior Wolf, and Gal Chechik. Add-it: Training-free object insertion in images with pretrained diffusion models. arXiv preprint arXiv:2411.07232, 2024.
- [29] Bahjat Kawar, Shiran Zada, Oran Lang, Omer Tov, Huiwen Chang, Tali Dekel, Inbar Mosseri, and Michal Irani. Imagic: Text-based real image editing with diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6007–6017, 2023.
- [30] Rahul Sajnani, Jeroen Vanbaar, Jie Min, Kapil Katyal, and Srinath Sridhar. Geodiffuser: Geometry-based image editing with diffusion models. In 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pages 472–482. IEEE, 2025.
- [31] Binxin Yang, Shuyang Gu, Bo Zhang, Ting Zhang, Xuejin Chen, Xiaoyan Sun, Dong Chen, and Fang Wen. Paint by example: Exemplar-based image editing with diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 18381–18391, 2023.
- [32] Jonathan Ho, Chitwan Saharia, William Chan, David J Fleet, Mohammad Norouzi, and Tim Salimans. Cascaded diffusion models for high fidelity image generation. arXiv preprint arXiv:2106.15282, 2021.
- [33] Omer BarTal, Hila Chefer, Omer Tov, Charles Herrmann, Roni Paiss, Shiran Zada, Ariel Ephrat, Junhwa Hur, Yuanzhen Li, Tomer Michaeli, Oliver Wang, Deqing Sun, Tali Dekel, and Inbar Mosseri. Lumiere: A space-time diffusion model for video generation. arXiv preprint arXiv:2401.12945, 2024.
- [34] Jie An, Songyang Zhang, Harry Yang, Sonal Gupta, Jia-Bin Huang, Jiebo Luo, and Xi Yin. Latent-Shift: Latent diffusion with temporal shift for efficient text-to-video generation. arXiv preprint arXiv:2304.08477, 2023.
- [35] Niket Agarwal, Arslan Ali, Maciej Bala, Yogesh Balaji, Erik Barker, Tiffany Cai, Prithvijit Chattopadhyay, Yongxin Chen, Yin Cui, Yifan Ding, et al. Cosmos world foundation model platform for physical ai. arXiv preprint arXiv:2501.03575, 2025.
- [36] Jake Bruce, Michael D Dennis, Ashley Edwards, Jack Parker-Holder, Yuge Shi, Edward Hughes, Matthew Lai, Aditi Mavalankar, Richie Steigerwald, Chris Apps, et al. Genie: Generative interactive environments. In Forty-first International Conference on Machine Learning, 2024.
- [37] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024.

- [38] Hedra. Hedra: Ai-powered character video generation. https://www.hedra.com/, 2025. Accessed: May 13, 2025.
- [39] OpenAI. Sora: Text-to-Video Generation. https://openai.com/sora/, 2024. Accessed: May 13, 2025.
- [40] Google DeepMind. Genie 2: A Large-Scale Foundation World Model. https://deepmind.google/ discover/blog/genie-2-a-large-scale-foundation-world-model/, 2024. Accessed: May 13, 2025.
- [41] World Labs. Generating worlds. https://www.worldlabs.ai/blog, 2024. Accessed: 2025-05-13.
- [42] William Peebles and Saining Xie. Scalable diffusion models with transformers. In ICCV, 2023.
- [43] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. SDXL: improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023.
- [44] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.
- [45] Yaohui Wang, Xinyuan Chen, Xin Ma, Shangchen Zhou, Ziqi Huang, Yi Wang, Ceyuan Yang, Yinan He, Jiashuo Yu, Peiqing Yang, et al. LaVie: High-quality video generation with cascaded latent diffusion models. arXiv preprint arXiv:2309.15103, 2023.
- [46] Wenliang Zhao, Lujia Bai, Yongming Rao, Jie Zhou, and Jiwen Lu. Unipc: A unified predictor-corrector framework for fast sampling of diffusion models. Advances in Neural Information Processing Systems, 36:49842–49869, 2023.
- [47] Hila Chefer, Yuval Alaluf, Yael Vinker, Lior Wolf, and Daniel Cohen-Or. Attend-and-excite: Attentionbased semantic guidance for text-to-image diffusion models. ACM transactions on Graphics (TOG), 42(4):1–10, 2023.
- [48] Omer Dahary, Or Patashnik, Kfir Aberman, and Daniel Cohen-Or. Be yourself: Bounded attention for multi-subject text-to-image generation. European Conference on Computer Vision (ECCV), pages 432–448, 2024.
- [49] Lital Binyamin, Yoad Tewel, Hilit Segev, Eran Hirsch, Royi Rassin, and Gal Chechik. Make it count: Text-to-image generation with an accurate number of objects. arXiv preprint arXiv:2406.10210, 2024.
- [50] Zhipeng Bao, Yijun Li, Krishna Kumar Singh, Yu-Xiong Wang, and Martial Hebert. Separate-and-enhance: Compositional finetuning for text-to-image diffusion models. In ACM SIGGRAPH 2024 Conference Papers, pages 1–10, 2024.
- [51] Jialu Li, Shoubin Yu, Han Lin, Jaemin Cho, Jaehong Yoon, and Mohit Bansal. Training-free guidance in text-to-video generation via multimodal planning and structured noise initialization. arXiv preprint arXiv:2504.08641, 2025.
- [52] Min Wei, Chaohui Yu, Jingkai Zhou, and Fan Wang. 3dv-ton: Textured 3d-guided consistent video try-on via diffusion models. arXiv preprint arXiv:2504.17414, 2025.
- [53] Haoxin Chen, Yong Zhang, Xiaodong Cun, Menghan Xia, Xintao Wang, Chao Weng, and Ying Shan. Videocrafter2: Overcoming data limitations for high-quality video diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7310–7320, 2024.
- [54] Xun Wu, Shaohan Huang, Guolong Wang, Jing Xiong, and Furu Wei. Boosting text-to-video generative model with mllms feedback. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024.
- [55] Yingqing He, Tianyu Yang, Yong Zhang, Ying Shan, and Qifeng Chen. Latent video diffusion models for high-fidelity long video generation. arXiv preprint arXiv:2211.13221, 2022.
- [56] Ruben Villegas, Mohammad Babaeizadeh, Pieter-Jan Kindermans, Hernan Moraldo, Han Zhang, Mohammad Taghi Saffar, Santiago Castro, Julius Kunze, and Dumitru Erhan. Phenaki: Variable length video generation from open domain textual description. arXiv preprint arXiv:2210.02399, 2022.
- [57] Fu-Yun Wang, Wenshuo Chen, Guanglu Song, Han-Jia Ye, Yu Liu, and Hongsheng Li. Gen-l-video: Multi-text to long video generation via temporal co-denoising. arXiv preprint arXiv:2305.18264, 2023.

- [58] Haonan Qiu, Menghan Xia, Yong Zhang, Yingqing He, Xintao Wang, Ying Shan, and Ziwei Liu. FreeNoise: Tuning-free longer video diffusion via noise rescheduling. arXiv preprint arXiv:2310.15169, 2023.
- [59] Tianxing Wu, Chenyang Si, Yuming Jiang, Ziqi Huang, and Ziwei Liu. Freeinit: Bridging initialization gap in video diffusion models. arXiv preprint arXiv:2312.07537, 2023.
- [60] Dohun Lee, Bryan S Kim, Geon Yeong Park, and Jong Chul Ye. Videoguide: Improving video diffusion models without training through a teacher’s guide. arXiv preprint arXiv:2410.04364, 2024.
- [61] Yuwei Guo, Ceyuan Yang, Anyi Rao, Yaohui Wang, Yu Qiao, Dahua Lin, and Bo Dai. AnimateDiff: Animate your personalized text-to-image diffusion models without specific tuning. arXiv preprint arXiv:2307.04725, 2023.
- [62] Black Forest Labs. FLUX, 2024.
- [63] Adam Polyak, Amit Zohar, Andrew Brown, Andros Tjandra, Animesh Sinha, Ann Lee, Apoorv Vyas, Bowen Shi, Chih-Yao Ma, Ching-Yao Chuang, David Yan, Dhruv Choudhary, Dingkang Wang, Geet Sethi, Guan Pang, Haoyu Ma, Ishan Misra, Ji Hou, Jialiang Wang, Kiran Jagadeesh, Kunpeng Li, Luxin Zhang, Mannat Singh, Mary Williamson, Matt Le, Matthew Yu, Mitesh Kumar Singh, Peizhao Zhang, Peter Vajda, Quentin Duval, Rohit Girdhar, Roshan Sumbaly, Sai Saketh Rambhatla, Sam Tsai, Samaneh Azadi, Samyak Datta, Sanyuan Chen, Sean Bell, Sharadh Ramaswamy, Shelly Sheynin, Siddharth Bhattacharya, Simran Motwani, Tao Xu, Tianhe Li, Tingbo Hou, Wei-Ning Hsu, Xi Yin, Xiaoliang Dai, Yaniv Taigman, Yaqiao Luo, Yen-Cheng Liu, Yi-Chiao Wu, Yue Zhao, Yuval Kirstain, Zecheng He, Zijian He, Albert Pumarola, Ali Thabet, Artsiom Sanakoyeu, Arun Mallya, Baishan Guo, Boris Araya, Breena Kerr, Carleigh Wood, Ce Liu, Cen Peng, Dimitry Vengertsev, Edgar Schonfeld, Elliot Blanchard, Felix Juefei-Xu, Fraylie Nord, Jeff Liang, John Hoffman, Jonas Kohler, Kaolin Fire, Karthik Sivakumar, Lawrence Chen, Licheng Yu, Luya Gao, Markos Georgopoulos, Rashel Moritz, Sara K. Sampson, Shikai Li, Simone Parmeggiani, Steve Fine, Tara Fowler, Vladan Petrovic, and Yuming Du. Movie gen: A cast of media foundation models, 2024.
- [64] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022.
- [65] Jieyu Zhang, Weikai Huang, Zixian Ma, Oscar Michel, Dong He, Tanmay Gupta, Wei-Chiu Ma, Ali Farhadi, Aniruddha Kembhavi, and Ranjay Krishna. Task me anything. arXiv preprint arXiv:2406.11775, 2024.
- [66] Weiming Ren, Huan Yang, Ge Zhang, Cong Wei, Xinrun Du, Wenhao Huang, and Wenhu Chen. Consisti2v: Enhancing visual consistency for image-to-video generation. arXiv preprint arXiv:2402.04324, 2024.
- [67] Xuanhua He, Quande Liu, Shengju Qian, Xin Wang, Tao Hu, Ke Cao, Keyu Yan, and Jie Zhang. Idanimator: Zero-shot identity-preserving human video generation. arXiv preprint arXiv:2404.15275, 2024.
- [68] Jianwen Jiang, Chao Liang, Jiaqi Yang, Gaojie Lin, Tianyun Zhong, and Yanbo Zheng. Loopy: Taming audio-driven portrait avatar with long-term motion dependency. arXiv preprint arXiv:2409.02634, 2024.

## A Comparison between FlowMo and FreeInit

In this section, we compare FlowMo with FreeInit [59], which is most closely related to our method. FreeInit was designed for earlier UNet-based models that employ DDPM or DDIM [61, 45]. Its approach is motivated by the observation that such models often exhibit significant spatio-temporal inconsistencies in scene elements across frames (e.g., character identities and backgrounds changing between frames). Their primary observation is that these models exhibit discrepancies in signal-tonoise ratios (SNR) between training and inference phases, causing temporal artifacts. In contrast, modern Transformer-based architectures, as used in our work, are generally more robust to these inconsistencies due to more powerful architectures, more stable training frameworks (using FM), and larger training datasets. Thus, these models are able to maintain consistent appearance across frames and are less susceptive to these types of artifacts.

To provide a fair comparison, we adapted FreeInit for use with FM-based DiT models. This involved re-noising a denoised latent and combining this re-noised latent with random noise to initialize the low-frequency components, before repeating the denoising process, as per FreeInit’s methodology. We then conducted both quantitative (user study and VBench automatic metrics) and qualitative comparisons between FlowMo and our adapted FreeInit. The experiments are demonstrated hereafter.

Implementation details. All experiments were done on the Wan2.1-1.3B model, for efficiency. The experimental setting for the vanilla baseline and FlowMo-guided model is the same as in Sec. 4. FreeInit was implemented based on its publicly available open-source code [59] and its default configuration, namely employing the butterworth filter with n = 4, ds = dt = 0.25. To remain comparable with our work, we performed one refinement iteration with each of the methods.

### A.1 Qualitative Experiments

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

###### FlowMoFreeInit

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

“A person lifts one knee high in a marching motion” “A close-up of a person’s feet as they walk”

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

FlowMoFreeInit

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

“A man jumping rope on a dark stage” “Young Adult Male Doing Handstand on the beach”

- Figure 7: Qualitative results. Text-to-video results of FreeInit [59] (1st, 3rd row) and FlowMo (2nd, 4th row) when applied on Wan2.1-1.3B [1]. FlowMo better mitigates severe temporal artifacts, e.g. distortions and object that appear and disappear.

Qualitative comparisons between FlowMo and the adapted FreeInit are presented in Fig. 7. These examples highlight FlowMo’s superiority in generating visually coherent motion. For instance, in

Wan2.1-1.3B vs FlowMo

FlowMo vs FreeInit

Text alignment

14.9%

77.9%

7.2%

16.5%

78.3%

5.2%

31.1%

54.9%

14.0%

28.1%

57.1%

14.8%

Quality

44.3%

39.5%

16.2%

38.7%

38.0%

23.4%

Motion

0 20 40 60 80 100

0 20 40 60 80 100

Win Rate

Win Rate

Ours Both Baseline

- Figure 8: User studies conducted on Wan2.1-1.3B [1] using VideoJAM-bench [3]. The studies compare three variants of the model: vanilla (Wan2.1-1.3B), with FlowMo, and with FreeInit. Our method significantly outperforms the baselines in both studies. 95%-confidence intervals were calculated using Dirichlet sampling, assuming a multinomial distribution with Laplace smoothing applied to the counts.

the top-left example, FlowMo successfully generates a plausible marching motion, whereas FreeInit produces disappearing feet and less convincing movement. Similarly, in the top-right example, FlowMo depicts coherent walking, while the rendition produced by FreeInit suffers from partially disappearing feet and a less natural gait. The bottom-left example shows FlowMo maintaining the integrity of the man and rope, while in the video refined with FreeInit, the rope distorts and disappears and the man has a less stable form. Finally, in the bottom-right example, FlowMo maintains a consistent orientation, whereas the front and back sides of the person flip spontaneously in the version generated with FreeInit.

Overall, these visual examples show that FlowMo demonstrates a significant advantage in producing more coherent and artifact-free motion compared to FreeInit.

### A.2 Quantitative Experiments

Consistent with the experiments presented in the main paper, we compare FlowMo to the FreeInit baseline using both the VBench benchmark [16] and human evaluation results.

User study. We conducted a human preference study that compared videos generated by Wan2.11.3B guided by FlowMo with those guided by FreeInit. Videos from both models were sampled given prompts from the VideoJAM benchmark [3] in the same setting described in Sec. 4.2 of the main paper.

The results, presented in Fig. 8, clearly indicate a strong human preference for FlowMo across all evaluated categories, in comparison to FreeInit as well as the vanilla Wan2.1-1.3B model. Comparing FreeInit and FlowMo, for Motion Coherence, FlowMo was preferred in 38.7% of comparisons, substantially more than FreeInit (23.4%). In terms of Aesthetic Quality, FlowMo was chosen in 28.1% of pairs, nearly double the preference for FreeInit (14.8%). Furthermore, FlowMo also outperformed FreeInit in Text-Video Alignment, with a preference rate of 16.5% compared to FreeInit’s 5.2%. These results demonstrate that human evaluators find FlowMo-guided videos to be significantly more coherent, aesthetically pleasing, and better aligned with textual prompts than those guided by the adapted FreeInit.

VBench Benchmark. We further evaluate FlowMo against the adapted FreeInit using the VBench benchmark. The detailed results per dimension are presented in Tab. 2.

The VBench metrics [16] corroborate the user study findings, showing FlowMo’s superiority. First, observe that across the comprehensive suite of VBench metrics detailed in Tab. 2, the adapted FreeInit does not achieve a superior score to both FlowMo and the baseline in any individual dimension.

Considering the main metrics related to motion coherence, FreeInit achieves only a marginal improvement in Motion Smoothness (+0.11% over baseline) compared to FlowMo’s substantial +2.13% gain. Critically, FreeInit significantly degrades the Dynamic Degree by -16.20% from the baseline

- Table 2: VBench evaluation results per dimension. Each column represents a model variant, Wan2.1 (Baseline), with FlowMo, and with FreeInit. Rows correspond to the 16 VBench evaluation dimensions. While FlowMo significantly increases the overall quality of the videos (+6.20%), FreeInit reduces it (-0.08%), and is unable to compare with the Baseline and FlowMo in any of the dimensions.

Dimension Wan2.1-1.3B Baseline + FlowMo + FreeInit

Subject Consistency 95.61% 96.54% 93.71% Background Consistency 97.25% 97.02% 97.12% Temporal Flickering 99.15% 98.93% 97.77% Motion Smoothness 96.43% 98.56% 96.54% Dynamic Degree 83.21% 81.96% 67.01% Aesthetic Quality 56.77% 58.03% 50.99% Imaging Quality 61.01% 64.89% 57.19% Object Class 91.37% 95.35% 92.11% Multiple Objects 77.98% 82.27% 73.26% Human Action 98.27% 97.23% 97.98% Color 87.94% 87.28% 86.53% Spatial Relationship 75.21% 78.42% 76.52% Scene 49.84% 49.41% 48.32% Appearance Style 20.95% 28.45% 21.59% Temporal Style 26.35% 27.30% 24.24% Overall Consistency 23.65% 25.53% 22.13%

Semantic Score 84.70% 89.11% 85.91% Quality Score 65.58% 73.58% 64.22% Final Score 75.14% 81.34% (+6.20%) 75.06 (-0.08%)

(from 83.21% to 67.01%), whereas FlowMo maintains a comparable dynamic level (81.96%). This large reduction in motion by FreeInit suggests that its apparent coherence might stem from producing less dynamic videos, which are inherently easier to keep coherent, rather than genuinely improving the quality of complex motion.

Furthermore, FreeInit performs worse than both the baseline and FlowMo in several other important quality aspects. For instance, its Aesthetic Quality (50.99%) is lower than both baseline (56.77%) and FlowMo (58.03%). Similar trends are observed for other important qualities, e.g. Temporal Flickering (FreeInit: 97.77% vs. FlowMo: 98.93%, Baseline: 99.15%), Imaging Quality (FreeInit: 57.19% vs. FlowMo: 64.89%), Appearance Style (FreeInit: 21.59% vs. FlowMo: 28.45%), Temporal Style (FreeInit: 24.24% vs. FlowMo: 27.30%), and Overall Consistency (FreeInit: 22.13% vs. FlowMo: 25.53%).

Finally, FlowMo significantly outperforms FreeInit in the aggregated VBench metrics. FlowMo achieves a Final Score of 81.34%, a +6.20% improvement over the baseline, while FreeInit scores 75.06%, slightly below the baseline. Similarly, FlowMo leads in Quality Score (73.58% vs. FreeInit’s 64.22%) and Semantic Score (89.11% vs. FreeInit’s 85.91%). These results underscore that FlowMo provides a more effective and well-rounded improvement to video generation quality compared to the adapted FreeInit on modern FM-based DiT architectures.

## B Additional Qualitative Motivation Results

To complement the qualitative observations in the main paper, we present additional visualizations of the latent-space predictions z¯1,c across timesteps during the generation process. The two figures below show a grid of frames from six different time indices (10–20) and ten representative diffusion steps (0–18), providing a spatio-temporal view of how motion emerges over time within the latent space.

Fig. 9 displays predictions for channel 0, used in the main paper, while Fig. 10 shows results for a randomly selected channel (channel 7). In both cases, we observe that coarse spatial structure appears in early steps, while coarse motion emerges primarily between steps 4 and 8. Although motion is refined in later steps, as seen in timesteps t = 10 onward, its coarse features are determined earlier, making the first timesteps the most crucial for coherent motion generation. These patterns reinforce our interpretation that motion is added into the generation during these intermediate steps, which underpins our focus on this range for motion-aware optimization in FlowMo.

Frame 10 Frame 12 Frame 14 Frame 16 Frame 18 Frame 20

|[Figure 94]|[Figure 95]|[Figure 96]|[Figure 97]|[Figure 98]|[Figure 99]|
|---|---|---|---|---|---|

t=0

|[Figure 100]|[Figure 101]|[Figure 102]|[Figure 103]|[Figure 104]|[Figure 105]|
|---|---|---|---|---|---|

t=2

|[Figure 106]|[Figure 107]|[Figure 108]|[Figure 109]|[Figure 110]|[Figure 111]|
|---|---|---|---|---|---|

t=4

|[Figure 112]|[Figure 113]|[Figure 114]|[Figure 115]|[Figure 116]|[Figure 117]|
|---|---|---|---|---|---|

t=6

|[Figure 118]|[Figure 119]|[Figure 120]|[Figure 121]|[Figure 122]|[Figure 123]|
|---|---|---|---|---|---|

t=8

|[Figure 124]|[Figure 125]|[Figure 126]|[Figure 127]|[Figure 128]|[Figure 129]|
|---|---|---|---|---|---|

t=10

|[Figure 130]|[Figure 131]|[Figure 132]|[Figure 133]|[Figure 134]|[Figure 135]|
|---|---|---|---|---|---|

t=12

|[Figure 136]|[Figure 137]|[Figure 138]|[Figure 139]|[Figure 140]|[Figure 141]|
|---|---|---|---|---|---|

t=14

|[Figure 142]|[Figure 143]|[Figure 144]|[Figure 145]|[Figure 146]|[Figure 147]|
|---|---|---|---|---|---|

t=16

|[Figure 148]|[Figure 149]|[Figure 150]|[Figure 151]|[Figure 152]|[Figure 153]|
|---|---|---|---|---|---|

t=18

- Figure 9: A visualization of channel 0 (selected arbitrarily, and used in the main paper) of the latent prediction at different timesteps of the generation.

Frame 10 Frame 12 Frame 14 Frame 16 Frame 18 Frame 20

|[Figure 154]|[Figure 155]|[Figure 156]|[Figure 157]|[Figure 158]|[Figure 159]|
|---|---|---|---|---|---|

t=0

|[Figure 160]|[Figure 161]|[Figure 162]|[Figure 163]|[Figure 164]|[Figure 165]|
|---|---|---|---|---|---|

t=2

|[Figure 166]|[Figure 167]|[Figure 168]|[Figure 169]|[Figure 170]|[Figure 171]|
|---|---|---|---|---|---|

t=4

|[Figure 172]|[Figure 173]|[Figure 174]|[Figure 175]|[Figure 176]|[Figure 177]|
|---|---|---|---|---|---|

t=6

|[Figure 178]|[Figure 179]|[Figure 180]|[Figure 181]|[Figure 182]|[Figure 183]|
|---|---|---|---|---|---|

t=8

|[Figure 184]|[Figure 185]|[Figure 186]|[Figure 187]|[Figure 188]|[Figure 189]|
|---|---|---|---|---|---|

t=10

|[Figure 190]|[Figure 191]|[Figure 192]|[Figure 193]|[Figure 194]|[Figure 195]|
|---|---|---|---|---|---|

t=12

|[Figure 196]|[Figure 197]|[Figure 198]|[Figure 199]|[Figure 200]|[Figure 201]|
|---|---|---|---|---|---|

t=14

|[Figure 202]|[Figure 203]|[Figure 204]|[Figure 205]|[Figure 206]|[Figure 207]|
|---|---|---|---|---|---|

t=16

|[Figure 208]|[Figure 209]|[Figure 210]|[Figure 211]|[Figure 212]|[Figure 213]|
|---|---|---|---|---|---|

t=18

- Figure 10: A visualization of channel 7 (selected randomly) of the latent prediction at different timesteps of the generation.

## C The Effect of FlowMo on Patch-Wise Variance

Max Patch-wise Variance 2

This section demonstrates the alignment between FlowMo’s optimization mechanism and our motivating insights from Sec. 3.2. Fig. 11 plots the maximal patch-wise temporal variance (FlowMo’s loss, Eq. (7)) for videos generated with and without FlowMo guidance.

Without FlowMo

1.3

With FlowMo

1.2

1.1

1.0

We observe that the results demonstrate a strong correlation with the conclusions from Sec. 3.2. First, the application of FlowMo results in a significant reduction and stabilization of this maximal variance, which is particularly evident from approximately timestep 5 onward. This observation is consistent with our earlier finding (Fig. 2) that the variance characteristics of coherent and incoherent videos begin to diverge at this stage.

0.9

0.8

0 5 10 15 20 25 30 35 40

Timestep

Figure 11: FlowMo effect on patch-wise variance. We plot the maximal temporal variance of spatial patches for videos with and without applying FlowMo guidance, and observe that our method significantly reduces and stabilizes the variance in the generation steps that impact motion the most by our analysis from Sec. 3.2. 95%-confidence interval was computed using the seaborn python package.

Second, FlowMo’s optimization is applied during the initial 12 timesteps of the generation process. This targeted intervention aligns with our qualitative motivation (Fig. 3), which indicates that coarse motion patterns are predominantly established within these early stages of the generation. As can be observed, applying the optimization at these steps indeed stabilizes the maximal variance in all other, non-optimized steps as well.

Consequently, Fig. 11 illustrates that FlowMo guidance leads to a notable decrease in the maximal patch-wise temporal variance. Videos generated with FlowMo exhibit consistently lower variance, especially within the critical timesteps 5-15, compared to the higher and more fluctuating variance observed in videos generated without FlowMo. This empirically validates that FlowMo operates as intended by reducing the target variance metric during the crucial phases of motion formation.

## D User Study: Instructions Provided to Participants

As part of the evaluations we performed on our method, we conducted a user study, as described in Sec. 4.2. The study was designed to assess human preferences on videos generated with and without FlowMo, using the videoJAM benchmark [3], which focuses on motion coherence.

The study was conducted using Google Forms. For each prompt, participants were shown a pair of videos—one with FlowMo and one without—generated with the same random seed (1024). The order of the videos was randomized to avoid positional bias. Each pair was evaluated by five different participants, resulting in 640 responses per baseline.

Participants were asked to evaluate the videos based on three criteria: text alignment, aesthetic quality, and motion coherence. The instructions provided to annotators are reproduced below, followed by a screenshot of the interface used:

Hello! We need your help to read a caption, and then watch two generated videos.

After watching the videos, we want you to answer a few questions about them:

- • Text alignment: Which video better matches the caption?
- • Quality: Aesthetically, which video is better?
- • Motion: Which video has more coherent and physically plausible motion? (Do note: it is OK if the quality is less impressive as long as the motion looks better.)

[Figure 214]

Figure 12: Screenshot of the Google Form used in the user study.

## E VBench Metrics Breakdown

- Table 3: VBench evaluation results per dimension. Each column represents a model variant, with and without FlowMo. Rows correspond to the 16 VBench evaluation dimensions.

Dimension Wan2.1-1.3B CogVideoX-5B Baseline + FlowMo Baseline + FlowMo

Subject Consistency 95.61% 96.54% 95.79% 97.02% Background Consistency 97.25% 97.02% 96.53% 97.53% Temporal Flickering 99.15% 98.93% 99.23% 96.21% Motion Smoothness 96.43% 98.56% 95.01% 97.29% Dynamic Degree 83.21% 81.96% 65.29% 63.92% Aesthetic Quality 56.77% 58.03% 55.51% 58.29% Imaging Quality 61.01% 64.89% 58.91% 58.75% Object Class 91.37% 95.35% 82.72% 88.41% Multiple Objects 77.98% 82.27% 60.17% 61.27% Human Action 98.27% 97.23% 97.81% 95.69% Color 87.94% 87.28% 82.75% 80.82% Spatial Relationship 75.21% 78.42% 67.89% 67.23% Scene 49.84% 49.41% 51.55% 54.13% Appearance Style 20.95% 28.45% 23.53% 30.29% Temporal Style 26.35% 27.30% 25.04% 31.26% Overall Consistency 23.65% 25.53% 26.43% 24.28%

Semantic Score 84.70% 89.11% 70.03% 69.26% Quality Score 65.58% 73.58% 60.83% 72.11% Final Score 75.14% 81.34% (+6.20%) 65.43% 70.69% (+5.26%)

In Sec. 4.2, we reported the aggregated VBench [16] metrics, as well as specific metrics that correspond to motion coherence and magnitude. Here, we provide the complete breakdown across all 16 individual evaluation dimensions for both Wan2.1 and CogVideoX.

Tab. 3 compares the baseline models with their FlowMo-guided counterparts. FlowMo leads to consistent improvements in key dimensions, including Subject Consistency, Motion Smoothness, and Object Class, while maintaining or slightly improving aesthetic and perceptual metrics such as Aesthetic Quality, Appearance Style, and Spatial Relationship. Although a small decrease is observed in Dynamic Degree, this aligns with our expectation that reducing motion artifacts also reduces spurious motion.

Critically, as mentioned in the main text, FlowMo consistently and significantly boosts the overall quality metric (Final Score) by at least 5% across all models. This is a clear indication of the positive impact our method has on the overall quality of the produced viseos.

