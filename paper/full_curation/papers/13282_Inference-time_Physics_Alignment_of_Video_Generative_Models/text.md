# arXiv:2601.10553v2[cs.CV]27Feb2026

## Inference-time Physics Alignment of Video Generative Models with Latent World Models

Jianhao Yuan1,2,∗, Xiaofeng Zhang1,3,4,‡, Felix Friedrich1,‡, Nicolas Beltran-Velez1,5,‡,∗, Melissa Hall1, Reyhane Askari-Hemmat1, Xiaochuang Han1, Nicolas Ballas1, Michal Drozdzal1,†, Adriana Romero-Soriano1,3,6,7,†

1FAIR, Meta Superintelligence Labs, 2University of Oxford, 3Mila - Québec AI Institute, 4Université de Montréal, 5Columbia University, 6McGill University, 7Canada CIFAR AI Chair ∗Work done at Meta, †Joint last author, ‡Equal contribution

State-of-the-art video generative models produce promising visual content yet often violate basic physics principles, limiting their utility. While some attribute this deficiency to insufficient physics understanding from pre-training, we find that the shortfall in physics plausibility also stems from suboptimal inference strategies. We therefore introduce WMReward and treat improving physics plausibility of video generation as an inference-time alignment problem. In particular, we leverage the strong physics prior of a latent world model (here, VJEPA-2) as a reward to search and steer multiple candidate denoising trajectories, enabling scaling test-time compute for better generation performance. Empirically, our approach substantially improves physics plausibility across imageconditioned, multiframe-conditioned, and text-conditioned generation settings, with validation from human preference study. Notably, in the ICCV 2025 Perception Test PhysicsIQ Challenge, we achieve a final score of 62.64%, winning first place and outperforming the previous state of the art by 7.42%. Our work demonstrates the viability of using latent world models to improve physics plausibility of video generation, beyond this specific instantiation or parameterization.

Date: March 2, 2026 Code: https://github.com/facebookresearch/WMReward Correspondence: {adrianars, mdrozdzal}@meta.com, jianhaoyuan@robots.ox.ac.uk

1 Introduction

State-of-the-art video generative models (Sand.ai et al., 2025; Brooks et al., 2024; OpenAI, 2025; Kondratyuk et al., 2024; Bar-Tal et al., 2024; Wan et al., 2025; Polyak et al., 2024) have shown remarkable capabilities in generating visually pleasing videos. Yet, the progress in generation quality has been hindered by the limited physics understanding of these models (Kang et al., 2024; Motamed et al., 2025; Yuan et al., 2025a), resulting in physically implausible video generation (Bansal et al., 2024, 2025). Ensuring physics correctness in video generation is not only crucial to increase user satisfaction, but also for reliable world modeling (LeCun, 2022) and downstream applications such as robotics (Yang et al., 2023) and autonomous driving (Hu et al., 2023).

Prior work attributes the observed physical implausibility to the pre-training stage of video generative models (Kang et al., 2024), which relies on minimizing pixel or feature-level reconstruction errors.

Following this hypothesis, substantial work has focused on improving pre-training or post-training of video generative models by injecting physics information (Yuan et al., 2025b; Chefer et al., 2025; Li et al., 2024b; Cao et al., 2024; Zhang et al., 2025c,a). By contrast, another line of work assumes that physically plausible videos may be found in the manifold learned by the generative model and therefore has focused on devising inference-time methods to improve physics. This is an underexplored line of research, with two contributions relying on visionlanguage models (VLMs) to rewrite prompts (Xue et al., 2025) and plan motion (Yang et al., 2025b) when a motion-controllable video generative model is available (Burgert et al., 2025). However, in the image generation literature, alternatives to prompt rewriting (Datta et al., 2024; Zhang et al., 2025d) have been extensively explored to search over the generated data manifold with the goal of boosting performance at inference-time. In particular, reward models have been used to perform candidate selection with search over random seeds (Ma et al., 2025;

[Figure 1]

- Figure 1 Left: We achieve the new state-of-the-art on PhysicsIQ benchmark, improving physics plausibility in both single (I2V) and multiframe (V2V) conditioned video generation by a considerable margin. Right: Our latent world model reward WMReward substantially outperforms VLM and other vision foundation model-based reward signals under BoN search; the proposed ∇+BoN sampling strategy further improves the scaling effect.

Li et al., 2025; Zhang et al., 2025b) and to guide the model’s sampling process towards high utility generations (Ye et al., 2024; Hemmat et al., 2023; Askari Hemmat et al., 2024; Dall’Asen et al., 2025; Askari-Hemmat et al., 2025; Ifriqi et al., 2025). Despite these advances, existing work has not explored the use of off-the-shelf reward models to improve the physics plausibility of video generation at inference time.

Recently, latent world models have demonstrated strong capabilities in physics understanding (Bordes et al., 2025; Luo et al., 2025; Garrido et al., 2025). A latent world model is a predictive model that encodes high-dimensional observations (e.g. , videos) into compact latent representations and learns the transition function in this latent space to forecast future states. By learning the transition function in this compressed representation rather than in pixel space, these models are trained to focus on features that are predictive of future latent states, such as motion and object dynamics, while ignoring irrelevant appearance details. This leads to representations that emphasize fundamental scene properties like structure, object permanence, and trajectory continuity (Garrido et al., 2025). As a result, they are ideally suited to serve as reward models for physics plausibility. VJEPA-2 (Assran et al., 2025) is one particular instantiation of a latent world model, which we employ in this work and which has demonstrated strong physics understanding. Yet, our approach of using latent world models as a source of reward is not tied to this particular instantiation; any suitable latent world model could potentially be used for this purpose.

In this paper, we formulate the problem of improving the physics correctness of video generation as an inference-time alignment problem (Uehara et al.,

2025; Singhal et al., 2025) that leverages a latent world model with high physics understanding as a reward model to search for physically plausible videos within the manifold learned by the generative model. In particular, we introduce WMReward by re-purposing VJEPA-2’s surprise score as a reward function and show that it can be effectively used both as a Best-of-N (BoN) selector and as a guidance signal during generation to improve the physics of videos.

We validate the effectiveness of WMReward by achieving substantial gains in physics plausibility of video generation using three state-of-the-art video generative models: MAGI-1 (Sand.ai et al., 2025), Sora2 (OpenAI, 2025), and a video latent diffusion model (vLDM). These models span diverse architectures, including large-scale autoregressive generation producing videos chunk-by-chunk and holistic diffusion-based generation. We obtain results across text-to-video (T2V), image-and-text-to-video (I2V), and video-and-text-to-video (V2V) settings. In particular, we achieve a new state of the art on the challenging PhysicsIQ benchmark (Motamed et al., 2025) with a final score of 62.0%, surpassing the previous best by 6.78%. This improvement is illustrated by the qualitative examples in Figure 1 and further validated by an additional 11.4% improvement over baselines in a human-preference study. Moreover, as shown in Figure 1, our VJEPA-based method demonstrates a strong scaling effect with the size of the search space, and outperforms VLM-based selectors (e.g. , Qwen3-VL (Yang et al., 2025a)) that perform near chance level. With these results, we demonstrate the viability of using latent world models to improve the physical plausibility of video generation, paving the way toward developing reliable reward models for video generation inference-time alignment. The

[Figure 2]

- Figure 2 We improve the physics plausibility of video generation by aligning a pre-trained diffusion model with a latent world model at inference time. Using a reward derived from latent world models, we perform search on guided denoising trajectories to sample from a tilted physics plausible distribution. Compared to the baseline (middle row of each quadrant), our generation (bottom) adheres more closely to real-world physics (top), exhibiting smoother temporal continuity, more accurate solid interactions, and improved fluid behavior.

key contributions of our work are as follows:

- • We show that latent world models are useful to improve video generation. To this end, we devise an effective physics plausibility reward model, WMReward, instantiated with VJEPA’s surprise score.
- • We highlight the scaling behavior of WMReward and show that model performance improves when increasing search space – with guided sampling showing more promising scaling.
- • We boost the physics plausibility in video generation across T2V, I2V and V2V settings and datasets, and achieve a final score of 62.0% on the challenging PhysicsIQ benchmark, outperforming prior state-of-the-art by 6.78%, which is further validated through a human-preference study showing a boost of 11.4% in win rate over the baseline.

### 2 Methodology: WMReward

In the following, we present WMReward, which transfers the physics prior from a latent world model (e.g. , VJEPA-2) into a physics plausibility reward signal. We leverage such reward signal to align the video generative model with the latent world model by sampling from a tilted distribution and thus generate physically more plausible videos.

- 2.1 Preliminaries

Diffusion and flow-matching models are two generative modeling paradigms that learn to transform Gaussian noise into samples from a target distribution pdata(x). They do this by learning score functions ∇xt

log pt(xt) of the random variables xt with distribution

xt = αtx0 + σtϵ, (1) where x0∼pdata and ϵ∼N(0,I). Here, αt and σt are real-valued time-dependent functions where at t=0, we have α0 = 1 and σ0 = 0 so that p0(x) = pdata(x), and at t=T, pT(x) is approximately Gaussian. This score is learned using either denoising score matching Vincent (2011) or conditional flow matching Lipman et al. (2023).

Once this score is available, we can construct an SDE or ODE whose marginals pt(x) depend only on αt, σt, and the learned score. We then can draw approximate samples from this learned distribution p(x) by solving this SDE/ODE backward from t = T to t = 0.

- 2.2 Steering for Better Physics Plausibility

We cast the problem of improving physics plausibility in video generation as sampling from a rewardweighted tilted distribution:

p∗(x) ∝ w(x)p(x), (2)

where p(x) is the pre-trained video model distribution and w(x) > 0 is a weighting function constructed from the reward r(x) that can evaluate the physics plausibility.

###### Generated Video

k - C k k + M

VideoGenerativeModel

Frame Index:

Prediction Context Prediction Future

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

... ...

...

VJEPA Surprise

X X

Eθ

Eθ VJEPA Encoder

VJEPA Encoder

Inference-time Alignment

X X X X

Pφ

VJEPA Predictor

###### X : Masking

|X|X| | | |
|---|---|---|---|---|
| | | | | |

Cosine Sim.

- Figure 3 Method Overview. We leverage a latent world model, VJEPA-2, to steer video generative models for better physics plausibility. During generation, we apply a sliding window approach and split the generative model’s output into sets of context and future frames. We encode generated context frames and predict the embedding of future frames using the latent world model’s predictor. Then, we encode the generated future frames and compute the cosine similarity between its embedding and the latent world model prediction, referred to as surprise score. The surprise score serves as reward to search and guide the denoising trajectories.

To successfully draw samples from this tilted distribution, two key questions arise: 1) how to define the reward function that captures physics plausibility from a latent world model (Section 2.3); and 2) how to realize sampling from p∗(x) given the pre-trained generative model p(x) and the reward function r(x) (Section 2.4). We depict an overview of our method in Figure 3.

- 2.3 Latent World Model Reward Signal

Recent work has demonstrated that latent world models—which learn to make predictions on compressed latent representations rather than raw pixelscan develop the ability to assess physical plausibility (Garrido et al., 2025; Luo et al., 2025; Bordes et al., 2025). By operating in a latent space, they tend to ignore superficial visual details and focus on underlying physical dynamics. One approach for training such models relies on self-supervised learning; VJEPA-2 (Assran et al., 2025) is a latent world model trained in this manner which achieves state-of-the-art performance on physics understanding benchmarks (Garrido et al., 2025), making it an ideal foundation for our reward model.

VJEPA learns representations through self-supervised prediction of unmasked video regions from masked ones. The architecture comprises a context encoder Eθ that embeds video frames, and a predictor net-

work Pϕ that reconstructs target representations from partial observations. During training, a masked view xmasked is created from a video x by removing certain spatiotemporal regions. The predictor Pϕ takes the encoder’s output Eθ(xmasked) along with learnable mask tokens ∆m, indicating masked positions, and attempts to reconstruct the target representation. The target is computed using an exponentially-movingaveraged (EMA) encoder E¯θ applied to the full video. The training objective is:

L = ∥Pϕ(∆m,Eθ(xmasked)) − sg(E¯θ(x))∥1 (3)

where sg(·) denotes the stop-gradient operation, preventing gradients from flowing through the target. The idea is that by making predictions in feature space, the model is encouraged to learn high-level spatiotemporal features that capture predictable dynamics rather than pixel-level details.

To turn VJEPA into a physics plausibility reward model, we leverage a simple intuition: being a world model, VJEPA should resonably predict the future in physically plausible videos. Hence, the more its predictions diverge from the generations, the more likely it is that the videos are less physically plausible.

We materialize this intuition by designing a reward function, WMReward, that measures prediction surprise by contrasting VJEPA-2 predictions with the generated video as shown in Figure 3. Concretely, we slide a window of length C +M across a generated video x, where C denotes context frames and M denotes prediction horizon. At each position k, VJEPA observes context frames xk−C+1:k and must predict representations for future frames xk+1:k+M. We produce latent world model’s future representations using only context, as follows:

zˆk = Pϕ(∆m,Eθ(xk−C+1:k)), (4)

where ∆m indicates that M future positions are masked. We obtain generated video’s representations by processing the complete window:

zk = Eθ(xk−C+1:k+M). (5) Both representations contain representations for all C + M positions; we extract and compare only the future portions corresponding to positions C + 1 through C + M (zˆkfut and zkfut):

1 |K| k∈K

1 − cos(ˆzkfut,zkfut) , (6)

r(x) =

where K contains all valid window positions. Videos receive higher rewards when their generated futures closely match VJEPA’s predictions, effectively measuring physical coherence.

L = 1 − cos(zpred, zgt)

- 2.4 Instantiation of Sampling Schemes

We now consider practical sampling schemes with different weighting functions w(x) to draw samples from the defined tilted distribution in Equation (2). As presented in Section 2.3, the reward function r(x) is differentiable with respect to the video x, which allows us to leverage both gradient-based and gradientfree sampling schemes. We focus on three sampling schemes chosen for their representativeness, simplicity, and performance: guidance as an example for gradient-based method, best-of-N for gradient-free method, and their combination.

- (i) Guidance (∇) uses w(x) = exp(λr(x)), where r(x) is our reward function and λ > 0 is a temperature parameter controlling how much we upweight highreward samples. Under the same noising process as in Equation (1), one can show that the score functions

∇xt

log p∗t(xt) of the time-dependent marginals p∗t(xt) are given by

∇xt

log pt(xt) + ∇xt

log E eλr(x

0) xt . (7)

Therefore, approximating E[eλr(x

0) | xt] as eλr(E[x

0|xt]), and using Tweedie’s formula (Robbins, 1992; Efron, 2011),

x0|t := E[x0 | xt] =

1 αt

(xt + σt∇xt

log pt(xt)), (8)

we obtain the following approximation to Equation (7)

∇xt

log p∗t(xt) ≈ ∇xt

log pt(xt) + λ∇xt

r(x0|t(xt)) (9)

We can therefore sample from p∗(x) by using this new score approximation in the SDE/ODE sampler.

- (ii) Best-of-N Search (BoN) involves generating N

independent samples {x(i)}Ni=1 (i.e. , particles) from the base model p(x) and selecting the sample with highest reward:

x∗ = argmaxx∈{x(i)}Ni=1r(x). (10) This procedure effectively samples from a distribution

p∗(x) ∝ p(x)[F(r(x))]N−1, (11)

where F(·) is the cumulative distribution function (CDF) of r(x) under p(x) and w(x) = [F(r(x))]N−1.

- (iii) ∇ + BoN combines both approaches by using the guidance scheme to generate N samples and then chooses the highest reward sample among them. This yields approximate samples

from the tilted distribution with weight function w(x) = exp(λr(x)) · [Fλ(r(x))]N−1, where Fλ(·) is the CDF of r(x) under the guided distribution. This combination is beneficial because it achieves stronger tilting without requiring large λ values (where the score approximation becomes inaccurate), and bestof-N selection provides an additional strategy for filtering samples where the approximation might have been sub-optimal. In this way, we benefit from the increased likelihood of high-reward samples provided by guidance, while mitigating approximation errors.

#### 3 Experiments

We show the effectiveness of sampling with WMReward for improving physics plausibility across three video generation setups: image-and-textto-video (I2V), video-and-text-to-video (V2V), and text-to-video (T2V). For all experiments, we apply WMReward to MAGI-1 24B (Sand.ai et al., 2025) and to vLDM 5B1, and additionally to Sora2 (OpenAI, 2025) on the PhysicsIQ benchmark.

- 3.1 Image and Multiframe-conditioned Generation

We start by evaluating WMReward on PhysicsIQ (Motamed et al., 2025), a benchmark for I2V and V2V generation. We follow their setup and use the text prompt along with a three-second context video (for V2V) or the last frame (for I2V). Given the conditioning, we generate five-second video continuations, evaluate it against the ground-truth and compute the PhysicsIQ score that relies on an aggregation of four metrics: spatial intersection over union (IoU), spatio-temporal IoU over motion masks, a weighted spatial IoU, and pixel MSE. For all methods, we use 16 particles for the search.

Table 1 demonstrates that sampling with WMReward consistently outperforms vanilla sampling for vLDM, MAGI-1, and Sora2, achieving state-ofthe-art performance (best and second-best) across baselines and evaluation dimensions. In particular, by using the appropriate sampling strategy, ∇+BoN, we achieve 62.0% final PhysicsIQ score on V2V generation, which significantly outperforms the previous state-of-the-art MAGI-1 model by 6.78%. For I2V, sampling with WMReward surpasses the previous state-of-the-art Sora2 (OpenAI, 2025) by

- 4.13% on PhysicsIQ.

1vLDM is a latent video diffusion model that utilizes a DiT backbone and jointly models spatial and temporal components at once.

Table 1 Image and Multiframe-conditioned Generation Performance on PhysicsIQ. Our results are highlighted in gray and performance changes against baseline sampling are indicated in green. Best results are highlighted in bold and second-best are underlined. For all search methods, we use 16 particles.

* The official results from the ICCV 2025 PhysicsIQ Challenge platform are MAGI-1 I2V 37.39 (+7.62) and V2V 62.64 (+7.42). † Sora2 is an API model; guidance (∇) is not available.

Spatial IoU ↑

Spatiotemp. IoU ↑

Weighted Spatial IoU ↑

PhysicsIQ Score ↑ I2V Generation

MSE ↓

Sora (Brooks et al., 2024) 0.138 0.047 0.063 0.030 10.00 Pika 1.0 (Pika Labs, 2023) 0.140 0.041 0.078 0.014 13.00 SVD (Blattmann et al., 2023) 0.132 0.076 0.073 0.021 14.80 Lumiere (Bar-Tal et al., 2024) 0.113 0.173 0.061 0.016 19.00 VideoPoet (Kondratyuk et al., 2024) 0.141 0.126 0.087 0.012 20.30

- Wan2.1 (Wan et al., 2025) 0.153 0.100 0.112 0.023 20.89 Gen 3 (Runway, 2024) 0.201 0.115 0.116 0.015 22.80 Kling1.6 (Kuaishou, 2024) 0.197 0.086 0.144 0.025 23.64 VLIPP (Yang et al., 2025b) N/A N/A N/A N/A 34.6 vLDM 0.221 0.120 0.144 0.008 27.76

+ VideoMAE(BoN) 0.229 0.132 0.152 0.009 29.42(+1.66) + Qwen2.5-VL(BoN) 0.228 0.092 0.143 0.009 26.21(-1.55) + Qwen3-VL(BoN) 0.214 0.139 0.141 0.008 28.51(+0.75) + WMReward (∇) 0.221 0.122 0.144 0.009 27.88(+0.12) + WMReward(BoN) 0.249 0.155 0.170 0.008 32.90(+5.14) + WMReward(∇+BoN) 0.234 0.175 0.165 0.007 33.44(+5.68)

MAGI-1 (Sand.ai et al., 2025) 0.252 0.146 0.151 0.011 29.77

+ VideoMAE(BoN) 0.251 0.151 0.153 0.012 29.95(+0.18) + Qwen2.5-VL(BoN) 0.241 0.093 0.141 0.013 24.99(-4.78) + Qwen3-VL(BoN) 0.247 0.161 0.146 0.010 30.21(+0.44) + WMReward(∇) 0.252 0.145 0.152 0.010 29.77(+0.00) + WMReward(BoN) 0.253 0.239 0.162 0.008 36.56(+6.79) + WMReward(∇+BoN) * 0.267 0.218 0.168 0.008 36.28(+6.51)

- Wan2.2 (Wan et al., 2025) 0.358 0.120 0.217 0.008 38.26

+ VideoMAE(BoN) 0.347 0.124 0.216 0.008 37.91(-0.35) + Qwen2.5-VL(BoN) 0.356 0.116 0.213 0.009 37.58(-0.68) + Qwen3-VL(BoN) 0.353 0.128 0.216 0.008 38.51(+0.25) + WMReward(BoN) 0.378 0.171 0.247 0.008 44.39(+6.13)

Sora2 (OpenAI, 2025) 0.363 0.187 0.231 0.007 42.30

+ VideoMAE(BoN) 0.365 0.193 0.234 0.007 42.94(+0.64) + Qwen2.5-VL(BoN) 0.340 0.166 0.210 0.007 38.69(-3.61) + Qwen3-VL(BoN) 0.349 0.206 0.226 0.007 42.54(+0.24) + WMReward(BoN) † 0.377 0.229 0.246 0.007 46.43(+4.13)

V2V Generation

Lumiere (Bar-Tal et al., 2024) 0.170 0.155 0.093 0.013 23.00 VideoPoet (Kondratyuk et al., 2024) 0.204 0.164 0.137 0.010 29.50

MAGI-1 (Sand.ai et al., 2025) 0.416 0.279 0.294 0.005 55.22

+ VideoMAE(BoN) 0.407 0.286 0.285 0.005 54.71(-0.51) + Qwen2.5-VL(BoN) 0.415 0.272 0.293 0.005 54.71(-0.51) + Qwen3-VL(BoN) 0.407 0.279 0.289 0.005 54.42(-0.80) + WMReward (∇) 0.414 0.279 0.292 0.005 55.02(-0.20) + WMReward(BoN) 0.435 0.324 0.316 0.005 60.34(+5.12) + WMReward(∇+BoN) * 0.439 0.339 0.325 0.005 62.00(+6.78)

Comparing Reward Signals. We compare WMReward against alternative physics plausibility signals based on foundation models such as VideoMAE and Qwen VL (2.5-7B-Instruct and 3-8B-Instruct). For VideoMAE (Tong et al., 2022), a self-supervised video masked autoencoder that learns spatiotemporal representations by reconstructing masked patches in pixel space, we use a surprise score given by the reconstruction error between the predicted pixels and the generated pixels as reward signal (Garrido et al., 2025). For Qwen2.5-VL (Bai et al., 2025) and Qwen3VL (Yang et al., 2025a), following Jang et al. (2025), we pose a physics-plausibility question and request a binary 0/1 answer, then follow practice in (Lin et al., 2024) and use the corresponding positive token logit as the reward.

Table 2 Human evaluation results on PhysicsIQ and VideoPhy. Pairwise human preference judgments across three criteria: Physics Plausibility, Visual Quality, and Prompt Alignment. Winning rate (Win) is the fraction of nonneutral comparisons won, winswins+losses × 100. Accuracy is computed as wins+0.5×neutrals

total × 100 to account for tie. Higher is better, best in bold.

Physics Plausibility

Visual Quality

Prompt Alignment

Overall

Win Acc. Win Acc. Win Acc. Win Acc.

PhysicsIQ vLDM 46.9 45.2 45.3 41.2 48.7 46.8 47.0 44.3

- + WMReward(∇+BoN) 53.1 54.8 54.7 58.8 51.3 53.2 53.0 55.7

MAGI-1 45.1 41.3 47.1 42.9 44.3 35.1 45.5 40.0

- + WMReward(∇+BoN) 54.9 58.7 52.9 57.1 55.7 64.9 54.5 60.0 VideoPhy

vLDM 43.8 38.0 47.4 41.9 51.0 53.3 47.4 43.2

+ WMReward(∇+BoN) 56.2 62.0 52.6 58.1 49.0 46.7 52.6 56.8 MAGI-1 40.7 30.4 42.8 33.3 49.5 48.6 44.3 36.8

+ WMReward(∇+BoN) 59.3 69.6 57.2 66.7 50.5 51.4 55.7 63.2

As shown in Table 1 and Figure 1, WMReward yields stronger performance than alternative signals, indicating that predictive latent-space surprise is a more effective proxy for physics plausibility than pixel reconstruction or VLM judgment, which aligns with recent work suggesting that latent world models exhibit better physics understanding (Garrido et al., 2025). With these results, we show that the physics knowledge in latent world models may be transferred as reward signal to improve video generation.

Human Study. We also supplement our evaluations with a human study to verify the effectiveness of WMReward. We conduct a human study on the full PhysicsIQ benchmark with a side-by-side comparison interface where five annotators view pairs of generated videos along with the original text prompt and additional conditioning information. For each video pair, annotators provide judgments across three criteria: Physics Plausibility, Visual Quality, and Prompt Alignment (further details in Appendix Section B). We obtain two groups of 198 annotations for both vLDM (I2V) and MAGI-1 (V2V) generations, comparing the baseline vanilla sampling strategy against WMReward(∇+BoN). For each criterion, the annotators select their preference among the shown videos or mark them as neutral. Results are aggregated using win rates (excluding neutrals) and accuracy scores to account for ties ((wins + 0.5 × neutrals) / total). Table 2 demonstrates that WMReward delivers significant improvement in all three evaluation criteria, with the winning rate in physics plausibility being most remarkable.

Table 3 Text-conditioned Generation Performance on VideoPhy. Physics Consistency (PC) and Semantic Adherence (SA) are computed with a VLM-based evaluator that measures how well generated videos follow physics laws and text prompts. Higher is better, best is bold. For all search methods, we use 8 particles.

Solid-Solid Solid-Fluid Fluid-Fluid Overall SA PC SA PC SA PC SA PC

VideoCrafter2 (Chen et al., 2024) 50.4 32.2 50.7 27.4 48.1 29.1 50.3 29.7 DreamMachine (Luma AI, 2024) 55.1 21.7 59.6 23.3 58.2 18.2 57.5 21.8 LaVIE (Wang et al., 2025b) 40.8 18.3 48.6 37.0 69.1 50.9 48.7 31.5 HunyuanVideo (Kong et al., 2024) 55.2 16.1 67.1 30.1 54.5 54.5 60.2 28.2

vLDM 46.6 20.7 67.5 28.6 52.4 48.6 56.9 28.0 + WMReward(BoN) 42.7 28.7 59.6 36.3 38.5 50.0 50.3 34.9 + WMReward(∇+BoN) 39.2 25.2 65.1 37.0 50.0 50.0 53.5 34.3

Wan2.2 (Wan et al., 2025) 52.7 13.9 71.9 17.8 67.5 30.2 63.5 18.1 + WMReward(BoN) 43.7 21.8 63.8 15.9 53.8 26.9 55.2 20.9

MAGI-1 (Sand.ai et al., 2025) 42.2 19.0 67.2 27.7 51.9 33.5 54.4 25.0 + WMReward(BoN) 30.8 28.0 59.6 36.3 45.5 36.4 45.3 32.8 + WMReward(∇+BoN) 29.4 28.7 61.6 30.1 40.0 52.7 44.8 33.1

3.2 Text-conditioned Generation

For evaluating the T2V setup, we rely on VideoPhy (Bansal et al., 2024). Following practice in Zhang et al. (2025c) and Wang et al. (2025a), we adopt the automatic VLM-based evaluator, which queries a VLM with templated questions to score semantic adherence (SA) and physics consistency (PC) on a scale of 1-5 for each generated video. We report the per-axis pass rates with each metric larger than 4 over the 344 benchmark prompts. For all methods, we use 8 particles for search.

As shown in Table 3, incorporating WMReward into sampling substantially improves the physics consistency of both MAGI-1 and vLDM by 8.1% and 6.9%, respectively, surpassing all baseline models. We also note that the semantic adherence decreases potentially due to the VJEPA surprise not including semantic information from the text-condition, and to further verify the trade-off between semantic adherence and physics consistency we perform a human study.

Human Study. To complement the VLM-based evaluation, we conduct a human study following the same protocol in Section 3.1 on a subset of 100 VideoPhy generations. As shown in Table 2, human judgement highlights improvements in both physics plausibility and visual quality when using WMReward for T2V generation. The slight decrease in prompt alignment for vLDM is consistent with the text-agnostic nature of the VJEPA’s surprise reward, but the drop is small relative to the gains, yielding a net improvement in the overall metric. We expect this limitation can be mitigated by developing compositional or text-conditioned physics plausibility rewards, an interesting direction for future work.

Table 4 Computation Cost. For each model, the first row shows absolute values (runtime in seconds, memory in GB per GPU). Subsequent rows show overhead multipliers relative to baseline. N is the search budget (number of particles). All experiments on H200 GPUs; MAGI-1 uses 8 GPUs, vLDM uses 1 GPU.

Method Time Memory N vLDM (baseline) 106.77 ± 0.43 s 27.64 GB 1

+ WMReward(BoN) ×N ×1.00 8 + WMReward(∇) ×5.02 ×4.27 1 + WMReward(∇+BoN) ×5.02N ×4.27 8

MAGI-1 (baseline) 265.66 ± 4.49 s 50.03 GB 1 + WMReward(BoN) ×N ×1.00 16 + WMReward(∇) ×4.96 ×2.07 1 + WMReward(∇+BoN) ×4.96N ×2.07 16

3.3 Scaling the Search Space ofWMReward

We now study the scaling behavior of WMReward with the size of the search space. In particular, we scale the number of particles and display results in Figure 1 (right). We observe that, by scaling up the number of particles, there is a steady improvement of PhysicsIQ score. The performance boost is most significant for N ≤4, and steadily increase for larger N. Additionally, as the number of particles increases, the performance variance is smaller and gradually stabilizes. Moreover, as shown in Figure 4, allocating more compute by increasing the number of particles and applying guidance steers the PhysicsIQ score distribution to concentrate in the high-score region. The shift of distributions from N =1 to N =16 shows the effectiveness of WMReward(BoN) by using more particles, while WMReward(∇+BoN) with guidance further sharpening the upper tail of the distribution, confirming its stronger scaling behavior. The consistent scaling behavior of WMReward on PhysicsIQ suggests that by increasing inference compute allocation, we improve the physical plausibility of video generation.

Runtime and Memory. Inference-time alignment methods trade additional computation for improved performance—a recent paradigm shift, from long reasoning traces to test-time search in general. In Table 4, we show the computational overhead induced by WMReward. WMReward(BoN) is parallelized as N vanilla sampling trajectories with the same memory bound, and its memory footprint scales linearly with the number of particles. Guidance introduces extra computation mainly from the gradient backpropagation, and is proportional to the number of steps where guidance is performed through the denoising process. One can choose whether to incorporate first-order information of reward signal

0.018

0.014

MAGI-1 V2V

vLDM I2V

BoN@1

BoN@1

0.015

0.012

+BoN@1

+BoN@1

BoN@16

BoN@16

0.013

0.010

+BoN@16

+BoN@16

Density

Density

0.010

0.008

0.007

0.006

0.005

0.004

0.003

0.002

0.000

0.000

0 20 40 60 80 100

0 20 40 60 80 100

PhysicsIQ Final Score

PhysicsIQ Final Score

- Figure 4 Improving Performance via Particle Scaling and Guidance. Visualization of PhysicsIQ score distributions with a Gaussian KDE for MAGI-1 V2V (left) and vLDM I2V (right) generations. Scaling the number of particles (here, 1 vs.

16) yields substantial gains for Best-of-N, and adding guidance further sharpens the distribution toward higher physics plausibility. This leads to overall higher average score shown as dashed vertical line.

Table 5 General Visual Quality with VBench Evaluators. Visual quality metrics computed on videos generated for the PhysicsIQ and VideoPhy datasets. Higher is better for all metrics.

Subject Consistency

Background Consistency

Motion Smoothness

Temporal Flickering

Imaging Quality

Aesthetic Quality

PhysicsIQ

vLDM 94.68 95.60 99.57 99.48 67.47 48.94 + WMReward(∇) 94.64 95.58 99.56 99.47 67.50 48.95 + WMReward(BoN) 95.29 95.89 99.62 99.61 67.71 49.19 + WMReward(∇+BoN) 95.25 95.89 99.62 99.61 67.83 49.03

MAGI-1 95.82 97.25 99.70 99.82 69.07 47.05 + WMReward(∇) 95.90 97.25 99.70 99.82 69.15 47.05 + WMReward(BoN) 95.87 97.30 99.71 99.84 69.24 46.88 + WMReward(∇+BoN) 95.88 97.30 99.71 99.85 69.25 46.94

VideoPhy

vLDM 93.73 95.79 98.57 97.80 60.34 49.71 + WMReward(∇) 93.72 95.78 98.57 97.80 60.05 49.69 + WMReward(BoN) 94.82 96.31 98.69 98.02 59.44 49.54 + WMReward(∇+BoN) 94.67 96.36 98.72 98.13 60.03 49.87

MAGI-1 94.43 95.87 98.80 98.08 56.15 46.39 + WMReward(∇) 94.58 95.92 98.81 98.12 56.17 46.46 + WMReward(BoN) 96.27 96.68 98.92 98.44 58.15 47.54 + WMReward(∇+BoN) 96.54 96.87 98.97 98.54 57.84 47.73

as budget allows. In general, WMReward can be adapted to the available computation budget with appropriate sampling methods, offering a spectrum of compute-performance tradeoffs.

3.4 Analysis and Ablation

Having established that WMReward achieves strong improvements in physics plausibility and scales effectively with computational budget, we now investigate several key questions about the method’s properties.

Does WMReward impair perceptual qualities? To assess whether physics improvements come at the cost of general perceptual quality, we use the VBench evaluators (Huang et al., 2024) to rate generated PhysicsIQ and VideoPhy videos across six key visual quality metrics: subject consistency, background consistency, motion smoothness, temporal flickering, imaging quality, and aesthetic quality. As shown

Table 6 Comparison of Sampling Methods. vLDM I2V PhysicsIQ generation under different sampling schemes. Columns indicate number of particles (N), showing scaling effects.

2 4 8 16

WMReward(SMC) 27.14 26.96 28.12 29.24 WMReward(SVDD) 27.31 28.57 28.54 28.87 WMReward(BoN) 27.76 31.87 32.32 32.90 WMReward(∇+BoN) 27.88 31.05 32.62 33.44

in Table 5, image quality and aesthetic quality tend to show small improvements, possibly due to the stronger physics plausibility achieved by WMReward. This trend aligns with the human-preference results in Table 2. We also observe improved temporal consistency, motion smoothness, and temporal flickering, suggesting that by following physics principles, such as artifact suppression, and adherence to mass conservation and continuity, the overall perceptual quality of generated videos are also improved.

How does WMReward perform with other sampling methods? In principle, one can use any valid sampling schemes to sample from the WMReward tilted distribution. We further demonstrate the compatibility of the proposed reward by alternative sampling scheme with SMC (Singhal et al., 2025) and SVDD (Li et al., 2024a) (detailed in Section 4) using vLDM on the PhysicsIQ benchmark. Empirically, both SMC and SVDD deliver gains over vanilla sampling, yet, they still underperform relative to BoN and ∇ + BoN, as demonstrated in Table 6. Intuitively, because VJEPA-2 surprise is differentiable, its gradients provide step-wise guidance that plays a role similar to, and often more accurate than, the multi-rollout importance estimates used by SVDD. Also, SMC shows smaller gains, likely because early-stage reward estimates are noisy, causing resampling to collapse onto suboptimal seeds. Practically, SMC also requires multiple parallel denoising trajectories per sample

###### Table 7 Robustness of V-JEPA Surprise Design. Comparison showing VJEPA architecture and reward hyperparameter against WMReward(BoN) results. We report results using 16 particles.

Arch Size Window Context Stride FPS Final Score↑

ViT-giant 32 16 16 24 60.78 ViT-giant 16 8 8 16 60.34 ViT-giant 16 8 8 24 60.09 ViT-giant 32 16 8 24 60.05 ViT-huge 48 24 8 24 60.04 ViT-giant 16 8 4 24 59.91 ViT-huge 16 8 8 24 59.77 ViT-huge 16 8 4 24 59.62 ViT-huge 32 16 16 24 57.84 ViT-huge 32 16 8 24 57.09

for resampling, which increases the memory footprint and is less trivial to parallelize than BoN search. For these reasons, we adopt WMReward(∇+BoN) as a more practically effective sampling scheme.

How robust is the WMReward to VJEPA size and hyper-parameters? One key design factor in our experiments is the choice of hyperparameters for the VJEPA surprise reward r(·) in Equation (6). We study how robust this reward is when transferring physics understanding to video generation. As shown in Table 7, such transfer remains relatively stable across the context length C, prediction horizon M, and stride s. We also observe that physics plausibility gains scale with the size of the reward model (from ViT-huge to ViT-giant), suggesting that stronger VJEPA backbones yield better performance without any fine-tuning of the underlying video generator.

- 4 Related Work

Video Diffusion Models. Diffusion and flow-based video generative models (Sand.ai et al., 2025; Brooks et al., 2024; OpenAI, 2025; Kondratyuk et al., 2024; Bar-Tal et al., 2024; Wan et al., 2025) learn to generate video sequences by reversing a noising process on temporal visual data. While they achieve promising visual quality in generation, they still often produce physically implausible videos (Bansal et al., 2024, 2025; Motamed et al., 2025; Yuan et al., 2025a). In order to mitigate this shortcoming, there has been substantial work devising pre-training and post-training approaches focused on explicitly injecting physics constraints (Yuan et al., 2025b), enabling motion prediction (Chefer et al., 2025), enhancing dynamics (Li et al., 2024b) or leveraging learned physics priors from foundation models through model finetuning or distillation (Cao et al., 2024; Zhang et al., 2025c,a). Yet, another relatively underexplored line of work

has focused on devising inference-time methods to improve physics. In particular, Xue et al. (2025) explore the manifold learned by the generative model through prompt rewriting and Yang et al. (2025b) leverage VLMs to perform motion planning on top of a motion-controllable video generative model (Burgert et al., 2025).

Physics Understanding in Vision Models. Physics understanding is essential to a model’s ability to reason and predict scenes under physical laws (Bear et al., 2021; Riochet et al., 2018; Bordes et al., 2025), and in particular, it is important to ensure physics plausibility in video generation (Motamed et al., 2025; Yuan et al., 2025a). Recent work (Bordes et al., 2025) has investigated the intuitive physics understanding of vision models, including VLMs (Bai et al., 2025; Comanici et al., 2025), generative models (Agarwal et al., 2025), and self-supervised learning (SSL) models (Tong et al., 2022; Wang et al., 2023; Assran et al., 2025), showing that video SSL approaches based on the Joint-Embedding-Predictive-Architecture (JEPA) (Bardes et al., 2024; Assran et al., 2025) exhibit the highest performance among predictive methods. Moreover, intuitive physics properties such as object permanence and shape consistency have been shown to emerge in the video JEPA (VJEPA) models (Garrido et al., 2025). VJEPA consists of a video encoder and a predictor trained on Internetscale data with a masked predictive objective in the representation space. Predicting the future in the representation space reduces the sensitivity to appearance details and holds the potential to emphasize other aspects such as dynamics and interaction.

Inference-Time Alignment for Image Diffusion Models.

Inference-time alignment steers a pre-trained diffusion model toward desired properties during generation without retraining. On the one hand, there are derivative-free approaches (Singhal et al., 2025; Wu et al., 2023; Lee et al., 2025; Li et al., 2024a, 2025; Jain et al., 2025; Park et al., 2025). Among those, BoN selects the highest-scoring candidates at the end of the denoising process (Ma et al., 2025). SMC-based approaches (Singhal et al., 2025; Wu et al., 2023) assign importance weights to multiple denoising trajectories and perform resampling during generation. Value-based importance sampling methods, such as SVDD (Li et al., 2024a), steer a single trajectory by resampling according to the local step reward. On the other hand, we have gradient-based approaches that adjust the score function (Song et al., 2020b; Bansal et al., 2023; Ye et al., 2024; Hemmat et al., 2023; Askari Hemmat et al., 2024; Dall’Asen et al., 2025; Askari-Hemmat et al., 2025), steering the denoising process toward high reward regions– resulting

in e.g. , high utility or high diversity samples. These assume access to differentiable reward functions. Despite their different implementations, these methods share a similar principle of using reward signals to refine local steps or enable better global exploration.

- 5 Conclusion and Discussion

In this work, we presented WMReward, an inference-time alignment method to improve physics plausibility of video generative models, and highlighted the viability of using latent world models (i.e. , VJEPA-2) as reward signal. Throughout our experimentation, we demonstrated the potential of leveraging latent world model’s to improve physics in video generation, yielding substantial improvements on the challenging PhysicsIQ and VideoPhy benchmarks without requiring any further training. There are three potential avenues for improving this line of work. (1) Improving video generative models: The quality of video generative model determines the set of potential solutions that can be explored with our method; developing stronger and more searchable models is a promising avenue. (2) Improving reward models: The strength of the reward model heavily influences the overall performance of the system. Although VJEPA-2 is already informative, its training data is currently limited and as a result the model is not able to cover all physics phenomena, e.g. , we observe that material understanding such as weight and friction is limited. Future work on latent world models could provide broader understanding of physical phenomena. (3) Improving search algorithms: Better and more efficient search algorithms offer a promising avenue to improve performance. Early in the diffusion process, the intermediate estimated video samples x0|t are blurry, potentially leading to unreliable predictions in the reward.

References

Niket Agarwal, Arslan Ali, Maciej Bala, Yogesh Balaji, Erik Barker, Tiffany Cai, Prithvijit Chattopadhyay, Yongxin Chen, Yin Cui, Yifan Ding, et al. Cosmos world foundation model platform for physical ai. arXiv preprint arXiv:2501.03575, 2025.

Reyhane Askari Hemmat, Melissa Hall, Alicia Sun, Candace Ross, Michal Drozdzal, and Adriana RomeroSoriano. Improving geo-diversity of generated images with contextualized vendi score guidance. In European Conference on Computer Vision, pages 213–229. Springer, 2024.

Reyhane Askari-Hemmat, Mohammad Pezeshki, Elvis Dohmatob, Florian Bordes, Pietro Astolfi, Melissa Hall, Jakob Verbeek, Michal Drozdzal, and Adriana RomeroSoriano. Improving the scaling laws of synthetic data with deliberate practice. In ICML, 2025.

Mido Assran, Adrien Bardes, David Fan, Quentin Garrido, Russell Howes, Mojtaba, Komeili, Matthew Muckley, Ammar Rizvi, Claire Roberts, Koustuv Sinha, Artem Zholus, Sergio Arnaud, Abha Gejji, Ada Martin, Francois Robert Hogan, Daniel Dugas, Piotr Bojanowski, Vasil Khalidov, Patrick Labatut, Francisco Massa, Marc Szafraniec, Kapil Krishnakumar, Yong Li, Xiaodong Ma, Sarath Chandar, Franziska Meier, Yann LeCun, Michael Rabbat, and Nicolas Ballas. V-jepa 2: Self-supervised video models enable understanding, prediction and planning, 2025. https://arxiv.org/abs/2506.09985.

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.

Arpit Bansal, Hong-Min Chu, Avi Schwarzschild, Soumyadip Sengupta, Micah Goldblum, Jonas Geiping, and Tom Goldstein. Universal guidance for diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 843–852, 2023.

Hritik Bansal, Zongyu Lin, Tianyi Xie, Zeshun Zong, Michal Yarom, Yonatan Bitton, Chenfanfu Jiang, Yizhou Sun, Kai-Wei Chang, and Aditya Grover. Videophy: Evaluating physical commonsense for video generation. arXiv preprint arXiv:2406.03520, 2024.

Hritik Bansal, Clark Peng, Yonatan Bitton, Roman Goldenberg, Aditya Grover, and Kai-Wei Chang. Videophy2: A challenging action-centric physical commonsense evaluation in video generation. arXiv preprint arXiv:2503.06800, 2025.

Omer Bar-Tal, Hila Chefer, Omer Tov, Charles Herrmann, Roni Paiss, Shiran Zada, Ariel Ephrat, Junhwa Hur, Guanghui Liu, Amit Raj, Yuanzhen Li, Michael Rubinstein, Tomer Michaeli, Oliver Wang, Deqing Sun, Tali Dekel, and Inbar Mosseri. Lumiere:

A space-time diffusion model for video generation, 2024. https://arxiv.org/abs/2401.12945.

Adrien Bardes, Quentin Garrido, Jean Ponce, Xinlei Chen, Michael Rabbat, Yann LeCun, Mahmoud Assran, and Nicolas Ballas. Revisiting feature prediction for learning visual representations from video. arXiv preprint arXiv:2404.08471, 2024.

Daniel M Bear, Elias Wang, Damian Mrowca, Felix J Binder, Hsiao-Yu Fish Tung, RT Pramod, Cameron Holdaway, Sirui Tao, Kevin Smith, Fan-Yun Sun, et al. Physion: Evaluating physical prediction from vision in humans and machines. arXiv preprint arXiv:2106.08261, 2021.

Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, Varun Jampani, and Robin Rombach. Stable video diffusion: Scaling latent video diffusion models to large datasets, 2023. https://arxiv.org/abs/2311.15127.

Florian Bordes, Quentin Garrido, Justine T Kao, Adina Williams, Michael Rabbat, and Emmanuel Dupoux. Intphys 2: Benchmarking intuitive physics understanding in complex synthetic environments. arXiv preprint arXiv:2506.09849, 2025.

Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, Clarence Ng, Ricky Wang, and Aditya Ramesh. Video generation models as world simulators, 2024. https://openai.com/research/ video-generation-models-as-world-simulators.

Ryan Burgert, Yuancheng Xu, Wenqi Xian, Oliver Pilarski, Pascal Clausen, Mingming He, Li Ma, Yitong Deng, Lingxiao Li, Mohsen Mousavi, et al. Go-withthe-flow: Motion-controllable video diffusion models using real-time warped noise. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 13–23, 2025.

Qinglong Cao, Ding Wang, Xirui Li, Yuntian Chen, Chao Ma, and Xiaokang Yang. Teaching video diffusion model with latent physical phenomenon knowledge. arXiv preprint arXiv:2411.11343, 2024.

Hila Chefer, Uriel Singer, Amit Zohar, Yuval Kirstain, Adam Polyak, Yaniv Taigman, Lior Wolf, and Shelly Sheynin. VideoJAM: Joint appearance-motion representations for enhanced motion generation in video models. In Forty-second International Conference on Machine Learning, 2025. https://openreview.net/ forum?id=yMJcHWcb2Z.

Haoxin Chen, Yong Zhang, Xiaodong Cun, Menghan Xia, Xintao Wang, Chao Weng, and Ying Shan. Videocrafter2: Overcoming data limitations for highquality video diffusion models, 2024.

Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Mar-

cel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.

Nicola Dall’Asen, Xiaofeng Zhang, Reyhane Askari Hemmat, Melissa Hall, Jakob Verbeek, Adriana RomeroSoriano, and Michal Drozdzal. Increasing the utility of synthetic images through chamfer guidance. arXiv preprint arXiv:2508.10631, 2025.

Siddhartha Datta, Alexander Ku, Deepak Ramachandran, and Peter Anderson. Prompt expansion for adaptive text-to-image generation. In In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2024.

Bradley Efron. Tweedie’s formula and selection bias. Journal of the American Statistical Association, 106

(496):1602–1614, 2011.

Quentin Garrido, Nicolas Ballas, Mahmoud Assran, Adrien Bardes, Laurent Najman, Michael Rabbat, Emmanuel Dupoux, and Yann LeCun. Intuitive physics understanding emerges from self-supervised pretraining on natural videos. arXiv preprint arXiv:2502.11831, 2025.

Reyhane Askari Hemmat, Mohammad Pezeshki, Florian Bordes, Michal Drozdzal, and Adriana Romero-Soriano. Feedback-guided data synthesis for imbalanced classification. arXiv preprint arXiv:2310.00158, 2023.

Anthony Hu, Lloyd Russell, Hudson Yeo, Zak Murez, George Fedoseev, Alex Kendall, Jamie Shotton, and Gianluca Corrado. Gaia-1: A generative world model for autonomous driving. arXiv preprint arXiv:2309.17080, 2023.

Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21807–21818, 2024.

Tariq Berrada Ifriqi, Adriana Romero-Soriano, Michal Drozdzal, Jakob Verbeek, and Karteek Alahari. Entropy rectifying guidance for diffusion and flow models, 2025.

Vineet Jain, Kusha Sareen, Mohammad Pedramfar, and Siamak Ravanbakhsh. Diffusion tree sampling: Scalable inference-time alignment of diffusion models. arXiv preprint arXiv:2506.20701, 2025.

Joel Jang, Seonghyeon Ye, Zongyu Lin, Jiannan Xiang, Johan Bjorck, Yu Fang, Fengyuan Hu, Spencer Huang, Kaushil Kundalia, Yen-Chen Lin, et al. Dreamgen: Unlocking generalization in robot learning through video world models. arXiv preprint arXiv:2505.12705, 2025.

Bingyi Kang, Yang Yue, Rui Lu, Zhijie Lin, Yang Zhao, Kaixin Wang, Gao Huang, and Jiashi Feng. How far is video generation from world model: A physical law perspective. arXiv preprint arXiv:2411.02385, 2024.

Dan Kondratyuk, Lijun Yu, Xiuye Gu, José Lezama, Jonathan Huang, Grant Schindler, Rachel Hornung, Vighnesh Birodkar, Jimmy Yan, Ming-Chang Chiu, Krishna Somandepalli, Hassan Akbari, Yair Alon, Yong Cheng, Josh Dillon, Agrim Gupta, Meera Hahn, Anja Hauth, David Hendon, Alonso Martinez, David Minnen, Mikhail Sirotenko, Kihyuk Sohn, Xuan Yang, Hartwig Adam, Ming-Hsuan Yang, Irfan Essa, Huisheng Wang, David A. Ross, Bryan Seybold, and Lu Jiang. Videopoet: A large language model for zeroshot video generation, 2024. https://arxiv.org/abs/ 2312.14125.

Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024.

Kuaishou. Kling video model, 2024. https://kling. kuaishou.com/en.

Yann LeCun. A path towards autonomous machine intelligence version 0.9. 2, 2022-06-27. Open Review, 62(1): 1–62, 2022.

Gyubin Lee, Truong Nhat Nguyen Bao, Jaesik Yoon, Dongwoo Lee, Minsu Kim, Yoshua Bengio, and Sungjin Ahn. Adaptive cyclic diffusion for inference scaling. arXiv preprint arXiv:2505.14036, 2025.

Xiner Li, Yulai Zhao, Chenyu Wang, Gabriele Scalia, Gokcen Eraslan, Surag Nair, Tommaso Biancalani, Shuiwang Ji, Aviv Regev, Sergey Levine, et al. Derivativefree guidance in continuous and discrete diffusion models with soft value-based decoding. arXiv preprint arXiv:2408.08252, 2024a.

Xiner Li, Masatoshi Uehara, Xingyu Su, Gabriele Scalia, Tommaso Biancalani, Aviv Regev, Sergey Levine, and Shuiwang Ji. Dynamic search for inferencetime alignment in diffusion models. arXiv preprint arXiv:2503.02039, 2025.

Zhengqi Li, Richard Tucker, Noah Snavely, and Aleksander Holynski. Generative image dynamics. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 24142– 24153, June 2024b.

Zhiqiu Lin, Deepak Pathak, Baiqi Li, Jiayao Li, Xide Xia, Graham Neubig, Pengchuan Zhang, and Deva Ramanan. Evaluating text-to-visual generation with image-to-text generation. In European Conference on Computer Vision, pages 366–384. Springer, 2024.

Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. In International Conference

on Learning Representations (ICLR), 2023. https: //arxiv.org/abs/2210.02747.

Luma AI. Dream machine | ai video generator. https: //lumalabs.ai/dream-machine, 2024.

Ge Ya Luo, Gian Mario Favero, ZhiHao Luo, Alexia Jolicoeur-Martineau, and Christopher Pal. Beyond FVD: An enhanced evaluation metrics for video generation distribution quality. In The Thirteenth International Conference on Learning Representations (ICLR), 2025.

Nanye Ma, Shangyuan Tong, Haolin Jia, Hexiang Hu, Yu-Chuan Su, Mingda Zhang, Xuan Yang, Yandong Li, Tommi Jaakkola, Xuhui Jia, et al. Inference-time scaling for diffusion models beyond scaling denoising steps. arXiv preprint arXiv:2501.09732, 2025.

Saman Motamed, Laura Culp, Kevin Swersky, Priyank Jaini, and Robert Geirhos. Do generative video models understand physical principles?, 2025. https://arxiv. org/abs/2501.09038.

OpenAI. Sora 2 system card. https://cdn.openai.com/

pdf/50d5973c-c4ff-4c2d-986f-c72b5d0ff069/sora_2_ system_card.pdf, 2025. Accessed: January 21, 2026.

Byeongjun Park, Hyojun Go, Hyelin Nam, Byung-Hoon Kim, Hyungjin Chung, and Changick Kim. Steerx: Creating any camera-free 3d and 4d scenes with geometric steering. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 27326–27337, 2025.

Pika Labs. Pika 1.0, 2023.

Adam Polyak, Amit Zohar, Andrew Brown, Andros Tjandra, Animesh Sinha, Ann Lee, Apoorv Vyas, Bowen Shi, Chih-Yao Ma, Ching-Yao Chuang, et al. Movie gen: A cast of media foundation models. arXiv preprint arXiv:2410.13720, 2024.

Ronan Riochet, Mario Ynocente Castro, Mathieu Bernard, Adam Lerer, Rob Fergus, Véronique Izard, and Emmanuel Dupoux. Intphys: A framework and benchmark for visual intuitive physics reasoning. arXiv preprint arXiv:1803.07616, 2018.

Herbert E Robbins. An empirical bayes approach to statistics. In Breakthroughs in Statistics: Foundations and basic theory, pages 388–394. Springer, 1992.

Runway. Gen-3 Alpha, 2024.

Sand.ai, Hansi Teng, Hongyu Jia, Lei Sun, Lingzhi Li, Maolin Li, Mingqiu Tang, Shuai Han, Tianning Zhang, W. Q. Zhang, Weifeng Luo, Xiaoyang Kang, Yuchen Sun, Yue Cao, Yunpeng Huang, Yutong Lin, Yuxin Fang, Zewei Tao, Zheng Zhang, Zhongshu Wang, Zixun Liu, Dai Shi, Guoli Su, Hanwen Sun, Hong Pan, Jie Wang, Jiexin Sheng, Min Cui, Min Hu, Ming Yan, Shucheng Yin, Siran Zhang, Tingting Liu, Xianping Yin, Xiaoyu Yang, Xin Song, Xuan Hu, Yankai Zhang,

and Yuqiao Li. Magi-1: Autoregressive video generation at scale, 2025. https://arxiv.org/abs/2505.13211.

Raghav Singhal, Zachary Horvitz, Ryan Teehan, Mengye Ren, Zhou Yu, Kathleen McKeown, and Rajesh Ranganath. A general framework for inference-time scaling and steering of diffusion models. arXiv preprint arXiv:2501.06848, 2025.

Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020a.

Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020b.

Zhan Tong, Yibing Song, Jue Wang, and Limin Wang. Videomae: Masked autoencoders are data-efficient learners for self-supervised video pre-training. Advances in neural information processing systems, 35: 10078–10093, 2022.

Masatoshi Uehara, Yulai Zhao, Chenyu Wang, Xiner Li, Aviv Regev, Sergey Levine, and Tommaso Biancalani. Inference-time alignment in diffusion models with reward-guided generation: Tutorial and review. arXiv preprint arXiv:2501.09685, 2025.

Pascal Vincent. A connection between score matching and denoising autoencoders. Neural Computation, 23

(7):1661–1674, 2011.

Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, Jianyuan Zeng, Jiayu Wang, Jingfeng Zhang, Jingren Zhou, Jinkai Wang, Jixuan Chen, Kai Zhu, Kang Zhao, Keyu Yan, Lianghua Huang, Mengyang Feng, Ningyi Zhang, Pandeng Li, Pingyu Wu, Ruihang Chu, Ruili Feng, Shiwei Zhang, Siyang Sun, Tao Fang, Tianxing Wang, Tianyi Gui, Tingyu Weng, Tong Shen, Wei Lin, Wei Wang, Wei Wang, Wenmeng Zhou, Wente Wang, Wenting Shen, Wenyuan Yu, Xianzhong Shi, Xiaoming Huang, Xin Xu, Yan Kou, Yangyu Lv, Yifei Li, Yijing Liu, Yiming Wang, Yingya Zhang, Yitong Huang, Yong Li, You Wu, Yu Liu, Yulin Pan, Yun Zheng, Yuntao Hong, Yupeng Shi, Yutong Feng, Zeyinzi Jiang, Zhen Han, Zhi-Fan Wu, and Ziyu Liu. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

Jing Wang, Ao Ma, Ke Cao, Jun Zheng, Zhanjie Zhang, Jiasong Feng, Shanyuan Liu, Yuhang Ma, Bo Cheng, Dawei Leng, et al. Wisa: World simulator assistant for physics-aware text-to-video generation. arXiv preprint arXiv:2503.08153, 2025a.

Limin Wang, Bingkun Huang, Zhiyu Zhao, Zhan Tong, Yinan He, Yi Wang, Yali Wang, and Yu Qiao. Videomae v2: Scaling video masked autoencoders with dual

masking. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 14549–14560, 2023.

Yaohui Wang, Xinyuan Chen, Xin Ma, Shangchen Zhou, Ziqi Huang, Yi Wang, Ceyuan Yang, Yinan He, Jiashuo Yu, Peiqing Yang, et al. Lavie: High-quality video generation with cascaded latent diffusion models. International Journal of Computer Vision, 133(5): 3059–3078, 2025b.

Luhuan Wu, Brian Trippe, Christian Naesseth, David Blei, and John P Cunningham. Practical and asymptotically exact conditional sampling in diffusion models. Advances in Neural Information Processing Systems, 36:31372–31403, 2023.

Qiyao Xue, Xiangyu Yin, Boyuan Yang, and Wei Gao. Phyt2v: Llm-guided iterative self-refinement for physics-grounded text-to-video generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 18826–18836, June 2025.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025a.

Mengjiao Yang, Yilun Du, Kamyar Ghasemipour, Jonathan Tompson, Dale Schuurmans, and Pieter Abbeel. Learning interactive real-world simulators. arXiv preprint arXiv:2310.06114, 1(2):6, 2023.

Xindi Yang, Baolu Li, Yiming Zhang, Zhenfei Yin, Lei Bai, Liqian Ma, Zhiyong Wang, Jianfei Cai, Tien-Tsin Wong, Huchuan Lu, et al. Vlipp: Towards physically plausible video generation with vision and language informed physical prior. arXiv preprint arXiv:2503.23368, 2025b.

Haotian Ye, Haowei Lin, Jiaqi Han, Minkai Xu, Sheng Liu, Yitao Liang, Jianzhu Ma, James Y Zou, and Stefano Ermon. Tfg: Unified training-free guidance for diffusion models. Advances in Neural Information Processing Systems, 37:22370–22417, 2024.

Jianhao Yuan, Fabio Pizzati, Francesco Pinto, Lars Kunze, Ivan Laptev, Paul Newman, Philip Torr, and Daniele De Martini. Likephys: Evaluating intuitive physics understanding in video diffusion models via likelihood preference. arXiv preprint arXiv:2510.11512, 2025a.

Yu Yuan, Xijun Wang, Tharindu Wickremasinghe, Zeeshan Nadir, Bole Ma, and Stanley H Chan. Newtongen: Physics-consistent and controllable text-to-video generation via neural newtonian dynamics. arXiv preprint arXiv:2509.21309, 2025b.

Ke Zhang, Cihan Xiao, Yiqun Mei, Jiacong Xu, and Vishal M Patel. Think before you diffuse: Llmsguided physics-aware video generation. arXiv preprint arXiv:2505.21653, 2025a.

Xiangcheng Zhang, Haowei Lin, Haotian Ye, James Zou, Jianzhu Ma, Yitao Liang, and Yilun Du. Inferencetime scaling of diffusion models through classical search. arXiv preprint arXiv:2505.23614, 2025b.

Xiangdong Zhang, Jiaqi Liao, Shaofeng Zhang, Fanqing Meng, Xiangpeng Wan, Junchi Yan, and Yu Cheng. Videorepa: Learning physics for video generation through relational alignment with foundation models. arXiv preprint arXiv:2505.23656, 2025c.

Xiaofeng Zhang, Aaron Courville, Michal Drozdzal, and Adriana Romero-Soriano. The intricate dance of prompt complexity, quality, diversity, and consistency in t2i models, 2025d. https://arxiv.org/abs/2510. 19557.

## Appendix

- A Implementation Details

In the following, we touch on the implementation details of WMReward, including generation settings and adaptation to different generation paradigms.

A.1 Generation Settings

For all experiments, we use a vLDM transformer with a spatiotemporal VAE for compression and text–video alignment, and MAGI-1-24B (Sand.ai et al., 2025), an autoregressive diffusion video model that generates videos chunk-by-chunk using block-causal attention for long-horizon consistency. Their corresponding generation hyperparameters are as follows.

Table 8 Generation hyperparameters.

VideoPhy PhysicsIQ

Hyperparameter MAGI-1 vLDM MAGI-1 (I2V) MAGI-1 (V2V) vLDM

Height 480 480 720 720 480 Width 832 720 1280 1280 720 Number of frames 48 49 120 120 49 FPS 24 8 24 24 8 Number of steps 16 50 32 32 50 CFG scale 7.5 6.0 7.5 7.5 6.0 Context guidance scale 1.5 – 1.5 1.5 – guidance frequency 3 3 5 3 1 VJEPA guidance scale 0.005 0.003 0.005 0.005 0.001

For generation resolution, FPS, we use the recommended settings from the official video generative model repository. Also, for CFG and other guidance implemented in MAGI-1, we follow the default settings in official codebase. The number of generated frames varies according to the specification of the evaluation dataset. For PhysicsIQ, the generated video is required to be exactly 5 seconds. Thus, we generate 49 frames with vLDM and trim them to 40 frames under 8 FPS. For MAGI-1, we generate 120 frames under 24 FPS. For VideoPhy, while there is no explicit requirement on duration and number of frames, we follow the paradigm in the official code to generate relative short video clips with the specification shown. The guidance frequency indicates the time-step interval in which we apply guidance during the denoising process. For vLDM, we use a DDIM (Song et al., 2020a) scheduler. For MAGI-1, we use standard linear rectified flow

- sampler. For MAGI-1, we use the distilled 24B checkpoint, and run the inference on 8 H200 GPUs in parallel. For vLDM, we run the inference on a single H200 GPU. Also, for sampling with WMReward, we use a VJEPA2 ViT-giant model. The input frame size is 256 × 256. We choose window size, context length, and stride to be 16, 8, 8 for all experiments.

- A.2 Adaptation to Different Generation Paradigms

Current video diffusion models follow two predominant paradigms: holistic generation (Wan et al., 2025; Agarwal et al., 2025), which denoises all frames simultaneously at the same noise level, and autoregressive generation (Sand.ai et al., 2025), which generates videos sequentially in temporal chunks, each with its own noise schedule. We implement WMReward on models based on both paradigms (vLDM and MAGI1). Practically, in both cases, the BoN search implementation is the same, while the implementation of guidance (∇) varies. For holistic generation, we use the the combined CFG and WMReward guidance in Equation (12).

∇xt

log pλ xt | txt = (1 − ωtxt)∇xt

log p(xt)

+ ωtxt ∇xt

log p xt | txt − ωs∇xt

r xt | txt .

(12)

Specifically, we adopt a sliding window approach to split the video into context and prediction target chunks, compute the VJEPA surprise on each window, and average over the whole sequence. For autoregressive generation, we perform guidance as follows:

∇xt

log pλ xt | x<kt ,txt = (1 − ω<k)∇xt

log p(xt)

+ (ω<k − ωtxt)∇xt

log p xt | x<kt

+ ωtxt∇xt

log p xt | x<kt ,txt − ωs∇xt

r xt | x<kt ,txt ,

(13) where we combine VJEPA surprise guidance with classifier-free guidance from both text and previous denoised chunks x<kt . In particular, we use previous denoised chunks as context for the VJEPA predictor, predict the next chunk, and calculate VJEPA’s surprise reward.

- A.3 VLM-based Reward Model Details

For VLM-based reward models, we use Qwen2.5-VL7B-Instruct (Bai et al., 2025) and Qwen3-VL-8BInstruct (Yang et al., 2025a), respectively. We use a question template "Does the video show good physics dynamics and showcase a good alignment with the physical world? Please be a strict judge. If it breaks the laws of physics, please answer 0. Answer 0 for No or 1 for Yes. Reply only 0 or 1.". Then, we extract the logit of token "1" and its variation " 1" as the reward signal.

- B Human Study Details

As shown in Figure 5, annotators are presented with a side-by-side comparison interface where they view two generated videos along with the original text prompt and conditioning frames describing the physical scenario. For each video pair, annotators provide judgments across three criteria: (1) Physics Plausibility, assessing whether the physical interactions and dynamics are realistic; (2) Visual Quality, evaluating the overall visual fidelity, clarity, and aesthetics of the generated video; and (3) Prompt Alignment, measuring how well the video content matches the given text description. For each criterion, the annotators select one of three options: preferring the video sample on the left, preferring the video sample on the right, or reporting a neutral preference when the difference is negligible or both videos are equally good/bad. To mitigate position bias, videos from each model are randomly assigned to the left or right positions. We collect evaluations from five annotators. Results are aggregated to compute win rates (percentage of comparisons where a model is preferred, excluding neutral judgments) and accuracy scores (computed as wins+0.5×neutrals

total ), providing a comprehensive assessment of relative model performance.

- C Qualitative Examples

We present additional qualitative visual samples to demonstrate the effectiveness of WMReward. As shown in Figures 6 to 8 for image- and multiframe-conditioned generation, and in Figure 9 for text-conditioned generation, we observe that our generated videos exhibit improved physics plausibility across spatial continuity, rigid-body dynamics, fluid behavior, buoyancy, temporal continuity, gravity, conservation of mass, and optical effects. These

- samples indicate that the latent world model contains non-trivial physics understanding that enables more physically plausible video generation. For a better view of the dynamics, we recommend viewing the videos attached in the supplementary material zip file.

- D Failure Mode Analysis

One of our core assumptions is that the latent world model VJEPA-2 captures stronger intuitive physics than current video generators (Garrido et al., 2025). However, this learned prior remains a proxy for true physics dynamics. VJEPA surprise is not exclusively measuring physics plausibility and entangles other perceptual factors. Practically, as shown in Figure 10,

we observe that sampling with VJEPA reward in some cases does not lead to substantial physics plausibility improvements. For example, the model fails to capture abrupt physical events, such as fluid overflowing from a bottle or a lit match igniting a balloon and causing it to explode, which require reasoning about sudden state changes. It sometimes struggles with more complex phenomena, including mirror reflections and siphon effects, which demand more complex reasoning and understanding of material properties. While VJEPA reward can correct some physics violations (e.g. , conservation-of-mass errors in certain siphon scenarios), these failures indicate that there remains room to improve the physics understanding of latent world models in order to obtain more reliable reward signals and, consequently, better physics-aware video generation.

[Figure 8]

###### Figure 5 Human Study Interface. Annotators view a side-by-side video comparison and indicate their preference on three criteria—Physics Plausibility, Visual Quality, and Prompt Alignment—choosing one of three preference options: Left, Right, or Neutral.

[Figure 9]

###### Figure 6 Additional Qualitative Results on Physics-IQ.

[Figure 10]

###### Figure 7 Additional Qualitative Samples on Physics-IQ.

[Figure 11]

###### Figure 8 Additional Qualitative Samples on Physics-IQ.

[Figure 12]

###### Figure 9 Additional Qualitative Samples on VideoPhy.

[Figure 13]

- Figure 10 Failure Mode Analysis. We observe some failure modes that persist even when leveraging VJEPA-2 for sampling. For example, the model often fails on abrupt physical events, such as fluid overflowing from a bottle (top left quadrant) or a lit match igniting a balloon and causing it to explode (bottom left quadrant). It also struggles with more complex phenomena that requires reasoning and understanding of material properties, including mirror reflections (top right) and siphon effects (bottom right), indicating that both the base model and the reward model still have room of improvement on physics understanding.

