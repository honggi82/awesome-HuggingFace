# arXiv:2505.07818v4[cs.CV]28Aug2025

[Figure 1]

[Figure 2]

## DanceGRPO: Unleashing GRPO on Visual Generation

### Zeyue Xue1,2, Jie Wu1‡, Yu Gao1, Fangyuan Kong1, Lingting Zhu2, Mengzhao Chen2, Zhiheng Liu2, Wei Liu1, Qiushan Guo1, Weilin Huang1†, Ping Luo2†

1ByteDance Seed, 2The University of Hong Kong †Corresponding authors, ‡Project lead

### Abstract

Recent advances in generative AI have revolutionized visual content creation, yet aligning model outputs with human preferences remains a critical challenge. While Reinforcement Learning (RL) has emerged as a promising approach for fine-tuning generative models, existing methods like DDPO and DPOK face fundamental limitations - particularly their inability to maintain stable optimization when scaling to large and diverse prompt sets, severely restricting their practical utility. This paper presents DanceGRPO, a framework that addresses these limitations through an innovative adaptation of Group Relative Policy Optimization (GRPO) for visual generation tasks. Our key insight is that GRPO’s inherent stability mechanisms uniquely position it to overcome the optimization challenges that plague prior RL-based approaches on visual generation. DanceGRPO establishes several significant advances: First, it demonstrates consistent and stable policy optimization across multiple modern generative paradigms, including both diffusion models and rectified flows. Second, it maintains robust performance when scaling to complex, real-world scenarios encompassing three key tasks and four foundation models. Third, it shows remarkable versatility in optimizing for diverse human preferences as captured by five distinct reward models assessing image/video aesthetics, text-image alignment, video motion quality, and binary feedback. Our comprehensive experiments reveal that DanceGRPO outperforms baseline methods by up to 181% across multiple established benchmarks, including HPS-v2.1, CLIP Score, VideoAlign, and GenEval. Our results establish DanceGRPO as a robust and versatile solution for scaling Reinforcement Learning from Human Feedback (RLHF) tasks in visual generation, offering new insights into harmonizing reinforcement learning and visual synthesis.

Date: May 1, 2025 Project Page: https://dancegrpo.github.io/ Code: https://github.com/XueZeyue/DanceGRPO

### 1 Introduction

Recent advances in generative models—particularly diffusion models [1–4] and rectified flows [5–7]—have transformed visual content creation by improving output quality and versatility in image and video generation. While pretraining establishes foundational data distributions, integrating human feedback during training proves critical for aligning outputs with human preferences and aesthetic criteria [8]. Existing methods face notable limitations: ReFL [9–11] relies on differentiable reward models, which introduce VRAM inefficiency in video generation and require several extensive engineering efforts, while DPO variants (Diffusion-DPO [12, 13], Flow-DPO [14], OnlineVPO [15]) achieve only marginal visual quality improvements. Reinforcement learning

(RL)-based methods [16, 17], which optimize rewards as black-box objectives, offer potential solutions but introduce three unresolved challenges: (1) the Ordinary Differential Equations (ODEs)-based sampling of rectified flow models conflict with Markov Decision Process formulations; (2) prior policy gradient approaches (DDPO [18], DPOK [19]) show instability when scaling beyond small datasets (e.g., <100 prompts); and (3) existing methods remain unvalidated for video generation tasks.

This work addresses these gaps by reformulating the sampling of diffusion models and rectified flows via Stochastic Differential Equations (SDEs) and applying Group Relative Policy Optimization (GRPO) [20, 21] to stabilize the training process. In this paper, we pioneer the adaptation of GRPO to visual generation tasks through the DanceGRPO framework, establishing a "harmonious dance" between GRPO and visual generation tasks. Our key insight is that GRPO’s architectural stability properties provide a principled solution to the optimization instabilities that have limited previous RL approaches to visual generation.

We extend a systematic study of DanceGRPO, evaluating its performance across generative paradigms (diffusion models, rectified flows) and tasks (text-to-image, text-to-video, image-to-video). Our analysis employs diverse foundation models [2, 22–24] and reward metrics to assess aesthetic quality, alignment, and motion dynamics. Furthermore, through the proposed framework, we discover insights regarding the rollout initialization noise, the reward model compatibility, the Best-of-N inference scaling, the timestep selection, and the learning on binary feedback.

Our contributions can be summarized as follows:

- • Stability and Pioneering. We present the first discovery that GRPO’s inherent stability mechanisms effectively address the core optimization challenges in visual generation that have persistently hindered prior RL-based approaches. We achieve seamless integration between GRPO and visual generation tasks by carefully reformulating the SDEs, selecting appropriate optimized timesteps, initializing noise, and noise scales.
- • Generalization and Scalability. To our knowledge, DanceGRPO is the first RL-based unified application framework capable of seamless adaptation across diverse generative paradigms, tasks, foundational models, and reward models. Unlike prior RL algorithms, primarily validated on text-to-image diffusion models on small-scale datasets, DanceGRPO demonstrates robust performance on large-scale datasets, showcasing both scalability and practical applicability.
- • High Effectiveness. Our experiments demonstrate that DanceGRPO achieves significant performance gains, outperforming baselines by up to 181% across multiple academic benchmarks, including HPSv2.1 [25], CLIP score [26], VideoAlign [14], and GenEval [27], in visual generation tasks. Notably, DanceGRPO also enables models to learn the denoising trajectory in Best-of-N inference scaling. We also make some initial attempts to enable models to capture the distribution of binary (0/1) reward models, showing its ability to capture sparse, thresholding feedback.

### 2 Approach

#### 2.1 Preliminary

Diffusion Model [1]. A diffusion process gradually destroys an observed datapoint x over timestep t, by mixing data with noise, and the forward process of the diffusion model can be defined as :

zt = αtx + σtϵ, where ϵ ∼ N(0,I), (1)

and αt and σt denote the noise schedule. The noise schedule is designed in a way such that z0 is close to clean data and z1 is close to Gaussian noise. To generate a new sample, we initialize the sample z1 and define the sample equation of the diffusion model given the denoising model output ϵˆ at time step t:

zs = αsxˆ + σsϵˆ, (2)

where xˆ can be derived via Eq.(1) and then we can reach a lower noise level s. This is also a DDIM sampler [28].

Rectified Flow [6]. In rectified flow, we view the forward process as a linear interpolation between the data x and a noise term ϵ:

zt = (1 − t)x + tϵ, (3)

where ϵ is always defined as a Gaussian noise. We define the u = ϵ − x as the "velocity" or "vector field". Similar to diffusion model, given the denoising model output uˆ at time step t, we can reach a lower noise level s by:

zs = zt + uˆ · (s − t). (4)

Analysis. Although the diffusion model and rectified flow have different theoretical foundations, in practice, they are two sides of a coin, as shown in the following formula:

z˜s = z˜t + Network output · (ηs − ηt). (5)

For ϵ-prediction (a.k.a. diffusion model), from Eq.(2), we have z˜s = zs/αs, z˜t = zt/αt, ηs = σs/αs, and ηt = σt/αt. For rectified flows, we have z˜s = zs, z˜t = zt, ηs = s, and ηt = t from Eq.(4).

#### 2.2 DanceGRPO

In this section, we first formulate the sampling processes of diffusion models and rectified flows as Markov Decision Processes. Then, we introduce the sampling SDEs and the algorithm of DanceGRPO.

Denoising as a Markov Decision Process. Following DDPO [18], we formulate the denoising process of the diffusion model and rectified flow as a Markov Decision Process (MDP):

st ≜ (c,t,zt), π(at | st) ≜ p(zt−1 | zt,c), P(st+1 | st,at) ≜ δc,δt−1,δz

t−1

, (6)

r(z0,c), if t = 0 0, otherwise

, ρ0(s0) ≜ (p(c),δT,N(0,I))

at ≜ zt−1, R(st,at) ≜

where c is the prompt, and π(at | st) is the probability from zt to zt−1. And δy is the Dirac delta distribution with nonzero density only at y. Trajectories consist of T timesteps, after which P leads to a termination state. r(z0,c) is the reward model, which is always parametrized by a Vision-Language model (such as CLIP [26] and Qwen-VL [29]).

Formulation of Sampling SDEs. Since GRPO requires stochastic exploration through multiple trajectory samples, where policy updates depend on the trajectory probability distribution and their associated reward signals, we unify the sampling processes of the diffusion model and rectified flows into the form of SDEs. For the diffusion model, as demonstrated in [30, 31], the forward SDE is given by: dzt = ftztdt + gtdw. The corresponding reverse SDE can be expressed as:

dzt = ftzt −

- 1 + ε2t

- 2

gt2∇log pt(zt) dt + εtgtdw, (7)

where dw is a Brownian motion, and εt introduces the stochasticity during sampling.

Similarly, the forward ODE of rectified flow is: dzt = utdt. The generative process reverses the ODE in time. However, this deterministic formulation cannot provide the stochastic exploration required for GRPO. Inspired by [32–34], we introduce an SDE case during the reverse process as follows:

dzt = (ut −

- 1

- 2

ε2t∇log pt(zt))dt + εtdw, (8)

where εt also introduces the stochasticity during sampling. Given a normal distribution pt(zt) = N(zt | αtx,σt2I), we have ∇log pt(zt) = −(zt − αtx)/σt2. We can insert this into the above two SDEs and obtain the π(at | st). More theoretical analysis can be found in Appendix B.

Algorithm. Motivated by Deepseek-R1 [20], given a prompt c, generative models will sample a group of outputs {o1,o2,...,oG} from the model πθ

, and optimize the policy model πθ by maximizing the following objective function:

old

J (θ) = E{o

i}Gi=1∼πθold(·|c) at,i∼πθold(·|st,i)

G

T

1 T

1 G

min ρt,iAi,clip ρt,i,1 − ϵ,1 + ϵ Ai , (9)

t=1

i=1

where ρt,i = ππθ(at,i|st,i)

θold(at,i|st,i), and πθ(at,i|st,i) is the policy function is MDP for output oi at time step t, ϵ is a hyper-parameter, and Ai is the advantage function, computed using a group of rewards {r1,r2,...,rG} corresponding to the outputs within each group:

ri − mean({r1,r2,··· ,rG}) std({r1,r2,··· ,rG})

. (10)

Ai =

Due to reward sparsity in practice, we apply the same reward signal across all timesteps during optimization. While traditional GRPO formulations employ KL-regularization to prevent reward over-optimization, we empirically observe minimal performance differences when omitting this component. So, we omit the KLregularization item by default. The full algorithm can be found in Algorithm 1. We also introduce how to train with Classifier-Free Guidance (CFG) [35] in Appendix C. In summary, we formulate the sampling processes of diffusion model and rectified flow as MDPs, use SDE sampling equations, adopt a GRPO-style objective, and generalize to text-to-image, text-to-video, and image-to-video generation tasks.

Initialization Noise. In the DanceGRPO framework, the initialization noise constitutes a critical component. Previous RL-based approaches like DDPO use different noise vectors to initialize training samples. However, as shown in Figure 8 in Appendix F, assigning different noise vectors to samples with the same prompts always leads to reward hacking phenomena in video generation, including training instability. Therefore, in our framework, we assign shared initialization noise to samples originating from the same textual prompts.

Timestep Selection. While the denoising process can be rigorously formalized within the MDP framework, empirical observations reveal that subsets of timesteps within a denoising trajectory can be omitted without compromising performance. This reduction in computational steps enhances efficiency while maintaining output quality, as further analyzed in Section 3.6.

Incorporating Multiple Reward Models. In practice, we employ more than one reward model to ensure more stable training and higher-quality visual results. As illustrated in Figure 9 in Appendix, models trained exclusively with HPS-v2.1 rewards [25] tend to generate unnatural ("oily") outputs, whereas incorporating CLIP scores helps preserve more realistic image characteristics. Rather than directly combining rewards, we aggregate advantage functions, as different reward models often operate on different scales. This approach stabilizes optimization and leads to more balanced generations.

Extension on Best-of-N Inference Scaling. As outlined in Section 3.6, our methodology prioritizes the use of efficient samples—specifically, those associated with the top k and bottom k candidates selected by the Best-of-N sampling. This selective sampling strategy optimizes training efficacy by focusing on high-reward and critical low-reward regions of the solution space. We use brute-force search to generate these samples. While alternative approaches, such as tree search or greedy search, remain promising avenues for further exploration, we defer their systematic integration to future research.

#### 2.3 Application to Different Tasks with Different Rewards

We verify the effectiveness of our algorithm in two generative paradigms (diffusion/rectified flow) and three tasks (text-to-image, text-to-video, image-to-video). For this, we choose four fundamental models (Stable Diffusion [2], HunyuanVideo [22], FLUX [23], SkyReels-I2V [24]) for the experiment. All of these methods can be precisely constructed within the framework of MDP during their sampling process. This allows us to unify the theoretical bases across these tasks and improve them via DanceGRPO. To our knowledge, this is the first work to apply the unified framework to diverse visual generation tasks.

Algorithm 1 DanceGRPO Training Algorithm

Require: Initial policy model πθ; reward models {Rk}Kk=1; prompt dataset D; timestep selection ratio τ; total sampling

steps T

Ensure: Optimized policy model πθ

- 1: for training iteration = 1 to M do
- 2: Sample batch Db ∼ D ▷ Batch of prompts
- 3: Update old policy: πθold ← πθ
- 4: for each prompt c ∈ Db do
- 5: Generate G samples: {oi}Gi=1 ∼ πθold(·|c) with the same random initialization noise
- 6: Compute rewards {rik}Gi=1 using each Rk
- 7: for each sample i ∈ 1..G do
- 8: Calculate multi-reward advantage: Ai ← Kk=1

rik−µk

σk ▷µk, σk per-reward statistics

- 9: end for
- 10: Subsample ⌈τT⌉ timesteps Tsub ⊂ {1..T}
- 11: for t ∈ Tsub do
- 12: Update policy via gradient ascent: θ ← θ + η∇θJ
- 13: end for
- 14: end for
- 15: end for

- Table 1 Comparison of alignment methods across key capabilities. VideoGen: Video generation generalization. Scalability: Scalability to datasets with a large number of prompts. Reward ↑ indicates a significant reward improvement. RFs: Applicable to Rectified Flows. No Diff-Reward: Don’t need differentiable reward models.

Method RL-based VideoGen Scalability Reward ↑ RFs No Diff-Reward DDPO/DPOK ReFL DPO Ours

We use five reward models to optimize visual generation quality: (1) Image Aesthetics quantifies visual appeal using a pretrained model fine-tuned on human-rated data [25]; (2) Text-image Alignment employs CLIP [26] to maximize cross-modal consistency between prompts and outputs; (3) Video Aesthetics Quality extends image evaluation to temporal domains using VLMs [14, 29], assessing frame quality and coherence; (4) Video Motion Quality evaluates motion realism through physics-aware VLM[14] analysis of trajectories and deformations; (5) Thresholding Binary Reward employs a binary mechanism motivated by [20], where rewards are discretized via a fixed threshold (values exceeding the threshold receive 1, others 0), specifically designed to evaluate generative models’ ability to learn abrupt reward distributions under threshold-based optimization.

- 2.4 Comparisons with DDPO, DPOK, ReFL, and DPO As evidenced by our comprehensive capability matrix in Table 1, DanceGRPO establishes new standards for diffusion model alignment. Our method achieves full-spectrum superiority across all evaluation dimensions: (1) seamless video generation, (2) large-scale dataset scalability, (3) significant reward improvements, (4) native compatibility with rectified flows, and (5) independence from differentiable rewards. This integrated capability profile – unobtainable by any single baseline method (DDPO/DPOK/ReFL/DPO) – enables simultaneous optimization across multiple generative domains while maintaining training stability. More comparisons can be found in Appendix D.

### 3 Experiments

#### 3.1 General Setup

Text-to-Image Generation. We employ Stable Diffusion v1.4, FLUX, and HunyuanVideo-T2I (using one latent frame in HunyuanVideo) as foundation models, with HPS-v2.1 [25] and CLIP score [26]—alongside their binary rewards—serving as reward models. A curated prompt dataset, balancing diversity and complexity, guides

optimization. For evaluation, we select 1,000 test prompts to assess CLIP scores and Pick-a-Pic performance in Section 3.2. We use the official prompts for GenEval and HPS-v2.1 benchmark.

Text-to-Video Generation. Our foundation model is HunyuanVideo [22], with reward signals derived from VideoAlign [14]. Prompts are curated using the VidProM [36] dataset, and an additional 1,000 test prompts are filtered to evaluate the VideoAlign scores in Section 3.3.

Image-to-Video Generation. We use SkyReels-I2V [24] as our foundation model. VideoAlign [14] serves as the primary reward metric, while the prompt dataset, constructed via ConsisID [37], is paired with reference images synthesized by FLUX [23] to ensure conditional fidelity. An additional 1000 test prompts are filtered to evaluate the VideoAlign score in Section 3.4.

Experimental Settings. We implemented all models with scaled computational resources appropriate to task complexity: 32 H800 GPUs for flow-based text-to-image models, 8 H800 GPUs for Stable Diffusion variants, 64 H800 GPUs for text-to-video generation systems, and 32 H800 GPUs for image-to-video transformation architectures. We develop our framework based on FastVideo [38, 39]. Comprehensive hyperparameter configurations and training protocols are detailed in Appendix A. We always use more than 10,000 prompts to optimize the models. All reward curves presented in our paper are plotted using a moving average for smoother visualization. We use ODEs-based samplers for evaluation and visualization.

- Table 2 Results on Stable Diffusion v1.4. This table presents the performance of three Stable Diffusion variants: (1) the base model, (2) the model trained with HPS score, and (3) the model optimized with both HPS and CLIP scores. For evaluation, we report HPS-v2.1 and GenEval scores using their official prompts, while CLIP score and Pick-a-Pic metrics are computed on our test set of 1,000 prompts.

Models HPS-v2.1 [25] CLIP Score [26] Pick-a-Pic [40] GenEval [27] Stable Diffusion 0.239 0.363 0.202 0.421

Stable Diffusion with HPS-v2.1 0.365 0.380 0.217 0.521 Stable Diffusion with HPS-v2.1&CLIP Score 0.335 0.395 0.215 0.522

- Table 3 Results on FLUX. In this table, we show the results of FLUX, FLUX trained with HPS score, and FLUX trained with both HPS score and CLIP score. We use the same evaluation prompts as Table 2.

Models HPS-v2.1 [25] CLIP Score [26] Pick-a-Pic [40] GenEval [27] FLUX 0.304 0.405 0.224 0.659

FLUX with HPS-v2.1 0.372 0.376 0.230 0.561 FLUX with HPS-v2.1&CLIP Score 0.343 0.427 0.228 0.687

- Table 4 Comparison of different methods trained on diffusion and solely (not combined) with HPS score and CLIP score. "Baseline" denotes the original results of Stable Diffusion. More comparisons with DDPO can be found in Appendix E.

Table 5 The results of HunyuanVideo on Videoalign and VisionReward trained with VideoAlign VQ&MQ. "Baseline" denotes the original results of HunyuanVideo. We use the weighted sum of the probability for VisionRewardVideo.

Approach Baseline Ours DDPO ReFL DPO HPS-v2.1 0.239 0.365 0.297 0.357 0.241

CLIP Score 0.363 0.421 0.381 0.418 0.367

Benchmarks VQ MQ TA VisionReward

Baseline 4.51 1.37 1.75 0.124 Ours 7.03 (+56%) 3.85 (+181%) 1.59 0.128

#### 3.2 Text-to-Image Generation

Stable Diffusion. Stable Diffusion v1.4, a diffusion-based text-to-image generation framework, comprises three core components: a UNet architecture for iterative denoising, a CLIP-based text encoder for semantic conditioning, and a variational autoencoder (VAE) for latent space modeling. As demonstrated in Table 2 and Figure 1(a), our proposed method, DanceGRPO, achieves a significant improvement in reward metrics, elevating the HPS score from 0.239 to 0.365, as well as the CLIP Score from 0.363 to 0.395. We also take the metric like the Pick-a-Pic [40] and GenEval [27] to evaluate our method. The results confirm the effectiveness

of our method. Moreover, as shown in Table 4, our method exhibits the best performance in terms of metrics compared to other methods. We implement DPO as an online version following [15].

Building on insights from rule-based reward models such as DeepSeek-R1, we conduct preliminary experiments with a binary reward formulation. By thresholding the continuous HPS reward at 0.28—assigning (CLIP score at 0.39) a value of 1 for rewards above this threshold and 0 otherwise, we construct a simplified reward model. Figure 3(a) illustrates that DanceGRPO effectively adapts to this discretized reward distribution, despite the inherent simplicity of the thresholding approach. These results indicate the effectiveness of binary reward models in visual generation tasks. In the future, we will also strive to explore more powerful rule-based visual reward models, for example, making judgments through a multimodal large language model.

Best-of-N Inference Scaling. We explore sample efficiency through Best-of-N inference scaling using Stable Diffusion, as detailed in Section 2.2. By training the model on subsets of 16 samples (with the top 8 and bottom 8 rewards) selected from progressively larger pools (16, 64, and 256 samples per prompt), we evaluate the impact of sample curation on convergence dynamics about Stable Diffusion. Figure 4(a) reveals that Best-of-N scaling substantially accelerates convergence. This underscores the utility of strategic sample selection in reducing training overhead while maintaining performance. For alternative approaches, such as tree search or greedy search, we defer their systematic integration to future research.

FLUX. FLUX.1-dev is a flow-based text-to-image generation model that advances the state-of-the-art across multiple benchmarks, leveraging a more complex architecture than Stable Diffusion. To optimize performance, we integrate two reward models: HPS score and CLIP score. As illustrated in Figure 1(b) and Table 3, the proposed training paradigm achieves significant improvements across all reward metrics.

HunyuanVideo-T2I. HunyuanVideo-T2I is a text-to-image adaptation of the HunyuanVideo framework, reconfigured by reducing the number of latent frames to one. This modification transforms the original video generation architecture into a flow-based image synthesis model. We further optimize the system using the publicly available HPS-v2.1 model, a human-preference-driven metric for visual quality. As demonstrated in Figure 1(c), this approach elevates the mean reward score from about 0.23 to 0.33, reflecting enhanced alignment with human aesthetic preferences.

[Figure 3]

[Figure 4]

[Figure 5]

(b) Reward on aesthetics of FLUX.1-dev (c) Reward on aesthetics of HunyuanVideo-T2I

(a) Reward on aesthetics of Stable Diffusion

- Figure 1 We visualize the reward curves of Stable Diffusion, FLUX.1-dev, and HunyuanVideo-T2I on HPS score from left to right. After applying CLIP score, the HPS score decreases, but the generated images become more natural (Figure 9 in Appendix), and the CLIP score improves (Tables 2 and 3).

[Figure 6]

[Figure 7]

[Figure 8]

#### 3.3 Text-to-Video Generation

HunyuanVideo. Optimizing text-to-video generation models presents significantly greater challenges compared to text-to-image frameworks, primarily due to elevated computational costs during training and inference, as well as slower convergence rates. In the pretraining protocol, we always adopt a progressive strategy: initial training focuses on text-to-image generation, followed by low-resolution video synthesis, and culminates in high-resolution video refinement. However, empirical observations reveal that relying solely on image-centric optimization leads to suboptimal video generation outcomes. To address this, our implementation employs training video samples synthesized at a resolution of 480×480 pixels, but we can visualize the samples with larger pixels.

(a) Reward on Videoalign motion quality (text-to-video)

(b) Reward on Videoalign visual aesthetics quality (text-to-video)

(c) Reward on Videoalign motion quality (image-to-video)

[Figure 9]

[Figure 10]

[Figure 11]

7

(a) Reward on inference scaling Best-of-N (b) Ablation study on timestep fraction (c) Ablation study on different noise level

(a) Reward on binary reward models (b) Human evaluation

[Figure 14]

[Figure 15]

[Figure 16]

(b) Reward on aesthetics of FLUX.1-dev (c) Reward on aesthetics of HunyuanVideo-T2I

(a) Reward on aesthetics of Stable Diffusion

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

(a) Reward on Videoalign motion quality (text-to-video)

(a) Reward on aesthetics of Stable Diffusion (b) Reward on aesthetics of FLUX.1-dev

(b) Reward on Videoalign visual aesthetics quality (text-to-video)

(c) Reward on Videoalign motion quality (image-to-video)

(c) Reward on aesthetics of HunyuanVideo-T2I

- Figure 2 We visualize the training curves of motion quality and visual aesthetics quality on HunyuanVideo, motion quality on SkyReels-I2V.

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

Furthermore, constructing an effective video reward model for training alignment poses substantial difficulties. Our experiments evaluated several candidates: the Videoscore [41] model exhibited unstable reward distributions, rendering it impractical for optimization, while Visionreward-Video [42], a 29-dimensional metric, yielded semantically coherent rewards but suffered from inaccuracies across individual dimensions. Consequently, we adopted VideoAlign [14], a multidimensional framework evaluating three critical aspects: visual aesthetics quality, motion quality, and text-video alignment. Notably, the text-video alignment dimension demonstrated significant instability, prompting its exclusion from our final analysis. We also increase the number of sampled frames per second for VideoAlign to improve the training stability. As illustrated in Figure 2(a) and Figure 2(b), our methodology achieves relative improvements of 56% and 181% in visual and motion quality metrics, respectively. Extended qualitative results are provided in the Table 5.

(a) Reward on inference scaling Best-of-N (b) Ablation study on timestep fraction (c) Ablation study on different noise level

(a) Reward on Videoalign motion quality (text-to-video)

(b) Reward on Videoalign visual quality (text-to-video)

(c) Reward on Videoalign motion quality (image-to-video)

[Figure 29]

[Figure 30]

#### 3.4 Image-to-Video Generation

[Figure 31]

[Figure 32]

[Figure 33]

SkyReels-I2V. SkyReels-I2V represents a state-of-the-art open-source image-to-video (I2V) generation framework, established as of February 2025 at the inception of this study. Derived from the HunyuanVideo architecture, the model is fine-tuned by integrating image conditions into the input concatenation process. A central finding of our investigation is that I2V models exclusively allow optimization of motion quality, encompassing motion coherence and aesthetic dynamics, since visual fidelity and text-video alignment are inherently constrained by the attributes of the input image rather than the parametric space of the model. Consequently, our optimization protocol leverages the motion quality metric from the VideoAlign reward model, achieving a 118% relative improvement in this dimension as shown in Figure 2(c). We must enable the CFG-training to ensure the sampling quality during the RLHF training process.

(a) Reward on binary reward models (b) Human evaluation

(a) Reward on inference scaling Best-of-N (b) Ablation study on timestep fraction (c) Ablation study on different noise level

[Figure 34]

[Figure 35]

(a) Reward on binary reward models (b) Human evaluation

###### Figure 3 (a) We visualize the training curves of binary rewards. (b) We show the human evaluation results using FLUX (T2I), HunyuanVideo (T2V), and SkyReel (I2V), respectively.

#### 3.5 Human Evaluation

We present the results of our human evaluation, conducted using in-house prompts and reference images. For text-to-image generation, we evaluate FLUX on 240 prompts. For text-to-video generation, we assess HunyuanVideo on 200 prompts, and for image-to-video generation, we test SkyReels-I2V on 200 prompts paired with their corresponding reference images. As shown in Figure 3(b), human artists consistently prefer outputs refined with RLHF. More visualization results can be found in Figure 10 in Appendix, and Appendix F.

[Figure 36]

[Figure 37]

[Figure 38]

#### 3.6 Ablation Study

Ablation on Timestep Selection. As detailed in Section 2.2, we investigate the impact of timestep selection on the training dynamics of the HunyuanVideo-T2I model. We conduct an ablation study across three experimental conditions: (1) training exclusively on the first 30% of timesteps from noise, (2) training on randomly sampled 30% of timesteps, (3) training on the final 40% of timesteps before outputs, (4) training on randomly sampled 60% of timesteps, and (5) training on sampled 100% of timesteps. As shown in Figure 4(b), empirical results indicate that the initial 30% of timesteps are critical for learning foundational generative patterns, as evidenced by their disproportionate contribution to model performance. However, restricting training solely to this interval leads to performance degradation compared to full-sequence training, likely due to insufficient exposure to late-stage refinement dynamics.

(b) Reward on aesthetics of FLUX.1-dev (c) Reward on aesthetics of HunyuanVideo-T2I

(a) Reward on aesthetics of Stable Diffusion

[Figure 39]

[Figure 40]

[Figure 41]

To reconcile this trade-off between computational efficiency and model fidelity, we always implement a 40% stochastic timestep dropout strategy during training. This approach randomly masks 40% of timesteps across all phases while preserving temporal continuity in the latent diffusion process. The findings suggest that strategic timestep subsampling can optimize resource utilization in flow-based generative frameworks.

Ablation on Noise Level εt. We systematically investigate the impact of noise level εt during training on FLUX. Our analysis reveals that reducing εt leads to a significant performance degradation, as quantitatively demonstrated in Figure 4(c). Notably, experiments with alternative noise decay schedules (e.g., those used in DDPM) show no statistically significant differences in output quality compared to our baseline configuration. Futhermore, the noise level larger than 0.3 sometimes leads to noisy images after RLHF training.

(a) Reward on Videoalign motion quality (text-to-video)

(b) Reward on Videoalign visual aesthetics quality (text-to-video)

(c) Reward on Videoalign motion quality (image-to-video)

[Figure 42]

[Figure 43]

[Figure 44]

(a) Reward on inference scaling Best-of-N (b) Ablation study on timestep fraction (c) Ablation study on different noise level

- Figure 4 This figure shows the rewards on Best-of-N inference scaling, the ablation on timestep selection, and the ablation on noise level, respectively. While Best-of-N inference scaling consistently improves performance with more samples, it reduces sampling efficiency. Therefore, we leave Best-of-N as an optional extension.

[Figure 45]

[Figure 46]

### 4 Related Work

Aligning Large Language Models. Large Language Models (LLMs) [43–47] are typically aligned with Reinforcement Learning from Human Feedback (RLHF) [46–51]. RLHF involves training a reward function based on comparison data of model outputs to capture human preferences, which is then utilized in reinforcement learning to align the policy model. While some approaches leverage policy gradient methods, others focus on Direct Policy Optimization (DPO) [52]. Policy gradient methods have proven effective but are computationally expensive and require extensive hyperparameter tuning. In contrast, DPO offers a more cost-efficient alternative but consistently underperforms compared to policy gradient methods.

(a) Reward on binary reward models (b) Human evaluation

Recently, DeepSeek-R1 [20] demonstrated that the application of large-scale reinforcement learning with formatting and result-only reward functions can guide LLMs toward the self-emergence of thought processes, enabling human-like complex chain-of-thought reasoning. This approach has achieved significant advantages in complex reasoning tasks, showcasing immense potential in advancing reasoning capabilities within Large Language Models.

Aligning Diffusion Models and Rectified Flows. Diffusion models and rectified flows can also benefit significantly from alignment with human feedback, but the exploration remains primitive compared with LLMs. Key approaches in this area include: (1) Direct Policy Optimization (DPO)-style [12, 14, 42, 53, 54] methods, (2) direct backpropagation with reward signals [55], such as ReFL [9], and (3) policy gradient-based methods, including DPOK [19] and DDPO [18]. However, production-level models predominantly rely on DPO and ReFL, as previous policy gradient methods have demonstrated instability when applied to large-scale settings. Our work addresses this limitation, providing a robust solution to enhance stability and scalability. We also hope our work offers insights into its potential to unify optimization paradigms across different modalities (e.g. image and text) [56, 57].

### 5 Conclusion and Future Work

This work pioneers the integration of Group Relative Policy Optimization (GRPO) into visual generation, establishing DanceGRPO as a unified framework for enhancing diffusion models and rectified flows across text-to-image, text-to-video, and image-to-video tasks. By bridging the gap between language and visual modalities, our approach addresses critical limitations of prior methods, achieving superior performance through efficient alignment with human preferences and robust scaling to complex, multi-task settings. Experiments demonstrate substantial improvements in visual fidelity, motion quality, and text-image alignment. Future work will explore GRPO’s extension to multimodal generation, further unifying optimization paradigms across Generative AI.

### References

- [1] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

- [2] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.

- [3] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023.

- [4] Zeyue Xue, Guanglu Song, Qiushan Guo, Boxiao Liu, Zhuofan Zong, Yu Liu, and Ping Luo. Raphael: Textto-image generation via large mixture of diffusion paths. Advances in Neural Information Processing Systems, 36:41693–41706, 2023.

- [5] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.

- [6] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022.

- [7] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning, 2024.

- [8] Lixue Gong, Xiaoxia Hou, Fanshi Li, Liang Li, Xiaochen Lian, Fei Liu, Liyang Liu, Wei Liu, Wei Lu, Yichun Shi, et al. Seedream 2.0: A native chinese-english bilingual image generation foundation model. arXiv preprint arXiv:2503.07703, 2025.

- [9] Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong. Imagereward: Learning and evaluating human preferences for text-to-image generation. Advances in Neural Information Processing Systems, 36:15903–15935, 2023.

- [10] Jiacheng Zhang, Jie Wu, Yuxi Ren, Xin Xia, Huafeng Kuang, Pan Xie, Jiashi Li, Xuefeng Xiao, Weilin Huang, Shilei Wen, et al. Unifl: Improve latent diffusion model via unified feedback learning. arXiv preprint arXiv:2404.05595, 2024.

- [11] Ming Li, Taojiannan Yang, Huafeng Kuang, Jie Wu, Zhaoning Wang, Xuefeng Xiao, and Chen Chen. Controlnet++: Improving conditional controls with efficient consistency feedback: Project page: liming-ai. github. io/controlnet_plus_plus. In European Conference on Computer Vision, pages 129–147. Springer, 2024.

- [12] Bram Wallace, Meihua Dang, Rafael Rafailov, Linqi Zhou, Aaron Lou, Senthil Purushwalkam, Stefano Ermon, Caiming Xiong, Shafiq Joty, and Nikhil Naik. Diffusion model alignment using direct preference optimization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8228–8238, 2024.

- [13] Ziyu Guo, Renrui Zhang, Chengzhuo Tong, Zhizheng Zhao, Peng Gao, Hongsheng Li, and Pheng-Ann Heng. Can we generate images with cot? let’s verify and reinforce image generation step by step. arXiv preprint arXiv:2501.13926, 2025.

- [14] Jie Liu, Gongye Liu, Jiajun Liang, Ziyang Yuan, Xiaokun Liu, Mingwu Zheng, Xiele Wu, Qiulin Wang, Wenyu Qin, Menghan Xia, et al. Improving video generation with human feedback. arXiv preprint arXiv:2501.13918, 2025.

- [15] Jiacheng Zhang, Jie Wu, Weifeng Chen, Yatai Ji, Xuefeng Xiao, Weilin Huang, and Kai Han. Onlinevpo: Align video diffusion model with online video-centric preference optimization. arXiv preprint arXiv:2412.15159, 2024.

- [16] Richard S Sutton, Andrew G Barto, et al. Reinforcement learning: An introduction, volume 1. MIT press Cambridge, 1998.

- [17] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

- [18] Kevin Black, Michael Janner, Yilun Du, Ilya Kostrikov, and Sergey Levine. Training diffusion models with reinforcement learning. arXiv preprint arXiv:2305.13301, 2023.

- [19] Ying Fan, Olivia Watkins, Yuqing Du, Hao Liu, Moonkyung Ryu, Craig Boutilier, Pieter Abbeel, Mohammad Ghavamzadeh, Kangwook Lee, and Kimin Lee. Dpok: Reinforcement learning for fine-tuning text-to-image diffusion models. Advances in Neural Information Processing Systems, 36:79858–79885, 2023.

- [20] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

- [21] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Y Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

- [22] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024.

- [23] Black Forest Labs. Flux. https://github.com/black-forest-labs/flux, 2024.
- [24] SkyReels-AI. Skyreels v1: Human-centric video foundation model. https://github.com/SkyworkAI/SkyReels-V1, 2025.
- [25] Xiaoshi Wu, Keqiang Sun, Feng Zhu, Rui Zhao, and Hongsheng Li. Better aligning text-to-image models with human preference. arXiv preprint arXiv:2303.14420, 1(3), 2023.

- [26] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021.

- [27] Dhruba Ghosh, Hannaneh Hajishirzi, and Ludwig Schmidt. Geneval: An object-focused framework for evaluating text-to-image alignment. Advances in Neural Information Processing Systems, 36:52132–52152, 2023.

- [28] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020.

- [29] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024.

- [30] Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. Advances in neural information processing systems, 32, 2019.

- [31] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020.

- [32] Michael S Albergo and Eric Vanden-Eijnden. Building normalizing flows with stochastic interpolants. arXiv preprint arXiv:2209.15571, 2022.

- [33] Michael S Albergo, Nicholas M Boffi, and Eric Vanden-Eijnden. Stochastic interpolants: A unifying framework for flows and diffusions, 2023. URL https://arxiv. org/abs/2303.08797, 3.

- [34] Ruiqi Gao, Emiel Hoogeboom, Jonathan Heek, Valentin De Bortoli, Kevin P. Murphy, and Tim Salimans. Diffusion meets flow matching: Two sides of the same coin. 2024.
- [35] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022.

- [36] Wenhao Wang and Yi Yang. Vidprom: A million-scale real prompt-gallery dataset for text-to-video diffusion models. arXiv preprint arXiv:2403.06098, 2024.

- [37] Shenghai Yuan, Jinfa Huang, Xianyi He, Yunyang Ge, Yujun Shi, Liuhan Chen, Jiebo Luo, and Li Yuan. Identitypreserving text-to-video generation by frequency decomposition. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 12978–12988, 2025.

- [38] Hangliang Ding, Dacheng Li, Runlong Su, Peiyuan Zhang, Zhijie Deng, Ion Stoica, and Hao Zhang. Efficient-vdit: Efficient video diffusion transformers with attention tile, 2025.
- [39] Peiyuan Zhang, Yongqi Chen, Runlong Su, Hangliang Ding, Ion Stoica, Zhenghong Liu, and Hao Zhang. Fast video generation with sliding tile attention, 2025.

- [40] Yuval Kirstain, Adam Polyak, Uriel Singer, Shahbuland Matiana, Joe Penna, and Omer Levy. Pick-a-pic: An open dataset of user preferences for text-to-image generation. Advances in Neural Information Processing Systems, 36:36652–36663, 2023.

- [41] Xuan He, Dongfu Jiang, Ge Zhang, Max Ku, Achint Soni, Sherman Siu, Haonan Chen, Abhranil Chandra, Ziyan Jiang, Aaran Arulraj, et al. Videoscore: Building automatic metrics to simulate fine-grained human feedback for video generation. arXiv preprint arXiv:2406.15252, 2024.

- [42] Jiazheng Xu, Yu Huang, Jiale Cheng, Yuanming Yang, Jiajun Xu, Yuan Wang, Wenbo Duan, Shen Yang, Qunlin Jin, Shurun Li, et al. Visionreward: Fine-grained multi-dimensional human preference learning for image and video generation. arXiv preprint arXiv:2412.21059, 2024.

- [43] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

- [44] Yupeng Chang, Xu Wang, Jindong Wang, Yuan Wu, Linyi Yang, Kaijie Zhu, Hao Chen, Xiaoyuan Yi, Cunxiang Wang, Yidong Wang, et al. A survey on evaluation of large language models. ACM transactions on intelligent systems and technology, 15(3):1–45, 2024.

- [45] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.

- [46] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

- [47] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, et al. Qwen2. 5 technical report. arXiv preprint arXiv:2412.15115, 2024.

- [48] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744, 2022.

- [49] Harrison Lee, Samrat Phatale, Hassan Mansoor, Kellie Ren Lu, Thomas Mesnard, Johan Ferret, Colton Bishop, Ethan Hall, Victor Carbune, and Abhinav Rastogi. Rlaif: Scaling reinforcement learning from human feedback with ai feedback. 2023.
- [50] Leo Gao, John Schulman, and Jacob Hilton. Scaling laws for reward model overoptimization. In International Conference on Machine Learning, pages 10835–10866. PMLR, 2023.

- [51] Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, et al. Training a helpful and harmless assistant with reinforcement learning from human feedback. arXiv preprint arXiv:2204.05862, 2022.

- [52] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems, 36:53728–53741, 2023.

- [53] Zhanhao Liang, Yuhui Yuan, Shuyang Gu, Bohan Chen, Tiankai Hang, Ji Li, and Liang Zheng. Step-aware preference optimization: Aligning preference with denoising performance at each step. arXiv preprint arXiv:2406.04314, 2(5):7, 2024.

- [54] Tao Zhang, Cheng Da, Kun Ding, Kun Jin, Yan Li, Tingting Gao, Di Zhang, Shiming Xiang, and Chunhong Pan. Diffusion model as a noise-aware latent reward model for step-level preference optimization. arXiv preprint arXiv:2502.01051, 2025.

- [55] Mihir Prabhudesai, Russell Mendonca, Zheyang Qin, Katerina Fragkiadaki, and Deepak Pathak. Video diffusion alignment via reward gradients. arXiv preprint arXiv:2407.08737, 2024.

- [56] Yiyang Ma, Xingchao Liu, Xiaokang Chen, Wen Liu, Chengyue Wu, Zhiyu Wu, Zizheng Pan, Zhenda Xie, Haowei Zhang, Liang Zhao, et al. Janusflow: Harmonizing autoregression and rectified flow for unified multimodal understanding and generation. arXiv preprint arXiv:2411.07975, 2024.

###### [57] Quan Sun, Yufeng Cui, Xiaosong Zhang, Fan Zhang, Qiying Yu, Yueze Wang, Yongming Rao, Jingjing Liu, Tiejun Huang, and Xinlong Wang. Generative multimodal models are in-context learners. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14398–14409, 2024.

## Appendix

### A Experimental Settings

We provide detailed experimental settings in Table 6, which apply exclusively to training without classifier-free guidance (CFG). When enabling CFG, we configure one gradient update per iteration. Additionally, the sampling steps vary by model: we use 50 steps for Stable Diffusion, and 25 steps for FLUX and HunyuanVideo.

Table 6 Our Hyper-paramters.

Learning rate 1e-5

Optimizer AdamW Gradient clip norm 1.0

Prompts per iteration 32

Images per prompt 12 Gradient updates per iteration 4

Clip range ϵ 1e-4 Noise level εt 0.3

Timestep Selection τ 0.6

### B More Analysis

#### B.1 Stochastic Interpolants

The stochastic interpolant framework, introduced by [33], offers a unifying perspective on generative models like rectified flows and score-based diffusion models. It achieves this by constructing a continuous-time stochastic process that bridges any two arbitrary probability densities, ρ0 and ρ1.

In our work, we connect with a specific type of stochastic interpolant known as spatially linear interpolants, defined in Section 4 of [33]. Given densities ρ0,ρ1 : Rd → R≥0, a spatially linear stochastic interpolant process xt is defined as:

zt = α(t)z0 + β(t)z1 + γ(t)ϵ, t ∈ [0,1], (11)

where z0 ∼ ρ0, z1 ∼ ρ1, and ϵ ∼ N(0,I) is a standard Gaussian random variable independent of z0 and z1. The functions α,β,γ : [0,1] → R are sufficiently smooth and satisfy the boundary conditions:

α(0) = β(1) = 1, α(1) = β(0) = γ(0) = γ(1) = 0, (12)

with the additional constraint that γ(t) ≥ 0 for t ∈ (0,1). The term γ(t)z introduces latent noise, smoothing the path between densities.

Specific choices within this framework recover familiar models:

- • Rectified Flow (RF): Setting γ(t) = 0 (removing the latent noise), α(t) = 1 − t, and β(t) = t yields the linear interpolation zt = (1 − t)z0 + tz1 used in Rectified Flow [6, 33]. The dynamics are typically governed by an ODE dzt = utdt, where ut is the learned velocity field.
- • Score-Based Diffusion Models (SBDM): The framework connects to SBDMs via one-sided linear interpolants (Section 4.4 of [33]), where ρ1 is typically Gaussian. The interpolant takes the form zt = α(t)z0+β(t)z1. The VP-SDE formulation [31] corresponds to choosing α(t) = √1 − t2 and β(t) = t after a time reparameterization.

A pivotal insight is that: "The law of the interpolant zt at any time t ∈ [0,1] can be realized by many different processes, including an ODE and forward and backward SDEs whose drifts can be learned from data."

The stochastic interpolant framework provides a probability flow ODE for RF:

dzt = utdt. (13) The backward SDE associated with the interpolant’s density evolution is given by:

dzt = bB(t,xt)dt + 2ϵ(t)dz, (14)

where bB(t,x) = ut − ϵ(t)s(t,x) is the backward drift, s(t,x) is the score function, and ϵ(t) ≥ 0 is a tunable diffusion coefficient (noise schedule). If we set εt = 2ϵ(t), the backward SDE becomes:

ε2t

2 ∇log pt(zt) dt + εtdz, (15) which is the same as [34].

dzt = ut −

#### B.2 Connections between Rectified Flows and Diffusion Models

We aim to demonstrate the equivalence between certain formulations of diffusion models and flow matching (specifically, stochastic interpolants) by deriving the hyperparameters of one model from the other.

The forward process of a diffusion model is described by an SDE: dzt = ftztdt + gtdw, (16)

where dw is a Brownian motion, and ft,gt define the noise schedule. The corresponding generative (reverse) process SDE is given by:

dzt = ftzt −

- 1 + ηt2

- 2

gt2∇log pt(zt) dt + ηtgtdw, (17)

where pt(zt) is the marginal probability density of zt at time t. For flow matching, we consider an interpolant path between data x = z0 and noise ϵ (typically ϵ ∼ N(0,I)):

zt = αtx + σtϵ. (18) This path satisfies the ODE:

dzt = utdt, where ut = α˙tx + σ˙tϵ. (19) This can be generalized to a stochastic interpolant SDE:

dzt = (ut −

1 2

ε2t∇log pt(zt))dt + εtdw. (20)

The core idea is to match the marginal distributions pt(zt) generated by the forward diffusion process Eq.(16) with those implied by the interpolant path Eq.(18). We will derive ft and gt from this requirement, and then relate the noise terms of the generative SDEs Eq.(17) and Eq.(20) to find ηt.

Deriving ft by Matching Means. From Eq.(18), assuming x = z0 is fixed and E[ϵ] = 0, the mean of zt is E[zt] = αtx. The mean mt = E[zt] of the process Eq.(16) starting from z0 = x satisfies the ODE dm

dt = ftmt. We require mt = αtx for all t. Substituting into the mean ODE:

t

d dt

(αtx) = ft(αtx), α˙tx = ftαtx. (21)

Assuming this holds for any x and αt ̸= 0, we divide by αtx:

α˙t αt

ft =

(22)

Using the identity ddt log(y) = y/y˙ , we get:

ft = ∂t log(αt) (23)

Deriving gt2 by Matching Variances. From Eq.(18), assuming x is fixed and V ar(ϵ) = I (identity matrix for standard Gaussian noise), the variance (covariance matrix) of zt is V ar(zt) = V ar(αtx+σtϵ) = σt2V ar(ϵ) = σt2I. Let Vt = σt2 be the scalar variance magnitude. The variance Vt = Tr(V ar(zt))/d for the process Eq.(16) evolves according to the Lyapunov equation: dV

dt = 2ftVt + gt2 (Here, gt2 represents the magnitude of the noise injection rate). We require Vt = σt2. Substitute Vt = σt2 and ft = α˙t/αt into the variance evolution equation:

t

d dt

(σt2) = 2

α ˙t αt

α˙t αt

σt2 + gt2, 2σtσ˙t = 2

σt2 + gt2. (24)

Solving for gt2:

α˙t αt

2 αt

2σt αt

gt2 = 2σtσ˙t − 2

σt2 =

(αtσtσ˙t − α˙tσt2) =

(αtσ˙t − α˙tσt) (25) Using the quotient rule for differentiation, ∂t(σt/αt) = α

tσ˙t−α˙ tσt

α2t , which implies αtσ˙t − α˙tσt = αt2∂t(σt/αt). Then we get:

2σt αt

σt αt

σt αt

gt2 =

αt2∂t

(26) Thus, we have:

= 2αtσt∂t

σt αt

gt2 = 2αtσt∂t

(27)

Deriving ηt by Matching Noise Terms in Generative SDEs. We compare the coefficients of the Brownian motion term (dw) in the reverse diffusion SDE Eq.(17) and the stochastic interpolant SDE Eq.(20). The diffusion coefficient (magnitude of the noise term) is Ddiff = ηtgt. The diffusion coefficient is Dint = εt. To match the noise structure in these specific SDE forms, we set Ddiff = Dint: ηtgt = εt. Solving for ηt (assuming gt ̸= 0):ηt = ε

gt. Substitute gt = gt2 using the result from Eq.(27):

t

εt 2αtσt∂t(σt/αt)

(28)

ηt =

Summary of Results. By requiring the forward diffusion process Eq.(16) to match the marginal mean and variance of the interpolant path Eq.(18) at all times t, we derived:

ft = ∂t log(αt), gt2 = 2αtσt∂t(σt/αt), ηt = εt/(2αtσt∂t(σt/αt))1/2. (29)

These relationships establish the equivalence between the parameters of the two frameworks under the specified conditions.

### C Classifier-Free Guidance (CFG) Training

Classifier-Free Guidance (CFG) [35] is a widely adopted technique for generating high-quality samples in conditional generative modeling. However, in our settings, integrating CFG into training pipelines introduces instability during optimization. To mitigate this, we empirically recommend disabling CFG during the sampling phase for models with high sample fidelity, such as HunyuanVideo and FLUX, as it reduces gradient oscillation while preserving output quality.

For CFG-dependent models like SkyReels-I2V and Stable Diffusion, where CFG is critical for reasonable sample quality, we identify a key instability: training exclusively on the conditional objective leads to divergent optimization trajectories. This necessitates the joint optimization of both conditional and unconditional outputs, effectively doubling VRAM consumption due to dual-network computations. Morever, we propose reducing the frequency of parameter updates per training iteration. For instance, empirical validation shows that limiting updates to one per iteration significantly enhances training stability for SkyReels-I2V, with minimal impact on convergence rates.

### D Advantages over DDPO and DPOK

Our approach differs from prior RL-based methods for text-to-image diffusion models (e.g., DDPO, DPOK) in three key aspects: (1) We employ a GRPO-style objective function, (2) we compute advantages within prompt-level groups rather than globally, (3) we ensure noise consistency across samples from the same prompt, (4) we generalize these improvements beyond diffusion models by applying them to rectified flows and scaling to video generation tasks.

### E Inserting DDPO into Rectified Flow SDEs

We also insert DDPO-style objective function into rectified flow SDEs, but it always diverges, as shown in Figure 5, which demonstrates the superiority of DanceGRPO.

[Figure 47]

[Figure 48]

- Figure 5 We visualize the results of DDPO and Ours. DDPO always diverges when applied to rectified flow SDEs.

### F More Visualization Results

We provide more visualization results on FLUX, Stable Diffusion, and HunyuanVideo as shown in Figure 6, 7, 10, 11, 12, 13, 14, 15, 16, 17, and 18.

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

Prompt: A man lying down on green grass, gazing at the stars during an evening at a countryside villa

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

Prompt: A sinister man with red eyes speaking, very close shot, Cthulhu

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

Prompt: A chubby baby playing with toys in the snow

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

Prompt: Generate a picture of a blue sports car parked on the road, metal texture

###### Figure 6 We visualize the results by selecting FLUX optimized with the HPS score at iterations 0, 60, 120, 180, 240, and 300. The optimized outputs tend to exhibit brighter tones and richer details. However, incorporating CLIP score regularization is crucial, as demonstrated in Figures 11, 12, and 13.

Prompt: A man lying down on green grass, gazing at the stars during an evening at a countryside villa

Prompt: A sinister man with red eyes speaking, very close shot, Cthulhu

Prompt: A chubby baby playing with toys in the snow

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

Prompt: Generate a picture of a blue sports car parked on the road, metal texture

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

Prompt: a basketball player wearing a jersey with the number _23_

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

Prompt: A boy with yellow hair is cycling on a country road, realistic, full shot, warm color palette

###### Figure 7 Visualization of the diversity of the model before and after RLHF. Different seed tends to generate similar images after RLHF.

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

Reward Hacking Without Reward Hacking

###### Figure 8 Visualization of the results of reward hacking (with different initialization noise) and without reward hacking (with the same initialization noise) on HunyuanVideo. Prompt: A splash of water in a clear glass, with sparkling, clear radiant reflections, sunlight, sparkle

[Figure 119]

[Figure 120]

[Figure 121]

Original Optimized with HPS score Optimized with HPS score and CLIP score

###### Figure 9 This figure demonstrates the impact of the CLIP score. The prompt is "A photo of cup". We find that the model trained solely with HPS-v2.1 rewards tends to produce unnatural ("oily") outputs, while incorporating CLIP scores helps maintain more natural image characteristics.

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

Before RLHF

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

After RLHF

A green cup and two blue cell phones, E-commerce poster, minimalist design

A long-haired man riding a giant sea monster in the deep ocean

We are painting the wall with cracks

A blue coloured pizza

(a) Visualization of FLUX.1-dev

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

Before RLHF

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

After RLHF

Video of a tiger walking in a forest at night

A man talking to a giant robot, fingers pointing at each other, wide shot

(b) Visualization of HunyuanVideo

###### Figure 10 Overall visualization. We visualize the results before and after RLHF of FLUX and HunyuanVideo.

[Figure 142]

[Figure 143]

[Figure 144]

Original Optimized with HPS score Optimized with HPS score

and CLIP score Prompt: Two boys and two girls, Y2K outfit

[Figure 145]

[Figure 146]

[Figure 147]

Original Optimized with HPS score Optimized with HPS score

and CLIP score Prompt: A tranquil park furnished with rows of benches made of marble

[Figure 148]

[Figure 149]

[Figure 150]

Original Optimized with HPS score Optimized with HPS score

and CLIP score Prompt: A dog with soft fur and bright eyes jumping after a toy, in a cozy living room

[Figure 151]

[Figure 152]

[Figure 153]

Original Optimized with HPS score Optimized with HPS score and CLIP score

Prompt: The lake surface mirrors the sky, surrounded by tall, lush trees, in the style of James Bidgood

[Figure 154]

[Figure 155]

[Figure 156]

Original Optimized with HPS score Optimized with HPS score

and CLIP score Prompt: A photo of a soft, plush teddy bear sits on a child's bed, diagonal composition

[Figure 157]

[Figure 158]

[Figure 159]

Original Optimized with HPS score Optimized with HPS score

and CLIP score Prompt: A man with a bunch of flowers is knocking on the door, Wong Kar-wai style, movie screenshot

[Figure 160]

[Figure 161]

[Figure 162]

Original Optimized with HPS score Optimized with HPS score

and CLIP score Prompt: The large ball crashed right through the table because it was made of styrofoam

[Figure 163]

[Figure 164]

[Figure 165]

Original Optimized with HPS score Optimized with HPS score and CLIP score

Prompt: A group of mongooses scuttle about, set against a desert backdrop, bathed in bright and warm earth tones

[Figure 166]

[Figure 167]

[Figure 168]

Original Optimized with HPS score Optimized with HPS score

and CLIP score Prompt: A television made of water that displays an image of a cityscape at night

[Figure 169]

[Figure 170]

[Figure 171]

Original Optimized with HPS score Optimized with HPS score

and CLIP score Prompt: A boy is smiling under the cherry blossom tree, Rick and Morty

[Figure 172]

[Figure 173]

[Figure 174]

Original Optimized with HPS score Optimized with HPS score

and CLIP score Prompt: A cup with a diameter of 5 cm and a cup with a diameter of 3cm

[Figure 175]

[Figure 176]

[Figure 177]

Original Optimized with HPS score Optimized with HPS score

and CLIP score Prompt: A family, medium shot

[Figure 178]

[Figure 179]

[Figure 180]

Original Optimized with HPS score Optimized with HPS score and CLIP score

Prompt: The feeling of a mirage in a house, the sunset shining on the lake surface, and the water flowing slowly

[Figure 181]

[Figure 182]

[Figure 183]

Original Optimized with HPS score Optimized with HPS score

and CLIP score Prompt: Meat all over the place and blood and a green plant in the middle, cinematic lighting, 4k

[Figure 184]

[Figure 185]

[Figure 186]

Original Optimized with HPS score Optimized with HPS score

and CLIP score Prompt: A bear is walking away in the forest, sunset

[Figure 187]

[Figure 188]

[Figure 189]

Original Optimized with HPS score Optimized with HPS score

and CLIP score Prompt: Daft punk driving through the city at night

###### Figure 14 We present the original outputs of HunyuanVideo-T2I, alongside optimizations driven solely by the HPS score and those enhanced by both the HPS and CLIP scores.

[Figure 190]

[Figure 191]

[Figure 192]

Original Optimized with HPS score Optimized with HPS score

and CLIP score Prompt: Tibetan terrier, chrysanthemum

[Figure 193]

[Figure 194]

[Figure 195]

Original Optimized with HPS score Optimized with HPS score

and CLIP score Prompt: Dhole, Cuon alpinus

[Figure 196]

[Figure 197]

[Figure 198]

Original Optimized with HPS score Optimized with HPS score

and CLIP score Prompt: Mongoose

[Figure 199]

[Figure 200]

[Figure 201]

Original Optimized with HPS score Optimized with HPS score

and CLIP score Prompt: Angora, Angora rabbit

###### Figure 15 We present the original outputs of Stable Diffusion, alongside optimizations driven solely by the HPS score and those enhanced by both the HPS and CLIP scores.

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

Before RLHF

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

After RLHF

Prompt: Dogs running with shoes 4k realism cinematic

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

Before RLHF

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

After RLHF

Prompt: Tobuscus wearing a green shirt, gliding through the sky on magic shoes

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

Before RLHF

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

After RLHF

Prompt: Realistic scene 4k HD angels in unique armor with gems and gold white on it with huge wings fighting among themselves

###### Figure 16 Visualization results of HunyuanVideo.

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

Before RLHF

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

After RLHF

Prompt: Man asking the girl

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

Before RLHF

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

After RLHF

Prompt: In a cold night, a girl is trying to sell the matches, hungry and tired, but still believe that it will become better

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

Before RLHF

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

After RLHF

Prompt: A horror scean at night inside an Indian palace

###### Figure 17 Visualization results of HunyuanVideo.

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

Before RLHF

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

After RLHF

Prompt: Man walking in an abandoned city in the rain

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

Before RLHF

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

After RLHF

Prompt: A man leaning on a tree at the beach

[Figure 298]

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

[Figure 303]

Before RLHF

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

After RLHF

Prompt: A pirate ship is sailing in the galaxy sky

###### Figure 18 Visualization results of HunyuanVideo.

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

Before RLHF

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

After RLHF

Prompt: A black Porsche drifting in the desert

[Figure 322]

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

[Figure 327]

Before RLHF

[Figure 328]

[Figure 329]

[Figure 330]

[Figure 331]

[Figure 332]

[Figure 333]

After RLHF

Prompt: Large light blue ice dragon breathing blue fire on town

[Figure 334]

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

Before RLHF

[Figure 340]

[Figure 341]

[Figure 342]

[Figure 343]

[Figure 344]

[Figure 345]

After RLHF

Prompt: A young black girl walking down a street with a lot of huge trees

[Figure 346]

[Figure 347]

[Figure 348]

[Figure 349]

[Figure 350]

[Figure 351]

Before RLHF

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

[Figure 356]

[Figure 357]

After RLHF

Prompt: a cinematic shot of a group of ultra marathon runners, 1980s, a street surrounded by fields, morning hours, anamorphic lens, Kodak Filmstock

###### Figure 19 Visualization results of SkyReels-I2V.

