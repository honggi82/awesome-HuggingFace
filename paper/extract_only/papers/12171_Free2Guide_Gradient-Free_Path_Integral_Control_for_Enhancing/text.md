## Free2Guide: Training-Free Text-to-Video Alignment using Image LVLM

# arXiv:2411.17041v2[cs.CV]19Oct2025

Jaemin Kim, Bryan Sangwoo Kim, Jong Chul Ye Kim Jaechul Graduate School of AI, KAIST

{kjm981995, bryanswkim, jong.ye}@kaist.ac.kr

##### Baseline Free2Guide Baseline Free2Guide

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

"A person is strumming guitar" "A dog and a horse"

##### Baseline Free2Guide Baseline Free2Guide

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

"A happy fuzzy panda playing guitar nearby a campfire, snow mountain in the background"

"The bund Shanghai, vibrant color"

Figure 1. Representative video results using Free2Guide, a novel framework that enables training-Free, gradient-Free video Guidance leveraging a Large Vision-Language Model. Each image shows the first frame of a video.

### Abstract

Diffusion models have achieved impressive results in generative tasks for text-to-video (T2V) synthesis. However, achieving accurate text alignment in T2V generation remains challenging due to the complex temporal dependencies across frames. Existing reinforcement learning (RL)based approaches to enhance text alignment often require differentiable reward functions trained for videos, hindering their scalability and applicability. In this paper, we propose Free2Guide, a novel gradient-free and training-free framework for aligning generated videos with text prompts. Specifically, leveraging principles from path integral control, Free2Guide approximates guidance for diffusion models using non-differentiable reward functions, thereby enabling the integration of powerful black-box Large Vision-Language Models (LVLMs) as reward models. To enable image-trained LVLMs to assess text-to-video alignment, we leverage stitching between video frames and use system prompts to capture sequential attributions. Our framework supports the flexi-

ble ensembling of multiple reward models to synergistically enhance alignment without significant computational overhead. Experimental results confirm that Free2Guide using image-trained LVLMs significantly improves text-to-video alignment, thereby enhancing the overall video quality. Our results and code are available at our project page 1.

### 1. Introduction

Diffusion models [21, 33, 34, 36] have emerged as powerful and versatile tools for generative modeling, achieving state-of-the-art results in tasks that require fine-grained control over content generation, such as text-to-image (T2I) [33] and text-to-video (T2V) generation [7, 15]. However, achieving perfect alignment with text conditions remains a significant challenge [12]. This issue becomes even more challenging in the video domain, where maintaining text-relevant content across frames requires handling complex temporal

1https://kjm981995.github.io/free2guide/

dependencies, often resulting in misalignment between generated frames and the given text prompt.

In the image domain, reinforcement learning (RL)-based methods have been introduced to address challenges in textguided T2I generation by using reward models to estimate human preferences within diffusion models [2, 10, 47, 48]. Previous works mainly focus on either directly fine-tuning the diffusion model with gradients derived from a reward function [6, 30, 31] or employing an RL-based policy gradient approach [2, 10]. While these fine-tuning methods can effectively improve sample alignment, they have notable limitations: the former requires a differentiable reward function, while the latter is typically limited to only few prompts.

Directly adapting these text alignment approaches for the video domain presents two main challenges. First, they often require a dedicated video-specific reward function or additional training on curated video datasets. Collecting large-scale, aligned text-video datasets is far more complex than gathering image data, and developing reward functions tailored to video tasks is similarly difficult. Second, even with trained reward models for the video domain, additional challenges such as substantial memory demands for backpropagation emerge, which grow proportionally as model scale increases (i.e., scaling laws) [19].

An alternative approach involves using differential reward models during inference time to guide diffusion models without fine-tuning model parameters [42]. However, guidance-based methods still require a differentiable reward function, which excludes non-differentiable options like state-of-the-art visual-language model APIs or human preference-based metrics. To address this, recent studies have explored stochastic optimization to guide diffusion models during the sampling process using non-differentiable objective functions in music generation [17], and concurrent research extends this idea within the image domain [50, 51]. However, such methods cannot be directly applied to video diffusion models due to the complex temporal dependencies involved.

To address these issues, here we introduce Free2Guidea novel text-to-video alignment method by leveraging the temporal understanding capabilities of Large VisionLanguage Models (LVLMs). Specifically, Free2Guide aligns text prompts in video generation without requiring gradients from the reward function. More specifically, drawing on principles from path integral control, Free2Guide approximates guidance to align generated videos with text prompts, regardless of the reward function’s differentiability. Another important contribution of this paper is a technique to adapt image-based LVLMs for temporal understanding. In particular, we concatenate video frames in a structured grid layout, and design prompts that explicitly indicate sequence order and reasoning to help LVLMs evaluate videos more comprehensively. By doing so, Free2Guide enables the use

of powerful black-box vision-language models as reward models, improving text-video alignment, as illustrated in Fig.

- 1. Finally, our framework allows for the flexible combination of reward models by eliminating the need for computationally intensive fine-tuning and backpropagation. As such, we explore several combinatorial approaches to collaborate LVLMs with existing large-scale image-based models. Extensive experiments show that our methods improve text alignment and the quality of generated videos.

Our contributions are summarized as follows:

- • We introduce Free2Guide, a novel framework for aligning generated videos with text prompts without requiring gradients from the reward function. To the best of our knowledge, Free2Guide is the first gradient-free guidance approach for text-to-video generation that requires no additional training.
- • We adapt non-differentiable image-based LVLM APIs to enhance text-video alignment by leveraging stitching and prompt design to capture video-specific attributes.
- • We develop an effective ensemble approach that integrates large-scale image-based models to improve video generation guidance.

- 2. Related Work

Text-to-Video diffusion model Text-to-Video diffusion models (e.g., LaVie [43], VideoCrafter [3, 4]) employ diffusion processes to generate coherent video sequences from textual prompts [13, 16, 27]. However, a notable limitation is that video diffusion models often struggle to generate videos that align accurately with the given text prompts, specifically in terms of spatial relationships (e.g., “A on B”) and the representation of temporal style (e.g., “zooming in”).

Diffusion model with LVLM feedback While several approaches have been proposed to improve the diffusion generation process with Large Language Models (LLMs) [11, 25, 46, 52], there has been limited exploration of methods leveraging Large Vision Language Models (LVLMs) that can also handle image domains. Recent works explore the integration of LVLMs as a feedback mechanism to image diffusion models to enhance control and guide diffusion processes. For instance, RPG [49] utilizes an LVLM as a planner to manipulate cross-attention layers in the diffusion model, while Demon [50] demonstrates that LVLMs can guide diffusion in alignment with a given persona. In contrast, our approach leverages LVLMs’ ability to comprehend stitched images, utilizing this capability to enhance frame-to-frame dynamic understanding and applying it within the video domain to improve text-video alignment.

Human Preference Alignment via Reward Models Aligning with human preferences has improved generative quality in diffusion models through fine-tuning diffusion model using reward model gradients (DRaFT [6], AlignProp [30]) or policy gradients (DDPO [2], DPOK [10]). On the other hand,

[Figure 9]

Figure 2. Overall pipeline of training-free gradient-free Free2Guide. Free2Guide leverages LVLMs’ ability to comprehend stitched images, utilizing this capability to enhance frame-to-frame dynamic understanding and applying it within the video domain to improve text-video alignment. It also enables an effective ensemble approach that integrates large-scale image-based models to improve video generation guidance.

DOODL [42] and Demon [50] guide the denoising process to achieve text alignment without training diffusion models. Note, however, that the previously mentioned methods all focus on the image domain. Recent work VADER [31] finetunes a pre-trained video diffusion model using gradients of reward models for aesthetic and text-aligned generation. While this approach shows promising results for using video reward models, it demands substantial memory and does not utilize LVLMs. We address these limitations by proposing a text-video alignment method that approximates image reward gradients without fine-tuning.

Zeroth-order gradient approximation Zeroth-order gradients, or gradient-free approaches, approximate gradients of non-differentiable functions by evaluating multiple points [26, 28]. In diffusion-based inverse problems, methods like EnKF [51] and SCG [17] leverage gradient-free approximations to guide sampling based on non-differentiable or black-box forward models. However, there is a lack of research specifically focused on gradient-free approaches to guide sampling for video diffusion models. In video diffusion models, approximating a black-box reward model with a zeroth-order gradient is advantageous, as gradients of the reward are unavailable and the high-dimensional space of video data imposes memory limitations.

### 3. Preliminaries

#### 3.1. Video Latent Diffusion Model

Video Latent Diffusion Models (VLDMs) learn a stochastic process by iteratively denoising random noise generated by the forward diffusion process [7]

q(zt|z0) = N(zt;√1 − α¯t z0,α¯tI), (1)

where z0 = E(x) is the latent encoding of the clean video with encoder E and α¯t is a noise scheduling coefficient at

timestep t. The VLDM estimates the noise in zt by minimizing the following objective:

0,ϵ,t,c ∥ϵ − ϵθ(zt,t,c)∥2 , (2) where ϵ ∼ N(0,I) and c represents the conditioning input.

Ez

To retrieve a clean latent representation, we use a reversetime Stochastic Differential Equation (SDE) sampling process:

dzt = f¯(zt)dt + g(zt)dw¯ = f(zt) − g(zt)2∇zt

(3)

log p(zt) dt + g(zt)dw¯ ,

where f and f¯are the drift term for the forward SDE and reverse SDE, respectively, g is the diffusion coefficient, and w¯ represents a reverse time Wiener process. The initial point for reverse SDE is sampled from a normal Gaussian distribution. By discretizing the reverse SDE with an appropriate noise schedule, the VLDM retrieves a clean latent representation based on the DDIM [35] trajectory,

1 − α¯t−1 1 − α¯t

α¯t α¯t−1 z0|t =

1 −

σt := η

√1 − α¯tϵθ(zt,t,c)

1 √α¯t

zt −

zt−1 = √α¯t−1z0|t + 1 − α¯t−1 − σt2ϵθ(zt,t,c) + σtϵ,

(4)

where σt controls the stochasticity of sampling, ϵ ∼ N(0,I) and z0|t = E[z0|zt] denotes the posterior mean or denoised version of zt, computed by Tweedie’s formula [8]. To transform the latent representation back to the video domain, a decoder D is used to decode the latent.

#### 3.2. Guidance in Diffusion Model

Given the reverse SDE in Eq. (3), our goal is to obtain the optimal control u(zt) :

dzt = f ¯(zt) + u(zt) dt + g(zt)dw¯ , (5)

which directs the sampling process toward target distribution p(zt|y), where y represent a desired condition, such as label, class or text prompt [45]. In classifier guidance [29], if an auxiliary classifier is available to estimate the likelihood p(y|zt), the control term can be defined as

u(zt) = −g(zt)2w∇zt

log p(y|zt), (6)

where w is a scaling factor that adjusts the strength of the guidance. This control term follows from applying the Bayes rule to express p(zt|y) ∝ p(zt|y)p(y|zt)w.

One might consider adapting classifier guidance by treating the reward model as a classifier. However, this approach presents two challenges: the reward model is not trained on noisy latent representations zt and requires differentiability. To alleviate these limitations, we utilize a path integral control approach with zeroth-order gradient approximation, as described in the following Section 3.3.

#### 3.3. Path Integral Control

Considering the diffusion model as an entropy regularized Markov Decision Process (MDP), we can conceptualize reverse SDE in the Reinforcement Learning (RL) framework [2, 10, 40] with the state st and the action at corresponding to the input zt. In this formula, the optimal policy p∗ maximizes the following objective:

Ep[r(z0) − α

τ=T

1

DKL(p(zτ−1|zτ)||pθ(zτ−1|zτ))], (7)

where α is a coefficient of KL divergence with original policy pθ defined by diffusion model. Let pθ(zt−1|zt) = N(µt,σt2I) be a reverse transition distribution in the SDE for the diffusion model and pθ(z0:t) := pθ(zt)Πtτ=1p(zτ−1|zτ). We can define a value function as

v(zt) α

v(zt−1) α

pθ(zt−1|zt)dzt−1

exp

= exp

r(z0)

= Ep

α |zt ,

θ(z0:t) exp

(8) satisfying v(z0) = r(z0) is a reward function [40].

The optimal control to address the entropy-regularized MDP system can be obtained by solving the HamiltonJacobi-Bellman (HJB) equation as follows [17, 41]:

σt2∇zt

v(zt) α

. (9)

u(zt) = −

However, this term requires the gradient of the value function. To bypass the gradient requirements, one can use path integral control, which is an approach to estimate the optimal control (or guidance) based on the principles of stochastic optimal control [20, 39, 41]. In [17], the optimal control can be approximated as

E exp r(z

0)

α (zt−1 − µt)|zt E exp r(z

. (10)

u(zt) ≃ −

0)

α |zt

While SCG [17] utilizes this optimal control with diffusion models to solve inverse problems in image domain, we aim to use LVLMs to guide videos toward improved text alignment.

### 4. Free2Guide

In this section, we introduce Free2Guide, a framework that uses a non-differentiable reward model to guide video generation during the sampling process. In Sec. 4.1, we discuss how to apply image-based reward models, including LVLM, for text-video alignment. Sec. 4.2 outlines methods for ensembling multiple reward models to achieve synergistic effects. Finally, we interpret the diffusion model as an entropy-regularized MDP and describe its practical implementation (Sec. 4.3).

#### 4.1. Video Guidance leveraging Image LVLMs

Motivation By leveraging the path integral control approach discussed in Sec. 3.3, we can guide the reverse process without relying on the gradient of the reward function. If the reward model r in Eq. (10) assesses the alignment of the generated video with the text prompt, it can help steer the video output to enhance fidelity to the prompt. However, due to the complexity of videos compared to static images, there are limited large-scale models specifically trained for video and text alignment. We analyze the impact of videobased reward models on video guidance and find that their effectiveness is limited (see Appendix D.1).

Applying these image-based reward models directly for video guidance, of course, presents challenges. Image-based models are not designed to process time-dependent features, such as motion, flow, and dynamics, so specific adaptations are required for these models to assess text-video alignment. As shown in Algorithm 1, we calculate the reward for a video by summing frame-by-frame rewards from the imagebased model. This approach enables alignment with spatial information within individual video frames but still lacks guidance on temporal dynamics.

Image-based LVLMs as a Video Reward Model Although LVLMs are trained on static image-text data, their extensive pretraining across diverse visual contexts enables them to implicitly capture elements of motion. As shown in

- Table 1 of [38], treating video as an image grid in LVLMs strongly correlates with human evaluation. Furthermore, results from [22, 24] demonstrate that image-based LVLMs achieve performance comparable to video-specific LLMs in video QA, validating our approach.

Accordingly, to adapt LVLMs for evaluating multiple frames simultaneously, we employ a method called stitching, which combines key frames into a single composite image (see Fig. 2). Specifically, we first sample key frames from the video and arrange them in a structured grid layout, labeling each frame with its index to indicate its position in the sequence. This approach allows LVLMs to process temporal information by leveraging spatial relationship between frames.

Then, to help LVLMs understand frame order within the composite image, we provide explicit sequence instructions through a system prompt. This efficient adaptation enables LVLMs to recognize frame order by referencing frame numbers rather than processing them linearly. We incorporate Zero-shot Chain-of-Thought [23] in the system prompt to enhance reasoning ability and mitigate hallucinations. In the user prompt, we instruct the LVLM to consider every key frame individually and evaluate the alignment score between the composite image and the text prompt on a scale of 1 to 9. The full system instructions and query templates are detailed in Appendix A.

#### 4.2. Ensembling Reward Functions

Unlike gradient-based guidance, our method significantly reduces memory requirements by avoiding the computationally intensive backpropagation process. This enables us to concurrently employ multiple rewards for sampling guidance, potentially leading to synergistic benefits with largescale image models. We explore ensemble methods that allow LVLMs to incorporate temporal information, thereby supporting more effective guidance for video alignment when combined with large-scale image models. Note that Demon [50], a concurrent work that also proposed ensemble rewards, failed to show the synergy effect of ensemble and did not have to handle temporal information.

Given the n videos {Vi}ni=1, we propose three ensembling methods to combine multiple reward models: Weighted Sum, Normalized Sum, and Consensus.

- • Weighted Sum: This method combines the outputs by computing a fixed weighted sum, allowing us to control the influence of each reward model.

Rewardens(Vi,r1,r2) = βr1(Vi) +(1 −β)r2(Vi), (11)

where β ∈ [0,1] is a constant weight factor that balances the contributions of reward models r1 and r2.

- • Normalized Sum: To ensure a balanced contribution of each reward models, we first normalize each reward’s

- Algorithm 1 Reward Model r(D(z0|t),c) Require: Reward function r, condition c, prompt p, de-

coded frames x0|t := D(z0|t), and key frames k ⊂ [1,N]

- 1: if r is CLIP then
- 2: reward ← i∈k sim(r(xi0|t),r(c))
- 3: else if r is ImageReward then
- 4: reward ← i∈k r(xi0|t,c)
- 5: else if r is LVLM then
- 6: reward ← r(concati∈k(xi0|t),c,p)
- 7: end if
- 8: return reward

- Algorithm 2 Free2Guide Require: Video diffusion model ϵθ, reward function r, de-

coder D, noise scheduling parameter {α¯t}Tt=1,{σt}Tt=1

- 1: for t = T to 1 do
- 2: z0|t ← √α¯1

t−1

zt −

√1 − α¯tϵθ(zt)

- 3: zˆt−1 ←

√α¯tz0|t + 1 − α¯t−1 − σt2ϵθ(zt)

- 4: ϵ1,··· ,ϵn ∼ N(0,I)
- 5: zti−1 ← zˆt−1 + σtϵi
- 6: z0i|t−1 ← √α¯1

t−1

zti−1 −

√1 − α¯t−1ϵθ(zti−1)

- 7: r1 ←LVLM
- 8: if Ensemble then
- 9: r2 ∈ {CLIP, ImageReward}
- 10: j ← argmaxi Rewardens(D(z0i|t−1),r1,r2) From Sec. 4.2.
- 11: else
- 12: j ← argmaxi r1(D(z0i|t−1),c)
- 13: end if
- 14: zt−1 ← ztj−1
- 15: end for
- 16: return z0

output to the range [0,1], then sum these normalized values to get the final ensemble reward.

r(Vi) − min(r(Vi)) max(r(Vi)) − min(r(Vi))

Rewardens(Vi,r1,r2) =

,

r

(12) where max(r),min(r) represents the maximum and minimum score from n reward outputs.

• Consensus: Inspired by the Borda count [9], each reward model ranks the videos from best to worst, assigning pointsr based on their rank. The top-ranked video receives the maximum points, down to 1 point for the lowest rank. The total reward for each video Vi is the sum of points

from both reward model.

Rewardens(Vi,r1,r2) = pointsr

(Vi) + pointsr

(Vi).

2

1

(13)

#### 4.3. Guidance using Path Integral Control

To guide the reverse sampling process without computing the gradient of the reward function, we utilize the framework outlined in Eq. (10). However, the expectation of the reward function in Eq. (10) demands extensive network function evaluations (NFE) by solving complex differential equations, such as PF-ODE [36]. Inspired by [17], we instead apply the DPS [5] approach to approximate Eq. (8) by using the posterior mean of zt, as defined in Eq. (4). Following DPS, we can set p(z0:t) = δ(z − E[z0|zt]) using Direc delta distribution δ in which case Eq. (10) becomes:

θ(zt−1|zt) exp r(z0α|t−1) (zt−1 − µt) exp r(zα0|t)

Ep

u(zt) ≃ −

.

(14) To approximate this expectation using the Monte Carlo method, we sample n different zt−1 through the reverse SDE as outlined in Eq. (4). Then we assume α → 0 to obtain optimal control. Under this assumption, Eq. (3) becomes equivalent to selecting the zt−1 that maximizes the reward of z0|t−1 [17]. While [17] arbitrarily weighted the reward function and assumed the weight to be zero, we interpret this as relaxing the entropy-regularization term in Eq. (7) by defining the diffusion process as an entropy-regularized MDP. In practical terms, this approach eliminates careful parameter exploration by selecting zt−1 with the largest reward.

By following this adjusted sampling strategy as described in Algorithm 2, Free2Guide can efficiently steer video generation towards better alignment with the reward signals.

### 5. Experiments

Baselines and Sampling Strategy. We use open-source textto-video diffusion models, LaVie [43] and VideoCrafter2 [4], as baseline models. The generated videos contain 16 frames with a resolution of 320 × 512. We employ LVLM as GPT-4o-2024-08-06 [1] using OpenAI APIs. We employ two large-scale models CLIP [32] and ImageReward [48], to validate that LVLM’s capability to account for temporal dynamics can enhance text-video alignment when used alongside large-scale image reward models. In CLIP, we can assess alignment by measuring cosine similarity between text and image embeddings. On the other hand, we can use ImageReward output as an reward since it predicts human preference for image-text pairs. For adaptation to the video domain, we extract key frames from each denoised video and sum the reward for each frame to evaluate overall alignment, as outlined in Algorithm 1.

We employ stochastic DDIM sampling with η = 1 in Eq. (4) for a total of T = 50 steps and apply classifier-free guidance [14] using a guidance scale of w = 7.5 for LaVie and w = 12 for VideoCrafter2. The number of samples at each guidance step is set to n = 5 for LaVie and n = 10 for VideoCrafter2. Guidance is applied during the early sampling steps, specifically within t ∈ [T,T − 5]. In the weighted sum ensemble, we assign a weight of β = 0.75 to the LVLM reward.

Text Alignment Evaluation. We conduct quantitative evaluation using VBench [18], a benchmark designed to evaluate the alignment of text-to-video (T2V) models with respect to a text prompt. Our evaluation protocol measures text alignment across six dimensions: Appearance Style, Temporal Style, Human Action, Multiple Objects, Spatial Relationship and Overall Consistency. For a fair comparison, we use standardized prompts for each metric, ensuring consistent conditions across different models.

General Video Quality Evaluation. In addition to text alignment, we evaluate the general quality of generated videos independently of text prompts using six metrics in VBench: Subject Consistency, Background Consistency, Motion Smoothness, Dynamic Degree, Aesthetic Quality, and Imaging Quality.

Video-specific Attributes. Since VBench prompts involve limited movement, we conducted additional experiments using T2V-CompBench [38] to analyze video-specific motion and dynamics. We measure Dynamic Attribution Binding, which evaluates how well models handle state changes (e.g. shape and texture) and color variations over time.

#### 5.1. Results

In this section, we present both qualitative and quantitative results to demonstrate the effectiveness of our method. The top four rows of Fig. 3 shows visual comparisons between the baseline and reward models. We observe that leveraging the GPT-4o model to assess text-video alignment improves alignment with respect to temporal dynamics (e.g. "tilt down") and semantic representation (e.g. "A and B"). These results indicate that LVLM can account for temporal information by processing multiple sub-frames of video simultaneously, with strong performance in spatial understanding.

Building on LVLMs’ capability to account for temporal dynamics, we validate the feasibility of ensembling techniques that integrate guidance from large-scale image mod-

Method Avg.

Method Avg.

LaVie + CLIP 0.5712 + GPTWeighted Sum 0.5738 + GPTNormalized Sum 0.5734 + GPTConsensus 0.5679

LaVie + ImageReward 0.5676 + GPTWeighted Sum 0.5726 + GPTNormalized Sum 0.5715 + GPTConsensus 0.5692

Table 1. Qualitative comparison between ensemble methods.

[Figure 10]

Figure 3. Qualitative results of our method. Comparison with LaVie on the left and VideoCrafter2 on the right.

Method Appearance Style Temporal Style Human Action Multiple Objects Spatial Relationship Overall Consistency Avg.

LaVie [43] 0.2312 0.2502 0.9300 0.2027 0.3496 0.2694 0.3722 + GPT 0.2366 (+2.3%) 0.2508 (+0.2%) 0.9300 (-0.0%) 0.2546 (+25.6%) 0.3531 (+1.0%) 0.2709 (+0.6%) 0.3827

+ CLIP 0.2370 (+2.5%) 0.2490 (-0.5%) 0.9400 (+1.1%) 0.2607 (+28.6%) 0.3074 (-12.1%) 0.2738 (+1.6%) 0.3780 ++ GPT 0.2350 (+1.6%) 0.2487 (-0.6%) 1.000 (+7.5%) 0.2447 (+20.7%) 0.3180 (-9.0%) 0.2742 (+1.7%) 0.3868

+ ImageReward 0.2360 (+2.1%) 0.2483 (-0.8%) 0.9300 (-0.0%) 0.2637 (+30.1%) 0.2614 (-25.2%) 0.2728 (+1.2%) 0.3687 ++ GPT 0.2373 (+2.6%) 0.2497 (-0.2%) 0.9400 (+1.1%) 0.2462 (+21.4%) 0.3014 (-13.8%) 0.2772 (+2.9%) 0.3753

VideoCrafter2 [4] 0.2490 0.2567 0.9300 0.3880 0.3760 0.2778 0.4129 + GPT 0.2504 (+0.6%) 0.2568 (+0.0%) 0.9500 (+2.2%) 0.4878 (+25.7%) 0.4225 (+12.4%) 0.2872 (+3.4%) 0.4425

+ CLIP 0.2542 (+2.1%) 0.2621 (+2.1%) 0.9300 (-0.0%) 0.4261 (+9.8%) 0.2923 (-22.3%) 0.2802 (+0.9%) 0.4075 ++ GPT 0.2490 (+0.0%) 0.2612 (+1.8%) 0.9600 (+3.2%) 0.4474 (+15.3%) 0.3361 (-10.6%) 0.2837 (+2.1%) 0.4229

+ ImageReward 0.2513 (+0.9%) 0.2574 (+0.3%) 0.9700 (+4.3%) 0.4733 (+22.0%) 0.4264 (+13.4%) 0.2826 (+1.7%) 0.4435 ++ GPT 0.2533 (+1.7%) 0.2607 (+1.6%) 0.9400 (+1.1%) 0.5160 (+33.0%) 0.4371 (+16.3%) 0.2828 (+1.8%) 0.4483

- Table 2. Quantitative evaluation on text alignment. Higher numbers indicate better alignment with the text prompt. The numbers in parentheses denote the performance difference from the baseline.

els to improve text-video alignment. This approach enables LVLMs to process temporal information, enhancing the quality of guidance. In Table 1, we explore the most effective ensemble method by comparing average scores on text alignment and general video quality evaluation from VBench. We

find that assigning more weight to LVLM outperformed the alternative of balancing model contributions equally in the ensemble, indicating that the role of LVLM is significant. Thus, we adopt the weighted sum ensemble as the default setting. The bottom four rows of Fig. 3 also illustrate qualita-

Method ConsistencySubject BackgroundConsistency SmoothnessMotion DynamicDegree AestheticQuality ImagingQuality Avg. LaVie [43] 0.9450 0.9689 0.9718 0.4799 0.5687 0.6611 0.7659 + GPT 0.9470 0.9693 0.9742 0.4725 0.5726 0.6615 0.7662 + CLIP 0.9495 0.9712 0.9735 0.4560 0.5727 0.6637 0.7644 ++ GPT 0.9622 0.9781 0.9804 0.3703 0.5951 0.6795 0.7609 + IR 0.9443 0.9681 0.9732 0.4872 0.5664 0.6605 0.7666 ++ GPT 0.9758 0.9813 0.9832 0.5165 0.5662 0.6530 0.7699 VC2 [4] 0.9658 0.9748 0.9818 0.3846 0.5860 0.6772 0.7617 + GPT 0.9746 0.9800 0.9827 0.2949 0.5977 0.6924 0.7537 + CLIP 0.9762 0.9816 0.9839 0.2491 0.6037 0.6886 0.7472 ++ GPT 0.9770 0.9823 0.9838 0.2399 0.6042 0.6878 0.7458 + IR 0.9739 0.9801 0.9828 0.2711 0.5994 0.6857 0.7488 ++ GPT 0.9758 0.9813 0.9832 0.2564 0.6039 0.6877 0.7480

- Table 3. Comparison of the general quality of the generated video independent of the text prompt. Higher numbers indicate better video quality. ‘VC2’ is VideoCrafter2 and ‘IR’ is ImageReward.

tive results for ensembling, showing that combining GPT-4o with other image reward models accurately resolves issues related to dynamics or multiple objects that standalone reward models struggle to properly identify, while maintaining overall structure.

For more detailed evaluations, we compare the quantitative results in Table 2 to assess text-video alignment. Analysis of the average evaluation scores reveals that incorporating LVLM consistently outperforms configurations that exclude it. Specifically, we observe the most significant improvement in handling Spatial Relationship across baselines. Since CLIP has a limited zero-shot spatial reasoning capability [37], the text alignment performance decreases in Spatial Relationship when using CLIP alone. However, ensembling with LVLM offers additional cues that help CLIP to better account for spatial semantics, leading to performance improvements. Furthermore, incorporating LVLM enhances Human action, Overall Consistency in overall case and Temporal Style, except when using CLIP as the reward model. Since LVLM can understand temporal nuances by processing multiple frames at once, it improves performance by supporting the alignment of temporal movement.

Additionally, we compare general video quality in Table 3. We confirm that even without explicit guidance for consistency or motion, alignment with text prompts improves most quality metrics except for Dynamic Degree. This metric often trades off with consistency but can be improved by ensembling GPT-4o with ImageReward in the LaVie model. This suggests that ImageReward compensates for the performance drop in Dynamic Degree that GPT-4o alone does not address, resulting in the best performance.

Method

Dynamic Attribution (↑)

LaVie 0.01242 + GPT 0.01360

VC2 0.00663 + GPT 0.00770

- Table 4. Results for T2V-CompBench. Figure 4. Example of T2V-CompBench.

[Figure 11]

As shown in Table 4, leveraging LVLM improves performance in Dynamic Attribution Binding. Figure 4 illustrates an example video where the water gradually fills up over time in response to a given prompt when utilizing LVLM, whereas the baseline model fails to capture this progression.

Method NFEs Avg.

Baseline 100 0.5815 Best-of-N 100 0.5802 Ours 100 0.5981

Table 5. Fixed NFE comparison on VBench.

Method CLIP (↑) IR (↑) GPT (↑)

VC2 30.39 -0.10 7.09 +GPT 30.90 0.23 7.28

+CLIP 30.96 0.14 7.11

- ++GPT 30.95 0.20 7.07

+IR 30.92 0.22 7.28

- ++GPT 30.96 0.28 7.33 Table 6. Reward robustness.

#### 5.2. Analysis

Computational efficiency To evaluate the computational efficiency of our method, we conduct experiments under a fixed NFE budget of 100 using VideoCrafter2, as shown in Table 5. The Baseline uses a single 100-step inference path, while Best-of-N selects the highest LVLM reward from two 50-step paths. Our approach uses 50 steps, with six samples in the first 10 steps, while the remaining 40 steps follow the baseline procedure. Notably, simply selecting from multiple final outputs is ineffective, as it does not influence the denoising process. In contrast, our method actively guides generation throughout sampling, leading to improved text alignment and coherence that cannot be achieved through post-hoc selection.

Robustness of Rewards We verify that our method achieves robust performance without overfitting to any particular reward, avoiding reward hacking, a common issue in RL literature. Table 6 compares the rewards for the final video outputs generated by each method. Video guidance ensembled with LVLM generally achieves higher metrics, exhibiting a trend similar to the text alignment results in Table 2. These findings indicate that the ensemble approach is not over-optimized for a particular reward, enhancing robustness across diverse evaluation criteria. Additional ablation studies can be found in Appendix C.

### 6. Conclusion

In this paper, we introduced Free2Guide, a novel gradientfree framework to enhance text-video alignment in diffusionbased generative models without relying on reward gradients. By approximating the gradient of the reward function, Free2Guide effectively integrates non-differentiable reward models, including powerful black-box LVLMs, to steer the video generation process towards better alignment. Our experiments demonstrate that Free2Guide consistently improves alignment with text prompts and general video quality. By enabling ensembling with LVLM, our method benefits from synergistic effects, further enhancing performance.

### 7. Acknowledgments

This work was supported by the National Research Foundation of Korea under Grant RS-2024-00336454, and by the Institute of Information & Communications Technology Planning & Evaluation (IITP) grant funded by the Korean government (MSIT) (No. RS-2024-00457882, AI Research Hub Project; No. RS-2025-02304967, AI Star Fellowship (KAIST))

### References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023. 6
- [2] Kevin Black, Michael Janner, Yilun Du, Ilya Kostrikov, and Sergey Levine. Training diffusion models with reinforcement learning. arXiv preprint arXiv:2305.13301, 2023. 2, 4
- [3] Haoxin Chen, Menghan Xia, Yingqing He, Yong Zhang, Xiaodong Cun, Shaoshu Yang, Jinbo Xing, Yaofang Liu, Qifeng Chen, Xintao Wang, Chao Weng, and Ying Shan. Videocrafter1: Open diffusion models for high-quality video generation, 2023. 2
- [4] Haoxin Chen, Yong Zhang, Xiaodong Cun, Menghan Xia, Xintao Wang, Chao Weng, and Ying Shan. Videocrafter2: Overcoming data limitations for high-quality video diffusion models, 2024. 2, 6, 7, 8
- [5] Hyungjin Chung, Jeongsol Kim, Michael Thompson Mccann, Marc Louis Klasky, and Jong Chul Ye. Diffusion posterior sampling for general noisy inverse problems. In International Conference on Learning Representations, 2023. 6
- [6] Kevin Clark, Paul Vicol, Kevin Swersky, and David J Fleet. Directly fine-tuning diffusion models on differentiable rewards. arXiv preprint arXiv:2309.17400, 2023. 2
- [7] Prafulla Dhariwal and Alexander Quinn Nichol. Diffusion models beat GANs on image synthesis. In Advances in Neural Information Processing Systems, 2021. 1, 3
- [8] Bradley Efron. Tweedie’s formula and selection bias. Journal of the American Statistical Association, 106(496):1602–1614,

2011. 3

- [9] Peter Emerson. The original borda count and partial voting. Social Choice and Welfare, 40(2):353–358, 2013. 5
- [10] Ying Fan, Olivia Watkins, Yuqing Du, Hao Liu, Moonkyung Ryu, Craig Boutilier, Pieter Abbeel, Mohammad Ghavamzadeh, Kangwook Lee, and Kimin Lee. Reinforcement learning for fine-tuning text-to-image diffusion models. Advances in Neural Information Processing Systems, 36, 2024. 2, 4
- [11] Weixi Feng, Wanrong Zhu, Tsu-jui Fu, Varun Jampani, Arjun Akula, Xuehai He, Sugato Basu, Xin Eric Wang, and William Yang Wang. Layoutgpt: Compositional visual planning and generation with large language models. Advances in Neural Information Processing Systems, 36, 2024. 2
- [12] Tejas Gokhale, Hamid Palangi, Besmira Nushi, Vibhav Vineet, Eric Horvitz, Ece Kamar, Chitta Baral, and Yezhou Yang. Benchmarking spatial relationships in text-to-image generation. arXiv preprint arXiv:2212.10015, 2022. 1

- [13] Yingqing He, Tianyu Yang, Yong Zhang, Ying Shan, and Qifeng Chen. Latent video diffusion models for high-fidelity long video generation. arXiv preprint arXiv:2211.13221,

2022. 2

- [14] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022. 6
- [15] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in Neural Information Processing Systems, 33:6840–6851, 2020. 1
- [16] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video diffusion models. Advances in Neural Information Processing Systems, 35:8633–8646, 2022. 2
- [17] Yujia Huang, Adishree Ghatare, Yuanzhe Liu, Ziniu Hu, Qinsheng Zhang, Chandramouli S Sastry, Siddharth Gururani, Sageev Oore, and Yisong Yue. Symbolic music generation with non-differentiable rule guided diffusion. arXiv preprint arXiv:2402.14285, 2024. 2, 3, 4, 6
- [18] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, Yaohui Wang, Xinyuan Chen, Limin Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. VBench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024. 6
- [19] Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361,

2020. 2

- [20] Hilbert J Kappen. Path integrals and symmetry breaking for optimal control theory. Journal of statistical mechanics: theory and experiment, 2005(11):P11011, 2005. 4
- [21] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. In Proc. NeurIPS, 2022. 1
- [22] Wonkyun Kim, Changin Choi, Wonseok Lee, and Wonjong Rhee. An image grid can be worth a video: Zero-shot video question answering using a vlm. IEEE Access, 2024. 5
- [23] Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large language models are zero-shot reasoners. Advances in neural information processing systems, 35:22199–22213, 2022. 5
- [24] K Li et al. Mvbench: A comprehensive multi-modal video understanding benchmark. In CVPR, 2024. 5
- [25] Long Lian, Baifeng Shi, Adam Yala, Trevor Darrell, and Boyi Li. Llm-grounded video diffusion models. arXiv preprint arXiv:2309.17444, 2023. 2
- [26] Sijia Liu, Pin-Yu Chen, Bhavya Kailkhura, Gaoyuan Zhang, Alfred O Hero III, and Pramod K Varshney. A primer on zeroth-order optimization in signal processing and machine learning: Principals, recent advances, and applications. IEEE Signal Processing Magazine, 37(5):43–54, 2020. 3
- [27] Simian Luo, Yiqin Tan, Longbo Huang, Jian Li, and Hang Zhao. Latent consistency models: Synthesizing highresolution images with few-step inference. arXiv preprint arXiv:2310.04378, 2023. 2

- [28] Yurii Nesterov and Vladimir Spokoiny. Random gradient-free minimization of convex functions. Foundations of Computational Mathematics, 17(2):527–566, 2017. 3
- [29] Weili Nie, Brandon Guo, Yujia Huang, Chaowei Xiao, Arash Vahdat, and Anima Anandkumar. Diffusion models for adversarial purification. arXiv preprint arXiv:2205.07460, 2022. 4
- [30] Mihir Prabhudesai, Anirudh Goyal, Deepak Pathak, and Katerina Fragkiadaki. Aligning text-to-image diffusion models with reward backpropagation. arXiv preprint arXiv:2310.03739, 2023. 2
- [31] Mihir Prabhudesai, Russell Mendonca, Zheyang Qin, Katerina Fragkiadaki, and Deepak Pathak. Video diffusion alignment via reward gradients. arXiv preprint arXiv:2407.08737,

2024. 2, 3

- [32] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 6
- [33] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10684–10695, 2022. 1
- [34] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International Conference on Machine Learning, pages 2256–2265. PMLR, 2015. 1
- [35] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In 9th International Conference on Learning Representations, ICLR, 2021. 3
- [36] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In International Conference on Learning Representations,

2021. 1, 6

- [37] Sanjay Subramanian, William Merrill, Trevor Darrell, Matt Gardner, Sameer Singh, and Anna Rohrbach. Reclip: A strong zero-shot baseline for referring expression comprehension. arXiv preprint arXiv:2204.05991, 2022. 8
- [38] K Sun et al. T2v-compbench: A comprehensive benchmark for compositional text-to-video generation. arXiv, 2024. 5, 6
- [39] Evangelos Theodorou, Jonas Buchli, and Stefan Schaal. A generalized path integral control approach to reinforcement learning. The Journal of Machine Learning Research, 11: 3137–3181, 2010. 4
- [40] Masatoshi Uehara, Yulai Zhao, Tommaso Biancalani, and Sergey Levine. Understanding reinforcement learning-based fine-tuning of diffusion models: A tutorial and review. arXiv preprint arXiv:2407.13734, 2024. 4
- [41] Masatoshi Uehara, Yulai Zhao, Kevin Black, Ehsan Hajiramezanali, Gabriele Scalia, Nathaniel Lee Diamant, Alex M Tseng, Tommaso Biancalani, and Sergey Levine. Fine-tuning of continuous-time diffusion models as entropy-regularized control. arXiv preprint arXiv:2402.15194, 2024. 4

- [42] Bram Wallace, Akash Gokul, Stefano Ermon, and Nikhil Naik. End-to-end diffusion latent optimization improves classifier guidance. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7280–7290, 2023. 2, 3
- [43] Yaohui Wang, Xinyuan Chen, Xin Ma, Shangchen Zhou, Ziqi Huang, Yi Wang, Ceyuan Yang, Yinan He, Jiashuo Yu, Peiqing Yang, et al. Lavie: High-quality video generation with cascaded latent diffusion models. arXiv preprint arXiv:2309.15103, 2023. 2, 6, 7, 8, 3
- [44] Yi Wang, Yinan He, Yizhuo Li, Kunchang Li, Jiashuo Yu, Xin Ma, Xinhao Li, Guo Chen, Xinyuan Chen, Yaohui Wang, et al. Internvid: A large-scale video-text dataset for multimodal understanding and generation. In The Twelfth International Conference on Learning Representations, 2023. 3
- [45] David Williams and L Chris G Rogers. Diffusions, Markov processes, and martingales. John Wiley & Sons, 1979. 4
- [46] Tsung-Han Wu, Long Lian, Joseph E Gonzalez, Boyi Li, and Trevor Darrell. Self-correcting llm-controlled diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6327–6336,

2024. 2

- [47] Xiaoshi Wu, Yiming Hao, Keqiang Sun, Yixiong Chen, Feng Zhu, Rui Zhao, and Hongsheng Li. Human preference score v2: A solid benchmark for evaluating human preferences of text-to-image synthesis. arXiv preprint arXiv:2306.09341,

2023. 2

- [48] Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong. Imagereward: Learning and evaluating human preferences for text-to-image generation. Advances in Neural Information Processing Systems, 36, 2024. 2, 6
- [49] Ling Yang, Zhaochen Yu, Chenlin Meng, Minkai Xu, Stefano Ermon, and CUI Bin. Mastering text-to-image diffusion: Recaptioning, planning, and generating with multimodal llms. In Forty-first International Conference on Machine Learning,

2024. 2

- [50] Po-Hung Yeh, Kuang-Huei Lee, and Jun-Cheng Chen. Training-free diffusion model alignment with sampling demons. arXiv preprint arXiv:2410.05760, 2024. 2, 3, 5
- [51] Hongkai Zheng, Wenda Chu, Austin Wang, Nikola Kovachki, Ricardo Baptista, and Yisong Yue. Ensemble kalman diffusion guidance: A derivative-free method for inverse problems. arXiv preprint arXiv:2409.20175, 2024. 2, 3
- [52] Shanshan Zhong, Zhongzhan Huang, Weushao Wen, Jinghui Qin, and Liang Lin. Sur-adapter: Enhancing text-to-image pre-trained diffusion models with large language models. In Proceedings of the 31st ACM International Conference on Multimedia, pages 567–578, 2023. 2

### A. Implementation Details

#### A.1. Model Checkpoints

We use the pre-trained T2V diffusion model LaVie and VideoCrafter2, available at https://github.com/Vchitect/ LaVie and https://github.com/AILab-CVC/VideoCrafter, respectively. For LaVie, the Stable Diffusion v1.4 model is employed to encode and decode latent. We also utilize CLIP from https://huggingface.co/openai/clip-vit-basepatch32 and the ImageReward model from https://github.com/THUDM/ImageReward.

#### A.2. Evaluation Details

During the video guidance process, we extract key frames from the video—specifically, the first, sixth, eleventh, and sixteenth frames—and assess the reward. When using an LVLM as the reward model, we concatenate the key frames using the following scripts:

- 1 fig , axes = plt.subplots(2, 2, figsize=(12, 8))

- 2 key_frames = [0, 5, 10, 15]

- 3

- 4 for idx , frame in enumerate(key_frames):

- 5 ax = axes[idx // 2, idx % 2]

- 6 ax.imshow(video[0, :, frame , :, :].permute(1, 2, 0).cpu().numpy())

- 7 ax.axis(’off’)

- 8 ax.set_title(f’Frame {frame + 1}’)

- 9

- 10 # Adjust the layout and show the plot

- 11 plt.tight_layout()

- 12 plt.savefig(f’frame_{i}_{j}.png’) Listing 1. Pseudo-code for stitching key frames at once.

Next, we provide a system instruction that allows the LVLM to understand the sequence order and explicitly describes the task it should perform.

- 1 You are a useful helper that responds to video quality assessments.

- 2 The given image is a grid of four key frames of a video: the top left is the first frame, the top right is the second

- 3 frame, the bottom left is the third frame, and finally the bottom right is the fourth frame.

- 4 Answer the reason first and the final answer later . Start the reason first with ‘Reasoning: ’ in front of the reason part

- 5 and review your reasoning logically .

- 6 After reviewing your reasoning , give the final answer with ‘Answer: ’.

- 7 You should check all frame and comparing them, and ensure your reasoning leads to a sound final answer.

- 8 Your final ‘answer’ should one score only and the score must be from 1 to 9 without decimals.

- 9 Let’s think step by step . Listing 2. System instruction for GPT-4o

For a given video, we input the user prompt to the LVLM as follows:

- 1 For a given image as keyframes of video, Rate the following questions :

- 2 Considering all four images, does the prompt, prompt, describe the video well enough?

- 3 Review your reasoning thoroughly and then respond with your final decision prefixed by Answer: ’. Listing 3. User prompt for GPT-4o

where prompt is the given text prompt (e.g. “a bird and a cat”)

### B. Limitation

Sampling in our approach requires additional processing time to approximate the gradient. While our approach extends sampling time compared to baseline, it uniquely enables guidance with non-differentiable reward models such as LVLM APIs. Additionally, the effectiveness of our framework is influenced by the accuracy of the reward function, which opens avenues for further improvements as reward models continue to advance.

### C. Additional Ablation Study

Number of Samples We analyze the effect of the sampling quantity on text alignment performance, evaluating the average text alignment score using the LaVie model with a CLIP reward model. As shown in Table 7, we find an optimal sampling size at n = 5. Increasing the number of samples increases the likelihood of selecting a denoised video that aligns with the desired control. However, excessive sampling introduces a risk: errors predicted by Tweedie’s formula in initial sampling steps may result in irreversible changes, affecting video quality negatively.

Guidance Range We also evaluate the effect of the guidance range with the same baseline. Table 8 reveals that applying guidance in the early stages is more effective than in later stages, as these initial steps establish the overall spatial structure of the video. However, extending the guidance range too far allows errors in the approximated optimal control to accumulate, ultimately degrading the quality of the final output video.

Assessment policy using LVLM We evaluate the impact of the assessment protocol in LVLM by analyzing the average scores generated with the VideoCrafter2 model. Specifically, we modify the system prompt to instruct LVLM to answer only with ‘yes’ or ‘no’ when assessing text-video alignment. The alignment score is then derived by calculating the percentage of the top 5 logits that correspond to ‘yes’. Table 9 reveals that scoring alignment on a scale from 1 to 9 achieves better performance in terms of text alignment. This is likely because a broader scale allows for more nuanced distinctions in fidelity, enabling LVLM to capture subtle differences in text-video alignment more effectively.

n Avg.

1 0.3722 3 0.3749 5 0.3780

10 0.3705

Table 7. Quantitative results on text alignment by sample size.

Guidance Step Avg.

None 0.3722 t ∈ [T,T − 5] 0.3780 t ∈ [T − 5,T − 10] 0.3769 t ∈ [T,T − 10] 0.3635

Table 8. Quantitative results on text alignment by range of guidance step.

Method Text Alignment General Quailty Avg.

VC2 0.4129 0.7617 0.5873 +GPT0/1 0.4358 0.7550 0.5954 +GPT1-9 0.4425 0.7537 0.5981

Table 9. Average results by assessment policy using LVLM.

### D. Additional Analysis

Method Appearance

Temporal Style

Human Action

Multiple Objects

Spatial Relationship

Overall Consistency

###### Avg.

Style

LaVie 0.2312 0.2502 0.9300 0.2027 0.3496 0.2694 0.3722 + GPT4o 0.2366 0.2508 0.9300 0.2546 0.3531 0.2709 0.3827 + Qwen2.5-VL 3B Image 0.2388 0.2447 0.9700 0.2477 0.3238 0.2647 0.3816 + Qwen2.5-VL 3B Video 0.2325 0.2464 0.9700 0.2431 0.3101 0.2738 0.3793

LTX-Video-2B 0.2189 0.1784 0.5303 0.1994 0.3436 0.1916 0.2770 + GPT4o 0.2202 0.1813 0.5051 0.2335 0.4177 0.1947 0.2921

Table 10. Baseline comparison with open-source Image and Video LVLM and longer video generation model.

##### Aspects Baselines Ours

Overall Quality 2.61 3.19 Temporal Quality 2.65 3.21 Text Alignment 2.60 3.94

Table 11. User study.

Method GPU Memory Computing Time Lavie 4.4 GiB 22.7 s/video

+Ours 7.5 GiB 154.5 s/video

Table 12. Computation.

Open-source LVLM. We leverage an open-source LVLM (QWen2.5-VL 3B) using both stitched image input and direct video input. As shown in Table 10, our framework consistently improves T2V alignment. Interestingly, image input demonstrated stronger performance than direct video input for this specific LVLM. We hypothesize this might be due to our frame stitching method effectively highlighting key temporal information for the LVLM.

Style Semantics Condition Consistency

Avg. Method Appearance Style Temporal Style Human Action Multiple Objects Spatial Relationship Overall Consistency

LaVie [43] 0.2312 0.2502 0.9300 0.2027 0.3496 0.2694 0.3722 + CLIP 0.2370 (+2.5%) 0.2490 (-0.5%) 0.9400 (+1.1%) 0.2607 (+28.6%) 0.3074 (-12.1%) 0.2738 (+1.6%) 0.3780 + ViCLIP 0.2348 (+1.6%) 0.2485 (-0.7%) 0.9600 (+3.2%) 0.2149 (+6.0%) 0.2872 (-17.9%) 0.2752 (+2.1%) 0.3701 + GPT 0.2366 (+2.3%) 0.2508 (+0.2%) 0.9300 (-0.0%) 0.2546 (+25.6%) 0.3531 (+1.0%) 0.2709 (+0.6%) 0.3827

Temporal Consistency Dynamics Frame-wise Quality

Avg. Method Subject Consistency Background Consistency Motion Smoothness Dynamic Degree Aesthetic Quality Imaging Quality

LaVie [43] 0.9450 0.9689 0.9718 0.4799 0.5687 0.6611 0.7659 + CLIP 0.9495 (+0.5%) 0.9712 (+0.2%) 0.9735 (+0.2%) 0.4560 (-5.0%) 0.5727 (0.7%) 0.6637 (+0.4%) 0.7644 + ViCLIP 0.9443 (-0.1%) 0.9694 (+0.0%) 0.9741 (+0.2%) 0.4707 (-1.9%) 0.5746 (1.0%) 0.6487 (-1.9%) 0.7636 + GPT 0.9470 (+0.2%) 0.9693 (+0.0%) 0.9742 (+0.2%) 0.4725 (-1.5%) 0.5726 (+0.7%) 0.6615 (+0.1%) 0.7662

Table 13. Comparison with video-based reward model. Higher numbers indicate better video quality. The numbers in parentheses denote the performance difference from the baselines.

Long Video Generation Model. To address concerns about generalization to longer videos, we applied Free2Guide to a long video generation model (LTX-video 2B), generating 15-second videos. As presented in Table 10, we measure VBench2beta-long metrics and our framework significantly improves performance over the baseline (which used stochastic sampling for fair comparison), demonstrating its effectiveness in longer videos.

User Study. We conducted a user study with 50 participants on Prolific, comparing videos from our method against the baseline (LaVie and VideoCrafter2). Participants rated videos on a 1-5 scale for overall quality, temporal quality, and text alignment. Our method was consistently preferred across all aspects, as shown in Table 11.

#### D.1. Video Reward Guidance

While using a video-based reward model to guide videos is a more natural approach, we claim that video reward models fail to capture the representation needed for guidance because the dataset of video-text pairs is relatively limited compared to images. To support this, we compare the results of using a video-based reward model for guidance with a video-based reward model for text alignment. We adopt ViCLIP [44], a pre-trained video-text representation learning model available at https://huggingface.co/OpenGVLab/ViCLIP, as the video reward model. Using LaVie as the baseline, we compute the reward based on eight video frames, measuring the similarity between the video and text embeddings.

Table 13 shows that the video-based reward model does not significantly outperform the image-based reward model. However, it specifically enhances the Overall Consistency and Dynamic Degree metrics. It is worth noting that the Overall Consistency metric is evaluated using ViCLIP itself, which could introduce a bias favoring the video reward model. In addition, we observe that ViCLIP struggles with spatial information processing compared to CLIP, leading to lower performance on the Multiple Objects and Spatial Relationship metrics. These results highlight the challenges of video reward models to fully capture the relationship between video and text due to the lack of training datasets.

#### D.2. Video Inverse Problems

Our framework can readily extend to inverse problems in the video domain, building on approaches from previous work [17, 51]. In Figure 5, we show a video reconstructed by our method using ×16 average pooling on spatial resolution. For the reward function, we use the L2 distance between the corrupted denoised video and the corrupted video, applying a sampling size of 10 at each step with DDIM over 500 steps, using VideoCrafter2. Our results demonstrate that, compared to unguided sampling, our method generates realistic videos that remain faithful to the input. We leave further extension to video inverse problems as future work.

### E. Additional Visual Results

[Figure 12]

Figure 5. The result of applying our method to the inverse problem. Baseline represents that no guidance is applied during sampling.

[Figure 13]

###### Figure 6. More qualitative comparison of different reward models. The red text highlights the difference between the models.

[Figure 14]

###### Figure 7. More qualitative results of ensembling with LVLMs. The red text highlights the difference between the models.

“The background is changing from blue to pink”

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

Lavie

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

Lavie + GPT4o

“The light bulb is turning off.”

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

Lavie

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

Lavie + GPT4o

“The glass is going from empty to full of water.”

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

VideoCrafter

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

VideoCrafter + GPT4o

Figure 8. More qualitative comparison of T2V-Compbench to analyze video-specific dynamics. The red text highlights the difference between the models.

