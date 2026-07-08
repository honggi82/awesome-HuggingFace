# arXiv:2604.15311v2[cs.CV]3May2026

[Figure 1]

[Figure 2]

## LeapAlign: Post-Training Flow Matching Models at Any Generation Step by Building Two-Step Trajectories

Zhanhao Liang1,2,∗, Tao Yang2,∗,†, Jie Wu2, Chengjian Feng2, Liang Zheng1 1The Australian National University, 2ByteDance Seed ∗Equal contribution, †Project lead

zhanhaoliang@outlook.com, {yangtao.mry, wujie.10, fengchengjian.cv}@bytedance.com, liang.zheng@anu.edu.au

#### Abstract

This paper focuses on the alignment of flow matching models with human preferences. A promising way is fine-tuning by directly backpropagating reward gradients through the differentiable generation process of flow matching. However, backpropagating through long trajectories results in prohibitive memory costs and gradient explosion. Therefore, direct-gradient methods struggle to update early generation steps, which are crucial for determining the global structure of the final image. To address this issue, we introduce LeapAlign, a fine-tuning method that reduces computational cost and enables direct gradient propagation from reward to early generation steps. Specifically, we shorten the long trajectory into only two steps by designing two consecutive leaps, each skipping multiple ODE sampling steps and predicting future latents in a single step. By randomizing the start and end timesteps of the leaps, LeapAlign leads to efficient and stable model updates at any generation step. To better use such shortened trajectories, we assign higher training weights to those that are more consistent with the long generation path. To further enhance gradient stability, we reduce the weights of gradient terms with large magnitude, instead of completely removing them as done in previous works. When fine-tuning the Flux model, LeapAlign consistently outperforms state-of-the-art GRPO-based and direct-gradient methods across various metrics, achieving superior image quality and image–text alignment.

Date: May 5, 2026 Project Page: https://rockeycoss.github.io/leapalign/

#### 1 Introduction

We study how to align flow matching models [6, 18, 19, 27, 30] with human preferences. GRPO-based methods, originally designed for large language model (LLM) post-training, are popular for flow matching [29, 42, 55]. Because the text generation process of LLMs is not differentiable, policy gradient forms the basis of these methods, which inevitably adds a considerable level of stochasticity and variance.

Essentially, what makes flow matching models differ from LLMs is that the sampling process of the former is continuous and differentiable, while the latter has discrete generation. This difference allows reward gradients to flow through the generation trajectory. That is, to increase the reward, the reward gradient can be backpropagated along intermediate image latents and used to update model weights by the chain rule. We refer to methods using this native gradient-based strategy as direct-gradient methods, since they directly

[Figure 3]

FLUX.1-Dev FLUX.1-Dev + DRTune FLUX.1-Dev + LeapAlign

Attribute Binding

ImageReward

[Figure 4]

[Figure 5]

(a) (b) (c)

[Figure 6]

Counting

Position

1.51

66.00

1.36

73.12 72.50

PickScore

55.50

HPSv2.1

30.25

27.50

66.88

1.05 25.8

0.41

45.25

25.1

0.39

19.50

22.8

Single Object

0.31

Overall

99.38

0.71 0.65 0.74

99.38

99.38

10.8 12.0

UnifiedReward

86.62 93.69 96.46

3.57

74.47 76.86

3.67 3.72

HPSv3

12.6

Two Object

Colors

80.59

- Figure 1 Performance overview of LeapAlign. (a) Comparison of reward improvement during fine-tuning on the compositional alignment task. LeapAlign achieves faster and higher reward gains than DRTune [52]. (b) LeapAlign consistently improves Flux across multiple evaluators. (c) LeapAlign shows clear gains on the GenEval benchmark. For clearer visualization of performance gains, we shift the radar chart origin to 60% of the FLUX.1-Dev performance and set the maximum radius to the best performance among the displayed methods.

backpropagate reward gradients through the differentiable generation trajectory [3, 52, 53]. They often allow flow matching post-training to converge faster and train more stably than policy-gradient-based methods.

However, backpropagation through long trajectories poses two significant challenges for direct-gradient methods: 1) prohibitive memory cost caused by the long chains of activations and 2) gradient explosion [3]. To avoid these challenges, existing methods typically update only one timestep close to the final image in each iteration [3, 52]. As a consequence, early steps that largely determine image layout [12, 25] are not updated. While it is possible to enable early-step updates by stopping the gradient at the model input [52], this method discards substantial gradient flow and leads to incomplete optimization. Moreover, while reducing the number of sampling steps may alleviate these problems, it would produce noisy or blurry images, making the rewards predicted by the reward model unreliable.

In this work, we introduce a flow matching post-training method, LeapAlign, which allows reward gradients to backpropagate to early timesteps while retaining useful gradients. LeapAlign performs training on a leap trajectory, a two-step trajectory constructed from a standard full-run trajectory, and can fine-tune any generation step. Specifically, at each iteration, we first sample a full trajectory from noise to image, choose two timesteps k > j, and build a leap trajectory that moves the first step from latent xk to xj and the second step from xj to the final latent x0. We compute the reward on the actual final image but backpropagate gradients only through the leap trajectory. The leap trajectory keeps the memory cost constant and allows us to directly update any generation step, whether early or late, because (k,j) are randomly selected across the full trajectory. Further, to address gradient explosion and stabilize training, we apply gradient discounting: we down-weight the large-magnitude gradient term instead of removing it, thereby preserving the learning signal that DRTune [52] removes. In addition, we add trajectory-similarity weighting to the loss so that leap trajectories closer to the real path receive higher weights. Together, LeapAlign makes early-step fine-tuning practical and stable.

We fine-tune Flux [18] with LeapAlign and show the performance gains in Fig. 1. Moreover, compared with the state-of-the-art GRPO-based methods [22, 55] and direct-gradient methods [3, 52, 53], LeapAlign consistently performs better in image generation, reflected by better scores in HPSv2.1 [51], HPSv3 [32], PickScore [17], UnifiedReward [48], ImageReward [53], and image–text alignment on GenEval [10]. In summary, this paper has the following key points.

- • We propose LeapAlign, which trains on a two-step leap trajectory carved from a full run. This reduces memory and allows for model updates at any step.

- • We further propose two techniques for improvement. We assign leap trajectories higher weights if they are similar to the real path. We also scale down gradient terms that potentially have a large magnitude instead

- of completely removing them to preserve their usefulness.
- • LeapAlign stably fine-tunes Flux and consistently outperforms existing post-training methods in improving image generation quality and image-text alignment.

#### 2 Related Work

The emergence of diffusion [14] and flow matching models [27, 30] has driven major progress in text-to-image generation [6, 18, 19, 34, 38, 41, 50]. Aligning such models with human preferences has become increasingly important. Inspired by RLHF [33], recent studies explore diverse post-training strategies for preference alignment.

Many methods are based on policy gradients [1, 7, 11, 20, 26, 63]. They generally fine-tune diffusion models using PPO [40] or REINFORCE [49]. Another popular line of work is based on direct preference optimization (DPO) [37] for LLM post-training. They include Diffusion-DPO [46], D3PO [56], SPO [25], and others [2, 15, 16, 23, 45, 57, 59–61]. They fine-tune diffusion models using preference pairs or sets. For flow matching models, Adjoint Matching [5] formulates reward fine-tuning as stochastic optimal control, whereas DiffusionNFT [64] and AWM [54] propose forward-process RL methods. DanceGRPO [55] and Flow-GRPO [29] adapt GRPO [42] to flow matching by converting deterministic ODE sampling into an equivalent SDE formulation and applying the GRPO loss across generation steps. MixGRPO [22] and other GRPO variants [24, 47, 66] further improve efficiency and performance.

Unlike the methods above, direct-gradient methods use the differentiability of diffusion and flow matching samplers to propagate reward gradients directly [3, 35, 43, 44, 52, 53, 62]. ReFL [53] randomly selects a timestep near the end of the generation trajectory and uses a one-step leap prediction to estimate the final image xˆ0. The reward is computed on xˆ0, and only the selected step is updated to maximize the reward. DRaFT-LV [3] updates only the last sampling step and reduces gradient variance by repeatedly noising the final image using the forward process and aggregating reward gradients across these noisy variants. DRTune [52] updates early steps by stopping the gradient at the model input, avoiding out-of-memory errors and gradient explosion when propagating through the full trajectory.

Notable differences with our method. Compared with ReFL and DRaFT-LV, which fine-tune only a single late step per trajectory, LeapAlign constructs a leap trajectory (Section 4.2) to propagate gradients to early generation steps, which are important for improving global layout. Moreover, while DRTune supports early-step updates and can fine-tune multiple steps per rollout, it removes the nested gradient (Section 4.3), which is useful for capturing dependencies across timesteps. Our method retains this term by lowering its weight in the full gradient, which is shown to be effective. Table 1 compares key differences among these methods. We also summarize these algorithms in Appendix D.

- Table 1 Comparison of direct-gradient methods. Early Steps: whether early generation steps can be updated. Nested Gradient: whether nested gradients (Eq. 8) that capture interactions across timesteps are preserved. Leap Trajectory: whether the method constructs leap trajectories for backpropogation. Multi-Step: whether multiple steps can be updated per trajectory.

Method Early Steps Nested Gradient Leap Trajectory Multi-Step

ReFL [53] ✗ ✗ ✗ ✗ DRaFT-LV [3] ✗ ✗ ✗ ✗ DRTune [52] ✓ ✗ ✗ ✓ LeapAlign (Ours) ✓ ✓ ✓ ✓

#### 3 Preliminaries

Flow matching models [27, 30] learn a continuous transformation that maps Gaussian noise to images by estimating a velocity field. Let x1 ∼ N(0,I) be a Gaussian noise sample and x0 ∼ pdata be a real image from

the data distribution. A forward noising process interpolates between them:

xt = αtx0 + βtx1, (1) where (αt,βt) is a scheduler [28] controlling the interpolation from data to noise. A neural network vθ is trained to predict the velocity field v = dx

dt by minimizing: Lfm = Et, x

t

0∼pdata, x1∼N(0,I) ∥vθ(xt,t) − v∥22 . (2)

In rectified flow matching [30], the scheduler takes the simple linear form αt = 1 − t, βt = t, making v = x1 − x0.

One-step leap prediction. As derived in Appendix F, a rectified flow matching model can estimate the latent xj at any timestep j from another timestep k by:

xˆj|k = xk − (k − j)vθ(xk,k), (3)

where k,j ∈ [0,1]. xˆj|k could be an approximation of xj. As detailed in Section 4.2, this property allows us to construct two adjacent one-step leaps, each directly connecting two timesteps along the full sampling trajectory and thus making backpropagation easier.

#### 4 Proposed Approach

𝑥  | 

𝑣 𝑥 𝑣 𝑥

### ···

𝑥  ∆  𝑥

𝑥  ∆  𝑥  | 

𝑥

···

···

ODE Sampling Step

𝑥 𝑥

Reward Loss

Input to 𝑣

[Figure 7]

[Figure 8]

One-Step Leap Prediction

Latent Connector

Backpropagation Path

- Figure 2 Overview of LeapAlign. x1, ..., x0 are the latents in the full generation trajectory, where x1 and x0 correspond to noise and the clean image, respectively. Our method builds two leaps: from xk we predict xˆj|k using the velocity predicted at xk, and from xj we predict xˆ0|j using the velocity predicted at xj. Here, all latents and velocity predictions are obtained during online sampling. We also compute the latent connector to connect the real latent and its one-step approximation. The two leaps and the two latent connectors form a two-step leap trajectory1. It is along the leap trajectory instead of the full trajectory that reward gradient can flow efficiently. Further, because k and j are randomly selected, ultimately LeapAlign can update any generation step.

- 4.1 Framework Overview To enable effective fine-tuning of early generation steps with direct-gradient methods, we propose LeapAlign.

- Figure 2 depicts its overall workflow. At each iteration, we first generate an image from Gaussian noise through standard ODE sampling steps. We then randomly select two timesteps (k and j) from this long generation trajectory to construct a shortened trajectory of two one-step leaps for fine-tuning (Section 4.2). To prevent gradient explosion during backpropagation through the leap trajectory, LeapAlign applies a gradient

1The latent connectors, e.g., from xˆj|k to xj, are not counted as a step because they do not involve prediction using the flow matching model.

discounting mechanism (Section 4.3) that scales down the gradient term with a large norm instead of removing it. The fine-tuning objective (Section 4.4) aims to maximize the expected reward of generated images. Finally, a trajectory-similarity weighting scheme (Section 4.5) amplifies learning signals from leap trajectories that better match the true generation process.

###### 4.2 Leap Trajectory Construction

As shown in Fig. 2, we shorten the long trajectory into only two steps by constructing two one-step leaps. Our design ensures that the shortened trajectory, named leap trajectory, preserves the step dynamics of the original trajectory while keeping memory cost constant and controlling gradient growth. Fine-tuning on leap trajectories allows for stable gradient backpropagation to any generation step.

Formally, we randomly select two timesteps k and j from the generation trajectory, where k > j. Using the one-step leap prediction property of rectified flow models (Eq. 3), we estimate the latent states at timesteps j and 0 as:

xˆj|k = xk − (k − j)vθ(xk), (4) xˆ0|j = xj − jvθ(xj), (5)

where xk and xj denote the latent states at timesteps k and j along the long generation trajectory, and vθ is the flow matching model being fine-tuned. For simplicity, we let vθ(xt) denote the velocity prediction after classifier-free guidance [13], and omit the explicit dependence on the text and timestep conditions.

To align the predicted states xˆ with the actual ones x while preserving differentiability, we introduce the latent connector:

xj = xˆj|k + stop_gradient(xj − xˆj|k), (6) x0 = xˆ0|j + stop_gradient(x0 − xˆ0|j). (7)

This process constructs a leap trajectory with two steps: xk → (ˆxj|k xj) → (ˆx0|j x0),

where solid arrows represent the one-step leap prediction performed by the flow matching model, while dashed arrows denote latent connectors that align the one-step predicted and real latents. Because the leap trajectory only has two steps, we achieve efficient gradient backpropagation to early steps with constant memory cost. Moreover, because k and j are randomly selected, we can fine-tune any step.

###### 4.3 Gradient Discounting

While the leap trajectory controls gradient growth, backpropagating through two flow matching steps still produces larger gradients than one-step direct-gradient methods [3, 53]. Appendix G shows the gradient propagated from the image x0 w.r.t. parameters θ can be written as:

∂x0 ∂θ

∂vθ(xj) ∂θ − (k − j)

∂vθ(xk) ∂θ

=−j

single-step gradients at k and j

∂vθ(xj) ∂xj

∂vθ(xk) ∂θ

+ j(k − j)

.

nested gradient

(8)

We refer to the first two terms as single-step gradients, since each arises from the gradient of a single one-step leap prediction (Eq. 4 and Eq. 5). The last term is the nested gradient, which arises when gradients are propagated through multiple steps. The nested gradient is useful for capturing interactions across different generation steps.

DRTune [52] mitigates gradient explosion by stopping the gradient of the model input, which effectively means removing the nested gradient term j(k − j)∂v∂xθ(xj)

∂vθ(xk)

∂θ in Eq. 8. Its drawback is that it loses useful signals

j

in the nested gradient. Instead of removing it, we propose a gradient discounting mechanism that reduces its magnitude so as to preserve the full gradient structure.

Specifically, using a discounting factor α ∈ [0,1], we modify Eq. 5 as:

xˆ0|j = xj − jvθ(αxj + (1 − α)stop_gradient(xj)). (9) This adjustment scales the nested gradient by α, producing:

∂x0 ∂θ

∂vθ(xj) ∂θ − (k − j)

∂vθ(xk) ∂θ

= − j

∂vθ(xj) ∂xj

∂vθ(xk) ∂θ

+ αj(k − j)

.

(10)

By adjusting α, we can moderate the gradient magnitude without discarding any component of the gradient flow. This, together with the leap trajectory design, stabilizes optimization while retaining full learning signals.

###### 4.4 Fine-Tuning Objective

The aim of fine-tuning is to maximize the reward predicted for the generated image. However, directly maximizing reward values often leads to reward hacking, where the model exploits the reward function rather than genuinely improving alignment quality. This paper therefore uses a simple hinge-style objective following Xu et al. [53]:

Lraw = max(0,λ − r(x0)), (11)

where r(·) is the reward model, and λ is a threshold that controls the strength of reward maximization. This loss encourages the model to increase rewards beyond the threshold while preventing unstable optimization toward excessively high or misleading reward values.

Unlike existing direct-gradient methods (e.g., ReFL, DRTune) that effectively estimate rewards from one-step leap predictions (e.g., xˆ0|j), we evaluate the reward using the generated image x0. While xˆ0|j is only an estimation of the final output and may contain noise and artifacts, x0 directly reflects the output quality of the full generation trajectory. Using x0 therefore allows the reward model to make more faithful assessments of visual and semantic quality, providing more reliable supervision signals for fine-tuning.

###### 4.5 Trajectory-Similarity Weighting

Since gradients are backpropagated through the leap trajectory to fine-tune the flow matching model, leap trajectories that deviate significantly from the original generation trajectory can yield misleading gradient signals. We therefore further introduce a trajectory-similarity weighting that emphasizes leap trajectories that are more consistent with the original trajectory.

We measure similarity by the average absolute difference between predicted states xˆ and actual states x at the two connection points:

dj = mean |xj − xˆj|k| ,d0 = mean |x0 − xˆ0|j| .

To avoid overemphasizing near-identical pairs, we clamp each distance with a minimum value τ and define the weighting factor as:

1 max(dj,τ) + max(d0,τ)

. (12) The final objective is formulated as:

wsim =

L = stop_gradient(wsim)Lraw. (13)

This weighting assigns higher importance to leap trajectories that better match the original generation dynamics, enabling more faithful and effective supervision.

#### 5 Discussions

LeapAlign reflects the key designs from DRTune and ReFL. All three methods directly backpropagate reward gradients. ReFL [53] uses one-step leap prediction to estimate xˆ0 from an intermediate latent, enabling gradient updates at that single timestep. Our method similarly employs one-step leap prediction, but extends it by constructing a leap trajectory from xk to xj and then to x0. DRTune [52] and LeapAlign both propagate gradient to early steps and try to address the nested gradients (but in different ways).

What reward models can be used with LeapAlign, any limitations? LeapAlign can accommodate any differentiable reward model. In our experiments (Section 6.2), we show that both CLIP-based [36] rewards (HPSv2.1 [51], PickScore [17]) and vision-language-model-based rewards (HPSv3 [32]) lead to effective finetuning results. Extending LeapAlign to non-differentiable rewards, perhaps via differentiable value models [4], is future work.

Is LeapAlign applicable to one-step or few-step image generation models? It is less important. While reward gradients can be propagated directly in one-step and few-step methods [8, 9, 39, 58, 65] because of the very short trajectories, these models fall short of multi-step methods in image quality and alignment. As such, it is more important to design fine-tuning methods for multi-step models.

#### 6 Experiments

###### 6.1 Experimental Setup

Training prompt datasets. We conduct experiments on two alignment tasks: general preference alignment and compositional alignment. For general preference alignment, following prior works [22, 55], we train on a set of 50,000 prompts sampled from the HPDv2 dataset [51]. We also use prompts from MJHQ-30k [21] for training. For compositional alignment, we use the 50,000-prompt dataset [29] generated with the official GenEval scripts [10]. This dataset spans six GenEval task categories, with ratio 7:5:3:1:1:0 for Position, Counting, Attribute Binding, Colors, Two Objects, and Single Object, respectively.

Test prompt datasets, evaluation protocols, and metrics. For general preference alignment, we follow the evaluation setup in MixGRPO [22] and generate images using the 400-prompt test set of the HPDv2 dataset. To reduce variance in evaluation, we generate four images per prompt, resulting in a total of 1,600 images. We assess the generated images using six automatic evaluators. Specifically, we employ HPSv2.1 [51], HPSv3 [32], PickScore [17], and ImageReward [53] to evaluate the degree to which generated images align with human preferences. In addition, we use UnifiedReward-Alignment and UnifiedReward-IQ [48] to assess image–text alignment and overall image quality, respectively. We further construct a 500-prompt test split by randomly sampling from MJHQ-30k [21] to evaluate models fine-tuned on the remaining prompts of the same dataset.

For compositional alignment, we evaluate on the GenEval benchmark [10], which consists of six compositional generation tasks: single-object generation, two-object generation, counting, colors, spatial position, and attribute binding. Following the official GenEval evaluation protocol, during testing we generate four images per prompt using its 553-prompt test set and employ the provided rule-based evaluators to automatically determine the correctness of each generated image.

Implementation details. We fine-tune FLUX.1-dev [18], a state-of-the-art open-source rectified flow matching model capable of generating high-quality images. During fine-tuning, by default we use HPSv2.1 [51] as the reward model and set the loss threshold λ = 0.55. We optimize all parameters of the Flux DiT using AdamW [31] with a learning rate of 1e−5, batch size 64, weight decay 1e−4, EMA decay rate 0.995, β1 = 0.9, and β2 = 0.999. The model is trained for 300 iterations on 16 GPUs. For online rollouts during training, we generate images at a resolution of 720 × 720 using 25 steps and a classifier-free guidance scale of 3.5 [13]. For evaluation, we sample images with the same resolution, 50 steps, and the same guidance scale. For our method LeapAlign, we set τ = 0.1 empirically. Since DRaFT-LV [3] and DRTune [52] do not have official implementations, we reproduce them based on the pseudo-code provided in their papers. We also adapt the official implementation of ReFL [53] to Flux for comparison. For additional implementation and training details, please refer to Appendix E.

- Table 2 Comparing different post-training methods. The base model is Flux. For the general preference alignment experiments, all post-training methods except MixGRPO use HPSv2.1 as the reward model, so the metric based on HPSv2.1 is marked as ‘in-domain’. † Fine-tuned using HPSv2.1, PickScore, and ImageReward as reward models for general preference alignment experiments. ∗ Implemented by us due to the absence of an official implementation. ‡ Adapted to Flux by us from the official implementation. Best scores are in bold, and second-best scores are underlined. PS: PickScore; UR: UnifiedReward; IR: ImageReward; Obj.: Object; Pos: Position; AttrB: Attribute Binding.

In-Domain Out-of-Domain GenEval Benchmark

Method HPSv2.1 ↑ HPSv3 ↑ PS ↑ UR-Align ↑ UR-IQ ↑ IR ↑ Overall Single Obj. Two Obj. Count Color Pos AttrB Pretrained Model Flux 0.3078 13.5020 22.7902 3.4514 3.5708 1.0455 0.6535 99.38 86.62 66.88 74.47 19.50 45.25 Policy-Gradient-Based Methods

DanceGRPO 0.3451 14.8336 23.1186 3.4660 3.6199 1.2347 0.6775 99.38 90.15 69.38 76.33 22.25 49.00 MixGRPO† 0.3692 14.7530 23.5184 3.4393 3.6241 1.6155 0.7232 99.69 93.69 80.00 80.05 24.25 56.25

Direct-Gradient Methods

ReFL‡ 0.3852 15.5127 23.6299 3.4786 3.6870 1.3468 0.7011 99.38 92.68 69.06 75.80 26.75 57.00 DRaFT-LV∗ 0.3859 15.3699 23.6437 3.4868 3.6887 1.3384 0.7024 99.69 92.42 74.06 75.53 24.00 55.75 DRTune∗ 0.3882 15.5606 23.5185 3.4793 3.6679 1.3562 0.7101 99.38 93.69 73.12 76.86 27.50 55.50 LeapAlign 0.4092 15.7678 23.7137 3.4984 3.7244 1.5104 0.7420 99.38 96.46 72.50 80.59 30.25 66.00

- Table 3 Comparison of post-training methods using various rewards and prompt sets. We fine-tune Flux with PickScore on HPDv2 and with HPSv3 on MJHQ-30k, respectively.

Method HPSv2.1 ↑ PickScore ↑ HPSv3 ↑ Pretrained Model

Flux 0.3078 22.7902 10.7624 Direct-Gradient Methods

ReFL‡ 0.3852 25.2373 11.7642 DRaFT-LV∗ 0.3859 24.9596 11.2701 DRTune∗ 0.3882 25.1021 12.0023

LeapAlign 0.4092 25.7589 12.5855

###### 6.2 Main Results

Comparing general preference alignment with state-of-the-art post-training methods. We use HPSv2.1 [51] as the reward model and compare LeapAlign with policy-gradient–based methods including DanceGRPO [55] and MixGRPO [22] and direct-gradient methods including ReFL [53], DRaFT-LV [3], and DRTune [52]. For DanceGRPO and MixGRPO, we use their official Flux checkpoints on Hugging Face2. They are trained on the same prompt set for the same number of iterations as ours. Note that DanceGRPO is trained with HPSv2.1 as the reward, while MixGRPO jointly optimizes HPSv2.1, PickScore, and ImageReward. We summarize the results in Table 2.

We have the following observations. First, LeapAlign demonstrates strong overall performance, achieving the highest average scores across both in-domain metric HPSv2.1 and out-of-domain metrics HPSv3, PickScore, UnifiedReward-Alignment, and UnifiedReward-IQ. Second, while MixGRPO is jointly fine-tuned with three reward models (HPSv2.1, PickScore, and ImageReward), LeapAlign, trained only with HPSv2.1, yields higher average scores on HPSv2.1 and PickScore and remains competitive on ImageReward. In summary, compared with the state-of-the-art, LeapAlign produces consistent in-domain and out-of-domain reward gains in human preference alignment, image–text consistency, and overall image quality.

2DanceGRPO: https://huggingface.co/xzyhku/flux_hpsv2.1_dancegrpo MixGRPO: https://huggingface.co/tulvgengenr/MixGRPO

LeapAlign Flux ReFL DRaFT-LV DRTune [0, 1/2]

LeapAlign [0, 1]

LeapAlign Flux ReFL DRaFT-LV DRTune [0, 1/2]

LeapAlign [0, 1]

[Figure 9]

[Figure 10]

a photo of a suitcase right of a boat

a photo of a red giraffe

[Figure 11]

[Figure 12]

a photo of a yellow bicycle and a red motorcycle a photo of a bicycle above a parking meter

a photo of two carrots

[Figure 13]

[Figure 14]

a photo of a green skis and a brown airplane

- Figure 3 Qualitative comparison on GenEval benchmark. We compare direct-gradient post-training methods and the base model Flux. These examples show our method can generate high-quality images aligned with text prompts. [ ·, · ] indicates the timestep range used for training.

Effectiveness of LeapAlign with different reward models, prompt sets, and flow matching models. To further validate the generality of LeapAlign across different reward models and prompt sets, we additionally fine-tune Flux using PickScore on HPDv2 and HPSv3 on MJHQ-30k, respectively. We then evaluate the fine-tuned models on the HPDv2 test set and on a non-overlapping, randomly sampled test split of MJHQ-30k. As shown in Table 3, LeapAlign again achieves the best performance across both settings, confirming its robustness. Additional results on SD3.5-M [6] are provided in Appendix B, further supporting the generality of LeapAlign across flow matching models.

Comparing compositional alignment with state-of-the-art post-training methods. To verify that LeapAlign effectively fine-tunes early generation steps which largely determine the image layout [12], we conduct evaluation on the GenEval benchmark [10] consisting of diverse compositional generation tasks. We use HPSv2.1 [51] as the reward model and adopt the GenEval training prompts from Liu et al. [29] to fine-tune Flux. For DanceGRPO and MixGRPO, we run experiments using their official codebases and recommended hyperparameter settings, with the same HPSv2.1 reward, GenEval training prompt set, and number of training iterations as ours. Results are reported in Table 2. The GenEval score improvement during fine-tuning for direct-gradient methods is visualized in Appendix A.

We observe that LeapAlign outperforms competitive post-training methods by a clear margin, e.g., overall score 0.7420, compared with 0.7232 for MixGRPO, the best policy-gradient-based baseline, and 0.7101 for DRTune, the strongest direct-gradient baseline. The GenEval performance is particularly strong under the ‘two objects’, ‘colors’, ‘position’, and ‘attribute binding’ categories. In fact, MixGRPO can use policy gradients to update early steps, and DRTune is also capable of fine-tuning early steps but discards critical gradients. These results indicate the benefit of fine-tuning early steps and the effectiveness of LeapAlign.

Fine-tuning reward curves. We plot the average HPSv2.1 reward curves during fine-tuning in Fig. 1. Rewards are computed from generated images x0 obtained from rollout trajectories during fine-tuning. Compared with DRTune, LeapAlign exhibits much stronger reward growth.

Qualitative results are shown in Fig. 3. For methods that can only fine-tune late generation steps, such as ReFL and DRaFT-LV, the generated layouts remain similar to those of the pretrained model. In comparison, LeapAlign substantially modifies the global structure, producing images with compositions more faithful to the text prompts.

###### 6.3 Further Analysis

If not specified, we use HPSv2.1 as the reward model when fine-tuning Flux, and adopt the prompt sets from the training and test splits of the HPDv2 dataset [51] for training and evaluation, respectively.

Effectiveness of gradient discounting. The gradient discounting factor α controls the scale of the nested gradient term (Eq. 10). To assess its effect, we compare LeapAlign with two variants: one that removes the nested gradient term entirely (α = 0) and another that applies no discounting (α = 1). As shown in Fig. 4a,

x0 x0 j d0 x0 j

(a) Effectiveness of gradient discounting. We vary the value of α (Eq. 10) and evaluate LeapAlign. Setting α = 0.3 yields the best performance.

(b) Comparing using one, two, and three steps in leap trajectories. Using two steps results in the best trade-off between performance and memory cost.

###### (c) Comparison of different inputs of the reward model. ‘xˆ0|j + d0’ computes trajectory-similarity weighting with dj and d0. We can see that using x0 is superior.

(k, j)

[0, 1/2] [0, 1]

dj d0 dj d0

k j

(d) Comparing trajectory similarity weighting methods. ‘dj and d0’: our method (Eq. 12). ‘dj’ and ‘d0’ only measure trajectory differences at xj and x0, respectively. ‘w/o’: this mechanism is not applied. Our method has the best result.

(e) Impact of training timestep range. We construct leap trajectories by randomly selecting timesteps from different ranges, where 1 is the earliest timestep. Random selection over the full timestep range [0, 1] has better performance.

(f) Comparing strategies for selecting k and j. ‘Fixed (k, j) Distance’: fixing the distance between k and j to 1/2. ‘Random’ means k and j are randomly selected (Section 4.2). Random selection performs better and is easier to implement.

- Figure 4 Further analysis of design components in LeapAlign, including gradient discounting, the number of steps in leap trajectories, the input of the reward model, the trajectory-similarity weighting scheme, the training timestep range, and the selection strategy of k and j.

setting α = 0.3 yields the best performance. Removing the nested gradient term (α = 0) leads to incomplete optimization and lower scores, while omitting discounting (α = 1) retains large gradients, making optimization difficult. See Appendix C for additional analysis of the nested gradient. Notably, even without the nested

- gradient (α = 0), LeapAlign still outperforms DRTune on HPSv2.1 (0.4064 vs. 0.3882 in Table 2), suggesting that its gains come not only from the nested gradient but also from the leap trajectory design.

Effectiveness of trajectory-similarity weighting. To evaluate the effectiveness of trajectory-similarity weighting, we compare our method (Eq. 12) with three variants. The first and second variants measure similarity only at xj and x0, respectively, while the last variant removes the weighting mechanism completely. As shown in Fig. 4d, variants that consider similarity at only a single step already improve the average HPSv2.1 score over the baseline without weighting. Our design, which incorporates similarity at both xj and x0, further enhances the average score.

Comparing leap trajectories with one, two, or three steps. We build one, two, or three one-step leaps and compare their fine-tuning performance. Results are shown in Fig. 4b. Our observation is that two-step leap trajectories provide the best trade-off between performance and memory usage. Using three steps increases memory consumption but yields no better result than our two step version. While using one step is not as good as two steps, its generation quality is still better than competing methods like DRTune and ReFL (Table 2). This demonstrates that the design of LeapAlign, including the leap trajectory (Section 4.2), reward evaluation on x0 (Section 4.4), and trajectory-similarity weighting (Section 4.5), boosts the performance of even the one-step variant.

Comparing range of training timesteps. Timesteps k and j define the position of the two leaps in the full trajectory for training (Section 4.2). Our default method is to randomly select k and j within the full timestep

range [0,1], where 1 is the earliest timestep with Gaussian noise as input. In Fig. 4e, we compare this range with [0,1/2] on the GenEval benchmark. We find that [0,1] is superior, highlighting that fine-tuning early generation steps is important for accurate layout and composition. As shown in Fig. 3, the [0,1] variant also yields qualitatively better results with stronger image-text alignment.

Comparing strategies for selecting k and j. This paper uses random selection within range [0,1]. We implement a variant: the distance between k and j is fixed to 12. The two methods are compared in Fig. 4f, where we observe that random selection is slightly better. For implementation simplicity, we use random selection in LeapAlign.

Comparing inputs of the reward model. We use the generated image x0 as the input to the reward model. To examine the impact of this choice, we implement two variants: one that uses xˆ0|j as input and measures trajectory-similarity only at xj, and another that also uses xˆ0|j but applies trajectory-similarity weighting considering similarities at both xj and x0. As shown in Fig. 4c, using x0 as input yields superior results, benefiting from more accurate reward evaluation and trajectory-similarity weighting.

#### 7 Conclusion

This paper introduces LeapAlign, a new post-training method that constructs two-step leap trajectories for efficient and stable reward gradient backpropagation. We find it useful to down-scale the large-magnitude gradient term and up-weight leap trajectories that are more similar to the original trajectories. Our method successfully addresses the challenge of propagating reward gradients to early generation steps without incurring excessive memory cost or sacrificing useful gradient terms. This is reflected by consistent improvements over existing post-training methods across a wide range of metrics, including general image preference and image-text alignment. In the future, we will implement and improve LeapAlign in video generation.

#### 8 Acknowledgement

We sincerely thank Jie Liu, Zeyue Xue, Kaiwen Zheng, and Xingjian Leng for insightful discussions. This work was partially supported by ARC Future Fellowship FT240100820.

#### References

- [1] Kevin Black, Michael Janner, Yilun Du, Ilya Kostrikov, and Sergey Levine. Training diffusion models with reinforcement learning. arXiv preprint arXiv:2305.13301, 2023.

- [2] Renjie Chen, Wenfeng Lin, Yichen Zhang, Jiangchuan Wei, Boyuan Liu, Chao Feng, Jiao Ran, and Mingyu Guo. Towards self-improvement of diffusion models via group preference optimization. arXiv preprint arXiv:2505.11070, 2025.

- [3] Kevin Clark, Paul Vicol, Kevin Swersky, and David J Fleet. Directly fine-tuning diffusion models on differentiable rewards. arXiv preprint arXiv:2309.17400, 2023.

- [4] Fengyuan Dai, Zifeng Zhuang, Yufei Huang, Siteng Huang, Bangyan Liao, Donglin Wang, and Fajie Yuan. Vard: Efficient and dense fine-tuning for diffusion models with value-based rl. arXiv preprint arXiv:2505.15791, 2025.

- [5] Carles Domingo-Enrich, Michal Drozdzal, Brian Karrer, and Ricky TQ Chen. Adjoint matching: Fine-tuning flow and diffusion generative models with memoryless stochastic optimal control. arXiv preprint arXiv:2409.08861, 2024.

- [6] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning, 2024.

- [7] Ying Fan, Olivia Watkins, Yuqing Du, Hao Liu, Moonkyung Ryu, Craig Boutilier, Pieter Abbeel, Mohammad Ghavamzadeh, Kangwook Lee, and Kimin Lee. Reinforcement learning for fine-tuning text-to-image diffusion models. Advances in Neural Information Processing Systems, 36, 2024.

- [8] Zhengyang Geng, Ashwini Pokle, and J Zico Kolter. One-step diffusion distillation via deep equilibrium models. Advances in Neural Information Processing Systems, 36:41914–41931, 2023.

- [9] Zhengyang Geng, Mingyang Deng, Xingjian Bai, J Zico Kolter, and Kaiming He. Mean flows for one-step generative modeling. arXiv preprint arXiv:2505.13447, 2025.

- [10] Dhruba Ghosh, Hannaneh Hajishirzi, and Ludwig Schmidt. Geneval: An object-focused framework for evaluating text-to-image alignment. Advances in Neural Information Processing Systems, 36:52132–52152, 2023.

- [11] Shashank Gupta, Chaitanya Ahuja, Tsung-Yu Lin, Sreya Dutta Roy, Harrie Oosterhuis, Maarten de Rijke, and Satya Narayan Shukla. A simple and effective reinforcement learning method for text-to-image diffusion fine-tuning. arXiv preprint arXiv:2503.00897, 2025.

- [12] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Prompt-to-prompt image editing with cross attention control. arXiv preprint arXiv:2208.01626, 2022.

- [13] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022.

- [14] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

- [15] Jiwoo Hong, Sayak Paul, Noah Lee, Kashif Rasul, James Thorne, and Jongheon Jeong. Margin-aware preference optimization for aligning diffusion models without reference. In First Workshop on Scalable Optimization for Efficient and Adaptive Foundation Models, 2024.

- [16] Shyamgopal Karthik, Huseyin Coskun, Zeynep Akata, Sergey Tulyakov, Jian Ren, and Anil Kag. Scalable ranked preference optimization for text-to-image generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 18399–18410, 2025.

- [17] Yuval Kirstain, Adam Polyak, Uriel Singer, Shahbuland Matiana, Joe Penna, and Omer Levy. Pick-a-pic: An open dataset of user preferences for text-to-image generation. Advances in neural information processing systems, 36:36652–36663, 2023.

- [18] Black Forest Labs. Flux. https://github.com/black-forest-labs/flux, 2024.
- [19] Black Forest Labs, Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, Sumith Kulal, Kyle Lacey, Yam Levi, Cheng Li, Dominik Lorenz, Jonas Müller, Dustin Podell, Robin Rombach, Harry Saini, Axel Sauer, and Luke Smith. Flux.1 kontext: Flow matching for in-context image generation and editing in latent space, 2025. URL https: //arxiv.org/abs/2506.15742.

- [20] Seung Hyun Lee, Yinxiao Li, Junjie Ke, Innfarn Yoo, Han Zhang, Jiahui Yu, Qifei Wang, Fei Deng, Glenn Entis, Junfeng He, et al. Parrot: Pareto-optimal multi-reward reinforcement learning framework for text-to-image generation. In European Conference on Computer Vision, pages 462–478. Springer, 2024.

- [21] Daiqing Li, Aleks Kamko, Ehsan Akhgari, Ali Sabet, Linmiao Xu, and Suhail Doshi. Playground v2.5: Three insights towards enhancing aesthetic quality in text-to-image generation, 2024.
- [22] Junzhe Li, Yutao Cui, Tao Huang, Yinping Ma, Chun Fan, Miles Yang, and Zhao Zhong. Mixgrpo: Unlocking flow-based grpo efficiency with mixed ode-sde. arXiv preprint arXiv:2507.21802, 2025.

- [23] Shufan Li, Konstantinos Kallidromitis, Akash Gokul, Yusuke Kato, and Kazuki Kozuka. Aligning diffusion models by optimizing human utility. Advances in Neural Information Processing Systems, 37:24897–24925, 2024.

- [24] Yuming Li, Yikai Wang, Yuying Zhu, Zhongyu Zhao, Ming Lu, Qi She, and Shanghang Zhang. Branchgrpo: Stable and efficient grpo with structured branching in diffusion models. arXiv preprint arXiv:2509.06040, 2025.

- [25] Zhanhao Liang, Yuhui Yuan, Shuyang Gu, Bohan Chen, Tiankai Hang, Mingxi Cheng, Ji Li, and Liang Zheng. Aesthetic post-training diffusion models from generic preferences with step-by-step preference optimization. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 13199–13208, 2025.

- [26] Xinyao Liao, Wei Wei, Xiaoye Qu, and Yu Cheng. Step-level reward for free in rl-based t2i diffusion model fine-tuning. arXiv preprint arXiv:2505.19196, 2025.

- [27] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.

- [28] Yaron Lipman, Marton Havasi, Peter Holderrieth, Neta Shaul, Matt Le, Brian Karrer, Ricky TQ Chen, David Lopez-Paz, Heli Ben-Hamu, and Itai Gat. Flow matching guide and code. arXiv preprint arXiv:2412.06264, 2024.

- [29] Jie Liu, Gongye Liu, Jiajun Liang, Yangguang Li, Jiaheng Liu, Xintao Wang, Pengfei Wan, Di Zhang, and Wanli Ouyang. Flow-grpo: Training flow matching models via online rl. arXiv preprint arXiv:2505.05470, 2025.

- [30] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022.

- [31] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.

- [32] Yuhang Ma, Xiaoshi Wu, Keqiang Sun, and Hongsheng Li. Hpsv3: Towards wide-spectrum human preference score. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 15086–15095, 2025.

- [33] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744, 2022.

- [34] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023.

- [35] Mihir Prabhudesai, Anirudh Goyal, Deepak Pathak, and Katerina Fragkiadaki. Aligning text-to-image diffusion models with reward backpropagation, 2023.
- [36] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021.

- [37] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in neural information processing systems, 36:53728–53741, 2023.

- [38] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.

- [39] Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. arXiv preprint arXiv:2202.00512, 2022.

- [40] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

- [41] Team Seedream, Yunpeng Chen, Yu Gao, Lixue Gong, Meng Guo, Qiushan Guo, Zhiyao Guo, Xiaoxia Hou, Weilin Huang, Yixuan Huang, et al. Seedream 4.0: Toward next-generation multimodal image generation. arXiv preprint arXiv:2509.20427, 2025.

- [42] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

- [43] Xiangwei Shen, Zhimin Li, Zhantao Yang, Shiyi Zhang, Yingfang Zhang, Donghao Li, Chunyu Wang, Qinglin Lu, and Yansong Tang. Directly aligning the full diffusion trajectory with fine-grained human preference. arXiv preprint arXiv:2509.06942, 2025.

- [44] Dmitrii Sorokin, Maksim Nakhodnov, Andrey Kuznetsov, and Aibek Alanov. Imagerefl: Balancing quality and diversity in human-aligned diffusion models. arXiv preprint arXiv:2505.22569, 2025.

- [45] Dipesh Tamboli, Souradip Chakraborty, Aditya Malusare, Biplab Banerjee, Amrit Singh Bedi, and Vaneet Aggarwal. Balanceddpo: Adaptive multi-metric alignment. arXiv preprint arXiv:2503.12575, 2025.

- [46] Bram Wallace, Meihua Dang, Rafael Rafailov, Linqi Zhou, Aaron Lou, Senthil Purushwalkam, Stefano Ermon, Caiming Xiong, Shafiq Joty, and Nikhil Naik. Diffusion model alignment using direct preference optimization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8228–8238, 2024.

- [47] Yibin Wang, Zhimin Li, Yuhang Zang, Yujie Zhou, Jiazi Bu, Chunyu Wang, Qinglin Lu, Cheng Jin, and Jiaqi Wang. Pref-grpo: Pairwise preference reward-based grpo for stable text-to-image reinforcement learning. arXiv preprint arXiv:2508.20751, 2025.

- [48] Yibin Wang, Yuhang Zang, Hao Li, Cheng Jin, and Jiaqi Wang. Unified reward model for multimodal understanding and generation. arXiv preprint arXiv:2503.05236, 2025.

- [49] Ronald J Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine learning, 8(3):229–256, 1992.

- [50] Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng-ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, et al. Qwen-image technical report. arXiv preprint arXiv:2508.02324, 2025.

- [51] Xiaoshi Wu, Yiming Hao, Keqiang Sun, Yixiong Chen, Feng Zhu, Rui Zhao, and Hongsheng Li. Human preference score v2: A solid benchmark for evaluating human preferences of text-to-image synthesis. arXiv preprint arXiv:2306.09341, 2023.

- [52] Xiaoshi Wu, Yiming Hao, Manyuan Zhang, Keqiang Sun, Zhaoyang Huang, Guanglu Song, Yu Liu, and Hongsheng Li. Deep reward supervisions for tuning text-to-image diffusion models. In European Conference on Computer Vision, pages 108–124. Springer, 2024.

- [53] Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong. Imagereward: Learning and evaluating human preferences for text-to-image generation. Advances in Neural Information Processing Systems, 36:15903–15935, 2023.

- [54] Shuchen Xue, Chongjian Ge, Shilong Zhang, Yichen Li, and Zhi-Ming Ma. Advantage weighted matching: Aligning rl with pretraining in diffusion models. arXiv preprint arXiv:2509.25050, 2025.

- [55] Zeyue Xue, Jie Wu, Yu Gao, Fangyuan Kong, Lingting Zhu, Mengzhao Chen, Zhiheng Liu, Wei Liu, Qiushan Guo, Weilin Huang, et al. Dancegrpo: Unleashing grpo on visual generation. arXiv preprint arXiv:2505.07818, 2025.

- [56] Kai Yang, Jian Tao, Jiafei Lyu, Chunjiang Ge, Jiaxin Chen, Weihan Shen, Xiaolong Zhu, and Xiu Li. Using human feedback to fine-tune diffusion models without any reward model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8941–8951, 2024.

- [57] Shentao Yang, Tianqi Chen, and Mingyuan Zhou. A dense reward view on aligning text-to-image diffusion with preference. arXiv preprint arXiv:2402.08265, 2024.

- [58] Tianwei Yin, Michaël Gharbi, Richard Zhang, Eli Shechtman, Fredo Durand, William T Freeman, and Taesung Park. One-step diffusion with distribution matching distillation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6613–6623, 2024.

- [59] Huizhuo Yuan, Zixiang Chen, Kaixuan Ji, and Quanquan Gu. Self-play fine-tuning of diffusion models for text-to-image generation. Advances in Neural Information Processing Systems, 37:73366–73398, 2024.

- [60] Daoan Zhang, Guangchen Lan, Dong-Jun Han, Wenlin Yao, Xiaoman Pan, Hongming Zhang, Mingxiao Li, Pengcheng Chen, Yu Dong, Christopher Brinton, et al. Seppo: Semi-policy preference optimization for diffusion alignment. arXiv preprint arXiv:2410.05255, 2024.

- [61] Tao Zhang, Cheng Da, Kun Ding, Huan Yang, Kun Jin, Yan Li, Tingting Gao, Di Zhang, Shiming Xiang, and Chunhong Pan. Diffusion model as a noise-aware latent reward model for step-level preference optimization. arXiv preprint arXiv:2502.01051, 2025.

- [62] Xinchen Zhang, Ling Yang, Guohao Li, Yaqi Cai, Jiake Xie, Yong Tang, Yujiu Yang, Mengdi Wang, and Bin Cui. Itercomp: Iterative composition-aware feedback learning from model gallery for text-to-image generation. arXiv preprint arXiv:2410.07171, 2024.

- [63] Yinan Zhang, Eric Tzeng, Yilun Du, and Dmitry Kislyuk. Large-scale reinforcement learning for diffusion models. In European Conference on Computer Vision, pages 1–17. Springer, 2024.

- [64] Kaiwen Zheng, Huayu Chen, Haotian Ye, Haoxiang Wang, Qinsheng Zhang, Kai Jiang, Hang Su, Stefano Ermon, Jun Zhu, and Ming-Yu Liu. Diffusionnft: Online diffusion reinforcement with forward process. arXiv preprint arXiv:2509.16117, 2025.

- [65] Mingyuan Zhou, Huangjie Zheng, Zhendong Wang, Mingzhang Yin, and Hai Huang. Score identity distillation: Exponentially fast distillation of pretrained diffusion models for one-step generation. In Forty-first International Conference on Machine Learning, 2024.

- [66] Yujie Zhou, Pengyang Ling, Jiazi Bu, Yibin Wang, Yuhang Zang, Jiaqi Wang, Li Niu, and Guangtao Zhai. Fine-grained grpo for precise preference alignment in flow models. arXiv preprint arXiv:2510.01982, 2025.

## Appendix

#### A Visualization of GenEval Score Improvement During Fine-Tuning for DirectGradient Methods

Figure 5 presents the GenEval score improvement curve evaluated during fine-tuning. LeapAlign exhibits both a more rapid increase and a higher final GenEval score compared to DRTune [52], DRaFT-LV [3], and ReFL [53]. Methods that update the early generation steps, such as DRTune, achieve stronger improvements than those that do not, underscoring the significance of early-step fine-tuning for the compositional alignment task. LeapAlign optimizes early generation steps more effectively, resulting in the greatest improvement across the entire fine-tuning process.

- Figure 5 Comparison of GenEval score improvement during fine-tuning among ReFL, DRaFT-LV, DRTune, and LeapAlign.

#### B Additional Results on Stable Diffusion 3.5 Medium

To verify that LeapAlign can also achieve strong performance on other flow matching models, we conduct experiments on Stable Diffusion 3.5 Medium [6]. We fine-tune and evaluate this model at a resolution of 512 × 512 for 200 iterations. All other settings follow those used for the general preference alignment task with HPSv2.1 in the main text. Results are shown in Table 4.

We observe that LeapAlign again achieves the best performance across all evaluators compared with other direct-gradient methods. These results demonstrate that LeapAlign generalizes well to other flow matching models and continues to deliver strong improvements.

#### C Additional Analysis

Analysis of the nested gradient. To better understand the role of the nested gradient, we conduct an additional experiment in which the first step of the leap trajectory is optimized only through the nested

- Table 4 Comparison of post-training methods on Stable Diffusion 3.5 Medium. For the general preference alignment experiments, all methods use HPSv2.1 as the reward model, so HPSv2.1 is reported as an ‘in-domain’ metric. ∗ Implemented by us due to the absence of an official implementation. ‡ Adapted to Stable Diffusion 3.5 Medium by us from the official implementation. Best scores are in bold, and second-best scores are underlined.

In-Domain Out-of-Domain

Method HPSv2.1 ↑ HPSv3 ↑ PickScore ↑ UnifiedReward-Alignment ↑ UnifiedReward-IQ ↑ ImageReward ↑ Pretrained Model

SD3.5-M 0.2967 12.2846 22.5189 3.4436 3.5565 1.0614 Direct-Gradient Methods

ReFL‡ 0.3833 15.2488 23.5015 3.4810 3.6872 1.4239 DRaFT-LV∗ 0.3506 14.6022 23.0747 3.4691 3.6519 1.3008 DRTune∗ 0.3828 15.2541 23.4711 3.4738 3.6667 1.4320

LeapAlign 0.3915 15.5780 23.6180 3.4896 3.7182 1.4736

gradient. Specifically, we remove the single-step gradient at timestep k, as shown in Eq. 14.

∂vθ(xj) ∂θ −(k − j)

∂x0 ∂θ

∂vθ(xk) ∂θ

=−j

single-step gradients at k and j

∂vθ(xj) ∂xj

∂vθ(xk) ∂θ

+ α j(k − j)

.

nested gradient

(14)

We fine-tune Flux with HPSv2.1 as the reward model and report results from the non-EMA checkpoint to better expose how the gradient magnitude affects optimization behavior. Fig. 6 shows the average test-set HPSv2.1 score together with the average gradient norm during fine-tuning. Directly using the full nested

- gradient (α = 1) substantially increases the gradient norm and degrades performance compared with removing the nested gradient (α = 0). In contrast, applying a moderate discount (α = 0.3) reduces the gradient norm and improves performance over α = 0. This observation is consistent with the main-paper analysis of gradient discounting and indicates that the nested gradient is beneficial when its magnitude is properly controlled.

- Figure 6 Analysis of the nested gradient. We fine-tune the first step of the leap trajectory using only the nested

gradient and vary α, which scales its magnitude. Left: average HPSv2.1 score on the test set. Right: average gradient norm during fine-tuning. Directly using the full nested gradient (α = 1) increases the gradient norm and hurts performance, while moderate gradient discounting factor (α = 0.3) provides the best trade-off.

###### Impact of the loss threshold λ. Table 5 presents an ablation study on the loss threshold λ, which controls

the strength of reward maximization. When λ is too small, the model is under-optimized, resulting in inferior performance. When λ is too large, optimization becomes overly aggressive, which hurts out-ofdomain generalization and lowers reward scores. Among the tested values, λ = 0.55 achieves the best overall performance, indicating the best trade-off between optimization strength and generalization.

Table 5 Impact of the loss threshold λ.

In-Domain Out-of-Domain λ HPSv2.1 HPSv3 PickScore ImageReward 0.35 0.3860 15.3635 23.4735 1.3510

0.55 0.4092 15.7678 23.7137 1.5104

0.75 0.4091 15.7274 23.7061 1.4844 0.95 0.4023 15.7254 23.5082 1.3888

#### D Summary of Direct-Gradient Methods

We summarize the direct-gradient methods, including ReFL [53], DRaFT-LV [3], DRTune [52], and our proposed LeapAlign, in Algorithm 1.

#### E Additional Implementation and Training Details

During fine-tuning, we set the gradient discounting factor α to 0.3 when using HPSv2.1 [51], and to 0.1 when using PickScore [17] or HPSv3 [32] as the reward model. We use a learning rate of 1e−5 when fine-tuning with HPSv2.1 [51] or PickScore [17] as the reward model. For HPSv3 [32], we empirically find that its backpropagated gradients are relatively large, so we adopt a smaller learning rate of 8e−6. The loss thresholds λ for HPSv2.1, PickScore, and HPSv3 are set to 0.55, 0.4, and 13.5, respectively.

Hyperparameters of baseline methods. We configure the hyperparameters of baseline direct-gradient methods following the recommended settings in their original papers. Specifically, for DRaFT-LV [3], we set the re-noising steps n to 2. For DRTune, we set the training timesteps K to 2. For both DRTune and ReFL [53], we randomly select the early-stop timestep from the last 11 generation steps out of the total 25. We do not include the pre-training loss when fine-tuning with ReFL, as EMA is sufficient to prevent overfitting.

#### F Derivation of the One-Step Leap Prediction

Let x1 ∼ N(0,I) be a Gaussian noise sample and x0 ∼ pdata be a real image drawn from the data distribution. Under a general scheduler (αt,βt), we can express

xt = αtx0 + βtx1. (15) Following the derivation of Domingo-Enrich et al. [5], the velocity field is defined as

v(xt, t) = E

= E

dxt dt

αtx0 + βtx1 = xt

dβt dt

dαt dt

x1 αtx0 + βtx1 = xt .

x0 +

(16)

A simple rearrangement of Eq. 15 gives

xt − βtx1 αt

x0 =

. Substituting this into Eq. 16 yields

xt − βtx1 αt

dαt dt

dβt dt

v(xt, t) = E

x1 αtx0 + βtx1 = xt

+

xt − βtxˆ1|t αt

dαt dt

dβt dt

=

+

xˆ1|t,

(17)

where xˆ1|t := E[x1 | αtx0 + βtx1 = xt]. Solving for xˆ1|t gives

v(xt,t) − dα

xt αt

t

dt

. (18)

xˆ1|t =

dβt

βt αt

dt − dα

t

dt

t−αtx0

Similarly, rewriting x1 = x

βt and substituting into Eq. 16 gives

v(xt,t) − dβ

xt βt

t

dt

. (19)

xˆ0|t =

dt − dβ

dαt

αt βt

t

dt

To extend the prediction to an arbitrary timestep j, we condition on xk at timestep t = k. Let

α˙k :=

dαt dt t=k

, β˙k :=

dβt dt t=k

,

denote the time derivatives of αt and βt evaluated at t = k, and let v(xk,k) be the velocity at xk. The one-step leap prediction is then

xˆj|k = αjxˆ0|k + βjxˆ1|k

 

  .

v(xk, k) − α˙k αxk

v(xk, k) − β˙k xβk

(20)

k α˙k − β˙k αβk

k β˙k − α˙k αβk

+ βj

= αj

k

k

Under rectified flow matching [30], the scheduler takes the form

αt = 1 − t, βt = t, so that

α˙k = −1, β˙k = 1. Substituting these into Eq. 20 yields the simplified expression

xˆj|k = xk − (k − j)v(xk,k). (21)

Finally, with a pretrained flow matching model vθ(xk,k) ≈ v(xk,k), the practical one-step leap prediction becomes

xˆj|k = xk − (k − j)vθ(xk,k). (22)

#### G Derivation of the Backpropagated Gradient Through the Leap Trajectory

Let k and j be two randomly selected timesteps from the full generation trajectory with k > j. The forward pass of the leap trajectory without gradient discounting is

xˆj|k = xk − (k − j)vθ(xk), (23) xj = xˆj|k + stop_gradient(xj − xˆj|k), (24)

xˆ0|j = xj − j vθ(xj), (25) x0 = xˆ0|j + stop_gradient(x0 − xˆ0|j). (26)

In the derivation below, the rollout states xk, xj, and x0 from the full trajectory are treated as detached constants, and gradients are propagated only through the leap trajectory.

The gradient of the final image x0 with respect to the parameters θ is

∂x0 ∂θ

=

=

∂xˆ0|j ∂θ

∂x0 ∂xˆ0|j

∂vθ(xj) ∂θ

∂x0 ∂xˆ0|j −j

+

∂vθ(xj) ∂xj

∂xj ∂θ − j

∂xj ∂θ

,

(27)

and

∂xˆ0|j = 1 and ∂∂xxˆ j

Since ∂x

0

j|k

∂xˆj|k ∂θ

∂xj ∂θ

∂xj ∂xˆj|k

=

∂xj ∂xˆj|k −(k − j)

∂vθ(xk) ∂θ

=

= 1, substituting Eq. 28 into Eq. 27 gives

.

∂x0 ∂θ

∂vθ(xj) ∂θ − (k − j)

∂vθ(xk) ∂θ

= −j

∂vθ(xj) ∂xj

∂vθ(xk) ∂θ

+ j(k − j)

.

(28)

(29)

When gradient discounting is applied with gradient discounting factor α ∈ [0,1], we modify Eq. 25 as xˆ0|j = xj − j vθ αxj + (1 − α)stop_gradient(xj) . (30)

In the forward pass we still have vθ(αxj + (1 − α)stop_gradient(xj)) = vθ(xj), but during backpropagation the gradient flowing through ∂v∂xθ(xj)

is scaled by a factor of α, since

j

As a result, Eq. 29 becomes

∂ αxj + (1 − α)stop_gradient(xj) ∂xj

= α.

∂x0 ∂θ

∂vθ(xj) ∂θ − (k − j)

∂vθ(xk) ∂θ

= −j

∂vθ(xj) ∂xj

∂vθ(xk) ∂θ

+ α j(k − j)

.

(31)

#### H Additional Qualitative Results on the GenEval Benchmark

We present additional qualitative comparisons on the GenEval benchmark across the pretrained Flux model, ReFL, DRaFT-LV, DRTune, and LeapAlign. As shown in Figure 7, LeapAlign more effectively adjusts the global structure of the generated images, leading to outputs that more faithfully follow the text prompts.

#### I Qualitative Results of Flux Fine-Tuned with LeapAlign

We present qualitative results of Flux fine-tuned with LeapAlign using HPSv3 as the reward model in Figures 8 and 9. The fine-tuned model generates visually compelling and realistic images across diverse styles, themes, and scenarios, demonstrating that LeapAlign effectively aligns flow matching models with human preferences.

Algorithm 1 Summary of direct-gradient methods

Inputs: pre-trained flow matching model vθ with parameters θ, reward r, prompt dataset pc, learning rate η, early-stop timestep range m (ReFL, DRTune), total number of discrete timesteps T, number of training timesteps K (DRTune), number of re-noising steps n (DRaFT-LV), and gradient discounting factor α (LeapAlign).

- 1: while not converged do
- 2: tmin =

randint(1, m) if ReFL ∥ DRTune 1 if LeapAlign ∥ DRaFT-LV

- 3: if DRTune then
- 4: s = randint(1, T − (K − 1)⌊T/K⌋)
- 5: ttrain = {s + i⌊T/K⌋ | i = 0, 1, . . . , K − 1}
- 6: if LeapAlign then
- 7: tk, tj ∼ {1, . . . , T} with tk > tj
- 8: c ∼ pc, xT ∼ N(0, I)
- 9: for t = T, . . . ,1 do
- 10: grad_on ←

 



t = tmin if ReFL ∥ DRaFT-LV t = tk∥t = tj if LeapAlign True if DRTune False otherwise

- 11: if grad_on then
- 12: enable_grad()
- 13: else
- 14: disable_grad()
- 15: if DRTune then
- 16: vt = vθ(stop_grad(xt), t, c)
- 17: if t ∈/ ttrain then
- 18: vt = stop_grad(vt)
- 19: else if LeapAlign && t = tj then
- 20: xt = xˆj|k + stop_grad(xt − xˆj|k)
- 21: vt = vθ(αxt + (1 − α)stop_grad(xt), t, c)
- 22: else
- 23: vt = vθ(xt, t, c)
- 24: if (ReFL ∥ DRTune) && t = tmin then
- 25: x0 ≈ one_step_leap_pred(xt, vt, t, 0)
- 26: break
- 27: if LeapAlign then
- 28: if t = tk then
- 29: xˆj|k = one_step_leap_pred(xt, vt, tk, tj)
- 30: else if t = tj then
- 31: xˆ0|j = one_step_leap_pred(xt, vt, tj, 0)
- 32: vt = stop_grad(vt)
- 33: xt = stop_grad(xt)
- 34: xt−1 = step(xt, vt, t)
- 35: enable_grad()
- 36: if LeapAlign then
- 37: w = stop_grad(1/(diff(xj, xˆj|k) + diff(x0, xˆ0|j)))
- 38: x0 = xˆ0|j + stop_grad(x0 − xˆ0|j)
- 39: else
- 40: w = 1
- 41: g = −w∇θr(x0, c)
- 42: if DRaFT-LV then
- 43: for i = 1, . . . , n do
- 44: ϵ ∼ N(0, I)
- 45: xi1 = α1stop_grad(x0) + β1ϵ
- 46: xi0 = step(xi1, vθ(xi1, 1, c), 1)
- 47: g = g − ∇θr(xi0, c)
- 48: θ ← θ − ηg
- 49: return θ

##### Flux ReFL DRaFT-LV DRTune LeapAlign

[Figure 15]

a photo of an elephant below a surfboard

[Figure 16]

a photo of a horse and a train

[Figure 17]

a photo of a black hot dog

[Figure 18]

a photo of a yellow stop sign and a blue potted plant

[Figure 19]

a photo of four hot dogs

[Figure 20]

a photo of four sinks

[Figure 21]

a photo of a bench left of a bear

[Figure 22]

a photo of a vase above a fire hydrant

[Figure 23]

a photo of a brown computer keyboard

[Figure 24]

a photo of a pink dining table and a black sandwich

###### Figure 7 Additional qualitative comparisons on the GenEval benchmark.

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

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

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

###### Figure 8 Qualitative results of Flux fine-tuned with LeapAlign using HPSv3 as the reward model.

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

###### Figure 9 Qualitative results of Flux fine-tuned with LeapAlign using HPSv3 as the reward model.

