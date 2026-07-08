### PRDP: Proximal Reward Difference Prediction for Large-Scale Reward Finetuning of Diffusion Models

# arXiv:2402.08714v2[cs.LG]27Mar2024

Fei Deng1,2*, Qifei Wang1, Wei Wei3†, Matthias Grundmann1, Tingbo Hou1

1Google, 2Rutgers University, 3Accenture https://fdeng18.github.io/prdp

a close up of a cat wearing a pikachu hat, reddit, gif, real life charmander, very aesthetic!!!!!!, soft!!

A painting of a girl standing on a mountain looking out at an approaching storm over the ocean, with wind blowing and ocean mist, surrounded by lightning.

A night scene of a lavender ﬁeld with a town and church in the background, reminiscent of Vincent van Gogh's style.

cinematic still of an adorable walking robot in the desert, at sunset

rural house with a garden and a swimming pool

An abandoned Segway in the forest

A corgi dressed as a bee costume.

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

StableDiffusionv1.4PRDP

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

Figure 1. Generation samples on complex, unseen prompts. Our proposed method, PRDP, achieves stable black-box reward finetuning for diffusion models for the first time on large-scale prompt datasets, leading to superior generation quality on complex, unseen prompts. Here, PRDP is finetuned from Stable Diffusion v1.4 on the training set prompts of Pick-a-Pic v1 dataset, using a weighted combination of rewards: PickScore = 10, HPSv2 = 2, Aesthetic = 0.05. The images within each column are generated using the same random seed.

###### Abstract

Reward finetuning has emerged as a promising approach to aligning foundation models with downstream objectives. Remarkable success has been achieved in the language domain by using reinforcement learning (RL) to maximize rewards that reflect human preference. However, in the vision domain, existing RL-based reward finetuning methods are limited by their instability in large-scale training, rendering them incapable of generalizing to complex, unseen prompts. In this paper, we propose Proximal Reward Difference Prediction (PRDP), enabling stable black-box reward finetuning for diffusion models for the first time on large-scale prompt datasets with over 100K prompts. Our key innovation is the Reward Difference Prediction (RDP) objective that has the same optimal solution as the RL ob-

*Work done during an internship at Google. †Work done while working at Google.

jective while enjoying better training stability. Specifically, the RDP objective is a supervised regression objective that tasks the diffusion model with predicting the reward difference of generated image pairs from their denoising trajectories. We theoretically prove that the diffusion model that obtains perfect reward difference prediction is exactly the maximizer of the RL objective. We further develop an online algorithm with proximal updates to stably optimize the RDP objective. In experiments, we demonstrate that PRDP can match the reward maximization ability of well-established RL-based methods in small-scale training. Furthermore, through large-scale training on text prompts from the Human Preference Dataset v2 and the Pick-a-Pic v1 dataset, PRDP achieves superior generation quality on a diverse set of complex, unseen prompts whereas RL-based methods completely fail.

###### 1. Introduction

Diffusion models have achieved remarkable success in generative modeling of continuous data, especially in photorealistic text-to-image synthesis [7, 15, 30, 36, 37, 40, 44, 46]. However, the maximum likelihood training objective of diffusion models is often misaligned with their downstream use cases, such as generating novel compositions of objects unseen during training, and producing images that are aesthetically preferred by humans.

A similar misalignment problem exists in language models, where exactly matching the model output to the training distribution tends to yield undesirable model behavior. For example, the model may output biased, toxic, or harmful content. A successful solution, called reinforcement learning from human feedback (RLHF) [2, 31, 47, 61], is to use reinforcement learning (RL) to finetune the language model such that it maximizes some reward function that reflects human preference. Typically, the reward function is defined by a reward model pretrained from human preference data.

Inspired by the success of RLHF in language models, researchers have developed several reward models in the vision domain [22, 23, 53–55] that are similarly trained to be aligned with human preference. Furthermore, two recent works, DDPO [4] and DPOK [10], have explored using RL to finetune diffusion models. They both view the denoising process as a Markov decision process [9], and apply policy gradient methods such as PPO [42] to maximize rewards.

However, policy gradients are notoriously prone to high variance, causing training instability. To reduce variance, a common approach is to normalize the rewards by subtracting their expected value [48, 51]. DPOK fits a value function to estimate the expected reward, showing promising results when trained on ∼200 prompts. Alternatively, DDPO maintains a separate buffer for each prompt to track the mean and variance of rewards, demonstrating stable training on ∼400 prompts and better performance than DPOK. Nevertheless, we find that DDPO still suffers from training instability on larger numbers of prompts, depriving it of the benefits offered by training on large-scale prompt datasets.

In this paper, we propose Proximal Reward Difference Prediction (PRDP), a scalable reward maximization algorithm that does not rely on policy gradients. To the best of our knowledge, PRDP is the first method that achieves stable large-scale finetuning of diffusion models on more than 100K prompts for black-box reward functions.

Inspired by the recent success of DPO [35] that converts the RLHF objective for language models into a supervised classification objective, we derive for diffusion models a new supervised regression objective, called Reward Difference Prediction (RDP), that has the same optimal solution as the RLHF objective while enjoying better training stability. Specifically, our RDP objective tasks the diffusion model with predicting the reward difference of generated

image pairs from their denoising trajectories. We prove that the diffusion model that obtains perfect reward difference prediction is exactly the maximizer of the RLHF objective. We further propose proximal updates and online optimization to improve training stability and generation quality.

Our contributions are summarized as follows:

- • We propose PRDP, a scalable reward finetuning method for diffusion models, with a new reward difference prediction objective and its stable optimization algorithm.
- • PRDP achieves stable black-box reward maximization for diffusion models for the first time on large-scale prompt datasets with over 100K prompts.
- • PRDP exhibits superior generation quality and generalization to unseen prompts through large-scale training.

###### 2. Preliminaries

In this section, we briefly introduce the generative process of denoising diffusion probabilistic models (DDPMs) [15, 44, 46]. Given a text prompt c, a text-to-image DDPM πθ with parameters θ defines a text-conditioned image distribution πθ(x0|c) as follows:

πθ(x0|c) = πθ(x0:T|c)dx1:T

(1)

T

πθ(xt−1|xt,c)dx1:T,

= p(xT)

t=1

where x0 is the image, and x1:T are latent variables of the same dimension as x0. Typically, p(xT) = N(0,I), and

πθ(xt−1|xt,c) = N(xt−1;µθ(xt,c),σt2I) (2) is a Gaussian distribution with learnable mean and fixed covariance. To generate an image x0 ∼ πθ(x0|c), DDPM uses ancestral sampling. That is, it samples the full denoising trajectory x0:T ∼ πθ(x0:T|c), by first sampling xT ∼ p(xT), and then sampling xt−1 ∼ πθ(xt−1|xt,c) for t = T,...,1. Conversely, given a denoising trajectory x0:T, we can analytically compute its log-likelihood as

T

log πθ(xt−1|xt,c) (3)

log πθ(x0:T|c) = log p(xT) +

t=1

T

∥xt−1 − µθ(xt,c)∥2 σt2

- 1

- 2

+ C, (4)

= −

t=1

where C is a constant independent of θ.

###### 3. Method

###### 3.1. Reward Difference Prediction for KL-Regularized Reward Maximization

We start derivation from the typical RLHF objective [10]:

0,c[r(x0,c) − βKL[πθ(x0|c)||πref(x0|c)]]. (5)

Ex

max

πθ

Groundtruth Reward Difference

Denoising Trajectories

- r(xa0,c)
- r(xb0,c)

xaT xa0

Reward Model

##  r

|[Figure 15]|
|---|

|[Figure 16]|
|---|

|[Figure 17]|
|---|

|[Figure 18]|
|---|

Prompt c

Diffusion Model

MSE Loss

A green colored rabbit

xbT xb0

##### ⇡✓

###### L(✓)

- rˆ✓(xa0:T,c)
- rˆ✓(xb0:T,c)

Diffusion Model

old

|[Figure 19]|
|---|

|[Figure 20]|
|---|

|[Figure 21]|
|---|

|[Figure 22]|
|---|

 ˆr✓

⇡✓

Model Snapshot

Predicted Reward Difference

- Figure 2. PRDP framework. PRDP mitigates the instability of policy gradient methods by converting the RLHF objective to an equivalent supervised regression objective. Specifically, given a text prompt, PRDP samples two images, and tasks the diffusion model with predicting the reward difference of these two images from their denoising trajectories. The diffusion model is updated by stochastic gradient descent on the MSE loss that measures the prediction error. We prove that the MSE loss and the RLHF objective have the same optimal solution.

Here, we seek to finetune the diffusion model πθ by maximizing a given reward function r(x0,c) with a KL regularization, whose strength is controlled by a hyperparameter β. The reward function can be a pretrained reward model (e.g., HPSv2 [53], PickScore [22]) that measures the generation quality, and the KL regularization discourages πθ from deviating too far from the pretrained diffusion model πref (e.g., Stable Diffusion [37]). This helps πθ to preserve the overall generation capability of πref, and keeps the generated images x0 close to the distribution where the reward model is accurate. The expectation is taken over text prompts c ∼ p(c) and images x0 ∼ πθ(x0|c), where p(c) is a predefined prompt distribution, usually a uniform distribution over a set of training prompts.

Z(c) is intractable, Eq. (7) cannot be directly used to compute πθ⋆. However, it reveals that πθ⋆ must satisfy

πθ⋆(x¯|c) πref(x¯|c)

1 β

r(x0,c) − log Z(c) (8)

log

=

for all x¯ and c. This allows us to cancel the log Z(c) term by considering two denoising trajectories x¯a and x¯b that correspond to the same text prompt c:

πθ⋆(x¯a|c) πref(x¯a|c) − log

πθ⋆(x¯b|c) πref(x¯b|c)

r(xa0,c) − r(xb0,c) β

=

. (9) Define

log

In contrast to language models, the KL regularization in Eq. (5) cannot be computed analytically, due to the intractable integral defined in Eq. (1). Hence, we instead maximize a lower bound of the objective in Eq. (5):

πθ(x¯|c) πref(x¯|c)

, (10) ∆ˆrθ(x¯a,x¯b,c) := rˆθ(x¯a,c) − rˆθ(x¯b,c), (11)

rˆθ(x¯,c) := log

∆r(xa0,xb0,c) := r(xa0,c) − r(xb0,c), (12) then Eq. (9) becomes

0,c[r(x0,c) − βKL[πθ(x¯|c)||πref(x¯|c)]], (6)

Ex

max

πθ

where x¯ := x0:T is the full denoising trajectory. We provide the proof of lower bound in Appendix A.1.

∆ˆrθ⋆(x¯a,x¯b,c) = ∆r(xa0,xb0,c)/β. (13)

While it is possible to apply REINFORCE [51] or more advanced policy gradient methods [4, 10, 42] to optimize Eq. (6), we empirically find they are hard to scale to large numbers of prompts due to training instability. Inspired by DPO [35], we propose to reformulate Eq. (6) into a supervised learning objective, allowing stable training on more than 100K prompts.

This motivates us to optimize πθ by minimizing the following mean squared error (MSE) loss:

L(θ) = Ex¯a,x¯b,c [lθ(x¯a,x¯b,c)] (14) := Ex¯a,x¯b,c ∆ˆrθ(x¯a,x¯b,c) − ∆r(xa0,xb0,c)/β 2 .

We call L(θ) the Reward Difference Prediction (RDP) objective, since we learn πθ by predicting the reward difference ∆r(xa0,xb0,c) instead of directly maximizing the reward. An illustration is provided in Fig. 2. We further show in Appendix A.3 that

First, we derive the optimal solution to Eq. (6) as:

1 Z(c)

1 β

r(x0,c) , (7)

πθ⋆(x¯|c) =

πref(x¯|c)exp

where Z(c) = πref(x¯|c)exp(r(x0,c)/β)dx¯ is the partition function. Proof can be found in Appendix A.2. Since

###### πθ = πθ⋆ ⇐⇒ L(θ) = 0. (15)

Training w/o Proximal Updates

Training w/ Proximal Updates

|[Figure 23]|[Figure 24]|[Figure 25]|[Figure 26]|
|---|---|---|---|

|[Figure 27]|[Figure 28]|[Figure 29]|[Figure 30]|
|---|---|---|---|

- Figure 3. Effect of proximal updates. We show generation samples during the PRDP training process. Here, we use the small-scale setup described in Sec. 4.1 and HPSv2 as the reward model. All samples use the same prompt “A painting of a deer” and the same random seed. (Left) Without proximal updates, training is quite unstable, and the generation quickly becomes meaningless noise. (Right) With proximal updates, the training stability is remarkably improved.

Algorithm 1 PRDP Training

Require: pretrained diffusion model πref, training prompt distribution p(c), reward model r(x0, c), training epochs E, gradient updates K per epoch, prompt batch size N, image batch size B per prompt

- 1: πθ ← πref ▷ Initialization
- 2: for epoch e = 1, . . . , E do
- 3: πθold ← πθ ▷ Model snapshot
- 4: {cn}Nn=1 iid∼ p(c) ▷ Sample text prompts
- 5: for each text prompt cn do
- 6: {x¯n,i}Bi=1 iid∼ πθold(x¯|cn) ▷ Denoising trajectories
- 7: end for
- 8: Obtain rewards r(xn,i0 , cn) for all n, i
- 9: for gradient step k = 1, . . . , K do
- 10: L(θ) ← N(1B2)

N n=1 1≤i<j≤B lθ(x¯n,i, x¯n,j, cn)

- 11: Update model parameters θ by gradient descent
- 12: end for
- 13: end for

###### 3.2. Online Optimization

To estimate the expectation in L(θ), we need samples of denoising trajectories x¯a and x¯b that correspond to the same prompt c. A straightforward approach, as similarly done in

DPO, is to sample x¯a,x¯b iid∼ πref(x¯|c). This can be implemented as uniform sampling from a fixed offline dataset generated by the pretrained model πref.

However, the offline dataset lacks sufficient coverage of samples from πθ(x¯|c) that keeps updating, leading to suboptimal generation quality. Therefore, we propose an online optimization procedure, inspired by online RL algorithms. Specifically, we sample x¯a,x¯b iid∼ πθ

(x¯|c), where θold is a snapshot of the diffusion model parameters θ, and we set θold ← θ every K gradient updates. In practice, we use πθ

old

to generate a batch of denoising trajectories, and then use all pairs of denoising trajectories in the batch to compute the loss L(θ). Details are provided in Algorithm 1. We will show in Sec. 4.3 that online optimization significantly improves generation quality.

old

###### 3.3. Proximal Updates for Stable Training

We find in our experiments that directly optimizing Eq. (14) is prone to training instability, as illustrated in Fig. 3 (Left). This is likely due to excessively large model updates during training. To resolve this issue, we propose proximal updates that remove the incentive for moving πθ too far away from πθ

. Inspired by PPO [42], we achieve this by clipping the log probability ratio log(πθ(x¯|c)/πθ

old

(x¯|c)) to be within a small interval [−ϵ′,ϵ′]. This can be implemented by clipping the rˆθ(x¯,c) as rˆθclip(x¯,c) :=

old

(x¯,c) + ϵ′), (16) because log(πθ(x¯|c)/πθ

(x¯,c) − ϵ′,rˆθ

clip(ˆrθ(x¯,c),rˆθ

old

old

(x¯,c). We then use rˆθclip(x¯,c) to compute the clipped MSE loss lθclip(x¯a,x¯b,c) :=

(x¯|c)) = rˆθ(x¯,c) − rˆθ

old

old

2

∆ˆrθclip(x¯a,x¯b,c) − ∆r(xa0,xb0,c)/β

, (17)

where ∆ˆrθclip(x¯a,x¯b,c) := rˆθclip(x¯a,c)−rˆθclip(x¯b,c). Similar to PPO [42], our final loss is the maximum of the

clipped and unclipped MSE loss:

lθ(x¯a,x¯b,c) ← max(lθ(x¯a,x¯b,c),lθclip(x¯a,x¯b,c)). (18) This ensures that we minimize an upper bound of the original loss, making the optimization problem well-defined.

In practice, the clipping in Eq. (16) is decomposed and applied at each denoising step t. First, rˆθ(x¯,c) can be decomposed as rˆθ(x¯,c) = Tt=1 rˆθ,t(x¯,c), where

rˆθ,t(x¯,c) := log(πθ(xt−1|xt,c)/πref(xt−1|xt,c)). (19) We apply clipping to each rˆθ,t(x¯,c) as rˆθ,tclip(x¯,c) :=

old,t(x¯,c) + ϵ), (20) where ϵ is the stepwise clipping range. Finally, we replace Eq. (16) with

old,t(x¯,c) − ϵ,rˆθ

clip(ˆrθ,t(x¯,c),rˆθ

T

rˆθclip(x¯,c) :=

rˆθ,tclip(x¯,c). (21)

t=1

As shown in Fig. 3 (Right), our proposed proximal updates can remarkably improve optimization stability.

snake

whale horse duck monkey goat

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

Diffusion DDPOPRDP

Stable

[Figure 37]

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

- Figure 4. Generation samples from small-scale training. DDPO and PRDP are finetuned from Stable Diffusion v1.4 on 45 prompts consisting of common animal names, with HPSv2 (Left) and PickScore (Right) as the reward model. Samples within each column use the same random seed. The prompt template is “A painting of a ⟨animal⟩”, where the ⟨animal⟩ is listed on top of each column. All prompts are seen during training. Both DDPO and PRDP significantly improve the generation quality, with PRDP being slightly better.

###### 4. Experiments

In our experiments, we first verify on a set of 45 prompts that PRDP can match the reward maximization ability of DDPO [4], which is based on the well-established PPO [42] algorithm. We then conduct a large-scale training on more than 100K prompts from the training set of HPDv2 [53], showing that PRDP can successfully handle large-scale training whereas DDPO fails. We further perform a largescale multi-reward finetuning on the training set prompts of Pick-a-Pic v1 dataset [22], highlighting the superior generation quality of PRDP on complex, unseen prompts. Finally, we showcase the advantages of our algorithm design, such as online optimization and KL regularization.

###### 4.1. Experimental Setup

To perform reward finetuning, we need a pretrained diffusion model, a pretrained reward model, and a training set of prompts. For all experiments, we use Stable Diffusion (SD) v1.4 [37] as the pretrained diffusion model, and finetune the full UNet weights. For sampling, during both training and evaluation, we use the DDPM sampler [15] with 50 denoising steps and a classifier-free guidance [14] scale of 5.0.

Small-scale setup. We use a set of 45 prompts, with the template “A painting of a ⟨animal⟩”, where the ⟨animal⟩ is taken from the list of common animal names used in DDPO.

Table 1. Reward score comparison on small-scale training.

SD v1.4 DDPO PRDP

HPSv2 0.2855 0.3398 0.3471 PickScore 0.2179 0.2664 0.2700

We conduct reward finetuning separately for two recently proposed reward models, HPSv2 [53] and PickScore [22]. We train for 100 epochs, where in each epoch, we sample 32 prompts and 16 images per prompt. The evaluation uses the same set of prompts as training. We report reward scores averaged over 256 random samples per prompt.

Large-scale setup. Following DRaFT [6], we use more than 100K prompts from the training set of HPDv2, and finetune for HPSv2 and PickScore separately. We train for 1000 epochs. In each epoch, we sample 64 prompts and 8 images per prompt. We evaluate the finetuned model on 500 randomly sampled training prompts, as well as a variety of unseen prompts, including 500 prompts from the Pick-a-Pic v1 test set, and 800 prompts from each of the four benchmark categories of HPDv2, namely animation, concept art, painting, and photo. We report reward scores averaged over 64 random samples per prompt.

###### Large-scale multi-reward setup. We mostly follow the

The image is a wooden sculpture of a cute robot with cat ears, displayed in a contemporary art gallery.

cinematic still of highly reﬂective stainless steel train in the desert, at sunset

An anthropomorphic frog wizard wearing a cape and holding a wand.

Digital art of a cherry tree overlooking a valley with a waterfall at sunset.

A monkey in a blue top hat painted in oil by Vincent van Gogh in the 1800s.

A chibi frog character surﬁng at the beach.

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

Diffusion DDPOPRDP

Stable

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

- Figure 5. Generation samples from large-scale training. DDPO and PRDP are finetuned from Stable Diffusion v1.4 on over 100K prompts from the training set of HPDv2, with HPSv2 (Left) and PickScore (Right) as the reward model. Samples within each column are generated from the prompt shown on top, using the same random seed. All prompts are unseen during training. PRDP significantly improves the generation quality over Stable Diffusion, whereas DDPO fails to generate reasonable results.

large-scale setup, except that we use the training set prompts of Pick-a-Pic v1 dataset, and a weighted combination of rewards: PickScore = 10, HPSv2 = 2, Aesthetic = 0.05, where Aesthetic is the LAION aesthetic score.

Baselines. DDPO [4] and DPOK [10] are the two most recent RL finetuning methods for black-box rewards. Since DDPO has demonstrated better performance than DPOK, we mainly compare to DDPO. To ensure a fair comparison, we train DDPO and PRDP for the same number of epochs, with the same number of reward queries per epoch. We also use the same random seeds to sample images for evaluation.

###### 4.2. Main Results

Small-scale finetuning. We show generation samples from small-scale finetuning in Fig. 4 and reward scores in Tab. 1. Both DDPO and PRDP can significantly improve the generation quality over Stable Diffusion, with more vivid colors and details. Quantitatively, PRDP achieves slightly better reward scores than DDPO. This verifies that PRDP can match the reward maximization ability of well-established policy gradient methods.

Large-scale finetuning. We present generation samples

from large-scale finetuning in Fig. 5 and reward scores in Tab. 2. We observe that Stable Diffusion generates images with relevant content but low quality. Meanwhile, DDPO fails to give reasonable results. It generates irrelevant, low quality images or even meaningless noise, leading to lower reward scores than Stable Diffusion. This is due to the instability of DDPO in large-scale training, which we further investigate in Appendix B. In contrast, PRDP maintains stability in the large-scale setup, and significantly improves the generation quality on both seen and unseen prompts.

Large-scale multi-reward finetuning. We provide generation samples in Figs. 1 and 11 to 15, and reward scores in Tab. 3, showing the superior generation quality of PRDP on a diverse set of complex, unseen prompts.

###### 4.3. Effect of Online Optimization

In this section, we show that online optimization has a great advantage over offline optimization. To ensure a fair comparison, we use the same number of reward queries and gradient updates for both methods. Specifically, following the small-scale setup, for online training, we use 100 epochs, where each epoch makes 512 queries to the reward model.

Table 2. Reward score comparison on large-scale training.

Seen Prompts Unseen Prompts

Reward Model

Method

HPD v2 Training Set

Pick-a-Pic v1 Test Set

HPD v2 Animation

HPD v2 Concept Art

HPD v2 Painting

HPD v2 Photo

SD v1.4 0.2685 0.2665 0.2737 0.2656 0.2654 0.2750 DDPO 0.2464 0.2501 0.2673 0.2558 0.2570 0.2093 PRDP 0.3175 0.3050 0.3223 0.3175 0.3172 0.3159

HPSv2

SD v1.4 0.2092 0.2082 0.2111 0.2062 0.2059 0.2172 DDPO 0.2032 0.1992 0.2077 0.2125 0.2124 0.1780 PRDP 0.2424 0.2344 0.2450 0.2441 0.2448 0.2387

PickScore

Online Optimization for PickScore

Online Optimization for HPSv2

|[Figure 67]|[Figure 68]|[Figure 69]|[Figure 70]|
|---|---|---|---|

|[Figure 71]|[Figure 72]|[Figure 73]|[Figure 74]|
|---|---|---|---|

|[Figure 75]|[Figure 76]|[Figure 77]|[Figure 78]|
|---|---|---|---|

|[Figure 79]|[Figure 80]|[Figure 81]|[Figure 82]|
|---|---|---|---|

- Figure 6. Effect of online optimization. We show generation samples during the PRDP training process, with HPSv2 (Left) and PickScore (Right) as the reward model. We follow the small-scale training setup. The prompts for the first and the second rows are “A painting of a squirrel” and “A painting of a bird”, respectively. Samples within each row use the same random seed. It can be observed that online optimization continually improves the generation quality.

For offline training, we sample 51200 images from the pretrained Stable Diffusion, obtain their rewards, and then perform the same total number of gradient updates as in online training. We show generation samples during the online optimization process in Fig. 6, and quantitative comparisons in Fig. 7. We observe that online optimization continually improves the generation quality, achieving significantly better reward scores than offline optimization.

###### 4.4. Effect of KL Regularization

A common limitation of reward finetuning is reward hacking, where the finetuned diffusion model exploits inaccuracies in the reward model, and produces undesired images with high reward scores. In this section, we show that the KL regularization in our PRDP formulation can help alleviate this issue. For this purpose, we use the LAION aesthetic predictor as the reward model. It only takes images as input, and can be exploited by disregarding text-image alignment. We follow the small-scale setup, except that we train for 250 epochs and directly use the 45 common animal names as prompts. As demonstrated in Fig. 8, DDPO, without KL regularization, is prone to reward hacking. It completely ig-

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

Figure 7. Comparison of online and offline optimization. We evaluate the reward scores of model checkpoints during online optimization and the final model obtained by offline optimization. We follow the small-scale training setup, and optimize the models for HPSv2 and PickScore separately. Online optimization matches the performance of offline optimization in ∼10 epochs, and keeps improving the reward score afterwards.

nores the text prompts and generates similar images for all prompts. In contrast, PRDP with β = 10 can successfully preserve the text-image alignment while improving the aesthetic quality. More analysis can be found in Appendix C.

cat butterﬂy frog

[Figure 83]

[Figure 84]

[Figure 85]

DDPOPRDP

[Figure 86]

[Figure 87]

[Figure 88]

Figure 8. Effect of KL regularization. We show generation samples from DDPO and PRDP when optimizing the LAION aesthetic score. We use the small-scale training setup, except that we train for 250 epochs. Samples within each column are generated from the prompt shown on top, using the same random seed. DDPO, without KL regularization, over-optimizes the reward, generating similar images for all prompts. In contrast, PRDP, formulated with KL regularization, successfully preserves text-image alignment.

###### 5. Related Work

Diffusion models. As a new class of generative models, diffusion models [15, 44, 46] have achieved remarkable success in a wide variety of data modalities, including images [7, 17, 30, 36, 37, 39–41], videos [16, 43], audios [25], 3D shapes [13, 32, 57, 60], and robotic trajectories [1, 5, 18]. To facilitate control over the content and style of generation, recent works have investigated finetuning diffusion models on various conditioning signals [11, 19, 20, 27, 28, 38, 45, 58]. However, it remains challenging to adapt diffusion models to downstream use cases that are misaligned with the training objective, such as generating novel compositions of objects unseen during training, and producing images that are aesthetically preferred by humans. Although classifier guidance [7] can help mitigate this issue, the classifier requires noisy images as input, making it hard to use off-the-shelf classifiers such as object detectors and aesthetic predictors for guidance. In contrast, we finetune the diffusion model to maximize rewards that reflect downstream objectives. Our method can work with generic off-the-shelf reward models that take clean images as input.

Language model learning from human feedback. The maximum likelihood training objective for language models tends to yield undesirable model behavior, due to the potentially biased, toxic, or harmful content in the training data. Reinforcement learning from human feedback (RLHF) has recently emerged as a successful remedy [2, 3, 12, 26, 29, 31, 47, 52, 61]. Typically, a reward model is first trained from human preference data (e.g., rankings of outputs from

a pretrained language model). Then, the language model is finetuned by online RL algorithms (e.g., PPO [42]) to maximize the score given by the reward model. More recently, DPO [35] proposes a supervised learning method that directly optimizes the language model from preference data, skipping the reward model training and avoiding the instability of RL algorithms. Our method is inspired by DPO and PPO, but designed specifically for diffusion models.

Reward finetuning for diffusion models. Inspired by the success of RLHF in the language domain, researchers have developed several reward models in the vision domain [21–24, 34, 53–56]. Moreover, recent works have explored using these reward models to improve the generation quality of diffusion models. A simple approach, called supervised finetuning [23, 54], is to finetune the diffusion model toward high-reward samples from an offline dataset. Its major drawback is that the generation quality is limited by the offline dataset. For further improvement, RAFT [8] proposes an online variant that iteratively re-generates the dataset. A more direct method for online optimization is to backpropagate the reward function gradient through the denoising process [6, 33, 49, 55]. However, this only works for differentiable rewards. For generic rewards, DDPO [4] and DPOK [10] propose RL finetuning. While they have shown promising results on small prompt sets, they are unstable in large-scale training. Our work addresses the training instability issue, achieving stable reward finetuning on largescale prompt datasets for generic rewards. Concurrent with our work, Diffusion-DPO [50] adapts DPO to efficiently align diffusion models from large-scale offline preference data, and [59] proposes to stabilize large-scale RL finetuning by combining the diffusion model pretraining loss.

###### 6. Conclusion

This paper presents PRDP, the first black-box reward finetuning method for diffusion models that is stable on largescale prompt datasets with over 100K prompts. We achieve this by converting the RLHF objective to an equivalent supervised regression objective and developing its stable optimization algorithm. Our large-scale experiments highlight the superior generation quality of PRDP on complex, unseen prompts, which is beyond the capability of existing RL finetuning methods. We also demonstrate that the KL regularization in the PRDP formulation can help alleviate the common issue of reward hacking. We hope that our work can inspire future research on large-scale reward finetuning for diffusion models.

###### Acknowledgments

We thank authors of DRaFT [6] for sharing their training prompts and reward models. We appreciate helpful discussion with Ligong Han, Yanwu Xu, Yaxuan Zhu, Zhonghao Wang, Yunzhi Zhang, Yang Zhao, and Zhisheng Xiao.

###### References

- [1] Anurag Ajay, Yilun Du, Abhi Gupta, Joshua B. Tenenbaum, Tommi S. Jaakkola, and Pulkit Agrawal. Is conditional generative modeling all you need for decision making? In International Conference on Learning Representations, 2023. 8
- [2] Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, Nicholas Joseph, Saurav Kadavath, Jackson Kernion, Tom Conerly, Sheer El-Showk, Nelson Elhage, Zac Hatfield-Dodds, Danny Hernandez, Tristan Hume, Scott Johnston, Shauna Kravec, Liane Lovitt, Neel Nanda, Catherine Olsson, Dario Amodei, Tom Brown, Jack Clark, Sam McCandlish, Chris Olah, Ben Mann, and Jared Kaplan. Training a helpful and harmless assistant with reinforcement learning from human feedback. arXiv preprint arXiv:2204.05862, 2022. 2, 8
- [3] Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, Carol Chen, Catherine Olsson, Christopher Olah, Danny Hernandez, Dawn Drain, Deep Ganguli, Dustin Li, Eli TranJohnson, Ethan Perez, Jamie Kerr, Jared Mueller, Jeffrey Ladish, Joshua Landau, Kamal Ndousse, Kamile Lukosuite, Liane Lovitt, Michael Sellitto, Nelson Elhage, Nicholas Schiefer, Noemi Mercado, Nova DasSarma, Robert Lasenby, Robin Larson, Sam Ringer, Scott Johnston, Shauna Kravec, Sheer El Showk, Stanislav Fort, Tamera Lanham, Timothy Telleen-Lawton, Tom Conerly, Tom Henighan, Tristan Hume, Samuel R. Bowman, Zac Hatfield-Dodds, Ben Mann, Dario Amodei, Nicholas Joseph, Sam McCandlish, Tom Brown, and Jared Kaplan. Constitutional AI: Harmlessness from AI feedback. arXiv preprint arXiv:2212.08073, 2022. 8
- [4] Kevin Black, Michael Janner, Yilun Du, Ilya Kostrikov, and Sergey Levine. Training diffusion models with reinforcement learning. In International Conference on Learning Representations, 2024. 2, 3, 5, 6, 8, 15, 16, 18
- [5] Chang Chen, Fei Deng, Kenji Kawaguchi, Caglar Gulcehre, and Sungjin Ahn. Simple hierarchical planning with diffusion. In International Conference on Learning Representations, 2024. 8
- [6] Kevin Clark, Paul Vicol, Kevin Swersky, and David J. Fleet. Directly fine-tuning diffusion models on differentiable rewards. In International Conference on Learning Representations, 2024. 5, 8, 17
- [7] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat GANs on image synthesis. In Advances in Neural Information Processing Systems, 2021. 2, 8
- [8] Hanze Dong, Wei Xiong, Deepanshu Goyal, Yihan Zhang, Winnie Chow, Rui Pan, Shizhe Diao, Jipeng Zhang, KaShun SHUM, and Tong Zhang. RAFT: Reward ranked finetuning for generative foundation model alignment. Transactions on Machine Learning Research, 2023. 8
- [9] Ying Fan and Kangwook Lee. Optimizing DDPM sampling with shortcut fine-tuning. In International Conference on Machine Learning, 2023. 2

- [10] Ying Fan, Olivia Watkins, Yuqing Du, Hao Liu, Moonkyung Ryu, Craig Boutilier, Pieter Abbeel, Mohammad Ghavamzadeh, Kangwook Lee, and Kimin Lee. DPOK: Reinforcement learning for fine-tuning text-to-image diffusion models. In Advances in Neural Information Processing Systems, 2023. 2, 3, 6, 8
- [11] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit Haim Bermano, Gal Chechik, and Daniel Cohen-or. An image is worth one word: Personalizing text-to-image generation using textual inversion. In International Conference on Learning Representations, 2023. 8
- [12] Amelia Glaese, Nat McAleese, Maja Tre˛bacz, John Aslanides, Vlad Firoiu, Timo Ewalds, Maribeth Rauh, Laura Weidinger, Martin Chadwick, Phoebe Thacker, Lucy Campbell-Gillingham, Jonathan Uesato, Po-Sen Huang, Ramona Comanescu, Fan Yang, Abigail See, Sumanth Dathathri, Rory Greig, Charlie Chen, Doug Fritz, Jaume Sanchez Elias, Richard Green, Soˇna Mokrá, Nicholas Fernando, Boxi Wu, Rachel Foley, Susannah Young, Iason Gabriel, William Isaac, John Mellor, Demis Hassabis, Koray Kavukcuoglu, Lisa Anne Hendricks, and Geoffrey Irving. Improving alignment of dialogue agents via targeted human judgements. arXiv preprint arXiv:2209.14375, 2022. 8
- [13] Jiatao Gu, Alex Trevithick, Kai-En Lin, Joshua M. Susskind, Christian Theobalt, Lingjie Liu, and Ravi Ramamoorthi. NerfDiff: Single-image view synthesis with NeRF-guided distillation from 3D-aware diffusion. In International Conference on Machine Learning, 2023. 8
- [14] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. In NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications, 2021. 5
- [15] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In Advances in Neural Information Processing Systems, 2020. 2, 5, 8
- [16] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P. Kingma, Ben Poole, Mohammad Norouzi, David J. Fleet, and Tim Salimans. Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303, 2022. 8
- [17] Jonathan Ho, Chitwan Saharia, William Chan, David J. Fleet, Mohammad Norouzi, and Tim Salimans. Cascaded diffusion models for high fidelity image generation. Journal of Machine Learning Research, 23(47):1–33, 2022. 8
- [18] Michael Janner, Yilun Du, Joshua Tenenbaum, and Sergey Levine. Planning with diffusion for flexible behavior synthesis. In International Conference on Machine Learning, 2022. 8
- [19] Jindong Jiang, Fei Deng, Gautam Singh, and Sungjin Ahn. Object-centric slot diffusion. In Advances in Neural Information Processing Systems, 2023. 8
- [20] Bahjat Kawar, Shiran Zada, Oran Lang, Omer Tov, Huiwen Chang, Tali Dekel, Inbar Mosseri, and Michal Irani. Imagic: Text-based real image editing with diffusion models. In CVPR, 2023. 8
- [21] Junjie Ke, Keren Ye, Jiahui Yu, Yonghui Wu, Peyman Milanfar, and Feng Yang. VILA: Learning image aesthetics from

user comments with vision-language pretraining. In CVPR,

2023. 8

- [22] Yuval Kirstain, Adam Polyak, Uriel Singer, Shahbuland Matiana, Joe Penna, and Omer Levy. Pick-a-Pic: An open dataset of user preferences for text-to-image generation. In Advances in Neural Information Processing Systems, 2023. 2, 3, 5, 16, 17
- [23] Kimin Lee, Hao Liu, Moonkyung Ryu, Olivia Watkins, Yuqing Du, Craig Boutilier, Pieter Abbeel, Mohammad Ghavamzadeh, and Shixiang Shane Gu. Aligning textto-image models using human feedback. arXiv preprint arXiv:2302.12192, 2023. 2, 8
- [24] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. BLIP: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In International Conference on Machine Learning, 2022. 8
- [25] Haohe Liu, Zehua Chen, Yi Yuan, Xinhao Mei, Xubo Liu, Danilo Mandic, Wenwu Wang, and Mark D Plumbley. AudioLDM: Text-to-audio generation with latent diffusion models. In International Conference on Machine Learning, 2023. 8
- [26] Hao Liu, Carmelo Sferrazza, and Pieter Abbeel. Chain of hindsight aligns language models with feedback. In International Conference on Learning Representations, 2024. 8
- [27] Ron Mokady, Amir Hertz, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Null-text inversion for editing real images using guided diffusion models. In CVPR, 2023. 8
- [28] Chong Mou, Xintao Wang, Liangbin Xie, Jian Zhang, Zhongang Qi, Ying Shan, and Xiaohu Qie. T2I-adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models. arXiv preprint arXiv:2302.08453, 2023. 8
- [29] Reiichiro Nakano, Jacob Hilton, Suchir Balaji, Jeff Wu, Long Ouyang, Christina Kim, Christopher Hesse, Shantanu Jain, Vineet Kosaraju, William Saunders, Xu Jiang, Karl Cobbe, Tyna Eloundou, Gretchen Krueger, Kevin Button, Matthew Knight, Benjamin Chess, and John Schulman. WebGPT: Browser-assisted question-answering with human feedback. arXiv preprint arXiv:2112.09332, 2021. 8
- [30] Alexander Quinn Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob Mcgrew, Ilya Sutskever, and Mark Chen. GLIDE: Towards photorealistic image generation and editing with text-guided diffusion models. In International Conference on Machine Learning,

2022. 2, 8

- [31] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul F Christiano, Jan Leike, and Ryan Lowe. Training language models to follow instructions with human feedback. In Advances in Neural Information Processing Systems, 2022. 2, 8
- [32] Ben Poole, Ajay Jain, Jonathan T. Barron, and Ben Mildenhall. DreamFusion: Text-to-3D using 2D diffusion. In International Conference on Learning Representations, 2023. 8
- [33] Mihir Prabhudesai, Anirudh Goyal, Deepak Pathak, and Katerina Fragkiadaki. Aligning text-to-image diffusion

- models with reward backpropagation. arXiv preprint arXiv:2310.03739, 2023. 8
- [34] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning, 2021. 8
- [35] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. In Advances in Neural Information Processing Systems, 2023. 2, 3, 8, 13
- [36] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with CLIP latents. arXiv preprint arXiv:2204.06125,

2022. 2, 8

- [37] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, 2022. 2, 3, 5, 8, 16, 17
- [38] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. DreamBooth: Fine tuning text-to-image diffusion models for subject-driven generation. In CVPR, 2023. 8
- [39] Chitwan Saharia, William Chan, Huiwen Chang, Chris Lee, Jonathan Ho, Tim Salimans, David Fleet, and Mohammad Norouzi. Palette: Image-to-image diffusion models. In ACM SIGGRAPH 2022 Conference Proceedings, 2022. 8
- [40] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, Jonathan Ho, David J Fleet, and Mohammad Norouzi. Photorealistic text-to-image diffusion models with deep language understanding. In Advances in Neural Information Processing Systems, 2022. 2
- [41] Chitwan Saharia, Jonathan Ho, William Chan, Tim Salimans, David J. Fleet, and Mohammad Norouzi. Image super-resolution via iterative refinement. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(4):4713– 4726, 2023. 8
- [42] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017. 2, 3, 4, 5, 8, 18
- [43] Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, Devi Parikh, Sonal Gupta, and Yaniv Taigman. Make-A-Video: Text-to-video generation without text-video data. In International Conference on Learning Representations, 2023. 8
- [44] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International Conference on Machine Learning, 2015. 2, 8
- [45] Kihyuk Sohn, Lu Jiang, Jarred Barber, Kimin Lee, Nataniel Ruiz, Dilip Krishnan, Huiwen Chang, Yuanzhen Li, Irfan

- Essa, Michael Rubinstein, Yuan Hao, Glenn Entis, Irina Blok, and Daniel Castro Chin. StyleDrop: Text-to-image synthesis of any style. In Advances in Neural Information Processing Systems, 2023. 8
- [46] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In International Conference on Learning Representations, 2021. 2, 8
- [47] Nisan Stiennon, Long Ouyang, Jeffrey Wu, Daniel Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, and Paul F Christiano. Learning to summarize with human feedback. In Advances in Neural Information Processing Systems, 2020. 2, 8
- [48] Richard S Sutton and Andrew G Barto. Reinforcement learning: An introduction. MIT press, 2018. 2
- [49] Bram Wallace, Akash Gokul, Stefano Ermon, and Nikhil Naik. End-to-end diffusion latent optimization improves classifier guidance. In ICCV, 2023. 8
- [50] Bram Wallace, Meihua Dang, Rafael Rafailov, Linqi Zhou, Aaron Lou, Senthil Purushwalkam, Stefano Ermon, Caiming Xiong, Shafiq Joty, and Nikhil Naik. Diffusion model alignment using direct preference optimization. In CVPR, 2024. 8
- [51] Ronald J Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine learning, 8:229–256, 1992. 2, 3
- [52] Jeff Wu, Long Ouyang, Daniel M Ziegler, Nisan Stiennon, Ryan Lowe, Jan Leike, and Paul Christiano. Recursively summarizing books with human feedback. arXiv preprint arXiv:2109.10862, 2021. 8
- [53] Xiaoshi Wu, Yiming Hao, Keqiang Sun, Yixiong Chen, Feng Zhu, Rui Zhao, and Hongsheng Li. Human preference score v2: A solid benchmark for evaluating human preferences of text-to-image synthesis. arXiv preprint arXiv:2306.09341,

2023. 2, 3, 5, 8, 15, 16, 17

- [54] Xiaoshi Wu, Keqiang Sun, Feng Zhu, Rui Zhao, and Hongsheng Li. Human preference score: Better aligning text-toimage models with human preference. In ICCV, 2023. 8
- [55] Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong. ImageReward: Learning and evaluating human preferences for textto-image generation. In Advances in Neural Information Processing Systems, 2023. 2, 8
- [56] Jiahui Yu, Zirui Wang, Vijay Vasudevan, Legg Yeung, Mojtaba Seyedhosseini, and Yonghui Wu. CoCa: Contrastive captioners are image-text foundation models. Transactions on Machine Learning Research, 2022. 8
- [57] Xiaohui Zeng, Arash Vahdat, Francis Williams, Zan Gojcic, Or Litany, Sanja Fidler, and Karsten Kreis. LION: Latent point diffusion models for 3D shape generation. In Advances in Neural Information Processing Systems, 2022. 8
- [58] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In ICCV, 2023. 8
- [59] Yinan Zhang, Eric Tzeng, Yilun Du, and Dmitry Kislyuk. Large-scale reinforcement learning for diffusion models. arXiv preprint arXiv:2401.12244, 2024. 8

- [60] Linqi Zhou, Yilun Du, and Jiajun Wu. 3D shape generation and completion through point-voxel diffusion. In ICCV,

2021. 8

- [61] Daniel M Ziegler, Nisan Stiennon, Jeffrey Wu, Tom B Brown, Alec Radford, Dario Amodei, Paul Christiano, and Geoffrey Irving. Fine-tuning language models from human preferences. arXiv preprint arXiv:1909.08593, 2019. 2, 8

### PRDP: Proximal Reward Difference Prediction for Large-Scale Reward Finetuning of Diffusion Models

#### Supplementary Material

###### A. Proofs

###### A.1. Lower Bound of RLHF Objective

- In Lemma A.1, we prove that the objective in Equation (6) is a lower bound of the RLHF objective in Equation (5).

- Lemma A.1. Given two diffusion models πθ,πref, a prompt distribution p(c), a reward function r(x0,c), and a constant β > 0, we have:

0∼πθ(x0|c)[r(x0,c)] − βKL[πθ(x0|c)||πref(x0|c)] (22) ≥ Ec∼p(c) Ex

Ec∼p(c) Ex

0∼πθ(x0|c)[r(x0,c)] − βKL[πθ(x¯|c)||πref(x¯|c)] , (23) where x¯ := x0:T is the full denoising trajectory, and πθ,πref are defined as:

π(x0|c) = π(x0:T|c)dx1:T = p(xT)

T

Proof. It suffices to show that for any c,

π(xt−1|xt,c)dx1:T. (24)

t=1

KL[πθ(x¯|c)||πref(x¯|c)] ≥ KL[πθ(x0|c)||πref(x0|c)]. (25) This can be proved similarly as the data processing inequality. We provide the proof below.

πθ(x0:T|c) πref(x0:T|c)

(26)

KL[πθ(x¯|c)||πref(x¯|c)] = Eπ

θ(x0:T|c) log

πθ(x0|c) πref(x0|c)

πθ(x1:T|x0,c) πref(x1:T|x0,c)

(27)

= Eπ

θ(x0:T|c) log

+ log

πθ(x1:T|x0,c) πref(x1:T|x0,c)

πθ(x0|c) πref(x0|c)

(28)

= Eπ

+ Eπ

θ(x0|c) Eπ

θ(x0|c) log

θ(x1:T|x0,c) log

θ(x0|c)[KL[πθ(x1:T|x0,c)||πref(x1:T|x0,c)]] (29) ≥ KL[πθ(x0|c)||πref(x0|c)]. (30)

= KL[πθ(x0|c)||πref(x0|c)] + Eπ

| |
|---|

###### A.2. Maximizer of the Lower Bound of RLHF Objective

- In Lemma A.2, we prove that Equation (7) maximizes the objective in Equation (6), a lower bound of the RLHF objective.

###### Lemma A.2. Define

where

1 Z(c)

πref(x¯|c)exp

πθ⋆(x¯|c) =

1 β

r(x0,c) , (31)

Z(c) = πref(x¯|c)exp

1 β

r(x0,c) dx¯ (32)

is the partition function. Then πθ⋆ is the optimal solution to the following maximization problem:

0∼πθ(x0|c)[r(x0,c)] − βKL[πθ(x¯|c)||πref(x¯|c)] . (33) Proof. We provide the proof below, which is inspired by DPO [35].

Ec∼p(c) Ex

max

πθ

0∼πθ(x0|c)[r(x0,c)] − βKL[πθ(x¯|c)||πref(x¯|c)] (34)

Ec∼p(c) Ex

max

πθ

θ(x¯|c)[r(x0,c)] − βKL[πθ(x¯|c)||πref(x¯|c)] (35)

Ec∼p(c) Ex¯∼π

= max

πθ

πθ(x¯|c) πref(x¯|c)

(36)

Ec∼p(c)Ex¯∼π

θ(x¯|c) r(x0,c) − β log

= max

πθ

πθ(x¯|c) πref(x¯|c) −

1 β

r(x0,c) (37)

Ec∼p(c)Ex¯∼π

= min

θ(x¯|c) log

πθ

 log

  (38)

πθ(x¯|c) πref(x¯|c)exp β 1r(x0,c)

Ec∼p(c)Ex¯∼π

= min

θ(x¯|c)

πθ

πθ(x¯|c) πθ⋆(x¯|c)Z(c)

(39)

Ec∼p(c)Ex¯∼π

= min

θ(x¯|c) log

πθ

πθ(x¯|c) πθ⋆(x¯|c) − log Z(c) (40)

Ec∼p(c) Ex¯∼π

= min

θ(x¯|c) log

πθ

Ec∼p(c)[KL[πθ(x¯|c)||πθ⋆(x¯|c)] − log Z(c)] (41)

= min

πθ

Ec∼p(c)[KL[πθ(x¯|c)||πθ⋆(x¯|c)]]. (42)

= min

πθ

Since KL[πθ(x¯|c)||πθ⋆(x¯|c)] ≥ 0, and KL[πθ(x¯|c)||πθ⋆(x¯|c)] = 0 if and only if πθ(x¯|c) = πθ⋆(x¯|c), we conclude that the optimal solution to Equation (33) is πθ(x¯|c) = πθ⋆(x¯|c) for all c.

| |
|---|

###### A.3. Necessary and Sufficient Conditions for the Optimal Solution

- In Lemma A.3, we provide theoretical justification for our proposed RDP objective in Equation (14).

- Lemma A.3. πθ(x¯|c) = πθ⋆(x¯|c), ∀x¯,c (43)

πθ(x¯a|c) πref(x¯a|c) − log

πθ(x¯b|c) πref(x¯b|c)

r(xa0,c) − r(xb0,c) β

, ∀x¯a,x¯b,c. (44)

⇐⇒ log

=

Proof. We have shown “ =⇒ ” in the main text. We provide the proof for “ ⇐= ” below. Equation (44) implies that

πθ(x¯|c) πref(x¯|c) −

1 β

r(x0,c) (45)

log

is a constant w.r.t. x¯. Therefore, we can write Equation (45) as a function of c alone:

Hence,

πθ(x¯|c) πref(x¯|c) −

log

1 β

r(x0,c) = f(c). (46)

It suffices to show that

πθ(x¯|c) = πref(x¯|c)exp

1 β

r(x0,c) exp(f(c)). (47)

exp(f(c)) =

1 Z(c)

, ∀c. (48)

This follows from the fact that the probability density function πθ(x¯|c) must satisfy:

1 = πθ(x¯|c)dx¯ (49)

1 β

r(x0,c) exp(f(c))dx¯ (50)

= πref(x¯|c)exp

1 β

r(x0,c) dx¯ (51) = exp(f(c))Z(c). (52)

= exp(f(c)) πref(x¯|c)exp

| |
|---|

###### B. Instability of DDPO in Large-Scale Reward Finetuning

- Figure 9. Analysis of the instability of DDPO in large-scale training. We plot the training curves of PRDP and DDPO on the large-scale Human Preference Dataset v2 (Left) and the small-scale Common Animals (Right). PRDP outperforms DDPO in the small-scale setting, and maintains stability in the large-scale setting where DDPO fails. Our ablation study suggests that the per-prompt reward normalization in DDPO is key to its stability, and the inability to perform such normalization in the large-scale setting likely causes its failure.

Figure 9 shows the training curve of PRDP and DDPO [4], where the reward model is HPSv2 [53]. From Figure 9 (Left), we observe that when trained on the large-scale Human Preference Dataset v2 (HPD v2) [53], DDPO fails to stably optimize the reward. We conjecture that this is because the per-prompt reward normalization is rarely enabled in the large-scale setting, since each prompt can only be seen a few times. Specifically, in each epoch, DDPO randomly samples 512 prompts, so on average, each prompt can be seen 512×1000/100K ≈ 5 times. This is insufficient to obtain a good estimate of the per-prompt expected reward. In this case, DDPO will compute a prompt-agnostic expected reward, by averaging the rewards across all 512 prompts. To verify that such prompt-agnostic reward normalization causes training instability, we conduct an ablation study of DDPO in our small-scale setting with 45 training prompts. As shown in Figure 9 (Right), DDPO without per-prompt reward normalization is unstable even in the small-scale setting, suggesting that the inability to perform per-prompt reward normalization can be a limiting factor in scaling DDPO to large prompt datasets. In contrast to DDPO, PRDP can steadily improve the reward score and maintain stability in both small-scale and large-scale settings.

###### C. Effect of KL Regularization

cat goat tiger butterﬂy frog

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

Diffusion DDPO

HPSv2: PickScore:

0.2836 0.2155 5.49

Stable

Aesthetic:

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

HPSv2: PickScore:

0.2629 0.1989 9.52

Aesthetic:

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

- (beta=0.1) PRDP

- (beta=1) PRDP

HPSv2: PickScore:

0.2647 0.2038 9.70

PRDP

Aesthetic:

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

HPSv2: PickScore:

0.2794 0.2199 8.62

Aesthetic:

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

(beta=10)

HPSv2: PickScore:

0.2841 0.2212 7.45

Aesthetic:

- Figure 10. Effect of KL regularization on optimizing aesthetic score. DDPO and PRDP are finetuned from Stable Diffusion v1.4 on 45 prompts of common animal names. Evaluation is performed on the same set of prompts. In addition to aesthetic score, we report HPSv2 and PickScore which reflect text-image alignment but are not used during training. Samples within each column are generated from the prompt shown on top, using the same random seed. PRDP with a large KL weight β can alleviate the reward over-optimization problem encountered by DDPO, significantly improving the aesthetic quality over Stable Diffusion while maintaining text-image alignment.

In contrast to DDPO [4] which only cares about maximizing the reward, PRDP is formulated with a KL regularization, allowing us to alleviate the problem of reward over-optimization by increasing the KL weight β. We demonstrate the effect of KL regularization in Figure 10. Here, the reward used for training is the aesthetic score given by the LAION aesthetic predictor. It only takes images as input, and therefore ignores the text-image alignment. We finetune DDPO and PRDP from Stable Diffusion v1.4 [37] for 250 epochs on 45 training prompts of common animal names as used in DDPO, with 512 reward queries in each epoch. For evaluation, we additionally use HPSv2 [53] and PickScore [22] that reflect text-image alignment. The reported reward scores are averaged over 64 random samples per training prompt, using the same random seed for Stable Diffusion v1.4, DDPO, and PRDP.

We observe that DDPO, without KL regularization, is prone to reward over-optimization. It ignores the text prompt and generates similar images for all prompts. PRDP with a small KL weight (e.g., β = 0.1) has the same problem, but achieves higher reward scores than DDPO, showing a better reward maximization capability. As the KL weight increases, PRDP is able to better preserve the text-image alignment, indicated by the increase in HPSv2 and PickScore. With β = 10, PRDP significantly improves the aesthetic score over Stable Diffusion v1.4 without sacrificing text-image alignment.

###### D. Large-Scale Multi-Reward Finetuning

Table 3. Reward score comparison on unseen prompts. We use a weighted combination of rewards: PickScore = 10, HPSv2 = 2, Aesthetic = 0.05. PRDP is finetuned from Stable Diffusion v1.4 on the training set prompts of Pick-a-Pic v1 dataset.

Pick-a-Pic v1 Test Set

HPD v2 Animation

HPD v2 Concept Art

HPD v2 Painting

HPD v2 Photo

SD v1.4 2.888 2.927 2.877 2.883 2.984 PRDP 3.208 3.296 3.264 3.274 3.214

In this section, we provide additional results for our large-scale multi-reward finetuning experiment. Following DRaFT [6], we use a weighted combination of rewards: PickScore = 10, HPSv2 = 2, Aesthetic = 0.05. We finetune Stable Diffusion v1.4 [37] on the training set prompts of Pick-a-Pic v1 dataset [22]. We evaluate our finetuned model on a variety of unseen prompts, including 500 prompts from the Pick-a-Pic v1 test set, and 800 prompts from each of the four benchmark categories of the Human Preference Dataset v2 (HPD v2) [53], namely animation, concept art, painting, and photo. Table 3 reports the reward scores before and after finetuning. The reward scores are averaged over 64 random samples per prompt, using the same random seed for Stable Diffusion v1.4 and PRDP. We further show generation samples for each test prompt set in Figures 11 to 15. As can be seen, PRDP significantly improves generation quality across all five prompt sets.

###### E. Hyperparameters

Table 4. PRDP training hyperparameters.

Small-Scale Finetuning

Large-Scale Finetuning

Large-Scale Multi-Reward Finetuning

Name Symbol

Training epochs E 100 1000 1000 Gradient updates per epoch K 10 1 1 Prompts per epoch N 32 64 64 Images per prompt B 16 8 8 KL weight β 3×10−5 3×10−6 3×10−5 DDPM steps T 50 50 50 Stepwise clipping range ϵ 1×10−6 1×10−4 1×10−4 Classifier-free guidance scale — 5.0 5.0 5.0 Optimizer — AdamW AdamW AdamW Gradient clipping — 1.0 1.0 1.0 Learning rate — 1×10−5 7×10−6 1×10−5 Weight decay — 1×10−4 1×10−4 1×10−4

###### F. Effect of Clipping

Table 5. Effect of clipping on training stability.

w/o Clipping w/ Clipping DDPO

Small scale: Unstable Large scale: Unstable

Small scale: Stable Large scale: Unstable

Small scale: Unstable Large scale: Unstable

Small scale: Stable Large scale: Stable

PRDP

Table 5 summarizes the effect of clipping on the training stability of both DDPO [4] and PRDP. For DDPO, we use PPObased clipping [42], while for PRDP, we use the proximal updates described in Section 3.3. We observe that clipping is key to stability of small-scale training, whereas using the PRDP objective and clipping are both indispensable for achieving stability in large-scale training.

###### G. Jax Implementation of PRDP Loss

- 1 import jax

- 2 import jax.numpy as jnp

- 3

- 4

- 5 def prdp_loss(

- 6 log_probs: jax.Array, # (B, T)

- 7 log_probs_old: jax.Array, # (B, T)

- 8 log_probs_ref: jax.Array, # (B, T)

- 9 rewards: jax.Array, # (B,)

- 10 clip_range: float,

- 11 kl_weight: float,

- 12 ) -> jax.Array:

- 13 """Computes PRDP loss for a batch of denoising trajectories with the same text prompt.

- 14

- 15 Args:

- 16 log_probs: Log probs of the denoising trajectories under pi_theta.

- 17 log_probs_old: Log probs of the denoising trajectories under pi_theta_old.

- 18 log_probs_ref: Log probs of the denoising trajectories under pi_ref.

- 19 rewards: Rewards of the generated clean images.

- 20 clip_range: Stepwise clipping range (epsilon).

- 21 kl_weight: KL weight (beta).

- 22

- 23 Returns:

- 24 loss: The PRDP loss.

- 25 """

- 26 log_ratios = log_probs - log_probs_ref

- 27 log_ratios_old = log_probs_old - log_probs_ref

- 28 clipped_log_ratios = jnp.clip(

- 29 log_ratios, log_ratios_old - clip_range, log_ratios_old + clip_range

- 30 )

- 31

- 32 log_ratios = jnp.mean(log_ratios, axis=-1)

- 33 clipped_log_ratios = jnp.mean(clipped_log_ratios, axis=-1)

- 34

- 35 log_ratio_diffs = log_ratios[:, None] - log_ratios

- 36 clipped_log_ratio_diffs = clipped_log_ratios[:, None] - clipped_log_ratios

- 37 reward_diffs = rewards[:, None] - rewards

- 38

- 39 mse_loss = (log_ratio_diffs - reward_diffs / kl_weight) ** 2

- 40 clipped_mse_loss = (clipped_log_ratio_diffs - reward_diffs / kl_weight) ** 2

- 41 loss = jnp.maximum(mse_loss, clipped_mse_loss)

- 42 loss = jnp.mean(loss, where=reward_diffs > 0)

- 43

- 44 return loss

Cute and adorable ferret wizard, wearing coat and suit, steampunk, lantern, anthromorphic, Jean paptiste monge, oil painting

A portrait of a bear wearing a suit in the style of a Baroque painting

cinematic still of highly reﬂective stainless steel train in the desert, at sunset

rural house with a garden and a swimming pool

Photo of a cat eating a burger like a person

An evil villain holding a mini earth

A cat in a space suit walking on the moon

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

###### StableDiffusionv1.4PRDP

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

a landscape with a river running down the middle in a forest with a sunset behind distant mountains

cinematic still of an adorable walking robot in the desert, at sunset

cubic building on clouds of colorful trees

Harry potter as a cat, pixar style, octane render, HD, high-detail

monkey climbing a skyscraper

cozy house to live in a mountain

Horses running on the Great Wall at sunset

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

###### StableDiffusionv1.4PRDP

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

Title image: A heartwarming illustration of a cute lion cub named Ladi and a very small koala named Carlo sitting together in the jungle, with an adventurous landscape unfolding in the background. Disney style

futuristic grand fort made out of white marble and extremely intricate carvings across the structure on a martian mountain with fountains and greenery all around

a close up of a cat wearing a pikachu hat, reddit, gif, real life charmander, very aesthetic!!!!!!, soft!!

wonderful image of a landscape and a medieval tower

A war weary hamster soldier

An abandoned Segway in the forest

A cute blue cat.

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

StableDiffusionv1.4PRDP

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

- Figure 11. Generation samples on unseen prompts from the Pick-a-Pic v1 test set. PRDP is finetuned from Stable Diffusion v1.4 on the training set prompts of Pick-a-Pic v1 dataset, using a weighted combination of rewards: PickScore = 10, HPSv2 = 2, Aesthetic

A portrait of a silver and white brindle persian cat dressed as a renaissance queen, standing atop a skyscraper overlooking a city.

A digital painting of an anthropomorphic corgi lifting weights in a dim gym with intricate details and a dynamic pose.

An anthropomorphic cat wearing sunglasses and a leather jacket rides a Harley Davidson in Arizona.

A cute anthropomorphic fox knight wearing a cape and crown in pale blue armor.

A toad baby sitting in a rose blossom, depicted in a humorous and detailed illustration.

A fox wearing a yellow dress.

A chibi frog character surﬁng at the beach.

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

###### StableDiffusionv1.4PRDP

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

Digital art of a female marten animal cartoon character wearing jewelry with a blonde hairstyle.

A bear in an astronaut suit sits on a rock on Mars surrounded by ﬂowers under a starry sky.

A cute little anthropomorphic bear knight wearing a cape and crown in pale blue armor.

A colorful cartoon tent in a bazaar with a borderlands-inspired aesthetic.

A knitted Capybara wearing sunglasses sips a Mojito at the beach during sunset.

A portrait of a cat wearing a samurai helmet.

A pikachu in a forest illustration.

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

###### StableDiffusionv1.4PRDP

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

A Fortnite poster featuring chibi kittens wearing cyberpunk headphones and shades, with anime-stylized art by Takeshi Murakami.

A cartoon satanic priest depicted as an anthropomorphic lamb in a highly detailed 3D render.

The image is a humorous illustration of a furry alien chick nesting in a ﬂoral cup.

A landscape with a Maya-style building and Winnie the Pooh on grass.

A ﬂuffy chick is nested in an antique coffee cup in a humorous illustration.

A corgi dressed as a bee costume.

A blue bear wearing cowboy boots.

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

StableDiffusionv1.4PRDP

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

- Figure 12. Generation samples on unseen prompts from the HPD v2 animation benchmark. PRDP is finetuned from Stable Diffusion v1.4 on the training set prompts of Pick-a-Pic v1 dataset, using a weighted combination of rewards: PickScore = 10, HPSv2 = 2, Aesthetic

= 0.05. For each prompt, the generation sample from Stable Diffusion v1.4 and PRDP use the same random seed.

An ancient Japanese temple located in a forest near a river, with dramatic lighting and a singular building centered in the image.

A giant burning pineapple illuminates the forest and mountain backdrop in this cinematic concept art for a video game.

An orange cat wearing magical ornate armor with a backdrop of Art Nouveau-inspired design.

A fox wearing a Maﬁa Hat, red Tie and white shirt in fantasy concept art.

Digital art of a cherry tree overlooking a valley with a waterfall at sunset.

Exterior image of a small magic items and curios shop in a busy fantasy city.

A landscape featuring a lone magic the gathering-style building.

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

###### StableDiffusionv1.4PRDP

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

The image depicts a forest with realistic gnomes and mushrooms on the ground, with warm lighting shining through the trees.

The image depicts a concept art of Schrodinger's cat in a box with an abstract background of waves and particles in a dynamic composition.

The image is a wooden sculpture of a cute robot with cat ears, displayed in a contemporary art gallery.

The Kremlin ruins are engulfed in ﬂames in a digital art illustration with a fantastical style and Morandi color scheme.

A concept art digital CG painting of a place in Bali, trending on ArtStation and created using Unreal Engine.

A path winding through a forest depicted in digital art.

An image of a fantastical city ﬂoating in the clouds.

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

###### StableDiffusionv1.4PRDP

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

A Halloween-themed TV show room with a big screen on the wall, designed by Disney Concept Artists with blunt borders and following the rule of thirds.

A futuristic modern house on a ﬂoating rock island surrounded by waterfalls, moons, and stars on an alien planet.

The image depicts an otherworldly landscape with a waterfall, trees, mountains, and lush greenery, under dramatic lighting.

Minimalistic surreal interior with arches, glass 3D objects, and abstract pools around.

Digital art featuring small white butterﬂies amidst a starry darkness.

This is a 3D isometric illustration with studio lighting.

A massive frog robot wreaking havoc on a city.

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

StableDiffusionv1.4PRDP

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

###### Figure 13. Generation samples on unseen prompts from the HPD v2 concept art benchmark. PRDP is finetuned from Stable Diffusion

A painting of a Persian cat dressed as a Renaissance king, standing on a skyscraper overlooking a city.

The image features a castle surrounded by a dreamy garden with roses and a cloudy sky in the background.

A digital painting of a fantasy kitchen environment with elements of cartoons, comics, and manga.

Colorful illustration of a forest tunnel illuminated by sunlight and ﬁlled with wildﬂowers.

A ﬂuffy owl sits atop a stack of antique books in a detailed and moody illustration.

A detailed painting of a futuristic spaceship with ornamental features.

The image features a surreal fox and skulls in highly detailed, liquid oilpaint style.

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

###### StableDiffusionv1.4PRDP

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

A night scene of a lavender ﬁeld with a town and church in the background, reminiscent of Vincent van Gogh's style.

A digital painting of a magical ritual location with volumetric lighting and elements from various artworks and games.

A serene meadow with a tree, river, bridge, and mountains in the background under a slightly overcast sunrise sky.

An oil painting of a vintage rally car, including a yellow Porsche with smoke and dirt from drifting.

A brownstone building located in a forest setting, painted by Eytan Zana.

A landscape featuring a unique digital painting-style building.

A watercolor painting of a galaxy in a jar.

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

###### StableDiffusionv1.4PRDP

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

A painting of a girl standing on a mountain looking out at an approaching storm over the ocean, with wind blowing and ocean mist, surrounded by lightning.

A solar eclipse is depicted over a ﬁeld of grass and ﬂowers with a small forest in the distance, as a matte painting on Art Station by Simon Stalenhag.

The image features an ancient Chinese landscape with a mountain, waterfalls, willow trees, and arch bridges set against a blue background.

A digital painting of a blue-skinned wizard with intricate and elegant details, created by multiple artists and posted on Artstation.

A train crosses a trestle bridge in the mountains in an optimistic and vibrant illustration.

A surreal cat with a smile and intricate details.

A landscape with an art nouveau building.

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

StableDiffusionv1.4PRDP

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

###### Figure 14. Generation samples on unseen prompts from the HPD v2 painting benchmark. PRDP is finetuned from Stable Diffusion

A small elephant toy sitting inside of a wooden car.

A wooden outhouse sitting in the grass near trees.

a man on a motorcycle that is in some grass

Two kittens curled up in a white sheet that looks soft.

A man standing in front of a bunch of doughnuts.

A wreath with a red bow on it hanging on a white door.

a vase with a ﬂower growing very well

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

###### StableDiffusionv1.4PRDP

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

A dim lit room consisting of many objects put together.

A motorcycle parked on a stone cobble road, in the sun.

A car sitting in the middle of the grass in the rain.

Sun shining through the blinds into a white bathroom.

a black cat that is sitting in a sink

A TV sitting on top of a wooden stand.

a couple of horse that are eating some grass

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

###### StableDiffusionv1.4PRDP

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

a black and white photo with a vase and ﬂower coming out of it

Ornate archway inset with matching ﬁreplace in room.

A TV sitting on top of a counter inside of a store.

A man wearing a black neck tie and glasses.

The motorcycle is tilting as he turns through a cave.

A table topped with lots of food and drinks.

a cat laying on the ﬂoor of a kitchen

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

StableDiffusionv1.4PRDP

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

[Figure 322]

[Figure 323]

- Figure 15. Generation samples on unseen prompts from the HPD v2 photo benchmark. PRDP is finetuned from Stable Diffusion v1.4 on the training set prompts of Pick-a-Pic v1 dataset, using a weighted combination of rewards: PickScore = 10, HPSv2 = 2, Aesthetic

= 0.05. For each prompt, the generation sample from Stable Diffusion v1.4 and PRDP use the same random seed.

