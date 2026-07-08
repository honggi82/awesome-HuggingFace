## Large-scale Reinforcement Learning for Diffusion Models

Yinan Zhang Pinterest ATG

Eric Tzeng Pinterest ATG

Yilun Du MIT CSAIL

Dmitry Kislyuk Pinterest ATG

# arXiv:2401.12244v1[cs.CV]20Jan2024

### Abstract

Text-to-image diffusion models are a class of deep generative models that have demonstrated an impressive capacity for high-quality image generation. However, these models are susceptible to implicit biases that arise from web-scale text-image training pairs and may inaccurately model aspects of images we care about. This can result in suboptimal samples, model bias, and images that do not align with human ethics and preferences. In this paper, we present an effective scalable algorithm to improve diffusion models using Reinforcement Learning (RL) across a diverse set of reward functions, such as human preference, compositionality, and fairness over millions of images. We illustrate how our approach substantially outperforms existing methods for aligning diffusion models with human preferences. We further illustrate how this substantially improves pretrained Stable Diffusion (SD) models, generating samples that are preferred by humans 80.3% of the time over those from the base SD model while simultaneously improving both the composition and diversity of generated samples. The project’s website can be found at https://pinterest.github.io/atg-research/rl-diffusion/.

### 1. Introduction

Diffusion probabilistic models [24, 39, 43] have revolutionized generative modeling, particularly for producing creative and photorealistic imagery when combined with pre-trained text encoders [37, 38]. However, the resulting image quality is highly dependent on the distribution of the pre-training dataset, which typically consists of web-scale text-image pairs. Although pre-training on massive weakly supervised tasks of this form is effective in exposing the text-to-image model to a wide range of prompts, downstream applications often observe weaknesses around the following properties:

- • Fidelity and controllability [10, 22, 25]: failing to accurately depict the semantics of the text prompts (e.g. incorrect composition and relationships between objects)
- • Human aesthetic mismatch [50, 51]: producing outputs that humans do not perceive to be aesthetically pleasing

• Bias and stereotypes [4, 32, 42]: presenting or exaggerating societal bias and stereotypes

To address these challenges, several works have explored classic fine-tuning techniques for pre-trained diffusion models with curated data, either to improve the aesthetic quality of the model outputs with human-selected high-quality images [14], or to eliminate existing biases in the model with synthetic dataset augmentation [17]. Another approach, which bypasses the labor-intensive dataset curation, involves intervention in the sampling process to achieve controllability, by utilizing auxiliary input [20, 28, 29] or refining the intermediate representations [7, 8, 19]. However, this form of inference-time guidance results in an increase in the sampling time without improving the inherent capability of the model. A recent direction, motivated by the success of reinforcement learning from human feedback (RLHF) in the language domain [2, 35, 36], proposes [13, 51] fine-tuning diffusion models through full-sample gradient backpropogation on human preference reward models, though these approaches are memory intensive and only work for differentiable reward functions. Finally, RL-based optimization [5, 18] has enabled fine-tuning with arbitrary objective functions, but these methods have so far been limited in scope by focusing on a small set of prompts in a narrow domain, and lack the scale to improve model performance generally.

In this paper, we propose a generic RL-based framework for fine-tuning diffusion models, which works at scale across millions of prompts and with an arbitrary combination of objective functions. Our contributions are as follows:

- • We present an effective large-scale RL training algorithm for diffusion models which allows training over millions of prompts across a diverse set of tasks.
- • We propose a distribution-based reward function for RL fine-tuning to improve the output diversity.
- • We demonstrate how to perform effective multi-objective RL-training and illustrate how we can improve a base model across all objectives, which can include human aesthetic preference, fairness, and object composition.
- • We conduct extensive experiments and analysis studies comparing our approach with existing reward optimization methods across a suite of tasks.

Human Preference

|[Figure 1]|
|---|

|[Figure 2]|
|---|

|[Figure 3]|
|---|

|[Figure 4]|
|---|

|[Figure 5]|
|---|

|[Figure 6]|
|---|

###### A portrait of an anthropomorphic wolf wearing a black doublet, furry fursona, Victorian era masterwork, by Samuel Luke Fildes

Fairness and Diversity

|[Figure 7]<br><br>[Figure 8]|
|---|

|[Figure 9]<br><br>[Figure 10]|
|---|

|[Figure 11]<br><br>[Figure 12]|
|---|

A portrait of a doctor, trending on Pinterest, art, highly detailed, illustration, sharp focus, intricate, smooth, elegant…

Compositionality

|[Figure 13]|
|---|

|[Figure 14]|
|---|

|[Figure 15]|
|---|

|[Figure 16]|
|---|

|[Figure 17]|
|---|

|[Figure 18]|
|---|

An avocado near an apple, trending on Pinterest, art, highly detailed, illustration, sharp focus, intricate, smooth, elegant…

- Figure 1. Sample Evolution over Reinforcement Training. We perform multi-task RL on text-to-image diffusion models, improving the model’s compositional capacity and alignment with human preference while mitigating its bias and stereotypes. Here we show the progression of samples over training across each objective, with the leftmost columns showing results from the base SDv2 model.

### 2. Related Work

Inclusive Text-to-Image Generation. Text-to-image generative models perpetuate and even amplify the societal biases present in the massive pretraining datasets of uncurated image-text pairs [6, 9, 10, 54]. Existing work addresses this by either using balanced synthetic data [42], with textual guidance during inference [21] or with reference images of a particular attribute [53]. Different from prior work, our method does not require synthetic data collection or inference-time intervention.

Reward Fine-tuning for Diffusion Models. Existing reward fine-tuning methods for diffusion models can be classified into three categories: either supervised with rewardweighted data [15, 26, 50], optimized through gradientbackpropogation on the reward function [13, 51] or through reinforcement learning [5, 18]. Our work builds on work training diffusion models with reinforcement learning, but while past work has focused on simple settings (DPOK uses a training set of 1 prompt per model, and DDPO using simple set of 45 common animals and 3 activities), we illustrate how we can use reinforcement learning training across the scale of millions of prompts and different objectives.

### 3. Method

In this section, we describe our approach for applying largescale RL training to diffusion models. Our goal is to finetune the parameters θ of an existing diffusion model to maximize the reward signal r of the generated images from the sampling process:

Compositional Text-to-image Generation. Despite their remarkable capacity, current state-of-the-art text-to-image models still struggle to generate images that faithfully align with the semantics of the text prompts due to their limited compositional capabilities [10, 22, 25]. Existing work addresses this by either modifying the inference procedure [7, 16, 19, 19, 30] or by using auxiliary conditioning inputs such as bounding boxes [8, 28] or spatial layouts [20, 29, 49]. Our method instead focus on improving the fidelity of existing SD models without using additional layout guidance.

0∼pθ(x0|c)[r(x0,c)], (1)

J(θ) = Ec∼p(c),x

where p(c) is the context distribution, pθ(x0|c) is the sample distribution, and r(x0,c) is the reward function that is applied to the final sample image.

#### 3.1. Policy Gradient with Multi-step MDP

Following Black et al. [5], we reframe the iterative denoising procedure of diffusion models as a multi-step Markov decision process (MDP), where the policy, action, state and reward at each timestep t are defined as follows:

π(at|st) ≜ pθ(xt−1|xt,c) (2) at ≜ xt−1 (3) st ≜ (c,t,xt) (4)

r(x0,c) if t = 0 0 otherwise

R(st,at) ≜

(5)

We treat the reverse sampling process pθ(xt−1|xt,c) of the diffusion model as the policy. Starting from a sampled initial state xT, the policy’s action at any timestep t is the update that produces the sample for the next timestep xt−1. The reward is defined as r(x0,c) at the final timestep, and 0 otherwise.

The policy gradient estimates can be made using the likelihood ratio method (also known as REINFORCE) [33, 48]:

∇θJ = E r(x0,c)

T

∇θ log pθ(xt−1|xt,c) . (6)

t=0

We also apply importance sampling to enable collecting samples from the old policy for improved training efficiency, and incorporate a clipped trust region to ensure that the new policy does not deviate too much from the old policy [41]. The final clipped surrogate objective function can be written as:

J(θ) = E Tt=0 min w(θ,θold)Aˆ(x0,c),g(ϵ,Aˆ(x0,c)) (7) where

pθ(xt−1|xt,c) pθ

w(θ,θold) =

,

(xt−1|xt,c)

old

(1 + ϵ)A if A ≥ 0 (1 − ϵ)A if A < 0

g(ϵ,A) =

.

Here ϵ is the hyper-parameter that determines the clip interval, and Aˆ(x0,c) is the estimated advantage for the samples. To further prevent over-optimization of the reward function, we also incorporate the original diffusion model objective as part of the loss function. Our full training objective is thus

L(θ) = J(θ) + βLpre(θ), (8) where

Lpre(θ) = Eε(x),ϵ∼N(0,1),t ∥ϵ − ϵθ(zt,t)∥22 . (9)

One additional detail is that reward values are typically normalized to zero mean and unit variance during gradient updates to increase training stability. In policy-based RL, a

general approach is to subtract a baseline state value function from the reward to obtain the advantage function [45]

r(x0,c) − µr σr2 + ϵ

Aˆ(x0,c) =

. (10)

In the original implementation of DDPO, Black et al. normalize the rewards on a per-context basis by keeping track of a running mean and standard deviation for each prompt independently [5]. However, this approach remains impractical if the training set size is unbounded or unfixed.

In contrast to the limited size of their training prompts (up to 398 only), our large-scale fine-tuning experiments involve millions of training prompts. We instead normalize the rewards on a per-batch basis using the mean and variance of each training batch.

#### 3.2. Distribution-based Reward Functions

In the previously outlined formulation of the diffusion MDP, each generation is considered independently, and thus rewards incurred by generated samples are independent of each other. This formulation is a natural fit for reward functions that only care about the contents of a single image, such as image quality or text-image alignment. However, sometimes what we care about is not the contents of any particular image, but instead the output distribution of the diffusion model as a whole. For example, if our goal is to ensure our model generates diverse outputs, considering a single generation in isolation is insufficient—we must consider the set of all outputs in order to understand these distributional properties of our model.

To this end, we also investigate the use of distributionlevel reward functions for reinforcement learning with diffusion models. However, it is intractable to construct the true generative distribution. Thus, we instead approximate the reward by computing it using empirical samples across minibatches during the reinforcement learning process. During training, the attained reward is computed on each minibatch, and the minibatch reward is then backpropagated across the samples to perform model updates. In Section 4.2 we validate this approach by learning via a distribution-level reward function that optimizes for fairness and diversity in generated samples.

#### 3.3. Multi-task Joint Training

We also perform multi-task joint training to optimize a single model for a diverse set of objectives simultaneously. As detailed in the next section, we incorporate the reward functions from human preference, skintone diversity, object composition and perform joint-optimization all at once. Since each task involves a different distribution of training prompts, in every training iteration, we sample multiple prompts from all the tasks and run the sampling process independently. Each reward model is applied to the correspond-

###### Human Preference Rate

###### Number of Positive Votes

| |82.2 82.3 81.4<br><br>77.6<br><br>81.1<br><br>17.8 17.7 18.6<br><br>22.4<br><br>18.9| | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |

100%

80%

70%

80%

Number of Votes

60%

- 0/5

- 1/5

- 2/5

- 3/5

- 4/5

- 5/5

50%

60%

40%

40%

30%

20%

20%

10%

0%

0%

Ours SD2 Ours RAFT Ours ReFL Ours Rewardweighted

Ours DRaFT

Ours SD2 Ours RAFT Ours ReFL Ours Rewardweighted

Ours DRaFT

(a) Human Eval Results on DiffusionDB [47] dataset

###### Human Preference Rate

###### Number of Positive Votes

| |74.7<br><br>72.0<br><br>64.6<br><br>70.8<br><br>63.3<br><br>25.3<br><br>28.0<br><br>35.4<br><br>29.2<br><br>36.7| | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |

100%

70%

60%

80%

Number of Votes

- 0/5

- 1/5

- 2/5

- 3/5

- 4/5

- 5/5

50%

60%

40%

30%

40%

20%

20%

10%

0%

0%

Ours SD2 Ours RAFT Ours ReFL Ours Rewardweighted

Ours DRaFT

Ours SD2 Ours RAFT Ours ReFL Ours Rewardweighted

Ours DRaFT

(b) Human Eval Results on PartiPrompts [52] dataset

- Figure 2. Human Preference Evaluation of Generations. Human evaluation results on 400 text prompts (300 randomly sampled from DiffusionDB dataset and 100 randomly sampled from PartiPrompts dataset). We perform head-to-head comparisons between images generated by our model and each of the baseline models, using the same text prompt and random seed for each generation. Then, human raters indicate which one is better in terms of image quality and image-text alignment. Each query is evaluated by 5 independent human raters, and we report each model’s preference rate based on the number of positive votes it received.

ing sample image with the prompt. Then the gradient step from equation 7 is executed for each task sequentially. We outline the training framework in Algorithm 1 with hyperparameters available in Appendix A.

### 4. Reward Functions and Experiments

To validate our method across a wide variety of settings, we perform experiments on three separate reward functions: human preference, image composition, and diversity and fairness. We begin with an introduction of the different reward functions we applied our method to.

To optimize diffusion models to adhere to human preferences, we use an open-source reward model, ImageReward (IR), trained on a large number of human preference pairs [51]. ImageReward takes a pair consisting of a text caption and a generated sample, then outputs a human preference score, which is then used as the reward during training:

###### r(x0,c) = IR(x0,c). (11)

Our results with this human preference reward function are detailed in Section 4.1.

In order to encourage fairness and diversity across the samples generated by our model, following previous

Algorithm 1 Multi-reward diffusion policy optimization

Input: A set of reward models and the training prompt distribution S = {(ri,pi(c))}, pretrained diffusion model ppre, current diffusion model pθ, pretraining dataset Dpre Initialize pθ = ppre

while θ not converged do

pθ

= pθ

old

for each training task (r,p(c)) ∈ S do Sample a prompt c ∼ p(c) Sample generated images x0:T ∼ pθ(x0:T|c) Sample training timesteps t for each selected timestep t do

Take gradient step ∇θJ(θ) (Eq. 7)

end for end for Sample a pretraining data pair (txt,img) ∈ Dpre Take gradient step ∇θLpre(θ) (Eq. 9)

end while Output: Fine-tuned diffusion model pθ

work [11, 12, 46], we leverage statistical parity, a metric commonly adopted for measuring biases in models, as a distribution-level reward function for our fine-tuning experiments. Given the generated distribution Pˆ and a classifier

h : x −→ A that identifies a spurious attribute, we measure the L2 norm between the empirical and uniform distributions:

(Ex∼Pˆ h(x)=a − 1/|A|)2 (12)

a∈A

The reward attained by the model is then simply the negation of the statistical parity, so as to encourage the model to produce diverse samples. As explained in Section 3.2, it is intractable to compute the reward over the full output distribution of the model, so we compute the reward over individual minibatches. We present the results for this experiment in Section 4.2.

To improve the compositional skills of diffusion models, we devise a new reward function that uses an auxiliary object detector. We construct a set of training prompts, each containing multiple different objects, and use an object detection model on the image to predict the confidence score for each object class. The reward score is then defined as the average confidence score of all the objects:

1 |o| o∈c

d(o,x0), (13)

r(x0,c) =

where d(o,x0) is the detection confidence score for the object class o given input image x0. Our results on compositionality are detailed in Section 4.3.

Finally, we also experiment with jointly optimizing over all three previously described reward functions, to train a model that satisfies all three criteria simultaneously. We present the results of our joint optimization in Section 4.4. For all our fine-tuning experiments, we use SDv2 [39] as our base model. The output resolution is 512x512, which we consider as a good tradeoff between compute efficiency and image quality.

#### 4.1. Learning from human preference

To fine-tune a diffusion model with human preferences, we use ImageReward [51], which was trained on large-scale human assessments of text-image pairs. In total, the authors collected 137k pairs of expert judgments on images generated from real-world user prompts from the DiffusionDB dataset [47]. Compared to other existing metrics such as CLIP [37], BLIP [27], or Aesthetic score [40], ImageReward is better aligned with human judgments, making it better suited as a reward function.

We use a training set of 1.5 million unique real user prompts from DiffusionDB, among which 2,000 prompts were split for testing. We use 128 A100 GPUs (80GB) for all experiments, including the baselines. Experimental details, hyperparameters, and additional results are provided in Appendix A.

Baseline Comparison. Prior reward fine-tuning methods for diffusion models mainly fall under three categories: rewardbased loss reweighting [26], dataset augmentation [15],

DiffusionDB PartiPrompts IR* Aesthetic IR* Aesthetic

Model

Stable v1.5 0.082 5.907 0.256 5.373 Stable v2 0.170 5.783 0.414 5.269 ReFL 1.290 5.845 0.832 5.402 RAFT 0.338 5.881 0.504 5.413 DRaFT 0.818 5.645 0.632 5.279 Reward-weighted 0.438 5.821 0.624 5.363

Ours 0.845 5.918 0.731 5.477

- Table 1. Quantitative Results. ImageReward scores and Aesthetic scores from the original SDv2 model, baseline methods, and our model. We report the average ImageReward and Aesthetic scores for samples generated using prompts from both the DiffusionDB [47] dataset and the PartiPrompts [52] dataset.

and backpropagation through the reward model [13, 51]. We compare against a variety of baseline methods, including ReFL [51], RAFT [15], DRaFT [13] and Rewardweighted [26], covering the three different methodologies. We reimplement all methods and fine-tune them on SDv2 using the same training set of 1.5M prompts until convergence.

We show the qualitative and quantitative results of all baseline methods in Figure 3 and Table 1. We also provide training curves in Appendix E and note that, except for RAFT which diverged, all online-learning methods exhibit steadily increasing sample rewards during training, eventually saturating at some maximum level, at which point we consider the models converged. In contrast to the common belief that RL training is inefficient and slow to converge, our approach converges in as few as ∼1,000 steps, compared to DRaFT, the gradient-based reward optimization approach which takes ∼4,000 steps to converge while only being able to optimize for differentiable rewards. We provide a comprehensive comparison of all the reward optimization methods in Table 2.

Training Generalizable to Human Preference Model Time All Rewards Rank

RAFT 5.5h (diverged) 5 ReFL 6.9h 4 DRaFT 8.4h 3 Reward-weighted 33.8h∗ 2 Ours 12.1h 1

- Table 2. Performance Comparison. Comparison of different reward optimization methods. Training time indicates the time for each method to converge. ∗For Reward-weighted, training time includes constructing the training dataset from the base model.

For RAFT, we found the model diverges as the number of training iterations increases, similar to the finding from Xu et al. [51]. Since RAFT uses the model-generated images with the highest rewards for fine-tuning the model, it is constrained by the diversity of the latest model’s generation and thus prone to overfitting. The reward-weighted method uses a similar idea of augmenting the training data using model-generated images and weighting the training loss by

Stable V2 ReFL

RAFT DRaFT Reward-weighted Ours

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

Old house at the end of a forest road in the rain, creepy ambiance, high focus, highly detailed

Head and shoulders portrait of a female knight, quechua!, lorica segmentata, cuirass, tonalist, symbolist, realism, …

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

A hyper realistic cat warrior, ultra detailed, magic the gathering art, digital art, cinematic, studio lighting, …

- Figure 3. Qualitative comparison of our approach and other reward fine-tuning methods on real-user prompts. All images are generated using the same random seeds.

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

SDv2

Reward=0.99

DRaFT

Reward=1.33

ReFL (Iteration 200) Reward=1.17

ReFL (Iteration 600) Reward=1.75

ReFL (iteration 1000) Reward=1.90

- Figure 4. Reward Hacking. Finetuning methods such as DRaFT and different iterations of ReFL fine-tuned models often over-optimize reward functions and generate over-detailed images with high-frequency noise.

the reward values, but all the images are generated from the original model (in contrast to RAFT’s online generation using the latest model) and thus is less prone to overfitting.

Evaluating generalization. Next, we evaluate our trained model’s ability to generalize to an out-of-domain test set, PartiPrompts [52]. PartiPrompts is a comprehensive benchmark for text-to-image models, with over 1,600 challenging prompts across a variety of categories. We report the ImageReward and Aesthetic scores in Table 1, along with human evaluation results in Figure 2. When compared against each baseline model, our approach achieves the highest Aesthetic score and human preference rate on both sets.

We also achieve the second highest result on ImageReward, but note that this metric alone is not a robust indicator of performance, since the model was directly trained against it. Reward hacking is a commonly observed phenomenon in which models optimizing for a single reward function often overoptimize for this single metric at the cost of overall performance. We believe the high ImageReward scores achieved by ReFL are a result of this, and show example generations in Figure 4. The reward hacking problem of

ReFL was observed by Clark et al. [13] in their DRaFT experiments as well, where their fine-tuned model optimizing for Aesthetic score collapses to generate very similar, highreward images. We hypothesize that gradient-based optimization methods (i.e. ReFL and DRaFT) are more prone to reward hacking due to their direct access to the gradients of the reward model. In contrast, our wins on human preference rate indicate that our method is more robust to these effects.

#### 4.2. Optimizing Fairness and Diversity

The training of diffusion models is highly data-driven, relying on billion-sized datasets that are randomly scraped from internet. As a result, the trained models may contain significant social bias and stereotypes. For example, it has been observed that text-to-image diffusion models commonly exhibit a tendency to generate humans with lighter skintones [9, 34]. We aim to mitigate this bias by explicitly guiding the model using a skintone diversity reward.

For fine-tuning, we collect a dataset of 240M human images from Pinterest and run BLIP [27] to generate captions for each image. Only the text prompts are used during train-

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

Stable v2

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

Ours

A portrait of a dentist

A portrait of a judge

- Figure 5. Skintone Diversity Visualization. Qualitative comparison of SDv2 and our model fine tuned for skintone diversity reward. All images are generated using the same random seeds.

ing, and the reward calculation is based on the generated samples. We further filter out the captions containing terms relating to ethnicity and race (e.g. African, Asian, Indian) to ensure that the training prompts are race agnostic. In each training iteration, we load 128 prompts and generate a minibatch of 16 images for each prompt, then run a pre-trained skintone classifier on the generated samples and calculate the statistical parity for each minibatch according to Equation 12. Since the classifier has 4 skintone categories ranging from dark to light, the optimal reward is achieved when the output distribution is entirely uniform (i.e. 4 samples in each skintone bucket).

Statistical Parity (↓) Occupation HRS-Bench

Model

Stable v1.5 0.575 0.578 Stable v2 0.556 0.576 RAFT 0.464 0.527 Reward-weighted 0.562 0.527 Ours 0.453 0.498

Table 3. Fairness and Equity Evaluation. Statistical Parity scores on out-of-domain test sets.

We show our qualitative results in Figures 1 and 5 and quantitative results in Table 3. We construct two test sets: a set of 100 randomly sampled occupations, for which we add the prefix “a portrait of” to produce the final prompts (e.g.“a portrait of a police officer”), and another set of 200 prompts from HRSBench [3], which are descriptions of people with random objects. We note that both are out-of-domain test sets, as their distribution is different from that of the BLIPgenerated training prompts.

Our fine-tuned model greatly reduces the skintone bias embedded in the pretrained SDv2 model, especially for occupations with more social stereotypes or biases inherent in the pretraining dataset. For example, in Figure 5, we show that the pretrained SDv2 model is biased towards light skintone for portraits of dentists and judges, whereas our finetuned model generates a much more balanced distribution.

#### 4.3. Optimizing Compositionality

While diffusion models are able to generate diverse images, they often fail to accurately generate different compositions of objects in a scene [10, 22, 25, 30]. We further explore using our RL framework in ensuring compositionality with diffusion models. We collect a list of 532 common object classes (e.g. apple, backpack, book, balloon, avocado; the full list is available in Appendix H) and use 450 of them for training. The remaining classes are withheld for testing. We then construct training prompts by combining two different objects using one of five relationship terms: “and,” “next to,” “near,” “on side of” and “beside,” producing captions that designate a spatial relationship between two objects, e.g. “an apple next to an avocado.” In total we create a training set of over 1M prompts. In order to compute our object composition reward function (Eq. 13), we use UniDet [55], an object detector trained on multiple large-scale datasets that supports a wide range of object classes.

Object Detection Score (↑) Unseen Objects Seen Objects

Model

Stable v1.5 0.072 0.056 Stable v2 0.102 0.094 RAFT 0.094 0.092 Reward-weighted 0.136 0.152 Ours 0.231 0.221

Table 4. Compositional Evaluation. Average detection scores of the objects appearing in the prompts; we report the results on 300 randomly sampled prompts consisting of objects seen by the model during training and another 300 for unseen objects.

We present qualitative and quantitative results in Figure 6 and Table 4. To evaluate generalizability, we also generate samples with our fine-tuned model on 300 randomly sampled prompts from both unseen and seen objects. Our trained model adheres better to compositional constraints in text captions when compared to SDv2, and the learned compositional abilities also generalize to unseen objects.

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

Stable v2

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

Ours

A pumpkin beside a balloon

A cherry on side of a teddy bear

An avocado near a donut

A rabbit next to a cat A donkey on side of A bird beside a lemon a juice

- Figure 6. Compositional Visualization. Qualitative comparison of SDv2 and our model fine-tuned for compositionality reward. All images are generated using the same random seeds.

Evaluation Metic ImageReward (↑) Object Detection Score (↑) Statistical Parity (↓)

Model / Fine-tuning Task

Stable v2 0.273 0.098 0.567 Ours – (ImageReward) 0.783 0.114 0.659 Ours – (Compositionality) 0.304 0.226 0.575 Ours – (Skintone Diversity) 0.093 0.076 0.479 Ours – Joint 0.701 0.182 0.499

- Table 5. Joint Optimization. We experiment with jointly optimizing a single model to satisfy three separate reward functions. Comparing with the original baseline model, we see that our jointly optimized model is able to satisfy all three objectives, achieving over 80% (relative) performance of the individually fine-tuned models across all three evaluation metrics simultaneously. 4.4. Multi-reward Joint Optimization

LLMs [1, 36]. Specifically, when models are trained with a reward function that is only concerned with one aspect of images, it may learn to neglect sample quality or overall diversity of outputs. Our jointly fine-tuned model, in contrast, is able to mitigate the alignment tax issue by incorporating multiple diverse reward functions during fine-tuning, thereby maintaining performance on all tasks in question.

As detailed in Algorithm 1, we also perform multi-reward RL with all three reward functions jointly, aiming to improve the model performance on all three tasks simultaneously. We compare the jointly-trained model with the base model and the models fine-tuned for each individual task. The quantitative results are shown in Table 5, with more qualitative results available in Appendix B. Following the same evaluation setting, we test the models on multiple datasets for each metric and report the average scores.

### 5. Conclusion

We present a scalable RL training framework for directly optimizing diffusion models with arbitrary downstream objectives, including distribution-based reward functions. We conducted large-scale multi-task fine-tuning to improve the general performance of an SDv2 model in terms of human preferences, fairness, and object composition simultaneously, and found that joint training also mitigated the alignment tax issue common in RLHF. By evaluating our trained model against several baseline models on diverse out-of-domain test sets, we demonstrated our method’s generality and robustness. We hope our work inspires future research on targeted tuning of diffusion models, with potential future topics including addressing more complex compositional relationships and mitigating bias along other social dimensions.

While the best score for each metric is achieved by the model fine-tuned specifically for that task, our jointly-trained model is able to satisfy over 80% (relative) performance of the individually fine-tuned models across all three metrics simultaneously. In addition, it significantly outperforms the original base model on all tasks.

Alignment Tax. We observed degraded performance for individually fine-tuned models on some of the tasks that the models were not fine-tuned for. For example, the model optimized for human preference exhibits a significant regression on statistical parity, indicating a drastic drop in skintone diversity. Similarly, the model optimized for skintone diversity degrades in terms of human preference as compared to the base model. This is akin to the “alignment tax” issue that has been observed during RLHF fine-tuning procedure of

### References

- [1] Amanda Askell, Yuntao Bai, Anna Chen, Dawn Drain, Deep Ganguli, Tom Henighan, Andy Jones, Nicholas Joseph, Ben Mann, Nova DasSarma, Nelson Elhage, Zac Hatfield-Dodds, Danny Hernandez, Jackson Kernion, Kamal Ndousse, Catherine Olsson, Dario Amodei, Tom Brown, Jack Clark, Sam McCandlish, Chris Olah, and Jared Kaplan. A general language assistant as a laboratory for alignment, 2021. 8
- [2] Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, Nicholas Joseph, Saurav Kadavath, Jackson Kernion, Tom Conerly, Sheer El-Showk, Nelson Elhage, Zac Hatfield-Dodds, Danny Hernandez, Tristan Hume, Scott Johnston, Shauna Kravec, Liane Lovitt, Neel Nanda, Catherine Olsson, Dario Amodei, Tom Brown, Jack Clark, Sam McCandlish, Chris Olah, Ben Mann, and Jared Kaplan. Training a helpful and harmless assistant with reinforcement learning from human feedback, 2022. 1
- [3] Eslam Mohamed Bakr, Pengzhan Sun, Xiaoqian Shen, Faizan Farooq Khan, Li Erran Li, and Mohamed Elhoseiny. Hrs-bench: Holistic, reliable and scalable benchmark for textto-image models, 2023. 7
- [4] Federico Bianchi, Pratyusha Kalluri, Esin Durmus, Faisal Ladhak, Myra Cheng, Debora Nozza, Tatsunori Hashimoto, Dan Jurafsky, James Zou, and Aylin Caliskan. Easily accessible text-to-image generation amplifies demographic stereotypes at large scale. In 2023 ACM Conference on Fairness, Accountability, and Transparency. ACM, 2023. 1
- [5] Kevin Black, Michael Janner, Yilun Du, Ilya Kostrikov, and Sergey Levine. Training diffusion models with reinforcement learning, 2023. 1, 2, 3
- [6] Soravit Changpinyo, Piyush Sharma, Nan Ding, and Radu Soricut. Conceptual 12m: Pushing web-scale image-text pretraining to recognize long-tail visual concepts, 2021. 2
- [7] Hila Chefer, Yuval Alaluf, Yael Vinker, Lior Wolf, and Daniel Cohen-Or. Attend-and-excite: Attention-based semantic guidance for text-to-image diffusion models, 2023. 1, 2
- [8] Minghao Chen, Iro Laina, and Andrea Vedaldi. Training-free layout control with cross-attention guidance, 2023. 1, 2
- [9] Marc Cheong, Ehsan Abedin, Marinus Ferreira, Ritsaart Willem Reimann, Shalom Chalson, Pamela Robinson, Joanne Byrne, Leah Ruppanner, Mark Alfano, and Colin Klein. Investigating gender and racial biases in dall-e mini images. manuscript. 2, 6
- [10] Jaemin Cho, Abhay Zala, and Mohit Bansal. Dall-eval: Probing the reasoning skills and social biases of text-to-image generation models, 2023. 1, 2, 7
- [11] Kristy Choi, Aditya Grover, Trisha Singh, Rui Shu, and Stefano Ermon. Fair generative modeling via weak supervision,

2020. 4

- [12] Ching-Yao Chuang, Varun Jampani, Yuanzhen Li, Antonio Torralba, and Stefanie Jegelka. Debiasing vision-language models via biased prompts, 2023. 4
- [13] Kevin Clark, Paul Vicol, Kevin Swersky, and David J Fleet. Directly fine-tuning diffusion models on differentiable rewards, 2023. 1, 2, 5, 6, 11

- [14] Xiaoliang Dai, Ji Hou, Chih-Yao Ma, Sam Tsai, Jialiang Wang, Rui Wang, Peizhao Zhang, Simon Vandenhende, Xiaofang Wang, Abhimanyu Dubey, Matthew Yu, Abhishek Kadian, Filip Radenovic, Dhruv Mahajan, Kunpeng Li, Yue Zhao, Vladan Petrovic, Mitesh Kumar Singh, Simran Motwani, Yi Wen, Yiwen Song, Roshan Sumbaly, Vignesh Ramanathan, Zijian He, Peter Vajda, and Devi Parikh. Emu: Enhancing image generation models using photogenic needles in a haystack, 2023. 1
- [15] Hanze Dong, Wei Xiong, Deepanshu Goyal, Yihan Zhang, Winnie Chow, Rui Pan, Shizhe Diao, Jipeng Zhang, Kashun Shum, and Tong Zhang. Raft: Reward ranked finetuning for generative foundation model alignment, 2023. 2, 5, 11, 26
- [16] Yilun Du, Conor Durkan, Robin Strudel, Joshua B Tenenbaum, Sander Dieleman, Rob Fergus, Jascha Sohl-Dickstein, Arnaud Doucet, and Will Sussman Grathwohl. Reduce, reuse, recycle: Compositional generation with energy-based diffusion models and mcmc. In International Conference on Machine Learning, pages 8489–8510. PMLR, 2023. 2
- [17] Piero Esposito, Parmida Atighehchian, Anastasis Germanidis, and Deepti Ghadiyaram. Mitigating stereotypical biases in text to image generative systems, 2023. 1
- [18] Ying Fan, Olivia Watkins, Yuqing Du, Hao Liu, Moonkyung Ryu, Craig Boutilier, Pieter Abbeel, Mohammad Ghavamzadeh, Kangwook Lee, and Kimin Lee. Dpok: Reinforcement learning for fine-tuning text-to-image diffusion models, 2023. 1, 2
- [19] Weixi Feng, Xuehai He, Tsu-Jui Fu, Varun Jampani, Arjun Akula, Pradyumna Narayana, Sugato Basu, Xin Eric Wang, and William Yang Wang. Training-free structured diffusion guidance for compositional text-to-image synthesis, 2023. 1, 2
- [20] Weixi Feng, Wanrong Zhu, Tsu jui Fu, Varun Jampani, Arjun Akula, Xuehai He, Sugato Basu, Xin Eric Wang, and William Yang Wang. Layoutgpt: Compositional visual planning and generation with large language models, 2023. 1, 2
- [21] Felix Friedrich, Manuel Brack, Lukas Struppek, Dominik Hintersdorf, Patrick Schramowski, Sasha Luccioni, and Kristian Kersting. Fair diffusion: Instructing text-to-image generation models on fairness, 2023. 2
- [22] Tejas Gokhale, Hamid Palangi, Besmira Nushi, Vibhav Vineet,Eric Horvitz,Ece Kamar,Chitta Baral,and Yezhou Yang. Benchmarking spatial relationships in text-to-image generation, 2023. 1, 2, 7
- [23] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance, 2022. 11
- [24] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models, 2020. 1
- [25] Kaiyi Huang, Kaiyue Sun, Enze Xie, Zhenguo Li, and Xihui Liu. T2i-compbench: A comprehensive benchmark for openworld compositional text-to-image generation, 2023. 1, 2, 7
- [26] Kimin Lee, Hao Liu, Moonkyung Ryu, Olivia Watkins, Yuqing Du, Craig Boutilier, Pieter Abbeel, Mohammad Ghavamzadeh, and Shixiang Shane Gu. Aligning text-toimage models using human feedback, 2023. 2, 5, 11

- [27] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pre-training for unified visionlanguage understanding and generation, 2022. 5, 6
- [28] Yuheng Li, Haotian Liu, Qingyang Wu, Fangzhou Mu, Jianwei Yang, Jianfeng Gao, Chunyuan Li, and Yong Jae Lee. Gligen: Open-set grounded text-to-image generation, 2023. 1, 2
- [29] Long Lian, Boyi Li, Adam Yala, and Trevor Darrell. Llmgrounded diffusion: Enhancing prompt understanding of textto-image diffusion models with large language models, 2023. 1, 2
- [30] Nan Liu, Shuang Li, Yilun Du, Antonio Torralba, and Joshua B Tenenbaum. Compositional visual generation with composable diffusion models. In European Conference on Computer Vision, pages 423–439. Springer, 2022. 2, 7
- [31] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization, 2019. 11
- [32] Alexandra Sasha Luccioni, Christopher Akiki, Margaret Mitchell, and Yacine Jernite. Stable bias: Analyzing societal representations in diffusion models, 2023. 1
- [33] Shakir Mohamed, Mihaela Rosca, Michael Figurnov, and Andriy Mnih. Monte carlo gradient estimation in machine learning, 2020. 3
- [34] Ranjita Naik and Besmira Nushi. Social biases through the text-to-image generation lens, 2023. 6
- [35] Reiichiro Nakano, Jacob Hilton, Suchir Balaji, Jeff Wu, Long Ouyang,Christina Kim,Christopher Hesse,Shantanu Jain,Vineet Kosaraju, William Saunders, Xu Jiang, Karl Cobbe, Tyna Eloundou, Gretchen Krueger, Kevin Button, Matthew Knight, Benjamin Chess, and John Schulman. Webgpt: Browserassisted question-answering with human feedback, 2022. 1
- [36] Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, and Ryan Lowe. Training language models to follow instructions with human feedback, 2022. 1, 8, 26
- [37] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision, 2021. 1, 5
- [38] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. Exploring the limits of transfer learning with a unified text-to-text transformer, 2023. 1
- [39] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models, 2022. 1, 5
- [40] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, Patrick Schramowski, Srivatsa Kundurthy, Katherine Crowson, Ludwig Schmidt, Robert Kaczmarczyk, and Jenia Jitsev. Laion5b: An open large-scale dataset for training next generation image-text models, 2022. 5, 11

- [41] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017. 3
- [42] Brandon Smith, Miguel Farinha, Siobhan Mackenzie Hall, Hannah Rose Kirk, Aleksandar Shtedritski, and Max Bain. Balancing the picture: Debiasing vision-language datasets with synthetic contrast sets, 2023. 1, 2
- [43] Jascha Sohl-Dickstein, Eric A. Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics, 2015. 1
- [44] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models, 2022. 11
- [45] Richard S Sutton, David McAllester, Satinder Singh, and Yishay Mansour. Policy gradient methods for reinforcement learning with function approximation. In Advances in Neural Information Processing Systems. MIT Press, 1999. 3
- [46] Christopher T. H Teo and Ngai-Man Cheung. Measuring fairness in generative models, 2021. 4
- [47] Zijie J. Wang, Evan Montoya, David Munechika, Haoyang Yang, Benjamin Hoover, and Duen Horng Chau. DiffusionDB: A large-scale prompt gallery dataset for text-toimage generative models. arXiv:2210.14896 [cs], 2022. 4, 5, 12, 13, 16, 17
- [48] Ronald J. Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine Learning, 8:229–256, 2004. 3
- [49] Qiucheng Wu, Yujian Liu, Handong Zhao, Trung Bui, Zhe Lin, Yang Zhang, and Shiyu Chang. Harnessing the spatialtemporal attention of diffusion models for high-fidelity textto-image synthesis, 2023. 2
- [50] Xiaoshi Wu, Keqiang Sun, Feng Zhu, Rui Zhao, and Hongsheng Li. Human preference score: Better aligning text-toimage models with human preference, 2023. 1, 2
- [51] Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong. Imagereward: Learning and evaluating human preferences for text-to-image generation, 2023. 1, 2, 4, 5, 11, 12, 18
- [52] Jiahui Yu, Yuanzhong Xu, Jing Yu Koh, Thang Luong, Gunjan Baid, Zirui Wang, Vijay Vasudevan, Alexander Ku, Yinfei Yang, Burcu Karagol Ayan, Ben Hutchinson, Wei Han, Zarana Parekh, Xin Li, Han Zhang, Jason Baldridge, and Yonghui Wu. Scaling autoregressive models for content-rich text-to-image generation, 2022. 4, 5, 6, 12, 14
- [53] Cheng Zhang, Xuanbai Chen, Siqi Chai, Henry Chen Wu, Dmitry Lagun, Thabo Beeler, and Fernando De la Torre. ITIGEN: Inclusive text-to-image generation. In ICCV, 2023. 2
- [54] Yanzhe Zhang, Lu Jiang, Greg Turk, and Diyi Yang. Auditing gender presentation differences in text-to-image models,

2023. 2

- [55] Xingyi Zhou, Vladlen Koltun, and Philipp Kr¨ahenb¨uhl. Simple multi-dataset detection, 2022. 7

### Appendix

This appendix is structured as follows:

- • In Appendix A, we provide more details of our experimental setup, including hyperparameters and baselines.
- • In Appendix B, we provide additional qualitative results and comparison of our method with the baselines.
- • In Appendix C, we provide evaluation guidelines and templates used for collecting human rating.
- • In Appendix D, we provide additional human evaluation results for skintone diversity and compositionality.
- • In Appendix E, we provide the training curves of all online-learning methods (including ours and other baselines) to demonstrate the training progress and convergence time.
- • In Appendix F, we illustrate the issue of reward hacking and provide visual examples.
- • In Appendix G, we provide an ablation study on the effect of pretraining dataset.
- • In Appendix H, we provide complete lists of 100 occupations for skintone diversity evaluation and 532 objects for training and evaluating the compositionality skill of the models.

### A. Experiment Details and Hyperparameters

All our experiments including baseline methods training were conducted on 128 80GB A100 GPUs. If a pretraining dataset is required, all fine-tuning methods use the same 12M subset of LAION-5B [40] filtered by the aesthetic score predictor with a threshold of 6. For optimization, we use the AdamW optimizer [31] with β1 = 0.9,β2 = 0.999,ϵ = 1e−8 and a weight decay of 1e − 2 for all the experiments. For inference, we run the diffusion process with 50 steps for each image with DDIM [44] noise scheduler. We use the default guidance scale of 7.0 for classifier-free guidance [23].

Implementation Details. For our RL fine-tuning experiments, we collect 16x128 samples per training iteration, with 50 samplings steps using DDIM scheduler [44]. We randomly sample 5 training timesteps and perform a gradient update across all the samples in the batch for each of the timesteps, resulting in 5 gradient updates per iteration. We use a small clip range of 1e − 4 for all the experiments.

Baseline Details. For the baseline methods including ReFL [51], Reward-weighted [26], RAFT [15] and DRaFT [13], we refer to the original implementation for the suggested hyperparameters and report our experiment details in Table 6. We use the same training set for all the baseline models training and fine-tune them until convergence. Since the experiments involve million-sized training prompts, for reward-weighted approach, instead of pre-generating the samples and storing the dataset offline, we generate the samples on the fly during training using the original SDv2 model and re-weigh them according to the reward values for fine-tuning. Following Xu et al. [51], we also map the reward values to the range of [0,1] using min-max normalization.

We note that DRaFT imposes a high memory burden by directly back-propagating the gradient from the reward model through the sampling process of diffusion model, allowing for a much smaller batch size compared to other optimization methods. We implement DRaFT-LV, which claimed to be the most efficient DRaFT variant.

|Hyperparameter<br><br>|ReFL|Reward-weighted<br><br>|RAFT|DRaFT<br><br>|Ours|
|---|---|---|---|---|---|
|Learning Rate Batch Size (Per GPU) Pretraining Batch Size (Per GPU) Sampling Scheduler Sampling Steps|1e-5 12 12 DDIM 40<br><br>|1e-5 16 16 DDIM 50|3e-6 32 32 DDIM 50<br><br>|5e-5 3 DDIM 50|2e-6 16 16 DDIM 50<br><br>|
|Method Specific|ϕ = ReLU [T1, T2] = [1, 10] λ = 1e − 3<br><br>|β = 0.5<br><br>|Acceptance ratio: 1/24<br><br>|LoRA rank: 32 ttruncate=1|clip range: 1e-4 Training timesteps: 5|

Table 6. Training Hyperparameters. We report the hyperparameters used in different experiments, where method-specific indicates the hyperparameters specific to each individual method.

### B. Additional Qualitative Results

We provide additional qualitative results in this section, including results from the models that were trained with single rewards (i.e., ImageReward [51], compositionality reward and skintone diversity reward), as well as the results from our model that was jointly trained with all three rewards simultaneously.

#### B.1. Results on Human Preference Fine-tuning

We show the visual samples from our model fine-tuned with ImageReward [51] on real-user prompts in Figure 7. We also provide more qualitative comparison of our model with other reward optimization methods in Figure 8. More results on the out-of-domain test set PartiPrompts [52] are availble in Figure 9. Our trained model generates more visually appealing images compared to the base SDv2 model, and it generalizes well to out-of-domain test sets with unseen text prompts that have a different distribution from that of the training prompts.

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

###### StableV2OursStableV2Ours

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

Little big planet by Joel Robison

When did the bright moon come to be? I raise my drink to ask the azure sky…

A fractal strong cat gentleman with mechanical glasses, ...

A portrait painting of a husky in cowboy costume, wearing a cowboy hat, …

Beautiful serene walk to the top of the hill to see the horizon…

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

Scifi action scene greg rutkowski and raymond swanland digital painting of a habitat on the moon, …

A unicorn flying over a city on fire, matte painting, long shot, , concept art, …

vintage photo of a fat robot sitting alone in a street bench, dieselpunk, cyberpunk, stray, neon lights

Portrait of a cyberpunk young witch in a black robe standing in the grass with full moon in the sky over her head, …

the squirrel thor ~ holding his hammer ~ dramatic thunder background ~ fighting scene ~

- Figure 7. Qualitative Comparison of SDv2 and Our Fine-tuned Model. All images are generated using real-user prompts from DiffusionDB [47] dataset with the same random seeds. Our outputs are better aligned with human aesthetic preferences, favoring finer details, focused composition, vivid colors, and high contrast.

###### Stable V2 ReFL RAFT DRaFT Reward-weighted Ours

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

A portrait of an anthropomorphic wolf wearing a black doublet, furry fursona, Victorian era masterwork, by Samuel Luke Fildes

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

Illustration, a study of a nordic village, post grunge concept art by Josan Gonzales and Wlop, highly detailed, intricate, sharp focus, Trending on Artstation HQ, deviantart-H 704

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

Kitten walks the empty street in a rainy day, led lights around the place, digital painting, ultra detailed, unreal engine 5

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

Woman with long red hair, very beautiful style, in a gold suit, night desert, dunes, photorealism, night in the desert, her face illuminated by golden rays, pensive, dreamy, red lips, john singer sargent, edgard maxence

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

- A portrait of a gothic princess in white baroque dress in a scenic environment by Henriette Ronner - Knip

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

Goddess of illusion, beautiful, stunning, breathtaking, mirrors, glass, magic circle, magic doorway, fantasy, mist, bioluminescence, hyper-realistic, unreal engine, by blizzard concept artists

- Figure 8. Additional Qualitative Comparison Results. We compare our fine-tuned model with other reward fine-tuning methods on real-user prompts from DiffusionDB [47] dataset. All images are generated using the same random seeds.

###### Stable V2 ReFL RAFT DRaFT Reward-weighted Ours

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

a peaceful lakeside landscape

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

a drawing of a man standing under a tree

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

a cat looking out of a window

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

a small house

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

a robot cooking in the kitchen

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

black hi-top sneakers

- Figure 9. Additional Qualitative Comparison on Out-of-domain Test Sets. We compare our fine-tuned model with other reward finetuning methods on PartiPrompts [52] dataset. Our model generates samples with higher aesthetic quality and better image-text alignment compared to other baseline models. All images are generated using the same random seeds.

#### B.2. Results on Optimizing Diversity

We provide more qualitative results of our model fine-tuned with skintone diversity reward in Figure 10. Our trained model effectively mitigates the inherent bias and stereotypes in the base SDv2 model with increased skintone diversity in the generated human samples.

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

Stable v2

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

Ours

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

Rewardweighted

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

RAFT

A portrait of a chemist

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

Stable v2

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

Ours

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

Rewardweighted

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

RAFT

A portrait of a police officer

- Figure 10. Skintone Diversity Visualization. We compare our model that was fine-tuned with skintone diversity reward with other baseline models. All images are generated using the same random seeds. We note that while RAFT also improves the skintone diversity of the output samples, it is prone to overfitting and generates over-saturated samples with decreased realism (e.g. the portraits of police officers in the second example).

#### B.3. Results on Optimizing Compositionality

We provide more qualitative results of our model fine-tuned with object composition reward in Figure 11. Our fine-tuned model demonstrates improved compositional skills compared to the base SDv2 model and other baseline models.

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

Stable v2

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

Ours

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

Rewardweighted

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

RAFT

a cake next to a desk

chips beside a blue jay

an apple and a pillow

a blue jay near a coffee

an insect next to a telephone

a flashlight beside a lily

Figure 11. Object Composition Visualization. We compare our model that was fine-tuned with object composition reward with other baseline models. All images are generated using the same random seeds.

#### B.4. Results on Multi-reward Joint Optimization

Next, we show more qualitative results from our jointly-fine-tuned model (with all three rewards simultaneously) on multiple test sets: DiffusionDB [47] (Figure 12), object composition (Figure 13) and occupation prompts (Figure 14). We demonstrate that our jointly-trained model has quite significant improvement over the base SDv2 model in terms of all three objectives: human preferences, skintone diversity and object composition. We further note that since joint training utilizes multiple reward signals (including ImageReward which reflects human preferences) during training, for portraits of occupations, we also observe additional increase in the aesthetic quality of the samples compared to single-reward training which optimizes for the skintone diversity only; see Figure 14.

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

Stable v2

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

Ours Single

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

Ours Joint

a woman with black hair and long pixie haircut in shorts with suspenders and white t-shirt drawn by frederic leighton…

a portrait of an anthropomorphic wolf wearing a black doublet, furry fursona, victorian era masterwork, …

beautiful serene walk to the top of the hill to see the wast horizon, healing through motion, life, minimalistic golden and ink airbrush…

small and dense intricate vines, moss, roots, colorful flowers, and tree branches take the detailed form of an elk standing in a lush forest…

anime picture, black hair, girl, digital art, octane render, trending on artstation

beautiful landscape photography by marc adamus, dolomites, a lake, mist, reflections, sunset, dramatic sky

- Figure 12. Visualization of Jointly-optimized Model on Real-user Prompts. We show the results from our jointly-fined-tuned model on real-user prompts from DiffusionDB [47] dataset. Our jointly-trained model generates more aesthetically pleasing images compared to the base SDv2 model.

Stable v2

Ours Single

Ours Joint

[Figure 252]

[Figure 253]

a cake next to a desk

[Figure 254]

[Figure 255]

chips beside a blue jay

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

a house on side of sunglasses

[Figure 261]

[Figure 262]

[Figure 263]

a truck next to a house

[Figure 264]

[Figure 265]

[Figure 266]

an apple next to a cup

[Figure 267]

[Figure 268]

[Figure 269]

a rabbit near a coconut

- Figure 13. Visualization of Jointly-optimized Model on Object Composition Prompts. Our jointly-trained model generates samples with improved compositionality compared to the base SDv2 model.

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

Stable v2

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

Ours Single

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

Ours Joint

A portrait of a chemist

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

Stable v2

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

Ours Single

[Figure 300]

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

Ours Joint

A portrait of a police officer

- Figure 14. Visualization of Jointly-optimized Model on Occupation Prompts. We show the results from our jointly-fined-tuned model on occupations prompts for skintone diversity evaluation. Our jointly-trained model has greatly reduced the inherent bias in the base SDv2 model and generates human samples with more diverse skintone. Compared to single-reward training, we also observe the additional increase in the aesthetic quality of the samples from our jointly-trained model.

### C. Human Evaluation Templates

We provide the detailed human evaluation guidelines document that were used to train our hired human labelers in section C.1, including the judging criteria and concrete examples for making trade-offs in order to help the evaluators better understand the task and make fair judgments. We use the annotation documents from ImageReward [51] as a reference. We also show our evaluation UI interface in section C.2.

#### C.1. Evaluation Criteria and Guidelines

You will be given a number of prompts/queries and there are several AI-generated images according to the prompt/query. Your annotation requirement is to evaluate these images in terms of Image Fidelity, Relevance to the Query, and Aesthetic Quality. Below are more details on each of the three mentioned factors.

##### C.1.1 Image Fidelity

Definition: The generated image should be true to the shape and characteristics of the object, and not generated haphazardly. Some examples of low-fidelity images are:

- • Dogs should have four legs and two eyes, generating an image with extra / fewer legs or eyes is considered low-fidelity.
- • “Spider-Man”” (or human) should only have two arms and five fingers each. Generating extra arms / fingers is considered low-fidelity.
- • “Unicorn” should only have one horn, generating an image with multiple horns is considered low-fidelity.
- • People eat noodles with utensils instead of grabbing them with their hands, generating an image of someone eating noodles with their hands is considered low-fidelity.

See Figure 15 for examples of low-fidelity generation. Images of low fidelity should be ranked as low preference.

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

Figure 15. Examples of Low-fidelity Generation. Note that these generated images have incorrect details with faces or body parts of human and animals, and would likely cause psychological discomfort. They should be ranked with lower-preference.

##### C.1.2 Relevance to the Query

Definition: the generated image should match the text in the query. Another term used for“Relevance” is “Text-alignment”. Some examples of inconsistent image generation are:

- • The subject described in the text does not appear in the image generated, for example, “A cat dressed as Napoleon Bonaparte” generates an image without the word “cat”.
- • The object properties generated in the image are different from the text description, for example, generating an image of “a little girl sitting in front of a sewing machine” with a boy (or many little girls) is incorrect.

See Figure 16 for examples of low-relevance generation. Images of low relevance to the query should be ranked as low preference.

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

(a) “A cucumber beside a peach ” (b) “A box and a canoe” (c) “set of 2 canvas paintings” (d) “sip and paint at home date night”

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

(e) “Matching wallpaper for two best friends”

(f) “Underwater congress art” (g) “Shoe design sketches draw ” (h) “Black cat minimalist art ”

Figure 16. Examples of Generation with Low-relevance to the Text Prompts. They should be ranked with lower-preference.

##### C.1.3 Aesthetic Quality

Definition: the generated images should look visually appealing and beautiful. Examples are provided in Figure 17, where two images are generated given the same text prompt and the one with higher aesthetic quality is highlighted.

[Figure 322]

[Figure 323]

(a) “Subset by the sea ” (b) “Dog art”

Figure 17. Illustration of Aesthetic Quality. The two images are generated given the same text prompt, and the highlighted one on the left is considered to have higher aesthetic quality (i.e. more visually appealing) and should be ranked with higher-preference.

##### C.1.4 Overall Preference Ranking

Guidelines for deciding boundary cases: which generated images would you prefer to receive from AI painters? Evaluating the output of the model may involve making trade-offs between the criteria we discussed. These trade-offs will depend on the task. When making these trade-offs, use the following guidelines to help choose between outputs.

- 1. For most tasks, fidelity & aesthetic quality are more important than image-text alignment. So, in most cases, the image having higher fidelity and aesthetic quality is rated higher than an output that is more image-text aligned.
- 2. However, if an output image:

- • clearly matches the text better than the other;

- • is only slightly lacking in the requirements of fidelity;
- • the content does not have significant artifacts that would cause psychological discomfort then the more consistent result is rated higher. We provide more examples below to illustrate how to make trade-off between the different criteria when making judge-

ments.

[Figure 324]

Figure 18. “Matching wallpaper for two best friends”

- In the example above (Figure 18), image A and B are the ones that match the text description best, and they are also the

most aesthetically appealing (A is better than B in both regards). The animals in image C look unnatural and have artifacts, also C does not align with the text very well. Image D does not match the text, and it has the lowest aesthetic quality too. Thus the overall ranking should be A > B > C > D.

[Figure 325]

Figure 19. “Anime wallpapers”

- In the example above (Figure 19), image A and B both match the text (they are wallpapers of some anime style), and image

- B looks slightly more appealing, so we rank B > A. Note that Image C has a lots of noticeable artifacts in the body parts of the anime character and it might cause psychological discomfort , so it should be ranked as the lowest. The overall ranking should be B > A > D > C (D is better than C because of the significant artifacts in C).

[Figure 326]

Figure 20. “Astronaut on Mars during sunset”

- In the example above (Figure 20), all four images are depiction of astronaut on mars during sunset, so they all match with

the text well. In this case we should mainly consider the fidelity and aesthetic quality of the images. Among the four images, Image A and C look the most beautiful (with C slightly better than A). Image D has the lowest aesthetic quality compared to others. So the overall ranking should be C > A > B > D.

[Figure 327]

Figure 21. “Forbidden city drawing”

- In the example above (Figure 21), image C is somehow a nonsense generation and does not match the text, so it is apparent

that C should be ranked the lowest. Image A, B and D all match with the text, and in terms of fidelity and aesthetic quality, they should be ranked as B > A > D (B looks the most appealing, followed by A, while D only shows part of the palace and is not as beautiful as B). The overall ranking should be B > A > D > C.

[Figure 328]

Figure 22. “Colorful art fire”

- In the example above (Figure 22), image A and C both have fire in it, and image A looks more visually appealing. Note that

although C is more colorful, we think image A matches with the text well enough; since A is much more visually appealing than C, we rank A > C. B and D both have lower image-text alignment and lower aesthetic quality, so we rank them as the lowest two. The overall ranking should be A > C > B > D.

#### C.2. Evaluation Interface

To compare our fine-tuned model with the base SDv2 model and models tuned with other baseline approaches, we perform head-to-head comparison of two images generated from different sources using the same text prompt. The two images are generated using the same random seed for fair comparison. The human evaluators were trained using the guidelines provided in section C.1. During evaluation, we show two generated images and the associated text query, and ask the evaluators to choose the preferred one based on image fidelity and aesthetic quality, as well as image-text relevance. We show the evaluation interface in Figure 23.

[Figure 329]

###### Figure 23. Human Evaluation Interface. We ask the hired evaluators to compare two generation from the same text prompt based on image fidelity and quality, as well as image-text relevance.

D. Additional Human Evaluation

- D.1. Additional Results

For a more thorough evaluation on the effectiveness of our method on improving compositionality and diversity, we also perform human evaluation on our models trained with compositionality reward and skintone diversity reward and provide the results in Figure 24. For the compositionality evaluation, the annotators were asked to rate the samples based on image-text relevance (how well the generated images match the text).

Text-Image Relevance Eval

Skintone Diversity Eval

| |84.2<br><br>68.5 66.4<br><br>61.3<br><br>15.8<br><br>31.5 33.6<br><br>38.7| | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |

| |84.4<br><br>57.0 56.2 56.1<br><br>15.6<br><br>43.0 43.8 43.9| | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |

80%

80%

70%

70%

60%

60%

50%

50%

40%

40%

30%

30%

20%

20%

10%

10%

0%

0%

Ours SD1_5 Ours SD2 Ours RAFT Ours Rewardweighted

Ours SD1_5 Ours SD2 Ours RAFT Ours Rewardweighted

###### Figure 24. Human evaluation results on our models fine-tuned with object composition reward and skintone diversity reward, respectively. For the object composition evaluation, the human evaluators were asked to make judgements based on the relevance of the generated images to the text prompts; for the skintone diversity evaluation, the evaluators were shown two groups of images from different models and were asked to make judgements based on the overall skintone diversity. We provide the detailed evaluation guidelines and interfaces for skintone diversity in section D.2.

#### D.2. Skintone Diversity Evaluation

We provide the guidelines we used to train the hired human evaluators on rating skintone diversity in section D.2.1 and the evaluation interface in section D.2.2.

##### D.2.1 Guidelines

You will be given a number of prompts and there are several AI-generated images according to the prompt. Your annotation requirement is to evaluate these outputs in terms of skintone diversity. What we mean by “skintone diversity” is that AI model should have minimum bias and stereotypes, so the generated images should have a diverse set of people with different skintone. For example, “a portrait of a police officer” is not supposed to generate only light skintone or dark skintone; what we want instead is a balanced distribution of light and dark skintone. For each prompt, there are two sets (set A and set B) of images, each with 6 images. Your job is to choose the one set with better skintone diversity. See the examples below:

[Figure 330]

[Figure 331]

[Figure 332]

[Figure 333]

[Figure 334]

[Figure 335]

- Set A
- Set B

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

[Figure 341]

a portrait of a chemist

[Figure 342]

[Figure 343]

[Figure 344]

[Figure 345]

[Figure 346]

[Figure 347]

- Set A
- Set B

[Figure 348]

[Figure 349]

[Figure 350]

[Figure 351]

[Figure 352]

[Figure 353]

a portrait of a builder

Figure 25. Skintone Diversity Human Evaluation. We provide two groups of images for each prompt and ask the annotators to choose the one with higher skintone diversity.

Note that in Figure 25 Set B is more diverse for both examples of portraits of a chemist and a builder, because it has a balanced distribution of light and dark skintone, while set A has mostly light skintone in it.

##### D.2.2 Interface

For evaluating skintone diversity, we perform head-to-head comparison of two groups of images generated from different sources using the same text prompt. The two groups are generated using the same random seed for fair comparison and the evaluators were asked to choose the one that has better diversity. The human evaluators were trained using the guidelines provided in section D.2.1. We show the evaluation interface in Figure 26.

[Figure 354]

###### Figure 26. Skintone Diversity Human Evaluation Interface. We ask the hired evaluators to compare two groups of generation from the same text prompt based on skintone diversity.

E. Training Curve

We plot the training curves of our method and other online learning baseline methods in Figure 27 and note that, except for RAFT which diverged, all online-learning methods exhibit steadily increasing sample rewards during training, eventually saturating at some maximum level, at which point we consider the models converged. Our method converged pretty quickly in as few as 1,000 steps.

1.75

1.50

1.25

TrainingReward

1.00

0.75

0.50

ReFL

0.25

Ours

DRaFT

0.00

RAFT

0 1000 2000 3000 4000 Iteration

###### Figure 27. Training Curve of All Online-learning Methods. Y-axis shows the average reward of the samples from each training batch, and x-axis is the training iteration. In contrast to the common belief that RL training is sample inefficient and slow to converge, our approach converges in as few as 1,000 steps, compared to DRaFT, the gradient-based reward optimiza-tion approach which takes 4,000 steps to converge while only being able to optimize for differentiable rewards. Our approach shows a steadily increasing sampling reward until convergence.

### F. Reward Hacking

We found that ReFL is prone to reward hacking, a well known issue in RLHF [15, 36]. Specifically, since the reward model trained from human annotation data is far from perfect, the imperfection can be exploited by the algorithms to chase for a high reward, leading to reward hacking [15]. We provide more visual examples of reward hacking from ReFL in Figure 28.

Stable V2 ReFL Iter 200 ReFL Iter 400 ReFL Iter 600 ReFL Iter 800 ReFL Iter 1000

[Figure 355]

[Figure 356]

[Figure 357]

[Figure 358]

[Figure 359]

[Figure 360]

- IM*=0.99 IM*=0.79 IM*=1.82 IM*=1.97 IM*=1.98 IM*=1.96

[Figure 361]

[Figure 362]

[Figure 363]

[Figure 364]

- IM*=1.83 IM*=1.94 IM*=1.99 IM*=2.00 IM*=2.00 IM*=1.99

realistic extremely detailed portrait painting of a wise ape, futuristic sci-fi landscape with a statue on background by Jean Delville, Amano, Yves Tanguy, Alphonse Mucha, Ernst Haeckel, Edward Robert Hughes, Roger Dean, rich moody colours, silver hair and beard, blue eyes

[Figure 365]

[Figure 366]

an anthropomorphic steampunk westie wearing scifi armor, diffuse lighting, fantasy, intricate, elegant, highly detailed, lifelike, photorealistic, digital painting, artstation, illustration, concept art, smooth, sharp focus, art by john collier and albert aublet and krenz cushart and artem demura and alphonse mucha

[Figure 367]

[Figure 368]

[Figure 369]

[Figure 370]

[Figure 371]

[Figure 372]

IM*=0.40 IM*=1.11 IM*=1.39 IM*=1.20 IM*=1.48 IM*=1.24

a cyberpunk city, digital painting by moebius

- Figure 28. Reward Hacking Examples from Different Iterations of ReFL-fine-tuned Models. While the reward fine-tuning method ReFL quickly increases ImageReward(IM*) values during training by backpropagating the gradients from pretrained ImageReward model, it learnt to generate over-detailed images with high-frequency noise. This issue is also known as reward hacking, a well-known issue in RLHF [15, 36].

G. Effect of Pretraining Dataset

As discussed in the paper, we incorporate the pretraining denoising loss Lpre to stabilize the training and to prevent reward over-optimization. In practice, we observe that the model is more prone to reward-hacking (i.e. producing unnatural artifacts and decreased photo-realism) without the pretraining loss. We experiment with removing Lpre and show the comparison in

- Figure 29 .

[Figure 373]

[Figure 374]

[Figure 375]

[Figure 376]

[Figure 377]

No Pretraining Loss

[Figure 378]

[Figure 379]

[Figure 380]

[Figure 381]

[Figure 382]

With Pretraining Loss

A portrait of a PR person

Figure 29. Effect of pretraining loss. The images are sampled from the models trained with and without pretraining loss after the same number of iterations, using the same random seeds. Without pretraining loss, the model is prone to grainy artifacts and decreased realism.

- H. Full List of Occupations and Objects We provide the full list of 100 occupations used to evaluate the skintone diversity of the generated samples in Table 7.

|Occupation List|
|---|
|accountant administrative assistant animator announcer architect assistant author economist editor engineer executive optician PR person TV presenter baker bartender biologist builder building inspector butcher<br><br>career counselor caretaker chef chemist chief executive officer childcare worker civil servant clerk comic book writer computer programmer<br><br>construction worker cook crane operator custodian decorator dentist designer diplomat director doctor drafter farmer film director flight attendant garbage collector<br><br>geologist hairdresser head teacher housekeeper jeweler journalist judge juggler lawyer lecturer librarian magician mail carrier makeup artist manager musician nurse nurse practitioner painter personal assistant<br><br>pharmacist photographer pilot plumber police officer porter primary school teacher printer prison officer puppeteer receptionist roofer sailor salesperson scientist<br><br>secretary security guard sign language interpreter singer software developer soldier solicitor surgeon tailor teacher technical writer telemarketer telephone operator telephonist travel agent trucker vet veterinarian waiter web designer<br><br>|

Table 7. Full List of 100 Occupations Used in Skintone Diversity Evaluation.

We also provide the full list of 532 common objects used to construct the training set for the compositionality experiments in Table 8. During training, two objects were randomly sampled and combined using one of the five relationship terms: “and”, “next to”, “near”, “on side of”, and “beside”.

##### Objects List

accordion air conditioner aircraft airplane alarm clock alpaca

ant antelope apple artichoke asparagus avocado backpack bagel ball balloon banana baozi

bar soap barbell barrel baseball baseball bat baseball glove basket basketball bat bathtub beaker bear

bed bee beer beetle bell pepper belt

bench bicycle bicycle helmet bicycle wheel bicyclist billboard binoculars bird blender blue jay boat book

bookcase boot boots bottle bow tie bowl

box boy bread broccoli broom brown bear brush bucket building bull burrito bus

butterfly cabbage cabinet cake cake stand calculator camel camera canary candle candy cannon canoe car caravan carpet carriage carrot

cart castle cat caterpillar cd cell phone chainsaw chair cheese cheetah cherry chicken

chips chopsticks christmas tree cigar clock clutch

coat cocktail coconut coffee coffee cup coffee table coffeemaker coin comb computer box computer monitor converter cookie corn couch cow cowboy hat crab crocodile croissant crosswalk crosswalk sign crosswalk zebra crown

crutch cucumber cup cupboard curtain cutting board cymbal dagger dates deer desk dessert

dice digital clock dining table dinosaur dog dolphin

donkey donut door dragonfly drawer dress drink drinking straw drum duck dumbbell durian eagle earphone earrings egg egg tart eggplant

electric drill elephant envelope eraser facial mask fedora fig filing cabinet fire extinguisher fire hydrant fire truck fireplace fish fishing rod flashlight flower flowerpot folder football football helmet fork fountain fox french fries french horn frisbee frog frying pan game board garlic

giraffe girl glasses globe glove goat goggles goldfish golf ball golf cart goose grape

grapefruit green beans green vegetables guitar hair drier hamburger

hamimelon hammer hamster handbag handgun hanger harbor seal harp hat headphones helicopter helmet high heels horn horse hot dog hotair balloon house

hurdle ice cream insect iron jacket jeans jellyfish jet ski jug juice kangaroo kettle

key keyboard kitchen knife kite kiwi fruit knife ladder lamp lantern laptop lavender lemon

leopard lettuce lifejacket light bulb lighter lighthouse lily lion liquid soap lizard lobster luggage

lynx mailbox man mango mangosteen manhole maple marker measuring cup meat balls mechanical fan medal

microphone microscope microwave microwave oven mirror missile

monkey mop motorcycle motorcyclist mouse muffin mug mule mushroom necklace nightstand nuts

office building okra onion orange ostrich otter

oven owl oyster paddle paint brush palm tree pancake papaya paper towel parachute parking meter parrot

pasta peach pear pen pencil case penguin pepper person phone booth piano picnic basket picture

pie pig pigeon pillow pineapple pitaya

pitcher pizza plastic bag plate platter plum poker card polar bear pole pomegranate pomelo popcorn porcupine poster pot potato potted plant power outlet

pressure cooker pretzel printer projector pumpkin punching bag rabbit raccoon race car racket radiator radio radish raven red cabbage refrigerator remote reptile

rhinoceros rice rice cooker rifle, gun ring rocket

rose router ruler sailboat salad sandal sandals sandwich saucer sausage saw saxophone

scale scallop scarf scissors scoreboard screwdriver sculpture sea turtle seahorse seal sewing machine shark

sheep shelf shellfish ship shirt shotgun

shrimp sink skateboard ski skirt skull skyscraper slide slippers snail snake sneakers snowboard snowman snowmobile snowplow sock sofa

sombrero sparrow speaker spider spoon sports car squirrel stairs stapler starfish stationary bicycle steak

stool stop sign strawberry street light stroller suitcase sun hat sunflower sunglasses surfboard surveillance camera sushi

suv swim cap swimming pool swimwear swing sword table tablet tank tape target tart

taxi tea teapot teddy bear telephone television tennis ball tennis racket tent tiara tick tie

tiger tin can tire tissue toaster toilet tomato tong toothbrush toothpaste tortoise towel tower toy traffic cone traffic light traffic sign trailer

train trash bin tree tricycle tripod trombone trophy trousers truck trumpet tuba turtle

tv umbrella utility pole van vase vegetable

vehicle violin volleyball waffle wall clock washing machine waste container watch watermelon weapon whale wheel

wheelchair whiteboard wild bird willow window window blind

wine wine glass winter melon wok woman woodpecker wrench yak zebra zucchini

Table 8. Full List of 532 Objects Used in Compositionality Training Experiments.

