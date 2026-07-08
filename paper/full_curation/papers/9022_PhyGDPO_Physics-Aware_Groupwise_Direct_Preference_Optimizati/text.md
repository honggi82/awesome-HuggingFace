# arXiv:2512.24551v4[cs.CV]18Jun2026

### PhyGDPO: Physics-Aware Groupwise Direct Preference Optimization for Physically Consistent Text-to-Video Generation

Yuanhao Cai1,2,∗, Kunpeng Li1, Menglin Jia3, Jialiang Wang1, Junzhe Sun1, Feng Liang1, Weifeng Chen1, Felix Juefei-Xu1, Chu Wang1, Ali Thabet1, Xiaoliang Dai1, Xuan Ju4, Alan Yuille2,†, Ji Hou1,†

1Meta Superintelligence Labs, 2Johns Hopkins University, 3Meta BizAI, 4CUHK ∗Work was done while Yuanhao Cai was an intern in Meta Superintelligence Labs, †Equal Advising

Recent advances in text-to-video (T2V) generation have achieved good visual quality, yet synthesizing videos that faithfully follow physical laws remains an open challenge. Existing methods mainly based on graphics or prompt extension struggle to generalize beyond simple simulated environments or learn implicit physics reasoning. The scarcity of training data with rich physics interactions and phenomena is also a problem. In this paper, we first introduce a Physics-Augmented video data construction Pipeline, PhyAugPipe, that leverages a vision–language model (VLM) with chain-of-thought reasoning to collect a training dataset, PhyVidGen-135K. Then we formulate a principled Physics-aware Groupwise Direct Preference Optimization, PhyGDPO, framework that uses real-world video as winning case to guarantee correct physics learning and builds upon the groupwise Plackett–Luce probabilistic model to capture holistic preferences beyond pairwise comparisons. In PhyGDPO, we design a Physics-Guided Rewarding (PGR) scheme that leverages VLM-based physical rewards to direct the optimization to focus on challenging physics cases. In addition, we propose a LoRA-Switch Reference (LoRA-SR) scheme that avoids full-model duplication as reference for efficient DPO training. Comprehensive experiments show that our method outperforms state-of-the-art methods on the PhyGenBench and VideoPhy2 datasets.

Project Page: https://caiyuanhao1998.github.io/project/PhyGDPO

(a)A gymnast performs a transition from a front support to a back support … (b) A soccer player runs, plants their foot, and drop kicks a soccer ball high into the air …

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

[Figure 12]

(c) A player dunks the basketball, the basketball soaring upward before slamming through the net. (d) A baseball bat smashes a glass bottle, sending shards flying in all directions.

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

Wan2.1-14B Wan2.1-14B + PhyGDPO OpenAI Sora2 Wan2.1-14B Wan2.1-14B + PhyGDPO Google Veo3.1

Figure 1 Text-to-video generation on four challenging action categories: (a) gymnastics, (b) soccer, (c) basketball, and (d) glass smashing. When using our post-training method, PhyGDPO, on Wan2.1-T2V-14B [58], the model yields more physically plausible results than OpenAI Sora2 [7] and Google Veo3.1 [20] by generating well-structured human bodies and reasonable physical interactions such as foot kicking the soccer, basketball passing through the hoop, and glass shattering.

1 Introduction

With the development of computing power and the increasing scale of training data, text-to-video (T2V) generation [7, 11, 58] has witnessed significant progress. However, accurately and consistently modeling the physics in the generated video still remains challenging and less explored. Improving the physics reasoning capability of video generation models can make them closer to real-world simulators, which can benefit a wide range of applications such as video

gaming [10, 55], autonomous driving [61, 65, 70], robotics [2, 3, 16], film making [51, 60, 66], etc.

To improve the physics modeling, graphics-based methods [31] rely on simulation engines to specify physical parameters of simple scenarios such as perfectly elastic collisions and basic rigid body dynamics. Nonetheless, it is impractical to apply them in real-world scenes as the environments are too complex to parameterize.

Another technical route [68, 73] is based on prompt extension with a large language model (LLM). These methods adopt an LLM to extend the prompts with explicit physics laws and phenomena, and then use the extended prompts as input to simulate the physics by iteratively generating the video or finetuning the model on a small subset. These methods simply follow the LLM-augmented prompts and lack the ability of thinking in physics. Instead, they use the LLM as their surrogate brain and outsource the reasoning process to it. Even so, the prompt following ability of current T2V models is still limited and current LLM’s physics reasoning ability is also weak and often erroneous, which may in turn mislead the T2V model when it follows such guidance to generate videos.

To learn the implicit physics, most current T2V foundation models are trained on massive collections of high-quality text–video data pairs. However, collecting and annotating such data is extremely costly and labor-intensive. Moreover, models obtained through such supervised fine-tuning (SFT) or training still exhibit limited physics reasoning ability. For instance, OpenAI Sora2 [7] and Google DeepMind Veo3.1 [20] often fail in complex human motions or physical phenomena, as shown in Fig. 1, 4, 5, and 6. The key reason is that there are no negative training data to provide contrastive signals discouraging physically inconsistent generations.

The emergence of direct preference optimization [53] (DPO) may provide a potential solution. However, directly using DPO may encounter three problems: (i) Lack of training data pairs that comprehensively capture physical activities, interactions, and phenomena. (ii) The correct physics preference optimization cannot be guaranteed because vanilla DPO uses generated video as winning case and the physical realism of the generated video is limited and some challenging cases remain difficult to generate with correct physics. The supervision is mainly based on the Bradley–Terry (BT) probabilistic model, which only compares a single pair of generated samples. Such single binary comparison struggles to capture inherently holistic global preference signals of physical plausibility. (iii) Vanilla DPO copies the full model as reference, which consumes substantial GPU memory and decreases efficiency.

To address these issues, we firstly propose a data construction pipeline, PhyAugPipe, that exploits a vision-language model (VLM) to filter text-video data pairs capturing rich physical interactions and phenomena from a large T2V data pool by parsing the entities and reasoning their actions with our designed chain-of-thought (CoT) [64] rule. We use PhyAugPipe to collect a training dataset, PhyVidGen-135K, containing 135K text-video pairs. Secondly, we formulate a novel Physics-aware Groupwise Direct Preference Optimization (PhyGDPO) framework for physically consistent T2V generation. PhyGDPO uses the real-world video that always follows physical laws as the winning case to guarantee correct physics learning. Different from vanilla pairwise DPO, PhyGDPO is based on the groupwise Plackett–Luce (PL) model, which captures the probability distribution over a group of candidate videos, enabling holistic preference adaptation beyond isolated pairwise comparisons. To further improve physics preference optimization, we propose a Physics-Guided Rewarding (PGR) scheme. PGR leverages a physics-aware VLM to guide data sampling and DPO training to focus on challenging actions and allows physics-violating samples to exert stronger influence. To improve DPO training efficiency and stability, we also propose a LoRA [28]-Switch Reference (LoRA-SR) scheme that does not need to copy the full model as reference like vanilla DPO, which occupies redundant GPU memory. Instead, we freeze the base model as reference and attach LoRA with an environment manager to flexibly switch between reference and action modes. Benefit from our dataset and techniques, PhyGDPO boosts the physical plausibility of the base T2V model and yields better results than the closed-source models OpenAI Sora2 [7] and Google Veo3.1 [20] on some challenging actions, as shown in Fig. 1, 4, 5, and 6.

In a nutshell, our contributions can be summarized as follows:

- • We formulate a principled DPO framework, PhyGDPO, based on the groupwise Plackett-Luce probabilistic model to capture holistic physics advantage signal for physically consistent text-to-video generation.
- • We design PGR to guide data sampling and preference optimization to focus on challenging physics. We propose LoRA-SR scheme to reduce GPU memory occupancy for more efficient and stable DPO training.
- • We present PhyAugPipe to construct physics-rich text-video pairs. We use PhyAugPipe to collect a training dataset, PhyVidGen-135K, with over 135K data pairs for studying physically consistent T2V generation.
- • Experiments demonstrate that our PhyGDPO outperforms SOTA methods on the PhyGenBench and VideoPhy2 datasets while yielding higher human preference in the user study of physical realism.

Challenging Action Categories

[Figure 25]

[Figure 26]

[Figure 27]

Element Parsing

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

Pre-sample

T2V Model

Playing Golf

Vision Checking

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

Original Prompt: a metal tool with several holes .…

Somersaulting

Original Prompt

Original Prompt: The video shows a cartoon animation of

###### …

[Figure 38]

Extended Prompt Physics Richness : 0.05

……

Extended Prompt: The

a yellow house with a red …

A golfer addresses the

A person folds a

person's hand steadily …

ball and hits the ball

damp cloth napkin

Billiard Shot

Physics Reasoning

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

Physics VLM

Folding Paper

Data Scoring

| | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |

Original Prompt: A man is

Original Prompt: Two dogs are playing with a water hose.

Original Prompt

- 0
- 1

SemanticsAdherence PhysicsCommonsense

skateboarding in a skate …

Extended Prompt Physics Richness : 0.85

Extended Prompt: The skateboarder jumps into ...

Semantics Matching

One is a golden retriever …

[Figure 49]

Score

Prompt Extending

…

PhysicsReward

[Figure 50]

[Figure 51]

…

Real Training Data

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

Self-Attention Block

HardActionRanking

x N

OriginalData Distribution

OriginalData Distribution

Number

Number

Qwen VLM

[Figure 56]

[Figure 57]

Linear Projection

Original Prompt

Original Prompt: a person

Data Sample

Extended Prompt Physics Richness : 0.88

[Figure 58]

pouring charcoal into a grill and then lighting it on fire …

…

…

Sentence Transformer

HardActionRanking

HardActionRanking

(a) T2V Data Pool

(b) CoT

(c) Physics Data Filtering

(d) Action Clustering

(e) Data Sampling with Physics Rewarding

Figure 2 Our physics-augmented video data construction pipeline (PhyAugPipe) first adopts a VLM, Qwen-2.5-72BInstruct [69], following our designed chain-of-thought (CoT) rule in (b) to select text-video data pairs that contain rich physics interactions and phenomena from a large-scale high-quality text-video data pool in (a). Then in (d), we perform action clustering on the filtered data pairs from (c) through the semantics matching via a sentence Transformer [54]. Subsequently, in (e), we adopt a physics-aware VLM, VideoCon-Physics [4], to evaluate the difficulty of different action categories and then sample the text-video pairs accordingly as the wining cases of our training data for preference optimization.

2 Method

- 2.1 Physics-Augmented Video Data Construction

Due to the scarcity of text-video data pairs with rich physics interactions and phenomena, we design a PhysicsAugmented video data construction Pipeline (PhyAugPipe). As shown in Fig. 2, we first use a VLM, Qwen2.572B-Instruct [69], equipped with our chain-of-thought (CoT) reasoning in (b) to derive physics-rich videos in (c) by filtering from a large T2V data pool in (a). Then in (d), we cluster the filtered data according to their action categories and sample the data with the rewards of a physics-aware VLM, VideoCon-Physics [4], in (e).

Data Filtering with CoT. As shown in Fig. 2 (b), Qwen2.5 with our CoT first parses the elements from the given prompts and video frames, including the physical objects with materials, actions and forces between them. Based on the parsed elements, Qwen2.5 reasons how the entities interact and what results, and rates the physics richness of the data with a score from 0 to 1. Eventually, Qwen2.5 extends the prompt with explicit physics reasoning. Although we explore the implicit physics reasoning ability of T2V models, but our dataset still include the extended prompts to support other research purposes, such as LLM-guided T2V. Please refer to Alg. 1 for the details of our CoT rules. Then in Fig. 2 (c), we threshold the data according to the estimated physics richness.

Action Clustering via Semantics Matching. After data filtering, the retained samples may exhibit distributional imbalance on different actions. To check this issue, we categorize the filtered data into semantically coherent action clusters. Specifically, in Fig. 2 (d), we first compile a list containing Ka challenging action categories. Then we feed the original text prompt of each filtered sample and the action list into a sentence Transformer [54] to perform fuzzy semantics matching, thereby determining the action category to which each data sample belongs. We count the number of data samples in each action category to obtain the distribution histogram Hf for the filtered data. Data Sampling with Physics Rewarding. As different action categories vary significantly in physical complexity thus leading to different generation difficulties for the T2V model, we propose to balance the training data accordingly. In particular, we first evaluate how well the T2V model performs on each action category. Based on the performance distribution, we adjust the sampling ratio by allocating more samples to the categories where the model performs poorly. As shown in Fig. 2 (d), we first select the top-nc samples with the highest semantics matching scores within each category as its representatives. Then we employ a physics-aware VLM, VideoCon-Physics [4], to evaluate each video representative. VideoCon-Physics outputs a semantics adherence score and a physics commonsense score ∈ [0,1] to measure the physics plaucibility of the video. We average the two scores to obtain the overall score for each video. Then we compute the mean score of all representatives within each category to obtain the

#### Algorithm 1 Brief Pipeline of our Chain-of-Thoughts Prompts Require: Original prompt p, video frames V Ensure: JSON object containing: original, parse, reason, extended, physics_richness, physics_label

- 1: Step 1: Element Parsing
- 2: Extract entities, actions, forces, and outcomes from (p,V ) using the VLM.
- 3: Ensure all agents are included and avoid unsupported or speculative items.
- 4: Save results into the dictionary parse.
- 5: Step 2: Vision Checking
- 6: Compare parse with video frames and the original prompt.
- 7: Remove hallucinated elements and add missing visible entities or interactions.
- 8: Update parse accordingly.
- 9: Step 3: Physics Reasoning
- 10: Produce a concise explanation describing how the parsed entities interact through physical forces and lead to observed outcomes.
- 11: Store the explanation as reason.
- 12: Step 4: Data Scoring
- 13: Compute physics_richness ∈ [0,1] based on:

- – number and interaction of entities;
- – presence of explicit forces and outcomes;
- – causal clarity in reason;
- – penalties from camera motion, stylization, or static-aftermath filters.

- 14: Set physics_label = 1 if physics_richness ≥ 0.60, otherwise 0.
- 15: Step 5: Prompt Extending
- 16: Extend the original prompt by inserting causal physical details based on reason, without adding new entities, forces, or sensory descriptions.
- 17: Limit the extension to at most 100 words and store it as extended.
- 18: return JSON object containing all computed fields.

category score. In this way, we derive the score distribution histogram Sf. Then we sample the data according to the difficulty of each action category. More specifically, we define the difficulty of the k-th action category as dk = 1 − Sf(k) and assign a sampling weight following an exponential form as rk = exp(τdk), where τ is a hyperparameter that controls how strongly the sampling favors difficult action categories. Given the total sampling

budget N, the number of data pairs sampled from the k-th action category is Hr(k) = min Hf(k), N · r

k Ka j=1 rj , where Hr denotes the distribution histogram after data sampling with physics rewarding. This sampling strategy allocates more data to challenging actions, encouraging the model to learn more complex physics during training.

Please note that our PhyAugPipe is a filtering-based data pipeline. The VLMs are used to select physics-rich scenes and hard actions from the real-world videos, which always follow the physical laws. Therefore, the VLM rewarding only impacts the physics richness rather than correctness. We use the original human-written prompts instead of the VLM rewritten ones to avoid the potential bias or errors of VLMs.

- 2.2 Physics-Aware Groupwise Direct Preference Optimization

Existing DPO algorithms are based on the Bradley–Terry (BT) probabilistic model, which compares pairwise data samples, showing limitations in capturing global and holistic preference signals and aligning with human feedback. Besides, vanilla DPO uses generated video as winning case but the physical realism of the generated video is limited and some challenging physics actions are difficult to generate. To address these issues, we formulate a Physics-Aware Groupwise Direct Preference Optimization (PhyGDPO), as illustrated in Fig. 3.

Groupwise Probability. We denote the reward as r(c,x0) with the generation x0 and condition c. Different from the

SA Score

SA Score

[Figure 59]

Groupwise Plackett-Luce (PL) probabilistic model

Physics VLM

|[Figure 60]<br><br>❄ 🔥 Trainable Weights Fixed Weights<br><br>[Figure 61]|
|---|

PC Score 𝑥

PC Score

𝑥

[Figure 62]

🔥

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

❄ LoRA

[Figure 67]

[Figure 68]

[Figure 69]

❄

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

Decoder

Prompts

Encoder

[Figure 79]

❄

Backbone

𝑥 SA Score: 0.8 PC Score: 0.6 𝑥 SA Score: 0.6 PC Score: 0.6 𝑥 SA Score: 0.6 PC Score: 0.8 𝑥 SA Score: 1.0 PC Score: 1.0

Text-to-Video Generation Model

Generated Data of Di erent Random Seeds – Losing Samples

Real Data – Winning Sample

(c) Groupwise Direct Preference Optimization with Physics-Guided Rewarding

(a) LoRA-Switch Reference

- Figure 3 Overview of PhyGDPO. (a) The LoRA-Switch Reference scheme can flexibly toggle between the reference and action modes for DPO advantage computation to save the GPU memory and increase the training stability. (b) our DPO framework is based on the groupwise Plackett-Luce (PL) probabilistic model and adopts a physics-aware VLM, VideoCon-Physics [4], to reward the DPO training. The winning case is the real-world video because it always captures correct physics as a guarantee.

normal DPO, we start from the groupwise Plackett-Luce (PL) probabilistic model, as shown in Fig. 3 (b). We adopt the real video as the winning case xw0 because it always follows physical laws and a set of generated videos as the losing cases Gl(c) = {xl

0 ,...,xl

0 }. The preference probability of the PL model is formulated as

1

m

exp r(c, xw0 ) exp r(c, xw0 ) + mj=1 exp r(c, xl0j)

pPL xw0 | Gl(c), c =

. (1)

r(c,x0) can be parameterized by a neural network ϕ [53] and estimated via maximum likelihood training as

exp rϕ(c,xw0 ) exp rϕ(c,xw0 ) + mj=1 exp rϕ(c,xl0j)

. (2)

LPL(ϕ) = −Ec,Gl(c) log

Vanilla DPO [39, 57] represents r(c,x0) by the T2V model itself as r(c,x0) = β log

p∗θ(x0|c) pψ(x0|c)

+ β log Z(c), (3)

where pθ and pψ denote the conditional probabilities of trained model θ and reference model ψ. p∗θ and Z(c) are the unique global optimal solution and the partition function. We plug Eq. (3) into Eq. (2) and drop the group-constant

term β log Z(c) as it does not affect the optimization direction to obtain the groupwise DPO (GDPO) loss as

exp βfθ(xw0 ,c) exp βfθ(xw0 ,c) + mj=1 exp βfθ(xl0j,c)

LGDPO(θ) = −Ec,Gl(c) log

exp βfθ(xw0 ,c) + mj=1 exp βfθ(xl0j,c) exp βfθ(xw0 ,c)

(4)

= Ec,Gl(c) log

m

exp β(fθ(xl0j,c) − fθ(xw0 ,c)) .

= Ec,Gl(c) log 1 +

j=1

where we define the function fθ(x0,c) as the log-likelihood ratio as

T

T

k−1 | xt

,c) pψ(xt

pθ(xt

pθ(x0|c) pψ(x0|c)

≜

∆k. (5)

k

fθ(x0,c) = log

=

log

k−1 | xt

,c)

k

k=1

k=1

Here the terminal distribution is inherently represented as the joint transition process over multiple timesteps tk by the definition of diffusion or flow matching models. Subsequently, we reformulate Eq. (4) and estimate its upper bound using the Jensen’s inequality into a single timestep for more efficient training as

m

exp βTEk[∆lkj − ∆wk ]

LGDPO(θ) = Ec,Gl(c) log 1 +

j=1

(6)

m

exp βT[∆lkj − ∆wk ] .

≤ Ec,Gl(c),k log 1 +

j=1

(a) A gymnast stands on a balance beam, then performs a forward roll, landing smoothly.

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

(b) A player rides their horse, preparing to strike the ball, their mallet poised.

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

VideoDPO PhyT2V Wan2.1-14B OpenAI Sora2 Google Veo3.1 PhyGDPO (Ours)

- Figure 4 Results of two challenging actions (gymnastics and polo) on VideoPhy2. Our method generates more physically consistent videos, showing coherent, deformation-free gymnastic movements and realistic ball–mallet striking interactions.

Although the upper bound in Eq. (6) allows single-timestep training, it needs to infer the model 2m + 2 times for each group, which requires a long time and large GPU memory. To handle this issue, we exploit an inequality as

m

ex

1 +

j

j=1

m

≤

j=1

jxj γj, 0 < αj ≤ 1, γj ≥ 1/αj. (7)

1 + eα

Please refer to the supplementary for the detailed proof process of this inequality. Subsequently, we can further formulate the upper bound of LGDPO(θ) via Ineq. (7) into a single sample within each group for efficient training as

m

γj log 1 + exp −αjβT[∆wk − ∆lkj])

LGDPO(θ) ≤ Ec,Gl(c),k

(8)

j=1

= Ec,Gl(c),k,j − mγj log σ αjβT(∆wk − ∆lkj) .

This upper bound allows efficient training with single data pair sampling in a single timestep for each iteration.

Physics-Guided Rewarding. To further improve physics preference optimization, we design a Physics-Guided Rewarding (PGR) to direct DPO training to focus on challenging physics cases. More specifically, we parameterize γj and αj in Eq. (8) by the normalized semantics adherence score ssaj ∈ [0,1] and physics commonsense score spcj ∈ [0,1] predicted by a physics-aware VLM, VideoCon-Physics, as

ssaj + spcj 2

1 + λ · σ κγ(vj − bγ) αmin

vj = 1 −

, αj = αmin + (1 − αmin) · σ κα(vj − bα) ,

, γj =

(9)

where vj measures the physics difficulty and mainly modulates γj and αj, αmin sets a lower bound to avoid vanishing gradients, and the sigmoid function σ(·) ensures a bounded and smooth adjustment of γj, stabilizing the optimization. αj adaptively adjusts the sharpness of the preference comparison, while γj amplifies the physicsguided reward for samples with lower physical plausibility. They are controlled by the hyperparameters λ, κα,γ, bα,γ, and αmin. Our designed physics rewards (αj,γj) balances the training stability with adaptivity, allowing physics-violating samples to exert stronger influence during optimization. Please note that the VLM rewarding only affects the importance of samples while the correctness of physical supervision is guaranteed by the core objective,

i.e., the DPO advantage of winning case (real-world video capturing correct physics) over losing case. Flow Matching Probabilistic Modeling. Our method is based on the standard rectified flow matching as

xt = (1 − t)x0 + tx1, x1 ∼ N(0,I), (10)

(a) A person plays squash on an indoor court.

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

(b) A person wearing a helmet performs a handspring over a platform.

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

[Figure 127]

VideoDPO PhyT2V Wan2.1-14B OpenAI Sora2 Google Veo3.1 PhyGDPO (Ours)

- Figure 5 Comparison on two challenging random user-input actions (squash and handspring). Our method generates more physically plausible videos, capturin racket–ball interactions in squash and well-coordinated body motion in handspring.

which induces the oracle velocity u∗(xt,t | c) = x1 − x0. Discretizing time with step size h (so t  → t − h), the backward update is xt−h = xt−hu∗(xt,t | c). Since the rectified flow one-step reverse transition has no intermediate noise, we follow a vanishing-noise regularization and approximate pθ(xt−h | xt,c) and pψ(xt−h | xt,c) as

pθ(xt−h | xt,c) ≈ N xt − hvθ(xt,t,c), εI , pψ(xt−h | xt,c) ≈ N xt − hvψ(xt,t,c), εI ,

(11)

then let ε → 0 and use the log-ratio between Gaussians with the identity (xt−h − xt)/h = −u∗(xt,t | c), we obtain

h2 2ε

pθ(xt−h | xt,c) pψ(xt−h | xt,c) ≈ −

ℓθ(xt,t) − ℓψ(xt,t) ,

log

ℓθ(xt,t) = ∥vθ(xt,t,c) − u∗(xt,t | c)∥22, ℓψ(xt,t) = ∥vψ(xt,t,c) − u∗(xt,t | c)∥22,

and reformulate the upper bound of Eq. (8) into the final overall training objective as

(12)

L = Ec,Gl(c),k,j − mγj log σ − αjβT[(ℓwθ − ℓwψ) − (ℓlθj − ℓlψj)] , (13)

where any global constant scale factor such as h2/2ε in Eq. (12) can be absorbed into the hyperparameter β. For simplicity, we denote ℓwθ = ℓθ(xwt

#### | tk), ℓlθj = ℓθ(xltj

#### | tk), and ℓlψj = ℓψ(xltj

#### | tk), ℓwψ = ℓψ(xwt

| tk) in Eq. (13).

k

k

k

k

LoRA-Switch Reference. Previous DPO methods usually copy the full model and fix it as the reference, which occupies redundant GPU memory, degrades computational efficiency, and hinders the scalability of model size. Plus, this full-copy strategy often leads to unstable and less effective DPO training because it updates the entire set of the weights and may cause the action model to quickly and dramatically deviate from the reference model.

To address this issue, we design a LoRA-Switch Reference (LoRA-SR) scheme. As shown in Fig. 3 (b), we freeze the backbone as the reference model ψ and attach trainable LoRA modules to ψ as the action model θ. Specifically, we attach LoRA to the query, key, value, and output linear projection layers of each self-attention in the Transformer backbone of the T2V model and implement an environment manager as the switch to flexibly toggle between the reference and action modes. Given the input tokens of the linear projection layer where the LoRA is attached to as X ∈ RB×L×C

in, the output tokens Y ∈ RB×L×C

in of our LoRA-SR scheme can be formulated as

α r

Y = X(W + 1action · ∆W)⊤ = X(W + 1action ·

BA)⊤, (14)

Methods Hard Activity Interaction Overall Vcrafter2 [12] 0.0222 0.1071 0.0943 0.1034 Wan2.1-14B [58] 0.0111 0.1190 0.1572 0.1288 Hunyuan [35] 0.0222 0.1286 0.1572 0.1356 VideoDPO [39] 0.0167 0.1310 0.1572 0.1373 PhyT2V [68] 0.0389 0.1405 0.1698 0.1492 Sora2 [7] 0.0389 0.1429 0.1698 0.1508 Veo3.1 [20] 0.0444 0.1405 0.1887 0.1525 PhyGDPO (Ours) 0.0500 0.1571 0.1761 0.1627

(a) Quantitative results on VideoPhy2

Methods Mechanics Optics Thermal Material Avg Lavie [62] 0.40 0.44 0.38 0.32 0.36 Wan2.1-14B [58] 0.36 0.53 0.36 0.33 0.40 Open-Sora [77] 0.43 0.50 0.44 0.37 0.44 Pika [50] 0.35 0.56 0.43 0.39 0.44 Vchitect-2.0 [21] 0.41 0.56 0.44 0.37 0.45 PhyT2V [68] 0.45 0.55 0.43 0.53 0.50 VideoDPO [39] 0.48 0.60 0.47 0.58 0.54 PhyGDPO (Ours) 0.55 0.60 0.58 0.47 0.55

Compared Methods Preference of Ours

Vcrafter2 [12] 94.2 % VideoDPO [39] 89.4 % PhyT2V [68] 88.5 % Wan2.1-14B [58] 86.5 % Hunyuan [35] 82.7 % Sora2 [7] 67.3 % Veo3.1 [20] 64.4 % PhyGDPO (Ours) –

(b) Quantitative results on PhyGenBench

(c) User preference of our method

- Table 1 Quantitative comparison and user preference (%) of our method on VideoPhy2 [5] and PhyGenBench [45] datasets.

(a) A wooden pencil is carefully dipped into a glass of crystal-clear water, …

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

(c) A small burning ball of paper was thrown into a pile of dry paper.

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

[Figure 150]

[Figure 151]

VideoDPO PhyT2V Wan2.1-14B OpenAI Sora2 Google Veo3.1 PhyGDPO (Ours)

- Figure 6 Results of two challenging physics phenomena (water refraction and paper combustion) on PhyGenBench [45]. Our method can model the magnifying convex-lens effect and the realistic light refraction caused by the curved water surface, as well as the physically correct ignition and flame propagation on paper, demonstrating stronger physics reasoning ability.

where W ∈ RC

out×Cin is the frozen weight of the linear layer, 1action ∈ {0,1} is a binary indicator governed by the environment manager to switch between the reference (1action = 0) and action (1action = 1) modes, α is the scaling factor, r is the rank, and B ∈ RC

in are the learnable LoRA parameters. Our LoRA-SR enables the reference and trainable models to share the same heavy backbone while flexibly switching the lightweight LoRA parameters for the action and reference models. Thus, we can compute the DPO advantage without storing or loading an extra full model. This significantly increases computational efficiency, improves the scalability of model size, and prevents the model from drifting too far from reference, thus stabilizing the training process.

out×r and A ∈ Rr×C

3 Experiment

Dataset. We first use PhyAugPipe to process one million high-quality text-video pairs. After physics data filtering in Fig. 2 (c), we obtain 135K data pairs with sufficient physics interaction and phenomena. Then we perform action clustering and data resampling with physics rewarding to derive 17K data pairs from the 135K filtered data samples. Subsequently, we use the pre-trained T2V model Wan2.1-14B to generate videos for the 17K text prompts with different random seeds and use VideoCon-Physics [4] to score them. We adopt the short unextended prompts from the two datasets, VideoPhy2 [5] and PhyGenBench [45], for evaluation.

Evaluation Metrics. VideoPhy2 uses another VLM - VideoPhy2-AutoRater [5], which is different from VideoConPhysics [4], to evaluate the semantics adherence and physics commonsense of the videos with scores ranging from 1 to 5, and then compute the ratio with both scores ≥ 4. PhyGenBench evaluation is based on GPT-4o [1], CLIP [52], InternVideo2 [63], and VideoLLaVA [37]. It firstly detects key physical phenomena required by the prompt and then assesses their order and naturalness, covering 27 physical laws in 4 basic physics domains.

Method Hard Activity Interaction Overall Baseline 0.0333 0.1405 0.1635 0.1475 + Chain-of-Thought 0.0389 0.1452 0.1698 0.1525 + Action Clustering 0.0500 0.1524 0.1698 0.1575 + Physics Rewarding 0.0500 0.1571 0.1761 0.1627

(a) Break-down ablation of PhyAugPipe on Wan2.1-14B

Method Hard Activity Interaction Overall Baseline 0.0118 0.1136 0.1447 0.1232 Flow-DPO [38] 0.0296 0.1247 0.1342 0.1283 VideoDPO [39] 0.0278 0.1262 0.1509 0.1305 PhyGDPO (Ours) 0.0444 0.1357 0.1635 0.1407

Method Hard Activity Interaction Overall Baseline 0.0111 0.1190 0.1572 0.1288 + LoRA-SR 0.0278 0.1381 0.1635 0.1458 + Groupwise Model 0.0389 0.1500 0.1698 0.1559 + Physics Rewarding 0.0500 0.1571 0.1761 0.1627

##### (b) Break-down ablation study of PhyGDPO on Wan2.1-14B

Method GPU Memory Storage Space Hard Score Overall Score

Baseline - - 0.0118 0.1232 LoRA-SFT 24.7GB 84MB 0.0167 0.1283 w/o LoRA-SR 48.7GB 5.3GB 0.0389 0.1373 with LoRA-SR 25.3GB 84MB 0.0444 0.1407

(c) Comparison with SOTA DPO methods on Wan2.1-1.3B

##### (d) Ablation of LoRA-SR mechanism on Wan2.1-1.3B

Method PhyT2V VideoDPO Wan2.1-14B Veo3.1 Sora2 PhyGDPO Overall 0.3186 0.3288 0.4119 0.5034 0.5373 0.5525

Method Vcrafter2 + Flow-DPO + VideoDPO + PhyGDPO Overall 0.1034 0.1128 0.1373 0.1426

(e) Cross-VLM evaluation with Gemini-2.5-pro

(f) Cross-T2V-model evaluation

- Table 2 Ablation study on VideoPhy2 [5]. (a) and (b) study the components of PhyAugPipe and PhyGDPO towards better performance. (c) compares VideoDPO with PhyGDPO with the same settings. (d) studies the effect of LoRA-SR scheme. (e) and (f) are cross evaluation by changing the VLM scorer to Gemini-2.5-pro [17] and T2V base model to Vcrafter2 [12].

A tennis ball is gently placed on the surface of a bucket filled with water.

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

(a) Baseline T2V Model Wan2.1-14B (b) Baseline + LoRA-SR

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

(c) Baseline + LoRA-SR + Groupwise Model (d) Baseline + LoRA-SR + Groupwise Model + Physics Rewarding

- Figure 7 Break-down visual analysis of PhyGDPO. When we progressively apply LoRA-SR, groupwise DPO modeling, and physics rewarding to Wan2.1-14B, the tennis ball in the generated video increasingly conforms to the physical law of buoyancy, floating stably on the water surface instead of sinking down or bouncing unrealistically above it.

Implementation Details. We implement our PhyGDPO by pytorch [48] based on Wan2.1-14B. Our model is finetuned for 10K steps in total at a batch size of 8 on 8 H100 GPUs for 6 days. To save GPU memory, we adopt mixed-precision training [47] with BF16 and sublinear memory training [15]. We adopt the AdamW optimizer [41] (β1 = 0.9, β2 = 0.999) with a weight decay of 0.01. The learning rate is initially set as 1e−5 and decays to 1e−6 using cosine annealing [40] algorithm. The training and inference spatial resolution of the video is 480×832. We set τ = 3, N = 20000, αmin = 0.5, kγ = 2.0, bγ = 0.4, λ = 0.6, kα = 5.0, bα = 0.5. The rank and scale factor are 48.

- 3.1 Comparison with State-of-the-Art Methods

Quantitative Results. We compare PhyGDPO with state-of-the-art (SOTA) methods in Tab. 1a and 1b. PhyGDPO outperforms SOTA methods by large margins. (i) Our method surpasses the two recent strongest closed-source models, Sora2 [7] and Veo3 [20] across the tracks of hard actions, activities and sports, and the overall score on VideoPhy2. Especially on the hard actions, PhyGDPO achieves 4.5× higher score than the base model, Wan2.1-14B, and is 29% and 13% higher than Sora2 and Veo3. (ii) Compared to the SOTA DPO algorithm for video generation (VideoDPO) and SOTA method for physically consistent video generatiion (PhyT2V), our method surpasses VideoDPO and PhyT2V and by 200% and 29% on the hard action score of VideoPhy2. Our method is also 22%/15% and 35%/23% higher than PhyT2V/VideoDPO on the mechanics and thermal tracks of PhyGenBench.

User Study. Since the auto-evaluation tools of VideoPhy2 and PhyGenBench are based on VLMs, their assessment of physical plausibility may not be perfect. Thus, we conduct a user study with 104 participants. Each participant is asked to pick up the video that better follows the physical laws. Tab. 1c reports the user preference (%) of our method over competing entries. Each participant completes 48 trials (one prompt per trial), with 24 prompts randomly sampled from VideoPhy2 and 24 from PhyGenBench. Each trial randomly selects one of the 8 baselines

αmin 0.3 0.4 0.5 0.6 0.7 Overall 0.1582 0.1591 0.1627 0.1615 0.1608

(a) Analysis of αmin

kα 4.0 4.5 5.0 5.5 6.0 Overall 0.1611 0.1579 0.1627 0.1593 0.1618

(b) Analysis of kα

bα 0.3 0.4 0.5 0.6 0.7 Overall 0.1582 0.1579 0.1627 0.1598 0.1602

##### (c) Analysis of bα

λ 0.3 0.4 0.5 0.6 0.7 Overall 0.1592 0.1586 0.1608 0.1627 0.1615

kγ 1.0 1.5 2.0 2.5 3.0 Overall 0.1588 0.1609 0.1627 0.1592 0.1598

bγ 0.3 0.4 0.5 0.6 0.7 Overall 0.1586 0.1627 0.1582 0.1607 0.1595

(d) Analysis of λ

(e) Analysis of kγ

(f) Analysis of bγ

- Table 3 Analysis of the hyperparameters in the physics rewarding of Eq.(9) on the VideoPhy2 [5] benchmark. When tweaking a hyperparameter to analyze its effectiveness, other hyperparameters are fixed at their optimal values.

A weightlifter completes a snatch with a 25kg barbell, holding it momentarily overhead.

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

(a) Baseline T2V Model Wan2.1-1.3B (b) Wan2.1-1.3B + VideoDPO

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

(c) Wan2.1-1.3B + Flow-DPO (d) Wan2.1-1.3B + PhyGDPO

- Figure 8 Visual comparison with state-of-the-art DPO algorithms on Wan2.1-T2V-1.3B under the same training settings and data for fairness. Using our PhyGDPO can generate more physically plausible video with undistorted human body motion.

(non-repeated) in Tab. 1c and compares it to our method given the same prompt. Video results are presented anonymously in random order. All results in Tab. 1c have 95% confidence intervals no wider than ±2.4%. PhyGDPO is preferred by human evaluators, indicating that PhyGDPO learns genuine physics beyond VLM agreement.

Qualitative Results. The visual comparisons on challenging physical actions or phenomena including gymnastics, soccer, basketball, glass smashing, polo, squash, handspring, refraction, and combustionare shown in Fig. 1, 4, 5, and 6. Our method generates videos with more realistic physical dynamics, deformation-free body motion, and accurate object interactions, effectively capturing phenomena such as force transfer, material deformation, light refraction, and flame propagation. Compared to existing open- and closed-source models, PhyGDPO shows superior generalization from human activities to complex physical events. Please refer to the project page for video results.

- 3.2 Ablation Study

Ablation of PhyAugPipe. We conduct a break-down ablation to study the effect of each component of PhyAugPipe in Tab. 2a. The baseline is directly training PhyGDPO on the randomly selected text-video data. When we keep the number of selected training data the same and apply the data filtering with CoT, action clustering with semantics matching, and data sampling with physics-guided rewarding, all track scores are improved. Especially the score on the hard actions gains by over 50%. These results suggest the effectiveness of our data construction techniques.

Ablation of PhyGDPO. Tab. 2b shows a break-down ablation for PhyGDPO. When progressively using LoRA-SR, PL groupwise probabilistic model, and physics-guided rewarding, the performance steadily gains by large margins. The score on the hard action categories yields 4.5× improvement, demonstrating the effectiveness of our designed methods in improving T2V physics plausibility. In addition, we conduct a visual analysis in Fig. 9 (a) for PhyGDPO. When gradually using our proposed techniques, the tennis in the generated video no longer presents ghosting artifacts, and its floating motion on the water surface better conforms to the physical laws of fluid buoyancy.

PhyGDPO vs. SOTA DPO. For fair comparison with two SOTA DPO methods: Flow-DPO [38] and VideoDPO, we use them to finetune Wan2.1-1.3B with the same data and settings. As reported in Tab. 2c, PhyGDPO surpasses Flow-DPO and VideoDPO across all tracks by large margins, especially on the hard action track, where it yields 50% improvement. We conduct a visual comparison in Fig. 9 (b) on the challenging weightlifting action. Flow-DPO and VideoDPO generate distorted or unstable body poses. In contrast, PhyGDPO generates coherent body motion with stable shapes and balanced force dynamics, highlighting its advantage in modeling complex physical actions.

Analysis of LoRA-SR. We conduct an ablation study of LoRA-SR in Tab. 2d. In our PhyGDPO training, using LoRA-SR reduces GPU memory consumption by 44% and achieves over 60× compression in storage size, while still improving the score on hard actions by 14% compared to the version without LoRA-SR. We also conduct a

A person shovels snow from a garden path, leaving a clear path.

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

(a) Wan2.1-T2V-1.3B + PhyGDPO w/o LoRA-SR (b) Wan2.1-T2V-1.3B + PhyGDPO with LoRA-SR

- Figure 9 Visual analysis of our LoRA-SR scheme. Applying our LoRA-SR can generate more accurate and plausible hand–shovel interaction in the snow shoveling scenario. In contrast, removing LoRA-SR generates distorted hand actions.

visual analysis in Fig. 9 (c), using our LoRA-SR in DPO training can generate more plausible physical interactions between human hands and the snow shovel. Besides, we also compare with SFT using the same LoRA in Tab. 2d. Our PhyGDPO with LoRA-SR uses almost the same GPU memory but surpasses LoRA-SFT by large margins, especially on the hard-action categories, where the score achieves 2.7× improvement.

Cross Evaluation. (i) To provide stronger evidence that our method improves the physical plausibility, we conduct a cross-VLM evaluation. We replace the VLM scorer of VideoPhy2 by a stronger physics evaluator, Gemini-2.5pro [17], to score the generated videos with the same protocol in Tab. 2e. Our method still outperforms SOTA T2V models, showing that our method learns better genuine physics. (ii) We change the T2V model to Vcrafter2 [12] and apply different DPO methods on it with the same setting for fairness in Tab. 2f. PhyGDPO consistently surpasses SOTA DPO methods, suggesting the superiority and generalization ability of our method.

Parameter Analysis. We analyze the effects of the hyperparameters in Eq. (9). When tweaking a hyperparameter, other hyperparameters are fixed at their optimal values. The results are shown in Tab. 3. The best performance is achieved when αmin = 0.5, kα = 5.0, bα = 0.5, λ = 0.6, kγ = 2.0, and bγ = 0.4. As shown in Tab. 2a, when we do not use the physics-guided rewarding, the model achieves 0.1575 on the overall score. And across all hyperparameter configurations of Tab. 3, our method consistently achieves performance gains.

- 4 Related Work

- 4.1 Text-to-Video Generation Models

T2V generation [6, 6, 7, 19, 25–27, 29, 62, 72, 75, 78] has witnessed significant progress. Many existing works follow DiT [49] to adopt a Transformer [56] to predict the noise in diffusion [9, 22, 24, 34, 36, 42–44, 51] or estimate the velocity field in flow matching [8, 14, 19, 23, 29, 30, 33, 58, 59, 74, 76]. Although high visual quality is achieved, accurately modeling the underlying physics-related effects remains challenging. Two recent works, DiffPhy [73] and PhyT2V [68], adopt an LLM to extend the text prompt with explicit physics laws and phenomena, and then iteratively generates the video or finetunes a T2V model with extended prompts. Yet, these prompt extension-based methods are easily misled by the mistakes of LLM and struggle to learn implicit physics. We aim to fill this gap.

- 4.2 Direct Preference Optimization

Direct Preference Optimization (DPO) [53] is proposed to align LLMs with human preferences and has been adopted in image [18, 32, 46, 57] and video [13, 38, 39, 67, 71] generation. For example, DiffusionDPO [57] finetunes a text-to-image diffusion model with human-annotated preference image pairs. VideoDPO [39] adapts DiffusionDPO into T2V generation. However, these DPO frameworks mainly improve the visual quality and aesthetics. In addition, these DPO methods suffer from a severe low-efficiency problem because they need to copy a full model as the reference, which occupies redundant GPU memory. Our work focused on solving these research problems.

- 5 Conclusion

In this paper, we focus on studying a challenging problem, physically consistent T2V generation, without using LLM for prompt extension in inference because our goal is to explore the implicit physics reasoning ability of the video generation model. To this end, we first propose a data construction method, PhyAugPipe, to collect a training dataset PhyVidGen-135K containing over 135K text-video data pairs with rich physics interaction and phenomena. Subsequently, we formulate a principled groupwise DPO framework, PhyGDPO, with two technical designs PGR and LoRA-SR. PhyGDPO efficiently post-trains Wan2.1-T2V-14B model on PhyVidGen-135K to boost the physics plausibility in video generation. Comprehenseiv experiments show that our method quantitatively and qualitatively outperforms SOTA algorithms and achieves higher human preference in the user study.

References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.
- [2] Niket Agarwal, Arslan Ali, Maciej Bala, Yogesh Balaji, Erik Barker, Tiffany Cai, Prithvijit Chattopadhyay, Yongxin Chen, Yin Cui, Yifan Ding, et al. Cosmos world foundation model platform for physical ai. arXiv preprint arXiv:2501.03575, 2025.
- [3] Hassan Abu Alhaija, Jose Alvarez, Maciej Bala, Tiffany Cai, Tianshi Cao, Liz Cha, Joshua Chen, Mike Chen, Francesco Ferroni, Sanja Fidler, et al. Cosmos-transfer1: Conditional world generation with adaptive multimodal control. arXiv preprint arXiv:2503.14492, 2025.
- [4] Hritik Bansal, Zongyu Lin, Tianyi Xie, Zeshun Zong, Michal Yarom, Yonatan Bitton, Chenfanfu Jiang, Yizhou Sun, Kai-Wei Chang, and Aditya Grover. Videophy: Evaluating physical commonsense for video generation. In ICLR, 2025.
- [5] Hritik Bansal, Clark Peng, Yonatan Bitton, Roman Goldenberg, Aditya Grover, and Kai-Wei Chang. Videophy-2: A challenging action-centric physical commonsense evaluation in video generation. In ICMLW, 2025.
- [6] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023.
- [7] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, et al. Video generation models as world simulators. OpenAI Blog, 2024.
- [8] Yuanhao Cai, He Zhang, Xi Chen, Jinbo Xing, Yiwei Hu, Yuqian Zhou, Kai Zhang, Zhifei Zhang, Soo Ye Kim, Tianyu Wang, et al. Omnivcus: Feedforward subject-driven video customization with multimodal control conditions. In NeurIPS, 2025.
- [9] Yuanhao Cai, He Zhang, Kai Zhang, Yixun Liang, Mengwei Ren, Fujun Luan, Qing Liu, Soo Ye Kim, Jianming Zhang, Zhifei Zhang, Yuqian Zhou, Yulun Zhang, Xiaokang Yang, Zhe Lin, and Alan Yuille. Baking gaussian splatting into diffusion denoiser for fast and scalable single-stage image-to-3d generation and reconstruction. In ICCV, 2025.
- [10] Haoxuan Che, Xuanhua He, Quande Liu, Cheng Jin, and Hao Chen. Gamegen-x: Interactive open-world game video generation. In ICLR, 2025.
- [11] Haoxin Chen, Menghan Xia, Yingqing He, Yong Zhang, Xiaodong Cun, Shaoshu Yang, Jinbo Xing, Yaofang Liu, Qifeng Chen, Xintao Wang, et al. Videocrafter1: Open diffusion models for high-quality video generation. arXiv preprint arXiv:2310.19512, 2023.
- [12] Haoxin Chen, Yong Zhang, Xiaodong Cun, Menghan Xia, Xintao Wang, Chao Weng, and Ying Shan. Videocrafter2: Overcoming data limitations for high-quality video diffusion models. In CVPR, 2024.
- [13] Harold Haodong Chen, Haojian Huang, Qifeng Chen, Harry Yang, and Ser-Nam Lim. Hierarchical fine-grained preference optimization for physically plausible video generation. In NeurIPS, 2025.
- [14] Shoufa Chen, Chongjian Ge, Yuqi Zhang, Yida Zhang, Fengda Zhu, Hao Yang, Hongxiang Hao, Hui Wu, Zhichao Lai, Yifei Hu, et al. Goku: Flow based video generative foundation models. In CVPR, 2025.
- [15] Tianqi Chen, Bing Xu, Chiyuan Zhang, and Carlos Guestrin. Training deep nets with sublinear memory cost. arXiv preprint arXiv:1604.06174, 2016.
- [16] Xiaoyu Chen, Junliang Guo, Tianyu He, Chuheng Zhang, Pushi Zhang, Derek Cathera Yang, Li Zhao, and Jiang Bian. Igor: Image-goal representations are the atomic control units for foundation models in embodied ai. arXiv preprint arXiv:2411.00785, 2024.
- [17] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.
- [18] Florinel-Alin Croitoru, Vlad Hondru, Radu Tudor Ionescu, Nicu Sebe, and Mubarak Shah. Curriculum direct preference optimization for diffusion and consistency models. In CVPR, 2025.
- [19] Aram Davtyan, Sepehr Sameni, and Paolo Favaro. Efficient video prediction via sparsely conditioned flow matching. In ICCV, 2023.
- [20] Google DeepMind. Veo-3: A scalable and controllable text-to-video model. https://storage.googleapis.com/ deepmind-media/veo/Veo-3-Tech-Report.pdf, 2025. Technical Report, Google DeepMind.

- [21] Weichen Fan, Chenyang Si, Junhao Song, Zhenyu Yang, Yinan He, Long Zhuo, Ziqi Huang, Ziyue Dong, Jingwen He, Dongwei Pan, et al. Vchitect-2.0: Parallel transformer for scaling up video diffusion models. arXiv preprint arXiv:2501.08453, 2025.
- [22] Peng Gao, Le Zhuo, Dongyang Liu, Ruoyi Du, Xu Luo, Longtian Qiu, Yuhang Zhang, Chen Lin, Rongjie Huang, Shijie Geng, et al. Lumina-t2x: Transforming text into any modality, resolution, and duration via flow-based large diffusion transformers. arXiv preprint arXiv:2405.05945, 2024.
- [23] Yu Gao, Haoyuan Guo, Tuyen Hoang, Weilin Huang, Lu Jiang, Fangyuan Kong, Huixia Li, Jiashi Li, Liang Li, Xiaojie Li, et al. Seedance 1.0: Exploring the boundaries of video generation models. arXiv preprint arXiv:2506.09113, 2025.
- [24] Agrim Gupta, Lijun Yu, Kihyuk Sohn, Xiuye Gu, Meera Hahn, Fei-Fei Li, Irfan Essa, Lu Jiang, and José Lezama. Photorealistic video generation with diffusion models. In ECCV, 2024.
- [25] Yingqing He, Tianyu Yang, Yong Zhang, Ying Shan, and Qifeng Chen. Latent video diffusion models for high-fidelity long video generation. arXiv preprint arXiv:2211.13221, 2022.
- [26] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, et al. Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303, 2022.
- [27] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video diffusion models. NeurIPS, 2022.
- [28] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. In ICLR, 2022.
- [29] Yang Jin, Zhicheng Sun, Ningyuan Li, Kun Xu, Hao Jiang, Nan Zhuang, Quzhe Huang, Yang Song, Yadong Mu, and Zhouchen Lin. Pyramidal flow matching for efficient video generative modeling. In ICLR, 2025.
- [30] Xuan Ju, Tianyu Wang, Yuqian Zhou, He Zhang, Qing Liu, Nanxuan Zhao, Zhifei Zhang, Yijun Li, Yuanhao Cai, Shaoteng Liu, et al. Editverse: Unifying image and video editing and generation with in-context learning. arXiv preprint arXiv:2509.20360, 2025.
- [31] Bingyi Kang, Yang Yue, Rui Lu, Zhijie Lin, Yang Zhao, Kaixin Wang, Gao Huang, and Jiashi Feng. How far is video generation from world model: A physical law perspective. In ICML, 2025.
- [32] Shyamgopal Karthik, Huseyin Coskun, Zeynep Akata, Sergey Tulyakov, Jian Ren, and Anil Kag. Scalable ranked preference optimization for text-to-image generation. In ICCV, 2025.
- [33] Lei Ke, Haohang Xu, Xuefei Ning, Yu Li, Jiajun Li, Haoling Li, Yuxuan Lin, Dongsheng Jiang, Yujiu Yang, and Linfeng Zhang. Proreflow: Progressive reflow with decomposed velocity. In CVPR, 2025.
- [34] Dan Kondratyuk, Lijun Yu, Xiuye Gu, José Lezama, Jonathan Huang, Grant Schindler, Rachel Hornung, Vighnesh Birodkar, Jimmy Yan, Ming-Chang Chiu, et al. Videopoet: A large language model for zero-shot video generation. In ICML, 2024.
- [35] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024.
- [36] Xin Li, Wenqing Chu, Ye Wu, Weihang Yuan, Fanglong Liu, Qi Zhang, Fu Li, Haocheng Feng, Errui Ding, and Jingdong Wang. Videogen: A reference-guided latent diffusion approach for high definition text-to-video generation. arXiv preprint arXiv:2309.00398, 2023.
- [37] Bin Lin, Yang Ye, Bin Zhu, Jiaxi Cui, Munan Ning, Peng Jin, and Li Yuan. Video-llava: Learning united visual representation by alignment before projection. In EMNLP, 2024.
- [38] Jie Liu, Gongye Liu, Jiajun Liang, Ziyang Yuan, Xiaokun Liu, Mingwu Zheng, Xiele Wu, Qiulin Wang, Wenyu Qin, Menghan Xia, et al. Improving video generation with human feedback. arXiv preprint arXiv:2501.13918, 2025.
- [39] Runtao Liu, Haoyu Wu, Ziqiang Zheng, Chen Wei, Yingqing He, Renjie Pi, and Qifeng Chen. Videodpo: Omni-preference alignment for video diffusion generation. In CVPR, 2025.
- [40] Ilya Loshchilov and Frank Hutter. Sgdr: Stochastic gradient descent with warm restarts. In ICLR, 2017.
- [41] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In ICLR, 2019.
- [42] Haoyu Lu, Guoxing Yang, Nanyi Fei, Yuqi Huo, Zhiwu Lu, Ping Luo, and Mingyu Ding. Vdt: General-purpose video diffusion transformers via mask modeling. In ICLR, 2024.
- [43] Xin Ma, Yaohui Wang, Gengyun Jia, Xinyuan Chen, Ziwei Liu, Yuan-Fang Li, Cunjian Chen, and Yu Qiao. Latte: Latent diffusion transformer for video generation. TMLR, 2024.

- [44] Willi Menapace, Aliaksandr Siarohin, Ivan Skorokhodov, Ekaterina Deyneka, Tsai-Shien Chen, Anil Kag, Yuwei Fang, Aleksei Stoliar, Elisa Ricci, Jian Ren, et al. Snap video: Scaled spatiotemporal transformers for text-to-video synthesis. In CVPR, 2024.
- [45] Fanqing Meng, Jiaqi Liao, Xinyu Tan, Wenqi Shao, Quanfeng Lu, Kaipeng Zhang, Yu Cheng, Dianqi Li, Yu Qiao, and Ping Luo. Towards world simulator: Crafting physical commonsense-based benchmark for video generation. In ICML, 2025.
- [46] Sanghyeon Na, Yonggyu Kim, and Hyunjoon Lee. Boost your human image generation model via direct preference optimization. In CVPR, 2025.
- [47] Sharan Narang, Gregory Diamos, Erich Elsen, Paulius Micikevicius, Jonah Alben, David Garcia, Boris Ginsburg, Michael Houston, Oleksii Kuchaiev, Ganesh Venkatesh, et al. Mixed precision training. In ICLR, 2018.
- [48] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, high-performance deep learning library. In NeurIPS, 2019.
- [49] William Peebles and Saining Xie. Scalable diffusion models with transformers. In ICCV, 2023.
- [50] Pika. Pika art. https://pika.art/, 2025. Accessed: 2025-05-04.
- [51] Adam Polyak, Amit Zohar, Andrew Brown, Andros Tjandra, Animesh Sinha, Ann Lee, Apoorv Vyas, Bowen Shi, Chih-Yao Ma, Ching-Yao Chuang, et al. Movie gen: A cast of media foundation models. arXiv preprint arXiv:2410.13720, 2024.
- [52] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, 2022.
- [53] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. In NeurIPS, 2023.
- [54] Nils Reimers and Iryna Gurevych. Sentence-bert: Sentence embeddings using siamese bert-networks. In EMNLP, 2019.
- [55] Dani Valevski, Yaniv Leviathan, Moab Arar, and Shlomi Fruchter. Diffusion models are real-time game engines. In ICLR, 2025.
- [56] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in neural information processing systems, 2017.
- [57] Bram Wallace, Meihua Dang, Rafael Rafailov, Linqi Zhou, Aaron Lou, Senthil Purushwalkam, Stefano Ermon, Caiming Xiong, Shafiq Joty, and Nikhil Naik. Diffusion model alignment using direct preference optimization. In CVPR, 2024.
- [58] Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, Jianyuan Zeng, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.
- [59] Duomin Wang, Wei Zuo, Aojie Li, Ling-Hao Chen, Xinyao Liao, Deyu Zhou, Zixin Yin, Xili Dai, Daxin Jiang, and Gang Yu. Universe-1: Unified audio-video generation via stitching of experts. arXiv preprint arXiv:2509.06155, 2025.
- [60] Qinghe Wang, Yawen Luo, Xiaoyu Shi, Xu Jia, Huchuan Lu, Tianfan Xue, Xintao Wang, Pengfei Wan, Di Zhang, and Kun Gai. Cinemaster: A 3d-aware and controllable framework for cinematic text-to-video generation. In SIGGRAPH, 2025.
- [61] Xiaofeng Wang, Zheng Zhu, Guan Huang, Xinze Chen, Jiagang Zhu, and Jiwen Lu. Drivedreamer: Towards real-worlddrive world models for autonomous driving. In ECCV, 2024.
- [62] Yaohui Wang, Xinyuan Chen, Xin Ma, Shangchen Zhou, Ziqi Huang, Yi Wang, Ceyuan Yang, Yinan He, Jiashuo Yu, Peiqing Yang, et al. Lavie: High-quality video generation with cascaded latent diffusion models. IJCV, 2025.
- [63] Yi Wang, Kunchang Li, Xinhao Li, Jiashuo Yu, Yinan He, Guo Chen, Baoqi Pei, Rongkun Zheng, Zun Wang, Yansong Shi, et al. Internvideo2: Scaling foundation models for multimodal video understanding. In ECCV, 2024.
- [64] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. In NeurIPS, 2022.
- [65] Yuqing Wen, Yucheng Zhao, Yingfei Liu, Fan Jia, Yanhui Wang, Chong Luo, Chi Zhang, Tiancai Wang, Xiaoyan Sun, and Xiangyu Zhang. Panacea: Panoramic and controllable video generation for autonomous driving. In CVPR, 2024.
- [66] Weijia Wu, Mingyu Liu, Zeyu Zhu, Xi Xia, Haoen Feng, Wen Wang, Kevin Qinghong Lin, Chunhua Shen, and Mike Zheng Shou. Moviebench: A hierarchical movie level dataset for long video generation. In CVPR, 2025.

- [67] Ziyi Wu, Anil Kag, Ivan Skorokhodov, Willi Menapace, Ashkan Mirzaei, Igor Gilitschenski, Sergey Tulyakov, and Aliaksandr Siarohin. Densedpo: Fine-grained temporal preference optimization for video diffusion models. arXiv preprint arXiv:2506.03517, 2025.
- [68] Qiyao Xue, Xiangyu Yin, Boyuan Yang, and Wei Gao. Phyt2v: Llm-guided iterative self-refinement for physics-grounded text-to-video generation. In CVPR, 2025.
- [69] An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, Guanting Dong, Haoran Wei, Huan Lin, Jialong Tang, Jialin Wang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Ma, Jin Xu, Jingren Zhou, Jinze Bai, Jinzheng He, Junyang Lin, Kai Dang, Keming Lu, Keqin Chen, Kexin Yang, Mei Li, Mingfeng Xue, Na Ni, Pei Zhang, Peng Wang, Ru Peng, Rui Men, Ruize Gao, Runji Lin, Shijie Wang, Shuai Bai, Sinan Tan, Tianhang Zhu, Tianhao Li, Tianyu Liu, Wenbin Ge, Xiaodong Deng, Xiaohuan Zhou, Xingzhang Ren, Xinyu Zhang, Xipin Wei, Xuancheng Ren, Yang Fan, Yang Yao, Yichang Zhang, Yu Wan, Yunfei Chu, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zhihao Fan. Qwen2 technical report. arXiv preprint arXiv:2407.10671, 2024.
- [70] Jiazhi Yang, Shenyuan Gao, Yihang Qiu, Li Chen, Tianyu Li, Bo Dai, Kashyap Chitta, Penghao Wu, Jia Zeng, Ping Luo, et al. Generalized predictive model for autonomous driving. In CVPR, 2024.
- [71] Xiaomeng Yang, Zhiyu Tan, and Hao Li. Ipo: Iterative preference optimization for text-to-video generation. arXiv preprint arXiv:2502.02088, 2025.
- [72] Yongsheng Yu, Ziyun Zeng, Haitian Zheng, and Jiebo Luo. Omnipaint: Mastering object-oriented editing via disentangled insertion-removal inpainting. In ICCV, 2025.
- [73] Ke Zhang, Cihan Xiao, Yiqun Mei, Jiacong Xu, and Vishal M Patel. Think before you diffuse: Llms-guided physics-aware video generation. arXiv preprint arXiv:2505.21653, 2025.
- [74] Shilong Zhang, Wenbo Li, Shoufa Chen, Chongjian Ge, Peize Sun, Yida Zhang, Yi Jiang, Zehuan Yuan, Binyue Peng, and Ping Luo. Flashvideo: Flowing fidelity to detail for efficient high-resolution video generation. arXiv preprint arXiv:2502.05179, 2025.
- [75] Shiwei Zhang, Jiayu Wang, Yingya Zhang, Kang Zhao, Hangjie Yuan, Zhiwu Qin, Xiang Wang, Deli Zhao, and Jingren Zhou. I2vgen-xl: High-quality image-to-video synthesis via cascaded diffusion models. arXiv preprint arXiv:2311.04145, 2023.
- [76] Yifu Zhang, Hao Yang, Yuqi Zhang, Yifei Hu, Fengda Zhu, Chuang Lin, Xiaofeng Mei, Yi Jiang, Bingyue Peng, and Zehuan Yuan. Waver: Wave your way to lifelike video generation. arXiv preprint arXiv:2508.15761, 2025.
- [77] Zangwei Zheng, Xiangyu Peng, Tianji Yang, Chenhui Shen, Shenggui Li, Hongxin Liu, Yukun Zhou, Tianyi Li, and Yang You. Open-sora: Democratizing efficient video production for all. arXiv preprint arXiv:2412.20404, 2024.
- [78] Daquan Zhou, Weimin Wang, Hanshu Yan, Weiwei Lv, Yizhe Zhu, and Jiashi Feng. Magicvideo: Efficient video generation with latent diffusion models. arXiv preprint arXiv:2211.11018, 2022.

## Appendix

Mathematical proof for Ineq. (8)

In the main paper, we use an inequality (8) as

m

m

exj ≤

1 +

j=1

j=1

1 + eαjxj γj, 0 < αj ≤ 1, γj ≥ 1/αj. (15)

Here we prove it. For each j, since 0 < αj ≤ 1, the function fj(t) = tα

j is concave on [0,+∞). For any u,v ≥ 0, let

s = u + v. If s = 0 the inequality is trivial. Assume s > 0 and set λ = us ∈ [0,1], so that u = λs and v = (1 − λ)s. By the concavity of fj, we can derive the following inequality:

fj(u) = fj(λs + (1 − λ) · 0) ≥ λfj(s) + (1 − λ)fj(0) = λfj(s),

and similarly, we derive the inequality for fj(v) as

(16)

fj(v) = fj (1 − λ)s + λ · 0 ≥ (1 − λ)fj(s). (17) Adding the two inequalities Ineq. (16) and Ineq. (17) yields

uα

j

#### + vα

j

= fj(u) + fj(v) ≥ λfj(s) + (1 − λ)fj(s)

= fj(s) = (u + v)α

,

j

(18)

this is,

(u + v)α

#### ≤ uα

#### + vα

. (19) Applying Ineq. (19) to u = 1 and v = ex

j

j

j

j yields 1 + ex

j αj ≤ 1 + eα

jxj, (20) which is equivalent to

jxj 1/αj. (21) Since γj ≥ 1/αj and the base 1 + eα

1 + ex

≤ 1 + eα

j

jxj ≥ 1, we have

jxj γj, (22) and thus from Ineq. (21), we can easily derive:

jxj 1/αj ≤ 1 + eα

1 + eα

jxj γj =⇒ ex

jxj γj − 1. (23)

1 + ex

≤ 1 + eα

≤ 1 + eα

j

j

jxj γj − 1 ≥ 0. Then Ineq. (23) gives ex

We define aj ≜ 1 + eα

#### ≤ aj for all j, and hence m

j

m

ex

aj. (24)

≤

j

j=1

j=1

On the other hand, for any nonnegative {aj}mj=1 we have

m

m

aj + (higher-order terms in {aj}), (25)

(1 + aj) = 1 +

j=1

j=1

where all higher-order terms are nonnegative, so

m

aj ≤

1 +

j=1

m

(1 + aj). (26)

j=1

Combining Ineq. (24) and Ineq. (26) and substituting the definition of aj yields

m

ex

1 +

j

j=1

≤ 1 +

m

=

j=1

m

=

j=1

m

m

aj ≤

(1 + aj)

j=1

j=1

jxj γj − 1

1 + 1 + eα

jxj γj,

1 + eα

which is exactly the claimed Ineq (7) used in the main paper.

(27)

