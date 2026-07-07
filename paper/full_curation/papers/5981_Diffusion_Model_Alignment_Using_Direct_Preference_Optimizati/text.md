## Diffusion Model Alignment Using Direct Preference Optimization

# arXiv:2311.12908v1[cs.CV]21Nov2023

Bram Wallace1 Meihua Dang2 Rafael Rafailov2 Linqi Zhou2 Aaron Lou2 Senthil Purushwalkam1 Stefano Ermon2 Caiming Xiong1 Shafiq Joty1 Nikhil Naik1 1Salesforce AI, 2Stanford University

{b.wallace,spurushwalkam,cxiong,sjoty,nnaik}@salesforce.com {mhdang,rafailov,lzhou907,aaronlou}@stanford.edu {ermon}@cs.stanford.edu

### Abstract

Large language models (LLMs) are fine-tuned using human comparison data with Reinforcement Learning from Human Feedback (RLHF) methods to make them better aligned with users’ preferences. In contrast to LLMs, human preference learning has not been widely explored in text-to-image diffusion models; the best existing approach is to fine-tune a pretrained model using carefully curated high quality images and captions to improve visual appeal and text alignment. We propose DiffusionDPO, a method to align diffusion models to human preferences by directly optimizing on human comparison data. Diffusion-DPO is adapted from the recently developed Direct Preference Optimization (DPO) [33], a simpler alternative to RLHF which directly optimizes a policy that best satisfies human preferences under a classification objective. We re-formulate DPO to account for a diffusion model notion of likelihood, utilizing the evidence lower bound to derive a differentiable objective. Using the Picka-Pic dataset of 851K crowdsourced pairwise preferences, we fine-tune the base model of the state-of-the-art Stable Diffusion XL (SDXL)-1.0 model with Diffusion-DPO. Our fine-tuned base model significantly outperforms both base SDXL-1.0 and the larger SDXL-1.0 model consisting of an additional refinement model in human evaluation, improving visual appeal and prompt alignment. We also develop a variant that uses AI feedback and has comparable performance to training on human preferences, opening the door for scaling of diffusion model alignment methods.

### 1. Introduction

Text-to-image diffusion models have been the state-of-theart in image generation for the past few years. They are typically trained in a single stage, using web-scale datasets of text-image pairs by applying the diffusion objective. This

stands in contrast to the state-of-the-art training methodology for Large Language Models (LLMs). The best performing LLMs [28, 48] are trained in two stages. In the first (“pretraining”) stage, they are trained on large webscale data. In the second (“alignment”) stage, they are fine-tuned to make them better aligned with human preferences. Alignment is typically performed using supervised fine-tuning (SFT) and Reinforcement Learning from Human Feedback (RLHF) using preference data. LLMs trained with this two-stage process have set the state-of-theart in language generation tasks and have been deployed in commercial applications such as ChatGPT and Bard.

Despite the success of the LLM alignment process, most text-to-image diffusion training pipelines do not incorporate learning from human preferences. Several models [9, 35, 36], perform two-stage training, where largescale pretraining is followed by fine-tuning on a highquality text-image pair dataset to strategically bias the generative process. This approach is much less powerful and flexible than the final-stage alignment methods of LLMs. Recent approaches [6, 7, 11, 31] develop more advanced ways to tailor diffusion models to human preferences, but none have demonstrated the ability to stably generalize to a fully open-vocabulary setting across an array of feedback. RL-based methods [6, 11] are highly effective for limited prompt sets, but their efficacy decreases as the vocabulary expands. Other methods [7, 31] use the pixel-level gradients from reward models on generations to tune diffusion models, but suffer from mode collapse and can only be trained on a relatively narrow set of feedback types.

We address this gap in diffusion model alignment for the first time, developing a method to directly optimize diffusion models on human preference data. We generalize Direct Preference Optimization (DPO) [33], where a generative model is trained on paired human preference data to implicitly estimate a reward model. We define a notion of data likelihood under a diffusion model in a novel formulation and derive a simple but effective loss resulting in stable

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

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

- Figure 1. We develop Diffusion-DPO, a method based on Direct Preference Optimization (DPO) [33] for aligning diffusion models to human preferences by directly optimizing the model on user feedback data. After fine-tuning on the state-of-the-art SDXL-1.0 model, our method produces images with exceptionally high visual appeal and text alignment, samples above.

and efficient preference training, dubbed Diffusion-DPO. We connect this formulation to a multi-step RL approach in the same setting as existing work [6, 11].

We demonstrate the efficacy of Diffusion-DPO by finetuning state-of-the-art text-to-image diffusion models, such as Stable Diffusion XL (SDXL)-1.0 [30]. Human evaluators prefer DPO-tuned SDXL images over the SDXL(base + refinement) model 69% of the time on the PartiPrompts dataset, which represents the state-of-the-art in text-to-image models as measured by human preference. Example generations shown in Fig. 1. Finally, we show that learning from AI feedback (instead of human preferences) using the Diffusion-DPO objective is also effective, a setting where previous works have been unsuccessful [7]. In sum, we introduce a novel paradigm of learning from human preferences for diffusion models and present the resulting state-of-the-art model.

### 2. Related Work

Aligning Large Language Models LLMs are typically aligned to human preferences using supervised fine-tuning on demonstration data, followed by RLHF. RLHF consists of training a reward function from comparison data on model outputs to represent human preferences and then using reinforcement learning to align the policy model. Prior work [4, 26, 29, 47] has used policy-gradient methods [27, 38] to this end. These methods are successful, but expensive and require extensive hyperparameter tuning [34, 59], and can be prone to reward hacking [10, 12, 41]. Alternative approaches sample base model answers and select based on predicted rewards [3, 5, 14] to use for supervised training [2, 16, 50]. Methods that fine-tune the policy model directly on feedback data [1, 10], or utilize a ranking loss on preference data to directly train the policy model [33, 49, 57, 58] have emerged. The latter set of

methods match RLHF in performance. We build on these fine-tuning methods in this work, specifically, direct preference optimization [33] (DPO). Finally, learning from AI feedback, using pretrained reward models, is promising for efficient scaling of alignment [4, 22].

Aligning Diffusion Models Alignment of diffusion models to human preferences has so far been much less explored than in the case of LLMs. Multiple approaches [30, 36] fine-tune on datasets scored as highly visually appealing by an aesthetics classifier [37], to bias the model to visually appealing generations. Emu [9] finetunes a pretrained model using a small, curated image dataset of high quality photographs with manually written detailed captions to improve visual appeal and text alignment. Other methods [15, 39] recaption existing web-scraped image datasets to improve text fidelity. Caption-aware human preference scoring models are trained on generation preference datasets [21, 52, 55], but the impact of these reward models to the generative space has been limited. DOODL [51] introduces the task of aesthetically improving a single generation iteratively at inference time. DRAFT [7] and AlignProp [31], incorporate a similar approach into training: tuning the generative model to directly increase the reward of generated images. These methods perform well for simple visual appeal criteria, but lack stability and do not work on more nuanced rewards such as text-image alignment from a CLIP model. DPOK and DDPO [6, 11] are RL-based approaches to similarly maximize the scored reward (with distributional constraints) over a relatively limited vocabulary set; the performance of these methods degrades as the number of train/test prompts increases. Diffusion-DPO is unique among alignment approaches in effectively increasing measured human appeal across an open vocabulary (DPOK, DDPO), without increased inference time (DOODL) while maintaining distributional guarantees and improving generic text-image alignment in addition to visual appeal (DRAFT, AlignProp). (see Tab. 1, further discussion in Supp. S1).

Equal

Open Inference Divergence Methods Vocab. Cost Control

DPOK[11] ✗ ✓ ✓ DDPO[6] ✗ ✓ ✗

DOODL[51] ✓ ✗ ✗ DRaFT[7],AlignProp[31] ✓ ✓ ✗

Diffusion-DPO (ours) ✓ ✓ ✓

- Table 1. Method class comparison. Existing methods fail in one or more of: Generalizing to an open vocabulary, maintaining the same inference complexity, avoiding mode collapse/providing distributional guarantees. Diffusion-DPO addresses these issues.

### 3. Background

#### 3.1. Diffusion Models

Given samples from a data distribution q(x0), noise scheduling function αt and σt (as defined in [36]), denoising diffusion models [17, 42, 46] are generative models pθ(x0) which have a discrete-time reverse process with a Markov structure pθ(x0:T) = Tt=1 pθ(xt−1|xt) where

σt2−1 σt2

pθ(xt−1|xt) = N(xt−1;µθ(xt),σt2|t−1

I). (1)

Training is performed by minimizing the evidence lower bound (ELBO) associated with this model [20, 45]:

0,ϵ,t,xt ω(λt)∥ϵ − ϵθ(xt,t)∥22 , (2)

LDM = Ex

with ϵ ∼ N(0,I), t ∼ U(0,T), xt ∼ q(xt|x0) = N(xt;αtx0,σt2I). λt = αt2/σt2 is a signal-to-noise ratio [20], ω(λt) is a pre-specified weighting function (typically chosen to be constant [17, 44]).

#### 3.2. Direct Preference Optimization

Our approach is an adaption of Direct Preference Optimization (DPO) [33], an effective approach for learning from human preference for language models. Abusing notation, we also use x0 as random variables for language.

Reward Modeling Estimating human partiality to a generation x0 given conditioning c, is difficult as we do not have access to the latent reward model r(c,x0). In our setting, we assume access only to ranked pairs generated from some conditioning xw0 ≻ xl0|c, where xw0 and xl0 denoting the “winning” and “losing” samples. The Bradley-Terry (BT) model stipulates to write human preferences as:

pBT(xw0 ≻ xl0|c) = σ(r(c,xw0 ) − r(c,xl0)) (3)

where σ is the sigmoid function. r(c,x0) can be parameterized by a neural network ϕ and estimated via maximum likelihood training for binary classification:

0 ,xl0 log σ rϕ(c,xw0 ) − rϕ(c,xl0)

LBT(ϕ) = −Ec,xw

(4)

where prompt c and data pairs xw0 ,xl0 are from a static dataset with human-annotated labels.

RLHF RLHF aims to optimize a conditional distribution pθ(x0|c) (conditioning c ∼ Dc) such that the latent reward model r(c,x0) defined on it is maximized, while regularizing the KL-divergence from a reference distribution pref

Ec∼D

max

c,x0∼pθ(x0|c) [r(c,x0)]

pθ

− βDKL [pθ(x0|c)∥pref(x0|c)] (5) where the hyperparameter β controls regularization.

DPO Objective In Eq. (5) from [33], the unique global optimal solution p∗θ takes the form:

p∗θ(x0|c) = pref(x0|c)exp(r(c,x0)/β)/Z(c) (6)

where Z(c) = x

pref(x0|c)exp(r(c,x0)/β) is the partition function. Hence, the reward function is rewritten as

0

p∗θ(x0|c) pref(x0|c)

+ β log Z(c) (7)

r(c,x0) = β log

Using Eq. (4), the reward objective becomes:

pθ(xl0|c) pref(xl0|c)

pθ(xw0 |c) pref(xw0 |c) − β log

LDPO(θ)=−Ec,xw

0 ,xl0 log σ β log

(8)

By this reparameterization, instead of optimizing the reward function rϕ and then performing RL, [33] directly optimizes the optimal conditional distribution pθ(x0|c).

### 4. DPO for Diffusion Models

In adapting DPO to diffusion models, we consider a setting where we have a fixed dataset D = {(c,xw0 ,xl0)} where each example contains a prompt c and a pairs of images generated from a reference model pref with human label xw0 ≻ xl0. We aim to learn a new model pθ which is aligned to the human preferences, with preferred generations to pref. The primary challenge we face is that the parameterized distribution pθ(x0|c) is not tractable, as it needs to marginalize out all possible diffusion paths (x1,...xT) which lead to x0. To overcome this challenge, we utilize the evidence lower bound (ELBO). Here, we introduce latents x1:T and define R(c,x0:T) as the reward on the whole chain, such that we can define r(c,x0) as

θ(x1:T|x0,c) [R(c,x0:T)]. (9)

r(c,x0) = Ep

As for the KL-regularization term in Eq. (5), following prior work [17, 42], we can instead minimize its upper bound joint KL-divergence DKL [pθ(x0:T|c)∥pref(x0:T|c)]. Plugging this KL-divergence bound and the definition of r(c,x0) (Eq. (9)) back to Eq. (5), we have the objective

Ec∼D

max

c,x0:T∼pθ(x0:T|c) [r(c,x0)] − βDKL [pθ(x0:T|c)∥pref(x0:T|c)]. (10)

pθ

This objective has a parallel formulation as Eq. (5) but defined on path x0:T. It aims to maximize the reward for reverse process pθ(x0:T), while matching the distribution of the original reference reverse process. Paralleling Eqs. (6) to (8), this objective can be optimized directly through the

conditional distribution pθ(x0:T) via objective:

LDPO-Diffusion(θ) = −E(xw

0 ,xl0)∼D logσ

pθ(xl0:T) pref(xl0:T)

pθ(xw0:T) pref(xw0:T) −log

βExw

log

1:T ∼pθ(xw1:T |xw0 ) xl1:T ∼pθ(xl1:T |xl0)

(11)

We omit c for compactness (details included in Supp. S2). To optimize Eq. (11), we must sample x1:T ∼pθ(x1:T|x0). Despite the fact that pθ contains trainable parameters, this sampling procedure is both (1) inefficient as T is usually large (T = 1000), and (2) intractable since pθ(x1:T) represents the reverse process parameterization pθ(x1:T) = pθ(xT) Tt=1 pθ(xt−1|xt). We solve these two issues next.

From Eq. (11), we substitute the reverse decompositions for pθ and pref, and utilize Jensen’s inequality and the convexity of function −log σ to push the expectation outside. With some simplification, we get the following bound

LDPO-Diffusion(θ) ≤ −E(xw

0 ,xl0)∼D,t∼U(0,T),

xwt−1,t∼pθ(xwt−1,t|xw0 ), xlt−1,t∼pθ(xlt−1,t|xl0)

pθ(xlt−1|xlt) pref(xlt−1|xlt)

pθ(xwt−1|xwt ) pref(xwt−1|xwt ) −βT log

log σ βT log

(12)

Efficient training via gradient descent is now possible. However, sampling from reverse joint pθ(xt−1,xt|x0,c) is still intractable and r of Eq. (9) has an expectation over pθ(x1:T|x0). So we approximate the reverse process

- pθ(x1:T|x0) with the forward q(x1:T|x0) (an alternative scheme in Supp. S2). With some algebra, this yields:

L(θ) = −E(xw

0 ,xl0)∼D,t∼U(0,T),xwt ∼q(xwt |xw0 ),xlt∼q(xlt|xl0)

log σ(−βT(

+ DKL(q(xwt−1|xw0,t)∥pθ(xwt−1|xwt )) − DKL(q(xwt−1|xw0,t)∥pref(xwt−1|xwt )) − DKL(q(xlt−1|xl0,t)∥pθ(xlt−1|xlt))

+ DKL(q(xlt−1|xl0,t)∥pref(xlt−1|xlt))).

(13) Using Eq. (1) and algebra, the above loss simplifies to:

L(θ) = −E(xw

0 ,xl0)∼D,t∼U(0,T),xwt ∼q(xwt |xw0 ),xlt∼q(xlt|xl0)

log σ (−βTω(λt)( ∥ϵw − ϵθ(xwt ,t)∥22 − ∥ϵw − ϵref(xwt ,t)∥22

− ∥ϵl − ϵθ(xlt,t)∥22 − ∥ϵl − ϵref(xlt,t)∥22 (14) where x∗t = αtx∗0 + σtϵ∗,ϵ∗ ∼ N(0,I) is a draw from

- q(x∗t|x∗0) (Eq. (2)). λt = αt2/σt2 is the signal-to-noise ratio,

Change.

0.4

0.0

0.4 0.0 0.4

Eror

0.4

x0l Error Change. || l− θ(xtl,t)||22−|| l− ref(xtl,t)||22

xw0

- Figure 2. Loss surface visualization. Loss can be decreased by

improving at denoising xw0 and worsening for xl0. A larger β increases surface curvature.

ω(λt) a weighting function (constant in practice [17, 20]). We factor the constant T into β. This loss encourages ϵθ to improve more at denoising xwt than xlt, visualization in Fig. 2. We also derive Eq. (14) as a multi-step RL approach in the same setting as DDPO and DPOK [6, 11] (Supp. S3) but as an off-policy algorithm, which justifies our sampling choice in Eq. 13. A noisy preference model perspective yields the same objective (Supp. S4).

### 5. Experiments 5.1. Setting

Models and Dataset: We demonstrate the efficacy of Diffusion-DPO across a range of experiments. We use the objective from Eq. (14) to fine-tune Stable Diffusion 1.5 (SD1.5) [36] and the state-of-the-art open-source model Stable Diffusion XL-1.0 (SDXL) [30] base model. We train on the Pick-a-Pic [21] dataset, which consists of pairwise preferences for images generated by SDXL-beta and Dreamlike, a fine-tuned version of SD1.5. The prompts and preferences were collected from users of the Pick-a-Pic web application (see [21] for details). We use the larger Pick-aPic v2 dataset. After excluding the ∼12% of pairs with ties, we end up with 851,293 pairs, with 58,960 unique prompts.

Hyperparameters We use AdamW [24] for SD1.5 experiments, and Adafactor [40] for SDXL to save memory. An effective batch size of 2048 (pairs) is used; training on 16 NVIDIA A100 GPUs with a local batch size of 1 pair and gradient accumulation of 128 steps. We train at fixed square resolutions. A learning rate of 2000β 2.048·10−8 is used with 25% linear warmup. The inverse scaling is motivated by the norm of the DPO objective gradient being proportional to β (the divergence penalty parameter) [33]. For both SD1.5 and SDXL, we find β ∈ [2000,5000] to offer good performance (Supp. S5). We present main SD1.5 results with β = 2000 and SDXL results with β = 5000.

Evaluation We automatically validate checkpoints with the 500 unique prompts of the Pick-a-Pic validation set: measuring median PickScore reward of generated images. Pickscore [21] is a caption-aware scoring model trained on Pick-a-Pic (v1) to estimate human-perceived image quality. For final testing, we generate images using the baseline and Diffusion-DPO-tuned models conditioned on captions from the Partiprompt [56] and HPSv2 [52] benchmarks (1632 and 3200 captions respectively). While DDPO [6] is a related method, we did not observe stable improvement when training from public implementations on Pick-a-Pic. We employ labelers on Amazon Mechanical Turk to compare generations under three different criteria: Q1 General Preference (Which image do you prefer given the prompt?), Q2 Visual Appeal (prompt not considered) (Which image is more visually appealing?) Q3 Prompt Alignment (Which image better fits the text description?). Five responses are collected for each comparison with majority vote (3+) being considered the collective decision.

#### 5.2. Primary Results: Aligning Diffusion Models

First, we show that the outputs of the Diffusion-DPOfinetuned SDXL model are significantly preferred over the baseline SDXL-base model. In the Partiprompt evaluation (Fig. 3-top left), DPO-SDXL is preferred 70.0% of the time for General Preference (Q1), and obtains a similar winrate in assessments of both Visual Appeal (Q2) and Prompt Alignment (Q3). Evaluation on the HPS benchmark (Fig. 3top right) shows a similar trend, with a General Preference win rate of 64.7%. We also score the DPO-SDXL HPSv2 generations with the HPSv2 reward model, achieving an average reward of 28.16, topping the leaderboard [53].

We display qualitative comparisons to SDXL-base in Fig. 3 (bottom). Diffusion-DPO produces more appealing imagery, with vivid arrays of colors, dramatic lighting, good composition, and realistic people/animal anatomy. While all SDXL images satisfy the prompting criteria to some degree, the DPO generations appear superior, as confirmed by the crowdsourced study. We do note that preferences are not universal, and while the most common shared preference is towards energetic and dramatic imagery, others may prefer quieter/subtler scenes. The area of personal or group preference tuning is an exciting area of future work.

After this parameter-equal comparison with SDXL-base, we compare SDXL-DPO to the complete SDXL pipeline, consisting of the base model and the refinement model (Fig. 4). The refinement model is an image-to-image diffusion model that improves visual quality of generations, and is especially effective on detailed backgrounds and faces. In our experiments with PartiPrompts and HPSv2, SDXLDPO (3.5B parameters, SDXL-base architecture only), handily beats the complete SDXL model (6.6B parameters). In the General Preference question, it has a benchmark win

[Figure 20]

[Figure 21]

DPO-SDXLSDXLDPO-SDXL

SDXL

A smiling beautiful sorceress wearing a high necked blue suit surrounded by swirling rainbow aurora, hyper-realistic, cinematic, post-production

A monk in an orange robe by a round window in a spaceship in dramatic lighting

Concept art of a mythical sky alligator with wings, nature documentary

A galaxy-colored figurine is floating over the sea at sunset, photorealistic

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

- Figure 3. (Top) DPO-SDXL significantly outperforms SDXL in human evaluation. (L) PartiPrompts and (R) HPSv2 benchmark results across three evaluation questions, majority vote of 5 labelers. (Bottom) Qualitative comparisons between SDXL and DPO-SDXL. DPOSDXL demonstrates superior prompt following and realism. DPO-SDXL outputs are better aligned with human aesthetic preferences, favoring high contrast, vivid colors, fine detail, and focused composition. They also capture fine-grained textual details more faithfully.

rate of 69% and 64% respectively, comparable to its win rate over SDXL-base alone. This is explained by the ability of the DPO-tuned model (Fig. 4, bottom) to generate fine-grained details and its strong performance across different image categories. While the refinement model is especially good at improving the generation of human details, the win rate of Diffusion-DPO on the People category in Partiprompt dataset over the base + refiner model is still an impressive 67.2% (compared to 73.4% over the base).

#### 5.3. Image-to-Image Editing

Image-to-image translation performance also improves after Diffusion-DPO tuning. We test DPO-SDXL on TEdBench [19], a text-based image-editing benchmark of 100 real image-text pairs, using SDEdit [25] with noise strength 0.6. Labelers are shown the original image and SDXL/DPO-SDXL edits and asked “Which edit do you prefer given the text?” DPO-SDXL is preferred 65% of the

time, SDXL 24%, with 11% draws. We show qualitative SDEdit results on color layouts (strength 0.98) in Fig. 5.

#### 5.4. Learning from AI Feedback

In LLMs, learning from AI feedback has emerged as a strong alternative to learning from human preferences [22]. Diffusion-DPO can also admit learning from AI feedback by directly ranking generated pairs into (yw,yl) using a pretrained scoring network. We use HPSv2 [52] for an alternate prompt-aware human preference estimate, CLIP (OpenCLIP ViT-H/14) [18, 32] for text-image alignment, Aesthetic Predictor [37] for non-text-based visual appeal, and PickScore. We run all experiments on SD 1.5 with β = 5000 for 1000 steps. Training on the promptaware PickScore and HPS preference estimates increase the win rate for both raw visual appeal and prompt alignment (Fig. 6). We note that PickScore feedback is interpretable as pseudo-labeling the Pick-a-Pic dataset—a form of data

|Anthropmorphic koala teaching a college class<br><br>|
|---|

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

DPO-SDXL

Refiner

SDXL

Base+

[Figure 40]

SDXL Base SDXL Base + Refiner DPO-SDXL

[Figure 41]

| |
|---|
| |
| |
| |
| |
| |
| |
| |
| |

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

| |
|---|

[Figure 49]

[Figure 50]

| |
|---|
| |
| |
| |
| |
| |

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

| |
|---|

[Figure 59]

[Figure 60]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

- Figure 4. DPO-SDXL (base only) significantly outperforms the much larger SDXL-(base+refinement) model pipeline in human evaluations on the PartiPrompts and HPS datasets. While the SDXL refinement model is used to touch up details from the output of SDXL-base, the ability to generate high quality details has been naturally distilled into DPO-SDXL by human preference. Among other advantages, DPO-SDXL shows superior generation of anatomical features such as teeth, hands, and eyes. Prompts: close up headshot, steampunk middle-aged man, slick hair big grin in front of gigantic clocktower, pencil sketch / close up headshot, futuristic young woman with glasses, wild hair sly smile in front of gigantic UFO, dslr, sharp focus, dynamic composition / A man and woman using their cellphones, photograph

cleaning [54, 60]. Training for Aesthetics and CLIP improves those capabilities more specifically, in the case of Aesthetics at the expense of CLIP. The ability to train for text-image alignment via CLIP is a noted improvement over prior work [7]. Moreover, training SD1.5 on the pseudolabeled PickScore dataset (β = 5000, 2000 steps) outperforms training on the raw labels. On the General Preference Partiprompt question, the win-rate of DPO increases from 59.8% to 63.3%, indicating that learning from AI feedback

can be a promising direction for diffusion model alignment.

#### 5.5. Analysis

Implicit Reward Model As a consequence of the theoretical framework, our DPO scheme implicitly learns a reward model and can estimate the differences in rewards between two images by taking an expectation over the inner term of Eq. (14) (details in Supp. S4.1). We estimate over 10 random t ∼ U{0,1} Our learned models (DPO-SD1.5

[Figure 72]

Original SDXL DPO-SDXL

- Figure 5. Diffusion-DPO generates more visually appealing images in the downstream image-to-image translation task. Comparisons of using SDEdit [25] from color layouts. Prompts are "A fantasy landscape, trending on artstation" (top) , "High-resolution rendering of a crowded colorful sci-fi city" (bottom).

PickScore

Training Ranker Automated Win Rate vs. SD1.5

HPS

CLIP

PickScore HPS CLIP Aesthetics

0.3

0.5

1.0 Aesthetics

Metric Model:

- Figure 6. Automated head-to-head win rates under reward models (x labels, columns) for SD1.5 DPO-tuned on the “preferences” of varied scoring networks (y labels, rows). Example: Tuning on Aesthetics preferences (bottom row) achieves high Aesthetics scores but has lower text-image alignment as measured by CLIP.

|Model|PS HPS CLIP Aes.<br><br>|DPO-SD1.5 DPO-SDXL|
|---|---|---|
|Acc.|64.2 59.3 57.1 51.4<br><br>|60.8 72.0|

- Table 2. Preference accuracy on the Pick-a-Pic (v2) validation set. The v1-trained PickScore has seen the evaluated data.

and DPO-SDXL) perform well at binary preference classification (Tab. 2), with DPO-SDXL exceeding all existing recognition models on this split. These results shows that the implicit reward parameterization in the Diffusion-DPO objective has comprable expressivity and generalization as the classical reward modelling objective/architecture.

Training Data Quality Fig. 7 shows that despite SDXL being superior to the training data (including the yw), as measured by Pickscore, DPO training improves its perfor-

MedianPickscore

0.23

0.22

0.21

mlike y: S D X Lyw : S D X L- SDXLDPO-SDXL

mlikeDreamlikeDPO-Dreamlike*

Drea

Drea

y:

:

yw

Figure 7. Diffusion-DPO improves on the baseline Dreamlike and SDXL models, when finetuned on both in-distribution data (in case of Dreamlike) and out-of-distribution data (in case of SDXL). yl and yw denote the Pickscore of winning and losing samples.

mance substantially. In this experiment, we confirm that Diffusion-DPO can improve on in-distribution preferences

- as well, by training (β = 5k, 2000 steps) the Dreamlike model on a subset of the Pick-a-Pic dataset generated by the Dreamlike model alone. This subset represents 15% of the original dataset. Dreamlike-DPO improves on the baseline model, though the performance improvement is limited, perhaps because of the small size of the dataset.

Supervised Fine-tuning (SFT) is beneficial in the LLM setting as initial pretraining prior to preference training. To evaluate SFT in our setting, we fine-tune models on the preferred (x,yw) pairs of the Pick-a-Pic dataset. We train for the same length schedule as DPO using a learning rate of 1e − 9 and observe convergence. While SFT improves vanilla SD1.5 (55.5% win rate over base model), any amount of SFT deteriorates the performance of SDXL, even

- at lower learning rates. This contrast is attributable to the much higher quality of Pick-a-Pic generations vs. SD1.5, as they are obtained from SDXL-beta and Dreamlike. In contrast, the SDXL-1.0 base model is superior to the Picka-Pic dataset models. See Supp. S6 for further discussion.

### 6. Conclusion

In this work, we introduce Diffusion-DPO: a method that enables diffusion models to directly learn from human feedback in an open-vocabulary setting for the first time. We fine-tune SDXL-1.0 using the Diffusion-DPO objective and the Pick-a-Pic (v2) dataset to create a new state-of-the-art for open-source text-to-image generation models as measured by generic preference, visual appeal, and prompt alignment. We additionally demonstrate that DPO-SDXL outperforms even the SDXL base plus refinement model pipeline, despite only employing 53% of the total model parameters. Dataset cleaning/scaling is a promising future direction as we observe preliminary data cleaning improving performance (Sec. 5.4). While DPO-Diffusion is an of-

fline algorithm, we anticipate online learning methods to be another driver of future performance. There are also exciting application variants such as tuning to the preferences of individuals or small groups.

Ethics The performance of Diffusion-DPO is impressive, but any effort in text-to-image generation presents ethical risks, particularly when data are web-collected. Generations of harmful, hateful, fake or sexually explicit content are known risk vectors. Beyond that, this approach is increasingly subject to the biases of the participating labelers (in addition to the biases present in the pretrained model); Diffusion-DPO can learn and propagate these preferences. As a result, a diverse and representative set of labelers is essential – whose preferences in turn become encoded in the dataset. Furthermore, a portion of user-generated Pick-a-Pic prompts are overtly sexual, and even innocuous prompts may deliver images that skew more suggestively (particularly for prompts that hyper-sexualize women). Finally, as with all text-to-image models, the image produced will not always match the prompt. Hearteningly though, some of these scenarios can be addressed at a dataset level, and data filtering is also possible. Regardless, we will not open source nor otherwise make available our model until we add additional safety filtering to ensure that toxic content is remediated.

### References

- [1] Model index for researchers, 2023. 2
- [2] Thomas Anthony, Zheng Tian, and David Barber. Thinking fast and slow with deep learning and tree search. Neural Information Processing Systems, 2017. 2
- [3] Amanda Askell, Yuntao Bai, Anna Chen, Dawn Drain, Deep Ganguli, Tom Henighan, Andy Jones, Nicholas Joseph, Ben Mann, Nova DasSarma, Nelson Elhage, Zac Hatfield-Dodds, Danny Hernandez, Jackson Kernion, Kamal Ndousse, Catherine Olsson, Dario Amodei, Tom Brown, Jack Clark, Sam McCandlish, Chris Olah, and Jared Kaplan. A general language assistant as a laboratory for alignment,

2021. 2

- [4] Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, Carol Chen, Catherine Olsson, Christopher Olah, Danny Hernandez, Dawn Drain, Deep Ganguli, Dustin Li, Eli TranJohnson, Ethan Perez, Jamie Kerr, Jared Mueller, Jeffrey Ladish, Joshua Landau, Kamal Ndousse, Kamile Lukosuite, Liane Lovitt, Michael Sellitto, Nelson Elhage, Nicholas Schiefer, Noemi Mercado, Nova DasSarma, Robert Lasenby, Robin Larson, Sam Ringer, Scott Johnston, Shauna Kravec, Sheer El Showk, Stanislav Fort, Tamera Lanham, Timothy Telleen-Lawton, Tom Conerly, Tom Henighan, Tristan Hume, Samuel R. Bowman, Zac Hatfield-Dodds, Ben Mann, Dario Amodei, Nicholas Joseph, Sam McCandlish, Tom Brown, and Jared Kaplan. Constitutional ai: Harmlessness

from ai feedback, 2022. 2, 3

- [5] Michiel A. Bakker, Martin J. Chadwick, Hannah R. Sheahan, Michael Henry Tessler, Lucy Campbell-Gillingham, Jan Balaguer, Nat McAleese, Amelia Glaese, John Aslanides, Matthew M. Botvinick, and Christopher Summerfield. Finetuning language models to find agreement among humans with diverse preferences. Neural Information processing systems, 2022. 2
- [6] Kevin Black, Michael Janner, Yilun Du, Ilya Kostrikov, and Sergey Levine. Training diffusion models with reinforcement learning. arXiv preprint arXiv:2305.13301, 2023. 1, 2, 3, 5, 4
- [7] Kevin Clark, Paul Vicol, Kevin Swersky, and David J Fleet. Directly fine-tuning diffusion models on differentiable rewards, 2023. 1, 2, 3, 7
- [8] Katherine Crowson, Stella Biderman, Daniel Kornis, Dashiell Stander, Eric Hallahan, Louis Castricato, and Edward Raff. Vqgan-clip: Open domain image generation and editing with natural language guidance, 2022. 1
- [9] Xiaoliang Dai, Ji Hou, Chih-Yao Ma, Sam Tsai, Jialiang Wang, Rui Wang, Peizhao Zhang, Simon Vandenhende, Xiaofang Wang, Abhimanyu Dubey, et al. Emu: Enhancing image generation models using photogenic needles in a haystack. arXiv preprint arXiv:2309.15807, 2023. 1, 3, 2
- [10] Yann Dubois, Xuechen Li, Rohan Taori, Tianyi Zhang, Ishaan Gulrajani, Jimmy Ba, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. Alpacafarm: A simulation framework for methods that learn from human feedback,

2023. 2

- [11] Ying Fan, Olivia Watkins, Yuqing Du, Hao Liu, Moonkyung Ryu, Craig Boutilier, Pieter Abbeel, Mohammad Ghavamzadeh, Kangwook Lee, and Kimin Lee. Dpok: Reinforcement learning for fine-tuning text-to-image diffusion models. arXiv preprint arXiv:2305.16381, 2023. 1, 2, 3, 5, 4
- [12] Leo Gao, John Schulman, and Jacob Hilton. Scaling laws for reward model overoptimization, 2022. 2
- [13] Divyansh Garg, Shuvam Chakraborty, Chris Cundy, Jiaming Song, Matthieu Geist, and Stefano Ermon. Iq-learn: Inverse soft-q learning for imitation. Neural Information Processing Systems, 2021. 5
- [14] Amelia Glaese, Nat McAleese, Maja Tre˛bacz, John Aslanides, Vlad Firoiu, Timo Ewalds, Maribeth Rauh, Laura Weidinger, Martin Chadwick, Phoebe Thacker, Lucy Campbell-Gillingham, Jonathan Uesato, Po-Sen Huang, Ramona Comanescu, Fan Yang, Abigail See, Sumanth Dathathri, Rory Greig, Charlie Chen, Doug Fritz, Jaume Sanchez Elias, Richard Green, Soˇna Mokrá, Nicholas Fernando, Boxi Wu, Rachel Foley, Susannah Young, Iason Gabriel, William Isaac, John Mellor, Demis Hassabis, Koray Kavukcuoglu, Lisa Anne Hendricks, and Geoffrey Irving. Improving alignment of dialogue agents via targeted human judgements, 2022. 2
- [15] Gabriel Goh, James Betker, Li Jing, Aditya Ramesh, Tim Brooks, Jianfeng Wang, Lindsey Li, Long Ouyang, Juntang Zhuang, Joyce Lee, Prafulla Dhariwal, Casey Chu, Joy Jiao, Jong Wook Kim, Alex Nichol, Yang Song, Lijuan Wang, and Tao Xu. Improving image generation with better captions.

2023. 3, 10, 12

- [16] Caglar Gulcehre, Tom Le Paine, Srivatsan Srinivasan, Ksenia Konyushkova, Lotte Weerts, Abhishek Sharma, Aditya Siddhant, Alex Ahern, Miaosen Wang, Chenjie Gu, Wolfgang Macherey, Arnaud Doucet, Orhan Firat, and Nando de Freitas. Reinforced self-training (rest) for language modeling, 2023. 2
- [17] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. pages 6840–6851, 2020. 3, 4,

- 5

[18] Gabriel Ilharco, Mitchell Wortsman, Ross Wightman, Cade Gordon, Nicholas Carlini, Rohan Taori, Achal Dave, Vaishaal Shankar, Hongseok Namkoong, John Miller, Hannaneh Hajishirzi, Ali Farhadi, and Ludwig Schmidt. Openclip, 2021. If you use this software, please cite it as below.

- 6

- [19] Bahjat Kawar, Shiran Zada, Oran Lang, Omer Tov, Huiwen Chang, Tali Dekel, Inbar Mosseri, and Michal Irani. Imagic: Text-based real image editing with diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6007–6017, 2023. 6
- [20] Diederik P. Kingma, Tim Salimans, Ben Poole, and Jonathan Ho. Variational diffusion models. 2021. 3, 5, 4
- [21] Yuval Kirstain, Adam Polyak, Uriel Singer, Shahbuland Matiana, Joe Penna, and Omer Levy. Pick-a-pic: An open dataset of user preferences for text-to-image generation. arXiv preprint arXiv:2305.01569, 2023. 3, 5, 8
- [22] Harrison Lee, Samrat Phatale, Hassan Mansoor, Kellie Lu, Thomas Mesnard, Colton Bishop, Victor Carbune, and Abhinav Rastogi. Rlaif: Scaling reinforcement learning from human feedback with ai feedback, 2023. 3, 6
- [23] Sergey Levine. Reinforcement learning and control as probabilistic inference: Tutorial and review, 2018. 4
- [24] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017. 5
- [25] Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. Sdedit: Guided image synthesis and editing with stochastic differential equations. arXiv preprint arXiv:2108.01073, 2021. 6, 8
- [26] Jacob Menick, Maja Trebacz, Vladimir Mikulik, John Aslanides, Francis Song, Martin Chadwick, Mia Glaese, Susannah Young, Lucy Campbell-Gillingham, Geoffrey Irving, and Nat McAleese. Teaching language models to support answers with verified quotes, 2022. 2
- [27] Volodymyr Mnih, Adrià Puigdomènech Badia, Mehdi Mirza, Alex Graves, Timothy P. Lillicrap, Tim Harley, David Silver, and Koray Kavukcuoglu. Asynchronous methods for deep reinforcement learning, 2016. 2
- [28] OpenAI. Gpt-4 technical report. ArXiv, abs/2303.08774,

2023. 1

- [29] Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, and Ryan Lowe. Training language models to follow instructions with human feedback, 2022. 2
- [30] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and

- Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis, 2023. 2, 3, 5, 1
- [31] Mihir Prabhudesai, Anirudh Goyal, Deepak Pathak, and Katerina Fragkiadaki. Aligning text-to-image diffusion models with reward backpropagation. arXiv preprint arXiv:2310.03739, 2023. 1, 3
- [32] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 6, 1
- [33] Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model, 2023. 1, 2, 3, 4, 5, 8
- [34] Rajkumar Ramamurthy, Prithviraj Ammanabrolu, Kianté Brantley, Jack Hessel, Rafet Sifa, Christian Bauckhage, Hannaneh Hajishirzi, and Yejin Choi. Is reinforcement learning (not) for natural language processing: Benchmarks, baselines, and building blocks for natural language policy optimization, 2022. 2
- [35] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. Stable diffusion

2. https://huggingface.co/stabilityai/ stable-diffusion-2. Accessed: 2023 - 11 - 16. 1

- [36] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10684–10695, 2022. 1, 3, 5
- [37] Christoph Schuhmann. Laion-aesthetics. https:// laion.ai/blog/laion-aesthetics/, 2022. Accessed: 2023 - 11- 10. 3, 6, 1
- [38] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms, 2017. 2
- [39] Eyal Segalis, Dani Valevski, Danny Lumen, Yossi Matias, and Yaniv Leviathan. A picture is worth a thousand words: Principled recaptioning improves image generation, 2023. 3
- [40] Noam Shazeer and Mitchell Stern. Adafactor: Adaptive learning rates with sublinear memory cost. In International Conference on Machine Learning, pages 4596–4604. PMLR,

2018. 5

- [41] Joar Skalse, Nikolaus H. R. Howe, Dmitrii Krasheninnikov, and David Krueger. Defining and characterizing reward hacking. Neural Information Processing Systems, 2022. 2
- [42] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International conference on machine learning, pages 2256–2265, 2015. 3, 4
- [43] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020. 6
- [44] Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. Advances in neural information processing systems, 32, 2019. 3

- [45] Yang Song, Conor Durkan, Iain Murray, and Stefano Ermon. Maximum likelihood training of score-based diffusion models. In Neural Information Processing Systems, 2021. 3
- [46] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In International Conference on Learning Representations, 2021. 3
- [47] Nisan Stiennon, Long Ouyang, Jeff Wu, Daniel M. Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, and Paul Christiano. Learning to summarize from human feedback. Neural Information Processing Systems, 18, 2020. 2
- [48] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. Llama 2: Open foundation and finetuned chat models, 2023. 1
- [49] Lewis Tunstall, Edward Beeching, Nathan Lambert, Nazneen Rajani, Kashif Rasul, Younes Belkada, Shengyi Huang, Leandro von Werra, Clémentine Fourrier, Nathan Habib, Nathan Sarrazin, Omar Sanseviero, Alexander M. Rush, and Thomas Wolf. Zephyr: Direct distillation of lm alignment, 2023. 2
- [50] Jonathan Uesato, Nate Kushman, Ramana Kumar, Francis Song, Noah Siegel, Lisa Wang, Antonia Creswell, Geoffrey Irving, and Irina Higgins. Solving math word problems with process- and outcome-based feedback, 2022. 2
- [51] Bram Wallace, Akash Gokul, Stefano Ermon, and Nikhil Naik. End-to-end diffusion latent optimization improves classifier guidance. arXiv preprint arXiv:2303.13703, 2023. 3, 1
- [52] Xiaoshi Wu, Yiming Hao, Keqiang Sun, Yixiong Chen, Feng Zhu, Rui Zhao, and Hongsheng Li. Human preference score v2: A solid benchmark for evaluating human preferences of text-to-image synthesis. arXiv preprint arXiv:2306.09341,

2023. 3, 5, 6

- [53] Xiaoshi Wu, Yiming Hao, Keqiang Sun, Yixiong Chen, Feng Zhu, Rui Zhao, and Hongsheng Li. Hpsv2 github. https: //github.com/tgxs002/HPSv2/tree/master,

2023. Accessed: 2023 - 11 - 15. 5

- [54] Qizhe Xie, Minh-Thang Luong, Eduard Hovy, and Quoc V Le. Self-training with noisy student improves imagenet

- classification. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10687– 10698, 2020. 7
- [55] Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong. Imagereward: Learning and evaluating human preferences for textto-image generation. arXiv preprint arXiv:2304.05977,

2023. 3

- [56] Jiahui Yu, Yuanzhong Xu, Jing Yu Koh, Thang Luong, Gunjan Baid, Zirui Wang, Vijay Vasudevan, Alexander Ku, Yinfei Yang, Burcu Karagol Ayan, Ben Hutchinson, Wei Han, Zarana Parekh, Xin Li, Han Zhang, Jason Baldridge, and Yonghui Wu. Scaling autoregressive models for content-rich text-to-image generation, 2022. 5
- [57] Zheng Yuan, Hongyi Yuan, Chuanqi Tan, Wei Wang, Songfang Huang, and Fei Huang. Rrhf: Rank responses to align language models with human feedback without tears. Neural Information Processing Systems, 2023. 2
- [58] Yao Zhao, Rishabh Joshi, Tianqi Liu, Misha Khalman, Mohammad Saleh, and Peter J. Liu. Slic-hf: Sequence likelihood calibration with human feedback, 2023. 2
- [59] Rui Zheng, Shihan Dou, Songyang Gao, Yuan Hua, Wei Shen, Binghai Wang, Yan Liu, Senjie Jin, Qin Liu, Yuhao Zhou, Limao Xiong, Lu Chen, Zhiheng Xi, Nuo Xu, Wenbin Lai, Minghao Zhu, Cheng Chang, Zhangyue Yin, Rongxiang Weng, Wensen Cheng, Haoran Huang, Tianxiang Sun, Hang Yan, Tao Gui, Qi Zhang, Xipeng Qiu, and Xuanjing Huang. Secrets of rlhf in large language models part i: Ppo, 2023. 2
- [60] Barret Zoph, Golnaz Ghiasi, Tsung-Yi Lin, Yin Cui, Hanxiao Liu, Ekin Dogus Cubuk, and Quoc Le. Rethinking pretraining and self-training. Advances in neural information processing systems, 33:3833–3845, 2020. 7

## Diffusion Model Alignment Using Direct Preference Optimization Supplementary Material

### S1. Comparisons to existing work

RL-Based Methods such as [6, 11] have shown effectiveness in operating on a limited set of prompts (< 10 and < 1000 respectively) but do not generalize as well to the open-vocabulary setting as shown in [7, 31]. We found this in our experiments as well, where training using the DDPO scheme did not improve PickScore over the baseline model over a sweep of hyperparameters.

While DDPO [6] is an RL-based method as is DPOK [11], their target objective and distributional guarantees are different. Specifically, DDPO purely aims to optimize the reward function without any KL-regularization

0∼pθ(x0|c)r(x0,c) (15)

Ec∼p(c),x

while DPOK adds in a term governing KL-regularization between the learned distribution and a reference distribution as in our setting. This means that DDPO is optimizing the same objective as DRaFT and AlignProp ([7, 31]) but via RL instead of gradient descent through the diffusion chain. DDPO uses early stopping in lieu of distributional control.

Additionally, through the score function policy gradient estimator employed by DDPO it is observable why the method struggles with open vocabulary generation. The gradient estimation used is

T

pθ(xt−1 | xt,c) pθ

(xt−1 | xt,c) ∇θ log pθ(xt−1 | xt,c) r(x0,c) (16) Here the trajectories {xT,xT−1,...,x0} are generated by the original model pθ

∇θJDDRL = E

old

t=0

. In this formulation, the term

old

pθ(xt−1|xt,c)

pθold simply is an importance weighter which scales gradient contributions based on the relevance of the sample (as determined by how aligned the learned and reference model predictions are). Since the trajectories are generated by the “old” (reference) model, r(x0,c) is only a weighting in the latter term ∇θ log pθ(xt−1 | xt,c) r(x0,c). The gradient encourages higher likelihoods for generations of high reward, but makes no distinction about the diversity of those generations. High-reward prompts can dominate the gradient trajectory, while generations considered lower-reward are ignored or discouraged. This stands in contrast to the DPO framework where the likelihood of a generation is contrasted against another with the same conditioning. This normalization across conditioning prevents sets of c being considered unimportant/undesirable and not being optimized for. In Diffusion-DPO, conditionings with all types of reward magnitudes are weighted equally towards the xw0 and away from the xl0.

Inference Time-Optimization namely DOODL [51], does not learn any new model parameters, instead optimizing diffusion latents to improve some criterion on the generated image similar to CLIP+VQGAN[8]. This runtime compute increases inference cost by more than an order of magnitude.

Reward Maximization Training such as [7, 31] amortize the cost of DOODL from runtime to training. They train by generating images from text prompts, computing a reward loss on the images, and backpropagating gradients through the generative process to improve the loss. While effective in the open-vocabulary setting (also training on Pick-a-Pic prompts), these methods provide no distributional guarantees (unlike the control via β in Diffusion-DPO) and suffer from mode collapse with over-training. These methods do not generalize to all reward functions, with [7] noting the inability of DRaFT to improve image-text alignment using CLIP[32] as a reward function. In contrast, Diffusion-DPO can improve image-text alignment using CLIP preference, as shown in Sec. 5.4. Furthermore, only differentiable rewards can be optimized towards in the reward maximization setting. This necessitates not only data collection but also reward model training.

Dataset Curation As discussed, models such as StableDiffusion variants [30, 36] train on laion-aesthetics [37] to bias the model towards more visually appealing outputs. Concurrent work Emu [9] takes this approach to an extreme. Instead of training on any images from a web-scale dataset which pass a certain model score threshold, they employ a multi-stage

pipeline where such filtering is only the first stage. Subsequently, crowd workers filter the subset down using human judgement and at the final stage expert in photography are employed to create the dataset. While effective, this process has several drawbacks compared to Diffusion-DPO. First, necessitating training on existing data can be a bottleneck, both in terms of scale and potential applications. While [9] reports lesser text faithfulness improvements as well, these are likely due to the hand-written captions, a much more costly data collection stage than preferences. The Emu pipeline is not generalizable to different types of feedback as DPO is (e.g. outside of recaptioning it is non-obvious how such an approach can improve text-image alignment).

### S2. Details of the Primary Derivation Starting from Eq. (5), we have

θ(x0|c)r(c,x0)/β + DKL(pθ(x0|c)||pref(x0|c)) ≤ min

− Ep

min

pθ

− Ep

θ(x0|c)r(c,x0)/β + DKL (pθ(x0:T|c)||pref(x0:T|c))

pθ

− Ep

θ(x0:T|c)R(c,x0:T)/β + DKL (pθ(x0:T|c)||pref(x0:T|c))

= min

pθ

pθ(x0:T|c) pref(x0:T|c)exp(R(c,x0:T)/β)/Z(c) − log Z(c)

Ep

= min

θ(x0:T|c) log

pθ

DKL (pθ(x0:T|c)∥pref(x0:T|c)exp(R(c,x0:T)/β)/Z(c)).

= min

pθ

(17)

where Z(c) = x pref(x0:T|c)exp(r(c,x0)/β) is the partition function. The optimal p∗θ(x0:T|c) of Equation (17) has a unique closed-form solution:

##### p∗θ(x0:T|c) = pref(x0:T|c)exp(R(c,x0:T)/β)/Z(c),

Therefore, we have the reparameterization of reward function

p∗θ(x0:T|c) pref(x0:T|c)

R(c,x0:T) = β log

+ β log Z(c).

Plug this into the definition of r, hence we have

p∗θ(x0:T|c) pref(x0:T|c)

r(c,x0) = βEp

θ(x1:T|x0,c) log

+ β log Z(c).

Substituting this reward reparameterization into maximum likelihood objective of the Bradly-Terry model as Eq. (4), the partition function cancels for image pairs, and we get a maximum likelihood objective defined on diffusion models, its per-example formula is:

pθ(xw0:T) pref(xw0:T) − log

pθ(xl0:T) pref(xl0:T)

LDPO-Diffusion(θ) = −log σ βExw

1:T∼pθ(x1:T|xw0 ),xl1:T∼pθ(xl1:T|xl0) log

where xw0 ,xl0 are from static dataset, we drop c for simplicity.

An approximation for reverse process Since sampling from pθ(x1:T|x0) is intractable, we utilize q(x1:T|x0) for approximation.

pθ(xw0:T) pref(xw0:T) − log

pθ(xl0:T) pref(xl0:T)

L1(θ) = −log σ βExw

1:T∼q(x1:T|xw0 ),xl1:T∼q(x1:T|xl0) log

T

pθ(xwt−1|xwt ) pref(xwt−1|xwt ) − log

pθ(xlt−1|xlt) pref(xlt−1|xt)

= −log σ βExw

log

1:T ∼q(x1:T |xw0 ),xl1:T ∼q(x1:T |xl0)

t=1

pθ(xwt−1|xwt ) pref(xwt−1|xwt ) − log

pθ(xlt−1|xlt) pref(xlt−1|xt)

= −log σ βExw

1:T∼q(x1:T|xw0 ),xl1:T∼q(x1:T|xl0)TEt log

pθ(xwt−1|xwt ) pref(xwt−1|xwt ) − log

pθ(xlt−1|xlt) pref(xlt−1|xt)

= −log σ βTEtExw

t−1,t∼q(xt−1,t|xw0 ),xlt−1,t∼q(xt−1,t|xl0) log

(18)

= −log σ βTEt,xw

t ∼q(xt|xw0 ),xlt∼q(xt|xl0)

pθ(xwt−1|xwt ) pref(xwt−1|xwt ) − log

pθ(xlt−1|xlt) pref(xlt−1|xt)

Exw

t−1∼q(xt−1|xwt ,xw0 ),xlt−1∼q(xt−1|xlt,xl0) log

By Jensen’s inequality, we have

L1(θ) ≤ −Et,xw

t ∼q(xt|xw0 ),xlt∼q(xt|xl0) log σ

pθ(xlt−1|xlt) pref(xlt−1|xt)

pθ(xwt−1|xwt ) pref(xwt−1|xwt ) − log

βTExw

t−1∼q(xt−1|xwt ,xw0 ),xlt−1∼q(xt−1|xlt,xl0) log

t ∼q(xt|xw0 ),xlt∼q(xt|xl0) log σ −βT DKL(q(xwt−1|xw0,t)∥pθ(xwt−1|xwt )) − DKL(q(xwt−1|xw0,t)∥pref(xwt−1|xwt ))

= −Et,xw

− DKL(q(xlt−1|xl0,t)∥pθ(xlt−1|xlt)) + DKL(q(xlt−1|xl0,t)∥pref(xlt−1|xlt) Using the Gaussian parameterization of the reverse process (Eq. (1)), the above loss simplifies to:

##### L1(θ) ≤ −Et,ϵw,ϵl log σ (−βTω(λt)( ∥ϵw−ϵθ(xwt ,t)∥2−∥ϵw−ϵref(xwt ,t)∥2 − ∥ϵl − ϵθ(xlt,t)∥2 − ∥ϵl − ϵref(xlt,t)∥2

where ϵw,ϵl ∼ N(0,I), xt ∼ q(xt|x0) thus xt = αtx0 + σtϵ. Same as Eq. (2), λt = αt2/σt2 is a signal-to-noise ratio term [20], in practice, the reweighting assigns each term the same weight [17].

An alternative approximation Note that for Eq. (18) we utilize q(x1:T|x0) to approximate pθ(x1:T|x0). For each step, it is to use q(xt−1,t|x0) to approximate pθ(xt−1,t|x0). Alternatively, we also propose to use q(xt|x0)pθ(xt−1|xt) for approximation. And this approximation yields lower error because DKL(q(xt|x0)pθ(xt−1|xt)∥pθ(xt−1,t|x0)) = DKL(q(xt|x0)∥pθ(xt|x0)) < DKL(q(xt−1,t|x0)∥pθ(xt−1,t|x0)).

pθ(xw0:T) pref(xw0:T) − log

pθ(xl0:T) pref(xl0:T)

LDPO-Diffusion(θ) = −log σ βExw

1:T∼pθ(x1:T|xw0 ),xl1:T∼pθ(x1:T|xl0) log

T

pθ(xwt−1|xwt ) pref(xwt−1|xwt ) − log

pθ(xlt−1|xlt) pref(xlt−1|xt)

= −log σ βExw

log

1:T ∼pθ(x1:T |xw0 ),xl1:T ∼pθ(x1:T |xl0)

t=1

pθ(xlt−1|xlt) pref(xlt−1|xt)

pθ(xwt−1|xwt ) pref(xwt−1|xwt ) − log

= −log σ βExw

1:T∼pθ(x1:T|xw0 ),xl1:T∼pθ(x1:T|xl0)TEt log

pθ(xwt−1|xwt ) pref(xwt−1|xwt ) − log

pθ(xlt−1|xlt) pref(xlt−1|xt)

= −log σ βTEtExw

t−1,t∼pθ(xt−1,t|xw0 ),xlt−1,t∼pθ(xt−1,t|xl0) log

.

By approximating pθ(xt−1,t|x0) with q(xt|x0)pθ(xt−1|xt), we have

pθ(xwt−1|xwt ) pref(xwt−1|xwt ) − log

pθ(xlt−1|xlt) pref(xlt−1|xt)

L2(θ) = −log σ βTEtExw

t−1,t∼q(xt|xw0 )pθ(xt−1|xwt ),xlt−1,t∼q(xt|xl0)pθ(xt−1|xlt) log

pθ(xwt−1|xwt ) pref(xwt−1|xwt ) − log

pθ(xlt−1|xlt) pref(xlt−1|xt)

= −log σ βTEt,xw

t ∼q(xt|xw0 ),xlt∼q(xt|xl0)Exw

t−1∼pθ(xt−1|xwt ),xlt−1∼pθ(xt−1|xlt) log

By Jensen’s inequality, we have

pθ(xwt−1|xwt ) pref(xwt−1|xwt ) − log

pθ(xlt−1|xlt) pref(xlt−1|xt)

t ∼q(xt|xw0 ),xlt∼q(xt|xl0) log σ βTExw

L2(θ) ≤ −Et,xw

t−1∼pθ(xt−1|xwt ),xlt−1∼pθ(xt−1|xlt) log

t ∼q(xt|xw0 ),xlt∼q(xt|xl0) log σ βT DKL(pθ(xwt−1|xwt )∥pref(xwt−1|xwt )) − DKL(pθ(xlt−1|xlt)∥pref(xlt−1|xlt) Using the Gaussian parameterization of the reverse process (Eq. (1)), the above loss simplifies to:

= −Et,xw

L2(θ) = −Et,ϵw,ϵl log σ −βTω(λt) ∥ϵθ(xwt ,t) − ϵref(xwt ,t)∥2 − ∥ϵθ(xlt,t) − ϵref(xlt,t)∥2

where ϵw,ϵl ∼ N(0,I), xt ∼ q(xt|x0) thus xt = αtx0 + σtϵ. Same as Eq. (2), λt = αt2/σt2 is a signal-to-noise ratio term [20], in practice, the reweighting assigns each term the same weight [17].

### S3. Alternate Derivation: Reinforcement Learning Perspective

We can also derive our objective as a multi-step RL approach, in the same setting as [6, 11]. A Markov Decision Process (MDP) is a tuple (S,A,ρ0,P,R), where S is the state space, A is the action space, ρ0 is an initial state distribution, P is the transition dynamics and R is the reward function. In this formulation, at each time step t a policy π(at|st) observes a state st ∈ S and takes an action at ∈ A. The environment then transitions to a next state st+1 ∼ P(st+1|st,at) and the returns a reward R(st,at).The goal of the policy is to maximize the total rewards it receives. Prior works [6, 11] map the denoising process in diffusion model generation to this formulation via:

st ≜ (c,xt,t) at ≜ xt

P(st+1|st,at) ≜ (δc,δt−1,δx

) ρ(s0) ≜ (p(c),δT,N(0,I))

t−1

R(st,at) =

r(c,x0) if t = 0 0 otherwise

(19)

where c is the prompt xt is the time-step t nosy image and δy is the Dirac delta function with unit density at y. That is in this formulation we consider the denoising model as a policy, with each denoising step a step in an MDP. The objective of the policy is to maximize the reward (alignment with human preference) of the final image. In the derivation below, we drop the time step t for brevity. In this formulation the generative model is a policy and the denoising process is a rollout in an MDP with a sparse reward received for the final generated image. Following [11] we optimize the following objective

Ec∼D,p

θ

0

r(c,xt) − βDKL[pθ(xt−1|xt,c)||pref(xt−1|xt,c)] (20)

t=T

While prior works [6, 11] use policy gradient approaches to optimize this objective, we’re going to use off-policy methods. Following Control as Variational Inference [23], we have the following

Q∗((xt,c),xt−1) = r(c,xt) + V ∗(xt−1,c) (21) V ∗(xt−1,c) = β log Ep

##### [expQ∗((xt,c),xt−1)/β] (22)

ref

.

∗((xt,c),xt−1)−V ∗(xt,c))/β (23)

p∗(xt−1|(xt,c)) = pref(xt−1|xt,c)e(Q

where V ∗ is the optimal value function and Q∗ is the optimal state-action value function (in tour definition of the denoising MDP, he policy is stochastic, but the dynamics is deterministic). Also notice that in Eq. 23 the equation is exact since the right-hand side integrates to 1. We then consider the inverse soft Bellman operator [13] and have the following

r(c,xt) = V ∗(xt−1,c) − Q∗((xt,c),xt−1) (24) However, from Eq. 23 we have

substituting in Eq. 24 we obtain:

p∗(xt−1|xt,c) pref(xt−1|xt,c)

Q∗((xt,c),xt−1) − V ∗(xt,c) = log

(25)

p∗(xt−1|xt,c) pref(xt−1|xt,c) − V ∗(xt,c) (26)

r(c,xt) = V ∗(xt−1,c) + log

Using a telescoping sum through the diffusion chain we are left with

T

r(c,x0) =

t=0

p∗(xt−1|xt,c) pref(xt−1|xt,c) − V ∗(xT,c) (27)

log

since by definition all intermediate rewards are zero. If we assume both diffusion chains start from the same state and plug this result into the preference formulation of Eq. 3 we obtain the objective of Eq. 11. Here we optimize the same objective

- as prior works [6, 11], but instead of a policy gradient approach we derive our objective as an off-policy learning problem in the same MDP. This not only simplifies the algorithm significantly, but justifies our sampling choices in Eq. 13 and we do not have to sample through the entire difussion chain.

S4. Alternative Derivation: Noise-Aware Preference Model

Paralleling the original DPO formulation we consider a policy trained on maximizing the likelihood of p(x0|c,t,xobs) where xobs is a noised version of x0. Here x0 is an image, c is a text caption, t is a noising scale, and xobs is a corruption (noised version) of x0. We initialize from a reference diffusion policy pref. We aim to optimize the same RL objective of Eq. (5), reprinted here for convenience:

max

pθ

Ec∼D,x

0∼pθ(x0|c) [r(c,x0)] − βDKL [pθ(x0|c)∥pref(x0|c)] (28) Our policy has additional conditioning (t,xobs). The latter is a noised version of x0. Define the space of noising operators

- at time t as Qt where qt ∼ Qt with qt(x0) = √αtx0 + √1 − αϵq

t ∼ N(0,I). Here qt refers to the linear transform corresponding with a specific gaussian draw ∼ N(0,I) and the set of qt is Qt. In general at some time level t we have yobs = qt(x0) for some qt ∼ Qt so can write the conditioning as p(x0|c,t,qt(y)). We rewrite Eq. (28) as

,ϵq

t

0∼p(θgen)(x0|c),t∼U{0,T},qt∼QT rϕ(c,x0) − βDKL pθ(x0 | c,t,qt(x0)) || pref(x0 | c,t,qt(x0)) (29) p(gen) denoting the generative process associated with p as a diffusion model. Note that the reward model is the same

Ec∼D,x

max

pθ

formulation as in DPO. The optimal policy now becomes

1 Z(c,t,qt)

p∗θ(x0 | c,t,qt(x0)) =

pref(x0 | c,t,qt(x0))exp

1 β

r(c,x0) (30)

with Z a partition over captions, timesteps, and noising draws. Rearranging for r(c,x0) now yields

p∗θ(x0 | c,t,qt) pref(x0 | c,t,qt)

+ β log Z(c,t,qt), ∀t,qt (31)

r(c,x0) = β log

We have not changed the reward model formulation at all, but our policies have extra conditioning as input (which ideally the likelihoods are constant with respect to). Putting this formulation into the original Bradley-Terry framework of Eq. (3) (re-printed here)

pBT(xw0 ≻ xl0|c) = σ(r(c,xw0 ) − r(c,xl0)) (32) results in the objective:

pθ(xw0 | c,t,qt(xw0 )) pref(xw0 | c,t,qt(xw0 )) − β log

pθ(xl0 | c,t,qt(xl0)) pref(xl0 | c,t,qt(xl0))

LDPO(pθ;pref) = −E(xw

0 ,xl0∼p(gen)(c),c∼D,t∼U{0,T},qt∼Qt log σ β log

(33)

We now consider how to compute these likelihoods. Using the notation at = √αt and bt = √1 − αt as shorthand for commonly-used diffusion constants (α are defined as in DDIM[43]) we have

xobs = qt(x0) = atx0 + btϵ,ϵ ∼ N(0,I) (34) We use Eq. 57 from DDIM[43] (along with their definition of σt):

p(x0|xt) = N(xpred0 ,σt2I) (35) Our xpred0 is:

xobs − btϵpredθ at

xpred0 =

atx0 + btϵ − btϵpredθ at

=

= x0 +

Here ϵpredθ is the output of ϵθ(c,t,xobs) Making the conditional likelihood:

bt at

(ϵ − ϵpredθ ) (36)

b2

bt at

1 (2πσt2)d/2

||ϵ−ϵpredθ ||22 (37) For convenience we define

t 2a2

e−

(ϵ − ϵpredθ ),σt2I) =

t σ2

pθ(x0|c,t,xobs) = N(x0;x0 +

t

1 (2πσt2)d/2

(38)

zt =

SE = ||ϵ − ϵpred||22 (39) We will decorate the latter quantity (SE) with sub/superscripts later. For now we get:

b2

t 2a2

pθ(x0|c,t,xobs) = zte−

SE (40) We see to minimize

t σ2

t

pθ(xw0 |c,t,qt(xw0 )) pref(xw0 |c,t,qt(xw0 ) − log

pθ(xl0|c,t,qt(xl0)) pref(xl0|c,t,qt(xl0)

##### E

−log σ β log

(xw0 ,xl0∼p(gen)(c);c∼D,;t∼U{0,T};qt∼Qt

b2

b2

SEθ(w)

SEθ(l)

t 2a2

t 2a2

zte−

zte−

t σ2

t σ2

t

t

##### E

−log σ β log

− log

b2

b2

SEref(w)

SEref(l)

(xw0 ,xl0∼p(gen)(c);c∼D,;t∼U{0,T};qt∼Qt

t 2a2

t 2a2

zte−

zte−

t σ2

t σ2

t

t

= (41)

(42)

##### Here we use SEψ(d) = ||ϵq

t − ψ(c,t,qt(xd0))||22 to denote the L2 error in the noise prediction of model ψ operating on

the noisy qt(xd0) with corresponding conditioning (c,t) (d ∈ {w,l}). Here the model associated with SEref∗ is the model of the reference policy pref. Note that these SE terms are the standard diffusion training objective from Eq. (2). Continuing to simplify the above yields:

  β

  log

− log σ

= − log σ −β

We can simplify the coefficient:

  

   (43)

b2

b2

SEθ(w)

SEθ(l)

t 2a2

t 2a2

zte−

zte−

t σ2

t σ2

t

t

− log

b2

b2

SEref(w)

SEref(l)

t 2a2

t 2a2

zte−

zte−

t σ2

t σ2

t

t

b2t 2a2tσt2

(SEθ(w) − SEref(w)) − (SEθ(l) − SEref(l)) (44)

σt2+1

b2t a2tσt2

1 − αt αt

1 σt2

σt2 ≈ 1 (45) Resulting in objective

=

=

β 2

(SEθ(w) − SEref(w)) − (SEθ(l) − SEref(l)) (46)

##### ≈ E

−log σ −

x,yw,yl∼D;t;ϵ∼N(0,I)

Up to the approximation of Eq. (45) this is the equivalent to Eq. (33). The log of the likelihood ratios simply take on the elegant form of a difference in diffusion training losses. Due to the equation negatives and log σ being a monotonic increasing function, by minimizing Eq. (46) we are aiming to minimize the inside term

(SEθ(w) − SEref(w)) − (SEθ(l) − SEref(l)) (47)

This can be done by minimizing SEθ(w) or maximizing SEθ(l), with the precise loss value depending on how these compare to the reference errors SEref(w),SEref(l). The asymmetry of the log σ function allows β to control the penalty for deviating from the reference distribution. A high β results in a highly assymetric distribution, disproportionately penalizing low SEθ(l) and high SEθ(w) and encouraging a pθ to make less mistakes in implicitly scoring yw,yl by deviating less from the reference policy pref. We visualize the log σ curves in Figure S1 for several values of β.

y = log (x)

- = 0.5

- = 1

- = 2

20.0

17.5

15.0

12.5

10.0

7.5

5.0

2.5

0.0

10.0 7.5 5.0 2.5 0.0 2.5 5.0 7.5 10.0

Figure S1. Visualization of y = − log σ(βx)

#### S4.1. Reward Estimation

Finally, we note that in this formulation that if we wish to compute the noise-aware reward difference r(c,xA0 ) − r(c,xB0 ), referring to Eq. (31) this now has form

r(c,xA0 ) − r(c,xB0 ) = β(SEθA − SErefA) + β log Z(c,t,qt) − β(SEθB − SErefB) + β log Z(c,t,qt) , ∀c,t,qt (48) = β (SEθA − SErefA) − (SEθB − SErefB) , ∀c,t,qt (49) (50)

Which means for two images (xA0 ,xB0 ) with the same conditioning c we can estimate the reward difference using Eq. (48). When doing this it improves the estimate to average over multiple draws (t,qt). We use this method in Table S2.

### S5. β Ablation

[Figure 73]

Figure S2. Median PickScores for generations on the Pick-a-Pic v2 validation set for different choices of β

For β far below the displayed regime, the diffusion model degenerates into a pure reward scoring model. Much greater, and the KL-divergence penalty greatly restricts any appreciable adaptation.

### S6. Further SFT Discussions

We also partially attribute this difference in effectiveness of SFT to the gap in pretraining vs. downstream task considered in the original DPO paper [33] vs. our work. On two of the DPO LLM tasks (sentiment generation, single-turn dialogue), generic off-the-shelf autoregressive language models are tuned on specific tasks in the SFT stage. In the final setting, summarization, the SFT model has been pretrained on a similar task/dataset. In this case, finetuning on the “preferred” dataset (preferred-FT) baseline performs comparably to the SFT initialization.

This final setting is most analogous to that of Diffusion-DPO. The generic pretraining, task, and evaluation setting are all text-to-image generation. There is no task-specific domain gap and all of the settings are open-vocabulary with a broad range of styles. As such, our findings are similar to that of summarization in [33] where an already task-tuned model does not benefit from preferred finetuning.

### S7. Additional Automated Metrics

Automated metrics on Pick-a-Pic validation captions are shown in Figure S3 for DPO-SDXL. The y-axis measures the fraction of head-to-head generation comparisions for a prompt that DPO-SDXL scores higher than the baseline SDXL.

### S8. PickScore Rejection Sampling

Rejection sampling was used in [21] as a powerful inference-time tool. 100 samples were drawn from variants of a prompt and PickScore-ranked, with the highest scored images being compared to a single random draw. PickScore selections were human-preferred 71.4% of the time. We compare using additional compute at inference vs. additional training in Figure S4. We plot the expected PickScore win rate of n draws from the reference model against a single draw from the learned (DPO) model. The mean inference compute for baseline rejection sampling to surpass the DPO-trained model is 10× higher in both cases. For 7% (SDXL) and 16% (SD1.5) of the prompts even 100 draws is insufficient.

[Figure 74]

Figure S3. Box plots of automated metrics vs. SDXL baseline. All 500 unique prompts from PickScore validation set.

[Figure 75]

- Figure S4. The number of draws from the reference model vs. the probability that maximum PickScore of the draws exceeds a single DPO generation. 500 PickScore validation prompts used. Mean (including 100s)/Median: SDXL (13.7, 3), SD1.5 (25.6, 7).

### S9. Pseudocode for Training Objective

def loss(model, ref_model, x_w, x_l, c, beta): """ # This is an example psuedo-code snippet for calculating the Diffusion-DPO loss # on a single image pair with corresponding caption

model: Diffusion model that accepts prompt conditioning c and time conditioning t ref_model: Frozen initialization of model x_w: Preferred Image (latents in this work) x_l: Non-Preferred Image (latents in this work) c: Conditioning (text in this work) beta: Regularization Parameter

returns: DPO loss value """ timestep = torch.randint(0, 1000) noise = torch.randn_like(x_w) noisy_x_w = add_noise(x_w, noise, t) noisy_x_l = add_noise(x_l, noise, t)

model_w_pred = model(noisy_x_w, c, t) model_l_pred = model(noisy_x_l, c, t) ref_w_pred = ref(noisy_x_w, c, t) ref_l_pred = ref(noisy_x_l, c, t)

model_w_err = (model_w_pred - noise).norm().pow(2) model_l_err = (model_l_pred - noise).norm().pow(2) ref_w_err = (ref_w_pred - noise).norm().pow(2) ref_l_err = (ref_l_pred - noise).norm().pow(2)

w_diff = model_w_err - ref_w_err l_diff = model_l_err - ref_l_err

inside_term = -1 * beta * (w_diff - l_diff) loss = -1 * log(sigmoid(inside_term)) return loss

### S10. Additional Qualitative Results

In Figure S6 we present generations from DPO-SDXL on complex prompts from DALLE3 [15]. Other generations for miscellaneous prompts are shown in Figure S5. In Fig. S 7 and 8 we display qualitative comparison results from HPSv2 with random seeds from our human evaluation for prompt indices 200, 600, 1000, 1400, 1800, 2200, 2600, 3000.

[Figure 76]

- Figure S5. DPO-SDXL gens on miscellaneous prompts Prompts (clockwise) (1) A bulldog mob boss, moody cinematic feel (2) A old historical notebook detailing the discovery of unicorns (3) A purple raven flying over a forest of fall colors, imaginary documentary (4) Small dinosaurs shopping in a grocery store, oil painting (5) A wolf wearing a sheep halloween costume going trick-or-treating at the farm

(6) A mummy studying hard in the library for finals, head in hands

[Figure 77]

- Figure S6. DPO-SDXL gens on prompts from DALLE3 [15] Prompts: (1): A swirling, multicolored portal emerges from the depths of an ocean of coffee, with waves of the rich liquid gently rippling outward. The portal engulfs a coffee cup, which serves as a gateway to a fantastical dimension. The surrounding digital art landscape reflects the colors of the portal, creating an alluring scene of endless possibilities. (2): In a fantastical setting, a highly detailed furry humanoid skunk with piercing eyes confidently poses in a medium shot, wearing an animal hide jacket. The artist has masterfully rendered the character in digital art, capturing the intricate details of fur and clothing texture"

[Figure 78]

- Figure S7. Prompts: (1) A kangaroo wearing an orange hoodie and blue sunglasses stands on the grass in front of the Sydney Opera House, holding a sign that says Welcome Friends. (2) Anime Costa Blanca by Studio Ghibli. (3) There is a secret museum of magical items inside a crystal greenhouse palace filled with intricate bookshelves, plants, and Victorian style decor. (4) A depiction of Hermione Granger from the Harry Potter series as a zombie.

[Figure 79]

###### Figure S8. (1) A portrait art of a necromancer, referencing DND and War craft. (2) Monalisa painting a portrait of Leonardo Da Vinci. (3) There is a cyclist riding above all the pigeons. (4) A woman holding two rainbow slices of cake.

