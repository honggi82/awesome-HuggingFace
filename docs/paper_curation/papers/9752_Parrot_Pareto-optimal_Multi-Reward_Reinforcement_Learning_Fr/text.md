Prompt Expansion

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

Kitchen painting, art and cuisine cookware review

# Parrot: Pareto-optimal Multi-Reward Reinforcement Learning Framework for Text-to-Image Generation

Seung Hyun Lee1,6∗ , Yinxiao Li1 , Junjie Ke1 , Innfarn Yoo2 , Han Zhang3 , Jiahui Yu4† , Qifei Wang2 , Fei Deng2,5∗ , Glenn Entis1 ,

Junfeng He1 , Gang Li1 , Sangpil Kim6 , Irfan Essa1 , Feng Yang1

Google Research1, Google2, Google DeepMind3, OpenAI4, Rutgers University5, Korea University6

“Primitive drawing of a heart with “A pineapple surfing on a wave” a smiley face in the middle”

“A small house with trees in the background”

“An F1 race car in a Manhattan street”

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

StableDiffusion1.5Parrot

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

###### ↑ ↑ ↑ ↑

Aesthetics Human Preference Text-Image Alignment Image Sentiment

- Fig. 1: Parrot visual examples. Parrot consistently improves the quality of generated images across multiple criteria: aesthetics, human preference, text-image alignment, and image sentiment. Each column shows generated images using the same seed.

“Cozy living room with a painting of a corgi on the wall above a couch and a round coffee table in front of a couch and a vase of flowers on a coffee table”

“A boat” “A painting of a horse in a field of flowers”

“A cartoon of a happy car on the road”

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

Abstract. Recent works have demonstrated that using reinforcement learning (RL) with multiple quality rewards can improve the quality of generated images in text-to-image (T2I) generation. However, manually adjusting reward weights poses challenges and may cause overoptimization in certain metrics. To solve this, we propose Parrot, which addresses the issue through multi-objective optimization and introduces an effective multi-reward optimization strategy to approximate Pareto optimal. Utilizing batch-wise Pareto optimal selection, Parrot automatically identifies the optimal trade-off among different rewards. We use the novel multi-reward optimization algorithm to jointly optimize the T2I model and a prompt expansion network, resulting in significant improvement of image quality and also allow to control the trade-off of

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

* This work was done during an internship at Google. † This work was done during working at Google.

A cocktail

A cat reading a book

→a cat reading a book in a beautiful garden.

different rewards using a reward related prompt during inference. Furthermore, we introduce original prompt-centered guidance at inference time, ensuring fidelity to user input after prompt expansion. Extensive experiments and a user study validate the superiority of Parrot over several baselines across various quality criteria, including aesthetics, human preference, text-image alignment, and image sentiment.

## 1 Introduction

Despite significant advancements in text-to-image (T2I) generation [17,29,38, 45,55], recent work like Imagen [42] and Stable Diffusion [39], still struggle to produce high quality images. Images in the first row in Fig. 1 illustrates such quality issues in Stable Diffusion [39], including poor composition (e.g. bad cropping), misalignment with input prompts (e.g. missing objects), or overall lack of aesthetic appeal. Assessing the quality of generated images can involve various metrics such as aesthetics [23,24,33], human preference [26], text-image alignment [35], and emotional appeal [44]. Enhancing T2I generation across multiple quality metrics remains a challenging task.

Recent works [4,12,15,27] have demonstrated that incorporating quality signals as reward functions in fine-tuning T2I with reinforcement learning (RL) can improve image quality. For instance, Promptist [15] fine-tunes prompt expansion model using RL with the sum of aesthetics and text-image alignment scores. However, the simple weighted sum approach may not effectively handle trade-offs among multiple quality metrics. As the number of rewards increases, manually adjusting reward weights becomes impractical. Moreover, optimizing one quality metric may inadvertently compromise others, as the model might prioritize aesthetics over relevance to the input prompt. Additionally, the trade-off for different reward is not controllable after training.

To address these challenges, we propose Parrot, a Pareto-optimal multireward reinforcement learning algorithm to improve text-to-image generation. Unlike previous approaches that treat T2I reward optimization as a single objective optimization problem, Parrot tackles this challenge through multi-objective optimization and introduces an effective multi-reward optimization strategy to achieve Pareto optimal approximation. Intuitively, each generated sample in a batch embodies a distinctive trade-off among various quality rewards, with some samples exhibiting superior trade-offs compared to others. Instead of updating gradients using all batch samples, Parrot uses non-dominated points [32] which have better trade-offs. Consequently, Parrot automatically learns from the optimal trade-off among different rewards. Moreover, Parrot learns reward-specific preference prompts, which can be utilized individually or in combination to control the trade-off among different rewards during inference time. Unlike prior work, which either solely fine-tunes the T2I model [12] or only tunes the prompt expansion network while freezing the T2I model [15], we employ the Parrot multi-reward optimization algorithm to jointly optimize both the T2I model and the prompt expansion network. This collaborative optimization unlocks the

full potential of Parrot by encouraging both more details from added context from the prompt expansion model, and the overall quality improvement on the T2I generation. During inference, we further introduce original prompt-centered guidance to ensure the output image is relevant to input prompts after prompt expansion.

In summary, our contributions can be listed as follows:

- – We propose Parrot, a novel multi-reward optimization algorithm for T2I RL fine-tuning. Leveraging batch-wise Pareto-optimal selection, it effectively optimizes multiple T2I rewards, enabling collaborative improvement in aesthetics, human preference, image sentiment, and text-image alignment and also allowing to control the trade-off of different rewards using a reward related prompt during inference.
- – We show the advantage of jointly optimizing both the prompt expansion network and the T2I model, which has never been explored before.
- – We introduce original prompt-centered guidance during inference time after prompt expansion, ensuring better alignment with the original prompt while enriching image details.
- – Extensive results and a user study validate that Parrot outperforms several baseline methods across various quality criteria.

## 2 Related Work

T2I Generation: The goal of T2I generation is to create an image given an input text prompt. Several T2I generative models have been proposed and have demonstrated promising results [5, 7, 14, 21, 22, 28, 37, 41, 42, 54]. Stable Diffusion [39] shows impressive generation performance in T2I generation, leveraging latent text representations from LLMs. Despite substantial progress, the images generated by those models still exhibit quality issues, such as bad cropping or misalignment with the input texts.

RL for T2I Fine-tuning: Starting by Fan et al. [11] to explore RL fine-tuning for T2I models, following works [4, 8, 9, 12, 16] have explored RL fine-tuning technique for T2I diffusion model, showcasing superior performance for human preference learning. DPOK [12] improves quality through RL using ImageReward [51] score as a reward with a few prompts. In addition to fine-tuning the T2I model directly using RL, Promptist [15] fine-tunes the prompt expansion model by using a simple sum of aesthetic and text-image alignment scores as reward. DRaFT [6] proposed not only differentiable rewards for efficient finetuning but also effectiveness of using linear summation of multi-rewards. These methods treat T2I RL as a single-objective optimization problem, while Parrot employs multi-objective optimization. Additionally, prior approaches either finetune the T2I model or the prompt expansion model while freezing the other. In contrast, Parrot proposes joint optimization of the prompt expansion model and the T2I model using multi-reward RL to foster better collaboration.

Multi-objective Optimization: Multi-objective optimization problem involves

optimizing multiple objective functions simultaneously. The scalarization technique [31,47] formulates multi-objective problem into single-objective problems with the weighted sum of each score, which requires pre-defined weights for each objective. Rame et al. [36] proposed weighted averaging method to find Pareto frontier, leveraging multiple fine-tuned models. Lin et al. [30] proposes to learn a set model to map trade-off preference vectors to their corresponding Pareto solutions. Inspired by this, Parrot introduces a language-based preference vector constructed from task identifiers for each reward, then encoded by the text-encoder. In the context of multi-reward RL for T2I diffusion models, Promptist [15] uses a simple weighted sum of two reward scores. This approach requires manual tuning of the weights, which makes it time-consuming and hard to scale when the number of rewards increases.

Generated Image Quality: The quality assessment of images generated by T2I models involves multiple dimensions, and various metrics have been proposed. In this paper, we consider using four types of quality metrics as rewards: aesthetics, human preference, text-image alignment, and image sentiment. Aesthetics captures the overall visual appealingness of the image, and it is learned using human ratings for aesthetics in real images [13,19,23,24,33,48,52]. Human preferences, rooted the concept of learning from human feedback [3,34,50], involves gathering preferences at scale by having raters to compare generated images [26,51]. Text-image alignment measures the extent to which the generated image aligns with the input prompt, CLIP [35] score is often employed, measuring the cosine distance of between contrastive image embedding and text embedding. Image sentiment is important for ensuring the generated image evokes positive emotions in the viewer. Serra et al. [44] predict average polarity of sentiments an image elicits and learn estimates for positive, neutral, and negative scores. In Parrot, we use its positive score as a reward for positive emotions.

## 3 Preliminary

Diffusion Probabilistic Models: Diffusion probabilistic models [17] generate the image by gradually denoising a noisy image. Specifically, given a real image x0 from the data distribution x0 ∼ q(x0), the forward process q(xt|x0,c) of diffusion probabilistic models produce a noisy image xt, which induces a distribution p(x0,c) conditioned on text prompt c. In classifier-free guidance [18], denoising model predicts noise ϵ¯θ with a linear combination of the unconditional score estimates ϵθ(xt,t) and the conditional score estimates ϵθ(xt,t,c) as follows:

ϵ¯θ = w · ϵθ(xt,t,c) + (1 − w) · ϵθ(xt,t,null), (1)

where t denotes diffusion time step, the null indicates a null text and w represents the guidance scale of classifier-free guidance where w ≥ 1. Note that ϵθ is typically parameterized by the UNet [40].

RL-based T2I Diffusion Model Fine-tuning: Given a reward signal from generated images, the goal of RL-tuning for T2I diffusion models is to optimize the policy defined as one denoising step of T2I diffusion models. In particular, Black et al. [4] apply policy gradient algorithm, which regards the denoising process of diffusion models as a Markov decision process (MDP) by performing multiple denoising steps iteratively. Subsequently, a black box reward model r(·,·) predicts a single scalar value from sampled image x0. Given text condition c ∼ p(c) and image x0, objective function J can be defined to maximize the expected reward as follows:

θ(x0|c)[r(x0,c)], (2) where the pre-trained diffusion model pθ produces a sample distribution pθ(x0|c) using text condition c. Modifying this equation, Fan et al. [12] demonstrate that the gradient of objective function ∇Jθ can be calculated through gradient ascent algorithm without using the gradient of reward model as follows:

Jθ = Ep(c)Ep

T

∇θ log pθ(xt−1|c,t,xt)], (3)

∇Jθ = E[r(x0,c)

t=1

where T denotes the total time step of the diffusion sampling. With parameters θ, the expectation value can be taken over the trajectories of diffusion sampling.

## 4 Method

### 4.1 Parrot Overview

Fig. 2 shows the overview of Parrot, which consists of the prompt expansion network (PEN) pϕ and the T2I diffusion model pθ. The PEN is first initialized from a supervised fine-tuning checkpoint on demonstrations of prompt expansion pairs, and the T2I model is initialized from pretrained diffusion model. Given the original prompt c, the PEN generates an expanded prompt cˆ, and the T2I model generates images based on this expanded prompt. During the multi-reward RL fine-tuning, a batch of N images is sampled, and multiple quality rewards are calculated for each image, encompassing aspects like text-image alignment, aesthetics, human preference, and image sentiment. Based on these reward scores, Parrot identifies the batch-wise Pareto-optimal set using a non-dominated sorting algorithm. This optimal set of images is then used for joint optimization of the PEN and T2I model parameters through RL policy gradient update. During inference, Parrot leverages both the original prompt and its expansion, striking a balance between maintaining faithfulness to the original prompt and incorporating additional details for higher quality.

### 4.2 Batch-wise Pareto-optimal Selection

Li et al. [30] has demonstrated that using batchwise Pareto-set learning and selecting good samples in a batch can approximate Pareto-optimality across multiple objectives. Backed up by this theory, we propose to select non-dominated

###### Training Inference

Original prompt Original prompt

Joint Optimization

[Figure 26]

“Old fashioned cocktail. ”

[Figure 27]

“A boat”

Policy Gradient Update

###### Prompt Expansion Network

Prompt Expansion Network

“A boat in the middle of a lake with a beautiful sunset.”

“A glass of old fashioned cocktail on a wooden table in a dark room”

[Figure 28]

Joint Guidance

###### Batch-wise Pareto Optimal Selection

T2I Model

[Figure 29]

Text-Image Alignment, Aesthetic Score, Human Preference, Image Sentiment

- Reward Scores 1

- Reward Scores 2

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

T2I Model

: Frozen

…

Multi-Reward

[Figure 35]

: Tunable

…

: Forward : Backward

Reward Scores N

SampleN

Sample1 Sample2

Fig. 2: Overview of Parrot. During the training, N images are sampled from the T2I model using the expanded prompt from the prompt expansion network. Multiple quality rewards are calculated for each image, and the Pareto-optimal set is identified using the non-dominated sorting algorithm. These optimal images are then used to perform policy gradient update of the parameters of T2I model and prompt expansion network jointly. During the inference, both the original prompt and the expanded prompt are provided to the T2I model, enabling better faithfulness while adding detail.

points in a batch for policy gradient update in RL to achieve Pareto-optimality. Algorithm 1 outlines the procedure of Parrot. Rather than updating the gradients using all images, Parrot focuses on high-quality samples, considering multiple quality rewards in each mini-batch. In the multi-reward RL, each sample generated by the T2I model presents distinct trade-offs for different rewards. Among these samples, a subset with varied optimal trade-offs across multiple objectives, also known as the Pareto set, exists. For a Pareto-optimal sample, none of its objective values can be further improved without damaging others. In other words, the Pareto-optimal set is not dominated by any data points, also known as the non-dominated set. To achieve a Pareto-optimal solution with textto-image generation diffusion model, Parrot selectively uses data points from the non-dominated set using non-dominated sorting algorithm. This naturally encourages the T2I model to produce Pareto-optimal samples with respect to the multi-reward objectives.

Reward-specific Preference: Inspired by the use of preference information in multi-objective optimization [30], Parrot incorporates the preference information through reward-specific identifiers. This enables Parrot to automatically determine the importance for each reward objective. Concretely, we enrich the expanded prompt cˆ by prepending reward-specific identifier “<reward k>” for k-th reward. Based on this reward-specific prompt, N images are generated and are used for maximizing the corresponding k-th reward model during gradient update. At inference time, a concatenation of all the reward identifiers “<reward 1>,...,<reward K>” is used for image generation.

Non-dominated Sorting: Parrot constructs Pareto set with non-dominated points based on trade-offs among multiple rewards. These non-dominated points are superior to the remaining solutions and are not dominated by each other. Formally, the dominance relationship is defined as follows: the image xa0 domi-

Algorithm Parrot: Pareto-optimal Multi-Reward RL Input: Prompt c, Batch size N, Total iteration E, the number of rewards: K, Prompt expansion network pϕ, T2I diffusion model: pθ, Total diffusion time step T, Non-dominated set: P

for e = 1 to E do Sample text prompt c ∼ p(c) for k = 1 to K do

function NDSet({x10, ..., xN0 }) P ← ∅ for i = 1 to N do

Expand text prompt cˆ ∼ pϕ(ˆc|c) Prepend reward-specific tokens “<reward k>” to cˆ

dominance ← True for j = 1 to N do

Sample a set of images {x10, ..., xN0 } ∼ pθ(x0|cˆ) A set of reward vector R = {R1, ..., RN}

if xj0 dominates xi0 then

dominance ← False

P ← NDSet({x10, ..., xN0 }) ∇Jϕ += −rk(xj0, cˆ) × ∇ log pϕ(ˆc|c)

if dominance is True then

Add i to P return P

Update the gradient pθ from Eq. 4 Update the gradient pϕ

Output: Fine-tuned diffusion model pθ, prompt expansion network pϕ

nates the image xb0, denoted as xb0 < xa0, if and only if Ri(xb0) ≤ Ri(xa0) for all i ∈ 1,2,...,m, and there exists j ∈ 1,2,...,m such that Rj(xb0) < Rj(xa0). For example, given the i-th generated image xi0 in a mini-batch, when no point in the mini-batch dominates xi0, it is referred to as a non-dominated point.

Policy Gradient Update: We assign a reward value of zero to the data points not included in non-dominated sets and only update the gradient of these nondominated data points as follows:

K

N

T

1 n(P)

rk(xi0,ck) × ∇θ log pθ(xit−1|ck,t,xit), (4)

∇Jθ =

t=1

i=1,xi0∈P

k=1

where i indicates the index of images in mini-batches, and P denotes batch-wise a set of non-dominated points. K and T are the total number of reward models and total diffusion time steps, respectively. The same text prompt is used when updating the diffusion model in each batch.

### 4.3 Original Prompt Centered Guidance

While prompt expansion enhances details and often improves generation quality, there is a concern that the added context may dilute the main content of the original input. To mitigate this during the inference, we introduce original prompt-centered guidance. When sampling conditioned on the original prompt, the diffusion model ϵθ typically predicts noises by combining the unconditioned score estimate and the prompt-conditioned estimate. Instead of relying solely on the expanded prompt from PEN, we propose using a linear combination of two guidances for T2I generation: one from the user input and the other from the expanded prompt. The strength of the original prompt is controlled by guidance scales w1 and w2. The noise ϵ¯θ is estimated, derived from Eq. 1, as follows:

ϵ¯θ = w1 · ϵθ(xt,t,c) + (1 − w1 − w2) · ϵθ(xt,t,null) + w2 · ϵθ(xt,t,cˆ), (5) where null denotes a null text.

## 5 Experiments

### 5.1 Experiment Setting

Dataset: The PEN is first supervised fine-tuned on a large-scale text dataset named the Promptist [15], which has 360K constructed prompt pairs for original prompt and prompt expansion demonstration. The original instruction “Rephrase” is included per pair in Promptist. We modify the instruction into “Input: <original prompt>. This is a text input for image generation. Expand prompt for improving image quality. Output: ”. Subsequently, we use the RL tuning prompts (1200K) from Promptist for RL training of the PEN and T2I model.

T2I Model: Our T2I model is based on the JAX version of Stable Diffusion 1.5 [39] pre-trained with the LAION-5B [43] dataset. We conduct experiments on a machine equipped with 16 NVIDIA RTX A100 GPUs. DDIM [46] with 50 denoising steps is used, and the classifier-free guidance weight is set to 5.0 with the resolution 512×512. Instead of updating all layers, we specifically update the cross-attention layer in the Denoising U-Net. For optimization, we employ the Adam [25] optimizer with a learning rate of 1×10−5.

Prompt Expansion Network: For prompt expansion, we use PaLM 2-L-IT [2], one of the PaLM2 variations, which is a multi-layer Transformer [49] decoder with casual language modeling. We optimize LoRA [20] weights for RL-based fine-tuning. The output token length of the PEN is set to 77 to match the maximum number of token length for Stable Diffusion. For original promptcentered guidance, we set both w1 and w2 to 5 in Eq. 5.

Reward Models: We incorporate four quality signals as rewards: Aesthetics, Human preference, Text-Image Alignment, Image Sentiment. For aesthetics, we use the VILA-R [24] pre-trained with the AVA [33] dataset. For human preference, we train a ViT-B/16 [10] using the Pick-a-Pic [26] dataset, which contains 500K examples for human feedback in T2I generation. The ViT-B/16 image encoder consists of 12 transformer layers, and the image resolution is 224×224 with a patch size of 16×16. For text-image alignment, we use CLIP [35] with the image encoder ViT-B/32. For image sentiment, we use the pre-trained model from [44], which outputs three labels: positive, neutral, negative. We use the positive score ranging from 0 to 1 as the sentiment reward.

### 5.2 Qualitative Analysis

Comparison with Baselines: Fig. 3 shows the visual comparison of Parrot and multiple baselines. We include results from Stable Diffusion 1.5, DPOK [12] with a weighted sum of rewards, Promptist [15], and Parrot. DPOK exclusively fine-tunes the T2I model, while Promptist focuses on fine-tuning only the prompt expansion network. Parrot shows visually better images, particularly in aspects like color combination, cropping, perspective, and fine details in the image. This improvement can be attributed to Parrot’s T2I model being fine-tuned together

[Figure 36]

[Figure 39]

[Figure 40]

A turtle in a beautiful garden with flowers and a pond

A turtle, RPG Reference, Oil Painting, Trending on Artstation, octane render, Insanely Detailed, 8k, HD

A turtle

A turtle

SD 1.5 Promptist Parrot

#### DPOK

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

A kitchen with a lot of natural light and a beautiful view of the mountains

A kitchen

A kitchen

A kitchen, a fantasy digital painting, trending on Artstation, highly detailed

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

A F1 F1, concept art oil painting by Jama Jurabaev, extremely detailed, brush hard, artstation

3d rendering of a f1 car in dark and moody environment with a red and orange color scheme.

A F1

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

A woman is looking out on a winter archipelago.

A woman is looking out on a winter archipelago.

Painting of a woman looking out on a winter archipelago, ultra realistic, concept art, intricate details, eerie

A woman is looking out on winter archipelago. The image is very peaceful and serene.

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

A Japanese Porcelain Imari vase, intricate, elegant, highly detailed, digital painting, artstation, concept art, smooth, sharp focus, illustration,

A photo of a Japanese Porcelain Imari vase on the ground with a beautiful floral pattern on the front

A Japanese Porcelain Imari vase

A Japanese Porcelain Imari vase

- Fig. 3: Comparison of Parrot and diffusion-based RL baselines. From left to right, we provide results of Stable diffusion 1.5 [39] (1st column), DPOK [12] (2nd column) with the weighted sum, Promptist [15] (3rd column), and Parrot (4th column).

[Figure 57]

[Figure 58]

An armchair A comfortable armchair in a living room with a fireplace and a view of the mountains.

A comfortable armchair in a living room with a fireplace and a view of the mountains.

An armchair, intricate, elegant, highly detailed, digital painting, artstation, concept art, smooth, sharp focus, illustration

An armchair

[Figure 64]

(a) CLIP Score (b) ViT-PickScore (c) VILA-AVA (d) Image Sentiment

[Figure 65]

[Figure 66]

[Figure 67]

Steps (K) Steps (K) Steps (K) Steps (K)

[Figure 68]

- Fig. 4: Training curve for fine-tuning on weighted sum and Parrot. For weighted sum, WS1 denotes {0.7, 0.1, 0.1, 0.1} and WS2 denotes {0.25, 0.25, 0.25, 0.25} for aesthetics, human preference, text-image alignment and image sentiment. Using weighted sum leads to decrease in human preference score and image sentiment score despite an improvement in the aesthetic score. In contrast, Parrot exhibits stable increases across all metrics.

Quality Metrics TIA (↑) Aesth. (↑) HP (↑) Sent. (↑) Average (↑)

Model

SD 1.5 [39] 0.2322 0.5755 0.1930 0.3010 0.3254 DPOK [12] (WS) 0.2337 0.5813 0.1932 0.3013 0.3273 (+0.58%) Parrot w/o PE 0.2355 0.6034 0.2009 0.3018 0.3354 (+3.07%) Parrot T2I Model Tuning Only 0.2509 0.7073 0.3337 0.3052 0.3992 (+22.6%)

Promptist [15] 0.1449 0.6783 0.2759 0.2518 0.3377 (+3.77 %) Parrot with HP Only 0.1543 0.5961 0.3528 0.2562 0.3398 (+4.42 %) Parrot PEN Tuning Only 0.1659 0.6492 0.2617 0.3131 0.3474 (+6.76 %) Parrot w/o Joint Optimization 0.1661 0.6308 0.2566 0.3084 0.3404 (+4.60 %) Parrot w/o ori prompt guidance 0.1623 0.7156 0.3425 0.3130 0.3833 (+17.8 %) Parrot 0.1667 0.7396 0.3411 0.3132 0.3901 (+19.8 %)

- Table 1: Quantitative comparison between Parrot and alternatives on the Parti dataset [53]. Abbreviations: WS - Weighted Sum; PE - Prompt Expansion; TIA Text-Image Alignment; Aesth. - Aesthetics; HP - Human Preference; Sent. - Image Sentiment. TIA score is measured against the original prompt without expansion.

with the prompt expansion model that incorporates aesthetic keywords during training. Parrot generates results that are more closely aligned with the input prompt, as well as more visually pleasing.

Weighted sum vs. Parrot: Fig. 4 shows the training curve comparison of Parrot and using a linear combination of the reward scores. Each subgraph represents a reward. WS1 and WS2 denote two different weights with multiple reward scores. WS1 places greater emphasis on the aesthetic score, while WS2 adopts balanced weights across aesthetics, human preference, text-image alignment, and image sentiment. Employing the weighted sum of multiple rewards leads to a decrease in the image sentiment score, despite notable enhancements in aesthetics and human preference. In contrast, Parrot consistently exhibits improvement across all metrics.

### 5.3 Quantitative Evaluation

Comparison with Baselines: Table 1 presents our results of the quality score across four quality rewards: text-image alignment score, aesthetic score, human preference score, and emotion score. The first group shows methods without prompt expansion, and the second group compares methods with expansion. The prompt expansion and T2I generation are performed on the PartiPrompts [53]. Using a set of 1632 prompts, we generate 32 images for each text input and calculate the average for each metric.

w/o PE indicates the generation of images solely based on original prompt without expansion. Note that w/o PE is not the same as Parrot T2I Model Tuning Only. The former takes a model trained with PEN and removes it during inference, which results in a notable disparity between prompts used in training and testing. The latter trains with the multi-objective RL using only the T2I model, and it shows substantial improvement upon DPOK [12] (weighted sum) with balanced weights of {0.25,0.25,0.25,0.25}. This shows that the proposed RL method is indeed effective in multi-objective optimization.

For Promptist [15], we generate prompt expansion from their model. Parrot with HP Only shows solely using a human prefernce reward leads to decline in others. Parrot PEN Tuning Only shows suboptimal aesthetics and human preference. Parrot w/o Joint Optimization shows suboptimal results than Parrot which demonstrates the necessity of jointly optimizing PEN and T2I models.

Our method outperforms both compared methods in aesthetics, human preference and sentiment scores. The text-image alignment score is measured with the original prompt before expansion for fair comparison. As a result, the group without prompt expansion generally shows a higher text-image alignment score. Parrot shows better text-image alignment in each subgroup.

Area Question Aesthetics “Which image shows better aesthetics without blurry texture, unnatural

focusing, and poor color combination?” Human Preference “Which generated image do you prefer?” Text-Image Alignment “Which image is well aligned with the text?” Image Sentiment “Which image is closer to amusement, excitement, and contentment?”

- Table 2: Questions for user study. For performing user study, we carefully design questions suitable to each quality.

User Study: We conduct a user study using MTurk [1] with generated images from 100 random prompts in the PartiPrompts [53]. Five models are compared: Stable Diffusion v1.5, DPOK [12] with an equal weighted sum, Promptist [15], Parrot without prompt expansion, and Parrot. Each rater is presented with the original prompt (before expansion) and a set of five generated images, with the image order being randomized. Raters are then tasked with selecting the best image from the group, guided by questions outlined in Table 2. Each question

[Figure 69]

0.54 0.26

0.32

##### 12 Lee, Seung Hyun et al.

0.24

0.32

0.10

(a) Aesthetic score (b) Human preference score

[Figure 70]

[Figure 71]

[Figure 72]

Average

%

[Figure 73]

Image Sentiment

Aesthetics

(c) Text-image alignment score (d) Image Sentiment score

[Figure 74]

[Figure 75]

Text-image Alignment

Human Preference

Fig. 6: Ablation study. We perform an ablation study by removing one of quality signals. We observe that each quality signal affects their improvement of (a) aesthetics, (b) human preference score, (c) text-image alignment score, (d) image sentiment score.

Fig. 5: User study results on PartiPrompts [53]. Parrot outperforms baselines across all metrics.

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

Fig. 7: Comparing Parrot with/without Pareto Optimal. By prepending different reward-specific preferences, Parrot can change the tradeoff between rewards. Specifically, “<reward 1>”, “<reward 2>”, “<reward 3>”, “<reward 4>” indicates aesthetics, human preference, text-image alignment, and image sentiment respectively. With Pareto-optimal selection, each reward increases based on reward-specific preference.

pertains to a specific quality aspect: aesthetics, human preference, text-image alignment, and image sentiment. For each prompt, 20 rounds of random sampling are conducted and sent to different raters. The user study results, illustrated in Fig. 5, show that Parrot outperforms other baselines across all dimensions.

[Figure 80]

[Figure 81]

[Figure 82]

Proportion of non-dominated points: Using batch size of 256, we observe that in a batch around 20% to 30% are non-dominated points and the proportion of non-dominated points in single batch slightly increase as training proceeds.

### 5.4 Ablations

Effect of Pareto-optimal Multi-reward RL: To show the efficacy of Paretooptimal Multi-reward RL, we conduct an ablation study by removing one reward model at a time. Fig. 6 shows quantitative results using one hundred random text

[Figure 83]

[Figure 88]

[Figure 89]

Landscape painting of old medieval cities, surrounded by lush green trees, with a river running beneath the city.

Landscape painting of old medieval cities, surrounded by lush green trees, with a river running beneath the city.

Landscape painting of old medieval cities, surrounded by lush green trees, with a river running beneath the city.

Landscape painting of old medieval cities, surrounded by lush green trees, with a river running beneath the city.

Landscape painting of old medieval cities, surrounded by lush green trees, with a river running beneath the city.

Landscape painting of old medieval cities, surrounded by lush green trees, with a river running beneath the city.

Landscape painting of old medieval cities, surrounded by lush green trees, with a river running beneath the city.

Parrot w/o Pareto Optimal

Stable Diffusion 1.5 Aesthetics Human Preference Text-Image Alignment Image Sentiment Parrot

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

A raccoon wearing formal clothes, wearing a tophat and holding a cane. Oil painting in the style of Rembrandt.

A raccoon wearing formal clothes, wearing a tophat and holding a cane. Oil painting in the style of Rembrandt.

A raccoon wearing formal clothes, wearing a tophat and holding a cane. Oil painting in the style of Rembrandt.

A raccoon wearing formal clothes, wearing a tophat and holding a cane. Oil painting in the style of Rembrandt.

A raccoon wearing formal clothes, wearing a tophat and holding a cane. Oil painting in the style of Rembrandt.

A raccoon wearing formal clothes, wearing a tophat and holding a cane. Oil painting in the style of Rembrandt.

A raccoon wearing formal clothes, wearing a tophat and holding a cane. Oil painting in the style of Rembrandt.

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

A painting of a pandas dressed as a chef, serving a cookie in a cozy kitchen.

A painting of a pandas dressed as a chef, serving a cookie in a cozy kitchen.

A painting of a pandas dressed as a chef, serving a cookie in a cozy kitchen.

A painting of a pandas dressed as a chef, serving a cookie in a cozy kitchen.

A painting of a pandas dressed as a chef, serving a cookie in a cozy kitchen.

A painting of a pandas dressed as a chef, serving a cookie in a cozy kitchen.

A painting of a pandas dressed as a chef, serving a cookie in a cozy kitchen.

Fig. 8: The comparisons of the diffusion fine-tuning between Pareto-optimal multi-reward RL and single reward RL. We show results with same seed from various methods: Stable Diffusion 1.5 [39] (1st column), T2I model fine-tuned with the aesthetic model (2nd column), the human preference model (3rd column), textimage alignment (4th column), image sentiment (5th column), Parrot without Paretooptimal selection (6th column) and Parrot (7th column). Parrot is effective to generate acceptable images without sacrificing one of quality signals. For example, T2I model fine-tuned with a single quality signal such as aesthetics, human preference and image sentiment results in text-image misalignment, while our method achieves a balanced visual outcome across multiple criteria.

prompts from the Promptist [15]. We observe that our training scheme improves multiple target objectives.

To verify whether Parrot achieved better trade-off for different reward scores, we generate 1000 images from common animal dataset [4], where each text prompt consists of the name of a common animal. As shown in Fig. 7, using only text-image alignment with reward-specific preference “<reward 3>” generates images with higher text-image alignment score, while using only aesthetic model with reward-specific preference “<reward 1>” yields images with higher aesthetic score. In the case of using two reward-specific preferences “<reward 1>, <reward 3>”, we observe that scores are balanced and show that better results across multiple rewards than Parrot without Pareto optimal selection.

Fig 8 shows the visual comparison between Parrot, Parrot with a single reward, and Parrot without selecting the batch-wise Pareto-optimal solution. Using a single reward model tends to result in degradation of other rewards, especially text-image alignment. In the third column, results of the first row miss the text a tophat in input prompt, even though the Stable Diffusion result includes that attribute. On the other hand, Parrot results capture all prompts, improving other quality signals, such as aesthetics, image sentiment and human preference.

Effect of Original Prompt Centered Guidance: Fig 9 shows the effect of the proposed original prompt-centered guidance. As evident from the figure, using only the expanded prompt as input often results in the main content be-

ing overwhelmed by the added context. For instance, given the original prompt “A shiba inu", the result from the expanded prompt shows a zoomed-out image and the intended main subject (shiba inu) becomes small. The proposed original prompt-centered guidance effectively addresses this issue, generating an image that faithfully captures the input prompt while incorporating visually more pleasing details.

[Figure 106]

[Figure 107]

A pumpkin A pumpkin with a beautiful autumn background

A pumpkin with a beautiful autumn background

W/o original prompt centered guidance

###### W/o original prompt centered guidance

Stable Diffusion 1.5 Parrot

###### Stable Diffusion 1.5 Parrot

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

Pickup truck under street lights at night with a beautiful cityscape in the background.

Pickup truck under street lights at night with a beautiful cityscape in the background.

Pickup truck under street lights at night

A shiba inu Shiba inu in a field of flowers with a beautiful sunset

Shiba inu in a field of flowers with a beautiful sunset

Fig. 9: Results of original prompt centered guidance. As we expand the prompt, the content in the generated image often fades away. This guidance is helpful for keeping the main content of the original prompt.

## 6 Conclusion and Limitation

We propose Parrot, a novel multi-reward optimization algorithm aimed to improve text-to-image generation by effectively optimizing multiple quality rewards using RL. With batch-wise Pareto-optimal selection, Parrot adaptively balance the optimization of multiple quality rewards. By applying Parrot to jointly finetune both the T2I model and the prompt expansion model, we achieve the generation of higher-quality images with richer details. Additionally, our original prompt centered guidance technique ensures that the generated image maintains fidelity to the user prompt after prompt expansion during inference. Results from the user study indicate that Parrot significantly improves the quality of generated images across multiple criteria, including text-image alignment, human preference, aesthetics, and image sentiment. While Parrot has shown effectiveness in enhancing generated image quality, its efficacy is limited by the quality metrics it relies on. Therefore, advancements of the generated image quality metrics will directly enhance the capabilities of Parrot. Additionally, Parrot is adaptable to a broader range of rewards that quantify generated image quality.

Societal Impact: Parrot could potentially raise ethical concerns related to the generation of immoral content. This concern stems from the user’s ability to influence T2I generation, allowing for the creation of visual content that may be deemed inappropriate. The risk may be tied to the potential biases in reward models inherited from various datasets.

## References

- 1. Amazon mechanical turk. https://www.mturk.com/ (2005)
- 2. Anil, R., Dai, A.M., Firat, O., Johnson, M., Lepikhin, D., Passos, A., Shakeri, S., Taropa, E., Bailey, P., Chen, Z., et al.: Palm 2 technical report. arXiv preprint arXiv:2305.10403 (2023)
- 3. Bai, Y., Jones, A., Ndousse, K., Askell, A., Chen, A., DasSarma, N., Drain, D., Fort, S., Ganguli, D., Henighan, T., et al.: Training a helpful and harmless assistant with reinforcement learning from human feedback. arXiv preprint arXiv:2204.05862

(2022)

- 4. Black, K., Janner, M., Du, Y., Kostrikov, I., Levine, S.: Training diffusion models with reinforcement learning. arXiv preprint arXiv:2305.13301 (2023)
- 5. Chang, H., Zhang, H., Barber, J., Maschinot, A., Lezama, J., Jiang, L., Yang, M.H., Murphy, K., Freeman, W.T., Rubinstein, M., et al.: Muse: Text-to-image generation via masked generative transformers. arXiv preprint arXiv:2301.00704

(2023)

- 6. Clark, K., Vicol, P., Swersky, K., Fleet, D.J.: Directly fine-tuning diffusion models on differentiable rewards. ICLR (2024)
- 7. Dai, X., Hou, J., Ma, C.Y., Tsai, S., Wang, J., Wang, R., Zhang, P., Vandenhende, S., Wang, X., Dubey, A., et al.: Emu: Enhancing image generation models using photogenic needles in a haystack. arXiv preprint arXiv:2309.15807 (2023)
- 8. Deng, F., Wang, Q., Wei, W., Grundmann, M., Hou, T.: Prdp: Proximal reward difference prediction for large-scale reward finetuning of diffusion models. CVPR

(2024)

- 9. Dong, H., Xiong, W., Goyal, D., Pan, R., Diao, S., Zhang, J., Shum, K., Zhang, T.: Raft: Reward ranked finetuning for generative foundation model alignment. arXiv preprint arXiv:2304.06767 (2023)
- 10. Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., et al.: An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929 (2020)
- 11. Fan, Y., Lee, K.: Optimizing ddpm sampling with shortcut fine-tuning. ICML

(2023)

- 12. Fan, Y., Watkins, O., Du, Y., Liu, H., Ryu, M., Boutilier, C., Abbeel, P., Ghavamzadeh, M., Lee, K., Lee, K.: Dpok: Reinforcement learning for fine-tuning text-to-image diffusion models. NeurIPS (2023)
- 13. Fang, Y., Zhu, H., Zeng, Y., Ma, K., Wang, Z.: Perceptual quality assessment of smartphone photography. In: CVPR (2020)
- 14. Han, L., Li, Y., Zhang, H., Milanfar, P., Metaxas, D., Yang, F.: Svdiff: Compact parameter space for diffusion fine-tuning. CVPR (2023)
- 15. Hao, Y., Chi, Z., Dong, L., Wei, F.: Optimizing prompts for text-to-image generation. arXiv preprint arXiv:2212.09611 (2022)
- 16. He, H., Wang, T., Yang, H., Fu, J., Yuan, N.J., Yin, J., Chao, H., Zhang, Q.: Learning profitable nft image diffusions via multiple visual-policy guided reinforcement learning. arXiv preprint arXiv:2306.11731 (2023)
- 17. Ho, J., Jain, A., Abbeel, P.: Denoising diffusion probabilistic models. NeurIPS

(2020)

- 18. Ho, J., Salimans, T.: Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598 (2022)

- 19. Hosu, V., Lin, H., Sziranyi, T., Saupe, D.: Koniq-10k: An ecologically valid database for deep learning of blind image quality assessment. TIP 29, 4041–4056

(2020)

- 20. Hu, E.J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W.: Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685 (2021)
- 21. Jeong, Y., Ryoo, W., Lee, S., Seo, D., Byeon, W., Kim, S., Kim, J.: The power of sound (tpos): Audio reactive video generation with stable diffusion. In: ICCV

(2023)

- 22. Kawar, B., Zada, S., Lang, O., Tov, O., Chang, H., Dekel, T., Mosseri, I., Irani, M.: Imagic: Text-based real image editing with diffusion models. In: CVPR (2023)
- 23. Ke, J., Wang, Q., Wang, Y., Milanfar, P., Yang, F.: Musiq: Multi-scale image quality transformer. In: ICCV (2021)
- 24. Ke, J., Ye, K., Yu, J., Wu, Y., Milanfar, P., Yang, F.: Vila: Learning image aesthetics from user comments with vision-language pretraining. In: CVPR (2023)
- 25. Kingma, D.P., Ba, J.: Adam: A method for stochastic optimization. ICLR (2015)
- 26. Kirstain, Y., Polyak, A., Singer, U., Matiana, S., Penna, J., Levy, O.: Pick-a-pic: An open dataset of user preferences for text-to-image generation. arXiv preprint arXiv:2305.01569 (2023)
- 27. Lee, K., Liu, H., Ryu, M., Watkins, O., Du, Y., Boutilier, C., Abbeel, P., Ghavamzadeh, M., Gu, S.S.: Aligning text-to-image models using human feedback. arXiv preprint arXiv:2302.12192 (2023)
- 28. Lee, S.H., Kim, S., Yoo, I., Yang, F., Cho, D., Kim, Y., Chang, H., Kim, J., Kim, S.: Soundini: Sound-guided diffusion for natural video editing. arXiv preprint arXiv:2304.06818 (2023)
- 29. Li, Y., Liu, H., Wu, Q., Mu, F., Yang, J., Gao, J., Li, C., Lee, Y.J.: Gligen: Open-set grounded text-to-image generation. In: CVPR (2023)
- 30. Lin, X., Yang, Z., Zhang, X., Zhang, Q.: Pareto set learning for expensive multiobjective optimization. NeurIPS (2022)
- 31. Mannor, S., Shimkin, N.: The steering approach for multi-criteria reinforcement learning. NeurIPS (2001)
- 32. Miettinen, K.: Nonlinear multiobjective optimization, vol. 12. Springer Science & Business Media (1999)
- 33. Murray, N., Marchesotti, L., Perronnin, F.: Ava: A large-scale database for aesthetic visual analysis. In: CVPR (2012)
- 34. Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., et al.: Training language models to follow instructions with human feedback. NeurIPS (2022)
- 35. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from natural language supervision. In: ICML (2021)
- 36. Rame, A., Couairon, G., Shukor, M., Dancette, C., Gaya, J.B., Soulier, L., Cord, M.: Rewarded soups: towards pareto-optimal alignment by interpolating weights fine-tuned on diverse rewards. NeurIPS (2023)
- 37. Ramesh, A., Dhariwal, P., Nichol, A., Chu, C., Chen, M.: Hierarchical textconditional image generation with clip latents. arXiv preprint arXiv:2204.06125

(2022)

- 38. Richardson, E., Goldberg, K., Alaluf, Y., Cohen-Or, D.: Conceptlab: Creative generation using diffusion prior constraints. arXiv preprint arXiv:2308.02669 (2023)
- 39. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image synthesis with latent diffusion models. In: CVPR (2022)

- 40. Ronneberger, O., Fischer, P., Brox, T.: U-net: Convolutional networks for biomedical image segmentation. In: MICCAI (2015)
- 41. Saharia, C., Chan, W., Chang, H., Lee, C., Ho, J., Salimans, T., Fleet, D., Norouzi, M.: Palette: Image-to-image diffusion models. In: SIGGRAPH (2022)
- 42. Saharia, C., Chan, W., Saxena, S., Li, L., Whang, J., Denton, E.L., Ghasemipour, K., Gontijo Lopes, R., Karagol Ayan, B., Salimans, T., et al.: Photorealistic textto-image diffusion models with deep language understanding. NeurIPS (2022)
- 43. Schuhmann, C., Beaumont, R., Vencu, R., Gordon, C., Wightman, R., Cherti, M., Coombes, T., Katta, A., Mullis, C., Wortsman, M., et al.: Laion-5b: An open large-scale dataset for training next generation image-text models. NeurIPS (2022)
- 44. Serra, A., Carrara, F., Tesconi, M., Falchi, F.: The emotions of the crowd: Learning image sentiment from tweets via cross-modal distillation. ECAI (2023)
- 45. Sohl-Dickstein, J., Weiss, E., Maheswaranathan, N., Ganguli, S.: Deep unsupervised learning using nonequilibrium thermodynamics. In: ICML (2015)
- 46. Song, J., Meng, C., Ermon, S.: Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502 (2020)
- 47. Tesauro, G., Das, R., Chan, H., Kephart, J., Levine, D., Rawson, F., Lefurgy, C.: Managing power consumption and performance of computing systems using reinforcement learning. NeurIPS (2007)
- 48. Tu, Z., Talebi, H., Zhang, H., Yang, F., Milanfar, P., Bovik, A., Li, Y.: Maxvit: Multi-axis vision transformer. In: ECCV (2022)
- 49. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, Ł., Polosukhin, I.: Attention is all you need. NeurIPS (2017)
- 50. Wu, X., Sun, K., Zhu, F., Zhao, R., Li, H.: Human preference score: Better aligning text-to-image models with human preference. In: ICCV (2023)
- 51. Xu, J., Liu, X., Wu, Y., Tong, Y., Li, Q., Ding, M., Tang, J., Dong, Y.: Imagereward: Learning and evaluating human preferences for text-to-image generation. arXiv preprint arXiv:2304.05977 (2023)
- 52. Ying, Z., Niu, H., Gupta, P., Mahajan, D., Ghadiyaram, D., Bovik, A.: From patches to pictures (paq-2-piq): Mapping the perceptual space of picture quality. In: CVPR (2020)
- 53. Yu, J., Xu, Y., Koh, J.Y., Luong, T., Baid, G., Wang, Z., Vasudevan, V., Ku, A., Yang, Y., Ayan, B.K., et al.: Scaling autoregressive models for content-rich text-to-image generation. arXiv preprint arXiv:2206.10789 (2022)
- 54. Yu, L., Cheng, Y., Sohn, K., Lezama, J., Zhang, H., Chang, H., Hauptmann, A.G., Yang, M.H., Hao, Y., Essa, I., et al.: Magvit: Masked generative video transformer. In: CVPR (2023)
- 55. Zhou, Y., Liu, B., Zhu, Y., Yang, X., Chen, C., Xu, J.: Shifted diffusion for textto-image generation. In: CVPR (2023)

## Parrot: Pareto-optimal Multi-Reward Reinforcement Learning Framework for Text-to-Image Generation Supplementary Material

This supplementary material provides:

- – Sec. A: implementation details, including the training details, and details of quantitative experiments.
- – Sec. B: more ablation studies on original prompt guidance and training scheme of Parrot.
- – Sec. C: more visual examples to show the advancements of Parrot.

## A. Implementation Details

Training Details. We conduct our experiments with Jax implementation of Stable Diffusion 1.5 [39]. In terms of diffusion-based RL, we sample 256 images per RL-tuning iteration. For policy gradient updates, we accumulate gradients across all denoising timesteps. Our experiments employ a small range of gradient clip 10−4. We keep negative prompt as null text.

Details of Quantitative Experiments. From Parti [53] prompts, we generate images of dimensions 512 × 512. In all experiments in the main paper, we apply reward-specific preference expressed as “<reward 1>, <reward 2>, <reward 3>, <reward 4>”, which is optional to select one or several rewards. Reward models are aesthetics, human preference, text-image alignment and image sentiment. During the inference stage, the guidance scale is also set as 5.0 for the diffusion sampling process. We employ the AdamW [25] as optimizer with β1 = 0.9, β2 = 0.999 and a weight decay of 0.1.

## B. Prompt Guidance and Training Scheme

Original Prompt Guidance. In Fig. 10, we provide additional ablation study of original prompt centered guidance by adjusting the guidance scale w1 and w2, which determines visual changes from original prompt and expanded prompt, respectively. We assigned 5 to both w1 and w2 in most of our experiments, which shows better text-image alignment performance in Fig. 10.

Ablation Study on Training Scheme. Fig. 11 provides ablation study comparing variation of the Parrot: Stable Diffusion, the prompt expansion network (PEN) tuning only, T2I model fine-tuning only, Parrot without joint optimization, and Parrot. In the second column, without fine-tuning the T2I diffusion model does not lead to significant improvements in terms of texture and composition. Furthermore, the third column demonstrates that fine-tuning the

diffusion model enhances texture and perspective, yet this improvement is hindered by the limited information in the text prompt. We also observe that the quality of images from joint optimization surpasses that of combining decoupled generative models.

## C. More Visual Examples

We show additional visual examples of Parrot in Figs. 12 to 17. Note that generated images from Parrot are improved across multiple-criteria. Fig. 12 highlights examples where the Parrot brings improvements in aesthetics. For example, Parrot effectively addresses issues such as poor cropping in the fourth column and improves color in the fifth column. Fig. 13 presents examples of images with improved human preference score generated by Parrot. In Fig. 14, we provide examples of improved text-image alignment achieved by Parrot. Fig. 15 shows examples where Parrot enhances image sentiment, producing emotionally rich images.

Finally, additional comparison results between diffusion-based RL baselines are described in Fig. 16, and Fig. 17. Diffusion-based RL baselines are listed: Stable Diffusion 1.5 [39], DPOK [12] with weighted sum of multiple reward scores, Promptist [11], Parrot without prompt expansion, and Parrot. For Parrot without prompt expansion, we only take original prompt as input.

Prompt Expansion

“Oil painting of a deer” →”Oil painting of a deer in a forest. the deer is standing in the middle of the painting, and the forest is behind it. the deer is brown, and the forest is green. the painting is done in a realistic style, and the texture is smooth. the atmosphere of the painting is peaceful and serene.”

“A teapot” →”a teapot with a textured surface and a warm atmosphere.”

“A parrot” →”A parrot with a green and blue body and a yellow beak. The parrot is sitting on a branch. The background is a blue sky with white clouds.”

“A clock” →”A clock with a wooden frame and a white face. The clock is sitting in a room with a white wall. The room is lit by a single light bulb”

“A robot cooking” →”A robot is cooking in a kitchen. The robot has a chef's hat on. The kitchen is clean and well-organized. The robot is cooking a meal for a family of four. The meal is a roast chicken, potatoes, and green beans. The robot is carefully preparing the meal and takes pride in its work.”

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

[Figure 127]

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

[Figure 140]

[Figure 141]

Kitchen painting, art and cuisine cookware review

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

##### Fig. 10: Original prompt centered guidance. We present visual comparison of 5 different pairs of w1 and w2 to demonstrate the effectiveness of guidance scales. For all experiments, we assign w1 = 5 and w2 = 5 (3rd row) for the best performance.

A cocktail

A cat reading a book

→a cat reading a book in a beautiful garden.

“A cocktail” →”A cocktail with a lime wedge and a cherry on top. The cocktail is in a glass with ice cubes and is surrounded by a light blue background. The cocktail is made with a clear liquid, such as vodka or gin, and is topped with a lime wedge and a cherry. The cocktail is surrounded by a light blue background, which creates a relaxing and inviting atmosphere.”

Only Pansion Only Diffusion Tuning Decoupled

Parrot w/o PEN Tuning Only Joint Optimization Parrot

T2I Model Tuning Only

SD 1.5

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

A tree A tree with a beautiful sky and clouds

A tree A tree with a beautiful sky

A tree with a beautiful sky and clouds

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

A crown A crown of flowers on a white background. The flowers are arranged in a circle and the crown is topped with a red rose. The background is a light blue color and the flowers are a variety of colors. The image is both beautiful and elegant.

3d render of a golden crown with a red jewel on a white background.

A crown 3d render of a golden crown with a red jewel on a white background.

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

The city of London The city of London at night with a beautiful view of the city lights.

The city of London The city of London is a beautiful city with a rich history. The city is also known for its vibrant nightlife.

The city of London at night with a beautiful view of the city lights.

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

A cartoon house with red roof 3d render of a cartoon house with a red roof and a blue sky

A cartoon house with red roof A cartoon house with a red roof and a blue sky

3d render of a cartoon house with a red roof and a blue sky

##### Fig. 11: Visual examples of Parrot under different settings. From left to right, we provide results of Stable Diffusion 1.5 [39], the only fine-tuned PEN, the only finetuned T2I diffusion model, the Parrot without joint optimization, and the Parrot.

Prompt Expansion

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

Prompt Expansion

[Figure 174]

Kitchen painting, art and cuisine cookware review

[Figure 175]

[Figure 176]

[Figure 177]

Kitchen painting, art and cuisine cookware review

“An F1 race car in a Manhattan

“A fantasy book style portrait of street” a giant dragon”

“A glowing magical portal inside a big wave made of sand fantasy desert”

“A cocktail”

“Suddenly there is a peach blossom forest”

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

StableDiffusion1.5Parrot

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

A cocktail

↑

Aesthetics

Fig. 12: More Examples of aesthetics improvement from the Parrot. Given the text prompt, we generate images with Stable Diffusion and Parrot. After fine-tuning, the Parrot alleviates quality issues such as poor composition (e.g. bad cropping), misalignment with the user input (e.g. missing objects), or generally less aesthetic pleasing.

A cat reading a book

→a cat reading a book in a beautiful garden.

“Sculptures of two lovers walking through paris”

“Legendary tower of light and “A pick-up truck” magic”

“A cat reading a book”

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

StableDiffusion1.5Parrot

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

↑

Human Preference

##### Fig. 13: More Examples of human preference improvement from the Parrot. Given the text prompt, we generate images with Stable Diffusion 1.5 [39] and Parrot.

[Figure 198]

[Figure 199]

“A small house with trees in the background”

[Figure 200]

[Figure 201]

A cocktail

A cat reading a book

→a cat reading a book in a beautiful garden.

Prompt Expansion

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

Prompt Expansion

[Figure 206]

Kitchen painting, art and cuisine cookware review

[Figure 207]

[Figure 208]

[Figure 209]

Kitchen painting, art and cuisine cookware review

“A room with two chairs and a painting of the Statue of Liberty” “A heart made of wood” “Inside of orange”

“A corgi’s head depicted as an explosion of a nebula”

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

StableDiffusion1.5Parrot

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

↑

Text-Image Alignment

Fig. 14: More examples of text-image alignment improvement from the Parrot. Given the text prompt, we generate images with the Stable Diffusion 1.5 [39] and the Parrot.

“A tiny cute cyberpunk monster metallic bodysuit big eyes smiling waving.”

“Cute little anthropomorphic furry dog.”

“Cute cat sticker” “A pig face”

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

StableDiffusion1.5Parrot

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

↑

Image Sentiment

##### Fig. 15: More examples of image sentiment improvement from the Parrot. Given the text prompt, we generate images with the Stable Diffusion 1.5 [39] and the Parrot.

“A pineapple surfing on a wave”

[Figure 226]

[Figure 227]

[Figure 228]

A cocktail

A cat reading a book

→a cat reading a book in a beautiful garden.

“Primitive drawing of a heart with a smiley face in the middle”

[Figure 231]

[Figure 232]

A cocktail

A cat reading a book

→a cat reading a book in a beautiful garden.

###### SD 1.5 Promptist Parrot w/o PE

###### DPOK Parrot

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

Abstract geometric cityscape

Abstract geometric cityscape

Abstract geometric cityscape

Abstract geometric cityscape with a beautiful sunset.

Abstract geometric cityscape, hyperdetailed, artstation, cgsociety, 8k

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

A painting of the city of nuremberg by john blanche

A painting of the city of nuremberg by john blanche

A painting of the city of nuremberg by john blanche, trending on artstation

A painting of the city of nuremberg by john blanche

A painting of the city of nuremberg by john blanche in a realistic style with a lot of detail and a warm color palette

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

A mix of paris and favelas. A mix of paris and favelas, A mix of paris and favelas. highly detailed, digital painting, artstation, concept art, sharp focus, illustration, art by greg rutkowski and alphonse mucha

A mix of paris and favelas. A mix of paris and favelas. a beautiful city with a lot of colors.

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

A large magical beast called an owlbear

A large magical beast called an owlbear

A large magical beast called an owlbear, highly detailed, digital painting, artstation, concept art, sharp focus, illustration, art by greg rutkowski and alphonse mucha

A large magical beast called an owlbear

A large magical beast called an owlbear in the sunset

###### SD 1.5 Promptist Parrot w/o PE

###### DPOK Parrot

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

A dungeon labyrinth A dungeon labyrinth, RPG A dungeon labyrinth

A dungeon labyrinth

A dungeon labyrinth with a dark and mysterious atmosphere

Reference, Oil Painting, Trending on Artstation, octane render, Insanely Detailed, 8k, HD

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

A red carpet floor A red carpet floor

A red carpet floor, a fantasy A red carpet floor digital painting by Greg Rutkowski and James Gurney, trending on Artstation, highly detailed

A red carpet floor with a beautiful floral pattern and a chandelier hanging from the ceiling.

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

Studio photo of a futuristic cute robot

Studio photo of a futuristic cute robot

Studio photo of a futuristic cute robot

Studio photo of a futuristic cute robot

Studio photo of a futuristic cute robot with a blue and white light.

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

A tree house in the jungle, highlyA tree house in the jungle detailed, digital painting, artstation, concept art, sharp focus, illustration, art by greg rutkowski and alphonse mucha

A tree house in the jungle A tree house in the jungle

A tree house in the jungle with a beautiful view of the mountains

