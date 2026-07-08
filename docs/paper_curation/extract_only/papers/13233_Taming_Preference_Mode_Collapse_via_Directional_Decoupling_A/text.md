### Taming Preference Mode Collapse via Directional Decoupling Alignment in Diffusion Reinforcement Learning

Chubin Chen1,∗,§ Sujie Hu1,∗ Jiashu Zhu2,∗ Meiqi Wu2 Jintao Chen2 Yanxun Li2 Nisha Huang1 Chengyu Fang1 Jiahong Wu2,‡ Xiangxiang Chu2 Xiu Li1,†

1Tsinghua University 2AMAP, Alibaba Group

∗Equal contribution †Corresponding author ‡Project lead §Work done during the internship at AMAP

# arXiv:2512.24146v2[cs.CV]10Mar2026

###### ID Collapse ID Diversity Preserved

###### D²-Align Breaks the Trade-off

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

Higher Preference

[Figure 5]

Eastern woman , with arched eyebrows

[Figure 6]

middle-aged, with a receding hairline

Similar ID w/ different prompt

Diverse ID w/ different prompt

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

Higher Diversity

middle-aged White woman, big nose

young Middle Eastern woman oval face

DanceGRPO

Ours

Style Collapse Style Diversity Preserved Layout Collapse Layout Diversity Preserved Tonal Collapse Tonal Diversity Preserved

[Figure 11]

[Figure 12]

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

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

Diverse Style w/ different prompt

Diverse Tone w/ different prompt

Homogeneous Style w/ different prompt

Diverse Layout w/ different prompt

Homogeneous Tone w/ different prompt

Homogeneous Layout w/ different prompt

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

Flow-GRPO Ours Flow-GRPO Ours

DanceGRPO Ours

Figure 1. D2-Align breaks the trade-off between human preference and generative diversity, mitigating Preference Mode Collapse (PMC). The top-right plot shows that while baselines struggle with a trade-off—either achieving low diversity or low preference—D2Align, achieves a state of both higher diversity and higher human preference. The qualitative examples below illustrate this phenomenon. For the same set of varied prompts, baseline methods exhibit severe PMC, generating homogeneous outputs for identity, style, layout, and tone. D2-Align successfully preserves diversity, generating distinct and high-quality images that align with each individual prompt. See Supp. for detail prompts.

###### Abstract

exposure), severely degrading generative diversity. In this work, we introduce and quantify this phenomenon, proposing DivGenBench, a novel benchmark designed to measure the extent of PMC. We posit that this collapse is driven by over-optimization along the reward model’s inherent biases. Building on this analysis, we propose Directional Decoupling Alignment (D2-Align), a novel framework that mitigates PMC by directionally correcting the reward signal. Specifically, our method first learns a directional correction within the reward model’s embedding space while

Recent studies have demonstrated significant progress in aligning text-to-image diffusion models with human preference via Reinforcement Learning from Human Feedback. However, while existing methods achieve high scores on automated reward metrics, they often lead to Preference Mode Collapse (PMC)—a specific form of reward hacking where models converge on narrow, high-scoring outputs (e.g., images with monolithic styles or pervasive over-

keeping the model frozen. This correction is then applied to the reward signal during the optimization process, preventing the model from collapsing into specific modes and thereby maintaining diversity. Our comprehensive evaluation, combining qualitative analysis with quantitative metrics for both quality and diversity, reveals that D2-Align achieves superior alignment with human preference.

###### 1. Introduction

Recent advances in generative models have enabled the generation of high-fidelity and creative visual content [4, 7, 12, 17, 25, 27, 32, 38, 41, 43, 55, 58, 61, 64, 69]. To better align model outputs with human preference, a pivotal approach has been the adoption of Reinforcement Learning from Human Feedback (RLHF) [31, 63]. A successful alignment process must consider not only image quality and fidelity [3, 46], but also generative diversity [44, 66]. Key motivations for ensuring generative diversity include its foundational role in creative content generation [47], data augmentation [18], and the enhancement of downstream task performance [5]. In practice, training T2I models to maximize a predefined reward often leads to reward hacking, where models learn to achieve high scores even for low-quality outputs [39, 60]. While existing methods have made progress in mitigating this outright quality degradation [10, 39, 60], they tend to steer the model toward a narrow high-score template [8, 48]. As shown in Fig. 1, this convergence produces highly homogeneous images characterized by a monolithic style, recurring visual features, or pervasive overexposure, severely degrading generative diversity. We term this phenomenon Preference Mode Collapse (PMC). This issue stems from two primary challenges: (1) Current approaches prioritize image quality, largely overlooking the crucial aspect of output diversity. Furthermore, there is a lack of standardized quantitative metrics for its evaluation; and (2) the methods used to counteract reward hacking are often empirical and hyperparameter-sensitive, serving only to temper its magnitude rather than fundamentally altering the model’s optimization direction.

Regarding the first challenge, existing methods for tackling reward hacking predominantly focus on image quality [39, 60, 62]. While they strive to maintain quality as the reward increases, the optimization process inadvertently drives the model into what we term PMC. This problem is compounded because, unlike fidelity [29, 60], generative diversity is inherently difficult to quantify. To address this, we re-evaluate the reward hacking problem through the dual lenses of quality and diversity. Furthermore, we propose DivGenBench, a novel benchmark designed to quantitatively measure the extent of this collapse, thereby offering a standardized means to assess generative diversity.

Regarding the second challenge, existing approaches to mitigate reward hacking often tackle the problem from distinct perspectives. For instance, Flow-GRPO [39] employs KL divergence to prevent over-optimization, but this method requires extensive manual tuning of its coefficient and often leads to a time-consuming training process. Another approach, DanceGRPO [62], involves ensembling multiple reward models; however, its sensitivity to the ensemble weights and thresholds can result in inconsistent performance and training instability. Despite these different strategies, a shared limitation is that they primarily modulate the reward’s magnitude, leaving the problematic optimization direction unaddressed. We hypothesize that the inherent biases of reward models are a primary cause of this phenomenon. To address this, we propose Directional Decoupling Alignment (D2-Align), a novel optimization framework that directionally corrects the reward signal to mitigate PMC. Specifically, our method first learns a directional correction within the reward model’s embedding space while keeping the generator frozen. During the subsequent optimization phase, applying the learned directional vector to correct the reward signal prevents the model from over-optimizing into specific modes, which in turn mitigates the collapse in diversity and guides the model toward a more genuine alignment with human preference.

In summary, our main contributions are as follows:

- • We introduce and quantify Preference Mode Collapse (PMC)—a specific form of reward hacking where existing methods cause a severe loss of diversity when aligning with human preference. To this end, we propose DivGenBench, a novel benchmark designed to serve as the primary tool for its measurement.
- • We propose Directional Decoupling Alignment (D2Align), a novel framework designed to mitigate PMC by directionally correcting the reward signal, which maintains high quality and preserves diversity throughout optimization to achieve a more genuine alignment with human preference.
- • Extensive quantitative and qualitative experiments, further corroborated by human evaluations, demonstrate that our method significantly outperforms existing approaches in terms of both generation quality and diversity.

###### 2. Related Work

###### 2.1. RL for Image Generation

The application of reinforcement learning (RL) to T2I generation has seen significant advancements. Early works in this area can be broadly categorized into two primary directions. The first [2, 15, 16, 21, 44, 67] integrated RL into diffusion models by optimizing the score function through policy gradient methods. An alternative approach [9, 31, 45, 60] focuses on direct fine-tuning with

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

##### 𝑋 𝑋

##### G

Addnoise 𝑋 𝑋

One-step denoise

[Figure 45]

Other SOTAs

Denoise

Ours

||Prompt|
|---|
| |
|---|---|
| | |

##### 𝑒

𝑏

RM

Text Encoder

𝑒̃

|min(ℒ (𝜃))|
|---|

[Figure 46]

𝑒

|min(ℒ (𝑏 ))|
|---|

Base Model

𝑒

Figure 2. Overview of D2-Align. (Left) Our framework first learns a directional vector (bv) to correct the reward signal while keeping the generator frozen (Stage 1). It then uses this learned direction to guide the optimization of the generator, steering it away from mode collapse (Stage 2). (Right) This visualization shows that while other methods converge to a narrow peak (low diversity), D2-Align finds a superior optimum that balances both quality and generative diversity.

###### 3. Preliminary

differentiable rewards. A key advantage of these methods is their use of analytical gradients rather than policy gradients, which allows for more efficient alignment. More recently, methods inspired by the principles of GRPO [6, 22, 33, 34, 36, 39, 56, 62] have elevated the performance to a new state-of-the-art, achieving substantially higher scores on given reward metrics. However, this success is accompanied by a significant challenge: reward hacking, where models over-optimize for superficial reward cues, producing outputs that misalign with true human preference.

###### 3.1. Diffusion and Flow Models

Diffusion models [12, 25, 46] learn to reverse a predefined forward noising process, transforming a simple prior distribution into a complex data distribution pdata. The forward process gradually adds noise to a clean sample x0 ∼ pdata according to:

xt = αtx0 + σtϵ, where ϵ ∼ N(0,I). (1)

###### 2.2. Human Preference Alignment

The functions αt and σt define a noise schedule that maps a clean sample x0 at t = 0 to a noisy sample x1 approximating the prior at t = 1. To learn the reverse process, Flow Matching [38, 41] trains a time-dependent vector field vθ(xt,t) by minimizing a loss function that matches it to the target velocity field of the forward process:

The inherent discrepancy between reward signals and true human preference poses a significant challenge, often leading to a degradation in both generation quality and diversity. From a quality perspective, DanceGRPO [62] observed that combining rewards like HPS-v2.1 [59] and CLIP [23] can mitigate their respective failure modes. Pref-GRPO [56] attributes this issue to the model’s exploitation of subtle imperfections within the reward function, while MixGRPO [33] alleviates it through hybrid inference at test time. From a diversity standpoint, while Flow-GRPO’s [39] proposed KL-regularization term can counteract the decline in sample diversity, its practical application is hampered by several limitations. The method not only relies on a coefficient that must be manually tuned via trial and error, but also introduces significant training overhead and is highly sensitive to this choice. Furthermore, a quantitative evaluation of diversity is notably absent from their framework. Crucially, prior works tend to address quality and diversity in isolation and do not provide a quantitative evaluation of diversity in the final, optimized model. In this work, we introduce a solution that holistically addresses both challenges.

0,ϵ w(t)∥vθ(αtx0 + σtϵ,t) − (α˙tx0 + σ˙tϵ)∥22 ,

Et,x

(2) where w(t) is a weighting function and f˙t denotes the time derivative dft/dt. Once trained, the learned field vθ governs the generative process via a probability flow Ordinary Differential Equation (ODE) [52]:

dxt dt

= vθ(xt,t). (3)

New samples are generated by numerically integrating this ODE backward in time from t = 1 to t = 0, starting from a random sample x1 drawn from the prior.

###### 3.2. One-Step Denoising for Reward Evaluation

A core challenge in aligning diffusion models [25] is that optimization occurs on noisy latents xt, whereas reward

[Figure 47]

models require a clean image x0 for evaluation. A common approach [10, 60] is to perform one-step denoising to predict an image xˆ0 from xt, but this prediction is often inaccurate for early, high-noise timesteps (large t).

[Figure 48]

The style is wrong !

[Figure 49]

###### Low Score

[Figure 50]

|Closer to human preference|
|---|

Exactly my taste !

To overcome this instability, we adopt the ground-truth noise prior technique [50, 68]. We begin with a clean image x0 and a known noise sample ϵgt to create a specific noisy state xt = αtx0 + σtϵgt. The model then predicts the noise ϵθ(xt,t), which allows us to reconstruct a high-fidelity, differentiable estimate of the clean image for reward evaluation:

Reward Model

High Score

###### +"Realistic"

[Figure 51]

That’s not it.

"A painting in the style of Minimalism, depicting a baby penguin"

Low Score

Figure 3. Correcting the Reward Signal via Prompt Perturbation. An image generated for a minimalism prompt is instead oily and overly-rendered. This style mismatch is identified by a human (low score), but the reward model assigns a high score due to its intrinsic bias. We counteract this by perturbing the prompt with descriptors like ”Realistic” to produce a more accurate reward signal aligned with human preference.

xt − σtϵθ(xt,t) αt

. (4)

xˆ0 =

This reliable reconstruction of xˆ0 provides a stable reward signal across all timesteps, enabling us to sample t uniformly from [0,1] for robust training.

###### 4. Methodology

by the reward model and lose diversity. Therefore, mitigating these inherent model biases to obtain a more faithful reward signal is a viable strategy to address this collapse.

Aligning T2I models with human preference aims to produce aesthetically appealing and diverse visual content. RL algorithms accomplish this by optimizing the model to maximize a predefined reward score. However, this process is prone to reward hacking, making its mitigation a critical challenge. In this section, we first analyze the challenge of reward hacking, identifying that the neglect of diversity during alignment leads to Preference Mode Collapse. Subsequently, to address this phenomenon, we propose Directional Decoupling Alignment, our novel framework. Finally, we introduce DivGenBench, a benchmark specifically for evaluating diversity.

To effectively mitigate PMC, it must first be quantitatively measured. We thus introduce DivGenBench (Sec. 4.3), a new benchmark to quantify generative diversity. Building on this, we propose D2-Align, a novel framework designed to mitigate the reward model’s intrinsic biases and prevent this collapse.

###### 4.2. Directional Decoupling Alignment

In the standard alignment process, a sampled image and its corresponding prompt are fed into a reward model, which outputs a feedback signal for gradient updates. However, as noted in [39, 50, 62], reward models such as HPS-v2.1 [59] can cause the T2I model to generate overly smooth and glossy images. This phenomenon leads to inflated reward scores, which genuine human evaluators would not rate as highly. As illustrated in Fig. 3, we observe that this bias can be counteracted by directionally perturbing the prompt. For instance, when we append the descriptor ”realistic” to the prompt of a glossy image, the corresponding reward is suppressed, better aligning the feedback with human aesthetic judgment. This demonstrates the efficacy of a corrected reward signal. While perturbing prompts with manually selected words can yield a bias-suppressed reward, this approach is inherently limited, as the vocabulary space is discrete and pre-defining such words is inefficient. This limitation motivates our strategy to identify a direction in the continuous prompt embedding space that can systematically adjust the reward signal. By operating within the continuous embedding space of the text encoder, we can apply a more principled correction to mitigate the biases of the reward model.

###### 4.1. Preference Mode Collapse

While reinforcement learning successfully aligns T2I models with human preference, it is prone to reward hacking. Existing methods mitigate this by preventing quality degradation, but overlook a critical side effect: a sharp decline in generative diversity. As illustrated in Fig. 1, this creates a problematic trade-off where models produce homogeneous outputs despite high reward scores. We identify this phenomenon—reward hacking from a diversity perspective—as Preference Mode Collapse (PMC).

The reasons for this oversight are twofold. First, evaluating diversity is considerably more challenging than quality. Unlike quality, evaluating diversity in T2I alignment remains an under-explored challenge. Defining standardized metrics is difficult, and existing ones are often too computationally expensive to be used as direct reward signals, thus hindering explicit regularization. Second, and more fundamentally, any reward model possesses intrinsic preferences. The optimization process naturally drives the T2I model to overfit these preferences, inevitably causing the generative distribution to ”collapse” towards a monolithic style favored

As illustrated in Fig. 2, D2-Align is a two-stage framework that addresses the challenge by decoupling the process

## … …

[Figure 52]

###### Benchmark Prompts Div. ID Style Layout Tonal

###### Four-Dimensional Templates

[Figure 53]

HPDv2 3200 ❌ — — — Geneval 553 ❌ — — — —

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

###### ID A high-quality[ethnicities] [gender]portraitwithphoto[feature]of an [age].

[Figure 58]

[Figure 59]

Style

## …

## …

DoFaiR 756 ✅ ✓ ✗ ✗ ✗ DIVBENCH 60 ✅ ✓ ✗ ✗ ✗

[Figure 60]

ID

[Figure 61]

[Figure 62]

style A paintingdepicting [basein theprompt].style of [style],

[Figure 63]

[Figure 64]

DivGenBench (Ours)

[Figure 65]

[Figure 66]

3200 ✅ ✓ ✓ ✓ ✓

[Figure 67]

[Figure 68]

Layout Tonal

[Figure 69]

A studio shot of [number] [object] on a clean white background.

[Figure 70]

[Figure 71]

(d) Comparison

Layout

[Figure 72]

Counting

[Figure 73]

ASC

[Figure 74]

[Figure 75]

[Figure 76]

- (b) Image Generation

T2I Model

[Figure 77]

- (c) Metric Evaluation

[Figure 78]

Tonal Anwithimage[tonal]ofproperties[base prompt], rendered

Layout

Style

Object

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

###### Four-Dimensional Metrics

DivGen Bench

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

3 Age Periods 6 Races 2 Genders 40 Features 300 prompts

SDI

1-IDS

27 Artistic Movements 300 prompts

4 Counts 80 Common Objects 300 prompts

IDS ASC SDI PVS

3 Visual Axes 18 Levels 300 prompts

Age

Brightness

Tonal

[Figure 87]

ID

Ethnicity

0.251

0.253

0.636

0.412

High Diversity

Contrast

<

>

>

>

[Figure 88]

Gender Features

Saturation

Low Diversity

0.348

0.130

0.389

0.168

[Figure 89]

[Figure 90]

PVS

(a) Prompt Crafting

(e) Statistic

Figure 4. Overview Construction and Evaluation Pipeline of Our DivGenBench. (a) A systematic process for prompt construction. (b) Generation of images across four distinct dimensions based on different prompts. (c) Quantitative evaluation using our four proposed metrics: Identity Divergence Score (IDS), Artistic Style Coverage (ASC), Spatial Dispersion Index (SDI), and Photographic Variance Score (PVS). (d) A comparative analysis positioning our benchmark against existing ones. (e) Detailed benchmark statistics and performance comparison of state-of-the-art methods.

[Figure 91]

into two distinct steps: reward signal correction and guided alignment. In Stage 1, we empirically identify a direction within the continuous embedding space to create a corrected reward signal. In Stage2, we leverage this learned direction to guide the generator’s gradient updates, preventing it from over-optimizing into specific modes.

DanceGRPO

D2-Align

Flow-GRPO

SRPO

Same score & Less steps

More Efficient

Stage 1. Following [33, 56, 60, 62, 68], we consider HPS-v2.1 [59] as our reward model, which is a CLIP-based model finetuned on human preference data. The original reward score R(x0,c) is computed by a scoring function:

Same steps & Higher score

More Effective

###### R(x0,c) = score(Φimg(x0),Φtext(c)) (5)

where the score function computes the cosine similarity between the input embeddings, and Φimg and Φtext are the image and text encoders of the model, respectively. We freeze the generator Gθ and introduce a learnable vector bv ∈ Rd. Inspired by [24, 35], we use this vector to construct a guided reward signal. First, we define two perturbed embeddings, e+ and e−, from the original text embedding etext = Φtext(c):

Figure 5. Training Efficiency and Effectiveness Comparison. D2-Align outperforms baselines by being both more effective and more efficient. It achieves a higher score in fewer steps, whereas methods like DanceGRPO and Flow-GRPO require over 250 steps to attain a similar level of performance.

To provide a clean image x0 for the reward calculation, we leverage the ground-truth noising strategy. Specifically, we add a known noise ϵ ∼ N(0,I) to a ground-truth image according to the forward process xt = αtx0 +σtϵ to obtain a noisy latent xt. The original image x0 is then recovered using the one-step denoising process described in Eq. (4). As illustrated in Fig. 5, this approach ensures an efficient and effective training workflow. The vector bv is optimized by minimizing the loss Lstage1(bv) = Ec,x

e+ = normalize(etext + bv) (6) e− = normalize(etext − bv) (7)

We then construct a new, guided text embedding e˜text that extrapolates from the negative direction towards the positive one, controlled by a guidance scale ω > 1:

0∼Gθfrozen[−Rguided]. We empirically verify that this learned directional vector provides a superior mechanism for reward correction compared to using hand-picked, discrete vocabulary (detailed in Sec. 5.4).

e˜text = e− + ω · (e+ − e−) (8)

The guided reward, Rguided, is computed by applying the same scoring function to the guided text embedding:

Stage 2. In this stage, we proceed with the alignment

Rguided(x0,c;bv) = score(eimg,e˜text) (9)

A comic portrait of an Indian goddess with realistic shading and fine details set in a nighttime anime style.

An image of a woman with sunglasses and red hair, , rendered with monochrome, black and white properties

The image features Brock Lesnar depicting Iron Man in a dynamic action pose, inspired by several artists.

A high-quality portrait photo of a young White man with rosy cheeks with arched eyebrows

An image of a smiling sloth, rendered with low key photography,

An image of a depiction of a canal in Venice, rendered with

A high-quality portrait photo of a young White man who is attractive with straight hair

A high-quality portrait photo of a young White man with narrow eyes with sideburns

A photo of Big Chungus from Looney Tunes.

dark scene properties

Silhouette properties

|Poor Detail and Weak Texture ❌<br><br>[Figure 92]|
|---|
|Low Diversity of Tonality ❌<br><br>[Figure 93]|
|[Figure 94]<br><br>[Figure 95]<br><br>[Figure 96]<br><br>Low Diversity of Tonality ❌<br><br>[Figure 97]|
|Poor Detail and Weak Texture ❌<br><br>[Figure 98]|
| |
|High Diversity of Tonality and Detailed Texture ✅<br><br>[Figure 99]|

|Poor Skin Texture ❌<br><br>[Figure 100]|
|---|
|Low Diversity of Face ❌<br><br>[Figure 101]|
|[Figure 102]<br><br>[Figure 103]<br><br>[Figure 104]<br><br>Low Diversity of Face ❌<br><br>[Figure 105]|
|Poor Skin Texture ❌<br><br>[Figure 106]|
| |
|High Diversity of Face and Detailed Skin Texture ✅<br><br>[Figure 107]|

[Figure 108]

[Figure 109]

[Figure 110]

Flux

[Figure 111]

[Figure 112]

[Figure 113]

Not Detailed Enough ❌ Oil Texture ❌ Not Detailed Enough ❌

DanceGRPO

[Figure 114]

Lack Character Fidelity ❌

[Figure 115]

[Figure 116]

Not Nighttime ❌ Lack Character Fidelity ❌

Flow-GRPO

[Figure 117]

Not Nighttime ❌ Lack Character Fidelity ❌

[Figure 118]

[Figure 119]

Lack Character Fidelity ❌

SRPO

[Figure 120]

[Figure 121]

[Figure 122]

Lack Character Fidelity ❌ Not Indian Goddess ❌ Not Detailed Enough ❌

| |
|---|
|High Fidelity & Detailed ✅<br><br>[Figure 123]|

| |
|---|
|Requirements Met ✅<br><br>[Figure 124]|

| |
|---|
|High Fidelity & Detailed ✅<br><br>[Figure 125]|

D2-Align

HPDv2 DivGenBench

- Figure 6. Qualitative Comparison of D2-Align against Baselines on the HPDv2 and Our DivGenBench Benchmarks. D2-Align demonstrates superior performance fidelity and diversity, overcoming the concept forgetting and stylistic limitations seen in other methods.

of the generator Gθ. Let b∗v denote the directional vector learned and frozen from Stage 1. The generator is then optimized by minimizing the following loss, which applies the guided reward function with the frozen vector b∗v:

0∼Gθ(c)[−Rguided(x0,c;b∗v)] (10)

Lstage2(θ) = Ec∼D,x

By incorporating the frozen vector b∗v, the optimization objective is fundamentally altered. Instead of naively maximizing the original reward, which is susceptible to the reward model’s intrinsic biases, the generator is now guided by a corrected reward signal. This corrected signal provides a more faithful representation of true human preference by suppressing the reward inflation often caused by these biases. Optimizing for this more credible reward naturally encourages the generator to produce outputs of higher fidelity. Moreover, by preventing the generator from converging on specific patterns that merely cater to the reward model’s biases, our approach forces it to explore a wider range of solutions, thereby preserving generative diversity and enhancing output quality.

###### 4.3. DivGenBench

To quantify Preference Mode Collapse (PMC), we introduce DivGenBench, a benchmark designed to evaluate

a model’s Generative Breadth—its ability to follow diverse, explicit instructions. Existing benchmarks are inadequate for diagnosing PMC, as they typically largely prioritize fidelity [20, 26, 49, 59], or measure output variance from ambiguous prompts[54], or lack comprehensive dimensions[19] and metrics[14, 39] for diversity.

DivGenBench addresses these gaps with a ”keyworddriven” prompt design that actively probes a model’s generative boundaries. Its hierarchical structure encompasses four key dimensions: ID (high-level semantics), Style (midlevel aesthetics), Layout (structure and relations), and Tonal (low-level physics). We systematically constructed our DivGenBench of 3,200 prompts, each dimension comprising 800 prompts, which are constructed by augmenting a set of base templates—variously derived from [28, 37, 65]—with explicit attribute keywords sourced from [42, 53] and other custom-designed keywords.

Recognizing that a single metric is insufficient, DivGenBench employs a suite of dimension-customized metrics for robust evaluation. We combine low-level image features with domain-specific extractors [11, 40, 51]. We then apply our four bespoke metric calculations to compute the final scores: the Identity Divergence Score (IDS), Artistic Style Coverage (ASC), Spatial Dispersion Index (SDI), and

- Table 1. Comprehensive Quantitative Evaluation. We compare FLUX and advanced methods from the combined perspectives of human preference alignmen, semantic consistency and accuracy, showcasing performance under two distinct reward configurations: HPS-v2.1 and HPS-v2.1 + CLIP. Ranking is performed independently for each reward configuration among the RL-based methods. The highest score is shown in bold, and the second-highest score is underlined.

Human Preference Alignment Semantic Consistency & Accuracy Aesthetic ↑ ImageReward ↑ Pick Score ↑ Q-Align ↑ HPS-v2.1 ↑ CLIP ↑ DeQA ↑ GenEval ↑

Method

FLUX 6.417 1.670 0.240 4.922 0.310 0.315 4.456 0.663 Reward Model: HPS-v2.1 DanceGRPO [62] 6.068 1.664 0.241 4.930 0.361 0.293 4.400 0.522

- Flow-GRPO [39] 5.888 1.703 0.239 4.969 0.367 0.283 4.432 0.517 SRPO [50] 6.614 1.533 0.241 4.866 0.296 0.302 4.357 0.623 Ours 6.450 1.771 0.246 4.969 0.343 0.323 4.484 0.636

Reward Model: HPS-v2.1 + CLIP DanceGRPO [62] 6.030 1.520 0.236 4.962 0.333 0.286 4.423 0.581

- Flow-GRPO [39] 6.060 1.744 0.239 4.950 0.343 0.317 4.454 0.636 SRPO [50] 6.394 1.616 0.240 4.958 0.310 0.309 4.495 0.659 Ours 6.671 1.762 0.246 4.970 0.314 0.328 4.498 0.660

Photographic Variance Score (PVS), inspired by [13, 48]. Detailed metric designs are provided in Supp.

###### 5. Experiment

###### 5.1. Implementation Details

We evaluate our method against several leading RL alignment baselines on the state-of-the-art T2I model, FLUX.1.Dev [30] . The primary reward signal is the HPS-v2.1 trained on the Human Preference Dataset (HPD) v2 [59].

Following DanceGRPO [62], we also conduct experiments with a multi-reward combination of HPS-v2.1 and CLIP Score [23] for a more comprehensive comparison Our evaluation is comprehensive, assessing model performance from two key perspectives: quality and diversity. For quality assessment, we employ a suite of metrics. Aesthetic appeal is measured using state-of-the-art predictors [1, 29, 57, 60]. To evaluate text-image alignment, we use CLIP Score. Furthermore, we assess semantic fidelity using [20]. For diversity evaluation, we utilize DivGenBench.

###### 5.2. Qualitative Evaluation

The qualitative comparisons on HPDv2 and our DivGenBench are presented in Fig. 6. Our method consistently outperforms the baselines in terms of fidelity, text-to-image alignment and diversity.

On HPDv2 (Cols. 1-3), baselines prone to PMC, such as DanceGRPO, FlowGRPO and SRPO, exhibit significant concept forgetting, failing to render well-known subjects like ”Big Chungus” (Col. 1) and ”Iron Man” (Col. 3). They also fail on complex semantic consistency, unable to synthesize all attributes for the ”Indian goddess... nighttime anime

style” prompt (Col. 2), which our method achieves. This degradation is magnified on our DivGenBench (Cols. 4-8). DanceGRPO and Flow-GRPO suffer severe PMC, generating near-identical faces despite prompts requiring ID diversity (Cols. 4-6). Similarly, they fail on tonal diversity (Cols. 7-8), defaulting to a homogeneous aesthetic instead of the requested ”low key” or ”black and white” styles. In contrast, our method not only preserves these concepts but also demonstrates superior fidelity and detail over other baselines like FLUX and SRPO. Our method successfully balances high fidelity with robust instruction-following capabilities. More qualitative results can be found in Supp.

###### 5.3. Quantitative Evaluation

We quantitatively evaluate the model’s performance from the combined perspectives of quality and diversity. From a quality perspective, as shown in Tab. 1, our method achieves superior performance on key metrics. This indicates that our approach successfully aligns with human preference while maintaining high fidelity. Furthermore, our method also outperforms all baselines in text-to-image alignment and semantic consistency, demonstrating its robust instruction-following capabilities. In terms of diversity, Tab. 2 shows that our method attains the highest scores on the DivGenBench. This result provides strong evidence that our proposed method effectively prevents the model from over-optimizing towards the reward model’s preferred modes, thereby mitigating the PMC. It is particularly noteworthy that DanceGRPO and FlowGRPO, which achieve artificially high scores on HPS-v2.1, exhibit a significant drop in diversity. This finding further validates the necessity of our approach and highlights the limitations of relying solely on a single reward score for evaluation. To provide

[Figure 126]

[Figure 127]

[Figure 128]

- Figure 7. Ablation Studies on the Key Components and Hyperparameters of D2-Align. (Left) Analysis of the bv directional vector’s convergence. (Middle) Sensitivity analysis of the guidance scale ω. (Right) Effectiveness of the learned directional vector. The radar chart compares our full method (using the learned bv) against variants using manually-defined discrete tokens (e.g., ”realistic”) and no directional correction.

a measure of alignment with human preference, we conducted a user study following [62]. D2-Align achieved the best win rate (see Supp. for details).

- Table 2. Quantitative Evaluation of Generative Diversity on DivGenBench. We compare the FLUX and other advanced methods, showcasing performance under two distinct reward configurations and four metrics, i.e., Identity Divergence Score (IDS), Artistic Style Coverage (ASC), Spatial Dispersion Index (SDI), and Photographic Variance Score (PVS). Ranking is performed independently for each reward configuration among the RLbased methods. The best score is shown in bold, and the secondbest score is underlined.

Method IDS ↓ ASC ↑ SDI ↑ PVS ↑ FLUX 0.280 0.179 0.563 0.408 Reward Model: HPS-v2.1

DanceGRPO 0.348 0.130 0.488 0.259 Flow-GRPO 0.391 0.044 0.389 0.168 SRPO 0.259 0.234 0.580 0.352 Ours 0.251 0.253 0.636 0.412

Reward Model: HPS-v2.1 + CLIP

DanceGRPO 0.252 0.108 0.523 0.344 Flow-GRPO 0.276 0.133 0.506 0.316 SRPO 0.306 0.198 0.559 0.402 Ours 0.237 0.247 0.631 0.418

###### 5.4. Ablation Study

In this section, we conduct a comprehensive ablation study on D2-Align to validate its effectiveness and robustness.

Convergence Analysis of the Learned Vector (bv). To analyze the convergence of the directional vector, we evaluated the performance of bv sampled from different training steps. As depicted in Fig. 7 (left), the corrective effect of bv becomes evident and robust after approximately 2000 training steps, yielding significant performance improvements

from that point onward.

Ablation on the Guidance Scale (ω). To investigate the impact of the guidance scale ω, we evaluate the optimization performance using the learned directional vector bv under various scale settings. We empirically set ω to 1.5, as this value yields superior results on both HPS-v2.1 and Pickscore, as demonstrated in Fig. 7 (middle).

Effectiveness of the Learned Vector (bv). To validate that our learned continuous vector, bv, more effectively mitigates over-optimization towards high rewards. We contrast our method against a baseline that uses manually selected discrete words representing fidelity and aesthetics. As illustrated in Fig. 7 (right), our approach consistently outperforms this baseline of hand-picked words across all evaluated metrics. Furthermore, compared to using the uncorrected reward signal, our method demonstrates a comprehensive improvement across the board.

Generalizability of the Learned Vector (bv). To evaluate the generalizability of our learned bv, we applied it as a corrective signal to other existing methods that are susceptible to PMC. The results show that even when integrated into these external frameworks, bv consistently mitigates the collapse by achieving a more favorable balance between fidelity and diversity. A detailed analysis of these crossmethod experiments is provided in Supp.

###### 6. Conclusion

We identified Preference Mode Collapse (PMC), a reward hacking from a diversity perspective. We proposed D2Align, a novel framework that counteracts PMC by mitigating the reward model’s intrinsic biases, thus breaking the common trade-off between fidelity and diversity. To measure this phenomenon, we introduced DivGenBench, a new diversity-centric benchmark. Our results show that D2-Align achieves state-of-the-art performance, simultaneously improving both human preference scores and generative diversity.

###### References

- [1] Aesthetic predictor v2.5. https://github.com/ discus0434/aesthetic-predictor-v2-5, 2025. Accessed: 2025-06-10. 7
- [2] Kevin Black, Michael Janner, Yilun Du, Ilya Kostrikov, and Sergey Levine. Training diffusion models with reinforcement learning. arXiv preprint arXiv:2305.13301, 2023. 2
- [3] Chubin Chen, Jiashu Zhu, Xiaokun Feng, et al. S2-guidance: Stochastic self guidance for training-free enhancement of diffusion models. arXiv preprint arXiv:2508.12880, 2025. 2, 14
- [4] Chubin Chen, Jiashu Zhu, Chenyang Zhu, Jiangshan Wang, Nisha Huang, Chengyu Fang, Jiahong Wu, Xiangxiang Chu, and Xiu Li. Storyctrl: Customized story visualization with fine-grained control, 2025. 2
- [5] Tingxiu Chen, Yilei Shi, Zixuan Zheng, Bingcong Yan, Jingliang Hu, Xiao Xiang Zhu, and Lichao Mou. Ultrasound image-to-video synthesis via latent dynamic diffusion models. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 764–774. Springer, 2024. 2
- [6] Xiangxiang Chu, Hailang Huang, Xiao Zhang, Fei Wei, and Yong Wang. Gpg: A simple and strong reinforcement learning baseline for model reasoning. arXiv preprint arXiv:2504.02546, 2025. 3
- [7] Xiangxiang Chu, Renda Li, and Yong Wang. Usp: Unified self-supervised pretraining for image generation and understanding. arXiv preprint arXiv:2503.06132, 2025. 2
- [8] Hyungjin Chung, Jeongsol Kim, Geon Yeong Park, Hyelin Nam, and Jong Chul Ye. Cfg++: Manifold-constrained classifier free guidance for diffusion models. arXiv preprint arXiv:2406.08070, 2024. 2
- [9] Kevin Clark, Paul Vicol, Kevin Swersky, and David J Fleet. Directly fine-tuning diffusion models on differentiable rewards. arXiv preprint arXiv:2309.17400, 2023. 2
- [10] Kevin Clark, Paul Vicol, Kevin Swersky, and David J Fleet. Directly fine-tuning diffusion models on differentiable rewards. arXiv preprint arXiv:2309.17400, 2023. 2, 4
- [11] Jiankang Deng, Jia Guo, Niannan Xue, and Stefanos Zafeiriou. Arcface: Additive angular margin loss for deep face recognition. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4690–4699, 2019. 6, 13
- [12] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in neural information processing systems, 34:8780–8794, 2021. 2, 3
- [13] Mischa Dombrowski, Weitong Zhang, Sarah Cechnicka, Hadrien Reynaud, and Bernhard Kainz. Image generation diversity issues and how to tame them. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 3029–3039, 2025. 7, 13
- [14] Jiajun Fan, Shuaike Shen, Chaoran Cheng, Yuxin Chen, Chumeng Liang, and Ge Liu. Online reward-weighted finetuning of flow matching with wasserstein regularization. In The Thirteenth International Conference on Learning Representations, 2025. 6

- [15] Ying Fan and Kangwook Lee. Optimizing ddpm sampling with shortcut fine-tuning. arXiv preprint arXiv:2301.13362,

2023. 2

- [16] Ying Fan, Olivia Watkins, Yuqing Du, Hao Liu, Moonkyung Ryu, Craig Boutilier, Pieter Abbeel, Mohammad Ghavamzadeh, Kangwook Lee, and Kimin Lee. Dpok: Reinforcement learning for fine-tuning text-to-image diffusion models. Advances in Neural Information Processing Systems, 36:79858–79885, 2023. 2
- [17] Chengyu Fang, Chunming He, Longxiang Tang, Yuelin Zhang, Chenyang Zhu, Yuqi Shen, Chubin Chen, Guoxia Xu, and Xiu Li. Integrating extra modality helps segmentor find camouflaged objects well, 2025. 2
- [18] Chun-Mei Feng, Kai Yu, Yong Liu, Salman Khan, and Wangmeng Zuo. Diverse data augmentation with diffusions for effective test-time prompt tuning. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2704–2714, 2023. 2
- [19] Felix Friedrich, Thiemo Ganesha Welsch, Manuel Brack, Patrick Schramowski, and Kristian Kersting. Beyond overcorrection: Evaluating diversity in t2i models with divbench. arXiv preprint arXiv:2507.03015, 2025. 6
- [20] Dhruba Ghosh, Hannaneh Hajishirzi, and Ludwig Schmidt. Geneval: An object-focused framework for evaluating textto-image alignment. Advances in Neural Information Processing Systems, 36:52132–52152, 2023. 6, 7
- [21] Shashank Gupta, Chaitanya Ahuja, Tsung-Yu Lin, Sreya Dutta Roy, Harrie Oosterhuis, Maarten de Rijke, and Satya Narayan Shukla. A simple and effective reinforcement learning method for text-to-image diffusion fine-tuning. arXiv preprint arXiv:2503.00897, 2025. 2
- [22] Xiaoxuan He, Siming Fu, Yuke Zhao, Wanli Li, Jian Yang, Dacheng Yin, Fengyun Rao, and Bo Zhang. Tempflow-grpo: When timing matters for grpo in flow models. arXiv preprint arXiv:2508.04324, 2025. 3
- [23] Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. Clipscore: A reference-free evaluation metric for image captioning. arXiv preprint arXiv:2104.08718,

2021. 3, 7, 12, 17

- [24] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022. 5
- [25] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 2, 3
- [26] Kaiyi Huang, Kaiyue Sun, Enze Xie, Zhenguo Li, and Xihui Liu. T2i-compbench: A comprehensive benchmark for open-world compositional text-to-image generation. Advances in Neural Information Processing Systems, 36:78723–78747, 2023. 6
- [27] Nisha Huang, Henglin Liu, Yizhou Lin, Kaer Huang, Chubin Chen, Jie Guo, Tong-yee Lee, and Xiu Li. Mate: Images are all you need for material transfer via diffusion transformer. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 15117–15126, 2025. 2
- [28] Kimmo Karkkainen and Jungseock Joo. Fairface: Face attribute dataset for balanced race, gender, and age for bias measurement and mitigation. In Proceedings of the

- IEEE/CVF Winter Conference on Applications of Computer Vision, pages 1548–1558, 2021. 6, 13
- [29] Yuval Kirstain, Adam Polyak, Uriel Singer, Shahbuland Matiana, Joe Penna, and Omer Levy. Pick-a-pic: An open dataset of user preferences for text-to-image generation. Advances in neural information processing systems, 36:36652– 36663, 2023. 2, 7
- [30] Black Forest Labs. Flux. https://github.com/ black-forest-labs/flux, 2024. 7, 12
- [31] Kimin Lee, Hao Liu, Moonkyung Ryu, Olivia Watkins, Yuqing Du, Craig Boutilier, Pieter Abbeel, Mohammad Ghavamzadeh, and Shixiang Shane Gu. Aligning textto-image models using human feedback. arXiv preprint arXiv:2302.12192, 2023. 2
- [32] Jiachen Lei, Keli Liu, Julius Berner, Haiming Yu, Hongkai Zheng, Jiahong Wu, and Xiangxiang Chu. There is no vae: End-to-end pixel-space generative modeling via selfsupervised pre-training, 2025. 2
- [33] Junzhe Li, Yutao Cui, Tao Huang, Yinping Ma, Chun Fan, Miles Yang, and Zhao Zhong. Mixgrpo: Unlocking flowbased grpo efficiency with mixed ode-sde. arXiv preprint arXiv:2507.21802, 2025. 3, 5
- [34] Renda Li, Hailang Huang, Fei Wei, Feng Xiong, Yong Wang, and Xiangxiang Chu. Adacurl: Adaptive curriculum reinforcement learning with invalid sample mitigation and historical revisiting. arXiv preprint arXiv:2511.09478, 2025. 3
- [35] Xiaomin Li, Yixuan Liu, Takashi Isobe, Xu Jia, Qinpeng Cui, Dong Zhou, Dong Li, You He, Huchuan Lu, Zhongdao Wang, et al. Reneg: Learning negative embedding with reward guidance. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 23636–23645, 2025. 5
- [36] Yuming Li, Yikai Wang, Yuying Zhu, Zhongyu Zhao, Ming Lu, Qi She, and Shanghang Zhang. Branchgrpo: Stable and efficient grpo with structured branching in diffusion models. arXiv preprint arXiv:2509.06040, 2025. 3
- [37] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In European conference on computer vision, pages 740–755. Springer, 2014. 6, 13
- [38] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling, 2023. 2, 3
- [39] Jie Liu, Gongye Liu, Jiajun Liang, Yangguang Li, Jiaheng Liu, Xintao Wang, Pengfei Wan, Di Zhang, and Wanli Ouyang. Flow-grpo: Training flow matching models via online rl. arXiv preprint arXiv:2505.05470, 2025. 2, 3, 4, 6, 7, 13
- [40] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Chunyuan Li, Jianwei Yang, Hang Su, Jun Zhu, et al. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. arXiv preprint arXiv:2303.05499, 2023. 6, 14
- [41] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022. 2, 3

- [42] Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang. Deep learning face attributes in the wild. In Proceedings of International Conference on Computer Vision (ICCV), 2015. 6, 13
- [43] Fangyuan Mao, Aiming Hao, Jintao Chen, Dongxia Liu, Xiaokun Feng, Jiashu Zhu, Meiqi Wu, Chubin Chen, Jiahong Wu, and Xiangxiang Chu. Omni-effects: Unified and spatially-controllable visual effects generation. arXiv preprint arXiv:2508.07981, 2025. 2
- [44] Zichen Miao, Jiang Wang, Ze Wang, Zhengyuan Yang, Lijuan Wang, Qiang Qiu, and Zicheng Liu. Training diffusion models towards diverse image generation with reinforcement learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10844– 10853, 2024. 2
- [45] Mihir Prabhudesai, Russell Mendonca, Zheyang Qin, Katerina Fragkiadaki, and Deepak Pathak. Video diffusion alignment via reward gradients. arXiv preprint arXiv:2407.08737,

2024. 2

- [46] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 2, 3
- [47] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 22500– 22510, 2023. 2
- [48] Seyedmorteza Sadat, Otmar Hilliges, and Romann M Weber. Eliminating oversaturation and artifacts of high guidance scales in diffusion models. In The Thirteenth International Conference on Learning Representations, 2024. 2, 7, 14
- [49] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in neural information processing systems, 35:36479–36494, 2022. 6
- [50] Xiangwei Shen, Zhimin Li, Zhantao Yang, Shiyi Zhang, Yingfang Zhang, Donghao Li, Chunyu Wang, Qinglin Lu, and Yansong Tang. Directly aligning the full diffusion trajectory with fine-grained human preference, 2025. 4, 7, 13
- [51] Gowthami Somepalli, Anubhav Gupta, Kamal Gupta, Shramay Palta, Micah Goldblum, Jonas Geiping, Abhinav Shrivastava, and Tom Goldstein. Measuring style similarity in diffusion models. arXiv preprint arXiv:2404.01292, 2024. 6, 13
- [52] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020. 3
- [53] Wei Ren Tan, Chee Seng Chan, Hernan Aguirre, and Kiyoshi Tanaka. Improved artgan for conditional synthesis of natural image and artwork. IEEE Transactions on Image Processing, 28(1):394–409, 2019. 6, 13

- [54] Revant Teotia, Candace Ross, Karen Ullrich, Sumit Chopra, Adriana Romero-Soriano, Melissa Hall, and Matthew Muckley. Dimcim: A quantitative evaluation framework for default-mode diversity and generalization in text-to-image generative models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 16431–16440,

2025. 6

- [55] Jiangshan Wang, Junfu Pu, Zhongang Qi, Jiayi Guo, Yue Ma, Nisha Huang, Yuxin Chen, Xiu Li, and Ying Shan. Taming rectified flow for inversion and editing. arXiv preprint arXiv:2411.04746, 2024. 2
- [56] Yibin Wang, Zhimin Li, Yuhang Zang, Yujie Zhou, Jiazi Bu, Chunyu Wang, Qinglin Lu, Cheng Jin, and Jiaqi Wang. Pref-grpo: Pairwise preference reward-based grpo for stable text-to-image reinforcement learning. arXiv preprint arXiv:2508.20751, 2025. 3, 5
- [57] Haoning Wu, Zicheng Zhang, Weixia Zhang, Chaofeng Chen, Liang Liao, Chunyi Li, Yixuan Gao, Annan Wang, Erli Zhang, Wenxiu Sun, et al. Q-align: Teaching lmms for visual scoring via discrete text-defined levels. arXiv preprint arXiv:2312.17090, 2023. 7
- [58] Meiqi Wu, Jiashu Zhu, Xiaokun Feng, Chubin Chen, Chen Zhu, Bingze Song, Fangyuan Mao, Jiahong Wu, Xiangxiang Chu, and Kaiqi Huang. Imagerysearch: Adaptive testtime search for video generation beyond semantic dependency constraints. arXiv preprint arXiv:2510.14847, 2025. 2
- [59] Xiaoshi Wu, Yiming Hao, Keqiang Sun, Yixiong Chen, Feng Zhu, Rui Zhao, and Hongsheng Li. Human preference score v2: A solid benchmark for evaluating human preferences of text-to-image synthesis. arXiv preprint arXiv:2306.09341,

2023. 3, 4, 5, 6, 7, 12, 17

- [60] Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong. Imagereward: Learning and evaluating human preferences for textto-image generation. Advances in Neural Information Processing Systems, 36:15903–15935, 2023. 2, 4, 5, 7
- [61] Ryan Xu, Dongyang Jin, Yancheng Bai, Rui Lan, Xu Duan, Lei Sun, and Xiangxiang Chu. Scalar: Scale-wise controllable visual autoregressive learning. arXiv preprint arXiv:2507.19946, 2025. 2
- [62] Zeyue Xue, Jie Wu, Yu Gao, Fangyuan Kong, Lingting Zhu, Mengzhao Chen, Zhiheng Liu, Wei Liu, Qiushan Guo, Weilin Huang, et al. Dancegrpo: Unleashing grpo on visual generation. arXiv preprint arXiv:2505.07818, 2025. 2, 3, 4, 5, 7, 8, 13, 14, 16, 17
- [63] Kai Yang, Jian Tao, Jiafei Lyu, Chunjiang Ge, Jiaxin Chen, Weihan Shen, Xiaolong Zhu, and Xiu Li. Using human feedback to fine-tune diffusion models without any reward model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8941– 8951, 2024. 2
- [64] Kaixing Yang, Xulong Tang, Ziqiao Peng, Yuxuan Hu, Jun He, and Hongyan Liu. Megadance: Mixture-of-experts architecture for genre-aware 3d dance generation, 2025. 2
- [65] Jiahui Yu, Yuanzhong Xu, Jing Yu Koh, Thang Luong, Gunjan Baid, Zirui Wang, Vijay Vasudevan, Alexander Ku, Yin-

- fei Yang, Burcu Karagol Ayan, et al. Scaling autoregressive models for content-rich text-to-image generation. arXiv preprint arXiv:2206.10789, 2(3):5, 2022. 6, 13
- [66] Mariia Zameshina, Olivier Teytaud, and Laurent Najman. Diverse diffusion: Enhancing image diversity in text-toimage generation. arXiv preprint arXiv:2310.12583, 2023. 2
- [67] Hanyang Zhao, Haoxian Chen, Ji Zhang, David D Yao, and Wenpin Tang. Score as action: Fine-tuning diffusion generative models by continuous-time reinforcement learning. arXiv preprint arXiv:2502.01819, 2025. 2
- [68] Kaiwen Zheng, Huayu Chen, Haotian Ye, Haoxiang Wang, Qinsheng Zhang, Kai Jiang, Hang Su, Stefano Ermon, Jun Zhu, and Ming-Yu Liu. Diffusionnft: Online diffusion reinforcement with forward process, 2025. 4, 5
- [69] Chenyang Zhu, Kai Li, Yue Ma, Longxiang Tang, Chengyu Fang, Chubin Chen, Qifeng Chen, and Xiu Li. Instantswap: Fast customized concept swapping across sharp shape differences, 2024. 2

### Taming Preference Mode Collapse via Directional Decoupling Alignment in Diffusion Reinforcement Learning

#### Supplementary Material

###### A. D2-Align Algorithm

Algorithm 2 D2-Align Stage 2: Guided Alignment

Require: Policy model Gθ (as ϵθ); reward model R (Φimg,Φtext,score); prompt dataset C; guidance scale ω; Stage 2 total timesteps T2 for training; diffusion coefficients αt,σ.

The D2-Align framework is a two-stage process that decouples reward signal correction from policy alignment. The detailed process for each stage is provided in Algorithm 1 and Algorithm 2.

Require: Frozen directional vector b∗v (from Stage 1). Ensure: Optimized policy model Gθ∗.

- Stage 1 (Algorithm 1) focuses on learning a directional

correction vector bv. In this stage, the policy model Gθ is frozen, and only the vector bv is optimized to create a guided reward signal that corrects for reward model biases.

- Stage 2 (Algorithm 2) then performs the guided align-

- 1: Unfreeze generator Gθ (ϵθ)
- 2: for timestep T2 to t = 1 do
- 3: Sample prompt c ∼ C
- 4: Generate clean image x0 ∼ Gθ(c)
- 5: Get text embedding etext ← Φtext(c)
- 6: Sample noise ϵgt ∼ N(0,I)
- 7: Create noisy latent xt ← αtx0 + σtϵgt
- 8: Predict noise ϵpred ← ϵθ(xt,t,c)
- 9: Perform one-step ODE sampling to get xt−1
- 10: Reconstruct xˆ0 ← (xt−1 − σt−1ϵgt)/αt−1
- 11: Get image embedding eimg ← Φimg(xˆ0)
- 12: Calculate e+ ← normalize(etext + b∗v)
- 13: Calculate e− ← normalize(etext − b∗v)
- 14: Construct guided embedding e˜text ← e−+ω·(e+− e−)
- 15: Compute guided reward Rguided ← score(eimg,e˜text)
- 16: Update generator θ by minimizing Lstage2(θ) = −Rguided
- 17: return Gθ∗ ← Gθ

ment of the policy model itself. The optimized vector b∗v from Stage 1 is frozen, and the policy model Gθ is unfrozen. The generator’s parameters θ are then updated by optimiz-

ing for the guided reward signal defined by b∗v.

Algorithm 1 D2-Align Stage 1: Learning Directional Correction

Require: Initial policy model Gθ (as ϵθ); reward model R (Φimg,Φtext,score); prompt dataset C; guidance scale ω; Stage 1 total timesteps T1 for training; diffusion coefficients αt,σ.

Ensure: Optimized directional vector b∗v.

- 1: Initialize learnable directional vector bv ∈ Rd
- 2: Freeze generator Gθ (ϵθ)
- 3: for timestep T1 to t = 1 do
- 4: Sample prompt c ∼ C
- 5: Generate clean image x0 ∼ Gθ(c)
- 6: Get text embedding etext ← Φtext(c)
- 7: Sample noise ϵgt ∼ N(0,I)
- 8: Create noisy latent xt ← αtx0 + σtϵgt
- 9: Predict noise ϵpred ← ϵθ(xt,t,c)
- 10: Perform one-step ODE sampling to get xt−1
- 11: Reconstruct xˆ0 ← (xt−1 − σt−1ϵgt)/αt−1
- 12: Get image embedding eimg ← Φimg(xˆ0)
- 13: Calculate e+ ← normalize(etext + bv)
- 14: Calculate e− ← normalize(etext − bv)
- 15: Construct guided embedding e˜text ← e−+ω·(e+− e−)
- 16: Compute guided reward Rguided ← score(eimg,e˜text)
- 17: Update bv by minimizing Lstage1(bv) = −Rguided
- 18: return b∗v ← bv

###### B. Experimental Setting Details

We conduct all reinforcement learning experiments using FLUX.1.Dev [30] as the base T2I model, following the state-of-the-art methodology. For fairness and comprehensive comparison, all alignment baselines are retrained on the Human Preference Dataset (HPD) v2 [59] using two standardized reward combinations: HPS-v2.1 [59] and HPSv2.1 [59] + CLIP [23] Score. All experiments are conducted on NVIDIA H20 GPUs with 96GB memory.

###### B.1. Baseline Implementation Details

To ensure a fair comparison, we standardize the training steps for major baselines and align all reward functions. For the baseline comparisons, we follow the original implementations provided in their official repositories. Details on the training steps for each competing method are provided below.

###### Table 3. Comprehensive Hyperparameters of D2-Align.

Category Parameter Value

Random Seed 42 Resolution (H × W) 720 × 720 Mixed Precision bf16 Gradient Checkpointing Enabled Dataloader Workers 4 Use EMA Disabled

GENERAL & MODEL

Optimizer AdamW Learning Rate 5 × 10−6 Weight Decay 1 × 10−4 Gradient Clip Norm 0.1 LR Warmup Steps 0 Gradient Accumulation Steps 2

OPTIMIZATION

ω 1.5

RL ALIGNMENT

- Stage 1 (bv) Max Steps 3000
- Stage 2 (Gθ) Max Steps 20

Sampling Steps 25 Inference Steps 50 Train Guidance 1.0 Shift 3 Train Batch Size 1 SP Size / Train SP Batch Size 1 / 1

INFERENCE

- • DanceGRPO [62]: We follow the default open-source hyperparameters and train the model for 300 steps.
- • Flow-GRPO [39]: Hyperparameters were adopted from a stable PickScore training configuration. The training length is set to 300 steps to maintain alignment with DanceGRPO.
- • SRPO [50]: The model is trained using its default configuration for the official recommended length of 20 steps.

###### B.2. D2-Align Hyperparameters

Our proposed D2-Align framework employs the two-stage training approach. Stage 1 (Directional Correction) trains the correction vector bv for 3000 steps, after which the policy model Gθ is aligned in Stage 2 with training 20 steps. Key hyperparameters specific to the D2-Align are listed in Tab. 3.

###### C. DivGenBench Construction and Metrics C.1. Prompt Construction and Examples

Our 3,200 ”keyword-driven” prompts are systematically generated using distinct templates for each of the four dimensions, as summarized in Tab. 4. Each dimension uses a specific templating strategy to augment base content with explicit attribute keywords.

• ID: Motivated by [28], prompts are built using the template: ”A high-quality portrait photo of a/an [age] [ethnicity] [gender] [features]”. We combine 3 ages, 6 ethnicities, 2 genders, and 40 physical features (e.g., ”with arched eyebrows”) derived from CelebA[42]. A conflictresolution mechanism ensures logical coherence.

- • Style: We pair 27 classic art styles from WikiArt[53] with base content prompts from Parti[65].
- • Layout: We combine 80 COCO[37] object classes with 4 counts (two, three, four, five) using designed template to test numerical control and spatial diversity.
- • Tonal: We augment Parti[65] base prompts with 18 finegrained keywords across 3 sub-dimensions: Saturation, Contrast, and Brightness.

###### C.2. Metric Calculation Details

To quantify the extent of PMC, we employ four dimensioncustomized metrics that measure the model’s generative breadth.

Identity Divergence Score (IDS): We use ArcFace[11] to extract a 512-D identity embedding vi for each of N generated faces. The score is the average pairwise cosine similarity between all unique identity vectors. A lower score signifies greater identity diversity.

N

N

vi · vj ∥vi∥∥vj∥

2 N(N − 1)

(11)

IDS =

i=1

j=i+1

Artistic Style Coverage (ASC): Inspired by the Image Retrieval Score (IRS)[13], this metric quantifies the retrievable Style diversity of a generative model relative to the ground-truth data’s diversity. We use the CSD[51] feature extractor (F) and define three datasets: a ground-truth (GT) training set Xtrain (with Ntrain images from WikiArt[53]), a GT reference set Xtest (with Nsample = Ntest from WikiArt[53]), and the generated synthetic set Xsynth (with Nsample = Nsynth from DivGenBench).The score is computed in three steps:

- • Retrieval: For a query set Xq (either Xtest or Xsynth), we first find Xlearned, the set of unique training images that are the nearest neighbor (in CSD feature space F) to at least one image in Xq, as defined in Eq. (12). We then get the count Nlearned = |Xlearned(Xq)|.

Xlearned(Xq) = {x ∈ Xtrain | ∃g ∈ Xq s.t. x = arg minx′∈Xtraind(F(g),F(x′))}

(12)

- • Estimation: Using Nlearned, Nsample, and Ntrain, we compute the maximum likelihood estimate of the total ”learnable” images s∗ (Eq. (13)), and the corresponding ”infinite” retrieval score, IRS∞ (Eq. (14)). This estimates the fraction of Xtrain that would be retrieved given infinite query samples.

s∗(Xq) = arg maxsP(Nlearned,Nsample,s) (13) IRS∞(Xq) = s∗(Xq)/Ntrain (14)

- • Adjustment: To correct for the feature extractor’s inherent ”measurement gap”, the final ASC score is the Adjusted IRS (IRS∞,a), shown in Eq. (15). This is the

###### Table 4. Prompt Construction Templates and Examples for Each Dimension of DivGenBench. Brackets indicate keywords sampled from curated attribute lists.

Dimension One of Templates Example ID A high-quality portrait photo of an [age] [ethnicities] [gen-

A high-quality portrait photo of an elderly South Asian woman with arched eyebrows wearing a necklace.

der] with [feature].

Style A painting in the style of [style], depicting [base prompt]. A painting in the style of Rococo, depicting a silver fire

hydrant next to a sidewalk. Layout A studio shot of [number] [object] on a clean white back-

A studio shot of three boats on a clean white background.

ground.

Tonal An image of [base prompt], rendered with [tonal] properties.

An image of ten children on a couch, rendered with dimly lit properties.

ratio of the synthetic set’s estimated diversity to the real reference set’s estimated diversity. A higher score is better.

IRS∞(Xsynth) IRS∞(Xtest)

(15)

ASC =

Spatial Dispersion Index (SDI): This metric evaluates the diversity of object layouts across multiple images generated from the same text prompt, effectively measuring the model’s ability to produce spatially varied results. For M images per prompt, we first use Grounding DINO[40] to detect the bounding boxes Li = {bj} of the target objects in each image. The Similarity SimLayout (Eq. 16) is calculated by finding the optimal bipartite matching of the detected bounding boxes via the Hungarian algorithm on the IoU matrix, normalized by the maximum number of objects. We then compute the average pairwise Layout Similarity (SimLayout) between all pairs of images, as defined in Eq. 17. Finally, the SDI is defined as one minus the average Layout Similarity across all M images, averaged over all prompts P (Eq. 18). A higher score signifies greater Layout diversity.

1 max(|Li|,|Lp|) ×

SimLayout(Li,Lp) =

(16)

IoU(bj ∈ Li,bl ∈ Lp)

(j,l)∈P

where P is the set of optimal matching pairs found via the Hungarian algorithm.

2 M(M − 1)

SimLayout =

M

M

SimLayout(Li,Lp)

i=1

p=i+1

(17)

P

1 P

SDI =

r=1

1 − Sim(Layoutr) (18)

Photographic Variance Score (PVS): This metric, inspired by APG[48], quantifies the spread of generated tonal values. For a set of N images G = {gi}Ni=1, we first extract a scalar value for each perceptual dimension. For each image gi, Saturation (si) is the mean of the S-channel (from an RGB-to-HSV conversion), Brightness (vi) is the mean of the V-channel, and Contrast (ci) is the standard deviation of the grayscale-converted image. We then form three value sets s = {si}Ni=1, v = {vi}Ni=1, and c = {ci}Ni=1. PVS is the sum of the standard deviations of these three sets. A higher score indicates greater tonal control.

PVS = std(s) + std(v) + std(c) (19)

###### D. Extended Experiments

###### D.1. User Study on HPDv2

To validate the effectiveness of D2-Align in alignment with human preferences, we conducted a comprehensive user study following [3, 62]. We compared our method against the base model (FLUX) and three competitive RL-based baselines: DanceGRPO, Flow-GRPO, and SRPO.

###### D.1.1. Experimental Setup

We randomly selected 100 prompts from HPDv2. For each prompt, we generated images using all five methods with the same random seed. We recruited 20 evaluators who were presented with the generated images in a randomized, blind manner. The evaluators were asked to select the best image based on the following criteria:

- • Detail Preservation: The clarity, sharpness, and richness of details in the generated image.
- • Color Consistency: The naturalness, harmony, and realism of the colors.
- • Image-Text Alignment: How well the generated image accurately reflects the content and intent of the text prompt.
- • Overall: Considering all the above factors, which image do you prefer?

- Table 5. Prompts Used for The Qualitative Examples in Figure 1 of The Main Paper Grouped by Their Corresponding Dimension. The numbering (1-16) corresponds to the images in Figure 1, read from left-to-right, top-to-bottom within each dimensional category.

###### No. Dimension Prompt

- 1 Face

A high-quality portrait photo of a young Middle Eastern woman who is attractive with arched eyebrows

- 2 A high-quality portrait photo of a middle-aged Middle Eastern woman with a receding hairline who is attractive
- 3 A high-quality portrait photo of a middle-aged White woman with a big nose
- 4 A high-quality portrait photo of a young Middle Eastern woman with an oval face

- 5 Style

An artwork of corgi pizza, in the Baroque style.

- 6 Imagine a panda bear playing ping pong using a blue paddle against an ostrich using a red paddle. Now, picture it in the style of Fauvism.
- 7 An image of a giant cobra snake made from salad, with strong Action painting influences.
- 8 A masterpiece of Pointillism, showing a hot air balloon

9-12 Layout A clear, top-down view of two tennis rackets arranged on a large white table.

- 13 Tonal

a woman with sunglasses and red hair, monochrome, black and white

- 14 a tiger in a forest, desaturated, muted colors
- 15 An image of an ornate treasure chest with a broad sword propped up against it, glowing in a dark cave, rendered with natural colors properties
- 16 Photograph of three red lego blocks, captured with neon colors, fluorescent

[Figure 129]

Detail Preservation Color Consistency Image Text Alignment Overall

[Figure 130]

- Figure 8. Human Preference Evaluation on HPDv2. We conducted a user study comparing D2-Align against the base model FLUX and state-of-the-art RL alignment methods (DanceGRPO, Flow-GRPO, SRPO). The evaluation spans four distinct dimensions: Detail Preservation, Color Consistency, Image-Text Alignment, and Overall Preference. D2-Align achieves a dominant lead in Detail Preservation (61.7%) and Image-Text Alignment (52.2%), significantly outperforming baselines that suffer from mode collapse artifacts. Ultimately, our method secures the highest Overall Preference rate of 48.2%.

[Figure 131]

- Figure 9. Human Preference on Diversity (DivGenBench). We evaluated user preferences across four key diversity dimensions: Identity, Style, Layout, and Tonal. The results reveal a severe PMC in existing RL baselines (DanceGRPO, Flow-GRPO), which often score lower than the Base Model (FLUX), particularly in Tonal and Style diversity. In contrast, D2-Align consistently achieves the highest preference rates (e.g., 37.3% in Style and 35.2% in Identity), demonstrating its ability to break the trade-off between human preference and generative diversity.

###### D.1.2. Analysis of Results

The results of the user study are presented in Figure 8. The findings demonstrate a clear and consistent preference for

our proposed method, D2-Align, across all evaluated metrics. Specifically, in the Detail Preservation category, D2Align was preferred in 61.7% of cases, significantly out-

performing the runner-up, SRPO (16.5%). A similar dominant trend is observed for Image-Text Alignment, where D2Align achieved a 52.2% preference rate. Furthermore, for Color Consistency, our method was chosen 29.0% of the time, again marking a lead over all baselines.

Aggregating the votes, the Overall preference for D2Align stands at 48.2%, confirming its comprehensive superiority. This strong performance in human evaluations validates that D2-Align not only improves alignment from a theoretical standpoint but also translates to tangible and perceptually superior generation quality that is easily recognized by human users.

###### D.2. User Study on DivGenBench

While the HPDv2 study confirms our method’s alignment quality, it does not explicitly measure the severity of PMC. To quantitatively assess whether models sacrifice diversity for higher scores, we conducted a second user study using our proposed DivGenBench.

###### D.2.1. Experimental Setup

We sampled 20 distinct templates from each of the four dimensions in DivGenBench: Identity, Style, Layout, and Tonal, totaling 80 evaluation sets. For each set, we generated images using varied prompts designed to probe the model’s generative boundaries (e.g., requesting specific ”Low-key” lighting or distinct ”Cubism” styles). 20 evaluators were asked to identify which model best reflected the requested diversity and avoided generating repetitive or homogeneous outputs.

###### D.2.2. Analysis of Results

The diversity preference results are presented in Figure 9. Two critical observations emerge from the data:

Evidence of Preference Mode Collapse. The results provide strong empirical evidence of PMC in existing RL methods. In several dimensions, the baseline RL models (DanceGRPO and Flow-GRPO) perform significantly worse than the unaligned Base Model (FLUX). For instance, in Tonal Diversity, Flow-GRPO drops to a mere 7.7% preference rate, and DanceGRPO to 10.3%, compared to FLUX’s 26.7%. Similarly, in Style Diversity, FlowGRPO (8.6%) lags behind FLUX (18.2%). This confirms that naively optimizing for reward models drives the generator into a narrow mode (e.g., always generating overexposed or realistic styles), actively destroying the inherent diversity of the base model.

Superior Diversity Preservation. In contrast, D2-Align effectively mitigates this collapse. Our method achieves the highest preference scores across all four dimensions, surpassing both the collapsed baselines and the Base Model.

- • Identity & Style: We achieve dominant preference rates of 35.2% for Identity and 37.3% for Style, significantly outperforming the runner-up SRPO (about 23%). This indicates our method can generate diverse faces and artistic styles without reverting to a mean template.
- • Layout & Tonal: Crucially, in the dimensions most susceptible to collapse, our method maintains robustness. In Tonal Diversity, where baselines fail, D2-Align leads with 33.7%, demonstrating successful disentanglement of ”quality” from ”lighting bias.”

These results, combined with the HPDv2 findings, validate that D2-Align can simultaneously improve human preference alignment while preserving and even enhancing generative diversity.

###### D.3. Generalizability Study with DanceGRPO

To further evaluate the intrinsic value and generalizability of our learned corrective signal bv, we conducted an extension experiment by applying it as a plug-and-play component to an external Reinforcement Learning framework. Specifically, we selected DanceGRPO [62], a representative method that, while effective in improving alignment, is susceptible to PMC. The objective of this experiment is to verify whether the directionality captured by bv acts as a universal corrective signal for the reward model and can effectively mitigate mode collapse in other algorithms.

###### D.3.1. Experimental Setup

We maintain the original training logic and hyperparameters of DanceGRPO. The only modification lies in the reward calculation mechanism. Let b∗v denote the optimal parameter learned and frozen from Stage 1 of our method. We substitute the naive reward of DanceGRPO with our proposed guided reward Rguided. Formally, during the DanceGRPO training process, for every generated sample (x0,c), the reward is computed as:

Rguided(x0,c;b∗v) = score(Φimg(x0),e˜text) (20)

where e˜text is the rectified text embedding constructed using b∗v via Eq. (8) (as defined in the main paper), with the guidance scale ω kept consistent.

###### D.3.2. Analysis of Results

The quantitative comparisons in Tab. 6 and Tab. 7 demonstrate that integrating our corrective signal significantly enhances the robustness of DanceGRPO. In terms of alignment, while vanilla DanceGRPO achieves the highest HPSv2.1 score, it suffers from regression in generalized metrics. In contrast, applying our learned bv effectively mitigates reward overfitting: it achieves a 4.7% improvement in Aesthetic Score and restores semantic consistency with a 6.1% gain in CLIP score, suggesting a shift towards true human preference. Crucially, for diversity, our method effectively

###### Table 6. Comprehensive Quantitative Evaluation of Metrics for Human Preference Alignment and Semantic Consistency. We compare FLUX, DanceGRPO, and DanceGRPO incorporated with our learned bv. All RL-based methods utilize HPS-v2.1 as the reward model. Ranking is performed between the RL-based methods. The best score is shown in bold.

Human Preference Alignment Semantic Consistency & Accuracy Aesthetic ↑ ImageReward ↑ Pick Score ↑ Q-Align ↑ HPS-v2.1 ↑ CLIP ↑ DeQA ↑ GenEval ↑

Method

FLUX 6.417 1.670 0.240 4.922 0.310 0.315 4.456 0.663 DanceGRPO 6.068 1.664 0.241 4.930 0.361 0.293 4.400 0.522 DanceGRPO + Our learned bv 6.353 1.677 0.242 4.947 0.319 0.311 4.496 0.641

resolves the mode collapse observed in the baseline. By filtering out the low-diversity manifold, our corrective signal reduces the IDS score by 20.1% and expands the ASC score by a remarkable 57.7%, surpassing both the baseline and the pre-trained FLUX. These results confirm that bv forces the external optimizer to explore a broader solution space without requiring complex re-tuning of the training configuration.

###### Table 7. Quantitative Evaluation of Generative Diversity on DivGenBench. We compare FLUX, DanceGRPO, and DanceGRPO enhanced with our learned bv. All RL-based methods utilize HPS-v2.1 as the reward model. We report Identity Divergence Score (IDS), Artistic Style Coverage (ASC), Spatial Dispersion Index (SDI), and Photographic Variance Score (PVS). Ranking is performed between the RL-based methods. The best score is shown in bold.

Method IDS ↓ ASC ↑ SDI ↑ PVS ↑ FLUX 0.280 0.179 0.563 0.408 DanceGRPO 0.348 0.130 0.488 0.259 DanceGRPO + Our learned bv 0.278 0.205 0.604 0.437

###### E.3. Results on DivGenBench

We provide a comprehensive visual evaluation of our method on DivGenBench. Observing a critical trade-off in existing work, we strategically focus our comparative visualization on the two state-of-the-art methods that achieved the highest performance on the HPDv2 benchmark yet simultaneously recorded the lowest diversity scores on DivGenBench.

As shown in Fig.12, our method generates distinct identities, avoiding the mode collapse seen in baselines. It further demonstrates a broad range of artistic styles without defaulting to generic aesthetics (Fig.13), produces diverse spatial layouts (Fig.14), and maintains a wide tonal spectrum in brightness and contrast (Fig.15), contrasting with the monotonic distributions of competing methods.

###### E. Visualization

###### E.1. Prompts for Figure 1

We present the example prompts used to generate the qualitative comparisons in Figure 1 of the main paper. The prompts, grouped by their corresponding dimension, are detailed in Tab. 5.

###### E.2. Results on HPDv2

In this section, we present qualitative comparisons on the HPDv2[59] benchmark to visually evaluate the performance of our method against the baseline approach and advanced RL-based methods. Fig.10 illustrates the visual outputs where all competing RL-based methods were trained using only HPS-v2.1[59] as the reward model. Fig.11 shows the results for the same set of methods, but where the models were trained using a combined reward signal from both HPS-v2.1[59] and CLIP[23], inspired by DanceGRPO[62].

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

No Name of Record or Artist ❌ No Name of Record or Artist ❌ No Name of Record or Artist ❌ No Name of Record or Artist ❌

"The Joy" by Shaun Tan features a colorful and detailed artistic record jacket design.

| |
|---|
|Accurate Logo ✅<br><br>[Figure 138]|

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

Incomplete Logo ❌ Wrong Logo ❌ Wrong Logo ❌ Incomplete Logo ❌

A dolphin swimming in front of a Studio Ghibli logo backdrop.

|Requirements Met ✅<br><br>[Figure 144]|
|---|

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

Not Scribbled Style ❌ Not Pencil Drawing ❌ Not Pencil Drawing ❌ Not Detailed Enough ❌

Tetsuo and Kaneda engage in a race through Neo Tokyo in a pencil drawing featuring a scribbled style.

|Requirements Met ✅<br><br>[Figure 150]|
|---|

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

Fails Plurality Constraint ❌ Fails Plurality Constraint ❌ Lacks Portrait Fidelity ❌ Fails Plurality Constraint ❌

A portrait of Rosalia painted by various artists.

|Extreme Agony Capture ✅<br><br>[Figure 156]|
|---|

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

Low Emotional Depth ❌ Low Emotional Depth ❌ Low Emotional Depth ❌ Low Emotional Depth ❌

A Ukrainian survivor takes a final selfie as they flee a nuclear blast, with their damaged body bleeding and running in fear.

|Requirements Met ✅<br><br>[Figure 162]|
|---|

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

No Shelf ❌ Not Dark ❌ Not Dark ❌ No Shelf ❌

This is a very dark picture of a room with a shelf.

- Figure 10. Qualitative comparison of D2-Align against SOTAs on the HPDv2 benchmark. All RL-based methods are trained using HPSv2.1 as the reward model. 18

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

Incomplete Text ❌ Wrong “AITH” for “The” ❌ Extra Words, “END” “The” ❌ No Text ❌

Image featuring the title of a manga, "The End of the World" by Eiichiro Oda.

| |
|---|
|Well-executed ✅<br><br>[Figure 174]|

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

No Glass Double Doors ❌ No Glass Double Doors ❌ No Glass Double Doors ❌ No Glass Double Doors ❌

Personal computer desk room with large glass double doors.

|Requirements Met ✅<br><br>[Figure 180]|
|---|

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

No Pokemon Trading Cards ❌ Lacks Character Fidelity ❌ Lacks Character Fidelity ❌ No Pokemon Trading Cards ❌

A mash-up of Tom and Jerry cartoon characters and Pokemon trading cards.

|Guideline Aligned ✅<br><br>[Figure 186]|
|---|

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

Not Intricate Enough ❌ Not Dark Ink Drawings ❌ Not Dark Ink Drawings ❌ Incomplete Face ❌

A horror movie inspired by Junji Ito's artwork, featuring intricate and dark ink drawings.

|Rich Detail ✅<br><br>[Figure 192]|
|---|

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

Too Simple ❌ Poor Resolution Output ❌ Irrelevant White Sofa ❌ Too Simple ❌

A wooden table sitting in the middle of a room.

|Well Subject Match ✅<br><br>[Figure 198]|
|---|

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

Lacks Character Fidelity ❌ Lacks Character Fidelity ❌ Lacks Character Fidelity ❌ Lacks Character Fidelity ❌

The image depicts John Egbert from Homestuck.

- Figure 11. Qualitative comparison of D2-Align against SOTAs on the HPDv2 benchmark. All RL-based methods are trained using HPSv2.1 and CLIP as the reward models. 19

|Low Diversity ❌<br><br>[Figure 204]|Low Diversity ❌<br><br>[Figure 205]|[Figure 206]<br><br>[Figure 207]<br><br>[Figure 208]|
|---|---|---|
| | |High Diversity ✅<br><br>[Figure 209]|

A high-quality portrait photo of a young White woman wearing earrings who is attractive

A high-quality portrait photo of a middle-aged South Asian woman with a receding hairline

A high-quality portrait photo of a middle-aged Latinx woman with straight hair with blond hair

|[Figure 210]<br><br>Low Diversity ❌<br><br>[Figure 211]<br><br>[Figure 212]<br><br>[Figure 213]|Low Diversity ❌<br><br>[Figure 214]| |
|---|---|---|
| | |High Diversity ✅<br><br>[Figure 215]|

A high-quality portrait photo of a young Middle Eastern man who is attractive with sideburns

A high-quality portrait photo of a young Latinx man with big lips with brown hair

A high-quality portrait photo of a middle-aged White man with wavy hair

- Figure 12. Qualitative comparison on the ID dimension of DivGenBench. Our method generates diverse identities adhering to required demographic features. 20

Imagine downtown Shanghai at sunrise. detailed ink wash. Now, picture it in the style of Ukiyo e.

Post Impressionism painting of the Eiffel Tower in a desert.

A painting in the style of Naive Art Primitivism, depicting a bowl of soup that looks like a monster knitted out of wool.

A painting in the style of Ukiyo e, depicting a tennis court with a basketball hoop in one corner.

An artwork of two people facing each other, in the Minimalism style.

A Baroque interpretation of food.

|Low Diversity ❌<br><br>[Figure 216]|Low Diversity ❌<br><br>[Figure 217]|[Figure 218]<br><br>[Figure 219]<br><br>[Figure 220]<br><br>[Figure 221]<br><br>[Figure 222]<br><br>[Figure 223]|
|---|---|---|
| | |High Diversity ✅<br><br>[Figure 224]|

- Figure 13. Qualitative comparison on the Style dimension of DivGenBench. Our method faithfully renders diverse artistic styles specified in the prompts. 21

|Low Diversity ❌<br><br>[Figure 225]<br><br>Wrong Numbers<br><br>Wrong Numbers|Low Diversity ❌<br><br>[Figure 226]<br><br>Wrong Numbers<br><br>Wrong Numbers<br><br>Wrong Numbers|[Figure 227]|
|---|---|---|
| | |High Diversity ✅<br><br>[Figure 228]|

A clear, top-down view of three sandwiches arranged on a large white table.

A clear, top-down view of three sandwiches arranged on a large white table.

A clear, top-down view of three sandwiches arranged on a large white table.

|[Figure 229]<br><br>Low Diversity ❌<br><br>[Figure 230]|Low Diversity ❌<br><br>[Figure 231]| |
|---|---|---|
| | |High Diversity ✅<br><br>[Figure 232]|

A simple scene featuring four bears on a large white table.

A simple scene featuring four bears on a large white table.

A simple scene featuring four bears on a large white table.

- Figure 14. Qualitative comparison on the Layout dimension of DivGenBench. Our method not only achieves precise adherence to object counts but also generates diverse and novel spatial arrangements. 22

DanceGRPO Flow-GRPO D2 –Align(Ours)

|Low Diversity ❌<br><br>[Figure 233]<br><br>Not cloud|Low Diversity ❌<br><br>[Figure 234]<br><br>Not cloud|[Figure 235]<br><br>[Figure 236]<br><br>[Figure 237]|
|---|---|---|
| | |High Diversity ✅<br><br>[Figure 238]|

a cloud in the shape of a elephant, high key photography

a bike rack with some bike locks attached to it but no bicycles, characterized by low key photography, dark scene

Brightness

Dimly lit, a depiction of a man with puppet

|[Figure 239]<br><br>[Figure 240]<br><br>[Figure 241]<br><br>Low Diversity ❌<br><br>[Figure 242]|Low Diversity ❌<br><br>[Figure 243]| |
|---|---|---|
| | |High Diversity ✅<br><br>[Figure 244]|

An image of The Oriental Pearl in sketch style, rendered with silhouette properties

a phone from the 20s, characterized by high contrast, hard lighting

Contrast

a tornado passing over a corn field, soft lighting, overcast day

|[Figure 245]<br><br>[Figure 246]<br><br>[Figure 247]<br><br>Low Diversity ❌<br><br>[Figure 248]<br><br>No bond|Low Diversity ❌<br><br>[Figure 249]<br><br>No bond| |
|---|---|---|
| | |High Diversity ✅<br><br>[Figure 250]|

A desaturated, muted colors image of a man standing on a street corner

Saturation

A neon colors, fluorescent image of bond

two cats, vibrant, saturated colors

- Figure 15. Qualitative comparison on the Tonal dimension of DivGenBench. Our method generates a diverse spectrum of brightness, contrast, and saturation levels while maintaining high image fidelity.23

