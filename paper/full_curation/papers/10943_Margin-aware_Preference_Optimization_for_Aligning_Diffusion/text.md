## Margin-Aware Preference Optimization for Aligning Diffusion Models without Reference

### Jiwoo Hong1*, Sayak Paul2*, Noah Lee1, Kashif Rasul2, James Thorne3, Jongheon Jeong4

1KAIST AI 2Hugging Face 3Theia Insights 4Korea University jiwoo hong@alumni.kaist.ac.kr, sayak@huggingface.co

# arXiv:2406.06424v2[cs.CV]3Dec2025

##### Abstract

Modern preference alignment methods, such as DPO, rely on divergence regularization to a reference model for training stability—but this creates a fundamental problem we call “reference mismatch.” In this paper, we investigate the negative impacts of reference mismatch in aligning text-to-image (T2I) diffusion models, showing that larger reference mismatch hinders effective adaptation given the same amount of data, e.g., as when learning new artistic styles, or personalizing to specific objects. We demonstrate this phenomenon across text-toimage (T2I) diffusion models and introduce margin-aware preference optimization (MaPO), a reference-agnostic approach that breaks free from this constraint. By directly optimizing the likelihood margin between preferred and dispreferred outputs under the Bradley-Terry model without anchoring to a reference, MaPO transforms diverse T2I tasks into unified pairwise preference optimization. We validate MaPO’s versatility across five challenging domains: (1) safe generation, (2) style adaptation, (3) cultural representation, (4) personalization, and (5) general preference alignment. Our results reveal that MaPO’s advantage grows dramatically with reference mismatch severity, outperforming both DPO and specialized methods like DreamBooth while reducing training time by 15%. MaPO thus emerges as a versatile and memory-efficient method for generic T2I adaptation tasks.

Warning: This paper contains examples of harmful content, including explicit text and images.

Code — https://github.com/mapo-t2i/mapo Project — https://mapo-t2i.github.io

### Introduction

Diffusion models have become dominant for modeling highdimensional data distributions due to their scalability (Ho, Jain, and Abbeel 2020; Kingma et al. 2021; Rombach et al. 2022; Podell et al. 2024; Peebles and Xie 2023; Esser et al. 2024), handling diverse conditioning modalities including text (Li et al. 2022; Strudel et al. 2022), images (Ho, Jain, and Abbeel 2020; Podell et al. 2024), and audio (Kong et al. 2021; Evans et al. 2024). Their capabilities in human-centered applications have motivated fine-tuning for preference alignment

*These authors contributed equally. Copyright © 2026, Association for the Advancement of Artificial Intelligence (www.aaai.org). All rights reserved.

[Figure 1]

Figure 1: Reference mismatch, heterogeneity between the training model’s distribution pθ and the data distribution pdata, leads to suboptimal preference learning. MaPO is a reference-free preference optimization method, more robust to such heterogeneity while adapting to the preference.

in areas like safe generation (Shen et al. 2024; Schramowski et al. 2023), stylistic rendering (Hertz et al. 2024), and personalization (Ruiz et al. 2023; von R¨utte et al. 2023).

T2I diffusion model alignment generates outputs reflecting desired attributes through preference optimization (Lee, Liu

- et al. 2023; Yoon et al. 2023; Fan et al. 2023; Wallace et al. 2023; Li et al. 2024b; Yuan et al. 2024), with reinforcement learning methods treating denoising as multi-step decisionmaking via proximal policy optimization (Schulman, Wolski et al. 2017). These methods employ reference models with divergence regularization to stabilize training, prevent overfitting, and preserve core capabilities (Ziegler et al. 2020; Wang
- et al. 2024a; Skalse et al. 2022; Pang et al. 2023).

However, constraining models to specific references limits flexibility in learning new content (Tajwar et al. 2024). We term this reference mismatch (Figure 1)—when reference model features differ from preference data, often triggered by using stronger proprietary models for dataset curation (Wu et al. 2023; Lambert et al. 2025). In T2I models, this manifests as stylistic preferences (”cartoon”) or distributional biases from limited personalization data (Bianchi et al. 2023; Liu et al. 2024a; Lu et al. 2023; Hertz et al. 2024), affecting

all task-specific methods aiming to induce new attributes while preserving general capabilities (Ruiz et al. 2023; Lee et al. 2024a). Addressing these discrepancies is crucial for applying preference alignment to diverse downstream tasks.

We investigate how reference mismatch hinders T2I diffusion model alignment when using direct alignment methods (Wallace et al. 2023), finding adverse effects particularly significant with large distributional gaps. We introduce marginaware preference optimization (MaPO), a novel referenceagnostic method that defines the Bradley-Terry model score function (Bradley and Terry 1952) directly from training model likelihood and incorporates DDPM loss (Ho, Jain, and Abbeel 2020) to incrementally align data and model distributions. While reference-free alignment has been studied in language modeling (Xu, Sharaf et al. 2024; Hong, Lee, and Thorne 2024; Meng, Xia, and Chen 2024; Gupta et al. 2025), we develop the first such objective for T2I diffusion models.

Our experiments on five representative T2I tasks—safe generation, style learning, cultural representation, personalization, and preference alignment—show that MaPO overcomes reference mismatch challenges while maintaining alignment benefits. For example, MaPO outperforms three preference alignment methods, including InPO (Lu et al. 2025a) and two personalization methods, including DreamBooth (Ruiz et al. 2023), while reducing training time by 15% compared to Diffusion-DPO (Wallace et al. 2023). This study provides a unified framework for T2I tasks and preference learning, empirically validated across five distinct tasks.

### Preliminaries

Text-to-image diffusion models Text-to-image (T2I) diffusion models (Rombach et al. 2022; Saharia et al. 2022; Ramesh et al. 2022) learn to denoise random noise xT ∼ N(0,I) into a data sample x0 ∼ pdata(x0) conditioned on text prompt c. They model a discrete Markov process pθ(xt−1|xt,c) that predicts xt−1 from xt for timesteps t = T,...,1, where xt follows the forward diffusion process:

xt ∼ q(xt|x0) where q(xt|x0) = N(αtx0,σt2I), (1) with noise schedule parameters αt and σt (Ho, Jain, and

Abbeel 2020). The backward denoising process is defined as:

T

pθ(xt−1|xt,c). (2)

pθ(x0:T|c) =

t=1

To maximize the likelihood of observed data x0 under model pθ(x0|c), the evidence lower bound is minimized. Ho, Jain, and Abbeel (2020) parameterized pθ as a noise predictor ϵθ(xt,c,t), yielding an MSE objective with random noise ϵ ∼ N(0,I):

LDDPM ≤ Ex

[−log pθ(x0 | c)] ≤ T · Ex

T

(3)

0,ϵ,t ω(λt)∥ϵ − ϵθ (xt,c,t)∥2 ,

where ω(λt) depends on the signal-to-noise ratio λt = log(αt2/σt2) (Song and Ermon 2019; Kingma et al. 2021). In practice, a simplified loss is used:

LMSE(c,x0) := Eϵ,t ∥ϵ − ϵθ (xt,c,t)∥2 . (4)

Preference optimization for alignment Alignment finetunes generative models to produce human-preferred outputs (Ouyang, Wu et al. 2022). Human preferences are often collected as pairs (xw,xl) given prompt c, where xw (“chosen”) is preferred over xl (“rejected”). The Bradley-Terry model (Bradley and Terry 1952) models the preference probability:

exp(r(xw,c)) exp(r(xw,c)) + exp(r(xl,c))

p(xw ≻ xl | c) =

, (5)

where r(x,c) denotes the reward function. This approach, popularized in language model alignment (Ziegler et al. 2020; Rafailov et al. 2023), is often combined with reinforcement learning like PPO (Schulman, Wolski et al. 2017) in RLHF. The RLHF objective maximizes:

θ(x|c) [r(x,c)]−βDKL (pθ(x|c) ∥ pref(x|c)), (6)

Ex∼p

max

θ

where pref is the reference model (typically the pre-trained initialization) and β weights the KL constraint. The optimal policy for objective (6) is:

1 Z(c) · pref(x | c) · exp

p∗(x | c) =

1 β · r(x,c) , (7)

where Z(c) is the partition function. Direct alignment algorithms (DAAs) like DPO (Rafailov et al. 2023) achieve this without RL by directly optimizing the implicit reward. For T2I diffusion models, Wallace et al. (2023) adapted DPO to preferences over diffusion paths xw1:T and xl1:T:

LDiff-DPO(c,xw1:T,xl1:T) := −log σ β log pθ(x

(8)

w 1:T |c)

l 1:T |c)

pref(xw1:T|c) − β log pθ(x

pref(xl1:T|c) .

### Margin-aware Preference Optimization

In this section, we first establish the concept of reference mismatch when aligning text-to-image (T2I) diffusion models and their negative impacts. Then, we propose margin-aware preference optimization (MaPO), a novel preference alignment method for diffusion models that aims to mitigate the issue by eliminating the need for a reference model.

#### Motivation: Reference mismatch problem

We define reference mismatch as the divergence (e.g., KL divergence) between the preference data distribution pdata and the initial reference model pref. The negative impacts of reference mismatch have been empirically observed in language models, particularly in DPO training (Guo et al. 2024; Tajwar et al. 2024; Xu et al. 2024; Tang et al. 2024). This issue mainly arises from the key assumption in DPO, namely, that the chosen and rejected samples (xw,xl) are drawn from the optimal policy p∗ (7) (Rafailov et al. 2023). However, in practice, preference data rarely originate from the optimal policy (Xu et al. 2024; Tang et al. 2024; Liu et al. 2024b), often being generated from external sources (Wallace et al. 2023; Li et al. 2024b; Zhu, Xiao, and Honavar 2025). This discrepancy violates this assumption and hinders optimal policy learning in DPO, highlighting the necessity of

[Figure 2]

- Figure 2: Reference mismatch between the model generation

xθ0 and data xD0 quantified by the cosine distance in the DINOv2 embeddings. The personalization task has the lowest similarity, implying the highest mismatch.

addressing the reference mismatch. A possible workaround to address the reference mismatch of DPO is lowering the hyperparameter β (8) to reduce the dependency of pθ to pref; however, this approach often triggers performance degradation in generation quality, due to that lowering β also weakens the log-likelihood objective of pθ(x|c) (Rafailov et al. 2024; Pal et al. 2024; Shi et al. 2024; Liu, Liu, and Cohan 2024). Therefore, lowering β does not mitigate reference mismatch and its negative impacts but deteriorates the model, making this scenario’s necessity of pref questionable.

Reference mismatch in T2I tasks Similarly, in T2I diffusion models, the optimality of Diffusion-DPO is prone to reference mismatch. As an instance, we quantify the reference mismatch in five representative downstream tasks in T2I diffusion models: general preference alignment (Wallace et al. 2023; Li et al. 2024b), cultural representation (Bianchi et al. 2023; Liu et al. 2024a), safe generation (Schramowski et al. 2023; Kim et al. 2023), style learning (Lu et al. 2023; Hertz et al. 2024), and personalization (Ruiz et al. 2023; Lee et al. 2024a). We measure the reference mismatch with image similarity score using DINOv2 (Oquab et al. 2024) between xθ0 ∼ pθ(x|c) and (xD0 ,c) ∼ pdata(x|c): i.e., less reference mismatch with higher score. In Figure 2, generic preference alignment and personalization tasks were shown to have the smallest and largest reference mismatch out of five tasks. This shows that the degree of reference mismatch significantly varies by task, limiting the versatility of direct alignment methods with reference models like Diffusion-DPO in the downstream tasks of T2I diffusion models.

#### Approach: Reference-free diffusion alignment

We propose a new preference optimization algorithm that eliminates the need for a reference model in diffusion alignment. Overall, the key idea is to define the reference-agnostic score function in the Bradley-Terry (BT) model.

Objective function of MaPO Given a preference dataset pdata of triplets of the form (c,xl0,xw0 ), each of which consists of a prompt c and a preference image pair (xw0 ,xl0) given

c. MaPO optimizes a T2I diffusion model pθ with: LMaPO(c,xw0 ,xl0) := LMSE(c,xw0 ) + β1LMargin(c,xl0,xw0 )

(9)

LMargin(c,xw0 ,xl0) := −log σ ϕβ(LMSE(c,xw0 )) − ϕβ(LMSE(c,xl0)) ,

(10)

and LMSE is the standard DDPM objective in (3) that maximizes the likelihood for “chosen” pairs (c,xw0 ) in (9), and LMargin (10) is the proposed margin-aware regularization that defines the score function in the BT model using the gap of LMSE between xw0 and xl0, modulated by a link function ϕβ:

β

ℓ exp(ℓ) − 1

. (11)

ϕβ(ℓ) :=

In a nutshell, (10) aims to regularize pθ to (i) ensure that xw and xl achieve sufficient likelihood margin, and (ii) fuse the term once they have the margin. In this way, MaPO incorporates preference pairs (xl,xw) upon simple distribution matching and defines a new preference optimization, which notably requires no reference model.

Joint matching and alignment Supervised fine-tuning (SFT) is one of straightforward approaches for matching the distribution of pθ to pdata (Kumar et al. 2022; Sun 2024). We incorporate the standard diffusion loss (4), computed with the “chosen” samples xw, into MaPO (9) as an SFT to incrementally match the distribution of pθ to pdata throughout the alignment. While SFT has been conventionally adopted to initially match pθ before preference learning (Bai et al. 2022; Rafailov et al. 2023; Meng, Xia, and Chen 2024), making overall training multi-stage, this often induces an additional distribution mismatch during the preference learning phase due to static (i.e., off-policy) preference data (Guo et al. 2024). Thus, we adopt SFT within the preference learning stage, consistently matching pθ to pdata while learning the preference to prevent additional mismatches.

Preference learning as a margin regularization We aim to eliminate the use of pref for preference optimization given the negative impacts of the noisy divergence penalty discussed above. Recall that under the Bradley-Terry model, a preference distribution can be modeled as follows:

p(x1 ≻ x2|c) = σ (f (c,x1) − f (c,x2)), (12) where f(c,x) represents the general representation of score function that assigns a scalar score to the prompt c and the image x pair. DPO parameterizes f with pθ and pref as rDPO,

pθ(x,c) pref(x,c)

+ log Z(c), (13)

rDPO(x,c) = β log

as Z(c) as a partition function for the prompt c from the maximum entropy reinforcement learning (Wallace et al. 2023; Rafailov et al. 2024). However, misguiding of pref is one factor that hinders desired preference learning as discussed previously. Furthermore, as implicit reward rDPO is not bounded either way, it is prone to overfitting by rDPO(c,xl) and rDPO(c,xw) easily diverging to maximize their margin

with logistic loss (12) (Azar et al. 2023; Kim et al. 2024) and eventually deteriorating the model in extreme cases (Liu, Liu, and Cohan 2024; Shi et al. 2024).

From this vein, we introduce bounded link function (11)

that can define the score function f in (12) without pref. Along with the reference-agnostic design, it prevents the excessive

divergence problem of rDPO by being bounded within (0,1). Here, the hyperparameter β of (11) controls the temperature of the score function, allowing (10) to be minimized with less likelihood margin between (c,xw0 ) and (c,xl0) when β gets larger. Finally, we weight (10) with β−1 to cancel out the proportional impact of β in ∇θLMargin, since the gradient of (10) is proportional to β (see Supplementary).

Theoretical justification for reference-free link functions The effectiveness of reference-free methods like MaPO, ORPO (Hong, Lee, and Thorne 2024), and SimPO (Meng, Xia, and Chen 2024) can be understood through the lens of the Bradley-Terry (BT) model’s flexibility. The BT model (12) only requires a score function f(c,x) that assigns scalar values preserving preference relationships—it does not mandate any specific functional form. While DPO’s log-ratio formulation (13) emerges from maximum entropy RL theory, it represents just one possible instantiation.

The key requirements for a valid score function in preference learning are: (i) monotonicity with respect to generation quality, (ii) bounded outputs to prevent optimization instabilities, and (iii) preservation of preference orderings (Sun, Shen, and Ton 2025; Gupta et al. 2025). MaPO’s link function ϕβ

(11) satisfies all these criteria using only the model’s likelihood (via LMSE), without requiring pref. The bounded nature of ϕβ ∈ (0,1) particularly addresses the divergence issues of unbounded rewards (Azar et al. 2023; Kim et al. 2024).

This theoretical understanding explains why diverse reference-free formulations succeed, i.e., ORPO’s log-odds and SimPO’s log-probability are alternative score functions that maintain preference relationships through modelintrinsic measures. By eliminating the divergence penalty, these methods avoid the pitfalls of reference mismatch while still effectively propagating preference signals. Our work extends this principle to diffusion models, demonstrating its applicability across generative modeling paradigms.

Unifying T2I fine-tuning as preference alignment Despite its broad formulation, it has been conventionally believed that applying preference optimization to diverse T2I fine-tuning tasks beyond general preference alignment, e.g., for style adaptation, is limited in practice; this is possibly due to the fact that reference mismatch in typical T2I finetuning can be more severe than in language alignment. By circumventing the reference mismatch through a referencefree alignment, MaPO expands the range of T2I diffusion model fine-tuning tasks where pairwise preference optimization can be effectively applied. Once we have a specific target image x0 to stipulate as chosen image xw0 and corresponding prompt c, the sampled generation xl0 ∼ pθ(x|c) from the T2I diffusion model to be trained can be rejected image xl0. Thereby, MaPO can be a versatile alignment method that could be generally used for the T2I fine-tuning tasks based on target datasets of the form (x0,c) ∼ pdata.

### Experiments

We validate the effectiveness and general applicability of MaPO across diverse text-to-image (T2I) diffusion model fine-tuning tasks with five baselines, three from preference alignment and two for personalization. Specifically, we construct a benchmark of five representative T2I downstream adaptation scenarios, each with varying degrees of reference mismatch, including the standard preference alignment task:

- 1. Safe generation (Schramowski et al. 2023)
- 2. Style learning (Lu et al. 2023; Hertz et al. 2024)
- 3. Cultural representation (Bianchi et al. 2023)
- 4. Personalization (Ruiz et al. 2023; Lee et al. 2024a)
- 5. Preference alignment (Wallace et al. 2023)

#### Experimental details

We compare MaPO and other methods by fine-tuning StableDiffusion XL (Podell et al. 2024, SDXL), and evaluate them with task-specific metrics, including HPSv2.1 (Wu et al. 2023), and DINOv2 (Oquab et al. 2024), and VLM-as-aJudge (Chen et al. 2024b,a; Lee et al. 2024b; Yasunaga, Zettlemoyer, and Ghazvininejad 2025). We state the detailed training configurations in the Supplementary.

Specialized preference alignment For a controlled comparison across the tasks under this category, we develop synthetic preference data on top of Pick-a-Pic v2. We sample 20,000 prompts from Pick-a-Pic v2 and extract the core contexts using GPT-3.5-Turbo.1 Then, we employ FLUX.1Schnell (Labs 2024) to generate high-quality images from these “context prompts” in Supplementary. For each task, we employ a vision language model (VLM) as an evaluator following the recent works (Chen et al. 2024b,a; Lee et al. 2024b; Yasunaga, Zettlemoyer, and Ghazvininejad 2025). We use Qwen2-VL-7B-Instruct (Wang et al. 2024b) and GPT-4o (Hurst et al. 2024) as VLM-as-a-judge with the 10point scale evaluation template provided in MJ-Bench (Chen et al. 2024b). By selecting the instances above a score of 5, we finally collect a filtered pairwise preference dataset for safe generation (Pick-Safety), cultural representation (PickCulture), and style learning (Pick-Cartoon). We compare MaPO against Diffusion-DPO (Wallace et al. 2023) by training on each task. To evaluate if the model is aligned to a particular aspect (e.g., if the generations are safer than before), we use the same evaluation template and VLM judge on the prompts in HPDv2.1 (Wu et al. 2023) test set.

Personalization We compare MaPO against direct consistency optimization (Lee et al. 2024a, DCO) and DreamBooth (Ruiz et al. 2023), which are designed specifically for this task. We test these methods on two low-shot DreamBooth datasets (Ruiz et al. 2023). We evaluate if the specific entity is well represented in the output through image-to-image similarities using DINOv2 (Oquab et al. 2024), instructionfollowing abilities with SigLIP (Zhai et al. 2023), and if the aesthetics in the original model are preserved with Aesthetics (Schuhmann 2023). We applied additional techniques

1https://platform.openai.com/docs/models/gpt-3.5-turbo

Qwen2-VL

8.8

SFT

DPO

8.7

MaPO

Score(1-10)

Score(1-10)

8.6

8.5

8.4

8.3

0 1,250 2,500 5,000

6.0

| |GPT-4o SFT<br><br>DPO| | | |
|---|---|---|---|---|
| |MaPO| | | |
| | | | | |

5.5

5.0

Score(1-10)

Score(1-10)

4.5

4.0

3.5

3.0

2.5

2.0

0 1,250 2,500 5,000

Train Set Size

9.25

| | | |Qwen2-VL| |
|---|---|---|---|---|
| | | |SFT<br><br>DPO<br><br>MaPO| |
| | | | | |
| | | | | |

9.00

8.75

Score(1-10)

8.50

8.25

8.00

7.75

7.50

7.25

0 1,250 2,500 5,000

| |GPT-4o SFT DPO MaPO| | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

9.25

9.00

8.75

Score(1-10)

8.50

8.25

8.00

7.75

7.50

0 1,250 2,500 5,000

Train Set Size

| |Qwen2-VL SFT| | | |
|---|---|---|---|---|
| |DPO<br><br>MaPO| | | |
| | | | | |
| | | | | |

8.0

7.5

7.0

6.5

6.0

5.5

5.0

0 1,250 2,500 5,000

| |GPT-4o SFT| | | |
|---|---|---|---|---|
| |DPO MaPO| | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

- 2
- 3
- 4
- 5
- 6
- 7

0 1,250 2,500 5,000

Train Set Size

(a) Cultural representation

(b) Safe generation

(c) Style learning

- Figure 3: Comparison between SFT, DPO, and MaPO on aligning SDXL for cultural representation, safe generation, and style learning tasks with increasing size of train set (train set size of 0 refers to the base SDXL). The tasks are presented in the ascending order of the degree of reference mismatch. We use Qwen2-VL-7B-Instruct (top) and GPT-4o (bottom) as judges.

[Figure 3]

(a) SDXL (b) DPO (c) MaPO (Ours)

[Figure 4]

[Figure 5]

- Figure 4: MaPO in safe generation - The base SDXL, Diffusion-DPO, and MaPO trained with Pick-Safety.

[Figure 6]

[Figure 7]

(a) DPO (b) MaPO (Ours)

Figure 5: MaPO in style learning - SDXL trained on 5,000 instances in Pick-Cartoon with MaPO and DPO.

introduced in DCO (e.g., textual inversion (Gal et al. 2023), low-rank adaptation (Hu et al. 2022)) for MaPO training.

General preference alignment We compare MaPO against four baseline methods, including simple supervised finetuning (SFT), Diffusion-DPO (Wallace et al. 2023), InPO (Lu et al. 2025a), and SmPO (Lu et al. 2025b), by training SDXL on Pick-a-Pic v2 (Kirstain et al. 2023). For fair comparison, we use the official checkpoint from each paper. The models are evaluated on the Pick-a-Pic v2 test set, with PickScore (Kirstain et al. 2023), HPSv2.1 (Wu et al. 2023), and Aesthetics (Schuhmann 2023).

### Results

We present results from evaluating MaPO on five tasks. We remind the reader that each task has a varying degree of reference mismatch. As we will show in this section, MaPO remains on par with or outperforms the task-specific methods while being versatile in diverse reference mismatch scenarios.

#### Safe generation

The performance trend for the safe generation task is similar to that of the cultural representation task. However, the gap

between MaPO and Diffusion-DPO gets larger, as shown in Figure 3b. While MaPO continues to improve as the training set increases, the performance of SFT incrementally decreases. This is expected since unsafe images are placed in rejected images in the pairwise preference dataset, and preparing safe images for SFT is not feasible. Figure 4 further supports the safety-aligned generations after training with MaPO when compared against SDXL and Diffusion-DPO. Although the prompt (symmetrical oil painting of full - body women by samokhvalov) does not contain adverse words or phrases, SDXL returns an unsafe image, and Diffusion-DPO induces minimal improvements over SDXL. Meanwhile, MaPO induces a safe image in Figure 4c by being fully clothed.

#### Style learning

For the style learning task, MaPO outperforms two methods with the largest gap, as shown in Figure 3c. Additionally, a qualitative comparison between Diffusion-DPO and MaPO shows a clear difference in generalizability in Figure 5. When trained on the same 5,000 preference pairs, MaPO renders the generation in a cartoon-style landscape. However, the injected style is not properly depicted in the landscape in Figure 5a, implying the limitation of Diffusion-DPO in large

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

(a) SDXL (b) SFT (c) Diffusion-DPO (d) MaPO (Ours)

- Figure 6: MaPO in cultural representation - While SFT fails to learn the demographic features, Diffusion-DPO and MaPO successfully capture demographic features of East-Asian culture.

Table 1: Assessment of personalized SDXL with DreamBooth, DCO, and MaPO. “Aesthetics”, “SigLIP”, and “DINOv2” measure the image quality (I), text-image alignment (T-I), and seed-wise image similarity (I-I), respectively.

Similarity DreamBooth DCO MaPO (Ours) Aesthetics (I) 5.91 5.92 5.97

SigLIP (T-I) 61.60 70.45 73.60 DINOv2 (I-I) 84.69 89.12 89.51

reference mismatch scenarios. We provide further support on the qualitative analysis in the Supplementary.

#### Cultural representation

In Figure 3a, the score for MaPO monotonically increases as the training set size doubles. While SFT fails to improve, Diffusion-DPO stays on par with MaPO but with a slower improvement rate than MaPO. The samples in Figure 6 empirically show that MaPO successfully induces facial characteristics of East-Asians as intended in Pick-Culture. Both quantitative and qualitative results highlight the effectiveness of alignment methods in low-reference mismatch settings.

#### Personalization

As presented in Figure 7, MaPO successfully induces specific entities depicted in Figure 7f. The examples in Figure 7 collectively demonstrate that MaPO can generalize diverse postures from different prompts in a low-shot personalization regime. We report more detailed samples for Figure 7.

Furthermore, the comparison between MaPO, DreamBooth, and DCO (Table 1) implies that MaPO best induces the appearance of the specific entity while preserving the aesthetics and instruction-following abilities of SDXL by outperforming the other methods in all three metrics measuring image quality, text-image alignment, and seed-level image similarity. This suggests that the reference model may not be required even in the largest reference mismatch, as it is competitive with DCO with a reference model.

Table 2: Four baselines and MaPO evaluation results on general alignment with aesthetic score, HPS v2.1 score, and PickScore on the Pick-a-Pic v2 test set prompt.

Aesthetic HPS v2.1 Pickscore

SDXL 6.03 30.0 22.4 SFT 5.95 29.6 22.0 Diffusion-DPO 6.03 31.1 22.9 InPO 6.14 30.2 22.5 SmPO 6.18 30.8 23.0 MaPO (Ours) 6.34 31.2 22.9

#### General preference alignment

MaPO better aligns the base SDXL with significant improvements in all three metrics (Table 2). The “Aesthetics” score especially highlights the improvements with MaPO compared with baselines, which measures the visual aesthetics of the generated images. In the meantime, HPS v2.1 and PickScore were on par with Diffusion-DPO and SmPO, outperforming SFT and InPO. From the scope of our analysis, Table 2 implies the effectiveness of MaPO in a low reference mismatch regime, adding to the clear strength of MaPO in high reference mismatch regimes in the previous four tasks. Figure 8 shows accurate instruction-following ability induced by MaPO, supporting Table 2.

### Analysis

Positive correlation between the state of reference mismatch and gain of MaPO over DPO Throughout the five tasks in this paper, we can find a positive correlation between the degree of reference mismatch and the performance gap between Diffusion-DPO and MaPO. While personalization in Section and preference alignment in Section employ taskspecific metrics, cultural representation, safe generation, and style learning are tested under controlled settings. In Figure 3, the gain from using MaPO instead of Diffusion-DPO consistently increases as reference mismatch gets larger. This aligns with our analysis, implying the negative impact of the divergence penalty when the reference mismatch is severe. In the task with large reference mismatch, matching the distri-

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

(f) Target dog image

(a) ⟨dog⟩ on a sandy beach

(b) ⟨dog⟩ swimming in a lake

(c) ⟨dog⟩ in a field of wildflowers

(d) ⟨dog⟩ with a lantern at dusk

(e) ⟨dog⟩ in warm, rustic kitchen

- Figure 7: MaPO in personalization - MaPO in the personalization task elicits strong fidelity and generalizability over diverse prompts as shown in Figure 7a to Figure 7e given the target image in Figure 7f, which aligns with quantitative results in Table 1.

[Figure 18]

(a) SDXL (b) SFT (c) Diffusion-DPO (d) MaPO (Ours)

[Figure 19]

[Figure 20]

[Figure 21]

- Figure 8: MaPO in general preference alignment - Given the prompt “Fairy market in giant mushroom forest, bioluminescent lighting, magical creatures trading goods, whimsical fantasy art style”, MaPO precisely depicts the detailed style instructions like “bioluminescent” and “magical creatures trading goods” compared to the base SDXL, SFT, and Diffusion-DPO.

[Figure 22]

(a) 128 (b) 256 (c) 512 (d) 1024 (e) Target

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

- Figure 9: Ablation with different β in personalization. While low β lumps the details of the target, higher β precisely depicts the specific target, e.g., β = 1,024 (Figure 9d).

Table 3: Computational costs of Diffusion-DPO and MaPO using 4 A100s. Training time (“Time”) and peak GPU memory without the model (“GPU Mem.”) measured with batch size 4 in fine-tuning SDXL for 1 epoch on Pick-a-Pic v2.

Diffusion-DPO MaPO (Ours)

Time (↓) 63.5 54.3 (-14.5%) GPU Mem. (↓) 55.9 46.1 (-17.5%) Max Batch (↑) 4 16 (×4)

bution through LMSE is more emphasized by having a larger β, as supported by Figure 9. This empirical result aligns with how DreamBooth (Ruiz et al. 2023) in a personalization task is mainly designed on top of supervised fine-tuning loss. We further analyze the correlation in the Supplementary.

Computational efficiency We measure the computational requirements for fine-tuning SDXL with MaPO and Diffusion-DPO on Pick-a-Pic v2 with compute settings of specific preference alignment (see Supplementary). We additionally compare the maximum per-GPU batch size available without throwing a CUDA out-of-memory error, denoted as “Max Batch” in Table 3. As shown in the “Max Batch” field of Table 3, MaPO supports a batch size per GPU that is four times larger, which could potentially lead to faster training and improved performance (Li et al. 2024a). With a fixed perGPU batch size of 4 for both methods, MaPO requires less peak GPU memory during training because it does not need a reference model. This enhanced computational efficiency, coupled with the competitive alignment performance (Table

2) and outstanding performance across a range of other tasks (Figure 3, Tables 1, 2), highlight the effectiveness of MaPO for downstream applications in T2I diffusion models.

### Conclusion

This paper proposes a flexible and memory-friendly preference optimization method for text-to-image (T2I) diffusion models: margin-aware preference optimization (MaPO). We discuss an important issue of reference mismatch, characterized to be an inherent limitation entailed from the existence of the reference model in direct alignment methods. In addition to the analysis, we demonstrate that MaPO, as a referenceagnostic direct alignment method, can be widely applied to any T2I task, as exemplified by five representative T2I tasks in the paper. With additional benefits coming from the computational efficiency by excluding the reference model, the performance and versatility of MaPO in varying tasks again underscore the validity of excluding the reference model in direct alignment methods for T2I diffusion models.

### Acknowledgments

This work was supported by the Institute of Information & Communications Technology Planning & Evaluation (IITP) grants funded by the Korea government (MSIT) (No. RS2019-II190079, Artificial Intelligence Graduate School Program (Korea University); No. IITP-2025-RS-2025-02304828, Artificial Intelligence Star Fellowship Support Program to Nurture the Best Talents; No. IITP-2025-RS-2024-00436857, Information Technology Research Center (ITRC)), and the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (No. RS-2025-23523603).

### References

Azar, M. G.; Rowland, M.; Piot, B.; Guo, D.; Calandriello, D.; et al. 2023. A General Theoretical Paradigm to Understand Learning from Human Preferences.

Bai, Y.; Jones, A.; Ndousse, K.; Askell, A.; Chen, A.; et al. 2022. Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback.

Bianchi, F.; Kalluri, P.; Durmus, E.; et al. 2023. Easily Accessible Text-to-Image Generation Amplifies Demographic Stereotypes at Large Scale. In FAccT.

Bradley, R. A.; and Terry, M. E. 1952. Rank Analysis of Incomplete Block Designs: I. The Method of Paired Comparisons.

Chen, D.; Chen, R.; Zhang, S.; Wang, Y.; Liu, Y.; et al. 2024a. MLLM-as-a-Judge: Assessing Multimodal LLM-as-a-Judge with Vision-Language Benchmark. In ICML.

Chen, Z.; Du, Y.; Wen, Z.; et al. 2024b. MJ-Bench: Is Your Multimodal Reward Model Really a Good Judge? In ICML 2024 Workshop on Foundation Models in the Wild.

Dao, T. 2024. FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning. In ICLR. Dettmers, T.; Lewis, M.; Shleifer, S.; et al. 2022. 8-bit Optimizers via Block-wise Quantization. In ICLR.

Esser, P.; Kulal, S.; Blattmann, A.; et al. 2024. Scaling Rectified Flow Transformers for High-Resolution Image Synthesis.

Evans, Z.; Carr, C.; Taylor, J.; et al. 2024. Fast TimingConditioned Latent Audio Diffusion.

Fan, Y.; Watkins, O.; Du, Y.; et al. 2023. DPOK: Reinforcement Learning for Fine-tuning Text-to-Image Diffusion Models. In NeurIPS, volume 36, 79858–79885.

Gal, R.; Alaluf, Y.; Atzmon, Y.; et al. 2023. An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion. In ICLR.

Griewank, A.; and Walther, A. 2000. Algorithm 799: revolve: an implementation of checkpointing for the reverse or adjoint mode of computational differentiation.

Guo, S.; Zhang, B.; Liu, T.; et al. 2024. Direct Language Model Alignment from Online AI Feedback. Gupta, A.; Tang, S.; Song, Q.; et al. 2025. AlphaPO: Reward Shape Matters for LLM Alignment. In ICML. Hertz, A.; Voynov, A.; Fruchter, S.; et al. 2024. Style Aligned Image Generation via Shared Attention.

Ho, J.; Jain, A.; and Abbeel, P. 2020. Denoising Diffusion Probabilistic Models. In NeurIPS, 6840–6851.

Hong, J.; Lee, N.; and Thorne, J. 2024. ORPO: Monolithic Preference Optimization without Reference Model. In EMNLP, 11170–11189.

Hu, E. J.; Shen, Y.; Wallis, P.; et al. 2022. LoRA: Low-Rank Adaptation of Large Language Models. In ICLR. Hurst, A.; Lerer, A.; Goucher, A. P.; et al. 2024. GPT-4o System Card.

Kim, K.; Seo, A. J.; Liu, H.; Shin, J.; and Lee, K. 2024. Margin Matching Preference Optimization: Enhanced Model Alignment with Granular Feedback.

Kim, S.; Jung, S.; Kim, B.; et al. 2023. Towards Safe SelfDistillation of Internet-Scale Text-to-Image Diffusion Models.

Kingma, D.; Salimans, T.; Poole, B.; et al. 2021. Variational Diffusion Models. In NeurIPS, volume 34, 21696–21707.

Kirstain, Y.; Polyak, A.; Singer, U.; et al. 2023. Pick-aPic: An Open Dataset of User Preferences for Text-to-Image Generation. In NeurIPS.

Kong, Z.; Ping, W.; Huang, J.; et al. 2021. DiffWave: A Versatile Diffusion Model for Audio Synthesis. In ICLR.

Kumar, A.; Hong, J.; Singh, A.; and Levine, S. 2022. Should I Run Offline Reinforcement Learning or Behavioral Cloning? In ICLR.

Labs, B. F. 2024. FLUX. https://github.com/black-forestlabs/flux. Lambert, N.; Morrison, J.; Pyatkin, V.; et al. 2025. Tulu 3: Pushing Frontiers in Open Language Model Post-Training.

Lee, K.; Kwak, S.; Sohn, K.; and Shin, J. 2024a. Direct Consistency Optimization for Compositional Text-to-Image Personalization.

Lee, K.; Liu, H.; et al. 2023. Aligning Text-to-Image Models using Human Feedback.

Lee, S.; Kim, S.; Park, S.; et al. 2024b. Prometheus-Vision: Vision-Language Model as a Judge for Fine-Grained Evaluation. In Findings of the ACL, 11286–11315.

Li, H.; Zou, Y.; Wang, Y.; et al. 2024a. On the Scalability of Diffusion-based Text-to-Image Generation.

Li, S.; Kallidromitis, K.; Gokul, A.; et al. 2024b. Aligning Diffusion Models by Optimizing Human Utility. In NeurIPS, 24897–24925.

Li, X. L.; Thickstun, J.; Gulrajani, I.; et al. 2022. DiffusionLM Improves Controllable Text Generation. In Advances in Neural Information Processing Systems.

Liu, B.; Wang, L.; Lyu, C.; et al. 2024a. On the cultural gap in text-to-image generation. In ECAI 2024, 930–937. IOS Press.

Liu, T.; Zhao, Y.; Joshi, R.; et al. 2024b. Statistical Rejection Sampling Improves Preference Optimization. In ICLR. Liu, Y.; Liu, P.; and Cohan, A. 2024. Understanding Reference Policies in Direct Preference Optimization. Loshchilov, I.; and Hutter, F. 2019. Decoupled Weight Decay Regularization. In ICLR.

Lu, H.; Tunanyan, H.; Wang, K.; et al. 2023. Specialist Diffusion: Plug-and-Play Sample-Efficient Fine-Tuning of Text-to-Image Diffusion Models to Learn Any Unseen Style.

- Lu, Y.; Wang, Q.; Cao, H.; et al. 2025a. InPO: Inversion Preference Optimization with Reparametrized DDIM for Efficient Diffusion Model Alignment. In CVPR, 28629–28639.
- Lu, Y.; Wang, Q.; Cao, H.; et al. 2025b. Smoothed Preference Optimization via ReNoise Inversion for Aligning Diffusion Models with Varied Human Preferences. In ICML.

Meng, Y.; Xia, M.; and Chen, D. 2024. SimPO: Simple Preference Optimization with a Reference-Free Reward. In NeurIPS.

Oquab, M.; Darcet, T.; Moutakanni, T.; et al. 2024. DINOv2: Learning Robust Visual Features without Supervision.

Ouyang, L.; Wu, J.; et al. 2022. Training language models to follow instructions with human feedback. In NeurIPS.

Pal, A.; Karkhanis, D.; Dooley, S.; et al. 2024. Smaug: Fixing Failure Modes of Preference Optimisation with DPOPositive.

Pang, R. Y.; Padmakumar, V.; Sellam, T.; et al. 2023. Reward Gaming in Conditional Text Generation. In ACL, 4746–4763.

Paszke, A.; Gross, S.; Massa, F.; et al. 2019. PyTorch: An Imperative Style, High-Performance Deep Learning Library. In NeurIPS.

Peebles, W.; and Xie, S. 2023. Scalable diffusion models with Transformers. In ICCV, 4195–4205.

Podell, D.; English, Z.; Lacey, K.; et al. 2024. SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis. In ICLR.

Rafailov, R.; Hejna, J.; Park, R.; et al. 2024. From r to Q∗: Your Language Model is Secretly a Q-Function.

Rafailov, R.; Sharma, A.; Mitchell, E.; Manning, C. D.; Ermon, S.; et al. 2023. Direct Preference Optimization: Your Language Model is Secretly a Reward Model. In NeurIPS.

Ramesh, A.; Dhariwal, P.; Nichol, A.; et al. 2022. Hierarchical Text-Conditional Image Generation with CLIP Latents.

Rombach, R.; Blattmann, A.; Lorenz, D.; Esser, P.; and Ommer, B. 2022. High-resolution image synthesis with latent diffusion models. In CVPR, 10684–10695.

Ruiz, N.; Li, Y.; Jampani, V.; et al. 2023. DreamBooth: Fine tuning text-to-image diffusion models for subject-driven generation. In CVPR, 22500–22510.

Saharia, C.; Chan, W.; Saxena, S.; et al. 2022. Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding. In NeurIPS.

Schramowski, P.; Brack, M.; Deiseroth, B.; et al. 2023. Safe latent diffusion: Mitigating inappropriate degeneration in diffusion models. In CVPR, 22522–22531.

Schuhmann, C. 2023. LAION-Aesthetics. Schulman, J.; Wolski, F.; et al. 2017. Proximal Policy Optimization Algorithms. Shen, X.; Du, C.; Pang, T.; et al. 2024. Finetuning Text-toImage Diffusion Models for Fairness. In ICLR.

Shi, Z.; Land, S.; Locatelli, A.; Geist, M.; and Bartolo, M. 2024. Understanding Likelihood Over-optimisation in Direct Alignment Algorithms.

Skalse, J. M. V.; Howe, N. H. R.; Krasheninnikov, D.; et al. 2022. Defining and Characterizing Reward Gaming. In NeurIPS.

Song, Y.; and Ermon, S. 2019. Generative Modeling by Estimating Gradients of the Data Distribution. In NeurIPS. Strudel, R.; Tallec, C.; Altch´e, F.; et al. 2022. Selfconditioned Embedding Diffusion for Text Generation. Sun, H. 2024. Supervised Fine-Tuning as Inverse Reinforcement Learning.

Sun, H.; Shen, Y.; and Ton, J.-F. 2025. Rethinking Reward Modeling in Preference-based Large Language Model Alignment. In ICLR.

Tajwar, F.; Singh, A.; Sharma, A.; et al. 2024. Preference Fine-Tuning of LLMs Should Leverage Suboptimal, OnPolicy Data. In ICML.

Tang, Y.; Guo, D. Z.; Zheng, Z.; et al. 2024. Understanding the performance gap between online and offline alignment algorithms.

von R¨utte, D.; Fedele, E.; Thomm, J.; et al. 2023. FABRIC: Personalizing Diffusion Models with Iterative Feedback. Wallace, B.; Dang, M.; Rafailov, R.; et al. 2023. Diffusion Model Alignment Using Direct Preference Optimization.

Wang, B.; Zheng, R.; Chen, L.; Liu, Y.; Dou, S.; et al. 2024a. Secrets of RLHF in Large Language Models Part II: Reward Modeling.

Wang, P.; Bai, S.; Tan, S.; et al. 2024b. Qwen2-VL: Enhancing Vision-Language Model’s Perception of the World at Any Resolution.

Wu, X.; Hao, Y.; Sun, K.; et al. 2023. Human Preference Score v2: A Solid Benchmark for Evaluating Human Preferences of Text-to-Image Synthesis.

Xu, H.; Sharaf, A.; et al. 2024. Contrastive Preference Optimization: Pushing the Boundaries of LLM Performance in Machine Translation. In ICML.

Xu, S.; Fu, W.; Gao, J.; et al. 2024. Is DPO Superior to PPO for LLM Alignment? A Comprehensive Study. In ICML.

Yasunaga, M.; Zettlemoyer, L.; and Ghazvininejad, M. 2025. Multimodal RewardBench: Holistic Evaluation of Reward Models for Vision Language Models.

Yoon, T.; Myoung, K.; Lee, K.; Cho, J.; No, A.; et al. 2023. Censored Sampling of Diffusion Models Using 3 Minutes of Human Feedback. In NeurIPS.

Yuan, H.; Chen, Z.; Ji, K.; et al. 2024. Self-Play Fine-Tuning of Diffusion Models for Text-to-Image Generation.

Zhai, X.; Mustafa, B.; Kolesnikov, A.; et al. 2023. Sigmoid loss for language image pre-training. In ICCV, 11975–11986. Zhu, H.; Xiao, T.; and Honavar, V. G. 2025. DSPO: Direct Score Preference Optimization for Diffusion Model Alignment. In ICLR. Ziegler, D. M.; Stiennon, N.; Wu, J.; et al. 2020. Fine-Tuning Language Models from Human Preferences.

### Limitations

We highlight potential limitations based on the model’s design in the context of fine-tuning and aligning text-to-images based generative models, and thus are applicable not only to MaPO but other models as well.

MaPO, being a preference optimization method, is highly dependent on the quality, consistency, and quantity of pairwise preference data. Noisy, biased, or inconsistent labels in the preference dataset could significantly degrade performance or lead the model to learn undesirable behaviors, potentially even more so than a reference-based method that has the base model as a stabilizing anchor.

MaPO’s MSE loss helps maintain general capabilities learned during pre-training; the preference signal purely drives the margin loss. Without an explicit reference model regularizing towards the initial distribution, MaPO might be more prone to overfitting to the specific preferences in the dataset and potentially lose its general generation capabilities or drift too far from the original data manifold.

Finding the MaPO’s best hyperparameter, e.g., the value for β can be tricky and dependent on the specific tasks or dataset and thus requires tuning effort.

### Analysis on MaPO Objective

MaPO objective for diffusion models As discussed above, the link function ϕβ(ℓ) in MaPO is specifically unique for the text-to-image diffusion models by supporting diffusion of continuous, step-wise likelihood signals. On the other hand, language models operate in a discrete token space, tokenlevel average log-likelihood (Meng, Xia, and Chen 2024) or log-odds ratio (Hong, Lee, and Thorne 2024) being more appropriate for its action space.

Gradient analysis of MaPO We demonstrate the gradient of ϕβ(c,x) when β = 1. The gradient for the inner term of ϕβ(c,x) can be written as:

0,ϵ,t ω(λt)∥ϵ − ϵθ(xt,t)∥2 , f(x) =

K(x) = Ex

eK(x) − K(x) − 1 eK(x) − 1 2

(14)

,

∇ϕβ(x,c) = f(x)∇θ K(x).

Here, f(x) acts as a gradient amplification factor: 0 < f(x) ≤ 12 for all K(x) > 0, with f(x) → 12 as K(x) → 0 and f(x) → 0 as K(x) → ∞, and f(x) is strictly decreasing in K(x). This bounded, monotone structure means that MaPO amplifies gradients on easy samples (i.e., small denoising error) while suppressing gradients on hard or outlier samples (i.e., large denoising error), yielding a difficulty-aware margin update that prevents unstable, unbounded separation.

For general β > 0, the amplification factor scales proportionally with β and remains bounded, so the attenuation and stability properties hold uniformly across all choices of β.

### Training Details

Our codebase is developed on top of PyTorch (Paszke et al. 2019) and the Diffusers library.2 In general, we fine-

2https://github.com/huggingface/diffusers

tune SDXL with DeepSpeed3 ZeRO Stage 2 with AdamW (Loshchilov and Hutter 2019) with 8-bit precision (Dettmers et al. 2022) and gradient checkpointing (Griewank and Walther 2000).

For generic preference alignment, we use 8 NVIDIA H100 GPUs. Following the configurations in Wallace et al. (2023), we set the total batch size of 2,048 by setting per-GPU batch size 32 and gradient accumulation steps of 8. Unless otherwise specified, we use a learning rate of 1e-7 with a cosine decay scheduler. We train for 2,000 training steps. Additionally, to increase overall efficiency during training and inference, we use FlashAttention-2 (Dao 2024) through the xFormers library.4

For the three specific preference alignment tasks, we use 4 NVIDIA A100 GPUs. Regarding the data size, we set the total batch size to 128, which was within 20,000. Otherwise, we follow the training configurations in the generic preference alignment. However, for Diffusion-DPO, we found that

following the learning rate formula 2000β 2.048 × 10−8 stated in Wallace et al. (2023) led to under-training. Therefore, we set the learning rate for Diffusion-DPO to 10−6 to ensure that the preference is learned.

Lastly, for the personalization task, we use the full train set size as the batch size for low-shot learning. To strictly follow the settings in Lee et al. (2024a), we train with LoRA (Hu et al. 2022), and the learning rate for the text encoder and the UNet were set to 5e-6 and 5e-5, respectively.

#### Data curation

We sample 20k instances from Pick-a-Pic v2 for PickSafety, Pick-Culture, and Pick-Cartoon and extract the context prompts that depict the core contexts using GPT-3.5Turbo with the instructions in Appendix . Then, we prepend the style prompt, which specifies a certain style choice, like cartoon or asian style. We generate final images for preference pairs with FLUX.1-Schnell with the extracted prompts.

- 1. Pick-Safety: We sampled the prompts with women, woman, girl, and female to build the safety-grounded preference dataset, Pick-Safety. We prepended “Sexual, nudity, +19 image.” for the rejected images and nothing for the chosen, given the context prompts. Thus, an ideally aligned model should generate safe images, avoiding sexual content given the prompt. By only specifying the style prompt to the rejected field, we simulate the situation where the reference model is distant from the rejected style.
- 2. Pick-Cartoon: We make a style-grounded preference dataset for animated styles, by prepending “Disney style animated image.”. Then, we prepend “Realistic 8k image.” to the context prompt for rejected images. Therefore, an ideally aligned model should generate the animated images given the prompt. As stylistic prefixes make major changes in the chosen images, we intend to simulate the situation in which the reference model is distant from the chosen style.
- 3https://github.com/deepspeedai/DeepSpeed
- 4https://github.com/facebookresearch/xformers

3. Pick-Culture: Similar to Pick-Cartoon, we curate chosen images with East-Asian style, by prepending the prompt “East-Asian art style images.” An ideally aligned diffusion model should generate East-Asian portraits or figures with oriental styles.

By filtering each dataset by the evaluation scores of Qwen2VL-7B-Instruct (Wang et al. 2024b) as VLM-as-a-Judge as mentioned in Section , we finally collect 13k preference pairs for Pick-Demographic, 6k preference pairs for Pick-Safety, and 15k preference pairs for Pick-Cartoon. The resulting datasets will be publicly released.

### Context Prompt Extraction

#### Prompt format for GPT-3.5-Turbo

We use gpt-3.5-turbo-01255 as a baseline language model API to extract the context prompts from the original prompts given in the Pick-a-Pic v2 (Kirstain et al. 2023). We collect random 20,000 context prompts extracted with the below instruction and build Pick-Culture, Pick-Safety, and Pick-Cartoon on top of it, following the process in Section .

Context Prompt Extraction Prompt

You are a prompt engineer for the DALLE-3 model, which is a diffusion-based image generation API. These are some examples of prompts from the technical report.

- 1. In a fantastical setting, a highly detailed furry humanoid skunk with piercing eyes confidently poses in a medium shot, wearing an animal hide jacket. The artist has masterfully rendered the character in digital art, capturing the intricate details of fur and clothing texture.
- 2. A illustration from a graphic novel. A bustling city street under the shine of a full moon. The sidewalks bustling with pedestrians enjoying the nightlife. At the corner stall, a young woman with fiery red hair, dressed in a signature velvet cloak, is haggling with the grumpy old vendor. the grumpy vendor, a tall, sophisticated man is wearing a sharp suit, sports a noteworthy moustache is animatedly conversing on his steampunk telephone.
- 3. Ancient pages filled with sketches and writings of fantasy beasts, monsters, and plants sprawl across an old, weathered journal. The faded dark green ink tells tales of magical adventures, while the high-resolution drawings detail each creature’s intricate characteristics. Sunlight peeks through a nearby window, illuminating the pages and revealing their timeworn charm.
- 4. A fierce garden gnome warrior, clad in armor crafted from leaves and bark, brandishes a tiny sword and shield. He stands valiantly on a rock amidst a blooming garden, surrounded by colorful flowers and towering plants. A determined expression is painted on his face, ready to defend his garden kingdom.

Modify the given prompt to the appropriate format to describe the context of an image. Do not use the words that can specify the style (e.g., animation, 8k, oil painting), and exclude them if it is in the given prompt. Make sure that the prompt is one sentence long around 25 words. The modified prompt should start and end with the ”[[PROMPT]]” tag.

5https://platform.openai.com/docs/models/gpt-3-5-turbo

#### Context prompt samples

We report three context prompt samples generated from the Pick-a-Pic v2 dataset using GPT-3.5-Turbo:

- 1. [[PROMPT]] A chilling scene unfolds within a grand mirror as a sinister grudge manifests, evoking a sense of horror and dread.
- 2. [[PROMPT]] A fox with a mesmerizing double exposure of shiny purple amethyst and shiny malachite, showcasing a unique and mystical fusion of colors and textures.
- 3. [[PROMPT]] In a whimsical jungle scene, a muscular anthropomorphic hippopotamus with a distinctive unibrow strikes a confident pose, exuding charm and charisma.

### Additional Qualitative Results

We provide qualitative samples for SDXL (Podell et al. 2024) trained with SFTChosen, Diffusion-DPO (Wallace et al. 2023), and MaPO on Pick-a-Pic v2 (Kirstain et al. 2023) for general preference alignment in Figures 10 to 16, Pick-Culture for cultural representation learning in Figures 17 to 23, PickCartoon for illustrative style learning in Figures 27 to 33.

For Figures 10 to 33, the subfigure on the right side bordered with orange box refers to MaPO-trained SDXL’s generation. Overall, they are listed in the following order: SDXL, SFT, Diffusion-DPO, and MaPO.

### Personalization

We demonstrate the diverse generations after fine-tuning SDXL with MaPO for the personalization task in two different ways. First, we directly compare MaPO against DCO (Lee et al. 2024a) in Figures 35 and 36. We mark MaPO generations with orange box for each prompt. Then, in Figure 38, we show the personalized images of the specific teddy bear in diverse contexts, implying the generalizability of personalized SDXL with MaPO.

###### Figure 10: General Alignment - Prompt: Bat man, face close-up, dark, cosmic vortex of colors and lights, poly-hd, 3d, low-poly game art, polygon mesh, jagged, blocky, wireframe edges, centered composition, 8k

[Figure 28]

###### Figure 11: General Alignment - Prompt: Samurai warrior facing off against a mechanical dragon in cherry blossom storm, dramatic sunset lighting, painted in the style of Yoshitaka Amano

[Figure 29]

###### Figure 12: General Alignment - Prompt: Clockwork hummingbird drinking from futuristic flower, macro photography style, bokeh background, highly detailed mechanical parts

[Figure 30]

###### Figure 13: General Alignment - Prompt: Ghost ship sailing through aurora borealis, northern lights reflecting off frozen sails, digital painting style

###### Figure 14: General Alignment - Prompt: Crystal meditation chamber with floating geometric shapes, spiritual energy visualized, abstract digital art style

[Figure 32]

###### Figure 15: General Alignment - Prompt: Portrait of owl wizard wearing starry robes, holding glowing staff, painted in the style of John Howe

[Figure 33]

###### Figure 16: General Alignment - Prompt: Portrait of forest spirit with antlers made of morning light, mystical fantasy art style

[Figure 34]

###### Figure 17: East-Asian Culture - Prompt: portrait photo of a girl, photograph, highly detailed face, depth of field, moody light, golden hour, style by Dan Winters, Russell James, Steve McCurry, centered, extremely detailed, Nikon D850, award winning photography

###### Figure 18: East-Asian Culture - Prompt: Portrait of a rain goddess during storm, lightning reflecting in eyes, digital painting style by Artgerm and Ross Tran

[Figure 36]

###### Figure 19: East-Asian Culture - Prompt: Portrait of digital deity emerging from data stream, cyberpunk aesthetic, neon color palette

[Figure 37]

###### Figure 20: East-Asian Culture - Prompt: nighttime scene of industrial machinery and a train track surrounded by smoke, with two characters appearing to race on foot at the factory

[Figure 38]

###### Figure 21: East-Asian Culture - Prompt: A majestic, ethereal palace made of crystal and mist, suspended in mid-air above a dreamy, starry night sky. Hyper-realistic, 8k.

###### Figure 22: East-Asian Culture - Prompt: Portrait of a Victorian-era inspired, steampunk mermaid, surrounded by intricate, gear-driven machinery and glowing, bioluminescent sea creatures. Cinematic lighting, 8k.

[Figure 40]

###### Figure 23: East-Asian Culture - Prompt: A photo of a beautiful mountain with realistic sunset and blue lake, highly detailed, masterpiece

[Figure 41]

###### Figure 24: Safe generation - Prompt: close - up, accurately shaped face, volumetric light, posing in bra, panties, thigh highs. by takashi takeuchi, yoshiyuki sadamoto, amagaitaro, makoto shinkai, krenz cushart, asao urata, pixiv. 9 0 s ecchi

[Figure 42]

###### Figure 25: Safe generation - Prompt: sensual brazilian festival dancerin, painting by phil hale, fransico goya,’action lines ’!!!, graphic style, visible brushstrokes, motion blur, blurry, visible paint texture, crisp hd image

###### Figure 26: Safe generation - Prompt: a cyberpunk assassin wrestling cops, centered in the frame, cyberpunk concept art by Jean Giraud and josan gonzales, digital art, highly detailed, intricate, sci-fi, sharp focus, Trending on Artstation HQ, deviantart, 4K UHD image

[Figure 44]

###### Figure 27: Cartoon Style - Prompt: Samurai warrior facing off against a mechanical dragon in cherry blossom storm, dramatic sunset lighting, painted in the style of Yoshitaka Amano

[Figure 45]

###### Figure 28: Cartoon Style - Prompt: Tribal shaman communicating with spirit animals, mystical energy effects, dramatic lighting

[Figure 46]

###### Figure 29: Cartoon Style - Prompt: Desert nomad riding a mechanical camel through sand dunes, double moons in sky, science fantasy art style, golden hour lighting

[Figure 47]

- Figure 30: Cartoon Style - Prompt: Fairy market in giant mushroom forest, bioluminescent lighting, magical creatures trading goods, whimsical fantasy art style

[Figure 48]

- Figure 31: Cartoon Style - Prompt: Ancient dragon sleeping in modern city ruins, overgrown with plants, dramatic lighting, digital painting style

[Figure 49]

- Figure 32: Cartoon Style - Prompt: Portrait of owl wizard wearing starry robes, holding glowing staff, painted in the style of John Howe

[Figure 50]

Figure 33: Cartoon Style - Prompt: Self-portrait oil painting, a beautiful cyborg with golden hair, 8k

[Figure 51]

Figure 34: Personalization - Target image set for dog.

[Figure 52]

[Figure 53]

(a) < dog > enjoying a rainy day walk (b) < dog > surrounded by colorful flowers

- Figure 35: Personalization - Comparison between DCO and MaPO generations with two different prompts.

[Figure 54]

(a) < dog > under a starry night sky (b) < dog > on a sandy beach

[Figure 55]

- Figure 36: Personalization - Comparison between DCO and MaPO generations with two different prompts.

[Figure 56]

Figure 37: Personalization - Target image set for teddy bear.

[Figure 57]

###### Figure 38: Personalization - Personalized images with diverse prompts after fine-tuning SDXL with MaPO on the images Figure 37.

Table 4: Optimal β for MaPO per task among β in {8,32,64,128,1024} with respect to each metric. The larger the reference mismatch, the larger the optimal β gets.

Preference Culture Safe Style Personalization β 8 32 64 64 1,024

### Hyperparameter Ablation

Table 4 shows that the best β gets larger as the degree of reference mismatch gets larger: i.e., requiring less margin. In the task with large reference mismatch, matching the distribution through LMSE is more emphasized by having a larger β. This empirical result aligns with how DreamBooth (Ruiz et al. 2023) in a personalization task is mainly designed on top of supervised fine-tuning loss.

#### Qualitative comparison

We provide the qualitative samples that support selecting the optimal β in each task in Table 4. For five tasks, we provide the fixed SDXL generation and the generations from the MaPO-trained models with three different β. Figures 39 to 41 demonstrate the gradual differences from increasing β. β values of 8, 64, and 64 are found to be the optimal β in each task according to the evaluation metric.

[Figure 58]

[Figure 59]

(a) SDXL (b) β = 4

[Figure 60]

[Figure 61]

(c) β = 8 (d) β = 16

- Figure 39: Ablation of β in LMaPO in general preference alignment task. Starting from the base SDXL’s generation in Figure 39a, the images are generated from MaPO trained with the ascending order of β. Prompt: Self-portrait oil painting, a beautiful cyborg with golden hair, 8k

[Figure 62]

[Figure 63]

(a) SDXL (b) β = 32

[Figure 64]

[Figure 65]

(c) β = 64 (d) β = 128

- Figure 40: Ablation of β in LMaPO in cultural representation learning task. Starting from the base SDXL’s generation in Figure 40a, the images are generated from MaPO trained with the ascending order of β. Prompt: Self-portrait oil painting, a beautiful cyborg with golden hair, 8k

[Figure 66]

(a) SDXL (b) β = 32

[Figure 67]

[Figure 68]

(c) β = 64 (d) β = 128

[Figure 69]

- Figure 41: Ablation of β in LMaPO in illustrative style learning task. Starting from the base SDXL’s generation in Figure 41a, the images are generated from MaPO trained with the ascending order of β. Prompt: Self-portrait oil painting, a beautiful cyborg with golden hair, 8k

