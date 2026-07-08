## Beyond Log Likelihood: Probability-Based Objectives for Supervised Fine-Tuning across the Model Capability Continuum

Gaotang Li*1 Ruizhong Qiu*1 Xiusi Chen*1 Heng Ji1 Hanghang Tong1

# arXiv:2510.00526v3[cs.CL]22May2026

### Abstract

Supervised fine-tuning (SFT) is the standard approach for post-training large language models (LLMs), yet it often shows limited generalization. We trace this limitation to its default training objective: negative log likelihood (NLL). While NLL is classically optimal when training from scratch, post-training operates in a different paradigm and could violate its optimality assumptions, where models already encode task-relevant priors and supervision can be long and noisy. In this work, we systematically study various probability-based objectives and characterize when and why different objectives succeed or fail under varying conditions. Through comprehensive experiments and extensive ablation studies across 8 model backbones, 27 benchmarks, and 7 domains, we uncover a critical dimension that governs objective behavior: the model-capability continuum. Near the model-strong end, prior-leaning objectives that downweight low-probability tokens (e.g., −p, −p10, thresholded variants) consistently outperform NLL; toward the model-weak end, NLL dominates; in between, no single objective prevails. Our theoretical analysis further elucidates how objectives trade places across the continuum, providing a principled foundation for adapting objectives to model capability. The code is available at https://github.com/ GaotangLi/Beyond-Log-Likelihood.

### 1. Introduction

Supervised fine-tuning (SFT) has become a standard approach for post-training large language models (LLMs),

*Equal contribution 1University of Illinois Urbana-Champaign. Correspondence to: Gaotang Li <gaotang3@illinois.edu>, Hanghang Tong <htong@illinois.edu>.

Proceedings of the 43rd International Conference on Machine Learning, Seoul, South Korea. PMLR 306, 2026. Copyright 2026 by the author(s).

| |17.00| | | | |
|---|---|---|---|---|---|
| | | |32.7|5 (+15.75)| |
| | | |31.50 (+|14.50)| |
| | | | | | |
| | | | | | |

- α = 0

NLL: −logp

- α = 1

1 − p

α = 10

(1 − p10)/10

16 20 24 28 32 34

Avg. accuracy (%)

Figure 1. Motivating model-strong SFT result on math reasoning. For the objective family fα(p) = (1 − pα)/α, where α → 0 recovers NLL (− log p). Prior-leaning objectives with α = 1 and α = 10 substantially improve average accuracy over NLL.

widely used to elicit and strengthen their capabilities (Zhang et al., 2026b; Chung et al., 2024). Despite its popularity, many existing studies find that SFT often exhibits limited generalization (Ouyang et al., 2022; Chu et al., 2025). Nevertheless, this limitation may not arise from the imitation learning paradigm itself. Instead, we find that it may stem from its default training objective: negative log likelihood (NLL, −log p). As a motivating case study, we generalize NLL into a parametrized family of learning objectives of the form fα(p) := −p

α−1

α , which includes NLL as a special case (fα(p) → −log p as α → 0). We surprisingly find that other objectives significantly outperform NLL on some tasks, as shown in Fig. 1.

This unexpected observation motivates us to fundamentally revisit the training objective of SFT in its simplest form. While NLL has been shown to be optimal in classical learning theory when training from scratch on smallscale classification tasks (Cox, 1958; Zhang, 2004; Bartlett et al., 2006), LLM post-training operates in a fundamentally different paradigm that could degrade the optimality of NLL. Post-training begins with a pretrained model that already encodes task-relevant priors, and typically involves long chain-of-thought supervision spanning thousands of tokens that may be noisy. Requiring the pretrained model to replicate every token verbatim can hinder generalization.

To this end, we conduct a comprehensive study to demystify which scenarios suit NLL and which suit other objectives, rather than advocating a single “one-size-fits-all” loss. Our study uncovers a critical dimension that governs

Math: Extensive Pretraining Tokens

Medical: General World Knowledge

Figfont Puzzle: No Pretraining Data

[Figure 1]

Alice has 3 apples, buys 2 more. How many now?

Patient has fever + cough. What’s the likely diagnosis?

What word is this?

[Figure 2]

Model-Strong (MS)

[Figure 3]

[Figure 4]

Model-Weak (MW)

Model Intermediate

Prior-leaning Objective Prior-averse Objective

- -p, -p10, threshold
- -log p, … Learning Objective -log p, -p0.1

Figure 2. The model capability continuum of SFT objectives in Post-Training. At the model-strong (MS) end, where base models already encode extensive priors (e.g., Llama 3 reports 25% math pretraining tokens (Grattafiori et al., 2024)), prior-leaning objectives that downweight low-probability tokens (e.g., −p, −p10, or thresholded variants) consistently outperform NLL by up to 16%. At the model-weak (MW) end, where no useful priors exist (e.g., no figfont puzzles in pretraining data), the standard NLL dominates. In the model-intermediate (MI) region (e.g., medical reasoning, where models rely on partial world knowledge), the gap between objectives narrows and no single choice consistently prevails. This continuum highlights how SFT objective effectiveness depends critically on base-model capability. Our results suggest that no single fixed SFT objective is universally optimal across settings.

the behavior of different objectives: the model-capability continuum. This continuum reflects the strength of prior signals inherited from pretraining: some domains (e.g., math with abundant pretraining tokens) align well with the model’s priors, while others (e.g., novel puzzles with no pretraining exposure) do not, as illustrated in Fig. 2. Accordingly, the effectiveness of a learning objective depends on prior strength: prior-leaning objectives excel when priors are reliable, whereas prior-averse ones remain necessary when priors are weak.

and the model capability interact.

### 2. Related Works

Improving SFT from RL perspective. Motivated by the success of reinforcement learning in reasoning tasks, a growing body of work reinterprets and improves SFT through an RL lens. Wang et al. (2026) cast SFT and DPO as instances of implicit reward learning, suggesting that smaller learning rates and alternative divergencebased objectives can improve performance. Qin & Springenberg (2025) further integrate importance sampling into SFT, while Zhu et al. (2026) introduces a PPO-style clipped to constrain policy drift. Closest to our setting, Wu et al. (2026) propose uniformly reweighting gradient coefficients, which is essentially equivalent to our −p objective. Collectively, these RL-inspired approaches can be interpreted as implementing prior-leaning updates within our framework. While prior work demonstrates their effectiveness in certain domains, our model-capability continuum view shows that such updates can fail in others, highlighting that no single loss is universally optimal across settings.

We validate this perspective through extensive experiments spanning 8 model backbones, 27 benchmarks, and 7 domains. Our results reveal a clear continuum in how objectives behave: at the model-strong end, where base models already provide reliable priors, probability-based objectives that downweight low-probability tokens (e.g., −p, −p10, or thresholded variants) consistently outperform NLL. At the model-weak end, where priors are misaligned with the data, NLL remains dominant by forcing the model to learn broadly from all tokens. In the intermediate region, the gap narrows and no single objective prevails. Further empirical analyses show that convexity and concavity of the learning objective, as a proxy for the degree to which model priors are respected, has opposite effects across the continuum. Likelihood estimation on the training set exhibits the same inversion.

Other SFT loss functions. Beyond RL-motivated objectives, a number of alternative losses to NLL have been explored in supervised learning, including mean squared error, focal loss (Lin et al., 2017), and Huber loss (Huber, 1992). These distribution-based losses can be naturally understood through our framework. More recent proposals, such as entropic distribution matching (Li et al., 2025), introduce additional regularization terms and can also be interpreted within our framework as prior-leaning objectives. Instead of adding objective sophistication or targeting specific applications, our work uncovers a modelcapability continuum through simple probability-based objectives across diverse settings. Additional discussion of

To elucidate these findings, we provide theoretical underpinnings that characterize when and why different objectives outperform others. We characterize a sufficient condition showing that a more prior-leaning (e.g., −p) achieve greater loss reduction than NLL in the model-strong end in gradient flow. The opposite holds in the model-weak end, where NLL achieves larger reductions. This theoretical characterization mirrors our empirical results and provides a principled explanation of how the objective form

related loss functions is provided in Appen. F.2, with a full version of the related work deferred to Appen. A.

Positioning of our work. We provide the first capabilitybased characterization of SFT objectives in LLM posttraining. Instead of promoting a single “best” loss, we establish a principled account of when and why objectives trade advantages across the model-capability continuum.

Prior-learning versus Prior-averse Objectives. The key distinction among these objectives lies in the form of their gradients with respect to the correct logit class, which governs the resulting learning dynamics.

Lemma 3.1 (Gradient Shape). Let f : [0,1]→R be differentiable and nonincreasing. Then the gradient of Eq. with respect to the logits at step t is

### 3. A Unified Categorization of SFT Objectives

Language Model Post-Training. We focus on the posttraining stage of large language models (LLMs). Let pθ denote a pretrained base model that has already undergone large-scale pretraining and accumulated extensive world knowledge. Such models typically produce predictions that are reasonably well-calibrated (Zhu et al., 2023; Xie et al., 2024), and their outputs encode task-relevant priors derived from pretraining corpora.

Standard Supervised Fine-Tuning. We consider supervised fine-tuning (SFT) on a dataset T of input-output pairs (x,y˜), where y˜ = (y1,...,yN) denotes the target sequence. The model defines token-level conditionals pθ(yt | y<t,x). At decoding step t, let zt ∈ RV denote the logits over the vocabulary, pt = softmax(zt), and pt,i = softmax(zt)i. For brevity, write y = yt, and denote by δi,y the Kronecker delta. In standard SFT, the training objective is to minimize the negative log likelihood, equivalently the cross-entropy loss, over the dataset:

Llog(p)(θ) = E(x,y˜)∼T − log pθ(˜y | x)

N

−log pθ(yt | y<t,x) . (1)

= E(x,y˜)∼T

t=1

A General Family of Probability-Based Objectives. We now extend beyond log likelihood by considering a broader family of objectives. For any differentiable and nonincreasing function f : [0,1] → R, we define

Lf(p)(θ) = E(x,y˜)∼T f pθ(˜y | x)

N

f pθ(yt | y<t,x) . (2)

= E(x,y˜)∼T

t=1

One useful general instance of f is given by

1 − pα α

fα(p) =

. (3)

As α → 0, it reduces to fα(p) → −log(p) (NLL). When α = 1, it yields the plain-p objective fα(p) = 1−p, which corresponds to maximizing the expected average prediction accuracy. More generally, the function is concave when α ≥ 1 and convex when 0 ≤ α ≤ 1.

∂ Lf ∂zt,i

= sf(pt,y) δi,y − pt,i , where sf(p) ≜ −f′(p)p ≥ 0, δiy = 1{i = y}.

In particular, for the correct class i = y,

∂ Lf ∂zt,y

= sf(pt,y)(1 − pt,y) = Wf(pt,y), Wf(p) ≜ −f′(p)p(1 − p).

Proposition 3.2 (Convex versus Concave Objectives). Let f ∈ C2[0,1] with f′(p) < 0 for all p ∈ (0,1). Define Wf(p) = −f′(p)p(1 − p). Then if f is concave, any maximizer of Wf lies in the interval [21,1]; if f is convex, any maximizer of Wf lies in the interval [0, 12].

In other words, convex objectives emphasize gradient contributions from low-probability tokens, while concave objectives emphasize high-probability tokens.

Logit Gradient Wf(p)

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | |log(p) (1 p0.5)/0.5<br><br>0.8| | | |
| | |(1 p )/0.8 p (1 p5)/5<br><br>| | | |
| | |(1 p10)/10 log(1 p)<br><br>| | | |
| | | | | | |
| | | | | | |

1.0

NormalizedValue

0.8

0.6

0.4

0.2

0.0

0.00 0.25 0.50 0.75 1.00

p

Figure 3. Logit-gradient weights Wf(p) for representative objectives. The marked peak of each curve shows the token-probability region where the correct-logit gradient is largest, distinguishing prior-averse objectives from prior-leaning objectives.

The weighting term Wf(p) determines how much learning signal each token contributes relative to the model’s prior belief. For the parametric family in Eq. 3, we have Wf(p) = pα(1 − p). As α → 0 (NLL), this reduces to Wf(p) → (1 − p), which strongly emphasizes lowprobability tokens. When α ≥ 1 (f(p) = 1 − p), the gradient signal from low-probability tokens quickly diminishes. For a special case f(p) = −log(1−p), we obtain Wf(p) =

p, which exhibits the opposite trend of −log(p) by emphasizing high-probability tokens. Fig. 3 visualizes these gradient shapes Wf(p) for different objectives: the dot marks where each logit-gradient weight is largest, and the dashed line at p = 0.5 serves as a reference point separating objectives that favor low- versus high-probability tokens. More formally, Prop. 3.2 shows that convex objectives (e.g., −log p) achieve their maximum within [0,0.5], thus prioritizing low-probability tokens (prior-averse); whereas concave objectives (e.g., −p2) peak within [0.5,1], thereby reinforcing already confident predictions (prior-leaning). This distinction illustrates how convexity modulates the degree to which an objective respects model priors. In particular, the family in Eq. 3 can be seen as providing a smooth transition between prior-averse and prior-leaning behavior. This leads to the following definition.

Definition 3.3 (Prior-leaning versus Prior-adverse Objectives). We classify objectives according to how Wf distributes its mass over p. We say the objective is:

- • Prior-leaning if the majority of gradient weight is concentrated on medium- to high-probability tokens (i.e., p above a threshold τ), thereby leveraging the model’s prior to refine already plausible predictions.
- • Prior-averse if the majority of gradient weight is concentrated on low-probability tokens (p below τ), pushing the model to learn from unlikely predictions.

This definition emphasizes that different objectives exploit the model’s prior in opposite ways. While the precise boundary between prior-leaning and prior-averse (e.g., the choice of threshold τ) is not unique and may depend on the task, some objectives exhibit clear contrasts (e.g., −log p versus −p), which form the primary focus of our study. To further probe their behavior, we also consider a hardthresholding variant:

LHT(I),f(p)(θ) = E(x,y˜)∼T f p(˜y | x) 1{p(˜y | x) ∈ I} ,

(4)

where HT(I) denotes restricting updates to tokens whose predicted probabilities fall within an interval I ⊆ [0,1]. This formulation is very useful for ablation, as it isolates the contribution of tokens in specific probability ranges.

The model capability continuum. Unlike traditional classification tasks, language model post-training spans a wide variety of domains that differ substantially in how well they are supported by pretraining. Consequently, not all tasks should be treated uniformly. We categorize tasks along a model-capability continuum, defined by the strength of the base model prior. A general categorization is shown in Fig. 2. Our classification relies on two complementary perspectives: (1) From the pretraining data side, tasks differ in

the portion of relevant data contained in the corpus. For example, the LLaMA-3 report indicates that ∼25% of its pretraining tokens are math-related, suggesting strong priors for mathematical reasoning (model-strong). By contrast, figfont puzzles fall entirely outside the pretraining corpus and thus represent model-weak tasks, while domains with partial coverage, such as medical reasoning, are considered intermediate. (2) From the model side, we use the mean predicted probability on the training set as a quantitative proxy for prior strength. This is analogous to how the widely-used benchmark lm eval evaluates base models by computing log-likelihood over answer tokens across many benchmarks (Gao et al., 2023). This measure aligns well with intuition: math tasks achieve high predicted likelihood of the training even before SFT (e.g., Qwen2.5Math-7B: 0.81, LLaMA-3.1-8B: 0.76), whereas medical reasoning lies in the middle (∼0.50), and figfont puzzles remain extremely low (∼0.01). Together, these perspectives motivate our continuum view and ground it in both qualitative and quantitative evidence. The details and the rationales about our classification are included in Appen. B.1.

At the model strong (MS) end, prior-leaning objectives can be leveraged to refine a small number of critical tokens by concentrating learning on mid- to high-probability tokens that are more likely to be correct. At the model weak (MW) end, prior-averse objectives are more suitable, as they encourage the model to improve predictions across all tokens. For models of intermediate capability (MI), both objectives may provide benefits, depending on the characteristics of the task and the base model.

### 4. Main Experiments

In this section, we empirically validate the proposed continuum view of SFT post-training and evaluate the performance of different probability-based objective functions.

#### 4.1. Experimental Setup

To empirically validate the continuum view, we conduct experiments across three representative domains: mathematical reasoning, medical reasoning, and textual puzzles, which serve as anchor settings for the model-capability continuum. As motivated in Sec. 3, these domains occupy different positions along the model-capability continuum. For the model-strong (MS) end, we use NuminaMath (LI et al., 2024) as training data. For the model-weak (MW) end, we generate synthetic figfont puzzles from Reasoning Gym (Stojanovski et al., 2025). For the intermediate (MI) region, we adopt m23k (Huang et al., 2026), a high-quality medical reasoning dataset. Additional statistics supporting this classification are provided in Appen. B.1. To test whether the same continuum behavior extends beyond these anchor domains, we also include fixed-domain ca-

Table 1. Main results in the Model Strong (MS) end. Both prior-leaning objectives −p and thresholded − log(p) consistently outperform the prior-averse − log(p) objective across models and datasets. Best results are in bold.

Models Math500 Minerva Math Olympiad Bench AIME24 AMC23 Avg.

LLaMA-3.1-8B Base 1.76 0.68 0.86 0.00 1.25 0.91

- -log(p) 17.59 5.84 3.04 0.21 5.78 6.49

- -log(p)1{p ≥ 0.2} 24.39 10.49 5.10 0.41 11.25 10.33

- -p 25.29 10.09 6.37 0.41 10.62 10.56 DeepSeekMath-7B

Base 5.70 2.89 1.51 0.00 2.34 2.49

- -log(p) 28.79 9.29 6.57 0.21 10.62 11.10

- -log(p)1{p ≥ 0.2} 40.38 19.38 13.98 0.62 18.91 18.65

- -p 39.55 20.14 13.99 1.24 20.62 19.11 Qwen2.5-Math-1.5B

Base 30.71 8.81 14.88 2.49 17.97 14.97

- -log(p) 42.52 12.71 12.09 0.62 17.03 17.00

- -log(p)1{p ≥ 0.2} 63.95 24.79 26.08 7.09 38.28 32.04

- -p 65.27 26.18 26.66 6.88 38.13 32.75 Qwen2.5-Math-7B Base 40.38 13.66 16.36 6.04 24.69 20.23
- -log(p) 51.90 18.88 17.37 2.70 22.50 22.67

- -log(p)1{p ≥ 0.2} 67.85 32.47 33.90 8.76 47.81 38.16

- -p 68.47 31.99 32.26 8.75 41.09 36.51

pability sweeps in general instruction tuning (Sec. 4.3 and Appen. C.2) and math (Appen. C.3), as well as additional domains that instantiate the MS and MW ends: coding and low-resource multilingual instruction tuning (Appen. C.3, C.4).

Across the main and appendix studies, our experiments cover a diverse set of advanced backbones, including LLaMA-3.1-3B, LLaMA-3.2-3B, LLaMA-3.1-8B, DeepSeekMath-7B, Qwen2.5-Math-1.5B, Qwen2.5-Math7B, Qwen2.5-1.5B/3B/7B/14B/32B, and Qwen2.5-Coder7B. We primarily compare the −p and −log p objectives, with one exception: on the MS end, we also evaluate a thresholded variant of −log p that excludes low-probability tokens. The evaluation datasets, training details, and further experimental configurations are provided in Appen. B.

#### 4.2. Main Results

Model-Strong Results Interpretation. Tab. 1 reports results in the model-strong (MS) end, where base models already exhibit strong priors aligned with the ground truth. In this setting, the −p objective consistently outperforms standard negative log-likelihood (−log p). This trend suggests that when model predictions are already reliable, a prior-leaning objective like −p better capitalizes on highconfidence tokens by suppressing the influence of lowprobability ones. To further dissect this effect, we evaluate a thresholded variant of −log p that excludes tokens with p < 0.2. This adjustment directly mitigates the effect

of low-confidence tokens and leads to consistent improvements over standard −log p. In many cases, it performs on par with, or even surpasses, −p applied to full tokens. Such evidence highlights that the weakness of standard NLL in this setting lies in its excessive emphasis on low-probability tokens. Prior-leaning objectives that explicitly downweight low-confidence tokens consistently yield the greatest gains at the MS end. This MS behavior is not restricted to the math domain: the coding experiments in Appen. C.3 show the same preference for −p over −log p, and the fixeddomain math scaling study further shows that the advantage of −p increases as the backbone becomes stronger. We provide further empirical analysis in Sec. 5 with a more careful study of the pattern.

Model-Intermediate Results Interpretation. In Tab. 2, results on medical reasoning reveal a strikingly different pattern: the performance of −p and −log p is nearly indistinguishable, with differences well within statistical variation. This neutrality arises from the nature of intermediate priors. On one hand, the priors are not strong enough for the prior-leaning objective −p to yield consistent refinements; on the other, they are not weak enough for the prior-averse objective −log p to offer a decisive corrective advantage. This observation is important because it indicates the existence of a region where gains are unlikely to come from altering the learning objective itself. Instead, improvements may rely on alternative directions, such as better data curation or targeted domain supervision.

- Table 2. Main results in the Model Intermediate (MI) region. Both −p and − log(p) result in similar performance.

Model MedMC MedQA PubMed MMLU-P GPQA Lancet MedB (4) MedB (5) MedX NEJM Avg. LLaMA-3.1-3B

- Base 21.30 21.92 22.60 11.40 23.08 25.00 23.05 15.26 10.35 23.22 19.48

- -log(p) 42.60 45.56 67.40 38.63 24.36 46.84 46.10 34.42 11.59 43.28 37.99

- -p 39.42 41.95 62.70 33.88 38.46 44.17 35.71 28.57 12.63 40.80 36.29 LLaMA-3.1-8B

Base 23.57 29.14 21.00 20.00 29.49 22.57 30.52 20.45 10.01 20.73 21.89

- -log(p) 55.08 59.47 74.00 53.62 32.05 57.28 52.27 46.10 15.87 59.20 47.23

- -p 54.10 58.44 76.50 52.70 44.87 54.13 42.21 42.53 13.80 54.73 45.89 Qwen2.5-1.5B

- Base 22.21 21.84 18.50 11.21 24.36 22.57 24.03 17.53 10.84 18.74 18.59

- -log(p) 39.64 39.59 66.70 34.92 33.33 38.83 38.31 27.60 10.56 34.16 35.13

- -p 38.58 36.68 68.00 38.37 35.90 35.68 36.69 28.90 11.94 39.97 35.02 Qwen2.5-Math-7B Base 35.84 27.26 49.30 30.23 35.90 30.34 24.03 18.18 10.21 24.71 27.55
- -log(p) 36.48 33.78 72.60 35.50 38.46 40.05 29.87 26.95 10.42 26.70 33.56

- -p 35.62 33.78 69.90 38.83 42.31 35.44 33.12 27.60 10.49 26.70 33.83

- Table 3. Main results in the Model Weak (MW) end. − log(p) consistently outperforms −p . Best results are in bold.

LLaMA-3.2-3B LLaMA-3.1-8B Qwen2.5-1.5B Qwen2.5-7B Metric Base -log(p) -p Base -log(p) -p Base -log(p) -p Base -log(p) -p Exact Match 0.00 1.08 0.00 0.00 1.34 0.00 0.00 0.60 0.0 0.00 35.20 0.00 Jaro-Winkler Similarity 41.89 44.39 2.43 30.17 43.59 10.15 35.32 32.98 8.36 44.92 82.48 10.15

Model-Weak Results Interpretation. Tab. 3 reveals the opposite trend at the MW end: −log p consistently outperforms −p, often by substantial margins. When priors are poorly aligned with the ground truth, the priorleaning objective −p allocates disproportionate weight to unreliable high-probability tokens, thereby reinforcing errors. By contrast, the prior-averse −log p ensures that lowprobability tokens, which often correspond to mistakes, receive stronger gradient signals, forcing the model to correct its errors and spread learning more broadly across the output distribution. This explains why NLL, despite its shortcomings elsewhere, remains the most effective objective in weak-prior settings. This MW pattern is also reproduced outside figfont puzzles: in low-resource multilingual instruction tuning, −log p consistently outperforms −p for both LLaMA-3.1-8B and Qwen2.5-7B (Appen. C.4). Consequently, progress on MW tasks is more likely to come from stronger or more targeted supervision or other methods of injecting knowledge. We provide further empirical analysis in Sec. 5 with a more careful study of the pattern.

#### 4.3. Experiments on General Instruction Tuning

To further demonstrate the breadth of our continuum view, we run additional general instruction-tuning experiments comparing the two objectives, −p and −log p. Experimental details and the full results are deferred to Appen. C.2. We fine-tune three Qwen2.5 base models spanning 3B to 14B parameters, and evaluate the resulting models trained

under each objective. Figure 4 reports the head-to-head win rate between the two final models at each scale. The results mirror the same continuum behavior observed elsewhere: as model scale increases, the preferred objective shifts across the MS, MI, and MW regimes, supporting the generality of our claim beyond previous settings.

Fixed-domain capability sweep. Here the data mixture, evaluation protocol, and objective comparison are held fixed, while only the base-model scale changes. The smooth transition from an NLL-favoring smaller model to a −p-favoring larger model therefore shows that the continuum behavior also appears within a single broad instruction-tuning setting.

60

−p −logp

Winrate(%)

| |
|---|

50

40

3B 7B 14B Qwen models

Figure 4. Head-to-head win rate of the −p -trained model against the − log(p) -trained model on AlpacaEval2. As backbone size increases from 3B to 14B, the prior-leaning objective becomes more effective.

#### 4.4. Additional Experiments

Generalization across scales and domains. The appendix experiments further show that the observed reversal is not limited to the three anchor domains. On the fixed math domain, Appen. C.3 shows that the performance gap between −p and −log p widens as the Qwen2.5 backbone scales from 3B to 32B. Across domains, it also reports coding results where −p again outperforms −log p, while Appen. C.4 reports low-resource multilingual instruction tuning results where −log p consistently outperforms −p. Together with the main experiments above, these results strongly support a capability-continuum interpretation.

### 5. Empirical Analysis

In this section, we provide a deeper empirical analysis of the findings in Sec. 4, with a particular emphasis on the MS and MW ends where the choice of training objective has the largest effect. Our goal is to move beyond merely reporting performance numbers and to analyze the mechanisms that drive the observed differences. To this end, we structure the analysis around three guiding questions:

- 1. In the MS end, what mechanisms explain the underperformance of NLL?
- 2. How do objectives with different emphasis on model priors behave across the two ends?
- 3. To what extent are these objectives consistent with likelihood estimation on the training set?

Answering these questions provides a deeper understanding of how different objectives interact with model capability from complementary perspectives.

Model Setup. For ablation studies in the MS end, we focus on Qwen-2.5-Math-1.5B, which shows the clearest gap between objectives. For the MW end, we use Qwen-2.5-7B. All training details and evaluation protocols remain identical to those in Sec. 4, ensuring that differences arise solely from the choice of objective.

###### Average Performance of Different Quantile Thresholding

34.22933.203 33.567 32.919

33.980

AveragePerformance

30

20

17.749

- -log p (>= Percentile)

- -log p (<= Percentile)

- -p (>= Percentile)

- -p (<= Percentile)

| |
|---|

| |
|---|

10

log(1-p) (<= Percentile) log(1-p) (>= Percentile)

0

| |
|---|

0 20 40 60 80 100

Percentile

Figure 5. Performance under quantile thresholding for − log(p), −p, and log(1 − p). Let Qpercentile denote the predicted probability at the specified percentile of the training set. (≥ Percentile) corresponds to I = [Qpercentile, 1] in Eq. 4, while (≤ Percentile) corresponds to I = [0, Qpercentile]. Key findings: (1) low-probability tokens consistently harm performance across all objectives; (2) when training on all tokens, objectives that deemphasize low-probability tokens (−p and log(1−p)) outperform − log(p); (3) restricting training to only the top 10% of tokens yields the strongest improvements across all objectives, surpassing standard SFT.

Bottom thresholds vary from 5% to 100%, and top thresholds vary from 0% to 90%.

Results Interpretation. The results in Fig. 5 reveal consistent patterns that align with our main experiments in Sec. 4. First, all objectives achieve strong performance when restricted to only the top 10% tokens, significantly exceeding standard NLL on all tokens. Second, performance drops sharply when training on low-probability tokens, confirming that they contribute adversarially to learning. Third, when applying bottom-thresholding, −p and log(1 − p) consistently outperform −log(p), illustrating the benefits of objectives that de-emphasize unreliable tokens. Finally, the degradation of log(p) performance when trained on all tokens (blue curve) can be largely attributed to the bottom 10% quantile. Overall, these results reinforce the main conclusion from Sec. 4: in the MS end, low-probability tokens act primarily as noise to the strong model.

- 5.1. Ablation on Quantile Thresholding with Different Objectives

Detailed Setup. This ablation examines how restricting training to different quantiles of tokens affects the relative performance of objectives. We compare three instances of f(p) in Eq. : −log(p), −p, and log(1 − p), which emphasize low-, mid-, and high-probability tokens, respectively (shown in Fig. 3). All experiments are identical except for the subset of tokens selected by the quantile thresholding rule in Eq. 4. Quantile thresholds are computed from the base model’s predicted token probabilities prior to training. We apply both bottom thresholding and top thresholding, denoted by (≥ Percentile) and (≤ Percentile), respectively.

#### 5.2. Objective Convexity and Performance Difference

Detailed Setup. To systematically examine the effect of objective on downstream performance, we study the parametric family in Eq. 3. This objective is concave when α ≥ 1 and convex when α ≤ 1. A “more concave” objective is more prior-leaning and vice versa, as shown in Fig. 3. We leverage the convexity of this objective as a proxy for assessing prior-leaning versus prior-averse objectives. We vary α from 0.1 to 1.0 in increments of 0.1, and from 1.0 to 10.0 in increments of 1.0.

Results Interpretation. As shown in Fig. 6, convexity affects performance in opposite directions across the SFT continuum. In the MS end, accuracy improves as α in-

###### Model-Strong Model-Weak

###### Performance vs

###### Performance vs

###### Likelihood Estimation

Likelihood Estimation

0.87

Jaro-WinklerSimilarity

0.65

| | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | |

| | |
|---|---|
| | |
| | |
| | |
| | |

32.5

0.8

0.6

0.86

###### MeanLikelihood

###### MeanLikelihood

MeanAccuracy

30.0

0.6

0.84

0.4

27.5

0.32

0.4

0.83

25.0

0.82

0.2

0.2

22.5

0.80

0.07

0.80

0.0

original -logp -p

original -logp -p

0.10.20.30.40.50.60.70.80.91.02.03.04.05.06.07.08.09.010.0

0.10.20.30.40.50.60.70.80.91.02.03.04.05.06.07.08.09.010.0

Method

Method

Power ( )

Power ( )

Figure 6. Analysis of MS and MW ends in terms of objective convexity (with Eq. 3) and likelihood estimation. In MS, more concave (prior-leaning) objectives yield better downstream accuracy, while in MW, more convex (prior-averse) objectives dominate. The likelihood estimation results align with these trends, suggesting that objective shape directly interacts with model prior strength.

### 6. Theoretical Analysis

creases, peaking near α = 1 and remaining stable for larger values. In the MW end, performance is maximized at α = 0.1 and deteriorates rapidly as α approaches 1 and exceeds the convexity boundary. This dichotomy highlights the importance of aligning objective shape with model prior strength: concave objectives (that emphasize model priors) are more effective when priors are strong, while convex objectives (that de-emphasize model priors) are preferable when priors are weak.

#### 6.1. Setup

Data. Let the input prompt be x ∈ X. The true conditional distribution over tokens y ∈ [V ] is denoted by r(y | x), with y∗ ∼ r(· | x). We write D for the marginal distribution over pairs x,r(· | x) , and let T(· | x) denote the empirical training distribution over contexts x, which we abuse the notation for writing (x,y˜) ∼ T. We use subscript p(·) to denote model predictions p(·).

#### 5.3. Likelihood Estimation on the Training Set

Model and objectives. Let pθ(· | x) = softmax zθ(x) be the next-token distribution of an autoregressive LM with parameters θ, and write p0(· | x) = pθ

Detailed Setup. In this ablation, we evaluate the empirical training performance of different objectives by computing the average predicted likelihood on the training set before and after fine-tuning:

(· | x) for the base model. We define the population risk to be

0

##### R(θ) = E(x,y∗)∼D,yp∼pθ(·|x) −{y ∗ = yp} ,

|y˜i|

n

1 N

pθ(˜yi,j). (5)

Likelihood Estimation(θ) :=

i=1

j=1

During SFT we minimize the empirical objective

where i denotes the i-th sample and j denotes the j-th token, and N = ni=1 |y˜i|, the total number of training tokens. We focus on comparing −p and −log(p) in both the MS and MW ends.

Lf(θ) = E(x,y˜)∼T f pθ(˜y | x)

where f : [0,1] → R is differentiable and decreasing in p. Our theoretical analysis mainly relies on the following assumption about the two ends of the continuum:

Results Interpretation. The likelihood estimation results, shown in Fig. 6, closely parallel the downstream accuracy trends. In the MS end, −p achieves higher mean predicted probabilities, confirming that they better align with strong model priors and effectively capture the training distribution. In contrast, in the MW end, −log(p) yield higher training performance, reflecting their ability to correct misaligned priors by emphasizing low-probability tokens. These findings indicate that the interaction between objective shape and regime governs not only generalization performance but also the model’s fit to the training data.

- Assumption 6.1 (Model-Capability Assumption). We make the following assumptions about model capability in the Model-Strong and Model-Weak ends:

- • Model-Weak. In the MW end, we assume that model predictions are uniform over the vocabulary V , where 2/V < 0.55.
- • Model-Strong. In the MS end, we assume that for

any given x, Pry∗,y˜ [(py∗ + py˜) ≥ 0.55] ≥ K with K ≥ 0.70.

- Assumption 6.2 (Trainable Base Model). We assume that the base model is still not perfect: for any given x, Pr[0.55 ≤ (py∗ + py˜) ≤ 0.95] ≥ 1 − K in the MS end.

Remark 6.3. The MW assumption captures the essential condition of weakness by modeling the base as uninformative. The MS assumption is grounded in practice: in Appen. C.1, we empirically validate this. For a uniform predictor, py∗ +py˜ = 2/V , so 2/V < 0.55 ensures that the MW assumption does not satisfy the MS threshold. This condition is trivially true for LLM-scale vocabularies. Assumption 6.2 is mild and simply guarantees that optimization is nontrivial. We choose 1−K for simplicity of proof.

- 6.2. Main Results We analyze the optimization dynamics of different objec-

tives under gradient flow. For an objective fi, let θ˙t(i) = −∇Lfi

(θ) denote the corresponding gradient flow, and let R(θt(i)) be the population risk at time t. Our goal is to maximize the reduction in risk, as captured by R˙ (θt(i)).

Theorem 6.4 (Characterization via Gradient Flow, Informal). Suppose that f2′(p) − f1′(p) < 0 for all p˜, and Assumptions 6.1–6.2 hold. Then, in a simplified setup, we have the following conclusions:

- • R˙ (θt(1)) t=0 ≥ R˙ (θt(2)) t=0 in Model Strong End.
- • R˙ (θt(1)) t=0 ≤ R˙ (θt(2)) t=0 in Model Weak End.

Remark 6.5. This theorem characterizes a sufficient condition for which the relative advantage of two objectives reverses across the MS and MW ends. For example, by setting f1(p) = 1 − p and f2(q) = −log p, we can conclude that in the model-strong end, the prior-leaning −p objective achieves larger risk reduction than NLL, whereas in the model-weak end, NLL is superior. This reversal mirrors our empirical observations and highlights the central theme of this work: the effectiveness of an SFT objective depends critically on model capability. The full version of the theoretical analysis is available at Appen. E.

- 7. Discussion

In this work, we revisited supervised fine-tuning (SFT) objectives for large language model post-training and showed that negative log likelihood (NLL), while classically optimal from scratch, is not universally effective once models already encode priors and supervision is long and noisy. Our central contribution is the model-capability continuum, instantiated through a general family of probability-based objectives, which shows that objective effectiveness depends critically on the prior strength of the base model. Across extensive analyses, we find that objectives reverse their relative advantage across different regions, yielding a unified explanation of how objective form interacts with model capability. Appen. F further situates several wellknown existing objectives in this lens: Focal loss is more

prior-averse than NLL, Huber-style probability losses are prior-leaning, and KL-regularized or RL-style objectives favor high-probability behaviors.

For practitioners, the continuum gives a simple diagnostic for objective choice. If the target task is well represented in pretraining, prior-leaning objectives such as −p or thresholded NLL are natural candidates. If the task is poorly covered and predictions are close to uninformative, NLL remains preferable because it gives strong corrective updates to low-probability tokens. This assessment can be guided by knowledge of the pretraining mixture when available, and more directly by measuring predicted probabilities on downstream examples. These observations also suggest that highly specialized models, such as strong mathematical reasoners, may benefit most from high-quality and relevant pretraining: once the prior is strong, lightweight SFT with an appropriate objective can be highly effective. Beyond SFT, prior-leaning objectives may also be useful during late-stage pretraining, where raw web-scale corpora contain substantial noise.

Our findings point to several natural future directions. First, adaptive objectives could adjust their prior-averse or priorleaning behavior as training progresses, rather than using a fixed objective throughout SFT. Such objectives may be especially useful in the intermediate regime, where neither prior-leaning nor prior-averse objective is uniformly preferred, and the right emphasis may change as the model improves. Second, a more general theoretical treatment could relax the stylized assumptions used in our analysis and characterize a broader range of training dynamics across the continuum. Third, developing better quantitative measures of model capability is an important direction. Predicted probability is a useful and accessible proxy, but it does not always coincide with true capability, especially in settings with miscalibration or confident hallucinations. More refined uncertainty quantification and capability diagnostics could therefore make objective selection more reliable in practice.

### Acknowledgement

We sincerely thank Ziqi Wang for his constructive advice and valuable engagement throughout this project. His sharp suggestions, thoughtful feedback, and sustained discussions substantially shaped and improved this work.

This work is supported by NSF (2134079) and DARPA ITM Program No. FA8650-23-C-7316. The content of the information in this document does not necessarily reflect the position or the policy of the Government, and no official endorsement should be inferred. The U.S. Government is authorized to reproduce and distribute reprints for Government purposes notwithstanding any copyright notation

here on. This research used both the DeltaAI advanced computing and data resource, which is supported by the National Science Foundation (award OAC 2320345) and the State of Illinois, and the Delta advanced computing and data resource which is supported by the National Science Foundation (award OAC 2005572) and the State of Illinois. Delta and DeltaAI are joint efforts of the University of Illinois Urbana-Champaign and its National Center for Supercomputing Applications. This work was supported by allocation CIS250611 from the Advanced Cyberinfrastructure Coordination Ecosystem: Services & Support (ACCESS) program, which is supported by National Science Foundation grants #2138259, #2138286, #2138307, #2137603, and #2138296.

### Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none of which we feel must be specifically highlighted here.

### References

Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I., Aleman, F. L., Almeida, D., Altenschmidt, J., Altman, S., Anadkat, S., et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

Aghajanyan, A., Shrivastava, A., Gupta, A., Goyal, N., Zettlemoyer, L., and Gupta, S. Better finetuning by reducing representational collapse. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum? id=OQ08SN70M1V.

AI Mathematical Olympiad Prize. Ai mathematical olympiad prize. https:// www.kaggle.com/competitions/ ai-mathematical-olympiad-prize, 2024. Accessed: 2025-09-24.

Bai, Y., Jones, A., Ndousse, K., Askell, A., Chen, A., DasSarma, N., Drain, D., Fort, S., Ganguli, D., Henighan, T., et al. Training a helpful and harmless assistant with reinforcement learning from human feedback. arXiv preprint arXiv:2204.05862, 2022.

Bartlett, P. L., Jordan, M. I., and McAuliffe, J. D. Convexity, classification, and risk bounds. Journal of the American Statistical Association, 101(473):138–156, 2006.

Casella, G. and Berger, R. Statistical inference. Chapman and Hall/CRC, 2024.

Chen, H., Fang, Z., Singla, Y., and Dredze, M. Benchmarking large language models on answering and ex-

plaining challenging medical questions. In Chiruzzo, L., Ritter, A., and Wang, L. (eds.), Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pp. 3563–3599, Albuquerque, New Mexico, April 2025. Association for Computational Linguistics. ISBN 979-8-89176-189-6. doi: 10.18653/v1/2025.naacl-long. 182. URL https://aclanthology.org/2025. naacl-long.182/.

Chu, T., Zhai, Y., Yang, J., Tong, S., Xie, S., Schuurmans, D., Le, Q. V., Levine, S., and Ma, Y. SFT memorizes, RL generalizes: A comparative study of foundation model post-training. In Forty-second International Conference on Machine Learning, 2025. URL https: //openreview.net/forum?id=dYur3yabMj.

Chung, H. W., Hou, L., Longpre, S., Zoph, B., Tay, Y., Fedus, W., Li, Y., Wang, X., Dehghani, M., Brahma, S., et al. Scaling instruction-finetuned language models. Journal of Machine Learning Research, 25(70):1– 53, 2024.

Cover, T. M. Elements of information theory. John Wiley & Sons, 1999.

Cox, D. R. The regression analysis of binary sequences. Journal of the Royal Statistical Society Series B: Statistical Methodology, 20(2):215–232, 1958.

Davis, D. and Recht, B. What is the objective of reasoning with reinforcement learning? ArXiv, abs/2510.13651, 2025. URL https://api.semanticscholar. org/CorpusID:282102854.

Dodge, J., Ilharco, G., Schwartz, R., Farhadi, A., Hajishirzi, H., and Smith, N. Fine-tuning pretrained language models: Weight initializations, data orders, and early stopping. arXiv preprint arXiv:2002.06305, 2020.

Dubois, Y., Liang, P., and Hashimoto, T. Length-controlled alpacaeval: A simple debiasing of automatic evaluators. In First Conference on Language Modeling, 2024. URL https://openreview.net/forum? id=CybBmzWBX0.

Gao, L., Tow, J., Abbasi, B., Biderman, S., Black, S., DiPofi, A., Foster, C., Golding, L., Hsu, J., Le Noac’h, A., Li, H., McDonell, K., Muennighoff, N., Ociepa, C., Phang, J., Reynolds, L., Schoelkopf, H., Skowron, A., Sutawika, L., Tang, E., Thite, A., Wang, B., Wang, K., and Zou, A. A framework for few-shot language model evaluation, 12 2023. URL https://zenodo.org/ records/10256836.

Gneiting, T. and Raftery, A. E. Strictly proper scoring rules, prediction, and estimation. Journal of the American statistical Association, 102(477):359–378, 2007.

Grattafiori, A., Dubey, A., Jauhri, A., Pandey, A., Kadian, A., Al-Dahle, A., Letman, A., Mathur, A., Schelten, A., Vaughan, A., et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

Hendrycks, D., Burns, C., Kadavath, S., Arora, A., Basart, S., Tang, E., Song, D., and Steinhardt, J. Measuring mathematical problem solving with the MATH dataset. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2), 2021. URL https://openreview. net/forum?id=7Bywt2mQsCe.

Howard, J. and Ruder, S. Universal language model finetuning for text classification. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 328– 339, 2018.

Huang, X., Wu, J., Liu, H., Tang, X., and Zhou, Y. m1: Unleash the potential of test-time scaling for medical reasoning with large language models. In Argaw, P., Zhang, H., Jabbour, S., Chandak, P., Ji, J., Mukherjee, S., Salaudeen, O., Chang, T., Healey, E., Gr¨oger, F., Adibi, A., Hegselmann, S., Wild, B., and Noori, A. (eds.), Proceedings of the Fifth Machine Learning for Health Symposium, volume 297 of Proceedings of Machine Learning Research, pp. 369–383. PMLR, 13– 14 Dec 2026. URL https://proceedings.mlr. press/v297/huang26a.html.

Huber, P. J. Robust estimation of a location parameter. In Breakthroughs in statistics: Methodology and distribution, pp. 492–518. Springer, 1992.

Jin, D., Pan, E., Oufattole, N., Weng, W.-H., Fang, H., and Szolovits, P. What disease does this patient have? a large-scale open domain question answering dataset from medical exams. Applied Sciences, 11(14):6421, 2021.

Jin, Q., Dhingra, B., Liu, Z., Cohen, W., and Lu, X. PubMedQA: A dataset for biomedical research question answering. In Inui, K., Jiang, J., Ng, V., and Wan, X. (eds.), Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pp. 2567–2577, Hong Kong, China, November 2019. Association for Computational Linguistics. doi: 10.18653/v1/D19-1259. URL https://aclanthology.org/D19-1259/.

Kirk, R., Mediratta, I., Nalmpantis, C., Luketina, J., Hambro, E., Grefenstette, E., and Raileanu, R. Understanding the effects of rlhf on llm generalisation and diversity. In The Twelfth International Conference on Learning Representations.

K¨oksal, A., Thaler, M., Imani, A., Ust¨¨ un, A., Korhonen, A., and Sch¨utze, H. Muri: High-quality instruction tuning datasets for low-resource languages via reverse instructions. Transactions of the Association for Computational Linguistics, 13:1032–1055, 2025.

Lewkowycz, A., Andreassen, A., Dohan, D., Dyer, E., Michalewski, H., Ramasesh, V., Slone, A., Anil, C., Schlag, I., Gutman-Solo, T., et al. Solving quantitative reasoning problems with language models. Advances in neural information processing systems, 35:3843–3857, 2022.

LI, J., Beeching, E., Tunstall, L., Lipkin, B., Soletskyi, R., Huang, S. C., Rasul, K., Yu, L., Jiang, A., Shen, Z., Qin, Z., Dong, B., Zhou, L., Fleureau, Y., Lample, G., and Polu, S. Numinamath. [https://huggingface.co/AI-MO/

NuminaMath-CoT](https://github.com/ project-numina/aimo-progress-prize/ blob/main/report/numina_dataset.pdf),

- 2024.

Li, Z., Chen, C., Xu, T., Qin, Z., Xiao, J., Luo, Z.-Q., and Sun, R. Preserving diversity in supervised finetuning of large language models. In The Thirteenth International Conference on Learning Representations,

- 2025. URL https://openreview.net/forum? id=NQEe7B7bSw.

Lightman, H., Kosaraju, V., Burda, Y., Edwards, H., Baker, B., Lee, T., Leike, J., Schulman, J., Sutskever, I., and Cobbe, K. Let’s verify step by step. In The Twelfth International Conference on Learning Representations, 2023.

Lin, T.-Y., Goyal, P., Girshick, R., He, K., and Doll´ar, P. Focal loss for dense object detection. In Proceedings of the IEEE international conference on computer vision, pp. 2980–2988, 2017.

Liu, J., Xia, C. S., Wang, Y., and Zhang, L. Is your code generated by chatgpt really correct? rigorous evaluation of large language models for code generation. Advances in Neural Information Processing Systems, 36:21558– 21572, 2023.

Liu, M., Farina, G., and Ozdaglar, A. E. UFT: Unifying supervised and reinforcement fine-tuning. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2026. URL https:// openreview.net/forum?id=usOkGv1S7M.

Mathematical Association of America. Math competitions. https://maa.org/math-competitions,

2023. Accessed: 2025-09-24.

Mathematical Association of America. Aime thresholds are available. https://maa.org/ aime-thresholds-are-available/, 2024. Accessed: 2025-09-24.

Mihaylov, T., Clark, P., Khot, T., and Sabharwal, A. Can a suit of armor conduct electricity? a new dataset for open book question answering. In Riloff, E., Chiang,

- D., Hockenmaier, J., and Tsujii, J. (eds.), Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pp. 2381–2391, Brussels, Belgium, October-November 2018. Association for Computational Linguistics. doi: 10.18653/v1/D18-1260. URL https://aclanthology.org/D18-1260/.

Milletari, F., Navab, N., and Ahmadi, S.-A. V-net: Fully convolutional neural networks for volumetric medical image segmentation. In 2016 fourth international conference on 3D vision (3DV), pp. 565–571. Ieee, 2016.

Mishra, S., Khashabi, D., Baral, C., and Hajishirzi, H. Cross-task generalization via natural language crowdsourcing instructions. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 3470–3487, 2022.

Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray,

- A., et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744, 2022.

Pal, A., Umapathi, L. K., and Sankarasubbu, M. Medmcqa: A large-scale multi-subject multi-choice dataset for medical domain question answering. In Conference on health, inference, and learning, pp. 248–260. PMLR, 2022.

Qin, C. and Springenberg, J. T. Supervised fine tuning on curated data is reinforcement learning (and can be improved). arXiv preprint arXiv:2507.12856, 2025.

Rege Cambrin, D., Gallipoli, G., Benedetto, I., Cagliero, L., and Garza, P. Beyond accuracy optimization: Computer vision losses for large language model fine-tuning. In Al-Onaizan, Y., Bansal, M., and Chen, Y.-N. (eds.), Findings of the Association for Computational Linguistics: EMNLP 2024, pp. 12060–12079, Miami, Florida, USA, November 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.findings-emnlp. 704. URL https://aclanthology.org/2024. findings-emnlp.704/.

Rein, D., Hou, B. L., Stickland, A. C., Petty, J., Pang, R. Y., Dirani, J., Michael, J., and Bowman, S. R. Gpqa: A graduate-level google-proof q&a benchmark. In First Conference on Language Modeling, 2024.

Savage, L. J. Elicitation of personal probabilities and expectations. Journal of the American Statistical Association, 66(336):783–801, 1971.

Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang, H., Zhang, M., Li, Y., Wu, Y., et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Sheng, G., Zhang, C., Ye, Z., Wu, X., Zhang, W., Zhang, R., Peng, Y., Lin, H., and Wu, C. Hybridflow: A flexible and efficient rlhf framework. arXiv preprint arXiv: 2409.19256, 2024.

Stojanovski, Z., Stanley, O., Sharratt, J., Jones, R., Adefioye, A., Kaddour, J., and K¨opf, A. Reasoning gym: Reasoning environments for reinforcement learning with verifiable rewards, 2025. URL https://arxiv. org/abs/2505.24760.

Taori, R., Gulrajani, I., Zhang, T., Dubois, Y., Li, X., Guestrin, C., Liang, P., and Hashimoto, T. B. Stanford alpaca: An instruction-following llama model. https://github.com/tatsu-lab/ stanford_alpaca, 2023.

Wang, B., Cheng, Q., Peng, R., Bao, R., Li, P., Guo, Q., Li, L., Zeng, Z., Zhou, Y., and Qiu, X. Implicit reward as the bridge: A unified view of SFT and DPO connections. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2026. URL https: //openreview.net/forum?id=xUx2B2NHvj.

Wang, Y., Ma, X., Zhang, G., Ni, Y., Chandra, A., Guo, S., Ren, W., Arulraj, A., He, X., Jiang, Z., et al. Mmlupro: A more robust and challenging multi-task language understanding benchmark. Advances in Neural Information Processing Systems, 37:95266–95290, 2024.

Wei, Y., Wang, Z., Liu, J., Ding, Y., and Zhang, L. Magicoder: Empowering code generation with OSS-instruct. In Salakhutdinov, R., Kolter, Z., Heller, K., Weller, A., Oliver, N., Scarlett, J., and Berkenkamp, F. (eds.), Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pp. 52632–52657. PMLR, 21– 27 Jul 2024. URL https://proceedings.mlr. press/v235/wei24h.html.

Wu, Y., Zhou, Y., Ziheng, Z., Peng, Y., Ye, X., Hu, X., Zhu, W., Qi, L., Yang, M.-H., and Yang, X. On the

generalization of SFT: A reinforcement learning perspective with reward rectification. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/forum? id=Lv7PjbcaMi.

Xie, J., Chen, A. S., Lee, Y., Mitchell, E., and Finn, C. Calibrating language models with adaptive temperature scaling. In Al-Onaizan, Y., Bansal, M., and Chen, Y.-N. (eds.), Proceedings of the

- 2024 Conference on Empirical Methods in Natural Language Processing, pp. 18128–18138, Miami, Florida, USA, November 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024. emnlp-main.1007. URL https://aclanthology. org/2024.emnlp-main.1007/.

Xu, C., Sun, Q., Zheng, K., Geng, X., Zhao, P., Feng, J., Tao, C., Lin, Q., and Jiang, D. Wizardlm: Empowering large pre-trained language models to follow complex instructions. In The Twelfth International Conference on Learning Representations, 2024.

Xu, Z., Jiang, F., Niu, L., Deng, Y., Poovendran, R., Choi, Y., and Lin, B. Y. Magpie: Alignment data synthesis from scratch by prompting aligned LLMs with nothing. In The Thirteenth International Conference on Learning Representations, 2025. URL https:// openreview.net/forum?id=Pnk7vMbznK.

Xuan, W., Yang, R., Qi, H., Zeng, Q., Xiao, Y., Feng,

- A., Liu, D., Xing, Y., Wang, J., Gao, F., Lu, J., Jiang, Y., Li, H., Li, X., Yu, K., Dong, R., Gu, S., Li, Y., Xie, X., Juefei-Xu, F., Khomh, F., Yoshie, O., Chen, Q., Teodoro, D., Liu, N., Goebel, R., Ma, L., MarreseTaylor, E., Lu, S., Iwasawa, Y., Matsuo, Y., and Li, I. MMLU-ProX: A multilingual benchmark for advanced large language model evaluation. In Christodoulopoulos, C., Chakraborty, T., Rose, C., and Peng, V. (eds.), Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 1513–1532, Suzhou, China, November 2025. Association for Computational Linguistics. ISBN 979-8-89176-332-6. doi: 10.18653/v1/2025.emnlp-main.79. URL https:// aclanthology.org/2025.emnlp-main.79/.

ACM Comput. Surv., 58(7), January 2026b. ISSN 03600300. doi: 10.1145/3777411. URL https://doi. org/10.1145/3777411.

Zhang, T. Statistical analysis of some multi-category large margin classification methods. Journal of Machine Learning Research, 5(Oct):1225–1251, 2004.

Zhong, T., Yang, Z., Liu, Z., Zhang, R., Liu, Y., Sun, H., Pan, Y., Li, Y., Zhou, Y., Jiang, H., et al. Opportunities and challenges of large language models for low-resource languages in humanities research. arXiv preprint arXiv:2412.04497, 2024.

Zhou, C., Liu, P., Xu, P., Iyer, S., Sun, J., Mao, Y., Ma, X., Efrat, A., Yu, P., Yu, L., et al. Lima: Less is more for alignment. Advances in Neural Information Processing Systems, 36:55006–55021, 2023.

Zhu, C., Xu, B., Wang, Q., Zhang, Y., and Mao, Z. On the calibration of large language models and alignment. In Bouamor, H., Pino, J., and Bali, K. (eds.), Findings of the Association for Computational Linguistics: EMNLP 2023, pp. 9778–9795, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.findings-emnlp. 654. URL https://aclanthology.org/2023. findings-emnlp.654/.

Zhu, W., Xie, R., Wang, R., Sun, X., Wang, D., and Liu, P. Proximal supervised fine-tuning. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/forum? id=hQtwQqYikp.

Zuo, Y., Qu, S., Li, Y., Chen, Z.-R., Zhu, X., Hua, E., Zhang, K., Ding, N., and Zhou, B. MedxpertQA: Benchmarking expert-level medical reasoning and understanding. In Forty-second International Conference on Machine Learning, 2025. URL https:// openreview.net/forum?id=IyVcxU0RKI.

Zhang, D., Dai, Q., and Peng, H. The best instructiontuning data are those that fit. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2026a. URL https://openreview.net/ forum?id=4jFSekBaDT.

Zhang, S., Dong, L., Li, X., Zhang, S., Sun, X., Wang, S., Li, J., Hu, R., Zhang, T., Wang, G., and Wu, F. Instruction tuning for large language models: A survey.

### Contents

- A Related Works 15
- B Detailed Experimental Setup 15

- B.1 Continuum Selection . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16
- B.2 Training and Evaluation Details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16

- C Additional Experiment Results 17

- C.1 Justification for Assumptions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
- C.2 General Instruction Tuning Experiments . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
- C.3 Additional Model Strong Experiments . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
- C.4 Additional Model Weak Experiments . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
- C.5 Additional Knowledge Memorization Experiments . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19

- D Proofs for Sec. 3 20
- E Main Theoretical Results 21

- E.1 Setup and Notations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21
- E.2 Assumptions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22
- E.3 Main Proofs . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22

- F Discussion with Existing Literature 27

- F.1 Connections with RL . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 27
- F.2 Connections with Other Loss Functions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 28

### A. Related Works

Language Model Post-training. Supervised Fine-Tuning (SFT) has emerged as the dominant paradigm for post-training, adapting pretrained models to tasks or domains by directly fitting labeled data (Zhang et al., 2026b; Chung et al., 2024). The availability of high-quality instruction datasets (Mishra et al., 2022; Zhou et al., 2023; Taori et al., 2023; Lightman et al., 2023) has further boosted SFT’s effectiveness. Nevertheless, abundament studies highlight that SFT alone often overfits, generalizes poorly, and yields sub-optimal models (Howard & Ruder, 2018; Dodge et al., 2020; Ouyang et al., 2022). To address these limitations while retaining SFT’s efficiency, the prevailing recipe is to combine SFT with RL, forming the de facto post-training paradigm (Bai et al., 2022; Achiam et al., 2023; Kirk et al.; Chu et al., 2025; Liu et al., 2026). Yet, existing SFT post-training consistently minimizes the negative log-likelihood objective, −log(p), whose suitability has rarely been questioned. In this work, we show that it is not universally optimal and argue for revisiting objectives that better exploit pretrained priors in SFT.

Improving SFT (from an RL perspective). Motivated by the success of reinforcement learning in reasoning tasks, a growing body of work seeks to reinterpret and improve SFT through an RL lens. Wang et al. (2026) cast both SFT and DPO as instances of implicit reward learning, showing that smaller learning rates and alternative divergence-based objectives can enhance performance. Qin & Springenberg (2025) integrates importance sampling into SFT, while Zhu et al. (2026) introduces a PPO-style clipped surrogate objective to constrain policy drift. Aghajanyan et al. (2021) introduces an explicit KL regularizer during fine-tuning, which can also be viewed as a prior-leaning mechanism that discourages unnecessary deviation from the pretrained model. Most closely related to our work, Wu et al. (2026) proposes reweighting gradient coefficients uniformly, essentially equivalent to our −p objective, for which we provide a deeper characterization and analysis. Overall, these approaches can be regarded as special cases of our proposed “prior-leaning” objectives, implemented through RL techniques to downweight low-probability tokens. In contrast, we show that the same effect can be achieved far more simply by applying a threshold. Moreover, these RL-inspired methods are only validated in a single domain, whereas we demonstrate the potential limitations of prior-leaning objectives in the model-weak end. Other than RL-inspired approaches, Zhang et al. (2026a) further explore data selection by favoring high-probability instances, a weaker form of our token-wise thresholding objective.

Classical views on SFT learning objectives. In the conventional view of classification, the NLL has long been regarded as the optimal training objective: it is the maximum likelihood estimator (statistical consistency) (Cox, 1958; Casella & Berger, 2024), equivalent to minimizing cross-entropy/KL-divergence (information-theoretic) (Cover, 1999), the unique strictly proper local scoring rule ensuring calibrated probabilities (decision-theoretic) (Savage, 1971; Gneiting & Raftery, 2007), and a convex surrogate to 0-1 loss guaranteeing Bayes consistency and tractable optimization (learningtheoretic) (Bartlett et al., 2006; Zhang, 2004). These arguments, however, assume training from scratch on simple classification tasks, whereas SFT in language model post-training starts from powerful pretrained models with long chain-ofthought supervision where only final answers are evaluated and intermediate tokens may be noisy. Under these conditions, the premises for −log(p) might no longer hold, and in this work, we provide the first systematic characterization of such settings. We provide a detailed discussion with other loss functions in Appen. F.2.

### B. Detailed Experimental Setup

Table 4. General experimental setup across different regions of the model-capability continuum.

Continuum Domain Signals Training Data Evaluation Data Objectives to Compare MS math-reasoning sparse NuminaMath CoT Math500, Minerva Math, Olympiad Bench, AIME24, AMC23 -p, -log(p), threshold(-log(p)) MI medical-reasoning sparse m23k

MedMC, MedQA, PubMed, MMLU-P, GPQA, Lancet, MedB(4), MedB(5), MedX, NEJM

-p, -log(p) MW text games dense synthetic synthetic -p, -log(p)

We now provide details of our experimental setup, including the rationale for the choice of datasets across the continuum, the corresponding training and evaluation benchmarks, and specific training protocols. An overview is summarized in Tab. 4.

- Table 5. Continuum selection based on mean predicted probability (Eq. 5). In the MS end, base models already achieve high likelihood on the training set before fine-tuning; in the MI region, predictions are around 0.5; in the MW end, predictions are near zero.

Model Strong (Math) Mean Predicted Probability 0.80 0.76 0.80 0.81

- Model Name LLaMA-3.1-8B DeepSeekMath-7B Qwen2.5-Math-1.5B Qwen2.5-Math-7B Model Intermediate (Med)

Mean Predicted Probability 0.50 0.53 0.56 0.59

- Model Name LLaMA-3.2-3B LLaMA-3.1-8B Qwen2.5-1.5B Qwen2.5-Math-7B Model Weak (Puzzles)

Mean Predicted Probability 0.01 0.01 0.01 0.07 Model Name LLaMA-3.2-3B LLaMA-3.1-8B Qwen2.5-1.5B Qwen2.5-7B

- B.1. Continuum Selection

We assign math tasks to the MS end, medical tasks to the MI region, and figfont puzzles to the MW end. For the MS end, we use LLaMA-3.1-8B, DeepSeekMath-7B, Qwen2.5-Math-1.5B, and Qwen2.5-Math-7B. For the MI region, we use LLaMA-3.2-3B, LLaMA-3.1-8B, Qwen2.5-1.5B, and Qwen2.5-Math-7B. For the MW end, we use LLaMA-3.2-3B, LLaMA-3.1-8B, Qwen2.5-1.5B, and Qwen2.5-7B. We rely on base models in all cases.

Our rationale for this selection is twofold.

First, evidence from pretraining corpora. Fig. 2 illustrates that some domains are strongly represented in pretraining while others are not. For example, open-sourced documentation of LLaMA-3 reports that ∼25% of pretraining tokens are math-related (Grattafiori et al., 2024), indicating strong priors for math reasoning. Similarly, DeepSeekMath and Qwen2.5Math were explicitly pretrained on math corpora. By contrast, medical corpora are only partially present in pretraining, yielding moderate priors, and figfont puzzles are completely absent, making them a natural MW task.

Second, quantitative evidence from model predictions. Tab. 5 shows mean predicted probabilities (Eq. 5) on the training set, which we use as a proxy for prior strength given that base LLMs are generally well-calibrated and their predictions more faithfully reflect inherent model capability (Zhu et al., 2023; Xie et al., 2024) . In the MS end, models already achieve very high likelihoods (around 0.8) before fine-tuning. In the MW end, predictions are close to zero, reflecting a lack of relevant prior knowledge. In between, predictions cluster around 0.5, reflecting an intermediate level of task familiarity. Together, these observations justify our continuum classification and ground it in both qualitative and quantitative evidence.

#### B.2. Training and Evaluation Details

General framework. All SFT experiments are conducted using verl (Sheng et al., 2024). We fix the optimizer to AdamW, with a base learning rate of 5 × 10−5 for all models except LLaMA-3.1-8B, where we use 2 × 10−5. We employ cosine decay scheduling with a warm-up ratio of 0.1, and train for a single epoch. All training runs are performed on 2 H200 GPUs with a single node. We have initially tuned the learning rate over {1e − 3,5e − 4,1e − 4,1e − 5,2e − 5,5e − 5,1e−6,5e−6} with one single model on each setting and do not observe notable performance changes when the learning rate is around 5e − 4 to 1e − 6 with the objectives we studied in this paper. Therefore, we generally fix the learning rate to ensure a fair and computationally efficient comparison.

Model-Strong (Math). Our setup for mathematical reasoning largely follows Wu et al. (2026). We train on NuminaMathCoT (LI et al., 2024), which contains 859k chain-of-thought problems collected from multiple sources. For efficiency, we sample a 67k subset, which we find to achieve equivalent performance to larger subsets (100k+ or more). We set the maximum training length to 3072 tokens and use a micro-batch size of 4. Evaluation covers five representative math benchmarks: Math500 (Hendrycks et al., 2021), Minerva Math (Lewkowycz et al., 2022), Olympiad Bench (AI Mathematical Olympiad Prize, 2024), AIME24 (Mathematical Association of America, 2024), and AMC23 (Mathematical Association of America, 2023). Each evaluation uses temperature 1.0, with results reported as the average of 16 generations per example and a maximum generation length of 4096 tokens.

Model-Intermediate (Medical). We train on m23k (Huang et al., 2026), a 23k-instance medical reasoning dataset. We experimented with two variants: (i) including long-form reasoning traces (maximum length 8192, micro-batch size 1) and (ii) using only standard chain-of-thought (maximum length 1024, micro-batch size 16). Since performance was similar, we report results from the standard CoT variant. Evaluation strictly follows the protocol in Huang et al. (2026), using temperature 0 and random seed 42. Benchmarks include MedMCQA (Pal et al., 2022), MedQA-USMLE (Jin et al., 2021), PubMedQA (Jin et al., 2019), MMLU-Pro (Wang et al., 2024), GPQA (Medical) (Rein et al., 2024), Lancet & NEJM (Huang et al., 2026), MedBullets (Chen et al., 2025), and MedXpertQA (Zuo et al., 2025). A detailed overview of these datasets is provided in Huang et al. (2026).

Model-Weak (Figfont). We generate synthetic figfont puzzles from ReasoningGym (Stojanovski et al., 2025). We generate synthetic figfont puzzle data from ReasoningGym (Stojanovski et al., 2025), creating 40k instances for training and 20k for evaluation. An example puzzle is shown in Fig. 2. Training mirrors the MI setup, with a maximum sequence length of 800 and a micro-batch size of 16. Inference uses temperature 0 and random seed 42. We evaluate with two metrics: (i) exact match and (ii) Jaro–Winkler similarity, a string-based similarity score that is more tolerant to small variations and complements the strictness of exact match.

### C. Additional Experiment Results

#### C.1. Justification for Assumptions

- Table 6. Fraction of training-set tokens whose initial predicted probability exceeds 0.55 in the MS end. The high fractions indicate that the base models already assign substantial probability mass to training targets before fine-tuning, supporting Assump. 6.1.

LLaMA-3.1-8B DeepSeekMath-7B Qwen2.5-Math-1.5B Qwen2.5-Math-7B

Percentage of tokens with initial predicted probability larger than 0.55

72.8% 76.7% 80.6% 81.2%

Tab. 6 reports this quantity across the MS backbones used in our experiments.

C.2. General Instruction Tuning Experiments

To demonstrate that our continuum extends beyond specialized domains (e.g., math or medical QA), we additionally consider general conversational alignment, where the goal is to match subtle human preferences and stylistic behaviors. We construct a mixed SFT corpus from two well-known, high-quality public datasets: Magpie-Prob-300K-Filtered (Xu et al., 2025) and EvoInstruct (Xu et al., 2024), which together target complex instruction following abilities. For efficiency, we subsample 70k instances from the Magpie-Prob-300K-Filtered dataset and we retain only instances whose total sequence length is at most 2048 tokens. This yields a combined training set of approximately 140K examples.

We consider three backbones, Qwen2.5-3B, Qwen2.5-7B, and Qwen2.5-14B, and fine-tune each for one epoch with batch sizes of 8, 4, and 2 and learning rates of 5e-5, 2e-6, and 1e-6, respectively. We then compare models trained with the standard NLL objective −log p versus the prior-leaning objective −p to probe the model-capability continuum in this more general alignment setting. For evaluation, due to limited API budget, we use AlpacaEval2 (Dubois et al., 2024) and directly compare responses from the −p model versus the −log p model. In the table below, we report the win rate of the prior-leaning model against the NLL baseline.

- Table 7. Win rates (%) of the prior-leaning objective −p against the NLL baseline − log p on AlpacaEval2. We report both lengthcontrolled win rate (LC WR) and standard win rate (WR), measuring the fraction of pairwise comparisons where the −p model is preferred over the − log p model. As the backbone scales from 3B to 14B parameters, the prior-leaning objective transitions from underperforming to outperforming NLL, consistent with our model-capability continuum.

Qwen2.5-3B Qwen2.5-7B Qwen2.5-14B

LC WR 41.0 49.6 57.5 WR 42.0 49.0 57.1

In Table 7, we observe a clear continuum-style pattern as the model becomes stronger. For the smaller Qwen2.5-3B

backbone, the prior-leaning objective −p underperforms NLL, achieving only about 41–42% win rate, indicating that aggressively trusting the model’s prior is detrimental when the backbone is relatively weak. For the intermediate Qwen2.57B model, the win rates are close to 50%, suggesting that the two objectives behave similarly in this regime. In contrast, for the larger Qwen2.5-14B backbone, the same prior-leaning objective attains around 57–58% win rate over NLL on AlpacaEval2, suggesting that once the model’s prior is sufficiently strong, emphasizing the model’s prior beliefs becomes beneficial even on broad alignment tasks.

#### C.3. Additional Model Strong Experiments

Additional Model-Strong: Coding Task. To further illustrate the generality of the model-strong end of our continuum, we examine the coding domain using the Qwen2.5-Coder-7B backbone, which is known to contain substantial codingrelated knowledge. For evaluation, we adopt the EvalPlus suite (Liu et al., 2023) and consider four standard benchmarks: HumanEval, HumanEval+, MBPP, and MBPP+ (all reported as pass@1 following Liu et al. (2023)). For training, we use Magicoder-OSS-Instruct-75K (Wei et al., 2024), a high-quality instruction-tuning corpus tailored specifically for coding tasks, and fine-tune the model for one epoch.

Tab. 8 shows that, in this coding-heavy, model-strong setting, the prior-leaning objective −p consistently outperforms NLL across all four benchmarks. Relative to the base model, −p delivers large absolute gains (e.g., from 63.6 to 77.8 average pass@1). On MBPP and MBPP+, the −p model nearly matches the performance of Qwen2.5-Coder-7B-Instruct (83.5 and 71.7, respectively), despite using only 75K training examples. This provides additional evidence that in domains where the backbone already encodes rich task-relevant knowledge, moving toward the model-strong end with a prior-leaning objective can yield substantial gains.

Table 8. Model-strong experiments on coding tasks with Qwen2.5-Coder-7B on the EvalPlus benchmarks (pass@1). The prior-leaning objective −p improves over the NLL objective − log p and substantially boosts performance over the base model. On MBPP and MBPP+, the −p model approaches the performance of the much more heavily tuned Qwen2.5-Coder-7B-Instruct variant, despite using only 75K instruction-tuning examples.

Case HumanEval MBPP Avg.

HE HE+ MBPP MBPP+

Qwen2.5-Coder-7B

Base 61.6 53.0 76.9 62.9 63.6

- -log(p) 80.4 71.3 78.8 68.3 74.7

- -p 83.5 75.6 83.1 69.0 77.8

Performance versus Model Scale in the Model-Strong End. We also study how the benefit of the prior-leaning objective −p evolves with model scale in the model-strong end. Concretely, we evaluate Qwen2.5 models at four sizes (3B, 7B, 14B, and 32B parameters), training and evaluating them on the same mathematical tasks and configurations as in the main paper. For the larger models (14B and 32B), we use slightly smaller learning rates of 2 × 10−5 and 10−5, respectively, to ensure stable optimization.

Fig. 7 plots the performance of −p versus NLL with both axes in log scale. As model size increases, the performance gap in favor of the prior-leaning objective widens monotonically: the gains are modest at 3B, more pronounced at 7B, and largest at 14B and 32B. This trend is consistent with our continuum view: as models become more capable and their pretrained priors more informative, they can exploit prior-leaning objectives more effectively, leading to larger improvements in the model-strong end.

#### C.4. Additional Model Weak Experiments

Low-resource Language Instruction Tuning. We study instruction tuning in low-resource language settings, which provide natural domains where existing language models remain very weak (Zhong et al., 2024). For evaluation, we use MMLU-ProX (Xuan et al., 2025), a challenging question answering benchmark adapted from MMLU-Pro that covers a wide range of languages and assesses general knowledge and reasoning abilities. Among these, we identify seven low-resource languages for which corresponding instruction-tuning data exist: Marathi (MR), Telugu (TE), Nepali (NE), Swahili (SW), Wolof (WO), Yoruba (YO), and Zulu (ZU).

Performance Improvement vs Model Size

| | | | | | | | | | | | |
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
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |

- 100
- 101

PerformanceImprovement

3B 7B 14B 32B

Model Size (Billions of Parameters)

Figure 7. Performance improvement of the objective −p over NLL across model scales. Both axes use log scale. Larger models exhibit larger gains, showing that the prior-leaning objective is increasingly effective as the underlying model becomes stronger.

For training, we use the muri-it dataset (K¨oksal et al., 2025), a 2M-example instruction-tuning corpus that spans nearly all languages. We extract the subset corresponding to the seven evaluation languages, resulting in a total of 95K training instances. We evaluate two backbones, LLaMA-3.1-8B and Qwen2.5-7B, both of which exhibit very limited pretraining exposure to these languages and have extremely low zero-shot performance. Each model is trained for one epoch using the same configuration described earlier in the paper.

In terms of our continuum, these settings are model-weak. Tab. 9 presents the results and indeed confirms this. In every low-resource evaluation language and for both backbones, the standard NLL objective substantially outperforms the priorleaning objective −p. This demonstrates that the model-weak end extends far beyond the puzzle cases discussed in the main paper and highlights low-resource multilingual instruction tuning as a natural and practically important instance of this end.

- Table 9. Additional benchmark results for the model-weak end. Low-resource languages provide a natural setting where pretrained models perform poorly. Best results are in bold.

Case Marathi (MR) Telugu (TE) Nepali (NE) Swahili (SW) Wolof (WO) Yoruba (YO) Zulu (ZU) Avg. LLaMA-3.1-8B Base 11.1 5.0 12.3 4.1 3.7 7.4 6.0 7.1

- -p 15.2 2.5 15.5 14.0 11.5 10.0 11.5 11.5
- -log(p) 18.7 13.3 18.7 18.4 13.9 13.6 15.9 16.1 Qwen2.5-7B Base 8.7 2.4 9.4 8.3 9.9 7.9 7.2 7.7
- -p 20.6 7.7 19.5 14.9 12.9 10.5 6.9 13.3
- -log(p) 24.8 9.1 24.4 19.9 13.1 12.6 9.6 16.2

#### C.5. Additional Knowledge Memorization Experiments

In this subsection, we demonstrate that our continuum view also applies to a classical knowledge memorization task. We use OpenBookQA (Mihaylov et al., 2018) to study commonsense and factual question answering. The dataset contains 5K training samples and 500 test samples. We fine-tune Qwen2.5-14B and Qwen2.5-7B, both of which achieve over 80% zero-shot accuracy on this benchmark, for one epoch using the same configurations described earlier in the paper.

Importantly, the labels in this setting are few, unambiguous, and essentially noise-free. This places the task in the regime where classical supervised classification applies and where NLL is expected to excel. In contrast, our motivation for priorleaning objectives comes from the observation that typical reasoning SFT datasets contain imperfect and potentially noisy demonstrations. To emulate the model strong end under such conditions, we perturb the training set with small amounts of label noise and train under the same protocol. This serves as a proxy for pretraining style noise, where incorrect or ambiguous labels are common.

Tab. 10 reports the results. In the clean, conventional-classification-like setting, NLL indeed performs best, fully consistent

with our framework. However, once noise is introduced, analogous to imperfections in pretraining corpora and reasoning SFT datasets, the prior-leaning objective −p remains robust while NLL collapses. Moreover, as the model scale increases (14B compared to 7B), the performance gap between prior-leaning and prior-averse objectives in the clean setting becomes smaller. These findings illustrate that the continuum we identify extends beyond long-form reasoning and also governs knowledge memorization under varying supervision quality.

- Table 10. Additional benchmark results on knowledge memorization (OpenBookQA). The clean setting uses the original labels and reflects standard classification with noise-free supervision. The noisy setting adds small label perturbations to emulate pretraining-style noise. In the clean setting, NLL performs best, whereas under noisy supervision, the prior-leaning objective −p is substantially more robust, illustrating our continuum beyond reasoning tasks.

###### Cases OpenBookQA

Clean (MW) Noisy (MS)

Qwen2.5-14B Base 82.2

-log(p) 95.0 27.0 -p 93.4 91.6

Qwen2.5-7B Base 81.0

- -log(p) 91.0 27.2
- -p 86.2 85.0

### D. Proofs for Sec. 3

- Lemma D.1 (Gradient Shape). Let f : [0,1] → R be differentiable and nonincreasing. Consider the objective in Eq. , whose step-t contribution depends on the correct-class probability pt,y = softmax(zt)y only through f(pt,y). Then the gradient of Lf with respect to the logits at step t satisfies

∂Lf ∂zt,i

= sf(pt,y) δi,y − pt,i , where sf(p) ≜ −f′(p)p ≥ 0.

In particular, for the correct class i = y,

∂Lf ∂zt,y

= sf(pt,y)(1 − pt,y) = Wf(pt,y), Wf(p) ≜ −f′(p)p(1 − p).

Proof. Write pt = softmax(zt), so pt,i = exp(zt,i)/ j exp(zt,j). The softmax Jacobian gives, for all i,

∂pt,y ∂zt,i

Since the step-t loss is f(pt,y), the chain rule yields

= pt,y (δi,y − pt,i).

∂Lf ∂zt,i

∂pt,y ∂zt,i

= f′(pt,y)

= f′(pt,y)pt,y (δi,y − pt,i) = −f′(pt,y)pt,y (δi,y − pt,i).

Define sf(p) = −f′(p)p. Because f is nonincreasing, f′(p) ≤ 0 on (0,1), hence sf(p) ≥ 0. The displayed formula then follows, and for i = y we obtain ∂z∂Lf

= sf(pt,y)(1 − pt,y) = −f′(pt,y)pt,y(1 − pt,y) = Wf(pt,y).

| |
|---|

t,y

Proposition D.2 (Convex versus Concave Objectives). Let f ∈ C2[0,1] with f′(p) < 0 for all p ∈ (0,1), and define Wf(p) = −f′(p)p(1 − p). If f is concave (f′′ ≤ 0), then any maximizer of Wf lies in [12,1]. If f is convex (f′′ ≥ 0), then any maximizer of Wf lies in [0, 12].

Proof. Set s(p) := −f′(p). Then s(p) > 0 on (0,1) by the hypothesis f′(p) < 0, and

Wf(p) = s(p)p(1 − p). Differentiate:

Wf′(p) = s′(p)p(1 − p) + s(p)(1 − 2p).

Concave case. If f′′ ≤ 0 on [0,1], then s′(p) = −f′′(p) ≥ 0. For p ∈ (0, 12) we have 1 − 2p > 0, hence both terms in Wf′(p) are nonnegative; since s(p) > 0, in fact Wf′(p) > 0 on (0, 12). Therefore Wf is strictly increasing on (0, 12), so no maximizer can lie in (0, 12); any global maximizer must belong to [12,1].

Convex case. If f′′ ≥ 0 on [0,1], then s′(p) = −f′′(p) ≤ 0. For p ∈ (12,1) we have 1 − 2p < 0; with s(p) > 0 the two terms in Wf′(p) are nonpositive, hence Wf′(p) < 0 on (12,1). Thus Wf is strictly decreasing on (12,1), so no maximizer can lie in (12,1); any global maximizer must belong to [0, 12].

Combining the two cases establishes the claim.

| |
|---|

### E. Main Theoretical Results

#### E.1. Setup and Notations

Data model. Let the input prompt x ∈ X. The true conditional distribution over tokens y ∈ [V ] is r(y | x). We let D denote the (marginal) distribution over pairs x,r(· | x) . We use T(· | x) to denote the empirical training distribution over contexts x.

Model and objectives. Let qθ(· | x) = softmax zθ(x) be the next-token distribution of an autoregressive LM with parameters θ, and write q0(· | x) = qθ

(· | x) for the base model. We note that we use different notations q (instead of p) to denote the model predictions in the appendix. The population risk is

0

R(θ) = E(x,y∗)∼D,q∼qθ(·|x) −{y ∗ = yq} During SFT we minimize the empirical objective

Lf(θ) = E(x,y˜)∼T f qθ(˜y | x) where f : [0,1] → R is differentiable and decreasing.

Notation. Let zθ(x) ∈ RV denote the pre-softmax logits and qθ(· | x) = softmax zθ(x) the next-token distribution. Fix x and suppress its dependence when clear. Define the logit feature map

(x,y) ∈ Rd, Φ(x) := [Φ(x,1),...,Φ(x,V )] ∈ Rd×V , and its Gram matrix over logits

Φ(x,y) := ∇θzθ

0

G(x) := Φ(x)⊤Φ(x) ∈ RV ×V , Gy,y′(x) = ⟨Φ(x,y),Φ(x,y′)⟩. Write q := qθ

(· | x), r := r(· | x), and T := T(· | x). For a differentiable, increasing fi : [0,1] → R, set

0

V

(βi)y := Ty qy fi′(qy), βi ∈ RV , Sf

Ty qy fi′(qy).

:= ⟨βi,1⟩ =

i

y=1

Define the discrepancy vectors

v∗ := r⊤q q − r ⊙ q, vi := βi − Sf

, v12 := v1 − v2 = β12 − S12q. Finally, let gi := ∇Lfi

q, β12 := β1 − β2, S12 := Sf

1 − Sf

i

2

(θ0), ki := ⟨∇R(θ0),gi⟩ and

1

∇2R θ0 − tη gi dt for a stepsize η > 0 (used later in second-order expansions).

Hi :=

0

#### E.2. Assumptions

- E.2.1. MAIN ASSUMPTIONS

- Assumption E.1 (Model-Capability Assumption). We make the following assumptions about data capability in the ModelStrong and Model-Weak ends:

- • Model-Weak. In the MW end, we assume that model predictions are uniform over the vocabulary V , where 2/V < 0.55.
- • Model-Strong. In the MS end, we assume that for any given x, Pry∗,y˜ [(qy∗ + qy˜) ≥ 0.55] ≥ K with K ≥ 0.70.

- Assumption E.2 (Trainable Base Model). We assume that the base model is still not perfect: for any given x, Pr[(0.55 ≤ qy∗ + qy˜) ≤ 0.95] ≥ α Pry∗,y˜ [(qy∗ + qy˜) ≤ 0.50] in the MS end.

These assumptions are mentioned in the main paper with justifications. For a uniform predictor, qy∗ + qy˜ = 2/V , so 2/V < 0.55 ensures that the MW assumption does not satisfy the MS threshold. This condition is trivially true for LLMscale vocabularies. The coefficient α could depend on the task itself, and this value ≥ 1 in practice. Assumption E.2 is a more general re-statement of Assumption 6.2.

E.2.2. ADDITIONAL SIMPLIFICATION ASSUMPTIONS

- Assumption E.3 (Model and Data Simplifications). We assume that the feature matrix Φ is preconditioned such that all of its singular values are equal to one, and that both the training distribution T and the true distribution r are one-hot.

- E.3. Main Proofs

This assumption is made purely for analytical convenience: it removes irrelevant conditioning factors in the proof and allows us to focus on the essential differences between objectives.

- Lemma E.4 (Gradient identities). We have the following identities:

∇R(θ0) = Ex Φ(x)v∗(x) , ∇Lfi

(θ0) = Ex Φ(x)vi(x) ,

Proof. Population risk. With R(θ) = Ex −r(x)⊤qθ(· | x) , for fixed x we have ∂R/∂q = −r. By the chain rule through softmax,

∂R ∂z

= J(q)(−r) = (q⊤r)q − q ⊙ r,

so ∇θR(θ0) = Φ(x) ∂∂zR = Φ(x)v∗(x). Taking expectation over x yields the first identity. General fi-objective. For Lfi

(θ) = Ex y Ty(x)fi(qy) , ∂Lfi

/∂q = mi with mi = (Tyfi′(qy))y. Again, ∂Lfi

/∂z = J(q)mi = vi, whence ∇θLfi

(θ0) = Φ(x)vi(x) and the claim follows after taking expectation over x. Lemma E.5 (Functional derivative). Define

| |
|---|

J(fi) := Ex v∗⊤Φ⊤Φvi −

η 2

vi⊤Φ⊤Hi Φvi , Hi :=

1

∇2R θ0 − tη gi dt,

0

= y Tyqyfi′(qy). For a perturbation h of fi (so that fi  → fi + ϵh), the first variation is

with gi := ∇Lfi

(θ0) = Ex[Φvi], v∗ := r⊤q q − r ⊙ q, vi := βi − Sf

q, (βi)y := Tyqyfi′(qy), Sf

i

i

1

η2 2

δJ(fi;h) = Ex v∗⊤Φ⊤Φ − η vi⊤Φ⊤HiΦ δvi +

t ∇3R θ0 − tηgi [δgi], gi ⊗ gi dt,

0

where δgi = Ex[Φδvi] and

)q = Diag(T ⊙ q) − q (T ⊙ q)⊤ h′(q).

δvi = δβi − (δSf

i

Proof. Write A := Φ(x) for brevity. Then

J = Ex v∗⊤A⊤Avi −

η 2

vi⊤A⊤HiAvi .

Vary fi  → fi + ϵh. Since v∗ is fixed, δ(v∗⊤A⊤Avi) = v∗⊤A⊤Aδvi. For the second term, use the product rule:

δ vi⊤A⊤HiAvi = 2vi⊤A⊤HiAδvi + vi⊤A⊤(δHi)Avi. Hence

η 2

vi⊤A⊤(δHi)Avi .

δJ = Ex v∗⊤A⊤Aδvi − η vi⊤A⊤HiAδvi −

Now Hi = 0 1 ∇2R(θ0 − tηgi)dt. Since δ∇2R(θ) = ∇3R(θ)[·] and the evaluation point depends on gi, the chain rule yields

1

−tη ∇3R θ0 − tηgi [δgi]dt, with δgi = Ex[Aδvi]. Therefore

δHi =

0

1

η2 2

η 2

vi⊤A⊤(δHi)Avi =

t ∇3R θ0 − tηgi [δgi], Avi ⊗ Avi dt.

−

0

Taking Ex and using trilinearity in the last two slots, Ex⟨T [δgi],Avi ⊗ Avi⟩ = ⟨T [δgi], (ExAvi) ⊗ (ExAvi)⟩ = ⟨T [δgi], gi ⊗ gi⟩, with T := ∇3R(·), gives the stated third-order term.

Finally, the variation of vi with respect to fi via h is

##### δβi = T ⊙ q ⊙ h′(q), δSf

##### = ⟨T ⊙ q, h′(q)⟩, δvi = δβi − (δSf

##### )q = Diag(T ⊙ q) − q (T ⊙ q)⊤ h′(q).

i

i

Collecting terms yields the claimed formula. Corollary E.6. Define the gradient flow of the following term:

| |
|---|

R(θ0) − R(θ1(i)) η

R˙ (θt(i)) t=0 := lim

(6) Then we have

η↘0

R˙ (θt(i)) t=0 = Ex v∗⊤Φ⊤Φvi (7) Proof. By Taylor Expansion, we have

1

η2 2 ∇Lfi

R(θ0) − R(θ1(i)) = η⟨∇R(θ0),∇Lfi

(θ0)⊤

∇2R(θ0 − tη∇Lfi

(θ0) (8)

(θ0)⟩ −

(θ0))dt ∇Lfi

0

Then this corollary follows immediately from Lem. E.5. Lemma E.7 (Useful Inequalities). Let q ∈ ∆V −1 be a probability vector and fix an index j.

| |
|---|

##### 1. For all q, qj2 ∥ej − q∥2 ≤ 2qj2(1 − qj)2, (9)

and the bound is tight (equality holds) when all mass i̸=j qi = 1 − qj is concentrated on a single coordinate.

##### 2. For fixed distinct i ̸= j, consider F(q) := qiqj −qi − qj + ∥q∥2 .

Then

11√33 − 59

768 ≤ 0.00546, and the maximizer is attained by a vector with

max

F(q) =

q∈∆V −1

√33 24

9 −

, all remaining mass 1 − 2qi placed on one coordinate.

qi = qj =

##### 3. If we know −qi − qj + ∥q∥2 ≤ 0, then −qi − qj + ∥q∥2 ≤ 1 + 2(qi + qj)2 − 3(qi + qj)

Proof. (1) Since q is a probability vector with nonnegative coordinates,

2

##### = 2(1 − qj)2,

∥ej − q∥2 = (1 − qj)2 +

qk2 ≤ (1 − qj)2 +

qk

k̸=j

k̸=j

because k̸=j qk2 ≤ ( k̸=j qk)2 for nonnegative terms. Multiplying by qj2 yields Eq. 9. Equality holds when the entire mass 1 − qj lies on a single coordinate distinct from j, in which case k̸=j qk2 = ( k̸=j qk)2 = (1 − qj)2.

##### (2) Set a = qi, b = qj, and s = 1 − a − b ≥ 0. Write ∥q∥2 = a2 + b2 + t with t := k̸=i,j qk2. For fixed a,b, the objective F(q) = ab −a − b + a2 + b2 + t

is increasing in t whenever ab > 0. Since t ≤ s2 with equality iff all the mass s is concentrated on a single coordinate, any maximizer (with ab > 0) must satisfy t = s2 = (1 − a − b)2. Thus we may reduce to the two-variable problem

##### G(a,b) := ab −a − b + a2 + b2 + (1 − a − b)2 , a ≥ 0, b ≥ 0, a + b ≤ 1.

It is convenient to reparametrize by

u := a + b ∈ [0,1], z := (a − b)2 ∈ [0,u2]. Then

u2 + z 2

u2 − z 4

, (1 − a − b)2 = (1 − u)2, and a short calculation gives

, a2 + b2 =

ab =

1 4

1 4

(u2 − z) 1 − 3u + 32u2 + z2 =

(u2 − z) K(u) + z2 ,

G(u,z) =

where K(u) := 1 − 3u + 32u2. For each fixed u, G(u,z) is a concave quadratic in z (its z2-coefficient is −18). Hence the z-maximizer is

z⋆(u) = min max 0, u2 − 2K(u) , u2 = min max 0, −α(u) , u2 , where α(u) := u2 − 3u + 1. Equivalently,

 

√5 2 ),

0, α(u) ≥ 0 (i.e. u ∈ 0, 3−

√5 2 , 12 ),

z⋆(u) =

−α(u), α(u) ≤ 0 and u ≤ 12 (i.e. u ∈ 3−



u2, u ≥ 21. Thus:

- • If u ∈ 0, 3−

√5 2 , then z⋆(u) = 0, so the maximizer over z occurs at a = b = u2 (the symmetric point), and

G(u,0) =

u2 4

K(u) =

u2 4

1 − 3u + 23u2 .

- • If u ∈ 3−

√5 2 , 12 , then z⋆(u) = −α(u), and a simplification yields

(u − 1)2(2u − 1)2 8

G(u,z) = G u,z⋆(u) =

.

##### max

z

Since dud (u − 1)2(2u − 1)2/8 = 14(u − 1)(2u − 1)(4u − 3) < 0 on this interval, the maximum over u here is attained at the left endpoint u = 3−

√5 2 .

- • If u ∈ [12,1], then z⋆(u) = u2, which gives ab = 0 and hence G = 0.

√5 2 ]. In this

Therefore the global maximizer must lie in the symmetric regime z = 0, i.e., a = b = x, with u = 2x ∈ [0, 3−

case

G(x) = x2 6x2 − 6x + 1 , x ∈ 0, 12 . Differentiating,

G′(x) = 2x(12x2 − 9x + 1), so the critical point in (0, 21) satisfies 12x2 − 9x + 1 = 0, i.e.

√33 24 ∈ 0, 12 .

9 −

x⋆ =

Since G(0) = 0, G(12) = −18 < 0, and G achieves a positive value at x⋆, the global maximum is attained at x⋆. Substituting and simplifying,

11√33 − 59

768 ≤ 0.00546. This value is realized by

max

F(q) = G(x⋆) =

q∈∆V −1

qi = qj = x⋆, qℓ = 1 − 2x⋆ for some ℓ ∈/ {i,j}, qk = 0 (k ∈/ {i,j,ℓ}), i.e., the remaining mass is concentrated on a single coordinate, as established at the start.

- (3) We have that

−qi − qj + ∥q∥2 ≤ −qi − qj + qi2 + qj2 + (1 − qi − qj)2

= 1 + 2qi2 + 2qj2 + 2qiqj − 3qi ≤ 1 + 2(qi + qj)2 − 3(qi + qj)

| |
|---|

Theorem E.8 (Characterization via Gradient Flow, Restatement of Thm. 6.4). Under Assumptions E.1- E.3, suppose that f2′−f1′(˜q) is negative for all q˜and that qy˜ (f2′ − f1′)(qy˜) > −c for some small positive constant c > 0 when q(˜y) ∈ [0,0.55] and qy˜ (f2′ − f1′)(qy˜) < −d for some small positive constant d when q(˜y) ∈ [0.55,0.95] and that c < 10d, with an appropriate choice of label noise ( e.g., when y∗ ̸= y˜) rate E, then we have the following conclusions:

- • R˙ (θt(1)) t=0 ≥ R˙ (θt(2)) t=0 in Model Strong End.
- • R˙ (θt(1)) t=0 ≤ R˙ (θt(2)) t=0 in Model Weak End.

Proof. By Assumption. E.3, we first expand the following term:

Note that

R˙ (θt(1)) t=0 − R˙ (θt(2)) t=0 = Ex v∗⊤ (v1 − v2) (10) = Ex r⊤q q − r ⊙ q ⊤ (v12) (11)

[Tyqy (f1′ − f2′)(qy)]ey −

(Tyqy)(f1′ − f2′)(qy) q (12)

v12 =

y

y

= qy˜ (f1′ − f2′)(qy˜)ey˜ − qy˜ (f1′ − f2′)(qy˜)q (Only consider T one-hot) = qy˜ (f1′ − f2′)(ey˜ − q) (13)

We can then proceeed as follows:

R˙ (θt(1)) t=0 − R˙ (θt(2)) t=0 = Ex qy˜ (f2′ − f1′)(qy˜)⟨r ⊙ q − r⊤q q,ey˜ − q⟩ (14) = Ex [qy˜ (f2′ − f1′)(qy˜)⟨qy∗ − qy∗q,ey˜ − q⟩] (r is also one-hot) = Ex [qy˜qy∗ (f2′ − f1′)(qy˜)⟨ey∗ − q,ey˜ − q⟩] (15) = Ex qy˜qy∗ (f2′ − f1′)(qy˜)∥ey∗ − q∥2 : y˜ = y∗ (16) + Ex qy˜qy∗ (f2′ − f1′)(qy˜) −qy∗ − qy˜ + ∥q∥2 : y˜ ̸= y∗ (17)

Then we first examine the weak model end, now the model is assumed to output uniform distribution over V . Denote the label noise rate to be E. Then we have that

V − 1 V 3

R˙ (θt(1)) t=0 − R˙ (θt(2)) t=0 =

(f2′ − f1′)

1 V 3

(f2′ − f1′)

−

= (f2′ − f1′)

1 V

1 V

(1 − E) (18)

1 V E (19)

1 V 3

((V − 1)(1 − E) − E) < 0 (20)

As long as E < VV−1 and (f2′ − f1′) V 1 < 0. Then we have the desired condition. Then we examine strong model end, applying Lemma E.7, we have

##### Ex qy˜qy∗ (f2′ − f1′)(qy˜)∥ey∗ − q∥2 : y˜ = y∗ ≥ 2(1 − E)E (f2′ − f1′)(qy∗)qy2∗ (1 − qy∗)2 (21)

and define R = qy˜ (f2′ − f1′)(qy˜) and Q = qy˜qy∗ −qy∗ − qy˜ + ∥q∥2 , then first we show the other term is positive.

1 E

Ex qy˜qy∗ (f2′ − f1′)(qy˜) −qy∗ − qy˜ + ∥q∥2 : y˜ ̸= y∗ (22) = Ex [QR] (23) = Ex [QR: Q ≥ 0] + Ex [QR: Q < 0] (24) ≥ −cEx [Q: Q ≥ 0] + Ex [QR: Q < 0] (25) ≥ −cPr[Q ≥ 0] ∗ 0.00546 + Ex [QR: Q < 0] (26) > 0 (27)

For the last inequality, we can proceed as follows:

Ex [QR: Q < 0] − cPr[Q ≥ 0] ∗ 0.00546 ≥ d ∗ Pr

|Q| − cPr[qy˜ + qy∗ ≤ 0.50] ∗ 0.00546

[0.95 ≥ qy˜ + qy∗ ≥ 0.55] ∗ min

y,y˜ ∗

0.95≥qy˜+qy∗≥0.55

= d ∗ Pr

[0.95 ≥ qy˜ + qy∗ ≥ 0.55] ∗ 0.045 − cPr[qy˜ + qy∗ ≤ 0.50] ∗ 0.00546 > 0

y,y˜ ∗

where the first inequality comes from the sufficient condition for guaranteeing Q > 0 is Pry,y˜ ∗ [qy˜ + qy∗ > 0.50], and by

(3) in Lem. E.7, we have that given Q < 0,

1 + 2(qy˜ + qy∗)2 − 3(qy˜ + qy∗) ≤ 0.045

|Q| ≤ − max

min

0.95≥qy˜+qy∗≥0.55

0.95≥qy˜+qy∗≥0.55

Also by Assumpion. 6.1 and 6.2, we have Pry,y˜ ∗ [0.95 ≥ qy˜ + qy∗ ≥ 0.55] ≥ α Pr[qy˜ + qy∗ ≤ 0.50]. Therefore, we have finished the claim.

Therefore, with an appropriate scale of E, specifically with E > B|A−|A where B = Ex qy˜qy∗ (f2′ − f1′)(qy˜) −qy∗ − qy˜ + ∥q∥2 : y˜ ̸= y∗ > 0 and A = Ex qy˜qy∗ (f2′ − f1′)(qy˜)∥ey∗ − q∥2 : y˜ = y∗ < 0, then we could achieve the desired result.

| |
|---|

### F. Discussion with Existing Literature

#### F.1. Connections with RL

Our analysis is formulated entirely in the supervised SFT setting, where we study probability-based token-level objectives of the form

Lf(θ) = E(x,y˜)∼D f pθ(˜y | x) , (28)

on a fixed offline dataset D. Here, the training distribution is independent of the current model πθ, and coverage is entirely determined by D.

By contrast, RL methods optimize a sequence-level objective

θ(·|x) r(x,y) , (29)

J(θ) = Ex∼D,y∼π

where r(x,y) is a scalar reward and πθ is updated using online trajectories sampled from itself. Gradient estimates typically take the form

θ(·|x) A(x,y),∇θ log πθ(y | x) , (30)

∇θJ(θ) = Ex∼D,y∼π

for some advantage term A(x,y). Because y is drawn from πθ, most gradient mass comes from sequences and tokens that are already high probability under the current policy. This online nature naturally biases updates toward existing high-probability behaviors.

Recent work on RL for LLM reasoning (Davis & Recht, 2025) shows that, for binary correctness rewards, several popular RL-style post-training algorithms can be interpreted as stochastic gradient ascent on a monotone transform of the probability of producing a correct answer given a prompt. If we denote

pcorrθ (x) :=

then these algorithms optimize an objective of the form

πθ(y | x), (31)

y∈Ycorr(x)

Jh(θ) := Ex∼D h pcorrθ (x) , (32)

for some monotonically increasing function h(·) determined by the algorithm design. From our perspective, Eq. 32 is another instance of the probability-based family in Eq. 28, but applied at the sequence level and coupled to an on-policy sampling scheme.

Practically, RLHF/RLVR pipelines start from a strong base model—typically after extensive pretraining and sometimes a specialized midtraining phase—and include a KL penalty that keeps πθ close to this base policy (Ouyang et al., 2022; Shao et al., 2024). Combined with the on-policy gradient, this means that updates are dominated by already high-probability sequences, while low-probability ones receive very little gradient signal. This behavior is closely aligned with our priorleaning objectives in the model-strong regime: both favor trusting the pretrained prior when it is reliable. At the same time, RL-based methods come with their own challenges (e.g., exploration versus exploitation, reward misspecification) that are largely orthogonal to the off-policy SFT setting we focus on. A full theoretical unification of RLHF/RLVR with our capability-based continuum is beyond the scope of this work, but Eq. 29–32 highlight that many RL objectives can be naturally interpreted within the same probability-based lens developed here, and extending our framework to fully encompass on-policy RL settings is an exciting direction for future work.

#### F.2. Connections with Other Loss Functions

Existing work on alternative SFT losses can be broadly divided into two categories. Distribution-based and NonDistribution-based losses. Distribution-based operate directly on the (scalar) probability assigned to the correct label (or a set of correct sequences), and thus fit exactly into our probability-based family Lf(p), while the latter ones are typically composite objectives (e.g., sums of multiple terms, or set/sequence-level surrogates) that depend on the joint behavior of many tokens and do not reduce to a clean function of pθ(˜y | x).

#### Distribution-based Losses.

We first discuss distribution-based losses, which are the most fundamental and admit a clean characterization through the logit-gradient weight

Wf(p) := −f′(p),p(1 − p), (33)

as we established in Lemma 3.1. As illustrative examples, we analyze the Focal loss by Rege Cambrin et al. (2024) and a Huber-style loss on probabilities, and interpret both within our Wf(p) view.

Focal loss: a prior-averse example. Focal loss (Lin et al., 2017; Rege Cambrin et al., 2024) was introduced to address class imbalance by downweighting easy examples and emphasizing hard ones. For a single correct token with probability p ∈ (0,1), the (binary) Focal loss can be written as

fFL(p) = −(1 − p)γ log p, γ > 0. (34) A direct calculation yields

(1 − p)γ−1 p

fFL′ (p) =

γplog p − (1 − p) , (35) and therefore

(p) = −fFL′ (p)p(1 − p) = (1 − p)γ (1 − p) − γplog p . (36)

Wf

FL

Compared to NLL, whose weight is WNLL(p) = 1 − p, Focal loss multiplies this factor by (1 − p)γ and introduces the additional term −γplog p. For small p (hard, low-probability tokens), (1 − p)γ is close to one and the −γplog p term is positive and large, so Wf

(p) can substantially exceed WNLL(p). For p near one (easy, high-probability tokens), both (1 − p)γ and −γplog p are small, and the weight decays quickly. In our terminology, Focal loss is therefore more prioraverse than NLL: it further shifts gradient mass toward low-probability tokens and away from high-probability ones. This behavior aligns with its original motivation of focusing learning on rare or difficult examples, and fits naturally into the model-weak end of our continuum.

FL

Huber-style loss: a prior-leaning example. To illustrate a contrasting, more prior-leaning distribution-based objective, we consider a Huber-style loss applied to the probability of the correct token. Let e = 1 − p denote the error in the correct-class probability and δ ∈ (0,1] be a threshold. The Huber loss on e is

ϕδ(e) =

1 2e2, e ≤ δ

δ e − 12δ e > δ,

(37)

and we define the probability-based loss

fHuber(p) := ϕδ(1 − p). (38) For e = 1 − p ≤ δ (i.e., p close to 1), we have ϕ′δ(e) = e and thus

(p) = −fHuber′ (p)p(1 − p) = p(1 − p)2. (39) For e = 1 − p > δ (i.e., low-probability, high-error region), ϕ′δ(e) = δ is constant and

fHuber′ (p) = −(1 − p), Wf

Huber

fHuber′ (p) = −δ, Wf

(p) = δ p(1 − p). (40)

Huber

Compared to NLL, this Huber-style loss strongly downweights both very low- and very high-probability tokens: in the high-confidence region, the weight decays as p(1 − p)2, which is smaller than 1 − p for p close to 1; in the low-confidence region, the weight is capped at δp(1−p), which can be much smaller than 1−p when p is small. As a result, gradients are concentrated on moderately confident tokens rather than on extremely low-probability ones. In our framework, this makes the Huber-style loss a prior-leaning objective, more conservative than NLL in correcting tokens the model currently deems very unlikely, which aligns with regimes where the pretrained prior is already informative but supervision may be noisy.

#### Non-Distribution-based Losses.

Beyond purely distribution-based objectives, several recent works have proposed losses that depend on sets of tokens or on both the data and model-generated distributions. These are not of the simple form f(pθ(˜y | x)) and therefore fall outside the characterization by our paper, but they are still informative for our continuum view.

Dice and region-based losses. (Rege Cambrin et al., 2024) transfer semantic-segmentation losses (Dice, Generalized Dice, Lov´asz, Self-Adjusting Dice) to LLM fine-tuning and combine them with cross-entropy via

Ltot = λLCE + (1 − λ)Lseg, λ ∈ [0,1],

where LCE is applied to all instruction and answer tokens and Lseg is applied only to answer tokens.1For binary Dice, given predicted probabilities pi ∈ [0,1] and labels yi ∈ {0,1}, the Dice score and loss are

2 i piyi i p2i + i yi2

DS =

, LDice = 1 − DS. (41)

As noted by both Milletari et al. (2016) and Rege Cambrin et al. (2024), Dice-type losses depend on global set-level quantities (e.g., intersection and union over all tokens). Consequently, the gradient with respect to a single token logit couples all tokens through the numerator and denominator of the Dice score. As a result, isolated misclassified tokens can receive relatively small updates when the overall overlap between predicted and gold token sets is already high. These region-based objectives therefore fall outside our token-wise probability-based characterization via Wf(p): the effective weight on each token cannot be written as a function of its own probability alone, and it is not meaningful to classify them as globally “prior-leaning” or “prior-averse” in our sense.

Entropic distribution matching (GEM). (Li et al., 2025) propose GEM, an SFT method based on entropic distribution matching. Conceptually, they formulate a reverse-KL objective with entropy regularization

Ex Ey∼f(·|x) log p(y | x) − Ey∼f(·|x) log f(y | x) + γ Ey∼f(·|x) log f(y | x) , (42)

max

f

which is equivalent to minimizing a reverse KL divergence DKL(f∥p) minus an entropy regularizer (i.e., maximizing entropy). Here p denotes the (unknown) data distribution on sequences and f is the model’s generative distribution. Li et al. show that, at the population level, the optimal solution to equation 42 satisfies

f⋆(y | x) ∝ p(y | x)1/(γ+1), (43)

i.e., a strictly concave power transform of p that flattens peaked distributions and increases entropy. Thus, at the sequence level, GEM behaves as a strongly prior-leaning objective: it preserves the modes of p while explicitly avoiding

1See their Sec. 3.4 and Fig. 2 for the combined loss design.

over-concentrating probability mass on any single sequence, which leads to less overfitting and higher output diversity in practice.

Algorithmically, GEM is implemented via a composite generative loss that contrasts log-probabilities of supervised “real” sequences and model-generated “fake” sequences, with the fake distribution q defined as a softened version of f (via a temperature β). This composite objective cannot be written as a simple f(pθ(˜y | x)) on ground-truth tokens, so it lies outside the Wf(p) characterization we develop. Nevertheless, Eq. 43 shows that GEM effectively implements a concave, entropy-increasing transform of the underlying sequence-level probabilities and hence sits naturally on the prior-leaning, model-strong side of our continuum, complementary to the token-level objectives we focus on in this paper.

