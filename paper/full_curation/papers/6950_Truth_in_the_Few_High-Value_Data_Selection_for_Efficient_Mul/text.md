### arXiv:2506.04755v2[cs.CV]12Feb2026

## Truth in the Few: High-Value Data Selection for Efficient Multi-Modal Reasoning

Shenshen Li, Xing Xu Member, IEEE, Kaiyuan Deng, Lei Wang, Heng Tao Shen Fellow, IEEE, Fumin Shen

[Figure 1]

[Figure 2]

Abstract—While multi-modal large language models (MLLMs) have made significant progress in complex reasoning tasks via reinforcement learning, it is commonly believed that extensive training data is necessary for improving multi-modal reasoning ability, inevitably leading to data redundancy and substantial computational costs. However, can smaller high-value datasets match or outperform full corpora for multi-modal reasoning in MLLMs? In this work, we challenge this assumption through a key observation: meaningful multi-modal reasoning is triggered by only a sparse subset of training samples, termed cognitive samples, whereas the majority contribute marginally. Building on this insight, we propose a novel data selection paradigm termed Reasoning Activation Potential (RAP), which identifies cognitive samples by estimating each sample’s potential to stimulate genuine multi-modal reasoning by two complementary estimators: 1) Causal Discrepancy Estimator (CDE) based on the potential outcome model principle, eliminates samples that overly rely on language priors by comparing outputs between multi-modal and text-only inputs; 2) Attention Confidence Estimator (ACE), which exploits token-level self-attention to discard samples dominated by irrelevant but over-emphasized tokens in intermediate reasoning stages. Moreover, we introduce a Difficulty-aware Replacement Module (DRM) to substitute trivial instances with cognitively challenging ones, thereby ensuring complexity for robust multimodal reasoning. Experiments on six datasets show that our RAP method consistently achieves superior performance using only 9.3% of the training data, while reducing computational costs by over 43%.

Qwen2.5-VL-7b

+2.5% accuracy -43% time cost

20%datayields only0.8% performance drop

(a) (b)

Fig. 1. Comparison of (a) accuracy under varying training dataset sizes and (b) performance–efficiency trade-offs on various methods.

training costs. Recent studies [8], [9] indicate that LLMs trained on high-quality curated datasets can outperform those trained on full corpora. However, it remains unclear whether this principle generalizes to multi-modal contexts, where effective cross-modal integration is important. This raises a critical question: can smaller high-value data achieve competitive or superior multi-modal reasoning compared to training on full corpora in MLLMs? To investigate this, as shown in Fig. 1(a), we empirically analyze the effect of data scale on multi-modal reasoning performance. Notably, training with only 20% of the data leads to merely a 0.8% performance degradation compared to the full dataset, suggesting that indiscriminate data scaling may have minimal or even negative effects. We hypothesize that such data augmentation diminishes the influence of high-value samples, termed cognitive samples, which are essential for guiding effective cross-modal integration during reasoning.

Index Terms—Large Vision-Language Model, High-Value Data, Efficient Multi-modal Reasoning

I. INTRODUCTION

# I

Mproving complex reasoning in multi-modal large language models (MLLMs) [1], [2] remains a fundamental

To validate this assumption, we analyze the characteristics of training samples and find that most fail to encourage joint attention to both modalities during reasoning. Specifically, we identify two main types of ineffective samples: 1) Languageprior biased samples (Fig. 2(a)), where the model produces nearly indistinguishable outputs given text-only and multimodal inputs due to over-reliance on language priors [10], [11]. Such samples enable models to solve tasks with minimal utilization of visual semantics, thus impairing their ability to cross-modal integration. (2) Attention-biased samples (Fig. 2(b)), where the model over-attends to semantically irrelevant tokens (e.g., the punctuation “.”), thereby obstructing the exploration of crucial cross-modal relationships. The above findings highlight the need to prioritize cross-modal interactions in data selection for multi-modal reasoning. However, existing methods rely on unimodal textual quality, such as human-annotated difficulty estimation [8], [9] or reward-based sampling [12]. These approaches not only incur substantial manual annotation costs [8], [12] and considerable filtering

challenge. While large-scale reinforcement learning (RL) [3]– [5] has shown promise in enhancing reasoning capability, the prevailing assumption [6], [7] suggests that scaling training data is a necessary condition for developing advanced reasoning ability, thus leading to data redundancy and substantial

This work was supported by the National Natural Science Foundation of China under Grants (No. 62476201), Chengdu Science and Technology Program (Grant No. 2023-XT00-00001-GX), Chengdu Science and Technology Bureau (Grant No. 2025ZDZX0008), the New Cornerstone Science Foundation through the XPLORER PRIZE (Corresponding author: Fumin Shen.)

Shenshen Li, Kaiyuan Deng and Fumin Shen are with the School of Computer Science and Engineering, University of Electronic Science and Technology of China, Chengdu 611731, China. (E-mail: lishenshen727@gmail.com, mantou.cloud@gmail.com, fumin.shen@gmail.com).

L. Wang is with MiroMind in Singapore 408621 (E-mail: lei.wang.2019@phdcs.smu.edu.sg).

X. Xu and H. T. Shen are with the School of Computer Science and Technology, Tongji University, Shanghai 201804, China. (E-mail: interxuxing@hotmail.com, shenhengtao@hotmail.com).

Question: In the image, a paper sector (arc length=6π, angle=120°) is folded into a cone. What is the height of the conical paper hat?

Question: In parallelogram ABCD, point E divides CD with DE:EC = 2:3. Lines AE and BD intersect at F. Find area ratio : .

Text-only: the height of conical paper hat is 6√ .

###### Overemphasis on non-informative token “.’’

Wrong answer: ABF. Consider the diagonal AC and BD of the ABCD. Hence, F is the midpoint of BD. The ratio is 1/2.

[Figure 3]

[Figure 4]

ABF

Multimodal: the height is 6√ .

Minimal difference in outputs

Self-attention matrix

the diagonals

ABF

(a) Language-prior biased samples

(b) Attention-biased samples

- Fig. 2. Illustrative examples for two ineffective training sample types: (a) language-prior biased samples and (b) attention-biased samples.

time [9], but also fail to estimate whether samples effectively facilitate cross-modal integration.

- • We propose two novel estimators: a Causal Discrepancy Estimator (CDE) to eliminate samples that overly rely on language priors, and an Attention Confidence Estimator (ACE) to filter out attention-biased samples with irrelevant semantic focus.
- • We introduce a Difficulty-aware Replacement Module (DRM) to preserve sufficient data complexity, effectively improving the model’s reasoning performance ceiling.

Motivated by the above observations, we propose a novel data selection paradigm termed Reasoning Activation Potential (RAP) for enhancing multi-modal reasoning while reducing training costs. RAP aims to identify cognitive samples that effectively trigger multi-modal reasoning during RL training. Specifically, RAP estimates the reasoning potential of each sample through two complementary perspectives: outputlevel reasoning discrepancy and process-level reasoning confidence. For the former, we are inspired by the intuition that if model predictions remain invariant regardless of visual input presence, the model may merely exploit linguistic biases rather than engaging in genuine multi-modal reasoning. We formalize this notion through Causal Discrepancy Estimator (CDE), which employs the Potential Outcome Model (POM) to estimate the causal effect of input modality on model predictions by simulating counterfactual outcomes, i.e., what the model would output if one modality were removed. Consequently, CDE effectively eliminates language-prior biased samples by measuring discrepancies between multi-modal and text-only predictions.

II. RELATED WORK

- A. Reinforcement Learning for (M)LLM Reasoning

Reinforcement learning (RL), initially popularized by RL from Human Feedback (RLHF) [4], [13], has become a key paradigm for improving the reasoning ability of both LLMs [14] and MLLMs [15], [16]. Recent methods [17]–[19] extend RL beyond human preference alignment, explicitly achieving reasoning improvement via policy-gradient algorithms such as Proximal Policy Optimization (PPO) [20] and reward-centric optimizations including Reinforce Leave-One-Out (RLOO) [21] and Group Relative Policy Optimization (GRPO) [3]. Moreover, several methods [5], [22]–[24] explore the use of RL to enhance the visual reasoning in MLLMs. Recent works have extended RL beyond alignment for enhancing reasoning capability, by utilizing policy gradient algorithms PPO [20] and optimizing reward signals, such as RLOO [21] and GRPO [3]. However, these methods typically rely on large-scale data [25], [26], which fails to adequately consider the quality of training samples. To this end, we introduce a novel RAP method to select valuable samples, ensuring that the training process stimulates effective multi-modal reasoning.

- B. Data Selection for Reasoning

However, relying solely on output-level measures neglects the reliability of internal reasoning dynamics. Therefore, we propose the Attention Confidence Estimator (ACE) to model the quality of internal reasoning behavior based on token-level attention distributions, thus removing attention-biased samples characterized by high attention to irrelevant tokens. Despite their efficacy, combining these two estimators alone may retain overly simplistic samples while discarding challenging yet valuable instances, thereby constraining the model’s reasoning upper bound. To address this limitation, we propose a Difficulty-aware Replacement Module (DRM) to replace trivial samples with suitable challenging alternatives, which ensure sufficient data complexity for robust multi-modal reasoning. Finally, the results in Fig. 1(b) demonstrate that our RAP method achieves state-of-the-art performance with only 5,159 samples, compared to the full dataset of 54,931 samples, while reducing training costs by over 43%. These findings validate our insight that data quality is more important than blind data scaling for multi-modal reasoning in RL, revealing the “truth in the few” phenomenon.

Traditional approaches [3], [7], [25], [27] generally highlight the importance of data scaling, suggesting that larger data volumes can lead to better model performance. In contrast, recent methods [8], [9], [12] demonstrate that carefully curated datasets can outperform models trained on full corpora for challenging textual reasoning tasks. For example, LIMO [12] and s1 [8] show that models trained with only 817 or 1000 curated samples can achieve better performance than those trained with substantially larger datasets, indicating that sample quality and representativeness may be more decisive than sheer quantity. Similarly, LIMR [9] suggests that selective data curation during RL training can further improve outcomes with fewer samples by prioritizing instances that provide clearer

Overall, our contributions can be summarized as follows:

• We reveal a “truth in the few” phenomenon that smaller high-quality datasets can outperform full corpora for multi-modal reasoning in LVLMs.

##### Casual Discrepancy Estimator

##### Attention Confidence Estimator Difficulty-aware Replacement Module

 =1

Prompt + Question: As shown in the figure, ..., if ∠O = 110.0, then the degree of ∠C is ?

( )

Overemphasis on non-informative token :

Promt + Question: As shown in the figure, ..., if ∠O = 110.0, then the degree of ∠C is ?

1

CognitiveSamples

=

 ( )

[Figure 5]

= 0

###### Groundtruth: 125°

- Output 1

- Output 2

Output M

- Output 3

- Output 1

- Output 2

Output M

- Output 3

[Figure 6]

As shown in the figure,

Image

D

by : ∠C

=

<img>+ prompt

<img>+ prompt

... if ∠O = 110.0,

Self-attention

C

0(  )

D

Unreliable reasoning: <think> ... at the circumference is given by : ∠C = 1/2 ∠O. Therefore, we can calculate ∠C = 55°. </think>

[Figure 7]

O

C

Easy samples

Difficult samples

A B

O

A B

- Fig. 3. The overall pipeline of our RAP method. First, the Causal Discrepancy Estimator (CDE) filters out samples that overly rely on language priors via output-level discrepancy. Then, the Attention Confidence Estimator (ACE) excludes attention-biased samples by token-level attention distributions. Finally, the Difficulty-aware Replacement Module (DRM) selectively replaces trivial instances with cognitively challenging ones, yielding a refined subset of cognitive samples.

optimization signals. Inspired by these, we focus on whether and how a minimal but valuable dataset can similarly enhance multi-modal reasoning within multi-modal contexts.

C. Causal Mechanism

Recent advances have highlighted the role of causal mechanisms in understanding and improving complex reasoning [28]–[31]. To capture causal relationships in MLLM reasoning, prior studies have leveraged established causal inference frameworks, such as Structural Causal Models (SCM) [32] and Potential Outcome Models (POM) [33]. Compared with purely correlational analyses, these frameworks facilitate intervention-based comparisons that help isolate factors genuinely responsible for a model’s decisions. In particular, POM provides a rigorous analytical structure to examine causal dependencies [28], [34], enabling the quantification of how an explicit change in input conditions affects model outputs. Therefore, we adopt the POM framework to systematically measure differences in MLLM predictions under multi-modal inputs versus text-only inputs, thereby assessing the causal contribution of visual information to the reasoning process.

III. METHOD Overview of RAP Method. As shown in Fig. 3, our RAP method aims to identify high-value training samples xcd, termed cognitive samples, that effectively stimulate multimodal reasoning in MLLMs during RL training. Given a training instance x = (xt,xv), we estimate its reasoning activation potential from two perspectives. First, we adopt the potential outcome model to quantify the output-level discrepancy D between the model predictions under multi-modal inputs, Y1(x), and text-only inputs, Y0(xt). Samples with low discrepancy values be considered as language-prior biased and are discarded accordingly. Second, we compute a confidence score ψ(A) from the self-attention matrix A ∈ Rd×d of the model’s final layer to assess the model’s focus on semantically meaningful tokens. Samples with attention focused on irrelevant tokens, below a threshold λa, are excluded as attention-biased. Moreover, we replace overly easy samples with an equal number of hard examples potentially missed by

the initial model due to limited reasoning capacity, thereby enhancing the model’s reasoning upper bound. The resulting cognitive samples form a refined training dataset that supports more efficient and robust multi-modal reasoning in MLLMs. Finally, we utilize these cognitive samples xcd to optimize the model by maximizing the objective of GRPO [3], [35].

A. Causal Discrepancy Estimator

To identify samples where the model genuinely engages in multi-modal reasoning, rather than overly relying on language priors, we interpret modality influence in reasoning as output discrepancy between multi-modal and text-only inputs, formulated under the Potential Outcome Model (POM).

Background of Potential Outcome Model. The foundation of causal inference is rooted in the Neyman-Rubin POM [33], [36], which aims to estimate the effect of a treatment T on an outcome Y for individuals described by covariates X. In this work, we consider the treatment variable T to be binary, i.e., T ∈ {0,1}. Under this framework, each unit u is associated with two potential outcomes: Y1(u), the outcome if the unit receives the treatment (T = 1), and Y0(u), the outcome under control (T = 0). The Individual Treatment Effect (ITE) [37], [38] is defined as the difference Y1(u)−Y0(u). However, due to the fundamental problem of causal inference [39], [40], only one of these outcomes can ever be observed for a given individual, rendering the ITE fundamentally unidentifiable. To address this, the prior work [41] proposes the Conditional Average Treatment Effect (CATE), which represents the expected treatment effect conditioned on covariates:

E[Y1−Y0 | X = x] = E[Y | T = 1,X = x]−E[Y | T = 0,X = x],

(1) where x denotes the observed covariates of the unit.

Output-level Discrepancy Calculation. Inspired by the intuition that a model generating nearly identical outputs in the presence or absence of visual input may fail to use multi-modal information for reasoning, we employ the POM to formalize the influence of the visual modality on model predictions, by defining outcomes under distinct treatment conditions. Specifically, we treat the presence of the image

Algorithm 1 The Process of Reasoning Activation Potential (RAP) Method.

Require: Original dataset D = {(xit,xiv,yi)}Ni=1, pretrained model θ, output discrepancy threshold λc, attention confidence threshold λa, training steps T

Ensure: Curated cognitive samples xcd Stage 1: Causal Discrepancy Estimator (CDE)

- 1: for all (xt,xv,y) ∈ D do
- 2: Generate M outputs with multimodal input (xt,xv) and text-only input (xt)
- 3: Compute output consistency:
- 4: D(x) = M1 Mj=1 I Y1(j) = y − I Y0(j) = y

- 5: end for
- 6: Form discrepancy-filtered set:
- 7: Xcde = {x | D(x) ≥ µD + λcσD } Stage 2: Attention Confidence Estimator (ACE)
- 8: Initialize Xace ← ∅
- 9: for all x ∈ Xcde do
- 10: Run a forward pass and extract final-layer selfattention A ∈ RL×L
- 11: Compute attention confidence:
- 12: ψj(A) = Li=j(σ · Ai,j), ψ(A) = maxj ψj(A)
- 13: if ψ(A) ≤ λa then
- 14: Xace ← Xace ∪ {x}
- 15: end if
- 16: end for Stage 3: Difficulty-aware Replacement Module (DRM)
- 17: for all x ∈ Xace do
- 18: Compute difficulty:
- 19: Ddiff(x) = 1 − M1 Mj=1 I output(j) is correct

- 20: end for
- 21: Identify trivial samples:
- 22: Xeasy = {x | Ddiff(x) = 0}
- 23: Select top |Xeasy| challenging samples from D \ Xace by descending difficulty to form Xhard
- 24: Construct final cognitive dataset:
- 25: xcd = (Xace \ Xeasy) ∪ Xhard Stage 4: Cognitive Dataset Training
- 26: Train θ on xcd via reinforcement learning algorithm (GRPO) for T steps

as a binary treatment variable T ∈ {0,1}, where T = 1 indicates the inclusion of the image and T = 0 denotes its absence. Given an input x = (xt,xv), we define two potential outcomes: Y1(x), the model’s output given both text and image, and Y0(xt), the counterfactual output when only the text is provided. The text-only output Y0(xt) can be calculated as follows:

- yi ∈ Y0(xt) ∼ softmax[logθ (yi | xt)], (2)

The multi-modal output Y1(x) can be calculated as follows:

- yi ∈ Y1(x) ∼ softmax[logθ (yi | xv,xt)], (3)

To quantify the discrepancy between model outputs Y0(xt) and Y1(x) under multi-modal and text-only inputs, we compute the consistency of these model outputs with the ground

truth Yg. If the model’s output matches the ground truth in a given condition, we assign a value of 1; otherwise, a value of 0. The discrepancy D(x) for each sample x is then quantified as the normalized difference in the number of correct predictions between these conditions, which can be formulated as:

D(x) = E[I(Y1 = Yg) − I(Y0 = Yg) | x]

M

1 M

I Y1(x(i)) = Yg(i) − I Y0(x(ti)) = Yg(i) ,

=

i=1

(4) where M is the number of rollout outputs generated for each sample set in GRPO, and I(·) is the indicator function that equals 1 if the condition is true, and 0 otherwise. Based on the mean and standard deviation of discrepancies across all samples, we set a threshold for the discrepancy score. Specifically, samples with discrepancy less than µc + λc · σc, where µc is the mean discrepancy, σc is the standard deviation, and λc is a tunable hyperparameter, are excluded from the training set.

B. Attention Confidence Estimator

While the CDE effectively identifies samples requiring multi-modal reasoning from an output-level perspective, it does not assess the quality of internal reasoning processes. Recent studies [42], [43] reveal an insightful phenomenon: tokens that receive excessive attention weights can dominate the prediction without using a meaningful semantic context. Motivated by this, we explicitly quantify the internal reasoning quality via self-attention distributions, thus filtering out attention-biased samples.

Attention Confidence Formulation. Given an input x = (xt,xv) to a transformer-based MLLM, we denote the selfattention matrix A ∈ Rd×d from its final transformer layer as:

QiK⊤ √ j

, (5)

Ai,j = softmax

d

where Qi,Kj ∈ Rd represent the query and key vectors for token positions i and j.

To systematically characterize attention-bias patterns, ACE analyzes the entire self-attention matrix A. An attention-biased pattern at token position j is identified if the corresponding attention column exhibits a pronounced concentration of attention weights, identifying excessive reliance on a single token. Formally, the degree of attention bias at position j, ψj(A) is quantified by computing a multiplicative attention score across subsequent token interactions:

L

ψj(A) =

(σ · Ai,j), (6)

i=j

where σ is a scaling factor ensuring numerical stability and emphasizing prominent attention patterns. L denotes the total length of the input sequence. The ψj(A) metric effectively quantifies attention confidence, with elevated values indicating unreliable multi-modal reasoning. We define a position j as attention-biased if ψj(A) exceeds a threshold λa. Instances containing more than one attention-biased position are filtered out as attention-biased samples.

TABLE I COMPARISON WITH STATE-OF-THE-ART METHODS. EXPERIMENTS ARE CONDUCTED USING THE QWEN2.5-VL-3B [1] AND QWEN2.5-VL-7B [1], EMPLOYING GRPO AS THE RL METHOD. “TIME” DENOTES THE TOTAL COMPUTATION COST, INCLUDING DATA SELECTION AND TRAINING. BOLD FONT DENOTES THE BEST RESULT.

|Method|Sample|Time (h) ↓ MathVista ↑ MMStar ↑ MathVerse ↑ WeMath ↑ MMVet ↑ LogicVista ↑ Avg. ↑<br><br>|
|---|---|---|
|Qwen2.5-VL-7b Qwen2.5-VL-7b-Full Qwen2.5-VL-7b-s1 [8] (2025) Qwen2.5-VL-7b-LIMO [12] (2025) Qwen2.5-VL-7b-LIMR [9] (2025) Qwen2.5-VL-7b-RAP (Ours)<br><br>|54,931 1,000<br><br>4,093 8,136<br><br>5,159<br>|- 68.70 56.07 39.31 35.90 59.13 44.52 50.61 93.2 70.70 61.53 48.43 38.67 60.51 46.09 54.32 55.9 68.50 61.80 45.79 35.05 61.05 45.86 53.01<br><br>111.9 69.90 61.33 45.74 34.67 59.08 44.74 52.58 122.0 71.10 62.12 48.02 41.21 62.86 45.81 55.19<br><br>52.8 73.20 62.53 48.65 42.00 63.31 46.53 56.04<br><br>|
|Qwen2.5-VL-3b Qwen2.5-VL-3b-Full Qwen2.5-VL-3b-s1 [8] (2025) Qwen2.5-VL-3b-LIMO [12] (2025) Qwen2.5-VL-3b-LIMR [9] (2025) Qwen2.5-VL-3b-RAP (Ours)<br><br>|54,931<br><br>1,000<br><br>2,679<br><br><br>21,303 4,374<br><br>|- 61.30 54.46 9.01 21.62 51.81 39.59 39.63 46.5 64.50 55.25 38.35 28.29 53.41 40.03 46.64 38.4 62.60 54.53 37.41 27.52 53.02 39.59 45.78<br><br>120.5 61.80 54.73 35.33 24.48 53.12 39.14 44.77 60.9 63.10 54.66 35.43 26.76 53.25 40.95 45.69 32.0 64.90 55.67 39.34 29.33 54.63 41.61 47.58<br><br>|

C. Difficulty-aware Replacement Module

Finally, by filtering through CDE and ACE and refining with DRM, we ensure that the model is trained with cognitive samples xcd, thereby enhancing the multi-modal reasoning ability, while simultaneously reducing training costs and data redundancy.

Although CDE and ACE select valuable samples, they inevitably limit the reasoning upper bound due to the exclusion of challenging yet informative samples. For example, in the CDE selection process, if the output discrepancy threshold is set above 0.2, a scenario may arise where a text-only model consistently produces incorrect outputs across all five trials, whereas multi-modal outputs succeed once in five trials. Such challenging yet valuable samples, which could significantly contribute to improving reasoning, are thus discarded. This exclusionary process reduces the complexity of the training data, thereby constraining the model to achieve more complex reasoning ability.

IV. EXPERIMENTS

Training dataset. Main results in the Table I are based on models trained with the MM-Eureka dataset [26], a highquality multi-modal dataset for mathematical reasoning. To further validate the generalization of our RAP method, we evaluate models trained on the subset of Mulberry-260k dataset [25], a multi-modal learning-to-reason-and-reflect dataset.

To address this, we introduce a Difficulty-aware Replacement Module (DRM) to refine the selected sample. First, we define the difficulty score Diffi to quantify the challenge of correctly answering a sample, which can be defined as:

Evaluation. Similar to [25], [26], we evaluate models on both mathematical and general multi-modal reasoning tasks using the pass@1 metric, where pass@1 measures the percentage of problems correctly solved on the first attempt, under a zero-shot setting. For mathematical reasoning, we assess the model’s ability on four benchmark datasets: MathVista [44], MMStar [45], MathVerse [46], and WeMath [47]. For universal multi-modal reasoning, we evaluate on MMVet [48] and LogicVista [49].

M j=1 ci,j

Diffi = 1 −

, (7)

M

where ci,j denotes the correctness of the j-th rollout generation for the i-th sample, and M is the total number of outputs in the

group. A higher Diffi indicates greater difficulty. In particular, the DRM involves two steps: First, we exclude easy samples,

Implementation details. Following prior methods [25], [26], we conduct our primary experiments on mathematical reasoning tasks, employing Qwen2.5-VL-3B and Qwen2.5-VL-7B [1] as baseline models. First, we apply RAP to select cognitive samples using the initial model without any training. These samples are then used to train models within the EasyR1 [50], employing the AdamW optimizer with a learning rate of 1e6. Full-data training requires 1 epoch (107 steps), all others use 40 training steps with a batch size of 512 across 8 GPUs. The system prompt is available at our supplementary material. For accelerated generation in GRPO [3], we utilize the vLLM package [51]. Finally, we set the hyperparameters σ to 2.0, λc and λa to 0.5 and 0.1 for the CDE and ACE, respectively.

denoted as Diffi = 0, which are characterized by consistent correct answers across all trials. Second, we reintroduce hard

samples that have been previously discarded due to difficulty but are still valuable for training. Specifically, based on the difficulty score Diffi , the set of reintroduced samples Shard is given by:

1 M

,1), I(ψi(A) > λa) = 0} ,

Shard = argmaxk {xi | Diffi ∈ [

(8) where k is the number of easy samples. This formulation identifies the top-k most challenging samples according to the difficulty metric Diffi , while excluding those that do not meet the specified criteria. Note that the hardest samples, i.e., Di = 1, would be neglected, as they are demonstrated to be meaningless for training [35]. This DRM can enhance the upper bound of the model’s ability to handle complex tasks, without introducing data redundancy and training costs.

A. Overall Comparison Results

Comparing methods. We compare our approach with existing data selection methods, including: 1) s1 [8], which utilizes the large-scale MLLMs to identify high-quality data; 2) LIMO

- TABLE II COMPARISON WITH STATE-OF-THE-ART METHODS USING QWEN2.5-VL-7B [1] WITH THE RL METHOD RLOO [21], EVALUATING OUR RAP METHOD ON DIFFERENT TRAINING DATASETS AND RL ALGORITHMS.

|Method<br><br>|MathVista MMVet We-Math Avg.|
|---|---|
|Qwen2.5-VL-7b Qwen2.5-VL-7b-Full Qwen2.5-VL-7b-s1 [8] Qwen2.5-VL-7b-LIMO [12] Qwen2.5-VL-7b-LIMR [9] Qwen2.5-VL-7b-RAP (Ours)|68.70 59.13 35.90 54.58<br><br>69.10 60.32 36.95 55.46<br><br><br>68.50 59.96 35.06 54.51 68.80 60.11 35.23 54.71<br><br>68.90 60.71 36.74 55.45<br><br>69.20 61.33 37.05 55.86<br>|

- TABLE III COMPARISON WITH STATE-OF-THE-ART METHODS USING INTERNVL3-2B [52] WITH THE RL METHOD GRPO [3], EVALUATING OUR RAP METHOD ACROSS DIFFERENT MODEL ARCHITECTURES.

|Method<br><br>|MathVista MMVet We-Math Avg.|
|---|---|
|InternVL3-2b InternVL3-2b-Full InternVL3-2b-s1 [8] InternVL3-2b-LIMO [12] InternVL3-2b-LIMR [9] InternVL3-2b-RAP (Ours)|56.10 58.22 12.06 42.13<br><br>57.20 59.86 12.84 43.30<br><br><br>56.80 60.47 11.63 42.97<br><br>56.70 59.82 11.92 42.81<br><br>57.10 61.33 12.44 43.62<br><br><br>57.40 62.02 13.05 44.16<br>|

[12] that designs a difficulty-aware selection method to identify crucial samples; and 3) LIMR [9], which employs learning impact measurement to select a subset of training samples. Moreover, we also evaluate models trained on the full dataset (Full) as the baseline.

Comparisons with state-of-the-art methods. We compare our RAP model with the latest data-efficient methods on six diverse datasets, including LIMO [12], s1 [8] and LIMR [9]. The results shown in Table I reveal several key findings: 1) RAP consistently outperforms models trained with full corpora on all datasets. Remarkably, these improvements are achieved using only 9.5% or 7.9% of training data while reducing training time by 43% or 31%, supporting our hypothesis “truth in the few” that selected cognitive samples can achieve more effective multi-modal reasoning. 2) Moreover, the RAP method demonstrates a significant improvement of 7.33% and 6.95% over the LIMO [12] and s1 [8] methods on WeMath [47], respectively, which rely on data quality and manual selection. Such results indicate that focusing on the potential of each sample to improve multi-modal reasoning.

#### Effectiveness of RAP on different base models. As shown in

- Table II, our method consistently surpasses other recent data selection methods when applied to the base model InternVL32b [52]. This outcome highlights the broad applicability and generalizability of the RAP framework, as the introduced CDE and ACE components effectively select training samples that activate the model’s multi-modal reasoning capability. Crucially, these components do not rely on exploiting modelspecific inductive biases, thereby ensuring RAP’s adaptability on a wide range of model architectures. More results are available in our supplementary material. Effectiveness of RAP for various RL methods and training datasets. To further validate the generalizability of RAP,
- Table III presents results using the Qwen2.5-VL-7B model,

TABLE IV ABLATION STUDY OF RAP USING QWEN2.5-VL-7B, TRAINED USING GRPO METHOD FOR 40 STEPS.

|No.|CDE ACE DRM<br><br>|MathVista MMStar MathVerse MMVet|
|---|---|---|
|0<br>1<br>2<br>3<br>4<br>5<br>6<br>|- - -<br><br>✔<br><br>✔<br><br>✔ ✔<br><br>✔ ✔<br><br>✔ ✔<br><br>✔ ✔ ✔<br><br><br>|69.10 59.32 46.07 58.91<br>70.80 60.64 47.13 60.92<br><br><br>70.20 60.21 46.86 60.68 72.00 61.28 47.82 61.74<br>71.50 60.93 47.90 61.33<br>72.60 61.76 48.74 62.78<br>73.20 62.53 48.65 63.31<br>|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Fig. 4. Cross-model generalization of cognitive samples selected by RAP. Performance with InternVL3-2B trained on samples from Qwen2.5-VL-3B (left), and vice versa (right).

trained under two different configurations: 1) the RLOO RL paradigm [21], and 2) the reduced Mulberry-10K dataset, a subset of Mulberry-260K [25]. Despite these variations, RAP maintains consistent superiority compared to other methods, suggesting its generalization to different RL strategies and training datasets. We attribute this robustness to cognitive samples that facilitate genuine multi-modal reasoning rather than simply fitting data distributions.

B. Further Analysis

Ablation study. As presented in Table IV, we list the following conclusions: 1) The comparison between No.0 and No.3 indicates that integrating CDE and ACE can improve multi-modal reasoning in MLLMs. These results underscore the efficacy of RAP in eliminating language-prior biased and attention-biased samples, thereby enabling models to focus on essential “cognitive samples” and improving reasoning performance. 2) Comparing No.1 with No.3 shows that only using the CDE can improve performance, but worse than the full RAP. We hypothesize the reason is that while CDE identifies critical samples by detecting output discrepancies, it overlooks the intermediate reasoning process. 3) Moreover, the comparison between No.2 and No.3 demonstrates that the RCC can further refine the reasoning performance by replacing easy samples with more appropriate hard samples. Such improvements are due to that the DRM addresses a limitation of the CDE and ACE approaches, i.e., the tendency to retain simpler instances while neglecting challenging yet informative samples.

Comparison on cross-modal reasoning utilization. We evaluate the effectiveness of cognitive samples in enhancing crossmodal reasoning utilization on multi-modal reasoning tasks. We define cross-modal reasoning utilization as the proportion

[Figure 8]

72 70 68

72

70

68

(b) Effect of varying hyperparameters and

(c) Comparison of RAP with baseline and the latest method LIMR [8] on cross-modal reasoning utilization

(a) Distribution of output discrepancy

- Fig. 5. (a) Visualization of output discrepancies between multi-modal and text-only inputs on the full MM-Eureka training dataset. (b) Performance variation with respect to the hyperparameters λa and λc on MMstar. (c) Comparative analysis of multi-modal reasoning utilization on four datasets.

of instances in the MMStar dataset where the model correctly uses multi-modal inputs to answer questions, but fails when relying solely on textual inputs. As shown in Fig. 5(c), models trained with cognitive samples show superior integration of cross-modal information, compared to baseline and the latest method LIMR [9]. Such results highlight that the proposed CDE can effectively reduce models’ reliance on superficial linguistic prior by filtering out training samples exhibiting excessive language bias. Hence, models are encouraged to discover the relationship between image and text, thus improving the cross-modal reasoning utilization.

TABLE V PERFORMANCE COMPARISON BETWEEN RAP AND RAP-MID USING QWENVL2.5-3B WITH THE GRPO RL ALGORITHM ACROSS THREE BENCHMARKS. RAP-MID REFERS TO THE VARIANT THAT UTILIZES INTERMEDIATE (6TH-LAYER) ATTENTION MAPS.

|Method<br><br>|MathVista MMVet LogicVista WeMath Avg.|
|---|---|
|QwenVL2.5-3b-RAP-mid Qwen2.5VL-3b-RAP (Ours)<br><br>|64.60 54.29 41.74 29.17 47.45 64.90 54.63 41.61 29.33 47.62|

tention allocation across subsequent token sequences. Existing research [28] has demonstrated that attention distributions in the final layer effectively capture causal reasoning behaviors. Such findings intuitively highlight the critical role of the lastlayer attention maps in evaluating the reliability of the model’s reasoning behavior. Inspired by this, we initially chose the last-layer attention distributions to effectively capture globally biased attention patterns that compromise genuine multi-modal reasoning.

Comparison on cross-model generalization. To examine the cross-model generalizability of cognitive samples identified by RAP, we evaluate whether cognitive samples obtained using the Qwen2.5-VL-3b are useful for improving the reasoning of a distinctly structured model, InternVL3-2b, and vice versa. Comparative results presented in Fig. 4 demonstrate that cognitive samples selected by our RAP method outperform the latest method LIMR [9], confirming the generalization of RAP in enhancing multi-modal reasoning for varying model frameworks. Note that we exclude comparisons with LIMO [12] due to its similarity with s1 [8].

However, we have conducted preliminary experiments to investigate the potential of attention confidence computed from intermediate layers. Recent work [53] has indicated layers 5 to 8 within large language models play a critical role in multi-modal fusion. Therefore, we selected the 6th transformer layer’s attention distribution as an alternative input for our ACE module and compared its efficacy against our original last-layer choice. The results presented in Table V demonstrate slight performance degradation when utilizing intermediate (6th-layer) attention maps. For most datasets, employing attention maps from the last layer consistently yielded superior performance. These results justify our choice of using last-layer attention maps.

Analysis on hyperparameter sensitivity. We further investigate the sensitivity of the hyperparameters λc and λa employed within the CDE and ACE components, respectively. As shown in Fig. 5(a), we visualize the distribution of output discrepancies between multi-modal and text-only inputs on the full MM-Eureka dataset, revealing a significant presence of samples where multi-modal reasoning is not necessary to solve the task. Experimental results depicted in Fig. 5(b) suggest that optimal performance is achieved with λc = 0.1 and λa = 0.5. Performance degradation is observed when both parameters fall below these optimal thresholds, with values lower than 0.1 for λc and 0.5 for λa leading to significant deterioration. This decline is attributed to the inherently uneven distribution of the output-level discrepancy and attention confidence, which are concentrated at the extremes.

Qualitative analysis. We provide a qualitative analysis by visualizing cognitive samples and comparing the reasoning processes of our RAP with the latest LIMR method [9] as follows: 1) Visualization of cognitive samples. As shown in Fig. 6(a), we present a typical example of the cognitive samples selected by our RAP method, where the model effectively integrates both geometric information and logical reasoning to infer the area. This case suggests that the cognitive samples selected by our RAP method exhibit two important characteristics: (a) the necessary of multi-modal information, as evidenced by the significant discrepancy between reasoning outcomes using

Analysis on the choice of attention layers. To clarify our rationale for using last-layer attention, we provide additional experiments exploring attention maps from other layers. As illustrated in Section III-B and Fig. 2(b), attention-biased distributions typically manifest as excessively persistent high at-

Question: <image> Find the area of the figure. A. 54 B. 72 C. 108 D. 36

Question: As shown in the figure, O is the circumscribed circle of the quadrilateral ABCD, if ∠O = 110.0, then the degree of ∠C is ？

Significant output-level discrepancy

Reliable process-level reasoning

###### Groundtruth: 125°

Text-only: <think> .... If the side length of the larger

square is 6 and the side length of the smaller square is 3, ..., and the area of the smaller square is 3^2 = 9. The total area = 36 + 9 = 45</think><answer> D </answer>

LIMR: ... The relationship between the central angle and the angle subtended by the same arc at the circumference is given by: ∠C = 1/2 ∠O. Therefore, we can calculate ∠C = 55°.

D

[Figure 9]

- 6

C

Multimodal: <think> The area of the given figure can be found by breaking it up into two shapes: a rectangle and a triangle. ... , thus the area of rectangle is: 6 * 6 =

[Figure 10]

RAP (Ours): ... The inscribed angle ∠A is half of the central angle ∠O. Therefore, ∠A = 1/2 * ∠O = 55°. 3. Use the

O

6

property of a cyclic quadrilateral: In a cyclic quadrilateral, the sum of the opposite angles is 180°. Therefore, ∠A+∠C=180° 4. Calculate ∠C: Since ∠A=55°, we have ∠C=180°-55°=125°

36. ... . The area of triangle is:1/2 * 6 * 6 = 18. Total area = Area of rectangle + Area of triangle = 36 + 18 = 54.</think>\n<answer> A </answer>

A B

6

Self-attention Cognitive sample matrix

(a) An illustrative example of cognitive drivers in training samples

(b) A comparison between our RAP and the latest method LIMR [9]

- Fig. 6. Illustration of (a) characteristics of cognitive samples selected by our RAP method and (b) Comparison of the reasoning processes between our RAP method and the state-of-the-art LIMR [9].

(a) Impact of data selection on upper bound (b) Data distribution by difficulty levels (c) Performance across difficulty levels (d) Comparison with RAP with dynamic selection

[Figure 11]

[Figure 12]

+2.5% accuracy

-60% training steps

Hard Easy

Limited upper bound

Update 5.1k to 6.6k cognitive drivers for training

- Fig. 7. (a) Trade-off analysis between efficiency and performance and effect of data selection on upper bound. (b-c) Impact of varying the proportion of samples with different difficulty levels on reasoning. (c) Comparison with our RAP method and RAP augmented with dynamic selection.

multi-modal and text-only inputs; and (b) the avoidance of overemphasis on irrelevant or meaningless tokens, ensuring that the model focuses on the most informative features for accurate reasoning. 2) Comparison case analysis. As shown

- in Fig. 6(b), we compare the reasoning process of our RAP method with that of the state-of-the-art LIMR approach [9]. For example, LIMR fails to integrate cross-modal information, leading to incorrect computations of both the central angle and the inscribed angle. In contrast, the model trained using samples selected by RAP correctly applies geometric principles and multi-modal integration to arrive at the correct solution. These comparisons underscore the advantage of training with cognitive samples derived through our RAP method, which enables the model to effectively leverage multi-modal information, resulting in more reliable reasoning outcomes.

C. Key Insights and Discussion Effect of RAP on reasoning upper bound. As illustrated

- in Fig. 7(a), our RAP method converges faster than the baseline, achieving optimal performance within 40 training steps compared to 100 for the baseline. These results demonstrate RAP’s efficiency in enhancing multi-modal reasoning while reducing training overhead and data redundancy. However, further analysis reveals differences between the full RAP model and its variant using only ACE and CDE, particularly during later training stages. This occurs because ACE and CDE inevitably retain easy examples, while discarding challenging yet informative samples. For example, multi-modal examples predicted correctly in only one out of five attempts but consistently mispredicted under text-only conditions may be mistakenly eliminated when the discrepancy threshold λc

exceeds 0.2. Such filtering reduces dataset complexity, limiting the model’s overall reasoning upper bound. To address this, we propose the Difficulty-aware Replacement Module (DRM) to explicitly replace easy samples with these informative yet challenging examples, thus elevating reasoning performance in later training stages.

Why does less data outperform the full dataset? The above empirical analysis shows that models trained on carefully curated data can notably outperform those trained on the entire dataset. To further elucidate the underlying mechanisms driving this “truth in the few” phenomenon, we examine sample difficulty distributions quantified by group-wise response accuracy in the GRPO generation paradigm in Fig. 7(b). The results reveal that cognitive samples contain significantly fewer easy samples compared to the full dataset. Therefore, under conventional large-scale training, which usually restricts training to 1 or 2 epochs, the models lack sufficient repeated exposure to challenging samples, thus limiting the improvements brought from these valuable instances.

Moreover, to further validate our hypothesis, we conduct a comparison in Fig. 7(c), where models are trained on equivalent numbers of easy, hard, and cognitive samples. The results reveal that the abundance of easy examples contributes little to advancing the model’s reasoning capability and predominantly introduces redundant information. While challenging samples yield better performance than easy samples, they still fall short compared to the cognitive samples selected through our RAP method. We argue that the superiority of cognitive samples arises not merely from sample difficulty but from their ability to activate the model’s multi-modal reasoning capacity. In multi-modal contexts, the training samples must facilitate the

TABLE VI PERFORMANCE COMPARISON ON TWO TEXT-ONLY BENCHMARKS MATH500 [54] AND AMC23 USING QWEN2.5-7B [1] AND LLAMA3.1-8B [55], WITH GRPO AS THE RL ALGORITHM. RANDOM AND FULL REFER TO TRAINING WITH RANDOMLY SAMPLED DATA AND THE COMPLETE DATASET, RESPECTIVELY.

|Method|MATH500 AMC23<br><br>|
|---|---|
|Qwen2.5-7b-Random Qwen2.5-7b-Full Qwen2.5-7b-RAP (Ours)<br><br>|72.40 50.25 74.40 51.25 74.60 51.75|
|LLaMA3.1-8b-Random LLaMA3.1-8b-Full LLaMA3.1-8b-RAP (Ours)<br><br>|46.60 20.50<br><br>48.60 22.75<br>49.90 23.25<br>|

activation of the model’s multi-modal reasoning mechanisms. Generalization on text-only scenarios. Our RAP method is primarily designed to enhance multi-modal reasoning during RL post-training, and thus its core CDE relies on cross-modal signals, which is not directly transferable to unimodal settings. To address potential concerns regarding applicability to LLMs, we adapt the ACE and DRM modules to text-only scenarios and evaluate them on two text-only benchmarks, MATH500 [54] and AMC23, using the GRPO [3] as the RL algorithm. We conduct experiments on two representative LLMs, Qwen2.5-

- 7B [1] and LLaMA3.1-8B [55]. In this adaptation, due to the absence of CDE, we implement DRM by directly removing

easy samples defined by Diff = 0, i.e., samples consistently answered correctly across all trials, which typically provide limited optimization signal. As shown in Table VI, ACE and DRM still yield consistent gains over Random and Full training across both models, indicating that these components capture a generally beneficial data selection principle beyond multi-modal settings. However, the improvements are smaller than those in the multi-modal scenario. We attribute this mainly to two reasons: (1) RAP is originally motivated and optimized for multi-modal contexts where cross-modal discrepancies provide stronger supervision; (2) without CDE, DRM reduces to a difficulty-based filtering strategy, which is less targeted than the original design, thereby limiting the achievable gains.

Discussion on potential optimization during RL training. The static nature of our current RAP method motivates us to explore potential optimizations through the dynamic adjustment of the training dataset during RL training. Specifically, we examine the evolving distribution of cognitive samples identified by RAP criteria across successive epochs. Initially, as depicted in Fig. 7(d), approximately 5,000 samples meet the cognitive sample criteria. However, the number of samples that continues to satisfy this criterion progressively declines as training advances. This observed trend suggests the promise of adaptive data sampling, wherein cognitive samples are resampled dynamically from the entire dataset after the first and third training epochs. Preliminary experiments indicate that such adaptive strategies yield tangible improvements in model performance. However, despite these advantages, the rescreening of the entire dataset after each epoch incurs a significant computational overhead. As a result, our final approach strategically refrains from adopting this adaptive data selection

in order to maintain computational efficiency. This highlights the inherent trade-off between performance enhancements and computational cost when incorporating dynamic data selection strategies into the RL training process.

V. CONCLUSION

In this work, we introduce a Reasoning Activation Potential (RAP) data selection paradigm, which reduces training costs and improves multi-modal reasoning in MLLMs. RAP utilizes the Causal Discrepancy Estimator (CDE) and Attention Confidence Estimator (ACE) to effectively eliminate attentionbiased samples and language-prior biased samples, leading to more accurate and efficient reasoning. For future work, we plan to investigate the efficacy of RAP in SFT training and introduce dynamic mechanisms to further improve the efficiency of multi-modal reasoning.

REFERENCES

- [1] Qwen, :, A. Yang, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Li, D. Liu, F. Huang, H. Wei, H. Lin, J. Yang, J. Tu, J. Zhang, J. Yang, J. Yang, J. Zhou, J. Lin, K. Dang, K. Lu, K. Bao, K. Yang, L. Yu, M. Li, M. Xue, P. Zhang, Q. Zhu, R. Men, R. Lin, T. Li, T. Tang, T. Xia, X. Ren, X. Ren, Y. Fan, Y. Su, Y. Zhang, Y. Wan, Y. Liu, Z. Cui, Z. Zhang, and Z. Qiu, “Qwen2.5 technical report,”

2025. [Online]. Available: https://arxiv.org/abs/2412.15115

- [2] OpenAI, “Gpt-4 technical report,” 2023.
- [3] D. Guo, D. Yang, H. Zhang, J. Song, R. Zhang, R. Xu, Q. Zhu, S. Ma, P. Wang, X. Bi et al., “Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning,” arXiv preprint arXiv:2501.12948, 2025.
- [4] J. Hu, X. Wu, Z. Zhu, Xianyu, W. Wang, D. Zhang, and Y. Cao, “Openrlhf: An easy-to-use, scalable and high-performance rlhf framework,” arXiv preprint arXiv:2405.11143, 2024.
- [5] Y. Yang, X. He, H. Pan, X. Jiang, Y. Deng, X. Yang, H. Lu, D. Yin, F. Rao, M. Zhu, B. Zhang, and W. Chen, “R1-onevision: Advancing generalized multimodal reasoning through cross-modal formalization,” CoRR, vol. abs/2503.10615, 2025.
- [6] OpenAI, “Learning to reason with llms,” 2024. [Online]. Available: https://openai.com/index/learning-to-reason-with-llms/
- [7] K. Team, A. Du, B. Gao, B. Xing, C. Jiang, C. Chen, C. Li, C. Xiao, C. Du, C. Liao, C. Tang, C. Wang, D. Zhang, E. Yuan, E. Lu, F. Tang, F. Sung, G. Wei, G. Lai, H. Guo, H. Zhu, H. Ding, H. Hu, H. Yang, H. Zhang, H. Yao, H. Zhao, H. Lu, H. Li, H. Yu, H. Gao, H. Zheng, H. Yuan, J. Chen, J. Guo, J. Su, J. Wang, J. Zhao, J. Zhang, J. Liu, J. Yan, J. Wu, L. Shi, L. Ye, L. Yu, M. Dong, N. Zhang, N. Ma, Q. Pan, Q. Gong, S. Liu, S. Ma, S. Wei, S. Cao, S. Huang, T. Jiang,

- W. Gao, W. Xiong, W. He, W. Huang, W. Wu, W. He, X. Wei,
- X. Jia, X. Wu, X. Xu, X. Zu, X. Zhou, X. Pan, Y. Charles, Y. Li,
- Y. Hu, Y. Liu, Y. Chen, Y. Wang, Y. Liu, Y. Qin, Y. Liu, Y. Yang,

- Y. Bao, Y. Du, Y. Wu, Y. Wang, Z. Zhou, Z. Wang, Z. Li, Z. Zhu,
- Z. Zhang, Z. Wang, Z. Yang, Z. Huang, Z. Huang, Z. Xu, and Z. Yang, “Kimi k1.5: Scaling reinforcement learning with llms,” 2025. [Online]. Available: https://arxiv.org/abs/2501.12599

- [8] N. Muennighoff, Z. Yang, W. Shi, X. L. Li, L. Fei-Fei, H. Hajishirzi, L. Zettlemoyer, P. Liang, E. J. Cand`es, and T. Hashimoto, “s1: Simple test-time scaling,” CoRR, vol. abs/2501.19393, 2025.
- [9] X. Li, H. Zou, and P. Liu, “LIMR: less is more for RL scaling,” CoRR, vol. abs/2502.11886, 2025.
- [10] Y. Han, L. Nie, J. Yin, J. Wu, and Y. Yan, “Visual perturbation-aware collaborative learning for overcoming the language prior problem,” CoRR, vol. abs/2207.11850, 2022.
- [11] S. Leng, H. Zhang, G. Chen, X. Li, S. Lu, C. Miao, and L. Bing, “Mitigating object hallucinations in large vision-language models through visual contrastive decoding,” in CVPR, 2024, pp. 13872–13882.
- [12] Y. Ye, Z. Huang, Y. Xiao, E. Chern, S. Xia, and P. Liu, “LIMO: less is more for reasoning,” CoRR, vol. abs/2502.03387, 2025.
- [13] M. G. Azar, Z. D. Guo, B. Piot, R. Munos, M. Rowland, M. Valko, and D. Calandriello, “A general theoretical paradigm to understand learning from human preferences,” in International Conference on Artificial Intelligence and Statistics, vol. 238. PMLR, 2024, pp. 4447–4455.

- [14] B. Lin, Z. Tang, Y. Ye, J. Huang, J. Zhang, Y. Pang, P. Jin, M. Ning, J. Luo, and L. Yuan, “Moe-llava: Mixture of experts for large visionlanguage models,” IEEE Transactions on Multimedia, pp. 1–14, 2026.
- [15] H. Xiong, Y. Zhuge, J. Zhu, L. Zhang, and H. Lu, “3ur-llm: An endto-end multimodal large language model for 3d scene understanding,” IEEE Transactions on Multimedia, vol. 27, pp. 2899–2911, 2025.
- [16] W. Tang, Y. Sun, Q. Gu, and Z. Li, “Visual position prompt for mllm based visual grounding,” IEEE Transactions on Multimedia, pp. 1–16, 2026.
- [17] X. Lai, Z. Tian, Y. Chen, S. Yang, X. Peng, and J. Jia, “Step-dpo: Stepwise preference optimization for long-chain reasoning of llms,” CoRR, vol. abs/2406.18629, 2024.
- [18] S. R. Chowdhury, A. Kini, and N. Natarajan, “Provably robust DPO: aligning language models with noisy feedback,” in Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024. OpenReview.net, 2024. [Online]. Available: https://openreview.net/forum?id=yhpDKSw7yA
- [19] R. Rafailov, A. Sharma, E. Mitchell, C. D. Manning, S. Ermon, and C. Finn, “Direct preference optimization: Your language model is secretly a reward model,” in NeurIPS, 2023.
- [20] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, “Proximal policy optimization algorithms,” vol. abs/1707.06347, 2017.
- [21] A. Ahmadian, C. Cremer, M. Gall´e, M. Fadaee, J. Kreutzer, O. Pietquin, A. Ust¨¨ un, and S. Hooker, “Back to basics: Revisiting reinforce-style optimization for learning from human feedback in llms,” in ACL, 2024, pp. 12248–12267.
- [22] Y. Peng, G. Zhang, M. Zhang, Z. You, J. Liu, Q. Zhu, K. Yang, X. Xu, X. Geng, and X. Yang, “LMM-R1: empowering 3b lmms with strong reasoning abilities through two-stage rule-based RL,” CoRR, vol. abs/2503.07536, 2025.
- [23] Z. Liu, Z. Sun, Y. Zang, X. Dong, Y. Cao, H. Duan, D. Lin, and J. Wang, “Visual-rft: Visual reinforcement fine-tuning,” arXiv preprint arXiv:2503.01785, 2025.
- [24] W. Wang, Z. Chen, W. Wang, Y. Cao, Y. Liu, Z. Gao, J. Zhu, X. Zhu, L. Lu, Y. Qiao et al., “Enhancing the reasoning ability of multimodal large language models via mixed preference optimization,” arXiv preprint arXiv:2411.10442, 2024.
- [25] H. Yao, J. Huang, W. Wu, J. Zhang, Y. Wang, S. Liu, Y. Wang, Y. Song, H. Feng, L. Shen, and D. Tao, “Mulberry: Empowering MLLM with o1-like reasoning and reflection via collective monte carlo tree search,” CoRR, vol. abs/2412.18319, 2024.
- [26] F. Meng, L. Du, Z. Liu, Z. Zhou, Q. Lu, D. Fu, B. Shi, W. Wang, J. He, K. Zhang, P. Luo, Y. Qiao, Q. Zhang, and W. Shao, “Mm-eureka: Exploring visual aha moment with rule-based large-scale reinforcement learning,” CoRR, vol. abs/2503.07365, 2025.
- [27] S. Li, X. Xu, W. Meng, J. Song, C. Peng, and H. T. Shen, “Mitigating hallucinations in large vision-language models via reasoning uncertaintyguided refinement,” IEEE Trans. Multim., vol. 27, pp. 7380–7391, 2025.
- [28] R. Y. Rohekar, Y. Gurwicz, and S. Nisimov, “Causal interpretation of self-attention in pre-trained transformers,” in NeurIPS, 2023.
- [29] C. Zhang, L. Zhang, and D. Zhou, “Causal walk: Debiasing multi-hop fact verification with front-door adjustment,” in AAAI, 2024, pp. 19533– 19541.
- [30] S. Zhou, J. Xiao, X. Yang, P. Song, D. Guo, A. Yao, M. Wang, and T. Chua, “Scene-text grounding for text-based video question answering,” TMM, 2025.
- [31] L. Z. Y. B. G. W. Y. Y. H. T. S. Zheng Wang, Xing Xu, “Evidence-based multi-feature fusion for adversarial robustness,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 47, no. 10, pp. 8923–8937, 2025.
- [32] B. Sch¨olkopf, D. Janzing, J. Peters, E. Sgouritsa, K. Zhang, and J. Mooij, “On causal and anticausal learning,” arXiv preprint arXiv:1206.6471, 2012.
- [33] D. B. Rubin, “Causal inference using potential outcomes: Design, modeling, decisions,” Journal of the American statistical Association, vol. 100, no. 469, pp. 322–331, 2005.
- [34] T. Du, A. Kanodia, H. Brunborg, K. Vafa, and S. Athey, “LABORLLM: language-based occupational representations with large language models,” CoRR, vol. abs/2406.17972, 2024.
- [35] Q. Huang, L. Chan, J. Liu, W. He, H. Jiang, M. Song, J. Chen, C. Yao, and J. Song, “Boosting MLLM reasoning with text-debiased hint-grpo,” CoRR, vol. abs/2503.23905, 2025.
- [36] D. B. Rubin, “Estimating causal effects of treatments in randomized and nonrandomized studies,” Journal of educational Psychology, vol. 66, no. 5, p. 688, 1974.
- [37] F. D. Johansson, U. Shalit, and D. A. Sontag, “Learning representations for counterfactual inference,” in ICML, vol. 48, 2016, pp. 3020–3029.

- [38] U. Shalit, F. D. Johansson, and D. A. Sontag, “Estimating individual treatment effect: generalization bounds and algorithms,” in ICML, vol. 70, 2017, pp. 3076–3085.
- [39] J. Pearl, “Causal inference in statistics: An overview,” 2009.
- [40] G. Hammerton and M. R. Munaf`o, “Causal inference with observational data: the need for triangulation of evidence,” Psychological medicine, vol. 51, no. 4, pp. 563–578, 2021.
- [41] J. Abrevaya, Y.-C. Hsu, and R. P. Lieli, “Estimating conditional average treatment effects,” Journal of Business & Economic Statistics, vol. 33, no. 4, pp. 485–505, 2015.
- [42] Q. Huang, X. Dong, P. Zhang, B. Wang, C. He, J. Wang, D. Lin, W. Zhang, and N. Yu, “OPERA: alleviating hallucination in multimodal large language models via over-trust penalty and retrospectionallocation,” in CVPR, 2024, pp. 13418–13427.
- [43] L. Wang, L. Li, D. Dai, D. Chen, H. Zhou, F. Meng, J. Zhou, and X. Sun, “Label words are anchors: An information flow perspective for understanding in-context learning,” in EMNLP, 2023, pp. 9840–9855.
- [44] P. Lu, H. Bansal, T. Xia, J. Liu, C. Li, H. Hajishirzi, H. Cheng, K. Chang, M. Galley, and J. Gao, “Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts,” in ICLR, 2024.
- [45] L. Chen, J. Li, X. Dong, P. Zhang, Y. Zang, Z. Chen, H. Duan, J. Wang, Y. Qiao, D. Lin, and F. Zhao, “Are we on the right way for evaluating large vision-language models?” in NeurIPS, 2024.
- [46] R. Zhang, D. Jiang, Y. Zhang, H. Lin, Z. Guo, P. Qiu, A. Zhou, P. Lu, K. Chang, Y. Qiao, P. Gao, and H. Li, “MATHVERSE: does your multimodal LLM truly see the diagrams in visual math problems?” in ECCV, 2024, pp. 169–186.
- [47] R. Qiao, Q. Tan, G. Dong, M. Wu, C. Sun, X. Song, Z. Gongque, S. Lei, Z. Wei, M. Zhang, R. Qiao, Y. Zhang, X. Zong, Y. Xu, M. Diao, Z. Bao, C. Li, and H. Zhang, “We-math: Does your large multimodal model achieve human-like mathematical reasoning?” CoRR, vol. abs/2407.01284, 2024.
- [48] W. Yu, Z. Yang, L. Li, J. Wang, K. Lin, Z. Liu, X. Wang, and L. Wang, “Mm-vet: Evaluating large multimodal models for integrated capabilities,” in ICML, 2024.
- [49] Y. Xiao, E. Sun, T. Liu, and W. Wang, “Logicvista: Multimodal LLM logical reasoning benchmark in visual contexts,” CoRR, vol. abs/2407.04973, 2024.
- [50] G. Sheng, C. Zhang, Z. Ye, X. Wu, W. Zhang, R. Zhang, Y. Peng, H. Lin, and C. Wu, “Hybridflow: A flexible and efficient rlhf framework,” arXiv preprint arXiv: 2409.19256, 2024.
- [51] W. Kwon, Z. Li, S. Zhuang, Y. Sheng, L. Zheng, C. H. Yu, J. E. Gonzalez, H. Zhang, and I. Stoica, “Efficient memory management for large language model serving with pagedattention,” in Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles, 2023.
- [52] Z. Chen, J. Wu, W. Wang, W. Su, G. Chen, S. Xing, M. Zhong, Q. Zhang, X. Zhu, L. Lu, B. Li, P. Luo, T. Lu, Y. Qiao, and J. Dai, “Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks,” CVPR, 2024.
- [53] H. Yin, G. Si, and Z. Wang, “Clearsight: Visual signal enhancement for object hallucination mitigation in multimodal large language models,” in CVPR, 2025, pp. 14625–14634.
- [54] D. Hendrycks, C. Burns, S. Kadavath, A. Arora, S. Basart, E. Tang, D. Song, and J. Steinhardt, “Measuring mathematical problem solving with the MATH dataset,” in NeurIPS Datasets and Benchmarks, 2021.
- [55] A. Dubey, A. Jauhri, A. Pandey, A. Kadian, A. Al-Dahle, A. Letman, A. Mathur, A. Schelten, A. Yang, A. Fan et al., “The llama 3 herd of models,” arXiv e-prints, pp. arXiv–2407, 2024.

