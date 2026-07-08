# arXiv:2603.12529v2[cs.LG]14May2026

## TERMINATOR: Learning Optimal Exit Points for Early Stopping in Chain-of-Thought Reasoning

##### Alliot Nagle˚

UT Austin

Jakhongir Saydaliev EPFL

Ashok Vardhan Makkuva† Télécom Paris (IP Paris)

Dhia Garbaya ENS Paris-Saclay

Michael Gastpar EPFL

Hyeji Kim† UT Austin

### Abstract

Large Reasoning Models (LRMs) achieve impressive performance on complex reasoning tasks via Chain-of-Thought (CoT) reasoning, which enables them to generate intermediate thinking tokens before arriving at the final answer. However, LRMs often suffer from significant overthinking, spending excessive compute time even after the answer is generated early on. Prior work has identified the existence of an optimal reasoning length such that truncating reasoning at this point significantly shortens CoT outputs with virtually no change in performance. However, determining optimal CoT lengths for practical datasets is highly non-trivial as they are fully task and model-dependent. In this paper, we precisely address this and design TERMINATOR, an early-exit strategy for LRMs at inference to mitigate overthinking. The central idea underpinning TERMINATOR is that the first arrival of an LRM’s final answer is often predictable, and we leverage these first answer positions to create a novel dataset of optimal reasoning lengths to train TERMINATOR. Powered by this approach, TERMINATOR achieves significant reductions in CoT lengths of 14%–55% on average across four challenging practical datasets: MATH-500, AIME 2025, HumanEval, and GPQA, while outperforming current state-of-the-art methods and reducing inference latency by more than 2ˆ compared to the original LRM.

### 1 Introduction

The advent of Large Reasoning Models (LRMs) has proven itself to be a critical next step for Large Language Models (LLMs) to surpass human-level performance. LRMs use test-time compute to “think” through a problem before answering, an approach that has led to significant performance gains across many challenging tasks [34]. However, this improvement does not come for free, as an LRM will generate thousands of additional thinking tokens to solve a single problem, compared to its non-reasoning counterparts [11]. Worse yet, LRMs spend a significant amount of their reasoning tokens double-checking their work and exploring different solutions when they have already generated the final answer, that they will eventually settle on, much earlier in the CoT, a phenomenon known as overthinking [27, 3]. Prior work has shown that the length of a CoT can be reduced by 50% or more on average with little drop in accuracy [18, 50, 46], demonstrating the extent to which compute is wasted during LRM inference.

Given that reasoning can be wasteful, a natural question to ask is, for any given accuracy, does there exist an optimal reasoning length? Previous works have shown that LRM performance, as a function of reasoning length, gradually increases, peaks, and then decreases, suggesting the existence

˚Corresponding Author: acnagle@utexas.edu

Preprint.

###### Pareto Frontiers: Accuracy vs. Compression Rate

###### MATH-500

###### AIME25

###### HumanEval

###### GPQA

###### Overall

60

80

Accuracy()

75

95

90

Qwen3-8B

70

50

50

90

80

60

25

85

40

0 25 50 75 100

0 25 50 75 100

0 25 50 75 100

0 25 50 75 100

0 25 50 75 100

Accuracy()

60

Qwen3-14B

75

80

95

90

90

50

70

50

85

80

25

60

0 25 50 75 100

0 25 50 75 100

0 25 50 75 100

0 25 50 75 100

0 25 50 75 100

100

90

100

Accuracy()

Ministral-8B

90

60

80

75

80

80

50

70

50

60

70

40

0 25 50 75 100

0 25 50 75 100

0 25 50 75 100

0 25 50 75 100

0 25 50 75 100

100

Ministral-14B

100

Accuracy()

90

60

80

80

50

50

50

60

70

40

0 25 50 75 100

0 25 50 75 100

0 25 50 75 100

0 25 50 75 100

0 25 50 75 100

Compression Rate ( )

Compression Rate ( )

Compression Rate ( )

Compression Rate ( )

Compression Rate ( )

Vanilla NoThinking DEER Thought-Calilbration

Dynasor Terminator (Ours)

| | |
|---|---|
| | |

- Figure 1: Pareto Frontier. TERMINATOR defines the Pareto frontier on 14 of the 16 (LRM, benchmark) pairings, outperforming prior work. Each point represents a method’s accuracy and compression rate, with lower compression rates indicating greater token savings and hence compute. The dashed line traces the Pareto frontier connecting non-dominated solutions. We refer to Sec. 5.2 and Table 7 in App. C for more details.

of an optimal reasoning length [44, 21]. Additionally, some recent works propose novel RL-training algorithms to fine-tune LRMs to produce shorter CoTs [27, 25, 9, 48, 39] and establish the Pareto frontier for those methods, showing that gaps still exist between them [9]. While these works focus on retraining an LRM, inference-time methods such as DEER [46] enable early termination of reasoning without retraining it. However, for practical tasks none of these methods either determines or utilizes the optimal-length reasoning, which in fact provides the best possible reduction in CoT length.

In this paper, we precisely address this by introducing the novel notion of hindsight-optimal reasoning length (Sec. 2.2): given a reasoning task, in hindsight, what is the fewest number of tokens that an LRM needs to generate before providing the same answer it would have provided without shortened reasoning? Namely, we mark the first logical arrival, as opposed to any other occurrence, of the LRM’s final answer as the hindsight-optimal exiting position. Leveraging this notion, we design a novel inference-time early-exit algorithm TERMINATOR that significantly outperforms current state-of-the-art methods in reductions to CoT lengths on challenging practical datasets (Fig. 4). In particular, TERMINATOR capitalizes on the fact that the first arrival of the final answer is (1) marked by a distinctive shift in the LRM’s token-level confidence and token usage distribution, and (2) can be used as a signal to train a binary probe classifier for effective early-exiting during reasoning.

Main Contributions. In summary, we make the following contributions:

- • We introduce the novel notion of hindsight-optimal reasoning, using which we show that the first arrival of an LRM’s final answer is marked by observable and meaningful signals (Fig. 2). To the best of our knowledge, this is the first such analysis of its kind.
- • We design TERMINATOR, a novel, lightweight, early-exit algorithm for LRMs that is trained on optimal-length CoTs (Sec. 4.2), resulting in more than a 2ˆ reduction in inference latency compared to the original LRM (Sec. 5.2).
- • We introduce a robust pipeline for identifying the first arrival of the final answer in CoTs, using which we construct a novel optimal-length CoT training dataset (Sec. 4.1).

### 2 Preliminaries

##### 2.1 Notation

A Large Reasoning Model LRM takes as input the prompt sequence x “ px1,x2,x3,...,xLq and produces two outputs r and s auto-regressively (Fig. 4). Here r “ pr1,r2,r3,...,rMq is the CoT

Single Sample and Averaged Time Series for Qwen3-8B - All Data Sources (n=3200)

###### Single Sample Token Confidence

###### Event-Locked Average Token Confidence

Final Answer Position

Final Answer Position

30

40

Mean

TokenConfidence

±1 SE

30

25

20

20

10

15

0

0 1000 2000 3000 4000

1000 750 500 250 0 250 500 750 1000

###### Single Sample Log Probabilities

###### Event-Locked Average Log Probabilities

- 0

- 1

- 2

- 3

- 4

Final Answer Position

Final Answer Position

0.3

Mean

LogProbability

±1 SE

0.2

0.1

0.0

0 1000 2000 3000 4000

1000 750 500 250 0 250 500 750 1000

Token Index

Relative Position from Answer

- Figure 2: Event-Locked Averaging of Token-Confidence. Event-locked averaging shows a consistent agreement on spiking behavior at the answer position in each CoT, but disagrees elsewhere. On the other hand, this phenomenon is not readily observable in the single-sample case. Figures on the left show the Token-Confidence [8] and log-probability trajectories throughout reasoning for a single, randomly selected sample; figures on the right show the effect of event-locked averaging on the position of the first arrival of the final answer across all CoTs. The 3200 CoTs used are a random subset of our training set, which combines AIME (1983–2024), MATH, OpenCoder-SFT, and OpenScience. Figs. 15 to 18 in App. C show similar trends for each dataset separately. The standard error (SE) is shown as a shaded region and becomes clearer when zoomed in.

sequence generated during the thinking stage, i.e. ri “ LRMpx,răiq for i P rMs fi t1,...,Mu, and s “ ps1,s2,s3,...,sNq is the solution that summarizes this CoT and contains a final answer aˆ, which could be a single numerical answer, a math expression, code, a multiple-choice option, etc. Here sj “ LRMpx,r,săjq for j P rNs. Note that the final answer aˆ is separate from the ground-truth answer a; they may or may not be in agreement with each other. Throughout the paper, aˆ always refers to the final answer of the full CoT, not the final answer generated after exiting a CoT early. Furthermore, when referring to aˆ with respect to its position in a CoT, we always mean the earliest logical arrival of aˆ unless stated otherwise explicitly. By the earliest logical arrival of aˆ, we are referring to the sequence of logical steps in the CoT that yields the final answer aˆ for the first time. For any early-exit strategy, a key metric to gauge its performance is the per-sample compression rate

(CR): MMearly, where Mearly P rMs is the token index of early exit in r. Accuracy (Acc) measures the proportion of problems where the correct answer is produced.

##### 2.2 Hindsight-optimality

We now formally define our novel notion of hindsight-optimality. Given an input prompt x P XL of length L over a vocabulary X, an LRM LRM generates a corresponding CoT r P XM and solution s P XN, where the solution contains a final answer aˆ P X. The hindsight-optimal reasoning length (HORL) is defined as the earliest position in the completed CoT at which the final answer aˆ has been logically reached. Mathematically,

HORLpx,r,s,aˆq fi minti P rMs : rďi contains the earliest logical arrival of aˆu. (1)

Here, rďi is said to contain the earliest logical arrival of aˆ if, by position i, the sequence of reasoning steps in r has produced the first derivation of the final answer aˆ. Thus, HORL is a retrospective property of the realized CoT r and final answer aˆ.

##### 2.3 Token-Confidence

Our analytical experiments require a measure of LRM’s confidence during the generation of a CoT. To this end, we use the Token-Confidence metric, which gauges the uncertainty of a chosen token. Mathematically, for every i P rMs, the corresponding Token-Confidence Ci is defined as

1 K ÿ

log PLRM pri “ k | x,răiq, (2)

Ci fi ´

kPTKpiq

Token Occurrence Ratios for Qwen3-8B - All Data Sources (n=3200)

###### Token: "hmm"

###### Token: "okay"

Token: "another"

Above Diagonal: 68.1% Below Diagonal: 19.8%

Above Diagonal: 5.2% Below Diagonal: 63.9%

Above Diagonal: 8.5% Below Diagonal: 91.5%

10 2

10 2

10 2

OccurrenceRateAfterAnswer

10 3

10 3

10 3

10 4

10 4

10 4

10 5

10 5

10 5

0

0

0

0 10 5 10 4 10 3 10 2

0 10 5 10 4 10 3 10 2

0 10 5 10 4 10 3 10 2

Occurrence Rate Before Answer

Occurrence Rate Before Answer

Occurrence Rate Before Answer

- Figure 3: Token Usage Frequency Shift. “Thinking token” usage changes depending on whether the final answer has been generated in the CoT. Rates are computed by counting the raw number of occurrences of the token before and after the answer, and then normalizing each count by the respective number of tokens in the before and after bins. The arrival of the final answer is hinted at by changes in the rates for these tokens. The relative length of a CoT is captured by its dot size, where a longer CoT has a larger dot. App. C demonstrates similar results for other “thinking tokens” in Fig. 19 and for each data source in Figs. 20 to 23.

where PLRM pri “ ¨ | x,răiq is the LRM prediction probability at position i and TKpiq fi Top-K rPLRM pri “ ¨ | x,răiqs is the set of vocabulary tokens corresponding to the Top-K probabilities. In other words, Token-Confidence is the average (negative) log-probability across the Top-K probabilities (we set K “ 20 in our experiments). The higher it is, the more confident the model is in its predictions.

This measure is based on the Self-Certainty metric [19], computed as the KL-divergence between the uniform distribution and the token distribution, and is based on the following idea: the higher the confidence of the model, the further its predictions should be from the uniform distribution. We also note that, while Token-level log-probabilities are commonly used as a proxy for confidence, we prefer the Token-Confidence measure [8] here, as it is principled and produces less noise. Both are used, and the same conclusions can be drawn using either.

### 3 Motivation

Shortly after the breakthrough of LRMs, it was observed that they exhibit an overthinking phenomenon where, despite arriving at the correct answer, they continue to consider alternative solution paths, possibly leading to other incorrect answers [3, 27]. While LRMs achieve greatly improved performance over their non-reasoning counterparts, they do so at a much higher inference-time cost: up to thousands of additional tokens are generated to form the CoT before arriving at the final solution [11]. Many follow-up works have observed the same overthinking phenomenon and developed methods to mitigate wasteful token expenditure [50, 24, 43, 49].

Towards designing an optimal early-exit strategy to stymie overthinking, we build upon the following key observation: once an LRM generates a CoT r and a final solution s, we can observe the final answer aˆ. Then, in hindsight, we can determine precisely where the LRM should have exited the CoT to avoid wasting tokens, i.e. HORL, and instead generate the final solution. To this end, we first only need to check for aˆ, not a, in r, since the LRM may never even have generated the correct (ground-truth) answer and thus may not exist in r. Second, by choosing to terminate reasoning after the arrival of aˆ in r, all steps that are useful to arriving at aˆ are kept, and anything after is skipped as it is usually redundant.

While the above procedure requires the explicit knowledge of aˆ to check for its arrival, are there meaningful markers to implicitly detect its arrival?

Detecting the Answer Early. We analyze trends in Token-Confidence during CoT reasoning on AIME (1983–2024), MATH, OpenCoder-SFT, and OpenScience, as shown in Figs. 2 and 3. Fig. 2 reveals a sharp transition in Token-Confidence around the first occurrence of aˆ in event-locked averages, formed by aligning each CoT’s first aˆ position to 0 and averaging across samples. Because these CoTs span math, science, and coding datasets, this transition suggests a consistent cross-domain signal; per-dataset versions appear in Figs. 15 to 18.

Fig. 3 complements this by comparing token frequencies before and after the first aˆ occurrence for three “thinking tokens”: hmm, okay, and another. These tokens are associated with ongoing reasoning, e.g., wait, so, alternatively, therefore, and we hypothesize that their usage changes once aˆ has been generated [40, 36, 6]. Indeed, hmm and okay occur more often before aˆ, while another occurs more often after it. Although not every thinking token shows a clear before/after bias, these shifts indicate that token-frequency distributions can signal when aˆ has appeared; additional examples are shown in Fig. 19. In each scatter plot, the axes show before- and after-aˆ token rates, computed as raw counts divided by the corresponding number of tokens, and point diameter indicates relative CoT length.

Moving to Online Inference and Challenges. While these results strongly indicate the early arrival of aˆ, using them during online inference remains a challenge. In the case of Fig. 2, event-locked averaging requires multiple CoTs to be generated simultaneously, each with reasonable estimates of the position of the answer. Under those circumstances, the spiking behavior will emerge. But attaining a reasonable estimate of the answer position for a single CoT during inference is the original problem we are tasked with. While applying the event-locked averaging signal to online inference is limited, it does indicate that an underlying trend can be extracted.

Similarly, each dot in Fig. 3 requires full knowledge of each r and its position of aˆ so that the rates can be calculated accordingly. Again, these results show a shift in the usage frequency of certain tokens before and after the first occurrence of aˆ, but translating the signal into an online inference algorithm remains challenging.

Our Approach. Under the hood of the LRM, there are clearly meaningful signals to indicate the earliest arrival of aˆ. However, as outlined above there are unique challenges in leveraging them in a hand-guided way to design an early-exit online inference algorithm. To address this, we approach this through the lens of prediction: i.e. predicting whether an LRM’s final answer aˆ has been generated or not. To this end, the core idea behind our method is to train a probe classifier—the TERMINATOR—on the hidden states of the final layer, thereby utilizing as many of the LRM’s underlying signals as possible (Fig. 4). Prior work has examined hidden states to assess whether LRMs know when their intermediate CoT answers are correct [49], a finding primarily aimed at understanding model internals rather than designing practical early-exit methods at inference. Our work adopts a fundamentally different, deployable approach by probing for the final answer aˆ, a signal that is fully self-contained within the LRM’s reasoning process and requires no ground-truth labels at inference time, thereby enabling a principled early-exit strategy.

last 10 bits

0 0 1 ... 0 1 Sumpbi´9:iq ą 5

bi´9:i

TERMINATOR

</think>Inject

h1 h2 h3 ... hi´1 hi

Earlyexit

Let’s start with ... get 5 In summary ... 5

Large Reasoning Model LRM

What is 2 ` 3? <think> Let’s start ... you get 5 </think> In ... is 5

Input x CoT r Solution s

Figure 4: Early stopping via TERMINATOR. TERMINATOR is a binary probe classifier that predicts whether to exit or not at every CoT token. Once the majority of prediction bits within a window (10 here) are 1, </think> is injected into the LRM’s token stream to stop thinking.

Advantages of TERMINATOR over Prior Methods. To train the probe classifier, we process inputs at the token level, offering much finer-grained predictions than prior work. That is, our dataset is curated for a token classification task. To the best of our knowledge, all previous methods use much coarser granularity in their training dataset where they chunk each CoT r according to some heuristic, such

- as “thinking tokens” or paragraph delimiters such as \n\n. Then, at inference time, they exit once the predicted probability crosses a data-calibrated threshold [24, 43, 49]. In contrast, our approach offers two-fold advantages at inference-time: (1) a probe classifier trained with our dataset has the ability to exit immediately after aˆ is generated, and (2) while our approach is amenable to using a data-calibrated threshold, it is not necessary. The main drawback of data-calibrated thresholding is that it requires additional samples from the evaluation data distribution, and the resulting threshold is therefore specific to that distribution and may not transfer well to other distributions.

However, obtaining the aˆ positions to create our HORL-dataset in a scalable way is challenging and highly non-trivial, which we precisely address in the next section.

False

###### px,r,i‹q

px,r,sq

###### Answer Position Identification

###### Answer Verification

###### Token-Index Extraction

###### Answer Extraction

Input: pr,aˆq Ask LRM to find a span of text d leading to the first logical arrival of aˆ Output: d

True

Input: (d,aˆ) Ask LRM whether aˆ P d Output: u P tTrue, Falseu

Input: (d,r,aˆ) Extract the token position of aˆ P r Output: Position i‹ P rMs

Input: s (final solution) Ask LRM to extract aˆ from s Output: aˆ

Figure 5: Training-Dataset Curation Process. We use an LRM to (1) extract final answer aˆ from final solution s, (2) identify the earliest position of aˆ in the CoT r, and (3) verify that the position was correct. If it was, then we can extract the exact position of aˆ from the CoT at the final token-index extraction step; otherwise, we retry the identification step with feedback.

### 4 TERMINATOR: Methodology

Given a full CoT and the corresponding final solution from an LRM, the earliest logical arrival to the LRM’s final answer can be detected in the CoT. However, reliable detection for tens of thousands of CoTs is a unique challenge, which we address through our pipeline in Sec. 4.1. We then present our method for training TERMINATOR, which is a probe classifier, in Sec. 4.2.

##### 4.1 Early Answer Extraction, Identification, and Verification

Our early answer extraction, identification, and verification pipeline (Fig. 5) is a critical component of our data curation process; at its core is an LRM that (1) extracts the final answer aˆ from final solution s (answer extraction), (2) identifies the earliest logical arrival to aˆ in r (answer identification), (3) verifies that the extraction step was successful (answer verification). Finally, (4) we extract the exact position of aˆ from the CoT (token-index extraction).

Rationale. Extracting the position of aˆ is not trivial. Human inspection and annotation of CoTs is one route, but it is expensive and not scalable. Our early attempts at answer extraction, identification, and verification relied solely on fuzzy pattern matching, resulting in many false positives despite our best efforts to accommodate as many edge cases as possible. The primary challenge is that identifying the answer position within a CoT is a semantic search problem that cannot be reliably solved with fuzzy or regex pattern matching for numerical answers, mathematical expressions, and code. Examples where pattern matching fails for these three is given in App. B.2. Using an LRM for all three cases confirmed that the earliest answer positions can be reliably extracted.

Our Extract-Identify-Verify Pipeline. We first extract the final answer aˆ from s, where it is explicitly marked, e.g. with \boxed{}, making extraction straightforward for the LRM. Next, the LRM identifies a span d that both precedes and includes aˆ. This ensures that d is a unique substring of r, allowing us to recover the exact token position of the earliest occurrence of aˆ. The LRM then verifies that d contains aˆ. If verification fails, the LRM repeats the identification step, providing textual feedback listing all previously selected spans that did not contain aˆ, reducing the chance of selecting the same span again. If no valid span is found within the retry limit, the corresponding CoT is excluded from the training set. Otherwise, we locate d in r and retrieve the earliest answer token position i‹. Algorithm 1 in App. B.1 gives pseudocode for this procedure. These steps scale the construction of TERMINATOR, which we use to train our probe classifier.

Computational Cost. Our cost-benefit analysis in App. C.4 shows that the inference-time benefits of TERMINATOR substantially outweigh the cost of running this pipeline.

##### 4.2 TERMINATOR: Binary Probing Classifier

Our approach entails training a small classification model θ on the LRM’s final-layer hidden states hi and making a binary prediction bi at each CoT position i P rMs (Fig. 4). More specifically, our model reuses the same transformer block from the LRM and adds a prediction head. The weights of the transformer block are copied from the final block of the LRM, which we found performs slightly better than random initialization, and the prediction head is randomly initialized. During training, the task is to predict whether the first occurrence of the final answer has been generated (label 1) or not (label 0). Given the causality of the transformer block, every prediction depends on the history of the CoT up to that point, but the predictions themselves are made independently of each other. Due to the inherent class imbalance of this early-exiting prediction task, our model is trained with

class-weighted binary cross-entropy loss, which for a single sample px,r,s,i‹q is computed as:

”w1 ¨ yi ¨ log pi ` w0 ¨ p1 ´ yiq ¨ logp1 ´ piqı, (3)

ÿM

1 M

Lpθq “ ´

i“1

where yi “ pi ă i ‹q P t0,1u denotes the ground-truth label corresponding to answer arrival and pi “ Pθ pbi “ 1 | x,rďiq is the predicted probability for each i P rMs, with M being the CoT length, and w0 and w1 the class weights. These weights are automatically computed from the training dataset using inverse frequency weighting as w0 “ pn0 ` n1q{2n0 and w1 “ pn0 ` n1q{p2n1q, where n0 and n1 are the total number of 0 and 1 labels in the training dataset, respectively.

Here we note that TERMINATOR is inspired by the findings of optimal-length reasoning literature; we seek to train a model on hindsight-optimal CoTs to encourage TERMINATOR to early-exit as soon as the final answer is generated. Unlike other methods, TERMINATOR is free of data-calibrated thresholding and is trained on several data sources (math, coding, and STEM problems) simultaneously.

### 5 Experiments

##### 5.1 Implementation Details

Models. We train and evaluate our method on LRMs from two different model families: Qwen3-8B and Qwen3-14B [45], and Ministral-3-8B-Reasoning-2512 and Ministral-3-14B-Reasoning-2512 [23]. We use Qwen3-30B-A3B-Thinking-2507 for our answer extraction, identification, and verification pipeline. Our trained models consist of a single transformer layer initialized from the final layer of the LRM and a binary prediction head. We compare TERMINATOR against (1) prompt-based approaches, including Vanilla, NoThinking [28], DEER [46], and Dynasor [7], and (2) a probe-based approach, Thought Calibration [43]. Vanilla is a direct evaluation of the LRM without any intervention. NoThinking prompts the model to skip the reasoning phase and generate the final solution s directly. DEER splits the reasoning into chunks, checks the average token probability after every chunk, and exits if it exceeds a threshold. Dynasor periodically prompts the model to produce intermediate answers at fixed token intervals and triggers early exit when 8 consecutive answers are consistent. Thought Calibration trains linear probes on the hidden representations of reasoning steps to automatically decide when to stop generation. We retrain these probes for our 4 models using their Supervised method. For further details of the baselines’ implementations, we refer to App. B.6.

Datasets. We form a training data mix with AIME (1983–2024) [1], MATH [22], OpenCoder-SFT [13], and OpenScience [33]. We form our training datasets by sampling three CoTs from each dataset, identifying the answer positions (see Sec. 4.1), and assigning the corresponding training labels. We evaluate our method and all baselines on AIME 2025 [1], MATH-500 [22], HumanEval [2], and GPQA [38]. Additional details on our training datasets are available in App. B.3.

Training and Inference. During training, we optimize for high performance on a holdout validation set for our prediction task; we choose our model based solely on how well it performs on the binary predictive task, without peeking at the evaluation dataset performance. Our validation metric of choice is the Macro-F1 score. Additional details on training TERMINATOR are provided in App. B.4.

We use vLLM [20] with asynchronous requests to sample CoTs when curating our training datasets. During inference with our trained model, a sliding window of the 10 most recent predictions is used, and the </think> token is injected when more than 50% of the labels are 1 (majority voting). Our main results reported in this paper use a threshold of 0.7 to predict 1. Additional details on TERMINATOR’s selected inference parameters are provided in App. B.5.

##### 5.2 TERMINATOR: Main Results

Fig. 1 shows the performance of TERMINATOR and relevant baselines with respect to the Compression Rate (lower is better) and Accuracy (higher is better). To ensure a fair comparison, all methods are evaluated using the same CoTs that are used in the vanilla baseline, except the NoThinking baseline, for which no CoT is generated. While Fig. 1 shows that TERMINATOR achieves a better overall compression-performance trade-off over prior methods, the same results—presented in table format—in Table 7 show that TERMINATOR also achieves best or second best performance on 28 out of 32 metrics. Notably, while methods such as Dynasor achieve aggressive token reduction, they

do so at the cost of significant accuracy degradation. TERMINATOR consistently occupies a favorable position on the accuracy-efficiency Pareto frontier across all four evaluated LRMs, demonstrating that its advantages are robust to model architecture and scale.

##### 5.3 Ablation Studies

We further run ablation studies on TERMINATOR with respect to its (1) latency and throughput over the vanilla LRM, (2) performance against the truncated early-exit baseline, (3) out-of-distribution (OOD) performance, and (4) TERMINATOR’s recovery of our observed answer signal phenomena (Figs. 2 and 3). Our results in this section will be reported mostly on Qwen3-8B alone, with results on Qwen3-14B, Ministral-3-8B, and Ministral-3-14B reported in Fig. 1 and Table 7.

Table 1: Latency Analysis. Latency and throughput benchmarks on MATH-500 problems for Qwen3-based vanilla and TERMINATOR models. Values are mean ˘ 95% CI.

Method Latency (s) Throughput

(tok/s) Qwen3-8B

Vanilla 32.68 ˘ 9.59 151.5 ˘ 4.4 Terminator 14.10 ˘ 6.27 135.2 ˘ 2.0

Latency Analysis. Table 1 shows the results of our latency analysis; TERMINATOR halves the average latency over the vanilla LRM, despite incurring a small overhead of 10.8% for Qwen3-8B and 7.5% for Qwen3-14B, respectively. Note that as the base LRM size increases, TERMI-

###### Qwen3-14B

Vanilla 43.38 ˘ 13.98 98.0 ˘ 2.0 Terminator 18.76 ˘ 6.52 90.6 ˘ 0.8

NATOR incurs a proportionally smaller overhead since its architecture (a single transformer layer and an FFN) remains fixed. For our analysis, we develop a vLLM-compatible implementation of TERMINATOR and compare with running the vanilla LRM in vLLM. Both methods are evaluated on the same subset of MATH-500 questions with a batch size of 1, disabled prefix caching, and on a single GH200.

Hindsight-Optimal CoTs. Since TERMINATOR is trained on hindsight-optimal reasoning length (HORL) CoTs, it is natural to ask where TERMINATOR lies on the accuracy-compression frontier relative to the ground-truth HORL, which is shown in Fig. 6. Each dot on the curves represents the accuracy when each CoT was truncated early, and the LRM was forced to give a final solution and final answer. The diamond-shaped markers represent the position of the first occurrence of aˆ, and therefore represent the points corresponding to hindsight-optimal reasoning. As expected, the accuracy remains constant after this point, showing that additional reasoning beyond aˆ does not yield better performance. We plot TERMINATOR alongside these curves to show how close it is to the hindsight-optimal CoT length and performance. Even though the HORL baseline is not achievable by any method in principle, TERMINATOR is notably close to it for all datasets.

###### Accuracy vs Compression Rate by Dataset

1.0

0.8

0.6

Accuracy

0.4

0.2

MATH-500

GPQA

Final Answer Position

AIME25

HumanEval

Terminator

0.0

0.0 0.2 0.4 0.6 0.8 1.0

Compression Rate

Figure 6: Effects of Early CoT Termination. Test set CoTs are evaluated after truncating them at various points via </think> and asking the LRM for a final solution and answer. Diamond-shaped markers show the hindsight-optimal answer position (not achievable). TERMINATOR is close to optimality on all datasets.

OOD Evaluation. We train separate models on each of the four tasks in our training dataset and evaluate them on the test datasets. Whereas our main results in Fig. 1 use TERMINATOR trained on all four tasks, this experiment trains on one task at a time to assess OOD generalization. Fig. 13 reports compression rate and accuracy, with rows denoting the training dataset, columns denoting the test dataset, and cell values denoting test performance. The results show that compression is best in-distribution, i.e., along the diagonal, but accuracy does not always follow the same pattern. For example, training on OpenScience yields the lowest GPQA accuracy despite GPQA being in-distribution, whereas training on the OOD OpenCoder-SFT dataset improves accuracy but worsens compression to 96%. Thus, OOD evaluation can slightly improve accuracy but often delays exiting, thereby reducing token savings on data not seen during training.

TERMINATOR Recovers Early-Exit Signals. We further highlight that using TERMINATOR’s predicted exit positions, we recover the same event-locked averaging (Fig. 2) and “thinking token” frequency (Fig. 3) phenomena. Fig. 10 mirrors Fig. 2, but uses all test samples rather than

3,200 randomly selected training samples. Its left and center panels show event-locked average Token-Confidence using ground-truth and TERMINATOR-predicted answer positions, respectively, while the right panel shows that prediction errors are concentrated near zero, with a median difference of 7. This alignment helps explain why TERMINATOR recovers most of the same signal. Similarly, Fig. 11 parallels Fig. 3 by overlaying scatter plots computed from ground-truth and predicted answer positions. The inset axes show nearly identical above-diagonal percentages, indicating that TERMINATOR preserves the same before/after “thinking token” usage biases. Together, Figs. 10 and 11 show that training TERMINATOR on the LRM’s hidden states is sufficient to independently recover the early-exit signals identified above, justifying our approach.

### 6 Related Work

Prompt Compression. This line of work is concerned with compressing the input prompt (or context) before passing it to an LLM. Some methods use soft-prompts [30, 5, 10, 37] to compress tokenized inputs into a sequence of embeddings. These embeddings serve as the LLM’s input, allowing richer expressivity, but they are not amenable to black-box LLMs and are difficult to analyze theoretically. Other methods use hard-prompts [17, 16, 35, 32], keeping the final compressed input prompt fixed to the same token vocabulary as the LLM.

Efficient Reasoning. Analogously to soft-prompt compression, latent or continuous reasoning is a technique where reasoning unfolds across latent hidden states rather than discrete tokens. Methods like Coconut [12], CCoT [4], and Soft Thinking [52] feed the LLM’s output embeddings back into the input of the LLM during the reasoning stage, which significantly decreases the number of passes through the LLM before arriving at the final answer. LightThinker [51] uses an idea similar to AutoCompressor [5], where each reasoning step is generated as discrete tokens first, compressed, and then the compressed summary of the reasoning thus far is fed back into the LLM to generate the next step.

Early-Exit Reasoning. These methods seek to make reasoning more efficient by terminating the CoT early. All existing methods use a consistency-based approach, injecting the </think> token

- at various points to force the model to generate an answer or a useful signal. Some methods, like EAT [41], DEER [46], ES-CoT [29], and Dynasor [7] are training-free; they track signals throughout the reasoning process and exit when a threshold is crossed. Other methods, like SpecExit [47], Learn To Stop [24], Thought Calibration [43], and FlashThink [15] rely on training a separate probe classifier by using consistency as the main approach for gathering their training signals. By contrast, our work constructs a training signal to predict the immediate arrival of aˆ, thereby training on hindsight-optimal length CoTs. In addition, our work does not require threshold tuning on validation data, which is needed for Learn To Stop and Thought Calibration.

### 7 Conclusion

We present TERMINATOR, an early-exit method for LRM reasoning. Training TERMINATOR requires an optimal-length dataset of CoTs, which can be obtained through our robust answer extraction, identification, and verification pipeline. Furthermore, we provide novel analysis and insights into the behaviors of an LRM’s (1) Token-Confidence during reasoning (Fig. 2), and (2) shift in “thinking token” usage. While our training data curation pipeline works well, future work can explore making training more efficient as tens of thousands of CoTs are used to train TERMINATOR.

Finally, we liken the averaged result in Fig. 2 to the field of event-related potential (ERP) research. An ERP is a measurable brain response elicited by a sensory, cognitive, or motor event, captured by electroencephalogram (EEG) recordings [26]. However, EEG recordings are often noisy, so ERPs are estimated using time-locked statistical estimators (e.g. averaging) across multiple EEG trials. While we do not claim that our findings will align exactly with ERP research, it is quite interesting that meaningful and observable signals can be extracted from LRMs using similar approaches, and we believe this warrants further exploration in future work.

### References

- [1] Art of Problem Solving. Aime problems and solutions. https://artofproblemsolving. com/wiki/index.php/AIME_Problems_and_Solutions. Accessed: 2026-01-07.

- [2] Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harrison Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian, Clemens Winter, Philippe Tillet, Felipe Petroski Such, Dave Cummings, Matthias Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss, William Hebgen Guss, Alex Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain, William Saunders, Christopher Hesse, Andrew N. Carr, Jan Leike, Joshua Achiam, Vedant Misra, Evan Morikawa, Alec Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie Mayer, Peter Welinder, Bob McGrew, Dario Amodei, Sam McCandlish, Ilya Sutskever, and Wojciech Zaremba. Evaluating large language models trained on code. CoRR, abs/2107.03374,

2021. URL https://arxiv.org/abs/2107.03374.

- [3] Xingyu Chen, Jiahao Xu, Tian Liang, Zhiwei He, Jianhui Pang, Dian Yu, Linfeng Song, Qiuzhi Liu, Mengfei Zhou, Zhuosheng Zhang, Rui Wang, Zhaopeng Tu, Haitao Mi, and Dong Yu. Do NOT think that much for 2+3=? on the overthinking of long reasoning models. In Forty-second International Conference on Machine Learning, 2025. URL https://openreview.net/ forum?id=MSbU3L7V00.
- [4] Jeffrey Cheng and Benjamin Van Durme. Compressed chain of thought: Efficient reasoning through dense representations, 2024. URL https://arxiv.org/abs/2412.13171.
- [5] Alexis Chevalier, Alexander Wettig, Anirudh Ajith, and Danqi Chen. Adapting language models to compress contexts. In Houda Bouamor, Juan Pino, and Kalika Bali, editors, Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 3829–3846, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023. emnlp-main.232. URL https://aclanthology.org/2023.emnlp-main.232/.
- [6] Bowen Ding, Yuhan Chen, Futing Wang, Lingfeng Ming, and Tao Lin. Do thinking tokens help or trap? towards more efficient large reasoning model, 2025. URL https://arxiv.org/ abs/2506.23840.
- [7] Yichao Fu, Junda Chen, Siqi Zhu, Zheyu Fu, Zhongdongming Dai, Yonghao Zhuang, Yian Ma, Aurick Qiao, Tajana Rosing, Ion Stoica, and Hao Zhang. Efficiently scaling llm reasoning with certaindex, 2025. URL https://arxiv.org/abs/2412.20993.
- [8] Yichao Fu, Xuewei Wang, Yuandong Tian, and Jiawei Zhao. Deep think with confidence, 2025. URL https://arxiv.org/abs/2508.15260.
- [9] Jiaxuan Gao, Shu Yan, Qixin Tan, Lu Yang, Shusheng Xu, Wei Fu, Zhiyu Mei, Kaifeng Lyu, and Yi Wu. How far are we from optimal reasoning efficiency?, 2025. URL https: //arxiv.org/abs/2506.07104.
- [10] Tao Ge, Hu Jing, Lei Wang, Xun Wang, Si-Qing Chen, and Furu Wei. In-context autoencoder for context compression in a large language model. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=uREj4ZuGJE.
- [11] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, Xiaokang Zhang, Xingkai Yu, Yu Wu, Z. F. Wu, Zhibin Gou, Zhihong Shao, Zhuoshu Li, Ziyi Gao, Aixin Liu, Bing Xue, Bingxuan Wang, Bochao Wu, Bei Feng, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chong Ruan, Damai Dai, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fucong Dai, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang, Hanwei Xu, Honghui Ding, Huazuo Gao, Hui Qu, Hui Li, Jianzhong Guo, Jiashi Li, Jingchang Chen, Jingyang Yuan, Jinhao Tu, Junjie Qiu, Junlong Li, J. L. Cai, Jiaqi Ni, Jian Liang, Jin Chen, Kai Dong, Kai Hu, Kaichao You, Kaige Gao, Kang Guan, Kexin Huang, Kuai Yu, Lean Wang, Lecong Zhang, Liang Zhao, Litong Wang, Liyue Zhang, Lei Xu, Leyi Xia, Mingchuan Zhang, Minghua Zhang, Minghui Tang, Mingxu Zhou, Meng Li, Miaojun Wang, Mingming Li, Ning Tian, Panpan Huang, Peng Zhang, Qiancheng Wang, Qinyu Chen, Qiushi Du, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan, Runji Wang, R. J. Chen, R. L. Jin, Ruyi Chen, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shengfeng Ye, Shiyu Wang, Shuiping Yu, Shunfeng Zhou, Shuting Pan, S. S. Li, Shuang Zhou, Shaoqing Wu, Tao Yun, Tian Pei, Tianyu Sun, T. Wang, Wangding Zeng, Wen Liu, Wenfeng Liang, Wenjun Gao,

- Wenqin Yu, Wentao Zhang, W. L. Xiao, Wei An, Xiaodong Liu, Xiaohan Wang, Xiaokang Chen, Xiaotao Nie, Xin Cheng, Xin Liu, Xin Xie, Xingchao Liu, Xinyu Yang, Xinyuan Li, Xuecheng Su, Xuheng Lin, X. Q. Li, Xiangyue Jin, Xiaojin Shen, Xiaosha Chen, Xiaowen Sun, Xiaoxiang Wang, Xinnan Song, Xinyi Zhou, Xianzu Wang, Xinxia Shan, Y. K. Li, Y. Q. Wang, Y. X. Wei, Yang Zhang, Yanhong Xu, Yao Li, Yao Zhao, Yaofeng Sun, Yaohui Wang, Yi Yu, Yichao Zhang, Yifan Shi, Yiliang Xiong, Ying He, Yishi Piao, Yisong Wang, Yixuan Tan, Yiyang Ma, Yiyuan Liu, Yongqiang Guo, Yuan Ou, Yuduan Wang, Yue Gong, Yuheng Zou, Yujia He, Yunfan Xiong, Yuxiang Luo, Yuxiang You, Yuxuan Liu, Yuyang Zhou, Y. X. Zhu, Yanping Huang, Yaohui Li, Yi Zheng, Yuchen Zhu, Yunxian Ma, Ying Tang, Yukun Zha, Yuting Yan, Z. Z. Ren, Zehui Ren, Zhangli Sha, Zhe Fu, Zhean Xu, Zhenda Xie, Zhengyan Zhang, Zhewen Hao, Zhicheng Ma, Zhigang Yan, Zhiyu Wu, Zihui Gu, Zijia Zhu, Zijun Liu, Zilin Li, Ziwei Xie, Ziyang Song, Zizheng Pan, Zhen Huang, Zhipeng Xu, Zhongyu Zhang, and Zhen Zhang. Deepseek-r1 incentivizes reasoning in llms through reinforcement learning. Nature, 645(8081):633–638, September 2025. ISSN 1476-4687. doi: 10.1038/s41586-025-09422-z. URL http://dx.doi.org/10.1038/s41586-025-09422-z.
- [12] Shibo Hao, Sainbayar Sukhbaatar, DiJia Su, Xian Li, Zhiting Hu, Jason Weston, and Yuandong Tian. Training large language models to reason in a continuous latent space, 2025. URL https://arxiv.org/abs/2412.06769.
- [13] Siming Huang, Tianhao Cheng, Jason Klein Liu, Jiaran Hao, Liuyihan Song, Yang Xu, J. Yang, J. H. Liu, Chenchen Zhang, Linzheng Chai, Ruifeng Yuan, Zhaoxiang Zhang, Jie Fu, Qian Liu, Ge Zhang, Zili Wang, Yuan Qi, Yinghui Xu, and Wei Chu. Opencoder: The open cookbook for top-tier code large language models. 2024. URL https://arxiv.org/pdf/2411.04905.
- [14] Hugging Face. Qwen3-8B model card. https://web.archive.org/web/ 20260502201803/https://huggingface.co/Qwen/Qwen3-8B, 2026. Archived May 2, 2026.
- [15] Guochao Jiang, Guofeng Quan, Zepeng Ding, Ziqin Luo, Dixuan Wang, and Zheng Hu. Flashthink: An early exit method for efficient reasoning, 2025. URL https://arxiv.org/ abs/2505.13949.
- [16] Huiqiang Jiang, Qianhui Wu, Xufang Luo, Dongsheng Li, Chin-Yew Lin, Yuqing Yang, and Lili Qiu. LongLLMLingua: Accelerating and enhancing LLMs in long context scenarios via prompt compression. In ICLR 2024 Workshop on Mathematical and Empirical Understanding of Foundation Models, 2024. URL https://openreview.net/forum?id=9YvfRrpmyw.
- [17] Hoyoun Jung and Kyung-Joong Kim. Discrete prompt compression with reinforcement learning. IEEE Access, 12:72578–72587, 2024. ISSN 2169-3536. doi: 10.1109/access.2024.3403426. URL http://dx.doi.org/10.1109/ACCESS.2024.3403426.
- [18] Yu Kang, Xianghui Sun, Liangyu Chen, and Wei Zou. C3ot: Generating shorter chainof-thought without compromising effectiveness. Proceedings of the AAAI Conference on Artificial Intelligence, 39(23):24312–24320, Apr. 2025. doi: 10.1609/aaai.v39i23.34608. URL https://ojs.aaai.org/index.php/AAAI/article/view/34608.
- [19] Zhewei Kang, Xuandong Zhao, and Dawn Song. Scalable best-of-n selection for large language models via self-certainty, 2025. URL https://arxiv.org/abs/2502.18581.
- [20] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles, 2023.
- [21] Celine Lee, Alexander M. Rush, and Keyon Vafa. Critical thinking: Which kinds of complexity govern optimal reasoning length?, 2025. URL https://arxiv.org/abs/2504.01935.
- [22] Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In ICLR,

###### 2024. URL https://openreview.net/forum?id=v8L0pN6EOi.

- [23] Alexander H. Liu, Kartik Khandelwal, Sandeep Subramanian, Victor Jouault, Abhinav Rastogi, Adrien Sadé, Alan Jeffares, Albert Jiang, Alexandre Cahill, Alexandre Gavaudan, Alexandre Sablayrolles, Amélie Héliou, Amos You, Andy Ehrenberg, Andy Lo, Anton Eliseev, Antonia Calvi, Avinash Sooriyarachchi, Baptiste Bout, Baptiste Rozière, Baudouin De Monicault, Clémence Lanfranchi, Corentin Barreau, Cyprien Courtot, Daniele Grattarola, Darius Dabert, Diego de las Casas, Elliot Chane-Sane, Faruk Ahmed, Gabrielle Berrada, Gaëtan Ecrepont, Gauthier Guinet, Georgii Novikov, Guillaume Kunsch, Guillaume Lample, Guillaume Martin, Gunshi Gupta, Jan Ludziejewski, Jason Rute, Joachim Studnia, Jonas Amar, Joséphine Delas, Josselin Somerville Roberts, Karmesh Yadav, Khyathi Chandu, Kush Jain, Laurence Aitchison, Laurent Fainsin, Léonard Blier, Lingxiao Zhao, Louis Martin, Lucile Saulnier, Luyu Gao, Maarten Buyl, Margaret Jennings, Marie Pellat, Mark Prins, Mathieu Poirée, Mathilde Guillaumin, Matthieu Dinot, Matthieu Futeral, Maxime Darrin, Maximilian Augustin, Mia Chiquier, Michel Schimpf, Nathan Grinsztajn, Neha Gupta, Nikhil Raghuraman, Olivier Bousquet, Olivier Duchenne, Patricia Wang, Patrick von Platen, Paul Jacob, Paul Wambergue, Paula Kurylowicz, Pavankumar Reddy Muddireddy, Philomène Chagniot, Pierre Stock, Pravesh Agrawal, Quentin Torroba, Romain Sauvestre, Roman Soletskyi, Rupert Menneer, Sagar Vaze, Samuel Barry, Sanchit Gandhi, Siddhant Waghjale, Siddharth Gandhi, Soham Ghosh, Srijan Mishra, Sumukh Aithal, Szymon Antoniak, Teven Le Scao, Théo Cachet, Theo Simon Sorg, Thibaut Lavril, Thiziri Nait Saada, Thomas Chabal, Thomas Foubert, Thomas Robert, Thomas Wang, Tim Lawson, Tom Bewley, Tom Bewley, Tom Edwards, Umar Jamil, Umberto Tomasini, Valeriia Nemychnikova, Van Phung, Vincent Maladière, Virgile Richard, Wassim Bouaziz, Wen-Ding Li, William Marshall, Xinghui Li, Xinyu Yang, Yassine El Ouahidi, Yihan Wang, Yunhao Tang, and Zaccharie Ramzi. Ministral 3, 2026. URL https://arxiv.org/abs/2601.08584.
- [24] Xin Liu and Lu Wang. Answer convergence as a signal for early stopping in reasoning, 2025. URL https://arxiv.org/abs/2506.02536.
- [25] Chenwei Lou, Zewei Sun, Xinnian Liang, Meng Qu, Wei Shen, Wenqi Wang, Yuntao Li, Qingping Yang, and Shuangzhi Wu. Adacot: Pareto-optimal adaptive chain-of-thought triggering via reinforcement learning, 2025. URL https://arxiv.org/abs/2505.11896.
- [26] Steven J. Luck. An introduction to the event-related potential technique / Steven J. Luck. The MIT Press, Cambridge, Massachusetts, second edition. edition, 2014. ISBN 0-262-32406-7.
- [27] Haotian Luo, Li Shen, Haiying He, Yibo Wang, Shiwei Liu, Wei Li, Naiqiang Tan, Xiaochun Cao, and Dacheng Tao. O1-pruner: Length-harmonizing fine-tuning for o1-like reasoning pruning, 2025. URL https://arxiv.org/abs/2501.12570.
- [28] Wenjie Ma, Jingxuan He, Charlie Snell, Tyler Griggs, Sewon Min, and Matei Zaharia. Reasoning models can be effective without thinking, 2025. URL https://arxiv.org/abs/2504. 09858.
- [29] Minjia Mao, Bowen Yin, Yu Zhu, and Xiao Fang. Early stopping chain-of-thoughts in large language models, 2025. URL https://arxiv.org/abs/2509.14004.
- [30] Jesse Mu, Xiang Lisa Li, and Noah Goodman. Learning to compress prompts with gist tokens. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. URL https://openreview.net/forum?id=2DtxPCL3T5.
- [31] Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi, Luke Zettlemoyer, Percy Liang, Emmanuel Candès, and Tatsunori Hashimoto. s1: Simple test-time scaling, 2025. URL https://arxiv.org/abs/2501.19393.
- [32] Alliot Nagle, Adway Girish, Marco Bondaschi, Michael Gastpar, Ashok Vardhan Makkuva, and Hyeji Kim. Fundamental limits of prompt compression: A rate-distortion framework for black-box language models. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang, editors, Advances in Neural Information Processing Systems, volume 37, pages 94934–94970. Curran Associates, Inc., 2024. doi: 10.52202/079017-3009. URL https://proceedings.neurips.cc/paper_files/paper/ 2024/file/ac8fbba029dadca99d6b8c3f913d3ed6-Paper-Conference.pdf.

- [33] NVIDIA. OpenScience dataset (v1). https://huggingface.co/datasets/nvidia/ OpenScience, 2025. Last updated: June 18 (per repository history). Accessed: 2026-0107.
- [34] OpenAI. Learning to reason with llms. https://openai.com/index/ learning-to-reason-with-llms/, September 12 2024. Accessed: 2025-01-19.
- [35] Zhuoshi Pan, Qianhui Wu, Huiqiang Jiang, Menglin Xia, Xufang Luo, Jue Zhang, Qingwei Lin, Victor Ruhle, Yuqing Yang, Chin-Yew Lin, H. Vicky Zhao, Lili Qiu, and Dongmei Zhang. LLMLingua-2: Data distillation for efficient and faithful task-agnostic prompt compression. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar, editors, Findings of the Association for Computational Linguistics ACL 2024, pages 963–981, Bangkok, Thailand and virtual meeting, August 2024. Association for Computational Linguistics. URL https://aclanthology. org/2024.findings-acl.57.
- [36] Chen Qian, Dongrui Liu, Haochen Wen, Zhen Bai, Yong Liu, and Jing Shao. Demystifying reasoning dynamics with mutual information: Thinking tokens are information peaks in llm reasoning. arXiv preprint arXiv:2506.02867, 2025.
- [37] Guanghui Qin, Corby Rosset, Ethan Chau, Nikhil Rao, and Benjamin Van Durme. Dodo: Dynamic contextual compression for decoder-only LMs. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar, editors, Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 9961–9975, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-long.536. URL https://aclanthology.org/2024.acl-long.536/.
- [38] David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R. Bowman. GPQA: A graduate-level googleproof q&a benchmark. In First Conference on Language Modeling, 2024. URL https: //openreview.net/forum?id=Ti67584b98.
- [39] Vaishnavi Shrivastava, Ahmed Awadallah, Vidhisha Balachandran, Shivam Garg, Harkirat Behl, and Dimitris Papailiopoulos. Sample more to think less: Group filtered policy optimization for concise reasoning, 2025. URL https://arxiv.org/abs/2508.09726.
- [40] Chenlong Wang, Yuanning Feng, Dongping Chen, Zhaoyang Chu, Ranjay Krishna, and Tianyi Zhou. Wait, we don’t need to “wait”! removing thinking tokens improves reasoning efficiency. In Christos Christodoulopoulos, Tanmoy Chakraborty, Carolyn Rose, and Violet Peng, editors, Findings of the Association for Computational Linguistics: EMNLP 2025, pages 7459–7482, Suzhou, China, November 2025. Association for Computational Linguistics. ISBN 979-889176-335-7. doi: 10.18653/v1/2025.findings-emnlp.394. URL https://aclanthology. org/2025.findings-emnlp.394/.
- [41] Xi Wang, James McInerney, Lequn Wang, and Nathan Kallus. Entropy after x/Thinky for reasoning model early exiting, 2025. URL https://arxiv.org/abs/2509.26522.
- [42] Chao Wu, Baoheng Li, Mingchen Gao, Yu Tian, and Zhenyi Wang. From efficiency to adaptivity: A deeper look at adaptive reasoning in large language models, 2026. URL https: //arxiv.org/abs/2511.10788.
- [43] Menghua Wu, Cai Zhou, Stephen Bates, and Tommi Jaakkola. Thought calibration: Efficient and confident test-time scaling. In Christos Christodoulopoulos, Tanmoy Chakraborty, Carolyn Rose, and Violet Peng, editors, Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 14302–14316, Suzhou, China, November 2025. Association for Computational Linguistics. ISBN 979-8-89176-332-6. doi: 10.18653/v1/2025.emnlp-main.722. URL https://aclanthology.org/2025.emnlp-main.722/.
- [44] Yuyang Wu, Yifei Wang, Tianqi Du, Stefanie Jegelka, and Yisen Wang. When more is less: Understanding chain-of-thought length in LLMs. In Workshop on Reasoning and Planning for Large Language Models, 2025. URL https://openreview.net/forum?id=W8dxn7hBkO.

- [45] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jing Zhou, Jingren Zhou, Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang, Le Yu, Lianghao Deng, Mei Li, Mingfeng Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin, Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, and Zihan Qiu. Qwen3 technical report, 2025. URL https://arxiv.org/abs/2505.09388.
- [46] Chenxu Yang, Qingyi Si, Yongjie Duan, Zheliang Zhu, Chenyu Zhu, Qiaowei Li, Minghui Chen, Zheng Lin, and Weiping Wang. Dynamic early exit in reasoning models, 2025. URL https://arxiv.org/abs/2504.15895.
- [47] Rubing Yang, Huajun Bai, Song Liu, Guanghua Yu, Runzhi Fan, Yanbin Dang, Jiejing Zhang, Kai Liu, Jianchen Zhu, and Peng Chen. Specexit: Accelerating large reasoning model via speculative exit, 2025. URL https://arxiv.org/abs/2509.24248.
- [48] Jingyang Yi, Justin Wang, and Sida Li. Shorterbetter: Guiding reasoning models to find optimal inference length for efficient reasoning. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. URL https://openreview.net/forum?id= MJvwM5dBZM.
- [49] Anqi Zhang, Yulin Chen, Jane Pan, Chen Zhao, Aurojit Panda, Jinyang Li, and He He. Reasoning models know when they’re right: Probing hidden states for self-verification. In Second Conference on Language Modeling, 2025. URL https://openreview.net/forum?id= O6I0Av7683.
- [50] Jiajie Zhang, Nianyi Lin, Lei Hou, Ling Feng, and Juanzi Li. AdaptThink: Reasoning models can learn when to think. In Christos Christodoulopoulos, Tanmoy Chakraborty, Carolyn Rose, and Violet Peng, editors, Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 3716–3730, Suzhou, China, November 2025. Association for Computational Linguistics. ISBN 979-8-89176-332-6. doi: 10.18653/v1/2025.emnlp-main.184. URL https://aclanthology.org/2025.emnlp-main.184/.
- [51] Jintian Zhang, Yuqi Zhu, Mengshu Sun, Yujie Luo, Shuofei Qiao, Lun Du, Da Zheng, Huajun Chen, and Ningyu Zhang. LightThinker: Thinking step-by-step compression. In Christos Christodoulopoulos, Tanmoy Chakraborty, Carolyn Rose, and Violet Peng, editors, Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 13307– 13328, Suzhou, China, November 2025. Association for Computational Linguistics. ISBN 979-8-89176-332-6. doi: 10.18653/v1/2025.emnlp-main.673. URL https://aclanthology. org/2025.emnlp-main.673/.
- [52] Zhen Zhang, Xuehai He, Weixiang Yan, Ao Shen, Chenyang Zhao, Shuohang Wang, Yelong Shen, and Xin Eric Wang. Soft thinking: Unlocking the reasoning potential of llms in continuous concept space, 2025. URL https://arxiv.org/abs/2505.15778.

### A Limitations and Broader Impacts

- A.1 Limitations

Our approach relies on the assumption that exiting at the earliest occurrence of the final answer is sufficient for the LRM to use that same answer in the final solution. Although rare, the LRM does not always commit to the exited answer in the CoT, which can cause the exited response to be marked incorrect even when the vanilla LRM, using full reasoning, eventually obtains the correct answer. Furthermore, the practicality of our approach depends on the final answer being present within the CoT, which might not always be the case, especially for open-ended generation tasks.

- A.2 Broader Impacts

This work aims to advance machine learning research by improving the efficiency of reasoning in large language models through early-exit mechanisms based on model confidence. Potential benefits include reduced energy consumption, faster inference, and lower deployment costs, thereby improving the scalability and environmental sustainability of language model systems.

At the same time, confidence-based early termination introduces the risk of over-confidence and premature conclusions, particularly in settings where incorrect outputs may have significant consequences. These risks underscore the importance of careful calibration, evaluation, and responsible deployment practices. Our work does not introduce new classes of societal harm beyond those already associated with large language models, but it highlights the need for continued research into reliable confidence estimation and safe adaptive inference.

### B Additional Details on Our Methods

##### B.1 Early Answer Extraction, Identification, and Verification

Algorithm 1 contains pseudocode for our pipeline. i‹, the index of the earliest token position containing aˆ, is used to construct the label set of our training data by setting all positions prior to i‹ to 0 and setting all positions after i‹ to 1. Each of the three steps (extraction, identification, and verification) requires separate calls to an LRM; please refer to our codebase for details on the exact system prompts that we used for each step.

##### B.2 Failure Modes of Pattern Matching for Answer Position Identification

The following are examples of failure modes for pattern matching while extracting the earliest arrival of aˆ from an CoT, thus motivating our LRM-based extraction pipeline:

- 1. Numerical answers. A numerical value may appear frequently throughout the CoT in intermediate calculations, problem restatements, or discarded solution attempts, making it impossible to distinguish these occurrences from the true final answer by pattern alone. For example, if the final answer is x = 42, the value 42 may appear dozens of times in prior reasoning steps without having any connection to a logical arrival to that answer. The CoT could simply consider 42 without steps for arriving to it as an answer, or 42 could appear out of pure coincidence as an intermediate value during calculation.
- 2. Mathematical expressions. The same mathematical object can be represented in many syntactically distinct forms. For instance, x2, x**2, pow(x,2), and x ¨ x are semantically

equivalent but would not be matched by any single pattern. Differences in LATEX formatting, Unicode symbols, and whitespace further compound this.

- 3. Python functions. A Python function may not appear as a contiguous block anywhere in the CoT; instead, it may be generated line by line, interspersed with commentary. The final reconstructed answer, therefore, does not exist verbatim in the text, making positional matching fundamentally ill-posed.

Algorithm 1 Answer Span Extraction, Identification, and Verification With Feedback

- 1: Input: CoT r, final solution s, LRM, tokenizer, max retries K
- 2: Output: answer position i‹ (token index where answer is reached)
- 3:
- 4: // Extract final answer value from model output
- 5: aˆ Ð LRMp“Extract the final answer from: ” ` sq
- 6:
- 7: // Iteratively extract and verify span with feedback
- 8: z Ð H // feedback provided to the LRM
- 9: for k “ 1,...,K do
- 10: // Ask LRM to identify a string span containing the first occurrence of aˆ in r
- 11: d Ð LRMp“Find first occurrence of ” ` aˆ ` “ in: ” ` r ` zq
- 12:
- 13: // Verify the identified span contains the answer
- 14: v Ð LRMp“Does ” ` d ` “ contain ” ` aˆ ` “?”q
- 15:
- 16: if v ““ true then
- 17: break // span verified, proceed
- 18: end if
- 19:
- 20: z Ð z ` “\n Previous span ” + d + “ was incorrect, try again”
- 21: end for
- 22:
- 23: // Pattern match span text to get character-wise positioning of the span
- 24: c Ð FuzzyMatchpd,rq // c is an integer-based character index of d in r
- 25:
- 26: // Convert to token position where answer is reached
- 27: i‹ Ð CharToTokenPospc ` lenpdq,r,tokenizerq
- 28:
- 29: return i‹

##### B.3 Training Dataset Details

Our training dataset consists of CoTs from AIME (1983–2024) [1], MATH [22], OpenCoder-SFT [13], and OpenScience [33]. All 933 samples and all 12,000 samples from the AIME (1983– 2024) and MATH datasets, respectively, are used. We randomly select 12,000 samples from the educational_instruct subset of the OpenCoder-SFT-Stage2 dataset, which we refer to as “OpenCoder-SFT” in the main paper. This subset consists of generated and validated Python coding examples. Our sampling procedure for this dataset was not uniform, unlike the others. Instead, problems are grouped by their entry_point field, and sampling is split into rounds, with each round randomly sampling one problem from that group without replacement. Finally, we randomly sample an additional 12,000 samples from the OS-Q3-235B-4 subset of the OpenScience dataset. This subset consists of multiple-choice STEM question-answer pairs that were synthetically generated from Qwen3-235B-A22B [45].

Three CoTs are sampled per problem by the target LRM (we used Qwen and Mistral models), yielding a set of approximately 110,799 CoTs per LRM. The respective answer positions for each set of CoTs is obtained with our extraction method outlined in Sec. 4.1. However, this procedure is not perfect, as even with our retry logic, the answer extractor, identifier, and verifier LRM (Qwen3-30B-A3BThinking-2507) cannot always identify the earliest final answer position. Thus, all three of these steps are successful for roughly 70%–80% of CoTs. Finally, a training-ready dataset for each LRM is formed by preparing label vectors (based on the answer positions), loss masks (based on the positions of <think> and </think>), and tokenizing the CoTs.

0.9

0.7

0.8

0.6

ValidationMacro-F1

TrainBatchLoss

0.7

0.5

0.6

0.4

0.5

0.3

0.2

0.4

0 2000 4000 6000 8000 10000

0 2000 4000 6000 8000 10000

Step

Step

1-transformer-layer-copy +1-mlp-layer (Proposed)

1-transformer-layer-copy

2-mlp-layer

+2-mlp-layer

1-transformer-layer-rand

1-mlp-layer

+1-mlp-layer

- Figure 7: Training Convergence. During training, our proposed TERMINATOR architecture achieves the best balance between the highest Macro-F1 validation scores and the lowest compute overhead.

##### B.4 TERMINATOR Training Details

All models are trained using a consistent set of hyperparameters: batch size of 2, gradient accumulation over 8 steps, no dropout, and a weight decay of 0.01. We initialize the learning rate at 2 ˆ 10´4. Training follows a cosine annealing schedule with 100 steps of linear warmup, decaying to a minimum learning rate of 1 ˆ 10´6.

Optimization is performed with the AdamW optimizer (β1 “ 0.9, β2 “ 0.999, ϵ “ 1 ˆ 10´8), and all runs use a fixed random seed of 1337. Each model consists of a single transformer layer and a single-layer binary classification head. Training proceeds for 2 epochs.

Fig. 7 shows the convergence curves of the training loss and the Macro-F1 scores on a small validation set during training. The validation set is a held-out set of examples from our curated dataset, but is not seen during training. Macro-F1 is the unweighted average of per-class F1 scores, treating all classes equally regardless of their frequency, making it a robust and fair metric for evaluating performance on class-imbalanced datasets. The following architectures are studied:

- 1. 1-transformer-layer-copy+1-mlp-layer: The proposed design, consisting of a single transformer layer with copied initialization followed by a one-layer MLP
- 2. 1-transformer-layer-rand+1-mlp-layer: A variant with random initialization of the transformer layer
- 3. 1-transformer-layer-copy+2-mlp-layer: A variant with a two-layer MLP head
- 4. 1-mlp-layer, 2-mlp-layer: Two MLP-only baselines with one and two layers, respectively.

The transformer block is critical: MLP-only architectures incur a substantial 10–17% drop in MacroF1. When the transformer layer is included, increasing the MLP head from one to two layers yields only a marginal improvement of „0.9%, despite additional computational cost. The choice of initialization for the transformer layer (copied vs. random) yields a tiny improvement („0.1%); we therefore adopt copied initialization by default, as it incurs no additional cost.

##### B.5 TERMINATOR Inference Details

Window Size and Majority Voting. Our proposed strategy for early-exiting based on TERMINATOR’s predictions uses majority voting over a window of the 10 most recent predictions. The window and majority vote serve as a consistency check, preventing a single spurious prediction from triggering an early exit. With Terminator scoring „91% Macro-F1 and „93% accuracy on the validation set, a window of 10 yields fewer than one incorrect prediction on average, making the strict majority (at least 6 of 10) a principled default. Fig. 8 explores the effect of varying the window size while fixing the threshold to 0.7. Performance is stable across all window sizes, confirming that the method is robust to this hyperparameter. As shown in Figures 17–21, once the predicted exit

probability exceeds the threshold, it rarely drops below, so the window size has minimal impact on the majority vote.

###### Compression Rate ( ) by Dataset and Window Size

###### Accuracy ( ) by Dataset and Window Size

1.0

1.0

Window Size w = 3 w = 5 w = 7 w = 10 w = 15

Window Size w = 3 w = 5 w = 7 w = 10 w = 15

| |
|---|

| |
|---|

| |
|---|

| |
|---|

0.8

0.8

| |
|---|

| |
|---|

| |
|---|

| |
|---|

CompressionRate

| |
|---|

| |
|---|

0.6

0.6

Accuracy

0.4

0.4

0.2

0.2

0.0

0.0

MATH-500 AIME25 HumanEval GPQA

MATH-500 AIME25 HumanEval GPQA

Dataset

Dataset

- Figure 8: Effect of Varying Window Size. We fix the threshold to 0.7 and vary the size of the window used by TERMINATOR at inference-time. TERMINATOR is robust to changes in the window size; Figures 17-21 in our manuscript show that the predicted probability of TERMINATOR during CoT generation tends to only increase. This means that once the predicted probability crosses the threshold, it tends not to fall back below it, and the choice of window size has little or no effect.

Thresholding. Fig. 9 shows how the compression rate and the accuracy change as a result of varying the threshold used by TERMINATOR to predict 0 or 1. For example, a threshold τ “ 0.7 requires TERMINATOR to have a predictive confidence of at least 0.7 to predict 1. Interestingly, varying the threshold has a greater impact on compression rate than on accuracy; setting τ “ 0.1 yields 8% and 17% better compression rates for MATH-500 and HumanEval, respectively, compared to τ “ 0.9, with little change in accuracy. For example, from τ “ 0.9 to τ “ 0.1 yields a 4% drop in accuracy but a 27% improvement in the compression rate for GPQA. Overall, this suggests that TERMINATOR learns a stable confidence signal for identifying viable early-exit points: lowering the threshold makes the method more aggressive, substantially improving compression while only modestly affecting accuracy.

MATH-500 AIME25 HumanEval GPQA

Dataset

0.0

0.2

0.4

0.6

0.8

1.0

CompressionRate

Compression Rate ( ) by Dataset and Threshold

Threshold

| |
|---|

= 0.1 = 0.3 = 0.5 = 0.7 = 0.9

| |
|---|

| |
|---|

| |
|---|

| |
|---|

MATH-500 AIME25 HumanEval GPQA

Dataset

0.0

0.2

0.4

0.6

0.8

1.0

Accuracy

Accuracy ( ) by Dataset and Threshold

Threshold

| |
|---|

= 0.1 = 0.3 = 0.5 = 0.7 = 0.9

| |
|---|

| |
|---|

| |
|---|

| |
|---|

- Figure 9: Effect of Varying Predictive Threshold. We fix the window size to 10 and vary the threshold τ used by TERMINATOR to predict whether to exit early. TERMINATOR is robust to changes in the predictive threshold: lowering τ makes the method more aggressive, leading to substantially higher compression rates, while accuracy changes only modestly across datasets. In particular, datasets such as MATH-500 and HumanEval exhibit little performance degradation across thresholds from τ “ 0.1 to τ “ 0.9, suggesting that TERMINATOR learns a stable confidence signal for identifying viable early-exit points.

TERMINATOR’s Context Length. In general, we stick to training and evaluating TERMINATOR with context lengths up to 32,768, since those are the lengths of the training CoTs anyway. However, Mistral models have unusually long CoTs only for AIME problems, so we train and evaluate TERMINATOR for a max length of 65,536 tokens on AIME for Mistral models.

##### B.6 Implementation of the Baseline Methods

Dynasor. Dynasor [7] works by interrupting reasoning at regular token intervals (e.g. every 32, or 64 tokens) by injecting the prompt “Oh, I suddenly got the answer to the whole problem, Final Answer: boxed{” to extract the model’s current answer. The method decides to exit early when consistent answers appear across multiple probing intervals for at least w times. In our experiments, we set w=8 and use a token interval of 64 tokens, following their so-called mild configuration setup.

Thought Calibration. [43] propose training a linear probe to predict optimal stopping points during reasoning generation. The method first segments reasoning trajectories (CoTs) into individual steps, delimited by “\n\n” and containing words like “wait” or “but”. Three probe variants are introduced: Supervised predicts whether the LRM is correct based on current thoughts; Consistent predicts whether the current answer (aˆ) matches the final answer (a); and Novel Leaf predicts whether the current step is a leaf node but not novel.

The stopping decision is controlled by two hyperparameters: tolerance δ (the maximum acceptable risk of stopping incorrectly) that implies a threshold λ (the calibrated probe score cutoff that triggers stopping), and window size (the number of consecutive reasoning steps averaged to smooth predictions before threshold comparison). We retrain the Supervised and Consistent probes on the S1-K dataset [31] for all four models in our experiments: Qwen3-8B, Qwen3-14B, Ministral-3-8B-Reasoning2512, and Ministral-3-8B-Reasoning-2512. For inference, we use the hyperparameters specified in Table 2. As shown in Table 2, the Qwen3 models require lower thresholds compared to the Ministral models, as their probe outputs yield systematically lower confidence scores. Similar model-specific calibration requirements have been observed in prior work [46].

- Table 2: Hyperparameters used for Supervised and Consistent linear probes for Thought Calibration. Ministral models’ full names are Ministral-3-xxB-Reasoning-2512. Tolerance and threshold refer to δ and λ, respectively.

|Model<br><br>|Supervised tolerance threshold window size<br><br>|Consistent tolerance threshold window size|
|---|---|---|
|Qwen3-8B Qwen3-14B Ministral-3-8B Ministral-3-14B<br><br>|0.25 0.6526 10 0.25 0.6377 10 0.10 0.8173 10 0.10 0.8306 10<br><br>|0.25 0.8790 10 0.25 0.8679 10<br><br>0.025 0.9973 10 0.025 0.9973 10|

We reported the results for the Supervised probe in Table 7, as it performed better across test datasets than the Consistent probe. For comparison, we report the results for both probes in Table 3. Note that for the Consistent probe with Ministral models, tolerance 0.025 represents the smallest feasible setting among values suggested in [43]. The smallest possible setting, being tolerance 0.01, yields threshold=1.0, resulting in no compression.

- Table 3: Performance comparison of Supervised and Consistent probes for Thought Calibration across models and tasks. Ministral models’ full names are Ministral-3-xxB-Reasoning-2512.

MATH-500 AIME 2025 GPQA HumanEval

Model Acc (Ò) CR (Ó) Acc (Ò) CR (Ó) Acc (Ò) CR (Ó) Acc (Ò) CR (Ó) Supervised Qwen3-8B 90.1% 93.89% 65.8% 81.54% 52.6% 78.87% 71.8% 92.92% Qwen3-14B 89.8% 91.92% 63.3% 71.29% 54.6% 81.90% 74.8% 87.12% Ministral-3-8B 87.7% 87.80% 83.7% 91.16% 47.3% 71.78% 47.2% 87.16% Ministral-3-14B 87.3% 95.86% 59.4% 79.77% 49.5% 73.81% 27.6% 96.25% Consistent Qwen3-8B 88.2% 72.09% 43.1% 54.64% 44.7% 45.17% 70.7% 73.90% Qwen3-14B 85.9% 64.40% 41.7% 45.57% 48.3% 53.83% 70.9% 39.60% Ministral-3-8B 75.8% 80.53% 23.3% 60.50% 42.4% 62.69% 34.2% 75.24% Ministral-3-14B 68.7% 77.40% 9.4% 34.42% 40.6% 59.97% 9.6% 95.90%

Token Confidence Comparison: Ground Truth vs. Terminator (Qwen3-8B) - All Test Samples (n=1132)

###### Ground Truth Answer Position

###### Terminator Answer Position

###### Answer Position Difference (Predicted GT)

Answer Position

Answer Position

Median Difference = 7

26

26

Mean

Mean

±1 SE

±1 SE

24

24

- 100

- 101

- 102

TokenConfidence

22

22

Count

20

20

18

18

16

16

1000 750 500 250 0 250 500 750 1000

1000 750 500 250 0 250 500 750 1000

2000 1500 1000 500 0 500 1000 1500 2000

Relative Position from Answer

Relative Position from Answer

Answer Position Difference

- Figure 10: TERMINATOR Recovers Event-Locked Average Spiking. The exit positions predicted by TERMINATOR (center) recovers similar spiking behavior in the event-locked averaged TokenConfidence as the ground-truth answer positions (left). The histogram of differences between the exit positions (right) shows that TERMINATOR’s predicted exit positions are close to the ground-truth. Note that the y-axis on the histogram is log-scaled.

0 10 5 10 4 10 3 10 2

Occurrence Rate Before Answer

0

10 5

10 4

10 3

10 2

OccurrenceRateAfterAnswer

Token: "hmm"

0

20

40

60

AboveDiag.%

4.3% 3.6%

Ground Truth

Terminator

0 10 5 10 4 10 3 10 2

Occurrence Rate Before Answer

0

10 5

10 4

10 3

10 2

Token: "okay"

0

20

40

60

AboveDiag.%

4.0% 5.4%

Ground Truth

Terminator

0 10 5 10 4 10 3 10 2

Occurrence Rate Before Answer

0

10 5

10 4

10 3

10 2

Token: "another"

0

20

40

60

AboveDiag.%

Ground Truth 63.1% 59.7%

Terminator

Token Occurrence Ratios: Ground Truth vs. Terminator (Qwen3-8B) - All Test Samples (n=1132)

- Figure 11: TERMINATOR Token Usage Biases. The exit positions predicted by TERMINATOR recover the same biases in the “thinking token” occurrence rates as the ground-truth answer positions. The inset axes on each panel show the percentage of dots that lie above the diagonal when the ground-truth and TERMINATOR answer positions are used.

### C Additional Experimental Results

##### C.1 Early-Exit Signal Analysis

TERMINATOR Recovers Early-Exit Signals. As discussed in Sec. 5.3, TERMINATOR’s early-exit predictions can be used to recover similar early-exit signals as when the ground-truth earliest answer position is used. Figs. 10 and 11 establish this claim. Discussion on our findings are provided in Sec. 5.3

Event-Locked Averaging Token-Confidence. The results shown in Figs. 2 and 3 motivate our approach by confirming that the first arrival of aˆ is (1) marked by spiking behavior in the TokenConfidence, which is most easily seen in the event-locked average case, and (2) by a shift in the “thinking token” usage distribution. Focusing on the averaged plots in Fig. 2, we observe that the confidence of the LRM grows up to the point when aˆ is first generated, where the confidence finally peaks. The confidence immediately drops after aˆ is generated, which is intuitive given that the LRM immediately begins to doubt itself, often producing “thinking tokens” like wait or but, signaling uncertainty about the answer that was just generated. The LRM’s confidence improves slightly as it continues to rethink the problem.

Figs. 15 to 18 show the Token-Confidence and log-probabilities for the single-sample and eventlocked averaging case, similar to what’s shown in Fig. 2, separately for each data source. While the exact contours of these two signal types vary for different data sources, the same idea applies to all: the LRM’s Token-Confidence has a sharp spike at the position of the first occurrence of aˆ, followed by a sharp decrease. In all cases, the Token-Confidence then has a quick recovery before plateauing or decaying.

Token Usage Frequency Shift. Beyond providing motivation for our method, the results of Fig. 3 offer some interesting insights on the usage of “thinking tokens.” These plots show the strong usage

- Table 4: Cross-Domain vs. Single-Domain Training. Comparison of accuracy and compression ratio when TERMINATOR is trained on a single-domain (based on Fig. 13) and all domains (based on Table 7). Cross-domain training (the proposed, unified model) and single-domain training are both valid approaches; cross-domain achieves a 1% increase in accuracy over single-domain, but at a 1.5% worse compression rate.

Math Coding Science Method MATH-500 AIME25 HumanEval GPQA Overall

AccÒ CRÓ AccÒ CRÓ AccÒ CRÓ AccÒ CRÓ AccÒ CRÓ

Single-domain training (Fig. 13) 91% 47% 66% 70% 96% 67% 55% 82% 77% 66.5% Cross-domain training (Table 7) 91% 45% 69% 71% 96% 70% 56% 86% 78% 68%

bias that can occur with respect to the first occurrence of aˆ. For example, 63.9% and 91.5% of CoTs contain the tokens hmm and okay more often before aˆ than after it, respectively. Other tokens, like another are more frequently used after. Moreover, Figs. 19 to 23 show that the occurrence rates can differ drastically between data sources. For example, the token alternatively has an above-diagonal rate of 80.4% for MATH, but only 19.2% for OpenCoder-SFT.

Fig. 3 also indicates the length of each CoT by its dot size. Upon closer inspection, it appears that there is some correlation between the dot size and the occurrence rates. We plot the before and after occurrence rates with respect to CoT length for these three tokens in Fig. 24. Interestingly, shorter CoTs do in fact correlate with higher token occurrences for these three “thinking tokens.”

Fig. 19 shows an expanded view of Fig. 3, including six additional “thinking tokens.” Figs. 20 to 23 reproduces this expanded plot, but with the data sources separated. Interestingly, the occurrence rates can vary substantially across different data sources. For example, for the token alternatively, only 19.2% of points lie above the diagonal (only 19.2% of CoTs have the token alternatively occurring more frequently after the answer than before), and 45.8% (nearly half!) of the points are at the origin (the token alternatively never occurs in 45.8% of CoTs) for OpenCoder-SFT. However, for MATH, 80.4% of points lie above the diagonal and only 4.1% lie at the origin for the same token. Other tokens, like therefore, are strongly biased toward occurring after the answer.

Fig. 24 shows that the average occurrence rate across CoTs from all data sources changes depending on how long the CoT is. We normalize by the number of tokens occurring before and after the first occurrence of aˆ for the before and after rates, respectively, so these plots suggest that the LRM uses hmm, okay, and another more frequently for shorter CoTs than longer ones.

##### C.2 Adaptive Strategies for TERMINATOR

In principle, adaptive early-exiting strategies offer a promising way to tailor inference to the target domain [42]. There are two design choices for TERMINATOR that are amenable to inference-time adaptation: (1) domain-based model selection, and (2) threshold selection. Domain-based model selection is concerned with training four separate TERMINATOR models on each of the four training data sets separately and choosing the appropriate one at inference time. This contrasts with our proposed approach, which trains TERMINATOR on all domains simultaneously. Similarly, threshold selection considers adaptivity at inference time by setting a separate threshold for each domain, rather than using a single threshold for all domains. We seek to understand how much performance TERMINATOR can gain, if any, by enabling domain adaptation on these two design choices.

- Table 4 shows the outcome of comparing domain-based model selection with our proposed approach. The domain-based model selection results (“single-domain training”) come from the diagonals of Fig. 13, and the results of our proposed method (“cross-domain training”) for TERMINATOR come from Table 7. Overall, both approaches yield good performance, without a clear winner: cross-domain training achieves 1% higher accuracy than single-domain training, but at a 1.5% worse compression rate.

Fig. 12 shows the best possible performance with a domain-adaptive thresholding approach for TERMINATOR for five possible thresholds: 0.1,0.3,0.5,0.7,and0.9. In the figure, the tuples correspond to the thresholds chosen for (MATH-500, AIME 25, HumanEval, GPQA) accordingly. For example, “(0.9,0.3,0.1,0.1)” corresponds to choosing thresholds 0.9,0.3,0.1,0.1 for MATH-500, AIME 25, HumanEval, and GPQA, respectively. Five thresholds across four datasets yield 54 “ 625

###### Pareto Frontier over Threshold Combinations for Qwen3-8B

80

75

Accuracy()

70

65

60

0 20 40 60 80 100

Compression Rate ( )

Terminator (Adaptive Threshold)

Baselines Vanilla

Terminator (Fixed Threshold)

Terminator (0.3, 0.3, 0.1, 0.1) Terminator (0.9, 0.3, 0.1, 0.1) Terminator (0.9, 0.3, 0.3, 0.1) Terminator (0.9, 0.3, 0.5, 0.1) Terminator (0.9, 0.3, 0.5, 0.9) Terminator (0.9, 0.9, 0.5, 0.9) Terminator (0.9, 0.9, 0.7, 0.9)

Terminator (0.1, 0.1, 0.1, 0.1) Terminator (0.3, 0.3, 0.3, 0.3) Terminator (0.5, 0.5, 0.5, 0.5) Terminator (0.7, 0.7, 0.7, 0.7) Terminator (0.9, 0.9, 0.9, 0.9)

| | | |
|---|---|---|
| | | |

NoThinking

DEER

Thought-Calilbration Dynasor

| | |
|---|---|
| | |

- Figure 12: Adaptive Thresholding for TERMINATOR. We plot TERMINATOR at various thresholds (blue stars, fixed for all domains) and compare with a domain-adaptive threshold selection (red stars). The tuples correspond to the thresholds chosen for (MATH-500, AIME 25, HumanEval, GPQA) accordingly. For example, “(0.9, 0.3, 0.1, 0.1)” corresponds to choosing thresholds 0.9, 0.3, 0.1, 0.1 for MATH-500, AIME 25, HumanEval, and GPQA, respectively. We enumerate all possible threshold combinations and retain those that define the Pareto frontier. Domain adaptation provides incremental gains but requires domain knowledge at inference time.

possible combinations, so we show only those that define the Pareto frontier. We use a threshold of 0.7 (represented with “(0.7,0.7,0.7,0.7)” in Fig. 12) for TERMINATOR. While domain-adaptive thresholding (red stars) generally lies above fixed thresholding (blue stars), the improvement in performance is marginal.

However, we note that the results for these domain-adaptive are an upper bound to the performance that can be realized in practice, since perfect knowledge of which domain a given problem belongs to is assumed. In practice, an inference-time strategy is required to appropriately choose the model or threshold, which may introduce error and additional computational overhead. Furthermore, the domain-based model selection approach requires four separately trained models rather than a single unified model, thereby incurring additional compute and memory overhead for model routing.

##### C.3 TERMINATOR Prediction Analysis

TERMINATOR Predictions During Reasoning. Fig. 25 shows the event-locked average of the predicted probabilities from TERMINATOR for each data source, and Fig. 26, Fig. 27, Fig. 28, and Fig. 29 show the predicted probabilities from four randomly chosen examples for AIME25, MATHH500, HumanEval, and GPQA, respectively. The event-locked average and the individual examples from MATH-500 and HumanEval show sharp transitions in predicted confidence at the exiting threshold, with good separation (dotted gray line). However, AIME25 and GPQA examples do not show such a sharp transition, suggesting that it is challenging for TERMINATOR to identify a good exit position for very hard tasks.

Answer Prediction Histograms. Fig. 30 gives an overview of the position of the first occurrence of aˆ in the CoTs for each data source we used for training. Fig. 31 shows the compression rate statistics of TERMINATOR for each test dataset.

###### Compression Rate ( )

###### Accuracy ( )

[Figure 1]

[Figure 2]

1.0

1.0

- 47% 74% 67% 85%
- 48% 70% 72% 85%

- 91% 69% 95% 56%

- 91% 66% 94% 55%
- 92% 74% 96% 58%

- 92% 70% 95% 55%

MATH

MATH

[Figure 3]

[Figure 4]

TrainingDataSource

0.8

0.8

AIME 1983-2024

AIME 1983-2024

0.6

0.6

0.4

0.4

89% 98% 67% 96%

OpenCoder-SFT

OpenCoder-SFT

0.2

0.2

67% 78% 83% 82%

OpenScience

OpenScience

0.0

0.0

MATH-500 AIME25HumanEval GPQA

MATH-500 AIME25HumanEval GPQA

Evaluation Data Source

Evaluation Dataset

- Figure 13: OOD Performance of TERMINATOR. The best trade-off between accuracy and compression rate is achieved when the evaluation set is in-distribution with the training dataset. Here the out-of-distribution performance of TERMINATOR with respect to the compression rate (left) and the accuracy (right) for Qwen3-8B is shown. Training datasets are listed along the row axis, and the evaluation sets are listed across the column axis. For example, training TERMINATOR on MATH and evaluating on HumanEval yields a compression rate of 67% and an accuracy of 95%. Every training dataset has an in-domain evaluation dataset, i.e. MATH Ñ MATH-500, AIME 1983–2024 Ñ AIME25, OpenCoder-SFT Ñ HumanEval, and OpenScience Ñ GPQA.

30% 40% 50% 60%

Token savings (%)

0

200

400

600

800

3Num.break-evenprompts()×10

30% 40% 50% 60%

Token savings (%)

0

10

20

30

40

50

60

GPU-secondssavedperprompt

L=4,000

| |
|---|

L=8,000

| |
|---|

L=16,000

- Figure 14: Pipeline Cost-Benefit Analysis. We report an estimate on the number of prompts to TERMINATOR needed to break even with the compute cost associated with running our data curation pipeline (left) and on the number of GPU-seconds saved per prompt (right). We use the Qwen3-8B throughput data from Table 1 and calculate the compute cost across fixed token savings (i.e. a reduction in the number of tokens by 30%,40%,50%, and 60%) and fixed prompt lengths of 4,000,8,000 and 16,000 tokens. At the time of writing, the number of monthly Qwen3-8B downloads from Hugging Face is „9.9 million [14]. If just a single prompt were run per download on Qwen3-8B with TERMINATOR, the compute offset would be compensated for many times over.

Case Study of TERMINATOR. By manual inspection, we have seen that easy problems have a clearer transition to overthinking, which is well detected by TERMINATOR. However, TERMINATOR may not always detect the appropriate answer position cleanly on harder questions. Figs. 32 to 35 demonstrate this behavior on Qwen3-14B CoTs.

##### C.4 Pipeline Cost-Benefit Analysis

Gathering 110,799 CoTs for training takes approximately 3 days with a single GH200. The pipeline for obtaining the position of aˆ on those CoTs takes roughly one week on 8 GH200s, totaling an estimated 1,416 GPU-hours. Fig. 14 provides a rough estimate of the number of prompts needed to offset the one-time data curation compute cost to train Terminator with Qwen3-8B (left), along with the number of seconds saved per prompt. The x-axis corresponds to the fixed token savings across all prompts, and L represents the length of the CoTs. For example, for a 30% savings, about „200,000 prompts with CoT lengths of 16,000 are sufficient to break even with the data curation step. The

number of prompts required to break even is small compared to the number of monthly downloads for Qwen3-8B on Hugging Face („9.9 million at the time of writing [14]). If just a single prompt were run per download on Qwen3-8B with TERMINATOR, the compute offset for the training data curation would be compensated for many times over.

##### C.5 Disentangling Hindsight-Optimal Labels and Token-Level Exit Prediction

We analyze the relative roles of two design choices in TERMINATOR: (1) hindsight-optimal exit labels and (2) token-level exit prediction. These two ingredients are closely coupled. Hindsight-optimal labels provide the supervision signal that makes token-level predictions meaningful, while token-level predictions are needed to fully exploit the fine-grained structure of those labels. To isolate their effects, we consider all four combinations of hindsight-optimal versus consistency-based labels and token-level versus chunk-level prediction.

We first describe the two alternatives considered in this analysis.

Consistency-based Labels. Consistency-based early-exit methods inject </think> at intermediate points in the CoT and force the model to produce a final answer. One common labeling strategy is to assign a positive exit label to the first intermediate point whose forced answer matches the model’s final answer aˆ. In contrast, our hindsight-optimal labeling assigns a positive label at the first point where aˆ itself appears in the CoT. Thus, rather than repeatedly probing whether the model can already reproduce the final answer, hindsight-optimal labeling directly identifies when the final answer has emerged in the reasoning trajectory.

Chunk-level Prediction. Prior early-exit approaches typically make exit decisions only at predefined checkpoints during generation. These checkpoints may occur at fixed intervals, after structural delimiters such as \n\n, or after heuristic “thinking tokens” such as wait, alternatively, or therefore. In contrast, TERMINATOR predicts whether to exit at every generated token.

The four possible combinations are summarized in Table 5.

- Table 5: Combinations of exit supervision and prediction granularity. TERMINATOR combines hindsight-optimal labels with token-level prediction. The remaining combinations either correspond to prior work, underperform our method, or are computationally infeasible.

Token-level prediction Chunk-level prediction

Hindsight-optimal labels TERMINATOR (proposed) Underperforms TERMINATOR Consistency-based labels Computationally infeasible Prior work (underperforms)

Hindsight-optimal Labels with Token-level Prediction. This is our proposed method, TERMINATOR. It uses hindsight-optimal labels to identify the earliest point at which the final answer appears in the CoT, and trains a token-level predictor to decide whether generation can safely terminate at each position.

Hindsight-optimal Labels with Chunk-level Prediction. To test whether hindsight-optimal labels alone are sufficient, we also train a variant that only makes exit predictions at chunk boundaries, following the prediction granularity used in prior early-exit work. Specifically, we use the \n\n delimiter to define chunks and assign the target exit to the end of the chunk containing the hindsight-optimal exit position. At inference time, predictions are made only at chunk boundaries.

- Table 6 shows that this variant substantially underperforms token-level TERMINATOR. Although the chunk-level variant exits much less aggressively, achieving substantially lower compression rates, it also suffers large accuracy drops across evaluation datasets. For example, the chunk-level variant achieves an overall accuracy of 56.7% and compression rate of 24.1%, compared to 73.7% accuracy and 67.8% compression for TERMINATOR. The gap is particularly large on AIME25 and GPQA, where coarse exit checkpoints appear unable to reliably capture the precise point at which the model has completed the necessary reasoning. These results indicate that token-level granularity is necessary to fully leverage hindsight-optimal supervision.

Consistency-based Labels with Token-level Prediction. A token-level consistency-based method would require injecting </think> after every generated token and forcing the model to produce

Table 6: Effect of chunk-level exit prediction. We train a chunk-level TERMINATOR variant using the \n\n delimiter so that, for each CoT, the positive exit label is assigned to the end of the chunk containing the hindsight-optimal exit position. At inference time, exit predictions are made only at chunk boundaries. All other hyperparameters, including the exit threshold, are kept the same as in TERMINATOR.

Math Coding Science Method MATH-500 AIME25 HumanEval GPQA Overall

AccÒ CRÓ AccÒ CRÓ AccÒ CRÓ AccÒ CRÓ AccÒ CRÓ

TERMINATOR (Proposed) 90.7% 45.1% 69.4% 70.7% 95.7% 69.9% 55.7% 85.7% 77.9% 67.8% Chunk Variant 80.8% 18.1% 34.9% 43.0% 86.6% 18.4% 45.4% 16.9% 61.9% 24.1%

Single Sample and Averaged Time Series for Qwen3-8B - AIME (n=800)

###### Single Sample Token Confidence

###### Event-Locked Average Token Confidence

Final Answer Position

Final Answer Position

40

Mean

30

TokenConfidence

±1 SE

30

25

20

20

10

15

0

0 2000 4000 6000 8000 10000

1000 750 500 250 0 250 500 750 1000

###### Single Sample Log Probabilities

###### Event-Locked Average Log Probabilities

0.5

- 0

- 1

- 2

- 3

- 4

Final Answer Position

Final Answer Position

0.4

Mean

LogProbability

±1 SE

0.3

0.2

0.1

0.0

0 2000 4000 6000 8000 10000

1000 750 500 250 0 250 500 750 1000

Token Index

Relative Position from Answer

- Figure 15: Event-Locked Averaging of Token-Confidence for AIME (1983–2024). A reproduction of Fig. 2, but only using CoTs from 800 randomly selected AIME (1983–2024) problems.

an intermediate final answer at each position. This is computationally prohibitive. For example, assuming a conservative overhead of 1 second per forced-answer generation, labeling a single CoT with 8,000 tokens would require roughly 2.22 hours of compute. Using the same compute budget as for our training data (1,416 GPU hours, see App. C.4), this would label only a small fraction (about 700) of the available CoTs (110,799 in total), yielding too little data for effective training. Thus, while token-level consistency labels are conceptually possible, they are not practical at scale.

Consistency-based labels with chunk-level prediction. This setting corresponds to the dominant design used in prior early-exit work, where the model is probed at coarse checkpoints and exits when an intermediate forced answer is consistent with the final answer. Our main results show that TERMINATOR outperforms these methods. Moreover, the hindsight-optimal chunk-level variant in

- Table 6 performs worse than token-level TERMINATOR, despite using our stronger labeling signal. This demonstrates that hindsight-optimal labels alone are not sufficient: they define a better target, but token-level prediction is needed to realize their benefit.

Overall, these results suggest that hindsight-optimal labels set the achievable performance ceiling by identifying meaningful early-exit positions, while token-level prediction provides the granularity needed to reach that ceiling.

- Table 7: Performance of TERMINATOR and Baselines. Ò Indicates that higher values are better, while Ó indicates that lower values are better. CR is the compression rate, reported here as the mean per-sample compression rate. Tok is the mean number of tokens per sample. Bold and Underlined values highlight the best and second-best performing early exit methods, respectively. TERMINATOR demonstrates superior accuracy-efficiency trade-offs (best or second-best performance across 28 out of 32 metrics). Fig. 1 in Sec. 5 shows the results of this table on the Pareto frontier.

Math Coding Science Method MATH-500 AIME25 HumanEval GPQA Overall

AccÒ TokÓ CRÓ AccÒ TokÓ CRÓ AccÒ TokÓ CRÓ AccÒ TokÓ CRÓ AccÒ CRÓ Qwen3-8B

- Vanilla 91.1% 5,037 100% 74.4% 14,499 100% 94.9% 3,792 100% 58.0% 8,594 100% 79.6% 100% NoThinking 80.7% 809 16.1% 22.0% 2,355 18.6% 84.6% 353 11.8% 46.0% 1,204 15.8% 58.3% 15.6%

- DEER 79.9% 2,602 52.0% 21.4% 10,349 67.8% 93.7% 3,275 83.6% 50.3% 8,553 99.6% 61.3% 75.8% Thought-Calib 90.1% 4,372 93.9% 65.8% 11,014 81.5% 93.9% 3,267 92.9% 56.6% 6,240 78.9% 76.6% 86.8%

- Dynasor 78.3% 1,850 41.0% 48.0% 7,479 48.8% 94.5% 2,883 78.4% 41.7% 2,455 28.4% 65.6% 49.2% TERMINATOR 90.7% 2,425 45.1% 69.4% 10,970 70.7% 95.7% 2,716 69.9% 55.7% 7,543 85.7% 77.9% 67.8%

Qwen3-14B

Vanilla 92.0% 4,598 100% 79.9% 14,255 100% 96.9% 3,296 100% 60.2% 7,628 100% 82.3% 100% NoThinking 84.1% 786 17.5% 26.3% 2,472 19.9% 83.7% 317 12.2% 49.8% 1,265 18.8% 61.0% 17.1% DEER 80.9% 2,501 56.2% 27.6% 10,497 71.0% 96.9% 2,961 87.3% 52.0% 7,451 97.4% 64.5% 78.0% Thought-Calib 89.8% 3,778 92.0% 63.3% 9,429 71.3% 94.3% 2,582 87.1% 57.3% 5,757 81.9% 76.2% 83.1%

- Dynasor 79.6% 1,702 42.4% 61.8% 7,937 52.8% 96.5% 2,611 82.2% 45.7% 2,101 29.1% 70.9% 51.6% TERMINATOR 90.7% 2,261 46.8% 74.2% 10,787 71.0% 97.1% 2,358 70.9% 59.6% 6,798 87.1% 80.4% 68.9%

- Vanilla 93.5% 6,212 100% 92.6% 22,124 100% 97.1% 4,367 100% 63.4% 11,765 100% 86.6% 100% NoThinking 83.2% 1,908 28.1% 43.6% 7,711 36.5% 87.6% 727 16.4% 42.4% 2,106 16.0% 64.2% 24.3% DEER 71.0% 3,791 60.3% 67.1% 17,481 77.0% 80.1% 3,606 84.0% 61.9% 11,312 94.1% 70.0% 78.9% Thought-Calib 87.7% 5,695 87.8% 83.7% 20,358 91.2% 57.9% 3,536 87.2% 50.6% 7,406 71.8% 70.0% 84.5% Dynasor 88.1% 2,967 56.8% 87.6% 15,407 66.3% 96.9% 3,931 88.6% 57.7% 9,766 83.7% 82.6% 73.9%

###### Ministral-3-8B-Reasoning-2512

- TERMINATOR 89.1% 2,863 47.8% 79.1% 15,748 67.8% 96.5% 2,960 66.6% 58.2% 9,588 77.4% 80.7% 64.9% Ministral-3-14B-Reasoning-2512

Vanilla 93.0% 6,385 100% 88.1% 23,694 100% 97.5% 3,918 100% 62.9% 9,539 100% 83.4% 100% NoThinking 79.1% 535 11.7% 20.5% 2,413 13.8% 88.5% 528 14.1% 42.8% 570 6.6% 57.75% 11.5% DEER 69.8% 4,279 74.6% 55.9% 20,049 84.9% 20.5% 1,684 46.5% 58.1% 9,185 95.0% 51.1% 75.3% Thought-Calib 87.3% 5,860 95.9% 59.4% 17,763 79.8% 45.3% 3,465 96.3% 56.2% 6,028 73.8% 62.1% 86.5% Dynasor 86.3% 3,240 55.5% 83.2% 17,920 70.6% 94.7% 3,538 88.9% 55.9% 7,917 85.2% 80.0% 75.1%

- TERMINATOR 90.2% 2,946 43.9% 84.2% 15,518 63.5% 97.8% 2,903 71.0% 61.2% 7,727 76.5% 83.4% 63.7%

Single Sample and Averaged Time Series for Qwen3-8B - MATH (n=800)

###### Single Sample Token Confidence

###### Event-Locked Average Token Confidence

50

35

Final Answer Position

Final Answer Position

40

Mean

TokenConfidence

30

±1 SE

30

25

20

20

10

15

0

0 500 1000 1500 2000 2500 3000 3500

1000 750 500 250 0 250 500 750 1000

###### Single Sample Log Probabilities

###### Event-Locked Average Log Probabilities

0.4

- 0

- 1

- 2

- 3

- 4

Final Answer Position

Final Answer Position

Mean

0.3

Token Index LogProbability

±1 SE

0.2

0.1

0.0

0 500 1000 1500 2000 2500 3000 3500

1000 750 500 250 0 250 500 750 1000

Relative Position from Answer

- Figure 16: Event-Locked Averaging of Token-Confidence for MATH. A reproduction of Fig. 2, but only using CoTs from 800 randomly selected MATH problems.

Single Sample and Averaged Time Series for Qwen3-8B - OpenCoder-SFT (n=800)

###### Single Sample Token Confidence

###### Event-Locked Average Token Confidence

60

Final Answer Position

Final Answer Position

35

Mean

TokenConfidence

±1 SE

40

30

25

20

20

0

15

0 200 400 600 800 1000 1200 1400

1000 750 500 250 0 250 500 750 1000

###### Single Sample Log Probabilities

###### Event-Locked Average Log Probabilities

0.3

Final Answer Position

- 0

- 1

- 2

- 3

LogProbability

0.2

0.1

Final Answer Position

Mean

0.0

±1 SE

0 200 400 600 800 1000 1200 1400

1000 750 500 250 0 250 500 750 1000

Token Index

Relative Position from Answer

- Figure 17: Event-Locked Averaging of Token-Confidence for OpenCoder-SFT. A reproduction of Fig. 2, but only using CoTs from 800 randomly selected OpenCoder-SFT problems.

0 1000 2000 3000 4000

0

10

20

30

40

TokenConfidence

Single Sample Token Confidence

Final Answer Position

1000 750 500 250 0 250 500 750 1000

12

14

16

18

20

Event-Locked Average Token Confidence

Final Answer Position

Mean

±1 SE

0 1000 2000 3000 4000

Token Index

- 0

- 1

- 2

- 3

- 4

- 5

LogProbability

Single Sample Log Probabilities

Final Answer Position

1000 750 500 250 0 250 500 750 1000

Relative Position from Answer

0.1

0.2

0.3

0.4

Event-Locked Average Log Probabilities

Final Answer Position

Mean

±1 SE

Single Sample and Averaged Time Series for Qwen3-8B - OpenScience (n=800)

- Figure 18: Event-Locked Averaging of Token-Confidence for OpenScience. A reproduction of Fig. 2, but only using CoTs from 800 randomly selected OpenScience problems.

Token Occurrence Ratios for Qwen3-8B - All Data Sources (n=3200)

- 10 2

OccurrenceRateAfterAnswer

Avg. Before Rate: 1.98e-04 Avg. After Rate: 2.11e-05 Above Diagonal: 5.2% ----------------------------------Only After: 4.2%

Only Before: 58.8% Before and After: 6.2%

Neither: 30.9% ----------------------------------Median Length Below: 3326 Median Length Above: 4455

Token: "hmm"

Equal Rate

0 10 5 10 4 10 3 10 2

0

10 5

10 4

10 3

10 2

Avg. Before Rate: 3.33e-04 Avg. After Rate: 6.75e-05 Above Diagonal: 8.5% ----------------------------------Only After: 0.0%

Only Before: 80.8% Before and After: 19.2%

Neither: 0.0% ----------------------------------Median Length Below: 2866 Median Length Above: 4290

Token: "okay"

Equal Rate

0 10 5 10 4 10 3 10 2

0

10 5

10 4

10 3

10 2

Avg. Before Rate: 1.75e-04 Avg. After Rate: 6.16e-04 Above Diagonal: 68.1% ----------------------------------Only After: 33.9%

Only Before: 13.6% Before and After: 40.2%

Neither: 12.2% ----------------------------------Median Length Below: 4028 Median Length Above: 3303

Token: "another"

Equal Rate

0 10 5 10 4 10 3 10 2

0

10 5

10 4

- 10 3

10 3

10 4

10 5

0

0 10 5 10 4 10 3 10 2

- 10 2

OccurrenceRateAfterAnswer

Avg. Before Rate: 1.82e-03 Avg. After Rate: 2.45e-03 Above Diagonal: 69.6% ----------------------------------Only After: 5.8%

Only Before: 11.6% Before and After: 82.2%

Neither: 0.3% ----------------------------------Median Length Below: 2569 Median Length Above: 3124

Token: "but"

Equal Rate

0 10 5 10 4 10 3 10 2

0

10 5

10 4

10 3

10 2

Avg. Before Rate: 3.27e-03 Avg. After Rate: 3.37e-03 Above Diagonal: 45.7% ----------------------------------Only After: 0.0%

Only Before: 8.7% Before and After: 91.3%

Neither: 0.0% ----------------------------------Median Length Below: 4056 Median Length Above: 2330

Token: "so"

Equal Rate

0 10 5 10 4 10 3 10 2

0

10 5

10 4

10 3

10 2

Avg. Before Rate: 2.77e-04 Avg. After Rate: 1.84e-04 Above Diagonal: 22.3% ----------------------------------Only After: 10.4%

Only Before: 36.6% Before and After: 26.3%

Neither: 26.8% ----------------------------------Median Length Below: 3657 Median Length Above: 3680

Token: "let's"

Equal Rate

0 10 5 10 4 10 3 10 2

Occurrence Rate Before Answer

0

10 5

10 4

- 10 3

###### Token: "wait"

###### Token: "alternatively"

###### Token: "therefore"

Equal Rate

Equal Rate

Equal Rate

10 2

10 2

10 2

OccurrenceRateAfterAnswer

10 3

10 3

10 4

10 4

Avg. Before Rate: 2.47e-04 Avg. After Rate: 4.24e-04 Above Diagonal: 46.2% ----------------------------------Only After: 23.3%

Avg. Before Rate: 1.18e-03 Avg. After Rate: 1.10e-03 Above Diagonal: 42.3% ----------------------------------Only After: 4.9%

Avg. Before Rate: 1.01e-03 Avg. After Rate: 1.72e-03 Above Diagonal: 68.3% ----------------------------------Only After: 17.2%

10 5

10 5

Only Before: 20.3% Before and After: 34.5%

Only Before: 16.7% Before and After: 77.4%

Only Before: 12.9% Before and After: 62.0%

Neither: 21.9% ----------------------------------Median Length Below: 4302 Median Length Above: 3592

Neither: 1.1% ----------------------------------Median Length Below: 3131 Median Length Above: 2903

Neither: 7.9% ----------------------------------Median Length Below: 3030 Median Length Above: 3344

0

0

0 10 5 10 4 10 3 10 2

0 10 5 10 4 10 3 10 2

Occurrence Rate Before Answer

Occurrence Rate Before Answer

- Figure 19: Token Usage Frequency Shift. An extension of the results shown in Fig. 3, highlighting additional “thinking tokens.” While most “thinking tokens” shown here have some bias, often occurring before and after the first occurrence of the final answer as indicated by the Above Diagonal statistic, some tokens, like “so,” are close to an equal rate on average.

Token Occurrence Ratios for Qwen3-8B - AIME (n=800)

###### Token: "hmm"

###### Token: "okay"

###### Token: "another"

Equal Rate

Equal Rate

Equal Rate

10 2

10 2

10 2

OccurrenceRateAfterAnswer

10 3

10 3

10 3

10 4

10 4

10 4

Avg. Before Rate: 1.23e-04 Avg. After Rate: 1.74e-05 Above Diagonal: 7.4% ----------------------------------Only After: 5.6%

Avg. Before Rate: 1.67e-04 Avg. After Rate: 4.53e-04 Above Diagonal: 76.4% ----------------------------------Only After: 27.6%

Avg. Before Rate: 2.00e-04 Avg. After Rate: 6.89e-05 Above Diagonal: 13.5% ----------------------------------Only After: 0.0%

10 5

10 5

10 5

Only Before: 72.8% Before and After: 27.2%

Only Before: 62.4% Before and After: 7.8%

Only Before: 11.5% Before and After: 58.2%

Neither: 0.0% ----------------------------------Median Length Below: 7580 Median Length Above: 7194

Neither: 24.2% ----------------------------------Median Length Below: 7977 Median Length Above: 6123

Neither: 2.6% ----------------------------------Median Length Below: 9661 Median Length Above: 7119

0

0

0

0 10 5 10 4 10 3 10 2

0 10 5 10 4 10 3 10 2

0 10 5 10 4 10 3 10 2

###### Token: "but"

###### Token: "so"

###### Token: "let's"

Equal Rate

Equal Rate

Equal Rate

10 2

10 2

10 2

OccurrenceRateAfterAnswer

10 3

10 3

10 3

10 4

10 4

10 4

Avg. Before Rate: 2.13e-04 Avg. After Rate: 1.25e-04 Above Diagonal: 24.5% ----------------------------------Only After: 9.5%

Avg. Before Rate: 1.55e-03 Avg. After Rate: 2.21e-03 Above Diagonal: 76.5% ----------------------------------Only After: 1.5%

Avg. Before Rate: 2.83e-03 Avg. After Rate: 2.36e-03 Above Diagonal: 32.0% ----------------------------------Only After: 0.0%

10 5

10 5

10 5

Only Before: 43.1% Before and After: 32.5%

Only Before: 1.8% Before and After: 96.8%

Only Before: 2.4% Before and After: 97.6%

Neither: 14.9% ----------------------------------Median Length Below: 7656 Median Length Above: 7930

Neither: 0.0% ----------------------------------Median Length Below: 9330 Median Length Above: 6994

Neither: 0.0% ----------------------------------Median Length Below: 7994 Median Length Above: 6336

0

0

0

0 10 5 10 4 10 3 10 2

0 10 5 10 4 10 3 10 2

0 10 5 10 4 10 3 10 2

###### Token: "wait"

###### Token: "alternatively"

###### Token: "therefore"

Equal Rate

Equal Rate

Equal Rate

10 2

10 2

10 2

OccurrenceRateAfterAnswer

10 3

10 3

10 3

10 4

10 4

10 4

Avg. Before Rate: 4.14e-04 Avg. After Rate: 5.56e-04 Above Diagonal: 58.4% ----------------------------------Only After: 15.5%

Avg. Before Rate: 1.44e-03 Avg. After Rate: 1.39e-03 Above Diagonal: 45.0% ----------------------------------Only After: 1.2%

Avg. Before Rate: 2.12e-03 Avg. After Rate: 3.14e-03 Above Diagonal: 79.0% ----------------------------------Only After: 0.2%

10 5

10 5

10 5

Only Before: 1.8% Before and After: 98.0%

Only Before: 5.6% Before and After: 93.1%

Only Before: 17.9% Before and After: 65.4%

Neither: 0.0% ----------------------------------Median Length Below: 10496 Median Length Above: 7058

Neither: 0.0% ----------------------------------Median Length Below: 8744 Median Length Above: 6534

Neither: 1.2% ----------------------------------Median Length Below: 9367 Median Length Above: 6721

0

0

0

0 10 5 10 4 10 3 10 2

0 10 5 10 4 10 3 10 2

0 10 5 10 4 10 3 10 2

Occurrence Rate Before Answer

Occurrence Rate Before Answer

Occurrence Rate Before Answer

- Figure 20: Token Usage Frequency Shift for AIME (1983–2024. A reproduction of Fig. 19, but only using CoTs from 800 randomly selected AIME (1983–2024) problems.

Token Occurrence Ratios for Qwen3-8B - MATH (n=800)

###### Token: "hmm"

###### Token: "okay"

###### Token: "another"

Equal Rate

Equal Rate

Equal Rate

10 2

10 2

10 2

OccurrenceRateAfterAnswer

10 3

10 3

10 3

10 4

10 4

10 4

Avg. Before Rate: 1.36e-04 Avg. After Rate: 5.86e-04 Above Diagonal: 80.0%

Avg. Before Rate: 3.99e-04 Avg. After Rate: 5.04e-05 Above Diagonal: 8.8% ----------------------------------Only After: 7.8%

Avg. Before Rate: 5.70e-04 Avg. After Rate: 6.52e-05 Above Diagonal: 4.2% ----------------------------------Only After: 0.0%

----------------------------------Only After: 58.2% Only Before: 5.5%

10 5

10 5

10 5

Only Before: 61.8% Before and After: 12.4%

Only Before: 76.9% Before and After: 23.1%

Before and After: 29.5%

Neither: 18.1% ----------------------------------Median Length Below: 2630 Median Length Above: 2696

Neither: 0.0% ----------------------------------Median Length Below: 2648 Median Length Above: 3479

Neither: 6.8% ----------------------------------Median Length Below: 3844 Median Length Above: 2670

0

0

0

0 10 5 10 4 10 3 10 2

0 10 5 10 4 10 3 10 2

0 10 5 10 4 10 3 10 2

###### Token: "but"

###### Token: "so"

###### Token: "let's"

Equal Rate

Equal Rate

Equal Rate

10 2

10 2

10 2

OccurrenceRateAfterAnswer

10 3

10 3

10 3

10 4

10 4

10 4

Avg. Before Rate: 3.93e-03 Avg. After Rate: 3.38e-03 Above Diagonal: 32.9% ----------------------------------Only After: 0.0%

Avg. Before Rate: 2.05e-04 Avg. After Rate: 1.43e-04 Above Diagonal: 27.8% ----------------------------------Only After: 22.0%

Avg. Before Rate: 1.06e-03 Avg. After Rate: 2.00e-03 Above Diagonal: 80.5%

----------------------------------Only After: 19.2% Only Before: 3.1%

10 5

10 5

10 5

Only Before: 1.2% Before and After: 98.8%

Only Before: 24.9% Before and After: 19.1%

Before and After: 76.8%

Neither: 0.0% ----------------------------------Median Length Below: 2808 Median Length Above: 2375

Neither: 34.0% ----------------------------------Median Length Below: 3277 Median Length Above: 2568

Neither: 0.9% ----------------------------------Median Length Below: 2433 Median Length Above: 2721

0

0

0

0 10 5 10 4 10 3 10 2

0 10 5 10 4 10 3 10 2

0 10 5 10 4 10 3 10 2

###### Token: "wait"

###### Token: "alternatively"

###### Token: "therefore"

Equal Rate

Equal Rate

Equal Rate

10 2

10 2

10 2

OccurrenceRateAfterAnswer

10 3

10 3

10 3

10 4

10 4

10 4

Avg. Before Rate: 1.21e-03 Avg. After Rate: 2.20e-03 Above Diagonal: 83.8%

Avg. Before Rate: 2.02e-04 Avg. After Rate: 7.37e-04 Above Diagonal: 80.4%

Avg. Before Rate: 1.05e-03 Avg. After Rate: 1.13e-03 Above Diagonal: 54.8%

----------------------------------Only After: 15.1% Only Before: 1.5%

----------------------------------Only After: 56.8% Only Before: 4.9%

----------------------------------Only After: 14.8% Only Before: 4.8%

10 5

10 5

10 5

Before and After: 80.2%

Before and After: 34.2%

Before and After: 83.0%

Neither: 0.2% ----------------------------------Median Length Below: 2765 Median Length Above: 2592

Neither: 4.1% ----------------------------------Median Length Below: 4205 Median Length Above: 2568

Neither: 0.4% ----------------------------------Median Length Below: 2527 Median Length Above: 2686

0

0

0

0 10 5 10 4 10 3 10 2

0 10 5 10 4 10 3 10 2

0 10 5 10 4 10 3 10 2

Occurrence Rate Before Answer

Occurrence Rate Before Answer

Occurrence Rate Before Answer

- Figure 21: Token Usage Frequency Shift for MATH. A reproduction of Fig. 19, but only using CoTs from 800 randomly selected MATH problems.

Token Occurrence Ratios for Qwen3-8B - OpenCoder-SFT (n=800)

###### Token: "hmm"

###### Token: "okay"

###### Token: "another"

Equal Rate

Equal Rate

Equal Rate

10 2

10 2

10 2

OccurrenceRateAfterAnswer

10 3

10 3

10 3

10 4

10 4

10 4

Avg. Before Rate: 1.44e-04 Avg. After Rate: 5.32e-06 Above Diagonal: 1.8% ----------------------------------Only After: 1.1%

Avg. Before Rate: 3.37e-04 Avg. After Rate: 1.31e-04 Above Diagonal: 15.5% ----------------------------------Only After: 0.0%

Avg. Before Rate: 2.73e-04 Avg. After Rate: 1.15e-03 Above Diagonal: 83.2%

----------------------------------Only After: 32.2% Only Before: 7.1%

10 5

10 5

10 5

Only Before: 75.0% Before and After: 25.0%

Only Before: 50.6% Before and After: 2.4%

Before and After: 55.0%

Neither: 0.0% ----------------------------------Median Length Below: 2156 Median Length Above: 2206

Neither: 45.9% ----------------------------------Median Length Below: 2479 Median Length Above: 4346

Neither: 5.6% ----------------------------------Median Length Below: 1942 Median Length Above: 2284

0

0

0

0 10 5 10 4 10 3 10 2

0 10 5 10 4 10 3 10 2

0 10 5 10 4 10 3 10 2

###### Token: "but"

###### Token: "so"

###### Token: "let's"

Equal Rate

Equal Rate

Equal Rate

10 2

10 2

10 2

OccurrenceRateAfterAnswer

10 3

10 3

10 3

10 4

10 4

10 4

Avg. Before Rate: 2.50e-03 Avg. After Rate: 3.61e-03 Above Diagonal: 73.9% ----------------------------------Only After: 1.8%

Avg. Before Rate: 3.95e-03 Avg. After Rate: 5.26e-03 Above Diagonal: 74.0% ----------------------------------Only After: 0.0%

Avg. Before Rate: 5.74e-04 Avg. After Rate: 4.44e-04 Above Diagonal: 32.1% ----------------------------------Only After: 7.6%

10 5

10 5

10 5

Only Before: 4.5% Before and After: 93.6%

Only Before: 0.2% Before and After: 99.8%

Only Before: 32.5% Before and After: 48.2%

Neither: 0.1% ----------------------------------Median Length Below: 2588 Median Length Above: 2048

Neither: 0.0% ----------------------------------Median Length Below: 2531 Median Length Above: 2010

Neither: 11.6% ----------------------------------Median Length Below: 2229 Median Length Above: 2479

0

0

0

0 10 5 10 4 10 3 10 2

0 10 5 10 4 10 3 10 2

0 10 5 10 4 10 3 10 2

###### Token: "wait"

###### Token: "alternatively"

###### Token: "therefore"

Equal Rate

Equal Rate

Equal Rate

10 2

10 2

10 2

OccurrenceRateAfterAnswer

10 3

10 3

10 3

10 4

10 4

10 4

Avg. Before Rate: 1.33e-04 Avg. After Rate: 1.18e-04 Above Diagonal: 19.2% ----------------------------------Only After: 11.9%

Avg. Before Rate: 1.25e-03 Avg. After Rate: 1.15e-03 Above Diagonal: 42.0% ----------------------------------Only After: 1.2%

Avg. Before Rate: 5.47e-05 Avg. After Rate: 3.66e-04 Above Diagonal: 60.0%

----------------------------------Only After: 47.6% Only Before: 9.8%

10 5

10 5

10 5

Only Before: 28.1% Before and After: 14.2%

Only Before: 10.0% Before and After: 88.8%

Before and After: 14.4%

Neither: 45.8% ----------------------------------Median Length Below: 3316 Median Length Above: 2150

Neither: 0.0% ----------------------------------Median Length Below: 2332 Median Length Above: 1748

Neither: 28.2% ----------------------------------Median Length Below: 4123 Median Length Above: 2288

0

0

0

0 10 5 10 4 10 3 10 2

0 10 5 10 4 10 3 10 2

0 10 5 10 4 10 3 10 2

Occurrence Rate Before Answer

Occurrence Rate Before Answer

Occurrence Rate Before Answer

- Figure 22: Token Usage Frequency Shift for OpenCoder-SFT. A reproduction of Fig. 19, but only using CoTs from 800 randomly selected OpenCoder-SFT problems.

Token Occurrence Ratios for Qwen3-8B - OpenScience (n=800)

###### Token: "hmm"

###### Token: "okay"

###### Token: "another"

Equal Rate

Equal Rate

Equal Rate

10 2

10 2

10 2

OccurrenceRateAfterAnswer

10 3

10 3

10 3

10 4

10 4

10 4

Avg. Before Rate: 1.27e-04 Avg. After Rate: 1.12e-05 Above Diagonal: 3.0% ----------------------------------Only After: 2.1%

Avg. Before Rate: 2.23e-04 Avg. After Rate: 4.99e-06 Above Diagonal: 0.8% ----------------------------------Only After: 0.0%

Avg. Before Rate: 1.23e-04 Avg. After Rate: 2.73e-04 Above Diagonal: 32.6% ----------------------------------Only After: 17.6%

10 5

10 5

10 5

Only Before: 98.8% Before and After: 1.2%

Only Before: 60.5% Before and After: 2.1%

Only Before: 30.4% Before and After: 18.2%

Neither: 0.0% ----------------------------------Median Length Below: 1540 Median Length Above: 4872

Neither: 35.2% ----------------------------------Median Length Below: 1748 Median Length Above: 2829

Neither: 33.8% ----------------------------------Median Length Below: 2831 Median Length Above: 2472

0

0

0

0 10 5 10 4 10 3 10 2

0 10 5 10 4 10 3 10 2

0 10 5 10 4 10 3 10 2

###### Token: "but"

###### Token: "so"

###### Token: "let's"

Equal Rate

Equal Rate

Equal Rate

10 2

10 2

10 2

OccurrenceRateAfterAnswer

10 3

10 3

10 3

10 4

10 4

10 4

Avg. Before Rate: 1.17e-04 Avg. After Rate: 2.36e-05 Above Diagonal: 5.0% ----------------------------------Only After: 2.4%

Avg. Before Rate: 2.17e-03 Avg. After Rate: 1.99e-03 Above Diagonal: 47.6% ----------------------------------Only After: 0.8%

Avg. Before Rate: 2.36e-03 Avg. After Rate: 2.47e-03 Above Diagonal: 43.8% ----------------------------------Only After: 0.0%

10 5

10 5

10 5

Only Before: 37.0% Before and After: 61.9%

Only Before: 45.8% Before and After: 5.4%

Only Before: 30.9% Before and After: 69.1%

Neither: 0.4% ----------------------------------Median Length Below: 1235 Median Length Above: 2069

Neither: 46.5% ----------------------------------Median Length Below: 2048 Median Length Above: 5284

Neither: 0.0% ----------------------------------Median Length Below: 1796 Median Length Above: 1399

0

0

0

0 10 5 10 4 10 3 10 2

0 10 5 10 4 10 3 10 2

0 10 5 10 4 10 3 10 2

###### Token: "wait"

###### Token: "alternatively"

###### Token: "therefore"

Equal Rate

Equal Rate

Equal Rate

10 2

10 2

10 2

OccurrenceRateAfterAnswer

10 3

10 3

10 3

10 4

10 4

10 4

Avg. Before Rate: 2.38e-04 Avg. After Rate: 2.85e-04 Above Diagonal: 27.0% ----------------------------------Only After: 9.1%

Avg. Before Rate: 6.41e-04 Avg. After Rate: 1.16e-03 Above Diagonal: 50.4% ----------------------------------Only After: 5.6%

Avg. Before Rate: 9.96e-04 Avg. After Rate: 7.04e-04 Above Diagonal: 27.6% ----------------------------------Only After: 2.2%

10 5

10 5

10 5

Only Before: 30.2% Before and After: 24.0%

Only Before: 46.4% Before and After: 47.4%

Only Before: 38.6% Before and After: 52.8%

Neither: 36.6% ----------------------------------Median Length Below: 2728 Median Length Above: 2865

Neither: 4.0% ----------------------------------Median Length Below: 1626 Median Length Above: 1777

Neither: 3.0% ----------------------------------Median Length Below: 1282 Median Length Above: 1890

0

0

0

0 10 5 10 4 10 3 10 2

0 10 5 10 4 10 3 10 2

0 10 5 10 4 10 3 10 2

Occurrence Rate Before Answer

Occurrence Rate Before Answer

Occurrence Rate Before Answer

- Figure 23: Token Usage Frequency Shift for OpenScience. A reproduction of Fig. 19, but only using CoTs from 800 randomly selected OpenScience problems.

0 2000 4000 6000 8000 10000 12000 14000

CoT Length

0.00000

0.00005

0.00010

0.00015

0.00020

0.00025

0.00030

0.00035

OccurrenceRate

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Token: "hmm"

Before Answer Rate

After Answer Rate

0 2000 4000 6000 8000 10000 12000 14000

CoT Length

0.0000

0.0001

0.0002

0.0003

0.0004

0.0005

0.0006

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Token: "okay"

0 2000 4000 6000 8000 10000 12000 14000

CoT Length

0.0000

0.0002

0.0004

0.0006

0.0008

0.0010

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Token: "another"

Occurrence Rates vs CoT Length for Qwen3-8B - All Data Sources

- Figure 24: Token Occurrence Rate vs CoT Length. For some tokens, such as these three “thinking tokens,” occurrence rates decrease rapidly as CoT length increases. Interestingly, the side of the answer (either “before” or “after”) with the highest rate is the one that decays the most (the “before” rate for hmm and okay and the “after” rate for another), while the side with the lower rate sees only a slight decrease or increase. For each plot, the lengths are placed into ten bins with percentile-based bin edges. In other words, each bin contains approximately 10% of the samples. Shaded regions indicate the 95% confidence interval.

Event-Locked Average Exit Predictions by Data Source for Qwen3-8B

###### AIME25 (n=30)

###### GPQA (n=448)

1.0

1.0

PredictedProbabilityforExiting

0.8

0.8

0.6

0.6

0.4

0.4

Exit Position

Exit Position

Exit Threshold

Exit Threshold

0.2

0.2

Mean ±1 SE

Mean ±1 SE

| |
|---|

| |
|---|

0.0

0.0

1000 750 500 250 0 250 500 750 1000

1000 750 500 250 0 250 500 750 1000

###### HumanEval (n=164)

###### MATH-500 (n=500)

1.0

1.0

PredictedProbabilityforExiting

0.8

0.8

0.6

0.6

0.4

0.4

Exit Position

Exit Position

Exit Threshold

Exit Threshold

0.2

0.2

Mean ±1 SE

Mean ±1 SE

| |
|---|

| |
|---|

0.0

0.0

1000 750 500 250 0 250 500 750 1000

1000 750 500 250 0 250 500 750 1000

Relative Position from Exit

Relative Position from Exit

- Figure 25: Predicted Probabilities Event-Locked Averaging. The dashed vertical line shows where TERMINATOR terminates the CoT with a sliding window of 10 and an exit threshold of 0.7, as indicated by the horizontal dotted line. We show the average predicted probability stream across all test problems from MATH-500, AIME25, HumanEval, and GPQA. Figs. 26 to 29 show predictions streams from individual, randomly drawn samples from each data source.

| | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|
| | |Predicted Exit Thr|Exit Position eshold| | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |

0 1000 2000 3000 4000 5000 6000 7000

0.0

0.2

0.4

0.6

0.8

1.0

PredictedProbabilityforExiting

AIME25 - Sample 502

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | |Predicted Exit Po<br><br>Exit Threshold|sition| | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

0 2000 4000 6000 8000

0.0

0.2

0.4

0.6

0.8

1.0

AIME25 - Sample 523

0 500 1000 1500 2000 2500 3000

Token Index

0.0

0.2

0.4

0.6

0.8

1.0

PredictedProbabilityforExiting

AIME25 - Sample 505

Predicted Exit Position

Exit Threshold

| | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|
| | |Predicted Exit Thres|Exit Positio hold|n| | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |

0 2000 4000 6000 8000 10000 12000 14000 16000

Token Index

0.0

0.2

0.4

0.6

0.8

1.0

AIME25 - Sample 508

Individual Sample Exit Predictions for Qwen3-8B - AIME25

- Figure 26: Predicted Probabilities for AIME25. TERMINATOR’s predicted probability stream for early-exiting on four randomly chosen samples from AIME25.

Individual Sample Exit Predictions for Qwen3-8B - MATH-500

###### MATH-500 - Sample 398

###### MATH-500 - Sample 107

| | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | |Predicted Exit Thre<br><br>|Exit Positio shold|n| | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |

1.0

1.0

Predicted Exit Position

PredictedProbabilityforExiting

Exit Threshold

0.8

0.8

0.6

0.6

0.4

0.4

0.2

0.2

0.0

0.0

0 500 1000 1500 2000 2500 3000 3500 4000

0 250 500 750 1000 1250 1500 1750

###### MATH-500 - Sample 56

###### MATH-500 - Sample 406

1.0

1.0

Predicted Exit Position

Predicted Exit Position

PredictedProbabilityforExiting

Exit Threshold

Exit Threshold

0.8

0.8

0.6

0.6

0.4

0.4

0.2

0.2

0.0

0.0

0 100 200 300 400 500 600 700

0 250 500 750 1000 1250 1500 1750

Token Index

Token Index

- Figure 27: Predicted Probabilities for MATH-500. TERMINATOR’s predicted probability stream for early-exiting on four randomly chosen samples from MATH-500.

0 200 400 600 800 1000

0.0

0.2

0.4

0.6

0.8

1.0

PredictedProbabilityforExiting

HumanEval - Sample 1036

Predicted Exit Position

Exit Threshold

0 200 400 600 800 1000 1200 1400

0.0

0.2

0.4

0.6

0.8

1.0

HumanEval - Sample 1113

Predicted Exit Position

Exit Threshold

0 200 400 600 800 1000 1200 1400

Token Index

0.0

0.2

0.4

0.6

0.8

1.0

PredictedProbabilityforExiting

HumanEval - Sample 1139

Predicted Exit Position

Exit Threshold

0 250 500 750 1000 1250 1500 1750

Token Index

0.0

0.2

0.4

0.6

0.8

1.0

HumanEval - Sample 1114

Predicted Exit Position

Exit Threshold

Individual Sample Exit Predictions for Qwen3-8B - HumanEval

- Figure 28: Predicted Probabilities for HumanEval. TERMINATOR’s predicted probability stream for early-exiting on four randomly chosen samples from HumanEval.

Individual Sample Exit Predictions for Qwen3-8B - GPQA

###### GPQA - Sample 825

###### GPQA - Sample 793

1.0

1.0

Predicted Exit Position

Predicted Exit Position

PredictedProbabilityforExiting

Exit Threshold

Exit Threshold

0.8

0.8

0.6

0.6

0.4

0.4

0.2

0.2

0.0

0.0

0 250 500 750 1000 1250 1500 1750 2000

0 1000 2000 3000 4000 5000

###### GPQA - Sample 867

###### GPQA - Sample 913

1.0

1.0

Predicted Exit Position

Predicted Exit Position

PredictedProbabilityforExiting

Exit Threshold

Exit Threshold

0.8

0.8

0.6

0.6

0.4

0.4

0.2

0.2

0.0

0.0

0 1000 2000 3000 4000 5000

0 500 1000 1500 2000 2500 3000 3500

Token Index

Token Index

- Figure 29: Predicted Probabilities for GPQA. TERMINATOR’s predicted probability stream for early-exiting on four randomly chosen samples from GPQA.

0 1000 2000 3000 4000 5000 6000

0

5

10

15

20

Count

AIME (n=800)

Mean: 5847.0

Median: 4169.5

0 1000 2000 3000 4000 5000 6000

0

10

20

30

40

50

60

70

80

MATH (n=800)

Mean: 1629.6

Median: 891.5

0 1000 2000 3000 4000 5000 6000

First Occurence of Final Answer

0

10

20

30

40

50

60

Count

OpenCoder-SFT (n=800)

Mean: 1703.1

Median: 1199.5

0 1000 2000 3000 4000 5000 6000

First Occurence of Final Answer

0

10

20

30

40

50

60

OpenScience (n=800)

Mean: 2036.0

Median: 1198.0

Distribution of First Occurence of Final Answer for Qwen3-8B by Data Source

- Figure 30: First Answer Occurrence Histogram. A histogram of the first occurrence of the final answer for each data source used in our training dataset is shown.

Compression Rate Distribution for Qwen3-8B

###### AIME25 (n=90)

###### MATH-500 (n=1500)

200

12

Mean: 0.693

Mean: 0.374

Median: 0.729

Median: 0.349

175

10

150

8

125

Count

100

6

75

4

50

2

25

0

0

0.0 0.2 0.4 0.6 0.8 1.0

0.0 0.2 0.4 0.6 0.8 1.0

###### HumanEval (n=492)

###### GPQA (n=1344)

50

Mean: 0.677

Mean: 0.845

600

Median: 0.699

Median: 0.932

500

40

400

30

Count

300

20

200

10

100

0

0

0.0 0.2 0.4 0.6 0.8 1.0

0.0 0.2 0.4 0.6 0.8 1.0

Compression Rate

Compression Rate

- Figure 31: Compression Ratio Histograms for TERMINATOR. Each histogram shows the frequency of an achieved compression rate when early-exiting with TERMINATOR. Ó indicates that a lower compression rate is better, as it results in more tokens saved with our method. Three CoTs are sampled per data source.

#### MATH-500, sample 77

Early exiting position

True: 4 | Pred: 4

... a ^\{ m \\ cdot n \} $. Applying that here , the right side

becomes : $$ 2 ^ 8 = 2 ^\{ 2 x \} $$ So now both sides have

the same base , which is 2 . If the bases are equal and the

expressions are equal , then their ex ponents must be equal as well

. That means : $$ 8 = 2 x $$ To solve for $ x $, divide

both sides by 2 : $$ x = \\ frac \{ 8 \}\{ 2 \} = 4 $$ Wait

|me|
|---|

, let

check if that makes sense . If $ x = 4 $, then $ 4

^ 4 $ should equal $ 2 ^ 8 $. Let me compute both : $ 4 ^ 4

= 4 \\ times 4 \\ times 4 \\ times 4 = 1 6 \\ times 1 6 = 2

5 6 $ $ 2 ^ 8 = 2 \\ times 2 \\ times 2 \\ times 2 \\ times

2 \\ times 2 \\ times 2 \\ times 2 = 2 5 6 $ Yes , both equal

2 5 6 . So that checks ...

|[Figure 5]| | | | |
|---|---|---|---|---|
| | | | | |

0.0 0.2 0.4 0.6 0.8 1.0

Early Exit Probability

- Figure 32: Predicted Probabilities for MATH-500. TERMINATOR’s predicted probabilities for early-exit on a randomly chosen sample from MATH-500. The beginning and the end are truncated for better visibility.

#### MATH-500, sample 195

Early exiting position

True: 24 | Pred: 24

... compute p ( 4 ). Pl ugging in x = 4 : p ( 4 ) = ( 4

- 1 )( 4 - 2 )( 4 - 3 )( 4 - r ) = ( 3 )( 2

)( 1 )( 4 - r ) = 3 * 2 * 1 *( 4 - r ) = 6

*( 4 - r ) = 2 4 - 6 r . Therefore , p ( 0 ) + p

( 4 ) = 6 r + 2 4 - 6 r = 2 4 . Wait , the r

|terms|
|---|

cancel out ? That 's interesting . So regardless of the value

of r , p ( 0 ) + p ( 4 ) is 2 4 ? But how is that

possible ? How come we don 't need to know the fourth root ? Is

there something wrong here ? Let me check my calculations again . For

p ( 0 ): ( 0 - 1 ) = - 1 ( 0 - 2 ) = - 2

( 0 - 3 ) = - 3 ( 0 - r ) = - r Multiply them : (-

...

|[Figure 6]| | | | |
|---|---|---|---|---|
| | | | | |

0.0 0.2 0.4 0.6 0.8 1.0

Early Exit Probability

- Figure 33: Predicted Probabilities for MATH-500. TERMINATOR’s predicted probabilities for early-exit on a randomly chosen sample from MATH-500. The beginning and the end are truncated for better visibility.

#### AIME-2025, sample 508

Early exiting position

True: 62 | Pred: 50

... - â 4 1 )/ 6 is not the correct y -coordinate . Therefore

, this suggests that my initial calculation is correct , and the

answer is (- 3 - â 4 1 )/ 6 , which is in the form ( a -

â b )/ c if we allow a to be negative , but the problem

specifies a , b , c are positive integers . This is a contradiction

, indicating a possible error in the problem or my solution . However

|it|
|---|

's more likely that

, since the problem is from a competition ,

I made a mistake . Let me revisit the quart ic equation . Earlier ,

after substit uting t = x â 3 , we had the quart ic equation t

^ 4 - 2 t Â³ - 2 5 t Â² + 1 8 t + 1 2 0 =

0 , which fact ored into ( t Â² + t - 1 0 )( t Â² - 3 t

- 1 2 ) = 0 . Let me double -check this factor ization . Multiply

( t Â² + t - 1 0 )( ...

|[Figure 7]| | | | |
|---|---|---|---|---|
| | | | | |

0.0 0.2 0.4 0.6 0.8 1.0

Early Exit Probability

- Figure 34: Predicted Probabilities for AIME25. TERMINATOR’s predicted probabilities for earlyexit on a randomly chosen sample from AIME-2025. The beginning and the end are truncated for better visibility.

###### AIME-2025, sample 517

Early exiting position

True: 82 | Pred: 82

... 's 8 2 . Seems small , but maybe correct ? Alternatively ,

think of the number of possibilities for internal edges and external

edges . Earlier , we found that for each internal edge configuration

( a ,b ,c ,d ), there are some number of external color ings . And

total over all internal configurations is 8 2 . Therefore , 8 2 is

the answer ? But let me check with another approach . Suppose we

model this as a graph and use linear algebra over GF ( 2 ). However

|linear|
|---|

, the constraints are not

equations but rather exact counts .

However , maybe we can find dependencies . Alternatively , think of

the problem as a constraint satisfaction problem where each square

imposes that the sum of its edges is 2 ( mod something ? No , it

's not mod , it 's exact count ). However , since each square has

four edges , two red and two blue , which is equivalent to saying

that the number of red edges is even ? No , two is even , but also

, for example , if you have three red edges ...

|[Figure 8]| | | | |
|---|---|---|---|---|
| | | | | |

0.0 0.2 0.4 0.6 0.8 1.0

Early Exit Probability

- Figure 35: Predicted Probabilities for AIME25. TERMINATOR’s predicted probabilities for earlyexit on a randomly chosen sample from AIME-2025. The beginning and the end are truncated for better visibility.

