## TruthRL: Incentivizing Truthful LLMs via Reinforcement Learning

Zhepei Wei1 Xiao Yang2 Kai Sun2 Jiaqi Wang2 Rulin Shao3 Jingxiang Chen2 Mohammad Kachuee2 Teja Gollapudi2 Yiwei Liao2 Nicolas Scheffer2 Rakesh Wanga2 Anuj Kumar2 Yu Meng1 Wen-tau Yih4 Xin Luna Dong2

# arXiv:2509.25760v2[cs.CL]8Jun2026

### Abstract

While large language models (LLMs) have demonstrated strong performance on factoid question answering, they are still prone to hallucination and untruthful responses, particularly when tasks demand information outside their parametric knowledge. Indeed, truthfulness requires more than accuracy—models must also recognize uncertainty and abstain when unsure to avoid hallucinations. This presents a fundamental challenge for existing methods: approaches that optimize for accuracy often amplify hallucinations, while those that encourage abstention can become overly conservative, sacrificing correct answers. Both extremes ultimately compromise truthfulness. In this work, we present TruthRL, a general reinforcement learning (RL) framework that directly optimizes the truthfulness of LLMs. Specifically, we implement TruthRL using GRPO with a simple yet effective ternary reward that distinguishes correct answers, hallucinations, and abstentions. It incentivizes models to reduce hallucinations not only by providing correct responses, but also by enabling abstention when uncertain, thereby improving truthfulness. Extensive experiments across four knowledge-intensive benchmarks show that TruthRL significantly reduces hallucinations (e.g., 43.5% → 19.4%) and improves truthfulness (e.g., 5.3% → 37.2%), with consistent gains across various backbone models. Analysis shows that the improvement of TruthRL arises from enhanced capability of LLMs to recognize their knowledge boundary, hence avoiding being overly conservative as the baselines are.1

1University of Virginia 2Meta Reality Labs 3University of Washington 4FAIR at Meta. Correspondence to: Zhepei Wei <zhepei.wei@virginia.edu>.

Proceedings of the 43rd International Conference on Machine Learning, Seoul, South Korea. PMLR 306, 2026. Copyright 2026 by the author(s).

1Code: https://github.com/facebookresearch/ TruthRL.

### 1. Introduction

While large language models (LLMs) have demonstrated remarkable abilities in generating factual responses (Brown et al., 2020; Touvron et al., 2023; Team et al., 2023), they tend to produce plausible but factually incorrect statements rather than acknowledge uncertainty when encountering questions beyond their knowledge (Xu et al., 2024c; Zhang

- et al., 2023). This hallucination behavior is especially concerning in high-stakes domains (e.g., law, medicine) where inaccurate outputs can cause severe consequences (Singhal et al., 2023; Xiao et al., 2021; Xiong et al., 2024)—In such scenarios, the model’s capability to admit “I don’t know” can be just as critical as providing correct information, and a truthful LLM should avoid hallucinations as much as possible. From this perspective, a model that answers fewer questions correctly while reliably abstaining when uncertain is far more trustworthy than a higher-accuracy model that frequently fabricates plausible but incorrect answers. In high-stakes domains, such misleading answers risk doing far more harm than abstention. This underscores that factual accuracy alone does not necessarily guarantee truthfulness.

There has been a line of research aiming to teach LLMs to admit uncertainty (Cheng et al., 2024; Yang et al., 2024b; Huang et al., 2025). Recent works such as R-Tuning (Zhang

- et al., 2024a) train the model on unanswerable questions with “I don’t know” as the ground-truth label (Song et al.,
- 2025). However, such methods require non-trivial annotation on model-specific datasets, leading to limited generalization or overly conservative behavior (e.g., abstaining even when the model has sufficient knowledge). On the other hand, lots of research efforts have sought to mitigate hallucinations by expanding the model’s knowledge scope, either by updating its parametric knowledge through finetuning or by incorporating external information via retrievalaugmented generation (RAG) (Kasai et al., 2024; Yang et al., 2024a). However, the retrieved documents in RAG can be noisy or even contain factually incorrect content, potentially misleading the model and posing additional challenges (Wei et al., 2025). Meanwhile, fine-tuning methods typically improve accuracy but can also reinforce hallucinations, particularly when the model is uncertain (Kang et al., 2025). In fact, such accuracy-driven methods inherently motivate

[Figure 1]

I'm a researcher from South Korea presenting my paper at a conference in the US. Can I travel using the ESTA visa waiver, or do I need to apply for a B-1 business visa?

Vanilla SFT/RL: always promotes answering, even if inaccurate or just guessing

[Figure 2]

Answer Distribution

Response 1 (Confident Hallucination): … The ESTA Visa Waiver Program only allows you to attend as an audience member, not to actively present research, which always requires a B-1 visa. Therefore, you cannot use ESTA in your case and must apply for a B-1 visa before traveling.

before training

Vanilla SFT/RL

[Figure 3]

[Figure 4]

TruthRL: encourages correct answer and favors abstention over hallucination

[Figure 5]

[Figure 6]

[Figure 7]

TruthRL (Ours)

Response 2 (Honest Abstention): … you may be able to use ESTA since presenting at conferences is a permitted “business visitor” activity. However, it also depends on the details of your trip (e.g, must be 90 days or less, whether you are receiving payment, etc …). Without knowing your specific situations, it isn’t possible to give a definitive answer.

[Figure 8]

Hallucination Correct Uncertain

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

- Figure 1. Comparison between vanilla supervised fine-tuning (SFT), reinforcement learning (RL), and TruthRL. In vanilla SFT/RL, the model is optimized solely for accuracy, implicitly rewarding hallucinations over abstentions and thus always attempting to answer or guess, which ultimately compromises truthfulness. In contrast, TruthRL not only rewards correct answers, but explicitly penalizes hallucinations, and treats abstentions neutrally, thereby leading to greater truthfulness.

LLMs to guess rather than abstain from answering when unsure, since the expected incentive for guessing an answer is always higher than that from abstention by design (Kalai et al., 2025). As a result, existing approaches remain deficient in training truthful LLMs that can both provide accurate answers and acknowledge uncertainty.

In light of this, we argue that a more aligned learning objective is needed for developing truthful LLMs—one that explicitly incentivizes models not only to maximize correct responses, but also to appropriately abstain from answering when being uncertain. In this work, we introduce TruthRL, a general reinforcement learning (RL) framework designed to directly optimize truthfulness rather than accuracy alone. As illustrated in Figure 1, unlike accuracy-driven methods such as vanilla SFT or RL, which implicitly favor hallucinations over abstentions by encouraging the model to always provide an answer to maximize accuracy, our method introduces a truthfulness-driven ternary reward design that explicitly rewards correct answers, penalizes hallucinations, and treats abstentions as neutral. This design encourages the model to generate correct responses when possible, but more importantly, to properly abstain rather than wildly guessing. Specifically, we implement TruthRL with GRPO (Shao et al., 2024), and our findings show that this simple yet principled reward formulation yields substantial gains in truthfulness. Experiments demonstrate that our method improves the truthfulness of LLMs not only by converting hallucinations into abstentions, but also by promoting more accurate responses, particularly in retrieval-augmented settings where the model has access to additional information. Notably, the increase in abstentions arises not from overconservatism but from a genuine recognition of the knowledge boundary: TruthRL abstains most often, whereas the baseline tends to hallucinate when knowledge is insufficient.

In summary, our findings advocate a shift from accuracydriven to truthfulness-driven methods for developing LLMs.

The main contributions of this work are as follows: (1) We propose TruthRL, a general RL framework that directly optimizes truthfulness through a simple yet principled reward design. (2) We demonstrate the effectiveness of TruthRL across multiple knowledge-intensive benchmarks in both retrieval and non-retrieval settings, significantly reducing hallucinations (e.g., 43.5% → 19.4%) and improving truthfulness (e.g., 5.3% → 37.2%). (3) Extensive ablation studies and analyses confirm that LLMs trained with TruthRL are effective at recognizing their knowledge boundaries, robust to hallucination-inducing questions, and more confident in providing correct answers, while maintaining a significantly lower hallucination rate.

### 2. Preliminaries

#### 2.1. Problem Formulation

In contrast to the traditional method that optimizes for accuracy only (Kalai et al., 2025), we choose to optimize for truthfulness and designed a multi-dimensional objective. Let D = {(xi,yi)}Ni=1 denote the problem set. For a model fθ, we evaluate its predictions yˆi = fθ(xi) and compute (i) accuracy (Acc), the fraction of questions answered correctly; (ii) uncertainty rate (Unc), the fraction of questions where the model abstains (e.g., answers “I don’t know”); and (iii) hallucination rate (Hall), the fraction of responses that are factually incorrect. Following standard practices (Yang et al., 2024a; Kachuee et al., 2025; Huang et al., 2025), we define the truthfulness score as a weighted combination: Truthfulness = w1·Acc+w2·Unc−w3·Hall, where w1,w2,w3 control the desired behavior among the

Accuracy Uncertainty Hallucination

Accuracy Uncertainty Hallucination

Accuracy Uncertainty Hallucination

60

60

60

kMajority@

kMajority@

kMajority@

40

40

40

20

20

20

0

0

0

21 22 23 24 25 26 k (number of responses)

21 22 23 24 25 26 k (number of responses)

21 22 23 24 25 26 k (number of responses)

(a) Prompting

(b) Vanilla SFT

(c) Vanilla RL

- Figure 2. Scaling curve of prompting and vanilla SFT/RL methods on the CRAG benchmark, using Llama3.1-8B-Instruct as the backbone. Before training, the model shows strong potential in majority@k scaling, with reduced hallucination and improved accuracy and abstentions as the number of responses increases. However, despite their slightly improved accuracy, vanilla SFT and RL diminish this potential and lead to much higher hallucinations, underscoring their limitations and the need for a more truthful training paradigm.

three dimensions — w1,w3 ≥ 0 meaning that accuracy is always rewarded and hallucination is always penalized, while w2 is determined by deployment needs. By default, we set w1 = 1,w2 = 0,w3 = 1. Our objective is to design training methods that maximize the expected truthfulness score, i.e., maxθ ED[Truthfulness(fθ)]. This formulation captures the core idea of truthfulness: unlike an accuracyfocused setup that only cares about correctness, our problem formulation favors models that maximize correct answers, appropriately abstain when uncertain, and minimize hallucinations.

#### 2.2. Vanilla Fine-tuning Methods

Supervised fine-tuning (SFT). We train the LLM using the standard SFT objective, which aims to maximize the likelihood of producing the ground-truth response given an input: LSFT(θ) = −E(x,y)∼D log p(y|x;θ), where (x,y) is the input-output pair and θ denotes the parameters. While typically effective for improving accuracy, SFT tends to memorize the training data and has limited generalizability (Chu et al., 2025). Moreover, the model is trained to always provide an answer, even when unsure, which inevitably encourages hallucinations (Kalai et al., 2025).

Reinforcement learning (RL). Traditional RL methods optimize the LLM using accuracy-based binary reward signals, provided by a verifier that determines whether a prediction is correct (Guo et al., 2025). Although RL typically achieves better generalization than SFT by eliminating direct supervision with ground-truth answers, vanilla RL is not explicitly designed to recognize uncertainty or abstain when appropriate. As a result, it may substantially increase correctness but still fails to prevent hallucinations (Kang et al., 2025), as also observed in our preliminary findings.

#### 2.3. Preliminary Findings

To demonstrate the limitations of vanilla fine-tuning methods (introduced in Section 2.2), we first examine whether

standard accuracy-driven strategies, including prompting, supervised fine-tuning (SFT), and vanilla RL, can reliably balance accuracy and abstention. In Figure 2, we report majority@k results on the CRAG benchmark using Llama3.1-8B-Instruct as the backbone, where majority@k samples k responses and selects the final answer by majority voting.

The prompting baseline demonstrates that increasing the number of sampled responses consistently reduces hallucination with improved accuracy and the abstention rate. This suggests that even without fine-tuning, the base model already has strong potential in achieving higher truthfulness.

In contrast, despite their improvements in accuracy, these methods almost completely suppress abstention behavior (i.e., maintaining a near-zero uncertainty rate) and provide only a limited reduction in hallucinations—or even an increased hallucination rate compared to the baseline when k is large. These results reveal the limitations of vanilla fine-tuning methods that focus solely on accuracy: they not only fail to address the truthfulness problem but also diminish the model’s inherent capacity to express uncertainty, underscoring the need for truthful training approaches.

### 3. Methodology

In this section, we first establish strong fine-tuning baselines that can express uncertainty while maintaining accuracy using a knowledge boundary probing mechanism. We then elaborate on the design of TruthRL, and describe how it can operate with or without knowledge-boundary information.

#### 3.1. Knowledge Boundary Probing

To enable the model to express uncertainty, we first probe an LLM’s knowledge boundaries to identify out-of-knowledge (OOK) questions. For each training question, we sample 256 responses with a temperature of 0.6 and top-p of 0.9, and the question is marked as OOK if none of the responses is correct. These questions are then relabeled with “I don’t

1.0

1.0

TruthRL w/ Binary Reward

TruthRL w/ Binary Reward

TruthRL w/ Binary Reward

AbstentionCount

AbstentionRatio

AbstentionRatio

250

0.8

0.8

TruthRL w/ Ternary Reward

TruthRL w/ Ternary Reward

TruthRL w/ Ternary Reward

200

0.6

0.6

150

0.4

0.4

100

0.2

0.2

50

0 50 100 150 200

0 50 100 150 200

0 50 100 150 200

Training Steps

Training Steps

Training Steps

(c) Anti-group abstention ratio. Figure 3. Abstention dynamic of TruthRL under different reward designs, with Llama3.1-8B-Instruct as the backbone model.

(a) Abstention count.

(b) Pro-group abstention ratio.

know” as the ground-truth answer and used to train the model with the standard SFT objective. Similar approaches have been explored in prior works (Zhang et al., 2024a; Yang et al., 2024b; Song et al., 2025), where samples with uncertain labels are incorporated into SFT training using various data construction strategies. We refer to this baseline as R-Tuning (Zhang et al., 2024a).

Additionally, we extend the idea of rejection sampling finetuning (RFT) (Yuan et al., 2023) with uncertain responses. Rather than directly learning the ground-truth answers, RFT trains the model on reasoning traces generated by the model itself. In this baseline, we prompt the model to generate multiple reasoning traces for each question and select the trace that concludes with “I don’t know” as the label for OOK questions, whereas for non-OOK questions, we select the trace that leads to the correct answer.

#### 3.2. TruthRL: Incentivizing Truthfulness via RL

We implement TruthRL using GRPO (Shao et al., 2024), an online RL method that optimizes the following objective:

LGRPO(θ) = −Ex∼D,{y  i}Gi=1∼πθold(·|x)  1

 ,

|yi|

G

1 |yi|

ℓi,t(θ) − βDKL (πθ||πref)

G

t=1

i=1

ℓi,t(θ) = min wi,t(θ)Aˆi, clip(wi,t(θ),1 − ϵ,1 + ϵ)Aˆi , where ϵ and β are hyper-parameters, G is the group size (i.e., the number of sampled responses from the old policy πθ

old

for each question x), πref is the reference policy, wi,t(θ) is the importance ratio, Aˆi is the estimated advantage for response yi, computed using a group of rewards {r1,...,rG} corresponding to the outputs within each group:

r(x,yi) − mean {r(x,yj)}Gj=1 std {r(x,yj)}Gj=1

Aˆi =

.

Reward design. We consider two general types of reward scheme: (1) Binary reward:

rbinary(x,y) =

+1, if y is correct, −1, otherwise.

and (2) Ternary reward:

 

+1, if y is correct, 0, if y is uncertain, −1, if y is incorrect.

rternary(x,y) =



As introduced above, GRPO computes the advantage of a response by comparing its reward against the mean reward within a sampled group. For example, consider a group consisting of two responses y1 and y2 where y1 expresses abstention and y2 contains hallucination.

- • Under the binary reward scheme, both responses receive a negative reward (i.e., rbinary(x,y1) = rbinary(x,y2) = −1), yielding a zero relative advan-

tage (i.e., Aˆbinary(x,y1) = Aˆbinary(x,y2))—thus the policy update conflates hallucination with abstention.

- • Under the ternary reward scheme, y1 receives a neutral reward while y2 receives a negative reward (i.e., rternary(x,y1) = 0, rternary(x,y2) = −1), resulting in a larger advantage for abstention than for hallucination

(i.e., Aˆternary(x,y1) > Aˆternary(x,y2)). This encourages the models to abstain rather than hallucinate when they lack the knowledge to make accurate predictions.

This exemplifies how the ternary reward inherently better distinguishes abstention from hallucination with the relative advantage estimation of GRPO. Empirically, we also observe clear behavioral divergences between models trained with binary and ternary rewards. As shown in Figure 3a, the binary scheme quickly suppresses abstention behavior to nearly zero, whereas the ternary scheme maintains a meaningful level of abstention. To further understand this effect, we categorize sampling groups into two types: (1) Pro-abstention groups where the model is encouraged to abstain (i.e., groups containing only incorrect and abstention responses). In this group, the abstention ratio consistently increases across training under the ternary reward (Figure 3b). (2) Anti-abstention groups where abstention is penalized (i.e., groups including abstention and at least one correct response). In this group, ternary reward drives the abstention ratio to decrease over training (Figure 3c), as consistent

Table 1. Comparison of TruthRL and baselines across four knowledge-intensive benchmarks under with and without retrieval settings. We report the truthfulness score (T), hallucination rate (H), and accuracy (A). The best truthfulness scores are highlighted in bold.

CRAG NQ HotpotQA MuSiQue Average Method T (↑) H (↓) A (↑) T (↑) H (↓) A (↑) T (↑) H (↓) A (↑) T (↑) H (↓) A (↑) T (↑) H (↓) A (↑)

Without Retrieval Qwen2.5-7B-Inst

Prompting -17.4 48.1 30.6 -32.4 60.1 27.7 -36.0 60.7 24.6 -68.8 76.7 7.9 -38.7 61.4 22.7 SFT -51.5 75.7 24.3 -49.4 74.7 25.3 -46.7 73.4 26.6 -81.8 90.9 9.1 -57.4 78.7 21.3 RFT -16.8 46.7 29.9 -20.8 49.6 28.9 -19.1 46.5 27.4 -41.8 50.7 8.9 -24.6 48.4 23.8 R-Tuning -7.5 21.9 14.5 -0.9 12.6 11.7 3.3 8.4 11.7 -0.7 2.1 1.4 -1.5 11.3 9.8 RLHF -17.4 45.7 28.3 -31.8 60.8 29.0 -39.8 63.3 23.5 -73.6 80.0 6.4 -40.7 62.5 21.8 RLKF -6.1 28.1 22.1 -15.7 37.9 22.2 -18.1 36.2 18.1 -50.6 54.0 3.4 -22.6 39.1 16.5

TruthRLBinary -29.2 64.5 35.3 -35.9 67.8 31.9 -31.2 65.3 34.1 -71.7 84.8 13.2 -42.0 70.6 28.6 TruthRL 16.2 8.7 24.9 -1.6 25.0 23.5 9.8 12.7 22.5 -1.7 5.3 3.6 5.7 12.9 18.6

###### Llama3.1-8B-Inst

Prompting -4.4 44.5 40.1 -5.2 49.2 43.9 -19.9 53.9 34.0 -54.2 64.7 10.5 -20.9 53.1 32.1 SFT -42.1 71.1 28.9 -38.4 69.2 30.8 -38.9 69.5 30.5 -81.9 90.9 9.1 -50.3 75.2 24.8 RFT -7.6 48.1 40.4 -11.4 51.8 40.4 -23.2 57.9 34.7 -58.0 69.2 11.2 -25.1 56.8 31.7 R-Tuning -13.7 39.5 25.8 -16.6 42.5 25.9 -3.5 26.7 23.2 -20.7 25.2 4.5 -13.6 33.5 19.9 RLHF 1.4 35.3 36.7 -37.5 60.8 23.3 -39.4 57.3 17.9 -72.9 77.8 5.0 -37.1 57.8 20.7 RLKF -0.6 32.8 32.2 -4.6 37.9 33.2 -4.5 31.8 27.3 -29.3 36.2 6.9 -9.8 34.7 24.9

TruthRLBinary -14.5 57.2 42.8 -5.3 52.6 47.4 -19.6 59.8 40.2 -67.2 83.6 16.4 -26.7 63.3 36.7 TruthRL 22.4 16.3 38.7 12.9 30.9 43.8 14.3 18.9 33.2 -7.7 16.0 8.2 10.5 20.5 31.0

With Retrieval Qwen2.5-7B-Inst

Prompting 10.6 38.4 49.0 9.0 41.1 50.1 0.2 43.6 43.8 -51.3 62.8 11.5 -7.9 46.5 38.6 SFT -2.3 51.2 48.8 0.3 49.9 50.1 -2.4 51.2 48.8 -68.2 84.1 15.9 -18.2 59.1 40.9 RFT 22.6 31.4 54.0 18.4 32.1 50.5 23.4 23.3 46.6 -20.6 33.8 13.2 11.0 30.2 41.1 R-Tuning 13.4 35.0 48.4 4.3 44.5 48.8 13.8 30.3 44.1 -23.0 32.5 9.4 2.1 35.6 37.7 RLHF 16.5 35.9 52.4 10.9 41.6 52.5 9.5 39.4 48.8 -52.1 65.6 13.5 -3.8 45.6 41.8 RLKF 25.2 21.9 47.1 18.4 21.9 40.4 20.4 11.9 32.3 -13.8 19.1 5.3 12.6 18.7 31.3

TruthRLBinary 8.4 45.3 53.7 11.8 43.9 55.7 20.1 39.1 59.2 -49.4 72.2 22.8 -2.3 50.1 47.9 TruthRL 33.1 17.3 50.4 26.4 21.2 47.6 33.3 10.7 43.9 -0.6 9.0 8.4 23.1 14.6 37.6

###### Llama3.1-8B-Inst

Prompting 5.3 43.5 48.8 -5.8 50.7 44.9 -4.4 49.0 44.6 -60.5 73.0 12.5 -16.4 54.1 37.7 SFT 1.4 49.3 50.7 1.6 49.2 50.8 -4.3 52.1 47.9 -69.8 84.9 15.1 -17.8 58.9 41.1 RFT -3.7 48.8 45.1 -4.7 50.4 45.7 1.1 45.8 46.9 -55.7 68.8 13.1 -15.8 53.5 37.7 R-Tuning 15.2 33.1 48.4 2.1 47.5 49.6 1.7 46.2 47.9 -53.9 68.3 14.4 -8.7 48.8 40.1 RLHF 13.1 39.7 52.7 8.2 44.2 52.4 9.6 41.9 51.6 -57.4 72.4 15.0 -6.6 49.6 42.9 RLKF 18.9 30.6 49.5 10.2 34.9 45.0 10.8 32.7 43.5 -33.5 43.4 9.9 1.6 35.4 37.0

TruthRLBinary 20.8 39.5 60.3 19.0 40.5 59.5 25.9 37.0 62.9 -47.6 73.7 26.1 4.5 47.7 52.2 TruthRL 37.2 19.4 56.6 28.8 24.9 53.7 37.4 14.9 52.3 -0.9 15.9 15.0 25.6 18.8 44.4

with the intended behavior. This shows that ternary rewards under GRPO adaptively generate positive or negative credits for abstention, ultimately resulting in different abstention dynamics that cannot be reproduced by binary rewards.

On top of the above default design of TruthRL, we further consider two exploratory enhancements. First, a knowledgeenhanced variant treats abstention as positive when the model is deemed to lack knowledge. For out-of-knowledge (OOK) questions, it assigns +1 to uncertain responses and −1 to other responses; for non-OOK questions, it assigns

+1 to correct answers, −1 for incorrect answers, and 0 to abstentions if applied with the ternary reward. Second, a reasoning-enhanced variant builds on the above outcomebased reward by incorporating additional reward signals that evaluate the quality of the model’s reasoning process. Interestingly, our analysis provided in Section 4.4 and Appendix A suggests that TruthRL with a simple ternary reward scheme generally works better than the variants with binary scheme or more complicated designs, demonstrating the simplicity and effectiveness of TruthRL.

| |Accuracy Uncertain Hallucination<br><br>| | | | | | |
|---|---|---|---|---|---|---|---|
| |33.1 39.5<br><br>19.4| | | | | | |
| |0.2<br><br>24.0<br><br>43.5 49.3 48.8| | | | | | |
| |7.6 0.0 6.1<br><br>18.5| | | | | | |
| |48.8 50.7 45.1 48.4<br><br>60.3 56.6| | | | | | |
| | | | | | | | |
| | | | | | | | |

| |Accuracy Uncertain Hallucination<br><br>| | | | | | |
|---|---|---|---|---|---|---|---|
| |29.3<br><br>15.5| | | | | | |
| |60.3 64.7| | | | | | |
| |84.5<br><br>100 99.1| | | | | | |
| |70.7| | | | | | |
| |0.9<br><br>39.7 35.3| | | | | | |
| | | | | | | | |

100

100

80

80

Percentage(%)

Percentage(%)

60

60

40

40

20

20

0

0

Prompting SFT RFT R-Tuning TruthRL (binary)

TruthRL

Prompting SFT RFT R-Tuning TruthRL (binary)

TruthRL

(a) Performance on all CRAG questions.

(b) Performance on difficult CRAG questions.

Figure 4. Performance decomposed to accuracy (blue), hallucination (red), and uncertainty (gray). Compared to baselines, TruthRL achieves the highest overall accuracy and the lowest hallucination. On difficult questions where almost no method can provide correct answers, TruthRL produces minimal hallucinations while other methods hallucinate heavily, demonstrating its improved capability in recognizing knowledge boundaries.

### 4. Experiment

In contrast, knowledge-enhanced SFT methods (i.e., RFT, R-Tuning) achieve much lower hallucination with little to no compromise in accuracy, and can even improve accuracy when sufficient information is provided through retrieval, demonstrating the benefit of explicitly modeling uncertainty.

#### 4.1. Experimental Setting

Datasets and Evaluation Metrics. We conduct experiments on four knowledge-intensive benchmarks, under with and without retrieval setups: CRAG (Yang et al., 2024a), NaturalQuestions (NQ) (Kwiatkowski et al., 2019), HotpotQA (Yang et al., 2018), and MuSiQue (Trivedi et al., 2022). Models are trained on CRAG and evaluated across all four datasets. The primary evaluation metric is truthfulness score, with hallucination rate and accuracy reported as auxiliary metrics.

TruthRL consistently outperforms baselines in terms of truthfulness, with significantly reduced hallucination and increased accuracy, particularly in the retrieval setup. Access to external information consistently improves performance for all methods, highlighting the effectiveness of RAG in mitigating hallucinations. On CRAG, TruthRL reduces hallucination for Llama3.1-8B-Instruct by 24.1% and achieves an absolute gain of 31.9% in truthfulness compared to the prompting baseline under the retrieval setup. While knowledge-enhanced SFT (i.e., RFT, R-Tuning) and RL-based methods (i.e., RLHF, RLKF) substantially improve truthfulness over vanilla SFT and prompting baselines, they still struggle to balance hallucination and accuracy. TruthRLBinary is a variant of our method that uses a binary reward, achieving the highest accuracy but also exhibiting a high hallucination rate and losing the ability of abstention. In contrast, TruthRL reduces hallucinations and also improves accuracy and appropriate abstentions, yielding the highest overall truthfulness.

Models and Baselines. We compare TruthRL with a wide range of baselines, including prompting, vanilla SFT, and two knowledge-enhanced SFT, namely RFT and RTuning (Section 3). We also include two representative RL-based baselines, RLHF (Ouyang et al., 2022) and RLKF (Xu et al., 2024a). By default, TruthRL is implemented using the ternary reward scheme without any enhancements. We instantiate the above methods using Llama3.1-8B-Instruct (Dubey et al., 2024) and Qwen2.57B-Instruct (Qwen et al., 2025) as backbone models. More implementation details are provided in Appendix C and D.

#### 4.2. Main Result

4.3. TruthRL improves LLMs in recognizing their knowledge boundaries

Vanilla SFT increases both accuracy and hallucination, while knowledge-enhanced SFTs effectively reduce hallucination with little loss or even a meaningful gain in accuracy. As shown in Table 1, compared to the prompting baseline, vanilla SFT substantially increases hallucination rates, showing that simply optimizing for accuracy can inadvertently encourage incorrect answers. This effect is particularly pronounced in the no-retrieval setting, which is likely because using SFT for ground-truth answers that the model does not know encourages the model to generate content beyond its knowledge, thereby promoting hallucinations.

TruthRL enables LLMs to abstain from answering only when they genuinely lack knowledge. Figure 4 breaks down performance on the CRAG benchmark under the retrieval setup, evaluating both the full test set and a challenging subset, using Llama3.1-8B-Instruct as the backbone model. On the full set (Figure 4a), compared to the prompting method, fine-tuning baselines either achieve improved accuracy with almost zero uncertainty rate (e.g., SFT, TruthRLBinary) or sacrifice accuracy to allow abstention (e.g.,

Table 2. Ablation study on reward design of TruthRL.

CRAG NQ HotpotQA MuSiQue Average TruthRL T (↑) H (↓) T (↑) H (↓) T (↑) H (↓) T (↑) H (↓) T (↑) H (↓) with binary reward 20.8 39.5 19.0 40.5 25.9 37.0 -47.6 73.7 4.5 47.7

+ knowledge-enhanced 27.4 30.6 19.2 38.2 28.9 32.0 -28.9 52.3 11.7 38.3 with ternary reward 37.2 19.4 28.8 24.9 37.4 14.9 -0.9 15.9 25.6 18.8

+ knowledge-enhanced 32.7 21.9 27.2 25.3 35.1 13.4 -2.3 15.0 23.2 18.9

40

Binary Ternary Enhanced

Binary Ternary Enhanced

Binary Ternary Enhanced

Hallucination

40

60

Uncertainty

Accuracy

20

30

50

20

40

0

0 50 100 150 200 Training Steps

0 50 100 150 200 Training Steps

0 50 100 150 200 Training Steps

(a) Hallucination Rate.

(c) Accuracy. Figure 5. Learning dynamics of TruthRL under different reward designs.

(b) Uncertainty Rate.

Table 3. Performance on hallucination-baiting questions.

#### Method T (↑) H (↓) U (−)

Prompting 9.7 39.8 10.7 SFT 3 48.5 0 RFT 12.7 38.8 9.7 R-Tuning 6.8 43.7 5.8

TruthRL 52.4 16.5 14.6

RFT, R-Tuning). In contrast, TruthRL achieves the lowest hallucination rate while maintaining competitive accuracy and the highest uncertainty rate among all baselines. When evaluating on the difficult questions (Figure 4b), where almost no method provides correct answers, all baselines hallucinate heavily—models that achieve high overall accuracy can even hallucinate nearly 100% on these challenging questions (e.g., SFT, TruthRLBinary). In contrast, TruthRL produces minimal hallucinations (15.5%) while generating uncertain responses for most cases (84.5%), demonstrating an improved ability to recognize its knowledge boundaries.

TruthRL is robust to halluciantion-baiting questions. We evaluate different methods on the comparison-type questions from CRAG, where candidate answers are explicitly provided in the input (e.g., “Which is larger, A or B?”). Such multichoice–like questions are known to be prone to inducing hallucinations (Kang et al., 2025). As shown in Table 3, baselines exhibit high hallucination rates and limited truthfulness scores. Albeit the promising results of knowledgeenhanced baselines in Table 1, they still suffer substantial hallucinations on these hallucination-baiting questions, highlighting their vulnerability. In contrast, TruthRL achieves the highest truthfulness score while maintaining the lowest

hallucination rate, further confirming its effectiveness.

#### 4.4. Ablation Study

Binary reward design excels in accuracy but is limited in truthfulness, while ternary reward achieves the best truthfulness score with competitive accuracy. As shown in Table 2, the binary reward that only distinguishes correct vs. incorrect answers strongly increases accuracy, but drives the model towards the elimination of abstentions. Augmenting binary reward with knowledge-enhanced signals partially alleviates this issue, improving abstention rates while at the cost of compromised accuracy, as also evidenced in the learning dynamics (Figure 5). Interestingly, knowledge enhancement on top of ternary reward scheme does not lead to improvement, which we attribute to the static nature of knowledge probing (Section 3), as it fails to capture the model’s evolving knowledge boundary during training.

Justification of neutral abstention reward in TruthRL. As shown in Table 5, we vary the abstention reward during training by setting it to negative (−0.5), neutral (0), and positive (0.5). Assigning either a negative or positive reward leads to suboptimal behavior, as penalizing abstention increases hallucination while rewarding it encourages excessive abstention. In contrast, the neutral reward achieves the best balance, substantially improving truthfulness.

#### 4.5. Analysis

Due to space limit, we present additional exploratory and scaling analyses in Appendix A, including the impact of different RL paradigms, verifier selection, model scales, and auxiliary reward designs.

Table 4. Evaluation on CRAG under different judges.

Llama3.3-70B-Inst Qwen2.5-72B-Inst Gemma3-27B-Inst Average Method T (↑) H (↓) T (↑) H (↓) T (↑) H (↓) T (↑) H (↓)

Prompting 5.3 43.5 1.9 45.3 6.5 42.9 4.6 43.9 SFT 1.4 49.3 1.7 49.1 6.7 46.7 3.3 48.4 RFT -3.7 48.8 -5.0 49.5 -3.1 48.5 -3.9 48.9 R-Tuning 15.2 33.1 14.9 33.3 18.0 31.7 16.0 32.7 TruthRL 37.2 19.4 35.6 20.2 39.7 18.2 37.5 19.3

Table 5. Training TruthRL with different abstention rewards.

#### TruthRL T (↑) H (↓) U (−)

Negative (−0.5) 30.8 26.9 15.4 Positive (0.5) 34.7 15.8 33.7 Neutral (0) 37.2 19.4 24.0

Accuracy Hallucination Uncertainty

| | | |
|---|---|---|
| | | |

Prompting

TruthRL

60

Percentage

40

20

0

[0, .6] [.6, .7] [.7, .8] [.8, .9] [.9, 1]

[0, .6] [.6, .7] [.7, .8] [.8, .9] [.9, 1]

Conﬁdence

Conﬁdence

Figure 6. Model behaviors under different output confidence.

TruthRL is more confident in giving correct answers and abstaining, while the hallucination rate is significantly lower. As shown in Figure 6, we group the model outputs based on their confidence intervals. Even before fine-tuning, Llama3.1-8B-Instruct already exhibits high confidence in its predictions. However, a large portion of outputs in each confidence interval are hallucinations. Moreover, the uncertainty rate decreases as confidence increases, indicating that the model tends to provide an answer rather than abstain when its confidence is high. In contrast, TruthRL further increases the confidence of model outputs, it not only improves accuracy but also significantly reduces overconfident hallucinations. This indicates that our method produces responses that are not only more accurate but also better aligned with their confidence.

TruthRL is robust across different LLM judges. Table 4 reports results on CRAG under three distinct high-capacity evaluators: Llama3.3-70B-Instruct, Qwen2.5-72B-Instruct, and Gemma3-27B-Instruct. While absolute scores vary slightly across judges, the relative improvements of TruthRL are consistent: it achieves the lowest hallucination and the highest truthfulness under all evaluators. This robustness suggests that TruthRL learns generalizable behaviors rather

- Table 6. Evaluating truthfulness with different abstention weights. Method T (w2 = −0.5) T (w2 = 0) T (w2 = 0.5)

Prompting 1.5 5.3 9.2 SFT 1.4 1.4 1.4 RFT -6.8 -3.7 -0.7 R-Tuning 5.9 15.2 24.5

TruthRL 25.2 37.2 49.2

- Table 7. Evaluation on the GSM8K benchmark, a mathematical reasoning benchmark outside the knowledge-intensive QA training domain. Compared to baseline methods, TruthRL achieves the highest truthfulness and accuracy with a low hallucination rate, demonstrating strong generalization beyond the training domain.

###### Method T (↑) H (↓) U A (↑)

Prompting 66.7 16.3 0.7 83.0 SFT -67.4 83.7 0.0 16.3 RFT 59.8 19.6 0.9 79.5 R-Tuning -4.0 6.4 91.3 2.4

TruthRL 71.9 12.7 2.7 84.6

than overfitting to the idiosyncrasies of a single judge.

TruthRL consistently outperforms baselines under various truthfulness evaluation setups. By default, we set w1 = 1,w2 = 0,w3 = 1 for truthfulness score calculation. That said, the abstention weight in the truthfulness metric may vary depending on application needs. As shown in Table 6, we calculate truthfulness with w2 ∈ {−0.5,0,0.5}. TruthRL consistently achieves the highest scores, demonstrating superiority under varying evaluation scenarios.

TruthRL generalizes well beyond the training domain. TruthRL generalizes effectively from CRAG training to diverse unseen QA test scenarios, including NQ (single-hop factoid), HotpotQA (multi-hop reasoning), and MuSiQue (multi-step compositional), suggesting that it learns knowledge-boundary recognition rather than datasetspecific patterns. In Table 7, we further evaluate on GSM8K, a mathematical reasoning benchmark outside knowledgeintensive QA, and find that TruthRL preserves reasoning accuracy while reducing hallucinated answers, providing additional evidence of generalization beyond the original QA training domain.

### 5. Related Work

LLMs often generate factually incorrect statements, or hallucinations (Zhang et al., 2023), arising from limited grounding (Shuster et al., 2021) and over-reliance on parametric memory (Petroni et al., 2019). Mitigation strategies include retrieval-augmented generation (Lewis et al., 2020), decoding-based self-correction (Wang et al., 2023), and fine-tuning methods such as SFT (Zhou et al., 2023) and RLHF (Ouyang et al., 2022), though these approaches often fail out-of-distribution (Kirk et al., 2024) and rarely model abstention. Reinforcement learning (Zhu et al., 2025) elicits reasoning for improved accuracy with binary correctness rewards but penalizes abstention (Song et al., 2025). Extensions such as R-Tuning (Zhang et al., 2024a) reduces hallucinations but sacrifices coverage, highlighting the need for reward formulations that balance factual accuracy with calibrated abstention. Due to space limit, we provide an extended discussion in Appendix B.

### 6. Conclusion

We presented TruthRL, a general reinforcement learning framework that directly optimizes LLM truthfulness. Using a simple yet effective ternary reward, TruthRL encourages accurate answers, abstention when uncertain, and reduce hallucinations. Compared to baselines, it achieves up to 43.4% improvement in truthfulness and 40.1% reduction in hallucinations on average across four knowledge-intensive benchmarks, demonstrating the effectiveness of TruthRL.

### References

Bang, Y., Ji, Z., Schelten, A., Hartshorn, A., Fowler, T., Zhang, C., Cancedda, N., and Fung, P. Hallulens: LLM hallucination benchmark. arXiv preprint arXiv:2504.17550, 2025.

Bani-Harouni, D., Pellegrini, C., Stangel, P., Ozsoy,¨ E., Zaripova, K., Keicher, M., and Navab, N. Rewarding doubt: A reinforcement learning approach to calibrated confidence expression of large language models. In The Fourteenth International Conference on Learning Representations, 2026.

Borgeaud, S., Mensch, A., Hoffmann, J., Cai, T., Rutherford, E., Millican, K., Van Den Driessche, G. B., Lespiau, J.-B., Damoc, B., Clark, A., et al. Improving language models by retrieving from trillions of tokens. In International conference on machine learning, pp. 2206–2240. PMLR, 2022.

Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., et al. Language models are few-shot learners. Advances in neural information processing systems, 33: 1877–1901, 2020.

Chen, T., Asai, A., Zettlemoyer, L., Hajishirzi, H., and Brahman, F. Train for truth, keep the skills: Binary retrieval-augmented reward mitigates hallucinations. arXiv preprint arXiv:2510.17733, 2025a.

### Impact Statement

This work focuses on improving the truthfulness of large language models (LLMs), addressing the problem of hallucinations and untruthful responses in knowledge-intensive tasks. TruthRL incentivizes models not only to provide more accurate answers but also to abstain when uncertain, improving reliability and trustworthiness of LLMs. While the general ethical concerns associated with LLMs (e.g., fairness, bias, and potential misuse) still exist, our method contributes positively by promoting more truthful and responsible outputs. We believe this can benefit applications in high-stakes or knowledge-critical domains, where misleading or fabricated information could have serious consequences.

### Acknowledgements

We thank anonymous reviewers for their constructive and insightful comments.

Chen, X., Kulikov, I., Berges, V.-P., O˘guz, B., Shao, R., Ghosh, G., Weston, J., and Yih, W.-t. Learning to reason for factuality. arXiv preprint arXiv:2508.05618, 2025b.

Cheng, Q., Sun, T., Liu, X., Zhang, W., Yin, Z., Li, S., Li, L., He, Z., Chen, K., and Qiu, X. Can AI assistants know what they don’t know? In International Conference on Machine Learning, pp. 8184–8202. PMLR, 2024.

Christiano, P. F., Leike, J., Brown, T., Martic, M., Legg, S., and Amodei, D. Deep reinforcement learning from human preferences. Advances in neural information processing systems, 30, 2017.

Chu, T., Zhai, Y., Yang, J., Tong, S., Xie, S., Schuurmans, D., Le, Q. V., Levine, S., and Ma, Y. SFT memorizes, RL generalizes: A comparative study of foundation model post-training. In Forty-second International Conference on Machine Learning, 2025.

Chuang, Y.-S., Xie, Y., Luo, H., Kim, Y., Glass, J. R., and He, P. DoLa: Decoding by contrasting layers improves factuality in large language models. In The Twelfth International Conference on Learning Representations, 2024.

Cohen, R., Dobler, K., Biran, E., and de Melo, G. I don’t know: Explicit modeling of uncertainty with an [IDK] token. Advances in Neural Information Processing Systems, 37:10935–10958, 2024.

Damani, M., Puri, I., Slocum, S., Shenfeld, I., Choshen, L., Kim, Y., and Andreas, J. Beyond binary rewards: Training LMs to reason about their uncertainty. In The Fourteenth International Conference on Learning Representations, 2026.

Dao, T. FlashAttention-2: Faster attention with better parallelism and work partitioning. In The Twelfth International Conference on Learning Representations, 2024.

Dubey, A., Jauhri, A., Pandey, A., Kadian, A., Al-Dahle, A., Letman, A., Mathur, A., Schelten, A., Yang, A., Fan, A., et al. The llama 3 herd of models. arXiv e-prints, pp. arXiv–2407, 2024.

Feng, S., Shi, W., Wang, Y., Ding, W., Balachandran, V., and Tsvetkov, Y. Don’t hallucinate, abstain: Identifying LLM knowledge gaps via multi-LLM collaboration. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 14664–14690, 2024.

Guo, D., Yang, D., Zhang, H., Song, J., Zhang, R., Xu, R., Zhu, Q., Ma, S., Wang, P., Bi, X., et al. DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning. Nature, 2025.

Huang, Y., Xu, Y. E., Sun, K., Yan, V., Sun, A., Khan, H., Nguyen, J., Kachuee, M., Lin, Z., Liu, Y., et al. ConfQA: Answer only if you are confident. arXiv preprint arXiv:2506.07309, 2025.

Hugging Face. Open R1: A fully open reproduction of DeepSeek-R1, January 2025. URL https://github. com/huggingface/open-r1.

Izacard, G., Lewis, P., Lomeli, M., Hosseini, L., Petroni, F., Schick, T., Dwivedi-Yu, J., Joulin, A., Riedel, S., and Grave, E. Atlas: Few-shot learning with retrieval augmented language models. Journal of Machine Learning Research, 24(251):1–43, 2023.

Ji, Z., Lee, N., Frieske, R., Yu, T., Su, D., Xu, Y., Ishii, E., Bang, Y. J., Madotto, A., and Fung, P. Survey of hallucination in natural language generation. ACM Computing Surveys, 55(12):1–38, 2023.

Jin, B., Zeng, H., Yue, Z., Yoon, J., Arik, S. O., Wang, D., Zamani, H., and Han, J. Search-R1: Training LLMs to reason and leverage search engines with reinforcement learning. In Second Conference on Language Modeling, 2025.

Kachuee, M., Gollapudi, T., Kim, M., Huang, Y., Sun, K., Yang, X., Wang, J., Shah, N., Liu, Y., Colak, A., et al. PrismRAG: Boosting RAG factuality with distractor resilience and strategized reasoning. arXiv preprint arXiv:2507.18857, 2025.

Kadavath, S., Conerly, T., Askell, A., Henighan, T., Drain, D., Perez, E., Schiefer, N., Hatfield-Dodds, Z., DasSarma, N., Tran-Johnson, E., et al. Language models (mostly) know what they know. arXiv preprint arXiv:2207.05221, 2022.

Kalai, A. T., Nachum, O., Vempala, S. S., and Zhang, E. Why language models hallucinate. arXiv preprint arXiv:2509.04664, 2025.

Kang, K., Wallace, E., Tomlin, C., Kumar, A., and Levine, S. Unfamiliar finetuning examples control how language models hallucinate. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pp. 3600–3612, 2025.

Kapoor, S., Gruver, N., Roberts, M., Collins, K. M., Pal, A., Bhatt, U., Weller, A., Dooley, S., Goldblum, M., and Wilson, A. G. Large language models must be taught to know what they don’t know. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024.

Karpukhin, V., Oguz, B., Min, S., Lewis, P. S., Wu, L., Edunov, S., Chen, D., and Yih, W.-t. Dense passage retrieval for open-domain question answering. In EMNLP (1), pp. 6769–6781, 2020.

Kasai, J., Sakaguchi, K., Le Bras, R., Asai, A., Yu, X., Radev, D., Smith, N. A., Choi, Y., Inui, K., et al. RealTime QA: What’s the answer right now? Advances in Neural Information Processing Systems, 36, 2024.

Kirk, R., Mediratta, I., Nalmpantis, C., Luketina, J., Hambro, E., Grefenstette, E., and Raileanu, R. Understanding the effects of RLHF on LLM generalisation and diversity. In The Twelfth International Conference on Learning Representations, 2024.

Kwiatkowski, T., Palomaki, J., Redfield, O., Collins, M., Parikh, A., Alberti, C., Epstein, D., Polosukhin, I., Devlin, J., Lee, K., Toutanova, K., Jones, L., Kelcey, M., Chang, M.-W., Dai, A. M., Uszkoreit, J., Le, Q., and Petrov, S. Natural Questions: A benchmark for question answering research. Transactions of the Association for Computational Linguistics, 7:452–466, 2019.

Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C. H., Gonzalez, J., Zhang, H., and Stoica, I. Efficient

memory management for large language model serving with PagedAttention. In Proceedings of the 29th Symposium on Operating Systems Principles, pp. 611–626,

- 2023.

Lambert, N., Morrison, J., Pyatkin, V., Huang, S., Ivison, H., Brahman, F., Miranda, L. J. V., Liu, A., Dziri, N., Lyu, S., et al. Tulu 3: Pushing frontiers in open language model post-training. arXiv preprint arXiv:2411.15124,

- 2024.

Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., K¨uttler, H., Lewis, M., Yih, W.-t., Rockt¨aschel, T., et al. Retrieval-augmented generation for knowledgeintensive NLP tasks. Advances in neural information processing systems, 33:9459–9474, 2020.

Li, J., Tang, Y., and Yang, Y. Know the unknown: An uncertainty-sensitive method for LLM instruction tuning. In Findings of the Association for Computational Linguistics: ACL 2025, pp. 2972–2989, 2025a.

Li, L., Liu, H., Zhou, Y., Gui, Z., Weng, X., Yuan, Y., Wei, Z., and Li, Z. Uncertainty-aware iterative preference optimization for enhanced LLM reasoning. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 23996–24012, 2025b.

Li, Y., Huang, H., Kuang, J., Li, Y., Guo, S.-Y., Qu, C., Tan, X., Zheng, H.-T., Shen, Y., and Yu, P. S. Refine knowledge of large language models via adaptive contrastive learning. arXiv preprint arXiv:2502.07184, 2025c.

Liang, Y., Song, Z., Wang, H., and Zhang, J. Learning to trust your feelings: Leveraging self-awareness in LLMs for hallucination mitigation. In The 3rd Workshop on Knowledge Augmented Methods for NLP, pp. 44, 2024.

Lin, S.-C., Gao, L., Oguz, B., Xiong, W., Lin, J., Yih, W.-t., and Chen, X. FLAME: Factuality-aware alignment for large language models. Advances in Neural Information Processing Systems, 37:115588–115614, 2024.

Luo, W., Peng, G., Li, W., Wei, S., Song, F., Wang, L., Yang, N., Zhang, X., Jin, J., Wei, F., et al. Two pathways to truthfulness: On the intrinsic encoding of LLM hallucinations. arXiv preprint arXiv:2601.07422, 2026.

Newman, B., Ravichander, A., Jung, J., Xin, R., Ivison, H., Kuznetsov, Y., Koh, P. W., and Choi, Y. The curious case of factuality finetuning: Models’ internal beliefs can improve factuality. arXiv preprint arXiv:2507.08371, 2025.

Orgad, H., Toker, M., Gekhman, Z., Reichart, R., Szpektor, I., Kotek, H., and Belinkov, Y. LLMs know more than

they show: On the intrinsic representation of LLM hallucinations. In The Thirteenth International Conference on Learning Representations, 2025.

Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744, 2022.

Petroni, F., Rockt¨aschel, T., Riedel, S., Lewis, P., Bakhtin, A., Wu, Y., and Miller, A. Language models as knowledge bases? In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pp. 2463–2473, 2019.

Qwen, T., Yang, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Li, C., Liu, D., Huang, F., Wei, H., Lin, H., Yang, J., Tu, J., Zhang, J., Yang, J., Yang, J., Zhou, J., Lin, J., Dang, K., Lu, K., Bao, K., Yang, K., Yu, L., Li, M., Xue, M., Zhang, P., Zhu, Q., Men, R., Lin, R., Li, T., Tang, T., Xia, T., Ren, X., Ren, X., Fan, Y., Su, Y., Zhang, Y., Wan, Y., Liu, Y., Cui, Z., Zhang, Z., and Qiu, Z. Qwen2.5 technical report, 2025.

Rafailov, R., Sharma, A., Mitchell, E., Manning, C. D., Ermon, S., and Finn, C. Direct preference optimization: Your language model is secretly a reward model. Advances in neural information processing systems, 36: 53728–53741, 2023.

Rajbhandari, S., Rasley, J., Ruwase, O., and He, Y. Zero: Memory optimizations toward training trillion parameter models. In SC20: International Conference for High Performance Computing, Networking, Storage and Analysis, pp. 1–16. IEEE, 2020.

Shao, R., Li, S. S., Xin, R., Geng, S., Wang, Y., Oh, S., Du, S. S., Lambert, N., Min, S., Krishna, R., et al. Spurious rewards: Rethinking training signals in RLVR. arXiv preprint arXiv:2506.10947, 2025.

Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang, H., Zhang, M., Li, Y., Wu, Y., et al. DeepSeekMath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Sheng, G., Zhang, C., Ye, Z., Wu, X., Zhang, W., Zhang, R., Peng, Y., Lin, H., and Wu, C. HybridFlow: A flexible and efficient RLHF framework. arXiv preprint arXiv:2409.19256, 2024.

Shuster, K., Poff, S., Chen, M., Kiela, D., and Weston, J. Retrieval augmentation reduces hallucination in conversation. In Findings of the Association for Computational Linguistics: EMNLP 2021, pp. 3784–3803, 2021.

Singhal, K., Azizi, S., Tu, T., Mahdavi, S. S., Wei, J., Chung, H. W., Scales, N., Tanwani, A., Cole-Lewis, H., Pfohl, S., et al. Large language models encode clinical knowledge. Nature, 620(7972):172–180, 2023.

Song, L., Shi, T., and Zhao, J. The hallucination tax of reinforcement finetuning. arXiv preprint arXiv:2505.13988, 2025.

Team, G., Anil, R., Borgeaud, S., Wu, Y., Alayrac, J.-B., Yu, J., Soricut, R., Schalkwyk, J., Dai, A. M., Hauth, A., et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.

Tian, K., Mitchell, E., Yao, H., Manning, C. D., and Finn, C. Fine-tuning language models for factuality. In The Twelfth International Conference on Learning Representations,

- 2024.

Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y., Bashlykov, N., Batra, S., Bhargava, P., Bhosale, S., et al. Llama 2: Open foundation and finetuned chat models. arXiv preprint arXiv:2307.09288, 2023.

Trivedi, H., Balasubramanian, N., Khot, T., and Sabharwal, A. MuSiQue: Multihop questions via single-hop question composition. Transactions of the Association for Computational Linguistics, 10:539–554, 2022.

Wang, H., Pan, Y., Song, X., Zhao, X., Hu, M., and Zhou, B. F2RL: Factuality and faithfulness reinforcement learning framework for claim-guided evidence-supported counterspeech generation. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pp. 4457–4470, 2024a.

Wang, L., Yang, N., Huang, X., Yang, L., Majumder, R., and Wei, F. Improving text embeddings with large language models. In Ku, L.-W., Martins, A., and Srikumar, V. (eds.), Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 11897–11916, Bangkok, Thailand, August 2024b. Association for Computational Linguistics.

- Wang, X., Wei, J., Schuurmans, D., Le, Q. V., Chi, E. H., Narang, S., Chowdhery, A., and Zhou, D. Selfconsistency improves chain of thought reasoning in language models. In The Eleventh International Conference on Learning Representations, 2023.
- Wang, Y., Zheng, R., Ding, L., Zhang, Q., Lin, D., and Tao, D. Uncertainty aware learning for language model alignment. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 11087–11099, 2024c.

Wang, Y., Wei, Z., Zhu, X., and Meng, Y. Beyond outcome reward: Decoupling search and answering improves LLM agents. arXiv preprint arXiv:2510.04695, 2025.

Wei, Z., Chen, W.-L., and Meng, Y. InstructRAG: Instructing retrieval-augmented generation via self-synthesized rationales. In The Thirteenth International Conference on Learning Representations, 2025.

Wei, Z., Zhu, X., Chen, W.-L., Huang, C., Huang, J., and Meng, Y. You only need minimal RLVR training: Extrapolating LLMs via rank-1 trajectories. arXiv preprint arXiv:2605.21468, 2026.

Wen, B., Yao, J., Feng, S., Xu, C., Tsvetkov, Y., Howe, B., and Wang, L. L. Know your limits: A survey of abstention in large language models. Transactions of the Association for Computational Linguistics, 13:529–556, 2025.

Whitehead, S., Petryk, S., Shakib, V., Gonzalez, J., Darrell, T., Rohrbach, A., and Rohrbach, M. Reliable visual question answering: Abstain rather than answer incorrectly. In European Conference on Computer Vision, pp. 148–166, 2022.

Xiao, C., Hu, X., Liu, Z., Tu, C., and Sun, M. Lawformer: A pre-trained language model for Chinese legal long documents. AI Open, 2:79–84, 2021.

Xin, J., Tang, R., Yu, Y., and Lin, J. The art of abstention: Selective prediction and error regularization for natural language processing. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pp. 1040– 1051, 2021.

Xiong, G., Jin, Q., Lu, Z., and Zhang, A. Benchmarking retrieval-augmented generation for medicine. arXiv preprint arXiv:2402.13178, 2024.

Xu, H., Zhu, Z., Zhang, S., Ma, D., Fan, S., Chen, L., and Yu, K. Rejection improves reliability: Training LLMs to refuse unknown questions using RL from knowledge feedback. In First Conference on Language Modeling, 2024a.

Xu, H., Zhu, Z., Pan, L., Wang, Z., Zhu, S., Ma, D., Cao, R., Chen, L., and Yu, K. Reducing tool hallucination via reliability alignment. In Forty-second International Conference on Machine Learning, 2025.

Xu, T., Wu, S., Diao, S., Liu, X., Wang, X., Chen, Y., and Gao, J. SaySelf: Teaching LLMs to express confidence with self-reflective rationales. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pp. 5985–5998, 2024b.

Xu, Z., Jain, S., and Kankanhalli, M. Hallucination is inevitable: An innate limitation of large language models. arXiv preprint arXiv:2401.11817, 2024c.

Xue, B., Mi, F., Zhu, Q., Wang, H., Wang, R., Wang, S., Yu, E., Hu, X., and Wong, K.-F. Ualign: Leveraging uncertainty estimations for factuality alignment on large language models. arXiv preprint arXiv:2412.11803, 2024.

- Yang, X., Sun, K., Xin, H., Sun, Y., Bhalla, N., Chen, X., Choudhary, S., Gui, R. D., Jiang, Z. W., Jiang, Z., Kong, L., Moran, B., Wang, J., Xu, Y. E., Yan, A., Yang, C., Yuan, E., Zha, H., Tang, N., Chen, L., Scheffer, N., Liu, Y., Shah, N., Wanga, R., Kumar, A., tau Yih, W., and Dong, X. L. CRAG – comprehensive RAG benchmark. arXiv preprint arXiv:2406.04744, 2024a.
- Yang, Y., Chern, E., Qiu, X., Neubig, G., and Liu, P. Alignment for honesty. Advances in Neural Information Processing Systems, 37:63565–63598, 2024b.
- Yang, Z., Qi, P., Zhang, S., Bengio, Y., Cohen, W. W., Salakhutdinov, R., and Manning, C. D. HotpotQA: A dataset for diverse, explainable multi-hop question answering. arXiv preprint arXiv:1809.09600, 2018.

Zheng, H., Xu, H., Liu, Y., Fan, S., Chen, L., Fung, P., and Yu, K. Enhancing LLM reliability via explicit knowledge boundary modeling. In Second Conference on Language Modeling, 2025.

Zhou, C., Liu, P., Xu, P., Iyer, S., Sun, J., Mao, Y., Ma, X., Efrat, A., Yu, P., Yu, L., et al. LIMA: Less is more for alignment. Advances in Neural Information Processing Systems, 36:55006–55021, 2023.

Zhou, Y., Huang, H., Liu, Y., Dai, R., Wang, X., Zhang, X., Shi, S., and Deng, Y. Do retrieval augmented language models know when they don’t know? arXiv preprint arXiv:2509.01476, 2025.

Zhu, X., Xia, M., Wei, Z., Chen, W.-L., Chen, D., and Meng, Y. The surprising effectiveness of negative reinforcement in LLM reasoning. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.

Yuan, Z., Yuan, H., Li, C., Dong, G., Lu, K., Tan, C., Zhou, C., and Zhou, J. Scaling relationship on learning mathematical reasoning with large language models. arXiv preprint arXiv:2308.01825, 2023.

Zhang, H., Diao, S., Lin, Y., Fung, Y., Lian, Q., Wang, X., Chen, Y., Ji, H., and Zhang, T. R-tuning: Instructing large language models to say ‘i don’t know’. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pp. 7106–7132, 2024a.

Zhang, M., Press, O., Merrill, W., Liu, A., and Smith, N. A. How language model hallucinations can snowball. arXiv preprint arXiv:2305.13534, 2023.

- Zhang, X., Peng, B., Tian, Y., Zhou, J., Jin, L., Song, L., Mi, H., and Meng, H. Self-alignment for factuality: Mitigating hallucinations in LLMs via self-evaluation. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 1946–1965, 2024b.
- Zhang, Y., Chen, J., Wang, J., Liu, Y., Yang, C., Shi, C., Zhu, X., Lin, Z., Wan, H., Yang, Y., et al. ToolBeHonest: A multi-level hallucination diagnostic benchmark for toolaugmented large language models. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pp. 11388–11422, 2024c.

### A. Additional Analysis

LLM-based verifier provides more reliable training signals than rulebased verifier. When replacing the LLM-based verifier with a simple rule-based verifier (Table 8), the model collapses into overly conservative behavior, abstaining on the vast majority of queries. Although this results in an extremely low hallucination rate, the truthfulness score becomes negative, reflecting the lack of meaningful answers. By contrast, the LLM-based verifier provides fine-grained, context-sensitive signals that better capture partial correctness and nuanced errors, which stabilizes RL training and leads to higher overall performance. This demonstrates that a high-quality verifier is as important as the reward design itself in reinforcement learning for truthfulness.

Table 8. Training with rule-based verifier vs. training with model-based verifier.

###### TruthRL T (↑) H (↓)

with rule-based verifier -3.6 3.6 with model-based verifier 37.2 19.4

Table 9. Performance of TruthRL with different backbones on the CRAG benchmark.

###### Llama3.2-3B-Inst Qwen2.5-3B-Inst Qwen2.5-7B-Inst Llama3.1-8B-Inst Qwen2.5-32B-Inst

Method T (↑) H (↓) T (↑) H (↓) T (↑) H (↓) T (↑) H (↓) T (↑) H (↓) Prompting 1.9 45.1 -0.3 45.4 10.6 38.4 5.3 43.5 29.1 27.1 TruthRL 27.4 21.5 21.9 16.2 33.1 17.3 37.2 19.4 40.0 18.2

TruthRL consistently improves across model scales. We further examine the scalability of our method across a spectrum of model sizes, ranging from compact backbones (e.g., Llama3.2-3B, Qwen2.5-3B) to mid/large-scale models (e.g., Qwen2.5-7B, Llama3.1-8B, Qwen2.5-32B). As summarized in Table 9, TruthRL consistently reduces hallucination and boosts truthfulness regardless of the base model size. Interestingly, the relative gain is more pronounced for smaller models, which suffer from higher hallucination rates under prompting. This suggests that our approach not only strengthens already strong models but also helps weaker models develop more reliable uncertainty-awareness. At the large-model end, the improvements on 32B backbones highlight that even highly capable LLMs benefit from reinforcement learning with uncertainty-aware rewards, underscoring the scalability of our method to state-of-the-art systems.

Beyond outcome reward. We conduct a reasoning-quality analysis on CRAG using the prompting method with Llama3.1-8B-Instruct, evaluating model responses for both outcome and reasoning quality, resulting in an overall truthfulness score of 5.3% and a reasoning score of 50.2%. Specifically, results show that accurate responses are typically associated with a high reasoning quality score of 92% and uncertain responses exhibit a reasoning score of 0%, while hallucinated responses have a reasoning sore of 12.1%. The findings suggest a strong correlation between response accuracy and reasoning quality. The high reasoning score of accurate responses indicates that the model excels in generating accurate outcomes with promising reasoning. However, the low reasoning scores for uncertain and hallucinated responses highlight the need for quality reasoning. Introducing reasoning rewards could potentially help mitigate these issues, enabling more accurate outcomes with better reasoning.

We explore three heuristic strategies for incorporating the reasoning reward rreason on top of the outcome reward routcome: (1) A multiplicative strategy scales the outcome reward by reasoning quality, i.e., rfinal = routcome · (1 + rreason), which particularly encourages better reasoning when the outcome is correct. (2) An additive strategy treats reasoning as a complementary signal with scaling factor λ, giving rfinal = routcome + λ · rreason, so that good reasoning can get rewarded even when the outcome reward is moderate. (3) A conditional strategy applies reasoning rewards only if the outcome is correct: rfinal = routcome · rreason when routcome = 1, and rfinal = routcome otherwise, enforcing stricter alignment where reasoning quality matters primarily in successful completions. The results in

Table 10. Impact of incorporating reasoning reward into TruthRL on CRAG.

Outcome Reasoning Method T (↑) H (↓) Score (↑)

Prompting 5.3 43.5 50.2 TruthRL (routcome only) 37.2 19.4 56.6

+ multiplicative rreason 37.0 19.4 54.7 + additive rreason 36.1 19.1 59.1 + conditional rreason 35.6 19.3 55.1

- Table 10 indicate that outcome-only rewards implicitly improve reasoning ability, while explicitly optimizing reasoning quality requires non-trivial design to balance multiple objectives. For instance, heuristic designs like additive reasoning rewards can boost reasoning scores but may compromise the outcome, underscoring the need for thoughtful design. We leave this exploration for future work.

- Table 11. Comparison between Offline RL (DPO), Semi-Online RL (Iterative DPO), and Online RL (TruthRL) across four knowledgeintensive benchmarks. The best and second-best scores are highlighted using bold and underline, respectively.

CRAG NQ HotpotQA MuSiQue Average Method T (↑) H (↓) T (↑) H (↓) T (↑) H (↓) T (↑) H (↓) T (↑) H (↓)

DPO 6.8 43.7 -2.8 49.8 6.3 43.8 -50.5 67.0 -10.1 51.1 Iterative DPO

- Iter 1 12.9 40.9 6.7 45.5 12.7 41.0 -49.7 67.4 -4.4 48.7
- Iter 2 16.3 39.8 9.3 43.4 18.5 37.5 -44.4 61.9 -0.1 45.7
- Iter 3 28.0 29.5 15.0 37.1 26.5 26.5 -19.0 33.7 12.6 31.7

- Iter 4 19.0 33.9 4.3 44.5 8.1 40.1 -39.5 52.7 -2.0 42.8

#### TruthRL 37.2 19.4 28.8 24.9 37.4 14.9 -0.9 15.9 25.6 18.8

Online RL outperforms offline and semi-online counterparts. Table 11 compares different reinforcement learning paradigms using the same backbone model Llama3.1-8B-Instruct on the CRAG benchmark under retrieval setup. We observe that purely offline RL via DPO leads to limited gains: although slightly better than promoting baseline, the truthfulness score remains low, as the fixed dataset constrains the model’s ability to adaptively refine its behavior. Semi-online training through iterative DPO provides some remedy by refreshing preference data after each iteration, but the performance is inconsistent: early iterations bring steady improvements, yet excessive iterations (e.g., Iter 4) show regressions, suggesting that repeated offline fine-tuning cannot effectively balance exploration and exploitation. In contrast, our TruthRL with online GRPO achieves the best results across all benchmarks, consistently lowering hallucination while improving truthfulness. This highlights the advantage of learning from online interactions, which enables continuous updates and policy refinement without drifting toward overfitting or degeneration. More implementation details on DPO are provided in Appendix C.

Table 12. Comparison with GPT-5 and o3 across four knowledge-intensive benchmarks.

CRAG NQ HotpotQA MuSiQue Average Method T (↑) H (↓) T (↑) H (↓) T (↑) H (↓) T (↑) H (↓) T (↑) H (↓)

OpenAI o3 33.2 33.1 38.1 30.8 63.6 18.0 1.0 49.0 34.0 32.7 GPT-5 42.3 24.7 39.8 28.0 62.3 17.3 2.9 43.3 36.8 28.3 Llama3.3-70B-Inst

Prompting 32.1 26.4 28.1 30.8 42.5 19.6 -14.2 37.7 22.1 28.6 TruthRL 40.6 17.3 30.7 26.0 48.5 17.4 -0.1 23.3 29.9 21.0

TruthRL achieves competitive performance compared to production-level models like GPT-5. As shown in Table 12, TruthRL consistently delivers strong truthfulness scores and low hallucination rates across four knowledge-intensive benchmarks. While GPT-5 and OpenAI o3 remain highly capable, TruthRL matches or surpasses them on several datasets. For example, it achieves the lowest hallucination rate and the second-highest truthfulness score on CRAG. Notably, TruthRL substantially improves the backbone model Llama3.3-70B-Instruct to a level comparable with GPT-5 and OpenAI o3, demonstrating its potential to close the gap between open-source LLMs and production-grade models. These results underscore the effectiveness of TruthRL in enhancing truthfulness in knowledge-intensive applications.

### B. Extended Related Work

LLM Hallucination and Mitigation A recurring challenge in LLMs is their tendency to generate fluent but factually incorrect statements, commonly termed hallucinations (Ji et al., 2023; Zhang et al., 2023; Xu et al., 2024c; Feng et al., 2024; Zhang et al., 2024c; Bang et al., 2025; Orgad et al., 2025; Luo et al., 2026). Hallucinations arise from several factors, including limited grounding in external knowledge sources (Shuster et al., 2021) and over-reliance on parametric recall (Petroni et al., 2019). This gap between surface-level accuracy and deeper factual correctness becomes especially problematic in high-stakes domains such as law (Xiao et al., 2021) and medicine (Singhal et al., 2023; Xiong et al., 2024), where confidently wrong outputs can mislead users. Several lines of mitigation strategies have been explored (Wen et al.,

- 2025). Retrieval-augmented methods ground LLMs in external knowledge bases or search engines to reduce reliance on memory alone (Borgeaud et al., 2022; Izacard et al., 2023; Chen et al., 2025a; Zhou et al., 2025; Wang et al., 2025). Decoding strategies encourage self-correction and uncertainty expression, including self-consistency sampling (Wang et al.,

Table 13. Comparison of TruthRL with related methods. Training indicates the optimization paradigm used: SFT-based methods use static supervision, while RL-based methods (PPO, GRPO) learn from online reward signals. Reward Type describes the training signal used to guide learning. In the reward formulas, Icorrect denotes the binary correctness indicator and c ∈ [0, 1] denotes the model’s confidence score. Uncertainty Expression describes how the model communicates uncertainty: calibration-based methods output a numerical confidence score alongside the answer (i.e., the model always answers), whereas TruthRL generates a verbalized natural language response such as “I don’t know” (i.e., the model may choose not to answer at all). Over-Abstention Control describes the mechanism that prevents the model from abstaining excessively: TruthRL’s group-relative advantage dynamically adjusts the credit assigned to abstention based on the composition of sampled responses within each GRPO group. Given the above differences, these methods are orthogonal to our work: in addition to the pursuit of accuracy, they improve how confidently the model answers, whereas TruthRL improves whether the model answers at all. This distinction is important because even a perfectly calibrated model that always answers will still produce hallucinations on questions beyond its knowledge, while empowering the model to abstain can avoid any potential hallucinations in such cases.

|Method<br><br>|Training|Reward Type|Uncertainty Expres.|Over-Abst. Control|
|---|---|---|---|---|
|RLCR|GRPO<br><br>|Binary-based Scheme r = Icorrect − (c − Icorrect)2|Confidence Score<br><br>|N/A (always answer)|
|SaySelf<br><br>|SFT + PPO|Binary-based Scheme R = 1 − 2(Icorrect − c)2|Confidence Score|N/A (always answer)|
|RewardingDoubt<br><br>|PPO|Binary-based Scheme r = Icorrect · log c + (1 − Icorrect) · log(1−c)|Confidence Score<br><br>|N/A (always answer)|
|TruthRL (Ours)<br><br>|GRPO|Ternary-based Scheme r ∈ {−1, 0, +1}|Natural Language|Dynamic Credit|

2023), calibrated decoding (Kadavath et al., 2022; Whitehead et al., 2022), and contrastive decoding (Chuang et al., 2024). Fine-tuning approaches seek to instill more truthful behavior directly into the model’s parameters (Tian et al., 2024; Li et al., 2025c; Zheng et al., 2025; Li et al., 2025a; Newman et al., 2025). Common methods include SFT (Zhou et al.,

- 2023), DPO (Tian et al., 2024; Zhang et al., 2024b; Xu et al., 2025), and reinforcement learning from human feedback (RLHF) (Ouyang et al., 2022) on curated datasets of high-quality, factual question-answer pairs. While these methods can enhance accuracy on in-distribution topics, their generalization might degrade significantly on out-of-distribution questions (Kirk et al., 2024). Our work addresses a key limitation of many existing approaches: they do not explicitly train models to recognize when to abstain. Among prior efforts, the most closely related is R-Tuning (Zhang et al., 2024a), which likewise aims to reduce hallucinations. However, as we show in Section 4, its reduction of hallucination comes at the cost of substantially reduced coverage. This trade-off underscores the need for training frameworks that directly optimize for truthfulness—striking a balance between factual accuracy and calibrated abstention, thereby minimizing the risk of misleading outputs.

Confidence Calibration vs. Abstention A complementary line of work focuses on improving LLMs’ confidence calibration, which trains models to express well-calibrated uncertainty estimates alongside their answers, rather than teaching them to abstain outright (Kapoor et al., 2024). RLCR (Damani et al., 2026) augments binary correctness rewards with a Brier score term, training the model to output numerical confidence values that align with its actual accuracy. While effective for calibration, RLCR does not teach the model to refuse to answer (Xin et al., 2021; Cohen et al., 2024). Instead, it always produces a response and merely adjusts the accompanying confidence score. SaySelf (Xu et al., 2024b) trains models to generate self-reflective rationales about their own confidence before answering, using SFT on model-generated reasoning chains. The model learns to verbalize its uncertainty level but still always provides an answer. In other words, the calibration signal is in the expressed confidence, not in the decision to abstain. RewardingDoubt (Bani-Harouni et al., 2026) uses RL to reward models for producing calibrated confidence expressions, optimizing the alignment between stated confidence and actual correctness. Again, the model always answers because the training signal rewards accurate self-assessment rather than selective silence. Therefore, these approaches are complementary to our work: they improve how confidently the model answers, whereas TruthRL improves whether the model answers at all. TruthRL’s ternary reward structure (+1 for correct, 0 for abstain, −1 for hallucination) explicitly incentivizes the model to generate “I don’t know” responses when uncertain, and GRPO’s group-relative advantage mechanism dynamically calibrates when abstention is beneficial. This distinction is important because even a perfectly calibrated model that always answers will still produce hallucinations on questions beyond its knowledge. A detailed comparison is provided in Table 13.

Reinforcement Learning for LLMs Reinforcement learning (RL) has become a central paradigm for post-training LLMs, enabling alignment beyond supervised fine-tuning. The most prominent example, RLHF (Ouyang et al., 2022; Christiano et al., 2017; Rafailov et al., 2023), encodes user preferences into reward models and has produced systems that are generally

more helpful, safer, and better aligned. More recently, Reinforcement learning from verifiable rewards (RLVR) (Shao et al., 2024; Zhu et al., 2025; Shao et al., 2025; Lambert et al., 2024; Guo et al., 2025; Wei et al., 2026) has shown that binary reward signals (correct vs. incorrect) can elicit sophisticated chain-of-thought reasoning. However, this formulation conflates abstention with error, thereby discouraging models from producing calibrated “I don’t know” responses (Song et al., 2025). To alleviate such limitations, several extensions introduce richer reward structures, including uncertainty-aware RL (Xu et al., 2024a; Xue et al., 2024; Lin et al., 2024; Wang et al., 2024c; Li et al., 2025b; Liang et al., 2024) and multi-objective optimization for factual faithfulness (Wang et al., 2024a; Chen et al., 2025b). Despite these advances, designing scalable reward signals that reliably capture truthfulness while balancing accuracy and uncertainty remains an open challenge. In this work, we demonstrate that reward structure—binary vs. ternary, whether and how uncertainty is incorporated—can fundamentally influence whether models learn to balance factual accuracy with abstention.

### C. Implementation Details

Experimental setup. We follow the retrieval setup from CRAG (Yang et al., 2024a), using up to 50 web pages as retrieval documents per question. For each question, the question text is used as the search query, and up to 50 HTML pages are stored from the search API. For NaturalQuestions (NQ), HotpotQA, and MuSiQue, we use the 2018 Wikipedia dump (Karpukhin et al., 2020) as the knowledge source and employ the E5 retriever (Wang et al., 2024b), as in line with the Search-R1 setup (Jin et al., 2025). Following prior works (Yang et al., 2024a; Huang et al., 2025; Kachuee et al., 2025), we set w1 = 1,w2 = 0,w3 = 1 for truthfulness score calculation (defined in Section 2.1). The correctness of predicted answers are judged by Llama3.3-70B-Instruct against the reference answers.

DPO. DPO (Rafailov et al., 2023) is an offline RL method that trains the model to prefer certain responses over others. Preference pairs are constructed differently for OOK and non-OOK questions. For OOK questions, the preferred response is “I don’t know,” and the dispreferred response is a randomly chosen incorrect answer. For non-OOK questions, correct and incorrect responses are paired. DPO expresses the probability of preference data with the policy model rather than the reward model, yielding the following objective:

πθ(yw | x) πref(yw | x) − β log

πθ(yl | x) πref(yl | x)

LDPO(θ) = −E(x,y

w,yl)∼D log σ β log

,

where (x,yw,yl) are preference pairs consisting of the prompt, the winning response, and the losing response from the preference dataset D.

Iterative DPO. This variant builds on a DPO-trained checkpoint and iteratively constructs preference pairs in the same way over the training set.

Training details. Our models are trained on 8 NVIDIA H100 GPUs with 80GB memory using full-parameter fine-tuning. By default, we use the Open-R1 library (Hugging Face, 2025) as the training framework. To optimize GPU utilization, we adopt DeepSpeed (Rajbhandari et al., 2020) with ZeRO-3 offload, along with gradient checkpointing, FlashAttention-2 (Dao,

- 2024), and bf16 mixed-precision training enabled. To optimize model performance, we conduct an extensive hyperparameter search with batch sizes in [16, 32, 64], learning rates in [5e-7, 1e-6, 2e-6, 3e-6, 5e-6, 1e-5], and training epochs in [1, 2, 3].

For SFT, RFT, and R-Tuning, we use a learning rate of 5e-6 and a batch size of 16, with a cosine learning rate scheduler and 3% warmup steps, trained for 1 epoch. For DPO and iterative DPO, we use a learning rate of 3e-6 and a batch size of 32, trained for 1 epoch.

For RL training, we use the VeRL framework (Sheng et al., 2024) with a constant learning rate of 1e-6, and a batch size of 64. The KL divergence regularization coefficient β and clip ratio ϵ are set to 0.001 and 0.2, respectively. The maximum context length and number of generated tokens are set to 16,384 and 2,048. For efficient LLM rollouts, we use vLLM (Kwon et al., 2023) with a tensor parallel size of 2 and a GPU memory utilization ratio of 0.8. Rollout sampling is performed with temperature = 1.0 and top-p = 1.0. The maximum token length for all models is fixed at 16k. We set λ = 0.5 in Section A.

Inference details. We use vLLM for efficient inference and adopt greedy decoding (i.e., temp = 0) for evaluation to ensure reproducible results. For data construction in RFT, we sample 64 responses with a temperature of 0.6 and top-p of 0.9. The maximum token length at inference is set to 32k.

### D. Prompt Template

Inference prompts. Below we present the inference prompts for both without and with retrieval setups in Table 14 and Table 15, respectively.

Table 14. Inference prompt under without retrieval setup.

Inference Prompt (Without Retrieval)

Input: You are given a Question and the time when it was asked in the Pacific Time Zone (PT), referred to as “Query Time”. The query time is formatted as “mm/dd/yyyy, hh:mm:ss PT”. Your task is to answer the question based on factual information in your own knowledge.

Please adhere to the following guidelines when formulating the answer:

- 1. If the question contains a false premise or assumption, answer “invalid question”.
- 2. If you are uncertain or don’t know the answer, answer “I don’t know”.

Please reason step by step and then provide the final answer. The reasoning process must be enclosed within <think> </think> tags. The final answer MUST be put in \boxed{}. For example, \boxed{I don’t know}, \boxed{invalid question}, \boxed{3 times}, \boxed{New York}, etc.

### Question: {question} ### Query Time: {query time}

Output: {answer}

Table 15. Inference prompt under with retrieval setup.

Inference Prompt (With Retrieval)

Input: You are given a Question, References and the time when it was asked in the Pacific Time Zone (PT), referred to as “Query Time”. The query time is formatted as “mm/dd/yyyy, hh:mm:ss PT”. The references may or may not help answer the question. Your task is to answer the question based on factual information in the references or your own knowledge.

Please adhere to the following guidelines when formulating the answer:

- 1. If the question contains a false premise or assumption, answer “invalid question”.
- 2. If you are uncertain or don’t know the answer, answer “I don’t know”.

Please reason step by step and then provide the final answer. The reasoning process must be enclosed within <think> </think> tags. The final answer MUST be put in \boxed{}. For example, \boxed{I don’t know}, \boxed{invalid question}, \boxed{3 times}, \boxed{New York}, etc.

### Question: {question} ### Query Time: {query time} ### References: {documents}

Output: {answer}

LLM-as-a-judge prompts. Below we present the judge prompts for outcome and reasoning quality in Table 16 and

- Table 17, respectively.

E. Case Study

- Table 18 presents a representative example illustrating the accuracy of the LLM-based judge.

Table 16. LLM-as-a-judge prompt for evaluating outcome.

##### LLM-as-a-judge Prompt (Outcome)

Input: Assume you are a human expert in grading predictions given by a model. You are given a question and a model prediction. Judge if the prediction matches the ground truth answer by following these steps:

- 1: Take it as granted that the Ground Truth is always correct.
- 2: If the Prediction exactly matches the Ground Truth, “score” is 1.
- 3: If the Prediction does not exactly match the Ground Truth, go through the following steps and likely give a score as 0.
- 4: If the Ground Truth is a number, “score” is 1 if and only if the Prediction gives a number that almost exactly matches the ground truth.
- 5: If the Prediction is self-contradictory, “score” must be 0.
- 6: If the prediction is not answering the question, “score” must be 0.
- 7: If the prediction is a concise and correct summary of the ground truth, “score” is 1.
- 8: If ground truth contains a set of items, prediction must contain exactly same items for the score to be 1.
- 9: Otherwise, “score” is 0. Output a JSON blob with an “explanation” field explaining your answer as short as possible and an “score” field with value 1 or 0. You should make the judgment based on provided examples.

### Examples: {examples} ### Question: {question} ### Ground Truth: {ground truth} ### Prediction: {predicted answer}

Output: {judgment}

Table 17. LLM-as-a-judge prompt for evaluating reasoning quality.

##### LLM-as-a-judge Prompt (Reasoning Quality)

Input: Assume you are a human expert in evaluating the usefulness of model-generated reasoning. You are given a question and a model-generated reasoning. Judge if the reasoning provides precise information to correctly answer the question by following these steps:

- 1: Evaluate if the reasoning directly addresses the question.
- 2: Check if the key points in the reasoning are relevant to the query.
- 3: If the reasoning provides precise and relevant information, “score” is 1.
- 4: If the reasoning is vague, unrelated, or does not address the question, “score” is 0. Output a JSON blob with an “explanation” field explaining your answer as short as possible and an “score” field with value 1 or 0. You should make the judgment based on provided examples.

### Examples: {examples} ### Question: {question} ### Ground Truth: {ground truth} ### Reasoning: {predicted reasoning}

Output: {judgment}

Table 18. An example of LLM-as-a-judge result.

##### LLM-as-a-judge Result

Question: How many times has Lizzo won the Grammy Award for Album of the Year? Ground Truth: Lizzo has never won the Grammy Award for Album of the Year, though she has been nominated twice.

Prediction: <think> To answer the question “How many times has Lizzo won the Grammy award for album of the year?”, we need to analyze the information provided in the references. Document [4] mentions that Lizzo has won 4 Grammy Awards, but it does not specifically mention the category of Album of the Year. However, Document [5] mentions that Lizzo’s album “Special” is nominated for Album of the Year, but it does not mention that she has won the award. After reviewing all the documents, I found that none of them explicitly state that Lizzo has won the Grammy Award for Album of the Year. Therefore, the answer is 0, as there is no evidence to suggest that Lizzo has won the Grammy Award for Album of the Year. </think> \boxed{0}.

LLM judge result: {“score”: 1, “explanation”: “The prediction correctly states the number of times Lizzo has won the Grammy Award for Album of the Year, which is zero.”}

