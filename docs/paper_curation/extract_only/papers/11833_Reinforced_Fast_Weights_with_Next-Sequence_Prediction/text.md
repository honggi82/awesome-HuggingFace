## Reinforced Fast Weights with Next-Sequence Prediction

Hee Seung Hwang* Xindi Wu* Sanghyuk Chun Olga Russakovsky Princeton University

# arXiv:2602.16704v1[cs.CL]18Feb2026

### Abstract

Fast weight architectures offer a promising alternative to attention-based transformers for longcontext modeling by maintaining constant memory overhead regardless of context length. However, their potential is limited by the next-token prediction (NTP) training paradigm. NTP optimizes single-token predictions and ignores semantic coherence across multiple tokens following a prefix. Consequently, fast weight models, which dynamically update their parameters to store contextual information, learn suboptimal representations that fail to capture long-range dependencies. We introduce REFINE (Reinforced Fast weIghts with Next sEquence prediction), a reinforcement learning framework that trains fast weight models under the next-sequence prediction (NSP) objective. REFINE selects informative token positions based on prediction entropy, generates multi-token rollouts, assigns selfsupervised sequence-level rewards, and optimizes the model with group relative policy optimization (GRPO). REFINE is applicable throughout the training lifecycle of pre-trained language models: mid-training, post-training, and test-time training. Our experiments on LaCT-760M and DeltaNet-1.3B demonstrate that REFINE consistently outperforms supervised fine-tuning with NTP across needle-in-a-haystack retrieval, longcontext question answering, and diverse tasks in LongBench. REFINE provides an effective and versatile framework for improving long-context modeling in fast weight architectures. Our code is publicly available.

*Equal contribution . Correspondence to: Hee Seung Hwang <hee.h@princeton.edu>.

[Figure 1]

Figure 1. Comparison of standard NTP and REFINE. Standard NTP (top) computes cross-entropy loss at each token position, providing only token-level supervision to fast weight models. REFINE (bottom) provides sequence-level supervision by generating multi-token rollouts at high-entropy positions, assigning sequencelevel rewards from hidden states, and optimizing with RL.

### 1. Introduction

Long-context modeling has become essential for large language models (LLMs). Tasks such as long-document understanding (Shaham et al., 2022; Dong et al., 2024), many-shot in-context learning (Agarwal et al., 2024; Li et al., 2024), and code generation (Liu et al., 2023; Nichols et al., 2024) require models to extract, store, and reuse information from contexts spanning thousands of tokens (Yen et al., 2024; Bai et al., 2024; 2025). While attention-based transformers demonstrate strong performance on these tasks, their computational and memory costs scale quadratically (Keles et al., 2023) with context length, creating a fundamental bottleneck for both training and inference.

Fast weight architectures offer a promising alternative by addressing this scaling problem through structural changes to the transformer block. Models such as DeltaNet (Yang et al., 2024), GatedDeltaNet (Yang et al., 2025), and LaCT (Zhang et al., 2025) replace global attention with a fixed-size memory that is dynamically updated as new tokens are processed, storing contextual information directly in model parameters

(Fig. 2). This design enables efficient inference with constant memory overhead regardless of context length (Tandon et al., 2025).

Despite their architectural differences, fast weight models are typically pre-trained with the same next-token prediction (NTP) objective used for standard transformer LLMs (Sun et al., 2024; Behrouz et al., 2024; 2025a; Yang et al., 2024; 2025; Zhang et al., 2025). In this work, we argue that NTP is a suboptimal objective for fast weight models. The NTP objective only has an immediate effect on the next token and disregards the quality of subsequent predictions that depend on the same internal state (Gloeckle et al., 2024). As a result, NTP’s token-level feedback encourages parameter updates that optimize only short-term likelihood, limiting the adaptive capacity of fast weights and model behavior over longer horizons.

To better align the training objective with the intended function of fast weights as long-context memory, we propose the next-sequence prediction (NSP) objective as a variation of NTP. NSP encourages a model to predict a semantically coherent sequence of future tokens conditioned on a given prefix. This objective directly reflects whether the information stored in fast weights enables accurate continuation over multiple steps, providing a more appropriate training signal for long-context adaptation compared to the standard NTP. However, training with NSP introduces two key challenges: (i) standard cross-entropy loss does not naturally extend to multi-token prediction without explicitly generating full continuations, and (ii) generating multiple tokens for every prefix is computationally prohibitive for long contexts.

We address these challenges by formulating NSP as a reinforcement learning (RL) problem: a model is trained to maximize sequence-level rewards derived from its predictions. Our method focuses on informative regions of the context and optimizes for NSP using policy gradient updates. Empirically, we show that RL for NSP yields superior fast weight initialization and consistently outperforms pure supervised fine-tuning (SFT) under the NTP objective.

We introduce Reinforced Fast Weights with Next Sequence Prediction (REFINE), a phase-agnostic framework that can be applied in multiple stages of the language model training lifecycle (Tab. 1). We demonstrate the effectiveness of REFINE in three stages: (i) mid-training, where we reinforce fast weight models on pretraining-like corpora to improve long-context adaptation; (ii) post-training, where RL is integrated into task-specific training loops to refine fast weights under downstream supervision; and (iii) test-time training, where REFINE reinforces fast weights directly on the prompt without additional labels.

Our experiments show that REFINE improves fast weight initialization in all three settings. For example, applying

[Figure 2]

Figure 2. Comparison of Standard Transformer and Fast Weight Models, adapted from Zhang et al. (2025). Fast weight models replace attention with a fixed-size memory implemented as a weight matrix (W), and updated according to Eq. (1).

REFINE on LaCT-760M notably improves the average performance on RULER (Hsieh et al., 2024) long-context QA tasks by 8.5% (mid-training), 15.3% (post-training), and 9.5% (test-time training), compared to pure NTP training with SFT. For DeltaNet-1.3B, we observe 20.3% (midtraining), 11.0% (post-training), and 15.0% (test-time training) gains with REFINE compared to SFT baselines. These results highlight REFINE’s flexibility and practicality in improving long-context modeling of fast weight architectures.

Contribution. Our contributions are as follows: (1) Introducing the next-sequence prediction (NSP) objective for fast weight language models, addressing the limitations of nexttoken prediction in sequence-level feedback. (2) Proposing an RL framework for optimizing NSP in fast weight models, combining entropy-based token selection and sequencelevel rewards. (3) Demonstrating that REFINE is effective across the language model training lifecycle, during mid-, post-, and test-time training.

### 2. Background

Fast Weight Architectures. Fast weight architectures replace global attention in standard transformers with fixedsize memory parameterized as weight matrices. Instead of keeping a growing key-value cache, fast weight models continually update the weight matrices as tokens are processed to store contextual information. As a result, fast weight models are often associated with test-time training (Behrouz et al., 2024) and meta-learning (Clark et al., 2022) due to the continual and task-agnostic nature of their weight updates. The update rule for fast weights can be generalized as follows (Zhang et al., 2025):

ℓ Wtkt,vt (1)

Wt+1 ← Wt − η∇Wt

where W denotes the fast weight, η is the learning rate, and kt,vt are the key, value representations of the input token

Table 1. Training Phases for Fast Weight Language Models. REFINE can be applied across all phases beyond pre-training to improve long-context modeling, using different data sources.

Training Phase Source of Training Data Pre-training General web-scraped data Mid-training Similar data to pre-training Post-training Task / instruction / preference data Test-Time Training Test data

at position t. This update rule can be viewed as learning the online mapping from key to value representations. The output representation is retrieved from fast weights via an apply operation, i.e., Wtqt, where qt is the token’s query representation. Fig. 2 illustrates the difference between standard transformer and fast weight language models.

Training Phases for Language Models. Pre-trained language models undergo additional training stages that rely on different sources of data and supervision (Tab. 1). We follow the standard taxonomy of three additional training phases: mid-training, post-training, and test-time training.

Mid-training is an extension (or continued version) of pretraining, generally used for the adaptation of a pre-trained model to specific domains or capabilities (Gururangan et al., 2020). In this paper, we apply REFINE to mid-train models on the same training dataset as pre-training to adapt pretrained models to the NSP objective and reward.

Post-Training fine-tunes pre-trained models to follow instructions and align their responses with human preferences (Ouyang et al., 2022; Rafailov et al., 2024; Guo et al., 2025). This phase typically involves SFT on task-specific instruction-response pairs, with relatively fewer gradient steps compared to pre-training. We apply REFINE during post-training through the nested learning (Behrouz et al., 2025b) technique: within each training loop, we first use REFINE to update the model on the instruction prompt alone, and then use SFT to fine-tune the model’s final response.

Test-Time Training (TTT) adapts model parameters at inference time using self-supervised objectives to handle distribution shifts from source to target (Sun et al., 2020; Wang et al., 2021; Gandelsman et al., 2022; Sun et al., 2024). TTT naturally integrates with fast weight architectures by design: each input token updates the fast weights via gradient-based rules, enabling the model to memorize and adapt to the given context on-the-fly (Zhang et al., 2025; Behrouz et al., 2024). However, fast weight models trained purely with NTP can still struggle on long-context retrieval (e.g., Needle-in-aHaystack (Hsieh et al., 2024)) due to unstable fast weight updates or insufficient long-horizon supervision.

RL for Language Modeling. Recent work has shown that NTP can be formulated as a reward maximization prob-

lem with RL (Dong et al., 2025; Hatamizadeh et al., 2025). These methods produce reasoning traces from the model before predicting the next token and provide rewards based on the similarity between the prediction and the ground truth. Existing works focus on applying RL on standard transformer LLMs with basic reasoning capability, but it is still an open question whether RL can be applied to pre-trained fast weight models. We demonstrate that RL can improve long-context capabilities of fast weight models during mid-, post-, and test-time training even without prior instruction tuning. More details are discussed in Appendix §B.

### 3. Method

Our goal is to obtain better fast weight initializations for long-context modeling by leveraging the NSP objective. We present RL as a solution to the limitations of SFT in optimizing sequence-level predictions, explained below.

#### 3.1. From Next-Token to Next-Sequence Prediction

Next Token Prediction (NTP). Standard language model pre-training involves minimizing the cross-entropy (CE) loss of the NTP objective. Given an input sequence S = (x1,...,xT), the CE loss is computed using the predicted probability distributions at each token position and the corresponding ground truth tokens:

−log p(xt+1 | x≤t). (2)

LNTP =

t

The NTP loss has two key limitations for long-context modeling. First, each term in the summation only considers single-token prediction, ignoring the semantic relationships among multiple tokens that follow the prefix. Second, NTP ignores local regions in the sequence that may be useful over the long-context by aggregating the terms uniformly.

Next Sequence Prediction (NSP). We aim to resolve the shortcomings of standard NTP by proposing the NSP objective for training fast weight models. Unlike NTP, which optimizes token-by-token predictions, NSP optimizes multi-token sequence alignment at selected positions T ∗ ⊆ {1,...,T}:

Lseq(ˆxt+1:t+k,xt+1:t+k), k > 1 (3)

LNSP =

t∈T ∗

where Lseq measures the discrepancy between the predicted sequence xˆt+1:t+k given prefix x≤t and the ground truth continuation xt+1:t+k.

A straightforward choice for Lseq is the CE loss. However, naively applying the CE loss at every position t requires generating k-token completions given all possible prefixes, which is computationally expensive especially for long contexts. Furthermore, directly matching a single reference will

[Figure 3]

- Figure 3. REFINE. We forward the sequence through the policy model and compute token-level entropy values. Sequences are split into chunks and a target token position is sampled from each chunk based on the entropy (Entropy-Based Token Selection). Prefixes are copied from the original sequence up to each target token. The policy model predicts continuations from the prefixes (Rollout Generation). Reward is computed based on the generated rollouts and ground truth tokens (Reward Assignment). Finally, we update the policy model with GRPO (Optimization with RL).

over-penalize plausible answers not exactly matching the ground truth. For example, for the ground truth sequence “cars are fast”, a semantically equivalent sequence “automobiles move quickly” may still result in a high CE loss.

We propose two approaches to tackle this issue. First, instead of unrolling tokens at every index t, we select informative positions T ∗ with high NTP entropy, which indicate high uncertainty. Second, we optimize Eq. (3) using an RL algorithm that maximizes the expected self-supervised reward R of sequence predictions. Let πθ denote the language model parameterized by θ. We define the sequence-level loss as follows:

Lseq = −Exˆ

t+1:t+k∼πθ(·|x≤t) R(ˆxt+1:t+k,xt+1:t+k) .

(4) This formulation has two advantages: (1) optimizing k-step continuations leverages higher information content compared to optimizing single-token predictions; (2) we can assign rewards to multiple plausible continuations based on their semantic similarity to the ground truth. For brevity, we use R(t) to denote R(ˆxt+1:t+k,xt+1:t+k).

While prior work has explored NSP for standard transformer LLMs (Gloeckle et al., 2024; Liu et al., 2025), we are the first to investigate RL-based NSP for fast weight models. More discussion can be found in Appendix §B.

#### 3.2. REFINE

Our REFINE framework (Fig. 3) consists of four key steps: (1) entropy-based token selection, (2) rollout generation, (3) reward assignment, and (4) optimization with RL.

Entropy-Based Token Selection. Given an input sequence S = (x1,...,xT), we forward S through the policy model πθ and compute the NTP entropy values Ht at each token position t:

Ht = H(πθ(· | x≤t−1)), xt ∈ S (5)

We smooth the entropy distribution within S using a 1D average pooling with kernel size k. We partition the input sequence S into c contiguous chunks of equal length:

- S = (S1,S2,...,Sc). For each chunk Si, we sample one token position with probability proportional to the softmax of its entropy. Concretely, we compute:

pi(t) =

eH

t/τ

t′∈Ti eHt′/τ , Ti = {t′ | xt′ ∈ Si} (6) where τ is a temperature parameter (we set τ = 1 if not specified). We then draw a sampling index ti ∼ pi(t). This produces one entropy-weighted token position for each chunk, yielding a set of sampled positions T ∗ = {t1,...,tc}. These positions represent regions that exhibit relatively high local uncertainty, allowing training to focus on challenging predictions within the context. Furthermore, sampling evenly from each chunk distributes training signals across the entire sequence length.

Rollout Generation. For each high-entropy position ti ∈

- T ∗, we construct a truncated prefix x≤t

, yielding c distinct partial sequences {x≤t

i

c} from S. These prefixes capture the full context leading up to each high-entropy position that is sampled. From each truncated prefix x≤t

,...,x≤t

1

, we generate a k-token continuation xˆt

i

i+1:ti+k using the current policy and extract the hidden states of the final layer before the logits:

##### hpredk (ti) = hpred(ti + 1),...,hpred(ti + k) . (7)

We also extract the hidden states of the ground-truth continuation xt

i+1:ti+k from the initial forward pass:

##### hgtk(ti) = hgt(ti + 1),...,hgt(ti + k) . (8)

We measure the discrepancy between the hidden states of the predicted and ground truth tokens to compute the reward.

Reward Assignment. Given the hidden states of predicted and ground truth continuations hpredk (ti),hgtk(ti) ∈ Rk×d, we assign a smooth similarity reward for an arbitrary similarity function φ defined as:

k

1 k

φ hpred(ti + j),hgt(ti + j) . (9)

Rkφ(ti) =

j=1

We use cosine similarity for φ. This reward encourages the model to produce hidden representations that align with those induced by the ground-truth tokens. The purpose of a similarity-reward of representations is to improve generalizability across contexts and stability, especially in the early training steps. It assigns smooth, non-zero rewards to semantically similar tokens that lead to hidden state embeddings that are closer in the latent space. Qualitative examples of the cosine similarity reward are discussed in Appendix §F.

Optimization with RL. Once we compute the reward for each rollout, we have a set of rollouts oT ∗ = {xˆt

c+1:tc+k} and corresponding rewards

1+1:t1+k,...,xˆt

RT ∗ = {Rkφ(t1),...,Rkφ(tc)}. The rewards from the same sequence S are standardized to compute the advantage fol-

lowing Shao et al. (2024). We employ the GRPO algorithm (Shao et al., 2024) to compute the NSP loss based on the rollouts and their relative advantages. The policy gradients therefore maximize the following objective:

≤t∼D,xˆt+1:t+k∼πθold(· |x≤t)[Rkφ(t)], (10)

J (θ) = Ex

where D is the set of all {x≤t}Tt=1. To prevent catastrophic forgetting, the final loss is a weighted sum of the NSP loss and the standard NTP loss (computed over the entire sequence S), with coefficients λRL and λSFT respectively. The weights are adjusted based on the training phase.

#### 3.3. Hybrid Reward for RL

TTT introduces unique constraints. First, evaluations are usually conducted in the low-data regime, which leads to smaller batch sizes and limited room for meta-adaptation across episodes. Second, effective memorization of the given context becomes more important than contextual generalization. For scenarios that require stronger context memorization (e.g., TTT), we introduce a binary exact match reward Rbinary defined as:

k

1 k

Rkbinary(ti) =

I[xt+j = xˆt+j]. (11)

j=1

We use a mixture of Rφ and Rbinary for post-training:

Rkhybrid(ti) = Rkφ(ti) + Rkbinary(ti). (12)

During post-training, the train and test datasets usually have a similar distribution. The hybrid reward is designed to balance contextual generalization and memorization.

Table 2. Training configurations. Detailed training configurations for Mid-training (MidTr), post-training (PostTr) and test-time training (TTT) are shown.

c k n batch size λSFT λRL reward lr

MidTr 8 5 1 128 1 0.2 Rφ 1e-6 PostTr 8 5 1 64 1 0.2 Rhybrid 1e-6 TTT 8 5 1 8 1 0.4 Rbinary 1e-6

### 4. Experiments

In this section, we conduct experiments to show that REFINE improves long-context modeling of fast weight models. We first establish the models and datasets we use for training and evaluation (§4.1). We then report results on three training phases: mid-training (§4.2), post-training (§4.3), and test-time training (§4.4). Finally, we provide analysis on reward assignment and entropy-based token selection (§4.5), followed by ablations on rollout length and the number of chunks (§4.6).

#### 4.1. Setup

Models. We use two fast weight language models, LaCT760M (Zhang et al., 2025) and DeltaNet-1.3B (Yang et al., 2024), as the pre-trained models. LaCT adapts the model by updating its fast weight parameters, whereas DeltaNet keeps parameters fixed but updates a parallelizable memory state. We show that REFINE can improve these distinct fast weight mechanisms during mid-training, post-training, and test-time training.

Datasets and Benchmarks. As shown in Tab. 1, data for each training phase comes from different sources. For midtraining, we employ a training dataset similar to that used to pre-train the fast weight models. Specifically, we perform mid-training with Long-Data-Collections (TogetherAI, 2024), which is the pre-training dataset for LaCT (Zhang et al., 2025). We evaluate the quality of the mid-trained models on RULER NIAH tasks (Hsieh et al., 2024) and Booksum (Kry´sci´nski et al., 2022).

We consider two additional scenarios: (1) multi-doc QA tasks and (2) long-context tasks. For multi-doc QA tasks, we conduct post-training on synthetically generated SQuADQA and HotpotQA tasks from RULER (Hsieh et al., 2024). Then, we apply TTT during evaluation. For the long-context tasks, we employ 12 tasks from LongBench (Bai et al., 2024) and apply TTT on the mid-trained models. More details on the datasets can be found in Appendix §C, Tab. C.1.

Training Configurations. Tab. 2 shows the training configuration for REFINE in each phase. c denotes the number of chunks per sequence, k denotes the number of tokens per rollout, and n denotes the number of rollouts per sampled

- Table 3. Performance on Long-Context Retrieval Tasks. We evaluate mid-trained (MidTr) models on the NIAH tasks in RULER at 4K, 8K, and 16K context lengths (standard SFT vs. REFINE). Highest scores in each category are highlighted in bold.

RULER S-NIAH RULER MK-NIAH RULER MQ-NIAH RULER MV-NIAH

4K 8K 16K Avg 4K 8K 16K Avg 4K 8K 16K Avg 4K 8K 16K Avg

LaCT-760M 98.8 91.2 95.8 95.3 70.2 44.8 24.2 46.2 39.2 24.9 17.4 27.1 40.6 27.0 17.7 28.4 + SFT MidTr 98.4 90.8 97.6 95.6 70.6 45.0 24.2 46.6 40.7 25.2 17.4 27.8 41.7 26.3 17.7 28.5

- + REFINE MidTr 99.0 93.0 96.8 96.3 70.4 46.0 26.6 47.7 40.5 25.8 18.0 28.1 41.2 29.5 18.6 29.8 DeltaNet-1.3B 100.0 100.0 100.0 100.0 23.6 17.8 3.4 14.9 27.7 3.9 1.8 11.1 23.9 5.4 2.0 10.4

+ SFT MidTr 100.0 100.0 100.0 100.0 23.8 19.2 7.8 16.9 33.2 16.8 3.5 17.8 28.3 19.3 3.7 17.1

- + REFINE MidTr 100.0 100.0 99.8 99.9 25.0 21.4 8.8 18.4 37.3 18.8 3.6 19.9 29.0 20.4 4.0 17.8

- Table 4. Performance on Multi-Doc QA Tasks. We evaluate various training strategies during mid-training (MidTr), post-training (PostTr), and test-time training (TTT) for multi-document question and answering tasks. For Nested SFT and Nested REFINE, we train the model with the training method on the prompt portion of the sample as described in §4.3. Higher is better. First and second highest scores on each task are highlighted in bold and underline, respectively.

RULER SQuADQA RULER HotpotQA MidTr PostTr TTT 4K 8K 16K Avg 4K 8K 16K Avg

- - - 17.5 14.0 6.5 12.7 20.5 14.5 12.0 15.7

SFT - - 19.5 14.0 6.0 13.2 21.5 15.0 12.0 16.2 REFINE - - 20.5 15.5 8.0 14.7 23.0 16.0 12.5 17.2 REFINE SFT - 30.5 20.0 7.5 19.3 25.0 16.5 11.5 17.7 REFINE Nested SFT - 38.0 20.0 7.5 21.8 24.0 16.0 12.5 17.5 REFINE Nested REFINE - 43.5 24.5 8.5 25.5 27.0 19.5 13.0 19.8 REFINE Nested REFINE SFT 43.5 21.5 8.5 24.5 26.5 24.0 13.0 21.2 REFINE Nested REFINE REFINE 45.5 25.5 10.0 27.0 28.5 25.5 15.0 23.0

LaCT 760M

- - - 11.0 5.0 2.0 6.0 10.0 3.0 2.5 5.2

SFT - - 9.0 6.0 3.0 6.0 9.0 8.0 4.5 7.2 REFINE - - 10.0 6.5 4.0 6.8 12.0 9.5 5.5 9.0 REFINE SFT - 11.0 7.0 5.0 7.7 14.0 11.0 6.5 10.5 REFINE Nested SFT - 12.0 8.0 5.0 8.3 16.5 10.0 7.5 11.3 REFINE Nested REFINE - 14.0 10.5 6.5 10.3 15.0 11.0 8.5 11.5 REFINE Nested REFINE SFT 16.5 10.5 6.5 11.2 16.0 10.5 6.5 11.0 REFINE Nested REFINE REFINE 17.5 12.5 7.0 12.3 18.0 13.5 8.0 13.2

DeltaNet 1.3B

position. While we fix c = 8, k = 5, and n = 1 in all phases, we vary λRL and the reward function to suit each phase. We provide the training hyperparameters used for REFINE during all training phases in Tab. D.1 (Section D).

#### 4.2. Impact of REFINE on Mid-Training

During mid-training, we train both models on Long-DataCollections (TogetherAI, 2024) for 100 steps using a batch size of 128 (≈200M training tokens). We compare against the pure SFT baseline trained under identical conditions.

We evaluate the mid-trained models on four NIAH tasks in RULER (Hsieh et al., 2024) (Single NIAH, Multi-key NIAH, Multi-query NIAH, and Multi-value NIAH) at 4K, 8K, and 16K context lengths using Language Model Evaluation Harness (Gao et al., 2024).

- Tab. 3 shows that the mid-trained model by REFINE consistently outperforms the original pre-trained model and the SFT mid-trained model on different tasks and models. For example, REFINE significantly improves DeltaNet in the Multi-key NIAH (+23.5% from no mid-training and +8.8%

from SFT mid-training). This suggests that REFINE leads to improvements in long-context retrieval.

We also report the validation NTP accuracy on the Booksum (Kry´sci´nski et al., 2022) dataset. Fig. 4 shows the evolution of validation NTP accuracy during mid-training. Interestingly, even though the SFT baseline directly optimizes NTP, REFINE’s gains in NTP accuracy are higher than those of SFT. Specifically, for LaCT, whose mid-training and pre-training use the same training dataset, REFINE consistently improves NTP, unlike SFT. We conjecture that the NSP objective’s sequence-level supervision provides learning signals that NTP does not.

In addition to long-context retrieval tasks, we also report the effectiveness of REFINE mid-training on multi-doc QA tasks and long-context tasks in Tab. 4 and Tab. 5. In Tab. 4, REFINE mid-trained models (3rd rows) outperform SFT mid-trained models (2nd rows) by large margins. For example, REFINE mid-training improves the average performance of pre-trained DeltaNet on RULER HotpotQA by 73.1% compared to no mid-training and 22.0% compared to

- Table 5. Performance on Long-Context Tasks in LongBench. We study the impact of the learning algorithm during mid-training (MidTr) and test-time training (TTT) on tasks with long-context. SFT denotes the supervised fine-tuning with next-token prediction. We evaluate on 12 tasks in LongBench, filtered for samples with at most 16K tokens. Details are similar to Tab. 4.

Single-doc QA Multi-doc QA Summarization Few-shot QA Coding MidTr TTT NQ QR MF HP 2W QM MN SS TC TQ LC RP Avg

- - 6.5 10.5 7.2 11.7 9.8 13.6 9.2 14.2 10.5 8.0 26.7 29.7 13.1

SFT - 5.8 10.1 7.4 12.6 9.2 13.1 10.5 13.8 7.5 12.2 29.8 30.1 13.5 REFINE - 6.5 11.1 12.6 19.6 18.0 14.3 15.9 17.0 11.0 12.9 32.9 31.1 16.9 REFINE SFT 5.1 12.6 13.5 15.9 18.4 13.2 16.2 17.4 12.5 16.2 32.0 31.4 17.0 REFINE REFINE 6.7 14.5 14.1 18.4 22.8 13.9 15.9 17.4 15.5 11.8 32.2 32.3 18.0

LaCT 760M

- - 6.5 8.7 10.0 4.8 6.4 12.4 16.3 9.4 17.7 15.1 33.8 29.0 14.2

SFT - 5.7 9.4 8.3 9.2 8.6 14.7 16.1 15.5 22.9 15.2 33.1 29.2 15.7 REFINE - 6.5 9.5 10.1 9.6 8.7 15.2 15.0 16.2 25.0 21.4 35.9 31.1 17.0 REFINE SFT 7.2 9.6 10.5 6.8 7.2 14.9 16.2 17.3 28.0 16.6 34.1 29.2 16.5 REFINE REFINE 7.5 9.2 11.5 9.5 8.6 14.7 16.1 16.5 31.5 24.7 35.2 30.0 17.9

DeltaNet 1.3B

(a) DeltaNet-1.3B

(b) LaCT-760M

33.0

SFT

SFT

30.5

ReFINE

ReFINE

Accuracy

Accuracy

32.9

30.0

29.5

32.8

0 20 40 60 80 100

0 20 40 60 80 100

Steps

Steps

- Figure 4. NTP Accuracy on Booksum. REFINE mid-training on DeltaNet-1.3B (a) and LaCT-760M (b) leads to a consistent increase in NTP accuracy on the validation dataset while that of SFT mid-training is stagnant. The error bars show the minimum and maximum values from three independent trials.

SFT mid-training. Tab. 5 shows a similar result; REFINE (3rd rows) consistently outperforms SFT (2nd rows).

#### 4.3. Impact of REFINE on Post-Training

We show that REFINE strengthens post-training, which aims to align the model’s responses to a given task. In our experiments, we fine-tune the mid-trained models in §4.2 on synthetically generated samples for the target tasks of RULER (Hsieh et al., 2024) SQuADQA and HotpotQA.

During post-training, we apply REFINE as a nested learning algorithm. Within each training loop, we first update the model on the prompt with REFINE before generating a final response, which is fine-tuned to align with a reference response. We compare three post-training scenarios. (1) SFT: we fine-tune the model directly on the post-training dataset with NTP (no nested learning). (2) Nested SFT: we apply nested training strategy with NTP loss. (3) Nested REFINE: we apply nested training strategy with REFINE.

- Tab. 4 shows that post-training with nested REFINE (6th rows) outperforms SFT (4th rows) and nested SFT (5th rows). For instance, nested REFINE on LaCT-760M improves the average score on SQuADQA by 17% compared to nested SFT (25.5 vs. 21.8), and by 24.1% for DeltaNet-

1.3B (10.3 vs. 8.3). These results suggest that NSP provides better task-agnostic learning signals than NTP to capture the context distribution in the fast weights.

#### 4.4. Impact of REFINE on Test-Time Training (TTT)

REFINE can be used during inference to improve performance on a target task. During inference, we apply REFINE on the prompt before letting the model generate the final response. In order to maximize memory capacity over long contexts, we provide more direct learning signals by using binary exact match reward (Rbinary) and a higher RL loss coefficient (λRL = 0.4). We replace REFINE with pure SFT on the prompt for comparison.

We apply TTT on the post-trained models for multi-doc QA tasks in §4.3. The 7th and 8th rows of Tab. 4 show the results of SFT TTT and REFINE TTT, respectively. REFINE consistently outperforms SFT during TTT similar to mid-training and post-training scenarios.

We observe a similar result in long-context tasks. Tab. 5 shows that TTT with REFINE yields superior performance compared to SFT across diverse subtasks in LongBench (Bai et al., 2024). This suggests that NSP provides stronger adaptation signals to facilitate compression of contextual

###### Ablation on k

###### Ablation on c

17.0

17.0

Averagescore

16.8

16.8

16.6

16.6

16.4

DeltaNet

DeltaNet

16.4

LaCT

LaCT

16.2

1 3 5 7

2 4 6 8

k

c

- Figure 5. Ablation on k and c. We mid-train models with different numbers of tokens per rollout k (left) and numbers of chunks per sequence c (right). We evaluate on 16K-context samples from 12 tasks in LongBench (Bai et al., 2024). With cosine similarity reward, there is an optimal k. Higher c leads to more NSP training per sequence, which leads to better overall performance.

- Table 6. Impact of reward function on mid-training. We compare Rbinary (binary exact match) and Rφ (ours) reward strategies on 12 tasks in LongBench.

Reward Avg. Score LaCT-760M +REFINE MidTr

Rbinary 16.6

Rφ (ours) 16.9 DeltaNet-1.3B +REFINE MidTr

Rbinary 16.5 Rφ (ours) 17.0

information in fast weights not only at the token-level as in NTP, but also at the sequence level.

#### 4.5. Analysis

Analysis of Reward Functions. We analyze the impact of alternative reward functions for REFINE. During midtraining, REFINE assigns a smooth, semantically driven reward to each rollout based on the cosine similarity of the hidden states of predicted and ground truth tokens (Eq. (9)). We repeat the mid-training process after replacing cosine similarity rewards (Rφ) with binary exact match rewards (Rbinary). Tab. 6 shows that Rφ achieves superior performance on both models for mid-training: +1.8% over Rbinary for LaCT-760M and +3.0% for DeltaNet-1.3B. This demonstrates that the similarity-based reward leads to better generalization under the NSP training objective. More discussion on reward functions in TTT is in Appendix §E, Tab. E.2.

Analysis of Entropy-Based Token Selection. We analyze the role of entropy-based token selection on REFINE mid-training. REFINE samples a target token from each chunk weighted by the token-level NTP entropy. Rollouts are generated to predict the local region following the sampled target tokens. We repeat the mid-training process after replacing entropy-based sampling with three alternative sampling methods: uniform sampling, maximum entropy selection, and minimum entropy selection. Tab. 7 shows that entropy-weighted sampling achieves the best performance on both models: +4.3% over uniform, +3.0% over max entropy, and +1.8% over min entropy for LaCT-760M;

Table 7. Impact of token selection. We compare various token selection strategies on 12 tasks in LongBench: uniform random, selecting the token with maximum entropy (arg max H) or minimum entropy (arg min H), and our entropy-weighted sampling.

Sampling Avg. Score

Uniform 16.2 arg max H 16.4 arg min H 16.6 Ours 16.9

LaCT-760M +REFINE MidTr

Uniform 15.9 arg max H 16.7 arg min H 16.8 Ours 17.0

DeltaNet-1.3B +REFINE MidTr

+6.9% over uniform, +1.8% over max entropy, and +1.2% over min entropy for DeltaNet-1.3B. This shows that NSP training is most effective when applied to regions with a balanced mixture of uncertainty levels.

#### 4.6. Ablations

Rollout Length. We examine the effect of rollout length k on REFINE mid-training, which is the number of tokens to unroll per rollout. k determines how far the model is expected to predict given a prefix. We mid-train both models with different values of k and evaluate on LongBench tasks in Tab. 5. We observe that the average score increases until k = 5 and decreases at k = 7 (Fig. 5, left). We hypothesize that the sharpness of the reward starts to degrade when the rewards are averaged over longer rollouts. More discussion on the reward distribution can be found in Appendix §E.

Number of Chunks per Sequence. We study the impact of number of chunks c per sequence on downstream performance by mid-training the models with different numbers of chunks. c determines the number of target tokens sampled based on entropy values, as well as the total number of rollouts per sequence. We evaluate the mid-trained models on LongBench tasks in Tab. 5. We find that the average score increases consistently as the number of chunks per sequence increases (Fig. 5, right). The average score increases from 16.5 (c = 2) to 16.9 (c = 8) for LaCT-760M and from 16.3 (c = 2) to 17.0 (c = 8) for DeltaNet-1.3B. This indicates that the quality of fast weight initializations increases as the frequency of sequence-level predictions increases.

### 5. Discussion

Conclusion. We introduce the NSP training objective for fast weight language models to address the limitations of NTP in providing sequence-level feedback. We propose REFINE, a RL framework which leverages entropy-based token selection and sequence-level rewards to efficiently train fast weight models under the NSP objective. Our ex-

periments demonstrate that REFINE is effective throughout the training lifecycle of fast weight models, showing consistent improvements in long-context benchmarks. REFINE presents RL for NSP as a flexible and practical pathway towards long-context modeling of fast weight architectures.

Limitations. REFINE’s cosine similarity reward starts to deteriorate for overly long rollouts. Introducing reward functions that capture richer semantic similarity, such as edit distance, could address the diminishing returns on k. Second, the optimal rollout length for a given prefix is contextdependent, suggesting that dynamic adjustment may isolate semantically meaningful regions more effectively.

Future Work. Fully incorporating NSP into the standard training framework of fast weight models requires architectural changes. Efficient transfer of fast weights across truncated prefixes will significantly accelerate rollout generation, allowing REFINE to scale data and compute further.

Impact Statement. This work aims to improve the longcontext modeling capabilities of fast weight architectures by introducing a novel training algorithm, rather than proposing new datasets or model architectures. The potential societal impacts of this work primarily depend on the data used to train models with our method. We encourage practitioners to consider the quality of and bias in the training data before deployment in real-world settings.

### Acknowledgments

We thank Yoonsang Lee and Tianyuan Zhang for helpful discussions and valuable insights throughout this project.

### References

Agarwal, R., Singh, A., Zhang, L., Bohnet, B., Rosias, L., Chan, S., Zhang, B., Anand, A., Abbas, Z., Nova, A., et al. Many-shot in-context learning. Advances in Neural Information Processing Systems, 37:76930–76966, 2024.

Ainslie, J., Lee-Thorp, J., De Jong, M., Zemlyanskiy, Y., Lebr´on, F., and Sanghai, S. Gqa: Training generalized multi-query transformer models from multi-head checkpoints. arXiv preprint arXiv:2305.13245, 2023.

Aky¨urek, E., Damani, M., Zweiger, A., Qiu, L., Guo, H., Pari, J., Kim, Y., and Andreas, J. The surprising effectiveness of test-time training for few-shot learning. arXiv preprint arXiv:2411.07279, 2024.

Arora, S., Yang, B., Eyuboglu, S., Narayan, A., Hojel, A., Trummer, I., and R´e, C. Language models enable simple systems for generating structured views of heterogeneous data lakes. arXiv preprint arXiv:2304.09433, 2023.

Bach, S., Sanh, V., Yong, Z.-X., Webson, A., Raffel, C., Nayak, N. V., Sharma, A., Kim, T., Bari, M. S., Fevry, T., et al. Promptsource: An integrated development environment and repository for natural language prompts. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics: System Demonstrations, pp. 93–104, 2022.

Bai, Y., Lv, X., Zhang, J., Lyu, H., Tang, J., Huang, Z., Du, Z., Liu, X., Zeng, A., Hou, L., et al. Longbench: A bilingual, multitask benchmark for long context understanding. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics, 2024.

Bai, Y., Tu, S., Zhang, J., Peng, H., Wang, X., Lv, X., Cao, S., Xu, J., Hou, L., Dong, Y., et al. Longbench v2: Towards deeper understanding and reasoning on realistic long-context multitasks. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 3639–3664, 2025.

Bansal, R., Zhang, A., Tiwari, R., Madaan, L., Duvvuri, S. S., Khatri, D., Brandfonbrener, D., Alvarez-Melis, D., Bhargava, P., Kale, M. S., et al. Let’s (not) just put things in context: Test-time training for long-context llms. arXiv preprint arXiv:2512.13898, 2025.

Behrouz, A., Zhong, P., and Mirrokni, V. Titans: Learning to memorize at test time. arXiv preprint arXiv:2501.00663, 2024.

Behrouz, A., Li, Z., Kacham, P., Daliri, M., Deng, Y., Zhong, P., Razaviyayn, M., and Mirrokni, V. Atlas: Learning to optimally memorize the context at test time. arXiv preprint arXiv:2505.23735, 2025a.

Behrouz, A., Razaviyayn, M., Zhong, P., and Mirrokni, V. Nested learning: The illusion of deep learning architectures. arXiv preprint arXiv:2512.24695, 2025b.

Beltagy, I., Peters, M. E., and Cohan, A. Longformer: The long-document transformer. arXiv preprint arXiv:2004.05150, 2020.

Bisk, Y., Zellers, R., Gao, J., Choi, Y., et al. Piqa: Reasoning about physical commonsense in natural language. In Proceedings of the AAAI conference on artificial intelligence, volume 34, pp. 7432–7439, 2020.

Child, R. Generating long sequences with sparse transformers. arXiv preprint arXiv:1904.10509, 2019.

Choromanski, K., Likhosherstov, V., Dohan, D., Song, X., Gane, A., Sarlos, T., Hawkins, P., Davis, J., Mohiuddin, A., Kaiser, L., et al. Rethinking attention with performers. arXiv preprint arXiv:2009.14794, 2020.

Clark, K., Guu, K., Chang, M.-W., Pasupat, P., Hinton, G., and Norouzi, M. Meta-learning fast weight language models. arXiv preprint arXiv:2212.02475, 2022.

Clark, P., Cowhey, I., Etzioni, O., Khot, T., Sabharwal, A., Schoenick, C., and Tafjord, O. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv preprint arXiv:1803.05457, 2018.

Dao, T. and Gu, A. Transformers are ssms: Generalized models and efficient algorithms through structured state space duality. arXiv preprint arXiv:2405.21060, 2024.

Dong, Q., Dong, L., Tang, Y., Ye, T., Sun, Y., Sui, Z., and Wei, F. Reinforcement pre-training. arXiv preprint arXiv:2506.08007, 2025.

Dong, Z., Tang, T., Li, J., Zhao, W. X., and Wen, J.-R. Bamboo: A comprehensive benchmark for evaluating long text modeling capacities of large language models. In Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024), pp. 2086–2099, 2024.

Gandelsman, Y., Sun, Y., Chen, X., and Efros, A. A. Testtime training with masked autoencoders. In Advances in Neural Information Processing Systems, volume 35, pp. 29374–29385, 2022.

Gao, L., Biderman, S., Black, S., Golding, L., Hoppe, T., Foster, C., Phang, J., He, H., Thite, A., Nabeshima, N., et al. The pile: An 800gb dataset of diverse text for language modeling. arXiv preprint arXiv:2101.00027, 2020.

Gao, L., Tow, J., Abbasi, B., Biderman, S., Black, S., DiPofi, A., Foster, C., Golding, L., Hsu, J., Le Noac’h, A., Li, H., McDonell, K., Muennighoff, N., Ociepa, C., Phang, J., Reynolds, L., Schoelkopf, H., Skowron, A., Sutawika, L., Tang, E., Thite, A., Wang, B., Wang, K., and Zou, A. The language model evaluation harness, 07 2024.

Gloeckle, F., Idrissi, B. Y., Rozi`ere, B., Lopez-Paz, D., and Synnaeve, G. Better & faster large language models via multi-token prediction. arXiv preprint arXiv:2404.19737, 2024.

Gu, A. and Dao, T. Mamba: Linear-time sequence modeling with selective state spaces. In First conference on language modeling, 2024.

Guo, D., Yang, D., Zhang, H., Song, J., Zhang, R., Xu, R., Zhu, Q., Ma, S., Wang, P., Bi, X., et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

Gururangan, S., Marasovi´c, A., Swayamdipta, S., Lo, K., Beltagy, I., Downey, D., and Smith, N. A. Don’t stop pretraining: Adapt language models to domains and tasks. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pp. 8342–8360, 2020.

Hatamizadeh, A., Akter, S. N., Prabhumoye, S., Kautz, J., Patwary, M., Shoeybi, M., Catanzaro, B., and Choi, Y. Rlp: Reinforcement as a pretraining objective. arXiv preprint arXiv:2510.01265, 2025.

Hsieh, C.-P., Sun, S., Kriman, S., Acharya, S., Rekesh, D., Jia, F., Zhang, Y., and Ginsburg, B. Ruler: What’s the real context size of your long-context language models? arXiv preprint arXiv:2404.06654, 2024.

Huang, A., Block, A., Foster, D. J., Rohatgi, D., Zhang, C., Simchowitz, M., Ash, J. T., and Krishnamurthy, A. Self-improvement in language models: The sharpening mechanism. arXiv preprint arXiv:2412.01951, 2024.

Katharopoulos, A., Vyas, A., Pappas, N., and Fleuret, F. Transformers are rnns: Fast autoregressive transformers with linear attention. In International conference on machine learning, pp. 5156–5165. PMLR, 2020.

Keles, F. D., Wijewardena, P. M., and Hegde, C. On the computational complexity of self-attention. In International conference on algorithmic learning theory, pp. 597–619. PMLR, 2023.

Kry´sci´nski, W., Rajani, N., Agarwal, D., Xiong, C., and Radev, D. Booksum: A collection of datasets for longform narrative summarization. In Findings of the association for computational linguistics: EMNLP 2022, pp. 6536–6558, 2022.

Li, T., Zhang, G., Do, Q. D., Yue, X., and Chen, W. Longcontext llms struggle with long in-context learning. arXiv preprint arXiv:2404.02060, 2024.

Liu, T., Xu, C., and McAuley, J. Repobench: Benchmarking repository-level code auto-completion systems. arXiv preprint arXiv:2306.03091, 2023.

Liu, Y., Cao, Y., Li, H., Luo, G., Chen, Z., Wang, W., Liang, X., Qi, B., Wu, L., Tian, C., et al. Sequential diffusion language models. arXiv preprint arXiv:2509.24007, 2025.

Lockard, C., Shiralkar, P., and Dong, X. L. Openceres: When open information extraction meets the semistructured web. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pp. 3047–3056, 2019.

Merity, S., Xiong, C., Bradbury, J., and Socher, R. Pointer sentinel mixture models. arXiv preprint arXiv:1609.07843, 2016.

Mishra, S., Khashabi, D., Baral, C., and Hajishirzi, H. Crosstask generalization via natural language crowdsourcing instructions. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 3470–3487, 2022.

Nichols, D., Davis, J. H., Xie, Z., Rajaram, A., and Bhatele, A. Can large language models write parallel code? In Proceedings of the 33rd International Symposium on HighPerformance Parallel and Distributed Computing, pp. 281–294, 2024.

Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., et al. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 35:27730–27744, 2022.

Paperno, D., Kruszewski, G., Lazaridou, A., Pham, N.-Q., Bernardi, R., Pezzelle, S., Baroni, M., Boleda, G., and Fern´andez, R. The lambada dataset: Word prediction requiring a broad discourse context. In Proceedings of the 54th annual meeting of the association for computational linguistics, 2016.

Rafailov, R., Sharma, A., Mitchell, E., Manning, C. D., Ermon, S., and Finn, C. Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems, 36, 2024.

Sakaguchi, K., Le Bras, R., Bhagavatula, C., and Choi, Y. Winogrande: An adversarial winograd schema challenge at scale. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, pp. 8732–8740, 2020.

Shaham, U., Segal, E., Ivgi, M., Efrat, A., Yoran, O., Haviv, A., Gupta, A., Xiong, W., Geva, M., Berant, J., et al. Scrolls: Standardized comparison over long language sequences. arXiv preprint arXiv:2201.03533, 2022.

Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang, H., Zhang, M., Li, Y., Wu, Y., et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Sun, Y., Wang, X., Liu, Z., Miller, J., Efros, A., and Hardt, M. Test-time training with self-supervision for generalization under distribution shifts. In International conference on machine learning, pp. 9229–9248. PMLR, 2020.

Sun, Y., Li, X., Dalal, K., Xu, J., Vikram, A., Zhang, G., Dubois, Y., Chen, X., Wang, X., Koyejo, S., et al. Learning to (learn at test time): Rnns with expressive hidden states. arXiv preprint arXiv:2407.04620, 2024.

Tandon, A., Dalal, K., Li, X., Koceja, D., Rød, M., Buchanan, S., Wang, X., Leskovec, J., Koyejo, S., Hashimoto, T., et al. End-to-end test-time training for long context. arXiv preprint arXiv:2512.23675, 2025.

Tay, Y., Dehghani, M., Tran, V. Q., Garcia, X., Wei, J., Wang, X., Chung, H. W., Shakeri, S., Bahri, D., Schuster, T., et al. Ul2: Unifying language learning paradigms. arXiv preprint arXiv:2205.05131, 2022.

Team, Q. Qwen3 technical report, 2025. URL https: //arxiv.org/abs/2505.09388.

TogetherAI. Long data collections database. https://huggingface. co/datasets/togethercomputer/ Long-Data-Collections, 2024.

van de Ven, G. M., Soures, N., and Kudithipudi, D. Continual learning and catastrophic forgetting. arXiv preprint arXiv:2403.05175, 2024.

Wang, D., Shelhamer, E., Liu, S., Olshausen, B., and Darrell, T. Tent: Fully test-time adaptation by entropy minimization. In International Conference on Learning Representations, 2021.

Wang, S., Li, B. Z., Khabsa, M., Fang, H., and Ma, H. Linformer: Self-attention with linear complexity. arXiv preprint arXiv:2006.04768, 2020.

Weber, M., Fu, D., Anthony, Q., Oren, Y., Adams, S., Alexandrov, A., Lyu, X., Nguyen, H., Yao, X., Adams, V., et al. Redpajama: an open dataset for training large language models. Advances in neural information processing systems, 37:116462–116492, 2024.

Yang, S., Wang, B., Zhang, Y., Shen, Y., and Kim, Y. Parallelizing linear transformers with the delta rule over sequence length. Advances in neural information processing systems, 37:115491–115522, 2024.

Yang, S., Kautz, J., and Hatamizadeh, A. Gated delta networks: Improving mamba2 with delta rule. In International Conference on Learning Representations, 2025.

Yen, H., Gao, T., Hou, M., Ding, K., Fleischer, D., Izsak, P., Wasserblat, M., and Chen, D. Helmet: How to evaluate long-context language models effectively and thoroughly. arXiv preprint arXiv:2410.02694, 2024.

Zaheer, M., Guruganesh, G., Dubey, K. A., Ainslie, J., Alberti, C., Ontanon, S., Pham, P., Ravula, A., Wang, Q., Yang, L., et al. Big bird: Transformers for longer sequences. Advances in neural information processing systems, 33:17283–17297, 2020.

Zellers, R., Holtzman, A., Bisk, Y., Farhadi, A., and Choi, Y. Hellaswag: Can a machine really finish your sentence? arXiv preprint arXiv:1905.07830, 2019.

Zhang, T., Bi, S., Hong, Y., Zhang, K., Luan, F., Yang, S., Sunkavalli, K., Freeman, W. T., and Tan, H. Test-time training done right. arXiv preprint arXiv:2505.23884, 2025.

Zuo, Y., Zhang, K., Sheng, L., Qu, S., Cui, G., Zhu, X., Li, H., Zhang, Y., Long, X., Hua, E., et al. Ttrl: Test-time reinforcement learning. arXiv preprint arXiv:2504.16084, 2025.

- A. Notation Table A.1. Glossary and notation.

Symbol Description Sequences and Tokens

- S = (x1,...,xT) Input sequence with T tokens xt Token at position t x<t, x≤t Token sequence up to (and including) position t xt+1:t+k k-token ground-truth continuation starting from position t + 1 xˆt+1:t+k Generated rollout token sequence from position t + 1
- T Total sequence length k Rollout length (number of tokens to predict per rollout) c Number of chunks per sequence n Number of rollouts per sampled position Si i-th chunk of the input sequence D Training data distribution

Model and Policy πθ Policy model with parameters θ πθ

Old policy (before update) p(xt | x<t) Conditional probability of token xt given context hpred(t) Hidden state vector of predicted token at position t hgt(t) Hidden state vector of ground-truth token at position t hpredk (t), hgtk(t) Matrix of k hidden state vectors starting at position t + 1 (∈ Rk×d) d Hidden state dimension

old

Entropy and Sampling Ht Entropy at token position t T ∗ = {t1,...,tc} Set of sampled high-entropy positions Ti Set of token positions in chunk i

pi(t) Entropy-weighted sampling probability in chunk i for token position t τ Temperature parameter for entropy-based sampling

Loss Functions LNTP Next-Token Prediction loss (cross-entropy) LNSP Next-Sequence Prediction loss Lseq Sequence-level discrepancy measure J (θ) RL objective to maximize

Table A.2. Glossary and notation (continued).

Symbol Description Rewards R(t) Self-supervised reward at position t (shorthand for R(ˆxt+1:t+k,xt+1:t+k)) Rkφ(t) Cosine similarity reward over k-token rollout Rkbinary(t) Binary exact match reward over k-token rollout Rkhybrid(t) Hybrid reward (Rkφ + Rkbinary) RT ∗ Set of rewards for all sampled positions in T ∗ cos(·,·) Cosine similarity function I[·] Indicator function oT ∗ Set of all rollouts at sampled positions Fast Weight Architecture Wt Fast weight matrix at token position t η Learning rate for fast weight update kt,vt,qt Key, value, query vector representations at position t

### B. Related Work

Multi-Token Prediction. Predicting more than one token as a training objective has been explored in previous studies. Gloeckle et al. (2024) tackled this problem by estimating k-tokens in parallel using k independent output heads. This approach achieves significant gains in throughput by applying architectural modifications to the language model, with minimal degradation of performance on downstream tasks. However, this approach is limited in capturing dependencies among the predicted tokens and relies on a fixed prediction horizon k. REFINE computes rewards based on each multi-token rollout as a whole, capturing the semantic connection among them.

Liu et al. (2025) uses diffusion-based generation to predict multiple masked tokens simultaneously and optimizes the cross-entropy loss between the predicted tokens and masked ground truth tokens. Their method is primarily designed for standard attention-based transformer architectures that predict the next token with masking. However, REFINE is designed to train fast weight language models that store information and update their parameters with a fundamentally different set of rules. We view this work as a valuable source of motivation experimented on a different class of models.

Continued Pre-Training with RL. Studies have explored using RL as a tool for the next-token prediction objective. Dong et al. (2025) samples reasoning traces before next-token prediction and assigns rewards based on the similarity of the bytesequences of predicted and ground truth tokens. Hatamizadeh et al. (2025) takes a similar approach by sampling reasoning traces for next-token prediction but assigning rewards by measuring the gap between the log-likelihood of the ground truth token with and without the reasoning trace as context. However, these works focus on attention-based transformer models (DeepSeek-R1-Distill-Qwen-14B (Guo et al., 2025) and Qwen3-1.7B-Base (Team, 2025), respectively) and assume basic reasoning capabilities that enable exploration with Chain-of-Thought. Whether RL can be used for pre-trained fast weight models without prior instruction tuning or human preference optimization, especially in long-context settings, is yet to be explored. Further, REFINE leverages RL for training on the NSP objective which optimizes sequence-level predictions rather than single-token predictions under the NTP objective.

TTT in Language Models. Recent work has explored training standard transformer-based language models during inference to improve performance on target tasks. These methods usually involve extracting task-related learning signals from the model itself for offline adaptation. Aky¨urek et al. (2024) generates relevant in-context examples from the given task and trains the model on those examples before generating the final answer to the actual task. RL-based methods extract

pseudo-labels by aggregating multiple responses to the same task and assigning rewards to each response based on their similarity to the pseudo-label (Zuo et al., 2025; Huang et al., 2024). Recently, context-based test-time training has also been explored in transformer-based language models. In an attempt to overcome the inherent limitations of static attention, Bansal et al. (2025) executes gradient updates on the query projection matrices using the context while keeping other parameters frozen. These approaches apply task-aware or task-agnostic TTT methods to standard transformer-based language models. TTT with REFINE, on the other hand, aims to improve contextual adaptation and memory of fast weights for long contexts, which is a novel setting that has not yet been explored.

Efficient Attention Variants. Standard transformer-based language models with full attention incur quadratic computational complexity as a function of context length. Recent work has developed techniques to reduce the computational and memory overhead in long-context modeling. Sparse attention addresses this problem by introducing computational sparsity in the original attention mechanism. Grouped Query Attention (Ainslie et al., 2023) employs sparsity along the head dimension to assign each query head to different groups that share a single key-value head. Sliding window attention (Child, 2019; Beltagy et al., 2020; Zaheer et al., 2020) architectures leverage sparsity along the context by computing local attention on a fixed number of contiguous tokens.

Linear attention approximates the softmax kernel in the attention formulation to achieve linear computational complexity. Linformer (Wang et al., 2020) replaces the attention mechanism with low rank matrix operations, which has been shown to be effective for sequence processing but limited in autoregressive generation. Performer (Choromanski et al., 2020) uses orthogonal random features to approximate softmax kernels in the attention, achieving linear complexity without employing low rank matrices. Similarly, Linear Transformer (Katharopoulos et al., 2020) approximates the softmax with linear dot-product of kernel feature maps. The success of linear attention has motivated the development of architectures that operate on linear computational complexity by design, including State Space Models (SSMs), such as Mamba (Gu & Dao, 2024; Dao & Gu, 2024). Unlike these attention variants that aim to approximate full attention, fast weight models rely on fixed-size memory with predefined online update rules to directly store contextual information in the parameters. We therefore propose REFINE as a training framework targeting fast weight models that are fundamentally different in terms of architecture compared to attention-based transformer models.

### C. Datasets and Benchmarks

We summarize the datasets and benchmarks used for training and evaluation in Tab. C.1. The Long-Data-Collections (TogetherAI, 2024) dataset contains a 68.8B-token pre-training corpus subsampled from RedPajama (Weber et al., 2024), Pile (Gao et al., 2020), UL2 Oscar (Tay et al., 2022), NI (Mishra et al., 2022), and P3 (Bach et al., 2022). We use a 200M-token subset of Long-Data-Collections for mid-training. For LongBench (Bai et al., 2024), we select 12 subtasks that are English-based. We leave out MuSiQue and GovReport tasks as they have fewer than 20 samples under 16K tokens.

### D. Training Configuration

Hyperparameters. We provide the training hyperparameters used for REFINE during all training phases in Tab. D.1. We only adjust the train batch size, reward function, and RL loss coefficient across training phases, while keeping all else equal.

Compute. We use 8 L40 GPUs for mid-training and post-training, and 4 L40 GPUs for TTT. We use fewer GPUs for TTT because the train batch size for TTT is smaller (8 samples per batch) compared to other training phases (128 for mid-training, 64 for post-training). Mid-training LaCT-760M and DeltaNet-1.3B with REFINE on 200M tokens from Long-Data-Collections (TogetherAI, 2024) at 16K context takes approximately 24 hours.

### E. Additional Analysis

Validation Loss. We report the validation loss on the Booksum (Kry´sci´nski et al., 2022) dataset in Long-DataCollections (TogetherAI, 2024) during mid-training. The validation loss for SFT on LaCT stays constant as the mid-training data is the same as its pre-training data. However, we see a notable decrease in validation loss with REFINE (Fig. E.1), indicating that NSP provides learning signal that is unique from standard NTP training.

Table C.1. Summary of datasets and benchmarks used across training phases. Phase Dataset Metric Context Size

Long-Data-Collections - 16K ∼200M tokens RULER NIAH recall 4K/8K/16K 500 per context Booksum NTP Accuracy, CE loss ≤16K 9600

Mid-training

RULER SQuADQA recall 4K/8K/16K 1600 train / 200 test RULER HotpotQA recall 4K/8K/16K 1600 train / 200 test

Post-training

NarrativeQA (NQ) F1 ≤16K 56 Qasper (QR) F1 ≤16K 184 MultiFieldQA (MF) F1 ≤16K 136 HotpotQA (HP) F1 ≤16K 96 2WikiMHQA (2W) F1 ≤16K 184 QMSum (QM) rouge ≤16K 104 MultiNews (MN) rouge ≤16K 192 SAMSum (SS) rouge ≤16K 152 TREC (TC) accuracy ≤16K 200 TriviaQA (TQ) accuracy ≤16K 120 LCC (LC) code similarity ≤16K 488 RepoBench-P (RP) code similarity ≤16K 320

Test-time

PIQA accuracy All HellaSwag(Hella.) normalized accuracy All WinoGrande(Wino.) accuracy All ARC-e accuracy All ARC-c normalized accuracy All Wikitext(Wiki.) perplexity All LAMBADA(LMB.) perplexity, accuracy All FDA recall All SWDE recall All

Commonsense

(b) LaCT-760M

(a) DeltaNet-1.3B

SFT

SFT

ReFINE

ReFINE

3.04

14.20

ValidationLoss

ValidationLoss

3.02

14.15

3.00

14.10

2.98

14.05

2.96

0 20 40 60 80 100

0 20 40 60 80 100

Steps

Steps

Figure E.1. Validation Loss on Booksum Dataset. We track the NTP loss on the Booksum validation dataset in Long-Data-Collections (TogetherAI, 2024) throughout mid-training on DeltaNet-1.3B (a) and LaCT-760M (b). The validation loss for SFT on LaCT-760M does not decrease, as the model has already been pre-trained on the mid-training dataset. The error bars show the minimum and maximum values from three independent trials.

Entropy Distribution. We report the NTP entropy distribution of a randomly selected sample in order to illustrate the effects of entropy-based token selection in REFINE (Fig. E.2). We find that there are no index-dependent patterns in the distribution, which justifies per-chunk target token sampling weighted by entropy.

Reward Distribution. In order to investigate the stability of mid-training with REFINE, we track the cosine similarity reward across training steps for different rollout lengths (k = 3,5,7), as shown in Fig. E.3. We find that the reward distribution remains stable throughout training. However, as rollout length increases, both the mean and variance of the

Table D.1. Training hyperparameters.

Parameter Value Parameter Value Actor gradient clip 0.2 Learning rate 10−6 Mid-Train batch size 128 Adam (β1, β2) (0.9, 0.999) Post-Train batch size 64 Weight decay 0.01 Test-Time-Train batch size 8 Sampling temperature τ 1.0 Mid-Train PPO mini-batch size 32 Max prompt length 16,384 Post-Train PPO mini-batch size 16 Entropy loss coefficient 0 Test-Time-Train PPO mini-batch size 4 KL loss coefficient 0

Mid-Train λRL 0.2 n (rollouts / position) 1 Post-Train λRL 0.2 k (rollout length) 5 Test-Time-Train λRL 0.4 c (chunks / sequence) 8

Mid-Train λSFT 1.0 Mid-Train reward Rφ Post-Train λSFT 1.0 Post-Train reward Rhybrid Test-Time-Train λSFT 1.0 Test-Time-Train reward Rbinary

Entropy

(a) LaCT-760M

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

4.0

3.0

2.0

1.0

0.0

-1.0

0k 4k 8k 12k 16k

Token Index

(b) DeltaNet-1.3B

5.0

4.0

Entropy

3.0

2.0

1.0

0.0

-1.0

0k 4k 8k 12k 16k

Token Index

- Figure E.2. NTP Entropy Distribution. We compute the token-level NTP entropy of a randomly selected sample using LaCT-760M (a) and DeltaNet-1.3B (b). We use the same sample to extract the entropy distribution from both models.

reward decrease, suggesting that the learning signal may lose sharpness for larger k.

Performance on Short-Context Tasks. We investigate whether enhancing long-context handling capabilities of fast weight models leads to degradation of performance on out-of-distribution tasks such as short-context in-context retrieval and commonsense reasoning. We evaluate mid-trained models on 9 relevant short-context tasks with lm-evaluation-harness (Gao et al., 2024): PIQA (Bisk et al., 2020), HellaSwag (Hella.) (Zellers et al., 2019), WinoGrande (Wino.) (Sakaguchi et al., 2020), ARC-easy (ARC-e), ARC-challenge (ARC-c) (Clark et al., 2018), Wikitext (Wiki.) (Merity et al., 2016), LAMBADA (LMB.) (Paperno et al., 2016), FDA (Arora et al., 2023), and SWDE (Lockard et al., 2019). Tab. E.1 shows that REFINE sustains performance in these tasks, suggesting that NSP complements NTP without inducing catastrophic forgetting (van de Ven et al., 2024).

Impact of Reward Functions on TTT. REFINE uses binary exact match reward Rbinary during test-time for offline adaptation of the model before generating the response. We repeat TTT with cosine similarity reward instead and report the average score on LongBench tasks in Tab. 5. Tab. E.2 shows that the binary reward is optimal for TTT, but the cosine similarity reward also performs well, higher than pure SFT for TTT.

RewardMean

(a) LaCT-760M (Mean)

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | |k=3 k=5<br><br>| | | |
| | | |k=7| | | |
| | | | | | | |

0.62

0.60

0.58

0.56

0.54

0.52

0 20 40 60 80 100

Steps

RewardMean

(b) DeltaNet-1.3B (Mean)

0.62

| | | | | | | |
|---|---|---|---|---|---|---|
| |k=3 k=5 k=7<br><br>| | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

0.60

0.58

0.56

0.54

0 20 40 60 80 100

Steps

RewardStd

(c) LaCT-760M (Std)

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | |k= k=<br><br>|3 5|
| | | | | |k=|7|
| | | | | | | |

0.27

0.26

0.25

0.24

0.23

0 20 40 60 80 100

Steps

RewardStd

(d) DeltaNet-1.3B (Std)

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| |k=3 k=5<br><br>| | | | | |
| |k=7| | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

0.21

0.20

0.19

0.18

0.17

0 20 40 60 80 100

Steps

- Figure E.3. Reward Distribution. We report the mean and standard deviation of the cosine similarity reward during mid-training for different values of k. As the rollout length increases, the mean reward (a, b) decreases and the standard deviation (c, d) also decreases.

- Table E.1. Performance on Short-Context Tasks. We evaluate mid-trained models on short-context benchmarks to verify that REFINE does not cause catastrophic forgetting.

Models Wiki LMB. FDA SWDE LMB. PIQA Hella. Wino. ARC-e ARC-c Avg

ppl ↓ ppl ↓ recall ↑ recall ↑ acc ↑ acc ↑ acc n ↑ acc ↑ acc ↑ acc n ↑

LaCT-760M 20.8 29.8 36.7 66.0 32.4 67.5 41.5 53.4 48.4 27.6 45.1 +SFT MidTr 20.7 28.9 37.2 66.2 32.9 67.4 41.4 52.5 48.4 27.1 45.0 +REFINE MidTr 20.8 30.2 36.1 66.5 32.3 67.3 41.4 52.8 48.8 27.5 45.0

DeltaNet-1.3B 16.7 14.7 18.0 54.4 43.0 70.8 50.5 53.7 58.3 25.9 50.4 +SFT MidTr 16.8 15.3 17.7 54.5 42.7 71.1 50.1 53.8 58.4 26.4 50.4 +REFINE MidTr 16.8 15.6 17.5 54.5 42.2 71.1 50.2 53.6 58.1 26.2 50.2

- Table E.2. Impact of reward function on TTT. We compare different TTT reward strategies on 12 tasks in LongBench: binary exact match between predicted and ground truth completion (Rbinary), and cosine similarity of the hidden states of predicted and ground truth completions (Rφ).

MidTr TTT MidTr Reward TTT Reward Avg. Score

REFINE - Rφ - 16.9 REFINE SFT Rφ - 17.0 REFINE REFINE Rφ Rφ 17.5 REFINE REFINE Rφ Rbinary 18.0

LaCT-760M

REFINE - Rφ - 17.0 REFINE SFT Rφ - 16.5 REFINE REFINE Rφ Rφ 17.6 REFINE REFINE Rφ Rbinary 17.9

DeltaNet-1.3B

### F. Qualitative Examples

We provide qualitative examples of cosine similarity reward assignment during mid-training in Tab. F.1. We randomly sample prefixes from the mid-training dataset and generate four k = 5 continuations each using the pre-trained LaCT-760M model. The cosine similarity reward is designed to capture the semantic similarity between the predicted and ground truth continuations. The highest reward values for each example are highlighted in bold. The examples demonstrate that the reward effectively captures semantic similarity beyond exact lexical matching. For instance, in example 2, “enjoyed every minute of it” achieves near-perfect alignment (0.961) with “loved every minute of it”, while semantically divergent predictions like “would not recommend it at” receive much lower scores (0.463). Similarly, example 6 shows sensitivity to mathematical concepts, where “is a convergent integral” receives a high score (0.758) for preserving the convergence concept from “is also convergent according”, while less relevant predictions such as “shall not be less than” receive lower rewards (0.512). These examples illustrate that the cosine similarity reward provides meaningful learning signals for training the model to generate semantically coherent continuations.

Table F.1. Qualitative examples of cosine similarity reward assignment during mid-training. GT denotes the ground truth continuation, and P1–P4 denote the four predicted continuations generated by the model.

Example Type Text Cosine Reward

- Ex. 1

GT very laid back lodge –

- P1 perfect hill 0.508
- P2 unique hotel to spend a 0.498
- P3 great place to stay 0.535

- P4 top-notch l 0.418

- Ex. 2

GT loved every minute of it –

- P1 would not recommend it at 0.463
- P2 were so impressed by this 0.566
- P3 enjoyed every minute of it 0.961

- P4 skied daily. Excell 0.461

- Ex. 3

GT screen washer fluid are –

- P1 screen washer software are 0.922

- P2 screen wipers are changed 0.641
- P3 screen washing and dipping 0.555
- P4 screen washer fluids 0.824

- Ex. 4

GT bound on the predictive –

- P1 bound on the trace of 0.840

- P2 -ranked trace- 0.535
- P3 bound on 0.594
- P4 bound on P( 0.582

- Ex. 5

GT continues to grow; and –

- P1 reflects wave after wave of 0.484
- P2 stretches into space. 0.531

- P3 heats ocean waters a 0.461
- P4 cushions us all from 0.500

- Ex. 6

GT is also convergent according –

- P1 is a convergent integral 0.758

- P2 shall not be less than 0.512
- P3 must have a value, 0.574
- P4 is convergent. Because 0.578

