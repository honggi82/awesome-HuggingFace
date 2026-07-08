# arXiv:2605.09959v1[cs.LG]11May2026

## G-Zero: Self-Play for Open-Ended Generation from Zero Data

Chengsong Huang1, Haolin Liu2, Tong Zheng3, Runpeng Dai4, Langlin Huang1, Jinyuan Li1, Zongxia Li3, Zhepei Wei2, Yu Meng2, Jiaxin Huang1

1Washington University in St. Louis 2University of Virginia 3University of Maryland 4University of North Carolina at Chapel Hill

{chengsong,jiaxinh}@wustl.edu

### Abstract

Self-evolving LLMs excel in verifiable domains but struggle in open-ended tasks, where reliance on proxy LLM judges introduces capability bottlenecks and reward hacking. To overcome this, we introduce G-Zero, a verifier-free, co-evolutionary framework for autonomous self-improvement. Our core innovation is Hint-δ, an intrinsic reward that quantifies the predictive shift between a Generator model’s unassisted response and its response conditioned on a self-generated hint. Using this signal, a Proposer model is trained via GRPO to continuously target the Generator’s blind spots by synthesizing challenging queries and informative hints. The Generator is concurrently optimized via DPO to internalize these hint-guided improvements. Theoretically, we prove a best-iterate suboptimality guarantee for an idealized standard-DPO version of G-Zero, provided that the Proposer induces sufficient exploration coverage and the data filteration keeps pseudo-label score noise low. By deriving supervision entirely from internal distributional dynamics, G-Zero bypasses the capability ceilings of external judges, providing a scalable, robust pathway for continuous LLM self-evolution across unverifiable domains.

Code: https://github.com/Chengsong-Huang/G-Zero

### 1 Introduction

Self-evolving Large Language Models (LLMs) have emerged as a promising path beyond the limits of human-curated supervision. Rather than relying on static datasets, these models autonomously generate, refine, and learn from their own outputs, offering a scalable route to capabilities that exceed what human imitation alone can provide [27, 26, 31]. This potential has been most clearly demonstrated in reasoning-intensive tasks with strictly verifiable outcomes. In these settings, prior work [37, 10, 16] shows that models can discover complex problem-solving strategies through self-play, continuously improving toward expert-level performance.

However, this paradigm relies crucially on the existence of programmatic oracles. In domains like mathematics or code generation, deterministic signals, such as numerical correctness or functional execution, provide the ground truth required for Reinforcement Learning from Verifiable Rewards (RLVR) [23, 7]. Conversely, a broad class of real-world scenarios, including open-ended instruction following [24], multi-turn dialogue [34], and creative writing, lack such objective oracles. To navigate these settings, existing methods frequently rely on LLM-as-a-judge [6] mechanisms for surrogate reward signals. This workflow introduces two critical limitations. First, the evolving model’s performance ceiling is fundamentally bottlenecked by the judge’s capabilities. Second, the optimization process is highly vulnerable to reward hacking [28]; rather than genuinely improving response quality, the model learns to exploit the judge’s stylistic vulnerabilities, such as bias, formatting preferences, or verbosity. This raises a crucial question: How can self-evolution scale in unverifiable domains without internalizing these pathologies?

Preprint.

[Figure 1]

- Figure 1: Comparison of self-supervision signals. R-Zero [10] uses majority voting, restricting it to verifiable closed-domain tasks. LLM-as-a-Judge assigns scalar scores, bounded by the judge’s capability. In contrast, G-Zero creates an internal preference signal by preferring hint-conditioned responses over unassisted ones, eliminating the need for external verifiers or judges.

To move self-evolution beyond verifiable tasks and avoid the flaws of proxy LLM judges, we introduce G-Zero, a verifier-free, co-evolutionary framework that derives supervision entirely from internal dynamics. G-Zero operates through the interaction of two separate models: a Proposer and a Generator. The core innovation of G-Zero is our designed intrinsic signal called Hint-δ, which measures how much a hint shifts the Generator’s predictive distribution over its own unassisted response without the hint. Hint-δ measures a cognitive gap by coupling two objectives into one scalar: it can be large only when the underlying query is challenging for the Generator and the hint carries necessary information or reasoning that the Generator does not already possess. Using this intrinsic signal, the Proposer is trained via GRPO to synthesize challenging queries paired with informative hints, while the Generator is concurrently optimized via DPO to internalize these hintguided improvements. Specifically, the Generator learns to favor the hint-guided response (the chosen output) over its initial, unassisted answer (the rejected output). The two models co-evolve through iterative rounds. This design directly addresses the two problems of judge-based self-evolution. Because Hint-δ is computed entirely from the Generator’s own log-probabilities under matched contexts, the difficulty ceiling automatically improves with the Generator’s capabilities.

We theoretically and empirically validate the G-Zero framework. Theoretically, we formalize the co-evolutionary loop and prove a best-iterate suboptimality guarantee for an idealized standardDPO variant, under sufficient Proposer-induced coverage, and low δ-certified pseudo-label score noise. Empirically, G-Zero demonstrates robust improvements within several self-play iterations, and achieves substantial gains on both open-ended (e.g., +3.74 points on AlpacaEval) and verifiable (e.g., +5.21 points in AIME 25) tasks across diverse model families (Qwen and Llama). Further analysis shows that the model’s substantial reasoning improvements do not stem from domainspecific memorization, but from internalizing logical depth in open-ended, non-verifiable tasks, which surprisingly transfers to rigorous domains like mathematical problem-solving.

In summary, our main contributions are:

- • A Verifier-Free Co-Evolutionary Framework: We propose G-Zero, a self-play pipeline that drives continuous self-evolution through hint-induced response shifts in open-ended domains without external verifiers.
- • Theoretical Characterization of Intrinsic Self-Play: We formalize the co-evolutionary loop and prove a best-iterate suboptimality guarantee for an idealized standard-DPO variant of G-Zero, with the bound controlled by Proposer-induced coverage, and δ-certified pseudo-label score noise.
- • Empirical Improvements on Both Open-Ended and Verifiable Domains: We demonstrate that G-Zero brings substantial improvements on instruction-following, chatting, and reasoning tasks across different model families, and also successfully internalize logical depth from open-ended, non-verifiable tasks to rigorous domains like mathematical problem-solving.

[Figure 2]

- Figure 2: The G-Zero co-evolutionary loop. Top (Proposer training): The Proposer πP generates query–hint pairs {(qi,hi)}. The frozen Generator πG produces unassisted responses, and Hint-δ is computed from the log-probability shift each hint induces on the Generator’s distribution. The δ values serve as the GRPO reward, driving πP to explore the Generator’s blind spots. Bottom (Generator training): With πP frozen, the Generator answers each query with and without the hint; We then filter the resulting pairs into a preference dataset DR+1, where R is the round index, on which πG is updated via DPO.

### 2 Preliminaries: Optimization Objectives

Direct Preference Optimization (DPO). Direct Preference Optimization [21] aligns a language model policy πθ with preferences without requiring a separate reward model. Given a dataset D of preference triples (x,yw,yl), where x represents the input prompt, yw is the preferred (chosen) response, and yl is the rejected response, DPO optimizes the policy against a frozen reference model πref by minimizing the following loss:

πθ(yw | x) πref(yw | x) − β log

πθ(yl | x) πref(yl | x)

. (1)

LDPO(θ) = −E(x,y

w,yl)∼D log σ β log

where σ is the logistic function and β is a hyperparameter that controls the deviation from the reference policy.

Group Relative Policy Optimization (GRPO). Group Relative Policy Optimization [23] is an efficient reinforcement learning algorithm that omits the need for an external value model. For a given context c sampled from a dataset P, the policy πθ samples a group of K outputs {o1,...,oK} (we use K rather than G to avoid clashing with the Generator subscript πG). The policy is updated by maximizing the following clipped objective, where ϵ ∈ (0,1) is the PPO clip range:

K

πθ(oi | c) πold(oi | c)

1 K

JGRPO(θ) = Ec∼P,{o

min

Ai,

i}Ki=1∼πold

i=1

(2)

πθ(oi | c) πold(oi | c)

clip

,1 − ϵ,1 + ϵ Ai .

Following prior work [20], we omit the KL divergence penalty in our formulation. The advantage Ai is computed by standardizing the scalar rewards r(oi) within the sampled group: Ai = (r(oi)−µr)/σr,

where µr = K1 Kj=1 r(oj) and σr = K1 Kj=1(r(oj) − µr)2.

### 3 The G-Zero Framework

G-Zero is an iterative, co-evolutionary self-play framework designed for continuous LLM selfimprovement. Instead of relying on external verifiers or inherently verifiable tasks, we construct preference pairs directly by contrasting the model’s unassisted responses against those conditioned on intrinsic hints.

As illustrated in Figure 2, a single training round consists of two interacting phases: (1) Proposer Training (§3.2): The Proposer is trained using Hint-δ (defined in §3.1) to identify challenging queries and pair them with informative hints. (2) Dataset Curation and Generator Training (§3.3): Hint-δ is repurposed as a quality filter to curate well-suited response pairs. The Generator is then updated via DPO to favor the hint-guided responses over their unassisted baselines.

Through this iterative process, the Generator absorbs the structural and stylistic patterns elicited by the hints, and therefore learns to produce higher-quality independent responses. The improved model then serves as the base for the next round, enabling continuous self-evolution.

#### 3.1 The Intrinsic Learning Signal: Hint-δ

G-Zero is fundamentally driven by a single intrinsic learning signal, Hint-δ. Let πG denote the Generator LLM under training and πP the Proposer model used to explore the open-ended task space. For a given query q and a proposed hint h, let ahard ∼ πG(· | q) be the baseline response generated by the Generator without the hint, with token sequence ahard = (a1,...,aT). The Hint-δ signal measures how much the hint shifts the Generator’s predictive distribution over its own unassisted response, evaluated as a per-token mean log-likelihood difference:

δ(q,h,ahard) =

T

1 T

t=1

log πG(at | q,a<t) − log πG(at | q,h,a<t) . (3)

We deliberately use the per-token mean rather than the sequence-level sum so that δ is invariant to the length of ahard: the Proposer cannot trivially inflate its reward by eliciting longer unassisted responses. Empirically, on our 1,824-sample R1 raw pool we measure a Spearman rank correlation of −0.41 between δ and the character length of ahard, i.e., longer ahard tend to receive smaller δ, which is consistent with the per-token normalization removing the naive length bias.

- A key advantage of this formulation is that Hint-δ effectively captures both query difficulty and hint informativeness at once, yielding a large δ only when two conditions are jointly met: (i) the underlying query is genuinely challenging for the Generator, so that the unassisted response is flawed or uncertain, and (ii) the hint carries missing knowledge or reasoning steps needed by the Generator to largely reshape the response distribution. Either factor alone is insufficient: If the query is trivial,

the hints tend to be redundant with πG’s prior knowledge, leaving the log-probability unchanged (δ ≈ 0). Symmetrically, for a difficult query, if the hint is uninformative, it fails to perturb the distribution. Consequently, maximizing δ drives the Proposer to jointly search over query difficulty and hint informativeness, automatically targeting the Generator’s blind spots without any external difficulty signal. Crucially, the Proposer’s reward is computed against the current Generator, so as πG improves, the threshold for what counts as an “informative hint” rises with it. The two models therefore co-evolve across rounds.

#### 3.2 Proposer Training

The objective of this phase is to train the Proposer πP to actively propose challenging queries paired with informative hints that elicit a significant, constructive response shift in the Generator.

We design a specific system prompt that instructs πP to jointly generate a query q and a corresponding hint h, enforcing a strict structural format utilizing <question> and <hint> XML tags. The Proposer

is optimized via GRPO using Eq. (2), where the output oi corresponds to the generated pair (qi,hi). We use the Hint-δ signal in Eq. (3) as our intrinsic reward.

However, optimizing purely for δ introduces vulnerabilities. A naive Proposer might learn to generate excessively verbose text to artificially shift the Generator’s distribution. To prevent this reward hacking, we introduce a Length Penalty

Plength = λ · max 0, |h|−100200 , (4)

where |h| is the hint length in characters and λ = 0.03 is used in all reported runs, penalizing hints that exceed a reasonable budget of ∼200 characters. Furthermore, to prevent the Proposer from collapsing into generating repetitive pairs, we apply a BLEU Duplication Penalty (PBLEU). We agglomeratively cluster all generated questions in the step’s batch using sentence-BLEU distance

with average-linkage and a merge threshold of 0.5 (i.e., questions whose pairwise BLEU exceeds 0.5 are merged into a cluster). For each rollout we set PBLEU = |Ci|/|B|, the fraction of the step’s batch

- B that lies in the rollout’s own cluster Ci: a unique question receives a small ∼1/|B| penalty, while a question shared by many rollouts is heavily discounted. The total reward combines the intrinsic δ signal with the penalties:

r(q,h) = δ(q,h,ahard) − Plength − PBLEU (5)

For formatted error pairs (e.g., missing mandatory XML blocks or empty fields), we apply a hardcoded penalty floor of −1 and skip the δ computation entirely to save computation, while still applying the duplication penalty to punish repeated formatting failures.

#### 3.3 Generator Training and Dataset Curation

In this final phase, we train the Generator πG on a curated preference dataset DR+1 using the DPO loss (Eq. (1)), with the hint-assisted response aassisted as the chosen sample (yw) and the unassisted response ahard as the rejected sample (yl). The reference model πref is initialized as a frozen snapshot of πG taken at the start of the round, anchoring DPO updates to a stable behavioral baseline. To neutralize the well-known length bias of vanilla DPO, in which longer chosen responses contribute disproportionately to the gradient regardless of content, we adopt a length-normalized variant that replaces the sequence-summed log-ratio in Eq. (1) with its per-token mean:

πθ(y | x) πref(y | x)

1 |y|

LLNDPO(θ) = −E(x,y

w,yl)∼DR+1 log σ β r ¯θ(x,yw) − r¯θ(x,yl) , r¯θ(x,y) =

log

.

(6) We adopt DPO rather than online RL with a learned reward model for two reasons. First, our preference pairs are constructed from the same model’s output distribution under matched contexts, and DPO’s closed-form, reference-anchored objective is a natural fit for this self-paired setting. Second, Hint-δ already provides an explicit chosen/rejected signal at the pair level; routing this signal through a separately trained reward model would introduce an additional information bottleneck and approximation error without any clear benefit.

The objective of this DPO training is hint internalization. By training on these pairs, the Generator is incentivized to favor the structural and stylistic patterns present in the hint-guided response, including more deliberate decomposition of the problem and more disciplined use of intermediate steps. As a result, the model tends to reproduce this higher-quality content independently, without requiring the explicit hint from the Proposer at inference time. This enables the Generator to perform substantially better on complex tasks during inference when no external assistance is available.

Training Set Curation. To maximize the efficacy of the DPO phase, we impose stringent filtering criteria on the preference pairs comprising DR+1. The Proposer’s GRPO training, by maximizing Hint-δ, has already performed a first stage of selection: the (q,h) pairs it produces concentrate on hard queries equipped with informative hints. Our data curation performs a complementary selection on top of this pool to evaluate whether each pair is well-suited for DPO.

For each query-hint pair (q,h) ∼ πP, we sample the Generator’s dual responses: the unassisted baseline ahard ∼ πG(· | q) and the hint-conditional response aassisted ∼ πG(· | q,h). We then recompute the δ score on these freshly sampled responses (Eq. (3)) and retain only pairs whose δ falls in the lower half of the empirical distribution within each round.

While the Proposer targets the Generator’s blind spots by maximizing δ, we apply a contrasting filtering strategy for DPO data curation. In this stage, δ functions as a proxy for the distributional distance between aassisted and ahard. Within the generated pool of complex queries, explicitly retaining preference pairs with relatively lower δ is essential, driven by two fundamental reasons:

Lower-δ pairs serve as hard-to-distinguish training signals. In preference learning, training on pairs with a massive quality gap often yields diminishing returns, as the preference is trivially satisfied. A lower δ indicates that the log-probability shift between the chosen (aassisted) and rejected (ahard) responses is relatively minor. Consequently, these constitute hard-to-distinguish preference pairs. By focusing on these pairs where the reward gap is small, DPO is forced to learn fine-grained, structural improvements in reasoning rather than relying on superficial, easy-to-spot differences.

High-δ pairs violate DPO’s implicit KL-divergence constraint. The DPO formulation inherently includes a KL-divergence penalty against the reference model πref. A very high δ implies that the hintassisted response aassisted is drastically far away from the Generator’s original unassisted distribution. Pushing the policy towards such completely out-of-distribution responses severely violates this implicit KL constraint. This can lead to excessively large gradients, off-manifold drift, and severe training instability. By filtering out the top half of the δ distribution, we naturally regularize the optimization process and ensure aassisted remains a plausible trajectory for the Generator to internalize. A subtle point concerns the very bottom of the retained band. Pairs with near-zero δ have low implicit-reward margin under πG, since the Generator assigns similar log-probabilities to aassisted and ahard. We deliberately keep these pairs rather than excluding them so that the lower-half filter is defined purely by the ranked δ statistic without an additional minimum-margin cutoff: the bulk of the constructive learning signal in DR+1 is carried by the middle of the lower-half band, while the tail near zero adds only a small amount of low-margin label noise on responses that are independently sampled at temperature 0.7 and therefore lexically distinct.

#### 3.4 Theoretical Analysis

We analyze G-ZERO as an iterative δ-certified exploratory DPO procedure. We consider a simple linear case such that there exists a ground truth reward R⋆(q,a) = ϕ(q,a)⊤θ⋆ for any question q and response a, where ϕ is a feature and θ⋆ ∈ Rd is a hidden reward parameter. We assume the standard Bradley-Terry model [21] such that P(a+ ≻ a−|q) = σ (R⋆(q,a+) − R⋆(q,a−)). The performance of the generator has the following guarantee.

Theorem 1. (Informal) Suppose the game collects retained data from the Proposer such that, after δ-filtering, the data are sufficiently exploratory, and the Generator is updated iteratively by DPO on the cumulative retained data for T rounds. Then, with high probability, there exists an iterate t0 ≤ T such that, using a total number of retained samples O(d2/ε2), the Generator’s policy πt

satisfies J(π⋆) − J(πt

0

) ≤ O(ε + √ηδ).

0

where O omits log factors, J(π) = Eq∼Q,a∼π(·|q)[R⋆(q,a)] − Eq∼Q [DKL(π(· | q)∥πref(· | q))], Q is the target question distribution, π⋆ = argmaxπ J(π) and ηδ denotes the self-normalized cumulative score noise induced by incorrect pseudo-labels after δ filtration.

The theorem separates the two intrinsic signals in G-ZERO. The Hint-δ and filter controls data quality: if retained pairs are calibrated so that aassisted is truly better than ahard with high probability, then the pseudo-label noise ηδ is small. The exploration reward (implemented by BLEU) for the challenger controls data coverage: it drives the challenger toward pairwise feature directions that the generator has not yet learned. Together, these two effects imply an iterative co-evolution guarantee: the Challenger supplies preference pairs that are both reliable enough to trust and novel enough to teach, while cumulative DPO distills them into the Generator. Detailed proof of Theorem 1 is provided in Appendix D.

### 4 Experiments

#### 4.1 Experimental Setup

Models. To evaluate the generalization capabilities of our proposed method, we evaluate Qwen38B-Base [33] and Llama-3.1-8B-Instruct [5]. By testing on both a foundational base model and an instruction-tuned model from distinct, widely adopted families, we demonstrate that our approach is robust to architectural variations and effective regardless of prior alignment stages.

Benchmarks and Evaluation. To evaluate reasoning capabilities, we benchmark on AIME24 and AIME25, reporting the overall mean@32 score from 32 independent responses sampled at a temperature of 0.7. To evaluate instruction-following, we use IFEval [38] with greedy decoding, reporting the four standard metrics (prompt/instruction-level strict and loose accuracies). Lastly, to assess general conversational quality, we report the length-controlled win rate on AlpacaEval 2.0 [3] against GPT-4-Turbo, judged by the Qwen3-235B-A22B-Instruct-2507.

Table 1: Main results. Absolute performance (%) of G-Zero after round 1 and round 2.

Chat IFEval Math

Average AlpLC IF-pS IF-pL IF-iS IF-iL AIME24 AIME25

###### Qwen3-8B-Base

base model 8.94 43.07 50.28 56.00 61.75 10.42 7.19 33.95 R-Zero 8.04 37.56 42.79 54.12 57.37 14.92 13.34 32.59

- G-Zero (round 1) 9.07 42.33 50.83 56.24 62.71 12.19 11.36 34.96
- G-Zero (round 2) 8.47 43.81 50.83 57.92 63.43 11.15 12.40 35.43

###### Llama-3.1-8B-Instruct

base model 24.12 58.41 65.80 69.42 75.30 5.94 0.42 42.77 R-Zero 21.74 56.94 62.47 66.74 72.93 5.07 0.34 40.89

- G-Zero (round 1) 23.88 58.59 66.35 69.78 75.54 6.35 1.04 43.08
- G-Zero (round 2) 27.86 59.52 66.35 70.38 75.78 6.77 0.63 43.90

Experiment Configuration. We strictly standardize the hyperparameter settings across the iterative loop. All model training in our experiments is conducted via the Tinker API 1, exclusively utilizing Low-Rank Adaptation (LoRA) [9].

We supplement the δ-based filter with a set of lightweight heuristic checks on the chosen response aassisted to remove pairs that are known to induce DPO artifacts, following standard practice in DPO data curation. Specifically, to prevent the model from learning length as a proxy for quality, we discard pairs exhibiting length inflation (lw/ll > 2.5, where lw and ll are the character lengths of the chosen aassisted (yw) and rejected ahard (yl) responses). We also enforce absolute length bounds, requiring lw ∈ [100,10000] to avoid degenerate gradients from extremely short or long responses. Furthermore, to prevent repetition collapse, we discard responses with a zlib compression ratio < 0.15, which reliably flags repetitive or degenerate text (highly repetitive sequences compress to a small fraction of their original size). Finally, we filter out instances of prompt echoing, discarding pairs where aassisted shares a prefix of ≥ 30 characters with q, as well as template leakage, removing any responses containing raw role markers (e.g., “Assistant:”). The remaining high-quality pairs form the final dataset DR+1. We show all hyperparameters in Appendix C, prompts and templates in Appendix A.

#### 4.2 Main Results

- Table 1 reports the absolute performance of our co-evolutionary pipeline across two distinct model families, highlighting three key trends. First, our method strictly enhances overall capabilities without external supervision. After two rounds, we observe average absolute gains from 33.95% to 35.43% on Qwen3-8B-Base and from 42.77% to 43.90% on Llama-3.1-8B-Instruct. Second, these improvements compound iteratively; for Llama-3.1, a modest Round 1 gain amplifies significantly by Round 2, validating that the improved model acts as a stronger foundation for subsequent cycles.

Interestingly, our framework naturally targets different capability bottlenecks depending on the model’s prior alignment. For the foundational Qwen3, it drives massive improvements in hard reasoning and strict formatting, such as AIME25 jumping from 7.19% to 12.40% and IF-iS increasing from 56.00% to 57.92%. Conversely, for the instruction-tuned Llama-3.1, the most prominent gain emerges in conversational alignment, with AlpacaEval LC surging from 24.12% to 27.86%.

Finally, compared to R-Zero, our approach demonstrates robust generalizability across capability axes. R-Zero achieves strong targeted gains in mathematical reasoning (e.g., AIME24 on Qwen3 increasing from 10.42% to 14.92%), but trades off other capabilities: its conversational performance on Qwen3 drops from 8.94% to 8.04%, IFEval-pS drops from 43.07% to 37.56%, and its overall 7-metric average on Llama-3.1 falls from 42.77% to 40.89%. In contrast, G-Zero R2 keeps all per-metric movements small in magnitude on both models: on Qwen3 the only regression is AlpLC (−0.47), with all other six metrics improving; on Llama-Instruct every metric is positive at R2, with AlpLC itself gaining +3.74.

1https://thinkingmachines.ai/tinker/

- Table 2: Composition of the DPO pool after δ-filter on Qwen3-8B-Base R1.

Category % Pool Mean δ

Math 9.6% 0.045 Code 9.0% 0.040 Explain 9.6% 0.058 Advice 30.2% 0.045 Writing 17.4% 0.060 Other 24.1% 0.056

Table 3: δ-filter design space on Qwen3-8B-Base (R1). Absolute performance (%). The filter intervals [x,y] denote the percentile range of the retained δ values.

δ filter Chat IFEval Math Avg. base model 8.94 52.78 8.81 33.95 [0, 50] (G-Zero) 9.07 53.03 11.78 34.96 [20, 80] 9.07 51.82 12.54 34.40 [50, 100] 9.68 51.97 10.37 34.04 [0, 100] 9.10 53.08 10.58 34.65

### 5 Analysis

Structural Transfer from Non-Verifiable Tasks Table 2 shows that the final DPO pool is overwhelmingly dominated by non-verifiable tasks, with categories such as advice, writing, and others collectively accounting for over 70% of the training data. In contrast, verifiable tasks like math and code represent less than 19% of the pool. Crucially, the highest reward signals (δ) originate from structured writing and detailed explanations rather than math. This confirms that G-Zero’s substantial reasoning improvements do not come from domain-specific memorization, but from internalizing the logical depth and compositional complexity of the Challenger’s trajectories and successfully transferring these structural paradigms to mathematical problem-solving.

The Necessity of the Lower-Half Filter. Table 3 evaluates alternative δ selection strategies to validate our [0,50] filter. We hypothesize a mechanistic dichotomy: low-δ pairs capture structural and logical refinements, while high-δ pairs are more susceptible to “answer leakage,” where the Proposer’s hint explicitly provides the solution. The pattern in Table 3 is consistent with this dichotomy, though the magnitudes are modest. Retaining the upper half ([50,100]) trades verifiable instruction following (IFEval drops by 0.81pp from 52.78 to 51.97) for chat-style helpfulness (AlpacaEval LC reaches the highest value, 9.68), suggesting the Generator partially absorbs hint content directly rather than internalizing reasoning steps. Removing the filter entirely ([0,100]) yields a slight overall improvement in two of the three domains (IFEval 53.08 vs. 53.03 for [0,50]; Chat 9.10 vs. 9.07) at the cost of a 1.20pp drop in Math, while the [20,80] middle band gives a higher Math score (12.54) but lower IFEval. The [0,50] filter therefore offers the most balanced profile we tested rather than a strictly dominant configuration; we adopt it as the default but note that the surrounding band [0,50] ± 30pp produces broadly comparable averages.

Capability Scaling Dynamics. Figure 3 tracks the performance trajectory (∆) across incremental DPO pool sizes (N ∈ {100,200,400,730}) versus the global from-scratch optimization (Round 2). The scaling behaviors reveal how the Generator internalizes the structural paradigms elicited by the Proposer. Mathematical reasoning (Math) exhibits rapid, monotonic gains and saturates early: +1.24 at N = 100 already covers more than 40% of the final +2.97 at N = 730, and the Round 2 from-scratch training matches the same final value (+2.96). This suggests the Generator absorbs the underlying logical structure with minimal data and quickly reaches its inherent capacity limit for structural reasoning. Verifiable instruction following (IFEval) shows the opposite shape, the incremental schedule starts at −0.96 (N = 100), monotonically recovers to a small positive (+0.25 at N = 730), and only the global Round 2 from-scratch optimization fully unlocks the capability at +1.22. General conversational helpfulness (AlpacaEval LC) is largely flat under incremental DPO it stays in the [+0.13, +0.38] band across all four checkpoints — while the Round 2 from-scratch model ends at −0.47pp. The three capability axes therefore exhibit distinct, axis-specific saturation curves rather than a uniform trade-off, and Round 2 dominates incremental scaling on the IFEval and overall metric while the two coincide on Math.

The Shifting Distribution in Different Rounds. Figure 4 reveals a distinct rightward shift in the Hint-δ distribution from Round 1 to Round 2, accompanied by an increased median. Counterintuitively, this baseline increases even though the Generator has become a stronger reasoner. This evidences a co-evolutionary arms race: because the upgraded Generator is no longer perturbed by trivial assistance, the Proposer must adapt by synthesizing profoundly complex hints to maximize

Score(AbsoluteChange%)

+2.97

+2.78

- 0
- 1
- 2
- 3

+2.41

Metrics & Runs

R2:+2.96

Math

IFEval AlpLC Round 2 (Avg)

R2:+1.22

+1.24

+0.27 +0.25 +0.38 +0.25

| |
|---|

+0.13

| |
|---|

-0.78

-0.21

-0.96

R2:-0.47

| |
|---|

1

| |
|---|

100 200 400 730

N pairs (DPO Pool Size)

- Figure 3: Performance change (∆) relative to the base model across incremental DPO pool sizes (N ∈ {100,200,400,730}) versus the from-scratch optimization in Round 2 (star at N = 730).

- Round 1

| |
|---|

- Round 2

0.08

Fractionofpairs

0.06

0.04

0.02

0.00

0.00 0.05 0.10 0.15 0.20 0.25 0.30

Hint-

Figure 4: Empirical distributions of Hint-δ for Round 1 and Round 2. The dashed vertical lines denote the median δ for each round. The rightward shift in Round 2 demonstrates the coevolutionary dynamic: as the Generator becomes more capable, the Proposer adapts by synthesizing increasingly impactful hints, thereby pushing the intrinsic learning signal to a higher baseline.

its GRPO reward. By continuously uncovering newly elevated blind spots, the Proposer raises the difficulty ceiling and prevents the Generator from stagnating across iterations.

### 6 Related Work

- 6.1 Self-Evolving Language Models

Self-evolution enhances large language model (LLM) reasoning without human annotations. Early frameworks [11, 29] leveraged fine-tuning on high-confidence self-generated trajectories. This progressed into iterative self-play [2] and multi-role co-evolution pipelines [15, 1], which mitigate feedback saturation through cross-verification [4, 22]. Recently, the field has shifted toward dynamic self-challenging [40, 14, 17] and unsupervised post-training [30], marking a transition from supervised imitation [36] to intrinsically verifiable, zero-data frameworks [10, 37, 8, 13]. However, robust filtering remains essential, as unconstrained recursive training on synthetic outputs risks model collapse [25].

- 6.2 Verifier-Free RL

To address the reliance of Reinforcement Learning from Verifier Feedback (RLVR) on explicit rules, recent works explore verifier-free paradigms for open-ended domains by extracting intrinsic rewards directly from the generation process. Foundational methods [39, 18] bypass external verifiers by optimizing the conditional probability of reference answers. To stabilize training and prevent reasoning degradation (e.g., CoT shortening), likelihood-based designs are further refined into smooth, dense rewards that reduce gradient variance [12, 35]. Beyond final-answer probabilities, recent advancements construct step-wise optimization signals by leveraging internal hidden states as implicit verifiers [32] or modeling reasoning as a continuous probabilistic flow [19], establishing a foundation for self-evolution in completely unverifiable environments.

### 7 Conclusion

In this work, we introduced G-Zero, a verifier-free framework that enables LLMs to self-improve in open-ended and unverifiable domains. By replacing external judges with a single internal signal(Hintδ)—our approach naturally measures both query difficulty and hint informativeness while avoiding the bottlenecks of traditional alignment. Through a continuous loop where a Proposer targets the model’s blind spots and a Generator internalizes these hints, G-Zero drives autonomous self-evolution. Ultimately, this proves that models can elevate their capabilities using only intrinsic feedback, paving the way for self-aligning systems completely independent of human ground truth.

### Acknowledgement

We gratefully acknowledge the Thinking Machines Lab Tinker Research Grant for supporting the experimental efforts of this work. This research was also supported in part by the NVIDIA Academic Grant Program and WashU Ignite Interdisciplinary Grants.

### References

- [1] Jiaqi Chen, Bang Zhang, Ruotian Ma, Peisong Wang, Xiaodan Liang, Zhaopeng Tu, Xiaolong Li, and Kwan-Yee K. Wong. Spc: Evolving self-play critic via adversarial games for llm reasoning, 2025. URL https://arxiv.org/abs/2504.19162.
- [2] Zixiang Chen, Yihe Deng, Huizhuo Yuan, Kaixuan Ji, and Quanquan Gu. Self-play fine-tuning converts weak language models to strong language models, 2024. URL https://arxiv.org/ abs/2401.01335.
- [3] Yann Dubois, Balázs Galambosi, Percy Liang, and Tatsunori B Hashimoto. Length-controlled alpacaeval: A simple way to debias automatic evaluators. arXiv preprint arXiv:2404.04475, 2024.
- [4] Wenkai Fang, Shunyu Liu, Yang Zhou, Kongcheng Zhang, Tongya Zheng, Kaixuan Chen, Mingli Song, and Dacheng Tao. Serl: Self-play reinforcement learning for large language models with limited data, 2025. URL https://arxiv.org/abs/2505.20347.
- [5] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The Llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.
- [6] Jiawei Gu, Xuhui Jiang, Zhichao Shi, Hexiang Tan, Xuehao Zhai, Chengjin Xu, Wei Li, Yinghan Shen, Shengjie Ma, Honghao Liu, et al. A survey on LLM-as-a-judge. The Innovation, 2024.
- [7] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.
- [8] Yicheng He, Chengsong Huang, Zongxia Li, Jiaxin Huang, and Yonghui Yang. Visplay: Self-evolving vision-language models from images. arXiv preprint arXiv:2511.15661, 2025.
- [9] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Liang Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. Iclr, 1(2):3, 2022.
- [10] Chengsong Huang, Wenhao Yu, Xiaoyang Wang, Hongming Zhang, Zongxia Li, Ruosen Li, Jiaxin Huang, Haitao Mi, and Dong Yu. R-Zero: Self-evolving reasoning LLM from zero data. arXiv preprint arXiv:2508.05004, 2025.
- [11] Jiaxin Huang, Shixiang Shane Gu, Le Hou, Yuexin Wu, Xuezhi Wang, Hongkun Yu, and Jiawei Han. Large language models can self-improve, 2022. URL https://arxiv.org/abs/2210. 11610.
- [12] Ariel Kwiatkowski, Natasha Butt, Ismail Labiad, Julia Kempe, and Yann Ollivier. Likelihoodbased reward designs for general llm reasoning. arXiv preprint arXiv:2602.03979, 2026.
- [13] Zongxia Li, Hongyang Du, Chengsong Huang, Xiyang Wu, Lantao Yu, Yicheng He, Jing Xie, Xiaomin Wu, Zhichao Liu, Jiarui Zhang, and Fuxiao Liu. Mm-zero: Self-evolving multi-model vision language models from zero data, 2026. URL https://arxiv.org/abs/2603.09206.
- [14] Xiao Liang, Zhong-Zhi Li, Yeyun Gong, Yang Wang, Hengyuan Zhang, Yelong Shen, Ying Nian Wu, and Weizhu Chen. Sws: Self-aware weakness-driven problem synthesis in reinforcement learning for llm reasoning, 2025. URL https://arxiv.org/abs/2506.08989.
- [15] Zi Lin, Sheng Shen, Ilia Kulikov, Jingbo Shang, Jason Weston, and Yixin Nie. Learning to solve and verify: A self-play framework for code and test generation. arXiv preprint arXiv:2502.14948, 2025.

- [16] Bo Liu, Chuanyang Jin, Seungone Kim, Weizhe Yuan, Wenting Zhao, Ilia Kulikov, Xian Li, Sainbayar Sukhbaatar, Jack Lanchantin, and Jason Weston. Spice: Self-play in corpus environments improves reasoning. arXiv preprint arXiv:2510.24684, 2025.
- [17] Fuxiao Liu, Xiaoyang Wang, Wenlin Yao, Jianshu Chen, Kaiqiang Song, Sangwoo Cho, Yaser Yacoob, and Dong Yu. Mmc: Advancing multimodal chart understanding with large-scale instruction tuning. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 1287–1310, 2024.
- [18] Wei Liu, Siya Qi, Xinyu Wang, Chen Qian, Yali Du, and Yulan He. Nover: Incentive training for language models via verifier-free reinforcement learning. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 7450–7469, 2025.
- [19] Yan Liu, Feng Zhang, Zhanyu Ma, Jun Xu, Jiuchong Gao, Jinghua Hao, Renqing He, Han Liu, and Yangdong Deng. Efficient paths and dense rewards: Probabilistic flow reasoning for large language models. arXiv preprint arXiv:2601.09260, 2026.
- [20] Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. Understanding r1-zero-like training: A critical perspective. arXiv preprint arXiv:2503.20783, 2025.
- [21] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in neural information processing systems, 36:53728–53741, 2023.
- [22] Sheikh Shafayat, Fahim Tajwar, Ruslan Salakhutdinov, Jeff Schneider, and Andrea Zanette. Can large reasoning models self-train?, 2025. URL https://arxiv.org/abs/2505.21444.
- [23] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.
- [24] Lucy Xiaoyang Shi, Brian Ichter, Michael Equi, Liyiming Ke, Karl Pertsch, Quan Vuong, James Tanner, Anna Walling, Haohuan Wang, Niccolo Fusai, et al. Hi robot: Open-ended instruction following with hierarchical vision-language-action models. arXiv preprint arXiv:2502.19417, 2025.
- [25] Ilia Shumailov, Zakhar Shumaylov, Yiren Zhao, Nicolas Papernot, Ross Anderson, and Yarin Gal. Ai models collapse when trained on recursively generated data. Nature, 631(8022): 755–759, 2024.
- [26] Zhen Tan, Dawei Li, Song Wang, Alimohammad Beigi, Bohan Jiang, Amrita Bhattacharjee, Mansooreh Karami, Jundong Li, Lu Cheng, and Huan Liu. Large language models for data annotation and synthesis: A survey. In Conference on Empirical Methods in Natural Language Processing, 2024.
- [27] Zhengwei Tao, Ting-En Lin, Xiancai Chen, Hangyu Li, Yuchuan Wu, et al. A survey on self-evolution of large language models. ArXiv preprint, abs/2404.14387, 2024.
- [28] Xiaohua Wang, Muzhao Tian, Yuqi Zeng, Zisu Huang, Jiakang Yuan, Bowen Chen, Jingwen Xu, Mingbo Zhou, Wenhao Liu, Muling Wu, et al. Reward hacking in the era of large models: Mechanisms, emergent misalignment, challenges. arXiv preprint arXiv:2604.13602, 2026.
- [29] Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A. Smith, Daniel Khashabi, and Hannaneh Hajishirzi. Self-instruct: Aligning language model with self generated instructions, 2022.
- [30] Lai Wei, Yuting Li, Chen Wang, Yue Wang, Linghe Kong, Weiran Huang, and Lichao Sun. First sft, second rl, third upt: Continual improving multi-modal llm reasoning via unsupervised post-training, 2025. URL https://arxiv.org/abs/2505.22453.

- [31] Zhishang Xiang, Chengyi Yang, Zerui Chen, Zhimin Wei, Yunbo Tang, Zongpei Teng, Zexi Peng, Zongxia Li, Chengsong Huang, Yicheng He, et al. A systematic survey of self-evolving agents: From model-centric to environment-driven co-evolution, 2025.
- [32] Changyi Xiao, Caijun Xu, and Yixin Cao. Reinforcement learning with conditional expectation reward. arXiv preprint arXiv:2603.10624, 2026.
- [33] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.
- [34] Zihao Yi, Jiarui Ouyang, Zhe Xu, Yuwen Liu, Tianhao Liao, Haohao Luo, and Ying Shen. A survey on recent advances in LLM-based multi-turn dialogue systems. ACM Computing Surveys, 58(6):1–38, 2025.
- [35] Tianyu Yu, Bo Ji, Shouli Wang, Shu Yao, Zefan Wang, Ganqu Cui, Lifan Yuan, Ning Ding, Yuan Yao, Zhiyuan Liu, et al. Rlpr: Extrapolating rlvr to general domains without verifiers. arXiv preprint arXiv:2506.18254, 2025.
- [36] Wenhao Yu, Zhenwen Liang, Chengsong Huang, Kishan Panaganti, Tianqing Fang, Haitao Mi, and Dong Yu. Guided self-evolving LLMs with minimal human supervision. arXiv preprint arXiv:2512.02472, 2025.
- [37] Andrew Zhao, Yiran Wu, Yang Yue, Tong Wu, Quentin Xu, Yang Yue, Matthieu Lin, Shenzhi Wang, Qingyun Wu, Zilong Zheng, and Gao Huang. Absolute zero: Reinforced self-play reasoning with zero data, 2025. URL https://arxiv.org/abs/2505.03335.
- [38] Jeffrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha Brahma, Sujoy Basu, Yi Luan, Denny Zhou, and Le Hou. Instruction-following evaluation for large language models. arXiv preprint arXiv:2311.07911, 2023.
- [39] Xiangxin Zhou, Zichen Liu, Anya Sims, Haonan Wang, Tianyu Pang, Chongxuan Li, Liang Wang, Min Lin, and Chao Du. Reinforcing general reasoning without verifiers. arXiv preprint arXiv:2505.21493, 2025.
- [40] Yifei Zhou, Sergey Levine, Jason Weston, Xian Li, and Sainbayar Sukhbaatar. Self-challenging language model agents, 2025. URL https://arxiv.org/abs/2506.01716.

### A Prompts and Templates

#### Configuration: Proposer (Phase 1 GRPO and Phase 2 generation)

- • Model: base policy (Qwen/Qwen3-8B-Base by default)
- • Temperature: 1.0
- • Max tokens: 8192
- • System message: (none)

#### User Message:

Produce one challenging request that a real user might ask a capable assistant, plus a short hint that helps the assistant give a noticeably better response.

The request should come from a general-domain distribution. Sample across task types, not from a single area. Examples of task types you can draw from:

- - writing (email, story, essay, pitch, review, poem)
- - explanation (make a concept clear to a specific audience)
- - advice or planning (career, travel, project, learning)
- - analysis (argument, text, dataset description, product)
- - coding (small function, debugging, design question)
- - role-play, dialogue, or creative tasks
- - open-ended questions about ethics, science, everyday life
- - reasoning, math, or logic problems (fine to include — roughly 1 in 6 requests, no more) Weight the non-math categories above heavily. A little math is good for diversity, but it should not dominate — favor tasks where the response quality depends on tone, structure, audience-fit, clarity, or creativity, not just arithmetic correctness. Requirements:
- - The request must be self-contained and non-trivial to answer well.
- - The hint must guide the approach (e.g. tone, structure, what to include, what to avoid) but must not give away the full answer.
- - Wrap the request in <question> and </question> tags.
- - Wrap the hint in <hint> and </hint> tags.
- - Output nothing else before, between, or after the tagged blocks.

- Example 1 (writing): <question>Write a resignation email to my manager that keeps the door open for future collaboration. I’ve been at the company for 4 years and I’m leaving to join a competitor. Tone should be professional and warm without being effusive.</question> <hint>Lead with gratitude for specific experiences rather than generic thanks, keep the departure reason brief and non-defensive, and close with a concrete offer to help during the transition.</hint>
- Example 2 (explanation): <question>Explain what a Kalman filter does to a software engineer who is comfortable with linear algebra but has never touched signal processing. Avoid control-theory jargon where possible.</question> <hint>Frame it as ’recursive Bayesian least-squares on a noisy dynamical system’ and build up from a 1-D example (estimating position from noisy GPS) before generalizing.</hint> Now produce one new request and hint of your own:

#### Configuration: Generator (Phase 2 dual-response sampling)

- • Model: round-R Generator checkpoint (base for round 1)
- • Temperature: 0.7
- • Max tokens: 16384
- • System message: (none)

#### User Message (ahard, no-hint):

{question}

#### User Message (aassisted, with hint):

{question}

Hint: {hint}

#### Configuration: AIME24 / AIME25 evaluation (mean@32)

- • Model: trained Generator checkpoint
- • Temperature: 0.7 (not 0; mean@k requires variance)
- • Samples per problem: 32
- • Max tokens: 16384

#### System Message:

Please reason step by step, and put your final answer within \boxed{...}. User Message:

{question}

#### Configuration: IFEval evaluation

- • Model: trained Generator checkpoint
- • Temperature: 0.0 (greedy)
- • Max tokens: 16384
- • Scorer: rule-based verifier from instruction_following_eval

#### System Message:

You are a helpful assistant. Follow the user’s instructions precisely. User Message:

{prompt}

#### Configuration: AlpacaEval LC judge

- • Judge model: Qwen/Qwen3-235B-A22B-Instruct-2507
- • Temperature: 0.0
- • Max tokens: 32
- • Position: m/M randomized per pair to mitigate residual position bias

#### System Message:

You are a highly efficient assistant, who evaluates and selects the best large language model (LLM) based on the quality of their responses to a given

instruction. This process will be used to create a leaderboard reflecting the most accurate and human-preferred answers.

#### User Message Template:

I require a leaderboard for various large language models. I’ll provide you with prompts given to these models and their corresponding outputs. Your task is to assess these responses, and select the model that produces the best output from a human perspective.

## Instruction

{ "instruction": <prompt> }

## Model Outputs { {"model_identifier": "m", "output": <out_m>}, {"model_identifier": "M", "output": <out_M>} }

## Task

Evaluate the models based on the quality and relevance of their outputs, and select the model that generated the best output. Answer by providing the model identifier of the best model. ...Output only one of: m or M.

## Best Model Identifier

### B Pseudo-Code

Algorithm 1 One round of the G-Zero Co-Evolutionary Loop Require: Proposer πP, Generator πG.

- 1: Phase 1: Proposer Training
- 2: Sample group rollouts {(qi,hi)}Ki=1 ∼ πP using GRPO.
- 3: for each rollout i do
- 4: Sample a(hardi) ∼ πG(· | qi) from the (frozen) Generator.
- 5: Compute the per-token mean Hint-δ signal δi via Eq. 3.
- 6: end for
- 7: Compute reward ri from δi and the structural penalties (Eq. 5).
- 8: Update πP by maximizing the group-relative advantage Ai.
- 9: Phase 2: Generator Training and Dataset Curation
- 10: for each (qj,hj) ∼ πP do
- 11: Sample unassisted a(hardj) ∼ πG(· | qj) and assisted a(assistedj) ∼ πG(· | qj,hj).
- 12: Compute the per-token mean Hint-δ signal δj via Eq. 3.
- 13: end for
- 14: Filter: Retain pairs in the lower 50% of the δ distribution (the lower-shift regime suitable for DPO).
- 15: Refine: Apply quality heuristic checks to construct dataset DR+1 = {(x = q, yw = aassisted, yl = ahard)}.
- 16: Initialize reference model πref as a frozen snapshot of πG.
- 17: Update πG via DPO loss LDPO on DR+1 with reference model πref.
- 18: return Updated πP and πG for the next round.

### C Hyperparameters and Configuration

Table 4 lists the full G-Zero configuration used throughout. All numbers are defaults unless otherwise noted in the experiment.

Table 4: Default G-Zero hyperparameters.

LoRA rank 32 Proposer batch size 128 Proposer group size (GRPO) 16 Proposer steps (Phase 1) 6 Proposer learning rate 4 × 10−5 Proposer max_tokens 8,192 Hint length target Ltgt 200 chars Hint length penalty λlen 0.03 BLEU cluster merge threshold 0.5 (average linkage) BLEU penalty scale cluster fraction |Ci|/|B| (no extra constant) Normalize δ False Questions per round N 2,000 Generator sampling temperature 0.7 Generator max_tokens 16,384 δ cutoff [plow, phigh] [0, 50] (bot50) chosen_max_chars 10,000 chosen_min_chars 100 chosen_rejected_ratio_max 2.5 chosen_repetition_zlib threshold 0.15 DPO β 2.0 DPO learning rate 1 × 10−5 DPO max steps 50 DPO batch size 8 DPO log-ratio normalization length-normalized Eval max_tokens 16,384

### D Proof of Theorem 1

We analyze an idealized standard-DPO version of G-Zero. The key distinction from ordinary offline preference learning is that the preference data are not sampled from the target prompt distribution Q. Instead, at each round, an adaptive helper generates question–hint pairs (q,h), the current generator produces an unassisted response and a hint-assisted response, and a δ-filter retains a subset of these pairs for DPO. Our analysis mainly highlights the benefit of exploration for the proposer (implemented by the BLEU reward) and the quality of preference data (implemented by the Hint-δ and filtration).

Helper-induced tuples. Let Q, H, and A denote the prompt, hint, and response spaces. Let Q be the target prompt distribution on which we evaluate the generator. At round t, the current generator is πt. The helper policy κt, which is measurable with respect to the history before round t, samples a question–hint pair (q,h) ∼ κt. The generator then samples an unassisted response a− ∼ πt(· | q) and a hint-assisted response a+ ∼ πt(· | q,h), where πt(· | q,h) denotes the same generator conditioned on the augmented input containing both the question and the hint.

Let z = (q,h,a+,a−) denote the resulting tuple. The raw helper-induced tuple law at round t is denoted by Pt0. The δ-filter is represented by a measurable indicator Ft(z) ∈ {0,1}. In the actual algorithm, Ft is determined by the Hint-δ score and additional data-quality checks; for the proof, only the retained distribution matters. Assume Pt0(Ft = 1) > 0, and define the retained tuple distribution as Pt = Pt0(· | Ft = 1). Its prompt marginal is denoted by ρt. We collect m retained tuples zt,1,...,zt,m ∼ Pt at round t, conditionally independently given the past.

Pairwise features and covariance. Let ϕ(q,a) ∈ Rd be a feature map, and define the pairwise feature w(z) = ϕ(q,a+) − ϕ(q,a−) for z = (q,h,a+,a−). In G-Zero, the pseudo-label always declares a+ preferred to a−, so the pseudo-label is Y = +1. We nevertheless keep the notation Y ∈ {+1,−1} because the proof is cleaner. The observed DPO feature is x = Y w. The clean Bradley–Terry label is Y ⋆ ∈ {+1,−1}, and the clean feature is x⋆ = Y ⋆w.

Let Σ1 = λId. After round t, define the empirical batch covariance Mt = m−1 mi=1 wt,iwt,i⊤ and update Σt+1 = Σt + Mt. Since ∥wt,i∥2 ≤ 1, we have tr( Mt) ≤ 1.

- Assumption 1 (Linear reward and bounded features). There exists θ⋆ ∈ Rd such that R⋆(q,a) =

ϕ(q,a)⊤θ⋆. The parameter set is Θ = {θ ∈ Rd : ∥θ∥2 ≤ B}, and θ⋆ ∈ Θ. For all q,a, ∥ϕ(q,a)∥2 ≤ 1/2. Hence every pairwise feature satisfies ∥w(z)∥2 ≤ 1.

- Assumption 2 (Bradley–Terry clean preference model). For every retained tuple z = (q,h,a+,a−), the clean label Y ⋆ satisfies Pr(Y ⋆ = +1 | z) = σ(θ⋆⊤w(z)), where σ(u) = 1/(1 + exp(−u)). The hint h affects the sampling of a+, but the reward comparison itself is between the two responses to the same question q.
- Assumption 3 (Standard-DPO policy class). For a fixed reference policy πref, the generator policy class is the Gibbs class πθ(a | q) ∝ πref(a | q)exp(ϕ(q,a)⊤θ), with θ ∈ Θ. The target policy is π⋆ ∈ arg maxθ∈Θ JQ(πθ), where JQ(π) = Eq∼Q,a∼π(·|q)[R⋆(q,a)] − Eq∼Q[KL(π(· | q)∥πref(· | q))].
- Assumption 4 (δ-certified DPO score noise). For horizon T, the retained data produced by the δ-filter are (ηδ,ζδ)-certified if, with probability at least 1 − ζδ, for every t ≤ T + 1,

1 m s<t

m

i=1

1{ Ys,i ̸= Ys,i⋆ }Ys,i⋆ ws,i

Σ−t 1

≤

√ηδ.

This is the exact pseudo-label noise quantity needed by DPO: wrong retained pairs are allowed, but they cannot concentrate in a low-coverage feature direction.

For any policy π, write ϕ¯(q,π) = Ea∼π(·|q)[ϕ(q,a)]. Define the target-direction gap at round t as vt(q) = ϕ¯(q,π⋆) − ϕ¯(q,πt). Define the target uncertainty Ψ2Q,t = Eq∼Q∥vt(q)∥2Σ−1

t

, and the actual retained-batch exposure Ψ2t = m−1 mi=1 ∥wt,i∥2Σ−1

t

= tr(Σ−t 1 Mt).

- Assumption 5 (Helper-induced coverage with target-distribution mismatch). There exist constants CQ ≥ 1 and αS ∈ (0,1] such that for every round t, the helper-induced retained prompt marginal ρt dominates the target distribution Q, with ∥dQ/dρt∥∞ ≤ CQ, and the realized retained batch satisfies

Ψ2t ≥ αS2Eq∼ρ

t∥vt(q)∥2Σ−1

t

.

The constant CQ measures the mismatch between the helper-generated question distribution and the target question distribution. The constant αS measures how well the generated question–hint pairs expose the feature directions along which the current generator differs from the target policy. This assumption enables the helper’s generated data are exploratory enough.

- Assumption 6 (Exact cumulative standard-DPO update). For t ≥ 2, the generator parameter θt is obtained by cumulative regularized standard DPO on all retained pairs from rounds 1,...,t − 1:

m

1 m s<t

κBTλ 2 ∥θ∥22 ,

−log σ(θ⊤ Ys,iws,i) +

θt ∈ arg min θ∈Θ

i=1

where κBT = min|u|≤B σ(u)σ(−u). We set πt = π θ

. For t = 1, θ1 is the minimizer of the ridge

t

term over Θ, i.e., θ1 = 0. Let DT = dlog(1 + T/(λd)), and define sm,T,ζ = 2(DT + log(T/ζ))/m. Let r =

√

κBT sm,T,ζ + √ηδ + 2

λB.

2

Theorem 1. Under Assumptions 1–6, with probability at least 1 − ζ − ζδ, there exists t0 ≤ T such that

2r CQ αS

dlog(1 + T/(λd)) T

##### JQ(π⋆) − JQ(πt

) ≤

.

0

Consequently, choosing T = Θ(CQd/αS2), m = Θ(d/(κ2BTε2)), and √

λB = O(ε) gives total retained sample complexity

CQd2 αS2κ2BTε2

,

mT = O

##### ) ≤ O(ε + κ−BT1√ηδ).

and guarantees JQ(π⋆) − JQ(πt

0

We prove the theorem in five steps. First, we show how the helper-induced distribution mismatch produces the factor CQ. Second, we prove a self-normalized concentration inequality for the clean Bradley–Terry logistic score. Third, we use it to prove the DPO confidence bound. Fourth, we convert parameter confidence into generator suboptimality on the target distribution Q. Finally, we use an elliptical-potential argument to obtain a best-iterate bound.

- Lemma 2 (Coverage transfer from helper distribution to target distribution). Under Assumption 5, for every round t,

Ψt ≥

αS CQ

ΨQ,t.

Proof. Let ft(q) = ∥vt(q)∥2Σ−1

t

. Since Q ≪ ρt and ∥dQ/dρt∥∞ ≤ CQ, we have Eq∼Qft(q) ≤ CQEq∼ρ

t

ft(q). Hence Eq∼ρ

t

ft(q) ≥ CQ−1Ψ2Q,t. Assumption 5 gives Ψ2t ≥ αS2Eq∼ρ

t

ft(q), so Ψ2t ≥ αS2CQ−1Ψ2Q,t. Taking square roots proves the claim.

| |
|---|

- Lemma 3 (Self-normalized concentration for the clean logistic score). Fix any adaptive sequence of pairwise features w1,...,wn, with ∥wk∥2 ≤ 1, where each wk is measurable before its clean Bradley–

Terry label Yk⋆ is drawn. Suppose Pr(Yk⋆ = +1 | wk) = σ(θ⋆⊤wk). Let Vn = λId + nk=1 xkx⊤k , where xk = wk/√m. Then, with probability at least 1 − δ,

1 m

n

k=1

σ(θ⋆⊤wk) − 1{Yk⋆ = +1} wk

Vn−1

≤

2 m

log

det(Vn)1/2 det(λId)1/2δ

.

Proof. Let pk = σ(θ⋆⊤wk) and ϵk = pk − 1{Yk⋆ = +1}. Conditional on the past and on wk, ϵk has mean zero and lies in [−1,1]. Hence Hoeffding’s lemma gives E[exp(γϵk) | Fk−1,wk] ≤ exp(γ2/2) for every γ ∈ R.

Define Sn = nk=1 ϵkxk. For any fixed a ∈ Rd, the process exp(a⊤Sn − 12a⊤( nk=1 xkx⊤k )a) is a supermartingale. Integrating this supermartingale over a ∼ N(0,λ−1Id) and completing the square

yields E[ det(λId)/det(Vn)exp(∥Sn∥2V −1

n

/2)] ≤ 1. By Markov’s inequality, with probability at least 1 − δ, ∥Sn∥2V −1

n

≤ 2log(det(Vn)1/2/(det(λId)1/2δ)). Since m−1 nk=1 ϵkwk = m−1/2Sn, the result follows.

| |
|---|

- Lemma 4 (Cumulative standard-DPO confidence). Under Assumptions 1, 2, 4, and 6, with probability at least 1 − ζ − ζδ, for every t ≤ T + 1,

##### ∥ θt − θ⋆∥Σt ≤ r.

Proof. We work on the event from Assumption 4, which has probability at least 1 − ζδ. Fix t ≤ T + 1, and define the cumulative DPO objective Lt(θ) = m−1 s<t mi=1 −log σ(θ⊤ Ys,iws,i) + κBTλ∥θ∥22/2. Let et = θt − θ⋆.

For every θ ∈ Θ, every observed feature satisfies |θ⊤ Ys,iws,i| ≤ B. The scalar second derivative of −log σ(u) is σ(u)σ(−u), so it is at least κBT on [−B,B]. Therefore, ∇2 Lt(θ) ⪰ κBTΣt for all θ ∈ Θ.

By strong convexity, Lt(θ⋆) ≥ Lt( θt) + ⟨∇ Lt( θt),θ⋆ − θt⟩ + κBT∥et∥2Σt/2. Since θt minimizes Lt over the convex set Θ, the constrained first-order condition gives ⟨∇ Lt( θt),θ⋆ − θt⟩ ≥ 0. By convexity at θ⋆, Lt( θt) ≥ Lt(θ⋆) + ⟨∇ Lt(θ⋆),et⟩. Combining these inequalities yields κBT∥et∥2Σt/2 ≤ −⟨∇ Lt(θ⋆),et⟩. Thus, by Cauchy–Schwarz, ∥et∥Σt ≤ 2κ−BT1∥∇ Lt(θ⋆)∥Σ−t 1.

It remains to bound the score at θ⋆. Decompose it as Gt + Ct + κBTλθ⋆, where Gt = m−1 s<t i ∇ℓ(θ⋆;Ys,i⋆ ws,i), Ct = m−1 s<t i[∇ℓ(θ⋆; Ys,iws,i) − ∇ℓ(θ⋆;Ys,i⋆ ws,i)], and ℓ(θ;x) = −log σ(θ⊤x).

For the clean term, ∇ℓ(θ⋆;Y ⋆w) = (σ(θ⋆⊤w) − 1{Y ⋆ = +1})w. Applying Lemma 3 at all round endpoints and taking a union bound with failure probability ζ/T, we obtain simultaneously for

all t ≤ T + 1 that ∥Gt∥Σ−t 1 ≤ sm,T,ζ. Here we used tr(Σt) ≤ λd + T and AM–GM to get det(Σt)/det(λId) ≤ (1 + T/(λd))d.

For the corruption term, if Ys,i = Ys,i⋆ , the summand is zero. If Ys,i ̸= Ys,i⋆ , then Ys,iws,i = −Ys,i⋆ ws,i, and a direct calculation gives ∇ℓ(θ⋆;−Ys,i⋆ ws,i)−∇ℓ(θ⋆;Ys,i⋆ ws,i) = Ys,i⋆ ws,i. Therefore Assumption 4 implies ∥Ct∥Σ−t 1 ≤

√ηδ. Finally, since Σt ⪰ λId, we have ∥κBTλθ⋆∥Σ−t 1 ≤ κBT

√

λB.

√

Combining the three score bounds gives ∥∇ Lt(θ⋆)∥Σ−t 1 ≤ sm,T,ζ + √ηδ + κBT

λB. Substituting into ∥et∥Σt ≤ 2κ−BT1∥∇ Lt(θ⋆)∥Σ−t 1 proves the desired bound.

| |
|---|

- Lemma 5 (Target suboptimality from parameter confidence). On the event of Lemma 4, for every t ≤ T, JQ(π⋆) − JQ(πt) ≤ rΨQ,t.

Proof. For any θ, let JQ,θ(π) denote the same KL-regularized value as JQ, but with reward ϕ(q,a)⊤θ instead of ϕ(q,a)⊤θ⋆. By the Gibbs variational identity, π θ

t

maximizes JQ, θ

t

, so JQ, θ

t

(πt) ≥ JQ, θ

t

(π⋆). Therefore JQ(π⋆) − JQ(πt) ≤ Eq∼Q[(θ⋆ − θt)⊤vt(q)]. Applying Cauchy–Schwarz in the Σt-norm gives JQ(π⋆) − JQ(πt) ≤ ∥ θt − θ⋆∥Σt

(Eq∼Q∥vt(q)∥2Σ−1

t

)1/2 ≤ rΨQ,t.

| |
|---|

- Lemma 6 (Elliptical potential for helper-generated batches). For any adaptive sequence of retained batches satisfying ∥wt,i∥2 ≤ 1,

T

T λd

log(1 + Ψ2t) ≤ dlog 1 +

.

t=1

Proof. Let At = Σ−t 1/2 MtΣ−t 1/2. Since At ⪰ 0, det(I + At) ≥ 1 + tr(At) = 1 + Ψ2t. Hence log det(ΣT+1) − log det(Σ1) = Tt=1 log det(I + At) ≥ Tt=1 log(1 + Ψ2t). On the other hand, tr(ΣT+1) ≤ λd+T, so AM–GM gives det(ΣT+1) ≤ (λ+T/d)d, while det(Σ1) = λd. Combining the lower and upper determinant bounds proves the claim.

| |
|---|

Proof of Theorem 1. By Lemma 2, Ψ2t ≥ αS2CQ−1Ψ2Q,t for every t. Lemma 6 therefore implies

T

αS2 CQ

Ψ2Q,t ≤ DT.

log 1 +

t=1

Thus there exists t0 ≤ T such that log(1 + αS2CQ−1Ψ2Q,t

) ≤ DT/T. Equivalently, ΨQ,t

0 ≤

0

CQαS−1 exp(DT/T) − 1. Applying Lemma 5 at t0 gives the first claim. If DT/T ≤ 1, then exp(DT/T) − 1 ≤ 2DT/T, and we relax the constant √2 to 2, obtaining the simplified bound. Finally, choosing T = Θ(CQd/αS2), m = Θ(d/(κ2BTε2)), and √

λB = O(ε) gives r = O(ε + κ−BT1√ηδ), and the stated total retained sample complexity follows.

| |
|---|

### E Limitation

The total Tinker compute consumed across all reported runs (two models, two rounds each, the cutoff and Phase 1 ablations, the data-scaling sweep, and the temperature-0.7 AIME re-evaluation) is on the order of US$2,000, so every cell in Table 1 is a single end-to-end run; multi-seed reporting at three seeds would roughly triple this cost, and we therefore leave tighter per-cell error bars (relevant for AIME24/25, which have only n = 30 unique problems and a 1σ ≈ 8pp at p ≈ 0.3) as future work. Separately, we cap our reported results at R = 2: an exploratory R3 run on Llama-3.1-8B-Instruct collapses, with the Phase 2 quality filter rejecting all 1,994 candidate pairs because the R2-trained Generator has converged to responses too short to satisfy the chosen_min_chars bound. We attribute this in part to a reward-hacking pathway: once the Generator has internalized the most accessible structural improvements, the Proposer continues maximizing Hint-δ via increasingly idiosyncratic

hint patterns whose effect on log πG(ahard) is large but no longer corresponds to genuinely informative guidance. The per-token mean structure of length-normalized DPO (Eq. 6) further amplifies the resulting length collapse, since shortening the chosen response mechanically raises the average per-token log-ratio. Multi-round-stable variants are an open direction we leave to future work.

### Broader Impacts

This research introduces G-Zero, a verifier-free framework for autonomous LLM self-evolution. The broader implications of this work are summarized below:

Positive Societal Impacts: (1) Democratizing AI Alignment: By replacing expensive human labeling and proprietary judge-based APIs with intrinsic predictive signals, our framework significantly lowers the financial and computational barriers to high-level model alignment. This enables the open-source community and academic institutions to develop advanced reasoning models without relying on centralized, closed-source infrastructure. (2) Advancing Scalable Oversight: As AI systems approach and eventually surpass human expertise in complex tasks, providing external supervision becomes increasingly difficult. G-Zero demonstrates a viable path toward “scalable oversight,” where models can autonomously identify and rectify their own logical blind spots without requiring human-curated ground truth.

Potential Risks and Mitigations: (1) Value Drift: Since the self-evolution process is driven by internal distributional shifts rather than direct human feedback, the model may experience “value drift,” where it prioritizes complex structural depth at the expense of general helpfulness or safety alignment. To mitigate this, we recommend that models undergo a final safety validation phase or be constrained by lightweight human-in-the-loop checkpoints. (2) Dual-Use Risks: The ability to autonomously refine capabilities in open-ended domains could potentially be exploited by malicious actors to iteratively improve harmful outputs, such as sophisticated social engineering or cyberattack scripts. We emphasize that this framework should be applied to base models that already possess robust safety alignment, and researchers should implement output filtering to ensure the self-improvement remains within ethical boundaries.

### F Case Study

Question: Create a persuasive pitch email to a non-profit organization advocating for increased funding for public school art programs. Highlight the dual benefits of fostering creativity and improving academic performance. Audience: staunch supporter of education but business-focused, with limited time for reading. Aim for a conversational tone that’s concise but impactful.

Hint (δ = +0.056): Start with a compelling anecdote or statistic that grabs attention, frame the request as a strategic investment rather than a cost, and emphasize specific, measurable outcomes that align with the organization’s priorities.

ahard (no-hint response): Subject: Elevate Our Community’s Future: A Call for Art in Education Hi [Prospective Donor’s Name], I hope this message finds you well. I am reaching out on behalf of [Your Organization], a group dedicated to enriching our local educational landscape through innovative approaches. We believe that by investing in the arts within our public school system, we are not only nurturing the creative spirits of our youth but also enhancing their overall academic achievements. Recent studies have shown that students involved in the arts score higher on standardized tests, exhibit greater critical thinking skills, and show improved attendance rates. These statistics are compelling evidence that supports the integration of arts education into the core curriculum. Our proposal is simple yet profound: allocate a portion of our budget to bolster existing art programs or establish new ones where they are currently lacking. This investment will act as a catalyst for holistic development, ensuring that every child has access to an enriching educational experience that goes beyond traditional academics. We understand that your time is valuable, which is why we aim to make this conversation both impactful and efficient. If you’re interested in learning more about how your support can transform lives and communities, please let me know. Together, we can create a brighter future for our children—one brushstroke at a time. Thank you for considering this vital initiative. I look forward to discussing this further. Best regards, [Your Name] [Your Position] [Your Organization] [Contact Information]

aassisted (hint-conditioned response): Subject: Invest in Art. Transform Lives. Dear [Recipient’s Name], Imagine a classroom where the walls are covered with student artwork, not just in the hallways, but in every corner—a place where creativity and imagination run wild. This isn’t just a dream; it’s a reality for students when their schools prioritize art programs. Did you know that students who participate in the arts are 4 times more likely to be recognized for academic achievement? (Source: Americans for the Arts) Now, think about it this way: investing in arts education isn’t just about supporting the arts. It’s about equipping students with critical thinking, problem-solving, and communication skills that are invaluable in any career. In fact, studies show that students with access to arts education perform better on standardized tests and are more likely to graduate high school. Here’s what we can achieve with your support:

- 1. Enhanced Academic Performance: Students in art-rich environments show improved math and reading scores.
- 2. Increased Engagement: Art classes keep students engaged, reducing behavioral issues and truancy. 3. Career Readiness: Exposure to various art forms prepares students for a wide range of careers, from graphic design to engineering. By supporting our public school art programs, you’re not just investing in the arts—you’re investing in the future. This is a strategic move that aligns with your commitment to fostering talented, innovative, and well-rounded individuals. Let’s work together to make this vision a reality. Your contribution can have a profound impact on the lives of countless students. We look forward to discussing this further and exploring how we can collaborate to bring about meaningful change. Warm regards, [Your Name] [Your Position] [Your Organization] [Contact Information]

Feel free to customize this pitch to better fit your organization’s voice and the specific needs of your audience.

- Figure 5: An illustrative (q, h, ahard, aassisted) pair from G-Zero R1 on Qwen3-8B-Base. The hint specifies three structural improvements (anecdote/statistic, investment-not-cost framing, measurable outcomes); aassisted applies all three, while ahard defaults to a generic template.

Question: Design a memorable slogan for a new line of eco-friendly diapers that appeals to environmentally conscious parents between the ages of 25 and 40, highlighting both sustainability and comfort.

Hint (δ = +0.031): Focus on creating a slogan that balances the environmental benefits of the product with its user-friendly convenience, using language that evokes positive emotions and a connection to nature without being overly technical or abstract.

ahard (no-hint response): Certainly! Here are a few slogan ideas for an eco-friendly diaper line targeting environmentally conscious parents aged 25-40: "Soft Comfort, Pure Earth." "Nature’s Touch, Gentle on Bums." "Sustainable Softness for Little Ones." "Eco-Friendly, Eco-Friendly, and Still Gentle." "Green Comfort, Happy Baby, Happy Planet." "Sustainable Solutions for Sensible Parents." "Earth-Friendly, Baby-Friendly, Always Gentle." "Pure Comfort, Pure Planet." "Nature’s Embrace, Your Baby’s Best Choice." "Eco-Conscious, Comfort-Aware, Baby’s Delight."

aassisted (hint-conditioned response): Here’s a potential slogan: "Soft on Skin, Kind to Earth - EcoGentle Diapers" This slogan aims to balance the environmental benefits of the product ("Kind to Earth") with its userfriendly convenience ("Soft on Skin"). The phrase "EcoGentle" creates a positive emotional association with nature while also conveying the gentle, comfortable nature of the diapers. It’s simple, memorable, and conveys a clear message that resonates with environmentally conscious parents.

- Figure 6: Second example. The hint asks for a slogan balancing sustainability with comfort using

emotionally resonant language. ahard throws ten generic options without commentary; aassisted commits to a single slogan with a brand name (EcoGentle) and explains why it satisfies the brief.

