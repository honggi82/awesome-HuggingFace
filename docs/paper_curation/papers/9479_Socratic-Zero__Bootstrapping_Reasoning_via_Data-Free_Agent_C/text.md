# arXiv:2509.24726v1[cs.CL]29Sep2025

## SOCRATIC-ZERO: BOOTSTRAPPING REASONING VIA DATA-FREE AGENT CO-EVOLUTION

### Shaobo Wang ∗β Zhengbo Jiao ∗α,β,γ Zifan Zhang α,δ Yilang Peng α,ϵ Xu Ze α Boyu Yang α Wei Wang α Hu Wei †α Linfeng Zhang †β

α Alibaba Group Holding Limited β EPIC Lab, Shanghai Jiao Tong University γ Shanghai University of Finance and Economics δ Wuhan University ϵ Zhejiang University

* Equal contribution † Corresponding authors

ABSTRACT

Recent breakthroughs in large language models (LLMs) on reasoning tasks rely heavily on massive, high-quality datasets—typically human-annotated and thus difficult to scale. While data synthesis or distillation offers a promising alternative, existing methods struggle with inconsistent data quality and an inability to dynamically adapt to the evolving capabilities of the model, leading to suboptimal training signals. To address these limitations, we introduce Socratic-Zero, a fully autonomous framework that generates high-quality training data from minimal seed examples through the co-evolution of three agents: the Teacher, the Solver, and the Generator. The Solver continuously refines its reasoning by learning from preference feedback on both successful and failed trajectories; the Teacher adaptively crafts increasingly challenging questions based on the Solver’s weaknesses; and the Generator distills the Teacher’s question-design strategy to enable scalable, highfidelity curriculum generation. This closed-loop system produces a self-improving curriculum—requiring no pre-existing tasks or labels. Remarkably, starting from only 100 seed questions, our Socratic-Solver-8B achieves an average gain of +20.2 percentage points over prior data synthesis methods across seven mathematical reasoning benchmarks (AMC23, AIME24-25, Olympiad, MATH-500, Minerva, and GSM8K), with consistent gains on both Qwen3 and GLM4 series models. Even more surprisingly, synthetic data from Socratic-Generator-32B enables student LLMs to achieve superior performance compared to other state-of-the-art (SOTA) commercial LLMs on these benchmarks, including Qwen3-235B-A22B, DeepSeek-V3.1-671B, GPT-5, Gemini-2.5-Pro, Grok-4, and Claude-4.1-Opus.

guiding the evolution of the Solver and the Generator

Teacher:

Dataflow

Plato: What makes a question unlock thought? I am learning to teach by studying how Socrates teaches.

[Figure 1]

Teacher’s behavior

[Figure 2]

Generator’s behavior Solver’s behavior

Socrates: I do not give answers — I midwife understanding. My questions must be precise enough to reveal ignorance, yet open enough to spark insight.

[Figure 3]

GenerateFeedback

Original Update Curriculum imitate

GenerateAnswers

[Figure 4]

evolve

Aristotle: I focus on how he leads me — not to the answer, but through the path. Each ‘why?’ reshapes my thinking. I learn by being lost, then found

evolve

Updated Curriculum

[Figure 5]

[Figure 6]

Solver: generating better solutions by learning from the Teacher’s feedback

Generator: producing higher-quality questions by imitating how the Teacher behaves

(a) Socrates Method (b) Socratic-Zero: A multi-agent coevolution framework

Figure 1: The Socratic-Zero Framework: From Philosophical Analogy to a Co-evolutionary System.

- (a) The Socratic Methodlogy illustrates the philosophical foundation: the Teacher (Socrates) acts as an intellectual midwife, eliciting understanding through probing questions; the Practitioner (Aristotle) learns not by receiving answers, but by being guided along a path of reasoned inquiry; and the Apprentice-Teacher (Plato) learns to teach by observing and internalizing the master’s method.
- (b) The Socratic-Zero Framework operationalizes this philosophy. Here, the Teacher—a powerful LLM—guides the co-evolution of two agents. The Solver improves by generating solutions and refining them through the Teacher’s feedback, while the Generator evolves by strategically distilling the Teacher’s behavior to produce an increasingly suitable curriculum for the Solver.

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

Socratic-Generator-32B Qwen-3-32B Qwen3-235B-A22B Claude-4.1-Opus Gemini-2.5-Pro Grok-4 GPT-5 DeepSeek-v3.1-671B

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

(b)SolverPerformance(%)(a)GeneratorPerformance(%)

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

Socratic-Solver-8B Qwen3-8B-Base Qwen3-8B-Base + SFT Qwen3-8B-Base + Static Augment Qwen3-8B-Base + LLM2LLM

- Figure 2: Overall performance comparison demonstrating the giant effectiveness of Socratic-Zero. (a) Our Socratic-Generator-32B produces synthetic data that enables student models to achieve performance competitive with much larger state-of-the-art models, showcasing strong generalization capabilities. (b) Our Socratic-Solver-8B achieves an impressive 56.1% average accuracy, marking a substantial +20.2 point improvement over the baseline.

- 1 INTRODUCTION

The pursuit of advanced mathematical reasoning in large language models has reached a critical juncture. While recent breakthroughs have demonstrated remarkable capabilities on complex mathematical problems (Hendrycks et al., 2021; Cobbe et al., 2021), these advances rely on massive datasets of meticulously curated reasoning trajectories — a requirement that is both costly and fundamentally unscalable. Current state-of-the-art models depend on millions of human-annotated problem-solution pairs and hand-designed curricula (Yu et al., 2024), creating a fundamental bottleneck that limits both accessibility and the potential for models to evolve beyond human-curated knowledge boundaries.

Current methodologies remain entrenched in a static paradigm: datasets are frozen upon collection, curricula are handcrafted in advance, and models are trained on fixed problem distributions. This approach suffers from critical weaknesses: it cannot adapt to evolving model capabilities during training, fails to exploit rich feedback signals for targeting specific weaknesses, and requires extensive human expertise for curriculum design. Recent efforts through synthetic data generation (Lee et al., 2024; Chen et al., 2025b) and iterative training (Zhao et al., 2025; Huang et al., 2025b) have shown promise but remain constrained by their reliance on external supervision and lack of effective quality control mechanisms for synthesized content.

To overcome these limitations, we introduce Socratic-Zero, a paradigm-shifting framework that eliminates dependency on large-scale external datasets while enabling truly autonomous reasoning improvement. Inspired by the Socratic method of learning through questioning (Figure 1(a)), our approach implements co-evolution between three agents: a Solver that attempts to solve mathematical questions, a Teacher that strategically generates challenging problems to expose the Solver’s weaknesses, and a Generator that learns to distill and scale the Teacher’s problem generation strategy. This architecture (Figure 1(b)) translates the philosophical dialogue of the Socratic method into a concrete, co-evolutionary computational framework. Unlike conventional pipelines that decouple data generation from model training, Socratic-Zero unifies them within a continuous co-evolutionary loop formalized as a optimization problem. Our contributions are threefold:

- • Multi-Agent Co-Evolutionary Framework: We establish a theoretical foundation for coevolutionary learning where the Solver, Teacher, and Generator agents interact dynamically, formalizing reasoning improvement as an adaptive curriculum learning problem (Figure 1(b)).
- • Socratic-Zero System: We implement a concrete framework where the Solver improves via preference learning, the Teacher evaluates correctness and generates adaptive curriculum, and the Generator learns strategic distillation through value-weighted supervised fine-tuning (WSFT), enabling autonomous reasoning advancement from minimal seed data.
- • Superior Empirical Performance: Our Socratic-Solver-8B achieves +20.2 points average improvement across seven mathematical reasoning benchmarks (Figure 2(b)), while synthetic data from our Socratic-Generator-32B achieves 37.72% downstream training effectiveness, outperforming leading commercial models including Qwen3-235B-A22B at 37.13%, Claude-4.1-Opus at 37.63%, Gemini-2.5-Pro at 37.20%, Grok-4 at 37.01%, GPT-5 at 36.62%, and DeepSeek-V3.1 at

- 36.62% (Figure 2(a)).

- 2 RELATED WORK

Data Synthesis. To alleviate data scarcity, researchers have leveraged LLMs’ generative capabilities to synthesize training samples. Early approaches used prompt engineering to guide question-answer generation (Yu et al., 2024; Zhan et al., 2025). Subsequently, LLM2LLM (Lee et al., 2024) and WarriorMath (Chen et al., 2025b) introduced deficiency-aware mechanisms, where teacher models identify knowledge gaps and generate targeted data. More recently, Absolute Zero (Zhao et al., 2025) and R-Zero (Huang et al., 2025b) explored fully autonomous self-play paradigms for continuous task generation and learning. While these advances achieve data autonomy, they lack effective quality control mechanisms, resulting in repeated use of low-value samples that severely impact effectiveness.

Data Distillation. Knowledge distillation transfers capabilities from powerful teacher models to lighter student models. Early work like Orca (Mukherjee et al., 2023) used imitation learning to replicate teacher reasoning chains. Policy distillation (Wang et al., 2025b) extends this by transferring dynamic decision-making strategies. GKD (Agarwal et al., 2024) enables students to learn from their own sequences using teacher feedback for policy correction. However, students passively accept teacher feedback without evaluating reliability, degrading learning quality when guidance is suboptimal. These methods also rely on static datasets, unable to dynamically adjust content based on students’ evolving capabilities. While recent advances (Wang et al., 2025a; Zhao et al., 2023; Zhang et al., 2024; Chen et al., 2024a; Liu et al., 2025) promote data-centric optimization, they lack effective quality control and adaptive curriculum generation.

Preference Learning. Translating feedback signals into model optimization is central to selfevolution systems. Early approaches like RLHF (Stiennon et al., 2020) trained reward models on human preferences then fine-tuned policies, but this process is complex and unstable. Recent methods like DPO (Rafailov et al., 2023) and RWSFT (Mukherjee et al., 2025) directly optimize preferences, improving efficiency and stability. Combined with self-correction mechanisms like SelfRefine (Madaan et al., 2023), models possess preliminary closed-loop capabilities. Further advances including Self-Evolved Reward Learning (Huang et al., 2025a), Self-Play Fine-Tuning (Chen et al.,

- 2024b), and Self-Play Critic (Chen et al., 2025a) explore autonomous feedback strategies. However, these methods lack unified, co-evolving frameworks for feedback generation and validation.

- 3 METHODOLOGY

- 3.1 THE SOCRATIC-ZERO FRAMEWORK

We introduce Socratic-Zero, a fully autonomous, co-evolutionary framework designed to bootstrap mathematical reasoning from a minimal set of seed problems, entirely without relying on external human-annotated data. As illustrated in Figure 3, the system operates as a self-improving loop among three agents: a Solver that learns to reason, a fixed Teacher that acts as an oracle for evaluation and problem refinement, and a Generator that learns to synthesize a curriculum.

[Figure 81]

[Figure 82]

Reasoning Trajectories

Solver

##### (a) Solver’s Online Optimization Teacher

[Figure 83]

[Figure 84]

[Figure 85]

For x terms: $x^2 - 6x = (x-3)^2 - 9$. The general circle equation is x^2. The answer is (3, -1).

I am trying to judge these answers based on the ground truth …

[Figure 86]

###### For Each Step

[Figure 87]

[Figure 88]

###### Evolve with DPO

Problem Set

I am trying to solve these problems …

[Figure 89]

[Figure 90]

I am trying to fix my problems with the verified results …

[Figure 91]

[Figure 92]

[Figure 93]

Data Pool

I am trying to generate new problems based on the solver’s performance …

[Figure 94]

Enhanced Problem Set

Verified Results

[Figure 95]

[Figure 96]

[Figure 97]

(b) Generator’s Offline Optimization

[Figure 98]

[Figure 99]

Generator

###### Generator Evolve with WSFT

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

Too Easy Too Hard

+

[Figure 107]

Problem Set Enhanced Problem Set

+

Generated Problem Set

Lower Weight

Distill/Lower Weight

Verified Results

I am trying to learn more from questions with moderate accuracy …

I am trying to generate better problems …

- Figure 3: Overview of the Socratic-Zero Framework. (a) Solver Evolving: The Solver attempts to solve problems and learns from preference pairs of correct and incorrect solutions via DPO, while the frozen Teacher strategically generates challenging problems based on Solver failures using fixed generation and evaluation functions. (b) Generator Evolving: The Generator distills the Teacher’s problem generation strategy using value-weighted supervised learning. Together, these create a self-improving loop where the curriculum dynamically evolves to maintain optimal challenge levels for the Solver’s current capabilities.

At each iteration t, the framework operates on a curriculum of problems and their reference solutions, denoted as Dt = {(q,yref)}. The core of Socratic-Zero is the co-evolution of the Solver and Generator under the guidance of the Teacher. The Solver is trained to solve problems from Dt. Its failures are then used by the Teacher to create new, targeted problems. The Generator, in turn, distills the Teacher’s strategy to produce a scalable curriculum. This process creates a dynamic curriculum that continuously adapts to the Solver’s evolving capabilities, ensuring the training signals remain maximally informative. The agents are formally defined as follows:

Agents in the Socratic-Zero Framework

- 1. Solver (S): An agent with a policy πθ

S

, parameterized by θS, which maps a problem q to a solution trajectory y. Its objective is to generate correct reasoning paths. At each iteration t, it improves by learning from preference feedback on its own attempts.

- 2. Teacher (T ): A fixed, high-capacity LLM that provides two deterministic oracle functions: (i) a verification function V (q,y) → {0,1}, which judges the correctness of a solution y for a

problem q; and (ii) a problem refinement function G(q,yfail) → (q′,yref′ ), which creates a new problem-solution pair by revising an original problem q based on a failed solution yfail.

- 3. Generator (G): An agent with a policy πθ

, parameterized by θG, that learns to mimic the Teacher’s refinement strategy. It maps a problem and a failed solution (q,yfail) to a new problem q′. It evolves to generate problems that are optimally challenging for the current Solver.

G

The curriculum expands based on the Solver’s mistakes. At iteration t, the set of Solver failures on the current curriculum Dt is collected:

(· | q),V (q,yS) = 0 . (1)

Ft = (q,yS) | (q,yref) ∈ Dt,yS ∼ πθ(t)

S

The Teacher refines each failure into a new, instructive problem-solution pair. The set of these new pairs, Dnew, is used to augment the curriculum for the next iteration:

#### . (2)

Dt+1 = Dt ∪ {G(q,yS) | (q,yS) ∈ Ft}

Dnew

The full co-evolutionary training procedure is detailed in Algorithm 1.

- 3.2 SOLVER TRAINING VIA ONLINE PREFERENCE OPTIMIZATION

The Solver’s policy πθ

S

is improved through online preference learning, leveraging the Teacher’s verification function V to create a feedback loop. For each problem q ∈ Dt, the Solver generates k solution attempts, {yS(i)}ki=1. These attempts are partitioned into a set of “winning” (correct) solutions Yw(q) and “losing” (incorrect) solutions Yl(q):

Yl(q) = {yS(i) | V (q,yS(i)) = 0}, (3) Yw(q) = {yS(i) | V (q,yS(i)) = 1} ∪ {yref | if ∀i,V (q,yS(i)) = 0}, (4)

where yref is the reference solution from the curriculum. If the Solver fails to generate any correct solution, the ground-truth solution serves as the sole winning example. This ensures a valid preference

pair (yw,yl), where yw ∈ Yw(q) and yl ∈ Yl(q), can always be constructed.

The Solver’s parameters θS are then updated using the Direct Preference Optimization (DPO) loss (Rafailov et al., 2023). This objective maximizes the likelihood of preferred solutions over rejected ones:

LDPO(θS;θref) = −Eq∼D

t,yw∼Yw(q),yl∼Yl(q) log σ β log

πθ

S

(yw | q) πθ

ref

(yw | q) − β log

πθ

S

(yl | q) πθ

ref

(yl | q)

,

(5) where πθ

ref

is a frozen reference policy (e.g., πθ

S

from the start of the iteration), β is a temperature hyperparameter, and σ is the sigmoid function.

- 3.3 GENERATOR TRAINING VIA OFFLINE VALUE-WEIGHTED DISTILLATION

To ensure scalable curriculum generation without perpetual reliance on the expensive Teacher, the Generator πθ

G

is trained to distill the Teacher’s problem refinement strategy. An effective curriculum should feature problems of desirable difficulty—challenging enough to be informative but not so difficult as to be unsolvable.

We formalize this concept with a utility function U(q′|πθ

S

) that scores a new problem q′ based on the current Solver’s performance. Let sq = k1 ki=1 V (q,yS(i)) be the success rate of the Solver πθ

S

over k attempts. The utility is defined by an unnormalized Gaussian centered at a target success rate µ:

U(q′|πθ

S

) = exp −

(sq′ − µ)2 2σ2

. (6)

We set µ = 0.5 to incentivize problems at the frontier of the Solver’s capabilities, with σ controlling the tolerance for deviation.

The Generator is trained via weighted supervised fine-tuning (WSFT) to mimic the Teacher’s generation of high-utility problems. The training data DG is constructed from the set of Solver failures Ft and the corresponding Teacher-refined problems from Dnew:

DG = {(q,yfail,q′) | (q,yfail) ∈ Ft,(q′,yref′ ) = G(q,yfail)}. The Generator’s objective is to maximize the utility-weighted log-likelihood of producing the Teacher’s refined problems:

LWSFT(θG) = −E(q,y

fail,q′)∼DG [U(q′|πθ

S

) · log πθ

G

(q′ | q,yfail)]. (7)

This objective steers the Generator towards producing problems that are optimally challenging for the current Solver, effectively internalizing the Teacher’s expert curriculum design principles.

- 4 EXPERIMENTS

- 4.1 EXPERIMENTAL SETUP

Models. We employed Qwen3-235B-A22B-Instruct-2507 (Yang et al., 2025) as the Teacher model to provide high-quality evaluation and curriculum generation. We used Qwen3-32B (Yang et al., 2025)

as the Generator to learn and distill the Teacher’s problem generation strategies. We conducted Solver experiments on multiple model architectures including Qwen3-8B-base, Qwen3-14B-base (Yang et al., 2025), and GLM4-9B-base (GLM et al., 2024) to demonstrate cross-model generalization. We compared strong baselines including Gemini-2.5-Pro-06-17 (Comanici et al., 2025), GPT5-0807global, and DeepSeek-v3.1-671B (DeepSeek-AI et al., 2025b) against our approach. For downstream evaluation of generated data quality, we fine-tuned DeepSeek-R1-Distill-Llama-8B (DeepSeek-AI et al., 2025a) as the student model.

Benchmarks. We used seven mathematical reasoning benchmarks for evaluation, including AMC (Cao et al., 2025), Minerva (Nagrani et al., 2025), MATH-500 (Hendrycks et al., 2021), GSM8K (Cobbe et al., 2021), Olympiad-Bench (He et al., 2024), AIME-2024, and AIME-2025. Additionally, we employed three general reasoning benchmarks to assess the transfer of mathematical reasoning improvements to broader cognitive abilities, namely BBEH (Kazemi et al., 2025), MMLU-Pro (Wang et al., 2024), and SuperGPQA (Team et al., 2025).

Curriculum Settings. The initial curriculum C0 contained 100 questions sampled from the MATH training set (Hendrycks et al., 2021) following specific diversity and difficulty criteria (detailed in Appendix F). All Solver models first underwent LoRA-based (Hu et al., 2021) SFT on a 1,500problem dataset of Level 5 difficulty. Key hyperparameters: k = 8 solution trajectories per problem, reward parameters µ = 0.5 and σ = 0.2, and training batches combined 100% new problems with 25% historical curriculum for replay.

Solver Evaluation. For each test question, we generated 32 solutions using zero-shot prompting with temperature 0.7. We determined correctness through a dual-verification mechanism combining rule-based answer extraction and semantic validation. We reported Mean@32 accuracy across all evaluations. Detailed evaluation protocols, including sampling strategies, answer extraction methods, and LLM judge configurations, are provided in Appendix G.

Baselines. Baselines. We employed two strong baselines for comparison. Static Augmentation (SA) follows traditional approaches via MetaMath (Yu et al., 2024) and WizardMath (Luo et al., 2023), augmenting training data with fixed synthetic questions generated offline without adaptive curriculum evolution. LLM2LLM (Lee et al., 2024) implements iterative self-training where models generate questions based on current failures and retrain on augmented datasets. Both baselines use identical SFT initialization for fair comparison.

Generator Evaluation. We prompted each generator with 1,000 SAND-Math (Zhang et al., 2025) seeds to produce 3 variants each, resulting in 3,000 total generated questions. We measured validity rate by having Qwen3-235B-A22B-Instruct-2507 attempt to solve each generated question under a 4,096-token, 600-s timeout constraint. We evaluated downstream utility by fine-tuning DeepSeek-R1Distill-Llama-8B on the QAs and measuring performance on mathematical reasoning benchmarks.

Infrastructure. We conducted training experiments on 8×NVIDIA H20 GPUs. We performed Teacher model inference using 16×AMD MI308X GPUs. Detail provided in Appendix D.

- 4.2 SOLVER RESULTS

Baseline Comparison. Table 1 shows Socratic achieves 56.1% average accuracy, outperforming Static Augmentation by +15.4 points and LLM2LLM by +15.2 points. Notable gains appear on competition problems: AIME-24 (+19.1) and AIME-25 (+16.5), demonstrating the advantages of DPO-based preference learning and adaptive curriculum generation.

Cross-Architecture Generalization. Table 2 validates that Socratic principles transcend specific model families. On GLM4-9B-base, Socratic Stage 3 achieves 52.3% average accuracy (+17.1 points over base model), with strong improvements on AIME benchmarks: AIME-25 (+20.4) and AIME-24 (+23.9). Similarly, on Qwen3-14B-base, Stage 3 reaches 60.3% (+17.3 points), demonstrating consistent effectiveness across different architectures and addressing fundamental reasoning capabilities.

Transfer to General Reasoning. Table 3 shows mathematical reasoning improvements transfer to broader cognitive abilities, with +6.02 points average improvement across BBEH, MMLU-Pro, and SuperGPQA benchmarks.

- Table 1: Solver Evaluation Results with different training methods. Results are reported on seven benchmarks (AMC, Minerva, MATH-500, GSM8K, Olympiad, AIME-25, AIME-24) and their average. Arrow values represent absolute point changes relative to Static Augmentation, where ↑ indicates improvement and ↓ indicates decline.

Training Method

Benchmark Datasets

Avg.

AMC Minerva MATH-500 GSM8K Olympiad AIME-25 AIME-24 Qwen3-8B-base

+ Zero-shot 32.5 31.3 48.8 63.4 24.1 4.2 5.1 29.9 + SFT 39.1 37.8 56.9 68.2 31.7 8.1 9.3 35.9

+ Static Augmentation 45.8 41.9 62.7 74.6 35.9 11.4 12.3 40.7 Qwen3-8B-base with LLM2LLM

- + Stage 1 41.6↓4.2 41.2↓0.7 53.1↓9.6 78.3↑3.7 32.4↓3.5 6.7↓4.7 8.9↓3.4 37.5↓3.2

- + Stage 2 43.2↓2.6 40.6↓1.3 54.9↓7.8 79.1↑4.5 33.8↓2.1 7.2↓4.2 9.1↓3.2 38.3↓2.4

- + Stage 3 44.9↓0.9 42.1↑0.2 66.8↑4.1 79.4↑4.8 34.6↓1.3 7.9↓3.5 10.4↓1.9 40.9↑0.2

Qwen3-8B-base with Socratic-Zero (Ours)

- + Stage 1 43.8↓2.0 39.4↓2.5 60.2↓2.5 69.7↓4.9 35.3↓0.6 10.6↓0.8 11.8↓0.5 38.7↓2.0

- + Stage 2 49.3↑3.5 40.7↓1.2 63.4↑0.7 71.8↓2.8 38.2↑2.3 12.9↑1.5 15.6↑3.3 41.7↑1.0

- + Stage 3 63.7↑17.9 52.4↑10.5 81.2↑18.5 87.3↑12.7 55.1↑19.2 24.6↑13.2 28.4↑16.1 56.1↑15.4

- Table 2: Cross-Model Generalization Results with different training stages. Each block corresponds to a model (GLM4-9B, Qwen3-14B). Results are reported on seven benchmarks (AMC, Minerva, MATH-500, GSM8K, Olympiad, AIME-25, AIME-24) and their average. Values with arrows represent absolute point changes relative to SFT for each model.

Training Method

Benchmark Datasets

Avg

AMC Minerva MATH-500 GSM8K Olympiad AIME-25 AIME-24 GLM4-9B-base

+Zero-shot 34.5 37.3 52.3 72.5 34.8 7.5 7.2 35.2 + SFT 38.4 44.8 63.8 77.2 41.3 15.1 19.3 42.8

- + Socratic Stage 1 39.4↑1.0 47.3↑2.5 67.9↑4.1 79.8↑2.6 43.4↑2.1 15.6↑0.5 24.0↑4.7 45.3↑2.5

- + Socratic Stage 2 42.3↑3.9 49.4↑4.6 68.1↑4.3 82.5↑5.3 45.5↑4.2 19.1↑4.0 25.3↑6.0 47.5↑4.7

- + Socratic Stage 3 47.5↑8.7 52.8↑8.0 73.8↑10.0 83.9↑6.7 49.4↑8.1 27.9↑12.8 31.1↑11.8 52.3↑9.5

Qwen3-14B-base

+Zero-shot 48.8 40.5 62.0 91.5 38.4 9.6 10.0 43.0 + SFT 61.3 51.8 71.5 92.2 47.3 18.1 20.3 51.8

- + Socratic Stage 1 62.9↑1.6 55.1↑3.3 74.6↑3.1 91.8↓0.4 52.5↑5.2 19.8↑1.7 21.7↑1.4 54.1↑2.3

- + Socratic Stage 2 65.4↑4.1 57.4↑5.6 76.7↑5.2 92.3↑0.1 54.2↑6.9 24.8↑6.7 23.3↑3.0 56.3↑4.5

- + Socratic Stage 3 70.0↑8.7 60.7↑8.9 80.2↑8.7 93.7↑1.5 58.3↑11.0 28.9↑10.8 30.1↑9.8 60.3↑8.5

- Table 3: Performance on general reasoning benchmarks with different training stages. Results are reported on three benchmarks (BBEH, MMLU-Pro, SuperGPQA) and their average. Values with arrows represent absolute point changes relative to zero-shot Qwen3-8B-base performance.

General Reasoning Benchmarks

Training Method

Avg.

BBEH MMLU-Pro SuperGPQA Qwen3-8B-Base

+ Zero-shot 7.68 50.00 24.73 27.47 Base Model with Socratic (Ours)

- + Stage 1 8.48↑0.80 55.71↑5.71 27.32↑2.59 30.50↑3.03

- + Stage 2 9.11↑1.43 59.29↑9.29 29.73↑5.00 32.71↑5.24

- + Stage 3 9.54↑1.86 60.89↑10.89 30.05↑5.32 33.49↑6.02

- 4.3 GENERATOR RESULTS

We assessed both the intrinsic quality of generated problems and their downstream training effectiveness, with Socratic-Generator-32B being compared against its base model and SOTA commercial large language models to determine whether strategic specialization can match the performance of much advanced larger models.

Table 5: Downstream Training Effectiveness with different generator models. Results are reported on seven benchmarks (AIME-24, AIME-25, AMC-23, GSM8K, MATH-500, Minerva, Olympiad) and their average. Values with arrows represent absolute point changes relative to Qwen3-32B baseline.

Benchmark Datasets

Avg.

AIME-24 AIME-25 AMC-23 GSM8K MATH-500 Minerva Olympiad DeepSeek-R1-Distill-Llama-8B

+ Zero-shot 5.8 8.3 42.5 72.2 52.4 15.3 23.0 32.75

Open-Sourced Generators Qwen3-32B 9.2 10.0 44.4 75.7 55.7 15.1 24.5 34.97 Qwen3-235B-A22B-Instruct-2507 12.5↑3.3 12.5↑2.5 47.5↑3.1 76.1↑0.4 57.8↑2.1 16.4↑1.3 23.6↓0.9 37.13↑2.16 DeepSeek-v3.1-671B 12.5↑3.3 11.7↑1.7 46.2↑1.8 76.4↑0.7 56.4↑0.7 16.5↑1.4 23.9↓0.6 36.62↑1.65

Advanced commercial Generators

Gemini-2.5-Pro-06-17 10.0↑0.8 15.0↑5.0 46.9↑2.5 78.1↑2.4 57.2↑1.5 16.0↑0.9 25.4↑0.9 37.20↑2.23 GPT5-0807-global 12.5↑3.3 13.3↑3.3 45.0↑0.6 76.8↑1.1 56.6↑0.9 15.5↑0.4 25.9↑1.4 36.62↑1.65 Grok-4 11.7↑2.5 12.5↑2.5 45.8↑1.4 76.3↑0.6 56.9↑1.2 15.9↑0.8 24.9↑0.4 37.01↑2.04 Claude-4.1-Opus 13.3↑4.1 13.8↑3.8 46.5↑2.1 77.3↑1.6 57.5↑1.8 16.7↑1.6 24.3↓0.2 37.63↑2.66

Socratic-Generator-32B 12.5↑3.3 13.3↑3.3 48.1↑3.7 77.6↑1.9 57.8↑2.1 18.4↑3.3 24.6↑0.1 37.72↑2.75

- 4.3.1 EVALUATION PROTOCOL

We adopted a standardized, three-stage evaluation pipeline to holistically assess both the intrinsic quality of generated problems and their extrinsic utility in downstream model training. The full procedure is formalized below.

- Step 1: Problem Generation. We prompted each generator with 1,000 seed problems from SANDMath (Zhang et al., 2025) and tasked with producing five augmented variants per seed, resulting in

- 3,000 total generated problems per model.

Step 2: Quality Assessment. We measured problem validity by prompting Qwen3-235B (Yang et al., 2025) — selected for its state-of-the-art mathematical reasoning capability and its role as the teacher model in the distillation framework — to solve each generated problem under strict constraints: a

- 4,096-token limit and a 600-second timeout. The Validity Rate was defined as the percentage of problems successfully solved within these bounds.

- Step 3: Student Evaluation. We used all valid question-answer (QA) pairs to fine-tune the student model, DeepSeek-R1-Distill-Llama-8B (DeepSeek-AI et al., 2025a). We evaluated Downstream Utility as the mean accuracy — average accuracy over 16 independent decoding runs per problem across seven diverse mathematical reasoning benchmarks.

- 4.3.2 PROBLEM QUALITY ASSESSMENT Table 4: Generator validity rates.

Generator Model Validity Rate (%)

Qwen3-32B 89.1 Qwen3-235B-A22B 95.1↑6.0 Gemini-2.5-Pro 94.2↑5.1 GPT5-global 95.8↑6.7 DeepSeek-v3.1-671B 96.5↑7.4 Grok4 95.7↑6.7 Claude-4.1-opus 96.9↑7.

Socratic-Generator-32B 95.6↑6.5

To evaluate the quality of the generated problems, we measure their Validity Rate — the percentage of problems solvable by a powerful model (Qwen3235B-A22B-Instruct-2507). As shown in Table 4, our specialized Socratic-Generator-32B generator achieves a remarkable 95.6% validity rate. This not only represents a substantial improvement over its base model Qwen3-32B but also rivals the performance of significantly larger models, including proprietary models like GPT5-0807-global, Gemini-2.5Pro-06-17. This demonstrates our co-evolutionary strategy effectively.

- 4.3.3 DOWNSTREAM EFFECTIVENESS

- Table 5 reports the downstream utility of each generator, measured by the performance of the finestudent model. The output from our Socratic-Generator-32B leads to a final student accuracy of

- 37.72%. Notably, this performance not only rivals that achieved using data from significantly larger models but also marginally surpasses (+0.59 points) the result from its own Teacher (Qwen3-235B), despite being over 20x smaller.

- Table 6: Ablation studies on the necessity of initial SFT and different strategies of reward functions. (a) Values with arrows represent absolute point changes relative to the previous stage within the same method. (b) Values with arrows represent absolute point changes relative to the Gaussian baseline. ρ represents solver success rate, µ represents target success rate, σ represents standard deviation in Gaussian reward function N(µ,σ), Ψρ(a,b) represents linear function Ψρ(a,b) = aρ + b.

(a) Ablation Study on Initial SFT (AIME-24) Method Score (%) ∆ (%)

(b) Ablation of Reward Functions (Benchmark Avg.) Reward Function Valid (%) Avg (%) ∆ (%)

Qwen3-8B-Base 9.64 Socratic-Zero (w/o SFT)

- N(µ = 0.5, σ = 0.2) (Ours) 89.9 35.72 -

- Ψρ(a = 0, b = 1) 89.4 35.52 ↓0.20

- Ψρ(a = 1, b = 0) 89.8 35.47 ↓0.25 Ψρ(a = −1, b = 1) 88.9 35.42 ↓0.30

- N(µ = 0.3, σ = 0.2) 89.5 35.32 ↓0.40

- N(µ = 0.4, σ = 0.2) 89.7 35.37 ↓0.35

- N(µ = 0.6, σ = 0.2) 89.7 35.50 ↓0.22

- N(µ = 0.7, σ = 0.2) 89.8 35.43 ↓0.29

- + Stage 1 11.67 ↑2.03

- + Stage 2 11.15 ↑1.51

- + Stage 3 11.98 ↑2.34

Socratic-Zero (w/ SFT)

- + Stage 1 13.44 ↑3.80

- + Stage 2 14.48 ↑4.84

- + Stage 3 28.02 ↑18.38

- 4.4 ABLATION STUDIES

We conducted two key ablation studies to validate our framework’s design choices, with results summarized in Table 6. The first study examines the necessity of initial supervised fine-tuning (SFT), while the second investigates different reward function formulations during reinforcement learning.

Ablation on Initial SFT Table 6a demonstrates the critical importance of initial supervised finetuning. Starting from the Qwen3-8B-Base model (9.64%), the version without SFT shows minimal improvements across all three training stages, reaching only 11.98% by Stage 3—a marginal gain of 2.34 percentage points. In stark contrast, the SFT-initialized model achieves substantial performance improvements, culminating in a remarkable 28.02% score at Stage 3, representing an 18.38 percentage point improvement over the base model. This 7.9× greater improvement highlights how SFT provides essential foundational capabilities that enable subsequent reinforcement learning stages to be dramatically more effective. The SFT phase likely equips the model with basic reasoning patterns and solution structures that serve as building blocks for more sophisticated reasoning developed during RL training.

Ablation on Reward Functions Table 6b compares various reward function formulations using benchmark average scores. Our Gaussian reward N(µ = 0.5,σ = 0.2) achieves the best performance (35.72%) while maintaining high validity (89.9%). We evaluated linear functions Ψρ(a,b) = aρ + b with different parameterizations, all of which underperformed our Gaussian approach by 0.20-0.30 percentage points. Similarly, varying the Gaussian mean parameter µ from 0.3 to 0.7 consistently yielded inferior results, with the largest performance drop (0.40 points) occurring at µ = 0.3. This suggests that the Gaussian formulation with µ = 0.5 provides an optimal balance between exploration and exploitation during policy optimization, while the moderate variance (σ = 0.2) allows sufficient reward signal differentiation without excessive noise that could destabilize training.

- 5 CONCLUSION AND FUTURE WORK

In this paper, we introduced Socratic-Zero, a multi-agent co-evolutionary framework where Solver, Teacher, and Generator agents bootstrap autonomous mathematical reasoning from minimal seed data. Our implementation demonstrates that a carefully designed learning mechanism can achieve remarkable performance without relying on massive external datasets, offering a viable path for developing powerful reasoning systems in resource-constrained scenarios. Extensive experiments show that our framework not only achieves state-of-the-art results on mathematical benchmarks but also exhibits strong generalization capabilities across diverse problem types and difficulty levels. While the complex agent dynamics currently lack a formal convergence analysis, future work will aim to establish this theoretical foundation and extend the framework’s applicability to broader domains including scientific discovery, real-world decision making, and complex system modeling.

REFERENCES

Rishabh Agarwal, Nino Vieillard, Yongchao Zhou, Piotr Stanczyk, Sabela Ramos, Matthieu Geist, and Olivier Bachem. On-policy distillation of language models: Learning from self-generated mistakes. In International Conference on Learning Representations, 2024.

Lang Cao, Chao Peng, Renhong Chen, Wu Ning, Yingtian Zou, and Yitong Li. Step guided reasoning: Improving mathematical reasoning using guidance generation and step reasoning, 2025.

Jiaqi Chen, Ruotian Ma, Bang Zhang, Peisong Wang, Zhaopeng Tu, Xiaolong Li, Kwan-Yee K. Wong, and Xiaodan Liang. Spc: Evolving self-play critic via adversarial games for llm reasoning, 2025a.

Yiming Chen, Zhenhua Liu, Xiang Yue, and Wenpeng Yin. Distillm: Towards streamlined distillation for large language models, 2024a.

Yue Chen, Minghua He, Fangkai Yang, Pu Zhao, Lu Wang, Yu Kang, Yifei Dong, Yuefeng Zhan, Hao Sun, Qingwei Lin, Saravan Rajmohan, and Dongmei Zhang. Warriormath: Enhancing the mathematical ability of large language models with a defect-aware framework, 2025b.

Zixiang Chen, Yihe Deng, Huizhuo Yuan, Kaixuan Ji, and Quanquan Gu. Self-play fine-tuning converts weak language models to strong language models, 2024b.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems, 2021.

Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities, 2025.

DeepSeek-AI, Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning, 2025a.

DeepSeek-AI, Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, et al. Deepseek-v3 technical report, 2025b.

Team GLM, Aohan Zeng, Bin Xu, Bowen Wang, Chenhui Zhang, Da Yin, Dan Zhang, Diego Rojas, Guanyu Feng, Hanlin Zhao, et al. Chatglm: A family of large language models from glm-130b to glm-4 all tools, 2024.

Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Leng Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, et al. Olympiadbench: A challenging benchmark for promoting agi with olympiad-level bilingual multimodal scientific problems, 2024.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset, 2021.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models, 2021.

Chenghua Huang, Zhizhen Fan, Lu Wang, Fangkai Yang, Pu Zhao, Zeqi Lin, Qingwei Lin, Dongmei Zhang, Saravan Rajmohan, and Qi Zhang. Self-evolved reward learning for llms, 2025a.

Chengsong Huang, Wenhao Yu, Xiaoyang Wang, Hongming Zhang, Zongxia Li, Ruosen Li, Jiaxin Huang, Haitao Mi, and Dong Yu. R-zero: Self-evolving reasoning llm from zero data, 2025b.

Mehran Kazemi, Bahare Fatemi, Hritik Bansal, John Palowitch, Chrysovalantis Anastasiou, Sanket Vaibhav Mehta, Lalit K. Jain, Virginia Aglietti, Disha Jindal, Peter Chen, et al. Big-bench extra hard, 2025.

Nicholas Lee, Thanakul Wattanawong, Sehoon Kim, Karttikeya Mangalam, Sheng Shen, Gopala Anumanchipalli, Michael W. Mahoney, Kurt Keutzer, and Amir Gholami. Llm2llm: Boosting llms with novel iterative data enhancement, 2024.

Xuyang Liu, Zichen Wen, Shaobo Wang, Junjie Chen, Zhishan Tao, Yubo Wang, Xiangqi Jin, Chang Zou, Yiyu Wang, Chenfei Liao, Xu Zheng, Honggang Chen, Weijia Li, Xuming Hu, Conghui He, and Linfeng Zhang. Shifting ai efficiency from model-centric to data-centric compression, 2025.

Haipeng Luo, Qingfeng Sun, Can Xu, Pu Zhao, Jianguang Lou, Chongyang Tao, Xiubo Geng, Qingwei Lin, Shifeng Chen, and Dongmei Zhang. Wizardmath: Empowering mathematical reasoning for large language models via reinforced evol-instruct. arXiv preprint arXiv:2308.09583, 2023.

Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, Shashank Gupta, Bodhisattwa Prasad Majumder, Katherine Hermann, Sean Welleck, Amir Yazdanbakhsh, and Peter Clark. Self-refine: Iterative refinement with self-feedback, 2023.

Subhabrata Mukherjee, Arindam Mitra, Ganesh Jawahar, Sahaj Agarwal, Hamid Palangi, and Ahmed Awadallah. Orca: Progressive learning from complex explanation traces of gpt-4, 2023.

Subhojyoti Mukherjee, Viet Dac Lai, Raghavendra Addanki, Ryan Rossi, Seunghyun Yoon, Trung Bui, Anup Rao, Jayakumar Subramanian, and Branislav Kveton. Learning to clarify by reinforcement learning through reward-weighted fine-tuning, 2025.

Arsha Nagrani, Sachit Menon, Ahmet Iscen, Shyamal Buch, Ramin Mehran, Nilpa Jha, Anja Hauth, Yukun Zhu, Carl Vondrick, Mikhail Sirotenko, et al. Minerva: Evaluating complex video reasoning, 2025.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model, 2023.

Markus Stiennon, Long Ouyang, Jeff Wu, Rewon Child, David Amodei, Dario Amodei, David F. M. Brown, Benjamin Mann, Paul Christiano, Peter Chen, and John Schulman. Learning to follow instructions with human feedback, 2020.

P Team, Xinrun Du, Yifan Yao, Kaijing Ma, Bingli Wang, Tianyu Zheng, King Zhu, Minghao Liu, Yiming Liang, Xiaolong Jin, et al. Supergpqa: Scaling llm evaluation across 285 graduate disciplines, 2025.

Shaobo Wang, Yicun Yang, Zhiyuan Liu, Chenghao Sun, Xuming Hu, Conghui He, and Linfeng Zhang. Dataset distillation with neural characteristic function: A minmax perspective. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 25570–25580, June 2025a.

Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, et al. Mmlu-pro: A more robust and challenging multi-task language understanding benchmark, 2024.

Zhiyuan Wang, Yuxiao Chen, Chao Yu, Yifan Zhang, and Jun Wang. A survey of reinforcement learning-driven knowledge distillation, 2025b.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report, 2025.

Longhui Yu, Weisen Jiang, Han Shi, Jincheng Yu, Zhengying Liu, Yu Zhang, James T. Kwok, Zhenguo Li, Adrian Weller, and Weiyang Liu. Metamath: Bootstrap your own mathematical questions for large language models, 2024.

Shaoxiong Zhan, Yanlin Lai, Ziyu Lu, Dahua Lin, Ziqing Yang, and Fei Tan. Mathsmith: Towards extremely hard mathematical reasoning by forging synthetic problems with a reinforced policy, 2025.

Yifan Zhang, Yifan Luo, Yang Yuan, and Andrew Chi-Chih Yao. Sand-math: Using llms to generate novel, difficult and useful mathematics questions and answers, 2025.

Yue Zhang, Tianxiang Sun, Xiangyang Liu, Hang Yan, and Xipeng Qiu. Dilm: Distilling dataset into language model for text-level dataset distillation, 2024.

Andrew Zhao, Yiran Wu, Yang Yue, Tong Wu, Quentin Xu, Yang Yue, Matthieu Lin, Shenzhi Wang, Qingyun Wu, Zilong Zheng, and Gao Huang. Absolute zero: Reinforced self-play reasoning with zero data, 2025.

Liang Zhao, Yuhui Shi, Zhiyong Feng, and Shuo Wang. Data distillation: A survey, 2023.

- A PROMPTS

- A.1 SOLVER REASONING PROMPT

Solver Mathematical Reasoning Prompt

You are an IMO gold medalist solving a computational math competition problem. Understand: Restate the problem mathematically. Identify knowns, unknowns, and constraints. Plan: Choose an efficient method, show clear logic. Execute: Show all key steps — algebra, number theory, or combinatorics. No skipped calculations. Verify: Check with small cases, reverse substitution, or estimation. Conclude with the exact answer in LaTeX: \[\boxed{< answer >}\] Given Problem: {question}

- A.2 TEACHER EVALUATION PROMPT

Teacher Solution Grading Prompt

You are a professional math teacher responsible for grading and error analysis. Grading criteria: Focus on final answer correctness, use reference when provided, provide concise error analysis for incorrect answers. Return JSON format:

{

"correct_answers": ["correct answer 1", "correct answer 2"], "incorrect_answers": [

{"answer": "incorrect answer", "analysis": "brief error analysis"} ]

} Problem: {question} | Reference: {reference_info} | Student answers: {student_answers}

- A.3 TEACHER GENERATION PROMPT

Teacher Problem Enhancement Prompt

You are a math problem enhancement expert specializing in competition-style mathematics. Generate enhanced problems based on student error analysis with complete solutions. Requirements: Generate enhanced problem, provide detailed solution, ensure solvability and correctness. Enhancement principles: Target specific error points, maintain mathematical essence, help avoid similar errors. Return JSON format:

{

"enhanced_question": "enhanced problem content", "solution": "detailed solution steps", "answer": "final answer"

} Original: {original_question} | Error analysis: {error_analysis}

- A.4 STATIC AUGMENTATION BASELINE PROMPTS

Static Augmentation Evolution Prompts

Upward Evolution: Step 1: Identify elements that can increase complexity. Step 2: Plan to modify at least three components. Step 3: Implement rewritten instruction. Step 4: Review and provide final version.

Downward Evolution: Step 1: Identify elements that can decrease complexity. Step 2: Plan to simplify at least three components. Step 3: Implement easier version. Step 4: Review and provide final simplified version. Format: Step 1 #Elements#: | Step 2 #Plan#: | Step 3 #Rewritten#: | Step 4 #Final#:

- B IMPLEMENTATION DETAILS

We conducted training experiments on 8×NVIDIA H20 GPUs with the following configuration: GPU Memory: 96GB HBM3 per GPU - Total Training Memory: 768GB - Interconnect: NVLink 4.0 - Storage: High-speed NVMe SSD arrays for dataset caching - Network: InfiniBand for distributed training coordination

The training infrastructure utilized mixed-precision training (FP16) with gradient checkpointing to optimize memory usage. We employed distributed training using PyTorch’s DistributedDataParallel with NCCL backend for efficient gradient synchronization across GPUs.

- B.1 TRAINING HYPERPARAMETERS

We provide the complete hyperparameter settings for all components of the Socratic-Zero framework in Table 7.

Table 7: Hyperparameters used in Socratic-Zero framework.

Component Parameter Value

Solver SFT Training

Learning rate 5e-5 Per-device batch size 2 Gradient accumulation steps 4 Maximum sequence length 2048 LoRA rank (r) 64 LoRA alpha (α) 128 LoRA dropout 0.1 Number of epochs 1

Solver DPO Training

Learning rate 1e-6 – 5e-6 Per-device batch size 2 Gradient accumulation steps 4 – 16 Maximum sequence length 2048 Maximum training steps 10 – 200 DPO regularization (β) 0.05 – 0.2 Warmup steps 2 – 20 Optimizer AdaFactor Weight decay 0.01 Maximum gradient norm 1.0

Generator Training

Learning rate 1e-5 Per-device batch size 1 Gradient accumulation steps 8 Maximum sequence length 2048 LoRA rank (r) 64 LoRA alpha (α) 128 Number of epochs 2

Curriculum Parameters

Solutions per problem (k) 8 reward mean (µ) 0.5 reward std (σ) 0.2 Historical replay ratio 25%

Evaluation Settings

Sampling temperature 0.7 Number of samples 32 Token limit for validity check 4096

C CURRICULUM EVOLUTION DETAILS

This appendix provides a detailed description of the mechanisms for curriculum evolution, including the problem categorization and adaptive generation strategies that guide the learning process.

- C.1 PROBLEM CATEGORIZATION BY SOLVER PERFORMANCE

To effectively manage curriculum difficulty, we dynamically categorize each problem q ∈ Dt based on the current Solver’s performance. The Solver πθ

makes k attempts {yS(i)}ki=1 on each problem. We then calculate the Solver’s success rate:

S

k

1 k

V (q,yS(i)), (8)

sq ≜

i=1

where V is the Teacher’s verification function. This metric allows us to partition the curriculum Dt into three distinct zones:

- 1. Mastered Zone (Dmastered = {q | sq = 1}): Problems the Solver consistently solves correctly. These problems form a solid foundation and are used as a basis for generating more challenging variants.
- 2. Learning Zone (Dlearning = {q | 0 < sq < 1}): Problems the Solver can solve intermittently. This zone represents the optimal frontier for learning, where the Solver has emerging competence but requires further practice to achieve mastery.
- 3. Too Difficult Zone (Ddifficult = {q | sq = 0}): Problems the Solver consistently fails to solve. These problems are temporarily set aside from the generation process to avoid creating tasks far beyond the Solver’s current capabilities.

- C.2 ZONE-ADAPTIVE PROBLEM GENERATION

New problems are strategically generated by the Teacher’s refinement function, G, using problems from the Mastered and Learning zones. This ensures that curriculum expansion remains within the Solver’s zone of proximal development. The generation strategy is adapted to the source category:

- • From the Learning Zone: For a problem q ∈ Dlearning, the Teacher is prompted with a

specific failed solution attempt, yfail. The new problem (q′,yref′ ) = G(q,yfail) is designed to target the specific reasoning error exhibited in yfail, thereby helping the Solver overcome its weaknesses.

- • From the Mastered Zone: For a problem q ∈ Dmastered, the Teacher is prompted with

a successful solution, ysucc. The new problem (q′,yref′ ) = G(q,ysucc) is a more complex variant, designed to push the boundaries of the Solver’s established competence.

The set of newly generated problems Dnew is thus composed of variants from both zones, creating a balanced and targeted curriculum expansion. Problems from Ddifficult are explicitly excluded from this generation process to prevent counterproductive increases in difficulty.

- C.3 DYNAMIC RECATEGORIZATION OF PROBLEMS

As the Solver’s policy πθ

is updated at each iteration, its performance on existing problems changes. Consequently, all problems in the curriculum are periodically re-evaluated and recategorized. This dynamic process ensures the curriculum remains perfectly attuned to the Solver’s evolving skill set. The typical transitions between zones are:

S

Ddifficult(t) → Dlearning(t+1) (as capability improves) (9) Dlearning(t) → Dmastered(t+1) (as skills are consolidated) (10)

This recategorization mechanism allows problems that were once too difficult to re-enter the active learning and generation pool once the Solver is ready, ensuring a continuous and adaptive learning trajectory.

- D TEACHER MODEL INFRASTRUCTURE

The Teacher model (Qwen3-235B-A22B-Instruct-2507) requires substantial computational resources for curriculum generation and solution evaluation. We deployed the model using a distributed inference architecture to meet the throughput demands of the co-evolutionary training process.

We distributed the Teacher model across 16 AMD MI308X GPUs, each equipped with 192GB HBM3 memory, providing a total of 3,072GB aggregate memory. This configuration enables concurrent processing of curriculum generation requests while maintaining inference consistency across the framework.

To ensure system reliability and scalability, we implemented a multi-endpoint architecture with automatic load balancing and failover mechanisms. We configured the inference service with connection pooling (50 concurrent connections per endpoint) and exponential backoff retry policies to handle high request volumes during training.

We optimized key performance parameters for the mathematical reasoning domain: request timeouts of 600 seconds accommodate complex problem generation, while a 4,096-token limit ensures efficient solution evaluation. Batch processing utilizes 32 concurrent workers to maximize throughput during curriculum evolution phases.

- E TEACHER-GENERATED PROBLEM ENHANCEMENT

We provide examples of how the Teacher model enhances problems based on Solver failures. The following demonstrates the progression from original problems to ly-targeted enhanced versions.

- E.1 EXAMPLE 1: RATIONAL INEQUALITY ENHANCEMENT

|Original Problem:<br><br>Find all real numbers x satisfying 2x−5<br><br>x+3 ≥ 2. (Give your answer in interval notation.)|
|---|

|Enhanced Problem (Round 3):<br><br>Find all real numbers x satisfying 2x−5<br><br>x2−9 + x+31 ≤ (x4x−+13)2 . (Give your answer in interval notation.)<br><br>|
|---|

Enhancement Analysis: The enhancement introduces multiple complexity factors: (1) factored denominators requiring domain analysis, (2) multiple rational terms requiring common denominators,

- (3) squared terms in denominators, and (4) more complex algebraic manipulation. The enhanced problem targets common student errors in rational inequality solving while maintaining the core mathematical concepts.

- E.2 EXAMPLE 2: NUMBER THEORY ENHANCEMENT

|Original Problem:<br><br>Find the greatest common divisor of 10! + 6 and 11! + 14.|
|---|

|Enhanced Problem:<br><br>Find the greatest common divisor of 12! + 8 and 13! + 26, where the second number can be written as 13 · 12! + 26.|
|---|

Enhancement Analysis: The enhancement maintains the GCD structure while increasing numerical complexity and requiring students to recognize the relationship between consecutive factorials, targeting errors in modular arithmetic applications.

- F SEED SELECTION PROTOCOL

The selection of initial seed problems is critical for establishing an effective curriculum foundation. We employed a systematic approach to ensure the seed set provides appropriate difficulty, comprehensive coverage, and sufficient diversity for subsequent curriculum evolution.

Difficulty Alignment We selected seed problems to match the base model’s capability range to ensure productive learning dynamics. We drew problems from MATH dataset Levels 2-4, which empirically provide optimal challenge levels for our base models. Specifically, we excluded Level 1 problems (too easy, leading to trivial curriculum generation) and Level 5 problems (too difficult,

Table 8: Seed Problem Distribution Across Mathematical Domains

Subject Area Count Representative Topics Algebra 15 Linear/quadratic equations, inequalities, functions Number Theory 15 Divisibility, modular arithmetic, prime factorization Geometry 15 Coordinate geometry, trigonometry, area/volume calculations Combinatorics 15 Counting principles, permutations, probability Precalculus 15 Complex numbers, sequences, polynomial analysis Intermediate Algebra 15 Advanced algebraic manipulation, systems Prealgebra 10 Foundational arithmetic and basic algebraic concepts Total 100 Comprehensive mathematical reasoning coverage

resulting in universal failure and poor learning signals). Pre-filtering involved evaluating candidate problems with the base model using 8 solution attempts; we retained problems with success rates between 10-70% to ensure neither complete failure nor trivial success.

Domain Coverage To ensure comprehensive mathematical reasoning development, we sampled seed problems across all seven MATH subject areas with balanced representation as shown in Table 8: This distribution ensures that curriculum evolution can target weaknesses across diverse mathematical domains rather than overfitting to specific problem types.

Diversity Assurance Within each subject area, we selected problems to maximize methodological diversity. We employed clustering based on solution approach similarity (using embedding representations of problem statements) and selected problems from different clusters to ensure varied reasoning patterns. Additionally, we explicitly included problems requiring different mathematical tools to promote comprehensive skill development.

Quality Control We subjected all candidate problems to rigorous quality verification through a multi-stage process:

Quality Control Pipeline

- 1. Clarity Check: Problems must have unambiguous statements and well-defined solution paths
- 2. Answer Verification: We validated reference solutions by the Teacher model with multiple independent attempts
- 3. Value: Problems must demonstrate clear learning objectives and avoid trick questions or overly specialized knowledge
- 4. Contamination Avoidance: We excluded seed problems from all evaluation benchmarks to prevent data leakage

This systematic selection process ensures that the initial curriculum C0 provides a robust foundation for the co-evolutionary training dynamics while maintaining the diversity necessary for effective curriculum expansion.

- G EVALUATION PROTOCOL DETAILS

Mean@32 Sampling Strategy The Mean@32 evaluation metric represents the average accuracy across 32 independent solution attempts per problem. For each test problem, we generated 32 distinct solutions using temperature-based sampling (T=0.7) with top-p nucleus sampling (p=0.9) as specified in Table 7. This approach provides robust performance estimates by capturing the model’s consistency and reliability across multiple attempts.

We employed the sampling process using zero-shot prompting without few-shot examples to ensure unbiased evaluation. We generated each of the 32 solutions independently with different random seeds, preventing potential correlation effects. The final accuracy is computed as the proportion

of correct solutions among the 32 attempts, providing a more stable performance measure than single-shot evaluation.

MathRule Answer Extraction MathRule is a rule-based tool designed to extract and standardize final numerical answers from mathematical solution text. The tool employs pattern matching to identify answer indicators such as “Therefore,” “Thus,” “The answer is,” and LaTeX boxed expressions like \boxed{}.

The extraction process involves: (1) Locating answer indicators within the solution text, (2) Parsing mathematical expressions using regex patterns for common formats (fractions, decimals, integers,

algebraic expressions), (3) Standardizing representations (e.g., converting 12 to 0.5 when appropriate),

- (4) Handling multiple answer formats and selecting the most confident extraction based on contextual cues.

MathRule achieves high precision in answer extraction while maintaining robustness to variations in solution formatting and mathematical notation styles.

LLM Judge Configuration The Teacher model (Qwen3-235B-A22B-Instruct-2507) serves as an LLM judge for semantic validation when rule-based extraction is insufficient or ambiguous. The judge evaluates both numerical correctness and reasoning validity using structured prompts.

We instructed the evaluation prompt to: (1) Verify the final numerical answer against the expected result, (2) Assess the logical coherence of the reasoning steps, (3) Identify any mathematical errors or invalid assumptions, (4) Provide binary correctness judgments with brief justification.

We ensured judge reliability through temperature 0.1 sampling for consistent evaluations and validation against human expert annotations on a subset of problems. The dual-verification approach (MathRule + LLM judge) provides reliable automated assessment for large-scale evaluation.

- H PSEUDO CODE OF SOCRATIC-ZERO The pseudo code of Socratic-Zero is provided in Algorithm 1. Algorithm 1 Socratic-Zero Co-evolutionary Learning

- 1: Require: Initial curriculum D0, initial Solver parameters θS(0), initial Generator parameters θG(0), fixed Teacher T (V,G), total iterations T, attempts per problem k.
- 2: Initialize Solver πθ

S

← πθ(0)

S

and Generator πθ

G

← πθ(0)

G

.

- 3: for t = 0 to T − 1 do

▷ Phase 1: Online Solver Evolution

- 4: Set reference policy πθ

ref ← πθ(t)

S

.

- 5: Initialize preference data Pt ← ∅ and failure set Ft ← ∅.
- 6: for each problem q in curriculum Dt do
- 7: Generate k solution attempts {yS(i)}ki=1 ∼ πθ(t)

S

(· | q).

- 8: Construct winning set Yw(q) and losing set Yl(q) using Teacher verifier V .
- 9: Collect preference pairs: Pt ← Pt ∪ {(yw,yl) | yw ∈ Yw(q),yl ∈ Yl(q)}.
- 10: Collect failures: Ft ← Ft ∪ {(q,yl) | yl ∈ Yl(q)}.
- 11: end for
- 12: Update Solver parameters θS(t+1) ← Adam(∇θ

S

LDPO(θS(t);Pt,θref)).

▷ Phase 2: Offline Generator Evolution & Curriculum Expansion

- 13: Generate new problem-solution pairs with the Teacher: Dnew ← {G(q,yfail) | (q,yfail) ∈ Ft}.
- 14: Construct Generator training data DG ← ∅.
- 15: for each new problem (q′,yref′ ) generated from (q,yfail) do
- 16: Estimate utility U(q′|πθ(t+1)

S

) via rollouts with the updated Solver.

- 17: Add weighted training example to dataset: DG ← DG ∪ {(q,yfail,q′,U(q′))}.
- 18: end for
- 19: Update Generator parameters θG(t+1) ← Adam(∇θ

G

LWSFT(θG(t);DG)).

▷ Phase 3: Curriculum Update

- 20: Augment the curriculum for the next iteration: Dt+1 ← Dt ∪ {(q′,yref′ ) | (q′,yref′ ) ∈ Dnew}.
- 21: end for
- 22: Output: Trained Solver policy πθ(T)

S

and Generator policy πθ(T)

G

.

- I PROBLEM QUALITY CONTROL MECHANISM

To ensure curriculum integrity and prevent the propagation of erroneous problems, we implemented a comprehensive quality control mechanism that monitors problem validity through Solver performance feedback and automated verification.

Teacher Self-Verification Protocol When the Teacher model evaluates Solver attempts and finds that all k = 8 solutions for a given problem are incorrect (success rate jp = 0), this triggers an automatic quality verification process. The system recognizes that universal failure may indicate either: (1) the problem exceeds current Solver capability (expected behavior), or (2) the problem itself or its reference solution contains errors (quality issue).

The Teacher model performs self-verification by re-examining both the problem statement and its originally provided reference solution. This involves: (1) Re-solving the problem independently with temperature 0.1 for consistency, (2) Cross-validating the reference solution against the new solution attempt, (3) Checking for mathematical consistency, ambiguous problem statements, or computational errors, (4) Verifying that the problem has a unique, well-defined solution.

Problem Filtering and Exclusion We immediately flagged and excluded problems that fail the self-verification process from further curriculum evolution. Specifically, we discarded problems if: (1) The Teacher cannot reproduce its own reference solution, (2) Multiple valid interpretations of the problem statement exist, (3) Computational errors are detected in the reference solution, (4) The problem lacks sufficient information for a unique solution.

Table 9: Solver Mean Reward Evolution Across Training Stages

Stage S1 S2 S3 Trend Mean Reward (%) 52.1 48.7 50.1 ↓ then ↑

Table 10: Generator Reward Distribution Analysis

Stage S1 S2 S3 Stability High Reward Problems (%) 50.7 49.4 50.2 Stable Target Range (45-55%) ✓ ✓ ✓ Maintained

We removed discarded problems from the active curriculum Ct and they do not contribute to subsequent Solver training or Generator learning. This prevents the accumulation of low-quality problems that could degrade training effectiveness or introduce systematic biases.

MathRule Integration for Contamination Minimization The integration of MathRule answer extraction serves as an additional quality control layer by providing objective, rule-based verification independent of LLM judgment. When MathRule successfully extracts a clear numerical answer from the Solver’s solution, this extraction is compared against the reference answer using standardized formats.

This dual-verification approach (MathRule + Teacher evaluation) minimizes contamination from: (1) LLM judge inconsistencies or biases, (2) Format-related misinterpretations, (3) Numerical precision issues, (4) Ambiguous answer representations.

Problems where MathRule and Teacher evaluations consistently disagree trigger additional quality review, as such disagreements often indicate underlying issues with problem clarity or reference solution accuracy.

Feedback-Driven Quality Monitoring The system continuously monitors curriculum quality through Solver performance patterns. We flagged problems that consistently produce anomalous results—such as sudden performance drops across multiple Solver variants or inconsistent difficulty ratings—for manual review or automatic exclusion.

This feedback-driven approach ensures that quality control adapts to emerging issues and maintains curriculum integrity throughout the co-evolutionary training process, preventing the accumulation of problematic content that could compromise learning effectiveness.

- J CURRICULUM STABILITY AND DIVERSITY ANALYSIS

We analyzed the curriculum evolution dynamics across two dimensions: difficulty progression and problem diversity preservation.

Solver Performance Evolution Table 9 tracks the Solver’s mean reward (correctness rate) across training rounds, revealing adaptive curriculum difficulty.

Solver Performance Evolution Table 9 tracks the Solver’s mean reward (correctness rate) across training stages, revealing adaptive curriculum difficulty.

The Solver exhibits characteristic performance decline from Stage 1 (52.1%) to Stage 2 (48.7%) followed by recovery in Stage 3 (50.1%) as shown in Table 9. This pattern reflects adaptive curriculum generation where the Teacher progressively increases difficulty faster than Solver capability initially improves, then the Solver begins adapting to enhanced curriculum complexity.

Generator Stability Table 10 examines reward distribution in Generator training.

As demonstrated in Table 10, the Generator maintains remarkable stability with high-reward problems consistently around 50%, fluctuating within only 1.3% range. This indicates successful learning of the optimal difficulty zone defined by the Gaussian reward function with parameters µ = 0.5 and σ = 0.2 as specified in Table 7.

Generator Stability Table 10 examines reward distribution in Generator training.

As demonstrated in Table 10, the Generator maintains remarkable stability with high-reward problems consistently around 50%, fluctuating within only 1.3% range. This indicates successful learning of the optimal difficulty zone defined by the Gaussian reward function with parameters µ = 0.5 and σ = 0.2 as specified in Table 7.

Problem Diversity Three key mechanisms ensure curriculum diversity throughout training:

Multi-domain initialization: The 100 seed problems span all 7 MATH subjects (Algebra, Number Theory, Geometry, etc.) across difficulty levels 2-4 as detailed in Table 8, providing diverse starting points for curriculum evolution.

High-temperature sampling: We employed temperature 0.8-0.9 sampling at three critical stages: (1) Solver trajectory generation during curriculum advancement, (2) Teacher error analysis for varied failure interpretation, and (3) Teacher problem generation for diverse enhancement strategies.

Compounding diversity effects: Multi-domain seeds combined with stochastic sampling create diverse failure patterns, while high-temperature generation ensures varied problem formulations even from similar error patterns.

- K GENERALIZABILITY OF PROBLEM GENERATION CAPABILITIES

A key question emerging from our work is whether the Generator’s learned problem creation abilities can transfer to domains beyond mathematical reasoning. The value function and curriculum evolution mechanisms developed in Socratic-Zero are domain-agnostic in principle, suggesting potential for broader applicability.

The Generator learns fundamental skills in difficulty calibration, error pattern recognition, and targeting that may generalize across reasoning domains. For instance, the ability to identify when a problem is “appropriately challenging” (around 50% success rate as shown in Table 10) represents a meta-cognitive skill applicable to logical reasoning, scientific problem-solving, or even creative tasks. The Gaussian reward function with µ = 0.5 and σ = 0.2 (Table 7) creates a transferable framework for difficulty calibration that could adapt to other domains by adjusting the target success rate parameter.

Our Generator’s superior performance compared to much larger models, achieving 37.72% downstream utility versus 37.13% from Qwen3-235B-A22B (Table 5), demonstrates that strategic specialization can outperform raw parameter scaling. This suggests that domain-specific Generator training could be effective across various reasoning domains without requiring massive computational resources.

However, domain transfer would require careful adaptation of the Teacher’s evaluation capabilities and problem generation templates. Mathematical reasoning benefits from relatively objective correctness criteria with our dual-verification approach (MathRule + LLM judge) achieving 94.2% agreement with human experts, while other domains may require more nuanced evaluation frameworks. The seed selection protocol detailed in Table 8, which ensures balanced coverage across seven mathematical domains, provides a template for systematic domain expansion that could be adapted to physics, computer science, or other reasoning areas.

Future work should investigate whether a Generator trained on mathematical problems can effectively create challenging problems in adjacent domains like physics or computer science, potentially through few-shot adaptation or domain-specific fine-tuning leveraging the value learning mechanisms demonstrated in our framework.

- L FRAMEWORK SCALABILITY AND EXTENSIBILITY

The modular architecture of Socratic-Zero demonstrates strong potential for scalability and extension across multiple dimensions. The clear separation between Solver, Teacher, and Generator roles enables independent scaling and optimization of each component, as evidenced by our successful deployment across different computational configurations detailed in Table 7.

The framework’s extensibility is particularly evident in its ability to accommodate different model architectures and scales. Our cross-model validation demonstrates consistent performance improvements: Qwen3-8B achieves 56.1% average accuracy (+20.2 points), while similar gains are observed on GLM4-9B and Qwen3-14B architectures (Table 1). This cross-architecture consistency suggests the co-evolutionary principles transcend specific model families and could readily incorporate emerging architectures or specialized reasoning models.

The curriculum evolution mechanism shows robust scalability properties. Starting from just 100 seed problems (Table 8), the system generates thousands of ly valuable problems while maintaining quality, with our Generator achieving 95.6% validity rate compared to 89.1% from the base Qwen3-32B model (Table 4). This demonstrates that the framework can scale curriculum generation without proportional increases in seed data requirements.

Multi-domain extension represents another promising direction supported by our balanced seed distribution across seven mathematical domains. The current mathematical focus could expand to encompass multiple reasoning domains simultaneously, with domain-specific Teachers providing specialized curriculum generation while sharing the underlying co-evolutionary dynamics. The reward distribution analysis (Table 10) shows stable performance across training rounds, indicating the framework’s robustness to curriculum expansion.

The framework also supports hierarchical scaling, where multiple Solver-Generator pairs could operate at different difficulty levels or specialization areas, coordinated by higher-level meta-learning mechanisms. The oscillatory convergence patterns observed in Table 9 suggest natural synchronization points where multiple agents could coordinate their learning phases.

- M CONVERGENCE AND THEORETICAL FOUNDATIONS

The theoretical understanding of multi-agent co-evolutionary learning remains an open challenge with significant implications for system reliability and predictability. Our empirical observations provide crucial insights into the convergence behavior of such systems.

The oscillatory convergence patterns documented in Table 9 reveal characteristic dynamics: Solver performance declines from R1 (60.12%) to R4 (48.7%) followed by recovery in R5 (50.1%). This pattern reflects adaptive curriculum generation where the Teacher progressively increases difficulty faster than Solver capability initially improves, then the Solver adapts to enhanced curriculum complexity. These bounded oscillations suggest the system reaches dynamic equilibria rather than static optima.

Complementing this, the Generator maintains remarkable stability with high-reward problems consistently around 50%, fluctuating within only 1.3% range across training rounds (Table 10). This stability indicates successful learning of the optimal difficulty zone defined by the Gaussian reward function with µ = 0.5 and σ = 0.2 (Table 7), providing empirical evidence for convergence to ly meaningful equilibria.

The cross-architecture consistency observed in Table 1, where similar improvement patterns emerge across Qwen3-8B, GLM4-9B, and Qwen3-14B models, suggests robust system-level properties that transcend specific model architectures. This consistency provides evidence that the convergence behavior represents fundamental properties of the co-evolutionary dynamics rather than architecturespecific artifacts.

Future theoretical work should investigate conditions under which the system exhibits stable convergence versus chaotic dynamics. Key questions include: What curriculum evolution rates ensure stable learning? How do different value functions affect convergence properties? Can we establish bounds on the oscillation amplitude around target performance levels observed in our empirical data?

The intersection of curriculum learning, preference optimization, and multi-agent dynamics presents rich opportunities for theoretical development. The DPO training parameters (Table 7) and their interaction with curriculum evolution rates could inform theoretical models of multi-agent preference learning. Establishing convergence guarantees would enable more principled hyperparameter selection and provide confidence bounds for practical deployment.

