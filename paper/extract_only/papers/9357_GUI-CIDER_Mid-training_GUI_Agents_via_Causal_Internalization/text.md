# arXiv:2605.28534v1[cs.CL]27May2026

## GUI-CIDER: Mid-training GUI Agents via Causal Internalization and Density-aware Exemplar Reselection

Zheng Wu1,2* Chengcheng Han2 Zhengxi Lu2,3 Tianjie Ju1,2 Yanyu Chen2,4 Qi Gu2† Xunliang Cai2 Zhuosheng Zhang1† 1School of Computer Science, Shanghai Jiao Tong University 2Meituan 3Zhejiang University 4The Chinese University of Hong Kong {wzh815918208,zhangzs}@sjtu.edu.cn guqi03@meituan.com

### Abstract

Despite the rapid progress of multimodal large language models in building Graphical User Interface (GUI) agents, their real-world task completion is fundamentally bottlenecked by a lack of world knowledge about GUI operations. Existing solutions typically rely on expensive multi-agent scaffolding or conventional post-training paradigms, such as Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL). However, post-training only allows agents to implicitly absorb world knowledge through action annotations or reward signals, leading to inefficient trajectory memorization rather than genuine comprehension. Therefore, an approach that enables explicit learning of this knowledge is imperative. To this end, we propose GUI-CIDER, a mid-training method that explicitly internalizes GUI world knowledge through Causal Internalization and Density-aware Exemplar Reselection. GUICIDER operates in three stages: (1) data synthesis, which distills static planning and dynamic causal knowledge from GUI trajectories into text; (2) exemplar reselection, which filters the corpus by rewarding causal structures and penalizing semantic redundancy; and (3) mid-training, where the refined data is used to embed the acquired knowledge. Extensive experiments on two GUI knowledge benchmarks and three task completion benchmarks demonstrate that GUI-CIDER consistently improves both the agent’s understanding of GUI operations and its task success rates.The codes are available at https://github.com/ Wuzheng02/GUI-CIDER.

### 1 Introduction

With the rapid advances of multimodal large language models (MLLMs) in reasoning (Bai et al., 2025), planning (Wei et al., 2025; Chen et al.,

*Work completed while Zheng Wu, Zhengxi Lu, Tianjie Ju, and Yanyu Chen were interns at Meituan.

†Corresponding authors.

[Figure 1]

Figure 1: The motivation of GUI-CIDER. (a) Current single-agent methods lack GUI world knowledge and cannot understand that the plus sign represents adding a task. (b) External scaffolding can obtain low-level instructions by calling a general agent, but this is timeconsuming. (c) GUI-CIDER enables a single agent to accomplish the task through mid-training on highquality GUI world knowledge.

2026b), perception (Yu et al., 2025), and decisionmaking (Sun et al., 2025), MLLM-based Graphical User Interface (GUI) agents (Tang et al., 2025) can now follow user instructions to autonomously control digital devices (e.g., computers (Sager et al., 2026) and smartphones (Wu et al., 2025a)) by simulating human actions (e.g., clicking and scrolling).

Existing work on GUI agents improves element grounding (Liu et al., 2026; Tang et al., 2026) and task completion (Bai et al., 2024; Xu et al., 2025) through post-training methods such as supervised fine-tuning (SFT) (Zhang and Zhang, 2024; Ma et al., 2024) and reinforcement learning (RL) (Lu et al., 2026; Luo et al., 2025). However, studies (Shi et al., 2025; Li et al., 2025) point out that as GUI agents continue to advance, the real capability bottleneck increasingly stems from a lack of world knowledge related to GUI operations.

Although plugging a capable general-purpose model into a multi-agent system (Yang et al., 2025;

Wang et al., 2024) can compensate for GUI agents’ deficiency in world knowledge, it introduces additional overhead and scaffolding.

In contrast, internalizing world knowledge within the agent is more efficient, yet conventional post-training (SFT/RL) only implicitly encodes such knowledge through action labels or reward signals, encouraging trajectory memorization rather than genuine comprehension.

An approach that enables explicit learning is therefore imperative.

Consequently, as shown in Figure 1, we propose GUI-CIDER, a mid-training method for GUI agents that explicitly internalizes world knowledge into them through Causal Internalization and Density-aware Exemplar Reselection.

GUI-CIDER consists of three stages: (1) data synthesis stage, (2) exemplar reselection stage, and (3) mid-training stage. In the data synthesis stage, GUI-CIDER employs a dedicated synthesis pipeline to generate static planning knowledge and dynamic causal knowledge for the GUI agent domain from publicly available GUI agent datasets (Li et al., 2024; Lu et al., 2025; Zhang et al., 2024). In the exemplar reselection stage, GUI-CIDER filters the data produced in the previous stage through causal-informed retention and relative density estimation based on K-nearest neighbors, resulting in a high-quality corpus that exhibits strong reasoning structures and low redundancy. In the mid-training stage, GUI-CIDER uses this high-quality corpus to train the GUI agent via mid-training, thereby explicitly internalizing world knowledge into the GUI agent.

We conduct extensive experiments on three benchmarks (Li et al., 2024; Lu et al., 2025; Zhang et al., 2024) for GUI agent task completion and two benchmarks (Wang et al., 2025; Shi et al., 2025) for GUI agent knowledge. Experimental results show that GUI-CIDER achieves an average relative improvement of 9.70% in task success rate compared to post-training baselines. Meanwhile, on the GUI knowledge bench, it enables an 8B-scale agent to reach a level close to that of Claude-Sonnet-4.5.

Additionally, through model comparison analysis, we show that the target of mid-training should be general agents rather than one that has been excessively post-trained specifically in the GUI agent domain. Furthermore, we validate the rationality of the GUI-CIDER pipeline through ablation studies.

To summarize, our contributions are three-fold: (i) We propose GUI-CIDER, a mid-training

method for GUI agents that explicitly internalizes world knowledge relevant to GUI agents into them through Causal Internalization and Density-aware Exemplar Reselection.

- (ii) We contribute a corpus of approximately 100M tokens generated from the data synthesis stage of GUI-CIDER, offering a valuable resource for related research in the community.
- (iii) Through extensive experiments, we demonstrate that GUI-CIDER can not only improve GUI agents’ world knowledge of GUI operations but also enhance their task completion performance.

### 2 Related Work

In this section, we first introduce recent improvements in GUI agents, and then we introduce related work on mid-training of (M)LLMs.

#### 2.1 GUI Agents

GUI agents are a type of agent that operate intelligent terminals such as computers (Sager et al., 2026), web (He et al., 2024), and smartphones (Zhang and Zhang, 2024; Wu et al., 2025a) by simulating human actions like clicking and scrolling (Tang et al., 2025; Hu et al., 2025). Existing work can be broadly divided into two categories for constructing GUI agents: single-agent based and multi-agent system based. Single-agent based GUI agents are typically developed through pretraining and post-training. Pretraining enhances the agent’s perception (Ma et al., 2024) and grounding capabilities (Wu et al., 2025b). Post-training methods, on the other hand, primarily improve the agent’s task completion ability through techniques such as SFT (Wu et al., 2025c) and RL (Lu et al., 2026; Zhou et al., 2026; Tang et al., 2026). Multi-agent system based GUI agents distribute capabilities such as planning (Wang et al., 2024), reflection (Li et al., 2026), and execution (Yang et al., 2025; Agashe et al.) across different agents to adapt to different tasks. However, few existing works enhance the world knowledge of GUI agents through mid-training.

#### 2.2 Mid-training for (M)LLM

Mid-training serves as a bridge (Tu et al., 2025; Mo et al., 2025) between pre-training and post-training, extending knowledge into specialized domains while preserving the general capabilities acquired during pre-training. Existing (M)LLMs (Team et al., 2025; Hu et al., 2024; Liu et al., 2024) conduct data collection, data synthesis, data selection,

[Figure 2]

- Figure 2: The pipeline of GUI-CIDER. In the data synthesis stage, GUI-CIDER synthesizes GUI world knowledge from raw GUI trajectories. In the exemplar reselection stage, GUI-CIDER selects a high-quality corpus with strong reasoning structures and low redundancy from the synthesized corpus. In the mid-training stage, GUI-CIDER internalizes the GUI world knowledge into the GUI agent through mid-training on the high-quality corpus.

and data decontamination from high-quality mathematical (Paster et al., 2024; Han et al., 2024), QA (Wei et al.; Ding et al., 2023) and coding (Kocetkov et al.; Lozhkov et al., 2024; Luo et al., 2024) domains. However, there is still very little work on internalizing domain knowledge for GUI agents through mid-training. UI-Venus-1.5 (Team et al., 2026) employed mid-training but did not opensource the data or provide specific details. Therefore, it is valuable to explore how GUI agents can internalize knowledge through mid-training.

### 3 GUI-CIDER

In this section, we introduce GUI-CIDER, a midtraining method for GUI agents, which stands for Causal Internalization and Density-aware Exemplar Reselection. As shown in Figure 2, GUICIDER consists of three stages: data synthesis, exemplar reselection, and mid-training. Next, we will introduce each of these stages in order.

#### 3.1 Stage 1: Data Synthesis

Given a raw GUI agent domain dataset D = {τ1,τ2,...,τN}, where each trajectory τ consists of a task instruction T and a sequence of screenshots and actions {(si,ai)}Li=1, we synthesize an augmented, knowledge-rich sample x. Specifically, the synthesized sample x encompasses two pri-

mary dimensions: static planning knowledge and dynamic causal knowledge.

Static Planning Knowledge Extraction. To operationalize hierarchical task decomposition, we leverage a high-capacity LLM as a latent knowledge prior, formalizing the planning process as a structured reasoning task. Specifically, the planning function P(·) is instantiated by an expert model that performs zero-shot reasoning to generate a hierarchical decomposition:

S = P(T;Mexp) = {g1,g2,...,gn}, (1)

where Mexp denotes the expert reasoning engine and gj represents a high-level sub-goal in natural language. This transformation converts abstract user intent into an actionable execution graph, providing dense supervisory signals for the agent’s long-term planning.

Dynamic Causal Knowledge Synthesis. To explicitly model environment transition dynamics and decision-making logic while producing a purely textual knowledge sample, we reformulate knowledge extraction as a text-grounded semantic and causal induction process. This is achieved through two specialized reasoning modules:

(i) Semantic Behavioral Grounding: A mapping function B(at,vt) that translates raw, lowlevel action primitives at and their corresponding

UI metadata vt (e.g., view hierarchy) into humaninterpretable semantic descriptions anlt . This stage bridges the gap between discrete pixel-level coordinates and high-level functional intent.

(ii) Textual State Abstraction and Causal Logic Induction: The visual screenshots st−1 and st are first converted into natural language state descriptions dt−1 and dt through a vision-language interface. For each transition under task T, we then employ a causal analyst C(·) that operates solely over textual representations. By prompting the expert model to perform retrospective and counterfactual analysis on the described states, we extract the underlying transition logic in a self-contained textual rationale Rt:

Rt = C(T,dt−1,anlt ,dt | Mexp)

(2)

= {dt−1,dt,Trigt,Mecht,CoTt},

where Trigt, Mecht, and CoTt denote the action trigger, the underlying UI mechanism, and the chain-of-thought rationale, respectively. The state descriptions dt−1 and dt are explicitly stored as part of Rt, making the rationale self-contained and eliminating the need for raw screenshots in the final sample.

The final synthesized sample x is thus defined as a purely textual, knowledge-rich tuple: x = ⟨T,S,anlt ,Rt⟩.

- 3.2 Stage 2: Exemplar Reselection To refine the synthesized corpus X =

{x1,...,xM}, we employ density-aware exemplar reselection. Let ϕ(x) be the embedding of sample x in a latent space Z ⊆ Rd.

Causal-Informed Retention. Following existing work (Chen et al., 2026a), we first define a causal saliency function f(x) based on the count of causallogic tokens:

f(x) = tanh K(x) γ

, (3)

where K(x) denotes the count of causal-logic tokens in Rt and γ controls the causal scaling. Here, causal-logic tokens broadly encompass words and phrases carrying causal or logical semantics (e.g., ’if’, ’unless’, ’because’, ’due to’). Detailed causallogic keywords can be found in Appendix E.

Relative Density Estimation. The local density d(x) is defined based on the ratio of the K-nearest

neighbor distance to the global mean distance. Let the raw ratio be

1 K z∈KNN(ϕ(x)) ∥ϕ(x) − z∥2

. (4)

r(x) =

1

M z′∈X ∥ϕ(x) − z′∥2

To obtain a density score in [0,1], we apply minmax normalization across all samples in the feature set X:

r(x) − minx′∈X r(x′) maxx′∈X r(x′) − minx′∈X r(x′)

. (5)

d(x) =

The retention probability g(x) for each sample x is then given by a non-linear combination of its semantic density d(x) and the causal saliency f(x):

1 1 + αd(x)

1 1 + αd(x)

+λ·f(x)· 1 −

g(x) =

,

(6) where α is a hyperparameter governing density sensitivity, and λ ∈ [0,1] is the weight for causal importance.

Finally, the high-quality corpus Xhigh is formed by retaining each sample x with probability g(x):

Xhigh = {x ∈ X | ξx ≤ g(x)}, (7)

where ξx ∼ Uniform(0,1) is sampled independently for every x.

#### 3.3 Stage 3: Mid-training

In the mid-training stage, we directly perform nexttoken prediction on the high-quality corpus Xhigh. For each synthesized sample x, we first format it into a single token sequence by concatenating its components in a fixed order. No distinction is made between input and output: the entire sequence is treated as a plain text stream for autoregressive language modeling. The training objective is the standard causal language modeling loss over all tokens in the sequence:

Lx

Lmid = −

x∈Xhigh

log Pθ(yi | y<i), (8)

i=1

where Lx is the total number of tokens in the serialized sequence of sample x, and yi denotes the i-th token. By optimizing Lmid, the model internalizes the transition dynamics P(st | st−1,at) and the underlying world knowledge directly into its parametric memory, achieving causal internalization without necessitating external runtime scaffolds.

### 4 Is the Retention Function g(x) a Good Function?

In this section, we first introduce the properties of a good retention function g(x) under our task setting, and then provide theoretical support to prove that GUI-CIDER’s retention function g(x) satisfies all these properties.

#### 4.1 Properties for the Retention Function g(x)

To effectively select high-value samples with strong reasoning structures and low redundancy, the retention function g(x) should possess the following four properties:

Property 1 (Causal Monotonicity) g(x) should be monotonically increasing with respect to the causal saliency f(x).

Samples with more causal-logic tokens contain richer reasoning structures and therefore deserve higher retention probabilities.

Property 2 (Density Penalty) g(x) should be monotonically decreasing with respect to the density d(x).

Higher density indicates that many different samples share similar semantics, leading to redundancy; thus, the retention probability should be penalized accordingly.

Property 3 (Density Order Preservation) g(x) should satisfy ∂d∂ d · g(x) > 0.

Although we filter the corpus, we must not invert the original density ordering of the semantic space, thereby preserving the relative density structure.

Property 4 (Density-Causal Synergy) g(x) should satisfy ∂

2g(x) ∂f∂d > 0.

In denser regions where semantic redundancy is high, the survival competition is fiercer. Therefore, an increase in causal saliency should provide a greater marginal benefit to the retention probability, enabling the most logically rigorous exemplars to stand out among highly redundant samples.

#### 4.2 Theoretical Support

We now prove that the retention function defined in GUI-CIDER satisfies the three properties.

- Proof of Property 1. For a fixed density d, the derivative of g with respect to f is

∂g ∂f

= λ 1 −

1 1 + αd

= λ

αd 1 + αd

. (9)

Since λ ≥ 0, α > 0, and d ≥ 0, we have ∂f∂g ≥ 0. Thus, g(x) is monotonically non-decreasing in f(x). For any sample with d > 0, the derivative is strictly positive, ensuring that higher causal saliency strictly increases the retention probability.

- Proof of Property 2. The derivative with respect to density d is

∂g ∂d

= −

α(1 − λf) (1 + αd)2

. (10)

Because f ∈ [0,1) and λ ∈ [0,1], we have 1 − λf ≥ 0. With α > 0, the derivative is non-positive, so g(x) is monotonically non-increasing in d(x). This directly imposes a redundancy penalty: denser samples receive lower retention probabilities.

- Proof of Property 3. For the product d · g(x): ∂

∂d

d · g = g + d ·

∂g ∂d

= λf +

1 − λf 1 + αd −

αd(1 − λf) (1 + αd)2

= λf + (1 − λf)

1 + αd − αd (1 + αd)2

= λf +

1 − λf (1 + αd)2

. (11)

Since λf ≥ 0 and 1 − λf ≥ 0 with a strictly

positive denominator, we obtain ∂d∂ (d · g) > 0 for all valid parameter settings. This guarantees

that if two samples have densities d1 < d2, then d1 · g(x1) < d2 · g(x2) (all else being equal), faithfully preserving the original density ordering of the semantic space.

- Proof of Property 4. The cross-partial derivative of g(x) is:

∂2g ∂f∂d

∂ ∂d

=

αd 1 + αd

λ

λα (1 + αd)2

. (12)

=

Given λ > 0 and α > 0, we strictly have

∂2g

∂f∂d > 0. This guarantees that the marginal utility of causal saliency f increases monotonically

with density d, effectively prioritizing high-quality reasoning structures within redundant clusters.

Task Category Dataset Evaluation Format Train Size Test Size GUI Agent Knowledge

MMBench-GUI L1 MCQ - 3,561 GUI Knowledge Bench T/F, MCQ - 3,483

AITZ

13,919 4,723 AndroidControl 69,670 7836 GUI-Odyssey 102,086 25,807

GUI Agent Task Completion

Action Generation

Table 1: Overview of datasets and evaluation format.

AITZ AndroidControl GUI-Odyssey

Method Type SR TSR Type SR TSR Type SR TSR Qwen3-VL-4B-Instruct Zero-shot 59.58 39.53 0.79 74.92 51.82 13.60 61.89 41.00 0.30 GUI-CIDER 60.46 41.22 0.99 76.24 53.58 14.43 64.83 43.45 0.42 Post-training 76.63 60.43 4.94 85.07 68.75 28.43 89.72 73.46 3.86 GUI-CIDER + Post-training 77.43 61.87 5.14 85.17 69.77 28.56 89.47 75.36 4.34 Qwen3-VL-8B-Instruct Zero-shot 56.91 40.82 0.99 73.92 52.49 13.96 67.86 44.16 0.36 GUI-CIDER 60.70 42.07 1.58 74.87 54.09 15.13 70.26 48.55 0.36 Post-training 72.98 58.16 4.15 83.50 65.34 23.46 88.82 71.74 3.32 GUI-CIDER + Post-training 73.70 60.33 5.14 83.41 66.82 25.35 89.65 73.36 3.63

- Table 2: Results of AITZ, AndroidControl and GUI-Odyssey benchmarks. Whether compared with zero-shot or post-training methods, adding a mid-training process leads to improvements.

### 5 Experiment

In this section, we first introduce the implementation of our GUI-CIDER experiments, then present the main results and provide analysis.

#### 5.1 Implementation

Dataset. As shown in Table 1, we conduct extensive experiments on three benchmarks, AITZ (Zhang et al., 2024), AndroidControl (Li et al., 2024), and GUI-Odyssey (Lu et al., 2025), for GUI agent task completion, where the agent is required to output actions to accomplish tasks, and two benchmarks, MMbench-GUI L1 (Wang et al., 2025) and GUI knowledge bench (Shi et al., 2025), for GUI agent knowledge, both of which adopt the formats of multiple-choice questions (MCQs) and true-false (T/F) questions.

Evaluation Method. We used GUI-CIDER for data synthesis on the AITZ, AndroidControl, and GUI-Odyssey datasets. The base models were Qwen3-VL-4B-Instruct and Qwen3-VL-8BInstruct (Bai et al., 2025). In the main results section, we refer to the models obtained by midtraining Qwen3-VL-4B-Instruct and Qwen3-VL-

8B-Instruct with GUI-CIDER as GUI-CIDER-4B and GUI-CIDER-8B, respectively. For evaluation on MMBench-GUI L1 and GUI Knowledge Bench, we performed mid-training using a mixture of all synthesized data. For evaluation on AITZ, AndroidControl, and GUI-Odyssey, we conducted mid-training using the data synthesized from the corresponding dataset. Meanwhile, we adopted SFT as the baseline method for post-training.

Metrics. For the AITZ, AndroidControl, and GUI-Odyssey datasets, we report action type accuracy (type), step-wise success rate (SR), and task success rate (TSR). For MMbench-GUI L1 and GUI Knowledge Bench, we compute the accuracy of multiple-choice questions and true-false questions under different subsets.

#### 5.2 Main Results

The results on the AITZ, AndroidControl, and GUIOdyssey datasets are shown in Table 2, the results on MMbench-GUI L1 are shown in Table 3, and the results on the GUI knowledge benchmark are shown in Table 4.

Based on the above results, we find:

Model Windows MacOS Linux iOS Android Web Overall Easy Level GPT-4o 62.47 67.89 62.38 58.52 56.41 58.51 60.16 Qwen-Max-VL 69.05 72.51 69.91 70.82 63.09 69.46 68.15

- Qwen2.5-VL-72B 65.86 75.23 73.02 67.24 58.09 72.08 66.98 UI-TARS-72B-DPO 41.59 28.52 35.16 31.08 52.25 35.33 40.18

- InternVL3-72B 74.67 78.72 79.16 83.57 80.10 81.18 79.15 GUI-CIDER-8B 95.19 97.62 96.91 90.43 93.44 94.98 94.69

Medium Level GPT-4o 56.33 63.13 59.70 54.06 57.69 54.98 57.24 Qwen-Max-VL 63.40 73.85 66.90 68.02 63.66 64.59 65.44 Qwen2.5-VL-72B 66.29 72.73 72.63 59.27 66.24 68.24 67.45 UI-TARS-72B-DPO 38.83 41.60 37.14 41.72 54.74 31.55 41.77 InternVL3-72B 71.46 78.58 79.88 78.43 81.36 78.67 77.89

GUI-CIDER-8B 95.56 91.67 96.39 89.57 89.84 88.13 92.00 Hard Level GPT-4o 60.69 60.38 52.42 45.27 50.93 50.83 53.49 Qwen-Max-VL 66.64 67.59 65.80 60.23 58.78 65.34 63.69 Qwen2.5-VL-72B 70.68 68.91 70.98 57.59 53.94 68.10 64.56 UI-TARS-72B-DPO 31.48 35.87 24.19 36.33 58.13 19.94 35.78

- InternVL3-72B 75.08 77.44 76.19 70.37 75.73 78.11 75.70 GUI-CIDER-8B 93.33 90.48 94.33 92.17 88.20 93.15 91.83

- Table 3: Results of MMBench-GUI L1 benchmark. GUI-CIDER enhances GUI understanding. Although the data sources do not cover every platform in MMBench-GUI, improvements are still observed across all of them.

(i) As shown in Table 2, mid-training yields gains in task completion capability across models of different parameter scales. Furthermore, when post-training is applied after mid-training, the benefits of GUI-CIDER still manifest. In addition, a 4Bscale GUI agent, after undergoing mid-training and post-training with GUI-CIDER, surpasses its 8Bscale counterpart, suggesting that for GUI agents, what matters may not be parameter scaling but rather knowledge scaling.

- (ii) As shown in Table 3, GUI-CIDER-8B signif-

icantly outperforms the baselines, indicating that GUI-CIDER brings improvements to the GUI content understanding capability of GUI agents.

- (iii) As shown in Table 4, overall, GUI-CIDER-

8B clearly bridges the knowledge gap in GUI tasks, achieving performance close to that of ClaudeSonnet-4.5 at the 8B scale (66.51 vs. 66.53). Moreover, GUI-CIDER-8B surpasses all larger-scale models (e.g., o3, Gemini-2.5-Pro) on the objective subset (which assesses whether a task is truly completed), demonstrating that GUI-CIDER equips the

GUI agent with a better understanding of tasks.

### 6 Further Analysis

In this section, we first compare the differences between models that have undergone post-training in the GUI agent domain and general models when used as the base model for GUI-CIDER, followed by an ablation study.

#### 6.1 Model Comparison Analysis

We conduct an analysis to verify whether a GUIspecialized model that has already been posttrained in the GUI agent domain can acquire new world knowledge again through mid-training. We perform experiments on the AITZ dataset with OS-Atlas-pro-7B following the GUI-CIDER, and report results with the amount of GUI-CIDERgenerated data increasing in 20% increments.

As shown in Figure 3, when using the general model Qwen3-VL-8B-Instruct as the base model, the GUI agent’s SR consistently improves as more GUI-CIDER-generated data are incorporated. In

Interface Knowledge Interaction Knowledge Procedure Knowledge

Model

Overall state widget layout effect type parameter objective workflow

O3 83.03 84.12 88.39 74.83 75.98 45.75 69.45 95.47 73.30 Gemini-2.5-Pro 81.19 84.36 87.10 71.03 73.25 46.97 67.72 92.56 71.69 GPT-5-Chat 78.90 84.12 88.39 71.55 71.55 43.85 68.98 91.26 70.97 Claude-Sonnet-4.5 74.77 81.52 82.58 49.83 70.19 43.33 70.30 91.56 66.53 Qwen3-VL-8B-Instruct 66.97 76.30 79.35 60.86 63.54 45.06 71.02 78.96 65.23 Qwen2.5-VL-72B 69.27 77.49 80.00 61.72 64.91 38.99 62.20 85.44 63.88 Doubao-V-Pro 72.48 83.65 81.29 67.24 75.64 41.07 33.07 94.17 63.42 Claude-Sonnet-4 70.18 78.44 78.06 41.90 62.52 42.11 65.20 94.82 62.16 Qwen2.5-VL-7B 53.21 67.77 60.00 51.72 50.60 39.34 16.22 48.87 45.16 UI-TARS-1.5-7B 49.54 59.48 59.35 22.24 59.11 34.32 38.74 55.34 44.27 GUI-owl-7B 60.09 64.93 63.23 21.55 55.37 36.05 21.26 39.81 40.74 GLM-4.5 49.54 48.10 53.55 27.07 17.55 35.53 28.98 91.91 38.10

GUI-CIDER-8B 72.61 75.83 80.13 61.55 65.42 46.45 71.81 80.58 66.51

- Table 4: Results of GUI knowledge benchmark. The performance of GUI-CIDER-8B is close to Claude-Sonnet-4.5.

[Figure 3]

- Figure 3: Comparison between a general model and a GUI-specialized model as the base model for midtraining, illustrating that models excessively posttrained in the GUI agent domain are no longer suitable for acquiring world knowledge through mid-training.

contrast, when using OS-Atlas-pro-7B as the base model, the GUI agent’s performance steadily declines. This is because OS-Atlas-pro-7B has undergone extensive post-training for GUI agents, which has already partially disrupted its original language representation capacity, making it difficult to learn new world knowledge through mid-training. Therefore, performing mid-training on world knowledge before conducting post-training in the GUI agent domain would be a reasonable paradigm.

#### 6.2 Ablation Study

We conduct an ablation study to investigate the necessity of the exemplar reselection stage in GUI-

Table 5: Ablation study of exemplar reselection stage.

#### Model w/ Stage 2 w/o Stage 2

Qwen3-VL-4B-Instruct 43.45 41.06 Qwen3-VL-8B-Instruct 48.55 42.34

CIDER. Specifically, we compare SR after midtraining with the complete GUI-CIDER pipeline against a variant that removes the exemplar reselection stage on the GUI-Odyssey dataset. As shown in Table 5, removing the exemplar reselection stage leads to a substantial drop in SR. This is because directly incorporating large-scale unscreened data into mid-training introduces a considerable amount of low-quality and redundant samples. Such noisy supervision can mislead the GUI agent and encourage shortcut or hacking behaviors, ultimately harming generalization and decision-making capability.

### 7 Conclusion

In this paper, we present GUI-CIDER, a midtraining framework that internalizes GUI world knowledge into GUI agents through causal internalization and density-aware exemplar reselection. Instead of relying on expensive external scaffolding or directly applying post-training to raw trajectories, GUI-CIDER synthesizes static planning knowledge and dynamic causal transition logic from GUI trajectories, then selects knowledge-rich and non-redundant exemplars with a retention func-

tion. Experiments on three task completion benchmarks and two GUI knowledge benchmarks show that GUI-CIDER improves both GUI operation understanding and downstream task success. These results suggest that knowledge scaling is a promising path toward more capable GUI agents.

### Limitations

Due to computational resource constraints, GUICIDER employs LoRA instead of full parameter tuning during the mid-training stage. Additionally, the model parameters used for training range from 4B to 8B. Future work will further explore the effectiveness under full parameter tuning and scale the approach to larger models.

### References

Saaket Agashe, Kyle Wong, Vincent Tu, Jiachen Yang, Ang Li, and Xin Eric Wang. Agent s2: A compositional generalist-specialist framework for computer use agents. In Second Conference on Language Modeling.

Hao Bai, Yifei Zhou, Mert Cemri, Jiayi Pan, Alane Suhr, Sergey Levine, and Aviral Kumar. 2024. Digirl: Training in-the-wild device-control agents with autonomous reinforcement learning. Advances in Neural Information Processing Systems, 37:12461– 12495.

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhifang Guo, Qidong Huang, Jie Huang, Fei Huang, Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng Li, and 45 others. 2025. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631.

Qiguang Chen, Yantao Du, Ziniu Li, Jinhao Liu, Songyao Duan, Jiarui Guo, Minghao Liu, Jiaheng Liu, Tong Yang, Ge Zhang, and 1 others. 2026a. The molecular structure of thought: Mapping the topology of long chain-of-thought reasoning. arXiv preprint arXiv:2601.06002.

Yanyu Chen, Jiyue Jiang, Jiahong Liu, Yifei Zhang, Xiao Guo, and Irwin King. 2026b. Trace: Trajectoryaware comprehensive evaluation for deep research agents. In Proceedings of the ACM Web Conference 2026, pages 2524–2534.

Ning Ding, Yulin Chen, Bokai Xu, Yujia Qin, Shengding Hu, Zhiyuan Liu, Maosong Sun, and Bowen Zhou. 2023. Enhancing chat language models by scaling high-quality instructional conversations. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 3029–3051.

Xiaotian Han, Yiren Jian, Xuefeng Hu, Haogeng Liu, Yiqi Wang, Qihang Fan, Yuang Ai, Huaibo Huang, Ran He, Zhenheng Yang, and 1 others. 2024. Infimmwebmath-40b: Advancing multimodal pre-training for enhanced mathematical reasoning. arXiv preprint arXiv:2409.12568, 4.

Hongliang He, Wenlin Yao, Kaixin Ma, Wenhao Yu, Yong Dai, Hongming Zhang, Zhenzhong Lan, and Dong Yu. 2024. Webvoyager: Building an end-toend web agent with large multimodal models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 6864–6890.

Shengding Hu, Yuge Tu, Xu Han, Chaoqun He, Ganqu Cui, Xiang Long, Zhi Zheng, Yewei Fang, Yuxiang Huang, Weilin Zhao, and 1 others. 2024. Minicpm: Unveiling the potential of small language models with scalable training strategies. arXiv preprint arXiv:2404.06395.

Xueyu Hu, Tao Xiong, Biao Yi, Zishu Wei, Ruixuan Xiao, Yurun Chen, Jiasheng Ye, Meiling Tao, Xiangxin Zhou, Ziyu Zhao, and 1 others. 2025. Os agents: A survey on mllm-based agents for computer, phone and browser use. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 7436–7465.

Denis Kocetkov, Raymond Li, Chenghao Mou, Yacine Jernite, Margaret Mitchell, Carlos Muñoz Ferrandis, Sean Hughes, Thomas Wolf, Dzmitry Bahdanau, Leandro Von Werra, and 1 others. The stack: 3 tb of permissively licensed source code. Transactions on Machine Learning Research.

Chunyi Li, Longfei Li, Zicheng Zhang, Xiaohong Liu, Min Tang, Weisi Lin, and Guangtao Zhai. 2025. Using gui agent for electronic design automation. arXiv preprint arXiv:2512.11611.

Ning Li, Xiangmou Qu, Jiamu Zhou, Muning Wen, Kounianhua Du, Xingyu Lou, Qiuying Peng, Jun Wang, and Weinan Zhang. 2026. Mobileuse: A hierarchical reflection-driven gui agent for autonomous mobile operation. Advances in Neural Information Processing Systems, 38:40361–40388.

Wei Li, William Bishop, Alice Li, Chris Rawles, Folawiyo Campbell-Ajala, Divya Tyamagundlu, and Oriana Riva. 2024. On the effects of data scale on ui control agents. Advances in Neural Information Processing Systems, 37:92130–92154.

Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, and 1 others. 2024. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437.

Yuhang Liu, Zeyu Liu, Shuanghe Zhu, Pengxiang Li, Congkai Xie, Jiasheng Wang, Xueyu Hu, Xiaotian Han, Jianbo Yuan, Xinyao Wang, and 1 others. 2026. Infigui-g1: Advancing gui grounding with adaptive

exploration policy optimization. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages 32267–32275.

Anton Lozhkov, Raymond Li, Loubna Ben Allal, Federico Cassano, Joel Lamy-Poirier, Nouamane Tazi, Ao Tang, Dmytro Pykhtar, Jiawei Liu, Yuxiang Wei, and 1 others. 2024. Starcoder 2 and the stack v2: The next generation. arXiv preprint arXiv:2402.19173.

Quanfeng Lu, Wenqi Shao, Zitao Liu, Lingxiao Du, Fanqing Meng, Boxuan Li, Botong Chen, Siyuan Huang, Kaipeng Zhang, and Ping Luo. 2025. Guiodyssey: A comprehensive dataset for cross-app gui navigation on mobile devices. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 22404–22414.

Zhengxi Lu, Yuxiang Chai, Yaxuan Guo, Xi Yin, Liang Liu, Hao Wang, Han Xiao, Shuai Ren, Pengxiang Zhao, Guangyi Liu, and 1 others. 2026. Ui-r1: Enhancing efficient action prediction of gui agents by reinforcement learning. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages 17608–17616.

Run Luo, Lu Wang, Wanwei He, Longze Chen, Jiaming Li, and Xiaobo Xia. 2025. Gui-r1: A generalist r1style vision-language action model for gui agents. arXiv preprint arXiv:2504.10458.

Ziyang Luo, Can Xu, Pu Zhao, Qingfeng Sun, Xiubo Geng, Wenxiang Hu, Chongyang Tao, Jing Ma, Qingwei Lin, and Daxin Jiang. 2024. Wizardcoder: Empowering code large language models with evolinstruct. In International Conference on Learning Representations, volume 2024, pages 27168–27188.

Xinbei Ma, Zhuosheng Zhang, and Hai Zhao. 2024. Coco-agent: A comprehensive cognitive mllm agent for smartphone gui automation. In Findings of the Association for Computational Linguistics: ACL 2024, pages 9097–9110.

Kaixiang Mo, Yuxin Shi, Weiwei Weng, Zhiqiang Zhou, Shuman Liu, Haibo Zhang, and Anxiang Zeng. 2025. Mid-training of large language models: A survey. arXiv preprint arXiv:2510.06826.

Keiran Paster, Marco Dos Santos, Zhangir Azerbayev, and Jimmy Ba. 2024. Openwebmath: An open dataset of high-quality mathematical web text. In International Conference on Learning Representations, volume 2024, pages 20357–20379.

Pascal J Sager, Benjamin Meyer, Peng Yan, Rebekka von Wartburg-Kottler, Layan Etaiwi, Aref Enayati, Gabriel Nobel, Ahmed Abdulkadir, Benjamin F Grewe, and Thilo Stadelmann. 2026. A comprehensive survey of agents for computer use: Foundations, challenges, and future directions. Journal of Artificial Intelligence Research, 85.

Chenrui Shi, Zedong Yu, Zhi Gao, Ruining Feng, Enqi Liu, Yuwei Wu, Yunde Jia, Liuyu Xiang, Zhaofeng

He, and Qing Li. 2025. Gui knowledge bench: Revealing the knowledge gap behind vlm failures in gui tasks. arXiv preprint arXiv:2510.26098.

Chuanneng Sun, Songjun Huang, and Dario Pompili. 2025. Llm-based multi-agent decision-making: Challenges and future directions. IEEE Robotics and Automation Letters.

Fei Tang, Zhangxuan Gu, Zhengxi Lu, Xuyang Liu, Shuheng Shen, Changhua Meng, Wen Wang, Wenqi Zhang, Yongliang Shen, Weiming Lu, and 1 others. 2026. Gui-g2: Gaussian reward modeling for gui grounding. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages 33214– 33222.

Fei Tang, Haolei Xu, Hang Zhang, Siqi Chen, Xingyu Wu, Yongliang Shen, Wenqi Zhang, Guiyang Hou, Zeqi Tan, Yuchen Yan, and 1 others. 2025. A survey on (m) llm-based gui agents. arXiv preprint arXiv:2504.13865.

Meituan LongCat Team, Bei Li, Bingye Lei, Bo Wang, Bolin Rong, Chao Wang, Chao Zhang, Chen Gao, Chen Zhang, Cheng Sun, and 1 others. 2025. Longcat-flash technical report. arXiv preprint arXiv:2509.01322.

Venus Team, Changlong Gao, Zhangxuan Gu, Yulin Liu, Xinyu Qiu, Shuheng Shen, Yue Wen, Tianyu Xia, Zhenyu Xu, Zhengwen Zeng, and 1 others. 2026. Ui-venus-1.5 technical report. arXiv preprint arXiv:2602.09082.

Chengying Tu, Xuemiao Zhang, Rongxiang Weng, Rumei Li, Chen Zhang, Yang Bai, Hongfei Yan, Jingang Wang, and Xunliang Cai. 2025. A survey on llm mid-training. arXiv preprint arXiv:2510.23081.

Junyang Wang, Haiyang Xu, Haitao Jia, Xi Zhang, Ming Yan, Weizhou Shen, Ji Zhang, Fei Huang, and Jitao Sang. 2024. Mobile-agent-v2: Mobile device operation assistant with effective navigation via multiagent collaboration. Advances in Neural Information Processing Systems, 37:2686–2710.

Xuehui Wang, Zhenyu Wu, JingJing Xie, Zichen Ding, Bowen Yang, Zehao Li, Zhaoyang Liu, Qingyun Li, Xuan Dong, Zhe Chen, and 1 others. 2025. Mmbench-gui: Hierarchical multi-platform evaluation framework for gui agents. arXiv preprint arXiv:2507.19478.

Hui Wei, Zihao Zhang, Shenghua He, Tian Xia, Shijia Pan, and Fei Liu. 2025. Plangenllms: A modern survey of llm planning capabilities. arXiv preprint arXiv:2502.11221.

Yuxiang Wei, Zhe Wang, Jiawei Liu, Yifeng Ding, and LINGMING ZHANG. Magicoder: Empowering code generation with oss-instruct. In Forty-first International Conference on Machine Learning.

Zheng Wu, Heyuan Huang, Yanjia Yang, Yuanyi Song, Xingyu Lou, Weiwen Liu, Weinan Zhang, Jun Wang, and Zhuosheng Zhang. 2025a. Quick on the uptake: Eliciting implicit intents from human demonstrations for personalized mobile-use agents. arXiv preprint arXiv:2508.08645.

Zhiyong Wu, Zhenyu Wu, Fangzhi Xu, Yian Wang, Qiushi Sun, Chengyou Jia, Kanzhi Cheng, Zichen Ding, Liheng Chen, Paul Pu Liang, and 1 others. 2025b. Os-atlas: Foundation action model for generalist gui agents. In International Conference on Learning Representations, volume 2025, pages 5090– 5108.

Zongru Wu, Rui Mao, Zhiyuan Tian, Pengzhou Cheng, Tianjie Ju, Zheng Wu, Lingzhong Dong, Haiyue Sheng, Zhuosheng Zhang, and Gongshen Liu. 2025c. See, think, act: Teaching multimodal agents to effectively interact with gui by identifying toggles. arXiv preprint arXiv:2509.13615.

Yifan Xu, Xiao Liu, Xinghan Liu, Jiaqi Fu, Hanchen Zhang, Bohao Jing, Shudan Zhang, Yuting Wang, Wenyi Zhao, and Yuxiao Dong. 2025. Mobilerl: Online agentic reinforcement learning for mobile gui agents. arXiv preprint arXiv:2509.18119.

Yan Yang, Dongxu Li, Yutong Dai, Yuhao Yang, Ziyang Luo, Zirui Zhao, Zhiyuan Hu, Junzhe Huang, Amrita Saha, Zeyuan Chen, and 1 others. 2025. Gta1: Gui test-time scaling agent. arXiv preprint arXiv:2507.05791.

Runpeng Yu, Xinyin Ma, and Xinchao Wang. 2025. Auto-controlled image perception in mllms via visual perception tokens. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 21822–21831.

Jiwen Zhang, Jihao Wu, Teng Yihua, Minghui Liao, Nuo Xu, Xiao Xiao, Zhongyu Wei, and Duyu Tang. 2024. Android in the zoo: Chain-of-action-thought for gui agents. In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 12016–12031.

Zhuosheng Zhang and Aston Zhang. 2024. You only look at screens: Multimodal chain-of-action agents. In Findings of the Association for Computational Linguistics: ACL 2024, pages 3132–3149.

Yuqi Zhou, Sunhao Dai, Shuai Wang, Kaiwen Zhou, Qinglin Jia, and Jun Xu. 2026. Gui-g1: Understanding r1-zero-like training for visual grounding in gui agents. Advances in Neural Information Processing Systems, 38:95683–95705.

### A End-to-End Algorithm

Algorithm 1 summarizes the full GUI-CIDER pipeline, including knowledge-rich sample synthesis, density-aware exemplar reselection, and midtraining on the retained corpus.

Algorithm 1 GUI-CIDER

Input: D, Mexp, ϕ, K, α, λ, γ Output: Xhigh, θ

- 1: Initialize synthesized corpus X ← ∅
- 2: for all τ = (T, {(st, at, vt)}Lt=1) ∈ D do
- 3: S ← P(T; Mexp)
- 4: for t = 1 to L do
- 5: anlt ← B(at, vt)
- 6: Convert st−1, st to text descriptions dt−1, dt
- 7: Rt ← C(T, dt−1, anlt , dt | Mexp)
- 8: xt ← ⟨T, S, anlt , Rt⟩
- 9: X ← X ∪ {xt}
- 10: end for
- 11: end for
- 12: for all x ∈ X do
- 13: f(x) ← tanh(K(x)/γ)
- 14: Compute r(x) by the ratio of local K-NN distance to global mean distance
- 15: end for
- 16: Min-max normalize {r(x)}x∈X into {d(x)}x∈X
- 17: for all x ∈ X do
- 18: bx ← 1+αd1 (x)

- 19: g(x) ← bx + λf(x)(1 − bx)
- 20: Sample ξx ∼ Uniform(0, 1)
- 21: if ξx ≤ g(x) then
- 22: Xhigh ← Xhigh ∪ {x}
- 23: end if
- 24: end for
- 25: Mid-train agent parameters θ on Xhigh with the causal LM loss Lmid
- 26: return Xhigh, θ

### B Additional Mathematical Proofs

In this appendix, we provide supplementary theoretical results for the density-aware exemplar reselection rule used in GUI-CIDER. Throughout, let

1 1 + αd

, (13)

b(d) =

so that the retention function in Section 4 can be rewritten as

g(x) = b(d(x)) + λf(x) 1 − b(d(x))

1 + αλf(x)d(x) 1 + αd(x)

.

=

(14)

This form makes explicit that g(x) interpolates between a density-based baseline b(d) and a causalsaliency correction controlled by λ.

#### B.1 Range, Boundary Cases, and Parameter Interpretation

Proposition A.1 (Range and Boundary Cases). For any sample x with f(x) ∈ [0,1], d(x) ∈ [0,1], α > 0, and λ ∈ [0,1], the retention score satisfies

1

1 + αd(x) ≤ g(x) ≤ 1 (15) and

λf(x) ≤ g(x) ≤ 1. (16) Moreover, the following boundary cases hold:

g(x) = 1 if d(x) = 0, (17)

1 1 + αd(x)

if f(x) = 0, (18) and

g(x) =

g(x) = 1 if λ = 1 and f(x) = 1. (19)

Proof. Since f(x) ∈ [0,1] and λ ∈ [0,1], we have

g(x) = b(d(x)) + λf(x) 1 − b(d(x)) ≥ b(d(x)) =

1 1 + αd(x)

.

Likewise,

(20)

g(x) ≤ b(d(x)) + 1 − b(d(x)) = 1. (21) For the second lower bound,

g(x) − λf(x) = b(d(x)) 1 − λf(x) ≥ 0

(22)

which implies g(x) ≥ λf(x). The boundary identities follow directly by substitution into the closed form of g(x).

#### B.2 Proofs of the Four Desiderata

Proposition A.2 (Causal Monotonicity). For fixed d(x), the retention function is monotonically non-decreasing in f(x).

Proof. For a fixed density d, the derivative of g with respect to f is

∂g ∂f

1 1 + αd

= λ 1 −

αd 1 + αd

. (23)

= λ

Since λ ≥ 0, α > 0, and d ≥ 0, we have ∂f∂g ≥ 0. Therefore, g(x) is monotonically non-decreasing

in f(x). It is strictly increasing whenever λ > 0 and d > 0.

Proposition A.3 (Density Penalty). For fixed f(x), the retention function is monotonically nonincreasing in d(x).

Proof. Differentiating g with respect to d gives

α(1 − λf) (1 + αd)2

∂g ∂d

. (24)

= −

Because f ∈ [0,1] and λ ∈ [0,1], we have 1 − λf ≥ 0. Hence ∂d∂g ≤ 0, so denser samples always receive no larger retention scores.

Proposition A.4 (Density Order Preservation). The reweighted density score d · g(x) is strictly increasing in d:

∂ ∂d

d · g(x) > 0. (25)

Proof. Using the closed form of g,

d + αλfd2 1 + αd

. (26)

d · g(x) =

Applying the quotient rule yields

∂ ∂d

(1 + 2αλfd)(1 + αd) (1 + αd)2 −

d · g(x) =

α(d + αλfd2) (1 + αd)2

.

After simplification,

(27)

∂ ∂d

1 + 2αλfd + α2λfd2 (1 + αd)2

. (28)

d · g(x) =

Every term in the numerator is non-negative, and the constant term is strictly positive. Therefore,

∂ ∂d

d · g(x) > 0. (29)

Thus, the reweighted density preserves the original ordering induced by d.

Proposition A.5 (Density-Causal Synergy). The retention rule exhibits positive interaction between causal saliency and density:

∂2g(x) ∂f∂d

> 0. (30)

Proof. From Proposition A.2,

∂g ∂f

αd 1 + αd

. (31)

= λ

Differentiating again with respect to d gives

∂2g ∂f∂d

(1 + αd) − αd (1 + αd)2

= λα

λα (1 + αd)2

=

.

(32)

For λ > 0 and α > 0, the above quantity is strictly positive. Hence the marginal value of causal saliency becomes larger in denser regions, which is exactly the desired synergy effect.

- B.3 Expected Retained Corpus Size and Information Preservation

The next result formalizes the intuition that stochastic retention preserves a non-trivial fraction of the original reasoning signal.

Proposition A.6 (Expectation of the Retained Corpus). Let 1x be the Bernoulli indicator of whether sample x is retained, namely 1x = 1 if ξx ≤ g(x) and 1x = 0 otherwise. Then

g(x). (33)

E |Xhigh| =

x∈X

In particular,

1 1 + αd(x) ≤ E |Xhigh| ≤ |X|. (34)

x∈X

Proof. By construction, 1x is a Bernoulli random variable with mean g(x). Therefore,

Proposition A.7 (Lower Bound on Preserved Causal Information). Define the causal information mass of a subset A ⊆ X as

f(x). (36)

If(A) =

x∈A

Then the expected preserved causal information ratio satisfies

f(x)g(x) x∈X f(x)

E[If(Xhigh)] If(X)

= x∈X

f(x)2 x∈X f(x)

(37)

≥ λ x∈X

≥ λf¯

where f¯= |X|1 x∈X f(x) is the average causal saliency of the original corpus.

Proof. Since If(Xhigh) = x 1xf(x), linearity of expectation gives

E[If(Xhigh)] =

f(x)E[1x]

x∈X

(38)

=

f(x)g(x).

x∈X

By Proposition A.1, g(x) ≥ λf(x) for every x, hence

f(x)2. (39)

f(x)g(x) ≥ λ

x∈X

x∈X

Dividing both sides by x f(x) proves the first inequality. For the second inequality, Cauchy–

Schwarz implies

f(x) 2 |X|

f(x)2 ≥ x∈X

, (40)

x∈X

and therefore

f(x)2 x f(x) ≥ λ x

f(x) |X|

λ x

= λf.¯

(41)

|Xhigh| =

1x,

x∈X

E |Xhigh| =

E[1x] =

g(x).

x∈X

x∈X

(35)

The bounds follow immediately from Proposition A.1.

This shows that the thinning process retains a guaranteed fraction of the original causal signal in expectation.

#### B.4 Stability Under Score Estimation Errors

In practice, both f(x) and d(x) are estimated from synthesized text and embedding geometry. The next result shows that moderate perturbations in

[Figure 4]

- Figure 4: Examples from GUI Knowledge Bench.

[Figure 5]

- Figure 5: Examples from MMBench-GUI L1.

these quantities induce controlled perturbations in g(x).

Proposition A.8 (Lipschitz Stability). Let fˆ(x) and dˆ(x) be perturbed estimates satisfying

|fˆ(x) − f(x)| ≤ εf, |dˆ(x) − d(x)| ≤ εd.

(42) If gˆ(x) = g(fˆ(x),dˆ(x)), then

α 1 + α

εf. (43)

|gˆ(x) − g(x)| ≤ αεd + λ

Proof. By the mean value theorem applied to the bivariate function g(f,d), there exists a point on the line segment joining (f,d) and (f,ˆ dˆ) such that

|gˆ − g| ≤ sup|∂dg| · |dˆ− d|

+ sup|∂fg| · |fˆ− f|.

From the derivatives established above,

(44)

and

α(1 − λf) (1 + αd)2 ≤ α (45)

|∂dg| =

αd 1 + αd ≤ λ

|∂fg| = λ

α 1 + α

(46)

because d ∈ [0,1]. Substituting the error bounds yields

|gˆ(x) − g(x)| ≤ αεd + λ

α 1 + α

εf.

(47)

Hence the retention rule is stable under bounded score estimation noise.

### C Benchmark Details and Examples

We briefly summarize the five benchmarks used in our experiments and point readers to the corresponding examples in the appendix.

GUI Knowledge Bench. GUI Knowledge Bench is a diagnostic benchmark for GUI-specific knowledge rather than end-to-end execution. It evaluates whether a model can understand widget functions, interface states, interaction effects, and workflow progress across diverse GUI platforms. Representative examples are shown in Figure 4.

MMBench-GUI L1. MMBench-GUI is a hierarchical cross-platform benchmark for GUI agents, and the L1 split used here focuses on GUI content understanding. Its examples test whether a model

[Figure 6]

Figure 6: Examples from GUI agent task completion benchmarks.

[Figure 7]

Figure 7: Examples of GUI-CIDER Synthetic Data.

can read interface content and reason about the semantics and relative placement of GUI elements. Representative examples are shown in Figure 5.

AITZ. AITZ (Android-In-The-Zoo) is an Android GUI navigation benchmark built from screenaction pairs with Chain-of-Action-Thought annotations. It evaluates whether an agent can infer the next GUI action from the current screen, prior context, and task instruction. A representative example is shown in Figure 6.

AndroidControl. AndroidControl studies realworld Android control at scale using human demonstrations paired with both high-level and low-level instructions. It is designed to evaluate how well agents follow everyday mobile tasks under realistic data diversity and task complexity.

GUI-Odyssey. GUI-Odyssey focuses on longhorizon cross-app mobile navigation, where successful completion requires carrying context across multiple apps and steps. Compared with single-app benchmarks, it places heavier demands on history tracking, planning, and cross-app reasoning.

### D Examples of Synthetic Data

Figure 7 presents a representative synthetic training example produced by GUI-CIDER. Starting from a task and a concrete GUI transition, GUI-CIDER organizes the annotation into a step list, action intent, pre-state description, post-state description, trigger, mechanism, and reasoning. This format converts raw interaction traces into structured supervision that captures not only what action should be taken, but also why the action is appropriate in the current

[Figure 8]

Figure 8: Keyword lexicon used in Stage 1 data synthesis (Part I). The listed markers target conditional and hypothetical reasoning, purpose and intent, and explicit causal chains or UI mechanisms. We use these lexical cues as lightweight anchors when eliciting and auditing reasoning-rich rationales from synthesized trajectories.

GUI context.

### E Keyword and Prompt Templates

Figures 8 and 9 summarize the lexical categories used in our data-synthesis pipeline to surface reasoning-rich textual patterns. These categories cover conditional and hypothetical statements, purpose and intent, explicit causal chains, temporal ordering, evidential language, verification cues, and comparison markers. In our pipeline, they are used as lightweight anchors for prompt design and quality inspection, helping the expert model generate rationales that more consistently expose triggers, mechanisms, and outcome-oriented reasoning.

Figure 10 presents the prompt templates used to instantiate the three key modules in Stage 1: the planning function P, the semantic behavioral grounding function B, and the causal analyst C. Together, these templates standardize how raw task descriptions, low-level GUI actions, and paired pre-/post-state screenshots are converted into the textual tuple ⟨T,S,anlt ,Rt⟩ used in mid-training.

### F Experimental Details

In our experiments, the Planning function is implemented using deepseek-v4-flash, while both the mapping function and the causal analyst are based on Qwen3-VL-32B-Instruct. The base models are Qwen3-VL-4B-Instruct and Qwen3-VL-8BInstruct. We set the per-device training batch size to 4, gradient accumulation steps to 2, learning rate

[Figure 9]

Figure 9: Keyword lexicon used in Stage 1 data synthesis (Part II). This part focuses on temporal markers, reasoning and evidence expressions, verification cues, and comparison or analogy terms, which help identify ordering constraints, supporting evidence, and contrastive reasoning in textualized GUI transitions.

Action Type Action Description

CLICK Click at specified position. TYPE Enter text at designated location.

SCROLL Scroll in direction. PRESS_BACK Go to previous screen. PRESS_HOME Go to home page.

ENTER Press enter button. OPEN_APP Open specified app.

WAIT Wait for screen to load. LONG_PRESS Long press at specified position. COMPLETE Indicate task finished. IMPOSSIBLE Indicate task impossible.

Table 6: Action space in our experiment.

to 1.0e-5, and train for 2 epochs. Moreover, for the post-training mentioned in the main text, we used a full-scale SFT approach, using the same training set as GUI-CIDER for training and the same test set for evaluation. For the same data, applying mid-training with GUI-CIDER followed by posttraining yields improvements compared to direct post-training. This demonstrates that GUI-CIDER can further unlock the potential of the data. The total computational cost amounts to 1,400 hours on 80G GPUs. For GUI agent task completion benchmarks, we adhere to the assessment methods of existing works: for actions with coordinates such as CLICK and LONG_PRESS, a error of less than 14% is considered correct. For TYPE actions, an F1 score greater than 0.5 is required to be counted as correct. In all other cases, exact matching is necessary for correctness. And TSR for a task will be 1 only if SR for every single frame within that task is 1. Action space is shown as Table 6.

[Figure 10]

###### Figure 10: Prompt templates used to instantiate the planning function P, the semantic mapping function B, and the causal analyst C.

