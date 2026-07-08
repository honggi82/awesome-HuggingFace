# arXiv:2510.25682v3[cs.CL]8Feb2026

[Figure 1]

[Figure 2]

## PairUni: Pairwise Training for Unified Multimodal Language Models

Jiani Zheng, Zhiyang Teng, Kunpeng Qiu, Xiangtai Li, Anran Wang, Yu Tian, Ye Tian, Haochen Wang, Zhuochen Wang

ByteDance

### Abstract

Unified Vision-Language Models (UVLMs) perform both understanding and generation within a single architecture. Since these models rely on heterogeneous data and supervision, balancing both generation and understanding in reinforcement learning (RL) is challenging. To address this challenge, we propose PairUni, a unified framework that reorganizes data into understanding–generation (UG) pairs and aligns optimization accordingly. Specifically, we construct a unified paired dataset by synthesizing aligned instances via cross-modal semantic completion and retrieving semantically related samples. These paired structures expose cross-task semantic correspondences and support consistent policy learning. To leverage this structure, we present PairGRPO, a pair-aware variant based on Group Relative Policy Optimization. It assigns a similarity score to each pair to modulate the advantage, strengthening learning from well-aligned examples and reducing task interference. Extensive experiments across diverse UVLM architectures (Autoregressive and Discrete Diffusion) and scales (1B to 14B) demonstrate that PairUni yields consistent improvements over strong baselines. Notably, our method also demonstrates strong generalization by improving performance on image editing tasks without using any editing-specific data. Code: https://github.com/Haochen-Wang409/PairUni

Date: February 10, 2026 †Correspondence: zhengjiani.0123@bytedance.com, zhiyang.teng@bytedance.com,

### 1 Introduction

UVLMs have demonstrated strong performance in both multimodal understanding and image generation tasks with different architectures [3, 8, 24, 41]. However, as evaluation protocols increasingly emphasize complex, multi-step reasoning—spanning mathematics, the natural sciences, and multi-hop visual question answering—the objective of training a single system that simultaneously balances these capabilities within a unified learning paradigm remains highly challenging [5, 20, 23].This challenge is particularly acute during reinforcement learning (RL) stages, since understanding and generation are supervised with heterogeneous objectives and data formats, making the optimization process highly sensitive to data batching and cross-task credit assignment.

This fundamental disparity has largely led to various advancements. For instance, text-to-image RL has focused on improving object controllability and prompt adherence [13, 29], while visual reasoning improvements have separately targeted accuracy on benchmarks for math or science. Consequently, attempts at unified RL encounter significant practical obstacles: (i) task interference during joint optimization, where gains on one

[Figure 3]

- Figure 1 Data Pairing Pipeline. Left: Illustrative examples of aligned quadruples constructed from generation and understanding tasks. Right: Schematic of the proposed pairing strategy, which utilizes retrieval and clustering techniques to synthesize high-quality paired data (PairUG-16k) for unified training.

objective cause regressions on the other [14]; (ii) a broad and diverse understanding task space (e.g., math, charts, OCR) that resists a single reward design; and (iii) limited guidance on how to select and align data for unified RL at scale, constraining both stability and ceiling performance [12].

Prevailing GRPO-based strategies often sidestep the central issue by focusing on a single capability, such as using understanding signals to improve generation quality [13], or by adopting multi-stage schemes that alternate between tasks to find a fragile balance [14]. While effective to a degree, these methods do not directly tackle the core source of conflict: the lack of data-level semantic alignment between understanding and generation supervision and the absence of an optimization rule that respects this alignment. As a result, the shared policy is driven by competing gradients from unrelated signals, leading to unstable updates and uneven performance gains across tasks. This optimization-level conflict is not merely theoretical; it is empirically measurable and provides the direct motivation for our work.

We address this with PairUni, a simple yet unified RL framework that aligns the problem at both the data and optimization levels. On the data side, we reorganize heterogeneous supervision into understanding–generation (UG) pairs centered on the same or closely related images. Two complementary pair types are constructed. Aligned pairs are formed by completing single-task samples into unified quadruples, using GPT-o3 to add the missing caption or prompt for understanding data or to synthesize a question–answer pair for generation data, so that both objectives share the same instance. We use clustering method to select representative high-quality medoids from the unified quadruples. Retrieved pairs link a generation sample to a semantically related understanding sample via similarity search over image embeddings, which expands coverage when exact matches are scarce. This paired view exposes cross-task correspondences, namely what to attend to for understanding and how to express it for generation, on related content rather than on unrelated batches.

On the optimization side, we develop PairGRPO, a pair-aware variant of GRPO that modulates the advantage by pair similarity. Aligned pairs receive full weight, and retrieved pairs are down-weighted by their pairsimilarity scores. This mechanism strengthens updates from high-quality supervision and tempers weaker matches, which reduces cross-task interference while preserving GRPO stability through the clipped objective and KL regularization. In effect, the policy update is made to respect the semantic alignment already present in the data.

Our contributions are summarized as follows:

- • We propose PairUni, a unified RL framework that organizes data into UG pairs and introduces PairGRPO to modulate optimization via similarity, effectively mitigating task interference.

- • We construct PairUG-16k, a high-quality paired dataset specifically designed to support unified RL fine-tuning.
- • Experiments show that PairUni achieves balanced improvements on Janus-Pro, outperforming strong baselines, and generalizes effectively to discrete diffusion models like Lumina-DiMOO.

These findings support a simple conclusion: pairing the data and weighting the advantage by pair-similarity is a general and effective ingredient for unified multimodal training, which improves understanding and generation together rather than trading one for the other.

### 2 Method

PairUni has two key components: a data pairing pipeline (as shown in Figure 1) to generate training pairs for unified models (Section 2.1) and a PairGRPO algorithm (as shown in Figure 2 of Section 2.2), which is specially designed for RL of understanding-generation pairs.

#### 2.1 Pairing Understanding–Generation Pairs

We aim to construct a unified paired dataset S = {(I,C,Q,A)}, where each data item supports both generation and understanding capabilities within a single multimodal model. I is an image input, C is a text prompt that describes or motivates the image (used in generation), Q is a visual understanding question and A is the corresponding answer to Q. We build this paired dataset from two distinct sources: 1) Understanding data U = {(I,Q,A)}, where the image is annotated with comprehension questions; 2) Generation data G = {(I,C)}, where an image is paired with a generative prompt. These two sources are inherently heterogeneous and rarely aligned. To unify them, we design a data pipeline that constructs either aligned or retrieval-based pairs (see Appendix B for detailed statistics).

##### 2.1.1 Generating Aligned Pairs

Given U and G, we first augment each element in the original datasets to the desired quadruples and then we design an algorithm to select representative pairs to construct S.

Algorithm 1 Pseudocode for the Data Pairing Algorithm.

- 1: Input: features Fu, Fg, clusters K, neighbors n
- 2: F ←L2Norm(Fu∪Fg)
- 3: C ←MiniBatchKMeans(F, K)
- 4: for k = 1..K do
- 5: Ik ←{i : assign(xi) = k}
- 6: i∗←arg maxi∈Ik⟨fi, ck⟩
- 7: Daligned←Daligned ∪ {(xi∗, yi∗)}
- 8: end for
- 9: Furem←Fu\Daligned; Fgrem←Fg\Daligned
- 10: for xgi ∈ Fgrem do
- 11: J ←top−n kNN(ϕgi , Furem)
- 12: Dret←Dret ∪ {(xuj , xgi ) : j ∈J };
- 13: Furem ← Furem \ J , Fgrem ← Fgrem \ {xgi }
- 14: end for
- 15: Output: Daligned ∪ Dret

Cross-Modal Semantic Completion We view the construction of aligned pairs not merely as augmentation, but as a semantic completion process that bridges the information gap between tasks. For understanding-only samples (I,Q,A) ∈ U, we employ GPT-o3 to generate the generation prompt C by conditioning on both the image I and the QA pair (Q,A). The instruction explicitly requires that C must include the visual details necessary to answer Q (e.g., if Q asks about counting cats, C must explicitly describe the number of cats). Conversely, for generation-only samples (I,C) ∈ G, we use GPT-4o to generate (Q,A) by conditioning on C, ensuring that the question targets the specific

visual elements described in the generation prompt. Crucially, we introduce a consistency check loop: after generation, we verify whether the synthesized component contradicts the original data (e.g., ensuring the generated C does not hallucinate objects absent in I). This generate-then-verify pipeline ensures that the resulting quadruples (I,C,Q,A) are strictly semantically bound, eliminating semantic drift and ensuring that reward signals for both tasks are grounded in the exact same visual context.

Data Selection Even after augmentation, many samples may be redundant or low-quality. To ensure coverage and diversity, we adopt a clustering-based strategy to identify representative aligned pairs (Daligned) from these datasets as shown in the first part of Algorithm 1. First, we extract image features using pretrained

[Figure 4]

- Figure 2 Framework of PairUni: Overview of the proposed architecture, featuring a dual-component design that integrates a sophisticated data processing pipeline with the GRPO reinforcement learning algorithm to achieve unified multimodal understanding and generation.

visual encoders, and then perform K-means clustering over the joint visual feature space of U ∪ G. Second, for each cluster, we select the most central sample (i.e., the medoid) as a canonical representation of that cluster’s content. This yields a curated set of self-referential pairs where understanding and generation annotations coexist for the same image. These samples are both semantically rich and geometrically representative of the data distribution, forming a strong backbone for joint training.

##### 2.1.2 Constructing Retrieval-Based Pairs

While aligned pairs are semantically precise, their quantity is limited. To scale supervision, we introduce retrieval-based pairs Dret, where understanding and generation samples come from different images but share visual similarity. The second part of Algorithm 1 shows the algorithm. The main idea is to extract visually similar image pairs from two datasets to establish correspondences between “understand” and “generate” data. First, cosine similarity is computed across all remaining generation–understanding image pairs. For each generation image, we retrieve the top-n most similar understanding images above a similarity threshold δ. A greedy matching algorithm is used to ensure that each image is only used once. This retrieval mechanism exploits the fact that semantically similar images often support related tasks, even if not identical. By leveraging these approximate matches, the model can learn cross-instance generalization, which enhances its robustness and expands training coverage.

Together, Daligned and Dret form the UG pair set S (referred to as PairUG-16k) used for policy optimization. These two pathways provide complementary benefits: aligned pairs deliver precise, high-quality supervision, while retrieval pairs enhance scale and semantic diversity, covering a wide spectrum of multimodal understanding and generation tasks.

#### 2.2 PairGRPO

This section describes: (1) vanilla GRPO with mixed tasks; (2) pairwise GRPO with UG-pairs; (3) PairGRPO, which incorporates pair similarity into advantage weighting. The formulation is model-agnostic and can be applied to both autoregressive and discrete diffusion backbones. Each step is designed to better align understanding and generation, reduce conflict, and stabilize learning.

- (1) Vanilla GRPO with mixed tasks. We consider a batch of trajectories B = {τi}Ni=1 mixed with understanding and generation tasks. For each trajectory τi with input q and output o1:T, we compute the token-wise

θ(ot|q,o<t)

importance ratio ρt(θ) = π

πθold(ot|q,o<t). Rewards r (rUnd or rGen) are normalized within groups sharing the

###### same prompt to obtain the advantage At = (r − µr)/σr. The objective maximizes the clipped surrogate loss:

T

Lclip(ρt, At) − βDKL(πθ||πold), (1)

Jvanilla(θ) = Eτ∼πθ

old

t=1

where ε is the clipping threshold and β controls the KL penalty (default β = 0).

Reward Functions Our reward functions are tailored to the specific goals of understanding and generation. For understanding tasks, typically formulated as multiple-choice question answering, we use prediction accuracy as our reward, a standard metric that directly measures correctness: rUnd = Acc(ypred,ytrue).

For generation tasks, we employ the HPSv2 reward model [47] to evaluate output quality. The reward is given by:rGen = RHPSv2(x,ygen), where x is the input prompt and ygen is the corresponding generated image.

- (2) Pairwise GRPO with UG data pairs. To mitigate the task interference inherent in mixed-batch training (as

discussed in Section 1), we reorganize training around a set of paired training examples P = {p}Mp=1. Each pair p consists of two datapoints: one generation example and one understanding example that are semantically aligned. This pairing is defined at the data level, not at the trajectory level: each data item in the pair can produce multiple trajectories through sampling.

For each paired datapoint p, we generate a set of trajectories {τp,k(u)}K

u

k=1 for the understanding side, and {τp,k(g)}Kk=1g for the generation side. These trajectories are grouped by task type and prompt to compute group-relative advantages. Specifically, we calculate A(tu) and A(tg) as the normalized reward within the respective task-type group, using GRPO’s group-based normalization.

The pairwise GRPO objective is then defined as:

Jpair(θ) = Ep∼P

τ∈{τp(u)} t∈τ

Lclip ρt, A(tu) +

τ∈{τp(g)} t∈τ

Lclip ρt, A(tg) ,

(2)

This formulation ensures that policy gradients from understanding and generation are derived from semantically related training examples, even when multiple trajectories are sampled per side. This pairing encourages more consistent policy updates across tasks.

- (3) PairGRPO: similarity-weighted advantage. This design is motivated by an empirical observation: when understanding and generation data are semantically aligned, the cosine similarity between their gradients increases (see Figure 8 in Appendix D). Higher gradient agreement correlates with better unified performance, suggesting that weighting updates by semantic alignment can reduce task interference. To further align training strength with semantic similarity, we introduce a pair-level similarity score sp ∈ [0,1] for each data pair p, computed via a pretrained image encoder. Based on this, we assign a pair weight wp:

1, if p ∈ Daligned, √sp, if p ∈ Dret

wp =

and apply this weight to all advantages computed from trajectories originating from the pair:

A(tu) = wp A(tu), A(tg) = wp A(tg).

(3)

We use the square root to amplify the relative differences between similarity scores, as all selected pairs are

drawn from a high-similarity candidate pool. The full PairGRPO objective becomes:

JPairUni(θ) = Ep∼P

Lclip ρt, A(tu) +

τ∈{τp(u)} t∈τ

Lclip ρt, A(tg) .

τ∈{τp(g)} t∈τ

(4)

This design modulates the trajectory-level credit assignment based on the quality of semantic pairing, strengthening updates from well-aligned pairs (aligned: wp = 1) while attenuating noisy or weakly aligned ones (retrieved: wp = √sp). By co-optimizing with the structure of PairUG-16k, PairGRPO retains the stability of GRPO while effectively leveraging data-level semantic alignment to resolve optimization-level task conflicts.

### 3 Experiments

Training We adopt Janus-Pro [6] as a primary baseline because it is widely used as a comparator for unified multimodal understanding and generation and exhibits competitive performance. All experiments are conducted on 8× H100 GPUs. For the 7B model, we use a rollout size of 4 for both text and image generation and train for at most 1200 steps. The per-device batch size is 2 (global batch size 16). We set the classifier-free guidance (CFG) weight to 5, β = 0, the learning rate to 1 × 10−6, and the sampling temperature to 1.0. For the 1B model, the rollout size for both modalities is increased to 8. All the visual features are extracted using a ResNet50 encoder by removing the classification head [16] and L2-normalized (we compare different image encoders in Appendix C). We use the Orsta data [25] as the understanding data U, which contains about 47K samples, and the BLIP3o data [3] as the generation dataset G, which contains about 60K samples. We exclude the original detection and grounding QA pairs from Orsta since Janus-Pro fails on these tasks. The similarity threshold is 0.6. Our constructed PairUG-16k dataset consists of 16,320 samples: including 4,971 aligned pairs and 11,349 retrieval-based pairs. The similarity score of each pair is provided (see Appendix B for dataset details).

Evaluation We evaluate multimodal understanding on MME Perception (basic reasoning), POPE [20] (hallucination), MMStar [5] (visual reasoning), and MMMU [54] (multi-discipline tasks). For generation, we use GenEval [10] to measure visual fidelity and WISE [28] to assess knowledge-grounded synthesis.

#### 3.1 Main Results

We first present the main results on Janus-Pro 1B and 7B, demonstrating that PairUni effectively addresses the trade-off typically observed in unified modeling, leading to simultaneous improvements in both understanding and generation.

Multimodal Understanding. Table 1 reports the performance of different models on four representative multimodal understanding benchmarks. On the broad-coverage MMMU benchmark, which emphasizes general visual reasoning across scientific, mathematical, and commonsense domains, PairUni achieves substantial improvements among unified models: at the 1B scale it attains a score of 40.4 (cf. Janus-Pro-1B: 36.3; ULM-R1†: 40.3), and at the 7B scale it reaches 47.0, surpassing prior unified baselines such as Janus-Pro-7B (41.1) and UniToken (32.8). Beyond MMMU, PairUni also improves performance on MMStar and MME: at 7B, it increases MMStar from 46.5 to 49.5 and MME from 1567.1 to 1597.7, while maintaining competitive POPE performance (88.0 vs. 87.4), thereby demonstrating gains on both perception-intensive and knowledge-intensive tasks. At the 1B scale, PairUni attains 46.4 on MMStar and 1483.2 on MME with stable POPE performance (86.4), indicating that the proposed data pairing strategy and PairGRPO optimization remain effective under stringent capacity constraints. These consistent improvements suggest that PairUni enhances unified visual reasoning capabilities without compromising perceptual robustness.

Text-to-Image Generation. We evaluate text-to-image generation on the WISE [28] and GenEval [10] benchmarks. On WISE, as shown in Table 2, PairUni achieves the highest overall performance among unified

- Table 1 Main Results on multimodal understanding benchmarks. Comprehensive comparison of PairUni against state-of-the-art methods across various tasks and model sizes. The PairUni method consistently achieves superior performance, validating the effectiveness of the unified approach.

Model LLM MMMU MMStar MME(P) POPE

Understanding Only InternVL3 [57] Qwen2.5-1.5B [1] 48.6 60.7 - 89.6 Qwen2.5-VL [1] Qwen2.5-3B 51.2 56.3 - 85.9 LMM-R1 [30] Qwen2.5-3B - 58.0 - -

Unified Understanding and Generation Show-o[49] Phi-1.3B [11] 26.7 - - 80.0 HermesFlow [52] Phi-1.3B 28.3 - - 81.4 Janus-Pro-1B [6] DeepSeek-LLM-1.5B [7] 36.3 - 1444.0 86.2 ULM − R1† [14] DeepSeek-LLM-1.5B 40.3 - - PairUni-1B DeepSeek-LLM-1.5B 40.4 46.4 1483.2 86.4 Orthus [17] Chameleon-7B [42] 28.2 - 1265.8 79.6 VILA-U [48] LLaMA-2-7B [43] - - - 85.8 UniToken [15] Chameleon-7B 32.8 46.1 - Janus-Pro-7B [6] DeepSeek-LLM-7B 41.1 46.5 1567.1 87.4 DSR [12] DeepSeek-LLM-7B 41.1 - - 86.6 PairUni-7B DeepSeek-LLM-7B 47.0 49.5 1597.7 88.0

[Figure 5]

- Figure 3 Case Study: Qualitative comparison of generated images between Janus-Pro-7B and PairUni. The results demonstrate PairUni’s superior ability to follow complex prompts and generate high-fidelity images.

models at both parameter scales, with scores of 0.38 (1B) and 0.45 (7B). At the 1B scale, it improves over Janus-1B (0.23) and ULM-R1 (0.33), and at the 7B scale it surpasses Janus-Pro-7B (0.35) and Emu3 (0.39), thereby substantially reducing the performance gap with generate-only models. Because ULM-R1 [14] is not open-sourced, we report its performance only after Unified-RL and Refined-RL training on approximately 40k samples. At the subtask level, PairUni notably enhances performance on Space (0.56 at 1B; 0.62 at 7B) and Physics (0.44 at 1B; 0.55 at 7B), indicating improved grounding in geographic and physical commonsense.

On GenEval (Table 3), PairUni demonstrates strong compositional generalization at both scales. At 1B, PairUni-1B attains the highest overall score (0.79), exceeding Janus-Pro-1B (0.73) and ULM-R1 (0.76). At 7B, PairUni-7B achieves 0.85, outperforming Janus-Pro-7B (0.79) and DSR (0.84), the latter exhibiting comparatively weaker semantic understanding and a substantially underperforming 1B variant. Although Janus-Pro-R1 slightly outperforms PairUni-7B by a margin of one point (0.86), its understanding metrics and WISE scores are not reported.

These improvements are consistent with our gains on WISE and suggest that PairGRPO more effectively enforces constraint adherence and object–attribute binding. Collectively, the results indicate that PairUni yields gains on both sides of the unified objective, in contrast to baseline models that typically specialize in either understanding or generation, but not both (e.g., InternVL3 for understanding, FLUX.1 for generation).

- Table 2 Results on the WISE Benchmark. PairUni achieves the highest overall score among compared methods, with particularly outstanding performance in the space and physics subtasks, demonstrating its robust generation capabilities.

Model Cultural↑ Time↑ Space↑ Biology↑ Physics↑ Chemistry↑ Overall

Generating Only PixArt-α [4] 0.45 0.50 0.48 0.49 0.56 0.34 0.47 playground-v2.5 [19] 0.49 0.58 0.55 0.43 0.48 0.33 0.49 SD-v1-5 [38] 0.34 0.35 0.32 0.28 0.29 0.21 0.32 SD-XL-base-0.9 [31] 0.43 0.48 0.47 0.44 0.45 0.27 0.43 FLUX.1-dev [18] 0.48 0.58 0.62 0.42 0.51 0.35 0.50

Unified Understanding and Generation VILA-U [48] 0.26 0.33 0.37 0.35 0.39 0.23 0.31 Janus-1B [6] 0.16 0.26 0.35 0.28 0.30 0.14 0.23 ULM − R1† [14] - - - - - - 0.33 PairUni-1B 0.31 0.39 0.56 0.38 0.44 0.22 0.38 Emu3 [45] 0.34 0.45 0.48 0.41 0.45 0.27 0.39 Janus-Pro-7B [6] 0.30 0.37 0.49 0.36 0.42 0.26 0.35 PairUni-7B 0.36 0.46 0.62 0.42 0.55 0.29 0.45

Additional qualitative examples are provided in Appendix F. These results not only validate the efficacy of our proposed framework but also highlight the potential of unified learning paradigms to surpass specialized models in complex multimodal tasks.

Generalization Across Architectures and Scales. To demonstrate that PairUni captures a general principle of unified learning rather than being specific to one architecture, we evaluate its effectiveness across different model families and scales. First, we test PairUni on Lumina-DiMOO, a discrete diffusion model that differs fundamentally from the autoregressive Janus-Pro. As shown in Table 4 (a), PairUni yields consistent gains: MMMU improves from 58.6 to 61.3 (+2.7), MMStar from 52.4 to 52.6 (+0.2), and GenEval from 0.88 to 0.89. Second, we verify scalability on Bagel, a larger 14B unified model. As presented in Table 4 (b), PairUni outperforms the random pairing baseline on Bagel-14B, boosting MMMU to 54.0 and POPE to 89.21, while also improving generation quality. These results, combined with our main experiments on Janus-Pro 1B and 7B, confirm that PairUni is effective across diverse architectures and scales ranging from 1B to 14B. More training details for these models are provided in Appendix E.

Zero-shot Improvement in Editing Capabilities. A key hypothesis is that bidirectional alignment should enhance the model’s ability to follow complex instructions, such as image editing, even in the absence of specific editing training data. We evaluate this zero-shot capability on Lumina-DiMOO and Bagel using image editing benchmarks. As shown in Table 4, PairUni consistently improves performance across all editing subtasks (Add, Replace, Remove, Style). For instance, on Lumina-DiMOO, PairUni achieves higher scores in every category compared to the random pairing baseline. This demonstrates that our method inherently improves the model’s capacity for instruction-following tasks by establishing better alignment between understanding and generation representations.

#### 3.2 Analysis: Effectiveness of PairUni Framework

To understand the source of our improvements, we conduct a series of ablation studies to isolate the impact of two core components: the data pairing strategy and the similarity-based advantage adjustment. We further analyze gradient similarity in Appendix D. We intervene on the semantic alignment between understanding and generation training signals by constructing six data-combination scenarios (PairUG, retrieval-based pairs, low-similarity unpaired data, understanding-only, generation-only, and random pairs), while keeping the training recipe fixed. As shown in Fig. 8(a), stronger semantic alignment consistently increases the median cosine similarity between the gradients from the understanding and generation objectives (e.g., 0.120 for PairUG vs. 0.059 for random/generation-only), indicating reduced gradient interference. Importantly, Fig. 8(b) shows that higher gradient agreement correlates with better joint downstream outcomes: PairUG achieves the best simultaneous performance on MMMU/MMStar and GenEval (40.4/46.1/79), whereas low-similarity or random pairing yields lower gradient agreement and inferior joint results (e.g., GenEval drops to 71 under low-similarity unpair). This controlled trend directly supports our motivation that aligned U–G pairs mitigate

- Table 3 Results on GenEval. PairUni achieves SOTA results among 1B models and competitive performance at 7B scale.

Method Single Obj. Two Obj. Counting Colors Position Color Attri. Overall

Generating Only PixArt-α [4] 0.98 0.50 0.44 0.80 0.08 0.07 0.48 SDXL [31] 0.98 0.74 0.39 0.85 0.15 0.23 0.55 DALL-E 3 [32] 0.96 0.87 0.47 0.83 0.43 0.45 0.67

Unified Understanding and Generation SEED-X [9] 0.97 0.58 0.26 0.80 0.19 0.14 0.49 Show-o [49] 0.95 0.52 0.49 0.82 0.11 0.28 0.53 ILLUME [44] 0.99 0.86 0.45 0.71 0.39 0.28 0.61 HermersFlow [52] 0.97 0.67 0.65 0.77 0.28 0.42 0.61 UniRL [26] 0.95 0.74 0.27 0.81 0.62 0.52 0.65 Janus-Pro-1B [6] 0.99 0.82 0.48 0.90 0.62 0.57 0.73 ULM-R1 [14] - - - - - - 0.76

- Janus-Pro-R1 [29] 0.98 0.80 0.51 0.84 0.59 0.55 0.71 PairUni-1B 0.98 0.91 0.44 0.75 0.95 0.69 0.79 Chameleon [42] – – – – – – 0.39 D-DiT [21] 0.97 0.80 0.54 0.76 0.32 0.50 0.65 LWM [22] 0.93 0.41 0.46 0.79 0.09 0.15 0.47 Transfusion [56] – – – – – – 0.63 TokenFlow-XL [35] 0.95 0.60 0.41 0.81 0.16 0.24 0.55 Janus-Pro-7B [6] 0.97 0.88 0.57 0.90 0.77 0.64 0.79 DSR [12] – – – – – – 0.84

- Janus-Pro-R1 [29] 0.99 0.94 0.66 0.92 0.87 0.78 0.86 PairUni-7B 0.97 0.75 0.78 0.97 0.91 0.69 0.85

gradient conflicts and improve unified generalization.

Disentangling Data Pairing from Data Quality. A central question is whether the observed performance improvements arise solely from the use of higher–quality data or from the pairing mechanism itself. Table 5 addresses this by comparing our method with random pairing strategies under an identical data source and budget.

Naive mixtures (unpaired or random pairing) under the same computational and data constraints systematically reduce generative fidelity (e.g., 0.71–0.73 on GenEval). This phenomenon is theoretically supported by our gradient analysis in Appendix D (Figure 8): gradients induced by semantically unrelated (randomly paired) samples exhibit low cosine similarity, effectively behaving as conflicting noise that destabilizes the shared policy. In contrast, our aligned and retrieved pairs maintain higher gradient agreement, fostering constructive interference. Furthermore, the construction of our dataset relies on selecting representative samples. As shown in Appendix C (Table 9), our K-means based medoid selection strategy significantly outperforms random selection, ensuring that the curated dataset PairUG-16k covers diverse and high-quality semantic regions, which is crucial for the consistent improvements observed. Figure 4 further demonstrates that PairUG-16k yields more stable training dynamics relative to random pairing.

[Figure 6]

Figure 4 Comparison of training reward curves between PairUG-16k and random pairing strategies.

Effect of Similarity-based Advantage Adjustment. Table 6 isolates the specific contribution of similarity-based weighting within PairGRPO. We perform a detailed investigation of weighting strategies in Appendix C (Table 7). Our results show that the square-root scaling (√sp) yields the strongest overall performance compared to linear scaling or no weighting. This non-linear scaling effectively balances the contribution of retrieved pairs: it down-weights weakly aligned pairs to prevent noise injection while preserving sufficient gradient magnitude for effective learning. Additionally, the choice of similarity threshold is critical for the quality-quantity trade-off. As detailed in Appendix C (Table 8), a threshold of 0.6 is optimal; it ensures sufficient data volume for training while excluding low-similarity pairs that could degrade performance. We also justify our choice of image encoder for similarity computation. In Appendix C.4 (Table 10), we compare

- Table 4 Extended Model Performance Analysis: (a) Improvements with PairUni on Lumina-DiMOO. (b) Performance gains on Bagel with PairUG-16k. Add, Replace, Remove, and Style are subtasks of ImgEdit [53].

(a) Lumina-DiMOO Performance Model MMMU MMStar GenEval Add Replace Remove Style

Lumina-DiMOO [41] 58.6 52.4 0.88 3.82 3.83 2.76 4.18 w/ Random Pair 56.1 52.1 0.86 3.75 3.66 2.75 3.98 Lumina-DiMOO w PairUni 61.3 52.6 0.89 3.84 3.94 2.78 4.22

(b) Bagel Performance Model MMMU POPE GenEval (S) GenEval (L) ImgEdit (sum)

Bagel (report) 55.3 - 82 88 3.38 Bagel (reproduced) 52.8 87.37 77.9 86.1 3.38 w/ Random Pair 51.3 87.62 81.7 86.3 3.39 w/ PairUG-16k (Ours) 54.0 89.21 84.6 87.2 3.51

- Table 5 Ablation study of Data Pairing. Comparison of different pairing strategies, highlighting the contribution of PairUG-16k.

###### Table 6 Ablation study on Trajectory-level credit assignment.

Model MMMU MMStar GenEval

Model MME(P) MMMU MMStar GenEval

Pairs from U only 38.2 43.7 0.75 Pairs from G only 36.4 41.9 0.74 Unpair 38.4 44.4 0.71 Random Pair 38.4 44.3 0.73 Aligned-based Pairs 39.2 44.6 0.76 Retrieval-based Pairs 40.1 44.9 0.77 PairUG (7.5K) 39.6 43.7 0.76 PairUG-16k 40.4 (+2.0) 46.1 (+1.8) 0.79 (+0.06)

PairUni-1B w/o sim 1469.87 40.0 45.1 0.79 PairUni-1B 1483.18 40.4 46.1 0.79 PairUni-7B w/o sim 1554.91 47.0 47.7 0.85 PairUni-7B 1597.71 47.0 49.5 0.85

different extractors and find that those emphasizing visual similarity (e.g., ResNet, DINOv3) outperform high-level semantic encoders (e.g., Perception Encoder), confirming that visual consistency is the primary driver for effective retrieval-based pairing. In summary, similarity-based weighting provides a principled mechanism to enhance comprehension-oriented performance while preserving generation quality by prioritizing supervision from strongly aligned pairs.

### 4 Conclusion

This paper introduces PairUni, a reinforcement learning framework for UVLMs that aligns understanding and generation via paired training signals, and PairUG-16k, a curated dataset of understanding–generation pairs that supports consistent policy learning. On standard UVLMs evaluations with Janus-Pro backbones, our approach achieves strong, balanced improvements in both understanding and generation, surpassing competitive RL baselines.

### References

- [1] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.

- [2] Daniel Bolya, Po-Yao Huang, Peize Sun, Jang Hyun Cho, Andrea Madotto, Chen Wei, Tengyu Ma, Jiale Zhi, Jathushan Rajasegaran, Hanoona Rasheed, Junke Wang, Marco Monteiro, Hu Xu, Shiyu Dong, Nikhila Ravi, Daniel Li, Piotr Dollár, and Christoph Feichtenhofer. Perception encoder: The best visual embeddings are not at the output of the network. arXiv:2504.13181, 2025.

- [3] Jiuhai Chen, Zhiyang Xu, Xichen Pan, Yushi Hu, Can Qin, Tom Goldstein, Lifu Huang, Tianyi Zhou, Saining

- Xie, Silvio Savarese, et al. Blip3-o: A family of fully open unified multimodal models-architecture, training and dataset. arXiv preprint arXiv:2505.09568, 2025.
- [4] Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Yue Wu, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, and Zhenguo Li. Pixart-α: Fast training of diffusion transformer for photorealistic text-to-image synthesis, 2023.
- [5] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, et al. Are we on the right way for evaluating large vision-language models? Advances in Neural Information Processing Systems, 37:27056–27087, 2024.

- [6] Xiaokang Chen, Zhiyu Wu, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, and Chong Ruan. Janus-pro: Unified multimodal understanding and generation with data and model scaling, 2025. URL https: //arxiv.org/abs/2501.17811.
- [7] DeepSeek-AI. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning, 2025. URL https://arxiv.org/abs/2501.12948.
- [8] Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, Guang Shi, and Haoqi Fan. Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683, 2025.

- [9] Yuying Ge, Sijie Zhao, Jinguo Zhu, Yixiao Ge, Kun Yi, Lin Song, Chen Li, Xiaohan Ding, and Ying Shan. Seed-x: Multimodal models with unified multi-granularity comprehension and generation. arXiv preprint arXiv:2404.14396, 2024.

- [10] Dhruba Ghosh, Hannaneh Hajishirzi, and Ludwig Schmidt. Geneval: An object-focused framework for evaluating text-to-image alignment. Advances in Neural Information Processing Systems, 36:52132–52152, 2023.

- [11] Suriya Gunasekar, Yi Zhang, Jyoti Aneja, Caio César Teodoro Mendes, Allie Del Giorno, Sivakanth Gopi, Mojan Javaheripi, Piero Kauffmann, Gustavo de Rosa, Olli Saarikivi, et al. Textbooks are all you need. arXiv preprint arXiv:2306.11644, 2023.

- [12] Jixiang Hong, Yiran Zhang, Guanzhong Wang, Yi Liu, Ji-Rong Wen, and Rui Yan. Reinforcing multimodal understanding and generation with dual self-rewards. arXiv preprint arXiv:2506.07963, 2025.

- [13] Dongzhi Jiang, Ziyu Guo, Renrui Zhang, Zhuofan Zong, Hao Li, Le Zhuo, Shilin Yan, Pheng-Ann Heng, and Hongsheng Li. T2i-r1: Reinforcing image generation with collaborative semantic-level and token-level cot. arXiv preprint arXiv:2505.00703, 2025.

- [14] Jingjing Jiang, Chongjie Si, Jun Luo, Hanwang Zhang, and Chao Ma. Co-reinforcement learning for unified multimodal understanding and generation, 2025. URL https://arxiv.org/abs/2505.17534.
- [15] Yang Jiao, Haibo Qiu, Zequn Jie, Shaoxiang Chen, Jingjing Chen, Lin Ma, and Yu-Gang Jiang. Unitoken: Harmonizing multimodal understanding and generation through unified visual encoding, 2025. URL https: //arxiv.org/abs/2504.04423.
- [16] Brett Koonce. Resnet 50. In Convolutional neural networks with swift for tensorflow: image recognition and dataset categorization, pages 63–72. Springer, 2021.

- [17] Siqi Kou, Jiachun Jin, Zhihong Liu, Chang Liu, Ye Ma, Jian Jia, Quan Chen, Peng Jiang, and Zhijie Deng. Orthus: Autoregressive interleaved image-text generation with modality-specific heads, 2025. URL https: //arxiv.org/abs/2412.00127.
- [18] Black Forest Labs, Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, Sumith Kulal, Kyle Lacey, Yam Levi, Cheng Li, Dominik Lorenz, Jonas Müller, Dustin Podell, Robin Rombach, Harry Saini, Axel Sauer, and Luke Smith. Flux.1 kontext: Flow matching for in-context image generation and editing in latent space, 2025. URL https: //arxiv.org/abs/2506.15742.
- [19] Daiqing Li, Aleks Kamko, Ehsan Akhgari, Ali Sabet, Linmiao Xu, and Suhail Doshi. Playground v2.5: Three insights towards enhancing aesthetic quality in text-to-image generation, 2024.
- [20] Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large vision-language models. arXiv preprint arXiv:2305.10355, 2023.

- [21] Zijie Li, Henry Li, Yichun Shi, Amir Barati Farimani, Yuval Kluger, Linjie Yang, and Peng Wang. Dual diffusion for unified image generation and understanding. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 2779–2790, 2025.

- [22] Hao Liu, Wilson Yan, Matei Zaharia, and Pieter Abbeel. World model on million-length video and language with blockwise ringattention, 2025. URL https://arxiv.org/abs/2402.08268.
- [23] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. arXiv preprint arXiv:2310.02255, 2023.

- [24] Shiyin Lu, Yang Li, Qing-Guo Chen, Zhao Xu, Weihua Luo, Kaifu Zhang, and Han-Jia Ye. Ovis: Structural embedding alignment for multimodal large language model. arXiv:2405.20797, 2024.

- [25] Yan Ma, Linge Du, Xuyang Shen, Shaoxiang Chen, Pengfei Li, Qibing Ren, Lizhuang Ma, Yuchao Dai, Pengfei Liu, and Junjie Yan. One rl to see them all: Visual triple unified reinforcement learning. arXiv preprint arXiv:2505.18129, 2025.

- [26] Weijia Mao, Zhenheng Yang, and Mike Zheng Shou. Unirl: Self-improving unified multimodal models via supervised and reinforcement learning, 2025. URL https://arxiv.org/abs/2505.23380.
- [27] Shen Nie, Fengqi Zhu, Zebin You, Xiaolu Zhang, Jingyang Ou, Jun Hu, Jun Zhou, Yankai Lin, Ji-Rong Wen, and Chongxuan Li. Large language diffusion models, 2025. URL https://arxiv.org/abs/2502.09992.
- [28] Yuwei Niu, Munan Ning, Mengren Zheng, Weiyang Jin, Bin Lin, Peng Jin, Jiaqi Liao, Chaoran Feng, Kunpeng Ning, Bin Zhu, et al. Wise: A world knowledge-informed semantic evaluation for text-to-image generation. arXiv preprint arXiv:2503.07265, 2025.

- [29] Kaihang Pan, Yang Wu, Wendong Bu, Kai Shen, Juncheng Li, Yingting Wang, Yunfei Li, Siliang Tang, Jun Xiao, Fei Wu, Hang Zhao, and Yueting Zhuang. Unlocking aha moments via reinforcement learning: Advancing collaborative visual comprehension and generation, 2025. URL https://arxiv.org/abs/2506.01480.
- [30] Yingzhe Peng, Gongrui Zhang, Miaosen Zhang, Zhiyuan You, Jie Liu, Qipeng Zhu, Kai Yang, Xingzhong Xu, Xin Geng, and Xu Yang. Lmm-r1: Empowering 3b lmms with strong reasoning abilities through two-stage rule-based rl, 2025. URL https://arxiv.org/abs/2503.07536.
- [31] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis, 2023. URL https://arxiv.org/abs/2307.01952.
- [32] MM Pooja, PM Sulfath, and KM Sheena. Dall-e 3: Advanced ai image generation model. Authorea Preprints.

- [33] Kunpeng Qiu, Zhiqiang Gao, Zhiying Zhou, Mingjie Sun, and Yongxin Guo. Noise-consistent siamese-diffusion for medical image synthesis and segmentation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 15672–15681, 2025.

- [34] Kunpeng Qiu, Zhiying Zhou, and Yongxin Guo. Adaptively distilled controlnet: Accelerated training and superior sampling for medical image synthesis. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 55–65. Springer, 2025.

- [35] Liao Qu, Huichao Zhang, Yiheng Liu, Xu Wang, Yi Jiang, Yiming Gao, Hu Ye, Daniel K Du, Zehuan Yuan, and Xinglong Wu. Tokenflow: Unified image tokenizer for multimodal understanding and generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 2545–2555, 2025.

- [36] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021.

- [37] Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model, 2024. URL https://arxiv.org/abs/ 2305.18290.
- [38] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10684–10695, June 2022.

- [39] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models, 2024. URL https://arxiv.org/abs/2402.03300.
- [40] Oriane Siméoni, Huy V. Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Michaël Ramamonjisoa, Francisco Massa, Daniel Haziza, Luca Wehrstedt, Jianyuan Wang, Timothée Darcet, Théo Moutakanni, Leonel Sentana, Claire Roberts, Andrea Vedaldi, Jamie Tolan, John Brandt, Camille Couprie, Julien Mairal, Hervé Jégou, Patrick Labatut, and Piotr Bojanowski. DINOv3, 2025. URL https://arxiv.org/abs/2508.10104.
- [41] Alpha VLLM Team. Lumina-dimoo: A unified masked diffusion model for multi-modal generation and understanding, 2025. URL https://github.com/Alpha-VLLM/Lumina-DiMOO.
- [42] Chameleon Team. Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818, 2024.

- [43] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. Llama 2: Open foundation and fine-tuned chat models, 2023. URL https://arxiv.org/abs/2307.09288.
- [44] Chunwei Wang, Guansong Lu, Junwei Yang, Runhui Huang, Jianhua Han, Lu Hou, Wei Zhang, and Hang Xu. Illume: Illuminating your llms to see, draw, and self-enhance. arXiv preprint arXiv:2412.06673, 2024.

- [45] Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Zhen Li, Qiying Yu, Yingli Zhao, Yulong Ao, Xuebin Min, Tao Li, Boya Wu, Bo Zhao, Bowen Zhang, Liangdong Wang, Guang Liu, Zheqi He, Xi Yang, Jingjing Liu, Yonghua Lin, Tiejun Huang, and Zhongyuan Wang. Emu3: Next-token prediction is all you need, 2024. URL https://arxiv.org/abs/2409.18869.
- [46] Yinjie Wang, Ling Yang, Bowen Li, Ye Tian, Ke Shen, and Mengdi Wang. Revolutionizing reinforcement learning framework for diffusion large language models. arXiv preprint arXiv:2509.06949, 2025.

- [47] Xiaoshi Wu, Yiming Hao, Keqiang Sun, Yixiong Chen, Feng Zhu, Rui Zhao, and Hongsheng Li. Human preference score v2: A solid benchmark for evaluating human preferences of text-to-image synthesis. arXiv preprint arXiv:2306.09341, 2023.

- [48] Yecheng Wu, Zhuoyang Zhang, Junyu Chen, Haotian Tang, Dacheng Li, Yunhao Fang, Ligeng Zhu, Enze Xie, Hongxu Yin, Li Yi, Song Han, and Yao Lu. Vila-u: a unified foundation model integrating visual understanding and generation, 2025. URL https://arxiv.org/abs/2409.04429.
- [49] Jinheng Xie, Weijia Mao, Zechen Bai, David Junhao Zhang, Weihao Wang, Kevin Qinghong Lin, Yuchao Gu, Zhijie Chen, Zhenheng Yang, and Mike Zheng Shou. Show-o: One single transformer to unify multimodal understanding and generation. arXiv preprint arXiv:2408.12528, 2024.

- [50] Jinheng Xie, Zhenheng Yang, and Mike Zheng Shou. Show-o2: Improved native unified multimodal models. arXiv preprint arXiv:2506.15564, 2025.

- [51] Ling Yang, Ye Tian, Bowen Li, Xinchen Zhang, Ke Shen, Yunhai Tong, and Mengdi Wang. Mmada: Multimodal large diffusion language models. arXiv preprint arXiv:2505.15809, 2025.

- [52] Ling Yang, Xinchen Zhang, Ye Tian, Chenming Shang, Minghao Xu, Wentao Zhang, and Bin Cui. Hermesflow: Seamlessly closing the gap in multimodal understanding and generation, 2025. URL https://arxiv.org/abs/ 2502.12148.
- [53] Yang Ye, Xianyi He, Zongjian Li, Bin Lin, Shenghai Yuan, Zhiyuan Yan, Bohan Hou, and Li Yuan. Imgedit: A unified image editing dataset and benchmark, 2025. URL https://arxiv.org/abs/2505.20275.

- [54] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9556–9567, 2024.

- [55] Xinjie Zhang, Jintao Guo, Shanshan Zhao, Minghao Fu, Lunhao Duan, Jiakui Hu, Yong Xien Chng, Guo-Hua Wang, Qing-Guo Chen, Zhao Xu, Weihua Luo, and Kaifu Zhang. Unified multimodal understanding and generation models: Advances, challenges, and opportunities, 2025. URL https://arxiv.org/abs/2505.02567.
- [56] Chunting Zhou, LILI YU, Arun Babu, Kushal Tirumala, Michihiro Yasunaga, Leonid Shamis, Jacob Kahn, Xuezhe Ma, Luke Zettlemoyer, and Omer Levy. Transfusion: Predict the next token and diffuse images with one multi-modal model. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=SI2hI0frk6.

- [57] Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, Zhangwei Gao, Erfei Cui, Xuehui Wang, Yue Cao, Yangzhou Liu, Xingguang Wei, Hongjie Zhang, Haomin Wang, Weiye Xu, Hao Li, Jiahao Wang, Nianchen Deng, Songze Li, Yinan He, Tan Jiang, Jiapeng Luo, Yi Wang, Conghui He, Botian Shi, Xingcheng Zhang, Wenqi Shao, Junjun He, Yingtong Xiong, Wenwen Qu, Peng Sun, Penglong Jiao, Han Lv, Lijun Wu, Kaipeng Zhang, Huipeng Deng, Jiaye Ge, Kai Chen, Limin Wang, Min Dou, Lewei Lu, Xizhou Zhu, Tong Lu, Dahua Lin, Yu Qiao, Jifeng Dai, and Wenhai Wang. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models, 2025. URL https://arxiv.org/abs/2504.10479.

## Appendix

[Figure 7]

- Figure 5 Distributional comparison between multimodal understanding and image generation datasets. The visualization highlights the domain gap and the necessity of the proposed alignment strategy.

Overview. Appendix A covers related work on UVLMs and the RL methods used in UVLMs. Appendix B provides data-centric details, including summary statistics of the original data distribution, representative cases of retrieved pairs, and the empirical distribution of PairUG-16k. Appendix C presents systematic ablation studies, including the choice of image extractor, weighting strategies, and threshold selection. Appendix E demonstrates the generalization capabilities of our method on additional models. Appendix F presents qualitative case studies on understanding tasks. Appendix G includes the prompts used by GPT-o3. The closing sections include notes on the use of large language models.

### A Related Work

Unified Vision-Language Model. The pursuit of unified multimodal frameworks has led to significant innovations in both architecture design and training paradigms. Early approaches [42, 49] like Show-o series [49, 50] establish the autoregressive foundations for joint vision-language processing. Meanwhile, Transfusion [56] introduces diffusion-based methodologies to enhance generation quality. These foundational works, as systematically analyzed in [55], demonstrate the potential of unifying modalities through shared representation learning. Recent advances have pushed the boundaries of unified modeling [6, 8]. For example, Janus-Pro [6] innovatively uses bidirectional encoder-decoder structures for understanding and generation, achieving stronger performance on both sides. Bagel [8] adopts transformer experts and is trained with massive image generation and understanding data, leading to the state-of-the-art performance. This architectural evolution aligns with the broader trend of developing modular yet integrated systems that can dynamically adapt to different modalities. In this context, our work enhances UVLMs post-training, in particular, during the reinforcement learning phase. We present a novel view of pair data generation and utilization of such pair data with proposed PairGRPO.

Reinforcement Learning in UVLMs. The integration of reinforcement learning (RL) has emerged as a critical component for advancing unified MLLMs during the post-training. Early RL applications focused on modalityspecific enhancements: step-by-step rule-based rewards for mathematical reasoning [39], and bbox IoU rewards for visual grounding [13]. For text-to-image generation, CLIP-based rewards [36] became standard for aligning visual outputs with textual descriptions. The paradigm shifted with unified RL approaches that exploit crossmodal synergies. T2I-R1 [13] pioneered iterative refinement through GRPO [39], using detailed descriptions as intermediate rewards. Recently, several works also explore the RL-based post-training for UVLMs. In

[Figure 8]

- Figure 6 Representative paired cases demonstrating the synergy between multimodal understanding and image generation tasks. The examples highlight how aligned data pairs can enhance model performance across both domains.

particular, UniRL [26] proposed a self-improving pipeline where generated QA pairs simultaneously serve as training data and reward signals, though this approach showed performance degradation in understanding benchmarks. More sophisticated reward mechanisms have since been developed. DSR [12] introduced dual-source rewards combining original image-caption pairs with generated content, while HermesFlow [52] implemented Pair DPO [37] to enforce consistency between modalities. Notably, CoRL [14] adopts a two-stage approach, first training unified RL on shared data before specializing in understanding/generation phases, demonstrating improved performance across multiple benchmarks. Different from these methods, which all focus on unified RL method design, our work provides a new view on understanding data and generation data. We propose to build the UG pairs to benefit both tasks. With proposed PairGRPO along with UG pairs, our work improves various UVLMs.

### B More details about data

#### B.1 Data Distribution of Understanding and Generation Data

We curate two complementary splits covering multimodal understanding and image generation. For understanding, we adopt Orsta-47k [25], a high-quality and diverse set that spans chart analysis, counting, object detection, grounding, mathematical reasoning, OCR, puzzles, and scientific reasoning. For image generation, BLIP-3o-60k [3] comprises curated AI-generated images paired with detailed textual descriptions, including single- and dual-object scenes as well as text-containing visuals. We deduplicate and ensure there is no overlap with the data used during pretraining.

To characterize their composition, we apply unsupervised clustering over the union of the two splits and examine cluster proportions (Figure 5). The two distributions exhibit pronounced divergence: categories

[Figure 9]

- Figure 7 Detailed distribution analysis of the PairUG-16k dataset. From left to right: (1) Source breakdown showing the composition of Aligned Pairs; (2) Class distribution within Aligned Pairs; (3) Class distribution for Retrieved Pairs; and (4) Similarity score distribution for Retrieved Pairs, validating the diversity and quality of the constructed dataset.

- Table 7 Ablation study on the Weighting Strategy. The table compares the base model (no weighting) with linear and square root similarity weighting schemes, demonstrating that the proposed square root weighting yields the best consistency and performance.

Weighting Strategy MME (p) MMMU MMStar GenEval

wo weighting (Base) 1469.87 40.0 45.1 0.79 linear similarity 1478.24 40.3 46.0 0.79

sqrt(similarity) (Ours) 1483.18 40.4 46.1 0.79

prevalent in understanding data—such as math- or OCR-intensive items—are rare in the generation split, whereas descriptive object-centric scenes are comparatively overrepresented in generation.

#### B.2 Retrieved Pairs Cases

- Figure 6 presents representative retrieved examples.

B.3 Distribution of PairUG-16k

- Figure 7 summarizes the composition of PairUG-16k. We construct two complementary splits: Aligned Pairs and Retrieved Pairs. For the Aligned Pairs, the data originate from two sources: generation (3,043 examples) and understanding (1,928 examples). The class distribution is long-tailed: the head classes—such as Math (646), Human Gestures (602), and Puzzle (402)—account for a substantial portion of the data, while several categories (e.g., object count with 65 instances) appear infrequently. This split provides high-quality supervision across 17 labeled categories. The Retrieved Pairs display a different profile. The largest classes are JourneyDB (2,476) and GenEval (1,787), followed by Human Gestures (1,387), Object1 (1,232), Text2 (1,193), Text1 (992), Mscoco Human (857), and Occupation2 (708), with smaller categories such as Object2

(252) and Occupation1 (238). The similarity histogram is unimodal with most pairs in the 0.55–0.75 range, indicating that retrieval yields semantically related pairs while retaining diversity.

### C Additional Ablation Studies

In this section, we provide detailed ablations on key design choices of PairUni, including the weighting strategy for retrieved pairs, the similarity threshold, the K-means medoid selection strategy, and the choice of image feature extractor. All experiments are conducted on the Janus-Pro-1B model.

#### C.1 Weighting Strategy for Retrieved Pairs

We compare different weighting schemes for the retrieved pairs in Eq. 2. As shown in Table 7, using the square root of the similarity score (‘sqrt(similarity)‘) provides the most robust performance boost. This is likely because it balances the contribution of high-similarity retrieval without dominating the loss.

- Table 8 Ablation study investigating the impact of the similarity threshold on model performance. The table reports results for thresholds of 0.5, 0.6 (Ours), and 0.7 across MMMU, MMStar, and GenEval benchmarks.

Similarity Threshold MMMU MMStar GenEval

- Sim ≥ 0.5 38.6 45.6 0.76

- Sim ≥ 0.6 (Ours) 40.4 46.1 0.79

- Sim ≥ 0.7 39.1 44.9 0.72

- Table 9 Ablation study on the K-means Medoid Selection strategy. The results compare the performance of the proposed K-means based selection against a random selection baseline across multiple benchmarks.

Selection Strategy MMMU MMStar GenEval

No K-means (Random) 38.4 44.0 0.73

K-means (Ours) 40.4 46.1 0.79

#### C.2 Similarity Threshold Selection

We investigate the impact of the similarity threshold used for filtering retrieved pairs. Table 8 shows that a threshold of 0.6 is optimal. Lower thresholds (e.g., 0.5) introduce noise, which can degrade performance, while higher thresholds (e.g., 0.7) reduce the diversity of the training data and limit the benefits of retrieval augmentation.

#### C.3 K-means Medoid Selection

We evaluate the effectiveness of using K-means medoids for data selection compared to random sampling. As demonstrated in Table 9, using K-means medoids significantly outperforms random sampling. We hypothesize that outliers are not necessarily useless but often represent data points that cannot simultaneously adapt to both understanding and generation tasks (e.g., tabular data may be suitable for understanding but show a significant gap in generation performance). By using clustering and selecting medoids, we prioritize data points that are compatible with both tasks, yielding better performance than random sampling.

#### C.4 Results with different image extractors.

We evaluate three alternatives for the image feature extractor: the Perception Encoder (PE) [2], DINOv3 [40], and ResNet [16]. The PE is designed for high-level semantic understanding, yet it underperforms on both the understanding and generation benchmarks (Table 10). In contrast, DINOv3 and ResNet—both emphasizing visual feature similarity—achieve comparable results. These findings indicate that, in our setting, enforcing consistency with respect to visual similarity is more critical than modeling semantic abstraction.

### D Gradient Similarity Analysis

- Figure 8 illustrates the gradient conflict mechanism. We observe that when the data fed to understanding and generation are semantically aligned, the cosine similarity between their gradients increases. Higher gradient agreement correlates with stronger downstream results.

### E Generalization Capabilities

As shown in Table 4, to assess the generalizability of our data and training method, we instantiate the PairGRPO framework on Lumina-DiMOO [41]. Unlike Janus-Pro [6], Lumina-DiMOO is a multimodal discrete diffusion model [51]. To enable parallel RL training for both understanding and generation, we set the rollout size to 2 and adopt a fixed-step diffusion sampling scheme: we use 2 diffusion steps for text generation and 35 steps for image generation. We build upon the diffusion Large Language Model (dLLM) [27] and its Proximal Policy Optimization (PPO) [46] implementation, extending the framework to a multimodal dLLM and adding an implementation of the GPRO algorithm for policy optimization. Because Lumina-DiMOO has not released its paper nor the official training and evaluation code, this integration is preliminary and

- Table 10 Ablation study on Image Feature Extractors. Comparison of model performance using different image feature extractors (PE, DINOv3, and ResNet). The results justify the choice of ResNet for the final implementation.

Model MMMU MMStar GenEval

PE [2] 40.1 45.5 0.77 DINOv3 [40] 40.4 46.0 0.79 ResNet [16] 40.4 46.1 0.79

[Figure 10]

- Figure 8 Performance Conflict Mechanism Analysis: Median gradient cosine similarity scores between understanding and generation components, alongside benchmark performance on two understanding benchmarks (MMMU [54], MMStar [5]) and one image generation benchmark (GenEval [10]). The analysis encompasses six distinct data combination scenarios: PairUG-16k, Retrieval-based Pairs, Unpair data with low similarity scores, pure Generationonly data, pure Understanding-only data, and Random Pairs.

may benefit from further refinement. In the current implementation, PairUni improves both the model’s comprehension and generation capabilities.

BAGEL [8] is a strong unified vision–language model (UVLM) that integrates both understanding and generation components. It adopts a diffusion head [33, 34] with a flow-matching strategy for visual generation, together with an auto-regressive text-generation architecture. Since BAGEL does not support reinforcement learning (RL) training under its default settings, we perform supervised fine-tuning (SFT) using the PairUG16k dataset. To satisfy the data format requirements of PairUG, the training data are constructed with a 1:1 ratio between understanding and generation tasks. For the generation task, the image resolution is set to 512, while all other hyperparameters follow the default configuration of BAGEL. The resulting SFT model achieves consistent performance improvements on the MMMU, POPE, GenEval (s) and GenEval (L), and ImgEdit benchmarks.

### F Case Studies on Understanding Tasks

As shown in Figure 9, we present representative cases comparing Janus-Pro and PairUni on understanding tasks.

### G Prompts for GPT-o3

We design our instruction templates to enforce the semantic completion logic described in Section 2.1. For the Understanding → Generation pathway, the prompt (Figure 10) instructs GPT-o3 to generate a caption C that is not only descriptive but also strictly consistent with the provided Question-Answer pair (Q,A). It explicitly asks the model to verify that the visual evidence required for Q is present in C. For the Generation → Understanding pathway, the prompt (Figure 11) directs the model to formulate questions Q that target the key entities and attributes described in the caption C, ensuring the question is answerable given the generation intent. Furthermore, both templates include a self-verification step where the model is asked to check for hallucinations or contradictions before finalizing the output. This rigorous prompting strategy ensures high-fidelity alignment within the constructed quadruples.

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

Janus-Pro-7B: <think> The image prominently features a tall, modern building with a unique design, which stands out against the backdrop of other buildings and a bridge. The city skyline is the most dominant feature, as it occupies a significant portion of the image and is the primary focus of the scene. The lake and the bridge are secondary elements that complement the cityscape, but the buildings are the most visually striking and central to the image. </think> <answer>C</answer>

Ours: <think> The image prominently features a tall, uniquely designed building in the center, which stands out against the backdrop of other buildings and a bridge. The lake in the foreground reflects the structures, adding to the visual appeal. The bridge and surrounding buildings are secondary elements, but they contribute to the overall urban landscape. </think><answer>D</answer>

question: What is the dominant feature in the image?

- A.The bridge
- B.The lake
- C.The city skyline
- D.The buildings ground turth: D

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

Janus-Pro-7B: <think> The image shows a mural with a naturalistic theme, featuring elements such as a river, plants, and a bird. The presence of a river and natural surroundings suggests an outdoor environment. The mural's style and content do not indicate a historical site, religious location, or residential area. Instead, it appears to be a garden or an outdoor space designed for aesthetic purposes. </think> <answer>D</answer>

Ours: <think> The image shows a mural with figures that appear to be indigenous people, possibly depicting a historical or cultural scene. The presence of totem poles and traditional attire suggests a connection to a specific cultural heritage. The setting includes natural elements like rocks and water, which are often found in outdoor environments. Based on these observations, the depicted environment is likely a historical site or a cultural heritage area. </think><answer>A</answer>

question: What kind of environment is depicted in the image?

- A.A historical site
- B.A religious location
- C.A residential area
- D.A garden ground turth: A

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

Janus-Pro-7B: <think> The upper right image shows the highest contrast. The image features a fly on a white surface, which has a stark contrast between the black and white colors of the fly and the background. The other images have more subtle contrasts, with the lower left image showing people in a dimly lit environment and the lower right image showing blue chairs in a similar setting. </think><answer>B</answer>

question: Which image shows the highest contrast?

Ours: <think> The image in the upper left (A) shows the highest contrast. This is evident from the stark difference between the light and dark areas, particularly the deer standing in the sunlight against the darker background. The other images (B, C, and D) have more balanced lighting and less pronounced contrast. </think><answer>A</answer>

- A.upper left
- B.upper right
- C.down left
- D.down right ground turth: A

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

Janus-Pro-7B: <think> The image predominantly features warm tones such as red and orange, which are evident in the lighting and the subject's clothing. The background has a mix of blue and green hues, adding to the overall color scheme. The lighting creates a vibrant and energetic atmosphere, which is characteristic of live performances. </think> <answer>C</answer>

question: What is the main color scheme of this image?

Ours: <think> The image predominantly features warm tones such as red and orange, which are evident in the lighting and the subject's clothing. The background has a mix of blue and black hues, adding contrast to the warm colors. The overall color scheme is a combination of red, orange, and blue tones. </think><answer>D</answer>

- A.Black and silver
- B.White and blue
- C.Brown and red
- D.Red and black ground turth: D

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

Janus-Pro-7B: <think> The image shows a person with multiple overlapping faces, each displaying different expressions. The overlapping and blending of faces suggest a sense of internal conflict or emotional turmoil. The expressions range from anger to frustration, which are typically associated with negative emotions. The background and lighting also contribute to a somber and intense atmosphere. Based on this analysis, the image conveys an anxious mood. </think><answer>B</answer>

question: Which mood does this image convey?

Ours: <think> The image shows a person with multiple overlapping faces, each displaying different expressions. The most prominent expression is one of anger, as indicated by the open mouth and intense facial features. The background and overall composition also contribute to a sense of tension and emotional intensity. </think><answer>D</answer>

- A.Sad
- B.Anxious
- C.Happy
- D.Angry ground turth: D

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

Ours: <think> The image depicts a bearded man sitting on a throne-like chair, surrounded by columns and a classical architectural setting. The text "Episode 2 Oedipus" suggests a narrative or story theme. Given the context of the image, the main theme is likely related to a story or character, specifically Oedipus, who is known for his role in Greek mythology. The other options (A, B, C, D) do not align with the visual and textual elements of the image. </think><answer>A</answer>

Janus-Pro-7B: <think> The image depicts a bearded man sitting on a throne-like chair, surrounded by columns and a fire pit, which suggests a setting related to royalty or a historical context. The title "Oedipus" indicates a connection to a well-known mythological character, further supporting the theme of royal mythology or historical storytelling. </think> <answer>C</answer>

question: What is the main theme of the image?

- A.Bearded men
- B.Bathroom hygiene
- C.Royal monarchy
- D.Home decor ground turth: A

###### Figure 9 Representative qualitative examples comparing the multimodal understanding capabilities of Janus-Pro and PairUni. The cases highlight PairUni’s improved accuracy in interpreting complex visual scenes.

[Figure 35]

###### Figure 10 Detailed prompts employed for generating synthetic training data. These prompts are designed to ensure diversity and relevance in the generated image-text pairs.

[Figure 36]

###### Figure 11 The prompt template used with GPT-o3 for constructing understanding-focused quadruple data. This prompt guides the model to generate relevant questions and answers based on the input image.

