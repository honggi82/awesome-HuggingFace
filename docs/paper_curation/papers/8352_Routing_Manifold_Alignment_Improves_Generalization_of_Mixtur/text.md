# arXiv:2511.07419v2[cs.LG]12Nov2025

## ROUTING MANIFOLD ALIGNMENT IMPROVES GENERALIZATION OF MIXTURE-OF-EXPERTS LLMS

Zhongyang Li∗, Ziyue Li†, Tianyi Zhou ∗Johns Hopkins University †University of Maryland, College Park zli300@jh.edu, litzy619@umd.edu Project: https://github.com/tianyi-lab/RoMA

ABSTRACT

Sparse Mixture-of-Experts (MoE) have been widely adopted in recent large language models since it can efficiently scale up the model capability without increasing the inference cost. However, evaluations on broad downstream tasks reveal a consistent suboptimality of the routers in existing MoE LLMs, which results in a severe performance gap (e.g., 10-20% in accuracy) to the optimal routing. In this paper, we show that aligning the manifold of routing weights with that of task embedding can effectively reduce the gap and improve MoE LLMs’ generalization performance. Our method, “Routing Manifold Alignment (RoMA)”, introduces an additional manifold regularization term in the post-training objective and only requires lightweight finetuning of routers (with other parameters frozen). Specifically, the regularization encourages the routing weights of each sample to be close to those of its successful neighbors (whose routing weights lead to correct answers) in a task embedding space. Consequently, samples targeting similar tasks will share similar expert choices across layers. Building such bindings between tasks and experts over different samples is essential to achieve better generalization. Moreover, RoMA demonstrates the advantage of unifying the task understanding (by embedding models) with solution generation (by MoE LLMs). In experiments, we finetune routers in OLMoE, DeepSeekMoE, and Qwen3-MoE using RoMA. Evaluations on diverse benchmarks and extensive comparisons with baselines show the substantial improvement brought by RoMA.

1 INTRODUCTION

Sparse Mixture-of-Experts (MoE) have emerged as a cornerstone architecture in scaling large language models (LLMs), enabling significant capacity increases without proportional computational overhead during inference (Fedus et al., 2022; Lepikhin et al., 2020). At the core of this mechanism lies the router, which assigns input tokens to a small subset of experts through routing weights in each layer. Despite the small portion of router parameters in MoE LLMs (e.g., 0.03% in a 7B model), they are the key to the success of expert usage in MoE (Shazeer et al., 2017). However, evaluations across broad downstream tasks reveal that routers in existing MoE LLMs cause major failures. As shown in Table 1, their suboptimal routing weights lead to a performance gap of 10-20% in accuracy when compared to the optimal routing weights (oracle). This gap underscores a major untapped bottleneck in MoE LLMs, suggesting that improving routing is critical to boosting MoE LLMs’ generalization performance on downstream tasks.

Our analysis further investigates the reasons behind the performance gap and the poor generalization capabilities of pretrained routers. As illustrated in Figures 3(a) and (b), pretrained routers assign semantically similar samples in the task embedding space to distinct experts with dramatically different routing weights. Such misalignment between the task embedding manifold and routing weight manifold hinders effective knowledge sharing across tasks and underutilizes the collective expertise of the experts. This misalignment between the targeted tasks and the assigned experts undermines the generalization of MoE and its core principle, which is to leverage specialized experts, share skills, and transfer knowledge for related inputs.

A natural solution is to finetune the routers. Existing approaches, such as Dense BP (Panda et al., 2025) developed more effective pretraining objectives for routers but do not address the manifold misalignment between the targeted tasks and the routing weights across samples.

This limitation motivates our exploration of incorporating manifold alignment into the fine-tuning objective. Specifically, our manifold alignment aims to enforce the consistency between task understanding (encoded by an embedding model) and task solving in an MoE LLM (encoded by the routing weights). As illustrated in Figure 2, for each training sample, in addition to minimizing its loss defined on the output, we encourage its intermediate layers’ routing weights to move to those of its “successful neighbors” (samples with correct MoE predictions) in the task embedding space. These neighbors are weighted by their similarity to the sample. This training objective can be formulated as manifold regularization (Belkin et al., 2006), a well-established technique in machine learning that aims to preserve the local neighborhood structure of high-dimensional inputs on the manifold of low-dimensional representations or outputs. Unlike its original setting, we apply such a regularization to the routing weights across MoE layers rather than the final outputs, and establish coherent bindings between the expert choices (weights) and the task embedding instead of the raw inputs.

Gemma2-3B Mistral-7B Vicuna-13B Llama2-34B OLMoE-7B-A1B OLMoE-7B-A1B + RoMA RoMA Improvement

ARC-C

67.2

ARC-E

HellaSwag

88.0

86.7

73.8 51.3

77.9

85.8

PIQA

80.7 57.8

MMLU

69.0

72.2

45.5 75.4

49.7

81.8

GSM-8K

WinoGrande

81.7

BoolQ

Figure 1: RoMA on OLMoE-7B-A1B vs. 734B dense LLMs across eight benchmarks. RoMA leads to 7-15% accuracy improvement, consistently outperforming all models over eight benchmarks, demonstrating the effectiveness of post-training by RoMA.

To this end, we propose “Routing Manifold Alignment (RoMA)”, a router post-training method that aligns the manifold of routing weights with task embeddings through lightweight fine-tuning of a few routers in MoE LLMs. RoMA introduces a manifold regularization term to the training objective that encourages routing weights of each sample to approximate those of its successful neighbors with similar task embedding, thereby promoting consistent expert selection for semantically related inputs. Extensive experiments on three recent MoE LLMs (OLMoE, DeepSeekMoE, Qwen3-MoE) demonstrate that RoMA brings substantial improvements (7-15% in accuracy) across diverse benchmarks and outperforms SOTA routing methods, as shown in Figure 1, by merely finetuning 0.0095% parameters of the base model, without affecting inference cost. Notably, RoMA-finetuned MoE LLMs with only 1-3B active parameters achieve competitive or superior performance over much larger dense models with 34B parameters. We

Training Sample Successful Neighbor

Successful Neighborhood

Manifold Alignment

Training Objective:

Task Embedding Manifold

Routing Weights Manifold

Expert1 Expert2 Expert3

❄ Frozen

❄ ❄ ❄

❄ Embedding Model

🔥 Trainable

Router

🔥

- Figure 2: Overview of RoMA. RoMA finetunes routers in MoE LLM (bottom, yellow) with a training objective defined on each sample (xi,yi), which is composed of

- (1) the task loss Ltask(i) defined on the model output f(xi,ri); and (2) the manifold alignment regularization Lmanifold(i), which aligns the manifolds of routing weights (right, green) and the task embedding (left, blue). It improves MoE’s generalization by unifying solution generation in MoE with task understanding.

conduct comprehensive ablation studies that further investigate the effects of key designs in RoMA, including layer selection, neighborhood configuration, and regularization strategies, validating the effectiveness of RoMA in bridging the performance gap between pretrained routers and optimal routing for MoE LLMs.

### 2 RELATED WORK

MoE LLMs Mixture of Experts (MoE) architectures have been extensively incorporated into large language models (LLMs) to enhance computational efficiency and task-specific specialization (Shazeer et al., 2017). Recent work such as OLMoE (Muennighoff et al., 2024) and DeepSeekMoE (Dai et al., 2024a) demonstrate the effectiveness of sparse MoE layers in reducing active parameters while maintaining model capacity. These MoE models fundamentally rely on routers to determine expert selection, typically employing token-choice routing that selectively activates subsets of experts for each input token (Fedus et al., 2022; Lepikhin et al., 2020). However, the quality of these routing decisions remains a critical bottleneck. Our study shows current routers often produce suboptimal routing weights that fail to fully leverage expert specialization, resulting in load imbalance and expert underutilization.

Manifold Regularization of LLMs Recent work reveals that LLM embeddings exhibit stratified manifold structures with varying dimensions across semantic domains (Li & Sarwate, 2025; Robinson et al., 2025). While traditional manifold regularization assumes smooth global structures (Belkin et al., 2006), LLMs require more sophisticated approaches. Methods like I-STAR (Rudman & Eickhoff, 2023) control isotropy in embedding spaces, while CROW (Min et al., 2024) enforces consistency across layers. However, these techniques do not explicitly leverage manifold structures to improve MoE routing. The geometric insights from stratified manifolds suggest that different experts naturally align with different embedding strata, yet current routing mechanisms fail to exploit this alignment. This gap motivates our routing manifold alignment approach, which guides routing decisions based on the data’s inherent geometric structure.

Routing Optimization in MoE architectures has emerged as a critical component for achieving efficient expert utilization and balanced computation. Routing optimization methods have evolved from simple load balancing (Fedus et al., 2022; Lepikhin et al., 2020) to sophisticated strategies including differentiable top-k selection (Zhou et al., 2022) and test-time optimization such as C3PO (Li et al., 2025a;b) that dynamically re-weights expert pathways. However, these approaches optimize routing without considering the embedding space’s geometric structure. Moreover, C3PO introduce additional computational overhead for task embedding and nearest neighbor search, requiring 6-7x the cost of standard inference by the base model.

### 3 TASK-EXPERT ROUTING MANIFOLD MISALIGNMENT

Task Cluster 1 Task Cluster 2 Task Cluster 3 Task Cluster 4

(a) Task Embedding (b) Routing Weights (Before RoMA) (c) Routing Weights (After RoMA) (d) Optimal Routing Weights (Oracle)

- Figure 3: UMAP visualization of task embedding and routing weights manifolds for samples in ARC-C. (a) Their task embedding shows cluster structures. (b) Routing weights by pretrained MoE are scattered and misaligned with the task embedding clusters. (c) RoMA aligns routing weights with the task embedding manifold’s cluster structure. (d) RoMA also achieves a similar manifold structure as that of the optimal routing weights (oracle), which explains the improvement in generalization.

MoE LLMs employ routers to assign input tokens to a small subset of experts through routing weights in each layer. However, evaluations across broad downstream tasks reveal that routers in existing MoE LLMs cause major failures. As shown in Table 1, their suboptimal routing weights lead to a performance gap of 10-20% in accuracy when compared to the optimal routing weights (oracle) ri∗, which is defined below for each sample (xi,yi) as

ri∗ ≜ arg min

LCE(f(xi,r),yi), (1)

r

where f(·,·) represents the MoE model that takes input xi and routing weights r to produce output, yi is the ground truth label for input xi, and LCE is the cross-entropy loss.

To investigate the root causes behind the observed performance gap in MoE LLMs, we conduct a comprehensive analysis of the relationship between task embeddings and routing weights in Figure 3. The comparison between task embeddings (Figure 3(a)) and pretrained routing weights (Figure 3(b)) reveals a severe misalignment. While the task embedding space presents clear cluster structures where semantically similar samples are grouped, the pretrained routing weights show no corresponding clustering patterns. Instead, samples from the same semantic cluster are scattered across the routing weights space. This manifold misalignment indicates that the pretrained routers fail to capture the underlying task structure, leading to inconsistent expert selection for semantically related inputs.

In contrast, the oracle routing weights (Figure 3(d)) demonstrate a clear cluster structure to the task embedding structure, with samples from the same semantic group receiving similar routing patterns. This alignment between task understanding and expert assignment is precisely what enables the oracle to achieve superior performance, highlighting that the task-expert routing manifold misalignment is the key bottleneck limiting router generalization in MoE LLMs.

- 4 ROUTING MANIFOLD REGULARIZATION (ROMA)

To address this limitation, we propose “Routing Manifold Alignment (RoMA)”, a post-training method that aligns the manifold of routing weights with task embeddings through lightweight router fine-tuning. Our key insight is that samples with similar task embeddings should share similar routing patterns to leverage specialized expertise effectively. To achieve this, we introduce a manifold regularization term that encourages alignment between the routing weight manifold and the task embedding manifold. Given a training set D = {(xi,yi)}ni=1 and their associated routing weights {ri}ni=1 (where ri denotes the concatenated routing weights across multiple layers), our goal is to optimize the routers such that samples with similar task embeddings share similar routing patterns.

- 4.1 SUCCESSFUL NEIGHBORHOOD TO IMITATE We first identify the subset of training samples where the MoE produces correct predictions:

S = {j ∈ [n] : f(xj,rj) = yj} (2)

This filtering ensures that our finetuning only imitates from routing patterns for samples in S that lead to successful outputs, preventing the propagation of suboptimal routing strategies.

Given the set of successful samples S, we construct a neighborhood N(xi) for each sample xi based on the task similarity in an embedding space. Let E(·) denote a pre-trained embedding model that

maps input task descriptions/instructions to a semantic representation space. The neighborhood of xi can be defined via k-Nearest Neighbors or ϵ-ball:

k-NN: N(xi) = arg max

A⊆S,|A|≤k

j∈A

sim(E(xi),E(xj)) (3)

ϵ-ball: N(xi) = {j ∈ S : sim(E(xi),E(xj)) ≥ ϵ} (4) where sim(·,·) is a similarity metric, for example, the Gaussian similarity is defined as

sim(E(xi),E(xj)) = exp −

∥E(xi) − E(xj)∥22 2σ2

. (5)

- 4.2 TRAINING OBJECTIVE WITH MANIFOLD REGULARIZATION

Having identified the successful neighborhood for each sample, our next step is to incorporate this structure into the training objective to align routing behaviors with the task embedding geometry. The key idea is that semantically similar samples should not only cluster in the embedding space but also share consistent routing patterns. To achieve this, we introduce a manifold regularization term that explicitly aligns the routing weights manifold with the task embedding manifold by encouraging samples to follow the routing patterns of their successful neighbors, weighted by their semantic similarity.

The (normalized) adjacency Wi,j between sample xi and xj is defined as

sim(E(xi),E(xj))

Wi,j ≜

, ∀j ∈ N(xi), (6)

j∈N(xi) sim(E(xi),E(xj))

where higher weights indicate stronger semantic similarity in the task embedding space. Given Wi,j, the manifold regularization applied to the routing weight ri of sample xi is defined as

Lmanifold(i) ≜

Wi,j∥ri − rj∥22. (7)

j∈N(xi)

By penalizing routing discrepancies ∥ri −rj∥2 between semantically similar samples with large Wi,j, Lmanifold(i) enforces the routing weights manifold to be aligned with the task embedding manifold. Moreover, it moves each sample’s routing weights to those of its “successful neighbors” in the task embedding space. As a consequence, the manifold regularization consolidates the bindings between tasks and their expert choices, and thus improves the generalization.

To ensure that aligned routing patterns also lead to correct predictions, the training objective in RoMA applies the manifold regularization to the cross-entropy loss LCE defined on the outputs.

Ltask(i) = LCE(f(xi,ri),yi). (8) With a regularization coefficient λ ≥ 0, the final objective on sample xi is

##### LRoMA(i) = Ltask(i) + λ · Lmanifold(i) (9)

During training, we only update router parameters while keeping all expert parameters frozen. The gradient update is performed via backpropagation on LRoMA with respect to router parameters:

θrouter(t+1) = θrouter(t) − η∇θrouterLRoMA, (10)

where θrouter represents the parameters of routers and η is the learning rate. While router parameters represent only a small fraction of the total model parameters (0.0095%), we empirically find that only finetuning routers in the last five layers achieves superior performance while significantly saves the training cost, as demonstrated in Figure 6.

- 5 EXPERIMENTS

- 5.1 EXPERIMENTAL SETTINGS

Models We evaluate three recent MoE LLMs: OLMoE-7B-A1B, DeepSeekMoE-16B-A3B, and Qwen3-30B-A3B. OLMoE features a 16-layer transformer with 64 experts per layer, activating 8 per token, totaling 6.9B parameters with 1.3B active per token. DeepSeekMoE uses a 28-layer transformer with 2 shared and 64 routed experts per layer, activating all shared plus 6 routed experts per token, totaling 16.4B parameters with 2.8B active per forward pass. Qwen3-30B-A3B employs a 48-layer transformer with 128 experts per layer, activating 8 per token, totaling 30.5B parameters with 3.3B active per token. These models exemplify distinct MoE designs and scales, enabling a comprehensive evaluation of routing dynamics and generalization behavior.

Baselines We evaluate RoMA against both different adaptation methods (See Table 1) and other models (See Table 2) across eight benchmarks. For adaptation methods, we compare with: (1) In-Context Learning (ICL) (Brown et al., 2020) with embedding-based retrieval for few-shot demonstrations;

- (2) Router Tuning that directly updates the routers; (3) Oracle Tuning that fine-tunes routers with access to optimal routing weights (oracle); (4) Prefix Tuning (Li & Liang, 2021) and Soft Prompt Tuning (Lester et al., 2021) that introduce lightweight trainable parameters while keeping the base model frozen; (5) Dense Backpropagation (Dense BP) (Panda et al., 2025) that enables gradient flow through the full model while updating few parameters; (6) C3PO (Li et al., 2025a), a state-of-the-art test-time routing weights optimization method. For model comparison, we evaluate against models grouped by active parameters (1B, 3B, 7-8B, 13-14B, 27-34B), including recent models like Llama3.2, Gemma2, Qwen2, and Mistral to assess the efficiency of MoE architectures enhanced with RoMA.

General Knowledge (40.8%)

BIG-Bench (10,000) SuperGLUE (10,000)

Training Set comprises 49,000 samples distributed across five task categories, as shown in Figure 4. The dataset includes General Knowledge tasks (BIG-Bench and SuperGLUE), Commonsense reasoning (CommonsenseQA and SocialIQA), Science QA (OpenBookQA and SciQ), Reading comprehension (MultiRC), and Coreference resolution (KnowRef). This diverse composition ensures comprehensive coverage across different reasoning capabilities for effective training.

Commonsense (32.7%)

CommonsenseQA (8,000) SocialIQA (8,000)

Training Set

Science QA (12.2%)

OpenbookQA (3,000) SciQ (3,000)

Coreference (6.1%)

KnowRef (3,000)

Reading (8.2%)

MultiRC (4,000)

Figure 4: Training set statistics.

Benchmarks We evaluate RoMA on eight diverse benchmarks. The evaluation suite includes MMLU, HellaSwag, PIQA, ARC-Challenge, ARC-Easy, WinoGrande, BoolQ and GSM8K. Notably, GSM8K serves as an Out-Of-Distribution (OOD) benchmark since our training set doesn’t contain math-related data. Details about training set and benchmarks is in Appendix A.3 and A.4.

OLMoE Base OLMoE + C3PO OLMoE + RoMA ARC-C

ARC-C

14.6

67.2 88.0

ARC-E

HellaSwag

ARC-E

HellaSwag

66.3 87.4

12.9

86.7

13.2

85.3

2.3 2.1

2.1

1.6

85.3

77.9

1.9

1.6

51.3

88.0

69.0

2.5

15.8

12.4

1.9

1.6

2.3

85.8 65.5

80.7

MMLU

PIQA

PIQA

MMLU

57.8

2.4

2.6 1.8

72.2

2.4

2.6

45.5 75.4

1.9

81.8 49.4 79.6

17.9

16.3

50.8

82.7

WinoGrande GSM-8K

WinoGrande GSM-8K

13.7

81.7

BoolQ

BoolQ

(a) Accuracy (b) Inference Cost

- Figure 5: Performance and inference cost of OLMoE (base model), OLMoE + C3PO and OLMoE

+ RoMA across eight benchmarks. (a) Accuracy: RoMA consistently improves the base model’s performance to be comparable or better than C3PO. (b) Inference cost (in FLOPs ×1011): RoMA maintains nearly the same efficiency as the base model, while C3PO requires test-time optimization and induces 6–7× more FLOPs. These results highlight the effectiveness and efficiency of RoMA.

- 5.2 MAIN RESULTS

Advantage of RoMA over different adaptation methods. Table 1 compares adaptation methods on OLMoE, DeepSeekMoE, and Qwen3-MoE across eight benchmarks. Lightweight methods (ICL, Router/Prefix/Prompt Tuning) yield only modest gains, while Oracle tuning and Dense BP achieve stronger, though still limited, improvements relative to the Oracle upper bound. C3PO performs better than these baselines, yet RoMA achieves the highest overall accuracy. On MMLU, RoMA boosts DeepSeekMoE from 46.2% to 56.8% (+10.6%) and OLMoE from 57.8% to 69.0% (+11.2%), surpassing C3PO by +1.4% and +3.5%, respectively. Although C3PO achieves comparable accuracy as RoMA, its inference cost is 6–7× higher than both RoMA and the base model (See Figure 5), highlighting RoMA ’s superior efficiency–effectiveness trade-off. In addition, RoMA shows more advantages over C3PO on larger models such as DeepSeekMoE and Qwen3-MoE. The accuracy and cost of the other two models are reported in Appendix A.1 and A.2.

Comparison of routing weights manifold before and after RoMA. Figure 3 illustrates the effect of RoMA on routing weights. After applying RoMA, routing weights form clear clusters (Figure 3(a)) that closely align with the task embedding structure (Figure 3(c)). In contrast, the pretrained routing weights show little alignment with task clusters in Figure 3(b), highlighting that RoMA effectively resolves the manifold misalignment problem. Furthermore, the post-RoMA routing patterns closely resemble the oracle routing weights as shown in Figure 3(d), suggesting that our optimization moves

- Table 1: Comparison of RoMA with the Base model, Oracle, test-time adaptation methods (ICL, C3PO), training-based methods (Router/Oracle/Prefix/Prompt Tuning), across eight benchmarks on DeepSeekMoE, OLMoE, and Qwen3-30B-A3B. Details of the baselines and benchmarks are provided in Section 5.1. Bold numbers denote the best performance (excluding Oracle), and underlined numbers denote the second best. RoMA improves DeepSeekMoE from 46.2% to 56.8% (+10.6%), improves OLMoE from 57.8% to 69.0% (+11.2%), and improves Qwen3-30B-A3B from 74.2% to 78.8% (+4.6%) on MMLU, outperforming C3PO on all three models.

HellaSwag

WinoGrande

Method MMLU

ARC-C ARC-E PIQA

BoolQ GSM8K Avg DeepSeekMoE-16B-A3B

Base model 46.2 78.0 50.3 73.8 79.9 70.1 72.3 62.2 66.6 Oracle 63.8 92.5 70.8 85.2 90.3 82.1 83.2 71.8 80.0

ICL 49.0 81.6 56.3 76.2 81.4 72.3 75.8 65.7 69.8 C3PO 55.4 85.7 61.6 80.7 85.8 77.5 78.2 68.5 74.2

Router Tuning 49.3 81.5 57.2 76.6 82.0 73.8 74.5 64.8 70.0 Oracle Tuning 54.2 84.3 60.1 79.5 84.0 76.0 77.5 66.2 72.7 Prefix Tuning 47.8 77.9 52.4 73.8 79.2 70.3 73.1 64.8 67.4 Prompt Tuning 49.3 78.6 55.1 74.7 80.5 72.0 74.2 65.5 68.7 Dense BP 50.1 80.2 54.8 77.3 81.7 74.2 76.1 63.9 69.8

RoMA (Ours) 56.8 87.9 61.4 81.5 85.2 76.8 80.6 67.4 74.7 OLMoE-7B-A1B

Base model 57.8 77.9 51.3 79.8 80.7 72.2 75.4 45.5 67.6 Oracle 72.2 91.5 74.8 91.4 93.6 87.7 84.5 53.2 81.1

ICL 60.3 80.6 58.1 82.5 83.6 76.8 78.9 48.5 71.2 C3PO 65.5 85.3 66.3 87.4 88.0 82.7 79.6 50.8 75.7

Router Tuning 63.2 81.7 62.5 83.8 80.9 75.3 77.8 47.2 71.6 Oracle Tuning 66.8 84.2 65.4 86.1 86.2 80.5 79.9 49.0 74.8 Prefix Tuning 59.3 78.2 54.5 80.4 82.1 73.5 76.8 46.7 68.9 Prompt Tuning 59.7 79.5 55.9 81.3 82.4 74.1 77.2 47.3 69.7 Dense BP 61.8 82.4 57.3 84.1 83.9 76.9 75.2 48.1 71.2

RoMA (Ours) 69.0 86.7 67.2 88.0 85.8 81.8 81.7 49.4 76.2 Qwen3-30B-A3B

Base model 74.2 68.5 56.8 84.3 78.5 65.2 81.3 83.4 74.0 Oracle 82.5 80.3 69.2 92.6 87.4 77.3 90.5 90.9 83.8

ICL 75.8 70.7 59.3 86.1 80.2 67.8 83.5 84.7 76.0 C3PO 77.9 74.1 63.4 88.1 81.7 71.9 85.4 86.0 78.6

Router Tuning 75.3 70.3 60.1 85.7 79.8 68.5 82.8 84.2 75.8 Oracle Tuning 77.2 73.5 62.8 87.6 81.3 71.2 84.9 85.5 78.0 Prefix Tuning 74.5 68.9 57.9 84.8 79.1 66.3 82.1 83.8 74.7 Prompt Tuning 75.0 69.6 58.6 85.2 79.6 67.0 82.7 84.0 75.2 Dense BP 76.1 71.4 59.8 86.5 80.5 69.2 83.8 84.9 76.5

RoMA (Ours) 78.8 74.8 65.5 88.6 83.1 73.8 85.1 86.3 79.5

the model toward theoretically optimal expert assignments. As a result, samples within the same task cluster receive similar routing patterns, enabling more consistent and efficient use of specialized expertise and bridging the performance gap between suboptimal pretrained routing and ideal oracle routing.

Advantage of RoMA over State-of-the-Art models. Table 2 reports LLM performance across eight benchmarks with varying active parameter counts. Notably, OLMoE-7B-A1B+RoMA, with only 1B active parameters, achieves 69.0% on MMLU and 86.7% on HellaSwag, surpassing several 7–8B and even 13B dense models. Similarly, DeepSeekMoE-16B-A3B+RoMA (3B active) delivers substantial gains, matching or exceeding the performance of dense LLMs up to 34B parameters. These results demonstrate that RoMA consistently improves routing quality, enabling small active-parameter MoEs to rival or outperform much larger dense counterparts. Details of models are in Appendix A.5.

- Table 2: Comparison of LLMs with varying active parameters (1B, 3B, 7–8B, 13–14B, 27–34B) evaluated on eight benchmarks. MoE models post-trained by RoMA achieve strong performance, surpassing or matching the performance of much larger dense models. For example, OLMoE-7B-A1B (1B active) achieves 69.0% on MMLU and 86.7% on HellaSwag, outperforming several 7–8B and even 13B dense counterparts. Qwen3-30B-A3B (3B active) achieves 78.8% on MMLU, surpassing even 27–34B dense models, highlighting the effectiveness of MoE+RoMA.

HellaSwag

WinoGrande

BoolQ GSM8K ∼1B active parameters

MMLU

ARC-C ARC-E PIQA

Llama3.2-1B 27.4 57.9 32.1 53.9 72.4 57.4 63.7 39.4 OLMo-1B 24.1 61.8 29.6 55.7 75.6 56.8 64.2 28.5 OLMoE-7B-A1B 57.8 77.9 51.3 79.8 80.7 72.2 75.4 45.5

###### ∼3B active parameters

Gemma2-3B 43.7 66.3 58.4 75.2 71.8 64.5 73.1 41.4 DeepSeekMoE-16B-A3B 46.2 78.0 50.3 73.8 79.9 70.1 72.3 62.2 Qwen3-30B-A3B 74.2 68.5 56.8 84.3 78.5 65.2 81.3 83.4

###### ∼7-8B active parameters

- Qwen2-7B 53.4 74.9 45.8 69.7 77.2 68.1 84.8 79.9 Mistral-7B 59.6 81.0 53.8 79.6 82.2 74.0 68.1 37.9 Llama3.1-8B 57.7 77.9 48.7 80.8 81.4 73.5 81.9 49.6 ∼13-14B active parameters

Llama2-13B 53.8 78.6 50.1 74.5 79.1 70.1 75.7 35.2 Vicuna-13B 51.3 76.2 47.4 72.8 78.0 68.2 71.5 32.2 Qwen1.5-14B 66.7 81.5 58.0 85.3 82.1 76.9 81.3 58.4

∼27-34B active parameters

Gemma2-27B 75.2 86.4 71.4 88.9 83.2 79.0 84.5 61.3 Yi-34B 73.5 83.1 58.2 82.6 82.6 78.9 83.1 63.5 Llama2-34B 62.6 79.4 54.5 77.5 81.9 76.0 78.1 42.2

RoMA (Ours)

DeepSeekMoE-16B-A3B 56.8 87.9 61.4 81.5 85.2 76.8 80.6 67.4 OLMoE-7B-A1B 69.0 86.7 67.2 88.0 85.8 81.8 81.7 49.4

- Qwen3-30B-A3B 78.8 74.8 65.5 88.6 83.1 73.8 85.1 86.3

- 5.3 ABLATION STUDY

We perform a series of ablation studies to systematically analyze the design choices behind RoMA. Specifically, we investigate: (i) which layers to regularize, (ii) which token positions to use for routing guidance, (iii) how to select neighbors for manifold alignment, (iv) the effect of training set size, and (v) the choice of regularization method. These ablations help identify the most effective and efficient configuration, revealing which factors are critical for performance. All experiments are conducted on OLMoE, and experiment results on DeepSeekMoE are provided in the Appendix A.6.

Layer Selection Figure 6 examines how applying routing manifold regularization to different subsets of layers affects model performance. Applying RoMA to a single layer yields only modest gains (69.1–69.7%), while extending it to two layers improves accuracy above 71%. Performance continues to increase as more layers are regularized, with the last five layers (L5) achieving the highest accuracy of 76.2%, even surpassing the All-Layer configuration (75.1%). These results highlight that the final layers are particularly critical for routing quality, and that selectively regularizing a small set of strategically important layers is both more effective and more efficient than uniformly applying RoMA across all layers.

Token Selection Figure 7 presents the effect of different token selection strategies when applying RoMA. Using multiple tokens (e.g., the first three or middle three) provides moderate improvements over the baseline, with the last 3 tokens (Last3) reaching 74.5%. Among single-token choices, the

76.2

77

75.1

Base

2 Layers 5 Layers

All Layers

AvgAccuracy(%)

| |
|---|

73.8

75

1 Layer

Ours

72.9

71.8 72.1 72.6 72.3

73

71.4 70.8 70.6 69.9

69.7 70.2

71

69.1 68.4

69

67.6

67

65

Base F1 M1 L1 F1M1F1L1M1L1 F2 M2 L2 F2M3F2L3M2L3 F5 M5L5(Ours) All16

- Figure 6: Applying RoMA at different layers (F: early layers, M: middle layers, L: late layers). Fine-tuning the routers in the last five layers (L5, RoMA) achieves the best performance.

Base First3 Middle3 Last3 First1 Middle1 Last1

65

70

75

AvgAccuracy(%)

67.6

70.1

68.7

74.5

71.4

69.2

76.2

| |
|---|

Baseline 3 Tokens 1 Token Ours

| |
|---|

| |
|---|

- Figure 7: Applying RoMA to routing weights of tokens at different positions. Regularizing the Last1 token’s routing weights performs the best.

76.2 Baseline 74.9

AvgAccuracy(%)

74.1

75

-neighbor k-neighbor

72.5 72.3

Ours

68.9

70

67.6 67.8

65

Base Rand =0.3 =0.5 =0.7 k=1 k=3 k=5

Figure 8: Comparing neighbor selection strategies in RoMA. Rand—random neighbors. k-NN with k = 3 achieves the best performance.

last 1 token (Last1) performs best (76.2%), outperforming both the first one token (First1) (71.4%) and the middle one token (Middle1) (69.2%). These results indicate that the final tokens contain richer task-relevant information for guiding expert routing than earlier or middle tokens. Moreover, the superiority of Last1 over Last3 highlights that a single, well-chosen token can be more effective and efficient than aggregating multiple tokens.

76.2

76.2

Baseline Set Size Ours

75.0

Baseline

AvgAccuracy(%)

AvgAccuracy(%)

75

75

73.6

Regularization

71.5

Ours

70.8

70.7

70

70

68.5

67.6 68.2

67.6

65

65

Base L1 L2 Entropy Manifold

Base 10% 30% 50% 70% 100%

Figure 9: Comparing different training set sizes for RoMA on OLMoE. While the full training set (100%) yields the best performance, 30% suffices to achieve substantial gains over the baselines.

Figure 10: Comparing different regularization methods with RoMA. RoMA’s manifold regularization achieves the best performance.

Neighborhood Selection Figure 8 compares different strategies for selecting neighbors in RoMA. Random neighbor selection yields almost no improvement over the baseline (67.8% vs. 67.6%). Using ϵ-neighborhoods shows sensitivity to the choice of radius: performance improves steadily from 68.9% (ϵ=0.3) to a peak of 74.1% at ϵ=0.5, but drops slightly when the radius grows larger (ϵ=0.7). In contrast, k-nearest neighbor selection provides more stable gains, with k=3 achieving the best overall accuracy of 76.2%. Notably, this surpasses both smaller (k=1) and larger (k=5) settings, suggesting that a moderate number of neighbors balances robustness and noise. These results highlight that careful neighborhood design is crucial for effective manifold alignment, and that our chosen k=3 strategy offers the most reliable improvement.

Training Set Size Figure 9 examines how the size of the training set used for RoMA affects performance. Starting from the baseline accuracy of 67.6%, even using only 10% of the training data yields a noticeable gain (68.5%). Performance improves steadily as more data is available, reaching 70.8% at 30% and 73.6% at 50%. With 70% of the data, accuracy rises further to 75.0%, and using the full dataset achieves the best performance of 76.2%. These results demonstrate that RoMA benefits consistently from additional training data, but also that substantial improvements can already be obtained with a relatively small fraction of the dataset, highlighting its data efficiency.

Regularization Methods Figure 10 compares different regularization strategies applied to the router. Standard techniques such as L1 and L2 penalties yield only modest improvements over the baseline (68.2% and 71.5%, respectively), while entropy regularization reaches a similar level (70.7%). In contrast, our proposed manifold regularization achieves the best result of 76.2%, substantially outperforming all alternatives. This demonstrates that aligning routing weights in the task embedding space provides a more effective inductive bias than generic sparsity or entropy-based constraints, highlighting the unique advantage of RoMA’s manifold perspective.

- 6 CONCLUSIONS

Our work introduces RoMA, a lightweight router post-training method for sparse Mixture-of-Experts LLMs. By aligning routing weights with the underlying task embedding manifold through manifold regularization, RoMA addresses the fundamental misalignment between task understanding and expert utilization in MoE models. Our approach requires updating only router parameters while keeping experts frozen, yet consistently improves accuracy across diverse benchmarks by 7–15% without increasing inference cost. Extensive experiments demonstrate that RoMA enables small active-parameter MoEs to rival or even surpass much larger dense models, highlighting both the efficiency and effectiveness of RoMA. Beyond performance gains, our findings underscore the importance of geometric alignment between task representation and expert selection, offering new insights for advancing routing strategies in future MoE architectures.

REFERENCES

Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023.

Mikhail Belkin, Partha Niyogi, and Vikas Sindhwani. Manifold regularization: A geometric framework for learning from labeled and unlabeled examples. Journal of machine learning research, 7

(11), 2006.

Yonatan Bisk, Rowan Zellers, Ronan Le Bras, Jianfeng Gao, and Yejin Choi. Piqa: Reasoning about physical commonsense in natural language. In Proceedings of the AAAI Conference on Artificial Intelligence, 2020.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.

Wei-Lin Chiang, Zhuohan Li, Zi Lin, et al. Vicuna: An open-source chatbot impressing gpt-4 with 90% chatgpt quality. Blog post, 2023.

Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael Collins, and Kristina Toutanova. Boolq: Exploring the surprising difficulty of natural yes/no questions. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics (NAACL), 2019.

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2018.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems. In arXiv preprint arXiv:2110.14168, 2021.

Damai Dai, Chengqi Deng, Chenggang Zhao, RX Xu, Huazuo Gao, Deli Chen, Jiashi Li, Wangding Zeng, Xingkai Yu, Yu Wu, et al. Deepseekmoe: Towards ultimate expert specialization in mixtureof-experts language models. arXiv preprint arXiv:2401.06066, 2024a.

Wenxuan Dai et al. Deepseekmoe: Towards ultimate expert specialization in mixture-of-experts language models. arXiv preprint arXiv:2401.06066, 2024b.

Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv e-prints, pp. arXiv–2407, 2024.

Ali Emami, Paul Trichelair, Adam Trischler, Kaheer Suleman, Hannes Schulz, and Jackie Chi Kit Cheung. The knowref coreference corpus: Removing gender and number cues for difficult pronominal anaphora resolution. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (ACL), 2019.

William Fedus, Barret Zoph, and Noam Shazeer. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. Journal of Machine Learning Research, 23(120):1–39, 2022.

Dirk Groeneveld, Kyle Lo, et al. Olmo: Accelerating the science of language models. arXiv preprint arXiv:2402.00838, 2024.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. In International Conference on Learning Representations (ICLR), 2021.

Albert Jiang et al. Mistral 7b. arXiv preprint arXiv:2310.06825, 2023.

Daniel Khashabi, Snigdha Chaturvedi, Michael Roth, Shyam Upadhyay, and Dan Roth. Looking beyond the surface: A challenge set for reading comprehension over multiple sentences. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics (NAACL), 2018.

Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang, Maxim Krikun, Noam Shazeer, and Zhifeng Chen. Gshard: Scaling giant models with conditional computation and automatic sharding. arXiv preprint arXiv:2006.16668, 2020.

Brian Lester, Rami Al-Rfou, and Noah Constant. The power of scale for parameter-efficient prompt tuning. arXiv preprint arXiv:2104.08691, 2021.

Xiang Lisa Li and Percy Liang. Prefix-tuning: Optimizing continuous prompts for generation. arXiv preprint arXiv:2101.00190, 2021.

Xin Li and Anand Sarwate. Unraveling the localized latents: Learning stratified manifold structures in llm embedding space with sparse mixture-of-experts. arXiv preprint arXiv:2502.13577, 2025.

Zhongyang Li, Ziyue Li, and Tianyi Zhou. C3po: Critical-layer, core-expert, collaborative pathway optimization for test-time expert re-mixing. arXiv preprint arXiv:2504.07964, 2025a.

Zhongyang Li, Ziyue Li, and Tianyi Zhou. R2-t2: Re-routing in test-time for multimodal mixture-ofexperts. arXiv preprint arXiv:2502.20395, 2025b.

Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. Can a suit of armor conduct electricity? a new dataset for open book question answering. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2018.

Nay Myat Min, Long H Pham, Yige Li, and Jun Sun. Crow: Eliminating backdoors from large language models via internal consistency regularization. arXiv preprint arXiv:2411.12768, 2024.

Niklas Muennighoff, Luca Soldaini, Dirk Groeneveld, Kyle Lo, Jacob Morrison, Sewon Min, Weijia Shi, Pete Walsh, Oyvind Tafjord, Nathan Lambert, et al. Olmoe: Open mixture-of-experts language models. arXiv preprint arXiv:2409.02060, 2024.

Ashwinee Panda, Vatsal Baherwani, Zain Sarwar, Benjamin Therien, Supriyo Chakraborty, and Tom Goldstein. Dense backpropagation improves training for sparse mixture-of-experts. arXiv preprint arXiv:2504.12463, 2025.

Michael Robinson, Sourya Dey, and Tony Chiang. Token embeddings violate the manifold hypothesis. arXiv preprint arXiv:2504.01002, 2025.

William Rudman and Carsten Eickhoff. Stable anisotropic regularization. arXiv preprint arXiv:2305.19358, 2023.

Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. Winogrande: An adversarial winograd schema challenge at scale. In Proceedings of the AAAI Conference on Artificial Intelligence, 2020.

Maarten Sap, Hannah Rashkin, Derek Chen, Ronan Le Bras, and Yejin Choi. Socialiqa: Commonsense reasoning about social interactions. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2019.

Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. arXiv preprint arXiv:1701.06538, 2017.

Alon Talmor, Jonathan Herzig, Nicholas Lourie, and Jonathan Berant. Commonsenseqa: A question answering challenge targeting commonsense knowledge. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics (NAACL), 2019.

Gemma Team. Gemma 2: Improving open language models at a practical size. arXiv preprint

arXiv:2408.00118, 2024a. Qwen Team. Qwen2 technical report. arXiv preprint arXiv:2407.10671, 2024b. Hugo Touvron et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint

arXiv:2307.09288, 2023.

Alex Wang, Yada Pruksachatkun, Nikita Nangia, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel Bowman. Superglue: A stickier benchmark for general-purpose language understanding systems. In Advances in Neural Information Processing Systems (NeurIPS), 2019.

Jason Wei et al. Beyond the imitation game: Quantifying and extrapolating the capabilities of language models. arXiv preprint arXiv:2206.04615, 2022.

Johannes Welbl, Nelson F. Liu, and Matt Gardner. Crowdsourcing multiple choice science questions. In Proceedings of the 3rd Workshop on Noisy User-generated Text (W-NUT), 2017.

Alex Young, Bei Chen, Chao Li, Chengen Huang, Ge Zhang, Guanwei Zhang, Guoyin Wang, Heng Li, Jiangcheng Zhu, Jianqun Chen, et al. Yi: Open foundation models by 01. ai. arXiv preprint arXiv:2403.04652, 2024.

Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. Hellaswag: Can a machine really finish your sentence? In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (ACL), 2019.

Yanqi Zhou, Tao Lei, Hanxiao Liu, Nan Du, Yanping Huang, Vincent Zhao, Andrew M Dai, Quoc V Le, James Laudon, et al. Mixture-of-experts with expert choice routing. Advances in Neural Information Processing Systems, 35:7103–7114, 2022.

A APPENDIX

- A.1 COMPARISON BETWEEN ROMA AND BASELINES ON DEEPSEEKMOE-16B-A3B AND QWEN3-30B-A3B

ARC-C

HellaSwag

MMLU

GSM-8K

BoolQ

WinoGrande

PIQA

ARC-E

87.9

61.4

81.5

85.2

76.8

80.6

67.4

56.8

50.3 73.8 78.0

79.9 46.2

70.1

72.3 62.2

Gemma2-3B Mistral-7B Vicuna-13B Llama2-34B OLMoE-7B-A1B OLMoE-7B-A1B + RoMA RoMA Improvement

ARC-C

HellaSwag

MMLU

GSM-8K

BoolQ

WinoGrande

PIQA

ARC-E

74.8

65.5

88.6

83.1

73.8

85.1

79.5

78.8

56.8 84.3

68.5 78.5

74.2

65.2

74.0 81.3

(a) DeepSeekMoE-16B-A3B (b) Qwen3-30B-A3B

- Figure 11: (a): Radar figure of DeepSeekMoE-16B-A3B, (b) Radar figure of Qwen3-30B-A3B. RoMA consistently improves model’s performance on multiple benchmarks.

A.2 ACCURACY AND COST OF DEEPSEEKMOE-16B-A3B AND QWEN3-30B-A3B

ARC-C

HellaSwag

MMLU

WinoGrande GSM-8K

PIQA

ARC-E

87.9

61.4 81.5

85.2

76.8

80.6

68.5

56.8

61.6

80.7 85.7

85.8 55.4

77.5

67.4 78.2

BoolQ

78.0

46.2

62.2 70.1 72.3

79.9

73.8

50.3

DeepSeekMoE Base DeepSeekMoE + C3PO DeepSeekMoE + RoMA ARC-C

HellaSwag

MMLU

WinoGrande GSM-8K

PIQA

ARC-E

39.2

47.5

54.2

41.8

49.4

37.8

40.3

44.1

BoolQ

8.3

6.4

5.9 4.7

7.3

5.5

7.8 6.8

9.2

6.2

7.5

5.4

6.7

7.0

5.0

5.3

(a) Accuracy (b) Inference Cost

- Figure 12: (a) Accuracy: RoMA achieves similar accuracy improvement as C3PO on DeepSeekMoE16B-A3B. (b) Inference cost (in FLOPs ×1011): RoMA maintains nearly the same efficiency as the base model, while C3PO requires test-time optimization and induces 6–7× more FLOPs.

- A.3 DETAILS OF TRAINING SET

Our training set comprises 49,000 samples distributed across five task categories, ensuring comprehensive coverage across diverse reasoning skills:

QwenMoE Base QwenMoE+ C3PO QwenMoE + RoMA ARC-C

ARC-C

63.5

48.7

ARC-E

HellaSwag

ARC-E

HellaSwag

74.8

63.4 88.1

45.3

88.6

58.8

74.1

8.5

56.8

10.1

6.2

68.5

7.5

84.3

8.9

5.9

83.1

78.8

7.9

44.1

47.2

6.8

6.0

7.7

78.5

MMLU

81.7 77.9

PIQA

PIQA

MMLU

74.2

7.7

6.2 5.0

65.2

8.9

7.3

83.4 81.3

5.9

71.9 86.0

41.8

53.6

86.3

73.8

WinoGrande GSM-8K

WinoGrande GSM-8K

85.1

40.2

85.4

BoolQ

BoolQ

(a) Accuracy (b) Inference Cost

- Figure 13: (a) Accuracy: RoMA achieves similar accuracy improvement as C3PO on Qwen3-30BA3B. (b) Inference cost (in FLOPs ×1011): RoMA maintains nearly the same efficiency as the base model, while C3PO requires test-time optimization and induces 6–7× more FLOPs.

Benchmarks Size MMLU 14,042 HellaSwag 10,042 PIQA 1,838 ARC-C 1,172 ARC-E 2,376 WinoGrande 1,267 BoolQ 3,227 GSM8k 1,000

General Knowledge (40.8%)

BIG-Bench (10,000) SuperGLUE (10,000)

Commonsense (32.7%)

CommonsenseQA (8,000) SocialIQA (8,000)

Training Set

Science QA (12.2%)

OpenbookQA (3,000) SciQ (3,000)

Coreference (6.1%)

KnowRef (3,000)

Reading (8.2%)

Figure 15: Overview of evaluation benchmarks we use for router post-training. Benchmarks are reserved strictly for evaluation.

MultiRC (4,000)

Figure 14: Training Set by Task

- • BIG-Bench (Wei et al., 2022): A large-scale collaborative benchmark covering a wide range of tasks such as logical reasoning, linguistic phenomena, and commonsense knowledge. It is designed to probe broad generalization and emergent capabilities in large language models.
- • SuperGLUE (Wang et al., 2019): A benchmark suite for general natural language understanding, consisting of challenging tasks such as natural language inference, word sense disambiguation, and question answering. It extends the original GLUE benchmark to push models toward higher-level reasoning.
- • CommonsenseQA (Talmor et al., 2019): A multiple-choice dataset focusing on commonsense reasoning, requiring models to connect concepts and apply everyday knowledge.
- • SocialIQA (Sap et al., 2019): A benchmark for social commonsense reasoning, where models must infer likely intents, reactions, and motivations of people in everyday situations.
- • OpenBookQA (Mihaylov et al., 2018): A science-oriented QA dataset requiring both retrieval from a small “open book” of facts and additional commonsense reasoning.
- • SciQ (Welbl et al., 2017): A dataset of science exam-style questions covering physics, biology, and chemistry, testing factual recall and reasoning in scientific contexts.
- • MultiRC (Khashabi et al., 2018): A reading comprehension benchmark with multi-sentence passages and multi-answer questions, requiring deeper reasoning across long contexts.

- • KnowRef (Emami et al., 2019): A coreference resolution dataset where multiple entities are mentioned, and models must resolve ambiguous pronouns using contextual cues.

- A.4 DETAILS OF BENCHMARKS

We evaluate our method on eight widely used benchmarks covering general knowledge, commonsense reasoning, science QA, and mathematical problem-solving:

- • MMLU (Hendrycks et al., 2021): A comprehensive benchmark of 57 subjects spanning STEM, humanities, social sciences, and professional domains. It measures models’ multitask accuracy and general world knowledge.
- • HellaSwag (Zellers et al., 2019): A commonsense benchmark requiring models to select the most plausible continuation of a given context. It emphasizes grounded reasoning about everyday scenarios.
- • PIQA (Bisk et al., 2020): A physical commonsense reasoning dataset, where models must infer the correct solution to physical problems from everyday settings.
- • ARC-Challenge (Clark et al., 2018): A benchmark of challenging grade-school science questions requiring reasoning, knowledge retrieval, and integration across multiple facts.
- • ARC-Easy (Clark et al., 2018): The easier subset of ARC focusing on factual recall and simpler reasoning in grade-school science.
- • WinoGrande (Sakaguchi et al., 2020): A large-scale benchmark for pronoun resolution and commonsense reasoning, designed to be adversarially filtered and less susceptible to dataset artifacts.
- • BoolQ (Clark et al., 2019): A reading comprehension dataset of yes/no questions paired with passages from Wikipedia, requiring models to integrate text understanding with factual reasoning.
- • GSM8k (Cobbe et al., 2021): A dataset of grade-school math word problems requiring multi-step numerical reasoning. We treat GSM8k as an out-of-distribution (OOD) evaluation since math is not included in our training set.

- A.5 MODEL DESCRIPTIONS

In this section, we provide additional details of the models reported in Table 2. These models cover a broad range of active parameters (1B, 3B, 7–8B, 13–14B, 27–34B), including dense LLMs and sparse Mixture-of-Experts (MoE) variants. All results are reported in the main text (see Table 2).

#### Models with ∼1B Active Parameters

- • Llama3.2-1B (Dubey et al., 2024): A 1B-parameter dense model in the Llama 3.2 family.
- • OLMo-1B (Groeneveld et al., 2024): Dense model from the AllenAI OLMo family.
- • OLMoE-7B-A1B Muennighoff et al. (2024): A sparse MoE variant of OLMoE with 7B total parameters and ∼1B active per token.

#### Models with ∼3B Active Parameters

- • Gemma2-3B (Team, 2024a): Google’s Gemma2 family dense model.
- • Qwen1.5-14B-A3B (Bai et al., 2023): Sparse MoE variant of Qwen1.5 with 14B total parameters and ∼3B active per token.
- • DeepSeekMoE-16B-A3B (Dai et al., 2024b): Sparse MoE model with 16B total parameters and ∼3B active per token.

#### Models with ∼7–8B Active Parameters

- • Qwen2-7B (Team, 2024b): Dense model from the Qwen2 family.
- • Mistral-7B (Jiang et al., 2023): Dense model emphasizing efficient training and inference.
- • Llama3.1-8B (Dubey et al., 2024): A dense model from the Llama 3.1 family.

#### Models with ∼13–14B Active Parameters

- • Llama2-13B (Touvron et al., 2023): Dense model from the Llama 2 release.
- • Vicuna-13B (Chiang et al., 2023): Instruction-tuned LLM based on Llama2-13B.
- • Qwen1.5-14B (Bai et al., 2023): Dense version of Qwen1.5.

#### Models with ∼27–34B Active Parameters

- • Gemma2-27B (Team, 2024a): Largest Gemma2 dense variant.
- • Yi-34B (Young et al., 2024): Dense model with 34B parameters.
- • Llama2-34B (Touvron et al., 2023): Large dense model from the Llama 2 family.

- A.6 DETAILS OF ABLATION STUDY ON DEEPSEEKMOE In this section, we provide a detailed ablation study of RoMA on the DeepSeekMoE model.

Layer configurations (Table 3). When applying RoMA to different numbers of layers, we find that using a single layer or two layers provides modest gains (e.g., F1 at 68.0%, L2 at 70.6%). Increasing to five layers yields further improvements (up to 72.4%), while applying RoMA on all layers gives 73.7%. Interestingly, restricting the method to the last five layers achieves the best result (74.7%), surpassing even the all-layer setting, which highlights the importance of critical-layer selection.

Table 3: DeepSeekMoE average accuracy (%) across different layer configurations

Group Configuration Avg. Acc. (%) Base Base 66.6

- 1 Layer

F1 68.0 M1 67.4 L1 68.6

- 2 Layers

F1M1 69.0 F1L1 70.2 M1L1 69.6 F2 69.4 M2 68.8 L2 70.6

F2M3 70.8 F2L3 71.3 M2L3 71.0 F5 72.4 M5 71.6

5 Layers

All Layers All16 73.7 Ours L5 (Ours) 74.7

Token positions (Table 4). We next evaluate applying RoMA to the routing weights of different tokens. Regularizing the first or middle tokens shows only limited improvements (69.0% and 67.6%, respectively), while the last token positions provide stronger performance. In particular, Last1 achieves the best result (74.7%), indicating that the most informative supervision signal for routing weights lies in the final tokens.

Neighbor selection (Table 5). We compare ε-neighbor and k-neighbor strategies. Small ε (0.3) gives a minor improvement (68.0%), while moderate ε (0.5) achieves a strong 72.8%. For k-neighbors, increasing k improves accuracy up to k = 3 (74.7%), after which performance saturates (k = 5 at

- 73.4%). This suggests that a balanced neighbor selection (neither too sparse nor too dense) is crucial for generalization.

Table 4: DeepSeekMoE average accuracy (%) when applying RoMA on different token positions.

Group Configuration Avg. Acc. (%) Baseline Base 66.6

First3 69.0 Middle3 67.6 Last3 73.1

3 Tokens

First1 70.2 Middle1 68.1

1 Token

Ours Last1 (Ours) 74.7

Table 6: DeepSeekMoE average accuracy (%) with different training set sizes.

Configuration Avg. Acc. (%) Base 66.6 10% 67.5 30% 69.8 50% 72.2 70% 73.6 100% (Ours) 74.7

###### Table 5: DeepSeekMoE average accuracy (%) with different neighbor selection strategies.

Group Configuration Avg. Acc. (%) Baseline Base 66.6 Baseline Rand 66.8

ε = 0.3 68.0 ε = 0.5 72.8 ε = 0.7 71.2

ε-neighbor

k=1 71.0 k=3 (Ours) 74.7 k=5 73.4

k-neighbor

Table 7: DeepSeekMoE average accuracy (%) with different regularization methods.

Configuration Avg. Acc. (%) Base 66.6 L1 67.2 L2 70.3 Entropy 69.8 Manifold (Ours) 74.7

Training set size (Table 6). We investigate different proportions of training data used for regularization. Performance grows steadily with larger set sizes, from 67.5% at 10% to 73.6% at 70%. Using the full dataset (100%) achieves the best result (74.7%), confirming that more data consistently strengthens the alignment of routing weights.

Regularization methods (Table 7). We compare different regularization objectives. L1 and L2 losses improve the baseline to 67.2% and 70.3%, respectively, while entropy regularization achieves 69.8%. Our proposed manifold regularization significantly outperforms all alternatives, reaching

- 74.7%, which demonstrates the effectiveness of aligning routing weights with the manifold structure of successful neighbors.

