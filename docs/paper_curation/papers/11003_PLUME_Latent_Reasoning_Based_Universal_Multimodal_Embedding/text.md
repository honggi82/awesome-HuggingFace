# arXiv:2604.02073v3[cs.CV]20Apr2026

## PLUME: Latent Reasoning Based Universal Multimodal Embedding

Chenwei He1* Xiangzhao Hao2,3* Tianyu Yang2,3* Yuxiang Ma1 Yuheng Jia1 Lingxiang Wu2,3 Chaoyang Zhao2,3 Haiyun Guo2,3† Jinqiao Wang2,3 1Southeast University 2Institute of Automation, Chinese Academy of Sciences 3University of Chinese Academy of Sciences

{hechenwei, 220256453, yhjia}@seu.edu.cn {haoxiangzhao2023, yangtianyu2024}@ia.ac.cn {lingxiang.wu, chaoyang.zhao, haiyun.guo, jqwang}@nlpr.ia.ac.cn

### Abstract

Universal multimodal embedding (UME) maps heterogeneous inputs into a shared retrieval space with a single model. Recent approaches improve UME by generating explicit chain-of-thought (CoT) rationales before extracting embeddings, enabling multimodal large language models to better infer complex query intent. However, explicit CoT incurs substantial inference overhead and can compress rich multimodal evidence into a narrow textual bottleneck. We propose PLUME, a latent reasoning framework that advances UME by replacing verbalized CoT with a short autoregressive rollout of continuous latent states. To support diverse multimodal queries, PLUME further introduces a semantic-anchor-guided transition adapter that steers latent rollout along different reasoning trajectories under the same fixed computation budget. To stabilize training, PLUME adopts a progressive explicit-to-latent curriculum that uses verbalized reasoning only as a temporary training scaffold and gradually transfers this behavior into hidden-state computation, eliminating explicit CoT at inference. On the 78-task MMEB-v2 benchmark, PLUME outperforms strong explicit-CoT UME baselines while reducing reasoning from hundreds of generated tokens to fewer than 10 latent steps, delivering over 30× faster inference. PLUME is especially well suited to retrieval settings where relevant evidence is dense, structurally complex, and difficult to organize through verbalized intermediate rationales, such as video and visual document retrieval. These results show that structured latent computation can preserve the benefits of intermediate reasoning without the overhead of explicit rationale generation, providing a stronger and more efficient paradigm for practical retrieval systems. Our code

*Equal contribution. †Corresponding author.

[Figure 1]

Figure 1. PLUME achieves a favorable accuracy–efficiency tradeoff on MMEB-v2. The x-axis shows inference throughput on a single H20 GPU and the y-axis shows average MMEB-v2 performance.

and data are publicly available at https://github. com/haoxiangzhao12138/PLUME.

### 1. Introduction

Universal multimodal embedding (UME) aims to map heterogeneous inputs, including text, images, videos, and visual documents, into a shared retrieval space with a single model [20, 31, 54]. In real-world retrieval, however, many queries cannot be resolved by surface-level similarity alone. They often require compositional spatial understanding, knowledge-intensive visual inference, or the aggregation of temporally and structurally dispersed evidence. These demands have made Multimodal Large Language Models (MLLMs) [9, 26, 29, 40] an increasingly attractive backbone for UME, thanks to their native multimodal grounding, strong semantic alignment, and broad world knowledge. Yet simply adopting an MLLM as the en-

[Figure 2]

- Figure 2. Comparison of three universal multimodal embedding paradigms. Left: early discriminative UME forms embeddings through single pass encoding, preserving efficiency but without explicitly modeling intermediate reasoning. Middle: explicit CoT UME improves reasoning by generating long textual traces before embedding extraction, but incurs substantial inference latency and token cost. Right: PLUME internalizes reasoning into a compact latent rollout and adapts the reasoning path with semantic-anchor-guided expert routing, achieving reasoning-aware embedding with substantially lower inference cost.

coder does not automatically translate its reasoning potential into stronger embeddings[46]. In most existing UME pipelines, the embedding is still formed in a single pass, leaving limited room for deliberate intermediate computation when query intent is complex. This raises a central question for UME: how can we leverage the reasoning capability of MLLMs during embedding formation without sacrificing retrieval efficiency?

Existing attempts to address this question mainly follow two directions, as illustrated in Figure 2. Single-pass MLLM-based methods [19, 20, 28, 54] are efficient, but they require the model to collapse complex query interpretation, evidence integration, and representation formation into one forward pass. To better handle such complexity, recent reasoning-enhanced methods, such as TTE [5], UMER1 [24], and TRACE [13], first generate an explicit chainof-thought (CoT) rationale [41, 43] before deriving the final embedding. While effective, this strategy introduces a dual bottleneck. Computationally, generating hundreds of reasoning tokens per sample incurs substantial autoregressive decoding overhead and severely limits inference throughput. Representationally, routing multimodal reasoning through discrete textual tokens creates a narrow bottleneck that may discard fine-grained continuous evidence and constrain how richly multimodal information is carried into the final embedding. As a result, explicit CoT ties the benefits of multi-step computation to a verbose interface that is fundamentally mismatched with the efficiency demands of retrieval.

In light of this, we take a different perspective on UME: what retrieval needs is intermediate computation, not necessarily verbalized intermediate text. As illustrated in Fig-

ure 2(c), the multi-step reasoning that helps embedding quality can instead unfold directly in the continuous hidden space of the backbone [4, 12, 36]. A short latent rollout can preserve the sequential dependency structure of reasoning while avoiding long-form text generation. Yet moving from explicit reasoning to latent reasoning is not a trivial substitution in the multimodal setting. Unlike pure language tasks, UME must handle videos, images, documents, and text within one shared framework, and these inputs demand different forms of intermediate computation over temporal dynamics, spatial relations, layout structure, and semantic abstraction. Once reasoning is executed within a short latent budget, the key challenge is no longer whether to reason, but how to allocate this compact latent computation adaptively across heterogeneous multimodal queries instead of forcing every input through the same fixed reasoning path.

To tackle above issues, we propose PLUME, a latent reasoning framework for universal multimodal embedding. PLUME internalizes reasoning for UME into a compact latent process inside the MLLM, allowing the model to perform multi-step computation without generating explicit rationales. To make this latent process suitable for structurally diverse multimodal inputs, PLUME further introduces a semantic-anchor-guided transition adapter that steers latent computation according to the semantic structure of the input, enabling different queries to follow different reasoning patterns under the same rollout budget. Finally, rather than viewing explicit CoT merely as a costly inference procedure, PLUME uses it as a temporary training scaffold. During training, the model is first exposed to verbalized intermediate reasoning and then gradually shifts reasoning process into latent rollouts, progressively replac-

ing explicit textual rationales with hidden-state computation until explicit CoT is no longer needed at inference time.

Experiments on MMEB-v2 show that PLUME outperforms strong explicit-CoT UME baselines while compressing reasoning from hundreds of generated tokens to fewer than ten latent steps and delivering over 30x faster inference, as shown in Figure1. PLUME is particularly effective on retrieval tasks where relevant evidence is dense, structurally complex, and difficult to organize through verbalized intermediate rationales, such as video and visual document retrieval. Taken together, these results suggest that strong UME benefits more from adaptive intermediate computation than from explicit verbalized rationales. By retaining reasoning quality inside a compact latent process, PLUME breaks the dual bottleneck of explicit CoT, bringing the benefits of intermediate reasoning back into the efficiency regime required by practical UME systems.

In summary, our contributions are:

- • A latent reasoning framework for UME. We introduce PLUME to internalize intermediate reasoning into a short continuous latent process for UME, replacing costly explicit chain-of-thought generation while preserving the benefits of intermediate computation.
- • An input-adaptive latent reasoning architecture. We design a semantic-anchor-guided transition adapter that allocates latent computation adaptively across heterogeneous multimodal queries, allowing the same compact rollout budget to support different reasoning patterns for images, videos, documents, and text.
- • Strong empirical gains in both effectiveness and efficiency. We show that latent reasoning can advance UME beyond explicit-CoT baselines, achieving stronger retrieval performance on MMEB-v2 while reducing reasoning from hundreds of generated tokens to fewer than ten latent steps and delivering over 30x faster inference, with particularly strong gains on video and visual document retrieval.

### 2. Related Work

#### 2.1. Universal Multimodal Embedding

Universal multimodal embedding (UME) maps heterogeneous inputs into a shared retrieval space with a single model. Early dual-encoder methods such as CLIP [34], ALIGN [17], SigLIP [51], and BLIP-2 [27] learn aligned image-text representations via contrastive objectives [32] but are less effective on complex multimodal compositions. UniIR [42] and MagicLens [52] begin to address multi-task multimodal retrieval within unified frameworks. Building on advances in LLM-based text embedding [25, 38, 39, 44], MLLM-based approaches [47] further overcome this limitation: E5-V [19] and MM-Embed [28] prompt MLLMs for universal embeddings; VLM2Vec [20] introduces the

MMEB benchmark; and VLM2Vec-V2 [31], GME [54], UniME [10], LamRA [30], LLaVE [23], MoCa [3], and DUME [55] further improve retrieval quality and modality coverage. More recent efforts explore multi-vector representations [6], large-scale data synthesis [56, 57], visual document retrieval [49], and reinforcement-learningbased alignment [48] to push the accuracy–efficiency frontier. However most methods derive embeddings from a single forward pass without modeling intermediate reasoning, limiting performance on complex retrieval query.

#### 2.2. Reasoning-Enhanced Embedding

Chain-of-thought (CoT) prompting [41, 43] elicits multistep reasoning in language models, and subsequent extensions such as multimodal CoT [45] and preferenceoptimized reasoning [53] further strengthen reasoning quality. Scaling this idea, reasoning-specialized LLMs such as DeepSeek-R1 [11] have demonstrated the power of longform reasoning. Recent work extends explicit reasoning to embeddings: Think-then-Embed (TTE) [5], UME-R1 [24], TRACE [13], and Embed-RL [18] generate an explicit reasoning trace before extracting the embedding, and reasoning-augmented retrieval representations [23] further confirm the value of extra reasoning computation. These methods yield consistent accuracy gains, yet producing hundreds of reasoning tokens per sample sharply increases latency and memory cost. In contrast, PLUME retains the benefits of structured reasoning without generating rationale tokens at inference time.

#### 2.3. Latent Reasoning in Large Language Models

A parallel line of work studies additional internal computation or latent reasoning beyond explicit CoT [4]. Pausetoken methods [8] increase internal compute before token prediction, Quiet-STaR [50] trains models to generate useful internal thoughts, and Coconut [12] and CODI [36] move reasoning into continuous hidden space. Fast QuietSTaR [33] further compresses thought traces to reduce inference overhead. In retrieval, LaSER [21] internalizes explicit reasoning into latent space for dense text retrieval. Different from LaSER, which focuses on text-only dense retrieval with a uniform latent reasoning process, PLUME targets universal multimodal embedding, where latent reasoning must operate over heterogeneous inputs and allocate a compact reasoning budget adaptively across diverse query structures.

### 3. Method

#### 3.1. Overview of PLUME

PLUME is a progressive latent reasoning framework for universal multimodal embedding. It replaces explicit reasoning tokens with a short latent rollout, adapts each la-

[Figure 3]

- Figure 3. Overview of PLUME. Starting from a multimodal prefix, PLUME replaces explicit CoT decoding with a compact latent rollout inside the backbone. The bottom panel illustrates the latent rollout process, where the model performs several latent transitions before extracting the final retrieval embedding from the hidden state at <gen>. The top-left panel expands the semantic-anchor-guided transition adapter, which routes each latent step through shared and specialized experts, while the top-right panel shows the progressive explicit-tolatent curriculum that gradually rewrites explicit reasoning segments into latent blocks across training stages. The example in the bottom panel corresponds to an intermediate curriculum stage.

tent transition with a semantic-anchor-guided transition adapter, and transfers explicit reasoning into hidden-space computation through a progressive curriculum. The backbone is fully fine-tuned; the only added components are the lightweight routed adapter and its anchor-conditioned router. Retrieval embeddings are taken directly from normalized hidden states of the backbone, without introducing separate retrieval heads. Figure 3 summarizes the framework.

#### 3.2. Problem Formulation

We consider universal multimodal embedding (UME) [20], where a single model maps heterogeneous inputs into a shared embedding space for retrieval. Given a query q and its corresponding positive target t+, together with a set of negative targets T − = {t−1 ,...,t−N

}, the goal is to maximize the similarity between q and t+ against all negatives. Both queries and targets may be text, images, videos, visual documents, or their combinations.

e

In practice, we sample a mini-batch of N query–target

pairs {(qi,ti)}Ni=1, where (qi,ti) forms the positive pair and all other targets {tj | j ̸= i} serve as in-batch negatives for

qi. We optimize the model with the InfoNCE objective [32]

N

1 N

LNCE =

i=1

exp(sim(qi, ti)/τ)

, (1)

− log

N j=1 exp(sim(qi, tj)/τ)

where sim(·,·) denotes cosine similarity between the normalized embeddings produced by the model, and τ is the temperature hyper-parameter. Unless otherwise specified, we apply this objective bidirectionally in both query-totarget and target-to-query directions. A causal language modeling loss LCE is further applied to the decoded suffix of both the query and its positive target, providing tokenlevel supervision that grounds the generative pathway, as detailed in Sec. 3.5.

The retrieval objective above is standard for UME; PLUME’s contribution lies in how the embedding is formed—through a compact latent reasoning process within the backbone, described next.

#### 3.3. Latent Rollout for Universal Multimodal Embedding

PLUME replaces explicit CoT decoding with a short autoregressive rollout in hidden space—retaining the iterative structure of multi-step reasoning while avoiding the need

to materialize intermediate tokens at inference time. Below we describe the four stages of this process: prefix encoding, latent initialization, iterative rollout, and suffix decoding.

Multimodal prefix encoding. Given an input x, the backbone first processes the multimodal prefix—which may interleave text tokens with image, video, or document features—and then encounters a special <slt> (start-latent-thinking) token that opens a latent block <slt><ct>K <elt>, where <ct> reserves K autoregressive positions and <elt> (end-latent-thinking) marks the end of the latent block. The <ct> tokens serve only as positional placeholders in the serialized sequence; at each latent step the model receives a continuous hidden state rather than a discrete token embedding.

Let h1,...,hL denote the hidden states produced by the backbone on the prefix, where hL corresponds to <slt>. This prefix pass yields two outputs that persist throughout the latent rollout. The first is the cached key–value states C(x) [22, 37], which allow every subsequent latent step to attend back to the full multimodal context via causal attention. The second is a semantic anchor c(x), extracted from the hidden state at a dedicated <anchor> token in the prefix; it provides a fixed summary of the input’s semantic intent for routing (Sec. 3.4). All visual features—image patches, video frames, and document renderings—are injected during this prefix encoding stage only; the latent rollout proceeds purely in hidden space.

Latent state initialization. We initialize the latent state from the hidden state at the <slt> position,

##### z(0) = hL, (2)

since this state already summarizes the multimodal context accumulated before the latent block.

Iterative latent rollout. At each latent step k ∈ {1,...,K}, PLUME performs two operations. First, the previous latent state z(k−1) is refined by the routed adapter (Sec. 3.4), yielding an adapted state z˜(k−1). Second, z˜(k−1) is fed into the backbone as the input embedding at position p<slt> +k, reusing the accumulated KV cache and advancing to the next causal position:

z(k) = Bθ z ˜(k−1), C(k−1), p<slt> + k , k = 1,...,K,

(3) where Bθ denotes one forward pass through the full transformer backbone for a single position. The KV cache grows incrementally: C(0) = C(x) is initialized from the prefix encoding, and each step appends its own key–value pair, yielding C(k) = C(k−1) ∪ {(k(k),v(k))}. Consequently, the output z(k)—the last-layer hidden state at position p<slt>+kattends to the full multimodal prefix and all preceding latent states z(1),...,z(k−1) through standard causal attention over the growing cache. The K outputs z(1),...,z(K) constitute the complete latent reasoning trace.

Each latent step occupies the same causal position where an explicit reasoning token would otherwise be decoded. The backbone sees a continuous vector where it would normally see a token embedding, but the attention mask, positional encoding, and KV-cache mechanics are identical to standard autoregressive generation. PLUME therefore preserves the sequential dependency structure of explicit CoT while replacing discrete token generation with a short sequence of continuous hidden-state transitions.

Suffix decoding and embedding extraction. After the K-step rollout, the latent block is closed by <elt>, and <gen> is placed immediately after <elt>. The final retrieval embedding is extracted from the hidden state at <gen> (Sec. 3.5). In practice, PLUME replaces hundreds of explicit reasoning tokens with as few as K latent steps while retaining full KV-cache compatibility.

#### 3.4. Semantic-Anchor-Guided Transition Adapter

A uniform latent transition cannot accommodate the diversity of UME instances, which vary in modality composition, grounding requirements, and reasoning structure. Inspired by mixture-of-experts (MoE) architectures [7, 35], PLUME inserts a lightweight routed adapter between adjacent latent steps, making each transition input-adaptive without modifying the backbone.

Routing is conditioned on the semantic anchor c(x) (Sec. 3.3), a fixed global signal that stabilizes expert selection against the rapidly evolving latent state.

At latent step k, the router concatenates the additively fused state z(k−1) + c(x) with a learnable step embedding e(k) ∈ RD. The routing weights over Me specialized experts are:

##### π(k) = Softmax Wr z(k−1) + c(x); e(k) + br , (4)

where [·; ·] denotes concatenation, Wr ∈ RM

e×2D and br ∈ RM

e are learnable router parameters. Additive fusion injects the anchor signal without altering the dimensionality of the router input, while the step embedding enables the router to distinguish early and late latent steps.

Each expert Em is a two-layer MLP with an expansion ratio of 2: a linear projection D → 2D followed by GELU activation [14], then a projection back to D. A layer normalization is applied to z(k−1) before it enters any expert, and dropout is applied after the activation for regularization. The adapter contains one shared expert E0 that captures broadly useful transition patterns, plus Me specialized experts {Em}M

m=1 from which the router selects the top Kr. The adapted latent state combines the shared expert output with a weighted mixture of the selected specialized experts

e

via a residual connection: z˜(k−1) = z(k−1)+E0 z ˆ(k−1) +

language modeling loss LCE is computed on the decoded suffix of both the query and its positive target, so that the generative pathway receives token-level supervision on both sides of each training pair. We apply the InfoNCE objective [15, 32] in Sec. 3.2 to both the generative and the anchor embeddings, producing LgenNCE and LancNCE, respectively. The overall objective is

πm(k) Em z ˆ(k−1) ,

m∈TopKr(π(k))

(5) where zˆ(k−1) = LN(z(k−1)) is the layer-normalized input. The resulting z˜(k−1) is passed to the backbone step in Eq. (3). Thus, routing in PLUME acts only on a lightweight latent transition module, rather than turning the backbone itself into a full MoE model.

L = LCE + λgenLgenNCE + λancLancNCE + λbalLbal, (8)

To prevent routing collapse, we impose a balance reg-

where λgen, λanc, and λbal are balancing coefficients.

ularizer. Let π¯m = NK1 Ni=1 Kk=1 πi,m(k) denote the average routing mass for expert m across the mini-batch and

The generative pathway thus receives primary embedding supervision, while the anchor pathway provides auxiliary training signal that improves routing quality and training stability; the anchor embedding itself is discarded at inference, and the retrieval embedding depends exclusively on the latent rollout.

latent steps. The balance loss penalizes deviation from uniform allocation:

Me

2

1 Me

1 Me

, (6)

Lbal =

π ¯m −

m=1

which reaches its minimum of zero when all experts receive equal routing mass. This design makes latent reasoning adaptive without sacrificing efficiency: the evolving latent state captures local step-wise computation, while the fixed semantic anchor provides a stable global cue, so that different multimodal inputs follow different latent reasoning paths under the same backbone and rollout budget.

#### 3.5.EmbeddingFormationandTrainingObjectives

The final retrieval embedding is taken from the generative pathway, because it reflects the full latent-to-suffix computation induced by reasoning.

Formally, for an input x, we define the final retrieval embedding as

egen(x) = Norm(h<gen>), (7)

where h<gen> denotes the last-layer hidden state at the <gen> position and Norm(·) denotes ℓ2 normalization. Additionally, an auxiliary anchor embedding eanc(x) = Norm(hanchor) is derived from the semantic anchor introduced in Sec. 3.3. During training, eanc receives its own contrastive supervision (LancNCE), which serves two purposes: it encourages the anchor to encode a semantically meaningful global summary of the input, thereby providing a higher-quality routing signal for the MoE adapter, and it supplies an additional gradient pathway that stabilizes early-stage training when the latent rollout has not yet converged. Crucially, eanc is discarded at inference: retrieval always relies exclusively on egen. Because egen is extracted after the full latent rollout and suffix decoding, it cannot be computed without a functioning latent trajectory, which prevents the model from taking a shortcut through the anchor embedding alone.

The full training objective combines suffix generation, retrieval alignment, and routing regularization. The causal

#### 3.6. Progressive Explicit-to-Latent Curriculum

A direct transition from explicit CoT supervision to latentonly execution is unstable, because semantic grounding does not transfer reliably into hidden-space rollout. Without an explicit scaffold, latent states may collapse into degenerate shortcuts instead of preserving the multi-step structure learned from verbalized reasoning. PLUME therefore adopts a progressive explicit-to-latent curriculum [2], which uses explicit CoT only as a transient training scaffold.

For each training instance, we split the explicit rationale into sentence-level segments and gradually replace them, from left to right, with a latent block of <ct> positions. At early stages, most reasoning steps remain explicit and are teacher-forced, which preserves semantic grounding. As training proceeds, a larger prefix of the rationale is absorbed into the latent block, while the remaining unreplaced steps are kept as supervised suffix tokens after <elt>. The model thus learns to continue explicit reasoning from partially latent prefixes before internalizing the full reasoning process.

The latent block itself receives no token-level supervision. Supervision is applied only to the remaining explicit rationale and the downstream answer. In the final stage, both the explicit rationale and the answer span are removed, so that the latent rollout connects directly to <gen>, matching the inference-time execution pattern.

We allocate more training to the final fully latent stage, while earlier stages serve mainly to stabilize the transfer from verbalized to latent reasoning. This curriculum provides a structured bridge from explicit CoT to compact latent rollout, allowing PLUME to retain the benefits of structured intermediate reasoning without preserving explicit CoT at inference time.

- Table 1. Main comparison on MMEB-v2. We compare PLUME with early UME baselines and reasoning-enhanced UME methods. All methods share the same Qwen2-VL-2B backbone. Best and second-best results are highlighted in bold and underline.

Image Video VisDoc

Model Venue

All CLS QA RET GD Overall CLS QA RET MRET Overall VDRv1 VDRv2 VR OOD Overall

# of Datasets – 10 10 12 4 36 5 5 5 3 18 10 4 6 4 24 78 Early UME Baselines LamRA CVPR’25 59.2 26.5 70.0 62.7 54.1 39.3 42.6 24.3 34.6 35.2 22.0 11.5 37.4 21.0 23.9 40.4 VLM2Vec ICLR’25 58.7 49.3 65.0 72.9 59.7 33.4 30.5 20.6 33.0 29.0 49.8 13.5 51.8 33.5 41.6 47.0 GME CVPR’25 54.4 29.9 66.9 55.5 51.9 34.9 42.0 25.6 32.4 33.9 86.1 54.0 82.5 43.1 72.7 54.1 VLM2Vec-V2 TMLR’26 62.0 56.3 69.5 77.3 64.9 39.3 34.3 28.8 38.5 34.9 75.5 44.9 79.4 39.4 65.4 58.0 DUME ICLR’26 59.3 55.0 66.3 78.0 62.5 37.7 46.6 17.1 30.0 33.2 67.6 43.3 47.1 33.8 52.8 52.7 Reasoning UME UME-R1 ICLR’26 64.8 62.8 67.6 77.2 66.6 44.3 51.2 32.9 39.7 42.2 72.4 46.2 79.2 37.2 63.9 60.1 PLUME Ours 66.5 59.2 67.6 79.7 66.3 45.0 52.3 33.5 46.7 44.1 72.1 49.8 78.1 57.4 67.5 61.6

### 4. Experiments

#### 4.1. Experimental Setup

Evaluation Metrics. We evaluate PLUME on MMEBv2 [31], a comprehensive benchmark for universal multimodal embedding. MMEB-v2 extends MMEB-V1 [20] by introducing video and visual-document retrieval scenarios, resulting in a benchmark with 9 meta-tasks and 78 test tasks in total. The benchmark covers a broad spectrum of vision-language retrieval settings, including image-level retrieval, video understanding and retrieval, visual-document retrieval, and reasoning-intensive multimodal matching. Following prior work, we report Hit@1 for image and video tasks, and NDCG@5 [16] for visualdocument retrieval tasks.

Baselines. We compare PLUME against two groups of representative baselines. The first group consists of early UME methods that form embeddings without explicit reasoning, including LamRA [30], VLM2Vec [20], GME [54], VLM2Vec-V2 [31], and DUME [55]. The second group consists of reasoning-enhanced UME methods, represented by UME-R1 [24], which generates explicit CoT rationales before embedding extraction. All methods share the same Qwen2-VL-2B backbone [40], ensuring that differences in performance reflect the reasoning mechanism rather than backbone capacity or data scale. We note that concurrent work TTE [5] employs a separate Qwen2.5-VL-72B [1] model as a dedicated reasoning module to produce CoT rationales before the 2B encoder extracts the final embedding; because this introduces additional large-scale model capacity that is unavailable to the other methods, we do not include it in the main comparison. We similarly exclude methods built on different backbone families or training corpora.

Implementation Details. Our model is built on the Qwen2VL-2B backbone [40]. For training data, we use the same

supervised fine-tuning corpus as UME-R1 [24], since it provides explicit multimodal reasoning traces that are well suited for progressive explicit-to-latent transfer. The InfoNCE temperature is set to 0.02, the global batch size is 1024, and the learning rate is 5 × 10−5. The latent rollout length is set to K = 8. Training lasts for 5 epochs in total. The curriculum begins with an initial fully explicitCoT warm-up stage (Stage 0) trained for 3 epochs, followed by four progressive latent curriculum stages (Stages 1–4). Among them, the intermediate transition stages (Stages 1– 3) are completed within 1 epoch in total, and the final fully latent stage (Stage 4) is trained for 1 additional epoch.

#### 4.2. Main Comparison on MMEB-v2

Table 1 presents the comparison across MMEB-v2’s three modality groups. Methods are organized into early UME baselines (single-pass encoding) and reasoning UME.

Under a controlled setting with the same backbone and training data, PLUME surpasses UME-R1 [24] by 1.5 points overall (61.6 vs. 60.1) while requiring only 8 latent steps instead of hundreds of generated reasoning tokens. Compared with early baselines, PLUME surpasses the strongest single-pass method VLM2Vec-V2 [31] by 3.6 points overall, with particularly large gains on Video (+9.2) where multi-step temporal reasoning is most beneficial.

Across modality groups, PLUME achieves 66.3 on Image (vs. UME-R1’s 66.6), 44.1 on Video (vs. 42.2), and 67.5 on VisDoc (vs. 63.9). While PLUME trails UMER1 slightly on Image (−0.3), it delivers clear gains on Video (+1.9) and VisDoc (+3.6). The Video advantage is consistent with our motivation that temporal dynamics and cross-frame relationships are difficult to linearize into discrete tokens, and that maintaining continuous state across reasoning steps preserves richer temporal semantics. This advantage is most visible on Video Multi-modal Retrieval (PLUME 46.7 vs. UME-R1 39.7, +7.0), where multi-step

Table 2. Inference efficiency on a single H20 GPU

Metric PLUME UME-R1 VLM2Vec-V2 Reasoning tokens/steps 8 403 0 Latency (ms/sample) 298±12 9023±187 156±8 Throughput (samples/s) 3.3±0.1 0.11±0.01 6.4±0.3 Speedup vs. UME-R1 30.3× 1.0× – Overhead vs. VLM2Vec-V2 1.9× – 1.0×

cross-modal alignment benefits from uninterrupted continuous reasoning. PLUME also sets the best score on Image Grounding (79.7) and VisDoc OOD (57.4), both of which involve compositional spatial reasoning or out-ofdistribution generalization that benefit from continuousstate computation. Figure 4 visualizes these per-task comparisons, confirming that PLUME consistently outperforms UME-R1 and single-pass baselines across most sub-tasks.

#### 4.3. Efficiency and Accuracy-Efficiency Tradeoff

- Table 2 provides a quantitative breakdown of inference costs. All measurements are conducted on a single NVIDIA H20 GPU. For each method, we randomly sample 500 inputs per modality as one evaluation set, preceded by 20 warm-up iterations, and repeat with 5 independently drawn evaluation sets. The table reports the mean across the 5 runs; ± denotes the standard deviation.

PLUME compresses reasoning from an average of 403 generated tokens (UME-R1 [24]) to 8 latent steps, reducing per-sample latency from 9023ms to 298ms, a 30.3× speedup. Compared with the single-pass baseline VLM2Vec-V2 [31] (156ms), PLUME adds less than 150ms of overhead yet improves overall accuracy by 2.1 points, confirming that a small latent computation budget delivers substantial reasoning gains at modest cost.

Figure 1 visualizes the accuracy–efficiency tradeoff. PLUME occupies a favorable position on the Pareto frontier: at K = 6 it already surpasses UME-R1’s accuracy while running over 30x faster, and at K = 4 it still outperforms all single-pass baselines with throughput comparable to VLM2Vec-V2. This confirms that latent reasoning effectively reconciles retrieval quality and inference efficiency.

#### 4.4. Ablation Studies

We conduct ablation studies to verify the contribution of each core component. Unless otherwise stated, all ablations use the same training recipe as the full model.

Component ablation. Table 3 shows that every component contributes meaningfully. Removing the progressive curriculum causes the largest overall drop (−6.8), with Video suffering the most (−7.6). In this ablation, the model first completes Stage 0 training with full explicit CoT and then directly trains with the Stage 4 setting (fully latent, no explicit tokens), bypassing the gradual transition of Stages 1–

[Figure 4]

Figure 4. Per task performance comparison on MMEB-v2.

Table 3. Ablation on the core components of PLUME.

Configuration Image Video VisDoc All

Full PLUME 66.3 44.1 67.5 61.6 w/o Latent Transition 63.6 41.0 64.8 58.8 w/o MoE (single MLP) 64.2 41.8 64.4 59.2 w/o Semantic Anchor 65.4 42.3 66.1 60.1 w/o Curriculum 60.2 36.5 60.2 54.8

Table 4. Effect of the latent steps K on accuracy and latency.

K Image Video VisDoc All Latency (ms)

4 64.3 43.3 65.7 59.9 232 6 65.9 43.6 66.7 61.1 268 8 66.3 44.1 67.5 61.6 300

3. The large degradation confirms that an abrupt switch from explicit to latent reasoning leads to training instability, and the progressive schedule is necessary for stable knowledge transfer.

Removing the latent transition entirely (reading the embedding from the last prefix token) reduces accuracy by 2.8 overall, with consistent losses across all modalities, validating that iterative hidden-space computation provides genuine reasoning benefit beyond additional parameters. Replacing the MoE adapter with a single shared MLP costs 2.4 points, and the degradation is most pronounced on VisDoc (−3.1), where document understanding benefits from specialized expert pathways. Removing the semantic anchor from the router hurts Video most (−1.8), indicating that global input context is important for routing temporal reasoning.

Latent steps K. Table 4 shows that accuracy improves steadily from K = 4 to K = 8, gaining 1.7 points overall. The gain from 6 to 8 (+0.5) is smaller than from 4

Table 5. Ablation on the transition adapter design.

Configuration Image Video VisDoc All

Default (Me=4, Kr =2, shared) 66.3 44.1 67.5 61.6 w/o Shared Expert 65.4 42.5 66.1 60.3 Top-1 expert (instead of Top-2) 65.8 43.0 66.8 60.8 Router: w/o c(x) 65.4 42.3 66.1 60.1 Router: w/o e(k) 65.7 41.9 66.2 60.4

[Figure 5]

Figure 5. Activation preferences of specialized experts across image and video retrieval sub-tasks.

to 6 (+1.2), exhibiting diminishing returns. Latency grows roughly linearly (232ms → 300ms), so K = 8 offers the best absolute accuracy while K = 6 provides a favorable accuracy–speed balance (see also Figure 1).

Transition adapter design. Table 5 validates individual design choices in the routed adapter. Removing the shared expert drops overall accuracy by 1.3, confirming that a modality-agnostic baseline pathway complements the specialized experts. Top-2 routing outperforms top-1 by 0.8, indicating that combining two experts captures richer transition patterns. Among routing inputs, removing the semantic anchor c(x) hurts more (−1.5) than removing the step embedding e(k) (−1.2), yet both contribute: c(x) provides global input context for expert selection, while e(k) encodes positional progression within the rollout.

#### 4.5. Diagnostic Analysis

We visualize the routing behavior of the four specialized experts to examine whether the routed adapter learns meaningful specialization. The shared expert is always active for every input and is therefore omitted from the heatmap; the figure shows only the top-Kr selected specialized experts.

Task-level routing. Figure 5 provides a fine-grained view across image and video sub-tasks. Expert 2 shows the highest activation on video classification (V-CLS: 75.2%) and video multi-modal retrieval (V-MRET: 70.6%), and remains elevated on image classification (I-CLS: 65.1%) and video retrieval (V-ReT: 61.8%). Expert 1 is preferentially activated on question-answering tasks: V-QA (63.8%) and I-QA (59.9%), consistent with an affinity for knowledge-

[Figure 6]

Figure 6. Average cosine similarity between intermediate states and the positive target over 200 samples, reported separately on image and video retrieval. PLUME shows a smoother trajectory with consistently smaller variance than UME-R1 across reasoning steps.

intensive reasoning. Expert 0 peaks on image grounding (I-GD: 62.9%) and image retrieval (I-RET: 60.4%), while being almost never selected for video classification (VCLS: 22.9%). Expert 3 remains a low-activation generalist without strong task-level peaks. These activation patterns emerge purely from the routing objective without any explicit task labels, confirming that the semantic-anchorguided router adapts latent computation to the structural demands of each input.

Latent trajectory visualization. Figure 6 compares the average cosine similarity between intermediate states and the positive target over 200 samples, reported for image and video retrieval. We use this metric as a diagnostic signal for trajectory stability rather than as a requirement that every intermediate step must monotonically approach the positive target. Across both subsets, PLUME exhibits a smoother trajectory with consistently smaller variance than UMER1, especially after the early reasoning steps. On image retrieval, the two methods are close at the beginning, but UME-R1 shows a larger mid-trajectory drop and substantially broader dispersion, whereas PLUME remains more stable throughout the rollout. On video retrieval, the advantage is clearer: PLUME maintains stronger alignment with the positive target across most steps, while UME-R1 stays lower and fluctuates more. These trends suggest that latent reasoning provides a more consistent intermediate computation path for retrieval, whereas explicit CoT produces more variable hidden-state trajectories under discrete token generation.

Limitations. Despite surpassing UME-R1 overall, PLUME shows a notable gap on the Image QA subset. This weakness is not uniform across all QA tasks: the gap is small on ScienceQA and WebQA, but much larger on text-rich or knowledge-intensive benchmarks such as ChartQA, InfographicsVQA and OK-VQA. We hypothesize that these tasks rely more heavily on preserving fine grained textual detail and explicit intermediate semantic organization, whereas PLUME compresses reasoning into a short latent rollout optimized for retrieval-oriented representation for-

mation. Besides, while the routed adapter develops differentiated activation patterns consistent with our design hypothesis, formal interpretability guarantees for continuous latent trajectories remain an open problem.

### 5. Conclusion

We introduced PLUME, a latent reasoning framework for universal multimodal embedding that replaces explicit chain-of-thought generation with a short hidden-space rollout. By combining latent multi-step reasoning, anchorguided routed adaptation, and a progressive explicit-tolatent curriculum, PLUME achieves better transfer of reasoning into compact embeddings. On the 78-task MMEBv2 benchmark, it surpasses UME-R1 trained on the same data, reduces reasoning from hundreds of tokens to fewer than ten latent steps, and delivers over 30× faster inference.

### References

- [1] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025. 7
- [2] Yoshua Bengio, J´erˆome Louradour, Ronan Collobert, and Jason Weston. Curriculum learning. In Proceedings of the 26th International Conference on Machine Learning, pages 41–48. ACM, 2009. 6
- [3] Haonan Chen, Hong Liu, Yuping Luo, Liang Wang, Nan Yang, Furu Wei, and Zhicheng Dou. Moca: Modality-aware continual pre-training makes better bidirectional multimodal embeddings. arXiv preprint arXiv:2506.23115, 2025. 3
- [4] Xinghao Chen, Anhao Zhao, Heming Xia, Xuan Lu, Hanlin Wang, Yanjun Chen, Wei Zhang, Jian Wang, Wenjie Li, and Xiaoyu Shen. Reasoning beyond language: A comprehensive survey on latent chain-of-thought reasoning. CoRR, abs/2505.16782, 2025. 2, 3
- [5] Xuanming Cui, Jianpeng Cheng, Hong-you Chen, Satya Narayan Shukla, Abhijeet Awasthi, Xichen Pan, Chaitanya Ahuja, Shlok Kumar Mishra, Yonghuan Yang, Jun Xiao, et al. Think then embed: Generative context improves multimodal embedding. arXiv preprint arXiv:2510.05014, 2025. 2, 3, 7
- [6] Manuel Faysse, Hugues Sibille, Tony Wu, Bilel Omrani, Gautier Viaud, C´eline Hudelot, and Pierre Colombo. Colpali: Efficient document retrieval with vision language models. arXiv preprint arXiv:2407.01449, 2024. 3
- [7] William Fedus, Barret Zoph, and Noam Shazeer. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. ArXiv, abs/2101.03961, 2021. 5
- [8] Sachin Goyal, Ziwei Ji, Ankit Singh Rawat, Aditya Menon, Sanjiv Kumar, and Vaishnavh Nagarajan. Think before you speak: Training language models with pause tokens. In International Conference on Learning Representations (ICLR),

2024. 3

- [9] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha

- Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024. 1
- [10] Tiancheng Gu, Kaicheng Yang, Ziyong Feng, Xingjun Wang, Yanzhao Zhang, Dingkun Long, Yingda Chen, Weidong Cai, and Jiankang Deng. Breaking the modality barrier: Universal embedding learning with multimodal llms. arXiv preprint arXiv:2504.17432, 2025. 3
- [11] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025. 3
- [12] Shibo Hao, Sainbayar Sukhbaatar, DiJia Su, Xian Li, Zhiting Hu, Jason Weston, and Yuandong Tian. Training large language models to reason in a continuous latent space. CoRR, abs/2412.06769, 2024. 2, 3
- [13] Xiangzhao Hao, Shijie Wang, Tianyu Yang, Tianyue Wang, Haiyun Guo, and Jinqiao Wang. Trace: Task-adaptive reasoning and representation learning for universal multimodal retrieval, 2026. 2, 3
- [14] Dan Hendrycks and Kevin Gimpel. Gaussian error linear units (gelus). arXiv: Learning, 2016. 5
- [15] Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebastian Riedel, Piotr Bojanowski, Armand Joulin, and Edouard Grave. Unsupervised dense information retrieval with contrastive learning. Trans. Mach. Learn. Res., 2022, 2022. 6
- [16] Kalervo J¨arvelin and Jaana Kek¨al¨ainen. Cumulated gainbased evaluation of ir techniques. ACM Transactions on Information Systems (TOIS), 20(4):422–446, 2002. 7
- [17] Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc Le, Yun-Hsuan Sung, Zhen Li, and Tom Duerig. Scaling up visual and vision-language representation learning with noisy text supervision. In International conference on machine learning, pages 4904–4916. PMLR,

2021. 3

- [18] Haonan Jiang, Yuji Wang, Yongjie Zhu, Xin Lu, Wenyu Qin, Meng Wang, Pengfei Wan, and Yansong Tang. Embedrl: Reinforcement learning for reasoning-driven multimodal embeddings. arXiv preprint arXiv:2602.13823, 2026. 3
- [19] Ting Jiang, Minghui Song, Zihan Zhang, Haizhen Huang, Weiwei Deng, Feng Sun, Qi Zhang, Deqing Wang, and Fuzhen Zhuang. E5-v: Universal embeddings with multimodal large language models. arXiv preprint arXiv:2407.12580, 2024. 2, 3
- [20] Ziyan Jiang, Rui Meng, Xinyi Yang, Semih Yavuz, Yingbo Zhou, and Wenhu Chen. Vlm2vec: Training vision-language models for massive multimodal embedding tasks. arXiv preprint arXiv:2410.05160, 2024. 1, 2, 3, 4, 7
- [21] Jiajie Jin, Yanzhao Zhang, Mingxin Li, Dingkun Long, Pengjun Xie, Yutao Zhu, and Zhicheng Dou. Laser: Internalizing explicit reasoning into latent space for dense retrieval,

2026. 3

- [22] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Pro-

- ceedings of the 29th Symposium on Operating Systems Principles, SOSP 2023, Koblenz, Germany, October 23-26, 2023, pages 611–626. ACM, 2023. 5
- [23] Zhibin Lan, Liqiang Niu, Fandong Meng, Jie Zhou, and Jinsong Su. Llave: Large language and vision embedding models with hardness-weighted contrastive learning. arXiv preprint arXiv:2503.04812, 2025. 3
- [24] Zhibin Lan, Liqiang Niu, Fandong Meng, Jie Zhou, and Jinsong Su. Ume-r1: Exploring reasoning-driven generative multimodal embeddings. arXiv preprint arXiv:2511.00405,

2025. 2, 3, 7, 8

- [25] Chankyu Lee, Rajarshi Roy, Mengyao Xu, Jonathan Raiman, Mohammad Shoeybi, Bryan Catanzaro, and Wei Ping. Nvembed: Improved techniques for training llms as generalist embedding models. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net, 2025. 3
- [26] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, et al. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024. 1
- [27] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In International conference on machine learning, pages 19730–

19742. PMLR, 2023. 3

- [28] Sheng-Chieh Lin, Chankyu Lee, Mohammad Shoeybi, Jimmy Lin, Bryan Catanzaro, and Wei Ping. Mm-embed: Universal multimodal retrieval with multimodal llms. arXiv preprint arXiv:2411.02571, 2024. 2, 3
- [29] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36:34892–34916, 2023. 1
- [30] Yikun Liu, Yajie Zhang, Jiayin Cai, Xiaolong Jiang, Yao Hu, Jiangchao Yao, Yanfeng Wang, and Weidi Xie. Lamra: Large multimodal model as your advanced retrieval assistant. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 4015–4025, 2025. 3, 7
- [31] Rui Meng, Ziyan Jiang, Ye Liu, Mingyi Su, Xinyi Yang, Yuepeng Fu, Can Qin, Zeyuan Chen, Ran Xu, Caiming Xiong, et al. Vlm2vec-v2: Advancing multimodal embedding for videos, images, and visual documents. arXiv preprint arXiv:2507.04590, 2025. 1, 3, 7, 8
- [32] Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748, 2018. 3, 4, 6
- [33] Jacob Pfau, William Merrill, and Samuel R. Bowman. Let’s think dot by dot: Hidden computation in transformer language models. ArXiv, abs/2404.15758, 2024. 3
- [34] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021. 3
- [35] Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. Outra-

- geously large neural networks: The sparsely-gated mixtureof-experts layer, 2017. 5
- [36] Zhenyi Shen, Hanqi Yan, Linhai Zhang, Zhanghao Hu, Yali Du, and Yulan He. CODI: compressing chain-ofthought into continuous space via self-distillation. CoRR, abs/2502.21074, 2025. 2, 3
- [37] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Neural Information Processing Systems, 2017. 5
- [38] Liang Wang, Nan Yang, Xiaolong Huang, Binxing Jiao, Linjun Yang, Daxin Jiang, Rangan Majumder, and Furu Wei. Text embeddings by weakly-supervised contrastive pretraining. CoRR, abs/2212.03533, 2022. 3
- [39] Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang, Rangan Majumder, and Furu Wei. Improving text embeddings with large language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2024, Bangkok, Thailand, August 11-16, 2024, pages 11897–11916. Association for Computational Linguistics, 2024. 3
- [40] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024. 1, 7
- [41] Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. arXiv preprint arXiv:2203.11171,

2022. 2, 3

- [42] Cong Wei, Yang Chen, Haonan Chen, Hexiang Hu, Ge Zhang, Jie Fu, Alan Ritter, and Wenhu Chen. Uniir: Training and benchmarking universal multimodal information retrievers. In European Conference on Computer Vision, pages 387–404. Springer, 2024. 3
- [43] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022. 2, 3
- [44] Shitao Xiao, Zheng Liu, Peitian Zhang, Niklas Muennighoff, Defu Lian, and Jian-Yun Nie. C-pack: Packed resources for general chinese embeddings. In Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR 2024, Washington DC, USA, July 14-18, 2024, pages 641–649. ACM, 2024. 3
- [45] Guowei Xu, Peng Jin, Ziang Wu, Hao Li, Yibing Song, Lichao Sun, and Li Yuan. Llava-cot: Let vision language models reason step-by-step. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2087– 2098, 2025. 3
- [46] Tianyu Yang, ChenWei He, Xiangzhao Hao, Tianyue Wang, Jiarui Guo, Haiyun Guo, Leigang Qu, Jinqiao Wang, and Tat-Seng Chua. Recall: Recalibrating capability degradation for mllm-based composed image retrieval. arXiv preprint arXiv:2602.01639, 2026. 2

- [47] Tianyu Yang, Chenwei He, Xiangzhao Hao, Tianyue Wang, Jiarui Guo, Haiyun Guo, Leigang Qu, Jinqiao Wang, and TatSeng Chua. Recall: Recalibrating capability degradation for mllm-based composed image retrieval, 2026. 3
- [48] Hao Yu, Zhuokai Zhao, Shen Yan, Lukasz Korycki, Jianyu Wang, Baosheng He, Jiayi Liu, Lizhu Zhang, Xiangjun Fan, and Hanchao Yu. Cafe: Unifying representation and generation with contrastive-autoregressive finetuning. arXiv preprint arXiv:2503.19900, 2025. 3
- [49] Shi Yu, Chaoyue Tang, Bokai Xu, Junbo Cui, Junhao Ran, Yukun Yan, Zhenghao Liu, Shuo Wang, Xu Han, Zhiyuan Liu, et al. Visrag: Vision-based retrieval-augmented generation on multi-modality documents. arXiv preprint arXiv:2410.10594, 2024. 3
- [50] E. Zelikman, Georges Harik, Yijia Shao, Varuna Jayasiri, Nick Haber, and Noah D. Goodman. Quiet-star: Language models can teach themselves to think before speaking. ArXiv, abs/2403.09629, 2024. 3
- [51] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF international conference on computer vision, pages 11975–11986, 2023. 3
- [52] Kai Zhang, Yi Luan, Hexiang Hu, Kenton Lee, Siyuan Qiao, Wenhu Chen, Yu Su, and Ming-Wei Chang. Magiclens: Self-supervised image retrieval with open-ended instructions. arXiv preprint arXiv:2403.19651, 2024. 3
- [53] Xuan Zhang, Chao Du, Tianyu Pang, Qian Liu, Wei Gao, and Min Lin. Chain of preference optimization: Improving chain-of-thought reasoning in llms. Advances in Neural Information Processing Systems, 37:333–356, 2024. 3
- [54] Xin Zhang, Yanzhao Zhang, Wen Xie, Mingxin Li, Ziqi Dai, Dingkun Long, Pengjun Xie, Meishan Zhang, Wenjie Li, and Min Zhang. Gme: Improving universal multimodal retrieval by multimodal llms. arXiv preprint arXiv:2412.16855, 2024. 1, 2, 3, 7
- [55] Xin Zhang, Yanzhao Zhang, Wen Xie, Mingxin Li, Ziqi Dai, Dingkun Long, Pengjun Xie, Meishan Zhang, Wenjie Li, and Min Zhang. Bridging modalities: Improving universal multimodal retrieval by multimodal large language models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 9274–9285, 2025. 3, 7
- [56] Junjie Zhou, Zheng Liu, Ze Liu, Shitao Xiao, Yueze Wang, Bo Zhao, Chen Jason Zhang, Defu Lian, and Yongping Xiong. Megapairs: Massive data synthesis for universal multimodal retrieval. arXiv preprint arXiv:2412.14475, 2024. 3
- [57] Junjie Zhou, Yongping Xiong, Zheng Liu, Ze Liu, Shitao Xiao, Yueze Wang, Bo Zhao, Chen Jason Zhang, and Defu Lian. Megapairs: Massive data synthesis for universal multimodal retrieval. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 19076–19095, 2025. 3

### A. Curriculum Ablation Details

We provide additional ablations on the curriculum design used in PLUME. Specifically, we vary the number of curriculum stages while keeping the backbone, training data, total number of training epochs, and all other optimization settings fixed. Unless otherwise specified, all variants use the same final latent rollout length and differ only in how progressively the explicit reasoning segments are rewritten into latent computation.

Table 6 compares curriculum schedules with 2, 4, 6, and 8 stages. Overall, we observe that curriculum granularity has a non-monotonic effect on performance. Using only 2 stages yields the weakest result, suggesting that an overly abrupt transition from explicit reasoning supervision to latent computation is suboptimal. With too few stages, the model must absorb a large distribution shift at each transition, which makes it harder to preserve stable representation quality during the handoff from token-level reasoning traces to latent reasoning steps.

Increasing the number of stages to 4 substantially improves performance across all three domains, indicating that a moderately progressive curriculum better supports optimization. A finer curriculum allows the model to adapt more smoothly as explicit reasoning is gradually replaced by latent rollout, reducing optimization difficulty and stabilizing embedding formation. This effect is particularly visible in the visual document setting, where the gap between 2 and 4 stages is the largest.

Further increasing the number of stages beyond 4 does not bring additional gains. Although 6 stages remains competitive, its overall result is slightly below the 4-stage setting, and 8 stages degrades more clearly. Since the total training budget is fixed for all variants, increasing the number of stages necessarily reduces the effective training time allocated to each individual stage. As a result, later stages may be under-optimized before the curriculum advances again, preventing the model from fully adapting to each intermediate supervision regime. In addition, an overly finegrained curriculum weakens the distinction between adjacent stages, which may reduce the practical benefit of each transition while introducing extra scheduling complexity.

Taken together, these results suggest that the curriculum should be progressive enough to avoid a sharp explicit-tolatent shift, but not so fragmented that each stage becomes

- too short to be fully exploited. We therefore adopt 4 stages as the default setting, as it provides the best overall trade-off between retrieval accuracy and curriculum efficiency in our experiments.

Implementation details. For all curriculum variants, we preserve the same overall training setup and only modify the number of transition stages. The rewritten portion of the reasoning sequence increases progressively across stages, while the latent rollout budget is scaled accordingly

Table 6. Ablation on the number of curriculum stages. All models are trained with the same data, backbone, and total training recipe unless otherwise specified. The best result is in bold and the second best is underlined.

# Stages Image Video VisDoc Avg. 2 64.0 42.1 64.3 59.0 4 66.3 44.1 67.5 61.6 6 66.4 43.7 67.3 61.4 8 65.6 43.6 66.7 60.9

to match the increasing degree of latent computation. This design isolates the effect of curriculum granularity without confounding it with changes in model capacity or total optimization budget.

### B. Training Time and Computational Cost Analysis

We briefly report the training cost of PLUME under different latent rollout lengths. Since all variants share the same backbone and training recipe, the main difference in computational cost comes from the number of latent reasoning steps. In our experiments, training PLUME with 4 latent steps requires approximately 2562 H20 GPU hours. The cost increases to about 2838 H20 GPU hours for 6 latent steps and 3119 H20 GPU hours for 8 latent steps. This increase is expected, as longer latent rollouts introduce additional computation during training. These results show that the training cost grows steadily with the latent rollout length, without introducing unexpected overhead beyond the added latent computation itself.

### C. Failure Case Analysis

We provide representative failure cases to better understand the remaining limitations of PLUME. In general, PLUME performs strongly on retrieval tasks that benefit from compact multi-step reasoning, but it can still struggle in cases requiring fine-grained textual preservation, dense document understanding, or externally grounded factual knowledge.

- Figure 7 presents representative failure cases from

ChartQA. We focus on this benchmark because it requires fine-grained numerical reading, localized text grounding, and compositional reasoning over structured visual content. In both examples, the correct answer remains semantically close to the retrieved candidates, but PLUME fails to rank it first. This pattern suggests that the main limitation is not a complete loss of task intent, but rather insufficient preservation of small numerical distinctions and multi-step relational structure after explicit reasoning is compressed into a compact latent rollout.

- Figure 8 presents a representative failure case from Info-

graphicsVQA. Compared with natural image question answering, infographic style inputs require the model to aggregate fine-grained textual cues distributed across multiple regions under a complex visual layout. In this example, PLUME correctly identifies the relevant semantic neighborhood, but ranks broader or compositionally related answers ahead of the exact target. This suggests that the main challenge lies not in coarse semantic localization, but in maintaining precise answer granularity when dense textual structure must be compressed into a compact latent reasoning trajectory. In many instances, PLUME still places semantically related candidates near the top, indicating that the latent rollout preserves a substantial portion of the relevant query intent even when the exact target is not ranked first.

[Figure 7]

[Figure 8]

- Figure 7. Failure cases on ChartQA. Each example shows the input chart and query at the top, and the top retrieved predictions from PLUME at the bottom. The green box marks the ground-truth target, while the preceding orange boxes denote higher-ranked distractors. In Case 1, PLUME fails on a count-over-threshold query, ranking 4 above the correct answer 5. In Case 2, PLUME struggles with multi-step numerical aggregation, ranking several nearby values ahead of the correct average 21.5. These examples suggest that although PLUME usually preserves the coarse semantic intent of the query, compressing reasoning into a short latent rollout can still weaken the preservation of fine-grained numerical relations and localized textual details required by chart-intensive question answering.

[Figure 9]

[Figure 10]

- Figure 8. Failure case on InfographicsVQA. The figure shows an infographic-style input, the query, the ground-truth target, and the

- top retrieved predictions from PLUME. The green box marks the correct target, while the orange boxes denote higher-ranked or nearby distractors. In this example, the target answer is facebook, but PLUME ranks a broader candidate, twitter, facebook, pinterest, above the exact target. It also retrieves semantically related but overly coarse or mismatched variants such as twitter, facebook and facebook ads. This case suggests that on text-dense infographic inputs, PLUME can capture the relevant semantic region but may fail to preserve the precise answer granularity needed for exact retrieval.

More broadly, the weakness on QA-style subsets should also be interpreted in light of the benchmark formulation itself. In MMEB, visual question answering is cast as an embedding retrieval problem, where the query consists of an instruction, an image, and a question, and the model must retrieve the correct answer from a fixed candidate pool. Prior work has noted that this type of evaluation pattern is not fully aligned with realistic question answering settings, where relevant evidence retrieval and answer generation are more tightly coupled, and where bounded candidate pools can simplify or distort the underlying retrieval–reasoning difficulty. We therefore view the larger gaps on text-rich QA benchmarks such as ChartQA and InfographicsVQA as reflecting both a model-side challenge in preserving finegrained textual and numerical structure, and a benchmarkside mismatch between QA and retrieval formulations.

These cases are consistent with the discussion in Sec. 4.5: while latent reasoning is more efficient and often sufficient for retrieval-oriented representation formation, explicit verbal reasoning may still preserve certain types of fine-grained semantic structure better in some challenging settings.

### D. Additional Routing Visualization

To further analyze the behavior of the semantic-anchorguided routed adapter, we visualize the average activation rate of each expert across different input modalities. Rather than focusing on individual examples, this aggregated view provides a coarse but informative picture of how the routed adapter allocates computation under different modality conditions.

Figure 9 summarizes expert activation patterns for six input types: text (T), image (I), video (V), document (D), text-image (TI), and text-video (TV). Several non-uniform trends can be observed. Expert 2 shows consistently high activation across almost all modalities, with especially strong responses on text and video inputs, suggesting that it serves as a broadly useful expert shared across diverse reasoning situations. In contrast, Expert 1 exhibits clearer specialization, reaching its highest activation on document and text-video inputs, which indicates a stronger preference for inputs with richer structured or cross-modal information. Expert 0 is comparatively more active on image and text-image inputs, while Expert 3 is most activated on text inputs and remains less preferred for document and mixedmodality cases.

These patterns suggest that the routed adapter does not collapse to a uniform expert usage distribution. Instead, different experts develop differentiated modality preferences, while still maintaining partial overlap that allows computation to be shared across related settings. This behavior is consistent with our design motivation: multimodal retrieval inputs are heterogeneous in structure, and thus benefit from

[Figure 11]

Figure 9. Average activation rate of each expert across input modalities. T, I, V, D, TI, and TV denote text, image, video, document, text-image, and text-video inputs, respectively. Darker cells indicate higher expert activation rates.

adaptive latent transition pathways rather than a single fixed computation pattern.

Overall, the visualization shows that the routed adapter learns structured but non-uniform expert allocation patterns across modalities, supporting the claim that different multimodal inputs benefit from differentiated latent computation.

### E. Full MMEB-V2 Results

We report the full MMEB-V2 evaluation results in this section. Following prior work, we provide per-task scores to give a more complete picture beyond the averaged categorylevel metrics reported in the main paper.

Table 7, 8 lists the detailed performance of all compared methods on MMEB-V2. These results show that the gains of PLUME are broadly distributed across task families, rather than arising from only a small subset of benchmarks.

### F. Additional Baseline Comparisons

In this section, we provide additional qualitative comparisons between explicit CoT and PLUME on representative retrieval cases. As shown in Figures 10, 11, 12, and 13, explicit CoT can fail in several different ways: it may overfocus on a spurious textual detail, compress the retrieval intent into an overly coarse verbal summary, rely on a misleading surface-level prior, or drift toward an incorrect action description.

For each PLUME example, the small line chart visualizes the cosine similarity between the hidden state at each latent rollout step and the embeddings of the top retrieved candidates at the end of inference. In other words, the curves show how strongly each intermediate latent state aligns with several final candidate targets as the rollout proceeds. We emphasize that these trajectories are used only as a diagnostic view of latent computation: they are not intended to imply that every step must monotonically ap-

proach the correct target or correspond to an interpretable verbal reasoning trace. Instead, they illustrate a different property from explicit CoT: PLUME does not externalize intermediate reasoning into a fixed textual path, and therefore is less likely to be locked into an early verbal mistake. Across these cases, although the relative similarities may fluctuate during rollout, the correct target remains recoverable and is ultimately selected at the end. These examples further support our claim that latent reasoning is less vulnerable to discrete verbal lock-in than explicit CoT in retrievaloriented embedding formation.

- Table 7. Full MMEB-V2 results with selected baselines (Part I: Overall and Image). “VLM2Vec” denotes VLM2Vec-7B.

###### Task GME-2B LamRA VLM2Vec VLM2Vec-V2.0 DUME-2B UME-R1-2B PLUME

Avg - All (78 tasks) 54.1 40.5 46.9 58.1 52.7 60.1 61.6 Avg - Image (36 tasks, Hit@1) 51.9 54.1 59.7 64.9 62.5 66.6 66.3

I-CLS (10) 54.6 59.3 58.6 62.9 59.3 64.8 66.5 I-QA (10) 29.8 26.5 49.2 56.4 54.9 62.8 59.2 I-RET (12) 66.8 70.1 65.0 69.6 66.3 67.6 67.6 I-VG (4) 55.6 62.6 73.1 77.1 78.0 77.2 79.7

ImageNet-1K 58.1 72.6 77.5 80.8 74.6 75.3 74.1 N24News 50.3 51.6 73.8 73.0 69.7 81.1 81.1 HatefulMemes 53.7 49.1 57.9 55.8 65.3 75.2 75.5 VOC2007 75.8 80.0 74.3 84.9 68.9 80.0 86.1 SUN397 67.0 68.7 73.7 70.9 71.4 79.4 76.9 Place365 36.1 40.5 35.0 36.1 41.0 42.6 42.4 ImageNet-A 28.4 47.2 50.6 47.6 41.3 50.4 50.8 ImageNet-R 78.5 88.4 84.7 89.3 90.7 88.7 87.5 ObjectNet 71.2 66.3 36.9 65.1 46.2 52.0 61.5 Country211 26.4 28.3 21.7 25.8 23.9 23.4 25.0 OK-VQA 30.1 37.9 48.2 51.7 56.8 62.4 60.5 A-OKVQA 18.6 26.9 39.7 44.0 46.9 51.1 49.9 DocVQA 30.0 22.4 82.7 90.1 86.0 92.2 89.9 InfographicsVQA 11.8 16.5 47.3 59.1 59.2 67.7 59.6 ChartQA 13.3 11.6 42.2 48.1 39.1 64.9 49.8 Visual7W 15.6 19.8 50.9 52.8 46.9 54.1 47.6 ScienceQA 27.1 26.5 30.5 38.1 38.7 42.7 42.9 VizWiz 37.1 31.9 38.8 43.3 42.0 46.8 46.5 GQA 75.3 38.3 48.1 65.4 60.2 67.3 69.1 TextVQA 39.5 33.1 63.2 71.6 73.9 78.6 78.9 VisDial 47.7 61.0 75.1 82.7 75.9 76.6 72.6 CIRR 43.1 52.1 46.8 57.3 52.0 53.7 54.6 VisualNews.t2i 74.8 70.9 73.4 74.7 71.2 71.7 71.3 VisualNews.i2t 77.7 84.1 73.8 78.3 72.5 74.2 72.7 MSCOCO.t2i 68.3 72.0 73.1 75.9 74.5 75.1 74.1 MSCOCO.i2t 63.2 73.6 68.3 71.1 68.3 68.9 69.8 NIGHTS 67.6 65.7 65.8 68.4 67.5 67.2 68.0 WebQA 88.8 81.2 85.8 90.6 90.2 90.0 89.1 FashionIQ 32.2 41.7 13.8 19.6 11.5 17.1 20.3 Wiki-SS-NQ 73.8 70.1 54.6 67.6 60.0 62.0 68.6 OVEN 72.1 82.2 68.4 64.8 65.2 66.9 68.4 EDIS 91.7 86.1 81.4 84.2 86.5 88.0 81.8 MSCOCO 28.4 44.7 65.7 66.2 68.1 69.5 66.9 RefCOCO 55.8 62.5 80.8 87.0 85.1 83.3 86.5 RefCOCO-Matching 73.7 76.2 76.6 86.3 89.3 84.4 88.4 Visual7W-Pointing 64.6 67.1 69.1 69.0 69.5 71.5 74.9

- Table 8. Full MMEB-V2 results with selected baselines (Part II: Video and VisDoc). “VLM2Vec” denotes VLM2Vec-7B.

Task GME-2B LamRA VLM2Vec VLM2Vec-V2.0 DUME-2B UME-R1-2B PLUME

Avg - Video (18 tasks, Hit@1) 33.6 35.2 28.6 34.7 33.2 42.2 44.1 Avg - Visdoc (24 tasks, NDCG@5) 72.7 23.9 41.6 65.4 52.8 63.9 67.5

V-CLS (5) 34.8 39.4 33.3 39.2 37.7 44.3 45.0 V-QA (5) 41.8 42.7 30.7 34.7 46.6 51.0 52.3 V-RET (5) 25.4 24.5 20.5 28.4 17.1 32.9 33.5 V-MR (3) 31.8 33.6 30.7 37.6 30.0 39.7 46.7

K700 34.9 42.3 31.0 38.2 22.7 35.8 42.2 SmthSmthV2 30.1 36.1 30.8 43.0 37.7 44.1 44.8 HMDB51 43.0 40.4 34.0 40.2 53.4 54.4 51.2 UCF101 52.1 60.7 57.6 60.0 55.7 67.2 66.5 Breakfast 13.6 17.6 12.9 14.8 18.9 20.1 20.1 MVBench 37.6 37.3 30.5 33.6 48.8 49.9 47.4 Video-MME 34.0 34.1 27.1 30.8 39.2 41.7 40.0 NExTQA 39.4 43.7 20.2 20.9 55.2 59.9 57.3 EgoSchema 40.6 44.8 25.6 35.0 23.2 45.4 47.8 ActivityNetQA 57.2 53.8 49.9 53.0 66.7 57.8 69.2 DiDeMo 21.5 25.0 19.1 30.0 16.9 32.4 32.7 MSR-VTT 27.0 22.6 25.6 27.8 16.2 34.3 36.2 MSVD 47.3 46.4 37.5 47.3 34.9 55.4 56.1 VATEX 23.1 19.1 15.7 26.2 11.1 29.9 28.2 YouCook2 7.9 9.3 4.4 10.6 0.06 12.7 14.5 QVHighlight 44.0 53.9 43.7 49.7 40.3 57.5 57.1 Charades-STA 14.3 10.9 12.9 20.1 16.1 20.4 19.4 MomentSeeker 37.1 36.0 35.4 42.9 33.7 41.2 63.5

VD-Vidore-V1 (10) 87.1 33.8 20.6 74.4 67.6 72.4 72.1 VD-Vidore-V2 (4) 53.9 11.5 13.2 44.6 43.3 46.2 49.8 VD-VisRAG (6) 82.4 37.6 52.2 79.3 47.1 79.2 78.1 VD-OOD (4) 43.1 21.0 33.6 39.4 33.8 37.2 57.4

ViDoRe.arxivqa 82.5 31.5 18.1 78.9 68.7 73.9 72.6 ViDoRe.docvqa 55.2 19.9 14.0 37.1 33.6 37.9 36.2 ViDoRe.infovqa 90.7 63.7 39.5 82.7 74.5 76.2 79.0 ViDoRe.tabfquad 93.3 53.5 36.0 87.8 78.3 86.1 88.8 ViDoRe.tatdqa 70.3 7.9 10.5 44.3 35.3 40.6 36.6 ViDoRe.shiftproject 92.9 16.0 8.4 61.0 61.8 66.8 64.8 ViDoRe.artificial intelligence 98.1 29.8 17.0 89.1 74.3 85.9 83.8 ViDoRe.energy 92.6 36.1 16.4 86.3 78.4 83.3 82.6 ViDoRe.government reports 97.2 41.2 25.2 85.6 83.0 82.6 83.2 ViDoRe.healthcare industry 98.5 38.8 20.8 91.1 88.2 90.8 91.1 ViDoRe.csg reports human labeled v2 60.3 6.9 13.1 45.8 48.0 50.2 52.1 ViDoRe.biomedical lectures v2 multilingual 54.0 13.4 6.5 44.6 39.8 46.2 48.2 ViDoRe.economics reports v2 multilingual 50.8 19.4 12.9 42.3 44.1 45.7 49.6 ViDoRe.csg reports v2 multilingual 50.4 6.4 20.3 45.7 41.1 42.7 49.0 VisRAG.ArxivQA 82.0 2.0 41.2 76.7 35.8 74.3 71.6 VisRAG.ChartQA 79.1 42.7 59.5 84.2 47.2 86.0 80.8 VisRAG.MP-DocVQA 84.3 33.4 43.6 71.8 35.3 75.6 74.9 VisRAG.SlideVQA 93.7 56.3 74.5 91.4 61.3 87.1 88.9 VisRAG.InfoVQA 91.4 56.9 71.1 85.9 64.7 84.4 85.7 VisRAG.PlotQA 64.1 34.1 23.5 65.9 38.5 68.0 66.2 ViDoSeck-page 21.6 11.3 17.7 21.9 20.0 21.2 80.6 ViDoSeck-doc - - - 79.0 69.5 75.9 76.9 MMLongBench-page 15.8 8.0 9.6 11.9 10.4 11.9 39.9 MMLongBench-doc - - - 63.0 35.4 39.7 32.0

[Figure 12]

[Figure 13]

- Figure 10. A case where explicit CoT gets trapped by a misleading textual detail while PLUME recovers the correct target. In this example, explicit CoT over-focuses on the spurious text cue “ARMYJUNK,” which biases the subsequent reasoning trace and leads to an incorrect retrieval result. In contrast, PLUME does not commit to a discrete verbal rationale and still retrieves the correct target half track after latent rollout.

[Figure 14]

[Figure 15]

###### Figure 11. A case where explicit CoT over-compresses the retrieval intent while PLUME preserves the correct target. In this example, explicit CoT summarizes the query into an overly coarse verbal description and is distracted toward an incorrect retrieval result. In contrast, PLUME avoids committing to a discrete textual rationale and successfully retrieves the correct target after latent rollout.

[Figure 16]

- Figure 12. A case where explicit CoT is misled by a surface-level prior while PLUME recovers the correct answer. In this example, explicit CoT over-relies on the visual prior that egg cartons are typically made of cardboard and therefore predicts an incorrect result. In contrast, PLUME avoids being locked into this explicit verbal path and successfully retrieves the correct answer styrofoam after latent rollout.

[Figure 17]

[Figure 18]

- Figure 13. A case where explicit CoT drifts toward an incorrect action description while PLUME recovers the correct target. In this example, explicit CoT over-compresses the video into a coarse description of two pens on a wooden floor and is distracted toward an incorrect retrieval result. In contrast, PLUME avoids committing to this explicit verbal path and successfully retrieves the correct target putting pen similar to others that are already on the table after latent rollout.

