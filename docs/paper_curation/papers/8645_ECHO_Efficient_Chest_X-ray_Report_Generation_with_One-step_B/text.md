[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

### ECHO: Efficient Chest X-ray Report Generation with One-step Block Diffusion

# arXiv:2604.09450v2[cs.LG]17May2026

###### Lifeng Chen1,2,∗,⋆, Tianqi You1,3,∗,⋆, Hao Liu1,†,‡, Zhimin Bao1, Jile Jiao1, Xiao Han1, Zhicai Ou1, Tao Sun1, Xiaofeng Mou1, Xiaojie Jin2, Yi Xu1,†

1AIRC, Midea Group, 2Beijing Jiaotong University, 3Dalian University of Technology ∗Equal Contribution, †Corresponding Author, ‡Project Leader, ⋆Work completed during an internship at Midea AI Research Center.

###### Abstract

Chest X-ray report generation (CXR-RG) has the potential to substantially alleviate radiologists’ workload. However, conventional autoregressive vision–language models (VLMs) suffer from high inference latency due to sequential token decoding. Diffusion-based models offer a promising alternative through parallel generation, but they still require multiple denoising iterations. Compressing multi-step denoising to a single step could further reduce latency, but often degrades textual coherence due to the mean-field bias introduced by token-factorized denoisers. To address this challenge, we propose ECHO, an efficient diffusion-based VLM (dVLM) for chest X-ray report generation. ECHO enables stable one-step-per-block inference via a novel Direct Conditional Distillation (DCD) framework, which mitigates the mean-field limitation by constructing unfactorized supervision from on-policy diffusion trajectories to encode joint token dependencies. In addition, we introduce a Response-Asymmetric Diffusion (RAD) training strategy that further improves training efficiency while maintaining model effectiveness. Extensive experiments demonstrate that ECHO surpasses state-of-the-art autoregressive methods, improving RaTE and SemScore by 64.33% and 60.58% respectively, while achieving up to 8× inference speedup with negligible degradation in clinical accuracy.

Keywords: CXR Report Generation , One-step Block Diffusion , Direct Conditional Distillation

Date: May 19, 2026 Correspondence: Hao Liu, Yi Xu Project Page: https://echo-midea-airc.github.io/

###### 1 Introduction

In recent years, Vision-Language Models (VLMs) [3, 19, 24, 26, 31, 42, 46], where visual features are aligned with language instructions to enable complex cross-modal understanding, have demonstrated significant progress in many fields, notably medical image analysis [6, 21, 25, 34, 37, 39, 44, 52, 59]. Within this field, automated chest X-ray report generation (CXR-RG) [7, 23, 35, 43, 48, 54] has emerged as a critical application. As one of the most common clinical imaging exams, CXR’s high volume places a heavy diagnostic burden on radiologists, creating strong demand for high-throughput automated reporting systems to ease workloads. Despite the promising performance achieved, canonical autoregressive (AR) VLMs often

∏  (  , |  ) — factorized

60

Vanilla Mean-field

Multi-step Diffusion One-step Diffusion

Autoregressive/ LLaDA-based

###### ECHO (Ours）

[Figure 5]

55

| | |
|---|---|
| | |

(TPF: 8.0, SemScore: 53.4)

SDAR*

###### CXR

[Figure 6]

50

T3D

CXR-ReportsSemScore(%)

no bilateral pleural focal

✗

CD4LM

45

 ,  , <  — conditional

SDAR*1step

d3LLM

DCD (Ours)

40

dParallel

Review the findings of this chest X-ray:

ECHO (Ours)

35

SDAR*

MedGemma-27B

Distillation Variants

[M] [M] [M] [M]

25

External Baselines

right lower lobe opacity

LLaDA-MedV

✓ (a) Illustration of the mean-field bias and our DCD

15

0 2 4 6 8 10

TPF (Tokens per forward)

(b) TPF vs. SemScore across methods

- Figure 1 Motivation and performance of ECHO. (a) Decoding all tokens simultaneously in one step produces incoherent outputs, as standard diffusion models predict each position independently. Our Direct Conditional Distillation (DCD) distills from a non-factorized target, yielding coherent one-step-per-block outputs. (b) Compared to both autoregressive and diffusion-based baselines, ECHO achieves a favorable trade-off between generation quality (SemScore) and decoding throughput (tokens per forward pass). Here, SDAR* denotes SDAR [9] implemented under the multimodal large language model setting.

suffer from high inference latency due to their sequential decoding mechanism, which becomes a primary bottleneck for producing reports rapidly and at scale. Fortunately, thanks to parallel decoding capabilities, emerging diffusion-based Vision Language Models (dVLMs) [9, 28, 57, 58] offer a highly promising route toward achieving fast report generation.

Despite the promise of parallel decoding, dVLMs in practice necessitate multiple denoising steps to ensure output coherence. This requirement stems from the mean-field approximation underlying their token-factorized denoisers, which introduces a structural bias that scales monotonically with the noise ratio. Although compressing decoding into a single step theoretically maximizes throughput, it compels the model to predict all tokens simultaneously from a fully masked input, precisely the scenario where mean-field bias is most acute. As illustrated in Fig. 1, the resulting inter-token incoherence leads to substantial quality degradation, prompting the critical question: Can we achieve the upper bound of decoding speed without compromising output fidelity?

To address this challenge, distillation emerges as a natural solution, seeking to transfer the quality of multistep denoising into a student model that operates in fewer passes. Specifically, several dLLMs [8, 12, 29, 36, 60] implement this via self-distillation, which aligns token-wise predictions with teacher’s ones from less corrupted states. However, these teacher targets remain factorized across positions, ignoring dependencies between concurrently predicted tokens. For few-step inference, this factorization remains tolerable, as progressive unmasking provides the inter-token context that token-independent targets fail to capture. In contrast, one-step decoding lacks such a corrective mechanism, allowing the full mean-field bias to resurface unchecked. Therefore, enabling reliable one-step decoding requires a fundamentally different training objective, one that is unfactorized and directly encodes the joint dependencies among tokens predicted in parallel. To achieve this goal, we propose a new Direct Conditional Distillation (DCD), which constructs supervision from the teacher’s on-policy remasking trajectory by conditioning each target on committed high-confidence context. The constructed supervision can encode inter-token dependencies into the training signal and enable stable one-step-per-block parallel inference.

Building upon DCD, we present ECHO (Efficient Chest X-ray Report Generation with One-step Block Diffusion), a new foundational dVLM for CXR report generation that achieves both clinical accuracy and one-step inference efficiency. More concretely, we first establish an enhanced AR-based CXR-RG VLM by training on our curated data, where clinical findings and symptoms are explicitly and comprehensively annotated. Based on this, we propose Response-Asymmetric Diffusion (RAD) adaptation to efficiently convert the AR model

into a block diffusion decoding paradigm. Subsequently, we apply DCD to distill this multi-step model into a one-step counterpart.

Notably, compared with state-of-the-art autoregressive methods, ECHO improves RaTE and SemScore by 64.33% and 60.58%, respectively, while achieving up to 8× theoretical and 5.1× practical inference speedup over autoregressive baselines. Our main contributions are summarized below:

- • We present ECHO, a novel dVLM for CXR report generation that delivers strong clinical accuracy through one-step-per-block parallel decoding, outperforming both pioneering autoregressive and diffusionbased models by a large margin across standard benchmarks.
- • We propose a novel Direct Conditional Distillation (DCD), to the best of our knowledge the first one-step distillation framework for discrete diffusion language models. By enabling one-step-per-block inference that reaches the upper bound of decoding speed, DCD achieves up to 390% inference speedup over the corresponding multi-step baseline at block size L=8 with only marginal quality degradation.
- • We design Response-Asymmetric Diffusion (RAD) adaptation, which further reduces theoretical training FLOPs by 72.3%, translating to a 3.61× speedup in training efficiency.

###### 2 Related Works

###### 2.1 Multimodal Medical Foundation Models

Large-scale vision-language models (VLMs) [3, 19, 24, 26, 31, 42, 46] have established strong multimodal understanding by aligning visual representations with language instructions, achieving remarkable performance across diverse visual reasoning tasks. Motivated by this progress, researchers have adapted VLMs to the medical domain [6, 21, 25, 34, 37, 39, 44, 52, 59], where precise alignment between clinical images and textual descriptions is essential. Within medical imaging, automated CXR report generation has emerged as a central task [7, 23, 35, 43, 48, 54], where existing methods predominantly follow an autoregressive paradigm and achieve strong clinical accuracy but at the cost of sequential decoding throughput, motivating the need for faster generation paradigms.

###### 2.2 Diffusion Language Model

Discrete diffusion language models (dLLMs) represent one such paradigm, enabling parallel token prediction in place of sequential decoding. Specifically, dLLMs [2, 4, 33, 40, 55] define a forward process that progressively masks tokens and a learned reverse process that recovers them over a discrete token vocabulary. Building on this foundation, dVLMs [9, 28, 57, 58] extend the framework to vision-language tasks through a two-stage process of visual encoder alignment followed by instruction fine-tuning, enabling image-conditioned parallel generation. Block diffusion [1, 14, 17] further refines these models by adopting a semi-autoregressive decoding scheme that generates outputs block by block in causal order, naturally supporting variable-length sequence generation. A further line of work specifically addresses the high training cost of building such models from scratch by directly adapting pretrained autoregressive models into block diffusion models [4, 5, 9, 15, 49], offering a practical and scalable alternative. Collectively, these advances have improved both the training scalability and inference throughput of dLLMs and dVLMs.

###### 2.3 Acceleration of dLLMs

As dLLMs and dVLMs continue to mature, further accelerating their inference has attracted growing research attention. This has been pursued from two complementary directions: inference-time optimization techniques [10, 18, 27, 32, 47, 50] that reduce per-step or per-token computation overhead, and trainingtime distillation [8, 12, 29, 36, 60] that compresses the multi-step denoising process into fewer steps. Along the distillation direction, methods such as SDTT [12] and dParallel [8] progressively align predictions at higher noise levels with those at lower ones, using either cross-entropy loss on self-generated trajectories or KL divergence between predicted distributions. Most recently, T3D [60] takes a distinct angle, employing a DPO-style optimization objective to directly penalize mean-field bias during training rather than matching noise-level predictions. Despite these advances, all existing methods retain token-factorized prediction targets

and therefore still require multiple denoising steps to produce coherent outputs, leaving dLLM and dVLM throughput far below its theoretical upper bound. ECHO addresses this gap with Direct Conditional Distillation (DCD), a distillation framework for dLLMs and dVLMs that supports one-step denoising, applied here to block diffusion.

###### 3 Preliminaries

To motivate our approach, we formalize the dLLM framework and analyze how its token-factorized parameterization gives rise to the parallelism bottleneck identified in Section 1. Let V denote the vocabulary and L the sequence length. A token sequence is x0 = (x0,1,...,x0,L) ∈ VL. Discrete diffusion language models (dLLMs) [38, 40, 55] define a forward process q(x1:T | x0) that progressively masks tokens, and a reverse process pθ(x0:T−1 | xT) that recovers them.

Mean-field parameterization. Since the joint posterior p(x0 | xt,t) over VL requires exponentially many parameters, existing dLLMs [1, 33, 38] universally adopt a token-factorized (mean-field) approximation [2, 51, 56, 62]:

L

pθ(x0,i | xt,t), (1)

pθ(x0 | xt,t) =

i=1

trained by minimizing the per-token denoising objective:

LMF(θ) = Ex

0, t,xt −

L

log pθ(x0,i | xt,t) . (2)

i=1

As Eq. (2) decomposes into L independent per-position cross-entropies, the global optimum is attained when each factor independently matches the true conditional marginal, i.e. p⋆θ(x0,i | xt,t) = q(x0,i | xt,t). The resulting optimal factorized joint p⋆MF = i q(x0,i | xt,t) aligns each position’s marginal with the true posterior, but cannot capture the cross-positional correlations present in q(x0 | xt,t). We measure this gap by the mean-field bias:

ϵMF(xt,t) ≜ KL q(x0 | xt,t) p⋆MF(x0 | xt,t) , (3) which quantifies the irreducible joint dependence structure across token positions that no factorized distribution can represent.

Why multi-step sampling mitigates the bias. The expected mean-field bias ϵ¯MF(t) ≜ Ex

[ϵMF(xt,t)] is governed by the corruption level: when xt has few masked tokens, the remaining unknowns are sparsely distributed and largely conditionally independent, so ϵ¯MF(t) is small; when all tokens are masked (t = T), the posterior exhibits strong cross-positional dependence and ϵ¯MF(T) reaches its maximum. In general, ϵ¯MF(t) grows monotonically with the number of masked positions. [56] Multi-step reverse sampling systematically exploits this monotonicity. At each reverse step s, tokens decoded with high confidence from pθ(· | xs,s) are committed as observed context, reducing the number of masked positions from ms to ms−1 < ms. Consequently, the model at step s − 1 operates under a strictly reduced corruption level and, accordingly, a lower expected bias ϵ¯MF(s − 1) ≤ ϵ¯MF(s). By iterating this process, the multi-step chain progressively shifts decoding from the high-bias fully-masked regime toward a low-bias, nearly-observed state, thereby recovering the inter-token dependence structure that Eq. (1) otherwise discards.

t

###### 4 Methodology

###### 4.1 Overview

As illustrated in Fig. 2, the training pipeline of ECHO comprises three successive stages. In Stage 1, we perform continued pre-training (CPT) on Lingshu-7B [52] using a curated CXR corpus. This produces ECHOAR, an autoregressive vision-language model specialized in radiology report generation. Subsequently, in Stage 2, we propose Response-Asymmetric Diffusion (RAD) adaptation, which converts ECHOAR into ECHOBase, a block diffusion decoding model that retains the domain knowledge of ECHOAR, achieving an initial inference speedup. Lastly, in Stage 3, to further push decoding speed toward its theoretical upper bound, we apply

Before

Data & Foundation

Training Paradigm

[Figure 7]

[Figure 8]

Noisy x Clean 𝑥

VP

VP VP R R VP VP VP R R

|[Figure 9]<br><br>| |[Figure 10]| |
|---|---|---|---|
| | | | |

VP

[Figure 11]

Stage 1

VP

[Figure 12]

[Figure 13]

VP

Base Model

KL Divergence

Continual Pretraining

R

R

ECHO-Base

VP

ECHO

[Figure 14]

VP

Normalized Report (Clean)

VP

R

Findings: \n The thorax is symmetric bilaterally. The mediastinum and trachea are midline; slight prominence of right paratracheal soft tissue noted. ... lungs are clear with no definite abnormal opacities in either lung field. Cardiac silhouette is mildly enlarged ... Impression: \n Mild cardiomegaly. No significant

Chest X-Ray

[Figure 15]

R

Stage 2

After (Ours)

[Figure 16]

Response-Asymmetric Diffusion Adaption

Unnormalized Report (Raw)

[Figure 17]

[Figure 18]

[Figure 19]

Context x 𝑥

VP VP R R R R ECHO-Base

VP R R

Findings: \n Lungs are clear. Cardiac silhouette mildly enlarged. Slight prominence of right paratracheal soft tissue possibly due to vascular structures. \n Impression: \n Mild cardiomegaly.

|[Figure 20]| | |
|---|---|---|
| | | |
| | | |

reserved tokens

condition tokens

VP

VP

mask tokens

logits distribution

VP

abnormalities in the lungs, hila ...

[Figure 21]

Stage 3

R

R

[Figure 22]

Direct Conditional Distillation

(I) Phase 1: Target Trajectory Construction (II) Phase 1: Student One-Step Align

R

Medical LLM

R

Response-Asymmetric

Diffusion Adaption (b) Direct Conditional Distillation (DCD)

(a)

- Figure 2 Overview of the ECHO training pipeline. ECHO is built in three successive stages: continued pre-training (Stage 1) produces ECHOAR; Response-Asymmetric Diffusion adaptation (Stage 2) converts it into the block diffusion model ECHOBase; and Direct Conditional Distillation (Stage 3) distills ECHOBase into the final one-step-per-block model ECHO. (a) RAD duplicates only the response portion of the training sequence, avoiding the redundant duplication of long vision token contexts required by prior two-stage conversion methods. (b) DCD proceeds in two phases: the teacher’s confidence-heuristic remasking trajectory is collected to form a joint, non-factorized supervision target, which is then used to align the student’s single-step prediction via KL divergence.

Direct Conditional Distillation (DCD) to train ECHOBase toward one-step-per-block decoding, yielding the final ECHO model.

###### 4.2 From ECHO-AR to ECHO

Response-Asymmetric Diffusion Adaptation. While ECHOAR achieves high-quality report generation, its tokenby-token decoding mechanism limits inference throughput. To address this, our first objective is to convert ECHOAR into ECHOBase, a block diffusion model that preserves the domain knowledge acquired during pretraining. Prior work [9] has implemented this through a two-stage adaptation, i.e., a pretraining phase followed by SFT. In these methods, the entire training sequence, covering vision, instruction, and response tokens, is duplicated across both stages to construct block-causal teacher-forcing targets. For CXR report generation, where vision token sequences are substantially long, this full-sequence duplication incurs prohibitive training costs. To overcome this inefficiency, we propose Response-Asymmetric Diffusion (RAD), which achieves this adaptation through a single SFT stage. As illustrated in Fig. 2, RAD duplicates only the response portion of the training sequence, and constructs a block attention mask such that each noisy response block attends to all vision and instruction tokens as well as all previously decoded blocks. As illustrated in Fig. 3(b), this asymmetric design eliminates the redundant duplication of long vision token contexts, substantially reducing training FLOPs while consolidating the originally two-stage conversion into one. Furthermore, we conduct a series of experiments (Sec. 5.4) to investigate the relationship between training data volume, model quality, and inference speed during the RAD conversion. We find that ECHOBase attains performance comparable to ECHOAR using only a small fraction of the original training data, demonstrating the high knowledge-transfer efficiency of RAD.

Direct Conditional Distillation. With ECHOBase established, our next objective is to convert it into a model capable of one-step-per-block decoding, fully maximizing inference throughput. However, as discussed in Sec. 3, discrete diffusion language models inherently rely on multi-step denoising to progressively reduce the mean-field bias, and our dVLM setting is no exception. This calls for a training target that is itself non-factorized: by aligning the student to such a target, the inter-token dependencies accumulated across the multi-step denoising trajectory can be captured in a single forward pass. To this end, we propose Direct Conditional Distillation (DCD), which constructs this non-factorized target by collecting and stitching the high-confidence token distributions at each step of the teacher’s multi-step denoising process, forming a joint supervision signal that encodes cross-token dependencies.

Algorithm 1: Direct Conditional Distillation (DCD) Input : Context xctx, teacher θ, student ϕ, N blocks, block length L, sampling confidence threshold τ Output: Updated student ϕ

- // Phase 1: On-policy Teacher Trajectory Collection

- 1 for n = 1,...,N do

- 2 Initialize B ← {1,...,L},;
- 3 xcurr ← (xctx, bˆ1:n−1);
- 4 while B ̸= ∅ do

- 5 Query teacher pθ(xi | xcurr) for all i ∈ B;
- 6 U ← {i ∈ B | ci ≥ τ};;
- 7 if U = ∅ then U ← {arg maxi∈B ci};
- 8 Ptch(n)[i] ← pθ(xi | xcurr),;
- 9 xˆi ← arg max pθ(xi | xcurr) for all i ∈ U;
- 10 xcurr ← xcurr ∪ {xˆi}i∈U,;
- 11 B ← B \ U;

- 12 bˆn ← (ˆx1,...,xˆL);

// Phase 2: Student One-Step Alignment (cf. Sec. 4.2 and Fig. 2)

- 13 Construct RAD training sequence xtrain using (xctx, bˆ1:N);
- 14 LDCD ← n,i

DKL Ptch(n)[i] pϕ(bn,i | xtrain) ;

- 15 Update ϕ ← Optimizer(ϕ, ∇ϕLDCD);

The full self-distillation procedure is detailed in Algorithm 1 and consists of the following two phases, applied iteratively during training.

- Phase 1: On-policy Teacher Trajectory Collection. The teacher trajectory is the trajectory sampled by its multi-step denoising process. To obtain this, we first run ECHOBase through a confidence-heuristic denoising process, where tokens are progressively unmasked in order of their prediction confidence ci ≜ maxv pθ(xi=v | xcurr). Specifically, at each step of a block’s denoising, we record the predicted distribution and the committed

token label for every position as it is unmasked. The collected token labels form the pseudo labels bˆn, which are used to construct the training sequence for the subsequent student alignment phase. The collected

distributions are then stitched together into the joint target Ptch(n) for the n-th block, as the distillation target in Phase 2.

- Phase 2: Student One-Step Alignment. With the pseudo-labels and stitched joint distribution collected in Phase 1, the next step is to align this distribution with the student’s one-step prediction. Concretely, we first use the pseudo-labels to construct a block teacher-forcing training sequence following the RAD scheme (Sec. 4.2), ensuring that the student makes its prediction under exactly the same conditions as the

teacher. Letting bn,i denote the token at position i of block n and Q(ϕn) ≜ pϕ(bn,i | xtrain) Li=1 the student’s one-step predicted distribution over that block, we minimize the forward Kullback–Leibler (KL) divergence

DKL Ptch(n) ∥Q(ϕn) for each block and aggregate the loss over all blocks. Intuitively, a token that is unmasked at a later step within a block requires stronger conditioning to reach a high-confidence prediction, indicating that its position is subject to more severe mean-field bias. Therefore, we introduce a token reweighting scheme that assigns each position a weight proportional to the step at which it is unmasked. Under this scheme, positions with larger mean-field bias receive stronger supervision, while tokens committed at early steps serve as a regularization signal for the alignment process.

After DCD, we can generate each block with only one step, achieving fast decoding speed without compromising clinical accuracy.

9×

| |[Figure 23]<br><br>[Figure 24]<br><br>[Figure 25]<br><br>[Figure 26]<br><br>[Figure 27]<br><br>[Figure 28]<br><br>[Figure 29]<br><br>0.877<br><br>[Figure 30]<br><br>0.822<br><br>[Figure 31]<br><br>0.880<br><br>[Figure 32]<br><br>0.753<br><br>[Figure 33]<br><br>Token type Content Tokens EOS Token<br><br>[Figure 34]<br><br>[Figure 35]| | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

1.0

80

7×

[Figure 40]

AttentionFLOPsSaved(%)

[Figure 41]

TrainingSpeedup()×

CXR (~2,870 vis. tokens) −73.0% FLOPs 3.70× speedup

0.8

Confidence(meanstd)±

60

5×

0.6

[Figure 42]

40

[Figure 43]

0.4

3×

20

-

[Figure 44]

High Resolution Image Section

0.2

Natural Image Inputs Section

0

1×

0 500 1,000 1,500 2,000 2,500 3,000 3,500 4,000

0.0

Vision Token Count

blk4 blk8

(a) EOS Token Confidence: blk4 vs blk8

(b) RAD Efficiency vs. Vision Context Length

- Figure 3 (a) Confidence analysis of <eos> tokens vs. content tokens under block sizes blk4 and blk8 in standard multi-step inference.(b) RAD attention FLOPs saved (%) and training speedup (×) as a function of vision token count.

###### 4.3 Hallucination Mitigation

Hallucination is a critical concern in CXR report generation, where inaccurate descriptions of clinical findings can directly undermine diagnostic reliability. During the development of our vision-language model, we identify two predominant hallucination patterns in medical dLLMs: inaccurate identification of symptoms, and degenerate repetition loops that fail to terminate.

For inaccurate symptom identification, we attribute this pattern to an implicit negative bias inherent in the training data. In routine clinical practice, radiologists follow a “reporting by exception” convention, describing only abnormal findings while omitting normal structures or dismissing them with a single catchall phrase such as “no other significant abnormality.” Consequently, for the majority of anatomical regions that are clinically unremarkable, their normal status is absent from the report rather than explicitly stated as negative. This systematic omission leaves the model without explicit negative evidence during training, leaving its conditional distribution under-constrained. At inference time, this under-constraint manifests as two complementary failure modes: false-positive hallucinations, where the model fabricates findings for normal regions, and false-negative omissions, where genuine abnormalities are overlooked. To address this, we propose a data normalization paradigm tailored for CXR report generation. Specifically, we reformulate every training report so that each predefined anatomical region receives an explicit annotation, either a positive finding or a negative assertion. This ensures unambiguous supervision at every position, directly eliminating the implicit omission bias described above. Our experiments in Sec. 5.6 show that the hallucination reduction brought by data normalization consistently benefits both ECHOAR and the final ECHO model.

Beyond inaccurate symptom identification, ECHO also suffers from degenerate repetition loops that fail to terminate. To understand the root cause, we examine ECHOBase’s prediction confidence of content tokens and <eos> tokens under standard confidence-heuristic inference. As shown in Fig. 3(a), <eos> tokens exhibit systematically lower mean confidence and substantially greater variance compared to content tokens. Using such a flat and unstable distribution as the distillation target for the <eos> token makes it difficult for the one-step model to terminate generation with high confidence. This problem is further exacerbated as the block size grows: as shown in Fig. 3(a), increasing the block size leaves content token confidence nearly unaffected while causing a notable decline in <eos> confidence. To address this, we apply an additional cross-entropy loss on the <eos> token during distillation, explicitly pulling its predicted distribution toward a sharp, high-confidence one-hot target. Together, the two hallucination mitigation strategies consistently reduce generation errors in ECHO, contributing to improved clinical accuracy.

Prompt Block 0 Block 1 Prompt Block 0 Block 1

Prompt token

Cached token

Decoded token

Compute cache

Decode cache

###### (a) Vanilla Block KV Cache (b) Fused Block KV Cache

- Figure 4 (a) Vanilla block KV cache: after all tokens of a block are committed, a dedicated forward pass is performed to update the KV cache. (b) Fused block KV cache: the KV cache update for the preceding block is fused into the current block’s denoising forward, eliminating the dedicated KV update pass.

###### 4.4 Inference Paradigm

To further maximize the inference throughput of ECHO, we optimize the inference paradigm used in dLLMs. A common technique in semi-autoregressive decoding is block KV cache [50], which caches the key-value states of previously decoded blocks to avoid redundant recomputation. As illustrated in Fig. 4(a), after all tokens of a block are committed, a dedicated forward pass is performed solely to update the KV cache with the newly decoded block. This additional forward pass incurs acceptable inference time overhead in multi-step denoising, but for our one-step model it doubles the total number of forward passes for each block. To eliminate this overhead, we propose Fused Block KV Cache. As illustrated in Fig. 4(b), after a block is decoded, we defer its KV cache update and fuse it into the next block’s denoising forward. In this fused forward, the model simultaneously computes and caches the key-value states for the preceding decoded block while denoising the current block’s fully-masked tokens, removing the dedicated KV update pass entirely. As proven in Sec. E, Fused Block KV Cache introduces no additional FLOPs while halving the number of forward passes, directly reducing inference latency.

###### 5 Experiments

In this section, we present experimental results to assess the effectiveness of ECHO for fast and reliable CXR report generation. We compare ECHO against state-of-the-art autoregressive baselines on report quality, and against diffusion-based distillation baselines on distillation efficiency. We then ablate the individual components of DCD, and analyze the effect of training data volume during RAD adaptation and the impact of report normalization across all training stages.

###### 5.1 Experimental Setups

###### 5.1.1 Dataset

We conduct experiments on a unified CXR report corpus built from multiple public datasets, including MIMIC-CXR [22], CheXpert-Plus [20], ReXGradient [61], and IU-Xray [11]. For training, we apply a standardized cleaning and normalization pipeline across all source datasets to construct a bilingual training set, with a small subset of LLaVA-ReCap-558K [26] included to mitigate catastrophic forgetting. For evaluation, we sample 2,000 English and 2,000 Chinese reports from each of normalized MIMIC-CXR, CheXpert-Plus, and ReXGradient, ensuring balanced coverage across datasets and languages. Full preprocessing details and data statistics are provided in Appendix. B.1.

###### 5.1.2 Models

All ECHO models are initialized from Lingshu-7B [52], an autoregressive VLM pretrained on large-scale medical corpora, which provides a strong domain-specific knowledge and vision alignment for CXR report generation.

###### 5.1.3 Implementation Details

For RAD, we use the same data as the SFT stage in continued pre-training. The vision encoder and projector weights of ECHOAR are frozen throughout, and only the LLM backbone is trained. For DCD, we randomly sample 30,000 examples from the RAD training set, drawn proportionally across datasets. During teacher trajectory collection, samples that produce degenerate repetition loops are automatically discarded.

###### 5.1.4 Evaluation Metrics

To provide a comprehensive assessment of the generated chest X-ray reports, we evaluate our model across four complementary dimensions: linguistic quality, clinical fidelity, structural stability, and decoding efficiency. These metrics enable a rigorous comparison between ECHO and existing state-of-the-art (SOTA) models while specifically addressing the unique challenges of diffusion-based generation.

- • Linguistic Quality Metrics (LQM). We use standard natural language generation (NLG) metrics, including ROUGE-L [30] and CIDEr [45], to measure lexical overlap and fluency between generated and reference reports, enabling direct comparison with prior methods.
- • Clinical Fidelity Metrics (CFM). Linguistic fluency does not guarantee accurate symptom identification, so we use RaTEScore [63] and SemScore [41] to assess clinical content fidelity. Since our evaluation set consists of normalized reports, a fair comparison across models requires focusing solely on symptom identification accuracy rather than reporting style. To this end, we only adopt the positive-finding score of RaTEScore, and SemScore is insensitive to negative findings by design.
- • Structural Stability Metrics (SSM). To assess generation stability specific to discrete diffusion models, we report Perplexity (PPL) computed with Qwen3-1.7B as the judge model. Lower PPL indicates more stable and fluent generation.
- • Efficiency Metrics (EM). To evaluate decoding efficiency, we report two complementary speed metrics. TPF (Tokens Per Forward Pass) measures theoretical throughput as the number of decoded tokens per forward pass, normalized relative to the AR baseline (×1.0). TPS (Tokens Per Second) measures practical decoding throughput as the number of tokens generated per second during inference.

###### 5.2 Comparison with State-of-the-art Methods

In Table 1, to evaluate the effectiveness of ECHO, we compare it against three categories of state-of-the-art models: (1) General-purpose proprietary models, including Gemini3-Pro [16] and Qwen3VL-Max [3]. (2) Autoregressive (AR) medical VLMs, such as LLaVA-Med v1.5-7B [25], Lingshu-7B/32B [52], MedGemma27B [39], and Hulu-Med-7B/32B [21]. (3) Diffusion-based medical models, including LLaDA-MedV-8B [13] and several distilled variants [8, 29, 36, 60]. These variants share the same base model ECHOBase,blk8 and distillation data, but with different distillation targets.

Overall Performance. ECHO consistently achieves superior performance compared to both general and specialized models. Notably, it significantly outperforms larger medical VLMs like MedGemma-27B, ranging from 17% to 40% in CFM. Even when compared to the most powerful models like Gemini3-Pro and Qwen3VL-Max, ECHO maintains a clear advantage in all medical clinical metrics.

Efficiency of Distillation. Since all diffusion-based methods compared here are ultimately derived from ECHOAR, the percentage drops in Tab. 1 are reported relative to ECHOAR, providing a unified basis for comparing distillation efficiency across methods and block size configurations. As shown in the table, ECHOblk8 achieves an 8× theoretical speedup over the AR baseline while incurring only 3–7% quality degradation across all metrics. By contrast, T3D delivers only a 2× theoretical speedup yet still incurs a 3–11% quality drop, and dParallel achieves a 4.4× speedup at the cost of 18–33% degradation in clinical fidelity metrics. ECHOblk4 further provides a more conservative working point, where a 4× theoretical speedup reduces clinical fidelity degradation

- Table 1 Performance comparison with state-of-the-art models. We report average metrics across the MIMICCXR, CheXpert-Plus, and ReXGradient test sets. Bold values indicate the best performance. Colored superscript percentages indicate relative quality degradation with respect to ECHOAR. Please refer to Appendix. D for detailed results on individual datasets.

Methods

CheXpert-Plus ReXGradient MIMIC-CXR Speed

ROUGE-L CIDEr RateScore SemScore ROUGE-L CIDEr RateScore SemScore ROUGE-L CIDEr RateScore SemScore TPF TPS

Proprietary general models

Gemini3-Pro [16] 26.95 1.53 40.86 29.72 32.10 1.81 33.52 39.02 27.08 1.55 41.42 30.45 ×1.0 – Qwen3-Max [3] 27.19 1.53 41.63 29.24 30.50 1.72 39.12 38.13 26.86 1.51 40.78 27.32 ×1.0 –

Autoregressive Medical Models

LLaVA-Med [25] 6.92 0.65 10.11 11.39 7.36 0.72 10.01 17.04 6.69 0.65 10.07 9.22 ×1.0 36.33 Lingshu-7B [52] 21.95 1.12 31.34 27.54 23.63 1.25 20.26 34.82 22.24 1.16 32.79 26.94 ×1.0 53.70 Hulu-Med-7B [21] 22.38 1.15 26.25 23.43 21.66 1.14 22.77 23.56 22.26 1.18 22.89 22.30 ×1.0 38.78 MedGemma-27B [39] 20.30 0.79 38.83 31.79 23.41 0.91 26.00 37.74 20.80 0.82 38.48 30.24 ×1.0 16.78 Lingshu-32B [52] 20.40 1.06 30.92 24.32 22.10 1.20 20.33 31.49 20.73 1.10 32.40 24.14 ×1.0 23.60 Hulu-Med-32B [21] 19.46 0.96 27.17 22.07 22.31 1.11 23.01 26.67 17.96 0.99 24.54 18.23 ×1.0 15.67

Diffusion Medical Methods

LLaDA-MedV [13] 7.69 0.22 26.72 17.76 9.72 0.23 18.19 22.09 9.60 0.23 27.47 17.26 ×1.30 12.86 ECHOAR 56.89 4.47 59.18 52.94 67.07 5.68 64.39 66.60 54.55 4.32 56.58 46.70 ×1.0 53.70 ECHOBase,blk4 56.90 4.25 59.12 52.64 66.46 5.64 63.02 66.68 54.45 4.16 55.85 46.10 ×1.89 48.86 ECHOBase,blk8 55.97 4.19 58.39 50.13 65.62 5.47 62.48 64.90 53.52 3.98 54.96 44.48 ×2.10 52.76 CD4LM [29] 49.2114%↓ 3.3824%↓ 55.906%↓ 43.8617%↓ 58.7312%↓ 4.4422%↓ 58.369%↓ 55.6217%↓ 46.2115%↓ 3.1328%↓ 52.118%↓ 38.7617%↓×3.61 98.49 d3LLM [36] 44.0423%↓ 2.9235%↓ 55.396%↓ 40.3524%↓ 54.1819%↓ 4.1727%↓ 57.9910%↓ 53.5020%↓ 41.1325%↓ 2.6539%↓ 50.9910%↓ 34.2027%↓×3.84 101.64 dParallel [8] 42.2226%↓ 3.1729%↓ 44.6525%↓ 36.7231%↓ 48.7327%↓ 4.0229%↓ 45.5029%↓ 54.5318%↓ 39.9827%↓ 2.9632%↓ 40.4729%↓ 31.1533%↓×4.42 110.12 T3D [60] 54.993%↓ 4.069%↓ 56.904%↓ 49.866%↓ 64.364%↓ 5.238%↓ 58.639%↓ 61.657%↓ 52.384%↓ 3.8511%↓ 52.387%↓ 42.519%↓ ×2.00 60.25 ECHOblk8 55.213%↓ 4.147%↓ 56.854%↓ 51.403%↓ 65.133%↓ 5.484%↓ 59.927%↓ 64.004%↓ 53.073%↓ 4.007%↓ 52.976%↓ 44.834%↓ ×8.00 274.21 ECHOblk4 56.141%↓ 4.206%↓ 57.403%↓ 49.576%↓ 66.391%↓ 5.543%↓ 62.183%↓ 66.280%↓ 54.121%↓ 4.056%↓ 53.955%↓ 45.572%↓ ×4.00 129.19

- Table 2 Ablation on DCD components. SW: step-wise token reweighting; CE: cross-entropy loss on the <eos> token; RKL: reverse KL divergence. Results are reported on CheXpert-Plus, ReXGradient, and MIMIC-CXR.

DCD settings CheXpert-Plus ReXGradient MIMIC-CXR

PPL SW CE RKL ROUGE-L CIDEr RateScore SemScore ROUGE-L CIDEr RateScore SemScore ROUGE-L CIDEr RateScore SemScore

52.57 3.61 54.87 46.93 63.28 5.17 61.32 58.36 51.73 3.63 51.19 43.50 23.72 ✓ 52.44 3.53 56.30 46.66 62.90 5.07 61.33 59.90 51.76 3.65 53.14 43.70 21.07 ✓ ✓ 56.14 4.20 57.40 49.57 66.39 5.54 62.18 66.28 54.12 4.05 53.95 45.57 18.83 ✓ ✓ 52.51 3.41 55.20 45.32 61.71 4.94 61.26 59.55 50.91 3.48 53.31 42.43 20.23 ✓ ✓ ✓ 53.25 3.72 57.25 47.07 63.82 5.21 61.47 63.46 51.48 3.67 53.42 43.86 21.32

to within 3% on average. Taken together, these results confirm that DCD achieves a consistently superior quality-speed trade-off across both block size configurations and against all existing distillation methods.

###### 5.3 Ablation Study of Distillation Component

To investigate the individual contributions of the components in our proposed Direct Conditional Distillation (DCD) strategy, we conduct extensive ablation studies. Specifically, we evaluate the impact of Step-wise Weighting (SW), Cross-Entropy loss for the <eos> token (CE), and Reverse KL divergence (RKL). The results across three benchmarks are summarized in Tab. 2.

Effect of Step-wise Weighting (SW). As shown in row 2 of Tab. 2, SW consistently improves generation stability across all datasets. Upweighting tokens that are unmasked later in the denoising trajectory, which tend to carry stronger inter-token dependencies, reduces PPL from 23.72 to 21.07. RaTEScore also improves steadily, e.g., from 54.87 to 56.30 on CheXpert-Plus, indicating that the reweighting helps the model better capture the conditioning provided by earlier committed tokens.

Effect of <eos> Cross-Entropy Loss (CE). Adding CE supervision on the <eos> token produces the largest single improvement across all settings. As discussed in Sec. 4.2, one-step diffusion models tend to assign low and unstable confidence to <eos> positions, leading to degenerate repetition loops. Explicitly supervising the <eos> token with a one-hot target directly addresses this failure mode, resulting in a substantial gain in ROUGE-L from 52.44 to 56.14 on CheXpert-Plus and CIDEr from 3.65 to 4.05 on MIMIC-CXR. PPL drops further to 18.83, its lowest value across all configurations, confirming that reliable termination is essential for

S[0,1]

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

| |
|---|

t

t

- Figure 5 Effect of training data scale during RAD adaptation. (a) Quality Metric Convergence During RAD. Each colored curve tracks one of four clinical and linguistic metrics across training steps, and dashed horizontal lines mark the corresponding ECHOAR baseline values. (b) Decoupled Convergence of Quality and Throughput. The black curve shows the mean quality score and the red curve shows inference throughput (TPF), and dashed horizontal lines mark respective final values.

overall report quality.

Effect of Reverse KL (RKL). We also replace the forward KL objective with reverse KL (RKL) to examine whether its mode-seeking behavior benefits distillation. As shown in rows 4 and 5 of Tab. 2, adding RKL consistently degrades performance. Compared to the SW-only baseline, CIDEr drops from 3.65 to 3.48 and SemScore from 43.70 to 42.43 on MIMIC-CXR. A likely cause is that clinical reports require the model to cover all plausible findings rather than concentrate probability mass on a single mode, which is the tendency encouraged by reverse KL. Forward KL, which preserves the full teacher distribution, is therefore better suited for this task.

###### 5.4 Data Scale Analysis for RAD

We study how the amount of training data used in the RAD stage affects both model quality and inference throughput, by evaluating checkpoints at different training steps across all benchmarks. Fig. 5 reports four clinical and linguistic metrics alongside inference throughput (TPF) as functions of training steps.

As shown in Fig. 5(a), all four quality metrics follow a rapid climb-and-plateau pattern. Within approximately 60 training steps, performance on all metrics reaches or surpasses the ECHOAR baseline, indicating that RAD transfers the domain knowledge of the pretrained AR model efficiently and requires only a small fraction of the full training data to do so.

Fig. 5(b) reveals that quality and throughput converge at different rates. Quality saturates early (around step 60, corresponding to only 2.2% of the full RAD training data), whereas TPF continues to improve throughout training, increasing from 1.62 to 2.17 (+33.95%). A plausible interpretation is that while crossmodal semantic alignment is established quickly, additional training is needed for the model to stabilize the joint token distribution within each block, which in turn enables more tokens to be committed with high confidence per denoising step and raises throughput.

###### 5.5 Effectiveness of DCD Against Native One-Step Decoding

To isolate the contribution of DCD, we compare ECHO against ECHOBase under native one-step decoding, where ECHOBase is forced to generate all response tokens simultaneously in a single forward pass without any distillation. This setting directly exposes the mean-field bias inherent to block diffusion models and provides a controlled baseline for measuring the gain attributable to DCD alone.

- Table 3 Comparison of ECHO and ECHOBase under native one-step decoding. Green percentages indicate the relative quality improvement of ECHO over ECHOBase across three benchmarks. Bold values mark the higher score between the two model types within each block size and metric.

Block size Model type

CheXpert-Plus ReXGradient MIMIC-CXR

ROUGE-L CIDEr RateScore SemScore ROUGE-L CIDEr RateScore SemScore ROUGE-L CIDEr RateScore SemScore Block 4

ECHOBase 41.39 2.91 56.39 39.17 53.60 4.32 61.72 57.42 41.50 2.76 53.86 35.41 ECHO 56.1436%↑ 4.2044%↑ 57.402%↑ 49.5727%↑ 66.3924%↑ 5.5428%↑ 62.181%↑ 66.2815%↑ 54.1230%↑ 4.0547%↑ 53.950%↑ 45.5729%↑

Block 8

ECHOBase 41.27 2.81 52.12 38.74 53.23 4.19 59.92 54.29 41.08 2.63 51.34 34.18 ECHO 55.2134%↑ 4.1447%↑ 56.859%↑ 51.4033%↑ 65.1322%↑ 5.4831%↑ 60.711%↑ 64.0018%↑ 53.0729%↑ 4.0052%↑ 52.973%↑ 44.8331%↑

- Table 4 Ablation on report data types across all training stages. Each stage contains two settings: normalized and unnormalized reports.

CheXpert-Plus ReXGradient MIMIC-CXR

Stage Data Type

ROUGE-L CIDEr RateScore SemScore ROUGE-L CIDEr RateScore SemScore ROUGE-L CIDEr RateScore SemScore

- Stage I

Normalized 56.89 4.47 59.18 52.94 67.07 5.68 64.39 66.60 54.55 4.32 56.58 46.70 Unnormalized 23.82 1.53 45.59 32.44 25.67 1.49 26.05 40.77 25.06 1.56 42.72 33.54

- Stage II

Normalized 56.90 4.25 59.12 52.64 66.46 5.64 63.02 66.68 54.45 4.16 55.85 46.10 Unnormalized 23.47 1.51 39.60 30.46 24.97 1.45 23.90 36.97 24.54 1.53 37.76 32.64

- Stage III

Normalized 56.14 4.20 57.40 49.57 66.39 5.54 62.18 66.28 54.12 4.05 53.95 45.57 Unnormalized 18.79 1.19 42.08 27.53 19.98 1.20 24.98 31.70 19.61 1.24 37.55 28.02

As shown in Tab. 3, DCD yields consistent and substantial improvements across both block sizes and all three benchmarks. For ECHOblk4 on CheXpert-Plus, ROUGE-L improves by 36% and CIDEr by 44% over the native one-step baseline. SemScore increases by up to 29% on MIMIC-CXR, indicating that DCD not only restores surface-level fluency but also substantially recovers the model’s ability to identify positive pathological findings. These gains are consistent across both block sizes, confirming that the benefit of DCD is not specific to a particular block configuration.

###### 5.6 Analysis on Normalization of Reports

To assess the impact of report normalization, we retrain the pipeline at each stage using unnormalized raw reports while keeping all other settings unchanged. Tab. 4 summarizes the results across all three training stages.

- Impact on Stage I (Continued Pre-training). Normalized data yields substantial improvements in Stage I across all metrics. On CheXpert-Plus, ROUGE-L increases from 23.82 to 56.89 and RaTEScore from 45.59 to 59.18. This large gap indicates that raw reports, which frequently contain inconsistent phrasing and systematically omit negative findings, provide ambiguous supervision that prevents the model from learning reliable visual-textual mappings.
- Impact on Stage II and Stage III (Adaptation and Distillation). The gap persists through Stage II and widens further in Stage III. Without normalization, ROUGE-L collapses to 18.79 on CheXpert-Plus and 19.98 on ReXGradient after distillation, substantially below the already-degraded Stage II unnormalized baseline. With normalization, the model retains strong clinical accuracy across all stages, with SemScore remaining above 45.0 on CheXpert-Plus even after one-step distillation. This progressive degradation under unnormalized training suggests that noisy supervision compounds across stages: the omission bias that distorts pretraining is further amplified by the mean-field bias of block diffusion, and becomes most severe when the student must reproduce the teacher’s full conditional distribution in a single step.

###### 6 Conclusion

We presented ECHO, a discrete diffusion VLM that demonstrates high clinical accuracy and extreme decoding efficiency are simultaneously achievable in automated chest X-ray reporting. By rethinking both the AR-todiffusion conversion and the distillation target, ECHO reduces training cost, eliminates multi-step denoising, and consistently outperforms autoregressive and diffusion-based state-of-the-art methods across standard benchmarks. We hope this work provides a strong foundation for efficient and reliable generation in broader medical multimodal settings beyond CXR reporting.

###### References

- [1] Marianne Arriola, Aaron Gokaslan, Justin T Chiu, Zhihan Yang, Zhixuan Qi, Jiaqi Han, Subham Sekhar Sahoo, and Volodymyr Kuleshov. Block diffusion: Interpolating between autoregressive and diffusion language models. In The Thirteenth International Conference on Learning Representations, 2025.
- [2] Jacob Austin, Daniel D Johnson, Jonathan Ho, Daniel Tarlow, and Rianne Van Den Berg. Structured denoising diffusion models in discrete state-spaces. Advances in neural information processing systems, 2021.
- [3] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025.
- [4] Tiwei Bie, Maosong Cao, Kun Chen, Lun Du, Mingliang Gong, Zhuochen Gong, Yanmei Gu, Jiaqi Hu, Zenan Huang, Zhenzhong Lan, et al. Llada2. 0: Scaling up diffusion language models to 100b. arXiv preprint arXiv:2512.15745, 2025.
- [5] Keshigeyan Chandrasegaran, Armin W. Thomas, Jerome Ku, Federico Berto, Jae Myung Kim, Garyk Brixi, Eric Nguyen, Stefano Massaroli, and Michael Poli. Rnd1: Simple, scalable ar-to-diffusion conversion. 2025.
- [6] Junying Chen, Chi Gui, Ruyi Ouyang, Anningzhe Gao, Shunian Chen, Guiming Hardy Chen, Xidong Wang, Zhenyang Cai, Ke Ji, Xiang Wan, et al. Towards injecting medical visual knowledge into multimodal llms at scale. In Proceedings of the 2024 conference on empirical methods in natural language processing, 2024.
- [7] Zhihong Chen, Maya Varma, Justin Xu, Magdalini Paschali, Dave Van Veen, Andrew Johnston, Alaa Youssef, Louis Blankemeier, Christian Bluethgen, Stephan Altmayer, et al. A vision-language foundation model to enhance efficiency of chest x-ray interpretation. arXiv preprint arXiv:2401.12208, 2024.
- [8] Zigeng Chen, Gongfan Fang, Xinyin Ma, Ruonan Yu, and Xinchao Wang. dparallel: Learnable parallel decoding for dllms. arXiv preprint arXiv:2509.26488, 2025.
- [9] Shuang Cheng, Yuhua Jiang, Zineng Zhou, Dawei Liu, Wang Tao, Linfeng Zhang, Biqing Qi, and Bowen Zhou. Sdar-vl: Stable and efficient block-wise diffusion for vision-language understanding. arXiv preprint arXiv:2512.14068, 2025.
- [10] Jacob K Christopher, Brian R Bartoldson, Tal Ben-Nun, Michael Cardei, Bhavya Kailkhura, and Ferdinando Fioretto. Speculative diffusion decoding: Accelerating language generation through diffusion. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), 2025.
- [11] Dina Demner-Fushman, Marc D Kohli, Marc B Rosenman, Sonya E Shooshan, Laritza Rodriguez, Sameer Antani, George R Thoma, and Clement J McDonald. Preparing a collection of radiology examinations for distribution and retrieval. Journal of the American Medical Informatics Association, 2016.
- [12] Justin Deschenaux and Caglar Gulcehre. Beyond autoregression: Fast llms via self-distillation through time, 2025.
- [13] Xuanzhao Dong, Wenhui Zhu, Xiwen Chen, Zhipeng Wang, Peijie Qiu, Shao Tang, Xin Li, and Yalin Wang. Llada-medv: Exploring large language diffusion models for biomedical image understanding. arXiv preprint arXiv:2508.01617, 2025.
- [14] Nima Fathi, Torsten Scholak, and Pierre-Andre Noel. Unifying autoregressive and diffusion-based sequence generation. In ICLR 2025 Workshop on Deep Generative Model in Machine Learning: Theory, Principle and Efficacy, 2025.
- [15] Shansan Gong, Shivam Agarwal, Yizhe Zhang, Jiacheng Ye, Lin Zheng, Mukai Li, Chenxin An, Peilin Zhao, Wei Bi, Jiawei Han, et al. Scaling diffusion language models via adaptation from autoregressive models. In The Thirteenth International Conference on Learning Representations, 2025.
- [16] Google DeepMind. Gemini 3 pro model card, 2025.
- [17] Xiaochuang Han, Sachin Kumar, and Yulia Tsvetkov. Ssd-lm: Semi-autoregressive simplex-based diffusion language model for text generation and modular control. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics, 2023.
- [18] Zhanqiu Hu, Jian Meng, Yash Akhauri, Mohamed S Abdelfattah, Jae-sun Seo, Zhiru Zhang, and Udit Gupta. Accelerating diffusion language model inference via efficient kv caching and guided diffusion. arXiv e-prints, 2025.

- [19] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024.
- [20] Jeremy Irvin, Pranav Rajpurkar, Michael Ko, Yifan Yu, Silviana Ciurea-Ilcus, Chris Chute, Henrik Marklund, Behzad Haghgoo, Robyn Ball, Katie Shpanskaya, et al. Chexpert: A large chest radiograph dataset with uncertainty labels and expert comparison. In Proceedings of the AAAI conference on artificial intelligence, 2019.
- [21] Songtao Jiang, Yuan Wang, Sibo Song, Tianxiang Hu, Chenyi Zhou, Bin Pu, Yan Zhang, Zhibo Yang, Yang Feng, Joey Tianyi Zhou, et al. Hulu-med: A transparent generalist model towards holistic medical vision-language understanding. arXiv preprint arXiv:2510.08668, 2025.
- [22] Alistair EW Johnson, Tom J Pollard, Seth J Berkowitz, Nathaniel R Greenbaum, Matthew P Lungren, Chihying Deng, Roger G Mark, and Steven Horng. Mimic-cxr, a de-identified publicly available database of chest radiographs with free-text reports. Scientific data, 2019.
- [23] Seowoo Lee, Jiwon Youn, Hyungjin Kim, Mansu Kim, and Soon Ho Yoon. Cxr-llava: a multimodal large language model for interpreting chest x-ray images. European Radiology, 2025.
- [24] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, et al. Llava-onevision: Easy visual task transfer. Transactions on Machine Learning Research, 2024.
- [25] Chunyuan Li, Cliff Wong, Sheng Zhang, Naoto Usuyama, Haotian Liu, Jianwei Yang, Tristan Naumann, Hoifung Poon, and Jianfeng Gao. Llava-med: Training a large language-and-vision assistant for biomedicine in one day. Advances in Neural Information Processing Systems, 2023.
- [26] Feng Li, Renrui Zhang, Hao Zhang, Yuanhan Zhang, Bo Li, Wei Li, Zejun Ma, and Chunyuan Li. Llava-nextinterleave: Tackling multi-image, video, and 3d in large multimodal models. arXiv preprint arXiv:2407.07895, 2024.
- [27] Guanghao Li, Zhihui Fu, Min Fang, Qibin Zhao, Ming Tang, Chun Yuan, and Jun Wang. Diffuspec: Unlocking diffusion language models for speculative decoding. arXiv preprint arXiv:2510.02358, 2025.
- [28] Shufan Li, Konstantinos Kallidromitis, Hritik Bansal, Akash Gokul, Yusuke Kato, Kazuki Kozuka, Jason Kuen, Zhe Lin, Kai-Wei Chang, and Aditya Grover. Lavida: A large diffusion language model for multimodal understanding. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.
- [29] Yihao Liang, Ze Wang, Hao Chen, Ximeng Sun, Jialian Wu, Xiaodong Yu, Jiang Liu, Emad Barsoum, Zicheng Liu, and Niraj K Jha. Cd4lm: Consistency distillation and adaptive decoding for diffusion language models. arXiv preprint arXiv:2601.02236, 2026.
- [30] Chin-Yew Lin. Rouge: A package for automatic evaluation of summaries. In Text summarization branches out, 2004.
- [31] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 2023.
- [32] Zhiyuan Liu, Yicun Yang, Yaojie Zhang, Junjie Chen, Chang Zou, Qingyuan Wei, Shaobo Wang, and Linfeng Zhang. dllm-cache: Accelerating diffusion large language models with adaptive caching. arXiv preprint arXiv:2506.06295, 2025.
- [33] Shen Nie, Fengqi Zhu, Zebin You, Xiaolu Zhang, Jingyang Ou, Jun Hu, JUN ZHOU, Yankai Lin, Ji-Rong Wen, and Chongxuan Li. Large language diffusion models. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.
- [34] Jiazhen Pan, Che Liu, Junde Wu, Fenglin Liu, Jiayuan Zhu, Hongwei Bran Li, Chen Chen, Cheng Ouyang, and Daniel Rueckert. Medvlm-r1: Incentivizing medical reasoning capability of vision-language models (vlms) via reinforcement learning. In International Conference on Medical Image Computing and Computer-Assisted Intervention, 2025.
- [35] Chantal Pellegrini, Ege Özsoy, Benjamin Busam, Nassir Navab, and Matthias Keicher. Radialog: A large visionlanguage model for radiology report generation and conversational assistance. arXiv preprint arXiv:2311.18681, 2023.
- [36] Yu-Yang Qian, Junda Su, Lanxiang Hu, Peiyuan Zhang, Zhijie Deng, Peng Zhao, and Hao Zhang. d3llm: Ultrafast diffusion llm using pseudo-trajectory distillation. arXiv preprint arXiv:2601.07568, 2026.

- [37] Khaled Saab, Tao Tu, Wei-Hung Weng, Ryutaro Tanno, David Stutz, Ellery Wulczyn, Fan Zhang, Tim Strother, Chunjong Park, Elahe Vedadi, et al. Capabilities of gemini models in medicine. arXiv preprint arXiv:2404.18416, 2024.
- [38] Subham Sahoo, Marianne Arriola, Yair Schiff, Aaron Gokaslan, Edgar Marroquin, Justin Chiu, Alexander Rush, and Volodymyr Kuleshov. Simple and effective masked diffusion language models. Advances in Neural Information Processing Systems, 2024.
- [39] Andrew Sellergren, Sahar Kazemzadeh, Tiam Jaroensri, Atilla Kiraly, Madeleine Traverse, Timo Kohlberger, Shawn Xu, Fayaz Jamil, Cían Hughes, Charles Lau, et al. Medgemma technical report. arXiv preprint arXiv:2507.05201, 2025.
- [40] Jiaxin Shi, Kehang Han, Zhe Wang, Arnaud Doucet, and Michalis Titsias. Simplified and generalized masked diffusion for discrete data. Advances in neural information processing systems, 2024.
- [41] Akshay Smit, Saahil Jain, Pranav Rajpurkar, Anuj Pareek, Andrew Y Ng, and Matthew Lungren. Combining automatic labelers and expert annotations for accurate radiology report labeling using bert. In Proceedings of the 2020 conference on empirical methods in natural language processing (EMNLP), pages 1500–1519, 2020.
- [42] Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.
- [43] Omkar Chakradhar Thawakar, Abdelrahman M Shaker, Sahal Shaji Mullappilly, Hisham Cholakkal, Rao Muhammad Anwer, Salman Khan, Jorma Laaksonen, and Fahad Khan. Xraygpt: Chest radiographs summarization using large medical vision-language models. In Proceedings of the 23rd workshop on biomedical natural language processing, 2024.
- [44] Tao Tu, Shekoofeh Azizi, Danny Driess, Mike Schaekermann, Mohamed Amin, Pi-Chuan Chang, Andrew Carroll, Charles Lau, Ryutaro Tanno, Ira Ktena, et al. Towards generalist biomedical ai. Nejm Ai, 2024.
- [45] Ramakrishna Vedantam, C Lawrence Zitnick, and Devi Parikh. Cider: Consensus-based image description evaluation. In Proceedings of the IEEE conference on computer vision and pattern recognition, 2015.
- [46] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024.
- [47] Xu Wang, Chenkai Xu, Yijie Jin, Jiachun Jin, Hao Zhang, and Zhijie Deng. Diffusion llms can do faster-than-ar inference via discrete diffusion forcing. arXiv preprint arXiv:2508.09192, 2025.
- [48] Chaoyi Wu, Xiaoman Zhang, Ya Zhang, Hui Hui, Yanfeng Wang, and Weidi Xie. Towards generalist foundation model for radiology by leveraging web-scale 2d&3d medical data. Nature Communications, 2025.
- [49] Chengyue Wu, Hao Zhang, Shuchen Xue, Shizhe Diao, Yonggan Fu, Zhijian Liu, Pavlo Molchanov, Ping Luo, Song Han, and Enze Xie. Fast-dllm v2: Efficient block-diffusion llm. arXiv preprint arXiv:2509.26328, 2025.
- [50] Chengyue Wu, Hao Zhang, Shuchen Xue, Zhijian Liu, Shizhe Diao, Ligeng Zhu, Ping Luo, Song Han, and Enze Xie. Fast-dllm: Training-free acceleration of diffusion llm by enabling kv cache and parallel decoding. arXiv preprint arXiv:2505.22618, 2025.
- [51] Minkai Xu, Tomas Geffner, Karsten Kreis, Weili Nie, Yilun Xu, Jure Leskovec, Stefano Ermon, and Arash Vahdat. Energy-based diffusion language models for text generation. In The Thirteenth International Conference on Learning Representations, 2025.
- [52] Weiwen Xu, Hou Pong Chan, Long Li, Mahani Aljunied, Ruifeng Yuan, Jianyu Wang, Chenghao Xiao, Guizhen Chen, Chaoqun Liu, Zhaodonghui Li, et al. Lingshu: A generalist foundation model for unified multimodal medical understanding and reasoning. arXiv preprint arXiv:2506.07044, 2025.
- [53] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.
- [54] Ling Yang, Zhanyu Wang, Zhenghao Chen, Xinyu Liang, and Luping Zhou. Medxchat: A unified multimodal large language model framework towards cxrs understanding and generation. In 2025 IEEE 22nd International Symposium on Biomedical Imaging (ISBI), 2025.

- [55] Jiacheng Ye, Zhihui Xie, Lin Zheng, Jiahui Gao, Zirui Wu, Xin Jiang, Zhenguo Li, and Lingpeng Kong. Dream 7b: Diffusion large language models. arXiv preprint arXiv:2508.15487, 2025.
- [56] Jaehoon Yoo, Wonjung Kim, and Seunghoon Hong. Redi: Rectified discrete flow. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.
- [57] Zebin You, Shen Nie, Xiaolu Zhang, Jun Hu, Jun Zhou, Zhiwu Lu, Ji-Rong Wen, and Chongxuan Li. Llada-v: Large language diffusion models with visual instruction tuning. arXiv preprint arXiv:2505.16933, 2025.
- [58] Runpeng Yu, Xinyin Ma, and Xinchao Wang. Dimple: Discrete diffusion multimodal large language model with parallel decoding. arXiv preprint arXiv:2505.16990, 2025.
- [59] Kai Zhang, Rong Zhou, Eashan Adhikarla, Zhiling Yan, Yixin Liu, Jun Yu, Zhengliang Liu, Xun Chen, Brian D Davison, Hui Ren, et al. A generalist vision–language foundation model for diverse biomedical tasks. Nature medicine, 2024.
- [60] Tunyu Zhang, Xinxi Zhang, Ligong Han, Haizhou Shi, Xiaoxiao He, Zhuowei Li, Hao Wang, Kai Xu, Akash Srivastava, Vladimir Pavlovic, et al. T3d: Few-step diffusion language models via trajectory self-distillation with direct discriminative optimization. arXiv preprint arXiv:2602.12262, 2026.
- [61] Xiaoman Zhang, Julián N Acosta, Josh Miller, Ouwen Huang, and Pranav Rajpurkar. Rexgradient-160k: A large-scale publicly available dataset of chest radiographs with free-text reports. arXiv preprint arXiv:2505.00228, 2025.
- [62] Yichi Zhang, Alex Schwing, and Zhizhen Zhao. Variational masked diffusion models. arXiv preprint arXiv:2510.23606, 2025.
- [63] Weike Zhao, Chaoyi Wu, Xiaoman Zhang, Ya Zhang, Yanfeng Wang, and Weidi Xie. Ratescore: A metric for radiology report generation. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, 2024.

## Appendix

This supplementary material provides additional details and results to complement the main paper, organized as follows.

- • Sec. A: Qualitative Examples. We present qualitative examples illustrating the effect of DCD on generation quality and the diagnostic capability of ECHO.
- • Sec. B: Experimental Details. We describe the data preprocessing pipeline (Sec. B.1), training configurations for all three stages (Sec. B.2), definitions and adaptations of all evaluation metrics (Sec. B.3), and the prompt templates used throughout the study (Sec. B.4).
- • Sec. C: Implementation of Baselines. We provide implementation details of all baseline methods, including hyperparameter settings and training loss combinations.
- • Sec. D: Detailed Results. We report extended quantitative results with per-dataset breakdowns across CheXpert-Plus, ReXGradient, and MIMIC-CXR.
- • Sec. E: Fused Block KV Cache Analysis. We formally show that Fused Block KV Cache preserves total FLOPs while halving the number of forward passes for one-step-per-block decoding.

###### A Qualitative examples

We present two sets of qualitative comparisons. Fig. A.3 analyzes differences in generation quality across model variants, and Fig. A.4 demonstrates the ability of ECHO to detect positive pathological findings.

- Fig. A.3 compares reports generated by ECHO-AR, ECHOBase, ECHO-Base_onestep, ECHOblk8, and ECHOblk4, where ECHO-Base_onestep denotes ECHOBase forced to decode in a single step without distillation. It is obvious that ECHOBase produces fluent and coherent reports without noticeable errors. In contrast, ECHO-Base_onestep exhibits severe token disorder and repetition caused by mean-field factorization. After applying DCD, ECHOblk8 and ECHOblk4 largely eliminate these artifacts, with only minor residual repetition. This comparison confirms that DCD effectively mitigates mean-field bias and yields more stable one-step generation.
- Fig. A.4 shows examples where ECHO correctly identifies abnormal findings, with correct predictions highlighted in green. Across these cases, ECHO accurately detects multiple abnormalities and describes them using appropriate clinical terminology, demonstrating reliable diagnostic capability for CXR report generation.

###### B Experimetal Details

###### B.1 Data

We aggregate reports from MIMIC-CXR [22], CheXpert-Plus [20], ReXGradient [61], and IU-Xray [11] to form a unified corpus for CXR report generation, and apply a five-stage preprocessing pipeline to ensure consistency and training stability across sources.

- 1. Modality filtering. we remove non-CXR studies and samples with incomplete clinical descriptions using clinical entity extraction.
- 2. Report standardization and normalization. We use a prompt-based rewriting pipeline with BaichuanM232B to standardize clinical terminology, enforce a structured Findings–Impression format with JSON output, and normalize each report by explicitly enumerating all negative findings in order.
- 3. Semantic deduplication. we embed reports with Qwen3-Embedding-8B [53] and remove near-duplicates using a cosine-similarity threshold. After deduplication, the corpus contains approximately 260k reports from MIMIC-CXR, 250k from CheXpert-Plus, 190k from ReXGradient, and 30k from IU-Xray.
- 4. Bilingual augmentation. we randomly sample 50% of the original reports and translate them from English into Chinese using BaichuanM2-32B. Specific prompts are used to preserve medical terminology and report structure.

- Table A.1 Detailed per-metric comparison on CheXpert-Plus across Chinese and English evaluation. Green percentages indicate the relative performance drop of ECHO compared to ECHOAR. Bold values denote the best result among all distillation methods.

Chinese English

Methods

ROUGE-L METEOR CIDEr RateScore SemScore ROUGE-L METEOR CIDEr RateScore SemScore

Proprietary general models

Gemini3-Pro [16] 26.79 29.40 2.14 41.87 32.21 27.12 25.71 0.92 39.84 27.23 Qwen3-Max [3] 26.33 30.17 2.12 43.20 29.88 28.04 26.82 0.93 40.06 28.60

Autoregressive Medical Models

LLaVA-Med [25] 10.58 5.50 1.13 10.03 11.69 3.25 0.56 0.18 10.00 11.09 Lingshu-7B [52] 19.96 16.81 1.50 28.46 26.51 23.93 14.19 0.73 34.22 28.56 Hulu-Med-7B [21] 21.18 18.45 1.67 22.09 27.34 23.57 14.34 0.63 30.41 19.52 MedGemma-27B [39] 15.56 20.76 0.95 34.83 29.91 25.04 23.59 0.62 42.84 33.68 Lingshu-32B [52] 16.61 12.92 1.39 27.68 21.25 24.19 15.86 0.73 34.16 27.39 Hulu-Med-32B [21] 19.93 14.37 1.45 22.85 22.57 19.00 11.06 0.47 31.49 21.57

Diffusion Medical Methods

LLaDA-MedV [13] 2.79 3.13 0.16 34.83 18.21 12.59 11.55 0.28 18.60 17.31 ECHOBase 55.30 54.52 4.62 54.68 50.69 58.50 52.02 3.88 63.57 54.59

CD4LM [29] 44.3320%↓ 49.579%↓ 3.7020%↓ 51.875%↓ 44.7412%↓ 54.098%↓ 47.259%↓ 3.0721%↓ 59.946%↓ 42.9821%↓ d3LLM [36] 34.5937%↓ 45.1417%↓ 2.7740%↓ 51.206%↓ 39.1323%↓ 53.499%↓ 47.459%↓ 3.0821%↓ 59.576%↓ 41.5724%↓ dParallel [8] 40.6127%↓ 41.9423%↓ 3.4725%↓ 36.0334%↓ 35.7030%↓ 43.8325%↓ 38.9425%↓ 2.8726%↓ 44.9129%↓ 37.7431%↓ T3D [60] 51.716%↓ 54.690%↓ 4.385%↓ 52.684%↓ 48.115%↓ 58.290%↓ 51.840%↓ 3.753%↓ 61.124%↓ 51.635%↓ ECHOblk8 52.455%↓ 51.675%↓ 4.473%↓ 52.953%↓ 48.894%↓ 57.981%↓ 49.205%↓ 3.802%↓ 60.754%↓ 53.921%↓ ECHOblk4 54.661%↓ 53.153%↓ 4.591%↓ 53.762%↓ 48.115%↓ 57.622%↓ 50.463%↓ 3.812%↓ 61.044%↓ 51.037%↓

5. Supplemental general multimodal data. we incorporate LLaVA-ReCap-558k [24] to preserve general vision-language grounding throughout the training process.

This pipeline yields a large-scale, standardized, and bilingual CXR report corpus.

###### B.2 Training details

All training stages are conducted on Alibaba PPUs. To balance the information retention of high-resolution chest X-ray images and GPU memory consumption, we enforce a maximum pixel constraint of 2,250,000 throughout all stages.

- Stage 1 (AR continual pretraining). We perform full-parameter SFT on the base model Lingshu [52]. We employ the AdamW optimizer with a learning rate of 1 × 10−5, weight decay of 0.01, and a linear warmup over the first 0.03 epochs.
- Stage 2 (Diffusion adaptation). To accommodate the GPU memory overhead introduced by the extended vision token sequences of high-resolution chest X-ray images, we adopt a neat-packing training strategy. This stage uses approximately 2.2% of the Stage 1 SFT data, with the vision encoder and projector frozen and only the LLM backbone trained.
- Stage 3 (Distillation). We randomly sample 2.3% of the SFT dataset as the distillation corpus. This stage also employs neat-packing-based training, with the vision encoder and projector kept frozen.

###### B.3 Metric Details

We evaluate generated chest X-ray reports across three dimensions using five established metrics. The calculation procedure and any task-specific adaptations for each metric are described below.

ROUGE-L [30] measures surface-level text overlap between the generated report and the reference. It computes the longest common subsequence (LCS), capturing sentence-level structure and word-order similarity.

CIDEr [45] evaluates report quality by emphasizing rare, clinically informative terms. It computes the cosine similarity between TF-IDF-weighted n-grams of the generated and reference reports, giving greater weight

- Table A.2 Detailed per-metric comparison on RexGradient across Chinese and English evaluation. Green percentages indicate the relative performance drop of ECHO compared to ECHOAR. Bold values denote the best result among all distillation methods.

Methods

Chinese English

ROUGE-L METEOR CIDEr RateScore SemScore ROUGE-L METEOR CIDEr RateScore SemScore

Proprietary general models

Gemini3-Pro [16] 34.10 36.70 2.60 41.78 42.59 30.10 26.34 1.03 25.26 35.46 Qwen3-Max [3] 31.32 36.62 2.48 45.86 39.13 29.67 29.84 0.97 32.38 37.14

Autoregressive Medical Models

LLaVA-Med [25] 11.35 5.92 1.23 10.01 18.46 3.37 0.62 0.21 10.00 15.62 Lingshu-7B [52] 22.24 19.27 1.68 20.99 32.56 25.01 14.87 0.83 19.51 37.07 Hulu-Med-7B [21] 21.35 18.96 1.70 22.48 24.36 21.96 14.48 0.58 23.06 22.76 MedGemma-27B [39] 16.97 23.29 1.08 24.16 34.87 29.85 26.64 0.72 27.84 40.61 Lingshu-32B [52] 18.45 14.29 1.53 21.07 27.75 25.75 20.06 0.87 19.58 35.22 Hulu-Med-32B [21] 21.46 17.15 1.63 22.72 28.06 23.17 15.49 0.59 23.30 25.28

Diffusion Medical Methods

LLaDA-MedV [13] 2.87 3.91 0.13 28.05 14.46 16.57 14.30 0.34 8.33 29.73 ECHOBase 64.59 64.53 5.65 53.09 61.64 68.32 65.18 5.63 72.95 71.72

CD4LM [29] 54.8315%↓ 58.3310%↓ 4.7416%↓ 48.569%↓ 59.244%↓ 62.628%↓ 58.8510%↓ 4.1426%↓ 68.157%↓ 52.0027%↓ d3LLM [36] 45.5330%↓ 53.4217%↓ 3.7933%↓ 46.5312%↓ 53.1114%↓ 62.828%↓ 60.617%↓ 4.5619%↓ 69.475%↓ 53.8825%↓ dParallel [8] 47.2627%↓ 47.7826%↓ 4.0928%↓ 37.4829%↓ 45.9126%↓ 50.2127%↓ 48.2926%↓ 3.9430%↓ 53.5227%↓ 50.1130%↓

- T3D [60] 61.565%↓ 63.402%↓ 5.326%↓ 48.289%↓ 59.254%↓ 67.172%↓ 64.581%↓ 5.139%↓ 68.696%↓ 64.0611%↓ ECHOblk8 62.443%↓ 62.024%↓ 5.444%↓ 50.655%↓ 60.362%↓ 67.821%↓ 64.052%↓ 5.532%↓ 69.205%↓ 67.646%↓ ECHOblk4 64.740%↓ 63.172%↓ 5.512%↓ 52.691%↓ 61.600%↓ 68.040%↓ 64.651%↓ 5.571%↓ 71.672%↓ 70.961%↓

to infrequent but diagnostically relevant terms over common stop words.

Modified RaTEScore [63] measures clinical entity-level accuracy. A Named Entity Recognition (NER) module extracts medical entities and classifies them into five types: Anatomy, Abnormality, Disease, NonAbnormality, and Non-Disease. These entities are mapped to dense embeddings via a medical text encoder to resolve synonymy, and the final score is computed as a weighted F1 using a predefined clinical importance matrix.

However, directly applying the standard metric is not appropriate in our setting. Since our normalized test set explicitly enumerates negative findings (see Sec. B.1), conventional CXR report generation models would be unfairly penalized for not doing the same. In clinical practice, what truly matters is whether a model correctly identifies positive pathologies. We therefore set the importance weights of negative-related entity types (Non-Abnormality and Non-Disease) to zero, so that all models are evaluated solely on their ability to identify true positive findings.

SemScore [41] evaluates semantic similarity between the generated and reference reports beyond exact lexical overlap. It encodes both reports into dense embedding vectors via a pre-trained sentence transformer, and uses their cosine similarity as the evaluation score.

Perplexity quantifies the fluency of the generated text. It is computed as the exponentiated average negative log-likelihood assigned to the token sequence by a standard causal language model. In addition to quality metrics, Tab. 1 also reports two efficiency metrics to evaluate decoding speed. TPF (Tokens Per Forward pass) measures the theoretical speedup relative to the AR baseline. It is computed

- as the number of decoded tokens divided by the number of model forward passes during decoding.

TPS (Tokens Per Second) measures the practical decoding throughput. It is computed as the number of tokens generated during the decoding stage divided by the elapsed time.

Table A.3 Detailed per-metric comparison on MIMIC across Chinese and English evaluation. Green percentages indicate the relative performance drop of ECHO compared to ECHOAR. Bold values denote the best result among all distillation methods.

Chinese English

Methods

ROUGE-L METEOR CIDEr RateScore SemScore ROUGE-L METEOR CIDEr RateScore SemScore

Proprietary general models

Gemini3-Pro [16] 25.86 28.37 2.13 41.46 30.62 28.30 26.28 0.98 41.38 30.28 Qwen3-Max [3] 25.62 29.10 2.08 41.93 26.95 28.10 27.21 0.94 39.64 27.69

Autoregressive Medical Models

LLaVA-Med [25] 10.32 5.42 1.12 10.02 10.42 3.05 0.55 0.18 10.00 8.02 Lingshu-7B [52] 18.93 16.27 1.48 28.69 24.04 25.55 14.79 0.83 36.69 29.84 Hulu-Med-7B [21] 21.64 16.37 1.68 24.36 23.56 22.88 14.78 0.68 21.41 21.04 MedGemma-27B [39] 15.46 20.82 0.98 33.74 27.35 26.14 24.06 0.65 43.22 33.13 Lingshu-32B [52] 16.52 12.63 1.41 28.35 19.65 24.94 14.61 0.79 36.45 28.62 Hulu-Med-32B [21] 18.25 13.51 1.46 26.36 15.41 17.66 10.53 0.52 22.72 21.06

Diffusion Medical Methods

LLaDA-MedV [13] 2.86 3.85 0.13 37.70 15.95 16.33 13.18 0.33 17.24 18.57 ECHOBase 52.59 52.08 4.51 50.98 43.63 56.32 51.40 3.81 60.71 48.58

CD4LM [29] 40.8522%↓ 46.5811%↓ 3.4324%↓ 47.167%↓ 39.789%↓ 51.568%↓ 45.3312%↓ 2.8326%↓ 57.076%↓ 37.7422%↓ d3LLM [36] 30.7342%↓ 42.3319%↓ 2.4745%↓ 45.5911%↓ 30.1431%↓ 51.548%↓ 45.6011%↓ 2.8226%↓ 56.417%↓ 38.2621%↓ dParallel [8] 37.8428%↓ 39.4924%↓ 3.2328%↓ 36.0329%↓ 27.9536%↓ 42.1225%↓ 37.7227%↓ 2.6929%↓ 44.9126%↓ 34.3429%↓ T3D [60] 48.657%↓ 51.840%↓ 4.158%↓ 47.267%↓ 39.0610%↓ 56.130%↓ 50.592%↓ 3.567%↓ 57.515%↓ 45.965%↓ ECHOblk8 50.893%↓ 48.637%↓ 4.373%↓ 48.455%↓ 42.772%↓ 55.252%↓ 47.348%↓ 3.635%↓ 57.505%↓ 46.883%↓ ECHOblk4 52.790%↓ 50.273%↓ 4.432%↓ 49.623%↓ 43.540%↓ 55.462%↓ 48.745%↓ 3.664%↓ 58.294%↓ 47.612%↓

###### B.4 Prompt templates

We provide the prompt templates used at each stage of our study below. Fig. A.1 shows the template for report structural standardization and report normalization. Fig. A.2 shows the templates for translating Chinese reports into English, as required by the RaTEScore and SemScore evaluation pipelines. For both inference and evaluation, we use the following unified instruction prompt: “Review this chest X-ray and write a report. Use this format: Findings: {}, Impression: {}.”

###### C Implementation of baselines

Autoregressive (AR) baselines. We adopt greedy decoding by setting the temperature to 0. The maximum number of generated tokens is set to 512. KV cache and FlashAttention are used to accelerate the decoding process.

LLaDA-MedV. Following the default configuration of LLaDA-MedV [13], we employ a low-confidence remasking strategy with the response length set to L = 256, the block length set to B = 64, and the total number of sampling steps set to Z = 256. The decoding process is further accelerated using its default fast_dllm configuration for efficient inference.

Other distillation baselines. To further compare the efficiency of DCD, we implement several additional distillation baselines, all built upon the same base model ECHO-Baseblk8 and training data. To enable direct comparison of inference speedup, all baselines adopt the semi-autoregressive decoding scheme aligned with BD3LM [1]. We categorize them into three groups:

• Trajectory-based distillation. Methods like dParallel [8] directly fit the predictions of the current noisy block to pseudo-labels produced by multi-step denoising via cross-entropy loss. Following its official implementation, we apply both a CE loss and an Entropy Minimization loss, which encourage trajectory self-consistency and output confidence, respectively. We set dparallel_entropy_weight to 2.0 and dparallel_temperature to 0.5. While this approach maintains trajectory consistency, directly applying CE loss does not resolve the mean-field bias problem.

###### Prompt for standardization and normalizaiton

You are an experienced radiologist tasked with rewriting non-standard medical imaging reports. These non-standard reports typically exhibit the following issues:

- 1. Incomplete "Findings" sections—often omitting descriptions of normal (negative) findings.
- 2. Inclusion of interpretive or inferential statements within the "Findings" section that should belong in the "Impression" section.

Your input will be an original report containing both "Findings" and "Impression" sections. Your output must be a standardized report in JSON format, without any additional explanations or comments.

Standardization requirements:

- 1. Findings:

- - Structure the findings in the following anatomical order: thorax, mediastinum and trachea, lung fields, cardiac silhouette, hila, diaphragm and costophrenic angles, and bony structures.
- - Retain all organ/structure descriptions present in the original report.
- - Retain any additional relevant details (e.g., presence of tubes or lines).
- - Retain all positive or abnormal findings from the original report.
- - Supplement any anatomical regions not mentioned in the original report by explicitly stating they are unremarkable (i.e., assume negative if not mentioned).
- - Remove any interpretive or inferential statements that belong in the Impression, not the Findings.

- 2. Impression:

- - Retain the original impression.
- - You may add diagnostic conclusions or clinical recommendations based on the standardized findings and original impression. Below are examples of standardized negative reports for reference:

- Standardized Negative Report (1): Findings: The thorax is symmetric bilaterally. The mediastinum and trachea are midline. Pulmonary vasculature is clear and follows a normal course; no definite abnormal opacities are seen in either lung field. Cardiac silhouette is within normal limits. Hilar shadows are not enlarged. Both diaphragms and costophrenic angles appear normal. No osseous abnormalities are identified in the imaged areas. Impression: No significant abnormalities detected in the lungs, heart, or diaphragm.
- Standardized Negative Report (2): Findings: Bilateral thorax is symmetric. Mediastinum is midline. Pulmonary markings are clear. No definite focal opacities or consolidations are seen in the lung fields. Hila and cardiac silhouette are normal. Diaphragms are smooth and well-defined; costophrenic angles are sharp. Bony thorax appears normal. Impression: No significant abnormalities on chest radiograph.

Here is the medical imaging report to be rewritten: input {content}

Please generate the standardized medical imaging report according to the above guidelines. outputformat <json> {{ "findings": "...", "impression": "..." }} </json>

- Figure A.1 Prompt template for report standardization and normalization. The prompt instructs the model to rewrite a raw radiology report by: (1) separating the content into Findings and Impression sections with standardized clinical terminology, (2) explicitly enumerating all negative findings for clinical completeness, and (3) returning the result in a structured JSON format.

- • Distribution-based distillation. Methods like CD4LM [29] align the predicted distribution of the same block at a higher noise level (more tokens masked) with its distribution at a lower noise level (fewer tokens masked), using KL divergence as the alignment objective. Following its official implementation, we implement Discrete-Space Consistency Distillation using a combination of CE loss and KL divergence. We set cdlm_lambda to 0.7 and cdlm_temperature to 2.0.
- • Optimization-based distillation. Methods like T3D [60] and d3LLM [36] optimize the model’s predicted distribution on the multi-step teacher pseudo-label to correct the mean-field bias. Specifically, T3D adopts a DPO-style training objective that rewards distributions collected on multi-step teacher pseudolabels, while penalizing distributions produced by the model’s own few-step predicted labels. For both methods, we implement their training loss combinations following their original codebases.

###### D Detailed results

Per-dataset results. Tab. A.1, Tab. A.2, and Tab. A.3 present a detailed per-metric comparison of our method against proprietary general models, autoregressive medical models, and other diffusion-based methods on each test set. ECHOBase achieves state-of-the-art performance across all datasets, and ECHO further maintains this performance with minimal quality degradation while reaching the maximum inference speedup.

###### Prompt for translation

Your task is to convert the following Chinese chest X-ray medical report into a standardized English radiology report. Please strictly follow these guidelines:

- 1. Extract and summarize the objective imaging observations into the FINDINGS section.

- - FINDINGS must contain only descriptive radiological observations.
- - Do NOT include any diagnostic conclusions in FINDINGS.

- 2. Extract and summarize the diagnostic interpretation into the IMPRESSION section.

- - IMPRESSION should reflect the clinician’s overall diagnostic assessment.
- - Do NOT introduce any new diagnoses that are not explicitly stated in the original report.

- 3. Translate the content accurately into English using standard radiology terminology.

- - Avoid literal word-by-word translation.
- - Use clinically accepted expressions (e.g., “increased bronchovascular markings” instead of “lung texture thickened”).

- 4. Preserve all expressions of uncertainty (e.g., “suggestive of”, “cannot exclude”, “likely”, “consider”).

- Do NOT convert uncertain statements into definitive conclusions.

- 5. If the original Chinese report contains only FINDINGS or only IMPRESSION, do NOT fabricate the missing section.

- Leave the missing section empty if necessary.

- 6. Standardized Output Format (strict): FINDINGS: <content>

IMPRESSION: <content>

- 7. Wrap your final output strictly within: ```output <your standardized report>

```

- 8. Output ONLY the standardized English report.

- - Do NOT include any explanation, notes, or additional commentary. Here is the Chinese medical report to be processed: ```input {content} ```
- - If the original content is ambiguous, incomplete, or poorly structured, you must translate it faithfully without attempting to correct or improve it. here is the output format example: output sample 1: ```output FINDINGS: Increased and thickened bronchovascular markings in both lungs without significant consolidation. No nodular opacities at the hilum. Normal cardiac size. Eggshell-like hyperdense calcification at the aortic knob. Sharp costophrenic angles and smooth diaphragmatic surfaces IMPRESSION: Atherosclerosis of the aorta.

```

- Figure A.2 Prompt template for translating Chinese radiology reports into English, used to prepare evaluation data for the RaTEScore and SemScore metrics.

###### E Fused Block KV Cache Analysis

We show that Fused Block KV Cache preserves the total FLOPs of Vanilla Block KV Cache while halving the number of forward passes from 2N to N for one-step-per-block decoding.

Notation. Let P denote the prompt length (including vision and instruction tokens), B the block size, and N the number of response blocks. For a Transformer forward pass processing q query tokens where each token attends to ℓ context positions, the FLOPs can be written as F(q,ℓ) = q·g(ℓ), where g(ℓ) is the per-token cost

- at context length ℓ. This linear decomposition in q holds because all major Transformer operations (QKV projection, attention, output projection, and FFN) scale linearly with the number of query tokens.

Vanilla Block KV Cache. For each block n (n = 0,...,N−1) to be decoded, the KV cache holds P +nB entries from the prompt and all previously decoded blocks. Two separate forward passes are performed: (i) Denoise: q=B masked tokens, each attending to P + nB + B = P + (n+1)B positions. (ii) KV Update: q=B decoded tokens, each attending to P + nB + B = P + (n+1)B positions, to compute and cache their key-value states. This yields 2N forward passes in total.

Fused Block KV Cache. Block 0 is handled with a standard denoise forward (q=B), and its KV update is deferred. For each subsequent block n (n = 1,...,N−1), the deferred KV update from block n−1 and the denoise of block n are merged into one fused forward (q=2B). This yields N forward passes in total.

FLOPs equivalence. Each fused forward for block n (n ≥ 1) replaces exactly two Vanilla operations: the KV update of block n−1 and the denoise of block n. At this point, the KV cache holds P +(n−1)B entries, since the KV update for block n−1 has not yet occurred. Under the block-causal attention mask, each token in the fused forward attends to the same context as in the corresponding Vanilla pass:

- • The B decoded tokens from block n−1 attend to P + (n−1)B + B = P + nB positions, matching the Vanilla KV update for block n−1.
- • The B masked tokens for block n attend to P + (n−1)B + B + B = P + (n+1)B positions, matching the Vanilla denoise for block n.

The combined FLOPs of the two Vanilla passes are:

Fvanilla = B · g(P + nB) + B · g P + (n+1)B . (4) Since the fused forward processes the same 2B tokens with identical per-token contexts, its FLOPs are:

Ffused = B · g(P + nB) + B · g P + (n+1)B = Fvanilla . (5)

This equality holds for every n ≥ 1. Therefore, Fused Block KV Cache introduces no additional FLOPs while reducing the number of forward passes from 2N to N, directly lowering inference latency.

CXR and GT

Findings: The thorax is symmetric bilaterally. The mediastinum and trachea are midline. Pulmonary vasculature is clear and follows a normal course; no definite abnormal opacities are seen in either lung field. Cardiac silhouette is within normal limits. Hilar shadows are not enlarged. Both diaphragms and costophrenic angles appear normal. NG tube is in place. No osseous abnormalities are identified in the imaged areas. Decrease in band atelectasis at right lung base. Linear opacity projecting over left base persists. Impression: Decrease in band atelectasis at right lung base. Linear opacity over left base persists, likely atelectasis though infiltrate cannot be completely excluded. No pulmonary edema. Clinical correlation recommended for the persistent linear opacity.

[Figure 45]

#### ECHO-AR

Finding: The thorax is symmetric bilaterally. The mediastinum and trachea are midline. Pulmonary vasculature is clear and follows a normal course; no definite abnormal opacities are seen in either lung field except for a persistent left basilar opacity. Cardiac silhouette is within normal limits. Hila shadows are not enlarged. Both diaphragms and costophrenic angles appear normal. No osseous abnormalities are identified in the imaged areas. Nasogastric tube is present, terminating below the diaphragm. Impression: Nasogastric tube in place. Persistent left basilar opacity.

ECHO-Base

Finding: The thorax is symmetric bilaterally. The mediastinum and trachea are midline. Pulmonary vasculature is clear and follows a normal course; no definite abnormal opacities are seen in either lung field except for a linear opacity in the left mid lung zone. Cardiac silhouette is within normal limits. Hilar shadows are not enlarged. Both diaphragms and costophrenic angles appear normal. No osseous abnormalities are identified in the imaged areas. Nasogastric tube is present. Impression: Linear opacity in left mid lung zone likely representing atelectasis. Recommend clinical correlation and follow-up imaging if symptoms persist.

ECHO-Base_onestep

Finding: The thorax is symmetric bilaterally. The mediastinum and trachea are midline. Pulmonary vasculature is clear and follows a normal course; no definite abnormal opacities are seen in either lung field. Cardiac silhouette is within normal limits. Hilar shadows are not enlarged. Both diaphragms and costophrenic angles appear normal. No osseous abnormalities are identified in the imaged areas. Fe lung tubeular central ven present. Low lung volumes. noted. Left retrocardiac opacityelect.\n. Impression: Fe lung tubeular central ven present. Low lung volumes. Left retrocardiac opacityelectasis. Consider clinical correlation for managementelectasis management and and related findings.

#### ECHO-blk8

Finding: The thorax is symmetric bilaterally. The mediastinum and trachea are midline. Pulmonary vasculature is clear and follows a normal course; no definite abnormal opacities are seen in either lung field for a left left at atasis..iac silhouette is within normal limits. Hilar shadows are not enlarged. Both diaphragms and costophrenic angles appear normal. No osseous abnormalities are identified in the imaged areas. Nasogastric tube is present. Impression: Nasogastric tube present. Left left at atasis.

ECHO-blk4

Finding: The thorax is symmetric bilaterally. The mediastinum and trachea are midline. Pulmonary vasculature is clear and follows a normal course; no definite abnormal opacities are seen in either lung field except for a linear opacity opacity in the left mid lung.Cardiac silhouette is within normal limits. Hilar shadows are not enlarged. Both diaphragms and costophrenic angles appear normal. No osseous abnormalities are identified in the imaged areas. Impression: Linear opacity opacity in left mid lung likely atelectasis. Recommend clinical correlation and follow-up imaging if symptoms persist.

###### Figure A.3 Qualitative comparison of report generation quality across model variants. ECHO-Base_onestep denotes ECHOBase forced to decode in a single step without distillation.

##### CXR and GT

[Figure 46]

Findings: The thorax is symmetric bilaterally. The mediastinum and trachea are midline. Left lung field is clear; right lung field demonstrates pleural effusion. Cardiac silhouette is enlarged. Hila are not enlarged. Both diaphragms and costophrenic angles appear normal; right costophrenic angle obscured by pleural effusion. Cardiac pacer device is present projecting over the right chest with intact leads in the right atrium and right ventricle; old pacer wires are present on the left. No osseous abnormalities are identified in the imaged areas. Impression: Pleural effusion at right side. Cardiomegaly.

##### ECHO-blk8

Finding: The thorax is asymmetric due to right righturaluralusion. Mediastinum and trachea are midline. Right lung field demonstrates large pleural effusion with associated atelectasis; left lung field is clear. Cardiac silhouette is within normal limits. Hila are not enlarged. Both diaphragms and costophrenic angles appear normal. Bony structures are unremarkable. Right-sided pac is inemaker. Impression: right-sided pacemaker effemaker. Large right pleural effusion with associated atelectasis. Recommend clinical evaluation for ple drainage for symptomatic and andomatic..

ECHO-blk4

Finding: The thorax is asymmetric due to right pleithural effusion. Mediastinum and trachea are midline. Right lung field shows large pleural effusion with associated atelectasis; left lung field is clear. Cardiac silhouette is within normal limits. Hila are not enlarged. Both diaphragms and costophrenic angles are unremarkable. Bony structures are unremarkable. Right pacemaker with two in place and left p pacemaker with in place are present. Impression: Large right pleural effusion with associated atelectasis. Recommend clinical evaluation for underlying cause.

##### CXR and GT

[Figure 47]

Findings: Lines and tubes are in position. The thorax is asymmetric. The mediastinum and trachea are midline. No definite abnormal opacities are seen in either lung field; however, diffuse subcutaneous air along the right lateral chest wall limits evaluation of the lung parenchyma. Cardiac silhouette is within normal limits. Hila are not enlarged. Both diaphragms and costophrenic angles appear normal. Probable small right apical pneumothorax is noted. Bony structures are unremarkable. Impression: Probable small right apical pneumothorax with subcutaneous air. Clinical correlation recommended.

##### ECHO-blk8

Finding: The thorax is asymmetric due to right chest tube. Mediastinum and trachea are midline. Lung lung field demonstrates small apical pneumothax; subcut field emphysema is present. Right lung chest field emphyphysese lung silhouette is within normal limits. Hila are not enlarged. Both diaphragms and costophrenic angles appear normal. Bony structures are unremarkable. Impression: Small right apical pneumothorax. Subcutcutaneous emphysema. clinical

ECHO-blk4

Finding: The thorax is asymmetric due to right chest tube and sub chest wall emphy. Mediastininum trachea are midline. Right lung field shows a ap apical pneumothorax and subcutaneous emphysema; left lung field is clear. Cardiac silhouette is within normal limits. Hila are not enlarged. Both diaphragms and costophrenic angles appear normal. Bony structures are unremarkable. Dual chest pac pacemaker are present. Impression: Right chest tube and subcutaneous chest wall emphysema. Right apical pneumothorax.

###### Figure A.4 Qualitative examples of positive pathology detection by ECHO. Correctly identified abnormal findings are highlighted in green.

