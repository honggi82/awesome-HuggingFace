# arXiv:2512.06776v2[cs.CL]30Jan2026

## From Next-Token to Next-Block: A Principled Adaptation Path for Diffusion LLMs

Peking University & Huawei Technologies

### Abstract

Diffusion Language Models (DLMs) enable fast generation, yet training large DLMs from scratch is costly. As a practical shortcut, adapting off-the-shelf Auto-Regressive (AR) model weights into a DLM could quickly equip the DLM with strong long-context generation capabilies. Prior “adaptation” attempts either modify logits or randomly grow attention masks to Full-Sequence diffusion, or simply transplant AR weights into a Block-Diffusion recipe, leaving two key questions unaddressed: where is the final destination of adaptation, and how to adapt better? For manifold benefits, we reframe the whole AR-to-DLM adaptation under the Block-Diffusion paradigm, transitioning from block size 1 to the final Block-Diffusion state. Concretely, the principled pathway of adaptation is designed as follows: we keep a context-causal path where causal attention is kept in the prefix, an efficient parallel adaptation procedure where an AR guidance is maintained, and gradual increment of the generation block size for a smoother transition. Built on these components, the adaptation is proved competitive on various models at different scales. With better adaptation, we propose NBDIFF-7B that could inherit the long-context modeling and reasoning capabilities, and achieve state-of-the-art performance among the 7B-class DLMs. Codes: https://github.com/YuchuanTian/NBDiff.

### 1 Introduction

Large language models (LLMs) are rapidly permeating real-world applications because of their strong generative capability. However, the dominance of AutoRegressive (AR) LLMs is built on a fundamental trade-off: powerful left-to-right causal generation at the cost of strictly sequential, token-by-token decoding. This trade-off creates an inference bottleneck that limits the decoding speed of AR LLMs. In contrast, Diffusion Language Models (DLMs) offer a promising alternative by enabling parallel generation, reducing sequential dependencies and yielding substantially higher throughput and lower wall-clock latency in practice.

Current diffusion approaches for language have largely converged on masked diffusion, with two dominant paradigms. Full-Sequence Diffusion [20; 34] starts from a fully masked sequence and denoises to a complete output where all tokens attend bidirectionally. Block-Diffusion [1; 6] decodes one block at a time: tokens are bidirectional within the active block, while blocks themselves follow left-to-right causal order, yielding a semi-autoregressive workflow. Training masked diffusion is intrinsically harder than AR pretraining because, unlike AR–where every token contributes a nexttoken-prediction loss–only masked tokens provide supervision, which slows optimization. Yet masked diffusion and AR models are strikingly similar in input–output format and transformer architecture. This naturally motivates the question: can we leverage powerful off-the-shelf AR checkpoints and rapidly adapt them into diffusion models, preserving their knowledge while avoiding the cost of training a DLM from scratch?

Existing adaptation methods are lacking. Early attempts used logit shifts and random attention mask growth to Full-Sequence Diffusion [8; 34]. More recent block-wise adaptations simply ’transplant’

Preprint. Under review.

the AR model into a Block-Diffusion training setup ’as is.’ [6]–they do not investigate the core mismatch between AR and Block-Diffusion. These methods leaves a clear gap: where should be our destination of adaptation, and how to adapt an AR model to Diffusion for better performance?

Our approach is grounded in a key insight: for longer-sequence training, efficient inference, and adaptation benefits, Block-Diffusion should be either the pathway and the destination of adaptation. AR generation could be viewed as a special case of Block-Diffusion with a blocksize of 1, reframing adaptation not as a crude switch, but as a smooth transition across a spectrum. Under this unified view, we look for a principled and smooth transition path from AR to Block-Diffusion. Our design consists of a context-causal attention mask that preserves AR inductive bias in committed context, parallel training with an auxiliary AR guidance that regularizes the path of adaptation, and gradual growth of block size. The design provides an efficient adaptation strategy from AR to DLM that progressively unlocks bidirectional attention and parallel decoding within the generating block while maintaining strict train–inference alignment.

GSM8K CMMLU

MATH

MMLU-Pro

###### Models

LLaDA-MoE 7B-A1B

LLaDA2.0-mini 16BA1B

Dream-v0-Instruct-7B

SDAR 8B

Ours-7B-Instruct

MBPP HumanEval

Figure 1: Comparison of our model with baselines. After adaptation from an open-sourced AR LLM, our model has good long-sequence and reasoning capabilities and shows outstanding performance in various benchmarks.

Our contributions are as follows:

- 1. After investigation, we propose to view the whole adaptation process under Block-Diffusion for its natural training, inference, and adaptation benefits. The transition from AR to DLM is then simply blocksize growth from causal (blocksize = 1) to target size.
- 2. We propose the Context-Causal mechanism tailored for this adaptation, which preserves AR knowledge in the context while enabling efficient bidirectional intra-block generation. We develop an efficient parallel training strategy that aligns with inference and incorporates an auxiliary AR loss. We also develop a gradual block growth approach that alleviates the gap between AR and Block-Diffusion models. These measures markedly improve adaptation performance.
- 3. We demonstrate the effectiveness of our approach with various models. We also propose NBDIFF-7B, which, after efficient adaptation from its strong AR counterparts, could model long contexts (up to 32K sequence lengths) and perform reasoning. Both NBDIFF-7BBASE and NBDIFF-7B-INSTRUCT outperform strong baselines like LLaDA [20; 38; 12], Dream [34], and SDAR [6] on general, math, and code benchmarks (Figure 1), achieving state-of-the-art performance.

### 2 Related Work

Discrete diffusion language models. Diffusion has been extended to categorical spaces, showing that discrete denoising objectives can effectively model text and clarifying connections to classical LM training, including transition design and absorbing states [2]. Two dominant paradigms have since emerged. Masked diffusion iteratively reveals tokens, enabling controllable generation with competitive likelihoods without left-to-right decoding [17], while recent MDLM variants substantially close the perplexity gap to AR LMs using simplified training recipes [23]. Absorbing-state diffusion instead corrupts tokens toward a sink symbol; recent analyses relate its objective to conditional modeling, calibration, and sampling behavior [21]. At scale, systems trained from scratch such as LLaDA demonstrate that masked-diffusion-style pretraining can rival strong AR baselines and extend naturally to multimodal instruction tuning [20; 35], establishing the viability of DLMs at billion-parameter scale.

Recent trend: Block-Diffusion, adaptation from AR, and test-time scaling. While FullSequence Diffusion provides fully bidirectional context, it is computationally inefficient for long texts

[BOS] Diffusion LLMs are fast and strong .

[Figure 1]

Causal Auto-Regressive LLM

Diffusion LLMs are fast and strong .

[EOS]

Attention Mask: Causal

[BOS] [MASK] LLMs [MASK] [MASK] [MASK] strong [MASK]

[Figure 2]

Diffusion LLM (Adapted from AR)

Diffusion Attention Mask: Full

LLMs [MASK] fast [MASK] strong [MASK] [MASK]

[BOS] Diffusion LLMs are [MASK] [MASK] strong [MASK]

[Figure 3]

Block Diffusion LLM

[BOS]

Diffusion LLMs are fast [MASK] strong [MASK]

Attention Mask: Intra-Block Bidirectional Blockwise Causal

Context

[BOS] Diffusion LLMs are fast [MASK] strong [MASK]

| |[Figure 4]|
|---|---|
| | |

ContextBlock

NBDiff (Ours)

Diffusion LLMs are fast [MASK] strong .

[EOS]

Attention Mask: Last-Block Bidirectional Context Causal

Context Token Decoded Token [M] Mask Token Newly Decoded Token

- Figure 2: The diffusion paradigm of our NBDIFF-7B model. We compare popular language generation paradigms. Diffusion LLMs adapted from AR adopt logit shift and attention mask growth; Block-Diffusion uses block-wise autoregressive and maintains an intra-block bidirectional mask; Our model adopts Block-Diffusion where bidirectional attention is used intra-block, but features a causal context.

and misaligned with left-to-right inductive biases. Attempts to amortize this cost via intermediate-state caching improve efficiency but do not fundamentally resolve the issue [18; 19; 31]. Block-Diffusion addresses this by fixing past context while denoising the current block bidirectionally, enabling parallel token updates and unbounded-length generation with tunable quality–efficiency trade-offs [10; 1]. Beyond training from scratch, several works adapt pretrained AR models into diffusion-style decoders, often at the block level, reporting objective connections, practical conversion recipes, and hybrid AR–diffusion paradigms that preserve AR quality while enabling parallel generation [8; 6; 31; 30]. In parallel, diffusion-based reasoning systems explore inference-time scaling or reinforcement learning to improve multi-step reasoning, particularly for math and code, but remain limited by short contexts or underutilized AR priors [33; 26; 13; 9; 36; 25; 27]. In contrast, our approach adapts strong AR models into block-diffusion generators via a smooth way, enabling longer context and better performance.

### 3 Rethinking DLM Adaptation from AR: to Where, and How?

#### 3.1 Revisiting Previous Adaptation

Prior adaptation work [8] mainly focuses on adapting a Full-Sequence Diffusion model from an AR model. The authors observe the difference in attention mechanism, and proposes random "annealing",

or random growth, of the attention mask from a lower-triangular causal attention mask to a full, bidirectional attention mask.

While the work [8] is trying to bridge the AR and Diffusion generation paradigms, we hold different opinions on random growth of attention masks: its transition is not "natural." In practice, training sees unknown future corpora; sporadically granting early tokens access to a random subset of future tokens yields incomplete and potentially misleading context, thus limiting adaptation potentials.

Hence, we try to answer two major questions regarding this transition, i.e. to where, and how. Firstly, what should be the destination of this transition? Secondly, is there a smoother and better way to transition from an AR model to a DLM model?

#### 3.2 Block-Diffusion and its Advantages

Unlike previous adaptation methods [8] that focuses merely on Full-Sequence Diffusion models, recent diffusion LLMs [6; 30] increasingly adopt Block-Diffusion [1], which is both more efficient and performant than Full-Sequence Diffusion and conceptually sits between Full-Sequence Diffusion and AR: generation proceeds left-to-right across blocks while remaining bidirectional within a block (Figure 2). Specifically, tokens within the same block attend to each other bidirectionally, whereas attention across blocks is strictly causal. Decoding is performed block by block, with all tokens in a block capable of generating in parallel.

We analyze the advantages from several aspects: either from training, from inference, and also from the perspective of adaptation from AR models.

Training advantages: stabler training at longer sequences. AR models excel at long-sequence reasoning, yet most diffusion LLMs still operate at modest context lengths (e.g., LLaDA [20]=1K, Dream [34]=512), raising the question of how sequence length impacts DLM training. We therefore pretrain Full-Sequence Masked Diffusion and Block-Diffusion under identical corpora at 1K/2K/4K/8K sequence lengths and compare their training losses (Figure 3). For fair comparison, we adjust batchsize accordingly with sequence length to ensure that each training iteration consumes the same number of tokens. As context grows, the full-sequence model exhibits increasingly large loss oscillations, whereas Block-Diffusion remains consistently stable. Notably, at long lengths the block-diffusion loss is also lower, suggesting that longer contexts provide tangible generation benefits when the training dynamics are well-conditioned.

- 2

- 3

- 4

- 5

- 6

8K 4K 2K 1K

Loss

- 0 100 200 300 400 500 600 700 800 Step

10

8K 4K 2K 1K

8

Loss

6

4

0 100 200 300 400 500 600 700 800 Step

- Figure 3: Loss curves at different sequence lengths (Left: Full-Sequence; Right: BlockDiffusion). As sequence length increases, Full-Sequence Diffusion suffers loss fluctuation while Block-Diffusion loss remains stable. Block-Diffusion is a better choice for long-sequence generation.

We hypothesize the instability of masked Full-Sequence Diffusion stems from the combinatorial explosion of the masking space: as sequence length increases, (i) the number of masked tokens per step can vary drastically, and (ii) the effective decoding/denoising orders proliferate, complicating optimization [14]. In contrast, Block-Diffusion constrains the space by fixing a constant block size, which regularizes the denoising schedule, preserves left-to-right structure, and stabilizes gradients, thereby enabling long-sequence advantages to manifest in masked DLMs.

Inference advantages: KV-Cache reuse. Different from Full-Sequence Diffusion where the whole sequence has to pass through the model together for each inference step, Block-Diffusion keeps previous tokens fixed while only performs decoding in the last block of the generated tokens. Thus it could re-use the KV-Cache from previous block generations and only the last block to be generated needs to be passed into the model, reducing significant inference costs. Besides, though BlockDiffusion has designated the causal (left-to-right) block generation sequence, the use of bidirectional attention within the last generating block still enables parallel token-decoding. In practice, we use the

blocksize of 32 tokens (larger than previous same-scale DLMs [6]) to tap the speed potential of the proposed model to the full.

Adaptation advantages: analogous paradigms and easier adaptation from AR. Apart from the performance advantages, Block-Diffusion helps easy adaptation from AR. Instead of forcing a global jump from causal to full-sequence bidirectional attention, we treat Block-Diffusion as the destination. By preserving left-to-right semantics at the block level and relaxing bidirectionality inside the active block, we try to partially align with the AR inductive bias for better adaptation, which greatly reduces the difficulty for alignment. In addition, the blockwise semi-AR manner could also enable parallel training, improving data utility and model convergence.

In the next section, we will stick to the paradigm of Block-Diffusion and seek a way for fast transition from AR model.

### 4 Designing Transition Paradigms

- 4.1 How Should the Unmasked Context Attend? Comparing the attention mechanisms of Block-Diffusion and AR (which could be roughly viewed

- as Block-Diffusion of blocksize = 1, as introduced in Sec. 1), the key difference that requires our adaptation efforts lies in the bidirectional attention within the last active block; namely, we have to grow the attention mask at the end of the generating sequence from blocksize = 1 to target block size. However, apart from the attention within the active block region, different transition solutions

arises from the decoded context: how tokens in the unmasked context (x<s

) should attend to each other? Here we analyze two possible attention pathways of Block-Diffusion as follows:

K

- • Block-Causal (widely used in Block-Diffusion [1] / D2F [28] / SDAR [6]). Tokens have bidirectional attention within every block (both past/committed blocks and the active block), and causal flow across blocks (each token can see all tokens in earlier blocks). This maximizes intra-block interaction everywhere, not only in the active block.
- • Context-Causal (our preferred setting). The context (prompt + already generated/committed blocks) remains strictly causal: each token only attends to itself and predecessors. Only the last (active) block is given bidirectional attention to support diffusion-style refinement; future blocks are hidden.

To examine the two schemes, we adapt two Block-Diffusion models from an AR model (based on Block-Causal and Context-Causal, respectively) by training 2000 iterations, and examine their performance on popular math and coding benchmarks. The results are shown in Table 1.

- Table 1: Comparison of Block-Causal and Context-Causal attention schemes. Context-Causal gains a clear advantage in adaptation from AR.

Scheme GSM8K MATH HumanEval MBPP Avg Block-Causal 60.1 1.6 24.4 39.4 31.4 Context-Causal 68.8 36.8 41.5 47.4 48.6

Empirical takeaway and intuition. In these preliminary adaptation experiments, Context-Causal consistently outperforms Block-Causal by large margins: the accuracy is significantly higher when the context keeps strict causality and only the active block is bidirectional.

We attribute this to: (i) Inductive-bias alignment with AR pretraining, which reduces the gap between AR and Block-Diffusion by preserving causal self-attention in the context; and (ii) Generationparadigm consistency: although the active block is refined bidirectionally (no fixed order inside the block), the overall decoding remains left-to-right across blocks. Keeping the context causal does not reduce the visibility required for the block being generated and avoids introducing spurious, partially bidirectional signals into earlier (already “finalized”) content.

Noised Sequence Unmasked Sequence

Noised Seq Unmasked Seq

|B|1|M|3|4|M|6|M|B|1|2|3|4|5|6|7|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|

Input

B 1 M 3 1 2 3 4 5 6 7

4 M 6 M B

[Figure 5]

[Figure 6]

UnmaskedSeqNoisedSeq

|𝐌𝐵𝐷|
|---|

|𝐌𝑂𝐵𝐶|
|---|

Diffusion LLM

Output

1 2 3 4 5 6 7 E 1 2 3 4 5

6 7 E 1 2 3 4 5 6 7

Shift

|𝐌𝐶𝐶|
|---|

GT Labels 1 2 3 4 5 6 7 1 2 3 4 5 6 7 E

|ℒ𝐴𝑅|
|---|

|ℒ𝑀𝐷𝑀|
|---|

- Figure 4: Our Parallel Training Diagram. The diagram shows the parallel training form of our Context-Causal setting (we use blocksize = 4 as an example; the actual blocksize is 32). We

concatenate a clean, unmasked token sequence to the noised sequence. The attention mask Mall is designed (shown in the right) such that strictly-causal attention is applied in the unmasked input; for the masked input, each token has bidirectional attention intra-block, but causal attention to past inter-block tokens that are unmasked. AR loss LAR is introduced in addition to the canonical masked loss LMDM for faster adaptation.

#### 4.2 Training Parallelism

The naive block-diffusion recipe is data-inefficient: random cropping wastes the remaining tokens of each sequence, and only a small subset of masked tokens inside the last block contributes to the loss. Unlike AR pretraining—where every token can supervise next-token prediction—switching to next-block prediction sharply reduces token utilization. We therefore restructure training so that all blocks provide learning signal in a single forward pass.

To reach this goal, the clean (unmasked) sequence is concatenated after the noised sequence and enforce a structured attention mask that lets the clean side provide context to the noised side. The mask (shown in Figure 4) has four quadrants: (upper-left, MBD) block-diagonal self-attention within the noised view (bidirectional inside each noised block); (upper-right, MOBC) attention from noised tokens to earlier clean blocks, so denoising conditions on stable context; (lower-left) zeros, preventing the clean sequence from reading the noised view (matching inference); and (lower-right, MCC) strictly causal self-attention within the clean sequence. Relative to prior block-diffusion training masks [1], we replace block-causal with fully causal (context-causal) in this lower-right quadrant, preserving the AR inductive bias in the context while still enabling intra-block bidirectionality only where generation happens. The detailed formulation of training and the structured attention mask in enclosed in Appendix A. This design optimizes per-step token utilization, and empirically stabilizes training.

#### 4.3 AR Loss Guidance

While our one-pass parallel recipe enables efficient Block-Diffusion training, the diffusion loss is only applied to logits on the noised (active) blocks; logits on the clean-context branch of the concatenated sequence mainly serve as conditioning signals. Meanwhile, our adaptation follows a path from Block-Diffusion with blocksize = 1 (i.e., AR) to a target block size, and we would like this path to remain anchored to the AR behavior rather than drifting too far away. To this end, we introduce an auxiliary AR objective as a lightweight constraint, which is naturally attached to the clean-context branch because it already follows strictly causal self-attention. This turns otherwise unused context predictions into supervised next-token targets, improving token utilization without changing diffusion-side conditioning.

Let C index tokens on the clean context, xi be the ground-truth token at position i, and MCC denote the context-causal mask; we attach a standard LM head and define an autoregressive objective over the context as

LAR(θ) = E −

log pθ xi+1 | x≤i; MCC . (1)

i∈C

Let LMDM(θ) be the masked/block-diffusion denoising loss computed on the noised view under Mall (as in Eq. (5)); we then train with an affine combination controlled by λ≥0:

Ltotal(θ) = LMDM(θ) + λLAR(θ). (2)

In practice, we set λ = 0.5 so that the number of tokens participating in the MDM and AR losses is kept at a comparable scale throughout adaptation. Overall, LAR provides a simple yet effective guidance signal along the AR→Block-Diffusion path, while being “free” to compute from the clean-context branch in our parallel training formulation.

#### 4.4 Gradual Block Growth

After determining the path for transition and guidance, we pursue a smoother adaptation by growing blocks in Block-Diffusion (Figure 5). As noted, AR could be viewed as Block-Diffusion of blocksize = 1; hence, the transition could be viewed under the Block-Diffusion paradigm, where we start from block size of 1 and end at the target block size. Naturally, we gradually increase the generation block size from AR’s single-token steps toward larger blocks, so that the model transitions continuously from next-token prediction to next-block refinement. This monotonic growth retains left-to-right causality while progressively introducing intra-block bidirectionality, narrowing the procedural gap and easing optimization.

Gradual Transition from AR to Block Diffusion

[Figure 7]

[Figure 8]

[Figure 9]

| |
|---|

| |
|---|

B M M

B M M M M

B

Diffusion LLM (Block=2)

Diffusion LLM (Block=4)

AR LLM

E

E

- Figure 5: The Diagram for Gradual Block Growth. Starting from an AR model (which could be viewed as a Block-Diffusion model of blocksize = 1), we gradually double the blocksize during training until reaching the target size, mitigating the adaptation gap between AR and Block-Diffusion.

Starting from blocksize b=1 (AR), we interpolate to larger bidirectional blocks by growing b in integer powers of a user-chosen base r ∈ {2,4,...} (on a normal basis the power of 2 is adopted) at fixed training intervals. Let s be the global training step, ∆ the interval (in steps) between growth events, s0 an optional warmup before the first growth, b0=1 the starting size, and bmax the inference target. The schedule is as follows.

b(s) = min bmax, b0 · r

max(0, s−s0)

∆ , (3)

which holds b constant on plateaus [s0+k∆, s0+(k+1)∆) and multiplies it by r whenever s crosses

- a multiple of ∆ (e.g., 1→r→r2→...) until capped by bmax. This integer-power curriculum reduces the AR → diffusion adaptation gap by aligning early optimization with the AR inductive bias (small
- b) and gradually unlocking intra-block bidirectionality and parallel supervision as b increases. In practice, we keep train-inference semantics matched at each plateau via the same context-causal mask family, optionally co-schedule compute by shrinking the refinement steps per block T(s) ∝ 1/b(s) to maintain a roughly stable token-update budget, and anneal the AR-loss weight λ as blocks grow to reallocate gradient capacity toward the diffusion objective.

- Table 2: Comparison of Adaptation Methods. We demonstrate the effectiveness of our method across different models and settings.We also show the contribution of each component in our adaptation methods.

Method GSM8K MATH HumanEval MBPP Avg Qwen3-4B-Base

Annealed Attention Mask [8] 76.57 28.66 25.61 59.40 47.56 Plain Finetuning 72.18 24.84 31.10 56.80 46.23 Ours 79.83 32.26 27.44 61.60 50.28

Qwen3-8B-Base

Annealed Attention Mask [8] 80.67 25.76 39.02 49.80 48.81 Plain Finetuning 78.32 24.22 42.68 58.20 50.85 Ours 82.26 34.68 31.71 64.40 53.26

openPangu-7B

Annealed Attention Mask [8] 70.05 35.46 26.83 45.00 44.34 Plain Finetuning 72.63 36.14 39.63 47.40 48.95 + AR loss 75.13 43.30 40.24 53.20 52.97 + AR loss & Gradual Growth (Ours) 76.57 44.06 44.51 54.60 54.94

### 5 Experiments

In this section, we empirically demonstrate the success of our proposed adaptation methods via manifold experiments. We test our method on various off-the-shelf model weights of different sizes, including Qwen3-4B-Base, Qwen3-8B-Base [32], and openPangu-Embedded-7B [4].

Experiment setup. For all experiments, we use sequence length ℓ=8k and global batch size B=1024; lr is set as 1e − 5 with the Adam optimizer. we train for 4000 iterations that consumes approximately 30B of the training data. Adaptation iteration for gradual growth is set as 2500. For the Qwen3 series, we use the FineWeb-100B [22] dataset. For openPangu, we use a large, high-quality 700B internal dataset.

Comparison with existing baselines. On the basis of logit shift introduced in DiffuLLaMA [8], we adopt two methods as baselines. "Annealed Attention Mask" [8] proposes random growth from the auto-regressive causal attention mask to the targeted Diffusion attention mask. In our setting, since we are using structured attention masks for parallel training, "Annealed Attention Mask" is applied as the random interpolation between structured attention masks for blocksize = 1 and blocksize = 32. We are aware that MBD and MOBC are closely dependent; hence, we "chain" the randomness of MBD and MOBC such that each token will not view each position twice in attention. "Plain Finetuning" directly uses the targeted attention mask for training.

As shown in Table 2, our method consistently outperforms both baselines across model scales and evaluation benchmarks. Compared to annealed attention masks and plain finetuning, our approach achieves the highest average performance on Qwen3-4B, Qwen3-8B, and openPangu-7B with particularly notable gains on GSM8K, MATH, and MBPP. These results indicate that a structured and progressive adaptation strategy is more effective than either random interpolation or directly applying the target attention mask, leading to more stable and reliable performance improvements across diverse tasks and model sizes.

Contribution of adaptation. In Table 2, we also ablate our two adaptation components: AR loss and gradual block-size growth—starting from a plain fine-tuning baseline. Adding AR loss lifts the overall Avg from 48.95 to 52.97 (+4%), with the largest gains on math and MBPP and modest improvements elsewhere. Stacking gradual block-size growth further raises Avg to 54.94, indicating that smoother progression from next-token to next-block generation improves stability and yields consistent, additional benefits—especially for coding and multi-step reasoning.

### 6 NBDiff-7B

We further demonstrate the effectiveness of our adaptation method via intensive data scaling. We start from the openPangu-Embedded-7B [4] base checkpoint and adapt it into a diffusion language model (DLM) using the training dataset in that release. With these efforts, we launch NBDIFF-7B, a State-of-the-Art diffusion model.

#### 6.1 Setup

Training. The pretraining adaptation stage uses a two-phase learning-rate schedule: we keep the learning rate constant for the first 24,000 iterations and then apply a learning-rate cooldown over the final 60,000 iterations, for a total of 84,000 iterations. We train with sequence length ℓ=8k and global batch size B=1024. The effective tokens processed per iteration are 8M tokens, so across 84,000 iterations the total token volume is approximately 700B tokens. NBDIFF-7B-BASE, the State-of-the-Art base model with 8K context, is completed after this stage. Then, to equip the model with long-sequence generative capabilities, we extend the pretraining sequence length ℓ=32k and train for 23,800 iterations (approximately 100B tokens), equipping the model with long-sequence modeling capabilities. Finally, we use 10B-token SFT data of sequence length ℓ=32k to finetune the model for 10 epochs (approximately 17,000 iterations) and equip it with reasoning capabilities.

We use a uniform masking strategy over the diffusion step t∼Uniform[0,1] (sampled and mapped to the discrete step index), and keep the inference/training mask families matched at each curriculum plateau. All other optimizer and system-level settings follow the default configuration of the openPangu-Embedded-7B [4] release.

Inference. Our inference sampling is mainly based on the implementation of Fast-DLLM-v2 [30]: at the macro level the sequence is generated left-to-right by blocks of size 32 (causal across blocks), while inside each block we permit bidirectional attention to refine tokens jointly. For speed, the inner refinement can follow the v2 “small-block” schedule or be collapsed into a single full-block bidirectional pass when latency matters. For general benchmarks, we adopt the standard configuration. Greedy strategies are used for math tasks and p = 0.9,T = 1 sampling strategies are used for coding benchmarks to optimize performance. The experiment results are all single-run outcomes.

Table 3: Comparison between NBDIFF-7B-INSTRUCT and the latest SFT (Instruct) version diffusion language models. Our model demonstrates strong performance on general, math, and coding tasks, and outcompetes latest diffusion baselines by large margins. * indicates non-official replications.

LLaDA-MoE LLaDA2.0-mini Dream-v0 SDAR Ours-7B Benchmark 7B-A1B preview 16BA1B Instruct-7B 8B Instruct General

MMLU 67.2 72.5 67.0 78.6 82.9 MMLU-Pro 44.6 49.2 43.3 56.9 71.9 CMMLU 64.3 67.5 58.8 75.7 79.8 CEval 63.9 66.5 58.0 72.7* 72.5 IFEval 59.3 62.5 62.5 61.4 60.8

Math

GSM8K 82.4 89.0 81.0 91.3 91.0 MATH 58.7 73.5 39.2 78.6 84.0

Coding

MBPP 70.0 77.8 58.8 72.0 87.6 HumanEval 61.6 80.5 55.5 78.7 89.0

Avg 61.1 71.0 58.2 74.0 79.9

#### 6.2 Evaluation

We primarily evaluate NBDIFF-7B across three capability areas—code, math, and general knowledge—and compare its performance against several baseline models to understand relative strengths

and trade-offs. We evaluate general capabilities on MMLU [11], MMLU-Pro [29], CEVAL [? ], CMMLU [16], and IFEval [37]; mathematical reasoning on GSM8K [7] and MATH [15]; and coding performance on MBPP [3] and HumanEval [5].

We present the SFT (Instruct) results (i.e. NBDIFF-7B-INSTRUCT) in Table 3. Due to page limits, the Base version, NBDIFF-7B-BASE, is presented in Appendix B. NBDIFF-7B-INSTRUCT delivers the highest macro average (79.9) among SFT baselines, substantially outperforming SDAR-8B [6] and LLaDA [20] variants [38; 12]. On general knowledge, it sets the pace on MMLU (82.9), MMLUPro (71.9), and CMMLU (79.8), and ranks second on CEval (72.5) while remaining competitive on IFEval (60.8) despite ties among baselines. For math, NBDIFF-7B-INSTRUCT achieves good GSM8K performance (91.0) and State-of-the-Art MATH performance (84.0), indicating strong multistep and competition-style reasoning under instruction following. In coding, it tops both MBPP (87.6) and HumanEval (89.0), narrowing and in most cases reversing the AR-favoring gap seen in some base models. Taken together, these results show that instruction tuning on a diffusion LLM not only preserves the Base model’s breadth, but amplifies performance across general, math, and coding by large margins, establishing NBDIFF-7B-INSTRUCT as a strong, balanced SFT model in the 7B class.

### 7 Conclusion

In this work, we propose a principled adaptation framework that bridges the gap between Autoregressive (AR) and Block-Diffusion models. By reframing adaptation as a continuous interpolation– viewing AR as a Block-Diffusion model with a block size of one–we introduce the context-causal attention mechanism and an efficient parallel training recipe with auxiliary AR supervision, which maximally preserves the pre-trained knowledge of the source model. We also propose a block-size growth curriculum that smoothly transitions the model from sequential to parallel generation.

Our resulting model, NBDIFF-7B, has achieves state-of-the-art performance among 7B-parameter diffusion models, outperforming strong baselines on math, code, and general reasoning benchmarks. These results demonstrate that expensive pre-training from scratch is not necessary to build high-quality diffusion LLMs. Instead, our method offers a compute-efficient pathway to unlock parallel generation capabilities in existing open-source AR checkpoints, potentially accelerating the deployment of faster and more flexible generative models.

### References

- [1] Marianne Arriola, Aaron Gokaslan, Justin T. Chiu, Zhihan Yang, Zhixuan Qi, Jiaqi Han, Subham Sekhar Sahoo, and Volodymyr Kuleshov. Block diffusion: Interpolating between autoregressive and diffusion language models. In ICLR, 2025.
- [2] Jacob Austin, Daniel D. Johnson, Jonathan Ho, Daniel Tarlow, and Rianne van den Berg. Structured denoising diffusion models in discrete state-spaces. In NeurIPS, 2021.
- [3] Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, et al. Program synthesis with large language models. arXiv preprint arXiv:2108.07732, 2021.
- [4] Hanting Chen, Yasheng Wang, Kai Han, Dong Li, Lin Li, Zhenni Bi, Jinpeng Li, Haoyu Wang, Fei Mi, Mingjian Zhu, Bin Wang, Kaikai Song, Yifei Fu, Xu He, Yu Luo, Chong Zhu, Quan He, Xueyu Wu, Wei He, Hailin Hu, Yehui Tang, Dacheng Tao, Xinghao Chen, and Yunhe Wang. Pangu embedded: An efficient dual-system llm reasoner with metacognition, 2025.
- [5] Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian, Clemens Winter, Philippe Tillet, Felipe Petroski Such, Dave Cummings, Matthias Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel HerbertVoss, William Hebgen Guss, Alex Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain, William Saunders, Christopher Hesse, Andrew N. Carr, Jan Leike, Josh Achiam, Vedant Misra, Evan Morikawa, Alec Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie Mayer, Peter Welinder, Bob McGrew, Dario Amodei, Sam McCandlish, Ilya Sutskever, and Wojciech Zaremba. Evaluating large language models trained on code. 2021.

- [6] Shuang Cheng, Yihan Bian, Dawei Liu, Yuhua Jiang, Yihao Liu, Linfeng Zhang, Wenhai Wang, Qipeng Guo, Kai Chen, Biqing Qi, and Bowen Zhou. Sdar: A synergistic diffusion–autoregression paradigm for scalable sequence generation. arXiv preprint arXiv:2510.06303, 2025.
- [7] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.
- [8] Shansan Gong, Shivam Agarwal, Yizhe Zhang, Jiacheng Ye, Lin Zheng, Mukai Li, Chenxin An, Peilin Zhao, Wei Bi, Jiawei Han, Hao Peng, and Lingpeng Kong. Scaling diffusion language models via adaptation from autoregressive models, 2025.
- [9] Shansan Gong, Ruixiang Zhang, Huangjie Zheng, Jiatao Gu, Navdeep Jaitly, Lingpeng Kong, and Yizhe Zhang. Diffucoder: Understanding and improving masked diffusion models for code generation, 2025.
- [10] Xiaochuang Han, Sachin Kumar, and Yulia Tsvetkov. Ssd-lm: Semi-autoregressive simplex-based diffusion language model for text generation and modular control, 2023.
- [11] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. Proceedings of the International Conference on Learning Representations (ICLR), 2021.
- [12] inclusionAI. Llada2.0-mini-preview. Hugging Face Model Card, 2025.
- [13] Xiangqi Jin, Yuxuan Wang, Yifeng Gao, Zichen Wen, Biqing Qi, Dongrui Liu, and Linfeng Zhang. Thinking inside the mask: In-place prompting in diffusion llms, 2025.
- [14] Jaeyeon Kim, Kulin Shah, Vasilis Kontonis, Sham Kakade, and Sitan Chen. Train for the worst, plan for the best: Understanding token ordering in masked diffusions, 2025.
- [15] Aitor Lewkowycz, Anders Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, Vinay Ramasesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo Gutman-Solo, Yuhuai Wu, Behnam Neyshabur, Guy Gur-Ari, and Vedant Misra. Solving quantitative reasoning problems with language models, 2022.
- [16] Haonan Li, Yixuan Zhang, Fajri Koto, Yifei Yang, Hai Zhao, Yeyun Gong, Nan Duan, and Timothy Baldwin. Cmmlu: Measuring massive multitask language understanding in chinese, 2023.
- [17] Xiang Lisa Li, John Thickstun, Ishaan Gulrajani, Percy Liang, and Tatsunori B. Hashimoto. Diffusion-lm improves controllable text generation. In NeurIPS, 2022.
- [18] Zhiyuan Liu, Yicun Yang, Yaojie Zhang, Junjie Chen, Chang Zou, Qingyuan Wei, Shaobo Wang, and Linfeng Zhang. dllm-cache: Accelerating diffusion large language models with adaptive caching. arXiv preprint arXiv:2506.06295, 2025.
- [19] Xinyin Ma, Runpeng Yu, Gongfan Fang, and Xinchao Wang. dkv-cache: The cache for diffusion language models. arXiv preprint arXiv:2505.15781, 2025.
- [20] Shen Nie, Fengqi Zhu, Zebin You, Xiaolu Zhang, Jingyang Ou, Jun Hu, Jun Zhou, Yankai Lin, Ji-Rong Wen, and Chongxuan Li. Large language diffusion models. arXiv preprint arXiv:2502.09992, 2025.
- [21] Jingyang Ou, Shen Nie, Kaiwen Xue, Fengqi Zhu, Jiacheng Sun, Zhenguo Li, and Chongxuan Li. Your absorbing discrete diffusion secretly models the conditional distributions of clean data. arXiv preprint arXiv:2406.03736, 2024.
- [22] Guilherme Penedo, Hynek Kydlíˇcek, Loubna Ben allal, Anton Lozhkov, Margaret Mitchell, Colin Raffel, Leandro Von Werra, and Thomas Wolf. The fineweb datasets: Decanting the web for the finest text data at scale, 2024.
- [23] Subham Sekhar Sahoo, Marianne Arriola, Yair Schiff, Aaron Gokaslan, Edgar Marroquin, Justin T Chiu, Alexander Rush, and Volodymyr Kuleshov. Simple and effective masked diffusion language models, 2024.
- [24] Mirac Suzgun, Nathan Scales, Nathanael Schärli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, Quoc V Le, Ed H Chi, Denny Zhou, , and Jason Wei. Challenging big-bench tasks and whether chain-of-thought can solve them. arXiv preprint arXiv:2210.09261, 2022.
- [25] Xiaohang Tang, Rares Dolga, Sangwoong Yoon, and Ilija Bogunovic. wd1: Weighted policy optimization for reasoning in diffusion language models, 2025.
- [26] Guanghan Wang, Yair Schiff, Subham Sekhar Sahoo, and Volodymyr Kuleshov. Remasking discrete diffusion models with inference-time scaling, 2025.

- [27] Guanghan Wang, Yair Schiff, Gilad Turok, and Volodymyr Kuleshov. d2: Improved techniques for training reasoning diffusion language models, 2025.
- [28] Xu Wang, Chenkai Xu, Yijie Jin, Jiachun Jin, Hao Zhang, and Zhijie Deng. Diffusion llms can do faster-than-ar inference via discrete diffusion forcing, 2025.
- [29] Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, et al. Mmlu-pro: A more robust and challenging multi-task language understanding benchmark. arXiv preprint arXiv:2406.01574, 2024.
- [30] Chengyue Wu, Hao Zhang, Shuchen Xue, Shizhe Diao, Yonggan Fu, Zhijian Liu, Pavlo Molchanov, Ping Luo, Song Han, and Enze Xie. Fast-dllm v2: Efficient block-diffusion large language model. arXiv preprint arXiv:2509.26328, 2025.
- [31] Chengyue Wu, Hao Zhang, Shuchen Xue, Zhijian Liu, Shizhe Diao, Ligeng Zhu, Ping Luo, Song Han, and Enze Xie. Fast-dllm: Training-free acceleration of diffusion llm by enabling kv cache and parallel decoding, 2025.
- [32] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jing Zhou, Jingren Zhou, Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang, Le Yu, Lianghao Deng, Mei Li, Mingfeng Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin, Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, and Zihan Qiu. Qwen3 technical report, 2025.
- [33] Jiacheng Ye, Shansan Gong, Liheng Chen, Lin Zheng, Jiahui Gao, Han Shi, Chuan Wu, Xin Jiang, Zhenguo Li, Wei Bi, and Lingpeng Kong. Diffusion of thoughts: Chain-of-thought reasoning in diffusion language models, 2024.
- [34] Jiacheng Ye, Zhihui Xie, Lin Zheng, Jiahui Gao, Zirui Wu, Xin Jiang, Zhenguo Li, and Lingpeng Kong. Dream 7b: Diffusion large language models, 2025.
- [35] Zebin You, Shen Nie, Xiaolu Zhang, Jun Hu, Jun Zhou, Zhiwu Lu, Ji-Rong Wen, and Chongxuan Li. Llada-v: Large language diffusion models with visual instruction tuning. arXiv preprint arXiv:2505.16933, 2025.
- [36] Siyan Zhao, Devaansh Gupta, Qinqing Zheng, and Aditya Grover. d1: Scaling reasoning in diffusion large language models via reinforcement learning, 2025.
- [37] Jeffrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha Brahma, Sujoy Basu, Yi Luan, Denny Zhou, and Le Hou. Instruction-following evaluation for large language models. arXiv preprint arXiv:2311.07911, 2023.
- [38] Fengqi Zhu, Zebin You, Yipeng Xing, Zenan Huang, Lin Liu, Yihong Zhuang, Guoshan Lu, Kangyu Wang, Xudong Wang, Lanning Wei, Hongrui Guo, Jiaqi Hu, Wentao Ye, Tieyuan Chen, Chenchen Li, Chengfu Tang, Haibo Feng, Jun Hu, Jun Zhou, Xiaolu Zhang, Zhenzhong Lan, Junbo Zhao, Da Zheng, Chongxuan Li, Jianguo Li, and Ji-Rong Wen. Llada-moe: A sparse moe diffusion language model, 2025.

### Contributors

Yuchuan Tian1∗, Yuchen Liang2∗, Shuo Zhang2, Yingte Shu1, Guangwen Yang2, Wei He1, Sibo Fang2, Tianyu Guo2, Kai Han2, Chao Xu1, Hanting Chen2†, Xinghao Chen2#, Yunhe Wang2#

- 1 State Key Lab of General AI, School of Intelligence Science and Technology, Peking University.
- 2 Huawei Technologies. ∗Equal Contribution. †Project Lead. #Corresponding Author.

### Acknowledgement

We are very grateful to Yulong Li, Xuechun Wang, Renjie Jiang, Chen Chen and Hang Zhou for their generous help.

### A Methodology Details

Attention mask for parallel training. The naive block-diffusion recipe is data-inefficient: random cropping wastes the remaining tokens of each sequence, and only a small subset of masked tokens inside the last block contributes to the loss. Unlike AR pretraining—where every token can supervise next-token prediction—switching to next-block prediction sharply reduces token utilization. We therefore restructure training so that all blocks provide learning signal in a single forward pass.

We seek to model all blockwise conditionals in parallel using a single transformer call. Instead of invoking the denoiser B times, we concatenate a noised view xt (partitioned into blocks) with the clean sequence x:

xall = xt ⊕ x (length 2L).

- A structured attention mask Mall ∈ {0, 1}2L×2L updates all token representations in one shot:

Mall =

MBD MOBC 0 MCC

. (4)

Within the noised view xt, attention is restricted to be block-wise (block-diagonal):

[MBD]ij =

1, i and j are in the same block, 0, otherwise.

From noised tokens to the clean context, we allow only earlier clean blocks as conditioning (offset block-causal):

[MOBC]ij =

 



1, clean position j lies in a block

strictly before the block of i, 0, otherwise.

Inside the clean context, we keep strict left-to-right causality (context-causal):

[MCC]ij =

1, j ≤ i, 0, j > i.

The lower-left tile is zero so the clean context never reads from the noised view, matching inference-time semantics.

Let B index all blocks and Mt be the step-dependent visibility inside the noised view. Under Eq. (4), one forward pass supplies gradients for all masked tokens across all blocks:

Lparallel(θ) = E

 −

B∈B i∈B: Mt(i)=0

log pθ xi | xt, x; Mall .

(5)

Processing xt and x jointly amortizes KV-cache construction, maximizes per-step token utilization, and empirically stabilizes training compared with randomly growing global masks. An example for L=16 and block size b=4 is illustrated in Fig. 4; but in reality, we use b=32.

- B Other Experiments

NBDiff-7B-Base. We also measure the performance of NBDIFF-7B-BASE. The comparative results for our NBDIFF-7B-BASE against strong 7B baselines is summarized in Table 4. Apart from the introduced benchmarks, we also include BBH (BIG-Bench Hard) [24], which is a curated set of particularly difficult tasks targeting abstraction, compositionality, and complex reasoning. Overall, NBDIFF-7B-BASE attains the highest macro average, surpassing Dream-v0-Base-7B and both LLaDA bases. On general knowledge, it leads on MMLU-Pro (52.7), CMMLU (76.9), CEval (75.9), and BBH (69.4), and remains competitive on MMLU (69.1, second only to Dream’s 69.5). In math, NBDIFF-7B-BASE ranks first on both GSM8K (79.6) and MATH (46.0), indicating strong multi-step and competition-style reasoning. In coding, it is consistently runner-up, slightly behind Dream-v0 [34] but ahead of the LLaDA [20] baselines. Taken together, these results show that a diffusion-style LLM can match or outperform autoregressive bases across diverse evaluations, with particularly clear gains on harder general-reasoning and Chinese benchmarks.

Base vs. SFT AR weight initialization. We attempted adapting both a pretrained Base AR checkpoint and an instruction-tuned (SFT) checkpoint into our DLM, with the comparison summarized in Table 5. Somewhat surprisingly, the Base-initialized model achieves the stronger overall balance (Though Avg 53.26 vs. 54.39 for

#### Table 4: Comparison between NBDIFF-7B-BASE and latest base-version diffusion language models. Our base model shows strong performance on general, math, and coding benchmarks.

LLaDA-8B LLaDA-MoE-7B Dream-v0 Ours-7B Benchmark Base A1B-Base Base-7B Base General

MMLU 65.9 64.6 69.5 70.1 MMLU-Pro 41.8 39.2 48.2 59.1 CMMLU 69.9 65.7 60.9 77.3 CEval 70.5 65.6 59.2 73.0 BBH 49.8 52.7 57.9 77.3

Math

GSM8K 70.7 66.4 77.8 78.8 MATH 27.3 36.1 39.6 46.0

Coding

MBPP 38.2 52.4 56.2 55.8 HumanEval 33.5 45.7 57.9 50.0

Avg 52.0 54.3 60.1 65.3

#### Table 5: Comparing DLMs adapted from Base and SFT version of loaded weights. ContextCausal gains a clear advantage in adaptation from AR.

Scheme GSM8K MATH HumanEval MBPP Avg

Qwen3-8B-Base 82.26 34.68 31.71 64.40 53.26 Qwen3-8B 81.05 32.24 42.68 61.60 54.39

SFT, SFT’s edge is driven largely by HumanEval’s small-set volatility; for other benchmarks, the Base model performs slightly better). We hypothesize two causes: 1. Objective alignment: the Base model is trained purely on next-token prediction, which better complements our masked-diffusion objective and auxiliary AR-loss, whereas SFT shifts the likelihood landscape toward instruction formats and response conventions; 2. Format priors: SFT injects stylistic and safety priors (headings, disclaimers, verbosity) that are beneficial for chat but act as spurious targets for diffusion.

