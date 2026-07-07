# arXiv:2512.14681v1[cs.CL]16Dec2025

## FAST AND ACCURATE CAUSAL PARALLEL DECODING USING JACOBI FORCING

##### Lanxiang Hu∗1 Siqi Kou∗2 Yichao Fu1 Samyam Rajbhandari3 Tajana Rosing1 Yuxiong He3 Zhijie Deng†2 Hao Zhang†1 1UC San Diego 2 Shanghai Jiao Tong University 3Snowflake

ABSTRACT

Multi-token generation has emerged as a promising paradigm for accelerating transformer-based large model inference. Recent efforts primarily explore diffusion Large Language Models (dLLMs) for parallel decoding to reduce inference latency. To achieve AR-level generation quality, many techniques adapt AR models into dLLMs to enable parallel decoding. However, they suffer from limited speedup compared to AR models due to a pretrain-to-posttrain mismatch. Specifically, the masked data distribution in post-training deviates significantly from the real-world data distribution seen during pretraining, and dLLMs rely on bidirectional attention, which conflicts with the causal prior learned during pretraining and hinders the integration of exact KV cache reuse. To address this, we introduce Jacobi Forcing, a progressive distillation paradigm where models are trained on their own generated parallel decoding trajectories, smoothly shifting AR models into efficient parallel decoders while preserving their pretrained causal inference property. The models trained under this paradigm, Jacobi Forcing Model, achieves 3.8× wall-clock speedup on coding benchmarks with minimal loss in performance. Based on Jacobi Forcing Model’s trajectory characteristics, we introduce multi-block decoding with rejection recycling, which enables up to 4.5× higher token acceptance count per iteration and nearly 4.0× wall-clock speedup, effectively trading additional compute for lower inference latency. Our code is available at https://github.com/hao-ai-lab/JacobiForcing.

1 INTRODUCTION

Modern large language models (LLMs), such as GPT-5 (OpenAI, 2025), Gemini-2.5 (DeepMind, 2025), and Kimi-K2 (Team et al., 2025), excel at complex and interactive agentic tasks. Yet, autoregressive (AR) decoding generates tokens sequentially, limiting parallelism and leading to high latency. To address this, recent work explores predicting multiple future tokens natively in transformer-based models without relying on auxiliary draft models. A popular approach is diffusion-based language models (dLLMs), which relax left-to-right generation by modeling the entire sequence jointly and decoding via full-sequence denoising (Nisonoff et al., 2024; Schiff et al., 2024; Inception Labs, 2025). This, in turn, enables highly parallelizable computation. However, open pretrained dLLMs (Ye et al., 2025; Zhu et al., 2025; Nie et al., 2025a) underperform AR models in generation quality, mainly due to their negative evidence lower bound (NELBO) training objective, a loose bound on AR’s negative log-likelihood (NLL) that is proven less efficient (Cheng et al., 2025; Nie et al., 2024; Arriola et al., 2025).

To preserve the generation quality of frontier AR models, the community has adapted high-quality AR models into dLLMs for parallel decoding (JetAstra, 2025; Wu et al., 2025b). Concretely, they perform block-wise perturbations of pretrained data by randomly masking tokens following the dLLMs recipe, and leverage these data to posttrain AR models by modifying the attention mask to enable block-wise bidirectional attention and replacing the training objective from NLL to NELBO. This adaptation delivers limited speedup under quality constraints, primarily due to a significant pretrain-to-posttrain mismatch. Specifically, enforcing block-wise bidirectional attention conflicts

∗Equal contributions. Part of work was done during Lanxiang’s internship at Snowflake. †Correspondence to Zhijie Deng, Hao Zhang <zhijied@sjtu.edu.cn, haozhang@ucsd.edu>.

with the causal prior in pretrained AR models. For instance, SDAR Cheng et al. (2025) suffers substantial quality drops when large block sizes (e.g., 64 or 128) are adopted. Moreover, the masked data distribution during post-training deviates sharply from the natural data distribution seen during pretraining, making the adaptation difficult to learn. Consequently, AR-adapted dLLMs are costly to train as shown in Figure 1, and fail to scale speedup reliably with larger block sizes, thereby underutilizing modern AI accelerators whose abundant FLOPs could otherwise be leveraged to decode more future tokens per iteration and further reduce end-to-end latency.

In this work, we introduce Jacobi Forcing, a progressive distillation technique that addresses this pretrain-to-posttrain mismatch. It trains AR models on their own generated data without any modifications of causal attention. This is made possible by collecting trajectories using Jacobi Decoding—a widely adopted parallel decoding technique for AR models (Song et al., 2021; Santilli et al., 2023). It first randomly initializes a block of n tokens and feeds it to the AR models to iteratively update it, and eventually, the block converges to the same n tokens generated by AR decoding, forming a trajectory between the randomly initialized point and the converged point. The full sequence is generated block by block. Prior works including CLLM (Kou et al., 2024) and CEED-VLA (Song et al., 2025a) design a consistency loss to map any point along the trajectory to the converged point, which in turn teaches AR models to predict multiple correct tokens in one iteration simultaneously. However, they face a similar limitation as AR-adapted dLLMs: as block size increases, the number of tokens correctly decoded per iteration remains essentially constant. Jacobi Forcing addresses this by introducing a noise-aware causal attention that teaches the model to predict the converged point within each block conditioned on previous unconverged blocks, and we show it enables more useful future tokens to emerge in each block’s trailing tails. Furthermore, Jacobi Forcing repeats this distillation procedure for the trained model and involves more noisy data with a larger block size for progressive distillation.

[Figure 1]

Figure 1: Baseline comparison.

We observe Jacobi Forcing Model has a stronger capability of generating correct future tokens conditioning on noisy context, consistent with our training objective. To better utilize this characteristic, we design a rejection-recycling and multi-block decoding algorithm for further inference optimization. Rejection recycling reuses high-quality consecutive tokens discarded from past Jacobi iterations to generate candidate token sequences, enabling the decoding of more accurate tokens via verifying multiple branches in a single iteration. Multi-block decoding maintains and refines multiple blocks simultaneously, where correct tokens are decoded in subsequent blocks even when preceding blocks remain unconverged for further speedup.

Experiments show Jacobi Forcing Model can serve as very efficient parallel decoders with up to 3.8× improvement in generation speed across coding and math benchmarks. It also effectively generates higher quality draft n-grams from future tokens within each block, as observed in Section 4. Using rejection-recycling and multi-block decoding makes use of future n-grams and further boost speedup to around 4×.

In summary, key contributions of this paper includes:

- • We introduce Jacobi Forcing to train AR models as fast parallel decoders, Jacobi Forcing Model, with up to 3.8× generation speedup.
- • We empirically observe and qualitatively verify Jacobi Forcing Model has both higher fastforwarded token count and a useful n-gram count in comparison with baseline models.
- • We propose rejection-recycling and multi-block decoding to make use of higher quality draft n-grams from future tokens within each block, and apply them to Jacobi Forcing Model boost generation speed nearly up to 4.0× across various benchmarks.

2 PRELIMINARY

This section reviews the basics of Jacobi decoding and consistency distillation training to accelerate Jacobi decoding of AR models.

- 2.1 JACOBI DECODING

Given a prompt x and a pre-trained LLM pθ(·|x) parametrized by θ, the standard AR decoding under the greedy strategy produces a response sequentially as follows:

yi = arg max

y

pθ(y | y<i,x), for i = 1,...,n, (1)

where y<i = {y1,...,yi−1}. This process requires n forward passes of the LLM to generate n tokens y≤n. The inherently sequential nature of AR decoding limits practical efficiency when generating long sequences. Jacobi decoding (Song et al., 2021; Santilli et al., 2023) addresses this bottleneck by reformulating token generation as solving a system of nonlinear equations:

f(yi,y<i,x) = 0, for i = 1,...,n, (2)

where f(yi,y<i,x) := yi − arg maxy pθ(y|y<i,x). This system can be solved in parallel using Jacobi fixed-point iteration (ort, 2000). Starting from a randomly initialized n-token sequence

y(0) = {y1(0),...,yn(0)}, the update at each iteration j is:







- y1(j+1) = arg max y

pθ(y|x)

- y2(j+1) = arg max y

pθ(y|y1(j),x)

.

yn(j+1) = arg max

y

pθ(y|y<n(j),x).

(3)

Notably, for LLM, the above n maximization problems can be solved in parallel by using a causal attention mask, i.e., only one forward pass of the LLM is required to obtain y(j+1) based on y(j). The iteration exits at some k such that y(k) = y(k−1) and we define y∗ := y(k) as the fixed point. Let J := {y(0),...,y(k)} denote the Jacobi trajectory. It can be proven that y∗ is identical to AR decoding under greedy strategy (Song et al., 2021).

To generate a long response l of length L ≫ n, Jacobi decoding is applied sequentially over blocks of size n until the <eos> token appears in a fixed point. Let yB∗

i

denote the fixed point obtained for the i-th block. The full output l is then constructed by concatenating fixed points from consecutive blocks:

l = [yB∗

1

,...,yB∗

N

], (4) where N = ⌈Ln⌉ denotes the number of blocks generated before termination.

- 2.2 CONSISTENCY DISTILLATION

Despite the promise, Jacobi decoding achieves little speedup over standard AR decoding (Santilli et al., 2023; Fu et al., 2024), as it rarely predicts more than one correct1 token within one fixedpoint iteration. To address this, recent works such as CLLMs (Kou et al., 2024) propose consistency distillation, a training approach designed to accelerate convergence to the fixed point from arbitrary states on a Jacobi trajectory. The key idea is to introduce a consistency loss that encourages an LLM pθ(·|x) to predict multiple tokens simultaneously:

Bi∼Ji DKL pθ−(yB∗

###### |x,yB∗

###### ,...,yB∗

i|x,yB∗

###### ,...,yB∗

Lc = Ei∼U{1,...,N},y

)||pθ(yB

###### ) ,

i−1

i−1

i

1

1

(5) where θ− = stopgrad(θ) and DKL denotes the KL divergence aggregated across the n tokens in a block. Here, i ∼ U{1,...,N} denotes sampling a block index uniformly at random, and yB

i ∼ Ji denotes randomly sampling from the Jacobi trajectory of the i-th block.

1By correctness, we mean alignment with the AR decoding result under a greedy sampling strategy.

CLLMs build upon this idea by first collecting Jacobi trajectories, obtained by running Jacobi decoding with pθ on a set of prompts. The model is then trained with a joint objective that combines the consistency loss in Eq. 5 with the standard AR loss, achieving up to a 2× speedup over AR decoding while maintaining quality. Similar training objectives have also been adopted for inference acceleration in other domains, such as action prediction in VLA models (Song et al., 2025a).

- 3 METHODOLOGY

In this section, we first discuss the training challenges of consistency distillation with larger block sizes n, and then present Jacobi Forcing, a progressive consistency distillation method designed to mitigate this bottleneck, and denote LLMs trained under this paradigm as Jacobi Forcing Model. Furthermore, by observing Jacobi Forcing Model’s trajectories under vanilla Jacobi decoding, we introduce rejection-recycling and multi-block decoding strategies to improve its efficiency.

3.1 JACOBI FORCING Progressive Noise Schedule. In Jacobi decoding, we maintain strict causality within each block, where each token is updated in accordance with Eq. 3. Consider the i-th block yB(j)

of size n is been decoded at some iteration step j. Assume the first c − 1 tokens have been accepted, and we denote yf as the future token as shown in Eq. 6.

i

p y | xc, yc′:f−1 , for f = c + 1,...,n, (6)

yf = arg max

y

where xc = [x,y<c] is the clean context, yc′:f−1 is the noisy2 context. While the training objective in Eq. 5 is designed to optimize correct token prediction in this setting, it’s observed from Kou et al.

(2024) that predicting yf is hard when it’s conditioned on a long noisy context yc′:f−1 under large block sizes (e.g., n = 256).

To address this challenge, we instead split a large block into smaller blocks (e.g., n = 16) with noise ratios determined by a predefined schedule {t1,...,tN}. Each ti denotes the fraction of noisy tokens in a block. The noise schedule follows a cyclic strategy with window size w, where the noise ratio linearly increases from 0 to 1 within each window, i.e.,

w − 1 w

1 w

, ti = W[j], j = i mod w. (7)

W = 0,

,...,

This progressive schedule ensures that each block retains a partially clean context, thereby shortening noisy tokens dependencies. In particular, it reduces the longest span of consecutive noisy inputs for any prediction from O(nN) assuming ti = 1 for all blocks using a random schedule to O(⌈tn⌉) using a progressive schedule, which facilitates learning. Empirically, we find this progressive schedule to be more effective than a purely random noise schedule (Table 4).

##### Progressive Distillation Loss. Let yt

bi denote the point along the i-th block Jacobi trajectory with several noisy tokens closest to ⌈tin⌉. The training objective is to predict tokens correctly within each block, aggregating losses across blocks to reduce gradient variance and stabilize optimization.

i

Accordingly, we introduce a new loss term, progressive consistency loss, which optimizes pθ under the progressive noise schedule in Eq. 7:

1 N

Lpc =

N

DKL pθ−(· | x,yB∗

,...,yB∗

1

i=1

B1,...,yBti−1

) pθ(· | x,yt

) . (8)

1

i−1

i−1

AR Loss. Kou et al. (2024) notes that using only the consistency loss (Eq. 5) must be supplemented with an AR loss to maintain generation quality. Our preliminary experiments show that using only the consistency objective (Eq. 8) produces the same effect. This motivates our inclusion of a conventional AR loss term in the final training objective to safeguard output quality:

L(θ) = Lpc + wLAR (9)

2By noisy, we refer to tokens in the non-converged point along the Jacobi trajectory that that differ from those in the fixed point at the same positions.

block 0 (unconverged)

block 0 (converged)

block 1 (unconverged)

block 1 (converged)

block 0 (unconverged)

block 0 (converged)

block 1 (unconverged)

block 1 (converged)

<bos> Implement bubble sort def \n def bubble for i _ sort

<bos> Implement bubble sort def \n def bubble for i _ sort

Noise-Aware Attention Mask (prompt length = 4, block size = 2, N = 2)

Clean Context Attention Mask (prompt length = 4, block size = 2, N = 2)

Query Index (q)

Query Index (q)

Key Index (k)

Key Index (k)

(b) noisy-context conditioned causal mask.

(a) clean-context conditioned causal mask.

Figure 2: Sequence packing with two attention mask implementations, both allow logits from clean blocks and noisy blocks to be generated with single forward pass to calculate the progressive consistency loss and AR loss in Eq. 9.

where w is a tunable weight that balances the two learning objectives.

Noise-aware Causal Attention. In CLLM, loss from each training step is computed based on KL divergance from one block instance in Eq. 5. This learning objective is to train correct token prediction in the setting where there is only a big block (Eq. 6). Moreover, in both Eq. 5 and Eq. 8, the loss term computation involves two forward passes using a conventional causal mask since each involves a distinction sequence. As a result, it requires O(2N) forward passes to compute all loss terms in Eq. 8 and O(N) backward passes to compute gradients, resulting in low training efficiency. We reduce the number of forward and backward passes from O(N) to O(1) by introducing a sequence packing technique and a block-wise sparse attention mask. We illustrate the sequence packing that interleaves yt

for the entire complete sequence in Figure 2b for Lpc computation, in contrast with conditioning each unconverged yb

bi and yb∗

i

i

only on clean tokens for consistency distillation with Lc in Figure 2a.

s

Progressive Distillation for Larger Block Sizes. In training Jacobi Forcing Model on trajectories from the original AR model, we find that speedup scales with training steps and saturates at large step counts, likely due to significant data distribution shifts from extensively trained models. To break this ceiling, we collect an additional round of Jacobi trajectories with progressively larger block sizes from the Jacobi Forcing Model empowered with multi-token prediction capability and further train it on newly generated trajectories. This yields a further 20% speedup with only minor performance degradation. Detailed training configurations are in Section 4.1.

- 3.2 INFERENCE OPTIMIZATION

Behavior of Jacobi Forcing Model. Jacobi Forcing Model is trained to have a stronger capability of generating correct future tokens conditioning on noisy tokens. Qualitative analysis in Figure 4 illustrates that it indeed brings the quality improvement: fixed-point segments emerge within the noisy tokens of the unconverged point. Furthermore, these segments progressively extend (e.g., the number of red tokens increases from point 1 to point 2 in Figure 4), even under noisy context, consistent with our training patterns. In this section, we focus on how to translating this qualitative observation of draft quality improvement into qualitative speedup.

Rejection Recycling. Prior work has shown that n-grams produced during Jacobi iterations can be verified in parallel and reused in subsequent iterations (Fu et al., 2024). As illustrated in Figure 4, such n-gram sizes could be large in Jacobi Forcing Model. If correctly verified, many tokens can be fast-forwarded in one iteration. In particular, we initialize a fixed-size n-gram pool constructed from noisy token sequences observed at unconverged points during Jacobi decoding. During decoding, if the pool contains an n-gram whose first token matches the last accepted token of the current point, we extend this token by concatenating it with its subsequent tokens to form new candidates. These candidates are then verified in parallel by appending them along the batch dimension. At

###### accepted tokens noisy tokens fixed point segments

|73594|......|12128|311|1817|1008|1817|262|262|2661|2661|2661|624|262|702|12704|22801|2561|13|13|13|55722|......|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|73594|......|12128|311|1817|1008|1091|198|262|262|12171|2661|624|262|12109|262|702|2561|16|13|15|11|......|
|73594|......|12128|311|1817|1008|1091|198|262|2661|12171|624|262|262|12109|702|12704|22081|2561|16|13|15|......|
|73594|......|12128|311|1817|1008|1091|198|262|2661|12171|624|262|12109|702|12704|22081|22081|16|16|13|15|......|
|73594|......|12128|311|1817|1008|1091|198|262|2661|12171|624|262|12109|702|12704|22801|2561|16|13|15|11|......|
|73594|......|12128|311|1817|1008|1091|198|262|2661|12171|624|262|12109|702|12704|22801|2561|16|13|15|11|......|

- 1:
- 2:
- 3:
- 4:
- 5:
- 6:

Rejectionrecycling helps me solve faster!

[Figure 2]

[Figure 3]

- Figure 4: Visualization of Jacobi Forcing Model’s trajectory under vanilla Jacobi decoding. The figure shows a partial segment of the trajectory. Blue tokens denote accepted tokens that match the fixed point at their positions. Black tokens denote unconverged noisy tokens, and we highlight them in red if more than three consecutive tokens match the fixed point regardless of position.

each iteration, we select the candidate that yields the largest number of newly accepted tokens. For instance, this strategy enables skipping from point 3 to point 5 in Figure 4, as the fixed-point segments in point 3 yield higher-quality candidates.

Multi-block Decoding. In addition to high-quality n-grams in the draft, we also observe the increasing number of stationary tokens, which are correctly predicted with preceding noisy tokens and remain unaltered through subsequent iterations. Together they yield higher quality drafts. To make use of the property, we introduce multi-block decoding, a new decoding paradigm that maintains and refines up to K blocks simultaneously. It marks the block closest to the effective KV cache boundary as the real-active block and all the other K − 1 blocks as pseudo-active blocks. Only tokens within the real-active block are accepted and committed to KV cache. Tokens in pseudo-active blocks are only pseudo-accepted, conditioning on prior blocks; once converged, pseudoactive blocks will wait until they are promoted as the real active block, where all tokens will be verified again, but now with a higher-quality draft. A detailed description is provided in Algorithm 1 (with rejection recycling) in Appendix A and with an example in Figure 3. Note that both rejection recycling and multi-block decoding are lossless as they employ greedy rejection sampling for token acceptance in the real-active block (Leviathan et al., 2022).

iteration index

committed tokens with KV cache

real-accepted tokens pseudo-accepted tokens rejected tokens

...

prompt

block 0 block 1

<bos> Implement bubble sort def bubble _ \n ( arr arr range

j

n-gram pool

accepted commit KV

real-active block threshold reached; spawning new block

<bos> Implement bubble sort def bubble _ sort

( arr arr \n

j + 1

n-gram matching and parallel verification

accepted commit KV

pseudo accepted

<bos> Implement bubble sort def bubble _ sort ( arr ) :

j + 2

accepted commit KV

n-gram caching

Figure 3: An example of multiblock decoding with rejection recycling at prompt length = 4, block size = 4, r = 0.5, K = 2.

- 4 EXPERIMENTS

- 4.1 EVALUATION SETTINGS

Models and Datasets. We evaluate Jacobi Forcing Model across coding benchmark. For coding benchmarks, we train Qwen2.5-Coder-Insutrct (Hui et al., 2024) on OpenCodeInstruct (Ahmad et al., 2025) and test on the HumanEval (Chen et al., 2021), MBPP (Austin et al., 2021). On OpenCodeInstruct, we curate question instances that come with generations that pass all unit tests, from where we use 450k prompts for trajectory generation and training. For mathematical tasks, we train Qwen2.5-Math-7B-Instruct (Yang et al., 2024) on the math split of Openthought2 (Guha et al., 2025) and test on GSM8K (Cobbe et al., 2021), and MATH (Hendrycks et al., 2021). On Openthought2, only mathematical prompts are considered, from where we apply the same training settings for trajectory generation and training.

Training Settings. All training and inference are conducted on instances equipped with 8x NVIDIA A100-80GB GPUs and 8x NVIDIA H200 GPUs. All models are trained with a learning rate of 10−6, a batch size of 4, and a max new sequence length of 2048. For Jacobi Forcing Model, we adopt a linear progressive noise schedule, initial block size at 16, window size at 16, and train for 10k steps,

- Table 1: Performance and efficiency on coding benchmarks, HumanEval and MBPP, grouped by decoding family. For AR-based models, all methods adopt Qwen2.5-Coder-7B-Instruct. For Jacobi Forcing Model, MR stands for employing the multi-block and rejection-recycling decoding algorithm introduced in Algorithm 1. DC stands for using bi-directional dual cache from fast-dLLM. For both Fast-dLLM and D2F, we choose the Dream-7B as it’s significantly faster with similar or better performance than LLaDA-7B. For CLLM*, we follow mostly the same recipe in CLLM but with new sequence packing technique (without progressive training on larger block sizes). The speedup ratio is relative to the AR baseline.

Benchmark Family Method TPF ↑ TPS ↑ Speedup ↑ Accuracy ↑

AR 1.00 41.3 1.00× 87.8 Jacobi 1.03 39.9 0.97× 87.8 CLLM* 2.68 103.3 2.50× 87.8 Jacobi Forcing Model 4.01 159.5 3.86× 83.5 Jacobi Forcing Model (MR) 4.09 163.9 3.97× 83.5

AR-based

HumanEval

LLaDA-Instruct 1.00 2.8 0.07× 36.0 Dream-Base 1.00 20.2 0.49× 54.3 Fast-dLLM (DC) 1.80 60.0 1.45× 53.0 D2F 2.50 73.2 1.77× 54.3

Diffusion-based

AR 1.00 43.1 1.00× 74.3 Jacobi 1.01 42.4 0.98× 74.3 CLLM* 2.10 80.1 1.94× 71.4 Jacobi Forcing Model 2.74 110.7 2.57× 70.4 Jacobi Forcing Model (MR) 2.84 113.0 2.62× 70.4

AR-based

MBPP

LLaDA-Instruct 1.00 0.9 0.02× 39.0 Dream-Base 1.00 10.4 0.24× 56.2 Fast-dLLM (DC) 1.90 73.2 1.70× 51.0 D2F 2.30 105.0 2.44× 55.2

Diffusion-based

and a second round of training with block size at 32, window size at 8, and train for another 10k steps. Ablation studies on parameter choices are presented in Section 4.3.

Baselines. Our main objective in this section is to compare performance and efficiency between diffusion-based parallel decoders and the AR-based parallel decoder, Jacobi Forcing Model. The dLLM baselines also have the capability of generating a single block of tokens or multiple consecutive blocks of tokens together. Specifically, we compare Jacobi Forcing Model with state-ofthe-art (SOTA) dLLMs including LLaDA-7B (Nie et al., 2025b), Dream-7B (Ye et al., 2025), fastdLLM (Wu et al., 2025c) and D2F (Wang et al., 2025). We also compare Jacobi Forcing Model with AR-based parallel decoder, including vanilla Jacobi decoding (Santilli et al., 2023) and CLLM (Kou et al., 2024). In this work, we do not focus on speculative decoding methods, because the models themselves don’t serve as parallel decoders without supplemental architecture modifications (e.g. via additional heads) (Cai et al., 2024; Li et al., 2024b;c; 2025) or separate draft models (Leviathan et al., 2022; Liu et al., 2024). In addition, to situate Jacobi Forcing Model among broader AR-acceleration techniques, we present in the Appendix B a complementary comparison with speculative decoding and consistency-distilled baselines.

4.2 RESULTS

Performance. The performance metrics are the greedy generations’ strict accuracy (pass@1) on HumanEval and MBPP. Table 1 compares Jacobi Forcing Model with both dLLMs and Jacobi decoding baselines. On A100 GPUs, our results show that on both benchmarks, Jacobi Forcing Model consistently achieves competitive accuracy with a much better speedup at the same parameter scale. In particular, for structured generations like Python coding, Jacobi Forcing Model achieves 3.6× speedup in comparison with the AR baseline, 53.3 ∼ 7.4× speedup comparing to dLLM baselines, and 2.0× comparing to optimized dLLM baselines including Fast-dLLM and D2F with techniques like adding block-wise KV cache, bidirectional KV cache and pipelined parallel decoding. For speedup evaluation, we run all evaluations with a block size of 128 except for Jacobi Forcing Model (MR) since MR takes extra FLOPs for multiblock decoding and parallel verification.

- Table 2: Performance and efficiency on math benchmarks, GSM8K and MATH, grouped by decoding family. For AR-based models, all methods adopt Qwen2.5-Math-7B-Instruct.

Benchmark Family Method TPF ↑ TPS ↑ Speedup ↑ Solve Rate ↑

AR 1.00 41.8 1.00× 92.4 Jacobi 1.05 42.2 1.02× 92.4 CLLM* 2.25 86.8 2.08× 92.2 Jacobi Forcing Model 3.72 146.1 3.50× 91.4 Jacobi Forcing Model (MR) 4.04 154.9 3.71× 91.4

AR-based

GSM8K

LLaDA-Instruct 1.00 7.2 0.17× 77.4 Dream-Base 1.00 9.5 0.23× 75.0 Fast-dLLM (DC) 2.10 49.8 1.19× 75.0 D2F 3.10 91.2 2.18× 77.6

Diffusion-based

AR 1.00 41.3 1.00× 77.0 Jacobi 1.02 41.0 0.99× 77.0 CLLM* 2.23 84.4 2.04× 77.2 Jacobi Forcing Model 3.82 150.7 3.65× 77.4 Jacobi Forcing Model (MR) 3.98 152.0 3.68× 77.4

AR-based

MATH

LLaDA-Instruct 1.00 21.1 0.51× 23.7 Dream-Base 1.00 9.9 0.24× 35.8 Fast-dLLM (DC) 1.90 67.0 1.62× 37.1 D2F 2.60 98.8 2.39× 35.4

Diffusion-based

Moreover, we report the speedup and problem solve rate (test@1) on GSM8K and MATH in Table 2. Across both benchmarks, the Jacobi Forcing Model substantially outperforms the AR baseline with

- 3.70× speedup while preserving competitive accuracy. In the MATH benchmark, Jacobi Forcing Model delivers a 150.7 TPS while even slightly improving the solve rate from 77.0% to 77.4%, highlighting its ability to achieve both high efficiency and accuracy.

We also present speedup comparison across different AR-based techniques with Jacobi Forcing Model on B200 in Table 3 as it comes with a better fast-forward count to TPS conversion rate with more compute on B200.

Table 3: Speedup on HumanEval tested on B200 using same settings and speedup ratio over A100.

Method TPF ↑ TPS ↑ Speedup ↑

AR 1.0 83.0 1.00× Jacobi 1.03 84.7 1.02× CLLM* 2.68 207.4 2.49× Jacobi Forcing Model 4.01 301.7 3.63× Jacobi Forcing Model (MR) 4.21 328.0 3.95×

On B200, with the block size at 128 and verification size at 4 (rationale provided in Section 4.3), we apply multi-block decoding using Jacobi Forcing Model and the results are presented in Figure 3. The running window method is an optimized variant of Jacobi decoding designed for settings where many tokens are accepted per iteration. It maintains a fixed-size active block by replenishing draft tokens to the original block size as accepted tokens are committed to the KV cache. The results demonstrate that multi-block decoding with rejection recycling consistently achieves the highest number of fastforwarded tokens per iteration, particularly in the larger block-size regime as shown in Figure 5b.

- 4.3 ABLATION STUDY

Training Noise schedules. We evaluate three types of noise schedules: random, linear progressive, and reverse progressive. In the random schedule, the noise step ti for each block is sampled uniformly as ti ∼ U(1,...,N) during sequence packing in Jacobi Forcing Model training. The linear progressive schedule follows Eq. 7, while the reverse progressive schedule applies a linearly decreasing noise ratio from 1 to 0 within each window. Results in Table 4 show that the linear progressive schedule significantly outperforms the other two when the window size is 8. Intuitively, with N = 16, this schedule corresponds to adding noise more aggressively across blocks within each window, roughly two additional noisy tokens per future block, until the final block where all tokens are noisy.

Training Mask types. We train Jacobi Forcing Model on the objective in Eq. 8 with noiseconditioned mask implementation (Figure 2b). An alternative implementation of the mask is to

Table 4: Inference results for block size = 256 with N = 16, tmin = 0.0 and tmax = 1.0. Acc. = pass@1 accuracy (%) on HumanEval. The checkpoints are trained with Qwen2.5-Coder-7BInstruct on 10k randomly sampled instances from our OpenCodeInstruct trajectory dataset. Notice that for ablation purpose, the checkpoints are not trained with full datascale as in Table 1. Reverse progressive is significantly worse than other schedule and we only conduct ablation for one choice of window size.

Random Linear Progressive Reverse Progressive Acc. iter/token Acc. iter/token Acc. iter/token

Window Size

8 82.9 0.53 84.7 0.48 – – 16 83.5 0.51 81.7 0.46 82.9 0.62 32 83.5 0.53 84.1 0.49 – –

[Figure 4]

[Figure 5]

(b) fast-forward count vs. block size on HumanEval using three decoding strategies on NVIDIA H200 GPU. Notice larger block size provides more fast-forward token count for multi-block decoding with rejection recycling.

(a) Speedup vs. (log-scaled) block size at fixed fastforwarding count per iteration on NVIDIA H200 GPU, using Jacobi decoding at prompt length = 128, generation length = 256 at varying TPF rates.

- Figure 5: Effect of block size choices on fast-forward counts and wall-clock speedup under different settings. We choose the maximum block size on hardware without sacrificing wall-clock speedup.

condition all blocks within a window on a clean context. In other words, for every query, it sees blocks from all preceding windows as of Figure 2a]), and all blocks within its own window as of Figure 2b. Intuitively, it makes token predictions in later windows and blocks easier to learn because now they are conditioned on a cleaner context. We summarize results in Table 5, where it shows noise-conditioned mask is more effective in empowering Jacobi Forcing Model with speedup while maintaining generation quality.

Inference FLOPs Utilization Analysis. Jacobi Forcing Model (MR) involves both multiblock decoding and rejection-recycling, where each technique consumes extra FLOPs for parallel drafting and parallel verification, respectively. To maximize hardware utilization, we experiment with how end-to-end decoding latency changes as the total number of decoded tokens changes. We use Jacobi decoding to run the experiments and the results are shown in Figure 5a. On H200 GPUs, Jacobi decoding with block sizes up to 64 shows no latency penalty and only minor degradation at 128, particularly in the high fast-forwarding regime. The result is consistent across accepted token counts fixed at 2,3,4,5, indicating that up to 126 tokens can be decoded in parallel with shared KV without significant latency overhead. We provide a more detailed analysis in Appendix D.

Table 5: Effects of applying noise-conditioned mask (NC) or noise-conditioned mask with intrawindow clean context (NC-IC) for Jacobi Forcing Model training, and evaluated on HumanEval with A100.

Method Speedup↑ Acc. NC 3.6× 82.3 NC-IC 1.9× 82.3

Inference Configuration Search. Beyond block size, the main tunable parameters for Jacobi Forcing Model (MR) inference are verification size (entries verified in parallel with shared KV for rejection recycling), number of blocks, and initialization threshold. We observe that performance gains

from additional blocks saturate at block size = 2 as later drafts degrade quickly. The initialization threshold, defined as the fraction of the first block completed before launching the next, can be optimized via grid search and shows consistently optimal performance at r = 0.85 for block size 64 across verification sizes 2 to 8. For maximum FLOPs utilization, we use block size = 64, verification size = 4, where wall-clock speedup remains stable until parallel decoding exceeds 256 tokens. More details on inference configuration search given the FLOPs budget can be found in Appendix E.

- 5 RELATED WORK

Discrete Text Diffusion. dLLMs represent a new paradigm that challenges traditional autoregressive (AR) modeling by replacing left-to-right causality with iterative denoising, enabling parallel multi-token generation (Li et al., 2024a; Nisonoff et al., 2024; Schiff et al., 2024). Closedsource dLLMs (e.g., Gemini Diffusion (Google DeepMind, 2025; Inception Labs, 2025; Song et al., 2025b)) show huge throughput improvement while maintaining competitive code and text quality, underscoring better accelerator utilization. On the open-source side, community dLLMs with released code and weights delivered strong throughput and controllability via parallel iterative denoising, yet remaining less efficient than autoregressive decoding (Ye et al., 2025; Zhu et al., 2025; Nie et al., 2025a; JetAstra, 2025; Gong et al., 2025). Recent efforts (Arriola et al., 2025; Wu et al., 2025c; Liu et al., 2025) further push the efficiency and scalability of dLLMs.

Jacobi Decoding. Jacobi decoding reframes AR generation as a parallel fixed-point update over all positions, with convergence linked to greedy AR, and has been instantiated using Jacobi (GaussSeidel) iterations (Song et al., 2021; Santilli et al., 2023). Building on this, follow-ups either refine the decoding procedure or train models as parallel decoders to exploit parallel: CLLMs (Kou et al.,

- 2024) fine-tune LLMs with consistency distillation to predict multiple correct tokens per iteration and speed convergence; CEED-VLA (Song et al., 2025a) brings the similar idea to robotics. Other strands adapt Jacobi to new regimes, including FastCoT (Zhang et al., 2023) for reasoning with parallel CoT updates, Speculative Jacobi Decoding (Teng et al., 2024) for sampling in AR Testto-Image, and MSN, TR-Jacobi (Wang et al., 2024) that injects denoising training and a retrievalaugmented Jacobi strategy.

Speculative Decoding. Speculative decoding speeds up AR generation by letting a lightweight drafter propose several future tokens and having the target model verify them in one pass (Leviathan et al., 2022; Chen et al., 2023). It preserves the target model’s distribution while reducing latency. Subsequent work improves proposal quality and verification efficiency: online speculative decoding (OSD) (Liu et al., 2024) adapts draft models to user query distributions via continual distillation, substantially improving token acceptance and reducing inference latency. Medusa (Cai et al., 2024) adds multi-head drafters to the base LM to produce verifiable token blocks; EAGLE, EAGLE-2 (Li et al., 2024b;c) reuse target features for feature-level drafting, and EAGLE-3 (Li et al., 2025) scales this idea with multi-layer fusion. Lookahead Decoding (Fu et al., 2024), PLD (Saxena, 2023; Somasundaram et al., 2024), and REST (He et al., 2023) dispense with a separate drafter, instead synthesizing speculative candidates directly from context or future tokens. The self-speculative decoding paradigm shares a close connection with the Jacobi decoding adopted in this work.

- 6 CONCLUSION

In this work, we propose a progressive distillation technique for training AR models as faster and more accurate parallel decoders compared to dLLMs. Unlike CLLM (Kou et al., 2024), which directly trains models to predict large blocks of tokens in parallel, our approach introduces a progressively more difficult learning objective. This is achieved through a progressive noise schedule, combined with a sequence packing strategy and a noise-aware causal mask, enabling parallel token prediction conditioned on noise. The model is further improved through iterative training, where trajectories are regenerated with progressively larger block sizes. The resulting model, Jacobi Forcing Model, achieves a 3.8× speedup while largely preserving accuracy. Analysis of its generated trajectories shows that Jacobi Forcing Model produces high-quality draft tokens toward the tail of sequences. In addition, we introduce rejection recycling and multi-block decoding, which together bring tokens accepted per iteration to 4.5× as high with nearly 4× speedup on HumanEval using on both A100 and B200 GPUs.

ETHICS STATEMENT

All authors have read and adhere to the ICLR Code of Ethics. This work does not involve human subjects, sensitive personal data, or experiments with the potential to cause harm. No confidential or proprietary data were used. The methods and experiments were conducted in accordance with principles of research integrity, fairness, and transparency. Potential societal impacts, including limitations and biases of large language models, are explicitly discussed in the paper. All conclusions are the sole responsibility of the authors.

REPRODUCIBILITY STATEMENT

We have made significant efforts to ensure the reproducibility of our results. Detailed descriptions of the models, datasets been used, as well as hyperparameter choices are included in the main text. All datasets used are publicly available, and the preprocessing steps are fully documented. Ablation studies are provided to validate robustness of results. These resources collectively allow independent researchers to verify and reproduce our work.

#### USE OF LLM

During the preparation of this manuscript, large language model was used to refine grammar and improve clarity. The authors carefully reviewed and revised all outputs to ensure the text reflects their original ideas and take full responsibility for the final content, including all statements and conclusions.

REFERENCES

Iterative solution of nonlinear equations in several variables. SIAM, 2000.

Wasi Uddin Ahmad, Aleksander Ficek, Mehrzad Samadi, Jocelyn Huang, Vahid Noroozi, Somshubra Majumdar, and Boris Ginsburg. Opencodeinstruct: A large-scale instruction tuning dataset for code llms. arXiv preprint arXiv:2504.04030, 2025. URL https://arxiv.org/ abs/2504.04030. Introduces the OpenCodeInstruct dataset of 5M instruction-code samples and reports fine-tuning results on code LLMs; improves performance on HumanEval, MBPP, LiveCodeBench, and BigCodeBench.

Marianne Arriola, Aaron Kerem Gokaslan, Justin T. Chiu, Zhihan Yang, Zhixuan Qi, Jiaqi Han, Subham Sekhar Sahoo, and Volodymyr Kuleshov. Block diffusion: Interpolating between autoregressive and diffusion language models. In Proceedings of the 2025 International Conference on Learning Representations (ICLR 2025), 2025. URL https://arxiv.org/abs/2503. 09573. Oral.

Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, and Charles Sutton. Program synthesis with large language models. arXiv preprint arXiv:2108.07732, 2021.

Tianle Cai, Yuhong Li, Zhengyang Geng, Hongwu Peng, Jason D. Lee, Deming Chen, and Tri Dao. Medusa: Simple llm inference acceleration framework with multiple decoding heads. Proceedings of Machine Learning Research, 235:5209–5235, 2024. URL https://arxiv.org/ abs/2401.10774. Introduces Medusa-1 and Medusa-2: augmenting LLMs with parallel decoding heads to speed inference.

Charlie Chen, Sebastian Borgeaud, Geoffrey Irving, Jean-Baptiste Lespiau, Laurent Sifre, and John Jumper. Accelerating large language model decoding with speculative sampling. arXiv preprint arXiv:2302.01318, 2023.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Gabriele Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray,

Nick Ryder, Michael Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian, Clemens Winter, Phil Tillet, Felipe Petroski Such, Reid Cummings, Matthias Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss, William Guss, Alex Nichol, Michael Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin, Suchir Balaji, Sukhdeep Jain, William Saunders, Christopher Hesse, Andrew Carr, Aitor Lewkowycz, Conor Durkan, Diego De Las Casas, Madeleine Li, Susan Hoffman, Bowen Wu, Frederick Kelton, Peter Jacobs, Rewon Chen, Sandhini Agrawal, Shantanu Sastry, Amanda Askell, Yuntao Bai, Daniel Ziegler, Michael Steinberg, Paul Smolensky, Gretchen Krueger, Sam McCandlish, Dario Amodei, Ilya Sutskever, Tom Brown, and Jared Kaplan. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021.

Zigeng Chen, Gongfan Fang, Xinyin Ma, Ruonan Yu, and Xinchao Wang. dparallel: Learnable parallel decoding for dllms, 2025. URL https://arxiv.org/abs/2509.26488.

Shuang Cheng, Yihan Bian, Dawei Liu, Yuhua Jiang, Yihao Liu, Linfeng Zhang, Wenhai Wang, Qipeng Guo, Kai Chen, Biqing Qi, and Bowen Zhou. Sdar: A synergistic diffusion-autoregression paradigm for scalable sequence generation. arXiv preprint arXiv:2510.06303, 2025.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

Google DeepMind. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next-generation agentic capabilities. arXiv preprint arXiv:2507.06261, July 2025. URL https://arxiv.org/abs/2507.06261. Also see “Gemini 2.5: Our most intelligent AI model” blog post, March 25, 2025.

Yichao Fu, Peter Bailis, Ion Stoica, and Hao Zhang. Break the sequential dependency of llm inference using lookahead decoding. arXiv preprint, (arXiv:2402.02057), 2024. URL https: //arxiv.org/abs/2402.02057.

Shansan Gong, Shivam Agarwal, Yizhe Zhang, Jiacheng Ye, Lin Zheng, Mukai Li, Chenxin An, Peilin Zhao, Wei Bi, Jiawei Han, Hao Peng, and Lingpeng Kong. Scaling diffusion language models via adaptation from autoregressive models. In Proceedings of the 2025 International Conference on Learning Representations (ICLR), 2025. URL https://arxiv.org/abs/ 2410.17891. Presents DiffuGPT and DiffuLLaMA (also “Diffullama”) – adapting AR models to diffusion LMs; shows performance competitive with AR counterparts using ¡200B tokens.

Google DeepMind. Gemini diffusion. Experimental research model / preview, 2025. URL https: //deepmind.google/models/gemini-diffusion/. Demonstration blog post; details such as full author list not publicly released.

Etash Guha, Ryan Marten, Sedrick Keh, Negin Raoof, Georgios Smyrnis, Hritik Bansal, Marianna Nezhurina, Jean Mercat, Trung Vu, Zayne Sprague, Ashima Suvarna, Benjamin Feuer, Liangyu Chen, Zaid Khan, Eric Frankel, Sachin Grover, Caroline Choi, Niklas Muennighoff, Shiye Su, Wanjia Zhao, John Yang, Shreyas Pimpalgaonkar, Kartik Sharma, Charlie Cheng-Jie Ji, Yichuan Deng, Sarah Pratt, Vivek Ramanujan, Jon Saad-Falcon, Jeffrey Li, Achal Dave, Alon Albalak, Kushal Arora, Blake Wulfe, Chinmay Hegde, Greg Durrett, Sewoong Oh, Mohit Bansal, Saadia Gabriel, Aditya Grover, Kai-Wei Chang, Vaishaal Shankar, Aaron Gokaslan, Mike A. Merrill, and Tatsunori Hashimoto. Openthoughts: Data recipes for reasoning models. 2025. URL https://arxiv.org/abs/2506.04178. Describes the OpenThoughts project and the OpenThoughts2 dataset and reasoning models (OpenThinker2).

Zhenyu He, Zexuan Zhong, Tianle Cai, Jason D Lee, and Di He. Rest: Retrieval-based speculative decoding. arXiv preprint arXiv:2311.08252, 2023.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874, 2021.

Binyuan Hui, Jian Yang, Zeyu Cui, Jiaxi Yang, Dayiheng Liu, Lei Zhang, Tianyu Liu, Jiajun Zhang, Bowen Yu, Kai Dang, et al. Qwen2.5-coder: A code-specialized instruction language model.

arXiv preprint arXiv:2409.12186, 2024. URL https://arxiv.org/abs/2409.12186. Describes the Qwen2.5-Coder family and its instruction-tuned versions; includes 7B instruct variant.

Inception Labs. Mercury: Ultra-fast language models based on diffusion. arXiv preprint, (arXiv:2506.17298), 2025. URL https://arxiv.org/abs/2506.17298.

JetAstra. Sdar: Synergy of diffusion & autoregression. https://github.com/JetAstra/ SDAR, 2025. Code repository.

Siqi Kou, Lanxiang Hu, Zhezhi He, Zhijie Deng, Hao Zhang, et al. Consistency large language models: A family of efficient parallel decoders. arXiv preprint arXiv:2403.00835, 2024. URL https://arxiv.org/abs/2403.00835. Introduces CLLMs, which are trained via a consistency loss so that they can decode multiple tokens in parallel while approximating AR decoding; shows 2.4× to 3.4× speedups with preserved generation quality.

Yaniv Leviathan, Matan Kalman, and Yossi Matias. Fast inference from transformers via speculative decoding. arXiv preprint arXiv:2211.17192, 2022. URL https://arxiv.org/abs/ 2211.17192. Introduces the speculative decoding (draft + verify) paradigm for accelerating autoregressive models without changing output distributions.

Xiner Li, Yulai Zhao, Chenyu Wang, Gabriele Scalia, Gokcen Eraslan, Surag Nair, Tommaso Biancalani, Shuiwang Ji, Aviv Regev, Sergey Levine, and Masatoshi Uehara. Derivative-free guidance in continuous and discrete diffusion models with soft value-based decoding. arXiv preprint, (arXiv:2408.08252), 2024a. URL https://arxiv.org/abs/2408.08252.

Yuhui Li, Fangyun Wei, Chao Zhang, et al. Eagle: Extrapolation algorithm for greater languagemodel efficiency. In ICML / appropriate venue, 2024b. URL https://github.com/ SafeAILab/EAGLE. Uses feature extrapolation on second-top hidden states to propose drafts, reducing forward passes.

Yuhui Li, Fangyun Wei, Chao Zhang, et al. Eagle-2: Faster inference of language models with dynamic draft trees. arXiv preprint arXiv:2406.16858, 2024c. URL https://arxiv.org/ abs/2406.16858. Introduces context-aware dynamic drafting (tree structure) to EAGLE, improving acceptance and speedups.

Yuhui Li, Fangyun Wei, Chao Zhang, and Hongyang Zhang. Eagle-3: Scaling up inference acceleration of large language models via training-time test. arXiv preprint arXiv:2503.01840, 2025. URL https://arxiv.org/abs/2503.01840. Improves speculative decoding by abandoning feature prediction, using multi-layer feature fusion, and enabling better scalability of draft performance.

Xiaoxuan Liu, Lanxiang Hu, Peter Bailis, Alvin Cheung, Zhijie Deng, Ion Stoica, and Hao Zhang. Online speculative decoding. In Proceedings of the 41st International Conference on Machine Learning, pp. 31131–31146. PMLR, 2024. URL https://proceedings.mlr.press/ v235/liu24y.html.

Zhiyuan Liu, Yicun Yang, Yaojie Zhang, Junjie Chen, Chang Zou, Qingyuan Wei, Shaobo Wang, and Linfeng Zhang. dllm-cache: Accelerating diffusion large language models with adaptive caching. arXiv preprint arXiv:2506.06295, 2025. URL https://arxiv.org/abs/2506. 06295. Adaptive caching for diffusion LLMs (LLaDA, Dream); reuse of computations across diffusion steps.

Shen Nie, Fengqi Zhu, Chao Du, Tianyu Pang, Qian Liu, Guangtao Zeng, Min Lin, and Chongxuan Li. Scaling up masked diffusion models on text. In The Thirteenth International Conference on Learning Representations, 2024.

Shen Nie, Fengqi Zhu, Zebin You, Xiaolu Zhang, Jingyang Ou, Jun Hu, Jun Zhou, Yankai Lin, Ji-Rong Wen, and Chongxuan Li. Large language diffusion models. arXiv preprint arXiv:2502.09992, 2025a. URL https://arxiv.org/abs/2502.09992. Introduces LLaDA, a diffusion model trained from scratch; competitive with autoregressive models of similar scale.

Shen Nie, Fengqi Zhu, Zebin You, Xiaolu Zhang, Jingyang Ou, Jun Hu, Jun Zhou, Yankai Lin, Ji-Rong Wen, and Chongxuan Li. Large language diffusion models. arXiv preprint arXiv:2502.09992, 2025b. URL https://arxiv.org/abs/2502.09992. LLaDA is a diffusion LLM trained from scratch under masking and reverse processes.

Hunter Nisonoff, Junhao Xiong, Stephan Allenspach, and Jennifer Listgarten. Unlocking guidance for discrete state-space diffusion and flow models. arXiv preprint, (arXiv:2406.01572), 2024. URL https://arxiv.org/abs/2406.01572.

OpenAI. Introducing gpt-5. OpenAI blog post, August 7 2025. URL https://openai.com/ index/introducing-gpt-5/. Also see the GPT-5 system card (PDF, August 13, 2025).

Andrea Santilli, Silvio Severino, Emilian Postolache, Valentino Maiorca, Michele Mancusi, Riccardo Marin, and Emanuele Rodol`a. Accelerating transformer inference for translation via parallel decoding. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (ACL 2023, Long Papers), pp. 12336–12355, Toronto, Canada, 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.acl-long.689. URL https: //aclanthology.org/2023.acl-long.689.

Apoorv Saxena. Prompt lookup decoding (pld). https://github.com/apoorvumang/ prompt-lookup-decoding, 2023. Training-free speculative decoding via prompt n-gram retrieval.

Yair Schiff, Subham Sekhar Sahoo, Hao Phung, Guanghan Wang, Sam Boshar, Hugo Dalla-torre, Bernardo P. de Almeida, Alexander Rush, Thomas Pierrot, and Volodymyr Kuleshov. Simple guidance mechanisms for discrete diffusion models. arXiv preprint, (arXiv:2412.10193), 2024. URL https://arxiv.org/abs/2412.10193.

Shwetha Somasundaram, Anirudh Phukan, and Apoorv Saxena. Pld+: Accelerating llm inference by leveraging language model artifacts. arXiv preprint arXiv:2412.01447, 2024.

Wenxuan Song, Jiayi Chen, Pengxiang Ding, Yuxin Huang, Han Zhao, Donglin Wang, and Haoang Li. Ceed-vla: Consistency vision-language-action model with early-exit decoding. arXiv preprint arXiv:2506.13725, 2025a. URL https://arxiv.org/abs/2506.13725. Presents methods for consistency distillation, mixed-label supervision, and early-exit decoding to accelerate inference in Vision-Language-Action models with minimal performance loss.

Yang Song, Chenlin Meng, Renjie Liao, and Stefano Ermon. Accelerating feedforward computation via parallel nonlinear equation solving. In International Conference on Machine Learning, pp. 9791–9800. PMLR, 2021.

Yuxuan Song, Zheng Zhang, Cheng Luo, Pengyang Gao, Fan Xia, Hao Luo, Zheng Li, Yuehang Yang, Hongli Yu, Xingwei Qu, Yuwei Fu, Jing Su, Ge Zhang, Wenhao Huang, Mingxuan Wang, Lin Yan, Xiaoying Jia, Jingjing Liu, Wei-Ying Ma, Ya-Qin Zhang, Yonghui Wu, and Hao Zhou. Seed diffusion: A large-scale diffusion language model with high-speed inference. arXiv preprint, 2025b. URL https://arxiv.org/abs/2508.02193.

Kimi Team, Yifan Bai, Yiping Bao, Guanduo Chen, Jiahao Chen, Ningxin Chen, Ruijue Chen, Yanru Chen, Yuankun Chen, Yutian Chen, et al. Kimi k2: Open agentic intelligence. arXiv preprint arXiv:2507.20534, 2025.

Yao Teng, Han Shi, Xian Liu, Xuefei Ning, Guohao Dai, Yu Wang, Zhenguo Li, and Xihui Liu. Accelerating auto-regressive text-to-image generation with training-free speculative jacobi decoding. arXiv preprint, (arXiv:2410.01699), 2024. URL https://arxiv.org/abs/2410.01699.

Xu Wang, Chenkai Xu, Yijie Jin, Jiachun Jin, Hao Zhang, and Zhijie Deng. Diffusion llms can do faster-than-ar inference via discrete diffusion forcing. arXiv preprint arXiv:2508.09192, 2025. URL https://arxiv.org/abs/2508.09192. Introduces D2F: a hybrid AR-diffusion approach enabling KV cache and inter-block parallel decoding for dLLMs.

Yixuan Wang, Xianzhen Luo, Fuxuan Wei, Yijun Liu, Qingfu Zhu, Xuanyu Zhang, Qing Yang, Dongliang Xu, and Wanxiang Che. Make some noise: Unlocking language model parallel inference capability through noisy training. arXiv preprint arXiv:2406.17404, 2024.

Chengyue Wu, Hao Zhang, Shuchen Xue, Shizhe Diao, Yonggan Fu, Zhijian Liu, Pavlo Molchanov, Ping Luo, Song Han, and Enze Xie. Fast-dllm v2: Efficient block-diffusion large language model. arXiv preprint arXiv:2509.26328, 2025a.

Chengyue Wu, Hao Zhang, Shuchen Xue, Shizhe Diao, Yonggan Fu, Zhijian Liu, Pavlo Molchanov, Ping Luo, Song Han, and Enze Xie. Fast-dllm v2: Efficient block-diffusion llm, 2025b. URL https://arxiv.org/abs/2509.26328.

Chengyue Wu, Hao Zhang, Shuchen Xue, Zhijian Liu, Shizhe Diao, Ligeng Zhu, Ping Luo, Song Han, and Enze Xie. Fast-dllm: Training-free acceleration of diffusion llm by enabling kv cache and parallel decoding. arXiv preprint arXiv:2505.22618, 2025c. URL https: //arxiv.org/abs/2505.22618. Inference acceleration for diffusion LLMs; block-wise KV cache; confidence-aware parallel decoding.

An Yang, Beichen Zhang, Binyuan Hui, Bofei Gao, Bowen Yu, Chengpeng Li, Dayiheng Liu, Jianhong Tu, Jingren Zhou, Junyang Lin, Keming Lu, Mingfeng Xue, Runji Lin, Tianyu Liu, Xingzhang Ren, and Zhenru Zhang. Qwen2.5-math technical report: Toward mathematical expert model via self-improvement. arXiv preprint arXiv:2409.12122, 2024.

Jiacheng Ye, Zhihui Xie, Lin Zheng, Jiahui Gao, Zirui Wu, Xin Jiang, Zhenguo Li, and Lingpeng Kong. Dream 7b: Diffusion large language models. arXiv preprint arXiv:2508.15487, 2025. URL https://arxiv.org/abs/2508.15487. Dream-Base and Dream-Instruct variants released; uses discrete diffusion modeling, AR initialization, context-adaptive token-level noise rescheduling.

Hongxuan Zhang, Zhining Liu, Yao Zhao, Jiaqi Zheng, Chenyi Zhuang, Jinjie Gu, and Guihai Chen. Fast chain-of-thought: A glance of future from parallel decoding leads to answers faster. arXiv preprint arXiv:2311.08263, 2023.

Lefan Zhang, Xiaodan Wang, Yanhua Huang, and Ruiwen Xu. Learning harmonized representations for speculative sampling. arXiv preprint arXiv:2408.15766, 2025.

Fengqi Zhu, Rongzhen Wang, Shen Nie, Xiaolu Zhang, Chunwei Wu, Jun Hu, Jun Zhou, Jianfei Chen, Yankai Lin, Ji-Rong Wen, and Chongxuan Li. Llada 1.5: Variance-reduced preference optimization for large language diffusion models. arXiv preprint arXiv:2505.19223, 2025. URL https://arxiv.org/abs/2505.19223. Applies RL-style alignment (preference optimization) to LLaDA, reducing variance in ELBO estimators.

Algorithm 1 MULTIBLOCK DECODING + REJECTION RECYCLING

- 1: Init: Create a set of blocks {b} with one real–active block RA: draft tokens qRA randomly initialized, accepted tokens aRA = ∅ ; For all other blocks b, set qb = ∅, ab = ∅, and mark as pseudo-active.
- 2: Initialize candidate pool N = ∅, spawn ratio r, threshold s = ⌈rn⌉, block size n.
- 3: while iters < max do
- 4: Assemble input y: Concatenate qRA, then for each pseudo-active b, append ab (no logits) and qb (collect logits). Resize cache to batch y.
- 5: Forward: Run model pθ(y) to produce logits.
- 6: for each block b with span (start,L) do
- 7: Verification (with rejection-recycling): Greedy prediction g = arg max logits; accept longest matching prefix of qb using g (or g ∪ N if b = RA); update ab.
- 8: if b = RA and EOS encountered in accepted region then
- 9: return committed output.
- 10: end if
- 11: Tail update: If partial accept, set qb ← [next∥gtail] (and if b = RA: push rejected tail to update N and qRA); else qb ← ∅.
- 12: end for
- 13: Cache trim: Delete false KV to committed length: prompt + verified ab (all accepted blocks) + aRA.
- 14: Spawn: If some block b reaches |ab| ≥ s and active {b} < K, clone and pad qRA to length n and add as new pseudo-active block.
- 15: Promote: If |aRA| ≥ n, choose a pseudo-active b with |ab| > 0, rebuild its draft to length n, mark as verified, set RA ← b.
- 16: Stop: If all |ab| ≥ n or EOS emitted by RA, break.
- 17: end while
- 18: Finalize: Concatenate output = verified ab for all non-RA blocks, then aRA; trim KV cache C;
- 19: Return: (output,C,iters)

- A DETAILED DECODING ALGORITHM

We present the detailed algorithm for multi-block decoding and rejection sampling introduced in Section 3.2. Rejection recycling reuses high-quality consecutive tokens discarded in previous Jacobi iterations to construct candidate token sequences. Multi-block decoding jointly maintains and refines multiple blocks, allowing correct tokens in later blocks to be decoded even when earlier blocks remain unconverged, thereby further improving decoding throughput. These two techniques are orthogonal and can be seamlessly combined. As shown in Table 3, their combination yields an improvement of over 30 TPS compared to vanilla Jacobi decoding on a B200 GPU.

- B FURTHER BASELINE COMPARISONS

The main text focuses on comparisons between Jacobi Forcing Model and diffusion-based parallel decoders, as well as AR-based parallel decoders, under a controlled setup where AR variants share the same backbone (Qwen2.5-Coder-7B-Instruct). This appendix extends the comparison to (i) distilled discrete diffusion models and (ii) state-of-the-art speculative decoding baselines.

Distilled dLLM baselines. A distilled dLLM baseline is useful for mapping Jacobi Forcing Model against contemporary training techniques for discrete diffusion models. dParallel (Chen et al., 2025) performs trajectory-level consistency distillation on a discrete diffusion model to accelerate token sampling while aiming to preserve quality. We adopt the technique as the latest distilled dLLM baseline.

As shown in Table 6, on HumanEval, Jacobi Forcing Model (MR) attains a noticeably stronger speed–quality profile than dParallel: Jacobi Forcing Model (MR) achieves 29% higher accuracy and achieves more than 80% higher TPF and TPS. On GSM8K, Jacobi Forcing Model improves accuracy by 8 absolute points with about 20% higher TPF and TPS (GSM8K numbers are omitted from the table below for brevity). These gaps indicate that, relative to latest consistency-distilled dLLM of

Table 6: Additional comparison on HumanEval across AR, speculative decoding, and dLLM-based methods. For the AR baseline and all Jacobi-decoding based methods, Qwen2.5-Coder-7B-Instruct is used as the backbone. Speedup is measured in TPS relative to the AR baseline on a single B200 GPU.

Family Method Acc. ↑ TPF ↑ TPS ↑ Speedup vs. AR ↑

AR AR (greedy) 87.8 1.00 83.00 1.00× dLLM Fast-dLLM v2 63.4 1.00 83.29 1.00× dLLM SDAR 78.7 2.36 31.46 0.38× dLLM (distilled) dParallel 54.3 2.90 175.15 2.11×

AR + Spec-Dec EAGLE-3∗ 68.9∗ 6.38 246.10 2.97× AR + Spec-Dec HASS∗ 61.6∗ 5.53 280.29 3.37×

AR + Jacobi Jacobi 87.8 1.05 84.70 1.02× AR + Jacobi CLLM 87.8 2.68 207.40 2.50× AR + Jacobi Jacobi Forcing Model 83.5 4.01 301.65 3.63× AR + Jacobi Jacobi Forcing Model (MR) 83.5 4.21 327.96 3.95×

∗Here we report the strongest checkpoints released by the authors, in principle EAGLE-3 and HASS are lossless in comparison with greedy AR checkpoints if they were trained with the Qwen2.5-7B backbone.

comparable scale, Jacobi Forcing Model occupies a more favorable point in the speed–quality tradeoff space.

Speculative decoding and recent dLLM baselines. Speculative decoding (SD) forms widely used family of AR acceleration methods. To place Jacobi Forcing Model among such approaches, this appendix includes comparisons against two recent SD methods, EAGLE-3 (Li et al., 2025) and HASS (Zhang et al., 2025), which represent stronger baselines than earlier methods such as Medusa and Medusa-2.

The comparison in Table 6 also includes two recent dLLM baselines, Fast-dLLM v2 (Wu et al.,

- 2025a) and SDAR (Cheng et al., 2025), in addition to the community dLLM and D2F variants discussed in the main text. Fast-dLLM v2 improves blockwise diffusion efficiency via enhanced scheduling and caching, while SDAR introduces a synergistic diffusion–autoregressive paradigm for scalable sequence generation.

- C MAPPING NOISE SCHEDULE TO TRAINING SEQUENCE FOR PROGRESSIVE CONSISTENCY DISTILLATION

We elaborate the process of mapping the noise schedule to arrive at the training sequence in Figure 2. For each training sample, let the target model’s complete generation of length L be y. Given a training-time block size n and a noise schedule W (e.g., the linear progressive schedule in Eq. 2), we partition y into N = ⌈L/n⌉ blocks of size n. The schedule W is applied over a window of w blocks, yielding noise ratios ti defined in Eq. 7. For each block, we select the point along its Jacobi trajectory whose fraction of unconverged tokens (number of unconverged tokens/n) is closest to ti, and use that point to form the corresponding noisy block. A full illustration is shown in Figure 6.

A complete training sequence contains both noisy and clean blocks. Clean blocks are the original partitions of y, while noisy blocks are constructed as above. We interleave each noisy block with its corresponding clean block so that a single forward pass, together with the custom attention mask in Figure 4, produces teacher logits on clean blocks for the AR loss and student logits on noisy blocks for the consistency loss. Under the progressive noise schedule, the longest consecutive noisy span within any block is O(⌈tn⌉), which is much smaller than the naive O(nN) worst case where every token in every block is noisy.

Generated Sequence from Target Model

| | |
|---|---|
| | |

Block 2 t_2^T = 0

Block 1 t_1^T = 0

Block 3

t_3^T = 0 ... t_N^TBlock=N0

Prompt

- t = T - 1
- t = T - 2

### ...

### ...

### ...

### ...

t = 1

t = 0

t_1^0 = 1

t_2^0 = 1

t_1^0 = 1

t_1^0 = 1

trajectories

Noise Schedule t_1= 1/N t_2 = 2 / N t_3 = 3 / N ... t_N = (N-1) / N

Training Sequence Packing

Block 1 noisy context

Block 1 clean context

... noisyBlockcontextN cleanBlockcontextN

Prompt

2N Blocks

- Figure 6: Illustration of the progressive noise schedule and training sequence packing. For each block i over a total of Ti decoding steps, we select the trajectory step whose fraction of unconverged tokens matches the scheduled noise ratio ti to form a noisy block (dashed red line), and pair it with the corresponding clean block (dashed dark line). The packed training sequence at the bottom interleaves all noisy and clean blocks, yielding 2N blocks so that a single forward pass can compute both AR and consistency losses.

- D UNDERSTANDING TPF AND FLOPS TRADE-OFF

To estimate how many tokens can be decoded in parallel before hitting the hardware roofline, we profile generation-only latency as a function of the total number of simultaneously decoded tokens (horizontal axis in Figure 7), sweeping several block sizes ntoken seq len. On H200 and B200 (left and middle panels), the curves for ntoken seq len ∈ {16,32,64,128} are essentially flat as we increase the parallel token count up to ≈ 256 tokens, and only start to grow noticeably when we push beyond that to 512 tokens. This plateau followed by an approximately linear region is the empirical roofline: up to ∼ 256 batched tokens the GPU has spare FLOPs and KV bandwidth, so extra tokens are almost “free,” whereas beyond that point the device becomes compute- or memory-bound and latency scales roughly linearly.

On A100 (right panel of Figure 7), the plateau is shorter: generation time is nearly constant up to ∼ 128 parallel tokens, but increases steeply once we go beyond 128 and approaches linear scaling by 256 tokens. Taken together, these measurements suggest operating near the “knee” of each roofline, which corresponds to ≈ 128 parallel tokens on A100 and ≈ 256 parallel tokens on H200/B200. This motivates our final configuration: block size 64 with verification size 4 on H200 and B200 (64×4 = 256 tokens), which maximizes FLOPs utilization without hurting wall-clock performance.

These roofline measurements imply a FLOPs budget on each GPU: once the parallel token count approaches the hardware knee, additional tokens incur an almost linear increase in cost. Consequently, there is an explicit TPF–FLOPs tradeoff: configurations with larger blocks and more aggressive parallelism achieve higher TPF, but the extra FLOPs consumption can saturate the hardware and even degrade wall-clock latency.

[Figure 6]

[Figure 7]

[Figure 8]

(a) H200 (b) B200 (c) A100

- Figure 7: Generation-only latency versus total number of parallel decoded tokens across three hardware platforms (A100, H200, B200).

[Figure 9]

(a) Best-Fit Surface with Interpolation. (b) TPS Contour Map.

[Figure 10]

- Figure 8: Tokens-per-second (TPS) as a function of block size ntoken seq len and n-gram verification size for K = 2 and r = 0.85. Black dots indicate measured configurations; the surface and contours are obtained by Gaussian-like Smoothing.

- E INFERENCE CONFIGURATION SEARCH

Because of this TPF-FLOPs trade-off, choosing an inference configuration is no longer a matter of simply maximizing block size or verification depth: the configuration must respect the FLOPs budget implied by the roofline of the target GPU. Once K = 2 and r = 0.85 (initialization threshold) are fixed as training-optimal values from a separate grid search (as discussed in Section 4.3, the remaining degrees of freedom at inference are the block size ntoken seq len and the n-gram verification size, which jointly determine how much parallel draft and verify work is done per step under a given hardware constraint.

To explore this space, we perform a grid search over block sizes ntoken seq len ∈ {8,16,32,64,128,256} and n-gram verification sizes ngram ∈ {1,2,4,8,12}, measuring the achieved tokens per second for each pair on the target GPU. Since the raw grid is relatively coarse, we fit a smooth surface over the discrete measurements and use it as a surrogate for continuous hyperparameter selection. Specifically, we construct a 2D polynomial design matrix in (block size,n-gram size) of total degree up to 6, select the best degree by mean squared error, and then interpolate the fitted surface onto a dense grid using scipy.interpolate.griddata with a light Gaussian-like smoothing pass.

The results are shown in Figure 8, and the resulting surfaces reveal a clear optimum region: tokensper-second peaks at moderate block sizes and medium n-gram verification, with the global maximum near ntoken seq len ≈ 64 and ngram ≈ 4. Very small blocks or n-gram verification size underutilize the available FLOPs, while very larger choices push the system closer to the roofline and begin to degrade wall-clock latency. This analysis justifies the final choice of using block size 64 and n-gram size 4 on B200, which lies near the empirical optimum under each GPU’s FLOPs budget.

