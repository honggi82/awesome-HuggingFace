arXiv:2505.22618v3[cs.CL]3Jul2025

[Figure 1]

2025-7-4

# Fast-dLLM: Training-free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding

Chengyue Wu1,2* Hao Zhang2* Shuchen Xue4 Zhijian Liu2 Shizhe Diao2 Ligeng Zhu2 Ping Luo1 Song Han2,3 Enze Xie2

1The University of Hong Kong 2NVIDIA 3MIT 4Independent Researcher

*Equal contribution.

Abstract: Diffusion-based large language models (Diffusion LLMs) have shown promise for non-autoregressive text generation. However, the practical inference speed of open-sourced Diffusion LLMs often lags behind autoregressive models due to the lack of Key-Value (KV) Cache and quality degradation when decoding multiple tokens simultaneously. To bridge this gap, we introduce Fast-dLLM, a method that incorporates a novel block-wise approximate KV Cache mechanism tailored for bidirectional diffusion models, enabling cache reuse with negligible performance drop. Additionally, we identify the root cause of generation quality degradation in parallel decoding as the disruption of token dependencies under the conditional independence assumption. To address this, Fast-dLLM also proposes a confidence-aware parallel decoding strategy that selectively decodes tokens exceeding a confidence threshold, mitigating dependency violations and maintaining generation quality. Experimental results on LLaDA and Dream models across multiple LLM benchmarks demonstrate up to 27.6× throughput improvement with minimal accuracy loss, closing the performance gap with autoregressive models and paving the way for practical deployment of Diffusion LLMs.

Links: Github Code | Project Page

## 1. Introduction

Diffusion-based large language models (Diffusion LLMs) have recently attracted increasing attention due to their potential for parallel token generation and the advantages of bidirectional attention mechanisms. Notably, Mercury [13] runs at over 1,000 tokens per second, and Gemini Diffusion [8] by Google DeepMind has demonstrated the ability to generate over 1,400 tokens per second, highlighting the promise of significant inference acceleration.

However, current open-source Diffusion LLMs [21, 36] have yet to close such throughput gap in practice, and their actual speed often falls short of autoregressive (AR) models. This is primarily due to two issues. First, diffusion LLMs do not support key-value (KV) caching, a critical component in AR models for speeding up inference. Second, the generation quality tends to degrade when decoding multiple tokens in parallel. For example, recent findings such as those from LLaDA [21] indicate that Diffusion LLMs perform best when generating tokens one at a time and soon degrades when decoding multiple tokens simultaneously.

To bridge the performance gap with AR models that benefit from KV Cache, we present Fast-dLLM, a fast and practical diffusion-based language modeling framework. First, Fast-dLLM introduces an approximate KV Cache tailored to Diffusion LLMs. While the bidirectional nature of attention in Diffusion LLMs precludes a fully equivalent KV Cache, our approximation closely resembles an ideal cache in practice. To support KV Cache, we adopt a block-wise generation manner. Before generating a block, we compute and store KV Cache of the other blocks to reuse. After generating the block, we recompute the KV Cache of all the blocks. Visualizations confirm the high similarity with adjacent inference steps within the block, and our experiments show that this approximation preserves model performance during inference. We further propose a DualCache version that caches Keys and Values for both prefix and suffix tokens.

In parallel, Fast-dLLM investigates the degradation in output quality when generating multiple tokens simultaneously. Through theoretical analysis and empirical studies, we identify that simultaneous sampling of interdependent tokens under a conditional independence assumption disrupts critical token dependencies. To address this issue and fully exploit the parallelism potential of Diffusion LLMs, we propose a novel confidence-thresholding strategy to select which tokens can be safely decoded simultaneously. Instead of selecting the tokens with top K confidence to decode as in LLaDA, we select tokens with confidence larger than a threshold. Our theoretical justification and experimental results demonstrate that this strategy maintains generation quality while achieving up to 13.3× inference speed-up.

In summary, our contributions are threefold:

© 2025 NVIDIA. All rights reserved.

86

| | | | | |Qwen2.5-7|B|
|---|---|---|---|---|---|---|
| | | | | | | |
| | |LLa|DA+KVCach|e| | |
| |LLaDA<br><br>Dre|LLaDA+P<br><br>am|L<br><br>arallel<br><br>8|LaMA-3-8B<br><br>.1× Faster<br><br>|LLaDA+K<br><br>+Paralle|VCache l|
| | | | | | | |
| | | |5.|3× Faster| | |
| | |Dream+Par|allel Dream+|KVCache|Dream+KV<br><br>+Parallel|Cache|
| | | | | | | |
| | | | | | | |

70

Tokens per step 3.25 3.01

3.0

84

Throughput (tokens/second)

60

Throughput(tokens/second)

54.40

Numberoftokensperstep

GSM8K(5-shot)Accuracy(%)

2.5

82

50

2.0

80

40

1.5

78

30

1.00 1.00

1.0

21.20

20

76

16.50

0.5

10

74

6.70

0.0

0

72

LLaDA LLaDA+KVCacheLLaDA+ParallelLLaDA+KVCache+Parallel

70

0 10 20 30 40 50

Throughput (Tokens/sec)

(a) Throughput vs. Accuracy across methods

(b) Throughput and tokens per step across methods

GSM8K (8-shot) Gen. Length=1024 Acc=77.3

Latency per step

LLaDA 1024 steps

0.26s

Latency per sample: 266s, Throughput: 0.7 tok./s

+Parallel ~100 steps

13.3x

0.26s

26s, 9.3 tok./s

+PreﬁxCache ~140 steps

1.4x

0.14s

20s, 13.0 tok./s

2.1x

+DualCache ~140 step

12s, 19.3 tok./s

GSM8K (8-shot) Gen. Length=1024 Acc=76.0

0.09s

27.6x

(c) End-to-end speedup over vanilla LLaDA baseline.

- Figure 1 | Effectiveness of components of Fast-dLLM across different approaches. We use NVIDIA A100 GPU with a single batch size and no inference speedup frameworks.. (a) Inference throughput (tokens/sec) and GSM8K (5-shot) accuracy across various designs and models under a maximum generation length of 256. Caching mechanism and parallel decoding can significantly accelerate inference, while the combination provides up to an 8.1× increase in throughput with negligible accuracy reduction. (b) We break down the contributions of each method by showing both the number of tokens generated per step (line) and total throughput (bars). (c) With long prefilling (8-shot) and a maximum generation length of 1024, our combined approach achieves up to 27.6× end-to-end speedup compared to the vanilla LLaDA baseline.

- 1. Key-Value Cache for Block-Wise Decoding We introduce a block-wise approximate KV Cache mechanism specifically designed for bidirectional attention. Our approach reuses cached activations from previously decoded blocks by exploiting the high similarity of KV activations between adjacent steps. By caching both prefix and suffix blocks, the DualCache strategy enables substantial computational reuse.
- 2. Confidence-Aware Parallel Decoding We propose a novel confidence-aware parallel decoding method. Unlike prior approaches that select a fixed number of tokens per step, our method dynamically selects tokens whose confidence exceeds a global threshold, enabling safe and effective parallel decoding. This approach significantly accelerates inference by 13.3× while preserving output quality.
- 3. State-of-the-Art Acceleration Results We conduct comprehensive experiments on multiple open-source Diffusion LLMs (LLaDA, Dream) and four mainstream benchmarks (GSM8K, MATH, HumanEval, MBPP). Results demonstrate that our Fast-dLLM consistently deliver order-of-magnitude speedups with minimal or no degradation in accuracy, confirming the generality and practical value of our approach for real-world deployment. Fast-dLLM achieves hgiher acceleration (up to 27.6×) when generation length is longer (1024).

## 2. Preliminary

### 2.1. Masked Diffusion Model

Diffusion models for discrete data were first explored in [29, 11]. Subsequently, D3PM [2] proposed a more general framework, defining the forward noising process via a discrete state Markov chain with specific transition matrices 𝑄𝑡, and parameterized 𝑝𝜃(𝑥0|𝑥𝑡) for learning the reverse process by maximizing the Evidence Lower Bound (ELBO). CTMC [3] further extended D3PM to continuous time, formalizing it within a continuous-time Markov Chain (CTMC) framework. In a different approach, SEDD [17] parameterizes the likelihood ratio 𝑝

𝑡(𝑦)

𝑝𝑡(𝑥) for learning the reverse process, and employs Denoising Score Entropy to train this ratio.

Among the various noise processes in discrete diffusion, Masked Diffusion Models (MDMs), also termed absorbing state discrete diffusion models, have gained considerable attention. MDMs employ a forward noising process where tokens are progressively replaced by a special [MASK] token. This process is defined by the transition probability:

Cat(︁𝑥𝑖𝑡;(1 − 𝑡)𝛿𝑥𝑖

+ 𝑡𝛿[MASK])︁. (1)

∏︁𝑛

∏︁𝑛

(︀

)︀

𝑞𝑡|0 (𝑥𝑡|𝑥0) =

=

𝑥𝑖𝑡|𝑥𝑖0

𝑞𝑡|0

0

𝑖=1

𝑖=1

Here, 𝑡 ∈ [0,1] denotes the diffusion time (or masking level), controlling the interpolation between the original data 𝑥0 (at 𝑡 = 0) and a fully masked sequence (at 𝑡 = 1).

More recently, work by MDLM [28, 27, 41] and RADD [22] has shown that for MDMs, different parameterizations are equivalent. Furthermore, they demonstrated that the training objective for MDMs can be simplified or directly derived from the data likelihood. This leads to the following objective function, an Evidence Lower Bound (ELBO) on log 𝑝𝜃(𝑥):

⎡ ⎣

−log 𝑝𝜃(𝑥𝑖0|𝑥𝑡)⎤

−log 𝑝𝜃 (𝑥) ≤ ∫︁ 1

- 0
- 1

∑︁

⎦d𝑡 := ℒMDM. (2)

E𝑞

𝑡|0(𝑥𝑡|𝑥0)

𝑡

𝑖:𝑥𝑖0=[MASK]

### 2.2. Generation Process of MDMs

The analytical reverse of the forward process defined in Equation 1 is computationally inefficient for generation, as it typically involves modifying only one token per step [3, 17]. A common strategy to accelerate this is to employ a 𝜏-leaping [6] approximation for the reverse process. In the context of MDMs, this allows for an iterative generation process where multiple masked tokens can be approximately recovered in a single step from a noise level 𝑡 to an earlier level 𝑠 < 𝑡.

⎧ ⎪⎨

1, 𝑥𝑖𝑡 ̸= [MASK],𝑥𝑖𝑠 = 𝑥𝑖𝑡 𝑠 𝑡, 𝑥𝑖𝑡 = [MASK],𝑥𝑖𝑠 = [MASK] 𝑡−𝑠

𝑛∏︁−1

𝑞𝑠|𝑡 =

𝑞𝑠|𝑡(𝑥𝑖𝑠|𝑥𝑡), where 𝑞𝑠|𝑡(𝑥𝑖𝑠|𝑥𝑡) =

(3)

⎪⎩

𝑡 𝑞0|𝑡(𝑥𝑖𝑠|𝑥𝑡), 𝑥𝑖𝑡 = [MASK],𝑥𝑖𝑠 ̸= [MASK].

𝑖=0

Here, 𝑞0|𝑡(𝑥𝑖𝑠|𝑥𝑡) (when 𝑥𝑖𝑡 = [MASK]) represents a distribution over the vocabulary for predicting a non-[MASK] token, provided by the model. In scenarios involving conditional data, such as generating a response 𝑥0 to a prompt 𝑝, the MDM’s reverse process, as defined in Equation 3, requires adaptation. Specifically, the model’s predictive

distribution 𝑞0|𝑡(𝑥𝑖𝑠|𝑥𝑡) for unmasking a token 𝑥𝑖𝑠 is now also conditioned on the prompt 𝑝, as 𝑞0|𝑡(𝑥𝑖𝑠|𝑥𝑡,𝑝).

Curse of Parallel Decoding Directly reversing the forward process from Equation 1 for generation is slow, typically altering just one token per step [3, 17]. A common strategy to accelerate this is to employ a 𝜏-leaping [6] approximation for the reverse process. For MDMs, this means multiple masked tokens will be generated in parallel in a single step. However, a significant challenge arises in multiple token prediction due to the conditional independence assumption. Consider an example from [30]: The list of poker hands that consist of two English words are: _ _. The subsequent two words could be, for instance, “high card,” “two pair,” “full house,” or “straight flush.” Notably, a correlation exists between these two words. However, the multi-token prediction procedure in MDMs first generates a probability distribution for each token and then samples from these distributions independently. This independent sampling can lead to undesirable combinations, such as “high house.”

To formalize this, consider unmasking two token positions, 𝑖 and 𝑗. MDMs sample these from 𝑝(𝑥𝑖𝑠|𝑥𝑡) · 𝑝(𝑥𝑗𝑠|𝑥𝑡) due to the conditional independence assumption. However, the true joint probability requires accounting for the dependency:

Prompt Block 0 Block 1

Prompt Block 0 Block 1

𝑡 = 1

𝑡 = 1

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

Prompt token

Cached token

MASK token

Decoded token Compute cache Decode token

𝑡 = 0

𝑡 = 0

(a) Prefix KV Cache for block-wise generation. (b) DualCache: Bidirectional KV cache contains prefix and suffix Cache.

- Figure 2 | Illustration of our Key-Value Cache for Block-Wise Decoding. (a) During prefix-only caching, the KV cache is computed once for the prompt and reused across multiple decoding steps within each block. The cache is updated after completing a block to maintain consistency, with negligible overhead. (b) DualCache extends this approach by caching both prefix and masked suffix tokens, further accelerating decoding. The high similarity of KV activations across steps allows effective reuse with minimal approximation error.

𝑝(𝑥𝑖𝑠,𝑥𝑗𝑠|𝑥𝑡) = 𝑝(𝑥𝑖𝑠|𝑥𝑡) · 𝑝(𝑥𝑗𝑠|𝑥𝑡,𝑥𝑖𝑠) (or symmetrically, by conditioning 𝑖 on 𝑗). This discrepancy between the assumed independent generation and the true dependent data distribution can degrade the quality and coherence of the generated sequences. The issue is more problematic when a large number of tokens are unmasked simultaneously in a single step.

- 3. Methodology

### 3.1. Pipeline Overview

Our approach, Fast-dLLM, builds on the Masked Diffusion Model (MDM) architecture to enable efficient and highquality sequence generation. To accelerate inference, the overall pipeline incorporates two key strategies: efficient attention computation through Key-Value (KV) Cache and a parallel decoding scheme guided by prediction confidence.

Specifically, we adopt Key-Value Cache for Block-Wise Decoding, which allows reusing attention activations across steps and significantly reduces redundant computation. Within each block, we further propose Confidence-Aware Parallel Decoding, enabling selective updates of tokens based on confidence scores to improve efficiency while maintaining output quality.

By combining these strategies, Fast-dLLM significantly speeds up inference for MDMs with minimal impact on generation performance. The overall procedure is summarized in Algorithm 1.

### 3.2. Key-Value Cache for Block-Wise Decoding

- As shown in Figure 2, we adopt a block-wise decoding strategy to support the use of a Key-Value (KV) Cache. Initially, we compute and store the KV Cache for the prompt, which is reused throughout Block 0. Within each block, the same cache is reused for multiple decoding steps. After completing the decoding of a block, we update the cache for all tokens (not just the newly generated ones). This cache update can be performed jointly with the decoding step, so compared to not using caching, there is no additional computational overhead. This approach results in an approximate decoding process, due to the use of full attention in masked diffusion models [21, 36].

The effectiveness of our approximate KV Cache approach stems from the observation that KV activations exhibit high similarity across adjacent inference steps, as illustrated in Figure 3. The red boxed region in Figure 3a highlights the similarity scores within a block, which are consistently close to 1. This indicates that the differences in prefix keys and values during block decoding are negligible, allowing us to safely reuse the cache without significant loss in accuracy.

Furthermore, we implement a bidirectional version of our KV caching mechanism, named DualCache, that caches not

only the prefix tokens but also the suffix tokens, which consist entirely of masked tokens under our block-wise decoding scheme. As shown in Table 4, DualCache results in further acceleration. The red boxed region in Figure 3b further demonstrates that the differences in suffix keys and values during block decoding are negligible.

###### Prompt's Key-Value Activation Cosine Similarity Heatmap

1.00

0

| |[Figure 9]<br><br>High Cosine similarity for (i, j) diagonal neighborhood blocks<br><br>Cosine similarity is very low when (i, j) are far apart| | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

|[Figure 10]| |
|---|---|
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |

0.98

20

0.96

40

0.94

CosineSimilarity

InferenceStep

0.92

60

0.90

80

0.88

100

0.86

0.84

120

0 20 40 60 80 100 120

Inference Step

(a) Prompt block

###### Last Block's Key-Value Activation Cosine Similarity Heatmap

1.00

0

|[Figure 11]| |
|---|---|
| | |
| | |
| | |
| | |
| | |
| | |

[Figure 12]

0.98

20

0.96

CosineSimilarity

###### InferenceStep

40

0.94

High Cosine similarity for (i, j) diagonal neighborhood blocks

60

0.92

Cosine similarity is very low when (i, j) are far apart

0.90

80

0.88

0 20 40 60 80

Inference Step

(b) Last block

Figure 3 | Heatmaps of Key-Value Activation Cosine Similarity Across Inference Steps in LLaDA-Instruct. (a) Cosine similarity heatmap for the prompt block, averaged over all prompt tokens. (b) Cosine similarity heatmap for the last block, averaged over all tokens in the last block (used to represent suffix tokens, as the last block always belongs to the suffix before its own decoding). In both (a) and (b), high similarity is observed near the diagonal (𝑖 ≈ 𝑗), indicating that Key-Value activations at adjacent inference steps within a block are highly similar. The red boxed regions highlight this effect, supporting the use of an approximate block-wise KV Cache: cached activations from previous steps can be safely reused during block decoding with minimal loss in accuracy. The DualCache strategy, which additionally caches suffix tokens, further demonstrates negligible differences in activations during block decoding, enabling greater acceleration with competitive accuracy.

### 3.3. Confidence-Aware Parallel Decoding

While approaches like employing auxiliary models to explicitly capture these dependencies exist [16, 34], they typically increase the complexity of the overall pipeline. In contrast to these approaches, we propose a simple yet effective confidence-aware decoding algorithm designed to mitigate this conditional independence issue.

Concretely, at each iteration, rather than aggressively unmasking all masked tokens using their independent marginal probabilities, we compute a confidence score for each token (e.g., the maximum softmax probability). Only those with confidence exceeding a threshold are unmasked in the current step; the rest remain masked and are reconsidered in future steps. If no token’s confidence exceeds the threshold, we always unmask the token with the highest confidence to ensure progress and prevent an infinite loop. This strategy accelerates generation while reducing errors from uncertain or ambiguous predictions.

A critical question, however, is: When is it theoretically justifiable to decode tokens in parallel using independent marginals, despite the true joint distribution potentially containing dependencies? We address this with the following formal result, which characterizes the conditions under which greedy parallel (product of marginal distribution) decoding is equivalent to greedy sequential (true joint distribution) decoding in the high-confidence regime, and quantifies the divergence between the two distributions.

Prior to presenting the theorem, we will define the mathematical notation used in its statement. Let 𝑝𝜃(·|𝐸) denote the conditional probability mass function (PMF) given by an MDM condition on 𝐸 (comprising a prompt 𝑝0 and previously generated tokens). Suppose the model is to predict 𝑛 tokens for positions 𝑖1,...,𝑖𝑛 not in 𝐸. Let 𝑋 = (𝑋𝑖

) be the vector of 𝑛 tokens, where each 𝑋𝑖

,...,𝑋𝑖

1

𝑛

𝑛|𝐸) be the joint conditional PMF according to the model. Let 𝑝𝑗(𝑋𝑖

takes values in vocabulary 𝒱. Let 𝑝(𝑋|𝐸) ≡ 𝑝𝜃(𝑋𝑖

,...,𝑋𝑖

1

𝑗

𝑗|𝐸) be the marginal conditional PMF for position 𝑖𝑗. Parallel decoding generates tokens using the product of marginals: 𝑞(𝑋|𝐸) =

𝑗|𝐸) ≡ 𝑝𝜃(𝑋𝑖

∏︀𝑛 𝑗=1 𝑝𝑗(𝑋𝑖

𝑗|𝐸). The proof of Theorem 1 and relevant discussions are in Appendix A.

Theorem 1 (Parallel Decoding under High Confidence). Suppose there exists a specific sequence of tokens 𝑥* = (𝑥𝑖

𝑗|𝐸) > 1 − 𝜖 for some small 𝜖 > 0. Then, the following results hold:

) such that for each 𝑗 ∈ {1,...,𝑛}, the model has high confidence in 𝑥𝑖

: 𝑝𝑗(𝑋𝑖

= 𝑥𝑖

,...,𝑥𝑖

1

𝑛

𝑗

𝑗

- 1. Equivalence for Greedy Decoding: If (𝑛 + 1)𝜖 ≤ 1 (i.e., 𝜖 ≤ 𝑛+11 ), then

argmax

𝑧

𝑝(𝑧|𝐸) = argmax

𝑧

𝑞(𝑧|𝐸) = 𝑥*. (4)

This means that greedy parallel decoding (selecting argmax 𝑞) yields the same result as greedy sequential decoding (selecting argmax 𝑝).

This bound is tight: if 𝜖 > 𝑛+11 , there exist distributions 𝑝(𝑋|𝐸) satisfying the high-confidence marginal assumption for which argmax𝑧 𝑝(𝑧|𝐸) ̸= argmax𝑧 𝑞(𝑧|𝐸).

- 2. Distance and Divergence Bounds: Let 𝑝(·|𝐸) and 𝑞(·|𝐸) be denoted as 𝑝 and 𝑞 for brevity.

𝐿𝑝 Distance (𝑝 ≥ 1): For 𝑛 > 1, 𝐷𝑝 (𝑝,𝑞) < ((𝑛 − 1)𝑝 + 2𝑛)1/𝑝𝜖. Specifically, for Total Variation Distance (𝐷𝑇𝑉 (𝑝,𝑞) = 12𝐷1 (𝑝,𝑞)): 𝐷𝑇𝑉 (𝑝,𝑞) < 3𝑛2−1𝜖. Forward KL Divergence: For 𝑛 > 1, 𝐷KL (𝑝‖𝑞) < (𝑛 − 1)(𝐻𝑏(𝜖) + 𝜖ln(|𝒱| − 1)), where 𝐻𝑏(𝜖) = −𝜖ln 𝜖 − (1 − 𝜖)ln(1 − 𝜖) is the binary entropy function, and |𝒱| is the size of the vocabulary.

Building on this theorem, we propose a practical factor-based parallel decoding strategy as an extension of the threshold strategy that adaptively selects how many tokens to decode in parallel based on the confidence levels. Concretely, given the model’s marginal confidence estimates for 𝑛 tokens in a block, we sort these confidences and select the largest 𝑛 such that (𝑛 + 1)(1 − 𝑐(𝑛)) < 𝑓, where 𝑓 is a fixed decoding factor hyperparameter and 𝑐(𝑛) is the 𝑛-th highest confidence.

- At each step, the top-𝑛 tokens are decoded in parallel. This formulation mirrors the bound in Theorem 1 and ensures that decoding only proceeds when the marginal confidence is sufficiently high to approximate the joint decoding reliably. In contrast to the static threshold-based strategy, factor-based decoding dynamically controls the degree of parallelism in a theoretically grounded manner.

Algorithm 1 Block-wise Confidence-aware Parallel Decoding with (Dual) KV Cache Require: 𝑝𝜃, prompt 𝑝0, answer length 𝐿, blocks 𝐾, block size 𝐵, steps per block 𝑇, threshold 𝜏, use_DualCache,

strategy ∈ {threshold, factor}, factor 𝑓

- 1: 𝑥 ← [𝑝0;[MASK],...,[MASK]]
- 2: Initialize KV Cache (single or dual) for 𝑥 (fuse with decoding). // KV Cache Init
- 3: for 𝑘 = 1 to 𝐾 do
- 4: 𝑠 ← |𝑝0| + (𝑘 − 1)𝐵, 𝑒 ← |𝑝0| + 𝑘𝐵
- 5: for 𝑡 = 1 to 𝑇 do
- 6: Use cache, run 𝑝𝜃 on 𝑥[𝑠,𝑒) if use_DualCache else 𝑥[𝑠,:) // Cache Reuse
- 7: For masked 𝑥𝑖, compute confidence 𝑐𝑖 = max𝑥 𝑝𝜃(𝑥𝑖|·) // Confidence scoring
- 8: if strategy == threshold then
- 9: Unmask all 𝑖 in [𝑠,𝑒) with 𝑐𝑖 ≥ 𝜏, always unmask max 𝑐𝑖
- 10: else if strategy == factor then
- 11: Sort 𝑐𝑖 in descending order as (𝑐(1),𝑐(2),...,𝑐(𝑚))
- 12: Find largest 𝑛 such that (𝑛 + 1)(1 − 𝑐(𝑛)) < 𝑓
- 13: Unmask top-𝑛 tokens, always unmask the max 𝑐𝑖
- 14: end if
- 15: if all 𝑥[𝑠,𝑒) unmasked then
- 16: break
- 17: end if
- 18: end for
- 19: Update KV cache: if use_DualCache: prefix & suffix; else: prefix. // Cache Update
- 20: end for
- 21: return 𝑥

- Table 1 | Comprehensive benchmark results on the LLaDA-Instruct suite. Each cell presents the accuracy and the decoding throughput in tokens per second with relative speedup to the LLaDA baseline (bottom row, blue: tokens per second/orange: relative speedup). The highest throughput and speedup for each configuration are highlighted.

Benchmark Gen Length LLaDA +Cache +Parallel +Cache+Parallel (Fast-dLLM)

256 79.3 79.5 79.2 78.5

6.7 (1×) 21.2 (3.2×) 16.5 (2.5×) 54.4 (8.1×) 512 77.5 77.0 77.6 77.2

GSM8K (5-shot)

3.2 (1×) 10.4 (3.3×) 18.6 (5.8×) 35.3 (11.0×)

256 33.5 33.3 33.4 33.2

9.1 (1×) 23.7 (2.6×) 24.8 (2.7×) 51.7 (5.7×) 512 37.2 36.2 36.8 36.0

MATH (4-shot)

8.0 (1×) 19.7 (2.5×) 23.8 (3.0×) 47.1 (5.9×)

256 41.5 42.7 43.9 43.3

30.5 (1×) 40.7 (1.3×) 101.5 (3.3×) 114.1 (3.7×) 512 43.9 45.7 43.3 44.5

HumanEval (0-shot)

18.4 (1×) 29.3 (1.6×) 57.1 (3.1×) 73.7 (4.0×)

256 29.4 29.6 28.4 28.2

6.0 (1×) 17.0 (2.8×) 24.8 (4.1×) 44.8 (7.5×) 512 14.8 13.4 15.0 13.8

MBPP (3-shot)

4.3 (1×) 10.1 (2.3×) 22.3 (5.1×) 39.5 (9.2×)

## 4. Experiments

### 4.1. Experimental Setup

All experiments are conducted on an NVIDIA A100 80GB GPU. The proposed approach, Fast-dLLM, comprises two components: a Key-Value Cache mechanism and a Confidence-Aware Parallel Decoding strategy. The KV Cache component introduces a hyperparameter, the cache block size, varied between 4 and 32. The parallel decoding strategy uses a confidence threshold hyperparameter, explored in the range of 0.5 to 1.0. Unless otherwise specified, we use PrefixCache with block size of 32 and the threshold to 0.9.

We evaluate Fast-dLLM on two recent diffusion-based language models: LLaDA [21], LLaDA-1.5 [44] and Dream [36]. Benchmarks include four widely-used datasets—GSM8K, MATH, HumanEval, and MBPP, to assess performance across diverse reasoning and code generation tasks. We also test under varying generation lengths to evaluate scalability and robustness.

In addition, we extend our evaluation to LLaDA-V [38], a multimodal variant of LLaDA tailored for vision-language reasoning tasks. For this, we use two challenging multimodal benchmarks: MathVista and MathVerse, which require solving math problems grounded in complex visual scenes.

Inference throughput is measured as the average number of output tokens generated per second, calculated over the full sequence until the end-of-sequence (<eos>) token is reached. This metric reflects true end-to-end decoding speed. All evaluations are conducted using the standardized lm-eval library to ensure consistency and reproducibility.

Selected

80

22

| |
|---|

| |
|---|

No cache

| |
|---|

20

GSM8K(5-shot)Accuracy

75

Throughput(tokens/s)

18

| |
|---|

70

16

3.3x Speedup

| |
|---|

14

65

12

60

10

8

55

No cache

6

4 8 16 32 64 128 256

Cache Block Size

Figure 4 | Impact of Cache Block Size on Accuracy and Throughput. The orange line illustrates the effect of varying cache block size on throughput, while the blue line depicts accuracy.

### 4.2. Main Results: Performance and Speed

We report decoding performance and efficiency gains for Fast-dLLM on both the LLaDA-Instruct and Dream-Base models across the four benchmarks in Tables 1 and 2.

- Table 2 | Comprehensive benchmark results on Dream-Base variants over four tasks with different generation lengths (256 and 512). Each cell shows accuracy (top row) and decoding throughput in tokens per second with relative speedup to Dream-Base baseline (bottom row, blue: tokens per second/orange: relative speedup). Numbers in yellow indicate the highest throughput and speedup per configuration.

Benchmark Gen Length Dream +Cache +Parallel +Cache+Parallel (Fast-dLLM)

256 75.0 74.3 74.2 74.8

9.1 (1×) 32.5 (3.6×) 14.2 (1.6×) 48.2 (5.3×) 512 76.0 74.3 73.4 74.0

GSM8K (5-shot)

7.7 (1×) 25.6 (3.3×) 14.6 (1.9×) 42.9 (5.6×)

256 38.4 36.8 37.9 37.6

11.4 (1×) 34.3 (3.0×) 27.3 (2.4×) 66.8 (5.9×) 512 39.8 38.0 39.5 39.3

MATH (4-shot)

9.6 (1×) 26.8 (2.8×) 31.6 (3.2×) 63.3 (6.5×)

256 49.4 53.7 49.4 54.3

23.3 (1×) 35.2 (1.5×) 45.6 (2.0×) 62.0 (2.8×) 512 54.3 54.9 51.8 54.3

HumanEval (0-shot)

16.3 (1×) 27.8 (1.7×) 29.8 (1.8×) 52.8 (3.2×)

256 56.6 53.2 53.8 56.4

11.2 (1×) 34.5 (3.1×) 31.8 (2.8×) 76.0 (6.8×) 512 55.6 53.8 55.4 55.2

MBPP (3-shot)

9.4 (1×) 26.7 (2.8×) 37.6 (4.0×) 73.6 (7.8×)

Overall, introducing the KV Cache mechanism yields significant speed improvements for all tasks and sequence lengths, typically achieving a 2× to 3.6× speedup compared to the vanilla backbone. When the parallel decoding strategy is applied individually, we see additional acceleration, often pushing speedups to 4×–6× for the evaluated settings, particularly as the generation length increases.

When both techniques are combined, the improvements become even more pronounced. On LLaDA, for example, combined KV Cache and parallel decoding methods boost throughput by up to 11× (GSM8K, length 512) and 9.2× (MBPP, length 512) over the standard baseline. Similarly, on Dream-Base, the largest throughput gains are observed on MBPP (7.8× at length 512) and GSM8K (5.6× at length 512). These results indicate that not only are our methods effective individually, but they are also highly complementary, resulting in the combined acceleration.

300

|Selected<br><br>6.17<br><br>5.12 4.24<br><br>3.25 1.00| | | | | | |
|---|---|---|---|---|---|---|
|7.01| | | | | | |
|Ours<br><br>2 tokens per step 4 tokens per step 8 tokens per step<br><br>| | | | | | |
| | | | | | | |
| | | | | | | |

|4.24<br><br>3.25| | | | |
|---|---|---|---|---|
|Selected<br><br>7.01<br><br>6.17<br><br>5.12<br><br>2<br><br>4<br><br>8<br><br>Ours<br><br>Fixed-Step Baseline (2/4/8 tokens)<br><br>Non-Parallel Baseline (1 token/step)| | | | |
| | | | | |

Ours

80

80

1.00

###### GSM8K(5-shot)Accuracy

GSM8K(5-shot)Accuracy

2 tokens per step 4 tokens per step 8 tokens per step

250

| |
|---|

70

70

###### InferenceSteps

200

60

60

150

50

50

100

3.25

Selected

4.24

40

40

| |
|---|

5.12

7.01 6.17

50

| |
|---|

| |
|---|

30

30

0

0.5 0.6 0.7 0.8 0.9 1.0

0.5 0.6 0.7 0.8 0.9 1.0

0 2 4 6 8

Threshold

Threshold

Average #Tokens per Step

(a) (b) (c)

Figure 5 | (a) The red line shows the GSM8K (5-shot) accuracy across different confidence thresholds. Numbers along the red line indicate the average number of tokens decoded at each step. The three dashed lines represent the accuracy of the baseline method when selecting the top 2, 4, or 8 tokens per step. (b) The number of inference steps required under varying confidence thresholds. (c) A comparison between our method and the baseline on GSM8K (5-shot) accuracy, plotted against the average number of tokens per step. Our method consistently outperforms the baseline.

Importantly, these efficiency gains are achieved with negligible impact on accuracy. Across all benchmarks and settings, the accuracy of our accelerated methods remains within 1–2 points of the backbone, and in several cases, accuracy is even slightly improved. This demonstrates that the speedup comes at almost no cost to task performance, ensuring reliability for practical deployment. We also observe that longer sequences, which are common in few-shot and code generation scenarios, benefit proportionally more from our caching and parallelization techniques due to greater opportunities for cache reuse and batch computation. We also evaluate an advanced version, LLaDA-1.5, which achieves consistently stronger accuracy and comparable or higher throughput across benchmarks (Table 12).

- Table 3 | Performance and Speedup Comparison of LLaDA-V on MathVista and MathVerse. Each benchmark includes results from Full Steps, Half Steps, and Fast-dLLM. Fast-dLLM significantly improves throughput (highlighted), with minimal accuracy loss.

Metric

MathVista MathVerse

| |Full Steps Half Steps Fast-dLLM<br><br>|Full Steps Half Steps Fast-dLLM|
|---|---|---|
|Accuracy (%) Throughput (Speedup)|59.2 59.7 56.6 2.84 (1×) 5.56 (1.96×) 28.2 (9.9×)<br><br>|28.5 28.3 28.6 2.75 (1×) 5.17 (1.88×) 23.3 (8.5×)<br><br>|

In addition to text-only models, we evaluate Fast-dLLM on the multimodal LLaDA-V using the MathVista and MathVerse datasets, which require complex vision-language reasoning. As shown in Table 9, LLaDA-V shows a strong sensitivity to block size, with accuracy dropping by over 8% when reducing from 96 to 8 on MathVista. To address this, we retain a full block length and apply refresh-based updates instead of small-block caching. This yields up to 9.9× speedup with minimal accuracy degradation (Table 3). On MathVerse, accuracy is even slightly improved under Fast-dLLM, demonstrating the broad applicability of our method to multimodal reasoning tasks.

Furthermore, the improvements generalize across model architectures (LLaDA and Dream), task types (math reasoning, program synthesis), and modalities (text and vision), confirming that Fast-dLLM is a practical and broadly applicable framework for accelerating masked diffusion-based language models.

4.3. Ablations and Analysis

- Table 4 | Performance and Speedup Comparison on LLaDA Between 5-Shot and 8-Shot Settings at Generation Length 1024. This table compares the accuracy and throughput speedups of different decoding strategies under 5-shot and 8-shot configurations using a generation length of 1024. The results demonstrate how increased prefill length enhances the effectiveness of caching strategies, particularly for DualCache.

Table 5 | Impact of Generation Length on Accuracy and Speedup Under 8-Shot for LLaDA. This table illustrates the effect of varying generation lengths (256, 512, and 1024) on decoding performance and efficiency for different caching strategies under the 8-shot setting. Longer generation lengths lead to higher throughput gains, especially for DualCache, validating the scalability of our approach.

Parallel Decoding

Setting. LLaDA

| | |No Cache PrefixCache DualCache<br><br>|
|---|---|---|
|5-shot 8-shot|77.0 1.1 (1×) 77.3 0.7 (1×)|77.4 75.2 74.7<br><br>11.7 (10.6×) 14.4 (13.1×) 21.6 (19.6×)<br><br>78.0 75.7 76.0<br><br><br>9.3 (13.3×) 13.0 (18.6×) 19.3 (27.6×)<br><br>|

Parallel Decoding

Len. LLaDA

| | |No Cache PrefixCache DualCache<br><br>|
|---|---|---|
|256 512 1024|77.6 4.9 (1×) 78.9 2.3 (1×) 77.3 0.7 (1×)|77.9 77.3 76.9<br><br>16.4 (3.3×) 49.2 (10.0×) 46.3 (9.4×)<br><br>78.9 74.8 75.4<br><br><br>14.0 (6.1×) 32.0 (13.9×) 36.4 (15.8×) 78.0 75.7 76.0 9.3 (13.3×) 13.0 (18.6×) 19.3 (27.6×)<br><br>|

We conduct extensive ablation studies to understand how different components of Fast-dLLM contribute to performance, focusing on factors such as prefill length, generation length, cache mechanism variants, cache block size, and confidence thresholds.

Influence of Prefill and Generation Length on Acceleration Table 4 and Table 5 indicate that both prefill length (n-shot) and generation length markedly impact overall speedup. Specifically, as the prefill length increases from 5-shot to 8-shot, the speedup obtained by both versions of KV Cache rises significantly (e.g., speedup for DualCache increases from 19.6× in 5-shot to 27.6× in 8-shot for generation length 1024). Similarly, extending the generation length amplifies the potential for cache reuse, leading to higher speedup. Notably, for 8-shot, speedup with DualCache grows from 9.4× (gen len 256) up to 27.6× (gen len 1024). This aligns with the theoretical expectation that amortizing computation over longer sequences yields more pronounced efficiency gains.

Comparison of prefix KV Cache vs. DualCache We further compare our prefix KV Cache and DualCache versions in multiple settings. As shown in Table 5, DualCache generally achieves higher speedup than the prefix KV Cache, especially for longer generation lengths. For gen len 512 and 1024, DualCache demonstrates up to 27.6× speedup,

outperforming the prefix KV Cache’s 18.6× in the same scenario. Importantly, DualCache maintains competitive accuracy, with only minor trade-offs relative to the cache-only variant. This highlights DualCache’s effectiveness in exploiting parallelism and cache locality for both efficiency and accuracy.

Effect of Cache Block Size Figure 4 analyzes the influence of the cache block size hyperparameter. We observe that smaller block sizes tend to maximize accuracy but incur overhead due to frequent cache updates. In contrast, larger block sizes may diminish accuracy owing to increased context mismatch. Block size of 32 achieves the best trade-off, substantially improving throughput while largely preserving accuracy. This hyperparameter thus offers a practical knob for balancing latency and precision in real deployments.

Dynamic Threshold vs. Fixed Token-per-Step Strategies We evaluate our Confidence-Aware Parallel Decoding method against fixed token-per-step baselines on GSM8K (Figure 5). Our adaptive strategy consistently outperforms fixed baselines across key metrics: it delivers higher accuracy at comparable or reduced number of function evaluations (NFE) and generates more tokens per step on average while closely tracking accuracy. In the rightmost panel, the dynamic method approaches or exceeds the accuracy of the 1-token (non-parallel) baseline, but with much greater throughput. The result demonstrates the effectiveness of Confidence-Aware Parallel Decoding, offering practical advantages.

Factor Decoding vs. Fixed Token-per-Step Strategies We further compare our factor-based parallel decoding approach with fixed token-per-step baselines on GSM8K (Figure 8) and with the threshold-based strategy (Table 11). Across a range of factor values, our method consistently achieves competitive or higher accuracy with fewer inference steps. As the factor increases, the number of tokens decoded per step grows steadily, reducing iteration count while maintaining performance. Compared to the threshold strategy, factor decoding achieves similar accuracy but significantly higher throughput by adaptively controlling decoding granularity. We also analyze parallel token counts across decoding step at Appendix C.4.

Decoding Efficiency Analysis and Limitations As discussed in Section C.5, PrefixCache significantly accelerates diffusion-based LLMs like LLaDA with up to 5× throughput improvement in compute-bound scenarios compared to LLaDA. At smaller batch sizes, PrefixCache achieves throughput comparable to or even exceeding that of autoregressive models like LLaMA. However, as batch sizes grow, PrefixCache struggles to match LLaMA, which transitions from memory-bound to compute-bound performance. This reflects a general challenge for diffusion-based LLMs, which tend to incur higher computational overhead due to full attention operations during decoding.

## 5. Related Work

### 5.1. Diffusion LLM

Diffusion models have emerged as a transformative paradigm in generative modeling, initially achieving remarkable success in continuous domains such as image [25, 19, 23, 26] and audio synthesis [35, 12] before expanding into natural language processing. Recent advancements in discrete diffusion models [2, 20, 21, 11, 3, 10, 18, 24, 31, 14, 42, 4, 37, 27, 28, 41, 5, 40, 39] have reshaped the landscape of text generation, offering a viable alternative to autoregressive (AR) paradigms in large language models (LLMs). These models address the inherent challenges of discrete data by redefining noise injection and denoising processes through innovative mathematical formulations.

Theoretical Foundations of Discrete Diffusion Diffusion models for discrete data were first explored in [29, 11]. Subsequently, D3PM [2] provided a more general framework. This framework models the forward noising process as a discrete state Markov chain using specific transition matrices. For the reverse process, D3PM learns a parameterized model of the conditional probability of the original data given a noised version by maximizing the Evidence Lower Bound (ELBO). CTMC [3] further extended D3PM to a continuous-time setting, formalizing it as a continuous-time Markov Chain (CTMC). In a distinct approach, SEDD [17] learns the reverse process by parameterizing the ratio of marginal likelihoods for different data instances at a given noising timestep. This ratio model is then trained using a Denoising Score Entropy objective. More recently, research on Masked Diffusion Models (MDMs) by MDLM [28, 27, 41] and RADD [22] has introduced significant clarifications. These studies have demonstrated that different parameterizations of MDMs can be equivalent.

Integration with Pre-trained Language Models A critical breakthrough involves combining discrete diffusion with existing LLM architectures. Diffusion-NAT [43] unifies the denoising process of discrete diffusion with BART’s [15] non-autoregressive decoding, enabling iterative refinement of masked tokens. By aligning BART’s inference with diffusion steps, this approach leverages pre-trained knowledge while maintaining generation speed 20× faster than comparable AR transformers. Similarly, the LLaDA [21] and DiffuLLaMA [7] framework scales diffusion to 7B parameters using masked denoising, while LLaDA and Dream [36] demonstrating competitive performance with autoregressive baselines like LLaMA3 [9] through recursive token prediction across diffusion timesteps.

### 5.2. LLM Acceleration

Key-Value Cache. Key-Value (KV) Cache is a fundamental optimization technique in modern large language model (LLM) inference with Transformer architecture [32]. It enables efficient autoregressive text generation by storing and reusing previously computed attention states. However, it is non-trival to apply KV Cache in diffusion langauge models such as LLaDA due to full attention. Block diffusion [1] overcomes key limitation of previous diffusion langauge models by generating block-by-block so that key and values of previously decoded blocks can be stored and reused.

Non-Autoregressive Generation Non-autoregressive (NAR) generation marks a fundamental shift from sequential token generation by enabling the simultaneous generation of multiple tokens, significantly accelerating inference [33]. Initially introduced for neural machine translation, NAR methods have since been extended to a variety of tasks, including grammatical error correction, text summarization, dialogue systems, and automatic speech recognition. Although NAR generation offers substantial speed advantages over autoregressive approaches, it often sacrifices generation quality. Diffusion LLMs represent a recent paradigm for non-autoregressive text generation; however, prior work [21] has struggled to realize the expected acceleration due to a notable drop in output quality.

## 6. Conclusion

In this work, we tackle key limitations in the inference efficiency of Diffusion-based Large Language Models (Diffusion LLMs), which have historically lacked support for KV Cache and exhibited performance degradation during parallel decoding. To bridge the gap with autoregressive models, we propose Fast-dLLM, a diffusion-based framework that introduces an approximate KV Cache mechanism tailored to the bidirectional attention characteristics of Diffusion LLMs, enabled by a block-wise generation scheme. Furthermore, we identify that the main obstacle to effective parallel decoding is the disruption of token dependencies arising from the conditional independence assumption. To address this, Fast-dLLM employs a Confidence-Aware Parallel Decoding strategy that facilitates safe and efficient multi-token generation. Extensive experiments across multiple benchmarks and model baselines (LLaDA and Dream) show that Fast-dLLM achieves up to a 27.6× speedup with minimal loss in accuracy. These findings offer a practical solution for deploying Diffusion LLMs as competitive alternatives to autoregressive models in real-world applications.

- A. Proof In this section, we will give the comprehensive proof and discussion of Theorem 1.

Proof. Step 1: Show that 𝑥* is the unique maximizer of 𝑞(𝑥). Let 𝑝*𝑗 = 𝑝𝑗(𝑋𝑖

𝑗|𝐸). Thus, 𝜖′𝑗 < 𝜖. The product-of-marginals probability mass function (PMF) is

= 𝑥𝑖

𝑗|𝐸). We are given 𝑝*𝑗 > 1 − 𝜖. Let 𝜖′𝑗 = 1 − 𝑝*𝑗 = 𝑝𝑗(𝑋𝑖

𝑗 ̸= 𝑥𝑖

𝑗

∏︁𝑛

𝑞(𝑧|𝐸) =

𝑝𝑗(𝑋𝑖

= 𝑧𝑗|𝐸).

𝑗

𝑗=1

To maximize 𝑞(𝑧|𝐸), we must maximize each term 𝑝𝑗(𝑋𝑖

= 𝑧𝑗|𝐸) independently. The condition (𝑛+1)𝜖 ≤ 1 implies 𝜖 ≤ 1/(𝑛 + 1). Since 𝑛 ≥ 1, it follows that 1/(𝑛 + 1) ≤ 1/2. So, 𝜖 ≤ 1/2. Therefore, for the chosen 𝑥𝑖

𝑗

: 𝑝*𝑗 = 𝑝𝑗(𝑋𝑖

𝑗

𝑗|𝐸) > 1 − 𝜖 ≥ 1 − 1/2 = 1/2. This means 𝑥𝑖

= 𝑥𝑖

𝑗

is the unique maximizer for 𝑝𝑗(·|𝐸). So, argmax

𝑗

𝑞(𝑧|𝐸) = (𝑥𝑖

) = 𝑥*.

,...,𝑥𝑖

1

𝑛

𝑧

- Step 2: Show that 𝑥* is the unique maximizer of 𝑝(𝑥). We want to show 𝑝(𝑥*|𝐸) > 𝑝(𝑧|𝐸) for all 𝑧 ̸= 𝑥*. Using the Bonferroni inequality:

𝑝(𝑥*|𝐸) = 𝑝(∩𝑛𝑗=1{𝑋𝑖

𝑗

= 𝑥𝑖

𝑗}|𝐸) ≥ 1 −

∑︁𝑛

𝑗=1

𝑝(𝑋𝑖

𝑗 ̸= 𝑥𝑖

𝑗|𝐸) = 1 −

∑︁𝑛

𝑗=1

𝜖′𝑗.

Since 𝜖′𝑗 < 𝜖 for all 𝑗, we have

∑︀𝑛 𝑗=1 𝜖′𝑗 < 𝑛𝜖. So,

𝑝(𝑥*|𝐸) > 1 − 𝑛𝜖. Now consider any 𝑧 = (𝑧1,...,𝑧𝑛) such that 𝑧 ̸= 𝑥*. This means there is at least one index 𝑘 such that 𝑧𝑘 ̸= 𝑥𝑖

𝑘

. The event {𝑋 = 𝑧} is a sub-event of {𝑋𝑖

𝑘

= 𝑧𝑘}. So, 𝑝(𝑧|𝐸) ≤ 𝑝𝑘(𝑋𝑖

𝑘

= 𝑧𝑘|𝐸). Since 𝑧𝑘 ̸= 𝑥𝑖

𝑘

,

𝑝𝑘(𝑋𝑖

𝑘

= 𝑧𝑘|𝐸) ≤ 𝑝𝑘(𝑋𝑖

𝑘 ̸= 𝑥𝑖

𝑘|𝐸) = 𝜖′𝑘 < 𝜖. Thus,

𝑝(𝑧|𝐸) < 𝜖. For 𝑝(𝑥*|𝐸) > 𝑝(𝑧|𝐸) to hold, it is sufficient that

1 − 𝑛𝜖 ≥ 𝜖,

which simplifies to 1 ≥ (𝑛 + 1)𝜖, or 𝜖 ≤ 𝑛+11 . The theorem assumes (𝑛 + 1)𝜖 < 1, which is exactly this condition. The strict inequalities 𝑝(𝑥*|𝐸) ≥ 1 − ∑︀

𝜖′𝑗 > 1 − 𝑛𝜖 and 𝑝(𝑧|𝐸) ≤ 𝜖′𝑘 < 𝜖 ensure that 𝑝(𝑥*|𝐸) > 𝑝(𝑧|𝐸). Thus, argmax

𝑧

𝑝(𝑧|𝐸) = 𝑥*.

Combined with the argmax of 𝑞, this proves the main statement of Part 1: argmax

𝑧

𝑝(𝑧|𝐸) = argmax

𝑧

𝑞(𝑧|𝐸) = 𝑥*.

- Step 3: Tightness of the bound 𝑛+11 .

The bound 𝜖 ≤ 𝑛+11 is tight. This means if 𝜖 > 𝑛+11 , one can construct a scenario where the marginal conditions 𝑝𝑗(𝑋𝑖

𝑗|𝐸) > 1 − 𝜖 hold, but argmax𝑧 𝑝(𝑧|𝐸) ̸= 𝑥* (which is argmax𝑧 𝑞(𝑧|𝐸) as long as 𝜖 ≤ 1/2). Consider a vocabulary 𝒱 = {0,1} and let 𝑥𝑖

= 𝑥𝑖

𝑗

= 0 for all 𝑗, so 𝑥* = (0,...,0). For each 𝑗 ∈ {1,...,𝑛}, let e𝑗 be the vector with 1 at position 𝑗 and 0 elsewhere. Let 𝜂 = 𝑛+11 (𝜖 − 𝑛+11 ) > 0. Set 𝑝(e𝑗|𝐸) = 𝑛+11 + 𝑛1𝜂, ∀1 ≤ 𝑗 ≤ 𝑛 and 𝑝(𝑥*|𝐸) = 𝑛+11 − 𝜂 , then 𝑥* ∈/ argmax𝑧 𝑝(𝑧|𝐸). The marginal probabilities are:

𝑗

1 𝑛 + 1

1 𝑛

= 1|𝐸) = 𝑝(e𝑗|𝐸) =

𝑝𝑗(𝑋𝑖

+

𝜂, ∀1 ≤ 𝑗 ≤ 𝑛.

𝑗

1 𝑛

𝑛 𝑛 + 1 −

𝑝𝑗(𝑋𝑖

= 0|𝐸) = 1 − 𝑝𝑗(𝑋𝑖

= 1|𝐸) = 1 − 𝜖𝑐 =

𝜂 > 1 − 𝜖,

𝑗

𝑗

because

1 𝑛 + 1 So, the marginal condition 𝑝𝑗(𝑋𝑖

1 𝑛

1 𝑛(𝑛 + 1)

1 𝑛 + 1

𝜂 =

(𝜖 −

) < 𝜖 −

= 0) holds. As shown, argmax𝑧 𝑝(𝑧|𝐸) can be made different from 𝑥*. Thus, if 𝜖 > 𝑛+11 , the argmax of 𝑝 and 𝑞 may not be the same.

= 𝑥𝑖

𝑗|𝐸) > 1 − 𝜖 (with 𝑥𝑖

𝑗

𝑗

= 𝑥𝑖

### Step 4: Bound the 𝐿𝑝 distance. Let 𝐴𝑗 be the event {𝑋𝑖

𝑗}. 𝐷𝑝 (𝑝,𝑞)𝑝 = |𝑝(𝑥*|𝐸) − 𝑞(𝑥*|𝐸)|𝑝 +

𝑗

∑︁

|𝑝(𝑧|𝐸) − 𝑞(𝑧|𝐸)|𝑝.

𝑧̸=𝑥*

The term |𝑝(∩𝑛𝑗=1𝐴𝑗|𝐸) − ∏︀𝑛

𝑗=1 𝑝(𝐴𝑗|𝐸)| (using 𝑝(𝐴𝑗|𝐸) for 𝑝𝑗(𝑋𝑖

= 𝑥𝑖

𝑗|𝐸)) can be bounded. Since

𝑗

∑︁𝑛

1 −

𝜖′𝑗 ≤ 𝑝(∩𝑛𝑗=1𝐴𝑗|𝐸) ≤ min

𝑝(𝐴𝑗|𝐸) = 1 − max

𝜖′𝑗,

1≤𝑗≤𝑛

1≤𝑗≤𝑛

𝑗=1

∑︁𝑛

∏︁𝑛

∏︁𝑛

(1 − 𝜖′𝑗) =

𝑝(𝐴𝑗|𝐸) ≤ 1 − max

1 −

𝜖′𝑗 ≤

𝜖′𝑗.

1≤𝑗≤𝑛

𝑗=1

𝑗=1

𝑗=1

Thus,

|𝑝(𝑥*|𝐸) − 𝑞(𝑥*|𝐸)| < (𝑛 − 1)𝜖. For 𝑧 ̸= 𝑥*: 𝑝(𝑧|𝐸) < 𝜖 and 𝑞(𝑧|𝐸) < 𝜖. So,

|𝑝(𝑧|𝐸) − 𝑞(𝑧|𝐸)| < 𝜖. The sum

∑︀

𝑧̸=𝑥* |𝑝(𝑧|𝐸) − 𝑞(𝑧|𝐸)| can be bounded:

∑︁

|𝑝(𝑧|𝐸) − 𝑞(𝑧|𝐸)| ≤ ∑︁

(𝑝(𝑧|𝐸) + 𝑞(𝑧|𝐸)) = 𝑝(𝑋 ̸= 𝑥*|𝐸) + 𝑞(𝑋 ̸= 𝑥*|𝐸).

𝑧̸=𝑥*

𝑧̸=𝑥*

∑︁𝑛

𝑝(𝑋 ̸= 𝑥*|𝐸) = 1 − 𝑝(𝑥*|𝐸) < 1 − (1 −

𝜖′𝑗) =

𝑗=1

∏︁𝑛

𝑞(𝑋 ̸= 𝑥*|𝐸) = 1 − 𝑞(𝑥*|𝐸) < 1 −

(1 − 𝜖′𝑗) ≤

𝑗=1

So, ∑︁

|𝑝(𝑧|𝐸) − 𝑞(𝑧|𝐸)| < 2𝑛𝜖.

𝑧̸=𝑥*

Then,

∑︁

∑︁

|𝑝(𝑧|𝐸) − 𝑞(𝑧|𝐸)|)𝑝−1

|𝑝(𝑧|𝐸) − 𝑞(𝑧|𝐸)|𝑝 ≤ ( sup

𝑧̸=𝑥*

𝑧̸=𝑥*

𝑧̸=𝑥*

< 𝜖𝑝−1(2𝑛𝜖) = 2𝑛𝜖𝑝.

∑︁𝑛

𝜖′𝑗 < 𝑛𝜖.

𝑗=1

∑︁𝑛

𝜖′𝑗 < 𝑛𝜖.

𝑗=1

|𝑝(𝑧|𝐸) − 𝑞(𝑧|𝐸)|

Therefore,

𝐷𝑝 (𝑝,𝑞)𝑝 < ((𝑛 − 1)𝜖)𝑝 + 2𝑛𝜖𝑝 = ((𝑛 − 1)𝑝 + 2𝑛)𝜖𝑝. So,

𝐷𝑝 (𝑝,𝑞) < ((𝑛 − 1)𝑝 + 2𝑛)1/𝑝𝜖. For 𝑝 = 1,

𝐷1 (𝑝,𝑞) < (𝑛 − 1 + 2𝑛)𝜖 = (3𝑛 − 1)𝜖. And for Total Variation Distance,

- 1

- 2

3𝑛 − 1 2

𝐷𝑇𝑉 (𝑝,𝑞) =

𝐷1 (𝑝,𝑞) <

𝜖.

### Step 4: Bound the forward KL divergence.

𝑝(𝑧|𝐸) 𝑞(𝑧|𝐸)

∑︁

𝐷KL (𝑝‖𝑞) =

𝑝(𝑧|𝐸)log

𝑛|𝐸). The conditional total correlation can be expanded using the chain rule:

= 𝐼(𝑋𝑖

;...;𝑋𝑖

1

𝑧

∑︁𝑛

𝐼(𝑋𝑖

;...;𝑋𝑖

𝑛|𝐸) =

𝐼(𝑋𝑖

;𝑋𝑖

𝑘−1|𝐸).

,...,𝑋𝑖

1

1

𝑘

𝑘=2

Each term is bounded by the conditional entropy: 𝐼(𝑋𝑖

;𝑋𝑖

𝑘−1|𝐸) ≤ 𝐻(𝑋𝑖

𝑘|𝐸). The conditional entropy 𝐻(𝑋𝑖

,...,𝑋𝑖

1

𝑘

𝑘|𝐸) is bounded. Since 𝑝𝑘(𝑋𝑖

𝑘|𝐸) > 1−𝜖, it implies 𝑝𝑘(𝑋𝑖

= 𝑥𝑖

𝑘 ̸= 𝑥𝑖

𝑘|𝐸) = 𝜖′𝑘 < 𝜖. The entropy is maximized when the remaining probability 𝜖′𝑘 is spread uniformly, leading to:

𝑘

𝐻(𝑋𝑖

𝑘|𝐸) ≤ 𝐻𝑏(𝜖′𝑘) + 𝜖′𝑘 ln(|𝒱| − 1) < 𝐻𝑏(𝜖) + 𝜖ln(|𝒱| − 1). Summing (𝑛 − 1) such terms (for 𝑘 = 2,...,𝑛):

𝐷KL (𝑝‖𝑞) < (𝑛 − 1)[𝐻𝑏(𝜖) + 𝜖ln(|𝒱| − 1)].

| |
|---|

### Remark 1. Assumption of a Well-Defined Joint 𝑝𝜃(𝑋𝑖

𝑛|𝐸): The theorem and proof rely on 𝑝𝜃(𝑋𝑖

𝑛|𝐸) being a well-defined joint probability mass function from which the marginals 𝑝𝜃(𝑋𝑖

,...,𝑋𝑖

,...,𝑋𝑖

1

1

𝑗|𝐸) are consistently derived. This implies that the joint PMF is coherent and its definition does not depend on a specific factorization order beyond what is captured by the conditioning on 𝐸. In practice, while MDM may not strictly satisfy this property, its behavior typically offers a close approximation. The theorem holds for an idealized 𝑝𝜃 that possesses these properties. As MDMs become larger and more powerful, their learned distributions might better approximate such consistency.

Worst-Case Analysis: The conditions and bounds provided in the theorem (e.g., (𝑛 + 1)𝜖 ≤ 1) are derived from a worst-case analysis. This means the bounds are guaranteed to hold if the conditions are met, regardless of the specific structure of 𝑝𝜃(𝑋|𝐸) beyond the high-confidence marginal property. In practice, the actual case might be "better behaved" than the worst-case scenario. For instance, the dependencies between 𝑋𝑖

(given 𝐸) might be weaker than what the worst-case construction assumes. Consequently, the argmax equivalence (Result 1) might still hold frequently even if (𝑛 + 1)𝜖 is slightly greater than 1 (but not much larger). The condition identifies a threshold beyond which guarantees break down in the worst case, but practical performance can be more robust. Similarly, the actual 𝐿𝑝 distances or KL divergence might be smaller than the upper bounds suggest if the true joint 𝑝𝜃(𝑋|𝐸) is closer to the product of marginals 𝑞(𝑋|𝐸) than the worst-case configurations.

and 𝑋𝑖

𝑗

𝑘

## B. Case Study

Table 6 | Qualitative comparison of responses across methods.

Prompt: A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take? Original PrefixCache DualCache

The robe takes 2 bolts of blue fiber. It also takes half that much white fiber, so it takes 2/2 = 1 bolt of white fiber. In total, the robe takes 2 + 1 = 3 bolts of fiber. So, the value is 3

The robe takes 2 bolts of blue fiber. It also takes half that much white fiber, so it takes 2/2 = 1 bolt of white fiber. In total, the robe takes 2 + 1 = 3 bolts of fiber. So, the value is 3

The robe takes 2 bolts of blue fiber. It also takes half that much white fiber, so it takes 2/2 = 1 bolt of white fiber. In total, it takes 2 bolts + 1 bolt = 3 bolts of fiber. The final result is 3

- Table 7 | Qualitative comparison of responses with varying block size for DualCache.

Prompt: A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take? Block Size 8 Block Size 16 Block Size 32

The robe takes 2 bolts of blue fiber. It also takes half that much white fiber, so it takes 2/2 = 1 bolt of white fiber. In total, the robe takes 2 + 1 = 3 bolts of fiber. So, the value is 3

The robe takes 2 bolts of blue fiber. It also takes half that much white fiber, so it takes 2/2 = 1 bolt of white fiber. In total, the robe takes 2 + 1 = 3 bolts of fiber. So, the value is 3

The robe takes 2 bolts of blue fiber. It also takes half that much white fiber, so it takes 2/2 = 1 bolt of white fiber. In total, the robe takes 2 + 1 = 3 bolts of fiber. So, the value is 3

- Table 8 | Qualitative comparison of responses under different threshold settings.

Prompt: A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take? Threshold 0.7 Threshold 0.8 Threshold 0.9

The robe takes 2 bolts of blue fiber. It also takes half that much white fiber, so it takes 2/2 = 1 bolt of white fiber. In total, it takes takes 2 + 1 = 3 bolts of fiber. So, the value is 3 (NFE: 9)

The robe takes 2 bolts of blue fiber. It also takes half that much white fiber, so it takes 2/2 = 1 bolt of white fiber. In total, the robe takes 2 + 1 = 3 bolts of fiber. So, the value is 3 (NFE: 12)

The robe takes 2 bolts of blue fiber. It also takes half that much white fiber, so it takes 2/2 = 1 bolt of white fiber. In total, the robe takes 2 + 1 = 3 bolts of fiber. So, the value is 3 (NFE: 20)

### B.1. Effect of Caching Strategies on Response Quality

- Table 6 qualitatively compares answers from the Original, PrefixCache, and DualCache methods for the arithmetic prompt. All correctly compute the answer (3 bolts), following similar step-by-step reasoning, with only minor differences in phrasing. This shows cache strategies maintain answer accuracy and logical clarity while improving efficiency; semantic fidelity and interpretability are unaffected.

B.2. Effect of Block Size in DualCache

- Table 7 examines different block sizes (8, 16, 32) in DualCache. For this arithmetic prompt, all settings yield correct, clearly explained answers with no meaningful output differences. Thus, DualCache is robust to block size for such problems, allowing efficiency improvements without compromising quality.

B.3. Impact of Dynamic Threshold Settings

- Table 8 investigates dynamic threshold values (0.7, 0.8, 0.9). The model consistently produces the correct answer and clear explanations, regardless of threshold. While higher thresholds increase computational effort (NFE from 9 to 20), answer quality remains stable, indicating threshold adjustment mainly affects efficiency, not correctness, for straightforward arithmetic questions.

### B.4. Multimodal Generation with LLAda-V

To qualitatively analyze the effectiveness of our Fast-dLLM framework in multimodal scenarios, we conduct a visual case study where the model is tasked with generating a highly detailed image description. As illustrated in Figure 6, both the baseline model and our Fast-dLLM are given the same visual input and user prompt: “Please describe the image in detail.”

[Figure 13]

User: Please describe the image in detail.

Baseline (63.0 secs): The image captures a serene rural landscape.

Dominating the scene is a wooden barn, its weathered exterior a testament to the passage of time. The barn is nestled in a field of green grass, which stretches out to meet the horizon. In the distance, a range of mountains rises, their peaks shrouded in a blanket of clouds. The sky above is a soft pink, suggesting either dawn or dusk. The perspective of the image is from a low angle, making the barn appear grand and imposing. The overall composition of the image creates a sense of depth and tranquility, inviting the viewer to

imagine themselves in this peaceful rural setting.

Fast-dLLM (6.8 secs): The image captures a serene rural landscape. Dominating the scene is a wooden barn, its rustic charm enhanced by the gray shingles on its roof. The barn is nestled in a field of tall

grass, which stretches out towards the horizon. The field is framed by a range of mountains, their peaks dusted with snow, adding a touch of tranquility to the scene. The sky above is a soft pink, suggesting either early morning or late afternoon. The perspective of the image is from a low angle, making the barn appear grand and imposing. The overall composition of the image creates a harmonious blend of man-made structures and the natural world.

- Figure 6 | Comparison between the baseline and Fast-dLLM on a visual description task. Fast-dLLM produces a comparable and faithful image caption in a fraction of the decoding time.

The baseline model requires 63.0 seconds to complete the generation, producing a detailed and poetic description of the rural landscape. It highlights elements such as the weathered wooden barn, the soft pink sky, and the tranquil atmosphere.

In contrast, our Fast-dLLM completes the task in just 6.8 seconds—a nearly 10× speedup—while maintaining rich visual detail. It further enhances the description with additional grounding (e.g., “gray shingles on its roof”, “touch of tranquility”), reflecting a strong alignment with both appearance and mood cues from the image. Notably, the generated caption retains compositional depth and stylistic fluency, illustrating the model’s ability to balance fluency and factuality even under diffusion-based parallel decoding.

This case highlights how LLAda-V with Fast-dLLM decoding enables high-quality vision-language generation at significantly improved efficiency, paving the way for faster and more interactive multimodal applications.

## C. Experiment Details

### C.1. Further Experiments with LLaDA-V

Table 9 | Effect of block length on performance (MathVista, 48 Steps)

### Block Length 4 8 16 32 96

Accuracy (%) 51.2 50.7 51.8 52.3 59.7 Throughput (tok./s) 6.1 6.2 5.5 5.5 5.6

Table 10 | MathVista Performance with Fast-dLLM at different refresh intervals (block length = 96)

### Refresh Interval 2 4 8 16 32

Accuracy (%) 59.2 59.2 58.2 57.1 56.6 Throughput (tok./s) 15.9 19.5 21.1 25.2 28.2

In Table 9, we investigate how the choice of block length affects the performance of LLaDA-V on MathVista under a fixed decoding length of 48 steps. The results show that the model achieves the highest accuracy with a block length of 96. However, when reducing the block size to 8 or 4, the accuracy drops significantly by over 8%.

Given this sensitivity to block length, we choose not to break the output into small blocks for updating caches individually. Instead, we keep the block length fixed at 96 and adopt a refresh-based strategy: the cache is updated only every 𝑟 decoding steps using the most recent full block. As shown in Table 10, increasing the refresh interval leads to consistent gains in throughput—from 15.9 tokens/s at interval 2 to 28.2 tokens/s at interval 32. While accuracy drops slightly with larger intervals, it remains above 56.6%, suggesting that aggressive refresh scheduling can yield substantial speedups with only minor performance degradation.

### C.2. Performance Comparison between Threshold and Factor Strategy

- Table 11 | Performance comparison between Threshold and Factor confidence-aware decoding on GSM8K and MATH benchmarks with generation lengths of 256 and 512. Each block shows accuracy (top row) and throughput with speedup (bottom row). Factor decoding provides favorable trade-offs in most settings.

### Benchmark Gen. Len Threshold Factor

| | | |
|---|---|---|
|GSM8K (5-shot)|256|78.5 77.5 54.4 (8.1×) 78.5 (11.7x)<br><br>|
| |512<br><br>|77.2 74.8 35.3 (11.0×) 47.1 (14.7x)|
|MATH (4-shot)|256|33.2 32.0 51.7 (5.7×) 78.3 (8.6x)<br><br>|
| |512|36.0 35.2 47.1 (5.9×) 64.6 (8.1x)|

We compare the performance of our threshold-based and factor-based confidence-aware parallel decoding strategies on GSM8K and MATH benchmarks (Table 11). While the threshold strategy achieves marginally better accuracy in most settings (e.g., 78.5% vs. 77.5% on GSM8K with 256 tokens), the factor strategy demonstrates substantially superior throughput performance.

Specifically, factor decoding achieves 1.4-1.5× higher throughput than threshold decoding across all settings. On GSM8K with 256 tokens, factor decoding reaches 78.5 tokens/sec (11.7× speedup) compared to 54.4 tokens/sec (8.1× speedup) for threshold decoding. This throughput advantage becomes even more pronounced on longer generation tasks—for GSM8K with 512 tokens, factor decoding attains 47.1 tokens/sec while threshold only achieves 35.3 tokens/sec.

The results demonstrate that factor decoding offers a compelling trade-off: it sacrifices minimal accuracy (typically 1-3%) in exchange for significant throughput improvements (40-50% higher). This makes factor decoding particularly attractive for latency-sensitive applications where the slight accuracy reduction is acceptable. The consistent pattern across both benchmarks and generation lengths validates the robustness of the factor strategy’s theoretical foundation, which adaptively controls parallelism based on the confidence bound (𝑛 + 1)𝜖 < 𝑓.

Decoding Steps vs Parallel Token Count

95% Confidence Interval Avg Parallel Token Count

12

10

AverageParallelTokenCount

8

6

4

2

0

0 20 40 60 80 100

Step Index

- Figure 7 | Average number of tokens generated at each decoding step. Blue line shows the mean token count, and the shaded area denotes the 95% confidence interval.

### C.3. Comparison between LLaDA and LLaDA-1.5

We compare the performance of LLaDA and its enhanced version LLaDA-1.5 across both GSM8K (5-shot) and MATH (4-shot) benchmarks under two generation length settings (256 and 512 tokens), as shown in Table 12. Each cell reports accuracy and decoding throughput (in tokens per second), along with the relative speedup over the greedy baseline.

Across GSM8K settings, LLaDA-1.5 consistently improves accuracy over the original LLaDA, achieving a notable +2.2% absolute gain at 256-token generation and +3.2% at 512-token generation. Furthermore, it maintains strong decoding efficiency, with throughput reaching 59.4 tokens/sec at 256 tokens, improving upon LLaDA’s 54.1 tokens/sec under the same setting.

On the MATH benchmark, accuracy between the two versions remains comparable. However, LLaDA-1.5 slightly improves throughput at 256 tokens (53.7 vs. 51.7) while incurring a mild efficiency regression at the 512-token setting (41.1 vs. 47.1). This suggests that while LLaDA-1.5 introduces enhancements beneficial for shorter or moderate decoding contexts, longer sequences may require further optimization.

Overall, LLaDA-1.5 consistently provides either superior accuracy or better decoding speed across settings, demonstrating better performance-efficiency trade-offs and highlighting the benefit of incorporating adaptive improvements on top of the base LLaDA architecture.

### C.4. Analysis of Parallel Token Counts across Decoding Steps

To better understand the behavior of factor-based parallel generation, we analyze the average number of tokens generated at each decoding step. Specifically, we collect statistics from all intermediate steps of the sampling process and compute the average number of tokens generated in parallel per step. The results are visualized in Figure 7, along with a 95% confidence interval indicating cross-sample variability.

As shown in Figure 7, the average number of tokens generated in parallel gradually increases during the early to middle stages of decoding, peaking roughly between step 30 to step 60. After this peak, the parallelism tends to slightly decline toward the end of generation. This suggests that the model becomes more confident in generating outputs during the mid-decoding phase, allowing it to produce more tokens simultaneously. Toward the final steps, the decoding process tends to become more conservative, reducing the number of tokens produced at each step.

The shaded confidence interval reveals greater variance in later decoding steps, indicating instability and inconsistent generation behavior across samples. This is expected since tail-end decoding steps tend to handle only a few remaining

- Table 12 | Performance comparison between LLaDA and LLaDA-1.5. Each cell presents the accuracy and the decoding throughput in tokens per second with relative speedup to the LLaDA baseline (bottom row, blue: tokens per second/orange: relative speedup).

Benchmark Gen Length LLaDA (Fast-dLLM) LLaDA 1.5 (Fast-dLLM)

256 78.5 80.7

54.1 (8.1×) 59.4 (8.9×) 512 77.2 80.4

GSM8K (5-shot)

35.3 (11.0×) 33.0 (10.3×)

256 33.2 32.6

51.7 (5.7×) 53.7 (5.9×) 512 36.0 35.1

MATH (4-shot)

47.1 (5.9×) 41.1 (5.1×)

85

|3.79 4.41| | | | | |
|---|---|---|---|---|---|
|Selected<br><br>4.92<br><br>5.26 5.68| | | | | |
| | | | | | |
|Ours<br><br>2 token/step 4 token/step Non-Parallel Baseline (1 token/step)<br><br>| | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
|Ours<br><br>2 token/step 4 token/step Non-Parallel Baseline (1 token/step)<br><br>| | | | | |
|3.79<br><br>4.41 4.92| | | | | |
|| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>Selected<br><br>5.26 5.68| | | | | |
| | | | | | |

|3.79<br><br>4.92<br><br>Selected| | | | | | | | |
|---|---|---|---|---|---|---|---|---|
|4.41<br><br>5.26<br><br>5.68<br><br>2<br><br>4<br><br>Ours<br><br>Fixed-Step Baseline (2/4 tokens)<br><br>Non-Parallel Baseline (1 token/step)| | | | | | | | |
| | | | | | | | | |

80

250

###### GSM8K(5-shot)Accuracy

GSM8K(5-shot)Accuracy

80

70

###### InferenceSteps

75

200

60

70

150

65

50

100

60

40

50

55

30

50

0

0.7 1.0 1.3 1.6 1.9

0.7 1.0 1.3 1.6 1.9

1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0 5.5 6.0

Factor

Factor

Average #Tokens per Step

(a) (b) (c)

- Figure 8 | (a) GSM8K (5-shot) accuracy across different factor values using our factor-based decoding strategy. Numbers above each point indicate the average number of tokens decoded per step. The dashed lines show the accuracy of the baseline method with 2 or 4 tokens per step, and the non-parallel (1 token/step) baseline. (b) The corresponding number of inference steps needed under each factor setting. Our method generally requires significantly fewer steps than fixed-step baselines. (c) Accuracy versus average number of tokens decoded per step on GSM8K (5-shot). Our factor-based decoding achieves better accuracy-efficiency trade-offs compared to baselines. The red “Selected” point represents the setting chosen in our main results.

tokens required to complete the output, and the number of remaining tokens could differ widely among different samples (e.g., due to early completion or padding).

These observations are important for understanding how decoding efficiency can be optimized: increasing parallelism during high-confidence phases (middle steps) offers computational savings, while conservative behavior near boundaries maintains quality.

### C.5. Throughput Comparison under Varying Batch Sizes

All experiments are conducted on an NVIDIA A100 GPU, with the prefill length fixed to 256 tokens. The generation length is varied among 16, 32, and 64 tokens, and batch sizes range from 1 to 32. This setup reflects realistic deployment scenarios, allowing the evaluation of decoding efficiency under diverse conditions.

It should be noted that parallel decoding allows multiple tokens to be generated simultaneously affected by dummy input tokens. To ensure fairness, we focus solely on the acceleration provided by caching techniques.

PrefixCache is designed as an acceleration mechanism for LLaDA, a diffusion-based LLM, and successfully boosts the throughput significantly. Figure 9 shows that PrefixCache achieves consistent improvements across all batch sizes and generation lengths, making it particularly suited for scenarios with smaller generation lengths and larger batch sizes. For instance, with a generation length of 16 and batch size of 32, PrefixCache achieves a throughput of over 211 tokens/s, significantly outperforming the native LLaDA which reaches only 43 tokens/s, demonstrating nearly 5× improvement.

While LLaDA exhibits limited scalability with increasing batch sizes—its throughput plateaus after batch size 8—this

Throughput vs Batch Size

Generation Length

Method

L = 16 L = 32 L = 64

PrefixCache

| |
|---|

LLaDA

600

LLaMA (AR)

| |
|---|

500

Throughput(Token/s)

400

300

200

100

0

0 5 10 15 20 25 30

Batch Size

- Figure 9 | Throughput comparison between PrefixCache, LLaDA, and LLaMA under different generation lengths and batch sizes. All models are evaluated on an NVIDIA A100 GPU with the prefill length fixed at 256.

limitation is inherent to diffusion-based LLMs, which are compute-bound by nature. In contrast, LLaMA, an autoregressive (AR) model, benefits greatly from large batch sizes. As the batch size increases, LLaMA shifts from being memory-bound to compute-bound, allowing it to achieve high absolute throughput at larger batch settings.

These results highlight the practical advantages of PrefixCache in accelerating compute-bound diffusion models like LLaDA, especially for latency-critical and high-throughput applications. Furthermore, the scalability and efficiency provided by PrefixCache bridge the gap between diffusion-based LLMs and AR models like LLaMA, showcasing its importance for large-scale deployment settings.

## References

- [1] Marianne Arriola, Aaron Gokaslan, Justin T. Chiu, Zhihan Yang, Zhixuan Qi, Jiaqi Han, Subham Sekhar Sahoo, and Volodymyr Kuleshov. Block diffusion: Interpolating between autoregressive and diffusion language models, 2025.
- [2] Jacob Austin, Daniel D Johnson, Jonathan Ho, Daniel Tarlow, and Rianne Van Den Berg. Structured denoising diffusion models in discrete state-spaces. Advances in Neural Information Processing Systems, 34:17981–17993, 2021.
- [3] Andrew Campbell, Joe Benton, Valentin De Bortoli, Thomas Rainforth, George Deligiannidis, and Arnaud Doucet. A continuous time framework for discrete denoising models. Advances in Neural Information Processing Systems, 35:28266–28279, 2022.
- [4] Zixiang Chen, Huizhuo Yuan, Yongqian Li, Yiwen Kou, Junkai Zhang, and Quanquan Gu. Fast sampling via de-randomization for discrete diffusion models. arXiv preprint arXiv:2312.09193, 2023.
- [5] Itai Gat, Tal Remez, Neta Shaul, Felix Kreuk, Ricky TQ Chen, Gabriel Synnaeve, Yossi Adi, and Yaron Lipman. Discrete flow matching. arXiv preprint arXiv:2407.15595, 2024.
- [6] Daniel T Gillespie. Approximate accelerated stochastic simulation of chemically reacting systems. The Journal of chemical physics, 115(4):1716–1733, 2001.
- [7] Shansan Gong, Shivam Agarwal, Yizhe Zhang, Jiacheng Ye, Lin Zheng, Mukai Li, Chenxin An, Peilin Zhao, Wei Bi, Jiawei Han, et al. Scaling diffusion language models via adaptation from autoregressive models. arXiv preprint arXiv:2410.17891, 2024.
- [8] Google DeepMind. Gemini diffusion. https://deepmind.google/models/gemini-diffusion, 2025. Accessed: 2025-05-24.
- [9] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, et al. The llama 3 herd of models, 2024.
- [10] Zhengfu He, Tianxiang Sun, Kuanning Wang, Xuanjing Huang, and Xipeng Qiu. Diffusionbert: Improving generative masked language models with diffusion models. arXiv preprint arXiv:2211.15029, 2022.
- [11] Emiel Hoogeboom, Didrik Nielsen, Priyank Jaini, Patrick Forré, and Max Welling. Argmax flows and multinomial diffusion: Learning categorical distributions. Advances in Neural Information Processing Systems, 34:12454–12465, 2021.
- [12] Rongjie Huang, Jiawei Huang, Dongchao Yang, Yi Ren, Luping Liu, Mingze Li, Zhenhui Ye, Jinglin Liu, Xiang Yin, and Zhou Zhao. Make-an-audio: Text-to-audio generation with prompt-enhanced diffusion models, 2023.
- [13] Inception Labs. Introducing mercury: The first commercial diffusion-based language model. https:// www.inceptionlabs.ai/introducing-mercury, 2025. Accessed: 2025-05-24.
- [14] Ouail Kitouni, Niklas Nolte, James Hensman, and Bhaskar Mitra. Disk: A diffusion model for structured knowledge. arXiv preprint arXiv:2312.05253, 2023.
- [15] Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Ves Stoyanov, and Luke Zettlemoyer. Bart: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension, 2019.
- [16] Anji Liu, Oliver Broadrick, Mathias Niepert, and Guy Van den Broeck. Discrete copula diffusion. arXiv preprint arXiv:2410.01949, 2024.
- [17] Aaron Lou, Chenlin Meng, and Stefano Ermon. Discrete diffusion language modeling by estimating the ratios of the data distribution. arXiv preprint arXiv:2310.16834, 2023.
- [18] Chenlin Meng, Kristy Choi, Jiaming Song, and Stefano Ermon. Concrete score matching: Generalized score matching for discrete data. Advances in Neural Information Processing Systems, 35:34532–34545, 2022.
- [19] Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models, 2022.
- [20] Shen Nie, Fengqi Zhu, Chao Du, Tianyu Pang, Qian Liu, Guangtao Zeng, Min Lin, and Chongxuan Li. Scaling up masked diffusion models on text, 2025.
- [21] Shen Nie, Fengqi Zhu, Zebin You, Xiaolu Zhang, Jingyang Ou, Jun Hu, Jun Zhou, Yankai Lin, Ji-Rong Wen, and Chongxuan Li. Large language diffusion models, 2025.

- [22] Jingyang Ou, Shen Nie, Kaiwen Xue, Fengqi Zhu, Jiacheng Sun, Zhenguo Li, and Chongxuan Li. Your absorbing discrete diffusion secretly models the conditional distributions of clean data. arXiv preprint arXiv:2406.03736, 2024.
- [23] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation, 2021.
- [24] Machel Reid, Vincent J. Hellendoorn, and Graham Neubig. Diffuser: Discrete diffusion via edit-based reconstruction, 2022.
- [25] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models, 2022.
- [26] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily Denton, Seyed Kamyar Seyed Ghasemipour, Burcu Karagol Ayan, S. Sara Mahdavi, Rapha Gontijo Lopes, Tim Salimans, Jonathan Ho, David J Fleet, and Mohammad Norouzi. Photorealistic text-to-image diffusion models with deep language understanding, 2022.
- [27] Subham Sekhar Sahoo, Marianne Arriola, Yair Schiff, Aaron Gokaslan, Edgar Marroquin, Justin T Chiu, Alexander Rush, and Volodymyr Kuleshov. Simple and effective masked diffusion language models. arXiv preprint arXiv:2406.07524, 2024.
- [28] Jiaxin Shi, Kehang Han, Zhe Wang, Arnaud Doucet, and Michalis K Titsias. Simplified and generalized masked diffusion for discrete data. arXiv preprint arXiv:2406.04329, 2024.
- [29] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International conference on machine learning, pages 2256–2265. PMLR, 2015.
- [30] Jiaming Song and Linqi Zhou. Ideas in inference-time scaling can benefit generative pre-training algorithms. arXiv preprint arXiv:2503.07154, 2025.
- [31] Haoran Sun, Lijun Yu, Bo Dai, Dale Schuurmans, and Hanjun Dai. Score-based continuous-time discrete diffusion models. arXiv preprint arXiv:2211.16750, 2022.
- [32] Ashish Vaswani. Attention is all you need. arXiv preprint arXiv:1706.03762, 2017.
- [33] Yisheng Xiao, Lijun Wu, Junliang Guo, Juntao Li, Min Zhang, Tao Qin, and Tie yan Liu. A survey on non-autoregressive generation for neural machine translation and beyond, 2023.
- [34] Minkai Xu, Tomas Geffner, Karsten Kreis, Weili Nie, Yilun Xu, Jure Leskovec, Stefano Ermon, and Arash Vahdat. Energy-based diffusion language models for text generation. arXiv preprint arXiv:2410.21357, 2024.
- [35] Dongchao Yang, Jianwei Yu, Helin Wang, Wen Wang, Chao Weng, Yuexian Zou, and Dong Yu. Diffsound: Discrete diffusion model for text-to-sound generation, 2023.
- [36] Jiacheng Ye, Zhihui Xie, Lin Zheng, Jiahui Gao, Zirui Wu, Xin Jiang, Zhenguo Li, and Lingpeng Kong. Dream 7b, 2025.
- [37] Jiasheng Ye, Zaixiang Zheng, Yu Bao, Lihua Qian, and Quanquan Gu. Diffusion language models can perform many tasks with scaling and instruction-finetuning. arXiv preprint arXiv:2308.12219, 2023.
- [38] Zebin You, Shen Nie, Xiaolu Zhang, Jun Hu, Jun Zhou, Zhiwu Lu, Ji-Rong Wen, and Chongxuan Li. Llada-v: Large language diffusion models with visual instruction tuning. arXiv preprint arXiv:2505.16933, 2025.
- [39] Runpeng Yu, Qi Li, and Xinchao Wang. Discrete diffusion in large language and multimodal models: A survey, 2025.
- [40] Runpeng Yu, Xinyin Ma, and Xinchao Wang. Dimple: Discrete diffusion multimodal large language model with parallel decoding, 2025.
- [41] Kaiwen Zheng, Yongxin Chen, Hanzi Mao, Ming-Yu Liu, Jun Zhu, and Qinsheng Zhang. Masked diffusion models are secretly time-agnostic masked models and exploit inaccurate categorical sampling. arXiv preprint arXiv:2409.02908, 2024.
- [42] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. Judging llm-as-a-judge with mt-bench and chatbot arena. Advances in Neural Information Processing Systems, 36:46595–46623, 2023.
- [43] Kun Zhou, Yifan Li, Wayne Xin Zhao, and Ji-Rong Wen. Diffusion-nat: Self-prompting discrete diffusion for non-autoregressive text generation, 2023.
- [44] Fengqi Zhu, Rongzhen Wang, Shen Nie, Xiaolu Zhang, Chunwei Wu, Jun Hu, Jun Zhou, Jianfei Chen, Yankai Lin, Ji-Rong Wen, and Chongxuan Li. Llada 1.5: Variance-reduced preference optimization for large language diffusion models, 2025.

