# arXiv:2509.04185v1[cs.LG]4Sep2025

## Set Block Decoding is a Language Model Inference Accelerator

Itai Gat∗, Heli Ben-Hamu∗, Marton Havasi, Daniel Haziza, Jeremy Reizenstein, Gabriel Synnaeve, David Lopez-Paz, Brian Karrer, Yaron Lipman

FAIR at Meta

∗Equal contribution

Autoregressive next token prediction language models offer powerful capabilities but face significant challenges in practical deployment due to the high computational and memory costs of inference, particularly during the decoding stage. We introduce Set Block Decoding (SBD), a simple and flexible paradigm that accelerates generation by integrating standard next token prediction (NTP) and masked token prediction (MATP) within a single architecture. SBD allows the model to sample multiple, not necessarily consecutive, future tokens in parallel, a key distinction from previous acceleration methods. This flexibility allows the use of advanced solvers from the discrete diffusion literature, offering significant speedups without sacrificing accuracy. SBD requires no architectural changes or extra training hyperparameters, maintains compatibility with exact KV-caching, and can be implemented by fine-tuning existing next token prediction models. By fine-tuning Llama-3.1 8B and Qwen-3 8B, we demonstrate that SBD enables a 3-5x reduction in the number of forward passes required for generation while achieving same performance as equivalent NTP training.

1 Introduction

Next token prediction (NTP) language models (LMs) have demonstrated extraordinary capabilities across a spectrum of tasks, from natural language understanding to complex reasoning and code generation. These models, built upon the Transformer architecture, have scaled to hundreds of billions and even trillions of parameters, a growth that has been directly correlated with their enhanced performance. However, this large scale presents a challenge for practical deployment. The process of using a trained language model to generate predictions, known as inference or sampling, is both computationally expensive and memoryintensive, creating a significant barrier for realworld applications where low latency and high throughput are important.

39

4.0x

37

4.7x

35

Pass@1

- 3.6x
- 4.6x

33

31

Llama-3.1 SBD Llama-3.1 NTP

QWEN-3 SBD QWEN-3 NTP

29

27

1 2 3 4 5 Speedup

Figure 1 LiveCodeBench-V6 acceleration with Set Block Decoding with no performance reductions.

Language model inference consists of two parts: the prefilling and the decoding stages. In the prefilling stage, the prompt is processed simultaneously and its attention keys and values (KVs) are cached, which usually achieves a high efficiency on GPUs because thousands of tokens are processed together. In the decoding stage, tokens are sequentially generated one after the other and require the attention KVs of preceding tokens: this stage requires few FLOPs per token, yet the entire model weights must be read from GPU memory, along with all cached keys and values, as many times as the number of tokens we want to generate. The decoding stage, which typically dominates the total inference time is the primary focus of many language model optimization efforts. Three main avenues for improvements are: Model compression, system-level optimization, and novel algorithms and modeling.

This <m> <m> <m> <m>

model was trained on a large corpus of public public --- and ---

Transformer

This model was trained on a large corpus of <m> text <m> code

Block Causal attention Causal attention - KV cached Bidirectional attention

#### (a) SBD training/finetuning (b) SBD inference

Figure 2 (a) Set Block Decoder fine-tunes any native NTP transformer architecture to predict k future tokens (in this illustration k = 4) conditioned on an arbitrary subset of the future tokens (in this case, “text” and “code”); where the special mask token ‘m’ is used to hide future tokens to be predicted. Past tokens (in white) use causal attention while the future block (in blue) uses bidirectional attention, allowing future tokens to attend to each other. (b) During inference, SBD decodes one block at a time by revealing some subset of independent future tokens, where each row represents a single forward in the model. Once a block is decoded it is KV-cached (in pink, with causal attention).

One central approach for accelerating language models within the algorithmic/modeling efforts is speculative decoding: generate many tokens with a fast draft model and then verify and accept them with the slower target model. A smaller language model can be used as the draft model (Leviathan et al., 2023), or additional heads can be attached to the target language model itself to predict multiple independent tokens into the future (Stern et al., 2018; Cai et al., 2024). In both cases the target model is used to verify and accept a consecutive subsequence of these tokens.

The goal of this work is to introduce Set Block Decoding (SBD) language models, a flexible and arguably simpler alternative to the draft/target-approach for accelerating language models. SBD models seamlessly combine the standard next token prediction (NTP) paradigm with the masked token prediction (MATP) in the same transformer architecture and, in contrast to previous draft/target-approaches, can sample future tokens in arbitrary order and in parallel. This extra degree of freedom unlocks the possibility to employ specialized solvers from the masked diffusion literature, e.g., Ben-Hamu et al. (2025), to achieve significant speedups at no accuracy loss and gain a refined control over the speedup-accuracy tradeoff, see figure 1.

Set Block Decoding models offer the following advantages:

- 1. Simplicity: Single language model, no architectural changes and additions; no added hyper-parameters during training; a single new hyper-parameter during inference.
- 2. Flexibility: Can use advanced solvers from the discrete diffusion literature.
- 3. Efficiency: Compatible with exact KV-caching while offering 3-5x fewer model decoding forwards per generated token.
- 4. Computational-cost-effective: Can be quickly fine-tuned from an existing NTP language model.

Experimentally, we have fine-tuned Llama-3.1 8B and Qwen-3 8B models to make them SBD models and compared to the corresponding NTP baselines trained on the same data and with the same training parameters and noticed that SBD preserve the performance of the NTP models while allows accelerating inference by requiring 3-5x less model forwards.

2 Approach

In this section we define SBD models as well as how to train and sample them, see figure 2. For that end, we start with a bit of background on parallel block decoding.

Notation. We will mostly follow standard notation denoting a sequence of tokens x = (x1,x2,...,xL), where L is the sequence length and each token is an element of a vocabulary set xi ∈ V = {1,2,...,V }. We will use I,J ,T to define sets of indices; accordingly if I = (i1,...,im) then xI = (xi

#### ).

#### ,...,xi

1

m

- 2.1 Parallel block decoding

Large language models generate a sequence of tokens x = (x1,...,xL) by repeatedly predicting the next token xt given all previous tokens x<t = (x1,...,xt−2,xt−1),

p(xt|x<t). (1)

This approach is called Next-Token-Prediction (NTP). NTP inference requires one model evaluation per generated token and is therefore slow. Parallel block decoding is an attempt to accelerate language models by making them predict k future tokens (called a block) in a single forward pass (Stern et al., 2018),

p(xI|x<t), I = {t,t + 1,...,t + k − 1} (2)

where xI = (xt,xt+1,...,xt+k−1). However, since modeling the joint probability of more than a single token is intractable with today’s vocabulary sizes |V|, e.g., the joint probability mass function of two tokens requires |V|2 values in general, previous works resorted to independent parallel decoding, that is learning only the marginals

p(xi|x<t), for i ∈ I. (3) Independent parallel decoding provides an independent joint for the future tokens,

p(xI|x<t) =

i∈I

p(xi|x<t), (4)

which is in general only a crude approximation to the true joint and therefore requires a verification step that is done with the NTP model and therefore can only accept some consecutive prefix xt,xt+1,...,xt+a.

- 2.2 Set parallel block decoding

A more general parallel decoding framework is the set, or equivalently mask parallel decoding (Ghazvininejad et al., 2019; Lou et al., 2023; Gat et al., 2024; Shi et al., 2024; Nie et al., 2025). In a nutshell, Set Block Decoding performs independent parallel decoding of the k future tokens I but allows the model to see an arbitrary subset J ⊂ I of the future tokens. Equivalently, mask some of the k future tokens, M = I \ J , and try to predict the masked tokens,

p(xi|x<t,xJ ), for i ∈ M. (5)

This model provides a lot of flexibility in sampling: for any sequence of indices I1 ⊂ I2 ⊂ ··· ⊂ Iℓ = I we can decode all the tokens in Dj = Ij \ Ij−1 simultaneously leading to

ℓ

). (6)

p(xi|x<t,xI

p(xI|x<t) =

j−1

j=1 i∈Dj

This sampling will be exact if all xi, i ∈ Dj, are conditionally independent, i.e., p(xD

). (7)

j|x<t,xI

p(xi|x<t,xI

) =

j−1

j−1

i∈Dj

This flexibility was recently explored by Ben-Hamu et al. (2025) who suggested a practical algorithm utilizing this principle named Entropy Bounded (EB) Sampler. In its simplest form, the EB-Sampler finds at each step a subset of tokens to decode in parallel, Dj, by identifying the masked tokens with low mutual information, which quantifies the degree of dependence among these tokens. Since the mutual information cannot be computed without the full joint probability, an upper bound is used instead utilizing the entropy of the marginals p(xi|x<t,xJ ). That is, given we already decoded the tokens corresponding to the indices J ⊂ I and we are left with masks M = I \ J , we sort the masked token indices, M, in ascending order according to entropy of their predicted probabilities, i.e., H(p(xi|x<t,xJ )); denote the sorted masked indices by i1,i2,...,i|M|. Now, decode in parallel the s tokens i1,i2,...,is where s ≥ 1 is the largest integer so that s−1

j|x<t,xJ )) ≤ γ, (8)

H(p(xi

j=1

###### Algorithm 1 Set Block Decoding training Require: Init model params θ, data D.

Algorithm 2 Set Block Decoding inference Require: Model fθ, prompt x<0, block size k, length L

- 1: for data sequence x ∈ D do
- 2: Random noising probability τ ∼ U(0,1)
- 3: Create a masked sequence xˆ, see 13
- 4: Compute loss L(x,xˆ;θ), see 14
- 5: Compute gradients and update parameters θ
- 6: end for
- 7: return θ

- 1: x ← x<0
- 2: Prefill fθ(x;)
- 3: for t = 0,k,2k,...,L − k do
- 4: x<t+k ← sample_block(x<t)
- 5: end for
- 6: return x<L

and γ > 0 is a user prescribed threshold. At least one token is revealed in each iteration of this algorithm. This algorithm, in contrast to the more classical parallel decoding, can control the efficiency-accuracy tradeoff with the single hyperparameter, γ, and demonstrates impressive reduction in model forwards during sampling. However, translating reductions in model forwards to wall-clock speedups for state-of-the-art autoregressive language models requires combining set parallel decoding with classical next token prediction in a seamless manner. This is what SBD models offer, and this capability is described next.

2.3 Set block decoding model

We devise a modeling that allows training of a next token prediction (NTP) model (equation 1) empowered with set block decoding (SBD) abilities (equation 5). The main benefit of SBD models is that they preserve NTP performance while allowing to accelerate inference by using KV-caching and decoding several tokens simultaneously at each step.

An SBD network is a transformer fθ(·; ·) defined by

zt = fθ(x1,...,xt−1 ; ) (9a) (ˆzt,...,zˆt+k−1) = fθ(x1,...,xt−1 ; xˆt,...,xˆt+k−1), (9b)

where all tokens before (left of) ‘;’ are using causal attention, while tokens after (right of) ‘;’ use bidirectional attention, see figure 3. The “noisy” tokens, xˆt,...,xˆt+k−1, can be either real tokens from V or a mask token denoted ‘m’, i.e., xˆi ∈ V ∪ {m}, for i ∈ I; see figure 2 (a) for an illustration of fθ for k = 4.

As usual, each of the logits zt,zˆt ∈ RV define a probability mass function for the generated token xt defined via softmax. We consequently denote by

| |x1 x2 x3 ··· xt−1<br><br>|xˆt xˆt+1 xˆt+2 xˆt+3<br><br>|
|---|---|---|
|x1 x2 x3 . xt−1| | |
|xˆt<br><br>xˆt+1<br><br>xˆt+2<br><br>xˆt+3<br>| | |

pθ(xt|x<t;) (10) pθ(xt+i|x<t;xˆt,...,xˆt+k−1), i = 0,...,k − 1 (11)

the probability mass functions defined by the logits zt (top equation) and zˆt,...,zˆt+k−1 (bottom equation), respectively.

Figure 3 The attention mask of the SBD transformer fθ, equation 9.

Inference with an SBD model. The SBD model can be sampled with the help of any masked parallel decoding method. We opt for the EB-Sampler where the masked probability in equation 5 is defined using the model as

p(xi|x<t,xJ ) = pθ(xi|x<t;xˆt,...,xˆt+k−1), i ∈ M, (12) and M ⊂ I are the masked indices, xˆj = xj if j ∈ J = I \ M (i.e., unmasked) and xˆj = m otherwise.

#### Algorithm 2 provides a sampling pseudo-code, where sample_block is the procedure for sampling the next block and is provided in algorithm 3, utilizing the EB-Sampler (section 2.2); see figure 2 (b) for an illustration of the sampling algorithm with block size k = 4.

TraininganSBDmodel. Training the SBD model combines next token prediction and masked prediction losses. In more detail, consider given a sequence of tokens x = (x1,...,xL), we create an additional masked sequence xˆ = (ˆx1,...,xˆL) by setting, for i ∈ [L], where [L] = {1,...,L},

m with prob η xi with prob 1 − η

(13)

xˆi =

where η is randomized uniformly in (0,1). Then the SBD loss for the sequence x and block size k is

Algorithm 3 Sample block Require: Model fθ, input sequence x<t, block size k

- 1: xˆt:t+k−1 = (m,...,m) ▷ init with k mask tokens
- 2: while xˆt:t+k−1 contains masks do
- 3: if first iteration and t ≥ k then
- 4: zˆt:t+k−1 ← fθ(x<t−k,

KV-cache xt−k:t−1;xˆt:t+k−1)

- 5: else
- 6: zˆt:t+k−1 ← fθ(x<t;xˆt:t+k−1)
- 7: end if
- 8: Compute probabilities from zˆt:t+k−1
- 9: Unmask tokens in xˆt:t+k−1 according to 8
- 10: end while
- 11: return (x<t,xˆt:t+k−1)

Next token prediction L

L(x,xˆ;θ) = −

log pθ(xt|x<t;)−

t=2

Masked token prediction

k−1

t+i=m log pθ(xt+i|x<t;xˆt,...,xˆt+k−1) (14)

1xˆ

t∈T

i=0

where 1C is 1 when condition C holds and otherwise 0, and T = 1 + ℓk | ℓ = 0,1,2,...,⌊Lk ⌋ − 1 . Algorithm 1 provide training pseudo-code, and figure 2 (a) and figure 8 illustrate the training (input, target and attention)

for the running example of block size k = 4.

### 3 Experiments

In this section we report experiments and evaluations of the Set Block Decoding method for accelerating large language models. In section 3.1 we fine-tune two leading 8B models (Llama-3.1 and Qwen-3) and benchmark them on a suite of generation tasks covering reasoning, coding, and mathematics. Next, in section 3.2 we investigate the method’s scaling properties compared to NTP models on smaller, 3B models. We use these experiments to formulate a training recipe for SBD models.

- 3.1 Benchmarks

Experimental setup. We fine-tune Llama-3.1 8B base (Meta, 2024) and Qwen-3 8B (Yang et al., 2025a) base models on 70B tokens. We train these models on a mix of reasoning and instruction data with a 32k token context length. The reasoning data mix is composed of mitigated versions of the OpenCodeReasoning (Ahmad et al., 2025) and OpenMathReasoning (Moshkov et al., 2025) datasets, where mitigations including algorithmic bias filtering and cybersecurity protections, were applied. For instruct data, we use a publicly available mix, similar to the one used for training Llama-3.1 Instruct. For optimization, we use AdamW (Loshchilov and Hutter, 2019) with a 3e-4 learning rate, a warmup of 200 iterations, and cosine annealing schedule. We use a batch size of 2M tokens and total 34k iterations of fine-tuning. To support our method, we train models with a variable block size by uniformly sampling a size from the range [2,16] at each training step. At inference, we use EB-Sampler as described in equation 8 with temperature 0.

Results. Table 1 presents results for our SBD models on reasoning benchmarks AIME25, LiveCodeBench v6 (Jain et al., 2024), Math500 (Lightman et al., 2023), as well as chat benchmarks GSM8K (Cobbe et al., 2021), HumanEval+ (Liu et al., 2023), and MBPP (Austin et al., 2021). For reasoning benchmarks, we generate with thinking and up to 32k tokens; for chat benchmarks we do not use thinking and generate 1024 tokens for HumanEval+ and GSM8K, and 256 for MBPP. The γ values we used for reasoning are γlow = 0.1,γhigh = 0.35; while for chat benchmarks we use γlow = 0.35,γhigh = 0.6.

To isolate the effect of training data quality on performance and to ensure a fair comparison between SBD and NTP, for each base model we trained an NTP baseline, corresponding to NTP in the loss column in table 1, in the exact same experimental setting and confirmed that the SBD performance is consistent with it.

Reasoning Chat MATH500 AIME25 LCB V6 GSM8K HE+ MBPP

Base Model Training Loss Sampling Speedup

Gemini diffusion - - Diffusion - - 23.3 30.9 - - 76.0 Mercury - - Diffusion - - - 25.0 - - 76.6 Diffucoder Scratch MATP Diffusion 1x - - - - 68.3 67.5 Llada 1.5 Scratch MATP Diffusion 1x - - - 83.3 52.4 42.8 Dream-coder FT MATP Diffusion 1x - - 21.4 - - 79.6

NTP NTP 1x 80.2 33.3 31.5 85.3 56.7 70.4 SBD NTP 1x 81.6 30.0 31.3 85.6 57.9 70.9 SBD SBD (γ

Llama-3.1 8B FT

###### ) 3.4x 80.4 (3.23x) 23.3 (4.50x) 29.9 (4.59x) 84.0 (2.34x) 54.9 (2.95x) 67.2 (2.66x) SBD SBD (γ

high

) 3.0x 81.0 (3.55x) 30.0 (3.35x) 31.7 (3.72x) 84.2 (2.20x) 57.9 (2.61x) 69.5 (2.49x)

low

NTP NTP 1x 86.6 33.3 36.6 90.1 69.5 78.0 SBD NTP 1x 86.6 33.3 37.4 90.4 66.5 77.7 SBD SBD (γ

Qwen-3 8B FT

###### ) 3.8x 85.4 (3.54x) 26.6 (5.06x) 33.3 (5.36x) 88.7 (2.86x) 65.2 (2.72x) 74.6 (3.03x) SBD SBD (γ

high

) 3.2x 85.0 (3.41x) 33.3 (3.85x) 37.2 (3.92x) 90.1 (2.77x) 66.5 (2.51x) 77.5 (2.61x)

low

- Table 1 Reasoning, coding, and math benchmark results. All baselines use greedy decoding. SBD models are sampled with a fixed entropy threshold and two γ settings: a low value that preserves accuracy and a high value that prioritizes speed. Speedup is measured as the reduction in NFEs (model forwards); see section 4 for the wall-clock time analysis.

We find that for low γ values, our method preserves the autoregressive performance while achieving a 3-5x speedup (benchmark-dependent) and higher γ values yield even greater speedups at the cost of a minor drop in performance. Figure 1 illustrates the tradeoff between speed and performance when tweaking the γ threshold of the EB-Sampler.

In all benchmarks, similar to the “generate until” logic presented in Ben-Hamu et al. (2025), we measure speedup only until the end of the problem’s solution.

- 3.2 Ablations

Experimental setup. For all experiments in this section we use a 3B transformer model, with 28 layers, and a hidden dimension of 3072. For pretraining experiments, we use AdamW with a peak learning rate of 1.5e-3, warmup of 2000 steps and a cosine annealing schedule, for a total of 1T tokens from a mitigated version of DCLM (Li et al., 2024a) and raw code data. For instruct SBD fine-tuning, we use AdamW with a peak learning rate of 1e-5, warmup of 200 steps and a cosine annealing schedule. We train with the same instruct data used in section 3.1.

Role of NTP loss term. We assess the significance of the NTP loss term (see equation 14) by training two variants: standard SBD, and a variant where we do not take gradients over the NTP term, but we do log its values during training. Both variants begin from the same intermediate training step, namely the 900B checkpoint of a standard NTP model pretraining. We train SBD with and without the NTP term for the remaining 100B tokens, continuing with the same exact optimization hyperparameters as the NTP training.

|NTP SBD<br><br>|SBD w/o NTP|
|---|---|
| | |

2.15

2.10

2.05

NTPLoss

2.00

1.95

1.90

1.85

1.80

0.90 0.92 0.94 0.96 0.98 1.00 Tokens (T)

- Figure 4 shows the NTP loss during training in three configurations: (i) standard NTP model training (gray), (ii) SBD training (orange), and (iii) SBD training w/o NTP term (blue). One can see that without the NTP loss term, SBD training will not maintain the AR capabilities of the base model, while training SBD with the full loss only results in a slightly higher NTP loss.

Figure 4 NTP loss during 3B model pretraining.

###### HumanEval MBPP GSM8K

60

60

40

55

55

35

Accuracy(%)

Accuracy(%)

Accuracy(%)

50

50

30

45

45

25

40

40

Training iterations

20

8.5k

17k

34k

35

35

EB

AR

15

200 400 600 800 1000 Mean NFE

200 400 600 800 1000 Mean NFE

200 300 400 500 600 700 800 900 1000 Mean NFE

##### Figure 5 3B model SFT training with varying number of training iterations.

MMLU GPQA Hellaswag Winogrande ARC-E ARC-C NTP 50.9 24.1 76.4 69.5 71.2 51.4 SBD 49.9 -1.0% 27.7 +3.6% 75.6 -0.8% 70.2 +0.7% 69.9 -1.3% 48.6 -2.8% SBD w/o NTP loss 43.2 -7.7% 27.2 +3.1% 71.1 -5.2% 67.2 -2.2% 58.8 -12.4% 43.2 -8.2%

- Table 2 Ablation on 3B-parameters pretrained models trained on 1T tokens. SBD models start from an intermediate training step of the NTP pretrained model and are trained for the last 10% of the tokens. The difference in accuracy compared to the NTP baseline is listed in subscript for the SBD models.

Table 2 further validates the observed behavior in figure 4 by evaluating the models on likelihood tasks: MMLU (Hendrycks et al., 2021), GPQA (Rein et al., 2023), Hellaswag (Zellers et al., 2019), Winogrande (Sakaguchi et al., 2019), Arc-E and ARC-C (Clark et al., 2018), showing significant decrease in accuracy when the NTP loss term is removed compared to full SBD training loss. For the SBD models, inference is done using the NTP prediction as in equation 9a.

Number of training steps. State-of-the-art language models are first fully pretrained and then undergo a supervised fine-tuning (SFT) in order to adjust them for specific use cases (e.g., instruct models). The SFT stage is typically significantly shorter than the full pretraining stage and is one of the most common post-training procedures. In this experiment we show SBD can be incorporated effectively into the SFT stage. That is, given a pretrained model, perform SFT with the SBD training scheme. We use the same instruct data used in section 3.1 and the same training hyperparameters for both NTP and SBD fine-tuning. Figure 5 shows SFT models’ performance for both NTP training and SBD training at 1024 generation length, for varying number of training iterations. We observe that SBD requires more steps to reach on-par performance as NTP training but closes the performance gap after roughly 34k training iterations. We also show the performance of the SBD models for different EB-Sampler γ values in {0,0.01,0.1,0.2,0.4,0.8,1.5}. This demonstrates that the EB-Sampler’s behavior, providing speed up at no accuracy loss, as shown in Ben-Hamu et al. (2025) for masked models, translates to the SBD paradigm. In appendix B.1 we also show comparison to the factor sampling algorithm introduced in Wu et al. (2025a).

- 4 Timing analysis 4.1 Roofline model for block inference

Standard NTP models perform L forwards to generate a sequence of length L. In contrast, SBD models reduce this number of model forwards by a factor of 3-5x (see factors in table 1) but each forward simultaneously computes a block of k tokens. In this section we provide a theoretical (“roofline”) analysis to justify why these model forward saving are expected to translate to almost identical factors of wall-clock savings in practice.

Our model assumes the H100 Nvidia GPU and standard 8B transformer model with the following parameters:

- 1 # --- H100 GPU Specifications ---
- 2 BYTES: int = 1 # model weights, we assume FP8
- 3 PEAK_FLOPS: float = 1978 * 1e12 # peak flops for FP8
- 4 PEAK_FLOPS_ATTENTION: float = 989 * 1e12 # attention is done in BF16
- 5 MEMORY_BANDWIDTH: float = 3.35 * (1024**4) # memory bandwidth in B/s
- 6
- 7 # --- Transformer Model Configuration ---
- 8 HIDDEN_DIM: int = 4096
- 9 FFN_DIM: int = 4 * HIDDEN_DIM
- 10 NUM_HEADS: int = 32
- 11 HEAD_DIM: int = HIDDEN_DIM // NUM_HEADS
- 12 NUM_KV_HEADS: int = 8
- 13 BYTES_PER_KV_ELEMENT: float = 1 # bytes used per KV element

Figure 6 Constants for roofline analysis.

The PEAK_FLOPS defines the maximal flops/sec for FP8 in H100 while MEMORY_BANDWIDTH is the maximal bytes/sec. The main equation to determine the theoretical time of an operation is to take the max upon memory transfers time and compute time via

theoretical_time = max

total_flops PEAK_FLOPS

,

total_memory MEMORY_BANDWIDTH

. (15)

If the maximum is achieved by the memory part then the computation is called memory bound and otherwise it is named compute bound (Williams et al., 2008).

To compare standard NTP and SBD decoding time we analyze different KV Cache lengths, supporting different stages of the generation process with potentially long generation sequences, different sizes of block sizes {1,2,4,8,16,32,64}, where block size of 1 corresponds to NTP sampling and serves as baseline, and batch size, batch_size ∈ {1,4,8,16}. To analyze the theoretical forward time of a transformer we sum the theoretical times of its two components, namely the multi-head attention operation and linear layer. For the attention, we assume a single fused kernel like in Dao et al. (2022). In the appendix, we provide Python code to calculate these theoretical runtimes (figure 11). Table 3 shows the slowdowns, i.e., the theoretical time ratios, of block sampling and NTP sampling for different block and batch sizes and KV cache length. For example, for block size of 16, the block inference slowdown over NTP is not more than few percents. In the next section, we use this analysis to provide theoretical wall-clock saving times for different NFE speedups.

4.2 Set block decoding theoretical speedups

Consider sampling a block of k tokens from the model. We want to estimate the wall-clock speedup of SBD sampling over NTP. Recall the SBD inference procedure in algorithm 2, which is illustrated in figure 2 (b), and assume that the SBD sampling uses l model forwards (l < k). Then the wall-clock speedup will be the ratio

k time(1) time(2k) + (l − 1)time(k)

, (16) where time(k) is the forward time of a block of size k in the network.

speedup(l,k) =

Block Size 1 2 4 8 16 32 64

Block Size 1 2 4 8 16 32 64

KV Cache Length

KV Cache Length

24 (16) 1.000 1.000 1.001 1.002 1.004 1.008 1.015 28 (256) 1.000 1.000 1.001 1.002 1.004 1.008 1.015 210 (1,024) 1.000 1.000 1.001 1.002 1.004 1.008 1.015 212 (4,096) 1.000 1.000 1.001 1.002 1.004 1.008 1.015 214 (16,384) 1.000 1.000 1.001 1.002 1.004 1.008 1.015 216 (65,536) 1.000 1.000 1.001 1.002 1.004 1.008 1.016 220 (1,048,576) 1.000 1.000 1.001 1.002 1.004 1.008 1.019 222 (4,194,304) 1.000 1.000 1.001 1.002 1.004 1.007 1.029

24 (16) 1.000 1.001 1.003 1.007 1.015 1.030 1.061 28 (256) 1.000 1.001 1.003 1.007 1.015 1.030 1.061 210 (1,024) 1.000 1.001 1.003 1.007 1.015 1.030 1.061 212 (4,096) 1.000 1.001 1.003 1.007 1.015 1.030 1.062 214 (16,384) 1.000 1.001 1.003 1.007 1.015 1.030 1.062 216 (65,536) 1.000 1.001 1.003 1.007 1.015 1.030 1.062 220 (1,048,576) 1.000 1.001 1.003 1.007 1.014 1.030 1.074 222 (4,194,304) 1.000 1.001 1.003 1.006 1.014 1.028 1.111

batch_size=1 batch_size=4

Block Size 1 2 4 8 16 32 64

Block Size 1 2 4 8 16 32 64

KV Cache Length

KV Cache Length

24 (16) 1.000 1.002 1.006 1.014 1.029 1.060 1.903 28 (256) 1.000 1.002 1.006 1.014 1.029 1.060 1.903 210 (1,024) 1.000 1.002 1.006 1.014 1.029 1.060 1.903 212 (4,096) 1.000 1.002 1.006 1.014 1.029 1.060 1.903 214 (16,384) 1.000 1.002 1.006 1.014 1.029 1.060 1.903 216 (65,536) 1.000 1.002 1.006 1.014 1.029 1.060 1.903 220 (1,048,576) 1.000 1.002 1.006 1.013 1.028 1.059 1.903 222 (4,194,304) 1.000 1.002 1.005 1.012 1.026 1.054 1.904

24 (16) 1.000 1.004 1.012 1.027 1.058 1.899 3.799 28 (256) 1.000 1.004 1.012 1.027 1.058 1.899 3.799 210 (1,024) 1.000 1.004 1.012 1.027 1.058 1.899 3.799 212 (4,096) 1.000 1.004 1.012 1.027 1.058 1.899 3.798 214 (16,384) 1.000 1.004 1.012 1.027 1.058 1.899 3.797 216 (65,536) 1.000 1.004 1.012 1.027 1.058 1.896 3.792 220 (1,048,576) 1.000 1.004 1.011 1.026 1.055 1.847 3.688 222 (4,194,304) 1.000 1.003 1.009 1.022 1.047 1.720 3.422

batch_size=8 batch_size=16

- Table 3 Slowdown factors of a single block forward (for different batch and block sizes and KV cache lengths) compared to a single token forward used in standard NTP sampling, according to theoretical “Roofline” analysis for an H100 NVIDIA GPU and a standard 8B transformer architecture. Uncolored cells exhibit small slowdown. Note: no NFE speedup is considered here.

To estimates these times, we will use the roofline analysis from section 4. Table 4 depicts speedup(l,k) for different batch and block sizes, KV cache length and

NFE_speedup =

- k

- l

. (17)

Note that for block size 16 the NFE speedups translate almost directly to theoretical wall-clock speedups, which covers the experimental setting and provides an evidence for potential 3-5x wall-clock speedup given the NFE speedups in table 1. Lastly, we note that as batch sizes and/or block size increase the roofline analysis indicates diminishing returns due to increased computational costs of block forwards (see colored cells).

- 5 Related work

Efficient large language modeling is a large topic that we do not attempt to cover comprehensively. Instead we summarize research that is most connected to SBD, describing recent diffusion language models as well as related efficiency efforts. Like SBD, this research proposes higher efficiency via computationally cheaper and fewer model evaluations, e.g., applying key-value caching, fusing sequential operations as in tree attention, and parallel decoding. We conclude by discussing hybrid language models.

Diffusion language models using masked discrete diffusion. Large language modeling via diffusion at the several billion (or larger) parameter scale has only recently become successful with models such as Dream (Wu et al., 2025b; Xie et al., 2025), LLaDa (You et al., 2025; Liu et al., 2025a), MMaDa (Yang et al., 2025b), Dimple (Yu et al., 2025), Mercury (Labs et al., 2025), DiffuCoder (Gong et al., 2025), Seed Diffusion (Song et al., 2025), and Gemini Diffusion (Deepmind, 2025). Prior to the arrival of these masked diffusion models, discrete diffusion for text had been mostly limited to smaller scales. These masked diffusion models are competitive with traditional autoregressive language models on task performance, but are still lacking on end-to-end efficiency, outside commercially developed models, such as Mercury, Seed Diffusion, and Gemini Diffusion, whose inner workings are undisclosed.

Efficient large language models with causal attention. Typical language models are autoregressive with a fixed left-to-right order and leverage Transformers using causal attention, enabling the reuse of model computation via key-value caching. This caching is crucial for efficient sampling from these models. Greedy decoding (Stern et al., 2018) and non-greedy speculative sampling (Chen et al., 2023; Leviathan et al.,

Block Size 8 16 32 64

Block Size 8 16 32 64

Block Size 8 16 32 64

KV Cache Length

KV Cache Length

KV Cache Length

24 (16) 1.996 1.992 1.984 1.969 28 (256) 1.996 1.992 1.984 1.969 210 (1,024) 1.996 1.992 1.984 1.969 212 (4,096) 1.996 1.992 1.984 1.969 214 (16,384) 1.996 1.992 1.984 1.969 216 (65,536) 1.996 1.992 1.984 1.968 220 (1,048,576) 1.996 1.992 1.984 1.962 222 (4,194,304) 1.996 1.992 1.983 1.941

24 (16) 1.983 1.967 1.938 1.839 28 (256) 1.983 1.967 1.938 1.839 210 (1,024) 1.983 1.967 1.938 1.838 212 (4,096) 1.983 1.967 1.938 1.838 214 (16,384) 1.983 1.967 1.938 1.838 216 (65,536) 1.983 1.967 1.938 1.837 220 (1,048,576) 1.983 1.968 1.937 1.816 222 (4,194,304) 1.984 1.969 1.935 1.755

24 (16) 1.966 1.936 1.797 1.019 28 (256) 1.966 1.936 1.797 1.019 210 (1,024) 1.966 1.936 1.797 1.019 212 (4,096) 1.966 1.936 1.797 1.019 214 (16,384) 1.966 1.936 1.797 1.019 216 (65,536) 1.966 1.936 1.797 1.019 220 (1,048,576) 1.967 1.938 1.800 1.019 222 (4,194,304) 1.969 1.943 1.807 1.019

batch_size=1, NFE_speedup=2 batch_size=4, NFE_speedup=2 batch_size=8, NFE_speedup=2

Block Size 8 16 32 64

Block Size 8 16 32 64

Block Size 8 16 32 64

KV Cache Length

KV Cache Length

KV Cache Length

24 (16) 3.989 3.982 3.966 3.936 28 (256) 3.989 3.982 3.966 3.936 210 (1,024) 3.989 3.982 3.966 3.936 212 (4,096) 3.989 3.982 3.966 3.936 214 (16,384) 3.989 3.982 3.966 3.935 216 (65,536) 3.989 3.982 3.966 3.935 220 (1,048,576) 3.989 3.982 3.965 3.920 222 (4,194,304) 3.989 3.982 3.960 3.876

24 (16) 3.958 3.927 3.868 3.590 28 (256) 3.958 3.927 3.868 3.590 210 (1,024) 3.958 3.927 3.868 3.590 212 (4,096) 3.958 3.927 3.868 3.590 214 (16,384) 3.958 3.927 3.868 3.589 216 (65,536) 3.958 3.927 3.868 3.587 220 (1,048,576) 3.958 3.928 3.863 3.545 222 (4,194,304) 3.960 3.931 3.851 3.425

24 (16) 3.916 3.857 3.431 1.978 28 (256) 3.916 3.857 3.431 1.978 210 (1,024) 3.916 3.857 3.431 1.978 212 (4,096) 3.916 3.857 3.431 1.978 214 (16,384) 3.916 3.857 3.431 1.978 216 (65,536) 3.916 3.857 3.431 1.978 220 (1,048,576) 3.918 3.861 3.436 1.978 222 (4,194,304) 3.925 3.872 3.448 1.978

batch_size=1, NFE_speedup=4 batch_size=4, NFE_speedup=4 batch_size=8, NFE_speedup=4

Block Size 8 16 32 64

Block Size 8 16 32 64

Block Size 8 16 32 64

KV Cache Length

KV Cache Length

KV Cache Length

24 (16) 7.971 7.955 7.925 7.864 28 (256) 7.971 7.955 7.925 7.864 210 (1,024) 7.971 7.955 7.925 7.864 212 (4,096) 7.971 7.955 7.925 7.864 214 (16,384) 7.971 7.955 7.924 7.863 216 (65,536) 7.971 7.955 7.924 7.862 220 (1,048,576) 7.971 7.956 7.918 7.830 222 (4,194,304) 7.971 7.956 7.898 7.732

24 (16) 7.885 7.824 7.707 6.856 28 (256) 7.885 7.824 7.707 6.856 210 (1,024) 7.885 7.824 7.707 6.856 212 (4,096) 7.885 7.824 7.707 6.855 214 (16,384) 7.885 7.824 7.706 6.854 216 (65,536) 7.885 7.825 7.705 6.850 220 (1,048,576) 7.886 7.827 7.685 6.768 222 (4,194,304) 7.891 7.834 7.625 6.534

24 (16) 7.773 7.657 6.294 3.736 28 (256) 7.773 7.657 6.294 3.736 210 (1,024) 7.773 7.657 6.294 3.736 212 (4,096) 7.773 7.657 6.294 3.736 214 (16,384) 7.773 7.657 6.294 3.736 216 (65,536) 7.773 7.657 6.294 3.736 220 (1,048,576) 7.779 7.667 6.300 3.736 222 (4,194,304) 7.797 7.693 6.318 3.736

batch_size=1, NFE_speedup=8 batch_size=4, NFE_speedup=8 batch_size=8, NFE_speedup=8

- Table 4 Theoretical wall-clock speedups of block decoding compared to standard NTP sampling for different NFE speedups based on roofline analysis. Uncolored cells exhibit wall-clock speedup roughly equivalent to NFE speedup.

2023) can improve efficiency further, by proposing candidate sequences from a cheap draft model and only evaluating those candidates with the NTP language model. For example, text diffusion was proposed as a draft in Christopher et al. (2025). Because multiple models adds system complexity, approaches such as multi-token prediction (Gloeckle et al., 2024), Medusa (Cai et al., 2024), and Eagle (Li et al., 2024b), add output heads to an existing language model to predict consecutive future tokens. This multi-head prediction enables self-speculative decoding without a separate draft. Unlike our approach, these methods require adding architectural constructions which introduce a tradeoff challenge and a large to explore hyperparameter space. Any-order autoregressive models (Uria et al., 2014; Hoogeboom et al., 2022) discards the left-to-right order but maintains causal attention and exact key-value caches, and have similar draft variants (Pannatier et al.,

- 2024; Guo and Ermon, 2025).

Efficient diffusion language models with bidirectional attention. As strongly performing masked diffusion models are recent, efficient sampling is even more recent and has focused on improvements to LLaDa and Dream. This research has introduced approximate key-value caching (Ma et al., 2025; Liu et al., 2025b) and adaptive multi-token sampling (Ben-Hamu et al., 2025), and explored both avenues simultaneously (Wu et al.,

- 2025a; Hu et al., 2025; Israel et al., 2025). Exact key-value caching is not possible with bidirectional attention, and heuristic approximations rely upon utilize empirical observations of slowly changing representations, especially for mask tokens. On the sampling side, (Ben-Hamu et al., 2025; Wu et al., 2025a) propose nongreedy and greedy decoding schemes designed to control error from parallel multi-token sampling, while Hu et al. (2025); Israel et al. (2025) consider leveraging autoregressive model outputs to correct for independent unmasking. Unmasking orders are often constrained, with semi-autoregressive blockwise decoding used in Wu et al. (2025a); Hu et al. (2025) and even left-to-right decoding suggested (Israel et al., 2025).

Hybrid language models. Most related to SBD are recent proposals to mix left-to-right and parallel modeling. Block diffusion (BD3-LM) (Arriola et al., 2025), proposes semi-autoregressive generation within blocks, uses block-causal attention and exact key-value caching for preceding blocks, proposes unmasking tokens within a block using diffusion and bidirectional attention. CtrlDiff built upon BD3-LM adding adaptive

block size selection (Huang and Tang, 2025). Fathi et al. (2025) concurrently built upon BD3-LM to combine NTP and block diffusion, however with the goal of testing hybrid probability paths with a mix of uniform and masked noising process for improving performance at the cost of longer inference. Finally Esoteric Language Models (Sahoo et al., 2025), claim to improve upon BD3-LM, by considering a hybrid construction that uses bidirectional attention over clean tokens and causal attention over masked tokens to enable KV-caching. While these methods focus on a similar goal, fusing NTP with MATP models, our work introduces an efficient method to fine-tune an existing NTP model, taking advantage of the efficient NTP training, while providing it with the ability for fast block decoding; this allows us to gain a 3-5x speedup without altering the model architecture or compromising its performance.

- 6 Conclusion and future work

This work introduces Set Block Decoding (SBD), a simple and effective paradigm for accelerating the inference of large language models. By integrating masked-token prediction directly into a standard autoregressive architecture, SBD models can decode multiple, non-consecutive tokens in parallel. SBD avoids the complexity of auxiliary models and complex architectural constructions and requires no architectural changes, making it a practical solution that can be readily implemented by fine-tuning existing language models. Our experiments with Llama-3.1 8B and Qwen-3 8B demonstrate that SBD reduces the number of required forward passes by 3x-5x without compromising the model’s original performance.

There are several interesting directions for future work. A key direction is scaling SBD to even larger models to investigate its scaling properties. Furthermore, developing hardware-aware inference implementations to match the theoretical roofline analysis, as well as exploring a wider range of advanced samplers from the discrete diffusion literature, could further unlock the potential of SBD to maximize wall-clock speedups.

- 7 Acknowledgements

We thank Shimon Nowik for his contributions to the method’s visualizations and Grigory Sizov for his support with the practical implementation.

References

Wasi Uddin Ahmad, Sean Narenthiran, Somshubra Majumdar, Aleksander Ficek, Siddhartha Jain, Jocelyn Huang, Vahid Noroozi, and Boris Ginsburg. Opencodereasoning: Advancing data distillation for competitive coding. 2025.

Marianne Arriola, Aaron Gokaslan, Justin T Chiu, Zhihan Yang, Zhixuan Qi, Jiaqi Han, Subham Sekhar Sahoo, and Volodymyr Kuleshov. Block diffusion: Interpolating between autoregressive and diffusion language models. arXiv preprint arXiv:2503.09573, 2025.

Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, et al. Program synthesis with large language models. arXiv preprint arXiv:2108.07732, 2021.

Heli Ben-Hamu, Itai Gat, Daniel Severo, Niklas Nolte, and Brian Karrer. Accelerated sampling from masked diffusion models via entropy bounded unmasking. arXiv preprint arXiv:2505.24857, 2025.

Tianle Cai, Yuhong Li, Zhengyang Geng, Hongwu Peng, Jason D Lee, Deming Chen, and Tri Dao. Medusa: Simple llm inference acceleration framework with multiple decoding heads. arXiv preprint arXiv:2401.10774, 2024.

Charlie Chen, Sebastian Borgeaud, Geoffrey Irving, Jean-Baptiste Lespiau, Laurent Sifre, and John Jumper. Accelerating large language model decoding with speculative sampling. arXiv preprint arXiv:2302.01318, 2023.

Jacob K Christopher, Brian R Bartoldson, Tal Ben-Nun, Michael Cardei, Bhavya Kailkhura, and Ferdinando Fioretto. Speculative diffusion decoding: Accelerating language generation through diffusion, 2025. https://arxiv.org/abs/ 2408.05636.

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. ArXiv, abs/1803.05457, 2018.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems, 2021. https://arxiv.org/abs/2110.14168.

Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. FlashAttention: Fast and memory-efficient

exact attention with IO-awareness. In Advances in Neural Information Processing Systems (NeurIPS), 2022. Google Deepmind. Gemini diffusion, 2025. https://deepmind.google/models/gemini-diffusion/. Juechu Dong, Boyuan Feng, Driss Guessous, Yanbo Liang, and Horace He. Flex attention: A programming model for

generating optimized attention kernels, 2024. https://arxiv.org/abs/2412.05496. Nima Fathi, Torsten Scholak, and Pierre-André Noël. Unifying autoregressive and diffusion-based sequence generation,

2025. https://arxiv.org/abs/2504.06416. Itai Gat, Tal Remez, Neta Shaul, Felix Kreuk, Ricky TQ Chen, Gabriel Synnaeve, Yossi Adi, and Yaron Lipman. Discrete flow matching. Advances in Neural Information Processing Systems, 37:133345–133385, 2024. Marjan Ghazvininejad, Omer Levy, Yinhan Liu, and Luke Zettlemoyer. Mask-predict: Parallel decoding of conditional masked language models. arXiv preprint arXiv:1904.09324, 2019. Fabian Gloeckle, Badr Youbi Idrissi, Baptiste Rozière, David Lopez-Paz, and Gabriel Synnaeve. Better & faster large language models via multi-token prediction. arXiv preprint arXiv:2404.19737, 2024.

Shansan Gong, Ruixiang Zhang, Huangjie Zheng, Jiatao Gu, Navdeep Jaitly, Lingpeng Kong, and Yizhe Zhang. Diffucoder: Understanding and improving masked diffusion models for code generation, 2025. https://arxiv.org/ abs/2506.20639.

Gabe Guo and Stefano Ermon. Reviving any-subset autoregressive models with principled parallel sampling and speculative decoding. arXiv preprint arXiv:2504.20456, 2025.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. Proceedings of the International Conference on Learning Representations (ICLR), 2021.

Emiel Hoogeboom, Alexey A. Gritsenko, Jasmijn Bastings, Ben Poole, Rianne van den Berg, and Tim Salimans. Autoregressive diffusion models. In International Conference on Learning Representations, 2022. https://openreview.net/forum?id=Lm8T39vLDTE.

Zhanqiu Hu, Jian Meng, Yash Akhauri, Mohamed S Abdelfattah, Jae-sun Seo, Zhiru Zhang, and Udit Gupta. Accelerating diffusion language model inference via efficient kv caching and guided diffusion. arXiv preprint arXiv:2505.21467, 2025.

Chihan Huang and Hao Tang. Ctrldiff: Boosting large diffusion language models with dynamic block prediction and controllable generation, 2025. https://arxiv.org/abs/2505.14455.

Daniel Israel, Guy Van den Broeck, and Aditya Grover. Accelerating diffusion llms via adaptive parallel decoding. arXiv preprint arXiv:2506.00413, 2025.

Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando Solar-Lezama, Koushik Sen, and Ion Stoica. Livecodebench: Holistic and contamination free evaluation of large language models for code. arXiv preprint arXiv:2403.07974, 2024.

Inception Labs, Samar Khanna, Siddhant Kharbanda, Shufan Li, Harshit Varma, Eric Wang, Sawyer Birnbaum, Ziyang Luo, Yanis Miraoui, Akash Palrecha, Stefano Ermon, Aditya Grover, and Volodymyr Kuleshov. Mercury: Ultra-fast language models based on diffusion, 2025. https://arxiv.org/abs/2506.17298.

Yaniv Leviathan, Matan Kalman, and Yossi Matias. Fast inference from transformers via speculative decoding. In International Conference on Machine Learning, pages 19274–19286. PMLR, 2023.

Jeffrey Li, Alex Fang, Georgios Smyrnis, Maor Ivgi, Matt Jordan, Samir Gadre, Hritik Bansal, Etash Guha, Sedrick Keh, Kushal Arora, Saurabh Garg, Rui Xin, Niklas Muennighoff, Reinhard Heckel, Jean Mercat, Mayee Chen, Suchin Gururangan, Mitchell Wortsman, Alon Albalak, Yonatan Bitton, Marianna Nezhurina, Amro Abbas, Cheng-Yu Hsieh, Dhruba Ghosh, Josh Gardner, Maciej Kilian, Hanlin Zhang, Rulin Shao, Sarah Pratt, Sunny Sanyal, Gabriel Ilharco, Giannis Daras, Kalyani Marathe, Aaron Gokaslan, Jieyu Zhang, Khyathi Chandu, Thao Nguyen, Igor Vasiljevic, Sham Kakade, Shuran Song, Sujay Sanghavi, Fartash Faghri, Sewoong Oh, Luke Zettlemoyer, Kyle Lo, Alaaeldin El-Nouby, Hadi Pouransari, Alexander Toshev, Stephanie Wang, Dirk Groeneveld, Luca Soldaini, Pang Wei Koh, Jenia Jitsev, Thomas Kollar, Alexandros G. Dimakis, Yair Carmon, Achal Dave, Ludwig Schmidt, and Vaishaal Shankar. Datacomp-lm: In search of the next generation of training sets for language models, 2024a.

Yuhui Li, Fangyun Wei, Chao Zhang, and Hongyang Zhang. Eagle: Speculative sampling requires rethinking feature uncertainty. In International Conference on Machine Learning, pages 28935–28948. PMLR, 2024b.

Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. arXiv preprint arXiv:2305.20050, 2023.

Jiawei Liu, Chunqiu Steven Xia, Yuyao Wang, and Lingming Zhang. Is your code generated by chatGPT really correct? rigorous evaluation of large language models for code generation. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. https://openreview.net/forum?id=1qvx610Cu7.

Xiaoran Liu, Zhigeng Liu, Zengfeng Huang, Qipeng Guo, Ziwei He, and Xipeng Qiu. Longllada: Unlocking long context capabilities in diffusion llms, 2025a. https://arxiv.org/abs/2506.14429.

Zhiyuan Liu, Yicun Yang, Yaojie Zhang, Junjie Chen, Chang Zou, Qingyuan Wei, Shaobo Wang, and Linfeng Zhang. dllm-cache: Accelerating diffusion large language models with adaptive caching, 2025b. https://arxiv.org/abs/ 2506.06295.

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization, 2019. https://arxiv.org/abs/1711.05101. Aaron Lou, Chenlin Meng, and Stefano Ermon. Discrete diffusion modeling by estimating the ratios of the data

distribution. arXiv preprint arXiv:2310.16834, 2023. Xinyin Ma, Runpeng Yu, Gongfan Fang, and Xinchao Wang. dkv-cache: The cache for diffusion language models.

arXiv preprint arXiv:2505.15781, 2025. Llama 3 Team Meta. The llama 3 herd of models, 2024. https://arxiv.org/abs/2407.21783. Ivan Moshkov, Darragh Hanley, Ivan Sorokin, Shubham Toshniwal, Christof Henkel, Benedikt Schifferer, Wei Du, and

Igor Gitman. Aimo-2 winning solution: Building state-of-the-art mathematical reasoning models with openmathreasoning dataset. arXiv preprint arXiv:2504.16891, 2025.

Shen Nie, Fengqi Zhu, Zebin You, Xiaolu Zhang, Jingyang Ou, Jun Hu, Jun Zhou, Yankai Lin, Ji-Rong Wen, and Chongxuan Li. Large language diffusion models. arXiv preprint arXiv:2502.09992, 2025.

Arnaud Pannatier, Evann Courdier, and François Fleuret. σ-gpts: A new approach to autoregressive models. In Joint European Conference on Machine Learning and Knowledge Discovery in Databases, pages 143–159. Springer, 2024.

David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R. Bowman. Gpqa: A graduate-level google-proof q&a benchmark, 2023.

Subham Sekhar Sahoo, Zhihan Yang, Yash Akhauri, Johnna Liu, Deepansha Singh, Zhoujun Cheng, Zhengzhong Liu,

Eric Xing, John Thickstun, and Arash Vahdat. Esoteric language models. arXiv preprint arXiv:2506.01928, 2025. Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. Winogrande: An adversarial winograd

schema challenge at scale. arXiv preprint arXiv:1907.10641, 2019. Jiaxin Shi, Kehang Han, Zhe Wang, Arnaud Doucet, and Michalis Titsias. Simplified and generalized masked diffusion for discrete data. Advances in neural information processing systems, 37:103131–103167, 2024.

Yuxuan Song, Zheng Zhang, Cheng Luo, Pengyang Gao, Fan Xia, Hao Luo, Zheng Li, Yuehang Yang, Hongli Yu, Xingwei Qu, Yuwei Fu, Jing Su, Ge Zhang, Wenhao Huang, Mingxuan Wang, Lin Yan, Xiaoying Jia, Jingjing Liu, Wei-Ying Ma, Ya-Qin Zhang, Yonghui Wu, and Hao Zhou. Seed diffusion: A large-scale diffusion language model with high-speed inference, 2025. https://arxiv.org/abs/2508.02193.

Mitchell Stern, Noam Shazeer, and Jakob Uszkoreit. Blockwise parallel decoding for deep autoregressive models. Advances in Neural Information Processing Systems, 31, 2018.

Benigno Uria, Iain Murray, and Hugo Larochelle. A deep and tractable density estimator. In Eric P. Xing and Tony Jebara, editors, Proceedings of the 31st International Conference on Machine Learning, volume 32 of Proceedings of Machine Learning Research, pages 467–475, Bejing, China, 22–24 Jun 2014. PMLR.

Samuel Webb Williams, Andrew Waterman, and David A Patterson. Roofline: An insightful visual performance model for floating-point programs and multicore architectures. Technical report, Technical Report UCB/EECS-2008-134, EECS Department, University of . . . , 2008.

Chengyue Wu, Hao Zhang, Shuchen Xue, Zhijian Liu, Shizhe Diao, Ligeng Zhu, Ping Luo, Song Han, and Enze Xie. Fast-dllm: Training-free acceleration of diffusion llm by enabling kv cache and parallel decoding. arXiv preprint arXiv:2505.22618, 2025a.

Zirui Wu, Lin Zheng, Zhihui Xie, Jiacheng Ye, Jiahui Gao, Yansong Feng, Zhenguo Li, Victoria W., Guorui Zhou, and Lingpeng Kong. Dreamon: Diffusion language models for code infilling beyond fixed-size canvas, 2025b. https://hkunlp.github.io/blog/2025/dreamon.

Zhihui Xie, Jiacheng Ye, Lin Zheng, Jiahui Gao, Jingwei Dong, Zirui Wu, Xueliang Zhao, Shansan Gong, Xin Jiang, Zhenguo Li, and Lingpeng Kong. Dream-coder 7b, 2025. https://hkunlp.github.io/blog/2025/dream-coder.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jing Zhou, Jingren Zhou, Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang, Le Yu, Lianghao Deng, Mei Li, Mingfeng Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin, Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, and Zihan Qiu. Qwen3 technical report, 2025a. https://arxiv.org/abs/2505.09388.

Ling Yang, Ye Tian, Bowen Li, Xinchen Zhang, Ke Shen, Yunhai Tong, and Mengdi Wang. Mmada: Multimodal large diffusion language models, 2025b. https://arxiv.org/abs/2505.15809.

Zebin You, Shen Nie, Xiaolu Zhang, Jun Hu, Jun Zhou, Zhiwu Lu, Ji-Rong Wen, and Chongxuan Li. Llada-v: Large language diffusion models with visual instruction tuning. arXiv preprint arXiv:2505.16933, 2025.

Runpeng Yu, Xinyin Ma, and Xinchao Wang. Dimple: Discrete diffusion multimodal large language model with parallel decoding. arXiv preprint arXiv:2505.16990, 2025.

Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. Hellaswag: Can a machine really finish your sentence? In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, 2019.

## Appendix

### A Implementation

A.1 Method

The following PyTorch snippets highlight the key modifications for our method, focusing on the changes to a standard autoregressive framework. Code Block 7 presents the adjustments to the training loop required to learn a hybrid model. The subsequent code blocks detail our custom attention mechanism, implemented using FlexAttention (Dong et al., 2024): code block 9 for training and code block 10 for inference.

- 1 input_ids = sequence[:, :-1]
- 2
- 3 ar_label_ids = sequence[:, 1:]
- 4 parallel_label_ids = input_ids.clone()
- 5
- 6 bsz, seq_len = input_ids.shape
- 7
- 8 block_len = torch.randint(max_block_size, (1, )).item()
- 9 attention_mask = create_attention_mask_train(seq_len=seq_len, block_len=block_len)
- 10
- 11 # 1. Compute masked_input
- 12 time = torch.rand(size=(bsz, 1), device="cuda")
- 13 input_ids_mask = torch.rand(size=input_ids.shape, device=input_ids.device) > time
- 14
- 15 masked_input = torch.where(condition=input_ids_mask, input=tokenizer.mask_id, other=input_ids)
- 16
- 17 input_ids = torch.cat([input_ids, masked_input], dim=1)
- 18
- 19 # 2. Compute label_ids
- 20 parallel_label_ids[:, :-1][input_ids_mask] = -100
- 21
- 22 label_ids = torch.cat([ar_label_ids, parallel_label_ids], dim=1)
- 23
- 24 # 3. Each unique token location (in AR and parallel) should get the same positional embeddings.
- 25 positional_embeddings = get_positional_embeddings(seq_len=seq_len)
- 26 positional_embeddings = positional_embeddings.repeat(2, 1)
- 27
- 28 loss = model(input_ids, mask=attention_mask, targets=label_ids, positional_embeddings=positional_embeddings)
- 29 loss.backward()
- 30
- 31 ...

##### Figure 7 The required modifications to the training loop of an autoregressive model. We first change how the attention mask is computed, then mask a portion of the input tokens. The block-length is chosen randomly.

|Input|Target|x1 x2 x3 x4<br><br>|x1 m x3 m<br><br>|x5 x6 x7 x8<br><br>|m m x7 m<br><br>|
|---|---|---|---|---|---|
|x1 x2 x3 x4|x2 x3 x4 x5<br><br>| | | | |
|x1 m x3 m|x2 x4<br><br>| | | | |
|x5 x6 x7 x8|x6 x7 x8 x9<br><br>| | | | |
|m m x7 m|x5 x6 x8<br><br>| | | | |

##### Figure 8 Training a hybrid model for block of size k = 4. We show input tokens (left column), target tokens used for Cross-Entropy loss (second columns) and attention matrix.

- 1 from torch.nn.attention.flex_attention import BlockMask, create_block_mask
- 2
- 3 def create_attention_mask_train(seq_len: int, block_len: int) -> BlockMask:
- 4 half_seq_len: int = seq_len // 2
- 5
- 6 def mask_mod(b, h, q_idx, kv_idx):
- 7 # Top-left quadrant (x1 -> x1, standard causal)
- 8 # True if query and key are in the first half and key is before or at query.
- 9 is_in_top_left_causal = (q_idx < half_seq_len) & (kv_idx < half_seq_len) & (kv_idx <= q_idx)
- 10
- 11 # Bottom-right quadrant (xt -> xt, block attention)
- 12 # True if query and key are in the second half and belong to the same block.
- 13 q_block_xt = (q_idx - half_seq_len) // block_len
- 14 kv_block_xt = (kv_idx - half_seq_len) // block_len
- 15 is_in_bottom_right_block = (q_idx >= half_seq_len) & (kv_idx >= half_seq_len) & (q_block_xt ==

→ kv_block_xt)

- 16
- 17 # Bottom-left quadrant (xt -> x1, block causal past)
- 18 # True if query is in the second half, key is in the first half, and the queries block index is

→ strictly greater than the keys block index.

- 19 q_block_idx_bl = (q_idx - half_seq_len) // block_len
- 20 kv_block_idx_bl = kv_idx // block_len
- 21 is_in_bottom_left_block_causal_past = (q_idx >= half_seq_len) & (kv_idx < half_seq_len) &

→ (q_block_idx_bl > kv_block_idx_bl)

- 22
- 23 return is_in_top_left_causal | is_in_bottom_right_block | is_in_bottom_left_block_causal_past
- 24 return create_block_mask(mask_mod, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len)

##### Figure 9 FlexAttention implementation of the attention mask at training time (see figure 8 for a visual representation).

- 1 from torch.nn.attention.flex_attention import BlockMask, create_block_mask
- 2
- 3 def create_attention_mask_inference(causal_point: int) -> BlockMask:
- 4 # causal_point is the index of the first token in the prediction block
- 5 def mask_mod(b, h, q_idx, kv_idx):
- 6 is_causal = (kv_idx <= q_idx)
- 7 is_past_causal_point = (causal_point <= q_idx)
- 8 return is_causal | is_past_causal_point
- 9 return create_block_mask(mask_mod, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len)

##### Figure 10 FlexAttention implementation of the attention mask at inference time. The attention is causal up until the point where the prediction block begins. The tokens in the prediction block attend to all other tokens in the sequence.

- A.2 Roofline model

- 1 def gemm_roofline(m: int, n: int, k: int) -> tuple[float, float]:
- 2 "Returns the runtime (in s) and flops of doing a C=A@B with A/B of shape [m, k]/[k, n] "
- 3 total_IO = (
- 4 m * k * BYTES + # read A
- 5 n * k * BYTES + # read B
- 6 m * n * BYTES # write C
- 7 )
- 8 total_flops = 2 * m * n * k
- 9 return max(total_IO / MEMORY_BANDWIDTH, total_flops / PEAK_FLOPS), total_flops
- 10
- 11 # Attention roofline analysis
- 12 def calculate_fused_attention_roofline(block_size: int, kv_length: int, batch_size: int) -> float:
- 13 num_tokens = block_size * batch_size
- 14 # Memory in Q: (batch_size, block_size, NUM_HEADS, HEAD_DIM)
- 15 bytes_Q = num_tokens * NUM_HEADS * HEAD_DIM * BYTES
- 16 # Memory in K: (batch_size, kv_length, NUM_KV_HEADS, HEAD_DIM)
- 17 bytes_K = batch_size * kv_length * NUM_KV_HEADS * HEAD_DIM * BYTES_PER_KV_ELEMENT
- 18 # Flops P=QK^T
- 19 _, flops_QKt = gemm_roofline(num_tokens, kv_length, NUM_HEADS * HEAD_DIM)
- 20 # Softmax memory transfer is ignored as it's not read or stored in memory, flops negliglbe
- 21 bytes_softmax = 0
- 22 # Memory in V: (batch_size, kv_length, NUM_KV_HEADS, HEAD_DIM)
- 23 bytes_V = batch_size * kv_length * NUM_KV_HEADS * HEAD_DIM * BYTES_PER_KV_ELEMENT
- 24 # Flops PV
- 25 _, flops_PV = gemm_roofline(num_tokens, kv_length, NUM_HEADS * HEAD_DIM)
- 26 # Memory out PV: (batch_size, block_size, NUM_HEADS, HEAD_DIM)
- 27 bytes_PV = num_tokens * NUM_HEADS * HEAD_DIM * BYTES
- 28
- 29 total_memory = bytes_Q + bytes_K + bytes_V + bytes_PV + bytes_softmax
- 30 total_flops = flops_QKt + flops_PV
- 31 return max(total_memory / MEMORY_BANDWIDTH, total_flops / PEAK_FLOPS_ATTENTION)
- 32
- 33
- 34 # FFN roofline analysis
- 35 def calculate_linear_layers_roofline(block_size: int, batch_size: int) -> float:
- 36 num_tokens = block_size * batch_size
- 37 # FFN: Up-projection
- 38 total_time = gemm_roofline(num_tokens, HIDDEN_DIM, HIDDEN_DIM * FFN_DIM)[0]
- 39 # FFN: Down-projection
- 40 total_time += gemm_roofline(num_tokens, HIDDEN_DIM * FFN_DIM, HIDDEN_DIM)[0]
- 41 # Attention linear layers
- 42 total_time += gemm_roofline(num_tokens, HIDDEN_DIM, NUM_HEADS * HEAD_DIM)[0] # Q-proj
- 43 total_time += gemm_roofline(num_tokens, HIDDEN_DIM, NUM_HEADS * NUM_KV_HEADS)[0] # K-proj
- 44 total_time += gemm_roofline(num_tokens, HIDDEN_DIM, NUM_HEADS * NUM_KV_HEADS)[0] # V-proj
- 45 total_time += gemm_roofline(num_tokens, NUM_HEADS * HEAD_DIM, HIDDEN_DIM)[0] # out-proj
- 46 return total_time

##### Figure 11 Transformer roofline analysis.

### B Additional experiments

- B.1 Sampling algorithm

In all our experiments we used EB-Sampler in its simplest form with entropy error proxy (see equation 8). In this section we ablate on two other sampling variants: (i) Factor parallel decoding (Wu et al., 2025a); and (ii) EB-Sampler with confidence error proxy (Ben-Hamu et al., 2025). For completeness, let us briefly describe the two.

Recently, Wu et al. (2025a) proposed the Factor parallel decoding approach for sampling. With a similar motivation as the EB-Sampler, the Factor method proposes an adaptive parallel decoding algorithm that in high confidence regimes is equivalent to greedy decoding. At each step, given we already decoded the tokens in indices J ⊂ I and are left with masks M = I \ J , compute the confidence for each masked index ci = maxx

i∈V p(xi|x<t,xJ ),i ∈ M, sort the masked indices by confidence i1,i2,...,i|M| and unmask the n first tokens. Where n is the largest such that (n + 1)cn < f and f is a predefined threshold hyperparameter. In figure 12 we compare the performance of the EB-Sampler with entropy error proxy to the factor method on the 3B SFT models from section 3.2. EB-Sampler with entropy error proxy compares favorably to the Factor method on HumanEval at all training budgets, where on MBPP and GSM8K both approaches perform similarly when models are trained for more steps.

HumanEval MBPP GSM8K

60

60

40

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

| |
|---|

55

55

| | | |
|---|---|---|

| | | |
|---|---|---|

| |
|---|

| |
|---|

35

| | |
|---|---|
| | |

| |
|---|

| |
|---|

| |
|---|

50

50

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Accuracy(%)

Accuracy(%)

Accuracy(%)

30

| | | |
|---|---|---|

| | | |
|---|---|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

45

45

25

| |
|---|

| |
|---|

40

40

| |
|---|

| |
|---|

| | |
|---|---|
| | |

| |
|---|

| |
|---|

| |
|---|

20

| | | |
|---|---|---|

| | | |
|---|---|---|

Training iterations

| |
|---|

| |
|---|

35

35

8.5k

17k

34k

15

EB

Factor

30

30

200 400 600 800 1000 Mean NFE

200 400 600 800 1000 Mean NFE

200 400 600 800 1000 Mean NFE

##### Figure 12 3B model SFT training with varying number of training steps. Factor vs EB-Sampler with entropy error proxy.

Interestingly, the Factor method is more closely related to the EB-Sampler instance that uses confidence as an error proxy. In each step this instance sorts the masked tokens, M, by confidence, similarly to the Factor method, but chooses how many to unmask according to the entropy bound on the mutual information, finding the largest s such that:

s

|x<t,xJ )) ≤ γ, (18)

j|x<t,xJ )) − max j′≤s

H(p(xi

H(p(xi

j′

j=1

#### Figure 13 shows the performance of the EB-Sampler with confidence error proxy compared to the Factor method at different γ and f values respectively, empirically demonstrating the similarity of the two approaches. Notably, figure 12 and figure 13 also show that entropy as an error proxy compares favorably for the tested models. The models used in this experiment are again the 3B SFT models from section 3.2.

###### HumanEval MBPP GSM8K

| |
|---|

| |
|---|

55

55

| | | |
|---|---|---|

| | | |
|---|---|---|

35

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

50

50

| |
|---|

| |
|---|

30

Accuracy(%)

Accuracy(%)

Accuracy(%)

| | | |
|---|---|---|

| | | |
|---|---|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

45

45

25

| |
|---|

| |
|---|

| |
|---|

| |
|---|

40

40

| |
|---|

| |
|---|

20

| |
|---|

| |
|---|

| |
|---|

| | | |
|---|---|---|

| | | |
|---|---|---|

| |
|---|

| |
|---|

Training iterations

35

35

8.5k

17k

34k

15

EB

Factor

30

30

| |
|---|

200 400 600 800 1000 Mean NFE

200 400 600 800 1000 Mean NFE

200 400 600 800 1000 Mean NFE

##### Figure 13 3B model SFT training with varying number of training steps. Factor vs EB-Sampler with confidence error proxy.

