## Randomized Autoregressive Visual Generation

Qihang Yu Ju He Xueqing Deng Xiaohui Shen Liang-Chieh Chen ByteDance https://yucornetto.github.io/projects/rar.html

# arXiv:2411.00776v1[cs.CV]1Nov2024

### Abstract

This paper presents Randomized AutoRegressive modeling (RAR) for visual generation, which sets a new state-of-theart performance on the image generation task while maintaining full compatibility with language modeling frameworks. The proposed RAR is simple: during a standard autoregressive training process with a next-token prediction objective, the input sequence—typically ordered in raster form—is randomly permuted into different factorization orders with a probability r, where r starts at 1 and linearly decays to 0 over the course of training. This annealing training strategy enables the model to learn to maximize the expected likelihood over all factorization orders and thus effectively improve the model’s capability of modeling bidirectional contexts. Importantly, RAR preserves the integrity of the autoregressive modeling framework, ensuring full compatibility with language modeling while significantly improving performance in image generation. On the ImageNet-256 benchmark, RAR achieves an FID score of 1.48, not only surpassing prior state-of-the-art autoregressive image generators but also outperforming leading diffusion-based and masked transformer-based methods. Code and models will be made available at https: //github.com/bytedance/1d-tokenizer.

### 1. Introduction

AutoRegressive (AR) models have driven remarkable advancements across both natural language processing and computer vision tasks in recent years. In language modeling, they serve as the fundamental framework for Large Language Models (LLMs) such as GPT [43], Llama [59, 60], and Gemini [57], along with other state-of-the-art models [1, 67]. In the realm of computer vision, autoregressive models1 have also shown substantial potential, delivering

1While MaskGIT-style models [10] could be classified as “generalized autoregressive models” as defined in [36], in this paper, we primarily use the term “autoregressive” to refer to GPT-style models [22, 52, 69], which are characterized by causal attention, next-token prediction, and operate without the need for mask tokens as placeholders.

TamingTrans

5.0

4.5

FID(lowerisbetter)

4.0

LlamaGen

3.5

VIM

3.0

Open-MAGVIT2

2.5

2.0

RAR (ours)

1.5

200 400 600 800 1000 1200 1400 1600

model size (M)

Figure 1. Comparison among different language modeling compatible autoregressive (AR) image generators. The proposed RAR demonstrates significant improvements over previous AR methods. RAR-B, with only 261M parameters, achieves an FID score of 1.95, outperforming both LlamaGen-XXL (1.4B parameters) and Open-MAGVIT2-XL (1.5B parameters).

competitive performance in image generation tasks [22, 35, 39, 52, 69, 70] to diffusion models [6, 18, 36, 45] or nonautoregressive transformers [10, 65, 71–73]. More importantly, autoregressive modeling is emerging as a promising pathway toward unified models across multiple modalities and tasks [5, 9, 14, 55, 56, 66].

Despite the dominance of autoregressive models in language modeling, they often yield suboptimal performance in comparison to diffusion models or non-autoregressive transformers in visual generation tasks [39, 52]. This discrepancy can be attributed to the inherent differences between text and visual signals. Text is highly compact and semantically meaningful, while visual data tends to be more low-level and redundant [29, 73], making bidirectional context modeling more critical. For instance, several studies [7, 21, 36] have demonstrated that causal attention applied to image tokens leads to inferior performance compared to bidirectional attention in vision tasks.

To address this, recent works [36, 58] have attempted to reintroduce bidirectional attention by redesigning the autoregressive formulation, achieving state-of-the-art re-

autoregressive transformer

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

|C C C C<br><br>|1 2 3 4 5<br><br>|
|---|---|
| |3 1 5 4 2 1 5 3 2 4<br><br>4 2 3 5 1<br><br><br>|

class tokensclass tokens

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

permuted tokens

permute with probability 𝑟

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

| | | |
|---|---|---|
|C C C C<br><br>|1 2 3 4 5 1 2 3 4 5 1 2 3 4 5 1 2 3 4 5<br><br>| |

input tokens

- Figure 2. Overview of the proposed Randomized AutoRegressive (RAR) model, which is fully compatible with language modeling frameworks. Left: RAR introduces a randomness annealing training strategy to enhance the model’s ability to learn bidirectional contexts. During training, the input sequence is randomly permuted with a probability r, which starts at 1 (fully random permutations) and linearly decreases to 0, transitioning the model to a fixed scan order, such as raster scan, by the end of training. Right: Randomly selected images generated by RAR, trained on ImageNet.

ing. At the same time, it significantly improves the generation quality of autoregressive models at no additional cost. On the ImageNet-256 benchmark [16], RAR achieves an FID score of 1.48, substantially outperforming previous state-of-the-art autoregressive image generators, as illustrated in Fig. 1. By addressing the limitations of unidirectional context modeling, RAR represents a critical step towards autoregressive visual generation and opens up new possibilities for further advancements in the field.

sults in image generation. However, these approaches often deviate from the traditional autoregressive paradigm. For example, VAR [58] shifts from next-token prediction to next-scale prediction, enabling bidirectional attention within each scale, and MAR [36] generalizes MaskGITstyle framework [10] to the autoregressive definition, which naturally introduces back the bidirectional attention. While effective, these modifications complicate their integration into universal transformer architectures that aim to unify different modalities, which proves to work well with conventional autoregressive modeling [55, 56].

### 2. Related Work

In this paper, we aim to enhance the generation quality of autoregressive image models while preserving the core autoregressive structure, maintaining compatibility with language modeling frameworks. Specifically, we enable bidirectional context learning within an autoregressive transformer by maximizing the expected likelihood over all possible factorization order. In this way, all tokens will be trained and predicted under all possible contexts, facilitating learning bidirectional representation. Moreover, we introduce a permutation probability r, which controls the ratio of training data between a random factorization order and the standard raster order. Initially, r is set to 1 (fully random factorization) and it linearly decays to 0 over the course of training, gradually reverting the model to the raster order commonly used by other autoregressive image generators.

Autoregressive Language Modeling. The advent of autoregressive language models [1–4, 9, 13, 20, 43, 46, 47, 57, 59, 60, 67] has paved a promising path toward generalpurpose AI systems. At the core of these models is a simple yet powerful next-token prediction paradigm, where the objective is to predict the next word or token in a sequence based on preceding inputs. This approach has demonstrated both scalability, as evidenced by scaling laws, and versatility through zero-shot generalization, enabling explorations beyond traditional language tasks to diverse modalities.

Autoregressive Visual Modeling. Pioneering research [12, 27, 44, 62, 63] in autoregressive visual modeling has focused on representing images as sequences of pixels. Nevertheless, inspired by advancements in autoregressive language modeling, a subsequent wave of studies has transitioned to modeling images as sequences of discrete-valued tokens [22, 48, 49, 64, 69], resulting in notable improvements in performance. This direction has been further explored through efforts [39, 52] aimed at enhancing tokeniza-

To this end, we present a simple, effective, and scalable autoregressive model training paradiam named Randomized AutoRegressive modeling (RAR). RAR retains the original autoregressive model architecture and formulation, ensuring full compatibility with language model-

tion quality and leveraging modern autoregressive architectures initially developed for language tasks. However, all of these works strictly adhere to a raster-scan order for processing pixels or tokens, resulting in a unidirectional information flow that is sub-optimal for visual modeling. In this work, we instead explore learning across all possible factorization orders to enhance bidirectional context learning while retaining the core autoregressive framework.

Other Visual Generation Models. In addition to autoregressive visual modeling, there have been numerous efforts in exploring other formats of visual generation models, including generative adversarial networks (GANs) [8, 26, 32], diffusion models [18, 23, 31, 37, 45, 50], masked transformers [10, 11, 65, 71, 73], scale-wise autoregressive modeling (VAR) [41, 54, 58, 75], and masked autoregressive modeling with diffusion loss (MAR) [24, 36]. It is worth noting that MAR [36] also experimented a random order based AR framework similar to the proposed RAR. However, as indicated in our experiments (see Sec. 4.2), simply replacing the raster order with random order only brings marginal improvement, coinciding the observation in [36]. This further demonstrates the importance on the randomness annealing strategy in RAR, leading to a substantial improvement for the AR image generators.

### 3. Method

In this section, we first provide an overview of autoregressive modeling in Sec. 3.1, followed by our proposed Randomized AutoRegressive modeling (RAR) in Sec. 3.2.

#### 3.1. Background

We provide a brief overview of autoregressive modeling with a next-token prediction objective. Given a discrete token sequence x = [x1,x2,··· ,xT], the goal of autoregressive modeling is to maximize the likelihood of the sequence under a forward autoregressive factorization. Specifically, the objective is to maximize the joint probability of predicting the current token xt based on all preceding tokens [x1,x2,··· ,xt−1], ∀t = 1,··· ,T:

T

pθ(xt|x1,x2,··· ,xt−1), (1)

max

pθ(x) =

θ

t=1

where pθ denotes a token distribution predictor with a model parameterized by θ.

As shown in the equation, each token xt at position t is conditioned solely on the preceding tokens, which limits context modeling to a unidirectional manner. This contrasts with methods such as masked transformer [10, 65, 71, 72] and diffusion models [31, 37, 45, 50], which can leverage bidirectional context at the training time. Additionally, while natural language has an inherent sequential order (left-to-right in most languages), image data lacks a clear,

predefined order for processing tokens. Among the possible orders for image generation, the row-major order (i.e., raster scan) is the most widely adopted and has demonstrated superior performance compared to other alternatives [22].

#### 3.2. RAR: Randomized AutoRegressive Modeling

Visual signals inherently exhibit bidirectional correlations, making effective global context modeling essential. However, conventional autoregressive models rely on causal attention masking, which enforces a unidirectional dependency on the token sequence, contradicting the nature of visual data, as noted in prior works [7, 21, 36], where bidirectional attention works significantly better than causal attention for visual modality. Furthermore, there is no universally “correct” way to arrange image tokens into a causal sequence. While the widely adopted raster order has achieved some success, it introduces biases in the autoregressive training process. For instance, each token is conditioned solely on the preceding tokens in the scanning order, restricting the model’s ability to learn dependencies from other directions.

To address these challenges, we propose a randomized autoregressive modeling approach that incorporates optimization objective with bidirectional context:

T

pθ(xt|x1,··· ,xt−1,xt+1,··· ,xT).

max

pθ(x) =

θ

t=1

(2)

Unlike BERT-style [17] or MaskGIT-style [10] methods, our method follows the permuted objective approach [61, 68], where the model is trained in an autoregressive manner across all possible factorization orders. This enables the model to gather bidirectional context while preserving the autoregressive framework in expectation. Formally, we have:

max

θ

pθ(x) = Eτ∼S

T

T

t|xτ

pθ(xτ

<t

t=1

) , (3)

where ST denotes the set of all possible permutations of the index sequence [1,2,··· ,T], and τ represents a randomly sampled permutation from ST. The notation τt refers to the t-th element in the permuted sequence, and τ<t represents all preceding positions to τt. Since the model parameters θ are shared across all sampled factorization orders, each token xt is exposed to every possible context and learns relationships with every other token xi ∀i ̸= t, during training. This allows the model to effectively capture bidirectional context while preserving the integrity of the autoregressive formulation.

Although simple, this modification significantly improves image generation performance, highlighting the power of bidirectional context in improving autoregressive

possible predictions

autoregressive transformer

- 4 5

- 5 4

C 2 3 1 ?

𝑝 𝑝 𝑝 𝑝 𝑝

C 2 3 1 5 4

|x same feature for the next token prediction|
|---|

add target-aware position embedding

- (b) w/o target-aware position embedding
- (c) w/ target-aware position embedding

- C 1 2 3 4 5

- C 2 3 1 5 4

permutation

C 2 3 1 ? 5 4 𝑝 𝑝 𝑝 𝑝

class condition

|√ feature is aware of which next token to be predicted|
|---|

[Figure 19]

tokenizer

|tokens w/ pos embed target-aware pos embed next token to predict unknown future tokens<br><br>|
|---|

(a) how does RAR work w/ target-aware position embedding

- Figure 3. Illustration of the target-aware positional embedding. Subfigure (a) shows the training process of the proposed Randomized AutoRegressive (RAR) model, along with the target-aware position embedding. Following Vision Transformer [19], images are tokenized into patches with original position embeddings (blue tokens). The token sequence is then randomly permuted, with the target-aware positional embeddings (green tokens) added to guide the model. Subfigures (b) and (c) highlight the importance of the target-aware positional embedding: (b) demonstrates a failure case where both permuted sequences yield identical prediction logits, while (c) shows that the target-aware positional embedding correctly guides the model to predict the next token accurately.

image generator capability. Our findings align with those observed in autoregressive training for language modeling in NLP [9, 17, 61, 68] as well.

identical features and thus identical prediction logits, even though they correspond to different ground-truth labels (i.e., pθ(xτ

) is the same for both permutations τa and τb). This problem, in a general randomized autoregressive training process and beyond this specific example, can happen for all token locations except the last one (since the last token does not need to predict next token). To address this issue, we introduce an additional set of positional embeddings, which we refer to as target-aware positional embeddings. These embeddings encode information about which token is being predicted next.

T−1|xτ

,··· ,xτ

##### ,xτ

T −2

1

2

Discussion. While the permutation objective allows for bidirectional context learning within the autoregressive framework in expectation, it remains challenging to fully capture “global context” during the generation process. This is because there are always some tokens generated before others, without having access to the full global context. This limitation is not unique to autoregressive methods [22, 52] but also present in non-autoregressive models [10]. Techniques such as resampling or refinement [28, 42] may help address this issue by ensuring that every token is generated with sufficient context. However, such designs may complicate the system; thus, exploring such solutions lies beyond the scope of this paper and is left for future work.

Formally, we define a set of target-aware positional embeddings pta = [p1,p2,··· ,pT]. The positional embedding corresponding to the next token is added to the current token embedding, resulting in a target-aware token embedding xˆτ:

,··· ,xτ

##### xˆτ = xτ+pτ = [xτ

##### +pτ

##### ,xτ

##### +pτ

+pτ

,xτ

],

T −1

1

2

2

3

T

T

(4) where xτ and pτ are permuted tokens for x and pta w.r.t. to the permutation τ, respectively. By associating the target token’s positional embedding with the next-token prediction, each token prediction is aware of the target token’s index, alleviating the potential confusion in permuted objective.

Target-aware Positional Embedding. One limitation of the permuted training objective is that standard positional embeddings may fail in certain scenarios. For instance, consider two different permutations: τa = [1,2,··· ,T − 2,T − 1,T] and τb = [1,2,··· ,T − 2,T,T − 1] (i.e., only the last two tokens’ positions are swapped). When predicting the second to last token, both permutations will yield

Notably, we omit the target-aware positional embedding for the final token xτ

, as it does not participate in the loss

T

computation and has no prediction target. A visual illustration of this concept is provided in Fig. 3. It is also noteworthy that the target-aware positional embedding can be merged with original positional embedding after the training is finished, because our method anneals to a fixed raster scan in the end, and thus leads to no increase on the parameters or computation during inference.

Randomness Annealing. While the proposed randomized autoregressive training with permutation enables the model to capture bidirectional context within a unidirectional framework, it may introduce sub-optimal behavior for visual generation due to two main factors: (1) The sheer number of possible permutations is vast, potentially causing the model to focus on learning how to handle the different permutation orders rather than improving generation quality. For example, for a token sequence of length 256, the number of possible permutations is 256! > 10506, which can overwhelm the model and reduce training efficiency. (2) Although images can be processed in arbitrary orders, certain scan orders tend to outperform others. For instance, [22] evaluated six different scan orders (row-major, spiral in, spiral out, z-curve, subsample, and alternate) and found that row-major (i.e., raster order) consistently performed the best, a result that has made it the most widely used order for visual generation.

To address these issues, we propose Randomness Annealing, a strategy designed to balance the randomness of permutations with the known effectiveness of the raster order. This method introduces a single parameter, r, which controls the probability of using a random permutation versus the raster order. At the start of training, r = 1, meaning that the model exclusively uses random permutations. Over the course of training, r linearly decays to 0, transitioning the model to the raster order by the end of training. Specifically, we define a training schedule for r, controlled by two hyper-parameters start and end indicating the training epoch when r starts to anneal and when the annealing ends. Formally, we have:

 

1.0, if epoch < start,

(5)

- 0.0, if epoch > end,
- 1.0 − epochend−−startstart, otherwise,

r =



where epoch is the current training epoch. We will ablate the hyper-parameters start and end in the experiments.

The schedule allows the model to initially explore the diverse random permutations for better bidirectional representation learning, and ultimately converge to the more effective row-major scan order for better visual generation quality, as is used by other typical autoregressive methods [22]. It is worth noting that this strategy not only improves generation performance but also maintains compatibility with the standard scan order used in previous works.

|model|depth width mlp heads #params|
|---|---|
|RAR-B RAR-L RAR-XL RAR-XXL|24 768 3072 16 261M 24 1024 4096 16 461M 32 1280 5120 16 955M 40 1408 6144 16 1499M<br><br>|

Table 1. Architecture configurations of RAR. We follow prior works scaling up ViT [19, 74] for different configurations.

### 4. Experimental Results

In this section, we outline the implementation details of our method in Sec. 4.1. Next, we present ablation studies on key design choices in Sec. 4.2. The main results are discussed in Sec. 4.3, followed by scaling study and visualizations.

#### 4.1. Implementation Details

We implement the RAR on top of language modeling autoregressive framework with minimal changes.

VQ Tokenizer. Following prior works [10, 22] which use a VQ tokenizer to tokenize the input images into discrete token sequences, we use the MaskGIT-VQGAN [10] with the official weight trained on ImageNet. This tokenizer is a purely CNN-based tokenizer which tokenizes a 256 × 256 image into 256 discrete tokens (i.e., downsampling factor 16) with a codebook size (i.e., vocabulary size) 1024.

Autoregressive Transformer. We use vision transformers [19] of different model configurations [74] including RAR-S (133M), RAR-B (261M), RAR-L (461M), RARXL (955M), and RAR-XXL (1499M). For all of these model variants, we apply causal attention masking in the self-attention module and QK LayerNorm [15] to stabilize the large-scale model training. We use plain ViT for all ablation studies to speed up the experiments, and we enhance the model with adaLN [45] for final models. The detailed architecture configuration and model size are available at Tab. 1.

Positional Embedding. We use learnable embeddings for both original positional embedding in ViT and target-aware positional embedding. Notably, as our model anneals to raster order-based autoregressive image generation after the training is finished, the two positional embeddings can be combined into one, making it identical to a conventional autoregressive image generator.

Dataset. We train our model on ImageNet-1K [16] training set, which contains 1,281,167 training images across 1000 object classes. We pre-tokenize the whole training set with MaskGIT-VQGAN tokenizer [10] to speed up the training. For ablation studies, we pre-tokenize the dataset with only center crop and horizontal flipping augmentation, while we further enhance the diversity in pretokenized datasets with ten-crop transformation [52, 53] for final models.

Training Protocols. We use the same training hyper-

|start epoch end epoch|FID↓ IS↑ Pre.↑ Rec.↑|
|---|---|
|0 0† 0 100 0 200 0 300 0 400<br><br>100 100 100 200 100 300 100 400 200 200 200 300 200 400 300 300 300 400 400 400‡<br><br>|3.08 245.3 0.85 0.52 2.68 237.3 0.84 0.54 2.41 251.5 0.84 0.54<br><br>2.40 258.4 0.84 0.54 2.43 265.3 0.84 0.53 2.48 247.5 0.84 0.54 2.28 253.1 0.83 0.55 2.33 258.4 0.83 0.54 2.39 266.5 0.84 0.54 2.39 259.7 0.84 0.54 2.18 269.7 0.83 0.55 2.55 241.6 0.84 0.54<br><br>2.41 269.1 0.84 0.53<br><br><br>2.74 236.4 0.83 0.54<br>3.01 305.6 0.84 0.52<br>|

Table 2. Different start and end epochs for randomness annealing, with a total of 400 training epochs and model size RAR-L. The final setting is labeled in gray. †: When start epoch and end epoch are both 0 (1st row), the training reverts to a standard raster order training. ‡: When start epoch and end epoch are both 400 (last row), the training becomes a purely random order training. After training is finished, all results are obtained with raster order sampling, except for the purely random order training (i.e., last row), where we also randomly sample the scan order following [36], which otherwise could not produce a reasonable result.

parameters for all model variants. The model is trained with batch size 2048 for 400 epochs (250k steps). The learning rate will be linearly increased from 0 to 4×10−4 at the first 100 epochs (warm-up), then it will be gradually decayed to 1 × 10−5 following a cosine decay schedule. We use AdamW [33, 38] optimizer with beta1 0.9, beta2 0.96, and weight decay 0.03. We perform gradient clipping with maximum gradient norm 1.0. During training, the class condition will be dropped at a probability 0.1. The training setting remain the same for both ablation studies and main results across all RAR model variants.

Sampling Protocols. We sample 50000 images for FID computation using the evaluation code from [18]. We do not use any top-k or top-p based filtering techniques. We also follow prior arts [11, 25, 73] to use classifier-free guidance [30]. In ablation study, we use a simpler linear guidance schedule [11] and for final models we use the improved power-cosine guidance schedule [25]. The final detailed hyper-parameters for each model variant can be found in appendix.

#### 4.2. Ablation Studies

We study different configurations for RAR, including the randomness annealing strategy and scan orders that RAR converges to.

Randomness Annealing Strategy. In Tab. 2 we compare different randomness annealing strategies. We adopt a linear decaying schedule and focus on when should the

|scan order<br><br>|FID↓ IS↑ Precision↑ Recall↑|
|---|---|
|row-major spiral in spiral out z-curve subsample alternate<br><br>|2.18 269.7 0.83 0.55 2.50 256.1 0.84 0.54 2.46 256.6 0.84 0.54 2.29 262.7 0.83 0.55 2.39 258.0 0.84 0.54 2.48 270.9 0.84 0.53<br><br>|

Table 3. Effect of different scan orders RAR-L converges to. We mainly consider 6 different scan orders (row major, spiral in, spiral out, z-curve, subsample, alternate) as studied in [22]. Our default setting is marked in gray. A visual illustration of different scan orders are available in the appendix.

randomization annealing starts and ends by changing two hyper-parameters start and end, as defined in Eq. (5). For a training lasting for 400 epochs, we enumerate all possible combinations for every 100 epochs. For example, when start = 200 and end = 300, the model is trained with random permutations from 0 to 200 epochs and raster order from 300 to 400 epochs. During 200 to 300 epoch, the model is trained via random permutation with probability r and raster order with probability 1−r, where r is computed as in Eq. (5). It is noteworthy that when start = end = 0, the model is trained with purely raster order, i.e., the standard autoregressive training. When start = end = 400, the model is always trained with randomly permuted input sequence. Both cases are important baselines of the proposed randomness annealing, and they achieve FID scores of 3.08 and 3.01, respectively. Interestingly, we observe all other variants achieve substantial improvement over these two baselines. For example, even simply replacing the first 100 epochs of raster order with random permutation, it (i.e., start = 100 and end = 100) improves the FID to 2.48 by 0.6. Besides, we also note that the model prefers to keep some beginning epochs for pure random permutation training and some last epochs for better adapting to raster scan order, which usually leads to a better performance compared to other variants. All the results demonstrate that adding randomized autoregressive training with a permuted objective is beneficial to the autoregressive visual generator and leads to a boosted FID score, thanks to the improved bidirectional representation learning process.

Additionally, among all variants, we found that the case, where start = 200 and end = 300, works the best, which improves the baseline (purely raster order) FID from 3.08 to 2.18. This strategy allocates slightly more computes on the training with random permutation order, and focuses on the purely raster order for the last 100 epochs. Therefore, we default to adopt this annealing strategy for all RAR models. Different Scan Orders Besides Raster. Although rowmajor order (i.e., raster scan) has been the de facto scan order in the visual generation, there lacks a systematic study on how good it is compared to other scan orders. We

note that the work [22] conducted a similar study 4 years ago. However, it is worth re-examining the conclusion considering the significant progress generative models have achieved in recent years. Specifically, we consider 6 different scan orders (row-major, spiral in, spiral out, z-curve, subsample, and alternative) following [22] that RAR may converge to. Instead of reporting the training loss and validation loss as the comparison metric [22], we directly evaluate their generation performance. The results are summarized in Tab. 3. Interestingly, we observe that all variants achieve a reasonably good score, which indicates that RAR is capable of handling different scan orders. Considering that the row-major (raster scan) still demonstrates advantages over the other scan orders, we thus use the raster scan order for all final RAR models.

#### 4.3. Main Results

We report RAR results against state-of-the-art image generators on ImageNet-1K 256 × 256 benchmark [16].

As shown in Tab. 4, RAR achieves significantly better performance compared to previous AR image generators. Specifically, the most compact RAR-B with 261M parameters only, achieves an FID score 1.95, already significantly outperforming current state-of-the-art AR image generators LlamaGen-3B-384 (3.1B, FID 2.18, crop size 384) [52] and Open-MAGVIT2-XL (1.5B, FID 2.33) [39], while using 91% and 81% fewer model parameters respectively. It also surpasses the widely used diffusion models such as DiTXL/2 (FID 1.95 vs. 2.27) and SiT-XL (FID 1.95 vs. 2.06) while only using 39% model parameters compared to them.

In Tab. 4, we further explore RAR at different model sizes (from 261M to 1.5B), where we observe strong scalability behavior with consistent performance improvement

- as model size scales up. Notably, the largest variant RARXXL sets a new state-of-the-art result on ImageNet benchmark, with an FID score 1.48. When compared to the other two recent methods VAR [58] and MAR [36], both of which attempt to amend AR formulation for better visual generation quality, RAR not only demonstrates a superior performance (FID 1.48 from RAR vs. 1.73 from VAR and 1.55 from MAR), but also keeps the whole framework compatible with language modeling and thus is more friendly for adapting the mature optimization and speed-up techniques for large language models to visual generation [52].

Moreover, RAR demonstrates superior performance to state-of-the-art visual generators in different frameworks. It performs better against the leading autoregressive models, diffusion models and masked transformer models, surpassing LlamaGen-3B-384 [52], MDTv2-XL/2 [25] and MaskBit [65] respectively (FID 1.48 from RAR vs. 2.18 from LlamaGen, 1.58 from MDTv2, and 1.52 from MaskBit). To the best of our knowledge, this is the first time that the language modeling style autoregressive visual

|tokenizer|type generator #params FID↓ IS↑ Pre.↑ Rec.↑|
|---|---|
|VQ [50]<br><br>VAE [50]<br><br>VAE [51]<br>|Diff. LDM-8 [50] 258M 7.76 209.5 0.84 0.35 Diff. LDM-4 [50] 400M 3.60 247.7 0.87 0.48<br><br>Diff.<br><br>UViT-L/2 [6] 287M 3.40 219.9 0.83 0.52 UViT-H/2 [6] 501M 2.29 263.9 0.82 0.57 DiT-L/2 [45] 458M 5.02 167.2 0.75 0.57<br><br>DiT-XL/2 [45] 675M 2.27 278.2 0.83 0.57<br><br>SiT-XL [40] 675M 2.06 270.3 0.82 0.59 DiMR-XL/2R [37] 505M 1.70 289.0 0.79 0.63 MDTv2-XL/2 [25] 676M 1.58 314.7 0.79 0.65<br><br>|
|VQ [10] VQ [73] VQ [72] VQ [65]<br><br>|Mask. MaskGIT [10] 177M 6.18 182.1 - Mask. TiTok-S-128 [73] 287M 1.97 281.8 - Mask. MAGVIT-v2 [72] 307M 1.78 319.4 - Mask. MaskBit [65] 305M 1.52 328.6 - -|
|VAE [36]|MAR<br><br>MAR-B [36] 208M 2.31 281.7 0.82 0.57 MAR-L [36] 479M 1.78 296.0 0.81 0.60 MAR-H [36] 943M 1.55 303.7 0.81 0.62<br><br>|
|VQ [58]<br><br>|VAR VAR-d30 [58]<br><br>2.0B 1.92 323.1 0.82 0.59 VAR-d30-re [58] 2.0B 1.73 350.2 0.82 0.60|
|VQ [22]<br><br>VQ [69]<br><br>VQ [39]<br><br>VQ [52]|AR GPT2 [22]<br><br>1.4B 15.78 74.3 - GPT2-re [22] 1.4B 5.20 280.3 - -<br><br>AR VIM-L [69]<br><br>1.7B 4.17 175.1 - VIM-L-re [69] 1.7B 3.04 227.4 - -<br><br>AR<br><br>Open-MAGVIT2-B [39] 343M 3.08 258.3 0.85 0.51 Open-MAGVIT2-L [39] 804M 2.51 271.7 0.84 0.54<br><br>Open-MAGVIT2-XL [39] 1.5B 2.33 271.8 0.84 0.54<br><br>AR<br><br>LlamaGen-L [52] 343M 3.80 248.3 0.83 0.51 LlamaGen-XL [52] 775M 3.39 227.1 0.81 0.54 LlamaGen-XXL [52] 1.4B 3.09 253.6 0.83 0.53 LlamaGen-3B [52] 3.1B 3.05 222.3 0.80 0.58 LlamaGen-L-384 [52] 343M 3.07 256.1 0.83 0.52 LlamaGen-XL-384 [52] 775M 2.62 244.1 0.80 0.57 LlamaGen-XXL-384 [52] 1.4B 2.34 253.9 0.80 0.59 LlamaGen-3B-384 [52] 3.1B 2.18 263.3 0.81 0.58<br><br>|
|VQ [10]<br><br>|AR<br><br>RAR-B (ours) 261M 1.95 290.5 0.82 0.58 RAR-L (ours) 461M 1.70 299.5 0.81 0.60<br><br>RAR-XL (ours) 955M 1.50 306.9 0.80 0.62 RAR-XXL (ours) 1.5B 1.48 326.0 0.80 0.63|

Table 4. ImageNet-1K 256 × 256 generation results evaluated with ADM [18]. “type” refers to the type of the generative model, where “Diff.” and “Mask.” stand for diffusion models and masked transformer models, respectively. “VQ” denotes discrete tokenizers and “VAE” stands for continuous tokenizers. “-re” stands for rejection sampling. “-384” denotes for generating images at resolution 384 and resize back to 256 for evaluation, as is used in [52].

generators outperform state-of-the-art diffusion models and masked transformer models.

Sampling Speed. One key advantage of AR methods is their ability to leverage established optimization techniques from LLMs, such as KV-caching. In Tab. 5, we compare the sampling speed (measured as images/sec) of RAR against other types of generative models, such diffusion models [45], masked transformers [65, 73], VAR [58], and MAR [36]. Among them, AR models (RAR) and VAR models (VAR-d30) are compatible with the KV-cache optimization, providing a significant advantage in generation speed over other methods. As shown in Tab. 5, RAR achieves a state-of-the-art FID score while also significantly

22.5

RAR-B RAR-L RAR-XL

RAR-B RAR-L RAR-XL

RAR-B RAR-L RAR-XL

7.0

20.0

FID(w/oclassifierfreeguidance)

- 2

- 3

- 4

- 5

FID(w/classifierfreeguidance)

RAR-XXL

RAR-XXL

RAR-XXL

6.5

17.5

6.0

trainingloss

15.0

5.5

12.5

10.0

5.0

7.5

4.5

5.0

4.0

0 50000 100000 150000 200000 250000

50000 75000 100000 125000 150000 175000 200000 225000 250000

50000 75000 100000 125000 150000 175000 200000 225000 250000

annealing starts annealing ends

annealing starts annealing ends

annealing starts annealing ends

steps

steps

steps

(a) training losses

(b) FID scores w/o classifier-free guidance

(c) FID scores w/ classifier-free guidance

Figure 4. Scaling behavior of RAR models. The scaled-up RAR models demonstrate (a) reduced training losses, and improved FID scores both (b) without and (c) with classifier-free guidance.

|method|type #params FID↓ steps images/sec<br><br>|
|---|---|
|DiT-XL/2 [45] TiTok-S-128 [73] VAR-d30 [58] MAR-B [36] RAR-B (ours)<br><br>|Diff. 675M 2.27 250 0.6<br><br>Mask. 287M 1.97 64 7.8 VAR 2.0B 1.92 10 17.3 MAR 208M 2.31 256 0.8<br><br>AR 261M 1.95 256 17.0|
|MAR-L [36] RAR-L (ours)<br><br>|MAR 479M 1.78 256 0.5 AR 461M 1.70 256 15.0|
|MaskBit [65] MAR-H [36] RAR-XL (ours) RAR-XXL (ours)<br><br>|Mask. 305M 1.52 256 0.7 MAR 943M 1.55 256 0.3<br><br>AR 955M 1.50 256 8.3 AR 1.5B 1.48 256 6.4|

Table 5. Sampling throughput comparison (including detokenization process) categorized by methods with similar FID scores. Throughputs are measured as samples generated per second on a single A100 using float32 precision and a batch size of 128, based on their official codebases. For VAR [58] and our RAR, KV-cache is applied. “Diff.” and “Mask.” refer to diffusion models and masked transformer models, respectively.

surpassing other methods in generation speed. For instance,

- at an FID score around 1.5, MaskBit [65] and MAR-H [36] generate image samples at 0.7 and 0.3 images per second, respectively. In comparison, RAR-XL not only achieves a better FID score but can generate 8.3 high-quality visual samples per second—11.9× faster than MaskBit and 27.7× faster than MAR-H. The largest RAR variant, RAR-XXL, further improves the FID score while maintaining a notable speed advantage, being 9.1× faster than MaskBit and 21.3× faster than MAR-H. Additionally, RAR may benefit further from LLM optimization techniques such as vLLM [34], as seen with other AR methods [52].

Scaling Behavior. We study the scaling behavior of RAR. Specifically, we plot the training loss curves and FID score curves (with and without classifier-free guidance [30]) in Fig. 4. As shown in the figure, we observe that RAR scales well at different model sizes, where larger model size leads to a consistently lower training loss and better FID score, regardless of using the enhancement of classifier-free guidance or not. We note that as RAR keeps the AR formulation and framework intact, it also inherits the scalability

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

RAR-B

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

RAR-L

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

RAR-XL

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

RAR-XXL

loggerhead turtle (33) otter (360) red panda(387) panda (388) balloon (417) dogsled (537)

Figure 5. Visualization of samples generated by RAR across various model sizes. RAR generates high-quality visual samples across all model sizes. As model size increases, fidelity and diversity improve, especially in challenging classes (e.g., dogsled).

from AR methods.

Visualization. We visualize generated samples by different RAR variants in Fig. 5, which shows that RAR is capable of generating high-quality samples with great fidelity and diversity. More visualizations are provided in the appendix.

### 5. Conclusion

In this paper, we introduced a simple yet effective strategy to enhance the visual generation quality of language modeling-compatible autoregressive image generators. By employing a randomized permutation objective, our approach enables improved bidirectional context learning while preserving the autoregressive structure. Consequently, the proposed RAR model not only surpasses previous state-of-the-art autoregressive image generation models but also outperforms leading non-autoregressive transformer and diffusion models. We hope this research contributes to advancing autoregressive transformers toward a unified framework for visual understanding and generation.

Acknowledgment. We sincerely thank Tianhong Li for his insightful discussion and feedback on this project.

## Randomized Autoregressive Visual Generation Supplementary Material

### Appendix

The supplementary material includes the following additional information:

- • Sec. A provides the detailed hyper-parameters for the final RAR models.
- • Sec. B provides the pseudo-code for randomized autoregressive modeling.
- • Sec. C visualizes the scan orders used in the ablation study.
- • Sec. D provides more visualization samples of RAR models.

### A. Hyper-parameters for Final RAR Models

We list the detailed training hyper-parameters and sampling hyper-parameters for all RAR models in Tab. 6.

config value

training hyper-params optimizer AdamW [33, 38] learning rate 4e-4 weight decay 0.03 optimizer momentum (0.9, 0.96) batch size 2048 learning rate schedule cosine decay ending learning rate 1e-5 total epochs 400 warmup epochs 100 annealing start epoch 200 annealing end epoch 300 precision bfloat16 max grad norm 1.0 dropout rate 0.1 attn dropout rate 0.1 class label dropout rate 0.1

sampling hyper-params guidance schedule pow-cosine [25] temperature 1.0 (B) / 1.02 (L, XL, XXL) scale power 2.75 (B) / 2.5 (L) / 1.5 (XL) / 1.2 (XXL) guidance scale 16.0 (B) / 15.5 (L) / 6.9 (XL) / 8.0 (XXL)

Table 6. Detailed hyper-parameters for final RAR models.

### B. Pseudo-Code for RAR

We provide a simple pseudo-code of RAR in PyTorch style in Algorithm 1.

### C. Visualization of Scan Orders

We visualize the 6 scan orders studied in the main paper ( Tab. 3) in Fig. 6.

Algorithm 1 PyTorch Pseudo-Code for Randomized AutoRegressive (RAR) Modeling

class RAR(nn.Module):

def sample orders(self, tokens, global step): # sample permutation order at training step global step. orders = [] # compute the randomized probability r as in Eq. (5). prob = 1.0 - min(1.0, max(0.0, (global step - self.anneal start) /

(self.anneal end - self.anneal start)))

for b in range(tokens.shape[0]): if random.random() <prob: # random permutation. orders.append(torch.randperm(tokens.shape[1]))

else: # raster order (no permutation). orders.append(torch.arange(tokens.shape[1]))

return torch.stack(orders)

def permute(self, inputs, orders): # permute inputs based on orders. B, L = inputs.shape[:2] indices = torch.arange(B).unsqueeze(1).expand(-1, L) return x[indices, orders]

def forward(self, tokens, condition, global step): # get permutation orders. orders = self.sample orders(global step, tokens) # permute labels for next-token prediction. labels = self.permute(tokens.clone(), orders) # token embeddings with positional embedding. x = self.tok emb(tokens) + self.pos emb # permute the token orders. x = self.permute(x, orders) # add target-aware postional embedding as in Eq. (4). target pos emb = self.target pos emb.repeat(x.shape[0], 1, 1) target pos emb = self.permute(target pos emb, orders) # shifting so each token will see next-token’s embedding. target pos emb = target pos emb[:, 1:] x = torch.cat([x[:, :-1] + target pos emb, x[:, -1:]], dim=1) # transformer forwarding. pred = self.transformers(x, condition) # next token prediction loss. loss = nn.CrossEntropy(pred[:, :-1], labels[:, 1:]) return loss

### D. Visualization on Generated Samples

We provide visualization results in Fig. 7, Fig. 8, and Fig. 9.

[Figure 44]

0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15

16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31

32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47

48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63

64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79

80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95

96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111

112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127

128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143

144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159

160 161 162 163 164 165 166 167 168 169 170 171 172 173 174 175

176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191

192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207

208 209 210 211 212 213 214 215 216 217 218 219 220 221 222 223

224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239

240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255

(a) row-major

[Figure 45]

0 1 4 5 16 17 20 21 64 65 68 69 80 81 84 85

2 3 6 7 18 19 22 23 66 67 70 71 82 83 86 87

8 9 12 13 24 25 28 29 72 73 76 77 88 89 92 93

10 11 14 15 26 27 30 31 74 75 78 79 90 91 94 95

32 33 36 37 48 49 52 53 96 97 100 101 112 113 116 117

34 35 38 39 50 51 54 55 98 99 102 103 114 115 118 119

40 41 44 45 56 57 60 61 104 105 108 109 120 121 124 125

42 43 46 47 58 59 62 63 106 107 110 111 122 123 126 127

128 129 132 133 144 145 148 149 192 193 196 197 208 209 212 213

130 131 134 135 146 147 150 151 194 195 198 199 210 211 214 215

136 137 140 141 152 153 156 157 200 201 204 205 216 217 220 221

138 139 142 143 154 155 158 159 202 203 206 207 218 219 222 223

160 161 164 165 176 177 180 181 224 225 228 229 240 241 244 245

162 163 166 167 178 179 182 183 226 227 230 231 242 243 246 247

168 169 172 173 184 185 188 189 232 233 236 237 248 249 252 253

170 171 174 175 186 187 190 191 234 235 238 239 250 251 254 255

(d) z-curve

[Figure 46]

- 0 59 58 57 56 55 54 53 52 51 50 49 48 47 46 45
- 1 60 111 110 109 108 107 106 105 104 103 102 101 100 99 44
- 2 61 112 155 154 153 152 151 150 149 148 147 146 145 98 43
- 3 62 113 156 191 190 189 188 187 186 185 184 183 144 97 42
- 4 63 114 157 192 219 218 217 216 215 214 213 182 143 96 41
- 5 64 115 158 193 220 239 238 237 236 235 212 181 142 95 40
- 6 65 116 159 194 221 240 251 250 249 234 211 180 141 94 39
- 7 66 117 160 195 222 241 252 255 248 233 210 179 140 93 38
- 8 67 118 161 196 223 242 253 254 247 232 209 178 139 92 37
- 9 68 119 162 197 224 243 244 245 246 231 208 177 138 91 36
- 10 69 120 163 198 225 226 227 228 229 230 207 176 137 90 35
- 11 70 121 164 199 200 201 202 203 204 205 206 175 136 89 34
- 12 71 122 165 166 167 168 169 170 171 172 173 174 135 88 33
- 13 72 123 124 125 126 127 128 129 130 131 132 133 134 87 32
- 14 73 74 75 76 77 78 79 80 81 82 83 84 85 86 31
- 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30

(b) spiral in

[Figure 47]

0 4 8 12 16 20 24 28 1 5 9 13 17 21 25 29

32 36 40 44 48 52 56 60 33 37 41 45 49 53 57 61

64 68 72 76 80 84 88 92 65 69 73 77 81 85 89 93

96 100 104 108 112 116 120 124 97 101 105 109 113 117 121 125

128 132 136 140 144 148 152 156 129 133 137 141 145 149 153 157

160 164 168 172 176 180 184 188 161 165 169 173 177 181 185 189

192 196 200 204 208 212 216 220 193 197 201 205 209 213 217 221

224 228 232 236 240 244 248 252 225 229 233 237 241 245 249 253

2 6 10 14 18 22 26 30 3 7 11 15 19 23 27 31

34 38 42 46 50 54 58 62 35 39 43 47 51 55 59 63

66 70 74 78 82 86 90 94 67 71 75 79 83 87 91 95

98 102 106 110 114 118 122 126 99 103 107 111 115 119 123 127

130 134 138 142 146 150 154 158 131 135 139 143 147 151 155 159

162 166 170 174 178 182 186 190 163 167 171 175 179 183 187 191

194 198 202 206 210 214 218 222 195 199 203 207 211 215 219 223

226 230 234 238 242 246 250 254 227 231 235 239 243 247 251 255

(e) subsample

[Figure 48]

255 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210

254 195 144 145 146 147 148 149 150 151 152 153 154 155 156 211

253 194 143 100 101 102 103 104 105 106 107 108 109 110 157 212

252 193 142 99 64 65 66 67 68 69 70 71 72 111 158 213

251 192 141 98 63 36 37 38 39 40 41 42 73 112 159 214

250 191 140 97 62 35 16 17 18 19 20 43 74 113 160 215

249 190 139 96 61 34 15 4 5 6 21 44 75 114 161 216

248 189 138 95 60 33 14 3 0 7 22 45 76 115 162 217

247 188 137 94 59 32 13 2 1 8 23 46 77 116 163 218

246 187 136 93 58 31 12 11 10 9 24 47 78 117 164 219

245 186 135 92 57 30 29 28 27 26 25 48 79 118 165 220

244 185 134 91 56 55 54 53 52 51 50 49 80 119 166 221

243 184 133 90 89 88 87 86 85 84 83 82 81 120 167 222

242 183 132 131 130 129 128 127 126 125 124 123 122 121 168 223

241 182 181 180 179 178 177 176 175 174 173 172 171 170 169 224

240 239 238 237 236 235 234 233 232 231 230 229 228 227 226 225

(c) spiral out

[Figure 49]

0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15

- 31 30 29 28 27 26 25 24 23 22 21 20 19 18 17 16
- 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47

- 63 62 61 60 59 58 57 56 55 54 53 52 51 50 49 48
- 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79

- 95 94 93 92 91 90 89 88 87 86 85 84 83 82 81 80
- 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111

- 127 126 125 124 123 122 121 120 119 118 117 116 115 114 113 112
- 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143

- 159 158 157 156 155 154 153 152 151 150 149 148 147 146 145 144
- 160 161 162 163 164 165 166 167 168 169 170 171 172 173 174 175

- 191 190 189 188 187 186 185 184 183 182 181 180 179 178 177 176
- 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207

- 223 222 221 220 219 218 217 216 215 214 213 212 211 210 209 208
- 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239

255 254 253 252 251 250 249 248 247 246 245 244 243 242 241 240

(f) alternate

- Figure 6. Different scan orders for a 16 × 16 grid (256 tokens). The number indicates the token’s indices in the scanning order.

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

class id 90, lorikeet class id 207, golden retriever

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

class id 928, ice cream class id 973, coral reef

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

- Figure 7. Visualization samples from RAR. RAR is capable of generating high-fidelity image samples with great diversity.

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

class id 360, otter class id 562, fountain

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

class id 933, cheeseburger class id 972, cliff

- Figure 8. Visualization samples from RAR. RAR is capable of generating high-fidelity image samples with great diversity.

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

class id 250, Siberian husky class id 812, space shuttle

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

[Figure 327]

[Figure 328]

[Figure 329]

[Figure 330]

[Figure 331]

[Figure 332]

[Figure 333]

[Figure 334]

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

[Figure 341]

[Figure 342]

[Figure 343]

[Figure 344]

[Figure 345]

[Figure 346]

[Figure 347]

[Figure 348]

[Figure 349]

[Figure 350]

[Figure 351]

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

[Figure 356]

[Figure 357]

[Figure 358]

class id 975, lakeside class id 980, volcano

[Figure 359]

[Figure 360]

[Figure 361]

[Figure 362]

[Figure 363]

[Figure 364]

[Figure 365]

[Figure 366]

[Figure 367]

[Figure 368]

[Figure 369]

[Figure 370]

[Figure 371]

[Figure 372]

[Figure 373]

[Figure 374]

[Figure 375]

[Figure 376]

[Figure 377]

[Figure 378]

[Figure 379]

[Figure 380]

[Figure 381]

[Figure 382]

[Figure 383]

[Figure 384]

[Figure 385]

- Figure 9. Visualization samples from RAR. RAR is capable of generating high-fidelity image samples with great diversity.

### References

- [1] Marah Abdin, Sam Ade Jacobs, Ammar Ahmad Awan, Jyoti Aneja, Ahmed Awadallah, Hany Awadalla, Nguyen Bach, Amit Bahree, Arash Bakhtiari, Harkirat Behl, et al. Phi-3 technical report: A highly capable language model locally on your phone. arXiv preprint arXiv:2404.14219, 2024. 1, 2
- [2] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.
- [3] Rohan Anil, Andrew M Dai, Orhan Firat, Melvin Johnson, Dmitry Lepikhin, Alexandre Passos, Siamak Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng Chen, et al. Palm 2 technical report. arXiv preprint arXiv:2305.10403, 2023.
- [4] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023. 2
- [5] Yutong Bai, Xinyang Geng, Karttikeya Mangalam, Amir Bar, Alan Yuille, Trevor Darrell, Jitendra Malik, and Alexei A Efros. Sequential modeling enables scalable learning for large vision models. In CVPR, 2024. 1
- [6] Fan Bao, Shen Nie, Kaiwen Xue, Yue Cao, Chongxuan Li, Hang Su, and Jun Zhu. All are worth words: A vit backbone for diffusion models. In CVPR, 2023. 1, 7
- [7] Lucas Beyer, Andreas Steiner, Andr´e Susano Pinto, Alexander Kolesnikov, Xiao Wang, Daniel Salz, Maxim Neumann, Ibrahim Alabdulmohsin, Michael Tschannen, Emanuele Bugliarello, et al. Paligemma: A versatile 3b vlm for transfer. arXiv preprint arXiv:2407.07726, 2024. 1, 3
- [8] Andrew Brock. Large scale gan training for high fidelity natural image synthesis. arXiv preprint arXiv:1809.11096,

2018. 3

- [9] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. NeurIPS, 2020. 1, 2, 4
- [10] Huiwen Chang, Han Zhang, Lu Jiang, Ce Liu, and William T Freeman. Maskgit: Masked generative image transformer. In CVPR, 2022. 1, 2, 3, 4, 5, 7
- [11] Huiwen Chang, Han Zhang, Jarred Barber, AJ Maschinot, Jose Lezama, Lu Jiang, Ming-Hsuan Yang, Kevin Murphy, William T Freeman, Michael Rubinstein, et al. Muse: Textto-image generation via masked generative transformers. In ICML, 2023. 3, 6
- [12] Mark Chen, Alec Radford, Rewon Child, Jeffrey Wu, Heewoo Jun, David Luan, and Ilya Sutskever. Generative pretraining from pixels. In ICML, 2020. 2
- [13] Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. Palm: Scaling language modeling with pathways. Journal of Machine Learning Research, 24(240): 1–113, 2023. 2
- [14] Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi Wang, Mostafa

- Dehghani, Siddhartha Brahma, et al. Scaling instructionfinetuned language models. JMLR, 25(70):1–53, 2024. 1
- [15] Mostafa Dehghani, Josip Djolonga, Basil Mustafa, Piotr Padlewski, Jonathan Heek, Justin Gilmer, Andreas Peter Steiner, Mathilde Caron, Robert Geirhos, Ibrahim Alabdulmohsin, et al. Scaling vision transformers to 22 billion parameters. In ICML, pages 7480–7512. PMLR, 2023. 5
- [16] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In CVPR, 2009. 2, 5, 7
- [17] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In NAACL, 2018. 3, 4
- [18] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. NeurIPS, 2021. 1, 3, 6, 7
- [19] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In ICLR, 2021. 4, 5
- [20] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783,

2024. 2

- [21] Alaaeldin El-Nouby, Michal Klein, Shuangfei Zhai, Miguel Angel Bautista, Alexander Toshev, Vaishaal Shankar, Joshua M Susskind, and Armand Joulin. Scalable pretraining of large autoregressive image models. ICML, 2024. 1, 3
- [22] Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image synthesis. In CVPR,

2021. 1, 2, 3, 4, 5, 6, 7

- [23] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first International Conference on Machine Learning, 2024. 3
- [24] Lijie Fan, Tianhong Li, Siyang Qin, Yuanzhen Li, Chen Sun, Michael Rubinstein, Deqing Sun, Kaiming He, and Yonglong Tian. Fluid: Scaling autoregressive text-to-image generative models with continuous tokens. arXiv preprint arXiv:2410.13863, 2024. 3
- [25] Shanghua Gao, Pan Zhou, Ming-Ming Cheng, and Shuicheng Yan. Masked diffusion transformer is a strong image synthesizer. In ICCV, 2023. 6, 7, 1
- [26] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. NeurIPS, 2014. 3
- [27] Karol Gregor, Ivo Danihelka, Andriy Mnih, Charles Blundell, and Daan Wierstra. Deep autoregressive networks. In International Conference on Machine Learning, pages 1242–1250. PMLR, 2014. 2

- [28] Jiatao Gu, Changhan Wang, and Junbo Zhao. Levenshtein transformer. NeurIPS, 32, 2019. 4
- [29] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Doll´ar, and Ross Girshick. Masked autoencoders are scalable vision learners. In CVPR, 2022. 1
- [30] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022. 6, 8
- [31] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. NeurIPS, 2020. 3
- [32] Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. In CVPR, 2019. 3
- [33] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In ICLR, 2015. 6, 1
- [34] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the 29th Symposium on Operating Systems Principles, pages 611–626, 2023. 8
- [35] Doyup Lee, Chiheon Kim, Saehoon Kim, Minsu Cho, and Wook-Shin Han. Autoregressive image generation using residual quantization. In CVPR, 2022. 1
- [36] Tianhong Li, Yonglong Tian, He Li, Mingyang Deng, and Kaiming He. Autoregressive image generation without vector quantization. NeurIPS, 2024. 1, 2, 3, 6, 7, 8
- [37] Qihao Liu, Zhanpeng Zeng, Ju He, Qihang Yu, Xiaohui Shen, and Liang-Chieh Chen. Alleviating distortion in image generation via multi-resolution diffusion models. NeurIPS,

2024. 3, 7

- [38] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. ICLR, 2019. 6, 1
- [39] Zhuoyan Luo, Fengyuan Shi, Yixiao Ge, Yujiu Yang, Limin Wang, and Ying Shan. Open-magvit2: An open-source project toward democratizing auto-regressive visual generation. arXiv preprint arXiv:2409.04410, 2024. 1, 2, 7
- [40] Nanye Ma, Mark Goldstein, Michael S Albergo, Nicholas M Boffi, Eric Vanden-Eijnden, and Saining Xie. Sit: Exploring flow and diffusion-based generative models with scalable interpolant transformers. ECCV, 2024. 7
- [41] Xiaoxiao Ma, Mohan Zhou, Tao Liang, Yalong Bai, Tiejun Zhao, Huaian Chen, and Yi Jin. Star: Scale-wise text-toimage generation via auto-regressive representations. arXiv preprint arXiv:2406.10797, 2024. 3
- [42] Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, et al. Self-refine: Iterative refinement with self-feedback. NeurIPS, 36, 2023. 4
- [43] OpenAI. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023. 1, 2
- [44] Niki Parmar, Ashish Vaswani, Jakob Uszkoreit, Lukasz Kaiser, Noam Shazeer, Alexander Ku, and Dustin Tran. Image transformer. In International conference on machine learning, pages 4055–4064. PMLR, 2018. 2
- [45] William Peebles and Saining Xie. Scalable diffusion models with transformers. In ICCV, 2023. 1, 3, 5, 7, 8

- [46] Alec Radford. Improving language understanding by generative pre-training. OpenAI, 2018. 2
- [47] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In ICML, 2021. 2
- [48] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In ICML, 2021. 2
- [49] Ali Razavi, Aaron Van den Oord, and Oriol Vinyals. Generating diverse high-fidelity images with vq-vae-2. NeurIPS,

2019. 2

- [50] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, 2022. 3, 7
- [51] stabilityai, 2023. 7
- [52] Peize Sun, Yi Jiang, Shoufa Chen, Shilong Zhang, Bingyue Peng, Ping Luo, and Zehuan Yuan. Autoregressive model beats diffusion: Llama for scalable image generation. arXiv preprint arXiv:2406.06525, 2024. 1, 2, 4, 5, 7, 8
- [53] Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, and Andrew Rabinovich. Going deeper with convolutions. In CVPR, 2015. 5
- [54] Haotian Tang, Yecheng Wu, Shang Yang, Enze Xie, Junsong Chen, Junyu Chen, Zhuoyang Zhang, Han Cai, Yao Lu, and Song Han. Hart: Efficient visual generation with hybrid autoregressive transformer. arXiv preprint arXiv:2410.10812,

2024. 3

- [55] Chameleon Team. Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818, 2024. 1, 2
- [56] Emu3 Team. Emu3: Next-token prediction is all you need. Tech Report, 2024. 1, 2
- [57] Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023. 1, 2
- [58] Keyu Tian, Yi Jiang, Zehuan Yuan, Bingyue Peng, and Liwei Wang. Visual autoregressive modeling: Scalable image generation via next-scale prediction. NeurIPS, 2024. 1, 2, 3, 7, 8
- [59] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023. 1, 2
- [60] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023. 1, 2
- [61] Benigno Uria, Marc-Alexandre Cˆot´e, Karol Gregor, Iain Murray, and Hugo Larochelle. Neural autoregressive distribution estimation. JMLR, 17(205):1–37, 2016. 3, 4

- [62] Aaron Van den Oord, Nal Kalchbrenner, Lasse Espeholt, Oriol Vinyals, Alex Graves, et al. Conditional image generation with pixelcnn decoders. NeurIPS, 2016. 2
- [63] A¨aron Van Den Oord, Nal Kalchbrenner, and Koray Kavukcuoglu. Pixel recurrent neural networks. In International conference on machine learning, pages 1747–1756. PMLR, 2016. 2
- [64] Aaron Van Den Oord, Oriol Vinyals, et al. Neural discrete representation learning. NeurIPS, 2017. 2
- [65] Mark Weber, Lijun Yu, Qihang Yu, Xueqing Deng, Xiaohui Shen, Daniel Cremers, and Liang-Chieh Chen. Maskbit: Embedding-free image generation via bit tokens. arXiv preprint arXiv:2409.16211, 2024. 1, 3, 7, 8
- [66] Jason Wei, Maarten Bosma, Vincent Y Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M Dai, and Quoc V Le. Finetuned language models are zero-shot learners. In ICLR, 2022. 1
- [67] An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, et al. Qwen2 technical report. arXiv preprint arXiv:2407.10671, 2024. 1, 2
- [68] Zhilin Yang, Zihang Dai, Yiming Yang, Jaime Carbonell, Ruslan Salakhutdinov, and Quoc V. Le. Xlnet: Generalized autoregressive pretraining for language understanding. NeurIPS, 2019. 3, 4
- [69] Jiahui Yu, Xin Li, Jing Yu Koh, Han Zhang, Ruoming Pang, James Qin, Alexander Ku, Yuanzhong Xu, Jason Baldridge, and Yonghui Wu. Vector-quantized image modeling with improved vqgan. In ICLR, 2022. 1, 2, 7
- [70] Jiahui Yu, Yuanzhong Xu, Jing Yu Koh, Thang Luong, Gunjan Baid, Zirui Wang, Vijay Vasudevan, Alexander Ku, Yinfei Yang, Burcu Karagol Ayan, et al. Scaling autoregressive models for content-rich text-to-image generation. TMLR,

2022. 1

- [71] Lijun Yu, Yong Cheng, Kihyuk Sohn, Jos´e Lezama, Han Zhang, Huiwen Chang, Alexander G Hauptmann, MingHsuan Yang, Yuan Hao, Irfan Essa, et al. Magvit: Masked generative video transformer. In CVPR, 2023. 1, 3
- [72] Lijun Yu, Jos´e Lezama, Nitesh B Gundavarapu, Luca Versari, Kihyuk Sohn, David Minnen, Yong Cheng, Agrim Gupta, Xiuye Gu, Alexander G Hauptmann, et al. Language model beats diffusion–tokenizer is key to visual generation. In ICLR, 2024. 3, 7
- [73] Qihang Yu, Mark Weber, Xueqing Deng, Xiaohui Shen, Daniel Cremers, and Liang-Chieh Chen. An image is worth 32 tokens for reconstruction and generation. NeurIPS, 2024. 1, 3, 6, 7, 8
- [74] Xiaohua Zhai, Alexander Kolesnikov, Neil Houlsby, and Lucas Beyer. Scaling vision transformers. In CVPR, pages 12104–12113, 2022. 5
- [75] Qian Zhang, Xiangzi Dai, Ninghua Yang, Xiang An, Ziyong Feng, and Xingyu Ren. Var-clip: Text-to-image generator with visual auto-regressive modeling. arXiv preprint arXiv:2408.01181, 2024. 3

