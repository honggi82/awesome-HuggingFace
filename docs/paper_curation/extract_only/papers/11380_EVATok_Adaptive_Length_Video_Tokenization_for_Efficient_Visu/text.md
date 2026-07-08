# arXiv:2603.12267v1[cs.CV]12Mar2026

## EVATok: Adaptive Length Video Tokenization for Efficient Visual Autoregressive Generation

Tianwei Xiong1 Jun Hao Liew2 Zilong Huang2 Zhijie Lin2 Jiashi Feng2 Xihui Liu1† 1The University of Hong Kong 2ByteDance Seed

#### Abstract

###### Reconstruction Generation

[Figure 1]

[Figure 2]

Autoregressive (AR) video generative models rely on video tokenizers that compress pixels into discrete token sequences. The length of these token sequences is crucial for balancing reconstruction quality against downstream generation computational cost. Traditional video tokenizers apply a uniform token assignment across temporal blocks of different videos, often wasting tokens on simple, static, or repetitive segments while underserving dynamic or complex ones. To address this inefficiency, we introduce EVATok, a framework to produce Efficient Video Adaptive Tokenizers. Our framework estimates optimal token assignments for each video to achieve the best quality-cost trade-off, develops lightweight routers for fast prediction of these optimal assignments, and trains adaptive tokenizers that encode videos based on the assignments predicted by routers. We demonstrate that EVATok delivers substantial improvements in efficiency and overall quality for video reconstruction and downstream AR generation. Enhanced by our advanced training recipe that integrates video semantic encoders, EVATok achieves superior reconstruction and stateof-the-art class-to-video generation on UCF-101, with at least 24.4% savings in average token usage compared to the prior state-of-the-art LARP and our fixed-length baseline.

Tokens ↓ LPIPS ↓ Traditional:

###### Assignment

[Figure 3]

- (a)
- (b)
- (c)
- (d)

1024 0.052

[Figure 4]

Ours: 352 0.052

Block1 Block2 Block3 Block4

[Figure 5]

[Figure 6]

Traditional:

1024 0.181

[Figure 7]

Ours: 608 0.162

Block1 Block2 Block3 Block4

[Figure 8]

[Figure 9]

Traditional:

1024 0.132

[Figure 10]

Ours: 928 0.112

Block1 Block2 Block3 Block4

[Figure 11]

[Figure 12]

Traditional:

1024 0.217

[Figure 13]

Ours: 1536 0.188

Block1 Block 2 Block3 Block4

[Figure 14]

#### 1. Introduction

Figure 1. EVATok highlights. Top: EVATok achieves superior video reconstruction and downstream generation quality with significant savings in token usage. Bottom: EVATok assigns tokens in an intuitive way. Clips with dynamic motion or complex layout will be encoded with more tokens, while clips that are repetitive or simple will be assigned fewer tokens.

Visual generation with autoregressive (AR) language models is rapidly advancing [11, 35, 62, 67], driven by the success of LLMs [1, 10, 14, 19, 34] and their potential for unified multi-modal generation [9, 30, 43, 65, 66]. AR visual generative models typically operate on sequences of discrete visual tokens, obtained by patch-wise compression of pixels via visual tokenizers [16, 73]. The tokenizer’s design directly influences reconstruction quality and token sequence length, thus determining the quality-cost trade-off and computational overhead for downstream AR models.

However, most existing visual tokenizers [16, 73, 75] produce fixed-length sequences regardless of input content complexity. This uniform budget allocation is especially inefficient for AR video generative models using causal video tokenizers [17, 30, 62, 64, 75], as information density in videos varies not only across samples but also temporally: simple-layout, near-static or repetitive segments receive ex-

Project page: https://silentview.github.io/EVATok/

cessive tokens, while dynamic or complex-layout segments are undeserved, compromising both efficiency and fidelity.

Ideally, given a video and specified preference between better quality or less token cost, we would predict an optimal assignment—specifying the total number of tokens used for video reconstruction and their distribution over temporal blocks—that maximizes the quality-cost trade-off. Prior video adaptive tokenizers [33, 70] enable variable length compression via tail-token-dropping [3, 41] training, with assignment selection by threshold-based search [70] or Integer Linear Programming (ILP) within video minibatches under fixed average budget constraints [33]. However, these approaches can yield suboptimal results: heuristic threshold-based searches may neglect global quality-cost balance, while mini-batch ILP ties per-sample decisions to the batch compositions and rigid average budgets. Critically, they do not address the core need: for each sample, determining the optimal assignment tailored to the samples’s inherent complexity, enabling optimal adaptive tokenization that allocates budgets where they are most needed, achieving the best balance for overall efficiency and quality.

The challenge for optimal assignment identification is that there was no estimation approach or even definition for it. To fill in this blank, we formulate the optimal assignment identification problem as a tractable maximum proxy reward assignment identification task, where the proxy reward is a novel metric measuring both the reconstruction quality and cost (token length) to quantify the quality-cost trade-off for a particular assignment. In other words, the optimal assignment with the maximum proxy reward achieves the best quality-cost trade-off.

To estimate the proxy reward, we introduce a proxy tokenizer that learns to reconstruct the input video under different token assignments. Once trained, we can simply iterate over all possible candidates to identify the optimal token assignment with maximum proxy reward. And to build a faster approach for optimal assignments prediction, we curate a dataset to train a lightweight model, named the router, which learns optimal assignment prediction in a classification task form. Equipped with this router, we train final adaptive tokenizers to encode videos using contentadaptive assignments, which in turn support downstream efficient adaptive length AR generative models. In summary, EVATok unfolds in four stages: (1) Train a proxy tokenizer for optimal assignment estimation; (2) Curate a dataset of (video, optimal assignment) pairs for router training; (3) Train a lightweight router for fast optimal assignment prediction; and (4) Train the final video adaptive tokenizer under assignments from the router.

For video reconstruction and downstream AR generation, EVATok yields substantial gains in efficiency and quality. Enhanced by our advanced recipe integrating video semantic encoders [2, 53] into tokenizer training,

EVATok achieves superior reconstruction and state-of-theart (SOTA) class-to-video generation quality with at least 24.4% token length savings compared to prior video tokenizers [33, 59], as shown in Fig. 1. The adaptive assignment examples of EVATok in Fig. 1 also correspond to intuitions: repetitive, simple-layout, and static content is assigned fewer tokens; in contrast, non-repetitive, complexlayout, and dynamic content is assigned more. EVATok highlights the promising potential of content-adaptive video tokenization for improving efficiency and quality for overall reconstruction and downstream AR generation.

We summarize our main contributions as follows:

- • A four-stage framework for efficient video adaptive tokenization, featuring a router that provides optimal budget assignment during training and inference of tokenizers.
- • Proxy reward: a novel metric utilizing a variable length tokenizer to identify optimal assignments for each video.
- • Extensive experiments showing that content-adaptive video tokenization can surpass fixed-length baselines, achieving superior performances in reconstruction and downstream AR generation with fewer tokens.

#### 2. Related Work

Discrete image and video tokenizers. Since the classic VQ-VAE [56] and VQ-GAN [16], extensive efforts have been made to better compress visual inputs into discrete token sequences for autoregressive modeling. LFQ [75] and FSQ [40] are proposed for large-scale codebook training. VAR [52] encodes token sequences in a residualstyle [31] multi-scale structure for efficient generation. For videos, while many works choose 3D CNN to implement video tokenizers [18, 62, 64, 75], recently more video tokenizers are being implemented using transformer architecture [33, 57, 59, 60, 70]. Transformers are beneficial to video tokenizers not only due to their known scalability, but also because their flexible attention mechanism naturally helps build 1D tokenizers [70, 76], which removes the grid-like spatial prior in token sequences, making the sequence length easy to adjust and convenient for adaptive tokenization. In this work, we use Q-Former-style [6, 32] 1D tokenizer design [68] to build adaptive video tokenizers. Adaptive visual tokenization. Based on the intuition that simple content needs fewer tokens while complex content needs more for efficient compression, the seminal work Dynamic VQ [24] encodes different regions across images with different granularity adaptively, utilizing Gumbel Softmax [26]. Differently, CAT [46] lets LLMs decide the compression granularity based on captions. Recently, techniques like tail-token-dropping [3, 41, 61, 70] or iterative token allocation [15, 39] have been used to enable tokenizers to encode visual inputs under different token assignments. Further on video tokenization, ElasticTok [70] and AdapTok [33] study how to determine these given assignments

in adaptive video tokenization. However, their adaptive assignment searching methods are heuristic and can potentially lead to suboptimal assignments. A concurrent work, InfoTok [72], masks less important tokens from pre-trained tokenizers with an ELBO-based method. In this work, EVATok directly predicts the optimal assignments given input videos and preferences between quality and cost.

Video representation alignment. The representation of pretrained semantic encoders [42, 44, 78] have been used to enhance image generative models [77] or image tokenizers [4, 38, 68, 69, 71]. Recently, similar approaches have been studied for video diffusion models [80] or reported in use for video tokenizers [11]. We further reveal that representation alignment is beneficial for video tokenizers, especially when combined with semantic video discriminators.

#### 3. Method

Problem setup. We first introduce the problem setup of our video adaptive tokenizer before presenting our proposed solution. Given a video x, we temporally downsample it by 4× and divide it into T causal blocks. Each block t is assigned kt tokens from a candidate set K (e.g., {32,...,512}) with m levels, forming an assignment a = (k1,...,kT) with total token length L(a) = Tt=1 kt. We identify that the primary challenge for an adaptive video tokenizer lies in predicting the optimal token assignment a∗ for each video to achieve the best quality-cost trade-off.

We formulate the optimal assignment prediction problem as a tractable maximum proxy reward assignment prediction task, where the proxy reward is a novel metric that we introduce to quantify the quality-cost trade-off performance for a particular token assignment. Centering on the idea of using proxy reward for optimal assignment prediction, we build our four-stage framework, as in Fig. 2, for efficient video adaptive tokenization: (Stage 1) train a proxy tokenizer that can reconstruct videos w.r.t. randomly sampled token assignments. This proxy tokenizer later serves for proxy reward computation; (Stage 2) curate a dataset comprising (video, optimal token assignment) pairs by searching proxy reward under different token assignments calculated with the proxy tokenizer. This dataset serves for training a router for fast optimal assignment prediction; (Stage 3) train the router on this curated dataset, to accelerate optimal assignment prediction largely against searching; (Stage 4) train an adaptive video tokenizer using the optimal assignments predicted by the router, hence achieving adaptive length video tokenization. Next, we will introduce each stage with more details.

###### 3.1. Stage 1: Training a Proxy Tokenizer

In stage 1, we train a proxy tokenizer that can reconstruct a video w.r.t. a randomly sampled assignment a. This proxy

tokenizer can serve as a proxy for the assessment of the quality of a particular token assignment, which we subsequently use to identify optimal assignments, train a router, and build the final adaptive length tokenizer.

Model architecture. We adopt a Q-Former [6, 32] style 1D tokenizer for its scalability [68] and flexibility in variable length tokenization. As shown in Fig. 3, input videos are first spatio-temporally patchified into 3D embeddings using a simple linear patch embedding module, consistent with prior video tokenizers [33, 59, 70]. Then, given a randomly sampled token assignment a = (q1,q2,q3,q4) specifying the number of 1D tokens per temporal block, a 1D query sequence is initialized, i.e., each 1D query is derived from the 2D pooling feature of the corresponding temporal block in the 3D embeddings. Through Q-Former encoder layers, these 1D queries encode visual information from the 3D embeddings and are vector-quantized into discrete tokens. Temporally causal attention masks ensure that 1D tokens do not encode information from subsequent blocks.

As for the decoder, 3D queries are initialized using the first 1D token in their corresponding temporal block. After a similar temporally causal decoding process, the final 3D features will be linearly projected and reshaped into video frames. We do not use the typical tail-tokendropping [33, 39, 41, 70] strategy to produce variable length tokens as it may lead to two concerns: (1) extra computation overhead caused by using many tokens that will just be dropped as register tokens; (2) the roles of tail 1D queries being ambiguous during encoding: tail 1D queries cannot be aware of whether they will be dropped after encoding. Since the two concerns potentially hurt efficiency and performance, in EVATok, the length of 1D tokens is decided and fixed during the initialization of 1D queries.

Training recipe. We enhance the tokenizer training by video semantic encoders [2, 53] through video representation alignment. Following typical image representation alignment approaches [68, 71, 77], we apply patch-wise alignment between the intermediate 3D features of the tokenizer decoder and the features from pre-trained V-JEPA2L [2]. We use a linear projection and reshape strategy, similar to depatchify, to address feature shape mismatches. Formally, let fdec,l be the output 3D features from the l-th decoder layer, and fsem the semantic features from the pretrained encoder. The representation alignment loss is:

1 N

Lalign = −

N

sim fndec,l,ϕ(fnsem) (1)

n=1

where N is the batch size, n is the batch item index, sim(·,·) is cosine similarity, and ϕ(·) combines an MLP and a depatchify module for shape matching. We use a transformer PatchGAN [25] discriminator as in LARP [59]. The final

###### Font, some too big, arrows alignment

Stage1 Proxy Tokenizer Training

Stage2 Router Training Data Curation

###### Stage3 Router Training

Stage4 Final Tokenizer Training

|[Figure 15]<br><br>[Figure 16]<br><br>[Figure 17]|
|---|

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

Proxy Tokenizer

[Figure 31]

Block1 Block2 Block3 Block4 Block1 Block2 Block3 Block4 Block1 Block 2 Block 3 Block 4

All Candidate Assignments

[Figure 32]

[Figure 33]

Traverse to Get

[Figure 34]

[Figure 35]

[Figure 36]

|[Figure 37]<br><br>Block4<br><br>[Figure 38]<br><br>[Figure 39]<br><br>[Figure 40]<br><br>[Figure 41]<br><br>All Candidate Assignments<br><br>[Figure 42]<br><br>Proxy Reward<br><br>-1.2<br><br>1.5<br><br>2.6<br><br>[Figure 43]<br><br>[Figure 44]<br><br>Block1|
|---|

Router Router

[Figure 45]

、

[Figure 46]

[Figure 47]

[Figure 48]

Find Optimal Assignments 𝑎∗ = argmax Proxy Reward

Providing Supervision

[Figure 49]

[Figure 50]

Proxy Tokenizer Final Tokenizer

|[Figure 51]<br><br>(Video, Optimal Assignment)|
|---|

|[Figure 52]<br><br>(Video, Optimal Assignment)|
|---|

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

Reconstruct Random sample Trainable Frozen Dataset

- Figure 2. Four-stage framework for adaptive video tokenizer training. Stage 1 trains a proxy tokenizer to reconstruct videos under all candidate assignments. Stage 2 compute proxy rewards for all candidate assignments across videos from a dataset. It identifies the ass rewards to curate a classification dataset of videos and their optimal assignments. Stage 3 trains a router t the optimal assignments for videos. Stage 4 trains the final tokenizer from scratch, with the router determining input video during training.

VQ

Q-Former Encoder

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

|Video|
|---|

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

Patchify

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

Depatchify

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

Assignment 𝑎 = (𝑞 , 𝑞 , 𝑞 , 𝑞 )

Q-Former Decoder

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

1D Queries

3D Queries

Query Initialization Quantized Token

[Figure 90]

- Figure 3. Architecture of 1D variable-length video tokenizer for EVATok. The input video is spatio-temporally patchified into 3D embeddings. According to a given assignment a, 1D variablelength query embeddings are initialized from these 3D embeddings. After Q-Former encoding and quantization, 1D discrete tokens are produced. Finally, 3D queries are initialized to reconstruct the video frames from the 1D tokens. training loss of our video tokenizer is:

|[Figure 91]<br><br>Max-proxy-reward Block4 Assignment<br><br>[Figure 92]<br><br>[Figure 93]<br><br>[Figure 94]<br><br>Block1<br><br>[Figure 95]<br><br>applies the proxy tokenizer to assignments with maximum proxy<br><br>on the curated dataset to predict the assignment for each|
|---|

[Figure 96]

(Videos, Optimal Assignments)

Lentropy is the entropy loss from [75] used for better codebook usage, with γ set empirically as 0.02.

###### 3.2. Stage 2: Dataset Curation for Router Training

With the proxy tokenizer, we can use it to assess the qualitycost trade-off performance of a specific assignment a for a video x, which will be illustrated in detail later. This means we can brute-force evaluate all the candidate assignments for x to find the optimal one. However, brute-force searching is undesirable due to the massive computational cost during adaptive video tokenization. Therefore, we aim to train a lightweight router that predicts optimal assignments in one pass. Towards this objective, we design stage 2 to curate a dataset to train such a lightweight router.

First, we illustrate how to evaluate the quality-cost tradeoff performance of a on x with the proxy tokenizer. We quantify this performance using the proxy reward:

Rproxy = wqQ(Eproxy,x,a) − wlL(a) (3) where Q(x,a) denotes the reconstruction quality of a for x, L(a) is the token length cost of a, and wq,wl are the weights reflecting preferences for better quality or less cost. For each x, its optimal assignment a∗ maximizes R, balancing token savings with minimal quality loss. Then, with this measurement, we resolve the challenging task of optimal assignment a∗ prediction by brute-force searching for the a with maximum proxy reward:

Ltotal = Lvqgan + λLalign + γLentropy (2)

with λ tuned as 0.7 by default. Here, Lvqgan combines l1 reconstruction loss Lrecon, perceptual loss Lpercp [27, 79], adversarial loss LGAN, and VQ codebook loss LVQ [16, 73];

a∗ = argmaxa∈ARproxy (4)

In practice, we collect 100k video clips from the diverse dataset WebVid-10M [5]. We record the reconstruction quality (LPIPS [79], PSNR, MSE) of each video clip under all candidate assignments. Then, with specified preference weights wq,wl, the proxy reward is calculated with Eq. 3 for all candidate assignments for each video. Specifically, we calculate Q(Eproxy,x,a) as the normalized LPIPS [79], and L(a) as the normalized length. Finally, only the assignment with the maximum proxy reward will be chosen for the video, resulting in a training dataset of 100k videos and their respective ground-truth assignments.

###### 3.3. Stage 3: Router Training

With the training dataset containing (video, optimal assignment) pairs from stage 2, we can train a lightweight router for one-pass optimal token assignment prediction in a classification formulation task. Our router adopts a ViT-like architecture [13] and is trained to classify input video x into one of the mT candidate assignment categories, which should be the optimal assignment for x. Given an input video, our router patchifies it to 3D visual embeddings, appends a [CLS] embedding. The router finally produces the probability of each candidate assignment a being the optimal one for the input video from the [CLS] embedding feature and is trained with cross-entropy loss.

###### 3.4. Stage 4: Adaptive Length Video Tokenizer

Integrating the router into our final video adaptive tokenization solution, we train an adaptive length video tokenizer conditioned on the token assignments predicted by the router. Specifically, the router predicts the optimal assignment for each video clip sample, which decides the token length and temporal distribution of the encoded token sequence. And the adaptive tokenizer learns to reconstruct each video using the predicted assignment from the router. Instead of combining the router with the proxy tokenizer as the final video adaptive tokenization solution, we choose to train a final tokenizer from scratch with the assignments from the tokenizer. This is to mitigate an issue in proxy tokenizer and prior video adaptive length tokenizers [33, 70]: the training-inference gap. For the proxy tokenizer, it is trained to encode videos across all mT possible assignments, yet during inference, only a few assignments might be used per video. This inefficiency in training can degrade tokenizer performance, as shown in Sec. 4.3, and is addressed by EVATok in our stage 4 training.

Different from the proxy tokenizer, which can suffice with a simpler training recipe, the final tokenizer training can further benefit from advanced training designs. Inspired by DINO [7, 42] discriminators in image tokenization [38, 52], in the final tokenizer training, we optionally employ video semantic encoders as discriminators for potentially better reconstruction and downstream AR gener-

ation. We use a frozen pretrained VideoMAE-B [53, 63] to process input videos and feed multi-layer features to trainable 1D CNN heads for fake/real logits. We avoid larger V-JEPA2 models [2] due to logit divergence instability risks [36] in adversarial training. We validate that a VideoMAE semantic discriminator, combined with video representation alignment, can significantly enhance both reconstruction and downstream AR generation quality for video tokenizers in Sec. 4.5.

#### 4. Experiments

###### 4.1. Settings

Dataset. We apply the commonly used combination of UCF [49] and K600 [8] datasets for video reconstruction and generation experiments. And for validation on more general data, we additionally experiment on WebVid10M [5] for video reconstruction. In all experiments, we use 16 × 128 × 128 video clips for training and evaluation. For router training, the video data is a randomly sampled subset of WebVid-10M, containing 100k video clips.

Implementation details. When patchifying videos in EVATok, the spatial downsample ratio is 8 and the temporal downsample ratio is 4. Therefore, a 16 × 128 × 128 video produces 4 × 16 × 16 features. When initializing 1D query embeddings, the candidate length of each temporal block is in {512,256,128,64,32}, so the number of all candidate assignments is 54 = 625. EVATok applies a codebook size of 16384 by default. But for the final tokenizers trained on UCF and K600 dataset, we use 8192 codebook size for fair comparison to previous methods [33, 59]. We train 19.9M ViT-S size routers in stage 3. We train Llama-like [50, 54] GPT models on variable length sequences. For class2video generation, the condition token corresponds to class labels, and for K600 frame prediction the conditions are tokens encoded from 8 frames padded from the 5 condition frames. Per-frame reconstruction fidelity metric LPIPS [79] and the overall distribution fidelity metric FVD [55] are used for evaluation. More details on the training and inference of the AR models can be found in the supplementary material.

###### 4.2. Validation on Quality-Cost Trade-off Curves

We validate the effect of our max-proxy-reward searching strategy and routers on proxy tokenizers. We compare these approaches against a commonly used baseline: fixed uniform assignment, which allocates the same number of tokens to different temporal blocks across videos. We plot quality-cost trade-off curves to show the overall trends of these assignment strategies under varying overall budgets.

To generate the quality-cost curves, we adjust the overall budgets to obtain multiple evaluation points. For the uniform assignment strategy, we vary the number of tokens allocated to each temporal block from 64 to 512. For the max-

Fixed Uniform Assign. Max-Proxy-Reward Assign. Router Assign.

WebVid: LPIPS

WebVid: rFVD

###### UCF: LPIPS

UCF: rFVD

0.2

| | | | | |
|---|---|---|---|---|
| | | | | |

180

0.2

180

0.18

0.18

0.16

0.16

100

100

0.14

90

90

LPIPS

LPIPS

0.14

rFVD

rFVD

80

80

70

0.12

70

0.12

60

60

50

0.1

50

0.1

40

0.09

500 1000 1500 2000

500 1000 1500 2000

500 1000 1500 2000

500 1000 1500 2000

Token Num.

Token Num.

Token Num.

Token Num.

- Figure 4. Quality-cost trade-off curves for different assignment strategies. By adaptively assigning token budgets to different temporal blocks across various videos, our max-proxy-reward strategy (green series) achieves superior performance under various overall budgets compared to the typical fixed uniform token assignment approach (red series). The router-based assignment (blue series) delivers performance close to that of the max-proxy-reward strategy on both WebVid and UCF datasets (the latter unseen during router training).

proxy-reward assignment strategy, we adjust wq in steps of 0.2 from 0.4 to 2.0, while setting wl = 2.0 − wq, thereby producing various wq,wl combinations that alter the overall budgets. For the router assignment strategy, we employ routers trained under different wq,wl combinations: wq ranging from 0.8 to 1.6 in steps of 0.2, with wl = 2.0 − wq. We evaluate on two benchmarks: the WebVid validation set and the UCF-101 training set, using two corresponding proxy tokenizers trained on either the WebVid training set or the UCF & K600 training sets.

As shown in Fig. 4, on both datasets, the max-proxyreward assignment strategy yields a quality-cost curve that achieves superior LPIPS and reconstruction FVD (rFVD) at equivalent overall budgets. Moreover, the routers closely align with the quality-cost curve of the max-proxy-reward strategy, demonstrating their ability to effectively simulate it. The routers also generalize well to different proxy tokenizers and datasets not seen during training, as evidenced in the last two columns of Fig. 4. The routers significantly reduce the total budget while delivering even better overall reconstruction quality. Focusing solely on rFVD, compared to traditional fixed uniform assignment approaches that allocate 1024 tokens on average to a 16 × 128 × 128 video clip, the routers can achieve 56% token savings on WebVid and 42% on UCF, with comparable or even better performance. We validate the performance and generalization of the routers on proxy tokenizers. Next, we demonstrate the advantages of using routers to guide final tokenizer training.

###### 4.3. Validation on Final Adaptive Tokenizer

We validate the benefits of using routers to provide optimal assignment guidance in both the training and inference of adaptive video tokenizers, which bridges the traininginference gap in previous adaptive length video tokenizers [33, 70]. For the choice of router, we choose the router conditioned on wq = 1.2,wl = 0.8, which achieves compa-

Settings PSNR↑ LPIPS↓ rFVD↓ #rTokens↓

- A1. Uniform (Proxy Tok.) 27.26 0.1178 73 1024
- A2. Uniform (Final Tok.) 27.77 0.1056 63 1024

- A2 + VideoMAE Disc. 26.68 0.1197 13 1024

- B1. Router (Proxy Tok.) 27.05 0.1182 50 721(-29.6%)
- B2. Router (Final Tok.) 27.68 0.1068 33 721(-29.6%)

- B2 + VideoMAE Disc. 26.90 0.1144 9.2 721(-29.6%)

- Table 1. Final tokenizer validation on WebVid. The tokenizers are trained for 400k iterations. With the router, final tokenizers achieve comparable LPIPS and better rFVD with 29.6% saving in token length (row A2 vs. B2 and row A2+ vs. B2+). Final tokenizers outperform proxy tokenizers with the same training efforts (row A2 vs. A1, B2 vs. row B1), showing the importance of bridging the training-inference gap for variable-length tokenizers.

Settings LPIPS↓ rFVD↓ #rTokens↓ gFVD↓ #gTokens↓ Uniform (Final) 0.1303 13 1024 98 1024 Router (Final) 0.1212 13 774(-24.4%) 96 740(-27.7%)

- Table 2. Final tokenizer validation on UCF. The final tokenizer with router beats the fixed uniform assignment baseline in both reconstruction and downstream AR generation, while saving 24.4% and 27.7% token length separately.

rable LPIPS and better rFVD with 24.4% token length saving against the uniform assignment. We conduct two sets of experiments on WebVid-10M and the combination of UCF & K600 separately.

On the WebVid-10M dataset, with the assignments provided the a router, we train two variants of final tokenizers, one not using the VideoMAE discriminator while the other uses it. As baselines, with fixed uniform assignments of 1024 tokens per video, we train two variants of final tokenizers correspondingly. And all tokenizers evaluated in this experiment are trained for 400k iterations for fair comparison. As shown in Tab. 1, on the WebVid validation set,

###### Method #Params #rTokens rFVD↓ gFVD↓ #gTokens

Tokenizer Generator K600 UCF UCF Diffusion-based generative models with continuous video tokenizers

VideoFusion [37] - 2B - - - 173 HPDM [48] - 725M - - - 66 W.A.L.T-L [20] - 313M - - 3.3 46 -

MLM generative models with discrete video tokenizers

MAGVIT-MLM [74] 158M 306M 1024 25 9.9 76 1024 MAGVIT-v2-MLM [75] - 307M 1280 8.6 4.3 58 1280

AR generative models with discrete video tokenizers

CogVideo [23] - 9.4B - - 109.2 626 TATS [18] 32M 321M 1024 162 - 332 1024 MAGVIT-AR [74] 158M 306M 1024 25 - 265 1024 MAGVIT-v2-AR [75] - 840M 1280 8.6 - 109 1280 OmniTokenizer [60] 82.2M 650M 1280 42 32.9 191 1280 AdapTok [33] 195M 633M 1024 36 11 67 1024 LARP-L-Long [59] 173M 343M 1024 20 6.2 102 1024 LARP-L-Long [59] 173M 632M 1024 20 5.1 57 1024 EVATok (ours) 145M 327M 774(-24.4%) 9.7 4.6 62 756(-26.2%) EVATok (ours) 145M 633M 774(-24.4%) 9.7 4.0 48 756(-26.2%)

Table 3. System-level comparison for tokenizers and downstream generation models. EVATok achieves superior performances in UCF101 video reconstruction, downstream class-to-video generation and K600 frame prediction, while saving 24.4% tokens in reconstruction and 26.2% tokens in UCF class-to-video generation.

we validate that for final tokenizers, the router-guided tokenizer achieves comparable LPIPS, better rFVD and 29.6% token length savings against the tokenizer trained with uniform assignments, no matter using VideoMAE discriminator or not (row A2 vs. B2 and A2+ vs. B2+). Besides, Tab. 1 also indicates that the final tokenizers outperform the proxy tokenizers with the same training iterations (row A2 vs. A1 and B2 vs. B1), proving that using optimal assignments in both tokenizer training and inference benefits performances.

On UCF & K600 dataset, we train two tokenizers with fixed uniform assignment or router-guided assignment, both using the VideoMAE discriminator. As shown in Tab. 2, under the predicted optimal assignment, the final tokenizer trained with the router achieves even better reconstruction with 24.4% savings in token length. Further, we train 99M GPT-B AR generation models on the two tokenizers separately on the UCF-101 class2video dataset, and evaluate them by generation FVD (gFVD) based on 10k generated samples. The AR model that adaptively decides the length of tokens it generates achieves even better gFVD while generating 740 tokens per video on average, saving 27.7% of tokens compared to the fixed-length AR model.

By training and evaluating final tokenizers with the adaptive assignments of our router, we show that a router helps

train a better adaptive tokenizer by eliminating the previous training-inference gap, beating baselines trained with fixed uniform assignment. And importantly, for the first time, we show that downstream AR models trained on adaptive length video token sequences can achieve better overall generation quality with significant savings in token length cost.

###### 4.4. System-Level Comparison

We compare EVATok with previous video generative models, evaluating performance in terms of video reconstruction, generation quality, and average token length. These aspects are assessed using the UCF-101 reconstruction, UCF101 class-to-video generation, and K600 frame prediction benchmarks. As shown in Tab. 3, EVATok achieves substantially better reconstruction FVD (rFVD) while reducing token length by 24.4% compared to LARP [59]. EVATok es-

Methods Tok. Param. #gTokens AR Param. gFVD↓

AdapTok [33] 195M 1024 633M 11 LARP [59] 173M 1024 632M 5.1 EVATok (ours) 145M 862 (-15.8%) 633M 4.0

Table 4. K600 frame prediction comparison. In similar settings, EVATok performs the best with 15.8% less generated tokens.

Configuration PSNR↑ LPIPS↓ rFVD↓ gFVD↓

|Final Recipe (Uniform) 25.05 0.1303 13|98<br><br>|
|---|---|
|- VideoMAE [53, 63] Disc. 26.21 0.1097 65<br>- V-JEPA2 [2] Align. 25.30 0.1253 18<br>- Both 26.41 0.1095 80<br>|155 144 230|

Table 5. Ablation study for video representation alignment and video semantic discriminator. Removing either design will lead to degradation in rFVD and downstream gFVD.

tablishes a new state-of-the-art (SOTA) on UCF-101 classto-video generation, with a generation FVD (gFVD) of 48 and 26.2% fewer tokens than the previous SOTA method, LARP [59]. EVATok also delivers the best results on K600 frame prediction. For a fair comparison of generation efficiency on K600 frame prediction, we benchmark against AdapTok [33] and LARP [59] in Tab. 4, as we employ the same approach of additionally generating conditioning frames during both training and inference. We fix the token length for conditioning frames as 512 + 128 = 640, therefore, during frame prediction training, we choose assignments with (512,128) prefix for the first two temporal blocks that have the highest probability predicted by the router, for the encoding of 16-frame samples. Under this setting, EVATok achieves the best gFVD on K600 frame prediction while saving 15.8% in generated tokens.

We demonstrate that adaptive video tokenization, combined with our advanced training recipe, enables EVATok to achieve both high efficiency and leading performance on video reconstruction and downstream AR generation.

###### 4.5. Ablation Study

Threshold vs. max-proxy-reward searching. For adaptive video tokenization, ElasticTok [70] applies a heuristic method to find adaptive assignments: searching the minimum token length to be kept that maintains the reconstruction quality above certain thresholds. This heuristic threshold-based method is not designed to optimize for the overall quality-cost trade-off. To compare this thresholdbased method to our max-proxy-reward strategy, we implement a similar baseline method in our setting, which finds the assignment with the minimum token length and satisfies a certain LPIPS threshold for each video. If no assignment can satisfy the threshold, the maximum length assignment will be chosen. Testing on our proxy tokenizer, by varying the LPIPS threshold, we can plot the quality-cost trade-off curve for this baseline strategy. As shown in Fig. 5, on the WebVid validation set, while the threshold-based baseline improves the rFVD compared to uniform assignment, it still lags behind our max-proxy-reward strategy.

Video semantic encoder for video tokenizers. Video semantic encoders can help video tokenizer training in two ways: (1) providing representation alignment; (2) giving

LPIPS

rFVD

###### PSNR

180

| || |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
| | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

| |
|---|

0.2

- 23
- 24
- 25
- 26
- 27
- 28

| |
|---|

| |
|---|

0.18

| |
|---|

| |
|---|

0.16

100

| |
|---|

| |
|---|

LPIPS

PSNR

90

0.14

rFVD

| |
|---|

| |
|---|

80

| |
|---|

70

| |
|---|

0.12

| |
|---|

60

| |
|---|

50

0.1

| |
|---|

| |
|---|

| |
|---|

1000 2000

1000 2000

1000 2000

Token Num.

Token Num.

Token Num.

Fixed Uniform Assign. Max-Proxy-Reward Assign. Threshold Assign.

Figure 5. Quality-cost curve: threshold based vs. max-proxyreward vs. uniform assignment. While threshold-based assignment improves rFVD against uniform assignment, it underperforms our max-proxy-reward strategy.

perceptual feedback as the discriminator for better reconstruction quality. We study the two designs and reveal that they are both important for reconstruction and downstream generation for video tokenizers. Here, for simplicity, we utilize the typical uniform assignment and train video tokenizers under different recipes on the UCF&K600 dataset for 400k iterations, and evaluate them by reconstruction and downstream GPT-B model generation on UCF-101. As shown in Tab. 5, removing either representation alignment or VideoMAE discriminator can lead to degradation of rFVD and downstream gFVD. While we do recognize a drop in the per-frame fidelity metric PSNR and LPIPS brought especially by the VideoMAE discriminator, in our qualitative checking, we identify that the drop in PSNR and LPIPS is traded for less blurriness and less temporal flickering, which corresponds to higher rFVD. More qualitative comparisons are in the supplementary material.

#### 5. Conclusions

In this work, we propose EVATok, a content-adaptive video tokenization framework to efficiently assign tokens across different temporal blocks and videos. We introduce proxy reward as a novel metric for finding the optimal assignments with the best quality-cost trade-off. By reformulating optimal assignment selection for video tokenization as a maximum proxy reward assignment classification task, we can curate supervision datasets to train routers to map each video to its optimal assignment. These routers help us train video adaptive tokenizers and downstream autoregressive (AR) video generative models with efficient token assignments. Enhanced by our advanced recipe incorporating video semantic encoders in tokenizer training, EVATok achieves superior reconstruction and downstream AR generation quality while significantly saving token length cost. EVATok has presented promising results on 16-frame videos in this work, and for future development, can potentially achieve higher efficiency for videos with longer duration. Please refer to the supplementary material for more discussions on future work and limitations of EVATok.

#### Acknowledgments

This work is supported in part by the Research Grant Council of Hong Kong through the NSFC-RGC Joint Research Scheme under grant N HKU769/25.

#### References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774,

2023. 1

- [2] Mido Assran, Adrien Bardes, David Fan, Quentin Garrido, Russell Howes, Matthew Muckley, Ammar Rizvi, Claire Roberts, Koustuv Sinha, Artem Zholus, et al. V-jepa 2: Selfsupervised video models enable understanding, prediction and planning. arXiv preprint arXiv:2506.09985, 2025. 2, 3, 5, 8
- [3] Roman Bachmann, Jesse Allardice, David Mizrahi, Enrico Fini, O˘guzhan Fatih Kar, Elmira Amirloo, Alaaeldin ElNouby, Amir Zamir, and Afshin Dehghan. Flextok: Resampling images into 1d token sequences of flexible length. arXiv preprint arXiv:2502.13967, 2025. 2
- [4] Zechen Bai, Jianxiong Gao, Ziteng Gao, Pichao Wang, Zheng Zhang, Tong He, and Mike Zheng Shou. Factorized visual tokenization and generation. arXiv preprint arXiv:2411.16681, 2024. 3
- [5] Max Bain, Arsha Nagrani, G¨ul Varol, and Andrew Zisserman. Frozen in time: A joint video and image encoder for end-to-end retrieval. In IEEE International Conference on Computer Vision, 2021. 5, 2
- [6] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-toend object detection with transformers. In European conference on computer vision, pages 213–229. Springer, 2020. 2, 3
- [7] Mathilde Caron, Hugo Touvron, Ishan Misra, Herv´e J´egou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9650–9660, 2021. 5
- [8] Joao Carreira, Eric Noland, Andras Banki-Horvath, Chloe Hillier, and Andrew Zisserman. A short note about kinetics-

600. 2018. 5

- [9] Xiaokang Chen, Zhiyu Wu, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, and Chong Ruan. Januspro: Unified multimodal understanding and generation with data and model scaling. arXiv preprint arXiv:2501.17811,

2025. 1

- [10] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025. 1
- [11] Yufeng Cui, Honghao Chen, Haoge Deng, Xu Huang, Xinghang Li, Jirong Liu, Yang Liu, Zhuoyan Luo, Jinsheng

- Wang, Wenxuan Wang, Yueze Wang, Chengyuan Wang, Fan Zhang, Yingli Zhao, Ting Pan, Xianduo Li, Zecheng Hao, Wenxuan Ma, Zhuo Chen, Yulong Ao, Tiejun Huang, Zhongyuan Wang, and Xinlong Wang. Emu3.5: Native multimodal models are world learners, 2025. 1, 3
- [12] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in neural information processing systems, 34:8780–8794, 2021. 10
- [13] Alexey Dosovitskiy. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020. 5, 10
- [14] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783,

2024. 1

- [15] Shivam Duggal, Phillip Isola, Antonio Torralba, and William T. Freeman. Adaptive length image tokenization via recurrent allocation. In The Thirteenth International Conference on Learning Representations, 2025. 2
- [16] Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12873–12883, 2021. 1, 2, 4
- [17] NVIDIA et. al. Cosmos world foundation model platform for physical ai. arXiv preprint arXiv:2501.03575, 2025. 1
- [18] Songwei Ge, Thomas Hayes, Harry Yang, Xi Yin, Guan Pang, David Jacobs, Jia-Bin Huang, and Devi Parikh. Long video generation with time-agnostic vqgan and timesensitive transformer. In European Conference on Computer Vision, pages 102–118. Springer, 2022. 2, 7
- [19] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025. 1
- [20] Agrim Gupta, Lijun Yu, Kihyuk Sohn, Xiuye Gu, Meera Hahn, Fei-Fei Li, Irfan Essa, Lu Jiang, and Jos´e Lezama. Photorealistic video generation with diffusion models. In European Conference on Computer Vision, pages 393–411. Springer, 2024. 7
- [21] Alexander H¨agele, Elie Bakouch, Atli Kosson, Loubna Ben Allal, Leandro Von Werra, and Martin Jaggi. Scaling laws and compute-optimal training beyond fixed training durations. arXiv preprint arXiv:2405.18392, 2024. 2
- [22] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017. 8, 10
- [23] Wenyi Hong, Ming Ding, Wendi Zheng, Xinghan Liu, and Jie Tang. Cogvideo: Large-scale pretraining for text-to-video generation via transformers. arXiv preprint arXiv:2205.15868, 2022. 7
- [24] Mengqi Huang, Zhendong Mao, Zhuowei Chen, and Yongdong Zhang. Towards accurate image coding: Improved autoregressive image generation with dynamic vector quantiza-

- tion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22596–22605, 2023. 2
- [25] Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, and Alexei A Efros. Image-to-image translation with conditional adversarial networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1125–1134,

2017. 3

- [26] Eric Jang, Shixiang Gu, and Ben Poole. Categorical reparameterization with gumbel-softmax. arXiv preprint arXiv:1611.01144, 2016. 2
- [27] Justin Johnson, Alexandre Alahi, and Li Fei-Fei. Perceptual losses for real-time style transfer and super-resolution. In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11-14, 2016, Proceedings, Part II 14, pages 694–711. Springer, 2016. 4
- [28] Diederik P Kingma. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013. 1
- [29] Diederik P Kingma, Max Welling, et al. An introduction to variational autoencoders. Foundations and Trends® in Machine Learning, 12(4):307–392, 2019. 1
- [30] Dan Kondratyuk, Lijun Yu, Xiuye Gu, Jose Lezama, Jonathan Huang, Grant Schindler, Rachel Hornung, Vighnesh Birodkar, Jimmy Yan, Ming-Chang Chiu, Krishna Somandepalli, Hassan Akbari, Yair Alon, Yong Cheng, Joshua V. Dillon, Agrim Gupta, Meera Hahn, Anja Hauth, David Hendon, Alonso Martinez, David Minnen, Mikhail Sirotenko, Kihyuk Sohn, Xuan Yang, Hartwig Adam, MingHsuan Yang, Irfan Essa, Huisheng Wang, David A Ross, Bryan Seybold, and Lu Jiang. Videopoet: A large language model for zero-shot video generation. In Proceedings of the 41st International Conference on Machine Learning, 2024. 1
- [31] Doyup Lee, Chiheon Kim, Saehoon Kim, Minsu Cho, and Wook-Shin Han. Autoregressive image generation using residual quantization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11523–11532, 2022. 2
- [32] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In International conference on machine learning, pages 19730–

19742. PMLR, 2023. 2, 3

- [33] Yan Li, Changyao Tian, Renqiu Xia, Ning Liao, Weiwei Guo, Junchi Yan, Hongsheng Li, Jifeng Dai, Hao Li, and Xue Yang. Learning adaptive and temporally causal video tokenization in a 1d latent space. 2025. 2, 3, 5, 6, 7, 8, 1
- [34] Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437, 2024. 1
- [35] Dongyang Liu, Shitian Zhao, Le Zhuo, Weifeng Lin, Yu Qiao, Hongsheng Li, and Peng Gao. Lumina-mgpt: Illuminate flexible photorealistic text-to-image generation with multimodal generative pretraining. arXiv preprint arXiv:2408.02657, 2024. 1
- [36] Jiasen Lu, Liangchen Song, Mingze Xu, Byeongjoo Ahn, Yanjun Wang, Chen Chen, Afshin Dehghan, and Yinfei

- Yang. Atoken: A unified tokenizer for vision. arXiv preprint arXiv:2509.14476, 2025. 5
- [37] Zhengxiong Luo, Dayou Chen, Yingya Zhang, Yan Huang, Liang Wang, Yujun Shen, Deli Zhao, Jingren Zhou, and Tieniu Tan. Videofusion: Decomposed diffusion models for high-quality video generation. arXiv preprint arXiv:2303.08320, 2023. 7
- [38] Chuofan Ma, Yi Jiang, Junfeng Wu, Jihan Yang, Xin Yu, Zehuan Yuan, Bingyue Peng, and Xiaojuan Qi. Unitok: A unified tokenizer for visual generation and understanding. arXiv preprint arXiv:2502.20321, 2025. 3, 5
- [39] Lingjun Mao, Rodolfo Corona, Xin Liang, Wenhao Yan, and Zineng Tang. Images are worth variable length of representations. arXiv preprint arXiv:2506.03643, 2025. 2, 3, 10
- [40] Fabian Mentzer, David Minnen, Eirikur Agustsson, and Michael Tschannen. Finite scalar quantization: Vq-vae made simple. arXiv preprint arXiv:2309.15505, 2023. 2
- [41] Keita Miwa, Kento Sasaki, Hidehisa Arai, Tsubasa Takahashi, and Yu Yamaguchi. One-d-piece: Image tokenizer meets quality-controllable compression. 2025. 2, 3
- [42] Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023. 3, 5
- [43] Liao Qu, Huichao Zhang, Yiheng Liu, Xu Wang, Yi Jiang, Yiming Gao, Hu Ye, Daniel K Du, Zehuan Yuan, and Xinglong Wu. Tokenflow: Unified image tokenizer for multimodal understanding and generation. arXiv preprint arXiv:2412.03069, 2024. 1
- [44] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021. 3
- [45] Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, et al. Imagenet large scale visual recognition challenge. International journal of computer vision, 115:211–252, 2015. 8
- [46] Junhong Shen, Kushal Tirumala, Michihiro Yasunaga, Ishan Misra, Luke Zettlemoyer, Lili Yu, and Chunting Zhou. Cat: Content-adaptive image tokenization. 2025. 2, 10
- [47] Oriane Sim´eoni, Huy V Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Micha¨el Ramamonjisoa, et al. Dinov3. arXiv preprint arXiv:2508.10104, 2025. 9
- [48] Ivan Skorokhodov, Willi Menapace, Aliaksandr Siarohin, and Sergey Tulyakov. Hierarchical patch diffusion models for high-resolution video generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7569–7579, 2024. 7
- [49] Khurram Soomro, Amir Roshan Zamir, and Mubarak Shah. Ucf101: A dataset of 101 human actions classes from videos in the wild. 2012. 5
- [50] Peize Sun, Yi Jiang, Shoufa Chen, Shilong Zhang, Bingyue Peng, Ping Luo, and Zehuan Yuan. Autoregressive model

- beats diffusion: Llama for scalable image generation. 2024. 5, 10
- [51] Anni Tang, Tianyu He, Junliang Guo, Xinle Cheng, Li Song, and Jiang Bian. Vidtok: A versatile and open-source video tokenizer. arXiv preprint arXiv:2412.13061, 2024. 1
- [52] Keyu Tian, Yi Jiang, Zehuan Yuan, BINGYUE PENG, and Liwei Wang. Visual autoregressive modeling: Scalable image generation via next-scale prediction. In The Thirtyeighth Annual Conference on Neural Information Processing Systems, 2024. 2, 5
- [53] Zhan Tong, Yibing Song, Jue Wang, and Limin Wang. Videomae: Masked autoencoders are data-efficient learners for self-supervised video pre-training. Advances in neural information processing systems, 35:10078–10093, 2022. 2, 3, 5, 8, 7
- [54] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023. 5
- [55] Thomas Unterthiner, Sjoerd Van Steenkiste, Karol Kurach, Raphael Marinier, Marcin Michalski, and Sylvain Gelly. Towards accurate generative models of video: A new metric & challenges. arXiv preprint arXiv:1812.01717, 2018. 5
- [56] Aaron Van Den Oord, Oriol Vinyals, et al. Neural discrete representation learning. Advances in neural information processing systems, 30, 2017. 2
- [57] Ruben Villegas, Mohammad Babaeizadeh, Pieter-Jan Kindermans, Hernan Moraldo, Han Zhang, Mohammad Taghi Saffar, Santiago Castro, Julius Kunze, and Dumitru Erhan. Phenaki: Variable length video generation from open domain textual description. arXiv preprint arXiv:2210.02399, 2022. 2
- [58] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, Jianyuan Zeng, Jiayu Wang, Jingfeng Zhang, Jingren Zhou, Jinkai Wang, Jixuan Chen, Kai Zhu, Kang Zhao, Keyu Yan, Lianghua Huang, Mengyang Feng, Ningyi Zhang, Pandeng Li, Pingyu Wu, Ruihang Chu, Ruili Feng, Shiwei Zhang, Siyang Sun, Tao Fang, Tianxing Wang, Tianyi Gui, Tingyu Weng, Tong Shen, Wei Lin, Wei Wang, Wei Wang, Wenmeng Zhou, Wente Wang, Wenting Shen, Wenyuan Yu, Xianzhong Shi, Xiaoming Huang, Xin Xu, Yan Kou, Yangyu Lv, Yifei Li, Yijing Liu, Yiming Wang, Yingya Zhang, Yitong Huang, Yong Li, You Wu, Yu Liu, Yulin Pan, Yun Zheng, Yuntao Hong, Yupeng Shi, Yutong Feng, Zeyinzi Jiang, Zhen Han, Zhi-Fan Wu, and Ziyu Liu. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025. 1
- [59] Hanyu Wang, Saksham Suri, Yixuan Ren, Hao Chen, and Abhinav Shrivastava. Larp: Tokenizing videos with a learned autoregressive generative prior. In ICLR, 2025. 2, 3, 5, 7, 8, 1
- [60] Junke Wang, Yi Jiang, Zehuan Yuan, BINGYUE PENG, Zuxuan Wu, and Yu-Gang Jiang. Omnitokenizer: A joint image-video tokenizer for visual generation. In The Thirtyeighth Annual Conference on Neural Information Processing Systems, 2024. 2, 7

- [61] Lingfeng Wang, Hualing Lin, Senda Chen, Tao Wang, Changxu Cheng, Yangyang Zhong, Dong Zheng, and Wuyue Zhao. Alto: Adaptive-length tokenizer for autoregressive mask generation. 2025. 2
- [62] Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Zhen Li, Qiying Yu, et al. Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869, 2024. 1, 2
- [63] Yi Wang, Kunchang Li, Yizhuo Li, Yinan He, Bingkun Huang, Zhiyu Zhao, Hongjie Zhang, Jilan Xu, Yi Liu, Zun Wang, et al. Internvideo: General video foundation models via generative and discriminative learning. arXiv preprint arXiv:2212.03191, 2022. 5, 8, 7
- [64] Yuqing Wang, Tianwei Xiong, Daquan Zhou, Zhijie Lin, Yang Zhao, Bingyi Kang, Jiashi Feng, and Xihui Liu. Loong: Generating minute-level long videos with autoregressive language models. arXiv preprint arXiv:2410.02757, 2024. 1, 2
- [65] Junfeng Wu, Yi Jiang, Chuofan Ma, Yuliang Liu, Hengshuang Zhao, Zehuan Yuan, Song Bai, and Xiang Bai. Liquid: Language models are scalable multi-modal generators. arXiv preprint arXiv:2412.04332, 2024. 1
- [66] Yecheng Wu, Zhuoyang Zhang, Junyu Chen, Haotian Tang, Dacheng Li, Yunhao Fang, Ligeng Zhu, Enze Xie, Hongxu Yin, Li Yi, et al. Vila-u: a unified foundation model integrating visual understanding and generation. arXiv preprint arXiv:2409.04429, 2024. 1
- [67] Yi Xin, Juncheng Yan, Qi Qin, Zhen Li, Dongyang Liu, Shicheng Li, Victor Shea-Jay Huang, Yupeng Zhou, Renrui Zhang, Le Zhuo, et al. Lumina-mgpt 2.0: Standalone autoregressive image modeling. arXiv preprint arXiv:2507.17801, 2025. 1
- [68] Tianwei Xiong, Jun Hao Liew, Zilong Huang, Jiashi Feng, and Xihui Liu. Gigatok: Scaling visual tokenizers to 3 billion parameters for autoregressive image generation. 2025. 2, 3, 8
- [69] Wanghan Xu, Xiaoyu Yue, Zidong Wang, Yao Teng, Wenlong Zhang, Xihui Liu, Luping Zhou, Wanli Ouyang, and Lei Bai. Exploring representation-aligned latent space for better generation. arXiv preprint arXiv:2502.00359, 2025. 3
- [70] Wilson Yan, Volodymyr Mnih, Aleksandra Faust, Matei Zaharia, Pieter Abbeel, and Hao Liu. Elastictok: Adaptive tokenization for image and video. In The Thirteenth International Conference on Learning Representations, 2025. 2, 3, 5, 6, 8
- [71] Jingfeng Yao and Xinggang Wang. Reconstruction vs. generation: Taming optimization dilemma in latent diffusion models. arXiv preprint arXiv:2501.01423, 2025. 3
- [72] Haotian Ye, Qiyuan He, Jiaqi Han, Puheng Li, Jiaojiao Fan, Zekun Hao, Fitsum Reda, Yogesh Balaji, Huayu Chen, Sheng Liu, Angela Yao, James Zou, Stefano Ermon, Haoxiang Wang, and Ming-Yu Liu. Infotok: Adaptive discrete video tokenizer via information-theoretic compression. In The Fourteenth International Conference on Learning Representations, 2026. 3
- [73] Jiahui Yu, Xin Li, Jing Yu Koh, Han Zhang, Ruoming Pang, James Qin, Alexander Ku, Yuanzhong Xu, Jason Baldridge, and Yonghui Wu. Vector-quantized image modeling with

- improved vqgan. arXiv preprint arXiv:2110.04627, 2021. 1, 4
- [74] Lijun Yu, Yong Cheng, Kihyuk Sohn, Jos´e Lezama, Han Zhang, Huiwen Chang, Alexander G Hauptmann, MingHsuan Yang, Yuan Hao, Irfan Essa, et al. Magvit: Masked generative video transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10459–10469, 2023. 7
- [75] Lijun Yu, Jos´e Lezama, Nitesh B Gundavarapu, Luca Versari, Kihyuk Sohn, David Minnen, Yong Cheng, Vighnesh Birodkar, Agrim Gupta, Xiuye Gu, et al. Language model beats diffusion–tokenizer is key to visual generation. arXiv preprint arXiv:2310.05737, 2023. 1, 2, 4, 7
- [76] Qihang Yu, Mark Weber, Xueqing Deng, Xiaohui Shen, Daniel Cremers, and Liang-Chieh Chen. An image is worth 32 tokens for reconstruction and generation. arXiv preprint arXiv:2406.07550, 2024. 2
- [77] Sihyun Yu, Sangkyung Kwak, Huiwon Jang, Jongheon Jeong, Jonathan Huang, Jinwoo Shin, and Saining Xie. Representation alignment for generation: Training diffusion transformers is easier than you think. arXiv preprint arXiv:2410.06940, 2024. 3
- [78] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF international conference on computer vision, pages 11975–11986, 2023. 3
- [79] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 586–595, 2018. 4, 5
- [80] Xiangdong Zhang, Jiaqi Liao, Shaofeng Zhang, Fanqing Meng, Xiangpeng Wan, Junchi Yan, and Yu Cheng. Videorepa: Learning physics for video generation through relational alignment with foundation models. arXiv preprint arXiv:2505.23656, 2025. 3

## EVATok: Adaptive Length Video Tokenization for Efficient Visual Autoregressive Generation

### Supplementary Material

#### Content of the Appendix

This supplementary material includes the following content:

- • Sec. F discusses the limitations of EVATok.
- • Sec. G provides the plan for the future work.
- • Sec. H gives the detailed implementation of the four-stage framework of EVATok and downstream adaptive length AR models.
- • Sec. I provides the reconstruction performances of our final tokenizers, including qualitative examples for the adaptive length reconstruction and generation, as well as cases of how VideoMAE discriminator affects video reconstruction perceptually.
- • Sec. J provides the compute cost analysis for our fourstage framework and downstream AR generation model training.
- • Sec. L analyzes the router’s max-proxy-reward assignment predictions in terms of accuracy and proxy reward.
- • Sec. K explains the attention mask mechanism in our QFormer style video adaptive tokenizers.
- • Sec. M shows the results for translating the solution of EVATok to image adaptive tokenization.

#### F. Limitations

In this work, we focus on addressing the key challenge in adaptive length video tokenization: identifying the optimal assignment. Although we have demonstrated the superiority of our method in video reconstruction, as well as downstream autoregressive (AR) class-to-video generation and frame prediction tasks, our experiments were limited to 16 × 128 × 128 video clips. We did not evaluate it on videos with higher resolution or longer duration that align with industry-level requirements [11, 58]. Additionally, due to limited computational resources, we have not validated EVATok on more complex downstream tasks, such as textto-video generation.

Furthermore, when extending video duration in adaptive length video tokenizers, the number of possible assignment choices can grow exponentially if exhaustive searching is naively applied. Although this issue is not addressed in the current work, we discuss a potential solution in our future work section (Sec. G), which can reduce the complexity of optimal assignment searching from O(mt) to around O(t2) with respect to the maximum video duration t.

#### G. Future Work

Adaptive length video tokenization on longer videos. In the main paper, when identifying the optimal assignment for a video clip with T temporal blocks and m possible token number choices for each temporal block, we search for mT possible assignments to find the optimal one. This approach will become unaffordable for larger T. To address this, in future work, we will explore a method that searches for optimal assignments approximately in an autoregressive way. For example, for a video with 2T temporal blocks, we can first search the mT possible assignments for the first T blocks, then, based on the optimal assignment for the first T blocks, we continue to search the mT possible assignments for the T blocks. Therefore, if we assume the reconstruction cost for the proxy tokenizer increases linearly with longer T, then the complexity for optimal assignment searching is estimated to be O(T2).

Extension to adaptive length video VAE and diffusion models. The idea of adaptive length video tokenization is not limited to discrete tokenizers, and can naturally transfer to adaptive length VAE [28, 29] training. While it is natural for AR models to learn on variable length sequences, the performance of diffusion models on denoising adaptive length sequences can be discussed in future work.

Router improvements. In our current implementation, the preference weights are implicitly fixed during the process of training data curation for routers. In the future work, we may want the preference weights to be able to be input explicitly for the routers for more flexible applications.

#### H. Implementation Details

Tokenizer training. On WebVid-10M, we train variable length tokenizer on 3 FPS video frames, following the approaches in VidTok [51]. On UCF & K600 dataset, we train tokenizers on video frames with their original FPS, following typical settings. We use a cosine learning rate schedule. The maximum learning rate is 1 × 10−4 and end learning rate is 1×10−6. The batch size is 128, and proxy tokenizers are all trained for only 400k iterations before being used for proxy reward calculation. The final video adaptive tokenizers are trained for 1000k iterations, whose training cost is aligned with previous work [33, 59].

Proxy reward calculation. For proxy reward calculation from the main paper:

###### Rproxy = wqQ(Eproxy,x,a) − wlL(a) (5)

Specifically, we calculate Q(Eproxy,x,a) as:

LPIPS(Eproxy(x,a),x) − MEANLPIPS STDLPIPS

Q(Eproxy,x,a) =

(6) where LPIPS(Eproxy(x,a),x) is the LPIPS value between original video x and the reconstruction result Eproxy(x,a) using proxy tokenizer Eproxy under assignment a. MEANLPIPS denotes the expectation of Eproxy(x,a) for randomly sampled x from all the training videos and randomly sampled a from all candidate assignments, and STDLPIPS represents the standard deviation of Eproxy(x,a) for random x and a. We choose LPIPS for per-video reconstruction quality measurement, as it is a metric designed to better align with human perception. We calculate L(a) as:

T k=1 a[k] − MEANL

L(a) =

STDL

(7)

where Tk=1 a[k] is the sum of the allocated tokens across all T temporal blocks. MEANL and STDL are the expectation of Tk=1 a[k] for randomly sampled a.

Router training. We train 19.9M ViT-S size routers with a batch size of 128 for 50k iterations. We optionally use frozen V-JEPA2 [2] to patchify raw frames into video embeddings. Otherwise, we use the typical learnable linear projection for patch embeddings. In practice, we find that there is no obvious performance gap between these two visual embedding strategies.

AR model training. For the adaptive length token sequences produced by EVATok, before each temporal block, a special token indicating the number of tokens for the upcoming temporal block will be inserted for AR training. Therefore, for AR inference, before generating the tokens of the next temporal block, the AR model first predicts the special tokens indicating the length of the next block. On UCF-101, AR models are trained for 3000 epochs using WSD [21] learning rate schedule, where the learning rate is kept constant for the first 80% of the training and quickly annealed to 0 in the rest 20% training iterations. On K600, the AR models are trained for 75 epochs with the same learning rate scheduler.

AR inference for adaptive length video generation. In adaptive length AR generation, we observe that even with a special token preceding each temporal block to indicate the number of tokens in the upcoming block, the AR model may still occasionally produce unexpected tokens during inference (e.g., sampling a special token when a visual token is expected, or vice versa). To ensure the model generates precisely the number of tokens specified by the preceding special token for each temporal block, we employ a logitmasking strategy. For instance, when sampling the first token of a variable length sequence, which is expected to be a special token denoting the token count for the initial

Settings / Dataset PSNR↑ LPIPS↓ rFVD↓ #rTokens↓

Final w/ VideoMAE Disc. (WebVid) 27.37 0.1063 7.3 721 Final w/o VideoMAE Disc. (WebVid) 28.18 0.0983 32 721

Final w/ VideoMAE Disc. (UCF&K600) 25.75 0.1140 9.7 774

Table 6. The detailed performances of final tokenizers reconstruction results. All models are trained for the full 1000k iterations. The tokenizers trained on WebVid-10M are evaluated on the WebVid validation set, and the tokenizers trained on UCF-101 and K600 are evaluated on UCF-101 training set.

block, all logit entries corresponding to visual tokens are masked to −inf before the softmax operation. This guarantees that only a special token is sampled. Subsequently, for the next k tokens (as indicated by the special token), the logits for special tokens are masked, ensuring only visual tokens are generated. This process continues until m special tokens and their corresponding temporal blocks are generated. This approach incurs nearly no additional computational overhead and guarantees the generated variable length sequence maintains the correct structure. We use constant classifier free guidance (CFG) schedules for AR model inference during class-to-video sampling. For GPTB models, the CFG value is 2.5, for larger GPT models, we use 1.75 CFG value.

#### I. More Results and Qualitative Analysis

In this section, we present the full metrics of our final adaptive tokenizers on reconstruction in Tab. 6, and their reconstruction examples on samples from WebVid and the UCF101 dataset. We also qualitatively present the effect of using VideoMAE [53] as part of the video discriminator. Besides, for video generation, we present generated samples on the UCF-101 class-to-video and K600 frame prediction task.

###### I.1. Adaptive Length Reconstruction Examples

We present the reconstruction results of our final video adaptive tokenizer, along with their token assignments decided by the router. In Fig. 6, we present the reconstruction results of the final adaptive tokenizer trained on WebVid10M [5]. And in Fig. 7 are the reconstruction results on UCF-101 dataset, using another final video adaptive tokenizer trained on UCF-101 and K600 datasets. The patterns of video and assignment pairs given by the router correspond to intuitions. The router typically assigns more tokens to the initial temporal block, which helps the reconstruction of subsequent frames to also benefit from more precise information encoded for the initial block. Content with simple layouts or largely repeats previous frames receives fewer tokens. In contrast, videos that vary intensely are assigned more.

[Figure 97]

GTRec.GTRec.GTRec.GTRec.

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

- Figure 6. Adaptive reconstruction results on WebVid. We downsample 16 frames into 8 frames for visualization, and each two frames represent a 4-frame temporal block. The router typically assigns more tokens to the initial temporal block, allowing the reconstruction of subsequent frames to also benefit from more precise information encoded for the initial block. Content with simple layouts receives fewer tokens (first example vs. other examples). If later frames largely repeat previous ones, they are assigned the minimum number of tokens. Video clips that vary constantly and intensely are allocated more tokens.

[Figure 105]

- GTRec.GTRec.GTRec.GTRec.
- Figure 7. Adaptive reconstruction results on UCF-101. We downsample 16 frames into 8 frames for visualization, and each two frames represent a 4-frame temporal block.

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

GT

[Figure 113]

w/ VideoMAE semantic discriminator: PSNR=23.06, LPIPS=0.1881

[Figure 114]

w/o VideoMAE semantic discriminator: PSNR=23.60, LPIPS=0.1791

[Figure 115]

GT

[Figure 116]

w/ VideoMAE semantic discriminator: PSNR=20.82, LPIPS=0.2480

[Figure 117]

w/o VideoMAE semantic discriminator: PSNR=21.72, LPIPS=0.2398

[Figure 118]

- Figure 8. Qualitative comparison for using VideoMAE discriminator or not. Using VideoMAE discriminator can degrade the PSNR/LPIPS, but in actual perceptual checking, we find that this degradation is traded for alleviated blurriness and artifact patterns. Zoom in to check the visual details.

- 5

[Figure 119]

w/ VideoMAE semantic discriminator: PSNR=20.91, LPIPS=0.1703

GT

w/o VideoMAE semantic discriminator: PSNR=21.77, LPIPS=0.1649

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

- 6

- Figure 9. Adaptive generation results on UCF-101. We use the 633M GPT model trained on EVATok. We use a constant 3.0 CFG for sampling.

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

##### condition prediction frames

- Figure 10. Adaptive generation results on K600 frame prediction. We use the 633M GPT model trained on EVATok. We don’t use CFG for sampling, following typical approaches. The 1, 3, 5 frames from the 5 condition frames are plotted as the condition part, and the rest 11 frames are downsampled into 6 frames for visualization.

###### I.2. VideoMAE Discriminator for Visual Quality

the pretrained VideoMAE discriminator, while another is trained with the PatchGAN discriminator. As shown in Fig. 8, although the reconstructed videos of the tokenizer trained with VideoMAE discriminator show worse PSNR and LPIPS, they are actually more perceptually preferable as they largely alleviate the blurriness or artifact patterns, especially for highly dynamic and challenging examples. Therefore, we conclude that despite the degradation in PNSR/LPIPS, VideoMAE discriminator still largely en-

In our ablation study in the main paper, the application of the VideoMAE [53, 63] discriminator significantly improves the rFVD and downstream gFVD but leads to degradation in PSNR and LPIPS. In this part, we aim to qualitatively examine the perceptual effect for improved rFVD and degraded PSNR/LPIPS. We compare two video adaptive length tokenizers on WebVid-10M, one is trained with

hances the reconstruction quality perceptually.

###### I.3. Adaptive Length Video Generation Examples

We present UCF-101 class-to-video generation results in Fig. 9 and K600 frame prediction results in Fig. 10. As in Fig. 9, the AR generation model learns an intuitive way for adaptive length generation. First, the model tends to pay more efforts for the generation of the first temporal block, which lays the foundation for the later generation. For later blocks, content with more variation tends to take more tokens to generate, while small-motion content takes fewer tokens.

#### J. Computational Overhead Analysis

We present the computation cost for the model training of our four-stage framework and the downstream AR models, as shown in Tab. 7. Compared to the previous fixed-length methods, the extra training cost of our four-stage framework comes from the first three stages. However, the first three stages only take around 27.8% of the total four-stage training cost. This percentage can be further reduced in realworld applications. The size of the proxy tokenizer and its training duration can be decreased for faster training, as we only need the proxy tokenizer to compare assignments, instead of performing well. The data curation can be processed by parallel and independent processes, without any GPU communication bottlenecks. And ultimately, the extra cost in adaptive tokenizer training is a one-time investment, but the savings for downstream deployment will consistently take effect. Therefore, the extra cost of the fourstage training is controllable and worthwhile.

#### K. Attention Mask for EVATok

In this section, we illustrate the specific details of the attention mask mechanism in our Q-Former style tokenizer, which ensures the temporal causal structure of our 1D token sequences. Each Q-Former layer consists of one selfattention module and one cross-attention module. The queries are first passed through the self-attention module, and then in the cross-attention module, the queries will attend to the reference embeddings. Next, we use an example to show what attention masks look like in the Q-Former encoder and Q-Former decoder. Assume a video clip is patchified into a 4 × 4 × 4 shape tensor, where the first 4 corresponds to the number of temporal blocks. And let the assignment of tokens across the 4 blocks for this video be (16,8,2,2). Then, the attention masks for the Q-Former encoder and Q-Former decoder will be the ones shown in Fig. 11. The query embeddings of each temporal block can only attend to query embeddings or reference embeddings that are no later than this temporal block.

#### L. Accuracy vs. Proxy Reward for Routers

In this part, we examine the accuracy of the router on the validation sets. We find that although the accuracy is relatively slow, the assignments given by the router still obtain decent proxy reward. We use preference weights wq = 1.2,wl = 0.8 for proxy reward calculation, which are the same as the weights used for the evaluated router training data curation. To evaluate relatively how good the predicted assignments are among all candidate assignments, we use a new metric, proxy reward percentile, defined as:

Ex(Rproxy(aeval,x)) − Ex(Rproxy(aworst,x)) Ex(Rproxy(abest,x)) − Ex(Rproxy(aworst,x)) × 100%

P =

(8) where aeval is the assignment to be evaluated for video x, abest is the searched max-proxy-reward assignment for x, and aworst is the min-proxy-reward assignment. In practice, aeval can be given by the router according to x or by some other manually designed strategy. Ex(Rproxy(aeval,x)) is the expectation for the proxy reward based on aeval and x. The range of P is [0,1] and the larger P, the better the assignment strategy is. We design a best-uniform baseline, which chooses the max-proxy-reward uniform assignment for x, to compare with the router assignment. As shown in Tab. 8, on WebVid validation set, the top1 accuracy of the router is relatively low, but the proxy reward percentile of the router is high. Moreover, when tested on the unseen UCF-101 dataset, although the top1 accuracy of the router significantly drops, its proxy reward percentile is largely maintained. This phenomenon indicates that the router does not need to be very precise to achieve good performance, implying that the optimal assignment prediction task is not demanding, and some deviation from the best choice won’t result in a large performance drop.

#### M. Image Adaptive Tokenization

Different from videos, images don’t have a temporal dimension, so intuitively images are much less redundant than videos. Our experiments on ImageNet [45] 256×256 show that in this setting, the improvement in overall reconstruction quality that can be brought by assigning different token lengths to different images could be limited. However, for downstream generation, adaptive image tokenization can still help produce better generation FID [22] with fewer tokens generated. This result highlights that training generative models on adaptive length sequences is not only efficient but also beneficial to their generation capability.

###### M.1. Implementation Details

We train image tokenizers on 256 × 256 ImageNet [45] dataset using a similar CNN + Q-Former hybrid architecture of GigaTok-S-B [68]. The basic training recipe is

###### Stage / Task Model Size Dataset Bsz Iters / Epochs GPUs Time

- Stage 1: Proxy tokenizer training 145M WebVid (or UCF& K600) 128 400k iters 64×V100 116 h
- Stage 2: Data curation – WebVid 100k-Subset – – 4×64×V100 12.5 h
- Stage 3: Router training 20M WebVid 100k-Subset 128 50k iters 32×V100 5 h
- Stage 4: Final adaptive tokenizer training 145M WebVid (or UCF& K600) 128 1000k iters 64×V100 347 h

AR training: UCF-101 class-to-video 633M UCF-101 128 3000 epochs 64×V100 88 h AR training: K600 frame prediction 633M Kinetics-600 128 75 epochs 64×V100 140 h

Table 7. Summary of compute and time for the four-stage tokenizer pipeline and subsequent AR model training.

Temporal Distribution (16, 16, 16, 16) (16, 8, 2, 2) Query Self-Attention Mask

Temporal Distribution (16, 8, 2, 2) (16, 16, 16, 16)

1D Query 3D Reference Embeddings

3D Query 1D Reference Embeddings

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

Query Self-Attention Mask

Q-FormerDecoder

Q-FormerEncoder

Query-Reference Cross-Attention Mask

Query-Reference Cross-Attention Mask

###### Figure 11. Example for the attention masks in our Q-Former style adaptive tokenizer.

Dataset Method Val top1/top5 acc. Proxy Reward Percentile WebVid

Best-Uniform - 84.88% Router 11.72% / 35.03% 96.96%

Best-Uniform - 88.46% Router 5.77% / 23.68% 96.19%

UCf-101

- Table 8. Accuracy vs. proxy reward percentile for the router assignment. In terms of accuracy. the assignments predicted by the router do not usually hit the top1 or top5 highest proxy reward assignments. However, in terms of proxy reward percentile, the router assignments achieve good results, and generalize to unseen dataset (UCF-101) well.

Fixed Uniform Assign. Max-Proxy-Reward Assign.

LPIPS

FID

PSNR

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

1.8

0.250

- 19
- 20
- 21

1.6

0.225

LPIPS

PSNR

1.4

FID

0.200

1.2

0.175

1.0

200 400 600

200 400 600

200 400 600

Token Num.

Token Num.

Token Num.

Figure 12. Image tokenization quality-cost trade-off curve. On ImageNet 256 × 256 reconstruction, the improvements of maxproxy-reward assignment can be marginal compared to uniform assignment.

also largely aligned to GigaTok except that we utilize DINOv3 [47] to provide semantic alignment. We train all image tokenizers with 256 batch size with only 400k itera-

tions, as we only target to validate the gain of adaptive tokenization on images compared to the fixed-length baseline.

###### LPIPS↓ rFID↓ #rTokens↓ gFID↓ #gTokens↓

Uniform (Final) 0.2205 1.22 256 4.72 256 Router (Final) 0.2455 1.46 205 (-19.9%) 4.51 197 (-23.0%)

- Table 9. Image final tokenizer validation. For ImageNet 256 × 256, saving 19.9% tokens by adaptive tokenization inevitably leads to worse rFID, but the performance and efficiency of downstream AR generation models can still benefit from our router. The AR generation models use a constant 1.5 CFG during inference.

Our four-stage framework smoothly translates from videos to image adaptive tokenization, because an image can be equivalent to a one-block video in the tokenization process.

For the proxy tokenizer, we predefine 8 candidate levels of token numbers, {512,384,256,192,128,96,64,32}, to be assigned to each image for variable length tokenizer training. For image router training, we train ViT-S [13] size routers on a subset of ImageNet training split of 100k images. They are trained for 25k iterations with a batch size of 256. We use normalized LPIPS as the quality metric in the proxy reward calculation. The reconstruction quality is evaluated on the 50k ImageNet validation set. For downstream AR generation validation, we train Llama-like [50] GPT-B models on each tokenizer for 300 epochs on ImageNet, and evaluate them with generation FID [22] with a constant 1.5 CFG, following the typical approaches [12, 50].

###### M.2. Results

Quality-cost trade-off curve. We use a similar way as for video proxy tokenizers, to plot the quality-cost tradeoff curve under different overall token budgets on an image proxy tokenizer. As shown in Fig. 12, the quality-cost trade-off curve evaluated on image proxy-tokenizers shows that the improvements brought by max-proxy-reward assignment are limited, which is different from the results on videos. This phenomenon corresponds to observations in previous adaptive image tokenization trials [39, 46] on ImageNet, where their adaptive length image tokenizers cannot outperform their fixed-length baselines even with the same overall token budgets.

Final image tokenizer validation. In the final image adaptive tokenizer training, as shown in Tab. 9, we utilize an image router trained with wq = 1.3,wl = 0.7 to save 19.8% tokens in reconstruction, but it inevitably leads to worse rFID. However, the AR model trained on our adaptive image tokenizer achieves better gFID with 23.0% fewer tokens generated, compared to the fixed-uniform baselines, which assign 256 tokens to all 256 × 256 images. This indicates that the performance and efficiency of downstream AR image generation can still benefit from image adaptive tokenization using our method.

