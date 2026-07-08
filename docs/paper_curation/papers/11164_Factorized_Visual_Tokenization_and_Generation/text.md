## Factorized Visual Tokenization and Generation

Zechen Bai1, Jianxiong Gao2, Ziteng Gao1, Pichao Wang3, Zheng Zhang3, Tong He3, Mike Zheng Shou1* 1Show Lab, National University of Singapore 2Fudan University 3Amazon

# arXiv:2411.16681v2[cs.CV]27Nov2024

#### Abstract

Visual tokenizers are fundamental to image generation. They convert visual data into discrete tokens, enabling transformer-based models to excel at image generation. Despite their success, VQ-based tokenizers like VQGAN face significant limitations due to constrained vocabulary sizes. Simply expanding the codebook often leads to training instability and diminishing performance gains, making scalability a critical challenge. In this work, we introduce Factorized Quantization (FQ), a novel approach that revitalizes VQ-based tokenizers by decomposing a large codebook into multiple independent sub-codebooks. This factorization reduces the lookup complexity of large codebooks, enabling more efficient and scalable visual tokenization. To ensure each sub-codebook captures distinct and complementary information, we propose a disentanglement regularization that explicitly reduces redundancy, promoting diversity across the sub-codebooks. Furthermore, we integrate representation learning into the training process, leveraging pretrained vision models like CLIP and DINO to infuse semantic richness into the learned representations. This design ensures our tokenizer captures diverse semantic levels, leading to more expressive and disentangled representations. Experiments show that the proposed FQGAN model substantially improves the reconstruction quality of visual tokenizers, achieving state-of-the-art performance. We further demonstrate that this tokenizer can be effectively adapted into auto-regressive image generation. https://showlab.github.io/FQGAN

#### 1. Introduction

In recent years, the success of discrete token-based approaches in natural language processing [3, 25] has sparked growing interest in discrete image tokenization and generation [7, 13, 38]. Visual tokenizers play a crucial role by converting image data into discrete tokens, thereby enabling the application of powerful transformer-based generative models. The quality of visual tokenization significantly impacts high-fidelity image reconstruction and generation.

*Corresponding author

VQ (Taming)

- 0

- 1

- 2

- 3

- 4

- 5

ReconstructionFID

VQ-LC

VQ (LlamaGen)

LFQ (Open-MAGVIT2)

###### FQGAN (Ours)

2048 30724096 8192 16384 3276849152 100000 262144

Codebook Size

Figure 1. Performance comparison of popular tokenizers at various codebook sizes, including VQ (Taming) [7], VQ (LlamaGen) [29], VQ-LC [46], LFQ (OpenMAGVIT2) [18], and FQGAN. Lower rFID values indicate better performance.

Popular visual tokenizers, such as VQGAN [7], adopt an encoder-quantizer-decoder structure, where the quantizer converts the latent feature into discrete tokens via vector quantization (VQ). These approaches have shown remarkable performance on image reconstruction and generation [4, 13, 29]. Despite these successes, visual tokenization involves inherently lossy compression, especially compared to continuous encoding, since visual data is naturally continuous and complex. A common strategy to address this is to enlarge the codebook, enhancing its capacity to approximate continuous representations. However, traditional VQ-based models are constrained by codebook size. Existing research [29, 46] indicates that increasing codebook sizes beyond 16,384 can lead to training instability, such as low codebook utilization and performance saturation.

Recent works have proposed innovative strategies to address these limitations. For example, FSQ [19] and LFQ [40] are introduced to eliminate the need for an explicit codebook, achieving state-of-the-art reconstruction quality using a massive codebook size. Among VQ tokenizers, VQGAN-LC [46] employs pre-trained feature clusters to help stabilize training with larger codebooks. Nevertheless, VQ tokenizers still exhibit inferior performance to LFQ ones and, more importantly, the inherent challenges of VQ

remain unresolved. Large codebooks complicate quantization by necessitating the calculation of pairwise distances between encoder outputs and all codebook entries, followed by an argmin() operation to select the nearest code. As the codebook size increases, the lookup process becomes more computationally expensive and unstable, leading to inconsistent results.

To tackle these challenges, we draw inspiration from the divide-and-conquer principle, breaking down a complex problem into smaller, more manageable components to enhance both stability and performance. We propose a novel factorized codebook design, wherein a large codebook is split into several smaller, independent sub-codebooks. This factorization simplifies the tokenization process, improving the stability of the quantization. By combining entries from multiple sub-codebooks, we construct a more expressive and scalable representation space. It provides greater flexibility for capturing image features at varying levels of granularity, improving overall tokenization quality.

However, to fully harness this expressiveness and ensure that each sub-codebook contributes uniquely to the representation, it is essential to disentangle the learned features. Factorizing the codebook alone is insufficient unless each sub-codebook learns to capture unique and complementary information. To address this, we introduce a disentanglement regularization mechanism that enforces orthogonality between sub-codebooks, encouraging each sub-codebook to focus on distinct aspects of the visual data, such as spatial structures, textures, colors, etc. This is akin to having different specialists analyzing various aspects of an image, ultimately resulting in a richer and more comprehensive representation.

To further enhance the specialization of the subcodebooks, we integrate representation learning as an essential part of the training framework. By seamlessly weaving into the training objective, the sub-codebook is guided to capture semantically meaningful features that contribute to the overall representation. Traditional reconstruction objectives often lead to overfitting on high-variance visual details, which results in features that lack semantic meaning for perception tasks [2]. Our representation learning objective addresses this issue by guiding the factorized codebooks to learn robust, semantically rich features capable of generalizing beyond simple reconstruction. Specifically, by leveraging different vision backbones, e.g., CLIP [25] and DINOv2 [20], the sub-codebooks essentially learn to establishes a complementary hierarchy of semantics: low-level structures (e.g., edges), mid-level details (e.g., textures), and high-level concepts (e.g., abstract appearance).

By seamlessly integrating factorized codebook design, disentanglement regularization, and representation learning objectives, our visual tokenizer captures a diverse and rich set of features. This holistic approach greatly enhances re-

construction quality, as each sub-codebook learns to represent different aspects of the image in a balanced and semantically meaningful way. Leveraging these three core innovations, our visual tokenizer achieves state-of-the-art performance in discrete image reconstruction, as illustrated in Fig. 1. Additionally, we extend our analysis to autoregressive (AR) generation tasks. Unlike conventional tokenizers that produce a single token per image patch, our tokenizer encodes each patch into multiple tokens, resulting in a richer and more expressive representation. Drawing inspiration from related works on handling extremely large codebooks [18] and multi-code [13], we design a factorized AR head that predicts sub-tokens for each patch, adapting our tokenizer effectively for downstream image generation.

In summary, our contributions include:

- • A novel factorized quantization design that revitalizes VQ-based tokenizers, achieving state-of-the-art performance on discrete image reconstruction.
- • Introduction of disentanglement and representation learning mechanisms that enable diverse and semantically meaningful codebook learning.
- • Demonstration that our tokenizer enhances downstream AR models, improving image generation quality on the ImageNet benchmark.

#### 2. Related Work

##### 2.1. Visual Tokenizers

Visual tokenizers map images into a latent space for downstream tasks, such as visual understanding [16] and visual generation [7]. Visual tokenizers generally fall into two categories: continuous feature-based models and discrete token-based models. We mainly discuss the discrete ones as they are closely related to our work. Popular visual tokenizers, exemplified by VQGAN [7], use an encoder-quantizerdecoder structure: the encoder maps image data to a latent space, the quantizer transforms this representation into discrete tokens using vector quantization, and the decoder reconstructs the image from these tokens. Building on VQGAN framework, numerous works [13, 19, 37, 38, 44] have been developed to improve performance from various perspectives. ViT-VQGAN [38] upgrades the encoder-decoder architecture from a CNN-based network to a transformerbased one. RQ-VAE [13] proposes modeling residual information using multiple codes to capture finer details.

Despite advancements, VQ tokenizers still struggle with the critical challenge of limited codebook size. Research [29, 46] indicates that expanding the codebook size beyond 16,384 can lead to performance saturation or even degradation. This issue is often accompanied by a low usage rate of the large codebook. To address this, FSQ (Finite Scalar Quantization) [19] and LFQ (Lookup Free Quantization) [40] are proposed to eliminate the need for an explicit

codebook and significantly alleviates this issue. Within the VQ family, VQGAN-LC [46] uses pre-trained feature clusters to implicitly regularize the large codebook, helping to maintain a higher usage rate. This work suggests that semantic information can benefit visual tokenizers, a concept further explored in recent studies [8, 15, 34, 35, 39, 45]. For instance, VILA-U [35] demonstrates that a pre-trained vision model can be fine-tuned into a visual tokenizer while preserving its semantic capabilities. LG-VQ [15] and VQKD [34] show that incorporating language supervision or image understanding models can improve visual tokenizers. A concurrent work, ImageFolder [14], proposes a folded quantization approach, improving the image reconstruction performance by a large margin. Our work, as part of the VQ family, aims to revitalize VQ tokenizers by addressing the large codebook problem through factorized quantization and leveraging semantic supervision. A more detailed discussion with related works is provided in the Appendix.

- 2.2. Auto-regressive Visual Generation

Auto-regressive visual generation uses a next-token prediction approach to sequentially generate images or videos. VQGAN [7], a pioneering model, utilizes a transformer to predict tokens sequentially. RQ-VAE [13] extends VQGAN by incorporating a residual tokenization mechanism and adding an AR transformer head to predict residual tokens at a finer depth. LlamaGen [29] extends the VQGAN transformer architecture to the Llama [32] framework, demonstrating promising scaling behaviors. VAR [30] extends next-token prediction to next-scale prediction, reducing auto-regressive steps and enhancing performance. Open-MAGVIT2 [18], similar to LlamaGen [29], adopts a Llama-style auto-regressive transformer as its backbone. To manage an extremely large codebook, it predicts two subtokens during the AR generation phase and composes them to obtain the original code. It also employs an RQ-like architecture, termed intra-block, to predict sub-tokens. In this work, our factorized codes share similarities with RQVAE [13] and Open-MAGVIT2 [18], specifically in predicting multiple tokens at each AR step. Consequently, we use a factorized AR head atop the AR backbone to predict subtokens for each patch.

- 3. Method

- 3.1. Preliminary

VQGAN [7] employs a learnable discrete codebook C ∈ RK×D to represent images, where K is the codebook size while D is the dimensionality of the codes. Given an input image x, the encoder transforms it into a latent feature h = Enc(x). Then, the closest codebook entry for each patch is retrieved from the codebook to serve as the quan-

tized representation:

hi − cj , (1)

qi = Quant(hi,C) := arg min

cj∈C

where hi ∈ RD, cj ∈ RD, qi ∈ RD denotes the encoded latent feature at patch i, codebook entry, and quantized feature at patch i, respectively. After that, VQGAN uses a decoder to reconstruct the image xˆ = Dec(q). The training objective of VQGAN is to identify the optimal compression model of {Enc,Dec,C}, involving the following loss:

LVQGAN = Lrec + LVQ + Lperceptual + LGAN, (2)

where Lrec denotes the pixel reconstruction loss between x and xˆ. LVQ denotes the codebook loss that pulls the latent features h and their closest codebook entries q closer. Lperceptual denotes the perceptual loss between x and xˆ by leveraging a pre-trained vision model [43]. LGAN introduces an adversarial training procedure with a patchbased discriminator [12] to calculate the GAN loss. As these losses are widely adapted in most VQ tokenizer designs [7, 13, 29, 46], we omit the detailed definitions for simplicity.

##### 3.2. Factorized Quantization

Despite the remarkable performance achieved by the classical VQGAN model, it is known to suffer from unstable training and low codebook usage rate when increasing the codebook size. One prominent issue is the unstable lookup process among a large set of embeddings. To alleviate this, we propose a factorized quantization approach that decomposes a singe large codebook C into k small sub-codebooks. The main framework is illustrated in Fig. 2.

Encoder. We regard the original VQGAN encoder as a base feature extractor. On top of that, k feature adapters are introduced to transform the base image features into their respective feature space. Formally,

hbase = Enc(x), (3) h1,h2,...,hk = F1(hbase),F2(hbase),...,Fk(hbase), (4)

where F1,...,Fk are the adapters for each factorized branch. Quantizer. Our method maintain a unique codebook for each factorized branch. After extracting the branch-specific features, the quantization process is conducted at each codebook independently. Formally,

q1,...,qk = Quant(h1,C1),...,Quant(hk,Ck), (5) where C1,...,Ck are the factorized sub-codebooks.

Decoder. Given the quantized feature from each subcodebook, we employ a simple yet effective aggregation approach that concatenates them along the latent (channel) dimension. After that, the aggregated features are fed into the

Disentangle. Loss

[Figure 1]

| |
|---|
| |
| |
| |

Quantization on

AR Head

AR Head

AR Head

AR Head

Share Weight

- Sub-Codebook 1

[Figure 2]

Enc

| | | |
|---|---|---|

| | |
|---|---|

…

[Figure 3]

Quantization on

- Sub-Codebook 2 …

- Adapter 1

- Adapter 2

[Figure 4]

### ⊕

Dec

Concat

| | |
|---|---|
| | |
| | |
| | |
| | |

Auto-Regressive Transformer

| | | |
|---|---|---|

| | |
|---|---|

Pixel Recon. Loss

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

…

❄

Vision Models

Feature Pred.

Rep. Learning Loss

Condition

CLIP, DINO, …

Figure 2. Illustration of the our method. The left part shows FQGAN-Dual, the factorized tokenizer design in an example scenario when k = 2. This framework is extendable to factorization of more codebooks. The right part demonstrate how we leverage an additional AR head to accommodate the factorized sub-codes based on standard AR generative transformer.

pixel decoder, which is inherited from the VQGAN model. Formally,

glement regularization mechanism as follows:

n

1 n

xˆ = Dec([q1;q2;...;qk]), (6) where “;” denotes the concatenation operation.

(q1⊤q2)2, (7)

Ldisentangle =

i=1

where n is the number of samples in a batch.

The factorized quantization design presents several appealing properties. First, the factorized and parallelized lookup process greatly alleviates the lookup instability in a single large codebook. Second, maintaining factorized sub-codebooks and independent feature adapters allow the model to learn more diverse features. Lastly, the code aggregation before decoding essentially builds a super large conceptual codebook with a size of |Ci|k. E.g., suppose k = 2, |C1| = |C2| = 1024, there are 1,0242 = 1,048,576 unique combinations of the sub-codes. Although the actual freedom of this conceptual codebook is smaller than a real codebook with the same size, it already provides much larger capacity, given that we only maintain |Ci| × k codes. Prior arts reaches reconstruction saturation with codebook size 16,384. In Tab. 3 of experiment, it is shown that factorizing the 32,768 codebook into two 16,384 sub-codebooks can further significantly improve the reconstruction performance.

This regularization mechanism minimizes the squared dot product between the two involved codes. The dot product directly measures the affinity between the two codes after L2 normalization, ranging from [−1,1], where -1/1 indicates negative/positive correlation and 0 denotes orthogonality. Minimizing the squaring function encourages the dot product value to approach 0. It also provides a smooth gradient for optimization. Note that this regularization does not directly apply to the entire codebook. Instead, it operates on patches of each image instance. In other words, for each patch, it encourages the involved sub-codes to capture different aspects.

###### 3.2.2. Representation Learning

Typically, the main training objective of visual tokenizers is pixel reconstruction. Research [2] suggests that the reconstruction objective can hardly learn meaningful semantic features for perception, as the features mainly capture highvariance details. However, recent work [42] finds that learning semantic features can benefit visual generation model training. In this work, we show that representation learning plays a crucial role in tokenizer training, especially in the context of factorized quantization.

###### 3.2.1. Disentanglement

The factorized quantization design allows diverse feature learning, given the sufficient capacity in the feature adapters and sub-codebooks. However, without explicit constraints, the sub-codebooks risk learning redundant and overlapping codes, particularly as the codebook size increases. To address this issue, we propose a disentanglement regularization mechanism for the factorized sub-codebooks.

Consider the example of an image patch depicting an ear. A traditional VQ code may capture its appearance, such as color, texture, etc. However, it is unaware of the species, e.g., cat or dog. While such a code may effectively reconstruct the patch, introducing semantic information is expected to be beneficial. When informed with semantics, the decoder (and generation model) can better handle the

For simplicity, we take k = 2 as an example scenario. Through Eq. 5, we obtain q1 ∈ RL×D and q2 ∈ RL×D, where L is the number of patches. We design the disentan-

corresponding visual reconstruction and generation tasks. Moreover, compared to high-variance signals, semantic information tends to generalize better.

Building on this intuition, we introduce representation learning as a training objective to encourage the model to learn meaningful semantic features. We continue to use k = 2 as an example scenario. Specifically, one subcodebook, say C2, is tasked with predicting the features of a pre-trained vision model using a lightweight feature prediction model. C2 essentially serves as the semantic codebook that embeds the semantic information. The other codebook C1 functions as the visual codebook that captures the visual details, complementing C2.

We note that semantic is still not a well-defined concept in the community. As studied in the multimodal domain [31], pre-trained vision models place varying emphasis on the semantic property. For instance, CLIP [25], which is pre-trained for cross-modal alignment, encodes high-level semantic features, while DINOv2 [20], a selfsupervised vision model, captures mid-level visual features. Incorporating diverse vision models into the factorized subcodebooks establishes a hierarchy of semantics: low-level structures (e.g., edges), mid-level details (e.g., textures), and high-level concepts (e.g., abstract appearance).

The total loss is a weighted sum of all the losses:

Ltotal = LVQGAN + λ1Ldisentangle + λ2Lrep, (8)

where λ1 and λ2 are weights. In this paper, we present two variants of the implementation of FQGAN, including k = 2 (FQGAN-Dual) and k = 3 (FQGAN-Triple). FQGANDual employs CLIP [24] as the pre-trained vision model to provide semantic features for the representation learning objective. For FQGAN-Triple, CLIP [24] and DINOv2 [20] are jointly adopted to form a semantic hierarchy.

##### 3.3. Auto-Regressive Model

The factorized quantization design produces multiple subtokens for each spatial position, represented as Zt = (zt1,zt2,...,ztk), where t denotes the time step. Standard AR transformers, such as those in VQGAN [7] and LlamaGen [29], predict only the index of the next token based on the hidden feature gt, which makes them inherently unsuitable for handling factorized sub-tokens. One simple solution is to apply k classifiers to the hidden feature gt, yielding the indices for the sub-tokens as zti = clsi(gt),i ∈ {1,...,k}. However, this method is shown to be suboptimal (see Tab. 4). To address this, we introduce a factorized AR head that sequentially predicts the distributions of these factorized sub-tokens, allowing for better modeling of their dependencies. Fig. 2 illustrates the full Factorized Auto-Regressive model (FAR). For each patch, the hidden feature gt serves as a prefix condition, which is processed by an additional AR head to au-

toregressively predict the list of sub-tokens, formulated as zti = headAR(gt;zt1,zt2,...,zti−1). Following a scaling pattern similar to previous works [18, 29], FAR has Base and Larger versions, differentiated by their parameter sizes. The detailed configurations are provided in the Appendix.

#### 4. Experiment

##### 4.1. Setup

In experiments, we follow previous works to use the standard benchmark, ImageNet [5], to train and evalaute the tokenizers and AR generation models. For the factorization configuration, we experiment with k = 2 and k = 3. λ1 and λ2 are empirically set to 0.1 and 0.5 respectively. The training schedule of the visual tokenizer is adapted from LlamaGen [29]. Specifically, the tokenizer is trained with a global batch size of 256 and a constant learning rate of 2e-4 across 8 A100 GPUs. For the AR model, we adopt a Llama-style [29, 32] transformer architecture as the backbone. To accommodate the factorized codes, the model employs k embedding layers on the input side, each embeds a separate sub-code, followed by a linear layer that aggregates these embeddings into a single representation. On the output side, we adapt a factorized AR head that predicts the factorized codes for each patch. The AR models are trained for 300 epochs with a constant learning rate of 2e-4 and a global batch size of 256 across 8 A100 GPUs.

Metric. We adopt Fr´echet inception distance (FID) [9] as the main metric to evaluate visual tokenizers and generation models. For tokenizers, we use the ImageNet validation set, consisting of 50k samples, to compute the reconstruction FID (rFID). Additionally, we use PSNR and Inception Score [28] as auxiliary metrics for comparison. For generation models, we follow the widely adapted ADM [6] evaluation protocol to compute the generation FID (gFID). Besides, Inception Score, Precision, and Recall are also used for comparison, following prior works. In both quantitative and qualitative evaluations, we use classifier-free guidance [10] (CFG), with the weight set to 2.0. We do not use any top-k or top-p sampling strategy unless specified.

##### 4.2. Comparison on Tokenizers

We first compare our method with popular visual tokenizers listed in Tab. 1. Our FQGAN model sets a new stateof-the-art performance in discretized image reconstruction across various settings, including different codebook sizes and downsample ratios. Compared to VQGAN and its advanced variants, our method outperforms them by a large margin. Note our method is also built based on the vectorquantization mechanism. This comparison effectively validates the advantage of our factorized quantization design.

Interestingly, compared to the state-of-the-art tokenizer Open-MAGVITv2, which employs an advanced lookup-

Table 1. Comparisons with other image tokenziers. Reconstruction performance of different tokenizers on 256 × 256 ImageNet 50k validation set. All models are trained on ImageNet, except “∗” on OpenImages and “†” on unknown training data. Bold denotes the best scores; underline denotes the second place.

Downsample Codebook Code Ratio Size Dim rFID↓ PSNR↑

Method

VQGAN [7] 16 16384 256 4.98 − SD-VQGAN [27] 16 16384 4 5.15 − RQ-VAE [13] 16 16384 256 3.20 − LlamaGen [29] 16 16384 8 2.19 20.79 Titok-B [41] − 4096 12 1.70 − VQGAN-LC [46] 16 100000 8 2.62 23.80 VQ-KD [34] 16 8192 32 3.41 VILA-U [35] 16 16384 256 1.80 Open-MAGVIT2 [18] 16 262144 1 1.17 21.90 FQGAN-Dual 16 16384 × 2 8 0.94 22.02 FQGAN-Triple 16 16384 × 3 8 0.76 22.73

SD-VAE† [27] 8 4 0.74 25.68 SDXL-VAE† [23] 8 − 4 0.68 26.04

ViT-VQGAN [38] 8 8192 32 1.28 − VQGAN∗ [7] 8 16384 4 1.19 23.38 SD-VQGAN∗ [27] 8 16384 4 1.14 − OmniTokenizer [33] 8 8192 8 1.11 − LlamaGen [29] 8 16384 8 0.59 25.45 Open-MAGVIT2 [18] 8 262144 1 0.34 26.19 FQGAN-Dual 8 16384 × 2 8 0.32 26.27 FQGAN-Triple 8 16384 × 3 8 0.24 27.58

free quantization mechanism, our method still exhibits superior image reconstruction performance, with a 0.41 rFID gap. This result suggests that VQ-based methods still hold great potential for visual tokenization, which may have been overlooked previously. Existing work often regards the codebook as a bottleneck, while our approach provides a novel perspective. An explicit codebook offers the opportunity for more sophisticated designs on code embeddings, such as disentanglement and representation learning.

Another key finding is the comparison between SDVAE [27], SDXL-VAE [23], and our FQGAN. SD-VAE and SDXL-VAE are advanced continuous visual tokenizers widely used in Stable Diffusion models [21, 23, 26]. We observe that our FQGAN, with a 16× downsample ratio, achieves performance comparable to these continuous models, which use an 8× downsample ratio. In a fairer comparison, with both methods using an 8× downsample ratio, our method achieves a significantly lower reconstruction FID, suggesting that discrete representation in image tokenization is no longer a bottleneck for image reconstruction.

##### 4.3. Comparison on Generation Models

We compare our FAR model with mainstream image generation models, including diffusion models, LFQ-based AR models, and VQ-based AR models, as shown in Tab. 2. Among VQ-based AR models, we observe that FAR achieves competitive image generation performance. When comparing models with similar parameter sizes, specifically

Table 2. Class-conditional generation on 256 × 256 ImageNet. Models with the suffix “-re” use rejection sampling. The evaluation protocol and implementation follow ADM [6]. Our model employs a cfg-scale of 2.0.

Type Model #Para. FID↓ IS↑ Precision↑ Recall↑

ADM [6] 554M 10.94 101.0 0.69 0.63

CDM [11] − 4.88 158.7 − − LDM-4 [27] 400M 3.60 247.7 − − DiT-XL/2 [21] 675M 2.27 278.2 0.83 0.57

Diffusion

Open-MAGVIT2-B [18] 343M 3.08 258.26 0.85 0.51 Open-MAGVIT2-L [18] 804M 2.51 271.70 0.84 0.54

LFQ AR

VQGAN [7] 227M 18.65 80.4 0.78 0.26

VQGAN [7] 1.4B 15.78 74.3 − − VQGAN-re [7] 1.4B 5.20 280.3 − − ViT-VQGAN [38] 1.7B 4.17 175.1 − − ViT-VQGAN-re [38] 1.7B 3.04 227.4 − − RQTran. [13] 3.8B 7.55 134.0 − − RQTran.-re [13] 3.8B 3.80 323.7 − −

VQ AR

LlamaGen-L [29] 343M 3.80 248.28 0.83 0.51 LlamaGen-XL [29] 775M 3.39 227.08 0.81 0.54 FAR-Base 415M 3.38 248.26 0.81 0.54 FAR-Large 898M 3.08 272.52 0.82 0.54

FAR-Base vs. LlamaGen-L and FAR-Large vs. LlamaGenXL, our FAR model consistently achieves superior performance in both FID and Inception Score. This validates the effectiveness of the proposed method. Among the other methods, RQ-Transformer [13] is similar to our method, as it also adopts an additional AR head to accommodate multiple sub-codes at each step. The performance gap between RQ-Transformer and FAR further validates the power of our FQGAN tokenizer and its transferability to the downstream generation model.

When comparing FAR with Open-MAGVIT2 [18], which shares a similar AR model design, our method exhibits a comparable or higher Inception Score, though with a slightly worse FID score. The Inception Score suggests that our FQGAN tokenizer has the potential to match LFQ performance, while the FID score gap still demonstrates the superiority of LFQ compared to VQ, as studied in MAGVITv2 [40]. Mitigating the generation performance gap between LFQ and VQ is a critical yet challenging problem, which is beyond the scope of this work. FQGAN is a crucial step toward this direction as it significantly improves image reconstruction performance, surpassing both VQ and LFQ tokenizers. Tab. 2 also suggests that the improvement on tokenization and reconstruction can be effectively transferred to AR generation. We hope the FQGAN tokenizer will inspire related further research.

Qualitative results of the FAR model is shown in Fig. 5. The FAR model in this section is trained with tokens from FQGAN-Dual tokenizer. More training details and settings of the FAR model are provided in the Appendix.

##### 4.4. Ablation Studies

Factorized Quantization. We investigate the design components of the FQGAN tokenizer, including the fac-

- Table 3. Ablation study on different components of the proposed factorized quantization, using the FQGAN-Dual variant.

Codebook Dis. Rep. Size Regular. Learn. rFID↓ IS↑ PSNR↑ Usage↑

Model

16384 − − 3.71 50.05 20.56 98% 32768 − − 3.60 50.60 20.56 84%

VQGAN

16384 × 2 ✗ ✗ 2.00 54.72 22.21 97% 16384 × 2 ✓ ✗ 1.84 55.04 22.04 98% 16384 × 2 ✗ ✓ 1.73 55.00 21.61 98% 16384 × 2 ✓ ✓ 1.66 55.21 21.62 98%

FQGAN

torized codebook, disentanglement regularization mechanism, and representation learning objective. In this study, we adopt FQGAN-Dual, i.e., k = 2. All experiments are conducted for 10 epochs on the ImageNet training set to ensure a fair comparison. As shown in Tab. 3, we start with a vanilla VQGAN tokenizer. Increasing the codebook size from 16,384 to 32,768 results in a drop in codebook usage, yielding only marginal performance gains even with double codebook size. Previous studies [29] have shown that with training schedule of more epochs, the 32,768 version ultimately performs worse than the 16,384 version. Next, we consider a vanilla factorized codebook design, which splits the single 32,768 codebook into two 16,384 subcodebooks. Such factorization brings a significant performance gain, as reflected in the rFID score change from 3.60 to 2.00. Compared to a single codebook with the same number of codes (i.e., capacity), the factorized design greatly reduces lookup complexity. It also yields more diverse code combinations, improving performance by a large margin.

Disentanglement and Representation Learning. Next, we gradually incorporate the proposed additional designs into the factorized codebooks. By making only one change in each experiment, we find that both the disentanglement regularization and the representation learning objective lead to better reconstruction results. When applied together, the two designs achieve even better performance. We attribute this performance gain to the fact that disentanglement regularization forces the factorized codes to learn more diverse aspects of the image, while the representation learning objective explicitly incorporates semantic information. It is worth mentioning that an rFID = 2.0 for the vanilla factorization version is already a very strong result, rarely achieved by previous VQ tokenizers. Pushing the performance further is particularly challenging, which effectively demonstrates the strength of the proposed designs.

What has each sub-codebook learned? To better understand the underlying behavior of the factorized subcodebooks, we provide a comprehensive visualization. Fig. 3 demonstrates the reconstruction results, including

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

Recon. with Sub-code 1

Recon. with Sub-code 2 (CLIP)

Recon. with Sub-code 1

Recon. with Sub-code 2 (DINO)

Recon. with Sub-code 3 (CLIP)

Input Reconstruction

FQGAN-Dual FQGAN-Triple

- Figure 3. Visualization of standard reconstruction by FQGANDual and reconstruction using only a single sub-codebook.

| |
|---|

t-SNE Visualization of Sub-Codebook 1

|hamster<br><br>nematode<br><br>paper_towel<br><br>cliff_dwelling|
|---|

t-SNE Visualization of Sub-Codebook 2CLIP

- Figure 4. T-SNE visualization of VQ codes from different subcodebooks in FQGAN-Dual.

standard reconstruction and reconstruction using only a single sub-codebook, achieved by setting the rest of the code embeddings to zero. In the two sub-codebooks of FQGANDual, we observe that sub-codebook 1 highlights low-level features, such as essential structures, shapes, and edges of the image. Sub-codebook 2, jointly supervised by CLIP features, presents a high-level abstract version of the original image, where colors are blurred together, and textures are preserved in a softened manner. When factorizing further into three sub-codebooks, i.e., FQGAN-Triple, we observe that sub-codebook 1 still emphasizes the low-level strong edges and overall shape. Sub-codebook 2, jointly supervised by DINO features, highlights textural elements, preserving surface patterns and fine details without clear structural outlines, representing mid-level features. Finally, subcodebook 3 concentrates on higher-level appearance and produces an abstract or blurry version of the original image. This visualization suggests that the factorized subcodebooks are indeed tasked with capturing different aspects of the image. With the supervision of representation learning, the sub-codebooks naturally form complementary hierarchical levels of visual features.

Furthermore, we illustrate the distribution of VQ codes from different sub-codebooks. Following previous practice [34], we randomly sample four classes from the ImageNet dataset, encode them with our tokenizer, and visualize the distribution using the t-SNE technique. The left part of Fig. 4 shows that VQ codes from sub-codebook 1, without additional regularization, are distributed in an unordered manner in the space. This is likely because this sub-codebook is solely trained for reconstruction, capturing high-variance detail while lacking awareness of semantic categories. In contrast, the right part suggests that the CLIP-

- Table 4. Ablation study on the generation model head design with the proposed FQGAN tokenizer. We use FAR-Large model with cfg-scale=1.75 in this study.

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

Generation Model Head Top-k Sampling gFID↓ k Linear Classifiers

4096 5.19 8192 6.90

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

4096 5.59 8192 8.88

k MLP Classifiers

4096 4.37 8192 3.74

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

Factorized AR Head

supervised sub-codebook 2 exhibits better semantic awareness, as its codes from the same category are distributed within a cluster. The two visualizations effectively demonstrate what each sub-codebook has learned qualitatively. We provide more visualizations in the Appendix.

Figure 5. Qualitative examples generated by our FAR model.

#### 5. Discussion and Future Work

Effect of AR Head. Adapting the FQGAN tokenizer to auto-regressive visual generation models presents the challenge of handling multiple sub-codes at each step. This is crucial, as predicting a wrong sub-code at a specific position can invalidate the entire patch. We present this investigation in Tab 4. We begin with a simple solution that employs k independent linear classifier heads to decode the hidden embedding of the AR backbone into their respective subcodebooks in parallel. This strategy yields decent results but lags behind auto-regressive models with the same parameter level. We hypothesize that this is due to the parallel decoding scheme placing too heavy a burden on the classifier. Therefore, we attempt to increase the capacity of the classifier by using multiple layers with a non-linear activation function in between. However, as shown in the table, the MLP version performs even worse, suggesting that simply increasing the capacity and computation is not the key to addressing this issue.

In factorized auto-regressive generation, the key issue is that the mismatch between sub-codes within a position (patch) can significantly affect the results. This suggests that an effective design is a module that not only decodes from the AR backbone but also models the dependency between sub-codes. To this end, we explore using an additional auto-regressive head to decode the factorized subcodes. The last row of Tab. 4 shows that this design can improve performance by a considerable margin. For example, when decoding code zt2, the vanilla classifier or MLP version only references the hidden embedding gt output by the AR backbone, whereas the AR module allows the decoding process to also attend to code zt1, strengthening the dependency among sub-codes of the current patch and improving overall generation quality.

In this work, we design a factorized quantization method and explore dual and triple sub-codebooks. Future research on factorizing more sub-codebooks could be a promising direction. Secondly, since the sub-codebook is jointly supervised by strong vision models, such as CLIP, it is interesting to probe its performance on multimodal understanding tasks. We provide a preliminary exploration in the Appendix. In the long term, building a truly unified tokenizer that excels at both generation and understanding tasks would be beneficial to the community. We believe the factorized design is a promising direction toward this ultimate goal, as it entails various levels of visual information. Regarding limitations, as discussed in Sec. 4.3, our method outperforms previous VQ-based methods in both reconstruction and generation. However, in downstream generation, our model still lags behind LFQ-based methods in generation FID metric. Our work, with a strong reconstruction performance, serves as an initial step toward bridging the gap between VQ and LFQ. We hope this work inspires future research to push the boundary further.

#### 6. Conclusion

We focus on a critical limitation of current VQ tokenizers: their difficulty in handling large codebooks. To address this, we propose a novel factorized quantization approach that decomposes a large codebook into multiple independent sub-codebooks. To facilitate learning of the sub-codebooks, we design a disentanglement regularization mechanism that reduces redundancy while promoting diversity. Additionally, we introduce a representation learning objective that explicitly guides the model to learn meaningful semantic features. The proposed visual tokenizer, FQGAN, effectively handles large codebooks and achieves state-of-theart performance in discrete image reconstruction, surpass-

ing both VQ and LFQ methods. Experimental results show that this tokenizer can be integrated into auto-regressive image generation models by adding a factorized AR head, demonstrating competitive image generation performance. Besides, we provide an in-depth analysis to unveil how the factorized codebooks function. Finally, we discuss several limitations to inspire future works.

Appendix

- A. More Generation Results

Figure 7 presents additional examples generated by our FAR model, highlighting its impressive image generation capabilities.

- B. More Training Details of Visual Tokenizers

In this section, we demonstrate the flexibility of the proposed FQGAN in terms of extending its codebook size and scaling its training schedule. Table 5 provides detailed experimental results for both FQGAN-Dual (k = 2) and FQGAN-Triple (k = 3). We observe that increasing the number of sub-codebooks from 2 to 3—effectively raising the total codebook size from 16384 × 2 to 16384 × 3—further improves reconstruction quality. With only 10 epochs of training, FQGAN-Triple achieves an rFID of 1.30, outperforming the FQGAN-Dual variant under the same training conditions. We attribute the performance gain to the larger codebook (16384 × 3), which introduces additional capacity, and the factorization design and associated training objectives enrich the new sub-codebook with more diverse features.

We observe that training the tokenizer for only 10 epochs does not fully utilize the large capacity of the subcodebooks. To address this, we extend the training schedule to further explore the capacity of the model. As shown in Tab. 5, increasing the training epochs from 10 to 40 significantly enhances performance. FQGAN-Dual improves from an rFID of 1.66 to 0.94, while FQGAN-Triple achieves an rFID of 0.76, comparable to the performance of continuous features. This study suggests that the FQGAN model has significant potential for scaling to achieve improved performance, owing to its factorization design. Importantly, training for 40 epochs does not indicate saturation. Due to limited time and resources, we did not extend training beyond 40 epochs; however, additional training could potentially yield even lower rFID values.

- C. Extended Analysis of Sub-codebooks

We present a visualization of the distribution of VQ code embeddings for the FQGAN-Triple model in Fig. 6. Specifically, FQGAN-Triple is equipped with three factorized sub-codebooks. Sub-codebook 2 is jointly supervised using

Table 5. The proposed FQGAN is extendable to multiple codebooks, i.e., k > 2, and demonstrate scaling behavior with increasing training schedule.

###### k Codebook Size Epoch rFID↓ IS↑ PSNR↑

- 2

16384 × 2 10 1.66 55.21 21.62 16384 × 2 20 1.25 56.39 22.00

- 16384 × 2 40 0.94 57.15 22.02

3

- 16384 × 3 10 1.30 56.41 21.85

16384 × 3 20 0.92 57.80 22.67 16384 × 3 40 0.76 58.05 22.73

t-SNE Visualization of Sub-Codebook 2DINO

t-SNE Visualization of Sub-Codebook 3CLIP

t-SNE Visualization of Sub-Codebook 1

| |
|---|

| |
|---|

|bull_mastiff<br><br>iron<br><br>flagpole<br><br>redshank capuchin<br><br>|
|---|

Figure 6. T-SNE visualization with FQGAN-Triple.

DINOv2 features, while sub-codebook 3 is jointly supervised using CLIP features. Following prior practice [34], we sample five classes from the ImageNet dataset and use the FQGAN-Triple model to encode these images into the latent space. Then, we use the t-SNE technique to visualize the code embeddings from each sub-codebook. We observe that the code embeddings from sub-codebook 1 appear unordered in the space, likely due to the dominant influence of high-variance details. This observation is consistent with the visualization of FQGAN-Dual in the main paper. In contrast to sub-codebook 1, the other two sub-codebooks are organized into clusters based on image categories. This clustered distribution likely reflects the influence of the representation learning objective.

Interestingly, we observe that sub-codebook 2 is distributed in a more compact manner compared to subcodebook 3, with embeddings from the same category clustering closer to the center. To better understand this phenomenon, we investigate the specific model instances and their performance on ImageNet. Specifically, we use facebook/dinov2-small and openai/clip-vit-base-patch16 checkpoints as the vision models. The DINOv2 checkpoint achieves 81.1% linear probing accuracy on the ImageNet validation set, while the CLIP checkpoint achieves 68.3% accuracy. This performance gap likely explains the observed differences in the visualization. The CLIP model is designed to capture cross-modal information between vision and language, while DINOv2 performs better in vision-centric classification tasks. These differing objectives lead the FQGANTriple model to naturally form a semantic hierarchy.

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

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

[Figure 78]

###### Figure 7. More qualitative examples generated by the FAR model.

Table 6. Exploration on multimodal understanding with LLaVA [16] framework and Flicker-30k [22] benchmark.

Feature Sub-Codebook CIDEr

Codebook 1 3.67 Codebook 2 (CLIP) 7.15

Continuous

Both 10.28

Codebook 1 2.22 Codebook 2 (CLIP) 7.40

Discrete

Both 7.37

#### D. Exploration on Multimodal Understanding

With the representation learning objective, the factorized sub-codebooks learn a semantic hierarchy, spanning from visual details to high-level concepts. Recent studies on unified multimodal models [17, 35, 36] demonstrate that their multimodal understanding performance is largely limited by traditional visual tokenizers. Compared to a standard tokenizer trained with a reconstruction objective, FQGAN demonstrates greater potential for supporting multimodal understanding tasks. We conduct a preliminary experiment to investigate its potential.

Specifically, we use a LLaVA [16]-style architecture with a Phi [1] LLM as the base model. The traditional LLaVA model undergoes two-stage training. In stage-1, a projector is trained to connect a vision model with the LLM for cross-modal alignment. In stage-2, the projector and LLM are trained jointly to develop instruction-following capabilities. In this study, we train only the stage-1 projector to evaluate the potential of vision features for crossmodal alignment. Subsequently, we evaluate the model on the Flickr-30k [22] test set using a simple image captioning task. The results are shown in Tab. 6.

Firstly, when comparing continuous features extracted from the VQ encoder to discrete code embeddings from the codebook, we observe that continuous features consistently perform better. This suggests that continuous features are still more suitable for cross-modal understanding, as they contain richer information. Secondly, when comparing different sub-codebooks, sub-codebook 2 consistently outperforms sub-codebook 1 in the captioning task. This demonstrates that joint supervision with CLIP features enhances the cross-modal potential of sub-codebook 2. We further observe that combining the two sub-codebooks results in comparable or better performance, particularly in the captioning task with continuous features. This phenomenon suggests that the visual details from sub-codebook 1 have the potential to complement information missing in subcodebook 2, enabling more effective cross-modal alignment and understanding.

The performance metrics of our model remain significantly lower than those of standard captioning models,

Table 7. Model configurations of FAR. We partially follow the scaling rule proposed in the previous work [29].

Model Parameters AR Backbone AR Head Widths Heads

FAR-Base 415M 24 3 1024 16 FAR-Large 898M 36 4 1280 20

likely due to the following factors. First, in our FQGAN model, the feature dimension is compressed to 8, which is significantly smaller than the typical dimensions of traditional vision features (512 or 768). While t-SNE visualizations demonstrate clear category separability, the features possess less semantic richness compared to the continuous representations generated by a standard vision backbone. Increasing the feature dimension further could enhance performance by capturing more detailed semantic information. Second, using CLIP features as auxiliary supervision introduces only a limited amount of cross-modal semantic information into the VQ encoder, especially given the relatively small scale of the training dataset. The VILA-U [35] study suggests that initializing the model with pre-trained CLIP weights could be a promising approach. However, it also highlights that training a single codebook to simultaneously optimize for reconstruction and semantic objectives can lead to feature conflicts.

Given these considerations, our FQGAN model shows strong potential to advance multimodal understanding and contribute to the development of unified multimodal frameworks. The factorized sub-codebook design effectively mitigates feature conflicts between high-variance visual details and high-level semantic concepts, naturally establishing a hierarchical structure. We hope this study serves as a foundation for further research in this area.

#### E. More Training Details of AR Generation Models

Table 7 presents the detailed configurations of the FARBase and FAR-Large models. The AR backbone and AR head architecture follows a standard auto-regressive transformer design with causal attention.

Next, we present the training details of the autoregressive generation model in Fig. 8. Specifically, we plot the gFID score curves using classifier-free guidance (cfg=2.0 for FAR-Dual and cfg=1.75 for FAR-Triple). Firstly, we observe that FAR scales well across different model sizes, with larger models consistently achieving better FID scores, regardless of whether dual or triple codes are used. Next, when comparing FAR-Dual and FARTriple models with the same number of parameters, we observe that FAR-Dual achieves a lower gFID score than FAR-Triple. For the “-Large” model size, FAR-Dual and FAR-Triple achieve comparable best gFID scores: 3.08 vs.

4.75

FAR-Dual-Base

FAR-Dual-Large

4.50

FID(withclassifierfreeguidance)

4.25

4.00

3.75

3.50

3.25

3.00

50 100 150 200 250 300

Training Epoches

(a) FAR-Dual generation FID with CFG.

6.5

FAR-Triple-Base

FAR-Triple-Large

6.0

FID(withclassifierfreeguidance)

5.5

5.0

4.5

4.0

3.5

3.0

50 100 150 200 250 300

Training Epoches

(b) FAR-Triple generation FID with CFG.

Figure 8. Training details of the FAR model. We demonstrate FAR-Dual and FAR-Triple with both Base and Large size.

3.09. For the “-Base” model size, FAR-Dual outperforms FAR-Triple, achieving gFID scores of 3.38 vs. 3.84. This performance gap suggests that handling multiple sub-codes in auto-regressive generation models remains challenging. The ablation study on the AR head in the main paper suggests that further scaling the AR head size could improve learning performance. We leave this for future work due to limited computational resources.

#### F. Discussion on Related Works

Our work is closely related to some existing studies. Residual Quantization [13] and Modulated Quantization [44] implicitly adopt the philosophy of factorization by decomposing visual features into primary and residual or modulated components. While this factorization improves image reconstruction performance, its potential is limited by reliance on a single codebook. Our approach explicitly decomposes a large codebook into multiple independent sub-codebooks, introducing greater flexibility and efficiency.

A concurrent work, ImageFolder [14], introduces a visual tokenization approach using two codebooks to encode semantics and details, achieving improvements in reconstruction quality. However, our work differs significantly with ImageFolder in objectives, representation design, generative modeling strategy, and downstream applicability. Below, we summarize the key differences:

- • Objective and Focus. ImageFolder focuses on improving the computational efficiency of autoregressive image generation through a Folded Tokenization mechanism, which compresses spatial information to reduce sequence length. Its primary goal is to optimize high-resolution image generation by addressing scalability challenges. In contrast, our work emphasizes building interpretable and disentangled visual representations through Factorized Tokenization, prioritizing semantic clarity and downstream usability. Our framework is designed not only for efficient generation but also for achieving more meaning-

ful and structured representations.

- • Representation Design. While ImageFolder employs two codebooks to separately encode semantics and details, it does not enforce explicit independence between these codebooks. Instead, its tokenization primarily focuses on spatial compression to maintain efficiency. In our work, we explicitly enforce independence between codebooks, disentangling semantic and detail representations. Furthermore, we introduce a hierarchical multi-codebook structure, enabling richer and more interpretable visual representations that support a broader range of tasks and downstream applications.
- • Generative Modeling Strategy. ImageFolder uses an autoregressive model to directly generate folded tokens, focusing on maintaining spatial coherence during generation. Its decoding relies on sequential predictions of compressed tokens. In contrast, our work explores how to effectively transfer the factorized tokenizer into downstream autoregressive generation tasks. By leveraging a factorized autoregressive prediction head, our framework enhances the generation process, enabling high-quality and consistent outputs. This demonstrates the adaptability of our approach for autoregressive generation tasks and highlights its advantages in maintaining both semantic integrity and visual fidelity.
- • Evaluation and Applicability. The evaluation in ImageFolder primarily measures reconstruction quality and generation efficiency using metrics like FID and inference latency. Our work takes a broader perspective, assessing the interpretability and usability of factorized representations. While our primary focus is on enhancing autoregressive generation, we also evaluate how the disentangled representations enable structured representation learning and multimodal understanding. This highlights the versatility of our approach in scenarios requiring more nuanced and interpretable models.
- • Role of Pre-trained Vision Models. Both works utilize

pre-trained vision models for supervision or regularization, but the integration differs. ImageFolder leverages these models to improve feature extraction for reconstruction tasks. In our framework, pre-trained models are utilized to guide the disentanglement process, enabling the creation of a hierarchical and factorized tokenization structure. This approach enhances the adaptability and generalizability of our representations to diverse downstream tasks.

- • Summary. While both ImageFolder and our work aim to improve visual tokenization and generation, their focus and methodologies diverge significantly. ImageFolder prioritizes computational efficiency and scalability in tokenized image generation, whereas our work introduces explicit disentanglement and hierarchical factorization for autoregressive generation. These innovations establish a more interpretable and versatile framework, extending the potential applications of visual tokenization beyond reconstruction and sequence efficiency to tasks requiring semantically meaningful and structured representations.

Together, these works, including our FQGAN, highlight factorization as a promising avenue for advancing visual tokenization and generation.

#### References

- [1] Marah Abdin, Sam Ade Jacobs, Ammar Ahmad Awan, Jyoti Aneja, Ahmed Awadallah, Hany Awadalla, Nguyen Bach, Amit Bahree, Arash Bakhtiari, Harkirat Behl, et al. Phi-3 technical report: A highly capable language model locally on your phone. arXiv preprint arXiv:2404.14219, 2024. 11
- [2] Randall Balestriero and Yann LeCun. Learning by reconstruction produces uninformative features for perception. arXiv preprint arXiv:2402.11337, 2024. 2, 4
- [3] Tom B Brown. Language models are few-shot learners. arXiv preprint arXiv:2005.14165, 2020. 1
- [4] Huiwen Chang, Han Zhang, Lu Jiang, Ce Liu, and William T. Freeman. Maskgit: Masked generative image transformer. In CVPR, pages 11305–11315, 2022. 1
- [5] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. ImageNet: A large-scale hierarchical image database. In CVPR, pages 248–255, 2009. 5
- [6] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. NeurIPS, 34:8780–8794,

2021. 5, 6

- [7] Patrick Esser, Robin Rombach, and Bj¨orn Ommer. Taming transformers for high-resolution image synthesis. In CVPR, pages 12873–12883, 2021. 1, 2, 3, 5, 6
- [8] Yuchao Gu, Xintao Wang, Yixiao Ge, Ying Shan, and Mike Zheng Shou. Rethinking the objectives of vectorquantized tokenizers for image synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7631–7640, 2024. 3
- [9] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. GANs trained by

- a two time-scale update rule converge to a local nash equilibrium. In NeurIPS, 2017. 5
- [10] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022. 5
- [11] Jonathan Ho, Chitwan Saharia, William Chan, David J Fleet, Mohammad Norouzi, and Tim Salimans. Cascaded diffusion models for high fidelity image generation. 23(1):2249–2281,

2022. 6

- [12] Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, and Alexei A. Efros. Image-to-image translation with conditional adversarial networks. In CVPR, pages 5967–5976, 2017. 3
- [13] Doyup Lee, Chiheon Kim, Saehoon Kim, Minsu Cho, and Wook-Shin Han. Autoregressive image generation using residual quantization. In CVPR, pages 11513–11522, 2022. 1, 2, 3, 6, 12
- [14] Xiang Li, Kai Qiu, Hao Chen, Jason Kuen, Jiuxiang Gu, Bhiksha Raj, and Zhe Lin. Imagefolder: Autoregressive image generation with folded tokens, 2024. 3, 12
- [15] Guotao Liang, Baoquan Zhang, Yaowei Wang, Xutao Li, Yunming Ye, Huaibin Wang, Chuyao Luo, Kola Ye, et al. Lg-vq: Language-guided codebook learning. arXiv preprint arXiv:2405.14206, 2024. 3
- [16] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36, 2024. 2, 11
- [17] Hao Liu, Wilson Yan, Matei Zaharia, and Pieter Abbeel. World model on million-length video and language with ringattention. arXiv preprint, 2024. 11
- [18] Zhuoyan Luo, Fengyuan Shi, Yixiao Ge, Yujiu Yang, Limin Wang, and Ying Shan. Open-magvit2: An open-source project toward democratizing auto-regressive visual generation, 2024. 1, 2, 3, 5, 6
- [19] Fabian Mentzer, David Minnen, Eirikur Agustsson, and Michael Tschannen. Finite scalar quantization: Vq-vae made simple. arXiv preprint arXiv:2309.15505, 2023. 1, 2
- [20] Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mahmoud Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Herv´e Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. Dinov2: Learning robust visual features without supervision, 2024. 2, 5
- [21] William Peebles and Saining Xie. Scalable diffusion models with transformers. In CVPR, pages 4195–4205, 2023. 6
- [22] Bryan A Plummer, Liwei Wang, Chris M Cervantes, Juan C Caicedo, Julia Hockenmaier, and Svetlana Lazebnik. Flickr30k entities: Collecting region-to-phrase correspondences for richer image-to-sentence models. In Proceedings of the IEEE international conference on computer vision, pages 2641–2649, 2015. 11
- [23] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. SDXL: improving latent diffusion models for high-resolution image synthesis. In ICLR, 2024. 6

- [24] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision, 2021. 5
- [25] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 1, 2, 5
- [26] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. pages 8821–8831, 2021. 6
- [27] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, pages 10684– 10695, 2022. 6
- [28] Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen. Improved techniques for training gans. In NeurIPS, 2016. 5
- [29] Peize Sun, Yi Jiang, Shoufa Chen, Shilong Zhang, Bingyue Peng, Ping Luo, and Zehuan Yuan. Autoregressive model beats diffusion: Llama for scalable image generation. arXiv preprint arXiv:2406.06525, 2024. 1, 2, 3, 5, 6, 7, 11
- [30] Keyu Tian, Yi Jiang, Zehuan Yuan, Bingyue Peng, and Liwei Wang. Visual autoregressive modeling: Scalable image generation via next-scale prediction. arXiv preprint arXiv:2404.02905, 2024. 3
- [31] Shengbang Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Manoj Middepogu, Sai Charitha Akula, Jihan Yang, Shusheng Yang, Adithya Iyer, Xichen Pan, et al. Cambrian1: A fully open, vision-centric exploration of multimodal llms. arXiv preprint arXiv:2406.16860, 2024. 5
- [32] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton-Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aur´elien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. Llama 2: Open foundation and finetuned chat models. arXiv preprint arXiv:2307.09288, 2023. 3, 5

- [33] Junke Wang, Yi Jiang, Zehuan Yuan, Binyue Peng, Zuxuan Wu, and Yu-Gang Jiang. Omnitokenizer: A joint image-video tokenizer for visual generation. arXiv preprint arXiv:2406.09399, 2024. 6
- [34] Luting Wang, Yang Zhao, Zijian Zhang, Jiashi Feng, Si Liu, and Bingyi Kang. Image understanding makes for a good tokenizer for image generation. arXiv preprint arXiv:2411.04406, 2024. 3, 6, 7, 9
- [35] Yecheng Wu, Zhuoyang Zhang, Junyu Chen, Haotian Tang, Dacheng Li, Yunhao Fang, Ligeng Zhu, Enze Xie, Hongxu Yin, Li Yi, et al. Vila-u: a unified foundation model integrating visual understanding and generation. arXiv preprint arXiv:2409.04429, 2024. 3, 6, 11
- [36] Jinheng Xie, Weijia Mao, Zechen Bai, David Junhao Zhang, Weihao Wang, Kevin Qinghong Lin, Yuchao Gu, Zhijie Chen, Zhenheng Yang, and Mike Zheng Shou. Show-o: One single transformer to unify multimodal understanding and generation. arXiv preprint arXiv:2408.12528, 2024. 11
- [37] Tackgeun You, Saehoon Kim, Chiheon Kim, Doyup Lee, and Bohyung Han. Locally hierarchical auto-regressive modeling for image generation. Advances in Neural Information Processing Systems, 35:16360–16372, 2022. 2
- [38] Jiahui Yu, Xin Li, Jing Yu Koh, Han Zhang, Ruoming Pang, James Qin, Alexander Ku, Yuanzhong Xu, Jason Baldridge, and Yonghui Wu. Vector-quantized image modeling with improved VQGAN. In ICLR, 2022. 1, 2, 6
- [39] Lijun Yu, Yong Cheng, Zhiruo Wang, Vivek Kumar, Wolfgang Macherey, Yanping Huang, David Ross, Irfan Essa, Yonatan Bisk, Ming-Hsuan Yang, et al. Spae: Semantic pyramid autoencoder for multimodal generation with frozen llms. Advances in Neural Information Processing Systems, 36, 2024. 3
- [40] Lijun Yu, Jose Lezama, Nitesh Bharadwaj Gundavarapu, Luca Versari, Kihyuk Sohn, David Minnen, Yong Cheng, Agrim Gupta, Xiuye Gu, Alexander G Hauptmann, Boqing Gong, Ming-Hsuan Yang, Irfan Essa, David A Ross, and Lu Jiang. Language model beats diffusion - tokenizer is key to visual generation. In ICLR, 2024. 1, 2, 6
- [41] Qihang Yu, Mark Weber, Xueqing Deng, Xiaohui Shen, Daniel Cremers, and Liang-Chieh Chen. An image is worth 32 tokens for reconstruction and generation. arXiv preprint arXiv:2406.07550, 2024. 6
- [42] Sihyun Yu, Sangkyung Kwak, Huiwon Jang, Jongheon Jeong, Jonathan Huang, Jinwoo Shin, and Saining Xie. Representation alignment for generation: Training diffusion transformers is easier than you think. arXiv preprint arXiv:2410.06940, 2024. 4
- [43] Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, pages 586– 595, 2018. 3
- [44] Chuanxia Zheng, Tung-Long Vuong, Jianfei Cai, and Dinh Phung. Movq: Modulating quantized vectors for highfidelity image generation. Advances in Neural Information Processing Systems, 35:23412–23425, 2022. 2, 12
- [45] Lei Zhu, Fangyun Wei, and Yanye Lu. Beyond text: Frozen large language models in visual signal comprehension. In

Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 27047–27057, 2024. 3

[46] Lei Zhu, Fangyun Wei, Yanye Lu, and Dong Chen. Scaling the codebook size of vqgan to 100,000 with a utilization rate of 99%. arXiv preprint arXiv:2406.11837, 2024. 1, 2, 3, 6

