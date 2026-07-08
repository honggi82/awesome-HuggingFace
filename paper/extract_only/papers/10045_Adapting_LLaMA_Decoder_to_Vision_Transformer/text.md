# arXiv:2404.06773v4[cs.CV]27May2024

## Adapting LLaMA Decoder to Vision Transformer

Jiahao Wang1, Wenqi Shao2*, Mengzhao Chen2, Chengyue Wu1, Yong Liu3, Taiqiang Wu1, Kaipeng Zhang2, Songyang Zhang2, Kai Chen2, Ping Luo1∗

1The University of HongKong. 2Shanghai AI Laboratory. 3Tsinghua Shenzhen International Graduate School. jiahao.wang@connect.hku.hk, shaowenqi@pjlab.org.cn

### Abstract

This work examines whether decoder-only Transformers such as LLaMA, which were originally designed for large language models (LLMs), can be adapted to the computer vision field. We first "LLaMAfy" a standard ViT step-by-step to align with LLaMA’s architecture, and find that directly applying a causal mask to the self-attention brings an attention collapse issue, resulting in the failure to the network training. We suggest to reposition the class token behind the image tokens with a post-sequence class token technique to overcome this challenge, enabling causal self-attention to efficiently capture the entire image’s information. Additionally, we develop a soft mask strategy that gradually introduces a causal mask to the self-attention at the onset of training to facilitate the optimization behavior. The tailored model, dubbed as image LLaMA (iLLaMA), is akin to LLaMA in architecture and enables direct supervised learning. Its causal self-attention boosts computational efficiency and learns complex representation by elevating attention map ranks. iLLaMA rivals the performance with its encoder-only counterparts, achieving 75.1% ImageNet top-1 accuracy with only 5.7M parameters. Scaling the model to ∼310M and pre-training on ImageNet21K further enhances the accuracy to 86.0%. Extensive experiments demonstrate iLLaMA’s reliable properties: shape-texture bias, calibration, quantization compatibility, ADE20K segmentation and CIFAR transfer learning. We hope our study can kindle fresh views to visual architectures in the wave of LLMs and inspire the development of unified multimodal models. Pre-trained models and codes are available https://github.com/techmonsterwang/iLLaMA.

### 1 Introduction

The year 2024 saw the meteoric rise of large language models (LLMs), as well as the 4th anniversary of the Vision Transformer (ViT) [13]. Born in 2020, ViT was influenced by the prevailing encoderonly text Transformers at the time, such as BERT [11], etc. Accordingly, ViT is allowed to borrow the encoder-only design, i.e., self-attention do not use any causal mask. As a result, advanced vision backbones [47, 55, 20] and training paradigms [3, 22] have followed such convention by default.

Meanwhile, the development of text Transformers did not stand still. A series of LLMs with a decoder-only architecture (e.g., LLaMA [49]), have sparked a new wave. Pre-trained decoder-only Transformers have demonstrated remarkable performance in diverse textual [50] and multimodal tasks [33, 32, 67, 6]. In this context, designing unified architectures for language and vision is a promising direction. Specifically, unified models employ shared types of operators to process both visual and textual data, reducing the cost of hardware implementation. Moreover, compared to using separate models for image and text, unified models [4, 29] simplify the inference process by handling different modalities simultaneously, thereby improving practical efficiency.

∗Corresponding author.

Preprint. Under review.

###### iLLaMA Block

###### Causal Self-Attention

###### Image LLaMA (iLLaMA)

ViT (Baseline) MLP → SwiGLU LN → RMSNorm

3.451 3.407 3.406

73.8 74.3 74.5

81.5

###### Class

Soft Mask

L×

82.0 81.7

Cls Head

| |
|---|

Training

SwiGLU

RMSNorm

Bi-di. SA → Causal SA Failed

iLLaMA

O

Transformer Decoder

71.9 72.6 73.2 74.3 75.0

+ Post-Sequence [cls] Learnable PE → RoPE

3.599

80.6

3.618 3.531

81.2 81.2

Causal Self-Attention

0 * … N *

1

Softmax

+ Learnable PE Data Augmentation

Post-Sequence [cls]

| | |
|---|---|
| | |

RMSNorm

2.990 2.955

81.3 81.6

Soft Mask

Image Patch Embedding

RoPE RoPE

+ Soft Mask

Image Patch + Learnable Positional Embedding

Embedded Patches

Q K V

- Figure 1: Left: iLLaMA architecture. Right: our design roadmap. Colored and gray bars represent the results of the tiny and base regimes, with the red line depicting the training loss of the tiny regime. iLLaMA strives to process visual tokens using standard LLaMa components, e.g., causal self-attention. The proposed PS [cls] and soft mask strategy help overcome training challenges. Block details of ViT [13], VisionLLaMA [8], and our iLLaMA is compared in Figure 5 in Appendix A.

Toward this goal, we took the initial step of exploring whether decoder-only Transformers can hold its effectiveness in the unimodal vision domain. In this study, we demonstrate that through straightforward supervised learning, LLaMA architecture itself can process input images with simple yet crucial modifications. We start by modifying a standard encoder-only ViT (e.g., ViT-T/16), progressively adapting its components to align with those in LLaMA. In practice, we observe an attention collapse issue, i.e., the training loss fails to converge by directly adding a causal mask to the attention map. The causal mask restricts the class token from accessing the image’s global information, thereby hindering the optimization of the training loss. To this end, we propose a postsequence class token technique, repositioning the class token to the end of image tokens (details in

- Section 3.3). As a result, causal mask can keep the attention score between the class token and others, allowing the model to optimize stably. We also evaluate the advantages of the causal self-attention in reducing computational complexity and enhancing the attention map rank.

Moreover, we explore several training techniques for the proposed causal Transformer. When observing things, humans start by broadly catching global connections, then narrow down to focus on specifics. Motivated by this, we develop a soft mask approach – bi-directional self-attention degenerates to a causal self-attention at the onset of training – to further boost the network performance. Soft mask does not alter the causal self-attention during inference but improves the initial training behavior of the network. (details in Section 3.6). We illustrate different types of masks in Figure 3.

Equipped with these modifications, we propose a decoder-only vision Transformer with causal self-attention inside, dubbed image LLaMA (iLLaMA), as shown in Figure 1. We conduct a thorough evaluation of iLLaMA’s properties, including ImageNet-1K classification [10], calibration, shapetexture bias, quantization compatibility, ADE20K semantic segmentation [66], and CIFAR transfer learning [28]. Experimental results show that iLLaMA delivers favorable and reliable performance to its encoder-only counterparts (i.e., ViT, VisionLLaMA), while maintaining a pure decoder design. More importantly, a spectral analysis on the attention map empirically shows that compared to bi-directional counterparts, causal self-attention has a higher rank (see Figure 4), which allows for learning complex image representation. The contribution of our work can be summarized as follows:

- • We explore the adaption of the LLaMA decoder to visual tasks and identify the attention collapse issue caused by causal self-attention. To address this, we introduce the PS [CLS] method.
- • We propose a soft mask strategy to optimize causal self-attention and analyzed the improvement in the rank of the causal attention map.
- • We develop a series of iLLaMA models and empirically validate its performance on ImageNet, along with practical properties such as quantization compatibility, calibration, shape-texture bias.

We hope our work to inspire a re-evaluation of vision backbone design in the era of LLMs and provide fresh insights for their architectural unification.

### 2 Preliminaries and Motivation

Encoder and decoder. We briefly summarize the encoder and decoder in Transformer [52]. Both of them basically consist of attention module and a MLP module, each followed by a residual

connection. The key difference between them is the mask scheme in their self-attention. Encoders use bi-directional self-attention, and decoders employ causal self-attention and cross-attention. However, the latter is typically omitted in decoder-only LLMs [49, 50], we thus focus on comparing causal and bi-directional self-attention as follows, in terms of the mask setting. Denote X ∈ RN×d,O ∈ RN×d as the input and output sequences, where N is the number of tokens and d is the embedding dimension. Wq,Wk,Wv ∈ Rd×d denotes the linear mapping of query, key and value, respectively. Generally, self-attention module can be formulated as (set the head number and batch size as 1 for simplicity):

0, i ≥ j −∞, i < j

1 √

(Wq(X) · Wk(X)⊤), O = Softmax(A + M) · Wv(X), Pi,j = 0, Qi,j =

(1)

A =

d

where i,j ∈ [1,N], A ∈ RN×N, M ∈ RN×N denote the attention map and mask. P ∈ RN×N, Q ∈ RN×N are masks in the encoder and decoder, respectively. For a causal self-attention, we have M = Q. Such design allows subsequent tokens only attend to the preceding ones, but not vice versa. For a bi-directional self-attention, we have M = P, ensuring mutual visibility for each token.

Recent LLMs-related image models. Recent image models [2, 21, 14] are trained with an autoregressive objective, targeting at solving visual tasks. Pang et al. [38] add a text pre-trained frozen LLM block to a ViT encoder to facilitate the performance. Our work, on the other hand, is motivated to explore in-depth how the decoder design in LLMs can be adapted to image models using simple supervised learning to achieve an architectural alignment. A concurrent work VisionLLaMA [8] proposes vision models for recognition and generation tasks based on the LLaMA components. Differently, we: 1) introduce causal self-attention from LLaMA, addressing the associated attention collapse issue, while VisionLLaMA retains an encoder architecture; 2) develop a soft mask technique to assist training the decoder; 3) expand the dataset to the larger ImageNet-21K to demonstrate scalability, achieving 86.0% ImageNet accuracy that outperforms VisionLLaMA’s best results.

### 3 A Roadmap: Solving Attention Collapse and Optimization Improvement

This section introduces the design roadmap of iLLaMA. As we aim to adapt LLMs to vision, we choose LLaMA [49] and ViT [13] as language and vision baselines due to their successful practices. The trajectory can be divided into two dimensions, i.e., architecture (Section 3.1-3.4) and training techniques (Section 3.5-3.6). First, we focus on block designs including 1) feed foward network, 2) normalization layer, 3) self-attention, 4) positional embedding, illustrated in Figure 1. Next, we study training techniques and develop a soft mask strategy to facilitate optimization. Finally, we provide an analysis in terms of efficiency and attention map rank (Section 3.7). We start with ViT-T/16 and ViTB/16 with around 5.7M and 86.4M parameters, respectively, and gradually replace the corresponding components with those from LLaMA. We conduct experiments on ImageNet-1K [10], following the training recipe adopted from [36] (details in Table 12 of Appendix C.1). Considering the differences between visual perception and text generation tasks, we maintain ViT’s non-autoregressive manner in our network. Each step change and the corresponding results are reported in Table 17 of Appendix D.

#### 3.1 Feed Forward Network (FFN)

FFN in Transformer are implemented differently in ViT and LLaMa, i.e., multi-layer perceptron (MLP) and SwiGLU [43]. MLP consists of two sequential linear mappings, with a GELU [24] function inserted. Meanwhile, SwiGLU combines three linear mappings, integrating a SiLU [24, 15, 40] function. This structure allows for the modulation of high-dimensional features through a gating mechanism. We substitute the Transformer’s MLPs with SwiGLUs, while maintaining comparable computational cost. As shown in Figure 1, this improves performance from 73.8% to 74.3%, and from 81.3% to 82.0% for the ViT-T/16 and ViT-B/16 regime. This highlights SwiGLU’s effectiveness not only in language models but also in vision, inspiring further exploration of other components.

We will now use SwiGLU to substitute MLP in each block.

#### 3.2 Normalization Layer

Transformers need normalization layer for stable training, i.e., layer normalization (LN) [1] in ViT and root mean square layer normalization (RMSNorm) [62] in LLaMA, respectively. We replace all LNs with RMSNorms in our network and empirically observed that the accuracy of the ViT-T/16

[cls] image tokens

image tokens [cls]

[cls] image tokens

| | | | | |
|---|---|---|---|---|

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| |
|---|
| |
| |
| |
| |

| |
|---|
| |
| |
| |
| |

[cls]

[cls]

[cls]

(a) causal mask (b) causal mask w/ PS [cls] (ours) (c) modified causal mask (ablation)

- Figure 2: (a) mask in causal self-attention. (b) mask in causal self-attention with our post-sequence class token (PS [cls]) method. (c) modified causal mask. Their ablation results are shown in Table 1.

regime increased from 74.3% to 74.5%. However, similar improvements in precision were not observed in the ViT-B/16 regime (from 82.0% to 81.7%). Nonetheless, compared to LN, RMSNorm removes the shift term computation, bringing simplicity to the network [50, 7, 42, 27].

We will use RMSNorm instead of LN as the normalization layer in each block.

#### 3.3 Causal Self-Attention Leads to Attention Collapse

Attention collapse issue. As a common practice for Transformer decoders, the key component for causal self-attention is the causal mask, i.e., a lower triangular mask matrix, illustrated in Eq. 1 and Figure 2(a). With such, each token can get the attention score of all its previous ones. We add the causal mask to our network via a non-autoregressive way. The reason is that visual perception tasks, unlike text generation, require only inference once. As a result, we observe that the training loss fails to converge in both ViT-T/16 and ViT-B/16 regimes (line 1 in Table 1). We posit that such issue stems from the influence of the lower triangular matrix, which prevents the class token from "seeing" other image tokens. As illustrated in Figure 2(a), when the class token is positioned at the start of the patch embedding, its attention score for all other image tokens gets zero due to a causal mask. We term this occurrence as the attention collapse issue, which leads to a loss of connection between the class token and other image patches, thereby hindering the optimization of the network.

Post-sequence class token (PS [cls]). The attention collapse issue stems from the inappropriate placement of the token. To this end, we suggest a PS [cls] strategy, by placing it at the end of the token sequence, without changing the causal mask, as shown in Figure 2(b) and Figure 1. Such modification ensures that the class token can achieve global information about all image tokens, while maintaining a causal self-attention property. As a result, we observe that the attention collapse issue is eliminated and the training process starts to stabilize, leading the network performance to 71.9% for ViT-T/16 and 80.6% for ViT-B/16 regime, respectively (line 2 in Table 1).

Table 1: Results of PS [cls] and the modified causal mask. Training converges in both settings.

Model Tiny Train Loss Base Train Loss

None 0.1 Failed 0.1 Failed PS [cls] 71.9 3.599 80.6 2.869 Modified 72.5 3.550 80.4 2.857

To test our hypothesis about the reason of the attention collapse issue, we also explore a mask setting in Figure 2(c). In this setting, we do not change the position of the class token. Instead, we unmask the first row of the mask (i.e., attention score of the class token) on the basis of the causal self-attention, termed as "modified causal mask". Ablation results (line 3 in Table 1) shows that both settings can solve the attention collapse issue as expected, and the "modified causal mask" leads to a better 72.5% accuracy for ViT-T/16 regime, validating our hypothesis about the reason. Although the results do not surpass the performance of bi-directional counterpart, they demonstrate the potential for optimizing causal self-attention in a decoder-only image model. We also observe that the PS [cls] method yields higher accuracy with a slightly larger training loss for ViT-B/16 regime, suggesting lower overfitting.

We will employ causal self-attention with the proposed PS [cls] method in each block.

#### 3.4 Positional Embedding

A standard ViT use learnable positional embedding (LPE) to preserve positional information, typically adding it directly to the patch embedding. Meanwhile, rotary positional embedding (RoPE) [45] is widely employed in LLMs [49, 50], which functions in the attention of each block. We first use RoPE alone, which boosts the accuracy of ViT-T/16 and ViT-B/16 regimes to 72.6% and 81.2%, from 71.9% and 80.6%, respectively. The encouraging results illustrate that the concepts of "position" in image

Tiny model: p74, 147

Bi-directional mask Causal mask

| |[Figure 1]<br><br>Soft mask ends<br><br>| | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

Start End

Training ViT

iLLaMA w/o soft mask

Constant schedule

iLLaMA w/ soft mask

Linear schedule

iLLaMA w/ soft mask

(a) soft mask scheme (b) training curves w/ or w/o soft mask

- Figure 3: (a) Soft mask gradually transitions from a bi-directional mask into a causal mask during training through a constant or linear schedule. (b) Ablation results of training loss and test accuracy.

and text do not exist an inherent gap. Since LPE functions only once before all Transformer blocks, keeping it does not disrupt the alignment with LLaMA within each block. Thus, we reintroduce the LPE, which improves the accuracy of ViT-T/16 regime from 72.6% to 73.2%, suggesting that the two positional embeddings are not redundant but rather synergistic, enhancing network performance.

We will use both LPE and RoPE for positional embedding. So far, we have investigated each block component, and thus fix the final architecture dubbed iLLaMA. Next, we explore training strategies.

#### 3.5 Data Augmentation

Mixup [63] and cutmix [61] that we used to train our iLLaMA (0.8 and 1.0), are borrowed from DeiT [47]’s recipe. Unlike the bi-directional self-attention used in DeiT, causal self-attention affects the connection between image tokens. Meanwhile, these two hyper-parameters affect the content of the input image, which further influences the subsequent embedding. Thus, we reevaluate their impact on iLLaMA optimization. Specifically, we discover that a combination of 0.1 mixup and 0.1 cutmix improves the performance of the iLLaMA-T/16 from 73.2% to 74.3%, whereas a combination of 0.95 and 1.0 leads the iLLaMA-B/16 to a 81.3% accuracy. Other ablations are detailed in Section 4.1.

#### 3.6 Soft Mask Strategy: Optimization Improvement

When observing objects, humans tend to perceive broad connections, then focus on specific details. Motivated by this, we propose a soft mask technique to improve the model’s optimization: starting with bi-directional self-attentions in the early training epochs and gradually shifting completely to causal self-attentions as the optimization goes. Specifically, self-attention can be formulated as:

1 √

(Wq(X) · Wk(X)⊤), O = (Softmax(A) ⊙ S) · Wv(X),

A =

d

1, i ≥ j 0, i < j

S = αB + (1 − α)C, Bi,j = 1, Ci,j =

(2)

where i,j ∈ [1,N], S ∈ RN×N denotes the soft mask, which is defined as a linear combination of a bi-directional mask B and a causal mask C. α is the hyper-parameter controlling the mask configuration, i.e., soft mask degenerates into B or C when α = 1 or α = 0, respectively. As illustrated in Figure 3(a), α involves three related hyper-parameters: 1) scheme: how α drops from 1 to 0: we try a linear or a constant scheme. 2) cutoff epochs: when will α drops to 0. 3) learning rate (lr) warmup [23, 18]: this hyper-parameter overlaps with the duration of soft mask. We initially set the lr warmup epochs at 50, consistent with previous settings. When using a linear scheme with 50 and 25 cutoff epochs, we observe an improvement in performance for both iLLaMA-T/16 and iLLaMA-B/16 models, reaching 74.9% and 81.6% from 74.3% and 81.3%, respectively. Ablations results are detailed in Section. 4.1. To intuitively observe the impact of soft mask, we plot the training curve of the iLLaMA-T/16 in Figure 3(b), using a constant scheme with 50 cutoff epochs. When soft mask ends, we observe that although there was a sharp drop in accuracy, the model ends up achieving better performance. Similar case of the iLLaMA-B/16 are shown in Figure 6 of Appendix F. Additionally, we discover that a lower learning rate warmup helps iLLaMA-T/16 achieve 75.0% top-1 accuracy, by using a constant scheme with 50 cutoff epochs. Therefore, we use this warmup method for iLLaMA-T/16. Notably, the final training loss within both iLLaMA-T/16 and iLLaMA-B/16 decreases when using soft masks, suggesting a mitigation of potential underfitting concerns.

#### 3.7 Analysis of causal Self-attention

Next, we analyze the advantages of using causal self-attention in iLLaMA, in terms of computational efficiency and expressive ability of visual representation through the lens of attention map rank.

Computational Complexity. We compare the efficiency of causal self-attention and bi-directional baseline. For a self-attention with a sequence length N and embedding dimension D, FLOPs are reported in Table 2 (RoPE is not involved as only attention computations are considered). causal self-attention, in light of the lower triangular property of its attention map, slightly reduces the FLOPs compared to the bi-directional baseline — the degree of reduction grows as sequence length increases.

Attention map rank. We examine the representation learning power of causal attention through a spectrum analysis. Following [54, 44], we perform singular value decomposition on the attention maps of the pre-trained ViT-T/16 and iLLaMA-T/16 models. Next, we sort the singular values and plot a curve illustrating the relationship between the cumulative normalized singular values and matrix indices. The results are conducted using 30 images randomly selected from the ImageNet-1K validation set. As shown in Figure 4, the curve of ViT exhibits concave function characteristics, while the curve of iLLaMA is close to a linear function, indicating a more uniform distribution of singular values in iLLaMA’s attention map. Approximating the matrix rank by the index at which the cumulative normalized singular value reaches 0.8, we observe that the index value of iLLaMA is about 48 higher than that of ViT (∼129-th v.s. ∼81-th). Under such premise, compared to ViT, the attention map of iLLaMA can be approximated with a certain error by a higher-rank matrix. Accordingly, the rank of the attention map may affect the expressive capabilities of the learned representations [12], suggesting that the causal self-attention in iLLaMA has the potential to learn complex visual representations, as empirically demonstrated in

Table 2: Computational complexity results. causal mask slightly reduces FLOPs required in the self-attention.

Type Bi-directional causal FLOPs 4ND2 + 2N2D 4ND2 + N2D + (⌊N2/2⌋ + 1)D

- Section 4.2. Detailed results for different layers are provided in Figure 11 of Appendix E.

Closing remarks. So far, we have finished the design roadmap of iLLaMA through architectural and training strategy modification. iLLaMA, a decoderonly Transformer, shows advantages in computational complexity and attention map rank through its causal self-attention mechanism. Notably, while all components of iLLaMA are essentially derived from LLaMA, relying only on them is insufficient for an effective weight optimization, as demonstrated in Section 3.3. In fact, the proposed PS [cls] and soft mask strategy effectively address this issue and assist in iLLaMA training. However, to achieve a comprehensive understanding of iLLaMA’s properties, some useful evaluation should be conducted: 1) Scalability for large model capacities (>300M parameters) and dataset sizes (>10M training images, e.g., ImageNet-21K). 2) Other practical evaluation dimensions, such as model calibration, shape-texture bias, downstream task performance, quantization compatibility, discussed below.

[Figure 2]

difference ≈ 48

Figure 4: Rank analysis of the attention map in head 1, layer 1 of the pretrained ViT-T and iLLaMA-T with N = 197. Difference between them is about 48.

### 4 Experiments

This section provide a comprehensive evaluation of iLLaMA. We first report ablation results, e.g., the effectiveness of data augmentation and different soft mask strategies. Next, we compare iLLaMA with other strong baselines on ImageNet classification. Beyond ImageNet accuracy, we also examine its efficacy on calibration, shape-texture bias, and evaluate its compatibility with quantization-aware training and downstream task performance.

Table 3: Soft mask scheduling for iLLaMAT/16 and iLLaMA-B/16 on ImageNet-1K.

Schedule Cutoff Epochs Tiny Base no softmask - 74.3 81.3 linear 25 74.8 81.6 linear 50 74.9 81.5 linear 100 74.9 81.5 constant 25 74.7 81.5 constant 50 74.8 81.5

#### 4.1 Ablation Study

Table 4: Soft mask for training loss and testing loss. Soft mask lowers both training and testing loss in tiny and base models, counteracting underfitting issue and thus leading to a better optimization.

Model Training Loss Testing Loss tiny 2.990 1.121

+ soft mask 2.955 (↓ 0.045) 1.092 (↓ 0.029) base 2.868 0.843

+ soft mask 2.828 (↓ 0.040) 0.831 (↓ 0.012)

Influence of data augmentation. Base on the observation in Section 3.5, we examined multiple sets of cutmix and mixup settings, as reported in Table 5. We empirically observe that the smaller iLLaMA-T/16 are more sensitive to two data augmentation strategies and perform better with lower hyper-parameters, whereas the larger iLLaMA-B/16 are suited to higher ones. This may be related to the architectural differences between LLaMA’s Transformer decoder and ViT’s encoder type.

Influence of soft mask scheduling strategies and epochs. As mentioned in Section 3.6, the proposed soft mask technique includes three hyper-parameters, i.e., schedule, cutoff epochs and lr warmup epochs. Here we evaluate the robustness of soft mask to hyper-parameter settings, with results detailed in Table 3. Beyond the linear schedule, inspired by [36], we also implemented a constant option. Additionally, we fixed the learning rate warm-up epochs at 50 and experimented with different cutoff epochs. The results reveal that the soft mask facilitates the optimization of iLLaMA under both linear and constant scheduling, suitable for models of both tiny and base sizes. Moreover, setting the cutoff epochs to span a wide range from 25 to 100 is advantageous. Notably, the soft mask can be easily integrated into existing code frameworks (e.g., timm [56]) with negligible additional training costs, thereby facilitating its effortless application on future related architectures.

Influence of soft mask for training and testing loss. Deep neural networks often face underfitting, marked by difficulty in continuously reducing training loss and resulting in poor test accuracy [36]. We compare the training and testing losses of the iLLaMA-T/16 and iLLaMA-B/16 models with and without the use of the soft mask strategy. As shown in Table 4, soft mask can reduce training loss in both regimes, mitigating potential underfitting issue and reducing testing loss.

Table 5: Mixup and cutmix ablation results.

Mixup Cutmix Tiny Mixup Cutmix Base

0.8 1.0 73.2 0.8 1.0 81.2 0.5 0.4 73.8 0.9 0.9 81.2 0.3 0.3 73.9 0.9 1.0 81.2 0.2 0.2 74.3 1.0 1.0 81.2 0.1 0.1 74.3 0.95 1.0 81.3

#### 4.2 Comparison with Recent Architectures on ImageNet-1K Classification

We conducted experiments on the ImageNet-1K [10] benchmark with different model sizes (i.e., iLLaMA-T/S/B/L). Detailed architecture configurations are shown in Table 11 of Appendix A. Our ImageNet-1K/21K (pre-)training and ImageNet-1K fine-tuning recipes are shown in Table 13 and Table 14 of Appendix C. We also study the use of LLaMA2-7B pre-trained weights for iLLaMA initialization, and the paradigm and results are detailed in Figure 9 and Table 18 in Appendix I.

ImageNet-1K training. We train iLLaMA-T/S/B on ImageNet-1K for 300 epochs with AdamW optimizer [37] and a batch size of 4096. The ImageNet-1K trained iLLaMA-T/B models are, in fact, the outcome of the explorations completed in Section 3.6. For the settings of soft mask schedule, cutoff epochs, and learning rate warmup epochs, we tune slightly for the iLLaMA-S model.

ImageNet-21K pre-training. We use the ‘Winter21 variant of ImageNet-21K-P’ (refered to as ImageNet-21K) dataset [41] 2 for the large-scale pre-training of our iLLaMA, which contains 11,060,223 training images and 522,500 testing images from 10,450 classes. Only the training set was used. We pre-train iLLaMA-B/L on ImageNet-21K for 90 epochs using a constant soft mask schedule, with cutoff epochs and learning rate warmup epochs set to 30 and 5, respectively.

ImageNet-1K fine-tuning. For iLLaMA-B model trained on ImageNet-1K, we fine-tune at a resolution of 384×384. Similarly, for the iLLaMA-B/L model trained on ImageNet-21K, we fine-

2downloaded from: https://www.image-net.org/download-images.php

- Table 6: ImageNet-1K accuracy. Throughput (images/s) are tested on Nvidia A100 GPU with a batch size of 1024. Hie.: Hierarchical, Iso.: Isotropic, Sup.: Supervised (pre-)training, AR.: Autoregressive pre-training. ♠ ConvNet, ■ Vision Transformer, ♣ MLP, ✠ LM-inspired visual model, ⋆ LLaMA. Model Dataset Used Objective Type Image Size Params MACs Throughput Acc

♠ ConvNeXt-S [35] IN-1K Sup. Hie. 224×224 50M 8.7G 1185 83.1 ♠ ConvNeXt-B [35] IN-1K Sup. Hie. 224×224 89M 15.4G 877 83.8 ♠ ConvNeXt-L [35] IN-1K Sup. Hie. 224×224 198M 34.4G 543 84.3 ♠ ConvNeXtV2-N [57] IN-1K Sup. Hie. 224×224 15.6M 2.45G 2120 81.2 ♠ ConvNeXtV2-T [57] IN-1K Sup. Hie. 224×224 28.6M 4.47G 1362 82.5 ♠ ConvNeXtV2-B [57] IN-1K Sup. Hie. 224×224 88.7M 15.4G 645 84.3

- ■ Swin-S [34] IN-1K Sup. Hie. 224×224 50M 8.7G 934 83.0

- ■ Swin-B [34] IN-1K Sup. Hie. 224×224 88M 15.4G 710 83.5

- ■ DeiT-Ti [47] IN-1K Sup. Iso. 224×224 5.7M 1.3G 6051 72.2

- ■ DeiT-S [47] IN-1K Sup. Iso. 224×224 22.1M 4.6G 3080 79.8

- ■ DeiT-B [47] IN-1K Sup. Iso. 224×224 86.4M 17.6G 1348 81.8

- ■ ViT-B/16 [13] IN-21K, IN-1K Sup., Sup. Iso. 384×384 86.4M 55.5G 349 84.0

- ■ ViT-L/16 [13] IN-21K, IN-1K Sup., Sup. Iso. 384×384 304.1M 191.2G 124 85.2

♣ PoolFormer-S12 [60] IN-1K Sup. Hie. 224×224 12M 1.8G 4354 77.2 ♣ PoolFormer-M48 [60] IN-1K Sup. Hie. 224×224 73M 11.6G 768 82.5 ♣ VanillaNet-5 [5] IN-1K Sup. Hie. 224×224 15.5M 5.2G - 72.5 ♣ VanillaNet-13-1.5×[5] IN-1K Sup. Hie. 224×224 127.8M 26.5G - 82.5

✠ AIM-0.6B [14] DFN-2B+, IN-1K AR., Sup. Iso. 224×224 0.6B - - 78.5 ✠ AIM-3B [14] DFN-2B+, IN-1K AR., Sup. Iso. 224×224 3B - - 82.2 ✠ AIM-7B [14] DFN-2B+, IN-1K AR., Sup. Iso. 224×224 7B - - 82.4 ✠ P-VisionLLaMA-S [8] IN-1K Sup. Hie. 224×224 24M - - 81.6 ✠ P-VisionLLaMA-B [8] IN-1K Sup. Hie. 224×224 56M - - 83.2 ✠ P-VisionLLaMA-L [8] IN-1K Sup. Hie. 224×224 99M - - 83.6 ✠ VisionLLaMA-L [8] IN-1K, IN-1K Sup., Sup. Iso. 224×224 310M - - 84.6

⋆ iLLaMA-T IN-1K Sup. Iso. 224×224 5.7M 1.3G 6958 75.0 ⋆ iLLaMA-S IN-1K Sup. Iso. 224×224 21.9M 4.6G 3222 79.9 ⋆ iLLaMA-B IN-1K Sup. Iso. 224×224 86.3M 17.6G 1345 81.6 ⋆ iLLaMA-B IN-1K Sup. Iso. 384×384 86.3M 55.5G 332 83.0 ⋆ iLLaMA-B IN-21K, IN-1K Sup., Sup. Iso. 224×224 86.3M 17.6G 1345 83.6 ⋆ iLLaMA-B IN-21K, IN-1K Sup., Sup. Iso. 384×384 86.3M 55.5G 332 85.0 ⋆ iLLaMA-L IN-21K, IN-1K Sup., Sup. Iso. 224×224 310.2M 62.8G 456 84.8 ⋆ iLLaMA-L IN-21K, IN-1K Sup., Sup. Iso. 384×384 310.2M 194.7G 116 86.0

tune at resolutions of 224×224 and 384×384, respectively. All fine-tuning was conducted for 30 epochs using the AdamW optimizer. We follow DeiT [47] for interpolating positional embeddings to allow our iLLaMA to handle inputs at a higher resolution.

Results. Table 6 shows a comparison between iLLaMA and other strong visual baselines, including ConvNets (ConvNeXt [35], ConvNeXt-V2 [57]), vision Transformers (ViT [13], Swin Transformer [34]), MLPs (PoolFormer [60], VanillaNet [5]), and recent language model inspired models (AIM [14], VisionLLaMA [8]). We present three observations: 1) The performance-parameter tradeoff of iLLaMA surpasses other LM-inspired models such as AIM and VisionLLaMA, presumably due to its use of causal attention and soft mask training techniques. 2) iLLaMA exhibits a superior accuracy-throughput trade-off compared to strong hierarchical baselines such as ConvNeXt-V2N/T/B and Swin-S/B. We attribute this to iLLaMA’s isotropic design (each intermediate block has the same feature resolution), which benefits from a straightforward and efficient architecture, enhancing inference speed. 3) Scalability of model capacity and dataset size: After comprehensive pre-training on the expanded ImageNet-21K dataset, the iLLaMA-B model achieves more than 85.0% accuracy on ImageNet-1K with under 100M parameters, significantly outperforming ViT-B’s 84.0%. Upon scaling up to the larger iLLaMA-L, accuracy reaches 86.0%, exceeding that of ViT-L pre-trained on ImageNet-21K and the AIM-7B pre-trained on the DFN-2B+ dataset. To our knowledge, this showcases SOTA performance for LLaMA-type architectures.

#### 4.3 Model Calibration and Shape-Texture Bias

Beyond ImageNet accuracy, we also examined iLLaMA’s calibration properties and shape-texture bias for a more detailed evaluation, following [53]. Besides iLLaMA, we also explore two preva-

- Table 7: Quantization results. #Bits (wa): w bit weights, a bit activations. 8-bit iLLaMA-T matches 32-bit DeiT-T.

Model #Bits Tiny Small

DeiT [47] 32-32 72.2 79.8 iLLaMA 32-32 75.0 79.9 iLLaMA 8-8 72.4 77.4

Table 8: Calibration (expected calibration error ↓) and shape-texture bias (ratio ↑) results of ConvNeXt-B [35], DeiT3-B [48] and iLLaMA-B. We use both IN-21K pretrained and IN-1K fine-tuned models.

Evaluation ConvNeXt-B DeiT3-B iLLaMA-B

Calibration 0.0281 0.0415 0.0335 Shape-Texture Bias 33.30% 39.86% 41.45%

Table 9: Soft mask for CIFAR transfer learning. Soft mask improves iLLaMA performance without changing the inference architecture.

Model CIFAR10 CIFAR100

ViT-T 98.0 85.5 iLLaMA-T 97.9 84.8 + soft mask 97.9 85.5

Table 10: ADE20K semantic segmentation results using UperNet [58]. We report mIoU with multi-scale testing. FLOPs calculation are based on input sizes of (512, 512).

Backbone Input Crop. mIoU #Param. FLOPs ViT-T 5122 39.8 10.88M 37.1G iLLaMA-T 5122 37.7 10.86M 37.1G ViT-B 5122 47.3 163.29M 585.7G iLLaMA-B 5122 45.1 163.22M 585.7G

lent architectures, i.e., ConvNeXt [35] and DeiT3 [48], representing ConvNets and Transformers, respectively. We apply ImageNet-21K pre-trained and ImageNet-1K fine-tuned models in this section.

Model calibration. Model calibration represents the relationship between a model’s precision and confidence across samples of varying difficulty, i.e., poor-calibrated models tend to produce overly confident yet incorrect predictions, whereas well-calibrated models demonstrate a strong correlation between confidence and accuracy [19]. Calibration is commonly measured using the Expected Calibration Error (ECE), where a lower ECE is favorable. ECE results for different models on ImageNet-1K are presented in Table 8. The calibration of iLLaMA is lower than that of DeiT3, suggesting a more reliable output confidence. We also plot the reliability diagrams [53] to intuitively compare the calibration of different models, as shown in Figure 7 of Appendix G.

Shape-texture bias. Shape-texture bias measures the extent to which the model relies on the shape or texture of the image when performing recognition [17]. We generally prefer models to mimic human eye behavior, relying more on shape rather than texture [51, 16]. We calculate the shape ratio for all models on cue-conflict images and report the results in Table 8, following [53]. Our iLLaMA shows the largest shape ratio of 41.45% among the three compared baselines, suggesting the potential of the LLM architecture for vision. Detailed results are provided in Figure 8 of Appendix H.

#### 4.4 Compatibility with Quantization

Since a practical goal for neural networks is deployment on low-bit hardware chips, we further examine iLLaMA’s compatibility with quantization. We basically follow Q-ViT [31] to apply quantization-aware training (QAT) to iLLaMA, with weights and activations of all blocks’ FFN and causal self-attention layers to 8 bits. Quantization recipes and results are shown in Table 15 of Appendix C.4 and Table 7. Different sizes of low-bit iLLaMA maintain accuracy well, and 8-bit iLLaMA-T is even compete favorably with the full-precision DeiT-T [47] (72.4% v.s. 72.2%).

#### 4.5 Transferability on Downstream Tasks

CIFAR transfer learning. We fine-tune ViT-T and iLLaMA-T on the CIFAR datasets [28], including an ablation of the soft mask on iLLaMA. Detailed recipes are shown in Table 16 of Appendix C.5. iLLaMA’s performance on CIFAR datasets is essentially on par with ViT, assuring that iLLaMA can be confidently applied in the transfer learning field as a practical alternative to ViT. Additionally, soft mask is helpful in the relatively complicated CIFAR100, demonstrating its generalizability.

ADE20K semantic segmentation. We fine-tune our ImageNet-1K pre-trained iLLaMA and ViT models on ADE20K [66] dataset using UperNet [58] to perform semantic segmentation task. For both iLLaMA and ViT, we set the learning rate as 6e-5 and weight decay as 0.01. Table 10 presents the results. iLLaMA’s performance is marginally lower than ViT’s, which we attribute to the potential impact of the masking mechanism in iLLaMA’s causal attention on high-resolution dense prediction tasks. This suggests there is still space for optimization, a subject for future investigation.

### 5 Conclusions

In this paper, we systematically studies whether Transformer decoder, an architecture that has shown amazing potential in LLMs, can also take root in learning visual representation through straightforward supervised training. The key component – causal self-attention we used – is not novel and is inherited from existing LLM architectures, but we propose pivotal techniques, i.e., PS [cls] and soft mask strategies, to effectively adapt them to visual tasks. The proposed iLLaMA outperforms many ConvNets, ViTs, and MLPs on imagenet, and demonstrates robust quantization compatibility, calibration, and shape-texture bias, thereby showing its practicality. We hope that this work will inspire a rethinking of generic yet practical architecture that can fully unify both vision and text.

### References

- [1] Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization. arXiv preprint arXiv:1607.06450, 2016.
- [2] Yutong Bai, Xinyang Geng, Karttikeya Mangalam, Amir Bar, Alan Yuille, Trevor Darrell, Jitendra Malik, and Alexei A Efros. Sequential modeling enables scalable learning for large vision models. arXiv preprint arXiv:2312.00785, 2023.
- [3] Hangbo Bao, Li Dong, Songhao Piao, and Furu Wei. Beit: Bert pre-training of image transformers. In ICLR, 2021.
- [4] Rohan Bavishi, Erich Elsen, Curtis Hawthorne, Maxwell Nye, Augustus Odena, Arushi Somani, and Sa˘gnak Ta¸sırlar. Introducing our multimodal models, 2023.
- [5] Hanting Chen, Yunhe Wang, Jianyuan Guo, and Dacheng Tao. Vanillanet: the power of minimalism in deep learning. In NeurIPS, 2023.
- [6] Jun Chen, Deyao Zhu, Xiaoqian Shen, Xiang Li, Zechun Liu, Pengchuan Zhang, Raghuraman Krishnamoorthi, Vikas Chandra, Yunyang Xiong, and Mohamed Elhoseiny. Minigpt-v2: large language model as a unified interface for vision-language multi-task learning. arXiv preprint arXiv:2310.09478, 2023.
- [7] Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E. Gonzalez, Ion Stoica, and Eric P. Xing. Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality, March 2023.
- [8] Xiangxiang Chu, Jianlin Su, bo Zhang, and Chunhua Shen. Visionllama: A unified llama interface for vision tasks. arXiv preprint arXiv:2403.00522, 2024.
- [9] Ekin D Cubuk, Barret Zoph, Jonathon Shlens, and Quoc V Le. Randaugment: Practical automated data augmentation with a reduced search space. In CVPR Workshops, 2020.
- [10] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In CVPR, 2009.
- [11] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In NAACL, 2019.
- [12] Yihe Dong, Jean-Baptiste Cordonnier, and Andreas Loukas. Attention is not all you need: Pure attention loses rank doubly exponentially with depth. In ICML, 2021.
- [13] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. In ICLR, 2020.
- [14] Alaaeldin El-Nouby, Michal Klein, Shuangfei Zhai, Miguel Angel Bautista, Alexander Toshev, Vaishaal Shankar, Joshua M Susskind, and Armand Joulin. Scalable pre-training of large autoregressive image models. arXiv preprint arXiv:2401.08541, 2024.
- [15] Stefan Elfwing, Eiji Uchibe, and Kenji Doya. Sigmoid-weighted linear units for neural network function approximation in reinforcement learning. Neural networks, 2018.
- [16] Robert Geirhos, Jörn-Henrik Jacobsen, Claudio Michaelis, Richard Zemel, Wieland Brendel, Matthias Bethge, and Felix A Wichmann. Shortcut learning in deep neural networks. Nature Machine Intelligence, 2020.
- [17] Robert Geirhos, Patricia Rubisch, Claudio Michaelis, Matthias Bethge, Felix A Wichmann, and Wieland Brendel. Imagenet-trained cnns are biased towards texture; increasing shape bias improves accuracy and robustness. arXiv preprint arXiv:1811.12231, 2018.
- [18] Priya Goyal, Piotr Dollár, Ross Girshick, Pieter Noordhuis, Lukasz Wesolowski, Aapo Kyrola, Andrew Tulloch, Yangqing Jia, and Kaiming He. Accurate, large minibatch sgd: Training imagenet in 1 hour. arXiv preprint arXiv:1706.02677, 2017.

- [19] Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q Weinberger. On calibration of modern neural networks. In ICML, 2017.
- [20] Jianyuan Guo, Kai Han, Han Wu, Yehui Tang, Xinghao Chen, Yunhe Wang, and Chang Xu. Cmt: Convolutional neural networks meet vision transformers. In CVPR, 2022.
- [21] Jianyuan Guo, Zhiwei Hao, Chengcheng Wang, Yehui Tang, Han Wu, Han Hu, Kai Han, and Chang Xu. Data-efficient large vision models through sequential autoregression. arXiv preprint arXiv:2402.04841, 2024.
- [22] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross Girshick. Masked autoencoders are scalable vision learners. In CVPR, 2022.
- [23] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In CVPR, 2016.
- [24] Dan Hendrycks and Kevin Gimpel. Gaussian error linear units (gelus). arXiv preprint arXiv:1606.08415, 2016.
- [25] Geoffrey E Hinton, Nitish Srivastava, Alex Krizhevsky, Ilya Sutskever, and Ruslan R Salakhutdinov. Improving neural networks by preventing co-adaptation of feature detectors. arXiv preprint arXiv:1207.0580, 2012.
- [26] Gao Huang, Yu Sun, Zhuang Liu, Daniel Sedra, and Kilian Q Weinberger. Deep networks with stochastic depth. In ECCV, 2016.
- [27] Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. Mistral 7b. arXiv preprint arXiv:2310.06825, 2023.
- [28] Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. 2009.
- [29] Bo Li, Peiyuan Zhang, Jingkang Yang, Yuanhan Zhang, Fanyi Pu, and Ziwei Liu. Otterhd: A highresolution multi-modality model. arXiv preprint arXiv:2311.04219, 2023.
- [30] Hao Li, Zheng Xu, Gavin Taylor, Christoph Studer, and Tom Goldstein. Visualizing the loss landscape of neural nets. In Advances in neural information processing systems, 2018.
- [31] Yanjing Li, Sheng Xu, Baochang Zhang, Xianbin Cao, Peng Gao, and Guodong Guo. Q-vit: Accurate and fully quantized low-bit vision transformer. In NeurIPS, 2022.
- [32] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. arXiv preprint arXiv:2310.03744, 2023.
- [33] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In NeurIPS, 2023.
- [34] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In ICCV, 2021.
- [35] Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, and Saining Xie. A convnet for the 2020s. In CVPR, 2022.
- [36] Zhuang Liu, Zhiqiu Xu, Joseph Jin, Zhiqiang Shen, and Trevor Darrell. Dropout reduces underfitting. In ICML, 2023.
- [37] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In ICLR, 2019.
- [38] Ziqi Pang, Ziyang Xie, Yunze Man, and Yu-Xiong Wang. Frozen transformers in language models are effective visual encoder layers. arXiv preprint arXiv:2310.12973, 2023.
- [39] Boris T Polyak and Anatoli B Juditsky. Acceleration of stochastic approximation by averaging. SIAM journal on control and optimization, 1992.
- [40] Prajit Ramachandran, Barret Zoph, and Quoc V Le. Searching for activation functions. In ICLR Workshop, 2018.
- [41] Tal Ridnik, Emanuel Ben-Baruch, Asaf Noy, and Lihi Zelnik-Manor. Imagenet-21k pretraining for the masses. arXiv preprint arXiv:2104.10972, 2021.
- [42] Baptiste Roziere, Jonas Gehring, Fabian Gloeckle, Sten Sootla, Itai Gat, Xiaoqing Ellen Tan, Yossi Adi, Jingyu Liu, Tal Remez, Jérémy Rapin, et al. Code llama: Open foundation models for code. arXiv preprint arXiv:2308.12950, 2023.
- [43] Noam Shazeer. Glu variants improve transformer. arXiv preprint arXiv:2002.05202, 2020.
- [44] Han Shu, Jiahao Wang, Hanting Chen, Lin Li, Yujiu Yang, and Yunhe Wang. Adder attention for vision transformer. In NeurIPS, 2021.
- [45] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 2024.

- [46] Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jon Shlens, and Zbigniew Wojna. Rethinking the inception architecture for computer vision. In CVPR, 2016.
- [47] Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, and Hervé Jégou. Training data-efficient image transformers & distillation through attention. In ICML, 2021.
- [48] Hugo Touvron, Matthieu Cord, and Hervé Jégou. Deit iii: Revenge of the vit. In ECCV, 2022.
- [49] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.
- [50] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.
- [51] Shikhar Tuli, Ishita Dasgupta, Erin Grant, and Thomas L Griffiths. Are convolutional neural networks or transformers more like human vision? arXiv preprint arXiv:2105.07197, 2021.
- [52] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In NIPS, 2017.
- [53] Kirill Vishniakov, Zhiqiang Shen, and Zhuang Liu. Convnet vs transformer, supervised vs clip: Beyond imagenet accuracy. arXiv preprint arXiv:2311.09215, 2023.
- [54] Sinong Wang, Belinda Z Li, Madian Khabsa, Han Fang, and Hao Ma. Linformer: Self-attention with linear complexity. arXiv preprint arXiv:2006.04768, 2020.
- [55] Wenhai Wang, Enze Xie, Xiang Li, Deng-Ping Fan, Kaitao Song, Ding Liang, Tong Lu, Ping Luo, and Ling Shao. Pyramid vision transformer: A versatile backbone for dense prediction without convolutions. In ICCV, 2021.
- [56] Ross Wightman. Pytorch image models. https://github.com/rwightman/ pytorch-image-models, 2019.
- [57] Sanghyun Woo, Shoubhik Debnath, Ronghang Hu, Xinlei Chen, Zhuang Liu, In So Kweon, and Saining Xie. Convnext v2: Co-designing and scaling convnets with masked autoencoders. In CVPR, 2023.
- [58] Tete Xiao, Yingcheng Liu, Bolei Zhou, Yuning Jiang, and Jian Sun. Unified perceptual parsing for scene understanding. In ECCV, 2018.
- [59] Zhiqiu Xu, Yanjie Chen, Kirill Vishniakov, Yida Yin, Zhiqiang Shen, Trevor Darrell, Lingjie Liu, and Zhuang Liu. Initializing models with larger ones. In ICLR, 2024.
- [60] Weihao Yu, Mi Luo, Pan Zhou, Chenyang Si, Yichen Zhou, Xinchao Wang, Jiashi Feng, and Shuicheng Yan. Metaformer is actually what you need for vision. In CVPR, 2022.
- [61] Sangdoo Yun, Dongyoon Han, Seong Joon Oh, Sanghyuk Chun, Junsuk Choe, and Youngjoon Yoo. Cutmix: Regularization strategy to train strong classifiers with localizable features. In ICCV, 2019.
- [62] Biao Zhang and Rico Sennrich. Root mean square layer normalization. In NeurIPS, 2019.
- [63] Hongyi Zhang, Moustapha Cisse, Yann N Dauphin, and David Lopez-Paz. mixup: Beyond empirical risk minimization. In ICLR, 2018.
- [64] Yiyuan Zhang, Xiaohan Ding, Kaixiong Gong, Yixiao Ge, Ying Shan, and Xiangyu Yue. Multimodal pathway: Improve transformers with irrelevant data from other modalities. arXiv preprint arXiv:2401.14405, 2024.
- [65] Zhun Zhong, Liang Zheng, Guoliang Kang, Shaozi Li, and Yi Yang. Random erasing data augmentation. In AAAI, 2020.
- [66] Bolei Zhou, Hang Zhao, Xavier Puig, Tete Xiao, Sanja Fidler, Adela Barriuso, and Antonio Torralba. Semantic understanding of scenes through the ade20k dataset. IJCV, 2019.
- [67] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. arXiv preprint arXiv:2304.10592, 2023.

### A Network Configuration

In Table 11, we provide detailed architecture configurations for iLLaMA models of various capacities. Our approach to scaling up the model size, from small to large, is similar to that of the ViT. Thus, akin to ViT, iLLaMA benefits from the simplicity of an isotropic architecture and high throughput, with its internal features remaining unchanged in resolution and number of channel as the depth increases.

Table 11: Detailed iLLaMA architecture configurations.

We provide a block-level comparison between iLLaMA and ViT model in Figure 5. VisionLLaMA uses SwiGLU, and AS2D RoPE to build LLaMA-style architecture. Differently, we further uses RMSNorm, modified causal self-attention and 1D RoPE from LLaMA to replace layer normalization, bi-directional self-attention, and proposes two pivotal strategies, i.e., PS [cls] and soft mask to help the optimization of our iLLaMA. We also keep the learnable positional embedding as ViT does.

Tiny (T) Small (S) Base (B) Large (L)

depth 12 12 12 24 embedding dim 192 384 768 1024 number of heads 3 6 12 16 #param. (M) 5.7 21.9 86.3 310.2 MACs (G) 1.3 4.6 17.6 62.8

###### ViT Block

###### VisionLLaMA Block

iLLaMA Block

L×

L×

L×

MLP

SwiGLU

SwiGLU

Layer Norm

Layer Norm

RMSNorm

Bi-Directional Self-Attention

Causal Self-Attention

Bi-Directional Self-Attention

AS2D RoPE

1D RoPE

Layer Norm

Layer Norm

RMSNorm

Embedded Patches

Embedded Patches

Embedded Patches

Figure 5: Comparison between ViT [13], VisionLLaMA [8], and iLLaMA blocks.

### B PyTorch-like Code of iLLaMA Causal Self-attention

The PyTorch-like implementation of our iLLaMA causal self-attention is shown as Algorithm 1. The iLLaMA code exhibits a high degree of similarity in structure and composition to the official LLaMA code 3 released by Meta, potentially offering considerable coding cost savings in developing a unified vision and language network with such architecture.

### C Experimental Settings

#### C.1 Training Recipe in Section 3

Our training recipe for training the tiny and base models during the “designing iLLaMA: a roadmap” (Section 3) is primarily adapted from ConvNeXt [35, 36], summarized in Table 12.

Basically, both regimes use the same experimental setup, with the only difference being the stochastic depth rate at 0.0 and 0.4, respectively. Notably, for the ViT baseline, our experimental results are 73.8% and 81.5%, as shown in Table 17, which slightly differ from the results of 73.9% and 81.6% reported in [36].

Utilizing only the basic training recipe with architectural modifications, the performance of iLLaMA’s tiny and base models achieves 73.2% and 81.2%, as shown in Table 17, yet remains below the ViT baseline. We attribute this to the impairing effect of causal self-attention on the information mixing among tokens. Thus, we enhance the training recipe, detailed next.

3https://github.com/meta-llama/llama

Algorithm 1 PyTorch code of iLLaMA causal self-attention

import torch import torch.nn as nn

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor): ndim = x.ndim assert 0 <= 1 < ndim assert freqs_cis.shape == (x.shape[1], x.shape[-1]) shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)] return freqs_cis.view(*shape)

def apply_rotary_emb(

xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor,

) -> Tuple[torch.Tensor, torch.Tensor]:

xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2)) xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2)) freqs_cis = reshape_for_broadcast(freqs_cis, xq_)

xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3) xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3) return xq_out.type_as(xq), xk_out.type_as(xk)

class Attention(nn.Module): def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,

proj_drop=0.): super().__init__() self.num_heads = num_heads head_dim = dim // num_heads # NOTE scale factor was wrong in my original version, can set manually to be

compat with prev weights self.scale = qk_scale or head_dim ** -0.5 self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias) self.proj = nn.Linear(dim, dim)

def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: Optional[torch.

Tensor]): B, N, C = x.shape qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2,

0, 1, 3, 4) # [3, B, N, self.num_heads, C // self.num_heads] q, k, v = qkv[0], qkv[1], qkv[2] # make torchscript happy (cannot use tensor as

tuple) # [B, N, self.num_heads, C // self.num_heads] q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)

q = q.transpose(1, 2) # [B, self.num_heads, N, C // self.num_heads] k = k.transpose(1, 2) # [B, self.num_heads, N, C // self.num_heads] v = v.transpose(1, 2) # [B, self.num_heads, N, C // self.num_heads] attn = (q @ k.transpose(-2, -1)) * self.scale # [B, self.num_heads, N, N] attn = attn.softmax(dim=-1) if mask is not None:

attn = attn * mask # (B, H, N, N)

x = (attn @ v).transpose(1, 2).reshape(B, N, C) x = self.proj(x)

return x

#### C.2 ImageNet (Pre-)training Recipe

As illustrated in Table 13, we provide the detailed ImageNet-1K training hyper-parameters and ImageNet-21K pre-training hyper-parameters for the experimental results in Table 6.

For the iLLaMA-T/S/B models, we train directly on ImageNet-1K and discover that models of different sizes are suited to different soft mask settings. For instance, the soft mask schedules are set to constant/linear/linear, respectively, with cutoff epochs designated as 50/50/25. We train the iLLaMA-T/S/B models using 8 A100 GPUs.

We pre-trained the iLLaMA-B/L models on ImageNet-21K for 90 epochs, adhering to the practices in [35]. We set the cutoff epochs to 30, indicating that the iLLaMA models’ self-attention fully transitions to causal self-attention after 30 epochs. We pre-train the iLLaMA-B/L models using 8 A100 GPUs.

Table 12: Our training recipe for Section 3 in the main paper, adapted from [36].

Training Configuration iLLaMA-T/B Initialization: weight init trunc. normal (0.2) Training recipe: optimizer AdamW [37] optimizer momentum β1, β2=0.9, 0.999 Learning hyper-parameters: base learning rate 4e-3 learning rate schedule cosine decay weight decay 0.05 batch size 4096 training epochs 300 lr warmup epochs 50 warmup schedule linear gradient clip None exp. mov. avg. (EMA) [39] None Dropout: dropout rate [25] 0.0 stochastic depth rate [26] 0.0/0.4 Data augmentation:

input resolution 2242 randAugment [9] (9, 0.5) random erasing [65] 0.25 label smoothing [46] 0.1 mixup [63] 0.8 cutmix [61] 1.0

#### C.3 ImageNet Fine-tuning Recipe

We present the results of fine-tuning models pre-trained on ImageNet-1K at a resolution of 384×384, as well as the outcomes of fine-tuning models pre-trained on ImageNet-21K at resolutions of 224×224 and 384 × 384, as shown in Table 14. All ImageNet-1K fine-tuning experiments were conducted for 30 epochs, following the convention in [35].

For the iLLaMA-B model pre-trained on ImageNet-1K, we used a relatively higher stochastic depth rate of 0.8. For the iLLaMA-B/L models pre-trained on ImageNet-21K, we employed relatively lower stochastic depth rates of 0.2 and 0.3, respectively.

Additionally, we standardized the cutoff epoch at 0 for our ImageNet-1K fine-tuning experiments, ensuring the application of a causal mask in self-attention to align with the LLaMA architecture. We also opted not to use learning rate warmup. We fine-tune the models using 8 A100 GPUs.

#### C.4 Quantization-aware Training Recipe

We provide our quantization-aware training recipe for iLLaMA in Table 15. Basically we follow the Q-ViT method proposed in [31], with only weights and activations in each basic block’s causal self-attention and FFN module are quantized to 8 bit width.

#### C.5 CIFAR Transfer Learning Recipe

We further provide our training recipe for transfer learning on the CIFAR10 and CIFAR100 datasets, as shown in Table 16.

In our transfer learning experiments, we consistently apply a linear soft mask schedule. However, for the CIFAR10 and CIFAR100 datasets, we use cutoff epochs of 25 and 50, respectively.

- Table 13: Our (pre-)training settings for iLLaMa model on ImageNet-1K/ImageNet-21K, respectively, adapted from [36]. Some key training techniques are highlighted .

iLLaMA-T/S/B iLLaMA-B/L (Pre-)Training Configuration ImageNet-1K ImageNet-21K Initialization: weight init trunc. normal (0.2) trunc. normal (0.2) Training recipe: optimizer AdamW AdamW optimizer momentum β1, β2=0.9, 0.999 β1, β2=0.9, 0.999 Learning hyper-parameters: base learning rate 4e-3 1e-3 learning rate schedule cosine decay cosine decay weight decay 0.05 0.01 batch size 4096 4096 training epochs 300 90 warmup schedule linear linear gradient clip None None exp. mov. avg. (EMA) None None Dropout:

dropout rate 0.0 0.0 stochastic depth rate 0.0/0.1/0.4 0.1

Data augmentation: input resolution 2242 2242 randAugment (9, 0.5) (9, 0.5) random erasing 0.25 0.25 label smoothing 0.1 0.1 mixup 0.1/0.5/0.95 0.8 cutmix 0.1/0.5/1.0 1.0

Soft mask: soft mask schedule constant/linear/linear constant cutoff epochs 50/50/25 30 lr warmup epochs 5/5/50 5

### D Designing iLLaMA: detailed results

We present the comprehensive experimental results of our exploration journey of iLLaMA in Table 17. This table not only delineates the stepwise accuracy of both the tiny and base models, as depicted in Figure 1, but also outlines the training loss at each step. The general trend observed is that as the training loss of the models decreases, their accuracy increases.

Overall, the trend in changes for the base model is broadly similar to that of the tiny model. However, in contrast to the tiny model, the implementation of RoPE coupled with subsequent integration of LPE does not affect the base model’s performance. This lack of impact, we theorize, stems from the base regime’s reduced susceptibility to underfitting compared to the tiny regime, hence the addition of extra learnable parameters offers less benefit to its performance.

Notably, vanilla causal self-attention mechanism proves inadequate for model optimization, an issue effectively addressed by implementing the PS [CLS] method. Additionally, the application of the soft mask technique significantly contributes to the training efficacy of both model sizes.

### E Rank Analysis of causal Self-attention

Detailed visualization results. We provide rank analysis results of all 3 heads in layer 1, 4, 8, 12 of ViT-T/16 and iLLaMA-T/16 in Figure 11. Besides the observation in Section 3.7, We make four observations: 1) Not each head in each layer of iLLaMA’s self-attention shows stronger concavity, suggesting that not every attention matrix of iLLaMA has a higher rank than its ViT counterpart. 2) In most cases, particularly in the shallow layers, the distribution of singular values in iLLaMA

- Table 14: Our fine-tuning settings for iLLaMa model on ImageNet-1K, adapted from [36]. Some key training techniques are highlighted .

iLLaMA-B iLLaMA-B/L iLLaMA-B/L (Pre-)Training Configuration

ImageNet-1K ImageNet-21K ImageNet-21K 2242 2242 2242 Fine-Tuning Configuration ImageNet-1K ImageNet-1K ImageNet-1K Initialization: weight init trunc. normal (0.2) trunc. normal (0.2) trunc. normal (0.2) Training recipe: optimizer AdamW AdamW AdamW optimizer momentum β1, β2=0.9, 0.999 β1, β2=0.9, 0.999 β1, β2=0.9, 0.999 Learning hyper-parameters: base learning rate 8e-5 8e-5/6e-5 1.1e-4/3.5e-5 learning rate schedule cosine decay cosine decay cosine decay weight decay 1e-8 1e-8 1e-8 batch size 512 512 512 training epochs 30 30 30 warmup schedule linear linear linear gradient clip None None None exp. mov. avg. (EMA) None None None Dropout: dropout rate 0.0 0.0 0.0 stochastic depth rate 0.8 0.2/0.3 0.2/0.3 Data augmentation:

input resolution 3842 2242 3842 randAugment (9, 0.5) (9, 0.5) (9, 0.5) random erasing 0.25 0.25 0.25 label smoothing 0.1 0.1 0.1 mixup 0 0 0 cutmix 0 0 0

Soft mask: soft mask schedule constant constant constant cutoff epochs 0 0 0 lr warmup epochs 0 0 0

appears more uniform than in ViT. 3) In certain attention maps (e.g., layer 8, head 2, and layer 8, head 3), the rank of ViT’s attention matrix is low, resulting in an skewed distribution of information. In contrast, such extreme cases were not observed in our iLLaMA. 4) The distribution of singular values in ViT varies significantly across different layers and heads (e.g., layer 1, head 1; layer 4, head 1; layer 8, head 1; layer 8, head 2), whereas iLLaMA’s distribution appears relatively more stable.

### F Analysis for Soft Mask Method

In this section, we plot the training results for iLLaMA-B/16 with and without the use of the soft mask technique in Figure 6. We can observe that the results display a similar pattern to those of iLLaMA-T/16 (Figure 3(b)).

We set the cutoff epochs to 50 and used a constant schedule. When the soft mask ends, there is a sudden increase in training loss and a steep decline in model accuracy. However, the final accuracy surpasses the baseline, and the training loss is also optimized to a lower value. Such phenomenon shows the versatility of the soft mask technique across models of varying capacities, and shows that causal self-attention can still effectively model even when a portion of the attention map is masked.

- Table 15: Our quantization-aware training settings for iLLaMa model on ImageNet-1K, adapted from [36, 31]. Some key training techniques are highlighted .

iLLaMA-T/S (Pre-)Training Configuration ImageNet-1K Initialization: weight init trunc. normal (0.2) Training recipe: optimizer AdamW optimizer momentum β1, β2=0.9, 0.999 Learning hyper-parameters: base learning rate 3e-3/4e-3 learning rate schedule cosine decay weight decay 0.05 batch size 4096 training epochs 300 warmup schedule linear gradient clip None exp. mov. avg. (EMA) None Dropout: dropout rate 0.0 stochastic depth rate 0.0/0.1 Data augmentation:

input resolution 2242 randAugment (9, 0.5) random erasing 0.25 label smoothing 0.1 mixup 0.1/0.5 cutmix 0.1/0.5

Soft mask: soft mask schedule constant/linear cutoff epochs 50/50 lr warmup epochs 5/5

- Table 16: Our transfer learning settings for ViT-T and iLLaMa-T model on CIFAR10/100, respectively, adapted from [59]. Some key training techniques are highlighted .

Transfer Learning Configuration CIFAR10 CIFAR100 For both ViT-T and iLLaMA-T: base learning rate 2e-3 2e-3 batch size 1024 1024 training epochs 300 300 stochastic depth rate 0.0 0.0 lr warmup epochs 50 50 For iLLaMA-T only: soft mask schedule linear linear cutoff epochs 25 50

### G Model Calibration

To qualitatively present the calibration property, we plot the reliability diagrams of ConvNeXt-B, DeiT3-B and the proposed iLLaMA-B using ImageNet-1K in Figure 7, following [53]. For wellcalibrated models, the direction of accuracy in their reliability diagrams show a roughly diagonal pattern, i.e., the difference between accuracy and confidence is small. Intuitively, the confidence of the early bins of the iLLaMA presents results below the accuracy level, indicating that iLLaMA

Base model: 103, 148

| |[Figure 3]<br><br>Soft mask ends<br><br>| | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

- Figure 6: Training curves for iLLaMA-B/16 regime w/ and w/o soft mask. When soft mask ends, the model experiences a similar pattern to the training curve of iLLaMA-T/16 regime, with eventually a lower test loss observed.

- Table 17: ImageNet-1K classification accuracy via gradually replacing components in ViT-T/16 and ViT-B/16 with counterparts in LLaMA, better or worse than the ViT baseline results with our basic training recipe. Components from or modified from LLaMA are highlighted . P.E.: positional embedding, Bd.: bi-directional self-attention, Cs.: causal self-attention.

Ablation FFN Norm Attention P.E. Tiny Train Loss Base Train Loss ViT [47] MLP LN Bd. LPE 72.2 - 81.8 -

results with our basic training recipe

ViT MLP LN Bd. LPE 73.8 3.451 81.5 2.828 + LLaMa FFN SwiGLU LN Bd. LPE 74.3 3.407 82.0 2.724 + LLaMa Norm SwiGLU RMS Bd. LPE 74.5 3.406 81.7 2.721 + LLaMa S.A. SwiGLU RMS Cs. LPE 0.1 Failed 0.1 Failed + LLaMa S.A. SwiGLU RMS Cs. + PS [CLS] LPE 71.9 3.599 80.6 2.869 + LLaMa P.E. SwiGLU RMS Cs. + PS [CLS] RoPE 72.6 3.618 81.2 2.861 + LPE P.E. SwiGLU RMS Cs. + PS [CLS] RoPE + LPE 73.2 3.531 81.2 2.839

modify the training techniques

+ data aug. SwiGLU RMS Cs. + PS [CLS] RoPE + LPE 74.3 2.990 81.3 2.868 + soft mask SwiGLU RMS Cs. + PS [CLS] RoPE + LPE 75.0 2.955 81.6 2.828

tends to be underconfident. This observation, akin to that observed in the DeiT3, may be a common characteristic of Transformer-based architectures and was also noted in [53].

### H Shape-Texture Bias

We visualize the shape-texture bias results on cue-conflict images of ConvNeXt-B, DeiT3-B and the proposed iLLaMA-B in Figure 8, following [53]. The three dashed lines of different colors represent the average shape decision of the three models over all categories. Generally, a more leftward average shape ratio indicates that the model relies more on global shape information for recognition tasks. iLLaMA shows higher shape scores relative to ConvNeXt and DeiT3.

### I Initializating iLLaMA Using Pre-trained LLaMA

Previous studies [64] have demonstrated that data unrelated to the image modality can be used to improve the performance of visual models. In fact, the pre-training dataset of LLaMA, which is entirely text, is irrelevant to the visual tasks that iLLaMA addresses. More important, the architectural components of iLLaMA are aligned with those of LLaMA. This alignment facilitates our exploration

convnext_21k imagenet

deit3_21k imagenet

illama_21k imagenet

1.0

1.0

1.0

ECE: 0.0281

ECE: 0.0415

ECE: 0.0335

0.8

0.8

0.8

0.6

0.6

0.6

Accuracy

Accuracy

Accuracy

0.4

0.4

0.4

0.2

0.2

0.2

0.0

0.0

0.0

0.2 0.4 0.6 0.8 1.0

0.2 0.4 0.6 0.8 1.0

0.2 0.4 0.6 0.8 1.0

Confidence

Confidence

Confidence

(a) ConvNeXt-B (b) DeiT3-B (c) iLLaMA-B

- Figure 7: Calibration results of (a) ConvNeXt-B (b) DeiT3-B and (c) iLLaMA-B pretrained on ImageNet-21K and fine-tuned on ImageNet-1K.

[Figure 4]

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

[Figure 19]

1 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 0

Fraction of shape decisions

0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 Fraction of texture decisions

Shapecategories

ConvNeXt-sup

DeiT3-sup

iLLaMA-sup

- Figure 8: Shape-texture bias results of ConvNeXt-B, DeiT3-B and iLLaMA-B pre-trained on ImageNet-21K and fine-tuned on ImageNet-1K. sup: supervised learning paradigm.

of using LLaMA’s parameters to initialize iLLaMA, allowing us to fully exploit the potential of the weights of pre-trained LLMs.

We use the pre-trained LLaMA2-7B [50] to initialize our iLLaMA, instead of training from scratch. To match the dimensions of the weights, we employ the weight selection [59] method to initialize iLLaMAT/S/B using a subset of the LLaMA2-7B pre-trained weights. Next, we train and evaluate the iLLaMA models, which are initialized using LLaMA2-7B, on the ImageNet-1K dataset. Other hyperparameter settings are consistent with Section 4.2. The results are shown in Table 18. Using LLaMA2 to initialize iLLaMA does not yield significant performance improvements. We attribute this to two main reasons: 1) The size difference between the two models is considerably large (LLaMA-2-7B’s 7B parameters vs. iLLaMA-T’s 5.7M parameters), resulting in a limited proportion of selected weights compared to meaningful pre-trained weights. 2) The training strategy was not adequately optimized. We believe that when using parameter inheritance, the corresponding training strategy should also be adjusted. However, we continued to use the training recipe designed for training from scratch.

Table 18: Ablation results of weight selection of iLLaMA using LLaMA2-7B pre-trained weights.

Model Initialization Tiny Small Base iLLaMA w/ weight selection 74.5 79.9 81.4 iLLaMA w/o weight selection 75.0 79.9 81.6

###### LLaMA2 Block

###### iLLaMA Block

Initialization

SwiGLU

SwiGLU

RMSNorm

RMSNorm

Causal Self-Attention

Causal Self-Attention

1D RoPE

1D RoPE

RMSNorm

RMSNorm

Embedded Patches

Embedded Patches

Initialization

LLaMA2-7B iLLaMA

Figure 9: iLLaMA initialization by pre-trained LLaMA2-7B [50] using weight selection [59].

### J Loss Landscape

10

10

8

8

6

6

4

4

2

2

1.0

1.0

0.5

0.5

1.0

1.0

0.0

0.0

0.5

0.5

0.5

0.5

0.0

0.0

0.5

0.5

1.0

1.0

1.0

1.0

(a) ViT-T/16 (b) iLLaMA-T/16 Figure 10: Loss landscape illustration of (a) ViT-T/16 and (b) iLLaMA-T/16.

As shown in Figure 10, we visualized the loss landscape [30] of the iLLaMA-T/16 and ViT-T/16. The loss landscape of ViT and iLLaMA exhibits similar patterns, with the steepness and bumps observed in ViT seeming to be softened.

### K Limitations

We have shown that the LLaMA architecture, enhanced by the developed post-sequence [cls] and soft mask techniques, is adept at adapting to tasks such as visual recognition and semantic segmentation. However, iLLaMA’s application remains predominantly within the realm of perception. In fact, such decoder-only architecture, favored by LLMs in the NLP field, can do more complex tasks, such

ViT-T

1.0

iLLaMa-T

normalizedcumulativesingularvalue

0.8

0.6

0.4

0.2

0.0

0 25 50 75 100 125 150 175 200 singular value index

(a) layer 1, head 1

ViT-T

1.0

iLLaMa-T

normalizedcumulativesingularvalue

0.8

0.6

0.4

0.2

0.0

0 25 50 75 100 125 150 175 200 singular value index

(b) layer 1, head 2

ViT-T

1.0

iLLaMa-T

normalizedcumulativesingularvalue

0.8

0.6

0.4

0.2

0.0

0 25 50 75 100 125 150 175 200 singular value index

(c) layer 1, head 3

ViT-T

1.0

iLLaMa-T

normalizedcumulativesingularvalue

0.8

0.6

0.4

0.2

0.0

0 25 50 75 100 125 150 175 200 singular value index

(d) layer 4, head 1

ViT-T

1.0

iLLaMa-T

normalizedcumulativesingularvalue

0.8

0.6

0.4

0.2

0 25 50 75 100 125 150 175 200 singular value index

(e) layer 4, head 2

ViT-T

1.0

iLLaMa-T

normalizedcumulativesingularvalue

0.8

0.6

0.4

0.2

0.0

0 25 50 75 100 125 150 175 200 singular value index

(f) layer 4, head 3

ViT-T

1.0

iLLaMa-T

normalizedcumulativesingularvalue

0.8

0.6

0.4

0.2

0.0

0 25 50 75 100 125 150 175 200 singular value index

(g) layer 8, head 1

1.0

0.9

normalizedcumulativesingularvalue

0.8

0.7

0.6

0.5

0.4

0.3

ViT-T

0.2

iLLaMa-T

0 25 50 75 100 125 150 175 200 singular value index

(h) layer 8, head 2

| |ViT-T<br><br>iLLaMa-T| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |

1.0

normalizedcumulativesingularvalue

0.8

0.6

0.4

0.2

0.0

0 25 50 75 100 125 150 175 200 singular value index

(i) layer 8, head 3

ViT-T

1.0

iLLaMa-T

0.9

normalizedcumulativesingularvalue

0.8

0.7

0.6

0.5

0.4

0.3

0 25 50 75 100 125 150 175 200 singular value index

(j) layer 12, head 1

ViT-T

1.0

iLLaMa-T

0.9

normalizedcumulativesingularvalue

0.8

0.7

0.6

0.5

0.4

0 25 50 75 100 125 150 175 200 singular value index

(k) layer 12, head 2

ViT-T

1.0

iLLaMa-T

0.9

normalizedcumulativesingularvalue

0.8

0.7

0.6

0.5

0.4

0.3

0 25 50 75 100 125 150 175 200 singular value index

(l) layer 12, head 3

Figure 11: Rank analysis of the self-attention matrix of all 3 heads in layer 1, 4, 8, 12 of the pretrained ViT-T and iLLaMA-T with N = 197. In most cases, especially in shallow layers, the singular values of iLLaMa show a more uniform distribution than ViT.

as reasoning and generation. This may be due to their massive training data and the next sentence prediction training paradigm, which is not explored by iLLaMA. Thus, a critical validation step of aligning the architectures of text and visual models would be to construct a multi-modal large language model that fully leverages LLaMA components. In this envisioned model, both visual and textual feature extractors would be realized through the LLaMA architecture. Futhermore, we strongly argue that iLLaMA’s successful attempts at basic supervised training strategies and classification tasks provide a foundation for more complex tasks, such as object detection and depth estimation. This represents a compelling avenue for future research.

### L Societal Impact

After the ChatGPT milestone in 2022, open-source architectures like LLaMA began to shine in the text domain. In the real world, images and text are the two main mediums of information and data types. For neural networks, having a unified architecture for language and vision models allows people to process these two distinct types of information using the same structure, which aids in the specialization of hardware implementation. This paper transfers the architecture widely used in language models to vision models, facilitating the achievement of this goal. The pretrained models and code provided in this paper can be used in a plug-and-play manner to serve this objective.

