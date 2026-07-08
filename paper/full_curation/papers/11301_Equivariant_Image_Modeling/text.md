# arXiv:2503.18948v1[cs.CV]24Mar2025

## Equivariant Image Modeling

Ruixiao Dong1,2, Mengde Xu2, Zigang Geng1,2, Li Li1, Han Hu2, Shuyang Gu2* 1University of Science and Technology of China 2Tencent Hunyuan Research

{dongruixiaoyx, zigang}@mail.ustc.edu.cn, lil1@ustc.edu.cn {jaredsheaxu, cientgu}@tencent.com

### Abstract

Current generative models, such as autoregressive and diffusion approaches, decompose high-dimensional data distribution learning into a series of simpler subtasks. However, inherent conflicts arise during the joint optimization of these subtasks, and existing solutions fail to resolve such conflicts without sacrificing efficiency or scalability. We propose a novel equivariant image modeling framework that inherently aligns optimization targets across subtasks by leveraging the translation invariance of natural visual signals. Our method introduces (1) column-wise tokenization which enhances translational symmetry along the horizontal axis, and (2) windowed causal attention which enforces consistent contextual relationships across positions. Evaluated on class-conditioned ImageNet generation at 256×256 resolution, our approach achieves performance comparable to state-of-the-art AR models while using fewer computational resources. Systematic analysis demonstrates that enhanced equivariance reduces intertask conflicts, significantly improving zero-shot generalization and enabling ultra-long image synthesis. This work establishes the first framework for task-aligned decomposition in generative modeling, offering insights into efficient parameter sharing and conflict-free optimization. The code and models are publicly available at https://github. com/drx-code/EquivariantModeling.

### 1. Introduction

Generative modeling has gained significant attention in computer vision, particularly for image generation tasks. Recent advances in autoregressive (AR) models [7, 18, 40] and diffusion processes [12, 30, 35] demonstrate remarkable capabilities in modeling complex data distributions through a shared strategy: decomposing the challenging problem of learning high-dimensional distributions into sequences of simpler conditional distribution estimations. AR

*Corresponding Author.

models achieve this through token-by-token prediction conditioned on previous outputs, while diffusion models employ iterative denoising across predefined noise levels. This multi-task learning paradigm raises critical questions about inter-task relationships - specifically, how decomposition strategies affect the synergies or conflicts between subtasks during joint optimization.

Recent investigations reveal inherent limitations in current decomposition frameworks. The MinSNR [10] identifies task conflicts in diffusion models and proposes Pareto optimization through careful loss weighting adjustments. An alternative approach, eDiff-I [1] attempts to mitigate conflicts via task-specific parameter groups; however, this leads to an explosion in the number of parameters and ignores the relevance of different tasks. These methods fundamentally fail to address the conflicts in multi-task optimization. These observations motivate us to consider the core research question: Can we establish a principled task decomposition framework that inherently aligns optimization target across subtasks?

In this paper, we present the first equivariant image modeling paradigm that systematically minimizes inter-task conflicts. This task decomposition method is inspired by the unbounded visual signals in nature. When people look around, they do not perceive that the visual signals mutate at a fixed position. Therefore, we propose a left-to-right modeling framework that includes:

Equivariant tokenization: We replace the conventional 2D patch grids with column-wise tokenization, enhancing spatial uniformity while better preserving natural image statistics (e.g., horizontal translation invariance in textures).

Equivariant modeling: A windowed causal attention mechanism that enforces consistent contextual relationships across positions.

We validate our framework through comprehensive experiments on class-conditioned image generation. When evaluated on ImageNet at 256×256 resolution, our approach achieves performance comparable to state-of-theart AR models while requiring substantially fewer computational resources. Through systematic analysis of the

model’s equivariance properties, we demonstrate that enhanced task alignment significantly improves parameter sharing efficiency across subtasks - particularly benefiting zero-shot generalization capabilities. This intrinsic equivariance proves especially advantageous for generating unbounded natural scenes, outperforming human-collected datasets that contain spatial inductive bias.

Our principal contributions can be summarized as follows:

- • The first equivariant image modeling paradigm that fundamentally aligns subtask optimization target.
- • A column-wise 1D tokenization scheme that eliminates the spatial constraints inherent in conventional 2D gridbased approaches, delivering competitive performance on class-conditioned generation with fewer computational cost than standard AR models.
- • An analytical framework for quantifying subtask conflicts, with empirical evidence showing that enhanced equivariance improves zero-shot generalization and enables ultra-long image generation.

### 2. Preliminaries

#### 2.1. Equivariance

Modern image generation paradigms, including autoregressive [7, 40] and diffusion models [12, 30], decompose the complex task of image distribution modeling into explicit and traceable subtasks. This decomposition inherently formulates image generation as a multi-task learning problem. Considering that these different tasks share identical parameters, maintaining equivariance among them becomes critical for learning efficiency [8]. Specifically, it is essential to ensure all subtasks exhibit congruent optimization trajectories under a shared parameter configuration. Formally, we define:

Definition 2.1 In a shared parameter space H, given a set of subtasks {Tt}Nt=1 with their performance measurements {Pt}Nt=1, the subtask group is termed equivariant if for any pair Ti,Tj ∈ T , their optimization directions coincide:

θ∗ = arg max

Pj(θ) (1)

Pi(θ) = arg max

θ∈H

θ∈H

where θ denotes the network parameters and H represents the optimization space.

For instance, in image generation, if predicting pixels at different spatial locations constitutes distinct subtasks, equivariance implies that the optimization direction for predicting any pixel should remain consistent, regardless of its position. This property enables facilitates transfer learning across all spatial locations and efficient parameter sharing.

#### 2.2. Autoregressive Image Modeling

Autoregressive models decompose images into token sequences {x1,...,xt} through the factorization:

n

p(xi|x<i,c), (2)

p(x1,...,xn|c) =

i=1

where c represents the external condition, and each conditional distribution mapping p(xi|x<i,c) corresponds to a subtask predicting the i-th token given the preceding tokens x<i and the condition c.

While enabling tractable likelihood estimation, conventional 2D grid-based autoregressive models face three critical challenges:

- • Subtask Heterogeneity: The fixed raster-scan ordering [7]* creates inherently distinct prediction tasks. For example, predicting border-region tokens typically proves more challenging than central-region tokens due to less local context.
- • Non-stationary Context Dependence: The standard autoregressive paradigm conditions each token generation on all preceding tokens, resulting in cumulatively increasing contextual dependencies. While wider context makes later positions easier to predict, it can cause the model to disproportionately optimize for these easier subtasks, potentially neglecting the more challenging predictions at earlier positions.
- • Architectural Non-Equivariance: Deep neural networks, particularly those employing self-attention mechanisms, can disrupt fundamental geometric symmetries. For example, position embeddings and attention patterns may introduce biases that violate translation invariance.

While recent work such as RAR [45] and TiTok [46] can partially address some challenges through stochastic ordering and 1D tokenization respectively, alleviating the inductive bias of AR image generation to a certain extent, there is still a lack of guarantee for the consistency of optimization directions across different tasks. This limitation potentially restricts their effectiveness in complex generation scenarios. Our work seeks to establish a subtask decomposition framework, where each subtask does not conflict and even promotes one another.

### 3. Towards Equivariant Autoregressive Image Generation

To address the challenges discussed in Sec. 2.2, we propose a column-based generation framework that explicitly removes the 2D grid structure and establishes a 1D equivariant modeling method, as illustrated in Fig. 1. Our approach includes two key components: a column-wise tokenizer that

*Though alternative orders like ”Z order” exist, they exhibit similar subtask decomposition properties.

naturally preserves vertical equivariance by organizing image features into column-based 1D tokens ( Sec. 3.1), and an autoregressive transformer equipped with equivariant context ( Sec. 3.2).

#### 3.1. Equivariant Tokenization via Columnization

Columnization and Rasterization. Visual signals in nature have no boundaries. When people look around, visual signals resemble a slowly unfolding scroll, without any sudden changes at fixed positions. Inspired by this, we consider tokenizing the image into a 1D latent sequence. Specifically, we employ columnization to transform the classical 2D tokenizer into a 1D sequence. This process involves reshaping the height dimension into channels, followed by a linear projection to compress the representation into column-wise tokens. Specifically, given an output feature map of shape H × W × C from a deep encoder, we transform it into a 1D token sequence F of shape W × C′ through:

W × (HC) −−−−−→Project W × C′.

H × W × C −−−−−−→Permute

Reshape

(3)

This columnization process eliminates the grid structure, resulting in each token representing a vertical visual signal, and adjacent tokens at any position transitioning naturally. Although the human-collected dataset may produce inductive biases due to camera settings and photographer preferences (e.g., the edge of the image may be darker due to the aperture), we leverage reflect padding to alleviate edge inconsistency. This semantically continuous transition between tokens provides the basic conditions for subsequent equivariant autoregressive modeling. We visualize the relationship between tokens and visual content in Fig. 2.

For image reconstruction, we rasterize the token sequence by projecting it to a higher-dimensional space and reshaping the channels back into the height dimension before processing it through a deep neural network:

W × C′ −−−−−→Project W × (HC) −−−−−−→Permute

###### H × W × C.

Reshape

(4) Semantic Aligned Tokenizer Training. Our columnization and rasterization operate on the deep features of an autoencoder inherited from [30]. Following their approach, we train the tokenizer using multiple loss components: pixel-wise reconstruction loss Lrec, adversarial loss Lgan, perceptual loss Lp, and KL divergence loss Lreg to regularize the token sequence distribution. Additionally, similar to [47], we introduce an alignment loss Lalign that aligns the decoder’s second-layer output features with those of a pretrained DINOv2 [25] model, helping to preserve the latent space’s semantic structure. The total loss combines the original autoencoder loss terms with semantic alignment loss:

Ltotal = λ1Lrec + λ2Lreg + λ3Lp + λ4Lgan + λ5Lalign (5)

where balancing coefficients λi are set to 1.0, 0.01, 1.0, 0.5, and 5.0 by default.

#### 3.2. Equivariant Autoregressive Modeling

Windowed Causal Attention. As shown in Eq. (2), autoregressive transformers employ causal attention to ensure that each token can attend to all its preceding tokens, making later subtasks significantly easier than earlier ones. A straightforward solution to address the potential imbalance in training is to limit each token’s context to a fixed window of k previous tokens:

p(x1,...,xn|c) =

n

p(xi|x≥i−k,<i,c) (6)

i=1

We named this approach as Real Equivariant Modeling. While this promotes equivariance across subtasks, it compromises the long-range dependencies that are often useful for image generation [48]. As an alternative, we implement windowed causal attention with a fixed context size w in each transformer layer. In this way, the network can increase the receptive field by stacking multiple layers, thereby implicitly modeling the long-range dependencies. For position i in each layer, the attention operation becomes:

qiKiT−w:i

)Vi−w:j (7)

Attn(qi,K,V ) = softmax(

√

d

where qi represents the query at position i, and Ki−w:i,Vi−w:i denote key-value pairs from the previous w positions. While this approach partially constrains the receptive field, empirical observations indicate that strategically leveraging the model’s equivariant properties enhances generation quality.

Our model architecture features a causal transformer with the windowed causal attention mechanism. In each self-attention layer, we utilize rotary position embedding [37] to encode token positions within the sequence. To inject conditioning information, we place a cross-attention layer between the self-attention layer and feed-forward network in each transformer block. Previous works [13, 48] find that the cross-attention requires absolute position embeddings to effectively convey global layout information. However, it may cause architectural non-equivariance. Therefore, we mitigate this issue through position embedding augmentation, randomly shifting the position index of the entire sequence with a randomly selected value. Following previous work [18], we implement a diffusion head above the transformer to predict the next token from random noise. To accelerate training, we employ the flow matching algorithm [19, 20], and the overall loss function is:

###### L = Ei∈[1,n],t||D(txi +(1−t)ϵi,t,zi)−(xi −ϵi)||2, (8)

Pred Tokens

[Figure 1]

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

Encoding

Modeling

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

Transformer Layer

Windowed Causal Attention

| | |
|---|---|
| | |

Columniation

Input Image

…

𝐊

| |
|---|

| |
|---|

| |
|---|

𝐐

1D Tokenizer Token Sequence

| |
|---|

| |
|---|

| |
|---|

Cross Attention

Transformer Layer

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

…

| |
|---|

| |
|---|

Rasterization

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

[Figure 2]

| |
|---|

Transformer Layer

| |
|---|

| |
|---|

| |
|---|

Decoding

| |
|---|

| |
|---|

| |
|---|

Start Token

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

| |
|---|

Reconstructed Image

Condition Input Tokens

- Figure 1. Illustration of Equivariant Image Generation Framework. The tokenizer translates the image into 1D tokens arranged in columns and an enhanced autoregressive model models the column-wise token distribution.

25% tokens 50% tokens 75% tokens 100% tokens

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

- Figure 2. Visual Meanings of 1D Tokens. By progressively replacing the randomly initialized token sequence with tokens encoded from the ground truth images, the decoder faithfully reconstructs the original images step by step.

lows [7, 30], employing a 2D encoder with a downsampling factor of 16, followed by a columniation operation to produce token tensors F′ ∈ Rw×c

′

where w = 16,c′ = 256. Training utilizes the Adam optimizer [15] for 320k iterations with a batch size of 192. The learning rate starts at 1.92 × 10−4, with a linear warm-up over 5,000 iterations and a 20% decay every 30,000 iterations. The semantic loss is introduced after 20,000 iterations.

We train the generator for 1,200 epochs using AdamW [21] with a batch size of 2,048. The initial learning rate is 8 × 10−4, which is linearly warmed up over 100 epochs and then maintained at a constant level. The weight decay is set to 0.02, along with the momentum parameters (β1,β2) = (0.9,0.95). An exponential moving average (EMA) of the parameters in the generator is maintained with a momentum of 0.9999.

where t denotes the noise level, which is sampled from U(0,1). Accordingly, xi and zi denote the ground truth token and the output of the causal transformer at position i. ϵi is randomly sampled Gaussian noise, and D denotes the denoising network, which predicts the vector field at noise level t.

Evaluation Metrics. We evaluate reconstruction fidelity using the dataset-level reconstruction Fr´echet Inception Distance (rFID)[11]. For generation quality, we measure both the generative Fr´echet Inception Distance (gFID)[11] and the Inception Score (IS) [31].

### 4. Experiments 4.1. Experimental Settings

We conduct experiments on the ImageNet-1k dataset [6]. All images are resized to 256×256 resolution, utilizing standard augmentation techniques, including random cropping and horizontal flipping. We follow the common train/validation split to ensure consistent evaluation with prior approaches.

Implementation Details. Our tokenizer architecture fol-

#### 4.2. Deep Analysis on Equivariance

Following the definition in Sec. 2, we examine the equivariance of our method by analyzing transferability across subtasks. We set up a MAR-AR variant model with 2D gridbased token sequences of length 16 as our baseline, denoted as AR-MAR-2D.

Zero-shot Transfer between Subtasks. We investigate model generalization across different generative subtasks. With our tokenizer producing sequences of length 16, we have 16 distinct generative subtasks (detailed in Sec. 2.2).

[Figure 11]

[Figure 12]

- Figure 3. Training Loss of Different Models. Left: the training loss of different methods at early (10 epoches) and late (100 epoches) training stage. Right: the relative loss improvement of different methods under different settings compared to the early stage of Multi-task setting. The higher value indicates better performance. The equivariant generation approach can transfer the improvement from a single task to other untrained tasks.

more uniform behavior across subtasks, confirming better inter-task consistency. Notably, both methods show elevated loss for middle subtasks (corresponding to image centers), probably due to ImageNet’s object-centric nature. We validates the data bias in Fig. 4 by comparing task-wise

Method # Num task. gFID AR-MAR-2D 16 7.93

Ours 16 5.57 AR-MAR-2D 8 92.46 Ours 8 8.99

1.02

0.92

| |ImageNet<br><br>LHQ| | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

Table 1. Performance under Zero-shot Setting. # Num task. is used to denote the number of trained subtasks. The total number of subtasks is 16 for all methods.

1.01

0.91

TrainingLoss

1.00

0.90

0.99

0.89

We train the model on 8 selected subtasks and evaluate performance across all 16 subtasks using gFID. As shown in Tab. 1, while both approaches perform well when trained on all subtasks, our method maintains competitive performance under the zero-shot setting, with only a 3.42 point drop in gFID. In contrast, the 2D baseline struggles significantly with untrained subtasks, with gFID deteriorating from 7.93 to 92.46. This suggests our framework creates more consistent generative patterns across subtasks, enabling better generalization.

0.98

0.88

4 6 8 10 12 Task ID

Figure 4. Converged Training Loss on ImageNet vs LHQ. Compared to ImageNet, the visual statics in LHQ demonstrates greater uniformity, as does the task-wise loss distribution.

loss across different datasets. When testing on LHQ [34], a landscape dataset with more uniform spatial distribution than ImageNet, the elevated middle loss phenomenon disappears. This motivates us to consider the dataset’s influence in future research studying the equivariance problem.

Training Dynamics Analysis. We further examine equivariance with training dynamics using two settings: 1) Multi-task training - the standard generative modeling setting, where a shared-parameter model is trained on all subtasks; 2) Single-task training - focusing on individual subtasks (4th and 11th) to isolate task interference effects.

Moreover, to isolate the influence of the data, we use the multi-task loss at the early training stage as our baseline and measure relative improvement across methods (with a lower loss indicating positive improvement). The results appear in the right panel of Fig. 3. In the multi-task setting, both our method and AR-MAR-2D converge to a 10% relative improvement. However, in single-task training, our method successfully transfers the performance gain to all other subtasks, while the 2D baseline demonstrates negative impacts on untrained subtasks. These dynamics align with overall

Fig. 3 (left) shows the evolution of the multi-task training loss. While both methods converge to lower loss with longer training iterations, they exhibit different patterns across subtasks. The baseline consistently shows higher loss for subtasks corresponding to the image’s boundary (5th, 9th, and 13th subtasks, which are the first tokens of each row in the 2D grid), while our method demonstrates

generation performance and suggest a promising approach for rapid equivariance verification.

#### 4.3. Equivariance Application

Long Content Image Generation. Our experiments demonstrate that, due to the equivariance of our method, the approach exhibits strong generalization ability across different subtasks. Inspired by the unbounded visual signals in nature, combining the generalization ability to generate training-unseen subtasks granted by the equivariant property, we test our models under long image generation scenarios. Specifically, we train our model on the Nature subset of the Places dataset [52] for class-conditional generation, which contains 30 categories. The training image size is 256 × 256, and we keep all other training parameters the same as in the ImageNet training phase.

Fig. 5 showcases some generated examples of extendedlength arbitrary resolution images produced by our model. These generated images exhibit a high spatial resolution, with lengths significantly greater than 256. The presented results demonstrably illustrate the zero-shot long-content capability of our method, primarily attributed to its inherent equivariance property: although we only leverage 256×256 resolution images to optimize the model, our approach effectively generates content at positional indices not encountered during training. Specifically, our method achieves the generation of images up to eight times longer than the input instances used during training, maintaining high visual fidelity and crucially avoiding discernible sharp edges between adjacent generated regions.

Interactive Generation From Human Feedback. Another advantageous aspect of our method is that each token corresponds to a visually trackable area. This allows for interactive, token-wise feedback from users. We illustrate some examples in Fig. 6. When a visually poor token is produced, we can promptly discard it and generate a new one. Ideally, our method could be integrated into an image editing pipeline, enabling complete control over the generation process. We leave this for future work.

#### 4.4. System Comparison with SOTA Methods

As detailed in Tab. 2, a comparative analysis was conducted between our method and state-of-the-art generative methodologies, including diffusion models and autoregressive methods along with their variants. Our approach demonstrates comparable or better performance than the other methods presented in the table. Significantly, our model outperforms the autoregressive variant of MAR methods, confirming that the introduction of equivariant properties indeed enhances the modeling capability. Furthermore, the reduced token length inherent in our method leads to substantial computational savings in both training and inference, as evidenced by the reduction of GFLOPs in

Method gFID↓ IS↑ #Para. #Len. GFLOPs↓ Diffusion DiT [26] 2.27 278.2 675M - 118.64 MaskGIT

TiTok [46] 1.97 281.8 287M 128 37.35 MAR [18] 1.78 296.0 479M 64 70.13

FractalMAR [17] 7.30 334.9 438M - 238.58

Autoregressive VQGAN [7] 15.78 74.3 1.4B 256 246.67 VAR [40] 3.30 274.4 310M 680 105.70 MAR [18] 4.69 244.6 479M 64 78.50

Ours-S 7.21 233.70 151M 16 5.41 Ours-B 5.57 260.05 294M 16 9.78 Ours-L 4.48 259.91 644M 16 19.66 Ours-H 4.17 290.66 1.2B 16 34.91

- Table 2. Class-conditional Generation Results on ImageNet 256×256 Benchmark. #Para. denotes the number of parameters in each generator, while #Len. indicates the token sequence length that generators are required to model.

Method Tokens Length gFID↓ GFLOPs↓ AR-MAR-2D-B 256 3.99 130.46 AR-MAR-2D-B 16 7.93 9.79 AR-MAR-2D-L 16 7.49 19.68

Ours-B 16 5.57 9.78 Ours-L 16 4.48 19.66

- Table 3. Comparison on the performance and computational efficiency. Our methods achieves better trade-off between performance and computation cost.

the table, particularly when compared to methods employing significantly longer token lengths (≥ 64).

We also compare our method with standard 2D variants in Tab. 3. With similar GFLOPs, our method attains superior generation performance. Moreover, our model achieves comparable gFID to the 256-token baseline while requiring only 15% of the GFLOPs. The results demonstrate the effectiveness of our equivariance-driven design in achieving strong outcomes with minimal computational overhead.

- 4.5. Ablation Studies

Windowed Causal Attention. To address the inconsistency in causal transformer models arising from subtasks adapting to varying context lengths, we introduce windowbased attention. Specifically, each token is restricted to attending to a fixed-size local window of ω tokens during attention computation. To evaluate the impact of this tech-

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

- Figure 5. Visual examples of long image generation. We present visual examples of long images with arbitrary lengths, which are generated by our model that has been trained on the Places datasets with fixed length of 256.

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

| | |
|---|---|

[Figure 24]

| | |
|---|---|

[Figure 25]

| |
|---|

| |
|---|

[Figure 26]

[Figure 27]

Generate

Generate

…

…

Drop Re-generate

Drop Re-generate

- Figure 6. Interactive Image Generation. During inference, each token is immediately visible and bad generated tokens (circled with orange rectangle) are dropped according to human feedback.

[Figure 28]

[Figure 29]

- (a) w/o random shift

[Figure 30]

[Figure 31]

- (b) with random shift

Figure 7. Visualization effect of the Augmented Position Embedding. The model is trained on a subset of the Places dataset.

nique, we perform comparative experiments on our small model with a window size of ω = 3 and compare it against standard causal attention, where each token attends to all preceding tokens. Additionally, we train a real equivariant variant by restricting the context window length during training; in this scenario, the model only observes randomly cropped sequences of length ω. Under these training conditions, the subsequences gain an explicit equivariant property: apart from the initial ω − 1 tokens, generated subsequences are invariant to shifted positions.

As summarized in Tab. 4, limiting the receptive field slightly weakens the modeling of long-range dependencies but enhances consistency and uniformity across tasks, thereby improving the overall training of the generator. In particular, our generator with window causal attention surpasses the baseline model. Moreover, due to the fixed context size in the transformer layer, it reduces the theoretical computational cost of attention by approximately 42.9%

Method Equ. L-Ctx. gFID↓ Attn FLOPs↓ Full Causal ✓ 7.35 4.2M

Real Equ. ✓ 8.87 0.26M Window Causal ✓ ✓ 7.21 1.8M

Table 4. Ablation Study of Windowed Causal Attention. ”Real Equ.” indicates the strictly equivariant model variant as described in Sec. 3.2; ”Equ.” refers to whether the model possesses equivariant properties; and ”L-Ctx.” denotes whether long-range contexts are utilized.

compared to causal attention. Conversely, while the real equivariant generator maintains highly equivariant properties, it exhibits significantly reduced performance due to overly limited receptive fields, demonstrating that preserving long-range context information remains crucial for generative modeling.

##### Position Embedding Augmentation. We evaluate our pro-

Method rFID↓ gFID ↓

- 1 baseline 1.11 7.10
- 2 +Stronger discriminator 0.62 6.29
- 3 + Decoder finetune 0.58 6.25 Ours +Semantic aligned loss 0.56 5.57

Table 5. Ablation on the Impact of Tokenizer Components. ”Ours” denotes the final setting we adopted in all other experiments.

posed position embedding augmentation by training our large model on the Places dataset and comparing visualization results with and without this augmentation, which keeps the left side of the images as the original images, as illustrated in Fig. 7. We observe that the model trained without augmented position embeddings can still extrapolate beyond the training lengths to generate longer images; however, noticeable artifacts and discontinuities begin to appear in regions beyond the model’s training positions. In contrast, when augmented position embeddings are utilized, the generated images exhibit stronger consistency across spatial locations, indicating that the model effectively generalizes beyond the observed spatial contexts. Specifically, as shown in the visual comparisons ( Fig. 7), our augmented model synthesizes coherent, artifact-free, and smooth images even at significantly increased lengths, demonstrating improved capability for extrapolation and robust generalization.

Components in Tokenizer. We analyze incremental improvements to our tokenizer within our base model, identified as (1) Baseline Tokenizer. The base tokenizer following LDM-tokenizer [30] achieves a gFID of 7.10 and an rFID of 1.11. (2) Enhanced Discriminator. Replacing the default discriminator with a DINO-small backbone [49] improves rFID by 0.49 and gFID by 0.81, emphasizing stronger feature discrimination. (3) Decoder Fine-tuning. Fine-tuning the decoder enhances reconstruction fidelity and marginally improves generation quality. (4) Alignment Loss Introduction. Recognizing that the semantic information contained in the latent space is insufficient for high-quality generation, we introduce an alignment loss Lalign to align the latent representations with a pretrained DINOv2 model [25], resulting in a significant improvement in the gFID metric to

- 5.57.

### 5. Related Work

Image distribution modeling is a long-standing topic in computer vision. One practical approach to solving this is visual signal decomposition, which breaks down the modeling task into subtasks.

Autoregressive models decompose each subtask as nexttoken generation. Early works like PixelCNN [33, 41] decomposed image generation into pixel-by-pixel prediction

in a raster-scan order. Later approaches [7, 42] introduced a two-stage process: first, encoding images into a latent space utilizing a tokenizer, and then employing autoregressive modeling to decompose latent modeling into sequential subtasks. While subsequent research [4, 9, 16, 22, 23, 27, 28, 38, 43, 44, 50, 51, 53, 54] has focused on enhancing tokenizer capacity, the fundamental decomposition logic remains unchanged and continues to face challenges associated with 2D grid structures. Various works have been proposed to eliminate prediction on a 2D grid. VAR [40] introduced scale-wise decomposition, predicting from small to large scales. However, this approach still exhibits significant non-equivariance, with earlier tokens primarily encoding low-frequency information and later tokens capturing high-frequency details. Concurrent work [29] proposes predicting block-wise or row-wise tokens in parallel, which is technically similar to our approach.

Diffusion models [12, 35] take a different approach by decomposing visual signals into progressively denoised image sequences through shared-parameter models. Standard noise schedules, however, can create distributional inconsistencies across denoising stages. Several techniques have been proposed to address this challenge: the reparameterization trick [24, 32] has proven vital in unifying output distributions across various denoising tasks; Flow Matching [19, 20] and Consistency Models [36] aim to develop improved noise strategies that maintain consistency throughout the denoising process; Min-SNR [39] employs Pareto optimization by carefully adjusting loss weights to identify conflicts among tasks; and Variational Diffusion Models [14] and the Diffusion Schr¨odinger Bridge [5, 39] explore learning the noise-adding strategy rather than relying on fixed heuristics. Despite these advances, real-world application of these methods remains constrained by unresolved distributional mismatches.

MaskGIT [3] decomposes the visual signals into depthwise generations that sequentially predict sets of tokens. TikTok [46] further simplifies the decomposition using a 1D tokenizer. Muse [2] alleviates subtask conflicts in depthwise generation by dividing the subtasks into two groups and modeling them with distinct parameters.

Although numerous techniques have been proposed to properly decompose visual signals, our work is the first to address this problem from an equivariance perspective, providing a more systematic analytical framework.

### 6. Conclusion

This work establishes Equivariant Image Modeling as a principled framework for mitigating subtask conflicts in generative models. By introducing column-wise tokenization and equivariant token modeling, we demonstrate that the spatial equivariant decomposition aligns optimization targets, enhancing parameter efficiency and zero-shot gen-

eralization. Experimental results on ImageNet-1k generation validate the computational advantages of our approach over conventional autoregressive models. The exploration of equivariant task decomposition opens new directions for the future development of generative models.

### References

- [1] Yogesh Balaji, Seungjun Nah, Xun Huang, Arash Vahdat, Jiaming Song, Qinsheng Zhang, Karsten Kreis, Miika Aittala, Timo Aila, Samuli Laine, et al. ediff-i: Text-to-image diffusion models with an ensemble of expert denoisers. arXiv preprint arXiv:2211.01324, 2022. 1
- [2] Huiwen Chang, Han Zhang, Jarred Barber, Aaron Maschinot, Jose Lezama, Lu Jiang, Ming-Hsuan Yang, Kevin Patrick Murphy, William T Freeman, Michael Rubinstein, et al. Muse: Text-to-image generation via masked generative transformers. In ICML, pages 4055–4075. PMLR,

2023. 8, 12

- [3] Huiwen Chang, Han Zhang, Lu Jiang, Ce Liu, and William T Freeman. Maskgit: Masked generative image transformer. In CVPR, pages 11315–11325, 2022. 8
- [4] Junyu Chen, Han Cai, Junsong Chen, Enze Xie, Shang Yang, Haotian Tang, Muyang Li, and Song Han. Deep compression autoencoder for efficient high-resolution diffusion models. In ICLR, 2025. 8
- [5] Valentin De Bortoli, James Thornton, Jeremy Heng, and Arnaud Doucet. Diffusion schr¨odinger bridge with applications to score-based generative modeling. NeurIPS, 34:17695– 17709, 2021. 8
- [6] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In CVPR, pages 248–255, 2009. 4, 11
- [7] Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image synthesis. In CVPR,

2021. 1, 2, 4, 6, 8

- [8] Shuyang Gu. Several questions of visual generation in 2024. arXiv preprint arXiv:2407.18290, 2024. 2
- [9] Yuchao Gu, Xintao Wang, Yixiao Ge, Ying Shan, and Mike Zheng Shou. Rethinking the objectives of vectorquantized tokenizers for image synthesis. In CVPR, pages 7631–7640, 2024. 8
- [10] Tiankai Hang, Shuyang Gu, Chen Li, Jianmin Bao, Dong Chen, Han Hu, Xin Geng, and Baining Guo. Efficient diffusion training via min-snr weighting strategy. In ICCV, pages 7441–7451, 2023. 1
- [11] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. NeurIPS, 30, 2017. 4
- [12] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. NeurIPS, 33:6840–6851, 2020. 1, 2, 8
- [13] Runtong Hou and Xu Zhao. High-quality talking face generation via cross-attention transformer. In 2024 IEEE International Conference on Real-time Computing and Robotics (RCAR), pages 194–199, 2024. 3
- [14] Diederik Kingma, Tim Salimans, Ben Poole, and Jonathan Ho. Variational diffusion models. NeurIPS, 34:21696–

- 21707, 2021. 8
- [15] Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In ICLR, 2015. 4
- [16] Doyup Lee, Chiheon Kim, Saehoon Kim, Minsu Cho, and Wook-Shin Han. Autoregressive image generation using residual quantization. In CVPR, pages 11523–11532, 2022. 8
- [17] Tianhong Li, Qinyi Sun, Lijie Fan, and Kaiming He. Fractal generative models. arXiv preprint arXiv:2502.17437, 2025. 6
- [18] Tianhong Li, Yonglong Tian, He Li, Mingyang Deng, and Kaiming He. Autoregressive image generation without vector quantization. NeurIPS, 37:56424–56445, 2024. 1, 3, 6
- [19] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow matching for generative modeling. In ICLR, 2023. 3, 8
- [20] Xingchao Liu, Chengyue Gong, and qiang liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. In ICLR, 2023. 3, 8, 12
- [21] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In ICLR, 2019. 4
- [22] Zhuoyan Luo, Fengyuan Shi, Yixiao Ge, Yujiu Yang, Limin Wang, and Ying Shan. Open-magvit2: An open-source project toward democratizing auto-regressive visual generation. arXiv preprint arXiv:2409.04410, 2024. 8
- [23] Fabian Mentzer, David Minnen, Eirikur Agustsson, and Michael Tschannen. Finite scalar quantization: VQ-VAE made simple. In ICLR, 2024. 8
- [24] Alexander Quinn Nichol and Prafulla Dhariwal. Improved denoising diffusion probabilistic models. In ICML, pages 8162–8171. PMLR, 2021. 8
- [25] Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy V. Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel HAZIZA, Francisco Massa, Alaaeldin El-Nouby, Mido Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Herve Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. DINOv2: Learning robust visual features without supervision. Transactions on Machine Learning Research, 2024. 3, 8
- [26] William Peebles and Saining Xie. Scalable diffusion models with transformers. In ICCV, pages 4195–4205, 2023. 6
- [27] Liao Qu, Huichao Zhang, Yiheng Liu, Xu Wang, Yi Jiang, Yiming Gao, Hu Ye, Daniel K Du, Zehuan Yuan, and Xinglong Wu. Tokenflow: Unified image tokenizer for multimodal understanding and generation. arXiv preprint arXiv:2412.03069, 2024. 8
- [28] Ali Razavi, Aaron Van den Oord, and Oriol Vinyals. Generating diverse high-fidelity images with vq-vae-2. NeurIPS, 32, 2019. 8
- [29] Shuhuai Ren, Shuming Ma, Xu Sun, and Furu Wei. Next block prediction: Video generation via semi-auto-regressive modeling. arXiv preprint arXiv:2502.07737, 2025. 8
- [30] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, 2022. 1, 2, 3, 4, 8, 11
- [31] Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki

- Cheung, Alec Radford, and Xi Chen. Improved techniques for training gans. NeurIPS, 29, 2016. 4
- [32] Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. In ICLR, 2022. 8
- [33] Tim Salimans, Andrej Karpathy, Xi Chen, and Diederik P Kingma. Pixelcnn++: Improving the pixelcnn with discretized logistic mixture likelihood and other modifications. arXiv preprint arXiv:1701.05517, 2017. 8
- [34] Ivan Skorokhodov, Grigorii Sotnikov, and Mohamed Elhoseiny. Aligning latent and image spaces to connect the unconnectable. In ICCV, pages 14144–14153, 2021. 5
- [35] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In ICML, pages 2256–

2265. PMLR, 2015. 1, 8

- [36] Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models. In ICML, pages 32211–

32252. PMLR, 2023. 8

- [37] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063,

2024. 3

- [38] Yuhta Takida, Yukara Ikemiya, Takashi Shibuya, Kazuki Shimada, Woosung Choi, Chieh-Hsin Lai, Naoki Murata, Toshimitsu Uesaka, Kengo Uchida, Wei-Hsiang Liao, and Yuki Mitsufuji. HQ-VAE: Hierarchical discrete representation learning with variational bayes. Transactions on Machine Learning Research, 2024. 8
- [39] Zhicong Tang, Tiankai Hang, Shuyang Gu, Dong Chen, and Baining Guo. Simplified diffusion schr\” odinger bridge. arXiv preprint arXiv:2403.14623, 2024. 8
- [40] Keyu Tian, Yi Jiang, Zehuan Yuan, Bingyue Peng, and Liwei Wang. Visual autoregressive modeling: Scalable image generation via next-scale prediction. NeurIPS, 37:84839–84865,

2025. 1, 2, 6, 8

- [41] Aaron Van den Oord, Nal Kalchbrenner, Lasse Espeholt, Oriol Vinyals, Alex Graves, et al. Conditional image generation with pixelcnn decoders. NeurIPS, 29, 2016. 8
- [42] Aaron Van Den Oord, Oriol Vinyals, et al. Neural discrete representation learning. NIPS, 30, 2017. 8
- [43] Mark Weber, Lijun Yu, Qihang Yu, Xueqing Deng, Xiaohui Shen, Daniel Cremers, and Liang-Chieh Chen. Maskbit: Embedding-free image generation via bit tokens. Transactions on Machine Learning Research, 2024. 8
- [44] Jiahui Yu, Xin Li, Jing Yu Koh, Han Zhang, Ruoming Pang, James Qin, Alexander Ku, Yuanzhong Xu, Jason Baldridge, and Yonghui Wu. Vector-quantized image modeling with improved VQGAN. In ICLR, 2022. 8
- [45] Qihang Yu, Ju He, Xueqing Deng, Xiaohui Shen, and LiangChieh Chen. Randomized autoregressive visual generation. arXiv preprint arxiv: 2411.00776, 2024. 2
- [46] Qihang Yu, Mark Weber, Xueqing Deng, Xiaohui Shen, Daniel Cremers, and Liang-Chieh Chen. An image is worth 32 tokens for reconstruction and generation. NeurIPS, 37:128940–128966, 2024. 2, 6, 8
- [47] Sihyun Yu, Sangkyung Kwak, Huiwon Jang, Jongheon Jeong, Jonathan Huang, Jinwoo Shin, and Saining Xie. Representation alignment for generation: Training diffusion transformers is easier than you think. In ICLR, 2025. 3

- [48] Bowen Zhang, Shuyang Gu, Bo Zhang, Jianmin Bao, Dong Chen, Fang Wen, Yong Wang, and Baining Guo. Styleswin: Transformer-based gan for high-resolution image generation. In CVPR, pages 11304–11314, 2022. 3
- [49] Hao Zhang, Feng Li, Shilong Liu, Lei Zhang, Hang Su, Jun Zhu, Lionel Ni, and Heung-Yeung Shum. DINO: DETR with improved denoising anchor boxes for end-to-end object detection. In ICLR, 2023. 8
- [50] Long Zhao, Sanghyun Woo, Ziyu Wan, Yandong Li, Han Zhang, Boqing Gong, Hartwig Adam, Xuhui Jia, and Ting Liu. ϵ-vae: Denoising as visual decoding. arXiv preprint arXiv:2410.04081, 2024. 8
- [51] Chuanxia Zheng, Tung-Long Vuong, Jianfei Cai, and Dinh Phung. Movq: Modulating quantized vectors for highfidelity image generation. NeurIPS, 35:23412–23425, 2022. 8
- [52] Bolei Zhou, Agata Lapedriza, Aditya Khosla, Aude Oliva, and Antonio Torralba. Places: A 10 million image database for scene recognition. IEEE TPAMI, 2017. 6, 11
- [53] Lei Zhu, Fangyun Wei, Yanye Lu, and Dong Chen. Scaling the codebook size of VQ-GAN to 100,000 with a utilization rate of 99%. In NeurIPS, 2024. 8
- [54] Yongxin Zhu, Bocheng Li, Yifei Xin, and Linli Xu. Addressing representation collapse in vector quantized models with one linear layer. arXiv preprint arXiv:2411.02038, 2024. 8

### A. Datasets

config value Base channels 128 Base channel multiplier per stage [1, 1, 2, 4, 4] Residual blocks per stage 2 Attention resolutions 16 Token channels 256 Adversarial loss enabled at iteration 5000 Discriminator loss weight 0.5 Discriminator loss hinge loss Perceptual loss weight 1.0 Semantic anlignment loss enabled at iteration 20000 Semantic anlignment loss weight 5.0 KL divergence loss weight 0.01 Gradient clipping by norm 1.0 Optimizer Adam

ImageNet-1k ImageNet [6] is a large-scale hierarchical image database that has served as a cornerstone dataset for modern computer vision research since its introduction in 2009. It contains more than one million annotated images across 1,000 object categories, providing a robust benchmark for numerous vision tasks, including image classification, object detection, and class-conditional image generation.

Dataset website: https://image-net.org/

Places The Places [52] dataset is curated according to the principles of human visual cognition, with the aim of creating a comprehensive resource to train artificial systems in high-level visual understanding tasks. Applications range from scene recognition and object detection in contextual environments, to sophisticated understanding tasks such as action recognition, event prediction, and theory-of-mind inference. The entire Places database includes more than 10 million images, covering over 400 unique scene categories. In particular, for our long-image generation experiments, we selected 30 nature categories from the Places-Challenge subset, which contains approximately 1 million images.

- Beta1 0.5
- Beta2 0.9 Base LR 1.92e-4 LR warmup iterations 5000 LR decay frequency 30000 LR decay ratio 0.2 EMA decay 0.9999

Dataset website: http://places2.csail.mit. edu/

Training epochs 50 Total Batchsize 192 GPU A100

### B. Implementation Details

#### B.1. Data Augmentation

Table 6. Detailed hyper-parameters for our equivariant 1D tokenizer.

We perform data augmentation by initially resizing the input images so that the smaller dimension is 256 pixels. Following this, random cropping is applied to the resized images. Additionally, horizontal flipping is performed with a probability of 0.5 to improve the robustness and generalization of the model.

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

#### B.2. Tokenizer Training

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

We follow the standard training recipe proposed in the Latent Diffusion Model (LDM) [30], and detailed hyperparameter configurations used for training our equivariant 1D tokenizer are provided in Tab. 6.

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

#### B.3. Generative Models Training

Detailed hyper-parameters utilized in our equivariant generator are summarized in Tab. 8. To comprehensively evaluate model capacity and performance trade-offs, we trained multiple variants of the generative model with different sizes and complexities. The complete architectural details for each of these model variants are provided in Tab. 7.

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

25% tokens 50% tokens 75% tokens 100% tokens

We utilize 32 A100 GPUs for training the tokenizer, with the training process spanning 3 days. The generator models are trained using 64 A100 GPUs, requiring 4.6 days for the longest schedule (training our huge model for 1200 epochs).

Figure 8. Additional Examples about Visual Meanings of 1D Tokens.

Model #Para. Layers Hidden dim Attn heads Diff. hidden dim Diff.layers

Small 151M 16 512 8 960 12 Base 294M 24 768 12 1024 12 Large 644M 32 1024 16 1280 12 Huge 1.2B 40 1280 16 1536 12

- Table 7. The model configurations of our generators. #Para. denotes the number of parameters in the respective generators and Diff. presents the diffusion head. We also use ”S”, ”B”, and ”L” and ”H” as shorthand for different models in the manuscript.

config value Token length 16 Token channels 256 MLP ratio 4 Norn layer in attention blocks nn.LayerNorm Class labels sequence length 16 Class labels dropout 0.1 Attention dropout 0.1 Projection layer dropout 0.1

Gradient clipping by norm 3.0 Optimizer Adam

- Beta1 0.9
- Beta2 0.95 Base LR 8.0e-4 LR scheduler constant LR warmup epochs 100 Weight decay 0.02 EMA decay 0.9999

Training epochs 1200 Total Batchsize 2048 GPU A100

- Table 8. Detailed hyper-parameters for our equivariant generator.

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

25% tokens 50% tokens 75% tokens 100% tokens

Figure 9. Visualization of the generation process.

select the optimal guidance scales individually for each trained model.

### C. Pseudo-Code for Our Equivariant 1D Tokenizer

#### B.4. Sampling Hyper-parameters

We generate results by sampling with 100 denoising steps, utilizing the first-order Euler solver following the Rectified Flow framework [20]. At inference time, the genertor employs classifier-free guidance (CFG). Specifically, the underlying transformer network produces two distinct outputs: the conditional output hc (conditioning context present) and the unconditional output hu (conditioning context absent). The predicted velocity v is obtained through interpolation of these two outputs as follows: v = vθ(xt,|t,hu) + ω · (vθ(xt,|t,hc) − vθ(xt,|t,hu)), where ω represents the guidance scale parameter. Inspired by Muse [2], we employ a dynamic CFG schedule in which the guidance scale ω increases linearly as the sampling sequence progresses. To maximize sampling quality, we systematically tune and

We have included PyTorch-style pseudo-code for our Equivariant 1D Tokenizer.

### D. Visualization of Our Huge Model on ImageNet

We showcase the uncurated 256×256 images generated by our huge model in Fig. 10.

### E. Visualization of Long Images

We provide uncurated long-content images produced by our large model in Fig. 11, which is trained exclusively on the Nature subset of the Places dataset. Owing to the equivariant property, our model effectively captures fine-grained spatial coherence, enabling it to generate high-fidelity landscape images. Remarkably, our approach demonstrates

Algorithm 1: Our Equivariant 1D Tokenizer PyTorch-style Pseudo-Code

class EquivariantTokenizer(nn.Module)

def init (token channels): # 2D Encoder and Decoder self.2DEncoder, self.2DDecoder = 2DEncoder(), 2DDecoder() # 1D Encoder and Decoder self.1DEncoder, self.1DDecoder = 1DEncoder(), 1DDecoder() self.token channels = token channels

def forward(self, x): z = self.2DEncoder(x) # Columnization z = z.permute(0,3,1,2) z = z.reshape(z.shape[0], z.shape[1], -1) # 1D Latent posterior = self.1DEncoder(x) latent = posterior.sample() z = self.1DDecoder(latent) # Rasterization z = z.reshape(z.shape[0], z.shape[1], self.token channels, -1) z = z.permute(0, 2, 3, 1) return self.2DDecoder(z)

this zero-shot capability, as the model was never explicitly trained on high-resolution images.

### F. Visualization of Tokens

To clearly illustrate the relationship between encoded tokens and corresponding visual content, we progressively replace the randomly initialized token sequences with encoded tokens. As demonstrated in Fig. 2 and further substantiated in Fig. 8, the decoder faithfully reconstructs the original images step by step. Furthermore, we visualize the generation process by decoding progressively generated token sequences in Fig. 9, thereby providing further clarification regarding the semantic interpretation of tokens in the latent space from a generative perspective.

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

###### Figure 10. Generation Results on the ImageNet-1k Dataset.

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

[Figure 134]

###### Figure 11. More Visual Examples of Generated Long Images.

