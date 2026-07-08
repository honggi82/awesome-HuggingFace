## “Principal Components” Enable A New Language of Images

Xin Wen1∗ Bingchen Zhao2∗† Ismail Elezi3 Jiankang Deng4 Xiaojuan Qi1‡

∗Equal Contribution †Project Lead ‡Corresponding Author

1University of Hong Kong 2University of Edinburgh 3Huawei London Research Centre 4Imperial College London

https://visual-gen.github.io/semanticist

# arXiv:2503.08685v2[cs.CV]28Jul2025

[Figure 1]

Figure 1. Image reconstruction using our structured visual tokenization approach, which uniquely enables decoding at any token count. Each column shows reconstructions resulting from progressively increasing the number of tokens, from a single token to 256 tokens. Unlike conventional tokenizers that require a fixed number of tokens for meaningful decoding, our method ensures that each token incrementally refines the image, with earlier tokens capturing the most salient features and later ones adding finer details. This demonstrates the flexibility and effectiveness of our approach in producing coherent images even with very few tokens (view more from Fig. 17 in the Appendix).

### Abstract

We introduce a novel visual tokenization framework that embeds a provable PCA-like structure into the latent token space. While existing visual tokenizers primarily optimize for reconstruction fidelity, they often neglect the structural properties of the latent space—a critical factor for both interpretability and downstream tasks. Our method generates a 1D causal token sequence for images, where each successive token contributes non-overlapping information with mathematically guaranteed decreasing explained variance, analogous to principal component analysis. This structural constraint ensures the tokenizer extracts the most salient visual features first, with each subsequent token adding diminishing yet complementary information. Additionally, we identified and resolved a semantic-spectrum coupling effect that causes the unwanted entanglement of high-level semantic content and low-level spectral details in the tokens by leveraging a diffusion decoder. Experiments demonstrate that our approach achieves state-of-the-art reconstruction

performance and enables better interpretability to align with the human vision system. Moreover, autoregressive models trained on our token sequences achieve performance comparable to current state-of-the-art methods while requiring fewer tokens for training and inference.

### 1. Introduction

The limits of my language mean the limits of my world. —Ludwig Wittgenstein Tractatus Logico-Philosophicus

The pursuit of compact visual representations has long been a fundamental goal, driving advancements in visual recognition [21, 52] and image generation [55, 62]. One of the earliest approaches, Principal Component Analysis (PCA) [48], achieves this by introducing decorrelated, orthogonal components that capture the most significant variations in the data in a progressively diminishing manner (i.e., orderliness), thereby reducing redundancies. This enables

[Figure 2]

[Figure 3]

[Figure 4]

|[Figure 5]|
|---|

|[Figure 6]|
|---|

|[Figure 7]|
|---|

|[Figure 8]|
|---|

|[Figure 9]|
|---|

|[Figure 10]|
|---|

|[Figure 11]|
|---|

|[Figure 12]|
|---|

|[Figure 13]|
|---|

|[Figure 14]|
|---|

|[Figure 15]|
|---|

|[Figure 16]|
|---|

(a) Semantic-spectrum coupling. Comparison of the frequency-power spectra for different tokenizers. Here, we decompose the tokens from the tokenizers to demonstrate their contribution to the spectrum of the generated image. The VQ-VAE tokenizer [50] is decomposed by performing PCA in its latent token space, and the 1D TiTok [62] is decomposed by replacing all but the first k tokens with a mean token. For SEMANTICIST, on the other hand, we can clearly see that with any number of tokens, the spectrum remains closely matched with the original image, demonstrating that SEMANTICIST can decouple semantics and spectrum in its tokenization process.

Figure 2. Spectrum analysis and the PCA-like structure of our tokenizer.

|[Figure 17]|
|---|

|[Figure 18]|
|---|

|[Figure 19]|
|---|

| | | | |
|---|---|---|---|
| |[Figure 20]| | |
| | | | |
| | | | |

(b) Our tokenizer decomposes the image into visual concepts following a PCA-like coarse-to-fine structure where first few tokens capture most semantic information and the rest refine the details.

PCA to effectively reduce dimensionality while preserving essential information, making it a powerful tool for compact representation. Building on this foundation, Hinton and Salakhutdinov [21] proposed a nonlinear generalization of PCA using autoencoders, which further emphasizes a structured latent space for effective learning and reconstruction error minimization.

While modern approaches such as (vector quantized) variational autoencoders [25, 55] share similar goals as earlier methods—compressing images into a compact, lowdimensional space while minimizing reconstruction errorsthey have largely abandoned the inherent structural properties, such as orthogonality and orderliness, that were critical to the success of earlier PCA-based techniques. For instance, mainstream methods employ a 2D latent space [25, 50, 55], where image patches are encoded into latent vectors arranged in a 2D grid. While this approach achieves high reconstruction quality, it introduces redundancies that scale poorly as image resolution increases. More recently, 1D tokenizers [13, 62] have been proposed to find a more compact set of latent codes for image representation. Although these methods more closely resemble earlier compression approaches [21, 52], they lack structural constraints on latent vectors, making optimization challenging and often resulting in high reconstruction errors.

We further investigate the latent space of state-of-theart methods, including VQ-VAE [50] and TiTok [62], and find that the lack of a structured latent space leads to an inherent tendency for their learned representations to couple significant semantic-level content with less significant lowlevel spectral information—a phenomenon we refer to as semantic-spectrum coupling. As shown in Fig. 2a, increasing the number of latent codes simultaneously affects both the power spectrum (reflecting low-level intensity information) and the reconstruction of semantic content in the image.

Further details on this coupling effect are presented in Fig. 5.

The above motivates us to ask: Can insights from classic PCA techniques be integrated with modern 1D tokenizers to achieve a compact, structured representation of imagesone that reduces redundancy while effectively decoupling semantic information from less important low-level details?

To this end, we reintroduce a PCA-like structure—

incorporating both orthogonality and orderliness—into 1D latent tokens. Specifically, we propose a dynamic nested classifier-free guidance (CFG) strategy during training to induce an orderliness bias in the tokens, enforcing the emergence of a PCA-like structure where token importance progressively decreases. This is achieved by incrementally replacing later tokens in the 1D sequence with a null condition token at an increasing probability, thereby encouraging earlier tokens to capture the most semantically significant features. This strategy also implicitly promotes orthogonal contributions among tokens (see Appendix A). By doing so, our approach ensures a coarse-to-fine token hierarchy with decreasing importance, where each token contributes unique information to reconstruct the image. This stands in contrast to previous 1D tokenizers [13, 62], which enforce a 1D structure but lack the orderliness and orthogonality properties that our method introduces (see Fig. 2a). Moreover, the PCA-like structural property enables a flexible encoding of the image by using the most significant tokens.

However, inducing a PCA-like structure alone is insufficient. The visual world exists in a high-dimensional space, and the nested CFG technique might converge to an arbitrary PCA-like structure that does not necessarily disentangle semantically significant content from less important low-level details. To ensure semantically meaningful features emerge in the latent tokens, we propose leveraging a diffusionbased decoder that follows a spectral autoregressive process [10, 44], which progressively reconstructs images from

low to high frequencies, conditioned on our 1D latent codes. By doing so, the 1D latent tokens are encouraged to focus on capturing semantically significant information while avoiding entanglement with low-level spectral information.

Finally, the 1D latent codes derived from our tokenizer exhibit a PCA-like hierarchical structure with progressively diminishing significance while also being semantically meaningful. As shown in Fig. 1, all reconstructed images retain the semantic essence of the original, while just 64 tokens are sufficient for high-quality reconstruction. Moreover, as illustrated in Fig. 2a, the power spectrum profile remains nearly identical to that of the original image, regardless of the number of tokens. This suggests that the latent tokens effectively capture semantically significant information while avoiding entanglement with low-level spectral details. Notably, the coarse-to-fine structure of latent tokens mirrors the global precedence effect in human vision [15, 34], a phenomenon corroborated by our human perceptual evaluations in Appendix D.1.

In our experiments, we demonstrate that SEMANTICIST can achieve state-of-the-art (SOTA) reconstruction FID scores [20] on the ImageNet validation set, surpassing the previous SOTA tokenizer by almost 10% in FID. SEMANTICIST maintains this SOTA reconstruction performance while maintaining a compact latent suitable for generative modeling. The autoregressive model trained on the tokens generated by SEMANTICIST can achieve comparable performance with the SOTA models while requiring only 32 tokens for training and inference. Additionally, linear probing in the latent space generated by SEMANTICIST also performs up to 63.5% top-1 classification accuracy, indicating SEMANTICIST can capture not only the essence for reconstructing high fidelity images but also the linear separable features.

### 2. Related Work

Image tokenization aims to transform images into a set of compact latent tokens that reduces the computation complexity for generative models like diffusion models and autoregressive models. Thus, we view image tokenization as a way of decomposing images into a learnable “language” for the generative model. VQ-VAE [55] is among the most widely used visual tokenizers. Combining vector quantization into the VAE [25] framework, VQ-VAE can generate discrete tokens for the input image. Improvements over the VQVAE have also been proposed, such as VQGAN [14], which introduces adversarial loss to improve the reconstruction quality, and RQ-VAE [27], which introduces multiple vector quantization stages. The insufficient usage of a codebook for vector quantization in VQ-VAE has also raised issues, and MAGVIT-v2 [60] introduces Look-up Free Quantization (LFQ) to alleviate this issue. Semantic information from a pre-trained visual foundation model [19, 36, 40] has also been shown to be beneficial for improving codebook

usage [66], improving reconstruction [58], and enabling better generation by diffusion models [6]. Maskbit [57] proposed a modernized VQGAN framework with a novel binary quantized token mechanism that enables state-of-the-art conditional image generation performance. Most recently, [18] demonstrates the scaling laws for ViT-based [12] visual tokenizers trained with perceptual and adversarial losses.

Though effective, these models tokenize the image into a 2-dimensional array of tokens where there is no obvious way of performing causal autoregressive modeling. TiTok [62] and SEED [16] are among the first works to introduce tokenizers that generate tokens with a 1-dimensional causal dependency. This dependency enables large language models to understand and generate images. The causal ordering has also been shown to be especially helpful for autoregressive generative models [41]. VAR [51] takes another approach by formulating visual autoregressive modeling as a nextscale prediction problem. A multi-scale residual quantization (MSRQ) tokenizer is proposed in [51] which tokenizes the image in a low-to-high resolution fashion. Following the development of the 1D tokenizers, ALIT [13] demonstrates that 1D tokens can be made adaptive to the image content, leading to an adaptive length tokenizer that can adjust its token length by considering the image entropy, familiarity, and downstream tasks. However, due to the semantic-spectrum coupling effect within these 1D causal tokenizers, the structure within the token sequence generated by these tokenizers is still not clear. In this work, we introduce a novel tokenizer that is able to encode an image to a 1D causal sequence with provable PCA-like structures. By decoupling the semantic features and the spectrum information with a diffusion decoder, this tokenizer is not only useful for vision generative modeling but also closely aligns with human perception and enables better interpretability and downstream performance. Concurrent works [1, 33] also employed similar dropout techniques as ours to form a variable-length 1D tokenizer. We refer interested readers on this topic to this fascinating blog that introduces many concepts in tokenization [11].

Modern generative vision modeling can be roughly divided to two categories, diffusion-based modeling [37] and autoregressive-based modeling [50]. Both modeling techniques typically require a visual tokenizer to compress the input visual signal to a compact space for efficient learning. Diffusion models demonstrate a strong performance since it was introduced [23]. They typically follow an iterative refinement process, which gradually denoises from a noisy image sampled from a Gaussian distribution to a clean image. Developments have made efforts toward sharper sample generation [9, 23], and faster generation [49]. The key development in diffusion models is the introduction of latent diffusion models [45], which allows the diffusion process to be performed in the latent space of a tokenizer [14]. This drastically reduces the computation cost of the diffusion pro-

cess and enables many more applications [37, 42]. Moreover, theoretical understanding of diffusion models has shown that the denoising process can be roughly described as a spectral autoregressive process [10, 44] where the model uses all previously seen lower frequency information to generate higher frequency information. Autoregressive models have also been developed for vision modeling, they typically follow a left-to-right generation process where the model predicts the next pixel given the previous pixels [53, 54]. Recent works have been developing more advanced autoregressive models leveraging the architecture improvements from NLP and advanced image tokenization [50]. Autoregressive models would naturally require an order for which to generate the tokens, some works apply random masking to allow the model to learn random order [4, 29, 61]. VAR [51] introduces a novel paradigm of next-scale-prediction formulation for visual autoregressive modeling, introducing a natural order—scale—to autoregressive modeling. In this work, in observation of the semantic spectrum coupling, we leverage diffusion as a decoder for our tokenizer for its spectral auto-regression property to decouple semantic from spectrum information. Additionally, in our experiments, we demonstrate that we can train autoregressive models on the tokens generated by our tokenizers to achieve a comparable performance with the state-of-the-art models.

### 3. Preliminary

In this section, we provide a concise summary of the denoising diffusion model [23] as a preliminary for understanding

- our SEMANTICIST architecture.

##### 3.1. Denoising Diffusion Models

A T-step Denoising Diffusion Probabilistic Model (DDPM) [23] consists of two processes: the forward process (also referred to as diffusion process), and the reverse inference process. The forward process from data x0 ∼ qdata(x0) to the latent variable xT can be formulated as a fixed Markov chain: q(x1,...,xT|x0) = Tt=1 q(xt|xt−1), where q(xt|xt−1) = N(xt;√1 − βtxt−1,βtI) is a normal distribution, βt is a small positive constant. The forward process gradually perturbs x0 to a latent variable with an isotropic Gaussian distribution platent(xT) = N(0,I).

The reverse process strives to predict the original data x0 from the latent variable xT ∼ N(0,I) through another Markov chain: pθ(x0,...,xT−1|xT) = Tt=1 pθ(xt−1|xt). The training objective of DDPM is to optimize the Evidence Lower Bound (ELBO): L = Ex

0,ϵ||ϵ − ϵθ(xt,t)||22, where ϵ is the Gaussian noise in xt which is equivalent to

▽x

lnq(xt|x0), ϵθ is the model trained to estimate ϵ. Conditional diffusion models [45] maintain the forward process and directly inject the condition z into the training objective:

t

0,ϵ||ϵ − ϵθ(xt,z,t)||22 ,

L = Ex

Concept Tokens (queries)

Null Tokens for CFG

|[Figure 21]|k,v|
|---|---|
| | |

Causal ViT Encoder

Nested CFG

| | |
|---|---|
|First k Concept Tokens| |

Condition

|[Figure 22]|
|---|

DiT Decoder

Figure 3. SEMANTICIST tokenizer architecture. The ViT encoder resamples the 2D image patch tokens into a 1D causal sequence of concept tokens. These concept tokens are then used as conditions to the DiT decoder to reconstruct the original image. To induce a PCA-like structure in the concept tokens, we apply nested CFG.

where z is the condition for generating an image with specific semantics. Except for the conditioning mechanism, the Latent Diffusion Model (LDM) [45] takes the diffusion and inference processes in the latent space of VQGAN [14], which is proven to be more efficient and generalizable than operating on the original image pixels.

### 4. SEMANTICIST Architecture

##### 4.1. Overview

The design of SEMANTICIST aims to generate a compact latent code representation of the visual content with a mathematically-guaranteed structure. As a first step to induce a PCA-like structure, we need to be able to encode the images to a sequence with 1D causal ordering. Thus we require an encoder E : H×W×C → K×D to map the input image of shape H × W × C to K causally ordered tokens, each with a dimension of D. We leverage the vision transformer [12] (ViT) architecture to implement this encoder for SEMANTICIST. Specifically, we first encode the input image x0 ∈ H×W×C to a sequence of image patches Xpatch, we also randomly initialize a set of concept tokens Xconcept = {z1,z2,...,zK} to be passed into the transformer model. Thus, the transformer model within the encoder E takes the below concatenated token sequence for processing:

X = [Xcls;Xpatch;Xconcept],

where Xcls is the [CLS] token. For the patch tokens Xpatch and the [CLS] we do not perform any masking to mimic the standard ViT behavior and for the concept tokens Xconcept, we apply a causal attention masking such that only preceding tokens are visible (as illustrated in Fig. 2b) to enforce them learn a causally ordered tokenization. After the ViT has processed the tokens, the output concept tokens Xconcept = {z1,z2,...,zK} are used as the condition input to the diffusion-based decoder D : K×D × H×W×C →

H×W×C to learn to denoise a noisy version xt of the input image x0. Doing this alone would allow the encoder E to encode the information about the image into the concept tokens Xconcept. However, this information is not structured. To induce a PCA-like structure with the concept tokens, we apply the nested CFG technique N : K×D → K×D that will be introduced in later sections. To summarize, the training process of SEMANTICIST is to minimize this training loss similar to the training of a conditional diffusion model:

0,ϵ||ϵ − D(xt,N(E(X)),t)||22 ,

L = Ex

where t is the condition for the forward process timestep, and ϵ is the noise at timestep t. Note that SEMANTICIST generates continuous tokens instead of quantized tokens like previous works [50, 62]. The reason is that we hope SEMANTICIST to capture the PCA-like variance decay structure, which is hard to capture when using quantized tokens. With the usage of Diffusion Loss [29], our experiments on autoregressive modeling with continuous tokens have shown that this design does not affect generative modeling performance.

##### 4.2. Diffusion-based Decoder

The decoder for SEMANTICIST is based on the conditional denoising diffusion model. The decoder is implemented using the Diffusion-Transformer (DiT) architecture [37], with the condition Xconcept injected by cross-attention. For efficient training, we adopt the LDM technique by training the decoder on the latent space of a pretrained VAE model [45]. This design choice stems from the observation of the semantic-spectrum coupling effect. Whereas, if a deterministic decoder is used to directly regress the pixel values like previous state-of-the-art visual tokenizers, the token space learned by the encoder E will entangle the semantic content and the spectral information of the image. This will prevent the encoder E from learning a semantic meaningful PCA-like structure where the first tokens capture the most important semantic contents. In Sec. 3, we describe the diffusion forward process as gradually corrupting the image with Gaussian noise, which is filtering out more and more highfrequency information. Since the training objective for the diffusion model is to reverse this forward process, the model naturally learns to generate low-frequency information first and then high-frequency details. This is described in [38, 44] as a spectral autoregressive process where the diffusion process itself can already generate the spectral information, leaving the conditions Xconcept to be able to encode most semantic information rather than spectral information.

##### 4.3. Inducing the PCA-like Structure

Although the encoder E and diffusion-based decoder D can produce high-quality reconstructions, their concept tokens Xconcept lack any explicit structural regularization beyond causal ordering. To impose a hierarchical variance-

decaying property similar to PCA [48], we introduce a nested classifier-free guidance (CFG) function

###### N : RK×D × {1,...,K} → RK×D ,

inspired by nested dropout [24, 43] and Matryoshka representations [26]. Specifically, we sample an integer k′ ∼ U{1,...,K} and apply N to the concept tokens Xconcept = (z1,...,zK) as follows:

N(Xconcept,k′) = (z1,...,zk′−1,z∅,...,z∅),

where z∅ are learnable null-condition tokens for masked positions. Intuitively, forcing positions k′,...,K to become

null tokens compels the earlier tokens z1,...,zk′−1 to encode the most salient semantic content. Over the course of training, the uniform sampling of k′ induces a coarse-tofine hierarchy in the concept tokens, mirroring the variancedecaying property of PCA [48].

From a classifier-free guidance [22] perspective, each token zk can be viewed as a conditional signal, and applying N effectively provides an “unconditional” pathway for later tokens. In Appendix A, we formally show that this procedure yields a PCA-like structure in which the earliest tokens capture the largest share of variance. A high-level illustration of our overall architecture is provided in Fig. 3.

##### 4.4. Autoregressive Modeling with SEMANTICIST

With the learned latent token sequences Xconcept = {z1,z2,...,zK} obtained from a well-trained encoder, we can train an autoregressive model for image generation. Specifically, we leverage the architecture of LlamaGen [50] for autoregressive modeling, which is a modern variant of GPT [39] where pre-norm [56] is applied with RMSNorm [64], and SwiGLU [47] activation function is used. As SEMANTICIST adopts continuous tokens, the prediction head of our autoregressive model is a denoising MLP following MAR [29] that is supervised by the diffusion process [23]. Specifically, the autoregressive model is modeling a next token prediction problem of p(zk|z<k,c) = Gdiscrete(z<k,c), where c is the class label embedding and Gdiscrete being the causal transformer to predict the next token with all previous tokens. If the latent tokens generated by E were quantized, we can directly leverage the softmax prediction head to obtain the next token prediction [50]. However, E generates continuous tokens. Thus we leverage the design from [29] to instead predict a condition mk from all previous tokens z<k and c, mk = G(z<k,c). This condition mk is used to condition a small diffusion MLP model to generate the k-th token zk from noise zkT. Specifically, the autoregressive model G and the diffusion MLP model M is trained with a similar diffusion loss as defined in Sec. 3:

k,ϵ||ϵ − M(zkt,G(z<k,c),t)||22 ,

LG = Ez0

|Method #Token Dim. VQ|rFID↓ PSNR↑ SSIM↑<br><br>|Gen. Model Type #Token #Step gFID↓ IS↑|
|---|---|---|
|MaskBit [57] 256 12 ✓ RCG (cond.) [28] 1 256 ✗ MAR [29] 256 16 ✗ TiTok-S-128 [62] 128 16 ✓ TiTok-L-32 [62] 32 8 ✓|1.61 – – – – – 1.22 – –<br><br>1.71 – –<br><br>2.21 – –<br>|MaskBit Mask. 256 256 1.52 328.6 MAGE-L Mask. 1 20 3.49 215.5<br><br>MAR-L Mask. 256 64 1.78 296.0 MaskGIT-L Mask. 128 64 1.97 281.8 MaskGIT-L Mask. 32 8 2.77 194.0<br><br>|
|VQGAN [14] 256 16 ✓ ViT-VQGAN [59] 1024 32 ✓ RQ-VAE [27] 256 256 ✓ VAR [51] 680 32 ✓ ImageFolder [30] 286 32 ✓ LlamaGen [50] 256 8 ✓ CRT [41] 256 8 ✓ Causal MAR [29] 256 16 ✗<br><br>|7.94 – –<br><br>1.28 – – 3.20 – – 0.90 – –<br><br>0.80 – –<br><br>2.19 20.79 0.675 2.36 – –<br><br>1.22 – –<br><br><br><br><br>|Tam. Trans. AR 256 256 5.20 280.3<br><br>VIM-L AR 1024 1024 4.17 175.1 RQ-Trans. AR 256 64 3.80 323.7 VAR-d16 VAR 680 10 3.30 274.4 VAR-d16 VAR 286 10 2.60 295.0<br><br>LlamaGen-L AR 256 256 3.80 248.3 LlamaGen-L AR 256 256 2.75 265.2<br><br>MAR-L AR 256 256 4.07 232.4|
|SEMANTICIST w/ DiT-L 256 16 ✗ SEMANTICIST w/ DiT-XL 256 16 ✗|0.78 21.61 0.626 0.72 21.43 0.613<br><br>|ϵLlamaGen-L AR 32 32 2.57 260.9 ϵLlamaGen-L AR 32 32 2.57 254.0<br><br>|

Table 1. Reconstruction and generation performance on ImageNet. “Dim.” denotes the dimension of the tokens, and “#Step” denotes the number of steps needed for generating the complete image. “#Token” stands for the number of tokens used for image reconstruction (left) and generation (right), respectively.

where zk0 is the ground truth next token zk. This continu-

- ous autoregressive modeling design enables the usage of the continuous tokens from SEMANTICIST to perform generative autoregressive modeling with comparable performance to the current state-of-the-art generative models. As this model uses the noise ϵ as a learning objective, we term this generative model as ϵLlamaGen.

### 5. Experiments

Following the common practice [29], we experiment on ImageNet [8] at a resolution of 256×256, and report FID [20] and IS [46] tested with the evaluation suite provided by [9].

##### 5.1. Implementation Details

SEMANTICIST autoencoder. The encoder of SEMANTICIST is a standard ViT-B/16 [12], except for additional concept tokens and causal attention masks applied to them. We fix the size (token count× dimension) of concept tokens representing an image to 4096 and consider four variants with token dimensions ranging from 16 to 256. Unless otherwise specified, we use 16-dimensional concept tokens (denoted as d16×256) by default, which are more friendly for reconstruction as shown in Fig. 14 in the Appendix. Before being fed to the decoder, the concept tokens are normalized by their own mean and variance following [28]. The decoder is a DiT [37] with a patch size of 2. We experiment across different scales (B, L, and XL) and take DiT-L as default. The decoder operates on the latent space of a publicly available KL-16 VAE provided by [29] to reduce computation cost. The VAE is frozen during training, and both the encoder and decoder are trained from scratch. To enforce the quality of the learned concept tokens and stabilize training, we apply REPA [63] with DINOv2-B [36] as a regularizer to the 8th layer of the DiT decoder.

Autoregressive image modeling. We validate the effectiveness of SEMANTICIST autoencoder by training autoregressive image generation models using LlamaGen [50] combined with diffusion loss [29]. The input sequence is prepended with a [CLS] token for class conditioning, which is randomly dropped out with probability 0.1 during training for classifier-free guidance. At inference time, we use a CFG schedule following [5, 29], and do not apply temperature sampling. Note that the implementation can be general. While our preliminary validation demonstrates promising results, we anticipate better configurations in future work.

##### 5.2. Reconstruction and Generation Quality

Tab. 1 presents the comparison of SEMANTICIST to state-ofthe-art image tokenizers and accompanying generative models. The comparisons on the image reconstructions are made with the variant of the state-of-the-art models with similarsized latent space (i.e., token count × token dimension—the number of floating point numbers used). SEMANTICIST demonstrates a superior reconstruction performance in terms of the rFID score compared to all previous works. Advancing the state-of-the-art performance in image reconstruction by 10% in rFID score compared to the next best model of ImageFolder [30] with a more compact latent space (286×32 for [30] and 256×16 for ours).

In terms of generative modeling, the results in Tab. 1 demonstrate that SEMANTICIST can obtain a gFID score comparable to the state-of-the-art tokenizers for standard autoregressive (AR) modeling. Remarkably, the unique PCA-like structure within the latent space of SEMANTICIST enables efficient generative modeling with fewer tokens. Specifically, ϵLlamaGen only requires being trained and evaluated on the first 32 tokens from SEMANTICIST to achieve a gFID score comparable with the state-of-theart generative models on ImageNet. This efficiency in the number of tokens used for generative modeling would allow

Metrics Linear Probe Accuracy

Model Configs

d256×16 d128×32

d64×64

| |
|---|

d16×256

PCA Percentage (%)

96.5

LinearProbeAccuracy

| | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |

PCAPercentage(%)

0.6

96.0

0.5

95.5

95.0

0.4

94.5

0.3

94.0

0.2

1 2 4 8 16 32 64 128 256

#Token

Figure 4. The explained variance ratio from SEMANTICIST’s PCAlike structure and the linear probing accuracy on the tokens.

ϵLlamaGen to use a much smaller number of inference steps to achieve better results.

##### 5.3. Representation and Structure

We study the property of the structured latent space of SEMANTICIST. In Fig. 4, we first generate a PCA-like variance explained plot by varying the number of tokens used to compute the averaged diffusion loss over the validation set and all diffusion timesteps. After obtaining the averaged loss, the loss value of using all null conditions is treated as the upper bound, and the loss value of using all conditions is treated as the lower bound. We plot the reduction percentage for the contribution of each token to the diffusion loss. It is clear that SEMANTICIST forms a PCA-like structure among the contributions of its tokens. Furthermore, we perform linear probing on the extracted tokens; the plots are also available in Fig. 4. Comparing this against the variance-explained curve, we can see the linear probing accuracy can reach the highest performance when using a low number of tokens and then gradually decrease as more tokens are used. This reveals that SEMANTICIST tends to store the most salient features (i.e., the object category) in the first few tokens, and thus, they benefit from linear probing. And yet, when adding more tokens, details of the scene are added, which causes the linear probing readout to lose category information.

##### 5.4. Semantic-spectrum Coupling

We further demonstrate the semantic-spectrum coupling effect in Fig. 5. The power-frequency plots are obtained by taking the 2D Fourier transform of the image, then averaging the magnitude-squared values over concentric circles in the frequency domain to get a 1D power distribution. The horizontal axis represents spatial frequency (higher values correspond to finer details), and the vertical axis shows the power at each frequency. As we can see from Fig. 5, for the TiTok [62] model, the semantic content only emerges from using 2/4 to 3/4 of all tokens. From the power-frequency plot,

[Figure 23]

- Figure 5. Reconstructed images and their corresponding powerfrequency plots, illustrating semantic-spectrum coupling. Each column shows reconstructions using only the first k tokens, increasing from left to right, alongside a plot of the reconstructed image’s frequency power (blue) overlaid on the ground-truth (red) image.

1 2 4 8 16 32 64 128 256

#Token

0.6

1.3

2.7

5.6

11.7

24.7

51.9

rFIDImageNet50K

DiT-B, CFG=1.0 DiT-B, CFG=3.0

DiT-L, CFG=1.0 DiT-L, CFG=3.0

DiT-XL, CFG=1.0 DiT-XL, CFG=3.0

- Figure 6. Scaling behavior of different-sized DiT decoder (qualitative results can be found in Fig. 14 in the Appendix).

it is clear that the model can not match the correct power distribution with fewer tokens and that when adding more tokens, not only is the power distribution shifting toward the ground truth distribution, but also, the semantic content emerges. This is what we term the “semantic-spectrum coupling” effect, where, when adding more tokens, both semantic content and spectral information are encoded. On the other hand, it is very clear that SEMANTICIST can match the power distribution of the ground truth with only 1/4 of the tokens, and later tokens contribute more to the semantic content of the image, successfully disentangling semantic content from the spectral information.

##### 5.5. Ablation Study

We ablate the effect of the scale of the diffusion decoder in SEMANTICIST, the number of tokens used for reconstruction,

[Figure 24]

Figure 7. Examples of the intermediate generation results of ϵLlamaGen-L trained on SEMANTICIST tokens (see more from Fig. 18).

and the strength of classifier-free-guidance in Fig. 6. SEMANTICIST follows a very clear scaling behavior in terms of the number of tokens and the model size. Notably, strengthening the classifier-free-guidance scale can greatly help the reconstruction performance with a smaller model and fewer tokens. More ablation studies on ϵLlamaGen and SEMANTICIST are available in Appendix D.4.

##### 5.6. Discussion on Qualitative Results

In Fig. 1, we have shown that SEMANTICIST can consistently produce semantic-meaningful high-quality images with any number of tokens that progressively refine towards the reconstruction target. Notably, the original image can be represented with as few as 32 tokens. This is especially helpful when we want to re-purpose the tokens for class-conditional image generation, in which the task is to produce semanticconsistent images instead of exact reconstruction. Thus, we can train the ϵLlamaGen autoregressive model efficiently with only the first few concept tokens (32 in our case). In Figs. 7 and 18, we demonstrate the effectiveness of autoregressive modeling following this strategy. The model is a ϵLlamaGen-L trained on the first 32 concept tokens of a SEMANTICIST (w/ DiT-XL) tokenizer. It is encouraging to see that the first token the model generates already sketches the majority of the scene well and even generates highly faithful

images in easier cases like animals. When more tokens are generated, the image is gradually refined and converges to a highly detailed and visually coherent generation.

### 6. Conclusion

We introduce SEMANTICIST, a PCA-like structured 1D tokenizer that addresses semantic-spectrum coupling through a dynamic nested classifier-free guidance strategy and a diffusion-based decoder. Our method enforces a coarse-tofine token hierarchy that captures essential semantic features while maintaining a compact latent representation. Our experiments demonstrate that SEMANTICIST achieves state-ofthe-art reconstruction FID scores on the ImageNet validation set, surpassing the previous SOTA tokenizer by almost 10% in FID. Moreover, the autoregressive model ϵLlamaGen trained on SEMANTICIST’s tokens attains comparable performance to current SOTA methods while requiring only 32 tokens for training and inference. Additionally, linear probing in the latent space yields up to 63.5% top-1 classification accuracy, confirming the effectiveness of our approach in capturing semantic information. These results highlight the promise of SEMANTICIST for high-fidelity image reconstruction and generative modeling, paving the way for more efficient and compact visual representations.

### Acknowledgment

This work has been supported in part by Hong Kong Research Grant Council—Early Career Scheme (Grant No. 27209621), General Research Fund Scheme (Grant No. 17202422, 17212923), Theme-based Research (Grant No. T45-701/22-R), and Shenzhen Science and Technology Innovation Commission (SGDX20220530111405040). Part of the described research work is conducted in the JC STEM Lab of Robotics for Soft Materials funded by The Hong Kong Jockey Club Charities Trust. We sincerely appreciate the dedicated support we received from the participants of the human study. We are also grateful to Anlin Zheng and Haochen Wang for helpful suggestions on the design of technical details.

### References

- [1] Roman Bachmann, Jesse Allardice, David Mizrahi, Enrico Fini, O˘guzhan Fatih Kar, Elmira Amirloo, Alaaeldin ElNouby, Amir Zamir, and Afshin Dehghan. FlexTok: Resampling images into 1d token sequences of flexible length. In ICML, 2025. 3, 2
- [2] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5VL technical report. arXiv:2502.13923, 2025. 4
- [3] Jerome S. Bruner and Mary C. Potter. Interference in visual recognition. Science, 144(3617):424–425, 1964. 2
- [4] Huiwen Chang, Han Zhang, Lu Jiang, Ce Liu, and William T. Freeman. MaskGIT: Masked generative image transformer. In CVPR, 2022. 4
- [5] Huiwen Chang, Han Zhang, Jarred Barber, Aaron Maschinot, Jose Lezama, Lu Jiang, Ming-Hsuan Yang, Kevin Patrick Murphy, William T. Freeman, Michael Rubinstein, Yuanzhen Li, and Dilip Krishnan. Muse: Text-to-image generation via masked generative Transformers. In ICML, 2023. 6, 3
- [6] Hao Chen, Yujin Han, Fangyi Chen, Xiang Li, Yidong Wang, Jindong Wang, Ze Wang, Zicheng Liu, Difan Zou, and Bhiksha Raj. Masked autoencoders are effective tokenizers for diffusion models. arXiv:2502.03444, 2025. 3
- [7] Yinbo Chen, Rohit Girdhar, Xiaolong Wang, Sai Saketh Rambhatla, and Ishan Misra. Diffusion autoencoders are scalable image tokenizers. arXiv:2501.18593, 2025. 2
- [8] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. ImageNet: A large-scale hierarchical image database. In CVPR, 2009. 6, 3
- [9] Prafulla Dhariwal and Alex Nichol. Diffusion models beat GANs on image synthesis. In NeurIPS, 2021. 3, 6
- [10] Sander Dieleman. Diffusion is spectral autoregression, 2024.

- 2, 4

[11] Sander Dieleman. Generative modelling in latent space, 2025.

- 3

- [12] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In ICLR, 2021. 3, 4, 6
- [13] Shivam Duggal, Phillip Isola, Antonio Torralba, and William T. Freeman. Adaptive length image tokenization via recurrent allocation. In ICLR, 2025. 2, 3
- [14] Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming Transformers for high-resolution image synthesis. In CVPR,

2021. 3, 4, 6

- [15] Li Fei-Fei, Asha Iyer, Christof Koch, and Pietro Perona. What do we perceive in a glance of a real-world scene? Journal of Vision, 7(1):10–10, 2007. 3, 2, 4
- [16] Yuying Ge, Yixiao Ge, Ziyun Zeng, Xintao Wang, and Ying Shan. Planting a seed of vision in large language model. arXiv:2307.08041, 2023. 3
- [17] Yuying Ge, Yizhuo Li, Yixiao Ge, and Ying Shan. Divot: Diffusion powers video tokenizer for comprehension and generation. arXiv:2412.04432, 2024. 2
- [18] Philippe Hansen-Estruch, David Yan, Ching-Yao Chung, Orr Zohar, Jialiang Wang, Tingbo Hou, Tao Xu, Sriram Vishwanath, Peter Vajda, and Xinlei Chen. Learnings from scaling visual tokenizers for reconstruction and generation. arXiv:2501.09755, 2025. 3
- [19] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Doll´ar, and Ross Girshick. Masked autoencoders are scalable vision learners. In CVPR, 2022. 3
- [20] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. GANs trained by a two time-scale update rule converge to a local Nash equilibrium. In NeurIPS, 2017. 3, 6
- [21] Geoffrey E. Hinton and Ruslan R. Salakhutdinov. Reducing the dimensionality of data with neural networks. Science, 313

(5786):504–507, 2006. 1, 2

- [22] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv:2207.12598, 2022. 5
- [23] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In NeurIPS, 2020. 3, 4, 5
- [24] Matthew Ho, Xiaosheng Zhao, and Benjamin Wandelt. Information-ordered bottlenecks for adaptive semantic compression. arXiv:2305.11213, 2023. 5
- [25] Diederik P. Kingma and Max Welling. Auto-encoding variational bayes. In ICLR, 2014. 2, 3
- [26] Aditya Kusupati, Gantavya Bhatt, Aniket Rege, Matthew Wallingford, Aditya Sinha, Vivek Ramanujan, William Howard-Snyder, Kaifeng Chen, Sham Kakade, Prateek Jain, and Ali Farhadi. Matryoshka representation learning. In NeurIPS, 2022. 5
- [27] Doyup Lee, Chiheon Kim, Saehoon Kim, Minsu Cho, and Wook-Shin Han. Autoregressive image generation using residual quantization. In CVPR, 2022. 3, 6
- [28] Tianhong Li, Dina Katabi, and Kaiming He. Return of unconditional generation: A self-supervised representation generation method. In NeurIPS, 2024. 6, 2, 3

- [29] Tianhong Li, Yonglong Tian, He Li, Mingyang Deng, and Kaiming He. Autoregressive image generation without vector quantization. In NeurIPS, 2024. 4, 5, 6, 3
- [30] Xiang Li, Kai Qiu, Hao Chen, Jason Kuen, Jiuxiang Gu, Bhiksha Raj, and Zhe Lin. Imagefolder: Autoregressive image generation with folded tokens. In ICLR, 2025. 6
- [31] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In ICLR, 2019. 3
- [32] Nanye Ma, Mark Goldstein, Michael S Albergo, Nicholas M Boffi, Eric Vanden-Eijnden, and Saining Xie. SiT: Exploring flow and diffusion-based generative models with scalable interpolant transformers. In ECCV, 2024. 3
- [33] Keita Miwa, Kento Sasaki, Hidehisa Arai, Tsubasa Takahashi, and Yu Yamaguchi. One-d-piece: Image tokenizer meets quality-controllable compression. In ICML Tokenization Workshop, 2025. 3
- [34] David Navon. Forest before trees: The precedence of global features in visual perception. Cognitive Psychology, 9(3): 353–383, 1977. 3, 2
- [35] Aude Oliva and Philippe G. Schyns. Diagnostic colors mediate scene recognition. Cognitive Psychology, 41(2):176–210,

2000. 2

- [36] Maxime Oquab, Timoth´ee Darcet, Theo Moutakanni, Huy V. Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Russell Howes, Po-Yao Huang, Hu Xu, Vasu Sharma, Shang-Wen Li, Wojciech Galuba, Mike Rabbat, Mido Assran, Nicolas Ballas, Gabriel Synnaeve, Ishan Misra, Herve Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. DINOv2: Learning robust visual features without supervision. TMLR, 2024. 3, 6
- [37] William Peebles and Saining Xie. Scalable diffusion models with transformers. In ICCV, 2023. 3, 4, 5, 6, 2
- [38] Konpat Preechakul, Nattanat Chatthee, Suttisak Wizadwongsa, and Supasorn Suwajanakorn. Diffusion autoencoders: Toward a meaningful and decodable representation. In CVPR,

2022. 5, 2

- [39] Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever. Improving language understanding by generative pre-training, 2018. 5
- [40] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In ICML, 2021. 3, 4
- [41] Vivek Ramanujan, Kushal Tirumala, Armen Aghajanyan, Luke Zettlemoyer, and Ali Farhadi. When worse is better: Navigating the compression-generation tradeoff in visual tokenization. arXiv:2412.16326, 2024. 3, 6
- [42] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with CLIP latents. arXiv:2204.06125, 2022. 4
- [43] Oren Rippel, Michael Gelbart, and Ryan Adams. Learning ordered representations with nested dropout. In ICML, 2014. 5

- [44] Severi Rissanen, Markus Heinonen, and Arno Solin. Generative modelling with inverse heat dissipation. arXiv:2206.13397, 2022. 2, 4, 5
- [45] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, 2022. 3, 4, 5
- [46] Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen. Improved techniques for training GANs. In NeurIPS, 2016. 6
- [47] Noam Shazeer. GLU variants improve Transformer. arXiv:2002.05202, 2020. 5
- [48] Jonathon Shlens. A tutorial on principal component analysis. arXiv:1404.1100, 2014. 1, 5
- [49] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In ICLR, 2021. 3
- [50] Peize Sun, Yi Jiang, Shoufa Chen, Shilong Zhang, Bingyue Peng, Ping Luo, and Zehuan Yuan. Autoregressive model beats diffusion: Llama for scalable image generation. arXiv:2406.06525, 2024. 2, 3, 4, 5, 6
- [51] Keyu Tian, Yi Jiang, Zehuan Yuan, Bingyue Peng, and Liwei Wang. Visual autoregressive modeling: Scalable image generation via next-scale prediction. In NeurIPS, 2024. 3, 4, 6
- [52] Matthew A. Turk and Alex P. Pentland. Face recognition using eigenfaces. In CVPR, 1991. 1, 2
- [53] A¨aron van den Oord, Nal Kalchbrenner, and Koray Kavukcuoglu. Pixel recurrent neural networks. In ICML,

2016. 4

- [54] A¨aron van den Oord, Nal Kalchbrenner, Oriol Vinyals, Lasse Espeholt, Alex Graves, and Koray Kavukcuoglu. Conditional image generation with PixelCNN decoders. In NeurIPS, 2016. 4
- [55] A¨aron van den Oord, Oriol Vinyals, and Koray Kavukcuoglu. Neural discrete representation learning. In NeurIPS, 2017. 1, 2, 3
- [56] Qiang Wang, Bei Li, Tong Xiao, Jingbo Zhu, Changliang Li, Derek F. Wong, and Lidia S. Chao. Learning deep Transformer models for machine translation. In ACL, 2019. 5
- [57] Mark Weber, Lijun Yu, Qihang Yu, Xueqing Deng, Xiaohui Shen, Daniel Cremers, and Liang-Chieh Chen. MaskBit: Embedding-free image generation via bit tokens. TMLR,

2024. 3, 6

- [58] Jingfeng Yao and Xinggang Wang. Reconstruction vs. generation: Taming optimization dilemma in latent diffusion models. arXiv:2501.01423, 2025. 3
- [59] Jiahui Yu, Xin Li, Jing Yu Koh, Han Zhang, Ruoming Pang, James Qin, Alexander Ku, Yuanzhong Xu, Jason Baldridge, and Yonghui Wu. Vector-quantized image modeling with improved VQGAN. In ICLR, 2022. 6
- [60] Lijun Yu, Jos´e Lezama, Nitesh B. Gundavarapu, Luca Versari, Kihyuk Sohn, David Minnen, Yong Cheng, Vighnesh Birodkar, Agrim Gupta, Xiuye Gu, Alexander G. Hauptmann, Boqing Gong, Ming-Hsuan Yang, Irfan Essa, David A. Ross, and Lu Jiang. Language model beats diffusion – tokenizer is key to visual generation. In ICLR, 2024. 3

- [61] Qihang Yu, Ju He, Xueqing Deng, Xiaohui Shen, and LiangChieh Chen. Randomized autoregressive visual generation. arXiv:2411.00776, 2024. 4
- [62] Qihang Yu, Mark Weber, Xueqing Deng, Xiaohui Shen, Daniel Cremers, and Liang-Chieh Chen. An image is worth 32 tokens for reconstruction and generation. In NeurIPS,

2024. 1, 2, 3, 5, 6, 7, 4

- [63] Sihyun Yu, Sangkyung Kwak, Huiwon Jang, Jongheon Jeong, Jonathan Huang, Jinwoo Shin, and Saining Xie. Representation alignment for generation: Training diffusion transformers is easier than you think. In ICLR, 2025. 6, 4, 5
- [64] Biao Zhang and Rico Sennrich. Root mean square layer normalization. In NeurIPS, 2019. 5
- [65] Long Zhao, Sanghyun Woo, Ziyu Wan, Yandong Li, Han Zhang, Boqing Gong, Hartwig Adam, Xuhui Jia, and Ting Liu. ϵ-VAE: Denoising as visual decoding. arXiv:2410.04081,

2024. 2

- [66] Lei Zhu, Fangyun Wei, Yanye Lu, and Dong Chen. Scaling the codebook size of VQ-GAN to 100,000 with a utilization rate of 99%. In NeurIPS, 2024. 3

## “Principal Components” Enable A New Language of Images Supplementary Material

### Contents

- A. Proof for PCA-like structure 1
- B. Additional Related Work 2

- B.1. Concurrent Related Work . . . . . . . . . . 2
- B.2. Related Work on Human Perception . . . . . 2
- B.3. Related Work on Diffusion-Based Tokenizers 2

- C. Additional Implementation Details 2

- C.1. Semanticist Autoencoder . . . . . . . . . . . 2
- C.2. Autoregressive Modeling . . . . . . . . . . 3
- C.3. Linear Probing . . . . . . . . . . . . . . . . 3

- D. Additional Experiment Results 3

- D.1. Human Perception Test . . . . . . . . . . . 3
- D.2. Zero-Shot CLIP on Reconstructed Images . 4
- D.3. Semantic Spectrum Coupling Effect Results 4
- D.4. Additional Ablation Study . . . . . . . . . . 4
- D.5. Qualitative Results . . . . . . . . . . . . . . 5

### Author Contribution Statement

X.W. and B.Z. conceived the study and guided its overall direction and planning. X.W. proposed the original idea of semantically meaningful decomposition for image tokenization. B.Z. developed the theoretical framework for nested CFG and the semantic spectrum coupling effect and conducted the initial feasibility experiments. X.W. further refined the model architecture and scaled the study to ImageNet. B.Z. led the initial draft writing, while X.W. designed the figures and plots. I.E., J.D., and X.Q. provided valuable feedback on the manuscript. All authors contributed critical feedback, shaping the research, analysis, and final manuscript.

### Limitations and Broader Impacts

Our tokenizer contributes to structured visual representation learning, which may benefit image compression, retrieval, and generation. However, like other generative models, it could also be misused for deepfake creation, misinformation, or automated content manipulation. Ensuring responsible use and implementing safeguards remains an important consideration for future research. SEMANTICIST also presents several limitations, for example, we employ a diffusionbased decoder, but alternative generative models like flow matching or consistency models could potentially improve efficiency. Additionally, our framework enforces a PCA-like structure, further refinements, such as adaptive tokenization or hierarchical models, could enhance flexibility.

### A. Proof for PCA-like structure

The conditional denoising diffusion model is using a neural network ϵθ(xt,z,t) to approximated the score function ▽x

lnq(xt|x0) which guides the transition from a noised image xt to the clean image x0. For the conditional diffusion decoder in SEMANTICIST, the score function can be decomposed as:

t

ϵθ(xt,z1,...,zk) = ϵθ(xt,∅) +

k

γi∆ϵθ(xt,zi),

i=1

where ∅ is the null condition, γi is the guidance scale, and ∆ϵθ(xt,zi) = ϵθ(xt,z1,...,zi) − ϵθ(xt,z1,...,zi−1) represents the increment contribution of the concept token condition zi to the score function. Thus, we can rewrite the diffusion training objective with k conditions with the following:

Lk = E ϵ − ϵθ(xt,∅) +

k

γi∆ϵθ(xt,zi)

i=1

2

.

Orthogonality between contribution of concept tokens. At the optimal convergence, the gradient of Lk w.r.t ∆ϵθ(xt,zi) is zero, thus give us:

∂Lk ∂∆ϵθ(xt,zi)

= E ϵ − ϵθ(xt,∅) −

= 0.

k

γj∆ϵθ(xt,zj) γi

j=1

Since model is at convergence, the residual term ϵ − ϵθ(xt,∅)− kj=1 γj∆ϵθ(xt,zj) can not be further reduced by making further changes to the adjustment from the i-th concept token ∆ϵθ(xt,zj). In other words, the residual term and all active conditions ∆ϵθ(xt,zj) are orthogonal to each other. Next, we can use induction to prove that at convergence, all ∆ϵθ(xt,zj) terms are orthogonal to each other similar to PCA. For the case of k = 1, we only use one concept token to condition the model, thus we can have:

E[(ϵ − ϵθ(xt,∅) − γ1∆ϵθ(xt,z1))∆ϵθ(xt,z1)] = 0. For the case of k = 2, for (i = 1,2), we have:

E ϵ − ϵθ(xt,∅) −

2

γj∆ϵθ(xt,zj) ∆ϵθ(xt,zi) = 0.

j=1

By substituting the k = 1 case into this, it can be seen that E ∆ϵθ(xt,z1)⊤∆ϵθ(xt,z2) = 0. Assuming this

orthogonality holds for the first k − 1 concept tokens:

- E ∆ϵθ(xt,zi)⊤∆ϵθ(xt,zj) = 0 ∀i,j < k,i ̸= j. Then for i < k, by substituting

ϵ − ϵθ(xt,∅) =

we can have:

k−1

γj∆ϵθ(xt,zj) + γk∆ϵθ(xt,zk),

j=1

###### E ∆ϵθ(xt,zi)⊤∆ϵθ(xt,zk) = 0.

Thus, the orthogonality propagates to all pairs (zi,zk) for i < k. By induction, we have orthogonality between all pairs of concept tokens.

Variance Explained Hierarchy. Assuming the true noise ϵ can be reconstructed using the conditional model, we have:

k

γi∆ϵθ(xt,zi) + residual.

ϵ ≈ ϵθ(xt,∅) +

i=1

Given the orthogonality of ∆ϵθ(xt,zi) we have proven earlier, the total variance can be decomposed as:

Var(ϵ) =

k

Var(γi∆ϵ(xt,zi)) + Var(residual).

i=1

Let λi = Var(γi∆ϵθ(xt,zi)), representing the variance explained by concept token condition zi. Our dropout design would have the training objective forces:

λ1 ≥ λ2 ≥ ··· ≥ λk ,

as each concept token zi is trained to explain the maximal residual variance after accounting for concept tokens z1,...,zi−1.

Thus, combining the orthogonality and the variance decay, SEMANTICIST provably grounds the emergence of a PCAlike hierarchical structure in the learned concept tokens. Providing a simple, effective, and explainable architecture for visual tokenization.

### B. Additional Related Work

##### B.1. Concurrent Related Work

Concurrent work [1] introduces a 1D tokenizer that focuses on adaptive-length tokenization by resampling sequences of 1D tokens from pre-trained 2D VAE tokens. In contrast, our

- encoder builds on raw RGB images. More importantly, our approach is motivated by a fundamentally different objective

— reintroducing a PCA-like structure into visual tokenization to enforce a structured, hierarchical latent representation. Furthermore, our tokenizer is continuous rather than discrete, setting it apart from [1] and allowing it to better capture the

variance-decaying properties inherent to PCA. Additionally, we identify and resolve the semantic-spectrum coupling effect, a key limitation in existing visual tokenization methods that have not been previously addressed.

##### B.2. Related Work on Human Perception

Human perception of visual stimuli has been shown to follow the global precedence effect [34], where the global information of the scene is processed before the local information. In [15], controlled experiments of presentation time on human perception of visual scenes have further confirmed the global precedence effect, where less information (presentation time) is needed to access the non-semantic, sensory-related information of the scene compared to the semantically meaningful, object- or scene-related information. Similar results have been reported in [3], where sensory attributes are more likely to be processed when the scene is blurred. Moreover, [35] has suggested that reliable structural information can be quickly extracted based on coarse spatial scale information. These results suggest that human perception of visual stimuli is hierarchical, where the global information of the scene is processed before the local information. As we have shown in the main paper, SEMANTICIST can naturally emerge a similar hierarchical structure in the token sequence, where the first few tokens encode the global information of the scene and the following tokens encode the local information of the scene. This hierarchical structure is provably PCA-like, similar to the hierarchical nature of human perception of visual stimuli.

##### B.3. Related Work on Diffusion-Based Tokenizers

The usage of a diffusion-based decoder has been explored by several works [7, 17, 65]. Zhao et al. [65] proposed the usage of a diffusion-based decoder as a paradigm shift from single-step reconstruction of previous tokenizers to the diffusion-based iterative refinement process. Chen et al. [7] further scale this idea on more modern DiT [37] architecture and describe the scaling law for such diffusion-based tokenizers. Ge et al. [17] applied this idea to a video tokenizer, enabling better reconstruction and understanding of video content. However, these previous works overlook the benefit of the diffusion-based decoder in that it can disentangle the semantic content from the spectral information. Additionally, these works still apply the 2D grid-based structure for encoding the image without considering the latent structure of the token space.

### C. Additional Implementation Details

##### C.1. Semanticist Autoencoder

Model architecture. As shown in Fig. 3, the SEMANTICIST tokenizer follows the diffusion autoencoder [28, 38] paradigm: a visual encoder takes RGB images as input and

- encodes them into latent embeddings to condition a diffusion model for reconstruction. In our case, the visual encoder is a ViT-B/16 [12] with a sequence of concept tokens concatenated with image patches as input. The concept tokens have full attention with patch tokens, but are causal to each other. Before being fed to the decoder, the concept tokens also go through a linear projector, and are then normalized by their mean and variance. To stabilize training, we also apply drop path with a probability of 0.1 to the ViT. For the DiT decoder, we concatenate the patch tokens (condition) with noisy patches as input, and the timesteps are still incorporated via AdaLN following common practice [37].

Nested classifier-free guidance (CFG). For the DiT decoder, we randomly initialize k (number of concept tokens) learnable null-conditioning tokens. During each training iteration, we uniformly sample a concept token index k′, and corresponding null tokens replace all tokens with larger indices. To facilitate the learning of the encoder, we do not enable nested CFG in the first 50 training epochs. During inference, CFG can be applied to concept tokens independently following the standard practice [37].

Training. We follow [28] for training details of the tokenizer. Specifically, the model is trained using the AdamW [31] optimizer on ImageNet [8] for 400 epochs with a batch size of 2048. The base learning rate is 2.5e-5, which is scaled by lr = lrbase×batch size/256. The learning rate is also warmed up linearly during the first 100 epochs, and then gradually decayed following the cosine schedule. No weight decay is applied, and β1 and β2 of AdamW are set to 0.9 and 0.95. During training, the image is resized so that the smaller side is of length 256, and then randomly flipped and cropped to 256×256. We also apply a gradient clipping of 3.0 to stabilize training. The parameters of the model are maintained using exponential moving average (EMA) with a momentum of 0.999.

Inference. Because of the nature of the PCA structure, it is possible to obtain reasonable reconstruction results with only the first few concept tokens. In implementation, we achieve this by padding missing tokens with their corresponding null conditioning tokens and then feeding the full sequence to the DiT decoder.

##### C.2. Autoregressive Modeling

Model architecture. The ϵLlamaGen roughly follows the LlamaGen architecture with the only change of using a diffusion MLP as the prediction head instead of a softmax head. To perform the classifier-free-guidance, we use one [CLS] token to guide the generation process of ϵLlamaGen. As certain configurations of SEMANTICIST can yield highdimensional tokens, we made a few adjustments to the model

architecture of ϵLlamaGen to allow it to learn with highdimensional tokens. Specifically, we use a 12-layer MLP with each layer having 1536 hidden neurons as the prediction head and use the stochastic interpolant formulation [32] to train the diffusion MLP. The classifier-free guidance is also slightly modified: we concatenate the [CLS] token with the input to the diffusion MLP along the feature axis and then project back to the original feature dimension to feed into the diffusion MLP. These changes allow us to train auto-regressive models on high-dimensional (e.g., 256dimensional) tokens with improved stability compared to the original version proposed in [29]. However, we expect future research to drastically simplify this model architecture.

Training. The ϵLlamaGen is trained for 400 epochs with cached latents generated by pretrained SEMANTICIST on the ImageNet dataset with TenCrop and random horizontal flipping augmentations. We use a batch size of 2048, and apply a 100-epoch warmup for the base learning rate of 1e-4, which is scaled similarly as the SEMANTICIST w.r.t. the batch size. After warmup, the learning rate is fixed. Weight decay of 0.05 and gradient clipping of 1.0 are applied. In our experiments, we find that later concept tokens have diminishing returns or are even harmful for ϵLlamaGen, thus only train ϵLlamaGen with the first few tokens. Specifically, the ϵLlamaGen-L model is trained with 32 concept tokens.

Inference. In the inference stage, we use the same linear classifier-free guidance schedule as MAR [29] and MUSE [5]. The schedule tunes down the guidance scale of small-indexed tokens to improve the diversity of generated samples, thus being more friendly for gFID. When reporting gFID, we disable CFG for SEMANTICIST’s DiT decoder, tune the guidance scale of the autoregressive model, and report the best performance.

##### C.3. Linear Probing

We utilized the sklearn library to perform the linear probing experiments, the encoder weights are frozen, and we encode each image to its token representation. The linear classifier is learned on the token space without applying any data augmentation.

### D. Additional Experiment Results

##### D.1. Human Perception Test

We are interested in understanding whether the tokens learned by SEMANTICIST follow a human-like perception effect, namely the global precedence effect [34] where the global shape and semantics are picked up within a very short period of exposure. Thus, we designed a human perception test to evaluate whether SEMANTICIST generates tokens that closely follow human perception. Specifically, we generate

0.56

#### PreferenceScore

0.54

0.52

0.50

0.48

Reveal 100ms Reveal 200ms Reveal 300ms

0.46

16 64 128 256

Token Dimension

Figure 8. The preference score from the human perception test, all models and test configurations obtained a score close to 0.5, indicating SEMANTICIST can encode images as effectively as human language does.

images by only reconstructing from the first two tokens from SEMANTICIST. Distractor images are also generated by first captioning the image with Qwen2.5VL [2] and then generate the image with a stable-diffusion model [45]. Following the setup of [15], we only reveal the generated images and the distractors by a very short reveal time, and then ask the participants to choose which images more closely align with the original image. For evaluation, we give the participant’s preference to distractor image zero points, the preference to the generated image one point, and in the case of a tie, we give 0.5 points. Fig. 8 presents the averaged preference score with different token dimensions and reveal time. SEMANTICIST is able to obtain a score close to 0.5 under all cases, indicating that SEMANTICIST can encode the image’s global semantic content close to how state-of-the-art vision language models [2] encode the image in language space. A web-based human perception test interface is provided along with this appendix.

##### D.2. Zero-Shot CLIP on Reconstructed Images

We also study the property of the SEMANTICIST latent space by reconstructing from it. Fig. 9 demonstrate the zero-shot accuracy of a pretrained CLIP [40] model on the imagenet validation set reconstructed by SEMANTICIST. For all model variants, the zero-shot performance improves with the number of tokens, with models using more dimensions per token achieving better performance with a smaller number of tokens, indicating that with more dimensions, SEMANTICIST is able to learn the semantic content with fewer tokens. Fig. 6 provides the rFID score on the ImageNet validation set with a varying number of tokens, similar conclusions can be drawn. Additionally, Fig. 6 also provides the scaling behavior of SEMANTICIST, we can observe that SEMANTICIST not only enjoys a structured latent space, but also demonstrates a promising scaling.

0.7

CLIPZeroshotAccuracy

0.6

0.5

0.4

GT Image

0.3

d256×16 d128×32 d64×64

0.2

d16×256

0.1

1 2 4 8 16 32 64 128 256

#Token

- Figure 9. CLIP zero-shot accuracy on reconstructed images.

[Figure 25]

|[Figure 26]|
|---|

|[Figure 27]|
|---|

|[Figure 28]|
|---|

|[Figure 29]|
|---|

- Figure 10. Frequency-power spectra of TiTok decomposed with PCA at feature dimensions. The learning of semantic contents and spectral information is coupled.

##### D.3. Semantic Spectrum Coupling Effect Results

In Fig. 10, we present the power frequency plot of performing PCA to decompose the latent token space of TiTok [62]. A similar effect as the PCA decomposition on VQ-VAE [50] and the first k token decomposition on TiTok [62] is observed. This result further demonstrates that the latent space of TiTok [62] entangles the semantic contents and the spectral information.

##### D.4. Additional Ablation Study

In Fig. 12, we show the results of SEMANTICIST with d64×64 tokens trained with or without REPA [63] evaluated by reconstruction FID on ImageNet 50K validation set. Despite the performance with full tokens being similar, adding REPA significantly improves the contribution of each (especially the first few) tokens. This naturally fits our need for PCA-like structure and is thus adopted as the default.

d256 × 16

###### d128 × 32

###### d64 × 64

d16 × 256

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| |C C<br><br>|FG=1.0 FG=1.5| | | | |
| |C|FG=2.0| | | | |
| | | | | | | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | |CFG= CFG=|1.0 1.5| | | | |
| | |CFG=|2.0| | | | |
| | | | | | | | |

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | |CFG CFG|=1.0 =1.5| | | | | |
| | | | | | | | | |
| | |CFG|=2.0| | | | | |
| | | | | | | | | |
| | | | | | | | | |

| | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | |CF CF|G= G=|1.0 1.5| |
| | | | | | | | | | |
| | | | | | |CF|G=|3.0| |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |

5.6

10.4

11.0

29.8

rFIDImageNet50K

4.0

6.6

6.9

15.9

2.8

4.2

4.4

8.5

2.0

2.7

2.8

4.6

1.4

1.7

1.8

2.4

1.0

1.1

1.1

1.3

0.7

0.7

0.7

0.7

1 2 4 8 16

1 2 4 8 16 32

1 2 4 8 16 32 64

1 2 4 8 16 32 64 128256

#Tokens (eval)

#Tokens (eval)

#Tokens (eval)

#Tokens (eval)

- Figure 11. Reconstruction performance of different encoder configurations on ImageNet val 50K benchmark. A larger number of lowerdimensional tokens is more friendly for reconstruction tasks.

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

1 2 4 8 16 32 64

#Token

1.0

2.0

4.0

8.2

16.6

33.9

69.0

rFIDImageNet50K

- w/o REPA, CFG=1.0

- w/o REPA, CFG=2.0

- w/ REPA, CFG=1.0

- w/ REPA, CFG=2.0

- Figure 12. Ablation on the use of REPA (with d64×64 concept tokens, DiT-L/2 decoder, see qualitative results in Fig. 16). REPA improves the information density in preceding tokens.

We can see that as the model scales up, the reconstructed images with fewer tokens become more and more realistic and appealing.

- Fig. 15 shows the reconstruction of the same SEMANTICIST tokenizer with different CFG guidance scales at inference time (CFG=1.0 indicates not applying CFG). It can be seen that the guidance scale has a very strong correlation with the aesthetics of generated images.
- Fig. 16 presents qualitative results with or without the usage of REPA [63]. It is clear that the usage of REPA did not visually improve the final reconstruction by much, yet with fewer tokens, the model with REPA demonstrates more faithful semantic details with the original image.
- Fig. 17 demonstrates the reconstruction results of more randomly sampled images, and Fig. 18 illustrates more intermediate results of auto-regressive image generation.

We also compared the reconstruction performance of different concept token dimensions. We fix the product between the number of tokens and the dimension per token to be 4096, and investigate 256-dimensional (d256×16), 128-dimensional (d128×32), 64-dimensional (d64×64), and 16-dimensional (d16×256) tokens. As shown in Fig. 11, all configurations can learn ordered representations, with higher-dimensional ones containing more information per token. However, lower-dimensional tokens are more friendly for reconstruction tasks as they achieve better rFID.

##### D.5. Qualitative Results

- In Fig. 13, reconstruction results from using different numbers of token dimensions are presented. As the dimension for one token becomes large, more semantic content can be encoded into it, thus allowing SEMANTICIST to generate faithful reconstructions of the original image.
- In Fig. 14, the reconstructed results for different scaled

DiT decoders are presented. These models are trained with the same dimension for the tokens that are 16-dimensional.

[Figure 30]

###### Figure 13. Qualitative results of different token dimensions. Higher-dimensional tokens encode more information, and lower-dimensional tokens achieve clearer semantic decoupling and better reconstruction.

[Figure 31]

###### Figure 14. Qualitative results of different DiT decoder scales (DiT-B/2, DiT-L/2, and DiT-XL/2) with d16×256 tokens. The quality of images generated with fewer tokens improves consistently as the decoder scales up.

[Figure 32]

Figure 15. Qualitative results of different CFG guidance scales for DiT decoder, which clearly controls image aesthetics.

[Figure 33]

Figure 16. Qualitative results on effects of REPA (with d64×64 concept tokens). Instead of improving final reconstruction much, the benefit of REPA is mainly attributed to more faithful semantics in intermediate results.

[Figure 34]

###### Figure 17. More reconstruction results of SEMANTICIST autoencoder (with d16×256 concepts tokens and DiT-XL/2 decoder).

[Figure 35]

###### Figure 18. More visualization of intermediate results of auto-regressive image generation.

