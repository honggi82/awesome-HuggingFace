# arXiv:2406.06525v1[cs.CV]10Jun2024

## Autoregressive Model Beats Diffusion: Llama for Scalable Image Generation

Peize Sun1 Yi Jiang2† Shoufa Chen1 Shilong Zhang1 Bingyue Peng2 Ping Luo1∗ Zehuan Yuan2∗

1The University of Hong Kong 2ByteDance Codes and models: https://github.com/FoundationVision/LlamaGen

[Figure 1]

Figure 1: Image generation with vanilla autoregressive models. We show samples from our class-conditional image (top row) and text-conditional image (bottom row) generation models.

### Abstract

We introduce LlamaGen, a new family of image generation models that apply original “next-token prediction” paradigm of large language models to visual generation domain. It is an affirmative answer to whether vanilla autoregressive models, e.g., Llama, without inductive biases on visual signals can achieve state-of-the-art image generation performance if scaling properly. We reexamine design spaces of image tokenizers, scalability properties of image generation models, and their training data quality. The outcome of this exploration consists of: (1) An image tokenizer with downsample ratio of 16, reconstruction quality of 0.94 rFID and codebook usage of 97% on ImageNet benchmark. (2) A series of class-conditional image generation models ranging from 111M to 3.1B parameters, achieving 2.18 FID on ImageNet 256×256 benchmarks, outperforming the popular diffusion models such as LDM, DiT. (3) A text-conditional image generation model with 775M parameters, from two-stage training on LAION-COCO and high aesthetics quality images, demonstrating competitive performance of visual quality and text alignment. (4) We verify the effectiveness of LLM serving frameworks in optimizing the inference speed of image generation models and achieve 326% - 414% speedup. We release all models and codes to facilitate open-source community of visual generation and multimodal foundation models.

∗: Corresponding authors, †: project lead

### 1 Introduction

Built upon autoregressive models, large language models (LLMs) [Vaswani et al. 2017; Devlin et al. 2018; Radford et al. 2018; Raffel et al. 2020; Radford et al. 2019; Brown et al. 2020; Zhang et al. 2022] generate the text by predicting the next token in a sequence. This “next-token prediction” paradigm presents unprecedented capabilities in solving language tasks in a human-like conversational manner [Ouyang et al. 2022; OpenAI 2022, 2023b; Google 2023; Anthropic 2023; Workshop et al.

- 2022; Touvron et al. 2023a,b; Bai et al. 2023a; Yang et al. 2023; Team 2023; Bi et al. 2024] and incredible scalability [Kaplan et al. 2020; Henighan et al. 2020; Hoffmann et al. 2022; Wei et al. 2022; Alabdulmohsin et al. 2022; Chowdhery et al. 2023; Anil et al. 2023], demonstrating a promising path toward general-purpose artificial intelligence models.

Witnessed the scalability of autoregressive models on large language models, pioneering works attempt to explore autoregressive models in image generation, for example, VQVAE [Van Den Oord et al. 2017; Razavi et al. 2019], VQGAN [Esser et al. 2021; Lee et al. 2022], DALL-E [Ramesh et al.

- 2021], Parti [Yu et al. 2021, 2022]. They introduce image tokenizers to convert continuous images to discrete tokens, and apply autoregressive models to generate image tokens in the way of next-token prediction. They demonstrate strong performance among their contemporaries [Brock et al. 2018; Ho et al. 2020; Dhariwal & Nichol 2021] in the year before 2022. However, their open-source communities are not well developed, which largely limits their further improvements.

At the same period, another image generation method, diffusion models [Song & Ermon 2019; Ho et al. 2020; Song et al. 2020; Dhariwal & Nichol 2021; Nichol et al. 2021; Lu et al. 2022a; Ho

- et al. 2022a; Ho & Salimans 2022; Rombach et al. 2022; Ramesh et al. 2022; Saharia et al. 2022; Rombach et al. 2022] develop rapidly. Along with their open-source communities, they dominate the field of visual generation up to today. However, diffusion models share distinct paradigms with autoregressive language models, which poses a huge challenge to building a unified model between language and vision.

In this work, we are committed to pushing the envelope of autoregressive models on image generation further: continuing its research methodology and contributing to open-source community. Reviewing the literature on image generation in the year before 2024, we identify three keys to existing advanced models [Peebles & Xie 2023; Podell et al. 2023; Xue et al. 2023; Chen et al. 2023b,c; Betker et al. 2023; Li et al. 2024; Esser et al. 2024]: 1) well-designed image compressors, 2) scalable image generation models and 3) high-quality training data. Motivated by this, we reexamine the designs of image tokenizers (image compressors for autoregressive models), the scalability properties of image generation models, and the effects of training data.

Towards a potential unified model between language and vision, our design is reducing the inductive biases on visual signals and adopting the same architecture as LLM. This belongs to a different research philosophy with recent works [Chang et al. 2022; Yu et al. 2023b; Tian et al. 2024] that modify the architectures under the guidance of vision-oriented designs. For example, MaskGIT [Chang et al.

- 2022], MAGVIT [Yu et al. 2023a,b] adopt the masked image modeling strategy, VAR [Tian et al. 2024] uses hierarchical multi-scale property. Although they have succeeded in achieving leading image generation performance, and even better than diffusion models, it is still not clear whether the original language model architectures are capable of this. Instead, our work reveals that vanilla autoregressive models that apply the exactly same “next-token prediction” as language models are also able to achieve state-of-the-art image generation performance. As a bonus, we can leverage the techniques [Dao et al. 2022; Rasley et al. 2020; Shoeybi et al. 2019; Zhao et al. 2023; Kwon et al. 2023; Chen et al. 2023a; Dettmers 2022] developed in LLM community to optimize the training recipes and inference speeds of our models. In summary, our contributions to the community include:

- 1. Image tokenizer: An image tokenizer with downsample ratio of 16, achieves reconstruction quality of 0.94 rFID and codebook usage of 97% on ImageNet benchmark. With the downsample ratio of 8, our tokenizer is competitive or even better than continuous VAE [Rombach et al. 2022; Podell et al. 2023; OpenAI 2023a] used in diffusion models. This shows that discrete representation in image tokenizers is no longer the bottleneck of the image reconstruction.
- 2. Scalable image generation model: A series of class-conditional image generation models, ranging from 111M to 3.1B parameters, are developed based on Llama architecture [Touvron

- et al. 2023a,b]. The largest model realizes 2.18 FID on ImageNet 256×256 benchmarks, outperforming the popular diffusion models such as LDM [Rombach et al. 2022], DiT [Peebles & Xie 2023]. This shows that vanilla autoregressive models without inductive biases on visual signals can serve as the basis of image generation systems.
- 3. Hiqh-quality training data: A text-conditional image generation model with 775M parameters, is firstly trained on a 50M subset of LAION-COCO [LAION 2022] and then fine-tuned on 10M internal high aesthetics quality images. It demonstrates competitive performance of visual quality and text alignment.
- 4. Optimized inference speed: We adopt vLLM [Kwon et al. 2023], one of the most popular LLM serving frameworks, to optimize the inference speed of our image generation models, and remarkable 326% - 414% speedup is achieved.

We release all models and codes to facilitate the open-source community of visual generation and multimodal foundation models. It is worth noticing that our released models are still behind state-ofthe-art visual generation models based on diffusion models [Alpha-VLLM 2024; Esser et al. 2024; Brooks et al. 2024]. When more training data and computation resources are available in the future, large-scale AR-based visual generation models, e.g., above 7B parameters, will be explored.

### 2 Autoregressive Models for Image Generation

#### 2.1 Overview

Firstly, image pixels x ∈ RH×W×3 are quantized into q ∈ Qh×w discrete tokens by the image tokenizer [Van Den Oord et al. 2017; Esser et al. 2021; Yu et al. 2021], where h=H/p, w=W/p, p is downsample ratio of the image tokenizer, q(i,j) is indices of the image codebook. Then, these image tokens are reshaped to a sequence of h·w tokens in raster scan ordering and used to train Transformer [Vaswani et al. 2017]-based autoregressive models.

During image generation, image tokens (q1,q2,...,qh·w) are generated by autoregressive models [Radford et al. 2018, 2019; Brown et al. 2020; Touvron et al. 2023a] in the way of next-token

prediction ht=1·w p(qt | q<t,c), where c is class label embedding or text embedding. Finally, these image tokens are converted to image pixels by the image tokenizer decoder.

#### 2.2 Image Tokenizer

Quantized-Autoencoder architecture. We use the same architecture as VQGAN [Esser et al. 2021], encoder-quantizer-decoder. The encoder and the decoder are ConvNet with downsample ratio p. The quantizer contains a codebook Z ∈ RK×C with K learnable vectors. The encoder projects image pixels x to the feature map f. The quantization process maps each vector f(i,j) in the feature map to the code index q(i,j) of its nearest vector z(i,j) in the codebook. During decoding, the code index q(i,j) is remapped to the feature vector z(i,j) and the decoder converts these feature vectors back to the image pixels xˆ.

The codebook has critical effects on image tokenization performance. Following [Yu et al. 2021], we use ℓ2-normalization to codebook vectors, low codebook vector dimension C, and large codebook size K. These designs significantly improve reconstruction quality and codebook usage. More details will be discussed in experiments.

Training losses. Since quantization is a non-differentiable operation, a straight-through gradient estimator [Bengio et al. 2013] is used to preserve the gradient from the decoder to the encoder z = sg[z − f] + f, sg[·] is stop-gradient operation. For codebook learning, LVQ = ∥sg[f] − z∥22 + β∥f −sg[z]∥22, where the second term is commitment loss [Van Den Oord et al. 2017] to force feature vectors extracted from the encoder to be close to codebook vectors, β is commitment loss weight. For simplicity, we don’t add entropy loss [Yu et al. 2023a; Chang et al. 2022] in codebook learning.

For image reconstruction training, LAE = ℓ2(x,xˆ)+LP(x,xˆ)+λGLG(ˆx), where ℓ2 is a reconstruction loss on image pixels, LP(·) is a perceptual loss from LPIPS [Zhang et al. 2018], LG(·) is an adversarial loss from a PatchGAN [Isola et al. 2017] discriminator trained at the same time with the image tokenizer, and λG is adversarial loss weight.

#### Model Parameters Layers Hidden Size Heads

LlamaGen-B 111M 12 768 12 LlamaGen-L 343M 24 1024 16 LlamaGen-XL 775M 36 1280 20 LlamaGen-XXL 1.4B 48 1536 24 LlamaGen-3B 3.1B 24 3200 32

- Table 1: Model sizes and architecture configurations of LlamaGen. The configurations are following previous works [Radford et al. 2019; Touvron et al. 2023a; OpenLM-Research 2023].

#### 2.3 Image Generation by Autoregressive Models

Llama architecture. Our model architecture is largely based on Llama [Touvron et al. 2023a,b], applying pre-normalization using RMSNorm [Zhang & Sennrich 2019], SwiGLU activation function [Shazeer 2020], and rotary positional embeddings [Su et al. 2024]. Specifically, we use 2D RoPE in at each layer of our model, following the implementation of [Lu et al. 2023; Fang et al. 2023]. We do not use the technique of AdaLN [Peebles & Xie 2023] to keep our structure the same as LLM.

Class-conditional image generation. The class embedding is indexed from a set of learnable embeddings [Peebles & Xie 2023; Esser et al. 2021] and is used as the prefilling token embedding. Starting from this token embedding, the model generates the sequence of image tokens by next-token prediction way, and stops at the location of the pre-defined maximum length.

Text-conditional image generation. To integrate the text condition into autoregressive models, we use FLAN-T5 XL [Chung et al. 2024] as the text encoder, the encoded text feature is projected by an additional MLP [Chen et al. 2023b,c] and is used as prefilling token embedding in autoregressive models. We note that this design is not an ultimate design for multimodal foundation models, where a unified vocabulary is established between language and vision [Lu et al. 2023; Team et al. 2023]. We leave it for future research.

Classifier-free guidance. Developed in the diffusion model community, classifier-free guidance [Ho & Salimans 2022] is well-known for its improving visual quality and text-image alignment. We adopt it in our models. During training, the conditional is randomly dropped and is replaced by a null unconditional embedding [Peebles & Xie 2023; Chen et al. 2023b]. In inference, for each token, its logit ℓg is formed by ℓg = ℓu + s(ℓc − ℓu), where ℓc is conditional logit, ℓu is unconditional logit, and s is scale of the classifier-free guidance.

It is worth noting that all design choices discussed so far are largely inspired by previous works, for example, image tokenizer is borrowed from [Rombach et al. 2022; Yu et al. 2021], image generation is from [Peebles & Xie 2023; Chen et al. 2023b; Esser et al. 2021]. A large portion of these techniques are well studied in diffusion models but little in AR models. Our work adapts these advanced designs collectively to AR-based visual generation models.

#### 2.4 Scale Up

Our model architecture is almost the same as Llama, which allows us to seamlessly adopt optimization techniques [Zhang & Sennrich 2019; Shazeer 2020; Su et al. 2024] and training recipes [Dao et al. 2022; Rasley et al. 2020; Shoeybi et al. 2019] in LLM community. As shown in Table 1, we scale the model size up to 3.1B parameters in this work. All models are implemented with PyTorch 2 [Ansel et al. 2024] and trained on 80GB A100 GPUs. For training the models with parameters below 1.4B, we directly use DDP, otherwise, we adopt PyTorch FSDP [Zhao et al. 2023] to optimize GPU memory usage.

#### 2.5 Serving

Autoregressive models have always suffered from its low inference speed. With the rapid development of large language models, advanced inference techniques [Kwon et al. 2023; Chen et al. 2023a; Dettmers 2022] are proposed in the LLM community to optimize the inference speed.

Similar to training, inference techniques developed in the LLM community can also be adopted to optimize our models. We verify the effectiveness of vLLM [Kwon et al. 2023], one of the most popular LLM serving frameworks, on our image generation methods. As shown in Table 7, 326% 414% speedup is achieved compared to the baseline setting.

### 3 Experiments

#### 3.1 Image Tokenizer

Training setup. The training is on ImageNet [Deng et al. 2009] train set, using the resolution of 256×256 and random crop data augmentation. The image tokenizer model size is 72M and 70M when the downsample ratio is 16 and 8, respectively. All models are trained with the same settings: constant learning rate of 10−4, AdamW optimizer with β1 = 0.9, β2 = 0.95, weight decay= 0.05, batch size of 128 and training epochs of 40. For the training losses, commitment loss weight is 0.25 and adversarial loss weight is 0.5. The adversarial loss is enabled after 20k training iterations.

Evaluation metrics. We use the popular ImageNet benchmark under the image resolution of 256 × 256. The image reconstruction quality is measured by r-FID, reconstruction-FID on 256×256 ImageNet 50k validation set. The codebook usage is calculated as the percentage of used codes in the queue of size 65536 over the whole codebook size. We also report PSNR and SSIM as the metrics of reconstruction quality, following SDXL [Podell et al. 2023].

dim rFID↓ PSNR↑ SSIM↑ usage↑ 256 9.21 18.32 0.575 0.29%

size rFID↓ PSNR↑ SSIM↑ usage↑

4096 3.02 19.99 0.643 100.0% 8192 2.91 20.41 0.654 75.0% 16384 2.19 20.79 0.675 97.0% 32768 2.26 20.59 0.663 85.0%

32 3.22 19.98 0.646 20.9% 8 2.19 20.79 0.675 97.0% 4 9.88 19.39 0.593 82.0%

(a) Codebook vector dimension. Lower vector dimension (from 256 to 8) improves both reconstruction quality and codebook usage significantly.

(b) Codebook size. Larger codebook size (from 4096 to 16384) benefits to the overall performance of image tokenizers.

- Table 2: Ablation studies on codebook designs in image tokenizers.. The evaluations are on 256×256 ImageNet 50k validation set. The default setting is codebook vector dimension is 8, codebook size is 16384, downsample ratio is 16.

ratio img size tokens size rFID↓ PSNR↑ SSIM↑ usage↑ 256 256 (16×16) 2.19 20.79 0.675 97.0%

16 384 576 (24×24) 0.94 21.94 0.726 97.0% 512 1024 (32×32) 0.70 23.03 0.772 97.0% 256 1024 (32×32) 0.59 24.45 0.813 97.6%

8 384 2304 (48×48) 0.37 25.63 0.852 97.6% 512 4096 (64×64) 0.39 26.98 0.888 97.6%

- Table 3: Number of tokens to represent the image. The number of tokens depends on downsample ratio and input image size. The reconstructed image is always resized to 256×256 when evaluating on ImageNet 50k validation set. The default setting is codebook vector dimension is 8, codebook size is 16384.

Effect of image codebook designs. As shown in Table 2, when the codebook vector dimension is reduced from 256 to 32 to 8, much better reconstruction quality and codebook usage are consistently achieved. For codebook size, a larger size from 4096 to 16384 benefits the overall performance. These observations are consistent with previous works [Yu et al. 2021, 2023b].

Effect of number of tokens to represent the image. Table 3 studies the effect of image token number on image reconstruction quality. Using the same image tokenizer, for example, downsample ratio as 16, representing an image with only 256 tokens (16×16) is not sufficient for good reconstruction quality, and increasing the number of tokens to 576 (24×24) could largely improve the image quality from 2.43 to 0.99 rFID.

ImageNet COCO

ratio method dim size

rFID↓ PSNR↑ SSIM↑ rFID↓ PSNR↑ SSIM↑

VQGAN 256 1024 8.30 19.51 0.614 16.95 19.08 0.613 VQGAN 256 16384 4.99 20.00 0.629 12.29 19.57 0.630 MaskGIT 256 1024 2.28 - - - - -

16

Ours 8 16384 2.19 20.79 0.675 8.11 20.42 0.678

VQGANoim. 4 256 1.44 22.63 0.737 6.58 22.289 0.744 VQGANoim. 4 16384 1.19 23.38 0.762 5.89 23.08 0.771 ViT-VQGAN 32 8192 1.28 - - - - -

8

Ours 8 16384 0.59 24.45 0.813 4.19 24.20 0.822

SD-VAEukn. 4 - 0.74 25.68 0.820 4.45 25.41 0.831 SDXL-VAEukn. 4 - 0.68 26.04 0.834 4.07 25.76 0.845 OAI-Decoderukn. 4 - 0.81 24.43 0.786 4.59 24.19 0.800

8

- Table 4: Comparisons with other image tokenizers. The evaluations are on 256×256 ImageNet 50k validation set and COCO 5k val2017 set. All models are trained on ImageNet except “oim.” is on OpenImage, “ukn.” is unknown training data.

Comparisons with other image tokenizers. We compare with other image tokenizers, including VQGAN [Esser et al. 2021], MaskGIT [Chang et al. 2022], ViT-VQGAN [Yu et al. 2021]. As shown in Table 4, our tokenizer outperforms previous image tokenizers. We also evaluate our tokenizer on COCO val2017 [Lin et al. 2014] of 256 × 256 image resolution to verify the image reconstruction quality, since COCO images contain more complex scenes. The comparison results are consistent with those in ImageNet validation set. This shows our tokenizer is a generalizable image tokenizer for both object-centric and scene-centric images.

Importantly, our tokenizer is competitive to continuous latent space representation, such as SD VAE [Rombach et al. 2022], SDXL VAE [Podell et al. 2023], and Consistency Decoder from OpenAI [OpenAI 2023a], which are widely used in diffusion models. This shows that discrete representation in the image tokenizer is no longer the bottleneck of the image reconstruction.

#### 3.2 Class-conditional Image Generation

Training setup. Our benchmark is the popular 256 × 256 ImageNet. All models are trained with the similar settings: base learning rate of 10−4 per 256 batch size, AdamW optimizer with β1 = 0.9, β2 = 0.95, weight decay = 0.05, gradient clipping of 1.0. The dropout is always 0.1 for input token embedding, attention module and FFN module. The class condition embedding dropout for classifier-free guidance is 0.1.

Precomputing image codes. To accelerate the model training, we use the image tokenizer to precompute image codes before training. To achieve the similar effect of random crop data augmentation, we extract image codes of ten crops of the original image. During training, we randomly select one copy code from the ten augmentations.

Evaluation metrics. We use Fréchet inception distance (FID) [Heusel et al. 2017] as the main metric. We also report Inception Score (IS) [Salimans et al. 2016], sFID [Nash et al. 2021] and Precision/Recall [Kynkäänniemi et al. 2019] as secondary metrics. All evaluations are implemented using ADM’s TensorFlow scripts [Dhariwal & Nichol 2021] for fair comparisons.

Effect of image tokens. Although increasing the image tokens brings better image reconstruction quality, it is not strongly correlated to image generation quality. As shown in Table 5, when the model parameter is smaller than 1B, 256 (16×16) tokens bring better image generation performance than 576 (24×24). This shows the synergistic effect of scaling up model parameters and token numbers. Nevertheless, fewer image tokens would limit the image generation performance, for example, 256 (16×16) tokens limit the FID at 3.06 FID, while 576 (24×24) could further improve the FID to a lower value.

image token model FID↓ IS↑ Precision↑ Recall↑

B 8.69 124.43 0.78 0.46 L 4.21 200.00 0.82 0.50 XL 3.39 227.08 0.81 0.54 XXL 3.09 253.60 0.82 0.52 3B 3.06 279.71 0.84 0.53

image size: 256×256 tokens: 256 (16×16) rFID: 2.19

B 12.89 92.44 0.73 0.48 L 5.01 167.31 0.78 0.52 XL 3.42 202.93 0.79 0.56 XXL 2.89 236.21 0.80 0.56 3B 2.61 251.90 0.80 0.56

image size: 384×384 tokens: 576 (24×24) rFID: 0.94

- Table 5: The effect of image tokens on image generation. The generated image is always resized to 256×256 when evaluating on ImageNet benchmark. We compare all models after training 50 epochs. The inference setting is cfg = 1.75, top-k = 0 (all), top-p = 1.0, temperature = 1.0 for all experiments.

[Figure 2]

[Figure 3]

(a) without classifier-free guidance (b) with classifier-free guidance

- Figure 2: Scaling model size. We show FID of 256×256 ImageNet benchmark over training epochs. Scaling model size brings consistent improvement on FID during the whole training process. More detailed evaluation metrics are in Appendix.

[Figure 4]

(a) classifier-free guidance (b) top-k sampling

[Figure 5]

- Figure 3: The effect of sampling configuration. We show FID and Inception Score of 256×256 ImageNet benchmark over different sampling configurations. The model is LlamaGen-L, and the default setting is cfg = 2.0, top-k = 0 (all), top-p = 1.0, temperature = 1.0.

Effect of model size. We train our models across five model sizes (B, L, XL, XXL, 3B) and evaluate their performance with and without classifier-free guidance. Figure 2 illustrates how FID changes as both the model sizes and the training epochs increase. Notable improvements in FID are observed when scaling the model from LlamaGen-B to LlamaGen-XXL. Further scaling to 3B yields only marginal improvements. A plausible explanation for this phenomenon could be the limitation in dataset size: ImageNet [Deng et al. 2009] comprises approximately only 1 million images, expanding the dataset or using stronger data augmentation could potentially lead to further improvements.

Type Model #Para. FID↓ IS↑ Precision↑ Recall↑

BigGAN [Brock et al. 2018] 112M 6.95 224.5 0.89 0.38 GigaGAN [Kang et al. 2023] 569M 3.45 225.5 0.84 0.61 StyleGan-XL [Sauer et al. 2022] 166M 2.30 265.1 0.78 0.53

GAN

ADM [Dhariwal & Nichol 2021] 554M 10.94 101.0 0.69 0.63

CDM [Ho et al. 2022b] − 4.88 158.7 − − LDM-4 [Rombach et al. 2022] 400M 3.60 247.7 − − DiT-XL/2 [Peebles & Xie 2023] 675M 2.27 278.2 0.83 0.57

Diffusion

MaskGIT [Chang et al. 2022] 227M 6.18 182.1 0.80 0.51

Mask.

MaskGIT-re [Chang et al. 2022] 227M 4.02 355.6 − −

VQGAN [Esser et al. 2021] 227M 18.65 80.4 0.78 0.26

VQGAN [Esser et al. 2021] 1.4B 15.78 74.3 − − VQGAN-re [Esser et al. 2021] 1.4B 5.20 280.3 − − ViT-VQGAN [Yu et al. 2021] 1.7B 4.17 175.1 − − ViT-VQGAN-re [Yu et al. 2021] 1.7B 3.04 227.4 − − RQTran. [Lee et al. 2022] 3.8B 7.55 134.0 − − RQTran.-re [Lee et al. 2022] 3.8B 3.80 323.7 − −

AR

LlamaGen-B (cfg=2.00) 111M 5.46 193.61 0.83 0.45 LlamaGen-L (cfg=2.00) 343M 3.07 256.06 0.83 0.52 LlamaGen-XL (cfg=1.75) 775M 2.62 244.08 0.80 0.57 LlamaGen-XXL (cfg=1.75) 1.4B 2.34 253.90 0.80 0.59 LlamaGen-3B (cfg=1.65) 3.1B 2.18 263.33 0.81 0.58

AR

- LlamaGen-3B (cfg=1.75) 3.1B 2.32 280.10 0.82 0.56

- LlamaGen-3B (cfg=2.00) 3.1B 2.81 311.59 0.84 0.54

- Table 6: Model comparisons on class-conditional ImageNet 256×256 benchmark. Metrics include Fréchet inception distance (FID), inception score (IS), precision and recall. “↓” or “↑” indicate lower or higher values are better. “-re” means using rejection sampling. “cfg” means using classifier-free guidance. More detailed results are in Appendix.

Effect of classifier-free guidance (CFG). First, as shown in Figure 2, using classifier-free guidance can significantly enhance the visual quality across all model sizes. Moreover, Figure 3a illustrates that the model achieves optimal FID at CFG = 2.0 and further increasing CFG would deteriorate FID, which is consistent with previous findings [Dhariwal & Nichol 2021]. Additionally, the increment in CFG results in a trade-off between diversity and fidelity, as evidenced by increased precision and decreased recall, demonstrated in Table 10.

Effect of top-k sampling. As shown in Figure 3b, a small top-k value is not beneficial for FID and IS. Increasing top-k continuously improves FID but decreases IS, which trades off fidelity for diversity. We observe a similar trend when changing the parameter of top-p and temperature in sampling. Since FID is our main metric, we use maximum value as the default top-k value, which is the whole codebook size.

Comparisons with other image generation methods. In Table 6, we compare with popular image generation models, including GAN [Brock et al. 2018; Kang et al. 2023; Sauer et al. 2022], Diffusion models [Dhariwal & Nichol 2021; Ho et al. 2022b; Rombach et al. 2022; Peebles & Xie 2023], and masked-prediction models [Chang et al. 2022]. Our models exhibit competitive performance in all metrics of FID, IS, Precision and Recall. Notably, our 3B model outperforms the popular diffusion models LDM [Rombach et al. 2022], DiT [Peebles & Xie 2023]. This shows that vanilla autoregressive models can serve as the basis of advanced image generation systems.

When comparing with autoregressive models [Esser et al. 2021; Yu et al. 2021; Lee et al. 2022], our model outperforms all previous models at different levels of model parameters. This benefits from better designs of image tokenizers and better scalability of image generation models. We hope our simple and effective implementation will serve as a solid baseline and help facilitate future research in autoregressive models for image generations.

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

- Stage I
- Stage II

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

|a big purple bus parked in a parking spot|
|---|

|A kitchen that is in the process of having the floors done|
|---|

|A furry, black bear standing in a rocky, weedy, area in the wild.|
|---|

|A cutting board topped with bread, meat and vegetables.|
|---|

- Figure 4: Visualization of two-stage training of text-conditional image generation models. Comparisons of generated images by models after stage I training and stage II training. The text prompts are from COCOPrompts.

#### 3.3 Text-conditional Image Generation

Training setup. We adopt a two-stage training strategy. In stage I, the model is trained on a 50M subset of LAION-COCO [LAION 2022] with the image resolution 256×256. In Stage II, the model is fine-tuned on 10M internal high aesthetic quality images with the image resolution 512×512. Examples of training data are shown in the Appendix. The maximum length of text token embedding is set to 120, and left padding is used to enable batch processing. The text condition embedding dropout for classifier-free guidance is 0.1. All models are trained with similar settings: model parameters of 775M, base learning rate of 10−4 per 256 batch size, AdamW optimizer with β1 = 0.9, β2 = 0.95, decay = 0.05, gradient clipping of 1.0.

Precomputing image codes and text embeddings. We use pre-trained FLAN-T5 XL [Chung et al. 2024] to precompute text embedding of the image captions. For image code, we only extract image codes of the original image center crop in text-conditional models training.

Fine-tune image tokenizer. Before two-stage training for text-conditional image generation models, we first fine-tune the image tokenizer on the joint of 50M LAION-COCO and 10M internal high aesthetic quality data.

Visualizations. In Figure 4, we select text prompts from COCOPrompts [Lin et al. 2014] to generate images using models after stage I training and stage II training. After stage I training, the model captures the text-image alignment, while its ability to represent image details is not clear. Stage II training improves the visual aesthetic quality by a significant margin. We explain this improvement comes from two aspects: high aesthetic quality images shift the domain, and high image resolution brings better visual details. We notice that further increasing the image resolution to 1024×1024 could bring better visual quality, and we leave it for future research.

More visualizations on PartiPrompts [Yu et al. 2022] are in Appendix. PartiPrompts have more longer captions than COCOPrompts, and our model demonstrates competitive performance in text-image alignment for long caption image generation tasks.

Limitation. Due to the training data and model parameters, our text-conditional models have several limitations, such as text rendering errors, counting errors, and common misconceptions. These problems are promising to be mitigated when more training data and computation resources are available in the future.

model parameters baseline (sec) vllm (sec) speed-up ratio

B 111M 7.80 2.39 326% L 343M 13.72 3.48 380%

XL 775M 19.76 4.84 408% XXL 1.4B 26.38 6.36 414%

- Table 7: Optimized inference speed by vLLM serving framework. The inference time is for a batch 16 images (generating 8 images with classifier-free guidance). The image resolution is 384×384 for all models.

#### 3.4 Inference Speed

We verify the effectiveness of vLLM [Kwon et al. 2023] serving framework on our methods. Since our models use the same architecture as Llama, which is already supported by vLLM, we can seamlessly adopt its implementation. As shown in Table 7, we achieve 326% - 414% speedup compared to the baseline setting in the models from 111M to 1.4B parameters. Please note that the baseline setting has already integrated KV-Cache technique. In the 3B model, its head size 100 is not supported by PagedAttention in vLLM.

### 4 Related Work

Visual generation. Generative adversarial network (GAN) [Goodfellow et al. 2014; Brock et al. 2018; Karras et al. 2019; Kang et al. 2023] is the first representative visual generation method in deep learning era. To improve the distribution coverage, several likelihood-based methods are proposed. Diffusion models [Ho et al. 2020; Song & Ermon 2019; Song et al. 2020; Dhariwal & Nichol 2021] view image generation as the reverse diffusion process from noises to images. Masked-prediction models [Chang et al. 2022, 2023; Yu et al. 2023a,b] apply language model BERT-style [Devlin et al. 2018] by learning to predict masked tokens. Instead, autoregressive models [Esser et al. 2021; Ramesh et al. 2021; Yu et al. 2022] leverage GPT-style [Radford et al. 2018] to predict the next token in a sequence. To ease the modeling and improve the generation quality, these methods always introduce the image tokenization process [Kingma & Welling 2013; Van Den Oord et al. 2017] to convert pixel space to semantic space.

Multimodal foundation models. Recently, vision-and-language models [Liu et al. 2024; Zhu

- et al. 2023; Dai et al. 2024; Peng et al. 2023; Zhang et al. 2023; Ma et al. 2024] have achieved versatile visual understanding through visual instruction tuning [Liu et al. 2024; Zhu et al. 2023]. However, unifying the understanding and generation in multimodal models is still in its early stages. Most existing methods [Sun et al. 2023b,a; Dong et al. 2024; Ge et al. 2023] try to collaborate a pre-trained diffusion model with existing models, rather than utilizing a unified next-token prediction paradigm. These methods need sophisticated designs to connect the two parts with distinct training paradigms, which makes scaling up challenging. Pioneering methods [Lu et al. 2022b, 2023; Bai et al. 2023b; Team et al. 2023; Team 2024] attempt to incorporate image generation into LLM using an autoregressive approach and achieve promising results. They do not specifically focus on demonstrating that a plain autoregressive approach can serve as a scalable image generator, which is our main argument in this work.

### 5 Conclusion

In this work, we delve into vanilla autoregressive models for scalable image generation. By reexamining their image tokenizers, image generation models and training data, our class-conditional models outperform the popular diffusion models, and our text-conditional models demonstrate competitive performance of visual quality and text alignment.

### References

Ibrahim M Alabdulmohsin, Behnam Neyshabur, and Xiaohua Zhai. Revisiting neural scaling laws in

language and vision. Advances in Neural Information Processing Systems, 35:22300–22312, 2022. Alpha-VLLM. Large dit. https://github.com/Alpha-VLLM/LLaMA2-Accessory/tree/

main/Large-DiT-ImageNet, 2024.

Rohan Anil, Andrew M Dai, Orhan Firat, Melvin Johnson, Dmitry Lepikhin, Alexandre Passos, Siamak Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng Chen, et al. Palm 2 technical report. arXiv preprint arXiv:2305.10403, 2023.

Jason Ansel, Edward Yang, Horace He, Natalia Gimelshein, Animesh Jain, Michael Voznesensky, Bin Bao, Peter Bell, David Berard, Evgeni Burovski, et al. Pytorch 2: Faster machine learning through dynamic python bytecode transformation and graph compilation. In Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2, pp. 929–947, 2024.

Anthropic. Claude. https://www.anthropic.com/index/introducing-claude, 2023. Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge,

Yu Han, Fei Huang, et al. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023a.

Yutong Bai, Xinyang Geng, Karttikeya Mangalam, Amir Bar, Alan Yuille, Trevor Darrell, Jitendra Malik, and Alexei A Efros. Sequential modeling enables scalable learning for large vision models. arXiv preprint arXiv:2312.00785, 2023b.

Yoshua Bengio, Nicholas Léonard, and Aaron Courville. Estimating or propagating gradients through stochastic neurons for conditional computation. arXiv preprint arXiv:1308.3432, 2013.

James Betker, Gabriel Goh, Li Jing, Tim Brooks, Jianfeng Wang, Linjie Li, Long Ouyang, Juntang Zhuang, Joyce Lee, Yufei Guo, et al. Improving image generation with better captions. Computer Science. https://cdn. openai. com/papers/dall-e-3. pdf, 2(3):8, 2023.

Xiao Bi, Deli Chen, Guanting Chen, Shanhuang Chen, Damai Dai, Chengqi Deng, Honghui Ding, Kai Dong, Qiushi Du, Zhe Fu, et al. Deepseek llm: Scaling open-source language models with longtermism. arXiv preprint arXiv:2401.02954, 2024.

Andrew Brock, Jeff Donahue, and Karen Simonyan. Large scale gan training for high fidelity natural image synthesis. arXiv preprint arXiv:1809.11096, 2018.

Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, Clarence Ng, Ricky Wang, and Aditya Ramesh. Video generation models as world simulators. OpenAI, 2024. URL https://openai.com/research/ video-generation-models-as-world-simulators.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.

Huiwen Chang, Han Zhang, Lu Jiang, Ce Liu, and William T Freeman. Maskgit: Masked generative image transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 11315–11325, 2022.

Huiwen Chang, Han Zhang, Jarred Barber, AJ Maschinot, Jose Lezama, Lu Jiang, Ming-Hsuan Yang, Kevin Murphy, William T Freeman, Michael Rubinstein, et al. Muse: Text-to-image generation via masked generative transformers. arXiv preprint arXiv:2301.00704, 2023.

Charlie Chen, Sebastian Borgeaud, Geoffrey Irving, Jean-Baptiste Lespiau, Laurent Sifre, and John Jumper. Accelerating large language model decoding with speculative sampling. arXiv preprint arXiv:2302.01318, 2023a.

Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Yue Wu, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, et al. Pixart: Fast training of diffusion transformer for photorealistic text-to-image synthesis. arXiv preprint arXiv:2310.00426, 2023b.

Shoufa Chen, Mengmeng Xu, Jiawei Ren, Yuren Cong, Sen He, Yanping Xie, Animesh Sinha, Ping Luo, Tao Xiang, and Juan-Manuel Perez-Rua. Gentron: Delving deep into diffusion transformers for image and video generation. arXiv preprint arXiv:2312.04557, 2023c.

Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. Palm: Scaling language modeling with pathways. Journal of Machine Learning Research, 24(240):1–113, 2023.

Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, et al. Scaling instruction-finetuned language models. Journal of Machine Learning Research, 25(70):1–53, 2024.

Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale N Fung, and Steven Hoi. Instructblip: Towards general-purpose visionlanguage models with instruction tuning. Advances in Neural Information Processing Systems, 36, 2024.

Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. Flashattention: Fast and memoryefficient exact attention with io-awareness. Advances in Neural Information Processing Systems, 35:16344–16359, 2022.

Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pp. 248–255. Ieee, 2009.

Tim Dettmers. bitsandbytes. https://github.com/TimDettmers/bitsandbytes, 2022. Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep

bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018. Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in neural information processing systems, 34:8780–8794, 2021.

Runpei Dong, Chunrui Han, Yuang Peng, Zekun Qi, Zheng Ge, Jinrong Yang, Liang Zhao, Jianjian Sun, Hongyu Zhou, Haoran Wei, Xiangwen Kong, Xiangyu Zhang, Kaisheng Ma, and Li Yi. DreamLLM: Synergistic multimodal comprehension and creation. In The Twelfth International Conference on Learning Representations, 2024.

Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 12873–12883, 2021.

Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, Dustin Podell, Tim Dockhorn, Zion English, Kyle Lacey, Alex Goodwin, Yannik Marek, and Robin Rombach. Scaling rectified flow transformers for high-resolution image synthesis, 2024.

Yuxin Fang, Quan Sun, Xinggang Wang, Tiejun Huang, Xinlong Wang, and Yue Cao. Eva-02: A visual representation for neon genesis. arXiv preprint arXiv:2303.11331, 2023.

Yuying Ge, Sijie Zhao, Ziyun Zeng, Yixiao Ge, Chen Li, Xintao Wang, and Ying Shan. Making llama see and draw with seed tokenizer. arXiv preprint arXiv:2310.01218, 2023.

Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. Advances in neural information processing systems, 27, 2014.

Google. Bard. https://bard.google.com/, 2023. Tom Henighan, Jared Kaplan, Mor Katz, Mark Chen, Christopher Hesse, Jacob Jackson, Heewoo

Jun, Tom B Brown, Prafulla Dhariwal, Scott Gray, et al. Scaling laws for autoregressive generative modeling. arXiv preprint arXiv:2010.14701, 2020.

Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017.

Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

Jonathan Ho, Chitwan Saharia, William Chan, David J Fleet, Mohammad Norouzi, and Tim Salimans. Cascaded diffusion models for high fidelity image generation. The Journal of Machine Learning

- Research, 23(1):2249–2281, 2022a.

Jonathan Ho, Chitwan Saharia, William Chan, David J Fleet, Mohammad Norouzi, and Tim Salimans. Cascaded diffusion models for high fidelity image generation. The Journal of Machine Learning

- Research, 23(1):2249–2281, 2022b.

Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. Training compute-optimal large language models. arXiv preprint arXiv:2203.15556, 2022.

Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, and Alexei A Efros. Image-to-image translation with conditional adversarial networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 1125–1134, 2017.

Minguk Kang, Jun-Yan Zhu, Richard Zhang, Jaesik Park, Eli Shechtman, Sylvain Paris, and Taesung Park. Scaling up gans for text-to-image synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10124–10134, 2023.

Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361, 2020.

Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 4401–4410, 2019.

Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles, 2023.

Tuomas Kynkäänniemi, Tero Karras, Samuli Laine, Jaakko Lehtinen, and Timo Aila. Improved precision and recall metric for assessing generative models. Advances in neural information processing systems, 32, 2019.

LAION. Laion-coco 600m. https://laion.ai/blog/laion-coco, 2022. Doyup Lee, Chiheon Kim, Saehoon Kim, Minsu Cho, and Wook-Shin Han. Autoregressive image

generation using residual quantization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 11523–11532, 2022.

Daiqing Li, Aleks Kamko, Ehsan Akhgari, Ali Sabet, Linmiao Xu, and Suhail Doshi. Playground v2. 5: Three insights towards enhancing aesthetic quality in text-to-image generation. arXiv preprint arXiv:2402.17245, 2024.

Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pretraining for unified vision-language understanding and generation. In International conference on machine learning, pp. 12888–12900. PMLR, 2022.

Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In Computer Vision– ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13, pp. 740–755. Springer, 2014.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36, 2024.

Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps. Advances in Neural Information Processing Systems, 35:5775–5787, 2022a.

Jiasen Lu, Christopher Clark, Rowan Zellers, Roozbeh Mottaghi, and Aniruddha Kembhavi. Unifiedio: A unified model for vision, language, and multi-modal tasks. arXiv preprint arXiv:2206.08916, 2022b.

Jiasen Lu, Christopher Clark, Sangho Lee, Zichen Zhang, Savya Khosla, Ryan Marten, Derek Hoiem, and Aniruddha Kembhavi. Unified-io 2: Scaling autoregressive multimodal models with vision, language, audio, and action. arXiv preprint arXiv:2312.17172, 2023.

Chuofan Ma, Yi Jiang, Jiannan Wu, Zehuan Yuan, and Xiaojuan Qi. Groma: Localized visual tokenization for grounding multimodal large language models. arXiv preprint arXiv:2404.13013, 2024.

Charlie Nash, Jacob Menick, Sander Dieleman, and Peter W Battaglia. Generating images with sparse representations. arXiv preprint arXiv:2103.03841, 2021.

Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. arXiv preprint arXiv:2112.10741, 2021.

OpenAI. Chatgpt. https://openai.com/blog/chatgpt, 2022. OpenAI. Consistency decoder. https://github.com/openai/consistencydecoder, 2023a. OpenAI. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023b. OpenLM-Research. Openllama 3b. https://huggingface.co/openlm-research/open_

##### llama_3b, 2023.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 35: 27730–27744, 2022.

William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 4195–4205, 2023.

Zhiliang Peng, Wenhui Wang, Li Dong, Yaru Hao, Shaohan Huang, Shuming Ma, and Furu Wei. Kosmos-2: Grounding multimodal large language models to the world. arXiv preprint arXiv:2306.14824, 2023.

Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023.

Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever, et al. Improving language understanding by generative pre-training. article, 2018.

Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. The Journal of Machine Learning Research, 21(1):5485–5551, 2020.

Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In International Conference on Machine Learning, pp. 8821–8831. PMLR, 2021.

Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical textconditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 1(2):3, 2022.

Jeff Rasley, Samyam Rajbhandari, Olatunji Ruwase, and Yuxiong He. Deepspeed: System optimizations enable training deep learning models with over 100 billion parameters. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pp. 3505–3506, 2020.

Ali Razavi, Aaron Van den Oord, and Oriol Vinyals. Generating diverse high-fidelity images with vq-vae-2. Advances in neural information processing systems, 32, 2019.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10684–10695, 2022.

Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in Neural Information Processing Systems, 35:36479–36494, 2022.

Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen. Improved techniques for training gans. Advances in neural information processing systems, 29, 2016.

Axel Sauer, Katja Schwarz, and Andreas Geiger. Stylegan-xl: Scaling stylegan to large diverse

datasets. In ACM SIGGRAPH 2022 conference proceedings, pp. 1–10, 2022. Noam Shazeer. Glu variants improve transformer. arXiv preprint arXiv:2002.05202, 2020. Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, and Bryan Catan-

zaro. Megatron-lm: Training multi-billion parameter language models using model parallelism. arXiv preprint arXiv:1909.08053, 2019.

Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020.

Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. Advances in neural information processing systems, 32, 2019.

Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024.

Quan Sun, Qiying Yu, Yufeng Cui, Fan Zhang, Xiaosong Zhang, Yueze Wang, Hongcheng Gao, Jingjing Liu, Tiejun Huang, and Xinlong Wang. Generative pretraining in multimodality. arXiv

- preprint arXiv:2307.05222, 2023a.

Quan Sun, Qiying Yu, Yufeng Cui, Fan Zhang, Xiaosong Zhang, Yueze Wang, Hongcheng Gao, Jingjing Liu, Tiejun Huang, and Xinlong Wang. Generative pretraining in multimodality. arXiv

- preprint arXiv:2307.05222, 2023b.

Chameleon Team. Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818, 2024.

Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.

InternLM Team. Internlm: A multilingual language model with progressively enhanced capabilities, 2023.

Keyu Tian, Yi Jiang, Zehuan Yuan, Bingyue Peng, and Liwei Wang. Visual autoregressive modeling: Scalable image generation via next-scale prediction. arXiv preprint arXiv:2404.02905, 2024.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023a.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023b.

Aaron Van Den Oord, Oriol Vinyals, et al. Neural discrete representation learning. Advances in neural information processing systems, 30, 2017.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, et al. Emergent abilities of large language models. arXiv preprint arXiv:2206.07682, 2022.

BigScience Workshop, Teven Le Scao, Angela Fan, Christopher Akiki, Ellie Pavlick, Suzana Ili´c, Daniel Hesslow, Roman Castagné, Alexandra Sasha Luccioni, François Yvon, et al. Bloom: A 176b-parameter open-access multilingual language model. arXiv preprint arXiv:2211.05100, 2022.

Zeyue Xue, Guanglu Song, Qiushan Guo, Boxiao Liu, Zhuofan Zong, Yu Liu, and Ping Luo. Raphael: Text-to-image generation via large mixture of diffusion paths. arXiv preprint arXiv:2305.18295, 2023.

Aiyuan Yang, Bin Xiao, Bingning Wang, Borong Zhang, Ce Bian, Chao Yin, Chenxu Lv, Da Pan, Dian Wang, Dong Yan, et al. Baichuan 2: Open large-scale language models. arXiv preprint arXiv:2309.10305, 2023.

Jiahui Yu, Xin Li, Jing Yu Koh, Han Zhang, Ruoming Pang, James Qin, Alexander Ku, Yuanzhong Xu, Jason Baldridge, and Yonghui Wu. Vector-quantized image modeling with improved vqgan. arXiv preprint arXiv:2110.04627, 2021.

Jiahui Yu, Yuanzhong Xu, Jing Yu Koh, Thang Luong, Gunjan Baid, Zirui Wang, Vijay Vasudevan, Alexander Ku, Yinfei Yang, Burcu Karagol Ayan, et al. Scaling autoregressive models for contentrich text-to-image generation. arXiv preprint arXiv:2206.10789, 2(3):5, 2022.

Lijun Yu, Yong Cheng, Kihyuk Sohn, José Lezama, Han Zhang, Huiwen Chang, Alexander G Hauptmann, Ming-Hsuan Yang, Yuan Hao, Irfan Essa, et al. Magvit: Masked generative video transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10459–10469, 2023a.

Lijun Yu, José Lezama, Nitesh B Gundavarapu, Luca Versari, Kihyuk Sohn, David Minnen, Yong Cheng, Agrim Gupta, Xiuye Gu, Alexander G Hauptmann, et al. Language model beats diffusion– tokenizer is key to visual generation. arXiv preprint arXiv:2310.05737, 2023b.

Biao Zhang and Rico Sennrich. Root mean square layer normalization. Advances in Neural Information Processing Systems, 32, 2019.

Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 586–595, 2018.

Shilong Zhang, Peize Sun, Shoufa Chen, Min Xiao, Wenqi Shao, Wenwei Zhang, Kai Chen, and Ping Luo. Gpt4roi: Instruction tuning large language model on region-of-interest. arXiv preprint arXiv:2307.03601, 2023.

Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, et al. Opt: Open pre-trained transformer language models. arXiv preprint arXiv:2205.01068, 2022.

Yanli Zhao, Andrew Gu, Rohan Varma, Liang Luo, Chien-Chin Huang, Min Xu, Less Wright, Hamid Shojanazeri, Myle Ott, Sam Shleifer, et al. Pytorch fsdp: experiences on scaling fully sharded data parallel. arXiv preprint arXiv:2304.11277, 2023.

Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models, 2023.

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

|An abstract painting on a pillow with pink, yellow and red flowers.|
|---|

|The dining room is decorated with elegant decor.|
|---|

|The card is attached to an external PCI.|
|---|

|The new tab in Powerpoint is highlighted.|
|---|

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

|A large building with columns and a clock tower.|
|---|

|An assortment of party decorations with owls and other items.|
|---|

|an image of the coordinate of two circles.|
|---|

|Two lions are laying under a tree in the wild.|
|---|

- Figure 5: Examples of stage I training data: 50M subset of LAION-COCO. The short caption is its original caption (generated from BLIP [Li et al. 2022]).

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

|A large, cartoon-like painting of a smiling Mickey Mouse. Mickey is wearing a red shirt and is holding a pair of white gloves. The painting is displayed on a wall, and the Mickey Mouse character appears to be the main focus of the artwork. There are no geometric patterns or overlays in the image.|
|---|

|A cozy bedroom with a large bed situated in the center of the room. The bed is covered with a white comforter and a fur blanket, adding warmth and comfort to the space. A fireplace can be seen in the room, providing additional warmth and ambiance. The bedroom also features a large window, allowing natural light to fill thee room and offering a beautiful view of the snowy landscape outside. There are several candles placed around the room, adding a touch of elegance and creating a serene atmosphere.|
|---|

|A beautiful blue flower with a white center, surrounded by a few other blue flowers. The blue flowers are adorned with water droplets, giving them a fresh and vibrant appearance. The scene appears to be set in a forest or a natural environment, with the flowers standing out against the backdrop. The combination of the blue flowers and the water droplets creates a visually appealing and serene atmosphere.|
|---|

|A cartoon drawing of a smiling lion, standing on a rock in a grassy field. The lion has a playful expression, and its mouth is open, possibly indicating that it is laughing or making a sound. The lion's mane is prominent, adding to its majestic appearance. The scene captures the lion’s joyful and carefree demeanor in a natural environment.|
|---|

- Figure 6: Examples of stage II training data: 10M internal high aesthetic quality images. The long caption is generated from LLaVA.

### A Examples of Image-Text Pair Data

- Training stage I: 50M subset of LAION-COCO [LAION 2022]. The original dataset has 600M image-text pair. We filter these images by valid image URL, aesthetic score, watermark score, CLIP image-text similarity score and image size. The remaining images are about 50M. Some examples are shown in Figure 5.
- Training stage II: 10M internal high aesthetic quality images. Each image is provided a long caption by LLaVA [Liu et al. 2024] using the prompt of “Describe this image in as much detail as possible”. Some examples are shown in Figure 6. We notice that the first sentence of the long caption is always a summary description of its image, so we use it as the short caption to augment the training of text-conditional image generation models.

### B More Results on ImageNet Benchmark

We provide more detailed performance on ImageNet 256×256 benchmark in Table 8 9 10. The generated image is always resized to 256×256 when evaluating.

Model #Para. epochs cfg FID↓ IS↑ sFID↓ Pre.↑ Rec.↑ B 111M 50 no 31.352 39.576 8.749 0.568 0.614

- B 111M 50 1.50 11.984 95.400 7.335 0.738 0.517

- B 111M 50 1.75 8.690 124.435 7.165 0.789 0.469

- B 111M 50 2.00 7.390 153.974 7.250 0.832 0.417

- B 111M 50 2.25 7.220 178.281 7.489 0.861 0.384 B 111M 50 2.50 7.824 197.511 7.857 0.882 0.349 B 111M 300 no 26.262 48.072 9.216 0.593 0.616

- B 111M 300 1.50 8.738 120.602 7.668 0.751 0.535

- B 111M 300 1.75 6.116 159.123 7.364 0.799 0.492

- B 111M 300 2.00 5.464 193.613 7.503 0.839 0.457

- B 111M 300 2.25 5.641 220.720 7.668 0.863 0.411 B 111M 300 2.50 6.390 246.565 8.041 0.883 0.382

L 343M 50 no 21.812 59.179 8.772 0.616 0.640 L 343M 50 1.50 5.781 153.792 7.096 0.774 0.555

- L 343M 50 1.75 4.218 200.001 7.015 0.824 0.509

- L 343M 50 2.00 4.317 242.112 7.077 0.859 0.468

L 343M 300 no 13.452 82.289 8.324 0.656 0.638 L 343M 300 1.50 4.079 198.504 8.157 0.800 0.552

- L 343M 300 1.75 3.805 248.280 8.487 0.833 0.515

- L 343M 300 2.00 4.407 288.170 8.871 0.858 0.481

XL 775M 50 no 19.417 66.196 8.911 0.610 0.665 XL 775M 50 1.50 4.808 172.170 7.298 0.767 0.585

- XL 775M 50 1.75 3.391 227.081 7.022 0.812 0.542

- XL 775M 50 2.00 3.642 268.779 7.244 0.846 0.502

XXL 1.4B 50 no 16.822 74.888 9.285 0.628 0.660 XXL 1.4B 50 1.50 3.844 195.527 7.496 0.781 0.577

- XXL 1.4B 50 1.75 3.094 253.609 7.305 0.825 0.529

- XXL 1.4B 50 2.00 3.644 296.521 7.410 0.857 0.511

3B 3.1B 50 no 13.581 87.902 7.781 0.648 0.666 3B 3.1B 50 1.50 3.050 222.330 6.489 0.801 0.575

- 3B 3.1B 50 1.75 3.063 279.716 6.686 0.843 0.538

- 3B 3.1B 50 2.00 4.212 325.150 7.027 0.869 0.492

validation data 1.684 231.811 3.692 0.752 0.671

- Table 8: Detailed performance on class-conditional ImageNet 256×256 benchmark. The generated image is 256×256. All experiments use the sampling configuration of top-k = 0 (all), top-p = 1.0, temperature = 1.0.

... ... ...

B 111M 50 no 41.025 30.788 9.825 0.523 0.605 B 111M 50 1.50 18.276 69.337 7.557 0.677 0.534 B 111M 50 1.75 12.899 92.447 6.900 0.738 0.487 B 111M 50 2.00 10.029 116.372 6.562 0.787 0.443 B 111M 50 2.25 8.674 136.621 6.428 0.818 0.413 B 111M 50 2.50 8.309 154.719 6.599 0.843 0.376 B 111M 50 2.75 8.391 168.629 6.708 0.860 0.345

B 111M 100 no 33.442 37.528 9.872 0.536 0.609

- B 111M 100 1.50 15.629 77.247 7.632 0.698 0.529

- B 111M 100 1.75 10.676 104.581 6.960 0.754 0.490

- B 111M 100 2.00 8.298 128.941 6.671 0.795 0.452

- B 111M 100 2.25 7.256 152.502 6.510 0.827 0.416 B 111M 100 2.50 7.151 172.677 6.517 0.850 0.390 B 111M 200 no 32.105 37.993 10.144 0.559 0.618

- B 111M 200 1.50 12.206 90.783 7.531 0.716 0.534

- B 111M 200 1.75 8.535 118.399 7.024 0.766 0.503

- B 111M 200 2.00 6.951 146.077 6.784 0.808 0.459

- B 111M 200 2.25 6.542 167.825 6.695 0.833 0.428 B 111M 200 2.50 6.632 188.157 6.811 0.853 0.393 B 111M 300 no 32.196 39.877 11.838 0.570 0.611

- B 111M 300 1.50 12.012 95.553 8.897 0.725 0.528

- B 111M 300 1.75 8.012 127.957 8.088 0.778 0.498

- B 111M 300 2.00 6.437 157.173 7.487 0.814 0.456

- B 111M 300 2.25 6.092 182.538 7.244 0.845 0.416 B 111M 300 2.50 6.249 203.886 6.981 0.861 0.389 B 111M 300 2.75 6.803 220.708 6.928 0.876 0.357 L 343M 50 no 25.889 48.053 9.612 0.570 0.655

- L 343M 50 1.50 7.905 123.830 7.381 0.732 0.569

- L 343M 50 1.75 5.018 167.310 6.786 0.784 0.524

- L 343M 50 2.00 4.240 206.739 6.483 0.825 0.491

- L 343M 50 2.25 4.589 238.890 6.325 0.850 0.451 L 343M 100 no 24.654 53.166 10.497 0.594 0.645

- L 343M 100 1.50 6.934 138.852 7.910 0.748 0.569

- L 343M 100 1.75 4.321 188.536 7.068 0.802 0.528

- L 343M 100 2.00 3.705 228.305 6.701 0.839 0.490

- L 343M 100 2.25 4.054 263.864 6.407 0.858 0.460 L 343M 200 no 19.742 61.715 7.286 0.601 0.667

- L 343M 200 1.50 4.929 158.546 6.066 0.759 0.588

- L 343M 200 1.75 3.249 209.372 5.927 0.805 0.544

- L 343M 200 2.00 3.220 250.697 5.879 0.841 0.512

- L 343M 200 2.25 3.939 288.217 6.076 0.865 0.479 L 343M 300 no 19.070 64.349 8.668 0.607 0.670

- L 343M 300 1.50 4.743 165.381 6.740 0.758 0.596

- L 343M 300 1.75 3.151 214.152 6.310 0.803 0.552

- L 343M 300 2.00 3.075 256.067 6.088 0.832 0.522

- L 343M 300 2.25 3.620 291.695 6.122 0.854 0.493

validation data 1.684 231.811 3.692 0.752 0.671

- Table 9: Detailed performance on class-conditional ImageNet 256×256 benchmark. The generated image is 384×384 and is resized to 256×256 when evaluating on ImageNet. All experiments use the sampling configuration of top-k = 0 (all), top-p = 1.0, temperature = 1.0.

XL 775M 50 no 19.820 61.363 8.067 0.601 0.669 XL 775M 50 1.50 5.231 154.249 6.284 0.746 0.592 XL 775M 50 1.75 3.420 202.939 6.090 0.796 0.560 XL 775M 50 2.00 3.238 245.680 6.023 0.826 0.529

XL 775M 100 no 18.037 69.879 8.388 0.616 0.665 XL 775M 100 1.50 4.563 173.749 6.591 0.759 0.588

- XL 775M 100 1.75 3.089 225.856 6.157 0.804 0.551

- XL 775M 100 2.00 3.105 267.608 6.001 0.833 0.531

XL 775M 200 no 14.772 80.826 6.840 0.620 0.681 XL 775M 200 1.50 3.388 193.477 5.753 0.771 0.603

- XL 775M 200 1.75 2.617 245.465 5.652 0.811 0.566

- XL 775M 200 2.00 2.859 285.900 5.758 0.840 0.527

XL 775M 300 no 15.549 79.157 7.049 0.616 0.689 XL 775M 300 1.50 3.479 194.448 5.816 0.763 0.606

- XL 775M 300 1.75 2.629 244.085 5.594 0.807 0.579

- XL 775M 300 2.00 2.785 286.875 5.567 0.836 0.542

XXL 1.4B 50 no 17.195 74.123 8.689 0.605 0.681 XXL 1.4B 50 1.50 4.363 178.228 6.818 0.758 0.600

- XXL 1.4B 50 1.75 2.893 236.210 6.263 0.805 0.564

- XXL 1.4B 50 2.00 3.049 285.390 6.053 0.842 0.522

XXL 1.4B 200 no 13.997 86.776 8.178 0.637 0.684 XXL 1.4B 200 1.50 3.137 207.870 6.060 0.774 0.605

- XXL 1.4B 200 1.75 2.331 262.995 5.714 0.816 0.579

- XXL 1.4B 200 2.00 2.678 304.631 5.587 0.840 0.545

XXL 1.4B 300 no 14.648 86.328 8.687 0.628 0.681 XXL 1.4B 300 1.50 3.295 202.586 6.476 0.770 0.626

- XXL 1.4B 300 1.75 2.340 253.906 5.977 0.809 0.596

- XXL 1.4B 300 2.00 2.523 295.374 5.736 0.836 0.559

3B 3.1B 50 no 16.431 72.622 7.217 0.611 0.677 3B 3.1B 50 1.50 3.472 191.979 5.955 0.768 0.600

- 3B 3.1B 50 1.75 2.611 251.903 6.167 0.807 0.568

- 3B 3.1B 50 2.00 3.222 300.887 5.764 0.847 0.523

3B 3.1B 200 no 9.949 108.083 7.088 0.667 0.672 3B 3.1B 200 1.50 2.400 237.683 5.548 0.794 0.600 3B 3.1B 200 1.65 2.264 268.180 5.426 0.817 0.581

- 3B 3.1B 200 1.75 2.381 286.091 5.390 0.828 0.569

- 3B 3.1B 200 2.00 3.011 321.563 5.514 0.851 0.538

3B 3.1B 300 no 9.380 112.877 8.242 0.685 0.668 3B 3.1B 300 1.50 2.388 233.246 6.145 0.798 0.601 3B 3.1B 300 1.60 2.216 251.338 6.002 0.811 0.584 3B 3.1B 300 1.65 2.189 263.334 5.965 0.819 0.581 3B 3.1B 300 1.75 2.329 280.104 5.818 0.828 0.566 3B 3.1B 300 1.80 2.370 287.452 5.825 0.834 0.570 3B 3.1B 300 2.00 2.816 311.597 5.845 0.848 0.544

validation data 1.684 231.811 3.692 0.752 0.671

- Table 10: Detailed performance on class-conditional ImageNet 256×256 benchmark. The generated image is 384×384 and is resized to 256×256 when evaluating on ImageNet. All experiments use the sampling configuration of top-k = 0 (all), top-p = 1.0, temperature = 1.0.

[Figure 26]

- Figure 7: 384×384 LlamaGen-3B samples. Classifier-free guidance scale = 4.0 Class label = "golden retriever" (207)

[Figure 27]

Figure 8: 384×384 LlamaGen-3B samples. Classifier-free guidance scale = 4.0 Class label = "husky " (250)

[Figure 28]

Figure 9: 384×384 LlamaGen-3B samples. Classifier-free guidance scale = 4.0 Class label = "cliff drop-off" (972)

[Figure 29]

Figure 10: 384×384 LlamaGen-3B samples. Classifier-free guidance scale = 4.0 Class label = "coral reef" (973)

[Figure 30]

Figure 11: 384×384 LlamaGen-3B samples. Classifier-free guidance scale = 4.0 Class label = "space shuttle" (812)

[Figure 31]

Figure 12: 384×384 LlamaGen-3B samples. Classifier-free guidance scale = 4.0 Class label = "sport car " (817)

[Figure 32]

[Figure 33]

|A plate on a wooden table full of bread.|
|---|

|A cat resting on an open laptop computer.|
|---|

[Figure 34]

[Figure 35]

|A few bags laying around in a living room.|
|---|

|A large green truck on a city street..|
|---|

[Figure 36]

[Figure 37]

|A box of donuts of different colors and varieties.|
|---|

|Two birds that are sitting in a marsh area.|
|---|

[Figure 38]

[Figure 39]

|A large body of water sitting below a mountain range.|
|---|

|A brown cow laying on top of a lush green field.|
|---|

[Figure 40]

[Figure 41]

|A pickup truck driving through a desert environment.|
|---|

|A bathroom with a blue shower curtain and blue walls..|
|---|

Figure 13: (Stage I) Text-conditional 256×256 image generation on COCOPrompts.

[Figure 42]

[Figure 43]

|a tiger|
|---|

|an illustration of a teapot.|
|---|

[Figure 44]

[Figure 45]

|Golden Gate bridge on the surface of Mars|
|---|

|a corgi wearing a red bowtie and a purple party hat|
|---|

[Figure 46]

[Figure 47]

|a view of the Kremlin with snow falling|
|---|

|the Eiffel Tower in winter|
|---|

[Figure 48]

[Figure 49]

|A photo of an Athenian vase with a painting of pandas playing soccer in the style of Egyptian hieroglyphics.|
|---|

|A bare kitchen has wood cabinets and white appliances|
|---|

[Figure 50]

[Figure 51]

|A close-up high-contrast photo of Sydney Opera House sitting next to Eiffel tower, under a blue night sky of roiling energy, exploding yellow stars, and radiating swirls of blue|
|---|

|A photograph of a portrait of a statue of a pharaoh wearing steampunk glasses, white t-shirt and leather jacket.|
|---|

###### Figure 14: (Stage I) Text-conditional 256×256 image generation on PartiPrompts.

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

|a baby penguin|
|---|

|a chimpanzee|
|---|

|an ostrich|
|---|

|a corgi’s head depicted as an explosion of a nebula|
|---|

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

|a photograph of a squirrel holding an arrow above its head and holding a longbow in its left hand|
|---|

|a racoon detective using a microscope while riding in a train|
|---|

|panda mad scientist mixing sparkling chemicals, high-contrast painting.|
|---|

|a gorilla climbing up the side of the Great Pyramid|
|---|

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

|A photo of a four-leaf clover made of water.|
|---|

|a sunken ship at the bottom of the ocean|
|---|

|a plant at the bottom of a shallow stream|
|---|

|red apples on a tree with green leaves|
|---|

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

|Dogs sitting around a poker table with beer bottles and chips. Their hands are holding cards.|
|---|

|A high resolution photo of a chicken working out in a gym.|
|---|

|Dogs sitting around a poker table|
|---|

|Siberian husky playing the piano.|
|---|

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

|a photograph of the mona lisa drinking coffee as she has her breakfast. her plate has an omelette and croissant|
|---|

|A blue Porsche 356 parked in front of a yellow brick wall|
|---|

|A portrait of a metal statue of a pharaoh wearing steampunk glasses and a leather jacket over a white t-shirt that has a drawing of a space shuttle on it.|
|---|

|A photo of an astronaut riding a horse in the forest. There is a river in front of them with water lilies.|
|---|

###### Figure 15: (Stage II) Text-conditional 512×512 image generation on PartiPrompts.

