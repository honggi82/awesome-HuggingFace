# arXiv:2501.07730v2[cs.CV]2Aug2025

## Democratizing Text-to-Image Masked Generative Models with Compact Text-Aware One-Dimensional Tokens

Dongwon Kim*†1,2 Ju He*1 Qihang Yu*1 Chenglin Yang1 Xiaohui Shen1 Suha Kwak2 Liang-Chieh Chen1 1 ByteDance Seed 2 POSTECH

https://tacju.github.io/projects/maskgen

#### Abstract

Image tokenizers form the foundation of modern text-toimage generative models but are notoriously difficult to train. Furthermore, most existing text-to-image models rely on large-scale, high-quality private datasets, making them challenging to replicate. In this work, we introduce Text-Aware Transformer-based 1-Dimensional Tokenizer (TA-TiTok), an efficient and powerful image tokenizer that can utilize either discrete or continuous 1-dimensional tokens. TA-TiTok uniquely integrates textual information during the tokenizer decoding stage (i.e., de-tokenization), accelerating convergence and enhancing performance. TA-TiTok also benefits from a simplified, yet effective, one-stage training process, eliminating the need for the complex two-stage distillation used in previous 1-dimensional tokenizers. This design allows for seamless scalability to large datasets. Building on this, we introduce a family of text-to-image Masked Generative Models (MaskGen), trained exclusively on open data while achieving comparable performance to models trained on private data. We aim to release both the efficient, strong TA-TiTok tokenizers and the open-data, open-weight MaskGen models to promote broader access and democratize the field of text-to-image masked generative models.

#### 1. Introduction

In recent years, text-to-image generation has seen remarkable progress across various frameworks, including diffusion models [23, 30, 53–55, 58, 69], autoregressive visual models [26, 51, 52, 62, 63], and masked generative models [7, 12, 24]. A crucial component of these models is a robust image tokenizer—either discrete [66] or continuous [36]—which transforms images into tokenized representations. These models then incorporate text conditions with

∗Equal contribution. †Work done as a research intern at ByteDance.

the tokenized representations using methods such as crossattention [54], concatenation [8], or conditioning embeddings [45], ultimately leveraging the tokenizer to generate high-quality images aligned with input text prompts.

Despite these advancements, replicating these models remains challenging due to the substantial resources required. Model sizes have grown from millions to billions of parameters, leading to prohibitively high training and inference costs. Furthermore, many of these models rely on large-scale, high-quality proprietary image-text datasets, which are critical for achieving high-fidelity generation but prevent the research community from fully reproducing results. Given these obstacles, a pivotal question arises: Can we develop a text-to-image generative model that is both efficient and effective using only open data, enabling reproducibility?

In this work, we address this question by building on the recent concept of compact one-dimensional tokenizers to facilitate both efficient and high-quality image tokenization. Traditional tokenizers [22, 36, 66] rely on 2D gridbased latent representations, which struggle to handle the inherent redundancies in images, as neighboring patches often display similarities. Recently, Yu et al. [74] introduced the Transformer-based 1-Dimensional Tokenizer (TiTok), which efficiently tokenizes images into compact 1D latent sequences by removing the fixed correspondence between latent representation and 2D image patches (i.e., each 1D token can represent any region in an image, rather than being tied to a specific patch). This approach results in a highly efficient tokenizer, significantly improving sampling speed compared to previous methods [11, 22, 45, 66].

However, extending TiTok to support text-to-image generation presents three main challenges: (1) its reliance on a complex two-stage training pipeline, which limits scalability to larger datasets necessary for diverse text-to-image generation beyond ImageNet [17]; (2) its restriction to a VectorQuantized (VQ) variant, leaving unexplored the potential benefits of a continuous Variational Autoencoder (VAE) representation; and (3) its focus on reconstructing low-level image details, which may lack the high-level semantics needed

[Figure 1]

- Figure 1. Text-to-Image (T2I) Generation Results by MaskGen. MaskGen, powered by the proposed compact text-aware 1D tokenizer TA-TiTok, is an efficient masked generative model that achieves state-of-the-art performance on multiple T2I benchmarks using only open data. The open-data, open-weight MaskGen models are designed to promote broader access and democratize T2I masked generative models.

for effective alignment with textual descriptions.

To address these limitations, we introduce several key innovations. First, we streamline the training process for the 1D tokenizer by developing an efficient one-stage training procedure, removing the need for the complex two-stage pipeline used in the original framework [74]. This improvement enables scalable training of the 1D tokenizer on largescale text-image datasets without multi-stage complexity.

Second, we extend the 1D tokens to continuous VAE representations, which allow for more consistent and accurate image reconstructions than the VQ counterpart. This approach combines the sampling efficiency of 1D tokens (due to the reduced number of tokens) with the improved reconstruction quality afforded by the continuous VAE representation, eliminating the quantization loss seen in VQ.

Third, we incorporate textual information during the detokenization stage to enhance semantic alignment with text prompts. Specifically, by concatenating CLIP [48] embeddings of captions with the tokenized image representations, we enable higher-quality image reconstructions that better retain fine details. This approach powers TA-TiTok, our novel and efficient Text-Aware Transformer-based 1-Dimensional Tokenizer, trained on the large-scale dataset (e.g., DataComp [25]) to capture a broad and diverse range of concepts.

Building upon TA-TiTok, we introduce MaskGen, a fam-

ily of text-to-image Masked Generative models. MaskGen is a versatile framework that supports both discrete and continuous token representations. For images represented by discrete tokens, MaskGen is trained using cross-entropy loss [11], while for images with continuous tokens, it leverages the recent diffusion loss [40]. To encode captions as text conditioning, we utilize CLIP [48], instead of the more resource-intensive T5-XXL encoder [49], used in other recent text-to-image models [12, 23, 55]. Although effective, T5-XXL incurs significantly higher computational and storage costs. CLIP provides a more efficient alternative, making our approach more accessible to research groups with limited compute resources. In terms of architecture, we adopt a straightforward design: we concatenate text conditions with image tokens before feeding them to the Diffusion Transformer [45], applying separate adaptive LayerNorm (adaLN [6, 45]) parameters for text and image modalities, as in MM-DiT [23]. Additionally, we find it beneficial to incorporate the aesthetic score as an additional conditioning signal via adaLN. This allows for more nuanced control over the generated images, beyond text input alone.

To ensure reproducibility, we exclusively use images from publicly available sources, including DataComp [25], LAION [57], and CC12M [13], as well as synthetic data from JourneyDB [61] and DALL-E 3 [9, 21]. Given the

noisiness of web-sourced text-image pairs, we filter images based on aesthetic scores (≥ 5), resolution (aspect ratio < 2 and longer side ≥ 256), and remove any images containing watermarks. Following the approach of DALL-E 3 [9], we further enhance text quality by recaptioning high-aesthetic subsets from DataComp and subsets from LAION (LAIONpop and LAION-art [57]) using the state-of-the-art vision language model Molmo [16].

Notably, despite being trained entirely on publicly available datasets, MaskGen achieves strong performance and efficiency in text-to-image generation. On MJHQ-30K [37], the lighter MaskGen-L (568M) with discrete tokens achieves a generation FID of 7.74, outperforming Show-o [68] (14.99) with 30.3× faster inference throughput. It also surpasses SD-2.1 [54] (26.96) and PixArt-α [14] (9.85) while requiring only 2% and 21% of their training times, respectively. Furthermore, the larger MaskGen-XL (1.1B) achieves FID scores of 7.51 and 6.53 on MJHQ-30K [37] and overall scores of 0.57 and 0.55 on the GenEval [27] benchmark using discrete and continuous tokens, respectively.

To promote further research on text-to-image masked generative models, we will release the training code and model weights for both TA-TiTok and MaskGen. To our knowledge, MaskGen is the first open-weight, open-data masked generative model for text-to-image synthesis to achieve performance comparable to state-of-the-art models, advancing the democratized access to high-performance masked generative models in this field.

#### 2. Related Work

Image Tokenization. Modern generative image models rely on image tokenization for efficient generation [11, 22, 54, 71]. During training, images are encoded into discrete [66] or continuous [36] tokens, allowing the model to focus on learning semantically meaningful information [54] rather than directly working with pixels [29].

Image tokenization approaches fall into two main paradigms. The first, discrete tokenization [66], maps each token to a codebook entry and is well-suited to autoregressive [22] or masked generative models [11], as it enables techniques directly from language models [10]. Introduced in VQ-VAE [66] and later improved in VQGAN [22] with adversarial loss [28], this approach has been further scaled through advanced codebook management techniques [70, 73, 78].

The second paradigm, continuous tokenization, follows the VAE [36] framework, enabling latent representations drawn from a normal distribution. While less common with masked generative models due to the simpler loss definitions with discrete tokens, continuous tokenization was recently adapted for such models in MAR [40], using a diffusion module to sample tokens from the normal distribution.

Image Generation with Sequence Models. Initially devel-

oped for language tasks, sequence models like BERT [18] and GPT [10] have been effectively adapted for image generation. Early approaches focused on autoregressive pixel generation [15, 29, 44, 65], but recent methods model the joint distribution of image tokens, leading to two main approaches: autoregressive [22] and masked generative models [11].

Autoregressive models predict tokens sequentially, following GPT’s strategy [43, 50, 62, 71, 75], while masked generative models adopt a BERT-like objective, predicting masked tokens simultaneously. This approach gives masked models a substantial edge in sampling speed, as they do not require token-by-token generation [11, 39, 67, 72, 73]. Building on these efficiency benefits, our work develops an open-source masked generative model that leverages 1D tokenization for efficient text-to-image generation.

Text-to-Image Generation. While diffusion models dominate text-to-image generation [8, 14, 42, 45, 46, 54, 55], sequence models have shown strong potential as well [12, 24, 26, 71]. Examples like Muse [12] and Parti [71] demonstrate the success of masked generative and autoregressive approaches for generating high-quality images from text. Recent innovations in diffusion models, such as improved architectures [23, 45], micro-conditioning for finer control [46], and advanced image recaptioning for better text-image alignment [9], offer potential benefits for masked generative models. In our work, we integrate these recent improvements from diffusion models into the masked generative model framework. By incorporating these improvements, we aim to enhance the quality of generation while maintaining their inherent advantages in sampling efficiency.

#### 3. Preliminary

TiTok [74] is a transformer-based, 1-dimensional VectorQuantized (VQ) tokenizer that diverges from traditional 2D grid-based latent space tokenization, instead opting for a compact 1D representation that bypasses 2D spatial structure preservation. Given an input image I ∈ RH×W×3, the tokenization phase of TiTok involves downscaling the image by a factor of f, resulting in patches P ∈ R

H f ×Wf ×D. These patches are concatenated with a set of latent tokens L ∈ RK×D. The combined sequence is then passed through a Vision Transformer (ViT) [20] encoder, Enc, to generate embeddings. Only the embeddings corresponding to the latent tokens, Z1D ∈ RK×D, are retained, forming a compact 1D latent representation. This representation is then quantized through a quantizer Quant by mapping it to the nearest codes in a learnable codebook.

In the de-tokenization phase, TiTok uses a sequence of mask tokens M ∈ R

H f ×Wf ×D, which are concatenated with the quantized codes. The resulting sequence is processed by a ViT decoder, Dec, to reconstruct the image ˆI. Formally, the tokenization and de-tokenization processes in TiTok can

[Figure 2]

- Figure 2. Overview of TA-TiTok (Text-Aware Transformer-based 1-Dimensional Tokenizer). (a) TA-TiTok introduces three key enhancements to TiTok [74]: First, an efficient one-stage training procedure replaces the need for a complex two-stage pipeline. Second, TA-TiTok supports 1D tokens in both discrete (VQ) and continuous (KL) formats. Third, it incorporates textual information (using CLIP’s text encoder) during de-tokenization to improve semantic alignment with text captions. (b) A comparison of reconstruction results shows that TA-TiTok achieves superior reconstruction quality over TiTok. be represented as:

models, predicted tokens are masked randomly in each iteration.

[_ ;Z1D] = Enc([P;L]), (1) [_ ;ˆI] = Dec([Quant(Z1D);M]), (2)

#### 4. Method

where [;] denotes concatenation along the sequence dimension, and _ represents ignored tokens that are not used in subsequent operations.

Masked Generative Models with Discrete Tokens [11, 72] adapt the masked language modeling framework [18] for image generation. During training, a portion of image tokens is masked, and a bidirectional transformer predicts these tokens using the surrounding context. The model employs a classification head to select tokens from a predefined codebook [66] and uses cross-entropy loss for training.

During sampling, the model iteratively predicts tokens for masked positions, retaining high-confidence tokens while re-masking uncertain ones until all positions are filled [11]. The completed sequence of tokens is then de-tokenized into pixel space to form the final image.

Masked Generative Models with Continuous Tokens maintain a conceptual similarity to discrete-token models but operate on continuous tokens, which reduces information loss from quantization. Recently, the diffusion loss [24, 40] was introduced, enabling these models to approximate the distribution of each image token independently. In this framework, Transformers generate a conditioning vector for each masked token, which is then input to a small multi-layer perceptron (MLP) that learns a denoising function [34] conditioned on it. This per-token conditioning and denoising allow the sampling process to directly apply to the probability distribution of each token [40]. Unlike discrete-token

In this section, we first present TA-TiTok, a text-aware transformer-based 1-dimensional tokenizer (Sec. 4.1), followed by MaskGen, a family of text-to-image (T2I) masked generative models built upon TA-TiTok (Sec. 4.2).

##### 4.1. TA-TiTok: Text-aware 1D Tokenizer

TA-TiTok is a novel text-aware image tokenizer that introduces three key enhancements to TiTok [74]. First, we develop an improved one-stage training approach; second, we extend TiTok to support both discrete and continuous tokens; and third, we incorporate textual information during de-tokenization to enhance semantic alignment. Each improvement is detailed below.

Improved One-Stage Training Recipe. The recently proposed class-conditional masked generative model, MaskBit [67], introduced several techniques to enhance VQGAN [22] training, two of which we incorporate in the training of our proposed TA-TiTok. First, MaskBit demonstrated that using ResNet50 [31] for perceptual loss [35] yields richer features than the VGG network [59] used in LPIPS [77], thereby improving tokenizer training. Second, we strengthen the PatchGAN [22] discriminator by replacing traditional average pooling with blur kernels [76] and adding LeCAM regularization [64] during training. Our experiments, consistent with MaskBit’s findings, confirm that these enhancements lead to improved image reconstruction quality through 1D tokens.

Extending TiTok to Support Continuous Tokens. To maximize the efficiency of the TiTok framework in diffusionbased models, we extend its discrete 1D tokens to continuous 1D VAE representations. Rather than using a quantizer Quant to map the 1D latents Z1D to the nearest codebook entries, we model Z1D as a Gaussian distribution and apply KL divergence regularization, resulting in a compact 1D VAE representation. This continuous representation retains the efficiency and structure of the TiTok framework, consistently improving reconstruction quality by avoiding the information loss associated with quantization. Moreover, this KL variant of TiTok integrates seamlessly with diffusion models, serving as a drop-in replacement for standard 2D VAEs [40, 54, 60]. In the supplementary materials (Tab. 13 in the Appendix), we validate this approach using the state-of-the-art image generation model MAR [40] on ImageNet [17], where our modification achieves a significant reduction in training costs and an increase in inference speed, all while maintaining comparable performance. This enhancement thus contributes to both the efficiency and flexibility of diffusion-based generation. For clarity, we refer to both variants collectively as TiTok, using the labels VQ and KL to denote the discrete and continuous versions, respectively, in the following text.

Text-aware De-tokenization. While TiTok effectively utilizes compact 1D tokens to capture richer semantic information than traditional 2D tokenizers, its primary focus remains on reconstructing low-level image details in the detokenization stage. Additionally, previous image tokenizers have largely overlooked the potential of textual information to enhance high-level semantic alignment, even when such information is available. This gap motivates us to introduce TA-TiTok, a text-aware, transformer-based 1D tokenizer designed to improve alignment with textual descriptions.

As shown in Fig. 2(a), the tokenization stage in TA-TiTok mirrors that of TiTok, transforming images into compact 1D tokens, either discrete or continuous. In the de-tokenization stage, however, TA-TiTok incorporates text guidance by using text embeddings generated by a pre-trained visionlanguage model (e.g., CLIP’s text encoder [48]). These text embeddings are projected through a linear layer to align with the channel dimensions of our TA-TiTok’s ViT decoder, resulting in T ∈ RN×D, where N is the number of context tokens predefined by the vision-language model (e.g., 77 for CLIP’s text encoder). This text embedding is then concatenated with the latent tokens Z1D and the mask tokens M before passing through the decoder Dec, yielding the reconstructed image ˆI.

Formally, the de-tokenization phase of the VQ variant of TA-TiTok can be expressed as:

[_ ;_ ;ˆI] = Dec([Quant(Z1D);T;M]). (3) For the KL variant, the de-tokenization phase follows a simi-

lar formulation but omits the quantizer Quant, as it operates on continuous representations directly.

TA-TiTok Design Discussion. Notably, TA-TiTok incurs minimal additional computational cost compared to TiTok. By extending the de-tokenization sequence length by N (i.e., the total number of tokens becomes N + K, where N = 77 for CLIP’s text tokens and K represents 32, 64, or 128 latent tokens), TA-TiTok still requires fewer computations than typical 2D tokenizers [40], which utilize 256 tokens. This design enables TA-TiTok to retain high efficiency while achieving reconstructions that closely align with text descriptions, effectively mitigating the information loss associated with compact 1D tokenization, as shown in Fig. 2(b).

The design of TA-TiTok incorporates text tokens exclusively into the tokenizer decoder with minimal modifications. To validate this approach, we also experimented with adding text tokens to both the encoder and decoder. The results showed similar performance to the decoder-only approach, suggesting that incorporating textual information during detokenization is sufficient for capturing high-level semantic information. Based on these findings, we adopt the simpler decoder-only design. For additional details, see Tab.10 in the Appendix.

##### 4.2. MaskGen: T2I Masked Generative Model

To fully leverage TA-TiTok’s capabilities, we propose MaskGen, a family of text-to-image (T2I) masked generative models. As illustrated in Fig. 3, MaskGen utilizes TA-TiTok to tokenize images into tokens and a CLIP [48] text encoder to extract both global and pooled text embeddings. Inspired by [23], we concatenate the global text embedding with the image tokens, feeding this combined sequence into multimodal Diffusion Transformer (MM-DiT) blocks [23, 45] for attention operations. To accommodate the distinct properties of text and image embeddings, we apply separate adaptive LayerNorm (adaLN [6, 45]) layers, where the scale and shift parameters are computed based on the pooled text embedding. We note that while a more powerful text encoder, such as T5-XXL [49], could be directly integrated into the pipeline to enhance performance, as demonstrated in studies like [12, 23, 55], we choose CLIP for its computational efficiency and reduced storage demands, making MaskGen more accessible in resource-constrained settings.

We also incorporate aesthetic scores [57] as another condition by projecting them into sinusoidal embeddings and appending them to the pooled text embedding. This feature provides an extra layer of control over the image quality and style during sampling, making the generation process more adaptable and flexible.

MaskGen is a versatile framework that accommodates both discrete and continuous tokens produced by TA-TiTok. For discrete tokens, MaskGen is trained with a cross-entropy loss, as in [11, 12], to predict the correct one-hot code-

[Figure 3]

- Figure 3. Overview of MaskGen. MaskGen is a family of text-to-image masked generative models that supports both discrete (VQ variant) and continuous (KL variant) token representations. For discrete tokens, MaskGen is trained with cross-entropy loss [11], while for continuous tokens, it employs diffusion loss [40]. The architecture is designed by concatenating text conditions with TA-TiTok’s latent tokens (both masked and unmasked) and feeding them into Diffusion Transformer blocks [45], with separate adaptive LayerNorms (adaLN), linear projections, and feedforward networks (FFN) for text and image modalities, following MM-DiT [23]. Additionally, aesthetic scores are incorporated as conditioning signals via adaLN. To encode captions, MaskGen uses the CLIP text encoder [48] instead of the more resource-intensive T5-XXL [49], making it more accessible to research groups with limited computational resources.

model depth width mlp heads #params MaskGen-L 16 1024 4096 16 568M MaskGen-XL 20 1280 5120 16 1.1B

- Table 1. Architecture Configuration of MaskGen. Following prior work, we scale up MM-DiT blocks across two configurations.

book index for masked tokens. In the case of continuous tokens, MaskGen leverages the recently proposed diffusion loss [40], applying a small MLP to directly approximate the distribution of each masked token. This adaptability makes MaskGen capable of handling various token types with ease. Additionally, thanks to the compact 1D token sequence produced by TA-TiTok, MaskGen is highly efficient, reducing training costs and enhancing sampling speed by minimizing token count. Together, these features help MaskGen to democratize access to efficient, high-performance masked generative models for text-to-image generation.

#### 5. Experimental Results

##### 5.1. Implementation Details

TA-TiTok Model Variants. We present three variants of our TA-TiTok, each containing K = 32, 64, or 128 1D latent tokens, following the architecture of TiTok. Both the tokenizer and de-tokenizer utilize a patch size of f = 16. For the VQ variant, the codebook is configured with 8192 entries, where each entry is a vector with 64 channels. For the KL variant, we use a continuous embedding with 16 channels, following MAR [40]. For the encoder (Enc) and decoder (Dec), we find that increasing the size of Dec to ViT-L [20] is beneficial when training on large-scale datasets across all variants. For Enc, we adopt ViT-B for all variants except the KL variant with 128 tokens, where ViT-S is sufficient.

MaskGen Model Variants. We introduce two variants of MaskGen: MaskGen-L (568M parameters) and MaskGenXL (1.1B parameters), with configurations detailed in Tab. 1. For continuous token processing in MaskGen, we incorporate an additional DiffLoss MLP [40], comprising 8 MLP layers with channel sizes aligned to the transformer’s, adding an extra 44M and 69M parameters for MaskGen-L and MaskGen-XL, respectively. To offset the additional training and inference cost introduced by the DiffLoss MLP, we use 128 tokens for the discrete MaskGen variant and 32 tokens for the continuous MaskGen variant.

Training Data. For training TA-TiTok and MaskGen, we utilized various open-source datasets: DataComp-1B [25], CC12M [13], LAION-aesthetic [1], LAION-art [3], LAIONpop [4], JourneyDB [61], and DALLE3-1M [21]. TA-TiTok is trained exclusively with DataComp-1B. MaskGen undergoes a two-stage training process: pre-training for image-text alignment and fine-tuning using aesthetic images. Details regarding the dataset preparation and recaptioning process are provided in the Appendix.

Evaluation Metrics. Our evaluation pipeline closely follows prior works [12, 71]. The images are generated without rejection sampling, and classifier-free guidance [33] is used to enhance generation quality. Unless specified otherwise, MaskGen uses 16 and 32 sampling steps for VQ and KL variants, respectively. For TA-TiTok, we measure reconstruction quality using reconstruction FID (rFID) [32] and inception score (IS) [56] on ImageNet [17] validation set. For MaskGen’s text-to-image generation capabilities, we utilize FID on MJHQ [38] to assess aesthetic quality, and GenEval [27] score to measure the alignment between text prompts and their corresponding generated images.

tokenizer arch training setting #tokens rFID↓ IS↑

TiTok [74] VQ 1-stage [22] 64 5.15 120.5 TiTok VQ our 1-stage 64 2.43 179.3

- Table 2. Ablation on Improved One-Stage Training Recipe. Both models are trained and evaluated on ImageNet.

[Figure 4]

- Figure 4. Visualization of Latent Token Attention Map and Latent Code Swapping. The results are from VQ variant of TATiTok with 32 tokens. Each latent token attends to prominent semantic and swapping the code leads to appearance changes in the corresponding semantic entity that the latent token focuses on.
- 5.2. Optimized Image Tokenization with TA-TiTok

Improved One-Stage Training Recipe. Tab. 2 summarizes the performance gains of our improved one-stage training recipe over the original schemes in [74]. As observed, the adopted one-stage training significantly outperforms the original, achieving an rFID improvement of 2.72.

Latent Token Attention Maps Reveal Semantic Focus. In the first row of Fig. 4, we overlay the attention maps of each latent token onto the image, where attention is computed from the encoder’s final layer. We observe that each latent token focuses on semantically meaningful regions within the image, rather than specific spatial locations or grid cells as in 2D tokenization. For instance, individual latents capture distinct semantic elements such as leaf, the bird’s head, and the bird’s torso. Notably, we find that at least one latent token (e.g., Latent 11) attends to the entire image, and subsequent experiments reveal that it captures the overall visual style.

Semantic Manipulation Through Latent Code Swapping. Fig. 4 also presents reconstruction results when we swap the code for latent tokens. Interestingly, swapping the code leads to changes in the semantic element corresponding to each latent token, with appearance changes ranging from low-level visual features (e.g., color changes) to high-level semantics (e.g., new objects, object removal, pose changes). For instance, swapping the code for the 4th latent token, which focuses on the leaf, leads to the removal of the leaf

|arch<br><br>tokens # c|TiTok rFID↓ IS↑|TA-TiTok rFID↓ IS↑<br><br>|
|---|---|---|
|VQ<br><br>32 64 -<br><br>128 -<br><br>|7.72 98.3 4.25 138.0 2.63 168.1|3.95 (-3.77) 219.6 (+121.3) 2.43 (-1.82) 218.8 (+80.8) 1.53 (-1.10) 222.8 (+54.7)|
|KL<br><br>32 16 64 16<br><br>128 16|2.56 171.7 1.64 198.0 1.02 209.7|1.53 (-1.03) 222.0 (+50.3) 1.47 (-0.17) 220.7 (+22.7) 0.90 (-0.12) 227.7 (+18.0)|

Table 3. Ablation on Text-Aware De-tokenization Design. All models are trained on DataComp with our improved one-stage recipe and evaluated in a zero-shot setting on the ImageNet validation set. Relative improvements for TA-TiTok are highlighted in blue. #: Number of tokens. c: Channels of continuous tokens.

or its transformation into a bird or flower. Swapping for the 28th latent token results to the change in bird’s head in terms of its spices or pose, and for 30th latent token swapping results to color change of bird’s torso. Unlike other latent tokens, swapping the 11th latent token focusing on the overall image leads to the change of overall style of the image. For instance, swapping to code 29 leads to the whole scene being redrawn in a flat, vector-graphic style; swapping to code 593 softens high-frequency details; swapping to code 9 produces a purple-tinted stylistic transformation.

Effect of Text-Aware De-Tokenization. We evaluate the impact of text-aware de-tokenization (TA-TiTok) in Tab. 3. For consistency, all models are trained using our improved one-stage training recipe. Tokenizers are trained on the DataComp dataset [25] and evaluated in a zero-shot setting on the ImageNet validation set [17], where the caption is simply represented as “A photo of class” without any prompt engineering. We compare two architectures—VQ and KL variants—and vary the token count between 32, 64, and 128. As shown in the table, continuous tokens consistently outperform discrete tokens, aligning with findings in [22]. Additionally, TA-TiTok consistently outperforms TiTok (the non-text-aware variant) across all configurations. Notably, the performance gain is most pronounced with a smaller number of tokens (e.g., 32) and with discrete tokens. We attribute this to the text embeddings supplementing high-level semantic information overlooked by latent tokens. Consequently, the improvement is more substantial with fewer tokens and vector-quantized tokens, where latent tokens experience greater information loss.

##### 5.3. Text-to-Image Generation with MaskGen

MJHQ-30K. We report zero-shot text-to-image generation results on MJHQ-30K [38] in Tab. 4. MaskGen-XL (discrete tokens) achieves a significantly better FID than recent autoregressive models like LlamaGen [62] (7.51 vs. 25.59) and Show-o [68] (7.51 vs. 14.99), both using VQ tokenizers. It also offers 18.5× higher inference throughput than Show-o, with MaskGen-L being 30.3× faster, though with a slight performance drop. MaskGen demonstrates impressive resource efficiency, with MaskGen-L completing training in

|tokenizer arch|type generator #params resolution<br><br>|open-data T↓ I↑|FID↓|
|---|---|---|---|
|VQGAN [62] VQ MAGVIT-v2 [68] VQ|AR LlamaGen [62] 775M 512 × 512 AR Show-o [68] 1.3B 256 × 256<br><br>|✗ - -<br><br>✓ - 1.0|25.59 14.99|
|TA-TiTok VQ TA-TiTok VQ|Mask. MaskGen-L (ours) 568M 256 × 256 Mask. MaskGen-XL (ours) 1.1B 256 × 256|✓ 20.0 30.3 ✓ 35.0 18.5|7.74 7.51|

|VAE [54] KL VAE [54] KL VAE [46] KL<br><br>|Diff. Stable-Diffusion-2.1 [54] 860M 768 × 768 Diff. PixArt-α [14] 630M 256 × 256 Diff. SDXL [46] 2.6B 1024 × 1024|✓ 1041.6 -<br><br>✗ 94.1 7.9<br>✗ - -<br>|26.96 9.85 8.76|
|---|---|---|---|
|TA-TiTok KL TA-TiTok KL|Mask. MaskGen-L (ours) 568M + 44M 256 × 256 Mask. MaskGen-XL (ours) 1.1B + 69M 256 × 256|✓ 18.5 11.1<br><br>✓ 30.5 9.1|7.24 6.53|

- Table 4. Zero-Shot Text-to-Image Generation Results on MJHQ-30K. Comparison of MaskGen with state-of-the-art open-weight models. “VQ” denotes discrete tokenizers and “KL” stands for continuous tokenizers. “type” indicates the generative model type, where “AR”, “Diff.” and “Mask.” refer to autoregressive models, diffusion models and masked transformer models, respectively. T: Generator training cost, measured in 8 A100 days using float16 precision. I: Generator inference throughput, measured in samples per second on a single A100 with batch size 64 using float16 precision. We compare inference throughput with methods using the same resolution.

|tokenizer arch<br><br>|generator #params|open-data|S. Obj. T. Obj. Count. Colors Position C. Attri.|Overall↑|
|---|---|---|---|---|
|VQGAN [62] VQ MAGVIT-v2 [68] VQ|LlamaGen [62] 775M Show-o [68] 1.3B|✗ ✓<br><br>|0.71 0.34 0.21 0.58 0.07 0.04 0.95 0.52 0.49 0.82 0.11 0.28|0.32 0.53|
|TA-TiTok VQ TA-TiTok VQ|MaskGen-L (ours) 568M MaskGen-XL (ours) 1.1B|✓ ✓|0.98 0.57 0.46 0.80 0.11 0.25<br>0.99 0.61 0.55 0.81 0.13 0.31<br>|0.53 0.57|

|VAE [54] KL VAE [54] KL VAE [54] KL VAE [46] KL|Stable-Diffusion-1.5 [54] 860M PixArt-α [14] 630M<br>Stable-Diffusion-2.1 [54] 860M SDXL [46] 2.6B<br>|✓ ✗ ✓ ✗|0.97 0.38 0.35 0.76 0.04 0.06 0.96 0.49 0.47 0.79 0.06 0.11<br>0.98 0.51 0.44 0.85 0.07 0.17 0.98 0.74 0.39 0.85 0.15 0.23<br><br><br>|0.43 0.48 0.50 0.55|
|---|---|---|---|---|
|TA-TiTok KL TA-TiTok KL|MaskGen-L (ours) 568M + 44M MaskGen-XL (ours) 1.1B + 69M|✓ ✓|0.99 0.57 0.36 0.80 0.11 0.29<br>0.99 0.58 0.47 0.77 0.13 0.34<br>|0.52 0.55|

- Table 5. Zero-Shot Text-to-Image Generation Results on GenEval. Comparison of MaskGen with state-of-the-art open-weight models. 20.0 8-A100 days and MaskGen-XL in 35.0 8-A100 days.

|tokenizer arch|generator|aesthetic<br><br>|MJHQ-30K FID↓|GenEval Overall↑|
|---|---|---|---|---|
|TiTok<br><br>KL<br><br>TiTok TA-TiTok TA-TiTok|MaskGen-L|✗ ✓ ✗ ✓|9.82 8.50 8.13 7.24|0.47 0.49 0.51 0.52|

For continuous tokens, MaskGen delivers competitive results against recent diffusion models. With just 32 tokens, MaskGen-L outperforms PixArt-α[14] (630M) (7.24 vs. 9.85), offering 1.4× faster inference throughput while using fewer parameters and requiring less than one-fifth of the training resources (18.5 vs. 94.1 8-A100 days). MaskGenXL further improves the FID score to 6.53, surpassing SDXL[46] (6.53 vs. 8.76), a 2.6B-parameter model trained on high-quality private data, despite MaskGen-XL being trained exclusively on open data for only 30.5 8-A100 days. GenEval. Tab. 5 summaries the zero-shot text-to-image generation results on GenEval [27]. Using discrete tokens, MaskGen-L (568M) achieves an overall score of 0.53, significantly outperforming the recent autoregressive model LlamaGen [62] by 0.21 and performing on par with the larger Show-o [68] (1.3B). Moreover, the larger MaskGenXL achieves the highest overall score on the benchmark, with a score of 0.57. This result notably surpasses SDXL [46], a 2.6B-parameter model (2.36× larger than MaskGen-XL) trained on proprietary data. Meanwhile, our MaskGen-XL with continuous tokens also achieves an overall score of 0.55, comparable to recent diffusion models [54], but with much lower training costs and exclusively trained on open data.

Table 6. Ablation of Text-Aware Design and Aesthetic Score Condition. Incorporating aesthetic scores and exploiting TA-TiTok enhance generation quality.

version of MaskGen-L. As shown in Tab. 6, adopting TATiTok significantly improves generation quality (Row 1 vs. Row 3). Additionally, incorporating aesthetic score conditioning further enhances performance (Row 3 vs. Row 4). Moreover, comparing Row 2 with other configurations suggests that while aesthetic score conditioning helps MaskGen better control image fidelity, its impact is less significant than that of the text-aware tokenizer TA-TiTok.

#### 6. Conclusion

We introduced TA-TiTok, a text-aware 1D tokenizer that enhances semantic alignment, and MaskGen, a versatile masked generative model supporting discrete and continuous tokens. With compact 1D tokenization, MaskGen lowers training costs and accelerates sampling, enabling efficient, high-quality generation. Leveraging public data, it broadens access to high-performance generative models. All code and model weights is released to promote future research.

Effect of Text-Aware Design and Aesthetic Score Conditioning. For efficient ablation studies, we use the continuous

#### References

- [1] LAION2B-en-aesthetic. https://huggingface.co/ datasets/laion/laion2B-en-aesthetic, . 6, 2
- [2] LAION-aesthetics predictor V2. https://github.com/ christophschuhmann / improved - aesthetic predictor, . 1
- [3] LAION-art. https://huggingface.co/datasets/ laion/laion-art, . 6, 1, 2
- [4] LAION-pop. https : / / huggingface . co / datasets/laion/laion-pop, . 6, 1, 2
- [5] LAION-5B-WatermarkDetection. https : / / github . com / LAION - AI / LAION - 5B WatermarkDetection, . 1
- [6] Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization. arXiv preprint arXiv:1607.06450, 2016. 2, 5
- [7] Jinbin Bai, Tian Ye, Wei Chow, Enxin Song, Qing-Guo Chen, Xiangtai Li, Zhen Dong, Lei Zhu, and Shuicheng Yan. Meissonic: Revitalizing masked generative transformers for efficient high-resolution text-to-image synthesis. arXiv preprint arXiv:2410.08261, 2024. 1
- [8] Fan Bao, Shen Nie, Kaiwen Xue, Yue Cao, Chongxuan Li, Hang Su, and Jun Zhu. All are worth words: A vit backbone for diffusion models. In CVPR, 2023. 1, 3
- [9] James Betker, Gabriel Goh, Li Jing, Tim Brooks, Jianfeng Wang, Linjie Li, Long Ouyang, Juntang Zhuang, Joyce Lee, Yufei Guo, et al. Improving image generation with better captions. Computer Science. https://cdn. openai. com/papers/dall-e-3. pdf, 2(3):8, 2023. 2, 3, 1
- [10] Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. NeurIPS, 2020. 3
- [11] Huiwen Chang, Han Zhang, Lu Jiang, Ce Liu, and William T Freeman. Maskgit: Masked generative image transformer. In CVPR, 2022. 1, 2, 3, 4, 5, 6
- [12] Huiwen Chang, Han Zhang, Jarred Barber, AJ Maschinot, José Lezama, Lu Jiang, Ming-Hsuan Yang, Kevin Murphy, William T Freeman, Michael Rubinstein, et al. Muse: Textto-image generation via masked generative transformers. In ICML, 2023. 1, 2, 3, 5, 6, 4
- [13] Soravit Changpinyo, Piyush Sharma, Nan Ding, and Radu Soricut. Conceptual 12m: Pushing web-scale image-text pre-training to recognize long-tail visual concepts. In CVPR,

2021. 2, 6

- [14] Junsong Chen, Jincheng YU, Chongjian GE, Lewei Yao, Enze Xie, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, and Zhenguo Li. Pixart-α: Fast training of diffusion transformer for photorealistic text-to-image synthesis. In ICLR,

2024. 3, 8

- [15] Mark Chen, Alec Radford, Rewon Child, Jeffrey Wu, Heewoo Jun, David Luan, and Ilya Sutskever. Generative pretraining from pixels. In ICML, 2020. 3
- [16] Matt Deitke, Christopher Clark, Sangho Lee, Rohun Tripathi, Yue Yang, Jae Sung Park, Mohammadreza Salehi, Niklas Muennighoff, Kyle Lo, Luca Soldaini, et al. Molmo and

- pixmo: Open weights and open data for state-of-the-art multimodal models. arXiv preprint arXiv:2409.17146, 2024. 3, 1, 2
- [17] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In CVPR, 2009. 1, 5, 6, 7, 2, 4
- [18] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In NAACL, 2018. 3, 4
- [19] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. NeurIPS, 2021. 3
- [20] Alexey Dosovitskiy. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020. 3, 6
- [21] Ben Egan, Alex Redden, XWAVE, and SilentAntagonist. Dalle3 1 Million+ High Quality Captions. https://huggingface.co/datasets/ ProGamerGov/synthetic-dataset-1m-dalle3high-quality-captions, 2024. 2, 6, 1
- [22] Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image synthesis. In CVPR,

2021. 1, 3, 4, 7

- [23] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In ICML,

2024. 1, 2, 3, 5, 6

- [24] Lijie Fan, Tianhong Li, Siyang Qin, Yuanzhen Li, Chen Sun, Michael Rubinstein, Deqing Sun, Kaiming He, and Yonglong Tian. Fluid: Scaling autoregressive text-to-image generative models with continuous tokens. arXiv preprint arXiv:2410.13863, 2024. 1, 3, 4
- [25] Samir Yitzhak Gadre, Gabriel Ilharco, Alex Fang, Jonathan Hayase, Georgios Smyrnis, Thao Nguyen, Ryan Marten, Mitchell Wortsman, Dhruba Ghosh, Jieyu Zhang, et al. Datacomp: In search of the next generation of multimodal datasets. NeurIPS, 2023. 2, 6, 7, 1
- [26] Oran Gafni, Adam Polyak, Oron Ashual, Shelly Sheynin, Devi Parikh, and Yaniv Taigman. Make-a-scene: Scene-based text-to-image generation with human priors. In ECCV, 2022. 1, 3
- [27] Dhruba Ghosh, Hannaneh Hajishirzi, and Ludwig Schmidt. Geneval: An object-focused framework for evaluating text-toimage alignment. NeurIPS, 2023. 3, 6, 8, 2
- [28] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. NeurIPS, 2014. 3
- [29] Karol Gregor, Ivo Danihelka, Andriy Mnih, Charles Blundell, and Daan Wierstra. Deep autoregressive networks. In ICML,

2014. 3

- [30] Ju He, Qihang Yu, Qihao Liu, and Liang-Chieh Chen. Flowtok: Flowing seamlessly across text and image tokens. In ICCV, 2025. 1
- [31] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In CVPR, 2016. 4

- [32] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. NeurIPS, 2017. 6
- [33] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022. 6, 1
- [34] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. NeurIPS, 2020. 4
- [35] Justin Johnson, Alexandre Alahi, and Li Fei-Fei. Perceptual losses for real-time style transfer and super-resolution. In ECCV, 2016. 4
- [36] Diederik P Kingma and Max Welling. Auto-encoding variational bayes. In ICLR, 2014. 1, 3, 4
- [37] Daiqing Li, Aleks Kamko, Ali Sabet, Ehsan Akhgari, Linmiao Xu, and Suhail Doshi. Playground v2. 3
- [38] Daiqing Li, Aleks Kamko, Ehsan Akhgari, Ali Sabet, Linmiao Xu, and Suhail Doshi. Playground v2.5: Three insights towards enhancing aesthetic quality in text-to-image generation. arXiv preprint arXiv:2402.17245, 2024. 6, 7, 2
- [39] Tianhong Li, Huiwen Chang, Shlok Mishra, Han Zhang, Dina Katabi, and Dilip Krishnan. Mage: Masked generative encoder to unify representation learning and image synthesis. In CVPR, 2023. 3
- [40] Tianhong Li, Yonglong Tian, He Li, Mingyang Deng, and Kaiming He. Autoregressive image generation without vector quantization. NeurIPS, 2024. 2, 3, 4, 5, 6
- [41] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In ECCV, 2014. 4
- [42] Qihao Liu, Zhanpeng Zeng, Ju He, Qihang Yu, Xiaohui Shen, and Liang-Chieh Chen. Alleviating distortion in image generation via multi-resolution diffusion models. NeurIPS, 2024. 3
- [43] Zhuoyan Luo, Fengyuan Shi, Yixiao Ge, Yujiu Yang, Limin Wang, and Ying Shan. Open-magvit2: An open-source project toward democratizing auto-regressive visual generation. arXiv preprint arXiv:2409.04410, 2024. 3
- [44] Niki Parmar, Ashish Vaswani, Jakob Uszkoreit, Lukasz Kaiser, Noam Shazeer, Alexander Ku, and Dustin Tran. Image transformer. In ICML, 2018. 3
- [45] William Peebles and Saining Xie. Scalable diffusion models with transformers. In ICCV, 2023. 1, 2, 3, 5, 6
- [46] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023. 3, 8
- [47] Ofir Press and Lior Wolf. Using the output embedding to improve language models. arXiv preprint arXiv:1608.05859,

2016. 1

- [48] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, 2021. 2, 5, 6

- [49] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. JMLR, 2020. 2, 5, 6
- [50] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In ICML, 2021. 3
- [51] Sucheng Ren, Qihang Yu, Ju He, Xiaohui Shen, Alan Yuille, and Liang-Chieh Chen. Beyond next-token: Next-x prediction for autoregressive visual generation. In ICCV, 2025. 1
- [52] Sucheng Ren, Qihang Yu, Ju He, Xiaohui Shen, Alan Yuille, and Liang-Chieh Chen. Flowar: Scale-wise autoregressive image generation meets flow matching. In ICML, 2025. 1
- [53] Sucheng Ren, Qihang Yu, Ju He, Alan Yuille, and LiangChieh Chen. Grouping first, attending smartly: Trainingfree acceleration for diffusion transformers. arXiv preprint arXiv:2505.14687, 2025. 1
- [54] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, 2022. 1, 3, 5, 8
- [55] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. NeurIPS, 2022. 1, 2, 3, 5
- [56] Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen. Improved techniques for training gans. NeurIPS, 2016. 6
- [57] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. NeurIPS, 2022. 2, 3, 5
- [58] Inkyu Shin, Chenglin Yang, and Liang-Chieh Chen. Deeply supervised flow-based generative models. In ICCV, 2025. 1
- [59] K Simonyan and A Zisserman. Very deep convolutional networks for large-scale image recognition. In ICLR, 2015. 4
- [60] stabilityai, 2023. 5
- [61] Keqiang Sun, Junting Pan, Yuying Ge, Hao Li, Haodong Duan, Xiaoshi Wu, Renrui Zhang, Aojun Zhou, Zipeng Qin, Yi Wang, et al. Journeydb: A benchmark for generative image understanding. NeurIPS, 2023. 2, 6, 1
- [62] Peize Sun, Yi Jiang, Shoufa Chen, Shilong Zhang, Bingyue Peng, Ping Luo, and Zehuan Yuan. Autoregressive model beats diffusion: Llama for scalable image generation. arXiv preprint arXiv:2406.06525, 2024. 1, 3, 7, 8
- [63] Chameleon Team. Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818, 2024. 1
- [64] Hung-Yu Tseng, Lu Jiang, Ce Liu, Ming-Hsuan Yang, and Weilong Yang. Regularizing generative adversarial networks under limited data. In CVPR, 2021. 4
- [65] Aaron Van den Oord, Nal Kalchbrenner, Lasse Espeholt, Oriol Vinyals, Alex Graves, et al. Conditional image generation with pixelcnn decoders. NeurIPS, 2016. 3

- [66] Aaron Van Den Oord, Oriol Vinyals, et al. Neural discrete representation learning. NeurIPS, 2017. 1, 3, 4
- [67] Mark Weber, Lijun Yu, Qihang Yu, Xueqing Deng, Xiaohui Shen, Daniel Cremers, and Liang-Chieh Chen. Maskbit: Embedding-free image generation via bit tokens. arXiv preprint arXiv:2409.16211, 2024. 3, 4
- [68] Jinheng Xie, Weijia Mao, Zechen Bai, David Junhao Zhang, Weihao Wang, Kevin Qinghong Lin, Yuchao Gu, Zhijie Chen, Zhenheng Yang, and Mike Zheng Shou. Show-o: One single transformer to unify multimodal understanding and generation. arXiv preprint arXiv:2408.12528, 2024. 3, 7, 8
- [69] Chenglin Yang, Celong Liu, Xueqing Deng, Dongwon Kim, Xing Mei, Xiaohui Shen, and Liang-Chieh Chen. 1.58-bit flux. arXiv preprint arXiv:2412.18653, 2024. 1
- [70] Jiahui Yu, Xin Li, Jing Yu Koh, Han Zhang, Ruoming Pang, James Qin, Alexander Ku, Yuanzhong Xu, Jason Baldridge, and Yonghui Wu. Vector-quantized image modeling with improved vqgan. In ICLR, 2022. 3
- [71] Jiahui Yu, Yuanzhong Xu, Jing Yu Koh, Thang Luong, Gunjan Baid, Zirui Wang, Vijay Vasudevan, Alexander Ku, Yinfei Yang, Burcu Karagol Ayan, et al. Scaling autoregressive models for content-rich text-to-image generation. TMLR,

2022. 3, 6

- [72] Lijun Yu, Yong Cheng, Kihyuk Sohn, José Lezama, Han Zhang, Huiwen Chang, Alexander G Hauptmann, MingHsuan Yang, Yuan Hao, Irfan Essa, et al. Magvit: Masked generative video transformer. In CVPR, 2023. 3, 4
- [73] Lijun Yu, José Lezama, Nitesh B Gundavarapu, Luca Versari, Kihyuk Sohn, David Minnen, Yong Cheng, Agrim Gupta, Xiuye Gu, Alexander G Hauptmann, et al. Language model beats diffusion–tokenizer is key to visual generation. In ICLR,

2024. 3

- [74] Qihang Yu, Mark Weber, Xueqing Deng, Xiaohui Shen, Daniel Cremers, and Liang-Chieh Chen. An image is worth 32 tokens for reconstruction and generation. NeurIPS, 2024. 1, 2, 3, 4, 7
- [75] Qihang Yu, Ju He, Xueqing Deng, Xiaohui Shen, and LiangChieh Chen. Randomized autoregressive visual generation. In ICCV, 2025. 3
- [76] Richard Zhang. Making convolutional networks shiftinvariant again. In ICML, 2019. 4
- [77] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, 2018. 4
- [78] Chuanxia Zheng and Andrea Vedaldi. Online clustered codebook. In ICCV, 2023. 3

## Democratizing Text-to-Image Masked Generative Models with Compact Text-Aware One-Dimensional Tokens

### Supplementary Material

#### Appendix

We provide additional information in the supplementary material, as outlined below:

- • Sec. A: Further implementation details, including dataset filtering, recaptioning, and MaskGen training hyperparameters.
- • Sec. B: Ablation studies on the type and place of text guidance in TA-TiTok.
- • Sec. C: Ablation studies on the number of tokens and aesthetic score condition in MaskGen.
- • Sec. D: Comparisons between MaskGen using discrete and continuous tokens.
- • Sec. E: Additional zero-shot text-to-image generation results on COCO validation set.
- • Sec. F: Results demonstrating the performance of our KL variant of TiTok on ImageNet.
- • Sec. G: More qualitative examples for latent code swapping with TA-TiTok.
- • Sec. H: More qualitative examples generated by MaskGen.
- • Sec. I: Discussions on limitations and future work of TATiTok and MaskGen.

#### A. More Implementation Details

Dataset Filtering. To prepare training data, we applied three filtering criteria: resolution, aesthetic, and watermark filtering. Details of applied filtering criteria and the total size of the dataset after filtering are presented in Tab. 7. Resolution filtering was applied to all datasets during the training of both TA-TiTok and MaskGen. This filtering ensured that the longer side of each image exceeded 256 pixels and maintained an aspect ratio below 2. For MaskGen training, we implemented aesthetic filtering using the LAION-aestheticv2 predictor [2] to retain only high-quality images. Images with scores above 5.0 were kept during the pre-training stage, while a stricter threshold of 6.0 was applied during finetuning to ensure even higher quality. Additionally, we employed watermark filtering for MaskGen using the LAIONWatermarkDetector [5], removing images with watermark probability exceeding 0.5 to prevent unwanted watermark artifacts in generated images. Synthetic datasets such as JourneyDB [61] and DALLE3-1M [21] were exempted from these filtering processes as they inherently met our high resolution and quality standards.

Dataset Recaptioning. To improve the text quality of DataComp [25], LAION-art [3], and LAION-pop [4], we utilize state-of-the-art vision-language models, Molmo-7B-D-

0924 [16], to enhance captions based on both the image and its original caption. Specifically, we randomly sample one of four prompts as shown in Fig. 5 to generate updated captions. Since the augmented captions are often significantly longer than the original ones and frequently start with similar patterns (e.g., “The image depicts/displays/showcases/shows/features...”), we apply prefix filtering to remove these repetitive prefixes, preventing information leakage. During training, we further address this by employing a 95:5 ratio, randomly sampling between augmented and original captions to ensure balanced learning, following [9]. A few recaption results are shown in Fig. 6, highlighting how the augmented captions provide richer details and align better with the image content.

Training. We adhere strictly to the hyperparameters used to train TiTok across all TA-TiTok variants. Specifically, TATiTok is trained with a batch size of 1024 for 1 epoch (650k steps) on the filtered DataComp dataset, using a maximum learning rate of 1e-4 and a cosine learning rate schedule. For MaskGen with discrete tokens, we employ a batch size of 4096, leveraging weight tying [47] to stabilize training, with a cosine learning rate schedule and a maximum learning rate of 4 × 10-4. For MaskGen with continuous tokens, to accommodate the diffusion loss, we use a constant learning rate schedule with a maximum rate of 1 × 10-4 and a batch size of 2048. Masked tokens are sampled by randomly selecting the masking rate from [0, 1] on a cosine schedule, following MaskGIT [11], and text conditioning is randomly dropped with a 0.1 probability to enable classifier-free guidance [33]. Tab. 8 provides the complete list of hyper-parameters used for training MaskGen with both discrete and continuous tokenizers.

#### B. Ablation Studies for TA-TiTok

Text Guidance Type in TA-TiTok. In our text-aware detokenization design, we can use either the numerical IDs from the CLIP text tokenizer or the embeddings from the CLIP text encoder. We ablate this design choice in Tab. 9, where the latter option yields a marginal improvement.

Text Guidance Place in TA-TiTok In the design of TATiTok, we only incorporate the text guidance (i.e., the text tokens from CLIP) into the tokenizer decoder to better capture high-level semantics and align with textual descriptions during both reconstruction and generation. In this study, we investigate whether injecting text guidance into both the encoder and decoder of TA-TiTok can further enhance the quality of the encoded latent tokens. This evaluation is

|model|dataset|filtering resolution aesthetic watermark|recaptioning<br><br>|samples|
|---|---|---|---|---|
|TA-TiTok: tokenizer<br><br>|DataComp [25]|✓| |685.8M|
|MaskGen: pre-training|DataComp [25] CC12M [13] LAION-aesthetic [1]|✓ ✓ (5.0) ✓ ✓ ✓ (5.0) ✓ ✓ ✓<br><br>| |219.8M 4.8M 28.3M|
|MaskGen: fine-tuning|DataComp [25] LAION-art [3] LAION-pop [4] DALLE3-1M [21] JourneyDB [61]|✓ ✓ (6.0) ✓ ✓ ✓ ✓ ✓|✓ ✓ ✓|3.6M<br>4.2M<br><br><br>0.4M<br>1.0M 4.1M<br>|

Table 7. Training Data Details. Filtering criteria applied to each publicly available dataset include resolution (aspect ratio < 2 and longer side ≥ 256), aesthetic score (predicted score exceeding the specified value in parentheses), and watermark detection (removal of images predicted to contain watermarks). For datasets with noisy web-crawled captions, Molmo [16] is used for recaptioning. The final column shows the number of text-image pairs remaining after filtering.

- 1. Describe the image in detail while considering the provided caption: '{original_caption}'. Correct any errors and improve the caption, ensuring the final description is in English and within 77 tokens. Return only the corrected caption.
- 2. Analyze the image and the caption '{original_caption}'. Write a detailed and accurate description of the image in English, correcting any mistakes or low-quality aspects of the original caption. Keep the final caption under 77 tokens, and return only the caption.
- 3. Use the caption '{original_caption}' as a reference to create a detailed and improved description of the image in English. Correct any errors and make sure the new caption is concise and within 77 tokens. Return only the revised caption.
- 4. Given the image and the original caption '{original_caption}', describe the image in a detailed and accurate way in English, improving upon the original caption where necessary. Ensure the description is within 77 tokens. Return only the corrected caption.

- Figure 5. Prompts Used for Recaptioning. One of four prompts is used to recaption each image, where {original_caption} is replaced with the original image caption.

conducted on the ImageNet validation set [17] using reconstruction metrics in a zero-shot setting, where the caption is represented as “A photo of class” without any prompt engineering. As shown in Tab. 10, injecting textual guidance into both the encoder and decoder has negligible impact on reconstruction quality. This finding suggests that incorporating text guidance in the decoder alone is sufficient to provide semantic information to the model.

hyper-parmeters discrete continuous optimizer AdamW AdamW

- β1 0.9 0.9
- β2 0.96 0.95 weight decay 0.03 0.02 lr (pre-training) 0.0004 0.0002 lr (fine-tuning) 0.0001 0.0002 lr scheduling cosine constant lr warmup steps 10K 50k batch size 4096 2048 training steps (pre-training) 500K 1000k training steps (fine-tuning) 250K 500k Table 8. Training Hyper-parameters for MaskGen. tokenizer arch #tokens text guidance rFID↓ IS↑

ID 1.62 213.6 Embedding 1.53 222.0

TA-TiTok KL 32

- Table 9. Ablation on Text Guidance Type. Models are trained on DataComp and zero-shot evaulated on ImageNet validation set. ID refers to numerical IDs extracted by CLIP text tokenizer, Embedding denotes text features extracted by CLIP text encoder.

arch

tokens Encoder + Decoder Decoder Only # c rFID↓ IS↑ rFID↓ IS↑ KL

32 16 1.65 218.4 1.53 222.0 64 16 1.39 221.5 1.47 220.7

128 16 0.92 227.1 0.90 227.7

- Table 10. Ablation on Text Guidance Place. In the TA-TiTok design, we ablate on adding the text guidance to both encoder and decoder or just decoder. Adding text guidance to only the decoder results in similar reconstruction performances but enjoys a simpler structure. Models are trained on DataComp and zero-shot evaulated on ImageNet validation set.

#### C. Ablation Studies for MaskGen

Experimental Setup. For efficient ablation studies, we use the discrete version of MaskGen to analyze the impact of token count. Performance is evaluated using the FID metric on MJHQ-30K [38] and the overall GenEval [27] score.

[Figure 5]

- Figure 6. Re-captioning Results. Captions augmented by Molmo [16] offer richer details and improved alignment with image content.

|arch generator|#tokens<br><br>|T↓ I↑|MJHQ-30K FID↓|GenEval Overall↑|
|---|---|---|---|---|
|VQ MaskGen-L|32 64 128|16.0 47.6<br>17.5 40.2 20.0 30.3<br>|9.11 7.85 7.74|0.43 0.50 0.53|

Table 11. Zero-Shot Text-to-Image Generation Results on MJHQ-30K and GenEval with Varying Number of Tokens. MaskGen achieves better generation quality with more tokens but incurs longer training times and slower inference speeds. T: Generator training cost, measured in 8 A100 days using float16 precision. I: Generator inference throughput, measured in samples per second on a single A100 with batch size 64 using float16 precision.

Additionally, we provide visual comparisons to illustrate the effect of aesthetic score conditioning during sampling.

Number of Tokens. Tab. 11 presents an ablation study on the number of tokens used for text-to-image generation with MaskGen. As observed, increasing the token count improves generation quality but comes at the expense of longer training times and slower inference speeds.

Aesthetic Score Conditioning. Fig. 7 visualizes images generated with different aesthetic scores while keeping other hyperparameters and prompts constant. The results indicate a strong correlation between higher aesthetic scores and enhanced dramatic lighting and fine-grained details. For instance, in the third row, a higher aesthetic score yields richer depictions of trees and stars in the night sky, whereas a lower score results in simpler representations. This enables precise control over image generation based on user preferences.

#### D. Comparisons Between MaskGen Using Discrete and Continuous Tokens

Performance Comparisons Between VQ and KL Variants. The KL variant of MaskGen consistently outperforms the VQ variant on MJHQ-30K FID but performs slightly worse on GenEval’s overall score. We hypothesize that the KL variant excels in generating diverse, high-aesthetic images, contributing to improved FID on MJHQ-30K. However, it falls behind on GenEval, which emphasizes object-focused compositional properties such as position, count, and color.

|tokenizer arch<br><br>|generator #params|open-data|FID-30K↓|
|---|---|---|---|
|MAGVIT-v2 [68] VQ|Show-o [68] 1.3B<br><br>|✓|9.24|
|TA-TiTok VQ TA-TiTok VQ|MaskGen-L (ours) 568M MaskGen-XL (ours) 1.1B|✓ ✓|13.62 13.01|

|VAE [54] KL VAE [54] KL VAE [54] KL|LDM [54] 1.4B Stable-Diffusion-1.5 [54] 860M PixArt-α [14] 630M<br><br>|✓ ✓ ✗|12.64 9.62 7.32|
|---|---|---|---|
|TA-TiTok KL TA-TiTok KL|MaskGen-L (ours) 568M + 44M MaskGen-XL (ours) 1.1B + 69M|✓ ✓|9.66 8.98|

- Table 12. Zero-Shot Text-to-Image Generation Results on COCO-30K. Comparison of MaskGen with state-of-the-art openweight models.

tokenizer arch

tokens

# c rFID↓ generator gFID↓ IS↑ T↓ I↑ VAE [40] KL 256 16 0.54 MAR [40] 2.45 275.5 8.0 1.0 TiTok (ours) KL

64 16 1.54

MAR [40]

2.96 246.9 2.1 8.1 128 16 1.31 2.70 252.9 3.2 3.2

- Table 13. Class-conditional ImageNet-1K 256×256 Generation Results Evaluated with ADM [19], using continuous tokens (i.e., KL architecture). #: Number of tokens. c: Channels of continuous tokens. T: Generator training cost, measured in 8 A100 days using float32 precision. I: Generator inference throughput, measured in samples per second on a single A100 with float32 precision.

In contrast, the VQ variant, constrained by a finite codebook, generates less diverse but more compositionally accurate images, leading to higher scores on GenEval. Fig. 9 visually compares generated samples, where the KL variant demonstrates slightly better overall generation quality.

Training and Inference Cost Comparisons Between VQ and KL Variants. The VQ variant of MaskGen benefits from faster training and significantly faster inference, primarily due to inherent differences in the diffusion process used in the KL variant. While the KL variant excels in generating more diverse and higher-aesthetic images, this advantage comes with increased computational demands. To address this gap in training and inference efficiency, we employ 128 tokens for the VQ variant and 32 tokens for the KL variant, effectively controlling the training cost to remain at a comparable level, as shown in Tab. 4 of the main paper.

[Figure 6]

- Figure 7. Generated Images with Varying Aesthetic Score Conditioning. Conditioning on higher aesthetic scores produces generated images with enhanced fine-grained details.

#### E. Zero-Shot Text-to-Image Generation Results on COCO

In Tab. 12, we evaluate zero-shot text-to-image generation on the COCO dataset [41] by randomly sampling 30K imagecaption pairs from the COCO 2014 validation split and reporting the FID, as is standard in the literature. Since the finetuning stage of MaskGen often generates more aesthetically appealing images that deviate from the COCO dataset distribution, we perform the evaluation using MaskGen at the pretraining stage to ensure consistency with the dataset’s characteristics. Notably, MaskGen-L (KL variant with continuous tokens) achieves an FID-30K of 9.66, while MaskGen-XL (KL variant) further improves to 8.98. These results demonstrate that MaskGen achieves performance comparable to other state-of-the-art text-to-image models, highlighting its effectiveness even in the zero-shot setting.

#### F. KL variant of TiTok on ImageNet

We evaluate the KL variant of TiTok as a drop-in replacement for standard 2D VAEs [36, 40] in class-conditional image generation on ImageNet [17]. Results, reported in Tab. 13, are based on the MAR [40] framework with its base model, after 400 epochs using unchanged MAR hyper-parameters. MAR with TiTok (our KL variant) achieves significant training time reductions (3.8× with 64 tokens, 2.7× with 128 tokens) and inference speedups (8.1× with 64 tokens, 3.2× with 128 tokens), thanks to its efficient 1D token design.

Despite the substantial reduction in computational overhead, MAR with TiTok maintains performance comparable to MAR with conventional 2D VAEs using 256 tokens, highlighting TiTok’s potential as an efficient and robust image tokenizer for class-conditional generation.

#### G. Qualitative Examples of Latent Code Swapping

Fig. 8 provides additional visualizations of latent code swapping in VQ variants of TA-TiTok. The results demonstrate that our proposed 1D tokenization encodes images such that each token captures meaningful semantic elements, enabling image manipulation through latent token swapping without even requiring the generator. This semantically rich tokenization explains why 1D tokenization achieves far more efficient encoding than its 2D counterpart while preserving competitive reconstruction fidelity: the encoder dynamically allocates tokens to perceptually informative regions in the image.

#### H. Qualitative Examples of MaskGen

Fig. 10, Fig. 11, Fig. 12, Fig. 13, Fig. 14, Fig. 15, and Fig. 16 showcase additional qualitative examples of text-to-image generation using MaskGen. By utilizing the efficient and compact text-aware tokenizer TA-TiTok, MaskGen demonstrates its ability to produce high-fidelity and diverse images.

#### I. Limitations and Future Work

While MaskGen achieves competitive generation quality and benchmark scores comparable to recent text-to-image models, including those leveraging proprietary training data, we acknowledge several aspects for future exploration.

First, the current KL variant of MaskGen is designed to use 32 tokens. While increasing the token count improves tokenization quality, leading to better-reconstructed samples, it also significantly raises training costs due to longer convergence times. Additionally, scaling up the generator remains a challenge, as the current MaskGen-XL is constrained to 1.1B parameters due to limited computational resources.

Second, the current implementation of MaskGen operates at a resolution of 256 × 256. However, the scalability of its core architectural design—1-dimensional tokenization and masked generation—has been validated in high-resolution implementations like Muse [12].

This work emphasizes establishing a fully open-source, open-data text-to-image masked generative model using compact text-aware 1-dimensional tokenization. Future work will focus on optimizing convergence speed, model scaling up, and enabling high-resolution outputs.

[Figure 7]

###### Figure 8. Visualization of Latent Token Attention Map and Latent Code Swapping. The results are from VQ variant of TA-TiTok with 32 tokens. Each latent token attends to prominent semantic and swapping the code leads to appearance changes in the corresponding semantic entity that the latent token focuses on.

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

KL

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

VQ

‘apple tree, you bow your head to the apple tree in quiet simplicity, receive …’

‘astronaut cat 100 Moon in

‘a beautiful wolf head with a

‘cartoon style, watercolor of

the background 60 fullcolor

surrounding floral design in

cute baby rabbit, huge eyes,

200 …’

detailed drawing style …’

vector style’

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

KL

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

VQ

‘vintage beautiful botanic

‘crazy psychotic darth vader

‘fireflies reflected off of a

‘A black MercedesBenz SL

flowers, graphic design,

with glowing red eyes with a

small forest pond, colorful

Roadster with the headlights

seamless repeating …’

beautiful sunset …’

sunset …’

illuminated parked …’

- Figure 9. Generated Images by MaskGen with Different Tokenizer Types. For each caption, the top row displays images generated using continuous tokens (KL), while the bottom row shows images generated using discrete tokens (VQ). Long prompts are truncated for brevity.

[Figure 24]

[Figure 25]

‘A golden hourglass, half-filled with flowing silver sand,

‘A person standing on the desert, desert waves, half red,

placed on a rich velvet cloth’

half blue, sand, illustration, outdoor’

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

‘A space explorer discovering an alien jungle planet under a purple sky’

[Figure 30]

[Figure 31]

‘A Pikachu with an angry expression and red eyes, with

‘A gorgeous mountain landscape at sunset. Masterful

lightning around it, hyper realistic style’

painting by Rembrandt’

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

‘A space explorer discovering an alien jungle planet under a purple sky’

[Figure 36]

[Figure 37]

‘A window with raindrops trickling down, overlooking a

‘A cloud dragon flying over mountains, its body swirling

blurry city’

with the wind’

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

‘A mountain village built into the cliffs of a canyon, where bridges connect houses carved into rock, and waterfalls flow

down into the valley below’

[Figure 42]

[Figure 43]

‘A vintage typewriter with paper spewing out like a waterfall’

‘A medieval knight standing on a cliff overlooking a vast

battlefield’

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

‘A cozy cabin in the middle of a snowy forest, surrounded by tall trees with lights glowing through the windows, a

northern lights display visible in the sky’

[Figure 48]

[Figure 49]

‘A dog that has been meditating all the time’ ‘A lion with a dragon's head’

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

‘Hot air balloons and flowers, collage art, photorealism, muted colors, 3D shading beautiful eldritch, mixed media,

vaporous’

[Figure 54]

[Figure 55]

‘A painting depicting a red wave outside, trapped

‘A cute fluffy sentient alien from planet Axor, in the

emotions depicted, full body, Jon Foster, depth, Dima Dmitriev, fisheye effects, Ray Collins’

andromeda galaxy, the alien have large innocent eyes and is digitigrade, high detail’

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

‘Isometric style farmhouse from RPG game, unreal engine, vibrant, beautiful, crisp, detailed, ultra detailed, intricate’

[Figure 60]

[Figure 61]

‘A snowy mountain’ ‘A man looks up at the starry sky, lonely’

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

‘A still life of a vase overflowing with vibrant flowers, painted in bold colors and textured brushstrokes, reminiscent of van

Gogh's iconic style’

[Figure 66]

[Figure 67]

‘A deep forest clearing with a mirrored pond reflecting a

‘knolling of a drawing tools and books, knowledge, white

galaxy-filled night sky’

background’

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

‘A vibrant yellow banana-shaped couch sits in a cozy living room, its curve cradling a pile of colorful cushions. on the

wooden floor, a patterned rug adds a touch of eclectic charm, and a potted plant sits in the corner, reaching towards the

sunlight filtering through the window’

[Figure 72]

[Figure 73]

‘Chinese painting of grapes’ ‘Crocodile in a sweater’

‘’

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

‘an astronaut rides a pig through in the forest. next to a river, with clouds in the sky’

[Figure 78]

[Figure 79]

‘Cthulhu, alien, in a huge towering church, an evil statue

‘A alpaca made of colorful building blocks, cyberpunk’

with a skeleton in his hand’

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

‘paper artwork, layered paper, colorful Chinese dragon surrounded by clouds’

[Figure 84]

[Figure 85]

‘An image of a chrome sphere reflecting a vibrant city

‘how world looks like in 100 A car made out of

skyline at sunset’

vegetables years, intricate, detailed’

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

‘Chinese architecture, ancient style,mountain, bird, lotus, pond, big tree, Unity, octane rendering’

[Figure 90]

[Figure 91]

‘stars, water, brilliantly, gorgeous large scale scene, a

‘a Emu, focused yet playful, ready for a competitive

little girl, in the style of dreamy realism’

matchup, photorealistic quality with cartoon vibes’

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

‘a traveler navigating via a boat in countless mountains, Chinese ink painting’

[Figure 96]

[Figure 97]

‘an illustration of an stylish swordsma’ ‘Futurist painting of the building’

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

‘beautiful scene with mountains and rivers in a small village’

[Figure 102]

[Figure 103]

‘A tranquil scene of a Japanese garden with a koi pond,

‘A dark forest under a full moon, with twisted, gnarled

painted in delicate brushstrokes and a harmonious blend

trees, shadows lurking behind every branch, and a lone

of warm and cool colors’

figure holding a glowing lantern’

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

‘A silhouette of a grand piano overlooking a dusky cityscape viewed from a top-floor penthouse, rendered in the bold and vivid style of a vintage travel poster’

