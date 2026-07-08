## Flowing from Words to Pixels: A Noise-Free Framework for Cross-Modality Evolution

# arXiv:2412.15213v2[cs.CV]24Mar2025

Qihao Liu1,2 Xi Yin1 Alan Yuille2 Andrew Brown1 Mannat Singh1

1GenAI, Meta 2Johns Hopkins University

https://cross-flow.github.io/

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

“A classic breakfast of egg and sausages on a white plate with two cups of coffee”

From image to text (image captioning)

[Figure 5]

[Figure 6]

‘Portrait of two anthropomorphic rabbits standing side by side, the left one is wearing a white coat and the right one is wearing a red coat holding a wooden weapon’

‘A oil painting of an ancient cat with yellow eyes, wearing a black wizard hat, red bow tie, and dark cloak.’

‘A white terrier wearing black headphones and speaking into a microphone in front of a computer’

[Figure 7]

[Figure 8]

[Figure 9]

From image to depth (monocular depth estimation)

[Figure 10]

[Figure 11]

From low-resolution to high-resolution image ‘A Shiba Inu dog riding a red motorcycle (image super-resolution)

‘A bird made of cheese, sitting on a plate’

‘A photo of butterfly standing on a yellow and white flower in the garden’

in the park, wearing sunglasses’

(a) Directly evolving text into images for Text-to-Image generation (b) CrossFlow for various tasks

Figure 1. We propose CrossFlow, a general and simple framework that directly evolves one modality to another using flow matching with no additional conditioning. This is enabled using a vanilla transformer without cross-attention, achieving comparable performance with state-of-the-art models on (a) text-to-image generation, and (b) various other tasks, without requiring task specific architectures.

### Abstract

Diffusion models, and their generalization, flow matching, have had a remarkable impact on the field of media generation. Here, the conventional approach is to learn the complex mapping from a simple source distribution of Gaussian noise to the target media distribution. For cross-modal tasks such as text-to-image generation, this same mapping from noise to image is learnt whilst including a conditioning mechanism in the model. One key and thus far relatively unexplored feature of flow matching is that, unlike Diffusion models, they are not constrained for the source distribution to be noise. Hence, in this paper, we propose a paradigm shift, and ask the question of whether we can instead train flow matching models to learn a direct mapping from the distribution of one modality to the distribution of another, thus obviating the need for both the noise

distribution and conditioning mechanism. We present a general and simple framework, CrossFlow, for cross-modal flow matching. We show the importance of applying Variational Encoders to the input data, and introduce a method to enable Classifier-free guidance. Surprisingly, for text-toimage, CrossFlow with a vanilla transformer without cross attention slightly outperforms standard flow matching, and we show that it scales better with training steps and model size, while also allowing for interesting latent arithmetic which results in semantically meaningful edits in the output space. To demonstrate the generalizability of our approach, we also show that CrossFlow is on par with or outperforms the state-of-the-art for various cross-modal / intra-modal mapping tasks, viz. image captioning, depth estimation, and image super-resolution. We hope this paper contributes to accelerating progress in cross-modal media generation.

### 1. Introduction

Diffusion models have achieved remarkable success in generating images [18, 63, 73, 77, 78], videos [8, 9, 35, 83], audio [42, 54], and 3D content [48, 69], revolutionizing the field of generative AI. Recently, flow matching [1, 51, 58] has been proposed as a generalization of diffusion models, where models are trained to find an optimal transport probability path between a source noise distribution and the target data distribution. This approach offers simpler, straight-line trajectories compared to the complex, curved trajectories in diffusion paths. As a result, it has been rapidly adopted in the latest state-of-the-art image and video generation models, including LDMs [23] and Movie Gen [68].

Both diffusion and flow-based models are typically trained to learn the mapping from noise to the target distribution. For cross-modal generation tasks such as text-toimage [11, 77], this same mapping from noise to the target modality distribution (i.e. the images) is learnt whilst adding a conditioning mechanism for the conditioning modality (i.e. the text) such as cross-attention. Unlike denoising diffusion models [34, 86], one relatively unexplored feature of flow matching models is that they are not constrained for the source distribution to be Gaussian noise; instead, the source distribution could be one that is correlated with the target distribution. Compared to noise, learning a mapping from such a distribution should intuitively be easier for the model because it has to learn shorter and more efficient probability paths. A question remains however as to what this correlated source distribution could be.

Interestingly, due to the information redundancy between different modalities arising from the same data point, for cross-modal generation tasks, the provided conditioning (e.g. the text in text-to-image) resembles such data that is correlated with the target distribution (e.g. the images). Hence, in this paper, we propose a paradigm shift for crossmodal generation, and ask the question of whether we can instead train flow matching models to learn a direct mapping from the distribution of one modality to the distribution of another, hence obviating the need for both the noise distribution and any conditioning mechanism.

Despite the exciting theoretical motivation, there are several key challenges in practice. First, both diffusion and flow-based models require the source and target distributions to be of the same shape; a requirement that is not satisfied for data distributions from different modalities. Secondly, state-of-the-art methods heavily rely on Classifierfree guidance (CFG) [33] for improved generation quality; a method that is not compatible with cross-modal flow matching due to the lack of a conditioning mechanism to turn on/off since the conditioning information instead lies within the source data. As a result, prior work [1, 30, 58] targets the simple setting of mapping between two similar intra-modal distributions, such as human faces to cat faces [58].

In this work, we present key architecture design solutions for overcoming these challenges: First, we employ a Variational Encoder for encoding the source modality data distribution to the same shape as the target modality, and show that the resulting regularization in the source distribution is essential for generation performance. Secondly, we enable CFG in cross-modal flow matching through the introduction of a binary conditioning indicator during training, and demonstrate the quantitative benefits of this approach compared to alternative CFG methods. We present CrossFlow; a general framework for mapping between two different modalities without the need for any conditioning mechanism or noise distribution. Typically, different crossmodal generation tasks require task-specific architectural and training modifications, but CrossFlow works for different tasks without any such changes.

Using the ubiquitous albeit challenging text-to-image (T2I) generation task as our primary setting, we show the significant result that CrossFlow outperforms commonly used flow matching baselines, given the same training data, model size, and training budget, all without requiring any cross-attention layers. CrossFlow exhibits improved scaling behavior over standard flow matching using crossattention when scaling training steps or model size, and is also compatible with a variety of Large Language Models (LLMs), including CLIP [70], T5 [71], and Llama3 [20]. Additionally, we demonstrate that since our approach encodes the source distribution into a regularized continuous space with semantic structure, CrossFlow enables exciting new latent arithmetic for the text-to-image task, e.g., L(“A dog with a hat”) + L(“Sunglasses”) – L(“A hat”) creates an image of a dog wearing sunglasses without a hat. Lastly, CrossFlow enables bi-directional mapping between modalities, allowing, for instance, the inversion of text-to-image models to serve as image-to-text (captioning) models.

We demonstrate the general-purpose nature of CrossFlow on various cross-modal/intra-modal tasks: imageto-text (image captioning), image-to-depth (depth estimation), and low-resolution to high-resolution image (superresolution). CrossFlow achieves comparable or superior performance to various state-of-the-art methods on all three tasks, without requiring task specific architectures. For example, in image captioning, CrossFlow directly projects images into a textual latent space to generate captions, achieving state-of-the-art performance using only a simple text decoder that maps textual latents to discrete tokens. Results are shown in Fig. 1. We hope this paper contributes to accelerating the progress in cross-modal media generation.

### 2. Related Work

Diffusion models and rectified flow. Starting from Gaussian noise, diffusion [34, 84] and score-based [37, 85] generative models progressively approximate the reverse ODE

of a stochastic forward process to generate data. These models have driven significant advances across various domains, particularly in high-fidelity image [6, 18, 36, 56, 66], video [8, 9, 35, 68, 83], and 3D generation [48, 55, 57, 69]. Recently, rectified flow models [1, 51, 58], such as flow matching, have been proposed to improve the generative process by enabling a transport map between two distributions. They enable faster training and sampling by avoiding complex probability flow ODEs.

Directly bridging distributions. Flow Matching theoretically allows for arbitrary distributions as the source distribution, which can then be used for direct evolution. Various approaches have been proposed in this direction, such as InterFlow [1], α-blending [30], data-dependent coupling [3], and Schr¨odinger Bridge [16, 52, 53, 81, 87, 88, 99]. They provide important theoretical support for using ODE-based methods to bridge two arbitrary distributions. However, they are still limited to similar distributions from the same domain, such as image-to-image translation (e.g., faces-tofaces [58, 99] or sketches-to-images [53]). As a step forward, CrossFlow focuses on learning the mapping between data distributions arising from even different modalities.

Text-to-image generation. Text-to-image generation [11, 15, 23, 63, 72, 73, 77, 78, 101] has rapidly advanced with diffusion and later flow matching models. This task bridges two critical and complex domains: language and vision. Existing methods typically integrate text encoders, such as LLMs, into diffusion models through additional conditioning mechanisms, with cross-attention being the most prevalent [23, 68]. However, these approaches increase model complexity and require extra parameters. We demonstrate that CrossFlow improves over standard flow matching with better scaling characteristics, and is comparable to prior work, despite a simpler architecture.

Cross-modal / intra-modal mapping. Various tasks can be framed as cross-modal/intra-modal mapping problems, including image captioning [25, 29, 44, 45, 61, 97, 100], depth estimation [7, 19, 40, 46, 47, 74, 94], and image super-resolution [24, 79]. However, due to the significant differences between modalities or distributions, previous methods have typically relied on task-specific designs. For example, Bit Diffusion [13] encodes text into binary bits and uses a diffusion model with self-conditioning for captioning. Flow-based super-resolution models, such as CFM [24], still require the low-resolution image as extra conditioning, and also add Gaussian noise to the input. In contrast, our CrossFlow uses the same unified framework across all these tasks without extra conditioning or noise.

### 3. Preliminaries

Flow Matching. We consider a generative model that defines a mapping between samples z0 from a source distribution p0 to samples z1 of a target distribution p1 via the ordi-

nary differential equation (ODE): dzt = vθ(zt,t)dt. Here, vθ is the velocity parameterized by the weights θ of a neural network, and t ∈ [0,1] is the time-step. Specifically, Flow Matching [1, 51, 58] defines the forward process as:

zt = tz1 + (1 − (1 − σmin)t)z0 (1) and σmin = 10−5. Ground truth velocity is computed as:

dzt dt

vˆt =

= z1 − (1 − σmin)z0 (2)

To achieve this, a network vθ(zt,t) is trained to predict velocity by minimizing the mean squared error (MSE) between its output and the target vˆt. This constructs a continuous path between z0 and z1 at any time-step t ∈ [0,1].

As discussed earlier, flow matching enables evolving a sample z1 from an arbitrary source distribution p0. But prior work [23, 68] typically starts from Gaussian noise z0 ∼ N(0,1), and computing the velocity with additional condition c incorporated through various methods, e.g., crossattention [23, 68], channel-wise concatenation [28].

Classifier-free guidance. CFG [33] is a broadly used technique that enhances sample quality in conditional generative models by jointly training a single model on conditional and unconditional objectives. This is achieved through randomly dropping the condition c during training with a certain probability p. Sampling is performed by extrapolating between conditional and unconditional denoising vθ(zt,c) and vθ(zt) with a scaling factor ω:

v˜θ(zt,c) = ωvθ(zt,c) + (1 − ω)vθ(zt) (3)

It significantly improves the generation quality and fidelity by guiding the samples towards higher likelihood of the condition c, which plays a crucial role in state-of-the-art media generation models [11, 23, 68, 73].

### 4. CrossFlow

In this section, we discuss the various components of our approach: a Variational Encoder (VE) to encode the inputs in Sec. 4.1, using flow matching to evolve from the source to the target distribution in Sec. 4.2, and finally, applying CFG in this setting for improving quality and fidelity in Sec. 4.3.

#### 4.1. Variational Encoder for Encoding Inputs

Flow matching models require the source distribution p0 to have the same shape as the target distribution p1. In particular, given an input x, we need to convert it to the source latent z0, which has the same shape as the target latent z1. An intuitive solution is to use an encoder E to convert x to z0, i.e., z0 = E(x), which can preserve most of the input information as shown in Appendix B.5. However, directly evolving from E(x) to z1 is problematic. We find that it is

|[Figure 12]<br><br>| |
|---|
<br><br>h x w x c<br><br>[Figure 13]<br><br>Transformer<br><br>(No cross attention)<br><br>[Figure 14]<br><br>N Sampling Steps (Flow Matching)<br><br>h x w x c|
|---|

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

Fresh ripe cherries on a wooden board

Text Variational Encoder

Image VAE Decoder

###### LLM

[Figure 23]

CLIP, T5, Llama3, etc.

[Figure 24]

[Figure 25]

[Figure 26]

n x d

- Figure 2. CrossFlow Architecture. CrossFlow enables direct evolution between two different modalities. Taking text-to-image generation as an example, our T2I model comprises two main components: a Text Variational Encoder and a standard flow matching model. At inference time, we utilize the Text Variational Encoder to extract the text latent z0 ∈ Rh×w×c from text embedding x ∈ Rn×d produced by any language model. Then we directly evolve this text latent into the image space to generate image latent z1 ∈ Rh×w×c.

essential to formulate z0 as a regularized distribution for the source in order for flow matching to work well. To address this, we propose using a VE to convert x to z0. Formally, instead of directly predicting z0, we predict its mean µ¯z

and variance σ¯z

0

). This enables us to convert the given input x into latent z0 with a regularized distribution, which can then be gradually evolved into the target distribution z1 with flow matching.

, and then sample the latent z0 ∼ N(¯µz

,σ¯z2

0

0

0

The VE can be trained with a standard Variational Autoencoding objective (VAE) [41] comprising of an encoding loss and the KL-divergence loss. For the encoding loss, the VE is trained to minimize a loss between the output z0 and a target zˆ. For a VAE this loss would be a reconstruction loss like MSE between the input x and the decoder D’s output, MSE(D(z0),x). But since we simply need a encoder and not an autoencoder, we don’t restrict ourselves to a VAE.

#### 4.2. Training CrossFlow

For each training sample, we start with an input-target pair (x,z1). We apply the VE to x to encode it to a latent z0 with the same shape as z1. Next, we employ a transformer model vθ trained for flow matching as per Equations 1 and 2. The VE can be trained prior to training vθ or concurrently. We show in Sec. 5.2 that jointly training the Variational Encoder with flow matching results in improved performance.

Specifically, we jointly train the VE with the flow matching model using a sum of flow matching MSE loss LFM, and the losses for Variational Encoder training (encoding loss LEnc and KL-divergence loss LKL):

L = LFM + LEnc + λLKL

= MSE(vθ(zt,t),vˆ) + Enc(z0,zˆ)

,σ¯z2

###### )||N(0,1)) (4)

+ λKL(N(¯µz

0

0

where λ is the KL-divergence loss weight. Eq. 4 outlines the general form of the loss function across tasks, where Lenc varies by task. Sec. 4.4 discusses text-to-image generation and choices for LEnc. More details in Appendix A.1.

#### 4.3. Classifier-Free Guidance with an Indicator

CFG [33] has become the standard low-temperature sampling method for enhancing multi-modal alignment and im-

proving quality. However, it can only be applied to generation methods that accept an additional conditioning input c, since the guidance signal relies on the difference between conditional and unconditional predictions vθ(zt,c) and vθ(zt). Recently, Autoguidance (AG) [39] has been introduced as a method to improve both conditional and unconditional generation, by guiding with a smaller, lesstrained ‘bad model’. However, it underperforms compared to standard CFG. AG also requires training a separate bad model, and its performance varies dramatically based on the choice of the bad model. While using an under-trained version of the same model narrows the search space, it affects performance and is impractical for large models due to the need to load two models during inference.

We instead aim to support CFG for CrossFlow, which is as accessible and performant as CFG is for standard flow matching. To enable CFG without the presence of an explicit conditioning input c, we introduce CFG with indicator. Specifically, our model is of the form vθ(zt,1c), where 1c ∈ {0,1} is an indicator to specify conditional vs. unconditional generation. The model evolves from z0 to z1 when 1c = 1, and from z0 to z1uc when 1c = 0, where z1uc represents any sample from the target distribution p1 other than z1. During training, we employ two learnable parameters, gc and guc, corresponding to conditional and unconditional generation, respectively. Depending on 1c, the appropriate learnable parameter is concatenated with the transformer input tokens along sequence dimension. We randomly sample the indicator with an unconditional rate of 10%, as per standard practice. The insight behind the CFG indicator is similar to that of standard CFG. In this approach, vθ(zt,1) is trained to map z0 to a specific region of the target manifold, while vθ(zt,0) is trained to map z0 to the entire target manifold to generate arbitrary unrelated images.

#### 4.4. Flowing from Text to Image

Now, we consider text-to-image generation as the archetypal task to leverage CrossFlow. We start with the input text embedding x ∈ Rn×d with token length n and dimension d, and use our Text VE to extract the corresponding text latent z0 ∼ N(¯µx,σ¯x2). While our approach is agnostic to

pixel vs. latent image generation, we consider image generation in the latent space for efficiency, and leverage a pretrained VAE to obtain the image latent from the input image I, which serves as our target z1. Then, we employ the vanilla flow matching [51] model to predict v(zt,t) between z0 and z1. The pipeline for performing text-to-image generation with CrossFlow is illustrated in Fig. 2. We discuss how to train a performant Text Variational Encoder next.

##### 4.4.1. Text Variational Encoder

Training the Text VE is challenging, as this involves compressing the text embeddings to small latent space (e.g., 77×768 CLIP tokens to 4×32×32 image latents for 256px generation, 14.4× compression). We explore various methods to train VEs for CrossFlow. The straightforward approach is to simply train a VAE with a MSE reconstruction loss. While this approach achieves very low reconstruction errors, we find that it does not capture semantic concepts well, leading to sub-optimal image generations.

Contrastive loss. We explore contrastive losses, which produce representations with strong semantic understanding when training on samples within the same modality [12, 64] and on different modality pairs [70]. To produce the contrastive targets for the VE, we either use the input text embedding x (text-text contrastive), or the paired image I for the text (image-text contrastive). Given the target, we employ a simple encoder to project it into a feature space with the same shape as z0, resulting in a representation denoted as zˆ. We then encourage semantic similarity between z0 and zˆ using the contrastive CLIP loss [70]. During training, the batch-wise contrastive loss is computed as LEnc = CLIP(z0,zˆ). We ablate this choice in Sec. 5.2 and find that contrastive loss works significantly better than the VAE reconstruction loss, with the image-text loss working slightly better than the text-text loss.

### 5. Experiments

We first evaluate CrossFlow on text-to-image generation, demonstrate its scalability, and showcase some interesting applications with latent arithmetic in Sec. 5.1. Then, we ablate our main design decisions through ablation studies in Sec. 5.2. Finally, we further explore CrossFlow’s performance on three distinct tasks: image captioning, monocular depth estimation, and image super-resolution in Sec. 5.3.

#### 5.1. Text-to-Image Generation

Experimental setup. Scientifically comparing T2I models is challenging due to diverse training datasets, often including proprietary data, and varying training conditions. In addition, our method represents a new paradigm for utilizing diffusion models, distinct from previous T2I approaches. Therefore, we primarily compare our model with the widely used “standard flow matching baseline” that starts from

Method #Params (B) #Steps (K) FID ↓ CLIP ↑

Standard FM (Baseline) 1.04 300 10.79 0.29 CrossFlow (Ours) 0.95 300 10.13 0.29

Table 1. Comparison between our CrossFlow and standard flow matching with cross-attention. Both models are trained with the same settings. We find that our model slightly outperforms standard flow matching baseline in terms of zero-shot FID30K and achieves comparable performance on the CLIP score.

noise and uses text cross-attention. For fairness, both CrossFlow and the baseline share the same codebase, training recipe, dataset, and budget. Unlike the baseline, which requires cross-attention after each self-attention layer, our model only relies on self-attention, reducing parameters per layer. To account for this, we adjust the number of layers to match model sizes. For both methods, we use a grid search to find the optimal CFG scale. We also compare CrossFlow with state-of-the-art T2I models to demonstrate that our approach is competitive with those established methods.

Architecture. Our model enables the use of vanilla Transformer [90] with self-attention layers and feed-forward layers. We use DiMR [56] as the flow matching backbone, a variant of Diffusion Transformer (DiT) [66] which replaces the parameter-heavy MLP in adaLN-Zero with a lightweight Time-Dependent Layer Normalization. For the Text VE, we apply stacked Transformer blocks, followed by a linear layer to project the output into the target shape.

Training details. We use a proprietary dataset with about 350M image-text pairs to train both CrossFlow and our ablations. Our text encoder is based on CLIP [70] with a fixed sequence length of 77 text tokens. We use a pre-trained and frozen VAE from LDM [77] to extract image latents. Logitnormal sampling [23] is used to bias the training timesteps. All T2I models are trained using the same settings: an image resolution of 256 × 256, a batch size of 1024, a base learning rate of 1 × 10−4 with 5000 warm-up steps, and an AdamW optimizer [60] with β1 = β2 = 0.9 and a weight decay of 0.03, and a KL loss weight of λ = 1 × 10−4. We train our largest model (0.95B) on 256 × 256 for 600K iterations, then finetune it on 512×512 for an additional 240K iterations for higher resolution generation.

Evaluation metrics. We evaluate all models on the COCO validation set [50] and report FID [32] and CLIP score [31, 70]. Following previous works, we report zero-shot FID30K, where 30K prompts are randomly sampled from the validation set, and the generated images are compared to reference images from the full validation set. Additionally, we also evaluate our models on GenEval benchmark as it exhibits strong alignment with human judgment [27].

##### 5.1.1. CrossFlow vs. Standard Flow Matching

We compare our CrossFlow with widely used crossattention baseline in Tab. 1. Both models are trained and tested under the same settings. The results show that Cross-

Standard FM

- 12
- 13
- 14
- 15
- 16
- 17

22

| | | | | | | | |C|ro|ss|Fl|ow| | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

18

FID-30K

FID-30K

14

10

6

50 100 150 200 250 300

70 150 300 500 1000

Training steps (thousands)

Model parameters (millions)

- Figure 3. Performance vs. Model Parameters and Iterations. We compare the baseline of starting from noise with text crossattention with CrossFlow, while controlling for data, model size and training steps. Left: Larger models are able to exploit the cross-modality connection better. Right: CrossFlow needs more steps to converge, but converges to better final performance. Overall, CrossFlow scales better than the baseline and can serve as the framework for future media generation models.

Flow achieves comparable performance, with slightly better zero-shot FID-30K compared with widely used flow matching baselines with cross-attention.

Scaling characteristics. We investigate the scalability of CrossFlow in Fig. 3 and compare it with standard flow matching. We train both approaches across 5 different model sizes, ranging from 70M to 1B parameters, with the same training settings, for 300K iterations. At smaller scales, CrossFlow underperforms the baseline, likely due to the lack of sufficient parameters to model the complex relationships between two modalities. But excitingly, as the model size increases, the zero-shot FID-30K improves more for our approach. Next, we evaluate the effect of varying the training iterations. We notice similarly that CrossFlow improves more as we increase training iterations.

While CrossFlow initially underperforms standard flow matching at small scales, increasing the model size and training iterations improves it significantly, even enabling it to surpass standard flow matching. We attribute this to the fact that CrossFlow generates images by directly evolving from the source distribution where different sub-regions correspond to different semantics. In contrast, standard flow matching may generate the same semantics from the entire source distribution, while exploiting the inductive biases afforded by text cross-attention. Ultimately, this works in favor of CrossFlow, as the learnt cross-modal paths and fewer inductive biases result in improved scaling characteristics with both model size and training iterations.

##### 5.1.2. State-of-the-art Comparison

Finally, we compare CrossFlow with state-of-the-art textto-image models and report results in Tab. 2. We additionally explore sin-cos matching [2] and find it improves over vanilla linear flow matching. We achieve a zero-shot FID-30K of 8.95 on COCO, and a GenEval score of 0.57, demonstrating performance comparable with the state-ofthe-art. Note that our model uses only 630 A100 GPU-days

FID-30K ↓ GenEval ↑

Method #Params.

zero-shot score

DALL·E [72] 12.0B 27.50 GLIDE [63] 5.0B 12.24 LDM [77] 1.4B 12.63 DALL·E 2 [73] 6.5B 10.39 0.52 LDMv1.5 [77] 0.9B 9.62 0.43 Imagen [78] 3.0B 7.27 RAPHAEL [92] 3.0B 6.61 PixArt-α [11] 0.6B 7.32 0.48 LDMv3 (5122) [23] 8.0B - 0.68

CrossFlow 0.95B 9.63 0.55 CrossFlow (Sin-Cos) 0.95B 8.95 0.57

Table 2. Comparison to recent T2I models. For GenEval, we report overall scores here and task-specific results in Appendix B.1. CrossFlow achieves comparable results to state-of-the-art models by evolving text directly into images. CrossFlow (Sin-Cos) replaces simple linear flow matching with sin-cos matching [2].

for training, whereas other methods like DALL·E 2 [73] typically require thousands of A100 GPU days. These results suggest that CrossFlow is a simple and promising direction for state-of-the-art media generation.

##### 5.1.3. Arithmetic Operations in Latent Space

Unlike previous diffusion or flow matching models, CrossFlow offers a unique property: arithmetic operations in the input latent space translate to similar operations in the output space. This is made possible since CrossFlow transforms the source space (i.e., the text latent space for T2I) into a regularized continuous space, where a uniform representation shape is shared across all texts. We showcase two examples of this, latent interpolation, and latent arithmetic. For latent interpolation, we use the Text Variational Encoder to generate text latents from two different text inputs, and then interpolate between them to produce images. As shown in Fig. 4, CrossFlow enables visually smooth linear interpolations, even between disparate prompts. Next, we showcase arithmetic operations in Fig. 5, in which we apply addition and subtraction in the text latent space, and find that the resulting images exhibit corresponding semantic modifications to the original image. This shows that CrossFlow formulates meaningful and well-structured semantic paths between the source and target distributions, providing additional capabilities and more control over standard flow matching approaches. See Appendix B.2 for further details.

#### 5.2. Ablation Study

We conduct various ablation experiments to verify the effectiveness of the proposed designs in Tab. 3.

Variational Encoder vs. standard encoder. Compared to a standard encoder or even adding Guassian noise like CFM [24], a Variational Encoder significantly improves the generation quality, with significant gains in the FID. This shows that forming a regularized distribution for the source domain is a crucial step for cross-modal flow matching.

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

A white dog wearing a white and black helmet riding a bike in the park

An orange cat wearing sunglasses on a ship

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

A robot cooking dinner in the kitchen

A panda eating hamburger in a classroom

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

A teddy bear dressed in black wizard hat and robes sitting on the bed

A corgi wearing a red hat in the park

- Figure 4. CrossFlow provides visually smooth interpolations in the latent space. We show images generated by linear interpolation between the first (left) and second (right) text latents. CrossFlow enables visually smooth transformations of object direction, composite colors, shapes, background scenes, and even object categories. Please zoom in for better visualization. For brevity, we display only 7 interpolating images here; additional interpolating images can be found in Appendix C (Fig. 11 and Fig. 12).

z0= VE(‘Awhitedog

wearing a black hat’)

z0=VE(‘Ahat’) z0=VE(‘Awhitedogwearinga

black hat’) + VE(‘Sunglasses’)

- VE(‘A hat’)

z0=VE(‘Alabrador’) z0=VE(‘snow’) z0=VE(‘Alabradorinfrontof Eiffel Tower’) - VE(‘A labrador’) + VE(‘snow’)

z0= VE(‘Sunglasses’)

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

= VE(‘A labrador in front of Eiffel Tower’)

z0

- Figure 5. CrossFlow allows arithmetic in text latent space. Using the Text Variational Encoder (VE), we first map the input text into the latent space z0. Arithmetic operations are then performed in this latent space, and the resulting latent representation is used to generate the corresponding image. The latent code z0 used to generate each image is provided at the bottom.

###### Loss FID ↓ CLIP ↑

Text encoder FID ↓ CLIP ↑

T-T Recon. 40.78 0.23 T-T Contrast. 34.67 0.24 I-T Contrast. 33.41 0.24

Encoder 66.65 0.20 Encoder + noise 59.91 0.21 Variational Encoder 40.78 0.23

(a) Variational Encoder *

(b) Text VE loss*

###### Method FID ↓ CLIP ↑

###### Model FID ↓ CLIP ↑

No guidance 33.41 0.24 AG 26.36 0.25 CFG indicator 24.33 0.26

CLIP (0.4B) 24.33 0.26 T5-XXL (11B) 22.28 0.27 Llama3 (7B) 21.20 0.27

(c) CFG with indicator

(d) Language Model

###### Train strategy FID ↓ CLIP ↑

2-stage separate training 32.55 0.24 Joint training 24.33 0.26 2-stage w/ joint finetuning 23.79 0.26

(e) Training strategy

Table 3. Ablation study on Text Variational Encoder, training objective, CFG, language models, and training strategy. We conduct ablation study on our smallest model (70M), reporting zero-shot FID-10K and CLIP scores. Final settings used for CrossFlow are underlined. AG: Autoguidance. *: results without applying CFG.

Joint training vs. two-stage training. We consider three training strategies: (1) jointly training the VE and flow matching from scratch, (2) training the VE first and then training flow matching with a fixed VE, and (3) training the VE first and then training the flow matching while jointly fine-tuning VE. We observe that it is important to update the VE when training the flow matching, either through joint training from scratch, or finetuning the VE jointly with flow matching. Initializing with a pre-trained VE and then jointly training improves convergence speed by about 35%, but we opt to jointly train both models from scratch on account of the simplicity, and for fair comparisons with baselines.

improves FID and also image-text CLIP alignment slightly, our CFG indicator works better than AG in terms of both FID and CLIP alignment while only using a single model trained with standard CFG settings. Qualitatively, our approach produces much higher fidelity images compared to both alternatives, as shared in Appendix B.5.

Text VE loss. We explore reconstruction and contrastive objectives for the encoder loss LEnc when training the text VE. We find that contrastive loss, which enhances semantic understanding, significantly outperforms reconstruction loss on input text embeddings. Moreover, image-text contrastive loss slightly surpasses text-text contrastive loss.

CFG indicator. We evaluate the performance of our model when leveraging our proposed CFG indicator techinuqe. We also evaluate Autoguidance (AG) [39], which utilizes two models for inference – we use an under-trained version of the same model as the bad model, while using a gridsearch to find the best under-trained checkpoint. While AG

Effect of different language models. We evaluate CrossFlow with various language models trained with different objectives. Specifically, we evaluate CLIP [70] (contrastive image-text), T5-XXL’s encoder [71] (encoder-decoder), Llama3-7B [20] (decoder-only). We use 77 tokens for

###### Method B@4 ↑ M ↑ R ↑ C ↑ S ↑

MNIC [25] 30.9 27.5 55.6 108.1 21.0 MIR [44] 32.5 27.2 - 109.5 20.6 NAIC-CMAL [29] 35.3 27.3 56.9 115.5 20.8 SATIC [100] 32.9 27.0 - 111.0 20.5 SCD-Net [61] 37.3 28.1 58.0 118.0 21.6

CrossFlow-T2I (Ours) 33.1 27.0 56.4 111.2 20.3 CrossFlow (Ours) 36.4 27.8 57.1 116.2 20.4

Table 4. Image captioning on COCO Karpathy split. CrossFlow directly evolves from image to text, achieving comparable performance to state-of-the-art models on image captioning. For a fair comparison, we consider non-autoregressive methods that are trained without CIDEr optimization. CrossFlow-T2I achieves captioning by simply inverting our text-to-image CrossFlow model.

all language models, resulting in text embeddings of size 77×768, 77×4096, 77×4096, respectively. We train a separate Text VE for each language model, projecting the text embeddings into the target image latent shape (4×32×32). CrossFlow works well with all language models regardless of their training objectives and embedding sizes. As expected, our performance improves with better text representations. Due to compute restrictions however, we use the light-weight CLIP model for our main experiments.

#### 5.3. CrossFlow for Various Tasks

We further evaluate CrossFlow on three distinct tasks that involve cross-modal / intra-modal evolution. We present the main results and key findings here, while additional details and qualitative results can be found in the Appendix.

Image to text (captioning). We first consider the task of image captioning. To achieve this, we train a new Text Variational Encoder on the captioning dataset to extract text latents from text tokens, and a separate text decoder with a reconstruction loss to convert text latents back into tokens. CrossFlow is then trained to map from the image latent space to the text latent space. Following previous work, we use the Karpathy split [38] of COCO dataset [50] for training and testing. In addition, we can also leverage the bi-directional flow property, and simply fine-tune our textto-image CrossFlow model on COCO and use its inversion for captioning. We report results in Tab. 4. CrossFlow enables direct evolution from image space to text space for image captioning, achieving state-of-the-art performance.

Image to depth (depth estimation). For monocular depth estimation, we train CrossFlow in pixel space. Specifically, we use a recontruction loss to train the Image Variational Encoder to map the original image into the shape of a depth map, followed by the flow matching model which generates the final depth maps. We train and evaluate our model on KITTI [26] (Eigen split [22]) and NYUv2 [82] (official split) for outdoor and indoor scenarios, respectively. As shown in Tab. 5, our model achieves comparable performance to state-of-the-art methods on both datasets. No-

KITTI NYUv2 AbsRel (↓) δ1 (↑) AbsRel (↓) δ1 (↑) TransDepth [93] 0.064 0.956 0.106 0.900 AdaBins [7] 0.058 0.964 0.103 0.903 DepthFormer [46] 0.052 0.975 0.096 0.921 BinsFormer [47] 0.052 0.974 0.094 0.925 DiffusionDepth [19] 0.050 0.977 0.085 0.939

Method

CrossFlow (Ours) 0.053 0.973 0.094 0.928

- Table 5. Monocular depth estimation on KITTI and NYUv2. CrossFlow enables direct mapping from image to depth, achieving comparable performance to state-of-the-art models.

Method FID ↓ IS ↑ PSNR ↑ SSIM ↑ Reference 1.9 240.8 - -

Regression 15.2 121.1 27.9 0.801 SR3 [79] 5.2 180.1 26.4 0.762 Flow Matching [51] 3.4 200.8 24.7 0.747

CrossFlow (Ours) 3.0 207.2 25.6 0.764

- Table 6. Image super-resolution on the ImageNet validation set. Our direct mapping method achieves better performance.

tably, DiffusionDepth [19] utilizes Swin Transformer [59] and specific designs such as Multi-Scale Aggregation and Monocular Conditioned Denoising Block. In contrast, our model achieves similar performance without any additional enhancements, demonstrating the efficiency and effectiveness of CrossFlow in mapping from images to depth.

Low-resolution to high-resolution (super-resolution). We compare CrossFlow with the standard flow-matching super-resolution method, which upsamples the lowresolution image, concatenates it with input noise as conditioning, and then processes it through the neural network. In contrast, we directly evolve the upsampled low-resolution image into a high-resolution image, without additional concatenation conditioning. We also compare against SR3 [79] which uses diffusion models for super-resolution. Following previous work [51, 79], we train and evaluate our model on ImageNet [17] for 64×64 → 256×256 super-resolution, and provide results in Tab. 6. Our method achieves better results compared to the standard flow matching and SR3, indicating that CrossFlow can also effectively evolve between similar distributions while achieving superior performance.

### 6. Conclusion

In this paper, we proposed CrossFlow, a simple and general framework for cross-modal flow matching that works well across a variety of tasks without requiring task specific architectural modifications. It outperforms conventional flow matching, while also enabling new capabilities such as latent arithmetic. We showcase that CrossFlow is a promising approach for the future thanks to its better scalablity. We hope our approach helps pave the way towards further research and applications of cross-modal flow matching.

Acknowledgements. We sincerely appreciate Ricky Chen and Saketh Rambhatla for their valuable discussions.

### References

- [1] Michael S Albergo and Eric Vanden-Eijnden. Building normalizing flows with stochastic interpolants. In ICLR, 2023. 2, 3
- [2] Michael S Albergo, Nicholas M Boffi, and Eric VandenEijnden. Stochastic interpolants: A unifying framework for flows and diffusions. arXiv preprint arXiv:2303.08797,

2023. 6

- [3] Michael S Albergo, Mark Goldstein, Nicholas M Boffi, Rajesh Ranganath, and Eric Vanden-Eijnden. Stochastic interpolants with data-dependent couplings. arXiv preprint arXiv:2310.03725, 2023. 3
- [4] Peter Anderson, Basura Fernando, Mark Johnson, and Stephen Gould. Spice: Semantic propositional image caption evaluation. In ECCV, 2016. 13
- [5] Satanjeev Banerjee and Alon Lavie. Meteor: An automatic metric for mt evaluation with improved correlation with human judgments. In Proceedings of the acl workshop on intrinsic and extrinsic evaluation measures for machine translation and/or summarization, 2005. 13
- [6] Fan Bao, Shen Nie, Kaiwen Xue, Yue Cao, Chongxuan Li, Hang Su, and Jun Zhu. All are worth words: A vit backbone for diffusion models. In CVPR, 2023. 3
- [7] Shariq Farooq Bhat, Ibraheem Alhashim, and Peter Wonka. Adabins: Depth estimation using adaptive bins. In CVPR,

2021. 3, 8

- [8] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023. 2, 3
- [9] Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. In CVPR, 2023. 2, 3
- [10] Yohann Cabon, Naila Murray, and Martin Humenberger. Virtual kitti 2. arXiv preprint arXiv:2001.10773, 2020. 14, 15
- [11] Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Yue Wu, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, et al. Pixart-α : Fast training of diffusion transformer for photorealistic text-to-image synthesis. arXiv preprint arXiv:2310.00426, 2023. 2, 3, 6, 14
- [12] Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations. In ICML, 2020. 5
- [13] Ting Chen, Ruixiang Zhang, and Geoffrey Hinton. Analog bits: Generating discrete data using diffusion models with self-conditioning. arXiv preprint arXiv:2208.04202, 2022. 3
- [14] Angela Dai, Angel X Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias Nießner. Scannet:

- Richly-annotated 3d reconstructions of indoor scenes. In CVPR, 2017. 14
- [15] Xiaoliang Dai, Ji Hou, Chih-Yao Ma, Sam Tsai, Jialiang Wang, Rui Wang, Peizhao Zhang, Simon Vandenhende, Xiaofang Wang, Abhimanyu Dubey, et al. Emu: Enhancing image generation models using photogenic needles in a haystack. arXiv preprint arXiv:2309.15807, 2023. 3
- [16] Valentin De Bortoli, Guan-Horng Liu, Tianrong Chen, Evangelos A Theodorou, and Weilie Nie. Augmented bridge matching. arXiv preprint arXiv:2311.06978, 2023. 3
- [17] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In CVPR, 2009. 8, 13
- [18] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. In NeurIPS, 2021. 2, 3
- [19] Yiqun Duan, Xianda Guo, and Zheng Zhu. Diffusiondepth: Diffusion denoising approach for monocular depth estimation. arXiv preprint arXiv:2303.05021, 2023. 3, 8
- [20] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783,

2024. 2, 7

- [21] Ainaz Eftekhar, Alexander Sax, Jitendra Malik, and Amir Zamir. Omnidata: A scalable pipeline for making multitask mid-level vision datasets from 3d scans. In ICCV,

2021. 15

- [22] David Eigen, Christian Puhrsch, and Rob Fergus. Depth map prediction from a single image using a multi-scale deep network. In NeurIPS, 2014. 8, 13
- [23] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In ICML, 2024. 2, 3, 5, 6, 14
- [24] Johannes S Fischer, Ming Gui, Pingchuan Ma, Nick Stracke, Stefan A Baumann, and Bj¨orn Ommer. Boosting latent diffusion with flow matching. arXiv preprint arXiv:2312.07360, 2023. 3, 6
- [25] Junlong Gao, Xi Meng, Shiqi Wang, Xia Li, Shanshe Wang, Siwei Ma, and Wen Gao. Masked non-autoregressive image captioning. arXiv preprint arXiv:1906.00717, 2019. 3, 8
- [26] Andreas Geiger, Philip Lenz, Christoph Stiller, and Raquel Urtasun. Vision meets robotics: The kitti dataset. The International Journal of Robotics Research, 2013. 8, 13, 14
- [27] Dhruba Ghosh, Hannaneh Hajishirzi, and Ludwig Schmidt. Geneval: An object-focused framework for evaluating textto-image alignment. In NeurIPS, 2024. 5
- [28] Rohit Girdhar, Mannat Singh, Andrew Brown, Quentin Duval, Samaneh Azadi, Sai Saketh Rambhatla, Akbar Shah, Xi Yin, Devi Parikh, and Ishan Misra. Emu video: Factorizing text-to-video generation by explicit image conditioning. arXiv preprint arXiv:2311.10709, 2023. 3
- [29] Longteng Guo, Jing Liu, Xinxin Zhu, Xingjian He, Jie Jiang, and Hanqing Lu. Non-autoregressive image captioning with counterfactuals-critical multi-agent learning. arXiv preprint arXiv:2005.04690, 2020. 3, 8

- [30] Eric Heitz, Laurent Belcour, and Thomas Chambon. Iterative α-(de) blending: A minimalist deterministic diffusion model. In SIGGRAPH, 2023. 2, 3
- [31] Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. Clipscore: A reference-free evaluation metric for image captioning. arXiv preprint arXiv:2104.08718, 2021. 5
- [32] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. In NeurIPS, 2017. 5
- [33] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022. 2, 3, 4
- [34] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In NeurIPS, 2020. 2
- [35] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, et al. Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303, 2022. 2, 3
- [36] Jonathan Ho, Chitwan Saharia, William Chan, David J Fleet, Mohammad Norouzi, and Tim Salimans. Cascaded diffusion models for high fidelity image generation. JMLR,

2022. 3

- [37] Aapo Hyv¨arinen and Peter Dayan. Estimation of nonnormalized statistical models by score matching. JMLR,

2005. 2

- [38] Andrej Karpathy and Li Fei-Fei. Deep visual-semantic alignments for generating image descriptions. In CVPR,

2015. 8, 13

- [39] Tero Karras, Miika Aittala, Tuomas Kynk¨a¨anniemi, Jaakko Lehtinen, Timo Aila, and Samuli Laine. Guiding a diffusion model with a bad version of itself. arXiv preprint arXiv:2406.02507, 2024. 4, 7, 15
- [40] Bingxin Ke, Anton Obukhov, Shengyu Huang, Nando Metzger, Rodrigo Caye Daudt, and Konrad Schindler. Repurposing diffusion-based image generators for monocular depth estimation. In CVPR, 2024. 3, 14, 15
- [41] Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013. 4
- [42] Zhifeng Kong, Wei Ping, Jiaji Huang, Kexin Zhao, and Bryan Catanzaro. Diffwave: A versatile diffusion model for audio synthesis. arXiv preprint arXiv:2009.09761, 2020. 2
- [43] Solomon Kullback and Richard A Leibler. On information and sufficiency. The annals of mathematical statistics,

1951. 13

- [44] Jason Lee, Elman Mansimov, and Kyunghyun Cho. Deterministic non-autoregressive neural sequence modeling by iterative refinement. arXiv preprint arXiv:1802.06901,

2018. 3, 8

- [45] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In ICML, 2022. 3
- [46] Zhenyu Li, Zehui Chen, Xianming Liu, and Junjun Jiang. Depthformer: Exploiting long-range correlation and local information for accurate monocular depth estimation. Machine Intelligence Research, 2023. 3, 8

- [47] Zhenyu Li, Xuyang Wang, Xianming Liu, and Junjun Jiang. Binsformer: Revisiting adaptive bins for monocular depth estimation. IEEE Transactions on Image Processing, 2024. 3, 8
- [48] Chen-Hsuan Lin, Jun Gao, Luming Tang, Towaki Takikawa, Xiaohui Zeng, Xun Huang, Karsten Kreis, Sanja Fidler, Ming-Yu Liu, and Tsung-Yi Lin. Magic3d: Highresolution text-to-3d content creation. In CVPR, 2023. 2, 3
- [49] Chin-Yew Lin. Rouge: A package for automatic evaluation of summaries. In Text summarization branches out, 2004. 13
- [50] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In ECCV, 2014. 5, 8, 13
- [51] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. In ICLR, 2022. 2, 3, 5, 8, 13
- [52] Guan-Horng Liu, Yaron Lipman, Maximilian Nickel, Brian Karrer, Evangelos A Theodorou, and Ricky TQ Chen. Generalized schr\” odinger bridge matching. arXiv preprint arXiv:2310.02233, 2023. 3
- [53] Guan-Horng Liu, Arash Vahdat, De-An Huang, Evangelos A Theodorou, Weili Nie, and Anima Anandkumar. Image-to-image schr\” odinger bridge. arXiv preprint arXiv:2302.05872, 2023. 3
- [54] Haohe Liu, Zehua Chen, Yi Yuan, Xinhao Mei, Xubo Liu, Danilo Mandic, Wenwu Wang, and Mark D Plumbley. Audioldm: Text-to-audio generation with latent diffusion models. arXiv preprint arXiv:2301.12503, 2023. 2
- [55] Minghua Liu, Chao Xu, Haian Jin, Linghao Chen, Mukund Varma T, Zexiang Xu, and Hao Su. One-2-3-45: Any single image to 3d mesh in 45 seconds without per-shape optimization. Advances in Neural Information Processing Systems, 36:22226–22246, 2023. 3
- [56] Qihao Liu, Zhanpeng Zeng, Ju He, Qihang Yu, Xiaohui Shen, and Liang-Chieh Chen. Alleviating distortion in image generation via multi-resolution diffusion models. arXiv preprint arXiv:2406.09416, 2024. 3, 5
- [57] Qihao Liu, Yi Zhang, Song Bai, Adam Kortylewski, and Alan Yuille. Direct-3d: Learning direct text-to-3d generation on massive noisy 3d data. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6881–6891, 2024. 3
- [58] Xingchao Liu, Chengyue Gong, and qiang liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. In ICLR, 2023. 2, 3
- [59] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In ICCV, 2021. 8
- [60] I Loshchilov. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017. 5
- [61] Jianjie Luo, Yehao Li, Yingwei Pan, Ting Yao, Jianlin Feng, Hongyang Chao, and Tao Mei. Semantic-conditional diffusion networks for image captioning. In CVPR, 2023. 3, 8

- [62] Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. Efficient estimation of word representations in vector space. arXiv preprint arXiv:1301.3781, 2013. 14
- [63] Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. arXiv preprint arXiv:2112.10741, 2021. 2, 3, 6
- [64] Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748, 2018. 5
- [65] Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. Bleu: a method for automatic evaluation of machine translation. In ACL, 2002. 13
- [66] William Peebles and Saining Xie. Scalable diffusion models with transformers. In ICCV, 2023. 3, 5
- [67] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023. 13, 14
- [68] Adam Polyak, Amit Zohar, Andrew Brown, Andros Tjandra, Animesh Sinha, Ann Lee, Apoorv Vyas, Bowen Shi, Chih-Yao Ma, Ching-Yao Chuang, et al. Movie gen: A cast of media foundation models. arXiv preprint arXiv:2410.13720, 2024. 2, 3
- [69] Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. arXiv preprint arXiv:2209.14988, 2022. 2, 3
- [70] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, 2021. 2, 5, 7
- [71] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. JMLR, 2020. 2, 7
- [72] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In ICML,

2021. 3, 6

- [73] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 2022. 2, 3, 6, 13, 14
- [74] Ren´e Ranftl, Katrin Lasinger, David Hafner, Konrad Schindler, and Vladlen Koltun. Towards robust monocular depth estimation: Mixing datasets for zero-shot crossdataset transfer. TPAMI, 2020. 3, 15
- [75] Ren´e Ranftl, Alexey Bochkovskiy, and Vladlen Koltun. Vision transformers for dense prediction. In ICCV, 2021. 15
- [76] Mike Roberts, Jason Ramapuram, Anurag Ranjan, Atulit Kumar, Miguel Angel Bautista, Nathan Paczan, Russ Webb, and Joshua M Susskind. Hypersim: A photorealistic synthetic dataset for holistic indoor scene understanding. In ICCV, 2021. 14, 15

- [77] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, 2022. 2, 3, 5, 6, 13, 14
- [78] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. In NeurIPS, 2022. 2, 3, 6
- [79] Chitwan Saharia, Jonathan Ho, William Chan, Tim Salimans, David J Fleet, and Mohammad Norouzi. Image super-resolution via iterative refinement. TPAMI, 2022. 3, 8, 13
- [80] Thomas Schops, Johannes L Schonberger, Silvano Galliani, Torsten Sattler, Konrad Schindler, Marc Pollefeys, and Andreas Geiger. A multi-view stereo benchmark with high-resolution images and multi-camera videos. In CVPR,

2017. 14

- [81] Yuyang Shi, Valentin De Bortoli, Andrew Campbell, and Arnaud Doucet. Diffusion schr¨odinger bridge matching. In NeurIPS, 2024. 3
- [82] Nathan Silberman, Derek Hoiem, Pushmeet Kohli, and Rob Fergus. Indoor segmentation and support inference from rgbd images. In ECCV, 2012. 8, 13, 14
- [83] Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, et al. Make-a-video: Text-to-video generation without text-video data. arXiv preprint arXiv:2209.14792,

2022. 2, 3

- [84] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In ICML, 2015. 2
- [85] Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. In NeurIPS,

2019. 2

- [86] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Scorebased generative modeling through stochastic differential equations. In ICLR, 2021. 2
- [87] Zhicong Tang, Tiankai Hang, Shuyang Gu, Dong Chen, and Baining Guo. Simplified diffusion schr\” odinger bridge. arXiv preprint arXiv:2403.14623, 2024. 3
- [88] Alexander Tong, Nikolay Malkin, Kilian Fatras, Lazar Atanackovic, Yanlei Zhang, Guillaume Huguet, Guy Wolf, and Yoshua Bengio. Simulation-free schr\” odinger bridges via score and flow matching. arXiv preprint arXiv:2307.03672, 2023. 3
- [89] Igor Vasiljevic, Nick Kolkin, Shanyi Zhang, Ruotian Luo, Haochen Wang, Falcon Z Dai, Andrea F Daniele, Mohammadreza Mostajabi, Steven Basart, Matthew R Walter, et al. Diode: A dense indoor and outdoor depth dataset. arXiv preprint arXiv:1908.00463, 2019. 14
- [90] A Vaswani. Attention is all you need. In NeurIPS, 2017. 5
- [91] Ramakrishna Vedantam, C Lawrence Zitnick, and Devi Parikh. Cider: Consensus-based image description evaluation. In CVPR, 2015. 13

- [92] Zeyue Xue, Guanglu Song, Qiushan Guo, Boxiao Liu, Zhuofan Zong, Yu Liu, and Ping Luo. Raphael: Text-toimage generation via large mixture of diffusion paths. In NeurIPS, 2024. 6
- [93] Guanglei Yang, Hao Tang, Mingli Ding, Nicu Sebe, and Elisa Ricci. Transformer-based attention networks for continuous pixel-wise prediction. In ICCV, 2021. 8
- [94] Lihe Yang, Bingyi Kang, Zilong Huang, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything: Unleashing the power of large-scale unlabeled data. In CVPR,

2024. 3

- [95] Wei Yin, Xinlong Wang, Chunhua Shen, Yifan Liu, Zhi Tian, Songcen Xu, Changming Sun, and Dou Renyin. Diversedepth: Affine-invariant depth prediction using diverse data. arXiv preprint arXiv:2002.00569, 2020. 15
- [96] Wei Yin, Jianming Zhang, Oliver Wang, Simon Niklaus, Long Mai, Simon Chen, and Chunhua Shen. Learning to recover 3d scene shape from a single image. In CVPR, 2021. 15
- [97] Quanzeng You, Hailin Jin, Zhaowen Wang, Chen Fang, and Jiebo Luo. Image captioning with semantic attention. In CVPR, 2016. 3
- [98] Chi Zhang, Wei Yin, Billzb Wang, Gang Yu, Bin Fu, and Chunhua Shen. Hierarchical normalization for robust monocular depth estimation. In NeurIPS, 2022. 15
- [99] Linqi Zhou, Aaron Lou, Samar Khanna, and Stefano Ermon. Denoising diffusion bridge models. arXiv preprint arXiv:2309.16948, 2023. 3
- [100] Yuanen Zhou, Yong Zhang, Zhenzhen Hu, and Meng Wang. Semi-autoregressive transformer for image captioning. In ICCV, 2021. 3, 8
- [101] Yufan Zhou, Bingchen Liu, Yizhe Zhu, Xiao Yang, Changyou Chen, and Jinhui Xu. Shifted diffusion for textto-image generation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10157–10166, 2023. 3

### Appendix

In the appendix, we provide additional information as listed below:

- – Sec. A. Method details
- – Sec. A.1. Loss function for text-to-image generation
- – Sec. A.2. Experimental details for various tasks
- – Sec. B. Additional experimental results
- – Sec. B.1. GenEval performance for text-to-image
- – Sec. B.2. Analysis of Arithmetic Operations
- – Sec. B.3. Zero-shot depth estimation
- – Sec. B.4. Image super-resolution
- – Sec. B.5. Ablations on Text VE and CFG indicator
- – Sec. C. Additional qualitative examples
- – Fig. 10. Text-to-image generation
- – Fig. 11, 12. Interpolation in latent space
- – Fig. 13. Arithmetic in latent space

### A. Method Details

#### A.1. Loss Function for T2I Generation

We jointly train the Text Variational Encoder with the flow matching model using the following training objective:

L = LFM + LEnc + λLKL

= MSE(vθ(zt,t),vˆ) + CLIP(z0,zˆ)

,σ¯z2

###### )||N(0,1)) (5)

+ λKL(N(¯µz

0

0

where λ is the weight of KL-divergence loss. For the flow matching loss LFM, we follow previous work [51] and compute the MSE loss between the predicted velocity vθ(zt,t) at time-step t and the ground-truth velocity vˆ. To train the Text Variational Encoder, we adopt a CLIP contrastive loss. Specifically, given a batch of N text and image pairs, we use our Text Variational Encoder to obtain text latents z0, and an image encoder to extract image features zˆ. Then, we compute the cosine similarity between all pairs of z0 and zˆ in the batch, resulting in a similarity matrix S, where each element sij represents the cosine similarity between the ith z0 and jth zˆ. The similarity scores are then scaled by a temperature parameter τ (a learnable parameter), denoted as logitsij = sij/τ. After that, a symmetric cross-entropy loss over the similarity scores is computed:

N

1 N

LI2T = −

i=1

N

1 N

LT2I = −

i=1

exp(logitsii)

log

N j=1 exp(logitsij)

exp(logitsii)

log

N j=1 exp(logitsji)

(6)

(7)

Finally, we compute the average of these two components to obtain the CLIP loss, which is then used to update our

Text Variational Encoder:

- 1

- 2

(LI2T + LT2I) (8)

LEnc = CLIP(z0,zˆ) =

For the KL loss LKL, we adopt the original KL divergence loss [43] with λ = 1 × 10−4.

#### A.2. Experimental Details for Various Tasks

Image captioning. We conduct our experiments on the popular Karpathy split [38] of COCO dataset [50], which contains 113,287 images for training, 5,000 images for validation, and 5,000 image for testing. We train our model with 351M parameters on the training split for 100 epochs, using a batch size of 256 and a base learning rate of 2×10−4 with 5 warm-up epochs. Following the standard evaluation setup, we compare the performance over five metrics: BLEU@4 [65] (B@4), METEOR [5] (M), ROUGE [49] (R), CIDEr [91] (C), and SPICE [4] (S).

Monocular depth estimation. We consider KITTI [26] and NYUv2 [82] for outdoor and indoor depth estimation. For KITTI, we use the Eigen split [22], consisting of 23,488 training images and 697 testing images. For NYUv2, we adopt the official split, which contains 24,231 training images and 654 testing images. We train our model with 527M parameters on the corresponding training splits for 50 epochs. We use a batch size of 64, and decay the learning rate from 1 × 10−4 to 1 × 10−8 with cosine annealing. Image super-resolution. We consider natural image superresolution, training our model on ImageNet 1K [17] for the task of 64 × 64 → 256 × 256 super-resolution. We use the dev split for evaluation. During training, we preprocess the images by removing those where the shorter side is less than 256 pixels. The remaining images are then centrally cropped and resized to 256 × 256. The low-resolution images are then generated by downsampling the 256×256 images using bicubic interpolation with anti-aliasing enabled. For a fair comparison with SR3 [79], we train our CrossFlow with 505M parameters (compared to 625M parameters in SR3). Our model is trained for 1M training steps with a batch size of 512 and a learning rate of 1 × 10−4, including 5,000 warm-up steps.

### B. Additional Experimental Results B.1. GenEval Performance

To compare with recent text-to-image models on GenEval, we report the overall score and task-specific scores in Tab. 7. Our model achieves comparable performance to state-ofthe-art models such as LDMv2.1 [77], LDM-XL [67], and DALL·E 2 [73]. This demonstrates that directly evolving from text space to image space with our approach is a simple and effective solution for text-to-image generation, indicating a novel and promising direction for state-of-the-art media generation.

Single Two

Attribute

Method Overall

Counting Colors Position

Object Object binding

DALL·E 2 [73] 0.52 0.94 0.66 0.49 0.77 0.10 0.19 LDMv1.5 [77] 0.43 0.97 0.38 0.35 0.76 0.04 0.06 LDMv2.1 [77] 0.50 0.98 0.51 0.44 0.85 0.07 0.17 LDM-XL [67] 0.55 0.98 0.74 0.39 0.85 0.15 0.23 PixArt-α [11] 0.48 0.98 0.50 0.44 0.80 0.08 0.07 LDMv3 (5122) [23] 0.68 0.98 0.84 0.66 0.74 0.40 0.43

CrossFlow 0.55 0.98 0.72 0.39 0.82 0.18 0.21

- Table 7. GenEval comparisons. Our model achieves comparable performance to state-of-the-art models such as LDM-XL and DALL·E 2, suggesting that CrossFlow is a simple and promising direction for state-of-the-art media generation.

[Figure 62]

𝛼=0.25 𝛼=0.33 𝛼=0.67 𝛼=1 𝛼=1.5 𝛼=2 𝛼=3

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

Figure 6. Arithmetic operation with different scaling terms. We show images generated by : VE(‘a white dog’)+αVE(‘a hat’)

Arithmetic Operation Success Rate (%)

Addition 95.3 Subtraction 92.7 Combination 87.5

Overall 91.4

- Table 8. Success rate of arithmetic operation We select 1,000 prompts from COCO-val to evaluate the success rate of arithmetic operations. The detection model is used to determine whether the target objects have been successfully added or removed. “Combination” refers to multiple operations involving a combination of both “addition” and “subtraction”.

#### B.2. Analysis of Arithmetic Operations

Our model encodes text into a continuous latent space with semantic structure. Prior work, such as word2vec [62], has shown that latent arithmetic can emerge without explicit training. Arithmetic on these latents effectively retains added and removes subtracted textual information, which our Flow Matching then correspondingly maps to images. We analyze latent arithmetic operations in more detail here. First, we consider addition (+) and test different scaling factors (Fig. 6), showing how they control the amount of information added or removed in the generated image.

In addition, we show qualitatively that the arithmetic works well across diverse concepts and multiple operations in Tab. 8. Specifically, we select 1,000 prompts from COCO-val to test arithmetic operations. A detection model confirms that 91.4% of the objects are accurately added or removed from the generations, providing quantitative evidence of effective feature disentanglement.

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

Input Ours Input Ours

Figure 7. Qualitative examples for zero-shot depth estimation. The input images in the first two rows are from the NYUv2 dataset, while the input images in the last row were generated by our T2I model. Our model provides robust zero-shot depth estimation across domains, whether indoor or outdoor, synthetic or real.

#### B.3. Zero-shot Depth Estimation

We also evaluate CrossFlow on zero-shot depth estimation. Following Marigold [40], we train our model on Hypersim [76] and Virtual KITTI [10], and evaluate our model on 5 real datasets that are not seen during training: KITTI [26], NYUv2 [82], ETH3D [80], ScanNet [14], and DIODE [89]. We follow Marigold [40] to prepare the training and testing data. Our model with 527M parameters is trained for 150K training steps, with a batch size of 512 and a learning rate of 1 × 10−4 with 5,000 warm-up steps. The results are reported in Tab. 9. Qualitative examples are provided in Fig. 7. Without specific design, CrossFlow achieves comparable or even superior performance compared to state-ofthe-art methods, demonstrating the general-purpose nature of our approach on various cross-modal tasks.

#### B.4. Image Super-resolution

We provide qualitative examples for image super-resolution in Fig. 8. Unlike traditional methods, which typically evolve from Gaussian noise and rely on concatenating upsampled low-resolution images as conditioning, our approach takes a more direct route: we demonstrate that it is possible to evolve a low-resolution image directly into a high-resolution image, eliminating the need for additional concatenation conditioning.

#### B.5. Ablation Study

Text compression. In this section, we show that we can compress the input text embedding x ∈ Rn×d into z0 ∈ Rh×w×c (e.g., 77 × 768 CLIP tokens to 4 × 32 × 32 latents for 256px generation, 14.4× compression) with a standard encoder or the proposed Variational Encoder while preserve most of the input information. We report the per-token re-

KITTI NYUv2 ETH3D ScanNet DIODE AbsRel ↓ δ1 ↑ AbsRel ↓ δ1 ↑ AbsRel ↓ δ1 ↑ AbsRel ↓ δ1 ↑ AbsRel ↓ δ1 ↑

Method # Training samples

DiverseDepth [95] 320K 0.117 0.875 0.190 0.704 0.228 0.694 0.109 0.882 0.376 0.631 MiDaS [74] 2M 0.111 0.885 0.236 0.630 0.184 0.752 0.121 0.846 0.332 0.715 LeReS [96] 300K + 54K 0.090 0.916 0.149 0.784 0.171 0.777 0.091 0.917 0.271 0.766 Omnidata [21] 11.9M + 310K 0.074 0.945 0.149 0.835 0.166 0.778 0.075 0.936 0.339 0.742 HDN [98] 300K 0.069 0.948 0.115 0.867 0.121 0.833 0.080 0.939 0.246 0.780 DPT [75] 1.2M + 188K 0.098 0.903 0.100 0.901 0.078 0.946 0.082 0.934 0.182 0.758 Marigold [40] 74K 0.060 0.959 0.105 0.904 0.071 0.951 0.069 0.945 0.310 0.772

CrossFlow (Ours) 74K 0.062 0.956 0.103 0.908 0.085 0.944 0.068 0.942 0.270 0.768

- Table 9. Zero-shot depth estimation. Baseline results are reported by Marigold [40]. We follow Marigold and train our CrossFlow on the same datasets, i.e., Hypersim [76] and Virtual KITTI [10]. We highlight the best, second best, and third best entries. With just a unified framework, CrossFlow achieves comparable or even superior performance on complex zero-shot depth estimation, demonstrating the general-purpose nature of CrossFlow on various cross-modal tasks.

Input Ours Input Ours

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

- Figure 8. Qualitative examples for image super-resolution.

Text encoder Recon. accuracy (%)

Text Encoder (1 × 1024) 95.12 Text Variational Encoder (1 × 1024) 94.53

- Table 10. Ablation on text compression. Both text encoder and Text Variational Encoder preserve most of the input information, despite the large compression ratio (77×768 → 1×1024, 14.4×).

with the help of the indicator. This allows our model to support standard CFG. Then, in the middle five columns, we show the images generated with different CFG scaling factors. Similar to the standard flow matching model, the CFG can significantly improve the image quality. Finally, in the last two columns, we compare our CFG with indicator to Autoguidance, using the same scaling factor. Like our approach, Autoguidance also enables low-temperature sampling for models without explicit conditioning. We observe that our CFG with indicator produces higher-fidelity images compared to Autoguidance.

### C. Additional Qualitative Examples

We provide additional qualitative examples for text-toimage generation here. Specifically, we first provide 512 × 512 images generated by our CrossFlow in Fig. 10. Next, we provide more examples for linear interpolation in latent space (Fig. 11 and Fig. 12) and arithmetic operation in latent space (Fig. 13).

construction accuracy, computed by cosine similarity, in Tab. 10. The results show that both methods are effective at preserving the input information, achieving high reconstruction accuracy despite a large compression ratio.

CFG indicator. In Fig. 9, we study the effect of our CFG with indicator, and then compare our approach with Autoguidance [39]. The left two columns show the images generated when the indicator 1c = 0 (for unconditional generation) and 1c = 1 (for conditional generation). It shows that despite generating an image by directly evolving from the text space into the image space without explicit conditioning, our model can still perform unconditional generation

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

Indicator 1c= 0

Indicator 1c= 1

Ours

Ours

Ours

Ours

Ours

Autoguidance

scaling factor ω = 2

scaling factor ω = 3

scaling factor ω = 4

scaling factor ω = 5

scaling factor ω = 6

scaling factor ω = 6

(uncond)

(cond)

- Figure 9. Ablation on CFG with indicator. The first two columns show the images generated when the indicator 1c = 0 (for unconditional generation) and 1c = 1 (for conditional generation), demonstrating that CrossFlow can still perform unconditional generation with the help of the indicator, thereby allowing for the use of standard CFG. We then demonstrate the improvement provided by CFG (middle five columns) and compare it with Autoguidance (last two columns). Prompts used to generate the images: ‘a corgi wearing a red hat in the park’,‘a cat playing chess’,‘a pair of headphones on a guitar’,‘a horse in a red car’

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

‘a glass of orange juice to the right of a plate with buttered toast on it’

‘a teddy bear on a skateboard in times square’

‘a painting of a rocket lifting off from the city’

‘a teddy bear sitting on a yellow toy pickup truck’

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

‘a black dog is playing chess with a white dog’

‘three birds standing on a wire stock’

‘five frosted glass bottles’ ‘two cats doing research’

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

‘a cute illustration of a horned owl with a graduation cap and diploma’

‘a close-up of milk pouring into a white bowl against a black background’

‘a close-up of the eyes of an owl’

‘a black and white landscape photograph of a black tree’

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

‘A spaceship made of cardboard’

‘a cup of cloud’

‘a Tyrannosaurus Rex roaring ‘a white goat in my room’ in front of a palm tree’

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

‘a cartoon of a train going to the moon’

‘a impressionistic painting of a lion reading books’

‘a watercolor painting of a tree and a building’

‘an abstract painting of a waterfall’

###### Figure 10. Qualitative examples for text-to-image with CrossFlow.

|[Figure 145]|
|---|

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

|[Figure 185]|
|---|

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

‘A white dog wearing a white and black helmet riding a bike in the park’(topleftinbluebox)

‘An orange cat wearing sunglasses on a ship’(bottomrightinorangebox)

|[Figure 193]|
|---|

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

|[Figure 233]|
|---|

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

‘A robot cooking dinner in the kitchen’(topleftinbluebox) ‘A panda eating hamburger in the classroom’(bottomrightinorangebox)

- Figure 11. Linear interpolation in latent space. We show images generated by linear interpolation between two text latents (i.e., interpolation between z0). Images generated by the first and second text latents are provided in the top-left and bottom-right corners.

|[Figure 241]|
|---|

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

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

|[Figure 281]|
|---|

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

‘A corgi wearing a red hat in the park’(topleftinbluebox) ‘A teddy bear dressed in black wizard hat and robes sitting on the bed’(bottomrightinorangebox)

- Figure 12. Linear interpolation in latent space. We show images generated by linear interpolation between two text latents (i.e., interpolation between z0). Images generated by the first and second text latents are provided in the top-left and bottom-right corners.

z0= VE(‘acorgiwitha

red hat in the park’)

z0=VE(‘ahat’) z0=VE(‘acorgiwitharedhatinthe

park’) + VE(‘book’) - VE(‘a hat’)

z0=VE(‘red’) z0=VE(‘yellow’) z0 =VE(‘aredcar’)-VE(‘red’)

+ VE(‘yellow’)

z0= VE(‘book’)

z0=VE(‘aredcar’)

z0=VE(‘car’) z0=VE(‘bike’) z0=VE(‘awhitedoginacar’)

- VE(‘car’) + VE(‘bike’)

= VE(‘a white dog in a car’)

z0

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

- Figure 13. Arithmetic in text latent space. We map the text into the text latent space, perform arithmetic operations to obtain new latent representation, and use the resulting representation to generate the image. Latent z0 used to generate each image is provided at the bottom.

