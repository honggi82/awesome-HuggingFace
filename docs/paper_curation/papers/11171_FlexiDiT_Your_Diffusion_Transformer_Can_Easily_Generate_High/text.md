# arXiv:2502.20126v1[cs.LG]27Feb2025

## FlexiDiT: Your Diffusion Transformer Can Easily Generate High-Quality Samples with Less Compute

Sotiris Anagnostidis†,§,∗ Gregor Bachmann†,§ Yeongmin Kim†,‡ Jonas Kohler† Markos Georgopoulos† Artsiom Sanakoyeu† Yuming Du† Albert Pumarola† Ali Thabet† Edgar Sch¨onfeld†

100.0 % FLOPs 84.3 % FLOPs 68.6 % FLOPs 52.9 % FLOPs 37.2 % FLOPs

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

Figure 1. We flexify DiTs and adjust the compute per diffusion step, generating high-quality samples with significantly less compute.

### Abstract

Despite their remarkable performance, modern Diffusion Transformers (DiTs) are hindered by substantial resource requirements during inference, stemming from the fixed and large amount of compute needed for each denoising step. In this work, we revisit the conventional static paradigm that allocates a fixed compute budget per denoising iteration and propose a dynamic strategy instead. Our simple and sample-efficient framework enables pre-trained DiT models to be converted into flexible ones — dubbed FlexiDiTallowing them to process inputs at varying compute budgets. We demonstrate how a single flexible model can generate images without any drop in quality, while reducing the required FLOPs by more than 40% compared to their static counterparts, for both class-conditioned and textconditioned image generation. Our method is general and agnostic to input and conditioning modalities. We show

∗Work done during an internship at Meta GenAI. †Meta GenAI. §ETH Z¨urich. ‡KAIST. Correspondence to: sanagnos@ethz.ch

how our approach can be readily extended for video generation, where FlexiDiT models generate samples with up to 75% less compute without compromising performance.

### 1. Introduction

Diffusion models [87] have recently become the core building block of major improvements in image generation [10, 13, 24, 90, 107]. These models gradually denoise a random sample xt, drawn typically from pnoise = N(0,I), by iteratively calling a neural network trained to reverse a pre-defined noise corruption process pθ(xt−1|xt). This process enables the generation of samples from the desired distribution pdata.The remarkable performance of these models is closely tied to the amount of computational resources invested in them, as evidenced by established scaling laws [52]. The Transformer architecture [95] has proven to be highly scalable across various modalities, leading to its adoption in diffusion models, in the form of the recent Diffusion Transformer [78]. In DiTs, the denoising pro-

###### Generative Denoising Process

| | | |
|---|---|---|
| | | |
| | | |
| | | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |

0.3

0.2

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

0.2

DenoisedImage

###### LPIPS

###### L2

0.1

0.1

0.0

0.0

pdata pnoise

pdata pnoise

L2

LPIPS

1.0

0.25

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |

0.9

0.20

SpatialFrequency

DreamSim

0.15

0.8

###### SSIM

0.7

0.10

0.6

0.05

0.5

0.00

pdata pnoise

pdata pnoise

SSIM

DreamSim

High Pass

Low Pass

| |
|---|

Figure 2. Diffusion can be viewed as spectral autoregression [25]. Left: Diffusion and its effect on the spatial frequency of images. Right: To investigate the role of different frequency components in image generation, we apply a low or high pass filter to a single diffusion step update (while keeping all other updates unchanged). With all other sources of randomness fixed, we compare the generated samples with and without filtering using LPIPS [110], L2 distance of the pixels, SSIM [99] and DreamSim [32]. Notably, the influence of low and high pass filters varies depending on whether they are applied early or late in the denoising process.

cess pθ(xt−1|xt) is parametrized using Transformer blocks instead of traditional convolutional layers. As popularized in Vision Transformers [26], (latent) images of dimension h × w are divided into patches of size p × p, which serve as input tokens that are transformed via a series of attention and feed-forward layers. The use of DiTs offers two key advantages: (i) a unified architecture and input processing framework that facilitates multimodal applications and generalization to other domains, such as audio and video; and (ii) exceptional scalability due to their signal propagation characteristics and efficient training on modern hardware.

The computational complexity of a DiT with hidden dimension d and depth L is O(LNd2 +LN2d), where N corresponds to the total number of tokens1. Given T steps of the diffusion process, function calls to the same monolithic DiT model are repeated for all T steps, and the total amount of compute is thus uniformly allocated. However, image generation via diffusion exhibits distinct, non-uniform, and temporally — with respect to the noise process — varying characteristics. Consistent with intuition, prior work has observed that high-level image features tend to emerge at early generative diffusion steps (i.e., large t’s). In contrast, later steps refine and progressively generate high-frequency and detailed visual features [14]. We illustrate further differences during the denoising generation in Fig. 2. Concisely, different denoising steps have profoundly different influence on high and low-level features of the resulting images.

In theory, different denoising methods and models can be employed for each step t, i.e. one can use separate parameters θt to learn pθ

(xt−1|xt). While this notion has been previously explored in the literature, prior works propose computationally intensive solutions, such as training

t

1For high-resolution image generation — even when diffusion is performed in a latent space — the second term O(LN2d) that corresponds to the attention operations, can quickly become the main bottleneck.

separate expert denoisers for different t’s [5] or using model cascades [91]. These approaches face two significant limitations: (i) they require multiple models to be managed during inference, increasing memory demands. This can quickly become a major issue, especially as current trends favor the use of larger and larger models. Moreover, (ii) using separate models restricts the opportunity for knowledge sharing across steps. Although one can treat denoising steps independently, the denoising process itself retains certain shared characteristics across steps. This inherent smoothness implies that separate models would need to learn these shared properties individually. In this work, we address these challenges and instead propose to perform different denoising steps with varying levels of compute using a single model.

In summary, our contributions are the following:

- • We present a simple framework that flexifies DiT models, allowing them to convert samples into different sequences of tokens by adjusting the patch size. Processing samples as different sequences enables us to control the compute budget being used for each denoising step.
- • By leveraging specific image characteristics at different denoising steps, we demonstrate that strategically allocating less compute to certain steps can yield significant computational savings (over 40%) without compromising the quality of generated samples for both classconditioned and text-conditioned image generation. We also show how denoising based on a larger patch size can serve as a more effective guidance signal.
- • We illustrate the versatility of our framework by extending it to other modalities. For video generation, we achieve substantial computational savings (up to 75%) with no considerable drop in performance or conceptual differences in the generated samples.

### 2. Background

For simplicity, we limit the discussion here to images, and detail later changes due to different modalities. DiTs use a Transformer encoder to process image patches as tokens. Hereafter, we refer as tokenization to the process of converting a (latent) image into a series of tokens and as de-tokenization to the opposite process, of transforming a series of tokens back into an image. Given an image of size h × w and a chosen patch size2 p, an input image is cropped into non-overlapping patches of dimensions Rp×p×c

[Figure 16]

FLOPs

[Figure 17]

FLOPs

[Figure 18]

(large patch size)

[Figure 19]

(small patch size)

Patch size

Figure 3. Tokenizing images into patches.

in, where cin denotes the number of channels of the input image. The total number of tokens is then equal to N = (h/p) × (w/p). This (flattened) sequence of patches is then projected using a linear layer Membed with weights Rp∗p∗c

in×d and potential biases Rd. This tokenization process is equivalent to performing a 2D convolution, where the kernel size and stride are both equal to p × p. The embedded N tokens are then processed using L Transformer encoder layers. The output tokens of dimension RN×d are projected back to the image space with a linear de-embedding layer Mde-embed with weights Rd×c

out∗p∗p and potential biases Rc

out∗p∗p. Here cout denotes the number of output channels, typically cout = 2cin if the prediction includes the variance, else cout = cin. DiTs adhere to the scaling properties of Transformers, as demonstrated in various other applications [4, 37, 41, 52, 109].

DiTs are an increasingly popular alternative to convolutional networks for denoising corrupted images during generation, i.e. modeling pθ(xt−1|xt). Diffusion defines two Markov chains, the forward and the backward process. During the forward process, Gaussian noise3 is added to samples of the real distribution x0 ∼ q(x0) [40]:

T

q(xt|xt−1), where

q(x1:T|x0) =

t=1

q(xt|xt−1) = N(xt−1; 1 − βtxt−1,βtI). (1)

In the backward process, samples are drawn from a Gaussian noise distribution p(xT) = pnoise = N(0,I) and then gradually denoised, using Tweedie’s Formula [28]:

- 2In reality, different patch sizes can be considered along the height and the width dimensions. We will however refrain from doing that.
- 3Although we focus on Gaussian noise here, other corruptions apart from Gaussian noise have also been analysed [6, 20, 75].

1

pθ(xt−1|xt), where

pθ(xT:0) = p(xT)

t=T

pθ(xt−1|xt) = N(xt;µθ(xt,t),Σθ(xt,t)). (2)

Typically, a single model is used to model the prediction pθ(xt−1|xt) for every t.

### 3. Flexible Diffusion Transformers

We flexify DiTs, via small architectural changes, that allow them to process images as different length sequences, by adjusting the patch size p used in tokenization. Flexible tokenization of images has been utilized before for singlestep inference applications [9] and to accelerate training [3]. We instead propose to use different patch sizes at different steps in the denoising process of the same image. This is based on the following intuition:

Early steps focus on low-frequency details, which can be performed with bigger patch sizes at the same quality.

Changing the patch size p directly affects the total number of tokens (recall that N = (h/p) × (w/p)) and thus the overall compute required for a function evaluation. Hence, by leveraging bigger patch sizes only for early steps, we can reduce the overall generation time while maintaining both low and high-frequency details in the produced images.

We obtain a flexible model — coined FlexiDiT — by modifying and fine-tuning a pre-trained DiT, which enables it to process and understand images with new patch sizes. In our experiments we focus on efficiency, so newly added patch sizes are always larger than the one of the pre-trained model, leading to fewer tokens and thus a smaller computational footprint. The single FlexiDiT model can be instantiated in different modes depending on the selected patch size. We will refer to instances of the model that use the original and smaller patch size ppowerful as powerful, compared to using a weak model with a larger patch size pweak. To simplify the discussion, we largely ignore how additional conditioning may be applied for now, and instead refer to specific implementation details in the experiments section.

DiTs — as any Transformer — can process sequences of any arbitrary length N. Fundamentally, we just need to modify (i) how tokenization and de-tokenization are performed to ensure that the input and output representation space stay unchanged and (ii) ensure that the model can interpret these different length sequences. In the following, we discuss and outline two different ways that FlexiDiTs can be derived, based on whether the forward pass for the pre-trained model (we refer to this as the target model) is preserved exactly or not. In both cases, we emphasize that any additional fine-tuning and new parameters are minimal.

[Figure 20]

| |Use weak model when Use powerful model when| |
|---|---|---|
| |denoising these steps denoising these steps| |
| | | |
| | | |
| | | |

oftheweakandpowerfulmodel

betweenerrorsinpredictionsL1

10 1

10 2

10 3

10 4

10 5

pnoise pdata

Denoising steps

- Figure 4. Left: We flexify DiTs by allowing them to process images with more patch sizes, by changing the lightweight embedding and de-embedding layers. We showcase this for a class-conditioned ImageNet model. Right: We plot the difference in predictions between a weak and a powerful model. For the first denoising steps, differences are small, and thus using the weak model there allows accelerated generation without performance degradation.

In all our experiments, training is stable, and no special tricks are necessary, while the total compute for fine-tuning is less than 5% of the original pre-training compute.

#### 3.1. Shared Parameters for all Sequences

When the original training data is available, it becomes possible to fine-tune the pre-trained model while preserving its performance, along with its existing abilities, biases, and potential safety features. We demonstrate that this approach enables processing images with varying patch sizes by introducing only minimal additional trainable parameters, as shown in Fig. 4 (left). Below, we detail the specific components of the architecture that require adaptation.

Tokenization: We introduce a new embedding layer Membedflex , for some underlying patch size p′ with new weights wembedflex ∈ Rp

′∗p′∗cin×d and biases bflexembed ∈ Rd. Then, when instantiating a FlexiDiT with a patch size pcurrent, we project these weights with a fixed projection matrix Qembed ∈ Rp

current∗pcurrent×p′∗p′ to the desired shape, and perform the 2D convolution with kernel size and stride equal to pcurrent × pcurrent. We use as Qembed the pseudo-inverse of the bi-linear interpolation projection, as this leads to better norm preservation of the output [9]. We initialize the weights based on the pre-trained parameters and the pre-trained patch size as wembedflex = Qembed†wembed4 and bflexembed = bembed, preserving the functional form of the model for the pre-trained patch size. When adding positional encodings, we identify for each patch its pixel coordinates in the original image [15, 103]. We additionally introduce a patch size embedding for each of the patch sizes used by the model, which we add to every token in the sequence, and patch size dependent layer-normalization layers. These layers help with signal propagation [36, 77, 104] by preserving the norms of the activations and let the model recover

4Here † denotes the pseudo-inverse. All projection matrices Q multiply each channel separately.

its expressivity [49, 100].

De-tokenization: We similarly adapt the de-embedding layer Mde-embed. For an underlying patch size p′, we define a new layer Mde-embedflex with weights wde-embedflex ∈ Rd×c

out∗p′∗p′. Depending on the current patch size pcurrent in the neural network evaluation, we project with a fixed projection matrix Qde-embed ∈ Rp

out∗p′∗p′ and biases bflexde-embed ∈ Rc

′∗p′×pcurrent∗pcurrent to wde-embedflex Qde-embed and

bflexde-embedQde-embed. Again, we use as Qde-embed the pseudoinverse of the bi-linear interpolation, now with flipped di-

mensions. We initialize the new parameters as wde-embedflex = wde-embedQde-embed† and bflexde-embed = bde-embedQde-embed†. In total, less than 0.005 % of auxiliary parameters are introduced to attain a FlexiDiT for the models tested in this case.

#### 3.2. Different LoRAs for each Sequence

In practice, however, pre-training often requires extensive computational resources and may span multiple stages using various datasets, which may not always be accessible, even for models with open weights. In such cases, fine-tuning can have unintended effects, potentially diminishing model capabilities [46] or compromising safety guarantees [80]. For such situations, where it is essential to preserve the original forward pass of a pre-trained model, we demonstrate a method that enables fine-tuning across different patch sizes with minimal additional compute and a small supplementary dataset. Fig. 5 illustrates our approach.

We perform similar changes to the model, but instead of training the DiT block parameters, we freeze the original weights and add trainable LoRAs [42]. These LoRAs are specialized for each new patch size and activated only when our FlexiDiT is instantiated with that. There are no LoRAs for the patch size of the pre-trained model. For the tokenization and de-tokenization layers, we simply add new layers Membed,Mde-embed for each new patch size. As before,

Frozen weights Trainable weights

Powerfull Patch De-embedder

Weak Patch De-embedder

Powerfull Patch De-embedder

Weak Patch De-embedder

Powerfull Patch De-embedder

Weak Patch Deembedder

A pyramind built by an advanced lost civilization

Shared parameters

[Figure 21]

Merged LoRAs Pointwise Feedforward

Pointwise Feedforward

Feedforward LoRAs

Pointwise Feedforward

Feedforward LoRAs

Pointwise Feedforward

Text Encoder

Multi-Head Cross-Attention

Multi-Head Cross-Attention

Multi-Head Cross-Attention

Multi-Head Cross-Attention

[Figure 22]

Merged-LoRAs Multi-Head Self-Attention

Multi-Head Self-Attention

Self-Attention LoRAs

Multi-Head Self-Attention

Self-Attention LoRAs

Multi-Head Self-Attention

[Figure 23]

Image Encoder

Powerfull Patch Embedder

Weak Patch Embedder

Powerfull Patch Embedder

Weak Patch Embedder

Powerfull Patch Embedder

Weak Patch Embedder

Inference with LoRAs

Inference without LoRAs

Training

- Figure 5. We preserve the functional form of the target model for the pre-trained patch size and add new trainable parameters (LoRAs) for each additional patch size we want to fine-tune the model to operate with. We showcase this for a text-to-image/video model that uses cross-attention for text conditioning. We find that freezing cross-attention layers without any additional LoRAs works the best. During inference, we can either keep the LoRAs unmerged (Inference with LoRAs) leading to a slight FLOPs increase that depends on the LoRAs’ dimensions, or create different copies of the model for each patch size, by merging the LoRAs (Inference without LoRAs). The latter leads to additional memory requirements. FLOPs and parameter numbers on the right correspond to our flexible T2I Emu model.

we use a patch size embedding, but only for the new patch sizes. Note that functional preservation is imposed for the pre-trained patch size, i.e. inference using only the powerful model generates exactly the same samples. We finetune by using the predictions of the original model (powerful model) as labels to distill knowledge [38] for the new patch sizes, i.e. we train to minimize:

Et,x

t∥ϵθ(xt−1|xt;ppowerful) − ϵθ(xt−1|xt;pweak)∥2.

We find that this leads to improved performance, faster convergence [18] and better alignment between predictions of different patch sizes, which can be important given potential discrepancies in the data used during pre-training and our fine-tuning. Here we use ϵθ to denote the model’s prediction parametrizing the distribution pθ. Note that ϵθ(xt−1|xt;ppowerful) has no trainable parameters.

During inference, we have two options: (i) keep the new LoRA parameters unmerged, which incurs a minor additional computational cost, or (ii) merge these weights into a copy of the original model parameters, with some additional memory cost. If we keep the LoRA parameters of dimension dlora unmerged, the computational complexity of the corresponding linear layer with input and output (dinput,doutput) will change from Ndinputdoutput to N(dinputdoutput + dinputdlora + dloradoutput). If we merge LoRAs, there is no computational overhead, but the new merged parameters need to be kept in memory. Depending on the model requirements and the available resources, one can choose between the two options. We note that compute overhead by keeping LoRAs unmerged is minimal, see also Fig. 5 (right). For all models tested, additional parameters

in this case are less than 5% of the original model parameters. Further details and ablations can be found in App. C.

#### 3.3. Inference Scheduler

At every denoising step, we need to decide how to instantiate our FlexiDiT, i.e. which patch size to use. Following our intuition, we find that for early steps of the denoising process, weak and powerful models produce similar predictions, and thus using the weak model preserves quality while reducing computational complexity, as also seen in Fig. 4 (right). We therefore propose an inference scheduler that, starting from random noise pnoise, first denoises using a weak model for the first Tweak steps and switches to the powerful model for the last Tpowerful = T − Tweak steps. By adjusting the steps performed by the weak model Tweak, we can adjust the amount of compute that we are saving. In practice, unless otherwise mentioned, we fine-tune models to process images with one additional patch size. We choose this patch size, corresponding to the weak model, as 2× larger than the one corresponding to the powerful model. Then, the sequence length corresponding to tokenizing with the new patch size is 4× smaller, and thus compute required for the powerful model is > 4× compared to the weak model.

#### 3.4. Generation Guidance

For conditional generation, classifier-free guidance (CFG) is typically employed [29, 39] to enhance sample quality. This entails performing two neural function evaluations (NFEs) (or one NFE with twice the batch size) to compute predictions with and without the conditioning c, i.e. ϵθ(xt−1|xt,c) and ϵθ(xt−1|xt,∅). Sampling can

[Figure 24]

[Figure 25]

[Figure 26]

- Figure 6. Left: As the weak model is used more extensively during generation, compute benefits increase, but at the cost of some performance degradation. Middle: The optimal CFG scale varies depending on the extent to which the weak model is used. Each line corresponds to an inference scheduler that applies the weak model for a different proportion of denoising steps. Right: Benefits from our inference scheduler are orthogonal to performing a smaller overall number of diffusion steps. We plot FID for different overall number of steps T and different number of weak steps Tweak, using in every case the DDPM scheduler.

then take place as ϵθ(xt−1|xt,∅) + scfg(ϵθ(xt−1|xt,c) − ϵθ(xt−1|xt,∅)), where scfg is the guidance scale. Recent work [53] has shown that using a smaller or less welltrained version of the model rather than an unconditional model can lead to better guidance signal [1, 83]. We adapt these findings in our setting, leading to better generation quality without the need to train or deploy additional models. For each denoising step, given a patch size used for the conditional pcond and a patch size used for guidance puncond, we compute:

 

ϵθ(xt−1|xt,∅;puncond) + scfg(ϵθ(xt−1|xt,c;pcond)− ϵθ(xt−1|xt,∅;puncond)), if pcond = puncond ϵθ(xt−1|xt,c;puncond) + scfg(ϵθ(xt−1|xt,c;pcond)− ϵθ(xt−1|xt,c;puncond)), if pcond < puncond



In this setup, we use the powerful model for the conditional prediction and leverage the weak model’s output as guidance. Unlike traditional approaches, our method applies guidance based on the conditional prediction from the weak model. Our guidance scheme requires performing inference using both the weak (for the unconditional) and the powerful (for the conditional) model, for some denoising steps. We show in Fig. 12 (appendix) how this can be efficiently implemented, making use of packing [23]. Depending on the guidance signal used, optimum generation quality can vary with respect to the guidance scale scfg. This will become more apparent in the following experiments.

### 4. Experiments

For clarity, we present efficiency gains with respect to FLOPs and point to Section 4.4 for a detailed analysis of the relationship between FLOPs and latency.

#### 4.1. Class-Conditioned Image Generation

We fine-tune models (DiT-XL/2) on ImageNet, using the same setup as in [78]. Since training data is publicly avail-

able, we fine-tune pre-trained models using the same parameters for all sequences, without the use of LoRAs, as described in Sec. 3.1. During training, we randomly noise images according to Eq. (1), and learn to denoise using one of the available patch sizes. We primarily report FID and point to App. B for more experiments and different metrics. In practice, we use a pre-trained model with a patch size of 2 (powerful), that we fine-tune to also process images with a patch size of 4 (weak). Since we are fine-tuning the powerful model, we can also “teach” it how to correct specific mistakes made by the weak model, accumulated in the backward process during the first Tweak steps. We provide more details in App. B.1 on how to reduce this exposure bias [57, 76]. Unless otherwise mentioned, reported metrics are computed by generating images at resolution 256×256, using 250 steps of the DDPM scheduler [40, 78].

Compute gains. We generate images with our FlexiDiT and the proposed inference scheduler, varying the amount of compute by adjusting the number of initial denoising steps Tweak performed with the weak model. For each level of compute, we report the FID of the generated images in Fig. 6 (left and middle). In general, performing a few steps with the weak model (60-100% of baseline compute) leads to no drop in performance. Saving even more compute is possible, albeit at the cost of a minor drop in the quality of the generated images. When using only the powerful mode of our FlexiDiT, we get the same performance — FID-50k = 2.25 — as the pre-trained DiTXL/2 model — FID-50k = 2.27. Thus, fine-tuning for more patch sizes does not reduce the capacity of the model with respect to the pre-trained one. We also show that other inference schedulers, such as starting with the powerful model and switching to the weak model, lead to worse results (appendix Fig. 19), validating our intuition.

Relation between T and Tweak. Naturally, the question arises whether doing fewer overall diffusion steps T leads to

35

Model (compute %) FID ↓ CLIP ↑ VQAScore ↑

Steps (Powerful/Weak) Votes (in %) Ours Baseline Win Tie Lose

30

T2I Transf. (100 %) 14.75 25.60 63.29 T2I Transf. ( 86 %) 14.72 25.62 63.30 T2I Transf. ( 72 %) 14.77 25.63 63.40 T2I Transf. ( 58 %) 14.71 25.58 63.26

MSCOCOFID-10k

50 (40,10) 42 (42,0) 33.5 42.5 24.0 50 (30,20) 34 (34,0) 35.5 41.5 23.0 50 (20,30) 26 (26,0) 35.5 44.0 20.5 50 (10,40) 18 (18,0) 43.0 41.5 15.5

25

20

Emu (100 %) 26.00 26.05 70.17 Emu ( 84 %) 25.96 26.06 70.19 Emu ( 69 %) 26.00 26.08 70.37 Emu ( 53 %) 26.10 26.07 70.09

15

Baseline Baseline 4.5 CFG

FLOPs % compared to baseline

We asked 8 annotators to express preferences between pairs of images for a given prompt. We generate images for 200 prompts and collect 3 different votes per each pair of images, collectively 2400 votes.

92.9% 85.8% 78.8% 71.7% 64.6% 57.5%

10

21 22 23 24 25 26 CLIP score (ViT/L-14)

- Figure 7. Left: We plot FID vs CLIP score for images generated with different CFG scales using the T2I Transf. model (we refer to the appendix for results regarding our Emu model). The red line represents images generated with varying CFG scales using only the (powerful) target model. By employing our dynamic scheduler, we can match image quality in terms of both FID and text alignment while significantly reducing compute requirements. Middle: Our flexible models can match the baseline (for a fixed pre-defined guidance scale scfg) across benchmarks, with significantly less compute. Right: Human study results show votes indicating a win, tie, or loss for our method compared to a baseline, which corresponds to running the pre-trained model (only the powerful model) for fewer steps. Comparisons are between Emu inference modes that require approximately equal FLOPs and time.

the same efficiency improvements compared to performing more steps with the weak model. After applying the DDPM scheduler for a different number of overall diffusion steps T, we plot in Fig. 6 (right) the efficiency gains across them. Results indicate that gains from performing weak steps are orthogonal to performing fewer overall diffusion steps. In other words, more steps are required to achieve better image fidelity, but performing some of the initial steps with our weak model is sufficient to achieve the targeted fidelity. In App. B.3, we provide further insights on how the predictions of the weak model closely align with the ones of the powerful model. This similarity supports parameter sharing, which not only reduces resource requirements during inference but also enables faster convergence during finetuning.

#### 4.2. Text-conditioned Image Generation

The universality of Transformers and the holistic view of different input sources as sequences of tokens implies that the generalization of the architecture to more modalities and a variety of different inputs is straightforward. This is also the case for DiTs, which have been already extended to accommodate text conditioning [7, 15], video generation [72] and speech synthesis [67]. We showcase how our framework can be applied out of the box to state-of-the-art text-toimage (T2I) model architectures. T2I DiTs have the same architecture as class-conditioned DiTs, with the exception that conditioning is imposed via cross-attention.

We fine-tune T2I models, by introducing new parameters in the form of LoRAs. Specifically, we use a DiT following PIXART [15] — we refer to this as T2I Transf. — generating 256 × 256 images and a 1.7B DiT based on EMU [19] — we refer to this as Emu — generating 1024 × 1024 images. Implementation details are provided in App. B.5, in short, we fine-tune both low and high-resolution target models with a pre-trained patch size of 2, to also support a patch

size of 4. For T2I Transf., we follow the inference scheduler protocol in [15], and use the DDPM solver for 100 steps. We point to the App. B for results with different solvers and number of steps (namely 20 steps of DPM, 25 steps of SAsolver, and a varying number of steps of the DDPM solver) showing that performance benefits are orthogonal to these choices. As before, we vary the percentage of denoising steps with the weak model for either the conditional or the guidance predictions.

CLIP-vs-FID. We report FID and CLIP score alignment for captions from the MS COCO dataset with varying CFG scale values in Fig. 7 (left). As before, we notice that lower compute versions of our model require a higher CFG scale to generate images that match similar FID vs CLIP score values of the baseline (full compute) model. We show that for a fixed CFG scale of the baseline model — we chose scfg = 4.5 for the T2I Transf. model as proposed in [15] and scfg = 6.0 for our Emu model — we can match the attained performance for less than 60% of the original compute. Detailed scores are given in Fig. 7 (middle).

Additional benchmarks. We perform additional evaluations on the final generated images. We follow [59] and present text alignment for visual question-answering (VQA). We generate images based on the captions of DrawBench, TIFA160, Pick-a-Pic, Winoground and report average VQA scores (we point to App. B for detailed results). We compare the baseline model (same CFG scales as before), against dynamic inference with our flexible models. Consistent with our previous findings, results indicate that performing a few denoising steps with our weak model generally generates high-quality images. We can again save up to 40% of compute without a performance drop. For highresolution images generated from our Emu model, we also embark on a human study, as shown in Fig. 7 (right). There we validate that human annotators prefer images from our dynamic scheduler instead of using similar compute, but

An astronaut running through an alley in Rio de Janeiro.

Baseline 70.4

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

70.4 70.4 70.5 70.3 70.4 70.4 70.4

70.4 70.4

70.5

| |
|---|

| |
|---|

70.2

| |
|---|

| |
|---|

70.0

VBenchscore(%)

69.5

A white and orange tabby cat is seen happily darting through a dense garden, as if chasing something. Its eyes are wide and happy as it jogs forward, scanning the branches, flowers, and leaves as it walks. The path is na rrow as it makes its way between all the plants. the scene is captured from a ground-level angle, following the cat closely, giving a low and intimate perspective. The image is cinematic with warm tones and a grainy text ure. The scattered daylight between the leaves and plants above creates a warm contrast, accentuating the cat s orange fur. The shot is clear and sharp, with a shallow depth of field.

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

69.0

Weak Model

Temporal (2,2,2)

Spatial (1,4,4)

68.7

68.5

10 % 25 % 50 % 75 % 100 % FLOPS % to baseline

- Figure 8. Left: Samples from our flexible Video DiT model using 25.2% of compute compared to the pre-trained baseline. Right: We perform a varying number of steps of the denoising process with a weak model, using either our spatial or our temporal weak model. Both weak models lead to significant compute savings with little to no degradation in performance.

this time uniformly allocated.

#### 4.3. Text-Conditioned Video Generation

Finally, we also explore other modalities, such as video generation. Text-to-video (T2V) generation typically follows the same setup as T2I [72, 79]. Given a video of dimensions Rf×h×w×c

in, where f is the temporal component corresponding to the number of frames, and a chosen patch size (pf,ph,pw) for each of the dimensions, the input (latent) video is cropped into non-overlapping spatialtemporal patches of dimensions Rp

f×ph×pw×cin. Patches are then embedded using now a 3D convolutional layer into tokens, subsequently transformed by the DiT blocks where every token attends to every other token in the sequence, and finally projected back to patches with a linear layer.

We perform the same changes as for our T2I models in Sec. 3.2, adding LoRAs for new patch sizes. We initialize tokenization and de-tokenization layers as before, and when the temporal patch size pf is increased, we duplicate weights along the temporal dimension. We fine-tune a 4.9B video model as the one in [79] — referred to as Video DiT and train with distillation as before. We generate videos of resolution (f × h × w) = (256,384,704) using 250 steps of the DDPM scheduler as in [79]. Diffusion is performed in a latent space that down-samples each dimension f,h,w by 8. More dimensions of the latent space now offer more possibilities in terms of determining the characteristics of our weak model. For our experiments, we fine-tune a pretrained model with patch size (pf,ph,pw) = (1,2,2) to also support a patch size (2,2,2), we denote this weak model as ‘temporal’, and (1,4,4), denoted as ‘spatial’. We then use either one of these modes as a weak model during inference. Both modes could also be used iteratively for the generation of a single sample, which we leave for future work.

By adjusting the number of steps performed with our weak model, we can again control the overall compute required per generated sample. We use VBench [45] to evaluate the quality of the generated videos and report results in Fig. 8. In this case, we can save up to 75% of compute with-

out a significant drop in performance. Recent training-free methods [51, 62, 113], while appealing due to their lack of training requirements, achieve significantly lower compute savings before performance begins to degrade noticeably. This demonstrates that allowing the model to learn optimized compute allocation is more effective than relying on predefined rules for inference efficiency. Additional comparisons to previous work are provided in the appendix.

#### 4.4. FLOPs vs Latency

High-resolution image/video generation is predominantly compute-bound. To verify, we propagate sequences of different lengths through a fixed size DiT and measure performance — FLOPs and latency — on a NVIDIA H100 SXM5 with a batch size of 2, simulating inference with CFG5. In Fig. 9 we show that the weak Emu and Video DiT models are also compute-bound, for the setup (generated image/video resolution) that we presented in the paper. Indeed, for T2V, FLOPs utilization is higher for our weak models, due to inefficiencies of the self-attention operation for large sequence lengths when using the powerful model. This effect can be expected to be even more predominant for multi-GPU inference. Consequently, latency benefits are even higher than FLOPs benefits presented so far.

### 5. Conclusion

We have demonstrated how regular DiTs can be converted into flexible ones, that can process samples with different patch sizes after minimal fine-tuning. Adjusting the compute for some denoising steps in the diffusion process readily allows accelerating inference without compromising. Notably, the efficiency benefits of our approach are independent of the chosen solver or the number of denoising steps. We have displayed how our approach is generic and can be straightforwardly applied to class-conditioned image

5In all cases, we compile using torch.compile with fullgraph=True and mode = ’reduce-overhead’.

T2I(1,4,4) (406.1T)

T2V(1,2,2) (345.1T)

T2V(1,4,4) (398.2T)

Performance(FLOPs/second)

1000T

100T

T2I(1,2,2) (417.2T)

T2V(2,2,2) (373.7T)

Memory Bandwidth

Peak FLOPs Limit

102 103 104 105 Operational Intensity (FLOPs / byte)

- Figure 9. GPU utilization for one denoising step, when propagating sequences with different overall number of tokens, corresponding to different patch sizes (pf, ph, pw). Our T2I model has no temporal dimension, but we overload notation and set pf = 1. For simplicity, we use a DiT of similar configuration (width and depth) for both the T2I and T2V reported in this plot numbers, but results generalize across model shapes.

generation, low and high-resolution text-conditioned image generation, and text-conditioned video generation. Looking ahead, we anticipate further applications of our flexible DiT framework across various modalities, such as audio and 3D modeling. As computational resources become increasingly in demand, developing efficient and adaptable models like ours will be crucial for enabling generative capabilities that are of high-quality, but also more scalable.

### References

- [1] Donghoon Ahn, Hyoungwon Cho, Jaewon Min, Wooseok Jang, Jungwoo Kim, SeonHwa Kim, Hyun Hee Park, Kyong Hwan Jin, and Seungryong Kim. Self-rectifying diffusion sampling with perturbed-attention guidance, 2024. 6
- [2] Sotiris Anagnostidis, Dario Pavllo, Luca Biggio, Lorenzo Noci, Aurelien Lucchi, and Thomas Hofmann. Dynamic context pruning for efficient and interpretable autoregressive transformers. Advances in Neural Information Processing Systems, 36:65202–65223, 2023. 14
- [3] Sotiris Anagnostidis, Gregor Bachmann, Imanol Schlag, and Thomas Hofmann. Navigating scaling laws: Compute optimality in adaptive model training. In Forty-first International Conference on Machine Learning, 2024. 3, 26
- [4] Gregor Bachmann, Sotiris Anagnostidis, and Thomas Hofmann. Scaling mlps: A tale of inductive bias. Advances in Neural Information Processing Systems, 36, 2024. 3
- [5] Yogesh Balaji, Seungjun Nah, Xun Huang, Arash Vahdat, Jiaming Song, Qinsheng Zhang, Karsten Kreis, Miika Aittala, Timo Aila, Samuli Laine, et al. ediff-i: Text-to-image diffusion models with an ensemble of expert denoisers. arXiv preprint arXiv:2211.01324, 2022. 2, 14
- [6] Arpit Bansal, Eitan Borgnia, Hong-Min Chu, Jie Li, Hamid Kazemi, Furong Huang, Micah Goldblum, Jonas Geiping,

- and Tom Goldstein. Cold diffusion: Inverting arbitrary image transforms without noise. Advances in Neural Information Processing Systems, 36, 2024. 3
- [7] James Betker, Gabriel Goh, Li Jing, Tim Brooks, Jianfeng Wang, Linjie Li, Long Ouyang, Juntang Zhuang, Joyce Lee, Yufei Guo, et al. Improving image generation with better captions. Computer Science. https://cdn. openai. com/papers/dall-e-3. pdf, 2(3):8, 2023. 7
- [8] Lucas Beyer, Xiaohua Zhai, Am´elie Royer, Larisa Markeeva, Rohan Anil, and Alexander Kolesnikov. Knowledge distillation: A good teacher is patient and consistent. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10925–10934, 2022. 14
- [9] Lucas Beyer, Pavel Izmailov, Alexander Kolesnikov, Mathilde Caron, Simon Kornblith, Xiaohua Zhai, Matthias Minderer, Michael Tschannen, Ibrahim Alabdulmohsin, and Filip Pavetic. Flexivit: One model for all patch sizes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14496–14506, 2023. 3, 4, 14, 26
- [10] BlackForestlabs AI. Flux. https : / / blackforestlabs.ai/#get-flux, 2024. [Online; accessed 17-Oct-2024]. 1
- [11] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023. 15
- [12] Daniel Bolya, Cheng-Yang Fu, Xiaoliang Dai, Peizhao Zhang, Christoph Feichtenhofer, and Judy Hoffman. Token merging: Your vit but faster. arXiv preprint arXiv:2210.09461, 2022. 14
- [13] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, Clarence Ng, Ricky Wang, and Aditya Ramesh. Video generation models as world simulators,

2024. 1

- [14] Angela Castillo, Jonas Kohler, Juan C P´erez, Juan Pablo P´erez, Albert Pumarola, Bernard Ghanem, Pablo Arbel´aez, and Ali Thabet. Adaptive guidance: Training-free acceleration of conditional diffusion models. arXiv preprint arXiv:2312.12487, 2023. 2, 21
- [15] Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Yue Wu, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, et al. Pixart-alpha: Fast training of diffusion transformer for photorealistic text-to-image synthesis. arXiv preprint arXiv:2310.00426, 2023. 4, 7, 22, 26
- [16] Junsong Chen, Chongjian Ge, Enze Xie, Yue Wu, Lewei Yao, Xiaozhe Ren, Zhongdao Wang, Ping Luo, Huchuan Lu, and Zhenguo Li. Pixart-\sigma: Weak-to-strong training of diffusion transformer for 4k text-to-image generation. arXiv preprint arXiv:2403.04692, 2024. 14
- [17] Zigeng Chen, Xinyin Ma, Gongfan Fang, Zhenxiong Tan, and Xinchao Wang. Asyncdiff: Parallelizing diffusion models by asynchronous denoising. arXiv preprint arXiv:2406.06911, 2024. 17
- [18] Jang Hyun Cho and Bharath Hariharan. On the efficacy of knowledge distillation. In Proceedings of the IEEE/CVF

- international conference on computer vision, pages 4794– 4802, 2019. 5
- [19] Xiaoliang Dai, Ji Hou, Chih-Yao Ma, Sam Tsai, Jialiang Wang, Rui Wang, Peizhao Zhang, Simon Vandenhende, Xiaofang Wang, Abhimanyu Dubey, et al. Emu: Enhancing image generation models using photogenic needles in a haystack. arXiv preprint arXiv:2309.15807, 2023. 7
- [20] Giannis Daras, Mauricio Delbracio, Hossein Talebi, Alexandros G Dimakis, and Peyman Milanfar. Soft diffusion: Score matching for general corruptions. arXiv preprint arXiv:2209.05442, 2022. 3
- [21] Giannis Daras, Yuval Dagan, Alex Dimakis, and Constantinos Daskalakis. Consistent diffusion models: Mitigating sampling drift by learning to be consistent. Advances in Neural Information Processing Systems, 36, 2024. 15
- [22] Mostafa Dehghani, Josip Djolonga, Basil Mustafa, Piotr Padlewski, Jonathan Heek, Justin Gilmer, Andreas Peter Steiner, Mathilde Caron, Robert Geirhos, Ibrahim Alabdulmohsin, et al. Scaling vision transformers to 22 billion parameters. In International Conference on Machine Learning, pages 7480–7512. PMLR, 2023. 27
- [23] Mostafa Dehghani, Basil Mustafa, Josip Djolonga, Jonathan Heek, Matthias Minderer, Mathilde Caron, Andreas Steiner, Joan Puigcerver, Robert Geirhos, Ibrahim M Alabdulmohsin, et al. Patch n’pack: Navit, a vision transformer for any aspect ratio and resolution. Advances in Neural Information Processing Systems, 36, 2024. 6
- [24] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in neural information processing systems, 34:8780–8794, 2021. 1, 22
- [25] Sander Dieleman. Diffusion is spectral autoregression,

2024. 2

- [26] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020. 2, 14
- [27] St´ephane d’Ascoli, Hugo Touvron, Matthew L Leavitt, Ari S Morcos, Giulio Biroli, and Levent Sagun. Convit: Improving vision transformers with soft convolutional inductive biases. In International conference on machine learning, pages 2286–2296. PMLR, 2021. 17
- [28] Bradley Efron. Tweedie’s formula and selection bias. Journal of the American Statistical Association, 106(496): 1602–1614, 2011. 3
- [29] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first International Conference on Machine Learning,

2024. 5

- [30] Haoqi Fan, Bo Xiong, Karttikeya Mangalam, Yanghao Li, Zhicheng Yan, Jitendra Malik, and Christoph Feichtenhofer. Multiscale vision transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 6824–6835, 2021. 14

- [31] Gongfan Fang, Xinyin Ma, and Xinchao Wang. Structural pruning for diffusion models, 2023. 14
- [32] Stephanie Fu, Netanel Tamir, Shobhita Sundaram, Lucy Chai, Richard Zhang, Tali Dekel, and Phillip Isola. Dreamsim: Learning new dimensions of human visual similarity using synthetic data. arXiv preprint arXiv:2306.09344,

2023. 2

- [33] Arthur Gretton, Karsten M Borgwardt, Malte J Rasch, Bernhard Sch¨olkopf, and Alexander Smola. A kernel twosample test. The Journal of Machine Learning Research, 13(1):723–773, 2012. 15
- [34] Ali Hatamizadeh, Greg Heinrich, Hongxu Yin, Andrew Tao, Jose M Alvarez, Jan Kautz, and Pavlo Molchanov. Fastervit: Fast vision transformers with hierarchical attention. arXiv preprint arXiv:2306.06189, 2023. 14
- [35] Ali Hatamizadeh, Jiaming Song, Guilin Liu, Jan Kautz, and Arash Vahdat. Diffit: Diffusion vision transformers for image generation. In European Conference on Computer Vision, pages 37–55. Springer, 2025. 15
- [36] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Delving deep into rectifiers: Surpassing human-level performance on imagenet classification. In Proceedings of the IEEE international conference on computer vision, pages 1026–1034, 2015. 4
- [37] Tom Henighan, Jared Kaplan, Mor Katz, Mark Chen, Christopher Hesse, Jacob Jackson, Heewoo Jun, Tom B Brown, Prafulla Dhariwal, Scott Gray, et al. Scaling laws for autoregressive generative modeling. arXiv preprint arXiv:2010.14701, 2020. 3
- [38] Geoffrey Hinton. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531, 2015. 5
- [39] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022. 5
- [40] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 3, 6, 15
- [41] Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. Training compute-optimal large language models. arXiv preprint arXiv:2203.15556, 2022. 3, 25
- [42] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models, 2021. 4
- [43] Yushi Hu, Benlin Liu, Jungo Kasai, Yizhong Wang, Mari Ostendorf, Ranjay Krishna, and Noah A Smith. Tifa: Accurate and interpretable text-to-image faithfulness evaluation with question answering. arXiv preprint arXiv:2303.11897,

2023. 24

- [44] Huaibo Huang, Xiaoqiang Zhou, Jie Cao, Ran He, and Tieniu Tan. Vision transformer with super token sampling. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 22690–22699, 2023. 14
- [45] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive

- benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21807–21818, 2024. 8, 27
- [46] Adam Ibrahim, Benjamin Th´erien, Kshitij Gupta, Mats L. Richter, Quentin Anthony, Timoth´ee Lesort, Eugene Belilovsky, and Irina Rish. Simple and scalable strategies to continually pre-train large language models, 2024. 4
- [47] Moritz Imfeld, Jacopo Graldi, Marco Giordano, Thomas Hofmann, Sotiris Anagnostidis, and Sidak Pal Singh. Transformer fusion with optimal transport. arXiv preprint arXiv:2310.05719, 2023. 14
- [48] Moritz Imfeld, Jacopo Graldi, Marco Giordano, Thomas Hofmann, Sotiris Anagnostidis, and Sidak Pal Singh. Transformer fusion with optimal transport. In The Twelfth International Conference on Learning Representations,

2024. 14

- [49] Sergey Ioffe. Batch normalization: Accelerating deep network training by reducing internal covariate shift. arXiv preprint arXiv:1502.03167, 2015. 4
- [50] Bowen Jing, Gabriele Corso, Renato Berlinghieri, and Tommi Jaakkola. Subspace diffusion generative models. In European Conference on Computer Vision, pages 274–289. Springer, 2022. 14
- [51] Kumara Kahatapitiya, Haozhe Liu, Sen He, Ding Liu, Menglin Jia, Michael S Ryoo, and Tian Xie. Adaptive caching for faster video generation with diffusion transformers. arXiv preprint arXiv:2411.02397, 2024. 8, 15
- [52] Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361,

2020. 1, 3

- [53] Tero Karras, Miika Aittala, Tuomas Kynk¨a¨anniemi, Jaakko Lehtinen, Timo Aila, and Samuli Laine. Guiding a diffusion model with a bad version of itself. arXiv preprint arXiv:2406.02507, 2024. 6
- [54] Dongjun Kim, Chieh-Hsin Lai, Wei-Hsiang Liao, Yuhta Takida, Naoki Murata, Toshimitsu Uesaka, Yuki Mitsufuji, and Stefano Ermon. Pagoda: Progressive growing of a one-step generator from a low-resolution diffusion teacher. arXiv preprint arXiv:2405.14822, 2024. 14
- [55] Yuval Kirstain, Adam Polyak, Uriel Singer, Shahbuland Matiana, Joe Penna, and Omer Levy. Pick-a-pic: An open dataset of user preferences for text-to-image generation. Advances in Neural Information Processing Systems, 36: 36652–36663, 2023. 24
- [56] Jonas Kohler, Albert Pumarola, Edgar Sch¨onfeld, Artsiom Sanakoyeu, Roshan Sumbaly, Peter Vajda, and Ali Thabet. Imagine flash: Accelerating emu diffusion models with backward distillation. arXiv preprint arXiv:2405.05224,

2024. 15

- [57] Mingxiao Li, Tingyu Qu, Ruicong Yao, Wei Sun, and Marie-Francine Moens. Alleviating exposure bias in diffusion models through sampling with shifted time steps. arXiv preprint arXiv:2305.15583, 2023. 6, 15
- [58] Yangming Li and Mihaela van der Schaar. On error propagation of diffusion models. In The Twelfth International Conference on Learning Representations, 2023. 15

- [59] Zhiqiu Lin, Deepak Pathak, Baiqi Li, Jiayao Li, Xide Xia, Graham Neubig, Pengchuan Zhang, and Deva Ramanan. Evaluating text-to-visual generation with imageto-text generation. arXiv preprint arXiv:2404.01291, 2024. 7
- [60] Zhiqiu Lin, Deepak Pathak, Baiqi Li, Jiayao Li, Xide Xia, Graham Neubig, Pengchuan Zhang, and Deva Ramanan. Evaluating text-to-visual generation with imageto-text generation. In European Conference on Computer Vision, pages 366–384. Springer, 2025. 23
- [61] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022. 15
- [62] Haozhe Liu, Wentian Zhang, Jinheng Xie, Francesco Faccio, Mengmeng Xu, Tao Xiang, Mike Zheng Shou, JuanManuel Perez-Rua, and J¨urgen Schmidhuber. Faster diffusion via temporal attention decomposition. arXiv e-prints, pages arXiv–2404, 2024. 8, 15
- [63] Qihao Liu, Zhanpeng Zeng, Ju He, Qihang Yu, Xiaohui Shen, and Liang-Chieh Chen. Alleviating distortion in image generation via multi-resolution diffusion models. arXiv preprint arXiv:2406.09416, 2024. 15
- [64] Xiangcheng Liu, Tianyi Wu, and Guodong Guo. Adaptive sparse vit: Towards learnable adaptive token pruning by fully exploiting self-attention. arXiv preprint arXiv:2209.13802, 2022. 14
- [65] Yixin Liu, Kai Zhang, Yuan Li, Zhiling Yan, Chujie Gao, Ruoxi Chen, Zhengqing Yuan, Yue Huang, Hanchi Sun, Jianfeng Gao, et al. Sora: A review on background, technology, limitations, and opportunities of large vision models. arXiv preprint arXiv:2402.17177, 2024. 15
- [66] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In Proceedings of the IEEE/CVF international conference on computer vision, pages 10012–10022, 2021. 14
- [67] Zhijun Liu, Shuai Wang, Sho Inoue, Qibing Bai, and Haizhou Li. Autoregressive diffusion transformer for textto-speech synthesis. arXiv preprint arXiv:2406.05551,

2024. 7

- [68] Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps. Advances in Neural Information Processing Systems, 35: 5775–5787, 2022. 22
- [69] Chenyang Lu, Daan de Geus, and Gijs Dubbelman. Content-aware token sharing for efficient semantic segmentation with vision transformers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 23631–23640, 2023. 14
- [70] Xinyin Ma, Gongfan Fang, Michael Bi Mi, and Xinchao Wang. Learning-to-cache: Accelerating diffusion transformer via layer caching. arXiv preprint arXiv:2406.01733,

2024. 20

- [71] Xinyin Ma, Gongfan Fang, and Xinchao Wang. Deepcache: Accelerating diffusion models for free. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15762–15772, 2024. 14, 17

- [72] Xin Ma, Yaohui Wang, Gengyun Jia, Xinyuan Chen, Ziwei Liu, Yuan-Fang Li, Cunjian Chen, and Yu Qiao. Latte: Latent diffusion transformer for video generation. arXiv preprint arXiv:2401.03048, 2024. 7, 8
- [73] Zhiyuan Ma, Yuzhu Zhang, Guoli Jia, Liangliang Zhao, Yichao Ma, Mingjie Ma, Gaofeng Liu, Kaiyan Zhang, Jianjun Li, and Bowen Zhou. Efficient diffusion models: A comprehensive survey from principles to practices. arXiv preprint arXiv:2410.11795, 2024. 15
- [74] Chenlin Meng, Robin Rombach, Ruiqi Gao, Diederik Kingma, Stefano Ermon, Jonathan Ho, and Tim Salimans. On distillation of guided diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14297–14306, 2023. 15
- [75] Eliya Nachmani, Robin San Roman, and Lior Wolf. Denoising diffusion gamma models. arXiv preprint arXiv:2110.05948, 2021. 3
- [76] Mang Ning, Mingxiao Li, Jianlin Su, Albert Ali Salah, and Itir Onal Ertugrul. Elucidating the exposure bias in diffusion models. arXiv preprint arXiv:2308.15321, 2023. 6, 15
- [77] Lorenzo Noci, Sotiris Anagnostidis, Luca Biggio, Antonio Orvieto, Sidak Pal Singh, and Aurelien Lucchi. Signal propagation in transformers: Theoretical perspectives and the role of rank collapse. Advances in Neural Information Processing Systems, 35:27198–27211, 2022. 4
- [78] William Peebles and Saining Xie. Scalable diffusion models with transformers, 2023. 1, 6, 17, 26
- [79] Adam Polyak, Amit Zohar, Andrew Brown, Andros Tjandra, Animesh Sinha, Ann Lee, Apoorv Vyas, Bowen Shi, Chih-Yao Ma, Ching-Yao Chuang, et al. Movie gen: A cast of media foundation models. arXiv preprint arXiv:2410.13720, 2024. 8, 15
- [80] Xiangyu Qi, Yi Zeng, Tinghao Xie, Pin-Yu Chen, Ruoxi Jia, Prateek Mittal, and Peter Henderson. Fine-tuning aligned language models compromises safety, even when users do not intend to!, 2023. 4
- [81] Maithra Raghu, Thomas Unterthiner, Simon Kornblith, Chiyuan Zhang, and Alexey Dosovitskiy. Do vision transformers see like convolutional neural networks? Advances in neural information processing systems, 34: 12116–12128, 2021. 17
- [82] Yongming Rao, Wenliang Zhao, Benlin Liu, Jiwen Lu, Jie Zhou, and Cho-Jui Hsieh. Dynamicvit: Efficient vision transformers with dynamic token sparsification. Advances in neural information processing systems, 34: 13937–13949, 2021. 14
- [83] Seyedmorteza Sadat, Manuel Kansy, Otmar Hilliges, and Romann M. Weber. No training, no problem: Rethinking classifier-free guidance for diffusion models, 2024. 6
- [84] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in neural information processing systems, 35:36479–36494, 2022. 24

- [85] Axel Sauer, Frederic Boesel, Tim Dockhorn, Andreas Blattmann, Patrick Esser, and Robin Rombach. Fast highresolution image synthesis with latent adversarial diffusion distillation. arXiv preprint arXiv:2403.12015, 2024. 16
- [86] Shuwei Shi, Wenbo Li, Yuechen Zhang, Jingwen He, Biao Gong, and Yinqiang Zheng. Resmaster: Mastering highresolution image generation via structural and fine-grained guidance. arXiv preprint arXiv:2406.16476, 2024. 14
- [87] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International conference on machine learning, pages 2256–2265. PMLR, 2015. 1
- [88] Yang Song and Prafulla Dhariwal. Improved techniques for training consistency models. arXiv preprint arXiv:2310.14189, 2023. 16
- [89] Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models. arXiv preprint arXiv:2303.01469, 2023. 15, 16, 18
- [90] Stability AI. Stablediffusion3. https://stability. ai/news/stable-diffusion-3, 2024. [Online; accessed 17-Oct-2024]. 1
- [91] Stability AI. Introducing stable cascade, 2024. 2
- [92] Vadim Sushko, Edgar Sch¨onfeld, Dan Zhang, Juergen Gall, Bernt Schiele, and Anna Khoreva. You only need adversarial supervision for semantic image synthesis. arXiv preprint arXiv:2012.04781, 2020. 17
- [93] Gerald Tesauro et al. Temporal difference learning and tdgammon. Communications of the ACM, 38(3):58–68, 1995. 15
- [94] Tristan Thrush, Ryan Jiang, Max Bartolo, Amanpreet Singh, Adina Williams, Douwe Kiela, and Candace Ross. Winoground: Probing vision and language models for visio-linguistic compositionality. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5238–5248, 2022. 24
- [95] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017. 1
- [96] Dimitri Von R¨utte, Sotiris Anagnostidis, Gregor Bachmann, and Thomas Hofmann. A language model’s guide through latent space. In International Conference on Machine Learning, pages 49655–49687. PMLR, 2024. 17
- [97] Jing Wang, Ao Ma, Jiasong Feng, Dawei Leng, Yuhui Yin, and Xiaodan Liang. Qihoo-t2x: An efficiency-focused diffusion transformer via proxy tokens for text-to-any-task. arXiv preprint arXiv:2409.04005, 2024. 14
- [98] Zhou Wang, Eero P Simoncelli, and Alan C Bovik. Multiscale structural similarity for image quality assessment. In The Thrity-Seventh Asilomar Conference on Signals, Systems & Computers, 2003, pages 1398–1402. Ieee, 2003. 17
- [99] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4):600–612, 2004. 2

- [100] Felix Wimbauer, Bichen Wu, Edgar Schoenfeld, Xiaoliang Dai, Ji Hou, Zijian He, Artsiom Sanakoyeu, Peizhao Zhang, Sam Tsai, Jonas Kohler, et al. Cache me if you can: Accelerating diffusion models through block caching. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6211–6220, 2024. 4, 14, 17
- [101] Xinjian Wu, Fanhu Zeng, Xiudong Wang, Yunhe Wang, and Xinghao Chen. Ppt: Token pruning and pooling for efficient vision transformers. arXiv preprint arXiv:2310.01812,

2023. 14

- [102] Tete Xiao, Mannat Singh, Eric Mintun, Trevor Darrell, Piotr Doll´ar, and Ross Girshick. Early convolutions help transformers see better. Advances in neural information processing systems, 34:30392–30400, 2021. 14, 17
- [103] Enze Xie, Lewei Yao, Han Shi, Zhili Liu, Daquan Zhou, Zhaoqiang Liu, Jiawei Li, and Zhenguo Li. Difffit: Unlocking transferability of large diffusion models via simple parameter-efficient fine-tuning. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4230–4239, 2023. 4
- [104] Ruibin Xiong, Yunchang Yang, Di He, Kai Zheng, Shuxin Zheng, Chen Xing, Huishuai Zhang, Yanyan Lan, Liwei Wang, and Tieyan Liu. On layer normalization in the transformer architecture. In International Conference on Machine Learning, pages 10524–10533. PMLR, 2020. 4
- [105] Yilun Xu, Mingyang Deng, Xiang Cheng, Yonglong Tian, Ziming Liu, and Tommi Jaakkola. Restart sampling for improving generative processes. Advances in Neural Information Processing Systems, 36:76806–76838, 2023. 27
- [106] Shuchen Xue, Mingyang Yi, Weijian Luo, Shifeng Zhang, Jiacheng Sun, Zhenguo Li, and Zhi-Ming Ma. Sa-solver: Stochastic adams solver for fast sampling of diffusion models. Advances in Neural Information Processing Systems, 36, 2024. 22
- [107] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-tovideo diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024. 1
- [108] Zhihang Yuan, Pu Lu, Hanling Zhang, Xuefei Ning, Linfeng Zhang, Tianchen Zhao, Shengen Yan, Guohao Dai, and Yu Wang. Ditfastattn: Attention compression for diffusion transformer models. arXiv preprint arXiv:2406.08552,

2024. 14, 17

- [109] Xiaohua Zhai, Alexander Kolesnikov, Neil Houlsby, and Lucas Beyer. Scaling vision transformers. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12104–12113, 2022. 3
- [110] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 586–595, 2018. 2, 17
- [111] Shiwei Zhang, Jiayu Wang, Yingya Zhang, Kang Zhao, Hangjie Yuan, Zhiwu Qin, Xiang Wang, Deli Zhao, and Jingren Zhou. I2vgen-xl: High-quality image-to-video synthesis via cascaded diffusion models. arXiv preprint arXiv:2311.04145, 2023. 14

- [112] Wangbo Zhao, Yizeng Han, Jiasheng Tang, Kai Wang, Yibing Song, Gao Huang, Fan Wang, and Yang You. Dynamic diffusion transformer. arXiv preprint arXiv:2410.03456,

2024. 14

- [113] Xuanlei Zhao, Xiaolong Jin, Kai Wang, and Yang You. Real-time video generation with pyramid attention broadcast. arXiv preprint arXiv:2408.12588, 2024. 8, 14, 15, 17
- [114] Mingjian Zhu, Yehui Tang, and Kai Han. Vision transformer pruning. arXiv preprint arXiv:2104.08500, 2021. 14

### A. Detailed Related Work

Efficiency in Vision Transformers [26] falls primarily into two broad categories: reducing the computation per token or the number of tokens overall.

Reducing the amount of computation per token. Reducing the amount of computation per token can be achieved by reducing the model size when training via distillation [8] or pruning the network after training [47, 114]. These methods again though typically work with a static inference protocol. As in [112], we compare in Fig. 10 our class-conditioned ImageNet FlexiDiT model, against popular based pruning techniques, based on Diff pruning [31], Taylor, magnitude or random pruning [48]. As one can see, our dynamic scheduler outperforms these baselines. We note that our method can also be

5.5

5.0

4.5

- 3.8: pruned w/ Magnitude
- 4.5: pruned w/ Random

4.0

3.5: pruned w/ Taylor

FID-50k

3.5

3.5: pruned w/ Diff

3.0

3.24

2.5

2.64

2.31 2.23 2.25 2.25

2.0

25 % 50 % 75 % 100 % FLOPs % compared to baseline

Figure 10. We compare our dynamic scheduler with more baselines.

applied in conjunction with pruning techniques to achieve even higher efficiency gains, offering potential orthogonal benefits. In concurrent work, [112] propose to adjust, in a dynamic way, the model during inference in different steps. In our work, we do not train separate models for each target FLOPs ratio like they do, meaning that we can train a single model and decide how many FLOPs we want to invest during inference. This makes our approach more versatile. Additionally, our training is very stable, and no specific training tricks are required to converge successfully. That is how we were able to extend experiments to high-resolution image and video generation, achieving significantly better speed-ups than the ones they reported. We also outperform their results when the compute budget is very small. Nonetheless, this work could inspire adaptive per-sample schedulers, that could open new future directions.

Given the large computational requirements of the attention operation, many methods nowadays focus on that to reduce the overhead imposed. These methods commonly use some form of hierarchical attention [34, 66], skip (usually the first) attention layers altogether [102], or reduce the number of the attended keys [16, 30, 108], by commonly aggregating keys in a spatial neighborhood or applying some form of windowed attention.

Reducing number of tokens. Our method primarily falls within the second category of reducing the overall number of tokens. Previous work here, typically relied on filtering [64, 82, 101], merging [12, 44, 69] or dropping [2]. Although merging works well for applications that eventually lead to some pooling operations (like classification tasks or for the task of creating an image-specific embedding), it works significantly less well for applications that require dense (token-level) predictions, where some un-merging operation has to be defined. In other concurrent work, [97] reduces the number of representative tokens to calculate the attention over. Our approach resembles most [9], where vision Transformers are trained to handle inputs with varying patch resolution. By applying less compute for some steps, we can reduce computational complexity significantly, without a drop in performance.

Image generation. In the context of image generation, diffusion has been largely established as the method for attaining state-of-the-art results. There have been previous works that try to take advantage of potential correlations between successive denoising step predictions [86], by either caching intermediate results in the activations [71, 100], or in the attention [113]. Caching has the advantage of a training-free method. Nonetheless, potential benefits are lower. Similar to our work, [5] use different experts for different denoising steps. Instead of using different experts that require separate training and separate deployment, we show how a single model can be easily formed into a flexible one that can be instantiated in different modes, with each of its modes corresponding essentially to a different expert. Similar in-spirit approaches have been proposed that rely on the smaller compute requirements for lower resolution image generation [54, 111]. [50] also adapt the computation per step, by projecting into smaller subspaces. We instead, keep the dimension of the latent space and the characteristics

of it the same across diffusion steps. Orthogonal gains to our approach are also possible through methods such as guidance distillation [56, 74] and consistency models [89]. Our approach is also largely agnostic to the diffusion process and can be applied out of the box for flow matching methods [61]. We point the interested reader to [73] for a survey for further efficiency in diffusion models. Finally, compared to other established techniques [35, 63] we do not fundamentally change the architecture, which allows us to apply our framework effortlessly for numerous pre-trained models across different modalities.

Video generation. Our approach can be easily extended for video generation, and in principle for the generation of any modality where some inductive bias (spatial, temporal, etc) is employed in the diffusion (latent) space. In video generation, typically, latent video tokens are processed in parallel [11, 65, 79]. Training-free methods [51, 62, 113] have been proposed in this case to accelerate video generation. Benefits with training-free methods are nonetheless minimal before performance degradation kicks in (see Table 1 in [51], where one can typically save less than 30%).

An interesting direction for future work involves adapting the inference scheduler, i.e. what patch size we are using for each denoising step, based on the requirements of each sample. It is natural to assume that when generating more static videos, increasing the temporal patch size, and thus decreasing the amount of compute along the temporal dimension, will result in smaller drops in performance. The same holds for the spatial patch sizes when generating images or videos that require less high-frequency details.

- B. Additional Experiments and Details We provide additional experiments, complementary to the main text.

#### B.1. Exposure Bias and Accumulation of Error

0

Without regularizer

[Figure 35]

With regularizer

Stepswithpowerfulmodel

DistributionDistance

T

pnoise pdata

[Figure 36]

- Figure 11. Left: We use maximum mean discrepancy to estimate the distribution mismatch between pθ(xt|xT:t+1) and q(xt|x0). Right: The proposed bootstrapped loss. During training, we perform a few denoising steps with a weak model followed by a few denoising steps with a powerful model (reminiscent of the scheduler during inference) and apply a distribution matching loss on the resulting samples.

Inference with diffusion suffers from exposure bias, due to the discrepancy of the input distribution during training and inference [21, 57, 58, 76]. Briefly, models are trained to denoise images sampled from the q(xt|x0) distribution [40]. Inference on the other hand is characterized by repeated model evaluations pθ(xt|xt+1) and any distribution mismatch between

- pθ(xt|xT:t+1) and q(xt|x0) accumulates, as also shown in Fig. 11 (left). The error at each iteration depends on the model, with a perfect model resulting in 0 error and thus no error accumulated. In our case, the accumulation of error is exacerbated by the characteristics of our model, where weak models could lead to higher, but also specific in nature, kinds of errors. Training with the standard denoising objective, where real samples are randomly noised for some t, does not make the more powerful model aware of the nature of the mistakes made by the weak model, rendering it unable to potentially correct them. We propose to mitigate this issue by introducing a bootstrapped distribution matching loss [93], as illustrated in Fig. 11 (right). The loss is applied in a patch size-dependent manner, according to the desired inference protocol (from weak to powerful model calls during inference).

Given natural images x0,x˜0 ∼ q(x0), we sample two time points t1 > t2 and corrupt the images with noise xtargett

1

∼

- q(xt

2|x˜0). We then apply a chain of denoising steps ϵθ(x˜predt−1|x˜predt ;p) for t ∈ (t1,t2] and a patch size p. Ultimately, we wish for the distributions of xtargett

1|x0),x˜predt

∼ q(x˜t

2

and x˜predt

to match, for which we employ the maximum mean discrepancy (MMD) [33]. To let the powerful model learn and potentially correct mistakes of the weak model and to simulate how our inference patch size scheduler works, we perform the first of these denoising steps with the weak model, followed by denoising steps with the powerful model. Given a set of patch sizes {pi}ki=1 where p1 < p2 < ··· < pk and a number of

1

1

denoising steps to perform with each s1,s2,...,sk, where kj=1 sj = t2 − t1, we denoise with a given patch size pi for all t’s in (t1 + kj=i+1 sj,t1 + kj=i sj]. An illustration of this process can also be seen in Fig. 11 (right). When sampling a time step t1, we bias our sampling similar to [85]. The proposed distribution matching loss, inspired by the notion of consistency [88, 89], provides a principled way to correct the errors accumulated during inference. We note that we are optimizing a simple distribution matching loss, instead of over-optimizing according to desired downstream metrics (namely FID), thus not violating Goodhart’s law6. Different distribution matching losses (including discriminator-based losses) can also be used. Note that we only correct this exposure bias for the class-conditioned image generation experiments, as this is the only case where we fine-tune the powerful pre-trained model.

#### B.2. Inference with Packing

Generate images: 1

200

| | | |Image|1 C| | | |
|---|---|---|---|---|---|---|---|

| | | |Image|1 C| | | |
|---|---|---|---|---|---|---|---|

180

Latency(relative)

160

140

|Imag UC|e 1|
|---|---|

| | |Padd|ing| | |
|---|---|---|---|---|---|

| | | |Image|2 C| | | |
|---|---|---|---|---|---|---|---|

120

100

| | | |Image|2 C| | | |
|---|---|---|---|---|---|---|---|

| | | |Image|3 C| | | |
|---|---|---|---|---|---|---|---|

80

80 100 120 140 160

Generate images: 4

200

|Imag UC|e 2|
|---|---|

| | |Padd|ing| | |
|---|---|---|---|---|---|

| | | |Image|4 C| | | |
|---|---|---|---|---|---|---|---|

Approach1

Approach2

180

Latency(relative)

160

| | | |Image|3 C| | | |
|---|---|---|---|---|---|---|---|

|Imag UC|e 1|
|---|---|

140

120

100

|Imag UC|e 3|
|---|---|

| | |Padd|ing| | |
|---|---|---|---|---|---|

|Imag UC|e 2|
|---|---|

80

80 100 120 140 160

| | | |Image|4 C| | | |
|---|---|---|---|---|---|---|---|

|Imag UC|e 3|
|---|---|

Generate images: 16

200

180

Latency(relative)

|Imag UC|e 4|
|---|---|

| | |Padd|ing| | |
|---|---|---|---|---|---|

|Imag UC|e 4|
|---|---|

160

140

120

100

80

| | | |Image|1 C| | | |
|---|---|---|---|---|---|---|---|

| | | |Image|1 C| | | |
|---|---|---|---|---|---|---|---|

|Imag UC|e 1 )|
|---|---|

80 100 120 140 160

Generate images: 64

| | | |Image|2 C| | | |
|---|---|---|---|---|---|---|---|

200

| | | |Image|2 C| | | |
|---|---|---|---|---|---|---|---|

|Imag UC|e 2 )|
|---|---|

Approach4

Approach3

- Approach 1

- Approach 2

- Approach 3

- Approach 4

180

Latency(relative)

| | | |Image|3 C| | | |
|---|---|---|---|---|---|---|---|

160

| | | |Image|3 C| | | |
|---|---|---|---|---|---|---|---|

|Imag UC|e 3 )|
|---|---|

140

| | | |Image|4 C| | | |
|---|---|---|---|---|---|---|---|

120

100

| | | |Image|4 C| | | |
|---|---|---|---|---|---|---|---|

|Imag UC|e 4 )|
|---|---|

80

|Imag UC|e 1|
|---|---|

|Imag UC|e 2|
|---|---|

|Imag UC|e 3|
|---|---|

|Imag UC|e 4|
|---|---|

80 100 120 140 160 FLOPs (relative)

- Figure 12. Different approaches can be employed to perform forward passes with CFG when the conditional (C) and unconditional (UC) predictions use different patch sizes. Here, each row corresponds to a sequence of tokens propagated through the DiT, and each bracket corresponds to a batch of sequences for a single NFE. Generally ‘Approach 2’ leads to the smallest amount of FLOPs, but for batch size 1, inference can be memory bound for low-resolution image generation. ‘Approach 4’ mostly leads to the smallest latency, as long as the number of generated images is larger than 4, i.e. the ratio of the sequence lengths between the powerful and the weak model. On the right, we plot FLOPs and Latency from the four different approaches of performing inference, for a different number of generated images. Batch size plays a role here (class-conditioned image generation experiments) as generated images are of lower resolution, namely 256 × 256, and thus sequence lengths through the Transformer are smaller. Normalized FLOPs are determined based on ‘Approach 2’ and normalized latency based on ‘Approach 3’. We use torch.compile with fullgraph=True and mode = ’reduce-overhead’.

We provide more details in Fig. 12, on how to perform inference with CFG when the conditional and unconditional predictions employ a different patch size. We show this for our class-conditioned model, but results easily generalize for all our FlexiDiT models. Performing CFG entails NFEs with double the batch size (or 2 distinct NFEs), for the conditional and unconditional input, respectively. Performing the conditional and unconditional calls with different patch sizes leads to propagating sequences of different lengths through the DiT. Depending on how these sequences are ‘packed’ together, and for lack of a hardware-specific implementation of masked attention, more FLOPs can be traded for better latency. Our weak model additionally leads to memory benefits, which can be traded for a bigger batch size when serving the model. Notice that current state-of-the-art image generation models in practice require much longer sequences compared to the 256 × 256 images generated here (see also Section 4.4) and so generation is compute-bound even when generating with batch size equal to 1.

6Goodhart’s law states that: ‘When a measure becomes a target, it ceases to be a good measure’.

#### B.3. What does the Model Learn?

Transformers are composed of a series of channel mixing components — feed-forward layers with shared weight applied to all tokens in the sequence — and token mixing components — attention applied to tokens in the sequence. By coercing the model to learn the denoising objective when applied to images processed with different patch sizes, we are enforcing inductive bias in its weights and helping it better understand global structures in the image [27, 81, 102]. We test this hypothesis and evaluate what the model is learning in the following ways in Fig. 13. (left) We visualize using t-SNE, centered kernel alignment (CKA) between feature maps across layers when performing NFEs with different patch sizes. Activations across layers exhibit similar transformations [96], except the early layers, where features are lower level, i.e. more patch specific. (right) We visualize the Jensen–Shannon divergence (JSD) between attention maps (interpolated to the same image space) when performing NFEs with different patch sizes. We compare using our FlexiDiT model with different patch sizes (Flexible) versus using two static models trained with different patch sizes (namely DiT-XL/2 and a trained from scratch DiT-XL/4). Our flexible model showcases lower JSD, demonstrating better knowledge transfer between the different patch sizes. We believe that this “transfer” of knowledge is crucial to (1) confirm that parameter sharing across patch sizes is valid and (2) ensure that fine-tuning can be fast and sample efficient.

[Figure 37]

[Figure 38]

- Figure 13. Interpretability of the model activations and attention scores, when propagating samples tokenized with different patch sizes.

#### B.4. Class-Conditioned Image Generation

Additional metrics. For class-conditioned experiments on the main text we focused on the DiT-XL/2 [78] and FID as a metric. Here, we report more metrics apart from FID, namely Inception Score (IS), sFID, and Precision/Recall. Results are presented in Fig. 14 for our flexible DiT-XL/2 model. We remind that for class-conditioned models, we fine-tune models using our distribution matching loss. As a result, the powerful model that we get after fine-tuning is different to the pretrained checkpoints we start from. To verify that our weak model does not lead to less diverse samples, we embark on a small experimental study to guarantee the diversity of generated images. We follow [92] and generate images from the same label map. We then calculate pairwise similarity/distance between these images and average across all similarities/distances and all label maps. We use MS-SSIM [98], LPIPS [110] and plot results in Fig. 15. Results indicate very similar values in terms of the diversity of the generated images. We also provide some sample images demonstrating diversity from the baseline model in Fig. 16 and our tuned model in Fig. 17. Note that the diversity of the generated images is in general high and not affected much by using the weak model for more of the initial denoising steps.

Caching distance. In the main text, we have shown how weak and powerful models generate more similar predictions during the early steps of the denoising process. Previous papers to accelerate diffusion have largely relied on caching [17, 71, 100, 108, 113] previous activations, by taking advantage of the similarity in activations between successive steps. For completeness, we also plot the caching distance between activations of the same layer between successive generation steps in Fig. 18. In this paper, we do not employ caching but focus on an orthogonal approach. We advocate that all steps are important for high-quality image generation, as demonstrated by our experiments on reducing the overall number of generation steps.

2.0

2.0

[Figure 39]

2.0

350

[Figure 40]

[Figure 41]

- 2

- 3

- 4

- 5

- 6

6.25

6.00

300

5.75

CFGscale

CFGscale

CFGscale

FID-50k

250

5.50

sFID-50k

IS-50k

5.25

200

3.24

5.00

2.64

4.75

150

2.31 2.23 2.25 2.25

4.50

1.25

1.25

1.25

50 % 75 % 100 % FLOPs % compared to baseline

25 % 50 % 75 % 100 % FLOPs % compared to baseline

25 % 50 % 75 % 100 % FLOPs % compared to baseline

2.0

2.0

0.625

[Figure 42]

[Figure 43]

0.900

0.600

0.875

0.575

0.850

Precision-50k

CFGscale

CFGscale

0.550

0.825

Recall-50k

0.525

0.800

0.500

0.775

0.475

0.750

0.450

0.725

1.25

1.25

25 % 50 % 75 % 100 % FLOPs % compared to baseline

25 % 50 % 75 % 100 % FLOPs % compared to baseline

- Figure 14. More metrics for our FlexiDiT based on the DiT-XL/2 model for class-conditioned generation on ImageNet. We plot (a) FID (b) sFID, (c) inception score, (d) precision, and (e) recall when generating 50, 000 samples with 250 steps of the DDPM schedule for various values of the CFG scales. Red lines correspond to the values that lead to the optimum FID scores for each compute level.

0.800

0.200

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

0.775

0.175

0.750

0.150

0.725

0.125

MS-SSIM

0.700

0.100

LPIPS

0.675

0.075

0.650

0.050

0.625

0.025

0.600

0.000

25 % 50 % 75 % 100 % FLOPs % compared to baseline

25 % 50 % 75 % 100 % FLOPs % compared to baseline

Figure 15. Average distance/similarity of images generated from the same label map. Both metrics take values between 0 and 1.

Instead of completely skipping steps, we simply invest less compute for them, and let the model decide how to allocate this compute.

Additional schedulers. Based on the results on activation distance between successive denoising steps (Fig. 18), one could argue that first denoising steps are also important and thus a better inference scheduler would deploy the powerful model for these as well. In practice, we found no benefit from deploying a scheduler that works like that. Notice though how activation distance is high for the first denoising steps only for some of the layers. We additionally experimented with dynamic schedulers that choose the patch size of each denoising step based on the activation distance of different layers between successive denoising steps. We did not find additional potential benefits.

In this paper, we are training a single model that can denoise images with any patch size for any denoising step. Given a fixed desired inference scheduler — i.e. if we know exactly which t’s to run with the powerful and the weak model —, one can train a model specifically based on that, leading to undoubtedly better quality images for the same compute. Similar techniques are regularly applied in consistency models [89]. Finally, we compare our scheduler — performing the first Tweak denoising steps with a weak model — versus the opposite scheduler, i.e. performing the last Tweak denoising steps with a weak model. Results in Fig. 19 indicate that, as expected, using the weak model in the last diffusion steps is suboptimal, leading to a loss in fine-grained details. We also provide qualitative examples of how these different schedulers affect image quality in Fig. 20.

[Figure 44]

Figure 16. Sample images generated with the baseline DiT-XL/2 for the ImageNet category ‘Brambling’.

[Figure 45]

- Figure 17. Sample images generated with our flexible DiT-XL model when performing inference using only the powerful model, for the ImageNet category ‘Brambling’.

More results on CFG. In the main text, we presented results on performing inference with different CFG scales and different invocations to our weak model for the unconditional and conditional part. The 4 generated curves in Fig. 6 (middle) correspond to performing our scheduler as 250/250, 130/130, 70/70, and 30/0 where x/y means using the powerful model for the last x denoising steps for the conditional and y denoising steps for the unconditional part. When performing CFG, we

10

60

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |
| | | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |

30

40

20

25

20

50

25

8

20

30

40

15

20

15

6

15

30

15

20

10

10

4

20

10

10

10

10

5

5

2

5

5

0

0

Block 0

Block 1

Block 2

Block 3

Block 4

Block 5

Block 6

35

40

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |
| | | |

35

30

50

40

30

40

30

25

40

30

25

30

25

30

20

30

20

20

20

20

15

20

15

15

20

10

10

10

10

10

10

10

5

5

5

Block 7

Block 8

Block 9

Block 10

Block 11

Block 12

Block 13

80

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |
| | | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |
| | | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |

60

50

60

50

80

60

50

60

40

40

60

40

40

40

30

30

40

30

40

20

20

20

20

20

20

20

10

10

10

0

0

Block 14

Block 15

Block 16

Block 17

Block 18

Block 19

Block 20

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |
| | | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |

1250

5000 Attetion layer MLP Layer

80

200

120

1250

100

| |
|---|

1000

4000

100

1000

60

80

150

80

3000

750

750

60

100

40

60

2000

500

500

40

40

50

1000

250

20

250

20

20

0

0

0

0

pdata pnoise Block 21

pdata pnoise Block 22

pdata pnoise Block 23

pdata pnoise Block 24

pdata pnoise Block 25

pdata pnoise Block 26

pdata pnoise Block 27

- Figure 18. We plot average distance (L2-norm) between activation of different layers during successive steps of the denoising process of the DiT-XL/2 model. Different layers exhibit different characteristics. Similar observations have been made in [70].

50 % 75 % 100 % FLOPs % compared to baseline

1.5

2.0

2.5

3.0

3.5

4.0

4.5

FID-50k

3.49

3.99

4.32 4.32 4.36 4.29

Weak model for the first denoising steps Weak model for the last denoising steps

2.25 2.24 2.25 2.25

2.45

2.64

3.24

- Figure 19. We compare our scheduler versus a different scheduler that uses the weak model for the last denoising steps when generating class-conditioned images. Points correspond to the minimum — concerning CFG scale — FID values.

use the update rule as presented in the main text

ϵθ(xt−1|xt,∅;puncond) + scfg1(ϵθ(xt−1|xt,c;pcond) − ϵθ(xt−1|xt,∅;puncond)), if pcond = puncond ϵθ(xt−1|xt,c;puncond) + scfg2(ϵθ(xt−1|xt,c;pcond) − ϵθ(xt−1|xt,c;puncond)), if pcond < puncond

.

This guidance scheme seeks to reduce errors made by the powerful model, enhancing potential differences in predictions of the corresponding weak model, when the two models disagree, indicating the general direction towards higher-quality samples. In practice, different values of scfg1 and scfg2 lead to the best results. We find that the rule (1−scfg1)/(1−scfg2) = 2.5,

84.9% FLOPs: Weak model for the first denoising steps

84.9% FLOPs: Weak model for the last denoising steps

69.7% FLOPs: Weak model for the first denoising steps

69.7% FLOPs: Weak model for the last denoising steps

54.6% FLOPs: Weak model for the first denoising steps

54.6% FLOPs: Weak model for the last denoising steps

[Figure 46]

- Figure 20. We compare our scheduler versus a different scheduler that uses the weak model for the last denoising steps when generating class-conditioned images. Using the weak model for the last denoising steps leads to images with lower image fidelity.

|Patch size for unconditional|
|---|

|Patch size for conditional|
|---|

4

2

denoising steps

works consistently across experiments. Although we fix the value of the CFG scale during inference, different combinations are likely to lead to higher quality images as demonstrated by previous work [14], which we leave for future exploration.

We point out that our scheduler is very stable in terms of performance attained for similar compute. For instance, performing inference with a 70/70 scheduler or a 90/50 scheduler, which both require the same overall compute, produces FID results of 2.64 and 2.65 respectively. Finally, we present detailed experiments on the effect of the CFG scale for different levels of compute and more metrics in Fig. 21.

400

8.5

| | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| | | |FLOP|s % com to base|pared line| | | | |
| | | |10<br><br>64|0 % %|46 29<br><br>|% %| | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |

FLOPs % compared to baseline

FLOPs % compared to baseline

10

8.0

350

100 % 64 %

46 % 29 %

100 % 64 %

46 % 29 %

7.5

300

8

7.0

sFID-50k

250

FID-50k

6.5

IS-50k

6

6.0

200

5.5

4

150

5.0

3.24

2.64

- 1.0 1.2 1.4 1.6 1.8 2.0 2.2 2.4 CFG scale

- 2

100

2.25 2.25

4.5

1.0 1.2 1.4 1.6 1.8 2.0 2.2 2.4 CFG scale

1.0 1.2 1.4 1.6 1.8 2.0 2.2 2.4 CFG scale

FLOPs % compared to baseline

0.65

0.90

100 % 64 %

46 % 29 %

0.60

0.85

0.55

Precision-50k

Recall-50k

0.80

0.50

0.75

0.45

FLOPs % compared to baseline

0.70

0.40

100 % 64 %

46 % 29 %

0.65

1.0 1.2 1.4 1.6 1.8 2.0 2.2 2.4 CFG scale

1.0 1.2 1.4 1.6 1.8 2.0 2.2 2.4 CFG scale

###### Figure 21. Effect of CFG scale on the generated images from our class-conditioned FlexiDiT model. We plot (a) FID, (b) sFID, (c) inception score, (d) precision, and (e) recall when generation 50, 000 samples with 250 steps of the DDPM scheduler.

B.5. Text-to-Image Experiments

Generally, T2I generation is performed for a fixed target CFG scale. For our experiments we choose scfg = 4.5 for the T2I Transf. model, as this is the value used in [15] and scfg = 6.0 for the Emu model, as for these values we observed the best quality images. In general, we can match with our dynamic inference other target values of the CFG scale. One simply needs to adjust the used CFG scale for the dynamic inference accordingly.

We follow the evaluation protocol of PIXART-α [15] and perform inference using the same solvers as they do, namely iDDPM [24] for 100 steps, DPM solver [68] for 20 steps, and SA solver [106] for 25 steps. In the main text — Fig. 7 (left) — we presented results for the iDDPM solver. We present results for all the schedulers with the settings used in PIXART-α in Fig. 22, 23 and 24. For all the schedulers, there are settings where we reach the Pareto front of FID vs CLIP score of the baseline model with a lot less required compute.

Baseline

92.9 % FLOPs 85.8 % FLOPs 78.8 % FLOPs 71.7 % FLOPs 64.6 % FLOPs 57.5 % FLOPs

35

30

MSCOCOFID-10k

25

20

15

21 22 23 24 25 26 CLIP score (ViT/L-14)

###### Figure 22. FID vs CLIP score using iDDPM for 100 steps for the T2I Transf. model.

Baseline

93.3 % FLOPs 86.5 % FLOPs 79.8 % FLOPs 73.0 % FLOPs 66.3 % FLOPs

35

30

MSCOCOFID-10k

25

20

15

21 22 23 24 25 26 CLIP score (ViT/L-14)

Figure 23. FID vs CLIP score using the DPM-solver for 20 steps for the T2I Transf. model.

32.5

Baseline

91.5 % FLOPs 83.0 % FLOPs 74.5 % FLOPs 66.0 % FLOPs 57.5 % FLOPs 49.1 % FLOPs 40.6 % FLOPs

30.0

27.5

MSCOCOFID-10k

25.0

22.5

20.0

17.5

15.0

21 22 23 24 25 CLIP score (ViT/L-14)

Figure 24. FID vs CLIP score using the SA-solver for 25 steps for the T2I Transf. model.

To better characterize the effect of reducing compute, i.e. heavier use of the weak model, we also present more detailed results for the DDPM scheduler in Fig. 25. Less compute-heavy inference schedulers, often produce images with smaller possible maximum CLIP scores (for large CFG guidance scales scfg). In practice, as large CFG scale values lead to larger values of FID, these are less preferred. In every case, our weak models can max the FID vs CLIP score tradeoff of the base model for the default configuration used, i.e. scfg = 4.5.

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |

Baseline

92.9 % FLOPs 85.8 % FLOPs 78.8 % FLOPs 71.7 % FLOPs 64.6 % FLOPs 57.5 % FLOPs

35

30

MSCOCOFID-10k

|2<br><br>|1 2<br><br>|2 2 CLIP sco|3 2 re (ViT/L-|4 2 14)|5 2|6|
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |

|2<br><br>|1 2<br><br>|2 2 CLIP sco|3 2 re (ViT/L|4 2<br><br>-14)|5 2|6|
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |

25

20

|2|1 2<br><br>|2 2 CLIP sco|3 2 re (ViT/L-|4 2 14)|5 2|6|
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |

|2|1 2<br><br>|2 2 CLIP sco|3 2 re (ViT/L|4 2<br><br>-14)|5 2|6|
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |

15

21 22 23 24 25 26 CLIP score (ViT/L-14)

21 22 23 24 25 26 CLIP score (ViT/L-14)

21 22 23 24 25 26 CLIP score (ViT/L-14)

Figure 25. FID vs CLIP score using DDPM for 100 steps for the T2I Transf. model, for different levels of compute. On the right, we present results separately for each compute level.

We also provide results on using a smaller overall number of steps with the DDPM solver in Fig. 27. To generate the baseline curves, we sample 10,000 samples using a CFG scale scfg from the set {1.0,1.125,1.25,1.375,1.5,1.625,1.75,2.0,2.25,2.5,2.75,3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0,7.5,8.0,8.5}. The same values are used when sampling with our flexible models. Finally, we also provide FID vs CLIP for the Emu model in Fig. 26. In this case, we take CFG scales scfg from the set {1.0,1.5,2.0,2.5,3.0,4.5,6.0,7.5,8.0,9.0}. We use captions from the training set of MS COCO to generate images. Neither of the models was trained on images from this dataset.

Baseline

84.3 % FLOPs 68.6 % FLOPs 52.9 % FLOPs 37.2 % FLOPs 21.5 % FLOPs Baseline 6.0 CFG

26

MSCOCOFID-10k

24

22

20

24.25 24.50 24.75 25.00 25.25 25.50 25.75 26.00 CLIP ViT-L/14

Figure 26. FID vs CLIP score using DDIM for 50 steps for the Emu model.

VQA results. VQA scores are calculated by querying a visual-question-answering model to produce an alignment score by computing the probability of a ”Yes” answer to a simple ”Does this figure show {text}?”. question. To calculate this score, we use the clip-flant5-xxl model from huggingface7 as suggested in [60]. We provide more detailed results on the VQA benchmark in Tables 1 and 2. More specifically, we provide per dataset VQA scores, along with the CFG scale scfg used to generate the images for each case . As we can see, using the weak model requires a bigger CFG scale to reach the same level of optimality (calculated from the FID vs CLIP score tradeoff). We also note that using the weak model often leads to images with better text alignment. We hypothesize that fewer tokens (as a result of larger patch sizes) help with spatial consistency at the beginning of the denoising process. To calculate VQA scores, we take the first 200 prompts from each dataset and

7https://huggingface.co/zhiqiulin/clip-flant5-xxl

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

- 14
- 15
- 16

Baseline

30steps

Baseline 4.5 CFG

MSCOCOFID-10k(100denoisingsteps)

35

92.9 % FLOPs 85.8 % FLOPs 78.8 % FLOPs 71.7 % FLOPs 64.6 % FLOPs 57.5 % FLOPs

30

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

- 14
- 15
- 16

50steps

25

20

- 14

- 15

- 16

100steps

15

21 22 23 24 25 26 CLIP score (ViT/L-14)

25.3 25.4 25.5 25.6 25.7 25.8 25.9 CLIP score (ViT/L-14)

Figure 27. We plot FID vs CLIP score when generating images with different CFG scales. (left) Overall Pareto front when generating images with 100 denoising steps. (right) Pareto front generating images with a different number of steps zoomed in the typical tradeoff generation values.

use the train split from the DrawBench [84], train split from the Pick-a-Pic [55], test split from the Winoground [94] and tifa v1.0 text inputs8 from the TIFA160 [43] dataset.

CFG scale scfg

DrawBench Pick-a-Pic Winoground TIFA160 Average

T2I Transf. (100 %) 4.5 58.93 50.56 62.01 81.65 63.29 T2I Transf. (92.9 %) 4.5 58.37 51.53 62.09 82.16 63.54 T2I Transf. (85.8 %) 4.5 57.58 51.67 62.41 81.53 63.30 T2I Transf. (78.8 %) 4.7 58.62 51.97 62.04 80.60 63.31 T2I Transf. (71.7 %) 4.7 58.44 51.56 63.03 80.57 63.40 T2I Transf. (64.6 %) 4.9 60.16 52.34 62.98 80.06 63.89 T2I Transf. (57.7 %) 5.0 59.04 50.80 63.27 79.90 63.26 T2I Transf. (50.5 %) 5.0 56.77 51.72 61.87 79.38 62.44 T2I Transf. (43.4 %) 5.0 56.87 51.92 61.30 78.33 62.11

- Table 1. Detailed VQA evaluations for the benchmarks tested with the T2I Transf. model. As in the class-conditioned experiments, using more of the weak model during denoising requires a higher CFG scale scfg to reach optimum performance.

CFG-scale scfg DrawBench Pick-a-Pic Winoground TIFA160 Average

Emu (100 %) 6.0 69.44 58.70 65.75 86.77 70.17 Emu (84.3 %) 6.0 68.00 58.93 67.33 86.51 70.19 Emu (68.6 %) 6.25 69.53 60.62 66.00 85.33 70.37 Emu (52.9 %) 6.5 69.79 58.14 66.23 86.20 70.09

- Table 2. Detailed VQA evaluations for the benchmarks tested with the Emu model. As in the class-conditioned experiments, using more of the weak model during denoising requires a higher CFG scale scfg to reach optimum performance.

Alignment between powerful and weak model. It is common practice nowadays to train images (and especially videos) in different stages, where a large (potentially lower quality) dataset is used for the first stage, followed by a shorter finetuning stage, characterized by higher quality and aesthetically more pleasing images. Although we are directly distilling the weak model from the predictions of the powerful model, the data used throughout training are still important. In practice,

8https://github.com/Yushi-Hu/tifa/blob/main/tifa_v1.0/tifa_v1.0_text_inputs.json

our fine-tuning is sample efficient, and we find that even a few thousand images (< 5000) are enough to succeed. We thus suggest fine-tuning on the last (potentially smaller) but higher-quality dataset. When generating images based on shorter prompts with Emu, we use a prompt re-writer, prompting a small LLM to expand on the information provided. We consider this prompt re-writer as part of the model.

### C. Implementation Details

We provide additional details on the experiments in the main text.

- C.1. Figure Details Prompts used for Fig. 1. We provide in Table 3 the exact prompts used to generate the images.

Prompts for Fig. 1

The image shows a frog wearing a golden crown with intricate designs, sitting on a wooden log in a serene environment reminiscent of a Japanese anime setting. The frog’s crown is adorned with small gems and its eyes are large and expressive. The log is covered in moss and surrounded by lush greenery, with a few cherry blossoms visible in the background. The frog’s skin is a vibrant shade of green with blue stripes, and it has a regal demeanor, as if it is a monarch of the forest. The overall atmosphere is peaceful and whimsical.

The image shows a serene waterfall cascading down a rocky slope in a lush tropical forest, reminiscent of Claude Monet’s impressionist style. Sunlight filters through the dense foliage above, casting dappled shadows on the misty veil surrounding the falls. The water plunges into a crystal-clear pool, surrounded by large rocks and vibrant greenery. The atmosphere is tranquil, with a warm color palette and soft brushstrokes evoking a sense of serenity. The forest floor is covered in a thick layer of leaves, and the sound of the waterfall echoes through the air.

Table 3. Details on the prompts used to generate the images in the paper.

Details on Fig. 2. In Fig. 2 (b), we fix randomness of the denoising process in terms of the initial sampled image p(xT), and from the denoising process in Eq. (2). Then we generate images with 250 steps using the DiT-XL/2 official public checkpoint. During denoising, we modify only 1 of the 250 denoising steps, bypassing the model predictions from that step through a high/low pass filter. We then compare the resulting generated images in terms of LPIPS, L2 distance, SSIM, and DreamSim. In general, modifying one of the first denoising steps, leads to larger final image differences, due to the accumulation of differences. We still note a distinctive pattern: a high-pass filter, i.e. removing low-pass components, leads to larger image differences during the first denoising steps. The opposite holds for the last denoising steps. We can thus argue, that low-pass components, i.e. ‘coarser’ image details are more important compared to high-frequency details, for the first denoising steps. To calculate spatial frequencies, we keep the corresponding values of the FFT of an image.

Details on Fig. 9. In Fig. 9, we plot GPU utilization when propagating different sequence lengths. All experiments are conducted in bfloat16 using PyTorch 2.5 and CUDA 12.4 and with our Emu DiT that has 24 layers and a hidden dimension of 2048. We also plot peak FLOPs and memory bandwidth for the GPU tested, an NVIDIA H100 SXM5 in this case. When we report FLOPs, we count additions and multiplications separately, as it is commonly done [41]. We count FLOPs as the theoretical required operations for a forward pass, and not the actual GPU operations, which might be a higher number due to the potential recalculation of partial results. As bytes for the x-axis, we only consider the model parameters (2 bytes per model parameter). Note that for our choice of weak models, the GPU is fully utilized (it reaches the maximum compute intensity that can be achieved for this application). In reality, compute intensity drops when larger sequence lengths are used, mainly due to the larger memory footprint of intermediate activations. Thus, latency benefits are indeed even larger than FLOPs benefits reported in the paper. Compiling and fusing operations is crucial to ensure that the GPU is not bottlenecked by waiting instructions from the CPU. When we are performing inference with the weak model without first merging the LoRAs, inference time is proportional to the additional FLOPs required. For the attention operation, we use the memory efficient attention operation from the xformers library. Other efficient implementations of attention do not lead to significant differences. Our T2I model operates in a 128 × 128 latent space, which means that using a patch size of (1,2,2) results in 4096 tokens, compared to 1024 tokens for the (1,4,4) patch size. Our T2V model operates in a 32×88×48 latent space, which means that using a patch size of (1,2,2), (2,2,2), (1,4,4) leads to 33792, 16896 and 8448 tokens respectively. We obtain similar results using different-sized models, like our Video DiT model.

When we report FLOPs in the paper, we report numbers for the denoising process, as it is commonly done. We thus ignore latency induced by decoders that map samples from a latent space, or potential prompt-rewrite modules. The added latency by the decoder, is in our settings negligible.

#### C.2. Flexifying Diffusion Transformers

Although for the class-conditioned experiments, we use a single embedding and de-embedding layer that we always project to the required patch size, we note that this projection can be done once to pre-calculate embedding and de-embeddding parameters during inference. These projected embeddings can then be used out of the box, for the tradeoff of some minuscule additional memory. The choice of p′ as the underlying patch size is not too important for our experiments. In practice, we use a value of p′ = 4. As mentioned in the main text, we add positional embeddings according to the coordinates of each patch in the original image. A schematic of this can be found on the right.

[Figure 47]

Apart from the architecture modifications listed in the main text, we experimented with adding patch size specific temperatures in the self-attention computation:

QK⊤ τp

softmax

√

##### V,

d

where Q,K,V are the queries, keys and values respectively and τp is a patch size specific temperature initialized as 1. We do not include this in the end, as it occasionally leads to instabilities during fine-tuning, even under different parametrizations.

Class-conditioned implementation details. We largely use the same hyperparameters as [78] to fine-tune. When finetuning to match distributions, we train to minimize the MMD loss, as introduced in Section B.1. During bootstrapping, we denoise images with a DDPM scheduler, operating on T = 250 steps, the same as the target inference scheduler. As we found that MMD distance is higher for diffusion steps closer to x0 ∼ q(x0) — see also Fig. 11 —, we bias the sampling to reflect that during training as well.

Parameter Value

training data ImageNet learning rate 10−4 weight decay 0.0

EMA update frequency every step EMA update rate 0.9999

Table 4. Class-conditioned implementation details.

In our experiments, we focused on fine-tuning pre-trained models, as we were interested in efficiency. We note that training with different patch sizes has been used in the past to also accelerate pre-training [3, 9]. We believe that flexible patch sizes can also be used in this application to accelerate pre-training.

T2I Transf. implementation details. For the T2I Transf. model, we follow exactly the recipe of [15], and fine-tune a 256 × 256 pre-trained variant9. For fine-tuning, we use the same image dataset, namely the SAM dataset10 with captions generated from a vision-language model. The model has overall the same parameters as the DiT-XL/2 model, with the addition of cross-attention blocks. When adding new embedding and de-embedding layers, we initialize them as we did for the class-conditioned experiments. Embedding layers are initialized to Qembed†wembed and de-embedding layers are initialized to wde-embedQde-embed†. Here wembed, wde-embed are the pre-trained model parameters and Qembed, Qde-embed are the same patch size dependent — fixed projection matrices that better preserve the norm of the output activations at initialization. As aforementioned, we add a patch size embedding that is added to all tokens in the sequences after the tokenization step. This embedding is equal to 0 for the pre-trained patch size, to ensure functional preservation. We fine-tune the T2I Transf. model on a small subset of the SAM dataset used to originally train the target model. We add LoRAs on the self-attention and feed-forward layers, with a LoRA dimension of 32. We use a higher learning rate, due to the different learning objectives distilling a powerful model’s predictions into the ones of a weal model.

- 9Our starting pre-trained model exactly matches the public checkpoint https://huggingface.co/PixArt-alpha/PixArt-XL-2-SAM256x256.
- 10https://segment-anything.com/.

Parameter Value training data SAM with captions from a VLM model

optimizer AdamW learning rate 8 × 10−4 weight decay 10−2

gradient clipping 0.02 batch size 512

EMA update frequency every step

EMA update rate 0.9999 LoRA rank 32

Table 5. Image text-conditioned implementation details.

Emu implementation details. Our Emu model is fundamentally identical to the T2I Transf. model. Small variations are due to different ways to calculate text embeddings, which lead to a different number and size of the cross-attention tokens, and slight architectural modifications — primarily the use of QK-normalization [22] and the use of learnable positional embeddings. We train using a high-quality aesthetic dataset. To calculate metrics based on the 1024×1024 images generated with this model, we follow the evaluation protocol of [105] and resize images to 512 × 512. For both our Emu and our Video DiT model, we use a LoRA rank of 64.

T2V implementation details. As aforementioned, our Video DiT model has a pre-trained patch size of (pf,ph,pw) = (1,2,2). Compared to our T2I experiments, we only change the 2D convolutional layers used for tokenization with a 3D convolution layer. When increasing the temporal patch size pf, we duplicate parameters along that dimension. When interpolating positional embeddings, we also do that along the temporal dimension. No additional changes are made. For evaluation, we use the prompts from https://github.com/facebookresearch/MovieGenBench/blob/main/ benchmark/MovieGenVideoBench.txt to generate videos of length equal to 256 frames. We evaluate according to VBench [45] and report the average over Subject Consistency, Background Consistency, Temporal Flickering, Motion Smoothness, Dynamic Degree, Aesthetic Quality, Imaging Quality, Temporal Style and Overall Consistency.

#### C.3. Human Evaluation Details

We prompt humans, asking them to: “Compare the two side-by-side images. Focus on visual appeal and flawlessness, considering factors like aesthetic appeal, clarity, composition, color harmony, lighting, and overall quality, then select ’left’ if the left image is better, ’right’ if the right image is better, or ’tie’ if both are equally appealing or you have no preference.”. In total, we collected votes for the 4 different settings presented in the paper and aggregated them across 200 prompts. For each setting and each prompt, we ask 3 people for votes. In cases where there are 3 votes for each of ’left’, ’right’, and ’tie’, we ask a fourth labeler to break the tie.

### D. Generated Samples

We provide more examples of generated samples.

- D.1. Text-Conditioned Image Generation

We showcase more examples with varying amounts of compute in Fig. 28, 29 and 30. We further show more examples of how performance and diversity in image generation are preserved in Fig. 31 and 32. Finally, we show examples of the effect of CFG scale scfg and reducing the overall number of FLOPs used to generate an image using our method, in Fig. 33. For all the images seen in the paper with our Emu model, we use 50 steps of the DDIM scheduler.

- D.2. Text-Conditioned Video Generation We showcase more examples of video generation with our flexible model in Fig. 34 and 35.
- D.3. Class-Conditioned ImageNet Generation

We provide a more comprehensive comparison for generated images from the same original sample from pnoise, using varying amounts of compute from our flexible model in Fig. 36. Note that for class-conditioned generation we do not use

100.0 % FLOPs 84.3 % FLOPs 68.6 % FLOPs 52.9 % FLOPs 37.2 % FLOPs

The image portrays a finely detailed portrait of a cat office worker, donning a sleek silk office suit and glasses, exuding an air of professionalism. The suit is rendered in intricate detail, with every fiber and fold meticulously captured. The cinematic lighting casts a dramati c glow, accentuating the cat's features and the textures of the suit. The dark shot is balanced by neutral colors, with hints of muted hues, evoking a sense of sophistication. The image boasts an epic, realistic quality, with intricate details that rival those of a high-end photo graphy studio, reminiscent of the work of renowned photographers like Oliver Wetter and Annie Leibovitz.

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

The image is a stunning underwater macro shot of a glowing jellyfish, perfectly centered and symmetrical. The jellyfish's translucent body glows with a soft, ethereal light, surrounded by vibrant corals and bulbs that add pops of color to the neutral-toned background. The image i s hyperdetailed and cinematic, with a complex and extremely detailed background that showcases the beauty of the ocean's depths. The bokeh effect adds a sense of depth and dimensionality to the image, drawing the viewer's eye to the jellyfish's glowing form.

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

The image depicts a stunning flower in a serene and lush botanical garden, captured with a DSLR camera. The flower is a vibrant shade of pink, with delicate petals and a prominent center. It is situated in the foreground, surrounded by an assortment of foliage and stems of varyin g textures and colors. The garden's tranquil atmosphere is accentuated by the soft, diffused light and the subtle mist rising from the ground. The DSLR camera's high-quality sensor has captured the intricate details of the flower and its surroundings, showcasing the beauty of nat ure.

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

The image presents a breathtakingly realistic portrait of an orange cat with vibrant, bright eyes and majestic angel wings. The cat's fur is intricately detailed, with subtle texture and shading that gives the impression of softness. The wings are radiant and ethereal, with delic ate feathers and a subtle glow. The photography is cinematic, with a shallow depth of field created by the 50mm lens, resulting in a beautiful bokeh effect in the background. The lighting is soft and warm, illuminating the cat's features with a gentle, heavenly glow.

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

The image shows a determined 35-year-old space colonist, wearing a high-tech space suit with a gleaming metallic finish, standing proudly on the Martian surface. The suit is adorned with various tools and gadgets, and the colonist holds a Martian soil sample in their gloved hand.

The historic portrait, captured by a space exploration photographer in 2055, is framed through the visor of the space helmet, providing an intimate and immersive view of the Martian landscape. The vast, crimson terrain stretches out behind the colonist, punctuated by rocky forma tions and the distant horizon.

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

Figure 28. More samples generated by our Emu model for varying amounts of compute.

LoRAs, and images generated from the original baseline model may not be exactly the same. They do however have the same characteristics (FID score) as seen in Sec 4.1. We also show samples of our flexible DiT-XL/2 model when using only 64% of the compute of the original model in Fig. 37, 38, 39, 40, 41, 42 and 43. All images are generated using 250 DDPM steps and a CFG-scale equal to 4.0.

100.0 % FLOPs 84.3 % FLOPs 68.6 % FLOPs 52.9 % FLOPs 37.2 % FLOPs

The image portrays a delectable arrangement of sushi and sashimi on a sleek, minimalist plate at a 3 Stars MICHELIN restaurant. The dish is artfully presented with honey and soy sauce, radiating an alluring aroma and appearance. Soft, dramatic lighting illuminates the culinary ma sterpiece, accentuating its indulgent appeal. The centered composition and clean layout evoke a sense of luxury and sophistication, highlighting the expertise of the food stylist. In the evening setting, the high-end dish exudes irresistible beauty, making it a true gastronomic d elight.

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

The image transports viewers to a mystical realm, reminiscent of Middle Earth's hidden hobbit towns. Amidst lush hills and gardens, cascading houses with steampunk flair seem to defy gravity. The atmosphere is both ancient and mysterious, with a soft, limpid color palette that ev okes a sense of wonder. Every detail, from the intricate architecture to the lush foliage, is rendered in hyper-realistic clarity, as if plucked from the imagination of Howard Behrens or djamilaknopf. The cinematic quality is further enhanced by the subtle glow of luminescent ref lections and the subtle sheen of crystalline structures.

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

The image shows a medium shot of a snow leopard in a snowy rocky mountain forest on a cloudy day. The forest is entirely covered in snow, and the snow leopard is in full-body view, showcasing its sleek white fur and black spots. The snow leopard's eyes are highly detailed and int ensely focused on its prey, with a predatory gaze. The image is captured in crisp focus and natural lighting, highlighting the snow leopard's majestic appearance. The surrounding environment is also highly detailed, with visible trees, rocks, and snow-covered mountains in the bac kground.

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

The image depicts a luxurious sakura-themed ring in a closeup product view, showcasing exquisite hyper details in 4K resolution. The ring features a combination of gemstones and diamonds, crafted with precision and elegance. The soft illumination highlights the intricate design, casting a dreamy glow. Rendered using Unreal Engine, the digital art exudes high-end fashion and luxury, evoking a sense of opulence. The overall effect is one of ultra-quality, making it a showstopper on platforms like ArtStation and CGSociety. The ring's beauty and craftsmanshi p are truly a work of art.

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

The image depicts a serene and adorable kitten, no more than a few weeks old, nestled in a cozy ball amidst a peaceful nocturnal setting. The kitten's fur is a soft gray, with distinctive white patches on its nose and paws, and its eyes are closed in tranquil slumber. The moon ca sts a soft, ethereal glow on the kitten's fur, illuminating the fine details of its whiskers and the gentle rise and fall of its chest as it breathes. The surrounding environment is dark, with only the faintest hint of stars and a crescent moon visible in the sky.

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

The image is a highly detailed portrait of an anthro frog mage, a human-like frog holding a wand, with intricate details and radiant light. The frog stands in a lush environment with dense foliage and twisted vines, its green skin glistening in the soft light. Its eyes are bright

and wise, and its long fingers grasp the wand with elegance. The scene is reminiscent of fantasy art by renowned artists such as Greg Rutkowski, Loish, and Lois van Baarle, with global illumination casting a warm glow on the entire setting. The atmosphere is mystical and enchant ing, inviting the viewer to step into the magical world.

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

Figure 29. More samples generated by our Emu model for varying amounts of compute.

100.0 % FLOPs 84.3 % FLOPs 68.6 % FLOPs 52.9 % FLOPs 37.2 % FLOPs

The image depicts a futuristic cityscape inspired by the mandelbulb, a three-dimensional representation of the Mandelbrot set. The city's skyline is dominated by towering structures with intricate, swirling patterns reminiscent of the mandelbulb's fractal design. The buildings se em to stretch on forever, with smaller, ornate details visible on their surfaces. The city's streets are bustling with activity, as strange, otherworldly vehicles move through the city, leaving trails of light behind them. In the distance, a massive, glowing mandelbulb serves as a beacon, illuminating the city's vibrant, dreamlike atmosphere.

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

The image depicts an ancient village at dusk, with traditional Chinese houses made of wood and stone standing along the misty streets. The walls and streets are covered in moss, adding a touch of mystique to the scene. Jacaranda trees with vibrant purple flowers tower above the r oofs, while a majestic, mystical mountain looms in the background. Three dogs lie sleeping on the cold, foggy street, seemingly undisturbed by the eerie silence. The atmosphere is ultra-realistic and cinematic, with every detail meticulously rendered to transport the viewer to a bygone era.

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

The image shows a flock of sheep gathered together, taking a selfie with a smartphone held by one of the sheep. They are standing on a lush grassland, surrounded by the majestic Himalayan mountains in the background. The landscape is depicted in extraordinary detail, with snow-ca pped peaks, rolling hills, and valleys. The sheep are all looking towards the camera, with some of them smiling and others looking curiously at the phone. The grassland is teeming with life, with wildflowers and other vegetation sprouting from the ground. The atmosphere is serene

and peaceful, with the sun shining down on the scene.

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

The image depicts the White Castle of the King, a majestic and serene structure surrounded by lush greenery and vibrant flowers, including a few roses. The sky above is a deep shade of pink, reminiscent of Miyazaki's Nausicaa, with dramatic clouds that evoke a sense of epic fanta sy, similar to Breath of the Wild. In the garden, a powerful warrior, akin to a berserker from anime, stands tall, exuding a sense of strength and protection. The scene is set in an ultra-wide shot, with an atmospheric and cinematic quality, boasting hyper-realistic details and p ost-processing that rivals the Unreal Engine, all in stunning 8k resolution and rendered with Octane.

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

The image depicts a chimpanzee wearing a sleek astronaut spacesuit, posing heroically in a grandiose setting reminiscent of classical oil paintings. The hyperrealistic and highly detailed 8k image is rendered in warm, cinematic lighting that accentuates the subject's textures and

contours. The chimpanzee's fur and the spacesuit's metallic sheen are meticulously captured, with each strand of hair and every rivet on the suit's surface visible in extraordinary detail. The background is a muted, gradient blue, evoking the vastness of space. The overall atmos phere is one of awe-inspiring wonder and discovery.

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

Figure 30. More samples generated by our Emu model for varying amounts of compute.

[Figure 132]

[Figure 133]

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

###### Figure 31. Samples for the prompt: ”A playful kitten just waking up.”. We showcase the image generated by the baseline and our flexible scheduler using only 53% of FLOPs.

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

###### Figure 32. Samples for the prompt: ”A baby hippo swimming in the river.”. We showcase the image generated by the baseline and our flexible scheduler using only 53% of FLOPs.

Decreasing FLOPs

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

IncreasingCFGscale

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

- Figure 33. Effect of CFG and total compute for the prompt: ‘The image shows a gigantic juicy burger placed on a white plate on a wooden table. The burger is composed of a large beef patty, crispy bacon, melted cheese, lettuce, tomato, onion, pickles, and a slice of red tomato, all sandwiched between a soft bun. The burger is so large that it occupies most of the plate, with some toppings falling out of the sides. The bun is slightly toasted, and the cheese is melted to perfection, giving off a savory aroma. The burger is garnished with a side of crispy fries and a refreshing glass of cola.‘.

Drone view of waves crashing against the rugged cliffs along Big Sur s garay point beach. The crashing blue waters create white-tipped waves, while the golden light of the setting sun illuminates the rocky shore. A small

island with a lighthouse sits in the distance, and green shrubbery covers the cliff s edge. The steep drop from the road down to the beach is a dramatic feat, with the cliff s edges jutting out over the sea. This is a v iew that captures the raw beauty of the coast and the rugged landscape of the Pacific Coast Highway.

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

A gorgeously rendered papercraft world of a coral reef, rife with colorful fish and sea creatures.

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

A litter of golden retriever puppies playing in the snow. Their heads pop out of the snow, covered in.

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

Several giant wooly mammoths approach treading through a snowy meadow, their long wooly fur lightly blows in the wind as they walk, snow covered trees and dramatic snow capped mountains in the distance, mid afternoon lig ht with wispy clouds and a sun high in the distance creates a warm glow, the low camera view is stunning capturing the large furry mammal with beautiful photography, depth of field.

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

This close-up shot of a Victoria crowned pigeon showcases its striking blue plumage and red chest. Its crest is made of delicate, lacy feathers, while its eye is a striking red color. The bird s head is tilted slightly t o the side, giving the impression of it looking regal and majestic. The background is blurred, drawing attention to the bird s striking appearance.

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

A tropical fish swimming in ocean reefs

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

Aerial shot of the ocean. a maelstrom forms in the water swirling around until it reveals the fiery depths below.

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

Dragon-toucan walking through the Serengeti.

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

A curious cat peering out from a cozy hiding spot.

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

bears figure out how to launch a rocket

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

A building collapsing into a puddle of lava.

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

A spaceship being pulled into a blackhole.

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

An orange cat jumps onto a kitchen counter after seeing butter there.

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

A gibbon swinging through the canopy.

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

DiT-XL/2 (256x256) Ours: 100% of Compute Ours: 85% of Compute Ours: 70% of Compute Ours: 55% of Compute Ours: 39% of Compute

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

###### Figure 36. Sample for the ImageNet dataset comparing the baseline model and varying inference schedulers of our model, using different levels of compute.

[Figure 314]

###### Figure 37. Samples for the ImageNet class ‘tree frog, tree-frog’ from our FlexiDiT model that uses only 46% of the compute compared to the baseline model.

[Figure 315]

###### Figure 38. Samples for the ImageNet class ‘prairie chicken, prairie grouse, prairie fowl’ from our FlexiDiT model that uses only 46% of the compute compared to the baseline model.

[Figure 316]

###### Figure 39. Samples for the ImageNet class ‘hummingbird’ from our FlexiDiT model that uses only 46% of the compute compared to the baseline model.

[Figure 317]

###### Figure 40. Samples for the ImageNet class ‘cairn, cairn terrier’ from our FlexiDiT model that uses only 46% of the compute compared to the baseline model.

[Figure 318]

###### Figure 41. Samples for the ImageNet class ‘French bulldog’ from our FlexiDiT model that uses only 46% of the compute compared to the baseline model.

[Figure 319]

###### Figure 42. Samples for the ImageNet class ‘Granny Smith’ from our FlexiDiT model that uses only 46% of the compute compared to the baseline model.

[Figure 320]

###### Figure 43. Samples for the ImageNet class ‘cock’ from our FlexiDiT model that uses only 46% of the compute compared to the baseline model.

