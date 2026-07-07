# arXiv:2307.06304v1[cs.CV]12Jul2023

## Patch n’ Pack: NaViT, a Vision Transformer for any Aspect Ratio and Resolution

Mostafa Dehghani∗ Basil Mustafa∗ Josip Djolonga† Jonathan Heek† Matthias Minderer Mathilde Caron Andreas Steiner Joan Puigcerver Robert Geirhos Ibrahim Alabdulmohsin Avital Oliver Piotr Padlewski Alexey Gritsenko Mario Lučić Neil Houlsby∗

Google DeepMind

Abstract

The ubiquitous and demonstrably suboptimal choice of resizing images to a fixed resolution before processing them with computer vision models has not yet been successfully challenged. However, models such as the Vision Transformer (ViT) offer flexible sequence-based modeling, and hence varying input sequence lengths. We take advantage of this with NaViT (Native Resolution ViT) which uses sequence packing during training to process inputs of arbitrary resolutions and aspect ratios. Alongside flexible model usage, we demonstrate improved training efficiency for large-scale supervised and contrastive image-text pretraining. NaViT can be efficiently transferred to standard tasks such as image and video classification, object detection, and semantic segmentation and leads to improved results on robustness and fairness benchmarks. At inference time, the input resolution flexibility can be used to smoothly navigate the test-time cost-performance trade-off. We believe that NaViT marks a departure from the standard, CNN-designed, input and modelling pipeline used by most computer vision models, and represents a promising direction for ViTs.

### 1 Introduction

The simple, flexible and scalable nature of the Vision Transformer (ViT) (Dosovitskiy et al., 2021) has rendered it an almost ubiquitous replacement to convolution based neural networks. Underpinning this model is a simple operation: splitting an image into patches, each of which is linearly projected to a token. Typically, input images are resized to a fixed square aspect ratio and then split into a fixed number of patches.

Recent works have explored alternatives to this paradigm: FlexiViT (Beyer et al., 2023) supports multiple patch sizes within one architecture, enabling smooth variation of sequence length and thus compute cost. This is achieved via random sampling of a patch size at each training step and a resizing algorithm to allow the initial convolutional embedding to support multiple patch sizes. Pix2Struct (Lee et al., 2022) introduced an alternative patching approach which preserves the aspect ratio, which is particularly useful for tasks such as chart and document understanding.

We present an alternative, NaViT. Multiple patches from different images are packed in a single sequencetermed Patch n’ Pack—which enables variable resolution while preserving the aspect ratio (Figure 2). This is inspired by example packing in natural language processing, where multiple examples are packed into a single sequence to accommodate efficient training on variable length inputs.

We demonstrate that: (i) Randomly sampling resolutions at training time significantly reduces training cost. (ii) NaViT results in high performance across a wide range of resolutions, enabling smooth cost-performance trade-off at inference time, and can be adapted with less cost to new tasks, (iii) Fixed batch shapes enabled by example packing lead to new research ideas, such as aspect-ratio preserving resolution-sampling, variable token dropping rates, and adaptive computation.

∗ Project lead. †Core contributor.

Pre-training linear eval

Finetuned acc vs. pretrain cost

Finetuned acc vs. inference cost

| | |
|---|---|
| | |
| | |
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |
| | |
| | |

| |Model<br><br>ViT<br><br>NaViT<br><br>Version<br><br>B/32 B/16 L/16<br><br>| |
|---|
|
|---|---|
| | |
| | |
| | |
| | |
| | |

88%

80%

80%

86%

75%

60%

Accuracy

84%

70%

40%

82%

65%

20%

80%

60%

0%

1012 1013

1012 1013

103 104

Pre-training TPU chips hours

Pre-training TPU chips hours

Inference cost (ms/image)

- Figure 1: NaViT offers notable computational efficiency during pre-training (left) which carries over to downstream fine-tuning (middle). A single NaViT can be applied successfully to multiple resolutions (right), smoothly trading off performance and inference cost.

Pooling Representations

Pad	examples	are	masked	out from	loss	computation

Figure 2: Example packing enables variable resolution images with preserved aspect ratio, reducing training time, improving performance and increasing flexibility. We show here the aspects of the data preprocessing and modelling that need to be modified to support Patch n’ Pack. The position-wise operations in the network, such as MLPs, residual connections, and layer normalisations, do not need to be altered.

Pad	tokens	are	masked	out from	pooling

Self-Attention

Pad	tokens	are	masked	out from	attention

Each token's receptive field is restricted to the tokens within the same example

query

key

Packed Sequence

| |Imag|e 1| | | |Imag|e 2| | | |Im|age|3| |Pa|d|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|

Data preprocessing

| | |
|---|---|
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |

| | | |
|---|---|---|
| | | |
| | | |

| | | |
|---|---|---|
| | | |
| | | |

| | | | |
|---|---|---|---|
| | | | |

| | | | |
|---|---|---|---|
| | | | |

Image1 Image2 Image2 Image1 Image2 Image2 Image1 Image2 Image2

Inputs

Patchify Token drop

These observations have major practical implications. At a fixed computational budget NaViT consistently outperforms ViT. For instance, we match the performance of the top-performing ViT with 4× less compute (Figure 1, left). We identify the substantial increase in the number of training examples processed within the allocated compute budget as the primary contributor to the improved performance over ViT—example packing coupled with variable resolution inputs and variable token dropping enable NaViT-L/16 to process five times more images during training (Table 2). This improved efficiency extends to the fine-tuning process (Figure 1, middle). Furthermore, by exposing NaViT to multiple resolutions during both pre-training and fine-tuning, a single model demonstrates excellent performance when evaluated on various resolutions, significantly advantaging NaViT in terms of inference cost (Figure 1, right).

NaViT’s training and adaptation efficiency, and flexible inference, presents a promising avenue for Vision Transformers. Patch n’ Pack empowers computer vision systems to transcend limitations imposed by current data and modeling pipelines, enabling ideas that were previously restricted by the constraints of fixed batch shapes, unlocking new possibilities for innovation and advancement.

ImageNet (85.9% non-square)

LVIS (92.2% non-square)

WebLI (57.3% non-square)

- 0

- 1

- 2

- 3

- 4

- 5

Density

h=2w h=w w=2h

h=2w h=w w=2h

h=2w h=w w=2h

Aspect ratio

### 2 Method

Figure 3: Height:width ratios of different datasets; most images are not squareish (> 20% deviation).

Deep neural networks are typically trained and run with batches of inputs. For efficient processing on the current hardware this implies fixed batch shapes which in turn imply fixed image sizes for computer vision applications. This coupled with architectural limitations historically associated with convolutional neural networks led to a practice of either resizing or padding images to a fixed size. Both of these have been shown to be flawed: the former harms performance and the latter is inefficient (Lee et al., 2022). An analysis of aspect ratios in ImageNet (Deng et al., 2009), LVIS (Gupta et al., 2019) and WebLI (Chen et al., 2022c) as representative examples of classification, detection and web image datasets, respectively, shows that most images are typivally not square (Figure 3).

In language modelling, it is common to bypass limitations of fixed sequence lengths via example packing: tokens from multiple distinct examples are combined in one sequence, which can significantly accelerate training of language models (Krell et al., 2021). By treating images as sequences of patches (tokens), we show that Vision Transformers (Dosovitskiy et al., 2021) can benefit from the same paradigm, which we call Patch n’ Pack. Using this technique ViTs can be trained on images at their “native” resolution, and we name this approach NaViT.

#### 2.1 Architectural changes

NaViT is built upon the original ViT, but in principle can use any ViT variant operating on a sequence of patches. To enable Patch n’ Pack, we make the following architectural modifications.

Masked self attention and masked pooling. To prevent examples attending to each other, additional self-attention masks are introduced. Similarly, masked pooling on top of encoder aims to pool the token representations within each example, resulting in a single vector representation per example in the sequence.

- Figure 2 presents how the receptive filed of attention is controlled via masking.

Factorized & fractional positional embeddings. To handle arbitrary resolutions and aspect ratios, we revisit the position embeddings. Given square images of resolution R×R, a vanilla ViT with patch size P learns 1-D positional embeddings of length (R/P)2 (Dosovitskiy et al., 2021). Linearly interpolating these embeddings is necessary to train or evaluate at higher resolution R.

Pix2struct (Lee et al., 2022) introduces learned 2D absolute positional embeddings, whereby positional embeddings of size [maxLen,maxLen] are learned, and indexed with (x,y) coordinates of each patch. This enables variable aspect ratios, with resolutions of up to R = P · maxLen. However, every combination of (x,y) coordinates must be seen during training.

To support variable aspect ratios and readily extrapolate to unseen resolutions, we introduce factorized positional embeddings, where we decompose into separate embeddings ϕx and ϕy of x and y coordinates. These are then summed together (alternative combination strategies explored in Section 3.4). We consider two schema: absolute embeddings, where ϕ(p) : [0,maxLen] → RD is a function of the absolute patch index, and fractional embeddings, where ϕ(r) : [0,1] → RD is a function of r = p/side-length, that is, the relative distance along the image. The latter provides positional embedding parameters independent of the image size, but partially obfuscates the original aspect ratio, which is then only implicit in the number of patches. We consider

simple learned embeddings ϕ, sinusoidal embeddings, and the learned Fourier positional embedding used by NeRF (Tancik et al., 2020).

#### 2.2 Training changes

Patch n’ pack enables new techniques to be used during training of NaViT.

Continuous Token dropping. Token dropping (random omission of input patches during training) (Akbari et al., 2021; Li et al., 2023) has been developed to accelerate training. However, typically the same proportion of tokens are dropped from all examples; packing enables continuous token dropping, whereby the token dropping rate can be varied per-image. This enables the benefits of faster throughput enabled by dropping while still seeing some complete images, reducing the train/inference discrepancy. Further, with packing, the drop-distribution can vary throughout training, following some pre-defined schedule. In Section 3.3, we explore different schedules and the benefits of flexible token dropping.

Resolution sampling. NaViT can be trained using the original resolution of each image. Alternatively, the total number of pixels can be resampled while preserving aspect ratio. In vanilla ViT, there is a tension between greater throughput (training on smaller images), and greater performance (training on larger images, to enable high-resolution at evaluation time). Oftentimes, models are pre-trained at a smaller resolution and finetuned at a higher one Touvron et al. (2019). NaViT is much more flexible; it allows mixed-resolution training by sampling from a distribution of image sizes, while retaining each images’ original aspect ratio. This allows both higher throughput and exposure to large images, yielding substantial improved performance over equivalent ViTs (in terms of models size and training duration). Section 3.2 explores different sampling strategies, and variable resolution training for pre-training and finetuning.

#### 2.3 Efficiency of NaViT

Here we discuss some implications of Patch n’ Pack on the computational efficiency of NaViT.

Self attention cost. The O(n2) cost of attention is a natural concern when packing multiple images into longer sequences. Though many works aim to remove this quadratic scaling (Tay et al., 2022, 2020), we demonstrate here that as the transformer hidden dimension is scaled, the attention becomes an increasingly smaller proportion of the the overall cost, which encompasses the computation cost of the MLP

- as well. Figure 4 illustrates this trend, indicating a corresponding reduction in the overhead associated with packing examples. In addition to speed considerations, the memory cost of self-attention can pose a challenge for extremely long sequences. However, this challenge can be also addressed by employing memory-efficient methods (Rabe and Staats, 2021; Dao et al., 2022).

- 1 packed

- 2 packed

50%

Attentionoverhead

40%

4 packed 8 packed

30%

20%

10%

0%

1024 2048 4096 6144

Model dimension

Figure 4: Overhead from extra attention due to packing, assuming 256 tokens per image; it diminishes with model scale.

##### Packing, and sequence-level padding. The final sequence

lengths containing multiple examples must be fixed. We use a greedy packing approach discussed in Appendix A.3; there typically is no perfect combination of examples exactly adding up to the fixed length and padding tokens have to be used. One could for example dynamically choose the resolution or token dropping

rate of the final example in a sequence to exactly fit the remaining tokens; however, we find typically less 2% of tokens are padding tokens, and thus the simple approach is sufficient.

Padding examples and the contrastive loss. Per-token losses are straightforward to implement with packed sequences. However, many computer vision models are trained with example-level losses, typically applied to a pooled representation. First, this requires modifications to the typical pooling heads to account for packing. Second, multiple pooled representations must be extracted from each sequence. Fixed batch shapes requires an assumption that, from a batch of B sequences, we extract at most B ×Emax pooled representations (i.e. Emax examples per sequence). If a sequence contains more than Emax images, the extra images will be dropped, wasting computation of the model’s encoder. If a sequence has less than Emax examples, then the loss will process lots of fake padding representations.

The latter is an issue for contrastive learning, where loss computation scales in time and memory ∼ O(n2). To avoid this, we used the chunked contrastive loss (Mustafa et al., 2023), which circumvents the need to gather all data points for the softmax by performing computations on local device subsets and efficiently accumulating the necessary statistics for global softmax normalization. This enable high values of Emax (and thus efficient use of the model encoder), without being bottlenecked by the loss.

### 3 Experiments

The base architecture we use for NaViT follows vanilla ViT (Dosovitskiy et al., 2021), with the changes to enable packing, described in Section2.1. In addition, we include small ViT improvements from previous works: query-key normalization and the omission of biases (Dehghani et al., 2023), and attention pooling (Zhai et al., 2022).

We pre-train NaViT in two setups: classification training on JFT-4B (Zhai et al., 2022) and contrastive languageimage training (Radford et al., 2021) on WebLI (Chen et al., 2022c). Typically, for JFT, inception crop is applied pre-training (Kolesnikov et al., 2020; Dosovitskiy et al., 2021), and in both cases, images are resized to a square (distorting aspect ratio). Unless otherwise specified, all NaViT models are pre-trained without these operations, and preserve aspect ratio. NaViT is implemented in JAX (Bradbury et al., 2018) using the FLAX library (Heek et al., 2020) and built within Scenic (Dehghani et al., 2022).

Classification pretraining. We pre-train NaViT with supervised classification objective, using a sigmoid cross-entropy loss, following the setup of (Dehghani et al., 2023) on JFT-4B (Zhai et al., 2022). Visual representations are evaluated following the linear evaluation protocol used for ViT (Dosovitskiy et al., 2021), where 10 examples per class are used to train a linear classifier on top of frozen representations.

Contrastive pre-training. Alongside the image model, we train a text encoder with the same architectural modifications using the contrastive image-text loss (Radford et al., 2021; Zhang et al., 2022) (details in Appendix A.2). Packing also provides efficiency improvements on the text-tower, as text sequences do not need to be padded to a fixed lengths, which is the normal setup. The contrastive models are evaluated on zero-shot ImageNet classification and COCO image-text retrieval.

#### 3.1 Improved training efficiency and performance

Figure 1 illustrates the JFT pretraining performance of different NaViT models compared to compute-matched ViT baselines (Dehghani et al., 2021). The experimental setup details are provided in Appendix A.1. NaViT consistently surpasses ViT in performance while using the same computational budget across different compute and parameter scales; for example, the performance of the top-performing ViT can be matched by a

Best ImageNet 10shot

- 76%

- 77%

- 78%

- 79%

Top-1Accuracy

Variable res U[64,Rmax]

Fixed res = Rmax

128 256 384 512

Max train resolution Rmax

native res eval

80%

60%

40%

128 256 384 512

256 res eval

75%

70%

128 256 384 512

Rmax

128 res eval

60%

40%

20%

128 256 384 512

384 res eval

80%

60%

40%

128 256 384 512

Rmax

224 res eval

75%

70%

65%

128 256 384 512

512 res eval

80%

60%

40%

20%

128 256 384 512

Rmax

Figure 5: At fixed computational cost, sampling lower resolutions increases throughput, improves performance and enables better use of models at varied resolutions. NaViT-B/16 models trained with variable vs. fixed resolutions demonstrate the benefit of mixed resolution.

NaViT with four times less compute. Conversely, the computationally lightest NaViT in Figure 1 is five times more cost-effective than its equivalent ViT counterpart.

The NaViT models benefit from preserved aspect ratios and the ability to evaluate over many resolutions, but the chief contributor here is the significant increase in the number of training examples processed by NaViT within the allocated compute budget. This is achieved through the combination of sampling multiple variable-resolution examples and token dropping, leading to variable size images that are efficiently packed into a similar sequence length as the original model. We ablate these factors below.

#### 3.2 Benefits of variable resolution

Here, we deep-dive the benefits of mixed-resolution training. Since we preserve the native aspect ratio, when we refer to “resolution” for NaViT, we mean “effective resolution”. That is, images with the same area as a square image with a given resolution. For example, for NaViT a resolution of “128” has the same area of a square 128 x 128 image, but could be 64 x 256, or 170 x 96, etc., and thus has the same inference cost as regular ViT on 128 x 128 images.

Variable-resolution pre-training. Lower resolution images require fewer FLOPs to process and hence small resolutions (like 224) are used with fixed-resolution training. With fixed-resolution training, there is a trade-off between throughput and ability to process details and high-resolution images. With NaViT we can mix lower resolution images with large ones to get the best of both worlds.

- Figure 5 shows a comparison between two NaViT variants trained at several different resolutions. Here, all trained for the same number of FLOPs. (1) Native aspect ratio, but fixed resolution R = Rmax for different chosen values of Rmax. (2) Variable resolution, where the resolution is distributed as R ∼ U(64,Rmax). Variable resolution models outperform models trained at only that resolution. Even in the best case for fixed resolution, where the train and evaluation resolutions are identical, variable resolution matches or outperforms fixed.

Variable-resolution finetuning. Prior works increase resolution late in pre-training or during finetuning, producing higher quality but more expensive models (Dosovitskiy et al., 2021; Touvron et al., 2019). We finetune NaViT and ViT models at different fixed resolutions, and additionally NaViT at variable resolutions.

- Figure 6 shows the results of fine-tuning pretrained ViT and NaViT on ImageNet-1k dataset. Performance gains during pretraining transfer well at all resolutions, but two phenomena are particularly interesting: First, NaViT finetuned with variable resolutions ("NaViT 64:512") is as good as a NaViT finetuned at a single resolution (and much better than single-resolution ViT), removing the need to pick a single downstream finetuning resolution. Second, NaViT finetuned at low resolution (64) still obtains good performance when evaluated at higher resolutions (Figure 6, right), enabling cheaper adaptation.This ability to perform cheap adaptation of a flexible pre-trained model corroborates findings in (Beyer et al., 2023).

Eval at Rft

ImageNetTop-1Accuracy

85%

80%

75%

70%

NaViT (one per res)

65%

ViT (one per res)

60%

NaViT-64:512

55%

64 128 256 384 512

Fine-tune resolution Rft

Best eval across R [64,512]

NaViT (one per res)

ViT (one per res)

NaViT-64:512

64 128 256 384 512

Fine-tune resolution Rft

Figure 6: Variable-resolution finetuning, JFT B/16 models finetuned on ImageNet at various resolutions. Overall NaViT in all settings (blue, red), outperforms ViT (orange) Left: A single NaViT finetuned with variable resolutions (red) is as good as models tuned on only one resolution (blue). Right: Mixed-resolution pretraining performs well at high resolution when finetuning at low resolution (lefthand end of blue curve).

Resolution sampling strategies. Packing examples enables diverse resolution sampling strategies. We first consider whether to sample the target side length (average height/width, R), or the target area (i.e. sequence length ∝ R2). Sampling the side length from a uniform distribution biases towards lower sequence lengths, whereas sampling the area from a uniform distribution biases towards higher side lengths.

63.3%

normal normal

- 62.6%
- 63.4%

- 63.6%

63.1%

- 64.3%

distribution

= 0.5 normal =0.5 uniform

63.8%

63.4%

For each image, we sample u ∼ D, where D is a distribution with support [−1,1]. We rescale u linearly to [64,384] for sampling side-lengths or [642,3842] for sampling areas. We consider four distributions D: uniform u ∼ U(−1,1), truncated (to [−1,1]) standard Normal u ∼ Nt(0,1), and then two other Normals which bias towards lower resolution u ∼ Nt(−0.5,1) and higher resolution u ∼ Nt(0.5,1). The results are shown in Figure 7. Here, the best resolution resampling strategy consistently performs over the default resolution. It is consistently better to sample side-lengths (orange) directly as opposed to area (blue), and the best distribution is the truncated normal biasing towards lower values; both of these increase throughput by preferentially sampling smaller sequences.

55% 58% 60% 62% 65% 68%

ImageNet zero-shot Accuracy

Sample side length

Native res

Sample area

Figure 7: Sampling side lengths directly with a bias towards lower resolutions gives overall best performance at a fixed computational budget.

#### 3.3 Benefits of variable token dropping

- 75.8%

72.2%

- 76.3% 76.5%

Fixed Increasing

Dropschedule

Token dropping strategies. We experimented with continuously sampled token dropping rates, and with resolution-dependent token dropping rates; both are explained in Appendix A.4.

Decreasing (fast) Decreasing (slow)

- Fig. 9a compares variable drop rates sampled from a Beta distribution to a constant drop rate, demonstrating consistent improvements from the former. Fig. 9b shows the use of a resolution dependent token dropping rate for models trained with R ∼ U(64,384) and dropping rates scaled between [0.5 − δ,0.5 + δ] ∝ R, which further improves over the beta distribution.

70.0% 80.0%

Top-1 accuracy

Figure 8: Time-varying token dropping rates improves performance and are easily done with Patch n’ Pack.

Scheduled token dropping rates. Packing also enables easy variation the token dropping rate during training. By changing the token dropping rate we can better tune the trade-off between number of images seen and information used per image, to maximize the final accuracy while keeping the total training cost constant. We varied the token dropping rate as a function of the number of images seen (details of the schedule in Appendix A.5). Fig. 8 demonstrates that further improvements are possible by reducing the token dropping rate during JFT pretraining of NaViT-B/16.

| | | | | |
|---|---|---|---|---|
| |no token drop<br><br>fixed for every image<br><br>beta: per image<br><br>| | | |
| | | | | |
| | | | | |

Resolution-dependent

- 78.5%

- 79.0%

Top-1Accuracy

Top-1accuracy

80%

Figure 9: Continuous token dropping strategies enabled by sequence packing improves performance

78%

76%

78.0%

fixed beta =0.25 =0.4

0.25 0.50 0.75

Token drop strategy

Mean token drop rate

(b) Resolution dependent token dropping sampled ∈ 0.5 ± δ

(a) Constant vs. Beta-distributed token dropping rates.

#### 3.4 Positional embeddings

We evaluate our factorized embeddings introduced in Section 2.1, and their design choices. We are interested in both absolute performance, and extrapolation to resolutions outside the training regime. To test this, we train NaViT-B/16 models for 200k steps on JFT, with resolutions R ∼ U(160,352). We evaluate performance at a range of resolutions, without modification of the embedding variables. We compare to a ViT-B/16 trained at fixed resolution 256 for the same amount of images seen, evaluated at new resolutions using standard interpolation of positional embeddings.

- Fig. 10 contains the results. First, it is clear that the factorized approaches outperform both the baseline ViT and the Learned 2D embeddings from Pix2struct. The latter in particular struggles to generalize to higher resolution, likely because this requires an increasingly long tail of unseen (x,y) pairs. Factorized embeddings are best combined additively (as opposed to stacking or multiplying).

| |64.4% 71.4%<br><br>64.7% 71.3%<br><br>64.9% 71.5%<br><br>64.8% 71.5%<br><br><br>64.2% 71.3%<br><br>53.8% 69.3%<br><br>63.5% 71.3%<br><br>63.9% 70.8%<br>64.8% 71.4%<br><br><br>64.2% 71.5%<br><br><br>| | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

Learned 1D (ViT) Learned 2D (Pix2struct) Factorized (+) Factorized (stack) Factorized (×) Sinusoidal

100%

80%

Relativeaccuracy

60%

40%

Fourier Factorized (+)

ViT

Train resolutions

Learned 2D (pix2struct)

20%

Sinusoidal Fourier

Factorized (+)

Fourier (fractional)

0%

64 160 256 352 512 640

50% 55% 60% 65% 70% 75%

Evaluation resolution

ImageNet 10shot accuracy

(a)

(b)

- Figure 10: Factorized position embeddings improve generalization to new resolutions and aspect ratios. (a) Best (faded) and average accuracies (dark) across resolutions. (b) Accuracy normalized w.r.t. resolution 256.

3.5 Other aspects of NaViT’s performance

Out of distribution generalization. We directly evaluate JFT-pretrained NaViT on downstream datasets, employing a label-map (Wortsman et al., 2022) from JFT-4B to ImageNet (Deng et al., 2009) and robustnessvariants (ObjectNet (Barbu et al., 2019) and ImageNet-A (Hendrycks et al., 2021)). We compare the performance to a compute-matched ViT baseline.

- Figure 11 shows that NaViT compares favorably both on ImageNet as well as datasets variants that were specifically designed to test out of distribution performance. It is interesting to note that NaViT performs much better on ImageNet-A, but ViT catches up on ObjectNet, even though both these datasets contain images that have extreme aspect ratios. We believe this is due to the aspect-preserving center crop that we

ViT NaViT B/32 B/16 L/16

ImageNet

ImageNet-A

ObjectNet

| | |
|---|---|
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
|
| || |
|---|
<br><br>|
| | |
| | |
| | |

| || |
|---|
<br><br>| |
|---|
| |
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>|
|---|---|
| || |
|---|
<br><br>|
| | |

75%

| |
|---|

| |
|---|

| |
|---|

60%

| |
|---|

| |
|---|

70%

50%

| |
|---|

65%

40%

40%

60%

20%

55%

1012 1013

1012 1013

1012 1013

Training TPU chip hours

Training TPU chip hours

Training TPU chip hours

- Figure 11: Out of distribution evaluation of ViT and NaViT models that were matched for training compute. In addition to improved performance due to more images seen (see also Figure 1), NaViT performs much better on ImageNet-A that has many images with an extreme aspect ratio and important information outside the center crop (Appendix F). Same data as in Table 5.

apply to images for ImageNet-A and ObjectNet classification (same as in (Dehghani et al., 2023)), which is a useful prior for ObjectNet, but less so for ImageNet-A (see Appendix F for details). If no crop is applied to images and they’re instead simply resized to the image resolution expected by the model (i.e., square for ViT, and aspect preserving resize to the same number of tokens for NaViT), then the observed difference is much larger (see Figure 20 in appendix).

Calibration. In addition to the in-depth accuracy analysis, we have also quantfied the quality of the uncertainty computed by the model. In particular, for our ImageNet1K-finetuned models, we computed the expected calibration error (Nixon et al., 2019) of the top prediction, as we vary the number of patches we assign per examples. We find that the calibration error remains very stable in the interval (0.045,0.047) as we vary the number of patches per image in the range [128,1024], without any post-hoc recalibration. We provide further details in the appendix Appendix E.

Inference trade-offs. Given the flexibility of the model, there are several viable approaches how one can maximize the aggregate accuracy under a given compute budget, which we quantify in terms of the latency measured on a Cloud TPUv3 chip. In an online inference setting, choosing an inference strategy translates to an algorithm how to allocate a fixed number of tokens per example. As we show in Fig. 1, NaViT offers much better trade-offs than ViT and it shows strong diminishing returns, s.t., even relatively few patches provide highly competitive results. In Appendix C we further study a cascading approach, which both provides Pareto optimal models and gives more precise trade-off opportunities.

Fairness signal annotation. We investigate annotating images with fairness-related signals, such as those pertaining to gender and ethnicity. Prior research has shown that metrics, such as group calibration, are susceptible to labeling inaccuracy, particularly for underrepresented groups. In addition, this problem persists even when accounting for label noise during training (Adebayo et al., 2023). Thus, reducing the labeling error of fairness signals improves the reliability of bias mitigation and post-hoc auditing (Raji and Buolamwini, 2019). To explore whether NaViT can help in overcoming these challenges, we train annotators on FairFace (Kärkkäinen and Joo, 2019) and CelebA (Liu et al., 2015) datasets as linear probes (i.e. using frozen features produced by NaViT or ViT), before comparing their accuracy.

First, NaViT provides representations of higher quality that improve the accuracy of fairness signal annotation, even when dealing with square images. Original images are of size 448 × 448 in FairFace and 178 × 218 in CelebA, and we resize them to area 2242 while preserving aspect ratios in NaViT. Despite having the same

fairface / gender

fairface / gender

fairface / ethnicity

fairface / ethnicity

- 95
- 96
- 97

97

72

72

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

| |Model<br><br>ViT<br><br>NaViT| | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

| |Square Native<br><br>| | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

accuracy

accuracy

accuracy

accuracy

95

68

68

93

64

64

100 250 500 1000

100 250 500 1000

100 250 500 1000

100 250 500 1000

Pretraining Steps (K)

Pretraining Steps (K)

Pretraining Steps (K)

Pretraining Steps (K)

- Figure 12: Evaluating the accuracy of annotators trained on fairness-related signals using either NaViT-L/16 or ViT-L/16: Left: NaViT offers better representations that improve the accuracy of annotators. Right: Using native aspect ratios in NaViT results in a higher performance when compared to resizing images to squares.

sequence length, NaViT provides a higher prediction accuracy than ViT, as shown in Figure 12 (left). We verify statistical significance using the Wilcoxon signed-rank test test (Wilcoxon, 1992), which reveals that the improvement in NaViT is significant with p = 3 × 10−4.

Second, we apply inception-style cropping (Szegedy et al., 2016) with a fixed minimum area of 50%. This changes the aspect ratio but maintains native resolution. Similar to before, we resize all cropped images to area 224×224, either as square images or with native aspect ratios. Figure 12 (right) shows that native aspect ratios in NaViT improve accuracy, which is statistically significant at the 95% confidence level (p = 0.02). Appendix G contains the full set of figures for other attributes in CelebA and Fairface.

#### 3.6 Other downstream tasks

Semantic segmentation. We finetune NaViT to semantic segmentation on ADE20k dataset (Zhou et al., 2017), following the linear decoder protocol of Segmenter (Strudel et al., 2021). We use ViT-L/16 as baseline and compare its performance with that of NaViT-L/16. Both models are pre-trained on JFT-4B (Zhai et al., 2022) with comparable compute budget.

Evaluated on native resolutions

53

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
|ViT (square R2 )| | | | |
|max| | | | |
|NaViT (res Rmax2 )| | | | |
| | | | | |

ADE20kmIoU

50

47

We experiment with different maximum resolution Rmax at finetuning time: ViT takes in random square crops of resolution Rmax×Rmax while NaViT’s inputs are randomly resized (with preserved aspect ratio) so that the total number of pixels is Rmax2 . This way, we have the same finetuning cost for the ViT and NaViT models. Note that following common practice, in order not to alter the ground truth segmentation maps, both models are evaluated at the native resolution (Strudel et al., 2021; Caron et al., 2022; Zhou et al., 2017). This means that for ViT, the square predictions are resized from square back to native resolution. We observe in Figure 13 that NaViT outperforms ViT when transferred to semantic segmentation with the same maximum finetuning resolution Rmax. Note that NaViT at R384 beats ViT at R512 while being twice as fast (see Appendix C). An advantage of NaViT over ViT is that it benefits from flexibility of resolution during training and can ingest seamlessly square and non-square images.

44

256 384 512 800

Max Finetuning Resolution

Figure 13: NaViT transfers competitively to semantic segmentation. We transfer ViTL/16 and NaViT-L/16 on ADE20k with fine-tuning at different resolutions. ViT consumes square images while NaViT preserves the aspect ratio of the original images, while maintaning the total number of pixels the same as ViT.

Table 1: NaViT improvements carry over to object detection.

Object detection. Native-resolution training may be especially beneficial for fine-grained tasks such as object detection, which require image understanding at many different spatial scales. We use compute matched NaViT and ViT models

ViT-L/14 NaViT-L/14

ImageNet zeroshot 68.3% 72.9% LVIS AP 23.3% 28.3% LVIS AP rare 17.2% 24.3%

- as backbones for OWL-ViT-L/14 object detectors (Minderer

et al., 2022), following the OWL-ViT protocol for training and evaluation. Results are presented in Table 1. The NaViT-based detector performs significantly better on both common and unseen LVIS “rare” classes. These experiments used shorter pre-training than the original OWL-ViT and thus reach lower absolute performance; the relative difference nonetheless suggests that NaViT produces strong representations for fine-grained vision tasks.

Video Classification. Video processing with transformers is inherently challenging due to the heterogeneity of the underlying spatio-temporal signals. In practice, cumbersome protocols have to be established both for training and evaluation (e.g. spatial and temporal multi-cropping) (Arnab et al., 2021). NaViT alleviates some of these challenges by not only allowing training over different resolutions, but also over different temporal durations. We fine-tuned NaViT trained on JFT for Kinetics400 (Kay et al., 2017) classification by extracting three different spatio-temporal patches “tubelets”’. (Piergiovanni et al., 2022), extending the positional embedding to include the temporal dimension and initializing the embedding kernel using “central frame embedding” (Arnab et al., 2021). We posit that NaViT is an excellent starting point for multi-scale training due to the resolution diversity at training time. We observe that NaViT-L achieves competitive performance with ViViT-L (80.4%) in approximately 6x less epochs, without multi-crop evaluation. Note that the Kinetics400 dataset used here contains less data than prior works (Arnab et al., 2021).

### 4 Related work

Flexible Vision Transformers. FlexiViT (Beyer et al., 2023) developed a novel kernel resizing approach which enables variable "resolution" via models which support multiple patch sizes. This is viewed as unifying multiple distinct models, and they study the relationship with distillation and neural architecture search. Pix2struct (Lee et al., 2022) supported variable aspect ratios with a novel positional embedding schema, and demonstrated significant efficiency and performance improvements for non-natural imagery such as chart and document understanding.

Multiscale Vision Transformers. Using feature maps at multiple spatial scales is a common approach to localization tasks such as segmentation and detection (Ronneberger et al., 2015). Many works developed Vision Transformers which do the same (Fan et al., 2021; Li et al., 2022), though some dispute the necessity for simple localization tasks (Chen et al., 2022a). NaViT does not build hierarchical representations using multiple scales; we believe combining the unique flexibility of our approach with the benefits of this modelling family is a promising avenue.

Accelerating training with mixed resolutions. Image modelling works considering resolution largely focus on accelerating pretraining with a fixed, low resolution (Touvron et al., 2022; Liu et al., 2022). FixRes (Touvron et al., 2019) is a popular technique whereby resolution is increased in a final stage of pretraining, and has been used in many subsequent works (Radford et al., 2021; Chen et al., 2022c). PaLI (Chen et al., 2022c), for example, successively increases the resolution of the vision backbone during its generative training. This approach’s main downside is its irreversibility: compute cannot be scaled back by reducing resolution after tuning at the higher and more expensive resolution.

Multigrid training (Wu et al., 2020) is the most closely related work. The main idea is accelerate video modelling by processing large batches with "coarse" spatiotemporal resolution early in the training, followed by a "finer" resolution later. To this end, the authors apply a hierarchical grid sampling schedule coupled with appropriate learning rate scaling. In contrast, Patch n’ Pack enables effortless incorporation of mixed resolutions without complex schedules or training pipelines.

Token dropping for improved efficiency. Research initially explored random token dropping (Akbari et al., 2021; Li et al., 2023; He et al., 2022). Follow ups demonstrated benefits from considering structured (Chen

et al., 2022b) or “importance” based strategies (Bolya et al., 2022; Yin et al., 2022). Better strategies will likely further boost performance, and Patch n’ Pack sidesteps the fixed minibatch shape restriction which limited these works.

### 5 Conclusions and future work

We havedemonstrated thatPatch n’ Pack—the simple applicationof sequence packing to vision transformerssignificantly improves training efficiency. The resultant NaViT models can be applied to many resolutions

- at inference time, and cheaply adapted to new tasks. Patch n’ Pack enables a wide variety of research previously hindered by the need for fixed batch shapes, including adaptive computation and new algorithms for improving training and inference efficiency.

### Acknowledgment

We would like to thank Marvin Ritter, Carlos Riquelme, Justin Gilmer, and Joelle Barral for invaluable technical support, insightful discussions around the challenges, and finally meticulous feedback on the paper.

### References

Samira Abnar, Mostafa Dehghani, Behnam Neyshabur, and Hanie Sedghi. Exploring the limits of large scale pre-training. arXiv preprint arXiv:2110.02095, 2021.

Julius Adebayo, Melissa Hall, Bowen Yu, and Bobbie Chern. Quantifying and mitigating the impact of label errors on model disparity metrics. In ICLR, 2023.

Hassan Akbari, Liangzhe Yuan, Rui Qian, Wei-Hong Chuang, Shih-Fu Chang, Yin Cui, and Boqing Gong. Vatt: Transformers for multimodal self-supervised learning from raw video, audio and text. Advances in Neural Information Processing Systems, 34:24206–24221, 2021.

Anurag Arnab, Mostafa Dehghani, Georg Heigold, Chen Sun, Mario Lučić, and Cordelia Schmid. ViViT: A video vision transformer. In CVPR, 2021.

Andrei Barbu, David Mayo, Julian Alverio, William Luo, Christopher Wang, Dan Gutfreund, Josh Tenenbaum, and Boris Katz. ObjectNet: A large-scale bias-controlled dataset for pushing the limits of object recognition models. In NeurIPS, pages 9448–9458, 2019.

Lucas Beyer, Pavel Izmailov, Alexander Kolesnikov, Mathilde Caron, Simon Kornblith, Xiaohua Zhai, Matthias Minderer, Michael Tschannen, Ibrahim Alabdulmohsin, and Filip Pavetic. Flexivit: One model for all patch sizes. In CVPR, 2023.

Daniel Bolya, Cheng-Yang Fu, Xiaoliang Dai, Peizhao Zhang, Christoph Feichtenhofer, and Judy Hoffman. Token merging: Your vit but faster. CoRR, abs/2210.09461, 2022.

James Bradbury, Roy Frostig, Peter Hawkins, Matthew James Johnson, Chris Leary, Dougal Maclaurin, George Necula, Adam Paszke, Jake VanderPlas, Skye Wanderman-Milne, and Qiao Zhang. JAX: composable transformations of Python+NumPy programs, 2018. URL http://github.com/google/jax.

Mathilde Caron, Neil Houlsby, and Cordelia Schmid. Location-aware self-supervised transformers for semantic segmentation. arXiv preprint arXiv:2212.02400, 2022.

Wuyang Chen, Xianzhi Du, Fan Yang, Lucas Beyer, Xiaohua Zhai, Tsung-Yi Lin, Huizhong Chen, Jing Li, Xiaodan Song, Zhangyang Wang, and Denny Zhou. A simple single-scale vision transformer for object detection and instance segmentation. In Shai Avidan, Gabriel J. Brostow, Moustapha Cissé, Giovanni Maria Farinella, and Tal Hassner, editors, Computer Vision - ECCV 2022, 2022a.

Wuyang Chen, Wei Huang, Xianzhi Du, Xiaodan Song, Zhangyang Wang, and Denny Zhou. Auto-scaling vision transformers without training. In The Tenth International Conference on Learning Representations, ICLR, 2022b.

Xi Chen, Xiao Wang, Soravit Changpinyo, A. J. Piergiovanni, Piotr Padlewski, Daniel Salz, Sebastian Goodman, Adam Grycner, Basil Mustafa, Lucas Beyer, Alexander Kolesnikov, Joan Puigcerver, Nan Ding, Keran Rong, Hassan Akbari, Gaurav Mishra, Linting Xue, Ashish Thapliyal, James Bradbury, Weicheng Kuo, Mojtaba Seyedhosseini, Chao Jia, Burcu Karagol Ayan, Carlos Riquelme, Andreas Steiner, Anelia Angelova, Xiaohua Zhai, Neil Houlsby, and Radu Soricut. PaLI: A jointly-scaled multilingual language-image model. arXiv preprint arXiv:2209.06794, 2022c.

Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. Flashattention: Fast and memoryefficient exact attention with io-awareness. In NeurIPS, 2022. URL http://papers.nips.cc/paper_files/ paper/2022/hash/67d57c32e20fd0a7a302cb81d36e40d5-Abstract-Conference.html.

Mostafa Dehghani, Anurag Arnab, Lucas Beyer, Ashish Vaswani, and Yi Tay. The efficiency misnomer. arXiv preprint arXiv:2110.12894, 2021.

Mostafa Dehghani, Alexey Gritsenko, Anurag Arnab, Matthias Minderer, and Yi Tay. Scenic: A jax library for computer vision research and beyond. In CVPR, 2022.

Mostafa Dehghani, Josip Djolonga, Basil Mustafa, Piotr Padlewski, Jonathan Heek, Justin Gilmer, Andreas Steiner, Mathilde Caron, Robert Geirhos, Ibrahim Alabdulmohsin, Rodolphe Jenatton, Lucas Beyer, Michael Tschannen, Anurag Arnab, Xiao Wang, Carlos Riquelme, Matthias Minderer, Joan Puigcerver, Utku Evci, Manoj Kumar, Sjoerd van Steenkiste, Gamaleldin F. Elsayed, Aravindh Mahendran, Fisher Yu, Avital Oliver, Fantine Huot, Jasmijn Bastings, Mark Patrick Collier, Alexey Gritsenko, Vighnesh Birodkar, Cristina Vasconcelos, Yi Tay, Thomas Mensink, Alexander Kolesnikov, Filip Pavetić, Dustin Tran, Thomas Kipf, Mario Lučić, Xiaohua Zhai, Daniel Keysers, Jeremiah Harmsen, and Neil Houlsby. Scaling vision transformers to 22 billion parameters. In International Conference on Machine Learning, 2023.

Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. ImageNet: A large-scale hierarchical image database. In CVPR, pages 248–255, 2009.

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In ICLR, 2021.

Haoqi Fan, Bo Xiong, Karttikeya Mangalam, Yanghao Li, Zhicheng Yan, Jitendra Malik, and Christoph Feichtenhofer. Multiscale vision transformers. In 2021 IEEE/CVF International Conference on Computer Vision, ICCV, 2021.

Robert Geirhos, Kantharaju Narayanappa, Benjamin Mitzkus, Tizian Thieringer, Matthias Bethge, Felix A Wichmann, and Wieland Brendel. Partial success in closing the gap between human and machine vision. In NeurIPS, pages 23885–23899, 2021.

Agrim Gupta, Piotr Dollár, and Ross B. Girshick. LVIS: A dataset for large vocabulary instance segmentation. In IEEE Conference on Computer Vision and Pattern Recognition, CVPR, 2019.

Foad Hamidi, Morgan Klaus Scheuerman, and Stacy M Branham. Gender recognition or gender reductionism? the social implications of embedded gender recognition systems. In Proceedings of the 2018 chi conference on human factors in computing systems, pages 1–13, 2018.

Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross B. Girshick. Masked autoencoders are scalable vision learners. In CVPR, pages 15979–15988. IEEE, 2022.

Jonathan Heek, Anselm Levskaya, Avital Oliver, Marvin Ritter, Bertrand Rondepierre, Andreas Steiner, and Marc van Zee. Flax: A neural network library and ecosystem for JAX, 2020. URL http://github.com/ google/flax.

Dan Hendrycks, Kevin Zhao, Steven Basart, Jacob Steinhardt, and Dawn Song. Natural adversarial examples. In CVPR, pages 15262–15271, 2021.

Will Kay, Joao Carreira, Karen Simonyan, Brian Zhang, Chloe Hillier, Sudheendra Vijayanarasimhan, Fabio Viola, Tim Green, Trevor Back, Paul Natsev, et al. The kinetics human action video dataset. arXiv preprint arXiv:1705.06950, 2017.

Os Keyes. The misgendering machines: Trans/hci implications of automatic gender recognition. Proceedings of the ACM on human-computer interaction, 2(CSCW):1–22, 2018.

Alexander Kolesnikov, Lucas Beyer, Xiaohua Zhai, Joan Puigcerver, Jessica Yung, Sylvain Gelly, and Neil Houlsby. Big Transfer (BiT): General visual representation learning. In ECCV, pages 491–507, 2020.

Mario Michael Krell, Matej Kosec, Sergio P Perez, and Andrew William Fitzgibbon. Efficient sequence packing

without cross-contamination: Accelerating large language models without impacting performance, 2021. Taku Kudo and John Richardson. SentencePiece: A simple and language independent subword tokenizer

and detokenizer for neural text processing. In EMNLP, pages 66–71, November 2018. Kimmo Kärkkäinen and Jungseock Joo. Fairface: Face attribute dataset for balanced race, gender, and age, 2019.

Kenton Lee, Mandar Joshi, Iulia Turc, Hexiang Hu, Fangyu Liu, Julian Eisenschlos, Urvashi Khandelwal, Peter Shaw, Ming-Wei Chang, and Kristina Toutanova. Pix2struct: Screenshot parsing as pretraining for visual language understanding, 2022.

Yanghao Li, Chao-Yuan Wu, Haoqi Fan, Karttikeya Mangalam, Bo Xiong, Jitendra Malik, and Christoph Feichtenhofer. Mvitv2: Improved multiscale vision transformers for classification and detection. In CVPR, 2022.

Yanghao Li, Haoqi Fan, Ronghang Hu, Christoph Feichtenhofer, and Kaiming He. Scaling language-image pre-training via masking, 2023.

Ze Liu, Han Hu, Yutong Lin, Zhuliang Yao, Zhenda Xie, Yixuan Wei, Jia Ning, Yue Cao, Zheng Zhang, Li Dong, et al. Swin Transformer V2: Scaling Up Capacity and Resolution. CVPR, 2022.

Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang. Deep learning face attributes in the wild. In ICCV, 2015.

Matthias Minderer, Alexey Gritsenko, Austin Stone, Maxim Neumann, Dirk Weissenborn, Alexey Dosovitskiy, Aravindh Mahendran, Anurag Arnab, Mostafa Dehghani, Zhuoran Shen, et al. Simple open-vocabulary object detection with vision transformers. arXiv preprint arXiv:2205.06230, 2022.

Basil Mustafa, Josip Djolonga, and Mostafa Dehghani. On efficient losses for distributed contrastive learning. arXiv preprint, 2023.

Jeremy Nixon, Michael W Dusenberry, Linchuan Zhang, Ghassen Jerfel, and Dustin Tran. Measuring calibration in deep learning. In CVPR Workshops, 2019.

AJ Piergiovanni, Weicheng Kuo, and Anelia Angelova. Rethinking video vits: Sparse video tubes for joint image and video learning, 2022.

Markus N Rabe and Charles Staats. Self-attention does not need O(n2) memory. arXiv preprint arXiv:2112.05682, 2021.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, pages 8748–8763, 2021.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. arXiv preprint arXiv:1910.10683, 2019.

Inioluwa Deborah Raji and Joy Buolamwini. Actionable auditing: Investigating the impact of publicly naming biased performance results of commercial ai products. In Proceedings of the 2019 AAAI/ACM Conference on AI, Ethics, and Society, pages 429–435, 2019.

Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In Nassir Navab, Joachim Hornegger, William M. Wells III, and Alejandro F. Frangi, editors, Medical Image Computing and Computer-Assisted Intervention - MICCAI, 2015.

Robin Strudel, Ricardo Garcia, Ivan Laptev, and Cordelia Schmid. Segmenter: Transformer for semantic segmentation. In ICCV, 2021.

Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jon Shlens, and Zbigniew Wojna. Rethinking the inception architecture for computer vision. In CVPR, 2016.

Matthew Tancik, Pratul P. Srinivasan, Ben Mildenhall, Sara Fridovich-Keil, Nithin Raghavan, Utkarsh Singhal, Ravi Ramamoorthi, Jonathan T. Barron, and Ren Ng. Fourier features let networks learn high frequency functions in low dimensional domains. In Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, 2020.

Yi Tay, Mostafa Dehghani, Samira Abnar, Yikang Shen, Dara Bahri, Philip Pham, Jinfeng Rao, Liu Yang, Sebastian Ruder, and Donald Metzler. Long range arena: A benchmark for efficient transformers. arXiv preprint arXiv:2011.04006, 2020.

Yi Tay, Mostafa Dehghani, Dara Bahri, and Donald Metzler. Efficient transformers: A survey. ACM Computing Surveys, 55(6):1–28, 2022.

Hugo Touvron, Andrea Vedaldi, Matthijs Douze, and Hervé Jégou. Fixing the train-test resolution discrepancy. In Hanna M. Wallach, Hugo Larochelle, Alina Beygelzimer, Florence d’Alché-Buc, Emily B. Fox, and Roman Garnett, editors, Advances in Neural Information Processing Systems 32: Annual Conference on Neural Information Processing Systems 2019, NeurIPS 2019, December 8-14, 2019, Vancouver, BC, Canada, 2019.

Hugo Touvron, Matthieu Cord, and Hervé Jégou. DeiT III: Revenge of the ViT. In ECCV, 2022. Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser,

and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017. Frank Wilcoxon. Individual comparisons by ranking methods. Springer, 1992.

Mitchell Wortsman, Gabriel Ilharco, Jong Wook Kim, Mike Li, Simon Kornblith, Rebecca Roelofs, Raphael Gontijo Lopes, Hannaneh Hajishirzi, Ali Farhadi, Hongseok Namkoong, and Ludwig Schmidt. Robust finetuning of zero-shot models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2022, New Orleans, LA, USA, June 18-24, 2022, pages 7949–7961. IEEE, 2022. doi: 10.1109/CVPR52688.2022.00780. URL https://doi.org/10.1109/CVPR52688.2022.00780.

Chao-Yuan Wu, Ross B. Girshick, Kaiming He, Christoph Feichtenhofer, and Philipp Krähenbühl. A multigrid method for efficiently training video models. In CVPR, pages 150–159. Computer Vision Foundation / IEEE, 2020.

Hongxu Yin, Arash Vahdat, Jose M. Alvarez, Arun Mallya, Jan Kautz, and Pavlo Molchanov. A-vit: Adaptive tokens for efficient vision transformer. In CVPR, 2022.

Xiaohua Zhai, Alexander Kolesnikov, Neil Houlsby, and Lucas Beyer. Scaling vision transformers. In CVPR, pages 12104–12113, 2022.

Yuhao Zhang, Hang Jiang, Yasuhide Miura, Christopher D. Manning, and Curtis P. Langlotz. Contrastive learning of medical visual representations from paired images and text. In Proceedings of the Machine Learning for Healthcare Conference, MLHC, 2022.

Bolei Zhou, Hang Zhao, Xavier Puig, Sanja Fidler, Adela Barriuso, and Antonio Torralba. Scene parsing through ade20k dataset. In CVPR, 2017.

### A Training details

#### A.1 Classification pretraining

The experiments and ablations in the paper are with ViT-B/32, ViT-B/16, and ViT-L/16. We use a reciprocal square-root learning rate schedule, with linear warmup and cooldown, with a maximum value of 8e − 4, and phases. We follow (Zhai et al., 2022; Abnar et al., 2021) and use a higher weight decay of 3.0 on the head compared to the body’s weight decay of 0.03 during upstream training to improve transfer to downstream tasks. For our experiments, we evaluated both NaViT and ViT models using configurations B/32, B/16, and L/16. Each ViT model was trained with varying compute budgets, with cooling down at different stages of training. We trained a corresponding NaViT model for each ViT size and computational budget, allowing us to perform “compute-matched” comparisons (Dehghani et al., 2021).

Table 2 presents the pretraining specifications for both ViT and NaViT models. During the pretraining phase, ViT models were trained using images of size 224×224. In contrast, NaViT models uniformly sampled a value, denoted as r, between 64 and 256 and resized the image to have a total of r2 pixels while preserving the aspect ratio. Although training NaViT models on native resolutions is possible, we empirically discovered that sampling a resolution provides greater control over maximizing the number of examples observed during pretaining within a fixed computational budget while maintaining performance across different resolutions in downstream tasks. Additionally, by controlling the resolution, we can ensure efficient packing by tuning the sequence length and limit padding to less than 2%.

#### A.2 Contrastive pretraining

We use the 32000 token T5 Raffel et al. (2019) sentencepiece Kudo and Richardson (2018) tokenizer. By default, text sequences are truncated to a maximum length of 24. No token dropping is used for text. Models are trained under the same optimization regime as the classification models, but with a learning rate of 3×10−3. Weight decay of 1×10−6 is applied consistently to all kernels in the model (no change for projection heads). By default, image side-lengths are sampled ∼ U(64,Rmax), and no other image augmentations are applied.

#### A.3 Packing algorithm

Packing of examples into sequences is done alongside batching. A simple greedy approach is used which adds examples to the first sequence with enough remaining space. Once no more examples can fit, sequences are filled with padding tokens, yielding the fixed sequence lengths needed for batched operations. Such simple packing algorithm can lead to a significant padding, depending on the distribution of length of inputs. There are several methods to address such limitations, like bin packing (Krell et al., 2021), which allows minimizing the padding. Here, in NaViT, since controlling the resolutions we sample, we can ensure efficient packing by tuning the sequence length and limit padding to less than 2%.

#### A.4 Sampling token dropping rates

Sampling with a beta distribution We use a parameterisation based on the mean dµ and standard deviation σ. We aim to sample dropout rate d ∈ [0.0,dmax], with some mean dµ.

Accordingly, we sample u ∈ [0,1] ∼ B(α,β) and set drop rate d = u × dmax. α and β are set such that the mean of u is uµ = ddµ

. The maximum supported variance for a beta distribution of mean uµ is uµ(1 − uµ); we pick by default a variance σ2 = 0.3uµ(1 − uµ), which we found to work well in practice. The resultant distributions of token dropouts for different settings of dµ and dmax are shown in Figure 14a.

max

Table 2: Pre-training details of ViT and NaViT with supervised classification.

Images Per Seq.

Batch Size

Name TPU Hours

Train Steps

Cooldown Steps

Sequence Length

Training Images

- 1.4 × 1011 1.0 × 105 1.0 × 104 49 1.0 ≈4.0 × 103 4.0 × 108 3.5 × 1011 2.5 × 105 5.0 × 104 49 1.0 ≈4.0 × 103 1.0 × 109 7.1 × 1011 5.0 × 105 1.0 × 105 49 1.0 ≈4.0 × 103 2.0 × 109
- 1.4 × 1012 1.0 × 106 1.0 × 105 49 1.0 ≈4.0 × 103 4.0 × 109

ViT-B/32

- 4.7 × 1011 1.0 × 105 1.0 × 104 196 1.0 ≈4.0 × 103 4.0 × 108

- 1.1 × 1012 2.5 × 105 5.0 × 104 196 1.0 ≈4.0 × 103 1.0 × 109
- 2.3 × 1012 5.0 × 105 1.0 × 105 196 1.0 ≈4.0 × 103 2.0 × 109

- 4.7 × 1012 1.0 × 106 1.0 × 105 196 1.0 ≈4.0 × 103 4.0 × 109

ViT-B/16

- 9.8 × 1011 1.0 × 105 1.0 × 104 196 1.0 ≈4.0 × 103 4.0 × 108 2.4 × 1012 2.5 × 105 5.0 × 104 196 1.0 ≈4.0 × 103 1.0 × 109 4.9 × 1012 5.0 × 105 1.0 × 105 196 1.0 ≈4.0 × 103 2.0 × 109
- 9.8 × 1012 1.0 × 106 1.0 × 105 196 1.0 ≈4.0 × 103 4.0 × 109

ViT-L/16

- 1.4 × 1011 9.8 × 104 1.0 × 104 64 5.41 ≈2.2 × 104 2.1 × 109 3.5 × 1011 2.4 × 105 5.0 × 104 64 5.41 ≈2.2 × 104 5.3 × 109 7.1 × 1011 4.8 × 105 1.0 × 105 64 5.41 ≈2.2 × 104 1.0 × 1010
- 1.4 × 1012 9.7 × 105 1.0 × 105 64 5.41 ≈2.2 × 104 2.1 × 1010

NaViT-B/32

- 4.7 × 1011 9.3 × 104 1.0 × 104 256 4.87 ≈1.9 × 104 1.8 × 109

- 1.1 × 1012 2.3 × 105 5.0 × 104 256 4.88 ≈1.9 × 104 4.6 × 109
- 2.3 × 1012 4.6 × 105 1.0 × 105 256 4.88 ≈1.9 × 104 9.2 × 109

- 4.7 × 1012 9.2 × 105 1.0 × 105 256 4.88 ≈1.9 × 104 1.8 × 1010

NaViT-B/16

- 9.8 × 1011 9.7 × 104 1.0 × 104 256 4.88 ≈1.9 × 104 1.9 × 109 2.4 × 1012 2.4 × 105 5.0 × 104 256 4.87 ≈1.9 × 104 4.8 × 109 4.9 × 1012 4.8 × 105 1.0 × 105 256 4.87 ≈1.9 × 104 9.6 × 109
- 9.8 × 1012 9.6 × 105 1.0 × 105 256 4.88 ≈1.9 × 104 1.9 × 1010

NaViT-L/16

Sampling resolution-dependent dropping rates Given input data with sequence lengths ranging from smin to smax, we sample dropout rate d from a truncated normal distribution d ∼ Ntrunc(µ,0.02), where samples more than two standard deviations away from µ are rejected.

The mean of this distribution µ is set according to the minimum and maximum token dropping rates dmin and dmax, and simply scales linearly with the sequence length s (such that s = smin has µ = dmin and s = smax has µ = dmax.

Figure 14b shows example distributions of sampled drop rates given inputs with resolution R ∼ U(64,384), and different values of dmin and dmax.

#### A.5 Scheduling token dropping rates

We experiment with a token dropping schedule which varies with total number of images seen. In particular, the rate applied for the n-th processed image during training is given by:

n − µ τ

, (1)

ρ(n;ρmin,ρmax,µ,τ) = ρmin + (ρmax − ρmin) · σ

|Max drop|rate = 90%|
|---|---|
| |=0.85p=0.1%<01<br><br>|
| | |
| | |
| | |
| | |
| | |

|Ma|x drop rate = 40%| | |
|---|---|---|---|
| |=0.35p=1.3%<01<br><br>| | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

Max drop rate = 70%

25%

- 0%

- 1%

- 2%

- 3%

- 4%

- 5%

=0.10p=60.3%<01

=0.25p=11.5%<01

=0.10p=66.9%<01

=0.25p=27.3%<01

=0.10p=70.4%<01

=0.25p=33.5%<01

=0.35p=1.3%<01

=0.50p=2.3%<01

=0.65p=0.1%<01

=0.50p=5.9%<01

=0.70p=0.9%<01

=0.85p=0.1%<01

d [0.1,0.9]

20%

d [0.25,0.75]

d [0.4,0.6]

15%

Percent

Percent

10%

5%

0%

25% 50% 75% 100%

25% 50% 75% 100%

25% 50% 75% 100%

25% 50% 75% 100%

Token drop rate

Token drop rate

Token drop rate

Token drop rate

(a) Beta-sampled token drop rates parameterised by the mean µ and the max drop rate dmax

(b) Sampled resolution-dependent token drop rates

1e9

ImageNet10shotacc

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

0.75

Totalimagesseen

Averagedroprate

Fixed =12

- 0

- 1

- 2

75%

=10N , =N2

0.50

= 10N , =N2 = N5, =23N

70%

0.25

65%

0 200 k 400 k

0 200 k 400 k

0 200 k 400 k

Total training steps

Total training steps

Total training steps

- Figure 15: Decreasing the token dropping rate along training improves the ImageNet 10shot accuracy using the same pre-training resources. N is the total number of training examples seen with a fixed token dropping rate of ρ = 12.

where σ represents the sigmoid function; ρmin, ρmax control the minimum and maximum dropping rate applied; and µ and τ control the shape of the schedule. We experimented with both increasing (τ > 0) and decreasing (τ < 0) schedules. In all cases we set ρmin = 0.2 and ρmax = 0.8. Figure 15 shows that, by decreasing the dropping rate throughout training, one can improve the final accuracy, at fixed training cost. Conversely, increasing the token dropping rate harms performance.

### B Model information

#### B.1 Positional embeddings

Extending ViTs to variable input sizes necessitates rethinking positional embeddings added to every token after embedding. We considered several variants of positional embeddings, and evaluated them based on (1) the best performance model using them achieve within training distribution of input sizes; and based on (2) how well these models perform when evaluated on image sizes outside of the training distribution. Results and discussion of these experiments can be found in Section 3.4.

Broadly, we considered positional embeddings that varied along three axes: (1) whether they were learned, parametric or fixed; (2) whether they were absolute or fractional; and (3) whether they are factorized.

Absolute and fractional coordinates A natural way of indexing token within an image is to select a priori a maximum possible image side length (shared for width and height) maxLen, and to assign to token integer coordinates (x,y) based on their original location within the image. Embedding coordinates defined in this way allow models to consume images with resolutions up to R = P · maxLen. However, when learned

###### Absolute coordinates Fractional coordinates (non aspect ratio preserving) Fractional coordinates (aspect ratio preserving)

| | |
|---|---|
|(0.0, 0.0)| |
| | |
| |[Figure 1]|

| | |
|---|---|
|(0.0, 0.0)| |
| | |
|[Figure 2]<br><br>| |

| | |
|---|---|
|(0.0, 0.0)| |
| | |
| |[Figure 3]|

|(0.0, 0.0)| | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | |[Figure 4]| | |
| | | | | | |

|(0, 0)| | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
|[Figure 5]<br><br>| | | | | |
| | | | | | |

|(0.0, 0.0)| | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
|[Figure 6]<br><br>| | | | | |
| | | | | | |

0.0 1.0 0.0

0 6 0

0.0 1.0 0.0

0.0 0.75 0.0

0 2 0

0.0 1.0 0.0

(0.75, 1.0) (1.0, .71)

(2, 3) (6, 4)

(1.0, 1.0) (1.0, 1.0)

.71

4

1.0

1.0

3

1.0

Image credit: Matthew Henry burst.shopify.com/photos/dog-staying-warm

- Figure 16: We use two different views of the same image, of resolutions 96×128 and 224×160, and demonstrate different coordinate systems when using patch size 32.

absolute coordinate embeddings are considered, extreme values of x and y must also be observed during training, which necessitates training on images with varied aspect ratios and limits models generalisation.

To alleviate the necessity of observing extreme aspect ratios and image size during learning of positional embeddings, we also consider fractional coordinates, which are normalized to the actual size of the input image and are obtained by dividing the absolute coordinates x and y above by the number number of columns and rows respectively, i.e. the corresponding side length. Doing this allows the model to observe extreme token coordinates during training, which intuitively should help with generalization to higher resolutions. However, this is accomplished at the cost of obfuscating the input images aspect ratio.

Factorized embeddings We further consider whether coordinates x and y should be embedded independently or jointly. In case of independent embedding, the two coordinates x and y are embedded independently, and their embeddings are combined via addition or by stacking. For joint embeddings and embedding for each position (x,y) is obtained directly.

Learned, parametric and fixed positional embeddings Finally, we also explored the relative benefits of fixed, learned and parametric embeddings. For fixed embeddings we followed Vaswani et al. (2017) and used sinusoidal positional embeddings, and learned embeddings were implemented as in Dosovitskiy et al. (2021).

For parametric positional embeddings we followed Tancik et al. (2020) and used Fourier embeddings. Specifically, coordinates (x,y) were mapped using a single linear layer before applying sin and cos activations to them, and stacking the results to obtained the positional embeddings.

Experiments Because not all combinations of the above embedding choices are equally promising or natural, we experimented only with subset of them shown in Table 3 and Figure 10a.

### C Inference strategies

We performed various experiments to measure model quality for given runtime cost. The runtime can be tuned by changing the number of processed patches, or by using choosing different size of the model.

We firstly looked at how model quality changes in respect to decreasing area of the image compared to native resolution, presented in Figure 17a. We observed that on ImageNet Deng et al. (2009) model retains most of the quality down to 40% of the image size. After that, the quality drastically decreases. On the other hand, increasing the size of the image have a diminishing return in quality. This can be directly compared with random token dropping as an alternative to resizing, which showed to be very ineffective way to decrease number of patches during inference - Figure 17b.

Table 3: Classification of positional embedding experiments from Figure 10a.

Name Coordinates Type Factorized

- Learned 1D (ViT) Absolute Learned No, position in flatted token sequence
- Learned 2D (Pix2struct) Absolute Learned No Factorized abs. (+) Absolute Learned Yes, sum Factorized abs. (stack) Absolute Learned Yes, stack Factorized abs. (×) Absolute Learned Yes, product Fourier abs. Absolute Parametric No Sinusoidal abs. Absolute Fixed Yes, stack Factorized frac. (+) Fractional Learned Yes, sum Fourier frac. Fractional Parametric No Sinusoidal frac. Fractional Fixed Yes, stack

In sinusoidal and factorised embeddings experiments with fractional coordinates fractional coordinate embeddings were obtained from absolute coordinate embeddings via bilinear interpolation.

Please note that this highly depends on the native resolution of the images in the dataset - e.g. dataset with twice as big images than ImageNet can probably be safely resized to 20% of area.

- 85

- 86

- 87

- 88

Imagenetaccuracy

NaViT-L/16 NaViT-B/16

0.25 0.50 0.75 1.00 1.25 1.50

Image area

(a)

88.5

88.0

Imagenetaccuracy

87.5

87.0

resize

86.5

drop tokens

0.25 0.50 0.75 1.00 1.25 1.50

Image area

(b)

88.6

Imagenetaccuracy

88.4

88.2

88.0

NaViT-L

87.8

NaViT-L(no upscale)

87.6

128 256 384 512 640 768 896 1024

sequence length

(c)

- Figure 17: (a) The effect of resizing the image. (b) Dropping random tokens is ineffective way to decrease number of patches compared to resizing the image. Data from NaViT-L/16. (c) Given number of patches as compute budget, it is beneficial to upscale the image.

A better way to quantify the performance is by giving a constant compute budget corresponding to number of patches. Figure 18a shows that resizing the image (while preserving aspect ratio) to 256 tokens retains most of the quality (within 0.3%). This corresponds to 256x256 area (given patch size of 16). At 128 tokens (181x181 area) the quality difference reaches 1% and drops significantly after that.

Here we also resized the image past its native resolution in case it already fit the given sequence length budget. We observed that it is beneficial to resize the image to the given sequence length past the native resolution to keep monotonic increase in quality, which is showed on Figure 17c.

Figure 18b presents the runtime of NaViT-L/16 and NaViT-B/16 for different sequence lengths. We can see that NaViT-L/16 at sequence length 128 is as fast as NaViT-B/16 with sequence length 512, while having almost 1% difference in quality.

### D Cascades

Another strategy to be more compute efficient would be to assign more tokens (and thus FLOPs) to the examples deemed hard by the model. This is in particular interesting for bulk inference workloads, where one can amortize over large datasets and where only the total inference time matters.

To evaluate the feasibility of this approach, we consider two sequence lengths n1 and n2 with respective

- 84

- 85

- 86

- 87

- 88

1%

6x reduction

Imagenetaccuracy

1%

5x reduction

navit resize mine resize

64128 256 384 512 640 768 896 1024

sequence length

(a)

512

640 768 896 1024

256 384

Imagenetaccuracy

88.0

87.5

128

87.0

640

8961024 NaViT-L/16 NaViT-B/16

512

768

384

86.5

256

5 10 15 20 25

average runtime in ms per image

(b)

- Figure 18: (a) Quality on ImageNet in respect to number of patches (sequence length). (b) Runtime of models compared to the accuracy on ImageNet.

0 5 10 15 20 25

Average inference time in milliseconds

85.5

86.0

86.5

87.0

87.5

88.0

88.5

ImageNet1K-Accuracy

| |
|---|

| |
|---|

| |
|---|

128

256

1024

Cascade (ViT-B)

ViT-B/16

Cascade (ViT-L) ViT-L/16

| |
|---|

Pareto frontier

- Figure 19: Performance of a model cascade versus the average inference time. The labels at the select points denote the number of tokens at that scale.

inference times t1 and t2. Then, we (i) send all examples though the model with n1 tokens, and send only the α ∈ (0,1)-fraction deemed hardest (those with the smallest maximum probability) to the model with n2 tokens. To have an input almost exactly fit into n1 or n2 tokens we perform and aspect ratio preserving resize. Hence, the total amortized inference time per-example is ti + αt2, while the accuracy obtained by combining the accuracy of the first model on the 1 − α most-confident fraction of the data, and the performance of the more expensive model on the remaining data. By considering several pairs of models and varying α we obtain the plot in Figure 19. As we can see this strategy is indeed useful and provides not only the best performing models at given compute budgets, but because α is a real parameter one can obtain very fine-grained trade-offs.

### E Calibration

To evaluate behaviour of the predicted uncertainties with scale, we compute the calibration error of a -B sized ImageNet-finetuned model (the -L model performs similarly). Note that these models were trained with sigmoid loss, i.e., the 1000 labels were predicted independently without enforcing that the probabilities should sum up to 1. As we varied the sequence of tokens per example between 128 and 1024, we obtained very stable calibration errors (top-1, using ℓ1 and 30 buckets, i.e., the settings from Nixon et al. (2019)), which we present in Table 4.

Table 4: Expected calibration error on ImageNet-1K with varying sequence lengths.

Sequence Length 128 256 384 512 640 768 1024 Calibration Error 0.047 0.046 0.048 0.047 0.047 0.046 0.045

### F Out of distribution evaluation

For ViT, we apply the “Crop” strategy from (Dehghani et al., 2023), namely an aspect-preserving crop of the central 75% of the image for ObjectNet and ImageNet-A, and square resize followed by a 87.5% central crop for the other datasets. We also apply a simple “Resize” strategy that does not crop the images. For NaViT, both the “Crop” and the “Resize” strategy do an aspect preserving resize of the target images.

Table 5: Detailed results of out evaluation of pretrained models with a label-map (see Section 3.5). Same data as in Figure 11 and Figure 20.

ImageNet ImageNet-A ObjectNet

ViT NaViT ViT NaViT ViT NaViT Compute

custom B/32 1.4 × 1011 54.4 63.5 14.7 32.7 32.9 41.2 1.4 × 1012 65.5 68.2 30.7 42.5 44.2 45.7 3.6 × 1011 60.2 65.6 22.0 37.0 38.0 43.5 7.2 × 1011 63.7 67.2 26.6 40.1 41.7 45.1

B/16 1.2 × 1012 66.4 68.5 37.3 52.3 47.2 48.4 2.4 × 1012 68.7 70.1 43.5 55.1 50.5 49.8

- 4.8 × 1011 61.7 66.0 27.0 44.2 41.3 45.9
- 4.8 × 1012 70.3 71.2 48.8 57.0 52.8 50.7

L/16 2.5 × 1012 70.7 73.6 51.5 65.5 53.3 55.0 4.9 × 1012 73.0 74.6 57.6 67.9 56.2 57.1

- 9.9 × 1011 66.4 71.1 39.2 58.6 47.6 52.1
- 9.9 × 1012 73.9 75.1 60.4 68.9 57.7 57.9

resize B/32 1.4 × 1011 51.7 64.0 12.6 26.7 15.9 31.6 1.4 × 1012 63.6 68.6 22.8 35.0 25.3 36.1 3.6 × 1011 57.9 66.2 17.2 30.1 20.0 33.8 7.2 × 1011 61.6 67.4 20.7 32.6 23.5 34.6

B/16 1.2 × 1012 65.4 69.2 27.4 43.9 28.7 38.7 2.4 × 1012 67.7 71.0 32.5 48.6 32.0 40.5 4.8 × 1011 60.4 66.5 20.4 36.8 23.6 35.3 4.8 × 1012 69.5 72.5 36.2 51.5 34.2 42.1

L/16 2.5 × 1012 70.0 73.9 38.5 59.9 35.3 45.6 4.9 × 1012 72.3 75.3 44.3 64.1 38.8 48.2

- 9.9 × 1011 65.1 71.5 28.2 51.6 29.3 41.5
- 9.9 × 1012 73.2 76.0 47.3 65.5 39.8 48.8

### G Fairness Signal Annotation

In Figure 21, we demonstrate that using native image resolution improves the performance of fairness signal annotation. Prior research has shown that metrics, such as group calibration, are vulnerable to labeling errors, particularly for underrepresented groups. Moreover, this problem persists even when accounting for label noise during training (Adebayo et al., 2023). Thus, reducing the labeling error of fairness signals has the potential of improving the reliability of bias mitigation and post-hoc auditing (Raji and Buolamwini, 2019).

ViT NaViT B/32 B/16 L/16

ImageNet

ImageNet-A

ObjectNet

50

| || |
|---|
<br><br>| |
|---|
|
|---|---|
| || |
|---|
|
| || |
|---|
<br><br>| |
|---|
<br><br>|
| || |
|---|
<br><br>|
| || |
|---|
<br><br>|
| | |

75

| |
|---|

| |
|---|

| |
|---|

60

45

| |
|---|

| |
|---|

| |
|---|

| |
|---|

70

| |
|---|

40

50

| |
|---|

35

65

40

30

| |
|---|

60

30

25

20

55

20

15

10

1012 1013

1012 1013

1012 1013

Training TPU chip hours

Training TPU chip hours

Training TPU chip hours

- Figure 20: Same evaluation as in Figure 11, but without any special preprocessing of the images before the evaluation. Employing a simple resize (square for ViT, aspect preserving for NaViT) results in much better performance on datasets that have images with an extreme aspect ratio. Same data as in table Table 5.

Nevertheless, we emphasize that while NaViT improves the annotation accuracy in these tasks, care must

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |

100 250 500 1000

Pretraining Steps (K)

- 98
- 99

accuracy

celeba / gender

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

100 250 500 1000

Pretraining Steps (K)

93

95

97

accuracy

fairface / gender

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

100 250 500 1000

Pretraining Steps (K)

54

57

60

accuracy

fairface / age

| |Model<br><br>ViT<br><br>NaViT| | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

100 250 500 1000

Pretraining Steps (K)

64

68

72

accuracy

fairface / ethnicity

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |

100 250 500 1000

Pretraining Steps (K)

- 98
- 99

accuracy

celeba / gender

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

100 250 500 1000

Pretraining Steps (K)

- 95
- 96
- 97

accuracy

fairface / gender

100 250 500 1000

Pretraining Steps (K)

55

57

59

accuracy

fairface / age

| |Square Native<br><br>| | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

100 250 500 1000

Pretraining Steps (K)

64

68

72

accuracy

fairface / ethnicity

- Figure 21: Summary of results of evaluating the accuracy of annotators trained on fairness-related signals using either NaViT-L/16 or ViT-L/16. top: NaViT offers better representations that improve the accuracy of annotators. bottom: Using native aspect ratios in NaViT results in a higher performance when compared to resizing images to squares.

be taken in such situations since classifiers can be inaccurate and lead to a broad categorization of people that misidentifies real identities. We encourage readers to delve into the comprehensive work outlining such potential risks, e.g. (Hamidi et al., 2018; Keyes, 2018), for further insight. In assessing the technical capabilities of NaViT, our intent is not to promote or encourage their application in inappropriate contexts. Rather, our objective is only to illuminate these technical findings for scenarios where they may be considered beneficial, such as when measuring the level of diversity in a dataset or auditing/mitigating biases in predictive models. We strongly advocate for responsible AI use, maintaining that the benefits of technological advancements should not overshadow the importance of user safety and privacy. AI tools, including ours, should always be deployed judiciously, with a keen awareness of potential risks and a commitment to avoiding harm.

### H Evaluation on model-vs-human OOD datasets on different resolutions

Just like NaViT, human visual perception works across flexible aspect ratios and resolutions (just imagine how strange the world would look like if we could only see it through a 224 × 224 pixel window!). We investigate how the ability to cope with variable resolutions affects performance on “model-vs-human”, a benchmark of 17 challenging datasets (Geirhos et al., 2021).1 For this purpose, we replicate the setup from Figure 6, but instead of evaluating ImageNet accuracy, we evaluate OOD accuracy on the model-vs-human benchmark.

colour

false-colour

power-equalisation

rotation

Accuracy(%)

Accuracy(%)

Accuracy(%)

Accuracy(%)

80

95

80

95

NaViT (one per res)

60

90

ViT (one per res)

90

60

NaViT-64:512

85

40

85

64 128 224 384 512 Fine-tune resolution Rft

64 128 224 384 512 Fine-tune resolution Rft

64 128 224 384 512 Fine-tune resolution Rft

64 128 224 384 512 Fine-tune resolution Rft

uniform-noise

contrast

low-pass

high-pass

60

Accuracy(%)

Accuracy(%)

Accuracy(%)

Accuracy(%)

50

60

80

40

50

40

20

60

64 128 224 384 512 Fine-tune resolution Rft

64 128 224 384 512 Fine-tune resolution Rft

64 128 224 384 512 Fine-tune resolution Rft

64 128 224 384 512 Fine-tune resolution Rft

eidolonI

phase-scrambling

eidolonIII

eidolonII

65

80

62.5

Accuracy(%)

Accuracy(%)

Accuracy(%)

Accuracy(%)

65

60.0

60

60

57.5

60

55.0

64 128 224 384 512 Fine-tune resolution Rft

64 128 224 384 512 Fine-tune resolution Rft

64 128 224 384 512 Fine-tune resolution Rft

64 128 224 384 512 Fine-tune resolution Rft

edge

sketch

stylized

silhouette

60

90

70

Accuracy(%)

Accuracy(%)

Accuracy(%)

Accuracy(%)

60

80

50

65

40

70

60

40

60

55

20

64 128 224 384 512 Fine-tune resolution Rft

64 128 224 384 512 Fine-tune resolution Rft

64 128 224 384 512 Fine-tune resolution Rft

64 128 224 384 512 Fine-tune resolution Rft

- Figure 22: OOD accuracy on “model-vs-human” datasets across different fine-tuning resolutions. A single NaViT model trained on varying resolutions (red) performs roughly on par as fine-tuning one NaViT model per test resolution (blue). The ViT baseline (orange) is mostly worse than NaViT models for lower resolutions and mostly better for higher resolutions.

This corresponds to testing JFT B/16 models finetuned on ImageNet at various resolutions. The test dataset has a fixed 224×224 square resolution; thus we resize the test images to fit each model’s fine-tuning resolution. Note that using square images has been standard practice designed for convolutional networks, but NaViT models no longer require square input, thus existing benchmarks are not tailored to those new possibilities. For datasets with multiple difficulty levels (such as different levels of blur), we average performance across levels while excluding levels that are too easy (not OOD) or too hard (human performance is close to chance), which follows the approach of Geirhos et al. (2021) as explained in their “Appendix G Benchmark scores”.

To ensure a fair comparison, we use models that are compute-matched for pretraining, and during fine-tuning, compute and data are identical for all models. The results of our comparison are shown in Figure 22. Overall, a single NaViT model trained on varying resolutions (red) performs roughly on par with fine-tuning one NaViT model per test resolution (blue).

The ViT baseline (orange) is mostly worse than NaViT models for lower resolutions and mostly better for

1For the purpose of our comparison, we exclude the “cue-conflict” dataset from the OOD evaluation, since there is no objective ground truth class in the case of images with a texture-shape cue conflict.

higher resolutions. It may be worth noting that ViT models have a bit of an advantage in this comparison since they are fine-tuned on square images, whereas NaViT models are fine-tuned on flexible resolution images (preserving the image’s aspect ratio) that have the same number of pixels, while not necessarily being square.

