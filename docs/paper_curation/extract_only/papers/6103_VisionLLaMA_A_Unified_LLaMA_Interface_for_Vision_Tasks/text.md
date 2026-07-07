# arXiv:2403.00522v2[cs.CV]8Jul2024

## VisionLLaMA: A Unified LLaMA Backbone for Vision Tasks

##### Xiangxiang Chu1, Jianlin Su2, Bo Zhang1, Chunhua Shen3

- 1 Meituan Inc.
- 2 Moonshot AI

3 Zhejiang University, China

[Figure 1]

Fig. 1: Generated images by DiT-LLaMA-XL of resolution (256, 256) with CFG.

Abstract. We all know that large language models are built on top of a transformer-based architecture to process textual inputs. For example, the LLaMA family of models stands out among many open-source implementations. Can the same transformer be used to process 2D images? In this paper, we answer this question by unveiling a LLaMA-like vision transformer in plain and pyramid forms, termed VisionLLaMA, which is tailored for this purpose. VisionLLaMA is a unified and generic modeling framework for solving most vision tasks. We extensively evaluate its effectiveness using typical pre-training paradigms in a good portion of downstream tasks of image perception and especially image generation. In many cases, VisionLLaMA has exhibited substantial gains over the previous state-of-the-art vision transformers. It is our hope that researchers in computer vision can apply VisionLLaMA presented here to solve various specific image generation and perception tasks.

Code is at: https://github.com/Meituan-AutoML/VisionLLaMA

Keywords: LLaMA · Diffusion Model · Vision Transformer

### 1 Introduction

Large language models have aroused great interest in the research community. One of the most influential and representative work is LLaMA [64, 65]. Many recent works have converged to this architecture and solutions for various applications are built upon the open-sourced models. Besides, we have witnessed the blooming of multimodal models, where many methods also heavily rely on LLaMA for text processing. Meanwhile, many endeavors [23,37,71] have been devoted to accelerating the inference speed and/or the memory cost of LLaMA. In a word, LLaMA is now the de facto architecture.

Observing its success, a straightforward and interesting question is whether the LLaMA architecture can be another victory in the vision modality. If the answer is affirmative, then both vision and language models can use the same unified architecture and enjoy various deployment techniques designed for LLaMA on the fly. Unfortunately, it is non-trivial to answer this question because there are some distinct differences between these two modalities. Firstly, it is common sense that text sequences are organized into one dimension, while vision requires two or more. Secondly, numerous vision tasks rely on pyramid backbones to perform better, while the LLaMA is a plain encoder. Thirdly, it is necessary to handle input images and videos with different resolutions. Our paper aims to resolve these difficulties and bridge the architectural gap between different modalities. Our main contributions are summarized as follows:

- 1. We propose VisionLLaMA, a vision transformer architecture similar to LLaMA to reduce the architectural differences between language and vision.
- 2. We investigate means to adapt VisionLLaMA to tackle common vision tasks, including image comprehension and creation (Figure 1). We examine two well-known vision architecture schemes (plain and pyramid) and assess their performance under supervised and self-supervised learning scenarios. Additionally, we introduce AS2DRoPE (i.e. auto-scaled 2D RoPE), which expands rotated positional encoding from 1D to 2D and utilizes interpolation scaling to accommodate arbitrary resolutions.
- 3. Without bells and whistles, VisionLLaMA significantly outperforms the widespread and carefully fine-tuned vision transformer by clear margins across many representative tasks such as image generation, classification, semantic segmentation, and object detection. Extensive experiments indicate that VisionLLaMA demonstrates faster convergence speed and better performance than existing vision transformers.

### 2 Related Work

Vision Transformer. ViT [22] successfully applied Transformer [66] from natural language processing to the vision world and many more efficient and powerful follow-up works are induced, like DeiT [63], Swin [42], PVT [68], and Twins [12].

The pre-training paradigm has been shifted from supervised learning on largescale categorically labeled datasets like ImageNet [19] to unsupervised learning [25]. DiT [49] adopts a transformer that operates on latent patches for diffusion models [28,58], outperforming the commonly used U-Net backbone [53].

Large Language/Multi-modal Models Proprietary models like GPT4 [47] have been taking the lead in the LLM competition, though their technical details are hidden from the public. In contrast, the community has blossomed to release a myriad of open-source counterparts. For instance, BLOOM [2] and LLaMA [64] catch up with the performance of the closed model GPT-3 [7]. Later in copious detail, LLaMA-2 [65] describes a pack of architectural tweakings including prenormalization called RMSNorm [78], the activation function SwiGLU [57], rotary positional embeddings RoPE [60], as well as a dedicated training pipeline, which comprises self-supervised pre-training and supervised fine-tuning enhanced by Reinforcement Learning with Human Feedback (RLHF). Many vision language models [35, 39, 40, 70, 81] are built on LLaMA and show impressive results on the visual dialog, reasoning, perception, and so on. The LLaMA architecture has also been applied in resource-limited multimodal scenarios such as mobile phones [10,11] recently and shows potential applications.

Positional Encoding for Transformers. Transformer [66] originally comes with 2D absolute position embeddings in sinusoidal forms. In contrast, the relative ones as in [56] pay attention to the relations of input tokens and can handle variable lengths of sequences. Rotary positional embeddings [60] are introduced to encode both absolute and relative positional information, which is proven to be effective in large language models [64]. Conditional positional embeddings [13] are proposed to add positional information for vision transformers according to the input image, with the benefit of boosted performance and generalizability to arbitrary input resolutions. As for LLMs, the models are usually pre-trained with a given fixed context length [64, 65, 75] and then fine-tuned to a larger context length to support long context inference. [8] extends the context length of LLaMA by simple positional interpolations. Base frequency adjustment of RoPE is also studied by [74] to enable long-context continued training. NTKAware scaled RoPE allows LLaMA to have an extended context size without fine-tuning and minimal perplexity degradation [54].

### 3 Method

#### 3.1 Plain Transformer

Our plain VisionLLaMA follows the pipeline of ViT [22] and we retain the architecture design of LLaMA as closely as possible. For an image of H × W, it’s firstly transformed and flattened into N = HP×2W non-overlapped patches X ∈ RN×C. Then a class token is prepended at the beginning of the sequence and the whole sequence is processed by L VisionLLaMA blocks. Unlike [22], we do not add positional encodings to the input sequence since our basic block readily contains positional encoding. Specifically, the basic block differs from the standard ViT block by two components: self-attention with positional encoding

SwiGLU

SwiGLU

SwiGLU

LayerNorm

LayerNorm

LayerNorm

MHSA

LSA

GSA

AS2DRoPE LayerNorm

AS2DRoPE LayerNorm

AS2DRoPE LayerNorm

Embedded Patches

Embedded Patches

(a) VisionLLaMA Block

(b) VisionLLaMA Pyramid Block

Fig. 2: VisionLLaMA block (a) in plain Transformer and (b) in pyramid Transformer.

(RoPE) [60] and SwiGLU activation [57]. We still utilize LayerNorm [3] instead of RMSNorm [78] since we find the former behave better through the classification experiment (see Table 7b). The basic block is illustrated in Figure 2 (a). It should be noted that directly applying 1D RoPE in vision tasks cannot well generalize to other resolutions, which is different from the training resolution. Therefore, we extend it to the 2D form. It can be formally written as,

zlij = MHSA AS2DRoPE LayerNorm zlij−1 + zlij−1, zlij = SwiGLU LayerNorm zlij + zlij, i ∈ {1,2,....,m},j ∈ {1,2,....,n}.

(1)

where zijl means the output of the l block at position (i,j).

#### 3.2 Pyramid Transformer

It is straightforward to apply VisionLLaMA to window-based transformers that utilize additive relative position encoding, such as Swin [42]. In this paper, we choose a stronger baseline Twins [12] to explore how to build a powerful pyramid transformer under strictly controlled settings. The original architecture of Twins exploits a conditional position encoding and interleaved local-global information exchange in the form of local and global attention. These components can be found in various transformers, which means it is not difficult to apply VisionLLaMA in other pyramid transformer variants by following our method. Note that our target is not to invent a novel pyramid vision transformer, but to show how we adapt the basic design of VisionLLaMA based on the existing

ones. Therefore, we simply conform to the smallest modifications to the architecture and hyperparameters. Following the name convention of [12], the two consecutive blocks can be written as,

zˆlij = LSA AS2DRoPE LayerNorm zlij−1 + zlij−1, zlij = SwiGLU LayerNorm z ˆlij + zˆlij, zˆl+1 = GSA AS2DRoPE LayerNorm zl + zl, zl+1 = SwiGLU LayerNorm z ˆl+1 + zˆl+1, i ∈ {1,2,....,m},j ∈ {1,2,....,n}.

(2)

where LSA is the local self-attention operation within a group and GSA is the global sub-sampled attention by interacting with the representative keys from each sub-window zˆij ∈ Rk1×k2×C and m × n is the sub-window shape.

We remove the conditional position encoding in our pyramid VisionLLaMA since AS2DRoPE already contains positional information. Besides, we also remove the class tokens and use GAP (global average pooling) before the classification head as [12,13]. The basic block in this setting is illustrated in Figure 2(b).

#### 3.3 Training or Inference Beyond the Sequence Length

From 1D RoPE to 2D. Handling different input resolutions is a common requirement in vision tasks. Convolutional neural networks use the sliding window mechanism to deal with the variable length. In contrast, most vision transformers apply local window operations or interpolations. For instance, DeiT [63] adopts bicubic interpolations when trained on different resolutions. CPVT [13] uses convolution-based position encoding. Here we evaluate the performance of

- 1D RoPE [60]. Specifically, our pyramid VisionLLaMA based on Twins-SVT-S with 1D RoPE achieves 81.5% top-1 accuracy on an input of 224×224. However, the performance severely degrades to zero when evaluated on 448×448. Therefore, we extend the 1D RoPE to 2D. As for the multi-head self-attention, the
- 2D RoPE is shared across different heads. Specifically, given a token xi,j ∈ Rd, we obtain its position-encoded token xPEi,j = Ri,jxi,j, and the diagonal matrix Ri,j ∈ Rd×d can be written as, 



cos(iθ0) − sin (iθ0) 0 0 . . . 0 0 0 sin(iθ0) cos (iθ0) 0 0 . . . 0 0 0

0 0 cos(jθ0) − sin (jθ0) . . . 0 0 0 0 0 sin(jθ0) cos (jθ0) . . . 0 0 0

0 0 0 . . . cos(iθd−4) − sin (iθd−4) 0 0

 

 

sin(iθd−4) cos (iθd−4) 0 0 0 0 0 . . . 0 0 cos(jθd−4) − sin (jθd−4) 0 0 0 . . . 0 0 cos(jθd−4) cos (jθd−4)

where θm = 10000−m/d and m ∈ {0,4,8,...,d−4}. Note that R is an orthogonal matrix. We make minor modifications to the frequency selection [60] and make two axes share the same frequency. It is easy to verify that

R⊤i

1−i2,j1−j2. (3)

1,j1Ri

2,j2 = Ri

Positional interpolation helps 2D RoPE to better generalize. Inspired by [8], which uses interpolation to extend the context window of LLaMA, involving higher resolution is analogous to extending the 2D context window of VisionLLaMA. Unlike the language task [8] with an enlarged fixed context length, vision tasks like object detection usually deal with different sampled resolutions at different iterations. We train our small model using an input resolution of 224×224 and evaluate the performance on the larger resolutions without re-training, which guides us to apply good strategies of interpolation or extrapolation. Consequently, we apply auto-scaled interpolation (so-called AS2DRoPE) based on an ‘anchor resolution’. Without loss of generality, we assume handling the square image of H ×H and an anchor resolution B ×B during the training, we calculate

R′i,jxi,j = Ri·B/H,j·B/H, (4) which can be efficiently implemented and does not introduce an extra cost. Note if the training resolution is kept unchanged, AS2DRoPE degenerates as a 2D RoPE.

As for the GSA under the pyramid setting, we require special treatments since we need to add positional information to the summarized keys. These subsampled keys are generated by abstraction on the feature maps. Without loss of generality, we use a convolution with a kernel size of k × k and stride of k. The coordinate of the generated key can be formulated as the average of the sampled features. We show a simple example in Figure 5 (appendix.).

### 4 Experiments

We evaluate the effectiveness of VisionLLaMA on image generation, classification, segmentation, and detection. Unless otherwise specified, all models are trained on 8 NVIDIA Tesla A100 GPUs.

#### 4.1 Image Generation

Image generation based on the DiT framework. We apply VisionLLaMA under the DiT framework [49], which is a representative work of image generation using vision transformers and DDPM [28]. Specifically, we replace the original vision transformer of DiT with VisionLLaMA while keeping other components unchanged. This controlled experiment manifests the generality of VisionLLaMA on the image generation task. Moreover, we do not change the original hyperparameters, although it may be sub-optimal to achieve the best performance. We also use the pre-trained VAE [33] (the ft-EMA VAE model) from SD [52], which has a down-sample factor of 8. For classifier-free guidance, we use a coefficient of 1.5. The training resolution of the image is 256 × 256. As suggested by [49], we choose the strongest adaLN-Zero version as our implementation. We also use flash attention [18] and mixed precisions to speed up the training. Note that FID is known to be sensitive to small implementation details [48]. To make accurate calculations and fair comparisons, we use the TensorFlow tool from [21] as [49].

We choose 250 sample steps of DDPM as [49] and show the result in Table 1. As a common practice, FID is regarded as a primary metric. We also report other secondary metrics such as sFID [46], Precision/Recall [34], and Inception Score [55]. Most experiments are controlled on 400k training steps. VisionLLaMA significantly outperforms DiT across various model sizes. We also extend the training steps of XL models to 2352k steps to evaluate whether our models have the faster convergence advantage or still behave better under the setting of longer training epochs. DiT-LLaMA-XL/2 has 0.83 lower FID [27] than DiTXL/2, indicating that VisionLLaMA not only has better computing efficiency but higher performance than DiT. We show some generated samples in Figure 1 using our XL model.

Table 1: Image generation comparisons using DiT [49].

Model CFG Flops Params Steps lr FID↓ sFID↓ Precision↑ Recall↑ IS↑ (G) (M) (K)

DiT-B/4 N 5.56 130 400 0.0001 68.38 12.66 36.07 54.71 20.27 DiT-LLaMA-B/4 N 5.56 130 400 0.0001 63.17 12.63 38.27 56.75 22.47 DiT-B/4 Y 5.56 130 400 0.0001 45.38 9.97 46.89 53.66 34.27 DiT-LLaMA-B/4 Y 5.56 130 400 0.0001 39.51 9.82 50.46 54.75 40.17

DiT-L/4 N 19.70 458 400 0.0001 44.37 8.97 48.16 61.53 32.25 DiT-LLaMA-L/4 N 19.70 458 400 0.0001 40.32 9.04 49.87 61.61 36.56 DiT-L/4 Y 19.70 458 400 0.0001 22.51 7.08 62.67 55.27 66.58 DiT-LLaMA-L/4 Y 19.70 458 400 0.0001 18.64 7.01 65.40 54.35 78.52

DiT-XL/4 N 29.05 675 400 0.0001 43.01 - - - DiT-LLaMA-XL/4 N 29.05 675 400 0.0001 35.99 8.48 52.31 61.65 41.18 DiT-XL/4 Y 29.05 675 400 0.0001 22.52 7.09 62.68 55.27 66.58 DiT-LLaMA-XL/4 Y 29.05 675 400 0.0001 18.69 7.02 65.67 55.57 78.32

DiT-XL/2 N 118.64 675 2352 0.0001 10.67 - - - DiT-LLaMA-XL/2 N 118.64 675 2352 0.0001 9.84 6.47 67.45 66.71 117.72 DiT-LLaMA-XL/2 Y 118.64 675 2352 0.0001 2.42 4.51 83.03 56.82 265.39

#### 4.2 Classification on ImageNet

Supervised Training In this section, we focus on supervised training on the ImageNet-1K dataset [19] to make fair comparisons. We exclude other datasets or distillation tricks. All the models are trained using the ImageNet-1K training set, and we report the accuracy of the validation set in Table 2.

Plain Vision Transformer Comparison. DeiT3 [63] is the state-of-the-art plain vision transformer, which proposes special data augmentations and performs extensive hyperparameter search to boost the performance of DeiT [62]. During the reproduction of DeiT3, we observe that it is sensitive to hyperparameters and prone to overfitting. Replacing the class token with GAP (global average pooling) [13] leads to a 0.7% top-1 accuracy drop for the DeiT3-Large model after 800 epochs of training. Therefore, we use the class token instead of GAP in the plain transformer and report the result in Table 2, where VisionLLaMA achieves a top-1 accuracy comparable to DeiT3. The detailed hyperparameter

Table 2: Comparisons on ImageNet-1K supervised classification. All the models are trained using the ImageNet-1K dataset. †: retrained using the official code. 160I 800E+224I 20E means two-stage training, the model is firstly trained for 800 epochs using 160×160, then trained for 20 epochs with higher image resolution 224×224.

Model Param Setting Top-1

(M) (%) DeiT-Small [62] 22 224I 300E 79.9 CPVT-Small-GAP [13] 23 224I 300E 81.5

DeiT3-Small [63] 22 224I 800E 81.4 VisionLLaMA-S [63] 22 224I 800E 81.6

Swin-T [42] 29 224I 300E 81.3 Twins-SVT-S [12] 24 224I 300E 81.7

Pyramid VisionLLaMA-S 24 224I 300E 81.6

Swin-S [42] 50 224I 300E 83.0 Twins-SVT-B [12] 56 224I 300E 83.2

Pyramid VisionLLaMA-B 56 224I 300E 83.2 DeiT3-Base [63] 86 192I 800E + 224I 20E 83.8 VisionLLaMA-B 86 192I 800E + 224I 20E 83.6

Swin-B [42] 88 224I 300E 83.3 Twins-SVT-L [13] 99 224I 300E 83.7

Pyramid VisionLLaMA-L 99 224I 300E 83.6 DeiT3-Large† 310 160I 800E+224I 20E 84.5 VisionLLaMA-L 310 160I 800E+224I 20E 84.6

is listed in the appendix. Note that the accuracy on a single resolution does not provide comprehensive comparisons, we also evaluate the performance across different image resolutions as [13] and report the result in Table 11(supp.). As for DeiT3, we use the bicubic interpolation for the learnable positional encoding. Although these two models have comparable performance at the resolution of 224×224, the gap is enlarged when the resolution is increased, which means our method generalizes better across different resolutions, which is a vital function for many downstream tasks such as object detection.

Pyramid Vision Transformer. We use the same architecture as Twins-SVT [12] and the detailed configuration is listed in Table 16 (supp.). We remove the conditional position encoding since VisionLLaMA already contains one kind of rotary position encoding. Therefore, VisionLLaMA is a convolution-free architecture. We do not tune the hyper-parameters and directly follow the setting provided in [12]. Although it’s suboptimal, it can still achieve competitive performance. As [12,13], we do not use the class token and apply GAP. The result is shown in Table 2 and our method achieves comparable performance as Twins across various levels of models and outperforms Swin [42] consistently. We further compare the pyramid transformers using popular downstream tasks, which are shown in the later sections.

Self-Supervised Training To make fair comparisons, we limit the training data to ImageNet-1K. We also exclude any component that utilizes CLIP [50], DALLE [51], or distillation, which can be orthogonally combined to further boost the performance. Our implementation is based on the MMPretrain frame-

work [14]. We utilize the MAE framework and replace the encoder using VisionLLaMA while keeping other components unchanged. This minor modified setting forms a controlled experiment to evaluate the role of our approaches. Moreover, we use the same hyperparameter as [25], which is suboptimal to our method. Fortunately, this simple setting still achieves a significant performance boost over the strong baseline.

Full fine-tuning. In such a setting, the model is first initialized using the pretrained weights and then trained for extra epochs with totally trainable parameters. Trained by 800 epochs on the ImageNet, VisionLLaMA-Base achieves 84.0% top-1 accuracy, which exceeds ViT-Base by 0.8%. Note that our method uses a mask ratio of 0.75 as [25], whose training speed is about 3 times faster than SimMIM [73]. We also increased the training epochs to 1600 to verify whether VisionLLaMA keeps the advantage given sufficient training resources. VisionLLaMA-Base achieves new state-of-art result among MAE variants, 84.3% top-1 accuracy, which outperforms ViT-Base by 0.9%. This result is even higher than MaskFeat [69] where new training objectives are proposed. Regarding full fine-tuning having a risk of performance saturation [41,67], our boost is significant. Next, we resort to the linear probing metric to provide extra evaluations, which is considered a more reliable evaluation for representative learning by a recent work [9].

Table 3: Comparison with MIM SSL methods. †: reproduced in MMPretrain.

Models Pretrain Epochs SFT Acc(%) LP Acc(%) ViT-Base-MAE† [25] 800 83.2 65.1

SemMAE [36] 800 83.4 65.0 SimMIM [73] 800 83.8 56.7

MFF-MAE [41] 800 83.6 67.0 VisionLLaMA-Base-MAE 800 84.0 69.7

ViT-Base-MAE [25] 1600 83.4 67.0 MaskFeat [69] 1600 84.0 62.3 VisionLLaMA-Base-MAE 1600 84.3 71.7

ViT-Large-MAE† [25] 800 85.4 73.7 VisionLLaMA-Large-MAE 800 85.5 77.3

Linear probing. In this setting, the model is initialized by the pre-trained weights from the SSL stage. Then, the whole backbone is frozen except for the classifier head during the training. The result is shown in Table 3. With a training cost of 800 epochs, VisionLLaMA-Base outperforms ViT-Base-MAE by

- 4.6%. It also exceeds ViT-Base-MAE, which is trained for 1600 epochs. When VisionLLaMA is trained for 1600 epochs, VisionLLaMA-Base achieves 71.7% top-1 accuracy. We also scale up to have VisionLLaMA-Large, where our method exceeds ViT-Large by 3.6%.

###### Table 4: Segmentation results on ADE20k.

(a) Performance comparisons with different backbones on ADE20K validation dataset. All backbones are pre-trained on ImageNet-1K with labels.

Models Param mIoU (ss)

(M) (%) Swin-S [42] 81.3 47.6 Twins-SVT-B [12] 88.5 47.7 Pyramid VisionLLaMA-B 88.5 49.1 Swin-B [42] 121 48.1 Twins-SVT-L [12] 133 48.8 Pyramid VisionLLaMA-L 133 50.0

(b) Performance comparisons with different SSL trained backbones on ADE20K validation dataset. All models are pre-trained on ImageNet-1K without labels.†: reproduced using [15].

Models Epochs mIoU(ss) ViT-B† 800 46.2

SemMAE [36] 800 46.3 MFF-MAE [41] 800 47.9

VisionLLaMA-B 800 49.0 ViT-B 1600 48.1 MaskFeat [69] 1600 48.3 VisionLLaMA-B 1600 50.2

#### 4.3 Semantic Segmentation on ADE20K

Supervised Training Following [12,42], we evaluate our method using semantic segmentation on the ADE20K [80] dataset. To make fair comparisons, we limit the baselines to only using ImageNet-1K in the pre-training stage. Specifically, we make use of the UperNet [72] framework and replace the backbone with pyramid VisionLLaMA. The detailed setting of the hyperparameter is shown in Section D.7(supp). We report the result in Table 4a. Our method outperforms both Swin and Twins by more than 1.2% mIoU.

Self-Supervised Training We use UperNet [72] to perform semantic segmentation on ADE20K. We carefully control the experiment and replace the ViT backbone with VisionLLaMA while keeping other components and hyperparameters unchanged. The detailed hyperparameters are provided in Section D.6(supp.). The result is given in Table 4b. As for the 800 epoch pretraining groups, VisionLLaMA-B significantly boosts ViT-Base by 2.8% mIoU. It also outperforms some other modifications such as introducing extra training objectives or features [41,69] by clear margins. Moreover, those approaches introduce extra overhead for the training process and slow down the training speed. We emphasize that the training speed of a method is becoming more and more important in the age of large models. In contrast, VisionLLaMA only involves the replacement of the base model and has the same fast training speed as [25]. In principle, our method can be seamlessly combined with these modifications. We further evaluate the performance of longer pre-training epochs of 1600, VisionLLaMA-B achieves 50.2% mIoU on the ADE20K validation set, which boosts ViT-B by 2.1% mIoU.

#### 4.4 Object Detection on COCO

Supervised Training. We evaluate the performance of pyramid VisionLLaMA on the COCO objection detection task. Specifically, we use the Mask RCNN

framework [26] and replace the backbone with pyramid VisionLLaMA, which is pre-trained for 300 epochs on ImageNet-1K as [12,42]. Since our target is not to achieve a new state-of-the-art detector, this carefully controlled experiment is used to verify the validity of our method. The hyperparameter setting is provided in Section D.8(supp.). We report the result in Table 5. Our model outperforms both Swin and Twins. Specifically, VisionLLaMA-B exceeds Swin-S by 1.5% box mAP and 1.0 mask mAP. Compared with the stronger baseline Twins-B, ours also has an advantage of 1.1% higher box mAP and 0.8% higher mask mAP.

Table 5: Performance on the COCO2017 dataset using Mask R-CNN.

Mask R-CNN 3× + MS APb APb50 APb75 APm APm50 APm75 Swin-S [42] ImageNet1k 300E 69.1 47.6 69.4 52.5 42.8 66.5 46.4

Setting Param (M)

Backbone

Twins-SVT-B [12] ImageNet1k 300E 76.3 48.0 69.5 52.7 43.0 66.8 46.6 Pyramid VisionLLaMA-B ImageNet1k 300E 76.3 49.1 70.5 54.0 43.8 67.4 47.0

Self-Supervised Training. We apply VisionLLaMA based on ViTDet [38], which utilizes plain transformers to achieve comparable performance as the pyramid counterpart. Specifically, we replace the vit-Base backbone (trained for 1600 epochs using MAE) with our VisionLLaMA-Base model, which is pre-trained for 800 epochs. The original ViTDet converges slowly and requires dedicated training strategies like longer training epochs (e.g., 100) to achieve good performance. During the training process, we find VisionLLaMA achieves similar performance after 30 epochs. Therefore, we directly utilize the standard 3× training strategy. Therefore, our training cost is only 36% of the baseline. Unlike [38], we do not search for the optimal hyperparameter. The result is shown in Table 6 and VisionLLaMA outperforms ViT-B by 0.6% Box mAP and 0.8% mask mAP.

Table 6: Object detection result on COCO 2017 dataset based on ViTDet [38].

Model Pretrained mAPBox mAPMask Epochs

Swin-S [42] ImageNet sup 300e 47.6 42.8 36 Twins-SVT-B [12] ImageNet sup 300e 48.0 43.0 36 ViT-B [38] MAE 1600e 51.6 45.7 100 VisionLLaMA-B MAE 800e 52.2 46.3 36

### 5 Ablation Study and Discussion

#### 5.1 Ablation Studies

Unless otherwise specified, we choose the ViT-Large model (160I 800E+224I 20E) to perform ablations because we observe that it generates small variance across multiple runs, where a performance gap of more than 0.2 suffices as a guide to choosing appropriate components. We give ablations in Table 7 and more ablations are provided in Section B (supp.).

- Table 7: Ablation experiments with plain transformer ViT-L/16 (DeiT3-L) on ImageNet-1K. We report the top-1 accuracy (%). If not specified, the default is: and the pre-training length is 800 epochs under an image resolution of 160×160 and 20 epochs using 224×224. Default settings are marked in gray . †: running the release code. All accuracies are top-1.

###### (d) RoPE base freq.

(a) SwiGLU/GELU

###### (c) RoPE Ratio

###### (b) Normalization.

Base Acc

Ratio Acc 25% 84.5 50% 84.5

case Acc SwiGLU 84.6 GELU 84.6

case Acc Speed

100 84.6 1000 84.6

LayerNorm [3] 84.6 0.4971s RMSNorm [78] 84.4 0.4874s

10000 84.6 100000 84.4

100% 84.6

(f) Feature extraction. The class token is better than GAP in the training setting of DeiT3 [63].

(g) PE comparison. Applying PEG [13] can further improve the performance. P-V-LLaMA: Pyramid VisionLLaMA, LPE: learnable PE

(e) Shared PE across different heads. Shared PE for all heads is better.

Method Class Head Acc VisionLLaMA-S Class Token 81.6 VisionLLaMA-S GAP 81.8

Shared PE Acc N 84.2 Y 84.6

case Acc

VisionLLaMA-B Class Token 83.6 VisionLLaMA-B GAP 83.6

P-LLaMA-S 81.6 P-V-LLaMA-S + LPE [62] 81.6 P-V-LLaMA-S + PEG [13] 81.8

VisionLLaMA-L Class Token 84.6 VisionLLaMA-L GAP 84.3

DeiT3-L [63] Class Token 84.5 DeiT3-L† GAP 84.2

Ablation of GELU and SwiGLU. We replace GELU with SwiGLU and report the result in Table 7a. We do not observe performance gaps, therefore, we utilize SwiGLU and avoid introducing extra modifications to the LLaMA architecture. This also motivates us to focus on the ablation of the self-attention block. As we apply multi-head self-attention, the remaining two differences become the normalization and positional encoding.

Ablation of the normalization strategy. We compare the two widely used normalization methods in transformers: RMSNorm [78] and LayerNorm [3] and report the result in Table 7b. The latter has a better final performance, which indicates that re-centering invariance is also important in the vision tasks. We also report the training speed by the average time spent per iteration, where LayerNorm is only 2% slower than RMSNorm. Therefore, we choose LayerNorm

instead of RMSNorm for better tradeoff. Note that the training speed might differ across different hardware devices and might also be affected by the overall architecture.

Partial PE. We adjust the ratio of overall channels using RoPE to report the result in Table 7c, which shows good performance can be achieved if the ratio is set above a small threshold value. We do not observe significant differences across these settings. Therefore, we keep the default setting of [64] and do not follow [5,30].

Positional encoding strategy. We also add other absolute position encoding strategies such as a learnable PE [62] and PEG [13] on pyramid VisionLLaMA-S. We use the ‘small’ model due to the existence of a strong baseline and report the result in Table 7g. While the learnable PE does not boost performance, PEG slightly improves the baseline from 81.6% to 81.8%. However, we do not include PEG as a basic component regarding three aspects. Firstly, we try to keep the smallest modifications on LLaMA [64]. Secondly, our target is proposing a universal approach for various tasks like ViT [22]. For masked image frameworks like MAE [25], it is non-trivial to keep the reduced training cost of masked tokens if the backbone contains PEG. If we mask patches in the input like [73], it would greatly slow down the training speed.

Sensitivity to the input size. We further compare the performance on the enlarged and commonly used resolutions without training to report the result in Table 12(supp.). Here we use the pyramid transformer since it is more popular in downstream tasks than the plain counterpart. It is not surprising that 1DRoPE severely suffers from the changed resolutions. NTK-Aware interpolation with α = 2 achieves similar performance as the 2D-RoPE, which is indeed NTKAware (α = 1). AS2DRoPE shows the best performance for larger resolution.

#### 5.2 Discussion

We further investigate the underlying mechanisms behind our method’s superior performance over ViT in various tasks. In this section, we discuss the boosted convergence speed and attempt to theoretically rationalize the mechanism.

Convergence speed. For image generation, we study the performance w.r.t the training steps. Specifically, we store the checkpoint at 100k, 200k, 300k, and 400k iterations to calculate the fidelity metrics. Since SDE is significantly slower than ODE, we opt to use the ODE sampler instead. The result of the strictly controlled experiment is listed in Table 10(supp.). It appears that VisionLLaMA converges much faster than ViT across all models. SiT-LLaMA with 300k training iterations even outperforms the baseline with 400k steps.

We also compare the convergence speed using the DeiT3-Large under the supervised training setting on ImageNet to show the top-1 validation accuracy during the 800 epochs in Figure 3(supp.). It also indicates that VisionLLaMA converges faster than DeiT3-L. We further compare the training loss across 800 epochs of the ViT-Base model under the MAE framework [25] and illustrate it in Figure 4(supp.). VisionLLaMA has lower training loss at the beginning and the trend is kept till the end.

Theoretical analysis. We dive into the mechanism of our positional encodings from the theoretical viewpoint. Without loss of generality, given an input embedding of dimension d = 4, the query at location (i,j) can be written as qi,j. We use ki,j to represent the key vector at (i,j) and pi,j to be the positional encoding using 2D sin-cos encoding [25,45]. The inner dot product between qi

1,j1 and ki

2,j2 using this additive encoding can be written as, q⊤i

1,j1)⊤(ki

1,j1ki

2,j2 = (qi

1,j1 + pi

2,j2 + pi

2,j2)

= q⊤i

2,j2 + p⊤i

2,j2 + q⊤i

2,j2 + p⊤i

(5)

1,j1ki

1,j1pi

1,j1pi

1,j1ki

2,j2

= q⊤i

2,j2 + f(i1 − i2,j1 − j2) + M.

1,j1ki

The first item is the inner dot product of contents. The second item reflects the positional effect in the form of f(i1 − i2,j1 − j2), which plays a long-distance decaying effect. However, the third item M = q⊤i

2,j2 means positions directly interacting with the content features, which slows down the learning process.

2,j2 + p⊤i

1,j1pi

1,j1ki

In contrast, the inner dot product using RoPE can be written as, (Ri

1,j1R⊤i

1,j1)⊤(Ri

2,j2) = q⊤i

1,j1Ri

1,j1qi

2,j2ki

2,j2ki

2,j2

(6)

= q⊤i

1,j1Ri

1−i2,j1−j2ki

2,j2.

#### Ri

1−i2,j1−j2 contributes a larger absolute value if the positions of q and k are close, and a smaller value if opposite. This introduces certain localities as a prior bias, which resembles the function of a convolution. Moreover, Ri

1−i2,j1−j2 adjusts the dot product by the multiplication of a factor between 0 and 1, which is more flexible and faster than the addition of f(i1 − i2,j1 − j2). We believe that this flexibility allows the transformer to leverage its model capacity effectively, learning a good representation without dedicating some of that capacity to introducing bias or separating position from content. In this way, VisionLLaMA not only converges faster but also has better final performance.

### 6 Conclusion

In a nutshell, we present VisionLLaMA to enjoy the benefits of the LLaMA architecture in the vision modality. It is trained either in supervised or selfsupervised schemes to validate the power in a myriad of downstream vision tasks including image classification, detection, and segmentation, among many others. We particularly explore its image generation capacity under the diffusion framework DiT and SiT to confirm its potency. We conclude that VisionLLaMA has a strong potential to serve as a new vision backbone to facilitate a large realm of downstream applications.

### Acknowledgements

This work was in part supported by National Key R&D Program of China (No. 2022ZD0118700).

### A More Experiments

- A.1 Image Generation

Image generation based on the SiT framework. SiT [45] has a flexible choice of drift and diffusion coefficients, which is supported by the recently proposed interpolant framework [1]. It improves the performance of image generation using vision transformers by clear margins. Orthogonally, we replace the vision transformer in SiT with VisionLLaMA to evaluate the benefits of better model architecture, which we call SiT-LLaMA. Our implementation is based on the released code of [45] with carefully controlled experiments. Specifically, we do not change the hyperparameters, although its default setting may be suboptimal. All the models are trained using the same number of steps. We use linear interpolant and the velocity model for all experiments. To make fair comparisons, we also rerun the released code and sample 50k 256×256 images using the 250 steps SDE sampler (Euler) and report the result in Table 8. SiT-LLaMA uniformly outperforms SiT across models with various levels of capacities by clear margins. Compared with SiT-L/2, SiT-LLaMA-L/2 decreases by 5.0 FID, whose magnitude is larger than the boost from the invention of a new framework (4.0 FID). We also report the more efficient ODE sampler (dopri5) in Table 9, our performance gap remains. Similar to the observation of [45], SDE has better performance than its ODE counterpart.

Table 8: Image generation of 256×256 using SiT [45] without CFG. The FID is calculated by a 250-step SDE Euler sampler. †: reproduced result using the released code.

Model Flops Params Steps lr FID↓ sFID↓ Precision↑ Recall↑ IS↑ (G) (M) (K)

SiT-S/2 † 6.06 33 400 0.0001 58.15 9.12 41.01 60.23 24.72 SiT-LLaMA-S/2 6.06 33 400 0.0001 53.90 8.78 42.98 60.36 26.74

SiT-B/2 † 23.01 130 400 0.0001 35.54 6.57 52.68 64.38 42.33 SiT-LLaMA-B/2 23.01 130 400 0.0001 29.53 6.32 56.07 64.07 50.13

DiT-L/2 80.71 458 400 0.0001 23.3 - - - -

SiT-L/2 † 80.71 458 400 0.0001 19.34 5.28 63.00 63.60 70.47 SiT-LLaMA-L/2 80.71 458 400 0.0001 14.32 5.17 66.39 63.64 86.85

SiT-XL/2 † 118.64 675 400 0.0001 16.98 5.07 65.12 64.10 77.06 SiT-LLaMA-XL/2 118.64 675 400 0.0001 12.20 5.03 67.86 63.08 95.28

We evaluate the image generation using the 250 steps ODE sampler (dopri5) based on the SiT framework in Table 9.

- A.2 Image Classification

### B More Ablations

Shared PE for each head. We find that sharing the same PE across different heads (the frequency varies from 1 to 10000 in each head) is better than inde-

- Table 9: Image generation comparisons using the SiT framework [45] All the models are trained using an image resolution of 256×256 with a global batch size of 256. Metrics are calculated using the sampled 50k images without classifier-free guidance. IS: inception score. The FID is calculated by a 250 steps ODE sampler because of the efficiency, which is a bit different from [45]. †: reproduced result using the released code.

Model Flops Params Steps lr FID↓ sFID↓ Precision↑ Recall↑ IS↑ (G) (M) (K)

SiT-S/2 † 6.06 33 400 0.0001 59.60 9.16 39.41 58.48 23.32 SiT-LLaMA-S/2 6.06 33 400 0.0001 54.62 8.81 41.69 61.16 26.07

SiT-B/2 † 23.01 130 400 0.0001 36.90 6.70 51.00 64.10 39.78 SiT-LLaMA-B/2 23.01 130 400 0.0001 30.23 6.36 54.99 64.90 48.34

DiT-L/2 80.71 458 400 0.0001 23.3 - - - -

SiT-L/2 † 80.71 458 400 0.0001 20.14 5.34 61.53 64.53 67.08 SiT-LLaMA-L/2 80.71 458 400 0.0001 14.91 5.16 64.97 64.30 82.23

SiT-XL/2 † 118.64 675 400 0.0001 17.83 5.13 63.52 64.61 73.64 SiT-LLaMA-XL/2 118.64 675 400 0.0001 12.79 5.02 66.77 64.37 90.93

- Table 10: FID calculated with the 250-step ODE sampler in view of efficiency based on the SiT framework.

Method 100k 200k 300k 400k SiT-S/2 89.9 71.9 64.5 59.6

SiT-LLaMA-S/2 82.88 67.1 59.3 54.6

SiT-B/2 65.76 48.37 41.05 36.90 SiT-LLaMA-B/2 56.60 40.62 34.09 30.22

SiT-L/2 45.07 29.11 23.40 20.14 SiT-LLaMA-L/2 35.39 21.82 17.23 14.91

SiT-XL/2 42.25 26.49 20.89 17.83 SiT-LLaMA-XL/2 40.46 19.00 14.84 12.79

pendent ones (the frequency varies from 1 to 10000 across all channels). The result is shown in Table 7e.

Frequency base. We change the base frequency and report the result in Table 7d, which means the performance is robust to a large range of frequencies. As a result, we keep the default value of [64] to avoid extra special treatments for deployment.

Feature abstraction strategy. We compare the two common feature extraction strategies: class token [22] and GAP [13] using the plain ‘large’ model and report the result in Table 7f. Using a class token is better than GAP, which is different from [13]. However, the training settings of the two cases are quite different. We also make an extra experiment using DeiT3-L to observe a similar performance gap of 0.3%. We further evaluate the performance of the ‘small’ and ‘base’ models. It’s interesting to see the opposite conclusions for the small

- Table 11: Top-1 accuracy comparison on different resolutions. The models are trained on 224 and directly evaluated on other resolutions.

Model 160 224 256 288 512 768 DeiT3-Large [63] 83.1 84.5 84.7 84.6 82.1 76.5 VisionLLaMA-L 83.1 84.6 84.7 84.8 83.5 79.1

80

60

Top-1Accuracy

40

20

VisionLLaMA-L

DeiT3-Large

0

0 100 200 300 400 500 600 700 800 Epoch

Fig. 3: Faster convergence of VisionLLaMA using the setting of DeiT3.

model. We suspect that the higher drop-path rate used in [63] makes it difficult for the parameter-free abstraction such as GAP to fit in the purpose.

### C More Related Work

Masked Image Modeling. Masked image modeling is a powerful pre-training scheme that learns strong representations. BEiT [4] extends BERT [20] to computer vision by pre-training a Transformer model with masked embeddings to predict discrete visual tokens. Masked Autoencoder (MAE) [25] is a selfsupervised learning approach that masks random patches of input images and trains an autoencoder to reconstruct the original images. SiMMIM [73] is a simplified version of the MAE approach that uses a lightweight one-layer head to predict raw pixel values. MaskFeat [69] is an extension of the MAE approach that involves predicting not only the raw pixel values of the masked patches but also additional features such as handcrafted HOG descriptor [17] and deep features, which can improve the performance of the model on downstream tasks. Diffusion Models. Diffusion models, represented by Denoising Diffusion Probabilistic Models (DDPMs) [28,58], score-based generative models (SGMs) [32,59] and classifier-free diffusion guidance [29], are the new de facto paradigm for im-

ViT-B

VisionLLaMA-B

0.7

0.6

Loss

0.5

0.4

0 100 200 300 400 500 600 700 800 Epoch

- Fig. 4: Loss curve of MAE pre-training on VisionLLaMA compared with ViT-B.

- Table 12: Top-1 accuracy on different resolutions of the pyramid small model. The models are trained on 224×224 and directly evaluated on other resolutions.

Model 224 448 512

- 1D-RoPE 81.5 0.01 0.01
- 2D-RoPE 81.6 79.5 78.4 NTK(α = 2) 81.6 79.6 78.5 NTK(α = 5) 81.3 79.6 78.6 NTK(α = 10) 81.1 79.6 78.6 AS2DRoPE 81.6 80.3 79.5

age generation, surpassing the previous methodology GAN [24]. The mechanism of diffusion models is based on the idea of gradually adding noise to data and then learning to denoise it. Challenges remain for the computationally expensive training and sampling process, the need for large amounts of data for training, and the difficulty in controlling the generation process. Most lately, OpenAI brings about transformer-based text-conditional diffusion models (the largest one called Sora) [6] jointly trained on videos and images of variable durations, resolutions, and aspect ratios to deliver high-fidelity videos simulating real-world scenes. The recent and concurrent work [44] explores how to deal with image generation with flexible target resolutions. Compared with [44], our target is to build a universal vision transformer for various vision tasks.

|(0,0)|(0,1)|(0,2)|(0,3)|
|---|---|---|---|
|(1,0)|(1,1)|(1,2)|(1,3)|
|(2,0)|(2,1)|(2,2)|(2,3)|
|(3,0)|(3,1)|(3,2)|(3,3)|

- Fig. 5: Position calibration for GSA’s keys using a simple case of 4 × 4 resolution and a kernel size of 2 × 2. The positions of the four points (abstraction keys) are (0.5, 0.5), (1, 2.5), (2.5, 0.5), (2.5, 2.5).

### D Hyperparameters

#### D.1 Supervised Training of VisionLLaMA on ImageNet-1K

As for the plain transformer, we use the same hyperparameters as [63]. The detailed setting is provided in Table 13. VisionLLaMA-S is trained on ImageNet1K for 800 epochs with a resolution of 224×224. VisionLLaMA-B is first trained for 800 epochs with an input size of 192× 192 and then fine-tuned for 20 epochs on 224 × 224. VisionLLaMA-L is first trained for 800 epochs with a resolution of 160 × 160 and then finetuned for 20 epochs on 224 × 224.

#### D.2 Supervised Training of Pyramid VisionLLaMA

We use the same setting as [12]. Specifically, all the models are trained on ImageNet-1K for 300 epochs with a global batch size of 1024 using the AdamW optimizer. The learning rate is increased to 0.001 within 5 warm-up epochs and decayed to zero following the cosine schedule. We use the same data augmentation as [12] and an image resolution of 224×224 for all models. To avoid overfitting, we use a weight decay of 0.05 and drop path [31] (0.2, 0.3, 0.5 for small base and large models respectively).

#### D.3 Mask Image Modeling on ImageNet

We use AdamW optimizer with momentum β1 = 0.9 and β2 = 0.95. The global batch size is 4096. The initial learning rate is 1.5×10−4 and decayed to zero within 800 or 1600 epochs. We also use 40 epochs to warm up the learning rate. We only use simple data augmentation RRC(random-resize-crops)as [25]. Besides, we use a weight decay of 0.05.

Table 13: Training procedures with ImageNet-1K.

Sec D.2 Sec D.1 Batch size 1024 2048 Optimizer AdamW LAMB LR 1.10−3 3.10−3 LR decay cosine cosine Weight decay 0.05 0.02 Warmup epochs 5 5 Label smoothing ε 0.1 ✗ Dropout ✗ ✗ Stoch. Depth ✓ ✓ Repeated Aug ✓ ✓ Gradient Clip. ✗ 1.0 H. flip ✓ ✓ RRC ✓ ✓ Rand Augment 9/0.5 ✗ 3 Augment ✗ ✓ LayerScale ✗ ✓ Mixup alpha 0.8 0.8 Cutmix alpha 1.0 1.0 Erasing prob. 0.25 ✗ ColorJitter ✗ 0.3 Test crop ratio 0.875 1.0 Loss CE BCE

#### D.4 Linear Probing on ImageNet

We follow the setting of [25] and show the details in Table 14.

#### D.5 SFT on ImageNet for SSL-pre-trained Models

We follow the same setting of [25] and show the details in Table 15. The only modification is the layer-wise learning rate decay because we find 0.75 of [25] is overfitting for our method, and we set it to 0.45.

#### D.6 Segmentation for SSL-pre-trained Models

We follow the default setting of [12]. We use AdamW [43] optimizer with β1 = 0.9 and β2 = 0.999. The global batch size is 16. The initial learning rate is 6× 10−5 and linearly decayed to zero. We also use 1500 iterations to warm up. We also utilize l2 weight decay of 0.05 and a drop-path rate [31] of 0.1.

###### Table 14: Linear probing setting.

config value

optimizer LARS [76] base learning rate 0.1

weight decay 0 optimizer momentum 0.9 batch size 16384

learning rate schedule cosine decay warm-up epochs 10

training epochs 90 augmentation RandomResizedCrop

###### Table 15: End-to-end fine-tuning setting for SSL.

config value

optimizer AdamW base learning rate 1e-3

weight decay 0.05 optimizer momentum β1, β2=0.9, 0.999 layer-wise lr decay [4] 0.45

batch size 1024

learning rate schedule cosine decay warmup epochs 5 training epochs 100 (B), 50 (L)

augmentation RandAug (9, 0.5) [16] label smoothing [61] 0.1

mixup [79] 0.8 cutmix [77] 1.0 drop path 0.1

#### D.7 Segmentation for Pyramid Transformers

We follow the setting of [12], which is almost the same as D.6. We use a drop-path rate of 0.2 for the pyramid VisionLLaMA-B model.

#### D.8 Object Detection of Pyramid Transformers

We use the same setting as [12]. We use the AdamW optimizer with β1 = 0.9 and β2 = 0.999. All the models are trained for 36 epochs with a global batch size of 16. The initial learning rate is 1×10−4 with 1000 iterations warm up and decayed by 10.0 at epoch 27 and 33. To avoid overfitting, we apply a l2 weight decay for all models.

#### D.9 Object Detection of Plain Transformers

We use the AdamW optimizer with β1 = 0.9 and β2 = 0.999. The training resolution is fixed as 1024 × 1024 as [38]. Our model is trained for 36 epochs

with a global batch size of 64. The initial learning rate is 1×10−4 with 1000 iterations warm up and decayed by 10.0 at epoch 27 and 33. We use a L2 weight decay of 0.1. We also apply layer-wise learning rate decay with 0.7 as [38].

#### D.10 Image Generation of DiT-LLaMA

We use the same VAE as [52]. We make use of the AdamW optimizer with momentum β1 = 0.9 and β2 = 0.999. We use a global batch size of 256 across all models. The learning rate is fixed as 1 × 10−4. The training resolution is 256 × 256. As for inference, we use 250 steps DDPM. We keep the default setting of ADM [21] without tuning. Specifically, we use tmax = 1000 linear schedule with β from 0.0001 to 0.02 and learnable variance σθ.

#### D.11 Image Generation of SiT-LLaMA

We use the same VAE as SD [52]. As for the ODE sampler, we utilize dopri5 and set ‘atol’ and ‘rtol’ to 1e−6 and 1e−3 respectively.

### E Architecture Setting

- E.1 Pyramid VisionLLaMA The detailed setting of the pyramid architecture is shown in Table 16.

Table 16: Configuration details of Pyramid VisionLLaMA.

Output Size Layer Name S B L

- Stage 1 H 4 × W4

|Patch Embedding<br><br>|P1 = 4; C1 = 64|P1 = 4; C1 = 96|P1 = 4; C1 = 128|
|---|---|---|---|
| |LSA GSA × 1<br><br>|LSA GSA × 1|LSA GSA × 1|

- Stage 2 H 8 × W8

|Patch Embedding|P2 = 2; C2 = 128<br><br>|P2 = 2; C2 = 192|P2 = 2; C2 = 256|
|---|---|---|---|
| |LSA GSA × 1<br><br>|LSA GSA × 1|LSA GSA × 1|

- Stage 3 H 16 × W16

|Patch Embedding|P3 = 2; C3 = 256<br><br>|P3 = 2; C3 = 384|P3 = 2; C3 = 512|
|---|---|---|---|
| |LSA GSA × 5|LSA GSA × 9<br><br>|LSA GSA × 9|

- Stage 4 H 32 × W32

|Patch Embedding|P4 = 2; C4 =512<br><br>|P4 = 2; C4 =768|P4 = 2; C4 =1024|
|---|---|---|---|
| |GSA × 4<br><br>|GSA × 2|GSA × 2|

- E.2 Plain Transformer for Vision Understanding

- The detailed setting of the architecture is shown in Table 17.

Table 17: Architecture settings for VisionLLaMA on image understanding tasks.

Model Layers Dims Heads

VisionLLaMA-S 12 384 6 VisionLLaMA-B 12 768 12 VisionLLaMA-L 24 1024 16

Table 18: Architecture settings for VisionLLaMA on image generation.

Model Layers Dims Heads

DiT-LLaMA-S / SiT-LLaMA-S 12 384 6 SiT-LLaMA-B / SiT-LLaMA-B 12 768 12 SiT-LLaMA-L / SiT-LLaMA-L 24 1024 16

SiT-LLaMA-XL / SiT-LLaMA-XL 28 1152 16

#### E.3 Plain Transformer for Generation

- The detailed setting of the architecture is shown in Table 18.

### References

- 1. Albergo, M.S., Boffi, N.M., Vanden-Eijnden, E.: Stochastic interpolants: A unifying framework for flows and diffusions. arXiv: Comp. Res. Repository (2023) 15
- 2. Sc ao, T.L., Fan, A., Akiki, C., Pavlick, E., Ilić, S., Hesslow, D., Castagné, R., Luccioni, A.S., Yvon, F., Gallé, M., et al.: Bloom: A 176b-parameter open-access multilingual language model. arXiv: Comp. Res. Repository (2022) 3
- 3. Ba, J.L., Kiros, J.R., Hinton, G.E.: Layer normalization. arXiv: Comp. Res. Repository (2016) 4, 12
- 4. Bao, H., Dong, L., Piao, S., Wei, F.: Beit: Bert pre-training of image transformers. arXiv: Comp. Res. Repository (2021) 17, 21
- 5. Biderman, S., Schoelkopf, H., Anthony, Q.G., Bradley, H., O’Brien, K., Hallahan, E., Khan, M.A., Purohit, S., Prashanth, U.S., Raff, E., et al.: Pythia: A suite for analyzing large language models across training and scaling. In: Proc. Int. Conf. Mach. Learn. (2023) 13
- 6. Brooks, T., Peebles, B., Homes, C., DePue, W., Guo, Y., Jing, L., Schnurr, D., Taylor, J., Luhman, T., Luhman, E., Ng, C.W.Y., Wang, R., Ramesh, A.: Video generation models as world simulators (2024) 18
- 7. Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J.D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., et al.: Language models are few-shot learners. Proc. Advances in Neural Inf. Process. Syst. (2020) 3
- 8. Chen, S., Wong, S., Chen, L., Tian, Y.: Extending context window of large language models via positional interpolation. arXiv: Comp. Res. Repository (2023) 3, 6
- 9. Chen, X., Liu, Z., Xie, S., He, K.: Deconstructing denoising diffusion models for self-supervised learning. arXiv: Comp. Res. Repository (2024) 9
- 10. Chu, X., Qiao, L., Lin, X., Xu, S., Yang, Y., Hu, Y., Wei, F., Zhang, X., Zhang, B., Wei, X., et al.: Mobilevlm: A fast, reproducible and strong vision language assistant for mobile devices. arXiv: Comp. Res. Repository (2023) 3

- 11. Chu, X., Qiao, L., Zhang, X., Xu, S., Wei, F., Yang, Y., Sun, X., Hu, Y., Lin, X., Zhang, B., et al.: Mobilevlm v2: Faster and stronger baseline for vision language model. arXiv: Comp. Res. Repository (2024) 3
- 12. Chu, X., Tian, Z., Wang, Y., Zhang, B., Ren, H., Wei, X., Xia, H., Shen, C.: Twins: Revisiting the design of spatial attention in vision transformers. In: Proc. Advances in Neural Inf. Process. Syst. (2021) 2, 4, 5, 8, 10, 11, 19, 20, 21
- 13. Chu, X., Tian, Z., Zhang, B., Wang, X., Shen, C.: Conditional positional encodings for vision transformers. In: Proc. Int. Conf. Learn. Representations (2023) 3, 5, 7, 8, 12, 13, 16
- 14. Contributors, M.: Openmmlab’s pre-training toolbox and benchmark (2023) 9
- 15. Contributors, M.: Mmsegmentation: Openmmlab semantic segmentation toolbox and benchmark (2020) 10
- 16. Cubuk, E.D., Zoph, B., Shlens, J., Le, Q.V.: Randaugment: Practical automated data augmentation with a reduced search space. In: Proc. IEEE Conf. Comp. Vis. Patt. Recogn. (2020) 21
- 17. Dalal, N., Triggs, B.: Histograms of oriented gradients for human detection. In: 2005 IEEE computer society conference on computer vision and pattern recognition (CVPR’05) (2005) 17
- 18. Dao, T.: Flashattention-2: Faster attention with better parallelism and work partitioning. arXiv: Comp. Res. Repository (2023) 6
- 19. Deng, J., Dong, W., Socher, R., Li, L.J., Li, K., Fei-Fei, L.: Imagenet: A largescale hierarchical image database. In: Proc. IEEE Conf. Comp. Vis. Patt. Recogn.

(2009) 3, 7

- 20. Devlin, J., Chang, M.W., Lee, K., Toutanova, K.: Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv: Comp. Res. Repository

(2018) 17

- 21. Dhariwal, P., Nichol, A.: Diffusion models beat gans on image synthesis. Proc. Advances in Neural Inf. Process. Syst. (2021) 6, 22
- 22. Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., et al.: An image is worth 16x16 words: Transformers for image recognition at scale. arXiv: Comp. Res. Repository (2020) 2, 3, 13, 16
- 23. Frantar, E., Ashkboos, S., Hoefler, T., Alistarh, D.: Gptq: Accurate post-training quantization for generative pre-trained transformers. arXiv: Comp. Res. Repository

(2022) 2

- 24. Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., Bengio, Y.: Generative adversarial nets. In: Proc. Advances in Neural Inf. Process. Syst. (2014) 18
- 25. He, K., Chen, X., Xie, S., Li, Y., Dollár, P., Girshick, R.: Masked autoencoders are scalable vision learners. In: Proc. IEEE Conf. Comp. Vis. Patt. Recogn. (2022) 3, 9, 10, 13, 14, 17, 19, 20
- 26. He, K., Gkioxari, G., Dollár, P., Girshick, R.: Mask r-cnn. In: Proc. IEEE Int. Conf. Comp. Vis. (2017) 11
- 27. Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., Hochreiter, S.: Gans trained by a two time-scale update rule converge to a local nash equilibrium. Proc. Advances in Neural Inf. Process. Syst. (2017) 7
- 28. Ho, J., Jain, A., Abbeel, P.: Denoising diffusion probabilistic models. Proc. Advances in Neural Inf. Process. Syst. (2020) 3, 6, 17
- 29. Ho, J., Salimans, T.: Classifier-free diffusion guidance. In: NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications (2021) 17

- 30. https://stability.ai/: Stable code 3b: Coding on the edge (2024) 13
- 31. Huang, G., Sun, Y., Liu, Z., Sedra, D., Weinberger, K.Q.: Deep networks with stochastic depth. In: Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11–14, 2016, Proceedings, Part IV 14

(2016) 19, 20

- 32. Hyv"arinen, A., Dayan, P.: Estimation of non-normalized statistical models by score matching. J. Mach. Learn. Res. (2005) 17
- 33. Kingma, D.P., Welling, M.: Auto-encoding variational bayes. arXiv: Comp. Res. Repository (2013) 6
- 34. Kynkäänniemi, T., Karras, T., Laine, S., Lehtinen, J., Aila, T.: Improved precision and recall metric for assessing generative models. Proc. Advances in Neural Inf. Process. Syst. (2019) 7
- 35. Lai, X., Tian, Z., Chen, Y., Li, Y., Yuan, Y., Liu, S., Jia, J.: Lisa: Reasoning segmentation via large language model. arXiv: Comp. Res. Repository (2023) 3
- 36. Li, G., Zheng, H., Liu, D., Wang, C., Su, B., Zheng, C.: Semmae: Semantic-guided masking for learning masked autoencoders. Proc. Advances in Neural Inf. Process. Syst. (2022) 9, 10
- 37. Li, L., Li, Q., Zhang, B., Chu, X.: Norm tweaking: High-performance low-bit quantization of large language models. In: Proc. AAAI Conf. Artificial Intell. (2024) 2
- 38. Li, Y., Mao, H., Girshick, R., He, K.: Exploring plain vision transformer backbones for object detection. In: Proc. Eur. Conf. Comp. Vis. (2022) 11, 21, 22
- 39. Liu, H., Li, C., Li, Y., Lee, Y.J.: Improved baselines with visual instruction tuning. arXiv: Comp. Res. Repository (2023) 3
- 40. Liu, H., Li, C., Wu, Q., Lee, Y.J.: Visual instruction tuning. Pattern Recogn. (2023) 3
- 41. Liu, Y., Zhang, S., Chen, J., Yu, Z., Chen, K., Lin, D.: Improving pixel-based mim by reducing wasted modeling capability. In: Proc. IEEE Int. Conf. Comp. Vis.

(2023) 9, 10

- 42. Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., Guo, B.: Swin transformer: Hierarchical vision transformer using shifted windows. In: Proc. IEEE Int. Conf. Comp. Vis. (2021) 2, 4, 8, 10, 11
- 43. Loshchilov, I., Hutter, F.: Decoupled weight decay regularization. arXiv: Comp. Res. Repository (2017) 20
- 44. Lu, Z., Wang, Z., Huang, D., Wu, C., Liu, X., Ouyang, W., Bai, L.: Fit: Flexible vision transformer for diffusion model. arXiv: Comp. Res. Repository (2024) 18
- 45. Ma, N., Goldstein, M., Albergo, M.S., Boffi, N.M., Vanden-Eijnden, E., Xie, S.: Sit: Exploring flow and diffusion-based generative models with scalable interpolant transformers. arXiv: Comp. Res. Repository (2024) 14, 15, 16
- 46. Nash, C., Menick, J., Dieleman, S., Battaglia, P.W.: Generating images with sparse representations. arXiv: Comp. Res. Repository (2021) 7
- 47. OpenAI: Gpt-4 technical report (2023) 3
- 48. Parmar, G., Zhang, R., Zhu, J.Y.: On aliased resizing and surprising subtleties in gan evaluation. In: Proc. IEEE Conf. Comp. Vis. Patt. Recogn. (2022) 6
- 49. Peebles, W., Xie, S.: Scalable diffusion models with transformers. In: Proc. IEEE Int. Conf. Comp. Vis. (2023) 3, 6, 7
- 50. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from natural language supervision. In: Proc. Int. Conf. Mach. Learn. (2021) 8
- 51. Ramesh, A., Dhariwal, P., Nichol, A., Chu, C., Chen, M.: Hierarchical textconditional image generation with clip latents (2022) 8

- 52. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image synthesis with latent diffusion models. In: Proc. IEEE Conf. Comp. Vis. Patt. Recogn. (2022) 6, 22
- 53. Ronneberger, O., Fischer, P., Brox, T.: U-net: Convolutional networks for biomedical image segmentation. In: Medical Image Computing and Computer-Assisted Intervention–MICCAI 2015: 18th International Conference, Munich, Germany, October 5-9, 2015, Proceedings, Part III 18 (2015) 3
- 54. Roziere, B., Gehring, J., Gloeckle, F., Sootla, S., Gat, I., Tan, X.E., Adi, Y., Liu, J., Remez, T., Rapin, J., et al.: Code llama: Open foundation models for code. arXiv: Comp. Res. Repository (2023) 3
- 55. Salimans, T., Goodfellow, I., Zaremba, W., Cheung, V., Radford, A., Chen, X.: Improved techniques for training gans. Proc. Advances in Neural Inf. Process. Syst. (2016) 7
- 56. Shaw, P., Uszkoreit, J., Vaswani, A.: Self-attention with relative position representations. arXiv: Comp. Res. Repository (2018) 3
- 57. Shazeer, N.: Glu variants improve transformer. arXiv: Comp. Res. Repository

(2020) 3, 4

- 58. Sohl-Dickstein, J., Weiss, E., Maheswaranathan, N., Ganguli, S.: Deep unsupervised learning using nonequilibrium thermodynamics. In: Proc. Int. Conf. Mach. Learn. (2015) 3, 17
- 59. Song, Y., Sohl-Dickstein, J., Kingma, D.P., Kumar, A., Ermon, S., Poole, B.: Scorebased generative modeling through stochastic differential equations. arXiv: Comp. Res. Repository (2020) 17
- 60. Su, J., Ahmed, M., Lu, Y., Pan, S., Bo, W., Liu, Y.: Roformer: Enhanced transformer with rotary position embedding. Int. J. Comput. Vision (2023) 3, 4, 5
- 61. Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., Wojna, Z.: Rethinking the inception architecture for computer vision. In: Proc. IEEE Conf. Comp. Vis. Patt. Recogn. (2016) 21
- 62. Touvron, H., Cord, M., Douze, M., Massa, F., Sablayrolles, A., Jegou, H.: Training data-efficient image transformers and distillation through attention. In: Proc. Int. Conf. Mach. Learn. (2021) 7, 8, 12, 13
- 63. Touvron, H., Cord, M., Jégou, H.: Deit iii: Revenge of the vit. In: Proc. Eur. Conf. Comp. Vis. (2022) 2, 5, 7, 8, 12, 17, 19
- 64. Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.A., Lacroix, T., Rozière, B., Goyal, N., Hambro, E., Azhar, F.: Llama: Open and efficient foundation language models. arXiv: Comp. Res. Repository (2023) 2, 3, 13, 16
- 65. Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y., Bashlykov, N., Batra, S., Bhargava, P., Bhosale, S., et al.: Llama 2: Open foundation and fine-tuned chat models. arXiv: Comp. Res. Repository (2023) 2, 3
- 66. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, Ł., Polosukhin, I.: Attention is all you need. Proc. Advances in Neural Inf. Process. Syst. (2017) 2, 3
- 67. Vishniakov, K., Shen, Z., Liu, Z.: Convnet vs transformer, supervised vs clip: Beyond imagenet accuracy. arXiv: Comp. Res. Repository (2023) 9
- 68. Wang, W., Xie, E., Li, X., Fan, D.P., Song, K., Liang, D., Lu, T., Luo, P., Shao, L.: Pyramid vision transformer: A versatile backbone for dense prediction without convolutions. In: Proc. IEEE Int. Conf. Comp. Vis. (2021) 2
- 69. Wei, C., Fan, H., Xie, S., Wu, C.Y., Yuille, A., Feichtenhofer, C.: Masked feature prediction for self-supervised visual pre-training. In: Proc. IEEE Conf. Comp. Vis. Patt. Recogn. (2022) 9, 10, 17

- 70. Wei, F., Zhang, X., Zhang, A., Zhang, B., Chu, X.: Lenna: Language enhanced reasoning detection assistant. arXiv: Comp. Res. Repository (2023) 3
- 71. Xiao, G., Lin, J., Seznec, M., Wu, H., Demouth, J., Han, S.: Smoothquant: Accurate and efficient post-training quantization for large language models. In: Proc. Int. Conf. Mach. Learn. (2023) 2
- 72. Xiao, T., Liu, Y., Zhou, B., Jiang, Y., Sun, J.: Unified perceptual parsing for scene understanding. In: Proc. Eur. Conf. Comp. Vis. (2018) 10
- 73. Xie, Z., Zhang, Z., Cao, Y., Lin, Y., Bao, J., Yao, Z., Dai, Q., Hu, H.: Simmim: A simple framework for masked image modeling. In: Proc. IEEE Conf. Comp. Vis. Patt. Recogn. (2022) 9, 13, 17
- 74. Xiong, W., Liu, J., Molybog, I., Zhang, H., Bhargava, P., Hou, R., Martin, L., Rungta, R., Sankararaman, K.A., Oguz, B., et al.: Effective long-context scaling of foundation models. arXiv: Comp. Res. Repository (2023) 3
- 75. Yang, A., Xiao, B., Wang, B., Zhang, B., Bian, C., Yin, C., Lv, C., Pan, D., Wang, D., Yan, D., et al.: Baichuan 2: Open large-scale language models. arXiv: Comp. Res. Repository (2023) 3
- 76. You, Y., Gitman, I., Ginsburg, B.: Large batch training of convolutional networks. arXiv: Comp. Res. Repository (2017) 21
- 77. Yun, S., Han, D., Oh, S.J., Chun, S., Choe, J., Yoo, Y.: Cutmix: Regularization strategy to train strong classifiers with localizable features. In: Proc. IEEE Int. Conf. Comp. Vis. (2019) 21
- 78. Zhang, B., Sennrich, R.: Root mean square layer normalization. Proc. Advances in Neural Inf. Process. Syst. (2019) 3, 4, 12
- 79. Zhang, H., Cisse, M., Dauphin, Y.N., Lopez-Paz, D.: mixup: Beyond empirical risk minimization. arXiv: Comp. Res. Repository (2017) 21
- 80. Zhou, B., Zhao, H., Puig, X., Fidler, S., Barriuso, A., Torralba, A.: Scene parsing through ade20k dataset. In: Proc. IEEE Conf. Comp. Vis. Patt. Recogn. (2017) 10
- 81. Zhu, D., Chen, J., Shen, X., Li, X., Elhoseiny, M.: MiniGPT-4: Enhancing visionlanguage understanding with advanced large language models. In: Proc. Int. Conf. Learn. Representations (2024) 3

