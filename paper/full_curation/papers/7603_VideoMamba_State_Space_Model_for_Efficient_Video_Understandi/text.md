# arXiv:2403.06977v2[cs.CV]12Mar2024

[Figure 1]

## VideoMamba: State Space Model for Efficient Video Understanding

Kunchang Li2,3,1♠, Xinhao Li4,1♠, Yi Wang1♡, Yinan He1 Yali Wang2,1♡, Limin Wang4,1♡, and Yu Qiao1♡

1 OpenGVLab, Shanghai AI Laboratory 2 Shenzhen Institute of Advanced Technology, Chinese Academy of Sciences 3 University of Chinese Academy of Sciences 4 State Key Laboratory for Novel Software Technology, Nanjing University https://github.com/OpenGVLab/VideoMamba

Abstract. Addressing the dual challenges of local redundancy and global dependencies in video understanding, this work innovatively adapts the Mamba to the video domain. The proposed VideoMamba overcomes the limitations of existing 3D convolution neural networks and video transformers. Its linear-complexity operator enables efficient long-term modeling, which is crucial for high-resolution long video understanding. Extensive evaluations reveal VideoMamba’s four core abilities: (1) Scalability in the visual domain without extensive dataset pretraining, thanks to a novel self-distillation technique; (2) Sensitivity for recognizing shortterm actions even with fine-grained motion differences; (3) Superiority in long-term video understanding, showcasing significant advancements over traditional feature-based models; and (4) Compatibility with other modalities, demonstrating robustness in multi-modal contexts. Through these distinct advantages, VideoMamba sets a new benchmark for video understanding, offering a scalable and efficient solution for comprehensive video understanding. All the code and models are available.

#### 1 Introduction

The core objective for video understanding lies in mastering spatiotemporal representations, which inherently presents two formidable challenges: the large spatiotemporal redundancy within short video clips, and the complex spatiotemporal dependencies among long contexts. Although the once-dominant 3D convolutional neural networks (CNNs) [9,19,76] and video transformers [2,4], effectively tackle one of the challenges mentioned by leveraging either local convolution or long-range attention, they fall short in addressing both simultaneously. UniFormer [44] attempts to integrate the advantages of both methods, but it struggles with modeling long videos, which has been the major trend in recent research on video understanding [48,72] and generation [5,92].

The emergence of low-cost operators such as S4 [26], RWKV [73], and RetNet [70] in the NLP domain, has carved a novel pathway for the vision model.

♠ Interns at Shanghai AI Laboratory. ♡ Corresponding authors.

+4.1%

[Figure 2]

+/00/$ ! "&10/$ ⚡

[Figure 3]

+4.6%

>>?

>>?

[Figure 4]

40×

2ℎ/&4/$ #

[Figure 5]

& = 8 & = 64

2×

20×

6×

- Fig. 1: Comparisons of throughput and memory. The TimeSformer-Ti [4] is built based on DeiT-Ti [75] with joint spatiotemporal attention. All the input frames are sized to 224×224. The testing is conducted on an NVIDIA A100-80G GPU, utilizing PyTorch 2.1 and CUDA 11.8, with a batch size of 128. Our VideoMamba is better, faster and cheaper for both short-term and long-term video understanding.

Mamba [25] stands out with its selective state space model (SSM), striking a balance between maintaining linear complexity and facilitating long-term dynamic modeling. This innovation has spurred its adoption in vision tasks, as evidenced by Vision Mamba [91] and VMamba [50], which leverage multi-directional SSMs for enhanced 2D image processing. These models rival attention-based architectures in performance while offering a significant reduction in memory usage. Given the inherently longer sequences produced by video, a natural question arises: Can Mamba work well for video understanding?

Inspired by this, we introduce VideoMamba, a purely SSM-based model tailored for video understanding. VideoMamba harmoniously merges the strengths of convolution and attention in vanilla ViT [15] style. It offers a linear-complexity method for dynamic spatiotemporal context modeling, ideal for high-resolution long videos. The related evaluation focuses on VideoMamba’s four key abilities:

##### (1) Scalability in the Visual Domain: We examine VideoMamba’s scal-

ability and find that, while the pure Mamba model tends to overfit as it scales, our introduction of a simple yet effective self-distillation strategy allows VideoMamba to achieve remarkable performance enhancements as the model and input sizes increase, without the need for large-scale dataset pretraining.

##### (2) Sensitivity for Short-term Action Recognition: Our analysis ex-

tends to assessing VideoMamba’s capability to accurately distinguish short-term actions, especially those with fine-grained motion differences, e.g., opening and closing. The findings reveal VideoMamba’s superior performance over existing attention-based models [2,4,52]. More importantly, it is also suitable for masked modeling, which further enhances its temporal sensitivity.

##### (3) Superiority in Long-term Video Understanding: We then assess

VideoMamba’s prowess in interpreting long videos. It showcases remarkable superiority over conventional feature-based methods [35, 47] through end-to-end training. Notably, VideoMamba operates 6× faster than TimeSformer [4] and demands 40× less GPU memory for 64-frame videos (see Fig. 1).

##### (4) Compatibility with Other Modalities: Lastly, we assess VideoMamba’s

adaptability with other modalities. Results in video-text retrievals show its im-

proved performance than ViT, particularly in long videos with complex scenarios. This underscores its robustness and multi-modal integration capacity.

In conclusion, our in-depth experiments reveal VideoMamba’s immense potential in understanding both short-term (K400 [36] and SthSthV2 [24]) and long-term (Breakfast [37], COIN [71], and LVU [84]) video contents. Given its efficiency and effectiveness, VideoMamba is poised to become a cornerstone in the realm of long-video comprehension. All the code and models are open-sourced to foster future research endeavors.

#### 2 Related Works

##### 2.1 State Space Models

Recently, the State Space Models (SSMs) have shown significant effectiveness of state space transformation in capturing the dynamics and dependencies of language sequences. [26] introduces a structured state-space sequence model (S4), specifically designed to model long-range dependencies, boasting the advantage of linear complexity. Based on it, various models have been developed (e.g., S5 [66], H3 [20] and GSS [56]), and Mamba [25] distinguishes itself by introducing a data-dependent SSM layer and a selection mechanism using parallel scan (S6). Compared to transformers [6,54] based on quadratic-complexity attention, Mamba excels at processing long sequences with linear complexity.

In the vision domain, [26] first applies SSM in pixel-level image classification, and [35] uses S4 to handle the long-range temporal dependencies for movie clip classification. Besides, the great potential of Mamba motivates a series of works [28,30,46,50,78,87,91], which demonstrates Mamba’s better performances and higher GPU efficiency than Transformer on visual downstream tasks like object detection and semantic segmentation. Different from the previous works, our VideoMamba is a purely SSM-based video model, showcasing great efficiency and effectiveness for both short-term and long-term video understanding.

##### 2.2 Video Understanding

Video understanding stands as a cornerstone in the domain of computer vision, whose significance is further amplified by the burgeoning growth of short video platforms. To bolster this field, numerous datasets equipped with extensive data and meticulous human annotations have been developed, aiming to enhance human action recognition capabilities. Notable examples include UCF101 [67] and Kinetics dataset [7,8,36], which have played pivotal roles in benchmarking progress. Furthermore, other datasets [22, 27, 31, 34, 49, 62] provide annotated activity videos tailored for action localization, fostering deeper research into human activities. Beyond action recognition, the advent of large-scale videotext datasets [10, 12, 57, 82, 86, 88] extends the utility of video understanding into the realm of multi-modality tasks, such as video captioning, retrieval and question answering, thereby broadening the application spectrum.

As for the architecture, it has evolved from using CNN which extracts features from video frames, to more advanced techniques. Initially, 3D CNNs [9,17, 76,77] expanded the traditional 2D CNN architecture to capture videos’ spatiotemporal information. Two-Stream [65], which combines spatial and temporal streams, TSN [80], which proposes sparse sampling, and SlowFast [19], which uses parallel networks to capture semantics and rapid movements, further enhance action recognition capacity. The introduction of attention-based models [2,4,59, 63,89], like TimeSformer [4] and ViViT [2], marked a significant advancement by effectively capturing long-range dependencies within video sequences, enhancing temporal relationship understanding. Recent developments [42, 44, 52, 83] have focused on accurate video transformer, with innovations like the VideoSwin’s window attention [52] and the UniFormer’s integration of convolution and selfattention mechanisms [44], aiming to balance computational efficiency with performance. Despite these models’ achievements in various tasks, they often come with high computational costs for long sequences. In contrast, our VideoMamba introduces a linear-complexity operator for efficient long-term modeling, outperforming existing methods with faster speed and lower GPU consumption.

#### 3 Method

##### 3.1 Preliminaries

SSM for 1D sequence. State Space Models (SSMs) are conceptualized based on continuous systems that map a 1D function or sequence, x(t) ∈ RL → y(t) ∈ RL through a hidden state h(t) ∈ RN. Formally, SSMs employ the following ordinary differential equation (ODE) to model the input data:

h′(t) = Ah(t) + Bx(t), (1) y(t) = Ch(t), (2)

where A ∈ RN×N represents the system’s evolution matrix, and B ∈ RN×1,C ∈ RN×1 are the projection matrices. This continuous ODE is approximated through discretization in modern SSMs. Mamba [25] is one of the discrete versions of the continuous system, which includes a timescale parameter ∆ to transform the continuous parameters A,B to their discrete counterparts A,B. The transformation typically employs the zero-order hold (ZOH) method, defined by:

- A = exp(∆A), (3)

- B = (∆A)−1(exp(∆A) − I) · ∆B (4) ht = Aht−1 + Bxt, (5) yt = Cht. (6)

Contrary to traditional models that primarily rely on linear time-invariant SSMs, Mamba distinguishes itself by implementing a Selective Scan Mechanism (S6) as its core SSM operator. Within S6, the parameters B ∈ RB×L×N, C ∈

Sequence Transformation

Multiplication Summarization

! Activation

Linear Projection

&.+/

&.+/

5

Conv

SSM

Conv 5

SSM

5

Conv

SSM

5

5

! 7!81! 1 ;'-'.#9)'*+!$ 7!81!

- Fig. 2: Mamba blocks for 1D [25] and 2D [91] sequence. We omit the initial normalization and the final residual for simplification.

[Figure 6]

[Figure 7]

Bidirectional Mamba Block

× L

Classification Head

3D Patch Embedding

0 1 2 3 4 1 2 3 4 1 2 3 4

*

Temporal Position Embedding Spatial Position Embedding

|[Figure 8]<br><br>|[Figure 9]<br><br>CLS|
|---|
<br><br>/!<br><br>/"<br><br>/#<br><br>|CLS|
|---|
<br><br>4 3 2 1<br><br>1 2 3 4<br><br>1 2 3 4<br><br>4 3 2 1<br><br>1 2 3 4<br><br>1 2 3 4<br><br>3456758 9:7; <7:=6758 9:7;|
|---|

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

[Figure 20]

[Figure 21]

[Figure 22]

[CLS] Token

1 2 3

! 6'-#*7!81! 1 "4!)'*)#84*.!$ "9!+

Class

$%&''()* +&%,-()* ./&0()* …

- Fig. 3: Framework of VideoMamba. We strictly follow the architecture of vanilla ViT [15], and adapt the bidirectional mamba block [91] for 3D video sequences.

RB×L×N, and ∆ ∈ RB×L×D are directly derived from the input data x ∈ RB×L×D, indicating an intrinsic capacity for contextual sensitivity and adaptive weight modulation. Fig. 2a shows the details of the Mamba block.

Bidirectional SSM for Vision. The original Mamba block, designed for 1D sequences, falls short for visual tasks requiring spatial awareness. Building on this, Vision Mamba introduces a bidirectional Mamba (B-Mamba) block in Fig. 2b, which adapts bidirectional sequence modeling for vision-specific applications. This block processes flattened visual sequences through simultaneous forward and backward SSMs, enhancing its capacity for spatially-aware processing. In this work, we extend the B-Mamba block for 3D video understanding.

##### 3.2 VideoMamba

Overview. Fig. 3 illustrates the overall framework of VideoMamba. Specifically, we first use 3D convolution (i.e., 1×16×16) to project the input videos Xv ∈ R3×T×H×W into L non-overlapping spatiotemporal patches Xp ∈ RL×C, where L=t×h×w (t=T, h=16H , and w=W16). The sequence of tokens input to the following VideoMamba encoder is

X = [Xcls,X] + ps + pt, (7)

where Xcls is a learnable classification token that is prepended to the start of the sequence. Following previous works [2, 4, 15], we added a learnable spatial position embedding ps ∈ R(hw+1)×C and the extra temporal one pt ∈ Rt×C to retain the spatiotemporal position information, since the SSM modeling is

S T

S T

S T

S T

"4!)'!$−:'.() ;'-'.#9)'*+!$

=#84*.!$−:'.() ;'-'.#9)'*+!$

"4!)'*)#84*.!$ ;'-'.#9)'*+!$ ,1

"4!)'*)#84*.!$ ;'-'.#9)'*+!$ ,2

!

1

9

-

Fig. 4: Different scan methods. We omit the [CLS] token for simplification.

sensitive to token position. The tokens X are then passed through by L stacked B-Mamba blocks, and the representation of [CLS] token at the final layer is processed by normalization and linear layer for classification.

Spatiotemporal Scan. To apply the B-Mamba layer for spatiotemporal input, we extend the original 2D scan into different bidirectional 3D scans in Fig. 4: (a) Spatial-First, organizing spatial tokens by location then stacking them frame by frame; (b) Temporal-First, arranging temporal tokens based on the frame then stacks along the spatial dimension; (c) Spatiotemporal, a hybrid of both SpatialFirst and Temporal-First, with v1 conducting half of them and v2 conducting full of them (2× computation). Moreover, our experiments in Fig. 7a demonstrate that the Spatial-First bidirectional scan is the most effective yet simple. Thanks to the linear complexity of Mamba, our VideoMamba is capable of handling long videos of high resolution efficiently.

Comparison to Vim [91] and VMamba [50]. Our VideoMamba builds upon Vim, yet streamlines its architecture by omitting features such as the middle [CLS] token and Rotary Position Embedding (RoPE [68]), resulting in superior performance on ImageNet-1K with gains of +0.8% and +0.7% for Vim-Ti and Vim-S, respectively. Unlike VMamba, which incorporates additional depthwise convolution, VideoMamba strictly follows the ViT design without downsampling layers. To counter the overfitting issues observed in VMamba, we introduce an effective self-distillation technique outlined in Section 3.3, demonstrate the isotropic VideoMamba’s great scalability for image and video tasks.

Comparison to TimeSformer [4] and ViViT [2]. Traditional attentionbased models like TimeSformer and ViViT have addressed the self-attention mechanism’s quadratic complexity by adopting divided spatiotemporal attention. Despite being more efficient, it introduces additional parameters and underperforms compared to joint attention, particularly in scenarios involving masked pretraining [43,74]. In contrast, VideoMamba processes spatiotemporal tokens with linear complexity, outperforming TimeSformer on Kinetics-400 by +2.6% and making significant strides on SthSthV2 with a +5.9% improvement (see Table 3 and 4). Furthermore, VideoMamba achieves a 6× increase in processing speed and requires 40× less GPU memory for long videos, as detailed in Fig. 1, demonstrating its efficiency and effectiveness in handling long-video tasks.

*+,-

*+,-

*+,-

| |[Figure 23]| | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| |[Figure 24]| | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| |[Figure 25]| | |
| | | | |

|[Figure 26]| | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
|[Figure 27]| | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
|[Figure 28]| | | |
| | | | |

|[Figure 29]| | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
|[Figure 30]| | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | |[Figure 31]| |
| | | | |

! '+4@) ,'-#*

1 .!+-*8 8!(A'+/

###### 9 )@1# 8!(A'+/

*+,-

*+,-

*+,-

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | |[Figure 32]| |

| | | | |
|---|---|---|---|
| | | | |
| | | |[Figure 33]|
| | | | |

| | | | |
|---|---|---|---|
| | | | |
|[Figure 34]| | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | |[Figure 35]|

| | | | |
|---|---|---|---|
| | | | |
|[Figure 36]| | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | |[Figure 37]|
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | |[Figure 38]| |

| | | | |
|---|---|---|---|
| | | | |
| |[Figure 39]| | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | |[Figure 40]|
| | | | |

- 9$'4−.*B 8!(A'+/

# %.!8#−.*B 8!(A'+/

% !))#+)'*+ 8!(A'+/

Fig. 5: Different masking strategies. Row masking, tailored for VideoMamba in light of the 1D convolution preceding SSM, enhances performance with continuous tokens. The difference between clip-row and frame-row masking is that the former masks the entire video clip, while the latter masks each frame individually.

##### 3.3 Architecture

For SSM in the B-Mamba layer, we adopt the default hyperparameters as in Mamba [25]. setting the state dimension and expansion ratio to 16 and 2, respectively. Following ViT [15], we adjust the depth and embedding dimensions to create models of comparable sizes in Table 1, including VideoMamba-Ti, VideoMamba-S and VideoMamba-M. However, we observe that larger VideoMamba tends to overfit during our experiments, leading to suboptimal performance as illustrated in Fig. 6a. This overfitting issue is not unique to our models but is also found in VMamba [50], where the optimal performance of VMamba-B was achieved at three-quarters of the total training epochs. To counteract the overfitting in larger Mamba models, we introduce an effective Self-Distillation strategy, which uses a smaller and well-trained model as the “teacher” to guide the training of the larger “student” model. The results, depicted in Fig. 6a, show that this strategy leads to expected better convergence.

Model #Depth #Dim #Param.

Tiny 24 192 7M Small 24 384 26M Middle 32 576 74M Base 24 768 98M

Table 1: Different model sizes. Base model is finally excluded due to its suboptimization.

##### 3.4 Masked Modeling

Recently, VideoMAE and ST-MAE [18,74] have showcased the significant benefits of masked modeling in enhancing a model’s capability for FINE-GRAINED temporal understanding. UMT [43] takes this further by introducing an efficient masked alignment technique that yields robust results across single and multimodal video tasks. To augment VideoMamba’s temporal sensitivity and verify its adaptability with text modalities, we adopt a masked alignment approach inspired by UMT. Firstly, VideoMamba is trained from scratch on video data alone, aligning unmasked tokens with those from CLIP-ViT. Subsequently, it is integrated with a text encoder and a cross-modal decoder (i.e., BERT [14]), for pretraining on both image-text and video-text datasets.

[Figure 41]

[Figure 42]

! "#$%-&'()'$$!)'*+ !,*'-( *,#.%'))'+/. 1 2!.$3 ()*44'+/ -*#( +*) ℎ#$4.

Fig. 6: Ablation studies of Self-Distillation and Early Stopping.

It’s important to note the distinction from UMT, which employs multi-layer alignment between the student and teacher models. In contrast, due to VideoMamba’s unique architecture (SSM vs. Transformer), we align only the final outputs. Regarding our masking strategy, we propose different row masking techniques, depicted in Fig. 5, tailored to the B-Mamba block’s preference for continuous tokens. Additionally, we explore attention masking to preserve meaningful adjacency among tokens, leveraging the inherent strengths of the 1D convolution within the B-Mamba block for improved performance.

#### 4 Experiments

##### 4.1 Scaling Up

Dataset and Settings. We first conduct experiments on ImageNet-1K [13], which includes 1.28M training images and 50K validation images across 1,000 categories. For fair comparisons, we follow most of the training strategies proposed in DeiT [75], but adopt weaker data augmentation for the tiny model variant. Furthermore, we adjust the stochastic depth ratio to 0/0.15/0.5 for VideoMamba-Ti/S/M. Our models are trained using the AdamW optimizer paired with a cosine learning rate schedule over 300 epochs. The initial 5 epochs serve as a period for linear warm-up. Default settings for the learning rate, weight decay, and batch size are 1e-3, 0.05, and 1024, respectively. Moreover, we use BFloat16 precision during training to enhance stability without relying on EMA. For the VideoMamba-M model, we employ a pretrained VideoMamba-S model as a “teacher” to guide the training process by aligning the final feature maps through L2 loss. For large resolution (>224) fine-tuning, we use a reduced learning rate (5e-6) and minimal weight decay (1e-8) for 30 epochs.

Effect of Self-Distillation. Fig. 6a reveals that when trained from scratch, VideoMamba-B tends to overfit more easily and underperforms compared to VideoMamba-S, whereas VideoMamba-M achieves similar performances. Fortunately, our self-distillation has shown to be effective in achieving the desired optimization with marginal additional computational cost. To mitigate teacher’s

|Arch.<br><br>|Model<br><br>|iso.|Input Size<br><br>|#Param (M)<br><br>|FLOPs (G)|IN-1K Top-1<br><br>|
|---|---|---|---|---|---|---|
|CNN<br><br>|ConvNeXt-T [53] ConvNeXt-S [53] ConvNeXt-B [53]<br><br>|✗ ✗ ✗|2242 2242 2242<br><br>|29 50 89<br><br>|4.5 8.7<br><br>15.4|82.1 83.1 83.8<br><br>|
|Trans.|SwinT-T [51]<br><br>Swin-S [51] Swin-B [51]<br><br>|✗ ✗ ✗<br><br>|2242 2242 2242|28 50 88<br><br>|4.5 8.7<br><br>15.4<br><br>|81.3 83.0 83.5|
|CNN+ SSM<br><br>|VMamba-T [50] VMamba-S [50] VMamba-B [50]|✗ ✗ ✗<br><br>|2242 2242 2242<br><br>|22 44 75<br><br>|5.6 11.2 18.0|82.2 83.5 83.7<br><br>|
|CNN|ConvNeXt-S [53] ConvNeXt-B [53]<br><br>|✓ ✓|2242 2242<br><br>|22 87<br><br>|4.3 16.9|79.7 82.0<br><br>|
|Trans.|DeiT-Ti [75]<br><br>DeiT-S [75] DeiT-B [75] DeiT-B [75]<br><br>|✓ ✓ ✓ ✓<br><br>|2242 2242 2242 3842|6 22 87 87<br><br>|1.3 4.6<br><br>17.6 55.5<br><br>|72.2 79.8 81.8 83.1|
|SSM|S4ND-ViT-B [58]<br><br>Vim-Ti [91]<br><br>Vim-S [91] VideoMamba-Ti VideoMamba-Ti VideoMamba-Ti VideoMamba-S VideoMamba-S VideoMamba-S VideoMamba-M VideoMamba-M VideoMamba-M<br><br>|✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓<br><br>|2242 2242 2242 2242 4482 5762 2242 4482 5762 2242 4482 5762<br><br>|89 7<br><br>26 7 7 7<br><br>26 26 26<br><br>74<br><br>75 75<br><br><br>|1.1 4.3 1.1 4.3 7.1 4.3<br><br>16.9 28.0 12.7 50.4 83.1<br><br>|80.4 76.1 80.5 76.9 79.3 79.6 81.2 83.2 83.5 82.8 83.8 84.0<br><br>|

- Table 2: Comparison with the state-of-the-art on ImageNet. “iso.” means isotropic architecture without downsampling layers.

potential overdirection, we experimented with early stopping [11] in Fig. 6b, although it did not yield beneficial outcomes. These findings indicate that selfdistillation offers a viable strategy for enhancing the scalability of the Mamba architecture without significant computational overhead.

- Results. Table 2 showcases the results on the ImageNet-1K dataset. Notably, VideoMamba-M outperforms other isotropic architectures by significant margins, achieving a +0.8% improvement over ConvNeXt-B [53] and a +2.0% increase compared to DeiT-B [75], while utilizing fewer parameters. Additionally, VideoMamba-M holds its ground against non-isotropic backbones that leverage hierarchical features for enhanced performance. Given Mamba’s efficiency in processing long sequences, we further enhance performance by increasing the resolution, achieving a top-1 accuracy of 84.0% with only 74M parameters. This remarkable improvement extends to video tasks, as detailed in Section 4.2, underscoring VideoMamba’s effectiveness and scalability.

##### 4.2 Short-term Video Understanding

Datasets and Settings. We evaluate our VideoMamba on the popular scenerelated Kinetics-400 [36] and temporal-related Something-Something V2 [24], the average video lengths of which are 10s and 4s. For supervised pretraining, we finetune those models pretrained on ImageNet-1K with the same training strategy

Extra Input #Param FLOPs K400

Arch. Model iso.

Data Size (M) (G) Top-1 Top-5 Supervised: Those models with extra data are under supervised training.

|CNN|SlowFastR101+NL [19]<br><br>X3D-M [17]<br><br>X3D-XL [17]<br><br>|✗ ✗ ✗| |80×2242 16×2242 16×3122|60 4 20<br><br>|234×3×10<br><br>6×3×10 194×3×10|79.8 93.9<br><br>76.0 92.3<br><br>80.4 94.6<br>|
|---|---|---|---|---|---|---|---|
|Trans.|Swin-T [52] Swin-B [52] Swin-B [52]<br><br>|✗ ✗ ✗<br><br>|IN-1K IN-1K IN-21K|32×2242 32×2242 32×2242<br><br>|28 88 88<br><br>|88×3×4 88×3×4<br><br>282×3×4|78.8 93.6 80.6 94.5 82.7 95.5<br><br>|
|CNN+ Trans.|MViTv1-B [16] MViTv2-S [45]<br><br>UniFormer-S [44] UniFormer-B [44] UniFormer-B [44]<br><br>|✗ ✗ ✗ ✗ ✗|IN-1K IN-1K IN-1K<br><br>|32×2242 16×2242 16×2242 16×2242 32×2242|37 35 21 50 50<br><br>|70×1×5 64×1×5 42×1×4 97×1×4<br><br>259×3×4|80.2 94.4<br><br>81.0 94.6<br><br>80.8 94.7<br><br>82.0 95.1<br><br>83.0 95.4<br>|
|Trans.<br><br>|STAM [63] TimeSformer-L [4] ViViT-L [2] Mformer-HR [59]|✓ ✓ ✓ ✓<br><br>|IN-21K IN-21K IN-21K IN-21K<br><br>|64×2242 96×2242 16×2242 16×3362|121 121 311 311<br><br>|1040×1×1 2380×3×1 3992×3×4 959×3×10|79.2 -<br><br>80.7 94.7<br><br>81.3 94.7<br><br><br>81.1 95.2|
|SSM|VideoMamba-Ti VideoMamba-Ti VideoMamba-Ti VideoMamba-S VideoMamba-S VideoMamba-S VideoMamba-M VideoMamba-M VideoMamba-M<br><br>|✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓<br><br>|IN-1K IN-1K IN-1K IN-1K IN-1K IN-1K IN-1K IN-1K IN-1K<br><br>|16×2242 32×2242 64×3842 16×2242 32×2242 64×3842 16×2242 32×2242 64×3842<br><br>|7 7 7<br><br>26 26 26 74 74 74<br><br>|17×3×4 34×3×4<br><br>202×3×4<br><br>68×3×4 135×3×4 395×3×4 202×3×4 403×3×4<br><br>2368×3×4|78.1 93.5 78.8 93.9<br><br>80.3 94.8<br><br>80.8 94.8<br><br>81.5 95.2<br><br>82.7 95.6<br><br><br>81.9 95.4<br><br>82.4 95.7<br><br>83.3 96.1<br><br><br>|

Self-supervised: For UMT, the CLIP-400M is used in pretrained teacher.

|Trans.<br><br>|BEVT-B800e [81]<br><br>|✗|IN-1K<br><br>|32×2242|88<br><br>|282×3×4<br><br>|81.1 -|
|---|---|---|---|---|---|---|---|
| |ST-MAE-B1600e [18] VideoMAE-S2400e [74] VideoMAE-B1600e [74] UMT-B800e [43]<br><br>|✓ ✓ ✓ ✓|CLIP-400M|16×2242 16×2242 16×2242<br><br>8×2242<br><br>|87 22 87 87<br><br>|180×3×7<br><br>57×3×5 180×3×5 180×3×5|81.3 94.9 79.0 93.8 81.5 95.1 85.7 97.0<br><br>|
|SSM|VideoMamba-M800e VideoMamba-M800e VideoMamba-M800e VideoMamba-M800e<br><br>|✓ ✓ ✓ ✓<br><br>|CLIP-400M CLIP-400M CLIP-400M CLIP-400M<br><br>|8×2242 16×2242 32×2242 64×3842<br><br>|74 74 74 74<br><br>|101×3×4 202×3×4 403×3×4<br><br>2368×3×4|82.0 95.4<br><br>83.4 95.9 83.9 96.2 85.0 96.9<br><br><br>|

- Table 3: Comparison with the state-of-the-art on scene-related Kinetics-400. “iso.” means isotropic architecture without downsampling layers. Masked modeling [43] also works for Mamba, but the inconsistent architecture leads to inferior alignment.

as VideoMAE [74]. Specifically, for VideoMamba-M, the warmup epoch, total epoch, stochastic depth rate, weight decay are set to 5, 50, 0.8, 0.05 for K400, and

- 5, 30, 0.8, 0.05 for SthSth. For the smaller models, all the hyper-parameters are the same unless we decrease the stochastic depth rate and increase the training epochs. Moreover, we linearly scale the base learning rates according to the

batch size, which are 2e−4 · batchsize256 for K400 and 4e−4 · batchsize256 for SthSth. As for self-supervised pretraining, we adopt the training recipe as in UMT [43],

employing CLIP-ViT-B [60] to distill VideoMamba-M over 800 epochs. During fine-tuning, we use similar hyperparameters as mentioned but opt for a small stochastic depth rate and learning rate for both datasets.

- Results. Table 3 and 4 list the results on short-term video datasets. (a) Supervised: Compared with the purely attention-based methods [2,4], our SSM-based VideoMamba-M secures a notable advantage, outperforming ViViT-L [2] by

+2.0% and +3.0% on the scene-related K400 and the temporally-related Sth-

Extra Input #Param FLOPs SSV2

Arch. Model iso.

Data Size (M) (G) Top-1 Top-5 Supervised: Those models with extra data are under supervised training.

|CNN<br><br>|SlowFastR101 [19] CT-NetR50 [41] TDNR50 [79]|✗ ✗ ✗<br><br>|K400 IN-1K IN-1K|32×2242 16×2242 16×2242<br><br>|53 21 26<br><br>|106×3×1 75×1×1 75×1×1<br><br>|63.1 87.6<br>64.5 89.3<br>65.3 91.6<br>|
|---|---|---|---|---|---|---|---|
|Trans.|Swin-B [52]|✗|K400<br><br>|32×2242|89<br><br>|88×3×1|69.6 92.7|
|CNN+ Trans.<br><br>|MViTv1-B [16] MViTv1-B [16] MViTv2-S [45] MViTv2-B [45] UniFormer-S [44] UniFormer-B [44]|✗ ✗ ✗ ✗ ✗ ✗<br><br>|K400 K400 K400 K400<br><br>IN-1K+K400 IN-1K+K400<br><br>|16×2242 32×2242 16×2242 32×2242 16×2242 16×2242<br><br>|37 37 35 51 21 50|71×3×1 170×3×1 65×3×1<br><br>225×3×1 42×3×1 97×3×1<br><br>|64.7 89.2<br><br>67.1 90.8<br><br>68.2 91.4<br><br><br>70.5 92.7 67.7 91.4 70.4 92.8<br><br>|
|Trans.|TimeSformer-HR [4]<br><br>ViViT-L [2]<br><br>Mformer-HR [59]|✓ ✓ ✓<br><br>|IN-21K<br><br>IN-21K+K400 IN-21K+K400|16×2242 16×2242 16×3362<br><br>|121 311 311<br><br>|1703×3×1 3992×3×4 1185×3×1|62.5 -<br><br>65.4 89.8 68.1 91.2<br><br>|
|SSM|VideoMamba-Ti VideoMamba-Ti VideoMamba-Ti VideoMamba-S VideoMamba-S VideoMamba-S VideoMamba-M VideoMamba-M VideoMamba-M<br><br>|✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓<br><br>|IN-1K IN-1K IN-1K IN-1K IN-1K IN-1K IN-1K IN-1K IN-1K<br><br>|8×2242 16×2242 16×2882<br><br>8×2242 16×2242 16×2882<br><br>8×2242 16×2242 16×2882<br><br>|7 7 7<br><br>26 26 26 74 74 74<br><br>|9×3×2 17×3×2 28×3×2 34×3×2 68×3×2<br><br>112×3×2 101×3×4 202×3×4 333×3×4<br><br>|65.1 89.1<br><br>66.0 89.6<br><br><br>66.2 90.0<br><br>66.6 90.4<br><br>67.6 90.9<br><br>68.1 91.2<br><br><br>67.3 91.0<br><br>68.3 91.4 68.4 91.6<br><br><br>|

Self-supervised: For UMT, the CLIP-400M is used in pretrained teacher.

|Trans.<br><br>|BEVT-B800e [81]<br><br>|✗<br><br>|IN-1K+K400|32×2242<br><br>|88<br><br>|321×3×1|70.6 -|
|---|---|---|---|---|---|---|---|
| |VideoMAE-S2400e [74] VideoMAE-B2400e [74] UMT-B800e [43]|✓ ✓ ✓<br><br>|CLIP-400M|16×2242 16×2242<br><br>8×2242|22 87 87<br><br>|57×3×2 180×3×2 180×3×2|66.8 90.3 70.8 92.4 70.8 92.6<br><br>|
|SSM|VideoMamba-M800e VideoMamba-M800e VideoMamba-M800e<br><br>|✓ ✓ ✓<br><br>|CLIP-400M CLIP-400M CLIP-400M<br><br>|8×2242 16×2242 16×2882<br><br>|74 74 74<br><br>|101×3×2 202×3×2 333×3×2<br><br>|70.2 92.6<br><br>71.0 92.7 71.4 92.9<br><br><br>|

- Table 4: Comparison with the state-of-the-art on temporal-related SthSth V2. “iso.” means isotropic architecture without downsampling layers. Masked modeling [43] also works for Mamba, and it performs better than VideoMAE.

SthV2 datasets, respectively. This improvement comes with significantly reduced computational demands and less pretraining data. Furthermore, VideoMamba-M delivers results that are on par with the SOTA UniFormer [44], which skillfully integrates convolution with attention in a non-isotropic structure. (b) Selfsupervised: The performance of VideoMamba under masked pretraining surpasses that of the VideoMAE [74], known for its proficiency in fine-grained action. This achievement underscores the potential of our purely SSM-based model in efficiently and effectively understanding short-term videos, highlighting its suitability for both supervised and self-supervised learning paradigms.

Ablation Studies. Through comprehensive ablation studies detailed in Fig. 7 and Table 5, we explore various aspects of our model. (a) Scan Type: Among all the methods, the spatial-first approach emerges as the most effective, in contrast, the temporal-first strategy is the worst. The superiority of the spatial-first method is attributed to its ability to seamlessly leverage 2D pretrained knowledge by scanning frame by frame. (b) Frame and Resolution: Contrary to findings from ImageNet (see Table 2), higher resolution does not uniformly lead to better performance. Increasing the number of frames consistently enhances

|Type|SSV2|
|---|---|
|SF-Bidirectional TF-Bidirectional ST-Bidirectional v1 ST-Bidirectional v2 Half-SF + Half-TF Half-TF + Half-SF Alternative SF&TF<br><br>|65.1 62.4 63.9 64.2 64.0 64.1 65.1|

66.0

64×224×224

16×224×224

79.5

SSV2Top-1Acc(%)

K400Top-1Acc(%)

32×224×224

79.0

65.8

32×224×224

78.5

16×224×224

65.6

Frame

8×288×288

Resolution

8×384×384

78.0

8×320×320

65.4

8×320×320

8×448×448

77.5

8×384×384

Frame

65.2

Resolution

77.0

8×224×224

8×224×224

2000 4000 6000 8000 10000 12000

2000 3000 4000 5000 6000

Token Number

Token Number

(a) Scan Type. Spatial-First scan is simple yet effective.

(b) Frame & Resolution for K400 and SSV2.

Fig. 7: Ablation studies of scan type, frame and resolution. All the models are fine-tuned from VideoMamba-Ti pretrained on ImageNet.

|Ratio|SSV2|
|---|---|
|50% 65% 80% 90%<br><br>|68.1 68.4 68.5 68.2<br><br>|

|DP<br><br>|SSV2|
|---|---|
|0.1 0.2 0.3 0.4<br><br>|68.0 68.2 68.4 68.5<br><br>|

|Layer<br><br>|SSV2|
|---|---|
|Last 1<br>Last 2 Last 6 Last 6×2<br>|68.5 68.4 68.2 67.7<br><br>|

|Type|SSV2<br><br>|
|---|---|
|Random Tube Clip-Row Frame-Row Attention<br><br>|67.4 66.3 68.2 67.8 68.5<br><br>|

(b) Alignment Layer.

(c) Mask Ratio.

(d) Droppath.

(a) Mask Type.

- Table 5: Ablation studies of masked pretraining. We adopt CLIP-ViT-B [60] as a teacher to distill VideoMamba-M for 200 epochs.

results on the K400 dataset. However, this is not the case with SthSthV2, possibly due to the brief duration of its videos, which may not accommodate longer inputs effectively. (c) Masked Pretraining: Our findings reveal that row masking, being particularly compatible with 1D convolution, outperforms commonly used random and tube masking. Clip-row masking excels owing to its higher degree of randomness. Moreover, attention masking stands out as the most efficient by favoring the preservation of adjacent meaningful content. Aligning solely the model’s final output proves most effective, likely due to architectural differences. Lastly, an optimal masking ratio (80%) combined with stronger regularization significantly benefits VideoMamba during masked pretraining.

##### 4.3 Long-term Video Understanding

Datasets and Settings. We rigorously assess VideoMamba’s proficiency in processing long-term videos by leveraging three comprehensive datasets, i.e., Breakfast [37], COIN [71] and Long-form Video Understanding (LVU [84]) benchmark. Specifically, Breakfast comprises 1,712 videos, encapsulating 10 intricate cooking activities over 77 hours. COIN features 11,827 videos across 180 unique procedural tasks, with an average duration of 2.36 minutes. The LVU benchmark includes approximately 30K movie clips, lasting between 1 to 3 minutes, and encompasses nine tasks across 3 primary categories: content understanding, metadata prediction, and user engagement. For the regression task among these, we evaluate using mean-squared error, while for the classification tasks, accuracy is the metric of choice. In contrast to prior studies [35,47] that rely on features derived from pretrained video models, such as Swin-B [51] trained on Kinetics-600, our method employs end-to-end training as detailed in Section 4.2. Additionally, for fair comparisons, we fine-tune our models pretrained on K400.

|Method<br><br>|e2e<br><br>|Backbone|Neck Type<br><br>|Pretraining Dataset<br><br>|BF COIN Top-1 Top-1|
|---|---|---|---|---|---|
|Timeception [32] VideoGraph [33] GHRM [90] Distant Supervision [47] ViS4mer [35]|✗ ✗ ✗ ✗ ✗<br><br>|3D-ResNet I3D I3D TimeSformer Swin-B<br><br>|Conv. Conv.+Atten. Graph Conv.. Atten. w/ KB SSM|IN-1K+K400 IN-1K+K400 IN-1K+K400<br><br>IN-21K+HTM IN-21K+K600<br><br>|71.3 69.5 75.5 -<br><br>89.9 90.0 88.2 88.4<br><br>|
|Turbof32 [29] Turbof32 [29] VideoMambaf32 VideoMambaf64 VideoMambaf32 VideoMambaf64 VideoMambaf32 VideoMambaf64 VideoMambaf32 VideoMambaf64<br><br>|✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓<br><br>|VideoMAE-B VideoMAE-B VideoMamba-Ti VideoMamba-Ti VideoMamba-S VideoMamba-S VideoMamba-M VideoMamba-M VideoMamba-M† VideoMamba-M†<br><br>| |K400 K400+HTM-AA K400 K400 K400 K400 K400 K400 K400 K400|86.8 82.3 91.3 87.5<br><br>94.3 86.2<br><br>94.3 87.0 95.3 88.4 97.4 88.7<br><br><br>94.8 88.3<br><br>95.8 89.5 97.9 89.6<br><br>96.9 90.4<br><br><br>|

- Table 6: Comparison with the state-of-the-art on Breakfast and COIN. “e2e” means end-to-end methods without exhausting feature extraction. “†” marks the backbone with masked pretraining.

|Method|e2e<br><br>|Backbone|Content(↑) Rel. Speak Scene|Metadata(↑) Dir. Genre Wtr. Year<br><br>|User(↓) Like View|
|---|---|---|---|---|---|
|VideoBERT [69] Object Trans. [84] LST [35] Performer [35] Orthoformer [35] ViS4mer [35]|✗ ✗ ✗ ✗ ✗ ✗<br><br>|S3D ResNet ViT-L ViT-L ViT-L ViT-L|52.80 37.90 54.90<br><br>53.10 39.40 56.90<br><br><br>52.38 37.31 62.79<br><br>50.00 38.80 60.46<br><br>50.00 39.30 66.27<br><br><br>57.14 40.79 67.44<br><br>|47.30 51.90 38.50 36.10 51.20 54.60 34.50 39.10 56.07 52.70 42.26 39.16 58.87 49.45 48.21 41.25 55.14 55.79 47.02 43.35 62.61 54.71 48.80 44.75<br><br>|0.32 4.46 0.23 3.55 0.31 3.83 0.31 3.93 0.29 3.86 0.26 3.63<br><br>|
|VideoMambaf32|✓|VM-Ti|62.50 40.43 70.37<br><br>|67.29 65.24 52.98 48.23<br><br>|0.26 2.90<br><br>|

- Table 7: Comparison with the state-of-the-art on LVU. “e2e” means end-toend methods without exhausting feature extraction. “Rel.”, “Dir.” and “Wtr.” refers to “Relation”, “Director” and “Writer”, respectively.

Results. As illustrated in Figure 1, the linear complexity of VideoMamba makes it well-suited for end-to-end training with long-duration videos. The comparisons in Tables 6 and 7 highlight VideoMamba’s simplicity and effectiveness against traditional feature-based methods [35,47] on these tasks. It yields significant performance improvements, achieving SOTA results even with smaller model sizes. For example, VideoMamba-Ti shows a notable increase of +6.1% over ViS4mer using Swin-B features and a +3.0% uplift against Turbo’s multi-modality alignment approach [29]. Notably, the results underscore the positive impact of the scaling model and frame numbers for long-term tasks. In the diverse and challenging set of nine tasks presented by LVU, our VideoMamba-Ti, fine-tuned in an end-to-end manner, delivers outstanding or comparable results to current SOTA methods. These outcomes not only highlight VideoMamba’s effectiveness but also its great potential for future long-video comprehension.

##### 4.4 Multi-modality Video Understanding

Datasets and Settings. Following UMT [43], we utilize WebVid-2M [3] videotext pairs and CC3M [64] image-text pairs for joint pretraining with four objectives: vision-text contrastive learning [3], vision-text matching [40], masked

|Method<br><br>|BB|#P<br><br>|MSRVTT @1 @5 @10|DiDeMo @1 @5 @10<br><br>|ANet @1 @5 @10<br><br>|LSMDC @1 @5 @10<br><br>|MSVD @1 @5 @10|
|---|---|---|---|---|---|---|---|
|Singularity [38] Frozen [3] ALPRO [39] BridgeFormer [23] UMT [43]|Swin<br><br>ViT ViT ViT ViT<br><br>|5M 5M 5M 5M 5M|28.4 50.2 59.5<br><br>18.7 39.5 51.6 24.1 44.7 55.4 26.0 46.4 56.4<br><br>29.6 52.8 61.9<br><br><br>|36.9 61.1 69.3 20.2 46.4 58.5 23.8 47.3 57.9 25.6 50.6 61.1 33.4 58.3 67.0<br><br>|30.8 55.9 66.3<br><br>- - -<br>- - -<br>- - -<br><br><br>28.3 53.0 64.2|- - -<br><br>- - -<br><br>- - -<br><br><br>12.2 25.9 32.2 16.8 30.5 37.6<br><br>|- - -<br>- - -<br>- - -<br><br><br>43.6 74.9 84.9 36.2 65.7 76.1|
|VideoMamba|VM|5M<br><br>|32.0 53.0 63.8<br><br>|36.6 61.7 70.3<br><br>|35.9 61.1 72.3<br><br>|18.0 36.1 43.4<br><br>|38.0 68.6 79.0<br><br>|
|VideoCLIP [85]<br><br>|S3D|136M|10.4 22.2 30.0<br><br>|16.6 46.9 -|- - -<br><br>|- - -<br><br>|- - -|
|VIOLET [21] Singularity [38] OmniVL [38] UMT [43] UMT [43] CLIP4Clip [55] InternVideo [83]|Swin Swin<br><br>ViT ViT ViT ViT ViT<br><br>|138M 17M 17M 17M 25M<br><br>400M 640M|25.9 49.5 59.7<br><br>34.0 56.7 66.7<br><br>34.6 58.4 66.6<br><br>35.5 59.3 68.6<br><br><br>35.2 57.8 66.0<br><br><br>30.6 54.4 64.3 40.0 65.3 74.1<br><br>|23.5 49.8 59.8 37.1 61.7 69.9 33.3 58.7 68.5 41.9 66.7 75.0 41.2 65.4 74.9 - - 31.5 57.6 68.2|- - -<br><br>30.6 55.6 66.9<br><br>- - 33.8 59.1 70.4 35.5 60.6 71.8 - - -<br><br>30.7 57.4 70.2<br>|- - -<br><br>- - -<br><br>- - -<br><br><br>18.1 33.1 42.2<br><br>19.1 33.4 42.2<br><br><br>13.6 27.9 35.5 17.6 32.4 40.2<br><br>|- - -<br>- - -<br>- - -<br><br><br>41.4 70.6 80.1<br>42.3 71.7 80.8 36.2 63.8 73.5<br><br>43.4 69.9 79.1<br>|
|VideoMamba VideoMamba<br><br>|VM VM<br><br>|17M 25M<br><br>|34.7 58.9 68.0<br><br>35.6 58.1 69.5<br><br><br>|42.0 67.3 76.8<br><br>43.1 68.1 77.7<br><br><br>|40.1 65.7 76.1<br><br>41.0 67.5 77.8<br><br><br>|18.4 35.3 43.0 20.4 37.1 45.7<br><br>|40.3 70.0 79.7 42.6 71.6 81.2<br><br>|

- Table 8: Zero-shot text-to-video retrieval on MSRVTT, DiDeMo, AcitivityNet, LSMDC, and MSVD. “BB” means the visual backbone. “#P” refers to the number of pretraining pairs. Models pretrained with large-scale pairs are noted in gray.

language modeling [14] and unmasked token alignment [43]. Initially, we mask 50% image tokens and 80% video tokens, conducting pretraining across 8 frames for 10 epochs. Given Mamba’s sensitivity to positional information, an additional unmasked tuning phase is carried out for one epoch to refine its comprehension further. For evaluation, we undertake zero-shot video-text retrieval tasks across five prominent benchmarks, including MSRVTT [86], DiDeMo [1], ActivityNet [31], LSMDC [61], and MSVD [10].

Results. As indicated in Table 8, under the same pretraining corpus and similar training strategies, our VideoMamba achieves superior zero-shot video retrieval performances to UMT [43] based on ViT [15]. It underscores Mamba’s comparable efficiency and scalability to the ViT in handling multi-modal video tasks. Notably, for datasets featuring longer video lengths (e.g., ANet and DiDeMo) and more complex scenarios (e.g., LSMDC), VideoMamba demonstrates a significant improvement. This demonstrates Mamba’s aptitude for the demands of cross-modality alignment even in challenging multimodal contexts.

#### 5 Conclusion

In this paper, we propose VideoMamba, a purely SSM-based model for efficient video understanding. Our extensive experiments demonstrate its scalability in the visual domain, sensitivity for short-term action recognition, superiority in long-term video understanding and compatibility with other modalities. We hope it can pave the way for future model design for long-video comprehension.

Limitations. Due to resource constraints, we have not yet fully validated the scalability of VideoMamba, such as extending VideoMamba to larger sizes (e.g., VideoMamba-g), incorporating additional modalities (e.g., audio), and integrating with large language models for hour-level video understanding. Despite these limitations, our findings confirm VideoMamba’s promising potential and we plan to conduct thorough explorations of its capabilities in the future.

#### References

- 1. Anne Hendricks, L., Wang, O., Shechtman, E., Sivic, J., Darrell, T., Russell, B.: Localizing moments in video with natural language. In: ICCV (2017) 14
- 2. Arnab, A., Dehghani, M., Heigold, G., Sun, C., Lučić, M., Schmid, C.: Vivit: A video vision transformer. In: ICCV (2021) 1, 2, 4, 5, 6, 10, 11, 21
- 3. Bain, M., Nagrani, A., Varol, G., Zisserman, A.: Frozen in time: A joint video and image encoder for end-to-end retrieval. In: ICCV (2021) 13, 14
- 4. Bertasius, G., Wang, H., Torresani, L.: Is space-time attention all you need for video understanding? In: ICML (2021) 1, 2, 4, 5, 6, 10, 11, 21
- 5. Brooks, T., Peebles, B., Holmes, C., DePue, W., Guo, Y., Jing, L., Schnurr, D., Taylor, J., Luhman, T., Luhman, E., Ng, C., Wang, R., Ramesh, A.: Video generation models as world simulators (2024), https://openai.com/research/videogeneration-models-as-world-simulators 1
- 6. Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J.D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., et al.: Language models are few-shot learners. In: NeurIPS (2020) 3
- 7. Carreira, J., Noland, E., Banki-Horvath, A., Hillier, C., Zisserman, A.: A short note about kinetics-600. ArXiv abs/1808.01340 (2018) 3
- 8. Carreira, J., Noland, E., Hillier, C., Zisserman, A.: A short note on the kinetics-700 human action dataset. ArXiv abs/1907.06987 (2019) 3
- 9. Carreira, J., Zisserman, A.: Quo vadis, action recognition? a new model and the kinetics dataset. In: CVPR (2017) 1, 4
- 10. Chen, D.L., Dolan, W.B.: Collecting highly parallel data for paraphrase evaluation. In: ACL (2011) 3, 14
- 11. Cho, J.H., Hariharan, B.: On the efficacy of knowledge distillation. In: ICCV (2019) 9
- 12. Das, P., Xu, C., Doell, R.F., Corso, J.J.: A thousand frames in just a few words: Lingual description of videos through latent topics and sparse object stitching. In: CVPR (2013) 3
- 13. Deng, J., Dong, W., Socher, R., Li, L.J., Li, K., Fei-Fei, L.: Imagenet: A large-scale hierarchical image database. In: CVPR (2009) 8
- 14. Devlin, J., Chang, M.W., Lee, K., Toutanova, K.: Bert: Pre-training of deep bidirectional transformers for language understanding. ArXiv abs/1810.04805 (2018) 7, 14
- 15. Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., Houlsby, N.: An image is worth 16x16 words: Transformers for image recognition at scale. In: ICLR (2021) 2, 5, 7, 14
- 16. Fan, H., Xiong, B., Mangalam, K., Li, Y., Yan, Z., Malik, J., Feichtenhofer, C.: Multiscale vision transformers. In: ICCV (2021) 10, 11
- 17. Feichtenhofer, C.: X3d: Expanding architectures for efficient video recognition. In: CVPR (2020) 4, 10
- 18. Feichtenhofer, C., Fan, H., Li, Y., He, K.: Masked autoencoders as spatiotemporal learners. NeurIPS (2022) 7, 10, 21
- 19. Feichtenhofer, C., Fan, H., Malik, J., He, K.: Slowfast networks for video recognition. In: ICCV (2019) 1, 4, 10, 11
- 20. Fu, D.Y., Dao, T., Saab, K.K., Thomas, A.W., Rudra, A., Ré, C.: Hungry hungry hippos: Towards language modeling with state space models. In: ICLR (2023) 3

- 21. Fu, T.J., Li, L., Gan, Z., Lin, K., Wang, W.Y., Wang, L., Liu, Z.: Violet: Endto-end video-language transformers with masked visual-token modeling. ArXiv abs/2111.12681 (2021) 14
- 22. Gao, J., Sun, C., Yang, Z., Nevatia, R.: Tall: Temporal activity localization via language query. In: ICCV (2017) 3
- 23. Ge, Y., Ge, Y., Liu, X., Li, D., Shan, Y., Qie, X., Luo, P.: Bridging video-text retrieval with multiple choice questions. In: CVP (2022) 14
- 24. Goyal, R., Kahou, S.E., Michalski, V., Materzynska, J., Westphal, S., Kim, H., Haenel, V., Fründ, I., Yianilos, P., Mueller-Freitag, M., Hoppe, F., Thurau, C., Bax, I., Memisevic, R.: The “something something” video database for learning and evaluating visual common sense. In: ICCV (2017) 3, 9
- 25. Gu, A., Dao, T.: Mamba: Linear-time sequence modeling with selective state spaces. ArXiv abs/2312.00752 (2023) 2, 3, 4, 5, 7
- 26. Gu, A., Goel, K., Ré, C.: Efficiently modeling long sequences with structured state spaces. In: ICLR (2022) 1, 3
- 27. Gu, C., Sun, C., Vijayanarasimhan, S., Pantofaru, C., Ross, D.A., Toderici, G., Li, Y., Ricco, S., Sukthankar, R., Schmid, C., Malik, J.: Ava: A video dataset of spatio-temporally localized atomic visual actions. CVPR (2017) 3
- 28. Guo, H., Li, J., Dai, T., Ouyang, Z., Ren, X., Xia, S.T.: Mambair: A simple baseline for image restoration with state-space model. ArXiv abs/2402.15648 (2024) 3
- 29. Han, T., Xie, W., Zisserman, A.: Turbo training with token dropout. In: BMVC

(2022) 13

- 30. He, X., Cao, K., Yan, K., Li, R., Xie, C., Zhang, J., Zhou, M.: Pan-mamba: Effective pan-sharpening with state space model. ArXiv abs/2402.12192 (2024) 3
- 31. Heilbron, F.C., Escorcia, V., Ghanem, B., Niebles, J.C.: Activitynet: A large-scale video benchmark for human activity understanding. In: CVPR (2015) 3, 14
- 32. Hussein, N., Gavves, E., Smeulders, A.W.M.: Timeception for complex action recognition. In: CVPR (2019) 13
- 33. Hussein, N., Gavves, E., Smeulders, A.W.M.: Videograph: Recognizing minuteslong human activities in videos. ArXiv abs/1905.05143 (2019) 13
- 34. Idrees, H., Zamir, A.R., Jiang, Y.G., Gorban, A., Laptev, I., Sukthankar, R., Shah, M.: The thumos challenge on action recognition for videos “in the wild”. Computer Vision and Image Understanding (2017) 3
- 35. Islam, M.M., Bertasius, G.: Long movie clip classification with state-space video models. In: ECCV (2022) 2, 3, 12, 13
- 36. Kay, W., Carreira, J., Simonyan, K., Zhang, B., Hillier, C., Vijayanarasimhan, S., Viola, F., Green, T., Back, T., Natsev, A., Suleyman, M., Zisserman, A.: The kinetics human action video dataset. ArXiv abs/1705.06950 (2017) 3, 9
- 37. Kuehne, H., Arslan, A., Serre, T.: The language of actions: Recovering the syntax and semantics of goal-directed human activities. In: CVPR (2014) 3, 12
- 38. Lei, J., Berg, T.L., Bansal, M.: Revealing single frame bias for video-and-language learning. ArXiv abs/2206.03428 (2022) 14
- 39. Li, D., Li, J., Li, H., Niebles, J.C., Hoi, S.C.: Align and prompt: Video-and-language pre-training with entity prompts. In: CVPR (2022) 14
- 40. Li, J., Selvaraju, R., Gotmare, A., Joty, S., Xiong, C., Hoi, S.C.H.: Align before fuse: Vision and language representation learning with momentum distillation. In: NeurIPS (2021) 13
- 41. Li, K., Li, X., Wang, Y., Wang, J., Qiao, Y.: Ct-net: Channel tensorization network for video classification. In: ICLR (2020) 11

- 42. Li, K., Wang, Y., He, Y., Li, Y., Wang, Y., Wang, L., Qiao, Y.: Uniformerv2: Spatiotemporal learning by arming image vits with video uniformer. ArXiv abs/2211.09552 (2022) 4
- 43. Li, K., Wang, Y., Li, Y., Wang, Y., He, Y., Wang, L., Qiao, Y.: Unmasked teacher: Towards training-efficient video foundation models. In: ICCV (2023) 6, 7, 10, 11, 13, 14, 21
- 44. Li, K., Wang, Y., Peng, G., Song, G., Liu, Y., Li, H., Qiao, Y.: Uniformer: Unified transformer for efficient spatial-temporal representation learning. In: ICLR (2022) 1, 4, 10, 11
- 45. Li, Y., Wu, C., Fan, H., Mangalam, K., Xiong, B., Malik, J., Feichtenhofer, C.: Improved multiscale vision transformers for classification and detection. ArXiv abs/2112.01526 (2021) 10, 11
- 46. Liang, D., Zhou, X., Wang, X., Zhu, X., Xu, W., Zou, Z., Ye, X., Bai, X.: Pointmamba: A simple state space model for point cloud analysis. ArXiv abs/2402.10739 (2024) 3
- 47. Lin, X., Petroni, F., Bertasius, G., Rohrbach, M., Chang, S.F., Torresani, L.: Learning to recognize procedural activities with distant supervision. CVPR (2022) 2, 12, 13
- 48. Liu, H., Yan, W., Zaharia, M., Abbeel, P.: World model on million-length video and language with ringattention. ArXiv abs/2402.08268 (2024) 1
- 49. Liu, Y., Wang, L., Wang, Y., Ma, X., Qiao, Y.: Fineaction: A fine-grained video dataset for temporal action localization. TIP (2022) 3
- 50. Liu, Y., Tian, Y., Zhao, Y., Yu, H., Xie, L., Wang, Y., Ye, Q., Liu, Y.: Vmamba: Visual state space model. ArXiv abs/2401.10166 (2024) 2, 3, 6, 7, 9
- 51. Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., Guo, B.: Swin transformer: Hierarchical vision transformer using shifted windows. In: ICCV (2021) 9, 12
- 52. Liu, Z., Ning, J., Cao, Y., Wei, Y., Zhang, Z., Lin, S., Hu, H.: Video swin transformer. In: CVPR (2022) 2, 4, 10, 11
- 53. Liu, Z., Mao, H., Wu, C., Feichtenhofer, C., Darrell, T., Xie, S.: A convnet for the 2020s. In: CVPR (2022) 9
- 54. Lu, J., Batra, D., Parikh, D., Lee, S.: Vilbert: Pretraining task-agnostic visiolinguistic representations for vision-and-language tasks. NeurIPS (2019) 3
- 55. Luo, H., Ji, L., Zhong, M., Chen, Y., Lei, W., Duan, N., Li, T.: Clip4clip: An empirical study of clip for end to end video clip retrieval and captioning. Neurocomputing (2022) 14
- 56. Mehta, H., Gupta, A., Cutkosky, A., Neyshabur, B.: Long range language modeling via gated state spaces. ArXiv abs/2206.13947 (2022) 3
- 57. Miech, A., Zhukov, D., Alayrac, J.B., Tapaswi, M., Laptev, I., Sivic, J.: Howto100m: Learning a text-video embedding by watching hundred million narrated video clips. In: ICCV (2019) 3
- 58. Nguyen, E., Goel, K., Gu, A., Downs, G.W., Shah, P., Dao, T., Baccus, S.A., Ré, C.: S4nd: Modeling images and videos as multidimensional signals with state spaces. In: NeurIPS (2022) 9
- 59. Patrick, M., Campbell, D., Asano, Y., Misra, I., Metze, F., Feichtenhofer, C., Vedaldi, A., Henriques, J.F.: Keeping your eye on the ball: Trajectory attention in video transformers. In: NeurIPS (2021) 4, 10, 11, 21
- 60. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., Sutskever, I.: Learning transferable visual models from natural language supervision. In: ICML (2021) 10, 12

- 61. Rohrbach, A., Torabi, A., Rohrbach, M., Tandon, N., Pal, C.J., Larochelle, H., Courville, A.C., Schiele, B.: Movie description. International Journal of Computer Vision (2016) 14
- 62. Shao, D., Zhao, Y., Dai, B., Lin, D.: Finegym: A hierarchical video dataset for fine-grained action understanding. CVPR (2020) 3
- 63. Sharir, G., Noy, A., Zelnik-Manor, L.: An image is worth 16x16 words, what is a video worth? ArXiv abs/2103.13915 (2021) 4, 10, 21
- 64. Sharma, P., Ding, N., Goodman, S., Soricut, R.: Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning. In: ACL

(2018) 13

- 65. Simonyan, K., Zisserman, A.: Two-stream convolutional networks for action recognition in videos. NeurIPS (2014) 4
- 66. Smith, J.T., Warrington, A., Linderman, S.W.: Simplified state space layers for sequence modeling. In: ICLR (2023) 3
- 67. Soomro, K., Zamir, A.R., Shah, M.: Ucf101: A dataset of 101 human actions classes from videos in the wild. arXiv preprint arXiv:1212.0402 (2012) 3
- 68. Su, J., Lu, Y., Pan, S., Wen, B., Liu, Y.: Roformer: Enhanced transformer with rotary position embedding. ArXiv abs/2104.09864 (2021) 6
- 69. Sun, C., Myers, A., Vondrick, C., Murphy, K., Schmid, C.: Videobert: A joint model for video and language representation learning. In: ICCV (2019) 13
- 70. Sun, Y., Dong, L., Huang, S., Ma, S., Xia, Y., Xue, J., Wang, J., Wei, F.: Retentive network: A successor to transformer for large language models. ArXiv abs/2307.08621 (2023) 1
- 71. Tang, Y., Ding, D., Rao, Y., Zheng, Y., Zhang, D., Zhao, L., Lu, J., Zhou, J.: Coin: A large-scale dataset for comprehensive instructional video analysis. In: CVPR

(2019) 3, 12

- 72. Team, G.: Gemini: A family of highly capable multimodal models. ArXiv abs/2312.11805 (2023) 1
- 73. Team, R.: Rwkv: Reinventing rnns for the transformer era. In: EMNLP (2023) 1
- 74. Tong, Z., Song, Y., Wang, J., Wang, L.: VideoMAE: Masked autoencoders are data-efficient learners for self-supervised video pre-training. In: NeurIPS (2022) 6, 7, 10, 11, 21
- 75. Touvron, H., Cord, M., Douze, M., Massa, F., Sablayrolles, A., J’egou, H.: Training data-efficient image transformers & distillation through attention. In: ICML (2021) 2, 8, 9
- 76. Tran, D., Bourdev, L.D., Fergus, R., Torresani, L., Paluri, M.: Learning spatiotemporal features with 3d convolutional networks. In: IEEE International Conference on Computer Vision (2015) 1, 4
- 77. Tran, D., xiu Wang, H., Torresani, L., Ray, J., LeCun, Y., Paluri, M.: A closer look at spatiotemporal convolutions for action recognition. In: CVPR (2018) 4
- 78. Wang, C., Tsepa, O., Ma, J., Wang, B.: Graph-mamba: Towards long-range graph sequence modeling with selective state spaces. ArXiv abs/2402.00789 (2024) 3
- 79. Wang, L., Tong, Z., Ji, B., Wu, G.: TDN: Temporal difference networks for efficient action recognition. In: CVPR (2021) 11
- 80. Wang, L., Xiong, Y., Wang, Z., Qiao, Y., Lin, D., Tang, X., Gool, L.V.: Temporal segment networks: Towards good practices for deep action recognition. In: ECCV

(2016) 4, 20

- 81. Wang, R., Chen, D., Wu, Z., Chen, Y., Dai, X., Liu, M., Jiang, Y.G., Zhou, L., Yuan, L.: Bevt: Bert pretraining of video transformers. CVPR (2022) 10, 11

- 82. Wang, Y., He, Y., Li, Y., Li, K., Yu, J., Ma, X.J., Chen, X., Wang, Y., Luo, P., Liu, Z., Wang, Y., Wang, L., Qiao, Y.: Internvid: A large-scale video-text dataset for multimodal understanding and generation. ArXiv (2023) 3
- 83. Wang, Y., Li, K., Li, Y., He, Y., Huang, B., Zhao, Z., Zhang, H., Xu, J., Liu, Y., Wang, Z., Xing, S., Chen, G., Pan, J., Yu, J., Wang, Y., Wang, L., Qiao, Y.: Internvideo: General video foundation models via generative and discriminative learning. ArXiv abs/2212.03191 (2022) 4, 14
- 84. Wu, C.Y., Krahenbuhl, P.: Towards long-form video understanding. In: CVPR

(2021) 3, 12, 13

- 85. Xu, H., Ghosh, G., Huang, P.Y., Okhonko, D., Aghajanyan, A., Metze, F., Zettlemoyer, L., Feichtenhofer, C.: Videoclip: Contrastive pre-training for zero-shot video-text understanding. ArXiv abs/2109.14084 (2021) 14
- 86. Xu, J., Mei, T., Yao, T., Rui, Y.: Msr-vtt: A large video description dataset for bridging video and language. In: CVPR (2016) 3, 14
- 87. Yang, Y., Xing, Z., Zhu, L.: Vivim: a video vision mamba for medical video object segmentation. ArXiv abs/2401.14168 (2024) 3
- 88. Yu, Z., Xu, D., Yu, J., Yu, T., Zhao, Z., Zhuang, Y., Tao, D.: Activitynet-qa: A dataset for understanding complex web videos via question answering. In: AAAI

(2019) 3

- 89. Zhang, D.J., Li, K., Wang, Y., Chen, Y., Chandra, S., Qiao, Y., Liu, L., Shou, M.Z.: Morphmlp: An efficient mlp-like backbone for spatial-temporal representation learning. In: ECCV (2022) 4
- 90. Zhou, J., Lin, K.Y., Li, H., Zheng, W.: Graph-based high-order relation modeling for long-term action recognition. CVPR (2021) 13
- 91. Zhu, L., Liao, B., Zhang, Q., Wang, X., Liu, W., Wang, X.: Vision mamba: Efficient visual representation learning with bidirectional state space model. ArXiv abs/2401.09417 (2024) 2, 3, 5, 6, 9
- 92. Zhuang, S., Li, K., Chen, X., Wang, Y., Liu, Z., Qiao, Y., Wang, Y.: Vlogger: Make your dream a vlog. ArXiv abs/2401.09414 (2024) 1

[Figure 43]

## VideoMamba: State Space Model for Efficient Video Understanding

### Appendix

https://github.com/OpenGVLab/VideoMamba

#### A More Results

In Table I, we present additional results on the Kinetics-400 dataset. These results clearly demonstrate that our SSM-based model outperforms all previous attention-based methods. We observe consistent performance improvements with increasing resolution and frame count.

#### B More Implementation Details

##### B.1 Training Details

We sparsely sample frames from the raw videos as in TSN [80] for all the datasets. Table II details the masked pretraining hyperparameters. For the unmasked multi-modality pretraining, we load the pretrained model and train it for an additional epoch with a learning rate of 8e-5. Moreover, Tables III, IV, V, and VI show the training details for the different datasets used for fine-tuning.

##### B.2 Dataset Descriptions

We show the statistics of multi-modality datasets in Table VII, and singlemodality datasets in Table VIII.

Extra Input #Param FLOPs K400

Arch. Model iso.

Data Size (M) (G) Top-1 Top-5 Supervised: Those models with extra data are under supervised training.

|Trans.<br><br>|STAM [63] TimeSformer-L [4] ViViT-L [2] Mformer-HR [59]|✓ ✓ ✓ ✓<br><br>|IN-21K IN-21K IN-21K IN-21K|64×2242 96×2242 16×2242 16×3362<br><br>|121 121 311 311<br><br>|1040×1×1 2380×3×1 3992×3×4 959×3×10<br><br>|79.2 -<br>80.7 94.7<br>81.3 94.7 81.1 95.2<br><br><br>|
|---|---|---|---|---|---|---|---|
|SSM|VideoMamba-Ti VideoMamba-Ti VideoMamba-Ti VideoMamba-Ti VideoMamba-Ti VideoMamba-S VideoMamba-S VideoMamba-S VideoMamba-S VideoMamba-S VideoMamba-M VideoMamba-M VideoMamba-M VideoMamba-M VideoMamba-M<br><br>|✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓<br><br>|IN-1K IN-1K IN-1K IN-1K IN-1K IN-1K IN-1K IN-1K IN-1K IN-1K IN-1K IN-1K IN-1K IN-1K IN-1K<br><br>|8×2242 16×2242 32×2242 64×2242 64×3842<br><br>8×2242 16×2242 32×2242 64×2242 64×3842<br><br>8×2242 16×2242 32×2242 64×2242 64×3842<br><br>|7 7 7 7 7<br><br>26 26 26 26 26 74 74 74 74 74<br><br>|9×3×4 17×3×4 34×3×4 69×3×4<br><br>202×3×4 34×3×4 68×3×4<br><br>135×3×4 271×3×4 395×3×4 101×3×4 202×3×4 403×3×4 806×3×4<br><br>2368×3×4|76.9 92.9<br><br>78.1 93.5<br><br>78.8 93.9<br><br>79.6 94.2<br><br>80.3 94.8<br><br>79.3 94.2 80.8 94.8 81.5 95.2<br><br>81.8 95.3<br><br>82.7 95.6<br><br><br>82.8 96.0<br><br>83.3 96.1<br><br><br>80.6 94.6<br><br>81.9 95.4<br><br>82.4 95.7<br><br><br>|

Self-supervised: For UMT, the CLIP-400M is used in pretrained teacher.

|Trans.|ST-MAE-B1600e [18]<br><br>VideoMAE-S2400e [74] VideoMAE-B1600e [74] UMT-B800e [43]<br><br>|✓ ✓ ✓ ✓<br><br>|CLIP-400M|16×2242 16×2242 16×2242<br><br>8×2242<br><br>|87 22 87 87|180×3×7<br><br>57×3×5 180×3×5 180×3×5<br><br>|81.3 94.9 79.0 93.8 81.5 95.1 85.7 97.0|
|---|---|---|---|---|---|---|---|
|SSM|VideoMamba-M800e VideoMamba-M800e VideoMamba-M800e VideoMamba-M800e VideoMamba-M800e<br><br>|✓ ✓ ✓ ✓ ✓<br><br>|CLIP-400M CLIP-400M CLIP-400M CLIP-400M CLIP-400M<br><br>|8×2242 16×2242 32×2242 64×2242 64×3842<br><br>|74 74 74 74 74<br><br>|101×3×4 202×3×4 403×3×4 806×3×4<br><br>2368×3×4|82.0 95.4<br><br>83.4 95.9<br><br><br>83.9 96.2<br><br>84.3 96.6<br><br>85.0 96.9<br><br><br>|

Table I: More results on scene-related Kinetics-400. “iso.” means isotropic architecture without downsampling layers.

|config<br><br>|Single-Modality SthSthV2 K400<br><br>|Multi-Modality 5M & 17M & 25M|
|---|---|---|
|optimizer optimizer momentum weight decay learning rate schedule learning rate minimal learning rate batch size warmup epochs total epochs mask ratio input frame drop path flip augmentation augmentation<br><br>|AdamW β1, β2=0.9, 0.95<br><br>0.05<br><br>cosine decay<br><br>1.2e-3 1e-5 2048<br><br><br>40 800 80% 8 0.4<br><br>no yes MultiScaleCrop[0.66, 0.75, 0.875, 1]<br><br>|AdamW β1, β2=0.9, 0.999 0.05 cosine decay 4e-4 4e-6 2048 I, 2048 V 1 10 50% I, 80% V, 50% T 8 0.25 yes MultiScaleCrop[0.5, 1]|

###### Table II: Masked pre-training settings. I-image, V-video, T-text.

|config<br><br>|224×224 448×448 512×512|
|---|---|
|optimizer optimizer momentum weight decay learning rate schedule base learning rate minimal learning rate base batch size repeated augmentation warmup epochs total epochs drop path label smoothing cutmix augmentation<br><br>|AdamW β1, β2=0.9, 0.999<br><br>0.1(Ti), 0.05(S,M) 1e-8 1e-8 cosine decay<br><br>5e-4 5e-6 5e-6 1e-5 5e-6 5e-6<br><br>512 no(Ti), yes(S,M)<br><br>5(Ti,S), 30(M) 5 2 300 30 10<br><br>0(Ti), 0.15(S), 0.5(M) 0.1 1.0 RandAug(7, 0.25)(Ti), RandAug(9, 0.5)(S,M)|

Table III: Training settings for ImageNet-1K.

|config|224×224 384×384<br><br>|
|---|---|
|optimizer optimizer momentum weight decay learning rate schedule base learning rate minimal learning rate base batch size repeated augmentation warmup epochs total epochs drop path layer-wise lr decay flip augmentation label smoothing cutmix augmentation|AdamW β1, β2=0.9, 0.999<br><br>0.1(Ti), 0.05(S,M,M†) 1e-8<br><br>cosine decay<br><br>4e-4(Ti,S), 2e-4(M), 1e-4(M†) 5e-6<br><br>1e-6<br><br>256<br><br>2<br><br><br>5 2<br><br>70(Ti), 50(S,M), 45(M†) 10<br><br>0.1(Ti), 0.35(S), 0.8(M), 0.4(M†) 0.75(S,M,M†), 0.8(M†) yes 0.1 1.0 RandAug(7, 0.25)(Ti), RandAug(9, 0.5)(S,M,M†)<br><br>|

- Table IV: Training settings for Kinetics-400. “†” means masked pretraining.

|config<br><br>|224×224 288×288|
|---|---|
|optimizer optimizer momentum weight decay learning rate schedule base learning rate minimal learning rate base batch size repeated augmentation warmup epochs total epochs drop path layer-wise lr decay flip augmentation label smoothing cutmix augmentation<br><br>|AdamW β1, β2=0.9, 0.999<br><br>0.1(Ti), 0.05(S,M,M†) 1e-8 cosine decay 4e-4(Ti,S,M) 1e-4(M†) 5e-6<br><br>1e-6 256<br>2<br><br><br>5 2 35(Ti), 30(S,M,M†) 10<br><br>0.1(Ti), 0.35(S), 0.8(M), 0.4(M†) 0.75(S,M,M†), 0.8(M†) no 0.1 1.0 RandAug(7, 0.25)(Ti), RandAug(9, 0.5)(S,M,M†)|

###### Table V: Training settings for SthSthV2. “†” means masked pretraining.

|config|BreakFast & LVU COIN<br><br>|
|---|---|
|optimizer optimizer momentum weight decay learning rate schedule base learning rate minimal learning rate base batch size repeated augmentation warmup epochs total epochs drop path layer-wise lr decay flip augmentation label smoothing cutmix augmentation<br><br>|AdamW β1, β2=0.9, 0.999 0.1(Ti), 0.05(S,M,M†) cosine decay 2e-4<br><br>1e-6 256<br>2 5<br><br><br>70(Ti), 50(S,M), 45(M†) 40(Ti), 35(S), 30(M,M†) 0.1(Ti), 0.35(S), 0.8(M), 0.4(M†) 0.75(S,M,M†), 0.8(M†) yes 0.1 1.0 RandAug(7, 0.25)(Ti), RandAug(9, 0.5)(S,M,M†)|

Table VI: Training settings for Breaskfast, COIN and LVU. “†” means masked pretraining. We directly sample the frames from the raw video sparsely.

Dataset #image/video #text Type COCO 113K 567K image Visual Genome 100K 768K image SBU Captions 860K 860K image CC3M 2.88M 2.88M image CC12M 11.00M 11.00M image WebVid-2M 2.49M 2.49M video WebVid-10M 10.73M 10.73M video

5M corpus = CC3M+WebVid-2M 5.37M 5.37M video+image 17M corpus = 5M+COCO+VG+SBU+CC12M 17.44M 18.57M video+image 25M corpus = 17M+WebVid-10M−WebVid-2M 25.68M 26.81M video+image

###### Table VII: Statistics of multi-modality datasets.

#video #text Avg Video Train Val Test Train Val Test Length (s)

Dataset

Image Classification ImageNet-1K 1,281,167 50,000 100,000 - - - -

###### Short-term Action Recognition

Kinetics-400 240,436 19,787 - - - - 10 Something-Something V2 168,913 24,777 - - - - 4

###### Long-term Action Recognition

Breakfast 1,577 - 410 - - - 137 COIN 9,026 - 2,796 - - - 142 LVU 7,619 1,666 1,551 - - - 134

Relation 138 49 41 - - - 127 Speak 871 196 188 - - - 133 Scene 514 107 81 - - - 132 Director 680 163 107 - - - 137 Genre 2807 569 584 - - - 130 Writer 748 174 168 - - - 142 Year 725 163 141 - - - 133 Like 658 142 139 - - - 159 View 478 103 102 - - - 112

###### Video-Text Retrieval

MSRVTT 7,010 - 1,000 140,200 - 1,000 15 DiDeMo 8,496 1,094 1,036 8,496 1,094 1,036 29 ActivityNet 10,009 4,917 - 10,009 4,917 - 180 LSMDC 101,055 - 1,000 101,055 - 1,000 5 MSVD 1,200 100 670 1,200 100 670 15

###### Table VIII: Statistics of single-modality datasets.

