# arXiv:2502.11663v1[cs.CV]17Feb2025

## MaskGWM: A Generalizable Driving World Model with Video Mask Reconstruction

Jingcheng Ni, Yuxin Guo, Yichen Liu, Rui Chen, Lewei Lu, Zehuan Wu SenseTime Research Project page: https://github.com/SenseTime-FVG/OpenDWM

#### Abstract

World models that forecast environmental changes from actions are vital for autonomous driving models with strong generalization. The prevailing driving world model mainly build on video prediction model. Although these models can produce high-fidelity video sequences with advanced diffusion-based generator, they are constrained by their predictive duration and overall generalization capabilities. In this paper, we explore to solve this problem by combining generation loss with MAE-style feature-level context learning. In particular, we instantiate this target with three key design: (1) A more scalable Diffusion Transformer (DiT) structure trained with extra mask construction task. (2) we devise diffusion-related mask tokens to deal with the fuzzy relations between mask reconstruction and generative diffusion process. (3) we extend mask construction task to spatial-temporal domain by utilizing row-wise mask for shifted self-attention rather than masked self-attention in MAE. Then, we adopt a row-wise cross-view module to align with this mask design. Based on above improvement, we propose MaskGWM: a Generalizable driving World Model embodied with Video Mask reconstruction. Our model contains two variants: MaskGWM-long, focusing on long-horizon prediction, and MaskGWM-mview, dedicated to multi-view generation. Comprehensive experiments on standard benchmarks validate the effectiveness of the proposed method, which contain normal validation of Nuscene dataset, long-horizon rollout of OpenDV-2K dataset and zero-shot validation of Waymo dataset. Quantitative metrics on these datasets show our method notably improving state-of-the-art driving world model.

#### 1. Introduction

As an pivotal application of artificial intelligence, autonomous driving technologies, which require comprehending the surrounding environment and executing correct actions, have achieved significant advancements following the

emergence of various learning models [20, 24] with scalability. However, the challenge of limited generalization to complex and varied scenarios remains unresolved for state-of-the-art methods [25]. For example, perception may encounter performance drops [42] in cases like weather changes, scene variations and motion blur. A promising solution for this problem is the use of world models, which directly predict environment changes under different actions. These models facilitate unraveling the complexities of data distributions and craft intricate regular patterns like human perception system [23].

Recently, advanced methods [12, 19, 39–41, 44, 47] developed world models through diffusion-based generation task, capitalizing on the rapid development of advanced image generation systems [1, 7, 33]. Despite generating highfidelity results, these approaches still struggle with longhorizon prediction and zero-shot generalization. To address this, GenAD [44] attempts to conduct training on largescale OpenDV-2K [44] dataset with carefully-designed temporal modules, while VISTA [12] further introduces explicit re-weighted generation loss on structural and moving areas. However, two problems still exist in building a generalizable world model for autonomous driving. First, the combination of large-scale training dataset with more scalable transformer architectures is still under exploration. Second, one fundamental question remains unanswered: Is diffusion-based generation sufficient to build a generalizable world model? Since diffusion loss targets at iterative de-noising, the learning of visual semantics may not be straightforward. For example, MaskDiT [48] has shown diffusion models are complementary to well-known self-supervised methods [14], benefiting both convergence speed and generation quality.

Based on these insights into the diffusion pipeline and data for our driving world model, we design a new model, dubbed as MaskGWM, aiming at improving the fidelity, generalizability and long-time series prediction of the existing methods. Additionally, our model can also generate multi-view cases, by incorporating a multi-view module. We adopt Diffusion Transformer (DiT) as our back-

Model Setup

Method

Data scale Framework Multi-view Traget

Drive-WM [40] 5h Unet √ Diff DiVE [21] 5h DiT √ Diff GenAD [44] 1740h Unet × Diff Vista [12] 1740h Unet × Diff

MaskGWM(Ours) 1740h DiT √ Diff+MR

(a) Real-world multi-view driving world models.

Spatial Context

[Figure 1]

Information

exchange

[Figure 2]

Same mask positions

Temporal Context

[Figure 3]

Information exchange

[Figure 4]

Different mask positions

(b) Different context for mask reconstruction.

Figure 1. (a). MaskGWM improve fidelity and generalization from web-scale dataset, scalable DiT architecture and Mask Reconstruction (MR) target. (b) proposed MR apply a two branch structure for spatial context (scene objects) and temporal context (object motions)

[Figure 5]

|0s|1s|3s|7s|12s|
|---|---|---|---|---|

[Figure 6]

[Figure 7]

Time

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

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

Zero Shot Transfer Long-Horizon Rollout on Web-Scale scenes Multi-View Generation

Figure 2. Our model facilitates zero-shot generation, consistent long-horizon prediction and multi-view video generation.

bone, which is more scalable and could take the information from a variety of datasets. Moreover, we introduce the mask reconstruction as a complementary task for generation. Several impactful works [14, 43] have demonstrated that masked autoencoder is a powerful self-supervised method for representation learning from large-scale data and it is also extended to some diffusion methods [11, 48, 50] as an additional supervision to improve the models’ performance. Additionally, the features obtained by self-supervised learning is more contextually meaningful [5], which can be used as an auxiliary supervision to further improve the generation quality [19] However, integrating existing mask reconstruction for image generation into driving world models is not straightforward. There are still two questions to answer: (1) How can we enhance the synergy between diffusion model and mask reconstruction? Though the Mask reconstruction improves the contextual reasoning ability, this task contradicts with diffusion steps which include high noise ratio obscuring the feature details. (2) What kind of mask strategy should we use for video data? Different from image generation, the prediction of driving future requires an understanding of not only the objects within the scene but also their dynamic movements.

Therefore, we develop several special designs to ad-

dress the aforementioned issues: (1) We make use of the mask tokens to improve the synergy between mask reconstruction and diffusion models. Specifically, we propose a diffusion-related mask tokens (Sec.3.2) to initialized the invisible patches after DiT encoder. This special mask token can balance the learning of global and local features. (2) We design a novel two-branch mask reconstruction strategy. For spatial modeling, we use a mask shared across all frames and reconstruct invisible tokens via spatial transformer, this mask strategy is similar to some video mask modeling methods [36, 38]. For temporal modeling, we introduce a frame-specific mask and recover masked tokens via temporal transformer. Unlike the spatial branch, we directly link the unaligned tokens on temporal dimension after masking, which can be seen as a shift augmentation restricted by a new-proposed row-wise policy. We find this temporal branch achieves both masked patches prediction in the temporal context and a reduction in training costs.

In summary, our main contributions are:

- • We propose MaskGWM, a generalizable DiT-based driving world model capable of forecasting long-term futures across web-scale scenes.
- • We introduce mask reconstruction as a complementary task for diffusion-based world model.

• Comprehensive experiments on nuScene, OpenDV-2K and Waymo datasets demonstrate the superior video quality and robust generalization capabilities across extended time spans and different viewpoints.

#### 2. Related Works

##### 2.1. World Models

World models aim to infer the dynamic environment and ego state from past observations for accurate future predictions and planning. Most studies achieve world understanding by enabling the model to generate realistic videos that align with physical principles. In autonomous driving area, world models primarily focus on generating controllable real-world driving scenarios. GAIA-1 [19] employs an autoregressive transformer to predict tokens based on past state and then leverages a diffusion decoder to generate high-quality videos. In contrast, DriveDreamer [39] directly use diffusion model to represent complex environments and generate driving videos from multi-modal inputs. Drive-WM [40] extends to consistent and controllable multi-view video generation, exploring its application in end-to-end planning. GenAD [44] enhances generalization capability of world models by training on large-scale datasets and Vista [12] achieves further improvements by introducing attention on structure and dynamic area.

##### 2.2. Diffusion Models with Self-supervised Learning

Recently, diffusion-based methods [13, 15, 18, 32, 46] have become the mainstream of image and video generation. One important advancement in this field is Diffusion Transformer(DiT) [30]. Due to its better scalability and lower computational cost, DiT has been successfully applied in various diffusion models, achieving state-of-the-art results [7, 8, 21, 49].

On the other hand, masking strategies from selfsupervised learning have been effectively applied to enhance generative models. With the development of DiT, research has focused on migrating this self-supervised approach to diffusion-based models. Initial works like MDT [11] modify DiT blocks to an asymmetric masking diffusion transformer architecture, where the encoder handles unmasked tokens only and a side-interpolater is introduced to recover the latnet to the original shape. Following MDT [11], MaskDiT [48] simply utilizes a learnable token to fill in the masked places. SD-DiT [50], noticing the training-inference discrepancy and fuzzy relations between mask strategy and diffusion process, introduce a novel masking DiT with self-supervised discrimination. Despite their succuss, none of them applied the masking diffusion to the video generation models. Our work make this attempt on driving world model by applying different

design of spatial context and temporal context, which focus on scene objects and nuance motion separately.

#### 3. Method

Fig. 3 illustrates an overview of the proposed pipeline. MaskGWM builds upon Stable Diffusion 3 (SD3) [7], which is a well-studied DiT-based Text-to-Image (T2I) generation model, and introduce additional spatial and temporal blocks to extract the cross-view and cross-frame information. Moreover, we deploy a mask reconstruction module during training to improve the performance of our model. In this section, we first briefly review our DiT-based driving world model in Sec. 3.1. Then, Sec. 3.2 details the pipeline for mask reconstruction and introduces novel diffusion-related mask tokens designed to enhance the synergy between diffusion generation and mask reconstruction. Following this, Sec. 3.3 describes the extension of mask reconstruction to the temporal dimension. Finally, in Sec. 3.4, we describe the details of our cross-view module.

##### 3.1. Preliminaries

Diffusion Models. A mutli-view video sampled from a video dataset pdata can be represented as x0 ∼ pdata, where x0 ∈ RT×K×C×H×W is a sequence of T frames with view K, height H and width W. We first transform x0 into video tokens z0 = P(Θ(x0)) ∈ RT×K×C×Hˆ×Wˆ via latent encoder Θ and patch encoding P. MaskGWM applies Rectified Flow [7, 26, 27] to model the generation process. Specifically, given a standard normal distribution ϵ ∼ N(0,I), Rectified Flow defines the intermediate noisy state as zτ = (1 − τ)z0 + τϵ, where τ ∈ [0,1] is the diffusion timestep. The training target of Rectified Flow adopts the v-prediction, defined as:

0,ϵ∼N(0,I),τ ∥Gθ(zτ,τ,c,M) − (z0 − ϵ)∥22 ,

L = Ez

(1) where c is the condition, M is a binary mask for mask reconstruction and Gθ is the DiT model.

Temporal Modeling. To facilitate temporal context learning, we attach a temporal transformer block after each 2D spatial transformer block, following common practice in video generation models [2, 29]. During the forward process, to standardize the inputs to the different self-attention layers in the transformer blocks, we reshape the video latent representation to (TK)(HˆW)Cˆ for spatial self-attention and to (KHˆW)TCˆ for temporal self-attention. Additionally, we introduce reference frames according to video DiT models [9, 29]. During diffusion process, diffusion timestep τ of the reference frames is always set as 0, while the following frames to be predicted are embedded with regular timestep. Unified Action Conditioning. Following VISTA, we provide nuanced control over low-level actions including angle, speed, trajectory and goal point, combining with high-level

[Figure 30]

|Decoder|
|---|

Encoder

Text condition

[Figure 31]

DiT-Block

TokenReconstruction

[Figure 32]

Daytime, city road

[Figure 33]

TemporalTB

TokenMask

SpatialTB

ViewTB

DiT-Block

Action

𝑇𝐾𝑊𝐻𝜌𝐶෡෡

𝐾𝜌𝑊𝐻𝐶𝑇෡෡

𝐻𝜌𝑇𝐶𝑊𝐾෢෡

Generative

Loss

Speed Trajectory

Angle

View

[Figure 34]

[Figure 35]

visible invisible

[Figure 36]

[Figure 37]

|if 𝑀 = 𝑀𝑠𝑝𝑎𝑡𝑖𝑎𝑙| | |
|---|---|---|
| |Spatial<br><br>Transformer| |

Time

if 𝑀 = 𝑀𝑠𝑝𝑎𝑡𝑖𝑎𝑙 if 𝑀 = 𝑀𝑡𝑖𝑚𝑒

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

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

|Temporal<br><br>Transformer|
|---|

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

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

|if 𝑀 = 𝑀𝑡𝑖𝑚𝑒|
|---|

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

|Only for Training|
|---|

Transformer Block Reshape & Froward

Mask Token 𝑚τ

Equal Mask

TB

- Figure 3. Overview of the MaskGWM. We propose mask reconstruction containing token mask and token reconstruction as a complementary task for training dring world model. Token Mask: we randomly sample tokens by temporal-shared Mspatial and temporal-unshared Mtime, specialized for spatial and temporal modeling. Token Reconstruction: we fill invisible tokens by diffusion-related mask tokens (Sec.3.2) and recover features by a two-branch transformer. Moreover, we introduce a row-wise mask strategy (Sec.3.3) for temporal branch. ρ = 1 − r is used for simplicity in encoder.

command capabilities. We construct action embedding by concatenating the Fourier embeddings [35] of all actions. Subsequently, these action embeddings are projected and added to the key and value features of the cross-attention layers in temporal transformer blocks.

##### 3.2. Diffusion-related Mask Reconstruction

Motivated by previous works [48, 50], Masked Image Modeling (MIM) [14, 43] with mask reconstruction object has been adopted to diffusion-based generation model and achieve improvement on training efficiency and local contextual perception. However, these methods fail to consider the influence of diffusion process, which incorporate noise schedule and complex training target. Therefore, we propose this novel pipeline integrated with mask reconstruction and then introduce our diffusion-related mask tokens for compatibility with diffusion process, which can reduce the effect of noise during diffusion process.

Mask Reconstruction. During the training phase, the DiT backbone is asymmetrically divided into an encoder E and a decoder D for mask reconstruction and includes two additional processing steps. In the encoding stage, MaskGWM produces a token mask by random sampling a

binary mask M ∈ RT×K×1×H×W, given the video latent zτ at timestep τ. Similar to [11, 50], M is only use to partition zτ into visible patch tokens zτv = zτ ⊙M and invisible patch tokens zτiv = zτ ⊙(1−M). In the decoding stage, an extra token reconstruction module is introduced to handle dropped invisible patches. Mask tokens mτ, representing invisible patches, are infilled at the positions of the dropped tokens. Then, a transformer block F is utilized to provide contextual awareness from visible patches. In details,

Gθ(zτ,M) = D(F(E(zτv) ⊙ M + mτ ⊙ (1 − M))), (2)

where τ,c in Eq.(1) are ignored for simplicity. Note that mask reconstruction is skipped for inference in which all tokens are visible, equivalent to Gθ(zτ) = D(E(zτ)) = D(E(zτv + zτiv)). In practice, invisible patches zτiv are directly dropped during the encoding of training to enable memory and speed benefits.

Diffusion-related Mask Tokens. SD-DiT [50] describes a fuzzy relationship between generation process and mask reconstruction. Concretely, mask reconstruction focuses on context reasoning while generation diffusion process aims to model the translations between real and fake distributions. From the viewpoint of diffusion model, the mask

M = 𝑀𝑠𝑝𝑎𝑡𝑖𝑎𝑙, Temporal Transformer Block M = 𝑀time, Temporal Transformer Block

M = 𝑀෡t𝑖𝑚𝑒, Temporal Transformer Block

Attention Mask

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

Shifted

|Self-Attention|
|---|

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

[Figure 185]

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

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

Each row mask exactly 1 token

- Figure 4. The comparison of different mask types and attention operations for temporal transformer block with mask reconstruction task. Attention mask is only applied when M = Mtime

reconstruction for represent learning can be regard as z0prediction task, whereas rectified flow employs v-prediction (pred z0 − ϵ). Therefore, simply combining these two distinct objectives may not lead to good performance of the diffusion model, as also demonstrated by MaskDiT [48].

Existing works either instantiate mask tokens as learnable parameters [11, 14] p or directly take noisy tokens zτ as input [50]. As discussed above, these mask tokens cannot balance these two targets due to the absence of explicit information for acquiring ϵ. We bridge this gap by introducing fm(ϵ) into the mask token, where fm is a small network for encoding noise ϵ. Given that ϵ is explicitly provided, it is easier to recover the original mask reconstruction target for representation learning within diffusion pipeline. Other than explicitly alignment for two prediction tasks, we further take τ into consideration. Overall, we define our mask token with learnable parameters p as:

mτ = (1 − τ)fm(ϵ) + τp. (3)

Based on our experiments in Section 4.4, we designed this mask token accordingly. When τ is large, the generation is performed at a high-noise level, and fine-grained image details are unknown. We hypothesize that the learnable parameters can estimate an average distribution under specific conditions (e.g., text) and assist in guiding the prediction direction when appearance details are lacking [45]. Conversely, on a low-noise level, mask reconstruction encourages the model to be attentive to local details of visible patches. Therefore, the model can leverage the visible information to recover the patches filled with noise (by fm(ϵ)).

##### 3.3. Mask Reconstruction Strategy

Temporal and Spatial Mask. Previous methods [36, 38] extend MIM to video domain by sharing the random mask across time, where the mask M = Mspatial = [M1,M2...,MT] satisfy Mi = Mj when i ̸= j. Despite

the effectiveness on understanding tasks in the aforementioned methods, this masking strategy may not be suitable for temporal modeling on driving video prediction. To incorporate temporal learning, we introduce a temporal unshared mask Mtime satisfy Mi ̸= Mj when i ̸= j. Then, we specialize Mspatial for spatial modeling and consider MIM with Mtime for temporal modeling. Then, we make tasks synergy by devising a two-branch transformer block:

- Fs if M = Mspatial
- Ft if M = Mtime

, (4)

F =

where M ∈ {Mspatial,Mtime}, Fs is a spatial transformer block and Ft is a temporal transformer block.

Row-wise Approximation. The most straightforward strategy for masking video data is to directly apply a random mask Mtime on each frame like [36, 38], as shown in the middle column of Fig. 4. However, this design cause tokens on masked positions should be masked rather than directly dropped, since temporal self-attention require all entries have the same sequence length and apply an attention mask to control sparsity. As a result, this masked temporal selfattention requires a 3D attention mask to skip masked tokens, which leads to additional computational cost and precludes the direct application of optimization operators like FlashAttention [6]. To address the aforementioned issues, we employ shifted temporal self-attention for Mtime. As illustrated in the right column of Fig. 4, we randomly mask the same number of tokens for each row. Similar to the spatial branch, all invisible tokens are dropped, and the visible tokens are directly connected. Consequently, this operation can be regarded as a shift in temporal self-attention, which adheres to the core idea of Masked Reconstruction (MR): masked tokens are invisible and are predicted from context during the decoding stage. Additionally, rearranging the visible tokens row by row ensures the relevance of information in the temporal attention block during the encoding

stage. In experiments, we find this design improves not only the training speed but also the generation metrics, especially on larger mask ratio r. Since this row-wise shifting allows nearby tokens to fill in the blanks of masked tokens, We analysis this phenomenon by token filling makes all tokens on temporal axis are retained and minor shift can facilitate the temporal block in context reasoning.

To formulate this row-wise temporal mask, we define the mask ratio as r. To generate the mask for a Hˆ × Wˆ image latent at frame t, we randomly generate Hˆ one-dimensional mask, each having rWˆ zero values, and concatenate them to formulate the final mask for this image, denoted by Mtime. We use Mtime to replace Mtime in our training.

##### 3.4. Mask Reconstruction for Cross-View

Our method can be extended to generate multi-view driving videos by introducing view transformer blocks. We propose a cross-view row-wise self-attention mechanism, which concatenates video features horizontally and computes attention across multi-view features on the same row. Specifically, given a feature tensor of shape TKHˆWˆ , we apply masking, resulting in a tensor of shape TKCHˆ[(1 − r)Wˆ ]. We then reshape it into size (THˆ)[K(1 − r)Wˆ ]C.

The key insights of the proposed cross-view row-wise attention are twofold: First, since the vertical context can be modeled by spatial transformer blocks, row-wise feature exchange provides sufficient receptive field to extract multi-view information. Second, the proposed row-wise masking for reconstruction can also be utilized as data augmentation, since KrWˆ tokens of each row are randomly dropped. Because the proposed masked modeling task focuses on spatio-temporal modeling, we do not apply mask reconstruction along the view dimension.

#### 4. Experiments 4.1. Setup

Datasets. We conduct comprehensive experiments on a single-view dataset OpenDV-2K [44] and two multi-views datasets nuScenes [3] and Waymo [34]. We follow the official splits to divide the training and validation sets.

Evaluation. The quality of the generated images and videos are assessed using the Frechet Inception Distance (FID) [16] for images and the Frechet Video Distance (FVD) [37] for videos. For fair comparison with previous works, we apply different evaluation settings for singleview model and multi-view model. For single view model, we align the evaluation setting of VISTA [12]. For multiview model, we adopt the setting of Drive-WM [40]. Please refer to Appendix.7.3 for more details. To assess generalization ability, we evaluate zero-shot performance on Waymo validation set, using 600 videos for FVD and 15K frames for FID.

Multiview

Futurelayout

FID↓ FVD↓ DriveDreamer [39] √

Method

× 14.9 340.8 MagicDrive [10] √ √ 19.1 218.1

DiVE [21] √ √ - 94.6 DriveDreamer-2 [47] √

× 11.2 55.7 Drive-WM [40] √

× 15.8 122.7 MaskGWM-mview √

× 8.9 65.4

DriveGAN [22] × × 73.4 502.3

GenAD [44] × × 15.4 184.0 Vista [12] × × 6.9 89.4 MaskGWM-long × × 5.6 92.5

MaskGWM-long† × × 4.0 59.4

- Table 1. Performance comparison with state-of-the-art methods on nuScene Dataset. The varying shades of gray indicate our multiview metric following Drive-WM and single-view metric following Vista for more fair comparison. Future layout refers to the availability of layout information for future time steps. † denotes training without action.

Method FVD↓ FID↓

VISTA [12]† 176.56 9.76 MaskGWM-long 118.83 9.55

- Table 2. Zero-shot metrics on 600 Waymo validation samples. † denotes inference by official checkpoint.

##### 4.2. Training Scheme

Our model is initialized with SD3 [7] medium checkpoint with 2B parameters. There are three stages for our training:

- Stage 1 for pre-training on large-scale OpenDV-2K dataset,
- Stage 2 for single-view model MaskGWM-long and Stage
- 3 for multi-view model MaskGWM-mview. For all stages, we resize the original images to 512×288, masking ratio r is set to 0.25.

- Stage 1. Following VISTA [12], we first pre-train our model on OpenDV-2K dataset. As our model starts from image backbone, we first train our model using single frame videos with batch size 768 for 18K iterations and then we train our temporal blocks with 16/20/24 frame videos and batch size 64. In particular, temporal blocks is initialized with zero and the training of temporal blocks takes 24K iterations in this step. Afterwards, we insert the zero-initialized reconstruction blocks F and train with masking strategy for extra 20K iterations. The reason for varying frame length during training is that the frame numbers for single-view and multi-view models are different.
- Stage 2 for MaskGWM-long. We devised MaskGWMlong to align with VISTA, where the frame length T is set to 25 and cross-view blocks are skipped. We also follow

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

[Figure 281]

###### Time: 0-12s

- Figure 5. Long-horizon prediction results of MaskGWM. Our model is capable of forecasting long video sequences with stability, devoid of collapse or blurring issues.

[Figure 282]

- Figure 6. Comparison of Long-horizon FVD metric on OpenDV2K validation set. Our method demonstrates superior performance in terms of both value and growth rate.

ting yields significantly improved results, with a FID of 4.0 and a FVD of 59.4. We present these results for reference. For multi-view cases, similar improvement can also be observed, with 8.9 FID and 65.4 FVD. It is worth noting that our model directly predict multi-view future without the requirement of future layouts, unlike those methods [21] deviating from video prediction task. Furthermore, our method represents a pioneering effort in extending a generalizable single-view model, trained on OpenDV-2K, to the domain of multi-view models.

Generalization ability. In Tab.2, we assess the generalization capability of our approach on the Waymo dataset, which is excluded from our training datasets. We conducted inference using the official checkpoint of VISTA [12]. The results indicate that our method attains superior FVD while maintaining comparable FID, thereby demonstrating the generalization ability of our method.

Long-horizon prediction. Fig.6 illustrates the comparison of long-time prediction with VISTA. Due to the computational cost of generating long videos, both FID and FVD metrics are calculated on 300 videos randomly sampled from the OpenDV-2K validation set. The slope of FVD curve is significantly lower than VISTA, showing the less degradation. Qualitative results are illustrated in Fig.5 and appendix.

the stage 2 of VISTA’s collaborative training, in which data were sampled equally from nuScene and OpenDV-2K, and the action module is zero-initialized and trained.

Stage 3 for MaskGWM-mview. Based on the welltrained MaskGWM-long, we enable multi-view ability of MaskGWM-mview by adding the cross-view blocks and train it on nuScene dataset for 6K steps and frame length T is reduced to 8.

##### 4.3. Comparison

Generation Quality. Tab.1 presents the quantitative comparison. In single-view generation, we achieve a remarkable FID of 5.6 and an FVD of 92.5, surpassing the previous state-of-the-art approaches [12, 44] that are also trained on web-scale dataset. It is worth noting that there is a discrepancy between the training and evaluation phases in Vista, which integrates action during training but excludes it during evaluation. To address this, we also experiment with an aligned setting that drops action information for both training and evaluation. As indicated in Table 1, this aligned set-

##### 4.4. Ablation Studies

We conduct comprehensive ablation study to verfiy the performance of every component in MaskGWM. Following the setting of GenAD for effective experiments, most ablation studies are conducted on stage 1 after mask reconstruction is inserted and the metrics are reported on the validation set of OpenDV-2K dataset with 3000 video clips for FVD and 18000 frames for FID. For the ablation of cross-view transformer blocks, we use multi-view nuScene metrics.

Effect of mask tokens. As indicated in Tab.3a, we make comparisons across different designs for mask tokens

mτ τ range contra. FVD↓ FID↓ p [0,1] 126.71 7.35 p [0.5,1] 120.35 7.12

###### fm(ϵ) [0,1] 116.85 6.40 fm(ϵ) [0,0.5] 109.87 5.92

zτ [0,1] √ 113.26 6.32 ours [0,1] 105.52 5.69

(a) Different design of mτ.

###### Mtime row shift att. Mspatial r FVD↓ FID↓

√√√ √√ √ √ 0.250.250.250.250 136.55142.68143.07121.38116.73 10.2810.7510.397.365.92 √ √ √ √ 0.25 105.52 5.69

(b) Ablations of two-branch mask reconstruction.

- Table 3. Ablations of the components in our mask reconstruction. “contra” stands for contrastive loss, “row” denotes whether Mtime satisfies above row-wise generation strategy (Mtime = Mˆ time) and ”att.” refers to self-attention

row&shift att r FVD↓ Time↓ 0.1 133.24 0.368d

√√ 0.250.250.1 142.68123.70121.38 0.352d0.357d0.329d (a) Different mask ratio r.

two-branch layers FVD↓ FID↓

√ 11 121.37105.52 5.855.69 √ 22 127.91107.34 6.035.60 (b) Different design of F.

att dim ratio r FVD↓ FID↓

KW 0 65.9 9.2 KW 0.25 65.4 8.9

K 0.25 71.5 8.7 KHW 0.25 64.7 8.5

(c) Ablation of cross-view block.

- Table 4. Ablations of the impact of mask ratio r, different view transformer block designs and the effect of mask reconstruction on convergence. ”share” stands for applying one shared model for Fs and Ft. Time indicate the training time for 10k steps.

mτ. To analysis the influence of diffusion timestep τ, we test only applying mask reconstruction on certain timestep range. We find learnable parameters p shown better results on high noisy level and proposed noise embedding fm(ϵ) perform better on low noisy level. This demonstrates the merit of noise embedding, which can recover the original mask reconstruction target for represent learning by explicitly giving ϵ, achieving better result on generation steps with local details. In addition, we also try the mask tokens used in SD-DiT [50], combining with extra contrastive [4] loss. The experiment shows our proposed diffusion-related mask tokens achieve best performance via combining the advantage of p and fm(ϵ).

Effect of mask reconstruction. In Tab.3b and Tab.4a, we explore the effectiveness of proposed mask reconstruction in spatial-temporal domain. According to Tab.4a, we find that our row-wise time mask with shift attention mechanism assists a lot the model’s convergence, while larger mask ratio can also make positive effect on the results. Thus, this setting is adopted in the ablation study on the mask strategy M. As shown in Tab.3b, combining Mspatial and Mtime can significantly improve the generation quality of the whole videos and each single images. We hypothesis this is because the temporal modeling is more sensitive to the dropout ratio than spatial counterpart (Please see more details on Appendix.6.1). When shifted temporal self-attention is applied with a row-wise mask, all tokens on temporal axis are retained and experience only minor spatial shifts. Whereas invisible tokens are skipped without shift temporal self-attention, which results in a discrepancy be-

tween the number of tokens used in training and inference. Moreover, the training speed is improved since invisible tokens dropped. From Tab.3b, we can see our two-branch mask reconstruction achieve best results, showing the effectiveness of introducing mask reconstruction on temporal context.

Effect of cross-view module. As shown in Tab.4c, we make comparisons across different types of cross-view modeling. We find that introducing mask reconstruction on stage-3 training also yields favorable results. This indicates that randomly masking certain tokens along the view-row dimension is not detrimental and can even enhance the final results. This is different from temporal counterpart that requires shift-attention to avoiding tokens dropping.

Furthermore, the experiment shows that proposed attention on dimensions KW outperforms view-attention on K used in previous methods [39, 40]. For view attention on KHW, despite the minor improvement, the computation complexity explodes significantly. Consequently, this design is not adopted in our experiments.

Effect of two-branch token reconstruction. We validate the effectiveness of two-branch transformer reconstruction structure for token reconstruction on Tab.4b. We remove two-branch structure by applying sequential spatialtemporal transformer blocks, which is shared by Fs and Ft. We find the two-branch structure produces better results, especially on FVD. Unshared design forces the model to reconstruct masked features by corresponding context, leading to better spatial and temporal modeling for different conditions.

#### 5. Conclusion

We introduce MaskGWM, the first DiT-based driving world model trained on web-scale datasets with masking strategy during training. By introducing a novel video dual-branch mask reconstruction, our model excels in both numeric metrics on fidelity and generation ability. Additionally, our mask policy also accelerate the training process and read memory consumption. Our extensive experiments showcase MaskGWM achieves the state-of-the-art performance on generation quality on nuScene, zero-shot ability on Waymo and longtime prediction ability on OpenDV-2K. These results furtehr indicates that our method can serve as a greate training programs to enable long driving video prediction.

#### References

- [1] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023. 1, 3
- [2] Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22563–22575, 2023. 3
- [3] Holger Caesar, Varun Bankiti, Alex H Lang, Sourabh Vora, Venice Erin Liong, Qiang Xu, Anush Krishnan, Yu Pan, Giancarlo Baldan, and Oscar Beijbom. nuscenes: A multimodal dataset for autonomous driving. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11621–11631, 2020. 6
- [4] Mathilde Caron, Hugo Touvron, Ishan Misra, Herv´e J´egou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9650–9660, 2021. 8
- [5] Mathilde Caron, Hugo Touvron, Ishan Misra, Herv´e J´egou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In Proceedings of the International Conference on Computer Vision (ICCV), 2021. 2
- [6] Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher R´e. Flashattention: Fast and memory-efficient exact attention with io-awareness. Advances in Neural Information Processing Systems, 35:16344–16359, 2022. 5
- [7] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first International Conference on Machine Learning, 2024. 1, 3, 6
- [8] Kunyu Feng, Yue Ma, Bingyuan Wang, Chenyang Qi, Haozhe Chen, Qifeng Chen, and Zeyu Wang. Dit4edit:

- Diffusion transformer for image editing. arXiv preprint arXiv:2411.03286, 2024. 3
- [9] Peng Gao, Le Zhuo, Ziyi Lin, Chris Liu, Junsong Chen, Ruoyi Du, Enze Xie, Xu Luo, Longtian Qiu, Yuhang Zhang, et al. Lumina-t2x: Transforming text into any modality, resolution, and duration via flow-based large diffusion transformers. arXiv preprint arXiv:2405.05945, 2024. 3
- [10] Ruiyuan Gao, Kai Chen, Enze Xie, Lanqing Hong, Zhenguo Li, Dit-Yan Yeung, and Qiang Xu. Magicdrive: Street view generation with diverse 3d geometry control. arXiv preprint arXiv:2310.02601, 2023. 6
- [11] Shanghua Gao, Pan Zhou, Ming-Ming Cheng, and Shuicheng Yan. Masked diffusion transformer is a strong image synthesizer. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 23164–23173,

2023. 2, 3, 4, 5

- [12] Shenyuan Gao, Jiazhi Yang, Li Chen, Kashyap Chitta, Yihang Qiu, Andreas Geiger, Jun Zhang, and Hongyang Li. Vista: A generalizable driving world model with high fidelity and versatile controllability. In Advances in Neural Information Processing Systems (NeurIPS), 2024. 1, 2, 3, 6, 7
- [13] Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized textto-image diffusion models without specific tuning. arXiv preprint arXiv:2307.04725, 2023. 3
- [14] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Doll´ar, and Ross Girshick. Masked autoencoders are scalable vision learners. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 16000– 16009, 2022. 1, 2, 4, 5
- [15] Roberto Henschel, Levon Khachatryan, Daniil Hayrapetyan, Hayk Poghosyan, Vahram Tadevosyan, Zhangyang Wang, Shant Navasardyan, and Humphrey Shi. Streamingt2v: Consistent, dynamic, and extendable long video generation from text. arXiv preprint arXiv:2403.14773, 2024. 3
- [16] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017. 6
- [17] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022. 2
- [18] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video diffusion models. Advances in Neural Information Processing Systems, 35:8633–8646, 2022. 3
- [19] Anthony Hu, Lloyd Russell, Hudson Yeo, Zak Murez, George Fedoseev, Alex Kendall, Jamie Shotton, and Gianluca Corrado. Gaia-1: A generative world model for autonomous driving. arXiv preprint arXiv:2309.17080, 2023. 1, 2, 3
- [20] Yihan Hu, Jiazhi Yang, Li Chen, Keyu Li, Chonghao Sima, Xizhou Zhu, Siqi Chai, Senyao Du, Tianwei Lin, Wenhai Wang, et al. Planning-oriented autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 17853–17862, 2023. 1

- [21] Junpeng Jiang, Gangyi Hong, Lijun Zhou, Enhui Ma, Hengtong Hu, Xia Zhou, Jie Xiang, Fan Liu, Kaicheng Yu, Haiyang Sun, et al. Dive: Dit-based video generation with enhanced control. arXiv preprint arXiv:2409.01595, 2024. 2, 3, 6, 7
- [22] Seung Wook Kim, Jonah Philion, Antonio Torralba, and Sanja Fidler. Drivegan: Towards a controllable high-quality neural simulation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5820–5829, 2021. 6
- [23] Yann LeCun. A path towards autonomous machine intelligence version 0.9. 2, 2022-06-27. Open Review, 62(1):1–62,

2022. 1

- [24] Zhiqi Li, Wenhai Wang, Hongyang Li, Enze Xie, Chonghao Sima, Tong Lu, Yu Qiao, and Jifeng Dai. Bevformer: Learning bird’s-eye-view representation from multi-camera images via spatiotemporal transformers. In European conference on computer vision, pages 1–18. Springer, 2022. 1
- [25] Zhiqi Li, Zhiding Yu, Shiyi Lan, Jiahan Li, Jan Kautz, Tong Lu, and Jose M Alvarez. Is ego status all you need for openloop end-to-end autonomous driving? In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14864–14873, 2024. 1
- [26] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022. 3
- [27] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022. 3
- [28] I Loshchilov. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017. 2
- [29] Open-Sora. Open-sora. URL: https://github.com/ hpcaitech/Open-Sora, 2024. 3
- [30] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4195–4205,

2023. 3

- [31] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023. 3
- [32] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 3
- [33] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 1
- [34] Pei Sun, Henrik Kretzschmar, Xerxes Dotiwalla, Aurelien Chouard, Vijaysai Patnaik, Paul Tsui, James Guo, Yin Zhou, Yuning Chai, Benjamin Caine, et al. Scalability in perception for autonomous driving: Waymo open dataset. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 2446–2454, 2020. 6

- [35] Matthew Tancik, Pratul Srinivasan, Ben Mildenhall, Sara Fridovich-Keil, Nithin Raghavan, Utkarsh Singhal, Ravi Ramamoorthi, Jonathan Barron, and Ren Ng. Fourier features let networks learn high frequency functions in low dimensional domains. Advances in neural information processing systems, 33:7537–7547, 2020. 4
- [36] Zhan Tong, Yibing Song, Jue Wang, and Limin Wang. Videomae: Masked autoencoders are data-efficient learners for self-supervised video pre-training. Advances in neural information processing systems, 35:10078–10093, 2022. 2, 5
- [37] Thomas Unterthiner, Sjoerd Van Steenkiste, Karol Kurach, Raphael Marinier, Marcin Michalski, and Sylvain Gelly. Towards accurate generative models of video: A new metric & challenges. arXiv preprint arXiv:1812.01717, 2018. 6
- [38] Limin Wang, Bingkun Huang, Zhiyu Zhao, Zhan Tong, Yinan He, Yi Wang, Yali Wang, and Yu Qiao. Videomae v2: Scaling video masked autoencoders with dual masking. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14549–14560, 2023. 2, 5
- [39] Xiaofeng Wang, Zheng Zhu, Guan Huang, Xinze Chen, Jiagang Zhu, and Jiwen Lu. Drivedreamer: Towards real-worlddriven world models for autonomous driving. arXiv preprint arXiv:2309.09777, 2023. 1, 3, 6, 8
- [40] Yuqi Wang, Jiawei He, Lue Fan, Hongxin Li, Yuntao Chen, and Zhaoxiang Zhang. Driving into the future: Multiview visual forecasting and planning with world model for autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14749–14759, 2024. 2, 3, 6, 8
- [41] Wei Wu, Xi Guo, Weixuan Tang, Tingxuan Huang, Chiyu Wang, Dongyue Chen, and Chenjing Ding. Drivescape: Towards high-resolution controllable multi-view driving video generation. arXiv preprint arXiv:2409.05463, 2024. 1
- [42] Shaoyuan Xie, Lingdong Kong, Wenwei Zhang, Jiawei Ren, Liang Pan, Kai Chen, and Ziwei Liu. Benchmarking and improving bird’s eye view perception robustness in autonomous driving. arXiv preprint arXiv:2405.17426, 2024. 1
- [43] Zhenda Xie, Zheng Zhang, Yue Cao, Yutong Lin, Jianmin Bao, Zhuliang Yao, Qi Dai, and Han Hu. Simmim: A simple framework for masked image modeling. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9653–9663, 2022. 2, 4
- [44] Jiazhi Yang, Shenyuan Gao, Yihang Qiu, Li Chen, Tianyu Li, Bo Dai, Kashyap Chitta, Penghao Wu, Jia Zeng, Ping Luo, et al. Generalized predictive model for autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14662–14672, 2024. 1, 2, 3, 6, 7
- [45] Zhongqi Yue, Jiankun Wang, Qianru Sun, Lei Ji, Eric I Chang, Hanwang Zhang, et al. Exploring diffusion timesteps for unsupervised representation learning. arXiv preprint arXiv:2401.11430, 2024. 5
- [46] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3836–3847, 2023. 3

- [47] Guosheng Zhao, Xiaofeng Wang, Zheng Zhu, Xinze Chen, Guan Huang, Xiaoyi Bao, and Xingang Wang. Drivedreamer-2: Llm-enhanced world models for diverse driving video generation. arXiv preprint arXiv:2403.06845,

2024. 1, 6

- [48] Hongkai Zheng, Weili Nie, Arash Vahdat, and Anima Anandkumar. Fast training of diffusion models with masked transformers. arXiv preprint arXiv:2306.09305, 2023. 1, 2, 3, 4, 5
- [49] Jun Zheng, Fuwei Zhao, Youjiang Xu, Xin Dong, and Xiaodan Liang. Viton-dit: Learning in-the-wild video try-on from human dance videos via diffusion transformers. arXiv preprint arXiv:2405.18326, 2024. 3
- [50] Rui Zhu, Yingwei Pan, Yehao Li, Ting Yao, Zhenglong Sun, Tao Mei, and Chang Wen Chen. Sd-dit: Unleashing the power of self-supervised discrimination in diffusion transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8435– 8445, 2024. 2, 3, 4, 5, 8

## MaskGWM: A Generalizable Driving World Model with Video Mask Reconstruction

### Supplementary Material

#### 6. More Ablation Experiments

##### 6.1. Additional Results of Mask Ratio

Table 5 illustrates the impact of the mask ratio r on mask reconstruction across various branches and temporal attention strategies. Our findings reveal several key insights: (1) The temporal branch equipped with masked temporal selfattention is more sensitive to mask ratio and necessitates a substantially lower mask ratio compared to the spatial branch. (2) The influence of the mask ratio on the proposed shifted temporal self-attention is more consistent with that observed on the spatial branch. As depicted in Fig.4, the main difference in the DiT Encoder with the spatial branch is the positional shift, which can be effectively handled by positional encoding. Consequently, this allows for the attainment of an well-performed mask ratio (e.g. 0.25) in both spatial MR and temporal MR.

r M = Mspatial M = Mtime M = Mˆ time 0 136.5 136.5 136.5

0.1 125.8 133.2 123.7 0.25 116.7 142.6 121.3

0.4 155.9 179.1 149.8

Table 5. FVD comparisons on mask ratio.

##### 6.3. Additional Results of Mask Reconstruction on Long-Horizon Prediction

To further analyse the influence of MR on auto-regressive generation, we extend the video duration to approximately 12 seconds and documented the metrics in Fig.7. The results indicate that MR is also effective in enhancing performance in long-sequence prediction. Although our baseline without MR still outperforms Vista, the quality of generation begins to deteriorate notably from about 8 seconds, and the FID score increases to 37.7 at the 10 second, making it also incapable to predict the distant future. Consequently, we conclude that this baseline’s improvements cannot translate to significant advancements in long-sequence. In contrast, when MR is integrated into our method, the fundamental enhancements in single-step generation lead to significantly alleviate quality degradation over time. As a result, MaskGWM is capable of generating 10 Hz videos with discernible scene elements for a long time, and even 60 second examples, which far exceeds both Vista and our nonMR baseline. Therefore, we regard MR as a pivotal design that enables the model to make generalized predictions over extended durations. Note that this evaluation is conducted on 300 videos of OpenDV-2K validation set only, due to their longer video sequence. Thus, the single-step (2.5 seconds) FID and FVD are higher than results in Tab.6, which is computed on 1800 videos.

##### 6.2. Additional Results of Mask Reconstruction on NuScene Dataset

As described in GenAD [44], the training and validation sets of OpenDV-2K are sourced from different YouTube videos with significant scene changes. Therefore, the model’s performance on this dataset can be used for the evaluation of its generalization ability. We also conduct ablation studies in the in-domain setting by evaluating metrics on the nuScenes validation dataset. As shown in Table 6, the proposed mask reconstruction method achieves significant improvements on both metrics.

row&shift att. r FVD↓ FID↓

0 107.2 7.5 √ 0.25 92.5 5.6

Table 6. Ablations on nuScene dataset.

#### 7. Implementation Details 7.1. Concrete DiT Structure

We adopt the framework of SD3 and start our model with 2B parameters. We make several modifications to the original spatial transformer block to facilitate temporal and cross-view context modeling. First, Due to the limited availability of high-quality text data in our training datasets, e.g. only scene-level descriptions on nuScene, we skip the update of text feature by new-initialized temporal and cross-view transformer blocks. Then, for temporal transformer block, we make another modification to accommodate condition frames. To streamline the explanation, we represent the transformation within a transformer block as zout′ = zin′ + fb(zin′ ), where zin′ and zout′ are the input and out features, respectively, and fb is the transformer block. Given the frame-level binary indicator mc with value 0 on condition frames, the diffusion time-step τ, and time-step aware embeddings for scale fscale and shift fshift, we in-

|242<br><br>407<br><br>547<br><br>789<br><br>865<br><br>163<br><br>236<br><br>322<br><br>353 343<br><br>181<br><br>333<br><br>464<br><br>493<br><br>576<br><br>2.5S 5S 7.5S 10S 12.5S<br><br>FVD<br><br>Prediction Time<br><br>VISTA MaskGWM MaskGWM w/o MR<br><br>|
|---|

|14<br><br>24<br><br>28.6<br><br>37.7<br><br>42.3<br><br>9.5<br><br>16.4<br><br>22.35<br><br>24.4<br><br>28<br><br>13.4<br><br>24.9<br><br>35.8<br><br>40<br><br>47<br><br>2.5S 5S 7.5S 10S 12.5S<br><br>FID<br><br>Prediction Time<br><br>VISTA MaskGWM MaskGWM w/o MR<br><br>|
|---|

FVD

FID

(a) FVD (b) FID

- Figure 7. Comparison of Long-horizon FVD metric on OpenDV-2K validation set. MR plays a crucial role in enhancing the capability to predict long video sequences, especially on FID.

troduce condition frames by:

zout′ = zin′ + fb(fscale(mcτ)zin′ + fshift(mcτ)) (5)

Here mcτ is employed to reset the time-step for conditional frames to zero and time-step aware embeddings is applied for linear transform.

We append one temporal transformer block and one view transformer block after each spatial transformer block following the common practice of previous works [12, 40].

##### 7.2. Detailed Training Parameters

We employ the Adam optimizer [28] for model training, using a learning rate of 5e-5. Throughout all training stages, we initiate the process with 1K warm-up steps and then maintain a constant learning rate. For condition frames, we randomly sample from zero to three frames following VISTA. We train Stage 1 for total 62K steps, Stage 2 for total 20K steps and Stage 3 for 6K steps. We select the training step based on numerical metrics from videos that are randomly sampled from the training set. Our training are conducted on 32 A800 GPUs with around 3 days on Stage 1.

##### 7.3. Detailed Sampling Parameters

Our sampling strategy does not incorporate any special designs. We generate the video by sampling 30 steps and utilize a classifier-free guidance scale [17] of 4.0. Following Vista, we generate 25-frame videos containing one reference frame on full nuScene validation set with 5369 samples for our single-view model. All generated videos and corresponding frames are used for computing FVD and FID respectively. For our multi-view model, we generate 150 6view videos for each nuScene scene, resulting in 900 singleview videos. Then, 10K frames are randomly sampled from these 900 videos for computing FID. This is align with the evaluation setting of DriveWM [40].

##### 7.4. Details of Comparisons with Vista

For comparisons with Vista, we use the official sample script and checkpoint. For zero-infer on Waymo dataset, we infer both models without action and the number of condition frame is set to 1. For long-horizon rollout on OpenDV2K dataset, we infer both models without action and the number of condition frame is set to 3 for better temporal continuity across auto-regressive steps. We find numeric improvement is similar for one condition frame but the qualitative continuity is reduced by one-frame auto-regression. For auto-regressive steps larger than 1, we randomly select 25 frames from the generated video sequences to calculate the FVD and FID metrics.

#### 8. Qualitative Results 8.1. Long-horizon rollouts

(what is rollout) Longer prediction We provide more qualitative and longer visualizations with 42-seconds videos in Fig.8. We find MaskGWM can predict stable and consistent driving future, combined with unseen scene with initial scope.

Qualitative comparisons In Fig.11, we make qualitative comparisons with Vista, which is previous state-of-the-art method on generalizable driving world model. Our method can both make stable prediction and generate dynamic objects according to the future, e.g. unseen cars in initial visual scope.

Diverse scenes In Fig.9, we present the extended generation results across various scenes, demonstrating the robust generalization capability of our approach.

Action control In Figure 12, we illustrate the controllability of our method on the OpenDV-2K dataset, adhering to the action module in Vista.

Multi-view generation In Figure 10, we show the multi-

view generation ability coming from lifting our single-view model by extra view transformer blocks.

#### 9. Discussions

##### 9.1. Differences to Vista

Although both our method and Vista [12] aim to construct a generalizable world model using the large-scale OpenDV2K dataset, we highlight several key distinctions here. First, our findings suggest that relying solely on the Diffusion loss may not be optimal for building a world model. We introduce a complementary MR task, which has demonstrated robust generalization capabilities in representation learning tasks. Second, our model enables multi-view video generation through an additional training stage. This also illustrates that multi-view generation can benefit from a welltrained single-view model trained on a dataset encompassing significantly longer durations—over 1,700 hours in the OpenDV-2K dataset. Third, our model achieves longer prediction durations than Vista. As indicated by the slope of the metric changes in Fig. 7, our method maintains stable video prediction results, up to 15 seconds by autoregressive generation, whereas the generation quality of Vista degrades notably at this point. Moreover, we have found that our model can sustain stable generation over longer time periods across diverse scenes. Regarding quantitative evaluation, our model exhibits superior generalization capabilities, as evidenced by results on both the OpenDV-2K and Waymo datasets. On the standard nuScene benchmark, our approach also yields better results, with a 19% improvement in FID and a 3% decrease in FVD.

##### 9.2. Usage of Stable Diffusion 3

Our baseline, built upon the SD3 [7] model, yields superior results compared to GenAD (trained on SDXL [31]) and performance slightly lower than Vista (initialized with SVD [1]). Since both GenAD and our baseline are derived from image generation models, the improved performance of our baseline demonstrates the effectiveness of SD3. The superiority of SVD is attributed to its wellinitialized temporal blocks, which have undergone multistage pre-training on extensive video datasets. Therefore, enhancing the data efficiency of SD3—as in our MR policy—and incorporating more video data present promising avenues to bridge this performance gap.

##### 9.3. Future impact of MR.

In our method, MR acts as a complementary task to the diffusion loss, incorporating better video prediction abilities. Within the scope of representation learning, MR conducts context reasoning in a self-supervised way and can be generalized to various tasks. This aligns with our design: recovering the original MR at low noise levels using de-

tailed local context. Our results show that diffusion models may excel in generating high-fidelity results but learn context reasoning slowly, which can be improved through the MR task. More generally, the effectiveness of MR shows that relying solely on diffusion may not be the optimal approach for driving world models. A similar inspiration can also be found in GAIA-1, where the prediction ability is decomposed into an auto-regressive model and a diffusion model. Exploring training targets for world models can be a promising direction.

##### 9.4. Limitations.

Although better generalization ability and quality are achieved, there still exist some limitations that call for future works. (1) Controllability. Since we focused our main improvements on generalization ability and long-duration prediction, the action module follows the design of Vista. We have found several challenging cases in control, such as unreasonable commands. Similar to Vista, our method relies on resampling the nuScenes dataset to learn control ability. As a result, finding better feedback strategies and larger datasets for action learning is a promising direction. (2) Prediction of Uncertain Future. This phenomenon mainly arises when encountering complex traffic scenarios, especially when predicting the movement of each vehicle is difficult. (3) Generation of Non-Front View Images. Since multi-view capability is introduced only at the last training stage with a single nuScenes dataset, the images of nonfront views lack exposure before this stage. Incorporating non-front view data at an earlier stage or adding more multiview datasets (e.g., Waymo) may help address this problem.

Time

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

0-12s

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

12-28s

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

[Figure 303]

28-42s

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

0-12s

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

12-28s

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

###### 28-42s

[Figure 325]

[Figure 326]

[Figure 327]

[Figure 328]

[Figure 329]

[Figure 330]

[Figure 331]

0-12s

[Figure 332]

[Figure 333]

[Figure 334]

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

###### 12-28s

[Figure 339]

[Figure 340]

[Figure 341]

[Figure 342]

[Figure 343]

[Figure 344]

[Figure 345]

###### 28-42s

- Figure 8. Generalization ability of MaskGWM with longer time.

[Figure 346]

[Figure 347]

[Figure 348]

[Figure 349]

[Figure 350]

[Figure 351]

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

[Figure 356]

[Figure 357]

[Figure 358]

[Figure 359]

[Figure 360]

[Figure 361]

[Figure 362]

[Figure 363]

[Figure 364]

[Figure 365]

[Figure 366]

[Figure 367]

[Figure 368]

[Figure 369]

[Figure 370]

[Figure 371]

[Figure 372]

[Figure 373]

[Figure 374]

[Figure 375]

[Figure 376]

[Figure 377]

[Figure 378]

[Figure 379]

[Figure 380]

[Figure 381]

[Figure 382]

[Figure 383]

[Figure 384]

[Figure 385]

[Figure 386]

[Figure 387]

[Figure 388]

[Figure 389]

[Figure 390]

[Figure 391]

[Figure 392]

[Figure 393]

###### Figure 9. Generalization ability of MaskGWM in more scenarios.

[Figure 394]

0s

[Figure 395]

2s

[Figure 396]

0s

[Figure 397]

2s

Figure 10. Generalization ability of multi-view videos.

|+0S|+2S|+4S|+6S|+8S|+10S|
|---|---|---|---|---|---|

[Figure 398]

[Figure 399]

[Figure 400]

[Figure 401]

[Figure 402]

[Figure 403]

Vista

[Figure 404]

[Figure 405]

[Figure 406]

[Figure 407]

[Figure 408]

[Figure 409]

Mask

GWM

[Figure 410]

[Figure 411]

[Figure 412]

[Figure 413]

[Figure 414]

[Figure 415]

Vista

[Figure 416]

[Figure 417]

[Figure 418]

[Figure 419]

[Figure 420]

[Figure 421]

Mask

GWM

Figure 11. Qualitative comparison with Vista.

[Figure 422]

[Figure 423]

[Figure 424]

[Figure 425]

[Figure 426]

[Figure 427]

Turn Left

[Figure 428]

[Figure 429]

[Figure 430]

[Figure 431]

[Figure 432]

[Figure 433]

Turn Right

[Figure 434]

[Figure 435]

[Figure 436]

[Figure 437]

[Figure 438]

[Figure 439]

Keep Forward

Figure 12. Action control ability of MaskGWM.

