## ViFeEdit: A Video-Free Tuner of Your Video Diffusion Transformer

##### Ruonan Yu1, Zhenxiong Tan1, Zigeng Chen1, Songhua Liu2⋆, Xinchao Wang1⋆

- 1 National University of Singapore
- 2 Shanghai Jiao Tong University

{ruonan, zhenxiong, zigeng99}@u.nus.edu, liusonghua@sjtu.edu.cn, xinchao@nus.edu.sg

# arXiv:2603.15478v1[cs.CV]16Mar2026

3D Chibi Style Ghibli Studio Style American Cartoon Style

A drone is hovering steadily

Source Video

Source Video A cat is walking steadily

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

ConsistentStyleTransfer

Non-RigidReplacement

ObjectReplacement

RigidReplacement

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

Source Video with a giant inflatable flamingo with a cyclist riding close

Source Video A woman in purple is greeting

Source Video without a red cap

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

ColorAlteration

ColorAlteration

ObjectRemoval

ObjectAddition

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

Fig. 1: The visualization results of our proposed method ViFeEdit. Our proposed method can adapt text-to-video DiTs to various video editing tasks without any video data. Here, we demonstrate our proposed method on six fine-grained video editing tasks, including style transfer, rigid replacement, non-rigid replacement, color alternation, object addition and object removal.

Abstract. Diffusion Transformers (DiTs) have demonstrated remarkable scalability and quality in image and video generation, prompting growing interest in extending them to controllable generation and editing tasks. However, compared to the image counterparts, progress in video control and editing remains limited, mainly due to the scarcity of paired video data and the high computational cost of training video diffusion models. To address this issue, in this paper, we propose a videofree tuning framework termed ViFeEdit for video diffusion transformers. Without requiring any forms of video training data, ViFeEdit achieves versatile video generation and editing, adapted solely with 2D images. At the core of our approach is an architectural reparameterization that decouples spatial independence from the full 3D attention in modern

⋆ Corresponding authors.

video diffusion transformers, which enables visually faithful editing while maintaining temporal consistency with only minimal additional parameters. Moreover, this design operates in a dual-path pipeline with separate timestep embeddings for noise scheduling, exhibiting strong adaptability to diverse conditioning signals. Extensive experiments demonstrate that our method delivers promising results of controllable video generation and editing with only minimal training on 2D image data. Codes are available here.

Keywords: Video editing · Video control · Video-free tuner

### 1 Introduction

Diffusion transformers (DiTs) [10,13,36] have recently emerged as a highly effective backbone for both image [37,39,53,54,62] and video generation [4,6,16, 24, 29, 32, 47, 58, 66], exhibiting strong generative quality and favorable scaling behavior. Recently, to further effectively accommodate the growing diversity of user requirements, attention has been devoted to controllable generation and editing tasks [40,63,64].

By training on large-scale paired datasets, current DiTs [5,17,26,45,46,50,65] have achieved high fidelity and strong usability in image control and editing tasks. However, its counterpart in video control and editing [3,8,11,15,18,20, 34,44,55,57,60,67] is substantially more challenging since it requires not only spatially coherent modifications, as in image editing, but also their temporally consistent propagation, achieving joint spatiotemporal coherence. Moreover, constructing paired video datasets [1,49,61] is substantially more demanding than for images, owing to the increased temporal complexity and the expensive framelevel annotation required for temporal alignment. For instance, recent efforts [1] to curate such datasets reportedly consumed over 10,000 GPU days. Even with these datasets available, training models for effective video editing and control remains highly resource-intensive due to the inherent multi-frame dependency of video data, typically feasible only for industrial laboratories equipped with large-scale GPU clusters.

Motivated by these drawbacks, we are curious about one question: Can a DiT for video editing be effectively tuned without videos, using only 2D images? In this paper, we answer this question affirmatively by introducing ViFeEdit, a videofree tuner for video diffusion transformers that enables DiT-based video editors to perform diverse video control and editing tasks with minimal training cost. At the core of ViFeEdit lies a structural decoupling of spatial and temporal modeling within a DiT-based video generator. Specifically, we disentangle the spatial token modeling from the temporal dimension, allowing the tuner to learn spatial editing behaviors purely from 2D images. Meanwhile, the pretrained temporal modules of the base video generator remain intact, preserving its inherent capability to maintain temporal coherence across frames. This design enables ViFeEdit to adapt to various video editing tasks without compromising temporal consistency or requiring any video-based supervision.

However, achieving a clean spatiotemporal decoupling in state-of-the-art DiT architectures, such as the Wan series, is highly non-trivial, as they typically adopt a 3D full-attention mechanism that jointly models spatial and temporal tokens in a unified interaction space. As a result, it is difficult to specify which parts of the computation correspond to spatial reasoning and which to temporal reasoning. Recent studies [52] further reveal that modern DiTs dynamically allocate spatial or temporal attention heads depending on the input prompt and diffusion timestep, which makes the decoupling problem even more challenging.

In this paper, we propose an architectural reparameterization technique to address the above challenge. Instead of explicitly enforcing a hard separation within the original 3D attention, we introduce a pair of mutually complementary

- 2D spatial attention blocks that are dedicated to spatial modeling. On the one hand, these two blocks are initialized to counteract each other, which enables sign-aware semantic editing through decoupled enhancement and suppression in spatial attention. Also, it allows the model to reuse the rich spatial priors of pretrained 3D attention layers and preserve its original behavior at initialization and thus providing a stable starting point for adaptation. On the other hand, since the original 3D attention components are entirely frozen, the pretrained temporal modeling capability remains untouched. Consequently, even when the model is trained solely on 2D images, it can still generate temporally stable and coherent videos during inference.

Moreover, to further enhance performance, we introduce a dual-path pipeline that separately processes latent states and conditional signals. By assigning distinct timestep embeddings for noise scheduling to each branch, this design facilitates more stable optimization and faster convergence. We conduct extensive experiments in six fine-grained video editing tasks [19,27], including style transfer, rigid object replacement, non-rigid object replacement, color alteration, object addition, and object removal, to validate the effectiveness of our method. Results demonstrate that our approach enables text-to-video diffusion models to perform diverse editing tasks with minimal computational cost, requiring only a limited amount of image data (100–250 pairs).

Our contributions can be summarized as follows:

- – To the best of our knowledge, we present the first approach that adapts text-to-video DiTs to diverse video editing tasks in a video-free scheme;
- – To preserve temporal consistency, we introduce an architectural reparameterization that decouples spatial interactions from the full 3D attention and operates within a dual-path pipeline using separate timestep embeddings.
- – Extensive experiments demonstrate that, with only limited image data and minimal computational cost, our proposed method achieves promising performance across a wide spectrum of video editing tasks.

### 2 Related Works

In this section, we summarize recent progress in diffusion-based video editing approaches. These approaches can be broadly categorized into three paradigms:

(1) temporal-adaptation method that explicitly incorporate temporal modules into image backbones [11,18,33,42,51], (2) training-free plug-and-play attentionand latent-modulation methods [12,23,28,31,38,41,48,57] that manipulate attention or latent representations during inference, and (3) end-to-end video editing methods [1,9,20,22,59,60,67] that train a video-conditioned generative model on paired or synthetic supervision to directly produce edited videos.

Temporal-Adaptation Methods. These approaches [11,18,33,42,51] extend pre-trained image diffusion models by explicitly incorporating temporal modeling to ensure cross-frame consistency. They typically inject temporal modules or recurrent connections into pre-trained image models to capture motion dynamics and temporal representations. While effective in improving temporal coherence, such pipelines are computationally expensive and typically require additional training or per-video fine-tuning to learn motion dynamics, which may limit their scalability in real-world applications.

Attention- and Latent-Modulation Methods. To enhance efficiency, attention- or latent-based strategies [12,23,28,31,38,41,48,57] modulate spatial or temporal attention within existing diffusion architectures. By reusing frozen image backbones without full temporal training, these approaches achieve higher efficiency lower memory cost. However, their editing capacity is largely confined to appearance-level refinements, making them insufficient for structural or largescale transformations that require deeper spatiotemporal understanding.

End-to-End Methods. More recently, a new class of high-capacity video editing frameworks [1,9,20,22,59,60,67] has emerged, trained in a fully supervised manner on large-scale paired video datasets. These models demonstrate impressive editing strength and robust generalization through joint optimization of content and motion. Nevertheless, such performance comes at the expense of massive computational and data requirements, as curating large-scale paired datasets remains both costly and time-consuming.

### 3 Method

In this section, we present the technical details of the proposed ViFeEdit. We first introduce the preliminaries of DiT-based text-to-video generators in Sec. 3.1. Next, Sec. 3.3 details the architectural reparameterization technique for spatiotemporal decoupling, which serves as the core of our video-free adaptation framework. Finally, Sec. 3.2 elaborates on the dual-path pipeline that enables text-tovideo DiTs to perform video editing, including the interaction between the two branches and the separate temporal embedding scheme. The overall framework of our proposed method is shown in Fig. 2.

#### 3.1 Preliminary

Effectively capturing spatial and temporal dependencies in the latent space, DiTs [36] have been widely adopted in modern video generators, e.g., Wan [47]. Typically, a text-to-video DiT takes noisy video latent maps Z ∈ RB×N×d and

text tokens CT ∈ RB×M×c as inputs. Here, B denotes the batch size, N and

- M represent the number of video and text tokens, respectively, while d and c denote the feature dimensions of the video and text embeddings. In particular,
- N = f × h × w, where f is the number of frames and h and w are the spatial dimensions.

To achieve coherent and temporally consistent video generation, modern DiTbased video generators adopt full 3D attention to jointly capture spatial and temporal dependencies for smooth and stable video results.

Attn3D(X) = Attention(XWQ,XWK,XWV )WO, Attention(Q,K,V ) = Softmax(

QK⊤ √

d′ )V,

(1)

where X denotes the hidden state at a given DiT layer, WQ, WK, WV , and WO are the learnable parameters, and d′ represents the dimensionality of this attention feature space.

During training, they apply the Flow Matching mechanism [30] and obtain noisy video latent maps Zt with t ∈ [0,1] by:

Zt = tϵ + (1 − t)Z0, vt = ϵ − Z0, ϵ ∼ N(0,I). (2)

The parameters θ of the DiT function u are optimized using the following objective:

0,CT),t[∥uθ(Zt,CT,t) − vt∥2]. (3) In this paper, with only minimal additional parameters, we adapt the text-tovideo DiTs to handle various video editing and control tasks without any video training data. We introduce our proposed video-free tuning framework ViFeEdit in the following sections.

L = Eϵ,(Z

#### 3.2 Spatio-Temporal Decoupling

As shown in Sec. 4, directly fine-tuning the full 3D attention using only 2D images can disrupt the temporal dynamics inherent in videos, leading to frozen frames during inference. The key to addressing this issue lies in a spatio-temporal decoupling mechanism that enables fine-tuning solely the spatial component with 2D images while preserving the model original temporal patterns.

Explicit decoupling within the full 3D attention module is incompatible with this setting, as the spatial-temporal roles of attention heads vary across denoising steps and conditioning prompts [52]. We tackle this challenge through an architectural reparameterization technique. Specifically, we keep the original 3D attention untouched and additionally introduce a pair of complementary 2D spatial attention modules. This positive–negative attention architecture facilitates sign-aware semantic editing, where positive and negative semantic signals are explicitly disentangled to enable controlled enhancement and suppression within spatial attention. Here, the 2D spatial attention modules are initialized with the parameters of the corresponding 3D attention module to reuse the rich spatial

###### Reshape

!,#! ∈ ℝ#×(&×'×()×*

!+,#!+ ∈ ℝ(#×&)×('×()×*

#

"!

"! #,""

!+,#!+ ∈ ℝ(#×&)×('×"()×*

!,#! ∈ ℝ"#×(&×'×()×*

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

T5-XXL

T5-XXL

Layer Norm

Layer Norm

[Figure 47]

[Figure 48]

3D Self-Attention

[Figure 49]

2D Spatial Attention

3D Self-Attention

###### +

N ×

N ×

[Figure 50]

[Figure 51]

Layer Norm

Layer Norm

[Figure 52]

[Figure 53]

Cross-Attention

Cross-Attention

[Figure 54]

[Figure 55]

2D Spatial Attention-Neg

2D Spatial Attention-Pos

[Figure 56]

[Figure 57]

Layer Norm

Layer Norm

-

[Figure 58]

[Figure 59]

FFN

FFN

Edited Video

Output Video

(a) Original (b) Our Video-Free Tuner

- Fig. 2: The architecture of DiT blocks of (a) the original text-to-video Wan2.1 model and (b) our proposed video-free tuner ViFeEdit for video editing and control tasks. Here, we enable text-to-video DiTs to handle diverse video editing and control tasks without any video data. Specifically, the source video CV is jointly fed into the model and interacts with the noisy video latent Z in the 2D spatial attention branch, providing explicit reference guidance.

priors of pre-trained 3D attention layers and ensure training stability. Also, these 2D attention modules are designed to interact in a residual manner, such that their combined output is zero at initialization, thereby preserving original performance of the model. Formally, the final result incorporating the original 3D attention can be written as:

Attn3D(X) + AttnSpaPos(X′) − AttnSpaNeg(X′), (4)

where X′ represents X with a consistent frame index used for the temporal position embedding across all latent frames, and AttnSpaPos and AttnSpaNeg denote the newly introduced spatial attention modules, each operating independently on individual frames and computing attention only within the spatial domain.

To enable fine-tuning using only 2D images, we only update the positive and negative spatial attention modules, AttnSpaPos and AttnSpaNeg, as well as the feed-forward layers to enhance performance. Again, the original 3D attention remains frozen during fine-tuning to preserve its pretrained temporal generation capability.

#### 3.3 Dual-Path Pipeline

Building upon the proposed spatio-temporal decoupling technique, the remaining challenge is to equip the DiT with the ability to take a source video as

input and effectively inject conditional information into the backbone features. Inspired by recent image editing approaches [45,46,65], instead of introducing a separate encoder, we reuse the DiT backbone to encode the conditional information. However, unlike previous approaches that directly concatenate conditional tokens with noisy latent tokens and allow them to interact throughout all attention layers, we adopt a dual-path pipeline. Specifically, the two streams are processed separately and only interact within the positive and negative spatial attention modules introduced above, ensuring that the original 3D attention remains intact and its temporal generation capability is preserved.

In other words, the 3D attention treats the noisy video latents Z ∈ RB×N×d, N = f × h × w, and the video condition CV ∈ RB×N×d as independent samples by concatenating them along the batch dimension, i.e., [Z,CV ] ∈ R2B×N×d, and assigning them separate 3D position embeddings as usual. For spatial attention modules, we flatten the inputs Z and CV into R(B×f)×(h×w)×d as the single-frame videos with batch size B × f and concatenated along the spatial dimension, i.e., either h or w, ensuring the interaction is within each frame. Without loss of generality, when concatenation occurs along the w dimension, we assign positional indices within [0,2w) for this axis, while setting the temporal positional indices to 0 for all tokens. This design enables the model to learn rich editing and control tasks mapping solely from 2D paired image training data, while strengthening the frame-wise consistency between the generated video Z and the original input video CV .

Optionally, in order to further enhance structural consistency, inspired by SDEdit [35], CV can be used as a noise prior to initialize the noisy latent during inference:

Zα = (1 − α)CV + αϵ, (5)

where α ∈ [0,1] is a hyper-parameter controlling the strength of the prior. The flow-matching schedule then starts from t = α.

Separate Timestep Embeddings. During training and inference, Z and CV correspond to the noisy latent map and the clean source video, respectively. As a result, they exhibit distinct noise levels, and using the same timestep input for both can blur the conditional guidance. To address this issue, we assign separate timestep embeddings to Z and CV , ensuring reliable conditional injection during both training and inference. Specifically, for Z, the timestep is the current flow-matching timestep t as usual, while for CV , the timestep is always 0, indicating a clean video input. These separate embeddings are concatenated along the batch dimension accordingly.

### 4 Experiments

#### 4.1 Settings and Implementation Details

In this paper, we propose a video-free tuning framework, ViFeEdit, to enable text-to-video diffusion transformers to handle various video editing and control

Source Video OmniConsistency VACE* Ours

Source Video OmniConsistency VACE* Ours

Consistency Style Consistency Style Consistency Style

Consistency Style Consistency Style Consistency Style

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

3DChibiStyleGhibliStyleAmericanCartoonStyle

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

- Fig. 3: The visualization results of baselines and our proposed method on consistent style transfer tasks. ∗ means that the pretrained model VACE is further finetuned on the paired image data for style transfer learning.

tasks with solely 2D paired image data. To validate the effectiveness of our proposed method, we conduct comprehensive experiments on 6 video editing tasks, i.e., consistent style transfer, rigid object replacement, non-rigid object replacement, color alteration, object addition, and object removal. We also conduct experiments on depth-to-video generation, please refer to the supplementary for more results. Here, we adopt the open-source text-to-video model Wan2.1-T2V1.3B [47] as the base model.

Finetuning Settings and Details Here, for the consistent style transfer task, we adopt the open-source image dataset OmniConsistency [43], which contains 100-200 paired samples for each style. With only the limited paired image data, our method achieves stable and high-quality video stylization results. For the remaining editing tasks, we adopt GPT-5 to randomly generate prompts for editing tasks and then adopt FLUX.1-dev to generate the source images and Qwen-Image-Edit-2509 [50] to generate the corresponding target edited images. Each task consists of 250 paired samples. Each image data is treated as singleframe video. During training, we employ LoRA fine-tuning [14], which is both efficient and lightweight. The rank is set to 32 for all tasks, and the training typically lasts within 20 epochs, yielding high-quality editing results for all tasks.

- Table 1: The evaluation results of baselines and our proposed method on style transfer task. Here, the experiments are conducted on three target styles, 3D Chibi style, Ghibli Studio style, and American Cartoon style. ∗ means that the pretrained VACE is further finetuned on the paired image data for style transfer learning. † means that the evaluation of consistency is conducted between the source and target video.

VBench VLM Score Subject Background Temporal Motion

Method/Setting

Structural Motion

Stylization Consistency Consistency Flickering Smoothness Consistency† Consistency†

Color

3D Chibi Style

- OmniConsistency 0.9711 0.9712 0.9948 0.9811 0.9113 90.86 89.67 93.06 VACE∗ 0.9751 0.9758 0.9945 0.9805 0.8737 84.68 86.29 91.13

Ours 0.9811 0.9785 0.9980 0.9872 0.9259 91.13 90.16 93.06 Ghibli Studio Style

OmniConsistency 0.9689 0.9715 0.9946 0.9805 0.9023 90.80 90.48 93.06 VACE∗ 0.9680 0.9709 0.9949 0.9828 0.7521 89.84 88.87 91.77 Ours 0.9773 0.9777 0.9978 0.9866 0.9106 93.39 92.42 93.39

American Cartoon Style

- OmniConsistency 0.9712 0.9699 0.9915 0.9702 0.9075 88.87 88.79 91.33 VACE∗ 0.9718 0.9778 0.9942 0.9797 0.8349 85.16 86.45 89.19

Ours 0.9802 0.9789 0.9974 0.9844 0.8764 91.46 90.81 91.46

Evaluation Settings and Details For the style transfer task, we follow the official VBench evaluation settings [19]. We generate five base videos for each prompt and apply consistent style transfer methods to obtain stylized videos. The resulting stylized videos are then evaluated using the subject consistency, background consistency, temporal flickering, motion smoothness and color metrics provided by VBench, which collectively measure visual quality, temporal consistency. Further, we evaluate VLM score with Qwen2.5-VL-7B-Instruct [2] on structural consistency and motion consistency between the base video and the stylized video, and stylization quality of the target video for style fidelity. As for other editing tasks, e.g., rigid and non-rigid object replacement, color alteration, object addition, and object removal, we adopt the FiVE-Bench [27], following its provided task prompts to generate base videos and perform edit. We evaluate the edited results using the FiVE-Acc metrics, which offers a comprehensive quantitative measure of editing accuracy. Specifically, to obtain more comprehensive results, the FiVE-Acc metrics are evaluated over entire videos rather than a few sampled frames, ensuring accuracy and stability.

Baseline Settings and Details For the consistent style transfer task, we adopt the powerful end-to-end model Wan2.1-VACE-1.3B [20], which is pretrained on large video datasets, as baseline. To enable the VACE model to handle unseen consistent style transfer tasks, we perform LoRA fine-tuning on the vace branch using the same image dataset OmniConsistency [43], with the rank set to 32 for all styles. Moreover, we adopt OmniConsistency method [43] to conduct frameby-frame style transfer on the base video for comparison, and all experiments are conducted following the official settings and checkpoint. Here, all videos are of 81 frames and the resolution is 480p. As for other editing tasks, we adopt

Source Video SDEdit VidToMe Pyramid-Edit Wan-Edit VACE*

###### Ours

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

ObjectReplacement(Rigid)

[Figure 145]

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

An Ultraman is running along a path through a park, with colorful flowers blooming on either side. The camera follows the Ultraman's movement with a steady pan.

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

ObjectReplacement(Non-Rigid)

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

A dragon is building a nest in a tree, carefully arranging twigs and leaves. The camera zooms in slowly to capture the dragon's meticulous work.

- Fig. 4: The visualization results of baselines and our proposed method on rigid and non-rigid replacement tasks. ∗ means that the pretrained VACE is further finetuned on the paired image data for editing tasks.

SDEdit [35], VidToMe [28], Pyramid-Edit [21,27], Wan-Edit [25,27], and Wan2.1VACE-1.3B as baselines. Here, SDEdit and Wan-Edit are based on Wan2.1T2V-1.3B model and the denoising strength is set to 0.7, following the default settings. VidToMe is based on Stable-Diffusion-1.5, and Pyramid-Edit is based on Pyramid-Flow-miniFLUX model. VACE is LoRA fine-tuned with the same paired image datasets as ours for each editing task to learn editing mappings. Here, all videos are also of 81 frames and 480p resolution, except VidToMe for 63 frames and Pyramid-Edit for 384p followed by official default settings.

#### 4.2 Results on Consistent Style Transfer

For the consistent style transfer task, we adopt the Wan2.1-T2V-1.3B model to generate the base videos with prompts provided by VBench. Following the default settings of VBench, we generate five videos for each prompt using seeds 0–4, and then apply style transfer methods to them. For this task, we evaluate the effectiveness of our proposed method on 3D Chibi, Ghibli, and American Cartoon styles. The evaluation results are reported in Table 1 and the visualization samples are shown in Fig. 3. For more experimental results and visualizations of comparison with Ditto-14B model [1], please refer to the supplementary.

As shown in the results, our proposed method can achieve high-quality style transfer with both temporal consistency and spatial consistency. Here, the OmniConsistency method can preserve the transferred style with stable color retention. However, as OmniConsistency is based on the image diffusion model FLUXdev, and it performs frame-by-frame style transfer, it suffers from poor temporal consistency and struggles to capture coherent motion. Also, when transferring

Source Video SDEdit VidToMe Pyramid-Edit Wan-Edit VACE*

###### Ours

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

ColorAlterationObjectAdditionObjectRemoval

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

A yellow Mini Cooper is smoothly navigating a roundabout in a bustling urban area, surrounded by historic buildings and other vehicles. The camera remains stationary, capturing the car's movement through the roundabout.

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

A brown and white cow wearing a flower garland is walking along a dirt path in a grassy field. The camera remains stationary, capturing the cow's steady movement.

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

A cat is pouncing playfully without a toy in a cozy living room, with a sofa and a window in the background. The camera remains fixed, focusing on the cat's playful behavior.

- Fig. 5: The visualization results of baselines and our proposed method on color alternation, object addition and object removal tasks. ∗ means that the pretrained VACE is further finetuned on the paired image data for editing tasks.

blurred or ambiguous frames, the stylized results often introduces noticeable abrupt changes. This issue can also be seen at the first example of the American Cartoon style in Fig. 3, where a sudden shift occurs due to the lack of temporal modeling. VACE can preserve temporal consistency. However, it suffers from substantial color drift and unstable style adherence, as shown in the second example of Ghibli studio style in Fig. 3. It may come from compensation for its limited style knowledge learned from the image data by over-relying on color changes, which leads to visible chromatic fluctuations. It is further supported by the color metrics in Table 1. With limited paired image data for training, our method achieves high-quality style transfer while simultaneously preserving spatial structure and temporal coherence. It maintains stable color behavior and captures the target style more faithfully, producing the most visually consistent and stylistically accurate results across all evaluated settings.

- Table 2: The evaluation results of baselines and our proposed method on FiVEBenchmark with object replacement (rigid and non-rigid), color alternation, object addition, and object removal tasks. ∗ means that the pretrained VACE is further finetuned on the paired image data for editing tasks.

Object Replacement (Rigid & Non-Rigid) Color Alteration YN-Acc MC-Acc ∪-Acc ∩-Acc FiVE-Acc YN-Acc MC-Acc ∪-Acc ∩-Acc FiVE-Acc

Method/Setting

SDEdit 21.00 32.50 33.50 20.00 26.75 13.00 22.00 23.00 12.00 17.50 VidToMe 36.50 51.00 51.50 36.00 43.75 47.00 51.00 54.00 44.00 49.00 Pyramid-Edit 48.00 67.50 68.00 47.50 57.75 52.00 49.00 59.00 42.00 50.50 Wan-Edit 39.50 52.50 53.00 39.00 46.00 37.00 41.00 44.00 34.00 39.00 VACE∗ 20.50 34.00 34.00 20.50 27.25 82.00 87.00 89.00 80.00 84.50

Ours 72.00 83.50 84.00 71.50 77.75 87.00 96.00 98.00 85.00 91.50 Method/Setting

Object Addition Object Removal

YN-Acc MC-Acc ∪-Acc ∩-Acc FiVE-Acc YN-Acc MC-Acc ∪-Acc ∩-Acc FiVE-Acc SDEdit 0.00 11.11 11.11 0.00 5.56 0.00 0.00 0.00 0.00 0.00

VidToMe 11.11 11.11 11.11 11.11 11.11 0.00 10.00 10.00 0.00 5.00 Pyramid-Edit 66.67 88.89 88.89 66.67 77.78 0.00 0.00 0.00 0.00 0.00

Wan-Edit 33.33 66.67 66.67 33.33 50.00 0.00 0.00 0.00 0.00 0.00 VACE∗ 22.22 22.22 22.22 22.22 22.22 0.00 0.00 0.00 0.00 0.00

Ours 100.00 100.00 100.00 100.00 100.00 80.00 80.00 80.00 80.00 80.00

#### 4.3 Results on Video Editing

We further conduct experiments on a broader set of video editing tasks, e.g., rigid and non-rigid replacement, color alteration, and object addition and removal. Here, we adopt FiVE-Bench as our evaluation benchmark. We compare our method with five video editing methods, SDEdit, VidToMe, Pyramid-Edit, WanEdit, and VACE. All methods are evaluated using their default settings without extra inputs such as depth maps or masked videos. Following the guidelines in FiVE-Bench, we employ the Qwen2.5-VL-7B-Instruct model for evaluation. Notably, our evaluation is performed on the entire video. The quantitative results are presented in Table 2, with visual examples shown in Fig. 4 and Fig. 5.

The results show that our method achieves effective and high-fidelity edits. For both rigid and non-rigid replacement tasks, VACE∗ can do certain editing but introduces background inconsistencies. SDEdit and Wan-Edit maintain background coherence but often provide partial edits. Pyramid-Edit supports replacement but degrades video quality. Our method can perform object replacement with high fidelity in both rigid and non-rigid scenarios, producing detailed target objects that preserve motion and integrate seamlessly with the background. For color alteration, transforming dark colors into bright ones is challenging. As shown in Fig. 5, SDEdit and Wan-Edit only lighten the car, while VidToMe and VACE∗ achieves the transformation but VidToMe lacks localized control, unintentionally affecting nearby cars and the background, and VACE∗ introduces background color inconsistencies. Our method enables precise color changes without disturbing other regions. For the fine-grained addition task, SDEdit and Wan-Edit produce only minor local edits and fail to achieve the required modifications. Pyramid-Edit is able to produce the intended additions, but the target video quality is degraded. Our method can perform fine-grained

- Table 3: The performance of multi-task LoRA and single-task LoRA on FiVE-Bench. Multi-task LoRA is trained with objective addition, removal and color alternation tasks at the same time.

Single-Task LoRA Multi-Task LoRA YN-Acc MC-Acc ∪-Acc ∩-Acc FiVE-Acc YN-Acc MC-Acc ∪-Acc ∩-Acc FiVE-Acc

Task/Setting

Color Alternation 87.00 96.00 98.00 85.00 91.50 87.00 95.00 97.00 85.00 91.00 Object Addition 100.00 100.00 100.00 100.00 100.00 100.00 100.00 100.00 100.00 100.00 Object Removal 80.00 80.00 80.00 80.00 80.00 80.00 90.00 90.00 80.00 85.00

Source Video +Condition Concatenation +Separate Timestep Embeddings +Spatio-Temporal Decoupling +Dual-Path Pipeline Direct Tuning Zero-Conv

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

###### (a) Effectiveness of Each Component (b) Effectiveness of Component Variants

- Fig. 6: The visualization results of ablation studies. (a) The effectiveness of each proposed component. From left to right, each key component cumulatively builds upon the previous configuration. (b) Comparison with other strategies, including Direct Tuning and Zero-Conv.

addition edits while preserving the video quality and overall consistency. For removal task, as indicated by Table 2 and Fig. 5, baseline methods struggle to fully remove target objects, whereas our method achieves complete removal with plausible background completion.

#### 4.4 Ablation Studies

Results on Multi-Task LoRA To further demonstrate the scalability and generalization capability of our proposed method, we conduct multi-task LoRA fine-tuning with color alternation, object addition, and object removal tasks at the same time. The results are shown in Table 3. We also conduct multi-style finetuning experiments in Supplementary Section B.2. The results show that our framework does not require training a separate LoRA or model for each editing task or style, and instead supports multiple tasks within a unified framework.

Effectiveness of Each Key Components We further conduct ablation studies on the key components of our method and provide visualizations to more intuitively illustrate the contribution of each component. As shown in Fig. 6(a), we take the 3D Chibi style transfer task as an example. Here, the starting point is direct tuning with image data and concatenating conditional tokens with noisy tokens. From left to right, each step introduces one more key component of our

method, cumulatively building upon the previous configuration. For the starting point, it is difficult for the model to learn clean representations, as the reference video remains noisy during training. To address this problem, we adopt separate timestep embeddings. It assures the clean reference video and helps the model to learn the target style rapidly. Here, the third column only takes less than half the training time of the second column. However, the motion remains weakened due to the image data training, and the structural consistency is still limited. To address the issue, we further introduce spatio-temporal decoupling. It isolates the learning process on image data to ensure the temporal consistency. To further enhance the spatial consistency, we propose dual-path pipeline. As illustrated in the final column, our proposed method achieves consistent motion, background preservation, and high-quality style transfer.

Comparison with Other Variants To further demonstrate the effectiveness of our method, we compare it with several alternative strategies. The visualization results are shown in Fig. 6(b). First, we directly fine-tune the attention layers (i.e., q, k, v, and o) as well as the feed-forward layers of the Wan2.1T2V-1.3B model. As shown in the results, first column of Fig. 6(b), since 3D attention jointly encodes spatial and temporal information, directly finetuning under image-only supervision can disrupt temporal generation capability, leading to degraded motion quality such as frozen frames. This can also be observed in the direct tuning with condition concatenation, as shown in the second column of Fig. 6. Our proposed method introduce 2D spatial attention branch, which decouples spatial-temporal attention from 3D attention and confines image-based updates only to the 2D spatial branch, preserving temporal modeling learned by the text-to-video backbone. Moreover, to further validate the effectiveness of our proposed positive-negative 2D spatial attention, we compare it with ControlNetstyle zero-convolution approaches, the results are shown in the second column of Fig. 6(b). Although ControlNet-style zero-convolution can achieve a similar initialization effect, it requires to train additional modules from scratch, which is challenging to converge under image-supervision, resulting in weaker alignment with the source video. Our proposed positive-negative 2D spatial attention reuses the rich spatial priors of pre-trained 3D attention layers, and preserve high consistency.

### 5 Conclusion

In this paper, we propose a video-free tuning framework termed ViFeEdit for video diffusion transformers. With solely 2D images, our proposed method can adapt text-to-video DiTs to diverse video editing tasks at minimal costs. Specifically, we propose an architectural reparameterization to decouple the spatial interactions from the full 3D attention. It can isolate the image tuning process and strengthen the visual control ability, without compromising the temporal consistency. Moreover, to further enhance background and motion consistency, we introduce a dual-path pipeline with separate timestep embeddings. It also

benefits training process for more stable optimization and faster convergence. Extensive experiments demonstrate that, with limited paired image data, our proposed method enables video diffusion models to achieve promising performance with a wide range video editing tasks.

### References

- 1. Bai, Q., Wang, Q., Ouyang, H., Yu, Y., Wang, H., Wang, W., Cheng, K.L., Ma, S., Zeng, Y., Liu, Z., et al.: Scaling instruction-based video editing with a high-quality synthetic dataset. arXiv preprint arXiv:2510.15742 (2025)
- 2. Bai, S., Chen, K., Liu, X., Wang, J., Ge, W., Song, S., Dang, K., Wang, P., Wang, S., Tang, J., et al.: Qwen2. 5-vl technical report. arXiv e-prints pp. arXiv–2502

(2025)

- 3. Bian, Y., Zhang, Z., Ju, X., Cao, M., Xie, L., Shan, Y., Xu, Q.: Videopainter: Any-length video inpainting and editing with plug-and-play context control. In: Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers. pp. 1–12 (2025)
- 4. Blattmann, A., Dockhorn, T., Kulal, S., Mendelevitch, D., Kilian, M., Lorenz, D., Levi, Y., English, Z., Voleti, V., Letts, A., et al.: Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127

(2023)

- 5. Brooks, T., Holynski, A., Efros, A.A.: Instructpix2pix: Learning to follow image editing instructions. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 18392–18402 (2023)
- 6. Chen, J., Zhao, Y., Yu, J., Chu, R., Chen, J., Yang, S., Wang, X., Pan, Y., Zhou, D., Ling, H., et al.: Sana-video: Efficient video generation with block linear diffusion transformer. arXiv preprint arXiv:2509.24695 (2025)
- 7. Chen, S., Guo, H., Zhu, S., Zhang, F., Huang, Z., Feng, J., Kang, B.: Video depth anything: Consistent depth estimation for super-long videos. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 22831–22840 (2025)
- 8. Chen, Y., Men, Y., Yao, Y., Cui, M., Bo, L.: Perception-as-control: Fine-grained controllable image animation with 3d-aware motion representation. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 14380–14389

(2025)

- 9. Cheng, J., Xiao, T., He, T.: Consistent video-to-video transfer using synthetic dataset. arXiv preprint arXiv:2311.00213 (2023)
- 10. Esser, P., Kulal, S., Blattmann, A., Entezari, R., Müller, J., Saini, H., Levi, Y., Lorenz, D., Sauer, A., Boesel, F., et al.: Scaling rectified flow transformers for high-resolution image synthesis. In: Forty-first international conference on machine learning (2024)
- 11. Gao, C., Ding, L., Cai, X., Huang, Z., Wang, Z., Xue, T.: Lora-edit: Controllable first-frame-guided video editing via mask-aware lora fine-tuning. arXiv preprint arXiv:2506.10082 (2025)
- 12. Geyer, M., Bar-Tal, O., Bagon, S., Dekel, T.: Tokenflow: Consistent diffusion features for consistent video editing. arXiv preprint arXiv:2307.10373 (2023)
- 13. Ho, J., Jain, A., Abbeel, P.: Denoising diffusion probabilistic models. Advances in neural information processing systems 33, 6840–6851 (2020)
- 14. Hu, E.J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W., et al.: Lora: Low-rank adaptation of large language models. Iclr 1(2), 3 (2022)

- 15. Hu, Z., Xu, D.: Videocontrolnet: A motion-guided video-to-video translation framework by using diffusion model with controlnet. arXiv preprint arXiv:2307.14073

(2023)

- 16. Huang, X., Li, Z., He, G., Zhou, M., Shechtman, E.: Self forcing: Bridging the train-test gap in autoregressive video diffusion. arXiv preprint arXiv:2506.08009

(2025)

- 17. Huang, Y., Huang, J., Liu, Y., Yan, M., Lv, J., Liu, J., Xiong, W., Zhang, H., Cao, L., Chen, S.: Diffusion model-based image editing: A survey. IEEE Transactions on Pattern Analysis and Machine Intelligence (2025)
- 18. Huang, Y., Xiong, W., Zhang, H., Chen, C., Liu, J., Yan, M., Chen, S.: Dive: Taming dino for subject-driven video editing. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 16004–16014 (2025)
- 19. Huang, Z., He, Y., Yu, J., Zhang, F., Si, C., Jiang, Y., Zhang, Y., Wu, T., Jin, Q., Chanpaisit, N., et al.: Vbench: Comprehensive benchmark suite for video generative models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 21807–21818 (2024)
- 20. Jiang, Z., Han, Z., Mao, C., Zhang, J., Pan, Y., Liu, Y.: Vace: All-in-one video creation and editing. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 17191–17202 (2025)
- 21. Jin, Y., Sun, Z., Li, N., Xu, K., Jiang, H., Zhuang, N., Huang, Q., Song, Y., Mu, Y., Lin, Z.: Pyramidal flow matching for efficient video generative modeling. arXiv preprint arXiv:2410.05954 (2024)
- 22. Ju, X., Wang, T., Zhou, Y., Zhang, H., Liu, Q., Zhao, N., Zhang, Z., Li, Y., Cai, Y., Liu, S., et al.: Editverse: Unifying image and video editing and generation with in-context learning. arXiv preprint arXiv:2509.20360 (2025)
- 23. Khachatryan, L., Movsisyan, A., Tadevosyan, V., Henschel, R., Wang, Z., Navasardyan, S., Shi, H.: Text2video-zero: Text-to-image diffusion models are zeroshot video generators. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 15954–15964 (2023)
- 24. Kong, W., Tian, Q., Zhang, Z., Min, R., Dai, Z., Zhou, J., Xiong, J., Li, X., Wu, B., Zhang, J., et al.: Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603 (2024)
- 25. Kulikov, V., Kleiner, M., Huberman-Spiegelglas, I., Michaeli, T.: Flowedit: Inversion-free text-based editing using pre-trained flow models. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 19721–19730

(2025)

- 26. Labs, B.F., Batifol, S., Blattmann, A., Boesel, F., Consul, S., Diagne, C., Dockhorn, T., English, J., English, Z., Esser, P., et al.: Flux. 1 kontext: Flow matching for in-context image generation and editing in latent space. arXiv preprint arXiv:2506.15742 (2025)
- 27. Li, M., Xie, C., Wu, Y., Zhang, L., Wang, M.: Five-bench: A fine-grained video editing benchmark for evaluating emerging diffusion and rectified flow models. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 16672–16681 (2025)
- 28. Li, X., Ma, C., Yang, X., Yang, M.H.: Vidtome: Video token merging for zero-shot video editing. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 7486–7495 (2024)
- 29. Lin, B., Ge, Y., Cheng, X., Li, Z., Zhu, B., Wang, S., He, X., Ye, Y., Yuan, S., Chen, L., et al.: Open-sora plan: Open-source large video generation model. arXiv preprint arXiv:2412.00131 (2024)

- 30. Lipman, Y., Chen, R.T., Ben-Hamu, H., Nickel, M., Le, M.: Flow matching for generative modeling. arXiv preprint arXiv:2210.02747 (2022)
- 31. Lu, T., Zhang, X., Gu, J., Pei, R., Xu, S., Ma, X., Xu, H., Wu, Z.: Fuse your latents: Video editing with multi-source latent diffusion models. In: Proceedings of the 32nd ACM International Conference on Multimedia. pp. 6745–6754 (2024)
- 32. Ma, G., Huang, H., Yan, K., Chen, L., Duan, N., Yin, S., Wan, C., Ming, R., Song, X., Chen, X., et al.: Step-video-t2v technical report: The practice, challenges, and future of video foundation model. arXiv preprint arXiv:2502.10248 (2025)
- 33. Ma, Y., Cun, X., Liang, S., Xing, J., He, Y., Qi, C., Chen, S., Chen, Q.: Magicstick: Controllable video editing via control handle transformations. In: 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV). pp. 9385–9395. IEEE (2025)
- 34. Ma, Y., Feng, K., Hu, Z., Wang, X., Wang, Y., Zheng, M., Wang, B., Wang, Q., He, X., Wang, H., et al.: Controllable video generation: A survey. arXiv preprint arXiv:2507.16869 (2025)
- 35. Meng, C., He, Y., Song, Y., Song, J., Wu, J., Zhu, J.Y., Ermon, S.: Sdedit: Guided image synthesis and editing with stochastic differential equations. arXiv preprint arXiv:2108.01073 (2021)
- 36. Peebles, W., Xie, S.: Scalable diffusion models with transformers. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 4195–4205 (2023)
- 37. Podell, D., English, Z., Lacey, K., Blattmann, A., Dockhorn, T., Müller, J., Penna, J., Rombach, R.: Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952 (2023)
- 38. Qi, C., Cun, X., Zhang, Y., Lei, C., Wang, X., Shan, Y., Chen, Q.: Fatezero: Fusing attentions for zero-shot text-based video editing. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 15932–15942 (2023)
- 39. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image synthesis with latent diffusion models. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 10684–10695 (2022)
- 40. Ruiz, N., Li, Y., Jampani, V., Pritch, Y., Rubinstein, M., Aberman, K.: Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 22500–22510 (2023)
- 41. Shen, T., Huang, Z., Li, X., Lin, Z., Liu, J., Wang, Y., Feng, J., Yang, M.H., Liew, J.H.: Qk-edit: Revisiting attention-based injection in mm-dit for image and video editing. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 19043–19053 (2025)
- 42. Shin, C., Kim, H., Lee, C.H., Lee, S.g., Yoon, S.: Edit-a-video: Single video editing with object-aware consistency. In: Asian Conference on Machine Learning. pp. 1215–1230. PMLR (2024)
- 43. Song, Y., Liu, C., Shou, M.Z.: Omniconsistency: Learning style-agnostic consistency from paired stylization data. arXiv preprint arXiv:2505.18445 (2025)
- 44. Sun, W., Tu, R.C., Liao, J., Tao, D.: Diffusion model-based video editing: A survey. arXiv preprint arXiv:2407.07111 (2024)
- 45. Tan, Z., Liu, S., Yang, X., Xue, Q., Wang, X.: Ominicontrol: Minimal and universal control for diffusion transformer. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 14940–14950 (2025)
- 46. Tan, Z., Xue, Q., Yang, X., Liu, S., Wang, X.: Ominicontrol2: Efficient conditioning for diffusion transformers. arXiv preprint arXiv:2503.08280 (2025)

- 47. Wan, T., Wang, A., Ai, B., Wen, B., Mao, C., Xie, C.W., Chen, D., Yu, F., Zhao, H., Yang, J., et al.: Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314 (2025)
- 48. Wang, W., Jiang, Y., Xie, K., Liu, Z., Chen, H., Cao, Y., Wang, X., Shen, C.: Zero-shot video editing using off-the-shelf image diffusion models. arXiv preprint arXiv:2303.17599 (2023)
- 49. Wang, Y., He, Y., Li, Y., Li, K., Yu, J., Ma, X., Li, X., Chen, G., Chen, X., Wang, Y., et al.: Internvid: A large-scale video-text dataset for multimodal understanding and generation. arXiv preprint arXiv:2307.06942 (2023)
- 50. Wu, C., Li, J., Zhou, J., Lin, J., Gao, K., Yan, K., Yin, S.m., Bai, S., Xu, X., Chen, Y., et al.: Qwen-image technical report. arXiv preprint arXiv:2508.02324 (2025)
- 51. Wu, J.Z., Ge, Y., Wang, X., Lei, S.W., Gu, Y., Shi, Y., Hsu, W., Shan, Y., Qie, X., Shou, M.Z.: Tune-a-video: One-shot tuning of image diffusion models for textto-video generation. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 7623–7633 (2023)
- 52. Xi, H., Yang, S., Zhao, Y., Xu, C., Li, M., Li, X., Lin, Y., Cai, H., Zhang, J., Li, D., et al.: Sparse videogen: Accelerating video diffusion transformers with spatialtemporal sparsity. arXiv preprint arXiv:2502.01776 (2025)
- 53. Xie, E., Chen, J., Chen, J., Cai, H., Tang, H., Lin, Y., Zhang, Z., Li, M., Zhu, L., Lu, Y., et al.: Sana: Efficient high-resolution image synthesis with linear diffusion transformers. arXiv preprint arXiv:2410.10629 (2024)
- 54. Xie, E., Chen, J., Zhao, Y., Yu, J., Zhu, L., Wu, C., Lin, Y., Zhang, Z., Li, M., Chen, J., et al.: Sana 1.5: Efficient scaling of training-time and inference-time compute in linear diffusion transformer. arXiv preprint arXiv:2501.18427 (2025)
- 55. Xing, Z., Feng, Q., Chen, H., Dai, Q., Hu, H., Xu, H., Wu, Z., Jiang, Y.G.: A survey on video diffusion models. ACM Computing Surveys 57(2), 1–42 (2024)
- 56. Yang, L., Kang, B., Huang, Z., Zhao, Z., Xu, X., Feng, J., Zhao, H.: Depth anything v2. Advances in Neural Information Processing Systems 37, 21875–21911 (2024)
- 57. Yang, X., Zhu, L., Fan, H., Yang, Y.: Videograin: Modulating space-time attention for multi-grained video editing. In: The Thirteenth International Conference on Learning Representations (2025)
- 58. Yang, Z., Teng, J., Zheng, W., Ding, M., Huang, S., Xu, J., Yang, Y., Hong, W., Zhang, X., Feng, G., et al.: Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072 (2024)
- 59. Ye, Z., He, X., Liu, Q., Wang, Q., Wang, X., Wan, P., Zhang, D., Gai, K., Chen, Q., Luo, W.: Unic: Unified in-context video editing. arXiv preprint arXiv:2506.04216

(2025)

- 60. Yu, S., Liu, D., Ma, Z., Hong, Y., Zhou, Y., Tan, H., Chai, J., Bansal, M.: Veggie: Instructional editing and reasoning video concepts with grounded generation. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 15147–15158 (2025)
- 61. Yuan, S., He, X., Deng, Y., Ye, Y., Huang, J., Lin, B., Luo, J., Yuan, L.: Opens2vnexus: A detailed benchmark and million-scale dataset for subject-to-video generation. arXiv preprint arXiv:2505.20292 (2025)
- 62. Zhang, C., Zhang, C., Zhang, M., Kweon, I.S., Kim, J.: Text-to-image diffusion models in generative ai: A survey. arXiv preprint arXiv:2303.07909 (2023)
- 63. Zhang, L., Rao, A., Agrawala, M.: Adding conditional control to text-to-image diffusion models. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 3836–3847 (2023)

- 64. Zhang, L., Rao, A., Agrawala, M.: Scaling in-the-wild training for diffusion-based illumination harmonization and editing by imposing consistent light transport. In: The Thirteenth International Conference on Learning Representations (2025)
- 65. Zhang, Y., Yuan, Y., Song, Y., Wang, H., Liu, J.: Easycontrol: Adding efficient and flexible control for diffusion transformer. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 19513–19524 (2025)
- 66. Zheng, Z., Peng, X., Yang, T., Shen, C., Li, S., Liu, H., Zhou, Y., Li, T., You, Y.: Open-sora: Democratizing efficient video production for all. arXiv preprint arXiv:2412.20404 (2024)
- 67. Zi, B., Peng, W., Qi, X., Wang, J., Zhao, S., Xiao, R., Wong, K.F.: Minimax-remover: Taming bad noise helps video object removal. arXiv preprint arXiv:2505.24873 (2025)

### A More Experimental Details

#### A.1 More Dataset Details

For the consistent style transfer task, we use OmniConsistency [43] as the training dataset. We select three style subsets, 3D Chibi, Ghibli, and American Cartoon. These subsets contain 140, 100, and 124 pairs of original and styled images, specifically. For the rest editing tasks, we use GPT-5 to generate 250 editing prompts for each editing task. For each prompt, we adopt FLUX.1-dev to generate the source image, and Qwen-Image-Edit-2509 [50] to produce the target image according to the specified edit instruction.

For evaluation, we use VBench [19] prompts and generate five source videos per prompt with Wan2.1-T2V-1.3B [47], using seeds 0–4. For the resting editing tasks, the source videos are produced by mixing Wan2.1-T2V-1.3B [47] and Wan2.1-T2V-14B [47] with prompts from FiVE Bench [27], and the seeds are randomly drawn from 0–4. The number of evaluation samples is 100, 100, 100, 9, 10 for object rigid replacement, object non-rigid replacement, color alternation, object addition, and object removal tasks, respectively.

#### A.2 More Model Details

In this paper, we adopt Wan2.1-T2V-1.3B [47] as the base model, and introduce a pair of complementary 2D spatial attention module applied to each 3D full attention block in residual manner. Both 2D spatial attention blocks are initialized with the corresponding 3D attention module to ensure training stability, and their outputs are combined by subtraction.

To establish strong baselines, we adopt the end-to-end model Wan2.1-VACE1.3B [20], which is pretrained on a large-scale video dataset and provides strong controllability and editing performance. To enable VACE to handle unseen consistent style transfer tasks, we apply LoRA fine-tuning on the VACE branch using the corresponding style subsets of the OmniConsistency dataset [43], with the LoRA rank fixed to 32 for all styles. We also include the OmniConsistency method applied to the base video frame-by-frame for comparison. OmniConsistency method is based on FLUX.1-dev model. Here, we directly use the corresponding LoRA modules provided officially for each specific style, and the rank for each LoRA module is 128. Here, all source and generated target videos contain 81 frames with 480p resolution.

For other editing tasks, we use SDEdit [35], VidToMe [28], Pyramid-Edit [21, 27], and Wan-Edit [25, 27] as baselines for comparison. Here, SDEdit [35] and Wan-Edit [25,27] are based on the Wan2.1-T2V-1.3B model, and we use a denoising strength of 0.7 following the default configuration stated in FiVE Bench. VidToMe [28] is based on Stable-Diffusion-1.5 model [39], while Pyramid-Edit [21,27] is based on the Pyramid-Flow-miniFLUX model. All methods generate videos with 81 frames at 480p resolution unless otherwise specified. VidToMe [28] outputs 63 frames, and Pyramid-Edit [21,27] produces 384p videos, both following their official default settings.

#### A.3 More Training Details

All tasks are fine-tuned with LoRA under the DiffSynth framework. Except for the number of training epochs, all other training hyperparameters are kept consistent across tasks. Specifically, the LoRA rank is set to 32, and only the 2D spatial attention modules and FFN modules are fine-tuned, with the rest part of network remains frozen. Training is performed on three NVIDIA RTX 6000 Ada GPUs. The peak GPU memory is about 18 GiB, and each epoch will take 5 min for consistent style transfer task and 9 min for the rest editing tasks. More implementation details are provided in Table 4.

- Table 4: The hyper-parameters of training for all tasks. Here, the training epoch from left to right is for consistent style transfer, object rigid replacement, object non-rigid replacement, color alternation, object addition and object removal tasks.

Hyperparameter Value

Optimizer AdamW

Learning Rate 1e-4 Weight Decay 0.01 LoRA Rank 32

Scheduler ConstantLR LoRA Target Modules

SpaPos.q, k, v, o SpaNeg.q, k, v, o ffn.0, ffn.2 CFG Scale 1.0

Training Epoch 20, 10, 20, 10, 5, 2

- Table 5: The evaluation results of Ditto and our proposed method on VBench with consistent style transfer task. Here, Ditto has 14B model size and pre-trained with large-scale paired video dataset, while our proposed method is 1.3B and trained on 100-200 paired image data.

VBench Subject Background Temporal Motion

Method/Setting Model Size Training Sets

Color Consistency Consistency Flickering Smoothness

3D Chibi Style

Ditto 14B Ditto-1M (Video) 0.9749 0.9790 0.9965 0.9759 0.7948 Ours 1.3B OmniConsistency (Image) 0.9855 0.9806 0.9981 0.9907 0.9357

###### Ghibli Studio Style

Ditto 14B Ditto-1M (Video) 0.9823 0.9803 0.9958 0.9853 0.7903 Ours 1.3B OmniConsistency (Image) 0.9828 0.9802 0.9979 0.9900 0.9054

###### American Cartoon Style

Ditto 14B Ditto-1M (Video) 0.9831 0.9820 0.9952 0.9829 0.6755 Ours 1.3B OmniConsistency (Image) 0.9850 0.9819 0.9974 0.9884 0.8720

Source Video Ditto

Ours Source Video Ditto Ours

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

3DChibiStyle

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

Source Video Ditto

Ours Source Video Ditto Ours

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

GhibliStyle

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

Source Video Ditto

Ours Source Video Ditto Ours

[Figure 300]

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

AmericanCartoonStyle

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

- Fig. 7: The visualization results of Ditto and our proposed method on consistent style transfer tasks. Here, our proposed method achieves high consistency and style fidelity with only paired image data.

#### A.4 More Evaluation Details

For evaluation, we adopt established benchmarks to assess performance across a variety of video editing tasks. All test videos consist of 81 frames. For consistent style transfer task, we follow the official VBench protocol [19]. We generate five base videos for each prompt, and apply the style-transfer methods produce stylized outputs. We report the subset of VBench metrics that appropriately capture style-transfer behavior, including subject consistency, background consistency, temporal flickering, motion smoothness, and color. We further compute VLM-based scores using Qwen2.5-VL-7B-Instruct [2] to measure structural and motion consistency between the base and stylized videos, as well as style fidelity in the final outputs. For VLM evaluation, we use the following prompts to evaluate structural and motion consistency, and style fidelity: for structural consistency, we use: ‘Score the spatial and structural consistency between the two

videos with prompt. Give a single score from 0 to 100, where 0 means completely inconsistent and 100 means perfectly consistent. Output only the number, with no other text’; for motion consistency, we use: ‘Score the temporal and motion consistency between the two videos with prompt. Give a single score from 0 to 100, where 0 means completely inconsistent and 100 means perfectly consistent. Output only the number, with no other text’; for style fidelity, we use: ‘Evaluate the style transfer quality between the two videos with prompt. The target style is style. Give a single score from 0 to 100, where 0 means the style is completely failed and 100 means the style is perfectly transferred while maintaining visual coherence. Output only the number, with no other text’. Here, we set the fps as 15 to help model thoroughly review the input source and target videos.

For other editing tasks such as rigid and non-rigid object replacement, color modification, object addition, and object removal, we adopt FiVE-Bench and use its task prompts to produce base videos and corresponding edits. Editing performance is quantified using the FiVE-Acc metrics. To improve the reliability of this benchmark, we review and refine the question set used by FiVE-Bench, and we compute FiVE-Acc over entire videos instead of a small set of sampled frames, which may provide more stable and representative quantitative results.

### B More Experimental Results

#### B.1 Comparison with End-to-End Video Editing Model Ditto

To further validate the effectiveness of our proposed method, we compare it against Ditto [1] on the consistent style transfer task. Ditto is built on Wan2.1VACE-14B [20] and trained on 1M high-quality video pairs. We use the officially released ditto_global_style LoRA, which yields its strongest style-transfer performance. Following the VBench protocol [19], we generate one video per prompt. Quantitative results are reported in Table 5, and visual comparisons are in Fig. 7.

Both the quantitative results and the visualization results demonstrate that our proposed method, requiring only 100-200 image pairs for fine-tuning, has high consistency and style fidelity style transfer performance across different styles. From the results, Ditto [1] preserves motion coherence and temporal stability but often fails to maintain accurate style, color correspondence, and spatial alignment. For example, in the second American Cartoon example, Ditto introduces clear color shifts in the car (from black to blue), the hair of the man (from yellow to brown), and the clothes of the man (from black to blue). This failure also reflected by the color metric in Table 5.

#### B.2 Results on Arbitrary Style Transfer

In previous experiments, we focus on transferring a single style. To further demonstrate the scalability and generalization capability of our proposed method, we conduct multi-style mixed fine-tuning with style Pixel Art, Oil Painting, and Van Gogh. Here, we combine the subsets with those three styles from OmniConsistency dataset [43]. The total number of paired image data is 317. We fine-tune

Source Video Ours Source Video Ours

Source Video Ours

Source Video Ours

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

[Figure 327]

[Figure 328]

[Figure 329]

[Figure 330]

[Figure 331]

[Figure 332]

[Figure 333]

PixelArtStyle

[Figure 334]

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

[Figure 341]

Source Video Ours Source Video Ours

Source Video Ours

Source Video Ours

[Figure 342]

[Figure 343]

[Figure 344]

[Figure 345]

[Figure 346]

[Figure 347]

[Figure 348]

[Figure 349]

OilPaintingStyle

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

Source Video Ours Source Video Ours

Source Video Ours

Source Video Ours

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

VanGoghStyle

[Figure 382]

[Figure 383]

[Figure 384]

[Figure 385]

[Figure 386]

[Figure 387]

[Figure 388]

[Figure 389]

- Fig. 8: The visualization results of our proposed method on mixed style transfer task. Here, we fine-tune Wan2.1-T2V-1.3B model with mixed style image data. Our proposed method can handle multiple styles transfer at one time.

the Wan2.1-T2V-1.3B model [47] following the settings stated in Table 4. The total fine-tuning stage lasts for 20 epochs. The visualization results are shown in Fig. 8.

From the results, our method is able to handle multiple styles jointly while maintaining high consistency in both spatial and temporal dimensions. Notably, our proposed method performs high-quality style transformation and can also reliably distinguish and reproduce even closely related styles, such as Van Gogh and classical oil painting, without introducing cross-style interference.

#### B.3 Results on Depth-to-Video Task

To further assess the applicability of our proposed method and its capability to handle different types of control signals, we perform additional experiments on controllable video generation tasks. Here, we conduct experiments on the depth-to-video task.

Specifically, we first use GPT-5 to randomly generate 250 prompts, and then employ FLUX.1-dev to produce the corresponding images as the target. These targets are paired with depth maps obtained from Depth-Anything-V2 [56], forming the supervision signals required for depth-conditioned video generation. For fine-tuning, we follows the hyper-parameter settings demonstrated in Table 4, and the total number of epochs for fine-tuning is 20. The visualization

Depth Condition Source Video VACE Ours Depth Condition Source Video VACE Ours

[Figure 390]

[Figure 391]

[Figure 392]

[Figure 393]

[Figure 394]

[Figure 395]

[Figure 396]

[Figure 397]

[Figure 398]

[Figure 399]

[Figure 400]

[Figure 401]

[Figure 402]

[Figure 403]

[Figure 404]

[Figure 405]

[Figure 406]

[Figure 407]

[Figure 408]

[Figure 409]

[Figure 410]

[Figure 411]

[Figure 412]

[Figure 413]

- Fig. 9: The visualization results of Wan2.1-VACE-1.3B and our proposed method on depth-to-video tasks. Here, the source videos are generated by Wan2.2-T2V-A14B, and the depth maps are generated by Video Depth Anything.

[Figure 414]

[Figure 415]

[Figure 416]

[Figure 417]

[Figure 418]

[Figure 419]

[Figure 420]

[Figure 421]

[Figure 422]

[Figure 423]

[Figure 424]

[Figure 425]

- Fig. 10: Visualization results of our method on more challenging editing cases. The left three columns are the source videos, and the right columns are edited videos via our proposed method.

results are shown in Fig. 9. Here, the source videos are generated by Wan2.2T2V-A14B [47], and the corresponding depth maps are generated by Video Depth Anything [7]. The prompts of the source video are randomly selected from VBench [19]. All source videos have 81 frames and are in 480p resolution. Here, to validate the effectiveness of our proposed method, we employ Wan2.1VACE-1.3B model [20] as the baseline for comparison. Wan2.1-VACE-1.3B [20] is pre-trained on large-scale paired video dataset, and has outstanding performance in controllable video generation tasks.

From the results, our proposed method closely adheres to the input depth conditions, capturing fine-grained structural cues with high precision. In addition, the aesthetic quality and overall visual fidelity of the generated videos are comparable to those produced by the Wan2.1-VACE-1.3B model [20], which is trained on large-scale video datasets.

#### B.4 Results on Challenging Cases

To further demonstrate the effectiveness of our proposed method, we apply ViFeEdit on more challenging cases, e.g., motion blur and significant occlusion. The edited results are shown in Fig. 10. From the results, our proposed method performs well and still keeps coherent temporal consistency and stable structural preservation under challenging cases.

### C Conclusion

Our method opens a new direction for the video editing and control tasks, and it achieves high-quality and highly consistent performance while using only a small amount of easily obtainable paired image, without requiring large-scale paired video data and extensive training. The experimental results further show that our proposed method attains performance on par with, and in many cases exceeding, large-scale models trained on massive amounts of paired video data with heavy training budgets. Moreover, our proposed method exhibits strong versatility, supporting a wide range of tasks, from global editing to local editing, and even controllable generation, while consistently delivering high-quality outputs.

