# arXiv:2503.23284v1[cs.GR]30Mar2025

## SketchVideo: Sketch-based Video Generation and Editing

Feng-Lin Liu1,2 Hongbo Fu3 Xintao Wang4 Weicai Ye4 Pengfei Wan4 Di Zhang4 Lin Gao1,2*

1Beijing Key Laboratory of Mobile Computing and Pervasive Device, Institute of Computing Technology, Chinese Academy of Sciences 2University of Chinese Academy of Sciences 3Hong Kong University of Science and Technology 4Kuaishou Technology

[Figure 1]

Figure 1. Our method enables high-quality video generation (a) and editing (b) based on sketch and text inputs. (a) Top: With the same text prompt, different keyframe sketches lead to results with similar semantics but diverse sketch-faithful geometry. (a) Bottom: With the same sketches, varied text prompts yield diverse appearances. (b) Users can also edit real videos by drawing on keyframe sketches, with edits automatically propagated even when edited objects in original videos have translation and rotation.

### Abstract

Video generation and editing conditioned on text prompts or images have undergone significant advancements. However, challenges remain in accurately controlling global layout and geometry details solely by texts, and supporting motion control and local modification through images. In this paper, we aim to achieve sketch-based spatial and motion control for video generation and support fine-grained editing of real or synthetic videos. Based on the DiT video generation model, we propose a memory-efficient control structure with sketch control blocks that predict residual features of skipped DiT blocks. Sketches are drawn on one or two keyframes (at arbitrary time points) for easy interaction. To propagate such temporally sparse sketch conditions across all frames, we propose an inter-frame at-

*Corresponding Author: Lin Gao (gaolin@ict.ac.cn)

tention mechanism to analyze the relationship between the keyframes and each video frame. For sketch-based video editing, we design an additional video insertion module that maintains consistency between the newly edited content and the original video’s spatial feature and dynamic motion. During inference, we use latent fusion for the accurate preservation of unedited regions. Extensive experiments demonstrate that our SketchVideo achieves superior performance in controllable video generation and editing. Homepage and code: http://geometrylearning.com/SketchVideo/

### 1. Introduction

Diffusion-based text-to-image [12, 51, 53] and text-to-video [27, 35, 43, 73, 84] models advance significantly due to improvements in datasets [4, 42, 55, 61] and denoising network architectures [12, 51]. While text prompts effectively

describe high-level semantics, they lack control of scene layouts and geometric details. To address this, existing video generation methods [2, 25, 73] utilize images as additional conditions but raise the questions of how to generate input images and achieve detailed editing. Sketching serves as a user-friendly interaction tool to capture spatial content and shape details accurately. One or two sketches are already sufficient to convey desired scene structures and motion information for short video clips (around 6 seconds), which are the target of our and most existing video generation methods. However, using such sparse keyframe sketches presents several challenges, including reasonably completing the missing frames, improving memory efficiency, and addressing the limited size of video datasets.

A na¨ıve solution is to translate the input keyframe sketches into images and then utilize interpolation methods [16, 37, 70] for video generation. However, it is nontrivial to ensure consistency during keyframe sketchto-image generation, which significantly affects the video quality. This approach also struggles to generate extrapolation frames when applying conditions in intermediate frames rather than beginning and ending time points. Another possible approach is to utilize white placeholders to fill missing condition frames and directly apply ControlNet [77] into video models, similar to SparseCtrl [22]. However, this requires the same network to process both sketches and white placeholders simultaneously, while the pretrained blocks handle tasks far from this sparse propagation. Additionally, for DiT-based video frameworks [27, 84], the traditional strategy [11] that copies half of the pretrained model as a condition network easily causes the outof-memory issue.

To address these issues, we propose a novel sketch condition network specifically designed for the DiT-based video generation architecture (CogVideoX [73] in our work). Following ControlNet [11, 77], we employ a trainable copy of CogVideoX’s DiT block to process only the sketch inputs and generate control features. No white placeholder is processed to align with the pretrained weights and reduce learning complexity. To propagate these keyframe features, we design an inter-frame attention mechanism that captures the relationship between the control keyframes and all video frames. Our approach computes query and key features from noisy latent while extracting value features from sketch conditions. This design leverages frame-toframe similarity for control propagation. The above components consist of a single sketch control block. Instead of copying a half number of the pretrained blocks to construct the control network [11], we use only 5 sketch control blocks out of the 30 DiT blocks available in CogVideoX-

- 2b. We design a novel uniformly distributed skip structure to add the control signals to different levels of features in discrete blocks (0, 6, 12, 18, and 24), achieving effective

spatial control while improving memory efficiency. During training, an external image dataset is incorporated to solve the challenge of limited video data.

Beyond generation, interactive editing of real or synthetic videos further enhances creative flexibility. Existing methods [34, 40, 45, 71] achieve interesting text-based editing or effectively propagate single image editing into videos. Despite their effectiveness in appearance modification, they struggle with shape manipulation and object insertion, as they preserve the original temporal motion. Such information is missing for newly introduced content. Moreover, precisely identifying and preserving unedited regions for localized editing remains a challenge.

We propose a sketch-based editing method for detailed local modification. Rather than editing a single image and propagating changes, we directly construct an editing network based on our sketch control network. To analyze the relationship between edited regions and the original video, we incorporate a video insertion module that takes the original video with masked edited regions as inputs. The modified sketch control blocks generate residual features that capture temporally and spatially coherent contents in the edited areas. To accurately preserve unedited regions and achieve seamless fusion, the newly edited regions are blended with the original video in the latent space.

Extensive experiments demonstrate that our method outperforms existing approaches in video generation and editing. Our contributions are summarized as follows: 1) We propose SketchVideo, a novel sketch-based video generation and editing framework that enables detailed geometry control and manipulation using keyframe sketches, as shown in Fig. 1. 2) We design a sketch condition network that predicts skipped control features for the DiT framework, with an inter-frame attention mechanism to propagate one or two sketch conditions across the video. 3) We propose a video insertion module that analyzes the relationship between drawn sketches and original videos, utilizing a latent fusion strategy to preserve unedited regions accurately.

### 2. Related Work

Sketch-based Image Generation. GAN-based methods [21] have achieved great success in category-restricted sketch-to-image translation [13, 14, 50, 65, 75, 85]. Recently, diffusion-based text-to-image models [51, 53] handle general categories with conditional models like ControlNet [77], T2I-Adapter [41] and further advances [30, 46, 48, 76, 83]. Beyond U-Net, the DiT backbone [12] enables image generation, with PIXART-δ [11] utilizing the first half of the pretrained model’s blocks as a trainable network to predict corresponding control residual features. Video generation, however, introduces additional challenges. For ease of interaction, we expect sketches specified only for a sparse set of keyframes, making it difficult to generate

frames without sketch inputs. Additionally, video generation costs significantly higher memory resources, making methods [11, 77] that replicate half of the base model as a sketch encoder easily out of memory.

Diffusion-based Video Generation. VDM [26] pioneered diffusion-based video generation with a 3D UNet denoising network. To improve quality, subsequent works [2, 3, 9, 10, 23, 25, 33, 64] integrated temporal modules into text-to-image models [51] to enable text- and image-conditioned video synthesis. Considering the efficiency, the DiT architecture is further used in Sora [43] and open-source projects [35, 60, 84]. Despite these advancements, subtle flickering artifacts remain in the results. CogVideoX [27, 73] further proposes a 3D full attention that merges the spatial and temporal attention, facilitating the generation of long-duration and high-resolution videos.

Building on existing video generation models, various conditions have been introduced to control the generation, such as camera movement [24, 66, 72], subject identity [31, 68], key-point trajectory [36, 56, 81] and example motions [67, 79]. However, they often lack control over spatial layouts and geometric details. Some methods [15, 18, 80] address this by extending image-based ControlNet [77] to videos but require all-frame conditions that are tedious for sketch interaction. SparseCtrl [22] tackles this by using white images for completions. However, due to the restriction of its base model [23] and a simple network design, its results suffer from temporal flickering. Similar ideas have been applied to cartoon interpolation [70] and colorization [28] with line art as input, but their outputs are limited to cartoon style. Our method utilizes sparse inputs, including hand-drawn sketches on one or two keyframes, to generate temporally stable and realistic videos. Additionally, our method further supports the sketch-based detailed editing of existing videos.

Deep Learning-based Video Editing. Pioneer works achieve video editing by style transfer [29, 52], GAN inversion [39, 62], and layered representations [32]. The advent of diffusion models further provides new editing paradigms. One category of such methods leverages image generation models to achieve compelling editing results, using temporal consistency techniques such as layered representations [6, 44], cross-frame attention [5, 17, 19, 57] and pixel warping [71]. A second category of methods employ video generation models. These works propagate the edits applied on the first frame into the other frames [34, 45], or utilize an inpainting strategy to achieve text-based editing [82] and motion modification [40]. To achieve text-based large-scale shape modification while maintaining motion features, a space-time feature loss [74] is designed to guide the inference process. Our method moves beyond traditional text- or image-based editing approaches, allowing users to draw one or two keyframe sketches at arbitrary time points for inter-

active video editing. Additionally, our method effectively handles sketch-based shape manipulation and dynamic object insertion, which are challenging for previous works.

Controllable Attention Mechanism. Attention across frames is initially used to ensure temporal consistency in AnimateDiff [23] and subsequent video generation models. Subsequent works, such as VideoBooth [31] and StillMoving [8], utilize it to learn identity-aware features for customized video generation. For video editing, existing works [17, 20, 57] utilize cross-frame attention to capture the temporal motion of input videos, enabling effective propagation of editing operations. Our method applies this idea to pixel-aligned sketch-based video generation and propagates the spatial geometric conditions instead of identity customization. We use a new feature derivation strategy for enhanced spatial control, differentiating our approach from traditional cross-attention mechanisms.

### 3. Methodology

This section introduces our sketch-based video generation and editing framework. In Sec.3.1, we provide an overview of CogVideoX-2b [73], a pretrained text-to-video generation method, which we will use for sketch-based video generation in our work. In Sec.3.2, we describe our sketch condition network specifically designed for the DiT architecture, which contains sketch control blocks to predict residual features. Within each control block, an interframe attention mechanism is designed to propagate the input sketches. In Sec.3.3, we detail the editing framework, which incorporates a video insertion module and latent fusion to preserve the features of the original video.

#### 3.1. Preliminary

CogVideoX [73] is a text-to-video generation model that employs a 3D causal VAE and builds diffusion within the latent space. In CogVideoX-2b, the VAE model performs 8×8 spatial and 4× temporal downsampling, followed by a DiT architecture with 30 blocks for video generation. Within each block, a 3D full attention processes the concatenated text embeddings and patchified video latents, followed by a feed-forward layer to output the features. The 3D full attention merges the commonly used separate spatial and temporal attention [2, 23, 84] to improve the temporal coherence. The training objective of diffusion is:

0 ,y,ϵ ϵ − ϵθ(√α¯tx1:0 N + √1 − α¯tϵ, t, y)

2

, (1)

L(θ) := Et,x1:N

where t is sampled between 1 and T (denosing steps), ϵ is the random noise, y is text prompts, and x1:0 N is the video data with N frames. The v-prediction [54] and zero SNR [38] are utilized for the diffusion setting.

[Figure 2]

- Figure 2. Our framework for sketch-based video generation and editing. (a) Our sketch condition network for the DiT-based video generation architecture has a skip structure and five sketch control blocks that predict residual features. (b) For generation, features are extracted from temporally sparse input sketches and propagated through inter-frame attention. The input sketches are provided for one or two keyframes (the second sketch is shown as a dotted line). In the top left corner of (b), the prompt and timestep inputs are shown. (c) For editing, the same sketch control block (b) is utilized, with an additional video insertion module and video masks M1:N to analyze the relationship between edited and unedited regions. The 3D causal VAE is omitted to save space.

#### 3.2. Sketch-based Video Generation

Given text prompts and one or two keyframe sketches with corresponding time points t1,t2, our method generates video clips that respect the input text prompts and sketches. As shown in Fig. 2, we design a sketch condition network for effective control. The input sketches are encoded into the latent space by the pretrained VAE, followed by patchifying and time-aware positional embedding to generate sketch latents st

1,t2 0 .

Skip Residual Structure. In sketch-based image generation, methods like ControlNet [77] and PIXART-δ [11] copy the half of the base model as their sketch encoder to fully utilize the pretrained text-to-image models. However, applying this to video generation, as done in SparseCtrl [22], is memory inefficient because it requires adding half of the base model’s parameters. Unlike the U-Net architecture, the DiT network does not have an explicit en-

- coder and decoder. Therefore, the assumption in PIXARTδ [11] that the first half of blocks serve as the encoder can be improved.

Instead of borrowing local consecutive blocks of the base model as the encoder and predicting their residual features, we recognize that blocks at different depths process distinct feature levels, which should be considered for sketch control. As illustrated in Fig. 2 (a), we propose a novel skip residual structure that reduces the number of blocks while enabling effective sketch control and high-quality generation. Our sketch condition network contains 5 sketch control blocks, uniformly distributed across the pretrained gen-

eration network to predict residual features for blocks 0, 6, 12, 18, and 24 of the original video generation model. This structure efficiently integrates sketch control information into multiple feature levels, enhancing the analysis of input conditions and original semantic features.

Sketch Control Block. In the i-th sketch control block, the input consists of hidden video features h1:i N and sketch features st

i−1 , and the output is the residual features h1:i N and updated sketch features st

1,t2

1,t2

i . As shown in Fig. 2 (b), the sketch feature st

1,t2

i−1 is processed: st

i = FeedForward(st

1,t2

1,t2

i−1 ), (2) where the output sketch feature st

1,t2

i is used for sketch control propagation and as the input to the next sketch control block.

To propagate the sketch inputs, a direct approach [22] replaces missing sketches with white images and employs a trainable copy of the pretrained DiT block to predict residual features. However, as shown in Fig. 5, this leads to fuzzy details in challenging cases, as the network randomly processes both sketches and white images. This mixed input is far from the pretrained model’s task. To address the above issue, we employ a trainable copy of pretrained DiT block to process only the sketch inputs (no white placeholder), aligning with the pretrained weights and reducing learning difficulty. The resulting keyframe sketch features, denoted as ct

1,t2

i , are propagated to all frames through an inter-frame attention approach.

Inter-frame Attention. We utilize the input hidden fea-

[Figure 3]

- Figure 3. The sketch-based video generation results. Left: The input text prompts and sketches. Right: The video generation results. It can be seen that the generated results show high quality and good faithfulness with the input sketches. Our method can handle one/two keyframe sketch(es) at arbitrary user-specified time points (the frames corresponding to the input time points are highlighted by orange).

tures of all frames to calculate Q and the hidden features corresponding to the control frames to calculate K. During attention computation, this approach captures the internal relationship between all frames and control keyframes, allowing propagation of V (derived from the keyframe sketch features). Our inter-frame attention is distinct from typical cross-frame attention, which uses both K and V from control conditions; instead, we leverage the inter-frame similarity within the input noisy hidden video features and progressively insert the sketches’ spatial and temporal information. The output c1:i N is computed as: Attention(Q,K,V ) = Softmax(QK

T

d ) · V , with Q = Wq · h1:i N,K = Wk · ht

√

i ,V = Wv · ct

1,t2

1,t2

i , (3)

where Wq,Wk,Wv are trainable linear projection weights. The output of the inter-frame attention is fed into a feed-

forward layer to generate final residual features h1:i N.

Training Strategy. To train the sketch condition network, we employ a hybrid training strategy in two stages. In the first stage, to accelerate convergence and address the issue of limited video data, the network is trained both for image generation at arbitrary time points and video generation with one or two keyframe sketches (randomly selected from corresponding sketch videos). In the second stage, video data alone is used to improve the temporal coherence.

#### 3.3. Sketch-based Video Editing

For editing, given real or synthetic videos, users select one or two keyframes at arbitrary time points and modify the extracted sketches, with additional inputs of text prompts and masks M1:N that label regions to be edited for all the frames. Our method then generates realistic, local editing results. The input video is multiplied by the inverted masks

to remove information from the edited regions and is subsequently encoded to generate a masked video latent representation v01:N.

Video Insertion Module. For sketch-based editing, newly generated contents within the mask regions should be coherent with the original spatial and temporal features in the unedited regions. Thus, we design a video insertion module that analyzes the relationship between the input sketches and the original video. The video insertion module takes vi1:−N1 (input video latent or generated by a previous control block) as input and predicts the updated video features vi1:N, similar to the sketch generation process. Since video features are not temporally sparse, we use a trainable copy of CogVideoX-2b’s DiT block to directly generate video insertion features vi1:N. The sketch branch output c1:i N and video branch output vi1:N are multiplied by their respective masks and concatenated:

Concat( c1:i N ∗ M1:N, vi1:N ∗ M1:N), (4) which serves as inputs to the feed-forward layers, producing final residual features incorporating the original video and sketch control information. This design ensures seamless integration of new contents with the original videos, effectively propagating edits across frames for dynamic motion.

Training Strategy. Directly training the video editing network would lead to low fidelity with the input sketches, possibly because of the challenging interaction between the input sketches and videos. So we finetune it from the pretrained sketch condition network for generation and add the new video insertion module. The pretrained model already has good sketch fidelity and only requires learning video information. The network is trained in a self-supervised inpainting manner, with randomly generated masks to imitate real-world editing.

[Figure 4]

- Figure 4. The sketch-based video editing results. For each example, the text prompts and sketches are shown on the left. On the right, the input real videos are shown at the top, while the edited results with the control keyframe highlighted in orange are shown at the bottom. The editing region masks are manually provided by users, highlighted as orange boxes. Our method generates realistic local editing results.

Inference Latent Fusion. Although the original videos are encoded in the condition network, as shown in Fig. 8, fine details might be lost during editing. To address this, we propose a latent fusion approach at inference. Specifically, we apply DDIM inversion [58] to generate noisy latent codes of input videos across the inference steps. At Steps 25 and 49 (out of 50 total steps), the latent codes in the unedited regions are replaced with these inversion latent

- codes, ensuring better preservation of the original video’s details and improving the coherence of the edited regions.

### 4. Evaluation

#### 4.1. Implementation Details

We implement SketchVideo based on CogVideoX-2b [73], trained on a subset of OpenVid [42] and LAION [55] datasets, with paired sketches from [7]. Training uses 8 NVIDIA H800 GPUs with a batch size of 8 and gradient accumulation of 4. For generation, one or two keyframe sketches are randomly sampled from the video, with 10,000 steps for each training stage. For editing, randomly drawn masks are applied and trained for 20,000 steps. For ease of reading, the input text prompts are simplified in figures. Additional implementation details and full input texts are available in the supplementary material.

#### 4.2. Results

Our method generates high-quality videos from one or two keyframe sketches and text prompts. As shown in Fig. 3, our method can accurately control the object shape and scene layout, such as the rabbit’s pose (1st row) and the position of lakes and buildings (2nd row). Text-only inputs cannot achieve such detailed geometry control. Our method also achieves interesting dynamic motion interpolation and extrapolation, generating smooth and realistic transitions, as seen in the cat’s head shaking (3rd row) and the building’s movement (last row). This allows control over both spatial

[Figure 5]

Figure 5. The comparison results of sketch-based video generation. Text prompts are shown on the top. On the left, we show the input sketches and sketch-based image generation results by ControlNet [77]. On the right, we show the results of the compared approaches, including AMT [37], SparseCtrl [22], CtrlCogVideo [77], and ours. Our method produces better results, especially for the intermediate frames.

layout and dynamic motion.

Our method also supports sketch-based video editing. Users specify bounding boxes and draw sketches in those regions, and our method generates photorealistic, seamlessly integrated content. As shown in Fig. 4, the new contents blend well with unedited regions and have dynamic motions like hat/hair rotation with original objects (1st row) or interesting fish swimming (2nd row). Our method can handle diverse editing cases, including object insertion, component replacement, and object removal. The unedited regions are well-preserved thanks to our latent fusion approach.

#### 4.3. Comparison

For sketch-based generation, we compare our methods with three methods given the same keyframe sketches text prompts as inputs. For SparseCtrl [22], we use the official

[Figure 6]

Figure 6. The comparison results of sketch-based video editing. On the left, we show the drawn editing sketches (for the frames highlighted in orange), text prompts, and sketch-based image generation results by ControlNet [77]. On the right, we show the original videos and editing results by the compared methods, including InsV2V [17], AnyV2V [34], and ours. Our method generates the most realistic results and preserves unedited regions well.

pretrained model and extract sketches by HED [69]. It is trained on videos in the WebVid-10M dataset [1] instead of the OpenVid-1M dataset [42] used in our method. We extend SparseCtrl to CogVideoX-2b [73], using PIXARTδ [11] as the sketch condition encoder (with 5 DiT blocks same as our method) and white images to complete the missing condition frames. We also compare with an interpolation baseline, which uses ControlNet to translate twoframe sketches into images and then interpolates them with AMT [37]. For the sketch-base editing task, we compare ours with a text-based video editing method InsV2V [17] and a first-frame editing method AnyV2V [34]. We compare with additional methods [16, 20, 45, 70], as shown in supplemental material.

As shown in Fig. 5, ControlNet [77] translates sketches into realistic images, which, however, lack temporal consistency and vary with shading and content. The interpolation results of AMT [37] exhibit fuzzy details and artifacts. SparseCtrl [22], based on AnimateDiff [23], exhibits temporal flickering, such as the suddenly appearing tree and distorted tower top (see the orange boxes in the 2nd row). Extending SparseCtrl to CogVideoX-2b [73] (Ctrl-CogVideo) still generates fuzzy details in intermediate frames, possibly due to the CogVideoX-2b’s pretrained selfattention being designed for dense inputs instead of sparse sketches. Our method generates realistic videos with clear details and good temporal coherence, with even small details like the electric wires in the top-right corner accurately propagated.

For sketch-based editing (Fig. 6), InsV2V [17] generates interesting results with birds but lacks control over shape and geometry through text prompts alone. For imagebased video editing method, we utilize ControlNet to edit the first frame and then propagate editing into the video by AnyV2V [34]. However, since the motion is borrowed from

Methods LPIPS ↓ CLIP ↑ Fidelity Consistency Realism

AMT 29.17 96.12 3.13 3.51 3.57 SparseCtrl 44.85 96.48 2.79 2.94 2.83

Ctrl-CogVideo 32.23 98.04 2.86 2.47 2.50 Ours 27.56 98.31 1.21 1.08 1.11

- Table 1. The quantitative results of sketch-based video generation comparison. The LPIPS and CLIP numbers are scaled up 100×, with each cell colored to indicate the best .

Methods LPIPS ↓ CLIP ↑ PSNR↑ Fidelity Preservation Realism InsV2V 13.61 95.39 16.84 2.58 2.26 2.61

AnyV2V 11.92 93.47 13.68 2.35 2.69 2.34 Ours 9.74 98.34 36.48 1.07 1.05 1.04

- Table 2. The quantitative results of sketch-based video editing comparison. The LPIPS and CLIP numbers are scaled up 100×, with each cell colored to indicate the best .

the original video, AnyV2V struggles with new content, leading to fuzzy details and distortion in the bird, as well as changes in unedited regions due to ControlNet’s inability to retain the original features. In contrast, our method produces more realistic results with faithful representations of the sketch. More generation and editing results are available in the supplemental material and video.

We follow SparseCtrl [22] and use the LPIPS [78] metric to measure sketch faithfulness between the input sketches and those extracted from the corresponding frames of the generated videos. We use the CLIP [49] similarity to assess temporal coherence. For sketch-based generation, we test 200 random examples from OpenVid [42], using the first and last frame sketches and the corresponding text prompts in the dataset. As shown in Table 1, our method achieves the lowest LPIPS and highest CLIP scores, indicating its superior performance. For sketch-based editing, we use additional an MSE metric to measure unedited region preservation. We utilize 10 examples with hand-drawn sketches as input. In Table 2, our method outperforms existing methods across all metrics, demonstrating its superiority.

[Figure 7]

Figure 7. The ablation study results of sketch-based video generation. Our method generates more realistic and sketch-faithful results than the baselines.

#### 4.4. Ablation Study

We conduct an ablation study to evaluate the effectiveness of each key component. Due to the restriction of computing resources, all the models are trained with a batch size of 2. For sketch-based generation, removing inter-frame attention and concatenating sketches with frame features at the temporal dimension results in strange control with low

[Figure 8]

Figure 8. The ablation study results of sketch-based video editing. Our method generates realistic results faithful to sketches. We utilize heat maps (top right) to visualize the difference between the edited and original frames.

Metric w/o Inter-Frame Sketch K,V w/o Skip w/o Image Ours

LPIPS ↓ 36.33 32.59 31.91 34.58 30.79 CLIP ↑ 98.10 98.19 97.60 98.24 98.48 Metric w/o Video module w/o Pretrain w/o Latent Fusion Ours

LPIPS ↓ 9.31 12.60 9.77 9.74

CLIP ↑ 97.97 98.12 98.44 98.34 PSNR ↑ 33.61 36.05 31.69 36.48

Table 3. The qualitative results of ablation study for sketch-based generation and editing, with each cell colored to indicate the best and second best .

sketch faithfulness (Fig. 7 (a)). Similarly, if we replace the inter-frame attention with a typical cross-attention that uses sketch features for both Key and Value (b), the sketch fidelity is also degraded due to the lack of effective control propagation. If we remove the skip structure and predict residual features for the first 5 blocks (c), the realism of the generated results is negatively influenced due to less control ability. Removing image data during training (d) causes significant geometry mismatch with input sketches. Compared to these alternatives, our method produces the most realistic and faithful results. As shown in Table 3, our method has the lowest LPIPS values (supporting the best sketch fidelity) while the CLIP metric has similar results (human eyes cannot recognize temporal coherence difference).

In sketch-based video editing (Fig. 8), removing the video insertion module (a) and solely utilizing the sketch condition encoder results in good sketch fidelity but obvious inconsistency with unedited regions. Training the sketch condition encoder from scratch without generation pretraining (b) reduces the faithfulness to the input sketches in the edited regions, which is also validated by the high LPIPS in Table 3. We use heat maps to reveal differences between edited and original frames. Removing the latent fusion strategy changes unedited region details (c), such as mountains and waves, as shown in the heat maps. The low PSNR value in unedited regions also supports this, as shown in Table 3. Our method maintains strong sketch fidelity, consistency, and unedited region preservation.

#### 4.5. User Study

We conducted a user study to validate that our method outperforms existing approaches. For video generation, we randomly selected 10 examples from the test set in Sec. 4.3. In the questionnaire, participants were shown the results of different methods in random order. 20 participants ranked the methods results by three criteria: Sketch Fidelity, Temporal Consistency, and Video Realism. The rankings provided by the participants were used as scores. As shown in Table 1, our method outperforms existing methods in all criteria, demonstrating its superior performance. For sketchbased editing, we used the same 10 examples from the test set in Sec. 4.3. The same participants ranked the methods based on three criteria: Sketch Fidelity, Unedited Region Preservation, and Video Realism. As shown in Table 2, our method also outperforms existing approaches in editing.

### 5. Conclusion and Discussion

We presented SketchVideo, a unified method for sketchbased video generation and editing. For generation, we proposed a sketch condition network that predicts residual features for skipped DiT blocks of the base models to save the memory and achieve effective control. An inter-frame attention is further proposed to propagate the keyframe sketches, achieving interesting motion interpolation or extrapolation. We also introduce a hybrid image and video training strategy. For editing, we incorporate a video insertion module to ensure the newly generated content is spatially and temporally coherent with the original video. During inference, a latent fusion approach preserves unedited regions accurately.

Limitation and Future Work. While our method generates high-quality videos, its capabilities are still limited by the pretrained text-to-video model. Enhancing performance with more powerful pretrained models and generating long videos instead of video clips is a potential avenue. Additionally, similar to image generation, our method struggles with too complex scenarios, such as human hands and interaction between multiple objects. Incorporating 3D priors [59, 63, 86] like SMPL-X [47] helps address these issues in human scenarios. Moreover, while our method focuses on geometry control, exploring appearance customization through tools like color strokes presents an intriguing area for future research.

### 6. Acknowledgments

This work was sponsored by CCF-Kuaishou Large Model Explorer Fund, Beijing Municipal Science and Technology Commission (No. Z231100005923031), Innovation Funding of ICT, CAS (No. E461020) and National Natural Science Foundation of China (No. 62322210). The authors would like to acknowledge

the Nanjing Institute of InforSuperBahn OneAiNexus for providing the training and evaluation platform.

### References

- [1] Max Bain, Arsha Nagrani, G¨ul Varol, and Andrew Zisserman. Frozen in time: A joint video and image encoder for end-to-end retrieval. In Int. Conf. Comput. Vis., pages 1708– 1718, 2021. 7
- [2] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, Varun Jampani, and Robin Rombach. Stable video diffusion: Scaling latent video diffusion models to large datasets. CoRR, abs/2311.15127, 2023. 2, 3
- [3] Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. In IEEE Conf. Comput. Vis. Pattern Recog., pages 22563–22575, 2023. 3
- [4] Minwoo Byeon, Beomhee Park, Haecheon Kim, Sungjun Lee, Woonhyuk Baek, and Saehoon Kim. Coyo-700m: Image-text pair dataset. https://github.com/ kakaobrain/coyo-dataset, 2022. 1
- [5] Duygu Ceylan, Chun-Hao Paul Huang, and Niloy J. Mitra. Pix2video: Video editing using image diffusion. In Int. Conf. Comput. Vis., pages 23149–23160, 2023. 3
- [6] Wenhao Chai, Xun Guo, Gaoang Wang, and Yan Lu. Stablevideo: Text-driven consistency-aware diffusion video editing. In Int. Conf. Comput. Vis., pages 22983–22993, 2023. 3
- [7] Caroline Chan, Fr´edo Durand, and Phillip Isola. Learning to generate line drawings that convey geometry and semantics. In IEEE Conf. Comput. Vis. Pattern Recog., pages 7915– 7925, 2022. 6
- [8] Hila Chefer, Shiran Zada, Roni Paiss, Ariel Ephrat, Omer Tov, Michael Rubinstein, Lior Wolf, Tali Dekel, Tomer Michaeli, and Inbar Mosseri. Still-moving: Customized video generation without customized video data. ACM Trans. Graph., 43(6):244:1–244:11, 2024. 3
- [9] Haoxin Chen, Menghan Xia, Yingqing He, Yong Zhang, Xiaodong Cun, Shaoshu Yang, Jinbo Xing, Yaofang Liu, Qifeng Chen, Xintao Wang, Chao Weng, and Ying Shan. Videocrafter1: Open diffusion models for high-quality video generation. CoRR, abs/2310.19512, 2023. 3
- [10] Haoxin Chen, Yong Zhang, Xiaodong Cun, Menghan Xia, Xintao Wang, Chao Weng, and Ying Shan. Videocrafter2: Overcoming data limitations for high-quality video diffusion models. In IEEE Conf. Comput. Vis. Pattern Recog., pages 7310–7320, 2024. 3
- [11] Junsong Chen, Yue Wu, Simian Luo, Enze Xie, Sayak Paul, Ping Luo, Hang Zhao, and Zhenguo Li. Pixart-δ: Fast and controllable image generation with latent consistency models. CoRR, abs/2401.05252, 2024. 2, 3, 4, 7
- [12] Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Zhongdao Wang, James T. Kwok, Ping Luo, Huchuan Lu, and Zhenguo Li. Pixart-α: Fast training of diffusion

- transformer for photorealistic text-to-image synthesis. In Int. Conf. Learn. Represent., 2024. 1, 2
- [13] Shu-Yu Chen, Wanchao Su, Lin Gao, Shihong Xia, and Hongbo Fu. Deepfacedrawing: deep generation of face images from sketches. ACM Trans. Graph., 39(4):72, 2020. 2
- [14] Shu-Yu Chen, Feng-Lin Liu, Yu-Kun Lai, Paul L. Rosin, Chunpeng Li, Hongbo Fu, and Lin Gao. Deepfaceediting: deep face generation and editing with disentangled geometry and appearance control. ACM Trans. Graph., 40(4):90:1– 90:15, 2021. 2
- [15] Weifeng Chen, Jie Wu, Pan Xie, Hefeng Wu, Jiashi Li, Xin Xia, Xuefeng Xiao, and Liang Lin. Control-a-video: Controllable text-to-video generation with diffusion models,

2023. 3

- [16] Xinyuan Chen, Yaohui Wang, Lingjun Zhang, Shaobin Zhuang, Xin Ma, Jiashuo Yu, Yali Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. Seine: Short-to-long video diffusion model for generative transition and prediction. In Int. Conf. Learn. Represent., 2023. 2, 7
- [17] Jiaxin Cheng, Tianjun Xiao, and Tong He. Consistent videoto-video transfer using synthetic dataset. In Int. Conf. Learn. Represent., 2024. 3, 7
- [18] Ernie Chu, Shuo-Yen Lin, and Jun-Cheng Chen. Video controlnet: Towards temporally consistent synthetic-to-real video translation using conditional image diffusion models. CoRR, abs/2305.19193, 2023. 3
- [19] Nathaniel Cohen, Vladimir Kulikov, Matan Kleiner, Inbar Huberman-Spiegelglas, and Tomer Michaeli. Slicedit: Zeroshot video editing with text-to-image diffusion models using spatio-temporal slices. arXiv preprint arXiv:2405.12211,

2024. 3

- [20] Michal Geyer, Omer Bar-Tal, Shai Bagon, and Tali Dekel. Tokenflow: Consistent diffusion features for consistent video editing. In Int. Conf. Learn. Represent., 2024. 3, 7
- [21] Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron C. Courville, and Yoshua Bengio. Generative adversarial networks. CoRR, abs/1406.2661, 2014. 2
- [22] Yuwei Guo, Ceyuan Yang, Anyi Rao, Maneesh Agrawala, Dahua Lin, and Bo Dai. Sparsectrl: Adding sparse controls to text-to-video diffusion models. In Eur. Conf. Comput. Vis., pages 330–348, 2024. 2, 3, 4, 6, 7
- [23] Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized text-toimage diffusion models without specific tuning. In Int. Conf. Learn. Represent., 2024. 3, 7
- [24] Hao He, Yinghao Xu, Yuwei Guo, Gordon Wetzstein, Bo Dai, Hongsheng Li, and Ceyuan Yang. Cameractrl: Enabling camera control for text-to-video generation. CoRR, abs/2404.02101, 2024. 3
- [25] Yingqing He, Tianyu Yang, Yong Zhang, Ying Shan, and Qifeng Chen. Latent video diffusion models for highfidelity video generation with arbitrary lengths. CoRR, abs/2211.13221, 2022. 2, 3
- [26] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video dif-

- fusion models. Adv. Neural Inform. Process. Syst., 35:8633– 8646, 2022. 3
- [27] Wenyi Hong, Ming Ding, Wendi Zheng, Xinghan Liu, and Jie Tang. Cogvideo: Large-scale pretraining for text-to-video generation via transformers. arXiv preprint arXiv:2205.15868, 2022. 1, 2, 3
- [28] Zhitong Huang, Mohan Zhang, and Jing Liao. LVCD: reference-based lineart video colorization with diffusion models. CoRR, abs/2409.12960, 2024. 3
- [29] Ondrej Jamriska, S´arka Sochorov´a, Ondrej Texler, Michal Luk´ac, Jakub Fiser, Jingwan Lu, Eli Shechtman, and Daniel S´ykora. Stylizing video by example. ACM Trans. Graph., 38

(4):107:1–107:11, 2019. 3

- [30] Rui Jiang, Guang-Cong Zheng, Teng Li, Tian-Rui Yang, Jing-Dong Wang, and Xi Li. A survey of multimodal controllable diffusion models. Journal of Computer Science and Technology, 39(3):509–541, 2024. 2
- [31] Yuming Jiang, Tianxing Wu, Shuai Yang, Chenyang Si, Dahua Lin, Yu Qiao, Chen Change Loy, and Ziwei Liu. Videobooth: Diffusion-based video generation with image prompts. In IEEE Conf. Comput. Vis. Pattern Recog., pages 6689–6700, 2024. 3
- [32] Yoni Kasten, Dolev Ofri, Oliver Wang, and Tali Dekel. Layered neural atlases for consistent video editing. ACM Trans. Graph., 40(6):210:1–210:12, 2021. 3
- [33] Levon Khachatryan, Andranik Movsisyan, Vahram Tadevosyan, Roberto Henschel, Zhangyang Wang, Shant Navasardyan, and Humphrey Shi. Text2video-zero: Textto-image diffusion models are zero-shot video generators. In Int. Conf. Comput. Vis., pages 15908–15918, 2023. 3
- [34] Max Ku, Cong Wei, Weiming Ren, Harry Yang, and Wenhu Chen. Anyv2v: A tuning-free framework for any video-tovideo editing tasks. arXiv preprint arXiv:2403.14468, 2024. 2, 3, 7
- [35] PKU-Yuan Lab and Tuzhan AI etc. Open-sora-plan, 2024. 1, 3
- [36] Yaowei Li, Xintao Wang, Zhaoyang Zhang, Zhouxia Wang, Ziyang Yuan, Liangbin Xie, Yuexian Zou, and Ying Shan. Image conductor: Precision control for interactive video synthesis. CoRR, abs/2406.15339, 2024. 3
- [37] Zhen Li, Zuo-Liang Zhu, Ling-Hao Han, Qibin Hou, ChunLe Guo, and Ming-Ming Cheng. Amt: All-pairs multi-field transforms for efficient frame interpolation. In IEEE Conf. Comput. Vis. Pattern Recog., 2023. 2, 6, 7
- [38] Shanchuan Lin, Bingchen Liu, Jiashi Li, and Xiao Yang. Common diffusion noise schedules and sample steps are flawed. In IEEE/CVF Winter Conference on Applications of Computer Vision, WACV, pages 5392–5399, 2024. 3
- [39] Feng-Lin Liu, Shu-Yu Chen, Yu-Kun Lai, Chunpeng Li, Yue-Ren Jiang, Hongbo Fu, and Lin Gao. Deepfacevideoediting: sketch-based deep editing of face videos. ACM Trans. Graph., 41(4):167:1–167:16, 2022. 3
- [40] Chong Mou, Mingdeng Cao, Xintao Wang, Zhaoyang Zhang, Ying Shan, and Jian Zhang. Revideo: Remake a video with motion and content control. CoRR, abs/2405.13865, 2024. 2, 3

- [41] Chong Mou, Xintao Wang, Liangbin Xie, Yanze Wu, Jian Zhang, Zhongang Qi, and Ying Shan. T2i-adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models. In AAAI, pages 4296–4304, 2024. 2
- [42] Kepan Nan, Rui Xie, Penghao Zhou, Tiehan Fan, Zhenheng Yang, Zhijie Chen, Xiang Li, Jian Yang, and Ying Tai. Openvid-1m: A large-scale high-quality dataset for text-tovideo generation. CoRR, abs/2407.02371, 2024. 1, 6, 7
- [43] OpenAI. Sora overview:https://openai.com/index/sora/,

2024. 1, 3

- [44] Hao Ouyang, Qiuyu Wang, Yuxi Xiao, Qingyan Bai, Juntao Zhang, Kecheng Zheng, Xiaowei Zhou, Qifeng Chen, and Qifeng Chen. Codef: Content deformation fields for temporally consistent video processing. In IEEE Conf. Comput. Vis. Pattern Recog., pages 8089–8099, 2024. 3
- [45] Wenqi Ouyang, Yi Dong, Lei Yang, Jianlou Si, and Xingang Pan. I2vedit: First-frame-guided video editing via image-tovideo diffusion models. CoRR, abs/2405.16537, 2024. 2, 3, 7
- [46] Gaurav Parmar, Taesung Park, Srinivasa Narasimhan, and Jun-Yan Zhu. One-step image translation with text-to-image models. CoRR, abs/2403.12036, 2024. 2
- [47] Georgios Pavlakos, Vasileios Choutas, Nima Ghorbani, Timo Bolkart, Ahmed A. A. Osman, Dimitrios Tzionas, and Michael J. Black. Expressive body capture: 3d hands, face, and body from a single image. In IEEE Conf. Comput. Vis. Pattern Recog., 2019. 8
- [48] Bohao Peng, Jian Wang, Yuechen Zhang, Wenbo Li, MingChang Yang, and Jiaya Jia. Controlnext: Powerful and efficient control for image and video generation. CoRR, abs/2408.06070, 2024. 2
- [49] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In ICML, pages 8748–8763, 2021. 7
- [50] Elad Richardson, Yuval Alaluf, Or Patashnik, Yotam Nitzan, Yaniv Azar, Stav Shapiro, and Daniel Cohen-Or. Encoding in style: A stylegan encoder for image-to-image translation. In IEEE Conf. Comput. Vis. Pattern Recog., pages 2287–2296,

2021. 2

- [51] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In IEEE Conf. Comput. Vis. Pattern Recog., pages 10674–10685, 2022. 1, 2, 3
- [52] Manuel Ruder, Alexey Dosovitskiy, and Thomas Brox. Artistic style transfer for videos and spherical images. Int. J. Comput. Vis., 126(11):1199–1219, 2018. 3
- [53] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L. Denton, Seyed Kamyar Seyed Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, Jonathan Ho, David J. Fleet, and Mohammad Norouzi. Photorealistic text-to-image diffusion models with deep language understanding. In Adv. Neural Inform. Process. Syst., 2022. 1, 2

- [54] Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. In Int. Conf. Learn. Represent., 2022. 3
- [55] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, Patrick Schramowski, Srivatsa Kundurthy, Katherine Crowson, Ludwig Schmidt, Robert Kaczmarczyk, and Jenia Jitsev. LAION-5B: an open large-scale dataset for training next generation image-text models. In Adv. Neural Inform. Process. Syst., 2022. 1, 6
- [56] Xiaoyu Shi, Zhaoyang Huang, Fu-Yun Wang, Weikang Bian, Dasong Li, Yi Zhang, Manyuan Zhang, Ka Chun Cheung, Simon See, Hongwei Qin, Jifeng Dai, and Hongsheng Li. Motion-i2v: Consistent and controllable image-to-video generation with explicit motion modeling. In ACM SIGGRAPH, page 111. ACM, 2024. 3
- [57] Uriel Singer, Amit Zohar, Yuval Kirstain, Shelly Sheynin, Adam Polyak, Devi Parikh, and Yaniv Taigman. Video editing via factorized diffusion distillation. CoRR, abs/2403.09334, 2024. 3
- [58] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In Int. Conf. Learn. Represent., 2021. 6
- [59] Jia-Mu Sun, Tong Wu, and Lin Gao. Recent advances in implicit representation-based 3d shape generation. Vis. Intell., 2(1), 2024. 8
- [60] Vchitect Team and Shanghai Artificial Intelligence Laboratory. Vchitect-2.0: Parallel transformer for scaling up video diffusion models, 2024. 3
- [61] Zheng Tianpeng, Chen Yanxiang, Wen Xinzhe, Li Yancheng, and Wang Zhiyuan. Research on diffusion model generated video datasets and detection benchmarks. Journal of Image and Graphics, pages 1–13, 2024. 1
- [62] Rotem Tzaban, Ron Mokady, Rinon Gal, Amit Bermano, and Daniel Cohen-Or. Stitch it in time: Gan-based facial editing of real videos. In SIGGRAPH Asia Conference Papers, pages 29:1–29:9. ACM, 2022. 3
- [63] Guangming Wang, Lei Pan, Songyou Peng, Shaohui Liu, Chenfeng Xu, Yanzi Miao, Wei Zhan, Masayoshi Tomizuka, Marc Pollefeys, and Hesheng Wang. Nerf in robotics: A survey. CoRR, abs/2405.01333, 2024. 8
- [64] Jiuniu Wang, Hangjie Yuan, Dayou Chen, Yingya Zhang, Xiang Wang, and Shiwei Zhang. Modelscope text-to-video technical report. CoRR, abs/2308.06571, 2023. 3
- [65] Ting-Chun Wang, Ming-Yu Liu, Jun-Yan Zhu, Andrew Tao, Jan Kautz, and Bryan Catanzaro. High-resolution image synthesis and semantic manipulation with conditional gans. In IEEE Conf. Comput. Vis. Pattern Recog., pages 8798–8807,

2018. 2

- [66] Zhouxia Wang, Ziyang Yuan, Xintao Wang, Yaowei Li, Tianshui Chen, Menghan Xia, Ping Luo, and Ying Shan. Motionctrl: A unified and flexible motion controller for video generation. In ACM SIGGRAPH, page 114, 2024. 3
- [67] Jianzong Wu, Xiangtai Li, Yanhong Zeng, Jiangning Zhang, Qianyu Zhou, Yining Li, Yunhai Tong, and Kai Chen. Motionbooth: Motion-aware customized text-to-video generation. CoRR, abs/2406.17758, 2024. 3

- [68] Tao Wu, Yong Zhang, Xintao Wang, Xianpan Zhou, Guangcong Zheng, Zhongang Qi, Ying Shan, and Xi Li. Customcrafter: Customized video generation with preserving motion and concept composition abilities. CoRR, abs/2408.13239, 2024. 3
- [69] Saining Xie and Zhuowen Tu. Holistically-nested edge detection. In Int. Conf. Comput. Vis., pages 1395–1403, 2015. 7
- [70] Jinbo Xing, Hanyuan Liu, Menghan Xia, Yong Zhang, Xintao Wang, Ying Shan, and Tien-Tsin Wong. Tooncrafter: Generative cartoon interpolation. CoRR, abs/2405.17933,

2024. 2, 3, 7

- [71] Shuai Yang, Yifan Zhou, Ziwei Liu, and Chen Change Loy. Rerender A video: Zero-shot text-guided video-to-video translation. In SIGGRAPH Asia Conference Papers, pages 95:1–95:11. ACM, 2023. 2, 3
- [72] Shiyuan Yang, Liang Hou, Haibin Huang, Chongyang Ma, Pengfei Wan, Di Zhang, Xiaodong Chen, and Jing Liao. Direct-a-video: Customized video generation with userdirected camera movement and object motion. In ACM SIGGRAPH, page 113, 2024. 3
- [73] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024. 1, 2, 3, 6, 7
- [74] Danah Yatim, Rafail Fridman, Omer Bar-Tal, Yoni Kasten, and Tali Dekel. Space-time diffusion features for zero-shot text-driven motion transfer. In IEEE Conf. Comput. Vis. Pattern Recog., pages 8466–8476, 2024. 3
- [75] Chen Yuan, Zhao Yang, Zhang Xiaojuan, and Liu Xiaoping. Sketch colorization with finite color space prior. Journal of Image and Graphics, 29(4):978–988, 2024. 2
- [76] Liang Yuan, Dingkun Yan, Suguru Saito, and Issei Fujishiro. Diffmat: Latent diffusion models for image-guided material generation. Visual Informatics, 8(1):6–14, 2024. 2
- [77] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Int. Conf. Comput. Vis., pages 3813–3824, 2023. 2, 3, 4, 6, 7
- [78] Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In IEEE Conf. Comput. Vis. Pattern Recog., pages 586–595, 2018. 7
- [79] Yuxin Zhang, Fan Tang, Nisha Huang, Haibin Huang, Chongyang Ma, Weiming Dong, and Changsheng Xu. Motioncrafter: One-shot motion customization of diffusion models. CoRR, abs/2312.05288, 2023. 3
- [80] Yabo Zhang, Yuxiang Wei, Dongsheng Jiang, Xiaopeng Zhang, Wangmeng Zuo, and Qi Tian. Controlvideo: Training-free controllable text-to-video generation. In Int. Conf. Learn. Represent., 2024. 3
- [81] Zhenghao Zhang, Junchao Liao, Menghao Li, Long Qin, and Weizhi Wang. Tora: Trajectory-oriented diffusion transformer for video generation. CoRR, abs/2407.21705, 2024. 3
- [82] Zhixing Zhang, Bichen Wu, Xiaoyan Wang, Yaqiao Luo, Luxin Zhang, Yinan Zhao, Peter Vajda, Dimitris N. Metaxas,

- and Licheng Yu. AVID: any-length video inpainting with diffusion model. In IEEE Conf. Comput. Vis. Pattern Recog., pages 7162–7172, 2024. 3
- [83] Shihao Zhao, Dongdong Chen, Yen-Chun Chen, Jianmin Bao, Shaozhe Hao, Lu Yuan, and Kwan-Yee K. Wong. Uni-controlnet: All-in-one control to text-to-image diffusion models. In Adv. Neural Inform. Process. Syst., 2023. 2
- [84] Zangwei Zheng, Xiangyu Peng, Tianji Yang, Chenhui Shen, Shenggui Li, Hongxin Liu, Yukun Zhou, Tianyi Li, and Yang You. Open-sora: Democratizing efficient video production for all, 2024. 1, 2, 3
- [85] Jun-Yan Zhu, Taesung Park, Phillip Isola, and Alexei A. Efros. Unpaired image-to-image translation using cycleconsistent adversarial networks. In Int. Conf. Comput. Vis., pages 2242–2251, 2017. 2
- [86] Siting Zhu, Guangming Wang, Dezhi Kong, and Hesheng Wang. 3d gaussian splatting in robotics: A survey. CoRR, abs/2410.12262, 2024. 8

# arXiv:2503.23284v1[cs.GR]30Mar2025

## SketchVideo: Sketch-based Video Generation and Editing Supplementary Material

Feng-Lin Liu1,2 Hongbo Fu3 Xintao Wang4 Weicai Ye4 Pengfei Wan4 Di Zhang4 Lin Gao1,2*

1Beijing Key Laboratory of Mobile Computing and Pervasive Device, Institute of Computing Technology, Chinese Academy of Sciences 2University of Chinese Academy of Sciences 3Hong Kong University of Science and Technology 4Kuaishou Technology

In this supplemental material, we provide additional implementation details (Sec. 1), showcase results of sketchbased video generation (Sec. 2) and editing (Sec. 3), and the results of consecutive generation and editing (Sec. 4). We also present additional ablation study (Sec. 5), memory and time analysis (Sec. 6), comparison results (Sec. 7), attention map visualization (Sec. 8), and failure cases (Sec. 9). We discuss ethical concerns in Sec. 10 and show the full text prompts of all examples in Sec. 11.

### 1. Implementation Details

Sketch-based Generation. During training, we use a learning rate of 1e-5 and employ mixed precision (fp16). We optimize the sketch-conditioned network using the AdamW optimizer [10], while fixing the weights of the pretrained CogVideoX-2b [6] model. Similar to ControlNet [19], we initialize the weights of the DiT block with those from CogVideoX-2b. In the hybrid training stage (image and video), each batch consists solely of images or videos, with one or two keyframe sketches conditioned on random time points. The training dataset includes 53w video clips and 90w images, with the labeled text prompts from OpenVid [11] and LAION [13].

During inference, generated videos have a resolution of 720×480 and 8 fps, spanning 6 seconds with 49 frames. The classifier-free guidance scale is set to 10.0. Super-resolution [14] and frame interpolation [7] can be optionally applied to improve the resolution and frame rate, as in CogVideo. All the results shown in the paper and supplemental material do not use them.

Sketch-based Editing. We initialize the editing network using the weights from the sketch-based generation model. During training, the rectangle masks have random position, height, and weight. Mask movement strategies include: 1) fixed position, 2) linear interpolation between two random positions, and 3) motion based on the mean optical flow of the pixels within the box. The masks are directly multiplied with the input videos at the pixel level to remove the information in the edited regions.

During inference, the classifier-free guidance scale is set to 20.0. Users may manually adjust mask positions through interpolation or dynamic movement using optical flow. The masks are further downsampled and merged to achieve latent fusion. The 3D VAE in CogVideoX-2b performs temporally 8×8 spatial and 4× temporal downsampling. So,

we also downsample the masks by 8×8 to match the latent code resolution and combine the masks of four frames corresponding to one latent code. The masks are shown with orange boxes in the original videos in Fig. 7 and 8.

### 2. Sketch-based Generation Results

We show additional sketch-based video generation results to validate the effectiveness of our method. All results in this section are generated from hand-drawn sketches rather than synthetic ones. Fig. 1 shows the video generation results using a single keyframe sketch as input, conditioned on different time points (0s, 1.5s, 3s, 4.5s, and 6s). Our method generates realistic video results across diverse categories. Fig. 2 shows the results from the same sketch and text prompts, but with varying time points (0s, 1.5s, 3s, 4.5s, and 6s). These results exhibit fidelity to the sketch at the specified time point while introducing diverse motion in other frames.

In Fig. 3, we present additional video generation results from two keyframe sketches. Our method generates realistic videos with detailed control of geometry and motion. Fig. 4 shows the results with the same two keyframe sketches and text prompts, but varying time points. It can be seen that our method generates interesting interpolation and extrapolation results.

Our method uses the sketch to define geometry and the text prompt to specify appearance. As shown in Fig. 5, given the identical text but varying sketches, our method generates results of similar objects but with different geometry components and layouts. As shown in Fig. 6, our method generates diverse results from identical sketches with different text prompts, varying the appearance while maintaining the same structure.

### 3. Sketch-based Editing Results

We show additional results of sketch-base video editing in Fig. 7. Our method effectively edits object components within the video, such as modifying a person’s face or changing the roof of a castle. Our method also supports editing based on two keyframe sketches at different time points, controlling the motion of the newly generated objects (e.g., the transformation of a hat or the posing of a fox).

[Figure 9]

- Figure 1. Video generation results using one keyframe sketch. Given the text prompts (a) and sketches (b), our method generates realistic video results (c). The frames corresponding to the given time points are highlighted in orange boxes.

[Figure 10]

- Figure 2. Video generation results using the same sketch and text inputs (a) with different time points. The frames at 0s, 1.5s, 3s, 4.5s, and 6s (from top to bottom) are marked with orange boxes. The results (b) have good faithfulness to the sketch at given time points, while exhibiting diverse motion in other frames.

### 4. Consecutive Editing Results

the content (Editing 1). Additional edits (Editing 2) can be made iteratively without degrading the quality of the generated video. This showcases the detailed control our method offers for video customization.

Our method allows for consecutive generation and editing of videos. As shown in Fig. 8, given an initial sketch and a text prompt, we first generate a realistic video (Generation 1). The generated video then serves as input for further editing, using a new text prompt and a new sketch to modify

[Figure 11]

- Figure 3. Video generation results using two keyframe sketches. Given the text prompts (a) and two sketches (b), our method generates realistic video results (c). The frames corresponding to the given time points are highlighted in orange and green boxes.

[Figure 12]

- Figure 4. Video generation results (b) using the same two keyframe sketches and text inputs (a) at diverse time points. Each column in

- (b) represents frames at 0s, 1.5s, 3s, 4.5s, and 6s (from left to right). The frames corresponding to the given time points are highlighted in orange and green boxes.

- 5. More Ablation Study

Metric 1-5 13-17 26-30 Ours LPIPS↓ 33.37 38.57 50.34 32.47

During the editing process, we employ a latent fusion strategy to seamlessly fuse the newly generated regions with the original video content. An alternative approach involves directly fusing the generated video with the input video in pixel level. As shown in Fig. 9 (b), this baseline approach results in obvious seams along the boundaries of the edited regions. In contrast, our method ensures a smooth fusion of the edited and unedited regions by accurately inserting information from the unedited regions during the denoising process. And the combination in latent space followed with VAE decoding generates more realistic images than direct pixel level fusion.

- Table 1. Different distributions of sketch control blocks. Our method has the best sketch faithfulness compared with other distributions.

Training Inference

Method Ctrl-10 Ctrl-5 Ours SpCtrl Ctrl-10 Ctrl-5 Ours Memory(GB) 70.79 55.37 56.84 11.94 21.23 19.95 20.35 Time(s) N/A N/A N/A 29 52 46 41

- Table 2. The memory and time efficiency of different methods. SpCtrl refers to SparseCtrl [5] method. Ctrl-10 and Ctrl-5 refer to the Ctrl-CogVideo baseline with 10 and 5 control blocks, respectively.

Distribution of control blocks. We constructed a control branch with blocks at positions 1-5, 13-17, 26-30, and

[Figure 13]

- Figure 5. Video generation results using the same text prompts (a) and different keyframe sketches (b). The generated results (c) have a similar appearance while exhibiting different geometry details. The keyframe sketches and corresponding generated frames are highlighted in orange and green box.

[Figure 14]

- Figure 6. The video generation results using the same two keyframe sketches (b) and different text prompts (a). The generated results

- (c) share a similar structure while exhibiting different appearances, demonstrating the influence of text prompts on visual details. The keyframe sketches and corresponding generated frames are highlighted in the orange and green boxes.

a uniform distribution (ours). To evaluate the effectiveness of these placements, we compute LPIPS between the input and extracted sketches from corresponding generated frames. Due to the restriction of computing resources, we trained 10,000 steps on our hybrid image and video dataset. As shown in Table 1, our method incorporates control signals at multiple feature levels, enabling a more comprehensive analysis and achieving superior performance compared to the baselines. Additionally, in the DiT-based video diffusion model (CogVideo-2b in our method), we observe that earlier blocks primarily influence geometric features, while later blocks exhibit reduced controllability in this aspect.

### 6. Memory and Time Efficiency

A direct comparison of memory and time with existing methods, such as SparseCtrl, is not meaningful due to different architectures (Unet for SparseCtrl, DiT for ours) and frame counts (16 vs. 49 frames). We applied SparseCtrl’s concept to CogVideo, but using half of the blocks (15 blocks) as in Pixart-δ [1] resulted in out-of-memory errors. In Table 2, we report its memory and time with 10 blocks (Ctrl-10) and 5 blocks (Ctrl-5) below. Our sketch control block consumes slightly more memory compared with Ctrl-5 due to additional inter-frame attention. Our method achieves faster inference speed compared with the Ctrl-5 and Ctrl-10 baselines.

### 7. More Comparison Results

We present additional comparison results for both generation and editing. For sketch-based generation, Fig. 11

shows comparison results of a single keyframe sketch. We exclude results from video interpolation methods [2, 9, 15] since they require two keyframes. Our method generates results that are most faithful to the input sketches, such as the wing shape and a notch in the cake. Fig. 12 shows comparison results for two keyframe sketches. We compare with two additional video interpolation methods, including Seine [2] and ToonCrafter [15], whose official codes and weights are utilized in our experiments. Since the text-toimage results of two keyframes are not consistent, these methods generate fuzzy details in intermediate frames. Our method generates the most realistic results with the best temporal coherence.

For sketch-based editing, Fig. 13 shows additional comparison results, including two additional methods: TokenFlow [4] and I2VEdit [12]. Our method generates the most realistic and geometrically consistent results. In contrast, TokenFlow [4] exhibits minor editing effects due to the complexity of geometry modifications, while InsV2V [3] has a similar problem with insufficient geometry control. Furthermore, AnyV2V [8] and I2VEdit [12] totally change the unedited regions and have fuzzy details.

Quantitative Comparison. We conducted quantitative experiments, including additional generation methods (i.e., Seine [2] and ToonCrafter [15]) and editing methods (i.e., TokenFlow [4] and I2VEdit [12]). For generations, Seine and ToonCrafter achieved similar sketch faithfulness to AMT, as they utilize the same beginning and ending frames. However, they exhibit lower temporal consistency due to the smaller number of frames and greater content variation between adjacent frames. For editing, existing methods sig-

[Figure 15]

- Figure 7. Results of sketch-based video editing. For each example, the input text and drawing sketch(es) are given in (a). In (b), the original video is at the top, while the edited video is at the bottom. The sketches and corresponding frames are highlighted in orange and green boxes.

Metrics ToonCrafter[15] Seine[2] AMT[9] SparseCtrl[5] Ctrl-CogVideo Ours LPIPS ↓ 29.09 29.54 29.17 44.85 32.23 27.56

CLIP ↑ 95.75 92.76 96.12 96.48 98.04 98.31

- Table 3. The quantitative results of sketch-based video generation comparison. The LPIPS and CLIP numbers are scaled up 100×, with each cell colored to indicate the best . Methods ToonCrafter [15], Seine [2], and AMT [9] interpolate ControlNet-generated images while the other method directly translate sketches into videos.

### 8. Sketch Propagation Illustration

nificantly change the unedited regions, as reflected in their low reconstruction values in these regions. Our method has the best sketch faithfulness, temporal consistency, and unedited region preservation.

As shown in Fig. 10, we visualize the attention maps during the generation of a frame highlighted in an orange box, both for input sketches (b, c) and itself (d). They are consistent

[Figure 16]

- Figure 8. Consecutive video generation and editing results. In Step 1 (Generation 1), a video is generated based on an input text and a sketch. In Step 2 (Editing 1), a flower is added to the generated video. Step 3 (Editing 2) adds a hummingbird to the edited video.

[Figure 17]

- Figure 9. Ablation study of latent fusion. Replacing the latent fusion strategy with direct image space fusion leads to edited results (b) with noticeable seams. Our method removes this artifact and generates a realistic edited video with the new component added into the original content.

with the input sketches and frame’s edge maps, supporting our inter-frame attention mechanism for edge analysis and sketch-based controllability.

[Figure 18]

- Figure 10. The visualization of inter-frame attention maps. The input sketches (a) and attention maps (b,c,d) are shown in the top row, while the generated video is shown in the bottom row.

### 9. Failure Cases

Fig. 14 shows failure cases of our method. Similar to image generation, our method generates fuzzy details in challenging cases, such as the human hands. Additionally, when the two keyframe sketches differ significantly in content, the generated video transitions may appear unnatural or incoherent due to the lack of such a dataset. 3D geometry information and data augmentation can be potentially used to solve these problems.

### 10. Ethical Discussion

Our method is designed for positive applications such as movie and short video production. We strongly condemn the abuse of AI video generation technologies. The meth-

Metrics InsV2V[3] AnyV2V[8] I2VEdit[12] TokenFlow[4] Ours LPIPS ↓ 13.61 11.92 10.98 12.92 9.74

CLIP ↑ 95.39 93.47 96.81 97.83 98.34 PSNR ↑ 16.84 13.68 12.98 21.15 36.48

- Table 4. The quantitative results of sketch-based video editing comparison. The LPIPS and CLIP numbers are scaled up 100×, with each cell colored to indicate the best . AnyV2V [8] and I2VEdit [12] propagate the image editing results into videos. InV2V [3] and TokenFLow [4] utilize text prompts for video editing.

[Figure 19]

- Figure 11. Comparison of sketch-based video generation results for one keyframe sketch input. We compare our method with SparseCtrl [5] and Ctrl-CogVideo (baseline discussion in Sec 4.3 in main paper). Our method generates the results that are most faithful to the input sketches.

ods [16–18] that detect the fake video may be helpful to avoid the potential misuse of existing video generation approaches. Our method can also be used to generate training data for these detection methods.

### 11. Full Text Prompt

For ease of reading, the text prompts shown in the figures are simplified. The full-text prompts of each figure are shown in the following:

Main Paper Figure 1: The camera meticulously frames a close-up of a majestic male lion’s face, his mane a regal cascade of tawny hues, eyes a piercing amber gaze that speaks of the wild spirit within. The sunlight bathes his features in a golden glow, highlighting the rugged texture of his fur and the subtle scars that tell tales of past battles. His whiskers are pro-

nounced, sensitive to the faintest breeze, and his mouth is slightly agape, revealing formidable teeth, a silent testament to his power as the king of the savannah.

The video is a drone shot of a serene landscape featuring a deep blue lagoon surrounded by lush green trees and rocky cliffs. The lagoon is nestled between the mountains, creating a picturesque scene. The drone captures the tranquility of the water and the vibrant colors of the surrounding nature. The video is a beautiful representation of the natural beauty of the area, showcasing the lagoon’s unique location and the stunning views it offers.

The video captures the breathtaking beauty of a mountainous landscape. The first frame shows a clear blue sky above the mountains, with a few clouds scattered across it. The second frame reveals a deep blue lake nestled between the mountains, its calm waters reflecting the surrounding scenery. The third frame offers a closer view of the mountains, showcasing their rugged terrain and the patches of

[Figure 20]

- Figure 12. Comparison of sketch-based video generation results for two keyframe sketches. The input text prompts and sketches are shown in the left region, while the results of AMT [9], Seine [2], ToonCrafter [15], SparseCtrl [5], Ctrl-CogVideo (baseline discussed in Sec 4.3 in the main paper), and our method are shown in the right region. Our method achieves the best temporal coherence in the generated video.

snow that still cling to their peaks. The overall style of the video is serene and majestic, capturing the natural beauty of the landscape in a way that is both awe-inspiring and tranquil.

A magnificent waterfall cascades down from a series of rocky cliffs. The waterfall occupies most of the area in the picture, which is very magnificent and spectacular. The water flow of the waterfall is very turbulent, with white splashes spanning the entire scene. The waterfall cascades

from the top of the cliff all the way to the bottom of the lake.

A fluffy Pomeranian dog wears a light-colored top hat made of cloth adorned with a pink satin bow tied neatly around the base of the hat. The dog has a light brown coat with white markings on its face and chest. The top hat, slightly tilted, gives the Pomeranian a charming and playful appearance, while the pink bow adds a touch of elegance and whimsy. The dog’s fur is well-groomed, showcasing its signature soft, voluminous coat that frames its small face.

[Figure 21]

- Figure 13. Comparison of video editing results for different methods. The input text prompts and sketches are shown on the left. The edited results from InsV2V [3], TokenFlow [4], AnyV2V [8], I2VEdit [12], and ours are shown on the right. Our method generates the most realistic video editing results.

Figure 3: A small, fluffy rabbit with soft, white fur and a pink nose hops gracefully from left to right across a lush, green meadow. The sun is setting, casting a warm, golden glow over the scene. The rabbit pauses momentarily, its ears twitching as it listens to the gentle rustling of the leaves in

the nearby trees. It then continues its playful dance, moving side to side with an air of innocence and joy, the grass beneath its paws a soft carpet of nature’s embrace.

A sweeping panoramic video captures the grandeur of a majestic city, where towering skyscrapers pierce the clouds, their glass facades shimmering in the sunlight. Nestled at

[Figure 22]

- Figure 14. Failure cases of our method. Our method generates fuzzy details in hands and eyeglasses. Performance degrades when the two keyframe sketches (highlighted in orange and green box) depict entirely different objects, resulting in strange transitions between frames.

the base of these architectural giants, quaint urban homes with red-tiled roofs and leafy gardens form a charming tapestry around the city’s perimeter. In the bottom left corner, a tranquil lake mirrors the sky with its rippling waves of azure water, while fluffy white clouds drift lazily across the expansive blue canvas above, completing the serene yet powerful urban landscape.

The video captures a close-up of a cat’s face in three frames. The cat has striking green eyes and a brown and black spotted coat. In the first frame, the cat is looking directly at the camera, its eyes wide and alert. In the second frame, the cat’s gaze is slightly averted, and its eyes are more relaxed. In the third frame, the cat is looking away from the camera, its eyes focused on something in the distance. The style of the video is a simple, yet intimate portrait of the cat, with a focus on its expressive eyes and fur pattern.

In a breathtaking cosmic event, a video captures the catastrophic explosion of a light blue planet, its surface fragmenting as rocks are catapulted into the void. The sky behind is a tapestry of stars, while the planet itself becomes a maelstrom of debris, with towering plumes of smoke and cascades of fiery orange flames painting a harrowing picture of celestial destruction. The detritus from the disintegrating planet creates a mesmerizing display against the serene backdrop of space, a stark contrast between the calm universe and the violent upheaval of the once-stable world.

The video is a time-lapse aerial shot of a large, ornate building with a distinctive dome and multiple balconies. The building is surrounded by lush greenery and a wellmanicured lawn. In the first frame, the building is bathed in daylight, with the sun shining brightly on its facade. In the second frame, the sun begins to set, casting a warm glow on the building and its surroundings. In the third frame, the sun has set, and the building is illuminated by artificial lights, creating a stark contrast with the darkening sky. The style of the video is realistic, with a focus on the architectural details of the building and the natural beauty of its surroundings.

Figure 4: Plumes of steam begin to billow from the volcano’s crest, hinting at the raw power simmering beneath the surface, a prelude to the impending eruption that promises to reshape the landscape.Rising majestically in the distance, a colos-

sal volcano stands as a silent sentinel, its peak dusted with a delicate blanket of white snow. The volcano’s flanks are lush with verdant vegetation, a vibrant green tapestry that contrasts starkly against the earthy tones of its rocky surface. In the foreground of the screen, the lake is rippling with water waves

A man is speaking. He has thick hair and is wearing a black suit with a white shirt and a red tie underneath. The camera focuses on his face and upper body.

The ornamental fish swims from left to right. A mesmerizing ornamental fish glides through the inky depths of the ocean, its vibrant scales shimmering with hues of electric blue, fiery orange, and iridescent green. With a sleek, flattened body that navigates the vast underwater expanse. Its delicate fins, adorned with intricate patterns, flutter and sway like rhythmic poetry in motion, leaving a trail of iridescence in the silent sea.

A green grassland and a small path extending into the

distance. The grassland is covered with gravel. Figure 5: The video captures a large cruise ship docked at a pier. The ship is painted in white and black, with a distinctive yellow and gold design on its side. The ship is moored to the pier, which is lined with several smaller boats. The sky above is filled with clouds, suggesting an overcast day. The water around the ship is calm, reflecting the ship’s grandeur. The pier extends into the water, providing a clear view of the ship’s size and scale. The video is a time-lapse, capturing the ship’s arrival and docking process. The style of the video is realistic, with a focus on the ship and its surroundings. The video does not contain any people or animals, and the focus is solely on the ship and its immediate environment.

- Figure 6: A vibrant magpie, its feathers a striking contrast of soft white and cool black, perches gracefully atop a slender branch of an ancient flower tree. The bird’s delicate talons grip the bark with precision, as soft morning light filters through the leafless canopy, casting gentle shadows that dance around it. The magpie’s head tilts slightly, its keen eye surveying the tranquil winter scene, while the rest of the forest lies quiet and still in the crisp, cool air of an early dawn.
- Figure 7:

Extreme close-up of chicken and green pepper kebabs grilling on a barbeque with flames. Shallow focus and light smoke. vivid colours

Figure 8: The video captures a wooden boat with coastal landscape. A small wooden boat is sailing on the sea surface. The wooden boat is composed of wooden boards with rich textures, with a rustic texture.

Supplemental Material Figure 1: A very cute little wild dog with a mouth and shiny fur, shining brightly towards the center of the picture, very lively. The dog takes a walk on the street with a natural flow of movements.

The video features a woman with long blonde hair, wearing a black and white dress, and large earrings. She is seated in a room with a white wall, which is adorned with various photographs. The woman is looking directly at the camera, and her expression is neutral. The room has a bright and airy feel to it. The style of the video is a combination of a personal interview and a motivational message.

The video features a red Tesla Model S driving on a highway with a mountainous landscape in the background. The car is sleek and modern, with a streamlined design and a large front grille. The highway is wide and well-maintained, with clear lane markings. The mountains rise majestically in the distance, their peaks shrouded in mist. The sky is a clear blue, and the sun is setting, casting a warm glow over the scene. The car is moving at a steady pace, and the driver appears to be focused on the road ahead. The overall style of the video is dynamic and energetic, capturing the thrill of driving a high-performance electric vehicle in a beautiful natural setting.

The video captures a serene coastal scene, showcasing the natural beauty of the area. The first frame shows a rocky shoreline with the calm ocean gently lapping against the rocks. The second frame reveals a small town nestled on the hillside, with houses and buildings scattered across the landscape. The third frame offers a closer view of the shoreline, with the waves crashing against the rocks, creating a dynamic contrast between the stillness of the ocean and the power of the waves. The overall style of the video is a blend of nature and human habitation, with a focus on the interplay between the land and the sea.

The video captures a sailboat journey across a vast body of water. The sailboat, with its white sail billowing in the wind, is the main focus of the video. The boat is seen from a high angle, providing a bird’s eye view of its journey. The water around the boat is choppy, indicating a windy day. The sky above is a clear blue, suggesting a sunny day. The boat is moving towards the right side of the frame, indicating its direction of travel. The overall style of the video is a dynamic and adventurous depiction of a sailboat journey

on a windy day. Figure 2: A fluffy, grey tabby kitten with bright, curious eyes and a white chest is playfully rolling on a patch of soft, green grass. The sun casts a gentle glow over its fur, emphasizing the subtle stripes and the warmth of its whiskers. The kitten pauses, sits up, and with a twitch of its little pink nose, it tilts its head inquisitively, capturing the essence of feline charm in a serene, outdoor setting.

Figure 3: In a breathtaking close-up, a majestic gray wolf with a coat of realistic, silver-tipped fur stands regally against the lush, green tapestry of an ancient forest. Its eyes, deep and expressive, seem to hold the secrets of the wild, locking gazes with the viewer through the lens. With a mouth agape, the wolf appears to be caught in a silent howl or a gentle pant, breathing life into the serene scene. The atmosphere is one of unspoken grandeur, as the video encapsulates the raw, wild essence of the wolf amidst the tranquility of its untouched habitat.

The video captures the majestic Mont Saint-Michel, a historic abbey and fortification located on a small island in Normandy, France. The first frame shows the abbey from a distance, its grandeur accentuated by the clear blue sky. The second frame offers a closer view, revealing the intricate details of the abbey’s architecture. The third frame provides a ground-level perspective, allowing viewers to appreciate the scale of the abbey and its surroundings. The video is a testament to the abbey’s historical significance and architectural beauty, set against the backdrop of a serene day.

At the bottom of the screen is a green lake surface, rippling with the wind. A small wooden boat moved from the left side of the lake to the right side. In the middle are continuous mountains, towering and magnificent, not satisfied with the green forest, with snow mixed at the top of the mountains. The background is a blue sky.

Figure 4: A bustling scene at the Piazza del Duomo in Milan, Italy. The focal point of the scene is the large archway entrance to the Galleria Vittorio Emanuele II, a historic shopping mall. The archway, constructed of white stone, stands majestically against the backdrop of a clear blue sky. Flanking the entrance are two buildings, one on each side. These buildings, also made of white stone, are adorned with green awnings, adding a touch of color to the otherwise monochrome structure. The piazza itself is teeming with life. People can be seen walking around, adding a dynamic element to the scene. In the background, a few trees can be spotted, providing a touch of nature amidst the urban setting. Despite the scene being taken from a distance, the details of the archway and the surrounding buildings are clearly visible, showcasing the architectural grandeur of the landmark. The scene does not contain any discernible text.

The relative positions of the objects suggest a well-planned architectural design, with the archway centrally located and the buildings symmetrically placed on either side.

- Figure 5: The video captures a brown owl in flight, soaring through a forested area. The owl’s wings are spread wide, showcasing its impressive wingspan. The owl’s eyes are wide open, alert and focused on its surroundings. The background is a blur of green and brown, indicating the presence of trees and foliage. The owl’s flight path is smooth and graceful, demonstrating the bird’s agility and control. The overall style of the video is a close-up, slow-motion shot, allowing viewers to appreciate the details of the owl’s flight and the beauty of its natural habitat.
- Figure 6: The video shows a chocolate cake being decorated with candy. The cake is placed on a white plate, and the candies are scattered on top of the cake. The candies are orange and yellow, and they are placed on top of the chocolate frosting. The cake is then drizzled with chocolate sauce, which is poured over the top of the cake. The sauce is thick and glossy, and it drips down the sides of the cake. The cake is then placed on a table, and the camera zooms in on the cake, showing the details of the decoration. The video is a close-up shot, focusing on the cake and the decoration. The style of the video is a food video, showcasing the process of decorating a cake.

A gleaming yellow cake, adorned with white frosting and delicate piping, sits at the center of a marble countertop. Above it, a cascade of vibrant pink strawberry juice, shimmering under the kitchen lights, gently pours from a height, drizzling down the sides of the cake, pooling slightly around the base, and creating a mesmerizing contrast between the yellow cake and the rosy liquid.

Figure 7: The video shows a man with a beard standing next to a black truck. He is wearing a black T-shirt. In the first frame, he is pointing at the truck. In the second frame, he is opening the door of the truck. In the third frame, he is sitting inside the truck. The truck is parked in a lot with trees in the background. The man appears to be in the process of getting into the truck. The style of the video is casual and informal.

The video captures the majestic Neuschwanstein Castle, a 19th-century Romanesque Revival palace, perched on a rugged hill above the village of Hohenschwangau near F¨ussen in southwest Bavaria, Germany. The castle, a symbol of fairy tales and fantasy, is seen in three different frames, each showcasing its grandeur against the backdrop of the snow-capped mountains and the surrounding forest. The first frame offers a distant view of the castle, its multiple towers and turrets reaching towards the sky. The second frame provides a closer perspective, revealing the intricate

details of the castle’s architecture. The third frame offers a panoramic view of the castle, its towers and turrets standing tall against the backdrop of the snow-capped mountains and the surrounding forest. The video is a testament to the castle’s historical significance and architectural beauty.

A close-up shot captures the innocent, yet adventurous expression of a young, blonde girl, her eyes a shimmering shade of blue. She’s adorned in a classic khaki canvas top hat, casting a gentle shadow over her bright, curious eyes. The scene is one of quiet wonder, with the girl’s face conveying a mix of mischief and wonder, as if she’s about to embark on a grand, unknown journey.

An ancient beige white temple, crafted from marble with a subtle yellow patina, stands majestically amidst a serene landscape. It boasts an array of towering pillars, each meticulously carved, supporting a grand, beige white roof. The temple’s weathered surface reveals the whispers of time, marked by intricate cracks and the subtle discolorations that speak to its storied past.

A small, light white fox with a fluffy tail and alert ears sits gracefully atop a lush green meadow, its head initially facing the viewer with a curious gaze. Gradually, the little fox tilts its head to the left, its eyes glinting with a mix of curiosity and wariness, as it surveys the surroundings. The soft sunlight filters through the nearby trees, casting dappled shadows that dance across the grass and the fox’s fur, creating a tranquil and picturesque scene. The grassland in the background presents a bright yellow green color.

Figure 8: A vibrant cluster of wildflowers, a kaleidoscope of reds, yellows, and purples, captures the foreground, their delicate petals swaying gently in the breeze. Beyond, a majestic range of snow-capped mountains stretches across the horizon, their peaks glistening under the midday sun, creating a breathtaking contrast against the clear blue sky. The flowers, a testament to the valley’s vibrant life, and the distant mountains, symbols of enduring strength, together paint a picture of serene natural beauty.

A majestic white chrysanthemum, its circular petals meticulously wrapped around a vibrant yellow core, sways gracefully in a gentle breeze. Each petal, delicate and perfectly formed, shimmers with the morning dew, catching the light as the flower dances with the rhythm of the wind. The scene captures the delicate balance of nature, the floral beauty moving in harmony with the unseen forces around it, a serene moment of natural elegance.

A vibrant yellow hummingbird, its iridescent feathers shimmering in the soft sunlight, hovers with exquisite precision above a pristine white flower. The scene is captured in side profile, showcasing the bird’s slender beak as it extends towards the flower’s nectar-rich center, wings blurred by the rapidity of their motion, a mesmerizing dance of nature’s delicate balance.

Figure 9: A fluffy Pomeranian dog weara light colored top hat made of cloth adorned with a pink satin bow tied neatly around the base of the hat. The dog has a light brown coat with white markings on its face and chest. The top hat, slightly tilted, gives the Pomeranian a charming and playful appearance, while the pink bow adds a touch of elegance and whimsy. The dog’s fur is well-groomed, showcasing its signature soft, voluminous coat that frames its small face.

- Figure 10: A mouthwatering round cake, adorned with a

dusting of powdered sugar, and a generous drizzle of glossy, dark chocolate ganache. The cake itself is a masterpiece of delicate sponge layers, each separated by a rich, velvety buttercream frosting that peeks through the sides, inviting everyone to take a slice. The scene is set with a soft focus on the cake, positioned on a marble countertop, with gentle lighting casting subtle shadows to accentuate its delectable texture.

- Figure 11: The video captures a brown owl in flight, soaring through

a forested area. The owl’s wings are spread wide, showcasing its impressive wingspan. The owl’s eyes are wide open, alert and focused on its surroundings. The background is a blur of green and brown, indicating the presence of trees and foliage. The owl’s flight path is smooth and graceful, demonstrating the bird’s agility and control. The overall style of the video is a close-up, slow-motion shot, allowing viewers to appreciate the details of the owl’s flight and the beauty of its natural habitat.

The video shows a close-up of three chocolate-covered desserts with various toppings, including nuts and a red filling, on a white plate. The desserts are presented in a way that suggests they are being eaten or sampled, with one dessert partially cut into and a spoon nearby. The style of the video is simple and straightforward, focusing on the desserts without any additional context or background. The lighting is bright, highlighting the textures and colors of the desserts. The video is likely intended to showcase the desserts’ presentation and appeal to viewers’ appetites.

Figure 12: The video features a large, gray teddy bear with a white nose and a blue ribbon around its neck. The bear is holding a small cake with a single candle on it. The bear is sitting in a room with a white wall in the background. The bear appears to be looking at the cake. The video is likely a celebration of a birthday or a special occasion. The bear’s fur is soft and fluffy, and the blue ribbon adds a touch of color to the gray bear. The cake is small and white, with a single candle on top. The bear’s eyes are closed, and it seems to be enjoying the moment. The room is brightly lit, and the white wall in the background provides a clean and simple backdrop for the scene. The bear is the main focus of the video, and its size and position in the frame make it the

most prominent object. The cake is smaller in comparison, but it still stands out due to its bright color and the candle on top. The bear’s position relative to the cake suggests that it is about to blow out the candle. The overall style of the video is simple and straightforward, with a focus on the bear and the cake. The lighting is bright and even, and the colors are soft and muted. The video does not contain any text or additional objects, and the focus is solely on the bear and the cake.

The video shows a serene waterfront scene with two large, multi-story houses situated on a grassy hillside. The houses are constructed with a combination of stone and wood, featuring traditional architectural elements such as pitched roofs and bay windows. The houses are surrounded by lush greenery, including a variety of trees and shrubs, which add to the natural beauty of the setting. In the foreground, there is a wooden dock extending into the water, with a small boat moored at the end. The boat appears to be a leisure vessel, possibly used for fishing or recreational purposes. The calm water reflects the houses and the surrounding landscape, creating a peaceful and idyllic atmosphere. The style of the video is realistic and naturalistic, capturing the tranquility of the waterfront location with a focus on the architecture and the natural environment. The camera angles and movements are steady and unhurried, allowing the viewer to appreciate the details of the houses and the surrounding landscape. The lighting is soft and diffused, suggesting either an overcast day or a time when the sun is not directly shining on the scene. Overall, the video presents a picturesque and serene waterfront setting, likely intended to evoke a sense of relaxation and escape from the hustle and bustle of everyday life.

Figure 13: Two pristine white flowers gently sway in the gentle breeze, their delicate petals catching the soft rays of sunlight that filter through the surrounding foliage. Each flower, with its subtle fragrance and intricate design, appears almost weightless as it dances gracefully in the air, creating a serene and tranquil atmosphere. The movement of these blossoms against the backdrop of lush greenery adds a dynamic element to the otherwise still scene, evoking a sense of peace and beauty in nature.

A fluffy Pomeranian dog weara light colored top hat made of cloth adorned with a pink satin bow tied neatly around the base of the hat. The dog has a light brown coat with white markings on its face and chest. The top hat, slightly tilted, gives the Pomeranian a charming and playful appearance, while the pink bow adds a touch of elegance and whimsy. The dog’s fur is well-groomed, showcasing its signature soft, voluminous coat that frames its small face.

Figure 14: The video features a bald man with a beard and glasses,

wearing a white shirt. He is standing against a blue back-

ground. The man appears to be speaking or presenting, as he is gesturing with his hands. The style of the video is professional and polished, suggesting it could be a corporate or educational video. The man’s attire and the background give the impression of a formal or professional setting.

In a whimsical transformation, a sleek black cat with piercing green eyes sits gracefully on a windowsill, bathed in the warm glow of the afternoon sun. As the scene progresses, the cat blinks slowly, and with a shimmering aura, begins to morph seamlessly into a majestic bird, its fur transitioning into feathers of iridescent brown and white. The newly formed bird spreads its powerful wings, perched confidently on the same sill, now surveying the open sky with an air of freedom and wild spirit.

### References

- [1] Junsong Chen, Yue Wu, Simian Luo, Enze Xie, Sayak Paul, Ping Luo, Hang Zhao, and Zhenguo Li. Pixart-δ: Fast and controllable image generation with latent consistency models. CoRR, abs/2401.05252, 2024. 4
- [2] Xinyuan Chen, Yaohui Wang, Lingjun Zhang, Shaobin Zhuang, Xin Ma, Jiashuo Yu, Yali Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. Seine: Short-to-long video diffusion model for generative transition and prediction. In Int. Conf. Learn. Represent., 2023. 4, 5, 8
- [3] Jiaxin Cheng, Tianjun Xiao, and Tong He. Consistent videoto-video transfer using synthetic dataset. In Int. Conf. Learn. Represent., 2024. 4, 7, 9
- [4] Michal Geyer, Omer Bar-Tal, Shai Bagon, and Tali Dekel. Tokenflow: Consistent diffusion features for consistent video editing. In Int. Conf. Learn. Represent., 2024. 4, 7, 9
- [5] Yuwei Guo, Ceyuan Yang, Anyi Rao, Maneesh Agrawala, Dahua Lin, and Bo Dai. Sparsectrl: Adding sparse controls to text-to-video diffusion models. In Eur. Conf. Comput. Vis., pages 330–348, 2024. 3, 5, 7, 8
- [6] Wenyi Hong, Ming Ding, Wendi Zheng, Xinghan Liu, and Jie Tang. Cogvideo: Large-scale pretraining for text-to-video generation via transformers. arXiv preprint arXiv:2205.15868, 2022. 1
- [7] Zhewei Huang, Tianyuan Zhang, Wen Heng, Boxin Shi, and Shuchang Zhou. Real-time intermediate flow estimation for video frame interpolation. In Eur. Conf. Comput. Vis., 2022. 1
- [8] Max Ku, Cong Wei, Weiming Ren, Harry Yang, and Wenhu Chen. Anyv2v: A tuning-free framework for any video-tovideo editing tasks. arXiv preprint arXiv:2403.14468, 2024. 4, 7, 9
- [9] Zhen Li, Zuo-Liang Zhu, Ling-Hao Han, Qibin Hou, ChunLe Guo, and Ming-Ming Cheng. Amt: All-pairs multi-field transforms for efficient frame interpolation. In IEEE Conf. Comput. Vis. Pattern Recog., 2023. 4, 5, 8
- [10] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In Int. Conf. Learn. Represent., 2019. 1
- [11] Kepan Nan, Rui Xie, Penghao Zhou, Tiehan Fan, Zhenheng Yang, Zhijie Chen, Xiang Li, Jian Yang, and Ying Tai.

- Openvid-1m: A large-scale high-quality dataset for text-tovideo generation. CoRR, abs/2407.02371, 2024. 1
- [12] Wenqi Ouyang, Yi Dong, Lei Yang, Jianlou Si, and Xingang Pan. I2vedit: First-frame-guided video editing via image-tovideo diffusion models. CoRR, abs/2405.16537, 2024. 4, 7, 9
- [13] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, Patrick Schramowski, Srivatsa Kundurthy, Katherine Crowson, Ludwig Schmidt, Robert Kaczmarczyk, and Jenia Jitsev. LAION-5B: an open large-scale dataset for training next generation image-text models. In Adv. Neural Inform. Process. Syst., 2022. 1
- [14] Xintao Wang, Liangbin Xie, Chao Dong, and Ying Shan. Real-esrgan: Training real-world blind super-resolution with pure synthetic data. In ICCVW, pages 1905–1914, 2021. 1
- [15] Jinbo Xing, Hanyuan Liu, Menghan Xia, Yong Zhang, Xintao Wang, Ying Shan, and Tien-Tsin Wong. Tooncrafter: Generative cartoon interpolation. CoRR, abs/2405.17933,

2024. 4, 5, 8

- [16] Zhiyuan Yan, Yong Zhang, Yanbo Fan, and Baoyuan Wu. Ucf: Uncovering common features for generalizable deepfake detection. In Int. Conf. Comput. Vis., pages 22412– 22423, 2023. 7
- [17] Zhiyuan Yan, Yong Zhang, Xinhang Yuan, Siwei Lyu, and Baoyuan Wu. Deepfakebench: A comprehensive benchmark of deepfake detection. In Adv. Neural Inform. Process. Syst., pages 4534–4565, 2023.
- [18] Zhiyuan Yan, Yuhao Luo, Siwei Lyu, Qingshan Liu, and Baoyuan Wu. Transcending forgery specificity with latent space augmentation for generalizable deepfake detection. In IEEE Conf. Comput. Vis. Pattern Recog., 2024. 7
- [19] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Int. Conf. Comput. Vis., pages 3813–3824, 2023. 1

