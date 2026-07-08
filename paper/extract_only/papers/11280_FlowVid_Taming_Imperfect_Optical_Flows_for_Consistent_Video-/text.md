# arXiv:2312.17681v1[cs.CV]29Dec2023

## FlowVid: Taming Imperfect Optical Flows for Consistent Video-to-Video Synthesis

Feng Liang*1, Bichen Wu†2, Jialiang Wang2, Licheng Yu2, Kunpeng Li2, Yinan Zhao2, Ishan Misra2, Jia-Bin Huang2, Peizhao Zhang2, Peter Vajda2, Diana Marculescu1 1The University of Texas at Austin, 2Meta GenAI {jeffliang,dianam}@utexas.edu, {wbc,stzpz,vajdap}@meta.com https://jeff-liangf.github.io/projects/flowvid

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

Input

InputStylization

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

Stylization

Input

Prompt: a woman wearing headphones, in flat 2d anime

Prompt: a Chinese ink painting of a panda eating bamboo

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

ObjectSwap

ObjectSwap

Prompt: a koala eating bamboo

Prompt: a Greek statue wearing headphones

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

LocalEdits

LocalEdits

Prompt: a woman with black hair wearing headphones

Prompt: a panda with a pig nose eating bamboo

Figure 1. We present FlowVid to synthesize a consistent video given an input video and a target prompt. Our model supports multiple applications: (1) global stylization, such as converting the video to 2D anime (2) object swap, such as turning the panda into a koala bear (3) local edit, such as adding a pig nose to a panda.

### Abstract

Diffusion models have transformed the image-to-image (I2I) synthesis and are now permeating into videos. However, the advancement of video-to-video (V2V) synthesis has been hampered by the challenge of maintaining temporal consistency across video frames. This paper proposes a consistent V2V synthesis framework by jointly leveraging spatial conditions and temporal optical flow clues within the source video. Contrary to prior methods that strictly adhere to optical flow, our approach harnesses its benefits while handling the imperfection in flow estimation. We encode the optical flow via warping from the first frame and serve it as a supplementary reference in the diffusion model. This enables our model for video synthesis by editing the first

*Work partially done during an internship at Meta GenAI. †Corresponding author.

frame with any prevalent I2I models and then propagating edits to successive frames. Our V2V model, FlowVid, demonstrates remarkable properties: (1) Flexibility: FlowVid works seamlessly with existing I2I models, facilitating various modifications, including stylization, object swaps, and local edits. (2) Efficiency: Generation of a 4-second video with 30 FPS and 512×512 resolution takes only 1.5 minutes, which is 3.1×, 7.2×, and 10.5× faster than CoDeF, Rerender, and TokenFlow, respectively. (3) High-quality: In user studies, our FlowVid is preferred 45.7% of the time, outperforming CoDeF (3.5%), Rerender (10.2%), and TokenFlow (40.4%).

### 1. Introduction

Text-guided Video-to-video (V2V) synthesis, which aims to modify the input video according to given text prompts, has

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

the occluded regions, the inaccurate legs estimation would persist, leading to an undesirable outcome. We seek to include an additional spatial condition, such as a depth map in Figure 2(c), along with a temporal flow condition. The legs’ position is correct in spatial conditions, and therefore, the joint spatial-temporal condition would rectify the imperfect optical flow, resulting in consistent results in Figure 2(d).

1stframe

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

10thframe20thframe

We build a video diffusion model upon an inflated spatial controlled I2I model. We train the model to predict the input video using spatial conditions (e.g., depth maps) and temporal conditions (flow-warped video). During generation, we employ an edit-propagate procedure: (1) Edit the first frame with prevalent I2I models. (2) Propagate the edits throughout the video using our trained model. The decoupled design allows us to adopt an autoregressive mechanism: the current batch’s last frame can be the next batch’s first frame, allowing us to generate lengthy videos.

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

(a). Input video (b). Flow warping

(c). Spatial control (d). FlowVid output

- Figure 2. (a) Input video: ’a man is running on beach’. (b) We edit the 1st frame with ’a man is running on Mars’, then conduct flow warping from the 1st frame to the 10th and 20th frames (using input video flow). Flow estimation of legs is inaccurate. (c) Our FlowVid uses spatial controls to rectify the inaccurate flow. (d) Our consistent video synthesis results.

We train our model with 100k real videos from ShutterStock [1], and it generalizes well to different types of modifications, such as stylization, object swaps, and local edits, as seen in Figure 1. Compared with existing V2V methods, our FlowVid demonstrates significant advantages in terms of efficiency and quality. Our FlowVid can generate 120 frames (4 seconds at 30 FPS) in high-resolution (512×512) in just 1.5 minutes on one A-100 GPU, which is 3.1×, 7.2× and 10.5× faster than state-of-the-art methods CoDeF [32] (4.6 minutes) Rerender [49] (10.8 minutes), and TokenFlow [13] (15.8 minutes). We conducted a user study on 25 DAVIS [34] videos and designed 115 prompts. Results show that our method is more robust and achieves a preference rate of 45.7% compared to CoDeF (3.5%) Rerender (10.2%) and TokenFlow (40.4%)

wide applications in various domains, such as short-video creation and more broadly in the film industry. Notable advancements have been seen in text-guided Image-to-Image (I2I) synthesis [4, 14, 31, 43], greatly supported by large pretrained text-to-image diffusion models [37, 39, 40]. However, V2V synthesis remains a formidable task. In contrast to still images, videos encompass an added temporal dimension. Due to the ambiguity of text, there are countless ways to edit frames so they align with the target prompt. Consequently, naively applying I2I models on videos often produces unsatisfactory pixel flickering between frames.

To improve frame consistency, pioneering studies edit multiple frames jointly by inflating the image model with spatial-temporal attention [6, 25, 35, 46]. While these methods offer improvements, they do not fully attain the soughtafter temporal consistency. This is because the motion within videos is merely retained in an implicit manner within the attention module. Furthermore, a growing body of research employs explicit optical flow guidance from videos. Specifically, flow is used to derive pixel correspondence, resulting in a pixel-wise mapping between two frames. The correspondence is later utilized to obtain occlusion masks for inpainting [19, 49] or to construct a canonical image [32] However, these hard constraints can be problematic if flow estimation is inaccurate, which is often observed when the flow is determined through a pre-trained model [42, 47, 48].

Our contributions are summarized as follows: (1) We introduce FlowVid, a V2V synthesis method that harnesses the benefits of optical flow, while delicately handling the imperfection in flow estimation. (2) Our decoupled editpropagate design supports multiple applications, including stylization, object swap, and local editing. Furthermore, it empowers us to generate lengthy videos via autoregressive evaluation. (3) Large-scale human evaluation indicates the efficiency and high generation quality of FlowVid.

### 2. Related Work

##### 2.1. Image-to-image Diffusion Models

Benefiting from large-scale pre-trained text-to-image (T2I) diffusion models [2, 11, 39, 40], progress has been made in text-based image-to-image (I2I) generation [10, 14, 24, 30, 31, 33, 43, 51]. Beginning with image editing methods, Prompt-to-prompt [14] and PNP [43] manipulate the attentions in the diffusion process to edit images according to target prompts. Instruct-pix2pix [4] goes a step further by training an I2I model that can directly interpret and follow

In this paper, we propose to harness the benefits of optical flow while handling the imperfection in flow estimation. Specifically, we perform flow warping from the first frame to subsequent frames. These warped frames are expected to follow the structure of the original frames but contain some occluded regions (marked as gray), as shown in Figure 2(b). If we use flow as hard constraints, such as inpainting [19, 49]

human instructions. More recently, I2I methods have extended user control by allowing the inclusion of reference images to precisely define target image compositions. Notably, ControlNet, T2I-Adapter [31], and Composer [20] have introduced spatial conditions, such as depth maps, enabling generated images to replicate the structure of the reference. Our method falls into this category as we aim to generate a new video while incorporating the spatial composition in the original one. However, it’s important to note that simply applying these I2I methods to individual video frames can yield unsatisfactory results due to the inherent challenge of maintaining consistency across independently generated frames (per-frame results can be found in Section 5.2).

##### 2.2. Video-to-video Diffusion Models

To jointly generate coherent multiple frames, it is now a common standard to inflate image models to video: replacing spatial-only attention with spatial-temporal attention. For instance, Tune-A-Video [46], Vid-to-vid zero [44], Text2videozero [25], Pix2Video [6] and FateZero [35] performs crossframe attention of each frame on anchor frame, usually the first frame and the previous frame to preserve appearance consistency. TokenFlow [13] further explicitly enforces semantic correspondences of diffusion features across frames to improve consistency. Furthermore, more works are adding spatial controls, e.g., depth map to constraint the generation. Zhang’s ControlVideo [50] proposes to extend image-based ControlNet to the video domain with full cross-frame attention. Gen-1 [12], VideoComposer [45], Control-A-Video [7] and Zhao’s ControlVideo [52] train V2V models with paired spatial controls and video data. Our method falls in the same category but it also includes the imperfect temporal flow information into the training process alongside spatial controls. This addition enhances the overall robustness and adaptability of our method.

Another line of work is representing video as 2D images, as seen in methods like layered atlas [23], Text2Live [3], shape-aware-edit [26], and CoDeF [32]. However, these methods often require per-video optimization and they also face performance degradation when dealing with large motion, which challenges the creation of image representations.

##### 2.3. Optical flow for video-to-video synthesis

The use of optical flow to propagate edits across frames has been explored even before the advent of diffusion models, as demonstrated by the well-known Ebsythn [22] approach. In the era of diffusion models, Chu’s Video ControlNet [9] employs the ground-truth (gt) optical flow from synthetic videos to enforce temporal consistency among corresponding pixels across frames. However, it’s important to note that ground-truth flow is typically unavailable in real-world videos, where flow is commonly estimated using pretrained models [42, 47, 48]. Recent methods like Reren-

der [49], MeDM [8], and Hu’s VideoControlNet [19] use estimated flow to generate occlusion masks for in-painting. In other words, these methods ”force” the overlapped regions to remain consistent based on flow estimates. Similarly, CoDeF [32] utilizes flow to guide the generation of canonical images. These approaches all assume that flow could be treated as an accurate supervision signal that must be strictly adhered to. In contrast, our FlowVid recognizes the imperfections inherent in flow estimation and presents an approach that leverages its potential without imposing rigid constraints.

### 3. Preliminary

Latent Diffusion Models Denoising Diffusion Probabilistic Models (DDPM) [16] generate images through a progressive noise removal process applied to an initial Gaussian noise, carried out for T time steps. Latent Diffusion models [39] conduct diffusion process in latent space to make it more efficient. Specifically, an encoder E compresses an image I ∈ RH×W×3 to a low-resolution latent code z = E(I) ∈ RH/8×W/8×4. Given z0 := z, the Gaussian noise is gradually added on z0 with time step t to get noisy sample zt. Text prompt τ is also a commonly used condition. A time-conditional U-Net ϵθ is trained to reverse the process with the loss function:

0,t,τ,ϵ∼N(0,1)∥ϵ − ϵθ(zt,t,τ)∥22 (1) ControlNet ControlNet provides additional spatial conditions, such as canny edge [5] and depth map [38], to control the generation of images. More specifically, spatial conditions C ∈ RH×W×3 are first converted to latents c ∈ RH/8×W/8×4 via several learnable convolutional layers. Spatial latent c, added by input latent zt, is passed to a copy of the pre-trained diffusion model, more known as ControlNet. The ControlNet interacts with the diffusion model in multiple feature resolutions to add spatial guidance during image generation. ControlNet rewrites Equation 1 to

LLDM = Ez

0,t,τ,c,ϵ∼N(0,1)∥ϵ − ϵθ(zt,t,τ,c)∥22 (2)

LCN = Ez

### 4. FlowVid

For video-to-video generation, given an input video with N frames I = {I1,...,IN} and a text prompt τ, the goal is transfer it to a new video I′ = {I1′,...,IN′ } which adheres to the provided prompt τ′, while keeping consistency across frame. We first discuss how we inflate the image-to-image diffusion model, such as ControlNet to video, with spatialtemporal attention [6, 25, 35, 46] (Section 4.1) Then, we introduce how to incorporate imperfect optical flow as a condition into our model (Section 4.2). Lastly, we introduce the edit-propagate design for generation (Section 4.3).

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

| | | |
|---|---|---|
| |[Figure 47]|+|

Conv

Conv

… Depth pred.

…

###### +

[Figure 48]

[Figure 49]

[Figure 50]

Spatial conditions

[Figure 51]

Spatial conditions

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

Concat

Concat

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

…

#### ~ 𝑓

[Figure 64]

[Figure 65]

Adding noise

…

[Figure 66]

[Figure 67]

[Figure 68]

𝑓

[Figure 69]

…

[Figure 70]

[Figure 71]

[Figure 72]

Input video

[Figure 73]

Input video

[Figure 74]

Flow warping from 1st frame

[Figure 75]

[Figure 76]

[Figure 77]

Loss

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

Synthesized video

…

CLIP

[Figure 84]

[Figure 85]

…

###### CLIP

|A man is running on Mars.|
|---|

[Figure 86]

|A man is running on beach.|
|---|

Flow warped edited video

[Figure 87]

[Figure 88]

Flow warped video

[Figure 89]

(a). Training (b). Generation

- Figure 3. Overview of our FlowVid. (a) Training: we first get the spatial conditions (predicted depth maps) and estimated optical flow from the input video. For all frames, we use flow to perform warping from the first frame. The resulting flow-warped video is expected to have a similar structure as the input video but with some occluded regions (marked as gray, better zoomed in). We train a video diffusion model with spatial conditions c and flow information f. (b) Generation: we edit the first frame with existing I2I models and use the flow in the input video to get the flow warped edited video. The flow condition spatial condition jointly guides the output video synthesis.
- 4.1. Inflating image U-Net to accommodate video

data. Yet, our empirical analysis indicates that this leads to sub-optimal results, as detailed in the ablation study in Section 5.4. We hypothesize that this method neglects the temporal clue within the video, making the frame consistency hard to maintain. While some studies, such as Rerender [49] and CoDeF [32], incorporate optical flow in video synthesis, they typically apply it as a rigid constraint. In contrast, our approach uses flow as a soft condition, allowing us to manage the imperfections commonly found in flow estimation.

The latent diffusion models (LDMs) are built upon the architecture of U-Net, which comprises multiple encoder and decoder blocks. Each block has two components: a residual convolutional module and a transformer module. The transformer module, in particular, comprises a spatial selfattention layer, a cross-attention layer, and a feed-forward network. To extend the U-Net architecture to accommodate an additional temporal dimension, we first modify all the 2D layers within the convolutional module to pseudo-3D layers and add an extra temporal self-attention layer [18]. Following common practice [6, 18, 25, 35, 46], we further adapt the spatial self-attention layer to a spatial-temporal self-attention layer. For video frame Ii, the attention matrix would take the information from the first frame I1 and the previous frame Ii−1. Specifically, we obtain the query feature from frame Ii, while getting the key and value features from I1 and Ii−1. The Attention(Q,K,V ) of spatial-temporal self-attention could be written as

Given a sequence of frames I, we calculate the flow between the first frame I1 and other frames Ii, using a pretrained flow estimation model UniMatch [48]. We denote the F1→i and Fi→1 as the forward and backward flow. Using forward-backward consistency check [29], we can derive forward and backward occlusion masks O1fwd→i and Oibwd→1. Use backward flow Fi→1 and occlusion Oibwd→1, we can perform Warp operation over the first frame I1 to get IiW. Intuitively, warped ith frame IiW has the same layout as the original frame Ii but the pixels are from the first frame I1. Due to occlusion, some blank areas could be in the IiW (marked as gray in Figure 3).

Q = WQzIi, K = WK zI1, zIi−1 , V = WV zI1, zIi−1 (3)

where WQ, WK, and WV are learnable matrices that project the inputs to query, key, and value. zI

is the latent for frame Ii. [·] denotes concatenation operation. Our model includes an additional ControlNet U-Net that processes spatial conditions. We discovered that it suffices only to expand the major U-Net, as the output from the ControlNet U-Net is integrated into this major U-Net.

We denote the sequence of warped frames as flow warped video IW = {I1W,...,INW}. We feed IW into the same encoder E to convert it into a latent representation f. This latent representation is then concatenated with the noisy input zt to serve as conditions. To handle the increased channel dimensions of f, we augment the first layer of the U-Net with additional channels, initializing these new channels with zero weights. We also integrate this concatenated flow information into the spatial ControlNet U-Net, reconfiguring its initial layer to include additional channels. With this

i

##### 4.2. Training with joint spatial-temporal conditions

Upon expanding the image model, a straightforward method might be to train the video model using paired depth-video

1st batch 4th batch 7th batch 13th batch

introduced flow information f, we modify Equation 2 as:

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

(a).w/ocalib.(b).w/calib.

0,t,τ,c,f,ϵ∼N(0,1)∥ϵ−ϵθ(zt,t,τ,c,f)∥22 (4)

LFlowV id = Ez

Throughout the development of our experiments, two particular design choices have been proven crucial for enhancing our final results. First, we opted for vparameterization [41], rather than the more commonly used ϵ-parameterization. This finding is consistent with other video diffusion models, such as Gen-1 [12] and Imagen Video [17] (see ablation in Section 5.4). Second, incorporating additional elements beyond the flow-warped video would further improve the performance. Specifically, including the first frame as a constant video sequence, I1st = {I1,...,I1}, and integrating the occlusion masks O = {O1bwd→1,...,ONbwd→1} enhanced the overall output quality. We process I1st by transforming it into a latent representation and then concatenating it with the noisy latent, similar to processing IW. For O, we resize the binary mask to match the latent size before concatenating it with the noisy latent. Further study is included in Section 5.4.

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

###### Figure 4. Effect of color calibration in autoregressive evaluation.

- (a) When the autoregressive evaluation goes from the 1st batch to the 13th batch, the results without color calibration become gray.
- (b) The results are more stable with the proposed color calibration.

Ij′ in the generated sequence {I1′,...,IM′ (N−1)+1}, where M is the number of autoregressive batches, we calibrate its

mean and variance to match those of I1′. The effect of calibration is shown in Figure 4(b), where the global color is preserved across autoregressive batches.

##### 4.3. Generation: edit the first frame then propagate

Ij′ − mean(Ij′)

During the generation, we want to transfer the input video I to a new video I′ with the target prompt τ′. To effectively leverage the prevalent I2I models, we adopt an editpropagate method. This begins with editing the first frame I1 using I2I models, resulting in an edited first frame I1′. We then propagate the edits to subsequent ith frame by using the flow Fi→1 and the occlusion mask Oibwd→1, derived from the input video I. This process yields the flow-warped edited video I′W = {I1′W,...,IN′W}. We input I′W into the same encoder E and concatenate the resulting flow latent f with a randomly initialized Gaussian noise zT drawn from the normal distribution N. The spatial conditions from the input video are also used to guide the structural layout of the synthesized video. Intuitively, the flow-warped edited video serves as a texture reference while spatial controls regularize the generation, especially when we have inaccurate flow. After DDIM denoising, the denoised latent z0 is brought back to pixel space with a decoder D to get the final output.

Ij′′ =

j) × std(I1′) + mean(I1′) (5)

std(I′

Another advantageous strategy we discovered is the integration of self-attention features from DDIM inversion, a technique also employed in works like FateZero [35] and TokenFlow [13]. This integration helps preserve the original structure and motion in the input video. Concretely, we use DDIM inversion to invert the input video with the original prompt and save the intermediate self-attention maps at various timesteps, usually 20. During the generation with the target prompt, we substitute the keys and values in the selfattention modules with these pre-stored maps. Then, during the generation process guided by the target prompt, we replace the keys and values within the self-attention modules with previously saved corresponding maps.

### 5. Experiments

In addition to offering the flexibility to select I2I models for initial frame edits, our model is inherently capable of producing extended video clips in an autoregressive manner. Once the first N edited frames {I1′,...,IN′ } are generated, the Nth frame IN′ can be used as the starting point for editing the subsequent batch of frames {IN,...,I2N−1}. However, a straightforward autoregressive approach may lead to a grayish effect, where the generated images progressively become grayer, see Figure 4(a). We believe this is a consequence of the lossy nature of the encoder and decoder, a phenomenon also noted in Rerender [49]. To mitigate this issue, we introduce a simple global color calibration technique that effectively reduces the graying effect. Specifically, for each frame

##### 5.1. Settings

Implementation Details We train our model with 100k videos from Shutterstock [1]. For each training video, we sequentially sample 16 frames with interval {2,4,8}, which represent videos lasting {1,2,4} seconds (taking videos with FPS of 30). The resolution of all images, including input frames, spatial condition images, and flow warped frames, is set to 512×512 via center crop. We train the model with a batch size of 1 per GPU and a total batch size of 8 with 8 GPUs. We employ AdamW optimizer [28] with a learning rate of 1e-5 for 100k iterations. As detailed in our method, we train the major U-Net and ControlNet U-Net joint branches

[Figure 105]

|Input|
|---|

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

|Input|
|---|

CoDeFInput

InputCoDeF

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

FlowVid(Ours)Per-frame

Per-frameFlowVid(Ours)

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

RerenderTokenFlow

TokenFlowRerender

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

(a). Prompt: a pirate is rowing a boat on a lake. (b). Prompt: a oil painting of a tiger walking.

Figure 5. Qualitative comparison with representative V2V models. Our method stands out in terms of prompt alignment and overall video quality. We highly encourage readers to refer to video comparisons in our supplementary videos.

with v-parameterization [41]. The training takes four days on one 8-A100-80G node.

We manually design 115 prompts for these videos, spanning from stylization to object swap. Besides, we also collect 50 Shutterstock videos [1] with 200 designed prompts. We conduct both qualitative (see Section 5.2) and quantitative comparisons (see Section 5.3) with state-of-the-art methods including Rerender [49], CoDeF [32] and TokenFlow [13]. We use their official codes with the default settings.

During generation, we first generate keyframes with our trained model and then use an off-the-shelf frame interpolation model, such as RIFE [21], to generate non-key frames. By default, we produce 16 key frames at an interval of 4, corresponding to a 2-second clip at 8 FPS. Then, we use RIFE to interpolate the results to 32 FPS. We employ classifier-free guidance [15] with a scale of 7.5 and use 20 inference sampling steps. Additionally, the Zero SNR noise scheduler [27] is utilized. We also fuse the self-attention features obtained during the DDIM inversion of corresponding key frames from the input video, following FateZero [35]. We evaluate our FlowVid with two different spatial conditions: canny edge maps [5] and depth maps [38]. A comparison of these controls can be found in Section 5.4.

##### 5.2. Qualitative results

In Figure 5, we qualitatively compare our method with several representative approaches. Starting with a per-frame baseline directly applying I2I models, ControlNet, to each frame. Despite using a fixed random seed, this baseline often results in noticeable flickering, such as in the man’s clothing and the tiger’s fur. CoDeF [32] produces outputs with significant blurriness when motion is big in input video, evident in areas like the man’s hands and the tiger’s face. Rerender [49] often fails to capture large motions, such as the movement of paddles in the left example. Also, the color of the edited

Evaluation We select the 25 object-centric videos from the public DAVIS dataset [34], covering humans, animals, etc.

Table 1. Quantitative comparison with existing V2V models. The preference rate indicates the frequency the method is preferred among all the four methods in human evaluation. Runtime shows the time to synthesize a 4-second video with 512×512 resolution on one A-100-80GB. Cost is normalized with our method.

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

1stframe10thframe

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

Preference rate Runtime (mean ± std %) ↑ (mins) ↓ Cost ↓

TokenFlow 40.4 ± 5.3 15.8 10.5 × Rerender 10.2 ± 7.1 10.8 7.2 × CoDeF 3.5 ± 1.9 4.6 3.1 × FlowVid (Ours) 45.7 ± 6.4 1.5 1.0 ×

(I). spatial controls (II). flow warped vid (III). flow occlusion (IV). first frame

(a) Condition types.

Condition choices

Winning rate ↑ (I) (II) (III) (IV)

tiger’s legs tends to blend in with the background. TokenFlow [13] occasionally struggles to follow the prompt, such as transforming the man into a pirate in the left example. It also erroneously depicts the tiger with two legs for the first frame in the right example, leading to flickering in the output video. In contrast, our method stands out in terms of editing capabilities and overall video quality, demonstrating superior performance over these methods. We highly encourage readers to refer to more video comparisons in our supplementary videos.

✓ × × × 9%

✓ ✓ × × 38% ✓ ✓ ✓ × 42 %

(b) Winning rate over our FlowVid (I + II + III + IV).

Figure 6. Ablation study of condition combinations. (a) Four types of conditions. (b) The different combinations all underperform our final setting which combines all four conditions.

The total runtime, including image processing, model operation, and frame interpolation, is approximately 1.5 minutes. This is significantly faster than CoDeF (4.6 minutes), Rerender (10.8 minutes) and TokenFlow (15.8 minutes), being 3.1×, 7.2×, and 10.5 × faster, respectively. CoDeF requires per-video optimization to construct the canonical image. While Rerender adopts a sequential method, generating each frame one after the other, our model utilizes batch processing, allowing for more efficient handling of multiple frames simultaneously. In the case of TokenFlow, it requires a large number of DDIM inversion steps (typically around 500) for all frames to obtain the inverted latent, which is a resource-intensive process. We further report the runtime breakdown (Figure 10) in the Appendix.

##### 5.3. Quantitative results

User study We conducted a human evaluation to compare our method with three notable works: CoDeF [32], Rerender [49], and TokenFlow [13]. The user study involves 25 DAVIS videos and 115 manually designed prompts. Participants are shown four videos and asked to identify which one has the best quality, considering both temporal consistency and text alignment. The results, including the average preference rate and standard deviation from five participants for all methods, are detailed in Table 1. Our method achieved a preference rate of 45.7%, outperforming CoDeF (3.5%), Rerender (10.2%), and TokenFlow (40.4%). During the evaluation, we observed that CoDeF struggles with significant motion in videos. The blurry constructed canonical images would always lead to unsatisfactory results. Rerender occasionally experiences color shifts and bright flickering. TokenFlow sometimes fails to sufficiently alter the video according to the prompt, resulting in an output similar to the original video.

##### 5.4. Ablation study

Condition combinations We study the four types of conditions in Figure 6(a): (I) Spatial controls: such as depth maps [38]. (II) Flow warped video: frames warped from the first frame using optical flow. (III) Flow occlusion: masks indicate which parts are occluded (marked as white). (IV) First frame. We evaluate combinations of these conditions in Figure 6(b), assessing their effectiveness by their winning rate against our full model which contains all four conditions. The spatial-only condition achieved a 9% winning rate, limited by its lack of temporal information. Including flow warped video significantly improved the winning rate to 38%, underscoring the importance of temporal guidance. We use gray pixels to indicate occluded areas, which might blend in with the original gray colors in the images. To avoid

Pipeline runtime We also compare runtime efficiency with existing methods in Table 1. Video lengths can vary, resulting in different processing times. Here, we use a video containing 120 frames (4 seconds video with FPS of 30). The resolution is set to 512 × 512. Both our FlowVid model and Rerender [49] use a key frame interval of 4. We generate 31 keyframes by applying autoregressive evaluation twice, followed by RIFE [21] for interpolating the non-key frames.

[Figure 143]

[Figure 144]

[Figure 145]

1st frame 20th frame 1st frame 20th frame

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

InputvideoFlowwarpingFinalprediction

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

(a). Input frame (b). Spatial control (c). Style transfer (d). Object swap

- Figure 7. Ablation study of different spatial conditions. Canny edge and depth map are estimated from the input frame. Canny edge provides more detailed controls (good for stylization) while depth map provides more editing flexibility (good for object swap).

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

(a).-prediction(b).-prediction

[Figure 164]

[Figure 165]

- Figure 8. Ablation study of different parameterizations. ϵprediction often predicts unnatural global color while v-prediction doesn’t. Prompt: ’a man is running on Mars’.

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

(a). Misaligned first frame (b). Big occlusion

Figure 9. Limitations of FlowVid. Failure cases include (a) the edited first frame doesn’t align structurally with the original first frame, and (b) large occlusions caused by fast motion.

our evaluation, we found canny edge works better when we want to keep the structure of the input video as much as possible, such as stylization. The depth map works better if we have a larger scene change, such as an object swap, which requires more considerable editing flexibility.

v-prediction and ϵ-prediction While ϵ-prediction is commonly used for parameterization in diffusion models, we found it may suffer from unnatural global color shifts across frames, as shown in Figure 8. Even though all these two methods use the same flow warped video, the ϵ-prediction introduces an unnatural grayer color. This phenomenon is also found in Imagen-Video [17].

potential confusion, we further include a binary flow occlusion mask, which better helps the model to tell which part is occluded or not. The winning rate is further improved to 42%. Finally, we added the first frame condition to provide better texture guidance, particularly useful when the occlusion mask is large and few original pixels remain.

Different control type: edge and depth We study two types of spatial conditions in our FlowVid: canny edge [5] and depth map [38]. Given an input frame as shown in Figure 7(a), the canny edge retains more details than the depth map, as seen from the eyes and mouth of the panda. The strength of spatial control would, in turn, affect the video editing. For style transfer prompt ’A Chinese ink painting of a panda eating bamboo’, as shown in Figure 7(c), the output of canny condition could keep the mouth of the panda in the right position while the depth condition would guess where the mouth is and result in an open mouth. The flexibility of the depth map, however, would be beneficial if we are doing object swap with prompt ’A koala eating bamboo’, as shown in Figure 7(d); the canny edge would put a pair of panda eyes on the face of the koala due to the strong control, while depth map would result in a better koala edit. During

##### 5.5. Limitations

Although our FlowVid achieves significant performance, it does have some limitations. First, our FlowVid heavily relies on the first frame generation, which should be structurally aligned with the input frame. As shown in Figure 9(a), the edited first frame identifies the hind legs of the elephant as the front nose. The erroneous nose would propagate to the following frame and result in an unsatisfactory final prediction. The other challenge is when the camera or the object moves so fast that large occlusions occur. In this case, our model would guess, sometimes hallucinate, the missing blank regions. As shown in Figure 9(b), when the ballerina turns her body and head, the entire body part is masked out. Our model manages to handle the clothes but turns the back of the head into the front face, which would be confusing if displayed in a video.

### 6. Conclusion

In this paper, we propose a consistent video-to-video synthesis method using joint spatial-temporal conditions. In contrast to prior methods that strictly adhere to optical flow, our approach incorporates flow as a supplementary reference in synergy with spatial conditions. Our model can adapt existing image-to-image models to edit the first frame and propagate the edits to consecutive frames. Our model is also able to generate lengthy videos via autoregressive evaluation. Both qualitative and quantitative comparisons with current methods highlight the efficiency and high quality of our proposed techniques.

### 7. Acknowledgments

We would like to express sincere gratitude to Yurong Jiang, Chenyang Qi, Zhixing Zhang, Haoyu Ma, Yuchao Gu, Jonas Schult, Hung-Yueh Chiang, Tanvir Mahmud, Richard Yuan for the constructive discussions.

Feng Liang and Diana Marculescu were supported in part by the ONR Minerva program, iMAGiNE - the Intelligent Machine Engineering Consortium at UT Austin, and a UT Cockrell School of Engineering Doctoral Fellowship.

### References

- [1] Stock footage video, royalty-free hd, 4k video clips, 2023. 2, 5, 6
- [2] Yogesh Balaji, Seungjun Nah, Xun Huang, Arash Vahdat, Jiaming Song, Karsten Kreis, Miika Aittala, Timo Aila, Samuli Laine, Bryan Catanzaro, et al. ediffi: Text-to-image diffusion models with an ensemble of expert denoisers. arXiv preprint arXiv:2211.01324, 2022. 2
- [3] Omer Bar-Tal, Dolev Ofri-Amar, Rafail Fridman, Yoni Kasten, and Tali Dekel. Text2live: Text-driven layered image and video editing. In European conference on computer vision, pages 707–723. Springer, 2022. 3
- [4] Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18392–18402, 2023. 2
- [5] John Canny. A computational approach to edge detection. IEEE Transactions on pattern analysis and machine intelligence, (6):679–698, 1986. 3, 6, 8
- [6] Duygu Ceylan, Chun-Hao P Huang, and Niloy J Mitra. Pix2video: Video editing using image diffusion. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 23206–23217, 2023. 2, 3, 4
- [7] Weifeng Chen, Jie Wu, Pan Xie, Hefeng Wu, Jiashi Li, Xin Xia, Xuefeng Xiao, and Liang Lin. Control-a-video: Controllable text-to-video generation with diffusion models. arXiv preprint arXiv:2305.13840, 2023. 3
- [8] Ernie Chu, Tzuhsuan Huang, Shuo-Yen Lin, and Jun-Cheng Chen. Medm: Mediating image diffusion models for videoto-video translation with temporal correspondence guidance. arXiv preprint arXiv:2308.10079, 2023. 3

- [9] Ernie Chu, Shuo-Yen Lin, and Jun-Cheng Chen. Video controlnet: Towards temporally consistent synthetic-to-real video translation using conditional image diffusion models. arXiv preprint arXiv:2305.19193, 2023. 3
- [10] Guillaume Couairon, Jakob Verbeek, Holger Schwenk, and Matthieu Cord. Diffedit: Diffusion-based semantic image editing with mask guidance. arXiv preprint arXiv:2210.11427,

2022. 2

- [11] Xiaoliang Dai, Ji Hou, Chih-Yao Ma, Sam Tsai, Jialiang Wang, Rui Wang, Peizhao Zhang, Simon Vandenhende, Xiaofang Wang, Abhimanyu Dubey, et al. Emu: Enhancing image generation models using photogenic needles in a haystack. arXiv preprint arXiv:2309.15807, 2023. 2
- [12] Patrick Esser, Johnathan Chiu, Parmida Atighehchian, Jonathan Granskog, and Anastasis Germanidis. Structure and content-guided video synthesis with diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7346–7356, 2023. 3, 5
- [13] Michal Geyer, Omer Bar-Tal, Shai Bagon, and Tali Dekel. Tokenflow: Consistent diffusion features for consistent video editing. arXiv preprint arXiv:2307.10373, 2023. 2, 3, 5, 6, 7, 11
- [14] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Prompt-to-prompt image editing with cross attention control. arXiv preprint arXiv:2208.01626, 2022. 2
- [15] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022. 6
- [16] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 3
- [17] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, et al. Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303, 2022. 5, 8
- [18] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video diffusion models. arXiv:2204.03458, 2022. 4
- [19] Zhihao Hu and Dong Xu. Videocontrolnet: A motionguided video-to-video translation framework by using diffusion model with controlnet. arXiv preprint arXiv:2307.14073,

2023. 2, 3

- [20] Lianghua Huang, Di Chen, Yu Liu, Yujun Shen, Deli Zhao, and Jingren Zhou. Composer: Creative and controllable image synthesis with composable conditions. arXiv preprint arXiv:2302.09778, 2023. 3
- [21] Zhewei Huang, Tianyuan Zhang, Wen Heng, Boxin Shi, and Shuchang Zhou. Real-time intermediate flow estimation for video frame interpolation. In Proceedings of the European Conference on Computer Vision (ECCV), 2022. 6, 7, 11
- [22] Ondˇrej Jamriˇska, S´ˇarka Sochorov´a, Ondˇrej Texler, Michal Luk´aˇc, Jakub Fiˇser, Jingwan Lu, Eli Shechtman, and Daniel S`ykora. Stylizing video by example. ACM Transactions on Graphics (TOG), 38(4):1–11, 2019. 3
- [23] Yoni Kasten, Dolev Ofri, Oliver Wang, and Tali Dekel. Layered neural atlases for consistent video editing. ACM Transactions on Graphics (TOG), 40(6):1–12, 2021. 3

- [24] Bahjat Kawar, Shiran Zada, Oran Lang, Omer Tov, Huiwen Chang, Tali Dekel, Inbar Mosseri, and Michal Irani. Imagic: Text-based real image editing with diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6007–6017, 2023. 2
- [25] Levon Khachatryan, Andranik Movsisyan, Vahram Tadevosyan, Roberto Henschel, Zhangyang Wang, Shant Navasardyan, and Humphrey Shi. Text2video-zero: Textto-image diffusion models are zero-shot video generators. arXiv preprint arXiv:2303.13439, 2023. 2, 3, 4
- [26] Yao-Chih Lee, Ji-Ze Genevieve Jang, Yi-Ting Chen, Elizabeth Qiu, and Jia-Bin Huang. Shape-aware text-driven layered video editing. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14317– 14326, 2023. 3
- [27] Shanchuan Lin, Bingchen Liu, Jiashi Li, and Xiao Yang. Common diffusion noise schedules and sample steps are flawed. arXiv preprint arXiv:2305.08891, 2023. 6
- [28] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017. 5
- [29] Simon Meister, Junhwa Hur, and Stefan Roth. Unflow: Unsupervised learning of optical flow with a bidirectional census loss. In Proceedings of the AAAI conference on artificial intelligence, 2018. 4
- [30] Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. Sdedit: Guided image synthesis and editing with stochastic differential equations. arXiv preprint arXiv:2108.01073, 2021. 2
- [31] Chong Mou, Xintao Wang, Liangbin Xie, Jian Zhang, Zhongang Qi, Ying Shan, and Xiaohu Qie. T2i-adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models. arXiv preprint arXiv:2302.08453, 2023. 2, 3
- [32] Hao Ouyang, Qiuyu Wang, Yuxi Xiao, Qingyan Bai, Juntao Zhang, Kecheng Zheng, Xiaowei Zhou, Qifeng Chen, and Yujun Shen. Codef: Content deformation fields for temporally consistent video processing. arXiv preprint arXiv:2308.07926, 2023. 2, 3, 4, 6, 7, 11
- [33] Gaurav Parmar, Krishna Kumar Singh, Richard Zhang, Yijun Li, Jingwan Lu, and Jun-Yan Zhu. Zero-shot image-to-image translation. In ACM SIGGRAPH 2023 Conference Proceedings, pages 1–11, 2023. 2
- [34] Jordi Pont-Tuset, Federico Perazzi, Sergi Caelles, Pablo Arbel´aez, Alex Sorkine-Hornung, and Luc Van Gool. The 2017 davis challenge on video object segmentation. arXiv preprint arXiv:1704.00675, 2017. 2, 6
- [35] Chenyang Qi, Xiaodong Cun, Yong Zhang, Chenyang Lei, Xintao Wang, Ying Shan, and Qifeng Chen. Fatezero: Fusing attentions for zero-shot text-based video editing. arXiv preprint arXiv:2303.09535, 2023. 2, 3, 4, 5, 6
- [36] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 11

- [37] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 1(2): 3, 2022. 2
- [38] Ren´e Ranftl, Katrin Lasinger, David Hafner, Konrad Schindler, and Vladlen Koltun. Towards robust monocular depth estimation: Mixing datasets for zero-shot cross-dataset transfer. IEEE transactions on pattern analysis and machine intelligence, 44(3):1623–1637, 2020. 3, 6, 7, 8
- [39] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 2, 3
- [40] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in Neural Information Processing Systems, 35:36479–36494, 2022. 2
- [41] Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. arXiv preprint arXiv:2202.00512, 2022. 5, 6
- [42] Zachary Teed and Jia Deng. Raft: Recurrent all-pairs field transforms for optical flow. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part II 16, pages 402–419. Springer, 2020. 2, 3
- [43] Narek Tumanyan, Michal Geyer, Shai Bagon, and Tali Dekel. Plug-and-play diffusion features for text-driven image-toimage translation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1921–1930, 2023. 2
- [44] Wen Wang, Kangyang Xie, Zide Liu, Hao Chen, Yue Cao, Xinlong Wang, and Chunhua Shen. Zero-shot video editing using off-the-shelf image diffusion models. arXiv preprint arXiv:2303.17599, 2023. 3
- [45] Xiang Wang, Hangjie Yuan, Shiwei Zhang, Dayou Chen, Jiuniu Wang, Yingya Zhang, Yujun Shen, Deli Zhao, and Jingren Zhou. Videocomposer: Compositional video synthesis with motion controllability. arXiv preprint arXiv:2306.02018,

2023. 3

- [46] Jay Zhangjie Wu, Yixiao Ge, Xintao Wang, Stan Weixian Lei, Yuchao Gu, Yufei Shi, Wynne Hsu, Ying Shan, Xiaohu Qie, and Mike Zheng Shou. Tune-a-video: One-shot tuning of image diffusion models for text-to-video generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7623–7633, 2023. 2, 3, 4
- [47] Haofei Xu, Jing Zhang, Jianfei Cai, Hamid Rezatofighi, and Dacheng Tao. Gmflow: Learning optical flow via global matching. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8121–8130,

2022. 2, 3

- [48] Haofei Xu, Jing Zhang, Jianfei Cai, Hamid Rezatofighi, Fisher Yu, Dacheng Tao, and Andreas Geiger. Unifying flow, stereo and depth estimation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2023. 2, 3, 4

- [49] Shuai Yang, Yifan Zhou, Ziwei Liu, and Chen Change Loy. Rerender a video: Zero-shot text-guided video-to-video translation. arXiv preprint arXiv:2306.07954, 2023. 2, 3, 4, 5, 6, 7, 11
- [50] Yabo Zhang, Yuxiang Wei, Dongsheng Jiang, Xiaopeng Zhang, Wangmeng Zuo, and Qi Tian. Controlvideo: Trainingfree controllable text-to-video generation. arXiv preprint arXiv:2305.13077, 2023. 3
- [51] Zhixing Zhang, Ligong Han, Arnab Ghosh, Dimitris N Metaxas, and Jian Ren. Sine: Single image editing with textto-image diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6027–6037, 2023. 2
- [52] Min Zhao, Rongzhen Wang, Fan Bao, Chongxuan Li, and Jun Zhu. Controlvideo: Adding conditional control for one shot text-to-video editing. arXiv preprint arXiv:2305.17098, 2023. 3

### A. Webpage Demo

We highly recommend looking at our demo web by opening the https://jeff-liangf.github.io/ projects/flowvid/ to check the video results.

### B. Quantitative comparisons

##### B.1. CLIP scores

Inspired by previous research, we utilize CLIP [36] to evaluate the generated videos’ quality. Specifically, we measure 1) Temporal Consistency (abbreviated as Tem-Con), which is the mean cosine similarity across all sequential frame pairs, and 2) Prompt Alignment (abbreviated as Pro-Ali), which calculates the mean cosine similarity between a given text prompt and all frames in a video. Our evaluation, detailed in Table 2, includes an analysis of 116 video-prompt pairs from the DAVIS dataset. Notably, CoDeF [32] and Rerender [49] exhibit lower scores in both temporal consistency and prompt alignment, aligning with the findings from our user study. Interestingly, TokenFlow shows superior performance in maintaining temporal consistency. However, it is important to note that TokenFlow occasionally underperforms in modifying the video, leading to outputs that closely resemble the original input. Our approach not only secures the highest ranking in prompt alignment but also performs commendably in temporal consistency, achieving second place.

##### B.2. Runtime breakdown

We benchmark the runtime with a 512 × 512 resolution video containing 120 frames (4 seconds video with FPS of 30). Our runtime evaluation was conducted on a 512 × 512 resolution video comprising 120 frames, equating to a 4-second clip at 30 frames per second (FPS). Both our methods, FlowVid, and Rerender [49], initially create key frames

Table 2. CLIP score comparisons. ’Tem-Con’ stands for temporal consistency, and ’Pro-Ali’ stands for prompt alignment.

Method Tem-Con ↑ Pro-Ali ↑

CoDeF [32] 96.98 30.83 Rerender [49] 96.88 31.84 TokenFlow [13] 97.30 33.11 Ours 97.08 33.20

| |1.2<br><br>3.5<br><br>1.1<br><br>5.2<br><br>5.6<br><br>10.4<br><br>5.4<br><br>1.5<br><br>4.6<br><br>10.8<br><br>15.8 FlowVid key-frame gen.<br><br>FlowVid frame interp.<br><br>CoDeF training<br><br>CoDeF inference<br><br>Rerender key-frame gen.<br><br>Rerender frame interp.<br><br>TokenFlow DDIM inversion<br><br>TokenFlow inference| | | | |
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

- 2

4

6

8

10

12

14

16

Time(minutes)

Figure 10. Runtime breakdown of generating a 4-second 512 × 512 resolution video with 30 FPS. Time is measured on one A10080GB GPU.

followed by the interpolation of non-key frames. For these techniques, we opted for a keyframe interval of 4. FlowVid demonstrates a marked efficiency in keyframe generation, completing 31 keyframes in just 1.1 minutes, compared to Rerender’s 5.2 minutes. This significant time disparity is attributed to our batch processing approach in FlowVid, which handles 16 images simultaneously, unlike Rerender’s sequential, single-image processing method. In the aspect of frame interpolation, Rerender employs a reference-based EbSynth method, which relies on input video’s non-key frames for interpolation guidance. This process is notably timeconsuming, requiring 5.6 minutes to interpolate 90 non-key frames. In contrast, our method utilizes a non-referencebased approach, RIFE [21], which significantly accelerates the process. Two other methods are CoDeF [32] and TokenFlow [13], both of which necessitate per-video preparation. Specifically, CoDeF involves training a model for reconstructing the canonical image, while TokenFlow requires a 500-step DDIM inversion process to acquire the latent representation. CoDeF and TokenFlow require approximately

- 3.5 minutes and 10.4 minutes, respectively, for this initial preparation phase.

0

FlowVid CoDeF Rerender TokenFlow

