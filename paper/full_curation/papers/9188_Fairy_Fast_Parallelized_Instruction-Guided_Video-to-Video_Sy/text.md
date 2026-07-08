## Fairy: Fast Parallellized Instruction-Guided Video-to-Video Synthesis

Bichen Wu Ching-Yao Chuang Xiaoyan Wang Yichen Jia Kapil Krishnakumar Tong Xiao Feng Liang Licheng Yu Peter Vajda GenAI, Meta Project page: https://fairy-video2video.github.io

# arXiv:2312.13834v1[cs.CV]20Dec2023

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[ Input Video ] [ Input Video ]

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

Turn into a metal knight sculpture Turn into a wood sculpture

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

In Van Gogh Style In Picasso Style

[Figure 29]

[Figure 30]

Figure 1. Fairy for Instruction-Guided Video Editing. Given a video and an instruction for editing, Fairy performs accurate edits while ensuring temporal coherence. Remarkably efficient, 120 frames of 512×384 video can be generated in just 14 seconds. We refer readers to our supplementary material to check the results in video format.

### Abstract

erates 120-frame 512×384 videos (4-second duration at 30 FPS) in just 14 seconds, outpacing prior works by at least 44×. A comprehensive user study, involving 1000 generated samples, confirms that our approach delivers superior quality, decisively outperforming established methods.

In this paper, we introduce Fairy, a minimalist yet robust adaptation of image-editing diffusion models, enhancing them for video editing applications. Our approach centers on the concept of anchor-based cross-frame attention, a mechanism that implicitly propagates diffusion features across frames, ensuring superior temporal coherence and high-fidelity synthesis. Fairy not only addresses limitations of previous models, including memory and processing speed. It also improves temporal consistency through a unique data augmentation strategy. This strategy renders the model equivariant to affine transformations in both source and target images. Remarkably efficient, Fairy gen-

### 1. Introduction

The advent of generative artificial intelligence has heralded a new era of creative potential, characterized by the ability to create or modify content in an effortless manner. In particular, image editing has undergone significant evolution, driven by text-to-image diffusion models pretrained on billion-scale datasets. This surge has catalyzed a vast array

of applications in image editing and content creation.

Building on the accomplishments of image-based models, the next natural frontier is transitioning these capabilities to the temporal dimension to enable effortless and creative video editing. A direct strategy to leap from image to video, is to simply process a video on a frame-by-frame basis using an image model. Nonetheless, generative image editing is inherently high-variance – there are countless ways to edit a given image based on the same text prompt. As a result, it is difficult to maintain temporal coherence if each frame is edited independently [30].

Previous and concurrent studies have proposed several ways to improve the temporal consistency, and one promising paradigm is what we call tracking-and-propagation: one first apply an image editing model on one or a few frames, then tracks pixels across all frames and propagate the edit to the entire video. Existing works [4, 6, 12, 14, 16, 19, 24, 33] track pixels mainly through optical flow or by reconstructing videos as some canonical layered representations. Despite some successful applications, this paradigm is not robust, since tracking is an unsolved computer vision challenge. Existing methods, including optical flow or layered video representation, often struggle with videos with large motion and complex dynamics.

In this work, we introduce Fairy, a versatile and efficient video-to-video synthesis framework that generates high-quality videos with remarkable speed (Figure 1). Our work re-examines the tracking-and-propagation paradigm under the context of diffusion model features. In particular, we bridge cross-frame attention with correspondence estimation, showing that it temporally tracks and propagates intermediate features inside a diffusion model. The crossframe attention map can be interpreted as a similarity metric assessing the correspondence between tokens throughout various frames, where features from one semantic region will assign higher attention to similar semantic regions in other frames, as shown in Figure 3. Consequently, the current feature representations are refined and propagated through a weighted sum of similar regions across frames via attention, effectively minimizing feature disparity between frames, which translates to improved temporal consistency.

The analysis gives rise to our anchor-based model, the central component of Fairy. To ensure temporal consistency, we sample K anchor frames from which we extract diffusion features, and the extracted features define a set global features to be propagated to successive frames. When generating each new frame, we replace the selfattention layer with cross-frame attention with respect to the cached features of anchor frames. With cross-frame attention, the tokens in each frame take the features in anchor frames that exhibit analogous semantic content, thereby enhancing consistency. In addition, by sampling K anchor frames instead of computing cross-attention with respect

to all frames, Fairy achieves several advantages: (1) it ensures temporal consistency by sharing the same global features, (2) it overcomes the memory issue due to extensive frame number, (3) it enhances processing speed through the caching of anchor frame features, and (4) it streamlines parallel computation, thereby facilitating remarkably rapid generation on multiple GPUs.

Despite the improvement from anchor-based cross-frame attention, the model is still sensitive to minor variations within the input frames, even with the same text prompt and initial latent noise. Such small changes could stem from natural movements within a video sequence or from affine transformations applied to the input. The gold standard solution is to train the model with pairs of original and edited videos, thereby accommodate it to recognize and adapt to such variations. However, collecting such a dataset is far from straightforward. To emulate these transformations, we employ a data augmentation strategy. Starting with an input image and its edited counterpart, we apply a sequence of affine transformations to both, generating successive frames. The assumption is that the affine transformations applied to input images should correspondingly affect the edited images. This method of equivariant finetuning leads to notable enhancements in temporal consistency.

To verify the effectiveness of Fairy, we conduct a largescale evaluation consists of 1000 generated videos. Both human evaluation and quantitative metrics confirm that our model achieves significantly better quality. Moreover, thanks to the simplicity of the design and the parallelizable architecture, Fairy achieves >44x speedup over baselines.

In short, this work makes the following contributions: (1) We adopt a series of simple yet effective adaptions that transform an image-editing model for video-to-video synthesis. (2) We evaluate our approach via extensive human study with 1000 generated videos, confirming that Fairy delivers superior quality over prior state-of-the art methods. (3) Fairy is blazing fast, achieving >44x speedup over previous methods when utilizing 8-gpu parallel generation.

### 2. Related works

Conditional video generation: Following the success of diffusion models in text-to-image generation [7, 22, 23, 25], there has been a surge in video generation. Based on a textto-video model, video-to-video generation can be achieved by conditioning the model on attributes extracted from a source video. For example, Gen-1 [9] conditions on the estimated depth while VideoComposer [31] integrates additional cues, such as depth, motion vectors, sketches, among others. Building such models requires training on video datasets, which are much more scarce than image datasets [26]. Training such models also imposes considerable computational demands. Consequently, these constraints confine video models to reduced resolution, shorter duration,

and smaller model size, leading to a decline in visual quality when contrasted with contemporary image generation models. In comparison, our model is adapted from a pretrained image-to-image model. Our finetuning only requires image data, and the training cost (30 hours on 8 A100 GPUs) is substantially smaller than video models.

Tracking and propagation: this paradigm involves initiating edits on a single image, identifying pixel correspondences across the video sequence, then propagating the edit. The key in this approach lies in tracking. Numerous efforts [12, 24, 33] have adopted optical flow, keypoint tracking, or other motion cues to tackle this. Another stream of efforts [4, 6, 14, 16, 19] reconstruct the video using a multi-layer canonical representation, associating pixels to canonical points on the representation. However, video tracking is an unsolved computer vision challenge and often fails on complex videos. Additionally, tracking-andpropagation does not allow edits to alter object contours, which breaks the pixel correspondence. Instead of tracking in pixel space, our model leverages cross-frame attention to implicitly track corresponding regions and propagate features to reduce frame discrepancy. Owing to the robustness and versatility of diffusion features, as also observed in Tang et al. [27], our approach accommodates a broader spectrum of videos and offers enhanced editing flexibility.

Image model adaptation: Many works also adapt image-to-image models to video. For example, [15] modifies self-attention in diffusion models. [32] performs pervideo finetuning and utilizes a inversion-denoising procedure for editing. [10, 17, 21, 29] adapt image-to-image pipelines [5, 11, 28] to edit videos, by modifying/adding cross-frame attention modules, null-text inversion, etc. Most of these methods can only generate video clips with a small number of frames, while [10] leverages a nearestneighbor field on diffusion features to propagate key frame features to more frames. Our model improves the design of spatial temporal attention [15, 17, 21, 29] to anchor-based cross-frame attention, which enables generating long videos with arbitrarily many frames. We further improves its temporal consistency by equivariant finetuning. Our work bears resemblance to the concurrent work [10]. To edit a video, [10] first performs a latent inversion on the original video, extract a nearest-neighbor field, which is then used for feature propagation to generate the target video. Our pipeline is much simpler and more efficient. We do not require latent inversion; and the feature propagation is achieved through attention; our architectures naturally allows parallel generation. As a result, our model is 53 times faster than [10].

### 3. Preliminaries

Video-to-Video Diffusion Models In this work, we primarily focus on instruction-guided video editing. Given an input video with N frames I = {I1,...,IN} ∈ IN, the

goal is to edit it into a new video I′ = {I1′,...,IN′} ∈ IN according to an natural language instruction c ∈ T that preserves the semantic of the original video. A straightforward baseline is to adopt an image-based editting model f : (I,T ) → I to edit the video frame by frame: I′ = {f(I1,c),...,f(IN,c)}. In this work, we build upon this line of work and improve the consistency with a variant of cross-frame attention.

Self-attention and Cross-frame attention Self-attention has played a crucial role in the diffusion networks. In a selfattention block, features of tokens are projected into queries Q ∈ Rn×d, keys K ∈ Rn×d, and values V ∈ Rn×d, where the output is defined as

SelfAttention(Q,K,V ) = softmax

##### QKT √

d

##### V .

The output from the softmax is commonly referred to as the attention score or attention map. Given N frames, to extend the self-attention to cross-frame attention, one can simply concat the keys and values from all frames, e.g., K∗ = [K1,··· ,KN], and compute the self-attention as Self-Attention(Q,K∗,V ∗). In particular, cross-frame attention provides temporal modeling capability by attending to other frames and shows encouraging results in improving temporal consistency [17, 30].

### 4. Implicit Tracking via Cross-frame Attention

We first bridge cross-frame attention with correspondence estimation, fostering a straightforward yet effective feature propagation mechanism for video-to-video generation.

The primary objective of self-attention is to select appropriate values V with the attention scores determined by QKT. In the case of cross-frame attention, given a token location p in a frame, the attention score is computed by the cosine similarity between Qp,: and each token in K∗, where the key values V ∗ are the features of tokens across both spatial and temporal dimension.

It is noteworthy that the mathematical formulation exhibits profound similarities to feature propagation mechanisms. Specifically, the attention score serves as the estimated correspondence, and the output of attention module could be interpreted as a fused representation of warped features derived from successive frames. We will empirically substantiate this hypothesis through analyses of the tracking behavior inherent in the attention score.

#### 4.1. Temporal Tracking with Attention Score

In this section, we provide evidences that the attention scores in cross-frame attention implicitly serve as a correspondence estimation across frames. In particular, we adopt a conditional image-to-image diffusion model and examine

[Figure 31]

| |[Figure 32]<br><br>[Figure 33]|
|---|---|

| |[Figure 34]<br><br>[Figure 35]|
|---|---|

[Figure 36]

PositionAccuracy

Figure 3. Visualization of Attention Score. The left image shows the query point p within the current frame, and the right image is the target frame. Cross-frame attention performs accurate temporal correspondence estimation without any finetuning.

Layers Layers

(a) δx = 16 (b) δx = 32

Figure 2. Position Accuracy δx on DAVIS. The cross-frame attention score demonstrates significant tracking proficiency, particularly evident in the initial and final stages of the UNet.

attributed primarily to the reduction in the spatial resolution of the feature maps. For instance, within the seventh layer of the network, the feature map dimensions are constrained to 4 × 4. Figure 3 visualizes the attention score on a target frame given a query point. We can see that the attention map locate the corresponding position in target frame.

the attention map between two frames of a video clip. Consider Qt and Kt as the respective query and key representations corresponding to the frame at time t. To corroborate our conjecture regarding the role of attention scores, we designate a specific query point p at time t and endeavor to ascertain its corresponding coordinate q at a subsequent time t′ through the expression:

Cross-frame Attention ≈ Tracking and Feature Propagation Our experimental findings disclose an unexpectedly potent tracking capability associated with the attention score. These results robustly validate our hypothesis: even in the absence of explicit finetuning, cross-frame attention implicitly executes a formidable feature propagation mechanism. In particular, features V ∗ from alternative frames are transmitted to the current frame based on the correspondence determined through the attention scores.

′T

QtKt

Ap,p′, where A = softmax(

√

q = arg max

),

d

p′

where Ap,p′ denotes the element of the matrix A located at the row index p and column index p′. The correspondence is estimated by selecting the location p′ with the highest attention score with respect to p. For multi-head attention, we average the attention scores from all heads. By evaluating the tracking ability of the proposed estimator, we can verify whether the attention scores are good correspondence estimator for feature propagation.

### 5. Fairy: Fast Video-to-Video Synthesis

Building on the analyses, we present Fairy, an efficacious video-to-video framework that leverages the inherent feature propagation of cross-frame attention. In particular, we propose to propagate the value features from a collection of anchor frames to a candidate frame using cross-frame attention. The performance can be further enhanced through the proposed equivariant finetuning method. We also demonstrate that Fairy is easily parallelizable, facilitating fast generation of arbitrarily long videos.

#### 4.2. Video Tracking Experiments: TAP-Vid

In our evaluation, we utilize the DAVIS datasets from the TAP-Vid [8, 20], with 30 videos clips ranging from 34-104 frames. The frames are resize to 256 × 256 for evaluation. We measure the < δx position accuracy proposed in TAPVid, which calculates the fraction of points that are within δx pixels of their ground truth position. The dimensions of the attention map inherently impose a constraint on the precision achievable in point tracking. Since diffusion UNets adopts spatial downsampling, we configure δx at the values of 16 and 32 for our experiments. We set the number of diffusion step to 10 with Euler ancestral sampler [13].

#### 5.1. Anchor-Based Model

Inspired by prior research in tracking-and-propagation, where the edits to one or a few frames are propagated to the entire video, we sample a set of anchor frames and edit them with an image-based model f : (I,T ) → I. Similarly, our objective is to extend the edits in the anchor frames to the successive frames, but utilizing crossframe attention mechanisms instead of optical flow or explicit point tracking. In particular, given a set of anchor frames Ianc = {I˜1,...,I˜K} ⊆ I = {I1,...,IN}, we treat them as a batch and feed them to the diffusion model f, where the self-attention in the model is replaced with cross-frame attention in a zero-shot manner. Throughout the diffusion process, for each anchor frame I˜n, we store its

Figure 2 shows the position accuracy for attention scores across different layers and diffusion step. We can see that the first and last few layers demonstrate a strong tracking results, achieving over 60%/70% accuracy for δx = 16/32. Interestingly, the results are consistent across different diffusion step, demonstrating the strong tracking ability of cross-frame attention. The observed degradation in accuracy at the central layers of the UNet architecture can be

- (a) Extract and cache anchor features
- (b) Anchor-based Cross-frame Attention

rior quality outputs without succumbing to memory-related constraints. This efficiency underscores our approach’s enhanced scalability and practicality, setting a new benchmark for performance in the realm of video editing.

[Figure 37]

Anchor

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

| | |
|---|---|
| | |

[Figure 42]

[Figure 43]

Q K V

[Figure 44]

anc anc anc

[Figure 45]

Anchor

[Figure 46]

#### 5.2. Equivariant Finetuning

[Figure 47]

While anchor-based attention greatly improves the quality, we still occasionally observed temporal inconsistency. In particular, we found that for generated contents that do not have semantic correspondence with the input, small changes in input frames can cause significant variances in the output frames.

[Figure 48]

|[Figure 49]<br><br>[Figure 50]|
|---|

[Figure 51]

###### Input Anchor

[Figure 52]

[Figure 53]

Q K V

[Figure 54]

To improve the consistency, we leverage the following intuition to design a data augmentation strategy. In particular, if an input frame It differs from It−1 only in the camera position, then the output frame Iˆt and Iˆt−1 should only be different in the camera position as well. This inspires us to come up with a data augmented strategy that can be applied to any image editing dataset to imporve the temporal consistency. Given a pair of images, the original and the edited, denoted as (I,I′), we randomly sample an affine transformation g : I → I and apply them to both images to obtain (g(I),g(I′)). We implement this using torchvision’s random affine transformation [2], setting random rotations degrees to < 5◦, random translation to [−0.05,0.05], random scaling factor to [0.95,1.05], and random shear degrees to [−5◦,5◦] on both axis. We also apply random resized crop, scaling the original image to 288pix, and randomly crop a square image with 256 pix. We then fine-tuned the base image-to-image model to generate the transformed g(I′) given the transformed g(I). The proposed fine-tuning process makes the model equivariant to affine transformations, leading us to denote our approach as equivariant finetuning. Empirically, we observe a notable enhancements in temporal consistency after finetuning (Section 6.3).

Figure 4. Illustration of Attention Blocks (a) Given a set of anchor frames, we extract and cache the attention feature Kanc and Vanc. (b) Given an input frame, we perform cross-frame attention with respect to the cached features of anchor frames.

key and value vectors Kn,l,t,V n,l,t for every cross-frame attention layer l and every diffusion step t in cache. Intuitively, V n,l,t defines a set global features to be propagated to successive frames. To simplify the notation, we will drop the subscript l and t in the following sections.

Let Kanc = [K1,...,KK] and Vanc = [V 1,...,V K] be the concatenated anchor key and value vectors. To edit any frame It ∈ I, we modify the self-attention module to the cross-frame attention with respect to the key and value vectors of anchor frames as follows:

Q[K,Kanc]T √

softmax

[V ,Vanc],

d

where Q,K and V are the self-attention vectors of the input frame It. The idea is that the attention score generated by the softmax facilitates cross-frame tracking by estimating the temporal correspondence between the input frame and anchor frames. The global value vectors then be propagated to input frame by multiplying the attention score with Vanc. By substituting the self-attention module with an anchor-based cross-frame attention mechanism, we found that the model could generate highly consistent video edits. In the default setting, we choose anchor frames uniformly across the video, and we did not notice consistent performance improvement or degradation when adopting different anchor-frame selection strategies.

### 6. Results

We implement Fairy based on an instruction-based image editing model, similar to [5], and replace the model’s selfattention with cross-frame attention. We set the number of anchor frames to 3. Anchor frames are uniformly selected with equal intervals among all frames. The model can accept input with different aspect ratios, and we rescale the input resolution with the longer size to be 512, and keep the aspect ratio unchanged. We edit all frames of the input video, without temporal downsampling. We distribute the computation to 8 A100 GPUs. We use the Euler Ancestral sampler with 10 diffusion steps.

Fast Generation via Parallelization Note that editing frame It does not require other frames as input except the cached features Kanc and Vanc from anchor frames. Therefore, we can edit arbitrary long videos by splitting them into segments and leverage multi-GPUs to parallize the generation, while the computation remains numerically identical. As a result, our method achieves significant speedup compared to previous works. Moreover, it delivers supe-

For equivariant finetuning, we use the same dataset that was used to train the image editing model, and apply the data augmentation discussed in Section 5.2. We load the image editing model’s pretrained checkpoint, and resumed

|[Figure 55]<br><br>[Figure 56]|[Figure 57]<br><br>[ Input Video ]|[Figure 58]|
|---|---|---|
|[Figure 59]<br><br>[Figure 60]<br><br>Ma|[Figure 61]<br><br>ke it a wood sculptu|[Figure 62]<br><br>re|

|[Figure 63]|[Figure 64]<br><br>[Figure 65]<br><br>[Figure 66]<br><br>[ Input Video ]| |
|---|---|---|
|[Figure 67]<br><br>[Figure 68]|[Figure 69]<br><br>Turn into lion|[Figure 70]|

|[Figure 71]|[Figure 72]<br><br>[Figure 73]<br><br>[Figure 74]<br><br>[ Input Video ]| |
|---|---|---|
|[Figure 75]<br><br>Tu|[Figure 76]<br><br>[Figure 77]<br><br>[Figure 78]<br><br>rn into a vintage ca|r|

|[Figure 79]|[Figure 80]<br><br>[ Input Video ]|[Figure 81]<br><br>[Figure 82]|
|---|---|---|
| |[Figure 83]<br><br>[Figure 84]<br><br>Make it Tokyo|[Figure 85]<br><br>[Figure 86]|

Figure 5. Diverse Video Editing via Fairy. Fairy enables a wide range of video edits with different types of subjects.

In this paper, we conduct a large-scale user study on an evaluation set consists of 1000 video-instruction samples. The evaluation set is divided into two parts: first, to test a model’s robustness across different videos, we construct the evaluation set of 50 videos × 10 instructions. And to test a model’s robustness across different instructions, we construct a dual evaluation set of 10 videos × 50 instructions. The videos are accessible from ShutterStock [3]. To our best knowledge, this is the largest evaluation in the videoto-video generation literature so far.

training for 50,000 steps with a batch size of 128, costing 30 hours on 8 A100 GPUs with 80GB memories.

- 6.1. Qualitative Evaluation We first show qualitative results of Fairy. Since most of the PDF readers do not render videos properly, we only show a small number of frames for each video. We strongly recommend readers to checkout our supplementary materials to watch the complete videos. In Figure 5, we show that our model is capable conducting edits on different subjects. In Figure 6, we show that our model is able to conduct different types of editing, including stylization, character swap, local editing, and attribute editing, following textual instructions. In Figure 9, we show that our model can transform the source character into different target characters based on instructions. Note that our model can adapt to different input aspect ratios without need for re-training. Our input videos contain large motions, occlusions, and other complex dynamics. Despite those challenges, videos generated by our model are temporally consistent and visually appealing. We also show our model’s capabilities to generate long videos in the supplementary material.
- 6.2. Quantitative Evaluation Quantitatively evaluating video generative models is challenging. First, the generation task is intrinsically highvariance – there are countless ways to edit an video given the instruction. Second, previous works have adopted metrics such as CLIP scores [9, 10] to evaluate the generation quality. However, these metrics are not necessarily aligned with human perception [18]. Lastly, human evaluation is still the golden standard to judge the quality. Yet, due to the cost of human evaluation, previous works have only conducted small scale human evaluations (< 100 samples).

We conduct a A/B comparison to compare our method with three previous works, Rerender [33] (tracking and propagation), Tokenflow [2] (image model adaptation), and Gen-1 [9] (conditional video model), which are the strongest representative of the three paradigms for videoto-video generation today. Results from baselines are collected from [1]. Prompts for baselines are descriptive, e.g., ”a dog running on grass, in Van Gogh style”. We re-write the prompt for our method as an edit instruction, e.g., ”in Van Gogh Style”. Since Gen-1 is not open-sourced, the evaluation is done on a smaller evaluation set of 100 videos. In each evaluation tuple, we show the input video, the editing instruction or prompt, and the output videos generated by Fairy and a baseline. We ask human evaluators to choose the better video in terms of their single frame quality, temporal consistency, prompt faithfulness, input faithfulness, and overall quality. Each comparison is rated by 3 different annotators and the decision is determined by the majority vote. We report the overall quality comparison in Figure 7, which demonstrates that videos produced by Fairy are more preferable, with a win rate of 41% vs 36% against Rerender, 73% vs. 16% against TokenFlow, and 72% vs 26% against Gen-1. More details in the supplementary material.

|[Figure 87]|[Figure 88]|[Figure 89]|
|---|---|---|

|[Figure 90]|[Figure 91]|[Figure 92]|
|---|---|---|

|[Figure 93]|[Figure 94]|[Figure 95]|[Figure 96]|
|---|---|---|---|

+ Anchor + Equi-Finetune

Input (Prompt: Turn into lion)

(Tem-Con: 0.974)

|[Figure 97]|[Figure 98]<br><br>[Figure 99]<br><br>[Figure 100]<br><br>|[Figure 101]<br><br>[Figure 102]<br><br>[Figure 103]<br><br>|
|---|---|---|

|[Figure 104]|[Figure 105]<br><br>[Figure 106]<br><br>|[Figure 107]<br><br>[Figure 108]<br><br>[Figure 109]<br><br>|
|---|---|---|

|[Figure 110]<br><br>[Figure 111]<br><br>Tu|[Figure 112]<br><br>rn into a car|[Figure 113]<br><br>toon lion cu|[Figure 114]<br><br>b|
|---|---|---|---|

stylizationattributeeditcharacterswaplocaledit

+ Anchor − Equi-Finetune − Anchor − Equi-Finetune

(Tem-Con: 0.968) (Tem-Con: 0.959)

Figure 8. Ablation Study. Without equivariant fine-tuning and anchor-based attention, we observed inconsistencies, particularly in the changing patterns of body and earscostumes over time. This inconsistency is further exacerbated upon the removal of anchorbased attention, leading to lower temporal consistency score.

|[Figure 115]|[Figure 116]<br><br>In Mone|[Figure 117]<br><br>[Figure 118]<br><br>t style|[Figure 119]|
|---|---|---|---|

|[Figure 120]|[Figure 121]<br><br>Add sun|[Figure 122]<br><br>[Figure 123]<br><br>glasses|[Figure 124]|
|---|---|---|---|

lows the instruction, delivering high-quality, temporal consistent, and authentic generations.

Lastly, we compute metrics adopted by previous works, Tem-Con and Frame-Acc [21, 33]. Tem-Con assesses temporal consistency by calculating the cosine similarity of CLIP feature across successive frame pairs, and Frame-Acc measures the percentage of frames where the edited image exhibits greater CLIP similarity to the target prompt than to the source prompt. The results in Table 1 demonstrates that Fairy achieves the best temporal consistency and framewise editing accuracy against the baselines.

|[Figure 125]|[Figure 126]<br><br>Make it|[Figure 127]<br><br>[Figure 128]<br><br>black|[Figure 129]|
|---|---|---|---|

- Figure 6. Different type of editing. Fairy is able to handle a diverse set of instructions and perform appropriate editing.

(a) vs. Rerender (b) vs. TokenFlow (c) vs. Gen-1

10% 1%

16%

73%

Fairy (Ours) Baseline Both Good Both Bad

17% 6%

36%

41%

2% 26%

72%

- Figure 7. A/B Comparison with Baselines. Fairy significantly surpassed baseline models, demonstrating its effectivity.

Speed Comparison In Table 1, we also compare the latency of different models. In particular, we calculate the inference time of editing a 4-seconds, 30 FPS, 512p×384p video on a server with 8 A100 GPUs. The key-frame interval of Rerender is set to 4 instead of the default 10, since the test videos contain faster motion. This leads to improved quality for Rerender. All other parameters were default. Due to its architecture simplicity, Fairy is already significantly faster than baselines using 1 GPU. Using a single GPU, Fairy completes inference in just 78 seconds, achieving 9.5× faster than TokenFlow and 7.5× faster than Rerender. When utilizing all 8 GPUs on the node, Fairy is 53× faster than TokenFlow and 44× faster than Rerender.

Latency (sec) ↓ Frame-Acc ↑ Tem-Con ↑

|TokenFlow Rerender<br><br>|744 608|0.537 0.775<br><br>|0.973 0.972|
|---|---|---|---|
|Ours<br><br>|13.8|0.819<br><br>|0.974|

#### 6.3. Ablation Studies

We conduct an ablation study to verify the effectiveness of our model’s component. We gradually remove equivariant fine-tuning and anchor-based attention, ultimately leading to the adoption of a standard frame-by-frame editing approach. The results are shown in Figure 8. The model becomes sensitive to the camera motion without equivariant finetuning, rendering inconsistency in the details. The subsequent removal of anchor-based attention, transitioning to a frame-based model, introduces further inconsistencies in the generated video. We compute the Tem-Con metric

Table 1. We assess our method’s temporal consistency and fidelity to the target text prompt using CLIP similarity metrics.

Figure 10 shows the visual comparison with the baselines. We observe that both Tokenflow [10] and Rerender [33] do not adhere closely to the provided instructions, resulting in evident inconsistencies. Outputs from Gen-1 often over-modify the entire scene and do not retain the original content effectively. In contrast, Fairy meticulously fol-

|[Figure 130]|[Figure 131]<br><br>[Figure 132]<br><br>[ Input Video ]|[Figure 133]|
|---|---|---|

|[Figure 134]<br><br>Turn in|[Figure 135]<br><br>to a bronze s|[Figure 136]<br><br>[Figure 137]<br><br>tatue|
|---|---|---|

|[Figure 138]<br><br>Make it m|[Figure 139]<br><br>[Figure 140]<br><br>arble roman s|[Figure 141]<br><br>culpture|
|---|---|---|

|[Figure 142]<br><br>Turn into a|[Figure 143]<br><br>[Figure 144]<br><br>metal knight|[Figure 145]<br><br>sculpture|
|---|---|---|

|[Figure 146]<br><br>T|[Figure 147]<br><br>[Figure 148]<br><br>urn into a yet|[Figure 149]<br><br>i|
|---|---|---|

|[Figure 150]<br><br>[Figure 151]<br><br>Tu|[Figure 152]<br><br>rn into a robo|[Figure 153]<br><br>t|
|---|---|---|

- Figure 9. Diverse Character Swap: Fairy possesses the capability to interchange the individual with a diverse array of characters.

|[Figure 154]|[Figure 155]|
|---|---|
|[Figure 156]|[Figure 157]|

|[Figure 158]|[Figure 159]|[Figure 160]|[Figure 161]|
|---|---|---|---|

|[Figure 162]|[Figure 163]|[Figure 164]|[Figure 165]|
|---|---|---|---|

|[Figure 166]|[Figure 167]|[Figure 168]|[Figure 169]|
|---|---|---|---|

|[Figure 170]|[Figure 171]|[Figure 172]|[Figure 173]|
|---|---|---|---|

TokenFlowRenderer

Fairy(ours)Gen-1

RendererTokenFlow

Fairy(ours)Gen-1

In Van Gogh style

Turn into a wood sculpture

|[Figure 174]|[Figure 175]|[Figure 176]|[Figure 177]|
|---|---|---|---|

|[Figure 178]|[Figure 179]|[Figure 180]|[Figure 181]|
|---|---|---|---|

|[Figure 182]|[Figure 183]|[Figure 184]|[Figure 185]|
|---|---|---|---|

|[Figure 186]|[Figure 187]|[Figure 188]|[Figure 189]|
|---|---|---|---|

|[Figure 190]|[Figure 191]|
|---|---|
|[Figure 192]|[Figure 193]|

- Figure 10. Comparison with Baselines. Fairy consistently outperform baselines in terms of consistency and instruction-faithfulness.

based on 150 videos and report in Figure 8. It confirms our observation that the proposed methodology effectively improves the temporal consistency, lifting the Tem-Con from 0.959 (baseline) to 0.968 (w/ anchors) to 0.974 (w/ anchor and equivariant finetuning).

#### 6.4. Limitations

The efficacy of Fairy is intrinsically tied to the underlying image-editing model. This means that any inherent constraints of this underlying model, e.g., face and text distortion, etc., will naturally manifest in the video editing capacities of Fairy . In our observations, a notable side effect of equivariant finetuning is the diminished ability to accurately render dynamic visual effects, such as lightning or flames. The process seems to overly focus on maintaining temporal

consistency, which often results in the depiction of lightning as static or stagnate, rather than dynamic and fluid. See the supplementary material for visualization.

### 7. Conclusion

Fairy offers a transformative approach to video editing, building on the strengths of image-editing diffusion models. By leveraging anchor-based cross-frame attention and equivariant finetuning, Fairy guarantees temporal consistency and superior video synthesis. Moreover, it tackles the memory and processing speed constraints observed in preceding models. With the capability to produce highresolution videos at a blazing speed, Fairy firmly establishes its superiority in terms of quality and efficiency, as further corroborated by our extensive user study.

### References

- [1] Exploring video-to-video synthesis: A comparative analysis of rerender, tokenflow, and gen-1. https://medium.com/@lwen9595/exploringvideo-to-video-synthesis-a-comparativeanalysis-of-rerender-tokenflow-and-gen1-9a63f281c4e1. 6
- [2] pytorch documentation. https://pytorch.org/ vision/main/transforms.html. Accessed: 202311-02. 5, 6
- [3] Stock footage video, royalty-free hd, 4k video clips, 2023. 6
- [4] Omer Bar-Tal, Dolev Ofri-Amar, Rafail Fridman, Yoni Kasten, and Tali Dekel. Text2live: Text-driven layered image and video editing. In European conference on computer vision, pages 707–723. Springer, 2022. 2, 3
- [5] Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18392–18402, 2023. 3, 5
- [6] Wenhao Chai, Xun Guo, Gaoang Wang, and Yan Lu. Stablevideo: Text-driven consistency-aware diffusion video editing. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 23040–23050, 2023. 2, 3
- [7] Xiaoliang Dai, Ji Hou, Chih-Yao Ma, Sam Tsai, Jialiang Wang, Rui Wang, Peizhao Zhang, Simon Vandenhende, Xiaofang Wang, Abhimanyu Dubey, et al. Emu: Enhancing image generation models using photogenic needles in a haystack. arXiv preprint arXiv:2309.15807, 2023. 2
- [8] Carl Doersch, Ankush Gupta, Larisa Markeeva, Adri`a Recasens, Lucas Smaira, Yusuf Aytar, Jo˜ao Carreira, Andrew Zisserman, and Yi Yang. Tap-vid: A benchmark for tracking any point in a video. Advances in Neural Information Processing Systems, 35:13610–13626, 2022. 4
- [9] Patrick Esser, Johnathan Chiu, Parmida Atighehchian, Jonathan Granskog, and Anastasis Germanidis. Structure and content-guided video synthesis with diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7346–7356, 2023. 2, 6
- [10] Michal Geyer, Omer Bar-Tal, Shai Bagon, and Tali Dekel. Tokenflow: Consistent diffusion features for consistent video editing. arXiv preprint arXiv:2307.10373, 2023. 3, 6, 7
- [11] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Prompt-to-prompt image editing with cross attention control. arXiv preprint arXiv:2208.01626, 2022. 3
- [12] Ondˇrej Jamriˇska, S´ˇarka Sochorov´a, Ondˇrej Texler, Michal Luk´aˇc, Jakub Fiˇser, Jingwan Lu, Eli Shechtman, and Daniel S`ykora. Stylizing video by example. ACM Transactions on Graphics (TOG), 38(4):1–11, 2019. 2, 3
- [13] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. Advances in Neural Information Processing Systems, 35:26565–26577, 2022. 4
- [14] Yoni Kasten, Dolev Ofri, Oliver Wang, and Tali Dekel. Layered neural atlases for consistent video editing. ACM Transactions on Graphics (TOG), 40(6):1–12, 2021. 2, 3

- [15] Levon Khachatryan, Andranik Movsisyan, Vahram Tadevosyan, Roberto Henschel, Zhangyang Wang, Shant Navasardyan, and Humphrey Shi. Text2video-zero: Text-toimage diffusion models are zero-shot video generators. arXiv preprint arXiv:2303.13439, 2023. 3
- [16] Yao-Chih Lee, Ji-Ze Genevieve Jang, Yi-Ting Chen, Elizabeth Qiu, and Jia-Bin Huang. Shape-aware text-driven layered video editing. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14317–14326, 2023. 2, 3
- [17] Shaoteng Liu, Yuechen Zhang, Wenbo Li, Zhe Lin, and Jiaya Jia. Video-p2p: Video editing with cross-attention control. arXiv preprint arXiv:2303.04761, 2023. 3
- [18] Yaofang Liu, Xiaodong Cun, Xuebo Liu, Xintao Wang, Yong Zhang, Haoxin Chen, Yang Liu, Tieyong Zeng, Raymond Chan, and Ying Shan. Evalcrafter: Benchmarking and evaluating large video generation models. arXiv preprint arXiv:2310.11440, 2023. 6
- [19] Hao Ouyang, Qiuyu Wang, Yuxi Xiao, Qingyan Bai, Juntao Zhang, Kecheng Zheng, Xiaowei Zhou, Qifeng Chen, and Yujun Shen. Codef: Content deformation fields for temporally consistent video processing. arXiv preprint arXiv:2308.07926, 2023. 2, 3
- [20] Jordi Pont-Tuset, Federico Perazzi, Sergi Caelles, Pablo Arbel´aez, Alex Sorkine-Hornung, and Luc Van Gool. The 2017 davis challenge on video object segmentation. arXiv preprint arXiv:1704.00675, 2017. 4
- [21] Chenyang Qi, Xiaodong Cun, Yong Zhang, Chenyang Lei, Xintao Wang, Ying Shan, and Qifeng Chen. Fatezero: Fusing attentions for zero-shot text-based video editing. arXiv:2303.09535, 2023. 3, 7
- [22] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 1

(2):3, 2022. 2

- [23] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 2
- [24] Alexander S. Disco diffusion v5.2 - warp fusion. https: //github.com/Sxela/DiscoDiffusion-Warp. 2, 3
- [25] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in Neural Information Processing Systems, 35:36479–36494, 2022. 2
- [26] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. Advances in Neural Information Processing Systems, 35:25278–25294, 2022. 2
- [27] Luming Tang, Menglin Jia, Qianqian Wang, Cheng Perng Phoo, and Bharath Hariharan. Emergent correspondence

from image diffusion. arXiv preprint arXiv:2306.03881,

2023. 3

- [28] Narek Tumanyan, Michal Geyer, Shai Bagon, and Tali Dekel. Plug-and-play diffusion features for text-driven image-to-image translation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1921–1930, 2023. 3
- [29] Wen Wang, kangyang Xie, Zide Liu, Hao Chen, Yue Cao, Xinlong Wang, and Chunhua Shen. Zero-shot video editing using off-the-shelf image diffusion models. arXiv preprint arXiv:2303.17599, 2023. 3
- [30] Wen Wang, Kangyang Xie, Zide Liu, Hao Chen, Yue Cao, Xinlong Wang, and Chunhua Shen. Zero-shot video editing using off-the-shelf image diffusion models. arXiv preprint arXiv:2303.17599, 2023. 2, 3
- [31] Xiang Wang, Hangjie Yuan, Shiwei Zhang, Dayou Chen, Jiuniu Wang, Yingya Zhang, Yujun Shen, Deli Zhao, and Jingren Zhou. Videocomposer: Compositional video synthesis with motion controllability. arXiv preprint arXiv:2306.02018, 2023. 2
- [32] Jay Zhangjie Wu, Yixiao Ge, Xintao Wang, Stan Weixian Lei, Yuchao Gu, Yufei Shi, Wynne Hsu, Ying Shan, Xiaohu Qie, and Mike Zheng Shou. Tune-a-video: One-shot tuning of image diffusion models for text-to-video generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7623–7633, 2023. 3
- [33] Shuai Yang, Yifan Zhou, Ziwei Liu, and Chen Change Loy. Rerender a video: Zero-shot text-guided video-to-video translation. arXiv preprint arXiv:2306.07954, 2023. 2, 3, 6, 7

### A. Webpage Demo

Fairy (Ours) Baseline Both Good Both Bad

The videos in the main paper and appendix can be viewed with our demo webpage by opening the webpage/ index.html in the supplementary material using a web browser.

7%

11%

7%

37%

28%

### B. Additional Results from Human Evaluation

82%

28%

As mentioned in Section 6.2, we conduct A/B comparison between Fairy with baselines. We ask annotators to compare our generated video with a baseline method’s result, and decide which one is better.Each video pair is evaluated by three independent annotators, and the majority vote is considered as the final rating. We ask raters to evaluate by four attributes: frame quality – visual quality of single frame; temporal consistency – whether the frames are coherent or flickering; prompt faithfulness – whether the output followed the editing instruction or target prompt; input faithfulness – whether the output video followed the contents of the original video. We reported the overall rating in Section 6.2, Figure 10. Here, we report a more detailed comparison along each attributes in Figure 11, 12, 13. Compared with Rerender (Figure 11), Fairy loses in terms of single frame visual quality. This is mainly due to the limitation of the foundational image editing model, while Rerender utilizes LoRA to enhance frame quality. Yet, Fairy significantly outperforms Rerender in terms of temporal consistency, achieves better prompt faithfulness, and performs similarly in terms of input faithfulness. Compared with TokenFlow, Fairy outperforms significantly in terms of frame quality, temporal consistency, and prompt faithfulness. They performs similarly in terms of input faithfulness. Compared with Gen-1, Fairy significantly outperforms in all attributes.

Prompt faithfulness

Input faithfulness

4% 25%

9% 21%

20%

55%

15%

51%

Temporal consistency

Frame visual quality

###### Figure 11. Comparison with Rerender.

Fairy (Ours) Baseline Both Good Both Bad

5% 2%

6% 14%

13%

67%

93%

In addition to the A/B comparison, in which we ask human annotators to compare our method with a baseline, we also conduct a standalone evaluation to examine output video’s quality. Each time we show an annotator the original video, an editing instruction, and the result video. We then ask the annotator to rate the output as good or bad by the same four attributes. We ask 3 annotators to rate each video, and the decision is determined by their majority vote. We report the success rate by each attributes in Figure 14.

Prompt faithfulness

Input faithfulness

5% 11%

6% 24%

55%

24% 60%

14%

### C. More Results

Temporal consistency

Frame visual quality

#### C.1. Character Swap

In Figure 15, we demonstrate more results of character swap, where Fairy is able to interchange individuals with various characters. Note that our model can adapt to different input aspect ratios without need for re-training.

###### Figure 12. Comparison with TokenFlow.

is less than 71.89 seconds via 6 A100 GPUs. In particular, the Fairy manage to retain decent temporal consistency even number of frames (664 frames) is way more than the number of anchor frames (3 frames).

Fairy (Ours) Baseline Both Good Both Bad

26%

43%

#### C.4. Ablation Study

56%

59%

- Figure 18 shows more ablation results by removing equivariant finetuning and anchor-based attention. We can see that without equivariant finetuning, the model is sensitive to local motion and movement of the subject and therefore degenerate the frame quality and temporal consistency. For instance, in the first video, the tail of the cat becomes the head of the lion in some of the frames, and the face of the cat in second video vary significantly between frames. Without anchor-based attention, the edit of each frame is completely independent, rendering in significant worse temporal consistency.
- Figure 19 demonstrates results generated with different

16%

1%

Prompt faithfulness

Input faithfulness

2% 9%

3% 15%

13%

30%

59%

69%

number of anchor frames. When number of anchor frames equals to 1, the global features model can leverage are too restricted, which lead to suboptimal edits. In contrast, we observe that when the number of anchor frames is greater than 7, the quality also gradually degrades, losing some visual details.

Temporal consistency

Frame visual quality

###### Figure 13. Comparison with Gen-1.

In Figure 20, we perform ablation study on the number of diffusion steps during generation. The model perform reasonably well when the number of diffusion step is above 10. We therefore set the diffusion step to 10 for all of our experiments to optimize the latency.

Success Rate of Fairy by Attributes

1.0

| | | |0.8|9|
|---|---|---|---|---|
|0.5|9<br><br>0.6|2<br><br>0.7|8| |
| | | | | |
| | | | | |
| | | | | |

0.8

#### C.5. Limitations

0.6

SuccessRate

Finally, Figure 21 demonstrates some limitations we point out in section 6.4. Since the model is never trained on video data, it does learn to generate concepts containing motion such as raining, lightning, or flames. Fairy also inherits the limit of the image editing model, where it is not able to follow the instructions that involve camera motion, such as zoom in or zoom out.

0.4

0.2

0.0

Frame quality Temporal consistency Prompt faithfulness Input faithfulness Evaluation Attributes

Figure 14. Standalone success rate by attributes. We report Fairy’s success rate in terms of frame quality, temporal consistency, prompt faithfulness, and input faithfulness.

- C.2. Stylization

- Figure 16 demonstrates more stylization results of Fairy. In particular, our model is able to recognize various styles, while perform high quality and temporal consistent edit based on the stylization instructions.

C.3. Arbitrary Long Videos

Fairy is able to scale to arbitrary long video without memory issue due to the proposed anchor-based attention. In

- Figure 17, we show that our model is able to generate a 27 second long video with high quality, while the latency

|[Figure 194]|[Figure 195]<br><br>[Figure 196]<br><br>[ Input Video ]|[Figure 197]|
|---|---|---|

|[Figure 198]|[Figure 199]<br><br>[Figure 200]<br><br>[ Input Video ]|[Figure 201]|
|---|---|---|

|[Figure 202]|[Figure 203]<br><br>[Figure 204]<br><br>[ Input Video ]|[Figure 205]|
|---|---|---|

|[Figure 206]<br><br>[Figure 207]<br><br>Turn|[Figure 208]<br><br>into a baby lion|[Figure 209]<br><br>cub|
|---|---|---|

|[Figure 210]<br><br>[Figure 211]<br><br>Turn into a|[Figure 212]<br><br>marble Roman|[Figure 213]<br><br>sculpture|
|---|---|---|

|[Figure 214]<br><br>[Figure 215]<br><br>Turn|[Figure 216]<br><br>into a bronze st|[Figure 217]<br><br>atue|
|---|---|---|

|[Figure 218]|[Figure 219]<br><br>[ Input Video ]|[Figure 220]<br><br>[Figure 221]|
|---|---|---|
|[Figure 222]<br><br>Tur|[Figure 223]<br><br>n into a wood sculpture|[Figure 224]<br><br>[Figure 225]|

| |[Figure 226]<br><br>[Figure 227]<br><br>[ Input Video ]|[Figure 228]<br><br>[Figure 229]|
|---|---|---|
|[Figure 230]|[Figure 231]<br><br>[Figure 232]<br><br>Turn into a fox|[Figure 233]|

Figure 15. Additional Results on Character Swap: Fairy is able to interchange the characters for videos with arbitrary ratios.

| |[Figure 234]<br><br>[Figure 235]<br><br>[Figure 236]<br><br>[Figure 237]<br><br>[ Input Video ]| |
|---|---|---|

|[Figure 238]<br><br>[Figure 239]|[Figure 240]<br><br>[Figure 241]<br><br>[ Input Video ]| |
|---|---|---|

|[Figure 242]|[Figure 243]<br><br>[Figure 244]<br><br>In Van Gogh style|[Figure 245]|
|---|---|---|

|[Figure 246]|[Figure 247]<br><br>[Figure 248]<br><br>In low poly art style|[Figure 249]|
|---|---|---|

| |[Figure 250]<br><br>[Figure 251]<br><br>[ Input Video ]|[Figure 252]<br><br>[Figure 253]|
|---|---|---|
|[Figure 254]|[Figure 255]<br><br>[Figure 256]<br><br>In Monet style|[Figure 257]|

|[Figure 258]<br><br>[Figure 259]|[Figure 260]<br><br>[Figure 261]<br><br>[ Input Video ]| |
|---|---|---|
|[Figure 262]|[Figure 263]<br><br>[Figure 264]<br><br>In cubism style|[Figure 265]|

Figure 16. Additional Results on Stylization: Fairy enables a wide range of style editing.

|[Figure 266]<br><br>[Figure 267]| |[Figure 268]<br><br>[Figure 269]<br><br>[ Input|Video ]|[Figure 270]<br><br>[Figure 271]<br><br>[Figure 272]| |
|---|---|---|---|---|---|

27sec(664frames)

|[Figure 273]|[Figure 274]|[Figure 275]<br><br>In low poly|[Figure 276]<br><br>[Figure 277]<br><br>art style|[Figure 278]|[Figure 279]|
|---|---|---|---|---|---|

|[Figure 280]<br><br>[Figure 281]| |[Figure 282]<br><br>[Figure 283]<br><br>[Figure 284]<br><br>[ Input|Video ]|[Figure 285]<br><br>[Figure 286]| |
|---|---|---|---|---|---|

26sec(633frames)

|[Figure 287]|[Figure 288]|[Figure 289]<br><br>[Figure 290]<br><br>Turn into|[Figure 291]<br><br>a knight|[Figure 292]|[Figure 293]|
|---|---|---|---|---|---|

|[Figure 294]| |[Figure 295]<br><br>[Figure 296]<br><br>[Figure 297]<br><br>[Figure 298]<br><br>[ Input|Video ]|[Figure 299]<br><br>[Figure 300]| |
|---|---|---|---|---|---|
|[Figure 301]|[Figure 302]|[Figure 303]<br><br>In Van Go|[Figure 304]<br><br>[Figure 305]<br><br>gh style|[Figure 306]|[Figure 307]|

25sec(610frames)

Figure 17. Any-length Video Editing. Fairy is able to scale to arbitrary long video without memory issue.

|[Figure 308]<br><br>[Figure 309]<br><br>Turn in|[Figure 310]<br><br>to a baby lion|[Figure 311]<br><br>cub|
|---|---|---|

|[Figure 312]<br><br>In|[Figure 313]<br><br>Van Gogh sty|[Figure 314]<br><br>[Figure 315]<br><br>le|
|---|---|---|

|[Figure 316]<br><br>[Figure 317]<br><br>T|[Figure 318]<br><br>urn into a fox|[Figure 319]|
|---|---|---|

|[Figure 320]|[Figure 321]|[Figure 322]|
|---|---|---|

|[Figure 323]|[Figure 324]|[Figure 325]|
|---|---|---|

|[Figure 326]|[Figure 327]|[Figure 328]|
|---|---|---|

+Equi-Finetune

+Anchor

|[Figure 329]<br><br>|[Figure 330]<br><br>|[Figure 331]<br><br>|
|---|---|---|

|[Figure 332]<br><br>|[Figure 333]<br><br>|[Figure 334]<br><br>|
|---|---|---|

|[Figure 335]<br><br>|[Figure 336]<br><br>|[Figure 337]<br><br>|
|---|---|---|

−Equi-Finetune −Anchor

+Anchor

|[Figure 338]|[Figure 339]|[Figure 340]|
|---|---|---|

|[Figure 341]<br><br>|[Figure 342]<br><br>|[Figure 343]<br><br>|
|---|---|---|

|[Figure 344]<br><br>|[Figure 345]<br><br>|[Figure 346]|
|---|---|---|

−Equi-Finetune

- Figure 18. Additional Results on Ablation Study. We demonstrate that both equivariant finetuning and anchor-based attention are crucial to Fairy .

|[Figure 347]|[Figure 348]<br><br>In cubis|[Figure 349]<br><br>[Figure 350]<br><br>m style|[Figure 351]|
|---|---|---|---|

|[Figure 352]<br><br>Turn in|[Figure 353]<br><br>[Figure 354]<br><br>to a Roman|[Figure 355]<br><br>marble scul|[Figure 356]<br><br>pture|
|---|---|---|---|

|[Figure 357]|[Figure 358]|[Figure 359]|[Figure 360]|
|---|---|---|---|

|[Figure 361]|[Figure 362]|[Figure 363]|[Figure 364]|
|---|---|---|---|

#Anchor=1#Anchor=3#Anchor=5#Anchor=7

|[Figure 365]|[Figure 366]|[Figure 367]|[Figure 368]|
|---|---|---|---|

|[Figure 369]|[Figure 370]|[Figure 371]|[Figure 372]|
|---|---|---|---|

|[Figure 373]|[Figure 374]|[Figure 375]|[Figure 376]|
|---|---|---|---|

|[Figure 377]|[Figure 378]|[Figure 379]|[Figure 380]|
|---|---|---|---|

|[Figure 381]|[Figure 382]|[Figure 383]|[Figure 384]|
|---|---|---|---|

|[Figure 385]|[Figure 386]|[Figure 387]|[Figure 388]|
|---|---|---|---|

- Figure 19. Ablation study on number of anchor frames. We found that setting number of anchor frames to 3 yields the best results.

|[Figure 389]<br><br>T|[Figure 390]<br><br>urn into a b|[Figure 391]<br><br>[Figure 392]<br><br>ronze statue|[Figure 393]|
|---|---|---|---|

|[Figure 394]|[Figure 395]<br><br>[Figure 396]<br><br>In Van Go|[Figure 397]<br><br>gh style|[Figure 398]|
|---|---|---|---|

|[Figure 399]|[Figure 400]|[Figure 401]|[Figure 402]|
|---|---|---|---|

|[Figure 403]|[Figure 404]|[Figure 405]|[Figure 406]|
|---|---|---|---|

#Step=5#Step=10#Step=20#Step=50

|[Figure 407]|[Figure 408]|[Figure 409]|[Figure 410]|
|---|---|---|---|

|[Figure 411]|[Figure 412]|[Figure 413]|[Figure 414]|
|---|---|---|---|

|[Figure 415]|[Figure 416]|[Figure 417]|[Figure 418]|
|---|---|---|---|

|[Figure 419]|[Figure 420]|[Figure 421]|[Figure 422]|
|---|---|---|---|

|[Figure 423]|[Figure 424]|[Figure 425]|[Figure 426]|
|---|---|---|---|

|[Figure 427]|[Figure 428]|[Figure 429]|[Figure 430]|
|---|---|---|---|

- Figure 20. Ablation study on number of diffusion steps. We found that diffusion steps above 5 generally yield good results.

|[Figure 431]|[Figure 432]<br><br>[Figure 433]<br><br>[ Input|[Figure 434]<br><br>Video ]|[Figure 435]|
|---|---|---|---|
|[Figure 436]|[Figure 437]<br><br>Add ligh|[Figure 438]<br><br>[Figure 439]<br><br>tning|[Figure 440]|

|[Figure 441]|[Figure 442]<br><br>[Figure 443]<br><br>[ Input|[Figure 444]<br><br>Video ]|[Figure 445]|
|---|---|---|---|
|[Figure 446]|[Figure 447]<br><br>Add fl|[Figure 448]<br><br>[Figure 449]<br><br>ames|[Figure 450]|

|[Figure 451]|[Figure 452]<br><br>[Figure 453]<br><br>[ Input|[Figure 454]<br><br>Video ]|[Figure 455]|
|---|---|---|---|

|[Figure 456]|[Figure 457]<br><br>[ Input|[Figure 458]<br><br>[Figure 459]<br><br>Video ]|[Figure 460]|
|---|---|---|---|

|[Figure 461]|[Figure 462]<br><br>[Figure 463]<br><br>Make it|[Figure 464]<br><br>rain|[Figure 465]|
|---|---|---|---|

|[Figure 466]|[Figure 467]<br><br>Zoom|[Figure 468]<br><br>[Figure 469]<br><br>out|[Figure 470]|
|---|---|---|---|

###### Figure 21. Limitations of Fairy. Our model cannot accurately render dynamic visual effects, such as lightning, flames, or rain.

