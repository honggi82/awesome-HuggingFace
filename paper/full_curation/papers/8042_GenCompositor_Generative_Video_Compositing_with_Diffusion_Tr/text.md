# arXiv:2509.02460v2[cs.CV]19Mar2026

## GENCOMPOSITOR: GENERATIVE VIDEO COMPOSITING WITH DIFFUSION TRANSFORMER

###### Shuzhou Yang1 Xiaoyu Li2∗ Xiaodong Cun4 Guangzhi Wang2 Lingen Li5 Ying Shan2 Jian Zhang1,3†

1School of Electronic and Computer Engineering, Peking University 2ARC Lab, Tencent PCG

- 3Guangdong Provincial Key Laboratory of Ultra High Definition Immersive Media Technology
- 4GVC Lab, Great Bay University 5The Chinese University of Hong Kong

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

bg.videoresultsbg.videoresultsfg.videofg.video

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

(a) (b)

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

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

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

3，19，31，48 0，28，42，48

(c) (d)

Figure 1: GenCompositor is capable of effortlessly compositing different videos following userspecified trajectories and scales. It could preserve the background video content and also seamlessly integrate the dynamic foreground elements into the background video, which not only strictly follows user-given instructions but also physically coordinates with background environments.

ABSTRACT

Video compositing combines live-action footage to create video production, serving as a crucial technique in video creation and film production. Traditional pipelines require intensive labor efforts and expert collaboration, resulting in lengthy production cycles and high manpower costs. To address this issue, we automate this process with generative models, called generative video compositing. This new task strives to adaptively inject identity and motion information of foreground video to the target video in an interactive manner, allowing users to customize the size, motion trajectory, and other attributes of the dynamic elements added in final video. Specifically, we designed a novel Diffusion Transformer (DiT) pipeline based on its intrinsic properties. To maintain consistency of the target video before and after editing, we revised a light-weight DiT-based background preservation branch with masked token injection. As to inherit dynamic elements from other sources, a DiT fusion block is proposed using full self-attention, along with a simple yet effective foreground augmentation for training. Besides, for fusing background and foreground videos with different layouts based on user control, we developed a novel position embedding, named Extended Rotary Position Embedding (ERoPE). Finally, we curated a dataset comprising 61K sets of videos for our new task, called VideoComp. This data includes complete dynamic elements and high-quality target videos. Experiments demonstrate that our method effectively realizes generative video compositing, outperforming existing possible solutions in fidelity and consistency. Project is available at https://gencompositor.github.io/

∗Project leader. †Corresponding author.

- 1 INTRODUCTION

Video compositing aims to edit target background video by adding foreground video from other sources, creating visually pleasing video assets. Compared to text- or image-guided video editing, this process edits video productions based on real-world captured videos, acting as a bridge between raw live-action video footage and final video work. However, classical process is labor intensive, which requires collaborative efforts of animators, videographers, and special effects artists. In this paper, we present generative video compositing, a new video editing task that attempts to automate compositing process with generative models (Ho et al., 2022; Blattmann et al., 2023; Li et al., 2025a). It allows direct editing of background videos using video footage under user control, aligning with traditional video creation process in video compositing stage.

To composite visually appealing video results, three main challenges are explored in this paper: 1) Ensuring background consistency of the video before and after editing. 2) Preserving the identity and motion of the injected dynamic elements while harmonizing with background content. 3) Enabling flexible user control, such as controllable size and motion trajectories. However, existing solutions cannot resolve these issues well. Controllable video generation (Hu, 2024; Li et al., 2026; Hu et al., 2025; Wang et al., 2025b) produces videos based on external conditions, including user-defined trajectories, images, and texts. Here, images provide identity information, and texts describe the motion to be generated. But images and text cannot precisely control the added elements at the pixel level, and current methods do not support external video condition. In Contrast, video harmonization methods (Lu et al., 2022; Chen et al., 2024; Xiao et al., 2024) directly paste foreground element onto the target video frame-by-frame, and adjust its RGB values to blend with background. However, these cannot adaptively specify the position, size, motion trajectory, and other attributes of the added element. Moreover, pasting foreground video requires accurate corresponding segmentation masks, which may not always be feasible in practical utilization.

To this end, we propose the first generative video compositing method, GenCompositor, aiming to automatically composite background video with dynamic foreground elements. As shown in Fig. 1, given a foreground video, a background video, and a user-provided trajectory in background (depicted by red line), GenCompositor injects foreground element to background video faithfully following trajectory. Trajectory can be specified by dragging a line or only clicking a point. For the latter, we track the movement of point with optical flow to simulate drag line. Our model can even predict realistic interaction between added element and background. For example, in Fig. 1 (a), the explosion effect influences the background content, i.e., the car’s fuel tank disappears after explosion in the last frame. In Fig. 1 (b), GenCompositor predicts realistic shadow of the added butterfly across frames, whose direction and effect consistent with background lighting.

We realize this by curating the first high-quality dataset for training, and proposing a novel Diffusion Transformer (DiT) pipeline specifically tailored for generative video compositing. Our pipeline consists of three main components. Firstly, a lightweight DiT-based background preservation branch is designed to ensure the consistency of the background in the edited results with the input videos. Secondly, to inject dynamic foreground elements, we propose a DiT fusion block with full selfattention to fuse tokens from foreground elements with that of background videos, and conduct detailed ablation study to demonstrate the advantages of this design over current cross-attention injection approaches (Ye et al., 2023; Cao et al., 2023; Wang et al., 2024a;c; Yang et al., 2025a). Considering that layouts of added foreground elements should follow user control and often differ from those of background videos, directly applying RoPE to foreground tokens introduces leakage artifacts. Therefore, we propose a novel position embedding method, Extended Rotary Position Embedding (ERoPE), to adaptively adjust the positions and scales of foreground tokens, which enables high-quality user-specified conditional generation even under layout-unaligned conditions. This technique is theoretically applicable to other tasks that exploit layout-unaligned video conditions.

Moreover, we develop some practical operations to enhance generalization and robustness. On the one hand, luminance augmentation is applied to foreground video for training, which strengthens the model’s generalization to diverse foreground conditions. On the other hand, we propose mask inflation to feed inaccurate masks to the model, enabling realistic interaction between newly added objects and environment. Experiments show that GenCompositor enables high-quality video compositing and can be used to make video effects. Overall, our contributions are as follows:

- • We propose a new practical video editing task, generative video compositing, and the first feasible solution that can automatically inject dynamic footage into the target video in a generative manner, utilizing the proposed novel diffusion architecture.
- • We design some novel techniques targeted to the unique characteristics of video compositing, including a revised position embedding, a full self-attention DiT fusion block, and a lightweight background preservation branch. These provide references for future research.
- • To train this model, we elaborate a data curation pipeline and develop some practical training operations, such as mask inflation and luminance augmentation, to improve the generalization ability and robustness of algorithm.
- • Experiments prove that the proposed method outperforms existing potential solutions, enables effortless generative video compositing based on user-given instructions, and can be used for automatic video effects creation. This inspires future research in this area.

- 2 RELATED WORK

- 2.1 DIFFUSION-BASED VIDEO EDITING In the field of AI-generated content, video editing (Brooks et al., 2023; Esser et al., 2023; Shin

- et al., 2024) has made great progress with the success of diffusion models (Yang et al., 2025d; Li et al., 2025b; Kong et al., 2024; Wang et al., 2025a; Ju et al., 2025; Li et al., 2025c). Early works attempted to use priors from pre-trained models (Ceylan et al., 2023; Yang et al., 2024; 2025b;c; Ma
- et al., 2025). Such as finetuning T2I models (Wu et al., 2023) or designing attention mechanisms tailored to video (Cai et al., 2025; QI et al., 2023; Wang et al., 2023; Yatim et al., 2025; Guo et al., 2024a). Recently, some methods trained specialized video editing models to realize more effective and prolific effect. Mou et al. (2024a) trained two ControlNet (Zhang et al., 2023) branches to specify motion and inject ID, respectively. Tu et al. (2025) followed a similar strategy, but enabled more fine-grained control through denser key points and trajectories. Liu et al. (2025) trained two control branches to guide spatial and temporal editing respectively. Bian et al. (2025) trained a video inpainting model, and edited videos through masking the target area, using Flux (Blackforestlabs,

2024) to edit the first frame, and painted subsequent frames to propagate the editing effect.

In general, training a dedicated model achieves better performance than training-free methods. However, existing approaches mainly edit videos based on images or textual prompts, making it hard to precisely control the appearance and motion details of the editing effects. To this end, we first propose generative video compositing task, which directly edits videos based on dynamic footage from other sources, realizing more practical and fine-grained video editing.

- 2.2 VIDEO HARMONIZATION

Video harmonization is a classical low-level vision task (Li et al., 2024; Dong et al., 2024; Tan et al., 2025), which aims to adjust the lighting of added foreground element in the composited video to make it visually harmonious. Similar to other low-level methods (Liu et al., 2022; Yang et al., 2023), Huang et al. (2020) first introduced adversarial training to video harmonization. Lu et al. (2022) collected a dedicated dataset for training. Ke et al. (2022) attempted to work in a white-box manner, which regresses the image-level filter argument to predict harmonized results. Guo et al. (2024b) directly trained a Triplet Transformer to simultaneously resolve multiple low-level video tasks, including harmonization. However, all of these methods focus only on adjusting the color of added elements. They all operate on composited videos and require a foreground video and its accurate and pixel-aligned mask as input. Besides, these methods do not allow users to freely modify the size and motion trajectory of the added elements in the final video. To address these limitations, we introduce generative video compositing task that accommodates arbitrary mask inputs. Using foreground videos, source background videos, and user-defined trajectories and scales, our model autonomously generates composited videos that adhere to all conditions.

- 3 TASK DEFINITION

Here, we define the task of generative video compositing. The inputs to this task are: a background video vb to be edited, a foreground video vf to be injected, and a user-given control c. The objective is to composite vb and vf following c, resulting in the final output z0.

Compared with other video editing methods, there are some unique characteristics of this problem. First, the layout of the videos to be edited vb is highly consistent with editing results z0. Meanwhile, layouts of condition vf are not pixel-aligned with vb. We should consider these two different conditions to generate results z0. Latent Diffusion Model (LDM) (Rombach et al., 2022) is employed to realize this, starting from a random latent noise zT, we aim to denoise zT to z0 based on the three inputs mentioned above. For training, given the Ground Truth video z0 containing desired foreground element. We add noise ϵ of various scales to it following a predefined schedule defined as:

zt = √α¯tz0 + √1 − α¯tϵ. (1) Training process aims to predict the added noise ϵ in zt by network ϵθ(·), objective function is:

min

θ

Ez

0,ϵ∼N(0,I),t∥ϵ − ϵθ(zt|vb,vf,c,t)∥22, (2)

where t is sampling step, vb, vf, and c are conditions. After training, ϵθ(·) receives a random noise zT, denoises it step by step, and decodes final latent through a VAE decoder for composited results.

- 4 METHOD

We introduce input conversion in Sect. 4.1, showing how we convert and augment user-given instructions. Then, we explain the proposed pipeline, including background preservation branch (Sect. 4.2), DiT fusion block (Sect. 4.3), and Extended Rotary Position Embedding (ERoPE) (Sect. 4.4).

- 4.1 INPUT CONVERSION

For utilization, our inputs include a background video vb, a foreground footage vf, as well as the user-given scale factor s and trajectory curve. We denote the user control as u = {γ,s}, where

γ is a 2D trajectory on the first frame of vb. We rasterize u into a per-frame binary mask video M = {Mt}Tt=1, Mt ∈ {0,1}H×W, and a masked video X = {Xt}Tt=1 defined by

Xt = (1 − Mt) ⊙ vb,t, t = 1,...,T, (3) where ⊙ denotes element-wise multiplication. In Eq. 2, we thus set the conditioning variable to c = (M,X), i.e., ϵθ(zt | vb,vf,c,t) = ϵθ(zt | vb,vf,M,X,t). In this way, the user trajectory and scale are encoded entirely through the mask and masked background videos fed into the network.

As shown in the left of Fig. 2, our compositing starts from two input videos (background and foreground). Given a background video to be edited, users drag a trajectory on its first frame, denoting the movement of added elements. Alternatively, users can click a point (as shown in Fig. 1 (a)), and our method will automatically track the trajectory of this point in subsequent frames based on video optical flow. On this basis, the remaining problem is to determine the size of the newly added element. For the foreground input, we get its corresponding binary mask video through Grounded SAM2(Ren et al., 2024), then adjust its region size according to the user-given rescale factor. Finally, we adjust the position of this rescaled mask video based on the designated trajectory curve. In this way, we adaptively rescale and reposition foreground masks to produce a mask video.

To integrate with environment realistically, a Gaussian filter is used to inflate mask. Given the binary foreground mask video Mbin = {Mbint }Tt=1, mask is inflated to define a buffered editable band around the object. We smooth Mbint with a 2D Gaussian kernel Gσ and then apply a threshold:

###### M˜ t = Gσ ∗ Mbint , Mt = 1[M˜ t > τ], t = 1,...,T, (4)

where ∗ denotes convolution. Intuitively, Mt slightly extends the foreground region beyond the object boundary, creating a narrow buffer band in which the model is allowed to modify background content and hallucinate interactions (e.g., shadows, glow, motion blur). At the same time, when SAM2 fails, this inflated mask explicitly absorbs pixel-level misalignments to support imperfect

fg. token noisy token

[Figure 49]

[Figure 50]

Bg. Preservation Branch

bg.tokenfg.tokennoisytoken

[Figure 51]

[Figure 52]

DiTBlock

[Figure 53]

[Figure 54]

Patchify

[Figure 55]

RoPEERoPE

c

scale=1.0

Block

Block

+

Layer Norm 3D Self-attention Gate

DiT

DiT

DiTFusionBlock

[Figure 56]

[Figure 57]

set traj

mask video

MaskInflation

[Figure 58]

++

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

Patchify

[Figure 68]

[Figure 69]

VAEDecoder

[Figure 70]

Layer Norm Feed Forward Net Gate

DiTFusionBlock

DiTFusionBlock

DiTFusionBlock

DiTFusionBlock

[Figure 71]

VAEEncoder

[Figure 72]

c ×…21 ×…21

Noise

×(1-)

bg. video

+

[Figure 73]

masked video

[Figure 74]

[Figure 75]

[Figure 76]

Patchify

[Figure 77]

Luminance Augmentation

0

+

composited video

Fg. Generation Mainstream

fg. video

- Figure 2: Workflow of GenCompositor. GenCompositor takes a background video and a foreground video as input. Users can specify the trajectory and scale of the added foreground elements. We firstly convert user-given instructions to model inputs, then generate with a background preservation branch and a foreground generation mainstream, consisting of the proposed ERoPE and DiT fusion blocks. Following this, our model automatically composites input videos.

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

……

×

……

……

……

……

t w

w

t

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

t

3D RoPE

h

[Figure 88]

[Figure 89]

[Figure 90]

t

w

h

[Figure 91]

[Figure 92]

[Figure 93]

×

Our ERoPE

2h

- A

- B

×(1- )

[Figure 94]

LayerNorm

3DSelf-attention

Gate

LayerNorm

FeedForwardNet

Gate

+ +

DiT Block

+

DiT Fusion Block

noisytokenfg.token

×

(t,0,0) (t,0,1) (t,0,w)

( t,2h-1,w)

(0,0,1)

(0,h,0) (0,h,1)

(0,2h-1,0)(0,2h-1,1) (0,2h-1,w)

(0,1,0)

(0,0,0) (0,0,w) (0,1,w)

(0,h,w)

…… (t,1,w)

……

(0,1,1) ( t,h-1,w)

(t,0,0) (t,0,1) (t,0,w)

( t,h-1,w)

( t,1,w)

- (0,0,0)
- (0,1,0)

(0,h-1,0)

(0,0,w)

(0,h-1,w)

mask video

- Figure 3: DiT fusion block receives concatenated tokens, some are to-be-generated tokens (blue) and some are unaligned conditional tokens (green). It fuses the two via pure self-attention. The final tokens with gradient color represent the mixture of generated tokens and masked conditional tokens from the BPBranch.

t

t

t w (t,0,0)

(t,0,w)

[Figure 95]

……

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

……

- (0,0,0)
- (0,1,0)

(0,0,1)

(0,0,w)

h

h

……

(t,h-1,w)

(0,h-1,w)

- A

- B

w/ RoPE

w

[Figure 103]

t w

(t,0,0) (t,0,w)

[Figure 104]

[Figure 105]

…… (t,1,w)

[Figure 106]

(0,0,0) (0,0,w) (0,1,w)

(0,0,1)

……

(0,1,0)

2h

……

[Figure 107]

(0,h,0) (0,h,1) (t,2h-1,w)

[Figure 108]

(0,h,w)

[Figure 109]

……

[Figure 110]

(0,2h-1,w)

w/ ERoPE

Figure 4: RoPE and ERoPE. (A) RoPE assigns labels to embeddings of videos. (B) For videos with inconsistent layout, our ERoPE extends RoPE by assigning unique labels to each embedding from different videos, avoiding interference.

masks. We also train our model with masks of SAM2, and the trained end-to-end model has a certain mask correction capability when utilization. Finally, masked video can be obtained by simply masking the object in the source video with a mask video for training. Our model actually receives (i) foreground video, (ii) inflated mask video, and (iii) masked video, to generate realistic output. This forces the model, during training, to explain pixels in the whole band around the object using both foreground and background cues, enabling local interaction.

- 4.2 BACKGROUND PRESERVATION BRANCH

As a video editing task, our main goal is to preserve non-editing content and only edit desired regions of vb. We propose a Background Preservation Branch (BPBranch) to ensure consistency between composited result and vb. Considering the overall layout of edited result is pixel-aligned with that of vb. Directly plus the latent of vb with model latent has been able to faithfully preserve non-editing content, which is proven by previous ControlNet-like methods (Mou et al., 2024b; Ju et al., 2024).

Intuitively, only using masked video has been able to inject background. As shown in Fig. 2, in masked video, the area where we aim to insert the element is marked black. However, some background videos naturally contain black content, which is not the desired masks and could confuse the network. Hence, we concatenate the masked video with its corresponding mask video as input. This guides the branch to focus on background preservation.As mask video and masked video are pixel-aligned, we apply the same Rotary Position Embedding (RoPE) (Su et al., 2024) to both, as shown in Fig. 4(A). This strictly aligns the positions of the two and enables precise mask guidance.

Subsequently, we inject these to foreground generation mainstream for compositing. As BPBranch is designed to inject background only, its main purpose is to align features of masked video with that of mainstream model. Instead of deeply extracting masked features, we simply design a lightweight control branch, which consists of two normal DiT blocks, to align with the latent of mainstream. Meanwhile, as we only want to use the background region, a masked token injection method is

applied to prevent interference from BPBranch to foreground region, which is formulated as:

zt = zt + (1 − M) ⊙ zBPBranch, (5)

where zt is the latent from mainstream, and zBPBranch is the output of BPBranch. This process is visualized in detail in the upper part of Fig. 3.

- 4.3 FOREGROUND GENERATION MAINSTREAM

To composite foreground footage, our goal is to faithfully preserve the identity and dynamic features of foreground elements from other sources in composited results. Cross-attention is commonly used to inject conditions (FU et al., 2025; Yu et al., 2025), such as texts or camera poses. However, we find that although cross-attention could address semantic conditions, it does not effectively utilize lowlevel conditional information for our task. To faithfully inherit foreground condition, we propose to concatenate the tokens of foreground with the tokens to be denoised as shown in Fig. 2, then calculate its self-attention to fully fuse these two messages through the DiT fusion block.

As depicted in Fig. 3, given the tokens of noisy latent and foreground condition, DiT fusion block concatenates them in a token-wise manner, instead of classical channel-wise concatenation. This is because the layout features of foreground condition and generated results are not pixel-aligned. Roughly concatenating their features in channel dimension will cause severe content interference, which leads to training collapse. DiT fusion block then predicts the noise in noisy latent by calculating self-attention on the concatenated tokens, which contain both foreground condition and masked background. Note that our generated results are the processed latent that is boxed by red in Fig. 3. Hence, we fuse the tokens of BPBranch only with the processed noisy tokens and pass them to the next block. Finally, we only decode the part corresponding to the input noisy tokens to obtain composited video, as shown in the right of Fig. 2.

Meanwhile, newly added content should visually coordinate with background. Some of its attributes (e.g., lighting), need to be properly adjusted during generation. To enable our model to learn this coordination adaptively, we develop a luminance augmentation strategy for training. In each iteration, we use gamma correction to the foreground video, with the gamma parameter randomly selected from a range of 0.4 to 1.9. This changes the lightness of foreground condition to offset it from source video. Consequently, based on our DiT fusion block that fully fuses foreground elements with model latent, foreground generation mainstream automatically learns the capabilities of foreground harmonization. Notice that luminance augmentation is only used in the training process.

- 4.4 EXTENDED ROTARY POSITION EMBEDDING

We distinguish two types of conditions with respect to the target layout. Given the background and target layout (vb,z0), we call a condition video v layout-aligned if v(x,y) and z0(x,y) refer to the same spatial location in the image plane, and layout-unaligned otherwise. In Fig. 2, the mask video M and masked video X are layout-aligned with (vb,z0), while the foreground video vf is layout-unaligned, since its dynamic content is centered and follows its own camera frame.

Layout-aligned conditions can be well utilized by sharing the same RoPE. However, this causes content interference for layoutunaligned conditions. As shown in Fig. 5, “w/ RoPE” directly shares RoPE between layoutaligned (vb,M,X) and layout-unaligned vf, which leads to obvious artifacts in composited results (boxed in red). Whose shapes and positions are consistent with that of foreground elements. Another optional way is cross-attention, which extracts abstract semantics to support layout-unaligned control signals. However, such a high-level feature cannot faithfully inherit detailed appearance and action features of foreground videos and is not suitable for video compositing, as proved in Sect. 5.4.

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

background video foreground video mask video w/ RoPE w/ ERoPE

Figure 5: ERoPE is superior in fusing layoutunaligned videos. As dynamic content in foreground footage is centered, using RoPE leads to content interference as shown in red box of “w/ RoPE”, while our ERoPE resolves this issue well.

To this end, we developed a new embedding strategy tailored to the unaligned nature between foreground footage and background video, named ERoPE. It still takes the same four inputs

manually paste

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

Harmonizer

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

VideoTripletTransformer

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

GenCompositor

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

- Figure 6: Visual comparison of video harmonization. The compared methods cannot achieve satisfactory results with jagged artifacts at the edges of foreground elements, inconsistent color or lighting, while our method achieves better performance.

(vb,M,X,vf) as in Fig. 2, but changes the position embedding: we assign distinct positional labels to layout-unaligned tokens so that foreground and background videos can be fused without occupying the same spatial positions in the latent space. Concretely, in RoPE (Su et al., 2024), each token at 1D position index p has its query/key components divided into 2D pairs and rotated by an angle proportional to p. For the k-th frequency ωk and 2D query component (q2k,q2k+1), we have:

2，22，40 2，22，32

q2′k q2′k+1

= R(θp,k)

q2k q2k+1

, θp,k = ωkp, R(θ) =

cosθ −sinθ sinθ cosθ

, (6)

and the same to the key component. Resulting attention score depends only on the relative position:

⟨qp′ ,kq′ ⟩ ∝ f {ωk(p − q)}k , (7) which implicitly assumes all tokens lie on a single spatio-temporal grid as shown in Fig. 4(A). In our setting, tokens come from ForeGround and BackGround videos s ∈ {BG,FG} with incompatible spatial coordinates. ERoPE introduces a stream-specific shift on top of RoPE:

θs,p,k = ωk(p + ∆s), s ∈ {BG,FG}, (8)

where ∆s is a constant offset for stream s. Let qs,p ∈ RD and ks′,q ∈ RD denote the unrotated query and key for a token from stream s at position p and a token from stream s′ at position q, respectively. Here s,s′ ∈ {BG,FG} indicate the source streams of the query and key tokens. They may be equal for within-stream attention or different for cross-stream attention. We write qs,p′ and ks′′,q for their ERoPE-rotated versions obtained by applying R(θs,p,k) and R(θs′,q,k) to each (2k,2k + 1) pair. Using the identity R(θ)⊤R(ϕ) = R(ϕ − θ) and writing qs,p,k = (qs,p,2k,qs,p,2k+1)⊤, ks′,q,k = (ks′,q,2k,ks′,q,2k+1)⊤ for 2D slices of qs,p and ks′,q, attention score between these two tokens is:

qs,p′ ,ks′′,q =

=

k

k

q⊤s,p,k R(θs′,q,k − θs,p,k) ks′,q,k

q⊤s,p,k R ωk (q + ∆s′) − (p + ∆s) ks′,q,k.

(9)

In other words, the contribution of each frequency k depends only on the relative effective positions (p + ∆s) and (q + ∆s′), and hence on their difference ωk[(p + ∆s) − (q + ∆s′)]. Choosing large, distinct shifts ∆s for BG and FG prevents “same-position” collisions across streams, while still allowing attention to learn meaningful cross-stream interactions. As shown in Fig. 4(B), ERoPE is implemented as stream-specific shifts ∆s on top of standard RoPE, introduces no additional parameters, and is crucial for stable fusion of layout-unaligned foreground and background videos in our compositing setting. As shown in Fig. 5, this strategy efficiently increases compositing performance and eliminates artifacts caused by interference of unaligned content. Moreover, considering that there are three optional shift directions for our ERoPE (e.g., width, height, and timing), this paper conducts ablation in Sect. F of appendix, and demonstrates their equality. We believe ERoPE is a practical tool for numerous layout-unaligned editing tasks, and here we use it for video compositing.

- 5 EXPERIMENTS

Given that there are no existing works for generative video compositing like ours, we compare with two related tasks, video harmonization, and trajectory-controlled video generation. We first introduce implementation details in Sect. 5.1, showcase comparisons in Sect. 5.2 and Sect. 5.3. To validate the effectiveness of our key components, we conduct an ablation study in Sect. 5.4.

Table 1: Quantitative comparison with video harmonization methods. The best results are highlighted in bold.

Table 2: Quantitative results of comparison with trajectory-controlled generation. The best results are highlighted in bold.

|Metrics<br><br>|Tora Revideo VACE GenCompositor|
|---|---|
|Subject ↑ Background ↑ Motion ↑ Aesthetic ↑ FVD ↓ KVD ↓<br><br>|88.44% 88.02% 89.51% 89.75% 92.45% 92.90% 92.63% 93.43% 98.03% 96.85% 98.21% 98.69% 49.33% 48.56% 49.30% 52.00% 1402.82 1342.56 942.52 535.71 94.94 64.53 120.92 45.91|

|Metrics<br><br>|Harmonizer VTT GenCompositor|
|---|---|
|PSNR ↑ SSIM ↑ CLIP ↑ LPIPS ↓|39.7558 40.0251 42.0010 0.9402 0.9297 0.9487 0.9614 0.9564 0.9713 0.0412 0.0455 0.0385<br><br>|

- 5.1 IMPLEMENTATION DETAILS

GenCompositor is a DiT model with a 6B Transformer, consisting of the proposed DiT fusion blocks and background preservation branch. We reuse pre-trained VAE module of CogVideoX to generate in latent space. We train our new architecture on 8 H20 GPUs from scratch. In inference, GenCompositor takes over 65s to generate a video at 480×720 with 49 frames within 34GB VRAM.

- 5.2 COMPARISON WITH VIDEO HARMONIZATION

Considering the limited number of open-source methods in video harmonization, we compare our method with two recent approaches whose codes are available: Harmonizer (Ke et al., 2022) and VTT (Guo et al., 2024b). As these methods cannot control the motion trajectory, we only compare the ability to harmonize foreground elements. We also manually paste foreground footage from other sources into the background video for comparison. As shown in left of Fig. 6, noticeable jagged artifacts appear at the edges of added elements in manually paste and Harmonizer, due to the imperfection of the segmentation mask of the foreground video. Additionally, the color style of the newly added explosion effect does not harmonize well with background video in these harmonization approaches. In contrast, our method effectively addresses these jagged artifacts caused by inaccurate masks and produces more harmonious results. In other examples of Fig. 6, we manually adjust the lighting of the foreground elements and test the video harmonization ability. Our method consistently outperforms the other methods, demonstrating its superiority.

For quantitative comparison, we use the well-known HYouTube (Lu et al., 2022) dataset, where the foreground videos, segmentation masks, and source videos are available. Four well-known metrics, PSNR, SSIM (Wang et al., 2004), CLIP (Radford et al., 2021) and LPIPS (Zhang et al., 2018) are used to measure performance in Tab. 1. One can see that GenCompositor outperforms other harmonization methods across all metrics, showing its superiority over specialized methods in video harmonization. We also conduct a user study in Sect. E of appendix.

- 5.3 COMPARISON WITH CONTROLLABLE GENERATION

Our method enables trajectory-controlled generation, where users can specify the motion trajectory and size of newly added elements in results. Considering our added element is a dynamic video and there is no method using the same condition as ours, we compare with SOTA trajectory-controlled video generation and editing methods, Tora (Zhang et al., 2025), Revideo (Mou et al., 2024a), and VACE (Jiang et al., 2025). Visual results are given in Fig. 7, where we report background videos, foreground videos, Tora generation results containing the red trajectory curves, Revideo editing results, VACE generation, and our results. Note that Tora and VACE still require additional textual prompts as condition, Revideo and VACE have to edit the first frame as image condition, but GenCompositor requires neither of these priors. Although Tora and VACE could generate results that follow the trajectory, Tora cannot maintain the ID consistency of the added element and cannot strictly follow the user-specified trajectory. VACE gets object information only from the first-frame reference image, tends to mistake the object we wanted to inject and cannot predicts faithful dynamics. In contrast, GenCompositor could strictly follow the trajectories to generate composited videos,

foreground video

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

background video

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

Tora

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

ReVideo

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

###### VACE

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

GenCompositor

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

- Figure 7: Visual comparison of trajectory-controlled video generation. All methods share the same trajectory, which is plotted as a red line in the results of Tora. All compared methods need to receive the pre-edited 1st-frame as condition image. Tora and Revideo generate temporally inconsistent foreground elements. VACE mistakes the element details we actually want to inject. Our method resolves these issues well without the preprocess of editing the first frame.

0，14，28，42

###### Table 3: Quantitative results of ablation study. The best results are highlighted in bold.

|Settings|PSNR ↑ SSIM ↑ CLIP ↑ LPIPS ↓<br><br>|Subject ↑ Background ↑ Motion ↑ Aesthetic ↑|
|---|---|---|
|w/o fusion block w/o BPBranch w/o augmentation w/o mask inflation fullmodel<br><br>|19.8940 0.8015 0.9341 0.1535<br><br>40.0099 0.9378 0.9709 0.0432 39.8040 0.9295 0.9629 0.0520<br><br>41.8553 0.9422 0.9701 0.0409<br><br>42.0010 0.9487 0.9713 0.0385<br><br><br>|88.85% 92.21% 98.34% 48.85%<br><br>88.77% 89.62% 97.25% 51.51%<br><br>88.00% 89.97% 98.30% 50.73%<br>89.72% 91.62% 98.28% 50.87%<br><br><br>89.75% 93.43% 98.69% 52.00%<br>|

and the ID and motion of the element are inherited from foreground videos faithfully. We believe these advantages come from different task settings. Compared with generating a video from an image such as Tora, or inpainting based on only single reference such as VACE, we composite foreground video following the trajectories. This is inherently more conducive to inheriting the ID and detailed motion of foreground elements. Meanwhile, Revideo also aims to drag the added element in the first frame to move along a given trajectory in subsequent frames, but its limited performance leads to non-robust results, elements may disappear in its predicted frames as shown in Fig. 7.

To conduct quantitative evaluations, we utilize two well-known generation quality metrics: FVD (Unterthiner et al., 2019) and KVD, and a common benchmark, i.e., VBench (Huang et al., 2024; Zheng et al., 2025), to analyze the quality of generation from 4 dimensions: 1) Subject Consistency: consistency about subjects in the video. 2) Background Consistency: consistency about the video background. 3) Motion Smoothness: motion quality of the generated video. 4) Aesthetic Quality: subjective visual quality of the video. We believe these four metrics to be most relevant to our task and generate 40 sets of videos using Tora, Revideo, VACE, and our method, respectively. As shown in Tab. 2, our method achieves the best average scores across all six metrics. In addition, we conduct user study in Sect. E of appendix to compare intuitive quality of different methods.

- 5.4 ABLATION STUDY

To analyze effectiveness of each component, we conduct ablation study with four settings. Specifically, we attempt two potential designs of GenCompositor, one removes the background preservation branch (w/o BPBranch), the other uses cross-attention to inject foreground elements without the proposed DiT fusion block (w/o fusion block). We discuss their architecture details in Sect. G. We also remove two designs. One does not inflate binary mask (w/o mask inflation), the other directly inputs original foreground videos for training, without luminance augmentation (w/o augmentation).

foreground element inflated mask

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

manually paste w/o fusion block

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

w/o BPBranch w/o augmentation

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

w/o mask inflation fullmodel

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

- Figure 8: Visual ablation results. Manually paste results own jagged artifacts. “w/o fusion block” cannot inject element faithfully. “w/o augmentation” and “w/o mask inflation” showcase jagged artifacts at the edge of element. “w/o BPBranch” cannot realistically adjust added element. “fullmodel” setting performs best, which naturally fuses foreground elements with background video.

消融visual result 0/14/28/48, 1.7

For visual comparison in Fig. 8, we provide the foreground element and inflated mask in the first row. Following three rows show the results of different settings. Where “manually paste”, “w/o augmentation” and “w/o mask inflation” all exhibit obvious jagged artifacts at the edges of foreground elements. We believe that, for “w/o augmentation”, due to the powerful learning ability of network, it totally inherits and overfits the content of the foreground video without any adjustment. For “w/o mask inflation”, as we provide pixel-aligned mask to model, it can only process the foreground element in this limited region and cannot adjust surrounding pixels to fuse with the background, leaving artifacts at the edge. The other two settings address the border artifacts, but present other limitations. “w/o fusion block” cannot faithfully inject ID and dynamics of foreground elements, but successfully predicts the flame, which is semantically consistent with the foreground condition, indicating that cross-attention is good at injecting semantic information, but is not applicable in our task. Although “w/o BPBranch” also produces a realistic background, this end-to-end learning of both background video and added foreground elements increases training difficulty, limiting its performance. In its results, the foreground element is not perceptually consistent with main video.

For quantitative evaluation, we use four ablation settings, together with “fullmodel”, to realize video harmonization and trajectory-controlled video generation, respectively. Related test data is the same as Sect. 5.2 and Sect. 5.3. As shown in Tab. 3, “fullmodel” outperforms all ablation settings on all metrics. We believe this objectively demonstrates the significance of each component in our method.

- 6 CONCLUSION

This paper introduces a novel video editing task, generative video compositing, which allows interactive video editing using dynamic visual elements. Specifically, we developed the first generative method, GenCompositor, which is designed to address three main challenges of this task: maintaining content consistency before and after editing, injecting video elements, and facilitating user control. It comprises three main contributions. Firstly, a lightweight background preservation branch is utilized to inject tokens of background videos into the mainstream. Secondly, the foreground generation mainstream incorporates novel DiT fusion blocks to effectively fuse the external video condition with the background latent. Finally, we revised a novel position embedding, ERoPE, to force the model to add external elements to the desired positions in results with desired scales, adaptively. Notice that ERoPE points out a new effective way to utilize layout-unaligned video conditions for generative model without any additional computational cost. The first paired dataset, VideoComp, is also proposed in this paper. Comprehensive experiments demonstrate the effectiveness and practicality of our proposed method.

Acknowledgments. This work was financially supported in part by National Natural Science Foundation of China (62372016), Guangdong Provincial Key Laboratory of Ultra High Definition Immersive Media Technology (2024B1212010006), Shenzhen Science and Technology Program (SYSPG20241211173440004) and Outstanding Talents Training Fund in Shenzhen.

REFERENCES

Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond. arXiv preprint arXiv:2308.12966, 2023.

Yuxuan Bian, Zhaoyang Zhang, Xuan Ju, Mingdeng Cao, Liangbin Xie, Ying Shan, and Qiang Xu. Videopainter: Any-length video inpainting and editing with plug-and-play context control. In Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference, 2025.

Blackforestlabs. Flux. https://github.com/black-forest-labs/flux, 2024.

Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, Varun Jampani, and Robin Rombach. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023.

Tim Brooks, Aleksander Holynski, and Alexei A. Efros. Instructpix2pix: Learning to follow image editing instructions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023.

Minghong Cai, Xiaodong Cun, Xiaoyu Li, Wenze Liu, Zhaoyang Zhang, Yong Zhang, Ying Shan, and Xiangyu Yue. Ditctrl: Exploring attention control in multi-modal diffusion transformer for tuning-free multi-prompt longer video generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025.

Mingdeng Cao, Xintao Wang, Zhongang Qi, Ying Shan, Xiaohu Qie, and Yinqiang Zheng. Masactrl: Tuning-free mutual self-attention control for consistent image synthesis and editing. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023.

Duygu Ceylan, Chun-Hao P. Huang, and Niloy J. Mitra. Pix2video: Video editing using image diffusion. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023.

Xiuwen Chen, Li Fang, Long Ye, and Qin Zhang. Deep video harmonization by improving spatialtemporal consistency. Machine Intelligence Research, 2024.

Yi Dong, Yuxi Wang, Zheng Fang, Wenqi Ouyang, Xianhui Lin, Zhiqi Shen, Peiran Ren, Xuansong Xie, and Qingming Huang. Movingcolor: Seamless fusion of fine-grained video color enhancement. In Proceedings of the ACM International Conference on Multimedia, 2024.

Patrick Esser, Johnathan Chiu, Parmida Atighehchian, Jonathan Granskog, and Anastasis Germanidis. Structure and content-guided video synthesis with diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023.

Xiao FU, Xian Liu, Xintao Wang, Sida Peng, Menghan Xia, Xiaoyu Shi, Ziyang Yuan, Pengfei Wan, Di ZHANG, and Dahua Lin. 3DTrajmaster: Mastering 3d trajectory for multi-entity motion in video generation. In Proceedings of the International Conference on Learning Representations, 2025.

Jiaqi Guo, Lianli Gao, Junchen Zhu, JiaxinZhang, Siyang Li, and Jingkuan Song. Magicvfx: Visual effects synthesis in just minutes. In Proceedings of the ACM International Conference on Multimedia, 2024a.

Zonghui Guo, Xinyu Han, Jie Zhang, Shiguang Shan, and Haiyong Zheng. Video harmonization with triplet spatio-temporal variation patterns. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024b.

Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video diffusion models. In Proceedings of the Advances in Neural Information Processing Systems, 2022.

Li Hu. Animate anyone: Consistent and controllable image-to-video synthesis for character animation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.

Teng Hu, Zhentao Yu, Zhengguang Zhou, Sen Liang, Yuan Zhou, Qin Lin, and Qinglin Lu. Hunyuancustom: A multimodal-driven architecture for customized video generation. arXiv preprint arXiv:2505.04512, 2025.

Hao-Zhi Huang, Sen-Zhe Xu, Jun-Xiong Cai, Wei Liu, and Shi-Min Hu. Temporally coherent video harmonization using adversarial networks. IEEE Transactions on Image Processing, 2020.

Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, Yaohui Wang, Xinyuan Chen, Limin Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. Vbench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.

Zeyinzi Jiang, Zhen Han, Chaojie Mao, Jingfeng Zhang, Yulin Pan, and Yu Liu. Vace: All-inone video creation and editing. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025.

Xuan Ju, Xian Liu, Xintao Wang, Yuxuan Bian, Ying Shan, and Qiang Xu. Brushnet: A plugand-play image inpainting model with decomposed dual-branch diffusion. In Proceedings of the European Conference on Computer Vision, 2024.

Xuan Ju, Weicai Ye, Quande Liu, Qiulin Wang, Xintao Wang, Pengfei Wan, Di Zhang, Kun Gai, and Qiang Xu. Fulldit: Multi-task video generative foundation model with full attention. arXiv preprint arXiv:2503.19907, 2025.

Zhanghan Ke, Chunyi Sun, Lei Zhu, Ke Xu, and Rynson W. H. Lau. Harmonizer: Learning to perform white-box image and video harmonization. In Proceedings of the European Conference on Computer Vision, 2022.

Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024.

Lingen Li, Guangzhi Wang, Zhaoyang Zhang, Yaowei Li, Xiaoyu Li, Qi Dou, Jinwei Gu, Tianfan Xue, and Ying Shan. Tooncomposer: Streamlining cartoon production with generative postkeyframing. arXiv preprint arXiv:2508.10881, 2025a.

Lingen Li, Zhaoyang Zhang, Yaowei Li, Jiale Xu, Wenbo Hu, Xiaoyu Li, Weihao Cheng, Jinwei Gu, Tianfan Xue, and Ying Shan. Nvcomposer: Boosting generative novel view synthesis with multiple sparse and unposed images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025b.

Weiqi Li, Shijie Zhao, Chong Mou, Xuhan Sheng, Zhenyu Zhang, Qian Wang, Junlin Li, Li Zhang, and Jian Zhang. Omnidrag: Enabling motion control for omnidirectional image-to-video generation. International Journal of Computer Vision, 2026.

Yaowei Li, Xintao Wang, Zhaoyang Zhang, Zhouxia Wang, Ziyang Yuan, Liangbin Xie, Ying Shan, and Yuexian Zou. Image conductor: Precision control for interactive video synthesis. In Proceedings of the AAAI Conference on Artificial Intelligence, 2025c.

Yuhang Li, Jincen Jiang, Xiaosong Yang, Youdong Ding, and Jian Jun Zhang. Harmony everything! masked autoencoders for video harmonization. In Proceedings of the ACM International Conference on Multimedia, 2024.

Risheng Liu, Zhiying Jiang, Shuzhou Yang, and Xin Fan. Twin adversarial contrastive learning for underwater image enhancement and beyond. IEEE Transactions on Image Processing, 2022.

Xinyu Liu, Ailing Zeng, Wei Xue, Harry Yang, Wenhan Luo, Qifeng Liu, and Yike Guo. Vfx creator: Animated visual effect generation with controllable diffusion transformer. arXiv preprint arXiv:2502.05979, 2025.

Xinyuan Lu, Shengyuan Huang, Li Niu, Wenyan Cong, and Liqing Zhang. Deep video harmonization with color mapping consistency. arXiv preprint arXiv:2205.00687, 2022.

Zhiyuan Ma, Xinyue Liang, Rongyuan Wu, Xiangyu Zhu, Zhen Lei, and Lei Zhang. Progressive rendering distillation: Adapting stable diffusion for instant text-to-mesh generation without 3d data. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025.

Chong Mou, Mingdeng Cao, Xintao Wang, Zhaoyang Zhang, Ying Shan, and Jian Zhang. Revideo: Remake a video with motion and content control. In Proceedings of the Advances in Neural Information Processing Systems, 2024a.

Chong Mou, Xintao Wang, Liangbin Xie, Yanze Wu, Jian Zhang, Zhongang Qi, and Ying Shan. T2i-adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models. In Proceedings of the AAAI Conference on Artificial Intelligence, 2024b.

Chenyang QI, Xiaodong Cun, Yong Zhang, Chenyang Lei, Xintao Wang, Ying Shan, and Qifeng Chen. Fatezero: Fusing attentions for zero-shot text-based video editing. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In Proceedings of the International Conference on Machine Learning, 2021.

Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman R¨adle, Chloe Rolland, Laura Gustafson, et al. SAM 2: Segment anything in images and videos. In Proceedings of the International Conference on Learning Representations, 2025.

Tianhe Ren, Shilong Liu, Ailing Zeng, Jing Lin, Kunchang Li, He Cao, Jiayu Chen, Xinyu Huang, Yukang Chen, Feng Yan, Zhaoyang Zeng, Hao Zhang, Feng Li, Jie Yang, Hongyang Li, Qing Jiang, and Lei Zhang. Grounded sam: Assembling open-world models for diverse visual tasks. arXiv preprint arXiv:2401.14159, 2024.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022.

Chaehun Shin, Heeseung Kim, Che Hyun Lee, Sang-gil Lee, and Sungroh Yoon. Edit-A-Video: Single video editing with object-aware consistency. In Proceedings of the Asian Conference on Machine Learning, 2024.

Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 2024.

Jianchao Tan, Jose Echevarria, and Yotam Gingold. Palette-based color harmonization. IEEE Transactions on Visualization and Computer Graphics, 2025.

Yuanpeng Tu, Hao Luo, Xi Chen, Sihui Ji, Xiang Bai, and Hengshuang Zhao. Videoanydoor: Highfidelity video object insertion with precise motion control. In Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference, 2025.

Thomas Unterthiner, Sjoerd van Steenkiste, Karol Kurach, Rapha¨el Marinier, Marcin Michalski, and Sylvain Gelly. FVD: A new metric for video generation. In Proceedings of the ICLR 2019 Workshop on DeepGenStruct, 2019.

Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, Jianyuan Zeng, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025a.

Hanlin Wang, Hao Ouyang, Qiuyu Wang, Wen Wang, Ka Leong Cheng, Qifeng Chen, Yujun Shen, and Limin Wang. Levitor: 3d trajectory oriented image-to-video synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025b.

Qian Wang, Weiqi Li, Chong Mou, Xinhua Cheng, and Jian Zhang. 360dvd: Controllable panorama video generation with 360-degree video diffusion model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024a.

Weihan Wang, Qingsong Lv, Wenmeng Yu, Wenyi Hong, Ji Qi, Yan Wang, Junhui Ji, Zhuoyi Yang, Lei Zhao, Xixuan Song, Jiazheng Xu, Keqin Chen, Bin Xu, Juanzi Li, Yuxiao Dong, Ming Ding, and Jie Tang. Cogvlm: Visual expert for pretrained language models. In Proceedings of the Advances in Neural Information Processing Systems, 2024b.

Wen Wang, Kangyang Xie, Zide Liu, Hao Chen, Yue Cao, Xinlong Wang, and Chunhua Shen. Zeroshot video editing using off-the-shelf image diffusion models. arXiv preprint arXiv:2303.17599, 2023.

Zhou Wang, A.C. Bovik, H.R. Sheikh, and E.P. Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE Transactions on Image Processing, 2004.

Zhouxia Wang, Ziyang Yuan, Xintao Wang, Yaowei Li, Tianshui Chen, Menghan Xia, Ping Luo, and Ying Shan. Motionctrl: A unified and flexible motion controller for video generation. In Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference, 2024c.

Jay Zhangjie Wu, Yixiao Ge, Xintao Wang, Stan Weixian Lei, Yuchao Gu, Yufei Shi, Wynne Hsu, Ying Shan, Xiaohu Qie, and Mike Zheng Shou. Tune-a-video: One-shot tuning of image diffusion models for text-to-video generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023.

Zeyu Xiao, Yurui Zhu, Xueyang Fu, and Zhiwei Xiong. Tsa2: Temporal segment adaptation and aggregation for video harmonization. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, 2024.

Shuzhou Yang, Moxuan Ding, Yanmin Wu, Zihan Li, and Jian Zhang. Implicit neural representation for cooperative low-light image enhancement. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023.

Shuzhou Yang, Chong Mou, Jiwen Yu, Yuhan Wang, Xiandong Meng, and Jian Zhang. Neural video fields editing. arXiv preprint arXiv:2312.08882, 2024.

Shuzhou Yang, Xiaodong Cun, Xiaoyu Li, Yaowei Li, and Jian Zhang. 4dvd: Cascaded dense-view video diffusion model for high-quality 4d content generation. arXiv preprint arxiv:2508.04467,

- 2025a.

Shuzhou Yang, Yu Wang, Haijie Li, Jiarui Meng, Yanmin Wu, Xiandong Meng, and Jian Zhang. Hybrid fourier score distillation for efficient one image to 3d object generation. Visual Intelligence,

- 2025b.

Shuzhou Yang, Xuanyu Zhang, Yinhuai Wang, Jiwen Yu, Yuhan Wang, and Jian Zhang. Difflle: Diffusion-based domain calibration for weak supervised low-light image enhancement. International Journal of Computer Vision, 2025c.

Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, Da Yin, Yuxuan.Zhang, Weihan Wang, Yean Cheng, Bin Xu, Xiaotao Gu, Yuxiao Dong, and Jie Tang. Cogvideox: Text-to-video diffusion models with an expert transformer. In Proceedings of the International Conference on Learning Representations, 2025d.

Danah Yatim, Rafail Fridman, Omer Bar-Tal, and Tali Dekel. Dynvfx: Augmenting real videos with dynamic content. In Proceedings of the SIGGRAPH Asia, 2025.

Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ip-adapter: Text compatible image prompt adapter for text-to-image diffusion models. arXiv preprint arxiv:2308.06721, 2023.

Mark Yu, Wenbo Hu, Jinbo Xing, and Ying Shan. Trajectorycrafter: Redirecting camera trajectory for monocular videos via diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025.

Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023.

Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2018.

Zhenghao Zhang, Junchao Liao, Menghao Li, ZuoZhuo Dai, Bingxue Qiu, Siyu Zhu, Long Qin, and Weizhi Wang. Tora: Trajectory-oriented diffusion transformer for video generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025.

Dian Zheng, Ziqi Huang, Hongbo Liu, Kai Zou, Yinan He, Fan Zhang, Lulu Gu, Yuanhan Zhang, Jingwen He, Wei-Shi Zheng, Yu Qiao, and Ziwei Liu. Vbench-2.0: Advancing video generation benchmark suite for intrinsic faithfulness. arXiv preprint arXiv:2503.21755, 2025.

Xianpan Zhou. Tiger200k: Manually curated high visual quality video dataset from ugc platform. arXiv preprint arXiv:2504.15182, 2025.

CONTENTS

- 1 Introduction 2
- 2 Related Work 3

- 2.1 Diffusion-based Video Editing . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 3
- 2.2 Video Harmonization . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 3

- 3 Task Definition 4
- 4 Method 4

- 4.1 Input Conversion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4
- 4.2 Background Preservation Branch . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5
- 4.3 Foreground Generation Mainstream . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6
- 4.4 Extended Rotary Position Embedding . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6

- 5 Experiments 8

- 5.1 Implementation Details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8
- 5.2 Comparison with Video Harmonization . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8
- 5.3 Comparison with Controllable Generation . . . . . . . . . . . . . . . . . . . . . . . . . . . 8
- 5.4 Ablation Study . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9

- 6 Conclusion 10

- A Statement 17
- B Dataset Construction 17

- B.1 Data curation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
- B.2 Data filtering . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
- B.3 Data type analysis . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18

- C Generalization Ability 19
- D Implementation Details 19
- E User Study 20
- F Loss Curve of Applying ERoPE 21
- G Ablation Architectures 22
- H Interactivity 22
- I More Visual Results 22
- J Discussion 23
- K Interactive Web App 24

Label Curation

Source Video

Collected Data (409K)

Describe this video in detail, ensuring … Focus on transformations throughout the video, including …

[Figure 255]

###### …… ……

NULL

Human, Coffee, ...

[Figure 256]

The video features a serene underwater scene where a grey nurse shark swims gracefully …

[Figure 257]

[Figure 258]

[Figure 259]

[Prompt]

###### QWen QWen

[Figure 260]

[Figure 261]

[Figure 262]

CogVLM

[Figure 263]

Identify the most prominent dynamic object in the video, such as … Answer with a set of nouns, separate with commas. Return NULL if none.

Human, Coffee, ...

[Question]

###### ……

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

Fish

[Figure 268]

[Figure 269]

Incomplete Over-broken

[Figure 270]

QWen

[Figure 271]

Complete

[Figure 272]

Grounded SAM2

[Figure 273]

[Figure 274]

Filter

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

……

Manual Screening

……

Foreground Video Mask Video

VideoComp (61K)

(a) Data Curation

(b) Data Filtering

- Figure 9: Dataset construction pipeline. We construct VideoComp dataset with two stages: data curation and data filtering. The former includes three steps: collection, labeling, and segmentation. Then, we select a high-quality subset based on several rules.

- A STATEMENT

Reproducibility statement. We have provided anonymous code in supplementary materials. Due to file size limitations, model weights are not attached, but one can understand any details of our method in our code itself. For reproducibility, we will open-source all contributions of this paper when it is published, including code, model weights, and organized dataset. In addition, we provide a complete description of the data processing steps in Sect. B. The related code and the complete dataset will be open-sourced when this paper is published.

Ethics statement. This work does not involve human subjects, personal data, or any experiments that may raise ethical concerns. All datasets used are curated from internal source videos with appropriate licenses. The proposed methodology is intended for research purposes in video editing and does not pose foreseeable risks regarding fairness, privacy, or potential misuse. The authors have carefully reviewed and adhered to the ICLR Code of Ethics.

The Use of Large Language Models (LLMs). This paper did not involve the use of Large Language Models (LLMs) for research ideation, content generation, or writing. All content presented in this submission was created by the authors without significant contribution from LLMs. We confirm that we take full responsibility for the accuracy and integrity of the contents in this paper.

- B DATASET CONSTRUCTION

Since there is no existing dataset for video compositing, we carefully build a dataset called VideoComp for this task. We present a scalable dataset construction pipeline that utilizes advanced models as tools for data construction (Wang et al., 2024b; Bai et al., 2023; Ren et al., 2024). The proposed VideoComp dataset contains 61K groups of videos, each containing three video samples, i.e., the source video, the foreground video, and its corresponding mask video. The source video is a high-quality raw video and the other two are extracted

[Figure 284]

[Figure 285]

[Figure 286]

- Figure 10: Data type analysis. We analyze data characteristics including category diversity, domain coverage, and motion distribution. Each is represented in a pie chart.

from it. As shown in Fig. 9, our dataset construction process consists of two main steps, data curation and data filtering.

- B.1 DATA CURATION

(1) We firstly collect about 240K cinematic HD videos that meet high aesthetic standards and large motion. Moreover, recently released Tiger200K (Zhou, 2025), which consists of 169K videos, is also included as our data source. For these total 409K videos, we propose an end-to-end workflow to process each of them as follows. (2) We distinguish the labels of prominent dynamic elements in the source video. We set up two questions to call CogVLM (Wang et al., 2024b) and QWen (Bai et al., 2023) respectively, where the former is a visual language model and the latter is a large language model. We input the video frames and then ask CogVLM to get the detailed description of video. Based on this answer and our second question, QWen is used to identify prominent dynamic elements, or return NULL if there are none. (3) Based on the output label of QWen, we employ Grounded SAM2 (Ren et al., 2024) to segment the elements in the video and save its foreground video and mask video. Note that the mask video shows the original motion trajectory. However, when saving the foreground video, we center the dynamic element in each frame, eliminating its global position and trajectory information. This approach allows us to control the trajectory of the resulting video totally based on the mask video, rather than relying on the position in the foreground video.

- B.2 DATA FILTERING

To ensure a high-quality dataset construction, we filter out unsatisfactory cases according to several rules. As illustrated in right of Fig. 9, our filtering principles are as follows: First, we exclude cases where QWen returns NULL, indicating that there is no significant object in the video. Second, for videos containing multiple elements, we select only the element with the highest probability. Third, we exclude suboptimal cases where elements have incomplete or excessively fragmented structures. Finally, we manually filter out videos that are visually unappealing.

- B.3 DATA TYPE ANALYSIS

To ease utilization, here we showcase three pie chats to analyze our VideoComp from three perspectives respectively, including category diversity, domain coverage, and motion distribution. As shown in Fig. 10.

- (1) Category diversity: we list 5 categories of topics. Where half (49.4%) of the videos feature various animals, and human subjects account for almost a quarter (23.0%). Various vehicles account for 14.5%, including plane, car, and bus. 6.8% of videos record landscape , and urban scene occupies 2.8%. The remaining 3.3% is about complex topics.
- (2) Domain coverage: we analyze the types of scenes displayed in the video data. The majority of these are nature outdoor pieces (64.6%), which typically offer a wider field of view and incorporate various real-world physical interactions and natural movements. This indirectly demonstrates the high quality of VideoComp. Indoor scenes account for 14.9%, urban outdoor accounts for 7.0%, and sports venue taks about 5.0%. The remaining 4.2% is about underwater scenes, and the other 4.3% includes various other scenes.
- (3) Motion distribution: we also list the type of motion in the right end of Fig. 10. Where extreme motion only occupies 9.4% and 1.9%, for fast and low motion, respectively. Over a quarter of videos (25.7%) contain moderate subject motion, and almost a half of videos (48.8%) actually have mixed motion, which include multiple subjects with different amplitudes of motion. These strongly support the validity of the content in the proposed VideoComp dataset. In additiion, our data also contains approximately 14.2% camera motion video, which makes the model practical in real application.

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

###### source videos blank foreground mask videos masked videos results

- Figure 11: Generalizability. After training for video compositing, our method seamlessly enables video inpainting and the removal of target objects from videos with blank foreground condition.

- C GENERALIZATION ABILITY

In addition to the effective video compositing capability of the proposed GenCompositor, our method also demonstrates impressive generalizability in video inpainting and the removal of target objects from videos, which can be achieved by simply replacing the foreground condition with a blank foreground video.

As shown in Fig. 11, we employ a blank foreground condition to remove target objects in source videos through the trained GenConpositor. We first use SAM2 (Ravi et al., 2025) to obtain mask videos corresponding to the objects to be removed in source videos. Then, as mentioned in Sect. 4.1, the same mask inflation operation is applied to get the inflated mask videos and masked videos. Given the blank foreground condition, GenCompositor removes objects and paints the masked region well. This successful generalization study demonstrates the significance of our proposed generative video compositing task, i.e., this new video editing task inherently supports other video-related downstream tasks.

- D IMPLEMENTATION DETAILS

We refer to CogVideoX-I2V-5B to design our model. Similarly to other DiT models, our GenCompositor also consists of 3 main components, Transformer, VAE, and text encoder, where we inherit the pre-trained weights of VAE and text encoder of CogVideoX in our model and do not train them. We only train Transformer here. Notice that as we provide null text during training, although GenCompositor inherits a text encoder, for inference, video compositing is totally based on the input videos and user-specified control, which is consistent with classical handmade process. The number of parameters of each component is: VAE (215,583,907), text encoder (4,762,310,656), Transformer (5,872,247,936), totaling 10,850,142,499. Following previous work, we only consider parameter number of Transformer in latent diffusion model, and claim that GenCompositor is a 6B model.

Our Diffusion Transformer contains two branches, one background preservation branch containing 301,568,576 parameters, and one foreground generation mainstream with 5,570,679,360 parameters. Since the original CogVideoX-I2V-5B is designed for image-to-video generation, we revised a novel module (DiT fusion block) to meet the characteristics of our task, video compositing, and train this new model from scratch. The foreground generation mainstream consists of 42 DiT fusion blocks.

As shown in Fig. 2 in main paper, our new DiT model has three patchify modules, which aim to patchify the features of input videos encoded by the VAE encoder into tokens that can be processed by Transformer, where the features encoded by VAE encoder contains 16 feature channels. Notice that for the background preservation branch, its inputs are a mask video and a masked video, which are concatenated on channel dimension. Hence, the input channel number of its patchify is 32. For foreground generation mainstream, its inputs are masked video, noise input, and foreground video condition. During training, the noise input is the combination of random noise and masked video, similar to other diffusion models. For inference, the noise input is pure gaussian noise. Notice that noise input is concatenate with masked video in channel dimension. Hence the input channel number of blue patchify in Fig. 2 is 32, and the input channel number of green patchify for foreground video condition is 16.

#### Video Harmonization

#### Trajectory-controlled Generation

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

Ours Harmonizer VideoTripletTransformer

Ours Tora Revideo

4.47%

13.95%

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

18.16%

[Figure 318]

[Figure 319]

14.47%

71.58%

77.37%

### (a) (b)

- Figure 12: User study comparison. Our method receives the most user preference for both video harmonization(a) and trajectory-controlled generation(b).

[Figure 320]

- Figure 13: Loss curves of applying ERoPE. We apply ERoPE by extending along different optional dimensions, height, width, and timing. For comparison, we showcase the loss curve of ablation setting that applies the same RoPE on both masked video and foreground video, which is marked in pink. One can see that the three extension directions have equal positive impact on performance.

- E USER STUDY

We conduct user study for both video harmonization and trajectory-controlled video generation, compared with Harmonizer and VideoTripletTransformer, Tora and Revideo, respectively. For each task, we provide 20 sets of visual comparisons and invite 19 professional volunteers to select their most preferred results. As shown in Fig. 12(a), most users prefer the video harmonization capability of our method, while Harmonizer and VideoTripletTransformer are equally favored. As to trajectory-controlled video generation, as shown in Fig. 12(b), our method obviously outperforms other related algorithms, which is consistent with the visual and quantitative results provided in the main paper.

[Figure 321]

fg.tokennoisytoken

[Figure 322]

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

[Figure 327]

[Figure 328]

Patchify

[Figure 329]

[Figure 330]

DiTFusionBlock

DiTFusionBlock

DiTFusionBlock

DiTFusionBlock

VAEEncoder

[Figure 331]

VAEDecoder

ERoPE 2*t*h*w

×…21 ×…21

[Figure 332]

Noise

c

masked video

[Figure 333]

[Figure 334]

[Figure 335]

Patchify

[Figure 336]

[Figure 337]

0

Fg. Generation Mainstream

fg. video

composited video

w/o Background Preservation Branch (BPB)

[Figure 338]

QueryKeyValue

Bg. Preservation Branch

RoPEt*h*w

bg.tokenfg.tokennoisytoken

LinearLinear

Patchify

[Figure 339]

Cross-attention

[Figure 340]

Self-attention

LayerNorm

[Figure 341]

[Figure 342]

###### c

Block

Block

DiT

DiT

……

Gate

mask video

[Figure 343]

[Figure 344]

[Figure 345]

[Figure 346]

[Figure 347]

[Figure 348]

[Figure 349]

Patchify

[Figure 350]

fg. token

VAEEncoder

[Figure 351]

DiTBlock

DiTBlock

DiTBlock

VAEDecoder

DiTBlock

[Figure 352]

[Figure 353]

[Figure 354]

×…21 ×…21

[Figure 355]

Noise

masked video

[Figure 356]

[Figure 357]

[Figure 358]

[Figure 359]

Patchify

0

composited video

fg. video

w/o fusion block

- Figure 14: Architectures of ablation settings. We visualize architecture details of the two ablation settings, “w/o BPBranch” (upper) and “w/o fusion block” (lower). The former deletes the control branch, and the latter uses cross-attention to inject foreground condition instead of the proposed fusion block. Their control signal conversion is the same as full model. So we omit it and mainly showcase model designs.

[Figure 360]

[Figure 361]

[Figure 362]

foreground element inflated mask

manually paste w/o fusion block

[Figure 363]

[Figure 364]

[Figure 365]

[Figure 366]

[Figure 367]

[Figure 368]

fullmodel

[Figure 369]

[Figure 370]

[Figure 371]

消融visual result 0/14/28/48, 1.7

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

w/o fusion block & w/ CLIP fg. encoding

[Figure 384]

[Figure 385]

[Figure 386]

[Figure 387]

- Figure 15: Ablation on foreground CLIP encoding. Based on the ablation setting of “w/o fusion block”, we further replace the VAE encoding of the foreground dynamics with CLIP encoding (i.e., “w/o fusion block & w/ CLIP fg. encoding”) to validate whether semantic feature is more suitable for cross-attention. But results demonstrate that our model cannot utilize such an abstract feature.

- F LOSS CURVE OF APPLYING EROPE

To study the impact of extended position embedding in different dimensions on performance, we visualize the training loss curves of four settings in Fig. 13, where we apply ERoPE by applying position embedding along three dimensions, i.e., height, width, and timing, which are marked as brown curve (w/ ERoPE h), blue curve (w/ ERoPE w), and green curve (w/ ERoPE t), respectively. For comparison, we also attach the loss curve of an ablation setting as pink curve in Fig. 13, which is the training loss of using the same RoPE for both masked and foreground videos (w/o ERoPE). One can see that the training loss curves of three settings that uses ERoPE are all obviously lower than w/o ERoPE. Meanwhile, the loss curves of the three settings are highly consistent with each other, i.e., the brown, blue, and green curves almost overlap. This comparison strongly prove the importance of our proposed ERoPE for utilizing layout-unaligned video conditions, and demonstrate the equality of these three extension directions. In the main paper, we use “w/ ERoPE h”. More importantly, we believe this discovery inspires future work for utilizing layout-unaligned video conditions.

foreground video

foreground video

[Figure 388]

[Figure 389]

[Figure 390]

[Figure 391]

[Figure 392]

[Figure 393]

[Figure 394]

background video

background video

[Figure 395]

[Figure 396]

[Figure 397]

[Figure 398]

[Figure 399]

[Figure 400]

[Figure 401]

- trajectory 1 results

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

- trajectory 2 results trajectory 2

trajectory 1

results

[Figure 413]

[Figure 414]

[Figure 415]

[Figure 416]

[Figure 417]

[Figure 418]

results

[Figure 419]

[Figure 420]

[Figure 421]

[Figure 422]

[Figure 423]

- Figure 16: Generating with different user-provided trajectories. We apply two different trajectories to the same foreground-background video pairs. We can see that, given the same foreground and background videos with different trajectories, GenCompositor could generate different contents that strictly follow their corresponding trajectories.

- G ABLATION ARCHITECTURES

As shown in Fig. 14, we visualize the architecture details of our two ablation settings, “w/o BPBranch” abd “w/o fusion block”, mentioned in the main paper. The former eliminates the background preservation branch, but still concatenates foreground condition with noisy tokens at the spatial-level and utilizes the proposed DiT fusion block to fusing them, combined with the EROPE. Whilst “w/o fusion block” remains the background branch, but replaces the DiT fusion block with normal DiT modules. To inject foreground control, it utilizes cross-attention as shown in the lower of Fig. 14.

In addition, considering that previous cross-attention based methods mainly employ semantic features as external conditions and inject with cross-attention. For the ablation setting of “w/o fusion block” in the lower of Fig. 14, we also replace the VAE encoding for the foreground videos with CLIP encoding to validate whether semantic feature is more suitable for cross-attention than low-level feature of VAE. The corresponding visual results are given in Fig. 15, where we attach the foreground element and inflated mask in the first row. The second row shows a manually paste version as baseline reference, and results of “w/o fusion block” as ablation baseline. In the setting of “w/o fusion block & w/ CLIP fg. encoding”, we use CLIP to encode foreground videos instead of VAE used in “w/o fusion block”. One can see that the main content of background is still preserved, but the foreground conditions cannot be injected, and even the abstract semantic information that could have been inherited by “w/o fusion block” cannot be preserved, directly generating noise. We believe this is because using such a high-level abstract information, such as semantics, to generate low-level and dense visual signals (i.e., video) is difficult. For example, using VAE features of foreground video can directly inject low-level messages of this dynamic condition, such as its shape and texture, and naturally only affects part region in generated video. But using CLIP features is theoretically equal to input an abstract description text, which directly impacts entire video, increasing learning difficulty. Moreover, fusing the abstract features (foreground video) extracted by CLIP with the low-level features (background video) extracted by VAE in a diffusion model is also challenging.

- H INTERACTIVITY

In order to validate the flexible interoperability of the proposed method, we provide additional examples where the user specifies different trajectories and scale factors, respectively. In Fig. 16, given the foreground and background videos, we manually specify two different trajectories to GenCompositor for generation. The added elements exhibit different trajectories, which are highly consistent with the input trajectories. In Fig. 17, we provide three different scale factors in the same example. Obviously, the user-specified factors directly affect our mask videos and control the size of elements in the final results.

- I MORE VISUAL RESULTS

We provide more visual results in Fig. 18 to showcase our superior video compositing capability. GenCompositor enables seamless video elements injection according to the user-given interaction. We visualize the designated trajectories as right dots and lines in the first frame of background videos. One can see that GenCompositor precisely injects foreground elements into the user-given positions and harmonizes their color. On the other hand, the background environmental changes caused by the inserted objects (such as the shadows in

foreground element

- results inflated mask (scale factor = 1.0)

[Figure 424]

[Figure 425]

[Figure 426]

[Figure 427]

[Figure 428]

[Figure 429]

[Figure 430]

[Figure 431]

[Figure 432]

[Figure 433]

[Figure 434]

[Figure 435]

[Figure 436]

[Figure 437]

[Figure 438]

[Figure 439]

background video

[Figure 440]

[Figure 441]

[Figure 442]

[Figure 443]

[Figure 444]

[Figure 445]

[Figure 446]

[Figure 447]

- results inflated mask (scale factor = 2.0)

results inflated mask (scale factor = 0.7)

[Figure 448]

[Figure 449]

[Figure 450]

[Figure 451]

[Figure 452]

[Figure 453]

[Figure 454]

[Figure 455]

- Figure 17: Generating with different user-provided scale factors. Our method produces mask videos that follow user-provided factors and control the size of added elements in final results.

0，17，22，42 0，21，38，48

[Figure 456]

[Figure 457]

[Figure 458]

[Figure 459]

[Figure 460]

[Figure 461]

[Figure 462]

[Figure 463]

6，20，27，48 0，8，24，48

[Figure 464]

[Figure 465]

[Figure 466]

[Figure 467]

[Figure 468]

[Figure 469]

[Figure 470]

[Figure 471]

[Figure 472]

[Figure 473]

[Figure 474]

[Figure 475]

[Figure 476]

[Figure 477]

[Figure 478]

[Figure 479]

[Figure 480]

[Figure 481]

[Figure 482]

[Figure 483]

[Figure 484]

[Figure 485]

[Figure 486]

[Figure 487]

[Figure 488]

[Figure 489]

[Figure 490]

[Figure 491]

[Figure 492]

[Figure 493]

[Figure 494]

[Figure 495]

[Figure 496]

[Figure 497]

(a) (b)

(c) (d)

bg.videoresultsbg.videoresultsfg.videofg.video

[Figure 498]

[Figure 499]

[Figure 500]

[Figure 501]

[Figure 502]

[Figure 503]

[Figure 504]

[Figure 505]

[Figure 506]

[Figure 507]

[Figure 508]

[Figure 509]

- Figure 18: More visual results of GenCompositor. Our method enables seamless video elements injection and user interaction. On the one hand, it precisely injects foreground elements into the user-given position and harmonizes their color. On the other hand, the background environmental changes caused by the inserted objects (such as the shadows in the red box) are also automatically predicted by our generative model, proving the superiority of generative video compositing.

the red box) are also automatically predicted by our generative model, proving the superiority of generative video compositing task.

##### J DISCUSSION

As the first attempt in generative video compositing, this paper proposed a new practical video editing task, provided the first feasible solution, and curated the first usable training dataset. Interestingly, we found that the proposed new task, generative video compositing, naturally supports other related tasks such as video inpainting and video element removal. More importantly, for video compositing itself, the proposed GenCompositor allows users to specify arbitrary trajectories and object sizes changing with timing. It means that users even simulate 3D spatial effects, such as near-large-and-far-small. We provide some examples in supplementary video to demonstrate this function. In addition, some techniques in this paper are novel and have the potential to be applied to other similar application scenarios. For example, the proposed ERoPE can fully utilize layoutunaligned video conditions, and experiments in this paper demonstrate that, for video signals, the optional three extension directions (e.g., width, height, and timing) are equal, as said in Sect. F. We believe this contribution also plays a significant role in other video editing tasks.

[Figure 510]

[Figure 511]

[Figure 512]

[Figure 513]

[Figure 514]

[Figure 515]

[Figure 516]

[Figure 517]

resultsfg.videobg.video

[Figure 518]

[Figure 519]

[Figure 520]

[Figure 521]

[Figure 522]

[Figure 523]

[Figure 524]

[Figure 525]

[Figure 526]

[Figure 527]

[Figure 528]

[Figure 529]

[Figure 530]

[Figure 531]

[Figure 532]

[Figure 533]

[Figure 534]

[Figure 535]

[Figure 536]

[Figure 537]

[Figure 538]

[Figure 539]

[Figure 540]

[Figure 541]

- Figure 19: Our performance in extreme background lightness. For the left, the freighter itself is not illuminated. The composited results shows a freighter illuminated from left to right, with a warm, sunset-like glow, which is consistent with background content. For the right, the added aircraft itself has a high exposure intensity but background is unexposed. In output, our model adaptively adjusts lightness of added element to make the entire result coordinated.

0,19,21,23

[Figure 542]

[Figure 543]

[Figure 544]

[Figure 545]

[Figure 546]

[Figure 547]

[Figure 548]

[Figure 549]

[Figure 550]

[Figure 551]

[Figure 552]

[Figure 553]

[Figure 554]

[Figure 555]

The 1st fg. video

bg. video

The 2nd results

The 1st composited results The 2nd composited results

[Figure 556]

[Figure 557]

[Figure 558]

[Figure 559]

[Figure 560]

[Figure 561]

- Figure 20: Our performance on multiple objects compositing. We showcase the background video to be edited in the first row, and two element videos in the second row. As shown in the third row, on the left, we use red box to highlight the inserted airplane. On the right, we additionally add a seagull; the airplane and the seagull are partially obscured.

Future work. Although this paper enables video injection and realistic interaction, some challenging topics still exist in extreme conditions. For example, we employ gamma correction, a simple yet effective augmentation, to address most common cases well. As shown in Fig. 19, GenCompositor has been able to generate harmonious results by adaptively adjusting lighting and filter effects of foreground elements to match challenging background brightness. But it may not be robust enough for sophisticated extreme background lighting. Meanwhile, the added elements cannot undergo complex occlusion changes with environment. But as the first attempt, GenCompositor actually has been able to generate videos with impressive occlusion effects. As shown in Fig. 20, we add an airplane and a seagull to a single background video, and GenCompositor tackles this case well. We believe more extreme background lighting can be solved by replacing other luminance augmentation methods, and the more challenging occlusion can be achieved by adding depth-aware or 3D priors. Since GenCompositor already works for most scenarios, we leave these issues for future work. We believe a potential research direction in generative video compositing is enabling novel-view foreground prediction and new-oriented compositing, which may be addressed by introducing 3D priors and we leave it in the future work.

##### K INTERACTIVE WEB APP

To ease utilization of the proposed GenCompositor, we realize an interactive web demo in our code, which can be found in supplementary materials. Specifically, our interactive demo is shown in Fig. 21. We firstly list the essential operating tips, and provide some available video samples in the first box. Users can directly specify background and foreground videos from them, or choose not to use them and upload their own materials. In step 1, users segment foreground element from the foreground video. In step 2, users specify the movement trajectory in the background video. Finally, in step 3, users can specify varying rescale parameters, which is defaulted as 0.4, to assign the scales of added element, and compositing final video through our model. Notice that we also provide an extended application in the box in the lower right corner, which can inpaint video or remove dynamic elements from the video. In this function, users do not need to specify foreground video. Our code automatically selects the blank video as foreground, and removes the specified element from the video smoothly.

[Figure 562]

[Figure 563]

[Figure 564]

[Figure 565]

[Figure 566]

[Figure 567]

[Figure 568]

[Figure 569]

[Figure 570]

[Figure 571]

[Figure 572]

[Figure 573]

[Figure 574]

[Figure 575]

[Figure 576]

[Figure 577]

[Figure 578]

[Figure 579]

[Figure 580]

[Figure 581]

[Figure 582]

[Figure 583]

[Figure 584]

[Figure 585]

[Figure 586]

[Figure 587]

[Figure 588]

[Figure 589]

[Figure 590]

[Figure 591]

[Figure 592]

[Figure 593]

[Figure 594]

[Figure 595]

[Figure 596]

[Figure 597]

[Figure 598]

[Figure 599]

[Figure 600]

[Figure 601]

[Figure 602]

[Figure 603]

[Figure 604]

[Figure 605]

[Figure 606]

[Figure 607]

[Figure 608]

[Figure 609]

[Figure 610]

[Figure 611]

[Figure 612]

[Figure 613]

[Figure 614]

[Figure 615]

[Figure 616]

[Figure 617]

[Figure 618]

[Figure 619]

[Figure 620]

[Figure 621]

[Figure 622]

[Figure 623]

[Figure 624]

[Figure 625]

[Figure 626]

[Figure 627]

[Figure 628]

[Figure 629]

[Figure 630]

[Figure 631]

[Figure 632]

[Figure 633]

[Figure 634]

[Figure 635]

[Figure 636]

[Figure 637]

[Figure 638]

[Figure 639]

[Figure 640]

[Figure 641]

[Figure 642]

[Figure 643]

[Figure 644]

[Figure 645]

[Figure 646]

[Figure 647]

[Figure 648]

[Figure 649]

[Figure 650]

[Figure 651]

[Figure 652]

[Figure 653]

[Figure 654]

[Figure 655]

[Figure 656]

[Figure 657]

[Figure 658]

[Figure 659]

[Figure 660]

[Figure 661]

[Figure 662]

|[Figure 663]<br><br>[Figure 664]|
|---|

|[Figure 665]<br><br>[Figure 666]|
|---|

|[Figure 667]<br><br>[Figure 668]|
|---|

[Figure 669]

[Figure 670]

[Figure 671]

[Figure 672]

[Figure 673]

[Figure 674]

[Figure 675]

[Figure 676]

[Figure 677]

[Figure 678]

[Figure 679]

[Figure 680]

[Figure 681]

[Figure 682]

[Figure 683]

[Figure 684]

[Figure 685]

[Figure 686]

[Figure 687]

[Figure 688]

[Figure 689]

[Figure 690]

[Figure 691]

[Figure 692]

[Figure 693]

[Figure 694]

[Figure 695]

[Figure 696]

[Figure 697]

[Figure 698]

[Figure 699]

[Figure 700]

[Figure 701]

[Figure 702]

[Figure 703]

[Figure 704]

[Figure 705]

[Figure 706]

[Figure 707]

[Figure 708]

[Figure 709]

[Figure 710]

[Figure 711]

[Figure 712]

[Figure 713]

[Figure 714]

[Figure 715]

[Figure 716]

[Figure 717]

[Figure 718]

[Figure 719]

[Figure 720]

[Figure 721]

[Figure 722]

[Figure 723]

[Figure 724]

[Figure 725]

[Figure 726]

[Figure 727]

[Figure 728]

[Figure 729]

[Figure 730]

[Figure 731]

[Figure 732]

[Figure 733]

[Figure 734]

[Figure 735]

[Figure 736]

[Figure 737]

[Figure 738]

[Figure 739]

[Figure 740]

[Figure 741]

[Figure 742]

[Figure 743]

[Figure 744]

[Figure 745]

[Figure 746]

[Figure 747]

[Figure 748]

[Figure 749]

[Figure 750]

[Figure 751]

[Figure 752]

[Figure 753]

[Figure 754]

[Figure 755]

[Figure 756]

[Figure 757]

[Figure 758]

[Figure 759]

[Figure 760]

[Figure 761]

[Figure 762]

[Figure 763]

[Figure 764]

[Figure 765]

[Figure 766]

[Figure 767]

[Figure 768]

[Figure 769]

[Figure 770]

[Figure 771]

[Figure 772]

[Figure 773]

[Figure 774]

[Figure 775]

[Figure 776]

[Figure 777]

[Figure 778]

[Figure 779]

[Figure 780]

[Figure 781]

[Figure 782]

[Figure 783]

[Figure 784]

[Figure 785]

[Figure 786]

[Figure 787]

[Figure 788]

[Figure 789]

[Figure 790]

[Figure 791]

[Figure 792]

[Figure 793]

[Figure 794]

[Figure 795]

[Figure 796]

[Figure 797]

[Figure 798]

[Figure 799]

[Figure 800]

[Figure 801]

[Figure 802]

[Figure 803]

[Figure 804]

[Figure 805]

[Figure 806]

[Figure 807]

[Figure 808]

[Figure 809]

[Figure 810]

[Figure 811]

[Figure 812]

[Figure 813]

[Figure 814]

[Figure 815]

[Figure 816]

[Figure 817]

[Figure 818]

[Figure 819]

[Figure 820]

[Figure 821]

[Figure 822]

[Figure 823]

[Figure 824]

[Figure 825]

[Figure 826]

[Figure 827]

[Figure 828]

[Figure 829]

[Figure 830]

[Figure 831]

[Figure 832]

[Figure 833]

[Figure 834]

[Figure 835]

[Figure 836]

[Figure 837]

[Figure 838]

[Figure 839]

[Figure 840]

[Figure 841]

[Figure 842]

[Figure 843]

[Figure 844]

[Figure 845]

[Figure 846]

[Figure 847]

[Figure 848]

[Figure 849]

[Figure 850]

[Figure 851]

[Figure 852]

[Figure 853]

[Figure 854]

[Figure 855]

[Figure 856]

[Figure 857]

[Figure 858]

[Figure 859]

[Figure 860]

[Figure 861]

[Figure 862]

###### Figure 21: Interactive interface of our web app. As an interactive video editing method, we realize an interactive web demo to ease the utilization of GenCompositor. Related code can be found in supplementary materials.

