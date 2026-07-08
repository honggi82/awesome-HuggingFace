# arXiv:2411.11045v1[cs.CV]17Nov2024

## STABLEV2V: Stablizing Shape Consistency in Video-to-Video Editing

Chang Liu1, Rui Li1, Kaidong Zhang1, Yunwei Lan1, Dong Liu1* 1University of Science and Technology of China {lc980413, liruid, richu, ywlan}@mail.ustc.edu.cn, dongeliu@ustc.edu.cn Project page: https://alonzoleeeooo.github.io/StableV2V

“A blue bag floating on the river.”

Text Description: “A turtle swimming on the river.”

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

[Figure 11]

[Figure 12]

StableV2V(Ours)SourceVideoDMTAnyV2V

###### User Instruction: “Make it lego.”

|[Figure 13]<br><br>[Figure 14]<br><br>[Figure 15]<br><br>[Figure 16]|
|---|

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

Reference Image: “A bag floating on the river.”

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

Hand-drawn Sketch: “A starfish floating, at dawn.”

|[Figure 30]<br><br>[Figure 31]<br><br>[Figure 32]<br><br>[Figure 33]|
|---|

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

|[Figure 38]|
|---|

Figure 1. Qualitative comparison (left) and results on different editing tasks by STABLEV2V (right). Herein, we highlight the words that depict the main edited contents and the modalities of external prompts in red and blue, respectively, and present the visualizations of several prompts (i.e., reference image and hand-drawn sketch) at the right-bottom corner of the corresponding first edited frames. Notably, AnyV2V [20] uses the same first edited frames as ours, where both results are highlighted in green and red bounding boxes, respectively.

### Abstract

Recent advancements of generative AI have significantly promoted content creation and editing, where prevailing studies further extend this exciting progress to video editing. In doing so, these studies mainly transfer the inherent motion patterns from the source videos to the edited ones, where results with inferior consistency to user prompts are often observed, due to the lack of particular alignments between the delivered motions and edited contents. To address this limitation, we present a shape-consistent video editing

method, namely StableV2V, in this paper. Our method decomposes the entire editing pipeline into several sequential procedures, where it edits the first video frame, then establishes an alignment between the delivered motions and user prompts, and eventually propagates the edited contents to all other frames based on such alignment. Furthermore, we curate a testing benchmark, namely DAVIS-Edit, for a comprehensive evaluation of video editing, considering various types of prompts and difficulties. Experimental results and analyses illustrate the outperforming performance, visual

consistency, and inference efficiency of our method compared to existing state-of-the-art studies.1

### 1. Introduction

Video editing aims to modify the source video contents according to user demands. With the prosper of diffusion models [17, 33] that demonstrates superior generative capabilities, recent studies have adopted this astonishing technique for video editing, making it possible for end users to interact with various types of external prompts, e.g., text [28, 50], instruction [40, 48], image [10, 30], sketches [24], and etc. They achieve significant success on this topic, bringing video editing to a prominent attractive research direction for the community of visual content generation.

To perform video editing, recent studies manage to transfer the motion patterns from the original video and adapt them to the editing process. In doing so, prevailing methods can be categorized into four main types, i.e., DDIM inversion-, one-shot tuning-, learning-, and firstframe-based methods. Specifically, DDIM inversion-based methods [28, 47] leverage DDIM inversion to store the motion patterns of videos in forms of latent features, which are then injected into the diffusion models when editing, thus enforcing the consistency between edited frames and the original ones. One-shot tuning-based solutions [25, 42] aim to tailor the motion patterns of each video through learning video-specific model weights. These two types of methods, however, often produce results that are inconsistent to the shapes that user prompts require, especially the ones with significant shape differences, e.g., the cases illustrated in Fig. 1. Learning-based methods [30, 50, 52] provide a more general solution for video editing by finetuning temporal-enhanced diffusion models on large-scale video-text datasets [8, 27], but these studies are highly restricted due to their inpainting paradigms. They normally require mask annotations to precisely localize the edited regions, thus becoming tough for users to interact with. Also, the inpainting paradigms limit them to regional editing scenarios, where the applications of global ones (e.g., video style transfer [21]) are neglected. First-frame-based methods [20, 29] offer a more flexible solution for video editing, where this paradigm decomposes video editing into image editing and motion transfer, enabling the potentials to perform both global and local editing with the same solution. Nevertheless, they suffer from similar limitations to the aforementioned studies due to their requirements of DDIM inversion [20] and video-specific tuning [29]. Recently, DMT [47], which proposes a space-time feature loss to constrain the motion consistency, serves as the most rel-

1We open-source our codebase at https://github.com/ AlonzoLeeeooo/StableV2V, and release the model weights and testing benchmark DAVIS-EDIT at https://huggingface.co/ AlonzoLeeeooo/StableV2V and https://huggingface.co/ datasets/AlonzoLeeeooo/DAVIS-Edit, respectively.

evant study to address such misalignment, but even so, inferior condition-following ability and detail loss of backgrounds are often observed in its results like the ones in Fig. 1, where effective paradigm is thus expected to ensure the consistency between delivered motions and user prompts.

Therefore in this paper, we propose STABLEV2V to perform video editing in a shape-consistent manner, with our method built based on the first-frame-based paradigm. In doing so, our method performs video editing with three main components, i.e., Prompted First-frame Editor (PFE), Iterative Shape Aligner (ISA), and Conditional Image-tovideo Generator (CIG). PFE serves as the first-frame image editor that converts external prompts into edited contents, which are then propagated to other frames in later processes to construct the entire edited video. To offer precise guidance that are well aligned with shapes required by user prompts, especially in scenarios that comprise complicated shape differences, we assume that the edited contents share the same motions with the ones of source video. Based on the assumption, we propose ISA, which manages to iteratively propagate the average motions, shapes, and depths from core elements (e.g., main objects) of each original video frame to the edited one, resulting in the simulated optical flow and depth map of all edited frames, along with a shape-guided depth refinement network to further calibrate the obtained depth map and ensure its preciseness. Eventually, we leverage the depth map as an intermediate vehicle to deliver precise motions from the source video, and utilize it to guide the image-to-video generation process of CIG, obtaining the final edited video. Furthermore, we collect a testing benchmark based on DAVIS [31], namely DAVISEDIT, to conduct a comprehensive evaluation for text- and image-based video editing. Experimental results compared to existing state-of-the-art studies demonstrate that STABLEV2V outperforms others from various perspectives, including visual quality, consistency, and inference efficiency.

### 2. Related Works

Video Synthesis. Modeling the high-dimensional distribution of video data is a challenging task for video generation. Early-proposed methods [37] mainly address this problem via Generative Adversarial Network (GAN), but suffering from inferior visual quality and training instability. Recent advancements of diffusion models [17, 33] have greatly promoted the development of various visual generation tasks, e.g., text-to-image and conditional generation [7, 13, 43], where this effective paradigm is also adopted for video generation [39, 44]. Particularly, existing studies leverage various model architectures upon the video modeling task, including U-net [12] and Diffusion Transformer [44]. These studies demonstrate outstanding generative abilities in producing photo-realistic videos with text prompts, and serve as strong foundation models for a wide

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

|[Figure 89]<br><br>[Figure 90]<br><br>[Figure 91]<br><br>[Figure 92]|
|---|

|[Figure 93]<br><br>[Figure 94]<br><br>[Figure 95]<br><br>[Figure 96]<br><br>[Figure 97]<br><br>[Figure 98]<br><br>[Figure 99]<br><br>[Figure 100]| |
|---|---|
| | |

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

Production

Extractor Depth

Hadamard

Compose

Average

Motion

Pasting

Flow

[Figure 113]

[Figure 114]

[Figure 115]

Dilate

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

isfloatingontheriver.bagA“”

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

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

###### Image Segmenter

[Figure 143]

Warping

|[Figure 144]<br><br>[Figure 145]|
|---|

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

|[Figure 159]<br><br>[Figure 160]<br><br>[Figure 161]<br><br>[Figure 162]<br><br>[Figure 163]<br><br>[Figure 164]<br><br>[Figure 165]<br><br>[Figure 166]<br><br>[Figure 167]<br><br>[Figure 168]<br><br>[Figure 169]<br><br>[Figure 170]<br><br>[Figure 171]<br><br>[Figure 172]<br><br>[Figure 173]<br><br>[Figure 174]|
|---|

|[Figure 175]<br><br>[Figure 176]<br><br>[Figure 177]<br><br>[Figure 178]<br><br>[Figure 179]<br><br>[Figure 180]<br><br>[Figure 181]<br><br>[Figure 182]<br><br>[Figure 183]<br><br>[Figure 184]<br><br>[Figure 185]<br><br>[Figure 186]<br><br>[Figure 187]<br><br>[Figure 188]<br><br>[Figure 189]<br><br>[Figure 190]|
|---|

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

ControlNet Blocks

Estimator

Production

Hadamard

Compose

Average

Pasting

Depth

Ctrl-Adapter Blocks

-

-

U-net Blocks

-

-

Motion Modules

Figure 2. Illustration of the overall pipeline of STABLEV2V, with three main components, i.e., Prompted First-frame Editor (PFE), Iterative Shape Aligner (ISA), and Conditional Image-to-video Generator (CIG), whose backgrounds are highlighted in red, yellow, and gray, respectively. Herein, the green bounding boxes refer to the first video frames; the blue bounding boxes represent the k-th optical flow, segmentation mask, and depth map in ISA. For simplicity, we only showcase the k-th to k + 1-th iteration process of ISA in this figure.

range of down-stream applications, e.g., text-to-video generation [46], image-to-video generation [2, 11, 15, 34] as well as video editing [3, 9, 20, 25, 28, 42, 47, 52].

significant shape differences, and the latter presents inferior capability of background preservation, where all issues above motivate STABLEV2V in this paper.

Video Editing. Recently, the research direction of video editing has attracted great attention. In performing this task, conventional works normally introduce external conditions to assist video editing, e.g., optical flow [9], Neural Layered Atlas (NLA) [4, 22], and etc., where limitations are usually observed due to the inherent problems of the used techniques. With the prosper of diffusion models, such task is significantly facilitated by their strong generative abilities, where we summarize existing methods into four categories, i.e., DDIM inversion-, one-shot tuning-, learning-, and firstframe-based methods. Specifically, DDIM inversion-based methods offer a way to represent the motion patterns of videos through inverted latent features, where these features are then utilized to enforce the temporal consistency in the generated video frames [28]. One-shot tuning-based methods [25, 42] mainly learn video-specific model weights to model the motion patterns, where diversified results can be then generated through adjusting the text prompts. Learning-based methods [30, 50, 52] solve the task via training particular networks on large datasets, where they integrate motion modules into pre-trained image diffusion models [5, 33], and optimize the enhanced model architectures with video-text data, enabling these networks to edit video contents in local regions. First-frame-based methods [20, 29] start with editing the first video frame, and propagate the results to all other frames through transferring the motions from the source video. Nevertheless, these studies obtain inferior performance since their delivered motions are inconsistent with user prompts. AnyV2V [20] and DMT [47] are the most relevant studies to our method. However, the former struggles to handle challenging scenarios with

### 3. Methods

STABLEV2V comprises three main components to perform video editing, i.e., Prompted First-frame Editor (PFE), Iterative Shape Aligner (ISA), and Conditional Image-to-video Generator (CIG), where the overall pipeline is shown in Fig.

- 2. Given an input video X = {X1,...,XN} with N video

frames in total, PFE edits the first video frame X1 into X1 according to an external prompt P. Then, ISA extracts the depth maps D, optical flows F, and segmentation masks M from X, and simulates the depth maps Dr of edited video based on D, F, M, and M1 of X1. Eventually, CIG serves as a depth-guided image-to-video generator, and leverages Dr and X1 to produce the entire edited video X, where the overall process of STABLEV2V is formulated by:

X = fCIG fPFE (X1, P) , fISA D, F, M, M1 , (1)

where fPFE (·), fISA (·), and fCIG (·) denote PFE, ISA, and CIG, respectively. In the following texts, we illustrate the details of each aforementioned component following the pipeline sequence of STABLEV2V.

- 3.1. Prompted First-frame Editor

Since STABLEV2V is built based on first-frame-based methods that decompose video editing into image editing and controlled image-to-video generation, the first step of STABLEV2V is to convert the external prompt into edited contents in the first video frame, with PFE serving as the core component in this step. Given an input video X = {X1,...,XN}, we send its first frame X1 and the external prompt P into PFE, where we formulate this process by:

X1 = fPFE (X1,P), (2) where X1 refers to the first edited video frame of X. Herein, we consider various categories of prompt inputs P, e.g., text descriptions, user instructions, reference images, and etc., where we adopt off-the-shelf image editors to process these prompts accordingly. For example, we utilize textguided editors, e.g., SD Inpaint [33] and InstructPix2Pix [36], to process text inputs, and adopt models like Paintby-Example [5] to integrate reference image prompts. Afterward in the subsequent processes, we build the alignment between motion controls and edited contents based on X1.

##### 3.2. Iterative Shape Aligner

Once we obtain the first edited frame X1, the next step is to propagate the edited contents to the remaining video frames. To conduct this step, we observe that existing studies often produce inferior results through directly propagation of motions from the source video, where the delivered motions in such case struggle to be consistent with contents that users expect, especially in the cases that user prompts may cause significant shape changes, as is shown in Fig. 1, thereby leading to artifacts in the edited video. Therefore, it is pivotal to propose an effective design to address such misalignment, so as to ensure the consistency in video editing.

In doing so, we propose ISA, which establishes the alignment between delivered motions and user prompts, and later offers precise guidance for CIG to produce the final video. Specifically, we assume that the edited and original contents share the same motion and depth information, and consider depth map as the intermediate media to deliver the motion information. Based on the assumption, ISA sequentially simulates the motion and depth information of all edited video frames, and leverages an additional refinement network to obtain precise motion guidance for CIG.

Motion Simulation. To simulate the motion information of the edited video, we use optical flows to represent its motions. Given the source video input X = {X1,...,XN} with N frames, we utilize an off-the-shelf flow extractor (i.e., RAFT [35]) to annotate the optical flows F = {F1→2,...,FN−1→N} from X. Besides, we use an image segmenter (i.e., SAM [19]) to obtain the segmentation masks of all frames in X, as well as the one of X1, resulting in M = {M1,...,MN} and M1, respectively. Considering that the edited contents and the original ones share the same motion information, we firstly compute the mean value of the k-th optical flow Fk→k+1 within Mk to represent the average motion, with the process formulated by:

1 Mk

F¯k→k+1 =

Fk→k+1 (i,j), (3)

(i,j)∈Mk

where (i,j) represents the pixel at the i-th row and the jth column of Mk. Then, we simulate the flow within the regions of edited contents through performing the motion pasting operation on Mk, where it is written as:

“A grape floating on the river.”

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

Source

Video Input

|[Figure 225]|
|---|

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

DepthMap Simulated

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

DepthMap

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

DepthMap

Refined

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

Edited

Video

Figure 3. Visualizations of the intermediate results by ISA.

F ¯k→k+1, (x,y) ∈ fd Mk 0, otherwise

Fkmp (x,y) =

(4)

Finally, we obtain Fk→k+1 of the k-th edited frame via:

Fk→k+1 = Fk→k+1 ⊙ 1 − fd Mk + Fkmp. (5)

Herein, fd (·) and ⊙ refer to the binary dilation and the Hadamard production operations, respectively, where we

apply them on Mk to ensure that the simulated motion covers all regions of the edited contents. Once Fk→k+1 is simulated, we obtain Mk+1 via warping Mk, written as:

Mk+1 = fw Mk, Fk→k+1 , (6)

where fw (·) denotes the warping operation. By iteratively simulating the optical flows from k = 1 to k = N − 1, we eventually obtain the optical flows F = { F1→2,..., FN−1→N} of all edited frames.

Depth Simulation. Once we simulate the motion information of the edited video, the next step is to obtain the guidance for the image-to-video generator, i.e., the depth maps. In doing so, we conduct procedures similar to that in motion simulation. Specifically, we firstly adopt an offthe-shelf depth estimator (i.e., MiDaS [32]) to extract the depth maps D = {D1,...,DN} from X. Given the k-th (k ∼ {1...N}) depth map Dk, we compute the average depth similar to the process of Eq. (3), formulated by:

1 Mk

D¯k =

Dk (i,j), (7)

(i,j)∈Mk

where (i,j) represents the pixel at the i-th row and j-th column. Then, we conduct the depth pasting operation on Mk to propagate the depth information, where the average depth

###### DAVIS-EDIT-S / DAVIS-EDIT-C(∆=|C−S|) Method DOVER↑ FVD↓ WE↓ CLIP-Temporal↑ CLIP Score↑ T¯↓

TokenFlow [28] 66.36 / 67.47(1.11) 17.33 / 17.45(0.12) 18.58 / 18.60(0.02) 95.84 / 95.61(0.23) 24.89 / 24.12(0.77) 5.81 FLATTEN [9] 63.86 / 61.18(2.68) 19.17 / 21.65(2.48) 17.29 / 17.75(0.46) 95.39 / 94.51(0.88) 24.07 / 23.24(0.83) 4.23 Tune-A-Video [42] 28.54 / 34.63(6.09) 25.89 / 26.76(0.87) 89.63 / 81.44(8.19) 91.82 / 90.91(0.91) 24.67 / 24.89(0.22) 20.23 Video-P2P [25] 55.10 / 51.22(3.88) 17.22 / 17.87(0.65) 19.95 / 18.82(1.13) 94.37 / 93.51(0.86) 24.72 / 24.11(0.61) 21.17 CoCoCo [52] 66.81 / 66.12(0.69) 18.13 / 18.41(0.28) 16.24 / 18.47(2.23) 96.07 / 94.97(1.10) 24.36 / 23.24(1.12) 1.55 AnyV2V [20] 66.82 / 65.01(1.72) 14.87 / 17.83(2.96) 15.35 / 18.26(2.91) 95.66 / 94.36(1.30) 25.09 / 24.32(0.77) 8.28 DMT [47] 59.27 / 57.45(1.82) 19.53 / 21.64(2.11) 16.65 / 19.89(3.24) 94.11 / 93.58(0.53) 24.91 / 24.51(0.40) 8.88 STABLEV2V 67.78 / 70.80(3.02) 13.77 / 17.18(3.41) 15.95 / 15.27(0.68) 96.34 / 96.83(0.49) 25.46 / 25.68(0.22) 3.14

AnyV2V [20] 65.83 / 64.56(1.27) 12.97 / 15.25(2.28) 24.47 / 25.61(1.14) 95.89 / 96.13(0.24) 25.41 / 24.79(0.62) 8.43 STABLEV2V 67.58 / 68.42(0.84) 12.36 / 14.87(2.51) 22.17 / 21.23(0.94) 96.51 / 96.71(0.20) 26.24 / 26.55(0.31) 3.23

- Table 1. Quantitative results of STABLEV2V on text- (top) and image-based (bottom) evaluation settings of DAVIS-EDIT, compared to existing methods [9, 20, 25, 28, 42, 47, 52] with respect to DOVER [41], FVD [38], Warping Error (WE), CLIP-Temporal [30], CLIP scores [16], and averaged inference time (termed T¯, in units of minutes), where the best and second best results are boldfaced and underlined. Results on DOVER, FVD, WE, CLIP-Temporal, and CLIP scores are scaled by 10−2, 102, 10−5, 10−2, and 10−2, respectively. Herein, performance gain and drop by comparing DAVIS-EDIT-C to DAVIS-EDIT-S are highlighted in blue and red, correspondingly.

##### 3.3. Conditional Image-to-video Generator

Dkdp (x,y) = D¯k if (x,y) ∈ Mk otherwise 0. Finally, we construct the k-th simulated depth map Dk via composing:

Once we obtain Dr, the final goal of CIG is to generate the edited video X. Specifically, CIG consists of two components, i.e., the controller model and the image-to-video generator, where we use Ctrl-Adapter [23] as a controller to inject Dr, and leverage I2VGen-XL [34] to propagate the edited contents from X1 to all other frames in X, respectively. Given the corresponding text prompt Pt and Dr, CIG produces the final edited video X through:

Dk = Dk ⊙ 1 − Mk + Dkdp. (8)

By iterating all depth maps D = {D1,...,DN−1}, we are able to obtain the simulated depth map D = { D1,..., DN} of all edited video frames. Since the simulated depth maps

- D are obtained via composing, we observe that D often contains unnecessary depth information in the regions of the original contents, as is shown in Fig. 3, indicating that D needs to be further refined to ensure its preciseness. Shape-guided Depth Refinement. To refine D, we draw inspirations from existing video inpainting methods [51] that adopt completion networks to repair optical flows, and propose a depth refinement network based on such paradigm.2 Furthermore, we integrate the first-frame shape mask M1 into it to ensure the shape consistency of refinement. Given M and M, the mask regions Mr and the masked depth maps Dm are obtained through:

X = { X1,..., XN} = fCIG X1,Pt,Ec Dr . (11)

### 4. Experimental Settings

In this section, we illustrate our experimental settings from aspects of evaluation setup, testing benchmark, baselines, and metrics, whose details are presented as follows.

Evaluation Setup. In our experiments, we summarize and evaluate existing video editing studies based on two mainstream setups, i.e., text- and image-based evaluation. For text-based evaluation, we adopt captions with only their object words modified to generate the edited videos. For image-based evaluation, we utilize reference images as external prompts to produce the edited videos.

Mr = fd 1 − M ⊙ M , Dm = Mr ⊙ D.

(9)

Testing Benchmark. For evaluation, we construct a testing benchmark, namely DAVIS-EDIT, based on DAVIS [31]. DAVIS-EDIT contains two subsets DAVIS-EDIT-S and DAVIS-EDIT-C, which address the scenarios with similar (S) and changing (C) shapes, respectively. Specifically, we select 26 videos from DAVIS, and annotate the captions and images for them, obtaining 100 cases eventually.3

Then, we send the concatenation of Dm, Mr, and M1 into the shape-guided refinement network fr (·), resulting in the final depth maps Dr, where the process is written as:

Dr = fr Dm,Mr, M1 . (10)

In this way, ISA is able to obtain the accurately simulated depth maps Dr of the edited video, where Dr later play a pivotal role in offering precise guidance for CIG.

Baselines. We compare STABLEV2V with several state-ofthe-art video editing methods, including TokenFlow [28],

2We illustrate the implementation details of the shape-guided depth refinement network in Sec. A of our supplementary materials.

3We illustrate more details of the proposed testing benchmark DAVISEDIT in Sec. B of our supplementary materials.

[Figure 266]

“A rooster walking on the grass.” “A ball floating on a pond.” “A tank driving up a hill.”

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

SourceVideoStableV2VFLATTENAnyV2VCoCoCoTune-A-VideoVideo-P2PTokenflow

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

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

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

[Figure 334]

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

[Figure 341]

[Figure 342]

[Figure 343]

[Figure 344]

[Figure 345]

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

|[Figure 365]<br><br>[Figure 366]<br><br>[Figure 367]<br><br>[Figure 368]|
|---|

|[Figure 369]<br><br>[Figure 370]<br><br>[Figure 371]<br><br>[Figure 372]|
|---|

|[Figure 373]<br><br>[Figure 374]<br><br>[Figure 375]<br><br>[Figure 376]<br><br>[Figure 377]|
|---|

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

SourceVideoStableV2VAnyV2VDMT

|[Figure 393]<br><br>[Figure 394]<br><br>[Figure 395]<br><br>[Figure 396]<br><br>[Figure 397]<br><br>[Figure 398]<br><br>[Figure 399]<br><br>[Figure 400]|
|---|

|[Figure 401]<br><br>[Figure 402]<br><br>[Figure 403]<br><br>[Figure 404]|
|---|

|[Figure 405]<br><br>[Figure 406]<br><br>[Figure 407]<br><br>[Figure 408]<br><br>[Figure 409]|
|---|

[Figure 410]

“A blackswan walking on the grass.”

“A sport car driving on the night road.”

“A motorboat surfing on the ocean.”

[Figure 411]

[Figure 412]

[Figure 413]

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

[Figure 426]

[Figure 427]

[Figure 428]

[Figure 429]

[Figure 430]

[Figure 431]

|[Figure 432]<br><br>[Figure 433]<br><br>[Figure 434]<br><br>[Figure 435]<br><br>[Figure 436]<br><br>[Figure 437]<br><br>[Figure 438]<br><br>[Figure 439]|
|---|

|[Figure 440]<br><br>[Figure 441]<br><br>[Figure 442]<br><br>[Figure 443]<br><br>[Figure 444]<br><br>[Figure 445]<br><br>[Figure 446]<br><br>[Figure 447]|
|---|

|[Figure 448]<br><br>[Figure 449]<br><br>[Figure 450]<br><br>[Figure 451]|
|---|

|[Figure 452]<br><br>[Figure 453]<br><br>[Figure 454]<br><br>[Figure 455]<br><br>[Figure 456]<br><br>[Figure 457]<br><br>[Figure 458]<br><br>[Figure 459]|
|---|

|[Figure 460]<br><br>[Figure 461]<br><br>[Figure 462]<br><br>[Figure 463]<br><br>[Figure 464]<br><br>[Figure 465]<br><br>[Figure 466]<br><br>[Figure 467]<br><br>[Figure 468]<br><br>[Figure 469]<br><br>[Figure 470]<br><br>[Figure 471]<br><br>[Figure 472]<br><br>[Figure 473]<br><br>[Figure 474]<br><br>[Figure 475]|
|---|

|[Figure 476]<br><br>[Figure 477]<br><br>[Figure 478]<br><br>[Figure 479]|
|---|

Figure 4. Qualitative comparison of text- and image-based editing, with their backgrounds highlighted in green and yellow, respectively. Note that results of AnyV2V [20] (green bounding boxes) use the same first edited frames as ours (red bounding boxes).

FLATTEN [9], Tune-A-Video [42], Video-P2P [25], CoCoCo [52], AnyV2V [20], and DMT [47]. Notably, we use the same first edited frames in comparison with other first-frame-based methods such as AnyV2V.4

Metrics. We evaluate all compared methods from four aspects, i.e., visual quality, temporal consistency, alignment,

4Since we have no access to AVID [50], VASE [30], and I2VEdit [29], we qualitatively compare STABLEV2V with them based on their demo videos, with details presented in Sec. C of our supplementary materials.

and efficiency. For visual quality, we utilize DOVER [41] and FVD [38] for evaluation. For temporal consistency, we compute the Warping Error (WE) of adjacent frames in the edited video, and adopt CLIP-Temporal following VASE [30]. For alignment, we leverage CLIP score [16] to measure the feature similarities of generated frames with the text prompts. For efficiency, we evaluate based on averaged inference time, where results are tested on the same A100 GPU with torch.float16 precision. Besides, we con-

###### Method D.-E.-S D.-E.-C Avg.

TokenFlow [28] 14.71% 7.49% 10.92% FLATTEN [9] 3.53% 1.60% 2.52% Tune-A-Video [42] 0.00% 5.88% 3.08% Video-P2P [25] 7.65% 2.14% 4.77% CoCoCo [52] 10.58% 8.56% 9.52% AnyV2V [20] 17.06% 23.53% 20.45% DMT [47] 21.18% 23.53% 22.41% STABLEV2V 25.29% 27.27% 26.33%

- Table 2. Human evaluation results on DAVIS-EDIT-S (“D.E.-S”) and DAVIS-EDIT-C (“D.-E.-C”). duct user study to analyze with human evaluation.56

### 5. Results and Applications

Performance Comparison and Human Evaluation. Tab. 1, Fig. 4, and Tab. 2 report the quantitative, qualitative comparisons, and human evaluation on DAVIS-EDIT, respectively, compared to several existing methods [9, 20, 25, 28, 42, 47, 52]. Specifically, TokenFlow [28] and FLATTEN [9] produce videos that are inconsistent with user prompts, and obtain inferior performance on most metrics, proving our motivation to address the shape inconsistency issue. Similar trends are observed in Tune-A-Video [42] and Video-P2P [25], with the video quality severely deteriorated, due to their incapabilities of modeling consistent motions with user prompts. Although CoCoCo [52] and AnyV2V [20] improve the aforementioned methods to some extents, they struggle to handle challenging cases with significant shape change, especially when AnyV2V uses the same edited frame as ours, suggesting the deficiencies in these methods. DMT [47] is the most related study to ours, where it fails to follow the edited text prompts in some scenarios, and tends to produce contents with information loss in the backgrounds. STABLEV2V consistently outperforms others with promising performance and video quality, where its results are also overwhelmingly preferred by users. Notably, we observe that most methods obtain worse performance on DAVIS-EDIT-C, whose cases comprise more complicated shape changes and are thus more challenging, however, STABLEV2V still obtains promising results and even gets improvements, owing to the fact that it ensures the consistency between the delivered motions and

- user prompts, thus will not be confused by misaligned motions when producing the final videos as others do.7

5In this paper, “D.”, “WE”, “C.-T”, and “C.S.” denote the abbreviations of DOVER [41], Warping Error, CLIP-Temporal [30], and CLIP score [1] unless otherwise specified. Besides, DOVER, FVD, WE, CLIP-Temporal, and CLIP scores are scaled by 10−2, 102, 10−5, 10−2, and 10−2.

- 6We recruit 17 users, and show them with the inputs, prompts, and re-

sults, with 10 and 11 cases from DAVIS-EDIT-S and DAVIS-EDIT-C, respectively. Each user is asked to choose the videos with best quality without knowing the corresponding methods. Then, we compute the averaged top-1 preference percentage of all cases for comparison.

- 7We present more results in Sec. C of our supplementary materials.

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

Instruction-based Editing: “Make it minecraft.”

[Figure 490]

[Figure 491]

[Figure 492]

[Figure 493]

[Figure 494]

[Figure 495]

[Figure 496]

[Figure 497]

[Figure 498]

[Figure 499]

Sketch-based Editing: “An elephant walking on the rocks.”

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

[Figure 510]

Video Style Transfer: “A bear walking, Van Gogh style.”

[Figure 511]

[Figure 512]

[Figure 513]

[Figure 514]

[Figure 515]

[Figure 516]

[Figure 517]

[Figure 518]

[Figure 519]

[Figure 520]

Video Inpainting: “A scenery of rocks.”

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

Figure 5. More applications performed by STABLEV2V, where the source video frames are shown in the first row.

Efficiency Comparison. In our experiments, we observe that STABLEV2V demonstrates outstanding efficiency compared to other methods, as is reported in Tab. 1. One can see that one-shot tuning-based methods [25, 42] take the most time (more than 20 minutes) to edit a video due to their requirements of video-specific training, but the corresponding performance is not satisfying. DDIM inversion-based methods [20, 28, 47] also require massive time (around 6 to 8 minutes) to perform an complete editing process, where they need to prepare CNN features and attention maps via the inversion process. FLATTEN [9] presents as an improved method that uses more efficient strategy to sample trajectories of optical flows, where STABLEV2V surpasses it with approximate 1.09 minutes. Eventually, CoCoCo serves as the best method in the comparison, however, it is worth noting that it also needs to train on Web10M [27] for one epoch in advance, while STABLEV2V plays as a training-free solution for video editing. Applications. Despite of the aforementioned results, STABLEV2V also support other applications as is demonstrated in Fig. 5. Herein, we adjust PFE according to the conducted application, where STABLEV2V consistently handles different tasks, especially the ones that are susceptible to cause shape differences (e.g., instructions and sketches). Notably in Fig. 1 and 5, sketch-based editing offers a way for users to customize the shapes of edited contents, indicating the great potentials of applying STABLEV2V for realworld cases. Notably, video inpainting represents an extreme scenario of shape differences in STABLEV2V, with

“An elephant walking in the zoo at dawn.”

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

SD+Con.(scribble)SD+Con.(Canny)SD+Con.(depth)SourceVideoSD

[Figure 541]

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

[Figure 556]

[Figure 557]

[Figure 558]

[Figure 559]

[Figure 560]

Figure 6. Text-guided results under different settings of PFE.

Method D.↑ FVD↓ WE↓ C.-T↑ C.S.↑ SD 46.03 21.06 17.69 92.22 19.72 SD + Con. (Canny) 61.16 19.90 16.67 94.24 21.55 SD + Con. (scribble) 64.08 14.70 16.69 95.66 24.75 SD + Con. (depth) 67.78 13.77 15.95 96.34 25.46

- Table 3. Evaluation scores under different settings of PFE, evaluated on text-based editing of DAVIS-EDIT-S. the foreground object completely removed from the source video. Particularly in ISA, M becomes all-zero maps since there is no foreground, and the pasting processes are subsequently skipped, where the shape-guided depth refinement

network fr (·) in such case aims to fully remove D and obtains depth maps of backgrounds to guide CIG.

### 6. Ablation Studies

To further analyze STABLEV2V, we ablate its different components through conducting experiments under different settings of PFE and the depth simulation strategies, where details are presented in the following texts.

Effect of PFE on Text-based Editing. We evaluate the effect of PFE using various types of text-guided editors, with the corresponding results shown in Tab. 3 and Fig. 6. Specifically, we use “SD” and “SD + Con.”, referring to the SD inpaint model [33] and the integrated framework that

- uses the ControlNet [26] to guide the inpainting process with conditions from the source video, respectively, where the condition types are illustrated in the parentheses. We observe that “SD” often produces unstable edited contents like the ones in Fig. 6, which later misguides the image-tovideo generator, and produces video with inferior quality. Using conditions significantly improves such limitation by

“A lamborghini driving on the road.”

[Figure 561]

[Figure 562]

[Figure 563]

[Figure 564]

[Figure 565]

withSourceVideoWarpingUsingUsingUsing

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

Figure 7. Results under different depth simulation strategies.

Depth Simulation D.↑ FVD↓ WE↓ C.-T↑ C.S.↑ Using D 62.00 22.93 17.25 94.73 22.55 Using D 66.46 16.62 16.36 95.94 24.55 Warping D1 with F 64.54 19.14 16.83 95.33 23.71 Using Dr (Ours) 67.78 13.77 15.95 96.34 25.46

Table 4. Evaluation scores under different depth simulation strategies, evaluated on text-based editing of DAVIS-EDIT-S.

enforcing the consistency, however, artifacts are observed due to the over-control by some conditions like Canny edge [6], with this situation alleviated in “SD + Con. (scribble)” and “SD + Con. (depth)” to some extents. This experiments highlight the vitalness of the first edited frame, which offers superior flexibility on one hand, while on the other hand, it also determines how subsequent processes perform.

Effect of the Depth Simulation Strategies. In STABLEV2V, depth map plays a vital role in transporting motions and guiding CIG, where we explore its effects via different simulation strategies, as is reported in Tab. 4 and Fig. 7. Directly using D of source video suffers from issues similar to existing studies, where such depth maps misalign with the user prompts, so that incorrect motions are used for editing, thus leading to artifacts in results of CIG. Similar results are shown when using D (w/o depth refinement), since D contain redundant regions like the ones in Fig. 3, indicating that depth refinement significantly boosts the accuracy of CIG guidance, thus ensuring that the edited video is consistent with user prompts. Warping-based solution produces results with varying shapes due to the lack of motion pasting, where F fail to fully cover D1, especially when edited objects comprise larger sizes than the original ones, e.g., the case of editing a black swan to a bag in Fig. 1.

### 7. Conclusion

In this work, we present STABLEV2V, a shape-consistent video editing method that sequentially edits the first video frame, aligns the motions with user prompts, and finally produces the edited video with such consistent motions, with superior performance demonstrated on challenging applications. Even so, STABLEV2V comprises several limitations due to the intrinsic problems of its paradigm, especially leading to potential working boundaries in cases with complicated motion patterns. In our future work, we expect to address such issue, and propose an improved paradigm with more fine-grained motion modeling for video editing.8

### References

- [1] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning Transferable Visual Models From Natural Language Supervision. In ICML, pages 8748–8763, 2021. 7
- [2] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, Varun Jampani, and Robin Rombach. Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets. arXiv, 2023. 3
- [3] Jianhong Bai, Tianyu He, Yuchi Wang, Junliang Guo, Haoji Hu, Zuozhu Liu, and Jiang Bian. UniEdit: A Unified TuningFree Framework for Video Motion and Appearance Editing. arXiv, 2024. 3
- [4] Omer Bar-Tal, Dolev Ofri-Amar, Rafail Fridman, Yoni Kasten, and Tali Dekel. Text2LIVE: Text-Driven Layered Image and Video Editing. In ECCV, pages 707–723, 2022. 3
- [5] Binxin Yang, Shuyang Gu, Bo Zhang, Ting Zhang, Xuejin Chen, Xiaoyan Sun, Dong Chen, and Fang Wen. Paint by Example: Exemplar-based Image Editing with Diffusion Models. In CVPR, pages 18381–18391, 2023. 3, 4, 14
- [6] John Canny. A Computational Approach to Edge Detection. TPAMI, (6):679–698, 1986. 8
- [7] Chang Liu, Rui Li, Kaidong Zhang, Xin Luo, and Dong Liu. LaCon: Late-Constraint Diffusion for Steerable Guided Image Synthesis. arXiv, 2024. 2
- [8] Tsai-Shien Chen, Aliaksandr Siarohin, Willi Menapace, Ekaterina Deyneka, Hsiang-wei Chao, Byung Eun Jeon, Yuwei Fang, Hsin-Ying Lee, Jian Ren, Ming-Hsuan Yang, and Sergey Tulyakov. Panda-70M: Captioning 70M Videos with Multiple Cross-Modality Teachers. In CVPR, pages 13320–13331, 2024. 2
- [9] Yuren Cong, Mengmeng Xu, Christian Simon, Shoufa Chen, Jiawei Ren, Yanping Xie, Juan-Manuel P´erez-R´ua, Bodo Rosenhahn, Tao Xiang, and Sen He. FLATTEN: Optical FLow-guided ATTENtion for Consistent Text-to-Video Editing. In ICLR, 2024. 3, 5, 6, 7

8We analyze and discuss the limitations of our proposed method in Sec.

- E of our supplementary materials.

- [10] Guangxuan Xiao, Tianwei Yin, William T. Freeman, Fr´edo Durand, and Song Han. FastComposer: Tuning-Free MultiSubject Image Generation with Localized Attention. arXiv,

2023. 2

- [11] Xun Guo, Mingwu Zheng, Liang Hou, Yuan Gao, Yufan Deng, Pengfei Wan, Di Zhang, Yufan Liu, Weiming Hu, Zhengjun Zha, Haibin Huang, and Chongyang Ma. I2VAdapter: A General Image-to-Video Adapter for Diffusion Models. In SIGGRAPH, page 112, 2024. 3
- [12] Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, and Bo Dai. AnimateDiff: Animate Your Personalized Text-toImage Diffusion Models without Specific Tuning. In ICLR,

2024. 2

- [13] Gwanghyun Kim, Taesung Kwon, and Jong Chul Ye. DiffusionCLIP: Text-Guided Diffusion Models for Robust Image Manipulation. In CVPR, pages 2416–2425, 2022. 2
- [14] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-Rank Adaptation of Large Language Models. In ICLR, 2022. 14
- [15] Li Hu. Animate Anyone: Consistent and Controllable Image-to-Video Synthesis for Character Animation. In CVPR, pages 8153–8163, 2024. 3
- [16] Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. CLIPScore: A Reference-free Evaluation Metric for Image Captioning. In EMNLP, pages 7514–7528,

2021. 5, 6

- [17] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising Diffusion Probabilistic Models. In NeurIPS, 2020. 2
- [18] Diederik P. Kingma and Jimmy Ba. Adam: A Method for Stochastic Optimization. In ICLR, 2015. 12
- [19] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chlo´e Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Doll´ar, and Ross B. Girshick. Segment Anything. In ICCV, pages 3992– 4003, 2023. 4
- [20] Max Ku, Cong Wei, Weiming Ren, Harry Yang, and Wenhu Chen. AnyV2V: A Tuning-Free Framework For Any Videoto-Video Editing Tasks. arXiv, 2024. 1, 2, 3, 5, 6, 7, 15
- [21] Wei-Sheng Lai, Jia-Bin Huang, Oliver Wang, Eli Shechtman, Ersin Yumer, and Ming-Hsuan Yang. Learning Blind Video Temporal Consistency. In ECCV, pages 179–195, 2018. 2
- [22] Yao-Chih Lee, Ji-Ze Genevieve Jang, Yi-Ting Chen, Elizabeth Qiu, and Jia-Bin Huang. Shape-Aware Text-Driven Layered Video Editing. In CVPR, pages 14317–14326, 2023. 3
- [23] Han Lin, Jaemin Cho, Abhay Zala, and Mohit Bansal. CtrlAdapter: An Efficient and Versatile Framework for Adapting Diverse Controls to Any Diffusion Model. arXiv, 2024. 5, 15
- [24] Chang Liu, Shunxin Xu, Jialun Peng, Kaidong Zhang, and Dong Liu. Towards interactive image inpainting via robust sketch refinement. TMM, pages 9973–9987, 2024. 2
- [25] Shaoteng Liu, Yuechen Zhang, Wenbo Li, Zhe Lin, and Jiaya Jia. Video-P2P: Video Editing with Cross-Attention Control. In CVPR, pages 8599–8608, 2024. 2, 3, 5, 6, 7

- [26] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding Conditional Control to Text-to-Image Diffusion Models. In ICCV, pages 3813–3824, 2023. 8
- [27] Max Bain, Arsha Nagrani, G¨ul Varol, and Andrew Zisserman. Frozen in Time: A Joint Video and Image Encoder for End-to-End Retrieval. In ICCV, pages 1708–1718, 2021. 2, 7
- [28] Michal Geyer, Omer Bar-Tal, Shai Bagon, and Tali Dekel. TokenFlow: Consistent Diffusion Features for Consistent Video Editing. In ICLR, pages 1–13, 2024. 2, 3, 5, 7
- [29] Wenqi Ouyang, Yi Dong, Lei Yang, Jianlou Si, and Xingang Pan. I2VEdit: First-Frame-Guided Video Editing via Imageto-Video Diffusion Models. arXiv, 2024. 2, 3, 6, 11, 12, 13, 14, 15
- [30] Elia Peruzzo, Vidit Goel, Dejia Xu, Xingqian Xu, Yifan Jiang, Zhangyang Wang, Humphrey Shi, and Nicu Sebe. VASE: Object-Centric Appearance and Shape Manipulation of Real Videos. arXiv, 2024. 2, 3, 5, 6, 7, 11, 12, 13, 14
- [31] Jordi Pont-Tuset, Federico Perazzi, Sergi Caelles, Pablo Arbel´aez, Alex Sorkine-Hornung, and Luc Van Gool. The 2017 DAVIS Challenge on Video Object Segmentation. arXiv,

2018. 2, 5

- [32] Rene Ranftl, Katrin Lasinger, David Hafner, Konrad Schindler, and Vladlen Koltun. Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-Shot CrossDataset Transfer. TPAMI, 44(03):1623–1637, 2022. 4, 12
- [33] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-Resolution Image Synthesis with Latent Diffusion Models. In CVPR, pages 10674–10685, 2022. 2, 3, 4, 8, 13, 15
- [34] Shiwei Zhang, Jiayu Wang, Yingya Zhang, Kang Zhao, Hangjie Yuan, Zhiwu Qin, Xiang Wang, Deli Zhao, and Jingren Zhou. I2VGen-XL: High-Quality Image-to-Video Synthesis via Cascaded Diffusion Models. arXiv, 2023. 3, 5, 15
- [35] Zachary Teed and Jia Deng. RAFT: Recurrent All-Pairs Field Transforms for Optical Flow. In ECCV, pages 402–419,

2020. 4

- [36] Tim Brooks, Aleksander Holynski, and Alexei A. Efros. InstructPix2Pix: Learning to Follow Image Editing Instructions. In CVPR, pages 18392–18402, 2023. 4
- [37] Sergey Tulyakov, Ming-Yu Liu, Xiaodong Yang, and Jan Kautz. MoCoGAN: Decomposing Motion and Content for Video Generation. In CVPR, pages 1526–1535, 2018. 2
- [38] Thomas Unterthiner, Sjoerd van Steenkiste, Karol Kurach, Rapha¨el Marinier, Marcin Michalski, and Sylvain Gelly. FVD: A New Metric for Video Generation. In ICLR Workshop, 2019. 5, 6
- [39] Xiang Wang, Shiwei Zhang, Hangjie Yuan, Zhiwu Qing, Biao Gong, Yingya Zhang, Yujun Shen, Changxin Gao, and Nong Sang. A Recipe for Scaling up Text-to-Video Generation with Text-free Videos. In CVPR, pages 6572–6582,

2024. 2

- [40] Bichen Wu, Ching-Yao Chuang, Xiaoyan Wang, Yichen Jia, Kapil Krishnakumar, Tong Xiao, Feng Liang, Licheng Yu, and Peter Vajda. Fairy: Fast Parallelized InstructionGuided Video-to-Video Synthesis. In CVPR, pages 8261– 8270, 2024. 2

- [41] Haoning Wu, Erli Zhang, Liang Liao, Chaofeng Chen, Jingwen Hou, Annan Wang, Wenxiu Sun, Qiong Yan, and Weisi Lin. Exploring Video Quality Assessment on User Generated Contents from Aesthetic and Technical Perspectives. In ICCV, pages 20087–20097, 2023. 5, 6, 7
- [42] Jay Zhangjie Wu, Yixiao Ge, Xintao Wang, Stan Weixian Lei, Yuchao Gu, Yufei Shi, Wynne Hsu, Ying Shan, Xiaohu Qie, and Mike Zheng Shou. Tune-A-Video: One-Shot Tuning of Image Diffusion Models for Text-to-Video Generation. In ICCV, pages 7589–7599, 2023. 2, 3, 5, 6, 7
- [43] Xihui Liu, Dong Huk Park, Samaneh Azadi, Gong Zhang, Arman Chopikyan, Yuxiao Hu, Humphrey Shi, Anna Rohrbach, and Trevor Darrell. More Control for Free! Image Synthesis with Semantic Diffusion Guidance. In WACV, pages 289–299, 2023. 2
- [44] Xin Ma, Yaohui Wang, Gengyun Jia, Xinyuan Chen, Ziwei Liu, Yuan-Fang Li, Cunjian Chen, and Yu Qiao. Latte: Latent Diffusion Transformer for Video Generation. arXiv,

2024. 2

- [45] Ning Xu, Linjie Yang, Yuchen Fan, Dingcheng Yue, Yuchen Liang, Jianchao Yang, and Thomas S. Huang. YouTubeVOS: A Large-Scale Video Object Segmentation Benchmark. CoRR, abs/1809.03327, 2018. 12
- [46] Yaohui Wang, Xinyuan Chen, Xin Ma, Shangchen Zhou, Ziqi Huang, Yi Wang, Ceyuan Yang, Yinan He, Jiashuo Yu, Peiqing Yang, Yuwei Guo, Tianxing Wu, Chenyang Si, Yuming Jiang, Cunjian Chen, Chen Change Loy, Bo Dai, Dahua Lin, Yu Qiao, and Ziwei Liu. LAVIE: High-Quality Video Generation with Cascaded Latent Diffusion Models. arXiv,

2023. 3

- [47] Danah Yatim, Rafail Fridman, Omer Bar-Tal, Yoni Kasten, and Tali Dekel. Space-Time Diffusion Features for ZeroShot Text-Driven Motion Transfer. In CVPR, pages 8466– 8476, 2024. 2, 3, 5, 6, 7, 15
- [48] Hangjie Yuan, Shiwei Zhang, Xiang Wang, Yujie Wei, Tao Feng, Yining Pan, Yingya Zhang, Ziwei Liu, Samuel Albanie, and Dong Ni. InstructVideo: Instructing Video Diffusion Models with Human Feedback. In CVPR, pages 6463– 6474, 2024. 2
- [49] Kaidong Zhang, Jingjing Fu, and Dong Liu. Flow-Guided Transformer for Video Inpainting. In ECCV, pages 74–90,

2022. 12

- [50] Zhixing Zhang, Bichen Wu, Xiaoyan Wang, Yaqiao Luo, Luxin Zhang, Yinan Zhao, Peter Vajda, Dimitris N. Metaxas, and Licheng Yu. AVID: Any-Length Video Inpainting with Diffusion Model. In CVPR, pages 7162–7172, 2024. 2, 3, 6, 11, 12, 13, 14
- [51] Shangchen Zhou, Chongyi Li, Kelvin C. K. Chan, and Chen Change Loy. ProPainter: Improving Propagation and Transformer for Video Inpainting. In ICCV, pages 10443– 10452, 2023. 5, 11
- [52] Bojia Zi, Shihao Zhao, Xianbiao Qi, Jianan Wang, Yukai Shi, Qianyu Chen, Bin Liang, Kam-Fai Wong, and Lei Zhang. CoCoCo: Improving Text-Guided Video Inpainting for Better Consistency, Controllability and Compatibility. arXiv,

2024. 2, 3, 5, 6, 7

[Figure 605]

[Figure 606]

[Figure 607]

[Figure 608]

[Figure 609]

[Figure 610]

[Figure 611]

O: “A boat on the river.” S: “A cruiseship on the river.” C: “A motorboat on the river.”

[Figure 612]

[Figure 613]

[Figure 614]

[Figure 615]

[Figure 616]

[Figure 617]

[Figure 618]

O: “A car is driving on the road.” S: “A truck is driving on the road.”

C: “A sport car is driving on the road.”

[Figure 619]

[Figure 620]

[Figure 621]

[Figure 622]

[Figure 623]

[Figure 624]

[Figure 625]

O: “A duck walking on the grass.”

S: “A blackswan walking on the grass.”

C: “A rabbit walking on the grass.”

[Figure 626]

[Figure 627]

[Figure 628]

[Figure 629]

[Figure 630]

[Figure 631]

[Figure 632]

[Figure 633]

O: “A duck swimming in a pond.”

S: “A goose swimming in a pond.”

C: “A ball floating on a pond.”

#### (a) Source Video Frames (b) Text Descriptions

(c) Reference Images

- Figure 8. Selected data samples and the corresponding annotations from DAVIS-EDIT, with visualizations of (a) source video frames, (b) text descriptions, and (c) reference images highlighted in orange, green, and red, respectively. Specifically in (b), “O” represents the original (O) text description of the source video; “S” and “C” refers to the annotated captions indicating similar (S) and changing (C) shapes in the edited contents, respectively. Besides, we highlight the words depicting the main edited contents in red. In (c), we show the annotated images indicating similar and changing shapes on the left and right sides, respectively.

### Overview

(both main paper and this document) at https : //alonzoleeeooo.github.io/StableV2V, and highly encourage readers to refer to them for a more intuitive experience of STABLEV2V.

In our supplementary materials, we provide more details and results of STABLEV2V, so as to offer more insights into the proposed method, where we construct the contents following the structures below:

### A. Implementation Details of Shape-guided Depth Refinement Network

- • Implementation Details of Shape-guided Depth Refinement Network. The proposed depth refinement network plays a pivotal role in ensuring preciseness of depth guidance for STABLEV2V, where we illustrate its detailed implementation details from perspectives of the motivation, network architecture, and training in Sec. A.
- • Implementation Details of the DAVIS-EDIT. DAVISEDIT serves as the testing benchmark for the evaluation of STABLEV2V, where we report its implementation details in Sec. B, illustrating the annotation process of different prompts and some samples for demonstration.
- • More Qualitative Comparison. In Sec. C, we conduct the qualitative comparison with more video editing methods, especially the ones that are not open-sourced, including AVID [50], VASE [30], and I2VEdit [29].
- • More Results. In Sec. D, we demonstrate more qualitative results generated by STABLEV2V, from aspects of text-, image-based editing, and applications.
- • Limitations. Although STABLEV2V achieves promising performance in various editing tasks, it also comprises several limitations due to its inherent problems, i.e., the paradigm based on pre-trained models and depth maps, where we discuss its working boundaries in Sec. E.

In this section, we introduce the implementation details of shape-guided depth refinement network from various aspects, including its motivation, network architecture, and training details, as is presented in the following texts.

Motivation and Network Architecture. The depth refinement network serves as a pivotal component in STABLEV2V, where it is highly associated with the preciseness of depth guidance for CIG, thus subsequently affecting the consisetncy of the edited video. The final goal of such network is to calibrate the input depth maps by removing its redundant regions, meanwhile ensuring the consistency of the refined depth map with the corresponding edited first frame. To build such network, we draw inspirations from the task of video inpainting [51], where optical flows, similar to the depth maps in STABLEV2V, normally serve as a pivotal guidance for the inpainting process. Recently, VASE [30] borrows the same network architecture of the flow completion network in ProPainter [51], and adds an additional channel to the input layer to integrate the shape guidance, where the resulting network is used to offer flow guidance for reference-guided video editing. Enlightened

Notably, we offer the video format of all results

“A large raccoon standing on a waterfall.”

“A flamingo walking on the grass.”

“Pink flamingo swimming in the pool.”

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

SourceVideoStableV2V(Ours)SourceVideoAVIDStableV2V(Ours)AVID

SourceVideoSourceVideoStableV2V(Ours)StableV2V(Ours)StableV2V(Ours)StableV2V(Ours)SourceVideoSourceVideoVASEVASEVASEVASE

SourceVideoSourceVideoStableV2V(Ours)StableV2V(Ours)StableV2V(Ours)StableV2V(Ours)SourceVideoSourceVideoI2VEditI2VEditI2VEditI2VEdit

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

[Figure 663]

[Figure 664]

[Figure 665]

[Figure 666]

[Figure 667]

[Figure 668]

[Figure 669]

[Figure 670]

[Figure 671]

[Figure 672]

[Figure 673]

[Figure 674]

“A goose swimming in a lake.”

“A zebra walking on the grass.”

“A blue rocket is taking off.”

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

###### (a) Comparison with AVID (b) Comparison with VASE (c) Comparison with I2VEdit

- Figure 9. More qualitative comparison of STABLEV2V, compared to (a) AVID [50], (b) VASE [30], and (c) I2VEdit [29]. Note that we use the same first frame as the ones of I2VEdit [29] for comparison.

“A motorboat driving on the river.” “A ball floating on a pond.” “An apple floating on the water.”

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

Source

Video Input

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

DepthMap

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

DepthMap

Simulated

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

[Figure 863]

[Figure 864]

[Figure 865]

[Figure 866]

[Figure 867]

DepthMap

Refined

[Figure 868]

[Figure 869]

[Figure 870]

[Figure 871]

[Figure 872]

[Figure 873]

[Figure 874]

[Figure 875]

[Figure 876]

[Figure 877]

[Figure 878]

[Figure 879]

[Figure 880]

[Figure 881]

[Figure 882]

[Figure 883]

[Figure 884]

[Figure 885]

[Figure 886]

[Figure 887]

[Figure 888]

[Figure 889]

[Figure 890]

[Figure 891]

[Figure 892]

[Figure 893]

[Figure 894]

[Figure 895]

[Figure 896]

[Figure 897]

[Figure 898]

[Figure 899]

Edited

Video

Figure 10. More visualizations of intermediate results in ISA, where we show the reference images at the right-bottom corners.

by the aforementioned studies, we adopt the same architecture as VASE does, and utilize the segmentation mask of the first edited frame as guidance for the refinement process.

Training Details. We train the shape-guided depth refinement network on YouTube-VOS [45] dataset, whose training set consists of 3,471 videos and the corresponding mask annotations in total. To obtain the depth maps of all videos, we use an off-the-shelf depth estimator, i.e., MiDaS [32], to automatically annotate depth maps for all videos. Once the data are pre-processed, we train the shape-guided

depth refinement network for 50,000 iterations, along with a batch size of 8. Specifically in each training step, we randomly sample 10 frames of depth maps, and adopt the random mask generation algorithm in Flow-guided Transformer [49]. We use AdamW [18] optimizer to update the model parameters, with the learning rate set to 0.99.

### B. Implementation Details of the DAVIS-EDIT

In this section, we illustrate more implementation details of our testing benchmark DAVIS-EDIT. DAVIS-EDIT

“A panda walking on rocks in a zoo.”

[Figure 900]

[Figure 901]

[Figure 902]

[Figure 903]

[Figure 904]

“A bus with blue windows driving down the night road.”

[Figure 905]

[Figure 906]

[Figure 907]

[Figure 908]

[Figure 909]

“A wolf is walking on the grass.”

[Figure 910]

[Figure 911]

[Figure 912]

[Figure 913]

[Figure 914]

“A robot with a backpack walking up a hill.”

[Figure 915]

[Figure 916]

[Figure 917]

[Figure 918]

[Figure 919]

[Figure 920]

“A man walking in the grass near a tree.”

[Figure 921]

[Figure 922]

[Figure 923]

[Figure 924]

[Figure 925]

[Figure 926]

“A basketball is rolling in the grass.”

[Figure 927]

[Figure 928]

[Figure 929]

[Figure 930]

[Figure 931]

“A wooden train is on the tracks.”

[Figure 932]

[Figure 933]

[Figure 934]

[Figure 935]

[Figure 936]

“A man riding a horse.”

[Figure 937]

[Figure 938]

[Figure 939]

[Figure 940]

[Figure 941]

Figure 11. More text-based results generated by STABLEV2V, where we show the first frame of the source video in the first row.

plays a crucial role in evaluating the performance of STABLEV2V, where we curate this testing benchmark to offer a standard to promote further studies in addressing the shape misalignment problem for video editing. Fig. 8 demonstrates some samples selected from DAVIS-EDIT, along with the example text prompts and reference images that we manually annotate. To obtain the text prompts, we only modify specific words that describe the main elements of videos, e.g., objects and foregrounds, and put emphasis on embodying the shape difference problem during annotation. For example, we use “duck” to replace “blackswan” to represent the setting with similar shapes of edited contents, and edit “duck” into “rabbit” for the scenario with changing shape. For the annotation of reference images, we follow the similar principles, considering the variety of shape

“A SUV car is driving on the road.”

[Figure 942]

[Figure 943]

[Figure 944]

[Figure 945]

[Figure 946]

[Figure 947]

“A golden duck statue floating on a pond.”

[Figure 948]

[Figure 949]

[Figure 950]

[Figure 951]

[Figure 952]

[Figure 953]

“A red panda is walking on rocks in a zoo.”

[Figure 954]

[Figure 955]

[Figure 956]

[Figure 957]

[Figure 958]

[Figure 959]

“A rabbit sitting on the grass near a river.”

[Figure 960]

[Figure 961]

[Figure 962]

[Figure 963]

[Figure 964]

[Figure 965]

“A lama walking on a dirt road.”

[Figure 966]

[Figure 967]

[Figure 968]

[Figure 969]

[Figure 970]

[Figure 971]

[Figure 972]

[Figure 973]

“A rock rolling on the grass.”

[Figure 974]

[Figure 975]

[Figure 976]

[Figure 977]

[Figure 978]

[Figure 979]

[Figure 980]

[Figure 981]

[Figure 982]

[Figure 983]

[Figure 984]

“A cat running through a fence in the yard.”

[Figure 985]

[Figure 986]

[Figure 987]

[Figure 988]

[Figure 989]

[Figure 990]

“A wooden boat on the water.”

[Figure 991]

[Figure 992]

[Figure 993]

[Figure 994]

[Figure 995]

[Figure 996]

Figure 12. More image-based results generated by STABLEV2V, where we show the first frame of the source video in the first row for simplicity. Note that reference images are shown at the right-bottom corners of the first row.

differences. On top of that, we focus on collecting reference images that are tough for texts to illustrate, e.g., the Transformer truck in Fig. 8, so as to highlight the impacts of image guidance in such setting.

### C. More Qualitative Comparison

In this section, we showcase more qualitative comparison with more methods, especially the ones that are not opensourced yet, including AVID [50], VASE [30], and I2VEdit [29]. Specifically, both AVID and VASE serve as learningbased solution for video editing, where AVID is a textguided video inpainting framework initialized from SD Inpaint [33]; VASE is fine-tuned based on a image-guided

“Make the elephant a marble sculpture.”

“A toy car driving up the hill.”

|[Figure 997]|
|---|

|[Figure 998]|
|---|

|[Figure 999]|
|---|

|[Figure 1000]<br><br>|[Figure 1001]|
|---|
|
|---|

|[Figure 1002]|
|---|

|[Figure 1003]|
|---|

|[Figure 1004]|
|---|

|[Figure 1005]|
|---|

|[Figure 1006]|
|---|

|[Figure 1007]|
|---|

“Make him a robot.”

“A UFO floating above the grassland.”

|[Figure 1008]<br><br>[Figure 1009]|
|---|

|[Figure 1010]<br><br>[Figure 1011]|
|---|

|[Figure 1012]<br><br>[Figure 1013]|
|---|

|[Figure 1014]<br><br>[Figure 1015]|
|---|

|[Figure 1016]<br><br>|[Figure 1017]|
|---|
|
|---|

|[Figure 1018]|
|---|

|[Figure 1019]|
|---|

|[Figure 1020]|
|---|

|[Figure 1021]|
|---|

[Figure 1022]

[Figure 1023]

“ .”

###### “Make it a Chinese ink painting.”

“A teddy bear floating on the river.”

|[Figure 1024]<br><br>[Figure 1025]|
|---|

|[Figure 1026]<br><br>[Figure 1027]|
|---|

|[Figure 1028]<br><br>[Figure 1029]|
|---|

|[Figure 1030]<br><br>[Figure 1031]|
|---|

|[Figure 1032]<br><br>|[Figure 1033]|
|---|
|
|---|

[Figure 1034]

[Figure 1035]

[Figure 1036]

[Figure 1037]

|[Figure 1038]<br><br>[Figure 1039]|
|---|

“ .”

“Make the man on Mars.”

“A pirate ship floating on the river.”

|[Figure 1040]<br><br>[Figure 1041]|
|---|

|[Figure 1042]<br><br>[Figure 1043]|
|---|

|[Figure 1044]<br><br>[Figure 1045]|
|---|

|[Figure 1046]<br><br>[Figure 1047]|
|---|

|[Figure 1048]<br><br>[Figure 1049]|
|---|

|[Figure 1050]<br><br>[Figure 1051]|
|---|

|[Figure 1052]<br><br>[Figure 1053]|
|---|

|[Figure 1054]<br><br>[Figure 1055]|
|---|

|[Figure 1056]<br><br>[Figure 1057]|
|---|

|[Figure 1058]<br><br>|[Figure 1059]<br><br>[Figure 1060]|
|---|
<br><br>[Figure 1061]<br><br>|
|---|

(a) Instruction-based Editing (b) Sketch-based Editing

“A scenery of river.”

“A boat floating on the river, Ukiyo-e style.”

[Figure 1062]

|[Figure 1063]<br><br>[Figure 1064]<br><br>[Figure 1065]<br><br>[Figure 1066]|
|---|

|[Figure 1067]|
|---|

|[Figure 1068]|
|---|

|[Figure 1069]|
|---|

|[Figure 1070]|
|---|

|[Figure 1071]|
|---|

|[Figure 1072]|
|---|

|[Figure 1073]|
|---|

|[Figure 1074]|
|---|

|[Figure 1075]|
|---|

“A scenery of road beside a grassland.”

“A camel walking on the ground, Picasso style.”

|[Figure 1076]|
|---|

|[Figure 1077]|
|---|

|[Figure 1078]|
|---|

|[Figure 1079]|
|---|

|[Figure 1080]|
|---|

|[Figure 1081]|
|---|

|[Figure 1082]<br><br>[Figure 1083]|
|---|

|[Figure 1084]<br><br>[Figure 1085]|
|---|

|[Figure 1086]|
|---|

|[Figure 1087]|
|---|

“A scenery of a dirt road.”

“A man riding a horse, Van Gogh style.”

|[Figure 1088]|
|---|

|[Figure 1089]|
|---|

|[Figure 1090]|
|---|

|[Figure 1091]|
|---|

|[Figure 1092]|
|---|

|[Figure 1093]|
|---|

|[Figure 1094]|
|---|

|[Figure 1095]|
|---|

|[Figure 1096]<br><br>[Figure 1097]|
|---|

|[Figure 1098]|
|---|

“A scenery of ocean.”

“A woman walking in the grass, Monet style.”

|[Figure 1099]<br><br>[Figure 1100]<br><br>[Figure 1101]<br><br>[Figure 1102]|
|---|

|[Figure 1103]<br><br>[Figure 1104]|
|---|

|[Figure 1105]<br><br>[Figure 1106]|
|---|

|[Figure 1107]<br><br>[Figure 1108]|
|---|

|[Figure 1109]<br><br>[Figure 1110]<br><br>[Figure 1111]|
|---|

|[Figure 1112]|
|---|

|[Figure 1113]<br><br>[Figure 1114]|
|---|

|[Figure 1115]|
|---|

|[Figure 1116]|
|---|

|[Figure 1117]|
|---|

[Figure 1118]

(c) Video Inpainting (d) Video Style Transfer

- Figure 13. More results of applications conducted by STABLEV2V, including instruction-based editing, sketch-based editing, video style transfer, and video inpainting, whose backgrounds are highlighted in green, blue, red and yellow, respectively.

editor, i.e., Paint-by-Example [5], and mainly puts emphasis on object-centric video editing. I2VEdit serves as a first-frame-based video editing method that trains videospecific LoRA [14] to model the motion patterns of the source video. Since we do not have access to their code and model weights, we mainly compare STABLEV2V to their released demo video, with results presented in Fig. 9. For fair comparison, we use the same reference images provided by VASE [30] in their demonstrated videos, and adopt the same first frame as the ones of I2VEdit [29].

Analyses. By comparing STABLEV2V to learning-based

methods, i.e., AVID [50] and VASE [30], it is observed that AVID [50] has possibilities in producing results with inconsistent textures, e.g., the case of editing a swan into a duck, suggesting its deficiencies in maintaining the temporal consistency. VASE [30] produces results that merely transfer the textures of reference images into the edited videos, e.g., the cow-shape zebra in Fig. 9, since it is highly restricted by the input masks used in its inpainting paradigm. The aforementioned results illustrate the typical issues in learning-based methods, where they are limited to editing scenarios with little shape changes due to the inpainting

“A Transformer truck is driving on the road.”

[Figure 1119]

[Figure 1120]

[Figure 1121]

[Figure 1122]

[Figure 1123]

StableV2V(Ours)StableV2V(Ours)SourceVideoSourceVideoAnyV2VAnyV2VDMT

[Figure 1124]

[Figure 1125]

[Figure 1126]

[Figure 1127]

[Figure 1128]

[Figure 1129]

[Figure 1130]

[Figure 1131]

[Figure 1132]

[Figure 1133]

[Figure 1134]

“A robot is doing break dancing.”

[Figure 1135]

[Figure 1136]

[Figure 1137]

[Figure 1138]

[Figure 1139]

[Figure 1140]

[Figure 1141]

[Figure 1142]

[Figure 1143]

[Figure 1144]

[Figure 1145]

[Figure 1146]

[Figure 1147]

[Figure 1148]

[Figure 1149]

[Figure 1150]

[Figure 1151]

[Figure 1152]

[Figure 1153]

[Figure 1154]

[Figure 1155]

[Figure 1156]

[Figure 1157]

[Figure 1158]

[Figure 1159]

[Figure 1160]

[Figure 1161]

[Figure 1162]

[Figure 1163]

[Figure 1164]

[Figure 1165]

[Figure 1166]

[Figure 1167]

[Figure 1168]

[Figure 1169]

[Figure 1170]

[Figure 1171]

[Figure 1172]

- Figure 14. Failure cases of STABLEV2V illustrating the limitations of inherent problems of pre-trained models (top) and complicated motion patterns (bottom).

paradigm of their foundation models, i.e., SD Inpaint and Paint-by-Example, where such issues are significantly alleviated in STABLEV2V, since our first-frame-based scheme offers more flexiblity. By comparing STABLEV2V to other first-frame-based method, i.e., I2VEdit [29], two limitations are observed, where I2VEdit either produces results with information loss in the backgrounds, e.g., the case of editing the blackswan into a flamingo, or generates edited contents with simple motions like the case of a rising rocket. Conversely, results generated by STABLEV2V comprise more detailed textures such as the waves on the river and the smoke emitted by the rocket, indicating that STABLEV2V not only offers robust consistency in the edited videos, but also ensures its video quality in details.

### D. More Results

In this section, we illustrate more results generated by STABLEV2V. Specifically, we offer more visualizations of the intermediate results of ISA in Fig. 10. Besides, we show several results on text- and image-based editing scenarios in Fig. 11 and Fig. 12, respectively. Also, we present more applications performed by STABLEV2V in Fig. 13.

### E. Limitations

Although outperforming performance and applications are demonstrated by STABLEV2V, we observe that our proposed method also comprises several limitations due to the inherent problems that are caused by its paradigm, especially leading to potential working boundaries in cases that contain complicated motion patterns. Therefore in this section, we analyze the limitations and working boundaries of STABLEV2V, with some failure cases shown in Fig. 14, and discuss several potential solutions. Details of the aforementioned analyses are illustrated in the following texts.

inherent Problems of Pre-trained Models. Since STABLEV2V presents a training-free solution in addressing the misalignment problem between the motion controls and the edited contents, it relies on the use of pre-trained models and also suffers from severl inherent problems of

- them. Specifically, this limitation occurs mostly in two components, i.e., PFE and CIG, where the former normally leverages off-the-shelf image editing methods; the latter is mainly designed based on a conditional generation paradigm for image-to-video generation, i.e., Ctrl-Adapter [23], since few studies are available in the existing literature. For PFE, as is analyzed in Sec. 6 in our main paper, it comprises a certain degree of randomness in some textguided editors such as SD Inpaint [33], where edited contents with undesired orientations might be produced, and
- then subsequently mis-guide the CIG module to produce inferior results. For CIG, we observe that Ctrl-Adapter might lead to slight color discrepancy in several cases, especially when the edited contents are biased to certain colors, e.g., the case of editing the car into a Transformer truck in Fig.

14. Such color bias might be caused by the limited diversity and quality in the training data of Ctrl-Adapter, since its fine-tuning process may not require as much data as that used for its foundation model, i.e., I2VGen-XL [34]. Meanwhile, we observe that the generated textures are much more consistent than other studies, especially compared to the ones that also leverage I2VGen-XL, e.g., AnyV2V [20], since ISA ensures the alignment between the edited contents and the delivered motions to CIG. This finding indicates a potential solution to the above issue by considering ISA as a plug-and-play plugin, where we can integrate it into more powerful methods in the future once available.

Working Boundaries in Complicated Motion Patterns. Another problem that STABLEV2V might suffer from is its limited capabilities in modeling motion patterns that are too complicated, e.g., the case of a man doing break dancing in Fig. 14. Similar results are observed in other studies like DMT [47] and AnyV2V [20], where it is also tough for these methods to produce consistent results. Such scenario serves as the challenging case that most existing methods struggle to handle, where the task of modeling fine-grained motions for video editing deserves studying in future works.

