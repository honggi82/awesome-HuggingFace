# arXiv:2502.08590v2[cs.CV]12Mar2025

## Light-A-Video: Training-free Video Relighting via Progressive Light Fusion

Yujie Zhou1,6*, Jiazi Bu1,6*, Pengyang Ling2,6*, Pan Zhang6†, Tong Wu5, Qidong Huang2,6, Jinsong Li3,6, Xiaoyi Dong6, Yuhang Zang6, Yuhang Cao6, Anyi Rao4, Jiaqi Wang6, Li Niu1†

1Shanghai Jiao Tong University 2University of Science and Technology of China 3The Chinese University of Hong Kong 4Hong Kong University of Science and Technology 5Stanford University 6Shanghai AI Laboratory

[Figure 1]

Input Video Light Relighted Video Frames

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

“..., sci-fi RGB glowing,

RelightForegroundSequenceRelightVideoSequence

”

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

“..., red and blue

”

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

“..., on the desk, ”

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

“..., , city night”

Figure 1. Training-free video relighting. Equipped with an image relighting model (e.g., IC-Light [62]) and a video diffusion model (e.g., CogVideoX [59] and AnimateDiff [13]), Light-A-Video enables training-free video relighting for given video sequences or foreground sequences.

### Abstract

Recent advancements in image relighting models, driven by large-scale datasets and pre-trained diffusion models, have enabled the imposition of consistent lighting. However, video relighting still lags, primarily due to the excessive training costs and the scarcity of diverse, high-quality

video relighting datasets. A simple application of image relighting models on a frame-by-frame basis leads to several issues: lighting source inconsistency and relighted appearance inconsistency, resulting in flickers in the generated videos. In this work, we propose Light-A-Video, a training-free approach to achieve temporally smooth video relighting. Adapted from image relighting models, Light-

A-Video introduces two key techniques to enhance lighting consistency. First, we design a Consistent Light Attention (CLA) module, which enhances cross-frame interactions within the self-attention layers of the image relight model to stabilize the generation of the background lighting source. Second, leveraging the physical principle of light transport independence, we apply linear blending between the source video’s appearance and the relighted appearance, using a Progressive Light Fusion (PLF) strategy to ensure smooth temporal transitions in illumination. Experiments show that Light-A-Video improves the temporal consistency of relighted video while maintaining the relighted image quality, ensuring coherent lighting transitions across frames. Project page: https://bujiazi.github. io/light-a-video.github.io/.

### 1. Introduction

Illumination plays a crucial role in shaping our perception of visual content, impacting both its aesthetic quality and human interpretation of scenes. Relighting tasks [17, 24, 32, 34, 44, 47, 54, 65, 66], which focus on adjusting lighting conditions in 2D and 3D visual content, have long been a key area of research in computer graphics due to their broad practical applications, such as film production, gaming, and virtual environments. Traditional image relighting methods, which rely on physical illumination models, are typically confined to controlled laboratory settings and struggle to generalize to complex, unconstrained real-world lighting and material estimation.

In order to address these limitations, data-driven approaches [10, 19, 22, 39, 60, 64] have emerged, leveraging large-scale, diverse relighting datasets combined with pre-trained diffusion models. As the state-of-the-art image relighting model, IC-Light [62] modifies only the illumination of an image while maintaining its albedo unchanged. Based on the physical principle of light transport independence, IC-Light allows for controllable and stable illumination editing, such as adjusting lighting effects and simulating complex lighting scenarios. However, video relighting is significantly more challenging as it needs to maintain temporal consistency across frames. The scarcity of video lighting datasets and the high training costs further complicate the task. Thus, existing video relighting methods [64] struggle to deliver consistently high-quality results or are limited to specific domains, such as portraits.

In this work, we propose a training-free approach for video relighting, named Light-A-Video, which enables the generation of smooth, high-quality relighted videos without any additional training or optimization. As shown in Fig. 1, given a text prompt that provides a general description of the video and specified illumination conditions, our Light-A-Video pipeline can relight the input video in a zero-shot manner, fully leveraging the relighting capa-

###### “a handsome man with glasses, sunlight through the blinds”

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

InputIC-LightIC-Light+CLA

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

16

###### MotionStrength

11

6

1

0 1 2 3 4 5 6 7 8 9 10 11 12 13 14

-4

Frame Index Input IC-Light IC-Light + CLA

man ICPF_man CLA_man

Figure 2. Relighted frames of vanilla IC-Light and “IC-Light + CLA” . The line chart depicts the average optical flow intensity between adjacent frames. Since IC-Light performs image relighting based on each independent frame, its results show a noticeable jitter between frames, especially in the generated background lighting. Conversely, the proposed CLA facilitates consistent lighting generation by forcing interaction between frames.

bilities of image-based models and the motion priors of the video diffusion model. To achieve this goal, we initially apply an image-relighting model to video-relighting tasks on a frame-by-frame basis, and observe that the generated lighting source is unstable across the video frames. This instability leads to inconsistencies in the relighting of the objects’ appearances and significant flickering across frames. To stabilize the generated lighting source and ensure consistent results, we design a Consistent Light Attention (CLA) module within the self-attention layers of the image relighting model. As shown in Fig. 2, by incorporating additional temporally averaged features into the attention computation, CLA facilitates cross-frame interactions, producing a structurally stable lighting source. To further enhance the appearance stability across frames, we utilize the motion priors of the video diffusion model with a novel Progressive Light Fusion (PLF) strategy. Adhering to the physical principles of light transport, PLF progressively employs linear blending to integrate relighted appearances from the CLA into each original denoising target, which gradually guides the video denoising process toward the desired relighting direction. Finally, Light-AVideo serves as a complete end-to-end pipeline, effectively achieving smooth and consistent video relighting. As a training-free approach, Light-A-Video is not restricted to specific video diffusion models, making it highly compatible with a range of popular video generation backbones, including UNet-based and DiT-based models such as Ani-

mateDiff [13] and CogVideoX [59]. Our contributions are summarized as follows:

- • We propose Light-A-Video, a novel training-free video relighting framework that generalizes the capabilities of image relighting models to the video domain, enabling flexible and temporally consistent video relighting.
- • We introduce two key designs: a consistent light attention module to enhance the stability of lighting sources across frames, and a progressive light fusion strategy gradually injects lighting information to facilitate temporal consistency in video appearance.
- • Extensive experiments under various scenarios demonstrate the effectiveness and versatility of the proposed method, which not only supports relighting the entire video sequences but also enables relighting for given foreground sequences.

### 2. Related Work

Video Diffusion Models. Video diffusion models [2– 6, 13, 16, 50, 53, 59, 61] aim to synthesize temporally consistent image frames based on provided conditions, such as a text prompt or an image prompt. In the realm of text-to-video (T2V) generation, the majority of methods [3, 5, 6, 13, 50, 61] train additional motion modeling modules from existing text-to-image architectures to model the correlation between video frames, while others [16, 59] train from scratch to learn video priors. For image-to-video (I2V) tasks that enhance still images with reasonable motions, a line of research [7, 56] proposes novel frameworks dedicated to image animation. Some approaches [12, 14, 63] serve as plug-to-play adapters. Stable Video Diffusion [2] fine-tune pre-trained T2V models for I2V generation, achieving impressive performance. Numerous works [26, 29, 33] focus on controllable generation, providing more controllability for users. Video diffusion models, due to their inherent video priors, are capable of synthesizing smooth and consistent video frames that are both content-rich and temporally harmonious.

Learning-based Illumination Control. Over the past few years, a variety of lighting control techniques [32, 34, 47] for 2D and 3D visual content based on deep neural networks have been proposed, especially in the field of portrait lighting [1, 22, 43, 45, 46], along with a range of baselines [17, 44, 54, 65, 66] aimed at improving the effectiveness, accuracy, and theoretical foundation of illumination modeling. Recently, owing to the rapid development of diffusion-based generative models, a number of lighting control methods [10, 19, 39, 60] utilizing diffusion models have also been introduced. Relightful Harmonization [39] focuses on harmonizing sophisticated lighting effects for the foreground portrait conditioning on a given background image. SwitchLight [22] suggests training a physically co-designed framework for human portrait relighting. IC-

Light [62] is a state-of-the-art approach for image relighting. LumiSculpt [64] enables consistent lighting control in T2V generation models for the first time. However, in the domain of video lighting, the aforementioned approaches fail to simultaneously ensure precise lighting control and exceptional visual quality. This work incorporates a pretrained image lighting control model into the denoising process of a T2V model through progressive guidance, leveraging the latter’s video priors to facilitate the smooth transfer of image lighting control knowledge, thereby enabling accurate and harmonized control of video lighting.

Video Editing with Diffusion Models. In recent years, diffusion-based video editing has undergone significant advancements. Some researches [28, 29, 31, 52, 55] adopt pre-trained text-to-image (T2I) backbones for video editing. Another line of approaches [8, 18, 57, 58] leverages pretrained optical flow models to enhance the temporal consistency of output video. Numerous studies [11, 20, 37] have concentrated on exploring zero-shot video editing approaches. COVE [51] leverages the inherent diffusion feature correspondence proposed by DIFT [48] to achieve consistent video editing. SDEdit [30] utilizes the intrinsic capability of diffusion models to refine details based on a given layout, enabling efficient editing for both image and video. Despite the remarkable performance of existing video editing techniques in various settings, there remains a lack of approaches specifically designed for controlling the lighting of videos.

### 3. Preliminary

##### 3.1. Diffusion Model

Given an image x0 that follows the real-world data distribution, we first encode x0 into latent space z0 = E(x0) using a pretrained autoencoder {E(·),D(·)}. The forward diffusion process is a T steps Markov chain [15], corresponding to the iterative introduction of Gaussian noise ϵ, which can be expressed as:

zt = 1 − βtzt−1 + βtϵ, (1)

where βt ∈ (0,1) determines the amount of Gaussian noise introduced at time step t. Mathematically, the above cumulative noise adding has the following closed-form:

zt = √α¯tz0 + √1 − α¯tϵ, (2) where α¯t = t1(1 − βt).

For numerical stability, v-prediction [41] approach is employed, where the diffusion model outputs a predicted velocity v to represent the denoising direction. Defined as:

√1 − α¯tz0. (3)

v = √α¯tϵ −

During inference, the noise-free component zˆ0←t can be recovered from the model’s output vt as follows:

√1 − α¯tvt. (4)

zˆ0←t = √α¯tzt −

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

Add Noise ...

###### …

###### ...

Video Diffusion Model

 − 

Source Video

[Figure 64]

1− 

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

V

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

Add

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

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

A car driving on the street,

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

Self Attn

[Figure 159]

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

K

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

###### h

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

Q

Consistent Target  ← 

[Figure 334]

Details Compensation

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

Temporal Average

[Figure 360]

[Figure 361]

Share Query

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

Image Relight Model

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

Relight Target  ← 

[Figure 394]

[Figure 395]

[Figure 396]

[Figure 397]

[Figure 398]

[Figure 399]

[Figure 400]

[Figure 401]

[Figure 402]

Q

Fusion Target  ← 

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

[Figure 432]

[Figure 433]

[Figure 434]

[Figure 435]

[Figure 436]

[Figure 437]

[Figure 438]

[Figure 439]

[Figure 440]

[Figure 441]

[Figure 442]

[Figure 443]

[Figure 444]

[Figure 445]

[Figure 446]

[Figure 447]

[Figure 448]

[Figure 449]

[Figure 450]

[Figure 451]

[Figure 452]

[Figure 453]

[Figure 454]

[Figure 455]

[Figure 456]

[Figure 457]

[Figure 458]

[Figure 459]

[Figure 460]

[Figure 461]

[Figure 462]

[Figure 463]

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

[Figure 498]

[Figure 499]

[Figure 500]

[Figure 501]

[Figure 502]

[Figure 503]

[Figure 504]

[Figure 505]

Self Attn

[Figure 506]

[Figure 507]

[Figure 508]

[Figure 509]

[Figure 510]

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

[Figure 542]

[Figure 543]

[Figure 544]

[Figure 545]

K

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

[Figure 561]

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

Relight Target  ← 

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

V

Consistent Light Attention Consistent Target

Figure 3. The pipeline of Light-A-Video. A source video is first noised and processed through the VDM for denoising across Tm steps. At each step, the predicted noise-free component with details compensation serves as the Consistent Target zv0←t, inherently representing the VDM’s denoising direction. Consistent Light Attention infuses zv0←t with unique lighting information, transforming it into the Relight Target zr0←t. The Progressive Light Fusion strategy then merges two targets to form the Fusion Target z˜0←t, which provides a refined direction for the current step.The bottom-right part illustrates the iterative evolution of zv0←t.

##### 4.1. Problem Formulation

zˆ0←t represents the denoising target at time step t.

##### 3.2. Light Transport

Given a source video and a lighting condition c, the objective of video relighting is to render the source video into the relighted video that maintains the motion in the source video while aligning the lighting condition c. Unlike image relighting that solely concentrates on appearance, video relighting raises extra challenges in maintaining temporal consistency and motion preservation, necessitating highquality visual coherence across frames.

Light transport theory [9, 62] demonstrates that arbitrary image appearance IL can be decomposed by the product of a light transport matrix T and environment illumination L, which can be expressed as:

IL = TL, (5)

where T is a single matrix for linear light transform [9] and L denotes variable environment illumination. Given the linearity of T, the merging between environment illumination L is equal to the fusion of image appearance IL, i.e.,

##### 4.2. Consistent Light Attention

Given the achievement in image relighting model [62], a straightforward approach for video relighting is to directly perform frame-by-frame image relighting under the same lighting condition. However, as illustrated in Fig. 2, this naive method fails to maintain appearance coherence across frames, resulting in frequent flickering of the generated light source and inconsistent temporal illumination.

. (6)

IL

1+L2 = T(L1 + L2) = IL

+ IL

1

2

Such characteristic suggests the feasibility of lighting control by indirectly constraining image appearance, i.e., the consistent image light constraint in IC-Light [62].

### 4. Light-A-Video

To improve inter-frame information integration and generate a stable light source, we propose a Consistent Light Attention (CLA) module. Specifically, for each selfattention layer in the IC-Light model, a video feature map h ∈ R(b×f)×(h×w)×d serves as the input, where b is the batch size and f is the number of video frames, h and w denote the height and width of the feature map, with h × w representing the number of tokens for attention computation. With linearly projections, h is projected into query,

Section 4.1 defines the objectives of the video relighting task. Section 4.2 reveals that per-frame image relighting for video sequences suffers from lighting source inconsistency and accordingly proposes the Consistent Lighting Attention (CLA) module for enhanced stability in generated lighting source across frames. Section 4.3 represents the Progressive Light Fusion (PLF) strategy for temporally consistent video appearance generation.

key and value features Q,K,V ∈ R(b×f)×(h×w)×d. The attention computation is defined as follows:

Self-Attn(Q,K,V ) = Softmax

QKT √

d

V. (7)

Note that the naive method treats the frame dimension as the batch size, performing self-attention frame by frame with the image relighting model, which results in each frame attending only to its features. For the CLA module, as shown in Fig. 3, a dual-stream attention fusion strategy is applied. Given the input feature h, the original stream directly feeds the feature map into the attention module to compute frame-by-frame attention, resulting in the output h′1. The average stream first reshapes h into Rb×f×(h×w)×d, averages it along the temporal dimension, then expands it f times to obtain h¯. Specifically, the average stream mitigates high-frequency temporal fluctuations, thereby facilitating the generation of a stable background light source across frames. Meanwhile, the original stream retains the original high-frequency details, thereby compensating for the detail loss incurred by the averaging process. Then, h¯ is input into the self-attention module and the output is h¯′2. The final output h′o of the CLA module is a weighted average between two streams, with the trade-off parameter γ,

h′o = (1 − γ)h′1 + γh¯′2. (8)

With the help of CLA, the result can capture global context across the entire video and generate a more stable lighting source, as shown in Fig. 2.

##### 4.3. Progressive Light Fusion

CLA module improves cross-frame consistency but lacks pixel-level constraints, leading to inconsistencies in appearance details. To address this, we leverage motion priors in the Video Diffusion Model (VDM), which are trained on large-scale video datasets and use a temporal attention module to ensure consistent motion and lighting changes. The novelty of our Light-A-Video lies in progressively injecting the relighting results as guidance into the denoising process.

In the pipeline as shown in Fig 3, a source video is first encoded into latent space, and then add Tm step noise to acquire the noisy latent zm. At each denoising step t, the noise-free component zˆ0←t in Eq. 4 is predicted, which serves as the denoising target for the current step. Prior work demonstrated the potential of applying tailored manipulation in denoising targets for guided generation, with significant achievements observed in high-resolution image synthesis [23] and text-based image editing [40].

Driven by the motion priors in the VDM, the denoising process encourage zˆ0←t to be temporally consistent. Thus, we define this target as the video Consistent Target zv0←t with environment illumination Lvt . However, discrepancies

 ← 

 ← 

###### PLF

- Figure 4. Visualization of the PLF strategy. During the denoising process of the VDM, the PLF strategy progressively replaces

the original Consistent Target zv0←t with the Fusion Target z˜0←t, guiding the denoising direction from vt to v˜t.

[Figure 658]

Predicted  ←  Result without

[Figure 659]

[Figure 660]

Detail Compensation

Add

[Figure 661]

Result with

Directly Relight

[Figure 662]

Consistent Target  ← 

A coin on the desk,

[Figure 663]

[Figure 664]

Source Video Predicted  ←  Detail Compensation

[Figure 665]

Relight

- Figure 5. Visualization of the detail compensation. ∆dm records the difference between zˆ0←m and the source video in the first denoising step, which is used as a detail compensation component for detail preservation in the consistent target.

still exist between the predicted zˆ0←t and the original video, resulting in detail loss in the relighted video. To address this issue, as shown in Fig. 5, details compensation ∆dm is incorporated into the zv0←t at each step. Then, zv0←t is sent into the CLA module to obtain the relighted latent, which serves as the Relight Target zr0←t with the illumination Lrt for the t-th denoising step. Aligning with the light transport theory in Section 3.2, a pre-trained VAE {E(·),D(·)} is used to decode the two targets into pixel level, yielding the image appearances Ivt = D(zv0←t) and Irt = D(zr0←t), respectively. Refer to Eq. 6, the fusing appearance Ift can be formulated as:

###### Ift = T(Lvt + Lrt) = Ivt + Irt. (9)

It is observed that directly using encoded latent E(Ift ) as the new target at each step results in suboptimal performance. This is attributed to the excessively large gap between the two targets, which exceeds the refinement capability of the VDM and consequently causes visible temporal lighting jitter. To mitigate this gap, a progressive lighting fusion strategy is proposed. Specifically, a fusion weight λt is intro-

duced, which decreases as denoising progresses, thereby gradually reducing the influence of the relight target. The progressive light fusion appearance is defined as Ipt, i.e.,

Ipt = Ivt + λt(Irt − Ivt ). (10)

The encoded latent z˜0←t = E(Ipt) is utilized as the Fusion Target for step t, replacing the original zv0←t. Based on the fusion target, the less noisy latent zt−1 can be computed with DDIM scheduler with v-prediction:

√α¯tat, (11)

,bt = √α¯t−1 −

1 − α¯t−1 1 − α¯t

at =

zt−1 = atzt + btz˜0←t. (12)

From Eq. 4, the fusion target z˜0←t determines a new denoising direction, denoted as v˜t,

√α¯tzt − z˜0←t √1 − α¯t

, (13)

v˜t =

which means PLF essentially refines vt iteratively and guides the denoising process towards the relighting direction, as shown in Fig 4. Other schedulers capable of modeling the denoising direction, such as Euler Scheduler [21] and Rectified Flow [27], are also applicable. As the denoising progresses, smooth and consistent illumination injection is achieved, ensuring coherent video relighting.

### 5. Experiments

##### 5.1. Experimental Details

Baselines. Given the lack of established video relighting methods, we adopt the state-of-the-art image relighting technique to perform frame-by-frame relighting on videos as a baseline. To verify the temporal smoothing effect of illumination using a VDM, we construct two comparative methods by applying SDEdit [30] to the per-frame results of IC-Light [62]. These two methods are named IC-Light + SDEdit-0.2 and IC-Light + SDEdit-0.6, corresponding to noise levels of 20% and 60%. Finally, IC-Light + AnyV2V [25] serves as another baseline. Specifically, IC-Light relights the first frame, and AnyV2V propagates the appearance information from the first frame to all subsequent frames, preserving the content of the source video. Evaluation metrics. Three widely adopted metrics are reported for quantitative evaluation. First, the temporal consistency of the generated video is assessed using the average CLIP [38] score across consecutive video frames. Second, the optical flow for each baseline video is estimated using RAFT [49], and the motion preservation score of each method is assessed by calculating the optical flow discrepancy with the source video. Third, to evaluate the quality of relighted image, a video test dataset is collected. The FID [42] score is then calculated between the results

of each method and the frame-by-frame IC-Light results, serving as the metric for relight quality evaluation. Finally, 52 volunteers are invited to conduct a user study across three aspects: Lighting Prompt Alignment (alignment between video content and lighting prompt), Video Smoothness (temporal consistency of the relighted video), and ID-Preservation (consistency of the object’s identity and albedo before and after relighting). The volunteers rank the results of five methods, and the average user ranking is used as a preference metric.

Datasets. We constructed a video test dataset consisting of 73 videos. The majority of these videos are selected from the DAVIS [36] public dataset, which contains a diverse collection of semantically rich videos with pronounced motion. Additionally, some videos are collected from Pixabay [35], featuring high-quality videos with significant motion. All quantitative metrics are evaluated on our collected dataset. For each video, two lighting prompts are applied, and three lighting directions are randomly chosen.

Implementation details. Unless otherwise specified, the default models employed for image relighting and VDM in the subsequent experiments are IC-Light [62] and AnimateDiff [13] Motion-Adapter-v3, respectively. In the ICLight model, the lighting conditions c for image relighting are derived from two components: First, a text prompt that describes the characteristics of the light source (e.g., neon light, sunshine, etc.). Second, a lighting map is utilized to represent the light intensity across the scene. This lighting map is then encoded by a VAE and serves as the initial latent for the denoising process. During the inference stage, the source video is added with 50% noise. Subsequently, the VDM employs a denoising process with Tm = 25 steps to progressively fuse the relight target. For the parameters in the pipeline, γ = 0.5 in the CLA module is used to balance the original attention feature and the cross-frame averaged feature. In the PLF strategy, the fusion weight λt decreases as denoising progresses, and we set λt = 1 − t/Tm.

##### 5.2. Qualitative Results

As depicted in Fig. 6, the frame-by-frame IC-Light method ensures high single-frame quality. However, the lack of consistency design and VDM temporal priors leads to significant flickering of the light source and overall appearance. By introducing VDM priors, IC-Light + SDEdit0.2 maintains content consistent with the source video, but still exhibits noticeable relight appearance jitter. IC-Light + SDEdit-0.6 further enhances temporal smoothness, yet object identity shifts occur. AnyV2V transfers the appearance of the first relight frame to subsequent frames, but this pixellevel migration method, lacking perception of the given light source, results in unreasonable illumination changes. In contrast, Light-A-Video achieves high-quality video relighting, demonstrating strong temporal consistency and high fidelity to the light source.

[Figure 666]

[Figure 667]

[Figure 668]

[Figure 669]

[Figure 670]

[Figure 671]

[Figure 672]

[Figure 673]

IC-Light+AnyV2VLight-A-VideoInputVideoIC-Light+SDEdit-0.2IC-LightIC-Light+SDEdit-0.6

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

“a sports car drives on the street, left soft ” “a handsome man walks in the factory, ”

- Figure 6. Qualitative comparison of baseline methods. Given a source video and guidance text prompt, Light-A-Video achieves high temporal consistency and fidelity to the light condition, outperforming other methods in avoiding flickering, jitter, and identity shifts. VDM used: AnimateDiff (Left), CogVideoX (Right).

(a) Relighted Image Quality (b) Temporal Consistency (c) User Preference FID Score (↓) CLIP Score (↑) Motion Preservation (↓) LPA (↑) VS (↑) IP (↑)

Evaluation Metric

IC-Light [62] / 0.9040 5.969 3.160 2.128 3.068 IC-Light + SDEdit-0.2 13.79 0.9199 5.959 2.850 2.752 3.014 IC-Light + SDEdit-0.6 62.61 0.9483 7.544 2.472 3.318 2.488 IC-Light + AnyV2V [25] 32.73 0.9436 8.854 2.766 3.300 2.774 Light-A-Video (Ours) 29.63 0.9667 1.833 3.752 3.502 3.656

Table 1. Quantitative comparison of baseline methods. We achieves better results in relighted image quality and temporal stability.

##### 5.3. Video Relighting with Background Generation

is progressively denoised, leveraging the VDM’s inpainting capability to generate the background. Subsequently, from step Tm to 0, the CLA module and PLF strategy are employed to achieve a temporally consistent relighting appearance of the video. These results illustrate that our pipeline can produce high-quality video relighting results with consistent background generation. 1

As depicted in Fig. 7, Light-A-Video can accept a video foreground sequence and a user-provided text prompt as input, generating a corresponding video background and illumination that aligns with the prompt descriptions. Specifically, the input foreground sequence is initially processed with IC-Light for frame-by-frame relighting, while the background is entirely noised to serve as the initialization latent for the VDM. From step T to Tm, the background

1More examples and ablation experiments are provided in the supplementary material.

[Figure 714]

[Figure 715]

[Figure 716]

[Figure 717]

InputInputLight-A-VideoLight-A-Video

[Figure 718]

[Figure 719]

[Figure 720]

[Figure 721]

“squirrel on the bed, moon light ”

[Figure 722]

[Figure 723]

[Figure 724]

[Figure 725]

[Figure 726]

[Figure 727]

[Figure 728]

[Figure 729]

“A cat walking on a runway with ”

- Figure 7. Text-conditioned video illumination modifying with background generation. Given a video foreground sequence and a text description of the target illumination, our method synthesizes suitable backgrounds and harmonious illumination.

##### 5.4. Quantitative Evaluation

The quantitative comparison of our method with various baselines is presented in Tab. 1. Given the addition of only 20% noise, IC-Light + SDEdit-0.2 exhibits performance in video relighting that is nearly identical to IC-Light, resulting in significant temporal flickering in both methods. IC-Light + SDEdit-0.6 provides enhanced temporal consistency but suffers from object identity shifts due to the introduction of excessive noise. For AnyV2V, the appearance of the first frame aligns well with the IC-Light results. However, its inability to perceive the light source, combined with inherent quality degradation in subsequent frames, leads to a low motion preservation score. In contrast, Light-A-Video achieves a low FID score while maintaining high temporal consistency, demonstrating its effectiveness in both relighted image quality and temporal stability.

##### 5.5. Ablation Study

An ablation study is conducted to assess the importance of our CLA and PLF modules. As illustrated in Fig. 8, for the video relighting task involving background generation, frame-by-frame IC-Light provides high-quality singleframe relighting but lacks control over temporal consistency. This results in inconsistencies in lighting sources and relighted appearances across frames. The CLA module enables cross-frame information exchange, which stabilizes the generation of background lighting sources. Additionally, by introducing VDM motion priors and employing

[Figure 730]

[Figure 731]

[Figure 732]

[Figure 733]

Input

[Figure 734]

[Figure 735]

[Figure 736]

[Figure 737]

w/oPLFLight-A-VideoIC-Lightw/oCLA

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

“a dog in the room, sunshine from window”

Figure 8. Ablation Study. Results of video relighting with the CLA module or the PLF strategy removed.

PLF’s strategy for progressive fusion of relight targets into the original denoising target, Light-A-Video ensures temporally smooth relighting. The overall video quality is also significantly improved with the aid of VDM priors.

##### 5.6. Limitation and Future Work

Despite the impressive results achieved by our training-free method, its performance is inherently constrained by the capabilities of the underlying image-relighting model and the VDM. While Light-A-Video demonstrates remarkable proficiency in ensuring stable lighting and temporal consistency, the CLA module, which is designed to stabilize background lighting, exhibits limitations when it comes to modeling dynamic lighting changes. To address this limitation, future work will focus on developing novel methods that can more effectively handle dynamic lighting conditions.

#### 6. Conclusion In summary, this paper introduces Light-A-Video, a training-free method that utilizes state-of-the-art image relighting models to achieve temporally consistent video relighting. By incorporating the Consistent Light Attention (CLA) module to stabilize lighting source generation and employing the Progressive Light Fusion (PLF) strategy for smooth appearance transitions, Light-A-Video significantly enhances the temporal coherence of relighted videos while preserving the high-quality relighting of individual frames.

### References

- [1] Jonathan T Barron and Jitendra Malik. Shape, illumination, and reflectance from shading. IEEE transactions on pattern analysis and machine intelligence, 37(8):1670–1687, 2014. 3
- [2] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023. 3
- [3] Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22563–22575, 2023. 3
- [4] Jiazi Bu, Pengyang Ling, Pan Zhang, Tong Wu, Xiaoyi Dong, Yuhang Zang, Yuhang Cao, Dahua Lin, and Jiaqi Wang. Broadway: Boost your text-to-video generation model in a training-free way. arXiv preprint arXiv:2410.06241, 2024.
- [5] Haoxin Chen, Menghan Xia, Yingqing He, Yong Zhang, Xiaodong Cun, Shaoshu Yang, Jinbo Xing, Yaofang Liu, Qifeng Chen, Xintao Wang, et al. Videocrafter1: Open diffusion models for high-quality video generation. arXiv preprint arXiv:2310.19512, 2023. 3
- [6] Haoxin Chen, Yong Zhang, Xiaodong Cun, Menghan Xia, Xintao Wang, Chao Weng, and Ying Shan. Videocrafter2: Overcoming data limitations for high-quality video diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7310– 7320, 2024. 3
- [7] Xi Chen, Zhiheng Liu, Mengting Chen, Yutong Feng, Yu Liu, Yujun Shen, and Hengshuang Zhao. Livephoto: Real image animation with text-guided motion control. In European Conference on Computer Vision, pages 475–491. Springer, 2025. 3
- [8] Yuren Cong, Mengmeng Xu, Christian Simon, Shoufa Chen, Jiawei Ren, Yanping Xie, Juan-Manuel Perez-Rua, Bodo Rosenhahn, Tao Xiang, and Sen He. Flatten: optical flowguided attention for consistent text-to-video editing. arXiv preprint arXiv:2310.05922, 2023. 3
- [9] Paul Debevec, Tim Hawkins, Chris Tchou, Haarm-Pieter Duiker, Westley Sarokin, and Mark Sagar. Acquiring the reflectance field of a human face. In Proceedings of the 27th annual conference on Computer graphics and interactive techniques, pages 145–156, 2000. 4
- [10] Kangle Deng, Timothy Omernick, Alexander Weiss, Deva Ramanan, Jun-Yan Zhu, Tinghui Zhou, and Maneesh Agrawala. Flashtex: Fast relightable mesh texturing with lightcontrolnet. In European Conference on Computer Vision, pages 90–107. Springer, 2025. 2, 3
- [11] Michal Geyer, Omer Bar-Tal, Shai Bagon, and Tali Dekel. Tokenflow: Consistent diffusion features for consistent video editing. arXiv preprint arXiv:2307.10373, 2023. 3
- [12] Xun Guo, Mingwu Zheng, Liang Hou, Yuan Gao, Yufan Deng, Chongyang Ma, Weiming Hu, Zhengjun Zha, Haibin

- Huang, Pengfei Wan, et al. I2v-adapter: A general imageto-video adapter for video diffusion models. arXiv preprint arXiv:2312.16693, 2023. 3
- [13] Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized textto-image diffusion models without specific tuning. arXiv preprint arXiv:2307.04725, 2023. 1, 3, 6
- [14] Yuwei Guo, Ceyuan Yang, Anyi Rao, Maneesh Agrawala, Dahua Lin, and Bo Dai. Sparsectrl: Adding sparse controls to text-to-video diffusion models. In European Conference on Computer Vision, pages 330–348. Springer, 2025. 3
- [15] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 3
- [16] Wenyi Hong, Ming Ding, Wendi Zheng, Xinghan Liu, and Jie Tang. Cogvideo: Large-scale pretraining for text-to-video generation via transformers. arXiv preprint arXiv:2205.15868, 2022. 3
- [17] Andrew Hou, Ze Zhang, Michel Sarkis, Ning Bi, Yiying Tong, and Xiaoming Liu. Towards high fidelity face relighting with realistic shadows. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 14719–14728, 2021. 2, 3
- [18] Zhihao Hu and Dong Xu. Videocontrolnet: A motion-guided video-to-video translation framework by using diffusion model with controlnet. arXiv preprint arXiv:2307.14073,

2023. 3

- [19] Haian Jin, Yuan Li, Fujun Luan, Yuanbo Xiangli, Sai Bi, Kai Zhang, Zexiang Xu, Jin Sun, and Noah Snavely. Neural gaffer: Relighting any object via diffusion. arXiv preprint arXiv:2406.07520, 2024. 2, 3
- [20] Ozgur Kara, Bariscan Kurtkaya, Hidir Yesiltepe, James M Rehg, and Pinar Yanardag. Rave: Randomized noise shuffling for fast and consistent video editing with diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6507–6516,

2024. 3

- [21] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. Advances in neural information processing systems, 35:26565–26577, 2022. 6
- [22] Hoon Kim, Minje Jang, Wonjun Yoon, Jisoo Lee, Donghyun Na, and Sanghyun Woo. Switchlight: Co-design of physicsdriven architecture and pre-training framework for human portrait relighting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 25096–25106, 2024. 2, 3
- [23] Younghyun Kim, Geunmin Hwang, Junyu Zhang, and Eunbyung Park. Diffusehigh: Training-free progressive highresolution image synthesis through structure guidance. arXiv preprint arXiv:2406.18459, 2024. 5
- [24] Peter Kocsis, Julien Philip, Kalyan Sunkavalli, Matthias Nießner, and Yannick Hold-Geoffroy. Lightit: Illumination modeling and control for diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9359–9369, 2024. 2

- [25] Max Ku, Cong Wei, Weiming Ren, Harry Yang, and Wenhu Chen. Anyv2v: A tuning-free framework for any video-tovideo editing tasks. arXiv preprint arXiv:2403.14468, 2024. 6, 7
- [26] Pengyang Ling, Jiazi Bu, Pan Zhang, Xiaoyi Dong, Yuhang Zang, Tong Wu, Huaian Chen, Jiaqi Wang, and Yi Jin. Motionclone: Training-free motion cloning for controllable video generation. arXiv preprint arXiv:2406.05338, 2024. 3
- [27] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022. 6
- [28] Shaoteng Liu, Yuechen Zhang, Wenbo Li, Zhe Lin, and Jiaya Jia. Video-p2p: Video editing with cross-attention control. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8599–8608, 2024. 3
- [29] Yue Ma, Yingqing He, Hongfa Wang, Andong Wang, Chenyang Qi, Chengfei Cai, Xiu Li, Zhifeng Li, HeungYeung Shum, Wei Liu, et al. Follow-your-click: Opendomain regional image animation via short prompts. arXiv preprint arXiv:2403.08268, 2024. 3
- [30] Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. Sdedit: Guided image synthesis and editing with stochastic differential equations. arXiv preprint arXiv:2108.01073, 2021. 3, 6
- [31] Ron Mokady, Amir Hertz, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Null-text inversion for editing real images using guided diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6038–6047, 2023. 3
- [32] Thomas Nestmeyer, Jean-Fran¸cois Lalonde, Iain Matthews, and Andreas Lehrmann. Learning physics-guided face relighting under directional light. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5124–5133, 2020. 2, 3
- [33] Muyao Niu, Xiaodong Cun, Xintao Wang, Yong Zhang, Ying Shan, and Yinqiang Zheng. Mofa-video: Controllable image animation via generative motion field adaptions in frozen image-to-video diffusion model. In European Conference on Computer Vision, pages 111–128. Springer, 2025. 3
- [34] Rohit Pandey, Sergio Orts-Escolano, Chloe Legendre, Christian Haene, Sofien Bouaziz, Christoph Rhemann, Paul E Debevec, and Sean Ryan Fanello. Total relighting: learning to relight portraits for background replacement. ACM Trans. Graph., 40(4):43–1, 2021. 2, 3
- [35] pixabay. pixabay. https://pixabay.com/videos/,

2025. 6

- [36] Jordi Pont-Tuset, Federico Perazzi, Sergi Caelles, Pablo Arbel´aez, Alex Sorkine-Hornung, and Luc Van Gool. The 2017 davis challenge on video object segmentation. arXiv preprint arXiv:1704.00675, 2017. 6
- [37] Chenyang Qi, Xiaodong Cun, Yong Zhang, Chenyang Lei, Xintao Wang, Ying Shan, and Qifeng Chen. Fatezero: Fusing attentions for zero-shot text-based video editing. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 15932–15942, 2023. 3
- [38] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry,

- Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 6
- [39] Mengwei Ren, Wei Xiong, Jae Shin Yoon, Zhixin Shu, Jianming Zhang, HyunJoon Jung, Guido Gerig, and He Zhang. Relightful harmonization: Lighting-aware portrait background replacement. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6452–6462, 2024. 2, 3
- [40] Litu Rout, Yujia Chen, Nataniel Ruiz, Constantine Caramanis, Sanjay Shakkottai, and Wen-Sheng Chu. Semantic image inversion and editing using rectified stochastic differential equations. arXiv preprint arXiv:2410.10792, 2024. 5
- [41] Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. arXiv preprint arXiv:2202.00512, 2022. 3
- [42] Maximilian Seitzer. pytorch-fid: FID Score for PyTorch. https://github.com/mseitzer/pytorch-fid,

2020. Version 0.3.0. 6

- [43] Soumyadip Sengupta, Angjoo Kanazawa, Carlos D Castillo, and David W Jacobs. Sfsnet: Learning shape, reflectance and illuminance of facesin the wild’. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 6296–6305, 2018. 3
- [44] Soumyadip Sengupta, Brian Curless, Ira KemelmacherShlizerman, and Steven M Seitz. A light stage on every desk. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2420–2429, 2021. 2, 3
- [45] YiChang Shih, Sylvain Paris, Connelly Barnes, William T Freeman, and Fr´edo Durand. Style transfer for headshot portraits. 2014. 3
- [46] Zhixin Shu, Sunil Hadap, Eli Shechtman, Kalyan Sunkavalli, Sylvain Paris, and Dimitris Samaras. Portrait lighting transfer using a mass transport approach. ACM Transactions on Graphics (TOG), 36(4):1, 2017. 3
- [47] Tiancheng Sun, Jonathan T Barron, Yun-Ta Tsai, Zexiang Xu, Xueming Yu, Graham Fyffe, Christoph Rhemann, Jay Busch, Paul Debevec, and Ravi Ramamoorthi. Single image portrait relighting. ACM Transactions on Graphics (TOG), 38(4):1–12, 2019. 2, 3
- [48] Luming Tang, Menglin Jia, Qianqian Wang, Cheng Perng Phoo, and Bharath Hariharan. Emergent correspondence from image diffusion. Advances in Neural Information Processing Systems, 36:1363–1389, 2023. 3
- [49] Zachary Teed and Jia Deng. Raft: Recurrent all-pairs field transforms for optical flow. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23– 28, 2020, Proceedings, Part II 16, pages 402–419. Springer,

2020. 6

- [50] Jiuniu Wang, Hangjie Yuan, Dayou Chen, Yingya Zhang, Xiang Wang, and Shiwei Zhang. Modelscope text-to-video technical report. arXiv preprint arXiv:2308.06571, 2023. 3
- [51] Jiangshan Wang, Yue Ma, Jiayi Guo, Yicheng Xiao, Gao Huang, and Xiu Li. Cove: Unleashing the diffusion feature correspondence for consistent video editing. arXiv preprint arXiv:2406.08850, 2024. 3

- [52] Wen Wang, Yan Jiang, Kangyang Xie, Zide Liu, Hao Chen, Yue Cao, Xinlong Wang, and Chunhua Shen. Zero-shot video editing using off-the-shelf image diffusion models. arXiv preprint arXiv:2303.17599, 2023. 3
- [53] Yaohui Wang, Xinyuan Chen, Xin Ma, Shangchen Zhou, Ziqi Huang, Yi Wang, Ceyuan Yang, Yinan He, Jiashuo Yu, Peiqing Yang, et al. Lavie: High-quality video generation with cascaded latent diffusion models. arXiv preprint arXiv:2309.15103, 2023. 3
- [54] Yifan Wang, Aleksander Holynski, Xiuming Zhang, and Xuaner Zhang. Sunstage: Portrait reconstruction and relighting using the sun as a light stage. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20792–20802, 2023. 2, 3
- [55] Jay Zhangjie Wu, Yixiao Ge, Xintao Wang, Stan Weixian Lei, Yuchao Gu, Yufei Shi, Wynne Hsu, Ying Shan, Xiaohu Qie, and Mike Zheng Shou. Tune-a-video: One-shot tuning of image diffusion models for text-to-video generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7623–7633, 2023. 3
- [56] Jinbo Xing, Menghan Xia, Yong Zhang, Haoxin Chen, Wangbo Yu, Hanyuan Liu, Gongye Liu, Xintao Wang, Ying Shan, and Tien-Tsin Wong. Dynamicrafter: Animating open-domain images with video diffusion priors. In European Conference on Computer Vision, pages 399–417. Springer, 2025. 3
- [57] Shuai Yang, Yifan Zhou, Ziwei Liu, and Chen Change Loy. Rerender a video: Zero-shot text-guided video-to-video translation. In SIGGRAPH Asia 2023 Conference Papers, pages 1–11, 2023. 3
- [58] Shuai Yang, Yifan Zhou, Ziwei Liu, and Chen Change Loy. Fresco: Spatial-temporal correspondence for zero-shot video translation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8703– 8712, 2024. 3
- [59] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024. 1, 3
- [60] Chong Zeng, Yue Dong, Pieter Peers, Youkang Kong, Hongzhi Wu, and Xin Tong. Dilightnet: Fine-grained lighting control for diffusion-based image generation. In ACM SIGGRAPH 2024 Conference Papers, pages 1–12, 2024. 2, 3
- [61] David Junhao Zhang, Jay Zhangjie Wu, Jia-Wei Liu, Rui Zhao, Lingmin Ran, Yuchao Gu, Difei Gao, and Mike Zheng Shou. Show-1: Marrying pixel and latent diffusion models for text-to-video generation. International Journal of Computer Vision, pages 1–15, 2024. 3
- [62] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Scaling in-the-wild training for diffusion-based illumination harmonization and editing by imposing consistent light transport. In The Thirteenth International Conference on Learning Representations, 2025. 1, 2, 3, 4, 6, 7
- [63] Yiming Zhang, Zhening Xing, Yanhong Zeng, Youqing Fang, and Kai Chen. Pia: Your personalized image animator

- via plug-and-play modules in text-to-image models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7747–7756, 2024. 3
- [64] Yuxin Zhang, Dandan Zheng, Biao Gong, Jingdong Chen, Ming Yang, Weiming Dong, and Changsheng Xu. Lumisculpt: A consistency lighting control network for video generation. arXiv preprint arXiv:2410.22979, 2024. 2, 3
- [65] Hao Zhou, Sunil Hadap, Kalyan Sunkavalli, and David W Jacobs. Deep single-image portrait relighting. In Proceedings of the IEEE/CVF international conference on computer vision, pages 7194–7202, 2019. 2, 3
- [66] Taotao Zhou, Kai He, Di Wu, Teng Xu, Qixuan Zhang, Kuixiang Shao, Wenzheng Chen, Lan Xu, and Jingyi Yu. Relightable neural human assets from multi-view gradient illuminations. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4315– 4327, 2023. 2, 3

## Light-A-Video: Training-free Video Relighting via Progressive Light Fusion Supplementary Material

[Figure 750]

- Figure 9. Evolution of λt over time steps t for different PLF strategies. λt determines the proportion of the relight target mixed into the fusion target.

### A. Comprehensive Ablation Studies

In this section, we conduct comprehensive ablation studies to explore the effects of the hyper-parameter γ of the Consistent Light Attention (CLA) and various Progressive Light Fusioin (PLF) strategies on the quality of relighted video generation. Specifically, the values of γ are uniformly sampled within the range of [0,1], where a larger γ indicates a higher proportion of the cross-frame averaged feature in the CLA. Notably, when γ = 0, it corresponds to the vanilla IC-Light with standard self-attention. For the PLF strategy, the parameter λt determines the proportion of the relight target mixed into the fusion target at each step. Several different PLF strategies are also proposed, with λt defined as:

k

t Tm

(14)

λt = 1 −

Here, Tm = 25 denotes the total number of noise-adding steps for the source video, and different values of k indicate different rates of decay for λt over time. λt ≡ 1 means directly replacing the fusion target with the relight target for all steps. Fig. 9 illustrates the curves of λt as it varies with time step t.

A quantitative comparison of various settings is provided in Fig. 10, where the three evaluation metrics (FID, Temporal Clip score, and Motion Preservation score) introduced in the main text are employed to evaluate the per-frame image quality and temporal consistency of the relighted video generated by our Light-A-Video method. Specifically, Fig. 10 (a) depicts the variation of the FID score with different values of the trade-off parameter γ. An excessively large γ

results in a significant degradation of the overall relighting image quality. This is attributed to the overemphasis on the cross-frame averaged feature in the CLA module, which leads to temporal over-smoothing and diminishes the lighting specificity, thereby negatively impacting the relighting effect. However, when γ is chosen appropriately (between 0.2 and 0.5), the FID score remains stable and can even be enhanced, especially when employing PLF strategies with k = 1 or k = 0.5.

The temporal consistency evaluation, as depicted in Fig. 10 (b), demonstrates a steady increase in the Temporal Clip score with the rise of the parameter γ. This trend underscores the remarkable efficacy of the CLA module in augmenting the temporal consistency of the relighted video. These results reflect that the CLA module is highly effective in enhancing the temporal consistency of the relighted video. In a parallel vein, the Motion Preservation score serves as an indicator of motion consistency with the source video. Specifically, when the value of γ is selected within the range of 0.2 to 0.5, the relighted video can achieve a high degree of motion consistency with the original video.

It is worth noting that, as evidenced by the three figures, employing a constant λt ≡ 1 significantly underperforms the method of progressively decreasing λt in PLF, both in terms of relight image quality and temporal consistency. Although a constant λt yields a higher Temporal Clip score when γ > 0.5, the overall motion deviates substantially from the source video, resulting in an unacceptable motion preservation effect. These results effectively demonstrate the efficacy of our PLF strategy. The explanation for this observation is twofold:

- • Compared to a dynamically mixed target, a constant target with rich additional illumination information in the denoising process is more likely to deviate from the sampling trajectory of the Video Diffusion Model (VDM). When this deviation exceeds the refinement capability of the VDM, it perturbs the motion priors, consequently leading to visible temporal jitter.
- • Repeatedly injecting constant relight appearance across multiple iterations is analogous to cyclically relighting the same image using the image relight model. This process causes the input distribution to progressively diverge from the training distribution of the image relight model, ultimately degrading the quality of the relighted images.

### B. Additional Results

In this section, we present additional qualitative results. In Fig. [11-12], we show examples of foreground sequences

[Figure 751]

- Figure 10. The relative effectiveness of different PLF strategy on Light-A-Video performance. (a) FID scores, (b) Temporal CLIP scores, and (c) Motion Preservation scores are shown for four strategies: PLF with constant λ (λt ≡ 1), and PLF with k = 0.5, 1, 2. Lower FID/Motion Preservation scores and higher Temporal Clip scores indicate better performance.

relighting with background generation on AnimateDiff. In Fig. [13-14], we showcase the application of Light-A-Video directly to the video relighting task. And finally, as illustrated in Fig. 15, we present the video relighting results on DiT-based video models, such as CogVideoX.

[Figure 752]

[Figure 753]

[Figure 754]

[Figure 755]

[Figure 756]

[Figure 757]

[Figure 758]

[Figure 759]

InputRelighted

a driving car,

[Figure 760]

[Figure 761]

[Figure 762]

[Figure 763]

[Figure 764]

[Figure 765]

[Figure 766]

[Figure 767]

…, on the beach, sunset over sea.

[Figure 768]

[Figure 769]

[Figure 770]

[Figure 771]

[Figure 772]

[Figure 773]

[Figure 774]

[Figure 775]

Relighted

…, on the street, neon light.

[Figure 776]

[Figure 777]

[Figure 778]

[Figure 779]

[Figure 780]

[Figure 781]

[Figure 782]

[Figure 783]

InputRelighted

a handsome man,

[Figure 784]

[Figure 785]

[Figure 786]

[Figure 787]

[Figure 788]

[Figure 789]

[Figure 790]

[Figure 791]

…, in the classroom, sunshine from the window.

[Figure 792]

[Figure 793]

[Figure 794]

[Figure 795]

[Figure 796]

[Figure 797]

[Figure 798]

[Figure 799]

Relighted

…, in a bar, yellow and purple neon lights.

###### Figure 11. More results of Light-A-Video in foreground sequences relighting with background generation.

[Figure 800]

[Figure 801]

[Figure 802]

[Figure 803]

[Figure 804]

[Figure 805]

[Figure 806]

[Figure 807]

InputRelightedInputRelighted

[Figure 808]

[Figure 809]

[Figure 810]

[Figure 811]

[Figure 812]

[Figure 813]

[Figure 814]

[Figure 815]

a tiny camera on a desk, cyberpunk style and light.

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

a glass of water, in the forest, magic golden lit.

[Figure 832]

[Figure 833]

[Figure 834]

[Figure 835]

[Figure 836]

[Figure 837]

[Figure 838]

[Figure 839]

InputRelightedInputRelighted

[Figure 840]

[Figure 841]

[Figure 842]

[Figure 843]

[Figure 844]

[Figure 845]

[Figure 846]

[Figure 847]

a red flower blooming in the river, nature lighting.

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

a wolf stands in an alley, detailed face, neon, Wong Kar-wai, warm.

###### Figure 12. More results of Light-A-Video in foreground sequences relighting with background generation.

InputRelightedInputRelighted

a bear walking on the rock, nature lighting.

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

a fox, sunlight filtering through trees, dappled light.

[Figure 896]

[Figure 897]

[Figure 898]

[Figure 899]

[Figure 900]

[Figure 901]

[Figure 902]

[Figure 903]

InputRelightedInputRelighted

[Figure 904]

[Figure 905]

[Figure 906]

[Figure 907]

[Figure 908]

[Figure 909]

[Figure 910]

[Figure 911]

a woman with curly hair, natural lighting, warm atmosphere.

[Figure 912]

[Figure 913]

[Figure 914]

[Figure 915]

[Figure 916]

[Figure 917]

[Figure 918]

[Figure 919]

[Figure 920]

[Figure 921]

[Figure 922]

[Figure 923]

[Figure 924]

[Figure 925]

[Figure 926]

[Figure 927]

handsome man with glasses, sunlight through the blinds.

InputRelightedInputRelighted

a boat floating on the sea, sunset over sea.

[Figure 944]

[Figure 945]

[Figure 946]

[Figure 947]

[Figure 948]

[Figure 949]

[Figure 950]

[Figure 951]

[Figure 952]

[Figure 953]

[Figure 954]

[Figure 955]

[Figure 956]

[Figure 957]

[Figure 958]

[Figure 959]

a plane on the runway, bottom neon light.

[Figure 960]

[Figure 961]

[Figure 962]

[Figure 963]

[Figure 964]

[Figure 965]

[Figure 966]

[Figure 967]

InputRelightedInputRelighted

[Figure 968]

[Figure 969]

[Figure 970]

[Figure 971]

[Figure 972]

[Figure 973]

[Figure 974]

[Figure 975]

a glass of ice water, candle light.

[Figure 976]

[Figure 977]

[Figure 978]

[Figure 979]

[Figure 980]

[Figure 981]

[Figure 982]

[Figure 983]

[Figure 984]

[Figure 985]

[Figure 986]

[Figure 987]

[Figure 988]

[Figure 989]

[Figure 990]

[Figure 991]

an anime girl, pink neon light.

InputRelightedInputRelighted

buildings in the city, right sunset.

[Figure 1008]

[Figure 1009]

[Figure 1010]

[Figure 1011]

[Figure 1012]

[Figure 1013]

[Figure 1014]

[Figure 1015]

[Figure 1016]

[Figure 1017]

[Figure 1018]

[Figure 1019]

[Figure 1020]

[Figure 1021]

[Figure 1022]

[Figure 1023]

a lioness is walking in the wild, nature lighting.

[Figure 1024]

[Figure 1025]

[Figure 1026]

[Figure 1027]

[Figure 1028]

[Figure 1029]

[Figure 1030]

[Figure 1031]

InputRelightedInputRelighted

[Figure 1032]

[Figure 1033]

[Figure 1034]

[Figure 1035]

[Figure 1036]

[Figure 1037]

[Figure 1038]

[Figure 1039]

a glass of water on the desk, home atmosphere, left light, bubbles in the water, warm sunshine

[Figure 1040]

[Figure 1041]

[Figure 1042]

[Figure 1043]

[Figure 1044]

[Figure 1045]

[Figure 1046]

[Figure 1047]

[Figure 1048]

[Figure 1049]

[Figure 1050]

[Figure 1051]

[Figure 1052]

[Figure 1053]

[Figure 1054]

[Figure 1055]

A girl in anime-game style, in a bar, home atmosphere, warm yellow and purple neon lights

###### Figure 15. More results of Light-A-Video in video sequences relighting on CogVideoX.

